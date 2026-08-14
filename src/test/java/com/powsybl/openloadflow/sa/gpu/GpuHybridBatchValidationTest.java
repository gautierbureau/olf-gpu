/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcOuterLoopContext;
import com.powsybl.openloadflow.ac.AcTargetVector;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop;
import com.powsybl.openloadflow.ac.solver.AcSolverResult;
import com.powsybl.openloadflow.ac.solver.AcSolverStatus;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuHybridOuterLoopBridge;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.lf.outerloop.OuterLoopStatus;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.NetworkState;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import com.powsybl.openloadflow.network.util.PreviousValueVoltageInitializer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * PROTOTYPE validation: drive the batched inner Newton on the device while making the reactive-limits
 * PV↔PQ decisions in OLF's validated Java {@link ReactiveLimitsOuterLoop} (via {@link
 * GpuHybridOuterLoopBridge}), and check that the two contingencies whose device-kernel batch diverged
 * from the CPU (LINE-4985-165, TWT-14-2925 — the VL-7616_0 limit-cycle) now agree.
 *
 * <p>The device context is built with an EMPTY reactive-limits pack (reactiveLimits=false) but the
 * distributed-slack pack ON, so {@code solveBatchContext} runs inner Newton + device DS to convergence
 * and never switches PV↔PQ. Between rounds, per scenario: reset the network, apply the contingency, load
 * the batched state, run OLF's RL check (mutates the activation), re-extract the device row mode, and
 * warm-restart the next round. Loop until every scenario's RL loop reports STABLE.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuHybridBatchValidationTest {

    private static final Path CASES = GpuTestPaths.caseDir();
    private static final String CASE = "case9241pegase.xiidm.gz";
    private static final List<String> CTG_IDS = List.of("LINE-4985-165", "TWT-14-2925");

    static {
        GpuTestPaths.init();
    }

    @BeforeEach
    void check() {
        assumeTrue(com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(),
                "GPU native lib not available");
        assumeTrue(Files.exists(CASES.resolve(CASE)), CASE + " not found");
        assumeTrue(Files.exists(Path.of("rte-loadflow-default-parameters.json")), "rte params not found");
    }

    @Test
    void hybridReactiveLimitsMatchesCpu() throws Exception {
        Network iidm = Network.read(CASES.resolve(CASE));
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(Path.of("rte-loadflow-default-parameters.json"));
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.get(lfp);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        acp.getNetworkParameters().setMaxSlackBusCount(1);

        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);
        EquationSystem<AcVariableType, AcEquationType> es =
                new AcEquationSystemCreator(network, acp.getEquationSystemCreationParameters()).create();
        AcSolverUtil.initStateVector(network, es, acp.getVoltageInitializer());
        AcTargetVector target = new AcTargetVector(network, es);

        AcLoadFlowContext context = new AcLoadFlowContext(network, acp);   // kept open — the outer-loop context needs it
        new AcloadFlowEngine(context).run();                              // converge the base case (network -> base state)
        AcSolverUtil.initStateVector(network, es, new PreviousValueVoltageInitializer());
        double[] baseState = es.getStateVector().get().clone();
        int n = baseState.length;

        GpuAcActivation baseAct = GpuAcActivation.fromEquationSystem(network, es);
        int[] baseRowMode = baseAct.rowMode();
        double[] baseTargets = baseAct.applyTargets(target.getArray());

        // contingencies -> per-scenario disabled-element masks
        List<BatchContingency> ctgs = CTG_IDS.stream()
                .map(id -> new BatchContingency(id, network.getBranchById(id)))
                .toList();
        int nS = ctgs.size();

        // device structure with the RL pack EMPTY (reactiveLimits=false) but DS ON -> inner + device DS, no PV<->PQ
        GpuAcData gpuData = GpuAcDataExtractor.extract(network, es, true, false);
        int nElements = gpuData.getTotalElementCount();
        GpuBatchElementMapper mapper = new GpuBatchElementMapper(network);

        int mode = 1;                                                      // batched API (simple, robust for a small nS)
        long structure = GpuAcNewtonSolver.createStructure(n, gpuData.cbIn(), gpuData.cbIdx(), gpuData.obIn(), gpuData.obIdx(),
                gpuData.shIn(), gpuData.shIdx(), gpuData.hvIn(), gpuData.hvIdx(),
                gpuData.dqElem(), gpuData.dqSide(), gpuData.dqWeight(), gpuData.dqRow(), gpuData.dqKind(),
                gpuData.ziIn(), gpuData.ziIdx(), gpuData.drRow(), gpuData.drCol(), gpuData.drCoef());
        long handle = GpuAcNewtonSolver.createBatchContext(structure, nS, mode,
                gpuData.rlVrow(), gpuData.rlMinQ(), gpuData.rlMaxQ(), gpuData.rlLoadQ(), gpuData.rlVtarget(),
                gpuData.dsPhiRow(), gpuData.dsFactor(), gpuData.dsSlackPhiRow(), gpuData.dsSlackTargetP(),
                gpuData.dsBaseTargetP(), gpuData.dsMaxDelta(), gpuData.dsMinDelta());

        // OLF's validated reactive-limits outer loop + a context bound to this network/loadflow
        AcOuterLoop rlLoop = acp.getOuterLoops().stream()
                .filter(o -> o instanceof ReactiveLimitsOuterLoop).findFirst().orElseThrow();
        // ONE outer-loop context (and thus PV<->PQ switch counter) PER SCENARIO — as OLF's SA gives each
        // contingency its own — so scenarios do not contaminate each other's switch bookkeeping.
        java.lang.reflect.Constructor<AcOuterLoopContext> olCtor = AcOuterLoopContext.class.getDeclaredConstructor(LfNetwork.class);
        olCtor.setAccessible(true);
        GpuHybridOuterLoopBridge[] bridges = new GpuHybridOuterLoopBridge[nS];
        for (int s = 0; s < nS; s++) {
            AcOuterLoopContext olCtx = olCtor.newInstance(network);
            olCtx.setLoadFlowContext(context);
            rlLoop.initialize(olCtx);
            olCtx.setLastSolverResult(new AcSolverResult(AcSolverStatus.CONVERGED, 1, java.util.Map.of()));
            bridges[s] = new GpuHybridOuterLoopBridge(network, es, n, rlLoop, olCtx);
        }

        NetworkState baseSnap = NetworkState.save(network);
        NetworkState[] snaps = new NetworkState[nS];       // each scenario's evolving PV/PQ state, persisted across rounds

        // per-scenario device inputs
        int[][] rowMode = new int[nS][];
        boolean[][] disabled = new boolean[nS][nElements];
        double[][] scenTargets = new double[nS][];
        boolean[] retired = new boolean[nS];
        for (int s = 0; s < nS; s++) {
            rowMode[s] = baseRowMode.clone();
            scenTargets[s] = baseTargets.clone();
            for (LfBranch b : ctgs.get(s).outagedBranches()) {
                disabled[s][mapper.getElementIndex(b)] = true;
            }
        }

        NewtonRaphsonParameters nrp = new NewtonRaphsonParameters().setMaxIterations(30);
        double[] x0 = null;                                               // first round: base state tiled on-device
        double[][] states = null;

        int rounds = 0;
        for (int round = 0; round < 40; round++) {
            rounds = round + 1;
            // sparse target overrides (rows where this scenario's target differs from the base)
            List<Integer> ovIdx = new ArrayList<>();
            List<Double> ovVal = new ArrayList<>();
            for (int s = 0; s < nS; s++) {
                for (int r = 0; r < n; r++) {
                    if (scenTargets[s][r] != baseTargets[r]) {
                        ovIdx.add(s * n + r);
                        ovVal.add(scenTargets[s][r]);
                    }
                }
            }
            states = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets, rowMode, baseRowMode, disabled,
                    ovIdx.stream().mapToInt(Integer::intValue).toArray(), ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    new int[0], new double[0], new double[0], new double[0], new double[0],
                    nrp.getMaxIterations(), 1e-6, 1e-4, 1e-2, 1e-4,       // tol, reactiveLimitTol(unused), slackDistTol, convEpsPerEq
                    0, 0.0, 0.0, null, x0,                                // lsMode, maxDv, maxDphi, varType, x0PerScenario
                    null, null, null, null, null, null,                  // shuntOv, rlStateIn/Out, r1Ov
                    30, 3, 0, 0, null, null);                            // maxOuterIterations(device DS), maxPqPvSwitch, dsRejectResidual, nzi...

            // warm restart for the next round
            x0 = new double[nS * n];
            for (int s = 0; s < nS; s++) {
                System.arraycopy(states[s], 0, x0, s * n, n);
            }

            boolean allStable = true;
            for (int s = 0; s < nS; s++) {
                if (retired[s]) {
                    continue;
                }
                if (snaps[s] != null) {
                    snaps[s].restore();                                 // resume THIS scenario's accumulated PV/PQ state
                } else {
                    baseSnap.restore();                                 // first round: from the base-converged state
                }
                for (LfBranch b : ctgs.get(s).outagedBranches()) {
                    b.setDisabled(true);                                // apply this scenario's outage
                }
                double[] st = new double[n];
                System.arraycopy(states[s], 0, st, 0, n);
                GpuAcActivation act = bridges[s].applyOuterLoopStep(st);  // set state -> OLF RL check (mutates network) -> re-extract
                snaps[s] = NetworkState.save(network);                   // persist this scenario's evolved PV/PQ for next round
                rowMode[s] = act.rowMode();
                scenTargets[s] = act.applyTargets(target.getArray());
                if (bridges[s].isStable()) {
                    retired[s] = true;
                } else {
                    allStable = false;
                }
            }
            if (allStable) {
                break;
            }
        }
        GpuAcNewtonSolver.destroyBatchContext(handle);
        GpuAcNewtonSolver.destroyStructure(structure);

        // Report: does the hybrid end with VL-7616_0 pinned to PQ (as CPU does) rather than held PV?
        boolean converged = rounds < 40;
        System.out.printf("%n=== HYBRID reactive-limits batch: %d scenarios, %d rounds, converged=%s ===%n", nS, rounds, converged);
        for (int s = 0; s < nS; s++) {
            snaps[s].restore();                                          // the hybrid's OWN evolved PV/PQ state for this scenario
            es.getStateVector().set(states[s]);
            AcSolverUtil.updateNetwork(network, es);
            LfBus b7616 = network.getBuses().stream().filter(b -> "VL-7616_0".equals(b.getId())).findFirst().orElse(null);
            String pv = b7616 == null ? "?" : (b7616.isGeneratorVoltageControlEnabled() ? "PV(diverges from CPU)" : "PQ(matches CPU)");
            double v7616 = b7616 == null ? Double.NaN : b7616.getV();
            System.out.printf("  %-16s conv=%d  VL-7616_0=%s  v=%.6f%n", CTG_IDS.get(s), (int) states[s][n], pv, v7616);
        }
        baseSnap.restore();
        context.close();
    }
}
