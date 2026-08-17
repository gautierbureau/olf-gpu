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

        // contingencies -> per-scenario disabled-element masks. Default: the 2 divergent ones; with
        // OLF_GPU_HYBRID_N=k, take the first k single-branch contingencies from the file (throughput sizing).
        List<String> ids = CTG_IDS;
        String nEnv = System.getenv("OLF_GPU_HYBRID_N");
        if (nEnv != null) {
            ids = Files.readAllLines(Path.of(System.getenv("OLF_GPU_CONT_FILE"))).stream()
                    .map(String::trim).filter(x -> !x.isEmpty())
                    .filter(id -> network.getBranchById(id) != null)
                    .limit(Integer.parseInt(nEnv)).toList();
        }
        List<BatchContingency> ctgs = ids.stream()
                .filter(id -> network.getBranchById(id) != null)
                .map(id -> new BatchContingency(id, network.getBranchById(id)))
                .toList();
        int nS = ctgs.size();

        // device structure with BOTH RL and DS packs EMPTY -> PURE batched inner Newton; ALL outer loops
        // (distributed slack, reactive limits, transformer VC) run in OLF's Java, so the state feeding every
        // decision is OLF's own, not a device-outer-loop approximation. (DISTR_Q/DISTR_RHO linear couplings stay.)
        GpuAcData gpuData = GpuAcDataExtractor.extract(network, es, false, false);
        int nElements = gpuData.getTotalElementCount();
        GpuBatchElementMapper mapper = new GpuBatchElementMapper(network);

        int mode = 3;   // UBATCH (production mode)
        long structure = GpuAcNewtonSolver.createStructure(n, gpuData.cbIn(), gpuData.cbIdx(), gpuData.obIn(), gpuData.obIdx(),
                gpuData.shIn(), gpuData.shIdx(), gpuData.hvIn(), gpuData.hvIdx(),
                gpuData.dqElem(), gpuData.dqSide(), gpuData.dqWeight(), gpuData.dqRow(), gpuData.dqKind(),
                gpuData.ziIn(), gpuData.ziIdx(), gpuData.drRow(), gpuData.drCol(), gpuData.drCoef());
        long handle = GpuAcNewtonSolver.createBatchContext(structure, nS, mode,
                gpuData.rlVrow(), gpuData.rlMinQ(), gpuData.rlMaxQ(), gpuData.rlLoadQ(), gpuData.rlVtarget(),
                gpuData.dsPhiRow(), gpuData.dsFactor(), gpuData.dsSlackPhiRow(), gpuData.dsSlackTargetP(),
                gpuData.dsBaseTargetP(), gpuData.dsMaxDelta(), gpuData.dsMinDelta());

        // OLF's FULL ordered outer-loop set (distributed slack, reactive limits, transformer VC, …) — one
        // context PER (scenario, loop), as OLF's SA gives each contingency its own, so scenarios never
        // contaminate each other's switch/redistribution bookkeeping.
        List<AcOuterLoop> allLoops = acp.getOuterLoops();
        java.lang.reflect.Constructor<AcOuterLoopContext> olCtor = AcOuterLoopContext.class.getDeclaredConstructor(LfNetwork.class);
        olCtor.setAccessible(true);
        GpuHybridOuterLoopBridge[] bridges = new GpuHybridOuterLoopBridge[nS];
        for (int s = 0; s < nS; s++) {
            List<AcOuterLoopContext> ctxs = new ArrayList<>();
            for (AcOuterLoop ol : allLoops) {
                AcOuterLoopContext olCtx = olCtor.newInstance(network);
                olCtx.setLoadFlowContext(context);
                ol.initialize(olCtx);
                olCtx.setLastSolverResult(new AcSolverResult(AcSolverStatus.CONVERGED, 1, java.util.Map.of()));
                ctxs.add(olCtx);
            }
            bridges[s] = new GpuHybridOuterLoopBridge(network, es, n, allLoops, ctxs);
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
        long tSolve = 0;                                                  // batched device inner solve (cuDSS) — SAME in both designs
        long tJava = 0;                                                   // essential Java outer-loop work (state load + OLF checks + re-extract)
        long tSnap = 0;                                                   // prototype-only NetworkState save/restore (a production hybrid avoids this)
        long nJavaCalls = 0;
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
            long ts0 = System.nanoTime();
            states = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets, rowMode, baseRowMode, disabled,
                    ovIdx.stream().mapToInt(Integer::intValue).toArray(), ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    new int[0], new double[0], new double[0], new double[0], new double[0],
                    nrp.getMaxIterations(), 1e-6, 1e-4, 1e-2, 1e-4,       // tol, reactiveLimitTol(unused), slackDistTol, convEpsPerEq
                    0, 0.0, 0.0, null, x0,                                // lsMode, maxDv, maxDphi, varType, x0PerScenario
                    null, null, null, null, null, null,                  // shuntOv, rlStateIn/Out, r1Ov
                    30, 3, 0, 0, null, null);                            // maxOuterIterations(device DS), maxPqPvSwitch, dsRejectResidual, nzi...
            tSolve += System.nanoTime() - ts0;

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
                long tr0 = System.nanoTime();
                if (snaps[s] != null) {
                    snaps[s].restore();                                 // resume THIS scenario's accumulated PV/PQ state
                } else {
                    baseSnap.restore();                                 // first round: from the base-converged state
                }
                tSnap += System.nanoTime() - tr0;
                for (LfBranch b : ctgs.get(s).outagedBranches()) {
                    b.setDisabled(true);                                // apply this scenario's outage
                }
                double[] st = new double[n];
                System.arraycopy(states[s], 0, st, 0, n);
                long tj0 = System.nanoTime();
                GpuAcActivation act = bridges[s].applyOuterLoopStep(st);  // set state -> all OLF outer loops -> re-extract
                rowMode[s] = act.rowMode();
                // fresh target vector: captures the P redistributed by the Java distributed-slack loop (and any tap change)
                scenTargets[s] = act.applyTargets(new AcTargetVector(network, es).getArray());
                tJava += System.nanoTime() - tj0;
                nJavaCalls++;
                long tsv0 = System.nanoTime();
                snaps[s] = NetworkState.save(network);                   // persist this scenario's evolved PV/PQ + targets
                tSnap += System.nanoTime() - tsv0;
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
        System.out.printf("%n=== HYBRID batch: %d scenarios, %d rounds, converged=%s ===%n", nS, rounds, converged);
        System.out.printf("  TIMING total: solve(cuDSS batched inner, SAME in both designs)=%.0f ms | java(OLF outer-loop work)=%.0f ms"
                + " | snapshot(prototype-only, a production hybrid avoids)=%.0f ms | javaCalls=%d%n",
                tSolve / 1e6, tJava / 1e6, tSnap / 1e6, nJavaCalls);
        System.out.printf("  PER-UNIT: java=%.3f ms/scenario-pass | solve=%.1f ms/round%n",
                nJavaCalls > 0 ? tJava / 1e6 / nJavaCalls : 0.0, rounds > 0 ? tSolve / 1e6 / rounds : 0.0);
        for (int s = 0; s < nS && nS <= 4; s++) {
            snaps[s].restore();                                          // the hybrid's OWN evolved PV/PQ state for this scenario
            es.getStateVector().set(states[s]);
            AcSolverUtil.updateNetwork(network, es);
            LfBus b7616 = network.getBuses().stream().filter(b -> "VL-7616_0".equals(b.getId())).findFirst().orElse(null);
            String pv = b7616 == null ? "?" : (b7616.isGeneratorVoltageControlEnabled() ? "PV(diverges from CPU)" : "PQ(matches CPU)");
            double v7616 = b7616 == null ? Double.NaN : b7616.getV();
            int vrow = -1;
            for (com.powsybl.openloadflow.equations.Variable<AcVariableType> var : es.getIndex().getSortedVariablesToFind()) {
                if (var.getType() == AcVariableType.BUS_V && b7616 != null && var.getElementNum() == b7616.getNum()) {
                    vrow = var.getRow();
                    break;
                }
            }
            double tgtV = b7616 == null ? Double.NaN : b7616.getGeneratorVoltageControl().map(c -> c.getTargetValue()).orElse(Double.NaN);
            System.out.printf("  %-16s conv=%d  VL-7616_0=%s  v=%.6f%n", ctgs.get(s).contingencyId(), (int) states[s][n], pv, v7616);
            System.out.printf("      DIAG row=%d rowMode=%d(1=power/PV,0=identity) target=%.6f deviceStateV=%.6f targetV=%.6f%n",
                    vrow, vrow >= 0 ? rowMode[s][vrow] : -9, vrow >= 0 ? scenTargets[s][vrow] : Double.NaN,
                    vrow >= 0 ? states[s][vrow] : Double.NaN, tgtV);
            // invariant: every mode-0 (identity) row must satisfy x == target on the device solve.
            int nMode0 = 0;
            int notPinned = 0;
            String ex = "";
            for (int r = 0; r < n; r++) {
                if (rowMode[s][r] == 0) {
                    nMode0++;
                    if (Math.abs(states[s][r] - scenTargets[s][r]) > 1e-5) {
                        notPinned++;
                        if (notPinned <= 3) {
                            ex += String.format(" [r=%d x=%.6f tgt=%.6f]", r, states[s][r], scenTargets[s][r]);
                        }
                    }
                }
            }
            System.out.printf("      PIN-CHECK: %d/%d mode-0 rows NOT pinned to target%s%n", notPinned, nMode0, ex);
        }
        baseSnap.restore();
        context.close();
    }
}
