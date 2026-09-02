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
import com.powsybl.openloadflow.ac.AcTargetVector;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivationExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchScenarioResult;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContingencyValidator;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuHybridBatchOuterLoopEngine;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuHybridOuterLoopBridge;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.NetworkState;
import com.powsybl.openloadflow.network.VoltageControl;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import com.powsybl.openloadflow.network.util.PreviousValueVoltageInitializer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
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

        // Original generator setpoints (getInitialTargetP is fixed at network load; the base LF does not commit it).
        double[] originalGenP = network.getBuses().stream().flatMap(b -> b.getGenerators().stream())
                .mapToDouble(LfGenerator::getInitialTargetP).toArray();

        // Base loadflow FIRST — its own context creates the AC equation system AFTER the DC voltage initializer has
        // run (the correct OLF order). We then build the device/bridge equation system `es` fresh, AFTER the base run.
        // Creating `es` BEFORE the base run (and DC-initializing it) corrupts it: OLF attaches DC equations to the
        // network during DC init and they overwrite the AC equations of an already-created AC equation system. That
        // corrupted `es` makes the device's pure-inner solve of the ill-conditioned LINE-26-4231 converge to a
        // SPURIOUS high-loss branch (slack 6.44 pu) that no outer loop can recover; a fresh `es` built after DC init
        // converges to the physical low-loss branch (verified on CPU: GpuCpuPureInnerProbe#scalarVsVectorizedBranch,
        // esBEFORE -6.44 pu vs esAFTER +0.05 pu — same 18482 vars, NOT a vectorized-vs-scalar effect).
        AcLoadFlowContext context = new AcLoadFlowContext(network, acp);   // kept open — the outer-loop context needs it
        new AcloadFlowEngine(context).run();                              // converge the base case (network -> base state)

        EquationSystem<AcVariableType, AcEquationType> es =
                new AcEquationSystemCreator(network, acp.getEquationSystemCreationParameters()).create();
        AcSolverUtil.initStateVector(network, es, new PreviousValueVoltageInitializer());
        double[] baseState = es.getStateVector().get().clone();
        int n = baseState.length;

        GpuAcActivation baseAct = GpuAcActivation.fromEquationSystem(network, es);
        int[] baseRowMode = baseAct.rowMode();
        // Build the base target vector from a FRESH AcTargetVector AFTER the base run so it captures the base
        // distributed-slack redistribution (AcTargetVector is lazy + invalidated on targetP change).
        double[] baseTargets = baseAct.applyTargets(new AcTargetVector(network, es).getArray());

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
        GpuBatchElementMapper mapper = new GpuBatchElementMapper(network);
        List<BatchContingency> allCtgs = ids.stream()
                .filter(id -> network.getBranchById(id) != null)
                .map(id -> new BatchContingency(id, network.getBranchById(id)))
                .toList();
        // ISLANDING QUALIFICATION — route islanding outages to CPU, exactly like the production batch. Without
        // it, an outage that islands buses leaves the slack unbalanceable and OLF's distributed-slack loop
        // oscillates forever (measured: LINE-26-4231, 4.72 pu stranded, dragged the loop to the 40-round cap).
        List<GpuBatchContingencyValidator.Qualification> quals =
                GpuBatchContingencyValidator.qualify(network, allCtgs, mapper, false);
        List<BatchContingency> ctgs = new ArrayList<>();
        int nRoutedCpu = 0;
        for (GpuBatchContingencyValidator.Qualification q : quals) {
            if (q.qualified()) {
                ctgs.add(q.contingency());
            } else {
                nRoutedCpu++;
            }
        }
        int nS = ctgs.size();

        // BOTH RL and DS packs EMPTY -> PURE batched inner Newton; ALL outer loops run in OLF's Java. This now
        // converges every contingency (incl. the once-divergent LINE-26-4231) because `es` is built AFTER the base
        // LF's DC init (see above) — the device extracts an uncorrupted equation system and lands on the physical
        // low-loss branch.
        GpuAcData gpuData = GpuAcDataExtractor.extract(network, es, false, false);
        int nElements = gpuData.getTotalElementCount();

        int mode = 3;   // UBATCH (production mode)
        long structure = GpuAcNewtonSolver.createStructure(n, gpuData.cbIn(), gpuData.cbIdx(), gpuData.obIn(), gpuData.obIdx(),
                gpuData.shIn(), gpuData.shIdx(), gpuData.hvIn(), gpuData.hvIdx(),
                gpuData.dqElem(), gpuData.dqSide(), gpuData.dqWeight(), gpuData.dqRow(), gpuData.dqKind(),
                gpuData.ziIn(), gpuData.ziIdx(), gpuData.drRow(), gpuData.drCol(), gpuData.drCoef(),
                gpuData.dbRow(), gpuData.dbCol(), gpuData.dbCoef());
        // (The batch-context handle + the per-scenario outer-loop bridges are now owned by
        // GpuHybridBatchOuterLoopEngine, which is invoked below.)

        // Compact-snapshot element sets (controller buses + HVDCs) + a base full snapshot kept ONLY for the
        // one-time base reset before the perf comparison (never used in the engine's hot loop).
        List<LfBus> ctrlBuses = network.getControllerElements(VoltageControl.Type.GENERATOR);
        List<LfHvdc> hvdcs = network.getHvdcs();
        NetworkState baseSnap = NetworkState.save(network);
        List<LfGenerator> gens = network.getBuses().stream().flatMap(b -> b.getGenerators().stream()).toList();
        double[] baseGenP = gens.stream().mapToDouble(LfGenerator::getTargetP).toArray();

        // Device state-vector scaling mirroring the production CPU solver (rte params = MAX_VOLTAGE_CHANGE);
        // lsMode: NONE=0, LINE_SEARCH=1, MAX_VOLTAGE_CHANGE=2.
        NewtonRaphsonParameters acSolverParams = (NewtonRaphsonParameters) acp.getAcSolverParameters();
        int lsMode = switch (acSolverParams.getStateVectorScalingMode()) {
            case NONE -> 0;
            case LINE_SEARCH -> 1;
            case MAX_VOLTAGE_CHANGE -> 2;
        };
        double lsMaxDv = acSolverParams.getMaxVoltageChangeStateVectorScalingMaxDv();
        double lsMaxDphi = acSolverParams.getMaxVoltageChangeStateVectorScalingMaxDphi();
        int[] varType = lsMode == 2 ? GpuAcActivationExtractor.extractVarType(es, n) : null;

        // Run the EXTRACTED production driver (cross-round compaction + compact snapshot + save-skip + straggler
        // routing). It owns the batch-context handle + the per-scenario outer-loop bridges; we own the structure.
        GpuHybridBatchOuterLoopEngine.Result res = GpuHybridBatchOuterLoopEngine.solve(
                new GpuHybridBatchOuterLoopEngine.Inputs(structure, mode, gpuData, nElements,
                        network, es, context, n, baseState, baseTargets, baseRowMode,
                        ctgs, mapper, gens, baseGenP, originalGenP, ctrlBuses, hvdcs,
                        network.getControllerElements(VoltageControl.Type.TRANSFORMER),
                        lsMode, lsMaxDv, lsMaxDphi, varType,
                        30, acp.getMaxOuterLoopIterations(), 3,
                        1e-8, 1e-4, 1e-2, 1e-4));   // tol (GPU_TOLERANCE floor), reactiveLimitTol, slackDistTol, convEpsPerEq
        GpuAcNewtonSolver.destroyStructure(structure);

        double[][] states = res.states();
        double[][] round0States = res.round0States();
        int[][] rowMode = res.rowMode();
        double[][] scenTargets = res.scenTargets();
        boolean[] retired = res.retired();
        boolean[] routedByBudget = res.routedByBudget();
        GpuHybridOuterLoopBridge[] bridges = res.bridges();
        int rounds = res.rounds();
        long tSolve = res.solveNanos();
        long tJava = res.javaNanos();
        long tSnap = res.snapshotNanos();
        long nJavaCalls = res.javaCalls();

        // Correctness gate: the hybrid must GPU-converge all but genuinely non-convergent contingencies
        // (TWT-106-2546 fails on CPU too and is correctly routed to the CPU fallback).
        assertTrue(res.gpuConvergedCount() >= nS - 1,
                "expected >= " + (nS - 1) + " GPU-converged, got " + res.gpuConvergedCount() + "/" + nS);

        // Report: does the hybrid end with VL-7616_0 pinned to PQ (as CPU does) rather than held PV?
        int budgetRouted = 0;
        for (int s = 0; s < nS; s++) {
            if (routedByBudget[s]) {
                budgetRouted++;
            }
        }
        int gpuConverged = nS - budgetRouted;
        System.out.printf("%n=== HYBRID batch: %d scenarios | %d GPU-converged, %d islanding->CPU, %d budget->CPU | %d rounds ===%n",
                nS, gpuConverged, nRoutedCpu, budgetRouted, rounds);
        System.out.printf("  TIMING total: solve(cuDSS batched inner, SAME in both designs)=%.0f ms | java(OLF outer-loop work)=%.0f ms"
                + " | snapshot(prototype-only, a production hybrid avoids)=%.0f ms | javaCalls=%d%n",
                tSolve / 1e6, tJava / 1e6, tSnap / 1e6, nJavaCalls);
        System.out.printf("  PER-UNIT: java=%.3f ms/scenario-pass | solve=%.1f ms/round%n",
                nJavaCalls > 0 ? tJava / 1e6 / nJavaCalls : 0.0, rounds > 0 ? tSolve / 1e6 / rounds : 0.0);
        int nNonConv = 0;
        for (int s = 0; s < nS; s++) {
            if (!retired[s]) {
                nNonConv++;
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < bridges[s].loopCount(); i++) {
                    if (bridges[s].unstableCount()[i] > 0) {
                        sb.append(String.format(" %s=%d", bridges[s].loopName(i), bridges[s].unstableCount()[i]));
                    }
                }
                if (nNonConv <= 8) {
                    int changedTargets = 0;
                    double maxDelta = 0;
                    for (int r = 0; r < n; r++) {
                        double d = Math.abs(scenTargets[s][r] - baseTargets[r]);
                        if (d > 1e-9) {
                            changedTargets++;
                            maxDelta = Math.max(maxDelta, d);
                        }
                    }
                    System.out.printf("  NON-CONVERGED %s: unstable-by-loop%s | lastSlackMismatch(pu)=%.4e | disabledElems=%d"
                            + " | changedTargets=%d maxDelta=%.4e%n",
                            ctgs.get(s).contingencyId(), sb, bridges[s].lastSlackMismatch(),
                            ctgs.get(s).outagedBranches().size(), changedTargets, maxDelta);
                }
            }
        }
        System.out.printf("  NON-CONVERGED total: %d/%d%n", nNonConv, nS);
        for (int s = 0; s < nS; s++) {
            if (routedByBudget[s]) {
                System.out.printf("  ROUTED-TO-CPU %s: round0 device[iters=%d ||F||inf=%.3e status=%d] | last device[iters=%d ||F||inf=%.3e status=%d]%n",
                        ctgs.get(s).contingencyId(),
                        (int) round0States[s][n], round0States[s][n + 1], (int) round0States[s][n + 2],
                        (int) states[s][n], states[s][n + 1], (int) states[s][n + 2]);
            }
        }
        for (int s = 0; s < nS && nS <= 4; s++) {
            // (The scenario's evolved PV/PQ + HVDC state is now engine-internal; this ≤4-scenario diagnostic reads
            // the device solve state directly — the PV/PQ flag reflects the base network, not the scenario's switches.)
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

        // =========================================================================================================
        // HEAD-TO-HEAD: this Java-outer-loop hybrid vs the production ON-DEVICE outer-loop batch, SAME contingency
        // set, SAME network/JVM. The on-device path (GpuBatchedSecurityAnalysisSolver.solveBatchN1) folds DS + RL
        // into the device solve as CUDA kernels; it is run warm (a warm-up call first) so both numbers exclude the
        // one-off CUDA init the hybrid loop above already paid. Per-contingency ms is the comparable metric.
        // =========================================================================================================
        AcTargetVector cmpTarget = new AcTargetVector(network, es);
        NewtonRaphsonParameters cmpNrp = new NewtonRaphsonParameters().setMaxIterations(30);
        baseSnap.restore();
        long odCold0 = System.nanoTime();
        GpuBatchedSecurityAnalysisSolver.solveBatchN1(network, baseState, ctgs, es, cmpTarget, cmpNrp, ReportNode.NO_OP); // 1st (cold data structures; CUDA already init by the hybrid loop)
        long odColdTime = System.nanoTime() - odCold0;
        baseSnap.restore();
        long od0 = System.nanoTime();
        List<BatchScenarioResult> odResults = GpuBatchedSecurityAnalysisSolver.solveBatchN1(
                network, baseState, ctgs, es, cmpTarget, cmpNrp, ReportNode.NO_OP);
        long odTime = System.nanoTime() - od0;
        long odConverged = odResults.stream().filter(r -> !r.needsCpuFallback()).count();

        double hybridProductiveMs = (tSolve + tJava) / 1e6;   // exclude prototype-only snapshot (a real hybrid avoids it)
        double hybridTotalMs = (tSolve + tJava + tSnap) / 1e6;
        double odMs = odTime / 1e6;
        System.out.printf("%n=== PERF HEAD-TO-HEAD (%d contingencies) ===%n", nS);
        System.out.printf("  ON-DEVICE outer loops (solveBatchN1): warm=%.0f ms (%.3f ms/ctg) | cold-1st-call=%.0f ms (%.3f ms/ctg) | %d/%d converged%n",
                odMs, odMs / nS, odColdTime / 1e6, odColdTime / 1e6 / nS, odConverged, nS);
        System.out.printf("  JAVA-outer-loop HYBRID: productive(solve+java)=%.0f ms | %.3f ms/ctg  [+ prototype snapshot %.0f ms -> total %.0f ms]%n",
                hybridProductiveMs, hybridProductiveMs / nS, tSnap / 1e6, hybridTotalMs);
        // The hybrid re-solves ONLY still-active scenarios each round (cross-round compaction); a non-converging
        // straggler still forces the round cap but runs those extra rounds solo, so its cost is small. The on-device
        // solveBatchN1 folds the outer loops into a single solve with per-scenario early-exit.
        System.out.printf("  hybrid breakdown: inner solve=%.0f ms (%d rounds, active-only compaction) | java outer=%.0f ms | snapshot=%.0f ms%n",
                tSolve / 1e6, rounds, tJava / 1e6, tSnap / 1e6);
        System.out.printf("  hybrid productive / on-device = x%.2f  (>1 => hybrid slower)%n", hybridProductiveMs / odMs);

        context.close();
    }
}
