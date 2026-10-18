/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivationExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataCache;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuEquationDump;
import com.powsybl.openloadflow.ac.outerloop.gpu.ContinuousPhaseControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.DeviceLoopKind;
import com.powsybl.openloadflow.ac.outerloop.gpu.DeviceLoopTables;
import com.powsybl.openloadflow.ac.outerloop.gpu.DistributedSlackWriteBack;
import com.powsybl.openloadflow.ac.outerloop.gpu.ForcedTargetQ;
import com.powsybl.openloadflow.ac.outerloop.gpu.HvdcAcEmulation;
import com.powsybl.openloadflow.ac.outerloop.gpu.HvdcNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.IncrementalFlowControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.IncrementalShuntVoltageControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.IncrementalTransformerVoltageControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveCapabilityCurves;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveLimitsWriteBack;
import com.powsybl.openloadflow.ac.outerloop.gpu.VoltageMonitoring;
import com.powsybl.openloadflow.ac.solver.AbstractAcSolver;
import com.powsybl.openloadflow.ac.solver.AcSolverResult;
import com.powsybl.openloadflow.ac.solver.AcSolverStatus;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphson;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.PerEquationTypeStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchStatus;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationSystemIndex;
import com.powsybl.openloadflow.equations.EquationSystemIndexListener;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkListener;
import com.powsybl.openloadflow.network.LfSynchronousNetwork;
import com.powsybl.openloadflow.network.util.VoltageInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Newton-Raphson run ENTIRELY on the GPU behind one JNI call ({@link GpuAcNewtonSolver}):
 * the fixed superset-pattern CSR Jacobian and the mismatch are filled on the device by
 * the generated fused kernels (closed-branch / open-branch / shunt / HVDC AC emulation),
 * cuDSS analyzes once and refactorizes/solves per iteration, the state update is a
 * kernel.
 *
 * <p>Invalidation is EVENT-DRIVEN, mirroring {@link JacobianMatrix}'s pattern (the same
 * {@link EquationSystemIndexListener} / {@link LfNetworkListener} interfaces, with this
 * consumer's own granularity): structural events (topology, branch parameters) force a
 * full re-extract + device context rebuild; activation events (PV→PQ switching, slack
 * change) only refresh the row-mode mask and target mapping; with no events, a rerun
 * goes straight to the device solve — targets and state are re-read every solve. Both
 * interfaces are implemented DIRECTLY (not via the no-op adapter) so any event added to
 * OLF later forces a compile-time classification here, keeping invalidation conservative.
 *
 * <p>Scope: plain AC power flow. {@link GpuAcDataExtractor} validates the equation
 * system on every structural rebuild and throws on anything not covered; an unsupported
 * equation type ACTIVATED between rebuilds fails loudly in the activation walk. The
 * device loop iterates to an infinity-norm mismatch of {@value GPU_TOLERANCE} (stricter
 * than the usual stopping criteria); the final CONVERGED/MAX_ITERATION_REACHED status is
 * then decided by the CONFIGURED stopping criteria, evaluated on the CPU from the
 * written-back state, exactly as for the CPU Newton-Raphson. State-vector scaling
 * options are not applied (no line search on the device).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuNewtonRaphson extends AbstractAcSolver {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuNewtonRaphson.class);

    private static final Cleaner CLEANER = Cleaner.create();

    public static final double GPU_TOLERANCE = 1e-8;

    /** The engine's AcLoadFlowContext, set by GpuDriverOuterLoopEngine before calling solver.run()
     *  so the solver's host-loop pause cycle can create outer-loop contexts that share the engine's
     *  equation system and Jacobian. Thread-local because the engine and solver run on the same thread. */
    public static final ThreadLocal<com.powsybl.openloadflow.ac.AcLoadFlowContext> GPU_ENGINE_CONTEXT = new ThreadLocal<>();

    /** Did this solver's LAST solve run on the device driver (true) or fall back to the CPU
     *  Newton-Raphson (false)? {@code GpuDriverOuterLoopEngine} reads it to decide whether OLF's own
     *  loop must run a check the driver would otherwise have owned. */
    private boolean lastRunOnDevice;

    public boolean lastRunOnDevice() {
        return lastRunOnDevice;
    }

    /** Set by {@code GpuDriverOuterLoopEngine} on the solver it created: only then does the driver own
     *  the outer loops. Under any other engine the solve is OLF's CPU Newton-Raphson. */
    private boolean driverEngineActive;

    public void setDriverEngineActive(boolean active) {
        this.driverEngineActive = active;
        this.driverFallbackForRun = false;                   // a new engine run: the driver is back on
        this.robustModeForRun = false;
    }

    /** A solve of the CURRENT engine run fell back to the CPU Newton-Raphson (a rejected shape): the
     *  driver made no decision there and OLF's real loops took over — they hold the loop state for
     *  the rest of the run, so the device must NOT drive the loops of the run's later solves (its
     *  machines would restart from INITIAL against OLF's contexts sitting at CONTROL/COMPLETE — the
     *  AFTER_GENERATOR carrier-clamp base measured 2.2e-3 kV that way). The rest of the run is OLF's CPU
     *  Newton-Raphson. */
    private boolean driverFallbackForRun;
    /** Set by the first device reject of a run: the batch-of-one context is rebuilt in mode 6 (the
     *  robust pivoted single-matrix solve) and stays there for the run. See the retry in runImpl. */
    private boolean robustModeForRun;

    // Diagnostics for the invalidation taxonomy: how the cached native context is reconciled across
    // solves — STRUCTURE rebuilds (pattern + cuDSS re-analysis) vs cheap VALUES re-uploads (a transformer
    // tap / shunt-susceptance move keeps the pattern, only the element packs change). Process-wide totals.
    private static final java.util.concurrent.atomic.AtomicLong CONTEXT_REBUILDS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong VALUE_REFRESHES = new java.util.concurrent.atomic.AtomicLong();

    public static long contextRebuildCount() {
        return CONTEXT_REBUILDS.get();
    }

    public static long valueRefreshCount() {
        return VALUE_REFRESHES.get();
    }

    protected final NewtonRaphsonParameters parameters;

    /**
     * The run's outer-loop settings, so the batch-of-one can drive the SAME device loops the batched
     * SA drives. A batch of one is still a batch: nothing about nS == 1 justifies restricting it.
     */
    private final double driverSlackDistTol;
    private final double driverReactiveLimitTol;
    private final int driverMaxOuterIterations;
    private final int driverMaxPqPvSwitch;
    /** slackDistributionFailureBehavior other than LEAVE_ON_SLACK_BUS: the device cannot apply it (FAIL, THROW,
     *  DISTRIBUTE_ON_REFERENCE_GENERATOR), so a distribution that leaves a residual is rejected and the solve
     *  goes to the CPU, where OLF's own loop decides — the same flag the batched SA passes per scenario. */
    private final boolean driverDsRejectResidual;

    /** The OLF outer loops of the run: the driver pauses for those it does not run in-kernel. */
    private final List<com.powsybl.openloadflow.ac.outerloop.AcOuterLoop> hostOuterLoops;
    private final com.powsybl.openloadflow.ac.AcLoadFlowParameters acLoadFlowParameters;

    /** Build the bitmask for {@link GpuAcNewtonSolver#setBatchHostLoop} from the configured outer loops. */
    private int buildHostLoopMask() {
        int mask = 0;
        for (var loop : hostOuterLoops) {
            int kind = DeviceLoopKind.loopKindFor(loop.getName());
            if (kind >= DeviceLoopKind.HOST_LOOP_SVC) {
                mask |= DeviceLoopKind.hostMaskBit(kind);
            }
        }
        return mask;
    }

    /** The driver's per-controller reactive-limit state after the last solve ({@code (limitType & 7) |
     *  (switchCount << 3)}), read back so the PV↔PQ decisions can be given to the network. The limit
     *  type takes three bits because robust mode adds MIN/MAX_REALISTIC_V to MIN_Q/MAX_Q. */
    private int[] rlStateOut;

    /**
     * Give the driver's outer-loop decisions back to the network — what OLF's own loops do as a side
     * effect of running: without it OLF's world still holds the pre-loop targets and controls, the
     * converged state is measured against them and the run is declared MAX_ITERATION_REACHED
     * (measured 170 BUS_TARGET_P rows, 14 BUS_TARGET_V rows and 2 BUS_TARGET_Q rows over 1e-4 on
     * rte6515 where the same solve with the driver off leaves none), and everything downstream —
     * the batched SA's base targets, the CPU-routed contingencies — starts from the wrong network.
     */
    private void writeBackDriverMutations(GpuAcData data) {
        long handle = context.handle.get();
        double[] deviceTargets = GpuAcNewtonSolver.readBatchTargets(handle);
        if (hvdcCtls != null && !hvdcCtls.isEmpty()) {
            HvdcAcEmulation.writeBack(network, hvdcCtls, HvdcNative.readBatchHvdcStatus(handle),
                    data.getTotalElementCount());
        }
        if (monCtls != null && !monCtls.isEmpty()) {
            VoltageMonitoring.writeBack(network, monCtls, GpuAcNewtonSolver.readBatchMonitoringTargets(handle),
                    data.rlBusNum());
        }
        if (svcCtls != null) {
            com.powsybl.openloadflow.ac.outerloop.gpu.SecondaryVoltageControlLoop.writeBack(network, svcCtls,
                    GpuAcNewtonSolver.readBatchSecondaryVcTargets(handle));
        }
        lastDistributedActivePower = DistributedSlackWriteBack.writeBack(network, equationSystem, handle, driverDsParams(),
                loadAnchorTargetP, deviceTargets,
                GpuAcNewtonSolver.readBatchDsGenCum(handle), GpuAcNewtonSolver.readBatchDsRetain(handle), data);
        ReactiveLimitsWriteBack.writeBack(network, rlStateOut, data, deviceTargets,
                data.rlRemote() != null ? GpuAcNewtonSolver.readBatchRowModes(handle) : null);
        if (tvcOwned && tvcCtls != null) {
            IncrementalTransformerVoltageControl.writeBackTaps(network, tvcCtls, GpuAcNewtonSolver.readBatchTvcTaps(handle));
        }
        if (shuntOwned && shuntCtls != null) {
            IncrementalShuntVoltageControl.writeBackPositions(network, shuntCtls, GpuAcNewtonSolver.readBatchShuntPositions(handle));
        }
        if (flowOwned && flowCtls != null) {
            IncrementalFlowControl.writeBackTaps(network, flowCtls, GpuAcNewtonSolver.readBatchFlowTaps(handle));
        }
        if (cphOwned && cphCtls != null) {
            ContinuousPhaseControl.writeBack(network, cphCtls, GpuAcNewtonSolver.readBatchContinuousPhaseTaps(handle));
        }
        if (tvdOwned && tvdCtls != null) {
            com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteTransformerVoltageControl.writeBack(network, tvdCtls,
                    GpuAcNewtonSolver.readBatchDiscreteTvcState(handle));
            if (tvdFreeze != null && tvdFreeze.hasCandidates()) {
                com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteTransformerVoltageControl.writeBackFreeze(network, tvdFreeze,
                        GpuAcNewtonSolver.readBatchDiscreteTvcFreeze(handle));
            }
        }
        if (svdOwned && svdGroups != null) {
            com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteShuntVoltageControl.writeBack(network, svdGroups,
                    GpuAcNewtonSolver.readBatchDiscreteShuntState(handle), GpuAcNewtonSolver.readBatchDiscreteShuntPositions(handle));
        }
    }

    /** Active power the device's distributed-slack passes placed in the last solve, from the run's anchor (pu):
     *  what OLF's DistributedSlackOuterLoop would have accumulated. NaN when that solve distributed nothing —
     *  the targets then still hold an earlier solve's distribution, whose amount stands. */
    private double lastDistributedActivePower = Double.NaN;

    public double lastDistributedActivePower() {
        return lastDistributedActivePower;
    }

    /** Each load's target P when the device's distribution anchor was extracted (load-based balance types). */
    private java.util.Map<com.powsybl.openloadflow.network.LfLoad, Double> loadAnchorTargetP = java.util.Map.of();

    /**
     * Number of solves that bailed out of the GPU path since JVM start. Tests assert on it because a
     * silent fallback is INVISIBLE otherwise: results stay CPU-identical (that is the whole point of the
     * contract) and the warning is logged through a logger the test config does not attach an appender to. A GPU test
     * that does not check this counter can pass while never touching the GPU at all — that is exactly how
     * the BRANCH_ALPHA1 packing looked "green but uncovered" until the counter existed.
     */
    public static final java.util.concurrent.atomic.AtomicLong CPU_FALLBACK_COUNT =
            new java.util.concurrent.atomic.AtomicLong();

    private final GpuNativeContext context = new GpuNativeContext();

    private final GpuSolverInvalidation invalidation = new GpuSolverInvalidation();

    private GpuAcData cachedData;

    /**
     * Writes the GPU equation system and OLF's own dump of the same state, once, when {@code OLF_GPU_EQ_DUMP}
     * names a path. Prefers a state that carries an RL pin (row mode 4): that is the assembly a rank-deficient
     * Jacobian was traced to, and a dump of a healthy pass would not show it.
     */
    private void dumpEquationsOnce(int[] rowMode, int n) {
        String path = System.getenv("OLF_GPU_EQ_DUMP");
        if (equationsDumped || path == null || path.isEmpty() || cachedData == null) {
            return;
        }
        boolean pinned = false;
        for (int m : rowMode) {
            pinned |= m >= 4;
        }
        if (!pinned && !"any".equals(System.getenv("OLF_GPU_EQ_DUMP_WHEN"))) {
            return;                                          // wait for the pinned pass, the one under suspicion
        }
        try (java.io.Writer w = java.nio.file.Files.newBufferedWriter(java.nio.file.Path.of(path))) {
            GpuEquationDump.write(w, network, equationSystem, cachedData, rowMode);
        } catch (java.io.IOException | RuntimeException e) {
            LOGGER.warn("OLF_GPU_EQ_DUMP failed: {}", e.toString());
        }
        try (java.io.Writer w = java.nio.file.Files.newBufferedWriter(java.nio.file.Path.of(path + ".olf"))) {
            equationSystem.write(w, true);                   // the CPU side's own reading, inactive ones in [ ]
        } catch (java.io.IOException | RuntimeException e) {
            LOGGER.warn("OLF_GPU_EQ_DUMP (.olf) failed: {}", e.toString());
        }
        equationsDumped = true;
        StringBuilder m4 = new StringBuilder();
        for (int r = 0; r < rowMode.length; r++) {
            if (rowMode[r] == 4) {
                m4.append(' ').append(r);
            }
        }
        // The HOST's pinned rows, to line up against the device's (OLF_GPU_VALS_SCAN prints mode4Rows per
        // fill): if they differ, the device moved a pin, and the row it moved it to is the defect.
        System.err.println("EQ_DUMP wrote " + path + " and " + path + ".olf (n=" + n + " pinned=" + pinned
                + ") hostMode4Rows=[" + m4.toString().trim() + "]");
    }

    /** The remote gen-VC pin set the cached extraction's DISTR_Q weights were built from (see runImpl). */
    private java.util.BitSet pinnedAtExtraction;

    /** OLF_GPU_EQ_DUMP: written once per run (see dumpEquationsOnce). */
    private boolean equationsDumped;

    /** OLF_GPU_F0_TRACE: written once per run (see runImpl). */
    private boolean f0Traced;

    private GpuAcActivation activation;

    // per-row variable kind (0=BUS_PHI, 1=BUS_V, 2=other) for the MAX_VOLTAGE_CHANGE device step
    // clamp; structural, recomputed whenever the activation/structure is rebuilt
    private int[] varType;

    // PER_EQUATION_TYPE row thresholds need (re)installing on the context: after a context rebuild
    // and after any activation change (the thresholds follow the variables at each row)
    private boolean criteriaDirty = true;

    /** The shared outer-loop tables ({@link DeviceLoopTables}) follow the extraction. */
    private boolean tablesDirty = true;

    // The driver phases with their own per-run tables (HVDC AC emulation, MonitoringVoltage, in-kernel
    // incremental TVC): built at every solve from the network as it stands, written back after it.
    private List<HvdcAcEmulation.Ctl> hvdcCtls;
    private List<VoltageMonitoring.Ctl> monCtls;
    private List<IncrementalTransformerVoltageControl.Controller> tvcCtls;
    private boolean tvcOwned;
    private List<IncrementalShuntVoltageControl.Controller> shuntCtls;
    private boolean shuntOwned;
    private List<IncrementalFlowControl.Controller> flowCtls;
    private boolean flowOwned;
    private List<ContinuousPhaseControl.Ctl> cphCtls;
    private boolean cphOwned;
    private List<com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteTransformerVoltageControl.TransformerController> tvdCtls;
    private boolean tvdOwned;
    private com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteTransformerVoltageControl.FreezeTables tvdFreeze;
    /** SecondaryVoltageControl's in-kernel tables — the same phase the batched SA drives by default. */
    private com.powsybl.openloadflow.ac.outerloop.gpu.SecondaryVoltageControlLoop.Controllers svcCtls;
    private List<com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteShuntVoltageControl.Group> svdGroups;
    private boolean svdOwned;
    /** The host-loop bits the CURRENT solve actually pauses for (in-kernel phases clear theirs). */
    private int activeHostMask;

    /** The host-paused loops' (loop, context) pairs of the CURRENT engine run: OLF keeps one context
     *  per loop for the whole run (iteration counters, per-loop data), so they must survive a restart
     *  of the device sequence — created on the first pause, cleaned up when the run completes. */
    private java.util.Map<Integer, org.apache.commons.lang3.tuple.Pair<
            com.powsybl.openloadflow.ac.outerloop.AcOuterLoop, com.powsybl.openloadflow.ac.AcOuterLoopContext>> hostLoopPairs;

    /** Per controller bus, the driver's RL state carried across a restart (OLF's switch counters
     *  persist for the whole outer-loop run), keyed by bus num since the entry order may change. */
    private java.util.Map<Integer, Integer> rlCarry;

    /** Newton iterations of this run's device sequences that a host-loop restart closed (see runNewtonIterations). */
    private int restartIterations;

    /**
     * The Newton iterations of the whole run, as OLF's engine counts them: the sum over every solve of the
     * outer-loop sequence. One device call runs many of those solves (the in-kernel passes and the host-loop
     * resumes), and the device sums them per scenario; a host-loop restart starts a new device sequence,
     * so the closed ones are carried. OLF writes a load flow's result back to the network only when this
     * is positive, so the count of the last pass alone (0 for a resume that only confirmed) lost results.
     */
    private int runNewtonIterations(int lastCallIterations) {
        int[] stats = GpuAcNewtonSolver.readBatchOlStats(context.handle.get());
        return restartIterations + (stats != null && stats.length >= 3 ? stats[2] : lastCallIterations);
    }

    /** On a restart, the host loop the sequence resumes at (the one that went UNSTABLE). */
    private int restartLoopKind;

    private com.powsybl.openloadflow.ac.outerloop.AcOuterLoop driverLoop(String name) {
        if (acLoadFlowParameters == null) {
            return null;
        }
        for (var ol : acLoadFlowParameters.getOuterLoops()) {
            if (name.equals(ol.getName())) {
                return ol;
            }
        }
        return null;
    }

    private com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop driverDsLoop() {
        if (acLoadFlowParameters == null) {
            return null;
        }
        for (var ol : acLoadFlowParameters.getOuterLoops()) {
            if (ol instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop ds) {
                return ds;
            }
        }
        return null;
    }

    private com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop driverRlLoop() {
        if (acLoadFlowParameters == null) {
            return null;
        }
        for (var ol : acLoadFlowParameters.getOuterLoops()) {
            if (ol instanceof com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop rl) {
                return rl;
            }
        }
        return null;
    }

    /** The run's own DistributedSlack settings (balance type, active limits) — never defaults. */
    private com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams driverDsParams() {
        var ds = driverDsLoop();
        return ds == null
                ? com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams.DEFAULT
                : new com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams(
                        ds.getActivePowerDistribution().getBalanceType(), ds.getActivePowerDistribution().isUseActiveLimits(),
                        true);                               // OLF re-derives from the INITIAL targets at every check
    }

    /** Install (or clear) the PER_EQUATION_TYPE per-row stopping thresholds on the batch context,
     *  from the run's actual criteria object — the device then stops on the SAME decision as OLF's
     *  PerEquationTypeStoppingCriteria (every row's |F| under its type threshold). */
    private void installRowCriteria(int n) {
        if (parameters.getStoppingCriteria() instanceof PerEquationTypeStoppingCriteria c) {
            double[][] thr = GpuAcActivationExtractor.extractRowCriteria(network, equationSystem, activation, n,
                    c.getMaxActivePowerMismatch(), c.getMaxReactivePowerMismatch(), c.getMaxVoltageMismatch(),
                    c.getMaxDefaultAngleMismatch(), c.getMaxDefaultRatioMismatch(),
                    c.getMaxDefaultSusceptanceMismatch(), c.getConvEpsPerEq());
            GpuAcNewtonSolver.installRowCriteria(context.handle.get(), thr);
        } else {
            GpuAcNewtonSolver.setBatchRowCriteria(context.handle.get(), null, null);
        }
    }

    @SuppressWarnings("java:S107")
    public GpuNewtonRaphson(LfNetwork network, NewtonRaphsonParameters parameters,
                            EquationSystem<AcVariableType, AcEquationType> equationSystem,
                            JacobianMatrix<AcVariableType, AcEquationType> j,
                            TargetVector<AcVariableType, AcEquationType> targetVector,
                            EquationVector<AcVariableType, AcEquationType> equationVector,
                            boolean detailedReport,
                            double driverSlackDistTol, double driverReactiveLimitTol,
                            int driverMaxOuterIterations, int driverMaxPqPvSwitch,
                            List<com.powsybl.openloadflow.ac.outerloop.AcOuterLoop> hostOuterLoops,
                            com.powsybl.openloadflow.ac.AcLoadFlowParameters acLoadFlowParameters) {
        super(network, equationSystem, j, targetVector, equationVector, detailedReport);
        this.parameters = Objects.requireNonNull(parameters);
        this.driverSlackDistTol = driverSlackDistTol;
        this.driverReactiveLimitTol = driverReactiveLimitTol;
        this.driverMaxOuterIterations = driverMaxOuterIterations;
        this.driverMaxPqPvSwitch = driverMaxPqPvSwitch;
        this.hostOuterLoops = Objects.requireNonNull(hostOuterLoops);
        this.acLoadFlowParameters = acLoadFlowParameters;
        this.driverDsRejectResidual = acLoadFlowParameters != null
                && acLoadFlowParameters.getSlackDistributionFailureBehavior()
                        != com.powsybl.openloadflow.OpenLoadFlowParameters.SlackDistributionFailureBehavior.LEAVE_ON_SLACK_BUS;
        EquationSystemIndex<AcVariableType, AcEquationType> index = equationSystem.getIndex();
        // Purge listeners of ALREADY-COLLECTED solvers synchronously (we are on the solve thread,
        // no dispatch in flight) — this caps accumulation on a long-lived LfNetwork at ~1 while
        // keeping the cleaner thread away from the (non-thread-safe) listener lists.
        for (LfNetworkListener l : List.copyOf(network.getListeners())) {
            if (l instanceof GpuSolverInvalidation inv && inv.dead) {
                network.removeListener(l);
            }
        }
        index.addListener(invalidation);
        network.addListener(invalidation);
        // the cleaning action must not reference the solver itself (it would never become
        // phantom-reachable); the Invalidation listener object is referenced instead. It only
        // dead-flags: the index-side listener stays registered (no-op) for the es lifetime.
        GpuSolverInvalidation listener = invalidation;
        context.removeListeners = () -> listener.dead = true;
        CLEANER.register(this, context);
    }

    /** The native batch context handle, for the engine's pause/resume protocol with the driver.
     *  Zero until the first {@link #run} builds the context. */
    public long getContextHandle() {
        return context.handle.get();
    }

    @Override
    public String getName() {
        return "GPU Newton-Raphson";
    }

    @Override
    public AcSolverResult run(VoltageInitializer voltageInitializer, ReportNode reportNode) {
        if (!driverEngineActive || driverFallbackForRun) {
            // Not driven by the driver engine, or an earlier solve of this run fell back: OLF's own
            // Newton-Raphson, with OLF's loops around it.
            lastRunOnDevice = false;
            return cpuNewtonRaphson(voltageInitializer, reportNode);
        }
        hostLoopPairs = null;
        rlCarry = null;
        restartIterations = 0;
        (parameters instanceof GpuNewtonRaphsonParameters gpu ? gpu.getGpuParameters() : new GpuLoadFlowParameters())
                .install();
        try {
            return runImpl(voltageInitializer, reportNode, false);
        } finally {
            if (hostLoopPairs != null) {                     // OLF's engine cleans its loops up in order
                for (var entry : hostLoopPairs.values()) {
                    entry.getLeft().cleanup(entry.getRight());
                }
                hostLoopPairs = null;
            }
        }
    }

    /**
     * One attempt of the device sequence. A host-paused loop that changes the equation layout, the
     * row modes or the element values (a tap or section move, a control it enabled or disabled)
     * voids the paused sequence: OLF re-solves on the mutated system and restarts its sweep, so this
     * returns through {@code restart = true} — no initializer (the state vector already holds the
     * device's last state), the extraction re-run, the driver's RL state carried by bus.
     */
    private AcSolverResult runImpl(VoltageInitializer voltageInitializer, ReportNode reportNode, boolean restart) {
        if (!GpuAcNewtonSolver.isAvailable()) {
            throw new PowsyblException("GPU Newton-Raphson requested but the native library is not available "
                    + "(build with native/build-gpu.sh and provide cuDSS — see GpuAcNewtonSolver)");
        }
        // RESUME: the engine's pause/resume cycle already called setBatchDriverResume and
        // setBatchTargetDeltas — just re-enter solveBatchContext (the native side reads
        // driverResume=1 and continues from its paused position). Skip initStateVector and
        // the context rebuild — the state is on the device where we left it.
        boolean resuming = GpuAcNewtonSolver.readBatchDriverPaused(context.handle.get()) != 0;
        if (resuming) {
            return runResume(reportNode);
        }
        // A restart is OLF's post-check re-solve: runOuterLoop starts it from the NETWORK's last state
        // (PreviousValueVoltageInitializer), which the pause cycle wrote with updateNetwork — and the
        // equation layout may have changed (a control the host loop disabled removes its variable),
        // so the previous state vector is not even the right length any more.
        AcSolverUtil.initStateVector(network, equationSystem,
                restart ? new com.powsybl.openloadflow.network.util.PreviousValueVoltageInitializer() : voltageInitializer);

        AcSolverStatus status;
        int iterations = 0;
        int n = equationSystem.getIndex().getSortedVariablesToFind().size();
        try {
            long t0 = System.nanoTime();
            boolean rebuilt = false;
            // A REACTIVE-LIMIT PIN of a remote gen-VC member (row mode 4) renormalizes OLF's DISTR_Q reactive
            // keys (updateReactiveKeys) and fires NO equation-system event, so the cached extraction would keep
            // its pre-pin static qPct. Watch the pinned SET and re-extract when it changes: the fresh pack's
            // dqWeight differs, sameStructureAs compares it, so the structure is rebuilt with OLF's renormalized
            // keys. That is what used to make every post-pin solve CPU-routed; it now solves on the device.
            java.util.BitSet pinnedNow = GpuAcActivationExtractor.pinnedRemoteMembers(network, equationSystem);
            if (!pinnedNow.equals(pinnedAtExtraction)) {
                invalidation.structureDirty = true;
            }
            if (invalidation.structureDirty) {
                // validate + build, through the shared cache (the batched SA reuses this extraction)
                // The RUN's outer-loop set and its DistributedSlack settings — the device must not
                // distribute with a default balance type, nor run a loop the run does not configure.
                GpuAcData data = GpuAcDataCache.extract(network, equationSystem,
                        driverDsLoop() != null, driverRlLoop() != null,
                        driverDsParams());
                // Three-level reconciliation, decided by the DATA not the event: a changed CSR pattern
                // (index packs) is STRUCTURE -> rebuild + cuDSS re-analysis; an unchanged pattern with
                // only changed element PARAMETERS (tap r1/a1, shunt b) is VALUES -> a cheap in-place
                // pack re-upload reusing the analysis; an unchanged data is a no-op (context reused).
                if (!data.sameStructureAs(cachedData)) {
                    GpuAcNewtonSolver.destroyBatchContext(context.handle.getAndSet(0));
                    // BATCH-OF-1: the load flow runs on a BATCH context of one scenario, so per-row
                    // PER_EQUATION_TYPE criteria, slope-SVC (mode 2), continuous phase (mode 3) and
                    // the vdl/ZIP fills come from the batch pipeline instead of being ported twice.
                    // The shared structure comes from the per-extraction cache — the batched SA
                    // building its context from the SAME extraction reuses it device-side.
                    context.handle.set(GpuAcNewtonSolver.createBatchContext(GpuAcStructures.handle(n, data), 1,
                            robustModeForRun ? GpuBatchContextCache.MODE_ROBUST : GpuBatchContextCache.resolveBatchMode(),
                            data.rlVrow(), data.rlMinQ(), data.rlMaxQ(), data.rlLoadQ(), data.rlVtarget(),
                            data.rlMonitored(), data.rowSlope(),
                            data.dsPhiRow(), data.dsFactor(), data.dsSlackPhiRow(), data.dsSlackTargetP(),
                            data.dsBaseTargetP(), data.dsMaxDelta(), data.dsMinDelta(),
                            data.dsMode(), data.dsMarginUp(), data.dsMarginDown(),
                            data.dsSlackMarginUp(), data.dsSlackMarginDown()));
                    // OLF's forced-PV election keys are part of creating a context with RL entries.
                    DeviceLoopTables.installRlElection(context.handle.get(), data);
                    cachedData = data;
                    loadAnchorTargetP = DistributedSlackWriteBack.captureLoadAnchors(network);
                    rebuilt = true;
                    criteriaDirty = true;
                    tablesDirty = true;
                    CONTEXT_REBUILDS.incrementAndGet();
                } else if (!data.sameAs(cachedData)) {
                    GpuAcNewtonSolver.refreshStructureValues(GpuAcStructures.handle(n, cachedData),
                            data.cbIn(), data.obIn(), data.shIn(), data.hvIn(), data.ziIn());
                    // the SHARED structure now holds the new values — re-key it to the new
                    // extraction so content matching never aliases it under the old one
                    GpuAcStructures.rebind(cachedData, data);
                    cachedData = data;
                    loadAnchorTargetP = DistributedSlackWriteBack.captureLoadAnchors(network);
                    // The election keys are per-extraction (nominal V, live target P, id rank), so a
                    // value refresh re-installs them too.
                    DeviceLoopTables.installRlElection(context.handle.get(), data);
                    tablesDirty = true;
                    VALUE_REFRESHES.incrementAndGet();
                }
                pinnedAtExtraction = pinnedNow;           // the packs now carry this pin set's qPct
                invalidation.structureDirty = false;
                invalidation.activationDirty = true;
            }
            if (invalidation.activationDirty) {
                activation = GpuAcActivationExtractor.extractActivation(network, equationSystem, n);
                varType = GpuAcActivationExtractor.extractVarType(equationSystem, n);
                invalidation.activationDirty = false;
                criteriaDirty = true;                        // per-row thresholds follow the variables
            }
            if (criteriaDirty) {
                installRowCriteria(n);
                criteriaDirty = false;
            }
            // OLF_GPU_EQ_DUMP=<path>: the GPU equation system as TEXT, next to OLF's own write() of the same
            // state (<path>.olf). OLF can print its equations, so a wrong one is readable there; ours only
            // existed as index packs until now. Written once per run, at the FIRST solve carrying a pin
            // (row mode 4) when there is one, since that is the assembly under suspicion.
            dumpEquationsOnce(activation.rowMode(), n);
            // Since the batch-of-1 route (2026-09-08) modes 2 (slope-SVC) and 3 (continuous phase) solve
            // on-device. Mode 4 (a PINNED remote gen-VC member) used to be CPU-routed here because the pack's
            // DISTR_Q weights were the STATIC extraction-time qPct while OLF's pin renormalizes them; the
            // pinned-set watch above re-extracts on a pin, so the weights are OLF's and the solve stays on
            // the device. Anything ABOVE the modes the kernels implement is still a fallback, not a guess.
            for (int mode : activation.rowMode()) {
                if (mode > 4) {
                    throw new UnsupportedOperationException(
                            "row mode " + mode + " is not implemented on-device");
                }
            }
            double[] target = activation.applyTargets(targetVector.getArray());
            double[] x0 = equationSystem.getStateVector().get().clone();
            // OLF_GPU_F0_TRACE=<path>: OLF's OWN mismatch vector at x0 (f(x0) - target, the residual its first
            // |f(x0)| line is the norm of), one line per equation with its type, element and id. Lined up
            // against the device's first fill (OLF_GPU_VALS_DUMP_ANY) it names the equation the two arms
            // evaluate differently BEFORE any step is taken — the only place a difference is not yet blurred
            // by different iterates. Written once per run.
            String f0Path = System.getenv("OLF_GPU_F0_TRACE");
            if (f0Path != null && !f0Path.isEmpty() && !f0Traced) {
                f0Traced = true;
                double[] fx0 = equationVector.getArray();
                double[] tg0 = targetVector.getArray();
                double n2 = 0;
                try (java.io.PrintWriter pw = new java.io.PrintWriter(java.nio.file.Files.newBufferedWriter(java.nio.file.Path.of(f0Path)))) {
                    for (int c = 0; c < fx0.length; c++) {
                        double v = fx0[c] - (c < tg0.length ? tg0[c] : 0.0);
                        n2 += v * v;
                        var eq = equationSystem.getIndex().getEquationAtColumn(c);
                        String id;
                        try {
                            id = switch (eq.getType().getElementType()) {
                                case BUS -> network.getBus(eq.getElementNum()).getId();
                                case BRANCH -> network.getBranch(eq.getElementNum()).getId();
                                default -> "#" + eq.getElementNum();
                            };
                        } catch (RuntimeException e) {
                            id = "#" + eq.getElementNum();
                        }
                        pw.println(c + " " + eq.getType() + " " + eq.getElementNum() + " " + id + " " + v);
                    }
                } catch (java.io.IOException e) {
                    LOGGER.warn("OLF_GPU_F0_TRACE failed: {}", e.toString());
                }
                System.err.println("F0_TRACE wrote " + f0Path + " rows=" + fx0.length + " norm2=" + Math.sqrt(n2));
            }
            rlStateOut = cachedData.rlVrow().length > 0 ? new int[cachedData.rlVrow().length] : null;
            int[] rlStateIn = null;
            if (restart && rlCarry != null && rlStateOut != null) {
                rlStateIn = new int[rlStateOut.length];
                for (int i = 0; i < rlStateIn.length; i++) {
                    rlStateIn[i] = rlCarry.getOrDefault(cachedData.rlBusNum()[i], 0);
                }
            }
            // Device-side state-vector scaling: NONE = undamped Newton; LINE_SEARCH backtracks on the
            // device (mirrors LineSearchStateVectorScaling); MAX_VOLTAGE_CHANGE clamps each step so the
            // largest |Δv|/|Δφ| stays within maxDv/maxDphi (mirrors MaxVoltageChangeStateVectorScaling).
            int lsMode = switch (parameters.getStateVectorScalingMode()) {
                case NONE -> 0;
                case LINE_SEARCH -> 1;
                case MAX_VOLTAGE_CHANGE -> 2;
            };
            long t1 = System.nanoTime();
            double convEpsPerEq = GpuSolveSettings.deviceConvEpsPerEq(parameters);
            if (tablesDirty) {
                // The SAME run-constant outer-loop tables the batched SA installs on its chunk
                // contexts — a batch of one drives the same loops as a batch of five hundred.
                var dsLoop = driverDsLoop();
                ReactiveCapabilityCurves.Channel rlCurves = driverRlLoop() != null
                        ? ReactiveCapabilityCurves.build(network, equationSystem, cachedData, driverDsParams(), dsLoop != null, target,
                                acLoadFlowParameters.getNetworkParameters().isExtrapolateReactiveLimits())
                        : null;
                // forceTargetQInReactiveLimits: the base load flow's slack moves P too, and OLF re-clamps.
                var lfNetworkParameters = acLoadFlowParameters.getNetworkParameters();
                ForcedTargetQ.Channel forcedTargetQ = driverRlLoop() != null && dsLoop != null
                        ? ForcedTargetQ.build(network, equationSystem, cachedData, driverDsParams(),
                                lfNetworkParameters.isForceTargetQInReactiveLimits() && lfNetworkParameters.isReactiveLimits(),
                                true, target, activation.rowMode(), lfNetworkParameters.isExtrapolateReactiveLimits())
                        : null;
                DeviceLoopTables.install(context.handle.get(), cachedData, rlCurves, forcedTargetQ,
                        acLoadFlowParameters.isVoltageRemoteControlRobustMode(),
                        acLoadFlowParameters.getMinRealisticVoltage(), acLoadFlowParameters.getMaxRealisticVoltage(),
                        acLoadFlowParameters.getMinNominalVoltageRealisticVoltageCheck(),
                        DeviceLoopTables.realisticFixingLoop(acLoadFlowParameters));
                tablesDirty = false;
            }
            if (lsMode == 1) {                               // batch loop LINE_SEARCH (ported 2026-09-08)
                GpuAcNewtonSolver.setBatchLineSearch(context.handle.get(),
                        parameters.getLineSearchStateVectorScalingMaxIteration(),
                        parameters.getLineSearchStateVectorScalingStepFold());
            }
            // AC-emulation HVDC statuses: OLF's freezing/limits loops DEACTIVATE the hvdc coupling
            // terms for any non-LINEAR status (the injections ride the target vector as VSC station
            // targetP, re-extracted every solve) — mirror it with the hv element's disabled bit,
            // which fillHvdcBatch honors. Without this the fill emitted the LINEAR coupling on top
            // of the frozen/saturated injections (the former frozen-HVDC cpu-base-only condition).
            boolean[][] elementMask = new boolean[1][cachedData.getTotalElementCount()];
            List<LfHvdc> hvPacked = network.getHvdcs().stream()
                    .filter(h -> h.getBus1() != null && h.getBus2() != null && h.isAcEmulation())
                    .toList();                               // the packer's exact enumeration order
            int hvOffset = cachedData.getTotalElementCount() - hvPacked.size();
            for (int h = 0; h < hvPacked.size(); h++) {
                if (hvPacked.get(h).getAcEmulationControl().getAcEmulationStatus()
                        != LfHvdc.AcEmulationControl.AcEmulationStatus.LINEAR_MODE) {
                    elementMask[0][hvOffset + h] = true;
                }
            }
            int[] ovIdx = new int[0];
            double[] ovVal = new double[0];
            hvdcCtls = null;
            monCtls = null;
            tvcCtls = null;
            tvcOwned = false;
            shuntCtls = null;
            shuntOwned = false;
            flowCtls = null;
            flowOwned = false;
            cphCtls = null;
            cphOwned = false;
            tvdCtls = null;
            tvdOwned = false;
            tvdFreeze = null;
            svdGroups = null;
            svdOwned = false;
            long handle = context.handle.get();
            int nElements = cachedData.getTotalElementCount();
            // Every phase's tables, from the SAME builder the batched SA calls (#58). The monitoring
            // and transformer builders probe controls with an on/off round trip that fires network
            // events, so the whole call sits inside our dirty-flag guard.
            var tvcAfterGenLoop = driverLoop("TransformerVoltageControl");
            var incTvcLoop = driverLoop("IncrementalTransformerVoltageControl");
            var incShuntLoop = driverLoop("IncrementalShuntVoltageControl");
            boolean sdAll = invalidation.structureDirty;
            boolean adAll = invalidation.activationDirty;
            var phases = com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases.build(
                    new com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases.Request(
                            driverLoop("VoltageMonitoring") != null,
                            incTvcLoop != null,
                            incShuntLoop != null,
                            driverLoop("ShuntVoltageControl") != null,
                            driverLoop("IncrementalTransformerReactivePowerControl") != null,
                            driverLoop("IncrementalPhaseControl") != null,
                            driverLoop("IncrementalTransformerReactivePowerControl")
                                    instanceof com.powsybl.openloadflow.ac.outerloop.IncrementalTransformerReactivePowerControlOuterLoop r
                                    ? r.getMaxTapShift() : 1,
                            driverLoop("PhaseControl") != null,
                            tvcAfterGenLoop != null || driverLoop("SimpleTransformerVoltageControl") != null,
                            tvcAfterGenLoop instanceof com.powsybl.openloadflow.ac.outerloop.TransformerVoltageControlOuterLoop tg
                                    && tg.isUseInitialTapPosition(),
                            tvcAfterGenLoop instanceof com.powsybl.openloadflow.ac.outerloop.TransformerVoltageControlOuterLoop tg2
                                    ? tg2.getMaxControlledNominalVoltageOverride() : -1,
                            driverLoop("SecondaryVoltageControl") != null,
                            acLoadFlowParameters.getNetworkParameters().getMinPlausibleTargetVoltage(),
                            acLoadFlowParameters.getNetworkParameters().getMaxPlausibleTargetVoltage()),
                    network, equationSystem, targetVector, activation, target,
                    cachedData.rlVrow(), cachedData.rlBusNum(), n);
            invalidation.structureDirty = sdAll;
            invalidation.activationDirty = adAll;
            // HVDC AC emulation (freeze at OLF order 4, limits at 5) as driver phases. The run starts from the
            // statuses the network holds: OLF's FreezingHvdcACEmulation.initialize has already run when the
            // engine built its loops, and froze exactly the emulations it freezes — those whose two bus angles
            // are known, at that flow (a fresh network's NaN angles freeze nothing), with the frozen injection
            // in the targets. A FROZEN status enters as the base status, unfrozen by the driver's own phase; a
            // LINEAR one stays linear, also on a host-loop restart, which OLF does not re-initialize either.
            boolean hvdcLimits = driverLoop("AcHvdcAcEmulationLimits") != null;
            boolean hvdcFreeze = driverLoop("FreezingHvdcACEmulation") != null;
            if (hvdcLimits || hvdcFreeze) {
                hvdcCtls = HvdcAcEmulation.buildControllers(network, equationSystem, nElements, cachedData);
            }
            if (hvdcCtls != null && !hvdcCtls.isEmpty()) {
                int nh = hvdcCtls.size();
                int[][] hvStatus = new int[1][nh];
                double[] appliedD1 = new double[nh];
                double[] appliedD2 = new double[nh];
                for (var ctl : hvdcCtls) {
                    elementMask[0][ctl.hvElemIndex()] = false;   // set back below from the status, as a BASE status
                }
                HvdcAcEmulation.seedBaseStatuses(1, 1, hvdcCtls, hvStatus, elementMask);
                HvdcAcEmulation.installDriverLoops(handle, hvdcCtls, 1, hvStatus, elementMask,
                        appliedD1, appliedD2, hvdcLimits);
            } else {
                HvdcAcEmulation.installDriverLoops(handle, null, 1, null, null, null, null, false);
            }
            // MonitoringVoltage (OLF order 6) as a driver phase.
            monCtls = phases.monitoring();
            VoltageMonitoring.installDriverLoop(handle, monCtls);
            int hostMask = buildHostLoopMask();
            // SecondaryVoltageControl (OLF order 4) IN-KERNEL — the same phase, built the same way,
            // as the batched security analysis. It used to be the one loop the base load flow never
            // handed to the driver, so every network with a zone paused it here while the chunk path
            // ran it on device; a pause whose mutation dirties the structure is also what forces the
            // driver RESTART, so the divergence was not only a throughput one.
            svcCtls = phases.secondaryVc();
            if (com.powsybl.openloadflow.ac.outerloop.gpu.SecondaryVoltageControlLoop.uploadToDevice(handle, svcCtls)) {
                hostMask &= ~1;                          // bit 0 = HOST_LOOP_SVC
            } else {
                svcCtls = null;                          // refused: OLF's own loop runs at the pause
            }
            // The incremental and discrete transformer, shunt and phase loops (OLF orders 7 to 10) IN-KERNEL,
            // installed exactly as for a security analysis chunk. The controls OLF's initialize enabled on the
            // network are in the activation already (entryOn): AFTER_GENERATOR starts with every control off
            // and the driver's INITIAL/CONTROL/COMPLETE machine enables, clamps and rounds; WITH_GENERATOR
            // starts with the controls ON and the first check rounds or dispatches.
            var itvc = driverLoop("IncrementalTransformerVoltageControl")
                    instanceof com.powsybl.openloadflow.ac.outerloop.IncrementalTransformerVoltageControlOuterLoop l ? l : null;
            var isvc = driverLoop("IncrementalShuntVoltageControl")
                    instanceof com.powsybl.openloadflow.ac.outerloop.IncrementalShuntVoltageControlOuterLoop l ? l : null;
            var rpcLoop = driverLoop("IncrementalTransformerReactivePowerControl");
            boolean incPhase = driverLoop("IncrementalPhaseControl") != null;
            boolean cph = driverLoop("PhaseControl") != null;
            var tvcAfterGen = driverLoop("TransformerVoltageControl");
            var tvcSimple = driverLoop("SimpleTransformerVoltageControl");
            boolean afterGen = tvcAfterGen instanceof com.powsybl.openloadflow.ac.outerloop.TransformerVoltageControlOuterLoop;
            boolean svd = driverLoop("ShuntVoltageControl") != null;
            var installed = com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhaseInstaller.install(handle,
                    new com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhaseInstaller.Request(
                            itvc != null ? phases.incrementalTvc() : null, itvc != null ? itvc.getMaxTapShift() : 1,
                            rpcLoop != null || incPhase ? phases.flow() : null, cph ? phases.continuousPhase() : null,
                            isvc != null ? phases.incrementalShunt() : null, isvc != null ? isvc.getMaxSectionShift() : 3,
                            tvcAfterGen != null || tvcSimple != null ? phases.discreteTvc() : null, tvcSimple != null,
                            afterGen, phases.freeze(), svd ? phases.simpleShunt() : null, true));
            if (itvc != null) {
                tvcCtls = phases.incrementalTvc();
            }
            if (isvc != null) {
                shuntCtls = phases.incrementalShunt();
            }
            if (rpcLoop != null || incPhase) {
                flowCtls = phases.flow();
            }
            if (cph) {
                cphCtls = phases.continuousPhase();
            }
            if (tvcAfterGen != null || tvcSimple != null) {
                tvdCtls = phases.discreteTvc();
            }
            if (svd) {
                svdGroups = phases.simpleShunt();
            }
            tvcOwned = installed.incrementalTvc();
            shuntOwned = installed.incrementalShunt();
            flowOwned = installed.flow();
            cphOwned = installed.continuousPhase();
            tvdOwned = installed.discreteTvc();
            svdOwned = installed.simpleShunt();
            if (tvdOwned && afterGen) {
                // Batch of one on the live network: its own topology IS the scenario, so the
                // connectivity masks come from an empty contingency.
                tvdFreeze = phases.freeze();
                com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteTransformerVoltageControl.uploadFixMasks(
                        handle, tvdFreeze, java.util.List.of(), 1);
            }
            if (tvcOwned || tvdOwned) {
                hostMask &= ~(DeviceLoopKind.hostMaskBit(DeviceLoopKind.HOST_LOOP_TVC));
            }
            if (shuntOwned || svdOwned) {
                hostMask &= ~(DeviceLoopKind.hostMaskBit(DeviceLoopKind.HOST_LOOP_SHUNT));
            }
            if (flowOwned && rpcLoop != null) {
                hostMask &= ~(DeviceLoopKind.hostMaskBit(DeviceLoopKind.HOST_LOOP_TRPC));
            }
            if (flowOwned && incPhase || cphOwned) {
                hostMask &= ~(DeviceLoopKind.hostMaskBit(DeviceLoopKind.HOST_LOOP_PHASE));
            }
            // Host-owned outer-loop slots: the driver pauses at their OLF positions and the
            // pause/resume cycle after solveBatchContext runs OLF's loop there. 0 clears a cached mask.
            // All-or-nothing, the same rule the batched SA applies: a loop the builders represented only
            // PARTIALLY goes back to OLF's engine at its own slot, rather than the device driving a
            // subset of it while the pause bit says nobody has to.
            hostMask = com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhaseOwnership.enforceComplete(
                    handle, hostMask);
            activeHostMask = hostMask;
            GpuAcNewtonSolver.setBatchHostLoop(handle, hostMask);
            if (restart && restartLoopKind > 0) {        // re-check the loop that went UNSTABLE first
                GpuAcNewtonSolver.setBatchDriverStartLoop(handle, restartLoopKind);
            }
            restartLoopKind = 0;
            if (driverDsRejectResidual && driverDsLoop() != null && cachedData.dsPhiRow().length == 0
                    && (cachedData.dsSlackPhiRow() < 0 || Double.isInfinite(cachedData.dsSlackRoomUp()))) {
                // No participant row and no slack generator to bound the slack's own share: either nobody
                // participates (the pack dropped the distribution, the mismatch simply stays at the slack) or the
                // slack's share has no generator limits (a load-based slack). The device cannot tell where OLF's
                // water-fill leaves a remaining mismatch, so OLF's own loop decides this failure behaviour. A
                // slack-only distribution over the slack's generators is bounded by their rooms on the device.
                throw new PowsyblException("GPU driver: a distribution without participant rows cannot apply "
                        + "slackDistributionFailureBehavior " + acLoadFlowParameters.getSlackDistributionFailureBehavior()
                        + "; the solve goes to the CPU");
            }
            double[] out = GpuAcNewtonSolver.solveBatchContext(context.handle.get(),
                    x0, target, new int[][] {activation.rowMode()}, activation.rowMode(),
                    elementMask,
                    ovIdx, ovVal,
                    new int[0], new double[0], new double[0], new double[0], new double[0], null, null,
                    null, null, null, null, null, null,
                    parameters.getMaxIterations(), GPU_TOLERANCE,
                    driverReactiveLimitTol, driverSlackDistTol, convEpsPerEq,
                    lsMode,
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDv(),
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDphi(),
                    varType, null,
                    null, null, rlStateIn, rlStateOut, null, null,
                    driverMaxOuterIterations, driverMaxPqPvSwitch, driverDsRejectResidual ? 1 : 0,
                    0,                                       // skipRlPhase: the driver owns RL
                    0, null, null)[0];
            long t2 = System.nanoTime();
            if (out.length > n + 2 && (int) out[n + 2] == GpuBatchStatus.STATUS_LINEAR_SOLVE_FAILED && !robustModeForRun) {
                // A FAILED LINEAR SOLVE (status -3, and only that: every other reject names a state
                // the device cannot lay out or OLF's own budget, and goes to the CPU as before -
                // retrying those on the device re-solved bases OLF's loops had to decide, 2.6e-3 kV on
                // the robust-realistic fixture): retry ON THE DEVICE in the robust single-matrix mode
                // (mode 6: BTF-COLAMD, global column pivoting, matching) for the rest of the run
                // before any CPU fallback. rte6515's full busbar structure does not factor in the
                // uniform batch (linear residual ratio 1e12, the base came back MAX_ITERATION_REACHED
                // and the whole analysis reported nothing) and converges on the device in mode 0.
                // A base load flow that works on the CPU has to work on the GPU.
                LOGGER.info("GPU Newton-Raphson: device rejected the batch-of-one solve; retrying on the device in the robust single-matrix mode");
                if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                    System.err.println("GPU_BASE_ROBUST_RETRY status=-2 -> mode 6 (pivoted single matrix)");
                }
                robustModeForRun = true;
                GpuAcNewtonSolver.destroyBatchContext(context.handle.getAndSet(0));
                cachedData = null;
                invalidation.structureDirty = true;
                return runImpl(voltageInitializer, reportNode, true);
            }
            if (out.length > n + 2 && (int) out[n + 2] <= GpuBatchStatus.STATUS_REJECTED) {
                // The kernel rejected the scenario (status -2, or -3 when the robust retry failed too): a linear solve that did not solve (the
                // structure does not factor unpivoted), the straggler exit, the outer-loop budget, an
                // unrealistic state or a device-loop reject. The chunk solver routes such a scenario
                // to the CPU; the batch-of-one does the same - the CPU Newton-Raphson reruns it and
                // OLF's loops (deferred by the engine) decide.
                throw new PowsyblException("GPU driver rejected the base solve (status -2: a failed linear solve (the structure does not factor unpivoted), straggler exit, outer-loop budget, unrealistic state or a device-loop reject - OLF_GPU_REJECT_TRACE=1 names the gate)");
            }
            LOGGER.debug("GPU Newton-Raphson timing: prepare {} ms (context {}), solve {} ms",
                    (t1 - t0) / 1_000_000, rebuilt ? "rebuilt" : "reused", (t2 - t1) / 1_000_000);
            // PAUSE/RESUME CYCLE: if the driver paused at a LOOP_HOST_* slot, run the corresponding
            // OLF outer loop on the host and resume. Repeat until the driver completes without pausing.
            if (!hostOuterLoops.isEmpty()) {
                out = handleDriverPauses(out, n, target, elementMask, lsMode, convEpsPerEq);
                if (out == null) {
                    restartIterations = runNewtonIterations(0);
                    return runImpl(voltageInitializer, reportNode, true);   // the host loop changed the system
                }
            }
            iterations = runNewtonIterations((int) out[n]);
            // The driver ran OLF's outer loops, which MUTATE TARGETS on the device. OLF's world
            // still holds the pre-loop ones, so without this the converged state is measured
            // against stale targets and the run is declared MAX_ITERATION_REACHED — measured as
            // 170 BUS_TARGET_P rows and 14 BUS_TARGET_V rows over 1e-4, where the same solve with
            // the driver off leaves none. Give the distribution back to the network, exactly as
            // OLF's own DistributedSlack does by setting generator targetP.
            writeBackDriverMutations(cachedData);
            // write the device solution back; the equation vector re-evaluates from it
            equationSystem.getStateVector().set(Arrays.copyOf(out, n));
            equationVector.minus(targetVector);
            NewtonRaphsonStoppingCriteria.TestResult result =
                    parameters.getStoppingCriteria().test(equationVector.getArray(), equationSystem);
            // OLF_GPU_RESIDUAL_TRACE: the device solved to ITS residual, this is OLF's own residual at
            // the state it returned — a large value here means the device's equations disagree with
            // OLF's (a packing/row-mapping bug), not that the solver failed to converge. The
            // by-equation-type breakdown names the family to look at. (Test logback is broken —
            // stderr lever.)
            if (System.getenv("OLF_GPU_RESIDUAL_TRACE") != null) {
                double[] fx = equationVector.getArray();
                int nanX = 0;
                int nanF = 0;
                int firstNanF = -1;
                for (int i2 = 0; i2 < n; i2++) {
                    if (Double.isNaN(out[i2])) {
                        nanX++;
                    }
                    if (i2 < fx.length && Double.isNaN(fx[i2])) {
                        nanF++;
                        if (firstNanF < 0) {
                            firstNanF = i2;
                        }
                    }
                }
                if (nanX > 0 || nanF > 0) {
                    System.err.println("RESID_NAN deviceStateNaN=" + nanX + " olfResidualNaN=" + nanF + " firstNanRow=" + firstNanF
                            + (firstNanF >= 0 ? " eq=" + equationSystem.getIndex().getEquationAtColumn(firstNanF).getType()
                            + " elem=" + equationSystem.getIndex().getEquationAtColumn(firstNanF).getElementNum() : "")
                            + " deviceIters=" + iterations + " devNorm=" + out[n + 1]);
                }
                double n2 = 0;
                int worstRow = -1;
                double worst = 0;
                for (int i2 = 0; i2 < fx.length; i2++) {
                    n2 += fx[i2] * fx[i2];
                    if (Math.abs(fx[i2]) > worst) {
                        worst = Math.abs(fx[i2]);
                        worstRow = i2;
                    }
                }
                System.err.println("RESID_TRACE rows=" + fx.length + " cpuNorm2=" + Math.sqrt(n2)
                        + " cpuInfNorm=" + worst + " worstRow=" + worstRow
                        + " deviceIters=" + iterations + " stop=" + result.isStop());
                var worstEq = equationSystem.getIndex().getEquationAtColumn(worstRow);
                System.err.println("RESID_TRACE worst equation type=" + worstEq.getType()
                        + " elementNum=" + worstEq.getElementNum());
                java.util.Map<String, Integer> byType = new java.util.TreeMap<>();
                java.util.Map<String, Double> maxByType = new java.util.TreeMap<>();
                for (int i2 = 0; i2 < fx.length; i2++) {
                    if (Math.abs(fx[i2]) > 1e-4) {
                        String t = equationSystem.getIndex().getEquationAtColumn(i2).getType().toString();
                        byType.merge(t, 1, Integer::sum);
                        maxByType.merge(t, Math.abs(fx[i2]), Math::max);
                    }
                }
                System.err.println("RESID_TRACE rows over 1e-4 by type=" + byType + " max=" + maxByType);
                double[] devT = GpuAcNewtonSolver.readBatchTargets(context.handle.get());
                double[] olfT = targetVector.getArray();
                for (int i2 = 0; i2 < fx.length; i2++) {
                    if (Math.abs(fx[i2]) <= 1e-4) {
                        continue;
                    }
                    var eq = equationSystem.getIndex().getEquationAtColumn(i2);
                    StringBuilder sb = new StringBuilder("RESID_ROW row=" + i2 + " type=" + eq.getType()
                            + " f=" + fx[i2] + " olfTarget=" + olfT[i2]
                            + " devTarget=" + (devT != null && i2 < devT.length ? devT[i2] : Double.NaN)
                            + " devMode=" + activation.rowMode()[i2] + " x=" + out[i2]);
                    LfBus bus = eq.getType().toString().startsWith("BUS_") ? network.getBus(eq.getElementNum()) : null;
                    if (bus != null) {
                        sb.append(" bus=").append(bus.getId()).append(" vcEnabled=").append(bus.isGeneratorVoltageControlEnabled())
                          .append(" qLimitType=").append(bus.getQLimitType().map(Object::toString).orElse("-"))
                          .append(" genTargetQ=").append(bus.getGenerationTargetQ()).append(" loadTargetQ=").append(bus.getLoadTargetQ())
                          .append(" minQ=").append(bus.getMinQ()).append(" maxQ=").append(bus.getMaxQ())
                          .append(" hasVc=").append(bus.getGeneratorVoltageControl().isPresent())
                          .append(" isVcController=").append(bus.isGeneratorVoltageControlled());
                        if (cachedData.rlBusNum() != null && rlStateOut != null) {
                            for (int k = 0; k < cachedData.rlBusNum().length; k++) {
                                if (cachedData.rlBusNum()[k] == bus.getNum()) {
                                    sb.append(" rl[").append(k).append("]: state=").append(rlStateOut[k])
                                      .append(" vRow=").append(cachedData.rlVrow()[k]).append(" remote=").append(cachedData.rlRemote()[k])
                                      .append(" rlMinQ=").append(cachedData.rlMinQ()[k]).append(" rlMaxQ=").append(cachedData.rlMaxQ()[k])
                                      .append(" rlLoadQ=").append(cachedData.rlLoadQ()[k])
                                      .append(" devTargetAtVrow=").append(devT != null ? devT[cachedData.rlVrow()[k]] : Double.NaN)
                                      .append(" devModeAtVrow=").append(activation.rowMode()[cachedData.rlVrow()[k]]);
                                }
                            }
                        }
                    }
                    System.err.println(sb);
                }
            }
            // OLF's NewtonRaphson.run reports MAX_ITERATION_REACHED whenever iterations >= maxIterations, a
            // test passing after the last allowed step included; the device applies the same rule and its
            // verdict is kept. OLF's own criterion on the returned state decides the rest.
            boolean deviceBudgetOut = out.length > n + 2 && (int) out[n + 2] == GpuBatchStatus.STATUS_MAX_ITERATION;
            status = result.isStop() && !deviceBudgetOut ? AcSolverStatus.CONVERGED : AcSolverStatus.MAX_ITERATION_REACHED;
            lastRunOnDevice = true;                          // the DRIVER decided the loops of this solve

            // DEVICE-BROKEN result safety net: a non-finite or absurd residual is a device-side
            // numerical failure (e.g. the open zi-with-loaded-far-bus-behind-a-remote-member shape,
            // where the sequential solve returns phi ~1e22 while the SAME system solves on the CPU
            // and in the batch context — see GpuSeqBaseFallbackTest). Returning it would poison
            // OLF's outer loops and fail the whole run (an SA base would come back FAILED where the
            // stock CPU provider succeeds); throw into the CPU-fallback path instead, so the solve
            // reruns on the CPU Newton-Raphson: never wrong, only un-accelerated. Ordinary
            // non-convergence (a plausible norm) is NOT retried — the CPU would reproduce it.
            if (status == AcSolverStatus.MAX_ITERATION_REACHED
                    && (!Double.isFinite(result.getNorm()) || result.getNorm() > 1e2)) {
                throw new PowsyblException("GPU Newton-Raphson returned a device-broken state (|f(x)|="
                        + result.getNorm() + ")");
            }
            LOGGER.debug("GPU Newton-Raphson: {} iterations, |f(x)|={}, {}", iterations, result.getNorm(), status);
            if (System.getenv("OLF_GPU_SEQ_TRACE") != null) {   // test logback is broken — stderr trace
                System.err.println("SEQ_TRACE solve iters=" + iterations + " norm=" + result.getNorm()
                        + " devNorm=" + out[n + 1] + " status=" + status);
                double[] fv = equationVector.getArray();
                int[] tm = activation.targetMap();
                var vars = equationSystem.getIndex().getSortedVariablesToFind();
                for (int c = 0; c < fv.length; c++) {
                    if (Math.abs(fv[c]) > 1e-4) {
                        int row = -1;
                        for (int r = 0; r < tm.length; r++) {
                            if (tm[r] == c) {
                                row = r;
                            }
                        }
                        final int rr = row;
                        var vv = vars.stream().filter(x -> x.getRow() == rr).findFirst().orElse(null);
                        System.err.println("SEQ_TRACE   |f|=" + Math.abs(fv[c]) + " col=" + c + " row=" + row
                                + " mode=" + (row >= 0 ? activation.rowMode()[row] : -9)
                                + " var=" + (vv != null ? vv.getType() + "#" + vv.getElementNum() : "?"));
                    }
                }
            }
        } catch (RuntimeException e) {
            CPU_FALLBACK_COUNT.incrementAndGet();
            lastRunOnDevice = false;
            driverFallbackForRun = true;
            // the native context may be mid-iteration inconsistent — rebuild next run
            GpuAcNewtonSolver.destroyBatchContext(context.handle.getAndSet(0));
            cachedData = null;
            invalidation.structureDirty = true;
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                System.err.println("GPU_BASE_FALLBACK " + e);   // test logback is broken — stderr lever
                // "equation row N has no diagonal pairing" names a row INDEX, which on its own says
                // nothing about which equation could not be paired. Resolve it against the CURRENT
                // active index -- and read it as a hint, not an identification: activation changes
                // between solves, so the same row index is a different equation on a later fallback.
                java.util.regex.Matcher m = java.util.regex.Pattern
                        .compile("equation row (\\d+) has no diagonal pairing").matcher(String.valueOf(e.getMessage()));
                if (m.find()) {
                    int row = Integer.parseInt(m.group(1));
                    var eq = equationSystem.getIndex().getEquationAtColumn(row);
                    var elem = eq == null ? null
                            : network.getElement(eq.getType().getElementType(), eq.getElementNum());
                    System.err.printf("GPU_BASE_FALLBACK   row=%d of n=%d eq=%s elem=%d id=%s"
                            + " (activation-dependent: verify before trusting)%n", row, n,
                            eq == null ? "?" : eq.getType(), eq == null ? -1 : eq.getElementNum(),
                            elem == null ? "?" : elem.getId());
                }
                e.printStackTrace();
            }
            // Graceful degradation: anything the GPU path cannot handle — an extractor validation rejection
            // (unsupported variable/equation/term, remote control, zero-impedance shapes) or a native/device
            // error — is re-run on the standard CPU Newton-Raphson with the SAME equation system and parameters:
            // never wrong, only un-accelerated.
            LOGGER.warn("GPU Newton-Raphson unavailable for this run ({}); falling back to the CPU Newton-Raphson",
                    e.toString());
            return cpuNewtonRaphson(voltageInitializer, reportNode);
        }

        if (status == AcSolverStatus.CONVERGED || parameters.isAlwaysUpdateNetwork()) {
            AcSolverUtil.updateNetwork(network, equationSystem);
        }

        Map<Integer, Double> slackBusActivePowerMismatch = new TreeMap<>();
        for (LfSynchronousNetwork lfScNetwork : network.getSynchronousNetworks()) {
            slackBusActivePowerMismatch.put(lfScNetwork.getNumSC(), lfScNetwork.getSlackBuses().stream().mapToDouble(LfBus::getMismatchP).sum());
        }
        return new AcSolverResult(status, iterations, slackBusActivePowerMismatch);
    }

    private AcSolverResult cpuNewtonRaphson(VoltageInitializer voltageInitializer, ReportNode reportNode) {
        return new NewtonRaphson(network, parameters, equationSystem, j, targetVector, equationVector, detailedReport)
                .run(voltageInitializer, reportNode);
    }

    /** Build a map from HOST_LOOP_* kind to the OLF outer loop instance. */
    private java.util.Map<Integer, com.powsybl.openloadflow.ac.outerloop.AcOuterLoop> buildHostLoopMap() {
        var map = new java.util.HashMap<Integer, com.powsybl.openloadflow.ac.outerloop.AcOuterLoop>();
        for (var loop : hostOuterLoops) {
            int kind = DeviceLoopKind.loopKindFor(loop.getName());
            if (kind >= DeviceLoopKind.HOST_LOOP_SVC) {
                map.put(kind, loop);
            }
        }
        return map;
    }

    /**
     * Handle the driver pause/resume cycle: while the native driver is paused at a LOOP_HOST_*
     * slot, load the device state into the network, run the corresponding OLF outer loop, write
     * any target changes back to the device, and resume. Returns the final device output (the
     * converged state when no more loops fire).
     */
    @SuppressWarnings("java:S3776")
    private double[] handleDriverPauses(double[] out, int n, double[] target,
                                         boolean[][] elementMask, int lsMode, double convEpsPerEq) {
        long handle = context.handle.get();
        int pausedKind = GpuAcNewtonSolver.readBatchDriverPaused(handle);
        if (pausedKind == 0) {
            return out;   // driver completed without pausing
        }
        var loopMap = buildHostLoopMap();
        // Per-loop persistent context data (iteration count, per-loop state). Outer loops store
        // their own state via context.setData(); we keep one context per loop for the whole run —
        // across pauses AND across restarts. Use the engine's own AcLoadFlowContext if available
        // (set by GpuDriverOuterLoopEngine before the solve), so the outer loops share the solver's
        // equation system and Jacobian.
        if (hostLoopPairs == null) {
            hostLoopPairs = new java.util.HashMap<>();
            var engineContext = GPU_ENGINE_CONTEXT.get();
            if (engineContext != null) {
                var pairs = com.powsybl.openloadflow.ac.AcloadFlowEngine.createOuterLoopsAndContexts(engineContext, false);
                for (var pair : pairs) {
                    int kind = DeviceLoopKind.loopKindFor(pair.getLeft().getName());
                    if (kind >= DeviceLoopKind.HOST_LOOP_SVC && loopMap.containsKey(kind)
                            && (activeHostMask & DeviceLoopKind.hostMaskBit(kind)) != 0) {
                        hostLoopPairs.put(kind, pair);
                        pair.getLeft().initialize(pair.getRight());
                    }
                }
            }
        }
        var ctxMap = hostLoopPairs;
        while (pausedKind != 0) {
            var pair = ctxMap.get(pausedKind);
            if (pair == null) {
                LOGGER.warn("Driver paused at host loop kind {} with no registered handler — resuming as STABLE", pausedKind);
                GpuAcNewtonSolver.setBatchDriverResume(handle, new byte[]{0});
            } else {
                var loop = pair.getLeft();
                var loopCtx = pair.getRight();
                // Load device state into the equation system and network — INCLUDING the driver's own
                // decisions so far (distributed P, RL pins, HVDC statuses, monitoring activations):
                // OLF's loop reads them from the network (SecondaryVoltageControl skips the RL-pinned
                // controllers of a zone; a check that still saw them enabled went UNSTABLE where OLF's
                // said STABLE, one extra adjust round, 2.5e-8 kV).
                writeBackDriverMutations(cachedData);
                equationSystem.getStateVector().set(Arrays.copyOf(out, n));
                AcSolverUtil.updateNetwork(network, equationSystem);
                // Run OLF's own outer loop check
                var result = loop.check(loopCtx, com.powsybl.commons.report.ReportNode.NO_OP);
                if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                    System.err.printf("HOST_LOOP kind=%d name=%s iter=%d status=%s%n",
                            pausedKind, loop.getName(), loopCtx.getIteration(), result.status());
                }
                if (result.status() == com.powsybl.openloadflow.lf.outerloop.OuterLoopStatus.UNSTABLE) {
                    // What did the loop change? A tap/section move changes element VALUES (pi rows,
                    // shunt b); enabling or disabling a control changes the equation LAYOUT and the
                    // row modes. None of that can be patched into a paused sequence (the resume
                    // channel carries targets only) — OLF re-solves the mutated system and restarts
                    // its sweep, and so does the driver: cancel the pause, carry the RL state, and
                    // let runImpl re-extract and start over. Target-only changes resume in place.
                    GpuAcData probe = GpuAcDataCache.extract(network, equationSystem,
                            driverDsLoop() != null, driverRlLoop() != null, driverDsParams());
                    var act2 = GpuAcActivationExtractor.extractActivation(network, equationSystem, n);
                    boolean modesChanged = !Arrays.equals(act2.rowMode(), activation.rowMode());
                    if (!probe.sameAs(cachedData) || modesChanged) {
                        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                            StringBuilder diff = new StringBuilder();
                            for (var rc : GpuAcData.class.getRecordComponents()) {   // which arrays changed?
                                try {
                                    Object x1 = rc.getAccessor().invoke(probe);
                                    Object x2 = rc.getAccessor().invoke(cachedData);
                                    boolean eq = x1 instanceof double[] d1 ? Arrays.equals(d1, (double[]) x2)
                                            : x1 instanceof int[] i1 ? Arrays.equals(i1, (int[]) x2)
                                            : x1 instanceof byte[] b1 ? Arrays.equals(b1, (byte[]) x2)
                                            : x1 instanceof Object[] o1 ? Arrays.equals(o1, (Object[]) x2)
                                            : java.util.Objects.equals(x1, x2);
                                    if (!eq) {
                                        diff.append(' ').append(rc.getName());
                                    }
                                } catch (ReflectiveOperationException e) {
                                    diff.append(" ?").append(rc.getName());
                                }
                            }
                            System.err.println("HOST_LOOP_RESTART loop=" + loop.getName() + " sameStructure="
                                    + probe.sameStructureAs(cachedData) + " modesChanged=" + modesChanged + " changed:" + diff);
                        }
                        loopCtx.setIteration(loopCtx.getIteration() + 1);
                        // The driver's own decisions so far (distributed P, RL pins, HVDC statuses,
                        // monitoring activations, taps) live on the device; the restart re-extracts
                        // from the NETWORK, so hand them back first or the restarted solve starts
                        // from the original targets and undoes the distribution (4.3e-3 kV on the
                        // AFTER_GENERATOR transformer-VC base).
                        writeBackDriverMutations(cachedData);
                        rlCarry = new java.util.HashMap<>();
                        if (rlStateOut != null && cachedData.rlBusNum() != null) {
                            for (int i = 0; i < rlStateOut.length && i < cachedData.rlBusNum().length; i++) {
                                rlCarry.put(cachedData.rlBusNum()[i], rlStateOut[i]);
                            }
                        }
                        GpuAcNewtonSolver.cancelBatchDriverPause(handle);
                        restartLoopKind = pausedKind;
                        invalidation.structureDirty = true;
                        invalidation.activationDirty = true;
                        return null;
                    }
                    // The loop mutated the network. Re-extract targets (the target vector evaluates
                    // against the updated network state, so getArray() reflects the loop's changes).
                    double[] newTargets = activation.applyTargets(targetVector.getArray());
                    // Write per-row target deltas to the device (only rows that changed)
                    List<Integer> deltaRows = new java.util.ArrayList<>();
                    List<Double> deltaVals = new java.util.ArrayList<>();
                    for (int r = 0; r < n; r++) {
                        if (newTargets[r] != target[r]) {
                            deltaRows.add(r);          // flat (scenario=0 * n + row) for batch-of-1
                            deltaVals.add(newTargets[r]);
                        }
                    }
                    if (!deltaRows.isEmpty()) {
                        GpuAcNewtonSolver.setBatchTargetDeltas(handle,
                                deltaRows.stream().mapToInt(Integer::intValue).toArray(),
                                deltaVals.stream().mapToDouble(Double::doubleValue).toArray());
                    }
                    // Update our local target copy for the next round's diff
                    System.arraycopy(newTargets, 0, target, 0, n);
                    GpuAcNewtonSolver.setBatchDriverResume(handle, new byte[]{1});
                    loopCtx.setIteration(loopCtx.getIteration() + 1);
                } else {
                    GpuAcNewtonSolver.setBatchDriverResume(handle, new byte[]{0});
                }
            }
            // Resume the driver — solveBatchContext continues from its paused position
            out = GpuAcNewtonSolver.solveBatchContext(context.handle.get(),
                    equationSystem.getStateVector().get().clone(), target,
                    new int[][] {activation.rowMode()}, activation.rowMode(),
                    elementMask,
                    new int[0], new double[0],
                    new int[0], new double[0], new double[0], new double[0], new double[0], null, null,
                    null, null, null, null, null, null,
                    parameters.getMaxIterations(), GPU_TOLERANCE,
                    driverReactiveLimitTol, driverSlackDistTol, convEpsPerEq,
                    lsMode,
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDv(),
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDphi(),
                    varType, null,
                    null, null, null, rlStateOut, null, null,
                    driverMaxOuterIterations, driverMaxPqPvSwitch, driverDsRejectResidual ? 1 : 0, 0,
                    0, null, null)[0];
            pausedKind = GpuAcNewtonSolver.readBatchDriverPaused(handle);
        }
        return out;
    }

    /**
     * Resume a paused driver. The engine's pause/resume cycle already called
     * {@link GpuAcNewtonSolver#setBatchDriverResume} and (if the host loop was UNSTABLE)
     * {@link GpuAcNewtonSolver#setBatchTargetDeltas} — this just re-enters
     * {@code solveBatchContext} which the native side resumes from its paused position.
     * The activation, context, and element mask from the first call are still valid (the
     * host loop mutated the NETWORK but the device-side structures are unchanged — only
     * targets are patched via the target-delta channel).
     */
    private AcSolverResult runResume(ReportNode reportNode) {
        int n = equationSystem.getIndex().getSortedVariablesToFind().size();
        // Re-use the activation and element mask from the paused call. The native side
        // ignores x0/target/rowMode/overrides on a resume (driverResume=1 skips all uploads),
        // but the JNI wrapper needs non-null arrays at the right sizes.
        double[] target = activation.applyTargets(targetVector.getArray());
        double[] x0 = equationSystem.getStateVector().get().clone();
        boolean[][] elementMask = new boolean[1][cachedData.getTotalElementCount()];
        double convEpsPerEq = GpuSolveSettings.deviceConvEpsPerEq(parameters);
        int lsMode = switch (parameters.getStateVectorScalingMode()) {
            case NONE -> 0;
            case LINE_SEARCH -> 1;
            case MAX_VOLTAGE_CHANGE -> 2;
        };
        double[] out = GpuAcNewtonSolver.solveBatchContext(context.handle.get(),
                x0, target, new int[][] {activation.rowMode()}, activation.rowMode(),
                elementMask,
                new int[0], new double[0],
                new int[0], new double[0], new double[0], new double[0], new double[0], null, null,
                null, null, null, null, null, null,
                parameters.getMaxIterations(), GPU_TOLERANCE,
                driverReactiveLimitTol, driverSlackDistTol, convEpsPerEq,
                lsMode,
                parameters.getMaxVoltageChangeStateVectorScalingMaxDv(),
                parameters.getMaxVoltageChangeStateVectorScalingMaxDphi(),
                varType, null,
                null, null, null, rlStateOut, null, null,
                driverMaxOuterIterations, driverMaxPqPvSwitch, driverDsRejectResidual ? 1 : 0,
                0,
                0, null, null)[0];
        int iterations = runNewtonIterations((int) out[n]);
        // Write the (possibly intermediate) state back to the equation system so the engine
        // can run updateNetwork + the host loop's check() against it.
        equationSystem.getStateVector().set(Arrays.copyOf(out, n));
        equationVector.minus(targetVector);
        NewtonRaphsonStoppingCriteria.TestResult result =
                parameters.getStoppingCriteria().test(equationVector.getArray(), equationSystem);
        if (GpuAcNewtonSolver.readBatchDriverPaused(context.handle.get()) == 0) {
            writeBackDriverMutations(cachedData);
            equationVector.minus(targetVector);               // re-measure against the written-back targets
            result = parameters.getStoppingCriteria().test(equationVector.getArray(), equationSystem);
        }
        boolean deviceBudgetOut = out.length > n + 2 && (int) out[n + 2] == GpuBatchStatus.STATUS_MAX_ITERATION;
        AcSolverStatus status = result.isStop() && !deviceBudgetOut ? AcSolverStatus.CONVERGED : AcSolverStatus.MAX_ITERATION_REACHED;
        LOGGER.debug("GPU Newton-Raphson resume: {} iterations, |f(x)|={}, {}", iterations, result.getNorm(), status);
        Map<Integer, Double> slackBusActivePowerMismatch = new TreeMap<>();
        for (LfSynchronousNetwork lfScNetwork : network.getSynchronousNetworks()) {
            slackBusActivePowerMismatch.put(lfScNetwork.getNumSC(), lfScNetwork.getSlackBuses().stream().mapToDouble(LfBus::getMismatchP).sum());
        }
        return new AcSolverResult(status, iterations, slackBusActivePowerMismatch);
    }
}
