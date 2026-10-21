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
import com.powsybl.openloadflow.ac.equations.vector.gpu.StructureNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.DeviceFlowControllers;
import com.powsybl.openloadflow.ac.outerloop.gpu.DeviceLoopTables;
import com.powsybl.openloadflow.ac.outerloop.gpu.ForceTargetQInReactiveLimits;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuDistributedSlackOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuHvdcAcEmulationLimitsOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuIncrementalShuntVoltageControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuIncrementalTransformerVoltageControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuMonitoringVoltageOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuPhaseControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuReactiveLimitsOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.HvdcNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.PhaseControlNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveCapabilityCurves;
import com.powsybl.openloadflow.ac.outerloop.gpu.SecondaryVoltageControlNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.ShuntVoltageControlNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.TransformerVoltageControlNative;
import com.powsybl.openloadflow.ac.outerloop.gpu.VoltageMonitoringNative;
import com.powsybl.openloadflow.ac.solver.AbstractAcSolver;
import com.powsybl.openloadflow.ac.solver.AcSolverResult;
import com.powsybl.openloadflow.ac.solver.AcSolverStatus;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphson;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.PerEquationTypeStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcSolverStatus;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuLinearSolveMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuOuterLoopKind;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuStateVectorScalingMode;
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
    private final boolean driverRejectSlackDistributionFailure;

    /** The OLF outer loops of the run: the driver pauses for those it does not run in-kernel. */
    private final List<com.powsybl.openloadflow.ac.outerloop.AcOuterLoop> hostOuterLoops;
    private final com.powsybl.openloadflow.ac.AcLoadFlowParameters acLoadFlowParameters;

    /** Build the bitmask for {@link GpuAcNewtonSolver#setBatchHostLoop} from the configured outer loops. */
    private int buildHostLoopMask() {
        int mask = 0;
        for (var loop : hostOuterLoops) {
            int kind = GpuOuterLoopKind.hostKindOf(loop.getName());
            if (kind >= GpuOuterLoopKind.HOST_SECONDARY_VOLTAGE_CONTROL) {
                mask |= GpuOuterLoopKind.hostMaskBit(kind);
            }
        }
        return mask;
    }

    /** The driver's per-controller reactive-limit state after the last solve (a GpuReactiveLimitsState: limit
     *  type and PV -> PQ switch count), read back so the PV↔PQ decisions can be given to the network. The limit
     *  type takes three bits because robust mode adds MIN/MAX_REALISTIC_V to MIN_Q/MAX_Q. */
    private int[] reactiveLimitsFinalState;

    /**
     * Give the driver's outer-loop decisions back to the network — what OLF's own loops do as a side
     * effect of running: without it OLF's world still holds the pre-loop targets and controls, the
     * converged state is measured against them and the run is declared MAX_ITERATION_REACHED
     * (measured 170 BUS_TARGET_P rows, 14 BUS_TARGET_V rows and 2 BUS_TARGET_Q rows over 1e-4 on
     * rte6515 where the same solve with the driver off leaves none), and everything downstream —
     * the batched SA's base targets, the CPU-routed contingencies — starts from the wrong network.
     */
    private void applyDeviceDecisionsToNetwork(GpuAcData data) {
        long handle = context.handle.get();
        double[] deviceTargets = BatchReadbackNative.readBatchTargets(handle);
        if (hvdcAcEmulationControllers != null && !hvdcAcEmulationControllers.isEmpty()) {
            GpuHvdcAcEmulationLimitsOuterLoop.applyToNetwork(network, hvdcAcEmulationControllers, HvdcNative.readBatchHvdcStatus(handle),
                    data.getTotalElementCount());
        }
        if (monitoringVoltageControllers != null && !monitoringVoltageControllers.isEmpty()) {
            GpuMonitoringVoltageOuterLoop.applyToNetwork(network, monitoringVoltageControllers, VoltageMonitoringNative.readBatchMonitoringTargets(handle),
                    data.reactiveLimits().busNum());
        }
        if (secondaryVoltageControlControllers != null) {
            com.powsybl.openloadflow.ac.outerloop.gpu.GpuSecondaryVoltageControlOuterLoop.applyToNetwork(network, secondaryVoltageControlControllers,
                    SecondaryVoltageControlNative.readBatchSecondaryVoltageControlTargets(handle));
        }
        // Only a loop the run has moved anything: its tables are packed either way.
        lastDistributedActivePower = driverDistributedSlackLoop() == null ? Double.NaN
                : GpuDistributedSlackOuterLoop.applyToNetwork(network, equationSystem, handle, driverDistributedSlackParameters(),
                        loadAnchorTargetP, deviceTargets,
                        BatchReadbackNative.readBatchDistributedSlackGeneratorMove(handle), BatchReadbackNative.readBatchSlackRetainedShare(handle), data);
        if (driverReactiveLimitsOuterLoop() != null) {
            GpuReactiveLimitsOuterLoop.applyToNetwork(network, reactiveLimitsFinalState, data, deviceTargets,
                    data.reactiveLimits().voltageControlRole() != null ? BatchReadbackNative.readBatchRowModes(handle) : null);
        }
        if (incrementalTransformerVoltageControlOwned && incrementalTransformerVoltageControlControllers != null) {
            GpuIncrementalTransformerVoltageControlOuterLoop.applyToNetwork(network, incrementalTransformerVoltageControlControllers,
                                                               TransformerVoltageControlNative.readBatchIncrementalTransformerVoltageControlTaps(handle));
        }
        if (shuntOwned && shuntControllers != null) {
            GpuIncrementalShuntVoltageControlOuterLoop.applyToNetwork(network, shuntControllers,
                                                              ShuntVoltageControlNative.readBatchIncrementalShuntVoltageControlPositions(handle));
        }
        if (flowOwned && flowControllers != null) {
            DeviceFlowControllers.applyToNetwork(network, flowControllers, PhaseControlNative.readBatchFlowTaps(handle));
        }
        if (phaseControlOwned && phaseControlControllers != null) {
            GpuPhaseControlOuterLoop.applyToNetwork(network, phaseControlControllers, PhaseControlNative.readBatchContinuousPhaseTaps(handle));
        }
        if (transformerVoltageControlOwned && transformerVoltageControlControllers != null) {
            com.powsybl.openloadflow.ac.outerloop.gpu.GpuTransformerVoltageControlOuterLoop.applyToNetwork(network, transformerVoltageControlControllers,
                    TransformerVoltageControlNative.readBatchTransformerVoltageControlState(handle));
            if (generatorVoltageControlFreeze != null && generatorVoltageControlFreeze.hasCandidates()) {
                com.powsybl.openloadflow.ac.outerloop.gpu.GpuTransformerVoltageControlOuterLoop.applyFreezeToNetwork(network, generatorVoltageControlFreeze,
                        TransformerVoltageControlNative.readBatchGeneratorVoltageControlFreeze(handle));
            }
        }
        if (shuntVoltageControlOwned && shuntVoltageControlGroups != null) {
            com.powsybl.openloadflow.ac.outerloop.gpu.GpuShuntVoltageControlOuterLoop.applyToNetwork(network, shuntVoltageControlGroups,
                    ShuntVoltageControlNative.readBatchShuntVoltageControlState(handle), ShuntVoltageControlNative.readBatchShuntVoltageControlPositions(handle));
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
            if (rowMode[r] == GpuRowMode.BUS_TARGET_Q) {
                m4.append(' ').append(r);
            }
        }
        // The HOST's pinned rows, to line up against the device's: if they differ, the device moved a pin,
        // and the row it moved it to is the defect.
        System.err.println("EQ_DUMP wrote " + path + " and " + path + ".olf (n=" + n + " pinned=" + pinned
                + ") hostMode4Rows=[" + m4.toString().trim() + "]");
    }

    /** The remote gen-VC pin set the cached extraction's DISTR_Q weights were built from (see runImpl). */
    private java.util.BitSet pinnedAtExtraction;

    /** OLF_GPU_EQ_DUMP: written once per run (see dumpEquationsOnce). */
    private boolean equationsDumped;

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
    private List<GpuHvdcAcEmulationLimitsOuterLoop.Controller> hvdcAcEmulationControllers;
    private List<GpuMonitoringVoltageOuterLoop.Controller> monitoringVoltageControllers;
    private List<GpuIncrementalTransformerVoltageControlOuterLoop.Controller> incrementalTransformerVoltageControlControllers;
    private boolean incrementalTransformerVoltageControlOwned;
    private List<GpuIncrementalShuntVoltageControlOuterLoop.Controller> shuntControllers;
    private boolean shuntOwned;
    private List<DeviceFlowControllers.Controller> flowControllers;
    private boolean flowOwned;
    private List<GpuPhaseControlOuterLoop.Controller> phaseControlControllers;
    private boolean phaseControlOwned;
    private List<com.powsybl.openloadflow.ac.outerloop.gpu.GpuTransformerVoltageControlOuterLoop.TransformerController> transformerVoltageControlControllers;
    private boolean transformerVoltageControlOwned;
    private com.powsybl.openloadflow.ac.outerloop.gpu.GpuTransformerVoltageControlOuterLoop.FreezeTables generatorVoltageControlFreeze;
    /** SecondaryVoltageControl's in-kernel tables — the same phase the batched SA drives by default. */
    private com.powsybl.openloadflow.ac.outerloop.gpu.GpuSecondaryVoltageControlOuterLoop.Controllers secondaryVoltageControlControllers;
    private List<com.powsybl.openloadflow.ac.outerloop.gpu.GpuShuntVoltageControlOuterLoop.Group> shuntVoltageControlGroups;
    private boolean shuntVoltageControlOwned;
    /** The host-loop bits the CURRENT solve actually pauses for (in-kernel phases clear theirs). */
    private int activeHostMask;

    /** The host-paused loops' (loop, context) pairs of the CURRENT engine run: OLF keeps one context
     *  per loop for the whole run (iteration counters, per-loop data), so they must survive a restart
     *  of the device sequence — created on the first pause, cleaned up when the run completes. */
    private java.util.Map<Integer, org.apache.commons.lang3.tuple.Pair<
            com.powsybl.openloadflow.ac.outerloop.AcOuterLoop, com.powsybl.openloadflow.ac.AcOuterLoopContext>> hostLoopPairs;

    /** Per controller bus, the driver's RL state carried across a restart (OLF's switch counters
     *  persist for the whole outer-loop run), keyed by bus num since the entry order may change. */
    private java.util.Map<Integer, Integer> reactiveLimitsStateCarry;

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
        int[] stats = BatchReadbackNative.readBatchOuterLoopStats(context.handle.get());
        return restartIterations + (stats != null && stats.length >= 3 ? stats[2] : lastCallIterations);
    }

    /** On a restart, the host loop the sequence resumes at (the one that went UNSTABLE). */
    private int restartLoopKind = GpuOuterLoopKind.NOT_PAUSED;  // else the paused host kind to restart at

    private com.powsybl.openloadflow.ac.outerloop.AcOuterLoop driverLoop(String name) {
        if (acLoadFlowParameters == null) {
            return null;
        }
        for (var outerLoop : acLoadFlowParameters.getOuterLoops()) {
            if (name.equals(outerLoop.getName())) {
                return outerLoop;
            }
        }
        return null;
    }

    private com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop driverDistributedSlackLoop() {
        if (acLoadFlowParameters == null) {
            return null;
        }
        for (var outerLoop : acLoadFlowParameters.getOuterLoops()) {
            if (outerLoop instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop distributedSlackOuterLoop) {
                return distributedSlackOuterLoop;
            }
        }
        return null;
    }

    private com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop driverReactiveLimitsOuterLoop() {
        if (acLoadFlowParameters == null) {
            return null;
        }
        for (var outerLoop : acLoadFlowParameters.getOuterLoops()) {
            if (outerLoop instanceof com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop reactiveLimitsOuterLoop) {
                return reactiveLimitsOuterLoop;
            }
        }
        return null;
    }

    /** The run's own DistributedSlack settings (balance type, active limits) — never defaults. */
    private com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters driverDistributedSlackParameters() {
        var distributedSlackOuterLoop = driverDistributedSlackLoop();
        return distributedSlackOuterLoop == null
                ? com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters.DEFAULT
                : new com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters(
                        distributedSlackOuterLoop.getActivePowerDistribution().getBalanceType(), distributedSlackOuterLoop.getActivePowerDistribution().isUseActiveLimits(),
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
            NewtonSettingsNative.setBatchRowCriteria(context.handle.get(), null, null);
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
        this.driverRejectSlackDistributionFailure = acLoadFlowParameters != null
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
        reactiveLimitsStateCarry = null;
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
        boolean resuming = DriverNative.readBatchDriverPaused(context.handle.get()) != GpuOuterLoopKind.NOT_PAUSED;
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
            // distrQWeight differs, sameStructureAs compares it, so the structure is rebuilt with OLF's renormalized
            // keys. That is what used to make every post-pin solve CPU-routed; it now solves on the device.
            java.util.BitSet pinnedNow = GpuAcActivationExtractor.pinnedRemoteMembers(network, equationSystem);
            if (!pinnedNow.equals(pinnedAtExtraction)) {
                invalidation.structureDirty = true;
            }
            rebuilt = rebuildStructureIfDirty(n, pinnedNow);
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
            // the RL state only a run with the ReactiveLimits loop moves (every entry stays NONE otherwise)
            reactiveLimitsFinalState = driverReactiveLimitsOuterLoop() != null && cachedData.reactiveLimits().voltageRow().length > 0
                    ? new int[cachedData.reactiveLimits().voltageRow().length] : null;
            int[] reactiveLimitsSeedState = null;
            if (restart && reactiveLimitsStateCarry != null && reactiveLimitsFinalState != null) {
                reactiveLimitsSeedState = new int[reactiveLimitsFinalState.length];
                for (int i = 0; i < reactiveLimitsSeedState.length; i++) {
                    reactiveLimitsSeedState[i] = reactiveLimitsStateCarry.getOrDefault(cachedData.reactiveLimits().busNum()[i], 0);
                }
            }
            // Device-side state-vector scaling: NONE = undamped Newton; LINE_SEARCH backtracks on the
            // device (mirrors LineSearchStateVectorScaling); MAX_VOLTAGE_CHANGE clamps each step so the
            // largest |Δv|/|Δφ| stays within maxDv/maxDphi (mirrors MaxVoltageChangeStateVectorScaling).
            int stateVectorScalingMode = GpuStateVectorScalingMode.of(parameters.getStateVectorScalingMode());
            long t1 = System.nanoTime();
            double convEpsPerEq = GpuSolveSettings.deviceConvEpsPerEq(parameters);
            installTablesAndLineSearch(target, stateVectorScalingMode);
            // AC-emulation HVDC statuses: OLF's freezing/limits loops DEACTIVATE the hvdc coupling
            // terms for any non-LINEAR status (the injections ride the target vector as VSC station
            // targetP, re-extracted every solve) — mirror it with the hv element's disabled bit,
            // which fillHvdcBatch honors. Without this the fill emitted the LINEAR coupling on top
            // of the frozen/saturated injections (the former frozen-HVDC cpu-base-only condition).
            boolean[][] elementMask = new boolean[1][cachedData.getTotalElementCount()];
            List<LfHvdc> packedHvdcs = network.getHvdcs().stream()
                    .filter(h -> h.getBus1() != null && h.getBus2() != null && h.isAcEmulation())
                    .toList();                               // the packer's exact enumeration order
            int hvdcAcEmulationOffset = cachedData.getTotalElementCount() - packedHvdcs.size();
            for (int h = 0; h < packedHvdcs.size(); h++) {
                if (packedHvdcs.get(h).getAcEmulationControl().getAcEmulationStatus()
                        != LfHvdc.AcEmulationControl.AcEmulationStatus.LINEAR_MODE) {
                    elementMask[0][hvdcAcEmulationOffset + h] = true;
                }
            }
            int[] ovIdx = new int[0];
            double[] ovVal = new double[0];
            resetDriverPhases();
            long handle = context.handle.get();
            int nElements = cachedData.getTotalElementCount();
            // Every phase's tables, from the SAME builder the batched SA calls (#58). The monitoring
            // and transformer builders probe controls with an on/off round trip that fires network
            // events, so the whole call sits inside our dirty-flag guard.
            var phases = buildDevicePhases(target, n);
            // HVDC AC emulation (freeze at OLF order 4, limits at 5) as driver phases. The run starts from the
            // statuses the network holds: OLF's FreezingHvdcACEmulation.initialize has already run when the
            // engine built its loops, and froze exactly the emulations it freezes — those whose two bus angles
            // are known, at that flow (a fresh network's NaN angles freeze nothing), with the frozen injection
            // in the targets. A FROZEN status enters as the base status, unfrozen by the driver's own phase; a
            // LINEAR one stays linear, also on a host-loop restart, which OLF does not re-initialize either.
            installHvdcDriverLoops(nElements, elementMask, handle);
            // MonitoringVoltage (OLF order 6) as a driver phase.
            monitoringVoltageControllers = phases.monitoring();
            GpuMonitoringVoltageOuterLoop.installDriverLoop(handle, monitoringVoltageControllers);
            int hostMask = buildHostLoopMask();
            // SecondaryVoltageControl (OLF order 4) IN-KERNEL — the same phase, built the same way,
            // as the batched security analysis. It used to be the one loop the base load flow never
            // handed to the driver, so every network with a zone paused it here while the chunk path
            // ran it on device; a pause whose mutation dirties the structure is also what forces the
            // driver RESTART, so the divergence was not only a throughput one.
            secondaryVoltageControlControllers = phases.secondaryVoltageControlControllers();
            // the run's own loops, before the phase uploads
            DriverNative.setBatchOuterLoops(handle, driverDistributedSlackLoop() != null, driverReactiveLimitsOuterLoop() != null);
            if (com.powsybl.openloadflow.ac.outerloop.gpu.GpuSecondaryVoltageControlOuterLoop.uploadToDevice(handle, secondaryVoltageControlControllers)) {
                hostMask &= ~GpuOuterLoopKind.hostMaskBit(GpuOuterLoopKind.HOST_SECONDARY_VOLTAGE_CONTROL);
            } else {
                secondaryVoltageControlControllers = null;                          // refused: OLF's own loop runs at the pause
            }
            hostMask = installControlPhases(handle, phases, hostMask);
            // Host-owned outer-loop slots: the driver pauses at their OLF positions and the
            // pause/resume cycle after solveBatchContext runs OLF's loop there. 0 clears a cached mask.
            // All-or-nothing, the same rule the batched SA applies: a loop the builders represented only
            // PARTIALLY goes back to OLF's engine at its own slot, rather than the device driving a
            // subset of it while the pause bit says nobody has to.
            hostMask = com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhaseOwnership.enforceComplete(
                    handle, hostMask);
            activeHostMask = hostMask;
            DriverNative.setBatchHostLoop(handle, hostMask);
            if (restart && restartLoopKind != GpuOuterLoopKind.NOT_PAUSED) {        // re-check the loop that went UNSTABLE first
                DriverNative.setBatchDriverStartLoop(handle, restartLoopKind);
            }
            restartLoopKind = GpuOuterLoopKind.NOT_PAUSED;
            if (driverRejectSlackDistributionFailure && driverDistributedSlackLoop() != null && cachedData.distributedSlack().busPhiRow().length == 0
                    && (cachedData.distributedSlack().slackPhiRow() < 0 || Double.isInfinite(cachedData.distributedSlack().slackRoomUp()))) {
                // No participant row and no slack generator to bound the slack's own share: either nobody
                // participates (the pack dropped the distribution, the mismatch simply stays at the slack) or the
                // slack's share has no generator limits (a load-based slack). The device cannot tell where OLF's
                // water-fill leaves a remaining mismatch, so OLF's own loop decides this failure behaviour. A
                // slack-only distribution over the slack's generators is bounded by their rooms on the device.
                throw new PowsyblException("GPU driver: a distribution without participant rows cannot apply "
                        + "slackDistributionFailureBehavior " + acLoadFlowParameters.getSlackDistributionFailureBehavior()
                        + "; the solve goes to the CPU");
            }
            double[] out = BatchSolveNative.solveBatchContext(context.handle.get(),
                    x0, target, new int[][] {activation.rowMode()}, activation.rowMode(),
                    elementMask,
                    ovIdx, ovVal,
                    new int[0], new double[0], new double[0], new double[0], new double[0], null, null,
                    null, null, null, null, null, null,
                    parameters.getMaxIterations(), GPU_TOLERANCE,
                    driverReactiveLimitTol, driverSlackDistTol, convEpsPerEq,
                    stateVectorScalingMode,
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDv(),
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDphi(),
                    varType, null,
                    null, null, reactiveLimitsSeedState, reactiveLimitsFinalState, null, null,
                    driverMaxOuterIterations, driverMaxPqPvSwitch, driverRejectSlackDistributionFailure ? 1 : 0,
                    0,                                       // skipReactiveLimitsPhase: the driver owns RL
                    0, null, null)[0];
            long t2 = System.nanoTime();
            if (out.length > n + 2 && (int) out[n + 2] == GpuAcSolverStatus.LINEAR_SOLVE_FAILED && !robustModeForRun) {
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
                BatchSolveNative.destroyBatchContext(context.handle.getAndSet(0));
                cachedData = null;
                invalidation.structureDirty = true;
                return runImpl(voltageInitializer, reportNode, true);
            }
            if (out.length > n + 2 && (int) out[n + 2] <= GpuAcSolverStatus.REJECTED) {
                // The kernel rejected the scenario (status -2, or -3 when the robust retry failed too): a linear solve that did not solve (the
                // structure does not factor unpivoted), the straggler exit, the outer-loop budget, an
                // unrealistic state or a device-loop reject. The chunk solver routes such a scenario
                // to the CPU; the batch-of-one does the same - the CPU Newton-Raphson reruns it and
                // OLF's loops (deferred by the engine) decide.
                throw new PowsyblException("GPU driver rejected the base solve (status -2: a failed linear solve (the structure does not factor unpivoted), straggler exit, outer-loop budget, "
                        + "unrealistic state or a device-loop reject - OLF_GPU_REJECT_TRACE=1 names the gate)");
            }
            LOGGER.debug("GPU Newton-Raphson timing: prepare {} ms (context {}), solve {} ms",
                    (t1 - t0) / 1_000_000, rebuilt ? "rebuilt" : "reused", (t2 - t1) / 1_000_000);
            // PAUSE/RESUME CYCLE: if the driver paused at a LOOP_HOST_* slot, run the corresponding
            // OLF outer loop on the host and resume. Repeat until the driver completes without pausing.
            if (!hostOuterLoops.isEmpty()) {
                out = handleDriverPauses(out, n, target, elementMask, stateVectorScalingMode, convEpsPerEq);
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
            applyDeviceDecisionsToNetwork(cachedData);
            // write the device solution back; the equation vector re-evaluates from it
            equationSystem.getStateVector().set(Arrays.copyOf(out, n));
            equationVector.minus(targetVector);
            NewtonRaphsonStoppingCriteria.TestResult result =
                    parameters.getStoppingCriteria().test(equationVector.getArray(), equationSystem);
            // OLF's NewtonRaphson.run reports MAX_ITERATION_REACHED whenever iterations >= maxIterations, a
            // test passing after the last allowed step included; the device applies the same rule and its
            // verdict is kept. OLF's own criterion on the returned state decides the rest.
            boolean deviceBudgetOut = out.length > n + 2 && (int) out[n + 2] == GpuAcSolverStatus.MAX_ITERATION_REACHED;
            status = result.isStop() && !deviceBudgetOut ? AcSolverStatus.CONVERGED : AcSolverStatus.MAX_ITERATION_REACHED;
            lastRunOnDevice = true;                          // the DRIVER decided the loops of this solve

            // DEVICE-BROKEN result safety net: a non-finite or absurd residual is a device-side
            // numerical failure (e.g. the open zero-impedance-with-loaded-far-bus-behind-a-remote-member shape,
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
        } catch (RuntimeException e) {
            CPU_FALLBACK_COUNT.incrementAndGet();
            lastRunOnDevice = false;
            driverFallbackForRun = true;
            // the native context may be mid-iteration inconsistent — rebuild next run
            BatchSolveNative.destroyBatchContext(context.handle.getAndSet(0));
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
            slackBusActivePowerMismatch.put(lfScNetwork.getNumSC(),
                                            lfScNetwork.getSlackBuses().stream().mapToDouble(LfBus::getMismatchP).sum());
        }
        return new AcSolverResult(status, iterations, slackBusActivePowerMismatch);
    }

    /** Install the run-constant outer-loop tables when they are dirty, and OLF's line-search budget in line-search
     *  mode. */
    private void installTablesAndLineSearch(final double[] target, final int stateVectorScalingMode) {
        if (tablesDirty) {
            // The SAME run-constant outer-loop tables the batched SA installs on its chunk
            // contexts — a batch of one drives the same loops as a batch of five hundred.
            var distributedSlackOuterLoop = driverDistributedSlackLoop();
            ReactiveCapabilityCurves.Channel reactiveCapabilityCurves = driverReactiveLimitsOuterLoop() != null
                    ? ReactiveCapabilityCurves.build(network, equationSystem, cachedData, driverDistributedSlackParameters(), distributedSlackOuterLoop != null, target,
                            acLoadFlowParameters.getNetworkParameters().isExtrapolateReactiveLimits())
                    : null;
            // forceTargetQInReactiveLimits: the base load flow's slack moves P too, and OLF re-clamps.
            var lfNetworkParameters = acLoadFlowParameters.getNetworkParameters();
            ForceTargetQInReactiveLimits.Channel forcedTargetQ = driverReactiveLimitsOuterLoop() != null && distributedSlackOuterLoop != null
                    ? ForceTargetQInReactiveLimits.build(network, equationSystem, cachedData, driverDistributedSlackParameters(),
                            lfNetworkParameters.isForceTargetQInReactiveLimits() && lfNetworkParameters.isReactiveLimits(),
                            true, target, activation.rowMode(), lfNetworkParameters.isExtrapolateReactiveLimits())
                    : null;
            DeviceLoopTables.install(context.handle.get(), cachedData, driverReactiveLimitsOuterLoop() != null, reactiveCapabilityCurves, forcedTargetQ,
                    acLoadFlowParameters.isVoltageRemoteControlRobustMode(),
                    acLoadFlowParameters.getMinRealisticVoltage(), acLoadFlowParameters.getMaxRealisticVoltage(),
                    acLoadFlowParameters.getMinNominalVoltageRealisticVoltageCheck(),
                    DeviceLoopTables.realisticFixingLoop(acLoadFlowParameters));
            tablesDirty = false;
        }
        if (stateVectorScalingMode == GpuStateVectorScalingMode.LINE_SEARCH) {   // batch loop LINE_SEARCH (ported 2026-09-08)
            NewtonSettingsNative.setBatchLineSearch(context.handle.get(),
                    parameters.getLineSearchStateVectorScalingMaxIteration(),
                    parameters.getLineSearchStateVectorScalingStepFold());
        }
    }

    /** Forget the previous solve's per-loop controllers and ownership flags: every solve installs its phases
     *  afresh. */
    private void resetDriverPhases() {
        hvdcAcEmulationControllers = null;
        monitoringVoltageControllers = null;
        incrementalTransformerVoltageControlControllers = null;
        incrementalTransformerVoltageControlOwned = false;
        shuntControllers = null;
        shuntOwned = false;
        flowControllers = null;
        flowOwned = false;
        phaseControlControllers = null;
        phaseControlOwned = false;
        transformerVoltageControlControllers = null;
        transformerVoltageControlOwned = false;
        generatorVoltageControlFreeze = null;
        shuntVoltageControlGroups = null;
        shuntVoltageControlOwned = false;
    }

    /** Every driver phase's tables, from the same builder the batched SA calls. The monitoring and transformer
     *  builders probe controls with an on/off round trip that fires network events, so the call sits inside a
     *  guard that keeps the dirty flags as they were. */
    private com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases.Built buildDevicePhases(final double[] target,
            final int n) {
        var transformerVoltageControlLoop = driverLoop("TransformerVoltageControl");
        var incrementalTransformerVoltageControlLoop = driverLoop("IncrementalTransformerVoltageControl");
        var incrementalShuntVoltageControlLoop = driverLoop("IncrementalShuntVoltageControl");
        boolean sdAll = invalidation.structureDirty;
        boolean adAll = invalidation.activationDirty;
        var phases = com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases.build(
                new com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases.Request(
                        driverLoop("VoltageMonitoring") != null,
                        incrementalTransformerVoltageControlLoop != null,
                        incrementalShuntVoltageControlLoop != null,
                        driverLoop("ShuntVoltageControl") != null,
                        driverLoop("IncrementalTransformerReactivePowerControl") != null,
                        driverLoop("IncrementalPhaseControl") != null,
                        driverLoop("IncrementalTransformerReactivePowerControl")
                                instanceof com.powsybl.openloadflow.ac.outerloop.IncrementalTransformerReactivePowerControlOuterLoop r
                                ? r.getMaxTapShift() : 1,
                        driverLoop("PhaseControl") != null,
                        transformerVoltageControlLoop != null || driverLoop("SimpleTransformerVoltageControl") != null,
                        transformerVoltageControlLoop instanceof com.powsybl.openloadflow.ac.outerloop.TransformerVoltageControlOuterLoop tg
                                && tg.isUseInitialTapPosition(),
                        transformerVoltageControlLoop instanceof com.powsybl.openloadflow.ac.outerloop.TransformerVoltageControlOuterLoop tg2
                                ? tg2.getMaxControlledNominalVoltageOverride() : -1,
                        driverLoop("SecondaryVoltageControl") != null,
                        acLoadFlowParameters.getNetworkParameters().getMinPlausibleTargetVoltage(),
                        acLoadFlowParameters.getNetworkParameters().getMaxPlausibleTargetVoltage(),
                        driverReactiveLimitsOuterLoop() != null),
                network, equationSystem, targetVector, activation, target,
                cachedData.reactiveLimits().voltageRow(), cachedData.reactiveLimits().busNum(), n);
        invalidation.structureDirty = sdAll;
        invalidation.activationDirty = adAll;
        return phases;
    }

    /** HVDC AC emulation (freeze at OLF order 4, limits at 5) as driver phases, started from the statuses the
     *  network holds. */
    private void installHvdcDriverLoops(final int nElements, final boolean[][] elementMask, final long handle) {
        boolean hvdcLimits = driverLoop("AcHvdcAcEmulationLimits") != null;
        boolean hvdcFreeze = driverLoop("FreezingHvdcACEmulation") != null;
        if (hvdcLimits || hvdcFreeze) {
            hvdcAcEmulationControllers = GpuHvdcAcEmulationLimitsOuterLoop.buildControllers(network, equationSystem, nElements, cachedData);
        }
        if (hvdcAcEmulationControllers != null && !hvdcAcEmulationControllers.isEmpty()) {
            int nh = hvdcAcEmulationControllers.size();
            int[][] hvdcAcEmulationStatus = new int[1][nh];
            double[] appliedDelta1 = new double[nh];
            double[] appliedDelta2 = new double[nh];
            for (var controller : hvdcAcEmulationControllers) {
                elementMask[0][controller.hvdcElementIndex()] = false;   // set back below from the status, as a BASE status
            }
            GpuHvdcAcEmulationLimitsOuterLoop.seedBaseStatuses(1, 1, hvdcAcEmulationControllers, hvdcAcEmulationStatus, elementMask);
            GpuHvdcAcEmulationLimitsOuterLoop.installDriverLoops(handle, hvdcAcEmulationControllers, 1, hvdcAcEmulationStatus, elementMask,
                    appliedDelta1, appliedDelta2, hvdcLimits);
        } else {
            GpuHvdcAcEmulationLimitsOuterLoop.installDriverLoops(handle, null, 1, null, null, null, null, false);
        }
    }

    /** When the structure is dirty: re-extract through the shared cache and rebuild or refresh the device context;
     *  true when it was rebuilt. */
    private boolean rebuildStructureIfDirty(final int n, final java.util.BitSet pinnedNow) {
        boolean rebuilt = false;
        if (invalidation.structureDirty) {
            // validate + build, through the shared cache (the batched SA reuses this extraction)
            // The RUN's DistributedSlack settings — the device must not distribute with a default balance type.
            // Which loops run is the run's own list (setBatchOuterLoops), not the extraction.
            GpuAcData data = GpuAcDataCache.extract(network, equationSystem, driverDistributedSlackParameters());
            // Three-level reconciliation, decided by the DATA not the event: a changed CSR pattern
            // (index packs) is STRUCTURE -> rebuild + cuDSS re-analysis; an unchanged pattern with
            // only changed element PARAMETERS (tap r1/a1, shunt b) is VALUES -> a cheap in-place
            // pack re-upload reusing the analysis; an unchanged data is a no-op (context reused).
            if (!data.sameStructureAs(cachedData)) {
                BatchSolveNative.destroyBatchContext(context.handle.getAndSet(0));
                // BATCH-OF-1: the load flow runs on a BATCH context of one scenario, so per-row
                // PER_EQUATION_TYPE criteria, slope-SVC (mode 2), continuous phase (mode 3) and
                // the load model (ZIP) fills come from the batch pipeline instead of being ported twice.
                // The shared structure comes from the per-extraction cache — the batched SA
                // building its context from the SAME extraction reuses it device-side.
                context.handle.set(BatchSolveNative.createBatchContext(GpuAcStructures.handle(n, data), 1,
                        robustModeForRun ? GpuLinearSolveMode.ROBUST : GpuBatchContextCache.resolveBatchMode(),
                        data.reactiveLimits().voltageRow(), data.reactiveLimits().minQ(), data.reactiveLimits().maxQ(), data.reactiveLimits().loadQ(), data.reactiveLimits().targetV(),
                        data.reactiveLimits().flags(), data.voltageRows().slope(),
                        data.distributedSlack().busPhiRow(), data.distributedSlack().busFactor(), data.distributedSlack().slackPhiRow(), data.distributedSlack().slackTargetP(),
                        data.distributedSlack().busBaseTargetP(), data.distributedSlack().busMaxDelta(), data.distributedSlack().busMinDelta(),
                        data.distributedSlack().factorMode(), data.distributedSlack().busMarginUp(), data.distributedSlack().busMarginDown(),
                        data.distributedSlack().slackMarginUp(), data.distributedSlack().slackMarginDown()));
                // OLF's forced-PV election keys are part of creating a context with RL entries.
                DeviceLoopTables.installReactiveLimitStrongestPvKeys(context.handle.get(), data);
                cachedData = data;
                loadAnchorTargetP = GpuDistributedSlackOuterLoop.captureLoadAnchors(network);
                rebuilt = true;
                criteriaDirty = true;
                tablesDirty = true;
                CONTEXT_REBUILDS.incrementAndGet();
            } else if (!data.sameAs(cachedData)) {
                StructureNative.refreshStructureValues(GpuAcStructures.handle(n, cachedData),
                        data.closedBranch().parameters(), data.openBranch().parameters(), data.shunt().parameters(), data.hvdcAcEmulation().parameters(), data.zeroImpedanceBranch().parameters());
                // the SHARED structure now holds the new values — re-key it to the new
                // extraction so content matching never aliases it under the old one
                GpuAcStructures.rebind(cachedData, data);
                cachedData = data;
                loadAnchorTargetP = GpuDistributedSlackOuterLoop.captureLoadAnchors(network);
                // The election keys are per-extraction (nominal V, live target P, id rank), so a
                // value refresh re-installs them too.
                DeviceLoopTables.installReactiveLimitStrongestPvKeys(context.handle.get(), data);
                tablesDirty = true;
                VALUE_REFRESHES.incrementAndGet();
            }
            pinnedAtExtraction = pinnedNow;           // the packs now carry this pin set's qPct
            invalidation.structureDirty = false;
            invalidation.activationDirty = true;
        }
        return rebuilt;
    }

    /** The incremental and discrete transformer, shunt and phase loops (OLF orders 7 to 10) installed on the device
     *  through {@link DevicePhaseInstaller}; returns {@code hostMask} without the bits of the loops the device owns. */
    private int installControlPhases(long handle, com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases.Built phases,
                                     int hostMaskIn) {
        int hostMask = hostMaskIn;
        // The incremental and discrete transformer, shunt and phase loops (OLF orders 7 to 10) IN-KERNEL,
        // installed exactly as for a security analysis chunk. The controls OLF's initialize enabled on the
        // network are in the activation already (entryOn): AFTER_GENERATOR starts with every control off
        // and the driver's INITIAL/CONTROL/COMPLETE machine enables, clamps and rounds; WITH_GENERATOR
        // starts with the controls ON and the first check rounds or dispatches.
        var incrementalTransformerVoltageControlLoop = driverLoop("IncrementalTransformerVoltageControl")
                instanceof com.powsybl.openloadflow.ac.outerloop.IncrementalTransformerVoltageControlOuterLoop l ? l : null;
        var incrementalShuntVoltageControlLoop = driverLoop("IncrementalShuntVoltageControl")
                instanceof com.powsybl.openloadflow.ac.outerloop.IncrementalShuntVoltageControlOuterLoop l ? l : null;
        var incrementalTransformerReactivePowerControlLoop = driverLoop("IncrementalTransformerReactivePowerControl");
        boolean hasIncrementalPhaseControlLoop = driverLoop("IncrementalPhaseControl") != null;
        boolean hasPhaseControlLoop = driverLoop("PhaseControl") != null;
        var transformerVoltageControlLoop = driverLoop("TransformerVoltageControl");
        var simpleTransformerVoltageControlLoop = driverLoop("SimpleTransformerVoltageControl");
        boolean afterGen = transformerVoltageControlLoop instanceof com.powsybl.openloadflow.ac.outerloop.TransformerVoltageControlOuterLoop;
        boolean hasShuntVoltageControlLoop = driverLoop("ShuntVoltageControl") != null;
        var installed = com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhaseInstaller.install(handle,
                new com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhaseInstaller.Request(
                        incrementalTransformerVoltageControlLoop != null ? phases.incrementalTransformerVoltageControllers() : null,
                        incrementalTransformerVoltageControlLoop != null ? incrementalTransformerVoltageControlLoop.getMaxTapShift() : 1,

                        incrementalTransformerReactivePowerControlLoop != null || hasIncrementalPhaseControlLoop ? phases.flow() : null, hasPhaseControlLoop ? phases.continuousPhase() : null,
                        incrementalShuntVoltageControlLoop != null ? phases.incrementalShunt() : null,
                        incrementalShuntVoltageControlLoop != null ? incrementalShuntVoltageControlLoop.getMaxSectionShift() : 3,

                        transformerVoltageControlLoop != null || simpleTransformerVoltageControlLoop != null ? phases.discreteTransformerVoltageControl() : null,
                        simpleTransformerVoltageControlLoop != null,
                        afterGen, phases.freeze(), hasShuntVoltageControlLoop ? phases.simpleShunt() : null, true));
        if (incrementalTransformerVoltageControlLoop != null) {
            incrementalTransformerVoltageControlControllers = phases.incrementalTransformerVoltageControllers();
        }
        if (incrementalShuntVoltageControlLoop != null) {
            shuntControllers = phases.incrementalShunt();
        }
        if (incrementalTransformerReactivePowerControlLoop != null || hasIncrementalPhaseControlLoop) {
            flowControllers = phases.flow();
        }
        if (hasPhaseControlLoop) {
            phaseControlControllers = phases.continuousPhase();
        }
        if (transformerVoltageControlLoop != null || simpleTransformerVoltageControlLoop != null) {
            transformerVoltageControlControllers = phases.discreteTransformerVoltageControl();
        }
        if (hasShuntVoltageControlLoop) {
            shuntVoltageControlGroups = phases.simpleShunt();
        }
        incrementalTransformerVoltageControlOwned = installed.incrementalTransformerVoltageControllers();
        shuntOwned = installed.incrementalShunt();
        flowOwned = installed.flow();
        phaseControlOwned = installed.continuousPhase();
        transformerVoltageControlOwned = installed.discreteTransformerVoltageControl();
        shuntVoltageControlOwned = installed.simpleShunt();
        if (transformerVoltageControlOwned && afterGen) {
            // Batch of one on the live network: its own topology IS the scenario, so the
            // connectivity masks come from an empty contingency.
            generatorVoltageControlFreeze = phases.freeze();
            com.powsybl.openloadflow.ac.outerloop.gpu.GpuTransformerVoltageControlOuterLoop.uploadFixMasks(
                    handle, generatorVoltageControlFreeze, java.util.List.of(), 1);
        }
        if (incrementalTransformerVoltageControlOwned || transformerVoltageControlOwned) {
            hostMask &= ~(GpuOuterLoopKind.hostMaskBit(GpuOuterLoopKind.HOST_TRANSFORMER_VOLTAGE_CONTROL));
        }
        if (shuntOwned || shuntVoltageControlOwned) {
            hostMask &= ~(GpuOuterLoopKind.hostMaskBit(GpuOuterLoopKind.HOST_SHUNT_VOLTAGE_CONTROL));
        }
        if (flowOwned && incrementalTransformerReactivePowerControlLoop != null) {
            hostMask &= ~(GpuOuterLoopKind.hostMaskBit(GpuOuterLoopKind.HOST_TRANSFORMER_REACTIVE_POWER_CONTROL));
        }
        if (flowOwned && hasIncrementalPhaseControlLoop || phaseControlOwned) {
            hostMask &= ~(GpuOuterLoopKind.hostMaskBit(GpuOuterLoopKind.HOST_PHASE_CONTROL));
        }
        return hostMask;
    }

    private AcSolverResult cpuNewtonRaphson(VoltageInitializer voltageInitializer, ReportNode reportNode) {
        return new NewtonRaphson(network, parameters, equationSystem, j, targetVector, equationVector, detailedReport)
                .run(voltageInitializer, reportNode);
    }

    /** Build a map from GpuOuterLoopKind.HOST_* kind to the OLF outer loop instance. */
    private java.util.Map<Integer, com.powsybl.openloadflow.ac.outerloop.AcOuterLoop> buildHostLoopMap() {
        var map = new java.util.HashMap<Integer, com.powsybl.openloadflow.ac.outerloop.AcOuterLoop>();
        for (var loop : hostOuterLoops) {
            int kind = GpuOuterLoopKind.hostKindOf(loop.getName());
            if (kind >= GpuOuterLoopKind.HOST_SECONDARY_VOLTAGE_CONTROL) {
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
    private double[] handleDriverPauses(double[] outIn, int n, double[] target,
                                         boolean[][] elementMask, int stateVectorScalingMode, double convEpsPerEq) {
        double[] out = outIn;
        long handle = context.handle.get();
        int pausedKind = DriverNative.readBatchDriverPaused(handle);
        if (pausedKind == GpuOuterLoopKind.NOT_PAUSED) {
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
                var pairs = com.powsybl.openloadflow.ac.AcloadFlowEngine.createOuterLoopsAndContexts(engineContext,
                        false);
                for (var pair : pairs) {
                    int kind = GpuOuterLoopKind.hostKindOf(pair.getLeft().getName());
                    if (kind >= GpuOuterLoopKind.HOST_SECONDARY_VOLTAGE_CONTROL && loopMap.containsKey(kind)
                            && (activeHostMask & GpuOuterLoopKind.hostMaskBit(kind)) != 0) {
                        hostLoopPairs.put(kind, pair);
                        pair.getLeft().initialize(pair.getRight());
                    }
                }
            }
        }
        var ctxMap = hostLoopPairs;
        while (pausedKind != GpuOuterLoopKind.NOT_PAUSED) {
            var pair = ctxMap.get(pausedKind);
            if (pair == null) {
                LOGGER.warn("Driver paused at host loop kind {} with no registered handler — resuming as STABLE", pausedKind);
                DriverNative.setBatchDriverResume(handle, new byte[]{0});
            } else {
                var loop = pair.getLeft();
                var loopCtx = pair.getRight();
                // Load device state into the equation system and network — INCLUDING the driver's own
                // decisions so far (distributed P, RL pins, HVDC statuses, monitoring activations):
                // OLF's loop reads them from the network (SecondaryVoltageControl skips the RL-pinned
                // controllers of a zone; a check that still saw them enabled went UNSTABLE where OLF's
                // said STABLE, one extra adjust round, 2.5e-8 kV).
                applyDeviceDecisionsToNetwork(cachedData);
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
                    GpuAcData probe = GpuAcDataCache.extract(network, equationSystem, driverDistributedSlackParameters());
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
                        applyDeviceDecisionsToNetwork(cachedData);
                        reactiveLimitsStateCarry = new java.util.HashMap<>();
                        if (reactiveLimitsFinalState != null && cachedData.reactiveLimits().busNum() != null) {
                            for (int i = 0; i < reactiveLimitsFinalState.length && i < cachedData.reactiveLimits().busNum().length; i++) {
                                reactiveLimitsStateCarry.put(cachedData.reactiveLimits().busNum()[i], reactiveLimitsFinalState[i]);
                            }
                        }
                        DriverNative.cancelBatchDriverPause(handle);
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
                        NewtonSettingsNative.setBatchTargetDeltas(handle,
                                deltaRows.stream().mapToInt(Integer::intValue).toArray(),
                                deltaVals.stream().mapToDouble(Double::doubleValue).toArray());
                    }
                    // Update our local target copy for the next round's diff
                    System.arraycopy(newTargets, 0, target, 0, n);
                    DriverNative.setBatchDriverResume(handle, new byte[]{1});
                    loopCtx.setIteration(loopCtx.getIteration() + 1);
                } else {
                    DriverNative.setBatchDriverResume(handle, new byte[]{0});
                }
            }
            // Resume the driver — solveBatchContext continues from its paused position
            out = BatchSolveNative.solveBatchContext(context.handle.get(),
                    equationSystem.getStateVector().get().clone(), target,
                    new int[][] {activation.rowMode()}, activation.rowMode(),
                    elementMask,
                    new int[0], new double[0],
                    new int[0], new double[0], new double[0], new double[0], new double[0], null, null,
                    null, null, null, null, null, null,
                    parameters.getMaxIterations(), GPU_TOLERANCE,
                    driverReactiveLimitTol, driverSlackDistTol, convEpsPerEq,
                    stateVectorScalingMode,
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDv(),
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDphi(),
                    varType, null,
                    null, null, null, reactiveLimitsFinalState, null, null,
                    driverMaxOuterIterations, driverMaxPqPvSwitch, driverRejectSlackDistributionFailure ? 1 : 0, 0,
                    0, null, null)[0];
            pausedKind = DriverNative.readBatchDriverPaused(handle);
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
        int stateVectorScalingMode = GpuStateVectorScalingMode.of(parameters.getStateVectorScalingMode());
        double[] out = BatchSolveNative.solveBatchContext(context.handle.get(),
                x0, target, new int[][] {activation.rowMode()}, activation.rowMode(),
                elementMask,
                new int[0], new double[0],
                new int[0], new double[0], new double[0], new double[0], new double[0], null, null,
                null, null, null, null, null, null,
                parameters.getMaxIterations(), GPU_TOLERANCE,
                driverReactiveLimitTol, driverSlackDistTol, convEpsPerEq,
                stateVectorScalingMode,
                parameters.getMaxVoltageChangeStateVectorScalingMaxDv(),
                parameters.getMaxVoltageChangeStateVectorScalingMaxDphi(),
                varType, null,
                null, null, null, reactiveLimitsFinalState, null, null,
                driverMaxOuterIterations, driverMaxPqPvSwitch, driverRejectSlackDistributionFailure ? 1 : 0,
                0,
                0, null, null)[0];
        int iterations = runNewtonIterations((int) out[n]);
        // Write the (possibly intermediate) state back to the equation system so the engine
        // can run updateNetwork + the host loop's check() against it.
        equationSystem.getStateVector().set(Arrays.copyOf(out, n));
        equationVector.minus(targetVector);
        NewtonRaphsonStoppingCriteria.TestResult result =
                parameters.getStoppingCriteria().test(equationVector.getArray(), equationSystem);
        if (DriverNative.readBatchDriverPaused(context.handle.get()) == GpuOuterLoopKind.NOT_PAUSED) {
            applyDeviceDecisionsToNetwork(cachedData);
            equationVector.minus(targetVector);               // re-measure against the written-back targets
            result = parameters.getStoppingCriteria().test(equationVector.getArray(), equationSystem);
        }
        boolean deviceBudgetOut = out.length > n + 2 && (int) out[n + 2] == GpuAcSolverStatus.MAX_ITERATION_REACHED;
        AcSolverStatus status = result.isStop() && !deviceBudgetOut ? AcSolverStatus.CONVERGED : AcSolverStatus.MAX_ITERATION_REACHED;
        LOGGER.debug("GPU Newton-Raphson resume: {} iterations, |f(x)|={}, {}", iterations, result.getNorm(), status);
        Map<Integer, Double> slackBusActivePowerMismatch = new TreeMap<>();
        for (LfSynchronousNetwork lfScNetwork : network.getSynchronousNetworks()) {
            slackBusActivePowerMismatch.put(lfScNetwork.getNumSC(),
                                            lfScNetwork.getSlackBuses().stream().mapToDouble(LfBus::getMismatchP).sum());
        }
        return new AcSolverResult(status, iterations, slackBusActivePowerMismatch);
    }
}
