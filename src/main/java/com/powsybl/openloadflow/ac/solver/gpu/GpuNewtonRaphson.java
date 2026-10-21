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
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcSolverStatus;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuLinearSolveMode;
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
import com.powsybl.openloadflow.util.gpu.GpuTrace;
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

    private final com.powsybl.openloadflow.ac.AcLoadFlowParameters acLoadFlowParameters;

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
                : GpuDistributedSlackOuterLoop.applyToNetwork(network, equationSystem, driverDistributedSlackParameters(),
                        loadAnchorTargetP, deviceTargets,
                        BatchReadbackNative.readBatchDistributedSlackGeneratorMove(handle), BatchReadbackNative.readBatchSlackRetainedShare(handle),
                        BatchReadbackNative.readBatchSlackAbsorbed(handle), data);
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
        String path = GpuTrace.EQ_DUMP.value();
        if (equationsDumped || path == null || path.isEmpty() || cachedData == null) {
            return;
        }
        boolean pinned = false;
        for (int m : rowMode) {
            pinned |= m >= 4;
        }
        if (!pinned && !"any".equals(GpuTrace.EQ_DUMP_WHEN.value())) {
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

    /**
     * The Newton iterations of the whole run, as OLF's engine counts them: the sum over every solve of the
     * outer-loop sequence. One device call runs many of those solves (the in-kernel passes), and the device
     * sums them per scenario. OLF writes a load flow's result back to the network only when this is
     * positive, so the count of the last pass alone lost results.
     */
    private int runNewtonIterations(int lastCallIterations) {
        int[] stats = BatchReadbackNative.readBatchOuterLoopStats(context.handle.get());
        return stats != null && stats.length >= 3 ? stats[2] : lastCallIterations;
    }

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
            NewtonSettingsNative.installRowCriteria(context.handle.get(), thr);
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
                            com.powsybl.openloadflow.ac.AcLoadFlowParameters acLoadFlowParameters) {
        super(network, equationSystem, j, targetVector, equationVector, detailedReport);
        this.parameters = Objects.requireNonNull(parameters);
        this.driverSlackDistTol = driverSlackDistTol;
        this.driverReactiveLimitTol = driverReactiveLimitTol;
        this.driverMaxOuterIterations = driverMaxOuterIterations;
        this.driverMaxPqPvSwitch = driverMaxPqPvSwitch;
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
        (parameters instanceof GpuNewtonRaphsonParameters gpu ? gpu.getGpuParameters() : new GpuLoadFlowParameters())
                .install();
        return runImpl(voltageInitializer, reportNode, false);
    }

    /**
     * One attempt of the device sequence; {@code restart = true} is the robust-mode retry, started from the
     * network's last state.
     */
    private AcSolverResult runImpl(VoltageInitializer voltageInitializer, ReportNode reportNode, boolean restart) {
        if (!GpuAcNewtonSolver.isAvailable()) {
            throw new PowsyblException("GPU Newton-Raphson requested but the native library is not available "
                    + "(build with native/build-gpu.sh and provide cuDSS — see GpuAcNewtonSolver)");
        }
        // A restart starts from the NETWORK's last state (PreviousValueVoltageInitializer).
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
            // its pre-pin static qPct. Watch the pinned SET and re-extract when it changes: the context is
            // rebuilt from the fresh extraction with OLF's renormalized keys. That is what used to make every
            // post-pin solve CPU-routed; it now solves on the device.
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
            // the run's own loops
            DriverNative.setBatchOuterLoops(handle, driverDistributedSlackLoop() != null, driverReactiveLimitsOuterLoop() != null);
            // SecondaryVoltageControl (OLF order 4) IN-KERNEL — the same phase, built the same way, as the batched
            // security analysis. The device owns every loop of the run or the solve goes to the CPU, the same rule
            // the batched security analysis applies to a whole run: a zone set the kernel refuses, a control loop
            // the installer refuses, a loop whose builder represented only PART of OLF's controllers.
            secondaryVoltageControlControllers = phases.secondaryVoltageControlControllers();
            boolean secondaryVoltageControlActive = secondaryVoltageControlControllers != null && !secondaryVoltageControlControllers.controllerBuses().isEmpty();
            if (!com.powsybl.openloadflow.ac.outerloop.gpu.GpuSecondaryVoltageControlOuterLoop.uploadToDevice(handle,
                    secondaryVoltageControlActive ? secondaryVoltageControlControllers : null) && secondaryVoltageControlActive) {
                throw new PowsyblException("GPU driver: the device cannot own SecondaryVoltageControl; the solve goes to the CPU");
            }
            List<String> refusedLoops = installControlPhases(handle, phases);
            if (!refusedLoops.isEmpty()) {
                throw new PowsyblException("GPU driver: the device cannot own " + refusedLoops + "; the solve goes to the CPU");
            }
            com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhaseOwnership.requireComplete();
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
                    null, null, null, reactiveLimitsFinalState, null, null,
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
                if (GpuTrace.BATCH_TRACE.isOn()) {
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
            if (GpuTrace.BATCH_TRACE.isOn()) {
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

    /** When the structure is dirty: re-extract through the shared cache and rebuild the device context from the
     *  new extraction; true when it was rebuilt. */
    private boolean rebuildStructureIfDirty(final int n, final java.util.BitSet pinnedNow) {
        if (!invalidation.structureDirty) {
            return false;
        }
        // validate + build, through the shared cache (the batched SA reuses this extraction)
        // The RUN's DistributedSlack settings — the device must not distribute with a default balance type.
        // Which loops run is the run's own list (setBatchOuterLoops), not the extraction.
        GpuAcData data = GpuAcDataCache.extract(network, equationSystem, driverDistributedSlackParameters());
        // Always a rebuild: comparing the new extraction with the old one to skip it missed arrays the tables are
        // installed from, and no solver re-extracted within a run (measured 2026-10-21 over the suite and the six
        // gate scopes).
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
        criteriaDirty = true;
        tablesDirty = true;
        pinnedAtExtraction = pinnedNow;           // the packs now carry this pin set's qPct
        invalidation.structureDirty = false;
        invalidation.activationDirty = true;
        return true;
    }

    /** The incremental and discrete transformer, shunt and phase loops (OLF orders 7 to 10) installed on the device
     *  through {@link DevicePhaseInstaller}; returns the loops of the run the device does not own. */
    private List<String> installControlPhases(long handle, com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases.Built phases) {
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
        return installed.refused();
    }

    private AcSolverResult cpuNewtonRaphson(VoltageInitializer voltageInitializer, ReportNode reportNode) {
        return new NewtonRaphson(network, parameters, equationSystem, j, targetVector, equationVector, detailedReport)
                .run(voltageInitializer, reportNode);
    }
}
