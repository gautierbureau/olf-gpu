/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.commons.config.PlatformConfig;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.SlackRelocation;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuStateVectorScalingMode;

/**
 * The run's outer-loop parameterization for the batched device path — a PER-CALL value, threaded
 * from {@link com.powsybl.openloadflow.sa.gpu.GpuBatchedAcSecurityAnalysis} into
 * {@link GpuBatchedSecurityAnalysisSolver#solveBatchN1} and down to the device call. It used to live
 * in JVM-global static fields set through setters before each run; production was safe only because
 * the provider always wrote every field first, but anything driving the solver DIRECTLY inherited
 * the previous run's values. That leak (a stale {@code dsRejectResidual} from an RTE run rejecting
 * every saturated scenario of the next direct run) is why GpuBatchedPegaseFullLoadflowTest passed
 * alone yet failed in a full-suite run. Making it a parameter removes the leak by construction; the
 * leak was one-directional (a stale {@code true} over-rejects — slow but correct; a stale
 * {@code false} would under-reject and leave a residual at the slack the run's failure behavior
 * forbids — a silent wrong answer).
 *
 * <ul>
 *   <li>{@code lsMode}/{@code maxDv}/{@code maxDphi}: MAX_VOLTAGE_CHANGE state-vector scaling per
 *       scenario (mirrors the CPU's MaxVoltageChangeStateVectorScaling); lsMode 0 = raw full Newton
 *       steps.</li>
 *   <li>{@code transformerControl}: run the per-scenario AFTER_GENERATOR transformer voltage-control
 *       orchestrator (set only when the resolved outer loops include it and the network is in scope).</li>
 *   <li>{@code maxOuterIterations}/{@code maxPqPvSwitch}: OLF's maxOuterLoopIterations and
 *       reactiveLimitsMaxPqPvSwitch replacing the device constants; 0 = device default (20 / 3).</li>
 *   <li>{@code dsRejectResidual}: a slackDistributionFailureBehavior other than LEAVE_ON_SLACK_BUS
 *       makes a saturated distribution REJECT the scenario (CPU fallback replays it with the real
 *       behavior) instead of silently leaving the residual at the slack.</li>
 *   <li>{@code forceTargetQ}: OLF's forceTargetQInReactiveLimits with reactive limits on - curve generators'
 *       target Q follows their P (ForceTargetQInReactiveLimits).</li>
 * </ul>
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record BatchConfig(int lsMode, double maxDv, double maxDphi, boolean transformerControl,
                          int maxOuterIterations, int maxPqPvSwitch, boolean dsRejectResidual,
                          com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams dsParams,
                          boolean incrementalTransformerControl, int incTvcMaxTapShift,
                          boolean incrementalShuntControl, int incShuntMaxSectionShift,
                          boolean simpleShuntControl,
                          boolean incrementalRpcControl, boolean incrementalPhaseControl,
                          boolean simpleTransformerControl, boolean tvcUseInitialTapPosition,
                          boolean hvdcFreezing, boolean hvdcLimits,
                          boolean secondaryVoltageControl, double minPlausibleTargetV, double maxPlausibleTargetV,
                          boolean voltageMonitoring, boolean continuousPhaseControl, boolean contingencyLossPreDistribution,
                          boolean rlCurveExtrapolate, boolean forceTargetQ,
                          PerEqCriteria perEqCriteria,
                          double genVcMinNominalV,
                          boolean rlRobustMode, double minRealisticVoltage, double maxRealisticVoltage,
                          double minNominalVoltageRealisticVoltageCheck, int realisticFixingLoop,
                          boolean robustSolveMode, SlackRelocation slackRelocation,
                          /** Where the chunk solver puts every CONVERGED scenario's state as its chunk completes, keyed by
                           *  contingency id, instead of carrying it in its result - a run whose states would not fit the
                           *  heap gives a disk-backed map (GpuStateStore); null keeps the states in the results. */
                          java.util.Map<String, double[]> stateSink,
                          /** The GPU solver's parameters (ordering, matching, pivoting, iterative refinement): the run's;
                           *  null for the configuration module's, which is what a run without them gets. Installed on the
                           *  device by every batched solve, so a direct caller cannot inherit another run's. */
                          GpuLoadFlowParameters gpuParameters) {

    /** PER_EQUATION_TYPE stopping thresholds (OLF PerEquationTypeStoppingCriteria ctor order;
     *  p/q in MW/MVAr, the rest pu/rad); null = UNIFORM_CRITERIA. */
    public record PerEqCriteria(double eps, double p, double q, double v, double angle, double ratio, double b) {
    }

    /** The device defaults — no state-vector scaling, no transformer control, device outer-loop
     *  constants and LEAVE_ON_SLACK_BUS behavior (never reject a saturated scenario). This is what every non-provider entry point (tests, direct callers) uses. */
    public static final BatchConfig DEVICE_DEFAULTS = new BatchConfig(GpuStateVectorScalingMode.NONE, 0, 0, false, 0,
            com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop.MAX_SWITCH_PQ_PV_DEFAULT_VALUE,   // OLF's cap: 0 means never switch back
            false,
            com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams.DEFAULT, false, 3, false, 3, false, false, false, false, false, false, false,
            false, 0.8, 1.2, false, false, false, false, false, null, -1,
            true, com.powsybl.openloadflow.ac.AcLoadFlowParameters.DEFAULT_MIN_REALISTIC_VOLTAGE,
            com.powsybl.openloadflow.ac.AcLoadFlowParameters.DEFAULT_MAX_REALISTIC_VOLTAGE,
            com.powsybl.openloadflow.ac.AcLoadFlowParameters.DEFAULT_MIN_NOMINAL_VOLTAGE_REALISTIC_VOLTAGE_CHECK, -1,
            false, null, null, null);

    /**
     * This config with the ROBUST single-matrix solve (mode 6) selected, for the SECOND device pass over
     * scenarios the uniform batch could not solve.
     *
     * <p>A phase marker rather than a run setting, which is why it is derived here instead of being set
     * at the two construction sites: the retry pass re-solves only the rejected scenarios, in mode 6's
     * per-scenario loop with matching and global column pivoting. The base load flow already falls into
     * that mode automatically after a linear-solve reject ({@code GpuNewtonRaphson}); this is the same
     * remedy for a CHUNK scenario, which previously had no route to it and went to the CPU instead.
     */
    public BatchConfig withRobustSolveMode() {
        return new BatchConfig(lsMode, maxDv, maxDphi, transformerControl, maxOuterIterations, maxPqPvSwitch,
                dsRejectResidual, dsParams, incrementalTransformerControl, incTvcMaxTapShift,
                incrementalShuntControl, incShuntMaxSectionShift, simpleShuntControl, incrementalRpcControl,
                incrementalPhaseControl, simpleTransformerControl, tvcUseInitialTapPosition, hvdcFreezing,
                hvdcLimits, secondaryVoltageControl, minPlausibleTargetV, maxPlausibleTargetV, voltageMonitoring,
                continuousPhaseControl, contingencyLossPreDistribution, rlCurveExtrapolate, forceTargetQ, perEqCriteria,
                genVcMinNominalV, rlRobustMode, minRealisticVoltage, maxRealisticVoltage,
                minNominalVoltageRealisticVoltageCheck, realisticFixingLoop, true, slackRelocation, stateSink,
                gpuParameters);
    }

    /** This config with the given state sink (see {@link #stateSink()}). */
    public BatchConfig withStateSink(java.util.Map<String, double[]> sink) {
        return new BatchConfig(lsMode, maxDv, maxDphi, transformerControl, maxOuterIterations, maxPqPvSwitch,
                dsRejectResidual, dsParams, incrementalTransformerControl, incTvcMaxTapShift,
                incrementalShuntControl, incShuntMaxSectionShift, simpleShuntControl, incrementalRpcControl,
                incrementalPhaseControl, simpleTransformerControl, tvcUseInitialTapPosition, hvdcFreezing,
                hvdcLimits, secondaryVoltageControl, minPlausibleTargetV, maxPlausibleTargetV, voltageMonitoring,
                continuousPhaseControl, contingencyLossPreDistribution, rlCurveExtrapolate, forceTargetQ, perEqCriteria,
                genVcMinNominalV, rlRobustMode, minRealisticVoltage, maxRealisticVoltage,
                minNominalVoltageRealisticVoltageCheck, realisticFixingLoop, robustSolveMode, slackRelocation, sink,
                gpuParameters);
    }

    /** This config with the given GPU solver parameters (see {@link #gpuParameters()}). */
    public BatchConfig withGpuParameters(GpuLoadFlowParameters parameters) {
        return new BatchConfig(lsMode, maxDv, maxDphi, transformerControl, maxOuterIterations, maxPqPvSwitch,
                dsRejectResidual, dsParams, incrementalTransformerControl, incTvcMaxTapShift,
                incrementalShuntControl, incShuntMaxSectionShift, simpleShuntControl, incrementalRpcControl,
                incrementalPhaseControl, simpleTransformerControl, tvcUseInitialTapPosition, hvdcFreezing,
                hvdcLimits, secondaryVoltageControl, minPlausibleTargetV, maxPlausibleTargetV, voltageMonitoring,
                continuousPhaseControl, contingencyLossPreDistribution, rlCurveExtrapolate, forceTargetQ, perEqCriteria,
                genVcMinNominalV, rlRobustMode, minRealisticVoltage, maxRealisticVoltage,
                minNominalVoltageRealisticVoltageCheck, realisticFixingLoop, robustSolveMode, slackRelocation, stateSink,
                parameters);
    }

    /** The GPU solver's parameters the device runs with: the run's, or the configuration module's. */
    public GpuLoadFlowParameters resolvedGpuParameters() {
        return gpuParameters != null ? gpuParameters : GpuLoadFlowParameters.load(PlatformConfig.defaultConfig());
    }

    public int dsRejectResidualFlag() {
        return dsRejectResidual ? 1 : 0;
    }
}
