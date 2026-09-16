/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

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
 *   <li>{@code outerLoopMode}: {@link OuterLoopMode#DEVICE} (CUDA-kernel outer loops folded into the
 *       device solve) or {@link OuterLoopMode#JAVA} (device inner solve only, OLF's Java outer loops
 *       make the decisions per scenario via {@link GpuHybridBatchOuterLoopEngine}).</li>
 * </ul>
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record BatchConfig(int lsMode, double maxDv, double maxDphi, boolean transformerControl,
                          int maxOuterIterations, int maxPqPvSwitch, boolean dsRejectResidual,
                          OuterLoopMode outerLoopMode,
                          com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams dsParams,
                          boolean incrementalTransformerControl, int incTvcMaxTapShift,
                          boolean incrementalShuntControl, int incShuntMaxSectionShift,
                          boolean simpleShuntControl,
                          boolean incrementalRpcControl, boolean incrementalPhaseControl,
                          boolean simpleTransformerControl, boolean tvcUseInitialTapPosition,
                          boolean hvdcFreezing, boolean hvdcLimits,
                          boolean secondaryVoltageControl, double minPlausibleTargetV, double maxPlausibleTargetV,
                          boolean voltageMonitoring, boolean continuousPhaseControl, boolean contingencyLossPreDistribution,
                          boolean rlCurveExtrapolate,
                          PerEqCriteria perEqCriteria,
                          double genVcMinNominalV,
                          boolean rlRobustMode, double minRealisticVoltage, double maxRealisticVoltage,
                          double minNominalVoltageRealisticVoltageCheck) {

    /** PER_EQUATION_TYPE stopping thresholds (OLF PerEquationTypeStoppingCriteria ctor order;
     *  p/q in MW/MVAr, the rest pu/rad); null = UNIFORM_CRITERIA. */
    public record PerEqCriteria(double eps, double p, double q, double v, double angle, double ratio, double b) {
    }

    /** The device defaults — no state-vector scaling, no transformer control, device outer-loop
     *  constants, LEAVE_ON_SLACK_BUS behavior (never reject a saturated scenario), and the DEVICE
     *  outer-loop mode. This is what every non-provider entry point (tests, direct callers) uses. */
    public static final BatchConfig DEVICE_DEFAULTS = new BatchConfig(0, 0, 0, false, 0, 0, false, OuterLoopMode.DEVICE,
            com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams.DEFAULT, false, 3, false, 3, false, false, false, false, false, false, false,
            false, 0.8, 1.2, false, false, false, false, null, -1,
            true, com.powsybl.openloadflow.ac.AcLoadFlowParameters.DEFAULT_MIN_REALISTIC_VOLTAGE,
            com.powsybl.openloadflow.ac.AcLoadFlowParameters.DEFAULT_MAX_REALISTIC_VOLTAGE,
            com.powsybl.openloadflow.ac.AcLoadFlowParameters.DEFAULT_MIN_NOMINAL_VOLTAGE_REALISTIC_VOLTAGE_CHECK);

    int dsRejectResidualFlag() {
        return dsRejectResidual ? 1 : 0;
    }
}
