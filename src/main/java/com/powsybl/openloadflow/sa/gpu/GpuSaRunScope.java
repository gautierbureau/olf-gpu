/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchConfig;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchTransformerControl;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.util.PerUnit;
import com.powsybl.security.SecurityAnalysisParameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * The GPU batched-SA run-scope decision (the DETECT-AND-FALLBACK safety gate): given the resolved
 * outer loops and loadflow parameters, decides whether the batched device path applies at all
 * ({@code batchDisabled} when any unsupported outer loop is present) and derives the per-call
 * {@link BatchConfig} + device outer-loop tolerances aligned to the CPU loadflow's criteria. Pure
 * function of the parameters — no side effects. Also probes whether the GPU full Newton-Raphson can
 * solve the network at all ({@link #gpuSupports}).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuSaRunScope {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuSaRunScope.class);

    private GpuSaRunScope() {
    }

    /** The batchability verdict + the run's captured outer-loop parameterization. {@code batchDisabled}
     *  routes every contingency to the CPU; the rest wire the device DS/RL passes and their tolerances. */
    record Scope(boolean batchDisabled, List<String> unsupportedOuterLoops, boolean distributedSlack,
                 boolean reactiveLimits, double slackDistTol, double reactiveLimitTol, double convEpsPerEq,
                 BatchConfig batchConfig) {
    }

    static Scope evaluate(LfNetwork lfNetwork, AcLoadFlowParameters acParameters,
                          SecurityAnalysisParameters securityAnalysisParameters) {
        boolean noHvdc = lfNetwork.getHvdcs().isEmpty();
        // TransformerVoltageControl is BATCHABLE since the per-scenario orchestrator port
        // (AFTER_GENERATOR mode only; stage-1 scope = no generator voltage controls under the
        // transformer-controlled level — the solver checks). Anything else falls back.
        OpenLoadFlowParameters olfGuardParams = OpenLoadFlowParameters.get(securityAnalysisParameters.getLoadFlowParameters());
        boolean transformerControlBatchable =
                olfGuardParams.getTransformerVoltageControlMode() == OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL
                && GpuBatchTransformerControl.transformerControlSupported(lfNetwork);
        boolean noSvc = lfNetwork.getBuses().stream()
                .flatMap(b -> b.getGenerators().stream())
                .noneMatch(g -> g instanceof com.powsybl.openloadflow.network.LfStaticVarCompensator);
        List<String> unsupportedOuterLoops = new java.util.ArrayList<>(acParameters.getOuterLoops().stream()
                .map(ol -> ol.getName())
                .filter(name -> !DistributedSlackOuterLoop.NAME.equals(name)
                        && !ReactiveLimitsOuterLoop.NAME.equals(name)
                        && !(transformerControlBatchable && "TransformerVoltageControl".equals(name))
                        && !(noHvdc && ("AcHvdcAcEmulationLimits".equals(name) || "FreezingHvdcACEmulation".equals(name)))
                        && !(noSvc && "VoltageMonitoring".equals(name)))
                .toList());
        // Parameterization corner cases the device DS loop does not model (see
        // docs/gpu-vs-cpu-capabilities.md Part II §A): the device weights are always maxP/droop with
        // active limits — any other balance type or unlimited distribution must fall back whole-run.
        boolean dsLoopActive = acParameters.getOuterLoops().stream()
                .anyMatch(ol -> DistributedSlackOuterLoop.NAME.equals(ol.getName()));
        com.powsybl.loadflow.LoadFlowParameters lfGuardParams = securityAnalysisParameters.getLoadFlowParameters();
        if (dsLoopActive && lfGuardParams.getBalanceType()
                != com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX) {
            unsupportedOuterLoops.add("DistributedSlack[balanceType=" + lfGuardParams.getBalanceType() + "]");
        }
        if (dsLoopActive && !olfGuardParams.isUseActiveLimits()) {
            unsupportedOuterLoops.add("DistributedSlack[useActiveLimits=false]");
        }
        // Parameterization the device loops DO model — wire the run's values in place of the
        // device defaults (maxOuterLoopIterations, reactiveLimitsMaxPqPvSwitch), and make a
        // saturated slack distribution reject its scenario to the CPU fallback whenever the
        // failure behavior is not LEAVE_ON_SLACK_BUS (e.g. RTE production uses FAIL).
        int cfgMaxOuterIterations = olfGuardParams.getMaxOuterLoopIterations();
        int cfgMaxPqPvSwitch = olfGuardParams.getReactiveLimitsMaxPqPvSwitch();
        boolean cfgDsRejectResidual = olfGuardParams.getSlackDistributionFailureBehavior()
                != com.powsybl.openloadflow.OpenLoadFlowParameters.SlackDistributionFailureBehavior.LEAVE_ON_SLACK_BUS;
        boolean batchDisabled = !unsupportedOuterLoops.isEmpty();
        // Capture which outer loops this run actually applies, so the batched device solver runs the
        // distributed-slack / reactive-limits passes only when OLF does (presence in the resolved outer-loop
        // list is the definitive signal — gens may report participation / buses may be PV regardless).
        boolean distributedSlack = acParameters.getOuterLoops().stream()
                .anyMatch(ol -> DistributedSlackOuterLoop.NAME.equals(ol.getName()));
        boolean reactiveLimits = acParameters.getOuterLoops().stream()
                .anyMatch(ol -> ReactiveLimitsOuterLoop.NAME.equals(ol.getName()));
        boolean cfgTransformerControl = transformerControlBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "TransformerVoltageControl".equals(ol.getName()));
        // Align the device outer-loop tolerances to the CPU loadflow's (per-unit on the 100 MVA base). The
        // reactive-limits PV→PQ switch threshold is NOT always maxReactivePowerMismatch: OLF derives it from the
        // stopping-criteria type exactly as AbstractAcOuterLoopConfig.createReactiveLimitsOuterLoop does —
        // UNIFORM_CRITERIA (the default) ties it to convEpsPerEq, PER_EQUATION_TYPE_CRITERIA to
        // maxReactivePowerMismatch/SB. Deriving it the same way keeps the device switch DECISIONS identical to
        // OLF when a user changes convEpsPerEq (with defaults both are 1e-4, which masked the divergence).
        OpenLoadFlowParameters olfParams = OpenLoadFlowParameters.get(securityAnalysisParameters.getLoadFlowParameters());
        double slackDistTol = olfParams.getSlackBusPMaxMismatch() / PerUnit.SB;
        double reactiveLimitTol = switch (olfParams.getNewtonRaphsonStoppingCriteriaType()) {
            case UNIFORM_CRITERIA -> olfParams.getNewtonRaphsonConvEpsPerEq();
            case PER_EQUATION_TYPE_CRITERIA -> olfParams.getMaxReactivePowerMismatch() / PerUnit.SB;
        };
        // Device inner Newton uses OLF's exact criterion for UNIFORM_CRITERIA (so switch decisions are made at
        // the same convergence depth as the CPU); the device infinity-norm floor otherwise.
        double convEpsPerEq = switch (olfParams.getNewtonRaphsonStoppingCriteriaType()) {
            case UNIFORM_CRITERIA -> olfParams.getNewtonRaphsonConvEpsPerEq();
            case PER_EQUATION_TYPE_CRITERIA -> 0.0;
        };
        // MAX_VOLTAGE_CHANGE state-vector scaling: apply the SAME per-step clamp in the batched device
        // solve as the CPU per-contingency Newton (per scenario). Without this the batch takes raw full
        // steps, PV/PQ switch decisions read different intermediate states, and converged states drift
        // (measured 2026-07-17 under the RTE parameters: 6,137 one-sided violation diffs).
        int cfgLsMode = 0;
        double cfgMaxDv = 0.0;
        double cfgMaxDphi = 0.0;
        if (olfParams.getStateVectorScalingMode() == com.powsybl.openloadflow.ac.solver.StateVectorScalingMode.MAX_VOLTAGE_CHANGE) {
            cfgLsMode = 2;
            cfgMaxDv = olfParams.getMaxVoltageChangeStateVectorScalingMaxDv();
            cfgMaxDphi = olfParams.getMaxVoltageChangeStateVectorScalingMaxDphi();
        }
        // Assemble the run's per-call batched config (replaces the solver's former JVM-global statics).
        BatchConfig batchConfig = new BatchConfig(cfgLsMode, cfgMaxDv, cfgMaxDphi,
                cfgTransformerControl, cfgMaxOuterIterations, cfgMaxPqPvSwitch, cfgDsRejectResidual);
        return new Scope(batchDisabled, unsupportedOuterLoops, distributedSlack, reactiveLimits,
                slackDistTol, reactiveLimitTol, convEpsPerEq, batchConfig);
    }

    /** True if the GPU full Newton-Raphson can solve this network — a throwaway extract (which VALIDATES and
     *  throws on anything uncovered) is the same gate the batch uses, so the base never runs on the GPU for a
     *  network whose contingencies would fall back to the CPU anyway. */
    static boolean gpuSupports(LfNetwork lfNetwork, AcLoadFlowParameters acParameters) {
        try {
            EquationSystem<AcVariableType, AcEquationType> es =
                    new AcEquationSystemCreator(lfNetwork, acParameters.getEquationSystemCreationParameters()).create();
            GpuAcDataExtractor.extract(lfNetwork, es);
            return true;
        } catch (RuntimeException e) {
            LOGGER.debug("GPU batched SA: base load flow stays on CPU (network not GPU-supported: {})", e.toString());
            return false;
        }
    }
}
