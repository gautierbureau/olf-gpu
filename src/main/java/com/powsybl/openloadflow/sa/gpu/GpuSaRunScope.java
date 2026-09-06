/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchConfig;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchTransformerControl;
import com.powsybl.openloadflow.ac.solver.gpu.batch.OuterLoopMode;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.util.PerUnit;
import com.powsybl.security.SecurityAnalysisParameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

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
        // INCREMENTAL (OLF's default TVC mode): the DEVICE runs the loop's decisions per scenario around
        // batched solves with in-kernel dV/drho sensitivities (GpuBatchIncrementalTvc); stage-1 scope is
        // the same as the continuous port (no generator VC under the transformer-controlled level).
        boolean incrementalTvcBatchable =
                olfGuardParams.getTransformerVoltageControlMode() == OpenLoadFlowParameters.TransformerVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL
                && GpuBatchTransformerControl.transformerControlSupported(lfNetwork);
        // INCREMENTAL shunt VC: same per-scenario orchestrator pattern (GpuBatchIncrementalShuntVc)
        // with in-kernel dV/dB sensitivities.
        boolean incrementalShuntBatchable =
                olfGuardParams.getShuntVoltageControlMode() == OpenLoadFlowParameters.ShuntVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL;
        // SIMPLE (WITH_GENERATOR) TVC: continuous first solve (VC-on activation deltas) then a single
        // round-to-closest-tap re-solve (GpuBatchSimpleTvc); same stage-1 structural scope.
        // HVDC AC-emulation loops WITH HVDCs present (GpuBatchHvdcAcEmulation): the device pack extracts
        // the LINEAR coupling, so every AC-emu HVDC must start LINEAR (a saturated/frozen base -> CPU).
        boolean hvdcBatchable = noHvdc
                || com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchHvdcAcEmulation.allLinearAtBase(lfNetwork);
        // Secondary voltage control (GpuBatchSecondaryVc): batched device sensitivities + the zone
        // system replayed per scenario on the host; stage-1 scope is LOCAL generator VC only.
        boolean svcBatchable = com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSecondaryVc.supported(lfNetwork);
        // Voltage monitoring (SVC stand-by automatons, GpuBatchVoltageMonitoring): batchable when every
        // monitoring SVC is local with an automaton — slope-less plain-VOLTAGE SVCs need no loop support
        // (ordinary generator-VC buses on the device).
        boolean monitoringBatchable = com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchVoltageMonitoring.supported(lfNetwork);
        // CONTINUOUS phase control (GpuBatchContinuousPhase): CONTROLLER mode batches LOCAL and REMOTE
        // (2026-09-05 — cbPcSide carries the controller's a1 column); LIMITER must stay LOCAL (OLF's
        // own loop reads the controller's I terms).
        boolean continuousPhaseBatchable = lfNetwork.getBranches().stream()
                .map(b -> b.getPhaseControl().orElse(null))
                .filter(java.util.Objects::nonNull)
                .allMatch(pc -> pc.getMode() == com.powsybl.openloadflow.network.TransformerPhaseControl.Mode.CONTROLLER
                        || pc.getControlledBranch() == pc.getControllerBranch());
        boolean simpleTvcBatchable =
                olfGuardParams.getTransformerVoltageControlMode() == OpenLoadFlowParameters.TransformerVoltageControlMode.WITH_GENERATOR_VOLTAGE_CONTROL
                && GpuBatchTransformerControl.transformerControlSupported(lfNetwork);
        // REMOTE/DISTRIBUTED generator VC groups + reactive limits: the device RL pack has no entry
        // for an unpinned remote controller (it has no own BUS_TARGET_V), a base-pinned member's
        // switch-BACK would restore a v-identity instead of the group's DISTR_Q row, and the q-scatter
        // routing (qRow suppression, free-controller role) is a per-structure SNAPSHOT of the base pin
        // state — so any scenario that could change a remote member's pin state is unrepresentable in
        // BOTH modes (the hybrid re-extracts activation only, never the scatter routing). A member can
        // change state only if its bus's Q limits are FINITE (an unlimited bus never pins); gate on that.
        boolean remoteRlPinnable = lfNetwork.getBuses().stream()
                .map(b -> b.getGeneratorVoltageControl()
                        .filter(vc -> vc.getMergeStatus() == com.powsybl.openloadflow.network.VoltageControl.MergeStatus.MAIN
                                && !vc.isLocalControl())
                        .orElse(null))
                .filter(java.util.Objects::nonNull)
                .distinct()
                .flatMap(vc -> vc.getMergedControllerElements().stream())
                .anyMatch(ctl -> ctl.getMinQ() > -1e9 || ctl.getMaxQ() < 1e9);
        boolean noSvc = lfNetwork.getBuses().stream()
                .flatMap(b -> b.getGenerators().stream())
                .noneMatch(g -> g instanceof com.powsybl.openloadflow.network.LfStaticVarCompensator);
        // DEVICE (CUDA outer loops) vs JAVA (OLF's Java outer loops per scenario) — §5.1 of the design.
        OuterLoopMode outerLoopMode = readOuterLoopMode();
        LoadFlowParameters lfGuardParams = securityAnalysisParameters.getLoadFlowParameters();
        List<String> unsupportedOuterLoops;
        if (outerLoopMode == OuterLoopMode.JAVA) {
            // JAVA runs OLF's OWN outer-loop code per scenario, so there is no device-port parameterization
            // limit — only not-yet-validated loops and config the compact snapshot cannot isolate fall back.
            unsupportedOuterLoops = javaUnsupportedOuterLoops(lfNetwork, acParameters, noHvdc, noSvc, remoteRlPinnable);
        } else {
            unsupportedOuterLoops = new ArrayList<>(acParameters.getOuterLoops().stream()
                    .map(AcOuterLoop::getName)
                    .filter(name -> !DistributedSlackOuterLoop.NAME.equals(name)
                            && !ReactiveLimitsOuterLoop.NAME.equals(name)
                            && !(transformerControlBatchable && "TransformerVoltageControl".equals(name))
                            && !(incrementalTvcBatchable && "IncrementalTransformerVoltageControl".equals(name))
                            && !(incrementalShuntBatchable && "IncrementalShuntVoltageControl".equals(name))
                            && !(simpleTvcBatchable && "SimpleTransformerVoltageControl".equals(name))
                            && !(svcBatchable && "SecondaryVoltageControl".equals(name))
                            && !"IncrementalPhaseControl".equals(name)
                            && !"IncrementalTransformerReactivePowerControl".equals(name)
                            && !(continuousPhaseBatchable && "PhaseControl".equals(name))
                            && !(hvdcBatchable && ("AcHvdcAcEmulationLimits".equals(name) || "FreezingHvdcACEmulation".equals(name)))
                            && !(monitoringBatchable && "VoltageMonitoring".equals(name)))
                    .toList());
            // Parameterization corner cases the device DS loop does not model (see
            // docs/gpu-vs-cpu-capabilities.md Part II §A): the device weights are always maxP/droop with
            // active limits — any other balance type or unlimited distribution must fall back whole-run.
            boolean dsLoopActive = acParameters.getOuterLoops().stream()
                    .anyMatch(ol -> DistributedSlackOuterLoop.NAME.equals(ol.getName()));
            // The device DS pass models every STATIC-factor generation balance type (P_MAX, P,
            // PARTICIPATION_FACTOR — see GpuDsParams) and both useActiveLimits settings. REMAINING_MARGIN
            // (mismatch-sign-dependent factors) and the load-based types still fall back whole-run.
            if (dsLoopActive && !com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams.supports(lfGuardParams.getBalanceType())) {
                unsupportedOuterLoops.add("DistributedSlack[balanceType=" + lfGuardParams.getBalanceType() + "]");
            }
            // loadPowerFactorConstant rescales each distributed load's targetQ alongside targetP — a per-round
            // bus Q-target mutation the device DS pass does not model.
            boolean loadBased = lfGuardParams.getBalanceType() == LoadFlowParameters.BalanceType.PROPORTIONAL_TO_LOAD
                    || lfGuardParams.getBalanceType() == LoadFlowParameters.BalanceType.PROPORTIONAL_TO_CONFORM_LOAD;
            if (dsLoopActive && loadBased && olfGuardParams.isLoadPowerFactorConstant()) {
                unsupportedOuterLoops.add("DistributedSlack[loadPowerFactorConstant]");
            }
            // Voltage-dependent load model + LOAD-based balance: distribution rescales each load's
            // targetP, which scales the vdl term coefficients per round — static device coefs would
            // go stale mid-run.
            if (dsLoopActive && loadBased && olfGuardParams.isUseLoadModel()) {
                unsupportedOuterLoops.add("DistributedSlack[useLoadModel+loadBased]");
            }
            if (remoteRlPinnable && acParameters.getOuterLoops().stream()
                    .anyMatch(ol -> ReactiveLimitsOuterLoop.NAME.equals(ol.getName()))) {
                unsupportedOuterLoops.add("ReactiveLimits[remoteGroupPinDynamics]");
            }
            // Curve-limit extrapolation: the device curve evaluator implements the CLAMP branch only
            // (OLF default); slope extrapolation outside [minP, maxP] -> CPU.
            if (olfGuardParams.isExtrapolateReactiveLimits()
                    && com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchRlCurves.hasCurveGenerators(lfNetwork)) {
                unsupportedOuterLoops.add("ReactiveLimits[extrapolateReactiveLimits+curves]");
            }
        }
        // Parameterization the device loops DO model — wire the run's values in place of the
        // device defaults (maxOuterLoopIterations, reactiveLimitsMaxPqPvSwitch), and make a
        // saturated slack distribution reject its scenario to the CPU fallback whenever the
        // failure behavior is not LEAVE_ON_SLACK_BUS (e.g. RTE production uses FAIL).
        int cfgMaxOuterIterations = olfGuardParams.getMaxOuterLoopIterations();
        int cfgMaxPqPvSwitch = olfGuardParams.getReactiveLimitsMaxPqPvSwitch();
        boolean cfgDsRejectResidual = olfGuardParams.getSlackDistributionFailureBehavior()
                != com.powsybl.openloadflow.OpenLoadFlowParameters.SlackDistributionFailureBehavior.LEAVE_ON_SLACK_BUS;
        // OLF's SA distributes each contingency's SCHEDULED active power loss BEFORE the post-contingency
        // solve (ContingencyActivePowerLossDistribution SPI). The batch replays the "Default" impl
        // host-side (GpuBatchLossPreDistribution); a custom SPI cannot be replayed -> whole-run CPU.
        String lossDistName = com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters
                .getOrDefault(securityAnalysisParameters).getContingencyActivePowerLossDistribution();
        boolean defaultLossDist = com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters
                .CONTINGENCY_ACTIVE_POWER_LOSS_DISTRIBUTION_DEFAULT_VALUE.equals(lossDistName);
        boolean dsActive = acParameters.getOuterLoops().stream()
                .anyMatch(ol -> DistributedSlackOuterLoop.NAME.equals(ol.getName()));
        if (!defaultLossDist && dsActive) {
            unsupportedOuterLoops.add("ContingencyActivePowerLossDistribution[" + lossDistName + "]");
        }
        boolean batchDisabled = !unsupportedOuterLoops.isEmpty();
        // Capture which outer loops this run actually applies, so the batched device solver runs the
        // distributed-slack / reactive-limits passes only when OLF does (presence in the resolved outer-loop
        // list is the definitive signal — gens may report participation / buses may be PV regardless).
        boolean distributedSlack = acParameters.getOuterLoops().stream()
                .anyMatch(ol -> DistributedSlackOuterLoop.NAME.equals(ol.getName()));
        boolean reactiveLimits = acParameters.getOuterLoops().stream()
                .anyMatch(ol -> ReactiveLimitsOuterLoop.NAME.equals(ol.getName()));
        // TransformerVoltageControl is the one loop DEVICE batches that JAVA does not yet (§5.7) — so it is
        // never wired when JAVA is selected (the JAVA gate above already routed such a run to CPU).
        boolean cfgTransformerControl = outerLoopMode == OuterLoopMode.DEVICE && transformerControlBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "TransformerVoltageControl".equals(ol.getName()));
        boolean cfgIncrementalTvc = outerLoopMode == OuterLoopMode.DEVICE && incrementalTvcBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "IncrementalTransformerVoltageControl".equals(ol.getName()));
        boolean cfgIncrementalShunt = outerLoopMode == OuterLoopMode.DEVICE && incrementalShuntBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "IncrementalShuntVoltageControl".equals(ol.getName()));
        // The flow-observed incremental family (GpuBatchIncrementalFlowControl): rho -> branch Q, and
        // a1 -> branch P / I. Both are decision-only loops over pinned identity rows — always batchable
        // when their loop is configured (continuous "PhaseControl" stays CPU via the extraction guard).
        boolean cfgIncrementalRpc = outerLoopMode == OuterLoopMode.DEVICE
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "IncrementalTransformerReactivePowerControl".equals(ol.getName()));
        boolean cfgIncrementalPhase = outerLoopMode == OuterLoopMode.DEVICE
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "IncrementalPhaseControl".equals(ol.getName()));
        boolean cfgSimpleTvc = outerLoopMode == OuterLoopMode.DEVICE && simpleTvcBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "SimpleTransformerVoltageControl".equals(ol.getName()));
        // HVDC freezing applies to CONTINGENCY solves whenever the SA-LEVEL startWithFrozenACEmulation
        // is on — and its DEFAULT IS TRUE (OpenSecurityAnalysisParameters), independently of the
        // loadflow-level flag: AcSecurityAnalysis.applyGenericContingencyParameters injects the
        // FreezingHvdcACEmulation loop into every post-contingency solve (freeze at the previous
        // state's flows -> solve -> unfreeze -> re-solve). The base loop list does NOT show it, so
        // keying on acParameters.getOuterLoops() alone silently skipped the freeze phase for the
        // batch (found 2026-09-05: ~1e-4 kV plateau gaps on live-hvdc branch outages — the arms
        // solved along different paths and stopped at different points inside the criterion).
        boolean saContingencyFreeze = com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters
                .getOrDefault(securityAnalysisParameters).isStartWithFrozenACEmulation();
        boolean cfgHvdcFreezing = outerLoopMode == OuterLoopMode.DEVICE && !noHvdc && hvdcBatchable
                && (saContingencyFreeze
                    || acParameters.getOuterLoops().stream().anyMatch(ol -> "FreezingHvdcACEmulation".equals(ol.getName())));
        boolean cfgHvdcLimits = outerLoopMode == OuterLoopMode.DEVICE && !noHvdc && hvdcBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "AcHvdcAcEmulationLimits".equals(ol.getName()));
        boolean cfgSecondaryVc = outerLoopMode == OuterLoopMode.DEVICE && svcBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "SecondaryVoltageControl".equals(ol.getName()));
        boolean cfgVoltageMonitoring = outerLoopMode == OuterLoopMode.DEVICE && monitoringBatchable && !noSvc
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "VoltageMonitoring".equals(ol.getName()));
        boolean cfgContinuousPhase = outerLoopMode == OuterLoopMode.DEVICE && continuousPhaseBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "PhaseControl".equals(ol.getName()));
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
                cfgTransformerControl, cfgMaxOuterIterations, cfgMaxPqPvSwitch, cfgDsRejectResidual, outerLoopMode,
                new com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams(
                        lfGuardParams.getBalanceType(), olfGuardParams.isUseActiveLimits()),
                cfgIncrementalTvc, olfGuardParams.getIncrementalTransformerRatioTapControlOuterLoopMaxTapShift(),
                cfgIncrementalShunt, olfGuardParams.getIncrementalShuntControlOuterLoopMaxSectionShift(),
                cfgIncrementalRpc, cfgIncrementalPhase, cfgSimpleTvc, cfgHvdcFreezing, cfgHvdcLimits,
                cfgSecondaryVc, olfGuardParams.getMinPlausibleTargetVoltage(), olfGuardParams.getMaxPlausibleTargetVoltage(),
                cfgVoltageMonitoring, cfgContinuousPhase, defaultLossDist && dsActive,
                olfParams.getNewtonRaphsonStoppingCriteriaType()
                        == com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteriaType.PER_EQUATION_TYPE_CRITERIA
                        ? new BatchConfig.PerEqCriteria(olfParams.getNewtonRaphsonConvEpsPerEq(),
                                olfParams.getMaxActivePowerMismatch(), olfParams.getMaxReactivePowerMismatch(),
                                olfParams.getMaxVoltageMismatch(), olfParams.getMaxAngleMismatch(),
                                olfParams.getMaxRatioMismatch(), olfParams.getMaxSusceptanceMismatch())
                        : null);
        return new Scope(batchDisabled, unsupportedOuterLoops, distributedSlack, reactiveLimits,
                slackDistTol, reactiveLimitTol, convEpsPerEq, batchConfig);
    }

    /** The outer-loop mode for this run, from the {@code olf.gpu.sa.outerLoopMode} system property
     *  ({@code DEVICE}|{@code JAVA}, default {@link OuterLoopMode#DEVICE}). An unrecognized value logs and
     *  defaults to DEVICE (never fails the run over a typo). */
    static OuterLoopMode readOuterLoopMode() {
        String prop = System.getProperty(GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY);
        if (prop == null || prop.isBlank()) {
            return OuterLoopMode.DEVICE;
        }
        try {
            return OuterLoopMode.valueOf(prop.trim().toUpperCase(java.util.Locale.ROOT));
        } catch (IllegalArgumentException e) {
            LOGGER.warn("Unrecognized {}={} — defaulting to DEVICE (valid: DEVICE, JAVA)",
                    GpuBatchedAcSecurityAnalysis.GPU_OUTER_LOOP_MODE_PROPERTY, prop);
            return OuterLoopMode.DEVICE;
        }
    }

    /** Outer loops the JAVA driver ({@link com.powsybl.openloadflow.ac.solver.gpu.batch.GpuHybridBatchOuterLoopEngine})
     *  is VALIDATED to run today (endgame: all of them, §5.7). JAVA runs OLF's own loop code, so for these there is
     *  NO device-port parameterization limit — the only constraints are (a) loops not yet validated and (b) config
     *  that mutates network state the compact per-scenario snapshot (controller buses + HVDC + generator targetP)
     *  does not yet capture. */
    private static final Set<String> JAVA_SUPPORTED_LOOPS = Set.of(
            DistributedSlackOuterLoop.NAME, ReactiveLimitsOuterLoop.NAME, "VoltageMonitoring",
            "AcHvdcAcEmulationLimits", "FreezingHvdcACEmulation");

    /** The JAVA-mode coverage gate (§5.6): the outer loops / config the JAVA driver cannot yet run, which route
     *  the WHOLE run to the CPU fallback. TEMPORARY — shrinks to empty as coverage lands (§5.7). Differs from the
     *  DEVICE gate: EVERY OLF balance type works (generation- AND load-based — the engine isolates both generator
     *  and load targets per scenario), and no useActiveLimits requirement (JAVA runs OLF's own ActivePowerDistribution). */
    private static List<String> javaUnsupportedOuterLoops(LfNetwork lfNetwork, AcLoadFlowParameters acParameters,
                                                          boolean noHvdc, boolean noSvc, boolean remoteRlPinnableJava) {
        List<String> unsupported = new ArrayList<>();
        for (AcOuterLoop ol : acParameters.getOuterLoops()) {
            String name = ol.getName();
            boolean ok = JAVA_SUPPORTED_LOOPS.contains(name)
                    && !(ReactiveLimitsOuterLoop.NAME.equals(name) && remoteRlPinnableJava)
                    // TransformerVoltageControl: the JAVA engine runs OLF's real transformer VC loop; the continuous
                    // ratio is solved INSIDE the device inner Newton (rho is a BRANCH_RHO1 variable in es), the loop
                    // makes only the discrete tap DECISIONS, and the compact snapshot now isolates the transformer
                    // BranchState (tap/r1/VC-enabled) per scenario. Stage-1 scope (no generator VC under the
                    // transformer-controlled level) is enforced exactly like the DEVICE path.
                    // ALL THREE transformer-VC loop flavors: AFTER_GENERATOR (TransformerVoltageControl),
                    // WITH_GENERATOR (SimpleTransformerVoltageControl) and INCREMENTAL — OLF's DEFAULT mode —
                    // (IncrementalTransformerVoltageControl). The JAVA engine replays OLF's real loop either
                    // way; the continuous ratio (when the loop uses one) is a BRANCH_RHO1 variable solved
                    // in-Newton, discrete tap moves reach the device via the per-scenario pi-model refresh
                    // (r1Ov carries all 8 statics), and the OLF-faithful re-solve initialization starts a
                    // just-pinned ratio exactly at its pinned value.
                    || ("TransformerVoltageControl".equals(name) || "SimpleTransformerVoltageControl".equals(name)
                            || "IncrementalTransformerVoltageControl".equals(name))
                            && GpuBatchTransformerControl.transformerControlSupported(lfNetwork)
                    // ShuntVoltageControl: the JAVA engine runs OLF's real shunt VC loop; the continuous
                    // susceptance is solved INSIDE the device inner Newton (SHUNT_B is a variable in es) and the
                    // compact snapshot isolates the controller-shunt b + VC-enabled per scenario (BusState).
                    // Dynamic Q-relocation (§11) — which keeps the shared structure correct as the one-terminal
                    // SHUNT_B toggles — is auto-enabled in the CUDA structure whenever a controllable bus is
                    // present, so no gate is needed here (unsupported shunt shapes throw at extraction and the
                    // precompute falls back to CPU).
                    || "ShuntVoltageControl".equals(name)
                    // IncrementalShuntVoltageControl: moves discrete sections directly; b follows via the pinned
                    // SHUNT_B variable (capability-based) and the per-scenario (g,b) shOv refresh carries any
                    // section-driven g change to the device.
                    || "IncrementalShuntVoltageControl".equals(name)
                    // IncrementalPhaseControl / IncrementalTransformerReactivePowerControl: both move discrete
                    // taps directly; a1/rho follow via their pinned variables (targets re-read per round +
                    // faithful re-init) and the per-scenario pi-model refresh carries the tap's R/X/Y/ksi/a1.
                    || "IncrementalPhaseControl".equals(name)
                    || "IncrementalTransformerReactivePowerControl".equals(name)
                    // CONTINUOUS PhaseControl (2026-09-03): the loop's enable/disable of the CONTROLLER-mode
                    // control toggles the BRANCH_TARGET_P row, which the activation extractor maps to the
                    // batch row mode 3 (the cb family re-emits the controlled side's P block at the α1 row) —
                    // the hybrid re-extracts activation after every loop mutation, and the rounding/limiter
                    // tap moves reach the device via the per-scenario pi-model refresh. CONTROLLER mode may
                    // be REMOTE (2026-09-05); LIMITER stays local (OLF's own loop is local-only).
                    || "PhaseControl".equals(name) && lfNetwork.getBranches().stream()
                            .map(b -> b.getPhaseControl().orElse(null))
                            .filter(java.util.Objects::nonNull)
                            .allMatch(pc -> pc.getMode() == com.powsybl.openloadflow.network.TransformerPhaseControl.Mode.CONTROLLER
                                    || pc.getControlledBranch() == pc.getControllerBranch())
                    // SecondaryVoltageControl: adjusts GENERATOR-VC TARGET VALUES only (pilot-point regulation);
                    // targets are re-read per round and the engine isolates every MAIN generator VC's target per
                    // scenario (vcSnap), so the hybrid replays OLF's loop verbatim.
                    || "SecondaryVoltageControl".equals(name);
            // conservative structural no-ops: allow the HVDC/SVC-touching loops only when they cannot
            // fire yet (widen once JAVA's HVDC/SVC handling is validated — the snapshot already captures
            // HvdcState, and SVCs are generator-voltage-control buses already in the controller set).
            // VoltageMonitoring now supported WITH SVCs present: the loop's mutations (re-enable generator VC,
            // move VC targets, flip an SVC's generator control type) are all isolated per scenario — BusState
            // captures generatorsControlType + VC-enabled, and the engine's vcSnap isolates VC target values.
            // (An SVC with a voltage SLOPE creates equation terms outside the device whitelist — extraction
            // rejects them and the precompute falls back to CPU by construction, so no gate is needed.)
            // HVDC loops now supported WITH HVDCs present: a non-LINEAR AC-emulation status deactivates the
            // hvdc coupling on the CPU; the engine mirrors it per scenario via the hv element's disabled bit
            // (fillHvdcBatch honors it) and the re-read bus targets carry the frozen/saturated VSC injections.
            if (!ok) {
                unsupported.add(name);
            }
        }
        // Distributed slack: the JAVA engine isolates BOTH generator targetP AND load targetP/targetQ per scenario
        // (explicit flat arrays), so every OLF balance type — generation-based AND load-based
        // (PROPORTIONAL_TO_[CONFORM_]LOAD) — is safe. This is strictly beyond the DEVICE gate, which is
        // PROPORTIONAL_TO_GENERATION_P_MAX-only. (No unsupported balance type remains for JAVA today.)
        return unsupported;
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
