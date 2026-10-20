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
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuTransformerVoltageControlOuterLoop;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchConfig;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.SlackRelocation;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuStateVectorScalingMode;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.util.PerUnit;
import com.powsybl.security.SecurityAnalysisParameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
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
        // Shared groups (plan-group-member-dynamics A3/A4): AFTER_GENERATOR and WITH_GENERATOR
        // orchestrate them with group-level controllers, INCLUDING the useInitialTapPosition
        // shared-ratio manager since 2026-09-07 (group-averaged clamp bounds + the spread-
        // preserving remap live in the machine; WITH_GENERATOR's loop never used the manager).
        // INCREMENTAL shared groups run since 2026-09-07 too: the loop keeps every branch VC
        // DISABLED (no DISTR_RHO coupling ever activates), so a shared group is purely the
        // one-tap-at-a-time round-robin over the merged controllers — ported in
        // GpuIncrementalTransformerVoltageControlOuterLoop.orchestrate, grouped by the MAIN control's controlled bus.
        // 2026-09-15: generator voltage controls UNDER the transformer-controlled level no longer gate the
        // run — the in-kernel phase runs OLF's GeneratorVoltageControlManager freeze/unfreeze and the
        // fixTransformerVoltageControls check, as long as every candidate is a LOCAL controller and every
        // transformer controller has a tap table (generatorFreezeSupported).
        boolean transformerControlBatchable =
                olfGuardParams.getTransformerVoltageControlMode() == OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL
                && (GpuTransformerVoltageControlOuterLoop.transformerControlSupported(lfNetwork)
                        || GpuTransformerVoltageControlOuterLoop.generatorFreezeSupported(lfNetwork,
                                olfGuardParams.getGeneratorVoltageControlMinNominalVoltage()));
        // INCREMENTAL (OLF's default TVC mode): an in-kernel driver phase (INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL). The loop never
        // freezes generators (no GeneratorVoltageControlManager), so the gen-under-the-controlled-level
        // gate that the AFTER_GENERATOR port needed never applied here — lifted 2026-09-15.
        boolean incrementalTvcBatchable =
                olfGuardParams.getTransformerVoltageControlMode() == OpenLoadFlowParameters.TransformerVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL;
        // INCREMENTAL shunt VC: same per-scenario orchestrator pattern (IncrementalShuntVoltageControl)
        // with in-kernel dV/dB sensitivities.
        boolean incrementalShuntBatchable =
                olfGuardParams.getShuntVoltageControlMode() == OpenLoadFlowParameters.ShuntVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL;
        // CONTINUOUS shunt VC (WITH_GENERATOR — OLF's DEFAULT shunt mode): the SimpleTvc two-phase
        // flow for shunts (GpuShuntVoltageControlOuterLoop: VC-on first solve + dispatchB round + re-solve).
        // Stage-1 scope: standalone of the other DECISION orchestrators — the CPU interleaves
        // simultaneously-unstable loops per outer iteration, which the sequential orchestrator
        // chain does not model; co-presence with any transformer/phase/RPC/SecondaryVc/monitoring
        // loop keeps the whole run on the CPU exactly as before (DS/RL/HVDC phases are in-kernel
        // and compose fine).

        // SIMPLE (WITH_GENERATOR) TVC: continuous first solve (VC-on activation deltas) then a single
        // round-to-closest-tap re-solve (SimpleTransformerVoltageControl); same stage-1 structural scope.
        // HVDC AC-emulation loops WITH HVDCs present (GpuFreezingHvdcACEmulationOuterloop,
        // GpuHvdcAcEmulationLimitsOuterLoop). A SATURATED/FROZEN
        // base is representable since 2026-09-08 (the orchestrator's base-frame overrides:
        // baseTargets - baseD + d(status), base statuses seeded with the hv bits) — but ONLY the
        // orchestrator seeds those bits, so a non-linear base still requires the freeze or limits
        // loop to actually RUN (controllers built); without either (startWithFrozenACEmulation
        // explicitly off and no limits loop) the base coupling would double-count -> keep the gate.
        boolean hvdcLoopsRun = com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters
                .getOrDefault(securityAnalysisParameters).isStartWithFrozenACEmulation()
                || acParameters.getOuterLoops().stream().anyMatch(ol ->
                        "AcHvdcAcEmulationLimits".equals(ol.getName()) || "FreezingHvdcACEmulation".equals(ol.getName()));
        boolean hvdcBatchable = noHvdc
                || com.powsybl.openloadflow.ac.outerloop.gpu.GpuHvdcAcEmulationLimitsOuterLoop.allLinearAtBase(lfNetwork)
                || hvdcLoopsRun;
        // Secondary voltage control (GpuSecondaryVoltageControlOuterLoop): batched device sensitivities + the zone
        // system replayed per scenario on the host; stage-1 scope is LOCAL generator VC only.
        boolean svcBatchable = com.powsybl.openloadflow.ac.outerloop.gpu.GpuSecondaryVoltageControlOuterLoop.supported(lfNetwork);
        // Voltage monitoring (SVC stand-by automatons, VoltageMonitoring): batchable when every
        // monitoring SVC is local with an automaton — slope-less plain-VOLTAGE SVCs need no loop support
        // (ordinary generator-VC buses on the device).
        boolean monitoringBatchable = com.powsybl.openloadflow.ac.outerloop.gpu.GpuMonitoringVoltageOuterLoop.supported(lfNetwork);
        // CONTINUOUS phase control (GpuPhaseControlOuterLoop): both modes batch LOCAL and REMOTE.
        // CONTROLLER since 2026-09-05 (cbPcSide carries the controller's a1 column); LIMITER since
        // 2026-09-16 — OLF's PhaseControlOuterLoop is local BY CONSTRUCTION there, reading the
        // CONTROLLER branch's own I terms and differentiating them, so the controlled branch never
        // enters the decision and a remote LIMITER is the same computation.
        boolean continuousPhaseBatchable = true;
        // (SimpleTransformerVoltageControlOuterLoop has no generator freeze either — no gen-under-level
        // gate, lifted 2026-09-15.)
        boolean simpleTvcBatchable =
                olfGuardParams.getTransformerVoltageControlMode() == OpenLoadFlowParameters.TransformerVoltageControlMode.WITH_GENERATOR_VOLTAGE_CONTROL;
        // REMOTE/DISTRIBUTED generator VC groups + reactive limits. Both modes run the full
        // pin/unpin dynamics: NON-free members via mode-4 transitions + own-q accumulation
        // (plan-B3), the FREE member via the pilot-Q relocation candidate groups (plan-B4), and
        // the ALL-PINNED corner (control lost) via the pilot-vEq flip — on-device by
        // updatePilotVeqBatch (mode-scan fallback = candidate 0, the pilot's own row). Still whole-run CPU:
        //  - a pinnable free member whose group has NO relocation table (fewer than 2 merged
        //    members). Half-open / zero-impedance q terms at members and pilots ride the ob/zi
        //    alternate-dest channels since 2026-09-07 — the unclean-member/pilot conditions are
        //    gone (the pre-channel gap was 3.3e-4 kV on a pinned member with one open line).
        boolean remoteRlUnsupported = remoteRlUnrepresentable(lfNetwork);
        boolean noSvc = lfNetwork.getBuses().stream()
                .flatMap(b -> b.getGenerators().stream())
                .noneMatch(g -> g instanceof com.powsybl.openloadflow.network.LfStaticVarCompensator);
        // CONTINUOUS shunt VC (WITH_GENERATOR — OLF's DEFAULT shunt mode): the SimpleTvc two-phase
        // flow for shunts (GpuShuntVoltageControlOuterLoop: VC-on first solve + dispatchB round + re-solve).
        // Stage-1 scope: standalone of the other DECISION orchestrators — the CPU interleaves
        // simultaneously-unstable loops per outer iteration, which the sequential orchestrator
        // chain does not model; co-presence with a transformer/phase/RPC/SecondaryVc loop (or
        // VoltageMonitoring with actual SVCs — the loop is inert otherwise) keeps the whole run on
        // the CPU exactly as before (DS/RL/HVDC phases are in-kernel and compose fine).
        // 2026-09-15: the co-presence gate is gone entirely. Every transformer, phase, RPC, monitoring and shunt
        // loop is an in-kernel DRIVER PHASE now, swept in OLF's own order with OLF's restart-on-UNSTABLE
        // semantics, so simultaneously unstable loops interleave exactly as on the CPU.
        boolean simpleShuntVcBatchable =
                olfGuardParams.getShuntVoltageControlMode() == OpenLoadFlowParameters.ShuntVoltageControlMode.WITH_GENERATOR_VOLTAGE_CONTROL;
        // SecondaryVoltageControl sits at OLF order 4, BEFORE every control loop (phase 7, transformer 8,
        // RPC 9, shunt 10). As a HOST ORCHESTRATOR it ran AFTER the whole device sweep, so each decided on
        // a state the other had already moved — the inverse of OLF's order (1.7e-3 kV on a zone whose pilot
        // bus feeds an AFTER_GENERATOR transformer; 2.4e-1 kV with the pre-2026-09-15 host chain, whose
        // re-solves do not compose). Since 2026-09-15 the chunk path runs SVC as a DRIVER PAUSE at its own
        // slot, so the order is OLF's and no co-presence gate is needed (tasks #48).
        LoadFlowParameters lfGuardParams = securityAnalysisParameters.getLoadFlowParameters();
        List<String> unsupportedOuterLoops;
        unsupportedOuterLoops = new ArrayList<>(acParameters.getOuterLoops().stream()
                .map(AcOuterLoop::getName)
                .filter(name -> !DistributedSlackOuterLoop.NAME.equals(name)
                        && !ReactiveLimitsOuterLoop.NAME.equals(name)
                        && !(transformerControlBatchable && "TransformerVoltageControl".equals(name))
                        && !(incrementalTvcBatchable && "IncrementalTransformerVoltageControl".equals(name))
                        && !(incrementalShuntBatchable && "IncrementalShuntVoltageControl".equals(name))
                        && !(simpleShuntVcBatchable && "ShuntVoltageControl".equals(name))
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
        // The device DS pass models EVERY balance type (GpuDsParams.supports returns true for all
        // six — incl. the load-based types and REMAINING_MARGIN via the margin factors, GpuDistributedSlackFactorMode)
        // and both useActiveLimits settings, so this guard is future-proofing for balance types
        // OLF adds later, not a live fallback.
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
        // targetP, which scales the load model term coefficients per round — static device coefs would
        // go stale mid-run.
        if (dsLoopActive && loadBased && olfGuardParams.isUseLoadModel()) {
            unsupportedOuterLoops.add("DistributedSlack[useLoadModel+loadBased]");
        }
        if (remoteRlUnsupported && acParameters.getOuterLoops().stream()
                .anyMatch(ol -> ReactiveLimitsOuterLoop.NAME.equals(ol.getName()))) {
            unsupportedOuterLoops.add("ReactiveLimits[remoteGroupPinDynamicsUnrepresentable]");
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
        boolean cfgTransformerControl = transformerControlBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "TransformerVoltageControl".equals(ol.getName()));
        boolean cfgIncrementalTvc = incrementalTvcBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "IncrementalTransformerVoltageControl".equals(ol.getName()));
        boolean cfgIncrementalShunt = incrementalShuntBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "IncrementalShuntVoltageControl".equals(ol.getName()));
        boolean cfgSimpleShunt = simpleShuntVcBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "ShuntVoltageControl".equals(ol.getName()));
        // The flow-observed incremental family (GpuIncrementalTransformerReactivePowerControlOuterLoop,
        // GpuIncrementalPhaseControlOuterLoop): rho -> branch Q, and a1 -> branch P / I. Both are decision-only loops
        // over pinned identity rows — always batchable when their loop is configured (continuous "PhaseControl" stays
        // CPU via the extraction guard).
        boolean cfgIncrementalRpc = acParameters.getOuterLoops().stream().anyMatch(ol -> "IncrementalTransformerReactivePowerControl".equals(ol.getName()));
        boolean cfgIncrementalPhase = acParameters.getOuterLoops().stream().anyMatch(ol -> "IncrementalPhaseControl".equals(ol.getName()));
        boolean cfgSimpleTvc = simpleTvcBatchable
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
        boolean cfgHvdcFreezing = !noHvdc && hvdcBatchable
                && (saContingencyFreeze
                    || acParameters.getOuterLoops().stream().anyMatch(ol -> "FreezingHvdcACEmulation".equals(ol.getName())));
        boolean cfgHvdcLimits = !noHvdc && hvdcBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "AcHvdcAcEmulationLimits".equals(ol.getName()));
        boolean cfgSecondaryVc = svcBatchable
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "SecondaryVoltageControl".equals(ol.getName()));
        boolean cfgVoltageMonitoring = monitoringBatchable && !noSvc
                && acParameters.getOuterLoops().stream().anyMatch(ol -> "VoltageMonitoring".equals(ol.getName()));
        boolean cfgContinuousPhase = continuousPhaseBatchable
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
        // LINE_SEARCH backtracks per lane on the device (the chunk solver installs OLF's max iteration and step fold).
        int cfgLsMode = GpuStateVectorScalingMode.of(olfParams.getStateVectorScalingMode());
        double cfgMaxDv = 0.0;
        double cfgMaxDphi = 0.0;
        if (cfgLsMode == GpuStateVectorScalingMode.MAX_VOLTAGE_CHANGE) {
            cfgMaxDv = olfParams.getMaxVoltageChangeStateVectorScalingMaxDv();
            cfgMaxDphi = olfParams.getMaxVoltageChangeStateVectorScalingMaxDphi();
        }
        // Assemble the run's per-call batched config (replaces the solver's former JVM-global statics).
        BatchConfig batchConfig = new BatchConfig(cfgLsMode, cfgMaxDv, cfgMaxDphi,
                cfgTransformerControl, cfgMaxOuterIterations, cfgMaxPqPvSwitch, cfgDsRejectResidual,
                new com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams(
                        lfGuardParams.getBalanceType(), olfGuardParams.isUseActiveLimits()),
                cfgIncrementalTvc, olfGuardParams.getIncrementalTransformerRatioTapControlOuterLoopMaxTapShift(),
                cfgIncrementalShunt, olfGuardParams.getIncrementalShuntControlOuterLoopMaxSectionShift(),
                cfgSimpleShunt,
                cfgIncrementalRpc, cfgIncrementalPhase, cfgSimpleTvc,
                olfGuardParams.isTransformerVoltageControlUseInitialTapPosition(), cfgHvdcFreezing, cfgHvdcLimits,
                cfgSecondaryVc, olfGuardParams.getMinPlausibleTargetVoltage(), olfGuardParams.getMaxPlausibleTargetVoltage(),
                cfgVoltageMonitoring, cfgContinuousPhase, defaultLossDist && dsActive,
                olfGuardParams.isExtrapolateReactiveLimits(),
                olfGuardParams.isForceTargetQInReactiveLimits() && lfGuardParams.isUseReactiveLimits(),
                olfParams.getNewtonRaphsonStoppingCriteriaType()
                        == com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteriaType.PER_EQUATION_TYPE_CRITERIA
                        ? new BatchConfig.PerEqCriteria(olfParams.getNewtonRaphsonConvEpsPerEq(),
                                olfParams.getMaxActivePowerMismatch(), olfParams.getMaxReactivePowerMismatch(),
                                olfParams.getMaxVoltageMismatch(), olfParams.getMaxAngleMismatch(),
                                olfParams.getMaxRatioMismatch(), olfParams.getMaxSusceptanceMismatch())
                        : null,
                olfGuardParams.getGeneratorVoltageControlMinNominalVoltage(),
                // voltageRemoteControlRobustMode (OLF default TRUE): the RL kernel's realistic-V criteria.
                acParameters.isVoltageRemoteControlRobustMode(),
                acParameters.getMinRealisticVoltage(), acParameters.getMaxRealisticVoltage(),
                acParameters.getMinNominalVoltageRealisticVoltageCheck(),
                com.powsybl.openloadflow.ac.outerloop.gpu.DeviceLoopTables.realisticFixingLoop(acParameters),
                false,                                       // the robust retry pass derives its own config
                slackRelocation(acParameters),
                null,                                        // the precomputer picks the state sink per run
                GpuLoadFlowParameters.get(securityAnalysisParameters.getLoadFlowParameters()));
        return new Scope(batchDisabled, unsupportedOuterLoops, distributedSlack, reactiveLimits,
                slackDistTol, reactiveLimitTol, convEpsPerEq, batchConfig);
    }

    /** The run's slack re-pick for contingencies that isolate the slack, when the device can follow it: one slack
     *  bus, and the reference bus being that slack (what the security analysis configures). */
    private static SlackRelocation slackRelocation(AcLoadFlowParameters acParameters) {
        var np = acParameters.getNetworkParameters();
        if (np.getMaxSlackBusCount() != 1
                || !(np.getReferenceBusSelector() instanceof com.powsybl.openloadflow.network.ReferenceBusFirstSlackSelector)) {
            return null;
        }
        return new SlackRelocation(np.getSlackBusSelector(), 1);
    }

    /** The remote-group pin-dynamics residual gate — RETIRED 2026-09-08: singleton remote groups
     *  get a relocation table too (their only member's pin IS the all-pinned corner, handled by
     *  the pilot-vEq flip), so every clean group shape is representable. Kept as a method so a
     *  future unrepresentable shape has a place to land. */
    private static boolean remoteRlUnrepresentable(LfNetwork lfNetwork) {
        return false;
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
