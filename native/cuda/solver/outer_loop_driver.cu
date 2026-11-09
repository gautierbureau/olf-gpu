// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See outer_loop_driver.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_ac_solver_status.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"
#include "../common/codes/gpu_realistic_voltage_fixing_loop.cuh"
#include "../common/codes/gpu_transformer_voltage_control_mode.cuh"
#include "../diagnostics/driver_traces.cuh"
#include "../diagnostics/matrix_scan.cuh"
#include "../equations/distribution_fill.cuh"
#include "../equations/fill_launch.cuh"
#include "../equations/scatter.cuh"
#include "../equations/scenario_overrides.cuh"
#include "../outerloop/forced_target_q.cuh"
#include "../outerloop/pilot_voltage.cuh"
#include "../outerloop/reactive_capability_curve.cuh"
#include "../outerloop/reactive_capability_curves.cuh"
#include "../outerloop/reactive_limits_seed.cuh"
#include "../outerloop/sensitivity.cuh"
#include "../outerloop/slack_participants.cuh"
#include "../solver/batch_context.cuh"
#include "../solver/compact_solve.cuh"
#include "../solver/convergence.cuh"
#include "../solver/driver_pass.cuh"
#include "../solver/driver_steps.cuh"
#include "../solver/iterative_refinement.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "../solver/newton_iteration.cuh"
#include "../solver/outer_loop_check.cuh"
#include "../solver/outer_loop_sequence.cuh"
#include "../solver/realistic_voltage_check.cuh"
#include "../solver/scenario_install.cuh"
#include "../solver/solve_seed.cuh"
#include "../solver/state_vector_scaling.cuh"
#include "../fixedmatrix/zero_diagonal_repair.cuh"
#include "outer_loop_driver.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

// =================================================================================
// BATCHED N-1 security analysis: many post-contingency scenarios on the SAME network
// structure, solved in one batched Newton-Raphson. The CSR pattern, the family index
// packs, the destination arrays and cuDSS's SYMBOLIC ANALYSIS are SHARED across all
// scenarios (built once); only the value/state/mismatch arrays gain a scenario
// dimension. Per-scenario element disablement (a branch outage) is a device mask the
// fill kernels honor (a disabled element simply does not scatter — its entries stay
// structurally present but numerically zero, and the diagonal-pairing guarantee keeps
// every row factorizable). Converged scenarios retire from the batch (an active mask),
// so the per-iteration cost shrinks as scenarios finish.
//
// cuDSS: in the default UBATCH mode one per-block CSR whose scenarios share the rowPtr/colIdx pattern with
// per-scenario value sets. ANALYSIS runs ONCE per context on the shared pattern; each Newton iteration is ONE
// batched FACTORIZATION + ONE batched SOLVE over all scenarios. Retired scenarios (converged/diverged) are frozen
// to an identity system so a singular sub-block never fails the batch. The robust mode repoints a single matrix
// and factorizes per scenario. Either way the symbolic analysis is amortized over the entire contingency set.
//
// Orientation: J is true CSR with row = equation, col = variable (OLF's indices), so cuDSS solves J dx = -F
// directly. The destination arrays built with the structure fold each scatter output's ROLE in, so the fill kernels
// never index the generated scatter tables: dest >= 0 -> J values slot; dest == -1 -> skip; dest <= -2 -> mismatch
// row (-2 - dest).
// =================================================================================


BatchResult solveBatchContext(BatchSolverContext* ctx, const double* x0, const double* target,
                              const int* rowMode, const int* rowModeSlots, int nRowModeSlots,
                              const int* baseRowMode, const char* disabled,
                              int nTargetOverrides, const int* targetOverrideIndex, const double* targetOverrideValue,
                              const int* distributedSlackOverrideBus, const double* distributedSlackOverrideFactorDelta, const double* distributedSlackOverrideBaseTargetPDelta,
                              const double* distributedSlackOverrideMaxDelta, const double* distributedSlackOverrideMinDelta, const double* distributedSlackBusAnchorShift,
                              const double* slackBusAnchorShift,
                              int maxIter, double tol,
                              double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                              int stateVectorScalingMode, double maxDv, double maxDphi, const int* varType,
                              const double* x0Full, int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                              const int* reactiveLimitsSeedState,
                              int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues,
                              int maxOuterProductive, int maxPqPvSwitch, int rejectSlackDistributionFailure, int skipReactiveLimitsPhase,
                              int nBaseClosedZeroImpedanceBranches, int nZeroImpedanceBranchCloseOverrides, const int* zeroImpedanceBranchCloseScenarioElement,
                              int nZeroImpedanceBranchOpenOverrides, const int* zeroImpedanceBranchOpenScenarioElement,
                              int pauseAtEnded) {
    DriverPass drv(ctx->slotState);
    drv.n = ctx->n;
    drv.nnz = ctx->nnz;
    drv.nS = ctx->nScenarios;
    drv.skipReactiveLimitsPhase = skipReactiveLimitsPhase;
    drv.rejectSlackDistributionFailure = rejectSlackDistributionFailure != 0;
    const int& n = drv.n;
    const int& nS = drv.nS;
    // MAX_VOLTAGE_CHANGE (GpuStateVectorScalingMode::MAX_VOLTAGE_CHANGE): per-scenario step clamp, same rule as the base solver's
    // (varType = OLF's AcVariableType ordinal: BUS_V vs maxDv, BUS_PHI vs maxDphi; whole step scaled). Buffers allocated lazily.
    uploadBatchOverrides(ctx, stateVectorScalingMode, maxDv, maxDphi, varType, nShuntAdmittanceOverrides, shuntAdmittanceOverrideScenarioElement, shuntAdmittanceOverrideDelta,
                         nBaseClosedZeroImpedanceBranches, nZeroImpedanceBranchOpenOverrides, zeroImpedanceBranchOpenScenarioElement, nZeroImpedanceBranchCloseOverrides, zeroImpedanceBranchCloseScenarioElement, nClosedBranchParameterOverrides, closedBranchParameterOverrideScenarioElement, closedBranchParameterOverrideValues);
    drv.nTot = static_cast<size_t>(n) * nS;
    const size_t& nTot = drv.nTot;
    drv.tol = tol;
    drv.maxIter = maxIter;
    const long long newtonIterationsAtStart = gNewtonIters;
    drv.stateVectorScalingMode = stateVectorScalingMode;
    drv.convEpsPerEq = convEpsPerEq;
    // The caller refilling the bank asks the call to return once that many of its scenarios have ended, the others still
    // solving (0: when every scenario has ended); the next call goes on with them and the scenarios it installs.
    drv.pauseAtEnded = pauseAtEnded;
    // The scenarios of this solve into their slots (scenario_install.cuh): each starts from the base state, targets and
    // row modes, then takes its own. The row modes are PER-SCENARIO and mutated in place by the outer loops; x0 is either
    // the base or a FULL per-scenario state (the warm restart of a strategy pair re-solving from its parent's state).
    ScenarioInputs in;
    in.x0 = x0;
    in.target = target;
    in.baseRowMode = baseRowMode;
    in.rowMode = rowMode;
    in.rowModeSlots = rowModeSlots;
    in.nRowModeSlots = nRowModeSlots;
    in.disabled = disabled;
    in.nTargetOverrides = nTargetOverrides;
    in.targetOverrideIndex = targetOverrideIndex;
    in.targetOverrideValue = targetOverrideValue;
    in.x0Full = x0Full;
    // Per-scenario distributed-slack reweight (generator loss at a DS-participant bus): one affected DS bus per scenario
    // whose factor/base/headroom/footroom shrink by the lost gen's share, and a per-scenario factor-sum drop so the other
    // buses pick up that share.
    in.distributedSlackOverrideBus = distributedSlackOverrideBus;
    in.distributedSlackOverrideFactorDelta = distributedSlackOverrideFactorDelta;
    in.distributedSlackOverrideBaseTargetPDelta = distributedSlackOverrideBaseTargetPDelta;
    in.distributedSlackOverrideMaxDelta = distributedSlackOverrideMaxDelta;
    in.distributedSlackOverrideMinDelta = distributedSlackOverrideMinDelta;
    // Per-scenario DS ANCHOR SHIFT (pair/strategy legs): shift_i = parent's final target at participant i minus the
    // extraction-time base - re-anchors the water-fill's cumulative reference at the PARENT contingency's distribution,
    // mirroring OLF's setGeneratorsInitialTargetPToTargetP before a strategy re-solve. The slack's own target move,
    // independent of the participants' anchors: a strategy pair's parent share, and also a contingency's load lost at
    // the slack (a plain scenario, no anchor shift).
    in.distributedSlackBusAnchorShift = distributedSlackBusAnchorShift;
    in.slackBusAnchorShift = slackBusAnchorShift;
    resetInstalledEntries(ctx, ctx->outerLoopFirstSolveIterations, 0);
    resetInstalledEntries(ctx, ctx->outerLoopIterations, 0);
    resetInstalledEntries(ctx, ctx->outerLoopSolvePasses, 0);
    resetInstalledEntries(ctx, ctx->outerLoopChecks, 0);
    installScenarios(ctx, in);

    // a new chunk's scenarios hold no identity system yet (a lane its scenarios leave unused keeps whatever the last one
    // left there)
    CUDA_CHECK(cudaMemsetAsync(ctx->dIdentityLane.p, 0, static_cast<size_t>(nS) * sizeof(char), ctx->stream));
    // The installed scenarios start their solve (SlotSolveState): solving, live, MAX_ITERATION_REACHED until they converge
    // or fail, at the first loop of OLF's sequence.
    ctx->slotState.start(nS, installedSlots(ctx));
    if (ctx->batchedLu) {                                    // ...on the fast engine, whatever their slots held
        forgetLaneRouting(*ctx->batchedLu, installedSlots(ctx));
    }
    std::vector<char>& active = drv.active;

    BatchResult& res = drv.res;                         // res.x is sized when the states are copied back, at the end

    std::vector<double>& hF = drv.hF;

    // Device-side fixed-iteration mode: tol <= 0 means "run exactly maxIter Newton steps
    // on the device with NO per-iteration host sync and NO convergence retirement" (the
    // a fixed-iteration NR loop). tol > 0 keeps the per-iteration retirement loop.
    drv.deviceFixedIter = (tol <= 0.0);
    const bool& deviceFixedIter = drv.deviceFixedIter;
    if (deviceFixedIter) {
        hF.assign(nTot, 0.0);                            // the fixed-iteration mode reads the whole mismatch back once
    }
    // A scenario that exhausts the budget is reported as MAX_ITERATION_REACHED by the device itself (the
    // precomputer's exhausted set), which is OLF's own answer: no early exit hands stragglers to the CPU.

    // Full-loadflow mode (tol > 0) with local PV controllers runs the reactive-limits (PV→PQ) OUTER loop:
    // converge the inner Newton, switch any generator over its Q limit, re-converge the switched scenarios,
    // repeat until a pass switches nothing. Fixed-iter mode (tol <= 0, the throughput benchmark) and
    // family-free networks with no controllers do ONE pass (no outer loop) — byte-identical to before. The
    // distributed-slack and reactive-limit tables are packed whatever the run's loops: they count only for a run
    // that has the loop (distributedSlackOuterLoop / reactiveLimitsOuterLoop).
    bool fullLoadflow = !deviceFixedIter && ((ctx->reactiveLimitsOuterLoop && ctx->nReactiveLimitEntries > 0)
                                             || (ctx->distributedSlackOuterLoop && (ctx->nDistributedSlackBuses > 0 || ctx->slackOnlyDistribution))
                                             || ctx->nMonitoringVoltageControllers > 0 || ctx->nIncrementalTransformerVoltageControlControllerBranches > 0 || ctx->nIncrementalShuntVoltageControlControllerShunts > 0 || ctx->nIncrementalFlowControlControllerBranches > 0 || ctx->nPhaseControlControllerBranches > 0
                                             || ctx->nTransformerVoltageControlMembers > 0 || ctx->nShuntVoltageControlMembers > 0 || ctx->nSecondaryVoltageControlControllerBuses > 0
                                             || ctx->nOverloadManagementSystems > 0);
    const int MAX_OUTER_LOOP_ITERATIONS = maxOuterProductive > 0 ? maxOuterProductive : 20;   // OLF maxOuterLoopIterations
    // per-bus PV→PQ count cap (OLF maxPqPvSwitch): 0 is OLF's "never switch back" and is honoured as such;
    // only a negative value (no cap given) takes OLF's default of 3.
    drv.maxPqPvSwitch = maxPqPvSwitch >= 0 ? maxPqPvSwitch : 3;
    static const char* reactiveLimitsDecisionLogEnv = traceValue(Trace::SW_LOG);
    drv.reactiveLimitsDecisionLogScenario = reactiveLimitsDecisionLogEnv == nullptr ? -1 : (std::string(reactiveLimitsDecisionLogEnv) == "all" ? -2 : std::atoi(reactiveLimitsDecisionLogEnv));
    // Outer-loop convergence tolerances (pu) — passed in so the device uses the SAME criteria as the CPU
    // loadflow that built the base case (maxReactivePowerMismatch / slackBusPMaxMismatch, per-unitized), NOT
    // hardcoded constants. With matching parameters and the same equations the GPU outer-loop fixed point
    // matches the CPU's to the inner-Newton precision; a loose tolerance (e.g. the 1 MW slack default) leaves
    // a redistribution band, so a test wanting machine-precision agreement passes tight values on both sides.
    drv.reactiveLimitTol = reactiveLimitTol;
    drv.slackDistTol = slackDistTol;
    drv.pResidueEps = 1e-5;                                  // OLF ActivePowerDistribution.P_RESIDUE_EPS: the
                                                            // water-filling distributes the mismatch down to this
    // MAX_OUTER_LOOP_ITERATIONS bounds the PRODUCTIVE outer iterations (an outer loop that re-solved), matching OLF's
    // outerLoopTotalIterations budget. The raw pass loop gets a generous safety bound on top, because the
    // sequential driver also spends non-productive passes (phase transitions / stability re-checks) that OLF
    // does not count.
    // The reactive-limit limits become PER-SCENARIO as soon as either channel is live: curve limits at
    // the runtime P, or a generator lost at a bus that keeps voltage control taking its capability away.
    // A lost voltage-controlling SVC also takes its share of the entry's quadratic limit coefficients (an SVC's
    // limit is its susceptance times v^2): per (scenario, entry) coefficients, read by the reactive-limits check
    // in place of the run-constant ones. Built only when some override carries such a share.
    const SlotTable& reactiveLimitDeltaOverrides = ctx->reactiveLimitDeltaOverrides;
    bool& hasPerScenarioSvcSusceptanceLimits = drv.hasPerScenarioSvcSusceptanceLimits;
    hasPerScenarioSvcSusceptanceLimits = false;
    for (int t = 0; t < nS && !hasPerScenarioSvcSusceptanceLimits; t++) {
        for (int k = 0; k < reactiveLimitDeltaOverrides.countOf(t) && !hasPerScenarioSvcSusceptanceLimits; k++) {
            hasPerScenarioSvcSusceptanceLimits = reactiveLimitDeltaOverrides.doubleOf(t, k, 3) != 0.0 || reactiveLimitDeltaOverrides.doubleOf(t, k, 4) != 0.0;
        }
    }
    if (hasPerScenarioSvcSusceptanceLimits && ctx->nReactiveLimitEntries > 0) {
        size_t nReactiveLimitEntries = static_cast<size_t>(ctx->nReactiveLimitEntries);
        std::vector<double> bmin(nReactiveLimitEntries), bmax(nReactiveLimitEntries);
        CUDA_CHECK(cudaMemcpy(bmin.data(), ctx->dReactiveLimitSvcMinB.p, nReactiveLimitEntries * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bmax.data(), ctx->dReactiveLimitSvcMaxB.p, nReactiveLimitEntries * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> dynMin(static_cast<size_t>(nS) * nReactiveLimitEntries), dynMax(static_cast<size_t>(nS) * nReactiveLimitEntries);
        for (int t = 0; t < nS; t++) {
            std::copy(bmin.begin(), bmin.end(), dynMin.begin() + static_cast<size_t>(t) * nReactiveLimitEntries);
            std::copy(bmax.begin(), bmax.end(), dynMax.begin() + static_cast<size_t>(t) * nReactiveLimitEntries);
        }
        for (int t = 0; t < nS; t++) {
            for (int k = 0; k < reactiveLimitDeltaOverrides.countOf(t); k++) {
                size_t at = static_cast<size_t>(t) * nReactiveLimitEntries + reactiveLimitDeltaOverrides.intOf(t, k);
                dynMin[at] += reactiveLimitDeltaOverrides.doubleOf(t, k, 3);
                dynMax[at] += reactiveLimitDeltaOverrides.doubleOf(t, k, 4);
            }
        }
        if (!ctx->dReactiveLimitSvcMinBPerScenario.p) {
            ctx->dReactiveLimitSvcMinBPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * nReactiveLimitEntries * sizeof(double));
            ctx->dReactiveLimitSvcMaxBPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * nReactiveLimitEntries * sizeof(double));
        }
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitSvcMinBPerScenario.p, dynMin.data(), dynMin.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitSvcMaxBPerScenario.p, dynMax.data(), dynMax.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    drv.usePerScenarioReactiveLimits = ctx->nReactiveLimitEntries > 0 && (ctx->nReactiveCapabilityCurves > 0 || reactiveLimitDeltaOverrides.entries() > 0 || ctx->nForcedTargetQLimitEntries > 0);
    // The per-scenario limits exist, and the seed computes each installed scenario's (solve_seed.cu), whether or not this
    // call's scenarios read them: an install whose overrides turn usePerScenarioReactiveLimits on, while scenarios of
    // earlier calls still solve, has them read theirs. Curves allocate these on their own table upload; a Q override
    // alone must too, and a guard that names only ONE of them silently leaves the others null on a curve network
    // (measured: the whole precompute fell back to the CPU).
    if (ctx->nReactiveLimitEntries > 0 && (!ctx->dReactiveLimitMinQPerScenario.p || !ctx->dReactiveLimitLoadQPerScenario.p)) {
        ctx->dReactiveLimitLoadQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitMinQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitMaxQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries * sizeof(double));
    }
    int maxOuter = fullLoadflow ? 8 * MAX_OUTER_LOOP_ITERATIONS : 1;
    // The two kernels that lay a group's rows out from the CURRENT PV/PQ states (the RL pass runs them after
    // its switches; the seeding below runs them once so the FIRST fill sees the seeded PV -> PQ switches on the rows).
    // `staticLimits`: use the STATIC reactive-limit arrays rather than the curve-evaluated ones. At SEED time the
    // dynamic arrays are still zero (reactiveCapabilityCurveLimitsBatch runs in the reactive-limits phase), so a seed-time flip
    // stamped the pilot's own-Q target as 0 instead of (limit - loadQ) and the whole slack phase ran against it —
    // measured on rte6515 BUS-6112_BBS_2: row 12834 converged at target 0 through outer 32, then the RL phase
    // re-stamped -0.444 and the solve had to catch up, one pass behind OLF. The re-stamp branch in the kernel
    // keeps the value current once the curve limits exist, so the static value is a correct starting point.
    std::vector<char>& outerLoopCheckActive = drv.outerLoopCheckActive;
    outerLoopCheckActive.assign(nS, 0);
    // OLF_GPU_RL_AUDIT: per-scenario decision history, kept in memory and printed only for the
    // scenarios the end-of-solve audit flags. The failing scenario is not known until the solve is
    // over and the flip is rare, so the only way to see its trajectory is to record every scenario's
    // and throw away the rest.
    drv.reactiveLimitsAuditOn = traceOn(Trace::RL_AUDIT);
    drv.reactiveLimitsAuditLog.assign(drv.reactiveLimitsAuditOn ? nS : 0, {});
    if (traceOn(Trace::BATCH_TRACE) && nS == 1) {
        std::fprintf(stderr, "CTX_FLAGS nS=1 nReactiveLimitEntries=%d nReactiveLimitRemoteEntries=%d hasDistrQRelocationGroups=%d nReactiveRelocationGroups=%d nDistrQ=%d nDistributedSlackBuses=%d nDistributedSlackGenerators=%d nZeroImpedanceBranch=%d nBaseClosedZeroImpedanceBranches=%d nRemoteMemberReactiveTerms=%d fullLoadflow=%d\n",
                     ctx->nReactiveLimitEntries, ctx->nReactiveLimitRemoteEntries, ctx->hasDistrQRelocationGroups ? 1 : 0, ctx->structure->nReactiveRelocationGroups, ctx->nDistrQ, ctx->nDistributedSlackBuses, ctx->nDistributedSlackGenerators, ctx->nZeroImpedanceBranch, ctx->nBaseClosedZeroImpedanceBranches, ctx->nRemoteMemberReactiveTerms, fullLoadflow ? 1 : 0);
    }
    if (fullLoadflow && ctx->nSecondaryVoltageControlControllerBuses > 0) {
        // BEFORE the reactive-limit seeding below, which consults it: no zone has moved a target yet.
        resetInstalledRows(ctx, ctx->dSecondaryVoltageControlReactiveLimitTargetV.as<double>(),
                           static_cast<size_t>(std::max(ctx->nReactiveLimitEntries, 1)), nullptr, std::nan(""));
    }
    // no inner solve has converged yet: the start
    copyInstalledRows(ctx, ctx->dXLastConv.as<double>(), ctx->dX.as<double>(), static_cast<size_t>(n));
    if (fullLoadflow) {                                      // per-solve PV↔PQ state: 0 switches, limitType from base
        seedSolve(ctx, drv, reactiveLimitsSeedState);
    }

    // SEQUENTIAL outer loops, exactly like OLF's AcloadFlowEngine: fully stabilize the reactive-limits loop
    // (switch + re-solve until it switches nothing), THEN fully stabilize the distributed-slack loop
    // (distribute + re-solve until balanced), and repeat the whole RL->DS sequence until a full sequence
    // triggers no inner Newton re-solve. Running RL and DS in the SAME pass converges to a different (still
    // in-tolerance) point when switching interacts with slack distribution. phase 0 = reactive limits, 1 = slack.
    // OLF's DefaultAcOuterLoopConfig orders DistributedSlack BEFORE ReactiveLimits (innermost first), and the
    // engine stabilizes them in that order. Match it: phase 0 = distributed slack, phase 1 = reactive limits.
    // (Switching PV->PQ before the slack is balanced leaves a large active imbalance that the undamped device
    // Newton diverges on — DS must stabilize first.)
    // The outer loops this solve drives, IN OLF'S CONFIGURED ORDER. OLF's engine
    // (AcloadFlowEngine.runOuterLoops) walks its configured list in order, drives each loop to
    // STABLE — re-solving after every unstable check — then moves to the next, restarts from the
    // top, and stops when a full sweep changes nothing. The driver below already has exactly those
    // semantics; what it lacked was the LIST: it hard-coded two loops, so every other loop had to be
    // bolted on as a host orchestrator running AFTER the whole solve, landing out of order. Naming
    // the order explicitly is what lets the remaining loops move into their proper slot.
    //
    // OLF's contingency order: DistributedSlack(1), FreezingHvdcACEmulation(2),
    // AcHvdcAcEmulationLimits(3), SecondaryVoltageControl(5), MonitoringVoltage(6),
    // ReactiveLimits(7). All but SecondaryVoltageControl are driver phases here, in that order.
    // The loops OLF runs AFTER ReactiveLimits (PhaseControl, transformer and shunt voltage control)
    // are still host orchestrators: their position relative to RL is right, but OLF restarts the
    // WHOLE sweep whenever one of them changes something, so an earlier loop is re-checked and here
    // it is not. Folding them needs the discrete tap/section tables on the device.
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    int& nLoops = drv.nLoops;
    nLoops = 0;
    // OLF drops the DistributedSlack loop from a run with AreaInterchangeControl, which runs after the HVDC loops instead
    bool distributionInLoopOrder = ctx->distributedSlackOuterLoop && (ctx->nDistributedSlackBuses > 0 || ctx->slackOnlyDistribution);
    if (distributionInLoopOrder && !ctx->areaInterchangeControlOuterLoop) {
        loopOrder[nLoops++] = GpuOuterLoopKind::DISTRIBUTED_SLACK;
    }
    if (ctx->nHvdcAcEmulation > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION;  // OLF: FreezingHvdcACEmulation
        if (ctx->hvdcAcEmulationLimitsActive) {
            loopOrder[nLoops++] = GpuOuterLoopKind::HVDC_AC_EMULATION_LIMITS;  // OLF: AcHvdcAcEmulationLimits
        }
    }
    if (distributionInLoopOrder && ctx->areaInterchangeControlOuterLoop) {
        loopOrder[nLoops++] = GpuOuterLoopKind::AREA_INTERCHANGE_CONTROL;  // OLF order 3: AreaInterchangeControl
    }
    if (ctx->nSecondaryVoltageControlControllerBuses > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::SECONDARY_VOLTAGE_CONTROL;  // OLF order 4: SecondaryVoltageControl
    }
    if (ctx->nMonitoringVoltageControllers > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::MONITORING_VOLTAGE;  // OLF order 5: MonitoringVoltage
    }
    // the run's loop, not the table alone: the table also holds the stand-by SVCs MonitoringVoltage activates
    if (ctx->reactiveLimitsOuterLoop && ctx->nReactiveLimitEntries > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::REACTIVE_LIMITS;  // OLF order 6: ReactiveLimits
    }
    if (ctx->nIncrementalPhaseControlControllerBranches > 0 || ctx->nPhaseControlControllerBranches > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::INCREMENTAL_PHASE_CONTROL;  // OLF order 7: (Incremental)PhaseControl, IN-KERNEL
    }
    if (ctx->nIncrementalTransformerVoltageControlControllerBranches > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL;  // OLF order 8: TransformerVoltageControl, IN-KERNEL
    } else if (ctx->nTransformerVoltageControlMembers > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL;  // OLF order 8: AFTER_GENERATOR / Simple TVC, IN-KERNEL
    }
    if (ctx->nIncrementalTransformerReactivePowerControlControllerBranches > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL;  // OLF order 9: TransformerReactivePowerControl, IN-KERNEL
    }
    if (ctx->nIncrementalShuntVoltageControlControllerShunts > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL;  // OLF order 10: ShuntVoltageControl, IN-KERNEL
    } else if (ctx->nShuntVoltageControlMembers > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL;  // OLF order 10: WITH_GENERATOR shunt VC, IN-KERNEL
    }
    if (ctx->nOverloadManagementSystems > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::AUTOMATION_SYSTEM;  // OLF's last loop: AutomationSystem
    }
    if (nLoops == 0) {
        // nothing to drive: one slot so the sweep runs its first solve. Its check is a ReactiveLimits one only
        // when the run has that loop (reactiveLimitsCheck tests reactiveLimitsOuterLoop); otherwise it is STABLE.
        loopOrder[nLoops++] = GpuOuterLoopKind::REACTIVE_LIMITS;
    }
    // OLF's AcloadFlowEngine.runOuterLoops, scenario by scenario, inside the batched Newton loop (outer_loop_sequence.cuh):
    // a scenario whose solve converges is checked at once along its own loops, and an UNSTABLE check restarts its Newton
    // while its chunk mates go on with theirs. The batch used to solve in passes (every scenario with a solve due, then
    // the checks of all of them), so each pass lasted as long as its slowest scenario's solve.
    // STABLE never re-solves: OLF's runOuterLoop re-solves only on an UNSTABLE check, and the device Newton forces >= 1
    // iteration per solve (matching OLF's per-run behavior), so a no-op solve would move a converged state off the
    // CPU's by about the stopping residual (measured 4e-4 kV on the DS->RL transition with a distributed base).
    OuterLoopSequence sequence(ctx->slotState);
    initOuterLoopSequence(sequence, ctx, drv, fullLoadflow, MAX_OUTER_LOOP_ITERATIONS);
    drv.loopIdx = 0;
    // Each scenario's work is bounded: its solves (its outer-loop budget of productive ones, and the sequence's others)
    // of at most maxIter + 1 iterations each. The cap only stops a loop that would never end.
    drv.maxBatchedIterations = maxOuter * (maxIter + 1);
    // OLF_GPU_SLOT_DUMP=<prefix>: the traced scenario's (OLF_GPU_LOOP_TRACE_CTG) row targets, initial state and row
    // modes as the driver starts - after every host override and every device seed - one file per solve
    // (<prefix>.<k>), "<row> <target> <x> <mode>" with every double exact. Two runs of one scenario that should be
    // identical are diffed row by row: the rows that differ name the writer.
    if (ctx->loopTraceSlot >= 0 && ctx->loopTraceSlot < nS) {
        static const char* slotDump = traceValue(Trace::SLOT_DUMP);
        static int slotDumps = 0;
        if (slotDump != nullptr) {
            const size_t off = static_cast<size_t>(ctx->loopTraceSlot) * n;
            std::vector<double> ht(n), hx(n);
            std::vector<int> hm(n);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(ht.data(), ctx->dTarget.as<double>() + off, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hx.data(), ctx->dX.as<double>() + off, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hm.data(), ctx->dMode.as<int>() + off, n * sizeof(int), cudaMemcpyDeviceToHost));
            std::string path = std::string(slotDump) + "." + std::to_string(slotDumps++);
            if (FILE* f = std::fopen(path.c_str(), "w")) {
                for (int r = 0; r < n; r++) {
                    std::fprintf(f, "%d %.17g %.17g %d\n", r, ht[r], hx[r], hm[r]);
                }
                std::fclose(f);
            }
        }
    }
    static const bool outerLoopTimingRequested = traceOn(Trace::PHASE_TIME);
    drv.outerLoopTiming = outerLoopTimingRequested;
    drv.loopRejected.assign(nS, 0);                         // a device loop routed the scenario to the CPU (checkOuterLoop)
    gNewtonLoops++;
    if (drv.outerLoopTiming) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    const double newtonStartMilliseconds = drv.outerLoopTiming ? nowMilliseconds() : 0.0;
    const long long newtonStartWidthIterations = gWidthIterSum;
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    newtonSolve(ctx, drv, sequence);
    if (drv.outerLoopTiming) {
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        gNewtonLoopMilliseconds += nowMilliseconds() - newtonStartMilliseconds;
        gNewtonLoopWidthIterations += gWidthIterSum - newtonStartWidthIterations;
    }

    traceSwitchLog(ctx, drv);
    // ---- OLF_GPU_RL_AUDIT=<pu threshold>: audit the FINAL state against the reactive-limit rule
    //      itself. OLF's outer loop ends only after a check found NOTHING to switch, so in OLF's
    //      answer every controller still PV sits inside its reactive limits (within the switch
    //      tolerance). A device scenario that returns with a PV controller beyond its limit
    //      did not reach that fixpoint - it stopped one check short - and that is a defect no
    //      comparison with the CPU is needed to see. Re-fills the diagnostic q at the final state
    //      for every live scenario, refreshes the curve limits at the final targets, and prints one
    //      line per violating (scenario, entry).
    if (drv.nActive == 0) {                                  // it recomputes limits: not while scenarios still solve
        traceReactiveLimitsAudit(ctx, drv, drv.reactiveLimitsAuditLog);
    }
    tracePhaseTime(ctx, drv);
    // The scenarios that ended - every one, or those that had when the call paused for a refill - are what the call hands
    // back: they become the selection, so the result rows and every readback return their rows, in the order they were
    // installed. A paused call has at least pauseAtEnded of them, a finished one every scenario it held.
    std::vector<int> ended = ctx->slotState.handBack();
    bool wholeBank = static_cast<int>(ended.size()) == nS;
    for (int j = 0; wholeBank && j < nS; j++) {
        wholeBank = ended[j] == j;
    }
    ctx->installSlots = wholeBank ? std::vector<int>() : std::move(ended);
    ctx->newtonLoopIterations = static_cast<int>(gNewtonIters - newtonIterationsAtStart);
    // Only the handed-back scenarios' states reach the host: they are the result rows. A call paused for a refill hands
    // back a few of the bank's slots, whose states are gathered on the device and copied alone; a call handing back more
    // than half the bank copies the whole state, which needs no device scratch as large as the state.
    const int handedBack = wholeBank ? nS : static_cast<int>(ctx->installSlots.size());
    const bool bySlot = wholeBank || 2 * handedBack > nS;
    if (bySlot) {
        res.x.resize(nTot);
        CUDA_CHECK(cudaMemcpy(res.x.data(), ctx->dX.p, nTot * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        const size_t gathered = static_cast<size_t>(handedBack) * n;
        res.x.resize(gathered);
        ctx->dHandBackSlots.ensure(static_cast<size_t>(handedBack) * sizeof(int));
        ctx->dHandBackStates.ensure(gathered * sizeof(double));
        CUDA_CHECK(cudaMemcpyAsync(ctx->dHandBackSlots.p, ctx->installSlots.data(), static_cast<size_t>(handedBack) * sizeof(int),
                                   cudaMemcpyHostToDevice, ctx->stream));
        gatherScenarios<<<grid(static_cast<int>(gathered)), BLOCK, 0, ctx->stream>>>(n, handedBack, ctx->dHandBackSlots.as<int>(),
                ctx->dX.as<double>(), ctx->dHandBackStates.as<double>());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(res.x.data(), ctx->dHandBackStates.p, gathered * sizeof(double), cudaMemcpyDeviceToHost,
                                   ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }
    // The state is handed over, not copied (the next call sizes it again); the per-slot status, iterations and norms are
    // copied: they stay on the context, where SlotSolveState::start reads them for every slot.
    BatchResult out;
    out.x = std::move(res.x);
    out.statesBySlot = bySlot;
    out.iters = res.iters;
    out.norm = res.norm;
    out.status = res.status;
    return out;
}

}  // namespace olfgpu
