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
#include "../common/codes/gpu_reactive_limits_state.cuh"
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
#include "../solver/realistic_voltage_check.cuh"
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


// Tile the base row (row 0, already in d[0..n)) across all nS scenarios: d[s*n+i] = d[i] for
// s in 1..nS-1. The base state/targets/row-mode are identical for every contingency, so the host
// uploads ONE n-row and this replicates it on-device — no 32x JNI marshalling / H2D of the base.
template <typename T>
__global__ void tileBaseRow(int total, int n, T* d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n && idx < total) {                       // rows 1..nS-1 copy from row 0; row 0 is the source
        d[idx] = d[idx % n];
    }
}

BatchResult solveBatchContext(BatchSolverContext* ctx, const double* x0, const double* target,
                              const int* rowMode, const int* rowModeSlots, int nRowModeSlots,
                              const int* baseRowMode, const char* disabled,
                              int nTargetOverrides, const int* targetOverrideIndex, const double* targetOverrideValue,
                              const int* distributedSlackOverrideBus, const double* distributedSlackOverrideFactorDelta, const double* distributedSlackOverrideBaseTargetPDelta,
                              const double* distributedSlackOverrideMaxDelta, const double* distributedSlackOverrideMinDelta, const double* distributedSlackBusAnchorShift,
                              const double* slackBusAnchorShift,
                              int nReactiveLimitTargetVOverrides, const int* reactiveLimitTargetVOverrideScenarioEntry, const double* reactiveLimitTargetVOverrideValue,
                              int nReactiveLimitDeltaOverrides, const int* reactiveLimitDeltaOverrideScenarioEntry, const double* reactiveLimitDeltaOverrideValues,
                              int nReactiveCapabilityCurveOverrides, const int* reactiveCapabilityCurveOverrideScenarioCurve, const double* reactiveCapabilityCurveOverrideStepScale,
                              int maxIter, double tol,
                              double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                              int stateVectorScalingMode, double maxDv, double maxDphi, const int* varType,
                              const double* x0Full, int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                              const int* reactiveLimitsSeedState, int* reactiveLimitsFinalState,
                              int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues,
                              int maxOuterProductive, int maxPqPvSwitch, int rejectSlackDistributionFailure, int skipReactiveLimitsPhase,
                              int nBaseClosedZeroImpedanceBranches, int nZeroImpedanceBranchCloseOverrides, const int* zeroImpedanceBranchCloseScenarioElement,
                              int nZeroImpedanceBranchOpenOverrides, const int* zeroImpedanceBranchOpenScenarioElement) {
    DriverPass drv;
    drv.n = ctx->n;
    drv.nnz = ctx->nnz;
    drv.nS = ctx->nScenarios;
    drv.nReactiveLimitTargetVOverrides = nReactiveLimitTargetVOverrides;
    drv.nReactiveLimitDeltaOverrides = nReactiveLimitDeltaOverrides;
    drv.nReactiveCapabilityCurveOverrides = nReactiveCapabilityCurveOverrides;
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
    drv.stateVectorScalingMode = stateVectorScalingMode;
    drv.convEpsPerEq = convEpsPerEq;
    DeviceBuf dTargetOverrideIndex, dTargetOverrideValue;                            // function-scoped: outlive the async override kernel
    DeviceBuf& dBaseMode = drv.dBaseMode;                   // the contingency-free base row mode (length n)
    DeviceBuf dModeSlots, dModeRows;                         // compact row modes: the scenarios that differ from the base
    DeviceBuf dX0SeedIdx, dX0SeedVal;                        // setBatchX0Seeds, function-scoped like the overrides
    // Per-scenario distributed-slack reweight (generator loss at a DS-participant bus): one affected DS bus per
    // scenario (distributedSlackOverrideBus[s], -1 if none) whose factor/base/headroom/footroom shrink by the lost gen's share, and
    // a per-scenario factor-sum drop so the other buses pick up that share. nullptr when no DS scenarios.
    DeviceBuf& dDistributedSlackOverrideBus = drv.dDistributedSlackOverrideBus;
    DeviceBuf& dDistributedSlackOverrideFactorDelta = drv.dDistributedSlackOverrideFactorDelta;
    DeviceBuf& dDistributedSlackOverrideBaseTargetPDelta = drv.dDistributedSlackOverrideBaseTargetPDelta;
    DeviceBuf& dDistributedSlackOverrideMaxDelta = drv.dDistributedSlackOverrideMaxDelta;
    DeviceBuf& dDistributedSlackOverrideMinDelta = drv.dDistributedSlackOverrideMinDelta;
    drv.hasDistributedSlackOverride = (distributedSlackOverrideBus != nullptr && ctx->nDistributedSlackBuses > 0);
    const bool& hasDistributedSlackOverride = drv.hasDistributedSlackOverride;
    // Per-scenario DS ANCHOR SHIFT (pair/strategy legs): shift_i = parent's final target at participant i
    // minus the extraction-time base — re-anchors the water-fill's cumulative reference at the PARENT
    // contingency's distribution, mirroring OLF's setGeneratorsInitialTargetPToTargetP before a strategy
    // re-solve. nullptr = anchored at the base (plain contingencies).
    DeviceBuf& dReactiveLimitTargetVOverrideScenarioEntry = drv.dReactiveLimitTargetVOverrideScenarioEntry;
    DeviceBuf& dReactiveLimitTargetVOverrideValue = drv.dReactiveLimitTargetVOverrideValue;
    if (nReactiveLimitTargetVOverrides > 0) {
        dReactiveLimitTargetVOverrideScenarioEntry.upload(reactiveLimitTargetVOverrideScenarioEntry, 2L * nReactiveLimitTargetVOverrides * sizeof(int));
        dReactiveLimitTargetVOverrideValue.upload(reactiveLimitTargetVOverrideValue, (size_t) nReactiveLimitTargetVOverrides * sizeof(double));
    }
    DeviceBuf& dDistributedSlackBusAnchorShift = drv.dDistributedSlackBusAnchorShift;
    DeviceBuf& dSlackBusAnchorShift = drv.dSlackBusAnchorShift;
    drv.hasDistributedSlackBusAnchorShift = (distributedSlackBusAnchorShift != nullptr && ctx->nDistributedSlackBuses > 0);
    const bool& hasDistributedSlackBusAnchorShift = drv.hasDistributedSlackBusAnchorShift;
    if (hasDistributedSlackBusAnchorShift) {
        dDistributedSlackBusAnchorShift.upload(distributedSlackBusAnchorShift, static_cast<size_t>(nS) * ctx->nDistributedSlackBuses * sizeof(double));
    }
    // The slack's own target move, independent of the participants' anchors: a strategy pair's parent share,
    // and also a contingency's load lost at the slack (a plain scenario, no anchor shift).
    drv.hasSlackBusAnchorShift = (slackBusAnchorShift != nullptr && ctx->nDistributedSlackBuses > 0);
    const bool& hasSlackBusAnchorShift = drv.hasSlackBusAnchorShift;
    if (hasSlackBusAnchorShift) {
        dSlackBusAnchorShift.upload(slackBusAnchorShift, static_cast<size_t>(nS) * sizeof(double));
    }

    // x0/target are the BASE row (length n), identical for every scenario: upload once and tile across
    // nS on-device. rowMode is PER-SCENARIO (length n*nS): a contingency can freeze islanded buses
    // (Phase 0b) or flip PV↔PQ via reactive limits (Phase 1), so it is NOT tiled — uploaded as-is and
    // mutated in place on-device by the outer-loop kernels. Only the disabled mask is also per-scenario.
    // x0: either the BASE row tiled across nS (default), or a FULL per-scenario n*nS state
    // (x0Full != null) — the warm-restart hook for multi-call outer sequencing (e.g. the
    // transformer voltage-control orchestrator re-solving scenarios from their previous states).
    ctx->outerLoopFirstSolveIterations.assign(nS, 0);
    ctx->outerLoopIterations.assign(nS, 0);
    ctx->outerLoopSolvePasses.assign(nS, 0);
    dBaseMode.upload(baseRowMode, n * sizeof(int));         // base (contingency-free) row mode for the RL limit-type seed
    if (x0Full != nullptr) {
        CUDA_CHECK(cudaMemcpy(ctx->dX.p, x0Full, (size_t) nTot * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(ctx->dX.p, x0, n * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(ctx->dTarget.p, target, n * sizeof(double), cudaMemcpyHostToDevice));
    // The BASE targets, kept aside before tiling and the per-scenario overrides: the pilot-Q
    // capture below reads the base carrier's row, and a scenario may have overridden that row
    // (a member islanded by a bus loss freezes it to a flat voltage identity). Capturing from the
    // base keeps the captured value the pilot's reactive target whatever the scenario did to the
    // row. (The carrier's target itself is stamped host-side when the base carrier is frozen -
    // GpuBatchChunkSolver - because the device stamp runs only inside the outer-loop passes.)
    if (!ctx->dTargetBase.p) ctx->dTargetBase.alloc((size_t) n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(ctx->dTargetBase.p, target, n * sizeof(double), cudaMemcpyHostToDevice));
    if (rowModeSlots == nullptr) {
        CUDA_CHECK(cudaMemcpy(ctx->dMode.p, rowMode, nTot * sizeof(int), cudaMemcpyHostToDevice));
    } else {
        // COMPACT: every scenario starts from the base row mode, then only those that differ (islanded buses,
        // switch outages, ...) get their own row — a few rows instead of nS * n ints through PCIe.
        CUDA_CHECK(cudaMemcpy(ctx->dMode.p, baseRowMode, n * sizeof(int), cudaMemcpyHostToDevice));
        tileBaseRow<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, n, ctx->dMode.as<int>());
        if (nRowModeSlots > 0) {
            dModeSlots.upload(rowModeSlots, nRowModeSlots * sizeof(int));
            dModeRows.upload(rowMode, (size_t) nRowModeSlots * n * sizeof(int));
            scatterRowsInt<<<grid(nRowModeSlots * n), BLOCK, 0, ctx->stream>>>(nRowModeSlots, n,
                    dModeSlots.as<int>(), dModeRows.as<int>(), ctx->dMode.as<int>());
        }
    }
    if (x0Full == nullptr) {
        tileBaseRow<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, n, ctx->dX.as<double>());
        int nSeeds = static_cast<int>(ctx->x0SeedIdx.size());
        if (nSeeds > 0) {                                // a few per-scenario starting values over the base
            dX0SeedIdx.upload(ctx->x0SeedIdx.data(), nSeeds * sizeof(int));
            dX0SeedVal.upload(ctx->x0SeedVal.data(), nSeeds * sizeof(double));
            applyTargetOverrideBatch<<<grid(nSeeds), BLOCK, 0, ctx->stream>>>(nSeeds, dX0SeedIdx.as<int>(),
                    dX0SeedVal.as<double>(), ctx->dX.as<double>());
        }
    }
    ctx->x0SeedIdx.clear();                              // consumed by this solve either way
    ctx->x0SeedVal.clear();
    tileBaseRow<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, n, ctx->dTarget.as<double>());
    if (nTargetOverrides > 0) {                                       // per-scenario target overrides (generator loss)
        dTargetOverrideIndex.upload(targetOverrideIndex, nTargetOverrides * sizeof(int));
        dTargetOverrideValue.upload(targetOverrideValue, nTargetOverrides * sizeof(double));
        applyTargetOverrideBatch<<<grid(nTargetOverrides), BLOCK, 0, ctx->stream>>>(nTargetOverrides, dTargetOverrideIndex.as<int>(), dTargetOverrideValue.as<double>(), ctx->dTarget.as<double>());
    }
    if (hasDistributedSlackOverride) {                                     // per-scenario generator-loss DS reweight
        dDistributedSlackOverrideBus.upload(distributedSlackOverrideBus, nS * sizeof(int));
        dDistributedSlackOverrideFactorDelta.upload(distributedSlackOverrideFactorDelta, nS * sizeof(double));
        dDistributedSlackOverrideBaseTargetPDelta.upload(distributedSlackOverrideBaseTargetPDelta, nS * sizeof(double));
        dDistributedSlackOverrideMaxDelta.upload(distributedSlackOverrideMaxDelta, nS * sizeof(double));
        dDistributedSlackOverrideMinDelta.upload(distributedSlackOverrideMinDelta, nS * sizeof(double));
    }
    CUDA_CHECK(cudaMemcpy(ctx->dDisabled.p, disabled,
                          static_cast<size_t>(ctx->nElements) * nS * sizeof(char), cudaMemcpyHostToDevice));

    // a new chunk's scenarios hold no identity system yet (a lane its scenarios leave unused keeps whatever the last one
    // left there)
    CUDA_CHECK(cudaMemsetAsync(ctx->dIdentityLane.p, 0, static_cast<size_t>(nS) * sizeof(char), ctx->stream));
    std::vector<char>& active = drv.active;
    active.assign(nS, 1);

    BatchResult& res = drv.res;
    res.x.resize(nTot);
    res.iters.assign(nS, 0);
    res.norm.assign(nS, 0.0);
    res.status.assign(nS, GpuAcSolverStatus::MAX_ITERATION_REACHED);  // default (overwritten on converge/diverge)

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
    DeviceBuf& dReactiveLimitDeltaOverrideScenarioEntry = drv.dReactiveLimitDeltaOverrideScenarioEntry;
    DeviceBuf& dReactiveLimitDeltaOverrideValues = drv.dReactiveLimitDeltaOverrideValues;
    DeviceBuf& dReactiveLimitDeltaOverrideStart = drv.dReactiveLimitDeltaOverrideStart;
    DeviceBuf& dReactiveLimitDeltaOverrideList = drv.dReactiveLimitDeltaOverrideList;
    if (nReactiveLimitDeltaOverrides > 0) {
        dReactiveLimitDeltaOverrideScenarioEntry.upload(reactiveLimitDeltaOverrideScenarioEntry, static_cast<size_t>(nReactiveLimitDeltaOverrides) * 2 * sizeof(int));
        dReactiveLimitDeltaOverrideValues.upload(reactiveLimitDeltaOverrideValues, static_cast<size_t>(nReactiveLimitDeltaOverrides) * 5 * sizeof(double));   // {dMinQ, dMaxQ, dLoadQ, dBmin, dBmax}
        // ...indexed by slot (CSR, entry order kept), so the dynamic-limits kernel reads its scenario's own
        // overrides instead of scanning the chunk's.
        std::vector<int> qPtr(nS + 1, 0);
        std::vector<int> qList(nReactiveLimitDeltaOverrides);
        for (int k = 0; k < nReactiveLimitDeltaOverrides; k++) qPtr[reactiveLimitDeltaOverrideScenarioEntry[2 * k] + 1]++;
        for (int t = 0; t < nS; t++) qPtr[t + 1] += qPtr[t];
        std::vector<int> qCursor(qPtr.begin(), qPtr.end() - 1);
        for (int k = 0; k < nReactiveLimitDeltaOverrides; k++) qList[qCursor[reactiveLimitDeltaOverrideScenarioEntry[2 * k]]++] = k;
        dReactiveLimitDeltaOverrideStart.upload(qPtr.data(), qPtr.size() * sizeof(int));
        dReactiveLimitDeltaOverrideList.upload(qList.data(), qList.size() * sizeof(int));
    }
    // A lost voltage-controlling SVC also takes its share of the entry's quadratic limit coefficients (an SVC's
    // limit is its susceptance times v^2): per (scenario, entry) coefficients, read by the reactive-limits check
    // in place of the run-constant ones. Built only when some override carries such a share.
    bool& hasPerScenarioSvcSusceptanceLimits = drv.hasPerScenarioSvcSusceptanceLimits;
    hasPerScenarioSvcSusceptanceLimits = false;
    for (int k = 0; k < nReactiveLimitDeltaOverrides && !hasPerScenarioSvcSusceptanceLimits; k++) {
        hasPerScenarioSvcSusceptanceLimits = reactiveLimitDeltaOverrideValues[5 * k + 3] != 0.0 || reactiveLimitDeltaOverrideValues[5 * k + 4] != 0.0;
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
        for (int k = 0; k < nReactiveLimitDeltaOverrides; k++) {
            size_t at = static_cast<size_t>(reactiveLimitDeltaOverrideScenarioEntry[2 * k]) * nReactiveLimitEntries + reactiveLimitDeltaOverrideScenarioEntry[2 * k + 1];
            dynMin[at] += reactiveLimitDeltaOverrideValues[5 * k + 3];
            dynMax[at] += reactiveLimitDeltaOverrideValues[5 * k + 4];
        }
        if (!ctx->dReactiveLimitSvcMinBPerScenario.p) {
            ctx->dReactiveLimitSvcMinBPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * nReactiveLimitEntries * sizeof(double));
            ctx->dReactiveLimitSvcMaxBPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * nReactiveLimitEntries * sizeof(double));
        }
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitSvcMinBPerScenario.p, dynMin.data(), dynMin.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitSvcMaxBPerScenario.p, dynMax.data(), dynMax.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    // Per-scenario curve repairs, CSR by scenario: the dynamic-limits kernel runs one thread per
    // (reactive-limit entry, scenario) and walks the curve entries inside, so a flat {slot, entry}
    // scan would be paid nReactiveCapabilityCurves times per thread. Bucketing by slot makes it a handful of comparisons.
    DeviceBuf& dReactiveCapabilityCurveOverrideStart = drv.dReactiveCapabilityCurveOverrideStart;
    DeviceBuf& dReactiveCapabilityCurveOverrideCurve = drv.dReactiveCapabilityCurveOverrideCurve;
    DeviceBuf& dReactiveCapabilityCurveOverrideStepScale = drv.dReactiveCapabilityCurveOverrideStepScale;
    if (nReactiveCapabilityCurveOverrides > 0) {
        std::vector<int> ptr(nS + 1, 0);
        for (int k = 0; k < nReactiveCapabilityCurveOverrides; k++) ptr[reactiveCapabilityCurveOverrideScenarioCurve[2 * k] + 1]++;
        for (int t = 0; t < nS; t++) ptr[t + 1] += ptr[t];
        std::vector<int> cursor(ptr.begin(), ptr.end() - 1);
        std::vector<int> entry(nReactiveCapabilityCurveOverrides);
        std::vector<double> val(static_cast<size_t>(nReactiveCapabilityCurveOverrides) * 2);
        for (int k = 0; k < nReactiveCapabilityCurveOverrides; k++) {
            int at = cursor[reactiveCapabilityCurveOverrideScenarioCurve[2 * k]]++;
            entry[at] = reactiveCapabilityCurveOverrideScenarioCurve[2 * k + 1];
            val[2 * at] = reactiveCapabilityCurveOverrideStepScale[2 * k];
            val[2 * at + 1] = reactiveCapabilityCurveOverrideStepScale[2 * k + 1];
        }
        dReactiveCapabilityCurveOverrideStart.upload(ptr.data(), ptr.size() * sizeof(int));
        dReactiveCapabilityCurveOverrideCurve.upload(entry.data(), entry.size() * sizeof(int));
        dReactiveCapabilityCurveOverrideStepScale.upload(val.data(), val.size() * sizeof(double));
    }
    drv.usePerScenarioReactiveLimits = ctx->nReactiveLimitEntries > 0 && (ctx->nReactiveCapabilityCurves > 0 || nReactiveLimitDeltaOverrides > 0 || ctx->nForcedTargetQLimitEntries > 0);
    const bool& usePerScenarioReactiveLimits = drv.usePerScenarioReactiveLimits;
    if (usePerScenarioReactiveLimits && (!ctx->dReactiveLimitMinQPerScenario.p || !ctx->dReactiveLimitLoadQPerScenario.p)) {   // curves allocate these on their own
                                                            // table upload; a Q override alone must too, and a
                                                            // guard that names only ONE of them silently leaves
                                                            // the others null on a curve network (measured: the
                                                            // whole precompute fell back to the CPU)
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
    std::vector<char> outerLoopUnstable(nS);
    std::vector<char>& wasActive = drv.wasActive;
    wasActive.assign(nS, 0);
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
        std::vector<double> nanv(static_cast<size_t>(nS) * std::max(ctx->nReactiveLimitEntries, 1), std::nan(""));
        CUDA_CHECK(cudaMemcpy(ctx->dSecondaryVoltageControlReactiveLimitTargetV.p, nanv.data(), nanv.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    // no inner solve has converged yet: the start
    CUDA_CHECK(cudaMemcpyAsync(ctx->dXLastConv.p, ctx->dX.p, static_cast<size_t>(nS) * n * sizeof(double),
                               cudaMemcpyDeviceToDevice, ctx->stream));
    if (fullLoadflow) {                                      // per-solve PV↔PQ state: 0 switches, limitType from base
        seedSolve(ctx, drv, reactiveLimitsSeedState, slackBusAnchorShift);
    }

    // SEQUENTIAL outer loops, exactly like OLF's AcloadFlowEngine: fully stabilize the reactive-limits loop
    // (switch + re-solve until it switches nothing), THEN fully stabilize the distributed-slack loop
    // (distribute + re-solve until balanced), and repeat the whole RL->DS sequence until a full sequence
    // triggers no inner Newton re-solve. Running RL and DS in the SAME pass converges to a different (still
    // in-tolerance) point when switching interacts with slack distribution. phase 0 = reactive limits, 1 = slack.
    std::vector<char>& live = drv.live;
    live = active;                          // scenarios still iterating outer loops (dropped on failure)
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
    // Where the realistic-voltage check STARTS applying in this sweep. AcloadFlowEngine keeps
    // checkUnrealisticStates false up to and including the last loop that can fix an unrealistic state
    // (that is the whole point of robust mode: give ReactiveLimits / the discrete
    // TransformerVoltageControl a chance to repair the state instead of failing the run on it), then
    // turns it on for the rest of the sweep and resets it at the top of the next one. A device pass at
    // index i applies loopOrder[i] to the state its solve produced, so "past the fixing loop" is
    // i > realisticFixingIdx. Not found, or no fixing loop at all, means check every solve — the
    // conservative direction, since a check that fires too early only routes to the CPU.
    int realisticFixingIdx = -1;
    for (int i = 0; i < nLoops; i++) {
        bool isFixing = ctx->realisticFixingLoop == GpuRealisticVoltageFixingLoop::REACTIVE_LIMITS
                ? loopOrder[i] == GpuOuterLoopKind::REACTIVE_LIMITS
                : (ctx->realisticFixingLoop == GpuRealisticVoltageFixingLoop::TRANSFORMER_VOLTAGE_CONTROL
                        && loopOrder[i] == GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL
                        && ctx->transformerVoltageControlMode != GpuTransformerVoltageControlMode::WITH_GENERATOR_VOLTAGE_CONTROL);
        if (isFixing) {
            realisticFixingIdx = i;
        }
    }
    int& loopIdx = drv.loopIdx;
    loopIdx = 0;
    bool seqChanged = false;                                 // did any outer loop re-solve during the current RL->DS sequence
    std::vector<int> prodIters(nS, 0);                       // PER-SCENARIO productive outer iterations: OLF's
                                                             // maxOuterLoopIterations is a per-RUN budget — a global
                                                             // counter capped busy chunks collectively (violation-diff
                                                             // root cause: a scenario solved solo matched the CPU
                                                             // exactly but diverged inside a full 500-chunk)
    // A phase transition after a NO-SWITCH pass must NOT re-solve: OLF's runOuterLoop re-solves only on an
    // UNSTABLE check — a STABLE loop is followed by checking the NEXT loop against the SAME state. The device
    // Newton forces >=1 iteration per solve (matching OLF's per-run behavior), so an extra no-op pass would
    // move an already-converged state off the CPU's by ~the stopping residual (measured 4e-4 kV on the
    // DS->RL transition with a distributed base). solvePass=false skips the inner Newton for such a pass.
    bool solvePass = true;
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
    const bool& outerLoopTiming = drv.outerLoopTiming;
    int& outerLoopPassSolvedScenarios = drv.outerLoopPassSolvedScenarios;
    outerLoopPassSolvedScenarios = 0;
    bool outerLoopFirstSolveDone = false;
    int outerLoopTriggerKind = GpuOuterLoopKind::NONE;  // the loop whose check made the next pass re-solve
    int& outer = drv.outer;
    for (outer = 0; outer < maxOuter && loopIdx < nLoops; outer++) {
    gOuterPasses++;
    if (outerLoopTiming) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    const double outerLoopPassStartMilliseconds = outerLoopTiming ? nowMilliseconds() : 0.0;
    const long long outerLoopPassStartActiveIterations = gActiveIterSum, outerLoopPassStartWidthIterations = gWidthIterSum;
    const bool outerLoopPassSolving = solvePass;
    if (solvePass) {
        gSolvePasses++;
    }
    std::copy(active.begin(), active.end(), wasActive.begin());   // scenarios (re)solved in this inner pass
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    int& nActive = drv.nActive;
    nActive = 0;
    for (char a : active) {
        if (a) nActive++;
    }
    std::vector<char>& loopRejected = drv.loopRejected;
    loopRejected.assign(nS, 0);                   // a device loop routed the scenario to the CPU this pass
    if (solvePass) {
        newtonSolve(ctx, drv);
    }   // ---- end if (solvePass) ----
    if (outerLoopTiming) {
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        double ms = nowMilliseconds() - outerLoopPassStartMilliseconds;
        long long ai = gActiveIterSum - outerLoopPassStartActiveIterations, wi = gWidthIterSum - outerLoopPassStartWidthIterations;
        if (outerLoopPassSolving && !outerLoopFirstSolveDone) {
            gOuterLoopFirstSolveMilliseconds += ms; gOuterLoopFirstSolveScenarios += outerLoopPassSolvedScenarios; gOuterLoopFirstSolveLaneIterations += ai; gOuterLoopFirstSolveWidthIterations += wi;
            gOuterLoopFirstSolvePasses++;
            outerLoopFirstSolveDone = true;
        } else if (outerLoopPassSolving) {
            int k = outerLoopTriggerKind >= 0 && outerLoopTriggerKind < GpuOuterLoopKind::COUNT ? outerLoopTriggerKind : 0;
            gOuterLoopReSolveMilliseconds[k] += ms; gOuterLoopReSolveScenarios[k] += outerLoopPassSolvedScenarios; gOuterLoopReSolveLaneIterations[k] += ai;
            gOuterLoopReSolveWidthIterations[k] += wi; gOuterLoopReSolvePasses[k]++;
        }
    }
    // the CHECK part of the pass, charged to the loop being checked when the pass body ends (break/continue too)
    struct OuterLoopCheckTimer {
        bool on; double t0; int kind;
        ~OuterLoopCheckTimer() {
            if (on && kind >= 0 && kind < GpuOuterLoopKind::COUNT) {
                gOuterLoopCheckMilliseconds[kind] += nowMilliseconds() - t0;
                gOuterLoopCheckPasses[kind]++;
            }
        }
    } outerLoopCheckTimer{outerLoopTiming, outerLoopTiming ? nowMilliseconds() : 0.0, -1};

    // ---- AcloadFlowEngine's realistic-voltage check, on this INTERMEDIATE state ----
    // OLF re-checks after every solver run once the last loop that can fix an unrealistic state has
    // run, and an unrealistic state ends the outer-loop cycle with UNREALISTIC_STATE: the run FAILS
    // and reports no violations. Which solve the check belongs to depends on the configured loop
    // order and on robust mode, so rather than reproduce that ordering the device routes any
    // scenario that is EVER unrealistic to the CPU replay, which applies OLF's rule itself. That is
    // stricter than OLF (a state it would have let a later loop fix also leaves the batch), so it
    // can cost batching share and can never change an answer.
    // One check over the scenarios gate selects: any whose state leaves the band goes to the CPU replay.
    if (ctx->realisticCheck && solvePass && loopIdx > realisticFixingIdx) {
        std::vector<char> gate(nS);
        for (int s = 0; s < nS; s++) {
            gate[s] = (wasActive[s] && res.status[s] == GpuAcSolverStatus::CONVERGED && live[s]) ? 1 : 0;
        }
        rejectUnrealistic(ctx, drv, gate);
    }

    if (!fullLoadflow) {
        break;                                              // fixed-iter / no controllers: one pass only
    }

    // ---- this pass's check: the scenarios the pass (re)solved, or every live one on a check-only pass, that converged ----
    for (int s = 0; s < nS; s++) {
        outerLoopCheckActive[s] = (wasActive[s] && res.status[s] == GpuAcSolverStatus::CONVERGED) ? 1 : 0;
    }
    outerLoopCheckTimer.kind = loopOrder[loopIdx];
    outerLoopTriggerKind = loopOrder[loopIdx];                     // a re-solve in the next pass was triggered by this check
    checkOuterLoop(ctx, drv, solvePass, outerLoopUnstable);
    int nChanged = 0;
    for (int s = 0; s < nS; s++) {
        if (outerLoopUnstable[s]) {
            nChanged++;
        }
    }
    // OLF_GPU_LOOP_TRACE=<slot>: the device side of OLF's OL_CHECK - which loop this pass ran, whether
    // it found the slot UNSTABLE (a re-solve follows) and whether the whole phase is still unstable.
    // Two arms that agree on every residual and still stop apart differ in this SEQUENCE.
    static const char* loopTraceEnv = traceValue(Trace::LOOP_TRACE);
    // a slot resolved per chunk from a contingency id (setBatchLoopTraceSlot) wins over the env slot:
    // chunk composition moves with the resolved chunk size, so a fixed slot traces a different
    // scenario from one run to the next (IZUZ: slot 170, then 18)
    const int loopTraceSlot = ctx->loopTraceSlot >= 0 ? ctx->loopTraceSlot : (loopTraceEnv != nullptr ? std::atoi(loopTraceEnv) : -1);
    traceLoopSlot(ctx, drv, loopTraceSlot, nChanged, outerLoopUnstable, solvePass);
    if (nChanged > 0) {                                     // this outer loop is still unstable
        for (int s = 0; s < nS; s++) {
            if (outerLoopUnstable[s] && ++prodIters[s] >= MAX_OUTER_LOOP_ITERATIONS) {
                // This scenario exhausted ITS outer-iteration budget mid-sequence. Never report it
                // converged on a half-finished RL/DS fixpoint — reject to the CPU fallback, which
                // replays it under OLF's own maxOuterLoopIterations/UNSTABLE handling.
                traceReject(s, "outer-loop-budget", outer, 0);
                res.status[s] = GpuAcSolverStatus::REJECTED;
                outerLoopUnstable[s] = 0;
                live[s] = 0;
            }
            active[s] = outerLoopUnstable[s];                      // re-solve only the scenarios it changed, stay in this phase
        }
        seqChanged = true;
        solvePass = true;                                   // mutated targets/modes -> a real re-solve is due
    } else {                                               // this outer loop stabilized
        // The last loop that can fix an unrealistic state is done for this sweep: AcloadFlowEngine checks the
        // state it leaves (runOuterLoop, after its do-while), whether or not the loop re-solved. The solve gate
        // above only sees the solves of the LATER loops, so with that loop last in the order (the discrete
        // TransformerVoltageControl under the validation parameters) only the final state was ever checked.
        if (ctx->realisticCheck && realisticFixingIdx >= 0 && loopIdx == realisticFixingIdx) {
            std::vector<char> gate(nS);
            for (int s = 0; s < nS; s++) {
                gate[s] = (res.status[s] == GpuAcSolverStatus::CONVERGED && live[s]) ? 1 : 0;
            }
            rejectUnrealistic(ctx, drv, gate);
        }
        if (++loopIdx >= nLoops) {                          // ...move to the next loop in OLF's order
            if (!seqChanged) {
                break;                                      // a full sweep re-solved nothing -> converged
            }
            seqChanged = false;
            loopIdx = 0;                                    // restart the sweep at the first loop
        }
        for (int s = 0; s < nS; s++) {
            active[s] = live[s];                            // re-CHECK every live scenario in the next phase...
        }
        solvePass = false;                                  // ...against the SAME state: STABLE never re-solves (OLF)
    }
    }   // ---- end sequential outer-loop driver ----

    traceSwitchLog(ctx, drv);
    // ---- OLF_GPU_RL_AUDIT=<pu threshold>: audit the FINAL state against the reactive-limit rule
    //      itself. OLF's outer loop ends only after a check found NOTHING to switch, so in OLF's
    //      answer every controller still PV sits inside its reactive limits (within the switch
    //      tolerance). A device scenario that returns with a PV controller beyond its limit
    //      did not reach that fixpoint - it stopped one check short - and that is a defect no
    //      comparison with the CPU is needed to see. Re-fills the diagnostic q at the final state
    //      for every live scenario, refreshes the curve limits at the final targets, and prints one
    //      line per violating (scenario, entry).
    traceReactiveLimitsAudit(ctx, drv, drv.reactiveLimitsAuditLog);
    if (reactiveLimitsFinalState != nullptr && ctx->nReactiveLimitEntries > 0) {             // export per-(controller, scenario) RL state
        std::vector<char> qLimitType(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS), sc(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        CUDA_CHECK(cudaMemcpy(qLimitType.data(), ctx->dReactiveLimitQLimitType.p, qLimitType.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sc.data(), ctx->dReactiveLimitPvPqSwitchCount.p, sc.size(), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < qLimitType.size(); k++) {
            reactiveLimitsFinalState[k] = reactiveLimitsState(qLimitType[k], sc[k]);
        }
    }
    tracePhaseTime(ctx, drv);
    CUDA_CHECK(cudaMemcpy(res.x.data(), ctx->dX.p, nTot * sizeof(double), cudaMemcpyDeviceToHost));
    return std::move(res);                               // drv is this call's: hand its n x nS state over, not a copy
}

}  // namespace olfgpu
