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
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_ac_solver_status.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"
#include "../common/codes/gpu_reactive_limits_state.cuh"
#include "../common/codes/gpu_realistic_voltage_fixing_loop.cuh"
#include "../common/codes/gpu_controlled_flow.cuh"
#include "../common/codes/gpu_transformer_voltage_control_mode.cuh"
#include "../diagnostics/driver_traces.cuh"
#include "../diagnostics/matrix_scan.cuh"
#include "../equations/branch_fill.cuh"
#include "../equations/bus_injection_fill.cuh"
#include "../equations/distribution_fill.cuh"
#include "../equations/fill_launch.cuh"
#include "../equations/scatter.cuh"
#include "../equations/scenario_overrides.cuh"
#include "../equations/zero_impedance_fill.cuh"
#include "../outerloop/distributed_slack.cuh"
#include "../outerloop/forced_target_q.cuh"
#include "../outerloop/hvdc_ac_emulation.cuh"
#include "../outerloop/incremental_flow_control.cuh"
#include "../outerloop/incremental_shunt_voltage_control.cuh"
#include "../outerloop/incremental_transformer_voltage_control.cuh"
#include "../outerloop/phase_control.cuh"
#include "../outerloop/pilot_voltage.cuh"
#include "../outerloop/reactive_capability_curve.cuh"
#include "../outerloop/reactive_capability_curves.cuh"
#include "../outerloop/reactive_limits.cuh"
#include "../outerloop/reactive_limits_seed.cuh"
#include "../outerloop/remote_voltage_control.cuh"
#include "../outerloop/secondary_voltage_control.cuh"
#include "../outerloop/sensitivity.cuh"
#include "../outerloop/shunt_voltage_control.cuh"
#include "../outerloop/slack_participants.cuh"
#include "../outerloop/transformer_voltage_control.cuh"
#include "../outerloop/voltage_monitoring.cuh"
#include "../solver/batch_context.cuh"
#include "../solver/compact_solve.cuh"
#include "../solver/convergence.cuh"
#include "../solver/driver_pass.cuh"
#include "../solver/driver_steps.cuh"
#include "../solver/iterative_refinement.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "../solver/newton_iteration.cuh"
#include "../solver/realistic_voltage_check.cuh"
#include "../solver/solve_seed.cuh"
#include "../solver/state_vector_scaling.cuh"
#include "../solver/zero_diagonal_repair.cuh"
#include "outer_loop_driver.cuh"

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
                              int nOv, const int* ovIndex, const double* ovValue,
                              const int* dsOvBus, const double* dsOvFactorDelta, const double* dsOvBaseDelta,
                              const double* dsOvMaxDelta, const double* dsOvMinDelta, const double* dsAnchorShift,
                              const double* dsSlackShift,
                              int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                              int nRlQOv, const int* rlQOvSlotEntry, const double* rlQOvMinMax,
                              int nCvOv, const int* cvOvSlotEntry, const double* cvOvStepScale,
                              int maxIter, double tol,
                              double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                              int lsMode, double maxDv, double maxDphi, const int* varType,
                              const double* x0Full, int nShOv, const int* shOvSlotElem, const double* shOvGb,
                              const int* rlStateIn, int* rlStateOut,
                              int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                              int maxOuterProductive, int rlMaxSwitch, int dsRejectResidual, int skipRlPhase,
                              int nziBase, int nZiOn, const int* ziOnSlotElem,
                              int nZiOff, const int* ziOffSlotElem) {
    DriverPass drv;
    drv.n = ctx->n;
    drv.nnz = ctx->nnz;
    drv.nS = ctx->nScenarios;
    drv.nRlVtOv = nRlVtOv;
    drv.nRlQOv = nRlQOv;
    drv.nCvOv = nCvOv;
    drv.skipRlPhase = skipRlPhase;
    const int& n = drv.n;
    const int& nnz = drv.nnz;
    const int& nS = drv.nS;
    // MAX_VOLTAGE_CHANGE (GpuStateVectorScalingMode::MAX_VOLTAGE_CHANGE): per-scenario step clamp, same rule as the base solver's
    // (varType 1 = V vs maxDv, 0 = PHI vs maxDphi; whole step scaled). Buffers allocated lazily.
    uploadBatchOverrides(ctx, lsMode, maxDv, maxDphi, varType, nShOv, shOvSlotElem, shOvGb,
                         nziBase, nZiOff, ziOffSlotElem, nZiOn, ziOnSlotElem, nR1Ov, r1OvSlotElem, r1OvVal);
    drv.nTot = static_cast<size_t>(n) * nS;
    const size_t& nTot = drv.nTot;
    drv.tol = tol;
    drv.maxIter = maxIter;
    drv.lsMode = lsMode;
    drv.convEpsPerEq = convEpsPerEq;
    DeviceBuf dOvIndex, dOvValue;                            // function-scoped: outlive the async override kernel
    DeviceBuf& dBaseMode = drv.dBaseMode;                   // the contingency-free base row mode (length n)
    DeviceBuf dModeSlots, dModeRows;                         // compact row modes: the scenarios that differ from the base
    DeviceBuf dX0SeedIdx, dX0SeedVal;                        // setBatchX0Seeds, function-scoped like the overrides
    // Per-scenario distributed-slack reweight (generator loss at a DS-participant bus): one affected DS bus per
    // scenario (dsOvBus[s], -1 if none) whose factor/base/headroom/footroom shrink by the lost gen's share, and
    // a per-scenario factor-sum drop so the other buses pick up that share. nullptr when no DS scenarios.
    DeviceBuf& dDsOvBus = drv.dDsOvBus;
    DeviceBuf& dDsOvFactorDelta = drv.dDsOvFactorDelta;
    DeviceBuf& dDsOvBaseDelta = drv.dDsOvBaseDelta;
    DeviceBuf& dDsOvMaxDelta = drv.dDsOvMaxDelta;
    DeviceBuf& dDsOvMinDelta = drv.dDsOvMinDelta;
    drv.hasDsOverride = (dsOvBus != nullptr && ctx->nds > 0);
    const bool& hasDsOverride = drv.hasDsOverride;
    // Per-scenario DS ANCHOR SHIFT (pair/strategy legs): shift_i = parent's final target at participant i
    // minus the extraction-time base — re-anchors the water-fill's cumulative reference at the PARENT
    // contingency's distribution, mirroring OLF's setGeneratorsInitialTargetPToTargetP before a strategy
    // re-solve. nullptr = anchored at the base (plain contingencies).
    DeviceBuf& dRlVtOvSlotIdx = drv.dRlVtOvSlotIdx;
    DeviceBuf& dRlVtOvVal = drv.dRlVtOvVal;
    if (nRlVtOv > 0) {
        dRlVtOvSlotIdx.upload(rlVtOvSlotIdx, 2L * nRlVtOv * sizeof(int));
        dRlVtOvVal.upload(rlVtOvVal, (size_t) nRlVtOv * sizeof(double));
    }
    DeviceBuf& dDsAnchorShift = drv.dDsAnchorShift;
    DeviceBuf& dDsSlackShift = drv.dDsSlackShift;
    drv.hasDsAnchor = (dsAnchorShift != nullptr && ctx->nds > 0);
    const bool& hasDsAnchor = drv.hasDsAnchor;
    if (hasDsAnchor) {
        dDsAnchorShift.upload(dsAnchorShift, static_cast<size_t>(nS) * ctx->nds * sizeof(double));
    }
    // The slack's own target move, independent of the participants' anchors: a strategy pair's parent share,
    // and also a contingency's load lost at the slack (a plain scenario, no anchor shift).
    drv.hasDsSlackShift = (dsSlackShift != nullptr && ctx->nds > 0);
    const bool& hasDsSlackShift = drv.hasDsSlackShift;
    if (hasDsSlackShift) {
        dDsSlackShift.upload(dsSlackShift, static_cast<size_t>(nS) * sizeof(double));
    }

    // x0/target are the BASE row (length n), identical for every scenario: upload once and tile across
    // nS on-device. rowMode is PER-SCENARIO (length n*nS): a contingency can freeze islanded buses
    // (Phase 0b) or flip PV↔PQ via reactive limits (Phase 1), so it is NOT tiled — uploaded as-is and
    // mutated in place on-device by the outer-loop kernels. Only the disabled mask is also per-scenario.
    // x0: either the BASE row tiled across nS (default), or a FULL per-scenario n*nS state
    // (x0Full != null) — the warm-restart hook for multi-call outer sequencing (e.g. the
    // transformer voltage-control orchestrator re-solving scenarios from their previous states).
    // RESUME: the driver is continuing from a LOOP_HOST pause, so x/targets/modes already hold the
    // sequence's current state on the device — re-uploading the base rows would throw away the
    // distributed slack, the HVDC deltas and the monitoring activations it has accumulated.
    const bool resume = ctx->driverResume != 0;
    if (!resume || ctx->olPasses.size() != static_cast<size_t>(nS)) {
        ctx->olFirst.assign(nS, 0);
        ctx->olIters.assign(nS, 0);
        ctx->olPasses.assign(nS, 0);
    }
    ctx->driverResume = 0;
    ctx->driverPaused = 0;
    if (resume && (ctx->nTvc > 0 || ctx->nFlc > 0 || ctx->nCph > 0 || ctx->nTvd > 0)) {
        tvcRebuildOverrides(ctx);                            // uploadBatchOverrides just erased the moves
    }
    if (resume && ctx->nShc > 0) {
        shcRebuildOverrides(ctx);
    }
    dBaseMode.upload(baseRowMode, n * sizeof(int));         // base (contingency-free) row mode for the RL limit-type seed
    if (!resume) {
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
        if (nOv > 0) {                                       // per-scenario target overrides (generator loss)
            dOvIndex.upload(ovIndex, nOv * sizeof(int));
            dOvValue.upload(ovValue, nOv * sizeof(double));
            applyTargetOverrideBatch<<<grid(nOv), BLOCK, 0, ctx->stream>>>(nOv, dOvIndex.as<int>(), dOvValue.as<double>(), ctx->dTarget.as<double>());
        }
    }
    if (hasDsOverride) {                                     // per-scenario generator-loss DS reweight
        dDsOvBus.upload(dsOvBus, nS * sizeof(int));
        dDsOvFactorDelta.upload(dsOvFactorDelta, nS * sizeof(double));
        dDsOvBaseDelta.upload(dsOvBaseDelta, nS * sizeof(double));
        dDsOvMaxDelta.upload(dsOvMaxDelta, nS * sizeof(double));
        dDsOvMinDelta.upload(dsOvMinDelta, nS * sizeof(double));
    }
    if (!resume) {
        // NOT on a resume: the HVDC AC-emulation driver phases MUTATE element bits on the device during
        // the paused sweep (a frozen or saturated emulation has its coupling deactivated), and the host's
        // mirror of the mask is the pre-solve one. Re-uploading it mid-sequence silently undoes those
        // loops' work — measured on a 400-contingency rte6515 slice: 5 differing violations against 1,
        // with NO host decision involved at all (the split alone did it). Contingency outages do not
        // change during a sweep, so the device copy stays authoritative across the pause.
        CUDA_CHECK(cudaMemcpy(ctx->dDisabled.p, disabled,
                              static_cast<size_t>(ctx->nElements) * nS * sizeof(char), cudaMemcpyHostToDevice));
    }

    std::vector<char>& active = drv.active;
    active.assign(nS, 1);

    BatchResult& res = drv.res;
    res.x.resize(nTot);
    res.iters.assign(nS, 0);
    res.norm.assign(nS, 0.0);
    res.status.assign(nS, GpuAcSolverStatus::MAX_ITERATION_REACHED);  // default (overwritten on converge/diverge)

    std::vector<double>& hF = drv.hF;
    hF.assign(nTot, 0.0);

    // Device-side fixed-iteration mode: tol <= 0 means "run exactly maxIter Newton steps
    // on the device with NO per-iteration host sync and NO convergence retirement" (the
    // a fixed-iteration NR loop). tol > 0 keeps the per-iteration retirement loop.
    drv.deviceFixedIter = (tol <= 0.0);
    const bool& deviceFixedIter = drv.deviceFixedIter;
    // A scenario that exhausts the budget is reported as MAX_ITERATION_REACHED by the device itself (the
    // precomputer's exhausted set), which is OLF's own answer: no early exit hands stragglers to the CPU.

    // Full-loadflow mode (tol > 0) with local PV controllers runs the reactive-limits (PV→PQ) OUTER loop:
    // converge the inner Newton, switch any generator over its Q limit, re-converge the switched scenarios,
    // repeat until a pass switches nothing. Fixed-iter mode (tol <= 0, the throughput benchmark) and
    // family-free networks with no controllers do ONE pass (no outer loop) — byte-identical to before.
    bool fullLoadflow = !deviceFixedIter && (ctx->nrl > 0 || ctx->nds > 0 || ctx->dsSlackOnly
                                             || ctx->nmon > 0 || ctx->nTvc > 0 || ctx->nShc > 0 || ctx->nFlc > 0 || ctx->nCph > 0
                                             || ctx->nTvd > 0 || ctx->nSvd > 0 || ctx->nSvc > 0);
    const int RL_MAX_OUTER = maxOuterProductive > 0 ? maxOuterProductive : 20;   // OLF maxOuterLoopIterations
    // per-bus PV→PQ count cap (OLF maxPqPvSwitch): 0 is OLF's "never switch back" and is honoured as such;
    // only a negative value (no cap given) takes OLF's default of 3.
    drv.rlMaxSwitch = rlMaxSwitch >= 0 ? rlMaxSwitch : 3;
    static const char* swLogEnv = std::getenv("OLF_GPU_SW_LOG");
    drv.swLogSlot = swLogEnv == nullptr ? -1 : (std::string(swLogEnv) == "all" ? -2 : std::atoi(swLogEnv));
    // Outer-loop convergence tolerances (pu) — passed in so the device uses the SAME criteria as the CPU
    // loadflow that built the base case (maxReactivePowerMismatch / slackBusPMaxMismatch, per-unitized), NOT
    // hardcoded constants. With matching parameters and the same equations the GPU outer-loop fixed point
    // matches the CPU's to the inner-Newton precision; a loose tolerance (e.g. the 1 MW slack default) leaves
    // a redistribution band, so a test wanting machine-precision agreement passes tight values on both sides.
    drv.reactiveLimitTol = reactiveLimitTol;
    drv.slackDistTol = slackDistTol;
    drv.pResidueEps = 1e-5;                                  // OLF ActivePowerDistribution.P_RESIDUE_EPS: the
                                                            // water-filling distributes the mismatch down to this
    // RL_MAX_OUTER bounds the PRODUCTIVE outer iterations (an outer loop that re-solved), matching OLF's
    // outerLoopTotalIterations budget. The raw pass loop gets a generous safety bound on top, because the
    // sequential driver also spends non-productive passes (phase transitions / stability re-checks) that OLF
    // does not count.
    // The reactive-limit limits become PER-SCENARIO as soon as either channel is live: curve limits at
    // the runtime P, or a generator lost at a bus that keeps voltage control taking its capability away.
    DeviceBuf& dRlQOvSlotEntry = drv.dRlQOvSlotEntry;
    DeviceBuf& dRlQOvMinMax = drv.dRlQOvMinMax;
    DeviceBuf& dRlQOvPtr = drv.dRlQOvPtr;
    DeviceBuf& dRlQOvList = drv.dRlQOvList;
    if (nRlQOv > 0) {
        dRlQOvSlotEntry.upload(rlQOvSlotEntry, static_cast<size_t>(nRlQOv) * 2 * sizeof(int));
        dRlQOvMinMax.upload(rlQOvMinMax, static_cast<size_t>(nRlQOv) * 5 * sizeof(double));   // {dMinQ, dMaxQ, dLoadQ, dBmin, dBmax}
        // ...indexed by slot (CSR, entry order kept), so the dynamic-limits kernel reads its scenario's own
        // overrides instead of scanning the chunk's.
        std::vector<int> qPtr(nS + 1, 0);
        std::vector<int> qList(nRlQOv);
        for (int k = 0; k < nRlQOv; k++) qPtr[rlQOvSlotEntry[2 * k] + 1]++;
        for (int t = 0; t < nS; t++) qPtr[t + 1] += qPtr[t];
        std::vector<int> qCursor(qPtr.begin(), qPtr.end() - 1);
        for (int k = 0; k < nRlQOv; k++) qList[qCursor[rlQOvSlotEntry[2 * k]]++] = k;
        dRlQOvPtr.upload(qPtr.data(), qPtr.size() * sizeof(int));
        dRlQOvList.upload(qList.data(), qList.size() * sizeof(int));
    }
    // A lost voltage-controlling SVC also takes its share of the entry's quadratic limit coefficients (an SVC's
    // limit is its susceptance times v^2): per (scenario, entry) coefficients, read by the reactive-limits check
    // in place of the run-constant ones. Built only when some override carries such a share.
    bool& svcCoefDyn = drv.svcCoefDyn;
    svcCoefDyn = false;
    for (int k = 0; k < nRlQOv && !svcCoefDyn; k++) {
        svcCoefDyn = rlQOvMinMax[5 * k + 3] != 0.0 || rlQOvMinMax[5 * k + 4] != 0.0;
    }
    if (svcCoefDyn && ctx->nrl > 0) {
        size_t nrl = static_cast<size_t>(ctx->nrl);
        std::vector<double> bmin(nrl), bmax(nrl);
        CUDA_CHECK(cudaMemcpy(bmin.data(), ctx->dRlSvcBmin.p, nrl * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bmax.data(), ctx->dRlSvcBmax.p, nrl * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> dynMin(static_cast<size_t>(nS) * nrl), dynMax(static_cast<size_t>(nS) * nrl);
        for (int t = 0; t < nS; t++) {
            std::copy(bmin.begin(), bmin.end(), dynMin.begin() + static_cast<size_t>(t) * nrl);
            std::copy(bmax.begin(), bmax.end(), dynMax.begin() + static_cast<size_t>(t) * nrl);
        }
        for (int k = 0; k < nRlQOv; k++) {
            size_t at = static_cast<size_t>(rlQOvSlotEntry[2 * k]) * nrl + rlQOvSlotEntry[2 * k + 1];
            dynMin[at] += rlQOvMinMax[5 * k + 3];
            dynMax[at] += rlQOvMinMax[5 * k + 4];
        }
        if (!ctx->dRlSvcBminDyn.p) {
            ctx->dRlSvcBminDyn.alloc(static_cast<size_t>(ctx->nScenarios) * nrl * sizeof(double));
            ctx->dRlSvcBmaxDyn.alloc(static_cast<size_t>(ctx->nScenarios) * nrl * sizeof(double));
        }
        CUDA_CHECK(cudaMemcpy(ctx->dRlSvcBminDyn.p, dynMin.data(), dynMin.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dRlSvcBmaxDyn.p, dynMax.data(), dynMax.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    // Per-scenario curve repairs, CSR by scenario: the dynamic-limits kernel runs one thread per
    // (reactive-limit entry, scenario) and walks the curve entries inside, so a flat {slot, entry}
    // scan would be paid ncv times per thread. Bucketing by slot makes it a handful of comparisons.
    DeviceBuf& dCvOvPtr = drv.dCvOvPtr;
    DeviceBuf& dCvOvEntry = drv.dCvOvEntry;
    DeviceBuf& dCvOvVal = drv.dCvOvVal;
    if (nCvOv > 0) {
        std::vector<int> ptr(nS + 1, 0);
        for (int k = 0; k < nCvOv; k++) ptr[cvOvSlotEntry[2 * k] + 1]++;
        for (int t = 0; t < nS; t++) ptr[t + 1] += ptr[t];
        std::vector<int> cursor(ptr.begin(), ptr.end() - 1);
        std::vector<int> entry(nCvOv);
        std::vector<double> val(static_cast<size_t>(nCvOv) * 2);
        for (int k = 0; k < nCvOv; k++) {
            int at = cursor[cvOvSlotEntry[2 * k]]++;
            entry[at] = cvOvSlotEntry[2 * k + 1];
            val[2 * at] = cvOvStepScale[2 * k];
            val[2 * at + 1] = cvOvStepScale[2 * k + 1];
        }
        dCvOvPtr.upload(ptr.data(), ptr.size() * sizeof(int));
        dCvOvEntry.upload(entry.data(), entry.size() * sizeof(int));
        dCvOvVal.upload(val.data(), val.size() * sizeof(double));
    }
    drv.useDynRlLimits = ctx->nrl > 0 && (ctx->ncv > 0 || nRlQOv > 0 || ctx->nfqLim > 0);
    const bool& useDynRlLimits = drv.useDynRlLimits;
    if (useDynRlLimits && (!ctx->dRlMinQDyn.p || !ctx->dRlLoadQDyn.p)) {   // curves allocate these on their own
                                                            // table upload; a Q override alone must too, and a
                                                            // guard that names only ONE of them silently leaves
                                                            // the others null on a curve network (measured: the
                                                            // whole precompute fell back to the CPU)
        ctx->dRlLoadQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
        ctx->dRlMinQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
        ctx->dRlMaxQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    }
    int maxOuter = fullLoadflow ? 8 * RL_MAX_OUTER : 1;
    // The two kernels that lay a group's rows out from the CURRENT pin states (the RL pass runs them after
    // its pins; the seeding below runs them once so the FIRST fill sees the seeded pins on the rows).
    // `staticLimits`: use the STATIC reactive-limit arrays rather than the curve-evaluated ones. At SEED time the
    // dynamic arrays are still zero (rlCurveLimitsBatch runs in the reactive-limits phase), so a seed-time flip
    // stamped the pilot's own-Q target as 0 instead of (limit - loadQ) and the whole slack phase ran against it —
    // measured on rte6515 BUS-6112_BBS_2: row 12834 converged at target 0 through outer 32, then the RL phase
    // re-stamped -0.444 and the solve had to catch up, one pass behind OLF. The re-stamp branch in the kernel
    // keeps the value current once the curve limits exist, so the static value is a correct starting point.
    std::vector<char>& rlActive = drv.rlActive;
    rlActive.assign(nS, 0);
    std::vector<char> rlSwitched(nS);
    std::vector<char>& wasActive = drv.wasActive;
    wasActive.assign(nS, 0);
    // OLF_GPU_RL_AUDIT: per-scenario decision history, kept in memory and printed only for the
    // scenarios the end-of-solve audit flags. The failing scenario is not known until the solve is
    // over and the flip is rare, so the only way to see its trajectory is to record every scenario's
    // and throw away the rest.
    const bool rlAuditOn = std::getenv("OLF_GPU_RL_AUDIT") != nullptr;
    std::vector<std::vector<std::string>> auditLog(rlAuditOn ? nS : 0);
    // ...but NOT on a resume: these are per-SOLVE state, and a resumed driver is continuing the SAME
    // solve. Re-seeding here would discard the RL pins, monitoring activations and absorbed slack the
    // sequence has already established — which is exactly what makes a resume cheaper than the
    // orchestrator pattern it replaces: that one resets and re-runs the whole driver every round.
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr && nS == 1) {
        std::fprintf(stderr, "CTX_FLAGS nS=1 nrl=%d nRlRemote=%d hasDqQrGroups=%d nQr=%d ndq=%d nds=%d ngen=%d nzi=%d nziBase=%d nRmq=%d resume=%d fullLoadflow=%d\n",
                     ctx->nrl, ctx->nRlRemote, ctx->hasDqQrGroups ? 1 : 0, ctx->st->nQr, ctx->ndq, ctx->nds, ctx->ngen, ctx->nzi, ctx->nziBase, ctx->nRmq, resume ? 1 : 0, fullLoadflow ? 1 : 0);
    }
    if (fullLoadflow && !resume && ctx->nSvc > 0) {
        // BEFORE the reactive-limit seeding below, which consults it: no zone has moved a target yet.
        std::vector<double> nanv(static_cast<size_t>(nS) * std::max(ctx->nrl, 1), std::nan(""));
        CUDA_CHECK(cudaMemcpy(ctx->dSvcRlVt.p, nanv.data(), nanv.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (!resume) {                                           // no inner solve has converged yet: the start
        CUDA_CHECK(cudaMemcpyAsync(ctx->dXLastConv.p, ctx->dX.p, static_cast<size_t>(nS) * n * sizeof(double),
                                   cudaMemcpyDeviceToDevice, ctx->stream));
    }
    if (fullLoadflow && !resume) {                           // per-solve PV↔PQ state: 0 switches, limitType from base
        seedSolve(ctx, drv, rlStateIn, dsSlackShift);
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
    int nLoops = 0;
    if (ctx->nds > 0 || ctx->dsSlackOnly) {
        loopOrder[nLoops++] = GpuOuterLoopKind::DISTRIBUTED_SLACK;
    }
    if (ctx->nhv > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION;  // OLF: FreezingHvdcACEmulation
        if (ctx->hvLimitsActive) {
            loopOrder[nLoops++] = GpuOuterLoopKind::HVDC_AC_EMULATION_LIMITS;  // OLF: AcHvdcAcEmulationLimits
        }
    }
    if (ctx->nSvc > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::SECONDARY_VOLTAGE_CONTROL;  // OLF order 4: SecondaryVoltageControl, IN-KERNEL
    } else if (ctx->hostLoopMask & hostLoopMaskBit(GpuOuterLoopKind::HOST_SECONDARY_VOLTAGE_CONTROL)) {
        loopOrder[nLoops++] = GpuOuterLoopKind::HOST_SECONDARY_VOLTAGE_CONTROL;  // OLF order 4: SecondaryVoltageControl, host pause
    }
    if (ctx->nmon > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::MONITORING_VOLTAGE;  // OLF order 5: MonitoringVoltage
    }
    if (ctx->nrl > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::REACTIVE_LIMITS;  // OLF order 6: ReactiveLimits
    }
    if (ctx->nFlcPhase > 0 || ctx->nCph > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::INCREMENTAL_PHASE_CONTROL;  // OLF order 7: (Incremental)PhaseControl, IN-KERNEL
    } else if (ctx->hostLoopMask & hostLoopMaskBit(GpuOuterLoopKind::HOST_PHASE_CONTROL)) {
        loopOrder[nLoops++] = GpuOuterLoopKind::HOST_PHASE_CONTROL;  // OLF order 7: PhaseControl
    }
    if (ctx->nTvc > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL;  // OLF order 8: TransformerVoltageControl, IN-KERNEL
    } else if (ctx->nTvd > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL;  // OLF order 8: AFTER_GENERATOR / Simple TVC, IN-KERNEL
    } else if (ctx->hostLoopMask & hostLoopMaskBit(GpuOuterLoopKind::HOST_TRANSFORMER_VOLTAGE_CONTROL)) {
        loopOrder[nLoops++] = GpuOuterLoopKind::HOST_TRANSFORMER_VOLTAGE_CONTROL;  // OLF order 8: TransformerVoltageControl, host pause
    }
    if (ctx->nFlcRpc > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL;  // OLF order 9: TransformerReactivePowerControl, IN-KERNEL
    } else if (ctx->hostLoopMask & hostLoopMaskBit(GpuOuterLoopKind::HOST_TRANSFORMER_REACTIVE_POWER_CONTROL)) {
        loopOrder[nLoops++] = GpuOuterLoopKind::HOST_TRANSFORMER_REACTIVE_POWER_CONTROL;  // OLF order 9: TransformerReactivePowerControl
    }
    if (ctx->nShc > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL;  // OLF order 10: ShuntVoltageControl, IN-KERNEL
    } else if (ctx->nSvd > 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL;  // OLF order 10: WITH_GENERATOR shunt VC, IN-KERNEL
    } else if (ctx->hostLoopMask & hostLoopMaskBit(GpuOuterLoopKind::HOST_SHUNT_VOLTAGE_CONTROL)) {
        loopOrder[nLoops++] = GpuOuterLoopKind::HOST_SHUNT_VOLTAGE_CONTROL;  // OLF order 10: ShuntVoltageControl
    }
    if (nLoops == 0) {
        loopOrder[nLoops++] = GpuOuterLoopKind::REACTIVE_LIMITS;  // nothing to drive: the old code's phase-1 fallback
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
                        && (loopOrder[i] == GpuOuterLoopKind::HOST_TRANSFORMER_VOLTAGE_CONTROL
                            || (loopOrder[i] == GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL && ctx->tvdMode != GpuTransformerVoltageControlMode::WITH_GENERATOR_VOLTAGE_CONTROL)));
        if (isFixing) {
            realisticFixingIdx = i;
        }
    }
    int& loopIdx = drv.loopIdx;
    loopIdx = 0;
    bool seqChanged = false;                                 // did any outer loop re-solve during the current RL->DS sequence
    if (!resume && ctx->startLoopKind != GpuOuterLoopKind::NONE) {
        // Restart after a host loop mutated the system: the first pass solves, then the sweep picks
        // up at THAT loop's slot (OLF's runOuterLoop re-checks it until STABLE, and only then does
        // the sweep restart from the first loop — seqChanged makes that restart happen).
        for (int i = 0; i < nLoops; i++) {
            if (loopOrder[i] == ctx->startLoopKind) {
                loopIdx = i;
                seqChanged = true;
            }
        }
        ctx->startLoopKind = GpuOuterLoopKind::NONE;
    }
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
    if (resume) {
        // Continue the paused sequence: the loop the host owned has just run, so step PAST it and
        // let its verdict drive the pass exactly as a device loop's would — UNSTABLE re-solves only
        // the scenarios it changed, STABLE advances without a solve (OLF's rule).
        loopIdx = ctx->pausedLoopIdx;
        seqChanged = ctx->pausedSeqChanged != 0;
        prodIters = ctx->pausedProdIters;
        live = ctx->pausedLive;
        if (ctx->pausedStatus.size() == res.status.size()) {
            res.status = ctx->pausedStatus;                  // the state is the converged one the pause left: a
        }                                                    // STABLE host verdict then re-CHECKS RL on it (rlActive)
        bool anyChanged = false;
        for (int s = 0; s < nS; s++) {
            char ch = s < (int) ctx->hostChanged.size() ? ctx->hostChanged[s] : 0;
            active[s] = ch ? 1 : 0;
            anyChanged |= ch != 0;
        }
        if (anyChanged) {
            seqChanged = true;
            solvePass = true;                                // the host loop moved targets -> re-solve
            // ...and it CONSUMES the outer-iteration budget, exactly as a device loop's change does.
            // Without this a host loop that keeps going UNSTABLE never retires a scenario: each resume
            // starts a fresh `outer` budget, so the pause/resume cycle spins forever (measured: a
            // 2000-contingency rte6515 slice with SecondaryVoltageControl did not finish in 28 min).
            for (int s = 0; s < nS; s++) {
                if (active[s] && ++prodIters[s] >= RL_MAX_OUTER) {
                    traceReject(s, "host-loop-outer-budget", -1, 0);
                    res.status[s] = GpuAcSolverStatus::REJECTED;                      // budget exhausted -> CPU replays it
                    live[s] = 0;
                    active[s] = 0;
                }
            }
        } else {
            solvePass = false;
            for (int s = 0; s < nS; s++) {
                active[s] = live[s];                         // stable: re-CHECK every live scenario
            }
            if (++loopIdx >= nLoops) {                       // ...and advance, as the stable branch does
                if (!seqChanged) {
                    loopIdx = nLoops;                        // a full sweep changed nothing -> converged
                } else {
                    seqChanged = false;
                    loopIdx = 0;
                }
            }
        }
    }
    // OLF_GPU_SLOT_DUMP=<prefix>: the traced scenario's (OLF_GPU_LOOP_TRACE_CTG) row targets, initial state and row
    // modes as the driver starts - after every host override and every device seed - one file per solve
    // (<prefix>.<k>), "<row> <target> <x> <mode>" with every double exact. Two runs of one scenario that should be
    // identical are diffed row by row: the rows that differ name the writer.
    if (!resume && ctx->loopTraceSlot >= 0 && ctx->loopTraceSlot < nS) {
        static const char* slotDump = std::getenv("OLF_GPU_SLOT_DUMP");
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
    static const bool olTimeEnv = std::getenv("OLF_GPU_PHASE_TIME") != nullptr;
    drv.olTime = olTimeEnv;
    const bool& olTime = drv.olTime;
    int& olPassSolved = drv.olPassSolved;
    olPassSolved = 0;
    bool olFirstSolveDone = resume;                          // a resumed call's passes are all re-solves
    int olTriggerKind = resume ? ctx->pausedLoopKind : GpuOuterLoopKind::NONE;  // the loop whose check made the next pass re-solve
    int& outer = drv.outer;
    for (outer = 0; outer < maxOuter && loopIdx < nLoops; outer++) {
    gOuterPasses++;
    if (olTime) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    const double olT0 = olTime ? olNowMs() : 0.0;
    const long long olAI0 = gActiveIterSum, olWI0 = gWidthIterSum;
    const bool olSolving = solvePass;
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
    // ---- batched fill for all active scenarios (J + F, or F-ONLY: the final residual fill and
    //      mode 4's solve-only iterations evaluate just the residual and skip the vals memset —
    //      the J buffer is never read before the next full fill re-zeroes it). A lambda so the
    //      LINE_SEARCH backtrack below can re-evaluate at a reduced step. ----
    // ---- IncrementalTransformerVoltageControl as a DRIVER PHASE (OLF order 8). OLF's own rule, per
    //      controlled bus: if |targetV - V| > halfDeadband, take dV/drho from the Jacobian at the converged
    //      state (unit RHS at the rho identity row, read the bus V entry), deltaR1 = diffV / sensi, move to
    //      the tap whose r1 is closest to r1 + deltaR1 within maxTapShift and the ratchet's allowed
    //      direction (MAX_DIRECTION_CHANGE 3), skip controllers with |sensi| < MIN_SENSI_FILTER (0.05).
    //      The sensitivity is the same forward solve GpuBatchIncrementalTvc used to make through
    //      solveBatchSensitivities — one host round trip per round — done here against the factors the
    //      driver already holds, with no host copy of the states. Single controller per controlled bus
    //      (the JNI upload refuses shared buses, which stay on the host orchestrator). ----
    // One unit-right-hand-side sensitivity solve on every scenario's CURRENT Jacobian (dVals, filled for all
    // scenarios by the caller; dB = the right-hand sides, dDx = the solutions), for the in-kernel outer loops.
    // The UBATCH mode (3) solves the whole batch in one call on the factors they hold - the analysis once per
    // context, the factorization with the first controller's solve. The robust mode (6) holds ONE matrix object
    // repointed per scenario and keeps no factors from one scenario to the next, so it factors and solves scenario
    // by scenario - a refactorization per (scenario, controller), affordable at the robust retry's chunk sizes and
    // the price of the loops running there at all: they used to report themselves STABLE in mode 6, so a scenario
    // retried in it (or a base load flow that fell into it) with a transformer, shunt, flow or secondary voltage
    // control out of its deadband got a wrong answer and nothing flagged it.
    std::vector<char>& loopRejected = drv.loopRejected;
    loopRejected.assign(nS, 0);                   // a device loop routed the scenario to the CPU this pass
    // ---- IncrementalShuntVoltageControl as a DRIVER PHASE (OLF order 10): deadband per controlled bus,
    //      dV/dB from a unit RHS at each needed controller's SHUNT_B identity row (the driver's own
    //      factors), OLF's adjustB round-robin per group, the section total pinned at the identity row
    //      and the static (dG, dB) through the shunt-override list. Groups with several controller
    //      shunts ARE handled (the round-robin runs in the group kernel). ----
    // ---- the FLOW-OBSERVED incremental loops as DRIVER PHASES: kindMask selects the phase controllers
    //      (OLF order 7: kinds 1+2) or the RPC controllers (OLF order 9: kind 0). Same skeleton as the
    //      voltage-observed phases; the observable is the controlled branch's flow gradient dotted with
    //      the unit-RHS solution (OLF's EquationTerm.calculateSensi), evaluated by the codegen'd branch
    //      functions on the device. ----
    // ---- CONTINUOUS PhaseControl as a DRIVER PHASE (OLF order 7), see cphCheckBatch. ----
    // ---- the DISCRETE voltage-control loops as DRIVER PHASES: TransformerVoltageControl (AFTER_GENERATOR /
    //      WITH_GENERATOR, OLF order 8) and ShuntVoltageControl (WITH_GENERATOR, OLF order 10). No sensitivities:
    //      OLF's decisions read the converged state only (deadband, clamp bounds, closest tap / section). ----
    // ---- SecondaryVoltageControl as a DRIVER PHASE (OLF order 4). The whole loop: the re-enable step and
    //      the per-zone trigger on the paused state's own qBuf, one unit-RHS solve per controlled bus on the
    //      driver's own factors for the K and pilot sensitivities, the small dense zone system per scenario,
    //      and the moved targets applied to the controlled buses' V rows (plus the switch-back comparison
    //      target for a pinned controller). Nothing goes back to the host, so the loop sits in OLF's order
    //      instead of after the sweep. ----
    if (solvePass) {
        newtonSolve(ctx, drv);
    }   // ---- end if (solvePass) ----
    if (olTime) {
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        double ms = olNowMs() - olT0;
        long long ai = gActiveIterSum - olAI0, wi = gWidthIterSum - olWI0;
        if (olSolving && !olFirstSolveDone) {
            gOlFirstSolveMs += ms; gOlFirstSolveScen += olPassSolved; gOlFirstLaneIters += ai; gOlFirstWidthIters += wi;
            gOlFirstPasses++;
            olFirstSolveDone = true;
        } else if (olSolving) {
            int k = olTriggerKind >= 0 && olTriggerKind < GpuOuterLoopKind::COUNT ? olTriggerKind : 0;
            gOlReSolveMs[k] += ms; gOlReSolveScen[k] += olPassSolved; gOlReSolveLaneIters[k] += ai;
            gOlReSolveWidthIters[k] += wi; gOlReSolvePasses[k]++;
        }
    }
    // the CHECK part of the pass, charged to the loop being checked when the pass body ends (break/continue too)
    struct OlCheckTimer {
        bool on; double t0; int kind;
        ~OlCheckTimer() {
            if (on && kind >= 0 && kind < GpuOuterLoopKind::COUNT) {
                gOlCheckMs[kind] += olNowMs() - t0;
                gOlCheckPasses[kind]++;
            }
        }
    } olCheck{olTime, olTime ? olNowMs() : 0.0, -1};

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

    // ---- REACTIVE LIMITS pass: switch any local PV controller over its Q limit to PQ ----
    // Only scenarios (re)solved this pass that converged can have new violations.
    for (int s = 0; s < nS; s++) {
        rlActive[s] = (wasActive[s] && res.status[s] == GpuAcSolverStatus::CONVERGED) ? 1 : 0;
    }
    CUDA_CHECK(cudaMemcpy(ctx->dRlActive.p, rlActive.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    // generator Q per controller = (Σ branch q + shunt q) at its BUS_V row + load Q. Compute the first
    // part by a fill into a ZEROED buffer: no −target init, no applyRowModeBatch, so qBuf[row] is the raw
    // bus reactive balance. Disabled branches are skipped (outage); dVals is scratch (the next inner
    // iteration memsets and refills it).
    CUDA_CHECK(cudaMemsetAsync(ctx->dQbuf.p, 0, nTot * sizeof(double), ctx->stream));
    // Reactive-limit Q buffer: physical bus-Q for PV-limit detection — keep the FROZEN routing
    // (dynReloc=0), the relocation only concerns the Newton residual, not this diagnostic Q.
    // Nothing reads a Jacobian here, and the fused J+F fill paid for one anyway: 46 of the 98 closed-branch
    // fills of a 485-contingency network.xiidm chunk, at 38 ms each, were this call. This kernel runs the fused
    // fill's own residual expressions and drops only the 24 Jacobian scatters (an angle-addition variant once
    // moved transformer-control cases by ~3e-5 kV: the bus balance must be the Newton residual's arithmetic).
    if (ctx->st->cb.elementCount) fillClosedBranchQbufBatch<<<grid(ctx->st->cb.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.elementCount, nS, n, ctx->nElements, ctx->cbOffset, ctx->st->cb.parameters.as<double>(), ctx->st->cb.indices.as<int>(), ctx->st->cb.destination.as<int>(), ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>());
    if (ctx->st->ob.elementCount) fillOpenBranchBatch<<<grid(ctx->st->ob.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.elementCount, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.parameters.as<double>(), ctx->st->ob.indices.as<int>(), ctx->st->ob.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), nullptr, nullptr, nullptr, 0, RelocTables{});
    if (ctx->st->sh.elementCount) fillShuntBatch<<<grid(ctx->st->sh.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.elementCount, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.parameters.as<double>(), ctx->st->sh.indices.as<int>(), ctx->st->sh.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), nullptr, nullptr, nullptr, 0, RelocTables{});
    if (ctx->st->nLoadModelTerms) fillLoadModelBatch<<<grid(ctx->st->nLoadModelTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nLoadModelTerms, nS, n, nnz, ctx->st->dLoadModelParameters.as<double>(), ctx->st->dLoadModelIndices.as<int>(), ctx->st->dLoadModelDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), nullptr, nullptr, nullptr);
    // Zero-impedance dummy transfers MUST be in the Q/P buffer too: the distributed-slack retain
    // reads the slack row's balance from dQbuf — without the coupler's dummy-P the DS sees a
    // phantom mismatch equal to the through-flow and distributes it FOREVER (measured: +392 MW
    // runaway on the first switch-close scenario until the per-scenario budget rejected it).
    if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), nullptr, nullptr, nullptr, 0, RelocTables{}, ctx->nZiTree, ctx->nZiTree ? ctx->dZiTreeSlotElemVal.as<int>() : nullptr, ctx->hasZiMasks ? ctx->dZiOffMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiOnMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiTreeMask.as<char>() : nullptr);
    // Remote-member OWN q (plan-B3): fill each member's own reactive injection into its (suppressed,
    // hence free) qBuf row so the RL kernel evaluates remote entries like local ones.
    if (ctx->nRmq > 0) {
        fillRemoteMemberQBatch<<<grid(ctx->nRmq * nS), BLOCK, 0, ctx->stream>>>(ctx->nRmq, nS, n,
                ctx->nElements, ctx->cbOffset, ctx->obOffset,
                ctx->dRmqRow.as<int>(), ctx->dRmqElem.as<int>(), ctx->dRmqSide.as<int>(),
                ctx->dRmqKind.as<int>(), ctx->dRmqWeight.as<double>(),
                ctx->st->cb.parameters.as<double>(), ctx->st->cb.indices.as<int>(), ctx->st->sh.parameters.as<double>(), ctx->st->sh.indices.as<int>(),
                ctx->st->ob.parameters.as<double>(), ctx->st->ob.indices.as<int>(),
                ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(),
                ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>());
    }
    if (ctx->nhv > 0 && ctx->dsSlackPhiRow >= 0) {           // a link on the slack bus, as OLF's slack mismatch sees it
        hvdcSlackBalanceBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, ctx->nhv, n, ctx->nElements, ctx->dsSlackPhiRow,
                ctx->hasDsSlackRowOv ? ctx->dDsSlackRowOv.as<int>() : nullptr,
                ctx->dHvBus1Row.as<int>(), ctx->dHvBus2Row.as<int>(), ctx->dHvBus1Phi.as<int>(), ctx->dHvBus2Phi.as<int>(),
                ctx->dHvElem.as<int>(), ctx->dHvIn.as<double>(), ctx->dHvStatus.as<int>(),
                ctx->dHvAppliedD1.as<double>(), ctx->dHvAppliedD2.as<double>(), ctx->dHvCtgOut.as<char>(),
                ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(), ctx->dQbuf.as<double>());
    }
    CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitched.p, 0, nS * sizeof(char), ctx->stream));
    if (outer == 0 && !resume && ctx->loopTraceSlot >= 0 && ctx->loopTraceSlot < nS) {
        // OLF_GPU_SLOT_DUMP, second point: the traced scenario's state and bus balances (qBuf) at the FIRST
        // outer-loop check, i.e. after the first solve - "<prefix>.<k>.pass0", "<row> <x> <qBuf>".
        static const char* slotDump0 = std::getenv("OLF_GPU_SLOT_DUMP");
        static int slotDumps0 = 0;
        if (slotDump0 != nullptr) {
            const size_t off = static_cast<size_t>(ctx->loopTraceSlot) * n;
            std::vector<double> hx(n), hq(n);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(hx.data(), ctx->dX.as<double>() + off, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hq.data(), ctx->dQbuf.as<double>() + off, n * sizeof(double), cudaMemcpyDeviceToHost));
            std::string path = std::string(slotDump0) + "." + std::to_string(slotDumps0++) + ".pass0";
            if (FILE* f = std::fopen(path.c_str(), "w")) {
                for (int r = 0; r < n; r++) {
                    std::fprintf(f, "%d %.17g %.17g\n", r, hx[r], hq[r]);
                }
                std::fclose(f);
            }
        }
    }
    // Apply ONLY the current phase's outer loop this pass (sequential, like OLF) — never both at once.
    bool inDsPhase = (loopOrder[loopIdx] == GpuOuterLoopKind::DISTRIBUTED_SLACK);
    olCheck.kind = loopOrder[loopIdx];
    olTriggerKind = loopOrder[loopIdx];                     // a re-solve in the next pass was triggered by this check
    traceBatchPass(ctx, drv, nLoops, solvePass);
    int loopKind = loopOrder[loopIdx];
    if (loopKind >= GpuOuterLoopKind::HOST_SECONDARY_VOLTAGE_CONTROL && loopKind <= GpuOuterLoopKind::HOST_SHUNT_VOLTAGE_CONTROL) {   // any LOOP_HOST_* kind: pause for the host
        // Hand the sequence back. Everything the device holds — x, targets, row modes, RL pins,
        // monitoring activations, absorbed slack, HVDC statuses — stays put in the context; only
        // these host-side locals need saving.
        ctx->pausedLoopIdx = loopIdx;
        ctx->pausedSeqChanged = seqChanged ? 1 : 0;
        ctx->pausedProdIters = prodIters;
        ctx->pausedLive = live;
        ctx->pausedActive = active;
        ctx->pausedStatus = res.status;
        ctx->driverPaused = 1;
        ctx->pausedLoopKind = loopKind;                      // WHICH host loop to run
        break;
    }
    if (loopKind == GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL) {
        tvcStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL) {
        shcStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::INCREMENTAL_PHASE_CONTROL) {
        if (ctx->nCph > 0) cphStep(ctx, drv); else flcStep(ctx, drv, controlledFlowMaskBit(GpuControlledFlow::ACTIVE_POWER) | controlledFlowMaskBit(GpuControlledFlow::CURRENT));  // continuous, or incremental phase
    } else if (loopKind == GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL) {
        flcStep(ctx, drv, controlledFlowMaskBit(GpuControlledFlow::REACTIVE_POWER));
    } else if (loopKind == GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL) {
        tvdStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL) {
        svdStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::SECONDARY_VOLTAGE_CONTROL) {
        svcStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::MONITORING_VOLTAGE) {
        monitoringStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION || loopKind == GpuOuterLoopKind::HVDC_AC_EMULATION_LIMITS) {
        hvdcStep(ctx, drv, loopKind);
    } else if (inDsPhase) {                                // distributed-slack outer loop: spread the slack mismatch
        dsCheck(ctx, drv);
    } else {                                               // reactive-limits outer loop: switch PV->PQ over Q limit
        rlCheck(ctx, drv, loopKind);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(rlSwitched.data(), ctx->dRlSwitched.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
    for (int s = 0; s < nS; s++) {
        if (loopRejected[s]) {                              // a device loop routed the scenario to the CPU fallback
            traceReject(s, "device-loop-routed", outer, 0);
            res.status[s] = GpuAcSolverStatus::REJECTED;
            live[s] = 0;
            active[s] = 0;
            rlSwitched[s] = 0;
            loopRejected[s] = 0;
        }
    }
    // OLF_GPU_SW_TRACE=<slot>: every controller this RL pass CHANGED for one scenario, with the
    // quantities its decision turned on. A scenario that ends one outer pass short of the CPU's
    // trajectory did so because one of these decisions went the other way; without the list there is
    // no way to name WHICH controller, and the margin of that one decision is the whole question.
    static const char* swTraceEnv = std::getenv("OLF_GPU_SW_TRACE");
    static const int swTraceSlotEnv = swTraceEnv != nullptr ? std::atoi(swTraceEnv) : -1;
    // the contingency-id slot of OLF_GPU_LOOP_TRACE_CTG wins (chunk composition moves fixed slots)
    const int swTraceSlot = ctx->loopTraceSlot >= 0 ? ctx->loopTraceSlot : swTraceSlotEnv;
    traceRlSwitches(ctx, drv, auditLog, rlAuditOn, swTraceSlot);
    traceRlState(ctx, drv, inDsPhase, reactiveLimitTol, rlSwitched);
    if (inDsPhase && (ctx->nds > 0 || ctx->dsSlackOnly) && dsRejectResidual) {
        // slackDistributionFailureBehavior != LEAVE_ON_SLACK_BUS: a scenario whose participants all
        // saturated must NOT be reported converged with the residual absorbed at the slack — route it
        // to the CPU fallback (status -2), which replays it with the run's real failure behavior.
        std::vector<char> dsResidual(nS);
        CUDA_CHECK(cudaMemcpy(dsResidual.data(), ctx->dDsResidual.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            if (dsResidual[s] && live[s]) {
                traceReject(s, "ds-residual-saturated", outer, 0);
                res.status[s] = GpuAcSolverStatus::REJECTED;
                active[s] = 0;
                live[s] = 0;
                rlSwitched[s] = 0;
            }
        }
    }
    int nChanged = 0;
    for (int s = 0; s < nS; s++) {
        if (rlSwitched[s]) {
            nChanged++;
        }
    }
    // OLF_GPU_LOOP_TRACE=<slot>: the device side of OLF's OL_CHECK - which loop this pass ran, whether
    // it found the slot UNSTABLE (a re-solve follows) and whether the whole phase is still unstable.
    // Two arms that agree on every residual and still stop apart differ in this SEQUENCE.
    static const char* loopTraceEnv = std::getenv("OLF_GPU_LOOP_TRACE");
    // a slot resolved per chunk from a contingency id (setBatchLoopTraceSlot) wins over the env slot:
    // chunk composition moves with the resolved chunk size, so a fixed slot traces a different
    // scenario from one run to the next (IZUZ: slot 170, then 18)
    const int loopTraceSlot = ctx->loopTraceSlot >= 0 ? ctx->loopTraceSlot : (loopTraceEnv != nullptr ? std::atoi(loopTraceEnv) : -1);
    traceLoopSlot(ctx, drv, loopTraceSlot, nChanged, rlSwitched, solvePass);
    if (nChanged > 0) {                                     // this outer loop is still unstable
        for (int s = 0; s < nS; s++) {
            if (rlSwitched[s] && ++prodIters[s] >= RL_MAX_OUTER) {
                // This scenario exhausted ITS outer-iteration budget mid-sequence. Never report it
                // converged on a half-finished RL/DS fixpoint — reject to the CPU fallback, which
                // replays it under OLF's own maxOuterLoopIterations/UNSTABLE handling.
                traceReject(s, "outer-loop-budget", outer, 0);
                res.status[s] = GpuAcSolverStatus::REJECTED;
                rlSwitched[s] = 0;
                live[s] = 0;
            }
            active[s] = rlSwitched[s];                      // re-solve only the scenarios it changed, stay in this phase
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
    //      answer every UNPINNED controller sits inside its reactive limits (within the switch
    //      tolerance). A device scenario that returns with an unpinned controller beyond its limit
    //      did not reach that fixpoint - it stopped one check short - and that is a defect no
    //      comparison with the CPU is needed to see. Re-fills the diagnostic q at the final state
    //      for every live scenario, refreshes the curve limits at the final targets, and prints one
    //      line per violating (scenario, entry).
    traceRlAudit(ctx, drv, auditLog);
    if (rlStateOut != nullptr && ctx->nrl > 0) {             // export per-(controller, scenario) RL state
        std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS), sc(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sc.data(), ctx->dRlSwitchCount.p, sc.size(), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < lt.size(); k++) {
            rlStateOut[k] = rlState(lt[k], sc[k]);
        }
    }
    tracePhaseTime(ctx, drv);
    CUDA_CHECK(cudaMemcpy(res.x.data(), ctx->dX.p, nTot * sizeof(double), cudaMemcpyDeviceToHost));
    return res;
}

}  // namespace olfgpu
