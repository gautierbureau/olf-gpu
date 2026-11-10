// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A batched context's life: the cuDSS matrices of the UBATCH and robust modes or our batched LU's factors, the
// creation on a shared structure, and the per-call upload of the scenario overrides. See batch_context.cuh.
#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_state_vector_scaling_mode.cuh"
#include "../common/codes/gpu_distributed_slack_factor_mode.cuh"
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../common/codes/gpu_two_sides.cuh"
#include "batch_context.cuh"
#include "driver_steps.cuh"
#include "linear_solver_settings.cuh"
#include "scenario_override_tables.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

void setupBatchMatricesUbatch(BatchSolverContext* ctx) {
    int n = ctx->n;
    int nScenarios = ctx->nScenarios;
    // ---- UBATCH: ONE per-block CSR (n, nnz, shared rowPtr/colIdx) with batched value/RHS pointers;
    //      cuDSS analyzes a single block (cheap) and factorizes/solves all nS via UBATCH_SIZE. ----
    {
        int ub = nScenarios;
        CUDSS_CHECK(cudssConfigSet(ctx->solver.config, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof(ub)));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->uniformBatchSystemMatrix, n, n, ctx->nnz,
                ctx->structure->dRowPtr.as<int>(), nullptr, ctx->structure->dColIdx.as<int>(), ctx->dVals.as<double>(),
                OLF_CUDSS_CSR_TYPES,
                CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->uniformBatchRightHandSideMatrix, n, 1, n, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->uniformBatchSolutionMatrix, n, 1, n, ctx->dDx.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        // adaptive refinement: the residual and the correction, the same single-block dense shape as B and X
        if (!ctx->dR.p) {
            ctx->dR.alloc(static_cast<size_t>(n) * ctx->nScenarios * sizeof(double));
            ctx->dDelta.alloc(static_cast<size_t>(n) * ctx->nScenarios * sizeof(double));
            // zeroed: a solve covers UBATCH_SIZE = nScenarios lanes while a short last chunk fills nS < nScenarios
            // of them - the padding lanes' residual must be a clean 0, not whatever the allocation held
            CUDA_CHECK(cudaMemset(ctx->dR.p, 0, static_cast<size_t>(n) * ctx->nScenarios * sizeof(double)));
            CUDA_CHECK(cudaMemset(ctx->dDelta.p, 0, static_cast<size_t>(n) * ctx->nScenarios * sizeof(double)));
        }
        if (!ctx->dIterativeRefinementScore.p) {
            ctx->dIterativeRefinementScore.alloc(static_cast<size_t>(ctx->nScenarios) * sizeof(double));
        }
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->uniformBatchResidualMatrix, n, 1, n, ctx->dR.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->uniformBatchDeltaMatrix, n, 1, n, ctx->dDelta.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    }
}

// BATCHED_LU: our batched LU's factors for every slot, and the refinement buffers the cuDSS batch has.
void setupBatchedLu(BatchSolverContext* ctx) {
    ctx->batchedLu = std::make_unique<BatchedLu>();
    createBatchedLu(*ctx->batchedLu, luPlanOf(*ctx->structure), ctx->nScenarios);
    const size_t vectorBytes = static_cast<size_t>(ctx->n) * ctx->nScenarios * sizeof(double);
    ctx->dR.alloc(vectorBytes);                              // zeroed: a lane that never solves keeps a clean 0
    ctx->dDelta.alloc(vectorBytes);
    ctx->dIterativeRefinementScore.alloc(static_cast<size_t>(ctx->nScenarios) * sizeof(double));
    if (ctx->structure->nZeroImpedanceBranch > 0) {
        ctx->dInternalRefinementResidual.alloc(vectorBytes);
        ctx->dInternalRefinementCorrection.alloc(vectorBytes);
    }
}

void setupBatchMatricesFallback(BatchSolverContext* ctx) {
    int n = ctx->n;
    // ---- fallback: a single matrix repointed per scenario inside the solve loop ----
    CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->solver.systemMatrix, n, n, ctx->nnz,
                                     ctx->structure->dRowPtr.as<int>(), nullptr, ctx->structure->dColIdx.as<int>(), ctx->dVals.as<double>(),
                                     OLF_CUDSS_CSR_TYPES,
                                     CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&ctx->solver.rightHandSideMatrix, n, 1, n, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&ctx->solver.solutionMatrix, n, 1, n, ctx->dDx.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
}

BatchSolverContext* createBatchContext(const SharedStructurePtr& structure, int nScenarios, int mode,
                                       int nReactiveLimitEntries, const int* reactiveLimitVoltageRow, const double* reactiveLimitMinQ,
                                       const double* reactiveLimitMaxQ, const double* reactiveLimitLoadQ, const double* reactiveLimitTargetV,
                                       const signed char* reactiveLimitFlags, const double* rowSlope,
                                       int nDistributedSlackBuses, const int* distributedSlackBusPhiRow, const double* distributedSlackBusFactor,
                                       int slackBusPhiRow, double slackBusTargetP,
                                       const double* distributedSlackBusBaseTargetP, const double* distributedSlackBusMaxDelta, const double* distributedSlackBusMinDelta,
                                       int distributedSlackFactorMode, const double* distributedSlackBusMarginUp, const double* distributedSlackBusMarginDown,
                                       double slackBusMarginUp, double slackBusMarginDown) {
    auto ctx = std::make_unique<BatchSolverContext>();
    ctx->structure = structure;
    int n = structure->n;
    ctx->n = n;
    ctx->nnz = structure->nnz;
    ctx->nScenarios = nScenarios;
    ctx->mode = mode;
    ctx->nElements = structure->nElements;
    ctx->closedBranchOffset = structure->closedBranchOffset;
    ctx->openBranchOffset = structure->openBranchOffset;
    ctx->shuntOffset = structure->shuntOffset;
    ctx->hvdcAcEmulationOffset = structure->hvdcAcEmulationOffset;
    ctx->nDistrQ = structure->nDistrQ;
    ctx->nDistrRho = structure->nDistrRho;
    ctx->nDistrShuntB = structure->nDistrShuntB;
    ctx->nZeroImpedanceBranch = structure->nZeroImpedanceBranch;
    const std::vector<int>& rowMap = structure->rowMap;
    const std::vector<int>& rowPtr = structure->rowPtr;
    const std::vector<int>& colIdx = structure->colIdx;

    // ---- per-scenario device state ----
    ctx->layout = mode == GpuLinearSolveMode::BATCHED_LU ? interleavedLayout(nScenarios) : scenarioMajorLayout(nScenarios);
    size_t nS = static_cast<size_t>(n) * nScenarios;
    size_t nnzS = static_cast<size_t>(ctx->nnz) * ctx->layout.stride;    // an interleaved stride rounds the width up
    ctx->dX.alloc(nS * sizeof(double));
    ctx->dXLastConv.alloc(nS * sizeof(double));
    ctx->dTarget.alloc(nS * sizeof(double));
    ctx->dF.alloc(nS * sizeof(double));
    ctx->dB.alloc(nS * sizeof(double));
    ctx->dDx.alloc(nS * sizeof(double));
    ctx->dVals.alloc(nnzS * sizeof(double));
    if (ctx->layout.interleaved) {
        ctx->dFillMismatch.alloc(static_cast<size_t>(n) * ctx->layout.stride * sizeof(double));
    }
    ctx->dNorms.alloc(static_cast<size_t>(nScenarios) * 4 * sizeof(double));
    ctx->dMode.alloc(nS * sizeof(int));
    ctx->dDisabled.alloc(static_cast<size_t>(ctx->nElements) * nScenarios * sizeof(char));
    ctx->dActive.alloc(static_cast<size_t>(nScenarios) * sizeof(char));
    ctx->dIdentityLane.alloc(static_cast<size_t>(nScenarios) * sizeof(char));
    CUDA_CHECK(cudaMemset(ctx->dIdentityLane.p, 0, static_cast<size_t>(nScenarios) * sizeof(char)));

    // ---- reactive-limits pack: the controller BUS_V rows mapped through the diagonal-pairing
    //      permutation (= identity in the batch path, no families, but mapped for correctness) ----
    ctx->nReactiveLimitEntries = nReactiveLimitEntries;
    ctx->dQbuf.alloc(static_cast<size_t>(n) * nScenarios * sizeof(double));
    ctx->dQbufClosed.alloc(static_cast<size_t>(n) * ctx->layout.stride * sizeof(double));
    ctx->dOuterLoopCheckActive.alloc(static_cast<size_t>(nScenarios) * sizeof(char));
    ctx->dOuterLoopUnstable.alloc(static_cast<size_t>(nScenarios) * sizeof(char));
    // The seed launch of updatePilotTargetVEquationBatch / updateDistrQTargetsBatch (before the first solve) READS
    // dOuterLoopCheckActive and relies on "not active yet" = 0; the RL pass is what first writes it. Explicit, not
    // left to the allocator's zeroing.
    CUDA_CHECK(cudaMemset(ctx->dOuterLoopCheckActive.p, 0, static_cast<size_t>(nScenarios) * sizeof(char)));
    CUDA_CHECK(cudaMemset(ctx->dOuterLoopUnstable.p, 0, static_cast<size_t>(nScenarios) * sizeof(char)));
    // OUTSIDE the `nReactiveLimitEntries > 0` block: a run with reactive limits DISABLED has no entries at all, and that is
    // exactly when every pilot-self group needs the gate — its pilot can never switch to PQ, so it controls throughout.
    // Built inside, the gate did not exist for such a run: the pilots' raw q was emitted at their carrier rows
    // on top of the DISTR_Q terms and rte6515's base (limits off) started at |f(x0)| 117.908 against OLF's
    // 117.845 and never converged.
    // Derive the PILOT-SELF groups from the candidate table itself: the fallback row repeated among the
    // member rows means the controlled bus is one of its own controllers. The pilot's RL entry is the one
    // whose (row-mapped) V row is that row; without an entry the pilot can never switch to PQ and the group needs
    // no gate. nReactiveRelocationGroups is small (groups), nReactiveLimitEntries a few thousand, and this runs once per context.
    int nRelocationGroups = ctx->structure != nullptr ? ctx->structure->nReactiveRelocationGroups : 0;
    if (nRelocationGroups > 0) {
        std::vector<int> pilotSelfReactiveLimitEntry(nRelocationGroups, -1);
        for (int g = 0; g < nRelocationGroups; g++) {
            int p0 = ctx->structure->hReactiveRelocationGroupStart[g];
            int count = ctx->structure->hReactiveRelocationGroupStart[g + 1] - p0;
            int fallbackRow = ctx->structure->hReactiveRelocationCandidateRows[p0];
            bool pilotSelf = false;
            for (int c = 1; c < count; c++) {
                if (ctx->structure->hReactiveRelocationCandidateRows[p0 + c] == fallbackRow) {
                    pilotSelf = true;
                    break;
                }
            }
            if (!pilotSelf) {
                continue;                                // not a pilot-self group: -1, no gate (raw q as always)
            }
            // PILOT_SELF_NO_REACTIVE_LIMIT_ENTRY (-2), not -1: a pilot-self pilot with NO reactive-limit entry can never SWITCH TO PQ
            // (limits off for the run, or an empty q range), so it controls for the whole solve and its raw q
            // must ALWAYS be skipped — the gate is needed MORE, not less, than for one that can switch. Read as
            // "no gate" (-1, the non-pilot-self value) it was emitted at the carrier row on top of the very
            // DISTR_Q terms it feeds: with reactive limits disabled on rte6515 the base started at
            // |f(x0)| 117.908 against OLF's 117.845 and the device's load flow never converged.
            pilotSelfReactiveLimitEntry[g] = PILOT_SELF_NO_REACTIVE_LIMIT_ENTRY;
            ctx->hasPilotSelf = true;
            for (int i = 0; i < nReactiveLimitEntries; i++) {
                if (rowMap[reactiveLimitVoltageRow[i]] == fallbackRow) {
                    pilotSelfReactiveLimitEntry[g] = i;
                    break;
                }
            }
        }
        if (ctx->hasPilotSelf) {
            ctx->dPilotSelfReactiveLimitEntry.upload(pilotSelfReactiveLimitEntry.data(), pilotSelfReactiveLimitEntry.size() * sizeof(int));
            ctx->dPilotSelfCarrierSkip.alloc(static_cast<size_t>(nRelocationGroups) * ctx->nScenarios * sizeof(char));
            if (traceOn(Trace::RL_TRACE)) {
                int count = 0, noEntry = 0;
                for (int g = 0; g < nRelocationGroups; g++) {
                    count += pilotSelfReactiveLimitEntry[g] >= 0 ? 1 : 0;
                    noEntry += pilotSelfReactiveLimitEntry[g] == PILOT_SELF_NO_REACTIVE_LIMIT_ENTRY ? 1 : 0;
                }
                std::fprintf(stderr, "PILOTSELF_GATE groups=%d(+%d without a reactive-limit entry) of %d nReactiveLimitEntries=%d\n",
                             count, noEntry, nRelocationGroups, nReactiveLimitEntries);
            }
        }
    }
    if (nReactiveLimitEntries > 0) {
        std::vector<int> reactiveLimitVoltageRowMapped(nReactiveLimitEntries);
        for (int i = 0; i < nReactiveLimitEntries; i++) {
            reactiveLimitVoltageRowMapped[i] = rowMap[reactiveLimitVoltageRow[i]];
        }
        ctx->dReactiveLimitVoltageRow.upload(reactiveLimitVoltageRowMapped.data(), nReactiveLimitEntries * sizeof(int));
        // A bus whose power equations the COUPLER-TREE relocation moved onto its parent coupler's dummy columns has
        // its reactive balance — and its BUS_TARGET_Q, which GpuAcActivationExtractor already puts there — at that
        // coupler's DUMMY_Q row, while its OWN V row carries the coupler's ZERO_V. The reactive-limit pass must read
        // q and make the PV -> PQ switch at THAT row; its VOLTAGE reads stay on the bus's own row (dReactiveLimitVoltageRow). Without the split
        // the check compared a voltage residual against reactive limits: measured on rte6515's full busbar scope
        // with the relocation enabled, entries read -78.0 and 4767.3 MVAr and 20 buses were switched to PQ that OLF never
        // switches, after which the solve with those switches burned all 30 iterations.
        std::vector<int> reactiveLimitReactivePowerRowMapped(reactiveLimitVoltageRowMapped);
        if (ctx->nZeroImpedanceBranch > 0 && ctx->structure->dZeroImpedanceBranchIndices.p != nullptr) {
            std::vector<int> zidx(static_cast<size_t>(ctx->nZeroImpedanceBranch) * 11);
            CUDA_CHECK(cudaMemcpy(zidx.data(), ctx->structure->dZeroImpedanceBranchIndices.p, zidx.size() * sizeof(int), cudaMemcpyDeviceToHost));
            std::map<int, int> qRowOfVRow;
            for (int e = 0; e < ctx->nZeroImpedanceBranch; e++) {
                const int* z = zidx.data() + static_cast<size_t>(e) * 11;
                int childSide = zeroImpedanceDeadEndSideOf(z);
                int ownVCol = childSide == GpuTwoSides::ONE ? z[6] : (childSide == GpuTwoSides::TWO ? z[7] : -1);
                if (childSide == GpuTwoSides::NONE || z[5] < 0 || ownVCol < 0) {
                    continue;
                }
                qRowOfVRow[rowMap[ownVCol]] = rowMap[z[5]];
            }
            int moved = 0;
            for (int i = 0; i < nReactiveLimitEntries; i++) {
                auto it = qRowOfVRow.find(reactiveLimitVoltageRowMapped[i]);
                if (it != qRowOfVRow.end()) {
                    reactiveLimitReactivePowerRowMapped[i] = it->second;
                    moved++;
                }
            }
            if (moved > 0 && traceOn(Trace::RL_TRACE)) {
                std::fprintf(stderr, "RL_QROW %d of %d entries follow the coupler relocation\n", moved, nReactiveLimitEntries);
            }
        }
        ctx->dReactiveLimitReactivePowerRow.upload(reactiveLimitReactivePowerRowMapped.data(), nReactiveLimitEntries * sizeof(int));
        ctx->dReactiveLimitMinQ.upload(reactiveLimitMinQ, nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitMaxQ.upload(reactiveLimitMaxQ, nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitLoadQ.upload(reactiveLimitLoadQ, nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitTargetV.upload(reactiveLimitTargetV, nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitFlags.upload(reactiveLimitFlags, nReactiveLimitEntries * sizeof(char));
        ctx->dReactiveLimitSavedTarget.alloc(static_cast<size_t>(nReactiveLimitEntries) * nScenarios * sizeof(double));
        ctx->dReactiveLimitCheckDecision.alloc(static_cast<size_t>(nReactiveLimitEntries) * nScenarios * sizeof(char));
        // OLF_GPU_SW_LOG=<slot>: record one scenario's reactive-limit decisions ON THE DEVICE (one
        // atomicAdd per decision, copied out once at the end) instead of reading the whole batch's
        // buffers over PCIe every pass. The host-side traces cost hundreds of MB per pass, which
        // serializes the stream and changes the very kernel scheduling whose effect is being traced:
        // a run-to-run flip disappears while you watch it.
        if (traceOn(Trace::SW_LOG)) {
            ctx->dReactiveLimitsDecisionLog.alloc(REACTIVE_LIMITS_DECISION_LOG_CAPACITY * REACTIVE_LIMITS_DECISION_LOG_WIDTH * sizeof(double));
            ctx->dReactiveLimitsDecisionLogCount.alloc(sizeof(int));
            CUDA_CHECK(cudaMemset(ctx->dReactiveLimitsDecisionLogCount.p, 0, sizeof(int)));
        }
        ctx->dReactiveLimitSavedRowMode.alloc(static_cast<size_t>(nReactiveLimitEntries) * nScenarios * sizeof(char));
        ctx->dReactiveLimitSwitchedRow.alloc(static_cast<size_t>(nReactiveLimitEntries) * nScenarios * sizeof(int));
        ctx->dReactiveLimitRemainingPvBusCount.alloc(static_cast<size_t>(nScenarios) * sizeof(int));
        ctx->dReactiveLimitQLimitType.alloc(static_cast<size_t>(nReactiveLimitEntries) * nScenarios * sizeof(char));
        ctx->dReactiveLimitPvPqSwitchCount.alloc(static_cast<size_t>(nReactiveLimitEntries) * nScenarios * sizeof(char));
        std::vector<double> svcZero(nReactiveLimitEntries, 0.0);              // neutral until setBatchReactiveLimitSvcSusceptanceLimits installs
        ctx->dReactiveLimitSvcMinB.upload(svcZero.data(), nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitSvcMaxB.upload(svcZero.data(), nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitBaseVSquared.upload(svcZero.data(), nReactiveLimitEntries * sizeof(double));
    }
    ctx->dRowSlope.upload(rowSlope, n * sizeof(double));
    ctx->nDistributedSlackBuses = nDistributedSlackBuses;
    ctx->slackBusTargetP = slackBusTargetP;
    ctx->distributedSlackFactorMode = distributedSlackFactorMode;
    ctx->slackBusMarginUp = slackBusMarginUp;
    ctx->slackBusMarginDown = slackBusMarginDown;
    if (distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN && nDistributedSlackBuses > 0) {
        ctx->dDistributedSlackBusMarginUp.upload(distributedSlackBusMarginUp, nDistributedSlackBuses * sizeof(double));
        ctx->dDistributedSlackBusMarginDown.upload(distributedSlackBusMarginDown, nDistributedSlackBuses * sizeof(double));
    }
    ctx->slackBusPhiRow = slackBusPhiRow >= 0 ? rowMap[slackBusPhiRow] : -1;
    ctx->slackOnlyDistribution = (nDistributedSlackBuses == 0 && ctx->slackBusPhiRow >= 0);
    if (ctx->slackOnlyDistribution) {
        ctx->dSlackOnlyAbsorbed.alloc(static_cast<size_t>(nScenarios) * sizeof(double));
    }
    if (nDistributedSlackBuses > 0) {
        std::vector<int> distributedSlackBusPhiRowMapped(nDistributedSlackBuses);
        for (int i = 0; i < nDistributedSlackBuses; i++) {
            distributedSlackBusPhiRowMapped[i] = rowMap[distributedSlackBusPhiRow[i]];
        }
        ctx->dDistributedSlackBusPhiRow.upload(distributedSlackBusPhiRowMapped.data(), nDistributedSlackBuses * sizeof(int));
        ctx->dDistributedSlackBusFactor.upload(distributedSlackBusFactor, nDistributedSlackBuses * sizeof(double));
        ctx->dDistributedSlackBusBaseTargetP.upload(distributedSlackBusBaseTargetP, nDistributedSlackBuses * sizeof(double));
        ctx->dDistributedSlackBusMaxDelta.upload(distributedSlackBusMaxDelta, nDistributedSlackBuses * sizeof(double));
        ctx->dDistributedSlackBusMinDelta.upload(distributedSlackBusMinDelta, nDistributedSlackBuses * sizeof(double));
        double factorSum = 0.0;
        for (int i = 0; i < nDistributedSlackBuses; i++) {
            factorSum += distributedSlackBusFactor[i];
        }
        ctx->distributedSlackFactorSum = factorSum > 0 ? factorSum : 1.0;
        ctx->dSlackRetainedShare.alloc(static_cast<size_t>(nScenarios) * sizeof(double));
        ctx->dDistributedSlackLastPassAccounting.alloc(static_cast<size_t>(nScenarios) * 8 * sizeof(double));   // see the gateLast write
        ctx->dSlackRetainedShareApplied.alloc(static_cast<size_t>(nScenarios) * sizeof(double));
    }

    // ---- cuDSS handle/config/data (shared by both solve paths) ----
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    CUDSS_CHECK(cudssCreate(&ctx->solver.handle));
    CUDSS_CHECK(cudssSetStream(ctx->solver.handle, ctx->stream));
    CUDSS_CHECK(cudssConfigCreate(&ctx->solver.config));
    int ir = ctx->structure->nZeroImpedanceBranch > 0 ? configuredIterativeRefinementBaseSteps() : ITERATIVE_REFINEMENT_STEPS_WITHOUT_COUPLERS;
    CUDSS_CHECK(cudssConfigSet(ctx->solver.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)));
    if (mode == GpuLinearSolveMode::ROBUST) {
        applyRobustSingleMatrixConfig(ctx->solver.config);       // its own ordering: never set the ordering twice
    } else {
        applyReorderAlg(ctx->solver.config, ctx->structure->nZeroImpedanceBranch > 0);
    }
    CUDSS_CHECK(cudssDataCreate(ctx->solver.handle, &ctx->solver.data));

    if (mode == GpuLinearSolveMode::UBATCH) {
        setupBatchMatricesUbatch(ctx.get());
    } else if (mode == GpuLinearSolveMode::BATCHED_LU) {
        setupBatchedLu(ctx.get());
    } else {
        setupBatchMatricesFallback(ctx.get());
    }
    return ctx.release();
}

void uploadBatchOverrides(BatchSolverContext* ctx, int stateVectorScalingMode, double maxDv, double maxDphi,
                          const int* varType, int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                          int nBaseClosedZeroImpedanceBranches, int nZeroImpedanceBranchOpenOverrides, const int* zeroImpedanceBranchOpenScenarioElement,
                          int nZeroImpedanceBranchCloseOverrides, const int* zeroImpedanceBranchCloseScenarioElement,
                          int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues) {
    int n = ctx->n;
    int nS = ctx->nScenarios;
    bool clampStep = stateVectorScalingMode == GpuStateVectorScalingMode::MAX_VOLTAGE_CHANGE && varType != nullptr;
    if (clampStep && !ctx->dVarType.p) {
        ctx->dVarType.upload(varType, static_cast<size_t>(n) * sizeof(int));
        ctx->dStepFactor.alloc(static_cast<size_t>(nS) * sizeof(double));
    }
    if (stateVectorScalingMode == GpuStateVectorScalingMode::LINE_SEARCH) {                                       // LINE_SEARCH backtracking buffers
        if (!ctx->dLineSearchPreviousX.p) ctx->dLineSearchPreviousX.alloc(static_cast<size_t>(n) * nS * sizeof(double));
        if (!ctx->dStepFactor.p) ctx->dStepFactor.alloc(static_cast<size_t>(nS) * sizeof(double));
    }
    ctx->clampStep = clampStep;
    ctx->maxDv = maxDv;
    ctx->maxDphi = maxDphi;
    // Per-scenario shunt admittance overrides (section shifts / reconnection), PACK-indexed elements (shunt family
    // index, not the global element index).
    installShuntAdmittanceOverrides(ctx, nShuntAdmittanceOverrides, shuntAdmittanceOverrideScenarioElement, shuntAdmittanceOverrideDelta);
    ctx->nBaseClosedZeroImpedanceBranches = nBaseClosedZeroImpedanceBranches > 0 ? nBaseClosedZeroImpedanceBranches : ctx->nZeroImpedanceBranch;         // default: every zero-impedance element always enabled
    if (ctx->nZeroImpedanceBranch > 0) {                                      // the dense (slot, element) forms (see zeroImpedanceBranchOpenMask)
        const size_t width = static_cast<size_t>(ctx->nZeroImpedanceBranch);
        ctx->zeroImpedanceBranchOpenMask.installEntries(nS, ctx->installSlots, width, nZeroImpedanceBranchOpenOverrides,
                                                        zeroImpedanceBranchOpenScenarioElement, nullptr, 1);
        ctx->zeroImpedanceBranchCloseMask.installEntries(nS, ctx->installSlots, width, nZeroImpedanceBranchCloseOverrides,
                                                         zeroImpedanceBranchCloseScenarioElement, nullptr, 1);
        if (ctx->spanningTreeOverrideMask.width() != width) {   // no tree override installed yet: the extraction's tree
            ctx->spanningTreeOverrideMask.installEntries(nS, {}, width, 0, nullptr, nullptr);
        }
        if (traceOn(Trace::PHASE_TIME)) {
            std::fprintf(stderr, "ZI_LISTS nZeroImpedanceBranch=%d nScenarios=%d nZeroImpedanceBranchOpenOverrides=%d nZeroImpedanceBranchCloseOverrides=%d\n", ctx->nZeroImpedanceBranch, ctx->nScenarios, nZeroImpedanceBranchOpenOverrides, nZeroImpedanceBranchCloseOverrides);
        }
    }
    installClosedBranchParameterOverrides(ctx, nClosedBranchParameterOverrides, closedBranchParameterOverrideScenarioElement, closedBranchParameterOverrideValues);
}

}  // namespace olfgpu
