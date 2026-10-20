// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A batched context's life: the cuDSS matrices of the UBATCH and robust modes, the creation on a shared structure,
// and the per-call upload of the scenario overrides. See batch_context.cuh.
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
#include "../common/codes/gpu_reactive_limits_entry_flag.cuh"
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "batch_context.cuh"
#include "driver_steps.cuh"
#include "linear_solver_settings.cuh"

namespace olfgpu {

void setupBatchMatricesUbatch(BatchSolverContext* ctx) {
    int n = ctx->n;
    int nScenarios = ctx->nScenarios;
    // ---- UBATCH: ONE per-block CSR (n, nnz, shared rowPtr/colIdx) with batched value/RHS pointers;
    //      cuDSS analyzes a single block (cheap) and factorizes/solves all nS via UBATCH_SIZE. ----
    {
        int ub = nScenarios;
        CUDSS_CHECK(cudssConfigSet(ctx->ds.config, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof(ub)));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->matUb, n, n, ctx->nnz,
                ctx->st->dRowPtr.as<int>(), nullptr, ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(),
                OLF_CUDSS_CSR_TYPES,
                CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbB, n, 1, n, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbX, n, 1, n, ctx->dDx.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        // adaptive refinement: the residual and the correction, the same single-block dense shape as B and X
        if (!ctx->dR.p) {
            ctx->dR.alloc(static_cast<size_t>(n) * ctx->nScenarios * sizeof(double));
            ctx->dDelta.alloc(static_cast<size_t>(n) * ctx->nScenarios * sizeof(double));
            // zeroed: a solve covers UBATCH_SIZE = nScenarios lanes while a short last chunk fills nS < nScenarios
            // of them - the padding lanes' residual must be a clean 0, not whatever the allocation held
            CUDA_CHECK(cudaMemset(ctx->dR.p, 0, static_cast<size_t>(n) * ctx->nScenarios * sizeof(double)));
            CUDA_CHECK(cudaMemset(ctx->dDelta.p, 0, static_cast<size_t>(n) * ctx->nScenarios * sizeof(double)));
        }
        if (!ctx->dIrScore.p) {
            ctx->dIrScore.alloc(static_cast<size_t>(ctx->nScenarios) * sizeof(double));
        }
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbR, n, 1, n, ctx->dR.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbDelta, n, 1, n, ctx->dDelta.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    }
}

void setupBatchMatricesFallback(BatchSolverContext* ctx) {
    int n = ctx->n;
    // ---- fallback: a single matrix repointed per scenario inside the solve loop ----
    CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->ds.matA, n, n, ctx->nnz,
                                     ctx->st->dRowPtr.as<int>(), nullptr, ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(),
                                     OLF_CUDSS_CSR_TYPES,
                                     CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&ctx->ds.matB, n, 1, n, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&ctx->ds.matX, n, 1, n, ctx->dDx.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
}

BatchSolverContext* createBatchContext(const SharedStructurePtr& st, int nScenarios, int mode,
                                       int nReactiveLimitEntries, const int* reactiveLimitVoltageRow, const double* reactiveLimitMinQ,
                                       const double* reactiveLimitMaxQ, const double* reactiveLimitLoadQ, const double* reactiveLimitTargetV,
                                       const signed char* reactiveLimitFlags, const double* rowSlope,
                                       int nDistributedSlackBuses, const int* distributedSlackBusPhiRow, const double* distributedSlackBusFactor,
                                       int slackBusPhiRow, double slackBusTargetP,
                                       const double* distributedSlackBusBaseTargetP, const double* dsMaxDelta, const double* dsMinDelta,
                                       int distributedSlackFactorMode, const double* distributedSlackBusMarginUp, const double* distributedSlackBusMarginDown,
                                       double slackBusMarginUp, double slackBusMarginDown) {
    auto ctx = std::make_unique<BatchSolverContext>();
    ctx->st = st;
    int n = st->n;
    ctx->n = n;
    ctx->nnz = st->nnz;
    ctx->nScenarios = nScenarios;
    ctx->mode = mode;
    ctx->nElements = st->nElements;
    ctx->closedBranchOffset = st->closedBranchOffset;
    ctx->openBranchOffset = st->openBranchOffset;
    ctx->shuntOffset = st->shuntOffset;
    ctx->hvdcAcEmulationOffset = st->hvdcAcEmulationOffset;
    ctx->nDistrQ = st->nDistrQ;
    ctx->nDistrRho = st->nDistrRho;
    ctx->nDistrShuntB = st->nDistrShuntB;
    ctx->nZeroImpedanceBranch = st->nZeroImpedanceBranch;
    const std::vector<int>& rowMap = st->rowMap;
    const std::vector<int>& rowPtr = st->rowPtr;
    const std::vector<int>& colIdx = st->colIdx;

    // ---- per-scenario device state ----
    size_t nS = static_cast<size_t>(n) * nScenarios;
    size_t nnzS = static_cast<size_t>(ctx->nnz) * nScenarios;
    ctx->dX.alloc(nS * sizeof(double));
    ctx->dXLastConv.alloc(nS * sizeof(double));
    ctx->dTarget.alloc(nS * sizeof(double));
    ctx->dF.alloc(nS * sizeof(double));
    ctx->dB.alloc(nS * sizeof(double));
    ctx->dDx.alloc(nS * sizeof(double));
    ctx->dVals.alloc(nnzS * sizeof(double));
    ctx->dNorms.alloc(static_cast<size_t>(nScenarios) * 4 * sizeof(double));
    ctx->dMode.alloc(nS * sizeof(int));
    ctx->dDisabled.alloc(static_cast<size_t>(ctx->nElements) * nScenarios * sizeof(char));
    ctx->dActive.alloc(static_cast<size_t>(nScenarios) * sizeof(char));

    // ---- reactive-limits pack: the controller BUS_V rows mapped through the diagonal-pairing
    //      permutation (= identity in the batch path, no families, but mapped for correctness) ----
    ctx->nReactiveLimitEntries = nReactiveLimitEntries;
    ctx->dQbuf.alloc(static_cast<size_t>(n) * nScenarios * sizeof(double));
    ctx->dOuterLoopCheckActive.alloc(static_cast<size_t>(nScenarios) * sizeof(char));
    ctx->dOuterLoopUnstable.alloc(static_cast<size_t>(nScenarios) * sizeof(char));
    // The seed launch of updatePilotVeqBatch / updateDistrQTargetsBatch (before the first solve) READS
    // dOuterLoopCheckActive and relies on "not active yet" = 0; the RL pass is what first writes it. Explicit, not
    // left to the allocator's zeroing.
    CUDA_CHECK(cudaMemset(ctx->dOuterLoopCheckActive.p, 0, static_cast<size_t>(nScenarios) * sizeof(char)));
    CUDA_CHECK(cudaMemset(ctx->dOuterLoopUnstable.p, 0, static_cast<size_t>(nScenarios) * sizeof(char)));
    // OUTSIDE the `nReactiveLimitEntries > 0` block: a run with reactive limits DISABLED has no entries at all, and that is
    // exactly when every pilot-self group needs the gate — its pilot can never pin, so it controls throughout.
    // Built inside, the gate did not exist for such a run: the pilots' raw q was emitted at their carrier rows
    // on top of the DISTR_Q terms and rte6515's base (limits off) started at |f(x0)| 117.908 against OLF's
    // 117.845 and never converged.
    // Derive the PILOT-SELF groups from the candidate table itself: the fallback row repeated among the
    // member rows means the controlled bus is one of its own controllers. The pilot's RL entry is the one
    // whose (row-mapped) V row is that row; without an entry the pilot can never pin and the group needs
    // no gate. nQr is small (groups), nReactiveLimitEntries a few thousand, and this runs once per context.
    int nQrG = ctx->st != nullptr ? ctx->st->nQr : 0;
    if (nQrG > 0) {
        std::vector<int> psEntry(nQrG, -1);
        for (int g = 0; g < nQrG; g++) {
            int p0 = ctx->st->hQrPtr[g];
            int cnt = ctx->st->hQrPtr[g + 1] - p0;
            int fallbackRow = ctx->st->hQrRows[p0];
            bool pilotSelf = false;
            for (int c = 1; c < cnt; c++) {
                if (ctx->st->hQrRows[p0 + c] == fallbackRow) {
                    pilotSelf = true;
                    break;
                }
            }
            if (!pilotSelf) {
                continue;                                // not a pilot-self group: -1, no gate (raw q as always)
            }
            // PS_NO_ENTRY (-2), not -1: a pilot-self pilot with NO reactive-limit entry can never be PINNED
            // (limits off for the run, or an empty q range), so it controls for the whole solve and its raw q
            // must ALWAYS be skipped — the gate is needed MORE, not less, than for a pinnable one. Read as
            // "no gate" (-1, the non-pilot-self value) it was emitted at the carrier row on top of the very
            // DISTR_Q terms it feeds: with reactive limits disabled on rte6515 the base started at
            // |f(x0)| 117.908 against OLF's 117.845 and the device's load flow never converged.
            psEntry[g] = PS_NO_ENTRY;
            ctx->hasPilotSelf = true;
            for (int i = 0; i < nReactiveLimitEntries; i++) {
                if (rowMap[reactiveLimitVoltageRow[i]] == fallbackRow) {
                    psEntry[g] = i;
                    break;
                }
            }
        }
        if (ctx->hasPilotSelf) {
            ctx->dPsPilotEntry.upload(psEntry.data(), psEntry.size() * sizeof(int));
            ctx->dPsSkip.alloc(static_cast<size_t>(nQrG) * ctx->nScenarios * sizeof(char));
            if (std::getenv("OLF_GPU_RL_TRACE") != nullptr) {
                int cnt = 0, noEntry = 0;
                for (int g = 0; g < nQrG; g++) {
                    cnt += psEntry[g] >= 0 ? 1 : 0;
                    noEntry += psEntry[g] == PS_NO_ENTRY ? 1 : 0;
                }
                std::fprintf(stderr, "PILOTSELF_GATE groups=%d(+%d never-pinnable) of %d nReactiveLimitEntries=%d\n",
                             cnt, noEntry, nQrG, nReactiveLimitEntries);
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
        // q and flip the pin at THAT row; its VOLTAGE reads stay on the bus's own row (dReactiveLimitVoltageRow). Without the split
        // the check compared a voltage residual against reactive limits: measured on rte6515's full busbar scope
        // with the relocation enabled, entries read -78.0 and 4767.3 MVAr and 20 buses were pinned that OLF never
        // pins, after which the pinned solve burned all 30 iterations.
        std::vector<int> reactiveLimitReactivePowerRowMapped(reactiveLimitVoltageRowMapped);
        if (ctx->nZeroImpedanceBranch > 0 && ctx->st->dZeroImpedanceBranchIndices.p != nullptr) {
            std::vector<int> zidx(static_cast<size_t>(ctx->nZeroImpedanceBranch) * 11);
            CUDA_CHECK(cudaMemcpy(zidx.data(), ctx->st->dZeroImpedanceBranchIndices.p, zidx.size() * sizeof(int), cudaMemcpyDeviceToHost));
            std::map<int, int> qRowOfVRow;
            for (int e = 0; e < ctx->nZeroImpedanceBranch; e++) {
                const int* z = zidx.data() + static_cast<size_t>(e) * 11;
                int childSide = (z[10] >> 1) & 3;
                int ownVCol = childSide == 1 ? z[6] : (childSide == 2 ? z[7] : -1);
                if (childSide == 0 || z[5] < 0 || ownVCol < 0) {
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
            if (moved > 0 && std::getenv("OLF_GPU_RL_TRACE") != nullptr) {
                std::fprintf(stderr, "RL_QROW %d of %d entries follow the coupler relocation\n", moved, nReactiveLimitEntries);
            }
        }
        ctx->dReactiveLimitReactivePowerRow.upload(reactiveLimitReactivePowerRowMapped.data(), nReactiveLimitEntries * sizeof(int));
        ctx->dReactiveLimitMinQ.upload(reactiveLimitMinQ, nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitMaxQ.upload(reactiveLimitMaxQ, nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitLoadQ.upload(reactiveLimitLoadQ, nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitTargetV.upload(reactiveLimitTargetV, nReactiveLimitEntries * sizeof(double));
        ctx->dReactiveLimitFlags.upload(reactiveLimitFlags, nReactiveLimitEntries * sizeof(char));
        ctx->allPvPinnable = 1;
        for (int i = 0; i < nReactiveLimitEntries; i++) {
            if (!(reactiveLimitFlags[i] & GpuReactiveLimitsEntryFlag::STAND_BY_AUTOMATON) && !((reactiveLimitMinQ[i] > -1e7 || reactiveLimitMaxQ[i] < 1e7) && reactiveLimitMinQ[i] < reactiveLimitMaxQ[i])) {
                ctx->allPvPinnable = 0;                      // an unlimited PV bus: the rule never fires
            }
        }
        ctx->dReactiveLimitSavedTarget.alloc(static_cast<size_t>(nReactiveLimitEntries) * nScenarios * sizeof(double));
        ctx->dReactiveLimitCheckDecision.alloc(static_cast<size_t>(nReactiveLimitEntries) * nScenarios * sizeof(char));
        // OLF_GPU_SW_LOG=<slot>: record one scenario's reactive-limit decisions ON THE DEVICE (one
        // atomicAdd per decision, copied out once at the end) instead of reading the whole batch's
        // buffers over PCIe every pass. The host-side traces cost hundreds of MB per pass, which
        // serializes the stream and changes the very kernel scheduling whose effect is being traced:
        // a run-to-run flip disappears while you watch it.
        if (std::getenv("OLF_GPU_SW_LOG") != nullptr) {
            ctx->dReactiveLimitsDecisionLog.alloc(REACTIVE_LIMITS_DECISION_LOG_CAPACITY * REACTIVE_LIMITS_DECISION_LOG_WIDTH * sizeof(double));
            ctx->dReactiveLimitsDecisionLogCount.alloc(sizeof(int));
            CUDA_CHECK(cudaMemset(ctx->dReactiveLimitsDecisionLogCount.p, 0, sizeof(int)));
        }
        ctx->dReactiveLimitSavedRowMode.alloc(static_cast<size_t>(nReactiveLimitEntries) * nScenarios * sizeof(char));
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
        ctx->dDistributedSlackBusMaxDelta.upload(dsMaxDelta, nDistributedSlackBuses * sizeof(double));
        ctx->dDistributedSlackBusMinDelta.upload(dsMinDelta, nDistributedSlackBuses * sizeof(double));
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
    CUDSS_CHECK(cudssCreate(&ctx->ds.handle));
    CUDSS_CHECK(cudssSetStream(ctx->ds.handle, ctx->stream));
    CUDSS_CHECK(cudssConfigCreate(&ctx->ds.config));
    int ir = ctx->st->nZeroImpedanceBranch > 0 ? irBaseSteps() : IR_N_STEPS;
    CUDSS_CHECK(cudssConfigSet(ctx->ds.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)));
    if (mode == GpuLinearSolveMode::ROBUST) {
        applyRobustSingleMatrixConfig(ctx->ds.config);       // its own ordering: never set the ordering twice
    } else {
        applyReorderAlg(ctx->ds.config, ctx->st->nZeroImpedanceBranch > 0);
    }
    CUDSS_CHECK(cudssDataCreate(ctx->ds.handle, &ctx->ds.data));

    if (mode == GpuLinearSolveMode::UBATCH) {
        setupBatchMatricesUbatch(ctx.get());
    } else {
        setupBatchMatricesFallback(ctx.get());
    }
    return ctx.release();
}

void uploadBatchOverrides(BatchSolverContext* ctx, int lsMode, double maxDv, double maxDphi,
                          const int* varType, int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                          int nBaseClosedZeroImpedanceBranches, int nZeroImpedanceBranchOpenOverrides, const int* zeroImpedanceBranchOpenScenarioElement,
                          int nZeroImpedanceBranchCloseOverrides, const int* zeroImpedanceBranchCloseScenarioElement,
                          int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal) {
    int n = ctx->n;
    int nS = ctx->nScenarios;
    bool clampStep = lsMode == GpuStateVectorScalingMode::MAX_VOLTAGE_CHANGE && varType != nullptr;
    if (clampStep && !ctx->dVarType.p) {
        ctx->dVarType.upload(varType, static_cast<size_t>(n) * sizeof(int));
        ctx->dStepFactor.alloc(static_cast<size_t>(nS) * sizeof(double));
    }
    if (lsMode == GpuStateVectorScalingMode::LINE_SEARCH) {                                       // LINE_SEARCH backtracking buffers
        if (!ctx->dXprevLs.p) ctx->dXprevLs.alloc(static_cast<size_t>(n) * nS * sizeof(double));
        if (!ctx->dStepFactor.p) ctx->dStepFactor.alloc(static_cast<size_t>(nS) * sizeof(double));
    }
    ctx->clampStep = clampStep;
    ctx->maxDv = maxDv;
    ctx->maxDphi = maxDphi;
    // Per-scenario shunt admittance overrides (section shifts / reconnection). PACK-indexed
    // elements (shunt family index, not the global element index). Empty = a 1-entry dummy
    // upload so the kernels' pointers are always valid.
    ctx->nShuntAdmittanceOverrides = nShuntAdmittanceOverrides;
    ctx->hShuntAdmittanceOverrideBaseScenarioElement.assign(shuntAdmittanceOverrideScenarioElement, shuntAdmittanceOverrideScenarioElement + (size_t) nShuntAdmittanceOverrides * 2);
    ctx->hShuntAdmittanceOverrideBaseDelta.assign(shuntAdmittanceOverrideDelta, shuntAdmittanceOverrideDelta + (size_t) nShuntAdmittanceOverrides * 2);
    if (nShuntAdmittanceOverrides > 0 || ctx->nIncrementalShuntVoltageControlControllerShunts > 0) {
        // room for the in-kernel shunt loop to append one (dG, dB) per (scenario, controller)
        size_t capEntries = std::max<size_t>(static_cast<size_t>(nShuntAdmittanceOverrides) + (ctx->nIncrementalShuntVoltageControlControllerShunts > 0 ? static_cast<size_t>(nS) * ctx->nIncrementalShuntVoltageControlControllerShunts : 0), 1);
        ctx->dShuntAdmittanceOverrideScenarioElement.ensure(capEntries * 2 * sizeof(int));
        ctx->dShuntAdmittanceOverrideDelta.ensure(capEntries * 2 * sizeof(double));
        if (nShuntAdmittanceOverrides > 0) {
            CUDA_CHECK(cudaMemcpy(ctx->dShuntAdmittanceOverrideScenarioElement.p, shuntAdmittanceOverrideScenarioElement, static_cast<size_t>(nShuntAdmittanceOverrides) * 2 * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dShuntAdmittanceOverrideDelta.p, shuntAdmittanceOverrideDelta, static_cast<size_t>(nShuntAdmittanceOverrides) * 2 * sizeof(double), cudaMemcpyHostToDevice));
        }
    } else if (!ctx->dShuntAdmittanceOverrideScenarioElement.p) {
        int dummyI[2] = {-1, -1};
        double dummyD[2] = {0, 0};
        ctx->dShuntAdmittanceOverrideScenarioElement.upload(dummyI, sizeof(dummyI));
        ctx->dShuntAdmittanceOverrideDelta.upload(dummyD, sizeof(dummyD));
    }
    ctx->nBaseClosedZeroImpedanceBranches = nBaseClosedZeroImpedanceBranches > 0 ? nBaseClosedZeroImpedanceBranches : ctx->nZeroImpedanceBranch;         // default: every zero-impedance element always enabled
    ctx->nZeroImpedanceBranchOpenOverrides = nZeroImpedanceBranchOpenOverrides;
    if (nZeroImpedanceBranchOpenOverrides > 0) {
        ctx->dZeroImpedanceBranchOpenScenarioElement.upload(zeroImpedanceBranchOpenScenarioElement, static_cast<size_t>(nZeroImpedanceBranchOpenOverrides) * 2 * sizeof(int));
    } else if (!ctx->dZeroImpedanceBranchOpenScenarioElement.p) {
        int noZeroImpedanceBranchOpenOverride[2] = {-1, -1};
        ctx->dZeroImpedanceBranchOpenScenarioElement.upload(noZeroImpedanceBranchOpenOverride, sizeof(noZeroImpedanceBranchOpenOverride));
    }
    ctx->nZeroImpedanceBranchCloseOverrides = nZeroImpedanceBranchCloseOverrides;
    if (nZeroImpedanceBranchCloseOverrides > 0) {
        ctx->dZeroImpedanceBranchCloseScenarioElement.upload(zeroImpedanceBranchCloseScenarioElement, static_cast<size_t>(nZeroImpedanceBranchCloseOverrides) * 2 * sizeof(int));
    } else if (!ctx->dZeroImpedanceBranchCloseScenarioElement.p) {
        int noZeroImpedanceBranchCloseOverride[2] = {-1, -1};
        ctx->dZeroImpedanceBranchCloseScenarioElement.upload(noZeroImpedanceBranchCloseOverride, sizeof(noZeroImpedanceBranchCloseOverride));
    }
    if (ctx->nZeroImpedanceBranch > 0) {                                      // the dense (scenario, element) forms (see dZeroImpedanceBranchOpenMask)
        size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nZeroImpedanceBranch;
        std::vector<char> off(len, 0), on(len, 0);
        for (int k = 0; k < nZeroImpedanceBranchOpenOverrides; k++) {
            int sl = zeroImpedanceBranchOpenScenarioElement[2 * k], e = zeroImpedanceBranchOpenScenarioElement[2 * k + 1];
            if (sl >= 0 && sl < ctx->nScenarios && e >= 0 && e < ctx->nZeroImpedanceBranch) off[(size_t) sl * ctx->nZeroImpedanceBranch + e] = 1;
        }
        for (int k = 0; k < nZeroImpedanceBranchCloseOverrides; k++) {
            int sl = zeroImpedanceBranchCloseScenarioElement[2 * k], e = zeroImpedanceBranchCloseScenarioElement[2 * k + 1];
            if (sl >= 0 && sl < ctx->nScenarios && e >= 0 && e < ctx->nZeroImpedanceBranch) on[(size_t) sl * ctx->nZeroImpedanceBranch + e] = 1;
        }
        ctx->dZeroImpedanceBranchOpenMask.upload(off.data(), len);
        ctx->dZeroImpedanceBranchCloseMask.upload(on.data(), len);
        if (!ctx->dSpanningTreeOverrideMask.p) {
            std::vector<char> tree(len, (char) -1);
            ctx->dSpanningTreeOverrideMask.upload(tree.data(), len);
        }
        ctx->hasZeroImpedanceBranchMasks = true;
        if (std::getenv("OLF_GPU_PHASE_TIME") != nullptr) {
            std::fprintf(stderr, "ZI_LISTS nZeroImpedanceBranch=%d nScenarios=%d nZeroImpedanceBranchOpenOverrides=%d nZeroImpedanceBranchCloseOverrides=%d\n", ctx->nZeroImpedanceBranch, ctx->nScenarios, nZeroImpedanceBranchOpenOverrides, nZeroImpedanceBranchCloseOverrides);
        }
    }
    ctx->nR1Ov = nR1Ov;
    ctx->hR1OvBaseSlotElem.assign(r1OvSlotElem, r1OvSlotElem + (size_t) nR1Ov * 2);
    ctx->hR1OvBaseVal.assign(r1OvVal, r1OvVal + (size_t) nR1Ov * 8);
    if (nR1Ov > 0 || ctx->nIncrementalTransformerVoltageControlControllerBranches > 0 || ctx->nIncrementalFlowControlControllerBranches > 0 || ctx->nPhaseControlControllerBranches > 0) {
        // Room for the in-kernel TVC loop to merge one moved tap per (scenario, controller) in
        // front of the caller's list without reallocating mid-solve.
        size_t capEntries = static_cast<size_t>(nR1Ov) + static_cast<size_t>(nS) * (ctx->nIncrementalTransformerVoltageControlControllerBranches + ctx->nIncrementalFlowControlControllerBranches + ctx->nPhaseControlControllerBranches + ctx->nTransformerVoltageControlMembers);
        capEntries = std::max<size_t>(capEntries, 1);
        ctx->dR1OvSlotElem.ensure(capEntries * 2 * sizeof(int));
        ctx->dR1OvVal.ensure(capEntries * 8 * sizeof(double));
        if (nR1Ov > 0) {
            CUDA_CHECK(cudaMemcpy(ctx->dR1OvSlotElem.p, r1OvSlotElem, static_cast<size_t>(nR1Ov) * 2 * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dR1OvVal.p, r1OvVal, static_cast<size_t>(nR1Ov) * 8 * sizeof(double), cudaMemcpyHostToDevice));
        }
    } else if (!ctx->dR1OvSlotElem.p) {
        int dummyI[2] = {-1, -1};
        double dummyD[1] = {0};
        ctx->dR1OvSlotElem.upload(dummyI, sizeof(dummyI));
        ctx->dR1OvVal.upload(dummyD, sizeof(dummyD));
    }
}

}  // namespace olfgpu
