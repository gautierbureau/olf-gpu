// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See sensitivity.cuh.
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
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../equations/branch_fill.cuh"
#include "../equations/bus_injection_fill.cuh"
#include "../equations/distribution_fill.cuh"
#include "../equations/fill_launch.cuh"
#include "../equations/zero_impedance_fill.cuh"
#include "../solver/driver_steps.cuh"
#include "../solver/zero_diagonal_repair.cuh"
#include "pilot_voltage.cuh"
#include "sensitivity.cuh"

namespace olfgpu {

// Distributed-slack pass: thread = (participating bus, scenario). The slack mismatch is
// qBuf[slackPhiRow] (= Σ branch p at the slack) − slackTargetP. If it exceeds the tolerance, each
// participating bus takes factor·mismatch into its BUS_TARGET_P target (factor normalized to sum 1, so
// the whole mismatch is absorbed), mirroring ActivePowerDistribution (one-shot, no Pmin/Pmax saturation
// — exact vs OLF when no generator saturates). A distribution marks its scenario for another inner solve.
// Freeze, on the FIRST distribution pass of a solve, the active-power share the slack/reference bus must
// RETAIN when its own generators participate: retain = (1 − Σfactor)·mismatch0. Σfactor over the emitted
// (non-slack) buses is < 1 exactly when the slack participates (its factor sits in the normalization
// denominator but emits no row); 1 − Σfactor is the slack's participation fraction ρ. Without this the
// geometric outer loop would drive the slack mismatch to 0 and leak the slack's whole share to the
// non-slack buses. ρ = 0 (non-participating slack) ⇒ retain = 0 ⇒ the slack keeps nothing extra.
// The slack/reference bus retains ρ = (1 − Σfactor) of the TOTAL active imbalance currently being shared,
// where total = the slack's residual mismatch (qBuf at slackPhiRow − slackTargetP) PLUS what the non-slack
// participating buses have already absorbed (Σ of their cumulative target deltas). Re-evaluated EVERY outer
// pass: freezing ρ·mismatch0 at the first pass ignores the AC loss change during redistribution, so the
// slack would keep ρ·mismatch0 and the non-slack side would over-absorb by ρ·Δlosses; tracking the current
// total instead makes the non-slack buses absorb exactly (1 − ρ)·total_final, matching OLF's per-pass
// distribution of the full mismatch. ρ = 0 (non-participating slack, incl. all batched gen-loss scenarios)
// ⇒ retain = 0, so this is a no-op there.
__global__ void setUnitRhsBatch(int n, int nScenarios, int unitRow, double* b) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nScenarios) return;
    b[t] = (t % n == unitRow) ? 1.0 : 0.0;
}

__global__ void gatherSensiBatch(int n, int nScenarios, int k, int m, int ki, const int* observeCols,
                                 const double* dx, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;                     // t = s*m + mi
    if (t >= nScenarios * m) return;
    int sIdx = t / m;
    int mi = t % m;
    out[(size_t) sIdx * k * m + (size_t) ki * m + mi] = dx[(size_t) sIdx * n + observeCols[mi]];
}

void sensitivitySolve(BatchSolverContext* ctx, DriverPass& drv, bool& first) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    if (ctx->mode == GpuLinearSolveMode::ROBUST) {
        for (int s2 = 0; s2 < nS; s2++) {
            CUDSS_CHECK(cudssMatrixSetValues(ctx->solver.systemMatrix, ctx->dVals.as<double>() + (size_t) s2 * ctx->nnz));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->solver.rightHandSideMatrix, ctx->dB.as<double>() + (size_t) s2 * n));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->solver.solutionMatrix, ctx->dDx.as<double>() + (size_t) s2 * n));
            if (!ctx->analyzed) {
                gAnalysesMain++;
                { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_ANALYSIS, ctx->solver.config, ctx->solver.data, ctx->solver.systemMatrix, ctx->solver.solutionMatrix, ctx->solver.rightHandSideMatrix)); }
                ctx->analyzed = true;
            }
            { AnalysisTimer at_(__LINE__, 'G'); CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                    ctx->solver.config, ctx->solver.data, ctx->solver.systemMatrix, ctx->solver.solutionMatrix, ctx->solver.rightHandSideMatrix)); }
        }
        first = false;
        return;
    }
    if (!ctx->analyzed) {
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_ANALYSIS, ctx->solver.config, ctx->solver.data, ctx->uniformBatchSystemMatrix, ctx->uniformBatchSolutionMatrix, ctx->uniformBatchRightHandSideMatrix)); }
        ctx->analyzed = true;
    }
    int phase = CUDSS_PHASE_SOLVE | (first ? CUDSS_PHASE_FACTORIZATION : 0);
    first = false;
    { AnalysisTimer at_(__LINE__, ((phase) & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) ? 'G' : ((phase) & CUDSS_PHASE_ANALYSIS) ? 'A' : 'S'); CUDSS_CHECK(cudssExecute(ctx->solver.handle, phase, ctx->solver.config, ctx->solver.data, ctx->uniformBatchSystemMatrix, ctx->uniformBatchSolutionMatrix, ctx->uniformBatchRightHandSideMatrix)); }
}

// Batched SENSITIVITY solve for the incremental outer-loop family (device port of OLF's
// SensitivityContext): J at the given per-scenario states (the same forward system the Newton
// solves — OLF's "solveTransposed" IS the forward direction in powsybl-math's transposed
// storage), factorized once, solved for a unit RHS at each controller's pinned-variable
// identity row (BRANCH_TARGET_RHO1 & co, storage row = rowMap[eqRow]); the requested variable
// columns of each solution are gathered per scenario. MUST be called on a context that has just
// completed solveBatchContext with the SAME per-scenario overrides/row modes — the fill reads
// them from the live device buffers. UBATCH mode (3) only.
std::vector<double> solveBatchSensitivities(BatchSolverContext* ctx, const double* states,
                                            int k, const int* unitEqRows, int m, const int* observeVarCols) {
    if (ctx->mode != GpuLinearSolveMode::UBATCH) {
        throw std::runtime_error("solveBatchSensitivities: UBATCH mode (3) only");
    }
    int n = ctx->structure->n;
    int nS = ctx->nScenarios;
    int nnz = ctx->structure->nnz;

    CUDA_CHECK(cudaMemcpyAsync(ctx->dX.p, states, (size_t) n * nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
    std::vector<char> allActive(nS, 1);
    CUDA_CHECK(cudaMemcpyAsync(ctx->dActive.p, allActive.data(), nS * sizeof(char), cudaMemcpyHostToDevice, ctx->stream));

    // ---- J fill at the converged states (same launches as the solve loop's full fill) ----
    CUDA_CHECK(cudaMemsetAsync(ctx->dVals.p, 0, (size_t) nnz * nS * sizeof(double), ctx->stream));
    initFBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dTarget.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->structure->closedBranch.elementCount) {
        fillClosedBranchBatch<<<grid(ctx->structure->closedBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->closedBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->closedBranchOffset, ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->closedBranch.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nClosedBranchParameterOverrides, ctx->dClosedBranchParameterOverrideScenarioElement.as<int>(), ctx->dClosedBranchParameterOverrideValues.as<double>(), ctx->structure->closedBranch.alternateDestination.as<int>(), ctx->structure->closedBranch.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->closedBranch.dynamicRelocation ? 1 : 0, ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlDestination.as<int>() : nullptr, ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlSelector.as<int>() : nullptr, withPilotSelf(relocationTablesOf(ctx->structure->closedBranch, ctx->structure.get()), ctx));
    }
    if (ctx->structure->openBranch.elementCount) {
        fillOpenBranchBatch<<<grid(ctx->structure->openBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->openBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->openBranchOffset, ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(), ctx->structure->openBranch.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->structure->openBranch.alternateDestination.as<int>(), ctx->structure->openBranch.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->openBranch.dynamicRelocation ? 1 : 0, withPilotSelf(relocationTablesOf(ctx->structure->openBranch, ctx->structure.get()), ctx));
    }
    if (ctx->structure->shunt.elementCount) {
        fillShuntBatch<<<grid(ctx->structure->shunt.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->shunt.elementCount, nS, n, nnz, ctx->nElements, ctx->shuntOffset, ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(), ctx->structure->shunt.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nShuntAdmittanceOverrides, ctx->dShuntAdmittanceOverrideScenarioElement.as<int>(), ctx->dShuntAdmittanceOverrideDelta.as<double>(), ctx->structure->shunt.alternateDestination.as<int>(), ctx->structure->shunt.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->shunt.dynamicRelocation ? 1 : 0, withPilotSelf(relocationTablesOf(ctx->structure->shunt, ctx->structure.get()), ctx));
    }
    if (ctx->structure->nLoadModelTerms) {
        fillLoadModelBatch<<<grid(ctx->structure->nLoadModelTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->nLoadModelTerms, nS, n, nnz, ctx->structure->dLoadModelParameters.as<double>(), ctx->structure->dLoadModelIndices.as<int>(), ctx->structure->dLoadModelDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->structure->loadModelRelocated ? ctx->structure->dLoadModelAlternateRow.as<int>() : nullptr, ctx->structure->loadModelRelocated ? ctx->structure->dLoadModelAlternateDestination.as<int>() : nullptr, ctx->structure->loadModelRelocated ? ctx->dMode.as<int>() : nullptr);
    }
    if (ctx->structure->hvdcAcEmulation.elementCount) {
        fillHvdcBatch<<<grid(ctx->structure->hvdcAcEmulation.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->hvdcAcEmulation.elementCount, nS, n, nnz, ctx->nElements, ctx->hvdcAcEmulationOffset, ctx->structure->hvdcAcEmulation.parameters.as<double>(), ctx->structure->hvdcAcEmulation.indices.as<int>(), ctx->structure->hvdcAcEmulation.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->structure->hvdcAcEmulation.alternateDestination.as<int>(), ctx->structure->hvdcAcEmulation.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->hvdcAcEmulation.dynamicRelocation ? 1 : 0);
    }
    int distrQTraceRow = gpuDistrQTraceRow();
    if (ctx->hasPilotSelf && ctx->dPilotSelfCarrierSkip.p != nullptr && ctx->structure->nReactiveRelocationGroups > 0) {
        // Recompute the PILOT-SELF gate from the CURRENT pin states: such a pilot has a Q balance to
        // relocate only once a reactive limit has pinned it, and the RL pass moves pins between passes.
        pilotSelfGateBatch<<<grid(ctx->structure->nReactiveRelocationGroups * nS), BLOCK, 0, ctx->stream>>>(
                ctx->structure->nReactiveRelocationGroups, nS, ctx->nReactiveLimitEntries, ctx->dPilotSelfReactiveLimitEntry.as<int>(),
                ctx->dReactiveLimitQLimitType.as<char>(),
                ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr, ctx->dPilotSelfCarrierSkip.as<char>());
    }
    if (ctx->nDistrQ) fillDistrQBatch<<<grid(ctx->nDistrQ * nS), BLOCK, 0, ctx->stream>>>(ctx->nDistrQ, nS, n, nnz, ctx->nElements, ctx->closedBranchOffset, ctx->openBranchOffset, ctx->structure->dDistrQElement.as<int>(), ctx->structure->dDistrQSide.as<int>(), ctx->structure->dDistrQKind.as<int>(), ctx->structure->dDistrQWeight.as<double>(), ctx->structure->dDistrQDestination.as<int>(), ctx->structure->dDistrQAlternateDestination.as<int>(), ctx->structure->dDistrQSelectorRow.as<int>(), ctx->structure->dDistrQDummyQRow.as<int>(), ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(), ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nDistrQWeightOverrides, ctx->nDistrQWeightOverrides ? ctx->dDistrQWeightOverrideScenarioTerm.as<int>() : nullptr, ctx->nDistrQWeightOverrides ? ctx->dDistrQWeightOverrideWeight.as<double>() : nullptr,
                ctx->nDistrQKeyOverrides, ctx->nDistrQKeyOverrides ? ctx->dDistrQKeyOverrideScenarioMember.as<int>() : nullptr, ctx->nDistrQKeyOverrides ? ctx->dDistrQKeyOverrideKey.as<double>() : nullptr, ctx->dMode.as<int>(), ctx->nDistrQInactiveMembers,
                ctx->nDistrQInactiveMembers ? ctx->dDistrQInactiveMembers.as<int>() : nullptr,
                ctx->nDistrQDynamicKeys ? ctx->dDistrQRowMember.as<int>() : nullptr, ctx->nDistrQDynamicKeys || ctx->hasDistrQRelocationGroups ? ctx->dDistrQInjectionMember.as<int>() : nullptr, ctx->nDistrQDynamicKeys ? ctx->dDistrQMemberKey.as<double>() : nullptr, ctx->nDistrQDynamicKeys ? ctx->dDistrQMemberReactiveLimitEntry.as<int>() : nullptr, ctx->nDistrQDynamicKeys ? ctx->dDistrQMemberGroupStart.as<int>() : nullptr, ctx->nDistrQDynamicKeys ? ctx->dDistrQMemberGroupCount.as<int>() : nullptr, ctx->dReactiveLimitQLimitType.as<char>(), ctx->nReactiveLimitEntries, ctx->hasPilotSelf && ctx->dPilotSelfCarrierSkip.p != nullptr ? ctx->dPilotSelfCarrierSkip.as<char>() : nullptr, ctx->structure->nReactiveRelocationGroups, ctx->hasDistrQRelocationGroups ? ctx->dDistrQMemberRelocationGroup.as<int>() : nullptr, ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->dReactiveRelocationGroupStart.as<int>() : nullptr, ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->dReactiveRelocationCandidateRows.as<int>() : nullptr, ctx->nClosedBranchParameterOverrides, ctx->dClosedBranchParameterOverrideScenarioElement.as<int>(), ctx->dClosedBranchParameterOverrideValues.as<double>(), ctx->nShuntAdmittanceOverrides, ctx->dShuntAdmittanceOverrideScenarioElement.as<int>(), ctx->dShuntAdmittanceOverrideDelta.as<double>(), distrQTraceRow,
                    ctx->hasDistrQSubgroups ? ctx->dDistrQSubgroup.as<char>() : nullptr, ctx->hasDistrQSubgroups ? ctx->dReactiveRelocationDistrQMemberStart.as<int>() : nullptr, ctx->nDistrQMembers);
    if (ctx->nDistrRho) fillDistrRhoBatch<<<grid(ctx->nDistrRho * nS), BLOCK, 0, ctx->stream>>>(ctx->nDistrRho, nS, n, nnz, ctx->structure->dDistrRhoColumn.as<int>(), ctx->structure->dDistrRhoCoefficient.as<double>(), ctx->structure->dDistrRhoDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nDistrRhoCoefficientOverrides, ctx->nDistrRhoCoefficientOverrides ? ctx->dDistrRhoOverrideScenarioTerm.as<int>() : nullptr, ctx->nDistrRhoCoefficientOverrides ? ctx->dDistrRhoOverrideCoefficient.as<double>() : nullptr, ctx->structure->hasDistrRhoRelocationGroups ? ctx->structure->dDistrRhoRelocationGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->structure->dReactiveRelocationGroupStart.as<int>(), ctx->structure->dReactiveRelocationCandidateRows.as<int>());
    if (ctx->nDistrShuntB) fillDistrShuntBBatch<<<grid(ctx->nDistrShuntB * nS), BLOCK, 0, ctx->stream>>>(ctx->nDistrShuntB, nS, n, nnz, ctx->structure->dDistrShuntBColumn.as<int>(), ctx->structure->dDistrShuntBCoefficient.as<double>(), ctx->structure->dDistrShuntBDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->nZeroImpedanceBranch) fillZeroImpBatch<<<grid(ctx->nZeroImpedanceBranch * nS), BLOCK, 0, ctx->stream>>>(ctx->nZeroImpedanceBranch, nS, n, nnz, ctx->structure->dZeroImpedanceBranchParameters.as<double>(), ctx->structure->dZeroImpedanceBranchIndices.as<int>(), ctx->structure->dZeroImpedanceBranchDestination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nBaseClosedZeroImpedanceBranches, ctx->nZeroImpedanceBranchCloseOverrides, ctx->dZeroImpedanceBranchCloseScenarioElement.as<int>(), ctx->nZeroImpedanceBranchOpenOverrides, ctx->dZeroImpedanceBranchOpenScenarioElement.as<int>(), ctx->structure->dZeroImpedanceAlternateDestination.as<int>(), ctx->structure->dZeroImpedanceSelectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->zeroImpedanceDynamicRelocation ? 1 : 0, withPilotSelf(zeroImpedanceRelocationTables(ctx->structure.get()), ctx), ctx->nSpanningTreeOverrides, ctx->nSpanningTreeOverrides ? ctx->dSpanningTreeOverrideScenarioElementValue.as<int>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dZeroImpedanceBranchOpenMask.as<char>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dZeroImpedanceBranchCloseMask.as<char>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dSpanningTreeOverrideMask.as<char>() : nullptr);
    applyRowModeBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->dMode.as<int>(), ctx->structure->dRowPtr.as<int>(), ctx->structure->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dRowSlope.as<double>(),
                    ctx->structure->dColIdx.as<int>(), ctx->structure->dRelocatedChildOwnRow.p != nullptr ? ctx->structure->dRelocatedChildOwnRow.as<int>() : nullptr);
    repairZeroDiagonalRows(ctx, nS, ctx->dVals.as<double>());

    if (!ctx->analyzed) {
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_ANALYSIS, ctx->solver.config, ctx->solver.data, ctx->uniformBatchSystemMatrix, ctx->uniformBatchSolutionMatrix, ctx->uniformBatchRightHandSideMatrix)); }
        ctx->analyzed = true;
    }
    DeviceBuf dObs, dOut;
    dObs.upload(observeVarCols, m * sizeof(int));
    dOut.alloc((size_t) nS * k * m * sizeof(double));
    for (int ki = 0; ki < k; ki++) {
        int row = ctx->structure->rowMap[unitEqRows[ki]];           // equation row -> CSR storage row
        if (row < 0) {
            throw std::runtime_error("solveBatchSensitivities: unit row not an active equation");
        }
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        repairZeroDiagonalRightHandSide(ctx, nS, ctx->dB.as<double>());
        int phase = CUDSS_PHASE_SOLVE | (ki == 0 ? CUDSS_PHASE_FACTORIZATION : 0);
        { AnalysisTimer at_(__LINE__, ((phase) & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) ? 'G' : ((phase) & CUDSS_PHASE_ANALYSIS) ? 'A' : 'S'); CUDSS_CHECK(cudssExecute(ctx->solver.handle, phase, ctx->solver.config, ctx->solver.data, ctx->uniformBatchSystemMatrix, ctx->uniformBatchSolutionMatrix, ctx->uniformBatchRightHandSideMatrix)); }
        gatherSensiBatch<<<grid(nS * m), BLOCK, 0, ctx->stream>>>(n, nS, k, m, ki, dObs.as<int>(),
                ctx->dDx.as<double>(), dOut.as<double>());
    }
    std::vector<double> out((size_t) nS * k * m);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), dOut.p, out.size() * sizeof(double), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    return out;
}

}  // namespace olfgpu
