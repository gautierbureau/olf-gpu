// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See fill_launch.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "scatter.cuh"
#include "../solver/driver_steps.cuh"
#include "branch_fill.cuh"
#include "bus_injection_fill.cuh"
#include "distribution_fill.cuh"
#include "zero_impedance_fill.cuh"
#include "../outerloop/pilot_voltage.cuh"
#include "fill_launch.cuh"

namespace olfgpu {

// The GROUP relocation tables of one family, or all-null when the structure has none (plan-A6).
ReactiveRelocationTables relocationTablesOf(Family& f, SharedStructure* structure) {
    ReactiveRelocationTables rt;
    if (f.relocationCandidateDestinations.p != nullptr) {
        rt.relocationGroup = f.relocationGroup.as<int>();
        rt.relocationCandidateOffset = f.relocationCandidateOffset.as<int>();
        rt.relocationCandidateDestinations = f.relocationCandidateDestinations.as<int>();
        rt.relocationDependentPilot = f.relocationDependentPilot.p != nullptr ? f.relocationDependentPilot.as<char>() : nullptr;
        rt.reactiveRelocationGroupStart = structure->dReactiveRelocationGroupStart.as<int>();
        rt.reactiveRelocationCandidateRows = structure->dReactiveRelocationCandidateRows.as<int>();
        rt.relocatedChildOwnRow = structure->dRelocatedChildOwnRow.p != nullptr ? structure->dRelocatedChildOwnRow.as<int>() : nullptr;
    }
    return rt;
}

// The zero-impedance family's relocation tables, or all-null when it has none.
// Attaches the PILOT-SELF gate to a structure-built ReactiveRelocationTables. Inert (and identical to before) on any
// context without such a group, so every existing candidate-table group keeps its exact behavior.
ReactiveRelocationTables withPilotSelf(ReactiveRelocationTables rt, BatchSolverContext* ctx) {
    if (ctx->hasPilotSelf && ctx->dPilotSelfCarrierSkip.p != nullptr) {
        rt.pilotSelfCarrierSkip = ctx->dPilotSelfCarrierSkip.template as<char>();
        rt.pilotSelfCarrierSkipStride = ctx->structure->nReactiveRelocationGroups;
    }
    if (ctx->hasDistrQSubgroups) {                                     // tasks #71: per-scenario subgroups of a split group
        rt.distrQSubgroup = ctx->dDistrQSubgroup.template as<char>();
        rt.reactiveRelocationDistrQMemberStart = ctx->dReactiveRelocationDistrQMemberStart.template as<int>();
        rt.nDistrQMembers = ctx->nDistrQMembers;
    }
    return rt;
}

ReactiveRelocationTables zeroImpedanceRelocationTables(SharedStructure* structure) {
    ReactiveRelocationTables rt;
    if (structure->dZeroImpedanceRelocationCandidateDestinations.p != nullptr) {
        rt.relocationGroup = structure->dZeroImpedanceRelocationGroup.as<int>();
        rt.relocationCandidateOffset = structure->dZeroImpedanceRelocationCandidateOffset.as<int>();
        rt.relocationCandidateDestinations = structure->dZeroImpedanceRelocationCandidateDestinations.as<int>();
        rt.relocationDependentPilot = structure->dZeroImpedanceRelocationDependentPilot.p != nullptr ? structure->dZeroImpedanceRelocationDependentPilot.as<char>() : nullptr;
        rt.reactiveRelocationGroupStart = structure->dReactiveRelocationGroupStart.as<int>();
        rt.reactiveRelocationCandidateRows = structure->dReactiveRelocationCandidateRows.as<int>();
        rt.relocatedChildOwnRow = structure->dRelocatedChildOwnRow.p != nullptr ? structure->dRelocatedChildOwnRow.as<int>() : nullptr;
    }
    return rt;
}

// Per-scenario slack/PV identity rows (rowMode is per scenario: a contingency can change
// which buses are PV/slack): a row in identity mode (mode 0) is zeroed, its diagonal set to 1 and its
// mismatch overwritten with x - target. It runs after the fills and overwrites them, which keeps the
// structure (and cuDSS's analysis) immutable across PV->PQ switching.
__global__ void applyRowModeBatch(int n, int nScenarios, int nnz, const int* mode, const int* rowPtr,
                                  const int* diagPos, const double* x, const double* target,
                                  double* vals, double* F, const char* active, const double* rowSlope,
                                  const int* colIdx, const int* relocatedChildOwnRow) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int r = tid % n;
    int s = tid / n;
    if (!active[s]) return;
    int m = mode[(size_t) s * n + r];
    double* valsS = vals + (size_t) s * nnz;
    if (m == GpuRowMode::CROSS_FIXED_VARIABLE) {
        // CROSS identity (tasks #70, VC_REMERGE_RELOC_VROW): this coupler DUMMY_Q row hosts the relocated child's
        // reactive equation, and the child is a re-merged voltage-controlled bus whose own V row carries the live
        // coupler's ZERO_V — so BUS_TARGET_V lands HERE on the child's V column (relocatedChildOwnRow[r] = its own V row =
        // its V column), the slot the child's own dq/dV term already reserves in the pattern.
        int col = relocatedChildOwnRow != nullptr ? relocatedChildOwnRow[r] : -1;
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
            valsS[k] = colIdx[k] == col ? 1.0 : 0.0;
        }
        F[(size_t) s * n + r] = (col >= 0 ? x[(size_t) s * n + col] : 0.0) - target[(size_t) s * n + r];
    } else if (m == GpuRowMode::FIXED_VARIABLE) {  // identity: v/phi/rho/a1/b pinned to target
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
            valsS[k] = 0.0;
        }
        valsS[diagPos[r]] = 1.0;
        F[(size_t) s * n + r] = x[(size_t) s * n + r] - target[(size_t) s * n + r];
    } else if (m == GpuRowMode::BUS_TARGET_V_WITH_SLOPE) {  // SLOPE-SVC: V + slope*sumQ = target — the natural
        double slope = rowSlope[r];                          // reactive fill stands, scaled, plus the v term
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
            valsS[k] *= slope;
        }
        valsS[diagPos[r]] += 1.0;
        size_t idx = (size_t) s * n + r;
        F[idx] = slope * (F[idx] + target[idx]) + x[idx] - target[idx];
    }
}

__global__ void initFBatch(int n, int nScenarios, const double* target, double* F, const char* active) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int s = tid / n;
    if (!active[s]) return;
    F[tid] = -target[tid];
}

// Seed the per-(controller, scenario) limitType from the BASE state, so a controller that OLF had already
// switched to PQ in the base case (packed by the extractor as a voltage-CAPABLE PQ bus, base row mode 1)
// starts pinned at the RIGHT limit and can be switched BACK to PV by a relieving contingency. A base-PV
// controller (mode 0) starts at limitType 0. For a base-PQ controller, v below its target ⇒ it is producing
// max reactive yet cannot reach target ⇒ maxQ-pinned (2); v above target ⇒ minQ-pinned (1). switchCount
// starts at 0 either way (a separate memset).
const int* pilotSelfReactiveLimitEntryOrNull(BatchSolverContext* ctx) {
    return ctx->hasPilotSelf && ctx->dPilotSelfReactiveLimitEntry.p != nullptr ? ctx->dPilotSelfReactiveLimitEntry.template as<int>() : nullptr;
}

// Make a RETIRED scenario's system an identity (diagonal 1, mismatch 0) so it stays
// well-posed inside the uniform batched factorization (a singular sub-block would fail the
// whole batch) and its state is frozen (dx = 0). A scenario retired THIS iteration still
// holds its filled values, so the row is zeroed before the diagonal is set — this is
// correct whether the scenario was just filled or has been inactive (zero) for a while.
__global__ void writeIdentityForInactive(int n, int nScenarios, int nnz, const int* rowPtr,
                                         const int* diagPos, const char* active, double* vals, double* F) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int r = tid % n;
    int s = tid / n;
    if (active[s]) return;
    double* valsS = vals + (size_t) s * nnz;
    for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
        valsS[k] = 0.0;
    }
    valsS[diagPos[r]] = 1.0;
    F[(size_t) s * n + r] = 0.0;
}

// Per-scenario target overrides: each entry writes target[targetOverrideIndex[k]] = targetOverrideValue[k], where targetOverrideIndex is the
// FLAT dTarget index (scenarioSlot*n + row). Used by generator contingencies to set a scenario's bus P/Q
// target (remove the lost generator's P, set BUS_TARGET_Q on a PV→PQ flip) after the base row is tiled.
__global__ void applyTargetOverrideBatch(int nTargetOverrides, const int* targetOverrideIndex, const double* targetOverrideValue, double* target) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nTargetOverrides) {
        target[targetOverrideIndex[k]] = targetOverrideValue[k];
    }
}

void fillActiveBatch(BatchSolverContext* ctx, DriverPass& drv, bool fOnly) {
    const int& n = drv.n;
    const int& nnz = drv.nnz;
    const int& nS = drv.nS;
    const bool& deviceFixedIter = drv.deviceFixedIter;
    if (ctx->hasPilotSelf && ctx->dPilotSelfCarrierSkip.p != nullptr && ctx->structure->nReactiveRelocationGroups > 0) {
        // Recompute the PILOT-SELF gate from the CURRENT pin states BEFORE any family fill: such a pilot
        // has a Q balance to relocate only once a reactive limit has pinned it, and the RL pass moves
        // pins between passes. The closed/open-branch and shunt fills are what route the pilot's raw q
        // through it (zeroImpedanceRelocationDestination), so it must be current when THEY run — launched after them (before the
        // DISTR_Q fill only) the FIRST fill of every context read an unset gate and put the unpinned
        // pilot's raw q on the carrier row on top of its DISTR_Q terms: row 0 of the 4-bus
        // pilot-as-controller fixture assembled -1.451 against OLF's -5.715 (= OLF + 1.0 * q_pilot), and
        // rte6515's base started from |f(x0)| 117.908 against OLF's 117.845.
        pilotSelfGateBatch<<<grid(ctx->structure->nReactiveRelocationGroups * nS), BLOCK, 0, ctx->stream>>>(
                ctx->structure->nReactiveRelocationGroups, nS, ctx->nReactiveLimitEntries, ctx->dPilotSelfReactiveLimitEntry.as<int>(),
                ctx->dReactiveLimitQLimitType.as<char>(),
                ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr, ctx->dPilotSelfCarrierSkip.as<char>());
    }
    if (!fOnly) {
        CUDA_CHECK(cudaMemsetAsync(ctx->dVals.p, 0, static_cast<size_t>(nnz) * nS * sizeof(double), ctx->stream));
    }
    initFBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dTarget.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->structure->closedBranch.elementCount) {
        if (fOnly) {
            fillClosedBranchBatchF<<<grid(ctx->structure->closedBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->closedBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->closedBranchOffset, ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->closedBranch.destination.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nClosedBranchParameterOverrides, ctx->dClosedBranchParameterOverrideScenarioElement.as<int>(), ctx->dClosedBranchParameterOverrideValues.as<double>(), ctx->structure->closedBranch.alternateDestination.as<int>(), ctx->structure->closedBranch.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->closedBranch.dynamicRelocation ? 1 : 0, ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlDestination.as<int>() : nullptr, ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlSelector.as<int>() : nullptr, withPilotSelf(relocationTablesOf(ctx->structure->closedBranch, ctx->structure.get()), ctx));
        } else {
            fillClosedBranchBatch<<<grid(ctx->structure->closedBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->closedBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->closedBranchOffset, ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->closedBranch.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nClosedBranchParameterOverrides, ctx->dClosedBranchParameterOverrideScenarioElement.as<int>(), ctx->dClosedBranchParameterOverrideValues.as<double>(), ctx->structure->closedBranch.alternateDestination.as<int>(), ctx->structure->closedBranch.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->closedBranch.dynamicRelocation ? 1 : 0, ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlDestination.as<int>() : nullptr, ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlSelector.as<int>() : nullptr, withPilotSelf(relocationTablesOf(ctx->structure->closedBranch, ctx->structure.get()), ctx));
        }
    }
    if (ctx->structure->openBranch.elementCount) {
        if (fOnly) {
            fillOpenBranchBatchF<<<grid(ctx->structure->openBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->openBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->openBranchOffset, ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(), ctx->structure->openBranch.destination.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->structure->openBranch.alternateDestination.as<int>(), ctx->structure->openBranch.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->openBranch.dynamicRelocation ? 1 : 0, withPilotSelf(relocationTablesOf(ctx->structure->openBranch, ctx->structure.get()), ctx));
        } else {
            fillOpenBranchBatch<<<grid(ctx->structure->openBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->openBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->openBranchOffset, ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(), ctx->structure->openBranch.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->structure->openBranch.alternateDestination.as<int>(), ctx->structure->openBranch.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->openBranch.dynamicRelocation ? 1 : 0, withPilotSelf(relocationTablesOf(ctx->structure->openBranch, ctx->structure.get()), ctx));
        }
    }
    if (ctx->structure->shunt.elementCount) {
        if (fOnly) {
            fillShuntBatchF<<<grid(ctx->structure->shunt.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->shunt.elementCount, nS, n, nnz, ctx->nElements, ctx->shuntOffset, ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(), ctx->structure->shunt.destination.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nShuntAdmittanceOverrides, ctx->dShuntAdmittanceOverrideScenarioElement.as<int>(), ctx->dShuntAdmittanceOverrideDelta.as<double>(), ctx->structure->shunt.alternateDestination.as<int>(), ctx->structure->shunt.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->shunt.dynamicRelocation ? 1 : 0, withPilotSelf(relocationTablesOf(ctx->structure->shunt, ctx->structure.get()), ctx));
        } else {
            fillShuntBatch<<<grid(ctx->structure->shunt.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->shunt.elementCount, nS, n, nnz, ctx->nElements, ctx->shuntOffset, ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(), ctx->structure->shunt.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nShuntAdmittanceOverrides, ctx->dShuntAdmittanceOverrideScenarioElement.as<int>(), ctx->dShuntAdmittanceOverrideDelta.as<double>(), ctx->structure->shunt.alternateDestination.as<int>(), ctx->structure->shunt.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->shunt.dynamicRelocation ? 1 : 0, withPilotSelf(relocationTablesOf(ctx->structure->shunt, ctx->structure.get()), ctx));
        }
    }
    // voltage-dependent load model: own guarded block (NOT inside the shunt guard — a modeled
    // load needs no shunt present); residual always, Jacobian slot unless the F-only pass
    if (ctx->structure->nLoadModelTerms) {
        fillLoadModelBatch<<<grid(ctx->structure->nLoadModelTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->nLoadModelTerms, nS, n, nnz, ctx->structure->dLoadModelParameters.as<double>(), ctx->structure->dLoadModelIndices.as<int>(), ctx->structure->dLoadModelDestination.as<int>(), ctx->dX.as<double>(), fOnly ? nullptr : ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->structure->loadModelRelocated ? ctx->structure->dLoadModelAlternateRow.as<int>() : nullptr, ctx->structure->loadModelRelocated ? ctx->structure->dLoadModelAlternateDestination.as<int>() : nullptr, ctx->structure->loadModelRelocated ? ctx->dMode.as<int>() : nullptr);
    }
    if (ctx->structure->hvdcAcEmulation.elementCount) {
        if (fOnly) {
            fillHvdcBatchF<<<grid(ctx->structure->hvdcAcEmulation.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->hvdcAcEmulation.elementCount, nS, n, nnz, ctx->nElements, ctx->hvdcAcEmulationOffset, ctx->structure->hvdcAcEmulation.parameters.as<double>(), ctx->structure->hvdcAcEmulation.indices.as<int>(), ctx->structure->hvdcAcEmulation.destination.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->structure->hvdcAcEmulation.alternateDestination.as<int>(), ctx->structure->hvdcAcEmulation.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->hvdcAcEmulation.dynamicRelocation ? 1 : 0);
        } else {
            fillHvdcBatch<<<grid(ctx->structure->hvdcAcEmulation.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->hvdcAcEmulation.elementCount, nS, n, nnz, ctx->nElements, ctx->hvdcAcEmulationOffset, ctx->structure->hvdcAcEmulation.parameters.as<double>(), ctx->structure->hvdcAcEmulation.indices.as<int>(), ctx->structure->hvdcAcEmulation.destination.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->structure->hvdcAcEmulation.alternateDestination.as<int>(), ctx->structure->hvdcAcEmulation.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->hvdcAcEmulation.dynamicRelocation ? 1 : 0);
        }
    }
int distrQTraceRow = gpuDistrQTraceRow();
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
    if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: stream-ordered into the solve, no host sync
}

}  // namespace olfgpu
