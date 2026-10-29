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
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "scatter.cuh"
#include "../fixedmatrix/reactive_relocation.cuh"
#include "../fixedmatrix/cross_identity.cuh"
#include "../solver/driver_steps.cuh"
#include "branch_fill.cuh"
#include "bus_injection_fill.cuh"
#include "distribution_fill.cuh"
#include "zero_impedance_fill.cuh"
#include "fill_layout.cuh"
#include "../outerloop/pilot_voltage.cuh"
#include "fill_launch.cuh"
#include "../solver/scenario_override_tables.cuh"

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
                                  const int* colIdx, const int* relocatedChildOwnRow, BatchLayout layout) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int r, s;
    itemAndScenario(tid, n, layout, r, s);
    if (!active[s]) return;
    int m = mode[(size_t) s * n + r];
    const ScenarioValues valsS = scenarioEntries(vals, s, nnz, layout);  // none on a mismatch-only fill: F alone
    const ScenarioValues FS = scenarioEntries(F, s, n, layout);
    if (m == GpuRowMode::CROSS_FIXED_VARIABLE) {
        writeCrossIdentityRow(r, s, n, rowPtr, colIdx, relocatedChildOwnRow, x, target, valsS, FS);
    } else if (m == GpuRowMode::FIXED_VARIABLE) {  // identity: v/phi/rho/a1/b held at target
        if (valsS) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
                valsS[k] = 0.0;
            }
            valsS[diagPos[r]] = 1.0;
        }
        FS[r] = x[(size_t) s * n + r] - target[(size_t) s * n + r];
    } else if (m == GpuRowMode::BUS_TARGET_V_WITH_SLOPE) {  // SLOPE-SVC: V + slope*sumQ = target — the natural
        double slope = rowSlope[r];                          // reactive fill stands, scaled, plus the v term
        if (valsS) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
                valsS[k] *= slope;
            }
            valsS[diagPos[r]] += 1.0;
        }
        size_t idx = (size_t) s * n + r;
        FS[r] = slope * (FS[r] + target[idx]) + x[idx] - target[idx];
    }
}

__global__ void initFBatch(int n, int nScenarios, const double* target, double* F, const char* active, double* qbuf) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int s = tid / n;
    if (!active[s]) return;
    F[tid] = -target[tid];
    if (qbuf != nullptr) {
        qbuf[tid] = 0.0;
    }
}

// Seed the per-(controller, scenario) limitType from the BASE state, so a controller that OLF had already
// switched to PQ in the base case (packed by the extractor as a voltage-CAPABLE PQ bus, base row mode 1)
// starts PQ at the RIGHT limit and can be switched BACK to PV by a relieving contingency. A base-PV
// controller (mode 0) starts at limitType 0. For a base-PQ controller, v below its target ⇒ it is producing
// max reactive yet cannot reach target ⇒ PQ at maxQ (2); v above target ⇒ PQ at minQ (1). switchCount
// starts at 0 either way (a separate memset).
const int* pilotSelfReactiveLimitEntryOrNull(BatchSolverContext* ctx) {
    return ctx->hasPilotSelf && ctx->dPilotSelfReactiveLimitEntry.p != nullptr ? ctx->dPilotSelfReactiveLimitEntry.template as<int>() : nullptr;
}

// Blocks per scenario of zeroActiveValues: each thread zeroes every (blocks x BLOCK)-th value of its scenario.
constexpr int ZERO_BLOCKS_PER_SCENARIO = 32;

// The Jacobian values of the ACTIVE scenarios zeroed before a fill, one row of blocks per scenario (blockIdx.y). A
// retired scenario's values are not read before writeIdentityForInactive rewrites them - the fills and the row modes
// skip it, the zero-diagonal repair finds what it finds there and the identity write replaces it (its F is zeroed, so
// the right-hand side's repair adds zeros) - so zeroing them was a pass over every scenario's values per fill, most of
// them retired: 0.19 s of memsets on rte6515's call 0, 0.40 s on network.xiidm's (cold and warm runs).
__global__ void zeroActiveValues(int nnz, const char* active, double* vals, char* identityLane) {
    if (!active[blockIdx.y]) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        identityLane[blockIdx.y] = 0;                        // its values are about to be a Jacobian's again
    }
    double* v = vals + static_cast<size_t>(blockIdx.y) * nnz;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x) {
        v[k] = 0.0;
    }
}

// The same on the interleaved layout: one thread per value and SECTOR_SCENARIOS consecutive scenario slots (one 32-byte
// sector: the stride is a multiple of it), written whole when they are all active.
__global__ void zeroActiveInterleavedValues(int nnz, BatchLayout layout, const char* active, double* vals,
                                            char* identityLane) {
    const int groups = layout.stride / SECTOR_SCENARIOS;
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(nnz) * groups) return;
    const int s0 = static_cast<int>(t % groups) * SECTOR_SCENARIOS;
    double* v = vals + t / groups * layout.stride + s0;
    bool on[SECTOR_SCENARIOS];
    bool all = true;
    for (int q = 0; q < SECTOR_SCENARIOS; q++) {
        on[q] = s0 + q < layout.nS && active[s0 + q];
        all = all && on[q];
    }
    if (t < static_cast<size_t>(groups)) {
        for (int q = 0; q < SECTOR_SCENARIOS; q++) {
            if (on[q]) {
                identityLane[s0 + q] = 0;                    // its values are about to be a Jacobian's again
            }
        }
    }
    if (all) {
        reinterpret_cast<double2*>(v)[0] = make_double2(0.0, 0.0);
        reinterpret_cast<double2*>(v)[1] = make_double2(0.0, 0.0);
    } else {
        for (int q = 0; q < SECTOR_SCENARIOS; q++) {
            if (on[q]) {
                v[q] = 0.0;
            }
        }
    }
}

// Make a RETIRED scenario's system an identity (diagonal 1, mismatch 0) so it stays
// well-posed inside the uniform batched factorization (a singular sub-block would fail the
// whole batch) and its state is frozen (dx = 0). A scenario retired THIS iteration still
// holds its filled values, so the row is zeroed before the diagonal is set — this is
// correct whether the scenario was just filled or has been inactive (zero) for a while. A scenario whose identity
// an earlier iteration of this solve already wrote (identityLane, see markIdentityLanes) holds it still and is skipped.
__global__ void writeIdentityForInactive(int n, int nScenarios, int nnz, const int* rowPtr,
                                         const int* diagPos, const char* active, const char* identityLane,
                                         double* vals, double* F, BatchLayout layout) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int r, s;
    itemAndScenario(tid, n, layout, r, s);
    if (active[s] || identityLane[s]) return;
    const ScenarioValues valsS = scenarioEntries(vals, s, nnz, layout);
    for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
        valsS[k] = 0.0;
    }
    valsS[diagPos[r]] = 1.0;
    F[(size_t) s * n + r] = 0.0;
}

// After writeIdentityForInactive: every retired scenario holds its identity system, every active one is to be filled.
__global__ void markIdentityLanes(int nScenarios, const char* active, char* identityLane) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < nScenarios) {
        identityLane[s] = active[s] ? 0 : 1;
    }
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

void fillTerms(BatchSolverContext* ctx, DriverPass& drv, bool fOnly, double* vals, double* F, double* qbufClosed,
               BatchLayout layout) {
    const int& n = drv.n;
    const int& nnz = drv.nnz;
    const int& nS = drv.nS;
    if (ctx->structure->closedBranch.elementCount) {
        if (fOnly) {
            fillClosedBranchBatchF<<<grid(ctx->structure->closedBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->closedBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->closedBranchOffset, ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->closedBranch.destination.as<int>(), ctx->dX.as<double>(), F, ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), closedBranchParameterOverrides(ctx), ctx->structure->closedBranch.alternateDestination.as<int>(), ctx->structure->closedBranch.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->closedBranch.dynamicRelocation ? 1 : 0, ctx->structure->closedBranch.relocatedElement.as<char>(), ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlDestination.as<int>() : nullptr, ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlSelector.as<int>() : nullptr, withPilotSelf(relocationTablesOf(ctx->structure->closedBranch, ctx->structure.get()), ctx), qbufClosed, layout);
        } else {
            fillClosedBranchBatch<<<grid(ctx->structure->closedBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->closedBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->closedBranchOffset, ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->closedBranch.destination.as<int>(), ctx->dX.as<double>(), vals, F, ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), closedBranchParameterOverrides(ctx), ctx->dMode.as<int>(), ctx->structure->closedBranch.dynamicRelocation ? 1 : 0, ctx->structure->closedBranch.relocatedElement.as<char>(), ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlDestination.as<int>() : nullptr, ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlSelector.as<int>() : nullptr, qbufClosed, layout);
            if (ctx->structure->closedBranch.dynamicRelocation && ctx->structure->closedBranch.relocatedCount > 0) {
                fillRelocatedClosedBranchBatch<<<grid(ctx->structure->closedBranch.relocatedCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->closedBranch.relocatedCount, ctx->structure->closedBranch.relocatedElements.as<int>(), nS, n, nnz, ctx->nElements, ctx->closedBranchOffset, ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->closedBranch.destination.as<int>(), ctx->dX.as<double>(), vals, F, ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), closedBranchParameterOverrides(ctx), ctx->structure->closedBranch.alternateDestination.as<int>(), ctx->structure->closedBranch.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlDestination.as<int>() : nullptr, ctx->structure->closedBranch.hasPhaseControl ? ctx->structure->closedBranch.phaseControlSelector.as<int>() : nullptr, withPilotSelf(relocationTablesOf(ctx->structure->closedBranch, ctx->structure.get()), ctx), qbufClosed, layout);
            }
        }
    }
    if (ctx->structure->openBranch.elementCount) {
        if (fOnly) {
            fillOpenBranchBatchF<<<grid(ctx->structure->openBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->openBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->openBranchOffset, ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(), ctx->structure->openBranch.destination.as<int>(), ctx->dX.as<double>(), F, ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->structure->openBranch.alternateDestination.as<int>(), ctx->structure->openBranch.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->openBranch.dynamicRelocation ? 1 : 0, withPilotSelf(relocationTablesOf(ctx->structure->openBranch, ctx->structure.get()), ctx), layout);
        } else {
            fillOpenBranchBatch<<<grid(ctx->structure->openBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->openBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->openBranchOffset, ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(), ctx->structure->openBranch.destination.as<int>(), ctx->dX.as<double>(), vals, F, ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->structure->openBranch.alternateDestination.as<int>(), ctx->structure->openBranch.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->openBranch.dynamicRelocation ? 1 : 0, withPilotSelf(relocationTablesOf(ctx->structure->openBranch, ctx->structure.get()), ctx), layout);
        }
    }
    if (ctx->structure->shunt.elementCount) {
        if (fOnly) {
            fillShuntBatchF<<<grid(ctx->structure->shunt.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->shunt.elementCount, nS, n, nnz, ctx->nElements, ctx->shuntOffset, ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(), ctx->structure->shunt.destination.as<int>(), ctx->dX.as<double>(), F, ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), shuntAdmittanceOverrides(ctx), ctx->structure->shunt.alternateDestination.as<int>(), ctx->structure->shunt.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->shunt.dynamicRelocation ? 1 : 0, withPilotSelf(relocationTablesOf(ctx->structure->shunt, ctx->structure.get()), ctx), layout);
        } else {
            fillShuntBatch<<<grid(ctx->structure->shunt.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->shunt.elementCount, nS, n, nnz, ctx->nElements, ctx->shuntOffset, ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(), ctx->structure->shunt.destination.as<int>(), ctx->dX.as<double>(), vals, F, ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), shuntAdmittanceOverrides(ctx), ctx->structure->shunt.alternateDestination.as<int>(), ctx->structure->shunt.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->shunt.dynamicRelocation ? 1 : 0, withPilotSelf(relocationTablesOf(ctx->structure->shunt, ctx->structure.get()), ctx), layout);
        }
    }
    // voltage-dependent load model: own guarded block (NOT inside the shunt guard — a modeled
    // load needs no shunt present); residual always, Jacobian slot unless the F-only pass
    if (ctx->structure->nLoadModelTerms) {
        fillLoadModelBatch<<<grid(ctx->structure->nLoadModelTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->nLoadModelTerms, nS, n, nnz, ctx->structure->dLoadModelParameters.as<double>(), ctx->structure->dLoadModelIndices.as<int>(), ctx->structure->dLoadModelDestination.as<int>(), ctx->dX.as<double>(), vals, F, ctx->dActive.as<char>(), ctx->structure->loadModelRelocated ? ctx->structure->dLoadModelAlternateRow.as<int>() : nullptr, ctx->structure->loadModelRelocated ? ctx->structure->dLoadModelAlternateDestination.as<int>() : nullptr, ctx->structure->loadModelRelocated ? ctx->dMode.as<int>() : nullptr, layout);
    }
    if (ctx->structure->hvdcAcEmulation.elementCount) {
        if (fOnly) {
            fillHvdcBatchF<<<grid(ctx->structure->hvdcAcEmulation.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->hvdcAcEmulation.elementCount, nS, n, nnz, ctx->nElements, ctx->hvdcAcEmulationOffset, ctx->structure->hvdcAcEmulation.parameters.as<double>(), ctx->structure->hvdcAcEmulation.indices.as<int>(), ctx->structure->hvdcAcEmulation.destination.as<int>(), ctx->dX.as<double>(), F, ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->structure->hvdcAcEmulation.alternateDestination.as<int>(), ctx->structure->hvdcAcEmulation.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->hvdcAcEmulation.dynamicRelocation ? 1 : 0, layout);
        } else {
            fillHvdcBatch<<<grid(ctx->structure->hvdcAcEmulation.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->hvdcAcEmulation.elementCount, nS, n, nnz, ctx->nElements, ctx->hvdcAcEmulationOffset, ctx->structure->hvdcAcEmulation.parameters.as<double>(), ctx->structure->hvdcAcEmulation.indices.as<int>(), ctx->structure->hvdcAcEmulation.destination.as<int>(), ctx->dX.as<double>(), vals, F, ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->structure->hvdcAcEmulation.alternateDestination.as<int>(), ctx->structure->hvdcAcEmulation.selectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->hvdcAcEmulation.dynamicRelocation ? 1 : 0, layout);
        }
    }
int distrQTraceRow = gpuDistrQTraceRow();
if (ctx->nDistrQ) fillDistrQBatch<<<grid(ctx->nDistrQ * nS), BLOCK, 0, ctx->stream>>>(ctx->nDistrQ, nS, n, nnz, ctx->nElements, ctx->closedBranchOffset, ctx->openBranchOffset, ctx->structure->dDistrQElement.as<int>(), ctx->structure->dDistrQSide.as<int>(), ctx->structure->dDistrQKind.as<int>(), ctx->structure->dDistrQWeight.as<double>(), ctx->structure->dDistrQDestination.as<int>(), ctx->structure->dDistrQAlternateDestination.as<int>(), ctx->structure->dDistrQSelectorRow.as<int>(), ctx->structure->dDistrQDummyQRow.as<int>(), ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(), ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(), ctx->dX.as<double>(), vals, F, ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->distrQWeightOverrides.view(),
            ctx->distrQKeyOverrides.view(), ctx->dMode.as<int>(), ctx->distrQInactiveMembers.view(),
            ctx->nDistrQDynamicKeys ? ctx->dDistrQRowMember.as<int>() : nullptr, ctx->nDistrQDynamicKeys || ctx->hasDistrQRelocationGroups ? ctx->dDistrQInjectionMember.as<int>() : nullptr, ctx->nDistrQDynamicKeys ? ctx->dDistrQMemberKey.as<double>() : nullptr, ctx->nDistrQDynamicKeys ? ctx->dDistrQMemberReactiveLimitEntry.as<int>() : nullptr, ctx->nDistrQDynamicKeys ? ctx->dDistrQMemberGroupStart.as<int>() : nullptr, ctx->nDistrQDynamicKeys ? ctx->dDistrQMemberGroupCount.as<int>() : nullptr, ctx->dReactiveLimitQLimitType.as<char>(), ctx->nReactiveLimitEntries, ctx->hasPilotSelf && ctx->dPilotSelfCarrierSkip.p != nullptr ? ctx->dPilotSelfCarrierSkip.as<char>() : nullptr, ctx->structure->nReactiveRelocationGroups, ctx->hasDistrQRelocationGroups ? ctx->dDistrQMemberRelocationGroup.as<int>() : nullptr, ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->dReactiveRelocationGroupStart.as<int>() : nullptr, ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->dReactiveRelocationCandidateRows.as<int>() : nullptr, closedBranchParameterOverrides(ctx), shuntAdmittanceOverrides(ctx), distrQTraceRow,
                ctx->hasDistrQSubgroups ? ctx->dDistrQSubgroup.as<char>() : nullptr, ctx->hasDistrQSubgroups ? ctx->dReactiveRelocationDistrQMemberStart.as<int>() : nullptr, ctx->nDistrQMembers, layout);
    if (ctx->nDistrRho) fillDistrRhoBatch<<<grid(ctx->nDistrRho * nS), BLOCK, 0, ctx->stream>>>(ctx->nDistrRho, nS, n, nnz, ctx->structure->dDistrRhoColumn.as<int>(), ctx->structure->dDistrRhoCoefficient.as<double>(), ctx->structure->dDistrRhoDestination.as<int>(), ctx->dX.as<double>(), vals, F, ctx->dActive.as<char>(), ctx->distrRhoCoefficientOverrides.view(), ctx->structure->hasDistrRhoRelocationGroups ? ctx->structure->dDistrRhoRelocationGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->structure->dReactiveRelocationGroupStart.as<int>(), ctx->structure->dReactiveRelocationCandidateRows.as<int>(), layout);
    if (ctx->nDistrShuntB) fillDistrShuntBBatch<<<grid(ctx->nDistrShuntB * nS), BLOCK, 0, ctx->stream>>>(ctx->nDistrShuntB, nS, n, nnz, ctx->structure->dDistrShuntBColumn.as<int>(), ctx->structure->dDistrShuntBCoefficient.as<double>(), ctx->structure->dDistrShuntBDestination.as<int>(), ctx->dX.as<double>(), vals, F, ctx->dActive.as<char>(), layout);
    if (ctx->nZeroImpedanceBranch) fillZeroImpBatch<<<grid(ctx->nZeroImpedanceBranch * nS), BLOCK, 0, ctx->stream>>>(ctx->nZeroImpedanceBranch, nS, n, nnz, ctx->structure->dZeroImpedanceBranchParameters.as<double>(), ctx->structure->dZeroImpedanceBranchIndices.as<int>(), ctx->structure->dZeroImpedanceBranchDestination.as<int>(), ctx->dX.as<double>(), vals, F, ctx->dActive.as<char>(), ctx->nBaseClosedZeroImpedanceBranches, ctx->nZeroImpedanceBranchCloseOverrides, ctx->dZeroImpedanceBranchCloseScenarioElement.as<int>(), ctx->nZeroImpedanceBranchOpenOverrides, ctx->dZeroImpedanceBranchOpenScenarioElement.as<int>(), ctx->structure->dZeroImpedanceAlternateDestination.as<int>(), ctx->structure->dZeroImpedanceSelectorRow.as<int>(), ctx->dMode.as<int>(), ctx->structure->zeroImpedanceDynamicRelocation ? 1 : 0, withPilotSelf(zeroImpedanceRelocationTables(ctx->structure.get()), ctx), ctx->nSpanningTreeOverrides, ctx->nSpanningTreeOverrides ? ctx->dSpanningTreeOverrideScenarioElementValue.as<int>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dZeroImpedanceBranchOpenMask.as<char>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dZeroImpedanceBranchCloseMask.as<char>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dSpanningTreeOverrideMask.as<char>() : nullptr, layout);
    applyRowModeBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->dMode.as<int>(), ctx->structure->dRowPtr.as<int>(), ctx->structure->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), vals, F, ctx->dActive.as<char>(), ctx->dRowSlope.as<double>(),
                ctx->structure->dColIdx.as<int>(), ctx->structure->dRelocatedChildOwnRow.p != nullptr ? ctx->structure->dRelocatedChildOwnRow.as<int>() : nullptr, layout);
}

// The batched fill of every active scenario: J and F, or F only (the iteration at the Newton budget tests the residual
// alone, and dVals keeps the last factorized Jacobian for the bad-solve score). The LINE_SEARCH backtrack re-evaluates
// through it at a reduced step.
void fillActiveBatch(BatchSolverContext* ctx, DriverPass& drv, bool fOnly) {
    const int& n = drv.n;
    const int& nnz = drv.nnz;
    const int& nS = drv.nS;
    const bool& deviceFixedIter = drv.deviceFixedIter;
    if (ctx->hasPilotSelf && ctx->dPilotSelfCarrierSkip.p != nullptr && ctx->structure->nReactiveRelocationGroups > 0) {
        // Recompute the PILOT-SELF gate from the CURRENT reactive-limit states BEFORE any family fill: such a pilot
        // has a Q balance to relocate only once a reactive limit has switched it to PQ, and the RL pass switches
        // buses between passes. The closed/open-branch and shunt fills are what route the pilot's raw q
        // through it (zeroImpedanceRelocationDestination), so it must be current when THEY run — launched after them (before the
        // DISTR_Q fill only) the FIRST fill of every context read an unset gate and put the PV
        // pilot's raw q on the carrier row on top of its DISTR_Q terms: row 0 of the 4-bus
        // pilot-as-controller fixture assembled -1.451 against OLF's -5.715 (= OLF + 1.0 * q_pilot), and
        // rte6515's base started from |f(x0)| 117.908 against OLF's 117.845.
        pilotSelfGateBatch<<<grid(ctx->structure->nReactiveRelocationGroups * nS), BLOCK, 0, ctx->stream>>>(
                ctx->structure->nReactiveRelocationGroups, nS, ctx->nReactiveLimitEntries, ctx->dPilotSelfReactiveLimitEntry.as<int>(),
                ctx->dReactiveLimitQLimitType.as<char>(),
                ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr, ctx->dPilotSelfCarrierSkip.as<char>());
    }
    // the mismatch the fills add into: dF itself on the scenario-major layout, the interleaved dFillMismatch, copied into
    // dF once the row modes are applied, on the interleaved one
    double* F = ctx->layout.interleaved ? ctx->dFillMismatch.as<double>() : ctx->dF.as<double>();
    if (!fOnly) {
        if (ctx->layout.interleaved) {
            const size_t threads = static_cast<size_t>(nnz) * (ctx->layout.stride / SECTOR_SCENARIOS);
            zeroActiveInterleavedValues<<<static_cast<unsigned>((threads + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(
                    nnz, ctx->layout, ctx->dActive.as<char>(), ctx->dVals.as<double>(), ctx->dIdentityLane.as<char>());
        } else if (ctx->mode == GpuLinearSolveMode::UBATCH || ctx->mode == GpuLinearSolveMode::BATCHED_LU) {
            // the uniform batches: a retired scenario gets its identity system before every solve (newton_iteration)
            const dim3 zeroGrid(std::min((nnz + BLOCK - 1) / BLOCK, ZERO_BLOCKS_PER_SCENARIO), nS);
            zeroActiveValues<<<zeroGrid, BLOCK, 0, ctx->stream>>>(nnz, ctx->dActive.as<char>(), ctx->dVals.as<double>(),
                                                                ctx->dIdentityLane.as<char>());
        } else {
            CUDA_CHECK(cudaMemsetAsync(ctx->dVals.p, 0, static_cast<size_t>(nnz) * nS * sizeof(double), ctx->stream));
        }
    }
    // A mismatch-only fill (the pass that ends a Newton budget) writes F alone. The values still hold the system the
    // last solve factorized, which the it == maxIter score of that solve reads (newton_iteration.cu): the DISTR_Q /
    // DISTR_RHO / DISTR_SHUNT_B, zero-impedance and row-mode passes added their Jacobian terms into them a second time
    // (a coupler's +-1 became +-2, a slope row was scaled twice), and the zero-diagonal repair took its copy of the
    // hazard rows and its choices again from the result. The repair has no F part: the right-hand side is repaired at
    // the solve.
    double* vals = fOnly ? nullptr : ctx->dVals.as<double>();
    if (ctx->layout.interleaved) {
        initInterleavedMismatch(n, ctx->layout, ctx->dTarget.as<double>(), F, ctx->dActive.as<char>(), ctx->dQbufClosed.as<double>(), ctx->stream);
    } else {
        initFBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dTarget.as<double>(), F, ctx->dActive.as<char>(), ctx->dQbufClosed.as<double>());
    }
    fillTerms(ctx, drv, fOnly, vals, F, ctx->dQbufClosed.as<double>(), ctx->layout);
    if (ctx->layout.interleaved) {
        copyToScenarioMajor(n, ctx->layout, F, ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->stream);
    }
    if (!fOnly) {
        repairZeroDiagonalRows(ctx, nS, vals);
    }
    if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: stream-ordered into the solve, no host sync
}

}  // namespace olfgpu
