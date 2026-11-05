// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// docs/gpu-vs-olf.md section 2.3 (equations placed on another bus's row): the per-scenario destination of a
// relocated output (carrier, stock pair, stock placement, two-way alternate) and the scatters that use it.
#ifndef OLF_GPU_FIXEDMATRIX_RELOCATED_SCATTER_CUH
#define OLF_GPU_FIXEDMATRIX_RELOCATED_SCATTER_CUH

#include <cuda_runtime.h>
#include "../common/codes/gpu_row_mode.cuh"
#include "../equations/scatter.cuh"
#include "reactive_relocation.cuh"
#include "stock_placement.cuh"

namespace olfgpu {

// zero-impedance q-slot dynamic pick (F/J slots {0..3} per element): group -> mode-scan candidate slot,
// two-way -> alt when the select row is an identity or a BUS_TARGET_Q this scenario, else the frozen dest.
__device__ inline int zeroImpedanceRelocationDestination(int frozen, int slot, const int* alt, const int* sel,
                                const int* mode, ReactiveRelocationTables rt) {
    int rg = rt.relocationGroup != nullptr ? rt.relocationGroup[slot] : -1;
    if (rg >= 0) {
        if (rt.relocationDependentPilot != nullptr && rt.relocationDependentPilot[slot]) {
            return dependentPilotDestination(slot, frozen, alt[slot], sel[slot], mode, rt);
        }
        if (rt.relocationCandidateOffset[slot] < 0) {                              // STOCK PAIR: group slot holds the dummy ROW
            return stockPairPick(mode[rg], rt.relocationCandidateDestinations, -rt.relocationCandidateOffset[slot] - 1, frozen, mode, rt.relocatedChildOwnRow, rg);
        }
        if (rt.pilotSelfCarrierSkip != nullptr && rt.pilotSelfCarrierSkip[rg]) {
            return -1;                                       // pilot-self group, pilot still PV: no raw q
        }
        return rt.relocationCandidateDestinations[rt.relocationCandidateOffset[slot] + chooseCarrierRowInSubgroup(rg, rt.reactiveRelocationGroupStart, rt.reactiveRelocationCandidateRows, mode, rt.distrQSubgroup, rt.reactiveRelocationDistrQMemberStart)];
    }
    int sr = sel != nullptr ? sel[slot] : -1;
    if (sr <= -2) {
        return mode[-2 - sr] == GpuRowMode::FIXED_VARIABLE ? alt[slot] : frozen;       // stock-placement: identity only (see stockPlacementAlternateRow)
    }
    return (sr >= 0 && (mode[sr] == GpuRowMode::FIXED_VARIABLE || mode[sr] == GpuRowMode::BUS_TARGET_Q)) ? alt[slot] : frozen;
}

// Dynamic Q-relocation scatter: when an output's frozen Q row is an identity this scenario
// (mode[selectorRow] == 0), route it to the alternate candidate row instead. mode is this scenario's
// row-mode slice (length n). dest/alternateDestination/selectorRow share the per-output layout of scatterAdd.
// Mode 4 ("remote gen-VC member switched to PQ" — the member row hosts its OWN Q balance, plan-B2) also
// selects the alternate: those entries carry a SUPPRESSED frozen dest and their own row as the
// alternate, with selectorRow = the member row itself. TVC/shunt select rows never carry mode 4.
static __device__ void scatterAddRelocated(int outputCount, const int* dest, const int* alternateDestination, const int* selectorRow,
                              const int* mode, const double* o, ScenarioValues vals, ScenarioValues F,
                              ReactiveRelocationTables rt, int eOff) {
    for (int k = 0; k < outputCount; k++) {
        int d;
        int rg = rt.relocationGroup != nullptr ? rt.relocationGroup[eOff + k] : -1;
        if (rg >= 0 && rt.relocationDependentPilot != nullptr && rt.relocationDependentPilot[eOff + k]) {
            d = dependentPilotDestination(eOff + k, dest[k], alternateDestination[k], selectorRow[k], mode, rt);
        } else if (rg >= 0 && rt.relocationCandidateOffset[eOff + k] < 0) {        // STOCK PAIR: group slot holds the dummy ROW
            d = stockPairPick(mode[rg], rt.relocationCandidateDestinations, -rt.relocationCandidateOffset[eOff + k] - 1, dest[k], mode, rt.relocatedChildOwnRow, rg);
        } else if (rg >= 0 && rt.pilotSelfCarrierSkip != nullptr && rt.pilotSelfCarrierSkip[rg]) {
            continue;                                        // pilot-self group, pilot still PV: no raw q
        } else if (rg >= 0) {
            d = rt.relocationCandidateDestinations[rt.relocationCandidateOffset[eOff + k] + chooseCarrierRowInSubgroup(rg, rt.reactiveRelocationGroupStart, rt.reactiveRelocationCandidateRows, mode, rt.distrQSubgroup, rt.reactiveRelocationDistrQMemberStart)];
        } else {
            int sr = selectorRow[k];
            bool useAlternate = sr >= 0 ? (mode[sr] == GpuRowMode::FIXED_VARIABLE || mode[sr] == GpuRowMode::BUS_TARGET_Q)
                                  : (sr <= -2 && mode[-2 - sr] == GpuRowMode::FIXED_VARIABLE);   // stock-placement: identity only
            d = useAlternate ? alternateDestination[k] : dest[k];
        }
        if (d >= 0) {
            atomicAdd(&vals[d], o[k]);
        } else if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

// Dynamic Q-relocation, residual-only variant: route each residual to its alternate F row when the
// frozen row is an identity this scenario. dest/alternateDestination/selectorRow are the element's per-output slices;
// pos maps each of the nres residuals to its CLOSED_BRANCH_OUTPUT_COUNT output index.
static __device__ void scatterAddFRelocated(int nres, const int* dest, const int* alternateDestination, const int* selectorRow,
                               const int* pos, const int* mode, const double* o, ScenarioValues F,
                               ReactiveRelocationTables rt, int eOff) {
    for (int k = 0; k < nres; k++) {
        int idx = pos[k];
        int d;
        int rg = rt.relocationGroup != nullptr ? rt.relocationGroup[eOff + idx] : -1;
        if (rg >= 0 && rt.relocationDependentPilot != nullptr && rt.relocationDependentPilot[eOff + idx]) {
            d = dependentPilotDestination(eOff + idx, dest[idx], alternateDestination[idx], selectorRow[idx], mode, rt);
        } else if (rg >= 0 && rt.relocationCandidateOffset[eOff + idx] < 0) {      // STOCK PAIR: group slot holds the dummy ROW
            d = stockPairPick(mode[rg], rt.relocationCandidateDestinations, -rt.relocationCandidateOffset[eOff + idx] - 1, dest[idx], mode, rt.relocatedChildOwnRow, rg);
        } else if (rg >= 0 && rt.pilotSelfCarrierSkip != nullptr && rt.pilotSelfCarrierSkip[rg]) {
            continue;                                        // pilot-self group, pilot still PV: no raw q
        } else if (rg >= 0) {
            d = rt.relocationCandidateDestinations[rt.relocationCandidateOffset[eOff + idx] + chooseCarrierRowInSubgroup(rg, rt.reactiveRelocationGroupStart, rt.reactiveRelocationCandidateRows, mode, rt.distrQSubgroup, rt.reactiveRelocationDistrQMemberStart)];
        } else {
            int sr = selectorRow[idx];
            bool useAlternate = sr >= 0 ? (mode[sr] == GpuRowMode::FIXED_VARIABLE || mode[sr] == GpuRowMode::BUS_TARGET_Q)
                                  : (sr <= -2 && mode[-2 - sr] == GpuRowMode::FIXED_VARIABLE);   // stock-placement: identity only
            d = useAlternate ? alternateDestination[idx] : dest[idx];
        }
        if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

}  // namespace olfgpu

#endif  // OLF_GPU_FIXEDMATRIX_RELOCATED_SCATTER_CUH
