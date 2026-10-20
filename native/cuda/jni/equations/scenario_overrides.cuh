// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Per-scenario override readers shared by the fill kernels and the outer-loop kernels: a scenario's tap
// row, shunt admittance delta, frozen identity rows, inactive DISTR_Q members, reactive keys and RL voltage
// targets. Each list is a handful of entries per run, scanned linearly.
#ifndef OLF_GPU_EQUATIONS_SCENARIO_OVERRIDES_CUH
#define OLF_GPU_EQUATIONS_SCENARIO_OVERRIDES_CUH

#include <cuda_runtime.h>

namespace olfgpu {

// Per-scenario closed-branch TAP override (tap/PST actions on FIXED-tap transformers): entries
// are (slot, cb elem) with the target tap's FULL static row (y g1 g2 b1 b2 ksi r1 a1)
// replacing the element's closedBranchParameters row — per-tap r/x/g/b changes are first-class (y/ksi move).
// Java guards that the branch's ratio/phase are NOT state variables (r1col/a1col == -1), so
// the state path never conflicts. An override invalidates the element's precomputed trig
// statics (ksi/a1 may differ) — the caller must take the raw-sincos path (same fallback as
// a state-variable a1). Returns true when out8 holds the replacement row.
__device__ inline bool findTapOverride(int nTapOv, const int* tapOvSlotElem, const double* tapOvVal,
                                       int s, int e, double* out8) {
    for (int k = 0; k < nTapOv; k++) {
        if (tapOvSlotElem[2 * k] == s && tapOvSlotElem[2 * k + 1] == e) {
            for (int c = 0; c < 8; c++) {
                out8[c] = tapOvVal[8 * k + c];
            }
            return true;
        }
    }
    return false;
}

// Per-scenario shunt admittance override (section shifts / reconnection): entries are
// (slot, PACK elem) pairs with (dG, dB) deltas added onto the static per-element g/b. Linear
// scan per thread — the list is a handful of entries per run.
__device__ inline void applyShuntOverride(int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                                          int s, int e, double& g, double& b) {
    for (int k = 0; k < nShuntAdmittanceOverrides; k++) {
        if (shuntAdmittanceOverrideScenarioElement[2 * k] == s && shuntAdmittanceOverrideScenarioElement[2 * k + 1] == e) {
            g += shuntAdmittanceOverrideDelta[2 * k];
            b += shuntAdmittanceOverrideDelta[2 * k + 1];
        }
    }
}

// A row an islanding outage FROZE to a flat identity state. Its target is not the outer loops' to
// write: OLF removes an islanded bus's equations altogether, so nothing there re-derives a target.
// Measured on rte6515 LINE-1525-590: the DISTR_Q target writer stamped 0.0 over a frozen voltage row
// still holding 1.0, which is an exact 1.0 residual on an identity row and the whole 1.192e-06 kV.
__device__ __forceinline__ bool rowIsFrozen(int nFrozen, const int* frozenSlotRow, int s, int row) {
    for (int k = 0; k < nFrozen; k++) {
        if (frozenSlotRow[2 * k] == s && frozenSlotRow[2 * k + 1] == row) return true;
    }
    return false;
}

// A DISTR_Q controller that stops controlling in ONE scenario (its bus lost its last voltage
// generator while the group's other controllers survive). OLF's updateReactiveKeys zeroes its key
// and renormalizes over the rest — the same treatment a PINNED member gets, except its target
// contribution stays its own targetQ instead of a reactive limit.
__device__ __forceinline__ bool dqMemberInactive(int nInact, const int* inactSlotMember, int s, int m) {
    for (int k = 0; k < nInact; k++) {
        if (inactSlotMember[2 * k] == s && inactSlotMember[2 * k + 1] == m) {
            return true;
        }
    }
    return false;
}

// A member's reactive key this scenario: the per-scenario override (OLF's updateReactiveKeys after the scenario's
// generator losses) when the plan wrote one, else the base pack's. Tiny lists, a linear scan per lookup.
__device__ inline double dqKeyOf(int nKeyOv, const int* keyOvIdx, const double* keyOvKey, int s, int m,
                                 const double* mKey) {
    for (int i = 0; i < nKeyOv; i++) {
        if (keyOvIdx[2 * i] == s && keyOvIdx[2 * i + 1] == m) return keyOvKey[i];
    }
    return mKey[m];
}

// Per-scenario RL V-target override (secondary voltage control moves generator-VC targets): entries are
// (slot, rl index) pairs with the REPLACEMENT target. Linear scan — the list is a handful of entries.
__device__ inline double rlTargetOf(int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    const double* svcRlVt, int nrl, int s, int i, double baseTarget) {
    if (svcRlVt != nullptr) {                                // a zone moved this controller's target: the
        double v = svcRlVt[(size_t) s * nrl + i];            // switch-back must compare against the MOVED one
        if (!isnan(v)) {
            return v;
        }
    }
    for (int k = 0; k < nRlVtOv; k++) {
        if (rlVtOvSlotIdx[2 * k] == s && rlVtOvSlotIdx[2 * k + 1] == i) {
            return rlVtOvVal[k];
        }
    }
    return baseTarget;
}

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_SCENARIO_OVERRIDES_CUH
