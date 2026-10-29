// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Per-scenario override readers shared by the fill kernels and the outer-loop kernels: a scenario's tap
// row, shunt admittance delta, frozen identity rows, inactive DISTR_Q members, reactive keys and RL voltage
// targets. The tap and shunt overrides are per-slot tables; the other lists are a handful of entries per run,
// scanned linearly.
#ifndef OLF_GPU_EQUATIONS_SCENARIO_OVERRIDES_CUH
#define OLF_GPU_EQUATIONS_SCENARIO_OVERRIDES_CUH

#include <cuda_runtime.h>

namespace olfgpu {

// Per-scenario closed-branch TAP overrides. Each is the target tap's FULL static row (y g1 g2 b1 b2 ksi r1 a1)
// replacing the element's closedBranchParameters row — per-tap r/x/g/b changes are first-class (y/ksi move).
// Java guards that the branch's ratio/phase are NOT state variables (r1col/a1col == -1), so the state path never
// conflicts. An override invalidates the element's precomputed trig statics (ksi/a1 may differ) — the caller must
// take the raw-sincos path (same fallback as a state-variable a1).
// Two sources, the first match wins:
// - the tap loops' moves (incremental TVC, continuous phase, incremental flow, discrete TVC members, in that order
//   on an element several act on): per (slot, controller) tables the loops publish on the device after each check
//   that moves a tap (publishClosedBranchParameterOverrides);
// - the caller's overrides (tap/PST actions on FIXED-tap transformers): a per-slot CSR, in the caller's order.
// A loop's move therefore shadows a caller's override of the same transformer.
struct ClosedBranchParameterOverrides {
    int nControllers;                                        // every tap loop's controllers, one numbering (0: no tap loop)
    const int* elementControllerStart;                       // per closed-branch element + 1: CSR into elementController
    const int* elementController;                            // the controllers acting on the element, in precedence order
    const char* tapMoved;                                    // nScenarios x nControllers: the tap is off its base tap
    const double* piModel;                                   // nScenarios x nControllers x 8: the moved tap's row
    const int* scenarioStart;                                // nScenarios + 1: CSR of the caller's overrides per slot
    const int* element;                                      // closed-branch element
    const double* values;                                    // 8 per entry
};

// Returns true when out8 holds the replacement row of closed-branch element e in slot s.
__device__ inline bool findTapOverride(const ClosedBranchParameterOverrides& o, int s, int e, double* out8) {
    if (o.nControllers > 0) {
        for (int k = o.elementControllerStart[e]; k < o.elementControllerStart[e + 1]; k++) {
            size_t sc = (size_t) s * o.nControllers + o.elementController[k];
            if (o.tapMoved[sc]) {
                for (int c = 0; c < 8; c++) {
                    out8[c] = o.piModel[8 * sc + c];
                }
                return true;
            }
        }
    }
    for (int k = o.scenarioStart[s]; k < o.scenarioStart[s + 1]; k++) {
        if (o.element[k] == e) {
            for (int c = 0; c < 8; c++) {
                out8[c] = o.values[8 * k + c];
            }
            return true;
        }
    }
    return false;
}

// Per-scenario shunt admittance overrides (section shifts / reconnection): (dG, dB) deltas added onto the static
// per-element g/b of a PACK shunt element. Every match adds, in this order: the caller's entries (a per-slot CSR,
// in the caller's order), then the incremental shunt loop's moved sections per controller (published on the device
// after each check that moves a section: publishShuntAdmittanceOverrides).
struct ShuntAdmittanceOverrides {
    int nControllers;                                        // the shunt loop's controllers (0: no shunt loop)
    const int* elementControllerStart;                       // per PACK shunt element + 1: CSR into elementController
    const int* elementController;                            // in controller order
    const char* sectionsMoved;                               // nScenarios x nControllers: (dG, dB) is not (0, 0)
    const double* controllerDelta;                           // nScenarios x nControllers x 2: (dG, dB)
    const int* scenarioStart;                                // nScenarios + 1: CSR of the caller's overrides per slot
    const int* element;                                      // PACK shunt element
    const double* delta;                                     // (dG, dB) per entry
};

__device__ inline void applyShuntOverride(const ShuntAdmittanceOverrides& o, int s, int e, double& g, double& b) {
    for (int k = o.scenarioStart[s]; k < o.scenarioStart[s + 1]; k++) {
        if (o.element[k] == e) {
            g += o.delta[2 * k];
            b += o.delta[2 * k + 1];
        }
    }
    if (o.nControllers > 0) {
        for (int k = o.elementControllerStart[e]; k < o.elementControllerStart[e + 1]; k++) {
            size_t sc = (size_t) s * o.nControllers + o.elementController[k];
            if (o.sectionsMoved[sc]) {
                g += o.controllerDelta[2 * sc];
                b += o.controllerDelta[2 * sc + 1];
            }
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
// and renormalizes over the rest — the same treatment a member SWITCHED TO PQ gets, except its target
// contribution stays its own targetQ instead of a reactive limit.
__device__ __forceinline__ bool distrQMemberInactive(int nInact, const int* inactSlotMember, int s, int m) {
    for (int k = 0; k < nInact; k++) {
        if (inactSlotMember[2 * k] == s && inactSlotMember[2 * k + 1] == m) {
            return true;
        }
    }
    return false;
}

// A member's reactive key this scenario: the per-scenario override (OLF's updateReactiveKeys after the scenario's
// generator losses) when the plan wrote one, else the base pack's. Tiny lists, a linear scan per lookup.
__device__ inline double distrQKeyOf(int nKeyOv, const int* keyOvIdx, const double* keyOvKey, int s, int m,
                                 const double* mKey) {
    for (int i = 0; i < nKeyOv; i++) {
        if (keyOvIdx[2 * i] == s && keyOvIdx[2 * i + 1] == m) return keyOvKey[i];
    }
    return mKey[m];
}

// Per-scenario RL V-target override (secondary voltage control moves generator-VC targets): entries are
// (slot, rl index) pairs with the REPLACEMENT target. Linear scan — the list is a handful of entries.
__device__ inline double reactiveLimitTargetVOf(int nReactiveLimitTargetVOverrides, const int* reactiveLimitTargetVOverrideScenarioEntry, const double* reactiveLimitTargetVOverrideValue,
                                    const double* secondaryVoltageControlReactiveLimitTargetV, int nReactiveLimitEntries, int s, int i, double baseTarget) {
    if (secondaryVoltageControlReactiveLimitTargetV != nullptr) {                                // a zone moved this controller's target: the
        double v = secondaryVoltageControlReactiveLimitTargetV[(size_t) s * nReactiveLimitEntries + i];            // switch-back must compare against the MOVED one
        if (!isnan(v)) {
            return v;
        }
    }
    for (int k = 0; k < nReactiveLimitTargetVOverrides; k++) {
        if (reactiveLimitTargetVOverrideScenarioEntry[2 * k] == s && reactiveLimitTargetVOverrideScenarioEntry[2 * k + 1] == i) {
            return reactiveLimitTargetVOverrideValue[k];
        }
    }
    return baseTarget;
}

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_SCENARIO_OVERRIDES_CUH
