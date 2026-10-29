// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Per-scenario override readers shared by the fill kernels and the outer-loop kernels: a scenario's tap
// row, shunt admittance delta, frozen identity rows, inactive DISTR_Q members, reactive keys and RL voltage
// targets. Each reads its own slot's entries (equations/slot_entries.cuh).
#ifndef OLF_GPU_EQUATIONS_SCENARIO_OVERRIDES_CUH
#define OLF_GPU_EQUATIONS_SCENARIO_OVERRIDES_CUH

#include <cuda_runtime.h>

#include "slot_entries.cuh"

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
// - the caller's overrides (tap/PST actions on FIXED-tap transformers): per slot, in the caller's order.
// A loop's move therefore shadows a caller's override of the same transformer.
struct ClosedBranchParameterOverrides {
    int nControllers;                                        // every tap loop's controllers, one numbering (0: no tap loop)
    const int* elementControllerStart;                       // per closed-branch element + 1: CSR into elementController
    const int* elementController;                            // the controllers acting on the element, in precedence order
    const char* tapMoved;                                    // nScenarios x nControllers: the tap is off its base tap
    const double* piModel;                                   // nScenarios x nControllers x 8: the moved tap's row
    SlotEntries caller;                                      // per slot: (closed-branch element; its 8-double row)
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
    int k = slotFirstEntryOf(o.caller, s, e);
    if (k < 0) {
        return false;
    }
    for (int c = 0; c < 8; c++) {
        out8[c] = o.caller.doubleOf(s, k, c);
    }
    return true;
}

// Per-scenario shunt admittance overrides (section shifts / reconnection): (dG, dB) deltas added onto the static
// per-element g/b of a PACK shunt element. Every match adds, in this order: the caller's entries (per slot, in the
// caller's order), then the incremental shunt loop's moved sections per controller (published on the device
// after each check that moves a section: publishShuntAdmittanceOverrides).
struct ShuntAdmittanceOverrides {
    int nControllers;                                        // the shunt loop's controllers (0: no shunt loop)
    const int* elementControllerStart;                       // per PACK shunt element + 1: CSR into elementController
    const int* elementController;                            // in controller order
    const char* sectionsMoved;                               // nScenarios x nControllers: (dG, dB) is not (0, 0)
    const double* controllerDelta;                           // nScenarios x nControllers x 2: (dG, dB)
    SlotEntries caller;                                      // per slot: (PACK shunt element; dG, dB)
};

__device__ inline void applyShuntOverride(const ShuntAdmittanceOverrides& o, int s, int e, double& g, double& b) {
    for (int k = 0; k < o.caller.countOf(s); k++) {
        if (o.caller.intOf(s, k) == e) {
            g += o.caller.doubleOf(s, k, 0);
            b += o.caller.doubleOf(s, k, 1);
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
__device__ __forceinline__ bool rowIsFrozen(const SlotEntries& frozenRows, int s, int row) {
    return slotFirstEntryOf(frozenRows, s, row) >= 0;
}

// A DISTR_Q controller that stops controlling in ONE scenario (its bus lost its last voltage
// generator while the group's other controllers survive). OLF's updateReactiveKeys zeroes its key
// and renormalizes over the rest — the same treatment a member SWITCHED TO PQ gets, except its target
// contribution stays its own targetQ instead of a reactive limit.
__device__ __forceinline__ bool distrQMemberInactive(const SlotEntries& inactiveMembers, int s, int m) {
    return slotFirstEntryOf(inactiveMembers, s, m) >= 0;
}

// A member's reactive key this scenario: the per-scenario override (OLF's updateReactiveKeys after the scenario's
// generator losses) when the plan wrote one, else the base pack's.
__device__ inline double distrQKeyOf(const SlotEntries& keyOverrides, int s, int m, const double* mKey) {
    int k = slotFirstEntryOf(keyOverrides, s, m);
    return k >= 0 ? keyOverrides.doubleOf(s, k) : mKey[m];
}

// Per-scenario RL V-target override (secondary voltage control moves generator-VC targets): per slot, (rl index;
// the REPLACEMENT target), the first one wins.
__device__ inline double reactiveLimitTargetVOf(const SlotEntries& reactiveLimitTargetVOverrides,
                                    const double* secondaryVoltageControlReactiveLimitTargetV, int nReactiveLimitEntries, int s, int i, double baseTarget) {
    if (secondaryVoltageControlReactiveLimitTargetV != nullptr) {                                // a zone moved this controller's target: the
        double v = secondaryVoltageControlReactiveLimitTargetV[(size_t) s * nReactiveLimitEntries + i];            // switch-back must compare against the MOVED one
        if (!isnan(v)) {
            return v;
        }
    }
    int k = slotFirstEntryOf(reactiveLimitTargetVOverrides, s, i);
    return k >= 0 ? reactiveLimitTargetVOverrides.doubleOf(s, k) : baseTarget;
}

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_SCENARIO_OVERRIDES_CUH
