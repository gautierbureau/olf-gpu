// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The per-slot tap and shunt override tables the fills and the loops read (equations/scenario_overrides.cuh): the
// caller's overrides as a CSR per slot, installed per solve, and the tap and shunt loops' moves per (slot, controller),
// published on the device after each check that moves a tap or a section.
#ifndef OLF_GPU_SOLVER_SCENARIO_OVERRIDE_TABLES_CUH
#define OLF_GPU_SOLVER_SCENARIO_OVERRIDE_TABLES_CUH

#include "batch_context.cuh"
#include "../equations/scenario_overrides.cuh"

namespace olfgpu {

// The caller's tap overrides of this solve, (slot, closed-branch element) + the 8-double row each, as a CSR per slot
// in the caller's order; no tap loop has moved anything yet.
void installClosedBranchParameterOverrides(BatchSolverContext* ctx, int nOverrides, const int* scenarioElement,
                                           const double* values);

// The caller's shunt overrides of this solve, (slot, PACK shunt element) + (dG, dB) each, as a CSR per slot in the
// caller's order; the shunt loop has moved no section yet.
void installShuntAdmittanceOverrides(BatchSolverContext* ctx, int nOverrides, const int* scenarioElement,
                                     const double* delta);

// Every tap loop's current taps: a controller whose tap is off its base tap overrides its transformer's row in that
// slot. Device kernels on the context's stream, no host round trip.
void publishClosedBranchParameterOverrides(BatchSolverContext* ctx);

// The incremental shunt loop's current sections: per (slot, controller), the (dG, dB) of its sections off their base.
void publishShuntAdmittanceOverrides(BatchSolverContext* ctx);

// The tables as the kernels take them.
ClosedBranchParameterOverrides closedBranchParameterOverrides(BatchSolverContext* ctx);
ShuntAdmittanceOverrides shuntAdmittanceOverrides(BatchSolverContext* ctx);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_SCENARIO_OVERRIDE_TABLES_CUH
