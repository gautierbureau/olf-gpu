// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that a contingency or a strategy changes in the network, installed per slot: shunt admittances,
// couplers opened or closed and spanning-tree flags, closed-branch tap and parameter changes, the transformer
// distribution coefficients, and the moves of the incremental loops on them.
#ifndef OLF_GPU_SOLVER_CONTEXT_SCENARIO_OVERRIDES_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_SCENARIO_OVERRIDES_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct ScenarioOverridesContext {
    // Per-slot shunt admittance overrides (scenario_override_tables.cuh): the caller's, a CSR per slot installed per
    // solve ((slot + 1) starts, PACK shunt element, (dG, dB)), and the incremental shunt loop's moved sections per
    // (slot, controller), with the controllers acting on each PACK shunt element.
    SlotTable shuntAdmittanceOverrides{1, 2};                            // per slot: the caller's (PACK shunt element; dG, dB)
    int nShuntAdmittanceOverrideControllers = 0;
    std::vector<int> hShuntAdmittanceOverrideControllerCounts;                       // the loop sizes the controller tables were built for
    DeviceBuf dShuntAdmittanceOverrideElementControllerStart, dShuntAdmittanceOverrideElementController;
    DeviceBuf dShuntAdmittanceOverrideSectionsMoved, dShuntAdmittanceOverrideControllerDelta;
    int nBaseClosedZeroImpedanceBranches = 0;                                         // zero-impedance elements [0,nBaseClosedZeroImpedanceBranches) always enabled; [nBaseClosedZeroImpedanceBranches,nZeroImpedanceBranch) are
                                                             // base-open closable switches gated by the zeroImpedanceBranchCloseOverrides list
    // Per (slot, zero-impedance element), dense: the switches the scenario OPENS and the base-open ones it CLOSES (1 = listed),
    // and OLF's spanning-tree flag where it differs from extraction (-1 = no override, else 0/1). The fill kernel read
    // them as (slot, element) lists, scanned PER THREAD - on network.xiidm's full non-busbar scope fillZeroImpBatch was the
    // top kernel, 32 s of 126 s (45 ms a launch).
    SlotRows<char> zeroImpedanceBranchOpenMask{0};
    SlotRows<char> zeroImpedanceBranchCloseMask{0};
    SlotRows<char> spanningTreeOverrideMask{-1};
    // Per-slot closed-branch tap overrides (scenario_override_tables.cuh): the caller's tap/PST actions, a CSR per slot
    // installed per solve ((slot + 1) starts, closed-branch element, full closedBranchParameters row), and the tap
    // loops' moved taps per (slot, controller), with the controllers acting on each closed-branch element.
    SlotTable closedBranchParameterOverrides{1, 8};                      // per slot: the caller's (closed-branch element; its row)
    int nClosedBranchParameterOverrideControllers = 0;
    std::vector<int> hClosedBranchParameterOverrideControllerCounts;                 // the loop sizes the controller tables were built for
    DeviceBuf dClosedBranchParameterOverrideElementControllerStart, dClosedBranchParameterOverrideElementController;
    DeviceBuf dClosedBranchParameterOverrideTapMoved, dClosedBranchParameterOverridePiModel;
    // DISTR_RHO coefficient overrides (setBatchDistrRhoCoefficientOverrides): a member outage renormalizes the
    // group's 1/count coefficients per scenario — per slot, (contribution; new coefficient), the last one wins.
    SlotTable distrRhoCoefficientOverrides{1, 1};
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_SCENARIO_OVERRIDES_CONTEXT_CUH
