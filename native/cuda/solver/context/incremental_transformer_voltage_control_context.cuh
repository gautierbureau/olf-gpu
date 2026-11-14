// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that IncrementalTransformerVoltageControlOuterLoop: the controller branches and groups, their taps and deadbands, and the
// per-scenario tap moves.
#ifndef OLF_GPU_SOLVER_CONTEXT_INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct IncrementalTransformerVoltageControlContext {
    // INCREMENTAL TRANSFORMER VOLTAGE CONTROL — in-kernel outer loop (GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL).
    // Per-controller static data uploaded once; per-(scenario, controller) mutable state allocated
    // at first use. Tap tables are CSR-packed: tapRows[tapStart[c]..tapStart[c+1]) holds nTaps rows of
    // 8 doubles [y, g1, g2, b1, b2, ksi, r1, a1] for controller c.
    int nIncrementalTransformerVoltageControlControllerBranches = 0;                                            // number of incremental TVC controllers
    DeviceBuf dIncrementalTransformerVoltageControlRhoRow, dIncrementalTransformerVoltageControlControlledBusVRow;                          // [nIncrementalTransformerVoltageControlControllerBranches] int: rho identity row, controlled bus V row
    DeviceBuf dIncrementalTransformerVoltageControlTargetV, dIncrementalTransformerVoltageControlHalfTargetDeadband;                       // [nIncrementalTransformerVoltageControlControllerBranches] double: target voltage, half deadband
    DeviceBuf dIncrementalTransformerVoltageControlElement;                                 // [nIncrementalTransformerVoltageControlControllerBranches] int: element index for disabled-mask check
    DeviceBuf dIncrementalTransformerVoltageControlTapStart;                                    // [nIncrementalTransformerVoltageControlControllerBranches+1] int: CSR pointer into flat tap table
    DeviceBuf dIncrementalTransformerVoltageControlTapRows;                                   // [totalTaps * 8] double: flat tap rows
    DeviceBuf dIncrementalTransformerVoltageControlBaseTap;                                   // [nIncrementalTransformerVoltageControlControllerBranches] int: base (extraction-time) tap index
    // Per-(scenario, controller) mutable state:
    DeviceBuf dIncrementalTransformerVoltageControlCurrentTap;                                    // [nS * nIncrementalTransformerVoltageControlControllerBranches] int: current tap position
    DeviceBuf dIncrementalTransformerVoltageControlCurrentDirection, dIncrementalTransformerVoltageControlDirectionChangeCount, dIncrementalTransformerVoltageControlAllowedDirection; // [nS * nIncrementalTransformerVoltageControlControllerBranches] int: ratchet state
    // Per-controller group structure (CSR): controllers sharing a controlled bus are grouped for
    // the one-tap-at-a-time round-robin (OLF's adjustWithSeveralControllers).
    DeviceBuf dIncrementalTransformerVoltageControlGroupControllerStart, dIncrementalTransformerVoltageControlGroupController;                   // CSR: groupControllerStart[nGroups+1], groupController[nIncrementalTransformerVoltageControlControllerBranches]
    int nIncrementalTransformerVoltageControlGroups = 0;
    int incrementalTransformerVoltageControlMaxTapShift = 1;                                  // OLF incrementalTransformerRatioTapControlOuterLoopMaxTapShift
    // Per-pass scratch and the previous tap (to apply only the pairs that moved this pass).
    DeviceBuf dIncrementalTransformerVoltageControlOutOfDeadband, dIncrementalTransformerVoltageControlDiffV, dIncrementalTransformerVoltageControlSensitivity, dIncrementalTransformerVoltageControlAppliedTap;
    // Host mirrors of the static tables, for the tap-override rebuild and the rho-row lookup.
    std::vector<int> hIncrementalTransformerVoltageControlRhoRow, hIncrementalTransformerVoltageControlElement, hIncrementalTransformerVoltageControlBaseTap, hIncrementalTransformerVoltageControlTapStart;
    std::vector<double> hIncrementalTransformerVoltageControlTapRows;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL_CONTEXT_CUH
