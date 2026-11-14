// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The part of a batched solver context that MonitoringVoltageOuterLoop: the SVCs standing by in monitoring mode, their voltage band and the row modes and targets
// they switch to when activated.
#ifndef OLF_GPU_SOLVER_CONTEXT_MONITORING_VOLTAGE_CONTEXT_CUH
#define OLF_GPU_SOLVER_CONTEXT_MONITORING_VOLTAGE_CONTEXT_CUH

#include <cublas_v2.h>

#include <vector>

#include "../../common/device_buf.cuh"
#include "../slot_rows.cuh"
#include "../slot_table.cuh"

namespace olfgpu {

struct MonitoringVoltageContext {
    // MonitoringVoltage (OLF order 6): the standby-SVC automaton, as a DRIVER PHASE. Per controller:
    // the monitored V row, its band and the two setpoints; plus a CSR list of (row, mode, target) the
    // activation writes. dMonitoringVoltageActivatedTargetV holds the chosen target per (controller, scenario), NaN until
    // it fires — OLF never re-checks a controller it has already switched.
    int nMonitoringVoltageControllers = 0;
    DeviceBuf dMonitoringVoltageBusVRow, dMonitoringVoltageBand, dMonitoringVoltageDeltaStart, dMonitoringVoltageDeltaRow, dMonitoringVoltageDeltaMode, dMonitoringVoltageDeltaTarget, dMonitoringVoltageActivatedTargetV;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_CONTEXT_MONITORING_VOLTAGE_CONTEXT_CUH
