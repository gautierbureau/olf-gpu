// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See voltage_monitoring.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

#include "voltage_monitoring.cuh"
#include "../solver/scenario_install.cuh"

namespace olfgpu {

/** MonitoringVoltage (OLF MonitoringVoltageOuterLoop): a standby SVC whose monitored bus leaves its
 *  band switches to voltage control at the corresponding setpoint. One-way — once a controller has
 *  fired for a scenario it is never re-checked, exactly as OLF's activatedTargetV bookkeeping. */
__global__ void monitoringVoltageLoopBatch(int nMonitoringVoltageControllers, int nScenarios, int n,
                                    const int* monitoringVoltageBusVRow, const double* monitoringVoltageBand,
                                    const int* deltaStart, const int* deltaRow,
                                    const int* deltaMode, const double* deltaTarget,
                                    const double* x, double* target, int* mode,
                                    double* activated, const char* outerLoopCheckActive, char* switched) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nMonitoringVoltageControllers * nScenarios) return;
    int c = k % nMonitoringVoltageControllers;
    int s = k / nMonitoringVoltageControllers;
    if (!outerLoopCheckActive[s]) return;
    size_t scenarioController = (size_t) s * nMonitoringVoltageControllers + c;
    if (!isnan(activated[scenarioController])) return;                     // already switched — a normal PV bus now
    double v = x[(size_t) s * n + monitoringVoltageBusVRow[c]];
    double lowThr = monitoringVoltageBand[c * 4];
    double highThr = monitoringVoltageBand[c * 4 + 1];
    double lowTgt = monitoringVoltageBand[c * 4 + 2];
    double highTgt = monitoringVoltageBand[c * 4 + 3];
    double chosen;
    if (v > highThr) {
        chosen = highTgt;
    } else if (v < lowThr) {
        chosen = lowTgt;
    } else {
        return;
    }
    activated[scenarioController] = chosen;
    for (int d = deltaStart[c]; d < deltaStart[c + 1]; d++) {
        int row = deltaRow[d];
        mode[(size_t) s * n + row] = deltaMode[d];
        target[(size_t) s * n + row] = row == monitoringVoltageBusVRow[c] ? chosen : deltaTarget[d];
    }
    switched[s] = 1;
}

// Per-solve reset. MonitoringVoltage: nothing has fired yet in this solve.
void monitoringVoltageResetSolve(BatchSolverContext* ctx, DriverPass&) {
    resetInstalledRows(ctx, ctx->dMonitoringVoltageActivatedTargetV.as<double>(), ctx->nMonitoringVoltageControllers, nullptr,
                       std::nan(""));
}

// One check of the loop on every live scenario (GpuOuterLoopKind::MONITORING_VOLTAGE).
void monitoringVoltageStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    monitoringVoltageLoopBatch<<<grid(ctx->nMonitoringVoltageControllers * nS), BLOCK, 0, ctx->stream>>>(
            ctx->nMonitoringVoltageControllers, nS, n, ctx->dMonitoringVoltageBusVRow.as<int>(), ctx->dMonitoringVoltageBand.as<double>(),
            ctx->dMonitoringVoltageDeltaStart.as<int>(), ctx->dMonitoringVoltageDeltaRow.as<int>(),
            ctx->dMonitoringVoltageDeltaMode.as<int>(), ctx->dMonitoringVoltageDeltaTarget.as<double>(),
            ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dMode.as<int>(),
            ctx->dMonitoringVoltageActivatedTargetV.as<double>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->dOuterLoopUnstable.as<char>());
}

}  // namespace olfgpu
