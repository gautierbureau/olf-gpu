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

namespace olfgpu {

/** MonitoringVoltage (OLF MonitoringVoltageOuterLoop): a standby SVC whose monitored bus leaves its
 *  band switches to voltage control at the corresponding setpoint. One-way — once a controller has
 *  fired for a scenario it is never re-checked, exactly as OLF's activatedTargetV bookkeeping. */
__global__ void monitoringLoopBatch(int nmon, int nScenarios, int n,
                                    const int* monVRow, const double* monBand,
                                    const int* deltaPtr, const int* deltaRow,
                                    const int* deltaMode, const double* deltaTarget,
                                    const double* x, double* target, int* mode,
                                    double* activated, const char* outerLoopCheckActive, char* switched) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nmon * nScenarios) return;
    int c = k % nmon;
    int s = k / nmon;
    if (!outerLoopCheckActive[s]) return;
    size_t slot = (size_t) s * nmon + c;
    if (!isnan(activated[slot])) return;                     // already switched — a normal PV bus now
    double v = x[(size_t) s * n + monVRow[c]];
    double lowThr = monBand[c * 4];
    double highThr = monBand[c * 4 + 1];
    double lowTgt = monBand[c * 4 + 2];
    double highTgt = monBand[c * 4 + 3];
    double chosen;
    if (v > highThr) {
        chosen = highTgt;
    } else if (v < lowThr) {
        chosen = lowTgt;
    } else {
        return;
    }
    activated[slot] = chosen;
    for (int d = deltaPtr[c]; d < deltaPtr[c + 1]; d++) {
        int row = deltaRow[d];
        mode[(size_t) s * n + row] = deltaMode[d];
        target[(size_t) s * n + row] = row == monVRow[c] ? chosen : deltaTarget[d];
    }
    switched[s] = 1;
}

// Per-solve reset. MonitoringVoltage: nothing has fired yet in this solve.
void monitoringResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    std::vector<double> nan(static_cast<size_t>(ctx->nmon) * nS, std::nan(""));
    CUDA_CHECK(cudaMemcpyAsync(ctx->dMonActivated.p, nan.data(),
            nan.size() * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

// One check of the loop on every live scenario (GpuOuterLoopKind::MONITORING_VOLTAGE).
void monitoringStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    monitoringLoopBatch<<<grid(ctx->nmon * nS), BLOCK, 0, ctx->stream>>>(
            ctx->nmon, nS, n, ctx->dMonVRow.as<int>(), ctx->dMonBand.as<double>(),
            ctx->dMonDeltaPtr.as<int>(), ctx->dMonDeltaRow.as<int>(),
            ctx->dMonDeltaMode.as<int>(), ctx->dMonDeltaTarget.as<double>(),
            ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dMode.as<int>(),
            ctx->dMonActivated.as<double>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->dOuterLoopUnstable.as<char>());
}

}  // namespace olfgpu
