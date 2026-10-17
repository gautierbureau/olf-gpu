// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See realistic_voltage_check.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../solver/driver_steps.cuh"
#include "realistic_voltage_check.cuh"

namespace olfgpu {

// AcloadFlowEngine.isStateUnrealistic, on the states INSIDE the sweep. OLF re-checks after every
// solver run once the last loop that can fix an unrealistic state has run, and a state outside
// [minRealisticVoltage, maxRealisticVoltage] on a non-fictitious bus at or above
// minNominalVoltageRealisticVoltageCheck ends the outer-loop cycle with UNREALISTIC_STATE — the run
// FAILS and reports no violations. The device only ever checked the FINAL restored state, so a
// scenario that passed through an unrealistic state and recovered came back CONVERGED where OLF
// reports FAILED. Flagged here after every inner solve and routed to the CPU replay, which applies
// OLF's own rule about WHICH solve the check belongs to (it depends on the configured loop order and
// on robust mode) instead of the device guessing at it. Rejecting is always sound: it changes the
// route, never the answer.
__global__ void realisticCheckBatch(int n, int nScenarios, const char* rowChecked, const double* x,
                                    const char* live, double minV, double maxV, char* bad) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nScenarios) return;
    int i = t % n;
    int s = t / n;
    if (!live[s] || !rowChecked[i]) return;
    double v = x[(size_t) s * n + i];
    if (v < minV || v > maxV) {
        bad[s] = 1;
    }
}

void rejectUnrealistic(BatchSolverContext* ctx, DriverPass& drv, const std::vector<char>& gate) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& active = drv.active;
    std::vector<char>& live = drv.live;
    int& loopIdx = drv.loopIdx;
    int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    BatchResult& res = drv.res;
    if (!ctx->dRealisticGate.p) ctx->dRealisticGate.alloc(nS * sizeof(char));
    if (!ctx->dRealisticBad.p) ctx->dRealisticBad.alloc(nS * sizeof(char));
    CUDA_CHECK(cudaMemcpyAsync(ctx->dRealisticGate.p, gate.data(), nS * sizeof(char),
                               cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemsetAsync(ctx->dRealisticBad.p, 0, nS * sizeof(char), ctx->stream));
    realisticCheckBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dRealisticRow.as<char>(),
            ctx->dX.as<double>(), ctx->dRealisticGate.as<char>(),
            ctx->realisticMinV, ctx->realisticMaxV, ctx->dRealisticBad.as<char>());
    std::vector<char> bad(nS);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(bad.data(), ctx->dRealisticBad.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
    static const bool traceUnrealistic = std::getenv("OLF_GPU_UNREALISTIC_TRACE") != nullptr;
    for (int s = 0; s < nS; s++) {
        if (bad[s]) {
            if (traceUnrealistic) {
                std::fprintf(stderr, "GPU_UNREALISTIC_SWEEP slot=%d outer=%d loop=%d band=[%.4f, %.4f]\n",
                             s, outer, loopOrder[loopIdx], ctx->realisticMinV, ctx->realisticMaxV);
            }
            traceReject(s, "unrealistic-state", outer, 0);
            res.status[s] = -2;                          // STATUS_REJECTED -> the CPU replays it
            live[s] = 0;
            active[s] = 0;
        }
    }
}

}  // namespace olfgpu
