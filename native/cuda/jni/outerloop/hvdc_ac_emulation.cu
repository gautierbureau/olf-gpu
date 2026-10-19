// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See hvdc_ac_emulation.cuh.
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
#include "../common/codes/gpu_ac_emulation_status.cuh"
#include "hvdc.cuh"

#include "hvdc_ac_emulation.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

// OLF AbstractHvdcAcEmulationFlowEquationTerm.getAbsActivePowerWithLosses + HvdcUtils losses.
__device__ inline double hvdcAbsActivePowerWithLosses(double pController, double lossController,
                                                      double lossNonController, double r) {
    double lineInputPower = (1.0 - lossController) * fabs(pController);
    double losses = r * lineInputPower * lineInputPower;     // HvdcUtils.getHvdcLineLosses(p, 1, r)
    return (1.0 - lossNonController) * (lineInputPower - losses);
}

/** OLF's FreezingHvdcACEmulation (mode 0) and AcHvdcAcEmulationLimits (mode 1) as one driver phase.
 *  Transitions are applied INCREMENTALLY to the converter-bus target rows — target += newDelta -
 *  appliedDelta — so they compose with whatever else already overrides that row (a lost load on a
 *  converter bus, the DS carry) without rebuilding the whole override list, which is what the host
 *  orchestrator had to do on every round. */
__global__ void hvdcAcEmulationLoopBatch(int nh, int nScenarios, int n, int nElements, int hvOffset,
                                         int mode,
                                         const int* hvElem, const int* hvBus1Row, const int* hvBus2Row,
                                         const int* hvBus1Phi, const int* hvBus2Phi,
                                         const int* rowMode, const int* relocOwn,
                                         const double* hvIn, const char* hvCtgOut,
                                         const double* x, double* target, char* disabled,
                                         int* status, double* appliedD1, double* appliedD2,
                                         const char* rlActive, char* switched, int traceSlot) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nh * nScenarios) return;
    int h = tid % nh;
    int s = tid / nh;
    if (!rlActive[s] || hvCtgOut[(size_t) s * nh + h]) return;    // outaged: the contingency owns that bit
    const double* p = hvIn + (size_t) h * 9;                 // p0 k pMax12 pMax21 loss1 loss2 r baseD1 baseD2
    int st = status[(size_t) s * nh + h];
    int next = st;
    if (s == traceSlot && mode == 1) {                       // OLF_GPU_HVDC_LOOP_TRACE=<slot>: the fork's HVDC_CHECK twin
        const double* xs = x + (size_t) s * n;
        double ph1 = xs[hvBus1Phi[h]];
        double ph2 = xs[hvBus2Phi[h]];
        printf("HVDC_CHECK s=%d h=%d elem=%d before=%d p1=%.12f p2=%.12f pmax12=%.9f pmax21=%.9f ph1=%.12f ph2=%.12f disabled=%d\n",
               s, h, hvElem[h], st, hvdc::p1(p[0], p[1], p[4], p[5], p[6], ph1, ph2),
               hvdc::p2(p[0], p[1], p[4], p[5], p[6], ph1, ph2), p[2], p[3], ph1, ph2,
               (int) disabled[(size_t) s * nElements + hvElem[h]]);
    }
    if (mode == 0) {                                         // FreezingHvdcACEmulation: unfreeze
        if (st != GpuAcEmulationStatus::FROZEN) return;
        next = GpuAcEmulationStatus::LINEAR_MODE;
    } else {                                                 // AcHvdcAcEmulationLimits
        size_t elemSlot = (size_t) s * nElements + hvElem[h];
        if (st == GpuAcEmulationStatus::LINEAR_MODE && disabled[elemSlot]) return;
        const double* xs = x + (size_t) s * n;
        // The ANGLE is the bus's own BUS_PHI column, whatever row its P balance lives on (tasks #68: reading
        // it at the target row made a relocated converter bus read a DUMMY_P variable as its angle).
        double ph1 = xs[hvBus1Phi[h]];
        double ph2 = xs[hvBus2Phi[h]];
        double pp1 = hvdc::p1(p[0], p[1], p[4], p[5], p[6], ph1, ph2);
        double pp2 = hvdc::p2(p[0], p[1], p[4], p[5], p[6], ph1, ph2);
        if (st == GpuAcEmulationStatus::LINEAR_MODE) {  // OLF checkAcEmulationMode
            if (pp1 >= 0) {
                if (pp1 > p[2]) { next = GpuAcEmulationStatus::SATURATION_MODE_FROM_CS1_TO_CS2; }
            } else if (pp2 > p[3]) {
                next = GpuAcEmulationStatus::SATURATION_MODE_FROM_CS2_TO_CS1;
            }
        } else if (st == GpuAcEmulationStatus::SATURATION_MODE_FROM_CS1_TO_CS2 || st == GpuAcEmulationStatus::SATURATION_MODE_FROM_CS2_TO_CS1) {
            if ((pp1 > 0 && pp1 < p[2]) || (pp2 > 0 && pp2 < p[3])) {
                next = GpuAcEmulationStatus::LINEAR_MODE;
            } else if (pp1 < 0 && st == GpuAcEmulationStatus::SATURATION_MODE_FROM_CS1_TO_CS2) {
                next = GpuAcEmulationStatus::SATURATION_MODE_FROM_CS2_TO_CS1;
            } else if (pp2 < 0 && st == GpuAcEmulationStatus::SATURATION_MODE_FROM_CS2_TO_CS1) {
                next = GpuAcEmulationStatus::SATURATION_MODE_FROM_CS1_TO_CS2;
            }
        }
    }
    if (next == st) return;
    double nd1 = 0.0;
    double nd2 = 0.0;
    size_t elemSlot = (size_t) s * nElements + hvElem[h];
    if (next == GpuAcEmulationStatus::LINEAR_MODE) {
        disabled[elemSlot] = 0;                              // re-enabled: its own equation terms carry the flow
    } else {
        disabled[elemSlot] = 1;
        if (next == GpuAcEmulationStatus::SATURATION_MODE_FROM_CS1_TO_CS2) {
            nd1 = -p[2];
            nd2 = hvdcAbsActivePowerWithLosses(p[2], p[4], p[5], p[6]);
        } else {
            nd1 = hvdcAbsActivePowerWithLosses(p[3], p[5], p[4], p[6]);
            nd2 = -p[3];
        }
    }
    double want1 = nd1 - p[7];                               // composition: base - baseD + d
    double want2 = nd2 - p[8];
    // ATOMIC: two links between the same bus pair (a doubled corridor) run as two threads of one
    // scenario and both add to the SAME converter-row target; a plain read-modify-write loses one
    // of the two unfreeze increments and leaves that link's frozen flow counted twice (measured:
    // 0.126 kV on every batched branch contingency of a network with a doubled 500 MW link).
    // The P TARGET goes to the row that hosts the bus's P balance THIS scenario: the parent coupler's DUMMY_P
    // row under the coupler-tree relocation, or the bus's own PHI row once a contingency has opened that
    // coupler (the dummy row is then an identity and the balance is back home — relocOwn, the same stock
    // placement every other target writer follows). An inactive P row (the slack, ~ encoded) absorbs it, as OLF's does.
    int w1 = hvBus1Row[h];
    int w2 = hvBus2Row[h];
    if (w1 >= 0 && relocOwn != nullptr && relocOwn[w1] >= 0 && rowMode[(size_t) s * n + w1] == 0) w1 = relocOwn[w1];
    if (w2 >= 0 && relocOwn != nullptr && relocOwn[w2] >= 0 && rowMode[(size_t) s * n + w2] == 0) w2 = relocOwn[w2];
    if (w1 >= 0) {
        atomicAdd(&target[(size_t) s * n + w1], want1 - appliedD1[(size_t) s * nh + h]);
    }
    if (w2 >= 0) {
        atomicAdd(&target[(size_t) s * n + w2], want2 - appliedD2[(size_t) s * nh + h]);
    }
    appliedD1[(size_t) s * nh + h] = want1;
    appliedD2[(size_t) s * nh + h] = want2;
    status[(size_t) s * nh + h] = next;
    switched[s] = 1;                                         // this outer loop is UNSTABLE for s
}

// One check of FreezingHvdcACEmulation (loopKind GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION) or AcHvdcAcEmulationLimits (GpuOuterLoopKind::HVDC_AC_EMULATION_LIMITS)
// on every live scenario.
void hvdcStep(BatchSolverContext* ctx, DriverPass& drv, int loopKind) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    // OLF_GPU_HVDC_LOOP_TRACE=<slot>: the status of every link of that scenario before and after this pass,
    // read back on the host (device printf is not relied on), beside the kernel's own per-link print.
    static const char* hvTraceEnv = std::getenv("OLF_GPU_HVDC_LOOP_TRACE");
    static const int hvTraceSlot = hvTraceEnv != nullptr ? std::atoi(hvTraceEnv) : -1;
    std::vector<int> hvBefore;
    if (hvTraceSlot >= 0 && hvTraceSlot < nS && ctx->nhv > 0) {
        hvBefore.resize(ctx->nhv);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        CUDA_CHECK(cudaMemcpy(hvBefore.data(), ctx->dHvStatus.as<int>() + (size_t) hvTraceSlot * ctx->nhv,
                              ctx->nhv * sizeof(int), cudaMemcpyDeviceToHost));
    }
    hvdcAcEmulationLoopBatch<<<grid(ctx->nhv * nS), BLOCK, 0, ctx->stream>>>(
            ctx->nhv, nS, n, ctx->nElements, ctx->hvOffset,
            loopKind == GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION ? 0 : 1,
            ctx->dHvElem.as<int>(), ctx->dHvBus1Row.as<int>(), ctx->dHvBus2Row.as<int>(),
            ctx->dHvBus1Phi.as<int>(), ctx->dHvBus2Phi.as<int>(),
            ctx->dMode.as<int>(), ctx->st->dRelocOwn.p != nullptr ? ctx->st->dRelocOwn.as<int>() : nullptr,
            ctx->dHvIn.as<double>(), ctx->dHvCtgOut.as<char>(),
            ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dDisabled.as<char>(),
            ctx->dHvStatus.as<int>(), ctx->dHvAppliedD1.as<double>(), ctx->dHvAppliedD2.as<double>(),
            ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>(), hvTraceSlot);
    if (!hvBefore.empty()) {
        std::vector<int> hvAfter(ctx->nhv);
        std::vector<double> d1(ctx->nhv), d2(ctx->nhv);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        CUDA_CHECK(cudaMemcpy(hvAfter.data(), ctx->dHvStatus.as<int>() + (size_t) hvTraceSlot * ctx->nhv,
                              ctx->nhv * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d1.data(), ctx->dHvAppliedD1.as<double>() + (size_t) hvTraceSlot * ctx->nhv,
                              ctx->nhv * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d2.data(), ctx->dHvAppliedD2.as<double>() + (size_t) hvTraceSlot * ctx->nhv,
                              ctx->nhv * sizeof(double), cudaMemcpyDeviceToHost));
        for (int h = 0; h < ctx->nhv; h++) {
            if (hvBefore[h] != hvAfter[h]) {
                std::fprintf(stderr, "HVDC_LOOP s=%d outer=%d kind=%d h=%d status %d -> %d appliedD1=%.9f appliedD2=%.9f\n",
                             hvTraceSlot, outer, loopKind, h, hvBefore[h], hvAfter[h], d1[h], d2[h]);
            }
        }
    }
}

}  // namespace olfgpu
