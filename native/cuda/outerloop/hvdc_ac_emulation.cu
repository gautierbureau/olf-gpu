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
#include "../common/codes/gpu_row_mode.cuh"
#include "../equations/generated/hvdc.cuh"

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
__global__ void hvdcAcEmulationLoopBatch(int nh, int nScenarios, int n, int nElements, int hvdcAcEmulationOffset,
                                         int mode,
                                         const int* hvdcElement, const int* hvdcBus1PRow, const int* hvdcBus2PRow,
                                         const int* hvdcBus1PhiColumn, const int* hvdcBus2PhiColumn,
                                         const int* rowMode, const int* relocatedChildOwnRow,
                                         const double* hvdcAcEmulationParameters, const char* hvdcContingencyOutage,
                                         const double* x, double* target, char* disabled,
                                         int* status, double* appliedDelta1, double* appliedDelta2,
                                         const char* outerLoopCheckActive, char* switched) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nh * nScenarios) return;
    int h = tid % nh;
    int s = tid / nh;
    if (!outerLoopCheckActive[s] || hvdcContingencyOutage[(size_t) s * nh + h]) return;    // outaged: the contingency owns that bit
    const double* p = hvdcAcEmulationParameters + (size_t) h * 9;                 // p0 k pMax12 pMax21 loss1 loss2 r baseD1 baseD2
    int acEmulationStatus = status[(size_t) s * nh + h];
    int next = acEmulationStatus;
    if (mode == 0) {                                         // FreezingHvdcACEmulation: unfreeze
        if (acEmulationStatus != GpuAcEmulationStatus::FROZEN) return;
        next = GpuAcEmulationStatus::LINEAR_MODE;
    } else {                                                 // AcHvdcAcEmulationLimits
        size_t elemSlot = (size_t) s * nElements + hvdcElement[h];
        if (acEmulationStatus == GpuAcEmulationStatus::LINEAR_MODE && disabled[elemSlot]) return;
        const double* xs = x + (size_t) s * n;
        // The ANGLE is the bus's own BUS_PHI column, whatever row its P balance lives on (tasks #68: reading
        // it at the target row made a relocated converter bus read a DUMMY_P variable as its angle).
        double ph1 = xs[hvdcBus1PhiColumn[h]];
        double ph2 = xs[hvdcBus2PhiColumn[h]];
        double pp1 = hvdc::p1(p[0], p[1], p[4], p[5], p[6], ph1, ph2);
        double pp2 = hvdc::p2(p[0], p[1], p[4], p[5], p[6], ph1, ph2);
        if (acEmulationStatus == GpuAcEmulationStatus::LINEAR_MODE) {  // OLF checkAcEmulationMode
            if (pp1 >= 0) {
                if (pp1 > p[2]) { next = GpuAcEmulationStatus::SATURATION_MODE_FROM_CS1_TO_CS2; }
            } else if (pp2 > p[3]) {
                next = GpuAcEmulationStatus::SATURATION_MODE_FROM_CS2_TO_CS1;
            }
        } else if (acEmulationStatus == GpuAcEmulationStatus::SATURATION_MODE_FROM_CS1_TO_CS2 || acEmulationStatus == GpuAcEmulationStatus::SATURATION_MODE_FROM_CS2_TO_CS1) {
            if ((pp1 > 0 && pp1 < p[2]) || (pp2 > 0 && pp2 < p[3])) {
                next = GpuAcEmulationStatus::LINEAR_MODE;
            } else if (pp1 < 0 && acEmulationStatus == GpuAcEmulationStatus::SATURATION_MODE_FROM_CS1_TO_CS2) {
                next = GpuAcEmulationStatus::SATURATION_MODE_FROM_CS2_TO_CS1;
            } else if (pp2 < 0 && acEmulationStatus == GpuAcEmulationStatus::SATURATION_MODE_FROM_CS2_TO_CS1) {
                next = GpuAcEmulationStatus::SATURATION_MODE_FROM_CS1_TO_CS2;
            }
        }
    }
    if (next == acEmulationStatus) return;
    double nd1 = 0.0;
    double nd2 = 0.0;
    size_t elemSlot = (size_t) s * nElements + hvdcElement[h];
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
    // coupler (the dummy row is then an identity and the balance is back home — relocatedChildOwnRow, the same stock
    // placement every other target writer follows). An inactive P row (the slack, ~ encoded) absorbs it, as OLF's does.
    int w1 = hvdcBus1PRow[h];
    int w2 = hvdcBus2PRow[h];
    if (w1 >= 0 && relocatedChildOwnRow != nullptr && relocatedChildOwnRow[w1] >= 0 && rowMode[(size_t) s * n + w1] == GpuRowMode::FIXED_VARIABLE) w1 = relocatedChildOwnRow[w1];
    if (w2 >= 0 && relocatedChildOwnRow != nullptr && relocatedChildOwnRow[w2] >= 0 && rowMode[(size_t) s * n + w2] == GpuRowMode::FIXED_VARIABLE) w2 = relocatedChildOwnRow[w2];
    if (w1 >= 0) {
        atomicAdd(&target[(size_t) s * n + w1], want1 - appliedDelta1[(size_t) s * nh + h]);
    }
    if (w2 >= 0) {
        atomicAdd(&target[(size_t) s * n + w2], want2 - appliedDelta2[(size_t) s * nh + h]);
    }
    appliedDelta1[(size_t) s * nh + h] = want1;
    appliedDelta2[(size_t) s * nh + h] = want2;
    status[(size_t) s * nh + h] = next;
    switched[s] = 1;                                         // this outer loop is UNSTABLE for s
}

// One check of FreezingHvdcACEmulation (loopKind GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION) or AcHvdcAcEmulationLimits (GpuOuterLoopKind::HVDC_AC_EMULATION_LIMITS)
// on every live scenario.
void hvdcStep(BatchSolverContext* ctx, DriverPass& drv, int loopKind) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    hvdcAcEmulationLoopBatch<<<grid(ctx->nHvdcAcEmulation * nS), BLOCK, 0, ctx->stream>>>(
            ctx->nHvdcAcEmulation, nS, n, ctx->nElements, ctx->hvdcAcEmulationOffset,
            loopKind == GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION ? 0 : 1,
            ctx->dHvdcElement.as<int>(), ctx->dHvdcBus1PRow.as<int>(), ctx->dHvdcBus2PRow.as<int>(),
            ctx->dHvdcBus1PhiColumn.as<int>(), ctx->dHvdcBus2PhiColumn.as<int>(),
            ctx->dMode.as<int>(), ctx->structure->dRelocatedChildOwnRow.p != nullptr ? ctx->structure->dRelocatedChildOwnRow.as<int>() : nullptr,
            ctx->dHvdcAcEmulationParameters.as<double>(), ctx->dHvdcContingencyOutage.as<char>(),
            ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dDisabled.as<char>(),
            ctx->dHvdcAcEmulationStatus.as<int>(), ctx->dHvdcAppliedDelta1.as<double>(), ctx->dHvdcAppliedDelta2.as<double>(),
            ctx->dOuterLoopCheckActive.as<char>(), ctx->dOuterLoopUnstable.as<char>());
}

}  // namespace olfgpu
