// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See branch_fill.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "closed_branch.cuh"
#include "closed_branch_scatter.h"
#include "open_branch.cuh"
#include "open_branch_scatter.h"
#include "shunt.cuh"
#include "shunt_scatter.h"
#include "hvdc.cuh"
#include "hvdc_scatter.h"
#include "vdl.cuh"
#include "scatter.cuh"
#include "scenario_overrides.cuh"
#include "branch_fill.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

// Batched fill kernels: thread = element x scenario. Skip if the scenario has retired
// (converged) or the element is outaged in that scenario. Values/mismatch are per-scenario.
__global__ void fillClosedBranchBatch(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                      const double* in, const int* ix, const int* dest,
                                      const double* x, double* vals, double* F,
                                      const char* disabled, const char* active,
                                      int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                      const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                                      const int* phaseControlDestination, const int* phaseControlSelector, ReactiveRelocationTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elementCount * nScenarios) return;
    int e = tid % elementCount;
    int s = tid / elementCount;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 8;                            // y g1 g2 b1 b2 ksi r1 a1
    const int* q = ix + e * 12;                              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) {
        p = pOv;
    }
    const double* xs = x + s * n;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    // OLF's own form (AcNetworkVector takes sin and cos of theta1 / theta2 directly). Every closed-branch fill -
    // this one, the F-only fill and the bus-balance fill - goes through the same generated assembler, so their
    // residuals agree bit for bit. The F-only fill used to derive the angles by angle addition from per-branch
    // statics: faster, but another rounding than OLF's.
    double o[CB_NOUT];
    evalClosedBranch(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    if (dynamicRelocation) {
        scatterAddRelocated(CB_NOUT, dest + e * CB_NOUT, alternateDestination + e * CB_NOUT, selectorRow + e * CB_NOUT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n,
                      relocationTablesForScenario(rt, s), e * CB_NOUT);
    } else {
        scatterAdd(CB_NOUT, dest + e * CB_NOUT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
    if (phaseControlDestination != nullptr) {                                 // continuous phase control: emit the controlled
        const int* pd = phaseControlDestination + e * CB_NOUT;                // side's P outputs AGAIN at the a1 row when this
        const int* psl = phaseControlSelector + e * CB_NOUT;                // scenario has the control enabled (row mode 3)
        for (int k = 0; k < CB_NOUT; k++) {
            int d = pd[k];
            if (d == -1 || mode[(size_t) s * n + psl[k]] != GpuRowMode::BRANCH_TARGET_P) {
                continue;
            }
            if (d >= 0) {
                atomicAdd(vals + (size_t) s * nnz + d, o[k]);
            } else {
                atomicAdd(F + (size_t) s * n + (-2 - d), o[k]);
            }
        }
    }
}

// The fused fill's residual half alone, for the reactive-limit / slack Q buffer: evalClosedBranchF evaluates
// p1 q1 p2 q2 from the SAME generated expressions and raw-angle trig as evalClosedBranch (bit-identical), with
// the frozen routing (dynamicRelocation 0, no phase-control re-emission) and no Jacobian at all - neither its 24
// derivatives nor their atomics.
__global__ void fillClosedBranchQbufBatch(int elementCount, int nScenarios, int n, int nElements, int elemOffset,
                                          const double* in, const int* ix, const int* dest,
                                          const double* x, double* F, const char* disabled, const char* active,
                                          int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elementCount * nScenarios) return;
    int e = tid % elementCount;
    int s = tid / elementCount;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 8;                            // y g1 g2 b1 b2 ksi r1 a1
    const int* q = ix + e * 12;
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) {
        p = pOv;
    }
    const double* xs = x + s * n;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    double o[4];                                             // p1 q1 p2 q2: evalClosedBranch's own expressions
    evalClosedBranchF(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0, o + 0, o + 1, o + 2, o + 3);
    const int pos[4] = {0, 7, 14, 21};                       // their output slots (static_assert-pinned below)
    const int* d = dest + e * CB_NOUT;
    for (int k = 0; k < 4; k++) {
        if (d[pos[k]] <= -2) {
            atomicAdd(F + (size_t) s * n + (-2 - d[pos[k]]), o[k]);
        }
    }
}

__global__ void fillOpenBranchBatch(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                    const double* in, const int* ix, const int* dest,
                                    const double* x, double* vals, double* F,
                                    const char* disabled, const char* active,
                                    const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                                    ReactiveRelocationTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elementCount * nScenarios) return;
    int e = tid % elementCount;
    int s = tid / elementCount;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 7;                            // y ksi g1 b1 g2 b2 r1
    const int* q = ix + e * 6;
    const double* xs = x + s * n;
    double v1 = q[4] >= 0 ? xs[q[4]] : 1.0;
    double v2 = q[5] >= 0 ? xs[q[5]] : 1.0;
    double o[OB_NOUT];
    evalOpenBranch(p[0], p[1], p[2], p[3], p[4], p[5], v1, p[6], v2, 0,
                   o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7);
    if (dynamicRelocation) {
        scatterAddRelocated(OB_NOUT, dest + e * OB_NOUT, alternateDestination + e * OB_NOUT, selectorRow + e * OB_NOUT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n,
                      relocationTablesForScenario(rt, s), e * OB_NOUT);
    } else {
        scatterAdd(OB_NOUT, dest + e * OB_NOUT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
}

// ---------------------------------------------------------------------------------
// F-ONLY batch fills: evaluate just the residual values (the generated eval<X>F
// routines) and scatter only the F rows — for iterations that never consume the
// Jacobian (the final residual/convergence fill). The residual positions in
// each family's dest stride are pinned to the generated scatter tables by
// static_asserts, so a codegen output-order change fails the build loudly.
// ---------------------------------------------------------------------------------

static_assert(CB_SCATTER[0].role == 0 && CB_SCATTER[7].role == 0
           && CB_SCATTER[14].role == 0 && CB_SCATTER[21].role == 0,
              "closed-branch residual outputs moved — update fillClosedBranchBatchF");
static_assert(OB_SCATTER[0].role == 0 && OB_SCATTER[2].role == 0
           && OB_SCATTER[4].role == 0 && OB_SCATTER[6].role == 0,
              "open-branch residual outputs moved — update fillOpenBranchBatchF");
static_assert(SH_SCATTER[0].role == 0 && SH_SCATTER[2].role == 0,
              "shunt residual outputs moved — update fillShuntBatchF");
static_assert(HVDC_SCATTER[0].role == 0 && HVDC_SCATTER[3].role == 0,
              "hvdc residual outputs moved — update fillHvdcBatchF");

__global__ void fillClosedBranchBatchF(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                       const double* in, const int* ix, const int* dest,
                                       const double* x, double* F,
                                       const char* disabled, const char* active,
                                       int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                       const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                                       const int* phaseControlDestination, const int* phaseControlSelector, ReactiveRelocationTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elementCount * nScenarios) return;
    int e = tid % elementCount;
    int s = tid / elementCount;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 8;                            // y g1 g2 b1 b2 ksi r1 a1
    const int* q = ix + e * 12;                              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    double pOv[8];
    bool tapOv = findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv);
    if (tapOv) {
        p = pOv;
    }
    const double* xs = x + s * n;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    double o[4];                                             // p1 q1 p2 q2: the fused fill's own expressions
    evalClosedBranchF(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0, o + 0, o + 1, o + 2, o + 3);
    const int pos[4] = {0, 7, 14, 21};
    if (dynamicRelocation) {
        scatterAddFRelocated(4, dest + e * CB_NOUT, alternateDestination + e * CB_NOUT, selectorRow + e * CB_NOUT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, rt, e * CB_NOUT);
    } else {
        scatterAddF(4, dest + e * CB_NOUT, pos, o, F + (size_t) s * n);
    }
    if (phaseControlDestination != nullptr) {                                 // continuous phase control: the P residual at a1
        for (int j = 0; j < 4; j += 2) {                     // outputs 0 (p1) / 2 (p2) at table pos 0 / 14
            int d = phaseControlDestination[e * CB_NOUT + pos[j]];
            if (d < -1 && mode[(size_t) s * n + phaseControlSelector[e * CB_NOUT + pos[j]]] == GpuRowMode::BRANCH_TARGET_P) {
                atomicAdd(F + (size_t) s * n + (-2 - d), o[j]);
            }
        }
    }
}

__global__ void fillOpenBranchBatchF(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                     const double* in, const int* ix, const int* dest,
                                     const double* x, double* F,
                                     const char* disabled, const char* active,
                                     const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                                     ReactiveRelocationTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elementCount * nScenarios) return;
    int e = tid % elementCount;
    int s = tid / elementCount;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 7;                            // y ksi g1 b1 g2 b2 r1
    const int* q = ix + e * 6;
    const double* xs = x + s * n;
    double v1 = q[4] >= 0 ? xs[q[4]] : 1.0;
    double v2 = q[5] >= 0 ? xs[q[5]] : 1.0;
    double o[4];
    evalOpenBranchF(p[0], p[1], p[2], p[3], p[4], p[5], v1, p[6], v2, 0, o + 0, o + 1, o + 2, o + 3);
    const int pos[4] = {0, 2, 4, 6};
    if (dynamicRelocation) {
        scatterAddFRelocated(4, dest + e * OB_NOUT, alternateDestination + e * OB_NOUT, selectorRow + e * OB_NOUT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, rt, e * OB_NOUT);
    } else {
        scatterAddF(4, dest + e * OB_NOUT, pos, o, F + (size_t) s * n);
    }
}

}  // namespace olfgpu
