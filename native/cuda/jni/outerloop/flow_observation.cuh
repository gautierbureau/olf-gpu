// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The flow of a controlled branch and its gradient, read by the incremental flow loops (phase, reactive
// power) and the continuous phase control.
#ifndef OLF_GPU_OUTERLOOP_FLOW_OBSERVATION_CUH
#define OLF_GPU_OUTERLOOP_FLOW_OBSERVATION_CUH

#include <cmath>

#include "closed_branch.cuh"
#include "closed_branch_scatter.h"
#include "../common/codes/gpu_controlled_flow.cuh"
#include "../common/codes/gpu_pi_model_column.cuh"
#include "../common/codes/gpu_two_sides.cuh"

namespace olfgpu {

using namespace powsybl::openloadflow;

// The controlled branch's flow and its gradient at the current state: P/Q through the codegen'd
// evalClosedBranch (the very functions the fill scatters), I through OLF's own reI/imI decomposition
// (ClosedBranchSide{1,2}CurrentMagnitudeEquationTerm, incl. its one-sided rule at I = 0 and its
// dr1-less sensitivity). side is a GpuTwoSides. g[0..5] = d/d(v1, v2, ph1, ph2, a1, r1).
__device__ inline double controlledFlowAndGradient(int side, int kind, const double* p, const double* xs, const int* q, double* g) {
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[GpuPiModelColumn::R1];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[GpuPiModelColumn::A1];
    double y = p[GpuPiModelColumn::Y], g1 = p[GpuPiModelColumn::G1], g2 = p[GpuPiModelColumn::G2], b1 = p[GpuPiModelColumn::B1], b2 = p[GpuPiModelColumn::B2];
    double ksi = p[GpuPiModelColumn::KSI];
    if (kind != GpuControlledFlow::CURRENT) {  // P or Q: the assembled outputs
        double o[CB_NOUT];
        evalClosedBranch(y, g1, g2, b1, b2, v1, r1, v2, ksi, a1, ph1, ph2, 0,
                         o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                         o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                         o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
        int blk = (kind == GpuControlledFlow::REACTIVE_POWER ? 7 : 0) + (side == GpuTwoSides::ONE ? 0 : 14);  // p1 0, q1 7, p2 14, q2 21: [val dv1 dv2 dph1 dph2 da1 dr1]
        for (int k = 0; k < 6; k++) g[k] = o[blk + 1 + k];
        return o[blk];
    }
    const double R2 = 1.0;
    if (side == GpuTwoSides::ONE) {
        double theta = ksi - a1 + ph2;                       // OLF: ksi - a1 + A2 + ph2, A2 = 0
        double ire = g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi);
        double iim = g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi);
        double re = r1 * (r1 * v1 * ire - y * R2 * v2 * sin(theta));
        double im = r1 * (r1 * v1 * iim + y * R2 * v2 * cos(theta));
        double dre[6] = { r1 * r1 * ire, r1 * (-y * R2 * sin(theta)),
                          r1 * r1 * v1 * (-g1 * sin(ph1) - b1 * cos(ph1) + y * cos(ph1 + ksi)),
                          r1 * (-y * R2 * v2 * cos(theta)), 0.0, 0.0 };
        double dim[6] = { r1 * r1 * iim, r1 * (y * R2 * cos(theta)),
                          r1 * r1 * v1 * ire, r1 * (-y * R2 * v2 * sin(theta)), 0.0, 0.0 };
        dre[4] = -dre[3];                                    // a1 enters theta with the opposite sign of ph2
        dim[4] = -dim[3];
        double i = hypot(re, im);
        if (i == 0.0) {                                      // one-sided limit: the gradient's MAGNITUDE — the
            for (int k = 0; k < 6; k++) g[k] = 0.0;          // dot with dx is done by the caller as hypot(dre·dx, dim·dx)
            g[0] = 1e300;                                    // flag (see incrementalFlowControlGatherSensitivityBatch)
            for (int k = 0; k < 6; k++) { g[6 + k] = dre[k]; g[12 + k] = dim[k]; }
            return 0.0;
        }
        for (int k = 0; k < 6; k++) g[k] = (re * dre[k] + im * dim[k]) / i;
        return i;
    }
    double theta = ksi + a1 + ph1;                           // OLF: ksi + a1 - A2 + ph1
    double ire = g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi);
    double iim = g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi);
    double re = R2 * (R2 * v2 * ire - y * r1 * v1 * sin(theta));
    double im = R2 * (R2 * v2 * iim + y * r1 * v1 * cos(theta));
    double dre[6] = { R2 * (-y * r1 * sin(theta)), R2 * R2 * ire,
                      R2 * (-y * r1 * v1 * cos(theta)),
                      R2 * R2 * v2 * (-g2 * sin(ph2) - b2 * cos(ph2) + y * cos(ph2 + ksi)), 0.0, 0.0 };
    double dim[6] = { R2 * (y * r1 * cos(theta)), R2 * R2 * iim,
                      R2 * (-y * r1 * v1 * sin(theta)), R2 * R2 * v2 * ire, 0.0, 0.0 };
    dre[4] = dre[2];                                         // a1 enters theta like ph1
    dim[4] = dim[2];
    double i = hypot(re, im);
    if (i == 0.0) {
        for (int k = 0; k < 6; k++) g[k] = 0.0;
        g[0] = 1e300;
        for (int k = 0; k < 6; k++) { g[6 + k] = dre[k]; g[12 + k] = dim[k]; }
        return 0.0;
    }
    for (int k = 0; k < 6; k++) g[k] = (re * dre[k] + im * dim[k]) / i;
    return i;
}

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_FLOW_OBSERVATION_CUH
