// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The flow of a controlled branch and its gradient, read by the incremental flow loops (phase, reactive
// power) and the continuous phase control.
#ifndef OLF_GPU_OUTERLOOP_FLOW_OBSERVATION_CUH
#define OLF_GPU_OUTERLOOP_FLOW_OBSERVATION_CUH

#include <cmath>

#include "../equations/generated/closed_branch.cuh"
#include "../equations/generated/closed_branch_i.cuh"
#include "../equations/generated/closed_branch_scatter.h"
#include "../common/codes/gpu_controlled_flow.cuh"
#include "../common/codes/gpu_pi_model_column.cuh"
#include "../common/codes/gpu_two_sides.cuh"

namespace olfgpu {

using namespace powsybl::openloadflow;

// The controlled branch's P or Q flow and its gradient at the current state, through the codegen'd
// updateClosedBranch (the very functions the fill scatters). side is a GpuTwoSides; kind is ACTIVE_POWER or
// REACTIVE_POWER. g[0..5] = d/d(v1, v2, ph1, ph2, a1, r1).
__device__ inline double controlledFlowAndGradient(int side, int kind, const double* p, const double* xs, const int* q, double* g) {
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[GpuPiModelColumn::R1];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[GpuPiModelColumn::A1];
    double y = p[GpuPiModelColumn::Y], g1 = p[GpuPiModelColumn::G1], g2 = p[GpuPiModelColumn::G2], b1 = p[GpuPiModelColumn::B1], b2 = p[GpuPiModelColumn::B2];
    double ksi = p[GpuPiModelColumn::KSI];
    double o[CB_NOUT];
    updateClosedBranch(y, g1, g2, b1, b2, v1, r1, v2, ksi, a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    int blk = (kind == GpuControlledFlow::REACTIVE_POWER ? 7 : 0) + (side == GpuTwoSides::ONE ? 0 : 14);  // p1 0, q1 7, p2 14, q2 21: [val dv1 dv2 dph1 dph2 da1 dr1]
    for (int k = 0; k < 6; k++) g[k] = o[blk + 1 + k];
    return o[blk];
}

// OLF's AbstractClosedBranchAcFlowEquationTerm.ZERO_CURRENT_THRESHOLD.
constexpr double ZERO_CURRENT_THRESHOLD = 1e-6;

// The controlled branch's current magnitude: OLF's ClosedBranchSide{1,2}CurrentMagnitudeEquationTerm.eval(), the
// generated i1 / i2.
__device__ inline double controlledCurrent(int side, const double* p, const double* xs, const int* q) {
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[GpuPiModelColumn::R1];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[GpuPiModelColumn::A1];
    double y = p[GpuPiModelColumn::Y], ksi = p[GpuPiModelColumn::KSI];
    return side == GpuTwoSides::ONE
            ? closed_branch_i::i1(y, ksi, p[GpuPiModelColumn::G1], p[GpuPiModelColumn::B1], v1, ph1, r1, a1, v2, ph2)
            : closed_branch_i::i2(y, ksi, p[GpuPiModelColumn::G2], p[GpuPiModelColumn::B2], v1, ph1, r1, a1, v2, ph2);
}

// The current's derivative with respect to a1: OLF's getI1()/getI2().der(a1Var), read by PhaseControlOuterLoop's
// limiter (NaN at I = 0, as in OLF).
__device__ inline double controlledCurrentDerivativeWithRespectToA1(int side, const double* p, const double* xs, const int* q) {
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[GpuPiModelColumn::R1];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[GpuPiModelColumn::A1];
    double y = p[GpuPiModelColumn::Y], ksi = p[GpuPiModelColumn::KSI];
    return side == GpuTwoSides::ONE
            ? closed_branch_i::di1da1(y, ksi, p[GpuPiModelColumn::G1], p[GpuPiModelColumn::B1], v1, ph1, r1, a1, v2, ph2)
            : closed_branch_i::di2da1(y, ksi, p[GpuPiModelColumn::G2], p[GpuPiModelColumn::B2], v1, ph1, r1, a1, v2, ph2);
}

// OLF's ClosedBranchSide1CurrentMagnitudeEquationTerm.calculateSensi, line for line on the generated methods.
// OLF throws on dr1 != 0 (no current derivative with respect to r1); the device has no dr1 term.
__device__ inline double closedBranchSide1CurrentSensitivity(double y, double ksi, double g1, double b1,
                                                             double v1, double ph1, double r1, double a1, double v2, double ph2,
                                                             double dph1, double dph2, double dv1, double dv2, double da1) {
    using namespace closed_branch_i;
    if (i1(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2) < ZERO_CURRENT_THRESHOLD) {
        // At I=0 the standard formula reI·∂reI/∂u / |I| is 0/0. One-sided limit: ∂|I1|/∂u = hypot(∂reI1/∂u, ∂imI1/∂u).
        using namespace closed_branch_i::ClosedBranchSide1CurrentMagnitudeEquationTerm;
        double dreI1Dph1 = dreI1dph1(y, ksi, g1, b1, v1, ph1, r1);
        double dreI1Dph2 = dreI1dph2(y, ksi, r1, a1, v2, ph2);
        double dreI1Dv1 = dreI1dv1(y, ksi, g1, b1, ph1, r1);
        double dreI1Dv2 = dreI1dv2(y, ksi, r1, a1, ph2);
        double dimI1Dph1 = dimI1dph1(y, ksi, g1, b1, v1, ph1, r1);
        double dimI1Dph2 = dimI1dph2(y, ksi, r1, a1, v2, ph2);
        double dimI1Dv1 = dimI1dv1(y, ksi, g1, b1, ph1, r1);
        double dimI1Dv2 = dimI1dv2(y, ksi, r1, a1, ph2);
        // da1 shifts theta the same way as -dph2 (a1 enters theta with opposite sign)
        double dreI1Da1 = -dreI1Dph2;
        double dimI1Da1 = -dimI1Dph2;
        return hypot(dreI1Dph1 * dph1 + dreI1Dph2 * dph2 + dreI1Dv1 * dv1 + dreI1Dv2 * dv2 + dreI1Da1 * da1,
                     dimI1Dph1 * dph1 + dimI1Dph2 * dph2 + dimI1Dv1 * dv1 + dimI1Dv2 * dv2 + dimI1Da1 * da1);
    }
    return di1dph1(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2) * dph1
            + di1dph2(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2) * dph2
            + di1dv1(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2) * dv1
            + di1dv2(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2) * dv2
            + di1da1(y, ksi, g1, b1, v1, ph1, r1, a1, v2, ph2) * da1;
}

// OLF's ClosedBranchSide2CurrentMagnitudeEquationTerm.calculateSensi, line for line on the generated methods.
__device__ inline double closedBranchSide2CurrentSensitivity(double y, double ksi, double g2, double b2,
                                                             double v1, double ph1, double r1, double a1, double v2, double ph2,
                                                             double dph1, double dph2, double dv1, double dv2, double da1) {
    using namespace closed_branch_i;
    if (i2(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2) < ZERO_CURRENT_THRESHOLD) {
        // At I=0 the standard formula reI·∂reI/∂u / |I| is 0/0. One-sided limit: ∂|I2|/∂u = hypot(∂reI2/∂u, ∂imI2/∂u).
        using namespace closed_branch_i::ClosedBranchSide2CurrentMagnitudeEquationTerm;
        double theta = ClosedBranchSide2CurrentMagnitudeEquationTerm::theta(ksi, ph1, a1);
        double dreI2Dph1 = dreI2dph1(y, ksi, v1, ph1, r1, a1);
        double dreI2Dph2 = dreI2dph2(y, ksi, g2, b2, v2, ph2);
        double dreI2Dv1 = dreI2dv1(y, r1, theta);
        double dreI2Dv2 = dreI2dv2(y, ksi, g2, b2, ph2);
        double dimI2Dph1 = dimI2dph1(y, v1, r1, theta);
        double dimI2Dph2 = dimI2dph2(y, ksi, g2, b2, v2, ph2);
        double dimI2Dv1 = dimI2dv1(y, ksi, ph1, r1, a1);
        double dimI2Dv2 = dimI2dv2(y, ksi, g2, b2, ph2);
        // da1 shifts theta the same way as dph1 (a1 enters theta with same sign as ph1)
        double dreI2Da1 = dreI2Dph1;
        double dimI2Da1 = dimI2Dph1;
        return hypot(dreI2Dph1 * dph1 + dreI2Dph2 * dph2 + dreI2Dv1 * dv1 + dreI2Dv2 * dv2 + dreI2Da1 * da1,
                     dimI2Dph1 * dph1 + dimI2Dph2 * dph2 + dimI2Dv1 * dv1 + dimI2Dv2 * dv2 + dimI2Da1 * da1);
    }
    return di2dph1(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2) * dph1
            + di2dph2(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2) * dph2
            + di2dv1(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2) * dv1
            + di2dv2(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2) * dv2
            + di2da1(y, ksi, g2, b2, v1, ph1, r1, a1, v2, ph2) * da1;
}

// The controlled current's sensitivity to the unit-RHS solution dx: OLF's calculateSensi(dx, column) on
// getI1()/getI2() (a non-variable a1 column contributes nothing).
__device__ inline double controlledCurrentSensitivity(int side, const double* p, const double* xs, const int* q, const double* dxs) {
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[GpuPiModelColumn::R1];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[GpuPiModelColumn::A1];
    double y = p[GpuPiModelColumn::Y], ksi = p[GpuPiModelColumn::KSI];
    double dph1 = dxs[q[4]], dph2 = dxs[q[6]], dv1 = dxs[q[5]], dv2 = dxs[q[7]];
    double da1 = q[8] >= 0 ? dxs[q[8]] : 0.0;
    return side == GpuTwoSides::ONE
            ? closedBranchSide1CurrentSensitivity(y, ksi, p[GpuPiModelColumn::G1], p[GpuPiModelColumn::B1], v1, ph1, r1, a1, v2, ph2, dph1, dph2, dv1, dv2, da1)
            : closedBranchSide2CurrentSensitivity(y, ksi, p[GpuPiModelColumn::G2], p[GpuPiModelColumn::B2], v1, ph1, r1, a1, v2, ph2, dph1, dph2, dv1, dv2, da1);
}

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_FLOW_OBSERVATION_CUH
