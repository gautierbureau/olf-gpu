// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See area_interchange_control.cuh.
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_area_boundary_kind.cuh"
#include "../common/codes/gpu_distributed_slack_factor_mode.cuh"
#include "../common/codes/gpu_two_sides.cuh"
#include "../equations/generated/closed_branch.cuh"
#include "../equations/generated/open_branch.cuh"
#include "../equations/scenario_overrides.cuh"
#include "../solver/scenario_install.cuh"
#include "../solver/scenario_override_tables.cuh"
#include "distributed_slack.cuh"
#include "slack_participants.cuh"
#include "area_interchange_control.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

namespace {

// Per (scenario, result slot) working values of one check: slot a < nAreas is area a, slot nAreas the buses without area.
constexpr int SLOT_INTERCHANGE = 0;         // LfArea.getInterchange(): Σ of its boundaries' P
constexpr int SLOT_FACTOR = 1;              // its slack distribution factor in distributeRemainingSlackMismatch
constexpr int SLOT_ORDER = 2;               // the area at this rank of distributeOnAreas' sort (by factor, descending)
constexpr int SLOT_PRESENT = 3;             // the check has a result for this slot
constexpr int SLOT_TYPE_SLACK = 4;          // 1: ActivePowerDistributionType.SLACK, 0: AREA_INTERCHANGE
constexpr int SLOT_DISTRIBUTED = 5;         // AreaActivePowerDistributionResult.distributedMismatch
constexpr int SLOT_REMAINING = 6;           // .remainingMismatch
constexpr int SLOT_MOVED = 7;               // .movedBuses
constexpr int SLOT_WIDTH = 8;

// Java's Math.toDegrees / Math.toRadians constants.
constexpr double RADIANS_TO_DEGREES = 57.29577951308232;
constexpr double DEGREES_TO_RADIANS = 0.017453292519943295;

struct Tables {
    int n, nS, nElements;
    int slackRowBase;
    const int* slackRowOverride;
    double slackTargetP;
    const double* slackBusAnchorShift;
    int nAreas;
    const double* interchangeTarget, *slackDistributionFactor;
    const int* boundaryStart, *boundaryKind, *boundaryElement, *boundaryMaskSlot, *boundarySide;
    const double* boundaryTieLineHalf;
    const double* closedBranchParameters;
    const int* closedBranchIndices;
    const double* openBranchParameters;
    const int* openBranchIndices;
    ClosedBranchParameterOverrides closedBranchParameterOverrides;
    int nBuses;
    const int* busPhiRow, *busRowOverride, *busArea, *busGeneratorStart, *busGeneratorEnd;
    const double* busBaseTargetP, *busAnchorShift;
    const char* islanded;
    int nGenerators;
    const int* generatorBus, *generatorArea;
    const double* generatorFactor, *generatorMaxDelta, *generatorMinDelta, *generatorMarginUp, *generatorMarginDown;
    int factorMode;
    SlotEntries lostGenerators;
    SlotEntries removedGenerators;
    double slackBusPMaxMismatch, areaInterchangePMaxMismatch, residueEps;
    const double* x, *qBuf;
    const char* disabled, *outerLoopCheckActive;
    double* target, *move, *workingFactor, *previousMove, *slots, *distributedActivePower;
    char* removed, *participant, *unstable, *failure, *moveValid;
};

struct DistributionResult {
    int iteration;
    double remainingMismatch;
    bool movedBuses;
};

// Apache Commons Math's Complex, as powsybl's SV and LinkData compute a tie line's boundary flow: the same operations in the
// same order, reciprocal and divide scaling by the larger component. Their NaN and infinite branches are never reached for
// a half with a non-zero impedance and finite flows.
struct Complex {
    double re, im;
};

__device__ inline Complex add(Complex a, Complex b) { return {a.re + b.re, a.im + b.im}; }
__device__ inline Complex subtract(Complex a, Complex b) { return {a.re - b.re, a.im - b.im}; }
__device__ inline Complex negate(Complex a) { return {-a.re, -a.im}; }
__device__ inline Complex conjugate(Complex a) { return {a.re, -a.im}; }
__device__ inline Complex multiply(Complex a, Complex b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }

__device__ inline Complex divide(Complex a, Complex b) {
    double c = b.re;
    double d = b.im;
    if (fabs(c) < fabs(d)) {
        double q = c / d;
        double denominator = c * q + d;
        return {(a.re * q + a.im) / denominator, (a.im * q - a.re) / denominator};
    }
    double q = d / c;
    double denominator = d * q + c;
    return {(a.im * q + a.re) / denominator, (a.im - a.re * q) / denominator};
}

__device__ inline Complex reciprocal(Complex a) {
    if (fabs(a.re) < fabs(a.im)) {
        double q = a.re / a.im;
        double scale = 1. / (a.re * q + a.im);
        return {scale * q, -scale};
    }
    double q = a.im / a.re;
    double scale = 1. / (a.im * q + a.re);
    return {scale, -scale * q};
}

__device__ inline Complex polar2Complex(double r, double theta) { return {r * cos(theta), r * sin(theta)}; }

// SV(p, q, u, a, side).otherSideP(boundaryLine, false) / SB: the active power at the boundary point of a tie line half, from
// the flow, voltage and angle at its network side (MW, MVar, kV, degrees) and the half's r, x, g, b (SI). The half is
// LinkData.calculateBranchAdmittance(r, x, 1, -0, 1, 0, g + jb, 0); side ONE solves for its end 2, side TWO for its end 1.
__device__ double tieLineBoundaryP(double pMw, double qMvar, double uKv, double aDeg, const double* half, int side, double sb) {
    double r = half[0];
    double xHalf = half[1];
    Complex a1 = polar2Complex(1.0, -0.0);
    Complex a2 = polar2Complex(1.0, 0.0);
    Complex ytr = r == 0.0 && xHalf == 0.0 ? Complex{0.0, 0.0} : reciprocal(Complex{r, xHalf});
    Complex ysh1 = {half[2], half[3]};
    Complex ysh2 = {0.0, 0.0};
    Complex y11 = divide(add(ytr, ysh1), multiply(conjugate(a1), a1));
    Complex y12 = divide(negate(ytr), multiply(conjugate(a1), a2));
    Complex y21 = divide(negate(ytr), multiply(conjugate(a2), a1));
    Complex y22 = divide(add(ytr, ysh2), multiply(conjugate(a2), a2));
    Complex v = polar2Complex(uKv, aDeg * DEGREES_TO_RADIANS);
    Complex s = {pMw, qMvar};
    Complex flow;
    if (side == GpuTwoSides::ONE) {
        Complex v2 = divide(subtract(divide(conjugate(s), conjugate(v)), multiply(y11, v)), y12);
        flow = conjugate(multiply(add(multiply(y21, v), multiply(y22, v2)), conjugate(v2)));
    } else {
        Complex v1 = divide(subtract(divide(conjugate(s), conjugate(v)), multiply(y22, v)), y21);
        flow = conjugate(multiply(add(multiply(y11, v1), multiply(y12, v)), conjugate(v1)));
    }
    return flow.re / sb;
}

// LfAreaImpl.BoundaryImpl.getP on this scenario's state: 0 for a disabled branch; a branch connected at both sides gives its
// closed p1 or p2 (OLF's getP1 / getP2), one connected at one side the open p of that side and 0 on the other, a tie line the
// flow at the boundary point of its half on the boundary's side.
__device__ double boundaryP(const Tables& t, int s, int b) {
    if (t.disabled[(size_t) s * t.nElements + t.boundaryMaskSlot[b]]) {
        return 0.0;
    }
    const double* xs = t.x + (size_t) s * t.n;
    int e = t.boundaryElement[b];
    int side = t.boundarySide[b];
    if (t.boundaryKind[b] == GpuAreaBoundaryKind::OPEN_BRANCH) {
        const double* p = t.openBranchParameters + (size_t) e * 7;      // y ksi g1 b1 g2 b2 r1
        const int* q = t.openBranchIndices + (size_t) e * 6;            // P1 Q1 P2 Q2 v1col v2col
        if (side == GpuTwoSides::ONE ? q[4] < 0 : q[5] < 0) {
            return 0.0;                                                  // OLF: EvaluableConstants.ZERO
        }
        double v1 = q[4] >= 0 ? xs[q[4]] : 1.0;
        double v2 = q[5] >= 0 ? xs[q[5]] : 1.0;
        double p2;
        double q2;
        double p1;
        double q1;
        updateOpenBranchWithoutDerivatives(p[0], p[1], p[2], p[3], p[4], p[5], v1, p[6], v2, 0, &p2, &q2, &p1, &q1);
        return side == GpuTwoSides::ONE ? p1 : p2;
    }
    const double* p = t.closedBranchParameters + (size_t) e * 8;        // y g1 g2 b1 b2 ksi r1 a1
    const int* q = t.closedBranchIndices + (size_t) e * 12;              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    double pOverride[8];
    if (findTapOverride(t.closedBranchParameterOverrides, s, e, pOverride)) {
        p = pOverride;
    }
    double v1 = xs[q[5]];
    double v2 = xs[q[7]];
    double ph1 = xs[q[4]];
    double ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    double p1;
    double q1;
    double p2;
    double q2;
    updateClosedBranchWithoutDerivatives(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0, &p1, &q1, &p2, &q2);
    if (t.boundaryKind[b] == GpuAreaBoundaryKind::CLOSED_BRANCH) {
        return side == GpuTwoSides::ONE ? p1 : p2;
    }
    // LfAreaImpl.BoundaryImpl: new SV(p * SB, q * SB, v * nominalV, toDegrees(angle), side).otherSideP(half, false) / SB
    const double* half = t.boundaryTieLineHalf + (size_t) b * 6;        // r x g b nominalV SB
    double sb = half[5];
    return side == GpuTwoSides::ONE
            ? tieLineBoundaryP(p1 * sb, q1 * sb, v1 * half[4], ph1 * RADIANS_TO_DEGREES, half, side, sb)
            : tieLineBoundaryP(p2 * sb, q2 * sb, v2 * half[4], ph2 * RADIANS_TO_DEGREES, half, side, sb);
}

// Java's Math.signum.
__device__ inline double signum(double v) {
    return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : v);
}

// Java's Double.compare: NaN above everything, -0.0 below 0.0.
__device__ inline int javaDoubleCompare(double a, double b) {
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    long long ab = isnan(a) ? 0x7ff8000000000000LL : __double_as_longlong(a);
    long long bb = isnan(b) ? 0x7ff8000000000000LL : __double_as_longlong(b);
    return ab == bb ? 0 : (ab < bb ? -1 : 1);
}

__device__ inline bool lessThanSlackBusMaxMismatch(const Tables& t, double mismatch) {
    return fabs(mismatch) <= t.slackBusPMaxMismatch || fabs(mismatch) <= t.residueEps;
}

__device__ inline bool lessThanInterchangeMaxMismatch(const Tables& t, double mismatch) {
    return fabs(mismatch) <= t.areaInterchangePMaxMismatch || fabs(mismatch) <= t.residueEps;
}

// A generator of this distribution set: an area's own (area >= 0) or a bus without area's (area -1), still a participant in
// this scenario.
__device__ inline bool inSet(const Tables& t, int s, int g, int area) {
    return t.generatorArea[g] == area && t.participant[(size_t) s * t.nGenerators + g];
}

// OLF's ActivePowerDistribution.run(null, buses, activePowerMismatch) with a GenerationActivePowerDistributionStep, over the
// participating generators of one set of buses: step.resetToInitialState puts every one back to its initial target and adds
// what that undid to the mismatch, then the water-fill normalizes the factors over the generators still in the list at each
// iteration, moves each by its share and removes one that strictly overshoots its limit. The moves are kept from the initial
// targets (dDistributedSlackGeneratorMove), and each bus row of the set is written back as its anchor plus its generators' moves.
__device__ DistributionResult runActivePowerDistribution(const Tables& t, int s, int area, double activePowerMismatch) {
    size_t gBase = (size_t) s * t.nGenerators;
    double previousMismatch = 0.0;
    for (int g = 0; g < t.nGenerators; g++) {
        if (!inSet(t, s, g, area)) {
            continue;
        }
        double m = t.move[gBase + g];
        t.previousMove[gBase + g] = m;
        previousMismatch += m;
        t.move[gBase + g] = 0.0;
    }
    double remainingMismatch = activePowerMismatch + previousMismatch;
    // getParticipatingElements: the generators with a non-zero factor, REMAINING_MARGIN's in the mismatch's direction
    bool up = remainingMismatch > 0.;
    int elementCount = 0;
    for (int g = 0; g < t.nGenerators; g++) {
        if (!inSet(t, s, g, area)) {
            continue;
        }
        double factor = t.factorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN
                ? fmax(0.0, up ? t.generatorMarginUp[g] : t.generatorMarginDown[g])
                : t.generatorFactor[g];
        t.workingFactor[gBase + g] = factor;
        t.removed[gBase + g] = factor != 0 ? 0 : 1;
        elementCount += factor != 0 ? 1 : 0;
    }
    int iteration = 0;
    while (elementCount > 0 && fabs(remainingMismatch) > t.residueEps) {
        double factorNorm = 0.0;
        for (int g = 0; g < t.nGenerators; g++) {
            if (inSet(t, s, g, area) && !t.removed[gBase + g]) {
                factorNorm += t.workingFactor[gBase + g];
            }
        }
        if (!(factorNorm > 0.0)) {
            break;
        }
        double done = 0.0;
        for (int g = 0; g < t.nGenerators; g++) {
            if (!inSet(t, s, g, area) || t.removed[gBase + g]) {
                continue;
            }
            double factor = t.workingFactor[gBase + g] / factorNorm;            // normalizeParticipationFactors
            t.workingFactor[gBase + g] = factor;
            double move = t.move[gBase + g];
            double newMove = move + remainingMismatch * factor;
            if (remainingMismatch > 0 && newMove > t.generatorMaxDelta[g]) {
                newMove = t.generatorMaxDelta[g];
                t.removed[gBase + g] = 1;
                elementCount--;
            } else if (remainingMismatch < 0 && newMove < t.generatorMinDelta[g]) {
                newMove = t.generatorMinDelta[g];
                t.removed[gBase + g] = 1;
                elementCount--;
            }
            if (newMove != move) {
                t.move[gBase + g] = newMove;
                done += newMove - move;
            }
        }
        remainingMismatch -= done;
        iteration++;
    }
    for (int i = 0; i < t.nBuses; i++) {
        if (t.busArea[i] != area || (t.islanded != nullptr && t.islanded[(size_t) s * t.nBuses + i])) {
            continue;
        }
        double sum = 0.0;
        for (int g = t.busGeneratorStart[i]; g < t.busGeneratorEnd[i]; g++) {
            sum += t.move[gBase + g];
        }
        t.target[(size_t) s * t.n + distributedSlackBusRowOf(t.busPhiRow, t.busRowOverride, s, t.nBuses, i)] =
                t.busBaseTargetP[i] + (t.busAnchorShift != nullptr ? t.busAnchorShift[(size_t) s * t.nBuses + i] : 0.0) + sum;
    }
    // PreviousStateInfo.moved: the sum of the generators' changes, against P_RESIDUE_EPS with OLF's 0.9 for rounding
    double changed = 0.0;
    for (int g = 0; g < t.nGenerators; g++) {
        if (inSet(t, s, g, area)) {
            changed += fabs(t.move[gBase + g] - t.previousMove[gBase + g]);
        }
    }
    return {iteration, remainingMismatch, changed > t.residueEps * 0.9};
}

// AbstractAreaInterchangeControlOuterLoop.distributeOnAreas: the areas by factor, highest first, each running the distribution
// on its share of the mismatch, or on 1.01 * P_RESIDUE_EPS when its share is below that; an area left with a remaining
// mismatch drops out (factor 0). An area this round does not reach keeps its earlier result with no remaining mismatch.
__device__ inline int areaAtRank(const double* slots, int rank) {
    return (int) slots[(size_t) rank * SLOT_WIDTH + SLOT_ORDER];
}

__device__ double distributeOnAreas(const Tables& t, int s, double* slots, double mismatch) {
    int nAreas = t.nAreas;
    for (int a = 0; a < nAreas; a++) {                                       // a stable insertion sort, factor descending
        double factor = slots[(size_t) a * SLOT_WIDTH + SLOT_FACTOR];
        int rank = a;
        while (rank > 0 && javaDoubleCompare(slots[(size_t) areaAtRank(slots, rank - 1) * SLOT_WIDTH + SLOT_FACTOR], factor) < 0) {
            slots[(size_t) rank * SLOT_WIDTH + SLOT_ORDER] = areaAtRank(slots, rank - 1);
            rank--;
        }
        slots[(size_t) rank * SLOT_WIDTH + SLOT_ORDER] = a;
    }
    double remainingMismatch = mismatch;
    int rank = 0;
    for (; rank < nAreas && fabs(remainingMismatch) > t.residueEps; rank++) {
        double* slot = slots + (size_t) areaAtRank(slots, rank) * SLOT_WIDTH;
        double factor = slot[SLOT_FACTOR];
        double areaActivePowerMismatch = fabs(mismatch * factor) > t.residueEps
                ? mismatch * factor
                : signum(mismatch) * 1.01 * t.residueEps;
        DistributionResult r = runActivePowerDistribution(t, s, areaAtRank(slots, rank), areaActivePowerMismatch);
        if (fabs(r.remainingMismatch) > t.residueEps) {
            slot[SLOT_FACTOR] = 0.;
        }
        if (slot[SLOT_PRESENT] == 0.0) {
            slot[SLOT_PRESENT] = 1.0;
            slot[SLOT_TYPE_SLACK] = 1.0;
            slot[SLOT_DISTRIBUTED] = 0.0;
            slot[SLOT_MOVED] = 0.0;
        }
        slot[SLOT_DISTRIBUTED] = slot[SLOT_DISTRIBUTED] + areaActivePowerMismatch - r.remainingMismatch;
        slot[SLOT_MOVED] = slot[SLOT_MOVED] != 0.0 || r.movedBuses ? 1.0 : 0.0;
        slot[SLOT_REMAINING] = r.remainingMismatch;
        remainingMismatch = remainingMismatch - areaActivePowerMismatch + r.remainingMismatch;
    }
    for (; rank < nAreas; rank++) {
        double* slot = slots + (size_t) areaAtRank(slots, rank) * SLOT_WIDTH;
        if (slot[SLOT_PRESENT] != 0.0) {
            slot[SLOT_REMAINING] = 0.;
        }
    }
    return remainingMismatch;
}

// normalizeSlackParticipationFactors: each factor over their sum (0 / 0 is NaN, which ends the distribution as in OLF).
__device__ void normalizeSlackParticipationFactors(const Tables& t, double* slots) {
    double sum = 0.0;
    for (int a = 0; a < t.nAreas; a++) {
        sum += slots[(size_t) a * SLOT_WIDTH + SLOT_FACTOR];
    }
    for (int a = 0; a < t.nAreas; a++) {
        slots[(size_t) a * SLOT_WIDTH + SLOT_FACTOR] /= sum;
    }
}

// AbstractAreaInterchangeControlOuterLoop.check on one scenario.
__global__ void areaInterchangeControlBatch(Tables t) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= t.nS || !t.outerLoopCheckActive[s]) {
        return;
    }
    size_t gBase = (size_t) s * t.nGenerators;
    // the generators a contingency removes and those on a bus left in another component take no part
    for (int g = 0; g < t.nGenerators; g++) {
        int i = t.generatorBus[g];
        bool participates = !(i >= 0 && t.islanded != nullptr && t.islanded[(size_t) s * t.nBuses + i]);
        participates = participates && slotFirstEntryOf(t.lostGenerators, s, g) < 0
                && slotFirstEntryOf(t.removedGenerators, s, g) < 0;
        t.participant[gBase + g] = participates ? 1 : 0;
    }
    // the slack bus mismatch: its P balance against its target, which carries its own generators' moves
    int slackRow = t.slackRowOverride != nullptr && t.slackRowOverride[s] >= 0 ? t.slackRowOverride[s] : t.slackRowBase;
    double slackTargetP = t.slackTargetP + (t.slackBusAnchorShift != nullptr ? t.slackBusAnchorShift[s] : 0.0);
    for (int g = 0; g < t.nGenerators; g++) {
        if (t.generatorBus[g] < 0) {
            slackTargetP += t.move[gBase + g];
        }
    }
    double slackBusActivePowerMismatch = t.qBuf[(size_t) s * t.n + slackRow] - slackTargetP;
    double* slots = t.slots + (size_t) s * (t.nAreas + 1) * SLOT_WIDTH;
    for (int k = 0; k <= t.nAreas; k++) {
        slots[(size_t) k * SLOT_WIDTH + SLOT_PRESENT] = 0.0;
    }
    bool anyAreaToBalance = false;
    for (int a = 0; a < t.nAreas; a++) {
        double interchange = 0.0;
        for (int b = t.boundaryStart[a]; b < t.boundaryStart[a + 1]; b++) {
            interchange += boundaryP(t, s, b);
        }
        slots[(size_t) a * SLOT_WIDTH + SLOT_INTERCHANGE] = interchange;
        double withSlack = interchange - t.interchangeTarget[a] + t.slackDistributionFactor[a] * slackBusActivePowerMismatch;
        anyAreaToBalance |= !lessThanInterchangeMaxMismatch(t, withSlack);
    }
    if (anyAreaToBalance) {
        // the areas whose interchange, with their share of the slack mismatch, is out of the band: each on its own buses
        for (int a = 0; a < t.nAreas; a++) {
            double* slot = slots + (size_t) a * SLOT_WIDTH;
            double withSlack = slot[SLOT_INTERCHANGE] - t.interchangeTarget[a] + t.slackDistributionFactor[a] * slackBusActivePowerMismatch;
            if (lessThanInterchangeMaxMismatch(t, withSlack)) {
                continue;
            }
            DistributionResult r = runActivePowerDistribution(t, s, a, withSlack);
            slot[SLOT_PRESENT] = 1.0;
            slot[SLOT_TYPE_SLACK] = 0.0;
            slot[SLOT_DISTRIBUTED] = withSlack - r.remainingMismatch;
            slot[SLOT_REMAINING] = r.remainingMismatch;
            slot[SLOT_MOVED] = r.movedBuses ? 1.0 : 0.0;
        }
    } else {
        bool anyInterchangeMismatch = false;
        for (int a = 0; a < t.nAreas; a++) {
            anyInterchangeMismatch |= !lessThanInterchangeMaxMismatch(t, slots[(size_t) a * SLOT_WIDTH + SLOT_INTERCHANGE] - t.interchangeTarget[a]);
        }
        if (!anyInterchangeMismatch && lessThanSlackBusMaxMismatch(t, slackBusActivePowerMismatch)) {
            t.moveValid[s] = 1;
            return;                                                            // already balanced: STABLE
        }
        // the slack bus mismatch on the buses without area
        double* noArea = slots + (size_t) t.nAreas * SLOT_WIDTH;
        DistributionResult r = runActivePowerDistribution(t, s, -1, slackBusActivePowerMismatch);
        if (lessThanSlackBusMaxMismatch(t, r.remainingMismatch)) {
            noArea[SLOT_PRESENT] = 1.0;
            noArea[SLOT_TYPE_SLACK] = 0.0;
            noArea[SLOT_DISTRIBUTED] = slackBusActivePowerMismatch - r.remainingMismatch;
            noArea[SLOT_REMAINING] = r.remainingMismatch;
            noArea[SLOT_MOVED] = r.movedBuses ? 1.0 : 0.0;
        } else {
            // distributeRemainingSlackMismatch: what they could not take, over the areas by their interchange margins
            double mismatch = r.remainingMismatch;
            for (int a = 0; a < t.nAreas; a++) {
                double* slot = slots + (size_t) a * SLOT_WIDTH;
                slot[SLOT_FACTOR] = signum(mismatch) * (slot[SLOT_INTERCHANGE] - t.interchangeTarget[a] + t.slackDistributionFactor[a] * mismatch)
                        + t.areaInterchangePMaxMismatch;
            }
            normalizeSlackParticipationFactors(t, slots);
            double remainingMismatch = mismatch;
            while (true) {
                double factorSum = 0.0;
                for (int a = 0; a < t.nAreas; a++) {
                    factorSum += slots[(size_t) a * SLOT_WIDTH + SLOT_FACTOR];
                }
                if (!(factorSum > 0) || !(fabs(remainingMismatch) > t.residueEps)) {
                    break;
                }
                remainingMismatch = distributeOnAreas(t, s, slots, remainingMismatch);
                normalizeSlackParticipationFactors(t, slots);
            }
        }
    }
    // buildOuterLoopResult
    bool remainingMismatches = false;
    double totalDistributedActivePower = 0.0;
    bool movedBuses = false;
    for (int k = 0; k <= t.nAreas; k++) {
        const double* slot = slots + (size_t) k * SLOT_WIDTH;
        if (slot[SLOT_PRESENT] == 0.0) {
            continue;
        }
        remainingMismatches |= slot[SLOT_TYPE_SLACK] != 0.0
                ? !lessThanSlackBusMaxMismatch(t, slot[SLOT_REMAINING])
                : !lessThanInterchangeMaxMismatch(t, slot[SLOT_REMAINING]);
        totalDistributedActivePower += slot[SLOT_DISTRIBUTED];
        movedBuses |= slot[SLOT_MOVED] != 0.0;
    }
    t.distributedActivePower[s] += totalDistributedActivePower;
    if (remainingMismatches) {
        t.failure[s] = 1;                                                  // the run's slackDistributionFailureBehavior decides
    }
    if (movedBuses) {
        t.unstable[s] = 1;
    }
    t.moveValid[s] = 1;
}

__global__ void areaInterchangeControlResetSolveBatch(int nS, int nGenerators, const double* initialMove, double* move, char* moveValid,
                                                     double* distributedActivePower, const char* installed) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) {
        return;
    }
    if (installed != nullptr && !installed[s]) return;   // a slot this install does not seed
    for (int g = 0; g < nGenerators; g++) {
        move[(size_t) s * nGenerators + g] = initialMove != nullptr ? initialMove[(size_t) s * nGenerators + g] : 0.0;
    }
    moveValid[s] = 1;
    distributedActivePower[s] = 0.0;
}

}  // namespace

void areaInterchangeControlResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& nS = drv.nS;
    ctx->dAreaInterchangeControlDistributedActivePower.alloc((size_t) nS * sizeof(double));
    if (ctx->nAreas == 0 || ctx->nDistributedSlackGenerators == 0) {
        resetInstalledRows(ctx, ctx->dAreaInterchangeControlDistributedActivePower.as<double>(), 1, nullptr, 0.0);
        return;
    }
    areaInterchangeControlResetSolveBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, ctx->nDistributedSlackGenerators,
            ctx->distributedSlackGeneratorInitialMove.deviceOrNull(),
            ctx->dDistributedSlackGeneratorMove.as<double>(), ctx->dDistributedSlackGeneratorMoveValid.as<char>(),
            ctx->dAreaInterchangeControlDistributedActivePower.as<double>(), installMaskOrNull(ctx));
}

void areaInterchangeControlCheck(BatchSolverContext* ctx, DriverPass& drv) {
    if (ctx->nAreas == 0) {
        distributedSlackCheck(ctx, drv);                                   // OLF's noAreaOuterLoop
        return;
    }
    const int& n = drv.n;
    const int& nS = drv.nS;
    int nGenerators = ctx->nDistributedSlackGenerators;
    if (!ctx->dSlackDistributionFailure.p) ctx->dSlackDistributionFailure.alloc(nS * sizeof(char));
    CUDA_CHECK(cudaMemsetAsync(ctx->dSlackDistributionFailure.p, 0, nS * sizeof(char), ctx->stream));
    ctx->dAreaInterchangeControlAreaScratch.alloc((size_t) nS * (ctx->nAreas + 1) * SLOT_WIDTH * sizeof(double));
    ctx->dAreaInterchangeControlPreviousMove.alloc((size_t) nS * nGenerators * sizeof(double));
    ctx->dAreaInterchangeControlParticipant.alloc((size_t) nS * nGenerators * sizeof(char));
    Tables t{};
    t.n = n;
    t.nS = nS;
    t.nElements = ctx->nElements;
    t.slackRowBase = ctx->slackBusPhiRow;
    t.slackRowOverride = ctx->slackBusRowOverride.deviceOrNull();
    t.slackTargetP = ctx->slackBusTargetP;
    t.slackBusAnchorShift = ctx->nDistributedSlackBuses > 0 ? ctx->slackBusAnchorShift.deviceOrNull() : nullptr;
    t.nAreas = ctx->nAreas;
    t.interchangeTarget = ctx->dAreaInterchangeTarget.as<double>();
    t.slackDistributionFactor = ctx->dAreaSlackDistributionFactor.as<double>();
    t.boundaryStart = ctx->dAreaBoundaryStart.as<int>();
    t.boundaryKind = ctx->dAreaBoundaryKind.as<int>();
    t.boundaryElement = ctx->dAreaBoundaryElement.as<int>();
    t.boundaryMaskSlot = ctx->dAreaBoundaryMaskSlot.as<int>();
    t.boundarySide = ctx->dAreaBoundarySide.as<int>();
    t.boundaryTieLineHalf = ctx->dAreaBoundaryTieLineHalf.as<double>();
    t.closedBranchParameters = ctx->structure->closedBranch.parameters.as<double>();
    t.closedBranchIndices = ctx->structure->closedBranch.indices.as<int>();
    t.openBranchParameters = ctx->structure->openBranch.parameters.as<double>();
    t.openBranchIndices = ctx->structure->openBranch.indices.as<int>();
    t.closedBranchParameterOverrides = closedBranchParameterOverrides(ctx);
    t.nBuses = ctx->nDistributedSlackBuses;
    t.busPhiRow = ctx->dDistributedSlackBusPhiRow.as<int>();
    t.busRowOverride = ctx->distributedSlackBusRowOverride.deviceOrNull();
    t.busArea = ctx->dDistributedSlackBusArea.as<int>();
    t.busGeneratorStart = ctx->dDistributedSlackBusGeneratorStart.as<int>();
    t.busGeneratorEnd = ctx->dDistributedSlackBusGeneratorEnd.as<int>();
    t.busBaseTargetP = ctx->dDistributedSlackBusBaseTargetP.as<double>();
    t.busAnchorShift = ctx->distributedSlackBusAnchorShift.deviceOrNull();
    t.islanded = ctx->distributedSlackIslanded.deviceOrNull();
    t.nGenerators = nGenerators;
    t.generatorBus = ctx->distributedSlackGeneratorBus.as<int>();
    t.generatorArea = ctx->dDistributedSlackGeneratorArea.as<int>();
    t.generatorFactor = ctx->distributedSlackGeneratorFactor.as<double>();
    t.generatorMaxDelta = ctx->distributedSlackGeneratorMaxDelta.as<double>();
    t.generatorMinDelta = ctx->distributedSlackGeneratorMinDelta.as<double>();
    t.generatorMarginUp = ctx->distributedSlackGeneratorMarginUp.as<double>();
    t.generatorMarginDown = ctx->distributedSlackGeneratorMarginDown.as<double>();
    t.factorMode = ctx->distributedSlackFactorMode;
    t.lostGenerators = ctx->distributedSlackLostGenerators.view();
    t.removedGenerators = ctx->distributedSlackRemovedGenerators.view();
    t.slackBusPMaxMismatch = drv.slackDistTol;
    t.areaInterchangePMaxMismatch = ctx->areaInterchangePMaxMismatch;
    t.residueEps = drv.pResidueEps;
    t.x = ctx->dX.as<double>();
    t.qBuf = ctx->dQbuf.as<double>();
    t.disabled = ctx->dDisabled.as<char>();
    t.outerLoopCheckActive = ctx->dOuterLoopCheckActive.as<char>();
    t.target = ctx->dTarget.as<double>();
    t.move = ctx->dDistributedSlackGeneratorMove.as<double>();
    t.workingFactor = ctx->dDistributedSlackGeneratorWorkingFactor.as<double>();
    t.previousMove = ctx->dAreaInterchangeControlPreviousMove.as<double>();
    t.slots = ctx->dAreaInterchangeControlAreaScratch.as<double>();
    t.distributedActivePower = ctx->dAreaInterchangeControlDistributedActivePower.as<double>();
    t.removed = ctx->dDistributedSlackGeneratorRemoved.as<char>();
    t.participant = ctx->dAreaInterchangeControlParticipant.as<char>();
    t.unstable = ctx->dOuterLoopUnstable.as<char>();
    t.failure = ctx->dSlackDistributionFailure.as<char>();
    t.moveValid = ctx->dDistributedSlackGeneratorMoveValid.as<char>();
    areaInterchangeControlBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(t);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace olfgpu
