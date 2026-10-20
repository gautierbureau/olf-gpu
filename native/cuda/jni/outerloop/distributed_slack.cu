// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See distributed_slack.cuh.
#include <cuda_runtime.h>

#include <limits>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_distributed_slack_factor_mode.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"
#include "../common/codes/gpu_ac_emulation_status.cuh"
#include "hvdc.cuh"
#include "forced_target_q.cuh"
#include "slack_participants.cuh"
#include "distributed_slack.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

// Whether the slack bus's own participating generators are in the per-generator pack (distributedSlackGeneratorBus -1, no row).
__host__ __device__ inline bool slackGeneratorsInPack(int nDistributedSlackGenerators, const int* distributedSlackGeneratorBus) {
    for (int g = 0; g < nDistributedSlackGenerators; g++) {
        if (distributedSlackGeneratorBus[g] < 0) {
            return true;
        }
    }
    return false;
}

// OLF's slack mismatch is slack.getP() - slack.getTargetP() (DistributedSlackOuterLoop): getP() carries the
// AC-emulation term of a link ending on the slack while the link is linear, and getTargetP() the converter
// station's target P while it is frozen or saturated. qBuf holds the branch flows only, and the slack's target P
// is the base one, so a link on the slack bus is put back here: its live term on the balance side, its station's
// move since the base (appliedD, see hvdcAcEmulationLoopBatch) on the target side. Without it the frozen link of
// a post-contingency solve read as an 84 MW slack mismatch where OLF saw 0.66 MW, and the device ran slack passes
// OLF never ran.
__global__ void hvdcSlackBalanceBatch(int nS, int nh, int n, int nElements, int slackRowBase, const int* slackRowOv,
                                      const int* hvdcBus1PRow, const int* hvdcBus2PRow, const int* hvdcBus1PhiColumn,
                                      const int* hvdcBus2PhiColumn, const int* hvdcElement, const double* hvdcAcEmulationParameters,
                                      const int* status, const double* d1, const double* d2, const char* ctgOut,
                                      const char* disabled, const double* x, const char* outerLoopCheckActive, double* qBuf) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS || !outerLoopCheckActive[s]) return;
    int slackRow = slackRowOv != nullptr && slackRowOv[s] >= 0 ? slackRowOv[s] : slackRowBase;
    const double* xs = x + (size_t) s * n;
    double corr = 0.0;
    for (int h = 0; h < nh; h++) {
        int r1 = hvdcBus1PRow[h] < 0 ? ~hvdcBus1PRow[h] : hvdcBus1PRow[h];
        int r2 = hvdcBus2PRow[h] < 0 ? ~hvdcBus2PRow[h] : hvdcBus2PRow[h];
        if (r1 != slackRow && r2 != slackRow) continue;
        size_t sh = (size_t) s * nh + h;
        const double* p = hvdcAcEmulationParameters + (size_t) h * 9;
        bool live = status[sh] == GpuAcEmulationStatus::LINEAR_MODE && !ctgOut[sh] && !disabled[(size_t) s * nElements + hvdcElement[h]];
        double ph1 = xs[hvdcBus1PhiColumn[h]];
        double ph2 = xs[hvdcBus2PhiColumn[h]];
        if (r1 == slackRow) {
            corr += (live ? hvdc::p1(p[0], p[1], p[4], p[5], p[6], ph1, ph2) : 0.0) - d1[sh];
        }
        if (r2 == slackRow) {
            corr += (live ? hvdc::p2(p[0], p[1], p[4], p[5], p[6], ph1, ph2) : 0.0) - d2[sh];
        }
    }
    qBuf[(size_t) s * n + slackRow] += corr;
}

// Slack-only distributed slack: the slack generator is the ONLY participant, so OLF's distribution
// puts the whole mismatch into the slack gen's own target — no flow change, but the outer loop reports
// UNSTABLE and triggers a re-solve that polishes the state by one more Newton run. Mirror it exactly:
// track the absorbed total, flag a switch while the residual mismatch stays above the outer gate.
// anchor: the slack generators' INITIAL target, relative to slackTargetP - 0 for a contingency (OLF's first pass
// undoes the loss pre-distribution), the parent's absorbed total for a strategy pair (OLF re-anchors there).
// OLF's ActivePowerDistribution resets to the anchor and places the whole remaining mismatch only above
// P_RESIDUE_EPS: below it the generators stay at the anchor, and the outer loop still re-solves.
// roomUp/roomDown: the slack generators' total rooms from their base targets. OLF's water-fill places at most
// that, from the anchor; what is left is its remaining mismatch, flagged in residual (FAIL / THROW decide on it),
// and a check that finds the generators already saturated moves nothing (movedBuses false: STABLE).
__global__ void slackOnlyDistributedSlackCheck(int nScenarios, int n, int slackPhiRowBase, const int* slackRowOv, double slackTargetP,
                                 double slackDistTol, double residueEps, const double* qBuf, const char* outerLoopCheckActive,
                                 const double* anchor, double roomUp, double roomDown,
                                 double* absorbed, char* switched, char* residual) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !outerLoopCheckActive[s]) return;
    int slackPhiRow = slackRowOv != nullptr && slackRowOv[s] >= 0 ? slackRowOv[s] : slackPhiRowBase;   // relocated slack
    double m = qBuf[(size_t) s * n + slackPhiRow] - slackTargetP - absorbed[s];
    if (fabs(m) > slackDistTol && fabs(m) > residueEps) {   // OLF's check: both bounds
        double a = anchor != nullptr ? anchor[s] : 0.0;
        double total = absorbed[s] + m - a;                  // this check's mismatch plus the undone previous move
        double placed = fmin(fmax(total, roomDown - a), roomUp - a);
        if (fabs(total - placed) > residueEps) {
            residual[s] = 1;
        }
        double next = a + (fabs(placed) > residueEps ? placed : 0.0);
        if (placed == total || next != absorbed[s]) {
            switched[s] = 1;
        }
        absorbed[s] = next;
    }
}

__global__ void computeSlackRetain(int nScenarios, int n, int nDistributedSlackBuses, int slackPhiRowBase, const int* slackRowOv, double slackTargetP,
                                   double oneMinusFactorSum, const int* distributedSlackBusPhiRow, const int* distributedSlackBusRowOverride,
                                   const double* distributedSlackBusBaseTargetP,
                                   const double* target, const double* qBuf, const char* outerLoopCheckActive,
                                   double* retain,
                                   const int* distributedSlackOverrideBus, const double* distributedSlackOverrideBaseTargetPDelta,
                                   const double* distributedSlackBusAnchorShift, const double* slackBusAnchorShift,
                                   int distributedSlackFactorMode, const double* distributedSlackBusMarginUp, const double* distributedSlackBusMarginDown,
                                   double slackMarginUp, double slackMarginDown,
                                   const char* distributedSlackIslanded, const double* distributedSlackBusFactor, const double* distributedSlackOverrideFactorDelta,
                                   int nDistributedSlackGenerators, const int* distributedSlackGeneratorBus, const double* distributedSlackGeneratorInitialMove, const double* retainApplied) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!outerLoopCheckActive[s]) return;
    if (slackGeneratorsInPack(nDistributedSlackGenerators, distributedSlackGeneratorBus)) {
        // The slack's generators are in the water-fill list: its share is what the last distribution gave them
        // (clamped like anyone's), not a fixed fraction of the total. Before any pass in this solve that is the
        // loss pre-distribution's move of them (a pair starts at its parent's, which its slack target holds).
        double placed = 0.0;
        if (retainApplied != nullptr && !isnan(retainApplied[s])) {
            placed = retainApplied[s];
        } else if (distributedSlackGeneratorInitialMove != nullptr) {
            for (int g = 0; g < nDistributedSlackGenerators; g++) {
                if (distributedSlackGeneratorBus[g] < 0) {
                    placed += distributedSlackGeneratorInitialMove[(size_t) s * nDistributedSlackGenerators + g];
                }
            }
        }
        retain[s] = placed;
        return;
    }
    int slackPhiRow = slackRowOv != nullptr && slackRowOv[s] >= 0 ? slackRowOv[s] : slackPhiRowBase;   // relocated slack
    double absorbed = 0.0;
    double fsActive = 0.0;                                   // Σ f over the participants still in the main component
    double fsAll = 0.0;
    int ov = distributedSlackOverrideBus != nullptr ? distributedSlackOverrideBus[s] : -1;
    for (int i = 0; i < nDistributedSlackBuses; i++) {
        fsAll += distributedSlackBusFactor[i];
        if (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + i]) {
            continue;                                        // in another component: not a participant here
        }
        // a lost participating generator leaves the participation like a stranded one: its factor goes
        fsActive += distributedSlackBusFactor[i] - (ov == i && distributedSlackOverrideFactorDelta != nullptr ? distributedSlackOverrideFactorDelta[s] : 0.0);
        // A composed injection override on a participant bus (redispatch action, load delta) shifts
        // the row target WITHOUT being distributed power — the effective base must shift with it,
        // or the retain math counts the override as absorbed slack and the waterfill CANCELS it
        // (measured: a +60 MW redispatch read back as a no-op, GeneratorAction drill 2026-07-19).
        double base = distributedSlackBusBaseTargetP[i];
        if (distributedSlackBusAnchorShift != nullptr) {
            base += distributedSlackBusAnchorShift[(size_t) s * nDistributedSlackBuses + i];     // pair leg: cumulative reference re-anchored at the
        }                                                    // PARENT contingency's distributed targets (OLF's
        if (ov == i) {                                       // setGeneratorsInitialTargetPToTargetP before the
            base -= distributedSlackOverrideBaseTargetPDelta[s];                        // strategy re-solve)
        }
        absorbed += target[(size_t) s * n + distributedSlackBusRowOf(distributedSlackBusPhiRow, distributedSlackBusRowOverride, s, nDistributedSlackBuses, i)] - base;
    }
    double slackTgt = slackTargetP + (slackBusAnchorShift != nullptr ? slackBusAnchorShift[s] : 0.0);
    double tot = (qBuf[(size_t) s * n + slackPhiRow] - slackTgt) + absorbed;   // total-from-anchor, OLF's
    double rho = oneMinusFactorSum;                                            // remainingMismatch at run() start
    if (fsActive != fsAll && distributedSlackFactorMode != GpuDistributedSlackFactorMode::REMAINING_MARGIN) {
        // The factors were normalized over ALL participants (W), so rho = w_slack/W. Dropping the
        // stranded or lost ones renormalizes over W' = W - Σ_dropped w, and w_slack/W' = rho/(Σ_active f + rho).
        // Leaving rho global is exactly what made the first attempt at this fail: the participants
        // were masked out of the distribution loops while the slack kept the share it had when they
        // were still in, so the pass retired with most of the mismatch undistributed.
        rho = fsActive + rho > 0 ? rho / (fsActive + rho) : 0.0;
    }
    if (distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN) {
        // REMAINING_MARGIN: the slack's retained share is its own margin share in the DIRECTION of the
        // total-from-anchor (OLF evaluates the direction-dependent factors once per run at the anchor).
        // Margins shift with a moved anchor: +delta of anchor target eats headroom, frees footroom.
        double slackM = tot > 0 ? slackMarginUp : slackMarginDown;
        if (slackBusAnchorShift != nullptr) {
            slackM = fmax(0.0, slackM - (tot > 0 ? slackBusAnchorShift[s] : -slackBusAnchorShift[s]));
        }
        double sum = slackM;
        for (int i = 0; i < nDistributedSlackBuses; i++) {
            if (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + i]) {
                continue;                                    // REMAINING_MARGIN needs no rescale: dropping
            }                                                // the stranded margins IS the renormalization
            double m = tot > 0 ? distributedSlackBusMarginUp[i] : distributedSlackBusMarginDown[i];
            if (distributedSlackBusAnchorShift != nullptr) {
                double sh = distributedSlackBusAnchorShift[(size_t) s * nDistributedSlackBuses + i];
                m = fmax(0.0, m - (tot > 0 ? sh : -sh));
            }
            sum += m;
        }
        rho = sum > 0 ? slackM / sum : 0.0;
    }
    retain[s] = rho * tot;
}

// Distribute the slack mismatch by FULL WATER-FILLING per scenario (one thread = one scenario), matching OLF's
// ActivePowerDistribution.run + GenerationActivePowerDistributionStep.run: distribute the whole mismatch among
// the participating buses, re-normalizing the factors over the UNSATURATED buses each round and removing buses
// that hit a limit, iterating until the residual falls below residueEps (P_RESIDUE_EPS) or all buses saturate —
// all WITHOUT a power-flow re-solve. The full water-fill at a FIXED operating point is what matches OLF under
// saturation: a single clamp step per outer pass (letting the saturated residual reappear at the slack, to be
// redistributed only AFTER the next re-solve) would drift — the loss change between re-solves converges to a
// different in-tolerance point than OLF.
__global__ void distributeSlackBatch(int nDistributedSlackBuses, int nScenarios, int n, double slackDistTol, double residueEps,
                                              int slackPhiRowBase, const int* slackRowOv, double slackTargetP, const int* distributedSlackBusPhiRow, const int* distributedSlackBusRowOverride,
                                              const double* distributedSlackBusFactor, const double* distributedSlackBusBaseTargetP, const double* distributedSlackBusAnchorShift, const double* slackBusAnchorShift,
                                              const double* dsMaxDelta, const double* dsMinDelta, const double* slackRetainedShare,
                                              const double* qBuf, const char* outerLoopCheckActive,
                                              const int* distributedSlackOverrideBus, const double* distributedSlackOverrideFactorDelta, const double* distributedSlackOverrideBaseTargetPDelta,
                                              const double* distributedSlackOverrideMaxDelta, const double* distributedSlackOverrideMinDelta,
                                              int distributedSlackFactorMode, const double* distributedSlackBusMarginUp, const double* distributedSlackBusMarginDown,
                                              double slackMarginUp, double slackMarginDown,
                                              const char* distributedSlackIslanded,
                                              int nDistributedSlackGenerators, const int* distributedSlackGeneratorBus, const double* distributedSlackGeneratorFactor,
                                              const double* distributedSlackGeneratorMaxDelta, const double* distributedSlackGeneratorMinDelta,
                                              const double* distributedSlackGeneratorMarginUp, const double* distributedSlackGeneratorMarginDown,
                                              const int* distributedSlackBusGeneratorStart, const int* distributedSlackBusGeneratorEnd,
                                              double* distributedSlackGeneratorMove, char* distributedSlackGeneratorMoveValid, const double* distributedSlackGeneratorAnchor, const double* distributedSlackGeneratorInitialMove, char* distributedSlackGeneratorRemoved, double* distributedSlackGeneratorWorkingFactor,
                                              int nDistributedSlackLostGenerators, const int* distributedSlackLostGenerators, int nDistributedSlackRemovedGenerators, const int* distributedSlackRemovedGenerators,
                                              double* target, char* switched, char* slackDistributionFailure, double* retainApplied, double* gateLast) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!outerLoopCheckActive[s]) return;
    int slackPhiRow = slackRowOv != nullptr && slackRowOv[s] >= 0 ? slackRowOv[s] : slackPhiRowBase;   // relocated slack
    // OLF's DistributedSlackOuterLoop.check distributes only when the slack mismatch exceeds BOTH
    // slackBusPMaxMismatch and ActivePowerDistribution.P_RESIDUE_EPS; the second only binds below 1 kW.
    double gateTol = fmax(slackDistTol, residueEps);
    // The participating buses share the mismatch in EXCESS of the slack's retained share (rem); only distribute
    // past the slackBusPMaxMismatch outer gate, exactly like OLF's DistributedSlackOuterLoop.check.
    double slackTgt = slackTargetP + (slackBusAnchorShift != nullptr ? slackBusAnchorShift[s] : 0.0);
    double rem = (qBuf[(size_t) s * n + slackPhiRow] - slackTgt) - slackRetainedShare[s];
    int ov = (distributedSlackOverrideBus != nullptr) ? distributedSlackOverrideBus[s] : -1;        // the bus losing a participating gen (gen contingency), -1 = none
    // REMAINING_MARGIN: factors are the per-bus margins in the DIRECTION of the pass's remaining mismatch,
    // picked ONCE at entry (OLF evaluates them once per run at the anchor); a moved anchor (+delta) eats
    // headroom and frees footroom symmetrically.
    bool dirUp = rem > 0;
    // rem is the NON-SLACK share of the CPU's slack mismatch: rem = (1 - rho) * m_cpu, with rho the slack's
    // GLOBAL retained share (the same constant the retain kernel uses — a per-scenario participant reweight
    // does not change it). OLF's outer gate tests |m_cpu| > slackBusPMaxMismatch, so the device gate scales
    // by (1 - rho) = the GLOBAL participant factor sum: unscaled it is 1/(1-rho) looser and lets the device
    // retire with up to (1/(1-rho) - 1) * tol still undistributed (measured 4.9e-4 kV on a 60 MW redispatch
    // pair with a participating slack, 2x); scaling by the per-scenario EFFECTIVE sum instead over-tightens
    // a gen-outage scenario with a non-participating slack and adds a round the CPU never runs (3.9e-4).
    double fsGlobal = 0.0;
    bool noCapacity = false;                                 // nobody — the slack included — can move this way
    if (distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN) {
        double sumM = 0.0;
        for (int i = 0; i < nDistributedSlackBuses; i++) {
            if (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + i]) {
                continue;
            }
            double m = dirUp ? distributedSlackBusMarginUp[i] : distributedSlackBusMarginDown[i];
            if (distributedSlackBusAnchorShift != nullptr) {
                double sh = distributedSlackBusAnchorShift[(size_t) s * nDistributedSlackBuses + i];
                m = fmax(0.0, m - (dirUp ? sh : -sh));
            }
            sumM += m;
        }
        double slackM = dirUp ? slackMarginUp : slackMarginDown;
        if (slackBusAnchorShift != nullptr) {
            slackM = fmax(0.0, slackM - (dirUp ? slackBusAnchorShift[s] : -slackBusAnchorShift[s]));
        }
        fsGlobal = sumM + slackM > 0 ? sumM / (sumM + slackM) : 0.0;
        noCapacity = !(sumM + slackM > 0);
    } else {
        double fsActive = 0.0;
        double fsAll = 0.0;
        for (int i = 0; i < nDistributedSlackBuses; i++) {
            fsAll += distributedSlackBusFactor[i];
            if (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + i]) {
                continue;
            }
            fsActive += distributedSlackBusFactor[i] - (ov == i && distributedSlackOverrideFactorDelta != nullptr ? distributedSlackOverrideFactorDelta[s] : 0.0);
        }
        // With every participant present this is Σf, bit-for-bit what it was. With some stranded or a
        // participating generator lost it is the survivors' share of the RENORMALIZED set, 1 - rho' = fsActive/(fsActive + rho) with
        // rho = 1 - Σf the slack's original share — the same formula computeSlackRetain uses for
        // rho', so the two halves of the split still add up to the whole mismatch. The gate below
        // then scales the tolerance by the share these participants really carry, which is what
        // OLF's gate tests on its surviving participant set.
        double rho = 1.0 - fsAll;
        fsGlobal = fsActive == fsAll
                ? fsActive
                : (fsActive + rho > 0 ? fsActive / (fsActive + rho) : 0.0);
        // every participant stranded and no slack share (rho is 1 minus a normalized sum: zero up to rounding)
        noCapacity = !(fsActive > 0.0) && rho < 1e-12;
    }
    // The slack's own generators in the list: the water-fill places the WHOLE total, theirs included, and slackRetainedShare
    // is what they hold now, so rem is OLF's slack mismatch itself and the gate needs no scaling.
    bool slackInPack = nDistributedSlackGenerators > 0 && slackGeneratorsInPack(nDistributedSlackGenerators, distributedSlackGeneratorBus);
    if (slackInPack) {
        fsGlobal = 1.0;
        noCapacity = false;
    }
    if (gateLast != nullptr) {
        // The WHOLE accounting of one slack pass, host-readable (device printf does not reach the test log).
        // [0] rem the gate compares, [1] fired, [2] the slack's own injection, [3] its target, [4] the
        // mismatch between them, [5] the share the slack is assumed to hold, [6] the participants' global
        // factor sum, [7] the amount the water-fill places (written below). OLF's counterpart is one line,
        // `DS_RUN mismatch=... participants=n`, and the two TOTALS must agree: a difference in the amount
        // distributed is a defect, not a different route.
        gateLast[8 * s] = rem;
        gateLast[8 * s + 1] = (fsGlobal > 0.0 && fabs(rem) > fsGlobal * gateTol) ? 1.0 : 0.0;
        gateLast[8 * s + 2] = qBuf[(size_t) s * n + slackPhiRow];
        gateLast[8 * s + 3] = slackTgt;
        gateLast[8 * s + 4] = qBuf[(size_t) s * n + slackPhiRow] - slackTgt;
        gateLast[8 * s + 5] = slackRetainedShare[s];
        gateLast[8 * s + 6] = fsGlobal;
        gateLast[8 * s + 7] = 0.0;
    }
    if (fsGlobal <= 0.0) {
        // No participant share. With the slack's own margin left this is OLF distributing onto the slack's
        // generators, which succeeds. With NOBODY able to move — every participant and the slack at a limit
        // in this direction (REMAINING_MARGIN), or every participant stranded with a non-participating
        // slack — OLF's participating set is empty, ActivePowerDistribution.run leaves the whole mismatch
        // and handleDistributionFailureBehavior decides. Flag it like a saturated water-fill: the host
        // rejects the scenario under a behaviour other than LEAVE_ON_SLACK_BUS and keeps today's answer
        // (the mismatch stays at the slack) under it. OLF's gate is on the raw slack mismatch, above
        // slackBusPMaxMismatch and above P_RESIDUE_EPS; nothing has been retained or absorbed here.
        double mis = qBuf[(size_t) s * n + slackPhiRow] - slackTgt;
        if (noCapacity && fabs(mis) > slackDistTol && fabs(mis) > residueEps) {
            slackDistributionFailure[s] = 1;
        }
        if (!noCapacity && retainApplied != nullptr) {
            // Only the slack bus's generators participate (every other participant lost or stranded). OLF's
            // check measures the slack's mismatch against the target its last distribution gave those
            // generators, distributes past slackBusPMaxMismatch onto them, and the outer loop re-solves. The
            // slack carries the whole total from the anchor here; what it took at the last pass that fired is
            // retainApplied (NaN before any), relative to a target the loss pre-distribution already shifted.
            double prev = isnan(retainApplied[s]) ? 0.0 : retainApplied[s];
            if (fabs(mis - prev) > gateTol) {
                retainApplied[s] = mis;
                switched[s] = 1;
            }
        }
        return;
    }
    if (fabs(rem) <= fsGlobal * gateTol) {
        return;
    }
    if (retainApplied != nullptr) {
        retainApplied[s] = slackRetainedShare[s];                      // the slack's share of THIS distribution (applied to the network)
    }
    // OLF's ActivePowerDistribution.run RESETS every participant to its INITIAL target
    // (step.resetToInitialState) and adds back what that undid (previousStateInfo.previousMismatch()),
    // then re-derives the WHOLE distribution from the anchor. Accumulating each pass on top of the
    // previous one is not the same function once anyone saturates: the re-normalisation over the
    // still-unsaturated participants is taken against a different remaining amount. Ported here for
    // FIDELITY — it is not what closes the residual rte6515 gap (that is the reactive-limits loop),
    // but the incremental form was not what OLF does.
    double absorbedNow = 0.0;
    for (int i = 0; i < nDistributedSlackBuses; i++) {
        if (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + i]) {
            continue;                                        // its row is frozen; it never absorbed anything
        }
        double base = distributedSlackBusBaseTargetP[i];
        if (distributedSlackBusAnchorShift != nullptr) { base += distributedSlackBusAnchorShift[(size_t) s * nDistributedSlackBuses + i]; }
        if (ov == i) { base -= distributedSlackOverrideBaseTargetPDelta[s]; }
        size_t idx = (size_t) s * n + distributedSlackBusRowOf(distributedSlackBusPhiRow, distributedSlackBusRowOverride, s, nDistributedSlackBuses, i);
        absorbedNow += target[idx] - base;
        target[idx] = base;                                 // reset to the anchor
    }
    rem = fsGlobal * ((qBuf[(size_t) s * n + slackPhiRow] - slackTgt) + absorbedNow);
    if (gateLast != nullptr) {
        // Slot 7, the documented `placed`. This read 6*s+5 — the stride of the 6-slot record this
        // one replaced: for s=0 it overwrote `slackRetainedShare` with the placed amount (so a FIRING pass
        // printed the wrong column and dsPlaced stayed 0), and for every s>=1 it scribbled into
        // ANOTHER scenario's slots (s=1 -> index 11 = scenario 1's slackTgt). Every LOOP_PASS number
        // read off a firing pass was therefore the wrong quantity.
        gateLast[8 * s + 7] = rem;                           // what this pass places over the participants
    }
    bool changed = false;
    // PER-GENERATOR water-fill, the shape of OLF's GenerationActivePowerDistributionStep.run: factors
    // renormalized over the elements still in the list at each iteration start, one pass using the
    // mismatch as of that start, each element clamped at its OWN limit and REMOVED when it strictly
    // overshoots. A bus's row is then the sum of its generators' cumulative deltas.
    //
    // Used whenever the pack carries generators and THIS scenario is not re-anchored. A pair leg's
    // distributedSlackBusAnchorShift is a BUS-level displacement (the parent contingency's absorption, carried so the
    // leg does not undo it) and nothing in the pack says how to split it across a bus's generators, so
    // a re-anchored scenario keeps the aggregate loop below — unchanged, and no less faithful than it
    // was. The test is on the VALUE, not on the pointer: an orchestrator re-solve passes the carry
    // arrays for every scenario, all zeros for a plain N-1, and taking that as "re-anchored" quietly
    // sent every post-orchestrator pass down the aggregate path (measured: the first two DS rounds of
    // LINE-3320-350 matched the CPU digit for digit, the third did not).
    bool anchored = false;
    if (distributedSlackBusAnchorShift != nullptr) {
        for (int i = 0; i < nDistributedSlackBuses; i++) {
            if (distributedSlackBusAnchorShift[(size_t) s * nDistributedSlackBuses + i] != 0.0) {
                anchored = true;
                break;
            }
        }
    }
    // A re-anchored scenario that carries its PER-GENERATOR anchors (distributedSlackGeneratorAnchor: a strategy pair, each generator's
    // post-contingency target) takes the per-generator water-fill too, from those anchors.
    bool perGen = nDistributedSlackGenerators > 0 && (!anchored || distributedSlackGeneratorAnchor != nullptr);
    if (distributedSlackGeneratorMoveValid != nullptr) {
        distributedSlackGeneratorMoveValid[s] = perGen ? 1 : 0;
    }
    if (perGen) {
        // The outaged bus's SURVIVORS join the water-fill like any others, with the generators this
        // scenario removes marked out by identity below. The gate is the BUS, not the lost unit: a
        // lost generator outside the DS pack never participated and has nothing to remove, but its
        // participating siblings still need their own P — leaving them on the aggregate entry is what
        // keeps their distributedSlackGeneratorMove at zero and sends the curve channel back to the share model.
        bool ovPerGen = ov >= 0;
        for (int g = 0; g < nDistributedSlackGenerators; g++) {                     // fresh list, fresh cumulative deltas
            distributedSlackGeneratorMove[(size_t) s * nDistributedSlackGenerators + g] = 0.0;
            bool lost = false;
            for (int k = 0; k < nDistributedSlackLostGenerators; k++) {
                if (distributedSlackLostGenerators[2 * k] == s && distributedSlackLostGenerators[2 * k + 1] == g) {
                    lost = true;
                    break;
                }
            }
            // A GeneratorAction can also take a generator out of the participation (OLF's
            // reApplyActivePowerControlChecks: its new target is outside its limits): it stays where the action
            // put it and takes no share, like a lost one, but it is still there for everything else.
            for (int k = 0; k < nDistributedSlackRemovedGenerators && !lost; k++) {
                lost = distributedSlackRemovedGenerators[2 * k] == s && distributedSlackRemovedGenerators[2 * k + 1] == g;
            }
            distributedSlackGeneratorRemoved[(size_t) s * nDistributedSlackGenerators + g] = lost ? 1 : 0;
            double a = distributedSlackGeneratorAnchor != nullptr ? distributedSlackGeneratorAnchor[(size_t) s * nDistributedSlackGenerators + g] : 0.0;   // a moved anchor eats headroom
            distributedSlackGeneratorWorkingFactor[(size_t) s * nDistributedSlackGenerators + g] = lost ? 0.0
                    : (distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN ? fmax(0.0, dirUp ? distributedSlackGeneratorMarginUp[g] - a : distributedSlackGeneratorMarginDown[g] + a) : distributedSlackGeneratorFactor[g]);
        }
        for (int iter = 0; iter < nDistributedSlackGenerators && fabs(rem) > residueEps; iter++) {
            double fsumActive = 0.0;                         // normalizeParticipationFactors, over the remaining
            for (int g = 0; g < nDistributedSlackGenerators; g++) {
                int i = distributedSlackGeneratorBus[g];                            // -1: a slack generator, never islanded nor outaged here
                if (distributedSlackGeneratorRemoved[(size_t) s * nDistributedSlackGenerators + g] || (i >= 0 && ((ov == i && !ovPerGen)
                        || (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + i])))) {
                    continue;
                }
                fsumActive += distributedSlackGeneratorWorkingFactor[(size_t) s * nDistributedSlackGenerators + g];
            }
            // The bus that lost a generator stays on the AGGREGATE treatment: the override channel
            // carries {factor, base, headroom, footroom} deltas for the bus, not the identity of the
            // generator, so a per-generator entry cannot express it. Its share is added to the same
            // denominator, so the normalization is the one the old loop used.
            double ovF = 0.0;
            double ovBase = 0.0;
            double ovMaxD = 0.0;
            double ovMinD = 0.0;
            bool ovActive = false;
            if (ov >= 0 && !ovPerGen && (distributedSlackIslanded == nullptr || !distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + ov])) {
                ovBase = distributedSlackBusBaseTargetP[ov] - distributedSlackOverrideBaseTargetPDelta[s];
                ovMaxD = dsMaxDelta[ov] - distributedSlackOverrideMaxDelta[s];
                ovMinD = dsMinDelta[ov] - distributedSlackOverrideMinDelta[s];
                ovF = distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN ? fmax(0.0, dirUp ? distributedSlackBusMarginUp[ov] : distributedSlackBusMarginDown[ov])
                                  : distributedSlackBusFactor[ov] - distributedSlackOverrideFactorDelta[s];
                double ovCum = target[(size_t) s * n + distributedSlackBusRowOf(distributedSlackBusPhiRow, distributedSlackBusRowOverride, s, nDistributedSlackBuses, ov)] - ovBase;
                ovActive = ovF > 0 && !((rem > 0) ? (ovCum >= ovMaxD) : (ovCum <= ovMinD));
                if (ovActive) {
                    fsumActive += ovF;
                }
            }
            if (fsumActive <= 0.0) {
                break;                                       // every participant removed — residual stays at the slack
            }
            double done = 0.0;
            for (int g = 0; g < nDistributedSlackGenerators; g++) {
                int i = distributedSlackGeneratorBus[g];                            // -1: a slack generator, never islanded nor outaged here
                if (distributedSlackGeneratorRemoved[(size_t) s * nDistributedSlackGenerators + g] || (i >= 0 && ((ov == i && !ovPerGen)
                        || (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + i])))) {
                    continue;
                }
                // normalizeParticipationFactors: divide the STORED factor, then use it.
                double f = distributedSlackGeneratorWorkingFactor[(size_t) s * nDistributedSlackGenerators + g] / fsumActive;
                distributedSlackGeneratorWorkingFactor[(size_t) s * nDistributedSlackGenerators + g] = f;
                double cum = distributedSlackGeneratorMove[(size_t) s * nDistributedSlackGenerators + g];
                double newCum = cum + rem * f;
                // its limits seen from its own anchor (the base target unless a strategy pair moved it)
                double a = distributedSlackGeneratorAnchor != nullptr ? distributedSlackGeneratorAnchor[(size_t) s * nDistributedSlackGenerators + g] : 0.0;
                double maxD = distributedSlackGeneratorMaxDelta[g] - a;
                double minD = distributedSlackGeneratorMinDelta[g] - a;
                // OLF removes an element only on a STRICT overshoot: one that lands exactly on its
                // limit stays in the list another iteration, where it takes weight and moves nothing.
                if (rem > 0 && newCum > maxD) {
                    newCum = maxD;
                    distributedSlackGeneratorRemoved[(size_t) s * nDistributedSlackGenerators + g] = 1;
                } else if (rem < 0 && newCum < minD) {
                    newCum = minD;
                    distributedSlackGeneratorRemoved[(size_t) s * nDistributedSlackGenerators + g] = 1;
                }
                if (newCum != cum) {
                    distributedSlackGeneratorMove[(size_t) s * nDistributedSlackGenerators + g] = newCum;
                    done += newCum - cum;
                    changed = true;
                }
            }
            if (ovActive) {
                size_t idx = (size_t) s * n + distributedSlackBusRowOf(distributedSlackBusPhiRow, distributedSlackBusRowOverride, s, nDistributedSlackBuses, ov);
                double cum = target[idx] - ovBase;
                double newCum = fmin(fmax(cum + rem * (ovF / fsumActive), ovMinD), ovMaxD);
                if (newCum != cum) {
                    target[idx] = ovBase + newCum;
                    done += newCum - cum;
                    changed = true;
                }
            }
            // The bus rows: anchor + Σ over the bus's generators, whose slice is contiguous.
            for (int i = 0; i < nDistributedSlackBuses; i++) {
                if ((ov == i && !ovPerGen) || (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + i])) {
                    continue;                                // ov keeps its own value; a frozen row is not written
                }
                double sum = 0.0;
                for (int g = distributedSlackBusGeneratorStart[i]; g < distributedSlackBusGeneratorEnd[i]; g++) {
                    sum += distributedSlackGeneratorMove[(size_t) s * nDistributedSlackGenerators + g];
                }
                // The outage's own move of the row target is not slack power: it shifts the anchor, and so
                // does a strategy pair's parent distribution (the bus's share of distributedSlackGeneratorAnchor).
                target[(size_t) s * n + distributedSlackBusRowOf(distributedSlackBusPhiRow, distributedSlackBusRowOverride, s, nDistributedSlackBuses, i)] =
                        distributedSlackBusBaseTargetP[i] + (distributedSlackBusAnchorShift != nullptr ? distributedSlackBusAnchorShift[(size_t) s * nDistributedSlackBuses + i] : 0.0)
                        - (ov == i ? distributedSlackOverrideBaseTargetPDelta[s] : 0.0) + sum;
            }
            rem -= done;
        }
        if (slackInPack && retainApplied != nullptr) {
            double placed = 0.0;                             // the slack's share of THIS distribution (applied to the network)
            for (int g = 0; g < nDistributedSlackGenerators; g++) {
                if (distributedSlackGeneratorBus[g] < 0) {
                    placed += distributedSlackGeneratorMove[(size_t) s * nDistributedSlackGenerators + g];
                }
            }
            retainApplied[s] = placed;
        }
    } else {
    for (int iter = 0; iter < nDistributedSlackBuses && fabs(rem) > residueEps; iter++) {
        double fsumActive = 0.0;                            // re-normalize factors over the still-unsaturated buses
        for (int i = 0; i < nDistributedSlackBuses; i++) {
            if (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + i]) {
                continue;
            }
            double f = distributedSlackBusFactor[i], base = distributedSlackBusBaseTargetP[i], maxD = dsMaxDelta[i], minD = dsMinDelta[i];
            double sh = distributedSlackBusAnchorShift != nullptr ? distributedSlackBusAnchorShift[(size_t) s * nDistributedSlackBuses + i] : 0.0;
            if (distributedSlackBusAnchorShift != nullptr) { base += sh; maxD -= sh; minD -= sh; }
            if (distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN) { f = fmax(0.0, (dirUp ? distributedSlackBusMarginUp[i] : distributedSlackBusMarginDown[i]) - (dirUp ? sh : -sh)); }
            if (ov == i) { f -= distributedSlackOverrideFactorDelta[s]; base -= distributedSlackOverrideBaseTargetPDelta[s]; maxD -= distributedSlackOverrideMaxDelta[s]; minD -= distributedSlackOverrideMinDelta[s]; }
            double cum = target[(size_t) s * n + distributedSlackBusRowOf(distributedSlackBusPhiRow, distributedSlackBusRowOverride, s, nDistributedSlackBuses, i)] - base;
            bool sat = (rem > 0) ? (cum >= maxD) : (cum <= minD);
            if (!sat) {
                fsumActive += f;
            }
        }
        if (fsumActive <= 0.0) {
            break;                                          // every participating bus saturated — residual stays at the slack
        }
        double done = 0.0;
        for (int i = 0; i < nDistributedSlackBuses; i++) {
            if (distributedSlackIslanded != nullptr && distributedSlackIslanded[(size_t) s * nDistributedSlackBuses + i]) {
                continue;                                    // writing here would land on a frozen row
            }
            double f = distributedSlackBusFactor[i], base = distributedSlackBusBaseTargetP[i], maxD = dsMaxDelta[i], minD = dsMinDelta[i];
            double sh = distributedSlackBusAnchorShift != nullptr ? distributedSlackBusAnchorShift[(size_t) s * nDistributedSlackBuses + i] : 0.0;
            if (distributedSlackBusAnchorShift != nullptr) { base += sh; maxD -= sh; minD -= sh; }
            if (distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN) { f = fmax(0.0, (dirUp ? distributedSlackBusMarginUp[i] : distributedSlackBusMarginDown[i]) - (dirUp ? sh : -sh)); }
            if (ov == i) { f -= distributedSlackOverrideFactorDelta[s]; base -= distributedSlackOverrideBaseTargetPDelta[s]; maxD -= distributedSlackOverrideMaxDelta[s]; minD -= distributedSlackOverrideMinDelta[s]; }
            size_t idx = (size_t) s * n + distributedSlackBusRowOf(distributedSlackBusPhiRow, distributedSlackBusRowOverride, s, nDistributedSlackBuses, i);
            double cum = target[idx] - base;
            bool sat = (rem > 0) ? (cum >= maxD) : (cum <= minD);
            if (sat) {
                continue;
            }
            double newCum = fmin(fmax(cum + rem * (f / fsumActive), minD), maxD);
            if (newCum != cum) {
                target[idx] = base + newCum;
                done += newCum - cum;
                changed = true;
            }
        }
        rem -= done;
    }
    }
    if (changed) {
        switched[s] = 1;
    }
    // OLF decides the distribution FAILED on ActivePowerDistribution.P_RESIDUE_EPS (1e-5 pu, 1 kW),
    // not on slackBusPMaxMismatch: handleDistributionFailureBehavior is entered whenever the water-fill
    // left more than P_RESIDUE_EPS unplaced. Flagging on the (much looser) entry band instead let a
    // saturated scenario with, say, 0.3 MW still unplaced come back CONVERGED where the CPU reports
    // FAILED. The flag only routes the scenario to the CPU fallback, so the tighter test can cost
    // batching share but can never make the device answer where OLF refuses to.
    if (fabs(rem) > residueEps) {
        slackDistributionFailure[s] = 1;                                   // saturated: residual stays at the slack — the host
    }                                                        // decides per the run's slackDistributionFailureBehavior
}

// One check of DistributedSlackOuterLoop on every live scenario (GpuOuterLoopKind::DISTRIBUTED_SLACK): slack-only, or the participants
// sharing the mismatch by water-filling.
void distributedSlackCheck(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const double& SLACK_DIST_TOL = drv.slackDistTol;
    const double& P_RESIDUE_EPS = drv.pResidueEps;
    const bool& hasDistributedSlackOverride = drv.hasDistributedSlackOverride;
    const bool& hasDistributedSlackBusAnchorShift = drv.hasDistributedSlackBusAnchorShift;
    const bool& hasSlackBusAnchorShift = drv.hasSlackBusAnchorShift;
    DeviceBuf& dDistributedSlackOverrideBus = drv.dDistributedSlackOverrideBus;
    DeviceBuf& dDistributedSlackOverrideFactorDelta = drv.dDistributedSlackOverrideFactorDelta;
    DeviceBuf& dDistributedSlackOverrideBaseTargetPDelta = drv.dDistributedSlackOverrideBaseTargetPDelta;
    DeviceBuf& dDistributedSlackOverrideMaxDelta = drv.dDistributedSlackOverrideMaxDelta;
    DeviceBuf& dDistributedSlackOverrideMinDelta = drv.dDistributedSlackOverrideMinDelta;
    DeviceBuf& dDistributedSlackBusAnchorShift = drv.dDistributedSlackBusAnchorShift;
    DeviceBuf& dSlackBusAnchorShift = drv.dSlackBusAnchorShift;
    if (ctx->slackOnlyDistribution) {
        if (!ctx->dSlackDistributionFailure.p) ctx->dSlackDistributionFailure.alloc(nS * sizeof(char));
        CUDA_CHECK(cudaMemsetAsync(ctx->dSlackDistributionFailure.p, 0, nS * sizeof(char), ctx->stream));
        slackOnlyDistributedSlackCheck<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, ctx->slackBusPhiRow,
                ctx->hasSlackBusRowOverride ? ctx->dSlackBusRowOverride.as<int>() : nullptr, ctx->slackBusTargetP,
                SLACK_DIST_TOL, P_RESIDUE_EPS, ctx->dQbuf.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
                ctx->hasSlackOnlyAnchor ? ctx->dSlackOnlyAnchor.as<double>() : nullptr,
                ctx->slackBusRoomUp, ctx->slackBusRoomDown,
                ctx->dSlackOnlyAbsorbed.as<double>(), ctx->dOuterLoopUnstable.as<char>(), ctx->dSlackDistributionFailure.as<char>());
    }
    if (ctx->nDistributedSlackBuses > 0) {
        // Recompute the slack's retained share from the CURRENT total imbalance every pass (NOT frozen at
        // pass 0) so a participating slack tracks the loss change during redistribution.
        computeSlackRetain<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, ctx->nDistributedSlackBuses, ctx->slackBusPhiRow,
                ctx->hasSlackBusRowOverride ? ctx->dSlackBusRowOverride.as<int>() : nullptr, ctx->slackBusTargetP,
                1.0 - ctx->distributedSlackFactorSum, ctx->dDistributedSlackBusPhiRow.as<int>(),
                ctx->nDistributedSlackBusRowOverrides > 0 ? ctx->dDistributedSlackBusRowOverride.as<int>() : nullptr, ctx->dDistributedSlackBusBaseTargetP.as<double>(),
                ctx->dTarget.as<double>(), ctx->dQbuf.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
                ctx->dSlackRetainedShare.as<double>(),
                hasDistributedSlackOverride ? dDistributedSlackOverrideBus.as<int>() : nullptr, dDistributedSlackOverrideBaseTargetPDelta.as<double>(),
                hasDistributedSlackBusAnchorShift ? dDistributedSlackBusAnchorShift.as<double>() : nullptr,
                hasSlackBusAnchorShift ? dSlackBusAnchorShift.as<double>() : nullptr,
                ctx->distributedSlackFactorMode, ctx->distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN ? ctx->dDistributedSlackBusMarginUp.as<double>() : nullptr,
                ctx->distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN ? ctx->dDistributedSlackBusMarginDown.as<double>() : nullptr,
                ctx->slackBusMarginUp, ctx->slackBusMarginDown,
                ctx->nDistributedSlackIslanded > 0 ? ctx->dDistributedSlackIslanded.as<char>() : nullptr,
                ctx->dDistributedSlackBusFactor.as<double>(), hasDistributedSlackOverride ? dDistributedSlackOverrideFactorDelta.as<double>() : nullptr,
                ctx->nDistributedSlackGenerators, ctx->nDistributedSlackGenerators > 0 ? ctx->distributedSlackGeneratorBus.as<int>() : nullptr,
                ctx->hasDistributedSlackGeneratorInitialMove ? ctx->dDistributedSlackGeneratorInitialMove.as<double>() : nullptr, ctx->dSlackRetainedShareApplied.as<double>());
        if (!ctx->dSlackDistributionFailure.p) ctx->dSlackDistributionFailure.alloc(nS * sizeof(char));
        CUDA_CHECK(cudaMemsetAsync(ctx->dSlackDistributionFailure.p, 0, nS * sizeof(char), ctx->stream));
        distributeSlackBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nDistributedSlackBuses, nS, n, SLACK_DIST_TOL, P_RESIDUE_EPS,
                ctx->slackBusPhiRow, ctx->hasSlackBusRowOverride ? ctx->dSlackBusRowOverride.as<int>() : nullptr, ctx->slackBusTargetP, ctx->dDistributedSlackBusPhiRow.as<int>(),
                ctx->nDistributedSlackBusRowOverrides > 0 ? ctx->dDistributedSlackBusRowOverride.as<int>() : nullptr,
                ctx->dDistributedSlackBusFactor.as<double>(), ctx->dDistributedSlackBusBaseTargetP.as<double>(),
                hasDistributedSlackBusAnchorShift ? dDistributedSlackBusAnchorShift.as<double>() : nullptr,
                hasSlackBusAnchorShift ? dSlackBusAnchorShift.as<double>() : nullptr,
                ctx->dDistributedSlackBusMaxDelta.as<double>(), ctx->dDistributedSlackBusMinDelta.as<double>(), ctx->dSlackRetainedShare.as<double>(),
                ctx->dQbuf.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
                hasDistributedSlackOverride ? dDistributedSlackOverrideBus.as<int>() : nullptr, dDistributedSlackOverrideFactorDelta.as<double>(),
                dDistributedSlackOverrideBaseTargetPDelta.as<double>(), dDistributedSlackOverrideMaxDelta.as<double>(), dDistributedSlackOverrideMinDelta.as<double>(),
                ctx->distributedSlackFactorMode, ctx->distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN ? ctx->dDistributedSlackBusMarginUp.as<double>() : nullptr,
                ctx->distributedSlackFactorMode == GpuDistributedSlackFactorMode::REMAINING_MARGIN ? ctx->dDistributedSlackBusMarginDown.as<double>() : nullptr,
                ctx->slackBusMarginUp, ctx->slackBusMarginDown,
                ctx->nDistributedSlackIslanded > 0 ? ctx->dDistributedSlackIslanded.as<char>() : nullptr,
                ctx->nDistributedSlackGenerators, ctx->distributedSlackGeneratorBus.as<int>(), ctx->distributedSlackGeneratorFactor.as<double>(),
                ctx->distributedSlackGeneratorMaxDelta.as<double>(), ctx->distributedSlackGeneratorMinDelta.as<double>(),
                ctx->distributedSlackGeneratorMarginUp.as<double>(), ctx->distributedSlackGeneratorMarginDown.as<double>(),
                ctx->dDistributedSlackBusGeneratorStart.as<int>(), ctx->dDistributedSlackBusGeneratorEnd.as<int>(),
                ctx->dDistributedSlackGeneratorMove.as<double>(), ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMoveValid.as<char>() : nullptr, ctx->hasDistributedSlackGeneratorAnchor ? ctx->dDistributedSlackGeneratorAnchor.as<double>() : nullptr, ctx->hasDistributedSlackGeneratorInitialMove ? ctx->dDistributedSlackGeneratorInitialMove.as<double>() : nullptr,
                ctx->dDistributedSlackGeneratorRemoved.as<char>(), ctx->dDistributedSlackGeneratorWorkingFactor.as<double>(),
                ctx->nDistributedSlackLostGenerators, ctx->nDistributedSlackLostGenerators > 0 ? ctx->dDistributedSlackLostGenerators.as<int>() : nullptr,
                ctx->nDistributedSlackRemovedGenerators, ctx->nDistributedSlackRemovedGenerators > 0 ? ctx->dDistributedSlackRemovedGenerators.as<int>() : nullptr,
                ctx->dTarget.as<double>(), ctx->dOuterLoopUnstable.as<char>(), ctx->dSlackDistributionFailure.as<char>(),
                ctx->dSlackRetainedShareApplied.as<double>(),
                ctx->dDistributedSlackLastPassAccounting.p ? ctx->dDistributedSlackLastPassAccounting.as<double>() : nullptr);
        if (ctx->nForcedTargetQ > 0) {                              // forced target Q follows the generators' new P
            forcedTargetQBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nForcedTargetQ, nS, n, ctx->nDistributedSlackBuses, ctx->nDistributedSlackGenerators,
                    ctx->forcedTargetQExtrapolate, ctx->dForcedTargetQIndices.as<int>(), ctx->dForcedTargetQParameters.as<double>(), ctx->dForcedTargetQPoints.as<double>(),
                    ctx->hasForcedTargetQPreDistribution ? ctx->dForcedTargetQPreDistribution.as<double>() : nullptr,
                    ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMove.as<double>() : nullptr,
                    ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMoveValid.as<char>() : nullptr, ctx->hasDistributedSlackGeneratorAnchor ? ctx->dDistributedSlackGeneratorAnchor.as<double>() : nullptr, ctx->hasDistributedSlackGeneratorInitialMove ? ctx->dDistributedSlackGeneratorInitialMove.as<double>() : nullptr,
                    ctx->dDistributedSlackBusPhiRow.as<int>(), ctx->nDistributedSlackBusRowOverrides > 0 ? ctx->dDistributedSlackBusRowOverride.as<int>() : nullptr,
                    ctx->nDistributedSlackLostGenerators, ctx->nDistributedSlackLostGenerators > 0 ? ctx->dDistributedSlackLostGenerators.as<int>() : nullptr,
                    ctx->nDistributedSlackIslanded > 0 ? ctx->dDistributedSlackIslanded.as<char>() : nullptr, ctx->dMode.as<int>(),
                    ctx->dOuterLoopCheckActive.as<char>(), ctx->dTarget.as<double>(), ctx->dForcedTargetQApplied.as<double>());
        }
    }
}

// Per-solve reset: the slack-retained share is 0 until frozen at the first pass, and the forced target Q is
// clamped at each scenario's start active power.
void distributedSlackResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    CUDA_CHECK(cudaMemsetAsync(ctx->dSlackRetainedShare.p, 0, nS * sizeof(double), ctx->stream));
    {
        std::vector<double> nanv(nS, std::numeric_limits<double>::quiet_NaN());
        CUDA_CHECK(cudaMemcpyAsync(ctx->dSlackRetainedShareApplied.p, nanv.data(), nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }
    if (ctx->nForcedTargetQ > 0) {                              // forced target Q: the clamp at each scenario's start P
        forcedTargetQInitBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nForcedTargetQ, nS, n, ctx->nDistributedSlackBuses, ctx->forcedTargetQExtrapolate,
                ctx->dForcedTargetQIndices.as<int>(), ctx->dForcedTargetQParameters.as<double>(), ctx->dForcedTargetQPoints.as<double>(),
                ctx->hasForcedTargetQPreDistribution ? ctx->dForcedTargetQPreDistribution.as<double>() : nullptr,
                ctx->nDistributedSlackLostGenerators, ctx->nDistributedSlackLostGenerators > 0 ? ctx->dDistributedSlackLostGenerators.as<int>() : nullptr,
                ctx->nDistributedSlackIslanded > 0 ? ctx->dDistributedSlackIslanded.as<char>() : nullptr, ctx->dMode.as<int>(),
                ctx->dTarget.as<double>(), ctx->dForcedTargetQApplied.as<double>());
    }
}

// Per-solve reset of the slack-only distribution: nothing absorbed yet, or a strategy pair's parent total.
void slackOnlyDistributionResetSolve(BatchSolverContext* ctx, DriverPass& drv, const double* slackBusAnchorShift) {
    const int& nS = drv.nS;
    if (slackBusAnchorShift != nullptr) {                   // strategy pairs: seed with the parent contingency's
        CUDA_CHECK(cudaMemcpyAsync(ctx->dSlackOnlyAbsorbed.p, slackBusAnchorShift,   // absorbed total (the device
                nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream)); // analogue of the CPU re-anchor)
    } else {
        CUDA_CHECK(cudaMemsetAsync(ctx->dSlackOnlyAbsorbed.p, 0, nS * sizeof(double), ctx->stream));
    }
}

}  // namespace olfgpu
