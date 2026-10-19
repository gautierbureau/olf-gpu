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
#include "hvdc.cuh"
#include "forced_target_q.cuh"
#include "slack_participants.cuh"
#include "distributed_slack.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

// Whether the slack bus's own participating generators are in the per-generator pack (dgBus -1, no row).
__host__ __device__ inline bool slackGeneratorsInPack(int ngen, const int* dgBus) {
    for (int g = 0; g < ngen; g++) {
        if (dgBus[g] < 0) {
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
                                      const int* hvBus1Row, const int* hvBus2Row, const int* hvBus1Phi,
                                      const int* hvBus2Phi, const int* hvElem, const double* hvIn,
                                      const int* status, const double* d1, const double* d2, const char* ctgOut,
                                      const char* disabled, const double* x, const char* rlActive, double* qBuf) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS || !rlActive[s]) return;
    int slackRow = slackRowOv != nullptr && slackRowOv[s] >= 0 ? slackRowOv[s] : slackRowBase;
    const double* xs = x + (size_t) s * n;
    double corr = 0.0;
    for (int h = 0; h < nh; h++) {
        int r1 = hvBus1Row[h] < 0 ? ~hvBus1Row[h] : hvBus1Row[h];
        int r2 = hvBus2Row[h] < 0 ? ~hvBus2Row[h] : hvBus2Row[h];
        if (r1 != slackRow && r2 != slackRow) continue;
        size_t sh = (size_t) s * nh + h;
        const double* p = hvIn + (size_t) h * 9;
        bool live = status[sh] == HV_LINEAR && !ctgOut[sh] && !disabled[(size_t) s * nElements + hvElem[h]];
        double ph1 = xs[hvBus1Phi[h]];
        double ph2 = xs[hvBus2Phi[h]];
        if (r1 == slackRow) {
            corr += (live ? hvdc::p1(p[0], p[1], p[4], p[5], p[6], ph1, ph2) : 0.0) - d1[sh];
        }
        if (r2 == slackRow) {
            corr += (live ? hvdc::p2(p[0], p[1], p[4], p[5], p[6], ph1, ph2) : 0.0) - d2[sh];
        }
    }
    qBuf[(size_t) s * n + slackRow] += corr;
}

// OLF_GPU_DS_DEBUG: the kernel-level water-fill prints ([DS-GATE]/[DS-WF]/[DS-SAT]) follow ONE slot,
// OLF_GPU_DS_DEBUG_SLOT (default 0); returns slot+1, 0 = off.
static int dsDebugTraceSlot() {
    if (std::getenv("OLF_GPU_DS_DEBUG") == nullptr) return 0;
    const char* e = std::getenv("OLF_GPU_DS_DEBUG_SLOT");
    return (e != nullptr ? std::atoi(e) : 0) + 1;
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
__global__ void slackOnlyDsCheck(int nScenarios, int n, int slackPhiRowBase, const int* slackRowOv, double slackTargetP,
                                 double slackDistTol, double residueEps, const double* qBuf, const char* rlActive,
                                 const double* anchor, double roomUp, double roomDown,
                                 double* absorbed, char* switched, char* residual) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !rlActive[s]) return;
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

__global__ void computeSlackRetain(int nScenarios, int n, int nds, int slackPhiRowBase, const int* slackRowOv, double slackTargetP,
                                   double oneMinusFactorSum, const int* dsPhiRow, const int* dsRowOv,
                                   const double* dsBaseTargetP,
                                   const double* target, const double* qBuf, const char* rlActive,
                                   double* retain,
                                   const int* dsOvBus, const double* dsOvBaseDelta,
                                   const double* dsAnchorShift, const double* dsSlackShift,
                                   int dsMode, const double* dsMarginUp, const double* dsMarginDown,
                                   double slackMarginUp, double slackMarginDown,
                                   const char* dsIslanded, const double* dsFactor, const double* dsOvFactorDelta,
                                   int ngen, const int* dgBus, const double* dgStart, const double* retainApplied) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!rlActive[s]) return;
    if (slackGeneratorsInPack(ngen, dgBus)) {
        // The slack's generators are in the water-fill list: its share is what the last distribution gave them
        // (clamped like anyone's), not a fixed fraction of the total. Before any pass in this solve that is the
        // loss pre-distribution's move of them (a pair starts at its parent's, which its slack target holds).
        double placed = 0.0;
        if (retainApplied != nullptr && !isnan(retainApplied[s])) {
            placed = retainApplied[s];
        } else if (dgStart != nullptr) {
            for (int g = 0; g < ngen; g++) {
                if (dgBus[g] < 0) {
                    placed += dgStart[(size_t) s * ngen + g];
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
    int ov = dsOvBus != nullptr ? dsOvBus[s] : -1;
    for (int i = 0; i < nds; i++) {
        fsAll += dsFactor[i];
        if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
            continue;                                        // in another component: not a participant here
        }
        // a lost participating generator leaves the participation like a stranded one: its factor goes
        fsActive += dsFactor[i] - (ov == i && dsOvFactorDelta != nullptr ? dsOvFactorDelta[s] : 0.0);
        // A composed injection override on a participant bus (redispatch action, load delta) shifts
        // the row target WITHOUT being distributed power — the effective base must shift with it,
        // or the retain math counts the override as absorbed slack and the waterfill CANCELS it
        // (measured: a +60 MW redispatch read back as a no-op, GeneratorAction drill 2026-07-19).
        double base = dsBaseTargetP[i];
        if (dsAnchorShift != nullptr) {
            base += dsAnchorShift[(size_t) s * nds + i];     // pair leg: cumulative reference re-anchored at the
        }                                                    // PARENT contingency's distributed targets (OLF's
        if (ov == i) {                                       // setGeneratorsInitialTargetPToTargetP before the
            base -= dsOvBaseDelta[s];                        // strategy re-solve)
        }
        absorbed += target[(size_t) s * n + dsRowOf(dsPhiRow, dsRowOv, s, nds, i)] - base;
    }
    double slackTgt = slackTargetP + (dsSlackShift != nullptr ? dsSlackShift[s] : 0.0);
    double tot = (qBuf[(size_t) s * n + slackPhiRow] - slackTgt) + absorbed;   // total-from-anchor, OLF's
    double rho = oneMinusFactorSum;                                            // remainingMismatch at run() start
    if (fsActive != fsAll && dsMode != 1) {
        // The factors were normalized over ALL participants (W), so rho = w_slack/W. Dropping the
        // stranded or lost ones renormalizes over W' = W - Σ_dropped w, and w_slack/W' = rho/(Σ_active f + rho).
        // Leaving rho global is exactly what made the first attempt at this fail: the participants
        // were masked out of the distribution loops while the slack kept the share it had when they
        // were still in, so the pass retired with most of the mismatch undistributed.
        rho = fsActive + rho > 0 ? rho / (fsActive + rho) : 0.0;
    }
    if (dsMode == 1) {
        // REMAINING_MARGIN: the slack's retained share is its own margin share in the DIRECTION of the
        // total-from-anchor (OLF evaluates the direction-dependent factors once per run at the anchor).
        // Margins shift with a moved anchor: +delta of anchor target eats headroom, frees footroom.
        double slackM = tot > 0 ? slackMarginUp : slackMarginDown;
        if (dsSlackShift != nullptr) {
            slackM = fmax(0.0, slackM - (tot > 0 ? dsSlackShift[s] : -dsSlackShift[s]));
        }
        double sum = slackM;
        for (int i = 0; i < nds; i++) {
            if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
                continue;                                    // REMAINING_MARGIN needs no rescale: dropping
            }                                                // the stranded margins IS the renormalization
            double m = tot > 0 ? dsMarginUp[i] : dsMarginDown[i];
            if (dsAnchorShift != nullptr) {
                double sh = dsAnchorShift[(size_t) s * nds + i];
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
__global__ void distributeSlackBatch(int nds, int nScenarios, int n, double slackDistTol, double residueEps,
                                              int slackPhiRowBase, const int* slackRowOv, double slackTargetP, const int* dsPhiRow, const int* dsRowOv,
                                              const double* dsFactor, const double* dsBaseTargetP, const double* dsAnchorShift, const double* dsSlackShift,
                                              const double* dsMaxDelta, const double* dsMinDelta, const double* dsRetain,
                                              const double* qBuf, const char* rlActive,
                                              const int* dsOvBus, const double* dsOvFactorDelta, const double* dsOvBaseDelta,
                                              const double* dsOvMaxDelta, const double* dsOvMinDelta,
                                              int dsMode, const double* dsMarginUp, const double* dsMarginDown,
                                              double slackMarginUp, double slackMarginDown,
                                              const char* dsIslanded,
                                              int ngen, const int* dgBus, const double* dgFactor,
                                              const double* dgMaxDelta, const double* dgMinDelta,
                                              const double* dgMarginUp, const double* dgMarginDown,
                                              const int* busGenStart, const int* busGenEnd,
                                              double* genCum, char* genCumValid, const double* dgAnchor, const double* dgStart, char* genRemoved, double* genFactorWork,
                                              int nDsLostGen, const int* dsLostGen, int nDsOutGen, const int* dsOutGen,
                                              double* target, char* switched, char* dsResidual, double* retainApplied, double* gateLast, int dbgTrace) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!rlActive[s]) return;
    int slackPhiRow = slackRowOv != nullptr && slackRowOv[s] >= 0 ? slackRowOv[s] : slackPhiRowBase;   // relocated slack
    // OLF's DistributedSlackOuterLoop.check distributes only when the slack mismatch exceeds BOTH
    // slackBusPMaxMismatch and ActivePowerDistribution.P_RESIDUE_EPS; the second only binds below 1 kW.
    double gateTol = fmax(slackDistTol, residueEps);
    // The participating buses share the mismatch in EXCESS of the slack's retained share (rem); only distribute
    // past the slackBusPMaxMismatch outer gate, exactly like OLF's DistributedSlackOuterLoop.check.
    double slackTgt = slackTargetP + (dsSlackShift != nullptr ? dsSlackShift[s] : 0.0);
    double rem = (qBuf[(size_t) s * n + slackPhiRow] - slackTgt) - dsRetain[s];
    int ov = (dsOvBus != nullptr) ? dsOvBus[s] : -1;        // the bus losing a participating gen (gen contingency), -1 = none
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
    if (dsMode == 1) {
        double sumM = 0.0;
        for (int i = 0; i < nds; i++) {
            if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
                continue;
            }
            double m = dirUp ? dsMarginUp[i] : dsMarginDown[i];
            if (dsAnchorShift != nullptr) {
                double sh = dsAnchorShift[(size_t) s * nds + i];
                m = fmax(0.0, m - (dirUp ? sh : -sh));
            }
            sumM += m;
        }
        double slackM = dirUp ? slackMarginUp : slackMarginDown;
        if (dsSlackShift != nullptr) {
            slackM = fmax(0.0, slackM - (dirUp ? dsSlackShift[s] : -dsSlackShift[s]));
        }
        fsGlobal = sumM + slackM > 0 ? sumM / (sumM + slackM) : 0.0;
        noCapacity = !(sumM + slackM > 0);
    } else {
        double fsActive = 0.0;
        double fsAll = 0.0;
        for (int i = 0; i < nds; i++) {
            fsAll += dsFactor[i];
            if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
                continue;
            }
            fsActive += dsFactor[i] - (ov == i && dsOvFactorDelta != nullptr ? dsOvFactorDelta[s] : 0.0);
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
    // The slack's own generators in the list: the water-fill places the WHOLE total, theirs included, and dsRetain
    // is what they hold now, so rem is OLF's slack mismatch itself and the gate needs no scaling.
    bool slackInPack = ngen > 0 && slackGeneratorsInPack(ngen, dgBus);
    if (slackInPack) {
        fsGlobal = 1.0;
        noCapacity = false;
    }
    if (dbgTrace && s == dbgTrace - 1) {
        printf("[DS-GATE] slackMismatch=%.9f retain=%.9f rem=%.9f fsGlobal=%.9f tol=%.9f slackTgt=%.9f fire=%d\n",
               qBuf[(size_t) s * n + slackPhiRow] - slackTgt, dsRetain[s], rem, fsGlobal, slackDistTol, slackTgt,
               (fsGlobal > 0.0 && fabs(rem) > fsGlobal * gateTol) ? 1 : 0);
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
        gateLast[8 * s + 5] = dsRetain[s];
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
            dsResidual[s] = 1;
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
        retainApplied[s] = dsRetain[s];                      // the slack's share of THIS distribution (applied to the network)
    }
    // OLF's ActivePowerDistribution.run RESETS every participant to its INITIAL target
    // (step.resetToInitialState) and adds back what that undid (previousStateInfo.previousMismatch()),
    // then re-derives the WHOLE distribution from the anchor. Accumulating each pass on top of the
    // previous one is not the same function once anyone saturates: the re-normalisation over the
    // still-unsaturated participants is taken against a different remaining amount. Ported here for
    // FIDELITY — it is not what closes the residual rte6515 gap (that is the reactive-limits loop),
    // but the incremental form was not what OLF does.
    double absorbedNow = 0.0;
    for (int i = 0; i < nds; i++) {
        if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
            continue;                                        // its row is frozen; it never absorbed anything
        }
        double base = dsBaseTargetP[i];
        if (dsAnchorShift != nullptr) { base += dsAnchorShift[(size_t) s * nds + i]; }
        if (ov == i) { base -= dsOvBaseDelta[s]; }
        size_t idx = (size_t) s * n + dsRowOf(dsPhiRow, dsRowOv, s, nds, i);
        absorbedNow += target[idx] - base;
        target[idx] = base;                                 // reset to the anchor
    }
    rem = fsGlobal * ((qBuf[(size_t) s * n + slackPhiRow] - slackTgt) + absorbedNow);
    if (gateLast != nullptr) {
        // Slot 7, the documented `placed`. This read 6*s+5 — the stride of the 6-slot record this
        // one replaced: for s=0 it overwrote `dsRetain` with the placed amount (so a FIRING pass
        // printed the wrong column and dsPlaced stayed 0), and for every s>=1 it scribbled into
        // ANOTHER scenario's slots (s=1 -> index 11 = scenario 1's slackTgt). Every LOOP_PASS number
        // read off a firing pass was therefore the wrong quantity.
        gateLast[8 * s + 7] = rem;                           // what this pass places over the participants
    }
    if (dbgTrace && s == dbgTrace - 1) {
        printf("[DS-WF] entry slackMismatch=%.9f absorbedNow=%.9f fsGlobal=%.9f rem=%.9f nds=%d ngen=%d anchorShift=%d\n",
               qBuf[(size_t) s * n + slackPhiRow] - slackTgt, absorbedNow, fsGlobal, rem, nds, ngen,
               dsAnchorShift != nullptr ? 1 : 0);
    }
    bool changed = false;
    // PER-GENERATOR water-fill, the shape of OLF's GenerationActivePowerDistributionStep.run: factors
    // renormalized over the elements still in the list at each iteration start, one pass using the
    // mismatch as of that start, each element clamped at its OWN limit and REMOVED when it strictly
    // overshoots. A bus's row is then the sum of its generators' cumulative deltas.
    //
    // Used whenever the pack carries generators and THIS scenario is not re-anchored. A pair leg's
    // dsAnchorShift is a BUS-level displacement (the parent contingency's absorption, carried so the
    // leg does not undo it) and nothing in the pack says how to split it across a bus's generators, so
    // a re-anchored scenario keeps the aggregate loop below — unchanged, and no less faithful than it
    // was. The test is on the VALUE, not on the pointer: an orchestrator re-solve passes the carry
    // arrays for every scenario, all zeros for a plain N-1, and taking that as "re-anchored" quietly
    // sent every post-orchestrator pass down the aggregate path (measured: the first two DS rounds of
    // LINE-3320-350 matched the CPU digit for digit, the third did not).
    bool anchored = false;
    if (dsAnchorShift != nullptr) {
        for (int i = 0; i < nds; i++) {
            if (dsAnchorShift[(size_t) s * nds + i] != 0.0) {
                anchored = true;
                break;
            }
        }
    }
    // A re-anchored scenario that carries its PER-GENERATOR anchors (dgAnchor: a strategy pair, each generator's
    // post-contingency target) takes the per-generator water-fill too, from those anchors.
    bool perGen = ngen > 0 && (!anchored || dgAnchor != nullptr);
    if (genCumValid != nullptr) {
        genCumValid[s] = perGen ? 1 : 0;
    }
    if (perGen) {
        // The outaged bus's SURVIVORS join the water-fill like any others, with the generators this
        // scenario removes marked out by identity below. The gate is the BUS, not the lost unit: a
        // lost generator outside the DS pack never participated and has nothing to remove, but its
        // participating siblings still need their own P — leaving them on the aggregate entry is what
        // keeps their genCum at zero and sends the curve channel back to the share model.
        bool ovPerGen = ov >= 0;
        for (int g = 0; g < ngen; g++) {                     // fresh list, fresh cumulative deltas
            genCum[(size_t) s * ngen + g] = 0.0;
            bool lost = false;
            for (int k = 0; k < nDsLostGen; k++) {
                if (dsLostGen[2 * k] == s && dsLostGen[2 * k + 1] == g) {
                    lost = true;
                    break;
                }
            }
            // A GeneratorAction can also take a generator out of the participation (OLF's
            // reApplyActivePowerControlChecks: its new target is outside its limits): it stays where the action
            // put it and takes no share, like a lost one, but it is still there for everything else.
            for (int k = 0; k < nDsOutGen && !lost; k++) {
                lost = dsOutGen[2 * k] == s && dsOutGen[2 * k + 1] == g;
            }
            genRemoved[(size_t) s * ngen + g] = lost ? 1 : 0;
            double a = dgAnchor != nullptr ? dgAnchor[(size_t) s * ngen + g] : 0.0;   // a moved anchor eats headroom
            genFactorWork[(size_t) s * ngen + g] = lost ? 0.0
                    : (dsMode == 1 ? fmax(0.0, dirUp ? dgMarginUp[g] - a : dgMarginDown[g] + a) : dgFactor[g]);
        }
        for (int iter = 0; iter < ngen && fabs(rem) > residueEps; iter++) {
            double fsumActive = 0.0;                         // normalizeParticipationFactors, over the remaining
            for (int g = 0; g < ngen; g++) {
                int i = dgBus[g];                            // -1: a slack generator, never islanded nor outaged here
                if (genRemoved[(size_t) s * ngen + g] || (i >= 0 && ((ov == i && !ovPerGen)
                        || (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i])))) {
                    continue;
                }
                fsumActive += genFactorWork[(size_t) s * ngen + g];
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
            if (ov >= 0 && !ovPerGen && (dsIslanded == nullptr || !dsIslanded[(size_t) s * nds + ov])) {
                ovBase = dsBaseTargetP[ov] - dsOvBaseDelta[s];
                ovMaxD = dsMaxDelta[ov] - dsOvMaxDelta[s];
                ovMinD = dsMinDelta[ov] - dsOvMinDelta[s];
                ovF = dsMode == 1 ? fmax(0.0, dirUp ? dsMarginUp[ov] : dsMarginDown[ov])
                                  : dsFactor[ov] - dsOvFactorDelta[s];
                double ovCum = target[(size_t) s * n + dsRowOf(dsPhiRow, dsRowOv, s, nds, ov)] - ovBase;
                ovActive = ovF > 0 && !((rem > 0) ? (ovCum >= ovMaxD) : (ovCum <= ovMinD));
                if (ovActive) {
                    fsumActive += ovF;
                }
            }
            if (fsumActive <= 0.0) {
                break;                                       // every participant removed — residual stays at the slack
            }
            double done = 0.0;
            for (int g = 0; g < ngen; g++) {
                int i = dgBus[g];                            // -1: a slack generator, never islanded nor outaged here
                if (genRemoved[(size_t) s * ngen + g] || (i >= 0 && ((ov == i && !ovPerGen)
                        || (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i])))) {
                    continue;
                }
                // normalizeParticipationFactors: divide the STORED factor, then use it.
                double f = genFactorWork[(size_t) s * ngen + g] / fsumActive;
                genFactorWork[(size_t) s * ngen + g] = f;
                double cum = genCum[(size_t) s * ngen + g];
                double newCum = cum + rem * f;
                // its limits seen from its own anchor (the base target unless a strategy pair moved it)
                double a = dgAnchor != nullptr ? dgAnchor[(size_t) s * ngen + g] : 0.0;
                double maxD = dgMaxDelta[g] - a;
                double minD = dgMinDelta[g] - a;
                // OLF removes an element only on a STRICT overshoot: one that lands exactly on its
                // limit stays in the list another iteration, where it takes weight and moves nothing.
                if (rem > 0 && newCum > maxD) {
                    newCum = maxD;
                    genRemoved[(size_t) s * ngen + g] = 1;
                } else if (rem < 0 && newCum < minD) {
                    newCum = minD;
                    genRemoved[(size_t) s * ngen + g] = 1;
                }
                if (newCum != cum) {
                    genCum[(size_t) s * ngen + g] = newCum;
                    done += newCum - cum;
                    changed = true;
                }
                if (dbgTrace && s == dbgTrace - 1 && genRemoved[(size_t) s * ngen + g]
                        && (newCum == maxD || newCum == minD)) {
                    printf("[DS-SAT] iter=%d gen=%d newCum=%.9f maxD=%.9f minD=%.9f\n",
                           iter, g, newCum, maxD, minD);
                }
            }
            if (ovActive) {
                size_t idx = (size_t) s * n + dsRowOf(dsPhiRow, dsRowOv, s, nds, ov);
                double cum = target[idx] - ovBase;
                double newCum = fmin(fmax(cum + rem * (ovF / fsumActive), ovMinD), ovMaxD);
                if (newCum != cum) {
                    target[idx] = ovBase + newCum;
                    done += newCum - cum;
                    changed = true;
                }
            }
            // The bus rows: anchor + Σ over the bus's generators, whose slice is contiguous.
            for (int i = 0; i < nds; i++) {
                if ((ov == i && !ovPerGen) || (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i])) {
                    continue;                                // ov keeps its own value; a frozen row is not written
                }
                double sum = 0.0;
                for (int g = busGenStart[i]; g < busGenEnd[i]; g++) {
                    sum += genCum[(size_t) s * ngen + g];
                }
                // The outage's own move of the row target is not slack power: it shifts the anchor, and so
                // does a strategy pair's parent distribution (the bus's share of dgAnchor).
                target[(size_t) s * n + dsRowOf(dsPhiRow, dsRowOv, s, nds, i)] =
                        dsBaseTargetP[i] + (dsAnchorShift != nullptr ? dsAnchorShift[(size_t) s * nds + i] : 0.0)
                        - (ov == i ? dsOvBaseDelta[s] : 0.0) + sum;
            }
            if (dbgTrace && s == dbgTrace - 1) {
                printf("[DS-WF] iter=%d fsumActive=%.9f rem=%.9f done=%.9f (per-generator)\n",
                       iter, fsumActive, rem, done);
            }
            rem -= done;
        }
        if (slackInPack && retainApplied != nullptr) {
            double placed = 0.0;                             // the slack's share of THIS distribution (applied to the network)
            for (int g = 0; g < ngen; g++) {
                if (dgBus[g] < 0) {
                    placed += genCum[(size_t) s * ngen + g];
                }
            }
            retainApplied[s] = placed;
        }
    } else {
    for (int iter = 0; iter < nds && fabs(rem) > residueEps; iter++) {
        double fsumActive = 0.0;                            // re-normalize factors over the still-unsaturated buses
        for (int i = 0; i < nds; i++) {
            if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
                continue;
            }
            double f = dsFactor[i], base = dsBaseTargetP[i], maxD = dsMaxDelta[i], minD = dsMinDelta[i];
            double sh = dsAnchorShift != nullptr ? dsAnchorShift[(size_t) s * nds + i] : 0.0;
            if (dsAnchorShift != nullptr) { base += sh; maxD -= sh; minD -= sh; }
            if (dsMode == 1) { f = fmax(0.0, (dirUp ? dsMarginUp[i] : dsMarginDown[i]) - (dirUp ? sh : -sh)); }
            if (ov == i) { f -= dsOvFactorDelta[s]; base -= dsOvBaseDelta[s]; maxD -= dsOvMaxDelta[s]; minD -= dsOvMinDelta[s]; }
            double cum = target[(size_t) s * n + dsRowOf(dsPhiRow, dsRowOv, s, nds, i)] - base;
            bool sat = (rem > 0) ? (cum >= maxD) : (cum <= minD);
            if (!sat) {
                fsumActive += f;
            }
        }
        if (fsumActive <= 0.0) {
            break;                                          // every participating bus saturated — residual stays at the slack
        }
        double done = 0.0;
        for (int i = 0; i < nds; i++) {
            if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
                continue;                                    // writing here would land on a frozen row
            }
            double f = dsFactor[i], base = dsBaseTargetP[i], maxD = dsMaxDelta[i], minD = dsMinDelta[i];
            double sh = dsAnchorShift != nullptr ? dsAnchorShift[(size_t) s * nds + i] : 0.0;
            if (dsAnchorShift != nullptr) { base += sh; maxD -= sh; minD -= sh; }
            if (dsMode == 1) { f = fmax(0.0, (dirUp ? dsMarginUp[i] : dsMarginDown[i]) - (dirUp ? sh : -sh)); }
            if (ov == i) { f -= dsOvFactorDelta[s]; base -= dsOvBaseDelta[s]; maxD -= dsOvMaxDelta[s]; minD -= dsOvMinDelta[s]; }
            size_t idx = (size_t) s * n + dsRowOf(dsPhiRow, dsRowOv, s, nds, i);
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
        if (dbgTrace && s == dbgTrace - 1) {
            printf("[DS-WF] iter=%d fsumActive=%.9f rem=%.9f done=%.9f\n", iter, fsumActive, rem, done);
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
        dsResidual[s] = 1;                                   // saturated: residual stays at the slack — the host
    }                                                        // decides per the run's slackDistributionFailureBehavior
}

// One check of DistributedSlackOuterLoop on every live scenario (LOOP_DS): slack-only, or the participants
// sharing the mismatch by water-filling.
void dsCheck(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const double& SLACK_DIST_TOL = drv.slackDistTol;
    const double& P_RESIDUE_EPS = drv.pResidueEps;
    const bool& hasDsOverride = drv.hasDsOverride;
    const bool& hasDsAnchor = drv.hasDsAnchor;
    const bool& hasDsSlackShift = drv.hasDsSlackShift;
    DeviceBuf& dDsOvBus = drv.dDsOvBus;
    DeviceBuf& dDsOvFactorDelta = drv.dDsOvFactorDelta;
    DeviceBuf& dDsOvBaseDelta = drv.dDsOvBaseDelta;
    DeviceBuf& dDsOvMaxDelta = drv.dDsOvMaxDelta;
    DeviceBuf& dDsOvMinDelta = drv.dDsOvMinDelta;
    DeviceBuf& dDsAnchorShift = drv.dDsAnchorShift;
    DeviceBuf& dDsSlackShift = drv.dDsSlackShift;
    if (ctx->dsSlackOnly) {
        if (!ctx->dDsResidual.p) ctx->dDsResidual.alloc(nS * sizeof(char));
        CUDA_CHECK(cudaMemsetAsync(ctx->dDsResidual.p, 0, nS * sizeof(char), ctx->stream));
        slackOnlyDsCheck<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, ctx->dsSlackPhiRow,
                ctx->hasDsSlackRowOv ? ctx->dDsSlackRowOv.as<int>() : nullptr, ctx->dsSlackTargetP,
                SLACK_DIST_TOL, P_RESIDUE_EPS, ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(),
                ctx->dsSlackHasAnchor ? ctx->dDsSlackAnchor.as<double>() : nullptr,
                ctx->dsSlackRoomUp, ctx->dsSlackRoomDown,
                ctx->dDsSlackAbsorbed.as<double>(), ctx->dRlSwitched.as<char>(), ctx->dDsResidual.as<char>());
    }
    if (ctx->nds > 0) {
        // Recompute the slack's retained share from the CURRENT total imbalance every pass (NOT frozen at
        // pass 0) so a participating slack tracks the loss change during redistribution.
        computeSlackRetain<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, ctx->nds, ctx->dsSlackPhiRow,
                ctx->hasDsSlackRowOv ? ctx->dDsSlackRowOv.as<int>() : nullptr, ctx->dsSlackTargetP,
                1.0 - ctx->dsFactorSum, ctx->dDsPhiRow.as<int>(),
                ctx->nDsRowOv > 0 ? ctx->dDsRowOv.as<int>() : nullptr, ctx->dDsBaseTargetP.as<double>(),
                ctx->dTarget.as<double>(), ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(),
                ctx->dDsRetain.as<double>(),
                hasDsOverride ? dDsOvBus.as<int>() : nullptr, dDsOvBaseDelta.as<double>(),
                hasDsAnchor ? dDsAnchorShift.as<double>() : nullptr,
                hasDsSlackShift ? dDsSlackShift.as<double>() : nullptr,
                ctx->dsMode, ctx->dsMode == 1 ? ctx->dDsMarginUp.as<double>() : nullptr,
                ctx->dsMode == 1 ? ctx->dDsMarginDown.as<double>() : nullptr,
                ctx->dsSlackMarginUp, ctx->dsSlackMarginDown,
                ctx->nDsIslanded > 0 ? ctx->dDsIslanded.as<char>() : nullptr,
                ctx->dDsFactor.as<double>(), hasDsOverride ? dDsOvFactorDelta.as<double>() : nullptr,
                ctx->ngen, ctx->ngen > 0 ? ctx->dgBus.as<int>() : nullptr,
                ctx->hasDgStart ? ctx->dDgStart.as<double>() : nullptr, ctx->dDsRetainApplied.as<double>());
        if (!ctx->dDsResidual.p) ctx->dDsResidual.alloc(nS * sizeof(char));
        CUDA_CHECK(cudaMemsetAsync(ctx->dDsResidual.p, 0, nS * sizeof(char), ctx->stream));
        distributeSlackBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nds, nS, n, SLACK_DIST_TOL, P_RESIDUE_EPS,
                ctx->dsSlackPhiRow, ctx->hasDsSlackRowOv ? ctx->dDsSlackRowOv.as<int>() : nullptr, ctx->dsSlackTargetP, ctx->dDsPhiRow.as<int>(),
                ctx->nDsRowOv > 0 ? ctx->dDsRowOv.as<int>() : nullptr,
                ctx->dDsFactor.as<double>(), ctx->dDsBaseTargetP.as<double>(),
                hasDsAnchor ? dDsAnchorShift.as<double>() : nullptr,
                hasDsSlackShift ? dDsSlackShift.as<double>() : nullptr,
                ctx->dDsMaxDelta.as<double>(), ctx->dDsMinDelta.as<double>(), ctx->dDsRetain.as<double>(),
                ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(),
                hasDsOverride ? dDsOvBus.as<int>() : nullptr, dDsOvFactorDelta.as<double>(),
                dDsOvBaseDelta.as<double>(), dDsOvMaxDelta.as<double>(), dDsOvMinDelta.as<double>(),
                ctx->dsMode, ctx->dsMode == 1 ? ctx->dDsMarginUp.as<double>() : nullptr,
                ctx->dsMode == 1 ? ctx->dDsMarginDown.as<double>() : nullptr,
                ctx->dsSlackMarginUp, ctx->dsSlackMarginDown,
                ctx->nDsIslanded > 0 ? ctx->dDsIslanded.as<char>() : nullptr,
                ctx->ngen, ctx->dgBus.as<int>(), ctx->dgFactor.as<double>(),
                ctx->dgMaxDelta.as<double>(), ctx->dgMinDelta.as<double>(),
                ctx->dgMarginUp.as<double>(), ctx->dgMarginDown.as<double>(),
                ctx->dBusGenStart.as<int>(), ctx->dBusGenEnd.as<int>(),
                ctx->dGenCum.as<double>(), ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->hasDgAnchor ? ctx->dDgAnchor.as<double>() : nullptr, ctx->hasDgStart ? ctx->dDgStart.as<double>() : nullptr,
                ctx->dGenRemoved.as<char>(), ctx->dGenFactorWork.as<double>(),
                ctx->nDsLostGen, ctx->nDsLostGen > 0 ? ctx->dDsLostGen.as<int>() : nullptr,
                ctx->nDsOutGen, ctx->nDsOutGen > 0 ? ctx->dDsOutGen.as<int>() : nullptr,
                ctx->dTarget.as<double>(), ctx->dRlSwitched.as<char>(), ctx->dDsResidual.as<char>(),
                ctx->dDsRetainApplied.as<double>(),
                ctx->dDsGateLast.p ? ctx->dDsGateLast.as<double>() : nullptr,
                dsDebugTraceSlot());
        if (ctx->nfq > 0) {                              // forced target Q follows the generators' new P
            forcedTargetQBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nfq, nS, n, ctx->nds, ctx->ngen,
                    ctx->fqExtrapolate, ctx->dFqIdx.as<int>(), ctx->dFqIn.as<double>(), ctx->dFqPts.as<double>(),
                    ctx->fqHasPre ? ctx->dFqPre.as<double>() : nullptr,
                    ctx->ngen > 0 ? ctx->dGenCum.as<double>() : nullptr,
                    ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->hasDgAnchor ? ctx->dDgAnchor.as<double>() : nullptr, ctx->hasDgStart ? ctx->dDgStart.as<double>() : nullptr,
                    ctx->dDsPhiRow.as<int>(), ctx->nDsRowOv > 0 ? ctx->dDsRowOv.as<int>() : nullptr,
                    ctx->nDsLostGen, ctx->nDsLostGen > 0 ? ctx->dDsLostGen.as<int>() : nullptr,
                    ctx->nDsIslanded > 0 ? ctx->dDsIslanded.as<char>() : nullptr, ctx->dMode.as<int>(),
                    ctx->dRlActive.as<char>(), ctx->dTarget.as<double>(), ctx->dFqApplied.as<double>());
        }
    }
}

// Per-solve reset: the slack-retained share is 0 until frozen at the first pass, and the forced target Q is
// clamped at each scenario's start active power.
void dsResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    CUDA_CHECK(cudaMemsetAsync(ctx->dDsRetain.p, 0, nS * sizeof(double), ctx->stream));
    {
        std::vector<double> nanv(nS, std::numeric_limits<double>::quiet_NaN());
        CUDA_CHECK(cudaMemcpyAsync(ctx->dDsRetainApplied.p, nanv.data(), nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }
    if (ctx->nfq > 0) {                              // forced target Q: the clamp at each scenario's start P
        forcedTargetQInitBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nfq, nS, n, ctx->nds, ctx->fqExtrapolate,
                ctx->dFqIdx.as<int>(), ctx->dFqIn.as<double>(), ctx->dFqPts.as<double>(),
                ctx->fqHasPre ? ctx->dFqPre.as<double>() : nullptr,
                ctx->nDsLostGen, ctx->nDsLostGen > 0 ? ctx->dDsLostGen.as<int>() : nullptr,
                ctx->nDsIslanded > 0 ? ctx->dDsIslanded.as<char>() : nullptr, ctx->dMode.as<int>(),
                ctx->dTarget.as<double>(), ctx->dFqApplied.as<double>());
    }
}

// Per-solve reset of the slack-only distribution: nothing absorbed yet, or a strategy pair's parent total.
void dsSlackOnlyResetSolve(BatchSolverContext* ctx, DriverPass& drv, const double* dsSlackShift) {
    const int& nS = drv.nS;
    if (dsSlackShift != nullptr) {                   // strategy pairs: seed with the parent contingency's
        CUDA_CHECK(cudaMemcpyAsync(ctx->dDsSlackAbsorbed.p, dsSlackShift,   // absorbed total (the device
                nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream)); // analogue of the CPU re-anchor)
    } else {
        CUDA_CHECK(cudaMemsetAsync(ctx->dDsSlackAbsorbed.p, 0, nS * sizeof(double), ctx->stream));
    }
}

}  // namespace olfgpu
