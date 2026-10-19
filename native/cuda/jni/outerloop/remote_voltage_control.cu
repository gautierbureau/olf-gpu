// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See remote_voltage_control.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_q_limit_type.cuh"
#include "../common/codes/gpu_reactive_term_kind.cuh"
#include "closed_branch.cuh"
#include "closed_branch_scatter.h"
#include "open_branch.cuh"
#include "open_branch_scatter.h"
#include "shunt.cuh"
#include "shunt_scatter.h"
#include "../equations/scenario_overrides.cuh"
#include "../solver/driver_steps.cuh"
#include "remote_voltage_control.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

// DISTR_Q target compensation (plan-B3): OLF's DISTR_Q row target is
// (qPct_k - 1)*targetQ_k + qPct_k*sum_{j!=k} targetQ_j, and targetQ_j is nonzero exactly for a
// FROZEN member (pinned at a limit). Recompute every UNPINNED member row's target from the
// per-scenario limitType so an in-flight pin/unpin keeps the targets consistent with the live
// renormalized weights. Idempotent per pass; one thread per (member, scenario).
// Per-scenario member targetQ. A member bus that loses reactive LOAD changes its own targetQ
// (targetQ = generation - load), and every DISTR_Q target in its group reads that value twice: once
// inside the shared sum and once as its own subtracted term. Seeding the row targets host-side is not
// enough, because this kernel RECOMPUTES them from mTargetQ on every reactive-limits pass and would
// put the base value straight back (measured on rte6515 LOAD-1263: the seeded row came back at
// final = base = 0.016, residual still 4.6e-02). The override list is tiny — one entry per outaged
// member bus — so a linear scan per lookup costs nothing.
__device__ inline double dqTargetQOf(int nTqOv, const int* tqOvKey, const double* tqOvVal,
                                     int s, int nMembers, int idx, const double* mTargetQ) {
    if (nTqOv > 0) {
        int want = s * nMembers + idx;
        for (int i = 0; i < nTqOv; i++) {
            if (tqOvKey[i] == want) {
                return tqOvVal[i];
            }
        }
    }
    return mTargetQ[idx];
}

// OLF's frozen targetQ (generation minus load) of a PINNED reactive-limit entry this scenario, the value its peers'
// DISTR_Q targets sum (AcTargetVector: otherControllerBus.getTargetQ()) and a pilot-self carrier hosts. The
// entry's own balance row carries that value whenever the pin gave it one (mode 4): the reactive-limit kernel's
// limit, rlInitTgtQ for a realistic-voltage freeze (OLF freezes at the INITIAL generation target Q), the injection
// channel's target for a bus that lost generators, the limit drift included - one source for the member's own
// row and for its peers. A pinned entry whose row is not a balance (a pilot-self pilot keeps its voltage identity)
// takes the pin rule directly. Rebuilding every pinned peer from `lt == 1 ? rlMinQ : rlMaxQ` put a realistic-
// voltage freeze at its CEILING (g1 frozen at 10 MVar with a 6000 MVar maxQ on the three-controller fixture)
// and a generator-lost member at its lost generator's maximum.
__device__ inline double pinnedTargetQ(int s, int e, int n, int nrl, const int* mode, const double* target,
                                       const int* rlVrowArr, const int* rlQrow, const int* relocOwn,
                                       const char* limitType, const double* rlMinQ, const double* rlMaxQ,
                                       const double* rlLoadQ, int cvStride, const double* rlInitTgtQ,
                                       const char* rlGenLost) {
    int vrow = rlVrowArr[e];
    int qrow = (rlQrow != nullptr && rlQrow[e] >= 0) ? rlQrow[e] : vrow;
    if (qrow >= 0 && mode[(size_t) s * n + qrow] == 4) {
        return target[(size_t) s * n + qrow];
    }
    if (vrow >= 0 && vrow != qrow && mode[(size_t) s * n + vrow] == 4) {   // the balance went home (coupler opened)
        return target[(size_t) s * n + vrow];
    }
    if (qrow >= 0 && relocOwn != nullptr && relocOwn[qrow] >= 0 && mode[(size_t) s * n + relocOwn[qrow]] == 4) {
        return target[(size_t) s * n + relocOwn[qrow]];
    }
    char lt = limitType[(size_t) s * nrl + e];
    size_t li = cvStride ? (size_t) s * nrl + e : (size_t) e;
    double lim = (lt == GpuQLimitType::MIN_REALISTIC_V || lt == GpuQLimitType::MAX_REALISTIC_V) && rlInitTgtQ != nullptr ? rlInitTgtQ[e]
               : isMinLimit(lt) ? rlMinQ[li] : rlMaxQ[li];
    if (lt == GpuQLimitType::FROZEN && rlGenLost != nullptr && rlGenLost[(size_t) s * nrl + e]) {
        lim = 0.0;                                       // every generator lost: no generation term at all
    }
    return lim - rlLoadQ[li];
}

__global__ void updateDistrQTargetsBatch(int nMembers, int nScenarios, int n, int nrl,
                                         const double* mKey, const int* mRlEntry,
                                         const int* mGroupStart, const int* mGroupCount,
                                         const int* mRow, const char* mHasRow,
                                         const int* rlVrowArr, const double* rlMinQ, const double* rlMaxQ,
                                         const double* rlLoadQ, const char* limitType, int cvStride,
                                         double* target, const char* rlActive, const double* mTargetQ,
                                         const int* mQrGroup, const int* qrPtr, const int* qrRows,
                                         const int* mode, const double* pilotQ, int nQr,
                                         int nDqInact, const int* dqInactSlotMember,
                                         int nFrozen, const int* frozenSlotRow,
                                         int nTqOv, const int* tqOvKey, const double* tqOvVal,
                                         const int* mTgtPtr, const int* mTgtPeer,
                                         const int* psEntry, const int* relocOwn,
                                const char* dqSub, const int* qrMemberStart, const char* rlGenLost,
                                const int* qrDepBase, const char* veqOff,
                                const double* rlInitTgtQ, const int* rlQrow,
                                int nDqKeyOv, const int* dqKeyOvIdx, const double* dqKeyOvKey) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nMembers * nScenarios) return;
    int k = tid % nMembers;
    int s = tid / nMembers;
    if (!rlActive[s]) return;
    int qg = mQrGroup != nullptr ? mQrGroup[k] : -1;
    if (!mHasRow[k] && qg < 0) return;                       // only members owning a packed DISTR_Q row
                                                             // (or linked to a relocation group, plan-B4)
    // STOCK PLACEMENT: this member's packed DISTR_Q row is its parent coupler's DUMMY_Q row, and a
    // scenario that kills that coupler pins it to an identity — the sharing equation (and so its target)
    // is then on the bus's OWN row, where fillDistrQBatch puts it. Without the same diversion here the
    // identity test below returned and the diverted row kept target 0: measured on the middle-section-loss
    // fixtures as 401.35 kV against OLF's 401.69 once the equation itself had moved.
    int mrow = mRow[k];
    if (mrow >= 0 && relocOwn != nullptr && relocOwn[mrow] >= 0 && mode[(size_t) s * n + mrow] == 0) {
        mrow = relocOwn[mrow];
    }
    if (mrow >= 0 && rowIsFrozen(nFrozen, frozenSlotRow, s, mrow)) {
        return;                                              // islanded: the row is frozen, not ours to write
    }
    if (mrow >= 0 && (mode[(size_t) s * n + mrow] == 0 || mode[(size_t) s * n + mrow] == 6)) {
        // The row is a VOLTAGE IDENTITY this scenario (6: a cross identity on the relocated child's V column): a control OLF re-merged onto this bus after a
        // zero-impedance split (GpuBatchQualification.reHomedControls — the bus is a plain PV bus at its
        // own target, and the member has left the group), or a frozen/outaged row. A DISTR_Q target
        // written here replaces the voltage target: measured on rte6515 BUS-6029_BBS_1, VL-6029_1's
        // target 1.039 became 0.000 at the first RL pass and the solve ran to MAX_ITERATION on it.
        return;
    }
    if (mrow >= 0 && mode[(size_t) s * n + mrow] == 4) {
        // The row hosts its OWN reactive balance this scenario (mode 4): an RL pin (the RL kernel
        // owns the target) or a control the contingency KILLED (the injection override owns it,
        // OLF's BUS_TARGET_Q). A DISTR_Q target has no business here — fillDistrQBatch already
        // skips the row's DISTR_Q emissions for the same reason. Measured on rte6515 BUS-1055_BBS_1:
        // the dead control's survivor converged at target 0, then this pass put the base DISTR_Q
        // target (0.02 pu) back and the re-solve balanced 2 MVar the CPU never asked for (0.65 kV).
        return;
    }
    int reK = mRlEntry[k];
    if (reK >= 0 && limitType[(size_t) s * nrl + reK] != GpuQLimitType::NONE) return;  // pinned: the RL kernel owns its row
    if (qg >= 0) {
        // plan-B4: the member whose row CURRENTLY hosts the relocated pilot Q carries the pilot's
        // Q target (captured at solve init); its DISTR_Q compensation does not apply. A non-carrier
        // member without a packed DISTR_Q row (the base free member, superseded mid-scenario) has
        // no row to maintain here — the RL kernel owns it while pinned.
        int carrier = qrRows[qrPtr[qg] + relocChooseSub(qg, qrPtr, qrRows, mode + (size_t) s * n,
                dqSub != nullptr ? dqSub + (size_t) s * nMembers : nullptr, qrMemberStart)];
        // A group whose PILOT ROW is frozen this scenario has lost its pilot to an island (tasks #69): its
        // surviving sections form a NEW group OLF re-elects a main for, and that main's sharing rows are
        // plain DISTR_Q rows renormalized over the survivors — there is NO base-pilot balance left to
        // relocate onto a carrier. Writing pilotQ (the dead pilot's BASE target) there would replace the
        // survivor's sharing equation with a stale number. updatePilotVeqBatch already exits on the same
        // test; this is its counterpart on the target side.
        bool pilotFrozen = rowIsFrozen(nFrozen, frozenSlotRow, s, qrRows[qrPtr[qg]]);
        if (pilotFrozen && qrDepBase != nullptr) {
            // The base pilot is lost to an island and OLF re-merged the survivors under a DEPENDENT pilot (a merged
            // controlled bus of the base group): the member row carrying that pilot's balance - its group picks on
            // the same modes as this one, so the same row - takes the dependent pilot's captured target.
            for (int g2 = 0; g2 < nQr; g2++) {
                if (qrDepBase[g2] != qg) continue;
                int pr = qrRows[qrPtr[g2]];
                if (relocOwn != nullptr && relocOwn[pr] >= 0 && mode[(size_t) s * n + pr] == 0) pr = relocOwn[pr];
                int pm = mode[(size_t) s * n + pr];
                if (pm != 0 && pm != 6 && !(veqOff != nullptr && veqOff[(size_t) s * nQr + g2])) continue;
                int c2 = qrRows[qrPtr[g2] + relocChooseSub(g2, qrPtr, qrRows, mode + (size_t) s * n,
                        dqSub != nullptr ? dqSub + (size_t) s * nMembers : nullptr, qrMemberStart)];
                if (c2 == mrow) {
                    target[(size_t) s * n + mrow] = pilotQ[(size_t) s * nQr + g2];
                    return;
                }
            }
        }
        if (mrow == carrier && !pilotFrozen) {
            int pe = psEntry != nullptr ? psEntry[qg] : -1;  // the group's own pilot, when it is one of its controllers
            if (pe < 0) {
                target[(size_t) s * n + mrow] = pilotQ[(size_t) s * nQr + qg];
                return;
            }
            char plt = limitType[(size_t) s * nrl + pe];
            if (plt != GpuQLimitType::NONE) {
                // pilot-self, pilot PINNED: the carrier hosts the pilot's own balance at OLF's frozen targetQ (the RL
                // kernel leaves the pilot's row a V identity in this case, so pinnedTargetQ takes the pin rule: the
                // limit, the initial generation target Q of a realistic-voltage freeze, nothing for a bus whose every
                // generator is lost - rte6515 GEN-676 with every busbar section in the list read its lost generator's
                // maximum here: the carrier's target 0.957 against OLF's -0.043, VL-676_1 supplying the 100 MVAr,
                // pinned, unpinned, and 1.133 MW distributed where OLF distributes nothing).
                target[(size_t) s * n + mrow] = pinnedTargetQ(s, pe, n, nrl, mode, target, rlVrowArr, rlQrow, relocOwn,
                        limitType, rlMinQ, rlMaxQ, rlLoadQ, cvStride, rlInitTgtQ, rlGenLost);
                return;
            }
            // pilot-self, pilot UNPINNED: its q feeds the DISTR_Q rows weighted and the carrier hosts its OWN
            // DISTR_Q (psSkip keeps the gate off), so this row wants the normal DISTR_Q target below — not a
            // relocated pilot balance that does not exist. Writing pilotQ here is what churned the batch.
        }
        if (!mHasRow[k]) return;
    }
    int gs = mGroupStart[k];
    int gc = mGroupCount[k];
    double sum = 0.0;
    const char* subS = dqSub != nullptr ? dqSub + (size_t) s * nMembers : nullptr;
    char subK = subS != nullptr ? subS[k] : 0;
    for (int m = gs; m < gs + gc; m++) {                     // live keys over the WHOLE merged group (updateReactiveKeys)
        int re = mRlEntry[m];
        bool pinned = re >= 0 && limitType[(size_t) s * nrl + re] != GpuQLimitType::NONE;
        bool inactive = dqMemberInactive(nDqInact, dqInactSlotMember, s, m);
        bool otherSub = subS != nullptr && subS[m] != subK;  // a split group's other subgroup (tasks #71)
        sum += (pinned || inactive || otherSub) ? 0.0 : dqKeyOf(nDqKeyOv, dqKeyOvIdx, dqKeyOvKey, s, m, mKey);
    }
    // targetQ over the peers AcTargetVector sums for THIS member: the elements of its OWN control
    // (getGeneratorVoltageControl().getMergedControllerElements()), which for a control merged INTO this
    // group is the member alone. Summing the whole merged group instead left a coupler-merged member's
    // target off by qPct x the other members' targetQ — rte6515 BUS-2226_BBS_1 alone, 1.85 MVar, and
    // OLF's own re-test of the device's base state then failed the load flow.
    double sumTargetQ = 0.0;
    for (int t = mTgtPtr[k]; t < mTgtPtr[k + 1]; t++) {
        int m = mTgtPeer[t];
        int re = mRlEntry[m];
        bool pinned = re >= 0 && limitType[(size_t) s * nrl + re] != GpuQLimitType::NONE;
        if (pinned) {
            // a FROZEN member's targetQ is what OLF froze it at (freezeGenerationTargetQ...: the limit on a
            // reactive-limit switch, the INITIAL generation target Q on a realistic-voltage freeze), minus its
            // load - the value the member's own balance row enforces. Unpinned members keep their packed targetQ.
            sumTargetQ += pinnedTargetQ(s, re, n, nrl, mode, target, rlVrowArr, rlQrow, relocOwn,
                    limitType, rlMinQ, rlMaxQ, rlLoadQ, cvStride, rlInitTgtQ, rlGenLost);
        } else {
            sumTargetQ += dqTargetQOf(nTqOv, tqOvKey, tqOvVal, s, nMembers, m, mTargetQ);
        }
    }
    double qpk = (sum == 0.0 || dqMemberInactive(nDqInact, dqInactSlotMember, s, k))
            ? 0.0 : dqKeyOf(nDqKeyOv, dqKeyOvIdx, dqKeyOvKey, s, k, mKey) / sum;
    // AcTargetVector.getGeneratorReactivePowerDistributionTarget, verbatim:
    //   (qPct_k - 1) * targetQ_k + qPct_k * sum_{j != k} targetQ_j
    // which is qPct_k * sum_over_its_PEERS targetQ_j - targetQ_k (k is always one of its own peers). k is never pinned here (the RL kernel
    // owns a pinned member's row and we returned above), so its own term is the packed targetQ.
    // This used to read `qpk * frozen` — the same expression with every UNPINNED targetQ assumed
    // zero. That assumption holds only for groups whose members carry no load and no generator
    // targetQ; anywhere else the device solved the row against a target short by exactly
    // qPct_k * sum_unpinned targetQ_j - targetQ_k, and drove the members to equal reactive
    // injections instead of equally-shared DEVIATIONS from their targets.
    target[(size_t) s * n + mrow] = qpk * sumTargetQ
            - dqTargetQOf(nTqOv, tqOvKey, tqOvVal, s, nMembers, k, mTargetQ);
}

// A pilot that STOPPED being one of its OWN controllers (this scenario outaged its generators) keeps
// its BUS_TARGET_V while the surviving controllers still hold its voltage — the reactive balance it
// just acquired rides one of THEIR rows. When every survivor PINS, each of those rows flips to mode 4
// to host its own pinned balance, and the pilot's balance is evicted with nowhere left to go: the
// solve converges happily on a system one equation short (rte6515 GEN-1522: OLF's residual 0.601 pu
// on BUS_TARGET_Q of the pilot, every device equation converged). OLF drops the now-uncontrollable
// BUS_TARGET_V and activates the pilot's own BUS_TARGET_Q on the freed row; this makes the same move
// INSIDE the solve, so the re-solve sees it.
//
// Mode 4 is "row hosts its own Q balance" via the member ALTERNATE dest. It only reaches the pilot's
// q because the extraction now gives a pilot-that-is-its-own-controller that alternate; without it
// the row assembles EMPTY and the flip is a measured no-op (state unchanged to 10 digits).
// Reversible: if a later pass unpins a survivor, the V identity and its target come back.
__global__ void pilotRowAllPinnedBatch(int nEntries, int nScenarios, int n, int nrl,
                                       const int* entSlot, const int* entRow, const double* entQTarget,
                                       const int* entOff, const int* entCount, const int* entRlPool,
                                       const char* limitType, int* mode, double* target,
                                       char* flipped, double* savedVTarget,
                                       const char* rlActive, char* switched) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEntries) return;
    int s = entSlot[e];
    if (s < 0 || s >= nScenarios || !rlActive[s]) return;
    int row = entRow[e];
    if (row < 0 || row >= n) return;
    size_t ri = (size_t) s * n + row;
    bool allPinned = entCount[e] > 0;
    for (int k = 0; k < entCount[e]; k++) {
        int rl = entRlPool[entOff[e] + k];
        if (rl < 0 || rl >= nrl || limitType[(size_t) s * nrl + rl] == GpuQLimitType::NONE) {
            allPinned = false;                              // this one still carries its DISTR_Q row
            break;
        }
    }
    if (allPinned && !flipped[e] && mode[ri] == 0) {
        savedVTarget[e] = target[ri];
        mode[ri] = 4;
        target[ri] = entQTarget[e];
        flipped[e] = 1;
        switched[s] = 1;
    } else if (!allPinned && flipped[e]) {
        mode[ri] = 0;
        target[ri] = savedVTarget[e];
        flipped[e] = 0;
        switched[s] = 1;
    }
}

// Remote-member OWN reactive injection (plan-B3): accumulate each remote member's own bus q into
// the diagnostic Q buffer at its OWN row (suppressed by the main qBuf fill), so the RL kernel's
// q = qBuf[row] + loadQ works verbatim for remote entries. Same family evaluation as fillDistrQBatch
// with weight 1 (±1 for zero-impedance DUMMY_Q terms), residual value only (no Jacobian).
__global__ void fillRemoteMemberQBatch(int nrmq, int nScenarios, int n, int nElements,
                                       int cbOffset, int obOffset,
                                       const int* rmqRow, const int* rmqElem, const int* rmqSide,
                                       const int* rmqKind, const double* rmqWeight,
                                       const double* cbIn, const int* cbIx, const double* shIn, const int* shIx,
                                       const double* obIn, const int* obIx,
                                       const double* x, double* qBuf,
                                       const char* disabled, const char* rlActive,
                                       int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                       int nShOv, const int* shOvSlotElem, const double* shOvGb,
                                       int traceRow) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrmq * nScenarios) return;
    int c = tid % nrmq;
    int s = tid / nrmq;
    if (!rlActive[s]) return;
    int e = rmqElem[c];
    int kind = rmqKind[c];
    if (kind == GpuReactiveTermKind::CLOSED_BRANCH_REACTIVE_FLOW && disabled[s * nElements + cbOffset + e]) {
        if (rmqRow[c] == traceRow) {
            printf("RMQ s=%d c=%d kind=0(closed) elem=%d SKIPPED disabled\n", s, c, e);
        }
        return;
    }
    if (kind == GpuReactiveTermKind::OPEN_BRANCH_REACTIVE_FLOW && disabled[s * nElements + obOffset + e]) {
        if (rmqRow[c] == traceRow) {
            printf("RMQ s=%d c=%d kind=2(halfOpen) elem=%d SKIPPED disabled\n", s, c, e);
        }
        return;
    }
    double w = rmqWeight[c];
    const double* xs = x + (size_t) s * n;
    double* q = qBuf + (size_t) s * n + rmqRow[c];
    if (kind == GpuReactiveTermKind::SHUNT_COMPENSATOR_REACTIVE_FLOW) {  // fixed shunt q at the member bus
        const double* sp = shIn + e * 2;
        double g = sp[0];
        double b = sp[1];
        applyShuntOverride(nShOv, shOvSlotElem, shOvGb, s, e, g, b);
        int vcol = shIx[e * 5 + 2];
        double so[SH_NOUT];
        evalShunt(xs[vcol], g, b, 0, so + 0, so + 1, so + 2, so + 3, so + 4);
        if (rmqRow[c] == traceRow) {
            printf("RMQ s=%d c=%d kind=1(shunt) elem=%d w=%.6f contrib=%.9f\n", s, c, e, w, w * so[2]);
        }
        atomicAdd(q, w * so[2]);
        return;
    }
    if (kind == GpuReactiveTermKind::DUMMY_Q) {  // zero-impedance DUMMY_Q linear term
        if (rmqRow[c] == traceRow) {
            printf("RMQ s=%d c=%d kind=3(ziDummyQ) dqCol=%d w=%.6f dummyQ=%.9f contrib=%.9f\n",
                   s, c, e, w, xs[e], w * xs[e]);
        }
        atomicAdd(q, w * xs[e]);
        return;
    }
    if (kind == GpuReactiveTermKind::OPEN_BRANCH_REACTIVE_FLOW) {  // half-open branch, connected side
        const double* op = obIn + e * 7;
        const int* oq = obIx + e * 6;
        int vcol = rmqSide[c] == 0 ? oq[4] : oq[5];
        double v1 = rmqSide[c] == 0 ? xs[vcol] : 1.0;
        double v2 = rmqSide[c] == 1 ? xs[vcol] : 1.0;
        double oo[OB_NOUT];
        evalOpenBranch(op[0], op[1], op[2], op[3], op[4], op[5], v1, op[6], v2, 0,
                       oo + 0, oo + 1, oo + 2, oo + 3, oo + 4, oo + 5, oo + 6, oo + 7);
        if (rmqRow[c] == traceRow) {
            printf("RMQ s=%d c=%d kind=2(halfOpen) elem=%d side=%d w=%.6f contrib=%.9f\n",
                   s, c, e, rmqSide[c], w, w * oo[rmqSide[c] == 0 ? 6 : 2]);
        }
        atomicAdd(q, w * oo[rmqSide[c] == 0 ? 6 : 2]);
        return;
    }
    // kind 0: closed branch q at the member's side (full pi row honors the tap override channel)
    const double* p0 = cbIn + e * 8;
    double p[8];
    for (int i = 0; i < 8; i++) {
        p[i] = p0[i];
    }
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) {
        for (int i = 0; i < 8; i++) {
            p[i] = pOv[i];
        }
    }
    const int* qx = cbIx + e * 12;
    double v1 = xs[qx[5]];
    double v2 = xs[qx[7]];
    double ph1 = xs[qx[4]];
    double ph2 = xs[qx[6]];
    double r1 = qx[9] >= 0 ? xs[qx[9]] : p[6];
    double a1 = qx[8] >= 0 ? xs[qx[8]] : p[7];
    double o[CB_NOUT];
    evalClosedBranch(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    if (rmqRow[c] == traceRow) {
        printf("RMQ s=%d c=%d kind=0(closed) elem=%d side=%d w=%.6f v1=%.9f v2=%.9f"
               " ph1=%.9f ph2=%.9f r1=%.6f y=%.6f b1=%.6f b2=%.6f contrib=%.9f\n",
               s, c, e, rmqSide[c], w, v1, v2, ph1, ph2, r1, p[0], p[3], p[4],
               w * o[rmqSide[c] == 0 ? 7 : 21]);
    }
    atomicAdd(q, w * o[rmqSide[c] == 0 ? 7 : 21]);           // q1 / q2 residual outputs
    return;
}

void launchDistrQTargetsInner(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const bool& useDynRlLimits = drv.useDynRlLimits;
    if (ctx->nRlRemote > 0 && ctx->nDqMembers > 0) {
        updateDistrQTargetsBatch<<<grid(ctx->nDqMembers * nS), BLOCK, 0, ctx->stream>>>(
                    ctx->nDqMembers, nS, n, ctx->nrl,
                    ctx->dDqMemberKey.as<double>(), ctx->dDqMemberRlEntry.as<int>(),
                    ctx->dDqMemberGroupStart.as<int>(), ctx->dDqMemberGroupCount.as<int>(),
                    ctx->dDqMemberRow.as<int>(), ctx->dDqMemberHasRow.as<char>(),
                    ctx->dRlVrow.as<int>(),
                    useDynRlLimits ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                    useDynRlLimits ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                    useDynRlLimits ? ctx->dRlLoadQDyn.as<double>() : ctx->dRlLoadQ.as<double>(), ctx->dRlLimitType.as<char>(), useDynRlLimits ? 1 : 0,
                    ctx->dTarget.as<double>(), ctx->dRlActive.as<char>(),
                    ctx->dDqMemberTargetQ.as<double>(),
                    ctx->hasDqQrGroups ? ctx->dDqMemberQrGroup.as<int>() : nullptr,
                    ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr,
                    ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr,
                    ctx->dMode.as<int>(),
                    ctx->hasDqQrGroups ? ctx->dQrPilotQ.as<double>() : nullptr, ctx->st->nQr,
                    ctx->nDqInact, ctx->nDqInact ? ctx->dDqInact.as<int>() : nullptr,
                    ctx->nFrozenRow, ctx->nFrozenRow ? ctx->dFrozenRow.as<int>() : nullptr,
                    ctx->nDqTqOv, ctx->nDqTqOv ? ctx->dDqTqOvKey.as<int>() : nullptr,
                    ctx->nDqTqOv ? ctx->dDqTqOvVal.as<double>() : nullptr,
                    ctx->dDqMemberTgtPtr.as<int>(), ctx->dDqMemberTgtPeer.as<int>(),
                    psEntryPtr(ctx),
                    ctx->st->dRelocOwn.p != nullptr
                            ? ctx->st->dRelocOwn.as<int>() : nullptr,
                ctx->hasDqSub ? ctx->dDqSub.as<char>() : nullptr, ctx->hasDqSub ? ctx->dQrMemberStart.as<int>() : nullptr,
                ctx->nRlGenLost > 0 ? ctx->dRlGenLost.as<char>() : nullptr,
                ctx->st->dQrDepBase.p ? ctx->st->dQrDepBase.as<int>() : nullptr,
                ctx->dQrVeqOff.p ? ctx->dQrVeqOff.as<char>() : nullptr,
                    ctx->dRlInitTgtQ.as<double>(), ctx->dRlQrow.as<int>(),
                    ctx->nDqKeyOv, ctx->nDqKeyOv ? ctx->dDqKeyOvIdx.as<int>() : nullptr, ctx->nDqKeyOv ? ctx->dDqKeyOvKey.as<double>() : nullptr);
    }
}

}  // namespace olfgpu
