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
#include "../common/codes/gpu_row_mode.cuh"
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
__device__ inline double distrQTargetQOf(int nTqOv, const int* tqOvKey, const double* tqOvVal,
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
// limit, reactiveLimitInitialTargetQ for a realistic-voltage freeze (OLF freezes at the INITIAL generation target Q), the injection
// channel's target for a bus that lost generators, the limit drift included - one source for the member's own
// row and for its peers. A pinned entry whose row is not a balance (a pilot-self pilot keeps its voltage identity)
// takes the pin rule directly. Rebuilding every pinned peer from `qLimitType == 1 ? reactiveLimitMinQ : reactiveLimitMaxQ` put a realistic-
// voltage freeze at its CEILING (g1 frozen at 10 MVar with a 6000 MVar maxQ on the three-controller fixture)
// and a generator-lost member at its lost generator's maximum.
__device__ inline double pinnedTargetQ(int s, int e, int n, int nReactiveLimitEntries, const int* mode, const double* target,
                                       const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow, const int* relocatedChildOwnRow,
                                       const char* limitType, const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ,
                                       const double* reactiveLimitLoadQ, int perScenarioReactiveLimits, const double* reactiveLimitInitialTargetQ,
                                       const char* reactiveLimitGeneratorsLost) {
    int vrow = reactiveLimitVoltageRow[e];
    int qrow = (reactiveLimitReactivePowerRow != nullptr && reactiveLimitReactivePowerRow[e] >= 0) ? reactiveLimitReactivePowerRow[e] : vrow;
    if (qrow >= 0 && mode[(size_t) s * n + qrow] == GpuRowMode::BUS_TARGET_Q) {
        return target[(size_t) s * n + qrow];
    }
    if (vrow >= 0 && vrow != qrow && mode[(size_t) s * n + vrow] == GpuRowMode::BUS_TARGET_Q) {  // the balance went home (coupler opened)
        return target[(size_t) s * n + vrow];
    }
    if (qrow >= 0 && relocatedChildOwnRow != nullptr && relocatedChildOwnRow[qrow] >= 0 && mode[(size_t) s * n + relocatedChildOwnRow[qrow]] == GpuRowMode::BUS_TARGET_Q) {
        return target[(size_t) s * n + relocatedChildOwnRow[qrow]];
    }
    char qLimitType = limitType[(size_t) s * nReactiveLimitEntries + e];
    size_t li = perScenarioReactiveLimits ? (size_t) s * nReactiveLimitEntries + e : (size_t) e;
    double lim = (qLimitType == GpuQLimitType::MIN_REALISTIC_V || qLimitType == GpuQLimitType::MAX_REALISTIC_V) && reactiveLimitInitialTargetQ != nullptr ? reactiveLimitInitialTargetQ[e]
               : isMinLimit(qLimitType) ? reactiveLimitMinQ[li] : reactiveLimitMaxQ[li];
    if (qLimitType == GpuQLimitType::FROZEN && reactiveLimitGeneratorsLost != nullptr && reactiveLimitGeneratorsLost[(size_t) s * nReactiveLimitEntries + e]) {
        lim = 0.0;                                       // every generator lost: no generation term at all
    }
    return lim - reactiveLimitLoadQ[li];
}

__global__ void updateDistrQTargetsBatch(int nMembers, int nScenarios, int n, int nReactiveLimitEntries,
                                         const double* mKey, const int* distrQMemberReactiveLimitEntry,
                                         const int* mGroupStart, const int* mGroupCount,
                                         const int* mRow, const char* mHasRow,
                                         const int* reactiveLimitVoltageRow, const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ,
                                         const double* reactiveLimitLoadQ, const char* limitType, int perScenarioReactiveLimits,
                                         double* target, const char* outerLoopCheckActive, const double* mTargetQ,
                                         const int* memberRelocationGroup, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                                         const int* mode, const double* pilotQ, int nReactiveRelocationGroups,
                                         int nDistrQInactiveMembers, const int* distrQInactiveScenarioMember,
                                         int nFrozen, const int* frozenSlotRow,
                                         int nTqOv, const int* tqOvKey, const double* tqOvVal,
                                         const int* mTgtPtr, const int* mTgtPeer,
                                         const int* pilotSelfReactiveLimitEntry, const int* relocatedChildOwnRow,
                                const char* distrQSubgroup, const int* reactiveRelocationDistrQMemberStart, const char* reactiveLimitGeneratorsLost,
                                const int* reactiveRelocationDependentBaseGroup, const char* pilotTargetVDeactivated,
                                const double* reactiveLimitInitialTargetQ, const int* reactiveLimitReactivePowerRow,
                                int nDistrQKeyOverrides, const int* distrQKeyOverrideScenarioMember, const double* distrQKeyOverrideKey) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nMembers * nScenarios) return;
    int k = tid % nMembers;
    int s = tid / nMembers;
    if (!outerLoopCheckActive[s]) return;
    int qg = memberRelocationGroup != nullptr ? memberRelocationGroup[k] : -1;
    if (!mHasRow[k] && qg < 0) return;                       // only members owning a packed DISTR_Q row
                                                             // (or linked to a relocation group, plan-B4)
    // STOCK PLACEMENT: this member's packed DISTR_Q row is its parent coupler's DUMMY_Q row, and a
    // scenario that kills that coupler pins it to an identity — the sharing equation (and so its target)
    // is then on the bus's OWN row, where fillDistrQBatch puts it. Without the same diversion here the
    // identity test below returned and the diverted row kept target 0: measured on the middle-section-loss
    // fixtures as 401.35 kV against OLF's 401.69 once the equation itself had moved.
    int mrow = mRow[k];
    if (mrow >= 0 && relocatedChildOwnRow != nullptr && relocatedChildOwnRow[mrow] >= 0 && mode[(size_t) s * n + mrow] == GpuRowMode::FIXED_VARIABLE) {
        mrow = relocatedChildOwnRow[mrow];
    }
    if (mrow >= 0 && rowIsFrozen(nFrozen, frozenSlotRow, s, mrow)) {
        return;                                              // islanded: the row is frozen, not ours to write
    }
    if (mrow >= 0 && (mode[(size_t) s * n + mrow] == GpuRowMode::FIXED_VARIABLE || mode[(size_t) s * n + mrow] == GpuRowMode::CROSS_FIXED_VARIABLE)) {
        // The row is a VOLTAGE IDENTITY this scenario (6: a cross identity on the relocated child's V column): a control OLF re-merged onto this bus after a
        // zero-impedance split (GpuBatchQualification.reHomedControls — the bus is a plain PV bus at its
        // own target, and the member has left the group), or a frozen/outaged row. A DISTR_Q target
        // written here replaces the voltage target: measured on rte6515 BUS-6029_BBS_1, VL-6029_1's
        // target 1.039 became 0.000 at the first RL pass and the solve ran to MAX_ITERATION on it.
        return;
    }
    if (mrow >= 0 && mode[(size_t) s * n + mrow] == GpuRowMode::BUS_TARGET_Q) {
        // The row hosts its OWN reactive balance this scenario (mode 4): an RL pin (the RL kernel
        // owns the target) or a control the contingency KILLED (the injection override owns it,
        // OLF's BUS_TARGET_Q). A DISTR_Q target has no business here — fillDistrQBatch already
        // skips the row's DISTR_Q emissions for the same reason. Measured on rte6515 BUS-1055_BBS_1:
        // the dead control's survivor converged at target 0, then this pass put the base DISTR_Q
        // target (0.02 pu) back and the re-solve balanced 2 MVar the CPU never asked for (0.65 kV).
        return;
    }
    int reK = distrQMemberReactiveLimitEntry[k];
    if (reK >= 0 && limitType[(size_t) s * nReactiveLimitEntries + reK] != GpuQLimitType::NONE) return;  // pinned: the RL kernel owns its row
    if (qg >= 0) {
        // plan-B4: the member whose row CURRENTLY hosts the relocated pilot Q carries the pilot's
        // Q target (captured at solve init); its DISTR_Q compensation does not apply. A non-carrier
        // member without a packed DISTR_Q row (the base free member, superseded mid-scenario) has
        // no row to maintain here — the RL kernel owns it while pinned.
        int carrier = reactiveRelocationCandidateRows[reactiveRelocationGroupStart[qg] + chooseCarrierRowInSubgroup(qg, reactiveRelocationGroupStart, reactiveRelocationCandidateRows, mode + (size_t) s * n,
                distrQSubgroup != nullptr ? distrQSubgroup + (size_t) s * nMembers : nullptr, reactiveRelocationDistrQMemberStart)];
        // A group whose PILOT ROW is frozen this scenario has lost its pilot to an island (tasks #69): its
        // surviving sections form a NEW group OLF re-elects a main for, and that main's sharing rows are
        // plain DISTR_Q rows renormalized over the survivors — there is NO base-pilot balance left to
        // relocate onto a carrier. Writing pilotQ (the dead pilot's BASE target) there would replace the
        // survivor's sharing equation with a stale number. updatePilotTargetVEquationBatch already exits on the same
        // test; this is its counterpart on the target side.
        bool pilotFrozen = rowIsFrozen(nFrozen, frozenSlotRow, s, reactiveRelocationCandidateRows[reactiveRelocationGroupStart[qg]]);
        if (pilotFrozen && reactiveRelocationDependentBaseGroup != nullptr) {
            // The base pilot is lost to an island and OLF re-merged the survivors under a DEPENDENT pilot (a merged
            // controlled bus of the base group): the member row carrying that pilot's balance - its group picks on
            // the same modes as this one, so the same row - takes the dependent pilot's captured target.
            for (int g2 = 0; g2 < nReactiveRelocationGroups; g2++) {
                if (reactiveRelocationDependentBaseGroup[g2] != qg) continue;
                int pr = reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g2]];
                if (relocatedChildOwnRow != nullptr && relocatedChildOwnRow[pr] >= 0 && mode[(size_t) s * n + pr] == GpuRowMode::FIXED_VARIABLE) pr = relocatedChildOwnRow[pr];
                int pm = mode[(size_t) s * n + pr];
                if (pm != 0 && pm != 6 && !(pilotTargetVDeactivated != nullptr && pilotTargetVDeactivated[(size_t) s * nReactiveRelocationGroups + g2])) continue;
                int c2 = reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g2] + chooseCarrierRowInSubgroup(g2, reactiveRelocationGroupStart, reactiveRelocationCandidateRows, mode + (size_t) s * n,
                        distrQSubgroup != nullptr ? distrQSubgroup + (size_t) s * nMembers : nullptr, reactiveRelocationDistrQMemberStart)];
                if (c2 == mrow) {
                    target[(size_t) s * n + mrow] = pilotQ[(size_t) s * nReactiveRelocationGroups + g2];
                    return;
                }
            }
        }
        if (mrow == carrier && !pilotFrozen) {
            int pe = pilotSelfReactiveLimitEntry != nullptr ? pilotSelfReactiveLimitEntry[qg] : -1;  // the group's own pilot, when it is one of its controllers
            if (pe < 0) {
                target[(size_t) s * n + mrow] = pilotQ[(size_t) s * nReactiveRelocationGroups + qg];
                return;
            }
            char plt = limitType[(size_t) s * nReactiveLimitEntries + pe];
            if (plt != GpuQLimitType::NONE) {
                // pilot-self, pilot PINNED: the carrier hosts the pilot's own balance at OLF's frozen targetQ (the RL
                // kernel leaves the pilot's row a V identity in this case, so pinnedTargetQ takes the pin rule: the
                // limit, the initial generation target Q of a realistic-voltage freeze, nothing for a bus whose every
                // generator is lost - rte6515 GEN-676 with every busbar section in the list read its lost generator's
                // maximum here: the carrier's target 0.957 against OLF's -0.043, VL-676_1 supplying the 100 MVAr,
                // pinned, unpinned, and 1.133 MW distributed where OLF distributes nothing).
                target[(size_t) s * n + mrow] = pinnedTargetQ(s, pe, n, nReactiveLimitEntries, mode, target, reactiveLimitVoltageRow, reactiveLimitReactivePowerRow, relocatedChildOwnRow,
                        limitType, reactiveLimitMinQ, reactiveLimitMaxQ, reactiveLimitLoadQ, perScenarioReactiveLimits, reactiveLimitInitialTargetQ, reactiveLimitGeneratorsLost);
                return;
            }
            // pilot-self, pilot UNPINNED: its q feeds the DISTR_Q rows weighted and the carrier hosts its OWN
            // DISTR_Q (pilotSelfCarrierSkip keeps the gate off), so this row wants the normal DISTR_Q target below — not a
            // relocated pilot balance that does not exist. Writing pilotQ here is what churned the batch.
        }
        if (!mHasRow[k]) return;
    }
    int gs = mGroupStart[k];
    int gc = mGroupCount[k];
    double sum = 0.0;
    const char* subS = distrQSubgroup != nullptr ? distrQSubgroup + (size_t) s * nMembers : nullptr;
    char subK = subS != nullptr ? subS[k] : 0;
    for (int m = gs; m < gs + gc; m++) {                     // live keys over the WHOLE merged group (updateReactiveKeys)
        int re = distrQMemberReactiveLimitEntry[m];
        bool pinned = re >= 0 && limitType[(size_t) s * nReactiveLimitEntries + re] != GpuQLimitType::NONE;
        bool inactive = distrQMemberInactive(nDistrQInactiveMembers, distrQInactiveScenarioMember, s, m);
        bool otherSub = subS != nullptr && subS[m] != subK;  // a split group's other subgroup (tasks #71)
        sum += (pinned || inactive || otherSub) ? 0.0 : distrQKeyOf(nDistrQKeyOverrides, distrQKeyOverrideScenarioMember, distrQKeyOverrideKey, s, m, mKey);
    }
    // targetQ over the peers AcTargetVector sums for THIS member: the elements of its OWN control
    // (getGeneratorVoltageControl().getMergedControllerElements()), which for a control merged INTO this
    // group is the member alone. Summing the whole merged group instead left a coupler-merged member's
    // target off by qPct x the other members' targetQ — rte6515 BUS-2226_BBS_1 alone, 1.85 MVar, and
    // OLF's own re-test of the device's base state then failed the load flow.
    double sumTargetQ = 0.0;
    for (int t = mTgtPtr[k]; t < mTgtPtr[k + 1]; t++) {
        int m = mTgtPeer[t];
        int re = distrQMemberReactiveLimitEntry[m];
        bool pinned = re >= 0 && limitType[(size_t) s * nReactiveLimitEntries + re] != GpuQLimitType::NONE;
        if (pinned) {
            // a FROZEN member's targetQ is what OLF froze it at (freezeGenerationTargetQ...: the limit on a
            // reactive-limit switch, the INITIAL generation target Q on a realistic-voltage freeze), minus its
            // load - the value the member's own balance row enforces. Unpinned members keep their packed targetQ.
            sumTargetQ += pinnedTargetQ(s, re, n, nReactiveLimitEntries, mode, target, reactiveLimitVoltageRow, reactiveLimitReactivePowerRow, relocatedChildOwnRow,
                    limitType, reactiveLimitMinQ, reactiveLimitMaxQ, reactiveLimitLoadQ, perScenarioReactiveLimits, reactiveLimitInitialTargetQ, reactiveLimitGeneratorsLost);
        } else {
            sumTargetQ += distrQTargetQOf(nTqOv, tqOvKey, tqOvVal, s, nMembers, m, mTargetQ);
        }
    }
    double qpk = (sum == 0.0 || distrQMemberInactive(nDistrQInactiveMembers, distrQInactiveScenarioMember, s, k))
            ? 0.0 : distrQKeyOf(nDistrQKeyOverrides, distrQKeyOverrideScenarioMember, distrQKeyOverrideKey, s, k, mKey) / sum;
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
            - distrQTargetQOf(nTqOv, tqOvKey, tqOvVal, s, nMembers, k, mTargetQ);
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
__global__ void pilotRowAllPinnedBatch(int nEntries, int nScenarios, int n, int nReactiveLimitEntries,
                                       const int* entSlot, const int* entRow, const double* entQTarget,
                                       const int* entOff, const int* entCount, const int* survivorReactiveLimitEntries,
                                       const char* limitType, int* mode, double* target,
                                       char* flipped, double* savedVTarget,
                                       const char* outerLoopCheckActive, char* switched) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEntries) return;
    int s = entSlot[e];
    if (s < 0 || s >= nScenarios || !outerLoopCheckActive[s]) return;
    int row = entRow[e];
    if (row < 0 || row >= n) return;
    size_t ri = (size_t) s * n + row;
    bool allPinned = entCount[e] > 0;
    for (int k = 0; k < entCount[e]; k++) {
        int survivorEntry = survivorReactiveLimitEntries[entOff[e] + k];
        if (survivorEntry < 0 || survivorEntry >= nReactiveLimitEntries || limitType[(size_t) s * nReactiveLimitEntries + survivorEntry] == GpuQLimitType::NONE) {
            allPinned = false;                              // this one still carries its DISTR_Q row
            break;
        }
    }
    if (allPinned && !flipped[e] && mode[ri] == GpuRowMode::FIXED_VARIABLE) {
        savedVTarget[e] = target[ri];
        mode[ri] = GpuRowMode::BUS_TARGET_Q;
        target[ri] = entQTarget[e];
        flipped[e] = 1;
        switched[s] = 1;
    } else if (!allPinned && flipped[e]) {
        mode[ri] = GpuRowMode::FIXED_VARIABLE;
        target[ri] = savedVTarget[e];
        flipped[e] = 0;
        switched[s] = 1;
    }
}

// Remote-member OWN reactive injection (plan-B3): accumulate each remote member's own bus q into
// the diagnostic Q buffer at its OWN row (suppressed by the main qBuf fill), so the RL kernel's
// q = qBuf[row] + loadQ works verbatim for remote entries. Same family evaluation as fillDistrQBatch
// with weight 1 (±1 for zero-impedance DUMMY_Q terms), residual value only (no Jacobian).
__global__ void fillRemoteMemberQBatch(int nRemoteMemberReactiveTerms, int nScenarios, int n, int nElements,
                                       int closedBranchOffset, int openBranchOffset,
                                       const int* remoteMemberReactiveTermRow, const int* remoteMemberReactiveTermElement, const int* remoteMemberReactiveTermSide,
                                       const int* remoteMemberReactiveTermKind, const double* remoteMemberReactiveTermWeight,
                                       const double* closedBranchParameters, const int* closedBranchIndices, const double* shuntParameters, const int* shuntIndices,
                                       const double* openBranchParameters, const int* openBranchIndices,
                                       const double* x, double* qBuf,
                                       const char* disabled, const char* outerLoopCheckActive,
                                       int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues,
                                       int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nRemoteMemberReactiveTerms * nScenarios) return;
    int c = tid % nRemoteMemberReactiveTerms;
    int s = tid / nRemoteMemberReactiveTerms;
    if (!outerLoopCheckActive[s]) return;
    int e = remoteMemberReactiveTermElement[c];
    int kind = remoteMemberReactiveTermKind[c];
    if (kind == GpuReactiveTermKind::CLOSED_BRANCH_REACTIVE_FLOW && disabled[s * nElements + closedBranchOffset + e]) {
        return;
    }
    if (kind == GpuReactiveTermKind::OPEN_BRANCH_REACTIVE_FLOW && disabled[s * nElements + openBranchOffset + e]) {
        return;
    }
    double w = remoteMemberReactiveTermWeight[c];
    const double* xs = x + (size_t) s * n;
    double* q = qBuf + (size_t) s * n + remoteMemberReactiveTermRow[c];
    if (kind == GpuReactiveTermKind::SHUNT_COMPENSATOR_REACTIVE_FLOW) {  // fixed shunt q at the member bus
        const double* sp = shuntParameters + e * 2;
        double g = sp[0];
        double b = sp[1];
        applyShuntOverride(nShuntAdmittanceOverrides, shuntAdmittanceOverrideScenarioElement, shuntAdmittanceOverrideDelta, s, e, g, b);
        int vcol = shuntIndices[e * 5 + 2];
        double so[SH_NOUT];
        evalShunt(xs[vcol], g, b, 0, so + 0, so + 1, so + 2, so + 3, so + 4);
        atomicAdd(q, w * so[2]);
        return;
    }
    if (kind == GpuReactiveTermKind::DUMMY_Q) {  // zero-impedance DUMMY_Q linear term
        atomicAdd(q, w * xs[e]);
        return;
    }
    if (kind == GpuReactiveTermKind::OPEN_BRANCH_REACTIVE_FLOW) {  // half-open branch, connected side
        const double* op = openBranchParameters + e * 7;
        const int* openBranch = openBranchIndices + e * 6;
        int vcol = remoteMemberReactiveTermSide[c] == 0 ? openBranch[4] : openBranch[5];
        double v1 = remoteMemberReactiveTermSide[c] == 0 ? xs[vcol] : 1.0;
        double v2 = remoteMemberReactiveTermSide[c] == 1 ? xs[vcol] : 1.0;
        double oo[OB_NOUT];
        evalOpenBranch(op[0], op[1], op[2], op[3], op[4], op[5], v1, op[6], v2, 0,
                       oo + 0, oo + 1, oo + 2, oo + 3, oo + 4, oo + 5, oo + 6, oo + 7);
        atomicAdd(q, w * oo[remoteMemberReactiveTermSide[c] == 0 ? 6 : 2]);
        return;
    }
    // kind 0: closed branch q at the member's side (full pi row honors the tap override channel)
    const double* p0 = closedBranchParameters + e * 8;
    double p[8];
    for (int i = 0; i < 8; i++) {
        p[i] = p0[i];
    }
    double pOv[8];
    if (findTapOverride(nClosedBranchParameterOverrides, closedBranchParameterOverrideScenarioElement, closedBranchParameterOverrideValues, s, e, pOv)) {
        for (int i = 0; i < 8; i++) {
            p[i] = pOv[i];
        }
    }
    const int* qx = closedBranchIndices + e * 12;
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
    atomicAdd(q, w * o[remoteMemberReactiveTermSide[c] == 0 ? 7 : 21]);           // q1 / q2 residual outputs
    return;
}

void launchDistrQTargetsInner(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const bool& usePerScenarioReactiveLimits = drv.usePerScenarioReactiveLimits;
    if (ctx->nReactiveLimitRemoteEntries > 0 && ctx->nDistrQMembers > 0) {
        updateDistrQTargetsBatch<<<grid(ctx->nDistrQMembers * nS), BLOCK, 0, ctx->stream>>>(
                    ctx->nDistrQMembers, nS, n, ctx->nReactiveLimitEntries,
                    ctx->dDistrQMemberKey.as<double>(), ctx->dDistrQMemberReactiveLimitEntry.as<int>(),
                    ctx->dDistrQMemberGroupStart.as<int>(), ctx->dDistrQMemberGroupCount.as<int>(),
                    ctx->dDistrQMemberRow.as<int>(), ctx->dDistrQMemberHasRow.as<char>(),
                    ctx->dReactiveLimitVoltageRow.as<int>(),
                    usePerScenarioReactiveLimits ? ctx->dReactiveLimitMinQPerScenario.as<double>() : ctx->dReactiveLimitMinQ.as<double>(),
                    usePerScenarioReactiveLimits ? ctx->dReactiveLimitMaxQPerScenario.as<double>() : ctx->dReactiveLimitMaxQ.as<double>(),
                    usePerScenarioReactiveLimits ? ctx->dReactiveLimitLoadQPerScenario.as<double>() : ctx->dReactiveLimitLoadQ.as<double>(), ctx->dReactiveLimitQLimitType.as<char>(), usePerScenarioReactiveLimits ? 1 : 0,
                    ctx->dTarget.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
                    ctx->dDistrQMemberTargetQ.as<double>(),
                    ctx->hasDistrQRelocationGroups ? ctx->dDistrQMemberRelocationGroup.as<int>() : nullptr,
                    ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->dReactiveRelocationGroupStart.as<int>() : nullptr,
                    ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->dReactiveRelocationCandidateRows.as<int>() : nullptr,
                    ctx->dMode.as<int>(),
                    ctx->hasDistrQRelocationGroups ? ctx->dReactiveRelocationPilotTargetQ.as<double>() : nullptr, ctx->structure->nReactiveRelocationGroups,
                    ctx->nDistrQInactiveMembers, ctx->nDistrQInactiveMembers ? ctx->dDistrQInactiveMembers.as<int>() : nullptr,
                    ctx->nFrozenRow, ctx->nFrozenRow ? ctx->dFrozenRow.as<int>() : nullptr,
                    ctx->nDistrQMemberTargetQOverrides, ctx->nDistrQMemberTargetQOverrides ? ctx->dDistrQMemberTargetQOverrideKey.as<int>() : nullptr,
                    ctx->nDistrQMemberTargetQOverrides ? ctx->dDistrQMemberTargetQOverrideValue.as<double>() : nullptr,
                    ctx->dDistrQMemberTargetPeerStart.as<int>(), ctx->dDistrQMemberTargetPeer.as<int>(),
                    pilotSelfReactiveLimitEntryOrNull(ctx),
                    ctx->structure->dRelocatedChildOwnRow.p != nullptr
                            ? ctx->structure->dRelocatedChildOwnRow.as<int>() : nullptr,
                ctx->hasDistrQSubgroups ? ctx->dDistrQSubgroup.as<char>() : nullptr, ctx->hasDistrQSubgroups ? ctx->dReactiveRelocationDistrQMemberStart.as<int>() : nullptr,
                ctx->nReactiveLimitGeneratorsLost > 0 ? ctx->dReactiveLimitGeneratorsLost.as<char>() : nullptr,
                ctx->structure->dReactiveRelocationDependentBaseGroup.p ? ctx->structure->dReactiveRelocationDependentBaseGroup.as<int>() : nullptr,
                ctx->dReactiveRelocationPilotTargetVDeactivated.p ? ctx->dReactiveRelocationPilotTargetVDeactivated.as<char>() : nullptr,
                    ctx->dReactiveLimitInitialTargetQ.as<double>(), ctx->dReactiveLimitReactivePowerRow.as<int>(),
                    ctx->nDistrQKeyOverrides, ctx->nDistrQKeyOverrides ? ctx->dDistrQKeyOverrideScenarioMember.as<int>() : nullptr, ctx->nDistrQKeyOverrides ? ctx->dDistrQKeyOverrideKey.as<double>() : nullptr);
    }
}

}  // namespace olfgpu
