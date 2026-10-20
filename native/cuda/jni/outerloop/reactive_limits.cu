// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See reactive_limits.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../equations/scatter.cuh"
#include "../equations/scenario_overrides.cuh"
#include "../solver/driver_steps.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_ac_solver_status.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"
#include "../common/codes/gpu_q_limit_type.cuh"
#include "../common/codes/gpu_voltage_control_role.cuh"
#include "../common/codes/gpu_reactive_limits_entry_flag.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "forced_target_q.cuh"
#include "pilot_voltage.cuh"
#include "reactive_capability_curves.cuh"
#include "remote_voltage_control.cuh"
#include "reactive_limits.cuh"

namespace olfgpu {

__global__ void reactiveLimitsBatch(const double* svcRlVt, int nrl, int nScenarios, int n, double tol, int maxSwitch,
                                    const int* rlVrow, const int* rlQrow, const double* rlMinQ, const double* rlMaxQ,
                                    const double* rlLoadQ, const double* rlVtarget, const double* qBuf,
                                    const double* x, const char* rlActive, int* mode, double* target,
                                    char* limitType, char* switchCount, char* switched,
                                    char* passSwitched, char* savedMode,
                                    int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    const double* rowSlope, double* savedTarget, int cvStride,
                                    const char* rlRemote, const int* rlCmpVrow,
                                    const double* rlSvcBmin, const double* rlSvcBmax, const double* rlV2Base,
                                    const double* svcBminDyn, const double* svcBmaxDyn,
                                    const char* rlIslanded,
                                    const double* rlInitTgtQ, double vLoRobust, double vHiRobust,
                                    int robust,
                                    int swLogSlot, double* swLog, int* swLogCnt, int passId,
                                    const int* psEntry, int nQr,
                                    const char* rlLocalOv, const double* rlLocalVt,
                                    const char* rlNoUnpin, char* unpinBad, const char* rlMonitored,
                                    const int* rlCmpOv) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    if (!rlActive[s]) return;
    // A controller the contingency put in ANOTHER synchronous component is not a controller at all:
    // OLF disables that bus and its outer loops never see it. Its rows here are FROZEN at a flat
    // 1<0, so the q read out of qBuf is a placeholder, not a measurement — and switching on it pins
    // a PV bus that OLF leaves alone, which perturbs the MAIN component through any remote-control
    // or DISTR_Q group the controller belongs to. Measured on LINE-3320-350: the device pinned
    // VL-6432_0, an ISLANDED bus, giving it 20 pinned controllers against OLF's 19 and one extra
    // LOW_VOLTAGE violation at VL-1624.
    if (rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + i]) return;
    // A controller FROZEN by the generator voltage control manager has its voltage control OFF, so it
    // is not one of this loop's controllers at all — OLF iterates buses with voltage control and never
    // reaches it. Leave the type (and the row) to the transformer phase that set it.
    if (limitType[(size_t) s * nrl + i] == GpuQLimitType::FROZEN) return;
    // SVC reactive limits are VOLTAGE-DEPENDENT: LfStaticVarCompensatorImpl evaluates
    // Bmin/Bmax*(v*nomV)^2 at the LIVE bus V on every CPU check (and a pinned bus re-freezes at
    // the moved limit via busesWithUpdatedQLimits). rlMinQ/rlMaxQ hold the base-V aggregates;
    // the per-entry svcB coefficients add the quadratic delta vs the packed base v^2 — exact for
    // the SVC part, identically zero on SVC-less buses.
    bool svcLim = rlSvcBmin[i] != 0.0 || rlSvcBmax[i] != 0.0;
    // ...minus a lost SVC's share, per scenario (svcBminDyn: built only when an outage takes one)
    double svcBMin = svcBminDyn != nullptr ? svcBminDyn[(size_t) s * nrl + i] : rlSvcBmin[i];
    double svcBMax = svcBmaxDyn != nullptr ? svcBmaxDyn[(size_t) s * nrl + i] : rlSvcBmax[i];
    // records one decision of the traced scenario; the branches below call it right where they act
    // swLogSlot >= 0: every check of that ONE scenario. swLogSlot == -2 ("all"): only the checks
    // that DECIDED something, for every scenario - the whole batch's decision set in ~40 records per
    // scenario, which is what two runs have to be compared on when the contingency that flips is not
    // known in advance.
    auto swLogPut = [&](int act, double q, double mn, double mx, double vOwn, double cmpV, double vt) {
        if (swLog == nullptr) return;
        if (swLogSlot == -2 ? act == 0 : s != swLogSlot) return;
        int k = atomicAdd(swLogCnt, 1);
        if (k >= SW_LOG_CAP) return;
        double* r = swLog + (size_t) k * SW_LOG_W;
        r[0] = passId; r[1] = i; r[2] = act; r[3] = q; r[4] = mn; r[5] = mx;
        r[6] = vOwn; r[7] = cmpV; r[8] = vt; r[9] = limitType[(size_t) s * nrl + i];
        r[10] = switchCount[(size_t) s * nrl + i]; r[11] = s;
    };
    // A RE-HOMED voltage control (GpuBatchQualification.reHomedControls): this scenario the bus is a plain PV
    // bus, so its entry — packed as a remote member of the base group — takes the LOCAL path below, at its
    // own voltage target. OLF checks it as the PV bus it now is (BUS-6113_BBS_1: VL-6113_1 pinned at RL#1).
    bool localOv = rlLocalOv != nullptr && rlLocalOv[(size_t) s * nrl + i];
    if (rlRemote != nullptr && rlRemote[i] != GpuVoltageControlRole::LOCAL && !localOv) {
        // REMOTE group member (plan-B3). Unpinned = its DISTR_Q row (mode 1, limitType 0);
        // pinned = mode 4 ("row hosts its own Q balance"). BOTH switch conditions test the
        // PILOT's voltage (rlCmpVrow) against the GROUP target — B0-verified CPU semantics.
        // The FREE member pins the same way (plan-B4): the fill kernels re-pick the pilot-Q
        // carrier by mode-scan over the relocation candidates, and updateDistrQTargetsBatch
        // hands the pilot's Q target to the new carrier's row. When the LAST live member pins,
        // updatePilotVeqBatch flips the pilot's V identity to a power row (control lost) and
        // restores it on the first unpin — the all-pinned corner runs on-device too.
        size_t ridx = (size_t) s * n + rlVrow[i];            // the bus's own V variable: every VOLTAGE read
        // ...and the row its reactive balance and pin live on. The coupler-tree relocation moves it to the parent
        // coupler's DUMMY_Q row — but PER SCENARIO: a contingency that OPENS that coupler sends the child's
        // equations back to its own rows and the host pins the dummy row with a DUMMY_TARGET identity, which is
        // exactly what mode 0 there means (the same test the zero-impedance scatter's stock-placement fallback uses).
        size_t qidx = ridx;
        if (rlQrow != nullptr && rlQrow[i] != rlVrow[i]) {
            size_t relocIdx = (size_t) s * n + rlQrow[i];
            qidx = mode[relocIdx] == GpuRowMode::FIXED_VARIABLE ? ridx : relocIdx;
        }
        size_t rst = (size_t) s * nrl + i;
        size_t rli = cvStride ? rst : (size_t) i;
        // The quadratic delta exists only for SVC entries: rlV2Base is the packed base v², NaN when
        // the extraction ran on a never-solved network (the base load flow's own context) — 0 × NaN
        // poisoned every entry's limits there and nothing ever pinned at base.
        double rdv2 = svcLim ? x[ridx] * x[ridx] - rlV2Base[i] : 0.0;
        double rliMin = rlMinQ[rli] + (svcLim ? svcBMin * rdv2 : 0.0);
        double rliMax = rlMaxQ[rli] + (svcLim ? svcBMax * rdv2 : 0.0);
        // rlRemote 2 = a PILOT that is one of its own controllers: unpinned, its row is the group's
        // voltage IDENTITY (mode 0), not a DISTR_Q share; pinned, its own Q balance (mode 4, the
        // alternate dest). Everything else (q via the remote q pack, the pilot's V as compare row)
        // is a remote member's. As a LOCAL entry its pin landed on mode 1, an EMPTY row.
        int pvForm = rlRemote[i] == GpuVoltageControlRole::PILOT_SELF ? GpuRowMode::FIXED_VARIABLE : GpuRowMode::EQUATION;
        bool enrolledPilot = rlRemote[i] == GpuVoltageControlRole::PILOT_SELF && isEnrolledPilot(psEntry, nQr, i);
        if (limitType[rst] == GpuQLimitType::NONE) {
            if (mode[qidx] != pvForm) return;                // islanded / frozen this scenario
            double q = qBuf[qidx] + rlLoadQ[rli];            // own q via fillRemoteMemberQBatch
            // voltageRemoteControlRobustMode (ReactiveLimitsOuterLoop.checkControllerBus): the criteria
            // below test this REMOTE controller's OWN voltage — the same x row the quadratic SVC delta
            // reads — against the realistic band shrunk by the 1.02 margin.
            // OLF's robust criterion applies to REMOTE controllers only, and "remote" is
            // ReactiveLimitsOuterLoop.isGeneratorRemoteController: the bus's OWN voltage control, not the
            // merged group's. Two classes are local in that respect and both are exempt here:
            //   2 = a pilot that controls its own bus;
            //   3 = a member of a MERGED group whose own (DEPENDENT) control points at ITSELF, which is
            //       every merged member — OLF skips the whole robust block for them (tasks #66; the
            //       device used to freeze rte6515's VL-2669_1..4 with MIN_REALISTIC_V where OLF froze
            //       nothing, replacing the DISTR_Q sharing with mode-4 own-Q balances in the STRUCTURE).
            bool robustRemote = rlRemote[i] != GpuVoltageControlRole::PILOT_SELF && rlRemote[i] != GpuVoltageControlRole::REMOTE_ROBUST_MODE_EXEMPT;
            bool unrealisticLo = robust && robustRemote && x[ridx] < vLoRobust;
            bool unrealisticHi = robust && robustRemote && x[ridx] > vHiRobust;
            if (q < rliMin - tol) {
                savedTarget[rst] = target[qidx];             // the DISTR_Q row target (NOT 0 in general:
                                                             // see updateDistrQTargetsBatch)
                if (!enrolledPilot) {
                    mode[qidx] = GpuRowMode::BUS_TARGET_Q;
                    target[qidx] = rliMin - rlLoadQ[rli];
                }
                limitType[rst] = GpuQLimitType::MIN_Q;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            } else if (q > rliMax + tol) {
                savedTarget[rst] = target[qidx];
                if (!enrolledPilot) {
                    mode[qidx] = GpuRowMode::BUS_TARGET_Q;
                    target[qidx] = rliMax - rlLoadQ[rli];
                }
                limitType[rst] = GpuQLimitType::MAX_Q;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            } else if (unrealisticLo || unrealisticHi) {
                // Q inside its limits but the controller's own voltage is already unrealistic: OLF
                // stops maintaining the REMOTE voltage and freezes the generation at its INITIAL
                // target Q, with a realistic-V limit type (3 = MIN_REALISTIC_V, 4 = MAX_REALISTIC_V)
                // that the switch-back reads like a min/max limit but the Q-limit drift ignores.
                savedTarget[rst] = target[qidx];
                if (!enrolledPilot) {
                    mode[qidx] = GpuRowMode::BUS_TARGET_Q;
                    target[qidx] = rlInitTgtQ[i] - rlLoadQ[rli];
                }
                limitType[rst] = unrealisticLo ? GpuQLimitType::MIN_REALISTIC_V : GpuQLimitType::MAX_REALISTIC_V;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            }
            // OLF also sets the controller's V to a safe 1 pu for the next computation, under BOTH
            // robust criteria. That WRITES the state vector every other entry is still reading here,
            // so it runs as its own pass (rlRobustVResetBatch) once this one has finished.
        } else {                                            // pinned — switch-back tests the PILOT
            // a controller re-homed onto a NEW pilot this scenario reads that pilot's voltage (rlCmpOv)
            double v = x[(size_t) s * n + (rlCmpOv != nullptr && rlCmpOv[rst] >= 0 ? rlCmpOv[rst] : rlCmpVrow[i])];
            double vt = rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
            bool minPin = isMinLimit(limitType[rst]);
            // OLF's checkPqBusWithQLimitType PROPOSES the unpin whenever the pilot's voltage asks for
            // it; the per-bus cap (maxPqPvSwitch) then blocks the switch itself in switchPqPv. Only
            // the ELSE of that test re-freezes a moved limit — so a capped bus that still WANTS to
            // unpin re-freezes on neither arm, while a capped bus that does not must still follow its
            // limit. Gating this whole block on the cap froze such a bus's target at a stale limit:
            // network.xiidm / GBMK, CQEM_0 and CQEM_5 kept -0.181175517 / -0.184950765 where OLF had
            // re-frozen at -0.181173957 / -0.184949017 (a reactive-capability curve moving with the
            // distributed slack), 2.079e-05 kV network-wide.
            bool wantsBack = (!minPin && v > vt) || (minPin && v < vt);
            bool back = wantsBack && switchCount[rst] < maxSwitch;
            if (back && rlNoUnpin != nullptr && rlNoUnpin[rst]) {
                // a pinned member of a re-homed cluster with no enabled controller: OLF would re-activate the
                // cluster's BUS_TARGET_V, which this table cannot host for a lost pilot — reject, CPU replay
                unpinBad[s] = 1;
                back = false;
            }
            if (back) {
                if (!enrolledPilot) {
                    mode[qidx] = pvForm;                     // back to the DISTR_Q share row (the V identity for a pilot)
                    target[qidx] = savedTarget[rst];
                }
                limitType[rst] = GpuQLimitType::NONE;
                switched[s] = 1;
                passSwitched[rst] = 2;
            } else if (!wantsBack && (cvStride || svcLim) && (limitType[rst] == GpuQLimitType::MIN_Q || limitType[rst] == GpuQLimitType::MAX_Q)) {  // curve/SVC limit drift while pinned
                // OLF re-freezes at a moved limit only for MIN_Q/MAX_Q (checkPqBusWithQLimitType tests
                // the type explicitly); a realistic-V freeze holds its initial target Q.
                double lim = limitType[rst] == GpuQLimitType::MIN_Q ? rliMin : rliMax;
                if (!enrolledPilot && fabs(lim - (target[qidx] + rlLoadQ[rli])) > tol) {
                    target[qidx] = lim - rlLoadQ[rli];
                    switched[s] = 1;
                    passSwitched[rst] = 3;
                }
            }
        }
        {
            double qRec = qBuf[qidx] + rlLoadQ[rli];
            int cmpR = rlCmpOv != nullptr && rlCmpOv[rst] >= 0 ? rlCmpOv[rst] : (rlCmpVrow != nullptr ? rlCmpVrow[i] : -1);
            double cmpRec = cmpR >= 0 ? x[(size_t) s * n + cmpR] : x[ridx];
            swLogPut(passSwitched[rst], qRec, rliMin, rliMax, x[ridx], cmpRec,
                     rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]));
        }
        return;
    }
    int vrow = rlVrow[i];
    size_t idx = (size_t) s * n + vrow;
    // The q this check READS is accumulated by fillRemoteMemberQBatch at rlQrow, which for a relocated
    // child is the parent coupler's DUMMY_Q row and NOT its own V row (see the rlQrow comment in
    // ReactiveLimitsPacker: filling the own row "left the check reading ZERO for every relocated member,
    // so it could never pin one"). That pack is STRUCTURAL -- it keeps writing there even in a scenario
    // that pins the dummy to a DUMMY_TARGET identity -- and qBuf is a separate accumulator, so reading it
    // there is valid whatever the row's equation MODE. Only the pin WRITE belongs on the live row, which
    // is idx. Conflating the two reproduced that base failure per scenario: measured on rte6515
    // BUS-6113_BBS_1, VL-6113_1 is re-homed LOCAL (reHomePv rlEntry=969) and OLF pins it MIN_Q at
    // q=-3.231 against a -0.774 limit, while the device read q~0 and never pinned. Its THIRD slack check
    // then measured 0.013280180 where OLF measures 0.006846832 -- straddling the 1 MW gate -- so the
    // device took a distribution OLF does not and reported 1.328 MW more distributed active power: the
    // worst such difference on the whole busbar scope.
    size_t qIdx = (rlQrow != nullptr && rlQrow[i] >= 0 && rlQrow[i] != vrow)
            ? (size_t) s * n + rlQrow[i] : idx;
    size_t st = (size_t) s * nrl + i;
    size_t li = cvStride ? st : (size_t) i;                 // curve limits are per (scenario, entry)
    bool slopeRow = rowSlope[vrow] != 0;                    // SLOPE-SVC bus: its PV form is row mode 2
    double dv2 = svcLim ? x[idx] * x[idx] - rlV2Base[i] : 0.0;   // see the remote branch: NaN base v² otherwise
    double liMin = rlMinQ[li] + (svcLim ? svcBMin * dv2 : 0.0);
    double liMax = rlMaxQ[li] + (svcLim ? svcBMax * dv2 : 0.0);
    // A DEGENERATE reactive range (min >= max) belongs to a bus with no reactive capability of its
    // own: the CONTROLLED (pilot) bus of a remote group carries a generator voltage control but no
    // generators, so bus.getMinQ()/getMaxQ() are a sum over nothing and both are 0. OLF never checks
    // such a bus - ReactiveLimitsOuterLoop walks getControllerElements(GENERATOR) and the pilot is
    // not one of them - while here its q is the bus's own reactive balance, which for a pure transit
    // node is zero up to rounding, so the pin test |q| > tol with tol = convEps (1e-12 under
    // UNIFORM_CRITERIA) fires on the LAST BITS. Measured on network.xiidm: two runs of the same
    // 2000-contingency window differed on 430 reactive-limit decisions and every one was such an
    // entry, q at +-1.2e-12 against limits of exactly 0. The pin flips the pilot's row to its own Q
    // balance for a pass, which perturbs the whole group, and once per ~2000 contingencies it
    // changes a real downstream pin (8.178e-05 kV on the KUMB group).
    // Neither direction is this entry's to decide. The pin is refused (above), and the switch-back
    // is not its either: the pilot's row follows its MEMBERS — updatePilotVeqBatch drops the V
    // identity when the last member pins and gives it back when one unpins, base-lost groups
    // included. Blocking both directions used to leave network.xiidm/LKPH 0.32 kV off because
    // nothing restored the row then; now something does, in the same pass OLF does it, and this
    // entry's own voltage test (a pin type guessed for an empty range) fired a pass late on LNER.
    // ...but ONLY such a bus: rlMonitored bit 2 (ReactiveLimitsPacker) marks an entry that is NOT one of its voltage
    // control's controller elements, the set OLF iterates (LfNetwork.getControllerElements(GENERATOR)). A real
    // generator bus whose capability curve collapses to [0, 0] at its current P is a controller OLF still checks:
    // network.xiidm/FBQV, FBQF_0 pinned MIN_Q at 0 with V 1.009793 under its 1.010221 target - OLF switched it
    // back to PV at the first reactive-limit pass, the device never tested it, and CQEM_0 then pinned a pass late
    // (the one contingency of 24,376 whose outer-loop sequence differed from the 1-thread CPU's).
    if (liMin >= liMax && (rlMonitored == nullptr || (rlMonitored[i] & GpuReactiveLimitsEntryFlag::NOT_CONTROLLER_ELEMENT) != 0)) {
        swLogPut(passSwitched[st], qBuf[qIdx] + rlLoadQ[li], liMin, liMax, x[idx], x[idx],
                 rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]));
        return;
    }
    // A re-homed LOCAL control whose own V row carries a live coupler's ZERO_V keeps its identity (mode 6, on its
    // own V column) and its pinned balance (mode 4) on the RELOCATED row — the parent coupler's DUMMY_Q (tasks #70,
    // VC_REMERGE_RELOC_VROW). The voltage is still read at the bus's own column (idx).
    size_t pinIdx = idx;
    if (localOv && rlQrow != nullptr && rlQrow[i] >= 0 && rlQrow[i] != vrow) {
        size_t r2 = (size_t) s * n + rlQrow[i];
        if (mode[r2] == GpuRowMode::CROSS_FIXED_VARIABLE || (mode[r2] == GpuRowMode::BUS_TARGET_Q && mode[idx] != GpuRowMode::BUS_TARGET_Q)) {
            pinIdx = r2;
        }
    }
    const bool crossRow = pinIdx != idx;
    if ((mode[pinIdx] == GpuRowMode::FIXED_VARIABLE || mode[pinIdx] == GpuRowMode::BUS_TARGET_V_WITH_SLOPE || mode[pinIdx] == GpuRowMode::CROSS_FIXED_VARIABLE) && limitType[st] == GpuQLimitType::NONE) {  // currently PV — check the reactive limits
        double q = qBuf[qIdx] + rlLoadQ[li];
        if (q < liMin - tol) {
            savedTarget[st] = target[pinIdx];                // remember the row target for the switch-back
            savedMode[st] = static_cast<char>(mode[pinIdx]);   // and the PV form (0, 2 for a slope SVC, 6 cross)
            mode[pinIdx] = localOv ? GpuRowMode::BUS_TARGET_Q : GpuRowMode::EQUATION;             // a re-homed member: own Q through its ALTERNATE dest
            target[pinIdx] = liMin - rlLoadQ[li];
            limitType[st] = GpuQLimitType::MIN_Q;
            switchCount[st]++;
            switched[s] = 1;
            passSwitched[st] = 1;
        } else if (q > liMax + tol) {
            savedTarget[st] = target[pinIdx];
            savedMode[st] = static_cast<char>(mode[pinIdx]);
            mode[pinIdx] = localOv ? GpuRowMode::BUS_TARGET_Q : GpuRowMode::EQUATION;             // a re-homed member: own Q through its ALTERNATE dest
            target[pinIdx] = liMax - rlLoadQ[li];
            limitType[st] = GpuQLimitType::MAX_Q;
            switchCount[st]++;
            switched[s] = 1;
            passSwitched[st] = 1;
        }
    } else if (limitType[st] != GpuQLimitType::NONE) {  // pinned at a limit — check switchback
        double v = x[idx];
        double vt = localOv ? rlLocalVt[(size_t) s * nrl + i]
                  : rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
        // OLF never back-switches a slope-SVC bus: checkPqBus runs with
        // canSwitchToPv = !bus.hasGeneratorsWithSlope() ("we don't support switching PQ to PV
        // for bus with one controller with slope") — once pinned, a slope row stays PQ.
        bool maxPin = isMaxLimit(limitType[st]);
        bool minPin = isMinLimit(limitType[st]);
        // the cap bounds the SWITCH, not the limit refresh below (see the remote branch)
        bool wantsBack = !slopeRow
                && ((maxPin && v > vt)                               // maxQ-pinned but v above target
                 || (minPin && v < vt));                             // minQ-pinned but v below target
        bool back = wantsBack && switchCount[st] < maxSwitch;
        if (back) {
            mode[pinIdx] = crossRow ? GpuRowMode::CROSS_FIXED_VARIABLE : GpuRowMode::FIXED_VARIABLE;               // PQ → PV: resume voltage control (cross identity stays cross)
            target[pinIdx] = vt;
            limitType[st] = GpuQLimitType::NONE;
            switched[s] = 1;
            passSwitched[st] = 2;
        } else if (!wantsBack && (cvStride || svcLim) && (limitType[st] == GpuQLimitType::MIN_Q || limitType[st] == GpuQLimitType::MAX_Q)) {  // a realistic-V freeze keeps its target Q
            // Q-limit UPDATE while pinned (OLF checkPqBusWithQLimitType): a CURVE limit that moved
            // (DS moved the gen's P) or an SVC limit that moved (B*v^2 at the live V) re-freezes
            // the pinned Q at the NEW limit when it drifted by more than the mismatch tol.
            // Unreachable with static limits — gated on the channels.
            double lim = limitType[st] == GpuQLimitType::MIN_Q ? liMin : liMax;
            if (fabs(lim - (target[pinIdx] + rlLoadQ[li])) > tol) {
                target[pinIdx] = lim - rlLoadQ[li];
                switched[s] = 1;
                passSwitched[st] = 3;
            }
        }
    }
    swLogPut(passSwitched[st], qBuf[qIdx] + rlLoadQ[li], liMin, liMax, x[idx], x[idx],
             rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]));
}

// voltageRemoteControlRobustMode, second half: "Set V to a safe 1 pu for the next computation".
// ReactiveLimitsOuterLoop does this to every REMOTE voltage controller it switched PV->PQ in this
// check whose OWN voltage is outside the realistic band — whether it switched on a reactive limit
// (the first robust block) or on the realistic-V criterion itself. It is a WRITE to the state
// vector the check kernel reads, so it gets its own pass: one thread per (entry, scenario), reading
// the same pre-pass x, which makes the result independent of the order the entries happen to run in.
// x is the next inner solve's warm start, so this is OLF's controllerBus.setV(1) verbatim.
__global__ void rlRobustVResetBatch(int nrl, int nScenarios, int n, const int* rlVrow, const char* rlRemote,
                                    const char* rlActive, const char* passSwitched, double* x,
                                    double vLoRobust, double vHiRobust) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    if (!rlActive[s]) return;
    // robust mode is remote-controller only, in OLF's sense: not a class-2 pilot and not a class-3
    // merged member whose own control is local (both are exempt from the gate that sets 3/4 at all)
    if (rlRemote == nullptr || rlRemote[i] != GpuVoltageControlRole::REMOTE) return;
    if (passSwitched[(size_t) s * nrl + i] != 1) return;     // switched PV -> PQ in THIS check
    size_t idx = (size_t) s * n + rlVrow[i];
    if (x[idx] < vLoRobust || x[idx] > vHiRobust) {
        x[idx] = 1.0;
    }
}

// OLF's GLOBAL forced-PV election (ReactiveLimitsOuterLoop.switchPvPq): when EVERY bus still PV
// would switch PV→PQ in this check (remainingPvBusCount == 0), the strongest candidate stays PV —
// highest controlled-bus nominal V, then highest bus target P (LIVE: the DistributedSlack may have
// moved it), then lowest id. One thread per scenario, after the pin kernel: it UNDOES the elected
// pin (row mode and target back, limit type and switch count back) and re-derives the scenario's
// "changed" flag from what the pass still did. Before this the device pinned every candidate and a
// per-scenario DETECTOR routed the scenario to the CPU (1.9 kV wrong on the forced-PV network).
__global__ void rlForcedPvElectBatch(int nrl, int nScenarios, int n,
                                     const int* rlVrow, const char* rlRemote, const char* rlIslanded,
                                     const double* nomV, const int* pRow, const double* staticP, const int* idRank,
                                     char* limitType, char* switchCount, char* passSwitched, const char* savedMode,
                                     const double* savedTarget, int* mode, double* target,
                                     const char* rlActive, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!rlActive[s]) return;
    int remaining = 0;
    int cand = 0;
    int best = -1;
    double bestV = -1.0;
    double bestP = -1e300;
    int bestId = 0x7fffffff;
    for (int i = 0; i < nrl; i++) {
        size_t st = (size_t) s * nrl + i;
        if (rlIslanded != nullptr && rlIslanded[st]) continue;   // not a controller in this component
        bool remote = rlRemote != nullptr && rlRemote[i] != GpuVoltageControlRole::LOCAL;
        int m = mode[(size_t) s * n + rlVrow[i]];
        if (passSwitched[st] == 1) {
            cand++;
            double v = nomV[i];
            double p = pRow[i] >= 0 ? target[(size_t) s * n + pRow[i]] : staticP[i];
            bool better = best < 0 || v > bestV
                    || (v == bestV && (p > bestP || (p == bestP && idRank[i] < bestId)));
            if (better) {
                best = i;
                bestV = v;
                bestP = p;
                bestId = idRank[i];
            }
        } else if (limitType[st] == GpuQLimitType::NONE && (rlRemote != nullptr && rlRemote[i] == GpuVoltageControlRole::PILOT_SELF ? m == GpuRowMode::FIXED_VARIABLE
                                          : remote ? m == GpuRowMode::EQUATION : (m == GpuRowMode::FIXED_VARIABLE || m == GpuRowMode::BUS_TARGET_V_WITH_SLOPE))) {
            remaining++;                                     // still PV after this pass
        }
    }
    if (cand == 0 || remaining > 0 || best < 0) return;
    size_t st = (size_t) s * nrl + best;
    size_t idx = (size_t) s * n + rlVrow[best];
    mode[idx] = (rlRemote != nullptr && rlRemote[best] == GpuVoltageControlRole::PILOT_SELF) ? 0
              : (rlRemote != nullptr && rlRemote[best] != GpuVoltageControlRole::LOCAL) ? 1 : savedMode[st];
    target[idx] = savedTarget[st];
    limitType[st] = GpuQLimitType::NONE;
    switchCount[st]--;
    passSwitched[st] = 0;
    char any = 0;
    for (int i = 0; i < nrl && !any; i++) {
        any = passSwitched[(size_t) s * nrl + i] != 0;
    }
    switched[s] = any;
}

// Warm-continuation reconcile: a SEEDED pinned controller (limitType != 0) must also carry its
// consequences — the BUS_V row flips to the reactive equation (mode 1) with the pinned Q target —
// else the solve holds the PV setpoint while the RL bookkeeping thinks the bus is at a limit.
// A PINNED pilot that is one of its own controllers (rlRemote 2) while a member of the same group
// (same compare row = the pilot's V) still controls: OLF then keeps the pilot's V identity AND
// activates the pilot's own Q balance, hosting it on the reference member's row - a relocation this
// group has no table for on the device. Flag the scenario; the host rejects it to the CPU replay.
// The other mixed state (members pinned, pilot controlling) is an ordinary layout: pilot identity,
// members at their own Q. One thread per (scenario, class-2 entry).
__global__ void rlPilotSelfMixedGate(int nrl, int nScenarios, const char* rlRemote, const int* rlCmpVrow,
                                     const char* limitType, const char* rlIslanded, char* bad,
                                     const int* psEntry, int nQr,
                                int nDistrQMembers, const int* distrQMemberReactiveLimitEntry, const char* distrQSubgroup) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nrl * nScenarios) return;
    int i = t % nrl;
    int s = t / nrl;
    if (rlRemote[i] != GpuVoltageControlRole::PILOT_SELF || limitType[(size_t) s * nrl + i] == GpuQLimitType::NONE) return;  // only a PINNED pilot
    // ...that is still ON this component: an islanded pilot keeps its base limit type but has no pin and no
    // balance (OLF removed the bus), and an islanded member is not "still controlling" either.
    if (rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + i]) return;
    if (isEnrolledPilot(psEntry, nQr, i)) return;            // enrolled (rotation): the carrier hosts its balance
    for (int j = 0; j < nrl; j++) {                          // ... with a member still controlling
        if (j != i && rlCmpVrow[j] == rlCmpVrow[i] && rlRemote[j] != GpuVoltageControlRole::LOCAL
                && limitType[(size_t) s * nrl + j] == GpuQLimitType::NONE
                && !(rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + j])) {
            if (distrQSubgroup != nullptr && distrQMemberReactiveLimitEntry != nullptr) {   // a member of ANOTHER subgroup is not this
                bool other = false;                                  // pilot's controller any more (tasks #71)
                for (int m = 0; m < nDistrQMembers; m++) {
                    if (distrQMemberReactiveLimitEntry[m] == j && distrQSubgroup[(size_t) s * nDistrQMembers + m] != 0) { other = true; break; }
                }
                if (other) continue;
            }
            bad[s] = 1;
            return;
        }
    }
}

// SecondaryVoltageControl's tryToReEnableHelpfulControllerBuses, applied to the DEVICE pin state at a
// driver pause: a zone whose controllers are only PARTLY at a reactive limit un-pins the limited ones so
// they can help the zone realign. OLF does it with setGeneratorVoltageControlEnabledAndRecomputeTargetQ
// (back to PV) + setQLimitType(null); here that is the row's saved PV form, the zone's CURRENT voltage
// target, and limitType 0. The host orchestrator got this for free by re-seeding rlStateIn on its own
// re-solve — a RESUME skips the RL seeding, so without this channel the un-pin never reached the device.
__global__ void rlUnpinBatch(int nPairs, int nS, int n, int nrl, const int* pairs, const double* newTarget,
                             const int* rlVrow, char* limitType, const char* savedMode, int* mode, double* target) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nPairs) return;
    int s = pairs[2 * k];
    int i = pairs[2 * k + 1];
    if (s < 0 || s >= nS || i < 0 || i >= nrl) return;
    size_t st = (size_t) s * nrl + i;
    if (limitType[st] == GpuQLimitType::NONE) return;  // not pinned: nothing to give back
    // A member FROZEN by the generator voltage control manager is not a reactive-limit pin: giving it
    // back would restore a V identity on a row that holds the group's DISTR_Q share, and would undo a
    // decision the transformer phase owns and will undo itself at COMPLETE.
    if (limitType[st] == GpuQLimitType::FROZEN) return;
    size_t idx = (size_t) s * n + rlVrow[i];
    char sm = savedMode[st];
    mode[idx] = (sm == GpuRowMode::BUS_TARGET_V_WITH_SLOPE) ? GpuRowMode::BUS_TARGET_V_WITH_SLOPE : GpuRowMode::FIXED_VARIABLE;                          // the PV form saved at pin time (2 = slope SVC)
    target[idx] = newTarget[k];
    limitType[st] = GpuQLimitType::NONE;
}

// One check of ReactiveLimitsOuterLoop on every live scenario (GpuOuterLoopKind::REACTIVE_LIMITS): the curve limits at the current
// targets, the PV->PQ pins and releases, robust mode's realistic-voltage pins and the forced-PV election.
void rlCheck(BatchSolverContext* ctx, DriverPass& drv, int loopKind) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& active = drv.active;
    std::vector<char>& live = drv.live;
    BatchResult& res = drv.res;
    const bool& useDynRlLimits = drv.useDynRlLimits;
    const double& REACTIVE_LIMIT_TOL = drv.reactiveLimitTol;
    const int& RL_MAX_SWITCH = drv.rlMaxSwitch;
    const int& swLogSlot = drv.swLogSlot;
    const int& skipRlPhase = drv.skipRlPhase;
    const bool& svcCoefDyn = drv.svcCoefDyn;
    const int& nRlVtOv = drv.nRlVtOv;
    const int& nRlQOv = drv.nRlQOv;
    const int& nCvOv = drv.nCvOv;
    DeviceBuf& dRlVtOvSlotIdx = drv.dRlVtOvSlotIdx;
    DeviceBuf& dRlVtOvVal = drv.dRlVtOvVal;
    DeviceBuf& dRlQOvSlotEntry = drv.dRlQOvSlotEntry;
    DeviceBuf& dRlQOvMinMax = drv.dRlQOvMinMax;
    DeviceBuf& dRlQOvPtr = drv.dRlQOvPtr;
    DeviceBuf& dRlQOvList = drv.dRlQOvList;
    DeviceBuf& dCvOvPtr = drv.dCvOvPtr;
    DeviceBuf& dCvOvEntry = drv.dCvOvEntry;
    DeviceBuf& dCvOvVal = drv.dCvOvVal;
    if (loopKind == GpuOuterLoopKind::REACTIVE_LIMITS && ctx->nrl > 0 && !skipRlPhase) {                  // skipRlPhase: a monitoring interleave solve — the
        if (useDynRlLimits) {                            // curve limits at the CURRENT (post-DS) targets,
            rlCurveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, ctx->ncv, n, ctx->cvExtrapolate,
                    ctx->dCvIdx.as<int>(), ctx->dCvIn.as<double>(), ctx->dCvPts.as<double>(),
                    ctx->dCvPtr.as<int>(), ctx->dCvList.as<int>(), cvHvdcView(ctx),
                    ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(), ctx->dRlLoadQ.as<double>(),
                    ctx->dTarget.as<double>(),
                    nRlQOv, nRlQOv > 0 ? dRlQOvSlotEntry.as<int>() : nullptr,
                    nRlQOv > 0 ? dRlQOvMinMax.as<double>() : nullptr,
                    nRlQOv > 0 ? dRlQOvPtr.as<int>() : nullptr, nRlQOv > 0 ? dRlQOvList.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvPtr.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvEntry.as<int>() : nullptr,
                    nCvOv > 0 ? dCvOvVal.as<double>() : nullptr,
                    ctx->ngen > 0 ? ctx->dGenCum.as<double>() : nullptr,
                    ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->hasDgAnchor ? ctx->dDgAnchor.as<double>() : nullptr, ctx->hasDgStart ? ctx->dDgStart.as<double>() : nullptr, ctx->ngen,
                    ctx->nds > 0 ? ctx->dDsPhiRow.as<int>() : nullptr,
                    ctx->nDsRowOv > 0 ? ctx->dDsRowOv.as<int>() : nullptr, ctx->nds,
                    ctx->ngen > 0 ? ctx->dgBus.as<int>() : nullptr,
                    ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>(),
                    ctx->dRlLoadQDyn.as<double>());
            launchForcedTargetQLimits(ctx, nS, n);
        }
        CUDA_CHECK(cudaMemsetAsync(ctx->dRlPassSwitched.p, 0, static_cast<size_t>(ctx->nrl) * nS, ctx->stream));
        reactiveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(
                ctx->nSvc > 0 ? ctx->dSvcRlVt.as<double>() : nullptr,
                ctx->nrl, nS, n, REACTIVE_LIMIT_TOL, RL_MAX_SWITCH,
                ctx->dRlVrow.as<int>(),
                ctx->dRlQrow.p ? ctx->dRlQrow.as<int>() : nullptr,
                useDynRlLimits ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                useDynRlLimits ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                useDynRlLimits ? ctx->dRlLoadQDyn.as<double>() : ctx->dRlLoadQ.as<double>(),
                ctx->dRlVtarget.as<double>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
                ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                ctx->dRlLimitType.as<char>(), ctx->dRlSwitchCount.as<char>(), ctx->dRlSwitched.as<char>(),
                ctx->dRlPassSwitched.as<char>(), ctx->dRlSavedMode.as<char>(),
                nRlVtOv, nRlVtOv > 0 ? dRlVtOvSlotIdx.as<int>() : nullptr,
                nRlVtOv > 0 ? dRlVtOvVal.as<double>() : nullptr,
                ctx->dRowSlope.as<double>(), ctx->dRlSavedTarget.as<double>(), useDynRlLimits ? 1 : 0,
                ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                ctx->nRlRemote > 0 ? ctx->dRlCmpVrow.as<int>() : nullptr,
                ctx->dRlSvcBmin.as<double>(), ctx->dRlSvcBmax.as<double>(), ctx->dRlV2Base.as<double>(),
                svcCoefDyn ? ctx->dRlSvcBminDyn.as<double>() : nullptr, svcCoefDyn ? ctx->dRlSvcBmaxDyn.as<double>() : nullptr,
                ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
                ctx->rlRobust ? ctx->dRlInitTgtQ.as<double>() : nullptr,
                ctx->rlVLoRobust, ctx->rlVHiRobust, ctx->rlRobust,
                swLogSlot, ctx->dSwLog.p ? ctx->dSwLog.as<double>() : nullptr,
                ctx->dSwLogCnt.p ? ctx->dSwLogCnt.as<int>() : nullptr, outer,
                psEntryPtr(ctx), ctx->st->nQr,
                ctx->nRlLocal > 0 ? ctx->dRlLocalOv.as<char>() : nullptr,
                ctx->nRlLocal > 0 ? ctx->dRlLocalVt.as<double>() : nullptr,
                ctx->nRlNoUnpin > 0 ? ctx->dRlNoUnpin.as<char>() : nullptr, ctx->nRlNoUnpin > 0 ? ctx->dRlUnpinBad.as<char>() : nullptr,
                ctx->dRlMonitored.p ? ctx->dRlMonitored.as<char>() : nullptr,
                ctx->nRlCmp > 0 ? ctx->dRlCmpOv.as<int>() : nullptr);
        if (ctx->nRlPilotSelf > 0) {
            // a pilot that is its own controller pinned while a member of its group is not (or
            // the reverse): the device has no row for OLF's layout of that state - CPU replay
            if (!ctx->dRlMixedBad.p) ctx->dRlMixedBad.alloc(nS * sizeof(char));
            CUDA_CHECK(cudaMemsetAsync(ctx->dRlMixedBad.p, 0, nS * sizeof(char), ctx->stream));
            rlPilotSelfMixedGate<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS,
                    ctx->dRlRemote.as<char>(), ctx->dRlCmpVrow.as<int>(), ctx->dRlLimitType.as<char>(),
                    ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
                    ctx->dRlMixedBad.as<char>(),
                    psEntryPtr(ctx), ctx->st->nQr,
                ctx->nDistrQMembers, ctx->dDistrQMemberReactiveLimitEntry.p != nullptr ? ctx->dDistrQMemberReactiveLimitEntry.as<int>() : nullptr, ctx->hasDistrQSubgroups ? ctx->dDistrQSubgroup.as<char>() : nullptr);
            std::vector<char> mixed(nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(mixed.data(), ctx->dRlMixedBad.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (mixed[s] && live[s]) {
                    traceReject(s, "pilot-self-mixed-pin", outer, 0);
                    res.status[s] = GpuAcSolverStatus::REJECTED;
                    active[s] = 0;
                    live[s] = 0;
                }
            }
        }
        if (ctx->nRlNoUnpin > 0) {
            std::vector<char> ub(nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(ub.data(), ctx->dRlUnpinBad.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (ub[s] && live[s]) {
                    traceReject(s, "unpin-unrepresentable", outer, 0);
                    res.status[s] = GpuAcSolverStatus::REJECTED;
                    active[s] = 0;
                    live[s] = 0;
                }
            }
        }
        if (ctx->rlRobust && ctx->nRlRemote > 0) {       // "set V to a safe 1 pu", as its own pass
            rlRobustVResetBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(
                    ctx->nrl, nS, n, ctx->dRlVrow.as<int>(), ctx->dRlRemote.as<char>(),
                    ctx->dRlActive.as<char>(), ctx->dRlPassSwitched.as<char>(), ctx->dX.as<double>(),
                    ctx->rlVLoRobust, ctx->rlVHiRobust);
        }
        if (ctx->hasRlElection && ctx->allPvPinnable) {  // the rule can only fire when every PV is pinnable
            rlForcedPvElectBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n,
                    ctx->dRlVrow.as<int>(),
                    ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                    ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
                    ctx->dRlElectNomV.as<double>(), ctx->dRlElectPRow.as<int>(),
                    ctx->dRlElectStaticP.as<double>(), ctx->dRlElectIdRank.as<int>(),
                    ctx->dRlLimitType.as<char>(), ctx->dRlSwitchCount.as<char>(), ctx->dRlPassSwitched.as<char>(),
                    ctx->dRlSavedMode.as<char>(), ctx->dRlSavedTarget.as<double>(),
                    ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>());
        }
        // The pilot's V-identity flip/restore runs BEFORE the DISTR_Q targets: the carrier hand-off picks the
        // carrier with relocChoose, which reads the pilot's own row (candidate 1 of a pilot-self group).
        // Measured with the order reversed (rte6515 busbar base, group VL-5997): the member that just
        // switched back was handed a target of 0 while the pilot's row was still flipped, its row then
        // hosted the pilot's balance against that 0, and it re-pinned every other pass until maxSwitch.
        // A pilot that LOST its own generator (GenOutage.flipsPilotRowWhenAllPinned) is laid out here, BEFORE
        // the pilot-self flip below: that kernel's "control lost" layout puts the pilot's row in mode 1, which
        // for a pilot-that-is-its-own-controller is an EMPTY row (no q term is routed to it - the RL kernel's
        // own note), so the voltage stayed exactly at the identity's value while OLF solved the pilot's
        // BUS_TARGET_Q = 0 there. Measured on rte6515 GEN-1522: VL-1522_0 at 1.122 (the target) against OLF's
        // 1.0811, 3.2 kV at VL-1525 and 1.98e-2 MW of distributed power. Mode 4 - the own-Q balance the
        // member ALTERNATE dest routes the pilot's q to - is what the outage composer armed; run first, the
        // pilot-self kernel then finds the row already off its identity and leaves it alone.
        if (ctx->nPilotFlip > 0) {                      // pilot stopped controlling AND every survivor pinned
            pilotRowAllPinnedBatch<<<grid(ctx->nPilotFlip), BLOCK, 0, ctx->stream>>>(
                    ctx->nPilotFlip, nS, n, ctx->nrl,
                    ctx->dPfSlot.as<int>(), ctx->dPfRow.as<int>(), ctx->dPfQTarget.as<double>(),
                    ctx->dPfOff.as<int>(), ctx->dPfCount.as<int>(), ctx->dPfRlPool.as<int>(),
                    ctx->dRlLimitType.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                    ctx->dPfFlipped.as<char>(), ctx->dPfSavedVT.as<double>(),
                    ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>());
        }
        if (ctx->hasDistrQRelocationGroups && ctx->st->nQr > 0) {    // all-pinned corner: pilot vEq flip/restore
            launchPilotVeqInner(ctx, drv);
        }
        if (ctx->nRlRemote > 0 && ctx->nDistrQMembers > 0) {
            // keep the survivors' DISTR_Q row targets in step with the pin states (plan-B3)
            launchDistrQTargetsInner(ctx, drv);
        }
    }
}

}  // namespace olfgpu
