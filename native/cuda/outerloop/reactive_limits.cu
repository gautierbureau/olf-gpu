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
#include "../fixedmatrix/reactive_relocation.cuh"
#include "../fixedmatrix/cross_identity.cuh"
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

__global__ void reactiveLimitsBatch(const double* secondaryVoltageControlReactiveLimitTargetV, int nReactiveLimitEntries, int nScenarios, int n, double tol, int maxSwitch,
                                    const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow, const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ,
                                    const double* reactiveLimitLoadQ, const double* reactiveLimitTargetV, const double* qBuf,
                                    const double* x, const char* outerLoopCheckActive, int* mode, double* target,
                                    char* limitType, char* switchCount, char* switched,
                                    char* passSwitched, char* savedMode,
                                    int nReactiveLimitTargetVOverrides, const int* reactiveLimitTargetVOverrideScenarioEntry, const double* reactiveLimitTargetVOverrideValue,
                                    const double* rowSlope, double* savedTarget, int perScenarioReactiveLimits,
                                    const char* reactiveLimitVoltageControlRole, const int* reactiveLimitComparedVoltageRow,
                                    const double* reactiveLimitSvcMinB, const double* reactiveLimitSvcMaxB, const double* reactiveLimitBaseVSquared,
                                    const double* svcBminDyn, const double* svcBmaxDyn,
                                    const char* reactiveLimitIslanded,
                                    const double* reactiveLimitInitialTargetQ, double vLoRobust, double vHiRobust,
                                    int robust,
                                    int reactiveLimitsDecisionLogScenario, double* reactiveLimitsDecisionLog, int* reactiveLimitsDecisionLogCount, int passId,
                                    const int* pilotSelfReactiveLimitEntry, int nReactiveRelocationGroups,
                                    const char* reactiveLimitLocalOverride, const double* reactiveLimitLocalTargetV,
                                    const char* reactiveLimitPqPvSwitchForbidden, char* unpinBad, const char* reactiveLimitFlags,
                                    const int* reactiveLimitComparedVoltageRowOverride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nReactiveLimitEntries * nScenarios) return;
    int i = tid % nReactiveLimitEntries;
    int s = tid / nReactiveLimitEntries;
    if (!outerLoopCheckActive[s]) return;
    // A controller the contingency put in ANOTHER synchronous component is not a controller at all:
    // OLF disables that bus and its outer loops never see it. Its rows here are FROZEN at a flat
    // 1<0, so the q read out of qBuf is a placeholder, not a measurement — and switching on it pins
    // a PV bus that OLF leaves alone, which perturbs the MAIN component through any remote-control
    // or DISTR_Q group the controller belongs to. Measured on LINE-3320-350: the device pinned
    // VL-6432_0, an ISLANDED bus, giving it 20 pinned controllers against OLF's 19 and one extra
    // LOW_VOLTAGE violation at VL-1624.
    if (reactiveLimitIslanded != nullptr && reactiveLimitIslanded[(size_t) s * nReactiveLimitEntries + i]) return;
    // A controller FROZEN by the generator voltage control manager has its voltage control OFF, so it
    // is not one of this loop's controllers at all — OLF iterates buses with voltage control and never
    // reaches it. Leave the type (and the row) to the transformer phase that set it.
    if (limitType[(size_t) s * nReactiveLimitEntries + i] == GpuQLimitType::FROZEN) return;
    // SVC reactive limits are VOLTAGE-DEPENDENT: LfStaticVarCompensatorImpl evaluates
    // Bmin/Bmax*(v*nomV)^2 at the LIVE bus V on every CPU check (and a pinned bus re-freezes at
    // the moved limit via busesWithUpdatedQLimits). reactiveLimitMinQ/reactiveLimitMaxQ hold the base-V aggregates;
    // the per-entry svcB coefficients add the quadratic delta vs the packed base v^2 — exact for
    // the SVC part, identically zero on SVC-less buses.
    bool svcLim = reactiveLimitSvcMinB[i] != 0.0 || reactiveLimitSvcMaxB[i] != 0.0;
    // ...minus a lost SVC's share, per scenario (svcBminDyn: built only when an outage takes one)
    double svcBMin = svcBminDyn != nullptr ? svcBminDyn[(size_t) s * nReactiveLimitEntries + i] : reactiveLimitSvcMinB[i];
    double svcBMax = svcBmaxDyn != nullptr ? svcBmaxDyn[(size_t) s * nReactiveLimitEntries + i] : reactiveLimitSvcMaxB[i];
    // records one decision of the traced scenario; the branches below call it right where they act
    // reactiveLimitsDecisionLogScenario >= 0: every check of that ONE scenario. reactiveLimitsDecisionLogScenario == -2 ("all"): only the checks
    // that DECIDED something, for every scenario - the whole batch's decision set in ~40 records per
    // scenario, which is what two runs have to be compared on when the contingency that flips is not
    // known in advance.
    auto putReactiveLimitsDecision = [&](int decision, double q, double minQ, double maxQ, double vOwn, double cmpV, double targetV) {
        if (reactiveLimitsDecisionLog == nullptr) return;
        if (reactiveLimitsDecisionLogScenario == -2 ? decision == 0 : s != reactiveLimitsDecisionLogScenario) return;
        int k = atomicAdd(reactiveLimitsDecisionLogCount, 1);
        if (k >= REACTIVE_LIMITS_DECISION_LOG_CAPACITY) return;
        double* r = reactiveLimitsDecisionLog + (size_t) k * REACTIVE_LIMITS_DECISION_LOG_WIDTH;
        r[0] = passId; r[1] = i; r[2] = decision; r[3] = q; r[4] = minQ; r[5] = maxQ;
        r[6] = vOwn; r[7] = cmpV; r[8] = targetV; r[9] = limitType[(size_t) s * nReactiveLimitEntries + i];
        r[10] = switchCount[(size_t) s * nReactiveLimitEntries + i]; r[11] = s;
    };
    // A RE-MERGED voltage control (GpuBatchQualification.reMergedControls): this scenario the bus is a plain PV
    // bus, so its entry — packed as a remote member of the base group — takes the LOCAL path below, at its
    // own voltage target. OLF checks it as the PV bus it now is (BUS-6113_BBS_1: VL-6113_1 pinned at RL#1).
    bool localOv = reactiveLimitLocalOverride != nullptr && reactiveLimitLocalOverride[(size_t) s * nReactiveLimitEntries + i];
    if (reactiveLimitVoltageControlRole != nullptr && reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::LOCAL && !localOv) {
        // REMOTE group member (plan-B3). Unpinned = its DISTR_Q row (mode 1, limitType 0);
        // pinned = mode 4 ("row hosts its own Q balance"). BOTH switch conditions test the
        // PILOT's voltage (reactiveLimitComparedVoltageRow) against the GROUP target — B0-verified CPU semantics.
        // The FREE member pins the same way (plan-B4): the fill kernels re-pick the pilot-Q
        // carrier by mode-scan over the relocation candidates, and updateDistrQTargetsBatch
        // hands the pilot's Q target to the new carrier's row. When the LAST live member pins,
        // updatePilotTargetVEquationBatch flips the pilot's V identity to a power row (control lost) and
        // restores it on the first unpin — the all-pinned corner runs on-device too.
        size_t ridx = (size_t) s * n + reactiveLimitVoltageRow[i];            // the bus's own V variable: every VOLTAGE read
        // ...and the row its reactive balance and pin live on. The coupler-tree relocation moves it to the parent
        // coupler's DUMMY_Q row — but PER SCENARIO: a contingency that OPENS that coupler sends the child's
        // equations back to its own rows and the host pins the dummy row with a DUMMY_TARGET identity, which is
        // exactly what mode 0 there means (the same test the zero-impedance scatter's stock-placement fallback uses).
        size_t qidx = ridx;
        if (reactiveLimitReactivePowerRow != nullptr && reactiveLimitReactivePowerRow[i] != reactiveLimitVoltageRow[i]) {
            size_t relocatedQRowIndex = (size_t) s * n + reactiveLimitReactivePowerRow[i];
            qidx = mode[relocatedQRowIndex] == GpuRowMode::FIXED_VARIABLE ? ridx : relocatedQRowIndex;
        }
        size_t rst = (size_t) s * nReactiveLimitEntries + i;
        size_t remoteLimitIndex = perScenarioReactiveLimits ? rst : (size_t) i;
        // The quadratic delta exists only for SVC entries: reactiveLimitBaseVSquared is the packed base v², NaN when
        // the extraction ran on a never-solved network (the base load flow's own context) — 0 × NaN
        // poisoned every entry's limits there and nothing ever pinned at base.
        double rdv2 = svcLim ? x[ridx] * x[ridx] - reactiveLimitBaseVSquared[i] : 0.0;
        double remoteMinQ = reactiveLimitMinQ[remoteLimitIndex] + (svcLim ? svcBMin * rdv2 : 0.0);
        double remoteMaxQ = reactiveLimitMaxQ[remoteLimitIndex] + (svcLim ? svcBMax * rdv2 : 0.0);
        // reactiveLimitVoltageControlRole 2 = a PILOT that is one of its own controllers: unpinned, its row is the group's
        // voltage IDENTITY (mode 0), not a DISTR_Q share; pinned, its own Q balance (mode 4, the
        // alternate dest). Everything else (q via the remote q pack, the pilot's V as compare row)
        // is a remote member's. As a LOCAL entry its pin landed on mode 1, an EMPTY row.
        int pvForm = reactiveLimitVoltageControlRole[i] == GpuVoltageControlRole::PILOT_SELF ? GpuRowMode::FIXED_VARIABLE : GpuRowMode::EQUATION;
        bool enrolledPilot = reactiveLimitVoltageControlRole[i] == GpuVoltageControlRole::PILOT_SELF && isEnrolledPilot(pilotSelfReactiveLimitEntry, nReactiveRelocationGroups, i);
        if (limitType[rst] == GpuQLimitType::NONE) {
            if (mode[qidx] != pvForm) return;                // islanded / frozen this scenario
            double q = qBuf[qidx] + reactiveLimitLoadQ[remoteLimitIndex];            // own q via fillRemoteMemberQBatch
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
            bool robustRemote = reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::PILOT_SELF && reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::REMOTE_ROBUST_MODE_EXEMPT;
            bool unrealisticLo = robust && robustRemote && x[ridx] < vLoRobust;
            bool unrealisticHi = robust && robustRemote && x[ridx] > vHiRobust;
            if (q < remoteMinQ - tol) {
                savedTarget[rst] = target[qidx];             // the DISTR_Q row target (NOT 0 in general:
                                                             // see updateDistrQTargetsBatch)
                if (!enrolledPilot) {
                    mode[qidx] = GpuRowMode::BUS_TARGET_Q;
                    target[qidx] = remoteMinQ - reactiveLimitLoadQ[remoteLimitIndex];
                }
                limitType[rst] = GpuQLimitType::MIN_Q;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            } else if (q > remoteMaxQ + tol) {
                savedTarget[rst] = target[qidx];
                if (!enrolledPilot) {
                    mode[qidx] = GpuRowMode::BUS_TARGET_Q;
                    target[qidx] = remoteMaxQ - reactiveLimitLoadQ[remoteLimitIndex];
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
                    target[qidx] = reactiveLimitInitialTargetQ[i] - reactiveLimitLoadQ[remoteLimitIndex];
                }
                limitType[rst] = unrealisticLo ? GpuQLimitType::MIN_REALISTIC_V : GpuQLimitType::MAX_REALISTIC_V;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            }
            // OLF also sets the controller's V to a safe 1 pu for the next computation, under BOTH
            // robust criteria. That WRITES the state vector every other entry is still reading here,
            // so it runs as its own pass (reactiveLimitsRobustVoltageResetBatch) once this one has finished.
        } else {                                            // pinned — switch-back tests the PILOT
            // a controller re-merged onto a NEW pilot this scenario reads that pilot's voltage (reactiveLimitComparedVoltageRowOverride)
            double v = x[(size_t) s * n + (reactiveLimitComparedVoltageRowOverride != nullptr && reactiveLimitComparedVoltageRowOverride[rst] >= 0 ? reactiveLimitComparedVoltageRowOverride[rst] : reactiveLimitComparedVoltageRow[i])];
            double targetV = reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]);
            bool minPin = isMinLimit(limitType[rst]);
            // OLF's checkPqBusWithQLimitType PROPOSES the unpin whenever the pilot's voltage asks for
            // it; the per-bus cap (maxPqPvSwitch) then blocks the switch itself in switchPqPv. Only
            // the ELSE of that test re-freezes a moved limit — so a capped bus that still WANTS to
            // unpin re-freezes on neither arm, while a capped bus that does not must still follow its
            // limit. Gating this whole block on the cap froze such a bus's target at a stale limit:
            // network.xiidm / GBMK, CQEM_0 and CQEM_5 kept -0.181175517 / -0.184950765 where OLF had
            // re-frozen at -0.181173957 / -0.184949017 (a reactive-capability curve moving with the
            // distributed slack), 2.079e-05 kV network-wide.
            bool wantsBack = (!minPin && v > targetV) || (minPin && v < targetV);
            bool back = wantsBack && switchCount[rst] < maxSwitch;
            if (back && reactiveLimitPqPvSwitchForbidden != nullptr && reactiveLimitPqPvSwitchForbidden[rst]) {
                // a pinned member of a re-merged cluster with no enabled controller: OLF would re-activate the
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
            } else if (!wantsBack && (perScenarioReactiveLimits || svcLim) && (limitType[rst] == GpuQLimitType::MIN_Q || limitType[rst] == GpuQLimitType::MAX_Q)) {  // curve/SVC limit drift while pinned
                // OLF re-freezes at a moved limit only for MIN_Q/MAX_Q (checkPqBusWithQLimitType tests
                // the type explicitly); a realistic-V freeze holds its initial target Q.
                double lim = limitType[rst] == GpuQLimitType::MIN_Q ? remoteMinQ : remoteMaxQ;
                if (!enrolledPilot && fabs(lim - (target[qidx] + reactiveLimitLoadQ[remoteLimitIndex])) > tol) {
                    target[qidx] = lim - reactiveLimitLoadQ[remoteLimitIndex];
                    switched[s] = 1;
                    passSwitched[rst] = 3;
                }
            }
        }
        {
            double qRec = qBuf[qidx] + reactiveLimitLoadQ[remoteLimitIndex];
            int cmpR = reactiveLimitComparedVoltageRowOverride != nullptr && reactiveLimitComparedVoltageRowOverride[rst] >= 0 ? reactiveLimitComparedVoltageRowOverride[rst] : (reactiveLimitComparedVoltageRow != nullptr ? reactiveLimitComparedVoltageRow[i] : -1);
            double cmpRec = cmpR >= 0 ? x[(size_t) s * n + cmpR] : x[ridx];
            putReactiveLimitsDecision(passSwitched[rst], qRec, remoteMinQ, remoteMaxQ, x[ridx], cmpRec,
                     reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]));
        }
        return;
    }
    int vrow = reactiveLimitVoltageRow[i];
    size_t idx = (size_t) s * n + vrow;
    // The q this check READS is accumulated by fillRemoteMemberQBatch at reactiveLimitReactivePowerRow, which for a relocated
    // child is the parent coupler's DUMMY_Q row and NOT its own V row (see the reactiveLimitReactivePowerRow comment in
    // ReactiveLimitsPacker: filling the own row "left the check reading ZERO for every relocated member,
    // so it could never pin one"). That pack is STRUCTURAL -- it keeps writing there even in a scenario
    // that pins the dummy to a DUMMY_TARGET identity -- and qBuf is a separate accumulator, so reading it
    // there is valid whatever the row's equation MODE. Only the pin WRITE belongs on the live row, which
    // is idx. Conflating the two reproduced that base failure per scenario: measured on rte6515
    // BUS-6113_BBS_1, VL-6113_1 is re-merged LOCAL (reMergePv reactiveLimitEntry=969) and OLF pins it MIN_Q at
    // q=-3.231 against a -0.774 limit, while the device read q~0 and never pinned. Its THIRD slack check
    // then measured 0.013280180 where OLF measures 0.006846832 -- straddling the 1 MW gate -- so the
    // device took a distribution OLF does not and reported 1.328 MW more distributed active power: the
    // worst such difference on the whole busbar scope.
    size_t qIdx = (reactiveLimitReactivePowerRow != nullptr && reactiveLimitReactivePowerRow[i] >= 0 && reactiveLimitReactivePowerRow[i] != vrow)
            ? (size_t) s * n + reactiveLimitReactivePowerRow[i] : idx;
    size_t scenarioEntry = (size_t) s * nReactiveLimitEntries + i;
    size_t li = perScenarioReactiveLimits ? scenarioEntry : (size_t) i;                 // curve limits are per (scenario, entry)
    bool slopeRow = rowSlope[vrow] != 0;                    // SLOPE-SVC bus: its PV form is row mode 2
    double dv2 = svcLim ? x[idx] * x[idx] - reactiveLimitBaseVSquared[i] : 0.0;   // see the remote branch: NaN base v² otherwise
    double liMin = reactiveLimitMinQ[li] + (svcLim ? svcBMin * dv2 : 0.0);
    double liMax = reactiveLimitMaxQ[li] + (svcLim ? svcBMax * dv2 : 0.0);
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
    // is not its either: the pilot's row follows its MEMBERS — updatePilotTargetVEquationBatch drops the V
    // identity when the last member pins and gives it back when one unpins, base-lost groups
    // included. Blocking both directions used to leave network.xiidm/LKPH 0.32 kV off because
    // nothing restored the row then; now something does, in the same pass OLF does it, and this
    // entry's own voltage test (a pin type guessed for an empty range) fired a pass late on LNER.
    // ...but ONLY such a bus: reactiveLimitFlags bit 2 (ReactiveLimitsPacker) marks an entry that is NOT one of its voltage
    // control's controller elements, the set OLF iterates (LfNetwork.getControllerElements(GENERATOR)). A real
    // generator bus whose capability curve collapses to [0, 0] at its current P is a controller OLF still checks:
    // network.xiidm/FBQV, FBQF_0 pinned MIN_Q at 0 with V 1.009793 under its 1.010221 target - OLF switched it
    // back to PV at the first reactive-limit pass, the device never tested it, and CQEM_0 then pinned a pass late
    // (the one contingency of 24,376 whose outer-loop sequence differed from the 1-thread CPU's).
    if (liMin >= liMax && (reactiveLimitFlags == nullptr || (reactiveLimitFlags[i] & GpuReactiveLimitsEntryFlag::NOT_CONTROLLER_ELEMENT) != 0)) {
        putReactiveLimitsDecision(passSwitched[scenarioEntry], qBuf[qIdx] + reactiveLimitLoadQ[li], liMin, liMax, x[idx], x[idx],
                 reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]));
        return;
    }
    // A re-merged LOCAL control whose own V row carries a live coupler's ZERO_V keeps its identity (mode 6, on its
    // own V column) and its pinned balance (mode 4) on the RELOCATED row — the parent coupler's DUMMY_Q (tasks #70,
    // VC_REMERGE_RELOC_VROW). The voltage is still read at the bus's own column (idx).
    size_t pinIdx = idx;
    if (localOv && reactiveLimitReactivePowerRow != nullptr && reactiveLimitReactivePowerRow[i] >= 0 && reactiveLimitReactivePowerRow[i] != vrow) {
        size_t r2 = (size_t) s * n + reactiveLimitReactivePowerRow[i];
        pinIdx = crossIdentityPinRow(mode, r2, idx, r2, idx);
    }
    const bool crossRow = pinIdx != idx;
    if ((mode[pinIdx] == GpuRowMode::FIXED_VARIABLE || mode[pinIdx] == GpuRowMode::BUS_TARGET_V_WITH_SLOPE || mode[pinIdx] == GpuRowMode::CROSS_FIXED_VARIABLE) && limitType[scenarioEntry] == GpuQLimitType::NONE) {  // currently PV — check the reactive limits
        double q = qBuf[qIdx] + reactiveLimitLoadQ[li];
        if (q < liMin - tol) {
            savedTarget[scenarioEntry] = target[pinIdx];                // remember the row target for the switch-back
            savedMode[scenarioEntry] = static_cast<char>(mode[pinIdx]);   // and the PV form (0, 2 for a slope SVC, 6 cross)
            mode[pinIdx] = localOv ? GpuRowMode::BUS_TARGET_Q : GpuRowMode::EQUATION;             // a re-merged member: own Q through its ALTERNATE dest
            target[pinIdx] = liMin - reactiveLimitLoadQ[li];
            limitType[scenarioEntry] = GpuQLimitType::MIN_Q;
            switchCount[scenarioEntry]++;
            switched[s] = 1;
            passSwitched[scenarioEntry] = 1;
        } else if (q > liMax + tol) {
            savedTarget[scenarioEntry] = target[pinIdx];
            savedMode[scenarioEntry] = static_cast<char>(mode[pinIdx]);
            mode[pinIdx] = localOv ? GpuRowMode::BUS_TARGET_Q : GpuRowMode::EQUATION;             // a re-merged member: own Q through its ALTERNATE dest
            target[pinIdx] = liMax - reactiveLimitLoadQ[li];
            limitType[scenarioEntry] = GpuQLimitType::MAX_Q;
            switchCount[scenarioEntry]++;
            switched[s] = 1;
            passSwitched[scenarioEntry] = 1;
        }
    } else if (limitType[scenarioEntry] != GpuQLimitType::NONE) {  // pinned at a limit — check switchback
        double v = x[idx];
        double targetV = localOv ? reactiveLimitLocalTargetV[(size_t) s * nReactiveLimitEntries + i]
                  : reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]);
        // OLF never back-switches a slope-SVC bus: checkPqBus runs with
        // canSwitchToPv = !bus.hasGeneratorsWithSlope() ("we don't support switching PQ to PV
        // for bus with one controller with slope") — once pinned, a slope row stays PQ.
        bool maxPin = isMaxLimit(limitType[scenarioEntry]);
        bool minPin = isMinLimit(limitType[scenarioEntry]);
        // the cap bounds the SWITCH, not the limit refresh below (see the remote branch)
        bool wantsBack = !slopeRow
                && ((maxPin && v > targetV)                               // maxQ-pinned but v above target
                 || (minPin && v < targetV));                             // minQ-pinned but v below target
        bool back = wantsBack && switchCount[scenarioEntry] < maxSwitch;
        if (back) {
            mode[pinIdx] = fixedVariableMode(crossRow);               // PQ → PV: resume voltage control (cross identity stays cross)
            target[pinIdx] = targetV;
            limitType[scenarioEntry] = GpuQLimitType::NONE;
            switched[s] = 1;
            passSwitched[scenarioEntry] = 2;
        } else if (!wantsBack && (perScenarioReactiveLimits || svcLim) && (limitType[scenarioEntry] == GpuQLimitType::MIN_Q || limitType[scenarioEntry] == GpuQLimitType::MAX_Q)) {  // a realistic-V freeze keeps its target Q
            // Q-limit UPDATE while pinned (OLF checkPqBusWithQLimitType): a CURVE limit that moved
            // (DS moved the gen's P) or an SVC limit that moved (B*v^2 at the live V) re-freezes
            // the pinned Q at the NEW limit when it drifted by more than the mismatch tol.
            // Unreachable with static limits — gated on the channels.
            double lim = limitType[scenarioEntry] == GpuQLimitType::MIN_Q ? liMin : liMax;
            if (fabs(lim - (target[pinIdx] + reactiveLimitLoadQ[li])) > tol) {
                target[pinIdx] = lim - reactiveLimitLoadQ[li];
                switched[s] = 1;
                passSwitched[scenarioEntry] = 3;
            }
        }
    }
    putReactiveLimitsDecision(passSwitched[scenarioEntry], qBuf[qIdx] + reactiveLimitLoadQ[li], liMin, liMax, x[idx], x[idx],
             reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]));
}

// voltageRemoteControlRobustMode, second half: "Set V to a safe 1 pu for the next computation".
// ReactiveLimitsOuterLoop does this to every REMOTE voltage controller it switched PV->PQ in this
// check whose OWN voltage is outside the realistic band — whether it switched on a reactive limit
// (the first robust block) or on the realistic-V criterion itself. It is a WRITE to the state
// vector the check kernel reads, so it gets its own pass: one thread per (entry, scenario), reading
// the same pre-pass x, which makes the result independent of the order the entries happen to run in.
// x is the next inner solve's warm start, so this is OLF's controllerBus.setV(1) verbatim.
__global__ void reactiveLimitsRobustVoltageResetBatch(int nReactiveLimitEntries, int nScenarios, int n, const int* reactiveLimitVoltageRow, const char* reactiveLimitVoltageControlRole,
                                    const char* outerLoopCheckActive, const char* passSwitched, double* x,
                                    double vLoRobust, double vHiRobust) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nReactiveLimitEntries * nScenarios) return;
    int i = tid % nReactiveLimitEntries;
    int s = tid / nReactiveLimitEntries;
    if (!outerLoopCheckActive[s]) return;
    // robust mode is remote-controller only, in OLF's sense: not a class-2 pilot and not a class-3
    // merged member whose own control is local (both are exempt from the gate that sets 3/4 at all)
    if (reactiveLimitVoltageControlRole == nullptr || reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::REMOTE) return;
    if (passSwitched[(size_t) s * nReactiveLimitEntries + i] != 1) return;     // switched PV -> PQ in THIS check
    size_t idx = (size_t) s * n + reactiveLimitVoltageRow[i];
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
__global__ void reactiveLimitsForcedPvBatch(int nReactiveLimitEntries, int nScenarios, int n,
                                     const int* reactiveLimitVoltageRow, const char* reactiveLimitVoltageControlRole, const char* reactiveLimitIslanded,
                                     const double* nomV, const int* pRow, const double* staticP, const int* idRank,
                                     char* limitType, char* switchCount, char* passSwitched, const char* savedMode,
                                     const double* savedTarget, int* mode, double* target,
                                     const char* outerLoopCheckActive, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!outerLoopCheckActive[s]) return;
    int remaining = 0;
    int cand = 0;
    int best = -1;
    double bestV = -1.0;
    double bestP = -1e300;
    int bestId = 0x7fffffff;
    for (int i = 0; i < nReactiveLimitEntries; i++) {
        size_t scenarioEntry = (size_t) s * nReactiveLimitEntries + i;
        if (reactiveLimitIslanded != nullptr && reactiveLimitIslanded[scenarioEntry]) continue;   // not a controller in this component
        bool remote = reactiveLimitVoltageControlRole != nullptr && reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::LOCAL;
        int m = mode[(size_t) s * n + reactiveLimitVoltageRow[i]];
        if (passSwitched[scenarioEntry] == 1) {
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
        } else if (limitType[scenarioEntry] == GpuQLimitType::NONE && (reactiveLimitVoltageControlRole != nullptr && reactiveLimitVoltageControlRole[i] == GpuVoltageControlRole::PILOT_SELF ? m == GpuRowMode::FIXED_VARIABLE
                                          : remote ? m == GpuRowMode::EQUATION : (m == GpuRowMode::FIXED_VARIABLE || m == GpuRowMode::BUS_TARGET_V_WITH_SLOPE))) {
            remaining++;                                     // still PV after this pass
        }
    }
    if (cand == 0 || remaining > 0 || best < 0) return;
    size_t scenarioEntry = (size_t) s * nReactiveLimitEntries + best;
    size_t idx = (size_t) s * n + reactiveLimitVoltageRow[best];
    mode[idx] = (reactiveLimitVoltageControlRole != nullptr && reactiveLimitVoltageControlRole[best] == GpuVoltageControlRole::PILOT_SELF) ? 0
              : (reactiveLimitVoltageControlRole != nullptr && reactiveLimitVoltageControlRole[best] != GpuVoltageControlRole::LOCAL) ? 1 : savedMode[scenarioEntry];
    target[idx] = savedTarget[scenarioEntry];
    limitType[scenarioEntry] = GpuQLimitType::NONE;
    switchCount[scenarioEntry]--;
    passSwitched[scenarioEntry] = 0;
    char any = 0;
    for (int i = 0; i < nReactiveLimitEntries && !any; i++) {
        any = passSwitched[(size_t) s * nReactiveLimitEntries + i] != 0;
    }
    switched[s] = any;
}

// Warm-continuation reconcile: a SEEDED pinned controller (limitType != 0) must also carry its
// consequences — the BUS_V row flips to the reactive equation (mode 1) with the pinned Q target —
// else the solve holds the PV setpoint while the RL bookkeeping thinks the bus is at a limit.
// A PINNED pilot that is one of its own controllers (reactiveLimitVoltageControlRole 2) while a member of the same group
// (same compare row = the pilot's V) still controls: OLF then keeps the pilot's V identity AND
// activates the pilot's own Q balance, hosting it on the reference member's row - a relocation this
// group has no table for on the device. Flag the scenario; the host rejects it to the CPU replay.
// The other mixed state (members pinned, pilot controlling) is an ordinary layout: pilot identity,
// members at their own Q. One thread per (scenario, class-2 entry).
__global__ void pilotSelfMixedPvPqGate(int nReactiveLimitEntries, int nScenarios, const char* reactiveLimitVoltageControlRole, const int* reactiveLimitComparedVoltageRow,
                                     const char* limitType, const char* reactiveLimitIslanded, char* bad,
                                     const int* pilotSelfReactiveLimitEntry, int nReactiveRelocationGroups,
                                int nDistrQMembers, const int* distrQMemberReactiveLimitEntry, const char* distrQSubgroup) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nReactiveLimitEntries * nScenarios) return;
    int i = t % nReactiveLimitEntries;
    int s = t / nReactiveLimitEntries;
    if (reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::PILOT_SELF || limitType[(size_t) s * nReactiveLimitEntries + i] == GpuQLimitType::NONE) return;  // only a PINNED pilot
    // ...that is still ON this component: an islanded pilot keeps its base limit type but has no pin and no
    // balance (OLF removed the bus), and an islanded member is not "still controlling" either.
    if (reactiveLimitIslanded != nullptr && reactiveLimitIslanded[(size_t) s * nReactiveLimitEntries + i]) return;
    if (isEnrolledPilot(pilotSelfReactiveLimitEntry, nReactiveRelocationGroups, i)) return;            // enrolled (rotation): the carrier hosts its balance
    for (int j = 0; j < nReactiveLimitEntries; j++) {                          // ... with a member still controlling
        if (j != i && reactiveLimitComparedVoltageRow[j] == reactiveLimitComparedVoltageRow[i] && reactiveLimitVoltageControlRole[j] != GpuVoltageControlRole::LOCAL
                && limitType[(size_t) s * nReactiveLimitEntries + j] == GpuQLimitType::NONE
                && !(reactiveLimitIslanded != nullptr && reactiveLimitIslanded[(size_t) s * nReactiveLimitEntries + j])) {
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
// target, and limitType 0. The host orchestrator got this for free by re-seeding reactiveLimitsSeedState on its own
// re-solve — a RESUME skips the RL seeding, so without this channel the un-pin never reached the device.
__global__ void reactiveLimitsPqPvSwitchBatch(int nPairs, int nS, int n, int nReactiveLimitEntries, const int* pairs, const double* newTarget,
                             const int* reactiveLimitVoltageRow, char* limitType, const char* savedMode, int* mode, double* target) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nPairs) return;
    int s = pairs[2 * k];
    int i = pairs[2 * k + 1];
    if (s < 0 || s >= nS || i < 0 || i >= nReactiveLimitEntries) return;
    size_t scenarioEntry = (size_t) s * nReactiveLimitEntries + i;
    if (limitType[scenarioEntry] == GpuQLimitType::NONE) return;  // not pinned: nothing to give back
    // A member FROZEN by the generator voltage control manager is not a reactive-limit pin: giving it
    // back would restore a V identity on a row that holds the group's DISTR_Q share, and would undo a
    // decision the transformer phase owns and will undo itself at COMPLETE.
    if (limitType[scenarioEntry] == GpuQLimitType::FROZEN) return;
    size_t idx = (size_t) s * n + reactiveLimitVoltageRow[i];
    char sm = savedMode[scenarioEntry];
    mode[idx] = (sm == GpuRowMode::BUS_TARGET_V_WITH_SLOPE) ? GpuRowMode::BUS_TARGET_V_WITH_SLOPE : GpuRowMode::FIXED_VARIABLE;                          // the PV form saved at pin time (2 = slope SVC)
    target[idx] = newTarget[k];
    limitType[scenarioEntry] = GpuQLimitType::NONE;
}

// One check of ReactiveLimitsOuterLoop on every live scenario (GpuOuterLoopKind::REACTIVE_LIMITS): the curve limits at the current
// targets, the PV->PQ pins and releases, robust mode's realistic-voltage pins and the forced-PV election.
void reactiveLimitsCheck(BatchSolverContext* ctx, DriverPass& drv, int loopKind) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& active = drv.active;
    std::vector<char>& live = drv.live;
    BatchResult& res = drv.res;
    const bool& usePerScenarioReactiveLimits = drv.usePerScenarioReactiveLimits;
    const double& REACTIVE_LIMIT_TOL = drv.reactiveLimitTol;
    const int& MAX_PQ_PV_SWITCH = drv.maxPqPvSwitch;
    const int& reactiveLimitsDecisionLogScenario = drv.reactiveLimitsDecisionLogScenario;
    const int& skipReactiveLimitsPhase = drv.skipReactiveLimitsPhase;
    const bool& hasPerScenarioSvcSusceptanceLimits = drv.hasPerScenarioSvcSusceptanceLimits;
    const int& nReactiveLimitTargetVOverrides = drv.nReactiveLimitTargetVOverrides;
    const int& nReactiveLimitDeltaOverrides = drv.nReactiveLimitDeltaOverrides;
    const int& nReactiveCapabilityCurveOverrides = drv.nReactiveCapabilityCurveOverrides;
    DeviceBuf& dReactiveLimitTargetVOverrideScenarioEntry = drv.dReactiveLimitTargetVOverrideScenarioEntry;
    DeviceBuf& dReactiveLimitTargetVOverrideValue = drv.dReactiveLimitTargetVOverrideValue;
    DeviceBuf& dReactiveLimitDeltaOverrideScenarioEntry = drv.dReactiveLimitDeltaOverrideScenarioEntry;
    DeviceBuf& dReactiveLimitDeltaOverrideValues = drv.dReactiveLimitDeltaOverrideValues;
    DeviceBuf& dReactiveLimitDeltaOverrideStart = drv.dReactiveLimitDeltaOverrideStart;
    DeviceBuf& dReactiveLimitDeltaOverrideList = drv.dReactiveLimitDeltaOverrideList;
    DeviceBuf& dReactiveCapabilityCurveOverrideStart = drv.dReactiveCapabilityCurveOverrideStart;
    DeviceBuf& dReactiveCapabilityCurveOverrideCurve = drv.dReactiveCapabilityCurveOverrideCurve;
    DeviceBuf& dReactiveCapabilityCurveOverrideStepScale = drv.dReactiveCapabilityCurveOverrideStepScale;
    if (loopKind == GpuOuterLoopKind::REACTIVE_LIMITS && ctx->reactiveLimitsOuterLoop && ctx->nReactiveLimitEntries > 0 && !skipReactiveLimitsPhase) {                  // skipReactiveLimitsPhase: a monitoring interleave solve — the
        if (usePerScenarioReactiveLimits) {                            // curve limits at the CURRENT (post-DS) targets,
            reactiveCapabilityCurveLimitsBatch<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS, ctx->nReactiveCapabilityCurves, n, ctx->reactiveCapabilityCurveExtrapolate,
                    ctx->dReactiveCapabilityCurveIndices.as<int>(), ctx->dReactiveCapabilityCurveParameters.as<double>(), ctx->dReactiveCapabilityCurvePoints.as<double>(),
                    ctx->dReactiveLimitEntryCurveStart.as<int>(), ctx->dReactiveLimitEntryCurves.as<int>(), curveHvdcAcEmulationView(ctx),
                    ctx->dReactiveLimitMinQ.as<double>(), ctx->dReactiveLimitMaxQ.as<double>(), ctx->dReactiveLimitLoadQ.as<double>(),
                    ctx->dTarget.as<double>(),
                    nReactiveLimitDeltaOverrides, nReactiveLimitDeltaOverrides > 0 ? dReactiveLimitDeltaOverrideScenarioEntry.as<int>() : nullptr,
                    nReactiveLimitDeltaOverrides > 0 ? dReactiveLimitDeltaOverrideValues.as<double>() : nullptr,
                    nReactiveLimitDeltaOverrides > 0 ? dReactiveLimitDeltaOverrideStart.as<int>() : nullptr, nReactiveLimitDeltaOverrides > 0 ? dReactiveLimitDeltaOverrideList.as<int>() : nullptr,
                    nReactiveCapabilityCurveOverrides > 0 ? dReactiveCapabilityCurveOverrideStart.as<int>() : nullptr,
                    nReactiveCapabilityCurveOverrides > 0 ? dReactiveCapabilityCurveOverrideCurve.as<int>() : nullptr,
                    nReactiveCapabilityCurveOverrides > 0 ? dReactiveCapabilityCurveOverrideStepScale.as<double>() : nullptr,
                    ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMove.as<double>() : nullptr,
                    ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMoveValid.as<char>() : nullptr, ctx->hasDistributedSlackGeneratorAnchor ? ctx->dDistributedSlackGeneratorAnchor.as<double>() : nullptr, ctx->hasDistributedSlackGeneratorInitialMove ? ctx->dDistributedSlackGeneratorInitialMove.as<double>() : nullptr, ctx->nDistributedSlackGenerators,
                    ctx->nDistributedSlackBuses > 0 ? ctx->dDistributedSlackBusPhiRow.as<int>() : nullptr,
                    ctx->nDistributedSlackBusRowOverrides > 0 ? ctx->dDistributedSlackBusRowOverride.as<int>() : nullptr, ctx->nDistributedSlackBuses,
                    ctx->nDistributedSlackGenerators > 0 ? ctx->distributedSlackGeneratorBus.as<int>() : nullptr,
                    ctx->dReactiveLimitMinQPerScenario.as<double>(), ctx->dReactiveLimitMaxQPerScenario.as<double>(),
                    ctx->dReactiveLimitLoadQPerScenario.as<double>());
            launchForcedTargetQLimits(ctx, nS, n);
        }
        CUDA_CHECK(cudaMemsetAsync(ctx->dReactiveLimitCheckDecision.p, 0, static_cast<size_t>(ctx->nReactiveLimitEntries) * nS, ctx->stream));
        reactiveLimitsBatch<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(
                ctx->nSecondaryVoltageControlControllerBuses > 0 ? ctx->dSecondaryVoltageControlReactiveLimitTargetV.as<double>() : nullptr,
                ctx->nReactiveLimitEntries, nS, n, REACTIVE_LIMIT_TOL, MAX_PQ_PV_SWITCH,
                ctx->dReactiveLimitVoltageRow.as<int>(),
                ctx->dReactiveLimitReactivePowerRow.p ? ctx->dReactiveLimitReactivePowerRow.as<int>() : nullptr,
                usePerScenarioReactiveLimits ? ctx->dReactiveLimitMinQPerScenario.as<double>() : ctx->dReactiveLimitMinQ.as<double>(),
                usePerScenarioReactiveLimits ? ctx->dReactiveLimitMaxQPerScenario.as<double>() : ctx->dReactiveLimitMaxQ.as<double>(),
                usePerScenarioReactiveLimits ? ctx->dReactiveLimitLoadQPerScenario.as<double>() : ctx->dReactiveLimitLoadQ.as<double>(),
                ctx->dReactiveLimitTargetV.as<double>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(), ctx->dOuterLoopCheckActive.as<char>(),
                ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                ctx->dReactiveLimitQLimitType.as<char>(), ctx->dReactiveLimitPvPqSwitchCount.as<char>(), ctx->dOuterLoopUnstable.as<char>(),
                ctx->dReactiveLimitCheckDecision.as<char>(), ctx->dReactiveLimitSavedRowMode.as<char>(),
                nReactiveLimitTargetVOverrides, nReactiveLimitTargetVOverrides > 0 ? dReactiveLimitTargetVOverrideScenarioEntry.as<int>() : nullptr,
                nReactiveLimitTargetVOverrides > 0 ? dReactiveLimitTargetVOverrideValue.as<double>() : nullptr,
                ctx->dRowSlope.as<double>(), ctx->dReactiveLimitSavedTarget.as<double>(), usePerScenarioReactiveLimits ? 1 : 0,
                ctx->nReactiveLimitRemoteEntries > 0 ? ctx->dReactiveLimitVoltageControlRole.as<char>() : nullptr,
                ctx->nReactiveLimitRemoteEntries > 0 ? ctx->dReactiveLimitComparedVoltageRow.as<int>() : nullptr,
                ctx->dReactiveLimitSvcMinB.as<double>(), ctx->dReactiveLimitSvcMaxB.as<double>(), ctx->dReactiveLimitBaseVSquared.as<double>(),
                hasPerScenarioSvcSusceptanceLimits ? ctx->dReactiveLimitSvcMinBPerScenario.as<double>() : nullptr, hasPerScenarioSvcSusceptanceLimits ? ctx->dReactiveLimitSvcMaxBPerScenario.as<double>() : nullptr,
                ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr,
                ctx->reactiveLimitsRobustMode ? ctx->dReactiveLimitInitialTargetQ.as<double>() : nullptr,
                ctx->reactiveLimitsRobustMinRealisticVoltage, ctx->reactiveLimitsRobustMaxRealisticVoltage, ctx->reactiveLimitsRobustMode,
                reactiveLimitsDecisionLogScenario, ctx->dReactiveLimitsDecisionLog.p ? ctx->dReactiveLimitsDecisionLog.as<double>() : nullptr,
                ctx->dReactiveLimitsDecisionLogCount.p ? ctx->dReactiveLimitsDecisionLogCount.as<int>() : nullptr, outer,
                pilotSelfReactiveLimitEntryOrNull(ctx), ctx->structure->nReactiveRelocationGroups,
                ctx->nReactiveLimitLocalOverrides > 0 ? ctx->dReactiveLimitLocalOverride.as<char>() : nullptr,
                ctx->nReactiveLimitLocalOverrides > 0 ? ctx->dReactiveLimitLocalTargetV.as<double>() : nullptr,
                ctx->nReactiveLimitPqPvSwitchForbidden > 0 ? ctx->dReactiveLimitPqPvSwitchForbidden.as<char>() : nullptr, ctx->nReactiveLimitPqPvSwitchForbidden > 0 ? ctx->dReactiveLimitPqPvSwitchRejected.as<char>() : nullptr,
                ctx->dReactiveLimitFlags.p ? ctx->dReactiveLimitFlags.as<char>() : nullptr,
                ctx->nReactiveLimitComparedVoltageRowOverrides > 0 ? ctx->dReactiveLimitComparedVoltageRowOverride.as<int>() : nullptr);
        if (ctx->nReactiveLimitPilotSelfEntries > 0) {
            // a pilot that is its own controller pinned while a member of its group is not (or
            // the reverse): the device has no row for OLF's layout of that state - CPU replay
            if (!ctx->dPilotSelfMixedPvPqRejected.p) ctx->dPilotSelfMixedPvPqRejected.alloc(nS * sizeof(char));
            CUDA_CHECK(cudaMemsetAsync(ctx->dPilotSelfMixedPvPqRejected.p, 0, nS * sizeof(char), ctx->stream));
            pilotSelfMixedPvPqGate<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS,
                    ctx->dReactiveLimitVoltageControlRole.as<char>(), ctx->dReactiveLimitComparedVoltageRow.as<int>(), ctx->dReactiveLimitQLimitType.as<char>(),
                    ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr,
                    ctx->dPilotSelfMixedPvPqRejected.as<char>(),
                    pilotSelfReactiveLimitEntryOrNull(ctx), ctx->structure->nReactiveRelocationGroups,
                ctx->nDistrQMembers, ctx->dDistrQMemberReactiveLimitEntry.p != nullptr ? ctx->dDistrQMemberReactiveLimitEntry.as<int>() : nullptr, ctx->hasDistrQSubgroups ? ctx->dDistrQSubgroup.as<char>() : nullptr);
            std::vector<char> mixed(nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(mixed.data(), ctx->dPilotSelfMixedPvPqRejected.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (mixed[s] && live[s]) {
                    traceReject(s, "pilot-self-mixed-pin", outer, 0);
                    res.status[s] = GpuAcSolverStatus::REJECTED;
                    active[s] = 0;
                    live[s] = 0;
                }
            }
        }
        if (ctx->nReactiveLimitPqPvSwitchForbidden > 0) {
            std::vector<char> ub(nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(ub.data(), ctx->dReactiveLimitPqPvSwitchRejected.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (ub[s] && live[s]) {
                    traceReject(s, "unpin-unrepresentable", outer, 0);
                    res.status[s] = GpuAcSolverStatus::REJECTED;
                    active[s] = 0;
                    live[s] = 0;
                }
            }
        }
        if (ctx->reactiveLimitsRobustMode && ctx->nReactiveLimitRemoteEntries > 0) {       // "set V to a safe 1 pu", as its own pass
            reactiveLimitsRobustVoltageResetBatch<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(
                    ctx->nReactiveLimitEntries, nS, n, ctx->dReactiveLimitVoltageRow.as<int>(), ctx->dReactiveLimitVoltageControlRole.as<char>(),
                    ctx->dOuterLoopCheckActive.as<char>(), ctx->dReactiveLimitCheckDecision.as<char>(), ctx->dX.as<double>(),
                    ctx->reactiveLimitsRobustMinRealisticVoltage, ctx->reactiveLimitsRobustMaxRealisticVoltage);
        }
        if (ctx->hasReactiveLimitStrongestPvKeys && ctx->allPvPinnable) {  // the rule can only fire when every PV is pinnable
            reactiveLimitsForcedPvBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS, n,
                    ctx->dReactiveLimitVoltageRow.as<int>(),
                    ctx->nReactiveLimitRemoteEntries > 0 ? ctx->dReactiveLimitVoltageControlRole.as<char>() : nullptr,
                    ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr,
                    ctx->dReactiveLimitNominalV.as<double>(), ctx->dReactiveLimitActivePowerRow.as<int>(),
                    ctx->dReactiveLimitTargetP.as<double>(), ctx->dReactiveLimitIdRank.as<int>(),
                    ctx->dReactiveLimitQLimitType.as<char>(), ctx->dReactiveLimitPvPqSwitchCount.as<char>(), ctx->dReactiveLimitCheckDecision.as<char>(),
                    ctx->dReactiveLimitSavedRowMode.as<char>(), ctx->dReactiveLimitSavedTarget.as<double>(),
                    ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->dOuterLoopUnstable.as<char>());
        }
        // The pilot's V-identity flip/restore runs BEFORE the DISTR_Q targets: the carrier hand-off picks the
        // carrier with chooseCarrierRow, which reads the pilot's own row (candidate 1 of a pilot-self group).
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
                    ctx->nPilotFlip, nS, n, ctx->nReactiveLimitEntries,
                    ctx->dPfSlot.as<int>(), ctx->dPfRow.as<int>(), ctx->dPfQTarget.as<double>(),
                    ctx->dPfOff.as<int>(), ctx->dPfCount.as<int>(), ctx->dPilotFlipSurvivorReactiveLimitEntries.as<int>(),
                    ctx->dReactiveLimitQLimitType.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                    ctx->dPfFlipped.as<char>(), ctx->dPfSavedVT.as<double>(),
                    ctx->dOuterLoopCheckActive.as<char>(), ctx->dOuterLoopUnstable.as<char>());
        }
        if (ctx->hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0) {    // all-pinned corner: pilot vEq flip/restore
            launchPilotTargetVEquationUpdate(ctx, drv);
        }
        if (ctx->nReactiveLimitRemoteEntries > 0 && ctx->nDistrQMembers > 0) {
            // keep the survivors' DISTR_Q row targets in step with the pin states (plan-B3)
            launchDistrQTargetsInner(ctx, drv);
        }
    }
}

}  // namespace olfgpu
