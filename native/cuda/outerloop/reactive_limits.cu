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
                                    const char* reactiveLimitFlags,
                                    const int* reactiveLimitComparedVoltageRowOverride,
                                    const char* reactiveLimitHoldsReMergedVoltageTarget, const int* reactiveLimitReMergedVoltageRow,
                                    int* remainingPvBusCount, int* switchedRow) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nReactiveLimitEntries * nScenarios) return;
    int i = tid % nReactiveLimitEntries;
    int s = tid / nReactiveLimitEntries;
    if (!outerLoopCheckActive[s]) return;
    // A controller the contingency put in ANOTHER synchronous component is not a controller at all:
    // OLF disables that bus and its outer loops never see it. Its rows here are FROZEN at a flat
    // 1<0, so the q read out of qBuf is a placeholder, not a measurement — and acting on it switches PV -> PQ
    // a bus that OLF leaves alone, which perturbs the MAIN component through any remote-control
    // or DISTR_Q group the controller belongs to. Measured on LINE-3320-350: the device switched
    // VL-6432_0, an ISLANDED bus, to PQ, giving it 20 controllers switched to PQ against OLF's 19 and one extra
    // LOW_VOLTAGE violation at VL-1624.
    if (reactiveLimitIslanded != nullptr && reactiveLimitIslanded[(size_t) s * nReactiveLimitEntries + i]) return;
    // A controller FROZEN by the generator voltage control manager has its voltage control OFF, so it
    // is not one of this loop's controllers at all — OLF iterates buses with voltage control and never
    // reaches it. Leave the type (and the row) to the transformer phase that set it.
    if (limitType[(size_t) s * nReactiveLimitEntries + i] == GpuQLimitType::FROZEN) return;
    // SVC reactive limits are VOLTAGE-DEPENDENT: LfStaticVarCompensatorImpl evaluates
    // Bmin/Bmax*(v*nomV)^2 at the LIVE bus V on every CPU check (and a bus switched to PQ re-freezes at
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
    // OLF's remainingPvBusCount (ReactiveLimitsOuterLoop.checkControllerBus): a bus this check finds PV and leaves PV,
    // counted where the check decides it. A bus that switches back to PV in this check is not one, as in OLF, which
    // counts before it switches anything. Nor is a pilot packed without being one of its control's controller elements
    // (GpuReactiveLimitsEntryFlag::NOT_CONTROLLER_ELEMENT): OLF's loop never sees it.
    auto countRemainingPvBus = [&]() {
        if (remainingPvBusCount != nullptr && (reactiveLimitFlags == nullptr || (reactiveLimitFlags[i] & GpuReactiveLimitsEntryFlag::NOT_CONTROLLER_ELEMENT) == 0)) {
            atomicAdd(&remainingPvBusCount[s], 1);
        }
    };
    // The row a PV -> PQ switch of this check writes (-1: none), whose mode and target reactiveLimitsForcedPvBatch gives
    // back to the bus it keeps PV. It is not always the bus's own V row: a relocated controller's switch lands on its
    // coupler's DUMMY_Q row, while its own V row carries the coupler's voltage equation.
    auto recordSwitchedRow = [&](size_t scenarioEntry, size_t rowIndex, bool written) {
        if (switchedRow != nullptr) {
            switchedRow[scenarioEntry] = written ? static_cast<int>(rowIndex - (size_t) s * n) : -1;
        }
    };
    // A RE-MERGED voltage control (GpuBatchQualification.reMergedControls): this scenario the bus is a plain PV
    // bus, so its entry — packed as a remote member of the base group — takes the LOCAL path below, at its
    // own voltage target. OLF checks it as the PV bus it now is (BUS-6113_BBS_1: VL-6113_1 switched to PQ at RL#1).
    bool localOv = reactiveLimitLocalOverride != nullptr && reactiveLimitLocalOverride[(size_t) s * nReactiveLimitEntries + i];
    if (reactiveLimitVoltageControlRole != nullptr && reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::LOCAL && !localOv) {
        // REMOTE group member (plan-B3). PV = its DISTR_Q row (mode 1, limitType 0);
        // switched to PQ = mode 4 ("row hosts its own Q balance"). BOTH switch conditions test the
        // PILOT's voltage (reactiveLimitComparedVoltageRow) against the GROUP target — B0-verified CPU semantics.
        // The FREE member switches to PQ the same way (plan-B4): the fill kernels re-pick the pilot-Q
        // carrier by mode-scan over the relocation candidates, and updateDistrQTargetsBatch
        // hands the pilot's Q target to the new carrier's row. When the LAST live member switches to PQ,
        // updatePilotTargetVEquationBatch flips the pilot's V identity to a power row (control lost) and
        // restores it on the first switch back to PV — the no-enabled-controller corner runs on-device too.
        size_t ridx = (size_t) s * n + reactiveLimitVoltageRow[i];            // the bus's own V variable: every VOLTAGE read
        // ...and the row its reactive balance and its frozen target Q live on. The coupler-tree relocation moves it to the parent
        // coupler's DUMMY_Q row — but PER SCENARIO: a contingency that OPENS that coupler sends the child's
        // equations back to its own rows and the host fixes the dummy row with a DUMMY_TARGET identity, which is
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
        // poisoned every entry's limits there and nothing ever switched to PQ at base.
        double rdv2 = svcLim ? x[ridx] * x[ridx] - reactiveLimitBaseVSquared[i] : 0.0;
        double remoteMinQ = reactiveLimitMinQ[remoteLimitIndex] + (svcLim ? svcBMin * rdv2 : 0.0);
        double remoteMaxQ = reactiveLimitMaxQ[remoteLimitIndex] + (svcLim ? svcBMax * rdv2 : 0.0);
        // reactiveLimitVoltageControlRole 2 = a PILOT that is one of its own controllers: PV, its row is the group's
        // voltage IDENTITY (mode 0), not a DISTR_Q share; switched to PQ, its own Q balance (mode 4, the
        // alternate dest). Everything else (q via the remote q pack, the pilot's V as compare row)
        // is a remote member's. As a LOCAL entry its PV -> PQ switch landed on mode 1, an EMPTY row.
        int pvForm = reactiveLimitVoltageControlRole[i] == GpuVoltageControlRole::PILOT_SELF ? GpuRowMode::FIXED_VARIABLE : GpuRowMode::EQUATION;
        bool enrolledPilot = reactiveLimitVoltageControlRole[i] == GpuVoltageControlRole::PILOT_SELF && isEnrolledPilot(pilotSelfReactiveLimitEntry, nReactiveRelocationGroups, i);
        // its sharing row may hold its re-merged control's voltage target (updateReMergedVoltageTargetRowBatch): it is
        // the control's first enabled controller then, and tested like any other
        bool holdsVoltageTarget = reactiveLimitHoldsReMergedVoltageTarget != nullptr && reactiveLimitHoldsReMergedVoltageTarget[rst];
        if (limitType[rst] == GpuQLimitType::NONE) {
            if (mode[qidx] != pvForm && !(holdsVoltageTarget && holdsFixedVariable(mode[qidx]))) return;   // islanded / frozen this scenario
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
                savedMode[rst] = static_cast<char>(mode[qidx]);
                recordSwitchedRow(rst, qidx, !enrolledPilot);
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
                savedMode[rst] = static_cast<char>(mode[qidx]);
                recordSwitchedRow(rst, qidx, !enrolledPilot);
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
                savedMode[rst] = static_cast<char>(mode[qidx]);
                recordSwitchedRow(rst, qidx, !enrolledPilot);
                if (!enrolledPilot) {
                    mode[qidx] = GpuRowMode::BUS_TARGET_Q;
                    target[qidx] = reactiveLimitInitialTargetQ[i] - reactiveLimitLoadQ[remoteLimitIndex];
                }
                limitType[rst] = unrealisticLo ? GpuQLimitType::MIN_REALISTIC_V : GpuQLimitType::MAX_REALISTIC_V;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            } else {
                countRemainingPvBus();
            }
            // OLF also sets the controller's V to a safe 1 pu for the next computation, under BOTH
            // robust criteria. That WRITES the state vector every other entry is still reading here,
            // so it runs as its own pass (reactiveLimitsRobustVoltageResetBatch) once this one has finished.
        } else {                                            // switched to PQ — switch-back tests the PILOT
            // a controller re-merged onto a NEW pilot this scenario reads that pilot's voltage (reactiveLimitComparedVoltageRowOverride)
            double v = x[(size_t) s * n + (reactiveLimitComparedVoltageRowOverride != nullptr && reactiveLimitComparedVoltageRowOverride[rst] >= 0 ? reactiveLimitComparedVoltageRowOverride[rst] : reactiveLimitComparedVoltageRow[i])];
            double targetV = reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]);
            bool atMinLimit = isMinLimit(limitType[rst]);
            // OLF's checkPqBusWithQLimitType PROPOSES the switch back to PV whenever the pilot's voltage asks for
            // it; the per-bus cap (maxPqPvSwitch) then blocks the switch itself in switchPqPv. Only
            // the ELSE of that test re-freezes a moved limit — so a capped bus that still WANTS to
            // switch back to PV re-freezes on neither arm, while a capped bus that does not must still follow its
            // limit. Gating this whole block on the cap froze such a bus's target at a stale limit:
            // network.xiidm / GBMK, CQEM_0 and CQEM_5 kept -0.181175517 / -0.184950765 where OLF had
            // re-frozen at -0.181173957 / -0.184949017 (a reactive-capability curve moving with the
            // distributed slack), 2.079e-05 kV network-wide.
            bool wantsBack = (!atMinLimit && v > targetV) || (atMinLimit && v < targetV);
            bool back = wantsBack && switchCount[rst] < maxSwitch;
            if (back) {
                if (!enrolledPilot) {
                    mode[qidx] = pvForm;                     // back to the DISTR_Q share row (the V identity for a pilot)
                    target[qidx] = savedTarget[rst];
                }
                limitType[rst] = GpuQLimitType::NONE;
                switched[s] = 1;
                passSwitched[rst] = 2;
            } else if (!wantsBack && (perScenarioReactiveLimits || svcLim) && (limitType[rst] == GpuQLimitType::MIN_Q || limitType[rst] == GpuQLimitType::MAX_Q)) {  // curve/SVC limit drift while switched to PQ
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
    // ReactiveLimitsPacker: filling the own row left the check reading ZERO for every relocated member,
    // so it could never switch one to PQ). That pack is STRUCTURAL -- it keeps writing there even in a scenario
    // that fixes the dummy with a DUMMY_TARGET identity -- and qBuf is a separate accumulator, so reading it
    // there is valid whatever the row's equation MODE. Only the PV -> PQ switch's WRITE belongs on the live row, which
    // is idx. Conflating the two reproduced that base failure per scenario: measured on rte6515
    // BUS-6113_BBS_1, VL-6113_1 is re-merged LOCAL (reMergePv reactiveLimitEntry=969) and OLF switches it to PQ at MIN_Q at
    // q=-3.231 against a -0.774 limit, while the device read q~0 and never switched it. Its THIRD slack check
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
    // node is zero up to rounding, so the PV -> PQ test |q| > tol with tol = convEps (1e-12 under
    // UNIFORM_CRITERIA) fires on the LAST BITS. Measured on network.xiidm: two runs of the same
    // 2000-contingency window differed on 430 reactive-limit decisions and every one was such an
    // entry, q at +-1.2e-12 against limits of exactly 0. The switch flips the pilot's row to its own Q
    // balance for a pass, which perturbs the whole group, and once per ~2000 contingencies it
    // changes a real downstream PV -> PQ switch (8.178e-05 kV on the KUMB group).
    // Neither direction is this entry's to decide. The PV -> PQ switch is refused (above), and the switch-back
    // is not its either: the pilot's row follows its MEMBERS — updatePilotTargetVEquationBatch drops the V
    // identity when the last member switches to PQ and gives it back when one switches back to PV, base-lost groups
    // included. Blocking both directions used to leave network.xiidm/LKPH 0.32 kV off because
    // nothing restored the row then; now something does, in the same pass OLF does it, and this
    // entry's own voltage test (a limit type guessed for an empty range) fired a pass late on LNER.
    // ...but ONLY such a bus: reactiveLimitFlags bit 2 (ReactiveLimitsPacker) marks an entry that is NOT one of its voltage
    // control's controller elements, the set OLF iterates (LfNetwork.getControllerElements(GENERATOR)). A real
    // generator bus whose capability curve collapses to [0, 0] at its current P is a controller OLF still checks:
    // network.xiidm/FBQV, FBQF_0 switched to PQ at MIN_Q 0 with V 1.009793 under its 1.010221 target - OLF switched it
    // back to PV at the first reactive-limit pass, the device never tested it, and CQEM_0 then switched to PQ a pass late
    // (the one contingency of 24,376 whose outer-loop sequence differed from the 1-thread CPU's).
    if (liMin >= liMax && (reactiveLimitFlags == nullptr || (reactiveLimitFlags[i] & GpuReactiveLimitsEntryFlag::NOT_CONTROLLER_ELEMENT) != 0)) {
        putReactiveLimitsDecision(passSwitched[scenarioEntry], qBuf[qIdx] + reactiveLimitLoadQ[li], liMin, liMax, x[idx], x[idx],
                 reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]));
        return;
    }
    // A re-merged LOCAL control whose own V row carries a live coupler's ZERO_V keeps its identity (mode 6, on its
    // own V column) and its own reactive balance (mode 4) on the RELOCATED row — the parent coupler's DUMMY_Q (tasks #70,
    // VC_REMERGE_RELOC_VROW). The voltage is still read at the bus's own column (idx).
    size_t targetQIdx = idx;
    if (localOv && reactiveLimitReactivePowerRow != nullptr && reactiveLimitReactivePowerRow[i] >= 0 && reactiveLimitReactivePowerRow[i] != vrow) {
        size_t r2 = (size_t) s * n + reactiveLimitReactivePowerRow[i];
        targetQIdx = crossIdentityTargetQRow(mode, r2, idx, r2, idx);
    }
    const bool crossRow = targetQIdx != idx;
    if ((mode[targetQIdx] == GpuRowMode::FIXED_VARIABLE || mode[targetQIdx] == GpuRowMode::BUS_TARGET_V_WITH_SLOPE || mode[targetQIdx] == GpuRowMode::CROSS_FIXED_VARIABLE) && limitType[scenarioEntry] == GpuQLimitType::NONE) {  // currently PV — check the reactive limits
        double q = qBuf[qIdx] + reactiveLimitLoadQ[li];
        if (q < liMin - tol) {
            savedTarget[scenarioEntry] = target[targetQIdx];                // remember the row target for the switch-back
            savedMode[scenarioEntry] = static_cast<char>(mode[targetQIdx]);   // and the PV form (0, 2 for a slope SVC, 6 cross)
            mode[targetQIdx] = localOv ? GpuRowMode::BUS_TARGET_Q : GpuRowMode::EQUATION;             // a re-merged member: own Q through its ALTERNATE dest
            target[targetQIdx] = liMin - reactiveLimitLoadQ[li];
            recordSwitchedRow(scenarioEntry, targetQIdx, true);
            limitType[scenarioEntry] = GpuQLimitType::MIN_Q;
            switchCount[scenarioEntry]++;
            switched[s] = 1;
            passSwitched[scenarioEntry] = 1;
        } else if (q > liMax + tol) {
            savedTarget[scenarioEntry] = target[targetQIdx];
            savedMode[scenarioEntry] = static_cast<char>(mode[targetQIdx]);
            mode[targetQIdx] = localOv ? GpuRowMode::BUS_TARGET_Q : GpuRowMode::EQUATION;             // a re-merged member: own Q through its ALTERNATE dest
            target[targetQIdx] = liMax - reactiveLimitLoadQ[li];
            recordSwitchedRow(scenarioEntry, targetQIdx, true);
            limitType[scenarioEntry] = GpuQLimitType::MAX_Q;
            switchCount[scenarioEntry]++;
            switched[s] = 1;
            passSwitched[scenarioEntry] = 1;
        } else {
            countRemainingPvBus();
        }
    } else if (limitType[scenarioEntry] != GpuQLimitType::NONE) {  // switched to PQ at a limit — check switchback
        // a re-merged main whose control's voltage target another controller holds reads the voltage that target holds
        int heldVoltageRow = reactiveLimitReMergedVoltageRow != nullptr ? reactiveLimitReMergedVoltageRow[scenarioEntry] : -1;
        double v = heldVoltageRow >= 0 ? x[(size_t) s * n + heldVoltageRow] : x[idx];
        double targetV = localOv ? reactiveLimitLocalTargetV[(size_t) s * nReactiveLimitEntries + i]
                  : reactiveLimitTargetVOf(nReactiveLimitTargetVOverrides, reactiveLimitTargetVOverrideScenarioEntry, reactiveLimitTargetVOverrideValue, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, i, reactiveLimitTargetV[i]);
        // OLF never back-switches a slope-SVC bus: checkPqBus runs with
        // canSwitchToPv = !bus.hasGeneratorsWithSlope() ("we don't support switching PQ to PV
        // for bus with one controller with slope") — once switched to PQ, a slope row stays PQ.
        bool atMaxLimit = isMaxLimit(limitType[scenarioEntry]);
        bool atMinLimit = isMinLimit(limitType[scenarioEntry]);
        // the cap bounds the SWITCH, not the limit refresh below (see the remote branch)
        bool wantsBack = !slopeRow
                && ((atMaxLimit && v > targetV)                               // at maxQ but v above target
                 || (atMinLimit && v < targetV));                             // at minQ but v below target
        bool back = wantsBack && switchCount[scenarioEntry] < maxSwitch;
        if (back) {
            mode[targetQIdx] = fixedVariableMode(crossRow);               // PQ → PV: resume voltage control (cross identity stays cross)
            target[targetQIdx] = targetV;
            limitType[scenarioEntry] = GpuQLimitType::NONE;
            switched[s] = 1;
            passSwitched[scenarioEntry] = 2;
        } else if (!wantsBack && (perScenarioReactiveLimits || svcLim) && (limitType[scenarioEntry] == GpuQLimitType::MIN_Q || limitType[scenarioEntry] == GpuQLimitType::MAX_Q)) {  // a realistic-V freeze keeps its target Q
            // Q-limit UPDATE while switched to PQ (OLF checkPqBusWithQLimitType): a CURVE limit that moved
            // (DS moved the gen's P) or an SVC limit that moved (B*v^2 at the live V) re-freezes
            // the frozen target Q at the NEW limit when it drifted by more than the mismatch tol.
            // Unreachable with static limits — gated on the channels.
            double lim = limitType[scenarioEntry] == GpuQLimitType::MIN_Q ? liMin : liMax;
            if (fabs(lim - (target[targetQIdx] + reactiveLimitLoadQ[li])) > tol) {
                target[targetQIdx] = lim - reactiveLimitLoadQ[li];
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

// OLF's GLOBAL forced-PV election (ReactiveLimitsOuterLoop.switchPvPq): when no bus that was PV at this check stays PV
// (remainingPvBusCount == 0, counted by reactiveLimitsBatch), the strongest candidate stays PV — highest controlled-bus
// nominal V, then highest bus target P (LIVE: the DistributedSlack may have moved it), then lowest id. One thread per
// scenario, after the reactive-limit check kernel: it UNDOES the elected candidate's PV -> PQ switch on the row that
// switch wrote (mode and target back, limit type and switch count back) and re-derives the scenario's "changed" flag
// from what the pass still did. Before this the device switched every candidate to PQ and a per-scenario DETECTOR
// routed the scenario to the CPU (1.9 kV wrong on the forced-PV network).
__global__ void reactiveLimitsForcedPvBatch(int nReactiveLimitEntries, int nScenarios, int n, const int* remainingPvBusCount,
                                     const char* reactiveLimitFlags, const double* nomV, const int* pRow, const double* staticP, const int* idRank,
                                     char* limitType, char* switchCount, char* passSwitched, const int* switchedRow, const char* savedMode,
                                     const double* savedTarget, int* mode, double* target,
                                     const char* outerLoopCheckActive, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!outerLoopCheckActive[s] || remainingPvBusCount[s] > 0) return;
    int best = -1;
    double bestV = -1.0;
    double bestP = -1e300;
    int bestId = 0x7fffffff;
    for (int i = 0; i < nReactiveLimitEntries; i++) {
        if (passSwitched[(size_t) s * nReactiveLimitEntries + i] != 1) continue;          // not switched PV -> PQ in this check
        if (reactiveLimitFlags != nullptr && (reactiveLimitFlags[i] & GpuReactiveLimitsEntryFlag::NOT_CONTROLLER_ELEMENT) != 0) continue;   // not OLF's
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
    }
    if (best < 0) return;
    size_t scenarioEntry = (size_t) s * nReactiveLimitEntries + best;
    if (switchedRow[scenarioEntry] >= 0) {                   // an enrolled pilot's switch writes no row
        size_t idx = (size_t) s * n + switchedRow[scenarioEntry];
        mode[idx] = savedMode[scenarioEntry];
        target[idx] = savedTarget[scenarioEntry];
    }
    limitType[scenarioEntry] = GpuQLimitType::NONE;
    switchCount[scenarioEntry]--;
    passSwitched[scenarioEntry] = 0;
    char any = 0;
    for (int i = 0; i < nReactiveLimitEntries && !any; i++) {
        any = passSwitched[(size_t) s * nReactiveLimitEntries + i] != 0;
    }
    switched[s] = any;
}

// Warm-continuation reconcile: a SEEDED controller switched to PQ (limitType != 0) must also carry its
// consequences — the BUS_V row flips to the reactive equation (mode 1) with the frozen Q target —
// else the solve holds the PV setpoint while the RL bookkeeping thinks the bus is at a limit.
// A pilot SWITCHED TO PQ that is one of its own controllers (reactiveLimitVoltageControlRole 2) while a member of the same group
// (same compare row = the pilot's V) still controls: OLF then keeps the pilot's V identity AND
// activates the pilot's own Q balance, hosting it on the reference member's row - a relocation this
// group has no table for on the device. Flag the scenario; the host rejects it to the CPU replay.
// The other mixed state (members switched to PQ, pilot controlling) is an ordinary layout: pilot identity,
// members at their own Q. One thread per (scenario, class-2 entry).
__global__ void pilotSelfMixedPvPqGate(int nReactiveLimitEntries, int nScenarios, const char* reactiveLimitVoltageControlRole, const int* reactiveLimitComparedVoltageRow,
                                     const char* limitType, const char* reactiveLimitIslanded, const char* outerLoopCheckActive, char* bad,
                                     const int* pilotSelfReactiveLimitEntry, int nReactiveRelocationGroups,
                                int nDistrQMembers, const int* distrQMemberReactiveLimitEntry, const char* distrQSubgroup) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nReactiveLimitEntries * nScenarios) return;
    int i = t % nReactiveLimitEntries;
    int s = t / nReactiveLimitEntries;
    if (!outerLoopCheckActive[s]) return;                    // the scenarios this check judges, not their chunk mates
    if (reactiveLimitVoltageControlRole[i] != GpuVoltageControlRole::PILOT_SELF || limitType[(size_t) s * nReactiveLimitEntries + i] == GpuQLimitType::NONE) return;  // only a pilot SWITCHED TO PQ
    // ...that is still ON this component: an islanded pilot keeps its base limit type but has no frozen target Q and no
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

// One check of ReactiveLimitsOuterLoop on every live scenario (GpuOuterLoopKind::REACTIVE_LIMITS): the curve limits at the current
// targets, the PV->PQ and PQ->PV switches, robust mode's realistic-voltage PV->PQ switches and the forced-PV election.
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
                    ctx->dReactiveLimitLoadQPerScenario.as<double>(), ctx->dOuterLoopCheckActive.as<char>());
            launchForcedTargetQLimits(ctx, nS, n, ctx->dOuterLoopCheckActive.as<char>());
        }
        CUDA_CHECK(cudaMemsetAsync(ctx->dReactiveLimitCheckDecision.p, 0, static_cast<size_t>(ctx->nReactiveLimitEntries) * nS, ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->dReactiveLimitRemainingPvBusCount.p, 0, static_cast<size_t>(nS) * sizeof(int), ctx->stream));
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
                ctx->dReactiveLimitFlags.p ? ctx->dReactiveLimitFlags.as<char>() : nullptr,
                ctx->nReactiveLimitComparedVoltageRowOverrides > 0 ? ctx->dReactiveLimitComparedVoltageRowOverride.as<int>() : nullptr,
                ctx->nReMergedVoltageTargets > 0 ? ctx->dReactiveLimitHoldsReMergedVoltageTarget.as<char>() : nullptr,
                ctx->nReMergedVoltageTargets > 0 ? ctx->dReactiveLimitReMergedVoltageRow.as<int>() : nullptr,
                ctx->dReactiveLimitRemainingPvBusCount.as<int>(), ctx->dReactiveLimitSwitchedRow.as<int>());
        if (ctx->nReactiveLimitPilotSelfEntries > 0) {
            // a pilot that is its own controller switched to PQ while a member of its group is not (or
            // the reverse): the device has no row for OLF's layout of that state - CPU replay
            if (!ctx->dPilotSelfMixedPvPqRejected.p) ctx->dPilotSelfMixedPvPqRejected.alloc(nS * sizeof(char));
            CUDA_CHECK(cudaMemsetAsync(ctx->dPilotSelfMixedPvPqRejected.p, 0, nS * sizeof(char), ctx->stream));
            pilotSelfMixedPvPqGate<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS,
                    ctx->dReactiveLimitVoltageControlRole.as<char>(), ctx->dReactiveLimitComparedVoltageRow.as<int>(), ctx->dReactiveLimitQLimitType.as<char>(),
                    ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr,
                    ctx->dOuterLoopCheckActive.as<char>(), ctx->dPilotSelfMixedPvPqRejected.as<char>(),
                    pilotSelfReactiveLimitEntryOrNull(ctx), ctx->structure->nReactiveRelocationGroups,
                ctx->nDistrQMembers, ctx->dDistrQMemberReactiveLimitEntry.p != nullptr ? ctx->dDistrQMemberReactiveLimitEntry.as<int>() : nullptr, ctx->hasDistrQSubgroups ? ctx->dDistrQSubgroup.as<char>() : nullptr);
            std::vector<char> mixed(nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(mixed.data(), ctx->dPilotSelfMixedPvPqRejected.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (mixed[s] && live[s]) {
                    traceReject(s, "pilot-self-mixed-pv-pq", outer, 0);
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
        if (ctx->hasReactiveLimitStrongestPvKeys) {
            reactiveLimitsForcedPvBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS, n,
                    ctx->dReactiveLimitRemainingPvBusCount.as<int>(), ctx->dReactiveLimitFlags.p ? ctx->dReactiveLimitFlags.as<char>() : nullptr,
                    ctx->dReactiveLimitNominalV.as<double>(), ctx->dReactiveLimitActivePowerRow.as<int>(),
                    ctx->dReactiveLimitTargetP.as<double>(), ctx->dReactiveLimitIdRank.as<int>(),
                    ctx->dReactiveLimitQLimitType.as<char>(), ctx->dReactiveLimitPvPqSwitchCount.as<char>(), ctx->dReactiveLimitCheckDecision.as<char>(),
                    ctx->dReactiveLimitSwitchedRow.as<int>(), ctx->dReactiveLimitSavedRowMode.as<char>(), ctx->dReactiveLimitSavedTarget.as<double>(),
                    ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->dOuterLoopUnstable.as<char>());
        }
        // The pilot's V-identity flip/restore runs BEFORE the DISTR_Q targets: the carrier hand-off picks the
        // carrier with chooseCarrierRow, which reads the pilot's own row (candidate 1 of a pilot-self group).
        // Measured with the order reversed (rte6515 busbar base, group VL-5997): the member that just
        // switched back was handed a target of 0 while the pilot's row was still flipped, its row then
        // hosted the pilot's balance against that 0, and it switched to PQ again every other pass until maxSwitch.
        // A pilot that LOST its own generator (GenOutage.flipsPilotRowWithoutEnabledController) is laid out here, BEFORE
        // the pilot-self flip below: that kernel's "control lost" layout puts the pilot's row in mode 1, which
        // for a pilot-that-is-its-own-controller is an EMPTY row (no q term is routed to it - the RL kernel's
        // own note), so the voltage stayed exactly at the identity's value while OLF solved the pilot's
        // BUS_TARGET_Q = 0 there. Measured on rte6515 GEN-1522: VL-1522_0 at 1.122 (the target) against OLF's
        // 1.0811, 3.2 kV at VL-1525 and 1.98e-2 MW of distributed power. Mode 4 - the own-Q balance the
        // member ALTERNATE dest routes the pilot's q to - is what the outage composer armed; run first, the
        // pilot-self kernel then finds the row already off its identity and leaves it alone.
        // A re-merged control's voltage target follows its first enabled controller first: the DISTR_Q targets below
        // are then written on the sharing rows that are left.
        launchReMergedVoltageTargetRowUpdate(ctx, drv, false);
        if (ctx->nPilotFlip > 0) {                      // pilot stopped controlling AND every survivor switched to PQ
            pilotRowWithoutEnabledControllerBatch<<<grid(ctx->nPilotFlip), BLOCK, 0, ctx->stream>>>(
                    ctx->nPilotFlip, nS, n, ctx->nReactiveLimitEntries,
                    ctx->dPfSlot.as<int>(), ctx->dPfRow.as<int>(), ctx->dPfQTarget.as<double>(),
                    ctx->dPfOff.as<int>(), ctx->dPfCount.as<int>(), ctx->dPilotFlipSurvivorReactiveLimitEntries.as<int>(),
                    ctx->dReactiveLimitQLimitType.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                    ctx->dPfFlipped.as<char>(), ctx->dPfSavedVT.as<double>(),
                    ctx->dOuterLoopCheckActive.as<char>(), ctx->dOuterLoopUnstable.as<char>());
        }
        if (ctx->hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0) {    // no-enabled-controller corner: pilot vEq flip/restore
            launchPilotTargetVEquationUpdate(ctx, drv);
        }
        if (ctx->nReactiveLimitRemoteEntries > 0 && ctx->nDistrQMembers > 0) {
            // keep the survivors' DISTR_Q row targets in step with their PV/PQ states (plan-B3)
            launchDistrQTargetsInner(ctx, drv);
        }
    }
}

}  // namespace olfgpu
