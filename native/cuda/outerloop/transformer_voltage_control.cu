// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See transformer_voltage_control.cuh.
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
#include "../common/codes/gpu_pi_model_column.cuh"
#include "../common/codes/gpu_discrete_control_member_state.cuh"
#include "../common/codes/gpu_transformer_voltage_control_mode.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "../solver/driver_steps.cuh"
#include "incremental_transformer_voltage_control.cuh"
#include "transformer_voltage_control.cuh"
#include "../common/trace.cuh"
#include "../solver/scenario_override_tables.cuh"
#include "../solver/scenario_install.cuh"

namespace olfgpu {

// OLF's non-incremental voltage-control loops as driver phases:
//  - TransformerVoltageControlOuterLoop (AFTER_GENERATOR_VOLTAGE_CONTROL): INITIAL (enable the out-of-deadband
//    controls, TransformerRatioManager bounds) -> CONTROL (clamp continuous ratios outside their range to
//    the extreme, round, disable; else updateContinuousRatio + roundVoltageRatios -> COMPLETE; UNSTABLE in
//    any case) -> COMPLETE (STABLE);
//  - SimpleTransformerVoltageControlOuterLoop (WITH_GENERATOR_VOLTAGE_CONTROL): control ON for the first solve,
//    every controller rounded to its closest tap and switched off at the first check (UNSTABLE);
// A group's ON/OFF activations are the toggle-diffed row deltas the Java side captured; member rows are
// laid out by ROLE per scenario — the first alive continuous member hosts the relocated pilot Q (the
// fills' mode scan picks the same row), the others host DISTR_RHO (target 0), fixed/outaged members are
// held by identities — so a member outage or a clamp re-forms the group exactly as OLF's "first enabled" rule.
struct TransformerVoltageControlTables {
    int nGroups, nMembers, nDeltaRows, mode;
    const int* groupMemberStart; const int* groupControlledBusVRow; const double* groupTargetV; const double* groupHalfTargetDeadband; const int* groupUseInitialTapPosition;
    const int* deltaStart; const int* deltaRow; const int* deltaModeOn; const int* deltaModeOff;
    const double* deltaTargetOn; const double* deltaTargetOff;
    const int* memberRhoRow; const int* memberElement; const int* memberBaseTap; const int* memberTapStart;
    const double* memberMinR1; const double* memberMaxR1; const double* memberInitialR1; const double* tapRows;
    int* scenarioStep; int* groupStep; double* groupSharedMinR1; double* groupSharedMaxR1; double* groupSharedInitialR1;
    int* memberState; int* memberTap; double* memberRoundingR1; double* savedDeltaTarget; char* rejected;
    // generator freeze + connectivity fix (AFTER_GENERATOR only)
    int nGeneratorVoltageControlFreezeCandidates, hasFixMasks, fixMaskWords, nReactiveLimitEntries;
    const int* generatorVoltageControlFreezeVRow; const double* generatorVoltageControlFreezeLoadTargetQ; const int* generatorVoltageControlFreezeReactiveLimitEntry; const int* fixPvRow;
    char* reactiveLimitQLimitType;                                       // the freeze marks a REMOTE member here
    const int* fixMinMask; const int* fixMaxMask;               // per (scenario, member), fixWords ints
    int* generatorVoltageControlFrozen; int* generatorVoltageControlFreezeSavedMode; double* generatorVoltageControlFreezeSavedTarget;
};

// Does any generator-voltage-CONTROLLED bus in this mask still carry an ENABLED control? Row mode 0 is
// a plain V identity and 2 a slope SVC — both are voltage control ON; a frozen bus or one its reactive limits
// switched PV -> PQ is a power row, which is exactly OLF's isGeneratorVoltageControlEnabled() == false.
__device__ inline bool anyGeneratorVoltageControlEnabled(const TransformerVoltageControlTables& T, size_t sn, const int* mode, const int* mask) {
    for (int w = 0; w < T.fixMaskWords; w++) {
        unsigned int bits = (unsigned int) mask[w];
        while (bits) {
            int b = __ffs((int) bits) - 1;
            bits &= bits - 1;
            int mr = mode[sn + T.fixPvRow[w * 32 + b]];
            if (mr == 0 || mr == 2) {
                return true;
            }
        }
    }
    return false;
}

// OLF PiModelArray.roundR1ToClosestTap: ClosestTapPositionFinder starts at the CURRENT (base) tap and
// takes a strictly closer one only.
__device__ inline double transformerVoltageControlRoundToClosestTap(const TransformerVoltageControlTables& T, int m, double r1, int* tapOut) {
    int base = T.memberTapStart[m];
    int nTaps = T.memberTapStart[m + 1] - base;
    int best = T.memberBaseTap[m];
    for (int t = 0; t < nTaps; t++) {
        if (fabs(T.tapRows[(size_t) (base + t) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1] - r1) < fabs(T.tapRows[(size_t) (base + best) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1] - r1)) best = t;
    }
    *tapOut = best;
    return T.tapRows[(size_t) (base + best) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1];
}

// Lay out one group's rows for one scenario from its members' states: ON (any alive continuous member)
// or OFF (none). Non-member delta rows (the pilot's V identity) apply as captured — the scenario's OFF
// target is saved on the way ON (a per-scenario override on the pilot's Q row must come back); member
// rows by role.
__device__ inline void transformerVoltageControlLayoutGroup(const TransformerVoltageControlTables& T, int s, int n, int nElements, int g,
                                      const char* disabled, int* mode, double* target, double* x) {
    size_t sn = (size_t) s * n;
    bool anyCont = false;
    for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
        if (!disabled[(size_t) s * nElements + T.memberElement[m]] && T.memberState[(size_t) s * T.nMembers + m] == GpuDiscreteControlMemberState::CONTINUOUS) anyCont = true;
    }
    int m0 = T.groupMemberStart[g];
    double pilotQ = 0.0;
    for (int k = T.deltaStart[g]; k < T.deltaStart[g + 1]; k++) {
        if (T.deltaRow[k] == T.memberRhoRow[m0]) pilotQ = T.deltaTargetOn[k];   // the BASE carrier's captured pilot-Q target
    }
    for (int k = T.deltaStart[g]; k < T.deltaStart[g + 1]; k++) {
        int r = T.deltaRow[k];
        bool memberRow = false;
        for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) if (T.memberRhoRow[m] == r) memberRow = true;
        if (memberRow) continue;                             // member rows are built by ROLE below
        size_t sk = (size_t) s * T.nDeltaRows + k;
        if (anyCont) {
            if (isnan(T.savedDeltaTarget[sk])) T.savedDeltaTarget[sk] = target[sn + r];
            mode[sn + r] = T.deltaModeOn[k];
            target[sn + r] = T.deltaTargetOn[k];
        } else {
            mode[sn + r] = T.deltaModeOff[k];
            target[sn + r] = isnan(T.savedDeltaTarget[sk]) ? T.deltaTargetOff[k] : T.savedDeltaTarget[sk];
            T.savedDeltaTarget[sk] = nan("");
        }
    }
    // The scenario's own move of the controlled bus's reactive balance: an injection lost there is written host-side
    // as that bus's Q target while the control is OFF, and saved above on the way ON. The carrier hosts the same
    // balance while ON, so it takes the same move on top of the captured BASE target - without it the lost injection
    // stayed in the balance the tap was decided on (a 5 MVar load at the controlled bus: 6.0e-3 kV).
    if (anyCont) {
        for (int k = T.deltaStart[g]; k < T.deltaStart[g + 1]; k++) {
            if (T.deltaRow[k] != T.groupControlledBusVRow[g] || T.deltaModeOff[k] != GpuRowMode::EQUATION) continue;
            double sv = T.savedDeltaTarget[(size_t) s * T.nDeltaRows + k];
            if (!isnan(sv)) pilotQ += sv - T.deltaTargetOff[k];
        }
    }
    bool carrier = false;
    for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
        if (disabled[(size_t) s * nElements + T.memberElement[m]]) continue;   // outaged: base identity as uploaded
        size_t mr = sn + T.memberRhoRow[m];
        int memberState = T.memberState[(size_t) s * T.nMembers + m];
        if (anyCont && memberState == GpuDiscreteControlMemberState::CONTINUOUS) {  // ALIVE continuous: carrier (pilot Q) or DISTR_RHO (0)
            mode[mr] = GpuRowMode::EQUATION;
            target[mr] = carrier ? 0.0 : pilotQ;
            carrier = true;
        } else if (memberState == GpuDiscreteControlMemberState::ROUNDED) {  // identity holding the rounded tap's r1
            int tap = T.memberTap[(size_t) s * T.nMembers + m];
            double r1 = T.tapRows[(size_t) (T.memberTapStart[m] + tap) * GpuPiModelColumn::COUNT + GpuPiModelColumn::R1];
            mode[mr] = GpuRowMode::FIXED_VARIABLE;
            target[mr] = r1;
            x[mr] = r1;
        }
    }
}

// Solve entry (OLF initialize): WITH_GENERATOR enables every connected controller before the first solve —
// applied here for a chunk (entryOn = 0), or read off the uploaded activation for a base load flow whose
// network OLF's own initialize already switched on (entryOn = 1). AFTER_GENERATOR starts with every
// control off.
__global__ void transformerVoltageControlEntryBatch(TransformerVoltageControlTables T, int nScenarios, int n, int nElements, int entryOn,
                              const char* disabled, int* mode, double* target, double* x, const char* installed) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (installed != nullptr && !installed[s]) return;   // a slot this install does not seed
    size_t sn = (size_t) s * n;
    for (int g = 0; g < T.nGroups; g++) {
        bool any = false;
        for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
            if (disabled[(size_t) s * nElements + T.memberElement[m]]) continue;
            size_t sm = (size_t) s * T.nMembers + m;
            if (entryOn) {
                T.memberState[sm] = mode[sn + T.memberRhoRow[m]] == GpuRowMode::EQUATION ? GpuDiscreteControlMemberState::CONTINUOUS : GpuDiscreteControlMemberState::DISABLED;
            } else if (T.mode == GpuTransformerVoltageControlMode::WITH_GENERATOR_VOLTAGE_CONTROL) {
                T.memberState[sm] = GpuDiscreteControlMemberState::CONTINUOUS;
                any = true;
            }
        }
        if (any) transformerVoltageControlLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
    }
}

// One thread per scenario (OLF keeps ONE step for all controls of a run).
__global__ void transformerVoltageControlCheckBatch(TransformerVoltageControlTables T, int nScenarios, int n, int nElements, const char* disabled, const char* outerLoopCheckActive,
                              const double* qBuf, int* mode, double* target, double* x, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !outerLoopCheckActive[s]) return;
    size_t sn = (size_t) s * n;
    if (T.mode == GpuTransformerVoltageControlMode::WITH_GENERATOR_VOLTAGE_CONTROL) {                                       // SimpleTransformerVoltageControl: iteration 0 rounds all
        bool anyCtl = false;
        for (int g = 0; g < T.nGroups; g++) {
            size_t sg = (size_t) s * T.nGroups + g;
            if (T.groupStep[sg] != 0) continue;
            bool touched = false;
            for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
                if (disabled[(size_t) s * nElements + T.memberElement[m]]) continue;
                anyCtl = true;
                size_t sm = (size_t) s * T.nMembers + m;
                if (T.memberState[sm] != GpuDiscreteControlMemberState::CONTINUOUS) continue;
                T.memberRoundingR1[sm] = x[sn + T.memberRhoRow[m]];
                int tap;
                transformerVoltageControlRoundToClosestTap(T, m, T.memberRoundingR1[sm], &tap);
                T.memberTap[sm] = tap;
                T.memberState[sm] = GpuDiscreteControlMemberState::ROUNDED;
                touched = true;
            }
            T.groupStep[sg] = 2;
            if (touched) transformerVoltageControlLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
        }
        if (anyCtl) switched[s] = 1;                         // OLF: UNSTABLE whenever a live controller exists
        return;
    }
    int step = T.scenarioStep[s];
    if (step == 2) return;                                   // COMPLETE: STABLE
    if (step == 0) {                                         // INITIAL
        bool needRun = false;
        for (int g = 0; g < T.nGroups; g++) {
            int alive = 0;
            for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) if (!disabled[(size_t) s * nElements + T.memberElement[m]]) alive++;
            if (alive == 0) continue;
            if (!(fabs(T.groupTargetV[g] - x[sn + T.groupControlledBusVRow[g]]) > T.groupHalfTargetDeadband[g])) continue;
            // TransformerRatioManager over the just-enabled members: group-averaged bounds and the
            // normalized-headroom initial (the exact identity for a single member)
            double sumMin = 0.0, sumMax = 0.0, a = 0.0, b = 0.0;
            for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
                if (disabled[(size_t) s * nElements + T.memberElement[m]]) continue;
                T.memberState[(size_t) s * T.nMembers + m] = GpuDiscreteControlMemberState::CONTINUOUS;
                sumMin += T.memberMinR1[m];
                sumMax += T.memberMaxR1[m];
                if (T.memberMaxR1[m] != T.memberMinR1[m]) {
                    a += (T.memberMaxR1[m] - T.memberInitialR1[m]) / (T.memberMaxR1[m] - T.memberMinR1[m]);
                    b += (T.memberInitialR1[m] - T.memberMinR1[m]) / (T.memberMaxR1[m] - T.memberMinR1[m]);
                } else {
                    a += 1.0;
                    b += 1.0;
                }
            }
            size_t sg = (size_t) s * T.nGroups + g;
            T.groupSharedMinR1[sg] = sumMin / alive;
            T.groupSharedMaxR1[sg] = sumMax / alive;
            T.groupSharedInitialR1[sg] = (T.groupSharedMinR1[sg] * a + T.groupSharedMaxR1[sg] * b) / alive;
            needRun = true;
        }
        if (!needRun) {
            T.scenarioStep[s] = 2;                                // every control in band: COMPLETE, STABLE
            return;
        }
        // GeneratorVoltageControlManager.disableGeneratorVoltageControlsUnderMaxControlledNominalVoltage:
        // every candidate whose control is ON right now (not switched to PQ by its reactive limits) freezes at its current Q
        for (int c = 0; c < T.nGeneratorVoltageControlFreezeCandidates; c++) {
            size_t sr = sn + T.generatorVoltageControlFreezeVRow[c];
            size_t sc = (size_t) s * T.nGeneratorVoltageControlFreezeCandidates + c;
            if (T.generatorVoltageControlFrozen[sc]) continue;
            int re = T.generatorVoltageControlFreezeReactiveLimitEntry[c];
            if (re >= 0) {
                // REMOTE controller. OLF freezes it exactly like a local one, but its own row is the
                // group's DISTR_Q SHARE, not a V identity — so the freeze is the reactive-limit remote
                // PV -> PQ SWITCH (mode 4, "the row hosts its own Q balance") with the CURRENT injection as the
                // target instead of a Q limit. The limit type is what the DISTR_Q key renormalization
                // reads to drop a member, so it has to be set, and GpuQLimitType::FROZEN keeps it
                // distinguishable from a real Q-limit PV -> PQ switch (which the switch-back may undo; this must
                // not be, OLF's reactive-limits loop never sees a bus whose voltage control is off).
                size_t scenarioEntry = (size_t) s * T.nReactiveLimitEntries + re;
                if (mode[sr] != GpuRowMode::EQUATION || T.reactiveLimitQLimitType[scenarioEntry] != GpuQLimitType::NONE) continue;  // switched to PQ already, or not live
                T.generatorVoltageControlFreezeSavedMode[sc] = mode[sr];
                T.generatorVoltageControlFreezeSavedTarget[sc] = target[sr];
                mode[sr] = GpuRowMode::BUS_TARGET_Q;
                target[sr] = qBuf[sr] - T.generatorVoltageControlFreezeLoadTargetQ[c];
                T.reactiveLimitQLimitType[scenarioEntry] = GpuQLimitType::FROZEN;
                T.generatorVoltageControlFrozen[sc] = 1;
                continue;
            }
            if (mode[sr] != GpuRowMode::FIXED_VARIABLE && mode[sr] != GpuRowMode::BUS_TARGET_V_WITH_SLOPE) continue;
            T.generatorVoltageControlFreezeSavedMode[sc] = mode[sr];
            T.generatorVoltageControlFreezeSavedTarget[sc] = target[sr];
            mode[sr] = GpuRowMode::EQUATION;
            target[sr] = qBuf[sr] - T.generatorVoltageControlFreezeLoadTargetQ[c];            // generationTargetQ = q.eval(); row target = gen − load
            T.generatorVoltageControlFrozen[sc] = 1;
        }
        // LfNetwork.fixTransformerVoltageControls: a just-enabled controller whose not-controlled side's
        // component holds no ENABLED generator voltage control (frozen buses and buses switched to PQ count as off)
        // is disabled again. OLF cuts the CONTINGENCY's branches and the ENABLED controller branches;
        // the host knows the first and brackets the second with two masks (see FreezeTables), so a PV
        // bus in the MIN mask is one OLF sees too, and no PV bus in the MAX mask means OLF sees none.
        // Between the two the answer depends on which controllers THIS scenario enabled, which is not
        // something the host could precompute — that scenario goes to the CPU.
        if (T.hasFixMasks) {
            for (int g = 0; g < T.nGroups; g++) {
                for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
                    size_t sm = (size_t) s * T.nMembers + m;
                    if (T.memberState[sm] != GpuDiscreteControlMemberState::CONTINUOUS) continue;
                    size_t base = ((size_t) s * T.nMembers + m) * T.fixMaskWords;
                    bool pvMin = anyGeneratorVoltageControlEnabled(T, sn, mode, T.fixMinMask + base);
                    if (pvMin) continue;                       // OLF keeps this control enabled
                    if (anyGeneratorVoltageControlEnabled(T, sn, mode, T.fixMaxMask + base)) {
                        T.rejected[s] = 1;                       // ambiguous without the enabled set
                        return;
                    }
                    T.memberState[sm] = GpuDiscreteControlMemberState::DISABLED;  // OLF disables it
                }
            }
        }
        for (int g = 0; g < T.nGroups; g++) {
            bool any = false;
            for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) any |= T.memberState[(size_t) s * T.nMembers + m] == GpuDiscreteControlMemberState::CONTINUOUS;
            if (any) transformerVoltageControlLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
        }
        T.scenarioStep[s] = 1;
        switched[s] = 1;
        return;
    }
    // CONTROL: roundR1ToExtremeTapPosition on every enabled controller
    bool outOfBound = false;
    for (int g = 0; g < T.nGroups; g++) {
        size_t sg = (size_t) s * T.nGroups + g;
        bool useInit = T.groupUseInitialTapPosition[g] != 0;
        bool touched = false;
        for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
            if (disabled[(size_t) s * nElements + T.memberElement[m]]) continue;
            size_t sm = (size_t) s * T.nMembers + m;
            if (T.memberState[sm] != GpuDiscreteControlMemberState::CONTINUOUS) continue;
            double r1 = x[sn + T.memberRhoRow[m]];
            double lo = useInit ? T.groupSharedMinR1[sg] : T.memberMinR1[m];
            double hi = useInit ? T.groupSharedMaxR1[sg] : T.memberMaxR1[m];
            if (!(r1 < lo || r1 > hi)) continue;
            double v = r1 > hi ? hi : lo;
            T.memberRoundingR1[sm] = v;
            int tap;
            transformerVoltageControlRoundToClosestTap(T, m, v, &tap);
            T.memberTap[sm] = tap;
            T.memberState[sm] = GpuDiscreteControlMemberState::ROUNDED;
            touched = true;
        }
        if (touched) {
            transformerVoltageControlLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
            outOfBound = true;
        }
    }
    if (outOfBound) {
        switched[s] = 1;                                     // stays CONTROL; OLF: UNSTABLE
        return;
    }
    // none out of range: updateContinuousRatio (useInitialTapPosition remap) + roundVoltageRatios -> COMPLETE
    for (int g = 0; g < T.nGroups; g++) {
        size_t sg = (size_t) s * T.nGroups + g;
        bool useInit = T.groupUseInitialTapPosition[g] != 0;
        bool touched = false;
        for (int m = T.groupMemberStart[g]; m < T.groupMemberStart[g + 1]; m++) {
            if (disabled[(size_t) s * nElements + T.memberElement[m]]) continue;
            size_t sm = (size_t) s * T.nMembers + m;
            if (T.memberState[sm] != GpuDiscreteControlMemberState::CONTINUOUS) continue;
            double v = x[sn + T.memberRhoRow[m]];
            if (useInit) {
                double init = T.groupSharedInitialR1[sg];
                v = v >= init
                        ? T.memberInitialR1[m] + (v - init) * (T.memberMaxR1[m] - T.memberInitialR1[m]) / (T.groupSharedMaxR1[sg] - init)
                        : T.memberInitialR1[m] - (init - v) * (T.memberInitialR1[m] - T.memberMinR1[m]) / (init - T.groupSharedMinR1[sg]);
            }
            T.memberRoundingR1[sm] = v;
            int tap;
            transformerVoltageControlRoundToClosestTap(T, m, v, &tap);
            T.memberTap[sm] = tap;
            T.memberState[sm] = GpuDiscreteControlMemberState::ROUNDED;
            touched = true;
        }
        if (touched) transformerVoltageControlLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
    }
    // enableGeneratorVoltageControlsUnderMaxControlledNominalVoltage: the frozen controllers resume
    for (int c = 0; c < T.nGeneratorVoltageControlFreezeCandidates; c++) {
        size_t sc = (size_t) s * T.nGeneratorVoltageControlFreezeCandidates + c;
        if (!T.generatorVoltageControlFrozen[sc]) continue;
        size_t sr = sn + T.generatorVoltageControlFreezeVRow[c];
        mode[sr] = T.generatorVoltageControlFreezeSavedMode[sc];
        target[sr] = T.generatorVoltageControlFreezeSavedTarget[sc];
        if (T.generatorVoltageControlFreezeReactiveLimitEntry[c] >= 0) {                           // a remote member rejoins the DISTR_Q shares
            T.reactiveLimitQLimitType[(size_t) s * T.nReactiveLimitEntries + T.generatorVoltageControlFreezeReactiveLimitEntry[c]] = GpuQLimitType::NONE;
        }
        T.generatorVoltageControlFrozen[sc] = 0;
    }
    T.scenarioStep[s] = 2;
    switched[s] = 1;                                         // OLF controlStep: UNSTABLE in any case (the COMPLETE solve)
}

static TransformerVoltageControlTables transformerVoltageControlTables(BatchSolverContext* ctx) {
    TransformerVoltageControlTables T;
    T.nGroups = ctx->nTransformerVoltageControlGroups; T.nMembers = ctx->nTransformerVoltageControlMembers; T.nDeltaRows = ctx->nTransformerVoltageControlDeltaRows; T.mode = ctx->transformerVoltageControlMode;
    T.groupMemberStart = ctx->dTransformerVoltageControlGroupMemberStart.as<int>(); T.groupControlledBusVRow = ctx->dTransformerVoltageControlGroupControlledBusVRow.as<int>();
    T.groupTargetV = ctx->dTransformerVoltageControlGroupTargetV.as<double>(); T.groupHalfTargetDeadband = ctx->dTransformerVoltageControlGroupHalfTargetDeadband.as<double>(); T.groupUseInitialTapPosition = ctx->dTransformerVoltageControlGroupUseInitialTapPosition.as<int>();
    T.deltaStart = ctx->dTransformerVoltageControlDeltaStart.as<int>(); T.deltaRow = ctx->dTransformerVoltageControlDeltaRow.as<int>();
    T.deltaModeOn = ctx->dTransformerVoltageControlDeltaModeOn.as<int>(); T.deltaModeOff = ctx->dTransformerVoltageControlDeltaModeOff.as<int>();
    T.deltaTargetOn = ctx->dTransformerVoltageControlDeltaTargetOn.as<double>(); T.deltaTargetOff = ctx->dTransformerVoltageControlDeltaTargetOff.as<double>();
    T.memberRhoRow = ctx->dTransformerVoltageControlMemberRhoRow.as<int>(); T.memberElement = ctx->dTransformerVoltageControlMemberElement.as<int>();
    T.memberBaseTap = ctx->dTransformerVoltageControlMemberBaseTap.as<int>(); T.memberTapStart = ctx->dTransformerVoltageControlMemberTapStart.as<int>();
    T.memberMinR1 = ctx->dTransformerVoltageControlMemberMinR1.as<double>(); T.memberMaxR1 = ctx->dTransformerVoltageControlMemberMaxR1.as<double>(); T.memberInitialR1 = ctx->dTransformerVoltageControlMemberInitialR1.as<double>();
    T.tapRows = ctx->dTransformerVoltageControlTapRows.as<double>();
    T.scenarioStep = ctx->dTransformerVoltageControlStep.as<int>(); T.groupStep = ctx->dTransformerVoltageControlGroupStep.as<int>();
    T.groupSharedMinR1 = ctx->dTransformerVoltageControlGroupSharedMinR1.as<double>(); T.groupSharedMaxR1 = ctx->dTransformerVoltageControlGroupSharedMaxR1.as<double>(); T.groupSharedInitialR1 = ctx->dTransformerVoltageControlGroupSharedInitialR1.as<double>();
    T.memberState = ctx->dTransformerVoltageControlMemberState.as<int>(); T.memberTap = ctx->dTransformerVoltageControlMemberTap.as<int>(); T.memberRoundingR1 = ctx->dTransformerVoltageControlMemberRoundingR1.as<double>();
    T.savedDeltaTarget = ctx->dTransformerVoltageControlSavedDeltaTarget.as<double>(); T.rejected = ctx->dTransformerVoltageControlRejected.as<char>();
    T.nGeneratorVoltageControlFreezeCandidates = ctx->nGeneratorVoltageControlFreezeCandidates; T.hasFixMasks = ctx->transformerVoltageControlFixMinMask.on() ? 1 : 0; T.fixMaskWords = ctx->transformerVoltageControlFixWords;
    T.generatorVoltageControlFreezeVRow = ctx->dGeneratorVoltageControlFreezeVRow.as<int>(); T.generatorVoltageControlFreezeLoadTargetQ = ctx->dGeneratorVoltageControlFreezeLoadTargetQ.as<double>();
    T.generatorVoltageControlFreezeReactiveLimitEntry = ctx->dGeneratorVoltageControlFreezeReactiveLimitEntry.as<int>(); T.nReactiveLimitEntries = ctx->nReactiveLimitEntries; T.reactiveLimitQLimitType = ctx->dReactiveLimitQLimitType.as<char>();
    T.fixPvRow = ctx->dTransformerVoltageControlFixPvRow.as<int>();
    T.fixMinMask = ctx->transformerVoltageControlFixMinMask.device(); T.fixMaxMask = ctx->transformerVoltageControlFixMaxMask.device();
    T.generatorVoltageControlFrozen = ctx->dGeneratorVoltageControlFrozen.as<int>(); T.generatorVoltageControlFreezeSavedMode = ctx->dGeneratorVoltageControlFreezeSavedMode.as<int>(); T.generatorVoltageControlFreezeSavedTarget = ctx->dGeneratorVoltageControlFreezeSavedTarget.as<double>();
    return T;
}

void transformerVoltageControlStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& loopRejected = drv.loopRejected;
    if (ctx->nTransformerVoltageControlMembers <= 0) return;
    transformerVoltageControlCheckBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(transformerVoltageControlTables(ctx), nS, n, ctx->nElements, ctx->dDisabled.as<char>(),
            ctx->dOuterLoopCheckActive.as<char>(), ctx->dQbuf.as<double>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>(),
            ctx->dOuterLoopUnstable.as<char>());
    // Freezing a REMOTE controller can take the LAST live member of its group, and then the group
    // controls nothing: OLF's freeze disables the voltage control, so the CONTROLLED bus stops
    // carrying a V identity. That flip lives in its own kernel, which the reactive-limits phase runs
    // after its PV -> PQ switches; the freeze happens two loops later and would otherwise leave the pilot holding
    // a target for a control that no longer exists until the next sweep reached RL again (measured
    // 0.6 kV). Same kernel, run again here — it restores the identity at COMPLETE on the same rule.
    if (ctx->hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0 && ctx->nGeneratorVoltageControlFreezeCandidates > 0) {
        launchPilotTargetVEquationUpdate(ctx, drv);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<char> rej(nS);
    CUDA_CHECK(cudaMemcpy(rej.data(), ctx->dTransformerVoltageControlRejected.p, nS, cudaMemcpyDeviceToHost));
    for (int s2 = 0; s2 < nS; s2++) if (rej[s2]) { loopRejected[s2] = 1; traceReject(s2, "tvd-kernel-reject", outer, 0); }
    publishClosedBranchParameterOverrides(ctx);
    if (traceOn(Trace::BATCH_TRACE)) {
        size_t nsm = static_cast<size_t>(nS) * ctx->nTransformerVoltageControlMembers;
        std::vector<int> memberState(nsm), step(nS);
        CUDA_CHECK(cudaMemcpy(memberState.data(), ctx->dTransformerVoltageControlMemberState.p, nsm * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(step.data(), ctx->dTransformerVoltageControlStep.p, nS * sizeof(int), cudaMemcpyDeviceToHost));
        int nCont = 0, nFixed = 0, nRej = 0;
        for (size_t k = 0; k < nsm; k++) { nCont += memberState[k] == GpuDiscreteControlMemberState::CONTINUOUS; nFixed += memberState[k] == GpuDiscreteControlMemberState::ROUNDED; }
        for (int s2 = 0; s2 < nS; s2++) nRej += rej[s2] ? 1 : 0;
        std::fprintf(stderr, "TVD_KERNEL mode=%d nS=%d nMembers=%d continuous=%d fixed=%d rejected=%d step0=%d\n",
                     ctx->transformerVoltageControlMode, nS, ctx->nTransformerVoltageControlMembers, nCont, nFixed, nRej, step[0]);
    }
}

// Per-solve reset. Discrete TVC: every control at its entry state.
void transformerVoltageControlResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const double nan = std::nan("");
    const size_t nm = static_cast<size_t>(ctx->nTransformerVoltageControlMembers);
    const size_t ng = static_cast<size_t>(ctx->nTransformerVoltageControlGroups);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    resetInstalledRows(ctx, ctx->dTransformerVoltageControlMemberTap.as<int>(), nm, ctx->hTransformerVoltageControlMemberBaseTap.data());
    resetInstalledRows(ctx, ctx->dTransformerVoltageControlMemberState.as<int>(), nm, nullptr, 0);
    resetInstalledRows(ctx, ctx->dTransformerVoltageControlMemberRoundingR1.as<double>(), nm, nullptr, nan);
    resetInstalledRows(ctx, ctx->dTransformerVoltageControlSavedDeltaTarget.as<double>(),
                       static_cast<size_t>(std::max(ctx->nTransformerVoltageControlDeltaRows, 1)), nullptr, nan);
    resetInstalledRows(ctx, ctx->dTransformerVoltageControlStep.as<int>(), 1, nullptr, 0);
    resetInstalledRows(ctx, ctx->dTransformerVoltageControlGroupStep.as<int>(), ng, nullptr, 0);
    resetInstalledRows(ctx, ctx->dTransformerVoltageControlGroupSharedMinR1.as<double>(), ng, nullptr, 0.0);
    resetInstalledRows(ctx, ctx->dTransformerVoltageControlGroupSharedMaxR1.as<double>(), ng, nullptr, 0.0);
    resetInstalledRows(ctx, ctx->dTransformerVoltageControlGroupSharedInitialR1.as<double>(), ng, nullptr, 0.0);
    resetInstalledRows(ctx, ctx->dTransformerVoltageControlRejected.as<char>(), 1, nullptr, char(0));
    if (ctx->nGeneratorVoltageControlFreezeCandidates > 0) {
        resetInstalledRows(ctx, ctx->dGeneratorVoltageControlFrozen.as<int>(), ctx->nGeneratorVoltageControlFreezeCandidates, nullptr, 0);
    }
    transformerVoltageControlEntryBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(transformerVoltageControlTables(ctx), nS, n, ctx->nElements, ctx->transformerVoltageControlEntryOn,
            ctx->dDisabled.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>(), installMaskOrNull(ctx));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

}  // namespace olfgpu
