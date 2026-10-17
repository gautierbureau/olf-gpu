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
#include "../solver/driver_steps.cuh"
#include "incremental_transformer_voltage_control.cuh"
#include "transformer_voltage_control.cuh"

namespace olfgpu {

// OLF's non-incremental voltage-control loops as driver phases:
//  - TransformerVoltageControlOuterLoop (AFTER_GENERATOR, tvdMode 0): INITIAL (enable the out-of-deadband
//    controls, TransformerRatioManager bounds) -> CONTROL (clamp continuous ratios outside their range to
//    the extreme, round, disable; else updateContinuousRatio + roundVoltageRatios -> COMPLETE; UNSTABLE in
//    any case) -> COMPLETE (STABLE);
//  - SimpleTransformerVoltageControlOuterLoop (WITH_GENERATOR, tvdMode 1): control ON for the first solve,
//    every controller rounded to its closest tap and switched off at the first check (UNSTABLE);
// A group's ON/OFF activations are the toggle-diffed row deltas the Java side captured; member rows are
// laid out by ROLE per scenario — the first alive continuous member hosts the relocated pilot Q (the
// fills' mode scan picks the same row), the others host DISTR_RHO (target 0), fixed/outaged members are
// identity pins — so a member outage or a clamp re-forms the group exactly as OLF's "first enabled" rule.
struct TvdTables {
    int nG, nM, nDelta, mode;
    const int* grpPtr; const int* grpVRow; const double* grpTargetV; const double* grpHalfDb; const int* grpUseInit;
    const int* deltaPtr; const int* deltaRow; const int* deltaModeOn; const int* deltaModeOff;
    const double* deltaTargetOn; const double* deltaTargetOff;
    const int* memRhoRow; const int* memElem; const int* memBaseTap; const int* memTapPtr;
    const double* memMin; const double* memMax; const double* memInit; const double* tapData;
    int* scnStep; int* grpStep; double* grpShMin; double* grpShMax; double* grpShInit;
    int* memState; int* memTap; double* memRoundFrom; double* saved; char* reject;
    // generator freeze + connectivity fix (AFTER_GENERATOR only)
    int nFz, hasFix, fixWords, nrl;
    const int* fzVRow; const double* fzLoadQ; const int* fzRlEntry; const int* fixPvRow;
    char* rlLimitType;                                       // the freeze marks a REMOTE member here
    const int* fixMinMask; const int* fixMaxMask;               // per (scenario, member), fixWords ints
    int* fzFrozen; int* fzSavedMode; double* fzSavedTarget;
};

// Does any generator-voltage-CONTROLLED bus in this mask still carry an ENABLED control? Row mode 0 is
// a plain V identity and 2 a slope SVC — both are voltage control ON; a frozen or reactive-limit-pinned
// bus is a power row, which is exactly OLF's isGeneratorVoltageControlEnabled() == false.
__device__ inline bool tvdAnyPv(const TvdTables& T, size_t sn, const int* mode, const int* mask) {
    for (int w = 0; w < T.fixWords; w++) {
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
__device__ inline double tvdRoundToTap(const TvdTables& T, int m, double r1, int* tapOut) {
    int base = T.memTapPtr[m];
    int nTaps = T.memTapPtr[m + 1] - base;
    int best = T.memBaseTap[m];
    for (int t = 0; t < nTaps; t++) {
        if (fabs(T.tapData[(size_t) (base + t) * 8 + 6] - r1) < fabs(T.tapData[(size_t) (base + best) * 8 + 6] - r1)) best = t;
    }
    *tapOut = best;
    return T.tapData[(size_t) (base + best) * 8 + 6];
}

// Lay out one group's rows for one scenario from its members' states: ON (any alive continuous member)
// or OFF (none). Non-member delta rows (the pilot's V identity) apply as captured — the scenario's OFF
// target is saved on the way ON (a per-scenario override on the pilot's Q row must come back); member
// rows by role.
__device__ inline void tvdLayoutGroup(const TvdTables& T, int s, int n, int nElements, int g,
                                      const char* disabled, int* mode, double* target, double* x) {
    size_t sn = (size_t) s * n;
    bool anyCont = false;
    for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
        if (!disabled[(size_t) s * nElements + T.memElem[m]] && T.memState[(size_t) s * T.nM + m] == 1) anyCont = true;
    }
    int m0 = T.grpPtr[g];
    double pilotQ = 0.0;
    for (int k = T.deltaPtr[g]; k < T.deltaPtr[g + 1]; k++) {
        if (T.deltaRow[k] == T.memRhoRow[m0]) pilotQ = T.deltaTargetOn[k];   // the BASE carrier's captured pilot-Q target
    }
    for (int k = T.deltaPtr[g]; k < T.deltaPtr[g + 1]; k++) {
        int r = T.deltaRow[k];
        bool memberRow = false;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) if (T.memRhoRow[m] == r) memberRow = true;
        if (memberRow) continue;                             // member rows are built by ROLE below
        size_t sk = (size_t) s * T.nDelta + k;
        if (anyCont) {
            if (isnan(T.saved[sk])) T.saved[sk] = target[sn + r];
            mode[sn + r] = T.deltaModeOn[k];
            target[sn + r] = T.deltaTargetOn[k];
        } else {
            mode[sn + r] = T.deltaModeOff[k];
            target[sn + r] = isnan(T.saved[sk]) ? T.deltaTargetOff[k] : T.saved[sk];
            T.saved[sk] = nan("");
        }
    }
    // The scenario's own move of the controlled bus's reactive balance: an injection lost there is written host-side
    // as that bus's Q target while the control is OFF, and saved above on the way ON. The carrier hosts the same
    // balance while ON, so it takes the same move on top of the captured BASE target - without it the lost injection
    // stayed in the balance the tap was decided on (a 5 MVar load at the controlled bus: 6.0e-3 kV).
    if (anyCont) {
        for (int k = T.deltaPtr[g]; k < T.deltaPtr[g + 1]; k++) {
            if (T.deltaRow[k] != T.grpVRow[g] || T.deltaModeOff[k] != 1) continue;
            double sv = T.saved[(size_t) s * T.nDelta + k];
            if (!isnan(sv)) pilotQ += sv - T.deltaTargetOff[k];
        }
    }
    bool carrier = false;
    for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
        if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;   // outaged: base identity as uploaded
        size_t mr = sn + T.memRhoRow[m];
        int st = T.memState[(size_t) s * T.nM + m];
        if (anyCont && st == 1) {                            // ALIVE continuous: carrier (pilot Q) or DISTR_RHO (0)
            mode[mr] = 1;
            target[mr] = carrier ? 0.0 : pilotQ;
            carrier = true;
        } else if (st == 2) {                                // FIXED: identity pin at the rounded tap's r1
            int tap = T.memTap[(size_t) s * T.nM + m];
            double r1 = T.tapData[(size_t) (T.memTapPtr[m] + tap) * 8 + 6];
            mode[mr] = 0;
            target[mr] = r1;
            x[mr] = r1;
        }
    }
}

// Solve entry (OLF initialize): WITH_GENERATOR enables every connected controller before the first solve —
// applied here for a chunk (entryOn = 0), or read off the uploaded activation for a base load flow whose
// network OLF's own initialize already switched on (entryOn = 1). AFTER_GENERATOR starts with every
// control off.
__global__ void tvdEntryBatch(TvdTables T, int nScenarios, int n, int nElements, int entryOn,
                              const char* disabled, int* mode, double* target, double* x) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    size_t sn = (size_t) s * n;
    for (int g = 0; g < T.nG; g++) {
        bool any = false;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;
            size_t sm = (size_t) s * T.nM + m;
            if (entryOn) {
                T.memState[sm] = mode[sn + T.memRhoRow[m]] == 1 ? 1 : 0;
            } else if (T.mode == 1) {
                T.memState[sm] = 1;
                any = true;
            }
        }
        if (any) tvdLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
    }
}

// One thread per scenario (OLF keeps ONE step for all controls of a run).
__global__ void tvdCheckBatch(TvdTables T, int nScenarios, int n, int nElements, const char* disabled, const char* rlActive,
                              const double* qBuf, int* mode, double* target, double* x, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !rlActive[s]) return;
    size_t sn = (size_t) s * n;
    if (T.mode == 1) {                                       // SimpleTransformerVoltageControl: iteration 0 rounds all
        bool anyCtl = false;
        for (int g = 0; g < T.nG; g++) {
            size_t sg = (size_t) s * T.nG + g;
            if (T.grpStep[sg] != 0) continue;
            bool touched = false;
            for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
                if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;
                anyCtl = true;
                size_t sm = (size_t) s * T.nM + m;
                if (T.memState[sm] != 1) continue;
                T.memRoundFrom[sm] = x[sn + T.memRhoRow[m]];
                int tap;
                tvdRoundToTap(T, m, T.memRoundFrom[sm], &tap);
                T.memTap[sm] = tap;
                T.memState[sm] = 2;
                touched = true;
            }
            T.grpStep[sg] = 2;
            if (touched) tvdLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
        }
        if (anyCtl) switched[s] = 1;                         // OLF: UNSTABLE whenever a live controller exists
        return;
    }
    int step = T.scnStep[s];
    if (step == 2) return;                                   // COMPLETE: STABLE
    if (step == 0) {                                         // INITIAL
        bool needRun = false;
        for (int g = 0; g < T.nG; g++) {
            int alive = 0;
            for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) if (!disabled[(size_t) s * nElements + T.memElem[m]]) alive++;
            if (alive == 0) continue;
            if (!(fabs(T.grpTargetV[g] - x[sn + T.grpVRow[g]]) > T.grpHalfDb[g])) continue;
            // TransformerRatioManager over the just-enabled members: group-averaged bounds and the
            // normalized-headroom initial (the exact identity for a single member)
            double sumMin = 0.0, sumMax = 0.0, a = 0.0, b = 0.0;
            for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
                if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;
                T.memState[(size_t) s * T.nM + m] = 1;
                sumMin += T.memMin[m];
                sumMax += T.memMax[m];
                if (T.memMax[m] != T.memMin[m]) {
                    a += (T.memMax[m] - T.memInit[m]) / (T.memMax[m] - T.memMin[m]);
                    b += (T.memInit[m] - T.memMin[m]) / (T.memMax[m] - T.memMin[m]);
                } else {
                    a += 1.0;
                    b += 1.0;
                }
            }
            size_t sg = (size_t) s * T.nG + g;
            T.grpShMin[sg] = sumMin / alive;
            T.grpShMax[sg] = sumMax / alive;
            T.grpShInit[sg] = (T.grpShMin[sg] * a + T.grpShMax[sg] * b) / alive;
            needRun = true;
        }
        if (!needRun) {
            T.scnStep[s] = 2;                                // every control in band: COMPLETE, STABLE
            return;
        }
        // GeneratorVoltageControlManager.disableGeneratorVoltageControlsUnderMaxControlledNominalVoltage:
        // every candidate whose control is ON right now (not RL-pinned) freezes at its current Q
        for (int c = 0; c < T.nFz; c++) {
            size_t sr = sn + T.fzVRow[c];
            size_t sc = (size_t) s * T.nFz + c;
            if (T.fzFrozen[sc]) continue;
            int re = T.fzRlEntry[c];
            if (re >= 0) {
                // REMOTE controller. OLF freezes it exactly like a local one, but its own row is the
                // group's DISTR_Q SHARE, not a V identity — so the freeze is the reactive-limit remote
                // PIN (mode 4, "the row hosts its own Q balance") with the CURRENT injection as the
                // target instead of a Q limit. The limit type is what the DISTR_Q key renormalization
                // reads to drop a member, so it has to be set, and RL_LIMIT_FROZEN keeps it
                // distinguishable from a real Q-limit pin (which the switch-back may undo; this must
                // not be, OLF's reactive-limits loop never sees a bus whose voltage control is off).
                size_t st = (size_t) s * T.nrl + re;
                if (mode[sr] != 1 || T.rlLimitType[st] != 0) continue;   // pinned already, or not live
                T.fzSavedMode[sc] = mode[sr];
                T.fzSavedTarget[sc] = target[sr];
                mode[sr] = 4;
                target[sr] = qBuf[sr] - T.fzLoadQ[c];
                T.rlLimitType[st] = RL_LIMIT_FROZEN;
                T.fzFrozen[sc] = 1;
                continue;
            }
            if (mode[sr] != 0 && mode[sr] != 2) continue;
            T.fzSavedMode[sc] = mode[sr];
            T.fzSavedTarget[sc] = target[sr];
            mode[sr] = 1;
            target[sr] = qBuf[sr] - T.fzLoadQ[c];            // generationTargetQ = q.eval(); row target = gen − load
            T.fzFrozen[sc] = 1;
        }
        // LfNetwork.fixTransformerVoltageControls: a just-enabled controller whose not-controlled side's
        // component holds no ENABLED generator voltage control (frozen and pinned buses count as off)
        // is disabled again. OLF cuts the CONTINGENCY's branches and the ENABLED controller branches;
        // the host knows the first and brackets the second with two masks (see FreezeTables), so a PV
        // bus in the MIN mask is one OLF sees too, and no PV bus in the MAX mask means OLF sees none.
        // Between the two the answer depends on which controllers THIS scenario enabled, which is not
        // something the host could precompute — that scenario goes to the CPU.
        if (T.hasFix) {
            for (int g = 0; g < T.nG; g++) {
                for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
                    size_t sm = (size_t) s * T.nM + m;
                    if (T.memState[sm] != 1) continue;
                    size_t base = ((size_t) s * T.nM + m) * T.fixWords;
                    bool pvMin = tvdAnyPv(T, sn, mode, T.fixMinMask + base);
                    if (pvMin) continue;                       // OLF keeps this control enabled
                    if (tvdAnyPv(T, sn, mode, T.fixMaxMask + base)) {
                        T.reject[s] = 1;                       // ambiguous without the enabled set
                        return;
                    }
                    T.memState[sm] = 0;                        // OLF disables it
                }
            }
        }
        for (int g = 0; g < T.nG; g++) {
            bool any = false;
            for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) any |= T.memState[(size_t) s * T.nM + m] == 1;
            if (any) tvdLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
        }
        T.scnStep[s] = 1;
        switched[s] = 1;
        return;
    }
    // CONTROL: roundR1ToExtremeTapPosition on every enabled controller
    bool outOfBound = false;
    for (int g = 0; g < T.nG; g++) {
        size_t sg = (size_t) s * T.nG + g;
        bool useInit = T.grpUseInit[g] != 0;
        bool touched = false;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;
            size_t sm = (size_t) s * T.nM + m;
            if (T.memState[sm] != 1) continue;
            double r1 = x[sn + T.memRhoRow[m]];
            double lo = useInit ? T.grpShMin[sg] : T.memMin[m];
            double hi = useInit ? T.grpShMax[sg] : T.memMax[m];
            if (!(r1 < lo || r1 > hi)) continue;
            double v = r1 > hi ? hi : lo;
            T.memRoundFrom[sm] = v;
            int tap;
            tvdRoundToTap(T, m, v, &tap);
            T.memTap[sm] = tap;
            T.memState[sm] = 2;
            touched = true;
        }
        if (touched) {
            tvdLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
            outOfBound = true;
        }
    }
    if (outOfBound) {
        switched[s] = 1;                                     // stays CONTROL; OLF: UNSTABLE
        return;
    }
    // none out of range: updateContinuousRatio (useInitialTapPosition remap) + roundVoltageRatios -> COMPLETE
    for (int g = 0; g < T.nG; g++) {
        size_t sg = (size_t) s * T.nG + g;
        bool useInit = T.grpUseInit[g] != 0;
        bool touched = false;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;
            size_t sm = (size_t) s * T.nM + m;
            if (T.memState[sm] != 1) continue;
            double v = x[sn + T.memRhoRow[m]];
            if (useInit) {
                double init = T.grpShInit[sg];
                v = v >= init
                        ? T.memInit[m] + (v - init) * (T.memMax[m] - T.memInit[m]) / (T.grpShMax[sg] - init)
                        : T.memInit[m] - (init - v) * (T.memInit[m] - T.memMin[m]) / (init - T.grpShMin[sg]);
            }
            T.memRoundFrom[sm] = v;
            int tap;
            tvdRoundToTap(T, m, v, &tap);
            T.memTap[sm] = tap;
            T.memState[sm] = 2;
            touched = true;
        }
        if (touched) tvdLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
    }
    // enableGeneratorVoltageControlsUnderMaxControlledNominalVoltage: the frozen controllers resume
    for (int c = 0; c < T.nFz; c++) {
        size_t sc = (size_t) s * T.nFz + c;
        if (!T.fzFrozen[sc]) continue;
        size_t sr = sn + T.fzVRow[c];
        mode[sr] = T.fzSavedMode[sc];
        target[sr] = T.fzSavedTarget[sc];
        if (T.fzRlEntry[c] >= 0) {                           // a remote member rejoins the DISTR_Q shares
            T.rlLimitType[(size_t) s * T.nrl + T.fzRlEntry[c]] = 0;
        }
        T.fzFrozen[sc] = 0;
    }
    T.scnStep[s] = 2;
    switched[s] = 1;                                         // OLF controlStep: UNSTABLE in any case (the COMPLETE solve)
}

static TvdTables tvdTables(BatchSolverContext* ctx) {
    TvdTables T;
    T.nG = ctx->nTvdGroups; T.nM = ctx->nTvd; T.nDelta = ctx->nTvdDelta; T.mode = ctx->tvdMode;
    T.grpPtr = ctx->dTvdGrpPtr.as<int>(); T.grpVRow = ctx->dTvdGrpVRow.as<int>();
    T.grpTargetV = ctx->dTvdGrpTargetV.as<double>(); T.grpHalfDb = ctx->dTvdGrpHalfDb.as<double>(); T.grpUseInit = ctx->dTvdGrpUseInit.as<int>();
    T.deltaPtr = ctx->dTvdDeltaPtr.as<int>(); T.deltaRow = ctx->dTvdDeltaRow.as<int>();
    T.deltaModeOn = ctx->dTvdDeltaModeOn.as<int>(); T.deltaModeOff = ctx->dTvdDeltaModeOff.as<int>();
    T.deltaTargetOn = ctx->dTvdDeltaTargetOn.as<double>(); T.deltaTargetOff = ctx->dTvdDeltaTargetOff.as<double>();
    T.memRhoRow = ctx->dTvdMemRhoRow.as<int>(); T.memElem = ctx->dTvdMemElem.as<int>();
    T.memBaseTap = ctx->dTvdMemBaseTap.as<int>(); T.memTapPtr = ctx->dTvdMemTapPtr.as<int>();
    T.memMin = ctx->dTvdMemMin.as<double>(); T.memMax = ctx->dTvdMemMax.as<double>(); T.memInit = ctx->dTvdMemInit.as<double>();
    T.tapData = ctx->dTvdTapData.as<double>();
    T.scnStep = ctx->dTvdScnStep.as<int>(); T.grpStep = ctx->dTvdGrpStep.as<int>();
    T.grpShMin = ctx->dTvdGrpShMin.as<double>(); T.grpShMax = ctx->dTvdGrpShMax.as<double>(); T.grpShInit = ctx->dTvdGrpShInit.as<double>();
    T.memState = ctx->dTvdMemState.as<int>(); T.memTap = ctx->dTvdMemTap.as<int>(); T.memRoundFrom = ctx->dTvdMemRoundFrom.as<double>();
    T.saved = ctx->dTvdSaved.as<double>(); T.reject = ctx->dTvdReject.as<char>();
    T.nFz = ctx->nTvdFz; T.hasFix = ctx->tvdFixTables ? 1 : 0; T.fixWords = ctx->tvdFixWords;
    T.fzVRow = ctx->dTvdFzVRow.as<int>(); T.fzLoadQ = ctx->dTvdFzLoadQ.as<double>();
    T.fzRlEntry = ctx->dTvdFzRlEntry.as<int>(); T.nrl = ctx->nrl; T.rlLimitType = ctx->dRlLimitType.as<char>();
    T.fixPvRow = ctx->dTvdFixPvRow.as<int>();
    T.fixMinMask = ctx->dTvdFixMinMask.as<int>(); T.fixMaxMask = ctx->dTvdFixMaxMask.as<int>();
    T.fzFrozen = ctx->dTvdFzFrozen.as<int>(); T.fzSavedMode = ctx->dTvdFzSavedMode.as<int>(); T.fzSavedTarget = ctx->dTvdFzSavedTarget.as<double>();
    return T;
}

void tvdStep(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const int& outer = drv.outer;
    std::vector<char>& loopRejected = drv.loopRejected;
    if (ctx->nTvd <= 0) return;
    tvdCheckBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(tvdTables(ctx), nS, n, ctx->nElements, ctx->dDisabled.as<char>(),
            ctx->dRlActive.as<char>(), ctx->dQbuf.as<double>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>(),
            ctx->dRlSwitched.as<char>());
    // Freezing a REMOTE controller can take the LAST live member of its group, and then the group
    // controls nothing: OLF's freeze disables the voltage control, so the CONTROLLED bus stops
    // carrying a V identity. That flip lives in its own kernel, which the reactive-limits phase runs
    // after its pins; the freeze happens two loops later and would otherwise leave the pilot holding
    // a target for a control that no longer exists until the next sweep reached RL again (measured
    // 0.6 kV). Same kernel, run again here — it restores the identity at COMPLETE on the same rule.
    if (ctx->hasDqQrGroups && ctx->st->nQr > 0 && ctx->nTvdFz > 0) {
        launchPilotVeqInner(ctx, drv);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<char> rej(nS);
    CUDA_CHECK(cudaMemcpy(rej.data(), ctx->dTvdReject.p, nS, cudaMemcpyDeviceToHost));
    for (int s2 = 0; s2 < nS; s2++) if (rej[s2]) { loopRejected[s2] = 1; traceReject(s2, "tvd-kernel-reject", outer, 0); }
    tvcRebuildOverrides(ctx);
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        size_t nsm = static_cast<size_t>(nS) * ctx->nTvd;
        std::vector<int> st(nsm), step(nS);
        CUDA_CHECK(cudaMemcpy(st.data(), ctx->dTvdMemState.p, nsm * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(step.data(), ctx->dTvdScnStep.p, nS * sizeof(int), cudaMemcpyDeviceToHost));
        int nCont = 0, nFixed = 0, nRej = 0;
        for (size_t k = 0; k < nsm; k++) { nCont += st[k] == 1; nFixed += st[k] == 2; }
        for (int s2 = 0; s2 < nS; s2++) nRej += rej[s2] ? 1 : 0;
        std::fprintf(stderr, "TVD_KERNEL mode=%d nS=%d nMembers=%d continuous=%d fixed=%d rejected=%d step0=%d nR1Ov=%d\n",
                     ctx->tvdMode, nS, ctx->nTvd, nCont, nFixed, nRej, step[0], ctx->nR1Ov);
    }
}

// Per-solve reset. Discrete TVC: every control at its entry state.
void tvdResetSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    size_t nsm = static_cast<size_t>(nS) * ctx->nTvd;
    size_t nsg = static_cast<size_t>(nS) * ctx->nTvdGroups;
    size_t nsd = static_cast<size_t>(nS) * std::max(ctx->nTvdDelta, 1);
    std::vector<int> tap(nsm);
    for (int s2 = 0; s2 < nS; s2++) for (int m = 0; m < ctx->nTvd; m++) tap[static_cast<size_t>(s2) * ctx->nTvd + m] = ctx->hTvdMemBaseTap[m];
    std::vector<double> nanv(std::max(nsm, nsd), std::nan(""));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(ctx->dTvdMemTap.p, tap.data(), nsm * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dTvdMemState.p, 0, nsm * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(ctx->dTvdMemRoundFrom.p, nanv.data(), nsm * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dTvdSaved.p, nanv.data(), nsd * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->dTvdScnStep.p, 0, nS * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dTvdGrpStep.p, 0, nsg * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx->dTvdGrpShMin.p, 0, nsg * sizeof(double)));
    CUDA_CHECK(cudaMemset(ctx->dTvdGrpShMax.p, 0, nsg * sizeof(double)));
    CUDA_CHECK(cudaMemset(ctx->dTvdGrpShInit.p, 0, nsg * sizeof(double)));
    CUDA_CHECK(cudaMemset(ctx->dTvdReject.p, 0, nS * sizeof(char)));
    if (ctx->nTvdFz > 0) CUDA_CHECK(cudaMemset(ctx->dTvdFzFrozen.p, 0, static_cast<size_t>(nS) * ctx->nTvdFz * sizeof(int)));
    tvdEntryBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(tvdTables(ctx), nS, n, ctx->nElements, ctx->tvdEntryOn,
            ctx->dDisabled.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

}  // namespace olfgpu
