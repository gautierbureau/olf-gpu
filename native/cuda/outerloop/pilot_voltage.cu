// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See pilot_voltage.cuh.
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
#include "../common/codes/gpu_row_mode.cuh"
#include "../equations/scenario_overrides.cuh"
#include "../solver/driver_steps.cuh"
#include "../fixedmatrix/reactive_relocation.cuh"
#include "../fixedmatrix/cross_identity.cuh"
#include "pilot_voltage.cuh"


namespace olfgpu {

// PILOT-SELF gate, per (scenario, group): 1 = skip this group's relocated raw q because its pilot is one of
// its own controllers and is NOT switched to PQ this scenario. While the pilot controls, its q belongs in the
// group's DISTR_Q rows WEIGHTED (it has no Q balance of its own), so emitting it raw at the carrier row
// would double-count it. Once a reactive limit switches it PV -> PQ, OLF activates its BUS_TARGET_Q and the carrier row
// is exactly where that balance goes — the relocation the CPU does statically, here per scenario.
__global__ void pilotSelfGateBatch(int nReactiveRelocationGroups, int nScenarios, int nReactiveLimitEntries, const int* pilotEntryOfPilotSelfGroup,
                                   const char* limitType, const char* reactiveLimitIslanded, char* pilotSelfCarrierSkip) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nReactiveRelocationGroups * nScenarios) return;
    int g = tid % nReactiveRelocationGroups;
    int s = tid / nReactiveRelocationGroups;
    int e = pilotEntryOfPilotSelfGroup[g];
    char skip;
    if (e == PILOT_SELF_NO_REACTIVE_LIMIT_ENTRY) {
        skip = 1;                                            // pilot-self, never switches to PQ ⇒ always controlling
    } else if (e >= 0 && e < nReactiveLimitEntries) {
        // An ISLANDED pilot (its section lost with the contingency) has no balance anywhere — OLF removed
        // the bus. Its limit type still reads its BASE switch to PQ (initReactiveLimitQLimitTypeBatch seeds an enrolled
        // pilot's kind from the base), and "switched to PQ" here means "its balance rides the carrier": the carrier's own
        // DISTR_Q was suppressed for a balance that does not exist and the row went empty. Measured on
        // rte6515 BUS-6194_BBS_1 (pilot VL-6194_0 switched to PQ at MAX_Q in the base, then islanded: FAILED where OLF
        // converges with 60 violations) and BUS-6003_BBS_1 (0.473 MW of distributed active power).
        // Still PV or islanded ⇒ skip; switched to PQ and alive ⇒ its balance rides the carrier.
        bool islanded = reactiveLimitIslanded != nullptr && reactiveLimitIslanded[(size_t) s * nReactiveLimitEntries + e];
        skip = (islanded || limitType[(size_t) s * nReactiveLimitEntries + e] == GpuQLimitType::NONE) ? 1 : 0;
    } else {
        skip = 0;                                            // not a pilot-self group
    }
    pilotSelfCarrierSkip[(size_t) s * nReactiveRelocationGroups + g] = skip;
}

// plan-B4: capture, per (scenario, relocation group), the pilot's Q target from the BASE carrier's
// row — the base target upload and the per-scenario injection overrides both address the pilot's
// reactive balance at that row (structure-time qRow). updateDistrQTargetsBatch hands the value to
// whichever member row currently carries the relocated pilot Q. Runs BEFORE the RL mode seeding so
// a warm-continuation PV -> PQ switch cannot overwrite the value first.
__global__ void capturePilotQBatch(int nReactiveRelocationGroups, int nScenarios, int n, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                                   const int* baseMode, const double* targetBase, double* pilotQ,
                                   const int* reactiveRelocationBaseTargetRow) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nReactiveRelocationGroups * nScenarios) return;
    int g = tid % nReactiveRelocationGroups;
    int s = tid / nReactiveRelocationGroups;
    // a DEPENDENT pilot's target sits on its own base reactive row, not on a carrier
    int row = reactiveRelocationBaseTargetRow != nullptr && reactiveRelocationBaseTargetRow[g] >= 0 ? reactiveRelocationBaseTargetRow[g]
            : reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g] + chooseCarrierRow(g, reactiveRelocationGroupStart, reactiveRelocationCandidateRows, baseMode)];
    pilotQ[(size_t) s * nReactiveRelocationGroups + g] = targetBase[row];          // the BASE carrier's BASE target (n, untiled)
}

// A scenario's outage that moves the PILOT's own reactive balance (a reactive load or a fixed-Q generator lost at
// the pilot, GpuBatchInjectionOutages): the captured value moves with it, and so does the target of the row that
// carries the balance at the first fill - whichever the scenario's modes pick, by the fill's own rule - because
// updateDistrQTargetsBatch stamps the captured value only inside the outer-loop passes. An identity there (the
// pilot's own row still holding its voltage target) carries no balance and keeps its target. One entry per
// (scenario, group): the host merges them.
__global__ void applyPilotQDeltasBatch(int nScenarios, SlotEntries deltas, int nReactiveRelocationGroups, int n,
                                       const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const int* mode,
                                       const char* distrQSubgroup, int nDistrQMembers, const int* reactiveRelocationDistrQMemberStart,
                                       double* pilotQ, double* target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nScenarios * deltas.capacity) return;
    int s = tid / deltas.capacity;
    int k = tid % deltas.capacity;
    if (k >= deltas.countOf(s)) return;
    int g = deltas.intOf(s, k);
    double delta = deltas.doubleOf(s, k);
    pilotQ[(size_t) s * nReactiveRelocationGroups + g] += delta;
    int c = chooseCarrierRowInSubgroup(g, reactiveRelocationGroupStart, reactiveRelocationCandidateRows, mode + (size_t) s * n,
            distrQSubgroup != nullptr ? distrQSubgroup + (size_t) s * nDistrQMembers : nullptr, reactiveRelocationDistrQMemberStart);
    int row = reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g] + c];
    int m = mode[(size_t) s * n + row];
    if (!holdsFixedVariable(m)) {
        target[(size_t) s * n + row] += delta;
    }
}

// DEVICE no-enabled-controller corner (plan-B4 follow-up): when the LAST live member of a remote gen-VC
// relocation group switches to PQ (the mode-scan finds no candidate with mode not in {0, 4} and falls back
// to candidate 0 = the pilot's own row), OLF deactivates the pilot's V equation — control lost.
// Flip the pilot row to a POWER row carrying the group's captured pilot-Q target (the scatter
// already lands the pilot's balance there via the same fallback); when a member switches back to PV,
// restore the V identity and its saved target. Flag-guarded per (scenario, group): only rows
// THIS kernel flipped are ever flipped back — with one exception. A group that lost control in the
// BASE (every member switched to PQ there) has its pilot row pushed as a power row with no flag, and when a
// member switches back to PV in a contingency OLF gives the pilot its V equation back IN THAT SAME PASS
// (updateRemoteVoltageControlEquations runs on every switch). Nothing did that here: the pilot's own
// local entry — an empty reactive range with a guessed limit type — restored the row on ITS OWN
// voltage test, which is the wrong trigger and fires a pass late or never. Measured on
// network.xiidm/LNER: the three members of IZUM_0's group switched back to PV on a 1.2e-6 pu knife-edge, the
// device solved one pass with three sharing rows and no voltage identity, switched two of the three to PQ again
// where OLF switched all three, and the distributed slack ended 0.151 MW apart. The restore below
// takes the group's V target from the pilot's local entry (reactiveRelocationPilotReactiveLimitEntry), per-scenario overrides
// included, and the local entry itself no longer decides anything. A flip (either direction) marks
// the scenario switched: the row swap needs a re-solve even when it rides a no-switch
// warm-continuation pass.
__global__ void updatePilotTargetVEquationBatch(int nReactiveRelocationGroups, int nScenarios, int n,
                                    const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, const char* reactiveRelocationIsGeneratorVoltageControl,
                                    const double* pilotQ, int* mode, double* target,
                                    char* pilotTargetVDeactivated, double* savedVTarget,
                                    const char* outerLoopCheckActive, char* switched,
                                    SlotEntries frozenRows,
                                    const int* pilotSelfReactiveLimitEntry,
                                    const char* limitType, int nReactiveLimitEntries, const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ,
                                    const double* reactiveLimitLoadQ, const double* reactiveLimitInitialTargetQ, int perScenarioReactiveLimits,
                                    const double* reactiveLimitTargetV,
                                    const int* reactiveRelocationPilotReactiveLimitEntry,
                                    SlotEntries reactiveLimitTargetVOverrides,
                                    const double* secondaryVoltageControlReactiveLimitTargetV,
                                    int nDistrQMembers, const int* distrQMemberRelocationGroup, const int* distrQMemberReactiveLimitEntry,
                                    const char* reactiveLimitIslanded,
                                const char* distrQSubgroup, const int* reactiveRelocationDistrQMemberStart, double tol,
                                const char* reactiveLimitGeneratorsLost, const int* reactiveRelocationDependentBaseGroup, const int* relocatedChildOwnRow,
                                const double* reactiveLimitSeedTargetQ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nReactiveRelocationGroups * nScenarios) return;
    int g = tid % nReactiveRelocationGroups;
    int s = tid / nReactiveRelocationGroups;
    if (!reactiveRelocationIsGeneratorVoltageControl[g]) return;
    // The scenarios being checked only: every one at the seed (it marks them all), and at a ReactiveLimits or discrete
    // TransformerVoltageControl check the scenarios standing at that loop. A scenario's layout follows its own facts at its
    // own checks; laid out at another scenario's check, it would depend on its chunk mates, and its own later check would
    // find no transition to flag. The seed also books a group lost in the base (below): without it the switch back to PV of
    // such a pilot at its first RL pass found pilotTargetVDeactivated = 0, restored nothing, and the pilot's row stayed a
    // power row with its raw q gated off (back on PV ⇒ pilotSelfCarrierSkip): an empty row and a bad solve (rte6515
    // [5000,6000] BUS-6003_BBS_2, slot 797, max-iter-bad-solve at RL#2).
    if (!outerLoopCheckActive[s]) return;
    int pilotRow = reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g]];
    // A DEPENDENT pilot (reactiveRelocationDependentBaseGroup): its identity sits on its own V row once its parent coupler is dead (the dummy
    // row an identity) and on the dummy row as a cross identity while the coupler lives; the group is live only
    // while that identity is there, or while this kernel holds it flipped.
    const bool dependent = reactiveRelocationDependentBaseGroup != nullptr && reactiveRelocationDependentBaseGroup[g] >= 0;
    if (dependent) {
        if (relocatedChildOwnRow != nullptr && relocatedChildOwnRow[pilotRow] >= 0 && mode[(size_t) s * n + pilotRow] == GpuRowMode::FIXED_VARIABLE) {
            pilotRow = relocatedChildOwnRow[pilotRow];
        }
        int pm = mode[(size_t) s * n + pilotRow];
        if (pm != 0 && pm != 6 && !pilotTargetVDeactivated[(size_t) s * nReactiveRelocationGroups + g]) return;
    }
    // PILOT-SELF group (the pilot is merged controller #0, so its OWN row is also candidate 1): "control
    // lost" is OLF's "no enabled controller" — every OTHER member switched to PQ/off AND the pilot itself switched to PQ.
    // The candidate scan must skip the pilot's row: once flipped to a power row it reads as a live member
    // and the next pass restores it — the kernel toggling its own verdict (rte6515 busbar base, groups 17
    // and 19: mode 1/0 every pass to the outer-loop budget). And the flipped row hosts the pilot's
    // BUS_TARGET_Q, whose target is its Q LIMIT (the target Q the RL switch froze), not the group's captured pilot-Q
    // target (0 here — a PV -> PQ switch with the wrong target moved V past the group target and switched it back to PV).
    int pe = pilotSelfReactiveLimitEntry != nullptr ? pilotSelfReactiveLimitEntry[g] : -1;
    char plt = pe >= 0 && limitType != nullptr ? limitType[(size_t) s * nReactiveLimitEntries + pe] : 0;
    double flipTarget = pilotQ[(size_t) s * nReactiveRelocationGroups + g];
    bool continuedFlip = false;
    if (pe >= 0 && plt != GpuQLimitType::NONE) {
        size_t li = perScenarioReactiveLimits ? (size_t) s * nReactiveLimitEntries + pe : (size_t) pe;
        double lim = (plt == GpuQLimitType::MIN_REALISTIC_V || plt == GpuQLimitType::MAX_REALISTIC_V) && reactiveLimitInitialTargetQ != nullptr ? reactiveLimitInitialTargetQ[pe]
                   : isMinLimit(plt) ? reactiveLimitMinQ[li] : reactiveLimitMaxQ[li];
        // A pilot whose bus LOST EVERY GENERATOR this scenario (seeded GpuQLimitType::FROZEN through setBatchReactiveLimitGeneratorsLost)
        // is not at any reactive limit: OLF's BUS_TARGET_Q there has no generator term at all, only the bus's
        // load. Falling through to reactiveLimitMaxQ asked the bus for its lost generator's MAXIMUM reactive output:
        // rte6515 GEN-1251 with every busbar section in the list (VL-1251_0, the pilot of a two-section group
        // whose other member is switched to PQ), target 1.0 pu against OLF's 0.0, the bus driven from 0.572 to 0.94 pu
        // for the first passes, 4.18 MW of slack read where OLF reads -0.33, and -0.384 MW distributed against 0.
        if (plt == GpuQLimitType::FROZEN && reactiveLimitGeneratorsLost != nullptr && reactiveLimitGeneratorsLost[(size_t) s * nReactiveLimitEntries + pe]) {
            lim = 0.0;
        }
        flipTarget = lim - reactiveLimitLoadQ[li];
        // A pilot whose group the PREVIOUS solve had already flipped (a strategy pair continuing its contingency):
        // OLF restores that solve's frozen target Q and re-freezes it only past the reactive power mismatch. The
        // seed launch reads the base limits, which a curve moved since (rte6515 BUS-6066_BBS_1, VL-6231_0: flipped
        // at 0.576013 against the parent's 0.574331, then re-frozen at the pair's limit - 3.8e-5 kV on every
        // strategy, even one changing nothing).
        if (reactiveLimitSeedTargetQ != nullptr && plt >= 1 && plt <= 4 && !isnan(reactiveLimitSeedTargetQ[(size_t) s * nReactiveLimitEntries + pe])) {
            flipTarget = reactiveLimitSeedTargetQ[(size_t) s * nReactiveLimitEntries + pe];
            continuedFlip = true;
        }
    }
    // An ISLANDED pilot has no equation at all on the CPU: OLF removes the bus. Its row here is
    // frozen to a flat voltage identity, and every member row is frozen to identity too, so the
    // mode scan below finds no carrier and reads that as "control lost" — flipping the frozen
    // identity into a power row. The target override then re-stamps the frozen VOLTAGE value over
    // that power row, leaving a residual of exactly 1.0 that no Newton step can ever clear, so
    // every later re-solve buys iterations OLF does not run. Measured on LINE-3938-646: the whole
    // network moved 8.8e-08 kV off the CPU answer with identical solve counts.
    if (rowIsFrozen(frozenRows, s, pilotRow)) return;
    size_t pi = (size_t) s * n + pilotRow;
    size_t fi = (size_t) s * nReactiveRelocationGroups + g;
    // OLF's own test for "this group has no controller": its ENABLED controllers
    // (VoltageControl.isControllerEnabled -> isGeneratorVoltageControlEnabled). On the device that fact is the
    // members' REACTIVE-LIMIT state, which is per (scenario, entry) and current from the seeding onwards —
    // unlike the row modes this kernel used to infer it from, which at seed time are still ambiguous (a member
    // row reads mode 0 both for a frozen island and for an identity that means the member controls). With the
    // fact in hand the layout can follow it from the FIRST fill, which is where OLF has it: a pilot whose only
    // sibling this contingency islands, and which the base itself switched to PQ, has no BUS_TARGET_V at all in OLF,
    // while the device used to hold that identity until its reactive-limits phase (rte6515 BUS-6112_BBS_2: the
    // pilot VL-6112_0 sat at v = 1.055 through the whole slack phase, VL-6055_0's q read 1.474 against OLF's
    // 1.902, and the run ended 3.7e-3 MW apart in distributed active power).
    bool entryKnown = false;
    bool anyEnabled = false;
    if (pe >= 0 && distrQMemberRelocationGroup != nullptr && distrQMemberReactiveLimitEntry != nullptr && limitType != nullptr && nReactiveLimitEntries > 0) {
        for (int m = 0; m < nDistrQMembers; m++) {
            if (distrQMemberRelocationGroup[m] != g) continue;
            if (distrQSubgroup != nullptr && distrQSubgroup[(size_t) s * nDistrQMembers + m] != 0) continue;   // another subgroup's controller (#71)
            int re = distrQMemberReactiveLimitEntry[m];
            if (re < 0 || re >= nReactiveLimitEntries) continue;
            if (reactiveLimitIslanded != nullptr && reactiveLimitIslanded[(size_t) s * nReactiveLimitEntries + re]) continue;   // OLF disables it
            entryKnown = true;
            if (limitType[(size_t) s * nReactiveLimitEntries + re] == GpuQLimitType::NONE) {  // still controlling
                anyEnabled = true;
                break;
            }
        }
    }
    // Only with at least one member entry to read: a group whose members have none (reactive limits disabled for
    // the run, an empty q range) cannot be judged this way, and its layout stays what the extraction gave it.
    const bool factKnown = entryKnown;
    bool lost;
    if (factKnown) {
        lost = !anyEnabled;
    } else if (pe >= 0) {
        lost = plt != GpuQLimitType::NONE;  // a pilot still PV controls its own bus
        int p0 = reactiveRelocationGroupStart[g];
        int count = reactiveRelocationGroupStart[g + 1] - p0;
        for (int c = 1; lost && c < count; c++) {
            int r = reactiveRelocationCandidateRows[p0 + c];
            if (r == pilotRow) continue;
            int m = mode[(size_t) s * n + r];
            if (m != GpuRowMode::FIXED_VARIABLE && m != GpuRowMode::BUS_TARGET_Q) lost = false;
        }
    } else {
        lost = chooseCarrierRowInSubgroup(g, reactiveRelocationGroupStart, reactiveRelocationCandidateRows, mode + (size_t) s * n,
                distrQSubgroup != nullptr ? distrQSubgroup + (size_t) s * nDistrQMembers : nullptr, reactiveRelocationDistrQMemberStart) == 0;
    }
    if (lost && !pilotTargetVDeactivated[fi] && mode[pi] == GpuRowMode::FIXED_VARIABLE) {
        savedVTarget[fi] = target[pi];
        mode[pi] = GpuRowMode::EQUATION;
        target[pi] = flipTarget;
        pilotTargetVDeactivated[fi] = 1;
        switched[s] = 1;
    } else if (!lost && pilotTargetVDeactivated[fi]) {
        mode[pi] = fixedVariableMode(dependent && relocatedChildOwnRow != nullptr && relocatedChildOwnRow[pilotRow] >= 0);  // a cross identity stays cross
        target[pi] = savedVTarget[fi];
        pilotTargetVDeactivated[fi] = 0;
        switched[s] = 1;
    } else if (lost && pilotTargetVDeactivated[fi] && pe >= 0 && fabs(target[pi] - flipTarget) > tol) {
        // The PQ pilot's limit moved (curve limits). OLF re-freezes a bus switched to PQ at its moved limit only
        // past maxReactivePowerMismatch (checkPqBusWithQLimitType: |maxQ - q| > tol), the same gate the member
        // path above this kernel applies; a 1e-12 gate here re-stamped a pilot OLF leaves at its switch-time
        // limit. Measured on rte6515 BUS-6029_BBS_1 (#73): VL-6231_0's curve moved 2.23e-5 pu with the second
        // slack distribution, under the 1e-4 pu threshold - OLF kept 0.570780403, the device stamped
        // 0.570758104 and the network sat 5.68e-5 kV off.
        target[pi] = flipTarget;
        switched[s] = 1;
    } else if (lost && !pilotTargetVDeactivated[fi] && pe >= 0 && mode[pi] == GpuRowMode::EQUATION && reactiveLimitTargetV != nullptr) {
        // Lost already in the BASE: the activation laid the pilot's row out as its own Q balance (OLF's
        // BUS_TARGET_V inactive, BUS_TARGET_Q active). Book it as flipped so a member switching back later
        // restores the V identity at the group's target — without this the restore never fired and the
        // pilot stayed a power row beside a member that had resumed control.
        pilotTargetVDeactivated[fi] = 1;
        savedVTarget[fi] = reactiveLimitTargetV[pe];
        // ...and if this scenario's outage took the pilot bus's every generator, its balance target loses the
        // generator's frozen output NOW, from the first fill: OLF's BUS_TARGET_Q there has no generator term.
        // The re-stamp branch above only runs inside the reactive-limits phase, so the slack phase solved with
        // the lost generator's frozen target Q still injected - rte6515 GEN-6003 with every busbar section in the list:
        // VL-6003_0 kept 0.7477 pu against OLF's 0.0 through the first passes, and the first slack read was
        // 0.049 MW off (238.816 MW distributed against OLF's 238.865).
        if (plt == GpuQLimitType::FROZEN && reactiveLimitGeneratorsLost != nullptr && reactiveLimitGeneratorsLost[(size_t) s * nReactiveLimitEntries + pe]) {
            target[pi] = flipTarget;
        }
        // ...and a group the previous solve had flipped at another limit (a strategy pair continuing its
        // contingency) holds THAT target, not the base one the activation laid out (rte6515 BUS-6230_BBS: VL-6229_0
        // switched to PQ at MAX_Q 0.0 in the base, at MIN_Q -0.06 after the contingency; the pair kept 0.0, 8.8e-5 kV)
        if (continuedFlip) {
            target[pi] = flipTarget;
        }
    } else if (!lost && pe < 0 && mode[pi] == GpuRowMode::EQUATION && reactiveRelocationPilotReactiveLimitEntry != nullptr && reactiveRelocationPilotReactiveLimitEntry[g] >= 0) {
        // The same base-lost case for a PURE-TRANSIT pilot (not one of its own controllers, so no
        // pilotSelfReactiveLimitEntry): the base carried no saved V target - the pilot's own local reactive-limit entry
        // holds the group's target (per-scenario overrides included). A member just switched back to PV, and
        // OLF gives the pilot its V equation back IN THAT PASS (network.xiidm/LNER: IZUM_0's three
        // members switched back to PV on a 7.85e-07 pu margin; restoring a pass late cost 0.151 MW of slack).
        int le = reactiveRelocationPilotReactiveLimitEntry[g];
        mode[pi] = GpuRowMode::FIXED_VARIABLE;
        target[pi] = reactiveLimitTargetVOf(reactiveLimitTargetVOverrides, secondaryVoltageControlReactiveLimitTargetV, nReactiveLimitEntries, s, le, reactiveLimitTargetV[le]);
        switched[s] = 1;
    }
}

void launchPilotTargetVEquationUpdate(BatchSolverContext* ctx, DriverPass& drv, bool staticLimits, bool useSeedTargetQ) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const bool& usePerScenarioReactiveLimits = drv.usePerScenarioReactiveLimits;
    const double& REACTIVE_LIMIT_TOL = drv.reactiveLimitTol;
    const bool useDyn = usePerScenarioReactiveLimits && !staticLimits;
    if (ctx->hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0) {
        updatePilotTargetVEquationBatch<<<grid(ctx->structure->nReactiveRelocationGroups * nS), BLOCK, 0, ctx->stream>>>(
                    ctx->structure->nReactiveRelocationGroups, nS, n,
                    ctx->structure->dReactiveRelocationGroupStart.as<int>(), ctx->structure->dReactiveRelocationCandidateRows.as<int>(), ctx->dReactiveRelocationIsGeneratorVoltageControl.as<char>(),
                    ctx->dReactiveRelocationPilotTargetQ.as<double>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                    ctx->dReactiveRelocationPilotTargetVDeactivated.as<char>(), ctx->dReactiveRelocationSavedPilotTargetV.as<double>(),
                    ctx->dOuterLoopCheckActive.as<char>(), ctx->dOuterLoopUnstable.as<char>(),
                    ctx->frozenRows.view(),
                    pilotSelfReactiveLimitEntryOrNull(ctx),
                ctx->dReactiveLimitQLimitType.p ? ctx->dReactiveLimitQLimitType.as<char>() : nullptr, ctx->nReactiveLimitEntries,
                useDyn ? ctx->dReactiveLimitMinQPerScenario.as<double>() : ctx->dReactiveLimitMinQ.as<double>(),
                useDyn ? ctx->dReactiveLimitMaxQPerScenario.as<double>() : ctx->dReactiveLimitMaxQ.as<double>(),
                useDyn ? ctx->dReactiveLimitLoadQPerScenario.as<double>() : ctx->dReactiveLimitLoadQ.as<double>(),
                ctx->reactiveLimitsRobustMode ? ctx->dReactiveLimitInitialTargetQ.as<double>() : nullptr, useDyn ? 1 : 0,
                ctx->dReactiveLimitTargetV.p ? ctx->dReactiveLimitTargetV.as<double>() : nullptr,
                ctx->dReactiveRelocationPilotReactiveLimitEntry.p ? ctx->dReactiveRelocationPilotReactiveLimitEntry.as<int>() : nullptr,
                ctx->reactiveLimitTargetVOverrides.view(),
                ctx->nSecondaryVoltageControlControllerBuses > 0 ? ctx->dSecondaryVoltageControlReactiveLimitTargetV.as<double>() : nullptr,
                ctx->nDistrQMembers, ctx->dDistrQMemberRelocationGroup.p ? ctx->dDistrQMemberRelocationGroup.as<int>() : nullptr,
                ctx->dDistrQMemberReactiveLimitEntry.p ? ctx->dDistrQMemberReactiveLimitEntry.as<int>() : nullptr,
                ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr,
                ctx->hasDistrQSubgroups ? ctx->dDistrQSubgroup.as<char>() : nullptr, ctx->hasDistrQSubgroups ? ctx->dReactiveRelocationDistrQMemberStart.as<int>() : nullptr,
                REACTIVE_LIMIT_TOL, ctx->nReactiveLimitGeneratorsLost > 0 ? ctx->dReactiveLimitGeneratorsLost.as<char>() : nullptr,
                ctx->structure->dReactiveRelocationDependentBaseGroup.p ? ctx->structure->dReactiveRelocationDependentBaseGroup.as<int>() : nullptr,
                ctx->structure->dRelocatedChildOwnRow.p ? ctx->structure->dRelocatedChildOwnRow.as<int>() : nullptr,
                useSeedTargetQ && ctx->hasReactiveLimitSeedTargetQ ? ctx->dReactiveLimitSeedTargetQ.as<double>() : nullptr);
    }
}

// OLF keeps a re-merged control's BUS_TARGET_V on its elected main while ANY merged controller is enabled, and the FIRST
// enabled one (merged order: the main's own controllers, then each dependent control's) carries no DISTR_Q
// (AcEquationSystemCreator.updateRemoteVoltageControlEquations). On the device the main's own row holds the voltage
// target while the main's controller is enabled, and that controller's PV -> PQ switch turns the row into its own reactive
// balance. The target then belongs on the first enabled controller's sharing row, as an identity on that controller's
// own voltage, which the cluster's couplers tie to the main's (a cross identity when the row is its relocated DUMMY_Q).
// Nothing used to move it, so a main switched to PQ left the cluster with no voltage equation: on the double-star
// fixture under nested dissection, g4 switched back to PV on a zero voltage margin in the pass g3 switched to PQ, and the
// sections floated to 399.78 kV against OLF's 400.00; under AMD, a main switching to PQ beside a controller with headroom
// made the solve diverge (status -2, CPU replay). The holder is recomputed from the reactive-limit states every pass, so
// it always is OLF's first enabled controller; with no enabled controller no row holds the target, as OLF deactivates
// BUS_TARGET_V. While a controller holds it, the main's switch-back reads the voltage that identity holds
// (reactiveLimitReMergedVoltageRow), as OLF reads the main's own, identity-held voltage: the main's is then only tied to
// it through the couplers, and a last-bit excess on a zero margin would switch it back to PV. One thread per re-merged
// control.
__global__ void updateReMergedVoltageTargetRowBatch(int nScenarios, int n, int nReactiveLimitEntries,
                                                    SlotEntries reMergedVoltageTargets, SlotEntries reMergedVoltageTargetControllers,
                                                    const int* reactiveLimitVoltageRow, const int* reactiveLimitReactivePowerRow, const int* relocatedChildOwnRow,
                                                    const char* limitType, const char* reactiveLimitIslanded, SlotEntries frozenRows,
                                                    int* reMergedVoltageTargetHolder, char* reactiveLimitHoldsReMergedVoltageTarget,
                                                    int* reactiveLimitReMergedVoltageRow, int* mode, double* target,
                                                    const char* outerLoopCheckActive, char* switched) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nScenarios * reMergedVoltageTargets.capacity) return;
    int s = tid / reMergedVoltageTargets.capacity;
    int control = tid % reMergedVoltageTargets.capacity;
    if (control >= reMergedVoltageTargets.countOf(s) || !outerLoopCheckActive[s]) return;
    size_t e = reMergedVoltageTargets.at(s, control);
    const int mainEntry = reMergedVoltageTargets.intOf(s, control, 0);
    const int firstController = reMergedVoltageTargets.intOf(s, control, 1);
    const int nControllers = reMergedVoltageTargets.intOf(s, control, 2);
    auto enabled = [&](int entry) {
        size_t k = (size_t) s * nReactiveLimitEntries + entry;
        return limitType[k] == GpuQLimitType::NONE && !(reactiveLimitIslanded != nullptr && reactiveLimitIslanded[k]);
    };
    // the row an entry's reactive equation lives on this scenario: its relocated DUMMY_Q row, or its own voltage row once
    // the contingency opened that coupler (the dummy row a DUMMY_TARGET identity then) - the reactive-limit kernel's rule
    auto sharingRow = [&](int entry) {
        int r = reactiveLimitReactivePowerRow != nullptr ? reactiveLimitReactivePowerRow[entry] : -1;
        if (r >= 0 && r != reactiveLimitVoltageRow[entry] && mode[(size_t) s * n + r] != GpuRowMode::FIXED_VARIABLE) {
            return r;
        }
        return reactiveLimitVoltageRow[entry];
    };
    auto controller = [&](int c) { return reMergedVoltageTargetControllers.intOf(s, firstController + c); };
    const int held = reMergedVoltageTargetHolder[e];
    int wanted = -1;
    if (!enabled(mainEntry)) {
        for (int c = 0; c < nControllers && wanted < 0; c++) {
            int entry = controller(c);
            int r = sharingRow(entry);
            int m = mode[(size_t) s * n + r];
            // an identity on the controller's own voltage: its own row, or the relocated row whose cross slot is that voltage
            bool hostable = r == reactiveLimitVoltageRow[entry]
                    || (relocatedChildOwnRow != nullptr && relocatedChildOwnRow[r] == reactiveLimitVoltageRow[entry]);
            if (enabled(entry) && hostable && !rowIsFrozen(frozenRows, s, r)
                    && (m == GpuRowMode::EQUATION || (held == c && holdsFixedVariable(m)))) {
                wanted = c;
            }
        }
    }
    if (held == wanted) return;
    if (held >= 0) {                                         // back to its sharing equation (its DISTR_Q target follows)
        int entry = controller(held);
        size_t r = (size_t) s * n + sharingRow(entry);
        if (holdsFixedVariable(mode[r])) {                   // ...unless its own switch to PQ made the row its balance already
            mode[r] = GpuRowMode::EQUATION;
        }
        reactiveLimitHoldsReMergedVoltageTarget[(size_t) s * nReactiveLimitEntries + entry] = 0;
    }
    size_t mainKey = (size_t) s * nReactiveLimitEntries + mainEntry;
    if (wanted >= 0) {
        int entry = controller(wanted);
        int r = sharingRow(entry);
        mode[(size_t) s * n + r] = fixedVariableMode(r != reactiveLimitVoltageRow[entry]);
        target[(size_t) s * n + r] = reMergedVoltageTargets.doubleOf(s, control);
        reactiveLimitHoldsReMergedVoltageTarget[(size_t) s * nReactiveLimitEntries + entry] = 1;
        reactiveLimitReMergedVoltageRow[mainKey] = reactiveLimitVoltageRow[entry];
    } else {
        reactiveLimitReMergedVoltageRow[mainKey] = -1;
    }
    reMergedVoltageTargetHolder[e] = wanted;
    switched[s] = 1;
}

void launchReMergedVoltageTargetRowUpdate(BatchSolverContext* ctx, DriverPass& drv, bool solveStart) {
    if (ctx->reMergedVoltageTargets.entries() == 0) {
        return;
    }
    const int& n = drv.n;
    const int& nS = drv.nS;
    SlotEntries reMergedVoltageTargets = ctx->reMergedVoltageTargets.view();
    if (solveStart) {
        size_t scenarioEntries = static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries;
        ctx->reMergedVoltageTargets.resetState({}, ctx->stream);
        CUDA_CHECK(cudaMemsetAsync(ctx->dReactiveLimitHoldsReMergedVoltageTarget.p, 0, scenarioEntries, ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->dReactiveLimitReMergedVoltageRow.p, 0xFF, scenarioEntries * sizeof(int), ctx->stream));
    }
    updateReMergedVoltageTargetRowBatch<<<grid(nS * reMergedVoltageTargets.capacity), BLOCK, 0, ctx->stream>>>(nS, n,
            ctx->nReactiveLimitEntries, reMergedVoltageTargets, ctx->reMergedVoltageTargetControllers.view(),
            ctx->dReactiveLimitVoltageRow.as<int>(), ctx->dReactiveLimitReactivePowerRow.p ? ctx->dReactiveLimitReactivePowerRow.as<int>() : nullptr,
            ctx->structure->dRelocatedChildOwnRow.p ? ctx->structure->dRelocatedChildOwnRow.as<int>() : nullptr,
            ctx->dReactiveLimitQLimitType.as<char>(), ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr,
            ctx->frozenRows.view(),
            ctx->reMergedVoltageTargets.state<int>(0), ctx->dReactiveLimitHoldsReMergedVoltageTarget.as<char>(),
            ctx->dReactiveLimitReMergedVoltageRow.as<int>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
            ctx->dOuterLoopCheckActive.as<char>(), ctx->dOuterLoopUnstable.as<char>());
}

}  // namespace olfgpu
