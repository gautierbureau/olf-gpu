// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See distribution_fill.cuh.
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
#include "../common/codes/gpu_two_sides.cuh"
#include "generated/closed_branch.cuh"
#include "generated/closed_branch_scatter.h"
#include "generated/open_branch.cuh"
#include "generated/open_branch_scatter.h"
#include "generated/shunt.cuh"
#include "generated/shunt_scatter.h"
#include "generated/hvdc_ac_emulation.cuh"
#include "generated/hvdc_ac_emulation_scatter.h"
#include "generated/load_model.cuh"
#include "scatter.cuh"
#include "../fixedmatrix/reactive_relocation.cuh"
#include "scenario_overrides.cuh"
#include "distribution_fill.cuh"
#include "../common/trace.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

int gpuDistrQTraceRow() {
    // OLF_GPU_DQ_TRACE=<deviceRow>: every DISTR_Q residual contribution landing in that row.
    static const char* e = traceValue(Trace::DQ_TRACE);
    static int v = e != nullptr ? std::atoi(e) : -1;
    return v;
}

// Batched DISTR_Q fill: one thread per (contribution, scenario). Each contribution is a closed
// branch's side-s reactive flow (value + the 5 derivatives), scaled by the controller weight; x/vals/F are
// strided per scenario, and the kernel DROPS a contribution whose underlying closed (kind 0) / open (kind 2)
// branch is outaged in this scenario — matching the CPU, which removes that branch's q from the controller's
// reactive sum (the controller weight qPercent is unchanged, so only the term vanishes). kinds 1 (shunt) and
// 3 (zero-imp dummy) cannot be a single-branch contingency in a qualified batched scenario, so no mask check.
__global__ void fillDistrQBatch(int nDistrQ, int nScenarios, int n, int nnz, int nElements,
                                int closedBranchOffset, int openBranchOffset,
                                const int* distrQElement, const int* distrQSide, const int* distrQKind,
                                const double* distrQWeight, const int* distrQDestination,
                                const int* distrQAlternateDestination, const int* distrQSelectorRow, const int* distrQDummyQRow,
                                const double* closedBranchParameters, const int* closedBranchIndices, const double* shuntParameters, const int* shuntIndices,
                                const double* openBranchParameters, const int* openBranchIndices,
                                const double* x, double* vals, double* F,
                                const char* disabled, const char* active,
                                int nDistrQWeightOverrides, const int* distrQWeightOverrideScenarioTerm, const double* distrQWeightOverrideWeight,
                                int nDistrQKeyOverrides, const int* distrQKeyOverrideScenarioMember, const double* distrQKeyOverrideKey,
                                const int* mode,
                                int nDistrQInactiveMembers, const int* distrQInactiveScenarioMember,
                                const int* distrQRowMember, const int* distrQInjectionMember, const double* mKey,
                                const int* distrQMemberReactiveLimitEntry, const int* mGroupStart, const int* mGroupCount,
                                const char* limitType, int nReactiveLimitEntries,
                                const char* pilotSelfCarrierSkip, int pilotSelfCarrierSkipStride,
                                const int* memberRelocationGroup, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows,
                                int nClosedBranchParameterOverrides, const int* closedBranchParameterOverrideScenarioElement, const double* closedBranchParameterOverrideValues,
                                int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                                int distrQTraceRow,
                                const char* distrQSubgroup, const int* reactiveRelocationDistrQMemberStart, int nDistrQMembersAll, BatchLayout layout) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nDistrQ * nScenarios) return;
    int c, s;
    itemAndScenario(tid, nDistrQ, layout, c, s);
    if (!active[s]) return;
    // STOCK PLACEMENT: a relocated child's sharing row moves to the bus's OWN row in any scenario that
    // kills its parent coupler, which turns that coupler's DUMMY_Q row into an identity (see buildDistrQDest).
    if (distrQSelectorRow != nullptr && distrQSelectorRow[c] >= 0 && distrQAlternateDestination != nullptr
            && mode[(size_t) s * n + distrQSelectorRow[c]] == GpuRowMode::FIXED_VARIABLE && distrQAlternateDestination[c * 6] <= -2) {
        distrQDestination = distrQAlternateDestination;
    }
    // the row of a remote member the RL check SWITCHED TO PQ (mode 4) hosts its OWN Q balance — its DISTR_Q emissions
    // stop for this scenario (the CPU deactivates only that member's DISTR_Q equation)
    {
        int d0 = distrQDestination[c * 6];
        if (d0 <= -2 && mode[(size_t) s * n + (-2 - d0)] == GpuRowMode::BUS_TARGET_Q) return;
        // plan-B4: the CURRENT pilot-Q carrier's row (a re-picked free role after the base free
        // member switched to PQ) hosts the group's relocated pilot balance — its DISTR_Q row is inactive
        // this scenario, exactly like OLF deactivating the new free member's DISTR_Q equation.
        if (memberRelocationGroup != nullptr && distrQInjectionMember[c] >= 0) {
            int g = memberRelocationGroup[distrQInjectionMember[c]];
            // A PILOT-SELF group carries no relocated balance while its pilot is still PV (its q feeds these
            // very DISTR_Q rows, weighted), so the carrier's row is NOT hosting anything and its DISTR_Q must
            // stay ACTIVE. Suppressing it there dropped an equation without gaining one: one short, singular.
            bool skipPilotSelfCarrier = pilotSelfCarrierSkip != nullptr && g >= 0 && pilotSelfCarrierSkip[(size_t) s * pilotSelfCarrierSkipStride + g];
            if (g >= 0 && !skipPilotSelfCarrier && d0 <= -2
                    && -2 - d0 == reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g] + chooseCarrierRowInSubgroup(g, reactiveRelocationGroupStart, reactiveRelocationCandidateRows, mode + (size_t) s * n,
                            distrQSubgroup != nullptr ? distrQSubgroup + (size_t) s * nDistrQMembersAll : nullptr, reactiveRelocationDistrQMemberStart)]) {
                return;
            }
        }
    }
    int e = distrQElement[c];
    int kind = distrQKind[c];
    if (kind == GpuReactiveTermKind::CLOSED_BRANCH_REACTIVE_FLOW && disabled[s * nElements + closedBranchOffset + e]) return;  // outaged closed contributor
    if (kind == GpuReactiveTermKind::OPEN_BRANCH_REACTIVE_FLOW && disabled[s * nElements + openBranchOffset + e]) return;  // outaged open contributor
    // kind 3 is a COUPLER's DUMMY_Q as part of this controller's reactive injection, and its element index
    // IS that dummy column (= its row). A scenario that kills the coupler turns that row into an identity, and
    // the coupler then carries no reactive flow at all — the contribution has to go with it. Measured on the
    // middle-section-loss fixtures: the surviving section's DISTR_Q row kept a coefficient 1 on the dead
    // coupler's DUMMY_Q and the solve landed 0.34 kV off (bus vl3_b 401.35 against OLF's 401.69).
    if (kind == GpuReactiveTermKind::DUMMY_Q && distrQDummyQRow != nullptr && distrQDummyQRow[c] >= 0 && mode[(size_t) s * n + distrQDummyQRow[c]] == GpuRowMode::FIXED_VARIABLE) return;
    double w = distrQWeight[c];
    int kM = distrQRowMember != nullptr ? distrQRowMember[c] : -1;
    if (kM >= 0) {
        // DYNAMIC reactive keys (plan-B3): an RL switch to PQ disables the member's controller, which zeroes
        // its key and renormalizes qPct over the survivors (updateReactiveKeys). Replicate OLF's
        // zero-then-sum arithmetic in member order, driven by the per-(entry, scenario) limitType.
        int gs = mGroupStart[kM];
        int gc = mGroupCount[kM];
        // SUBGROUPS (tasks #71): a split group's members pair only within their own subgroup — the other
        // subgroup's keys leave this denominator and its members' q terms leave this row (below).
        const char* subS = distrQSubgroup != nullptr ? distrQSubgroup + (size_t) s * nDistrQMembersAll : nullptr;
        char subK = subS != nullptr ? subS[kM] : 0;
        if (subS != nullptr && distrQInjectionMember[c] >= 0 && subS[distrQInjectionMember[c]] != subK) return;
        double sum = 0.0;
        for (int m = gs; m < gs + gc; m++) {
            int re = distrQMemberReactiveLimitEntry[m];
            bool switchedToPq = re >= 0 && limitType[(size_t) s * nReactiveLimitEntries + re] != GpuQLimitType::NONE;
            bool otherSub = subS != nullptr && subS[m] != subK;
            sum += (switchedToPq || otherSub || distrQMemberInactive(nDistrQInactiveMembers, distrQInactiveScenarioMember, s, m))
                    ? 0.0 : distrQKeyOf(nDistrQKeyOverrides, distrQKeyOverrideScenarioMember, distrQKeyOverrideKey, s, m, mKey);
        }
        // the q terms of a member SWITCHED TO PQ STAY in every survivor's row (weight = the renormalized
        // qPct of the ROW OWNER); its frozen target Q is balanced on the TARGET side instead — OLF's
        // getGeneratorReactivePowerDistributionTarget adds qPct_k * targetQ_j for each frozen j
        // (updateDistrQTargetsBatch keeps the per-scenario targets in step with the reactive-limit states)
        int reK = distrQMemberReactiveLimitEntry[kM];
        bool kSwitchedToPq = reK >= 0 && limitType[(size_t) s * nReactiveLimitEntries + reK] != GpuQLimitType::NONE;
        double qpk = (sum == 0.0 || kSwitchedToPq || distrQMemberInactive(nDistrQInactiveMembers, distrQInactiveScenarioMember, s, kM))
                ? 0.0 : distrQKeyOf(nDistrQKeyOverrides, distrQKeyOverrideScenarioMember, distrQKeyOverrideKey, s, kM, mKey) / sum;
        w = qpk - (distrQInjectionMember[c] == kM ? 1.0 : 0.0);
        if (kind == GpuReactiveTermKind::DUMMY_Q && distrQSide[c] == GpuTwoSides::TWO) {
            // The host folds the SIDE SIGN into a kind-3 weight (+dummyQ where the controller is
            // bus1, -dummyQ where it is bus2 — exactly createReactiveTerms' term); this live
            // recomputation rebuilds qPct_k - delta_jk and must put that sign back. Without it a
            // coupler whose BOTH ends are controllers of the same group cancels to zero at
            // qPct = 0.5: measured on rte6515 BUS-1014_COUPLER_1, where DUMMY_Q then appeared in NO
            // equation at all and the Jacobian had a structurally zero column, against OLF's
            // coefficient of +1.0 in DISTR_Q of VL-1014_1.
            w = -w;
        }
    }
    for (int k = 0; k < nDistrQWeightOverrides; k++) {                        // per-scenario qPct renormalization (gen outage)
        if (distrQWeightOverrideScenarioTerm[k * 2] == s && distrQWeightOverrideScenarioTerm[k * 2 + 1] == c) {
            w = distrQWeightOverrideWeight[k];
        }
    }
    const int* dst = distrQDestination + c * 6;
    const double* xs = x + (size_t) s * n;
    const ScenarioValues valsS = scenarioEntries(vals, s, nnz, layout);  // none on a mismatch-only fill: F alone
    const ScenarioValues FS = scenarioEntries(F, s, n, layout);

    if (kind == GpuReactiveTermKind::SHUNT_COMPENSATOR_REACTIVE_FLOW) {  // shunt q: {q -> F, dq/dv -> J(row, vcol)}
        const double* sp = shuntParameters + e * 2;
        int vcol = shuntIndices[e * 5 + 2];
        double g = sp[0];
        double b = sp[1];
        applyShuntOverride(nShuntAdmittanceOverrides, shuntAdmittanceOverrideScenarioElement, shuntAdmittanceOverrideDelta, s, e, g, b);
        double so[SHUNT_OUTPUT_COUNT];
        updateShunt(xs[vcol], g, b, 0, so + 0, so + 1, so + 2, so + 3, so + 4);
        if (dst[0] <= -2 && (-2 - dst[0]) == distrQTraceRow) {
            printf("DQF kind=1(shunt) elem=%d w=%.6f contrib=%.9f\n", e, w, w * so[2]);
        }
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * so[2]);
        if (valsS && dst[1] >= 0) atomicAdd(&valsS[dst[1]], w * so[3]);
        return;
    }
    if (kind == GpuReactiveTermKind::DUMMY_Q) {  // zero-impedance DUMMY_Q: linear w*dummyQ
        if (dst[0] <= -2 && (-2 - dst[0]) == distrQTraceRow) {
            printf("DQF kind=3(ziDummyQ) col=%d w=%.6f contrib=%.9f\n", e, w, w * xs[e]);
        }
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * xs[e]);
        if (valsS && dst[1] >= 0) atomicAdd(&valsS[dst[1]], w);
        return;
    }
    if (kind == GpuReactiveTermKind::OPEN_BRANCH_REACTIVE_FLOW) {  // half-open branch: connected-side q + dq/dv
        const double* op = openBranchParameters + e * 7;
        const int* openBranch = openBranchIndices + e * 6;
        int vcol = distrQSide[c] == GpuTwoSides::ONE ? openBranch[4] : openBranch[5];
        double v1 = distrQSide[c] == GpuTwoSides::ONE ? xs[vcol] : 1.0;
        double v2 = distrQSide[c] == GpuTwoSides::TWO ? xs[vcol] : 1.0;
        double oo[OPEN_BRANCH_OUTPUT_COUNT];
        updateOpenBranch(op[0], op[1], op[2], op[3], op[4], op[5], v1, op[6], v2, 0,
                       oo + 0, oo + 1, oo + 2, oo + 3, oo + 4, oo + 5, oo + 6, oo + 7);
        int qi = distrQSide[c] == GpuTwoSides::ONE ? 6 : 2;
        if (dst[0] <= -2 && (-2 - dst[0]) == distrQTraceRow) {
            printf("DQF kind=2(halfOpen) elem=%d w=%.6f contrib=%.9f\n", e, w, w * oo[qi]);
        }
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * oo[qi]);
        if (valsS && dst[1] >= 0) atomicAdd(&valsS[dst[1]], w * oo[qi + 1]);
        return;
    }
    // kind 0: closed-branch q1/q2 + its derivatives.
    // The per-scenario pi override MUST be honoured here, exactly as fillRemoteMemberQBatch does.
    // It is not a tap-only channel: an outaged branch on this path is nulled by zeroing its
    // admittance through it, NOT by the disabled mask. Reading the base pi instead kept three
    // transformers into an outaged busbar live against a far bus frozen at v=1.0 and injected
    // +0.813908 of phantom reactive into one DISTR_Q row (rte6515 BUS-1265_BBS, measured residual
    // -0.814738) - which moved the sharing solution 2.1 kV and then tripped a spurious RL switch to PQ.
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
    const int* q = closedBranchIndices + e * 12;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    double o[CLOSED_BRANCH_OUTPUT_COUNT];
    updateClosedBranch(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    int qi = distrQSide[c] == GpuTwoSides::ONE ? 7 : 21;  // side-1 q1[7]..dq1dr1[13]; side-2 q2[21]..dq2dr1[27]
    double out6[6] = {o[qi], o[qi + 1], o[qi + 2], o[qi + 3], o[qi + 4], o[qi + 6]};
    for (int m = 0; m < 6; m++) {
        int d = dst[m];
        double val = w * out6[m];
        if (d <= -2 && (-2 - d) == distrQTraceRow) {
            printf("DQF kind=0(closed) elem=%d w=%.6f v1=%.9f v2=%.9f y=%.6f contrib=%.9f\n",
                   e, w, v1, v2, p[0], val);
        }
        addAt(d, val, valsS, FS);
    }
}

// Batched DISTR_RHO fill: one thread per (contribution, scenario). Linear coef*r1 into the controller's
// BRANCH_RHO1 row — strided per scenario, active-gated, no mask (controller-outage contingencies are
// rejected to the CPU because the 1/count coefficients reform).
__global__ void fillDistrRhoBatch(int nDistrRho, int nScenarios, int n, int nnz,
                                  const int* distrRhoColumn, const double* distrRhoCoefficient, const int* distrRhoDestination,
                                  const double* x, double* vals, double* F, const char* active,
                                  int nDistrRhoCoefficientOverrides, const int* distrRhoOverrideScenarioTerm, const double* distrRhoOverrideCoefficient,
                                  const int* distrRhoRelocationGroup, const int* mode, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRows, BatchLayout layout) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nDistrRho * nScenarios) return;
    int c, s;
    itemAndScenario(tid, nDistrRho, layout, c, s);
    if (!active[s]) return;
    // the CURRENT pilot-Q carrier's row hosts the relocated pilot balance — its DISTR_RHO row is
    // inactive PER SCENARIO (same rule as fillDistrQBatch; the
    // machine's zeroed coefficient overrides covers machine steps, but a bare batch solve — the batch-of-1
    // sequential route on a clamp-in-the-base network — has no coefficient overrides and needs the in-kernel skip)
    if (distrRhoRelocationGroup != nullptr && distrRhoRelocationGroup[c] >= 0) {
        int g = distrRhoRelocationGroup[c];
        if (-2 - distrRhoDestination[c * 2] == reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g] + chooseCarrierRow(g, reactiveRelocationGroupStart, reactiveRelocationCandidateRows, mode + (size_t) s * n)]) return;
    }
    double coef = distrRhoCoefficient[c];
    for (int k = 0; k < nDistrRhoCoefficientOverrides; k++) {                        // per-scenario 1/count' renormalization
        if (distrRhoOverrideScenarioTerm[k * 2] == s && distrRhoOverrideScenarioTerm[k * 2 + 1] == c) {
            coef = distrRhoOverrideCoefficient[k];
        }
    }
    int dF = distrRhoDestination[c * 2 + 0];
    int dJ = distrRhoDestination[c * 2 + 1];
    if (dF <= -2) atomicAdd(&scenarioEntries(F, s, n, layout)[-2 - dF], coef * x[(size_t) s * n + distrRhoColumn[c]]);
    if (vals != nullptr && dJ >= 0) atomicAdd(&scenarioEntries(vals, s, nnz, layout)[dJ], coef);
}

// Batched DISTR_SHUNT_B fill: the SHUNT_B analogue of fillDistrRhoBatch. Linear coef*b into the controller's
// SHUNT_B row, strided per scenario, active-gated.
__global__ void fillDistrShuntBBatch(int nDistrShuntB, int nScenarios, int n, int nnz,
                                     const int* distrShuntBColumn, const double* distrShuntBCoefficient, const int* distrShuntBDestination,
                                     const double* x, double* vals, double* F, const char* active, BatchLayout layout) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nDistrShuntB * nScenarios) return;
    int c, s;
    itemAndScenario(tid, nDistrShuntB, layout, c, s);
    if (!active[s]) return;
    double coef = distrShuntBCoefficient[c];
    int dF = distrShuntBDestination[c * 2 + 0];
    int dJ = distrShuntBDestination[c * 2 + 1];
    if (dF <= -2) atomicAdd(&scenarioEntries(F, s, n, layout)[-2 - dF], coef * x[(size_t) s * n + distrShuntBColumn[c]]);
    if (vals != nullptr && dJ >= 0) atomicAdd(&scenarioEntries(vals, s, nnz, layout)[dJ], coef);
}

}  // namespace olfgpu
