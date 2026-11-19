// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: ReactiveLimitsOuterLoop tables, curves, forced target Q and per-scenario overrides.
#include "jni_common.cuh"
#include "../solver/scenario_install.cuh"
#include "slot_table_install.cuh"
#include "../common/codes/gpu_reactive_limits_state.cuh"
#include "../common/codes/gpu_voltage_control_role.cuh"

// Reactive-capability CURVES at runtime P: install (or clear, with nulls) the per-gen curve channel
// on a batch context — see reactiveCapabilityCurveLimitsBatch. reactiveCapabilityCurveIndices stride 5 {reactiveLimitEntry, pRow, ptsOff, npts, distributedSlackGeneratorIndex}; reactiveCapabilityCurveParameters
// stride 5 {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu}; reactiveCapabilityCurvePoints stride 3
// {p_MW, minQ_MW, maxQ_MW} sorted by p; reactiveCapabilityCurveHvdcStation (null when no entry is an AC-emulation VSC) stride 2 {hvdc, side}.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveCapabilityCurves)(
        JNIEnv* env, jclass, jlong handle, jintArray jReactiveCapabilityCurveIndices, jdoubleArray jReactiveCapabilityCurveParameters, jdoubleArray jReactiveCapabilityCurvePoints,
        jintArray jReactiveCapabilityCurveHvdcStation, jint extrapolate) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->reactiveLimits.reactiveCapabilityCurveHasHvdc = false;
    if (jReactiveCapabilityCurveIndices == nullptr || jReactiveCapabilityCurveParameters == nullptr || jReactiveCapabilityCurvePoints == nullptr) {
        ctx->reactiveLimits.nReactiveCapabilityCurves = 0;
        return;
    }
    int nReactiveCapabilityCurves = env->GetArrayLength(jReactiveCapabilityCurveIndices) / 5;   // stride 5: {reactiveLimitEntry, pRow, ptsOff, npts, distributedSlackGeneratorIndex}
    if (nReactiveCapabilityCurves == 0) {
        ctx->reactiveLimits.nReactiveCapabilityCurves = 0;
        return;
    }
    if (jReactiveCapabilityCurveHvdcStation != nullptr) {                      // stride 2 per entry: {hvdc, GpuTwoSides side}, -1 / NONE for a plain generator
        const JniInts reactiveCapabilityCurveHvdcStation(env, jReactiveCapabilityCurveHvdcStation);
        ctx->reactiveLimits.dReactiveCapabilityCurveHvdcStation.upload(reactiveCapabilityCurveHvdcStation, static_cast<size_t>(nReactiveCapabilityCurves) * 2 * sizeof(int));
        ctx->reactiveLimits.reactiveCapabilityCurveHasHvdc = true;
    }
    const JniInts reactiveCapabilityCurveIndices(env, jReactiveCapabilityCurveIndices);
    const JniDoubles reactiveCapabilityCurveParameters(env, jReactiveCapabilityCurveParameters);
    const JniDoubles reactiveCapabilityCurvePoints(env, jReactiveCapabilityCurvePoints);
    ctx->reactiveLimits.dReactiveCapabilityCurveIndices.upload(reactiveCapabilityCurveIndices, static_cast<size_t>(nReactiveCapabilityCurves) * 5 * sizeof(int));
    ctx->reactiveLimits.dReactiveCapabilityCurveParameters.upload(reactiveCapabilityCurveParameters, static_cast<size_t>(nReactiveCapabilityCurves) * 5 * sizeof(double));
    ctx->reactiveLimits.dReactiveCapabilityCurvePoints.upload(reactiveCapabilityCurvePoints, static_cast<size_t>(env->GetArrayLength(jReactiveCapabilityCurvePoints)) * sizeof(double));
    // The curves of each reactive-limit entry, CSR (pack order kept within an entry): reactiveCapabilityCurveLimitsBatch runs
    // one thread per (entry, scenario) and walked EVERY curve to find its own.
    {
        std::vector<int> ptr(static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries) + 1, 0);
        std::vector<int> list(nReactiveCapabilityCurves);
        for (int g = 0; g < nReactiveCapabilityCurves; g++) {
            int e = reactiveCapabilityCurveIndices[g * 5];
            if (e < 0 || e >= ctx->reactiveLimits.nReactiveLimitEntries) {
                ctx->reactiveLimits.nReactiveCapabilityCurves = 0;
                env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                              "GPU batch: a curve's reactive-limit entry is outside the pack");
                return;
            }
            ptr[e + 1]++;
        }
        for (int e = 0; e < ctx->reactiveLimits.nReactiveLimitEntries; e++) ptr[e + 1] += ptr[e];
        std::vector<int> cursor(ptr.begin(), ptr.end() - 1);
        for (int g = 0; g < nReactiveCapabilityCurves; g++) list[cursor[reactiveCapabilityCurveIndices[g * 5]]++] = g;
        ctx->reactiveLimits.dReactiveLimitEntryCurveStart.upload(ptr.data(), ptr.size() * sizeof(int));
        ctx->reactiveLimits.dReactiveLimitEntryCurves.upload(list.data(), list.size() * sizeof(int));
    }
    ctx->reactiveLimits.dReactiveLimitMinQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->reactiveLimits.nReactiveLimitEntries * sizeof(double));
    ctx->reactiveLimits.dReactiveLimitMaxQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->reactiveLimits.nReactiveLimitEntries * sizeof(double));
    ctx->reactiveLimits.dReactiveLimitLoadQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->reactiveLimits.nReactiveLimitEntries * sizeof(double));
    ctx->reactiveLimits.nReactiveCapabilityCurves = nReactiveCapabilityCurves;
    ctx->reactiveLimits.reactiveCapabilityCurveExtrapolate = extrapolate;
}

// forceTargetQInReactiveLimits channel (GpuBatchForcedTargetQ): forcedTargetQIndices stride 7 {qRow, distributedSlackGeneratorIndex, ptsOff, npts,
// distributedSlackBusIndex, baseQRowMode, reactiveLimitEntry}; forcedTargetQParameters stride 5 {basePpu, rawTargetQPu, baseClampedPu, share, baseRowTargetP}; forcedTargetQPoints
// stride 3 {p_MW, minQ_MW, maxQ_MW} sorted by p. Null clears the channel (a cached context reused by a run
// without it).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchForcedTargetQ)(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jIn, jdoubleArray jPts, jint extrapolate) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->forcedTargetQ.forcedTargetQPreDistribution.clear();
    if (jIdx == nullptr || jIn == nullptr || jPts == nullptr || env->GetArrayLength(jIdx) == 0) {
        ctx->forcedTargetQ.nForcedTargetQ = 0;
        ctx->forcedTargetQ.nForcedTargetQLimitEntries = 0;
        return;
    }
    int nForcedTargetQ = env->GetArrayLength(jIdx) / 7;
    const JniInts idx(env, jIdx);
    int nForcedTargetQLimitEntries = 0;
    for (int e = 0; e < nForcedTargetQ; e++) {
        nForcedTargetQLimitEntries += idx[e * 7 + 6] >= 0 ? 1 : 0;
    }
    const JniDoubles in(env, jIn);
    const JniDoubles pts(env, jPts);
    ctx->forcedTargetQ.dForcedTargetQIndices.upload(idx, static_cast<size_t>(nForcedTargetQ) * 7 * sizeof(int));
    ctx->forcedTargetQ.dForcedTargetQParameters.upload(in, static_cast<size_t>(nForcedTargetQ) * 5 * sizeof(double));
    ctx->forcedTargetQ.dForcedTargetQPoints.upload(pts, static_cast<size_t>(env->GetArrayLength(jPts)) * sizeof(double));
    ctx->forcedTargetQ.dForcedTargetQApplied.alloc(static_cast<size_t>(ctx->nScenarios) * nForcedTargetQ * sizeof(double));
    ctx->forcedTargetQ.nForcedTargetQ = nForcedTargetQ;
    ctx->forcedTargetQ.nForcedTargetQLimitEntries = nForcedTargetQLimitEntries;
    ctx->forcedTargetQ.forcedTargetQExtrapolate = extrapolate;
}

// The installed scenarios' loss pre-distribution moves of the forced generators ([nInstalled * nForcedTargetQ], pu), or null
// when none of them pre-distributes. Set before each solve that carries contingencies.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchForcedTargetQPre)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jPre) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installSlotRows(env, ctx, ctx->forcedTargetQ.forcedTargetQPreDistribution, ctx->forcedTargetQ.nForcedTargetQ, ctx->forcedTargetQ.nForcedTargetQ > 0 ? jPre : nullptr,
                        "setBatchForcedTargetQPre");
    }
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitStrongestPvKeys)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jNomV, jintArray jPRow, jdoubleArray jStaticP, jintArray jIdRank) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->reactiveLimits.hasReactiveLimitStrongestPvKeys = 0;
    if (ctx->reactiveLimits.nReactiveLimitEntries == 0 || jNomV == nullptr || jPRow == nullptr || jStaticP == nullptr || jIdRank == nullptr) {
        return;
    }
    if (env->GetArrayLength(jNomV) != ctx->reactiveLimits.nReactiveLimitEntries || env->GetArrayLength(jPRow) != ctx->reactiveLimits.nReactiveLimitEntries
            || env->GetArrayLength(jStaticP) != ctx->reactiveLimits.nReactiveLimitEntries || env->GetArrayLength(jIdRank) != ctx->reactiveLimits.nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchReactiveLimitStrongestPvKeys: array length != nReactiveLimitEntries");
        return;
    }
    const JniDoubles nv(env, jNomV);
    const JniInts pr(env, jPRow);
    const JniDoubles sp(env, jStaticP);
    const JniInts ir(env, jIdRank);
    ctx->reactiveLimits.dReactiveLimitNominalV.upload(nv, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries) * sizeof(double));
    ctx->reactiveLimits.dReactiveLimitActivePowerRow.upload(pr, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries) * sizeof(int));
    ctx->reactiveLimits.dReactiveLimitTargetP.upload(sp, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries) * sizeof(double));
    ctx->reactiveLimits.dReactiveLimitIdRank.upload(ir, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries) * sizeof(int));
    ctx->reactiveLimits.hasReactiveLimitStrongestPvKeys = 1;
}

// voltageRemoteControlRobustMode (OLF's default): install the robust-mode channel on a batch context —
// the per-entry initial generation target Q that a realistic-V freeze holds, and the two thresholds
// ReactiveLimitsOuterLoop tests the controller's OWN voltage against (minRealisticVoltage ·
// REALISTIC_VOLTAGE_MARGIN and maxRealisticVoltage / REALISTIC_VOLTAGE_MARGIN, margin 1.02). robust = 0
// clears the channel, so a cached context reused by a run with the mode off stops applying it.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitsRobustMode)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jInitTgtQ, jdouble vLo, jdouble vHi, jint robust) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->reactiveLimits.reactiveLimitsRobustMode = 0;
    if (ctx->reactiveLimits.nReactiveLimitEntries == 0 || robust == 0 || jInitTgtQ == nullptr) {
        return;
    }
    if (env->GetArrayLength(jInitTgtQ) != ctx->reactiveLimits.nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchReactiveLimitsRobustMode: array length != nReactiveLimitEntries");
        return;
    }
    const JniDoubles q0(env, jInitTgtQ);
    ctx->reactiveLimits.dReactiveLimitInitialTargetQ.upload(q0, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries) * sizeof(double));
    ctx->reactiveLimits.reactiveLimitsRobustMinRealisticVoltage = vLo;
    ctx->reactiveLimits.reactiveLimitsRobustMaxRealisticVoltage = vHi;
    ctx->reactiveLimits.reactiveLimitsRobustMode = 1;
}

// AcloadFlowEngine's realistic-voltage check: install (or clear) the per-row "this row is a
// checked bus voltage" mask and the two bounds. enabled = 0 clears it on a cached context.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchRealisticCheck)(
        JNIEnv* env, jclass, jlong handle, jbyteArray jRowChecked, jdouble minV, jdouble maxV, jint enabled,
        jint fixingLoop) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->realisticVoltage.realisticCheck = 0;
    if (enabled == 0 || jRowChecked == nullptr) {
        return;
    }
    if (env->GetArrayLength(jRowChecked) != ctx->n) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRealisticCheck: array length != n");
        return;
    }
    const JniBytes rc(env, jRowChecked);
    ctx->realisticVoltage.dRealisticRow.upload(rc, static_cast<size_t>(ctx->n) * sizeof(char));
    ctx->realisticVoltage.realisticMinV = minV;
    ctx->realisticVoltage.realisticMaxV = maxV;
    ctx->realisticVoltage.realisticFixingLoop = fixingLoop;
    ctx->realisticVoltage.realisticCheck = 1;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitSvcSusceptanceLimits)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jBmin, jdoubleArray jBmax, jdoubleArray jV2Base) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->reactiveLimits.nReactiveLimitEntries == 0) {
        return;
    }
    if (env->GetArrayLength(jBmin) != ctx->reactiveLimits.nReactiveLimitEntries || env->GetArrayLength(jBmax) != ctx->reactiveLimits.nReactiveLimitEntries
            || env->GetArrayLength(jV2Base) != ctx->reactiveLimits.nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchReactiveLimitSvcSusceptanceLimits: array length != nReactiveLimitEntries");
        return;
    }
    const JniDoubles bmin(env, jBmin);
    const JniDoubles bmax(env, jBmax);
    const JniDoubles v2b(env, jV2Base);
    ctx->reactiveLimits.dReactiveLimitSvcMinB.upload(bmin, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries) * sizeof(double));
    ctx->reactiveLimits.dReactiveLimitSvcMaxB.upload(bmax, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries) * sizeof(double));
    ctx->reactiveLimits.dReactiveLimitBaseVSquared.upload(v2b, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries) * sizeof(double));
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitBaseQLimitType)(
        JNIEnv* env, jclass, jlong handle, jbyteArray jKinds) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKinds == nullptr || env->GetArrayLength(jKinds) == 0) {
        ctx->reactiveLimits.hasReactiveLimitBaseQLimitType = 0;
        return;
    }
    if (env->GetArrayLength(jKinds) != ctx->reactiveLimits.nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchReactiveLimitBaseQLimitType: array length != nReactiveLimitEntries");
        return;
    }
    const JniBytes kinds(env, jKinds);
    ctx->reactiveLimits.dReactiveLimitBaseQLimitType.alloc(static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries));
    CUDA_CHECK(cudaMemcpy(ctx->reactiveLimits.dReactiveLimitBaseQLimitType.p, kinds, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries), cudaMemcpyHostToDevice));
    ctx->reactiveLimits.hasReactiveLimitBaseQLimitType = 1;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitRemoteEntries)(
        JNIEnv* env, jclass, jlong handle, jbyteArray jReactiveLimitVoltageControlRole, jintArray jReactiveLimitComparedVoltageRow,
        jintArray jRemoteMemberReactiveTermRow, jintArray jRemoteMemberReactiveTermElement, jintArray jRemoteMemberReactiveTermSide, jintArray jRemoteMemberReactiveTermKind, jdoubleArray jRemoteMemberReactiveTermWeight,
        jintArray jDistrQRowMember, jintArray jDistrQInjectionMember, jdoubleArray jDistrQMemberKey, jintArray jDistrQMemberReactiveLimitEntry,
        jintArray jDistrQMemberGroupStart, jintArray jDistrQMemberGroupCount, jintArray jDistrQMemberRow, jbyteArray jDistrQMemberHasRow,
        jintArray jDistrQMemberRelocationGroup, jdoubleArray jDistrQMemberTargetQ, jintArray jDistrQMemberTargetPeerStart, jintArray jDistrQMemberTargetPeer) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jReactiveLimitVoltageControlRole == nullptr || jReactiveLimitComparedVoltageRow == nullptr || env->GetArrayLength(jReactiveLimitVoltageControlRole) == 0) {
        ctx->reactiveLimits.nReactiveLimitRemoteEntries = 0;
        ctx->remoteVoltageControl.nRemoteMemberReactiveTerms = 0;
        ctx->remoteVoltageControl.nDistrQDynamicKeys = 0;
        ctx->remoteVoltageControl.hasDistrQRelocationGroups = 0;
        return;
    }
    int nReactiveLimitEntries = env->GetArrayLength(jReactiveLimitVoltageControlRole);
    if (nReactiveLimitEntries != ctx->reactiveLimits.nReactiveLimitEntries || env->GetArrayLength(jReactiveLimitComparedVoltageRow) != ctx->reactiveLimits.nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchReactiveLimitRemoteEntries: array length != nReactiveLimitEntries");
        return;
    }
    const JniBytes voltageControlRole(env, jReactiveLimitVoltageControlRole);
    const JniInts comparedVoltageRow(env, jReactiveLimitComparedVoltageRow);
    int count = 0;
    int selfCount = 0;
    for (int i = 0; i < nReactiveLimitEntries; i++) {
        if (voltageControlRole[i] != GpuVoltageControlRole::LOCAL) {
            count++;
        }
        if (voltageControlRole[i] == GpuVoltageControlRole::PILOT_SELF) {
            selfCount++;
        }
    }
    ctx->reactiveLimits.nReactiveLimitPilotSelfEntries = selfCount;
    ctx->reactiveLimits.dReactiveLimitVoltageControlRole.upload(voltageControlRole, static_cast<size_t>(nReactiveLimitEntries) * sizeof(char));
    ctx->reactiveLimits.dReactiveLimitComparedVoltageRow.upload(comparedVoltageRow, static_cast<size_t>(nReactiveLimitEntries) * sizeof(int));
    ctx->reactiveLimits.nReactiveLimitRemoteEntries = count;
    ctx->remoteVoltageControl.nRemoteMemberReactiveTerms = 0;
    if (jRemoteMemberReactiveTermRow != nullptr && env->GetArrayLength(jRemoteMemberReactiveTermRow) > 0) {
        int nRemoteMemberReactiveTerms = env->GetArrayLength(jRemoteMemberReactiveTermRow);
        const JniInts remoteMemberReactiveTermRow(env, jRemoteMemberReactiveTermRow);
        const JniInts remoteMemberReactiveTermElement(env, jRemoteMemberReactiveTermElement);
        const JniInts remoteMemberReactiveTermSide(env, jRemoteMemberReactiveTermSide);
        const JniInts remoteMemberReactiveTermKind(env, jRemoteMemberReactiveTermKind);
        const JniDoubles remoteMemberReactiveTermWeight(env, jRemoteMemberReactiveTermWeight);
        ctx->remoteVoltageControl.dRemoteMemberReactiveTermRow.upload(remoteMemberReactiveTermRow, static_cast<size_t>(nRemoteMemberReactiveTerms) * sizeof(int));
        ctx->remoteVoltageControl.dRemoteMemberReactiveTermElement.upload(remoteMemberReactiveTermElement, static_cast<size_t>(nRemoteMemberReactiveTerms) * sizeof(int));
        ctx->remoteVoltageControl.dRemoteMemberReactiveTermSide.upload(remoteMemberReactiveTermSide, static_cast<size_t>(nRemoteMemberReactiveTerms) * sizeof(int));
        ctx->remoteVoltageControl.dRemoteMemberReactiveTermKind.upload(remoteMemberReactiveTermKind, static_cast<size_t>(nRemoteMemberReactiveTerms) * sizeof(int));
        ctx->remoteVoltageControl.dRemoteMemberReactiveTermWeight.upload(remoteMemberReactiveTermWeight, static_cast<size_t>(nRemoteMemberReactiveTerms) * sizeof(double));
        ctx->remoteVoltageControl.nRemoteMemberReactiveTerms = nRemoteMemberReactiveTerms;
    }
    ctx->remoteVoltageControl.nDistrQDynamicKeys = 0;
    ctx->remoteVoltageControl.hasDistrQRelocationGroups = 0;
    // Keyed on the MEMBER list, not on the DISTR_Q rows: a SINGLETON remote group has members (and a
    // pilot-Q relocation group, so its pilot's control-loss flip must be armed) but no DISTR_Q row at
    // all — gating on distrQRowMember left hasDistrQRelocationGroups at 0 for it, and the base load flow on the driver
    // kept the pilot's V equation after switching its only member to PQ (2.4e-3 pu on the pilot's Q).
    if (jDistrQMemberKey != nullptr && env->GetArrayLength(jDistrQMemberKey) > 0) {
        int nDistrQTermMembers = jDistrQRowMember != nullptr ? env->GetArrayLength(jDistrQRowMember) : 0;
        int nDistrQMembers = env->GetArrayLength(jDistrQMemberKey);
        const JniInts distrQRowMember(env, nDistrQTermMembers > 0 ? jDistrQRowMember : nullptr);
        const JniInts distrQInjectionMember(env, nDistrQTermMembers > 0 ? jDistrQInjectionMember : nullptr);
        const JniDoubles distrQMemberKey(env, jDistrQMemberKey);
        const JniInts distrQMemberReactiveLimitEntry(env, jDistrQMemberReactiveLimitEntry);
        const JniInts distrQMemberGroupStart(env, jDistrQMemberGroupStart);
        const JniInts distrQMemberGroupCount(env, jDistrQMemberGroupCount);
        const JniInts distrQMemberRow(env, jDistrQMemberRow);
        const JniBytes distrQMemberHasRow(env, jDistrQMemberHasRow);
        ctx->remoteVoltageControl.dDistrQMemberRow.upload(distrQMemberRow, static_cast<size_t>(nDistrQMembers) * sizeof(int));
        ctx->remoteVoltageControl.dDistrQMemberHasRow.upload(distrQMemberHasRow, static_cast<size_t>(nDistrQMembers) * sizeof(char));
        if (jDistrQMemberTargetQ == nullptr || env->GetArrayLength(jDistrQMemberTargetQ) != nDistrQMembers) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchReactiveLimitRemoteEntries: distrQMemberTargetQ missing or length != members");
            return;
        }
        const JniDoubles distrQMemberTargetQ(env, jDistrQMemberTargetQ);
        ctx->remoteVoltageControl.dDistrQMemberTargetQ.upload(distrQMemberTargetQ, static_cast<size_t>(nDistrQMembers) * sizeof(double));
        if (jDistrQMemberTargetPeerStart == nullptr || env->GetArrayLength(jDistrQMemberTargetPeerStart) != nDistrQMembers + 1 || jDistrQMemberTargetPeer == nullptr) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchReactiveLimitRemoteEntries: distrQMemberTargetPeerStart missing or length != members + 1");
            return;
        }
        const JniInts distrQMemberTargetPeerStart(env, jDistrQMemberTargetPeerStart);
        ctx->remoteVoltageControl.dDistrQMemberTargetPeerStart.upload(distrQMemberTargetPeerStart, static_cast<size_t>(nDistrQMembers + 1) * sizeof(int));
        int nDistrQMemberTargetPeers = env->GetArrayLength(jDistrQMemberTargetPeer);
        if (nDistrQMemberTargetPeers > 0) {
            const JniInts distrQMemberTargetPeer(env, jDistrQMemberTargetPeer);
            ctx->remoteVoltageControl.dDistrQMemberTargetPeer.upload(distrQMemberTargetPeer, static_cast<size_t>(nDistrQMemberTargetPeers) * sizeof(int));
        }
        ctx->remoteVoltageControl.nDistrQMembers = nDistrQMembers;
        if (nDistrQTermMembers > 0) {
            ctx->remoteVoltageControl.dDistrQRowMember.upload(distrQRowMember, static_cast<size_t>(nDistrQTermMembers) * sizeof(int));
            ctx->remoteVoltageControl.dDistrQInjectionMember.upload(distrQInjectionMember, static_cast<size_t>(nDistrQTermMembers) * sizeof(int));
        }
        ctx->remoteVoltageControl.dDistrQMemberKey.upload(distrQMemberKey, static_cast<size_t>(nDistrQMembers) * sizeof(double));
        ctx->remoteVoltageControl.dDistrQMemberReactiveLimitEntry.upload(distrQMemberReactiveLimitEntry, static_cast<size_t>(nDistrQMembers) * sizeof(int));
        ctx->remoteVoltageControl.dDistrQMemberGroupStart.upload(distrQMemberGroupStart, static_cast<size_t>(nDistrQMembers) * sizeof(int));
        ctx->remoteVoltageControl.dDistrQMemberGroupCount.upload(distrQMemberGroupCount, static_cast<size_t>(nDistrQMembers) * sizeof(int));
        ctx->remoteVoltageControl.nDistrQDynamicKeys = nDistrQTermMembers;
        ctx->remoteVoltageControl.hasDistrQRelocationGroups = 0;
        if (jDistrQMemberRelocationGroup != nullptr && env->GetArrayLength(jDistrQMemberRelocationGroup) == nDistrQMembers) {
            const JniInts distrQMemberRelocationGroup(env, jDistrQMemberRelocationGroup);
            std::vector<char> isGeneratorVoltageControl(ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->nReactiveRelocationGroups : 1, 0);
            for (int i = 0; i < nDistrQMembers; i++) {
                if (distrQMemberRelocationGroup[i] >= 0) {
                    ctx->remoteVoltageControl.hasDistrQRelocationGroups = 1;                  // plan-B4: relocation-linked group present
                    isGeneratorVoltageControl[distrQMemberRelocationGroup[i]] = 1;   // this relocation group is a remote generator voltage control
                }
            }
            for (int g = 0; g < ctx->structure->nReactiveRelocationGroups; g++) {         // a DEPENDENT pilot's group is as generator-controlled
                if (g < static_cast<int>(ctx->structure->hReactiveRelocationDependentBaseGroup.size()) && ctx->structure->hReactiveRelocationDependentBaseGroup[g] >= 0) {   // as its base
                    isGeneratorVoltageControl[g] = isGeneratorVoltageControl[ctx->structure->hReactiveRelocationDependentBaseGroup[g]];
                }
            }
            ctx->remoteVoltageControl.dDistrQMemberRelocationGroup.upload(distrQMemberRelocationGroup, static_cast<size_t>(nDistrQMembers) * sizeof(int));
            if (ctx->remoteVoltageControl.hasDistrQRelocationGroups) {
                ctx->remoteVoltageControl.dReactiveRelocationIsGeneratorVoltageControl.upload(isGeneratorVoltageControl.data(), isGeneratorVoltageControl.size() * sizeof(char));
            }
        }
  }
}

// Per (installed scenario, entry): the frozen target Q a continued solve restores, NaN = the limit. Null or empty: none.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitSeedTargetQ)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jTargets) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installSlotRows(env, ctx, ctx->reactiveLimitSeedTargetQ, ctx->reactiveLimits.nReactiveLimitEntries,
                        jTargets != nullptr && env->GetArrayLength(jTargets) > 0 ? jTargets : nullptr, "setBatchReactiveLimitSeedTargetQ");
    }
}

extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, readBatchReactiveLimitTargetQ)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->reactiveLimits.nReactiveLimitEntries <= 0) {
        return nullptr;
    }
    int nS = ctx->nScenarios;
    int n = ctx->structure->n;
    DeviceBuf out;
    out.alloc(static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries) * nS * sizeof(double));
    exportReactiveLimitTargetQ<<<grid(ctx->reactiveLimits.nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->reactiveLimits.nReactiveLimitEntries, nS, n, ctx->reactiveLimits.dReactiveLimitVoltageRow.as<int>(),
            ctx->reactiveLimits.dReactiveLimitReactivePowerRow.p ? ctx->reactiveLimits.dReactiveLimitReactivePowerRow.as<int>() : nullptr, ctx->reactiveLimits.dReactiveLimitQLimitType.as<char>(), ctx->dMode.as<int>(),
            ctx->dTarget.as<double>(), ctx->reactiveLimits.reactiveLimitIslanded.deviceOrNull(),
            ctx->remoteVoltageControl.hasDistrQRelocationGroups ? ctx->structure->nReactiveRelocationGroups : 0, pilotSelfReactiveLimitEntryOrNull(ctx),
            ctx->remoteVoltageControl.hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->dReactiveRelocationGroupStart.as<int>() : nullptr,
            ctx->remoteVoltageControl.hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->dReactiveRelocationCandidateRows.as<int>() : nullptr,
            ctx->remoteVoltageControl.hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0 ? ctx->remoteVoltageControl.dReactiveRelocationPilotTargetVDeactivated.as<char>() : nullptr,
            ctx->structure->dReactiveRelocationDependentBaseGroup.p ? ctx->structure->dReactiveRelocationDependentBaseGroup.as<int>() : nullptr, out.as<double>());
    std::vector<double> host;
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(readInstalledRows(ctx, out.p, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries), host));
    jdoubleArray arr = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(arr, 0, static_cast<jsize>(host.size()), host.data());
    return arr;
}

// Per (scenario, reactive-limit entry): the PV/PQ state the last solve ended with, a GpuReactiveLimitsState (its limit type
// and PV -> PQ switch count), in the install's order; null without entries.
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, readBatchReactiveLimitsFinalState)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->reactiveLimits.nReactiveLimitEntries <= 0) {
        return nullptr;
    }
    std::vector<char> qLimitType, switchCount;
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(readInstalledRows(ctx, ctx->reactiveLimits.dReactiveLimitQLimitType.p, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries), qLimitType));
    CUDA_CHECK(readInstalledRows(ctx, ctx->reactiveLimits.dReactiveLimitPvPqSwitchCount.p, static_cast<size_t>(ctx->reactiveLimits.nReactiveLimitEntries), switchCount));
    std::vector<int> state(qLimitType.size());
    for (size_t k = 0; k < state.size(); k++) {
        state[k] = reactiveLimitsState(qLimitType[k], switchCount[k]);
    }
    jintArray arr = env->NewIntArray(static_cast<jsize>(state.size()));
    if (arr != nullptr) {
        env->SetIntArrayRegion(arr, 0, static_cast<jsize>(state.size()), state.data());
    }
    return arr;
}

// (scenario, entry) pairs whose bus lost EVERY generator this scenario: seeded FROZEN by initReactiveLimitQLimitTypeBatch.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitGeneratorsLost)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        std::vector<int> pairs = jniInts(env, jPairs);
        ctx->reactiveLimits.reactiveLimitGeneratorsLost.installEntries(ctx->nScenarios, ctx->installSlots, ctx->reactiveLimits.nReactiveLimitEntries, static_cast<int>(pairs.size() / 2),
                                           pairs.data(), nullptr, 1);
    }
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitIslanded)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        std::vector<int> pairs = jniInts(env, jPairs);
        ctx->reactiveLimits.reactiveLimitIslanded.installEntries(ctx->nScenarios, ctx->installSlots, ctx->reactiveLimits.nReactiveLimitEntries, static_cast<int>(pairs.size() / 2),
                                           pairs.data(), nullptr, 1);
    }
}

// Re-merged voltage controls (GpuBatchQualification.reMergedControls): per (slot, entry) pair the entry is
// checked by the reactive-limits pass as a LOCAL controller at the given voltage target (pu) — the bus is
// a plain PV bus this scenario, whatever its base packing. Null: none in the installed slots.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitLocalOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jdoubleArray jVt) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    std::vector<int> pairs = jVt != nullptr ? jniInts(env, jPairs) : std::vector<int>();
    std::vector<double> targetV = jniDoubles(env, jVt);
    int nPairs = static_cast<int>(std::min(pairs.size() / 2, targetV.size()));
    ctx->reactiveLimits.reactiveLimitLocalOverride.installEntries(ctx->nScenarios, ctx->installSlots, ctx->reactiveLimits.nReactiveLimitEntries, nPairs, pairs.data(), nullptr, 1);
    ctx->reactiveLimits.reactiveLimitLocalTargetV.installEntries(ctx->nScenarios, ctx->installSlots, ctx->reactiveLimits.nReactiveLimitEntries, nPairs, pairs.data(), targetV.data());
}

// Remote controllers OLF re-merges onto a NEW pilot after a zero-impedance split: per (scenario, entry) the V row
// the reactive-limit switch-back compares against the (per-scenario) group target - the new pilot's, not the base
// pilot's whose row is frozen with its island. Null or empty: none in the installed slots.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitComparedVoltageRowOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jintArray jCmp) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    std::vector<int> pairs = jCmp != nullptr ? jniInts(env, jPairs) : std::vector<int>();
    std::vector<int> rows = jniInts(env, jCmp);
    int nPairs = static_cast<int>(std::min(pairs.size() / 2, rows.size()));
    ctx->reactiveLimits.reactiveLimitComparedVoltageRowOverride.installEntries(ctx->nScenarios, ctx->installSlots, ctx->reactiveLimits.nReactiveLimitEntries, nPairs,
                                                               pairs.data(), rows.data());
}
