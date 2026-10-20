// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: ReactiveLimitsOuterLoop tables, curves, forced target Q and per-scenario overrides.
#include "jni_common.cuh"
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
    ctx->reactiveCapabilityCurveHasHvdc = false;
    if (jReactiveCapabilityCurveIndices == nullptr || jReactiveCapabilityCurveParameters == nullptr || jReactiveCapabilityCurvePoints == nullptr) {
        ctx->nReactiveCapabilityCurves = 0;
        return;
    }
    int nReactiveCapabilityCurves = env->GetArrayLength(jReactiveCapabilityCurveIndices) / 5;   // stride 5: {reactiveLimitEntry, pRow, ptsOff, npts, distributedSlackGeneratorIndex}
    if (nReactiveCapabilityCurves == 0) {
        ctx->nReactiveCapabilityCurves = 0;
        return;
    }
    if (jReactiveCapabilityCurveHvdcStation != nullptr) {                      // stride 2 per entry: {hvdc, side}, -1 for a plain generator
        jint* reactiveCapabilityCurveHvdcStation = env->GetIntArrayElements(jReactiveCapabilityCurveHvdcStation, nullptr);
        ctx->dReactiveCapabilityCurveHvdcStation.upload(reactiveCapabilityCurveHvdcStation, static_cast<size_t>(nReactiveCapabilityCurves) * 2 * sizeof(int));
        env->ReleaseIntArrayElements(jReactiveCapabilityCurveHvdcStation, reactiveCapabilityCurveHvdcStation, JNI_ABORT);
        ctx->reactiveCapabilityCurveHasHvdc = true;
    }
    jint* reactiveCapabilityCurveIndices = env->GetIntArrayElements(jReactiveCapabilityCurveIndices, nullptr);
    jdouble* reactiveCapabilityCurveParameters = env->GetDoubleArrayElements(jReactiveCapabilityCurveParameters, nullptr);
    jdouble* reactiveCapabilityCurvePoints = env->GetDoubleArrayElements(jReactiveCapabilityCurvePoints, nullptr);
    ctx->dReactiveCapabilityCurveIndices.upload(reactiveCapabilityCurveIndices, static_cast<size_t>(nReactiveCapabilityCurves) * 5 * sizeof(int));
    ctx->dReactiveCapabilityCurveParameters.upload(reactiveCapabilityCurveParameters, static_cast<size_t>(nReactiveCapabilityCurves) * 5 * sizeof(double));
    ctx->dReactiveCapabilityCurvePoints.upload(reactiveCapabilityCurvePoints, static_cast<size_t>(env->GetArrayLength(jReactiveCapabilityCurvePoints)) * sizeof(double));
    // The curves of each reactive-limit entry, CSR (pack order kept within an entry): reactiveCapabilityCurveLimitsBatch runs
    // one thread per (entry, scenario) and walked EVERY curve to find its own.
    {
        std::vector<int> ptr(static_cast<size_t>(ctx->nReactiveLimitEntries) + 1, 0);
        std::vector<int> list(nReactiveCapabilityCurves);
        for (int g = 0; g < nReactiveCapabilityCurves; g++) {
            int e = reactiveCapabilityCurveIndices[g * 5];
            if (e < 0 || e >= ctx->nReactiveLimitEntries) {
                env->ReleaseIntArrayElements(jReactiveCapabilityCurveIndices, reactiveCapabilityCurveIndices, JNI_ABORT);
                env->ReleaseDoubleArrayElements(jReactiveCapabilityCurveParameters, reactiveCapabilityCurveParameters, JNI_ABORT);
                env->ReleaseDoubleArrayElements(jReactiveCapabilityCurvePoints, reactiveCapabilityCurvePoints, JNI_ABORT);
                ctx->nReactiveCapabilityCurves = 0;
                env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                              "GPU batch: a curve's reactive-limit entry is outside the pack");
                return;
            }
            ptr[e + 1]++;
        }
        for (int e = 0; e < ctx->nReactiveLimitEntries; e++) ptr[e + 1] += ptr[e];
        std::vector<int> cursor(ptr.begin(), ptr.end() - 1);
        for (int g = 0; g < nReactiveCapabilityCurves; g++) list[cursor[reactiveCapabilityCurveIndices[g * 5]]++] = g;
        ctx->dReactiveLimitEntryCurveStart.upload(ptr.data(), ptr.size() * sizeof(int));
        ctx->dReactiveLimitEntryCurves.upload(list.data(), list.size() * sizeof(int));
    }
    ctx->dReactiveLimitMinQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries * sizeof(double));
    ctx->dReactiveLimitMaxQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries * sizeof(double));
    ctx->dReactiveLimitLoadQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries * sizeof(double));
    env->ReleaseIntArrayElements(jReactiveCapabilityCurveIndices, reactiveCapabilityCurveIndices, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jReactiveCapabilityCurveParameters, reactiveCapabilityCurveParameters, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jReactiveCapabilityCurvePoints, reactiveCapabilityCurvePoints, JNI_ABORT);
    ctx->nReactiveCapabilityCurves = nReactiveCapabilityCurves;
    ctx->reactiveCapabilityCurveExtrapolate = extrapolate;
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
    ctx->hasForcedTargetQPreDistribution = false;
    if (jIdx == nullptr || jIn == nullptr || jPts == nullptr || env->GetArrayLength(jIdx) == 0) {
        ctx->nForcedTargetQ = 0;
        ctx->nForcedTargetQLimitEntries = 0;
        return;
    }
    int nForcedTargetQ = env->GetArrayLength(jIdx) / 7;
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    int nForcedTargetQLimitEntries = 0;
    for (int e = 0; e < nForcedTargetQ; e++) {
        nForcedTargetQLimitEntries += idx[e * 7 + 6] >= 0 ? 1 : 0;
    }
    jdouble* in = env->GetDoubleArrayElements(jIn, nullptr);
    jdouble* pts = env->GetDoubleArrayElements(jPts, nullptr);
    ctx->dForcedTargetQIndices.upload(idx, static_cast<size_t>(nForcedTargetQ) * 7 * sizeof(int));
    ctx->dForcedTargetQParameters.upload(in, static_cast<size_t>(nForcedTargetQ) * 5 * sizeof(double));
    ctx->dForcedTargetQPoints.upload(pts, static_cast<size_t>(env->GetArrayLength(jPts)) * sizeof(double));
    ctx->dForcedTargetQApplied.alloc(static_cast<size_t>(ctx->nScenarios) * nForcedTargetQ * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jIn, in, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jPts, pts, JNI_ABORT);
    ctx->nForcedTargetQ = nForcedTargetQ;
    ctx->nForcedTargetQLimitEntries = nForcedTargetQLimitEntries;
    ctx->forcedTargetQExtrapolate = extrapolate;
}

// Per-chunk loss pre-distribution moves of the forced generators ([nS * nForcedTargetQ], pu), or null when no scenario of the
// chunk pre-distributes. Set before each solve that carries contingencies.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchForcedTargetQPre)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jPre) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPre == nullptr || ctx->nForcedTargetQ == 0) {
        ctx->hasForcedTargetQPreDistribution = false;
        return;
    }
    jdouble* pre = env->GetDoubleArrayElements(jPre, nullptr);
    ctx->dForcedTargetQPreDistribution.upload(pre, static_cast<size_t>(env->GetArrayLength(jPre)) * sizeof(double));
    env->ReleaseDoubleArrayElements(jPre, pre, JNI_ABORT);
    ctx->hasForcedTargetQPreDistribution = true;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitStrongestPvKeys)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jNomV, jintArray jPRow, jdoubleArray jStaticP, jintArray jIdRank) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hasReactiveLimitStrongestPvKeys = 0;
    if (ctx->nReactiveLimitEntries == 0 || jNomV == nullptr || jPRow == nullptr || jStaticP == nullptr || jIdRank == nullptr) {
        return;
    }
    if (env->GetArrayLength(jNomV) != ctx->nReactiveLimitEntries || env->GetArrayLength(jPRow) != ctx->nReactiveLimitEntries
            || env->GetArrayLength(jStaticP) != ctx->nReactiveLimitEntries || env->GetArrayLength(jIdRank) != ctx->nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchReactiveLimitStrongestPvKeys: array length != nReactiveLimitEntries");
        return;
    }
    jdouble* nv = env->GetDoubleArrayElements(jNomV, nullptr);
    jint* pr = env->GetIntArrayElements(jPRow, nullptr);
    jdouble* sp = env->GetDoubleArrayElements(jStaticP, nullptr);
    jint* ir = env->GetIntArrayElements(jIdRank, nullptr);
    ctx->dReactiveLimitNominalV.upload(nv, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(double));
    ctx->dReactiveLimitActivePowerRow.upload(pr, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(int));
    ctx->dReactiveLimitTargetP.upload(sp, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(double));
    ctx->dReactiveLimitIdRank.upload(ir, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(int));
    env->ReleaseDoubleArrayElements(jNomV, nv, JNI_ABORT);
    env->ReleaseIntArrayElements(jPRow, pr, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jStaticP, sp, JNI_ABORT);
    env->ReleaseIntArrayElements(jIdRank, ir, JNI_ABORT);
    ctx->hasReactiveLimitStrongestPvKeys = 1;
}

// voltageRemoteControlRobustMode (OLF's default): install the robust-mode channel on a batch context —
// the per-entry initial generation target Q that a realistic-V freeze pins at, and the two thresholds
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
    ctx->reactiveLimitsRobustMode = 0;
    if (ctx->nReactiveLimitEntries == 0 || robust == 0 || jInitTgtQ == nullptr) {
        return;
    }
    if (env->GetArrayLength(jInitTgtQ) != ctx->nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchReactiveLimitsRobustMode: array length != nReactiveLimitEntries");
        return;
    }
    jdouble* q0 = env->GetDoubleArrayElements(jInitTgtQ, nullptr);
    ctx->dReactiveLimitInitialTargetQ.upload(q0, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(double));
    env->ReleaseDoubleArrayElements(jInitTgtQ, q0, JNI_ABORT);
    ctx->reactiveLimitsRobustMinRealisticVoltage = vLo;
    ctx->reactiveLimitsRobustMaxRealisticVoltage = vHi;
    ctx->reactiveLimitsRobustMode = 1;
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
    ctx->realisticCheck = 0;
    if (enabled == 0 || jRowChecked == nullptr) {
        return;
    }
    if (env->GetArrayLength(jRowChecked) != ctx->n) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRealisticCheck: array length != n");
        return;
    }
    jbyte* rc = env->GetByteArrayElements(jRowChecked, nullptr);
    ctx->dRealisticRow.upload(rc, static_cast<size_t>(ctx->n) * sizeof(char));
    env->ReleaseByteArrayElements(jRowChecked, rc, JNI_ABORT);
    ctx->realisticMinV = minV;
    ctx->realisticMaxV = maxV;
    ctx->realisticFixingLoop = fixingLoop;
    ctx->realisticCheck = 1;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitSvcSusceptanceLimits)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jBmin, jdoubleArray jBmax, jdoubleArray jV2Base) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nReactiveLimitEntries == 0) {
        return;
    }
    if (env->GetArrayLength(jBmin) != ctx->nReactiveLimitEntries || env->GetArrayLength(jBmax) != ctx->nReactiveLimitEntries
            || env->GetArrayLength(jV2Base) != ctx->nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchReactiveLimitSvcSusceptanceLimits: array length != nReactiveLimitEntries");
        return;
    }
    jdouble* bmin = env->GetDoubleArrayElements(jBmin, nullptr);
    jdouble* bmax = env->GetDoubleArrayElements(jBmax, nullptr);
    jdouble* v2b = env->GetDoubleArrayElements(jV2Base, nullptr);
    ctx->dReactiveLimitSvcMinB.upload(bmin, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(double));
    ctx->dReactiveLimitSvcMaxB.upload(bmax, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(double));
    ctx->dReactiveLimitBaseVSquared.upload(v2b, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(double));
    env->ReleaseDoubleArrayElements(jBmin, bmin, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jBmax, bmax, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jV2Base, v2b, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitBaseQLimitType)(
        JNIEnv* env, jclass, jlong handle, jbyteArray jKinds) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKinds == nullptr || env->GetArrayLength(jKinds) == 0) {
        ctx->hasReactiveLimitBaseQLimitType = 0;
        return;
    }
    if (env->GetArrayLength(jKinds) != ctx->nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchReactiveLimitBaseQLimitType: array length != nReactiveLimitEntries");
        return;
    }
    jbyte* kinds = env->GetByteArrayElements(jKinds, nullptr);
    ctx->dReactiveLimitBaseQLimitType.alloc(static_cast<size_t>(ctx->nReactiveLimitEntries));
    CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitBaseQLimitType.p, kinds, static_cast<size_t>(ctx->nReactiveLimitEntries), cudaMemcpyHostToDevice));
    env->ReleaseByteArrayElements(jKinds, kinds, JNI_ABORT);
    ctx->hasReactiveLimitBaseQLimitType = 1;
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
        ctx->nReactiveLimitRemoteEntries = 0;
        ctx->nRemoteMemberReactiveTerms = 0;
        ctx->nDistrQDynamicKeys = 0;
        ctx->hasDistrQRelocationGroups = 0;
        return;
    }
    int nReactiveLimitEntries = env->GetArrayLength(jReactiveLimitVoltageControlRole);
    if (nReactiveLimitEntries != ctx->nReactiveLimitEntries || env->GetArrayLength(jReactiveLimitComparedVoltageRow) != ctx->nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchReactiveLimitRemoteEntries: array length != nReactiveLimitEntries");
        return;
    }
    jbyte* voltageControlRole = env->GetByteArrayElements(jReactiveLimitVoltageControlRole, nullptr);
    jint* comparedVoltageRow = env->GetIntArrayElements(jReactiveLimitComparedVoltageRow, nullptr);
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
    ctx->nReactiveLimitPilotSelfEntries = selfCount;
    ctx->dReactiveLimitVoltageControlRole.upload(voltageControlRole, static_cast<size_t>(nReactiveLimitEntries) * sizeof(char));
    ctx->dReactiveLimitComparedVoltageRow.upload(comparedVoltageRow, static_cast<size_t>(nReactiveLimitEntries) * sizeof(int));
    env->ReleaseByteArrayElements(jReactiveLimitVoltageControlRole, voltageControlRole, JNI_ABORT);
    env->ReleaseIntArrayElements(jReactiveLimitComparedVoltageRow, comparedVoltageRow, JNI_ABORT);
    ctx->nReactiveLimitRemoteEntries = count;
    ctx->nRemoteMemberReactiveTerms = 0;
    if (jRemoteMemberReactiveTermRow != nullptr && env->GetArrayLength(jRemoteMemberReactiveTermRow) > 0) {
        int nRemoteMemberReactiveTerms = env->GetArrayLength(jRemoteMemberReactiveTermRow);
        jint* remoteMemberReactiveTermRow = env->GetIntArrayElements(jRemoteMemberReactiveTermRow, nullptr);
        jint* remoteMemberReactiveTermElement = env->GetIntArrayElements(jRemoteMemberReactiveTermElement, nullptr);
        jint* remoteMemberReactiveTermSide = env->GetIntArrayElements(jRemoteMemberReactiveTermSide, nullptr);
        jint* remoteMemberReactiveTermKind = env->GetIntArrayElements(jRemoteMemberReactiveTermKind, nullptr);
        jdouble* remoteMemberReactiveTermWeight = env->GetDoubleArrayElements(jRemoteMemberReactiveTermWeight, nullptr);
        ctx->dRemoteMemberReactiveTermRow.upload(remoteMemberReactiveTermRow, static_cast<size_t>(nRemoteMemberReactiveTerms) * sizeof(int));
        ctx->dRemoteMemberReactiveTermElement.upload(remoteMemberReactiveTermElement, static_cast<size_t>(nRemoteMemberReactiveTerms) * sizeof(int));
        ctx->dRemoteMemberReactiveTermSide.upload(remoteMemberReactiveTermSide, static_cast<size_t>(nRemoteMemberReactiveTerms) * sizeof(int));
        ctx->dRemoteMemberReactiveTermKind.upload(remoteMemberReactiveTermKind, static_cast<size_t>(nRemoteMemberReactiveTerms) * sizeof(int));
        ctx->dRemoteMemberReactiveTermWeight.upload(remoteMemberReactiveTermWeight, static_cast<size_t>(nRemoteMemberReactiveTerms) * sizeof(double));
        env->ReleaseIntArrayElements(jRemoteMemberReactiveTermRow, remoteMemberReactiveTermRow, JNI_ABORT);
        env->ReleaseIntArrayElements(jRemoteMemberReactiveTermElement, remoteMemberReactiveTermElement, JNI_ABORT);
        env->ReleaseIntArrayElements(jRemoteMemberReactiveTermSide, remoteMemberReactiveTermSide, JNI_ABORT);
        env->ReleaseIntArrayElements(jRemoteMemberReactiveTermKind, remoteMemberReactiveTermKind, JNI_ABORT);
        env->ReleaseDoubleArrayElements(jRemoteMemberReactiveTermWeight, remoteMemberReactiveTermWeight, JNI_ABORT);
        ctx->nRemoteMemberReactiveTerms = nRemoteMemberReactiveTerms;
    }
    ctx->nDistrQDynamicKeys = 0;
    ctx->hasDistrQRelocationGroups = 0;
    // Keyed on the MEMBER list, not on the DISTR_Q rows: a SINGLETON remote group has members (and a
    // pilot-Q relocation group, so its pilot's control-loss flip must be armed) but no DISTR_Q row at
    // all — gating on distrQRowMember left hasDistrQRelocationGroups at 0 for it, and the base load flow on the driver
    // kept the pilot's V equation after pinning its only member (2.4e-3 pu on the pilot's Q).
    if (jDistrQMemberKey != nullptr && env->GetArrayLength(jDistrQMemberKey) > 0) {
        int nDistrQTermMembers = jDistrQRowMember != nullptr ? env->GetArrayLength(jDistrQRowMember) : 0;
        int nDistrQMembers = env->GetArrayLength(jDistrQMemberKey);
        jint* distrQRowMember = nDistrQTermMembers > 0 ? env->GetIntArrayElements(jDistrQRowMember, nullptr) : nullptr;
        jint* distrQInjectionMember = nDistrQTermMembers > 0 ? env->GetIntArrayElements(jDistrQInjectionMember, nullptr) : nullptr;
        jdouble* distrQMemberKey = env->GetDoubleArrayElements(jDistrQMemberKey, nullptr);
        jint* distrQMemberReactiveLimitEntry = env->GetIntArrayElements(jDistrQMemberReactiveLimitEntry, nullptr);
        jint* distrQMemberGroupStart = env->GetIntArrayElements(jDistrQMemberGroupStart, nullptr);
        jint* distrQMemberGroupCount = env->GetIntArrayElements(jDistrQMemberGroupCount, nullptr);
        jint* distrQMemberRow = env->GetIntArrayElements(jDistrQMemberRow, nullptr);
        jbyte* distrQMemberHasRow = env->GetByteArrayElements(jDistrQMemberHasRow, nullptr);
        ctx->dDistrQMemberRow.upload(distrQMemberRow, static_cast<size_t>(nDistrQMembers) * sizeof(int));
        ctx->dDistrQMemberHasRow.upload(distrQMemberHasRow, static_cast<size_t>(nDistrQMembers) * sizeof(char));
        env->ReleaseIntArrayElements(jDistrQMemberRow, distrQMemberRow, JNI_ABORT);
        env->ReleaseByteArrayElements(jDistrQMemberHasRow, distrQMemberHasRow, JNI_ABORT);
        if (jDistrQMemberTargetQ == nullptr || env->GetArrayLength(jDistrQMemberTargetQ) != nDistrQMembers) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchReactiveLimitRemoteEntries: distrQMemberTargetQ missing or length != members");
            return;
        }
        jdouble* distrQMemberTargetQ = env->GetDoubleArrayElements(jDistrQMemberTargetQ, nullptr);
        ctx->dDistrQMemberTargetQ.upload(distrQMemberTargetQ, static_cast<size_t>(nDistrQMembers) * sizeof(double));
        env->ReleaseDoubleArrayElements(jDistrQMemberTargetQ, distrQMemberTargetQ, JNI_ABORT);
        if (jDistrQMemberTargetPeerStart == nullptr || env->GetArrayLength(jDistrQMemberTargetPeerStart) != nDistrQMembers + 1 || jDistrQMemberTargetPeer == nullptr) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchReactiveLimitRemoteEntries: distrQMemberTargetPeerStart missing or length != members + 1");
            return;
        }
        jint* distrQMemberTargetPeerStart = env->GetIntArrayElements(jDistrQMemberTargetPeerStart, nullptr);
        ctx->dDistrQMemberTargetPeerStart.upload(distrQMemberTargetPeerStart, static_cast<size_t>(nDistrQMembers + 1) * sizeof(int));
        env->ReleaseIntArrayElements(jDistrQMemberTargetPeerStart, distrQMemberTargetPeerStart, JNI_ABORT);
        int nDistrQMemberTargetPeers = env->GetArrayLength(jDistrQMemberTargetPeer);
        if (nDistrQMemberTargetPeers > 0) {
            jint* distrQMemberTargetPeer = env->GetIntArrayElements(jDistrQMemberTargetPeer, nullptr);
            ctx->dDistrQMemberTargetPeer.upload(distrQMemberTargetPeer, static_cast<size_t>(nDistrQMemberTargetPeers) * sizeof(int));
            env->ReleaseIntArrayElements(jDistrQMemberTargetPeer, distrQMemberTargetPeer, JNI_ABORT);
        }
        ctx->nDistrQMembers = nDistrQMembers;
        if (nDistrQTermMembers > 0) {
            ctx->dDistrQRowMember.upload(distrQRowMember, static_cast<size_t>(nDistrQTermMembers) * sizeof(int));
            ctx->dDistrQInjectionMember.upload(distrQInjectionMember, static_cast<size_t>(nDistrQTermMembers) * sizeof(int));
        }
        ctx->dDistrQMemberKey.upload(distrQMemberKey, static_cast<size_t>(nDistrQMembers) * sizeof(double));
        ctx->dDistrQMemberReactiveLimitEntry.upload(distrQMemberReactiveLimitEntry, static_cast<size_t>(nDistrQMembers) * sizeof(int));
        ctx->dDistrQMemberGroupStart.upload(distrQMemberGroupStart, static_cast<size_t>(nDistrQMembers) * sizeof(int));
        ctx->dDistrQMemberGroupCount.upload(distrQMemberGroupCount, static_cast<size_t>(nDistrQMembers) * sizeof(int));
        if (nDistrQTermMembers > 0) {
            env->ReleaseIntArrayElements(jDistrQRowMember, distrQRowMember, JNI_ABORT);
            env->ReleaseIntArrayElements(jDistrQInjectionMember, distrQInjectionMember, JNI_ABORT);
        }
        env->ReleaseDoubleArrayElements(jDistrQMemberKey, distrQMemberKey, JNI_ABORT);
        env->ReleaseIntArrayElements(jDistrQMemberReactiveLimitEntry, distrQMemberReactiveLimitEntry, JNI_ABORT);
        env->ReleaseIntArrayElements(jDistrQMemberGroupStart, distrQMemberGroupStart, JNI_ABORT);
        env->ReleaseIntArrayElements(jDistrQMemberGroupCount, distrQMemberGroupCount, JNI_ABORT);
        ctx->nDistrQDynamicKeys = nDistrQTermMembers;
        ctx->hasDistrQRelocationGroups = 0;
        if (jDistrQMemberRelocationGroup != nullptr && env->GetArrayLength(jDistrQMemberRelocationGroup) == nDistrQMembers) {
            jint* distrQMemberRelocationGroup = env->GetIntArrayElements(jDistrQMemberRelocationGroup, nullptr);
            std::vector<char> isGeneratorVoltageControl(ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->nReactiveRelocationGroups : 1, 0);
            for (int i = 0; i < nDistrQMembers; i++) {
                if (distrQMemberRelocationGroup[i] >= 0) {
                    ctx->hasDistrQRelocationGroups = 1;                  // plan-B4: relocation-linked group present
                    isGeneratorVoltageControl[distrQMemberRelocationGroup[i]] = 1;   // this relocation group is a remote generator voltage control
                }
            }
            for (int g = 0; g < ctx->structure->nReactiveRelocationGroups; g++) {         // a DEPENDENT pilot's group is as generator-controlled
                if (g < static_cast<int>(ctx->structure->hReactiveRelocationDependentBaseGroup.size()) && ctx->structure->hReactiveRelocationDependentBaseGroup[g] >= 0) {   // as its base
                    isGeneratorVoltageControl[g] = isGeneratorVoltageControl[ctx->structure->hReactiveRelocationDependentBaseGroup[g]];
                }
            }
            ctx->dDistrQMemberRelocationGroup.upload(distrQMemberRelocationGroup, static_cast<size_t>(nDistrQMembers) * sizeof(int));
            if (ctx->hasDistrQRelocationGroups) {
                ctx->dReactiveRelocationIsGeneratorVoltageControl.upload(isGeneratorVoltageControl.data(), isGeneratorVoltageControl.size() * sizeof(char));
            }
            env->ReleaseIntArrayElements(jDistrQMemberRelocationGroup, distrQMemberRelocationGroup, JNI_ABORT);
        }
  }
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitSeedTargetQ)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jTargets) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jTargets == nullptr || env->GetArrayLength(jTargets) == 0) {
        ctx->hasReactiveLimitSeedTargetQ = false;
        return;
    }
    int len = env->GetArrayLength(jTargets);
    jdouble* t = env->GetDoubleArrayElements(jTargets, nullptr);
    ctx->dReactiveLimitSeedTargetQ.upload(t, static_cast<size_t>(len) * sizeof(double));
    env->ReleaseDoubleArrayElements(jTargets, t, JNI_ABORT);
    ctx->hasReactiveLimitSeedTargetQ = true;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, readBatchReactiveLimitTargetQ)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nReactiveLimitEntries <= 0) {
        return nullptr;
    }
    int nS = ctx->nScenarios;
    int n = ctx->structure->n;
    DeviceBuf out;
    out.alloc(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS * sizeof(double));
    exportReactiveLimitTargetQ<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS, n, ctx->dReactiveLimitVoltageRow.as<int>(),
            ctx->dReactiveLimitReactivePowerRow.p ? ctx->dReactiveLimitReactivePowerRow.as<int>() : nullptr, ctx->dReactiveLimitQLimitType.as<char>(), ctx->dMode.as<int>(),
            ctx->dTarget.as<double>(), ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr,
            ctx->hasDistrQRelocationGroups ? ctx->structure->nReactiveRelocationGroups : 0, pilotSelfReactiveLimitEntryOrNull(ctx),
            ctx->hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->dReactiveRelocationGroupStart.as<int>() : nullptr,
            ctx->hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0 ? ctx->structure->dReactiveRelocationCandidateRows.as<int>() : nullptr,
            ctx->hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0 ? ctx->dReactiveRelocationPilotTargetVDeactivated.as<char>() : nullptr,
            ctx->structure->dReactiveRelocationDependentBaseGroup.p ? ctx->structure->dReactiveRelocationDependentBaseGroup.as<int>() : nullptr, out.as<double>());
    std::vector<double> host(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(host.data(), out.p, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    jdoubleArray arr = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(arr, 0, static_cast<jsize>(host.size()), host.data());
    return arr;
}

/** Un-pin reactive-limit entries at a driver pause (SecondaryVoltageControl's re-enable step):
 *  {@code pairs} is (scenario, RL entry) flattened, {@code newTarget} the voltage target each row goes
 *  back to. No-op for an entry that is not pinned. */
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitPqPvSwitches)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jdoubleArray jTargets) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nReactiveLimitEntries <= 0 || jPairs == nullptr || jTargets == nullptr) {
        return;
    }
    int nPairs = env->GetArrayLength(jTargets);
    if (nPairs == 0 || env->GetArrayLength(jPairs) != 2 * nPairs) {
        return;
    }
    std::vector<int> pairs(2 * nPairs);
    std::vector<double> targets(nPairs);
    env->GetIntArrayRegion(jPairs, 0, 2 * nPairs, pairs.data());
    env->GetDoubleArrayRegion(jTargets, 0, nPairs, targets.data());
    DeviceBuf dPairs;
    DeviceBuf dTargets;
    dPairs.upload(pairs.data(), pairs.size() * sizeof(int));
    dTargets.upload(targets.data(), targets.size() * sizeof(double));
    reactiveLimitsPqPvSwitchBatch<<<grid(nPairs), BLOCK, 0, ctx->stream>>>(nPairs, ctx->nScenarios, ctx->n, ctx->nReactiveLimitEntries,
            dPairs.as<int>(), dTargets.as<double>(), ctx->dReactiveLimitVoltageRow.as<int>(), ctx->dReactiveLimitQLimitType.as<char>(),
            ctx->dReactiveLimitSavedRowMode.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

// Reactive-limits controllers STRANDED in another synchronous component: install (or clear, with
// null) the per-scenario mask as {slot, controller} pairs.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitPqPvSwitchForbidden)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nReactiveLimitEntries == 0) {
        ctx->nReactiveLimitPqPvSwitchForbidden = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries;
    std::vector<char> mask(len, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nReactiveLimitEntries) {
            mask[(size_t) s * ctx->nReactiveLimitEntries + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dReactiveLimitPqPvSwitchForbidden.upload(mask.data(), len);
    std::vector<char> zero(ctx->nScenarios, 0);
    ctx->dReactiveLimitPqPvSwitchRejected.upload(zero.data(), ctx->nScenarios);
    ctx->nReactiveLimitPqPvSwitchForbidden = nPairs;
}

// (scenario, entry) pairs whose bus lost EVERY generator this scenario: seeded FROZEN by initReactiveLimitQLimitTypeBatch.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitGeneratorsLost)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nReactiveLimitEntries == 0) {
        ctx->nReactiveLimitGeneratorsLost = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries;
    std::vector<char> mask(len, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nReactiveLimitEntries) {
            mask[(size_t) s * ctx->nReactiveLimitEntries + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dReactiveLimitGeneratorsLost.upload(mask.data(), len);
    ctx->nReactiveLimitGeneratorsLost = nPairs;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitIslanded)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nReactiveLimitEntries == 0) {
        ctx->nReactiveLimitIslanded = 0;
        return;
    }
    size_t maskLen = static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries;
    std::vector<char> mask(maskLen, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nReactiveLimitEntries) {
            mask[(size_t) s * ctx->nReactiveLimitEntries + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dReactiveLimitIslanded.upload(mask.data(), maskLen);
    ctx->nReactiveLimitIslanded = nPairs;
}

// Re-merged voltage controls (GpuBatchQualification.reMergedControls): per (slot, entry) pair the entry is
// checked by the reactive-limits pass as a LOCAL controller at the given voltage target (pu) — the bus is
// a plain PV bus this scenario, whatever its base packing. Null clears the context's list.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitLocalOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jdoubleArray jVt) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || jVt == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nReactiveLimitEntries == 0) {
        ctx->nReactiveLimitLocalOverrides = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries;
    std::vector<char> mask(len, 0);
    std::vector<double> targetV(len, 0.0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    jdouble* vts = env->GetDoubleArrayElements(jVt, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nReactiveLimitEntries) {
            mask[(size_t) s * ctx->nReactiveLimitEntries + i] = 1;
            targetV[(size_t) s * ctx->nReactiveLimitEntries + i] = vts[k];
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVt, vts, JNI_ABORT);
    ctx->dReactiveLimitLocalOverride.upload(mask.data(), len);
    ctx->dReactiveLimitLocalTargetV.upload(targetV.data(), len * sizeof(double));
    ctx->nReactiveLimitLocalOverrides = nPairs;
}

// Remote controllers OLF re-merges onto a NEW pilot after a zero-impedance split: per (scenario, entry) the V row
// the reactive-limit switch-back compares against the (per-scenario) group target - the new pilot's, not the base
// pilot's whose row is frozen with its island. Null or empty clears the context's list.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitComparedVoltageRowOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jintArray jCmp) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || jCmp == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nReactiveLimitEntries == 0) {
        ctx->nReactiveLimitComparedVoltageRowOverrides = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries;
    std::vector<int> cmp(len, -1);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    jint* rows = env->GetIntArrayElements(jCmp, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nReactiveLimitEntries) {
            cmp[(size_t) s * ctx->nReactiveLimitEntries + i] = rows[k];
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    env->ReleaseIntArrayElements(jCmp, rows, JNI_ABORT);
    ctx->dReactiveLimitComparedVoltageRowOverride.upload(cmp.data(), len * sizeof(int));
    ctx->nReactiveLimitComparedVoltageRowOverrides = nPairs;
}
