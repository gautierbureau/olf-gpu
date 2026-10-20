// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: ReactiveLimitsOuterLoop tables, curves, forced target Q and per-scenario overrides.
#include "jni_common.cuh"
#include "../common/codes/gpu_voltage_control_role.cuh"

// Reactive-capability CURVES at runtime P: install (or clear, with nulls) the per-gen curve channel
// on a batch context — see reactiveCapabilityCurveLimitsBatch. cvIdx stride 5 {reactiveLimitEntry, pRow, ptsOff, npts, dsGenSlot}; cvIn
// stride 5 {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu}; cvPts stride 3
// {p_MW, minQ_MW, maxQ_MW} sorted by p; cvHv (null when no entry is an AC-emulation VSC) stride 2 {hvdc, side}.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveCapabilityCurves)(
        JNIEnv* env, jclass, jlong handle, jintArray jCvIdx, jdoubleArray jCvIn, jdoubleArray jCvPts,
        jintArray jCvHv, jint extrapolate) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->cvHasHvdc = false;
    if (jCvIdx == nullptr || jCvIn == nullptr || jCvPts == nullptr) {
        ctx->ncv = 0;
        return;
    }
    int ncv = env->GetArrayLength(jCvIdx) / 5;   // stride 5: {reactiveLimitEntry, pRow, ptsOff, npts, dsGenSlot}
    if (ncv == 0) {
        ctx->ncv = 0;
        return;
    }
    if (jCvHv != nullptr) {                      // stride 2 per entry: {hvdc, side}, -1 for a plain generator
        jint* hv = env->GetIntArrayElements(jCvHv, nullptr);
        ctx->dCvHv.upload(hv, static_cast<size_t>(ncv) * 2 * sizeof(int));
        env->ReleaseIntArrayElements(jCvHv, hv, JNI_ABORT);
        ctx->cvHasHvdc = true;
    }
    jint* cvIdx = env->GetIntArrayElements(jCvIdx, nullptr);
    jdouble* cvIn = env->GetDoubleArrayElements(jCvIn, nullptr);
    jdouble* cvPts = env->GetDoubleArrayElements(jCvPts, nullptr);
    ctx->dCvIdx.upload(cvIdx, static_cast<size_t>(ncv) * 5 * sizeof(int));
    ctx->dCvIn.upload(cvIn, static_cast<size_t>(ncv) * 5 * sizeof(double));
    ctx->dCvPts.upload(cvPts, static_cast<size_t>(env->GetArrayLength(jCvPts)) * sizeof(double));
    // The curves of each reactive-limit entry, CSR (pack order kept within an entry): reactiveCapabilityCurveLimitsBatch runs
    // one thread per (entry, scenario) and walked EVERY curve to find its own.
    {
        std::vector<int> ptr(static_cast<size_t>(ctx->nReactiveLimitEntries) + 1, 0);
        std::vector<int> list(ncv);
        for (int g = 0; g < ncv; g++) {
            int e = cvIdx[g * 5];
            if (e < 0 || e >= ctx->nReactiveLimitEntries) {
                env->ReleaseIntArrayElements(jCvIdx, cvIdx, JNI_ABORT);
                env->ReleaseDoubleArrayElements(jCvIn, cvIn, JNI_ABORT);
                env->ReleaseDoubleArrayElements(jCvPts, cvPts, JNI_ABORT);
                ctx->ncv = 0;
                env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                              "GPU batch: a curve's reactive-limit entry is outside the pack");
                return;
            }
            ptr[e + 1]++;
        }
        for (int e = 0; e < ctx->nReactiveLimitEntries; e++) ptr[e + 1] += ptr[e];
        std::vector<int> cursor(ptr.begin(), ptr.end() - 1);
        for (int g = 0; g < ncv; g++) list[cursor[cvIdx[g * 5]]++] = g;
        ctx->dCvPtr.upload(ptr.data(), ptr.size() * sizeof(int));
        ctx->dCvList.upload(list.data(), list.size() * sizeof(int));
    }
    ctx->dReactiveLimitMinQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries * sizeof(double));
    ctx->dReactiveLimitMaxQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries * sizeof(double));
    ctx->dReactiveLimitLoadQPerScenario.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries * sizeof(double));
    env->ReleaseIntArrayElements(jCvIdx, cvIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCvIn, cvIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCvPts, cvPts, JNI_ABORT);
    ctx->ncv = ncv;
    ctx->cvExtrapolate = extrapolate;
}

// forceTargetQInReactiveLimits channel (GpuBatchForcedTargetQ): fqIdx stride 7 {qRow, dsGenSlot, ptsOff, npts,
// dsBusIdx, baseQRowMode, reactiveLimitEntry}; fqIn stride 5 {basePpu, rawTargetQPu, baseClampedPu, share, baseRowTargetP}; fqPts
// stride 3 {p_MW, minQ_MW, maxQ_MW} sorted by p. Null clears the channel (a cached context reused by a run
// without it).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchForcedTargetQ)(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jIn, jdoubleArray jPts, jint extrapolate) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->fqHasPre = false;
    if (jIdx == nullptr || jIn == nullptr || jPts == nullptr || env->GetArrayLength(jIdx) == 0) {
        ctx->nfq = 0;
        ctx->nfqLim = 0;
        return;
    }
    int nfq = env->GetArrayLength(jIdx) / 7;
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    int nfqLim = 0;
    for (int e = 0; e < nfq; e++) {
        nfqLim += idx[e * 7 + 6] >= 0 ? 1 : 0;
    }
    jdouble* in = env->GetDoubleArrayElements(jIn, nullptr);
    jdouble* pts = env->GetDoubleArrayElements(jPts, nullptr);
    ctx->dFqIdx.upload(idx, static_cast<size_t>(nfq) * 7 * sizeof(int));
    ctx->dFqIn.upload(in, static_cast<size_t>(nfq) * 5 * sizeof(double));
    ctx->dFqPts.upload(pts, static_cast<size_t>(env->GetArrayLength(jPts)) * sizeof(double));
    ctx->dFqApplied.alloc(static_cast<size_t>(ctx->nScenarios) * nfq * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jIn, in, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jPts, pts, JNI_ABORT);
    ctx->nfq = nfq;
    ctx->nfqLim = nfqLim;
    ctx->fqExtrapolate = extrapolate;
}

// Per-chunk loss pre-distribution moves of the forced generators ([nS * nfq], pu), or null when no scenario of the
// chunk pre-distributes. Set before each solve that carries contingencies.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchForcedTargetQPre)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jPre) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPre == nullptr || ctx->nfq == 0) {
        ctx->fqHasPre = false;
        return;
    }
    jdouble* pre = env->GetDoubleArrayElements(jPre, nullptr);
    ctx->dFqPre.upload(pre, static_cast<size_t>(env->GetArrayLength(jPre)) * sizeof(double));
    env->ReleaseDoubleArrayElements(jPre, pre, JNI_ABORT);
    ctx->fqHasPre = true;
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
        JNIEnv* env, jclass, jlong handle, jbyteArray jRemote, jintArray jCmpVrow,
        jintArray jRemoteMemberReactiveTermRow, jintArray jRemoteMemberReactiveTermElement, jintArray jRemoteMemberReactiveTermSide, jintArray jRemoteMemberReactiveTermKind, jdoubleArray jRemoteMemberReactiveTermWeight,
        jintArray jDistrQRowMember, jintArray jDistrQInjectionMember, jdoubleArray jMKey, jintArray jDistrQMemberReactiveLimitEntry,
        jintArray jMGroupStart, jintArray jMGroupCount, jintArray jMRow, jbyteArray jMHasRow,
        jintArray jMQrGroup, jdoubleArray jMTargetQ, jintArray jMTgtPtr, jintArray jMTgtPeer) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jRemote == nullptr || jCmpVrow == nullptr || env->GetArrayLength(jRemote) == 0) {
        ctx->nReactiveLimitRemoteEntries = 0;
        ctx->nRemoteMemberReactiveTerms = 0;
        ctx->nDistrQDynamicKeys = 0;
        ctx->hasDistrQRelocationGroups = 0;
        return;
    }
    int nr = env->GetArrayLength(jRemote);
    if (nr != ctx->nReactiveLimitEntries || env->GetArrayLength(jCmpVrow) != ctx->nReactiveLimitEntries) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchReactiveLimitRemoteEntries: array length != nReactiveLimitEntries");
        return;
    }
    jbyte* rem = env->GetByteArrayElements(jRemote, nullptr);
    jint* cmp = env->GetIntArrayElements(jCmpVrow, nullptr);
    int count = 0;
    int selfCount = 0;
    for (int i = 0; i < nr; i++) {
        if (rem[i] != GpuVoltageControlRole::LOCAL) {
            count++;
        }
        if (rem[i] == GpuVoltageControlRole::PILOT_SELF) {
            selfCount++;
        }
    }
    ctx->nReactiveLimitPilotSelfEntries = selfCount;
    ctx->dReactiveLimitVoltageControlRole.upload(rem, static_cast<size_t>(nr) * sizeof(char));
    ctx->dReactiveLimitComparedVoltageRow.upload(cmp, static_cast<size_t>(nr) * sizeof(int));
    env->ReleaseByteArrayElements(jRemote, rem, JNI_ABORT);
    env->ReleaseIntArrayElements(jCmpVrow, cmp, JNI_ABORT);
    ctx->nReactiveLimitRemoteEntries = count;
    ctx->nRemoteMemberReactiveTerms = 0;
    if (jRemoteMemberReactiveTermRow != nullptr && env->GetArrayLength(jRemoteMemberReactiveTermRow) > 0) {
        int nq = env->GetArrayLength(jRemoteMemberReactiveTermRow);
        jint* qr = env->GetIntArrayElements(jRemoteMemberReactiveTermRow, nullptr);
        jint* qe = env->GetIntArrayElements(jRemoteMemberReactiveTermElement, nullptr);
        jint* qsd = env->GetIntArrayElements(jRemoteMemberReactiveTermSide, nullptr);
        jint* qk = env->GetIntArrayElements(jRemoteMemberReactiveTermKind, nullptr);
        jdouble* qw = env->GetDoubleArrayElements(jRemoteMemberReactiveTermWeight, nullptr);
        ctx->dRemoteMemberReactiveTermRow.upload(qr, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRemoteMemberReactiveTermElement.upload(qe, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRemoteMemberReactiveTermSide.upload(qsd, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRemoteMemberReactiveTermKind.upload(qk, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRemoteMemberReactiveTermWeight.upload(qw, static_cast<size_t>(nq) * sizeof(double));
        env->ReleaseIntArrayElements(jRemoteMemberReactiveTermRow, qr, JNI_ABORT);
        env->ReleaseIntArrayElements(jRemoteMemberReactiveTermElement, qe, JNI_ABORT);
        env->ReleaseIntArrayElements(jRemoteMemberReactiveTermSide, qsd, JNI_ABORT);
        env->ReleaseIntArrayElements(jRemoteMemberReactiveTermKind, qk, JNI_ABORT);
        env->ReleaseDoubleArrayElements(jRemoteMemberReactiveTermWeight, qw, JNI_ABORT);
        ctx->nRemoteMemberReactiveTerms = nq;
    }
    ctx->nDistrQDynamicKeys = 0;
    ctx->hasDistrQRelocationGroups = 0;
    // Keyed on the MEMBER list, not on the DISTR_Q rows: a SINGLETON remote group has members (and a
    // pilot-Q relocation group, so its pilot's control-loss flip must be armed) but no DISTR_Q row at
    // all — gating on distrQRowMember left hasDistrQRelocationGroups at 0 for it, and the base load flow on the driver
    // kept the pilot's V equation after pinning its only member (2.4e-3 pu on the pilot's Q).
    if (jMKey != nullptr && env->GetArrayLength(jMKey) > 0) {
        int nDistrQTermMembers = jDistrQRowMember != nullptr ? env->GetArrayLength(jDistrQRowMember) : 0;
        int nm = env->GetArrayLength(jMKey);
        jint* km = nDistrQTermMembers > 0 ? env->GetIntArrayElements(jDistrQRowMember, nullptr) : nullptr;
        jint* jm = nDistrQTermMembers > 0 ? env->GetIntArrayElements(jDistrQInjectionMember, nullptr) : nullptr;
        jdouble* mk = env->GetDoubleArrayElements(jMKey, nullptr);
        jint* mre = env->GetIntArrayElements(jDistrQMemberReactiveLimitEntry, nullptr);
        jint* mgs = env->GetIntArrayElements(jMGroupStart, nullptr);
        jint* mgc = env->GetIntArrayElements(jMGroupCount, nullptr);
        jint* mrw = env->GetIntArrayElements(jMRow, nullptr);
        jbyte* mhr = env->GetByteArrayElements(jMHasRow, nullptr);
        ctx->dDistrQMemberRow.upload(mrw, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDistrQMemberHasRow.upload(mhr, static_cast<size_t>(nm) * sizeof(char));
        env->ReleaseIntArrayElements(jMRow, mrw, JNI_ABORT);
        env->ReleaseByteArrayElements(jMHasRow, mhr, JNI_ABORT);
        if (jMTargetQ == nullptr || env->GetArrayLength(jMTargetQ) != nm) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchReactiveLimitRemoteEntries: distrQMemberTargetQ missing or length != members");
            return;
        }
        jdouble* mtq = env->GetDoubleArrayElements(jMTargetQ, nullptr);
        ctx->dDistrQMemberTargetQ.upload(mtq, static_cast<size_t>(nm) * sizeof(double));
        env->ReleaseDoubleArrayElements(jMTargetQ, mtq, JNI_ABORT);
        if (jMTgtPtr == nullptr || env->GetArrayLength(jMTgtPtr) != nm + 1 || jMTgtPeer == nullptr) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchReactiveLimitRemoteEntries: distrQMemberTargetPeerStart missing or length != members + 1");
            return;
        }
        jint* mtp = env->GetIntArrayElements(jMTgtPtr, nullptr);
        ctx->dDistrQMemberTargetPeerStart.upload(mtp, static_cast<size_t>(nm + 1) * sizeof(int));
        env->ReleaseIntArrayElements(jMTgtPtr, mtp, JNI_ABORT);
        int nPeer = env->GetArrayLength(jMTgtPeer);
        if (nPeer > 0) {
            jint* mpe = env->GetIntArrayElements(jMTgtPeer, nullptr);
            ctx->dDistrQMemberTargetPeer.upload(mpe, static_cast<size_t>(nPeer) * sizeof(int));
            env->ReleaseIntArrayElements(jMTgtPeer, mpe, JNI_ABORT);
        }
        ctx->nDistrQMembers = nm;
        if (nDistrQTermMembers > 0) {
            ctx->dDistrQRowMember.upload(km, static_cast<size_t>(nDistrQTermMembers) * sizeof(int));
            ctx->dDistrQInjectionMember.upload(jm, static_cast<size_t>(nDistrQTermMembers) * sizeof(int));
        }
        ctx->dDistrQMemberKey.upload(mk, static_cast<size_t>(nm) * sizeof(double));
        ctx->dDistrQMemberReactiveLimitEntry.upload(mre, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDistrQMemberGroupStart.upload(mgs, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDistrQMemberGroupCount.upload(mgc, static_cast<size_t>(nm) * sizeof(int));
        if (nDistrQTermMembers > 0) {
            env->ReleaseIntArrayElements(jDistrQRowMember, km, JNI_ABORT);
            env->ReleaseIntArrayElements(jDistrQInjectionMember, jm, JNI_ABORT);
        }
        env->ReleaseDoubleArrayElements(jMKey, mk, JNI_ABORT);
        env->ReleaseIntArrayElements(jDistrQMemberReactiveLimitEntry, mre, JNI_ABORT);
        env->ReleaseIntArrayElements(jMGroupStart, mgs, JNI_ABORT);
        env->ReleaseIntArrayElements(jMGroupCount, mgc, JNI_ABORT);
        ctx->nDistrQDynamicKeys = nDistrQTermMembers;
        ctx->hasDistrQRelocationGroups = 0;
        if (jMQrGroup != nullptr && env->GetArrayLength(jMQrGroup) == nm) {
            jint* mqg = env->GetIntArrayElements(jMQrGroup, nullptr);
            std::vector<char> genVc(ctx->st->nQr > 0 ? ctx->st->nQr : 1, 0);
            for (int i = 0; i < nm; i++) {
                if (mqg[i] >= 0) {
                    ctx->hasDistrQRelocationGroups = 1;                  // plan-B4: relocation-linked group present
                    genVc[mqg[i]] = 1;                       // this qr group is a remote gen-VC group
                }
            }
            for (int g = 0; g < ctx->st->nQr; g++) {         // a DEPENDENT pilot's group is as generator-controlled
                if (g < static_cast<int>(ctx->st->hQrDepBase.size()) && ctx->st->hQrDepBase[g] >= 0) {   // as its base
                    genVc[g] = genVc[ctx->st->hQrDepBase[g]];
                }
            }
            ctx->dDistrQMemberRelocationGroup.upload(mqg, static_cast<size_t>(nm) * sizeof(int));
            if (ctx->hasDistrQRelocationGroups) {
                ctx->dQrGenVc.upload(genVc.data(), genVc.size() * sizeof(char));
            }
            env->ReleaseIntArrayElements(jMQrGroup, mqg, JNI_ABORT);
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
    int n = ctx->st->n;
    DeviceBuf out;
    out.alloc(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS * sizeof(double));
    exportReactiveLimitTargetQ<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS, n, ctx->dReactiveLimitVoltageRow.as<int>(),
            ctx->dReactiveLimitReactivePowerRow.p ? ctx->dReactiveLimitReactivePowerRow.as<int>() : nullptr, ctx->dReactiveLimitQLimitType.as<char>(), ctx->dMode.as<int>(),
            ctx->dTarget.as<double>(), ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr,
            ctx->hasDistrQRelocationGroups ? ctx->st->nQr : 0, psEntryPtr(ctx),
            ctx->hasDistrQRelocationGroups && ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr,
            ctx->hasDistrQRelocationGroups && ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr,
            ctx->hasDistrQRelocationGroups && ctx->st->nQr > 0 ? ctx->dQrVeqOff.as<char>() : nullptr,
            ctx->st->dQrDepBase.p ? ctx->st->dQrDepBase.as<int>() : nullptr, out.as<double>());
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

// Re-homed voltage controls (GpuBatchQualification.reHomedControls): per (slot, entry) pair the entry is
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
    std::vector<double> vt(len, 0.0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    jdouble* vts = env->GetDoubleArrayElements(jVt, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nReactiveLimitEntries) {
            mask[(size_t) s * ctx->nReactiveLimitEntries + i] = 1;
            vt[(size_t) s * ctx->nReactiveLimitEntries + i] = vts[k];
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVt, vts, JNI_ABORT);
    ctx->dReactiveLimitLocalOverride.upload(mask.data(), len);
    ctx->dReactiveLimitLocalTargetV.upload(vt.data(), len * sizeof(double));
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
