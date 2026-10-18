// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Remote generator voltage control (DISTR_Q) and its pilot: per-scenario members, keys and targets.
#include "jni_common.cuh"

// DISTR_Q weight overrides: install (or clear, with nulls) the per-scenario qPct renormalization
// pairs {s, contrib} + new weights on a batch context (gen outage at a remote-VC controller bus).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDqInactiveMembers)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotMember) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotMember == nullptr || env->GetArrayLength(jSlotMember) == 0) {
        ctx->nDqInact = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotMember);
    jint* sm = env->GetIntArrayElements(jSlotMember, nullptr);
    ctx->dDqInact.upload(sm, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotMember, sm, JNI_ABORT);
    ctx->nDqInact = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDqSubgroups)(
        JNIEnv* env, jclass, jlong handle, jbyteArray jDense, jintArray jQrMemberStart) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jDense == nullptr || env->GetArrayLength(jDense) == 0) {
        ctx->hasDqSub = false;
        return;
    }
    jbyte* d = env->GetByteArrayElements(jDense, nullptr);
    ctx->dDqSub.upload(d, static_cast<size_t>(env->GetArrayLength(jDense)));
    env->ReleaseByteArrayElements(jDense, d, JNI_ABORT);
    jint* q = env->GetIntArrayElements(jQrMemberStart, nullptr);
    ctx->dQrMemberStart.upload(q, static_cast<size_t>(env->GetArrayLength(jQrMemberStart)) * sizeof(int));
    env->ReleaseIntArrayElements(jQrMemberStart, q, JNI_ABORT);
    ctx->hasDqSub = true;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDqWeightOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jW) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jW == nullptr || env->GetArrayLength(jW) == 0) {
        ctx->nDqOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jW);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* w = env->GetDoubleArrayElements(jW, nullptr);
    ctx->dDqOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDqOvW.upload(w, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jW, w, JNI_ABORT);
    ctx->nDqOv = nOv;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDqKeyOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jKey) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jKey == nullptr || env->GetArrayLength(jKey) == 0) {
        ctx->nDqKeyOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jKey);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* key = env->GetDoubleArrayElements(jKey, nullptr);
    ctx->dDqKeyOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDqKeyOvKey.upload(key, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jKey, key, JNI_ABORT);
    ctx->nDqKeyOv = nOv;
}

// Per-scenario member targetQ overrides for the DISTR_Q target kernel: {key, value} where key is
// slot * nDqMembers + memberIndex. Set when an outaged bus is a packed group member whose own targetQ
// moved (a reactive LOAD loss). Nulls clear.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDqMemberTargetQOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKeys == nullptr || env->GetArrayLength(jKeys) == 0) {
        ctx->nDqTqOv = 0;
        return;
    }
    int ne = env->GetArrayLength(jKeys);
    jint* keys = env->GetIntArrayElements(jKeys, nullptr);
    jdouble* vals = env->GetDoubleArrayElements(jVals, nullptr);
    ctx->dDqTqOvKey.upload(keys, static_cast<size_t>(ne) * sizeof(int));
    ctx->dDqTqOvVal.upload(vals, static_cast<size_t>(ne) * sizeof(double));
    ctx->nDqTqOv = ne;
    env->ReleaseIntArrayElements(jKeys, keys, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVals, vals, JNI_ABORT);
}

// Per-scenario moves of a pilot's captured reactive target (applyPilotQDeltasBatch): {key, delta} with key
// slot * nQr + group, one entry per key. Nulls clear.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchPilotQDeltas)(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKeys == nullptr || env->GetArrayLength(jKeys) == 0) {
        ctx->nQrPqDelta = 0;
        return;
    }
    int ne = env->GetArrayLength(jKeys);
    jint* keys = env->GetIntArrayElements(jKeys, nullptr);
    jdouble* vals = env->GetDoubleArrayElements(jVals, nullptr);
    ctx->dQrPqDeltaKey.upload(keys, static_cast<size_t>(ne) * sizeof(int));
    ctx->dQrPqDeltaVal.upload(vals, static_cast<size_t>(ne) * sizeof(double));
    ctx->nQrPqDelta = ne;
    env->ReleaseIntArrayElements(jKeys, keys, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVals, vals, JNI_ABORT);
}

// Pilot-row flip entries (see pilotRowAllPinnedBatch): one per (scenario, pilot) where the outage
// made the pilot stop being one of its own controllers. slots/rows/qTargets are per entry; rlCounts
// says how many surviving-controller RL entries each one owns and rlPool concatenates them in order.
// Nulls clear. Flip state (flipped/savedVTarget) is allocated zeroed here, so each solve starts with
// every pilot row unflipped.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchPilotAllPinnedFlip)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlots, jintArray jRows, jdoubleArray jQTargets,
        jintArray jRlCounts, jintArray jRlPool) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlots == nullptr || env->GetArrayLength(jSlots) == 0) {
        ctx->nPilotFlip = 0;
        return;
    }
    int ne = env->GetArrayLength(jSlots);
    jint* slots = env->GetIntArrayElements(jSlots, nullptr);
    jint* rows = env->GetIntArrayElements(jRows, nullptr);
    jdouble* qt = env->GetDoubleArrayElements(jQTargets, nullptr);
    jint* cnt = env->GetIntArrayElements(jRlCounts, nullptr);
    jint* pool = env->GetIntArrayElements(jRlPool, nullptr);
    int poolLen = env->GetArrayLength(jRlPool);
    std::vector<int> off(ne);
    int acc = 0;
    for (int i = 0; i < ne; i++) {
        off[i] = acc;
        acc += cnt[i];
    }
    ctx->dPfSlot.upload(slots, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfRow.upload(rows, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfQTarget.upload(qt, static_cast<size_t>(ne) * sizeof(double));
    ctx->dPfCount.upload(cnt, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfOff.upload(off.data(), static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfRlPool.upload(pool, static_cast<size_t>(poolLen > 0 ? poolLen : 1) * sizeof(int));
    std::vector<char> zeroC(ne, 0);
    std::vector<double> zeroD(ne, 0.0);
    ctx->dPfFlipped.upload(zeroC.data(), static_cast<size_t>(ne) * sizeof(char));
    ctx->dPfSavedVT.upload(zeroD.data(), static_cast<size_t>(ne) * sizeof(double));
    ctx->nPilotFlip = ne;
    env->ReleaseIntArrayElements(jSlots, slots, JNI_ABORT);
    env->ReleaseIntArrayElements(jRows, rows, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jQTargets, qt, JNI_ABORT);
    env->ReleaseIntArrayElements(jRlCounts, cnt, JNI_ABORT);
    env->ReleaseIntArrayElements(jRlPool, pool, JNI_ABORT);
}

// DISTR_RHO coefficient overrides: install (or clear, with nulls) the per-scenario 1/count'
// renormalization pairs {s, contrib} + new coefficients on a batch context (a shared-TVC group
// member outaged in that scenario — plan-group-member-dynamics A5).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDrCoefOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jC) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jC == nullptr || env->GetArrayLength(jC) == 0) {
        ctx->nDrOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jC);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* c = env->GetDoubleArrayElements(jC, nullptr);
    ctx->dDrOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDrOvC.upload(c, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jC, c, JNI_ABORT);
    ctx->nDrOv = nOv;
}
