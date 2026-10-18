// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: DistributedSlackOuterLoop tables and per-scenario overrides.
#include "jni_common.cuh"

// Remote-member RL entries (plan-B1): install (or clear) the per-entry remote kinds + compare rows
// on a batch context. The RL kernel SKIPS remote entries until the B3 transition machinery lands.
// Slack-only distribution: the slack generators' initial target per scenario (relative to the slack's target P)
// for the next solves - a strategy pair's parent total; null clears it (zero: a contingency's base).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchSlackOnlyAnchor)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jAnchor) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jAnchor == nullptr || !ctx->dsSlackOnly) {
        ctx->dsSlackHasAnchor = false;
        return;
    }
    jdouble* a = env->GetDoubleArrayElements(jAnchor, nullptr);
    ctx->dDsSlackAnchor.upload(a, static_cast<size_t>(env->GetArrayLength(jAnchor)) * sizeof(double));
    env->ReleaseDoubleArrayElements(jAnchor, a, JNI_ABORT);
    ctx->dsSlackHasAnchor = true;
}

// Per scenario and participating generator (s * ngen + g, pu): anchor = its move from its base target P to where the
// slack's water-fill starts it (a strategy pair: the parent's final target, or a GeneratorAction's value; null = the
// base), start = its move from that anchor at the solve's start (a contingency's loss pre-distribution; null = none).
// The curve and forced-target-Q channels evaluate a generator at anchor + (genCum once a pass ran, else start).
// seedAnchor (pairs): the anchor WITHOUT the strategy's GeneratorActions - where the continued pins are stamped.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchGenAnchor)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jAnchor, jdoubleArray jStart, jdoubleArray jSeedAnchor) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    size_t want = static_cast<size_t>(ctx->nScenarios) * std::max(ctx->ngen, 0);
    auto put = [&](jdoubleArray arr, DeviceBuf& buf, bool& has) {
        has = false;
        if (arr == nullptr || ctx->ngen <= 0) {
            return true;
        }
        if (static_cast<size_t>(env->GetArrayLength(arr)) != want) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "setBatchGenAnchor: expected nScenarios * ngen");
            return false;
        }
        jdouble* a = env->GetDoubleArrayElements(arr, nullptr);
        buf.upload(a, want * sizeof(double));
        env->ReleaseDoubleArrayElements(arr, a, JNI_ABORT);
        has = true;
        return true;
    };
    if (put(jAnchor, ctx->dDgAnchor, ctx->hasDgAnchor) && put(jStart, ctx->dDgStart, ctx->hasDgStart)) {
        put(jSeedAnchor, ctx->dDgSeedAnchor, ctx->hasDgSeedAnchor);
    }
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDsLostGenerators)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotGen) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotGen == nullptr || env->GetArrayLength(jSlotGen) == 0) {
        ctx->nDsLostGen = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotGen);
    jint* sg = env->GetIntArrayElements(jSlotGen, nullptr);
    ctx->dDsLostGen.upload(sg, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotGen, sg, JNI_ABORT);
    ctx->nDsLostGen = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDsOutGenerators)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotGen) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotGen == nullptr || env->GetArrayLength(jSlotGen) == 0) {
        ctx->nDsOutGen = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotGen);
    jint* sg = env->GetIntArrayElements(jSlotGen, nullptr);
    ctx->dDsOutGen.upload(sg, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotGen, sg, JNI_ABORT);
    ctx->nDsOutGen = n2 / 2;
}

// Per-GENERATOR distributed-slack participants: the factors and limits OLF's water-fill actually
// saturates on, grouped by bus (a bus's row value is the sum over its contiguous slice).
// The slack bus's participating generators' total rooms: the bound of a slack-only distribution (slackOnlyDsCheck).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDsSlackRooms)(
        JNIEnv*, jclass, jlong handle, jdouble roomUp, jdouble roomDown) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->dsSlackRoomUp = roomUp;
        ctx->dsSlackRoomDown = roomDown;
    }
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDsGenerators)(
        JNIEnv* env, jclass, jlong handle, jintArray jBus, jdoubleArray jFactor,
        jdoubleArray jMaxDelta, jdoubleArray jMinDelta, jdoubleArray jMarginUp, jdoubleArray jMarginDown,
        jintArray jBusGenStart, jintArray jBusGenEnd) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jBus == nullptr || env->GetArrayLength(jBus) == 0) {
        ctx->ngen = 0;
        return;
    }
    int ngen = env->GetArrayLength(jBus);
    auto upI = [&](jintArray a, DeviceBuf& buf) {
        jint* p = env->GetIntArrayElements(a, nullptr);
        buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, p, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& buf) {
        if (a == nullptr || env->GetArrayLength(a) == 0) {
            return;
        }
        jdouble* p = env->GetDoubleArrayElements(a, nullptr);
        buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, p, JNI_ABORT);
    };
    upI(jBus, ctx->dgBus);
    upD(jFactor, ctx->dgFactor);
    upD(jMaxDelta, ctx->dgMaxDelta);
    upD(jMinDelta, ctx->dgMinDelta);
    upD(jMarginUp, ctx->dgMarginUp);
    upD(jMarginDown, ctx->dgMarginDown);
    upI(jBusGenStart, ctx->dBusGenStart);
    upI(jBusGenEnd, ctx->dBusGenEnd);
    ctx->dGenCum.alloc((size_t) ctx->nScenarios * ngen * sizeof(double));
    ctx->dGenCumValid.alloc((size_t) ctx->nScenarios * sizeof(char));
    ctx->dGenFactorWork.alloc((size_t) ctx->nScenarios * ngen * sizeof(double));
    ctx->dGenRemoved.alloc((size_t) ctx->nScenarios * ngen * sizeof(char));
    ctx->ngen = ngen;
}

// Per slot, the row the distributed-slack kernels read the slack on (host row, -1 = the base slack row): a
// contingency that isolates the base slack makes OLF re-pick one, whose angle row becomes the reference. Null clears.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDsSlackRows)(
        JNIEnv* env, jclass, jlong handle, jintArray jRows) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jRows == nullptr || env->GetArrayLength(jRows) == 0) {
        ctx->hasDsSlackRowOv = 0;
        return;
    }
    int ne = env->GetArrayLength(jRows);
    std::vector<int> rows(ne);
    env->GetIntArrayRegion(jRows, 0, ne, rows.data());
    for (int& r : rows) {
        r = r >= 0 ? ctx->st->rowMap[r] : -1;
    }
    ctx->dDsSlackRowOv.upload(rows.data(), static_cast<size_t>(ne) * sizeof(int));
    ctx->hasDsSlackRowOv = 1;
}

// Distributed-slack participants STRANDED in another synchronous component: install (or clear, with
// null) the per-scenario mask as {slot, participant} pairs. A contingency that fragments the network
// leaves some participating buses outside the main component; OLF drops them from the participant
// set, so the device must renormalize over the survivors — see distributeSlackBatch.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDsIslanded)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nds == 0) {
        ctx->nDsIslanded = 0;
        return;
    }
    // Expanded to a dense (scenario, participant) mask on the host: the kernel reads it once per
    // participant inside loops it already walks, so a scan over a pair list would be the only cost
    // the mask adds to scenarios that have no islanded participant at all.
    size_t maskLen = static_cast<size_t>(ctx->nScenarios) * ctx->nds;
    std::vector<char> mask(maskLen, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nds) {
            mask[(size_t) s * ctx->nds + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dDsIslanded.upload(mask.data(), maskLen);
    ctx->nDsIslanded = nPairs;
}

// Per-scenario P-balance row of a DS participant this scenario MOVES: {slot, participant, row}
// triples, expanded to a dense (scenario, participant) row array (-1 = the structural dsPhiRow).
// The rows arrive in the same space dsPhiRow is given in and are taken through the structure's
// rowMap here, exactly as createBatchContext maps dsPhiRow itself — so the two are always in one
// space whatever rowMap is.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDsRowOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jTriples) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jTriples == nullptr || env->GetArrayLength(jTriples) == 0 || ctx->nds == 0) {
        ctx->nDsRowOv = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nds;
    std::vector<int> rows(len, -1);
    int nTriples = env->GetArrayLength(jTriples) / 3;
    jint* t = env->GetIntArrayElements(jTriples, nullptr);
    const std::vector<int>& rowMap = ctx->st->rowMap;
    int nOk = 0;
    for (int k = 0; k < nTriples; k++) {
        int s = t[3 * k];
        int i = t[3 * k + 1];
        int row = t[3 * k + 2];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nds
                && row >= 0 && row < static_cast<int>(rowMap.size()) && rowMap[row] >= 0) {
            rows[(size_t) s * ctx->nds + i] = rowMap[row];
            nOk++;
        }
    }
    env->ReleaseIntArrayElements(jTriples, t, JNI_ABORT);
    ctx->dDsRowOv.upload(rows.data(), len * sizeof(int));
    ctx->nDsRowOv = nOk;
}
