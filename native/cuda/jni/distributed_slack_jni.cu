// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: DistributedSlackOuterLoop tables and per-scenario overrides.
#include "jni_common.cuh"
#include "slot_table_install.cuh"

// Remote-member RL entries (plan-B1): install (or clear) the per-entry remote kinds + compare rows
// on a batch context. The RL kernel SKIPS remote entries until the B3 transition machinery lands.
// Slack-only distribution: the slack generators' initial target per installed scenario (relative to the slack's target
// P) for the next solves - a strategy pair's parent total; null: zero, a contingency's base.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchSlackOnlyAnchor)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jAnchor) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installSlotRows(env, ctx, ctx->distributedSlack.slackOnlyAnchor, 1, ctx->distributedSlack.slackOnlyDistribution ? jAnchor : nullptr, "setBatchSlackOnlyAnchor");
    }
}

// Per scenario and participating generator (s * nDistributedSlackGenerators + g, pu): anchor = its move from its base target P to where the
// slack's water-fill starts it (a strategy pair: the parent's final target, or a GeneratorAction's value; null = the
// base), start = its move from that anchor at the solve's start (a contingency's loss pre-distribution; null = none).
// The curve and forced-target-Q channels evaluate a generator at anchor + (distributedSlackGeneratorMove once a pass ran, else start).
// seedAnchor (pairs): the anchor WITHOUT the strategy's GeneratorActions - where the continued PV -> PQ switches are stamped.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDistributedSlackGeneratorAnchors)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jAnchor, jdoubleArray jStart, jdoubleArray jSeedAnchor) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    size_t width = static_cast<size_t>(std::max(ctx->distributedSlack.nDistributedSlackGenerators, 0));
    const char* who = "setBatchDistributedSlackGeneratorAnchors";
    installSlotRows(env, ctx, ctx->distributedSlack.distributedSlackGeneratorAnchor, width, jAnchor, who);
    if (!env->ExceptionCheck()) {
        installSlotRows(env, ctx, ctx->distributedSlack.distributedSlackGeneratorInitialMove, width, jStart, who);
    }
    if (!env->ExceptionCheck()) {
        installSlotRows(env, ctx, ctx->distributedSlack.distributedSlackGeneratorSeedAnchor, width, jSeedAnchor, who);
    }
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDistributedSlackLostGenerators)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotGen) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installSlotTable(env, ctx, ctx->distributedSlack.distributedSlackLostGenerators, jSlotGen, nullptr);
    }
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDistributedSlackRemovedGenerators)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotGen) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installSlotTable(env, ctx, ctx->distributedSlack.distributedSlackRemovedGenerators, jSlotGen, nullptr);
    }
}

// Per-GENERATOR distributed-slack participants: the factors and limits OLF's water-fill actually
// saturates on, grouped by bus (a bus's row value is the sum over its contiguous slice).
// The slack bus's participating generators' total rooms: the bound of a slack-only distribution (slackOnlyDistributedSlackCheck).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchSlackBusRooms)(
        JNIEnv*, jclass, jlong handle, jdouble roomUp, jdouble roomDown) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->distributedSlack.slackBusRoomUp = roomUp;
        ctx->distributedSlack.slackBusRoomDown = roomDown;
    }
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDistributedSlackGenerators)(
        JNIEnv* env, jclass, jlong handle, jintArray jBus, jdoubleArray jFactor,
        jdoubleArray jMaxDelta, jdoubleArray jMinDelta, jdoubleArray jMarginUp, jdoubleArray jMarginDown,
        jintArray jBusGenStart, jintArray jBusGenEnd) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jBus == nullptr || env->GetArrayLength(jBus) == 0) {
        ctx->distributedSlack.nDistributedSlackGenerators = 0;
        return;
    }
    int nDistributedSlackGenerators = env->GetArrayLength(jBus);
    auto upI = [&](jintArray a, DeviceBuf& buf) {
        const JniInts p(env, a);
        buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(int));
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& buf) {
        if (a == nullptr || env->GetArrayLength(a) == 0) {
            return;
        }
        const JniDoubles p(env, a);
        buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(double));
    };
    upI(jBus, ctx->distributedSlack.distributedSlackGeneratorBus);
    upD(jFactor, ctx->distributedSlack.distributedSlackGeneratorFactor);
    upD(jMaxDelta, ctx->distributedSlack.distributedSlackGeneratorMaxDelta);
    upD(jMinDelta, ctx->distributedSlack.distributedSlackGeneratorMinDelta);
    upD(jMarginUp, ctx->distributedSlack.distributedSlackGeneratorMarginUp);
    upD(jMarginDown, ctx->distributedSlack.distributedSlackGeneratorMarginDown);
    upI(jBusGenStart, ctx->distributedSlack.dDistributedSlackBusGeneratorStart);
    upI(jBusGenEnd, ctx->distributedSlack.dDistributedSlackBusGeneratorEnd);
    ctx->distributedSlack.dDistributedSlackGeneratorMove.alloc((size_t) ctx->nScenarios * nDistributedSlackGenerators * sizeof(double));
    ctx->distributedSlack.dDistributedSlackGeneratorMoveValid.alloc((size_t) ctx->nScenarios * sizeof(char));
    ctx->distributedSlack.dDistributedSlackGeneratorWorkingFactor.alloc((size_t) ctx->nScenarios * nDistributedSlackGenerators * sizeof(double));
    ctx->distributedSlack.dDistributedSlackGeneratorRemoved.alloc((size_t) ctx->nScenarios * nDistributedSlackGenerators * sizeof(char));
    ctx->distributedSlack.nDistributedSlackGenerators = nDistributedSlackGenerators;
}

// Per installed slot, the row the distributed-slack kernels read the slack on (host row, -1 = the base slack row): a
// contingency that isolates the base slack makes OLF re-pick one, whose angle row becomes the reference. Null: the
// base slack row everywhere.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchSlackBusRows)(
        JNIEnv* env, jclass, jlong handle, jintArray jRows) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    std::vector<int> rows = jniInts(env, jRows);
    for (int& r : rows) {
        r = r >= 0 ? ctx->structure->rowMap[r] : -1;
    }
    if (!rows.empty() && static_cast<int>(rows.size()) != installedSlotCount(ctx)) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "setBatchSlackBusRows: expected one row per installed scenario");
        return;
    }
    ctx->distributedSlack.slackBusRowOverride.install(ctx->nScenarios, ctx->installSlots, 1, rows.empty() ? nullptr : rows.data());
}

// Distributed-slack participants STRANDED in another synchronous component: the installed slots' mask as {slot,
// participant} pairs, null for none. A contingency that fragments the network
// leaves some participating buses outside the main component; OLF drops them from the participant
// set, so the device must renormalize over the survivors — see distributeSlackBatch.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDistributedSlackIslanded)(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    // Expanded to a dense (scenario, participant) mask on the host: the kernel reads it once per
    // participant inside loops it already walks, so a scan over a pair list would be the only cost
    // the mask adds to scenarios that have no islanded participant at all.
    std::vector<int> pairs = jniInts(env, jPairs);
    ctx->distributedSlack.distributedSlackIslanded.installEntries(ctx->nScenarios, ctx->installSlots, ctx->distributedSlack.nDistributedSlackBuses,
                                                 static_cast<int>(pairs.size() / 2), pairs.data(), nullptr, 1);
}

// Per-scenario P-balance row of a DS participant this scenario MOVES: {slot, participant, row}
// triples, expanded to a dense (scenario, participant) row array (-1 = the structural distributedSlackBusPhiRow).
// The rows arrive in the same space distributedSlackBusPhiRow is given in and are taken through the structure's
// rowMap here, exactly as createBatchContext maps distributedSlackBusPhiRow itself — so the two are always in one
// space whatever rowMap is.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, DistributedSlackNative, setBatchDistributedSlackBusRowOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jTriples) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    std::vector<int> triples = jniInts(env, jTriples);
    const std::vector<int>& rowMap = ctx->structure->rowMap;
    std::vector<int> pairs;
    std::vector<int> rows;
    for (size_t k = 0; k + 2 < triples.size(); k += 3) {
        int row = triples[k + 2];
        if (row >= 0 && row < static_cast<int>(rowMap.size()) && rowMap[row] >= 0) {
            pairs.push_back(triples[k]);
            pairs.push_back(triples[k + 1]);
            rows.push_back(rowMap[row]);
        }
    }
    ctx->distributedSlack.distributedSlackBusRowOverride.installEntries(ctx->nScenarios, ctx->installSlots, ctx->distributedSlack.nDistributedSlackBuses,
                                                       static_cast<int>(rows.size()), pairs.data(), rows.data());
}
