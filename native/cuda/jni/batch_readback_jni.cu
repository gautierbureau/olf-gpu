// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: What a batched solve leaves on the device, read back.
#include "jni_common.cuh"
#include "../solver/scenario_install.cuh"

// Read back the batch context's per-scenario TARGET buffer (n doubles per installed scenario) after a solve —
// the device outer loops (distributed slack) mutate targets in place, and the strategy/pair phase needs
// the PARENT contingency's final distributed targets to re-anchor its own distribution (OLF's
// setGeneratorsInitialTargetPToTargetP semantics).
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchTargets)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    std::vector<double> host;
    readInstalledRows(ctx, ctx->dTarget.p, static_cast<size_t>(ctx->n), host);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}

// The per-scenario ROW MODES as the device ENDED with them, the companion to readBatchTargets.
// A row's equation is (mode, target) together, and the outer-loop kernels rewrite BOTH: the row modes the
// host installed are only what SETUP asked for, so reading a target without its final mode can
// say "the target is right" about a row that is no longer the equation it was set up to be. That is
// the whole failure shape of the device-target-writer class, where a wrong row converges cleanly and
// nothing flags it.
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchRowModes)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    std::vector<int> host;
    readInstalledRows(ctx, ctx->dMode.p, static_cast<size_t>(ctx->n), host);
    jintArray out = env->NewIntArray(static_cast<jsize>(host.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}

// Per-scenario slack-only-DS absorbed totals after a solve (see slackOnlyDistributedSlackCheck) — the pair phase
// seeds its scenarios with the parent contingency's value (the device analogue of OLF's
// setGeneratorsInitialTargetPToTargetP before a strategy re-solve). Null when not slack-only.
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchSlackAbsorbed)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->slackOnlyDistribution) {
        return nullptr;
    }
    std::vector<double> host;
    readInstalledRows(ctx, ctx->dSlackOnlyAbsorbed.p, 1, host);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchDistributedSlackGeneratorMove)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nDistributedSlackGenerators <= 0 || !ctx->dDistributedSlackGeneratorMove.p) {
        return nullptr;
    }
    std::vector<double> host;
    readInstalledRows(ctx, ctx->dDistributedSlackGeneratorMove.p, static_cast<size_t>(ctx->nDistributedSlackGenerators), host);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}

// Per installed scenario of the last solveBatchContext call: {solve passes, first pass's
// Newton iterations, total Newton iterations, outer-loop checks}, flattened s*4. The per-contingency counterpart of
// GPU_OL_HIST, for matching OLF's CPU_OL_STATS contingency by contingency.
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchOuterLoopStats)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->outerLoopSolvePasses.size() != static_cast<size_t>(ctx->nScenarios)) {
        return nullptr;
    }
    std::vector<jint> host(static_cast<size_t>(installedSlotCount(ctx)) * 4);
    for (int j = 0; j < installedSlotCount(ctx); j++) {
        int s = installedSlot(ctx, j);
        host[j * 4] = ctx->outerLoopSolvePasses[s];
        host[j * 4 + 1] = ctx->outerLoopFirstSolveIterations[s];
        host[j * 4 + 2] = ctx->outerLoopIterations[s];
        host[j * 4 + 3] = ctx->outerLoopChecks[s];
    }
    jintArray out = env->NewIntArray(static_cast<jsize>(host.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}

// The batched Newton iterations of the last solveBatchContext call, every scenario's solves together: the length of the
// chunk's solve, in factorization rounds.
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchNewtonLoopIterations)(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    return ctx == nullptr ? -1 : ctx->newtonLoopIterations;
}

// How many solveBatchContext calls this context has run. The per-solve DS read-backs (retain, distributedSlackGeneratorMove,
// slack absorbed) are reset at every solve, so a caller summing them must know they span its solve.
extern "C" JNIEXPORT jlong JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchSolveCount)(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    return ctx == nullptr ? -1 : static_cast<jlong>(ctx->solveCount);
}

// One scenario's state at its last converged inner solve of the last solve (its starting state if none converged), by
// its slot in the last install.
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchLastConvergedState)(
        JNIEnv* env, jclass, jlong handle, jint slot) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dXLastConv.p || slot < 0 || slot >= installedSlotCount(ctx)) {
        return nullptr;
    }
    int n = ctx->structure->n;
    std::vector<double> host(n);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(host.data(), ctx->dXLastConv.as<double>() + (size_t) installedSlot(ctx, slot) * n, n * sizeof(double),
                          cudaMemcpyDeviceToHost));
    jdoubleArray out = env->NewDoubleArray(n);
    env->SetDoubleArrayRegion(out, 0, n, host.data());
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchSlackRetainedShare)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dSlackRetainedShareApplied.p) {
        return nullptr;
    }
    std::vector<double> host;
    readInstalledRows(ctx, ctx->dSlackRetainedShareApplied.p, 1, host);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}
