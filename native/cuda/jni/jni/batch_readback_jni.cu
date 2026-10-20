// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: What a batched solve leaves on the device, read back.
#include "jni_common.cuh"

// Read back the batch context's per-scenario TARGET buffer (n * nScenarios doubles) after a solve —
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
    size_t nTot = static_cast<size_t>(ctx->n) * ctx->nScenarios;
    std::vector<double> host(nTot);
    cudaMemcpy(host.data(), ctx->dTarget.p, nTot * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(nTot));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(nTot), host.data());
    return out;
}

// The per-scenario ROW MODES as the device ENDED with them, the companion to readBatchTargets.
// A row's equation is (mode, target) together, and the outer-loop kernels rewrite BOTH: the host's
// rowModePerScenario is only what SETUP asked for, so reading a target without its final mode can
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
    size_t nTot = static_cast<size_t>(ctx->n) * ctx->nScenarios;
    std::vector<int> host(nTot);
    cudaMemcpy(host.data(), ctx->dMode.p, nTot * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(nTot));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(nTot), host.data());
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
    std::vector<double> host(ctx->nScenarios);
    cudaMemcpy(host.data(), ctx->dSlackOnlyAbsorbed.p, ctx->nScenarios * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(ctx->nScenarios));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(ctx->nScenarios), host.data());
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchDistributedSlackGeneratorMove)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nDistributedSlackGenerators <= 0 || !ctx->dDistributedSlackGeneratorMove.p) {
        return nullptr;
    }
    size_t count = static_cast<size_t>(ctx->nScenarios) * ctx->nDistributedSlackGenerators;
    std::vector<double> host(count);
    cudaMemcpy(host.data(), ctx->dDistributedSlackGeneratorMove.p, count * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(count));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(count), host.data());
    return out;
}

// Per scenario of the last solveBatchContext call (and its host-loop resumes): {solve passes, first pass's
// Newton iterations, total Newton iterations}, flattened s*3. The per-contingency counterpart of
// GPU_OL_HIST, for matching OLF's CPU_OL_STATS contingency by contingency.
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchOuterLoopStats)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->outerLoopSolvePasses.size() != static_cast<size_t>(ctx->nScenarios)) {
        return nullptr;
    }
    std::vector<jint> host(static_cast<size_t>(ctx->nScenarios) * 3);
    for (int s = 0; s < ctx->nScenarios; s++) {
        host[s * 3] = ctx->outerLoopSolvePasses[s];
        host[s * 3 + 1] = ctx->outerLoopFirstSolveIterations[s];
        host[s * 3 + 2] = ctx->outerLoopIterations[s];
    }
    jintArray out = env->NewIntArray(static_cast<jsize>(host.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}

// How many solveBatchContext calls this context has run. The per-solve DS read-backs (retain, distributedSlackGeneratorMove,
// slack absorbed) are reset at every solve, so a caller summing them must know they span its solve.
extern "C" JNIEXPORT jlong JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchSolveCount)(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    return ctx == nullptr ? -1 : static_cast<jlong>(ctx->solveCount);
}

// One scenario's state at its last converged inner solve of the last solve (its starting state if none converged).
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_solver_gpu, BatchReadbackNative, readBatchLastConvergedState)(
        JNIEnv* env, jclass, jlong handle, jint slot) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dXLastConv.p || slot < 0 || slot >= ctx->nScenarios) {
        return nullptr;
    }
    int n = ctx->st->n;
    std::vector<double> host(n);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(host.data(), ctx->dXLastConv.as<double>() + (size_t) slot * n, n * sizeof(double),
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
    std::vector<double> host(ctx->nScenarios);
    cudaMemcpy(host.data(), ctx->dSlackRetainedShareApplied.p, ctx->nScenarios * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(ctx->nScenarios));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(ctx->nScenarios), host.data());
    return out;
}
