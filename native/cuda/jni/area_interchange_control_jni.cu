// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of AreaInterchangeControlNative: the AreaInterchangeControl phase's tables and its read-back.
#include "jni_common.cuh"

// Install (or clear, with outerLoop false) the run's AreaInterchangeControl: per area its interchange target and slack
// distribution factor (pu) and the slice of its boundaries (boundaryStart, nAreas + 1); per boundary its
// GpuAreaBoundaryKind, family element, disabled-mask slot and side, and for a tie line half {r, x, g, b, nominalV, SB};
// per distributed-slack bus and generator its area (-1: without area).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, AreaInterchangeControlNative, setBatchAreaInterchangeControl)(
        JNIEnv* env, jclass, jlong handle, jboolean outerLoop, jdouble areaInterchangePMaxMismatch,
        jdoubleArray jInterchangeTarget, jdoubleArray jSlackDistributionFactor, jintArray jBoundaryStart,
        jintArray jBoundaryKind, jintArray jBoundaryElement, jintArray jBoundaryMaskSlot, jintArray jBoundarySide,
        jdoubleArray jBoundaryTieLineHalf, jintArray jBusArea, jintArray jGeneratorArea) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->areaInterchangeControlOuterLoop = outerLoop == JNI_TRUE;
    ctx->nAreas = 0;
    ctx->nAreaBoundaries = 0;
    if (!ctx->areaInterchangeControlOuterLoop || jInterchangeTarget == nullptr || env->GetArrayLength(jInterchangeTarget) == 0) {
        return;                                                     // no area: the phase is OLF's DistributedSlackOuterLoop
    }
    auto upI = [&](jintArray a, DeviceBuf& buf) {
        jint* p = env->GetIntArrayElements(a, nullptr);
        buf.upload(p, std::max<size_t>(1, static_cast<size_t>(env->GetArrayLength(a))) * sizeof(int));
        env->ReleaseIntArrayElements(a, p, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& buf) {
        jdouble* p = env->GetDoubleArrayElements(a, nullptr);
        buf.upload(p, std::max<size_t>(1, static_cast<size_t>(env->GetArrayLength(a))) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, p, JNI_ABORT);
    };
    upD(jInterchangeTarget, ctx->dAreaInterchangeTarget);
    upD(jSlackDistributionFactor, ctx->dAreaSlackDistributionFactor);
    upI(jBoundaryStart, ctx->dAreaBoundaryStart);
    upI(jBoundaryKind, ctx->dAreaBoundaryKind);
    upI(jBoundaryElement, ctx->dAreaBoundaryElement);
    upI(jBoundaryMaskSlot, ctx->dAreaBoundaryMaskSlot);
    upI(jBoundarySide, ctx->dAreaBoundarySide);
    upD(jBoundaryTieLineHalf, ctx->dAreaBoundaryTieLineHalf);
    upI(jBusArea, ctx->dDistributedSlackBusArea);
    upI(jGeneratorArea, ctx->dDistributedSlackGeneratorArea);
    ctx->areaInterchangePMaxMismatch = areaInterchangePMaxMismatch;
    ctx->nAreas = env->GetArrayLength(jInterchangeTarget);
    ctx->nAreaBoundaries = env->GetArrayLength(jBoundaryKind);
}

// Per scenario, the active power the AreaInterchangeControl checks of the last solve distributed (pu), as OLF's context data
// accumulates it.
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, AreaInterchangeControlNative, readBatchAreaInterchangeControlDistributedActivePower)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dAreaInterchangeControlDistributedActivePower.p) {
        return nullptr;
    }
    std::vector<double> host(static_cast<size_t>(ctx->nScenarios));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(host.data(), ctx->dAreaInterchangeControlDistributedActivePower.p, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}
