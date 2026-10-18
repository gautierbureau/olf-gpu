// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: HVDC AC emulation loops.
#include "jni_common.cuh"

// The HVDC AC-emulation outer loops' tables + their per-scenario starting state (the statuses and
// applied converter-row deltas the host's pre-solve freeze produced). Install to run those loops as
// DRIVER PHASES in OLF's order; clear (null) to leave them to the host orchestrator.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, HvdcNative, setBatchHvdcLoops)(
        JNIEnv* env, jclass, jlong handle, jintArray jElem, jintArray jBus1Row, jintArray jBus2Row,
        jintArray jBus1Phi, jintArray jBus2Phi,
        jdoubleArray jIn, jbyteArray jCtgOut, jintArray jStatus,
        jdoubleArray jAppliedD1, jdoubleArray jAppliedD2, jint limitsActive) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jElem == nullptr || env->GetArrayLength(jElem) == 0) {
        ctx->nhv = 0;
        return;
    }
    int nh = env->GetArrayLength(jElem);
    auto upI = [&](jintArray a, DeviceBuf& b) {
        jint* q = env->GetIntArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, q, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& b) {
        jdouble* q = env->GetDoubleArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, q, JNI_ABORT);
    };
    upI(jElem, ctx->dHvElem);
    upI(jBus1Row, ctx->dHvBus1Row);
    upI(jBus2Row, ctx->dHvBus2Row);
    upI(jBus1Phi, ctx->dHvBus1Phi);
    upI(jBus2Phi, ctx->dHvBus2Phi);
    upD(jIn, ctx->dHvIn);
    upI(jStatus, ctx->dHvStatus);
    upD(jAppliedD1, ctx->dHvAppliedD1);
    upD(jAppliedD2, ctx->dHvAppliedD2);
    jbyte* co = env->GetByteArrayElements(jCtgOut, nullptr);
    ctx->dHvCtgOut.upload(co, (size_t) env->GetArrayLength(jCtgOut));
    env->ReleaseByteArrayElements(jCtgOut, co, JNI_ABORT);
    ctx->nhv = nh;
    ctx->hvLimitsActive = limitsActive;
}

/** Read the HVDC statuses back after a solve, so the host keeps its own view in step. */
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, HvdcNative, readBatchHvdcStatus)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nhv == 0) {
        return nullptr;
    }
    size_t len = (size_t) ctx->nScenarios * ctx->nhv;
    std::vector<int> host(len);
    if (cudaMemcpy(host.data(), ctx->dHvStatus.p, len * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return nullptr;
    }
    jintArray out = env->NewIntArray((jsize) len);
    env->SetIntArrayRegion(out, 0, (jsize) len, host.data());
    return out;
}
