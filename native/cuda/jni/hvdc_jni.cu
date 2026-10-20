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
        JNIEnv* env, jclass, jlong handle, jintArray jElem, jintArray jHvdcBus1PRow, jintArray jHvdcBus2PRow,
        jintArray jHvdcBus1PhiColumn, jintArray jHvdcBus2PhiColumn,
        jdoubleArray jIn, jbyteArray jHvdcContingencyOutage, jintArray jHvdcAcEmulationStatus,
        jdoubleArray jHvdcAppliedDelta1, jdoubleArray jHvdcAppliedDelta2, jint limitsActive) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jElem == nullptr || env->GetArrayLength(jElem) == 0) {
        ctx->nHvdcAcEmulation = 0;
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
    upI(jElem, ctx->dHvdcElement);
    upI(jHvdcBus1PRow, ctx->dHvdcBus1PRow);
    upI(jHvdcBus2PRow, ctx->dHvdcBus2PRow);
    upI(jHvdcBus1PhiColumn, ctx->dHvdcBus1PhiColumn);
    upI(jHvdcBus2PhiColumn, ctx->dHvdcBus2PhiColumn);
    upD(jIn, ctx->dHvdcAcEmulationParameters);
    upI(jHvdcAcEmulationStatus, ctx->dHvdcAcEmulationStatus);
    upD(jHvdcAppliedDelta1, ctx->dHvdcAppliedDelta1);
    upD(jHvdcAppliedDelta2, ctx->dHvdcAppliedDelta2);
    jbyte* co = env->GetByteArrayElements(jHvdcContingencyOutage, nullptr);
    ctx->dHvdcContingencyOutage.upload(co, (size_t) env->GetArrayLength(jHvdcContingencyOutage));
    env->ReleaseByteArrayElements(jHvdcContingencyOutage, co, JNI_ABORT);
    ctx->nHvdcAcEmulation = nh;
    ctx->hvdcAcEmulationLimitsActive = limitsActive;
}

/** Read the HVDC statuses back after a solve, so the host keeps its own view in step. */
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, HvdcNative, readBatchHvdcStatus)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nHvdcAcEmulation == 0) {
        return nullptr;
    }
    size_t len = (size_t) ctx->nScenarios * ctx->nHvdcAcEmulation;
    std::vector<int> host(len);
    if (cudaMemcpy(host.data(), ctx->dHvdcAcEmulationStatus.p, len * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return nullptr;
    }
    jintArray out = env->NewIntArray((jsize) len);
    env->SetIntArrayRegion(out, 0, (jsize) len, host.data());
    return out;
}
