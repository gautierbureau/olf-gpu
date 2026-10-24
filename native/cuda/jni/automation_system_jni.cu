// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of AutomationSystemNative: the AutomationSystem phase's tables and its read-back.
#include "jni_common.cuh"

// Install (or clear, with null systems) the run's overload management systems: per system its monitored closed-branch
// element and side (GpuTwoSides) and the slice of its trippings (trippingStart, nSystems + 1); per tripping the
// disabled-mask slot of the closed branch it operates, whether it opens it, and its current threshold (pu).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, AutomationSystemNative, setBatchAutomationSystem)(
        JNIEnv* env, jclass, jlong handle, jintArray jMonitoredElement, jintArray jMonitoredSide, jintArray jTrippingStart,
        jintArray jOperatedMaskSlot, jbyteArray jOpenAction, jdoubleArray jThreshold) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->nOverloadManagementSystems = 0;
    ctx->nOverloadManagementSystemTrippings = 0;
    if (jMonitoredElement == nullptr || env->GetArrayLength(jMonitoredElement) == 0) {
        return;
    }
    auto upI = [&](jintArray a, DeviceBuf& buf) {
        jint* p = env->GetIntArrayElements(a, nullptr);
        buf.upload(p, std::max<size_t>(1, static_cast<size_t>(env->GetArrayLength(a))) * sizeof(int));
        env->ReleaseIntArrayElements(a, p, JNI_ABORT);
    };
    upI(jMonitoredElement, ctx->dOverloadManagementSystemMonitoredElement);
    upI(jMonitoredSide, ctx->dOverloadManagementSystemMonitoredSide);
    upI(jTrippingStart, ctx->dOverloadManagementSystemTrippingStart);
    upI(jOperatedMaskSlot, ctx->dOverloadManagementSystemOperatedMaskSlot);
    jbyte* open = env->GetByteArrayElements(jOpenAction, nullptr);
    ctx->dOverloadManagementSystemOpenAction.upload(open, std::max<size_t>(1, static_cast<size_t>(env->GetArrayLength(jOpenAction))));
    env->ReleaseByteArrayElements(jOpenAction, open, JNI_ABORT);
    jdouble* threshold = env->GetDoubleArrayElements(jThreshold, nullptr);
    ctx->dOverloadManagementSystemThreshold.upload(threshold, std::max<size_t>(1, static_cast<size_t>(env->GetArrayLength(jThreshold))) * sizeof(double));
    env->ReleaseDoubleArrayElements(jThreshold, threshold, JNI_ABORT);
    std::vector<int> operatedMaskSlot(static_cast<size_t>(env->GetArrayLength(jOperatedMaskSlot)));
    env->GetIntArrayRegion(jOperatedMaskSlot, 0, static_cast<jsize>(operatedMaskSlot.size()), operatedMaskSlot.data());
    ctx->hOverloadManagementSystemOperatedMaskSlot = operatedMaskSlot;
    ctx->nOverloadManagementSystems = env->GetArrayLength(jMonitoredElement);
    ctx->nOverloadManagementSystemTrippings = static_cast<int>(operatedMaskSlot.size());
}

// Per scenario and tripping (s * nTrippings + t), whether the solve left the tripping's branch disabled (1) or enabled (0).
extern "C" JNIEXPORT jbyteArray JNICALL
OLF_JNI(ac_outerloop_gpu, AutomationSystemNative, readBatchAutomationSystemOperatedDisabled)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nOverloadManagementSystems <= 0) {
        return nullptr;
    }
    int nTrippings = ctx->nOverloadManagementSystemTrippings;
    std::vector<char> disabled(static_cast<size_t>(ctx->nScenarios) * ctx->nElements);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(disabled.data(), ctx->dDisabled.p, disabled.size(), cudaMemcpyDeviceToHost));
    std::vector<jbyte> host(static_cast<size_t>(ctx->nScenarios) * nTrippings);
    for (int s = 0; s < ctx->nScenarios; s++) {
        for (int t = 0; t < nTrippings; t++) {
            host[(size_t) s * nTrippings + t] = disabled[(size_t) s * ctx->nElements + ctx->hOverloadManagementSystemOperatedMaskSlot[t]];
        }
    }
    jbyteArray out = env->NewByteArray(static_cast<jsize>(host.size()));
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}
