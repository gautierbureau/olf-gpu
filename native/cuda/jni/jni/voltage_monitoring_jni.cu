// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: MonitoringVoltageOuterLoop.
#include "jni_common.cuh"

/** MonitoringVoltage controller table (OLF order 6). Empty array clears it. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchMonitoringLoops(
        JNIEnv* env, jclass, jlong handle, jintArray jVRow, jdoubleArray jBand,
        jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaMode, jdoubleArray jDeltaTarget) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jVRow == nullptr || env->GetArrayLength(jVRow) == 0) {
        ctx->nmon = 0;
        return;
    }
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
    upI(jVRow, ctx->dMonVRow);
    upD(jBand, ctx->dMonBand);
    upI(jDeltaPtr, ctx->dMonDeltaPtr);
    upI(jDeltaRow, ctx->dMonDeltaRow);
    upI(jDeltaMode, ctx->dMonDeltaMode);
    upD(jDeltaTarget, ctx->dMonDeltaTarget);
    ctx->nmon = env->GetArrayLength(jVRow);
    ctx->dMonActivated.alloc(static_cast<size_t>(ctx->nmon) * ctx->nScenarios * sizeof(double));
}

/** How many (controller, scenario) monitoring automatons FIRED in the last solve. The host keeps
 *  the ACTIVATIONS counter meaningful now that the loop is a driver phase rather than an
 *  orchestrator — it is the coverage guard that a test case really exercises an activation. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchMonitoringActivations(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nmon == 0 || !ctx->dMonActivated.p) {
        return 0;
    }
    size_t cnt = static_cast<size_t>(ctx->nmon) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dMonActivated.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jint fired = 0;
    for (double v : host) {
        if (!std::isnan(v)) {
            fired++;
        }
    }
    return fired;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchMonitoringTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nmon == 0 || !ctx->dMonActivated.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nmon) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dMonActivated.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(cnt));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}
