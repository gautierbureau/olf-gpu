// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: MonitoringVoltageOuterLoop.
#include "jni_common.cuh"
#include "../solver/scenario_install.cuh"

/** MonitoringVoltage controller table (OLF order 6). Empty array clears it. */
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, VoltageMonitoringNative, setBatchMonitoringLoops)(
        JNIEnv* env, jclass, jlong handle, jintArray jVRow, jdoubleArray jBand,
        jintArray jDeltaStart, jintArray jDeltaRow, jintArray jDeltaMode, jdoubleArray jDeltaTarget) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jVRow == nullptr || env->GetArrayLength(jVRow) == 0) {
        ctx->nMonitoringVoltageControllers = 0;
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
    upI(jVRow, ctx->dMonitoringVoltageBusVRow);
    upD(jBand, ctx->dMonitoringVoltageBand);
    upI(jDeltaStart, ctx->dMonitoringVoltageDeltaStart);
    upI(jDeltaRow, ctx->dMonitoringVoltageDeltaRow);
    upI(jDeltaMode, ctx->dMonitoringVoltageDeltaMode);
    upD(jDeltaTarget, ctx->dMonitoringVoltageDeltaTarget);
    ctx->nMonitoringVoltageControllers = env->GetArrayLength(jVRow);
    ctx->dMonitoringVoltageActivatedTargetV.alloc(static_cast<size_t>(ctx->nMonitoringVoltageControllers) * ctx->nScenarios * sizeof(double));
}

/** How many (controller, installed scenario) monitoring automatons FIRED in the last solve. The host keeps
 *  the ACTIVATIONS counter meaningful now that the loop is a driver phase rather than an
 *  orchestrator — it is the coverage guard that a test case really exercises an activation. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, VoltageMonitoringNative, readBatchMonitoringActivations)(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nMonitoringVoltageControllers == 0 || !ctx->dMonitoringVoltageActivatedTargetV.p) {
        return 0;
    }
    std::vector<double> host;
    readInstalledRows(ctx, ctx->dMonitoringVoltageActivatedTargetV.p, static_cast<size_t>(ctx->nMonitoringVoltageControllers), host);
    jint fired = 0;
    for (double v : host) {
        if (!std::isnan(v)) {
            fired++;
        }
    }
    return fired;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, VoltageMonitoringNative, readBatchMonitoringTargets)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nMonitoringVoltageControllers == 0 || !ctx->dMonitoringVoltageActivatedTargetV.p) {
        return nullptr;
    }
    std::vector<double> host;
    readInstalledRows(ctx, ctx->dMonitoringVoltageActivatedTargetV.p, static_cast<size_t>(ctx->nMonitoringVoltageControllers), host);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}
