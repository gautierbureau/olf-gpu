// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: The outer-loop driver: host-owned loops and the pause/resume cycle.
#include "jni_common.cuh"

/** Enable HOST-owned outer-loop slots via a bitmask.
 *  Bit 0 = SecondaryVoltageControl (OLF order 4, pre-RL),
 *  Bit 1 = PhaseControl (OLF order 7, post-RL),
 *  Bit 2 = TransformerVoltageControl (OLF order 8),
 *  Bit 3 = TransformerReactivePowerControl (OLF order 9),
 *  Bit 4 = ShuntVoltageControl (OLF order 10).
 *  0 disables all host loops; the driver behaves exactly as if the mechanism did not exist. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchHostLoop(
        JNIEnv*, jclass, jlong handle, jint mask) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->hostLoopMask = mask;
    }
}

/** Did the last solve stop at a host-owned slot? Returns 0 if not paused, or the
 *  DeviceLoopKind value (LOOP_HOST_SVC=5, LOOP_HOST_PHASE=6, ...) of the paused slot. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDriverPaused(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    return (ctx == nullptr || !ctx->driverPaused) ? 0 : ctx->pausedLoopKind;
}

/** Resume the paused driver: {@code hostChanged[s] != 0} is the host loop's UNSTABLE verdict for
 *  that scenario, which drives the next pass exactly as a device loop's verdict would. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDriverResume(
        JNIEnv* env, jclass, jlong handle, jbyteArray jHostChanged) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hostChanged.assign(ctx->nScenarios, 0);
    if (jHostChanged != nullptr) {
        jsize len = std::min<jsize>(env->GetArrayLength(jHostChanged), ctx->nScenarios);
        env->GetByteArrayRegion(jHostChanged, 0, len, reinterpret_cast<jbyte*>(ctx->hostChanged.data()));
    }
    ctx->driverResume = 1;
}

/** Drop a paused driver sequence without resuming it: the host loop changed the equation layout or
 *  the row modes, so the next solveBatchContext must START OVER on the re-extracted system. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_cancelBatchDriverPause(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->driverPaused = 0;
    ctx->driverResume = 0;
    ctx->pausedLoopKind = 0;
}

/** The next FRESH solve resumes its sweep at this loop kind after its first solve (a restarted
 *  sequence: OLF re-checks the loop that went UNSTABLE before restarting the sweep). The context may
 *  have been rebuilt since the pause, so the Java side carries the kind across. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDriverStartLoop(
        JNIEnv*, jclass, jlong handle, jint kind) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->startLoopKind = kind;
}
