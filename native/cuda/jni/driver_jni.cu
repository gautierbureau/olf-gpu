// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: The outer-loop driver: host-owned loops and the pause/resume cycle.
#include "jni_common.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

/** Enable HOST-owned outer-loop slots via a bitmask.
 *  Bit 0 = SecondaryVoltageControl (OLF order 4, pre-RL),
 *  Bit 1 = PhaseControl (OLF order 7, post-RL),
 *  Bit 2 = TransformerVoltageControl (OLF order 8),
 *  Bit 3 = TransformerReactivePowerControl (OLF order 9),
 *  Bit 4 = ShuntVoltageControl (OLF order 10).
 *  0 disables all host loops; the driver behaves exactly as if the mechanism did not exist. */
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, DriverNative, setBatchHostLoop)(
        JNIEnv*, jclass, jlong handle, jint mask) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->hostLoopMask = mask;
    }
}

/** The run's own outer loops: does its loop list hold OLF's DistributedSlackOuterLoop, its
 *  ReactiveLimitsOuterLoop? The driver installs the DISTRIBUTED_SLACK / REACTIVE_LIMITS phases only for
 *  a loop the run has (and a table to drive it). */
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, DriverNative, setBatchOuterLoops)(
        JNIEnv*, jclass, jlong handle, jboolean distributedSlackOuterLoop, jboolean reactiveLimitsOuterLoop) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->distributedSlackOuterLoop = distributedSlackOuterLoop == JNI_TRUE;
        ctx->reactiveLimitsOuterLoop = reactiveLimitsOuterLoop == JNI_TRUE;
    }
}

/** Did the last solve stop at a host-owned slot? Returns GpuOuterLoopKind::NOT_PAUSED if not paused, or the
 *  GpuOuterLoopKind::HOST_* kind of the paused slot. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_solver_gpu, DriverNative, readBatchDriverPaused)(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    return (ctx == nullptr || !ctx->driverPaused) ? GpuOuterLoopKind::NOT_PAUSED : ctx->pausedLoopKind;
}

/** Resume the paused driver: {@code hostChanged[s] != 0} is the host loop's UNSTABLE verdict for
 *  that scenario, which drives the next pass exactly as a device loop's verdict would. */
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, DriverNative, setBatchDriverResume)(
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
OLF_JNI(ac_solver_gpu, DriverNative, cancelBatchDriverPause)(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->driverPaused = 0;
    ctx->driverResume = 0;
    ctx->pausedLoopKind = GpuOuterLoopKind::NOT_PAUSED;
}

/** The next FRESH solve resumes its sweep at this loop kind after its first solve (a restarted
 *  sequence: OLF re-checks the loop that went UNSTABLE before restarting the sweep). The context may
 *  have been rebuilt since the pause, so the Java side carries the kind across. */
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, DriverNative, setBatchDriverStartLoop)(
        JNIEnv*, jclass, jlong handle, jint kind) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->startLoopKind = kind;
}
