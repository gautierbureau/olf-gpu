// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: the outer-loop driver's run settings.
#include "jni_common.cuh"

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
