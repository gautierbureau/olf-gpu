// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: HVDC AC emulation loops.
#include "jni_common.cuh"
#include "../solver/scenario_install.cuh"
#include "slot_table_install.cuh"

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
        ctx->hvdcAcEmulation.nHvdcAcEmulation = 0;
        return;
    }
    int nh = env->GetArrayLength(jElem);
    auto upI = [&](jintArray a, DeviceBuf& b) {
        const JniInts q(env, a);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(int));
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& b) {
        const JniDoubles q(env, a);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(double));
    };
    upI(jElem, ctx->hvdcAcEmulation.dHvdcElement);
    upI(jHvdcBus1PRow, ctx->hvdcAcEmulation.dHvdcBus1PRow);
    upI(jHvdcBus2PRow, ctx->hvdcAcEmulation.dHvdcBus2PRow);
    upI(jHvdcBus1PhiColumn, ctx->hvdcAcEmulation.dHvdcBus1PhiColumn);
    upI(jHvdcBus2PhiColumn, ctx->hvdcAcEmulation.dHvdcBus2PhiColumn);
    upD(jIn, ctx->hvdcAcEmulation.dHvdcAcEmulationParameters);
    // the scenarios' starting state, per (installed slot, hvdc): the HVDC phases write it on the device as they go
    ctx->hvdcAcEmulation.nHvdcAcEmulation = nh;
    const char* who = "setBatchHvdcLoops";
    installSlotRows(env, ctx, ctx->hvdcAcEmulation.hvdcAcEmulationStatus, nh, jHvdcAcEmulationStatus, who);
    installSlotRows(env, ctx, ctx->hvdcAcEmulation.hvdcAppliedDelta1, nh, jHvdcAppliedDelta1, who);
    installSlotRows(env, ctx, ctx->hvdcAcEmulation.hvdcAppliedDelta2, nh, jHvdcAppliedDelta2, who);
    installSlotRows(env, ctx, ctx->hvdcAcEmulation.hvdcContingencyOutage, nh, jHvdcContingencyOutage, who);
    ctx->hvdcAcEmulation.hvdcAcEmulationLimitsActive = limitsActive;
}

/** Read the HVDC statuses back after a solve, so the host keeps its own view in step. */
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, HvdcNative, readBatchHvdcStatus)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->hvdcAcEmulation.nHvdcAcEmulation == 0) {
        return nullptr;
    }
    std::vector<int> host;
    if (readInstalledRows(ctx, ctx->hvdcAcEmulation.hvdcAcEmulationStatus.device(), static_cast<size_t>(ctx->hvdcAcEmulation.nHvdcAcEmulation), host) != cudaSuccess) {
        return nullptr;
    }
    jsize len = static_cast<jsize>(host.size());
    jintArray out = env->NewIntArray(len);
    env->SetIntArrayRegion(out, 0, len, host.data());
    return out;
}
