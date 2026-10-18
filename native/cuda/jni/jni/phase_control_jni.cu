// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Phase control and transformer reactive power control, incremental and continuous.
#include "jni_common.cuh"

/** Upload the FLOW-OBSERVED incremental controllers (RPC / incremental phase). Per controller: kind
 *  (0 RPC, 1 phase CONTROLLER, 2 phase LIMITER), pinRow (the rho/a1 identity row), pinCol (6 r1 / 7 a1),
 *  side (1/2) of the controlled flow, targetValue, halfDb, elem (controller's global element),
 *  ctlElem (controlled branch's global element — a closed branch), tapPtr/tapData ([y g1 g2 b1 b2 ksi r1
 *  a1] per tap), baseTap, maxShift. Returns 1 when the kernel loops own them. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, PhaseControlNative, setBatchFlowControllers)(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jPinRow, jintArray jPinCol, jintArray jSide, jdoubleArray jTargetValue, jdoubleArray jHalfDb,
        jintArray jElem, jintArray jCtlElem, jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap, jintArray jMaxShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nFlc = 0;
    ctx->nFlcPhase = 0;
    ctx->nFlcRpc = 0;
    if (jKind == nullptr || env->GetArrayLength(jKind) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> pinCol, side, ctlElem, maxShift;
    std::vector<double> targetValue, halfDb;
    readI(jKind, ctx->hFlcKind);
    readI(jPinRow, ctx->hFlcPinRow);
    readI(jPinCol, pinCol);
    readI(jSide, side);
    readD(jTargetValue, targetValue);
    readD(jHalfDb, halfDb);
    readI(jElem, ctx->hFlcElem);
    readI(jCtlElem, ctlElem);
    readI(jTapPtr, ctx->hFlcTapPtr);
    readD(jTapData, ctx->hFlcTapData);
    readI(jBaseTap, ctx->hFlcBaseTap);
    readI(jMaxShift, maxShift);
    int nc = static_cast<int>(ctx->hFlcKind.size());
    std::vector<int> ctlPack(nc);
    for (int c = 0; c < nc; c++) {
        ctlPack[c] = ctlElem[c] - ctx->cbOffset;             // the controlled branch's index in the closed-branch pack
        if (ctlPack[c] < 0 || ctlPack[c] >= ctx->st->cb.ne) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
        if (ctx->hFlcKind[c] == 0) ctx->nFlcRpc++; else ctx->nFlcPhase++;
    }
    ctx->dFlcKind.upload(ctx->hFlcKind.data(), nc * sizeof(int));
    ctx->dFlcPinRow.upload(ctx->hFlcPinRow.data(), nc * sizeof(int));
    ctx->dFlcPinCol.upload(pinCol.data(), nc * sizeof(int));
    ctx->dFlcSide.upload(side.data(), nc * sizeof(int));
    ctx->dFlcTargetValue.upload(targetValue.data(), nc * sizeof(double));
    ctx->dFlcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dFlcElem.upload(ctx->hFlcElem.data(), nc * sizeof(int));
    ctx->dFlcCtlPack.upload(ctlPack.data(), nc * sizeof(int));
    ctx->dFlcTapPtr.upload(ctx->hFlcTapPtr.data(), ctx->hFlcTapPtr.size() * sizeof(int));
    ctx->dFlcTapData.upload(ctx->hFlcTapData.data(), ctx->hFlcTapData.size() * sizeof(double));
    ctx->dFlcBaseTap.upload(ctx->hFlcBaseTap.data(), nc * sizeof(int));
    ctx->dFlcMaxShift.upload(maxShift.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dFlcCurTap.ensure(nst * sizeof(int));
    ctx->dFlcPrevTap.ensure(nst * sizeof(int));
    ctx->dFlcRatDir.ensure(nst * sizeof(int));
    ctx->dFlcRatCount.ensure(nst * sizeof(int));
    ctx->dFlcRatAllowed.ensure(nst * sizeof(int));
    ctx->dFlcSensi.ensure(nst * sizeof(double));
    ctx->dFlcDiff.ensure(nst * sizeof(double));
    ctx->dFlcOutOfDb.ensure(nst * sizeof(char));
    ctx->nFlc = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, PhaseControlNative, readBatchFlowTaps)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nFlc <= 0 || !ctx->dFlcCurTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nFlc) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dFlcCurTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Upload the CONTINUOUS phase controllers (kind 0 CONTROLLER / 1 LIMITER, a1 identity row, controlled
 *  side, current limit, global element, tap rows, base tap). Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, PhaseControlNative, setBatchContinuousPhase)(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jA1Row, jintArray jSide, jdoubleArray jLimit, jintArray jElem,
        jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nCph = 0;
    if (jKind == nullptr || env->GetArrayLength(jKind) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> kind, a1Row, side;
    std::vector<double> limit;
    readI(jKind, kind);
    readI(jA1Row, a1Row);
    readI(jSide, side);
    readD(jLimit, limit);
    readI(jElem, ctx->hCphElem);
    readI(jTapPtr, ctx->hCphTapPtr);
    readD(jTapData, ctx->hCphTapData);
    readI(jBaseTap, ctx->hCphBaseTap);
    int nc = static_cast<int>(kind.size());
    for (int c = 0; c < nc; c++) {
        int e = ctx->hCphElem[c] - ctx->cbOffset;
        if (e < 0 || e >= ctx->st->cb.ne) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
    }
    ctx->dCphKind.upload(kind.data(), nc * sizeof(int));
    ctx->dCphA1Row.upload(a1Row.data(), nc * sizeof(int));
    ctx->dCphSide.upload(side.data(), nc * sizeof(int));
    ctx->dCphLimit.upload(limit.data(), nc * sizeof(double));
    ctx->dCphElem.upload(ctx->hCphElem.data(), nc * sizeof(int));
    ctx->dCphTapPtr.upload(ctx->hCphTapPtr.data(), ctx->hCphTapPtr.size() * sizeof(int));
    ctx->dCphTapData.upload(ctx->hCphTapData.data(), ctx->hCphTapData.size() * sizeof(double));
    ctx->dCphBaseTap.upload(ctx->hCphBaseTap.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dCphTap.ensure(nst * sizeof(int));
    ctx->dCphPrevTap.ensure(nst * sizeof(int));
    ctx->dCphIter.ensure(static_cast<size_t>(ctx->nScenarios) * sizeof(int));
    ctx->nCph = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, PhaseControlNative, readBatchContinuousPhaseTaps)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nCph <= 0 || !ctx->dCphTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nCph) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dCphTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}
