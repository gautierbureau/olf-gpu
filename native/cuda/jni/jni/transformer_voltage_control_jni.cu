// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Transformer voltage control, incremental and discrete.
#include "jni_common.cuh"

/** Upload the incremental TVC controller data for the in-kernel outer loop.
 *  rhoRow/vRow/elemIndex: [nCtrl] int; targetV/halfDb: [nCtrl] double;
 *  tapPtr: [nCtrl+1] int (CSR into tapData); tapData: [totalTaps*8] double;
 *  baseTap: [nCtrl] int; groupPtr: [nGroups+1] int (CSR into groupCtrl);
 *  groupCtrl: [nCtrl] int (controller indices per group); maxTapShift: OLF's per-pass bound.
 *  Returns 1 when the kernel loop OWNS the controllers (shared controlled buses included — the
 *  one-tap-at-a-time round robin runs in tvcGroupDecisionBatch), 0 when there are none. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, setBatchTvcControllers)(
        JNIEnv* env, jclass, jlong handle,
        jintArray jRhoRow, jintArray jVRow, jdoubleArray jTargetV, jdoubleArray jHalfDb,
        jintArray jElemIndex, jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap,
        jintArray jGroupPtr, jintArray jGroupCtrl, jint maxTapShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nTvc = 0;
    if (jRhoRow == nullptr || env->GetArrayLength(jRhoRow) == 0) {
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
    std::vector<int> groupPtr;
    readI(jGroupPtr, groupPtr);                              // shared controlled buses run in-kernel too
    std::vector<int> vRow, groupCtrl;
    std::vector<double> targetV, halfDb;
    readI(jRhoRow, ctx->hTvcRhoRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfDb, halfDb);
    readI(jElemIndex, ctx->hTvcElemIndex);
    readI(jTapPtr, ctx->hTvcTapPtr);
    readD(jTapData, ctx->hTvcTapData);
    readI(jBaseTap, ctx->hTvcBaseTap);
    readI(jGroupCtrl, groupCtrl);
    int nc = static_cast<int>(ctx->hTvcRhoRow.size());
    ctx->dTvcRhoRow.upload(ctx->hTvcRhoRow.data(), nc * sizeof(int));
    ctx->dTvcVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->dTvcTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->dTvcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dTvcElemIndex.upload(ctx->hTvcElemIndex.data(), nc * sizeof(int));
    ctx->dTvcTapPtr.upload(ctx->hTvcTapPtr.data(), ctx->hTvcTapPtr.size() * sizeof(int));
    ctx->dTvcTapData.upload(ctx->hTvcTapData.data(), ctx->hTvcTapData.size() * sizeof(double));
    ctx->dTvcBaseTap.upload(ctx->hTvcBaseTap.data(), nc * sizeof(int));
    ctx->nTvcGroups = static_cast<int>(groupPtr.size()) - 1;
    ctx->dTvcGroupPtr.upload(groupPtr.data(), groupPtr.size() * sizeof(int));
    ctx->dTvcGroupCtrl.upload(groupCtrl.data(), groupCtrl.size() * sizeof(int));
    ctx->tvcMaxTapShift = maxTapShift > 0 ? maxTapShift : 1;
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dTvcCurTap.ensure(nst * sizeof(int));
    ctx->dTvcPrevTap.ensure(nst * sizeof(int));
    ctx->dTvcRatchetDir.ensure(nst * sizeof(int));
    ctx->dTvcRatchetCount.ensure(nst * sizeof(int));
    ctx->dTvcRatchetAllowed.ensure(nst * sizeof(int));
    ctx->nTvc = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, readBatchTvcTaps)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvc <= 0 || !ctx->dTvcCurTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nTvc) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dTvcCurTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Install the DISCRETE transformer voltage-control tables (OLF order 8): mode 0 = AFTER_GENERATOR
 *  (TransformerVoltageControlOuterLoop), 1 = WITH_GENERATOR (SimpleTransformerVoltageControlOuterLoop);
 *  entryOn = the uploaded activation already carries the control ON (a base load flow whose network
 *  OLF's initialize switched on). Groups = controlled buses (CSR over members), each with its toggle-
 *  diffed ON/OFF row deltas (CSR); members with rho row, element, base tap, tap table, r1 range and
 *  initial r1. Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, setBatchDiscreteTvc)(
        JNIEnv* env, jclass, jlong handle, jint mode, jint entryOn,
        jintArray jGrpPtr, jintArray jGrpVRow, jdoubleArray jGrpTargetV, jdoubleArray jGrpHalfDb, jintArray jGrpUseInit,
        jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaModeOn, jintArray jDeltaModeOff,
        jdoubleArray jDeltaTargetOn, jdoubleArray jDeltaTargetOff,
        jintArray jMemRhoRow, jintArray jMemElem, jintArray jMemBaseTap, jintArray jMemTapPtr,
        jdoubleArray jMemMin, jdoubleArray jMemMax, jdoubleArray jMemInit, jdoubleArray jTapData) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nTvd = 0;
    ctx->nTvdFz = 0;
    ctx->tvdFixTables = false;
    ctx->tvdFixWords = 0;
    if (jMemRhoRow == nullptr || env->GetArrayLength(jMemRhoRow) == 0) {
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
    std::vector<int> grpPtr, grpVRow, grpUseInit, deltaPtr, deltaRow, deltaModeOn, deltaModeOff, memRhoRow;
    std::vector<double> grpTargetV, grpHalfDb, deltaTargetOn, deltaTargetOff, memMin, memMax, memInit;
    readI(jGrpPtr, grpPtr);
    readI(jGrpVRow, grpVRow);
    readD(jGrpTargetV, grpTargetV);
    readD(jGrpHalfDb, grpHalfDb);
    readI(jGrpUseInit, grpUseInit);
    readI(jDeltaPtr, deltaPtr);
    readI(jDeltaRow, deltaRow);
    readI(jDeltaModeOn, deltaModeOn);
    readI(jDeltaModeOff, deltaModeOff);
    readD(jDeltaTargetOn, deltaTargetOn);
    readD(jDeltaTargetOff, deltaTargetOff);
    readI(jMemRhoRow, memRhoRow);
    readI(jMemElem, ctx->hTvdMemElem);
    readI(jMemBaseTap, ctx->hTvdMemBaseTap);
    readI(jMemTapPtr, ctx->hTvdMemTapPtr);
    readD(jMemMin, memMin);
    readD(jMemMax, memMax);
    readD(jMemInit, memInit);
    readD(jTapData, ctx->hTvdTapData);
    int nM = static_cast<int>(memRhoRow.size());
    int nG = static_cast<int>(grpPtr.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    for (int m = 0; m < nM; m++) {
        if (ctx->hTvdMemTapPtr[m + 1] - ctx->hTvdMemTapPtr[m] <= 0) {
            return 0;                                        // no tap table: the host orchestrator keeps the control
        }
    }
    ctx->dTvdGrpPtr.upload(grpPtr.data(), grpPtr.size() * sizeof(int));
    ctx->dTvdGrpVRow.upload(grpVRow.data(), nG * sizeof(int));
    ctx->dTvdGrpTargetV.upload(grpTargetV.data(), nG * sizeof(double));
    ctx->dTvdGrpHalfDb.upload(grpHalfDb.data(), nG * sizeof(double));
    ctx->dTvdGrpUseInit.upload(grpUseInit.data(), nG * sizeof(int));
    ctx->dTvdDeltaPtr.upload(deltaPtr.data(), deltaPtr.size() * sizeof(int));
    ctx->dTvdDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->dTvdDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->dTvdMemRhoRow.upload(memRhoRow.data(), nM * sizeof(int));
    ctx->dTvdMemElem.upload(ctx->hTvdMemElem.data(), nM * sizeof(int));
    ctx->dTvdMemBaseTap.upload(ctx->hTvdMemBaseTap.data(), nM * sizeof(int));
    ctx->dTvdMemTapPtr.upload(ctx->hTvdMemTapPtr.data(), ctx->hTvdMemTapPtr.size() * sizeof(int));
    ctx->dTvdMemMin.upload(memMin.data(), nM * sizeof(double));
    ctx->dTvdMemMax.upload(memMax.data(), nM * sizeof(double));
    ctx->dTvdMemInit.upload(memInit.data(), nM * sizeof(double));
    ctx->dTvdTapData.upload(ctx->hTvdTapData.data(), ctx->hTvdTapData.size() * sizeof(double));
    size_t nS = ctx->nScenarios;
    ctx->dTvdScnStep.ensure(nS * sizeof(int));
    ctx->dTvdGrpStep.ensure(nS * nG * sizeof(int));
    ctx->dTvdGrpShMin.ensure(nS * nG * sizeof(double));
    ctx->dTvdGrpShMax.ensure(nS * nG * sizeof(double));
    ctx->dTvdGrpShInit.ensure(nS * nG * sizeof(double));
    ctx->dTvdMemState.ensure(nS * nM * sizeof(int));
    ctx->dTvdMemTap.ensure(nS * nM * sizeof(int));
    ctx->dTvdMemRoundFrom.ensure(nS * nM * sizeof(double));
    ctx->dTvdSaved.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->dTvdReject.ensure(nS * sizeof(char));
    ctx->nTvdGroups = nG;
    ctx->nTvdDelta = nD;
    ctx->tvdMode = mode;
    ctx->tvdEntryOn = entryOn;
    ctx->nTvd = nM;
    return 1;
}

/** GeneratorVoltageControlManager tables for the AFTER_GENERATOR phase installed by setBatchDiscreteTvc:
 *  the freeze candidates (controller bus V row, load Q) and fixTransformerVoltageControls' per-member
 *  PV-candidate row lists (CSR, [nM+1]; null = no connectivity fix). Returns 1 when installed. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, setBatchDiscreteTvcFreeze)(
        JNIEnv* env, jclass, jlong handle, jintArray jFzVRow, jdoubleArray jFzLoadQ, jintArray jFzRlEntry,
        jintArray jFixPvRow) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> vRow, fixRow;
    std::vector<double> loadQ;
    if (jFzVRow != nullptr) {
        readI(jFzVRow, vRow);
        loadQ.resize(env->GetArrayLength(jFzLoadQ));
        env->GetDoubleArrayRegion(jFzLoadQ, 0, static_cast<jsize>(loadQ.size()), loadQ.data());
    }
    int nFz = static_cast<int>(vRow.size());
    if (nFz > 0) {
        std::vector<int> rlEntry;
        if (jFzRlEntry != nullptr) {
            readI(jFzRlEntry, rlEntry);
        }
        if (static_cast<int>(rlEntry.size()) != nFz) {
            rlEntry.assign(nFz, -1);                         // no remote candidate
        }
        ctx->dTvdFzVRow.upload(vRow.data(), nFz * sizeof(int));
        ctx->dTvdFzLoadQ.upload(loadQ.data(), nFz * sizeof(double));
        ctx->dTvdFzRlEntry.upload(rlEntry.data(), nFz * sizeof(int));
        size_t nsc = static_cast<size_t>(ctx->nScenarios) * nFz;
        ctx->dTvdFzFrozen.ensure(nsc * sizeof(int));
        ctx->dTvdFzSavedMode.ensure(nsc * sizeof(int));
        ctx->dTvdFzSavedTarget.ensure(nsc * sizeof(double));
    }
    ctx->nTvdFz = nFz;
    ctx->tvdFixTables = false;                               // armed by setBatchDiscreteTvcFixMasks
    ctx->tvdFixWords = 0;
    if (jFixPvRow != nullptr) {
        readI(jFixPvRow, fixRow);
        if (!fixRow.empty()) {
            ctx->dTvdFixPvRow.upload(fixRow.data(), fixRow.size() * sizeof(int));
        }
    }
    return 1;
}

// The per-(scenario, member) connectivity masks of fixTransformerVoltageControls. Uploaded per CHUNK,
// unlike the tables above: OLF re-runs the check on the POST-CONTINGENCY network, so the component of a
// not-controlled side is a property of the scenario, not of the run.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, setBatchDiscreteTvcFixMasks)(
        JNIEnv* env, jclass, jlong handle, jintArray jMin, jintArray jMax, jint words) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0) {
        return;
    }
    ctx->tvdFixTables = false;
    ctx->tvdFixWords = 0;
    if (jMin == nullptr || jMax == nullptr || words <= 0) {
        return;
    }
    size_t want = static_cast<size_t>(ctx->nScenarios) * ctx->nTvd * words;
    if (static_cast<size_t>(env->GetArrayLength(jMin)) != want
            || static_cast<size_t>(env->GetArrayLength(jMax)) != want) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchDiscreteTvcFixMasks: array length != nScenarios * nMembers * words");
        return;
    }
    std::vector<int> mn(want), mx(want);
    env->GetIntArrayRegion(jMin, 0, static_cast<jsize>(want), mn.data());
    env->GetIntArrayRegion(jMax, 0, static_cast<jsize>(want), mx.data());
    ctx->dTvdFixMinMask.upload(mn.data(), want * sizeof(int));
    ctx->dTvdFixMaxMask.upload(mx.data(), want * sizeof(int));
    ctx->tvdFixWords = words;
    ctx->tvdFixTables = true;
}

/** Per (scenario, freeze candidate): {frozen (0/1), the row target while frozen (gen − load)} — layout
 *  {@code (s * nFz + c) * 2}; null without candidates. */
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, readBatchDiscreteTvcFreeze)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0 || ctx->nTvdFz <= 0 || !ctx->dTvdFzFrozen.p) {
        return nullptr;
    }
    int n = ctx->n;
    size_t cnt = static_cast<size_t>(ctx->nTvdFz) * ctx->nScenarios;
    std::vector<int> fz(cnt), vrow(ctx->nTvdFz);
    std::vector<double> tgt(static_cast<size_t>(n) * ctx->nScenarios), out(cnt * 2);
    cudaMemcpy(fz.data(), ctx->dTvdFzFrozen.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vrow.data(), ctx->dTvdFzVRow.p, vrow.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tgt.data(), ctx->dTarget.p, tgt.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (int s = 0; s < ctx->nScenarios; s++) {
        for (int c = 0; c < ctx->nTvdFz; c++) {
            size_t i = static_cast<size_t>(s) * ctx->nTvdFz + c;
            out[i * 2] = fz[i];
            out[i * 2 + 1] = tgt[static_cast<size_t>(s) * n + vrow[c]];
        }
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Per (scenario, member): {state (0 off, 1 continuous, 2 fixed), tap index, the r1 fed to the
 *  rounding} — layout {@code (s * nM + m) * 3}; null without the table. */
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, readBatchDiscreteTvcState)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0 || !ctx->dTvdMemState.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nTvd) * ctx->nScenarios;
    std::vector<int> st(cnt), tap(cnt);
    std::vector<double> rf(cnt), out(cnt * 3);
    cudaMemcpy(st.data(), ctx->dTvdMemState.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tap.data(), ctx->dTvdMemTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(rf.data(), ctx->dTvdMemRoundFrom.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < cnt; i++) {
        out[i * 3] = st[i];
        out[i * 3 + 1] = tap[i];
        out[i * 3 + 2] = rf[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}
