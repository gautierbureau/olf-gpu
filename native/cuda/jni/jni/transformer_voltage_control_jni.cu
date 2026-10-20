// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Transformer voltage control, incremental and discrete.
#include "jni_common.cuh"

/** Upload the incremental TVC controller data for the in-kernel outer loop.
 *  rhoRow/vRow/elemIndex: [nControllers] int; targetV/halfTargetDeadband: [nControllers] double;
 *  tapStart: [nControllers+1] int (CSR into tapRows); tapRows: [totalTaps*8] double;
 *  baseTap: [nControllers] int; groupControllerStart: [nGroups+1] int (CSR into groupController);
 *  groupController: [nControllers] int (controller indices per group); maxTapShift: OLF's per-pass bound.
 *  Returns 1 when the kernel loop OWNS the controllers (shared controlled buses included — the
 *  one-tap-at-a-time round robin runs in incrementalTransformerVoltageControlGroupDecisionBatch), 0 when there are none. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, setBatchIncrementalTransformerVoltageControl)(
        JNIEnv* env, jclass, jlong handle,
        jintArray jRhoRow, jintArray jVRow, jdoubleArray jTargetV, jdoubleArray jHalfTargetDeadband,
        jintArray jControllerElement, jintArray jTapStart, jdoubleArray jTapRows, jintArray jBaseTap,
        jintArray jGroupControllerStart, jintArray jGroupController, jint maxTapShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nIncrementalTransformerVoltageControlControllerBranches = 0;
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
    std::vector<int> groupControllerStart;
    readI(jGroupControllerStart, groupControllerStart);                              // shared controlled buses run in-kernel too
    std::vector<int> vRow, groupController;
    std::vector<double> targetV, halfTargetDeadband;
    readI(jRhoRow, ctx->hIncrementalTransformerVoltageControlRhoRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfTargetDeadband, halfTargetDeadband);
    readI(jControllerElement, ctx->hIncrementalTransformerVoltageControlElement);
    readI(jTapStart, ctx->hIncrementalTransformerVoltageControlTapStart);
    readD(jTapRows, ctx->hIncrementalTransformerVoltageControlTapRows);
    readI(jBaseTap, ctx->hIncrementalTransformerVoltageControlBaseTap);
    readI(jGroupController, groupController);
    int nc = static_cast<int>(ctx->hIncrementalTransformerVoltageControlRhoRow.size());
    ctx->dIncrementalTransformerVoltageControlRhoRow.upload(ctx->hIncrementalTransformerVoltageControlRhoRow.data(), nc * sizeof(int));
    ctx->dIncrementalTransformerVoltageControlControlledBusVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->dIncrementalTransformerVoltageControlTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->dIncrementalTransformerVoltageControlHalfTargetDeadband.upload(halfTargetDeadband.data(), nc * sizeof(double));
    ctx->dIncrementalTransformerVoltageControlElement.upload(ctx->hIncrementalTransformerVoltageControlElement.data(), nc * sizeof(int));
    ctx->dIncrementalTransformerVoltageControlTapStart.upload(ctx->hIncrementalTransformerVoltageControlTapStart.data(), ctx->hIncrementalTransformerVoltageControlTapStart.size() * sizeof(int));
    ctx->dIncrementalTransformerVoltageControlTapRows.upload(ctx->hIncrementalTransformerVoltageControlTapRows.data(), ctx->hIncrementalTransformerVoltageControlTapRows.size() * sizeof(double));
    ctx->dIncrementalTransformerVoltageControlBaseTap.upload(ctx->hIncrementalTransformerVoltageControlBaseTap.data(), nc * sizeof(int));
    ctx->nIncrementalTransformerVoltageControlGroups = static_cast<int>(groupControllerStart.size()) - 1;
    ctx->dIncrementalTransformerVoltageControlGroupControllerStart.upload(groupControllerStart.data(), groupControllerStart.size() * sizeof(int));
    ctx->dIncrementalTransformerVoltageControlGroupController.upload(groupController.data(), groupController.size() * sizeof(int));
    ctx->incrementalTransformerVoltageControlMaxTapShift = maxTapShift > 0 ? maxTapShift : 1;
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dIncrementalTransformerVoltageControlCurrentTap.ensure(nst * sizeof(int));
    ctx->dIncrementalTransformerVoltageControlAppliedTap.ensure(nst * sizeof(int));
    ctx->dIncrementalTransformerVoltageControlCurrentDirection.ensure(nst * sizeof(int));
    ctx->dIncrementalTransformerVoltageControlDirectionChangeCount.ensure(nst * sizeof(int));
    ctx->dIncrementalTransformerVoltageControlAllowedDirection.ensure(nst * sizeof(int));
    ctx->nIncrementalTransformerVoltageControlControllerBranches = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, readBatchIncrementalTransformerVoltageControlTaps)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nIncrementalTransformerVoltageControlControllerBranches <= 0 || !ctx->dIncrementalTransformerVoltageControlCurrentTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nIncrementalTransformerVoltageControlControllerBranches) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dIncrementalTransformerVoltageControlCurrentTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
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
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, setBatchTransformerVoltageControl)(
        JNIEnv* env, jclass, jlong handle, jint mode, jint entryOn,
        jintArray jGroupMemberStart, jintArray jGroupControlledBusVRow, jdoubleArray jGroupTargetV, jdoubleArray jGroupHalfTargetDeadband, jintArray jGroupUseInitialTapPosition,
        jintArray jDeltaStart, jintArray jDeltaRow, jintArray jDeltaModeOn, jintArray jDeltaModeOff,
        jdoubleArray jDeltaTargetOn, jdoubleArray jDeltaTargetOff,
        jintArray jMemberRhoRow, jintArray jMemberElement, jintArray jMemberBaseTap, jintArray jMemberTapStart,
        jdoubleArray jMemberMinR1, jdoubleArray jMemberMaxR1, jdoubleArray jMemberInitialR1, jdoubleArray jTapRows) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nTransformerVoltageControlMembers = 0;
    ctx->nGeneratorVoltageControlFreezeCandidates = 0;
    ctx->hasTransformerVoltageControlFixMasks = false;
    ctx->transformerVoltageControlFixWords = 0;
    if (jMemberRhoRow == nullptr || env->GetArrayLength(jMemberRhoRow) == 0) {
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
    std::vector<int> groupMemberStart, groupControlledBusVRow, groupUseInitialTapPosition, deltaStart, deltaRow, deltaModeOn, deltaModeOff, memberRhoRow;
    std::vector<double> groupTargetV, groupHalfTargetDeadband, deltaTargetOn, deltaTargetOff, memberMinR1, memberMaxR1, memberInitialR1;
    readI(jGroupMemberStart, groupMemberStart);
    readI(jGroupControlledBusVRow, groupControlledBusVRow);
    readD(jGroupTargetV, groupTargetV);
    readD(jGroupHalfTargetDeadband, groupHalfTargetDeadband);
    readI(jGroupUseInitialTapPosition, groupUseInitialTapPosition);
    readI(jDeltaStart, deltaStart);
    readI(jDeltaRow, deltaRow);
    readI(jDeltaModeOn, deltaModeOn);
    readI(jDeltaModeOff, deltaModeOff);
    readD(jDeltaTargetOn, deltaTargetOn);
    readD(jDeltaTargetOff, deltaTargetOff);
    readI(jMemberRhoRow, memberRhoRow);
    readI(jMemberElement, ctx->hTransformerVoltageControlMemberElement);
    readI(jMemberBaseTap, ctx->hTransformerVoltageControlMemberBaseTap);
    readI(jMemberTapStart, ctx->hTransformerVoltageControlMemberTapStart);
    readD(jMemberMinR1, memberMinR1);
    readD(jMemberMaxR1, memberMaxR1);
    readD(jMemberInitialR1, memberInitialR1);
    readD(jTapRows, ctx->hTransformerVoltageControlTapRows);
    int nM = static_cast<int>(memberRhoRow.size());
    int nG = static_cast<int>(groupMemberStart.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    for (int m = 0; m < nM; m++) {
        if (ctx->hTransformerVoltageControlMemberTapStart[m + 1] - ctx->hTransformerVoltageControlMemberTapStart[m] <= 0) {
            return 0;                                        // no tap table: the host orchestrator keeps the control
        }
    }
    ctx->dTransformerVoltageControlGroupMemberStart.upload(groupMemberStart.data(), groupMemberStart.size() * sizeof(int));
    ctx->dTransformerVoltageControlGroupControlledBusVRow.upload(groupControlledBusVRow.data(), nG * sizeof(int));
    ctx->dTransformerVoltageControlGroupTargetV.upload(groupTargetV.data(), nG * sizeof(double));
    ctx->dTransformerVoltageControlGroupHalfTargetDeadband.upload(groupHalfTargetDeadband.data(), nG * sizeof(double));
    ctx->dTransformerVoltageControlGroupUseInitialTapPosition.upload(groupUseInitialTapPosition.data(), nG * sizeof(int));
    ctx->dTransformerVoltageControlDeltaStart.upload(deltaStart.data(), deltaStart.size() * sizeof(int));
    ctx->dTransformerVoltageControlDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTransformerVoltageControlDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTransformerVoltageControlDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTransformerVoltageControlDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->dTransformerVoltageControlDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->dTransformerVoltageControlMemberRhoRow.upload(memberRhoRow.data(), nM * sizeof(int));
    ctx->dTransformerVoltageControlMemberElement.upload(ctx->hTransformerVoltageControlMemberElement.data(), nM * sizeof(int));
    ctx->dTransformerVoltageControlMemberBaseTap.upload(ctx->hTransformerVoltageControlMemberBaseTap.data(), nM * sizeof(int));
    ctx->dTransformerVoltageControlMemberTapStart.upload(ctx->hTransformerVoltageControlMemberTapStart.data(), ctx->hTransformerVoltageControlMemberTapStart.size() * sizeof(int));
    ctx->dTransformerVoltageControlMemberMinR1.upload(memberMinR1.data(), nM * sizeof(double));
    ctx->dTransformerVoltageControlMemberMaxR1.upload(memberMaxR1.data(), nM * sizeof(double));
    ctx->dTransformerVoltageControlMemberInitialR1.upload(memberInitialR1.data(), nM * sizeof(double));
    ctx->dTransformerVoltageControlTapRows.upload(ctx->hTransformerVoltageControlTapRows.data(), ctx->hTransformerVoltageControlTapRows.size() * sizeof(double));
    size_t nS = ctx->nScenarios;
    ctx->dTransformerVoltageControlStep.ensure(nS * sizeof(int));
    ctx->dTransformerVoltageControlGroupStep.ensure(nS * nG * sizeof(int));
    ctx->dTransformerVoltageControlGroupSharedMinR1.ensure(nS * nG * sizeof(double));
    ctx->dTransformerVoltageControlGroupSharedMaxR1.ensure(nS * nG * sizeof(double));
    ctx->dTransformerVoltageControlGroupSharedInitialR1.ensure(nS * nG * sizeof(double));
    ctx->dTransformerVoltageControlMemberState.ensure(nS * nM * sizeof(int));
    ctx->dTransformerVoltageControlMemberTap.ensure(nS * nM * sizeof(int));
    ctx->dTransformerVoltageControlMemberRoundingR1.ensure(nS * nM * sizeof(double));
    ctx->dTransformerVoltageControlSavedDeltaTarget.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->dTransformerVoltageControlRejected.ensure(nS * sizeof(char));
    ctx->nTransformerVoltageControlGroups = nG;
    ctx->nTransformerVoltageControlDeltaRows = nD;
    ctx->transformerVoltageControlMode = mode;
    ctx->transformerVoltageControlEntryOn = entryOn;
    ctx->nTransformerVoltageControlMembers = nM;
    return 1;
}

/** GeneratorVoltageControlManager tables for the AFTER_GENERATOR phase installed by setBatchTransformerVoltageControl:
 *  the freeze candidates (controller bus V row, load Q) and fixTransformerVoltageControls' per-member
 *  PV-candidate row lists (CSR, [nM+1]; null = no connectivity fix). Returns 1 when installed. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, setBatchGeneratorVoltageControlFreeze)(
        JNIEnv* env, jclass, jlong handle, jintArray jGeneratorVoltageControlFreezeVRow, jdoubleArray jGeneratorVoltageControlFreezeLoadTargetQ, jintArray jGeneratorVoltageControlFreezeReactiveLimitEntry,
        jintArray jFixPvRow) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTransformerVoltageControlMembers <= 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> vRow, fixRow;
    std::vector<double> loadQ;
    if (jGeneratorVoltageControlFreezeVRow != nullptr) {
        readI(jGeneratorVoltageControlFreezeVRow, vRow);
        loadQ.resize(env->GetArrayLength(jGeneratorVoltageControlFreezeLoadTargetQ));
        env->GetDoubleArrayRegion(jGeneratorVoltageControlFreezeLoadTargetQ, 0, static_cast<jsize>(loadQ.size()), loadQ.data());
    }
    int nGeneratorVoltageControlFreezeCandidates = static_cast<int>(vRow.size());
    if (nGeneratorVoltageControlFreezeCandidates > 0) {
        std::vector<int> reactiveLimitEntry;
        if (jGeneratorVoltageControlFreezeReactiveLimitEntry != nullptr) {
            readI(jGeneratorVoltageControlFreezeReactiveLimitEntry, reactiveLimitEntry);
        }
        if (static_cast<int>(reactiveLimitEntry.size()) != nGeneratorVoltageControlFreezeCandidates) {
            reactiveLimitEntry.assign(nGeneratorVoltageControlFreezeCandidates, -1);                         // no remote candidate
        }
        ctx->dGeneratorVoltageControlFreezeVRow.upload(vRow.data(), nGeneratorVoltageControlFreezeCandidates * sizeof(int));
        ctx->dGeneratorVoltageControlFreezeLoadTargetQ.upload(loadQ.data(), nGeneratorVoltageControlFreezeCandidates * sizeof(double));
        ctx->dGeneratorVoltageControlFreezeReactiveLimitEntry.upload(reactiveLimitEntry.data(), nGeneratorVoltageControlFreezeCandidates * sizeof(int));
        size_t nsc = static_cast<size_t>(ctx->nScenarios) * nGeneratorVoltageControlFreezeCandidates;
        ctx->dGeneratorVoltageControlFrozen.ensure(nsc * sizeof(int));
        ctx->dGeneratorVoltageControlFreezeSavedMode.ensure(nsc * sizeof(int));
        ctx->dGeneratorVoltageControlFreezeSavedTarget.ensure(nsc * sizeof(double));
    }
    ctx->nGeneratorVoltageControlFreezeCandidates = nGeneratorVoltageControlFreezeCandidates;
    ctx->hasTransformerVoltageControlFixMasks = false;                               // armed by setBatchTransformerVoltageControlFixMasks
    ctx->transformerVoltageControlFixWords = 0;
    if (jFixPvRow != nullptr) {
        readI(jFixPvRow, fixRow);
        if (!fixRow.empty()) {
            ctx->dTransformerVoltageControlFixPvRow.upload(fixRow.data(), fixRow.size() * sizeof(int));
        }
    }
    return 1;
}

// The per-(scenario, member) connectivity masks of fixTransformerVoltageControls. Uploaded per CHUNK,
// unlike the tables above: OLF re-runs the check on the POST-CONTINGENCY network, so the component of a
// not-controlled side is a property of the scenario, not of the run.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, setBatchTransformerVoltageControlFixMasks)(
        JNIEnv* env, jclass, jlong handle, jintArray jMin, jintArray jMax, jint words) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTransformerVoltageControlMembers <= 0) {
        return;
    }
    ctx->hasTransformerVoltageControlFixMasks = false;
    ctx->transformerVoltageControlFixWords = 0;
    if (jMin == nullptr || jMax == nullptr || words <= 0) {
        return;
    }
    size_t want = static_cast<size_t>(ctx->nScenarios) * ctx->nTransformerVoltageControlMembers * words;
    if (static_cast<size_t>(env->GetArrayLength(jMin)) != want
            || static_cast<size_t>(env->GetArrayLength(jMax)) != want) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchTransformerVoltageControlFixMasks: array length != nScenarios * nMembers * words");
        return;
    }
    std::vector<int> mn(want), mx(want);
    env->GetIntArrayRegion(jMin, 0, static_cast<jsize>(want), mn.data());
    env->GetIntArrayRegion(jMax, 0, static_cast<jsize>(want), mx.data());
    ctx->dTransformerVoltageControlFixMinMask.upload(mn.data(), want * sizeof(int));
    ctx->dTransformerVoltageControlFixMaxMask.upload(mx.data(), want * sizeof(int));
    ctx->transformerVoltageControlFixWords = words;
    ctx->hasTransformerVoltageControlFixMasks = true;
}

/** Per (scenario, freeze candidate): {frozen (0/1), the row target while frozen (gen − load)} — layout
 *  {@code (s * nGeneratorVoltageControlFreezeCandidates + c) * 2}; null without candidates. */
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, readBatchGeneratorVoltageControlFreeze)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTransformerVoltageControlMembers <= 0 || ctx->nGeneratorVoltageControlFreezeCandidates <= 0 || !ctx->dGeneratorVoltageControlFrozen.p) {
        return nullptr;
    }
    int n = ctx->n;
    size_t cnt = static_cast<size_t>(ctx->nGeneratorVoltageControlFreezeCandidates) * ctx->nScenarios;
    std::vector<int> fz(cnt), vrow(ctx->nGeneratorVoltageControlFreezeCandidates);
    std::vector<double> tgt(static_cast<size_t>(n) * ctx->nScenarios), out(cnt * 2);
    cudaMemcpy(fz.data(), ctx->dGeneratorVoltageControlFrozen.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vrow.data(), ctx->dGeneratorVoltageControlFreezeVRow.p, vrow.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tgt.data(), ctx->dTarget.p, tgt.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (int s = 0; s < ctx->nScenarios; s++) {
        for (int c = 0; c < ctx->nGeneratorVoltageControlFreezeCandidates; c++) {
            size_t i = static_cast<size_t>(s) * ctx->nGeneratorVoltageControlFreezeCandidates + c;
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
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, readBatchTransformerVoltageControlState)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTransformerVoltageControlMembers <= 0 || !ctx->dTransformerVoltageControlMemberState.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nTransformerVoltageControlMembers) * ctx->nScenarios;
    std::vector<int> st(cnt), tap(cnt);
    std::vector<double> rf(cnt), out(cnt * 3);
    cudaMemcpy(st.data(), ctx->dTransformerVoltageControlMemberState.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tap.data(), ctx->dTransformerVoltageControlMemberTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(rf.data(), ctx->dTransformerVoltageControlMemberRoundingR1.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < cnt; i++) {
        out[i * 3] = st[i];
        out[i * 3 + 1] = tap[i];
        out[i * 3 + 2] = rf[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}
