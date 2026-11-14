// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Transformer voltage control, incremental and discrete.
#include "jni_common.cuh"
#include "../solver/scenario_install.cuh"
#include "slot_table_install.cuh"

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
    ctx->incrementalTransformerVoltageControl.nIncrementalTransformerVoltageControlControllerBranches = 0;
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
    readI(jRhoRow, ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlRhoRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfTargetDeadband, halfTargetDeadband);
    readI(jControllerElement, ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlElement);
    readI(jTapStart, ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlTapStart);
    readD(jTapRows, ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlTapRows);
    readI(jBaseTap, ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlBaseTap);
    readI(jGroupController, groupController);
    int nc = static_cast<int>(ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlRhoRow.size());
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlRhoRow.upload(ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlRhoRow.data(), nc * sizeof(int));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlControlledBusVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlHalfTargetDeadband.upload(halfTargetDeadband.data(), nc * sizeof(double));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlElement.upload(ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlElement.data(), nc * sizeof(int));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlTapStart.upload(ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlTapStart.data(), ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlTapStart.size() * sizeof(int));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlTapRows.upload(ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlTapRows.data(), ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlTapRows.size() * sizeof(double));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlBaseTap.upload(ctx->incrementalTransformerVoltageControl.hIncrementalTransformerVoltageControlBaseTap.data(), nc * sizeof(int));
    ctx->incrementalTransformerVoltageControl.nIncrementalTransformerVoltageControlGroups = static_cast<int>(groupControllerStart.size()) - 1;
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlGroupControllerStart.upload(groupControllerStart.data(), groupControllerStart.size() * sizeof(int));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlGroupController.upload(groupController.data(), groupController.size() * sizeof(int));
    ctx->incrementalTransformerVoltageControl.incrementalTransformerVoltageControlMaxTapShift = maxTapShift > 0 ? maxTapShift : 1;
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlCurrentTap.ensure(nst * sizeof(int));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlAppliedTap.ensure(nst * sizeof(int));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlCurrentDirection.ensure(nst * sizeof(int));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlDirectionChangeCount.ensure(nst * sizeof(int));
    ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlAllowedDirection.ensure(nst * sizeof(int));
    ctx->incrementalTransformerVoltageControl.nIncrementalTransformerVoltageControlControllerBranches = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, readBatchIncrementalTransformerVoltageControlTaps)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->incrementalTransformerVoltageControl.nIncrementalTransformerVoltageControlControllerBranches <= 0 || !ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlCurrentTap.p) {
        return nullptr;
    }
    std::vector<int> host;
    readInstalledRows(ctx, ctx->incrementalTransformerVoltageControl.dIncrementalTransformerVoltageControlCurrentTap.p,
                      static_cast<size_t>(ctx->incrementalTransformerVoltageControl.nIncrementalTransformerVoltageControlControllerBranches), host);
    jintArray out = env->NewIntArray(static_cast<jsize>(host.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
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
    ctx->transformerVoltageControl.nTransformerVoltageControlMembers = 0;
    ctx->nGeneratorVoltageControlFreezeCandidates = 0;
    ctx->transformerVoltageControl.transformerVoltageControlFixMinMask.clear();
    ctx->transformerVoltageControl.transformerVoltageControlFixMaxMask.clear();
    ctx->transformerVoltageControl.transformerVoltageControlFixWords = 0;
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
    readI(jMemberElement, ctx->transformerVoltageControl.hTransformerVoltageControlMemberElement);
    readI(jMemberBaseTap, ctx->transformerVoltageControl.hTransformerVoltageControlMemberBaseTap);
    readI(jMemberTapStart, ctx->transformerVoltageControl.hTransformerVoltageControlMemberTapStart);
    readD(jMemberMinR1, memberMinR1);
    readD(jMemberMaxR1, memberMaxR1);
    readD(jMemberInitialR1, memberInitialR1);
    readD(jTapRows, ctx->transformerVoltageControl.hTransformerVoltageControlTapRows);
    int nM = static_cast<int>(memberRhoRow.size());
    int nG = static_cast<int>(groupMemberStart.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    for (int m = 0; m < nM; m++) {
        if (ctx->transformerVoltageControl.hTransformerVoltageControlMemberTapStart[m + 1] - ctx->transformerVoltageControl.hTransformerVoltageControlMemberTapStart[m] <= 0) {
            return 0;                                        // no tap table: the host orchestrator keeps the control
        }
    }
    ctx->transformerVoltageControl.dTransformerVoltageControlGroupMemberStart.upload(groupMemberStart.data(), groupMemberStart.size() * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlGroupControlledBusVRow.upload(groupControlledBusVRow.data(), nG * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlGroupTargetV.upload(groupTargetV.data(), nG * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlGroupHalfTargetDeadband.upload(groupHalfTargetDeadband.data(), nG * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlGroupUseInitialTapPosition.upload(groupUseInitialTapPosition.data(), nG * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlDeltaStart.upload(deltaStart.data(), deltaStart.size() * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlMemberRhoRow.upload(memberRhoRow.data(), nM * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlMemberElement.upload(ctx->transformerVoltageControl.hTransformerVoltageControlMemberElement.data(), nM * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlMemberBaseTap.upload(ctx->transformerVoltageControl.hTransformerVoltageControlMemberBaseTap.data(), nM * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlMemberTapStart.upload(ctx->transformerVoltageControl.hTransformerVoltageControlMemberTapStart.data(), ctx->transformerVoltageControl.hTransformerVoltageControlMemberTapStart.size() * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlMemberMinR1.upload(memberMinR1.data(), nM * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlMemberMaxR1.upload(memberMaxR1.data(), nM * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlMemberInitialR1.upload(memberInitialR1.data(), nM * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlTapRows.upload(ctx->transformerVoltageControl.hTransformerVoltageControlTapRows.data(), ctx->transformerVoltageControl.hTransformerVoltageControlTapRows.size() * sizeof(double));
    size_t nS = ctx->nScenarios;
    ctx->transformerVoltageControl.dTransformerVoltageControlStep.ensure(nS * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlGroupStep.ensure(nS * nG * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlGroupSharedMinR1.ensure(nS * nG * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlGroupSharedMaxR1.ensure(nS * nG * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlGroupSharedInitialR1.ensure(nS * nG * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlMemberState.ensure(nS * nM * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlMemberTap.ensure(nS * nM * sizeof(int));
    ctx->transformerVoltageControl.dTransformerVoltageControlMemberRoundingR1.ensure(nS * nM * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlSavedDeltaTarget.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->transformerVoltageControl.dTransformerVoltageControlRejected.ensure(nS * sizeof(char));
    ctx->transformerVoltageControl.nTransformerVoltageControlGroups = nG;
    ctx->transformerVoltageControl.nTransformerVoltageControlDeltaRows = nD;
    ctx->transformerVoltageControl.transformerVoltageControlMode = mode;
    ctx->transformerVoltageControl.transformerVoltageControlEntryOn = entryOn;
    ctx->transformerVoltageControl.nTransformerVoltageControlMembers = nM;
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
    if (ctx == nullptr || ctx->transformerVoltageControl.nTransformerVoltageControlMembers <= 0) {
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
    ctx->transformerVoltageControl.transformerVoltageControlFixMinMask.clear();                               // armed by setBatchTransformerVoltageControlFixMasks
    ctx->transformerVoltageControl.transformerVoltageControlFixMaxMask.clear();
    ctx->transformerVoltageControl.transformerVoltageControlFixWords = 0;
    if (jFixPvRow != nullptr) {
        readI(jFixPvRow, fixRow);
        if (!fixRow.empty()) {
            ctx->transformerVoltageControl.dTransformerVoltageControlFixPvRow.upload(fixRow.data(), fixRow.size() * sizeof(int));
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
    if (ctx == nullptr || ctx->transformerVoltageControl.nTransformerVoltageControlMembers <= 0) {
        return;
    }
    bool none = jMin == nullptr || jMax == nullptr || words <= 0;
    size_t width = static_cast<size_t>(ctx->transformerVoltageControl.nTransformerVoltageControlMembers) * std::max(static_cast<int>(words), 0);
    const char* who = "setBatchTransformerVoltageControlFixMasks";
    installSlotRows(env, ctx, ctx->transformerVoltageControl.transformerVoltageControlFixMinMask, width, none ? nullptr : jMin, who);
    if (!env->ExceptionCheck()) {
        installSlotRows(env, ctx, ctx->transformerVoltageControl.transformerVoltageControlFixMaxMask, width, none ? nullptr : jMax, who);
    }
    ctx->transformerVoltageControl.transformerVoltageControlFixWords = none ? 0 : words;
}

/** Per (scenario, freeze candidate): {frozen (0/1), the row target while frozen (gen − load)} — layout
 *  {@code (s * nGeneratorVoltageControlFreezeCandidates + c) * 2}; null without candidates. */
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, TransformerVoltageControlNative, readBatchGeneratorVoltageControlFreeze)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->transformerVoltageControl.nTransformerVoltageControlMembers <= 0 || ctx->nGeneratorVoltageControlFreezeCandidates <= 0 || !ctx->dGeneratorVoltageControlFrozen.p) {
        return nullptr;
    }
    int n = ctx->n;
    const int nRows = installedSlotCount(ctx);
    size_t count = static_cast<size_t>(ctx->nGeneratorVoltageControlFreezeCandidates) * nRows;
    std::vector<int> frozen, voltageRow(ctx->nGeneratorVoltageControlFreezeCandidates);
    std::vector<double> target, out(count * 2);
    readInstalledRows(ctx, ctx->dGeneratorVoltageControlFrozen.p, static_cast<size_t>(ctx->nGeneratorVoltageControlFreezeCandidates), frozen);
    cudaMemcpy(voltageRow.data(), ctx->dGeneratorVoltageControlFreezeVRow.p, voltageRow.size() * sizeof(int), cudaMemcpyDeviceToHost);
    readInstalledRows(ctx, ctx->dTarget.p, static_cast<size_t>(n), target);
    for (int s = 0; s < nRows; s++) {
        for (int c = 0; c < ctx->nGeneratorVoltageControlFreezeCandidates; c++) {
            size_t i = static_cast<size_t>(s) * ctx->nGeneratorVoltageControlFreezeCandidates + c;
            out[i * 2] = frozen[i];
            out[i * 2 + 1] = target[static_cast<size_t>(s) * n + voltageRow[c]];
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
    if (ctx == nullptr || ctx->transformerVoltageControl.nTransformerVoltageControlMembers <= 0 || !ctx->transformerVoltageControl.dTransformerVoltageControlMemberState.p) {
        return nullptr;
    }
    const size_t nMembers = static_cast<size_t>(ctx->transformerVoltageControl.nTransformerVoltageControlMembers);
    std::vector<int> memberState, tap;
    std::vector<double> roundingR1;
    readInstalledRows(ctx, ctx->transformerVoltageControl.dTransformerVoltageControlMemberState.p, nMembers, memberState);
    readInstalledRows(ctx, ctx->transformerVoltageControl.dTransformerVoltageControlMemberTap.p, nMembers, tap);
    readInstalledRows(ctx, ctx->transformerVoltageControl.dTransformerVoltageControlMemberRoundingR1.p, nMembers, roundingR1);
    const size_t count = memberState.size();
    std::vector<double> out(count * 3);
    for (size_t i = 0; i < count; i++) {
        out[i * 3] = memberState[i];
        out[i * 3 + 1] = tap[i];
        out[i * 3 + 2] = roundingR1[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}
