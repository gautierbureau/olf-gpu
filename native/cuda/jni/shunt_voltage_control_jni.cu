// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Shunt voltage control, incremental and discrete.
#include "jni_common.cuh"

/** Upload the incremental SHUNT voltage-control tables for the in-kernel loop. Per controller:
 *  bRow (SHUNT_B identity row), vRow (controlled bus), targetV, halfTargetDeadband, controllerElement (global element, -1),
 *  shuntAdmittanceOverrideElement (shunt-pack element for the static (dG, dB) override, -1); groups: groupControllerStart[nGroups+1] ->
 *  groupController (controllers, |b|-descending); subs: shuntCompensatorStart[nControllers+1] -> per sub sectionStart[nSubs+1] into
 *  sectionB/sectionG, shuntCompensatorMinPosition, shuntCompensatorBasePosition. Returns 1 when the kernel loop owns them, 0 when there are none. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, setBatchIncrementalShuntVoltageControl)(
        JNIEnv* env, jclass, jlong handle,
        jintArray jShuntBRow, jintArray jVRow, jdoubleArray jTargetV, jdoubleArray jHalfTargetDeadband, jintArray jControllerElement, jintArray jShuntAdmittanceOverrideElement,
        jintArray jGroupControllerStart, jintArray jGroupController, jintArray jShuntCompensatorStart, jintArray jSectionStart, jdoubleArray jSectionB, jdoubleArray jSectionG,
        jintArray jShuntCompensatorMinPosition, jintArray jShuntCompensatorBasePosition, jint maxSectionShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nIncrementalShuntVoltageControlControllerShunts = 0;
    if (jShuntBRow == nullptr || env->GetArrayLength(jShuntBRow) == 0) {
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
    std::vector<int> vRow, controllerElement, groupControllerStart, groupController, shuntCompensatorMinPosition;
    std::vector<double> targetV, halfTargetDeadband;
    readI(jShuntBRow, ctx->hIncrementalShuntVoltageControlShuntBRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfTargetDeadband, halfTargetDeadband);
    readI(jControllerElement, controllerElement);
    readI(jShuntAdmittanceOverrideElement, ctx->hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement);
    readI(jGroupControllerStart, groupControllerStart);
    readI(jGroupController, groupController);
    readI(jShuntCompensatorStart, ctx->hIncrementalShuntVoltageControlShuntCompensatorStart);
    readI(jSectionStart, ctx->hIncrementalShuntVoltageControlSectionStart);
    readD(jSectionB, ctx->hIncrementalShuntVoltageControlSectionB);
    readD(jSectionG, ctx->hIncrementalShuntVoltageControlSectionG);
    readI(jShuntCompensatorMinPosition, shuntCompensatorMinPosition);
    readI(jShuntCompensatorBasePosition, ctx->hIncrementalShuntVoltageControlShuntCompensatorBasePosition);
    int nc = static_cast<int>(ctx->hIncrementalShuntVoltageControlShuntBRow.size());
    int nSubs = static_cast<int>(ctx->hIncrementalShuntVoltageControlShuntCompensatorBasePosition.size());
    ctx->dIncrementalShuntVoltageControlShuntBRow.upload(ctx->hIncrementalShuntVoltageControlShuntBRow.data(), nc * sizeof(int));
    ctx->dIncrementalShuntVoltageControlControlledBusVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->dIncrementalShuntVoltageControlTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->dIncrementalShuntVoltageControlHalfTargetDeadband.upload(halfTargetDeadband.data(), nc * sizeof(double));
    ctx->dIncrementalShuntVoltageControlElement.upload(controllerElement.data(), nc * sizeof(int));
    ctx->dIncrementalShuntVoltageControlShuntAdmittanceOverrideElement.upload(ctx->hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement.data(), nc * sizeof(int));
    ctx->dIncrementalShuntVoltageControlGroupControllerStart.upload(groupControllerStart.data(), groupControllerStart.size() * sizeof(int));
    ctx->dIncrementalShuntVoltageControlGroupController.upload(groupController.data(), groupController.size() * sizeof(int));
    ctx->dIncrementalShuntVoltageControlShuntCompensatorStart.upload(ctx->hIncrementalShuntVoltageControlShuntCompensatorStart.data(), ctx->hIncrementalShuntVoltageControlShuntCompensatorStart.size() * sizeof(int));
    ctx->dIncrementalShuntVoltageControlSectionStart.upload(ctx->hIncrementalShuntVoltageControlSectionStart.data(), ctx->hIncrementalShuntVoltageControlSectionStart.size() * sizeof(int));
    ctx->dIncrementalShuntVoltageControlSectionB.upload(ctx->hIncrementalShuntVoltageControlSectionB.data(), ctx->hIncrementalShuntVoltageControlSectionB.size() * sizeof(double));
    ctx->dIncrementalShuntVoltageControlSectionG.upload(ctx->hIncrementalShuntVoltageControlSectionG.data(), ctx->hIncrementalShuntVoltageControlSectionG.size() * sizeof(double));
    ctx->dIncrementalShuntVoltageControlShuntCompensatorMinPosition.upload(shuntCompensatorMinPosition.data(), nSubs * sizeof(int));
    ctx->dIncrementalShuntVoltageControlShuntCompensatorBasePosition.upload(ctx->hIncrementalShuntVoltageControlShuntCompensatorBasePosition.data(), nSubs * sizeof(int));
    ctx->nIncrementalShuntVoltageControlGroups = static_cast<int>(groupControllerStart.size()) - 1;
    ctx->nIncrementalShuntVoltageControlShuntCompensators = nSubs;
    ctx->incrementalShuntVoltageControlMaxSectionShift = maxSectionShift >= 0 ? maxSectionShift : 3;
    size_t nsu = static_cast<size_t>(ctx->nScenarios) * nSubs;
    size_t nsc = static_cast<size_t>(ctx->nScenarios) * nc;
    size_t nsg = static_cast<size_t>(ctx->nScenarios) * ctx->nIncrementalShuntVoltageControlGroups;
    ctx->dIncrementalShuntVoltageControlPosition.ensure(nsu * sizeof(int));
    ctx->dIncrementalShuntVoltageControlCurrentDirection.ensure(nsu * sizeof(int));
    ctx->dIncrementalShuntVoltageControlDirectionChangeCount.ensure(nsu * sizeof(int));
    ctx->dIncrementalShuntVoltageControlAllowedDirection.ensure(nsu * sizeof(int));
    ctx->dIncrementalShuntVoltageControlSectionShift.ensure(nsu * sizeof(int));
    ctx->dIncrementalShuntVoltageControlSensitivity.ensure(nsc * sizeof(double));
    ctx->dIncrementalShuntVoltageControlBaseB.ensure(nsc * sizeof(double));
    ctx->dIncrementalShuntVoltageControlAppliedDeltaB.ensure(nsc * sizeof(double));
    ctx->dIncrementalShuntVoltageControlOutOfDeadband.ensure(nsg * sizeof(char));
    ctx->dIncrementalShuntVoltageControlDiffV.ensure(nsg * sizeof(double));
    ctx->nIncrementalShuntVoltageControlControllerShunts = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, readBatchIncrementalShuntVoltageControlPositions)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nIncrementalShuntVoltageControlControllerShunts <= 0 || !ctx->dIncrementalShuntVoltageControlPosition.p) {
        return nullptr;
    }
    size_t count = static_cast<size_t>(ctx->nIncrementalShuntVoltageControlShuntCompensators) * ctx->nScenarios;
    std::vector<int> host(count);
    cudaMemcpy(host.data(), ctx->dIncrementalShuntVoltageControlPosition.p, count * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(count));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(count), host.data());
    return out;
}

/** Install the DISCRETE (WITH_GENERATOR) shunt voltage-control tables (OLF order 10): groups =
 *  controlled buses (CSR over member shunts) with their toggle-diffed ON/OFF row deltas; members with
 *  SHUNT_B row, element (-1 = unpacked, never outaged) and sub-controllers (CSR) whose section-B tables,
 *  min and base positions drive LfShuntImpl.dispatchB. Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, setBatchShuntVoltageControl)(
        JNIEnv* env, jclass, jlong handle, jint entryOn,
        jintArray jGroupMemberStart, jintArray jDeltaStart, jintArray jDeltaRow, jintArray jDeltaModeOn, jintArray jDeltaModeOff,
        jdoubleArray jDeltaTargetOn, jdoubleArray jDeltaTargetOff,
        jintArray jMemberShuntBRow, jintArray jMemberElement, jintArray jMemberShuntCompensatorStart,
        jintArray jSectionStart, jdoubleArray jSectionB, jintArray jShuntCompensatorMinPosition, jintArray jShuntCompensatorBasePosition) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nShuntVoltageControlMembers = 0;
    if (jMemberShuntBRow == nullptr || env->GetArrayLength(jMemberShuntBRow) == 0) {
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
    std::vector<int> groupMemberStart, deltaStart, deltaRow, deltaModeOn, deltaModeOff, memberShuntBRow, memberElement, memberShuntCompensatorStart, sectionStart, shuntCompensatorMinPosition, shuntCompensatorBasePosition;
    std::vector<double> deltaTargetOn, deltaTargetOff, sectionB;
    readI(jGroupMemberStart, groupMemberStart);
    readI(jDeltaStart, deltaStart);
    readI(jDeltaRow, deltaRow);
    readI(jDeltaModeOn, deltaModeOn);
    readI(jDeltaModeOff, deltaModeOff);
    readD(jDeltaTargetOn, deltaTargetOn);
    readD(jDeltaTargetOff, deltaTargetOff);
    readI(jMemberShuntBRow, memberShuntBRow);
    readI(jMemberElement, memberElement);
    readI(jMemberShuntCompensatorStart, memberShuntCompensatorStart);
    readI(jSectionStart, sectionStart);
    readD(jSectionB, sectionB);
    readI(jShuntCompensatorMinPosition, shuntCompensatorMinPosition);
    readI(jShuntCompensatorBasePosition, shuntCompensatorBasePosition);
    int nM = static_cast<int>(memberShuntBRow.size());
    int nG = static_cast<int>(groupMemberStart.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    int nSubs = static_cast<int>(shuntCompensatorBasePosition.size());
    ctx->dShuntVoltageControlGroupMemberStart.upload(groupMemberStart.data(), groupMemberStart.size() * sizeof(int));
    ctx->dShuntVoltageControlDeltaStart.upload(deltaStart.data(), deltaStart.size() * sizeof(int));
    ctx->dShuntVoltageControlDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->dShuntVoltageControlDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->dShuntVoltageControlDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->dShuntVoltageControlDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->dShuntVoltageControlDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->dShuntVoltageControlMemberShuntBRow.upload(memberShuntBRow.data(), nM * sizeof(int));
    ctx->dShuntVoltageControlMemberElement.upload(memberElement.data(), nM * sizeof(int));
    ctx->dShuntVoltageControlMemberShuntCompensatorStart.upload(memberShuntCompensatorStart.data(), memberShuntCompensatorStart.size() * sizeof(int));
    ctx->dShuntVoltageControlSectionStart.upload(sectionStart.data(), sectionStart.size() * sizeof(int));
    ctx->dShuntVoltageControlSectionB.upload(sectionB.data(), std::max<size_t>(sectionB.size(), 1) * sizeof(double));
    ctx->dShuntVoltageControlShuntCompensatorMinPosition.upload(shuntCompensatorMinPosition.data(), std::max(nSubs, 1) * sizeof(int));
    ctx->dShuntVoltageControlShuntCompensatorBasePosition.upload(shuntCompensatorBasePosition.data(), std::max(nSubs, 1) * sizeof(int));
    size_t nS = ctx->nScenarios;
    ctx->dShuntVoltageControlGroupStep.ensure(nS * nG * sizeof(int));
    ctx->dShuntVoltageControlMemberState.ensure(nS * nM * sizeof(int));
    ctx->dShuntVoltageControlMemberContinuousB.ensure(nS * nM * sizeof(double));
    ctx->dShuntVoltageControlShuntCompensatorPosition.ensure(nS * std::max(nSubs, 1) * sizeof(int));
    ctx->dShuntVoltageControlSavedDeltaTarget.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->nShuntVoltageControlGroups = nG;
    ctx->nShuntVoltageControlDeltaRows = nD;
    ctx->nShuntVoltageControlShuntCompensators = nSubs;
    ctx->shuntVoltageControlEntryOn = entryOn;
    ctx->nShuntVoltageControlMembers = nM;
    return 1;
}

/** Per (scenario, member shunt): {state (0 off, 1 continuous, 2 dispatched), the continuous b the
 *  dispatch started from} — layout {@code (s * nM + m) * 2}; null without the table. */
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, readBatchShuntVoltageControlState)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nShuntVoltageControlMembers <= 0 || !ctx->dShuntVoltageControlMemberState.p) {
        return nullptr;
    }
    size_t count = static_cast<size_t>(ctx->nShuntVoltageControlMembers) * ctx->nScenarios;
    std::vector<int> memberState(count);
    std::vector<double> continuousB(count), out(count * 2);
    cudaMemcpy(memberState.data(), ctx->dShuntVoltageControlMemberState.p, count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(continuousB.data(), ctx->dShuntVoltageControlMemberContinuousB.p, count * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < count; i++) {
        out[i * 2] = memberState[i];
        out[i * 2 + 1] = continuousB[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Per (scenario, sub-controller): the dispatched section — layout {@code s * nSubs + u}. */
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, readBatchShuntVoltageControlPositions)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nShuntVoltageControlMembers <= 0 || ctx->nShuntVoltageControlShuntCompensators <= 0 || !ctx->dShuntVoltageControlShuntCompensatorPosition.p) {
        return nullptr;
    }
    size_t count = static_cast<size_t>(ctx->nShuntVoltageControlShuntCompensators) * ctx->nScenarios;
    std::vector<int> host(count);
    cudaMemcpy(host.data(), ctx->dShuntVoltageControlShuntCompensatorPosition.p, count * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(count));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(count), host.data());
    return out;
}
