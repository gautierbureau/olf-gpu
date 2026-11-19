// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Shunt voltage control, incremental and discrete.
#include "jni_common.cuh"
#include "../solver/scenario_install.cuh"

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
    ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlControllerShunts = 0;
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
    readI(jShuntBRow, ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntBRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfTargetDeadband, halfTargetDeadband);
    readI(jControllerElement, controllerElement);
    readI(jShuntAdmittanceOverrideElement, ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement);
    readI(jGroupControllerStart, groupControllerStart);
    readI(jGroupController, groupController);
    readI(jShuntCompensatorStart, ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntCompensatorStart);
    readI(jSectionStart, ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlSectionStart);
    readD(jSectionB, ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlSectionB);
    readD(jSectionG, ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlSectionG);
    readI(jShuntCompensatorMinPosition, shuntCompensatorMinPosition);
    readI(jShuntCompensatorBasePosition, ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntCompensatorBasePosition);
    int nc = static_cast<int>(ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntBRow.size());
    int nSubs = static_cast<int>(ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntCompensatorBasePosition.size());
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntBRow.upload(ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntBRow.data(), nc * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlControlledBusVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlHalfTargetDeadband.upload(halfTargetDeadband.data(), nc * sizeof(double));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlElement.upload(controllerElement.data(), nc * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntAdmittanceOverrideElement.upload(ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntAdmittanceOverrideElement.data(), nc * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlGroupControllerStart.upload(groupControllerStart.data(), groupControllerStart.size() * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlGroupController.upload(groupController.data(), groupController.size() * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntCompensatorStart.upload(ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntCompensatorStart.data(), ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntCompensatorStart.size() * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionStart.upload(ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlSectionStart.data(), ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlSectionStart.size() * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionB.upload(ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlSectionB.data(), ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlSectionB.size() * sizeof(double));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionG.upload(ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlSectionG.data(), ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlSectionG.size() * sizeof(double));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntCompensatorMinPosition.upload(shuntCompensatorMinPosition.data(), nSubs * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlShuntCompensatorBasePosition.upload(ctx->incrementalShuntVoltageControl.hIncrementalShuntVoltageControlShuntCompensatorBasePosition.data(), nSubs * sizeof(int));
    ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlGroups = static_cast<int>(groupControllerStart.size()) - 1;
    ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlShuntCompensators = nSubs;
    ctx->incrementalShuntVoltageControl.incrementalShuntVoltageControlMaxSectionShift = maxSectionShift >= 0 ? maxSectionShift : 3;
    size_t nsu = static_cast<size_t>(ctx->nScenarios) * nSubs;
    size_t nsc = static_cast<size_t>(ctx->nScenarios) * nc;
    size_t nsg = static_cast<size_t>(ctx->nScenarios) * ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlGroups;
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlPosition.ensure(nsu * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlCurrentDirection.ensure(nsu * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlDirectionChangeCount.ensure(nsu * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlAllowedDirection.ensure(nsu * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSectionShift.ensure(nsu * sizeof(int));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlSensitivity.ensure(nsc * sizeof(double));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlBaseB.ensure(nsc * sizeof(double));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlAppliedDeltaB.ensure(nsc * sizeof(double));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlOutOfDeadband.ensure(nsg * sizeof(char));
    ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlDiffV.ensure(nsg * sizeof(double));
    ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlControllerShunts = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, readBatchIncrementalShuntVoltageControlPositions)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlControllerShunts <= 0 || !ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlPosition.p) {
        return nullptr;
    }
    std::vector<int> host;
    readInstalledRows(ctx, ctx->incrementalShuntVoltageControl.dIncrementalShuntVoltageControlPosition.p, static_cast<size_t>(ctx->incrementalShuntVoltageControl.nIncrementalShuntVoltageControlShuntCompensators), host);
    jintArray out = env->NewIntArray(static_cast<jsize>(host.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
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
    ctx->shuntVoltageControl.nShuntVoltageControlMembers = 0;
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
    ctx->shuntVoltageControl.dShuntVoltageControlGroupMemberStart.upload(groupMemberStart.data(), groupMemberStart.size() * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlDeltaStart.upload(deltaStart.data(), deltaStart.size() * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->shuntVoltageControl.dShuntVoltageControlDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->shuntVoltageControl.dShuntVoltageControlMemberShuntBRow.upload(memberShuntBRow.data(), nM * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlMemberElement.upload(memberElement.data(), nM * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlMemberShuntCompensatorStart.upload(memberShuntCompensatorStart.data(), memberShuntCompensatorStart.size() * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlSectionStart.upload(sectionStart.data(), sectionStart.size() * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlSectionB.upload(sectionB.data(), std::max<size_t>(sectionB.size(), 1) * sizeof(double));
    ctx->shuntVoltageControl.dShuntVoltageControlShuntCompensatorMinPosition.upload(shuntCompensatorMinPosition.data(), std::max(nSubs, 1) * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlShuntCompensatorBasePosition.upload(shuntCompensatorBasePosition.data(), std::max(nSubs, 1) * sizeof(int));
    size_t nS = ctx->nScenarios;
    ctx->shuntVoltageControl.dShuntVoltageControlGroupStep.ensure(nS * nG * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlMemberState.ensure(nS * nM * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlMemberContinuousB.ensure(nS * nM * sizeof(double));
    ctx->shuntVoltageControl.dShuntVoltageControlShuntCompensatorPosition.ensure(nS * std::max(nSubs, 1) * sizeof(int));
    ctx->shuntVoltageControl.dShuntVoltageControlSavedDeltaTarget.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->shuntVoltageControl.nShuntVoltageControlGroups = nG;
    ctx->shuntVoltageControl.nShuntVoltageControlDeltaRows = nD;
    ctx->shuntVoltageControl.nShuntVoltageControlShuntCompensators = nSubs;
    ctx->shuntVoltageControl.shuntVoltageControlEntryOn = entryOn;
    ctx->shuntVoltageControl.nShuntVoltageControlMembers = nM;
    return 1;
}

/** Per (scenario, member shunt): {state (0 off, 1 continuous, 2 dispatched), the continuous b the
 *  dispatch started from} — layout {@code (s * nM + m) * 2}; null without the table. */
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, readBatchShuntVoltageControlState)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->shuntVoltageControl.nShuntVoltageControlMembers <= 0 || !ctx->shuntVoltageControl.dShuntVoltageControlMemberState.p) {
        return nullptr;
    }
    const size_t nMembers = static_cast<size_t>(ctx->shuntVoltageControl.nShuntVoltageControlMembers);
    std::vector<int> memberState;
    std::vector<double> continuousB;
    readInstalledRows(ctx, ctx->shuntVoltageControl.dShuntVoltageControlMemberState.p, nMembers, memberState);
    readInstalledRows(ctx, ctx->shuntVoltageControl.dShuntVoltageControlMemberContinuousB.p, nMembers, continuousB);
    const size_t count = memberState.size();
    std::vector<double> out(count * 2);
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
    if (ctx == nullptr || ctx->shuntVoltageControl.nShuntVoltageControlMembers <= 0 || ctx->shuntVoltageControl.nShuntVoltageControlShuntCompensators <= 0 || !ctx->shuntVoltageControl.dShuntVoltageControlShuntCompensatorPosition.p) {
        return nullptr;
    }
    std::vector<int> host;
    readInstalledRows(ctx, ctx->shuntVoltageControl.dShuntVoltageControlShuntCompensatorPosition.p, static_cast<size_t>(ctx->shuntVoltageControl.nShuntVoltageControlShuntCompensators), host);
    jintArray out = env->NewIntArray(static_cast<jsize>(host.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}
