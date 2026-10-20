// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Shunt voltage control, incremental and discrete.
#include "jni_common.cuh"

/** Upload the incremental SHUNT voltage-control tables for the in-kernel loop. Per controller:
 *  bRow (SHUNT_B identity row), vRow (controlled bus), targetV, halfDb, elemAbs (global element, -1),
 *  shuntAdmittanceOverrideElement (shunt-pack element for the static (dG, dB) override, -1); groups: groupPtr[nGroups+1] ->
 *  groupCtrl (controllers, |b|-descending); subs: subPtr[nCtrl+1] -> per sub secPtr[nSubs+1] into
 *  secB/secG, subMin, subBase. Returns 1 when the kernel loop owns them, 0 when there are none. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, setBatchShuntControllers)(
        JNIEnv* env, jclass, jlong handle,
        jintArray jBRow, jintArray jVRow, jdoubleArray jTargetV, jdoubleArray jHalfDb, jintArray jElemAbs, jintArray jShuntAdmittanceOverrideElement,
        jintArray jGroupPtr, jintArray jGroupCtrl, jintArray jSubPtr, jintArray jSecPtr, jdoubleArray jSecB, jdoubleArray jSecG,
        jintArray jSubMin, jintArray jSubBase, jint maxSectionShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nShc = 0;
    if (jBRow == nullptr || env->GetArrayLength(jBRow) == 0) {
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
    std::vector<int> vRow, elemAbs, groupPtr, groupCtrl, subMin;
    std::vector<double> targetV, halfDb;
    readI(jBRow, ctx->hShcBRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfDb, halfDb);
    readI(jElemAbs, elemAbs);
    readI(jShuntAdmittanceOverrideElement, ctx->hShcShOvElem);
    readI(jGroupPtr, groupPtr);
    readI(jGroupCtrl, groupCtrl);
    readI(jSubPtr, ctx->hShcSubPtr);
    readI(jSecPtr, ctx->hShcSecPtr);
    readD(jSecB, ctx->hShcSecB);
    readD(jSecG, ctx->hShcSecG);
    readI(jSubMin, subMin);
    readI(jSubBase, ctx->hShcSubBase);
    int nc = static_cast<int>(ctx->hShcBRow.size());
    int nSubs = static_cast<int>(ctx->hShcSubBase.size());
    ctx->dShcBRow.upload(ctx->hShcBRow.data(), nc * sizeof(int));
    ctx->dShcVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->dShcTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->dShcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dShcElemAbs.upload(elemAbs.data(), nc * sizeof(int));
    ctx->dShcShOvElem.upload(ctx->hShcShOvElem.data(), nc * sizeof(int));
    ctx->dShcGroupPtr.upload(groupPtr.data(), groupPtr.size() * sizeof(int));
    ctx->dShcGroupCtrl.upload(groupCtrl.data(), groupCtrl.size() * sizeof(int));
    ctx->dShcSubPtr.upload(ctx->hShcSubPtr.data(), ctx->hShcSubPtr.size() * sizeof(int));
    ctx->dShcSecPtr.upload(ctx->hShcSecPtr.data(), ctx->hShcSecPtr.size() * sizeof(int));
    ctx->dShcSecB.upload(ctx->hShcSecB.data(), ctx->hShcSecB.size() * sizeof(double));
    ctx->dShcSecG.upload(ctx->hShcSecG.data(), ctx->hShcSecG.size() * sizeof(double));
    ctx->dShcSubMin.upload(subMin.data(), nSubs * sizeof(int));
    ctx->dShcSubBase.upload(ctx->hShcSubBase.data(), nSubs * sizeof(int));
    ctx->nShcGroups = static_cast<int>(groupPtr.size()) - 1;
    ctx->nShcSubs = nSubs;
    ctx->shcMaxSectionShift = maxSectionShift >= 0 ? maxSectionShift : 3;
    size_t nsu = static_cast<size_t>(ctx->nScenarios) * nSubs;
    size_t nsc = static_cast<size_t>(ctx->nScenarios) * nc;
    size_t nsg = static_cast<size_t>(ctx->nScenarios) * ctx->nShcGroups;
    ctx->dShcPos.ensure(nsu * sizeof(int));
    ctx->dShcRatDir.ensure(nsu * sizeof(int));
    ctx->dShcRatCount.ensure(nsu * sizeof(int));
    ctx->dShcRatAllowed.ensure(nsu * sizeof(int));
    ctx->dShcShift.ensure(nsu * sizeof(int));
    ctx->dShcSensi.ensure(nsc * sizeof(double));
    ctx->dShcBaseB.ensure(nsc * sizeof(double));
    ctx->dShcAppliedDb.ensure(nsc * sizeof(double));
    ctx->dShcOutOfDb.ensure(nsg * sizeof(char));
    ctx->dShcDiffV.ensure(nsg * sizeof(double));
    ctx->nShc = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, readBatchShuntPositions)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nShc <= 0 || !ctx->dShcPos.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nShcSubs) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dShcPos.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Install the DISCRETE (WITH_GENERATOR) shunt voltage-control tables (OLF order 10): groups =
 *  controlled buses (CSR over member shunts) with their toggle-diffed ON/OFF row deltas; members with
 *  SHUNT_B row, element (-1 = unpacked, never outaged) and sub-controllers (CSR) whose section-B tables,
 *  min and base positions drive LfShuntImpl.dispatchB. Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, setBatchDiscreteShunt)(
        JNIEnv* env, jclass, jlong handle, jint entryOn,
        jintArray jGrpPtr, jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaModeOn, jintArray jDeltaModeOff,
        jdoubleArray jDeltaTargetOn, jdoubleArray jDeltaTargetOff,
        jintArray jMemBRow, jintArray jMemElem, jintArray jMemSubPtr,
        jintArray jSecPtr, jdoubleArray jSecB, jintArray jSubMin, jintArray jSubBase) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nSvd = 0;
    if (jMemBRow == nullptr || env->GetArrayLength(jMemBRow) == 0) {
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
    std::vector<int> grpPtr, deltaPtr, deltaRow, deltaModeOn, deltaModeOff, memBRow, memElem, memSubPtr, secPtr, subMin, subBase;
    std::vector<double> deltaTargetOn, deltaTargetOff, secB;
    readI(jGrpPtr, grpPtr);
    readI(jDeltaPtr, deltaPtr);
    readI(jDeltaRow, deltaRow);
    readI(jDeltaModeOn, deltaModeOn);
    readI(jDeltaModeOff, deltaModeOff);
    readD(jDeltaTargetOn, deltaTargetOn);
    readD(jDeltaTargetOff, deltaTargetOff);
    readI(jMemBRow, memBRow);
    readI(jMemElem, memElem);
    readI(jMemSubPtr, memSubPtr);
    readI(jSecPtr, secPtr);
    readD(jSecB, secB);
    readI(jSubMin, subMin);
    readI(jSubBase, subBase);
    int nM = static_cast<int>(memBRow.size());
    int nG = static_cast<int>(grpPtr.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    int nSubs = static_cast<int>(subBase.size());
    ctx->dSvdGrpPtr.upload(grpPtr.data(), grpPtr.size() * sizeof(int));
    ctx->dSvdDeltaPtr.upload(deltaPtr.data(), deltaPtr.size() * sizeof(int));
    ctx->dSvdDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->dSvdDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->dSvdMemBRow.upload(memBRow.data(), nM * sizeof(int));
    ctx->dSvdMemElem.upload(memElem.data(), nM * sizeof(int));
    ctx->dSvdMemSubPtr.upload(memSubPtr.data(), memSubPtr.size() * sizeof(int));
    ctx->dSvdSecPtr.upload(secPtr.data(), secPtr.size() * sizeof(int));
    ctx->dSvdSecB.upload(secB.data(), std::max<size_t>(secB.size(), 1) * sizeof(double));
    ctx->dSvdSubMin.upload(subMin.data(), std::max(nSubs, 1) * sizeof(int));
    ctx->dSvdSubBase.upload(subBase.data(), std::max(nSubs, 1) * sizeof(int));
    size_t nS = ctx->nScenarios;
    ctx->dSvdGrpStep.ensure(nS * nG * sizeof(int));
    ctx->dSvdMemState.ensure(nS * nM * sizeof(int));
    ctx->dSvdMemContB.ensure(nS * nM * sizeof(double));
    ctx->dSvdSubPos.ensure(nS * std::max(nSubs, 1) * sizeof(int));
    ctx->dSvdSaved.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->nSvdGroups = nG;
    ctx->nSvdDelta = nD;
    ctx->nSvdSubs = nSubs;
    ctx->svdEntryOn = entryOn;
    ctx->nSvd = nM;
    return 1;
}

/** Per (scenario, member shunt): {state (0 off, 1 continuous, 2 dispatched), the continuous b the
 *  dispatch started from} — layout {@code (s * nM + m) * 2}; null without the table. */
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, readBatchDiscreteShuntState)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvd <= 0 || !ctx->dSvdMemState.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvd) * ctx->nScenarios;
    std::vector<int> st(cnt);
    std::vector<double> cb(cnt), out(cnt * 2);
    cudaMemcpy(st.data(), ctx->dSvdMemState.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cb.data(), ctx->dSvdMemContB.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < cnt; i++) {
        out[i * 2] = st[i];
        out[i * 2 + 1] = cb[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Per (scenario, sub-controller): the dispatched section — layout {@code s * nSubs + u}. */
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, ShuntVoltageControlNative, readBatchDiscreteShuntPositions)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvd <= 0 || ctx->nSvdSubs <= 0 || !ctx->dSvdSubPos.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvdSubs) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dSvdSubPos.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}
