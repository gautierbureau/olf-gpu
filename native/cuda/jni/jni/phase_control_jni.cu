// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Phase control and transformer reactive power control, incremental and continuous.
#include "jni_common.cuh"
#include "../common/codes/gpu_controlled_flow.cuh"

/** Upload the FLOW-OBSERVED incremental controllers (RPC / incremental phase). Per controller: kind
 *  (0 RPC, 1 phase CONTROLLER, 2 phase LIMITER), pinRow (the rho/a1 identity row), tapColumn (6 r1 / 7 a1),
 *  side (1/2) of the controlled flow, targetValue, halfTargetDeadband, elem (controller's global element),
 *  controlledElement (controlled branch's global element — a closed branch), tapStart/tapRows ([y g1 g2 b1 b2 ksi r1
 *  a1] per tap), baseTap, maxShift. Returns 1 when the kernel loops own them. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, PhaseControlNative, setBatchFlowControllers)(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jPinRow, jintArray jTapColumn, jintArray jSide, jdoubleArray jTargetValue, jdoubleArray jHalfTargetDeadband,
        jintArray jElem, jintArray jControlledElement, jintArray jTapStart, jdoubleArray jTapRows, jintArray jBaseTap, jintArray jMaxTapShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nIncrementalFlowControlControllerBranches = 0;
    ctx->nIncrementalPhaseControlControllerBranches = 0;
    ctx->nIncrementalTransformerReactivePowerControlControllerBranches = 0;
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
    std::vector<int> tapColumn, side, controlledElement, maxShift;
    std::vector<double> targetValue, halfTargetDeadband;
    readI(jKind, ctx->hIncrementalFlowControlControlledFlow);
    readI(jPinRow, ctx->hIncrementalFlowControlPinRow);
    readI(jTapColumn, tapColumn);
    readI(jSide, side);
    readD(jTargetValue, targetValue);
    readD(jHalfTargetDeadband, halfTargetDeadband);
    readI(jElem, ctx->hIncrementalFlowControlElement);
    readI(jControlledElement, controlledElement);
    readI(jTapStart, ctx->hIncrementalFlowControlTapStart);
    readD(jTapRows, ctx->hIncrementalFlowControlTapRows);
    readI(jBaseTap, ctx->hIncrementalFlowControlBaseTap);
    readI(jMaxTapShift, maxShift);
    int nc = static_cast<int>(ctx->hIncrementalFlowControlControlledFlow.size());
    std::vector<int> controlledClosedBranch(nc);
    for (int c = 0; c < nc; c++) {
        controlledClosedBranch[c] = controlledElement[c] - ctx->closedBranchOffset;             // the controlled branch's index in the closed-branch pack
        if (controlledClosedBranch[c] < 0 || controlledClosedBranch[c] >= ctx->st->closedBranch.elementCount) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
        if (ctx->hIncrementalFlowControlControlledFlow[c] == GpuControlledFlow::REACTIVE_POWER) ctx->nIncrementalTransformerReactivePowerControlControllerBranches++; else ctx->nIncrementalPhaseControlControllerBranches++;
    }
    ctx->dIncrementalFlowControlControlledFlow.upload(ctx->hIncrementalFlowControlControlledFlow.data(), nc * sizeof(int));
    ctx->dIncrementalFlowControlPinRow.upload(ctx->hIncrementalFlowControlPinRow.data(), nc * sizeof(int));
    ctx->dIncrementalFlowControlTapColumn.upload(tapColumn.data(), nc * sizeof(int));
    ctx->dIncrementalFlowControlControlledSide.upload(side.data(), nc * sizeof(int));
    ctx->dIncrementalFlowControlTargetValue.upload(targetValue.data(), nc * sizeof(double));
    ctx->dIncrementalFlowControlHalfTargetDeadband.upload(halfTargetDeadband.data(), nc * sizeof(double));
    ctx->dIncrementalFlowControlElement.upload(ctx->hIncrementalFlowControlElement.data(), nc * sizeof(int));
    ctx->dIncrementalFlowControlControlledClosedBranch.upload(controlledClosedBranch.data(), nc * sizeof(int));
    ctx->dIncrementalFlowControlTapStart.upload(ctx->hIncrementalFlowControlTapStart.data(), ctx->hIncrementalFlowControlTapStart.size() * sizeof(int));
    ctx->dIncrementalFlowControlTapRows.upload(ctx->hIncrementalFlowControlTapRows.data(), ctx->hIncrementalFlowControlTapRows.size() * sizeof(double));
    ctx->dIncrementalFlowControlBaseTap.upload(ctx->hIncrementalFlowControlBaseTap.data(), nc * sizeof(int));
    ctx->dIncrementalFlowControlMaxTapShift.upload(maxShift.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dIncrementalFlowControlCurrentTap.ensure(nst * sizeof(int));
    ctx->dIncrementalFlowControlAppliedTap.ensure(nst * sizeof(int));
    ctx->dIncrementalFlowControlCurrentDirection.ensure(nst * sizeof(int));
    ctx->dIncrementalFlowControlDirectionChangeCount.ensure(nst * sizeof(int));
    ctx->dIncrementalFlowControlAllowedDirection.ensure(nst * sizeof(int));
    ctx->dIncrementalFlowControlSensitivity.ensure(nst * sizeof(double));
    ctx->dIncrementalFlowControlDiff.ensure(nst * sizeof(double));
    ctx->dIncrementalFlowControlOutOfDeadband.ensure(nst * sizeof(char));
    ctx->nIncrementalFlowControlControllerBranches = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, PhaseControlNative, readBatchFlowTaps)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nIncrementalFlowControlControllerBranches <= 0 || !ctx->dIncrementalFlowControlCurrentTap.p) {
        return nullptr;
    }
    size_t count = static_cast<size_t>(ctx->nIncrementalFlowControlControllerBranches) * ctx->nScenarios;
    std::vector<int> host(count);
    cudaMemcpy(host.data(), ctx->dIncrementalFlowControlCurrentTap.p, count * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(count));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(count), host.data());
    return out;
}

/** Upload the CONTINUOUS phase controllers (kind 0 CONTROLLER / 1 LIMITER, a1 identity row, controlled
 *  side, current limit, global element, tap rows, base tap). Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, PhaseControlNative, setBatchContinuousPhase)(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jA1Row, jintArray jSide, jdoubleArray jCurrentLimit, jintArray jElem,
        jintArray jTapStart, jdoubleArray jTapRows, jintArray jBaseTap) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nPhaseControlControllerBranches = 0;
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
    readD(jCurrentLimit, limit);
    readI(jElem, ctx->hPhaseControlElement);
    readI(jTapStart, ctx->hPhaseControlTapStart);
    readD(jTapRows, ctx->hPhaseControlTapRows);
    readI(jBaseTap, ctx->hPhaseControlBaseTap);
    int nc = static_cast<int>(kind.size());
    for (int c = 0; c < nc; c++) {
        int e = ctx->hPhaseControlElement[c] - ctx->closedBranchOffset;
        if (e < 0 || e >= ctx->st->closedBranch.elementCount) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
    }
    ctx->dPhaseControlMode.upload(kind.data(), nc * sizeof(int));
    ctx->dPhaseControlA1Row.upload(a1Row.data(), nc * sizeof(int));
    ctx->dPhaseControlControlledSide.upload(side.data(), nc * sizeof(int));
    ctx->dPhaseControlCurrentLimit.upload(limit.data(), nc * sizeof(double));
    ctx->dPhaseControlElement.upload(ctx->hPhaseControlElement.data(), nc * sizeof(int));
    ctx->dPhaseControlTapStart.upload(ctx->hPhaseControlTapStart.data(), ctx->hPhaseControlTapStart.size() * sizeof(int));
    ctx->dPhaseControlTapRows.upload(ctx->hPhaseControlTapRows.data(), ctx->hPhaseControlTapRows.size() * sizeof(double));
    ctx->dPhaseControlBaseTap.upload(ctx->hPhaseControlBaseTap.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dPhaseControlTap.ensure(nst * sizeof(int));
    ctx->dPhaseControlAppliedTap.ensure(nst * sizeof(int));
    ctx->dPhaseControlCheckCount.ensure(static_cast<size_t>(ctx->nScenarios) * sizeof(int));
    ctx->nPhaseControlControllerBranches = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, PhaseControlNative, readBatchContinuousPhaseTaps)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nPhaseControlControllerBranches <= 0 || !ctx->dPhaseControlTap.p) {
        return nullptr;
    }
    size_t count = static_cast<size_t>(ctx->nPhaseControlControllerBranches) * ctx->nScenarios;
    std::vector<int> host(count);
    cudaMemcpy(host.data(), ctx->dPhaseControlTap.p, count * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(count));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(count), host.data());
    return out;
}
