// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Phase control and transformer reactive power control, incremental and continuous.
#include "jni_common.cuh"
#include "../solver/scenario_install.cuh"
#include "../common/codes/gpu_controlled_flow.cuh"

/** Upload the FLOW-OBSERVED incremental controllers (RPC / incremental phase). Per controller: kind
 *  (0 RPC, 1 phase CONTROLLER, 2 phase LIMITER), targetRow (the rho/a1 identity row), tapColumn (6 r1 / 7 a1),
 *  side (a GpuTwoSides) of the controlled flow, targetValue, halfTargetDeadband, elem (controller's global element),
 *  controlledElement (controlled branch's global element — a closed branch), tapStart/tapRows ([y g1 g2 b1 b2 ksi r1
 *  a1] per tap), baseTap, maxShift. Returns 1 when the kernel loops own them. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, PhaseControlNative, setBatchFlowControllers)(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jTargetRow, jintArray jTapColumn, jintArray jSide, jdoubleArray jTargetValue, jdoubleArray jHalfTargetDeadband,
        jintArray jElem, jintArray jControlledElement, jintArray jTapStart, jdoubleArray jTapRows, jintArray jBaseTap, jintArray jMaxTapShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->incrementalFlowControl.nIncrementalFlowControlControllerBranches = 0;
    ctx->incrementalFlowControl.nIncrementalPhaseControlControllerBranches = 0;
    ctx->incrementalFlowControl.nIncrementalTransformerReactivePowerControlControllerBranches = 0;
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
    readI(jKind, ctx->incrementalFlowControl.hIncrementalFlowControlControlledFlow);
    readI(jTargetRow, ctx->incrementalFlowControl.hIncrementalFlowControlTargetRow);
    readI(jTapColumn, tapColumn);
    readI(jSide, side);
    readD(jTargetValue, targetValue);
    readD(jHalfTargetDeadband, halfTargetDeadband);
    readI(jElem, ctx->incrementalFlowControl.hIncrementalFlowControlElement);
    readI(jControlledElement, controlledElement);
    readI(jTapStart, ctx->incrementalFlowControl.hIncrementalFlowControlTapStart);
    readD(jTapRows, ctx->incrementalFlowControl.hIncrementalFlowControlTapRows);
    readI(jBaseTap, ctx->incrementalFlowControl.hIncrementalFlowControlBaseTap);
    readI(jMaxTapShift, maxShift);
    int nc = static_cast<int>(ctx->incrementalFlowControl.hIncrementalFlowControlControlledFlow.size());
    std::vector<int> controlledClosedBranch(nc);
    for (int c = 0; c < nc; c++) {
        controlledClosedBranch[c] = controlledElement[c] - ctx->closedBranchOffset;             // the controlled branch's index in the closed-branch pack
        if (controlledClosedBranch[c] < 0 || controlledClosedBranch[c] >= ctx->structure->closedBranch.elementCount) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
        if (ctx->incrementalFlowControl.hIncrementalFlowControlControlledFlow[c] == GpuControlledFlow::REACTIVE_POWER) ctx->incrementalFlowControl.nIncrementalTransformerReactivePowerControlControllerBranches++; else ctx->incrementalFlowControl.nIncrementalPhaseControlControllerBranches++;
    }
    ctx->incrementalFlowControl.dIncrementalFlowControlControlledFlow.upload(ctx->incrementalFlowControl.hIncrementalFlowControlControlledFlow.data(), nc * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlTargetRow.upload(ctx->incrementalFlowControl.hIncrementalFlowControlTargetRow.data(), nc * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlTapColumn.upload(tapColumn.data(), nc * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlControlledSide.upload(side.data(), nc * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlTargetValue.upload(targetValue.data(), nc * sizeof(double));
    ctx->incrementalFlowControl.dIncrementalFlowControlHalfTargetDeadband.upload(halfTargetDeadband.data(), nc * sizeof(double));
    ctx->incrementalFlowControl.dIncrementalFlowControlElement.upload(ctx->incrementalFlowControl.hIncrementalFlowControlElement.data(), nc * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlControlledClosedBranch.upload(controlledClosedBranch.data(), nc * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlTapStart.upload(ctx->incrementalFlowControl.hIncrementalFlowControlTapStart.data(), ctx->incrementalFlowControl.hIncrementalFlowControlTapStart.size() * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlTapRows.upload(ctx->incrementalFlowControl.hIncrementalFlowControlTapRows.data(), ctx->incrementalFlowControl.hIncrementalFlowControlTapRows.size() * sizeof(double));
    ctx->incrementalFlowControl.dIncrementalFlowControlBaseTap.upload(ctx->incrementalFlowControl.hIncrementalFlowControlBaseTap.data(), nc * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlMaxTapShift.upload(maxShift.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->incrementalFlowControl.dIncrementalFlowControlCurrentTap.ensure(nst * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlAppliedTap.ensure(nst * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlCurrentDirection.ensure(nst * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlDirectionChangeCount.ensure(nst * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlAllowedDirection.ensure(nst * sizeof(int));
    ctx->incrementalFlowControl.dIncrementalFlowControlSensitivity.ensure(nst * sizeof(double));
    ctx->incrementalFlowControl.dIncrementalFlowControlDiff.ensure(nst * sizeof(double));
    ctx->incrementalFlowControl.dIncrementalFlowControlOutOfDeadband.ensure(nst * sizeof(char));
    ctx->incrementalFlowControl.nIncrementalFlowControlControllerBranches = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_outerloop_gpu, PhaseControlNative, readBatchFlowTaps)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->incrementalFlowControl.nIncrementalFlowControlControllerBranches <= 0 || !ctx->incrementalFlowControl.dIncrementalFlowControlCurrentTap.p) {
        return nullptr;
    }
    std::vector<int> host;
    readInstalledRows(ctx, ctx->incrementalFlowControl.dIncrementalFlowControlCurrentTap.p, static_cast<size_t>(ctx->incrementalFlowControl.nIncrementalFlowControlControllerBranches), host);
    jintArray out = env->NewIntArray(static_cast<jsize>(host.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}

/** Upload the CONTINUOUS phase controllers (kind 0 CONTROLLER / 1 LIMITER, a1 identity row, controlled
 *  side (a GpuTwoSides), current limit, global element, tap rows, base tap). Returns 1 when the kernel loop owns them. */
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
        if (e < 0 || e >= ctx->structure->closedBranch.elementCount) {
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
    std::vector<int> host;
    readInstalledRows(ctx, ctx->dPhaseControlTap.p, static_cast<size_t>(ctx->nPhaseControlControllerBranches), host);
    jintArray out = env->NewIntArray(static_cast<jsize>(host.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}
