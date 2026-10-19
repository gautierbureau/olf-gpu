// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: The Newton and linear-solver settings of a batched solve.
#include "jni_common.cuh"

// The linear-solver settings of the run (GpuLoadFlowParameters): read by every context built after this call and by
// the adaptive refinement of every solve. -1 reorder / pivotType = AUTO; NaN threshold = the library's.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, NewtonSettingsNative, setLinearSolverSettings)(
        JNIEnv*, jclass, jint reorder, jint matching, jint pivotType, jdouble pivotThreshold,
        jint irBase, jint irMax, jdouble irTol) {
    gLinearSolver.reorder = reorder;
    gLinearSolver.matching = matching;
    gLinearSolver.pivotType = pivotType;
    gLinearSolver.pivotThreshold = pivotThreshold;
    gLinearSolver.irBase = irBase;
    gLinearSolver.irMax = irMax;
    gLinearSolver.irTol = irTol;
}

// The installed linear-solver settings, in setLinearSolverSettings's order (tests: the parameters reach the device).
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_solver_gpu, NewtonSettingsNative, readLinearSolverSettings)(JNIEnv* env, jclass) {
    const double v[7] = {static_cast<double>(gLinearSolver.reorder), static_cast<double>(gLinearSolver.matching),
                         static_cast<double>(gLinearSolver.pivotType), gLinearSolver.pivotThreshold,
                         static_cast<double>(gLinearSolver.irBase), static_cast<double>(gLinearSolver.irMax),
                         gLinearSolver.irTol};
    jdoubleArray out = env->NewDoubleArray(7);
    env->SetDoubleArrayRegion(out, 0, 7, v);
    return out;
}

// LINE_SEARCH backtracking parameters for a batch context (OLF LineSearchStateVectorScaling's
// maxIteration and stepFold): consumed by solves called with lsMode == GpuStateVectorScalingMode::LINE_SEARCH. Context-scoped like
// setBatchRowCriteria, so the solveBatchContext signature stays unchanged.
// Per-scenario starting values for the next solve given no full per-scenario state: flat index s*n+row -> value,
// written over the base state tiled on the device. Replaces sending nScenarios * n doubles when only a few rows
// differ (the frozen rows of islanded buses). Null or empty arrays clear them.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, NewtonSettingsNative, setBatchX0Seeds)(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jVal) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->x0SeedIdx.clear();
    ctx->x0SeedVal.clear();
    if (jIdx == nullptr || jVal == nullptr) {
        return;
    }
    jsize k = env->GetArrayLength(jIdx);
    ctx->x0SeedIdx.resize(k);
    ctx->x0SeedVal.resize(k);
    env->GetIntArrayRegion(jIdx, 0, k, ctx->x0SeedIdx.data());
    env->GetDoubleArrayRegion(jVal, 0, k, ctx->x0SeedVal.data());
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, NewtonSettingsNative, setBatchLineSearch)(
        JNIEnv*, jclass, jlong handle, jint lsMaxIter, jdouble lsStepFold) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->lsMaxIter = lsMaxIter > 0 ? lsMaxIter : 10;
    ctx->lsStepFold = lsStepFold > 1.0 ? lsStepFold : 4.0 / 3.0;
}

// PER_EQUATION_TYPE stopping criterion: install (or clear, with nulls) the per-row threshold
// arrays on a batch context — thrIdent for identity/slope rows, thrPower for power rows, both
// length n. The convergence test then requires EVERY row's |F| under its threshold (OLF's
// PerEquationTypeStoppingCriteria), replacing the 2-norm/infnorm rule for this context.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, NewtonSettingsNative, setBatchRowCriteria)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jThrIdent, jdoubleArray jThrPower) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jThrIdent == nullptr || jThrPower == nullptr) {
        ctx->perRowCriteria = false;
        ctx->perRowAlt = false;
        return;
    }
    int n = ctx->st->n;
    jdouble* thrIdent = env->GetDoubleArrayElements(jThrIdent, nullptr);
    jdouble* thrPower = env->GetDoubleArrayElements(jThrPower, nullptr);
    ctx->dThrIdent.upload(thrIdent, static_cast<size_t>(n) * sizeof(double));
    ctx->dThrPower.upload(thrPower, static_cast<size_t>(n) * sizeof(double));
    env->ReleaseDoubleArrayElements(jThrIdent, thrIdent, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jThrPower, thrPower, JNI_ABORT);
    ctx->perRowCriteria = true;
}

// The per-row ALTERNATE threshold and its selector row (-1: none) of the PER_EQUATION_TYPE criterion, read by
// residualNormsBatch for a cross identity (mode 6) and for a relocated child's row once its coupler's dummy row
// is an identity. Nulls clear them. Install after setBatchRowCriteria, which clears them too.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, NewtonSettingsNative, setBatchRowCriteriaAlt)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jThrAlt, jintArray jThrSel) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jThrAlt == nullptr || jThrSel == nullptr) {
        ctx->perRowAlt = false;
        return;
    }
    int n = ctx->st->n;
    jdouble* thrAlt = env->GetDoubleArrayElements(jThrAlt, nullptr);
    jint* thrSel = env->GetIntArrayElements(jThrSel, nullptr);
    ctx->dThrAlt.upload(thrAlt, static_cast<size_t>(n) * sizeof(double));
    ctx->dThrSel.upload(thrSel, static_cast<size_t>(n) * sizeof(int));
    env->ReleaseDoubleArrayElements(jThrAlt, thrAlt, JNI_ABORT);
    env->ReleaseIntArrayElements(jThrSel, thrSel, JNI_ABORT);
    ctx->perRowAlt = true;
}

// Per-scenario ACTIVE equation counts for the L2 stopping threshold (see hNActive). Per-chunk state.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, NewtonSettingsNative, setBatchActiveRowCounts)(
        JNIEnv* env, jclass, jlong handle, jintArray jNActive) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hNActive.clear();
    if (jNActive != nullptr) {
        int len = env->GetArrayLength(jNActive);
        ctx->hNActive.resize(len);
        env->GetIntArrayRegion(jNActive, 0, len, ctx->hNActive.data());
    }
}

/** Write absolute target values at (scenario, row) positions — how a host-owned loop applies its
 *  decision without disturbing the device-side targets the sequence has already accumulated. */
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, NewtonSettingsNative, setBatchTargetDeltas)(
        JNIEnv* env, jclass, jlong handle, jintArray jRows, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || jRows == nullptr) {
        return;
    }
    jsize cnt = env->GetArrayLength(jRows);
    if (cnt == 0) {
        return;
    }
    std::vector<int> rows(cnt);
    std::vector<double> vals(cnt);
    env->GetIntArrayRegion(jRows, 0, cnt, rows.data());
    env->GetDoubleArrayRegion(jVals, 0, cnt, vals.data());
    DeviceBuf dR;
    DeviceBuf dV;
    dR.upload(rows.data(), (size_t) cnt * sizeof(int));
    dV.upload(vals.data(), (size_t) cnt * sizeof(double));
    applyTargetOverrideBatch<<<grid(cnt), BLOCK, 0, ctx->stream>>>(cnt, dR.as<int>(), dV.as<double>(),
                                                                  ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, NewtonSettingsNative, setBatchFrozenRows)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotRow) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotRow == nullptr || env->GetArrayLength(jSlotRow) == 0) {
        ctx->nFrozenRow = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotRow);
    jint* sr = env->GetIntArrayElements(jSlotRow, nullptr);
    ctx->dFrozenRow.upload(sr, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotRow, sr, JNI_ABORT);
    ctx->nFrozenRow = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, NewtonSettingsNative, setBatchLoopTraceSlot)(
        JNIEnv*, jclass, jlong handle, jint slot) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->loopTraceSlot = slot;
    }
}
