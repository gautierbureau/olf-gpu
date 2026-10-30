// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of BatchedLuNative: our batched LU alone, on an explicit CSR pattern - its plan's statistics, a
// factorization and solve on a lane map, and the elimination order cuDSS's analysis picks for the same pattern.
#include "jni_common.cuh"

namespace {

void throwJava(JNIEnv* env, const std::exception& e) {
    jclass ex = env->FindClass("java/lang/RuntimeException");
    if (ex) {
        env->ThrowNew(ex, e.what());
    }
}

std::vector<int> intsOf(JNIEnv* env, jintArray a) {
    std::vector<int> v(env->GetArrayLength(a));
    env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    return v;
}

}  // namespace

// The factorizations our batched LU ran in this process, and the lanes of them its pivoting engine factorized again
// (tests: a run really used it, and how).
extern "C" JNIEXPORT jlongArray JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, factorizationCounts)(JNIEnv* env, jclass) {
    const jlong counts[2] = {static_cast<jlong>(gBatchedLuFactorizations), static_cast<jlong>(gBatchedLuPivotedLanes)};
    jlongArray out = env->NewLongArray(2);
    env->SetLongArrayRegion(out, 0, 2, counts);
    return out;
}

// The fast engine's plan: {LU entries, contributions, factorization steps, forward levels, backward levels}
extern "C" JNIEXPORT jlongArray JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, scalarPlanStatistics)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder) {
    try {
        const std::vector<int> rowPtr = intsOf(env, jRowPtr);
        const std::vector<int> colIdx = intsOf(env, jColIdx);
        const std::vector<int> rowOrder = intsOf(env, jRowOrder);
        const std::vector<int> colOrder = intsOf(env, jColOrder);
        const ScalarLuPlan plan = buildScalarLuPlan(n, rowPtr.data(), colIdx.data(), rowOrder.data(), colOrder.data());
        const jlong stats[5] = {plan.nnzLu, plan.contributions(), plan.factorSteps(), plan.forwardLevels(),
                                plan.backwardLevels()};
        jlongArray out = env->NewLongArray(5);
        env->SetLongArrayRegion(out, 0, 5, stats);
        return out;
    } catch (const std::exception& e) {
        throwJava(env, e);
        return nullptr;
    }
}

// The pivoting engine's plan: {stored entries, gather multiply-adds, levels, supernodes, columns of the largest
// supernode, L + U entries of the symmetrized structure}
extern "C" JNIEXPORT jlongArray JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, supernodalPlanStatistics)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder) {
    try {
        const std::vector<int> rowPtr = intsOf(env, jRowPtr);
        const std::vector<int> colIdx = intsOf(env, jColIdx);
        const std::vector<int> rowOrder = intsOf(env, jRowOrder);
        const std::vector<int> colOrder = intsOf(env, jColOrder);
        const SupernodalLuPlan plan = buildSupernodalLuPlan(n, rowPtr.data(), colIdx.data(), rowOrder.data(),
                                                            colOrder.data());
        int largest = 0;
        for (int s = 0; s < plan.nSupernodes; s++) {
            largest = std::max(largest, plan.supernodeSize(s));
        }
        const jlong stats[6] = {plan.nStored, plan.gatherMultiplyAdds(), plan.levels(), plan.nSupernodes, largest,
                                plan.choleskyEntries};
        jlongArray out = env->NewLongArray(6);
        env->SetLongArrayRegion(out, 0, 6, stats);
        return out;
    } catch (const std::exception& e) {
        throwJava(env, e);
        return nullptr;
    }
}

// values: capacity * nnz, rightHandSides and solutions: capacity * n, scenario-major; laneMap: the scenarios to
// factorize and solve; pivoted (1 when the pivoting engine factorized the lane) and pivotClamps: per entry of laneMap,
// written.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, factorizeAndSolve)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder,
        jint capacity, jdoubleArray jValues, jdoubleArray jRightHandSides, jintArray jLaneMap, jdoubleArray jSolutions,
        jintArray jPivoted, jintArray jPivotClamps) {
    try {
        const std::vector<int> rowPtr = intsOf(env, jRowPtr);
        const std::vector<int> colIdx = intsOf(env, jColIdx);
        const std::vector<int> rowOrder = intsOf(env, jRowOrder);
        const std::vector<int> colOrder = intsOf(env, jColOrder);
        const std::vector<int> laneMap = intsOf(env, jLaneMap);
        const int nnz = rowPtr[n];
        std::vector<double> values(static_cast<size_t>(capacity) * nnz);
        std::vector<double> rightHandSides(static_cast<size_t>(capacity) * n);
        std::vector<double> solutions(static_cast<size_t>(capacity) * n);
        env->GetDoubleArrayRegion(jValues, 0, static_cast<jsize>(values.size()), values.data());
        env->GetDoubleArrayRegion(jRightHandSides, 0, static_cast<jsize>(rightHandSides.size()), rightHandSides.data());
        env->GetDoubleArrayRegion(jSolutions, 0, static_cast<jsize>(solutions.size()), solutions.data());
        for (int s : laneMap) {
            if (s < 0 || s >= capacity) {
                throw std::runtime_error("lane map entry " + std::to_string(s) + " outside 0.."
                                         + std::to_string(capacity - 1));
            }
        }
        const LuPlan plan = buildLuPlan(n, rowPtr.data(), colIdx.data(), rowOrder.data(), colOrder.data());
        BatchedLu lu;
        createBatchedLu(lu, plan, capacity);
        DeviceBuf dValues;
        DeviceBuf dRightHandSides;
        DeviceBuf dSolutions;
        dValues.upload(values.data(), values.size() * sizeof(double));
        dRightHandSides.upload(rightHandSides.data(), rightHandSides.size() * sizeof(double));
        dSolutions.upload(solutions.data(), solutions.size() * sizeof(double));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        std::vector<int> clamps;
        std::vector<int> pivoted(laneMap.size(), 0);
        try {
            factorizeLanes(lu, dValues.as<double>(), laneMap.data(), static_cast<int>(laneMap.size()), stream);
            solveLanes(lu, dRightHandSides.as<double>(), dSolutions.as<double>(), stream);
            clamps = readPivotClamps(lu, stream);
            size_t next = 0;
            for (size_t a = 0; a < laneMap.size() && next < lu.hPivoted.size(); a++) {
                if (laneMap[a] == lu.hPivoted[next]) {
                    pivoted[a] = 1;
                    next++;
                }
            }
            CUDA_CHECK(cudaMemcpy(solutions.data(), dSolutions.p, solutions.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        } catch (...) {
            cudaStreamDestroy(stream);
            throw;
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        env->SetDoubleArrayRegion(jSolutions, 0, static_cast<jsize>(solutions.size()), solutions.data());
        env->SetIntArrayRegion(jPivoted, 0, static_cast<jsize>(pivoted.size()), pivoted.data());
        env->SetIntArrayRegion(jPivotClamps, 0, static_cast<jsize>(clamps.size()), clamps.data());
    } catch (const std::exception& e) {
        throwJava(env, e);
    }
}

// The row and column orders cuDSS's ANALYSIS picks for the pattern under the run's ordering (CUDSS_DATA_PERM_REORDER_*,
// as cuDSS returns them), and the LU fill it reports.
extern "C" JNIEXPORT jlong JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, cudssReorder)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder) {
    try {
        const std::vector<int> rowPtr = intsOf(env, jRowPtr);
        const std::vector<int> colIdx = intsOf(env, jColIdx);
        DeviceBuf dRowPtr;
        DeviceBuf dColIdx;
        dRowPtr.upload(rowPtr.data(), rowPtr.size() * sizeof(int));
        dColIdx.upload(colIdx.data(), colIdx.size() * sizeof(int));
        std::vector<int> rowOrder;
        std::vector<int> colOrder;
        int64_t luNnz = 0;
        cudssEliminationOrder(n, rowPtr[n], dRowPtr.as<int>(), dColIdx.as<int>(), false, rowOrder, colOrder, luNnz);
        env->SetIntArrayRegion(jRowOrder, 0, n, rowOrder.data());
        env->SetIntArrayRegion(jColOrder, 0, n, colOrder.data());
        return static_cast<jlong>(luNnz);
    } catch (const std::exception& e) {
        throwJava(env, e);
        return -1;
    }
}
