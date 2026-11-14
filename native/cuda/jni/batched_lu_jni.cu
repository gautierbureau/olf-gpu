// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of BatchedLuNative: our batched LU alone, on an explicit CSR pattern - its plan's statistics, a
// factorization and solve on a lane map, and the elimination order cuDSS's analysis picks for the same pattern.
#include "jni_common.cuh"
#include "../solver/batched_lu_solve.cuh"
#include "../lu/amd_order.cuh"
#include "../lu/minimum_degree.cuh"
#include "../lu/supernodal_lu.cuh"

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

// The factorizations our batched LU ran in this process, the lanes of them its pivoting engine factorized again, and
// the fast-engine orders built with learned pivot delays, the fast-engine steps and levels launched with their long
// rows split, the pivoting-engine forward levels launched with their long rows split, the lanes sent straight to the
// pivoting engine, past the fast one, the lanes the fast engine's triangular solves ran on, the pivoting-engine backward
// levels solved a supernode row per warp, its split forward levels summed a supernode row per block, and its diagonal
// blocks factorized a row per warp (tests: a run really used it, and how).
extern "C" JNIEXPORT jlongArray JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, factorizationCounts)(JNIEnv* env, jclass) {
    const jlong counts[10] = {static_cast<jlong>(gBatchedLuFactorizations), static_cast<jlong>(gBatchedLuPivotedLanes),
                              static_cast<jlong>(gBatchedLuPivotDelayOrders), static_cast<jlong>(gBatchedLuSplitLaunches),
                              static_cast<jlong>(gBatchedLuSupernodalSplitLaunches),
                              static_cast<jlong>(gBatchedLuRoutedLanes), static_cast<jlong>(gBatchedLuFastLaneSolves),
                              static_cast<jlong>(gBatchedLuSupernodalRowLaunches),
                              static_cast<jlong>(gBatchedLuSupernodalForwardRowLaunches),
                              static_cast<jlong>(gBatchedLuSupernodalRowFactorizations)};
    jlongArray out = env->NewLongArray(10);
    env->SetLongArrayRegion(out, 0, 10, counts);
    return out;
}

// Tests: the pivoting engine's one-thread-per-supernode solve kernels (setPivotingEngineOneThreadKernelsForTests).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, setPivotingEngineOneThreadKernelsForTests)(JNIEnv*, jclass, jboolean on) {
    setPivotingEngineOneThreadKernelsForTests(on == JNI_TRUE);
}

extern "C" JNIEXPORT jlong JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, cudssContexts)(JNIEnv*, jclass) {
    return static_cast<jlong>(gCudssContexts);
}

extern "C" JNIEXPORT jlong JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, cudssOrders)(JNIEnv*, jclass) {
    return static_cast<jlong>(gCudssOrders);
}

// The approximate minimum-degree elimination order of A + A^T (position -> original index), as luPlanOf takes it.
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, amdOrder)(JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx) {
    try {
        const std::vector<int> rowPtr = intsOf(env, jRowPtr);
        const std::vector<int> colIdx = intsOf(env, jColIdx);
        const std::vector<int> order = approximateMinimumDegreeOrder(n, rowPtr.data(), colIdx.data());
        jintArray out = env->NewIntArray(n);
        env->SetIntArrayRegion(out, 0, n, order.data());
        return out;
    } catch (const std::exception& e) {
        throwJava(env, e);
        return nullptr;
    }
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

// The exact minimum-degree elimination order of A + A^T (position -> original index), as luPlanOf takes it.
extern "C" JNIEXPORT jintArray JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, minimumDegreeOrder)(JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx) {
    try {
        const std::vector<int> rowPtr = intsOf(env, jRowPtr);
        const std::vector<int> colIdx = intsOf(env, jColIdx);
        const std::vector<int> order = minimumDegreeOrder(n, rowPtr.data(), colIdx.data());
        jintArray out = env->NewIntArray(n);
        env->SetIntArrayRegion(out, 0, n, order.data());
        return out;
    } catch (const std::exception& e) {
        throwJava(env, e);
        return nullptr;
    }
}

// The elimination-tree height of A + A^T eliminated in order, as FEWEST_LEVELS compares the orders.
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, eliminationLevels)(JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx,
                                                           jintArray jOrder) {
    try {
        const std::vector<int> rowPtr = intsOf(env, jRowPtr);
        const std::vector<int> colIdx = intsOf(env, jColIdx);
        const std::vector<int> order = intsOf(env, jOrder);
        return eliminationLevels(n, rowPtr.data(), colIdx.data(), order);
    } catch (const std::exception& e) {
        throwJava(env, e);
        return -1;
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
// written. With interleavedValues, the values reach the factorization interleaved by scenario, as a batch context
// holds them (the engines read them in place), instead of scenario-major. With transposedRightHandSides and
// transposedSolutions (capacity * n each, scenario-major), the transposed systems are also solved on the same factors.
static void factorizeAndSolveIn(JNIEnv* env, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder,
                                jintArray jColOrder, jint capacity, jdoubleArray jValues, jdoubleArray jRightHandSides,
                                jintArray jLaneMap, jdoubleArray jSolutions, jintArray jPivoted, jintArray jPivotClamps,
                                bool interleavedValues, jdoubleArray jTransposedRightHandSides = nullptr,
                                jdoubleArray jTransposedSolutions = nullptr) {
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
        const bool transposed = jTransposedRightHandSides != nullptr && jTransposedSolutions != nullptr;
        std::vector<double> transposedRightHandSides(transposed ? static_cast<size_t>(capacity) * n : 0);
        std::vector<double> transposedSolutions(transposed ? static_cast<size_t>(capacity) * n : 0);
        if (transposed) {
            env->GetDoubleArrayRegion(jTransposedRightHandSides, 0, static_cast<jsize>(transposedRightHandSides.size()),
                                      transposedRightHandSides.data());
            env->GetDoubleArrayRegion(jTransposedSolutions, 0, static_cast<jsize>(transposedSolutions.size()),
                                      transposedSolutions.data());
        }
        for (int s : laneMap) {
            if (s < 0 || s >= capacity) {
                throw std::runtime_error("lane map entry " + std::to_string(s) + " outside 0.."
                                         + std::to_string(capacity - 1));
            }
        }
        const std::shared_ptr<LuPlan> plan = buildLuPlan(n, rowPtr.data(), colIdx.data(), rowOrder.data(),
                                                         colOrder.data());
        BatchedLu lu;
        createBatchedLu(lu, *plan, capacity);
        DeviceBuf dValues;
        DeviceBuf dRightHandSides;
        DeviceBuf dSolutions;
        const BatchLayout layout = interleavedValues ? interleavedLayout(capacity) : scenarioMajorLayout(capacity);
        if (interleavedValues) {
            std::vector<double> interleaved(static_cast<size_t>(layout.stride) * nnz, 0.0);
            for (int s = 0; s < capacity; s++) {
                for (int k = 0; k < nnz; k++) {
                    interleaved[static_cast<size_t>(k) * layout.stride + s] = values[static_cast<size_t>(s) * nnz + k];
                }
            }
            dValues.upload(interleaved.data(), interleaved.size() * sizeof(double));
        } else {
            dValues.upload(values.data(), values.size() * sizeof(double));
        }
        dRightHandSides.upload(rightHandSides.data(), rightHandSides.size() * sizeof(double));
        dSolutions.upload(solutions.data(), solutions.size() * sizeof(double));
        DeviceBuf dTransposedRightHandSides;
        DeviceBuf dTransposedSolutions;
        if (transposed) {
            dTransposedRightHandSides.upload(transposedRightHandSides.data(),
                                             transposedRightHandSides.size() * sizeof(double));
            dTransposedSolutions.upload(transposedSolutions.data(), transposedSolutions.size() * sizeof(double));
        }
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        std::vector<int> clamps;
        std::vector<int> pivoted(laneMap.size(), 0);
        try {
            factorizeLanes(lu, dValues.as<double>(), layout, laneMap.data(), static_cast<int>(laneMap.size()), stream);
            solveLanes(lu, dRightHandSides.as<double>(), dSolutions.as<double>(), stream);
            if (transposed) {
                solveLanesTransposed(lu, dTransposedRightHandSides.as<double>(), dTransposedSolutions.as<double>(),
                                     stream);
            }
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
            if (transposed) {
                CUDA_CHECK(cudaMemcpy(transposedSolutions.data(), dTransposedSolutions.p,
                                      transposedSolutions.size() * sizeof(double), cudaMemcpyDeviceToHost));
            }
        } catch (...) {
            cudaStreamDestroy(stream);
            throw;
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        env->SetDoubleArrayRegion(jSolutions, 0, static_cast<jsize>(solutions.size()), solutions.data());
        if (transposed) {
            env->SetDoubleArrayRegion(jTransposedSolutions, 0, static_cast<jsize>(transposedSolutions.size()),
                                      transposedSolutions.data());
        }
        env->SetIntArrayRegion(jPivoted, 0, static_cast<jsize>(pivoted.size()), pivoted.data());
        env->SetIntArrayRegion(jPivotClamps, 0, static_cast<jsize>(clamps.size()), clamps.data());
    } catch (const std::exception& e) {
        throwJava(env, e);
    }
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, factorizeAndSolve)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder,
        jint capacity, jdoubleArray jValues, jdoubleArray jRightHandSides, jintArray jLaneMap, jdoubleArray jSolutions,
        jintArray jPivoted, jintArray jPivotClamps) {
    factorizeAndSolveIn(env, n, jRowPtr, jColIdx, jRowOrder, jColOrder, capacity, jValues, jRightHandSides, jLaneMap,
                        jSolutions, jPivoted, jPivotClamps, false);
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, factorizeAndSolveInterleaved)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder,
        jint capacity, jdoubleArray jValues, jdoubleArray jRightHandSides, jintArray jLaneMap, jdoubleArray jSolutions,
        jintArray jPivoted, jintArray jPivotClamps) {
    factorizeAndSolveIn(env, n, jRowPtr, jColIdx, jRowOrder, jColOrder, capacity, jValues, jRightHandSides, jLaneMap,
                        jSolutions, jPivoted, jPivotClamps, true);
}

// factorizeAndSolve's factorization, then both solves on it: solutions = A^-1 rightHandSides and transposedSolutions =
// A^-T transposedRightHandSides (solveLanes, solveLanesTransposed: the adjoint's solve on the Newton's factors).
// pivoted and pivotClamps as factorizeAndSolve's.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, factorizeAndSolveTransposed)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder,
        jint capacity, jdoubleArray jValues, jdoubleArray jRightHandSides, jdoubleArray jTransposedRightHandSides,
        jintArray jLaneMap, jdoubleArray jSolutions, jdoubleArray jTransposedSolutions, jintArray jPivoted,
        jintArray jPivotClamps) {
    factorizeAndSolveIn(env, n, jRowPtr, jColIdx, jRowOrder, jColOrder, capacity, jValues, jRightHandSides, jLaneMap,
                        jSolutions, jPivoted, jPivotClamps, false, jTransposedRightHandSides, jTransposedSolutions);
}

// factorizeAndSolve's factorization, then the solve of the lanes whose scenario is in selected only (selectLanes,
// solveSelectedLanes: the adaptive refinement's corrections), every other scenario's solution left as it is.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, factorizeAndSolveSelected)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder,
        jint capacity, jdoubleArray jValues, jdoubleArray jRightHandSides, jintArray jLaneMap, jintArray jSelected,
        jdoubleArray jSolutions) {
    try {
        const std::vector<int> rowPtr = intsOf(env, jRowPtr);
        const std::vector<int> colIdx = intsOf(env, jColIdx);
        const std::vector<int> rowOrder = intsOf(env, jRowOrder);
        const std::vector<int> colOrder = intsOf(env, jColOrder);
        const std::vector<int> laneMap = intsOf(env, jLaneMap);
        const std::vector<int> selected = intsOf(env, jSelected);
        const int nnz = rowPtr[n];
        std::vector<double> values(static_cast<size_t>(capacity) * nnz);
        std::vector<double> rightHandSides(static_cast<size_t>(capacity) * n);
        std::vector<double> solutions(static_cast<size_t>(capacity) * n);
        env->GetDoubleArrayRegion(jValues, 0, static_cast<jsize>(values.size()), values.data());
        env->GetDoubleArrayRegion(jRightHandSides, 0, static_cast<jsize>(rightHandSides.size()), rightHandSides.data());
        env->GetDoubleArrayRegion(jSolutions, 0, static_cast<jsize>(solutions.size()), solutions.data());
        std::vector<char> scenarios(capacity, 0);
        for (const std::vector<int>* list : {&laneMap, &selected}) {
            for (int s : *list) {
                if (s < 0 || s >= capacity) {
                    throw std::runtime_error("scenario " + std::to_string(s) + " outside 0.."
                                             + std::to_string(capacity - 1));
                }
            }
        }
        for (int s : selected) {
            scenarios[s] = 1;
        }
        const std::shared_ptr<LuPlan> plan = buildLuPlan(n, rowPtr.data(), colIdx.data(), rowOrder.data(),
                                                         colOrder.data());
        BatchedLu lu;
        createBatchedLu(lu, *plan, capacity);
        DeviceBuf dValues;
        DeviceBuf dRightHandSides;
        DeviceBuf dSolutions;
        dValues.upload(values.data(), values.size() * sizeof(double));
        dRightHandSides.upload(rightHandSides.data(), rightHandSides.size() * sizeof(double));
        dSolutions.upload(solutions.data(), solutions.size() * sizeof(double));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        try {
            factorizeLanes(lu, dValues.as<double>(), scenarioMajorLayout(capacity), laneMap.data(),
                           static_cast<int>(laneMap.size()), stream);
            selectLanes(lu, scenarios, stream);
            solveSelectedLanes(lu, dRightHandSides.as<double>(), dSolutions.as<double>(), stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(solutions.data(), dSolutions.p, solutions.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        } catch (...) {
            cudaStreamDestroy(stream);
            throw;
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        env->SetDoubleArrayRegion(jSolutions, 0, static_cast<jsize>(solutions.size()), solutions.data());
    } catch (const std::exception& e) {
        throwJava(env, e);
    }
}

// The scenarios of laneMap factorized `rounds` times on one plan - the fast engine's order learning its pivot delays
// between rounds - then solved with the last factorization. values: capacity * nnz, rightHandSides and solutions:
// capacity * n, scenario-major; installRound: the round before which every lane's scenario is installed anew
// (forgetLaneRouting), -1 for none; pivoted (rounds * laneMap entries, round-major): 1 when the pivoting engine
// factorized the lane that round; delays (rounds): the pivot delays the fast engine's order had in each round. With
// interleavedValues, the values reach the factorizations interleaved by scenario, as a batch context holds them.
static void factorizeRoundsIn(JNIEnv* env, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder,
                              jintArray jColOrder, jint capacity, jdoubleArray jValues, jdoubleArray jRightHandSides,
                              jintArray jLaneMap, jint rounds, jint installRound, jdoubleArray jSolutions,
                              jintArray jPivoted, jintArray jDelays, bool interleavedValues) {
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
        const std::shared_ptr<LuPlan> plan = buildLuPlan(n, rowPtr.data(), colIdx.data(), rowOrder.data(),
                                                         colOrder.data());
        BatchedLu lu;
        createBatchedLu(lu, *plan, capacity);
        DeviceBuf dValues;
        DeviceBuf dRightHandSides;
        DeviceBuf dSolutions;
        const BatchLayout layout = interleavedValues ? interleavedLayout(capacity) : scenarioMajorLayout(capacity);
        if (interleavedValues) {
            std::vector<double> interleaved(static_cast<size_t>(layout.stride) * nnz, 0.0);
            for (int s = 0; s < capacity; s++) {
                for (int k = 0; k < nnz; k++) {
                    interleaved[static_cast<size_t>(k) * layout.stride + s] = values[static_cast<size_t>(s) * nnz + k];
                }
            }
            dValues.upload(interleaved.data(), interleaved.size() * sizeof(double));
        } else {
            dValues.upload(values.data(), values.size() * sizeof(double));
        }
        dRightHandSides.upload(rightHandSides.data(), rightHandSides.size() * sizeof(double));
        dSolutions.upload(solutions.data(), solutions.size() * sizeof(double));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        std::vector<int> pivoted(static_cast<size_t>(rounds) * laneMap.size(), 0);
        std::vector<int> delays(rounds, 0);
        try {
            for (int round = 0; round < rounds; round++) {
                if (round == installRound) {
                    forgetLaneRouting(lu, laneMap);
                }
                factorizeLanes(lu, dValues.as<double>(), layout, laneMap.data(), static_cast<int>(laneMap.size()),
                               stream);
                delays[round] = lu.fast->delayed;
                size_t next = 0;
                for (size_t a = 0; a < laneMap.size() && next < lu.hPivoted.size(); a++) {
                    if (laneMap[a] == lu.hPivoted[next]) {
                        pivoted[static_cast<size_t>(round) * laneMap.size() + a] = 1;
                        next++;
                    }
                }
            }
            solveLanes(lu, dRightHandSides.as<double>(), dSolutions.as<double>(), stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(solutions.data(), dSolutions.p, solutions.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        } catch (...) {
            cudaStreamDestroy(stream);
            throw;
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        env->SetDoubleArrayRegion(jSolutions, 0, static_cast<jsize>(solutions.size()), solutions.data());
        env->SetIntArrayRegion(jPivoted, 0, static_cast<jsize>(pivoted.size()), pivoted.data());
        env->SetIntArrayRegion(jDelays, 0, static_cast<jsize>(delays.size()), delays.data());
    } catch (const std::exception& e) {
        throwJava(env, e);
    }
}

// The Newton step's linear solve of a structure with couplers (solveNewtonStep: the solve with cuDSS's in-solve
// refinement steps, then the adaptive refinement) on the scenarios of laneMap, every other scenario's solution left as
// it is. values: capacity * nnz, rightHandSides and solutions: capacity * n, scenario-major.

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, factorizeRounds)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder,
        jint capacity, jdoubleArray jValues, jdoubleArray jRightHandSides, jintArray jLaneMap, jint rounds,
        jint installRound, jdoubleArray jSolutions, jintArray jPivoted, jintArray jDelays) {
    factorizeRoundsIn(env, n, jRowPtr, jColIdx, jRowOrder, jColOrder, capacity, jValues, jRightHandSides, jLaneMap,
                      rounds, installRound, jSolutions, jPivoted, jDelays, false);
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, factorizeRoundsInterleaved)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder,
        jint capacity, jdoubleArray jValues, jdoubleArray jRightHandSides, jintArray jLaneMap, jint rounds,
        jint installRound, jdoubleArray jSolutions, jintArray jPivoted, jintArray jDelays) {
    factorizeRoundsIn(env, n, jRowPtr, jColIdx, jRowOrder, jColOrder, capacity, jValues, jRightHandSides, jLaneMap,
                      rounds, installRound, jSolutions, jPivoted, jDelays, true);
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchedLuNative, factorizeAndSolveNewtonStep)(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx, jintArray jRowOrder, jintArray jColOrder,
        jint capacity, jdoubleArray jValues, jdoubleArray jRightHandSides, jintArray jLaneMap, jdoubleArray jSolutions) {
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
        std::vector<char> lanes(capacity, 0);
        for (int s : laneMap) {
            if (s < 0 || s >= capacity) {
                throw std::runtime_error("lane map entry " + std::to_string(s) + " outside 0.."
                                         + std::to_string(capacity - 1));
            }
            lanes[s] = 1;
        }
        const std::shared_ptr<LuPlan> plan = buildLuPlan(n, rowPtr.data(), colIdx.data(), rowOrder.data(),
                                                         colOrder.data());
        BatchedLu lu;
        createBatchedLu(lu, *plan, capacity);
        const size_t vectorBytes = static_cast<size_t>(capacity) * n * sizeof(double);
        DeviceBuf dRowPtr, dColIdx, dValues, dRightHandSides, dSolutions, dLanes, dResidual, dCorrection,
                dInternalResidual, dInternalCorrection, dScore, dMask;
        dRowPtr.upload(rowPtr.data(), rowPtr.size() * sizeof(int));
        dColIdx.upload(colIdx.data(), colIdx.size() * sizeof(int));
        dValues.upload(values.data(), values.size() * sizeof(double));
        dRightHandSides.upload(rightHandSides.data(), rightHandSides.size() * sizeof(double));
        dSolutions.upload(solutions.data(), solutions.size() * sizeof(double));
        dLanes.upload(lanes.data(), lanes.size());
        dResidual.alloc(vectorBytes);
        dCorrection.alloc(vectorBytes);
        dInternalResidual.alloc(vectorBytes);
        dInternalCorrection.alloc(vectorBytes);
        dScore.alloc(static_cast<size_t>(capacity) * sizeof(double));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        try {
            factorizeLanes(lu, dValues.as<double>(), scenarioMajorLayout(capacity), laneMap.data(),
                           static_cast<int>(laneMap.size()), stream);
            const BatchedLuSystems systems{&lu, n, capacity, nnz, dRowPtr.as<int>(), dColIdx.as<int>(), dValues.as<double>(),
                                           true, dLanes.as<char>(), stream, dResidual.as<double>(), dCorrection.as<double>(),
                                           dInternalResidual.as<double>(), dInternalCorrection.as<double>(), dScore, dMask,
                                           scenarioMajorLayout(capacity)};
            solveNewtonStep(systems, dRightHandSides.as<double>(), dSolutions.as<double>());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(solutions.data(), dSolutions.p, solutions.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        } catch (...) {
            cudaStreamDestroy(stream);
            throw;
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        env->SetDoubleArrayRegion(jSolutions, 0, static_cast<jsize>(solutions.size()), solutions.data());
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
