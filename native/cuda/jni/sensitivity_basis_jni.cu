// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI glue of SensitivityBasisNative (ac.solver.gpu): a batched sensitivity analysis's basis tables on a batch context,
// each install's participation columns, and the basis solutions of the handed-back scenarios (sensitivityBasisStep).
// Rows arrive as STABLE equation rows and are mapped to device rows here, through the structure's rowMap, as the
// incremental loops map theirs.
#include "jni_common.cuh"
#include "../solver/scenario_install.cuh"
#include "slot_table_install.cuh"

namespace {

int deviceRow(const BatchSolverContext* ctx, int stableRow) {
    const std::vector<int>& rowMap = ctx->structure->rowMap;
    return stableRow >= 0 && stableRow < static_cast<int>(rowMap.size()) ? rowMap[stableRow] : -1;
}

}  // namespace

// The basis every converged scenario solves: unit columns (stable rows), weighted columns (CSR by column over stable rows),
// the gather rows (state rows), and whether a per-scenario participation column follows. Null unit or gather rows: none.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, SensitivityBasisNative, setBatchSensitivityBasis)(
        JNIEnv* env, jclass, jlong handle, jintArray jUnitRows, jintArray jWeightedStart, jintArray jWeightedRows,
        jdoubleArray jWeightedValues, jintArray jGatherRows, jboolean participation) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    SensitivityBasisTables& basis = ctx->sensitivityBasis;
    basis.nUnits = 0;
    basis.nWeighted = 0;
    basis.nColumns = 0;
    basis.nGather = 0;
    basis.hUnitDeviceRows.clear();
    if (jUnitRows == nullptr || jGatherRows == nullptr) {
        return;
    }
    const int n = ctx->structure->n;
    std::vector<int> unitRows = jniInts(env, jUnitRows);
    std::vector<int> weightedStart = jniInts(env, jWeightedStart);
    std::vector<int> weightedRows = jniInts(env, jWeightedRows);
    std::vector<double> weightedValues = jniDoubles(env, jWeightedValues);
    std::vector<int> gatherRows = jniInts(env, jGatherRows);
    for (int r : gatherRows) {
        if (r < 0 || r >= n) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "setBatchSensitivityBasis: a gather row out of the state");
            return;
        }
    }
    for (int r : unitRows) {
        basis.hUnitDeviceRows.push_back(deviceRow(ctx, r));
    }
    int nWeighted = weightedStart.empty() ? 0 : static_cast<int>(weightedStart.size()) - 1;
    std::vector<double> weighted(static_cast<size_t>(nWeighted) * n, 0.0);
    for (int w = 0; w < nWeighted; w++) {
        for (int k = weightedStart[w]; k < weightedStart[w + 1] && k < static_cast<int>(weightedRows.size()); k++) {
            int row = deviceRow(ctx, weightedRows[k]);
            if (row >= 0) {
                weighted[static_cast<size_t>(w) * n + row] += weightedValues[k];
            }
        }
    }
    if (nWeighted > 0) {
        basis.dWeighted.ensure(weighted.size() * sizeof(double));
        CUDA_CHECK(cudaMemcpy(basis.dWeighted.p, weighted.data(), weighted.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    basis.dGatherRows.ensure(gatherRows.size() * sizeof(int));
    if (!gatherRows.empty()) {
        CUDA_CHECK(cudaMemcpy(basis.dGatherRows.p, gatherRows.data(), gatherRows.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    basis.nUnits = static_cast<int>(unitRows.size());
    basis.nWeighted = nWeighted;
    basis.nGather = static_cast<int>(gatherRows.size());
    basis.nColumns = basis.nUnits + basis.nWeighted + (participation ? 1 : 0);
}

// Per installed scenario, in the install's order, its participation column: n values over stable rows (null: none).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, SensitivityBasisNative, setBatchSensitivityParticipation)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jRows) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    const int n = ctx->structure->n;
    if (jRows == nullptr) {
        ctx->sensitivityParticipation.install(ctx->nScenarios, ctx->installSlots, n, nullptr);
        return;
    }
    const int installed = installedSlotCount(ctx);
    const size_t want = static_cast<size_t>(installed) * n;
    if (static_cast<size_t>(env->GetArrayLength(jRows)) != want) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "setBatchSensitivityParticipation: expected one row per installed scenario");
        return;
    }
    std::vector<double> stable(want);
    env->GetDoubleArrayRegion(jRows, 0, static_cast<jsize>(want), stable.data());
    std::vector<double> mapped(want, 0.0);
    for (int j = 0; j < installed; j++) {
        for (int r = 0; r < n; r++) {
            double value = stable[static_cast<size_t>(j) * n + r];
            if (value != 0) {
                int row = deviceRow(ctx, r);
                if (row >= 0) {
                    mapped[static_cast<size_t>(j) * n + row] += value;
                }
            }
        }
    }
    ctx->sensitivityParticipation.install(ctx->nScenarios, ctx->installSlots, n, mapped.data());
}

// The handed-back scenarios' basis solutions, in the install's order: nColumns x nGather each. Null without a basis.
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_solver_gpu, SensitivityBasisNative, readBatchSensitivityBasis)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->sensitivityBasis.nColumns <= 0 || ctx->sensitivityBasis.nGather <= 0
            || ctx->dSensitivityBasisOut.p == nullptr) {
        return nullptr;
    }
    std::vector<double> host;
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(readInstalledRows(ctx, ctx->dSensitivityBasisOut.p,
                                 static_cast<size_t>(ctx->sensitivityBasis.nColumns) * ctx->sensitivityBasis.nGather, host));
    jdoubleArray arr = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(arr, 0, static_cast<jsize>(host.size()), host.data());
    return arr;
}
