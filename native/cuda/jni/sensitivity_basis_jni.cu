// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI glue of SensitivityBasisNative (ac.solver.gpu): a batched sensitivity analysis's basis tables on a batch context,
// its contraction into the factors' sensitivities, each install's participation columns, and what the handed-back
// scenarios' step left (sensitivityBasisStep): their sensitivities with a contraction, their basis solutions without.
// Rows arrive as STABLE equation rows and are mapped to device rows here, through the structure's rowMap, as the
// incremental loops map theirs.
#include <algorithm>

#include "jni_common.cuh"
#include "../common/codes/gpu_sensitivity_basis_kind.cuh"
#include "../common/codes/gpu_sensitivity_function.cuh"
#include "../solver/scenario_install.cuh"
#include "slot_table_install.cuh"

namespace {

int deviceRow(const BatchSolverContext* ctx, int stableRow) {
    const std::vector<int>& rowMap = ctx->structure->rowMap;
    return stableRow >= 0 && stableRow < static_cast<int>(rowMap.size()) ? rowMap[stableRow] : -1;
}

template <typename T>
void uploadTable(DeviceBuf& buf, const std::vector<T>& host) {
    buf.ensure(std::max<size_t>(host.size(), 1) * sizeof(T));
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(buf.p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
}

bool allWithin(const std::vector<int>& values, int low, int high) {
    for (int v : values) {
        if (v < low || v >= high) {
            return false;
        }
    }
    return true;
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
    basis.nFactors = 0;                                      // a contraction follows the basis it contracts
    basis.nGroups = 0;
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

// The basis's contraction (SensitivityBasisRequest.Contraction), for the calls that follow: per unit column the stable row
// of its bus's active power balance, the slack's; per group (CSR) its columns and their kinds; per factor its group, its
// function, its closed branch and its four gather positions. Null factor groups: none (the solutions are read back).
// Installed after setBatchSensitivityBasis, which clears it.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, SensitivityBasisNative, setBatchSensitivityContraction)(
        JNIEnv* env, jclass, jlong handle, jintArray jUnitBusPRows, jint slackPRow, jintArray jGroupColumnStart,
        jintArray jGroupColumns, jintArray jGroupColumnKind, jintArray jFactorGroup, jintArray jFactorFunction,
        jintArray jFactorElement, jintArray jFactorGather) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    SensitivityBasisTables& basis = ctx->sensitivityBasis;
    basis.nFactors = 0;
    basis.nGroups = 0;
    if (jFactorGroup == nullptr || basis.nColumns <= 0) {
        return;
    }
    std::vector<int> unitBusPRows = jniInts(env, jUnitBusPRows);
    std::vector<int> groupColumnStart = jniInts(env, jGroupColumnStart);
    std::vector<int> groupColumns = jniInts(env, jGroupColumns);
    std::vector<int> groupColumnKind = jniInts(env, jGroupColumnKind);
    std::vector<int> factorGroup = jniInts(env, jFactorGroup);
    std::vector<int> factorFunction = jniInts(env, jFactorFunction);
    std::vector<int> factorElement = jniInts(env, jFactorElement);
    std::vector<int> factorGather = jniInts(env, jFactorGather);
    const int nFactors = static_cast<int>(factorGroup.size());
    const int nGroups = static_cast<int>(groupColumnStart.size()) - 1;
    // every index the kernel follows, checked here: a table that does not fit the basis and the structure is refused
    bool consistent = static_cast<int>(unitBusPRows.size()) == basis.nUnits && nGroups >= 0
            && groupColumnStart.front() == 0 && groupColumnStart.back() == static_cast<int>(groupColumns.size())
            && groupColumnKind.size() == groupColumns.size() && factorFunction.size() == factorGroup.size()
            && factorElement.size() == factorGroup.size() && factorGather.size() == 4 * factorGroup.size()
            && allWithin(groupColumns, 0, basis.nColumns) && allWithin(groupColumnKind, GpuSensitivityBasisKind::P, GpuSensitivityBasisKind::ONE + 1)
            && allWithin(factorGroup, 0, nGroups) && allWithin(factorFunction, GpuSensitivityFunction::ACTIVE_POWER_1, GpuSensitivityFunction::BUS_VOLTAGE + 1)
            && allWithin(factorGather, -1, basis.nGather);
    for (int g = 0; consistent && g < nGroups; g++) {
        consistent = groupColumnStart[g] <= groupColumnStart[g + 1];
    }
    for (int i = 0; consistent && i < nFactors; i++) {
        const int* gather = factorGather.data() + 4 * i;
        bool voltage = factorFunction[i] == GpuSensitivityFunction::BUS_VOLTAGE;
        consistent = voltage ? gather[0] >= 0
                : factorElement[i] >= 0 && factorElement[i] < ctx->structure->closedBranch.elementCount
                  && gather[0] >= 0 && gather[1] >= 0 && gather[2] >= 0 && gather[3] >= 0;
    }
    if (!consistent) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "setBatchSensitivityContraction: tables that do not fit the basis or the structure");
        return;
    }
    std::vector<int> unitDeviceBusPRows(unitBusPRows.size());
    std::vector<char> unitSlack(unitBusPRows.size());
    for (size_t u = 0; u < unitBusPRows.size(); u++) {
        unitDeviceBusPRows[u] = deviceRow(ctx, unitBusPRows[u]);
        unitSlack[u] = unitBusPRows[u] == slackPRow ? 1 : 0;   // on stable rows, as Contraction.coefficient's caller
    }
    uploadTable(basis.dUnitRows, basis.hUnitDeviceRows);
    uploadTable(basis.dUnitBusPRows, unitDeviceBusPRows);
    uploadTable(basis.dUnitSlack, unitSlack);
    uploadTable(basis.dGroupColumnStart, groupColumnStart);
    uploadTable(basis.dGroupColumns, groupColumns);
    uploadTable(basis.dGroupColumnKind, groupColumnKind);
    uploadTable(basis.dFactorGroup, factorGroup);
    uploadTable(basis.dFactorFunction, factorFunction);
    uploadTable(basis.dFactorElement, factorElement);
    uploadTable(basis.dFactorGather, factorGather);
    basis.nGroups = nGroups;
    basis.nFactors = nFactors;
}

// What the handed-back scenarios' step left, in the install's order: with a contraction, each one's sensitivities then its
// unit columns' row modes (nFactors + 2 x nUnits); without, its basis solutions (nColumns x nGather). Null without a basis.
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_solver_gpu, SensitivityBasisNative, readBatchSensitivityBasis)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->sensitivityBasis.nColumns <= 0 || ctx->sensitivityBasis.nGather <= 0
            || ctx->dSensitivityBasisOut.p == nullptr) {
        return nullptr;
    }
    const SensitivityBasisTables& basis = ctx->sensitivityBasis;
    const bool contracted = basis.nFactors > 0;
    if (contracted && ctx->dSensitivityValuesOut.p == nullptr) {
        return nullptr;
    }
    std::vector<double> host;
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(readInstalledRows(ctx, contracted ? ctx->dSensitivityValuesOut.p : ctx->dSensitivityBasisOut.p,
                                 contracted ? static_cast<size_t>(basis.nFactors) + 2 * static_cast<size_t>(basis.nUnits)
                                            : static_cast<size_t>(basis.nColumns) * basis.nGather, host));
    jdoubleArray arr = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(arr, 0, static_cast<jsize>(host.size()), host.data());
    return arr;
}
