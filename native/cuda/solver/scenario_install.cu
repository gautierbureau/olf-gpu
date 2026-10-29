// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See scenario_install.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../equations/fill_launch.cuh"
#include "compact_solve.cuh"
#include "scenario_install.cuh"

namespace olfgpu {

namespace {

// The base row src (n values) into every listed slot's row.
template <typename T>
__global__ void copyRowToSlots(int n, int nSlots, const int* slots, const T* src, T* d) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nSlots) return;
    int j = t / n;
    int i = t % n;
    d[static_cast<size_t>(slots[j]) * n + i] = src[i];
}

// Host rows, `width` values per installed slot in the install's order, into the bank's device array.
template <typename T>
void writeInstalledRows(const BatchSolverContext* ctx, T* device, const T* rows, size_t width) {
    if (ctx->installSlots.empty()) {
        CUDA_CHECK(cudaMemcpy(device, rows, static_cast<size_t>(ctx->nScenarios) * width * sizeof(T), cudaMemcpyHostToDevice));
        return;
    }
    for (int j = 0; j < installedSlotCount(ctx); j++) {
        CUDA_CHECK(cudaMemcpy(device + static_cast<size_t>(installedSlot(ctx, j)) * width, rows + static_cast<size_t>(j) * width,
                              width * sizeof(T), cudaMemcpyHostToDevice));
    }
}

}  // namespace

void installScenarios(BatchSolverContext* ctx, const ScenarioInputs& in) {
    const int n = ctx->n;
    const int nS = ctx->nScenarios;
    const int nLocal = installedSlotCount(ctx);
    std::vector<int> slots(nLocal);
    for (int j = 0; j < nLocal; j++) {
        slots[j] = installedSlot(ctx, j);
    }
    ctx->dInstallSlots.upload(slots.data(), slots.size() * sizeof(int));
    const int* dSlots = ctx->dInstallSlots.as<int>();
    ctx->dXBase.upload(in.x0, static_cast<size_t>(n) * sizeof(double));
    ctx->dBaseMode.upload(in.baseRowMode, static_cast<size_t>(n) * sizeof(int));
    // The BASE targets, kept aside before the per-scenario overrides: the pilot-Q capture reads the base carrier's row,
    // and a scenario may have overridden that row (a member islanded by a bus loss freezes it to a flat voltage
    // identity). Capturing from the base keeps the captured value the pilot's reactive target whatever the scenario did
    // to the row. (The carrier's target itself is stamped host-side when the base carrier is frozen - GpuBatchChunkSolver
    // - because the device stamp runs only inside the outer-loop passes.)
    ctx->dTargetBase.upload(in.target, static_cast<size_t>(n) * sizeof(double));

    // The state: a full per-scenario row (the warm restart of a strategy pair), or the base with a few per-scenario
    // starting values over it (the frozen rows of islanded buses, setBatchX0Seeds, already in bank slots).
    if (in.x0Full != nullptr) {
        writeInstalledRows(ctx, ctx->dX.as<double>(), in.x0Full, n);
    } else {
        copyRowToSlots<<<grid(n * nLocal), BLOCK, 0, ctx->stream>>>(n, nLocal, dSlots, ctx->dXBase.as<double>(), ctx->dX.as<double>());
        int nSeeds = static_cast<int>(ctx->x0SeedIdx.size());
        if (nSeeds > 0) {
            DeviceBuf dX0SeedIdx, dX0SeedVal;
            dX0SeedIdx.upload(ctx->x0SeedIdx.data(), nSeeds * sizeof(int));
            dX0SeedVal.upload(ctx->x0SeedVal.data(), nSeeds * sizeof(double));
            applyTargetOverrideBatch<<<grid(nSeeds), BLOCK, 0, ctx->stream>>>(nSeeds, dX0SeedIdx.as<int>(), dX0SeedVal.as<double>(),
                                                                             ctx->dX.as<double>());
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));  // the buffers are this scope's
        }
    }
    ctx->x0SeedIdx.clear();                                  // consumed by this install either way
    ctx->x0SeedVal.clear();

    // The targets: the base, then each scenario's overrides (a generator loss).
    copyRowToSlots<<<grid(n * nLocal), BLOCK, 0, ctx->stream>>>(n, nLocal, dSlots, ctx->dTargetBase.as<double>(), ctx->dTarget.as<double>());
    if (in.nTargetOverrides > 0) {
        std::vector<int> index(in.targetOverrideIndex, in.targetOverrideIndex + in.nTargetOverrides);
        for (int& idx : index) {
            idx = installedSlot(ctx, idx / n) * n + idx % n;
        }
        DeviceBuf dIndex, dValue;
        dIndex.upload(index.data(), index.size() * sizeof(int));
        dValue.upload(in.targetOverrideValue, static_cast<size_t>(in.nTargetOverrides) * sizeof(double));
        applyTargetOverrideBatch<<<grid(in.nTargetOverrides), BLOCK, 0, ctx->stream>>>(in.nTargetOverrides, dIndex.as<int>(),
                                                                                      dValue.as<double>(), ctx->dTarget.as<double>());
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }

    // The row modes: PER-SCENARIO, mutated in place by the outer loops. Every installed slot starts from the base row,
    // then only those that differ (islanded buses, switch outages, ...) get their own - a few rows instead of one per
    // scenario through PCIe.
    if (in.rowModeSlots == nullptr) {
        writeInstalledRows(ctx, ctx->dMode.as<int>(), in.rowMode, n);
    } else {
        copyRowToSlots<<<grid(n * nLocal), BLOCK, 0, ctx->stream>>>(n, nLocal, dSlots, ctx->dBaseMode.as<int>(), ctx->dMode.as<int>());
        if (in.nRowModeSlots > 0) {
            std::vector<int> modeSlots(in.nRowModeSlots);
            for (int k = 0; k < in.nRowModeSlots; k++) {
                modeSlots[k] = installedSlot(ctx, in.rowModeSlots[k]);
            }
            DeviceBuf dModeSlots, dModeRows;
            dModeSlots.upload(modeSlots.data(), modeSlots.size() * sizeof(int));
            dModeRows.upload(in.rowMode, static_cast<size_t>(in.nRowModeSlots) * n * sizeof(int));
            scatterRowsInt<<<grid(in.nRowModeSlots * n), BLOCK, 0, ctx->stream>>>(in.nRowModeSlots, n, dModeSlots.as<int>(),
                                                                                  dModeRows.as<int>(), ctx->dMode.as<int>());
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        }
    }
    writeInstalledRows(ctx, ctx->dDisabled.as<char>(), in.disabled, static_cast<size_t>(ctx->nElements));

    // The distributed slack's per-scenario channels exist only on a context with DS buses.
    const bool distributedSlack = ctx->nDistributedSlackBuses > 0;
    auto rowsIf = [&](const auto* rows) { return distributedSlack ? rows : nullptr; };
    ctx->distributedSlackOverrideBus.install(nS, ctx->installSlots, 1, rowsIf(in.distributedSlackOverrideBus));
    ctx->distributedSlackOverrideFactorDelta.install(nS, ctx->installSlots, 1, rowsIf(in.distributedSlackOverrideFactorDelta));
    ctx->distributedSlackOverrideBaseTargetPDelta.install(nS, ctx->installSlots, 1, rowsIf(in.distributedSlackOverrideBaseTargetPDelta));
    ctx->distributedSlackOverrideMaxDelta.install(nS, ctx->installSlots, 1, rowsIf(in.distributedSlackOverrideMaxDelta));
    ctx->distributedSlackOverrideMinDelta.install(nS, ctx->installSlots, 1, rowsIf(in.distributedSlackOverrideMinDelta));
    ctx->distributedSlackBusAnchorShift.install(nS, ctx->installSlots, static_cast<size_t>(std::max(ctx->nDistributedSlackBuses, 0)),
                                                rowsIf(in.distributedSlackBusAnchorShift));
    // The slack's own move also seeds a slack-only distribution, which has no DS bus.
    ctx->slackBusAnchorShift.install(nS, ctx->installSlots, 1, in.slackBusAnchorShift);
}

}  // namespace olfgpu
