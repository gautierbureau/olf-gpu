// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The per-scenario inputs of a solve, written into the install slots (ctx->installSlots, every slot of the bank when
// empty): local slot j of each array is bank slot installedSlot(ctx, j), and only the installed slots' rows change. An
// installed slot starts from the base state, targets and row modes, then takes its own. A readback returns the same
// rows in the same order: a caller reads its scenarios back as it installed them, wherever they sit in the bank.
#ifndef OLF_GPU_SOLVER_SCENARIO_INSTALL_CUH
#define OLF_GPU_SOLVER_SCENARIO_INSTALL_CUH

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "../common/check.cuh"
#include "batch_context.cuh"

namespace olfgpu {

// The installed slots as a device mask (1: installed), or nullptr when the install covers the bank: a kernel seeding or
// resetting per-slot state then does every slot, as it did before installs per slot.
inline const char* installMaskOrNull(const BatchSolverContext* ctx) {
    return ctx->installSlots.empty() ? nullptr : static_cast<const char*>(ctx->dInstallMask.p);
}

// The installed slots' bank slots, in the install's order.
inline std::vector<int> installedSlots(const BatchSolverContext* ctx) {
    std::vector<int> slots(installedSlotCount(ctx));
    for (int j = 0; j < static_cast<int>(slots.size()); j++) {
        slots[j] = installedSlot(ctx, j);
    }
    return slots;
}

// Per slot of the bank, 1 for an installed one (every slot when the install covers the bank).
inline std::vector<char> installedFlags(const BatchSolverContext* ctx) {
    std::vector<char> flags(ctx->nScenarios, ctx->installSlots.empty() ? 1 : 0);
    for (int s : ctx->installSlots) {
        flags[s] = 1;
    }
    return flags;
}

// Host rows, `width` values per installed slot in the install's order, into the bank's per-slot device array.
template <typename T>
void writeInstalledRows(const BatchSolverContext* ctx, T* device, const T* rows, size_t width) {
    if (ctx->installSlots.empty()) {
        CUDA_CHECK(cudaMemcpy(device, rows, static_cast<size_t>(ctx->nScenarios) * width * sizeof(T), cudaMemcpyHostToDevice));
        return;
    }
    ctx->installRowUpload.write(device, width, installedSlotCount(ctx), ctx->installSlots.data(),
                                [&](int j) { return rows + static_cast<size_t>(j) * width; });
}

// The installed slots' rows of a per-slot device array (`width` values per slot) into `host`, in the install's order.
// Returns the first copy's error (cudaSuccess when every copy went), which each readback handles as it handled its own
// copy.
template <typename T>
cudaError_t readInstalledRows(const BatchSolverContext* ctx, const void* device, size_t width, std::vector<T>& host) {
    const int nRows = installedSlotCount(ctx);
    host.resize(static_cast<size_t>(nRows) * width);
    const T* from = static_cast<const T*>(device);
    if (ctx->installSlots.empty()) {
        return cudaMemcpy(host.data(), from, host.size() * sizeof(T), cudaMemcpyDeviceToHost);
    }
    for (int j = 0; j < nRows; j++) {
        cudaError_t e = cudaMemcpy(host.data() + static_cast<size_t>(j) * width,
                                   from + static_cast<size_t>(installedSlot(ctx, j)) * width, width * sizeof(T),
                                   cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            return e;
        }
    }
    return cudaSuccess;
}

// The installed slots' rows of a per-slot device array copied from another one of the same layout, on the stream.
template <typename T>
void copyInstalledRows(const BatchSolverContext* ctx, T* device, const T* from, size_t width) {
    if (ctx->installSlots.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(device, from, static_cast<size_t>(ctx->nScenarios) * width * sizeof(T), cudaMemcpyDeviceToDevice,
                                   ctx->stream));
        return;
    }
    for (int s : ctx->installSlots) {
        CUDA_CHECK(cudaMemcpyAsync(device + static_cast<size_t>(s) * width, from + static_cast<size_t>(s) * width, width * sizeof(T),
                                   cudaMemcpyDeviceToDevice, ctx->stream));
    }
}

// Every byte of the installed slots' rows of a per-slot device array (`rowBytes` per slot) set to `byte`, on the stream.
inline void memsetInstalledRows(const BatchSolverContext* ctx, void* device, size_t rowBytes, int byte) {
    auto* p = static_cast<unsigned char*>(device);
    if (ctx->installSlots.empty()) {
        CUDA_CHECK(cudaMemsetAsync(p, byte, static_cast<size_t>(ctx->nScenarios) * rowBytes, ctx->stream));
        return;
    }
    for (int s : ctx->installSlots) {
        CUDA_CHECK(cudaMemsetAsync(p + static_cast<size_t>(s) * rowBytes, byte, rowBytes, ctx->stream));
    }
}

// A parameter of type T that does not take part in deducing T (C++20's std::type_identity_t): the array a helper writes
// names the type, and a null row or a literal fill value converts to it.
template <typename T>
struct SlotValue {
    using Type = T;
};

// The installed slots' rows of a per-slot device array (`width` values per slot) back to `row` (the same `width` values
// for every slot) or, row null, every value to `value`: a solve's per-slot state starting afresh for the scenarios it
// installs, the other slots' left as they are.
template <typename T>
void resetInstalledRows(const BatchSolverContext* ctx, T* device, size_t width, const typename SlotValue<T>::Type* row,
                        typename SlotValue<T>::Type value = T()) {
    if (device == nullptr || width == 0) {
        return;
    }
    T zero{};
    if (row == nullptr && std::memcmp(&value, &zero, sizeof(T)) == 0) {
        memsetInstalledRows(ctx, device, width * sizeof(T), 0);
        return;
    }
    const int nRows = installedSlotCount(ctx);
    std::vector<T> rows(static_cast<size_t>(nRows) * width, value);
    for (int j = 0; j < nRows && row != nullptr; j++) {
        std::copy(row, row + width, rows.begin() + static_cast<size_t>(j) * width);
    }
    writeInstalledRows(ctx, device, rows.data(), width);
}

// The installed slots' entries of a per-slot host vector, sized for the bank, back to `value`.
template <typename T>
void resetInstalledEntries(const BatchSolverContext* ctx, std::vector<T>& entries, T value) {
    entries.resize(ctx->nScenarios, value);
    if (ctx->installSlots.empty()) {
        std::fill(entries.begin(), entries.end(), value);
        return;
    }
    for (int s : ctx->installSlots) {
        entries[s] = value;
    }
}

struct ScenarioInputs {
    const double* x0 = nullptr;                              // the base state (n)
    const double* target = nullptr;                          // the base targets (n)
    const int* baseRowMode = nullptr;                        // the base row modes (n)
    // Where an installed scenario's row modes differ from the base: (local slot, row, mode) per entry.
    int nRowModeOverrides = 0;
    const int* rowModeOverrides = nullptr;
    const char* disabled = nullptr;                          // per installed slot, nElements
    int nTargetOverrides = 0;                                // (local slot * n + row, value)
    const int* targetOverrideIndex = nullptr;
    const double* targetOverrideValue = nullptr;
    const double* x0Full = nullptr;                          // per installed slot, n; null: the base state and the x0 seeds
    // Per installed slot, null when none of them has it: a generator loss at a DS participant bus...
    const int* distributedSlackOverrideBus = nullptr;
    const double* distributedSlackOverrideFactorDelta = nullptr;
    const double* distributedSlackOverrideBaseTargetPDelta = nullptr;
    const double* distributedSlackOverrideMaxDelta = nullptr;
    const double* distributedSlackOverrideMinDelta = nullptr;
    // ...a strategy pair's anchor shift per DS participant, and the slack's own target move.
    const double* distributedSlackBusAnchorShift = nullptr;
    const double* slackBusAnchorShift = nullptr;
};

void installScenarios(BatchSolverContext* ctx, const ScenarioInputs& in);

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_SCENARIO_INSTALL_CUH
