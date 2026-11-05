// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Where a scenario's entries of a batched array live. Scenario-major - entry d of scenario s at s * count + d - is
// what cuDSS's uniform batch reads. Scenario-interleaved - at d * stride + s - puts one entry of consecutive
// scenarios side by side: a warp of 32 consecutive scenarios adding into the same Jacobian entry, which is what a fill
// thread per (element, scenario) does with the scenarios fastest, touches 8 sectors of L2 instead of 32. Our batched
// LU keeps its lanes interleaved the same way.
#ifndef OLF_GPU_COMMON_BATCH_LAYOUT_CUH
#define OLF_GPU_COMMON_BATCH_LAYOUT_CUH

#include <cstddef>

#include <cuda_runtime.h>

namespace olfgpu {

// Scenarios per 32-byte sector of an interleaved array: the interleaved stride is a multiple of it, so the entries of
// scenarios 4q .. 4q + 3 share one sector.
constexpr int SECTOR_SCENARIOS = 4;

struct BatchLayout {
    int nS = 0;                                              // the batch width: scenario slots
    int stride = 0;                                          // interleaved: the distance between two entries
    bool interleaved = false;
};

inline int interleavedStride(int nS) {
    return (nS + SECTOR_SCENARIOS - 1) / SECTOR_SCENARIOS * SECTOR_SCENARIOS;
}

inline BatchLayout scenarioMajorLayout(int nS) {
    return BatchLayout{nS, nS, false};
}

inline BatchLayout interleavedLayout(int nS) {
    return BatchLayout{nS, interleavedStride(nS), true};
}

// One scenario's entries of a batched array, or none (null): a fill that writes no Jacobian (a check's refill, the
// mismatch-only fill that ends a Newton budget) passes none for the values.
template <typename T>
struct ScenarioEntries {
    T* at = nullptr;                                         // the scenario's entry 0
    int step = 1;                                            // from one entry to the next
    __host__ __device__ T& operator[](int d) const { return at[static_cast<long long>(d) * step]; }
    __host__ __device__ explicit operator bool() const { return at != nullptr; }
};

template <typename T>
__host__ __device__ inline ScenarioEntries<T> scenarioEntries(T* base, int s, int count, BatchLayout layout) {
    if (base == nullptr) {
        return {};
    }
    if (layout.interleaved) {
        return {base + s, layout.stride};
    }
    return {base + static_cast<size_t>(s) * count, 1};
}

// The (item, scenario) of a thread of a one-thread-per-(item, scenario) kernel: the scenarios fastest on the
// interleaved layout (a warp: one item, consecutive scenarios), the items fastest on the scenario-major one.
__device__ inline void itemAndScenario(int tid, int itemCount, BatchLayout layout, int& item, int& s) {
    if (layout.interleaved) {
        s = tid % layout.nS;
        item = tid / layout.nS;
    } else {
        item = tid % itemCount;
        s = tid / itemCount;
    }
}

}  // namespace olfgpu

#endif  // OLF_GPU_COMMON_BATCH_LAYOUT_CUH
