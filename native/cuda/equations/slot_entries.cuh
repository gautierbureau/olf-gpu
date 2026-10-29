// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// One per-scenario channel's entries as the kernels read them (solver/slot_table.cuh installs them): slot s holds
// count[s] entries at [s * capacity, s * capacity + count[s]) of each field, in the order its scenario's plan gave them.
// A lookup reads its own slot's entries only, and a channel's first-match, last-match or every-match rule keeps its
// meaning within the slot.
#ifndef OLF_GPU_EQUATIONS_SLOT_ENTRIES_CUH
#define OLF_GPU_EQUATIONS_SLOT_ENTRIES_CUH

#include <cstddef>

namespace olfgpu {

struct SlotEntries {
    int capacity = 0;                                        // entries per slot
    int intWidth = 0;                                        // ints per entry
    int doubleWidth = 0;                                     // doubles per entry
    const int* count = nullptr;                              // per slot; null when no slot holds an entry
    const int* ints = nullptr;
    const double* doubles = nullptr;

    __host__ __device__ int countOf(int s) const {
        return count != nullptr ? count[s] : 0;
    }
    __host__ __device__ size_t at(int s, int k) const {
        return static_cast<size_t>(s) * capacity + k;
    }
    __device__ int intOf(int s, int k, int field = 0) const {
        return ints[at(s, k) * intWidth + field];
    }
    __device__ double doubleOf(int s, int k, int field = 0) const {
        return doubles[at(s, k) * doubleWidth + field];
    }
};

// The first entry of slot s whose int field 0 is `key`, -1 when none.
__device__ inline int slotFirstEntryOf(const SlotEntries& t, int s, int key) {
    int n = t.countOf(s);
    for (int k = 0; k < n; k++) {
        if (t.intOf(s, k) == key) {
            return k;
        }
    }
    return -1;
}

// The last entry of slot s whose int field 0 is `key`, -1 when none.
__device__ inline int slotLastEntryOf(const SlotEntries& t, int s, int key) {
    for (int k = t.countOf(s) - 1; k >= 0; k--) {
        if (t.intOf(s, k) == key) {
            return k;
        }
    }
    return -1;
}

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_SLOT_ENTRIES_CUH
