// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The host side of one per-scenario channel (equations/slot_entries.cuh): each slot's entries in its own region of
// `capacity` entries, so installing a scenario into a slot rewrites that slot's region alone and the other slots keep
// theirs. The host keeps the whole table: a slot that outgrows the capacity lays every region out again here, and the
// table goes to the device whole.
#ifndef OLF_GPU_SOLVER_SLOT_TABLE_CUH
#define OLF_GPU_SOLVER_SLOT_TABLE_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "../common/device_buf.cuh"
#include "../equations/slot_entries.cuh"

namespace olfgpu {

class SlotTable {
public:
    // A field the kernels write per entry (its size in bytes and the byte every one of its bytes starts at), laid out
    // like the entries: slot s's entry k at s * capacity + k.
    struct StateField {
        size_t elementBytes;
        unsigned char initByte;
    };

    SlotTable(int intWidth, int doubleWidth, std::vector<StateField> stateFields = {});

    // Rewrite the regions of `slots` (every slot of an nSlots-wide bank when empty) with the entries given in the plan's
    // order: per entry, 1 + intWidth ints - its LOCAL slot j (bank slot slots[j], or j when slots is empty), then its int
    // fields - and doubleWidth doubles. An installed slot without an entry is left empty; an entry whose local slot is
    // out of range is dropped (no lookup could match it).
    void install(int nSlots, const std::vector<int>& slots, int nEntries, const int* entryInts, const double* entryDoubles);

    SlotEntries view() const;
    // State field i of every slot's entries; the entries of an installed slot start at the field's initial bytes, and a
    // slot that outgrows the capacity carries the other slots' values along.
    template <typename T> T* state(int i) const { return static_cast<T*>(states_[i]->p); }
    // Every state field of `slots` (every slot when empty) back to its initial bytes, on `stream`.
    void resetState(const std::vector<int>& slots, cudaStream_t stream);
    int intWidth() const { return intWidth_; }
    int doubleWidth() const { return doubleWidth_; }
    int entries() const { return total_; }
    int countOf(int s) const { return s >= 0 && s < static_cast<int>(hCount_.size()) ? hCount_[s] : 0; }
    int intOf(int s, int k, int field = 0) const { return hInts_[at(s, k) * intWidth_ + field]; }
    double doubleOf(int s, int k, int field = 0) const { return hDoubles_[at(s, k) * doubleWidth_ + field]; }

private:
    size_t at(int s, int k) const { return static_cast<size_t>(s) * capacity_ + k; }
    void layOut(int capacity);                               // every region at a new capacity, on the host
    void layOutState(int capacity);                          // the state fields' too, through the host

    int intWidth_;
    int doubleWidth_;
    int nSlots_ = 0;
    int capacity_ = 0;
    int total_ = 0;
    std::vector<int> hCount_;
    std::vector<int> hInts_;
    std::vector<double> hDoubles_;
    DeviceBuf count_;
    DeviceBuf ints_;
    DeviceBuf doubles_;
    std::vector<StateField> stateFields_;
    std::vector<std::unique_ptr<DeviceBuf>> states_;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_SLOT_TABLE_CUH
