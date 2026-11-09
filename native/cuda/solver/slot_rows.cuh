// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A per-scenario channel kept densely, `width` values per slot (slot s's at [s * width, (s + 1) * width)): installing
// writes the installed slots' rows - the caller's values, or the fallback for a slot the install gives none - and leaves
// the other slots' rows as they are. The channel is on while any slot of the bank holds the caller's values: a kernel
// gets the array, or nullptr as before when no scenario of the bank has the channel. A slot holding no values always
// has the fallback in its row, so a kernel reading the array for it sees what it saw without the channel.
#ifndef OLF_GPU_SOLVER_SLOT_ROWS_CUH
#define OLF_GPU_SOLVER_SLOT_ROWS_CUH

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "../common/check.cuh"
#include "../common/device_buf.cuh"
#include "../common/slot_row_upload.cuh"

namespace olfgpu {

template <typename T>
class SlotRows {
public:
    explicit SlotRows(T fallback) : fallback_(fallback) {}

    // One row of `width` values per installed slot, local slot j's at rows + j * width (bank slot slots[j], or j when slots
    // is empty); rows null: the installed slots hold none.
    void install(int nSlots, const std::vector<int>& slots, size_t width, const T* rows) {
        begin(nSlots, slots, width);
        for (int j = 0; j < nLocal_ && rows != nullptr; j++) {
            std::copy_n(rows + static_cast<size_t>(j) * width, width, host_.begin() + static_cast<size_t>(bank(j)) * width);
            holds_[bank(j)] = 1;
        }
        end();
    }

    // The same from entries over the fallback - a list expanded to the dense form: per entry its local slot and its index in
    // the row, and its value (values null: `one` for every entry). An entry out of range is dropped; a slot holds values when
    // the install gave it at least one entry.
    void installEntries(int nSlots, const std::vector<int>& slots, size_t width, int nEntries, const int* slotIndex,
                        const T* values, T one = T()) {
        begin(nSlots, slots, width);
        for (int k = 0; k < nEntries; k++) {
            int j = slotIndex[2 * k];
            int i = slotIndex[2 * k + 1];
            if (j < 0 || j >= nLocal_ || i < 0 || static_cast<size_t>(i) >= width) {
                continue;
            }
            host_[static_cast<size_t>(bank(j)) * width + i] = values != nullptr ? values[k] : one;
            holds_[bank(j)] = 1;
        }
        end();
    }

    // Every slot back to the fallback, holding no values.
    void clear() {
        std::fill(holds_.begin(), holds_.end(), 0);
        std::fill(host_.begin(), host_.end(), fallback_);
        if (buf_.p != nullptr && !host_.empty()) {
            buf_.upload(host_.data(), host_.size() * sizeof(T));
        }
    }

    bool on() const { return std::find(holds_.begin(), holds_.end(), 1) != holds_.end(); }
    bool holds(int s) const { return s >= 0 && s < static_cast<int>(holds_.size()) && holds_[s] != 0; }
    size_t width() const { return width_; }
    T* device() const { return static_cast<T*>(buf_.p); }
    T* deviceOrNull() const { return on() ? device() : nullptr; }
    const T& at(int s, size_t i) const { return host_[static_cast<size_t>(s) * width_ + i]; }

private:
    int bank(int j) const { return slots_->empty() ? j : (*slots_)[j]; }

    void begin(int nSlots, const std::vector<int>& slots, size_t width) {
        reset_ = nSlots != nSlots_ || width != width_;
        if (reset_) {                                        // another bank or another row: nothing to keep
            nSlots_ = nSlots;
            width_ = width;
            holds_.assign(nSlots, 0);
            host_.assign(static_cast<size_t>(nSlots) * width, fallback_);
        }
        slots_ = &slots;
        nLocal_ = slots.empty() ? nSlots : static_cast<int>(slots.size());
        for (int j = 0; j < nLocal_; j++) {
            std::fill_n(host_.begin() + static_cast<size_t>(bank(j)) * width, width, fallback_);
            holds_[bank(j)] = 0;
        }
    }

    void end() {
        if (width_ == 0 || nSlots_ == 0) {
            return;
        }
        if (reset_ || slots_->empty() || buf_.p == nullptr) {
            buf_.upload(host_.data(), host_.size() * sizeof(T));
            return;
        }
        bankSlots_.resize(nLocal_);
        for (int j = 0; j < nLocal_; j++) {
            bankSlots_[j] = bank(j);
        }
        upload_.write(device(), width_, nLocal_, bankSlots_.data(),
                      [&](int j) { return host_.data() + static_cast<size_t>(bankSlots_[j]) * width_; });
    }

    T fallback_;
    int nSlots_ = 0;
    size_t width_ = 0;
    bool reset_ = false;
    int nLocal_ = 0;
    const std::vector<int>* slots_ = nullptr;
    std::vector<char> holds_;
    std::vector<T> host_;
    DeviceBuf buf_;
    std::vector<int> bankSlots_;
    SlotRowUpload upload_;
};

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_SLOT_ROWS_CUH
