// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See slot_table.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "../common/check.cuh"
#include "slot_table.cuh"

namespace olfgpu {

SlotTable::SlotTable(int intWidth, int doubleWidth, std::vector<StateField> stateFields)
        : intWidth_(intWidth), doubleWidth_(doubleWidth), stateFields_(std::move(stateFields)) {
    for (size_t i = 0; i < stateFields_.size(); i++) {
        states_.push_back(std::make_unique<DeviceBuf>());
    }
}

void SlotTable::layOutState(int capacity) {
    for (size_t f = 0; f < stateFields_.size(); f++) {
        const size_t eb = stateFields_[f].elementBytes;
        std::vector<unsigned char> to(static_cast<size_t>(nSlots_) * capacity * eb, stateFields_[f].initByte);
        DeviceBuf& buf = *states_[f];
        if (buf.p != nullptr && capacity_ > 0) {
            std::vector<unsigned char> from(static_cast<size_t>(nSlots_) * capacity_ * eb);
            CUDA_CHECK(cudaMemcpy(from.data(), buf.p, from.size(), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nSlots_; s++) {
                std::copy_n(from.begin() + static_cast<size_t>(s) * capacity_ * eb, static_cast<size_t>(hCount_[s]) * eb,
                            to.begin() + static_cast<size_t>(s) * capacity * eb);
            }
        }
        buf.upload(to.data(), to.size());
    }
}

void SlotTable::resetState(const std::vector<int>& slots, cudaStream_t stream) {
    if (capacity_ == 0) {
        return;
    }
    for (size_t f = 0; f < stateFields_.size(); f++) {
        const size_t region = static_cast<size_t>(capacity_) * stateFields_[f].elementBytes;
        auto* p = static_cast<unsigned char*>(states_[f]->p);
        if (slots.empty()) {
            CUDA_CHECK(cudaMemsetAsync(p, stateFields_[f].initByte, region * nSlots_, stream));
            continue;
        }
        for (int s : slots) {
            CUDA_CHECK(cudaMemsetAsync(p + region * s, stateFields_[f].initByte, region, stream));
        }
    }
}

void SlotTable::layOut(int capacity) {
    std::vector<int> ints(static_cast<size_t>(nSlots_) * capacity * intWidth_);
    std::vector<double> doubles(static_cast<size_t>(nSlots_) * capacity * doubleWidth_);
    for (int s = 0; s < nSlots_; s++) {
        size_t from = static_cast<size_t>(s) * capacity_;
        size_t to = static_cast<size_t>(s) * capacity;
        std::copy_n(hInts_.begin() + from * intWidth_, static_cast<size_t>(hCount_[s]) * intWidth_, ints.begin() + to * intWidth_);
        std::copy_n(hDoubles_.begin() + from * doubleWidth_, static_cast<size_t>(hCount_[s]) * doubleWidth_,
                    doubles.begin() + to * doubleWidth_);
    }
    hInts_.swap(ints);
    hDoubles_.swap(doubles);
    capacity_ = capacity;
}

void SlotTable::install(int nSlots, const std::vector<int>& slots, int nEntries, const int* entryInts, const double* entryDoubles) {
    if (nSlots != nSlots_) {                                 // another bank: nothing to keep
        nSlots_ = nSlots;
        capacity_ = 0;
        hCount_.assign(nSlots, 0);
        hInts_.clear();
        hDoubles_.clear();
    }
    const int nLocal = slots.empty() ? nSlots : static_cast<int>(slots.size());
    auto bankSlot = [&](int j) { return slots.empty() ? j : slots[j]; };
    const int stride = 1 + intWidth_;
    std::vector<int> localCount(nLocal, 0);
    for (int k = 0; k < nEntries; k++) {
        int j = entryInts[static_cast<size_t>(k) * stride];
        if (j >= 0 && j < nLocal) {
            localCount[j]++;
        }
    }
    const int need = localCount.empty() ? 0 : *std::max_element(localCount.begin(), localCount.end());
    const bool grown = need > capacity_;
    if (grown) {
        layOutState(need);                                   // with the counts the other slots still have
        layOut(need);
    }
    for (int j = 0; j < nLocal; j++) {
        hCount_[bankSlot(j)] = 0;
    }
    for (int k = 0; k < nEntries; k++) {
        const int* e = entryInts + static_cast<size_t>(k) * stride;
        if (e[0] < 0 || e[0] >= nLocal) {
            continue;
        }
        int s = bankSlot(e[0]);
        size_t i = at(s, hCount_[s]++);
        std::copy_n(e + 1, intWidth_, hInts_.begin() + i * intWidth_);
        std::copy_n(entryDoubles + static_cast<size_t>(k) * doubleWidth_, doubleWidth_, hDoubles_.begin() + i * doubleWidth_);
    }
    total_ = 0;
    for (int c : hCount_) {
        total_ += c;
    }
    if (capacity_ == 0) {
        return;                                              // no slot ever held an entry: nothing on the device
    }
    resetState(slots, nullptr);                              // the installed slots' state starts afresh
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    // The counts go up whole (one int per slot), so a slot emptied while the device copy was skipped cannot keep stale
    // entries; the regions go up whole after a re-layout or an install of every slot, else slot by slot.
    count_.upload(hCount_.data(), hCount_.size() * sizeof(int));
    if (grown || slots.empty() || ints_.p == nullptr) {
        if (intWidth_ > 0) {
            ints_.upload(hInts_.data(), hInts_.size() * sizeof(int));
        }
        if (doubleWidth_ > 0) {
            doubles_.upload(hDoubles_.data(), hDoubles_.size() * sizeof(double));
        }
        return;
    }
    // Slot s's region is its capacity_ entries: row s of a per-slot array `capacity_ * width` wide.
    std::vector<int> bankSlots(nLocal);
    for (int j = 0; j < nLocal; j++) {
        bankSlots[j] = bankSlot(j);
    }
    if (intWidth_ > 0) {
        const size_t width = static_cast<size_t>(capacity_) * intWidth_;
        intUpload_.write(ints_.as<int>(), width, nLocal, bankSlots.data(),
                         [&](int j) { return hInts_.data() + static_cast<size_t>(bankSlots[j]) * width; });
    }
    if (doubleWidth_ > 0) {
        const size_t width = static_cast<size_t>(capacity_) * doubleWidth_;
        doubleUpload_.write(doubles_.as<double>(), width, nLocal, bankSlots.data(),
                            [&](int j) { return hDoubles_.data() + static_cast<size_t>(bankSlots[j]) * width; });
    }
}

SlotEntries SlotTable::view() const {
    SlotEntries v;
    v.capacity = capacity_;
    v.intWidth = intWidth_;
    v.doubleWidth = doubleWidth_;
    if (total_ > 0) {                                        // no count: every slot reads as empty
        v.count = static_cast<const int*>(count_.p);
        v.ints = static_cast<const int*>(ints_.p);
        v.doubles = static_cast<const double*>(doubles_.p);
    }
    return v;
}

}  // namespace olfgpu
