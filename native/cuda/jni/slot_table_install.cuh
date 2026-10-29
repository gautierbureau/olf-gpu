// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A per-scenario channel's Java lists into its slot table (solver/slot_table.cuh): per entry, its slot then the table's
// int fields, and the table's doubles per entry in a second array, installed into the context's install slots. A null
// or empty list leaves those slots empty.
#ifndef OLF_GPU_JNI_SLOT_TABLE_INSTALL_CUH
#define OLF_GPU_JNI_SLOT_TABLE_INSTALL_CUH

#include <jni.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../solver/batch_context.cuh"
#include "../solver/slot_rows.cuh"
#include "../solver/slot_table.cuh"

namespace olfgpu {

inline std::vector<int> jniInts(JNIEnv* env, jintArray a) {
    std::vector<int> v(a != nullptr ? env->GetArrayLength(a) : 0);
    if (!v.empty()) {
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    }
    return v;
}

inline std::vector<double> jniDoubles(JNIEnv* env, jdoubleArray a) {
    std::vector<double> v(a != nullptr ? env->GetArrayLength(a) : 0);
    if (!v.empty()) {
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    }
    return v;
}

// Entries that each own a run of a second list (a pilot's surviving controllers, a re-merged control's other
// controllers): per entry its slot and int fields, then the run's start within its slot's region of the second table
// and its length, and the run's items as (slot, item) entries of the second table, in the entries' order.
struct SlotRuns {
    std::vector<int> entryInts;
    std::vector<int> itemInts;
};

inline SlotRuns slotRuns(const std::vector<int>& slots, const std::vector<std::vector<int>>& fields,
                         const std::vector<int>& runLength, const std::vector<int>& items) {
    SlotRuns r;
    std::vector<int> next;                                   // per local slot: the items given so far
    size_t item = 0;
    for (size_t e = 0; e < slots.size(); e++) {
        int j = slots[e];
        if (j >= static_cast<int>(next.size()) && j >= 0) {
            next.resize(j + 1, 0);
        }
        r.entryInts.push_back(j);
        for (const auto& f : fields) {
            r.entryInts.push_back(f[e]);
        }
        r.entryInts.push_back(j >= 0 ? next[j] : 0);
        r.entryInts.push_back(runLength[e]);
        for (int k = 0; k < runLength[e]; k++) {
            r.itemInts.push_back(j);
            r.itemInts.push_back(items[item + k]);
        }
        item += runLength[e];
        if (j >= 0) {
            next[j] += runLength[e];
        }
    }
    return r;
}

inline void installSlotTable(JNIEnv* env, BatchSolverContext* ctx, SlotTable& table, jintArray jEntryInts, jdoubleArray jEntryDoubles) {
    std::vector<int> ints;
    std::vector<double> doubles;
    int nEntries = 0;
    if (jEntryInts != nullptr) {
        ints.resize(env->GetArrayLength(jEntryInts));
        env->GetIntArrayRegion(jEntryInts, 0, static_cast<jsize>(ints.size()), ints.data());
        nEntries = static_cast<int>(ints.size()) / (1 + table.intWidth());
    }
    if (table.doubleWidth() > 0) {
        int have = jEntryDoubles != nullptr ? env->GetArrayLength(jEntryDoubles) / table.doubleWidth() : 0;
        nEntries = std::min(nEntries, have);                 // an entry without its doubles is not one
        doubles.resize(static_cast<size_t>(nEntries) * table.doubleWidth());
        if (nEntries > 0) {
            env->GetDoubleArrayRegion(jEntryDoubles, 0, static_cast<jsize>(doubles.size()), doubles.data());
        }
    }
    table.install(ctx->nScenarios, ctx->installSlots, nEntries, ints.data(), doubles.data());
}

// The same from {key, value} lists whose key is slot * width + field (one double per entry).
inline void installKeyedSlotTable(JNIEnv* env, BatchSolverContext* ctx, SlotTable& table, int width, jintArray jKeys,
                                  jdoubleArray jValues) {
    std::vector<int> entries;
    std::vector<double> values;
    if (jKeys != nullptr && jValues != nullptr && width > 0) {
        int nEntries = std::min(env->GetArrayLength(jKeys), env->GetArrayLength(jValues));
        std::vector<int> keys(nEntries);
        values.resize(nEntries);
        env->GetIntArrayRegion(jKeys, 0, nEntries, keys.data());
        env->GetDoubleArrayRegion(jValues, 0, nEntries, values.data());
        entries.resize(2 * static_cast<size_t>(nEntries));
        for (int k = 0; k < nEntries; k++) {
            entries[2 * k] = keys[k] / width;
            entries[2 * k + 1] = keys[k] % width;
        }
    }
    table.install(ctx->nScenarios, ctx->installSlots, static_cast<int>(values.size()), entries.data(), values.data());
}

namespace detail {
inline void regionOf(JNIEnv* env, jdoubleArray a, jsize n, double* to) { env->GetDoubleArrayRegion(a, 0, n, to); }
inline void regionOf(JNIEnv* env, jintArray a, jsize n, int* to) { env->GetIntArrayRegion(a, 0, n, to); }
inline void regionOf(JNIEnv* env, jbyteArray a, jsize n, char* to) { env->GetByteArrayRegion(a, 0, n, reinterpret_cast<jbyte*>(to)); }
}  // namespace detail

// A dense per-scenario channel's Java rows into its SlotRows: `width` values per installed scenario, in the install's
// order; null leaves the installed slots without values. Another length is the caller's bug: an IllegalArgumentException.
template <typename T, typename JArray>
inline void installSlotRows(JNIEnv* env, BatchSolverContext* ctx, SlotRows<T>& rows, size_t width, JArray jRows, const char* who) {
    std::vector<T> values;
    if (jRows != nullptr && width > 0) {
        size_t want = static_cast<size_t>(installedSlotCount(ctx)) * width;
        if (static_cast<size_t>(env->GetArrayLength(jRows)) != want) {
            std::string message = std::string(who) + ": expected one row per installed scenario";
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), message.c_str());
            return;
        }
        values.resize(want);
        detail::regionOf(env, jRows, static_cast<jsize>(want), values.data());
    }
    rows.install(ctx->nScenarios, ctx->installSlots, width, values.empty() ? nullptr : values.data());
}

}  // namespace olfgpu

#endif  // OLF_GPU_JNI_SLOT_TABLE_INSTALL_CUH
