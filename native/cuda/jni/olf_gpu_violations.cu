// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// On-device limit-violation SCREENING.
//
// Detection was the largest block of GPU security-analysis wall time that is not a device solve
// (rte6515 at 2000 contingencies, 1 thread: 10.6 s of 39.3 s), and 98.4% of it is comparison
// arithmetic over data the device already holds — 34714 limit groups and 138817 thresholds per
// contingency on that network.
//
// This kernel SCREENS, it does not decide. It compares against a RELAXED threshold and emits EVERY
// exceeded entry of a group rather than stopping at the first, so it can only ever hand the host
// MORE candidates than OLF would report. The host then re-tests each with OLF's exact `>` on OLF's
// own value (GpuLimitScanner.detectFromCandidates). That is deliberate: a value computed here
// differs from the host's in the last bits (FMA contraction, a different sin/cos), and violations of
// an element that already violates at base are decided with ZERO tolerance — so the arithmetic here
// must not be allowed to decide anything.
//
// The flows come from the SAME generated device functions the fill kernels use — parsed from OLF's
// own equation-term Java — and are assembled the way OLF's vectorized network assembles them:
// i = hypot(p, q) / v but s = sqrt(p*p + q*q), which are not the same function in floating point.
#include <cuda_runtime.h>
#include <jni.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../closed_branch.cuh"
#include "../open_branch.cuh"

using namespace powsybl::openloadflow;

namespace {

constexpr int BLOCK = 256;
constexpr int FAMILY_CLOSED = 0;
constexpr int FAMILY_OPEN = 1;
constexpr int QTY_I = 0;
constexpr int QTY_P = 1;
constexpr int QTY_S = 2;

inline int grid(int nThreads) {
    return (nThreads + BLOCK - 1) / BLOCK;
}

#define VIOL_CUDA_CHECK(expr)                                                        \
    do {                                                                             \
        cudaError_t err_ = (expr);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            std::fprintf(stderr, "olfgpu violations: %s\n", cudaGetErrorString(err_));\
            return 0;                                                                \
        }                                                                            \
    } while (0)

/** One device allocation, freed with the table. */
struct Buf {
    void* p = nullptr;
    size_t bytes = 0;
    bool upload(const void* host, size_t n) {
        bytes = n;
        if (n == 0) {
            return true;
        }
        if (cudaMalloc(&p, n) != cudaSuccess) {
            return false;
        }
        return cudaMemcpy(p, host, n, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    bool alloc(size_t n) {
        bytes = n;
        return n == 0 || cudaMalloc(&p, n) == cudaSuccess;
    }
    template <typename T> T* as() const { return static_cast<T*>(p); }
    ~Buf() { if (p) { cudaFree(p); } }
};

/** The run's limit thresholds and the branch model they are compared against — uploaded once,
 *  reused by every chunk of every contingency, because a contingency cannot change a threshold. */
struct ViolationTable {
    int nBranches = 0, nGroups = 0, nLimits = 0, nBuses = 0;
    Buf brFamily, brElem, brGroupStart, brGroupEnd;
    Buf grpSide, grpQty, grpLimStart, grpLimEnd;
    Buf limThreshold;
    Buf busVRow, busHigh, busLow;
    Buf cbIn, cbIdx, obIn, obIdx;
    // Scratch, grown on demand: the states of one chunk and its records.
    Buf states, records, counts, values;
    size_t statesCap = 0, recordsCap = 0, countsCap = 0, valuesCap = 0;
};

/** Emit one candidate, or count past the capacity so the host can fall back for this scenario. With
 *  {@code values} non-null (DECIDE mode) the value the violation reports goes alongside. */
__device__ inline void emit(int* records, int* counts, int scenario, int capacity, int record,
                            double* values = nullptr, double value = 0.0) {
    int slot = atomicAdd(counts + scenario, 1);
    if (slot < capacity) {
        records[(size_t) scenario * capacity + slot] = record;
        if (values != nullptr) {
            values[(size_t) scenario * capacity + slot] = value;
        }
    }
}

__global__ void screenBranches(int nBranches, int nScenarios, int n, int capacity, double band,
                               const int* brFamily, const int* brElem, const int* brGroupStart, const int* brGroupEnd,
                               const int* grpSide, const int* grpQty, const int* grpLimStart, const int* grpLimEnd,
                               const double* limThreshold,
                               const double* cbIn, const int* cbIdx, const double* obIn, const int* obIdx,
                               const double* x, int* records, int* counts, double* values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nBranches * nScenarios) {
        return;
    }
    int b = tid % nBranches;
    int s = tid / nBranches;
    const double* xs = x + (size_t) s * n;

    double p1 = nan(""), q1 = nan(""), p2 = nan(""), q2 = nan(""), v1 = nan(""), v2 = nan("");
    int family = brFamily[b];
    int e = brElem[b];
    if (family == FAMILY_CLOSED) {
        const double* p = cbIn + (size_t) e * 8;             // y g1 g2 b1 b2 ksi r1 a1
        const int* q = cbIdx + (size_t) e * 12;              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
        if (q[4] < 0 || q[5] < 0 || q[6] < 0 || q[7] < 0) {
            return;
        }
        double y = p[0], g1 = p[1], g2 = p[2], b1 = p[3], b2 = p[4], ksi = p[5];
        // A controlled ratio / phase shift is a STATE variable, so this scenario's own value is in
        // the state vector and the pack carries only the base tap — the same choice the fill kernels
        // make. Reading the pack here would evaluate the wrong tap for every controlled transformer.
        double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
        double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
        double ph1 = xs[q[4]];
        v1 = xs[q[5]];
        double ph2 = xs[q[6]];
        v2 = xs[q[7]];
        double theta1 = ksi - a1 + A2 - ph1 + ph2;
        double theta2 = ksi + a1 - A2 + ph1 - ph2;
        double sinKsi = sin(ksi);
        double cosKsi = cos(ksi);
        p1 = closed_branch::p1(y, sinKsi, g1, v1, r1, v2, sin(theta1));
        q1 = closed_branch::q1(y, cosKsi, b1, v1, r1, v2, cos(theta1));
        p2 = closed_branch::p2(y, sinKsi, g2, v1, r1, v2, sin(theta2));
        q2 = closed_branch::q2(y, cosKsi, b2, v1, r1, v2, cos(theta2));
    } else if (family == FAMILY_OPEN) {
        const double* p = obIn + (size_t) e * 7;             // y ksi g1 b1 g2 b2 r1
        const int* q = obIdx + (size_t) e * 6;               // P1 Q1 P2 Q2 v1col v2col
        double y = p[0], ksi = p[1], g1 = p[2], b1 = p[3], g2 = p[4], b2 = p[5], r1 = p[6];
        double sinKsi = sin(ksi);
        double cosKsi = cos(ksi);
        if (q[4] >= 0) {                                     // side 2 open: only side 1 carries a flow
            v1 = xs[q[4]];
            p1 = open_branch::p1(y, cosKsi, sinKsi, g1, g2, b2, v1, r1);
            q1 = open_branch::q1(y, cosKsi, sinKsi, b1, g2, b2, v1, r1);
        }
        if (q[5] >= 0) {                                     // side 1 open
            v2 = xs[q[5]];
            p2 = open_branch::p2(y, cosKsi, sinKsi, g1, b1, g2, v2);
            q2 = open_branch::q2(y, cosKsi, sinKsi, g1, b1, b2, v2);
        }
    } else {
        return;                                              // not in the device model — host scan
    }

    for (int g = brGroupStart[b]; g < brGroupEnd[b]; g++) {
        double p = grpSide[g] == 1 ? p1 : p2;
        double q = grpSide[g] == 1 ? q1 : q2;
        double v = grpSide[g] == 1 ? v1 : v2;
        double value;
        switch (grpQty[g]) {
            case QTY_I: value = hypot(p, q) / v; break;
            case QTY_P: value = fabs(p); break;
            default: value = sqrt(p * p + q * q); break;
        }
        if (isnan(value)) {
            continue;
        }
        if (values != nullptr) {
            // DECIDE: OLF's own rule — the FIRST limit of the group, in table order, whose threshold the value
            // exceeds with a strict >; reported with the value OLF reports (active power keeps its sign).
            for (int l = grpLimStart[g]; l < grpLimEnd[g]; l++) {
                if (value > limThreshold[l]) {
                    emit(records, counts, s, capacity, l, values, grpQty[g] == QTY_P ? p : value);
                    break;
                }
            }
            continue;
        }
        for (int l = grpLimStart[g]; l < grpLimEnd[g]; l++) {
            double threshold = limThreshold[l];
            if (value > threshold - fabs(threshold) * band) {
                emit(records, counts, s, capacity, l);
            }
        }
    }
}

__global__ void screenBuses(int nBuses, int nScenarios, int n, int capacity, double band, int busRecordBase,
                            const int* busVRow, const double* busHigh, const double* busLow,
                            const double* x, int* records, int* counts, double* values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nBuses * nScenarios) {
        return;
    }
    int i = tid % nBuses;
    int s = tid / nBuses;
    int row = busVRow[i];
    if (row < 0) {
        return;
    }
    double v = x[(size_t) s * n + row];
    double high = busHigh[i];
    double low = busLow[i];
    // A NaN limit makes the comparison false, which is exactly what OLF's !isNaN(limit) guard does.
    if (values != nullptr) {                                 // DECIDE: the exact comparison, on the state's own V
        if (v > high) {
            emit(records, counts, s, capacity, busRecordBase + i * 2, values, v);
        } else if (v < low) {
            emit(records, counts, s, capacity, busRecordBase + i * 2 + 1, values, v);
        }
        return;
    }
    if (v > high - fabs(high) * band) {
        emit(records, counts, s, capacity, busRecordBase + i * 2);
    }
    if (v < low + fabs(low) * band) {
        emit(records, counts, s, capacity, busRecordBase + i * 2 + 1);
    }
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_uploadViolationTable(
        JNIEnv* env, jclass,
        jintArray jBrFamily, jintArray jBrElem, jintArray jBrGroupStart, jintArray jBrGroupEnd,
        jintArray jGrpSide, jintArray jGrpQty, jintArray jGrpLimStart, jintArray jGrpLimEnd,
        jdoubleArray jLimThreshold,
        jintArray jBusVRow, jdoubleArray jBusHigh, jdoubleArray jBusLow,
        jdoubleArray jCbIn, jintArray jCbIdx, jdoubleArray jObIn, jintArray jObIdx) {
    auto* t = new ViolationTable();
    t->nBranches = env->GetArrayLength(jBrFamily);
    t->nGroups = env->GetArrayLength(jGrpSide);
    t->nLimits = env->GetArrayLength(jLimThreshold);
    t->nBuses = env->GetArrayLength(jBusVRow);

    auto uploadI = [&](jintArray a, Buf& buf) {
        jint* p = env->GetIntArrayElements(a, nullptr);
        bool ok = buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, p, JNI_ABORT);
        return ok;
    };
    auto uploadD = [&](jdoubleArray a, Buf& buf) {
        jdouble* p = env->GetDoubleArrayElements(a, nullptr);
        bool ok = buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, p, JNI_ABORT);
        return ok;
    };

    bool ok = uploadI(jBrFamily, t->brFamily) && uploadI(jBrElem, t->brElem)
            && uploadI(jBrGroupStart, t->brGroupStart) && uploadI(jBrGroupEnd, t->brGroupEnd)
            && uploadI(jGrpSide, t->grpSide) && uploadI(jGrpQty, t->grpQty)
            && uploadI(jGrpLimStart, t->grpLimStart) && uploadI(jGrpLimEnd, t->grpLimEnd)
            && uploadD(jLimThreshold, t->limThreshold)
            && uploadI(jBusVRow, t->busVRow) && uploadD(jBusHigh, t->busHigh) && uploadD(jBusLow, t->busLow)
            && uploadD(jCbIn, t->cbIn) && uploadI(jCbIdx, t->cbIdx)
            && uploadD(jObIn, t->obIn) && uploadI(jObIdx, t->obIdx);
    if (!ok) {
        delete t;
        return 0;
    }
    return reinterpret_cast<jlong>(t);
}

JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_destroyViolationTable(
        JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<ViolationTable*>(handle);
}

/** Run the screen (values == false) or the decision (values == true) over one chunk of states, leaving counts,
 *  records and, when deciding, values on the device. False on any CUDA failure. */
static bool runChunk(JNIEnv* env, ViolationTable* t, jdoubleArray jStates, jint nScenarios, jint n,
                     jdouble band, jint capacity, bool decide) {
    size_t stateBytes = (size_t) nScenarios * n * sizeof(double);
    size_t recordBytes = (size_t) nScenarios * capacity * sizeof(int);
    size_t valueBytes = decide ? (size_t) nScenarios * capacity * sizeof(double) : 0;
    size_t countBytes = (size_t) nScenarios * sizeof(int);
    // The chunk buffers are reused across chunks and only ever grow, so a run pays one allocation.
    auto grow = [](Buf& buf, size_t& cap, size_t bytes) {
        if (cap >= bytes) {
            return true;
        }
        if (buf.p) { cudaFree(buf.p); buf.p = nullptr; }
        if (!buf.alloc(bytes)) { return false; }
        cap = bytes;
        return true;
    };
    if (!grow(t->states, t->statesCap, stateBytes) || !grow(t->records, t->recordsCap, recordBytes)
            || !grow(t->counts, t->countsCap, countBytes) || !grow(t->values, t->valuesCap, valueBytes)) {
        return false;
    }
    jdouble* states = env->GetDoubleArrayElements(jStates, nullptr);
    cudaError_t err = cudaMemcpy(t->states.p, states, stateBytes, cudaMemcpyHostToDevice);
    env->ReleaseDoubleArrayElements(jStates, states, JNI_ABORT);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "olfgpu violations: state upload: %s\n", cudaGetErrorString(err));
        return false;
    }
    if (cudaMemset(t->counts.p, 0, countBytes) != cudaSuccess) {
        return false;
    }
    double* values = decide ? t->values.as<double>() : nullptr;
    screenBranches<<<grid(t->nBranches * nScenarios), BLOCK>>>(
            t->nBranches, nScenarios, n, capacity, band,
            t->brFamily.as<int>(), t->brElem.as<int>(), t->brGroupStart.as<int>(), t->brGroupEnd.as<int>(),
            t->grpSide.as<int>(), t->grpQty.as<int>(), t->grpLimStart.as<int>(), t->grpLimEnd.as<int>(),
            t->limThreshold.as<double>(),
            t->cbIn.as<double>(), t->cbIdx.as<int>(), t->obIn.as<double>(), t->obIdx.as<int>(),
            t->states.as<double>(), t->records.as<int>(), t->counts.as<int>(), values);
    if (t->nBuses > 0) {
        screenBuses<<<grid(t->nBuses * nScenarios), BLOCK>>>(
                t->nBuses, nScenarios, n, capacity, band, t->nLimits,
                t->busVRow.as<int>(), t->busHigh.as<double>(), t->busLow.as<double>(),
                t->states.as<double>(), t->records.as<int>(), t->counts.as<int>(), values);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "olfgpu violations: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * Screen a chunk of scenarios. Returns nScenarios counts followed by nScenarios * capacity records;
 * a count ABOVE the capacity means the scenario overflowed and the host must scan it in full.
 */
JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_screenViolationsBatch(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jStates, jint nScenarios, jint n,
        jdouble band, jint capacity) {
    auto* t = reinterpret_cast<ViolationTable*>(handle);
    if (t == nullptr || nScenarios <= 0 || !runChunk(env, t, jStates, nScenarios, n, band, capacity, false)) {
        return nullptr;
    }
    size_t recordBytes = (size_t) nScenarios * capacity * sizeof(int);
    size_t countBytes = (size_t) nScenarios * sizeof(int);
    jintArray out = env->NewIntArray(nScenarios + (jint) ((size_t) nScenarios * capacity));
    if (out == nullptr) {
        return nullptr;
    }
    jint* host = env->GetIntArrayElements(out, nullptr);
    cudaError_t err = cudaMemcpy(host, t->counts.p, countBytes, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        err = cudaMemcpy(host + nScenarios, t->records.p, recordBytes, cudaMemcpyDeviceToHost);
    }
    env->ReleaseIntArrayElements(out, host, 0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "olfgpu violations: record readback: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return out;
}

/**
 * DECIDE a chunk of scenarios: OLF's own comparison on the device's values (EXPERIMENTAL — the values differ
 * from the host's in the last bits, so a value exactly at a threshold may be decided differently). Returns
 * nScenarios counts, then nScenarios * capacity records, then nScenarios * capacity values, all as doubles
 * (a record index is exact in a double); a count above the capacity means the host must decide that scenario.
 */
JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_decideViolationsBatch(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jStates, jint nScenarios, jint n, jint capacity) {
    auto* t = reinterpret_cast<ViolationTable*>(handle);
    if (t == nullptr || nScenarios <= 0 || !runChunk(env, t, jStates, nScenarios, n, 0.0, capacity, true)) {
        return nullptr;
    }
    size_t slots = (size_t) nScenarios * capacity;
    std::vector<int> counts(nScenarios);
    std::vector<int> records(slots);
    std::vector<double> out((size_t) nScenarios + 2 * slots);
    cudaError_t err = cudaMemcpy(counts.data(), t->counts.p, nScenarios * sizeof(int), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        err = cudaMemcpy(records.data(), t->records.p, slots * sizeof(int), cudaMemcpyDeviceToHost);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(out.data() + nScenarios + slots, t->values.p, slots * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "olfgpu violations: decision readback: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    for (int i = 0; i < nScenarios; i++) {
        out[i] = counts[i];
        int emitted = counts[i] < capacity ? counts[i] : capacity;
        for (int k = 0; k < emitted; k++) {
            out[nScenarios + (size_t) i * capacity + k] = records[(size_t) i * capacity + k];
        }
    }
    jdoubleArray jout = env->NewDoubleArray((jsize) out.size());
    if (jout == nullptr) {
        return nullptr;
    }
    env->SetDoubleArrayRegion(jout, 0, (jsize) out.size(), out.data());
    return jout;
}

}  // extern "C"
