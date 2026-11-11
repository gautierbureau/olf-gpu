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
// This kernel SCREENS, it does not decide. It compares against a RELAXED threshold and keeps emitting a
// group's exceeded entries until one is exceeded BEYOND the band (certain to confirm on the host, which
// reports the first confirmed entry of a group), so it can only ever hand the host MORE candidates than
// OLF would report. The host then re-tests each with OLF's exact `>` on OLF's
// own value (GpuLimitScanner.detectFromCandidates). That is deliberate: a value computed here
// differs from the host's in the last bits (FMA contraction, a different sin/cos), and violations of
// an element that already violates at base are decided with ZERO tolerance — so the arithmetic here
// must not be allowed to decide anything.
//
// The flows come from the SAME generated device functions the fill kernels use — parsed from OLF's
// own equation-term Java — and are assembled the way OLF's vectorized network assembles them:
// i = hypot(p, q) / v but s = sqrt(p*p + q*q), which are not the same function in floating point.
//
// Each scenario is screened as the device solved it (GpuScreenScenario): a branch it takes out emits nothing, a
// base-open branch emits only where the scenario closes it, a branch it opens at one side is evaluated at the other with
// OLF's open-branch formula (as OLF's open terms are, not with the dangling row the solve uses, whose active power loses
// relative precision to a cancellation), and a branch it gives another pi row (a tap action, a tap the outer loops
// moved) is evaluated with that row, a ratio or phase shift that is a state variable still coming from the state — the
// solve's fill (branch_fill.cu) makes the same choices.
#include <cuda_runtime.h>
#include <jni.h>
#include "../jni/olf_jni.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../equations/generated/closed_branch.cuh"
#include "../equations/generated/open_branch.cuh"
#include "../common/codes/gpu_two_sides.cuh"
#include "../solver/handed_back_states.cuh"

using namespace powsybl::openloadflow;

namespace {

constexpr int BLOCK = 256;
constexpr int FAMILY_CLOSED = 0;
constexpr int FAMILY_OPEN = 1;
constexpr int QTY_I = 0;
constexpr int QTY_P = 1;
constexpr int QTY_S = 2;
// GpuScreenScenario's entry bits
constexpr int ENTRY_OUT = 1;
constexpr int ENTRY_IN = 2;
constexpr int ENTRY_ROW = 4;
constexpr int ENTRY_OPEN_1 = 8;
constexpr int ENTRY_OPEN_2 = 16;

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
    Buf brFamily, brElem, brBaseOpen, brGroupStart, brGroupEnd;
    Buf grpSide, grpQty, grpLimStart, grpLimEnd;
    Buf limThreshold;
    Buf busVRow, busHigh, busLow;
    Buf closedBranchParameters, closedBranchIndices, openBranchParameters, openBranchIndices;
    // Scratch, grown on demand: the states of one chunk, its scenario entries and its records; for a batch context's
    // handed-back states, their slots, and the rows gathered from them.
    Buf states, records, counts;
    Buf entryStart, entryBranch, entryKind, entryRow;
    Buf slots, gatherScenario, gatherRow, gathered;
    size_t statesCap = 0, recordsCap = 0, countsCap = 0;
    size_t entryStartCap = 0, entryBranchCap = 0, entryKindCap = 0, entryRowCap = 0;
    size_t slotsCap = 0, gatherScenarioCap = 0, gatherRowCap = 0, gatheredCap = 0;
};

/** Grow-only scratch: keeps the buffer when it is big enough (an empty request still gets a few bytes, so a kernel never
 *  reads a null pointer). */
bool grow(Buf& buf, size_t& cap, size_t bytes) {
    if (bytes == 0) {
        bytes = sizeof(double);
    }
    if (cap >= bytes) {
        return true;
    }
    if (buf.p) { cudaFree(buf.p); buf.p = nullptr; }
    if (!buf.alloc(bytes)) { return false; }
    cap = bytes;
    return true;
}

/** Scenario s's state: row s of a chunk's own states, or the bank slot slotOf[s] of a batch context's (n per slot). */
__device__ inline const double* stateOf(const double* x, const int* slotOf, int s, int n) {
    return x + (size_t) (slotOf != nullptr ? slotOf[s] : s) * n;
}

/** Emit one candidate, or count past the capacity so the host can fall back for this scenario. */
__device__ inline void emit(int* records, int* counts, int scenario, int capacity, int record) {
    int recordIndex = atomicAdd(counts + scenario, 1);
    if (recordIndex < capacity) {
        records[(size_t) scenario * capacity + recordIndex] = record;
    }
}

/** Scenario s's entry for table branch b (its entries are ascending by branch), or -1. */
__device__ inline int entryOf(const int* entryStart, const int* entryBranch, int s, int b) {
    for (int k = entryStart[s]; k < entryStart[s + 1] && entryBranch[k] <= b; k++) {
        if (entryBranch[k] == b) {
            return k;
        }
    }
    return -1;
}

__global__ void screenBranches(int nBranches, int nScenarios, int n, int capacity, double band,
                               const int* brFamily, const int* brElem, const int* brBaseOpen, const int* brGroupStart,
                               const int* brGroupEnd,
                               const int* grpSide, const int* grpQty, const int* grpLimStart, const int* grpLimEnd,
                               const double* limThreshold,
                               const double* closedBranchParameters, const int* closedBranchIndices, const double* openBranchParameters, const int* openBranchIndices,
                               const int* entryStart, const int* entryBranch, const int* entryKind, const double* entryRow,
                               const double* x, const int* slotOf, int* records, int* counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nBranches * nScenarios) {
        return;
    }
    int b = tid % nBranches;
    int s = tid / nBranches;
    const double* xs = stateOf(x, slotOf, s, n);
    int entry = entryOf(entryStart, entryBranch, s, b);
    int kind = entry >= 0 ? entryKind[entry] : 0;
    if ((kind & ENTRY_OUT) != 0 || (brBaseOpen[b] != 0 && (kind & ENTRY_IN) == 0)) {
        return;                                              // out of service in this scenario
    }

    double p1 = nan(""), q1 = nan(""), p2 = nan(""), q2 = nan(""), v1 = nan(""), v2 = nan("");
    int family = brFamily[b];
    int e = brElem[b];
    if (family == FAMILY_CLOSED && (kind & (ENTRY_OPEN_1 | ENTRY_OPEN_2)) != 0) {
        // opened at one side by the scenario: OLF's open terms at the other, from the pack's row or the scenario's
        const double* p = (kind & ENTRY_ROW) != 0 ? entryRow + (size_t) entry * 8
                : closedBranchParameters + (size_t) e * 8;   // y g1 g2 b1 b2 ksi r1 a1
        const int* q = closedBranchIndices + (size_t) e * 12;              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
        double y = p[0], g1 = p[1], g2 = p[2], b1 = p[3], b2 = p[4], ksi = p[5], r1 = p[6];
        double sinKsi = sin(ksi);
        double cosKsi = cos(ksi);
        if ((kind & ENTRY_OPEN_2) != 0) {                    // side 2 open: only side 1 carries a flow
            if (q[5] < 0) {
                return;
            }
            v1 = xs[q[5]];
            p1 = open_branch::p1(y, cosKsi, sinKsi, g1, g2, b2, v1, r1);
            q1 = open_branch::q1(y, cosKsi, sinKsi, b1, g2, b2, v1, r1);
        } else {                                             // side 1 open
            if (q[7] < 0) {
                return;
            }
            v2 = xs[q[7]];
            p2 = open_branch::p2(y, cosKsi, sinKsi, g1, b1, g2, v2);
            q2 = open_branch::q2(y, cosKsi, sinKsi, g1, b1, b2, v2);
        }
    } else if (family == FAMILY_CLOSED) {
        const double* p = (kind & ENTRY_ROW) != 0 ? entryRow + (size_t) entry * 8   // this scenario's row
                : closedBranchParameters + (size_t) e * 8;   // y g1 g2 b1 b2 ksi r1 a1
        const int* q = closedBranchIndices + (size_t) e * 12;              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
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
        const double* p = openBranchParameters + (size_t) e * 7;             // y ksi g1 b1 g2 b2 r1
        const int* q = openBranchIndices + (size_t) e * 6;               // P1 Q1 P2 Q2 v1col v2col
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
        double p = grpSide[g] == GpuTwoSides::ONE ? p1 : p2;
        double q = grpSide[g] == GpuTwoSides::ONE ? q1 : q2;
        double v = grpSide[g] == GpuTwoSides::ONE ? v1 : v2;
        double value;
        switch (grpQty[g]) {
            case QTY_I: value = hypot(p, q) / v; break;
            case QTY_P: value = fabs(p); break;
            default: value = sqrt(p * p + q * q); break;
        }
        if (isnan(value)) {
            continue;
        }
        for (int l = grpLimStart[g]; l < grpLimEnd[g]; l++) {
            double threshold = limThreshold[l];
            if (value > threshold - fabs(threshold) * band) {
                emit(records, counts, s, capacity, l);
                // Exceeded BEYOND the band: the host's exact re-test (its value differs from ours by up to 2.1e-10
                // relative, measured on currents; the band is 1e-7) confirms this limit, and it reports the first
                // confirmed limit of a group and never looks further — so the group's remaining limits can never be
                // reported.
                if (value > threshold + fabs(threshold) * band) {
                    break;
                }
            }
        }
    }
}

__global__ void screenBuses(int nBuses, int nScenarios, int n, int capacity, double band, int busRecordBase,
                            const int* busVRow, const double* busHigh, const double* busLow,
                            const double* x, const int* slotOf, int* records, int* counts) {
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
    double v = stateOf(x, slotOf, s, n)[row];
    double high = busHigh[i];
    double low = busLow[i];
    // A NaN limit makes the comparison false, which is exactly what OLF's !isNaN(limit) guard does.
    if (v > high - fabs(high) * band) {
        emit(records, counts, s, capacity, busRecordBase + i * 2);
    }
    if (v < low + fabs(low) * band) {
        emit(records, counts, s, capacity, busRecordBase + i * 2 + 1);
    }
}

// The buses outside the realistic-voltage band, for the host's realistic check (OLF's own comparisons, no arithmetic: the
// host decides each one exactly as it would over every bus). A NaN voltage compares false, as on the host.
__global__ void screenRealistic(int nBuses, int nScenarios, int n, int capacity, double minV, double maxV, int realisticRecordBase,
                                const int* busVRow, const double* x, const int* slotOf, int* records, int* counts) {
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
    double v = stateOf(x, slotOf, s, n)[row];
    if (v < minV || v > maxV) {
        emit(records, counts, s, capacity, realisticRecordBase + i);
    }
}

// The full states of the handed-back scenarios which[j] (indices in the result rows), one after the other.
__global__ void gatherStates(int count, int n, const int* which, const int* slotOf, const double* x, double* out) {
    size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t) count * n) {
        return;
    }
    size_t j = tid / n;
    size_t k = tid % n;
    out[tid] = stateOf(x, slotOf, which[j], n)[k];
}

// One gathered value per (scenario, row) request.
__global__ void gatherRows(int nRows, int n, const int* gatherScenario, const int* gatherRow, const double* x,
                           const int* slotOf, double* out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nRows) {
        return;
    }
    out[k] = stateOf(x, slotOf, gatherScenario[k], n)[gatherRow[k]];
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
OLF_JNI(sa_gpu, ViolationScreenNative, uploadViolationTable)(
        JNIEnv* env, jclass,
        jintArray jBrFamily, jintArray jBrElem, jintArray jBrBaseOpen, jintArray jBrGroupStart, jintArray jBrGroupEnd,
        jintArray jGrpSide, jintArray jGrpQty, jintArray jGrpLimStart, jintArray jGrpLimEnd,
        jdoubleArray jLimThreshold,
        jintArray jControllerBusVRow, jdoubleArray jBusHigh, jdoubleArray jBusLow,
        jdoubleArray jClosedBranchParameters, jintArray jClosedBranchIndices, jdoubleArray jOpenBranchParameters, jintArray jOpenBranchIndices) {
    auto* t = new ViolationTable();
    t->nBranches = env->GetArrayLength(jBrFamily);
    t->nGroups = env->GetArrayLength(jGrpSide);
    t->nLimits = env->GetArrayLength(jLimThreshold);
    t->nBuses = env->GetArrayLength(jControllerBusVRow);

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

    bool ok = uploadI(jBrFamily, t->brFamily) && uploadI(jBrElem, t->brElem) && uploadI(jBrBaseOpen, t->brBaseOpen)
            && uploadI(jBrGroupStart, t->brGroupStart) && uploadI(jBrGroupEnd, t->brGroupEnd)
            && uploadI(jGrpSide, t->grpSide) && uploadI(jGrpQty, t->grpQty)
            && uploadI(jGrpLimStart, t->grpLimStart) && uploadI(jGrpLimEnd, t->grpLimEnd)
            && uploadD(jLimThreshold, t->limThreshold)
            && uploadI(jControllerBusVRow, t->busVRow) && uploadD(jBusHigh, t->busHigh) && uploadD(jBusLow, t->busLow)
            && uploadD(jClosedBranchParameters, t->closedBranchParameters) && uploadI(jClosedBranchIndices, t->closedBranchIndices)
            && uploadD(jOpenBranchParameters, t->openBranchParameters) && uploadI(jOpenBranchIndices, t->openBranchIndices);
    if (!ok) {
        delete t;
        return 0;
    }
    return reinterpret_cast<jlong>(t);
}

JNIEXPORT void JNICALL
OLF_JNI(sa_gpu, ViolationScreenNative, destroyViolationTable)(
        JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<ViolationTable*>(handle);
}

/** Upload a chunk's scenario entries (GpuScreenScenario as one CSR) and size its record buffers. False on any failure. */
static bool prepareChunk(JNIEnv* env, ViolationTable* t, jint nScenarios, jint capacity, jintArray jEntryStart,
                         jintArray jEntryBranch, jintArray jEntryKind, jdoubleArray jEntryRow) {
    size_t recordBytes = (size_t) nScenarios * capacity * sizeof(int);
    size_t countBytes = (size_t) nScenarios * sizeof(int);
    if (env->GetArrayLength(jEntryStart) != nScenarios + 1) {
        return false;
    }
    size_t nEntries = (size_t) env->GetArrayLength(jEntryBranch);
    // The chunk buffers are reused across chunks and only ever grow, so a run pays one allocation.
    if (!grow(t->records, t->recordsCap, recordBytes) || !grow(t->counts, t->countsCap, countBytes)
            || !grow(t->entryStart, t->entryStartCap, ((size_t) nScenarios + 1) * sizeof(int))
            || !grow(t->entryBranch, t->entryBranchCap, nEntries * sizeof(int))
            || !grow(t->entryKind, t->entryKindCap, nEntries * sizeof(int))
            || !grow(t->entryRow, t->entryRowCap, nEntries * 8 * sizeof(double))) {
        return false;
    }
    auto copyI = [&](jintArray a, Buf& buf) {
        size_t bytes = (size_t) env->GetArrayLength(a) * sizeof(int);
        if (bytes == 0) {
            return true;
        }
        jint* p = env->GetIntArrayElements(a, nullptr);
        bool ok = cudaMemcpy(buf.p, p, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        env->ReleaseIntArrayElements(a, p, JNI_ABORT);
        return ok;
    };
    size_t rowBytes = (size_t) env->GetArrayLength(jEntryRow) * sizeof(double);
    bool rowsOk = true;
    if (rowBytes > 0) {
        jdouble* p = env->GetDoubleArrayElements(jEntryRow, nullptr);
        rowsOk = cudaMemcpy(t->entryRow.p, p, rowBytes, cudaMemcpyHostToDevice) == cudaSuccess;
        env->ReleaseDoubleArrayElements(jEntryRow, p, JNI_ABORT);
    }
    if (!rowsOk || !copyI(jEntryStart, t->entryStart) || !copyI(jEntryBranch, t->entryBranch)
            || !copyI(jEntryKind, t->entryKind)) {
        std::fprintf(stderr, "olfgpu violations: scenario entries upload failed\n");
        return false;
    }
    return cudaMemset(t->counts.p, 0, countBytes) == cudaSuccess;
}

/** Screen nScenarios states of n doubles (x, row s or bank slot slotOf[s]) into the chunk's counts and records; the
 *  realistic-voltage band too when realistic. False on any CUDA failure. */
static bool launchScreen(ViolationTable* t, jint nScenarios, jint n, jdouble band, jint capacity, const double* x,
                         const int* slotOf, bool realistic, double minV, double maxV) {
    screenBranches<<<grid(t->nBranches * nScenarios), BLOCK>>>(
            t->nBranches, nScenarios, n, capacity, band,
            t->brFamily.as<int>(), t->brElem.as<int>(), t->brBaseOpen.as<int>(), t->brGroupStart.as<int>(),
            t->brGroupEnd.as<int>(),
            t->grpSide.as<int>(), t->grpQty.as<int>(), t->grpLimStart.as<int>(), t->grpLimEnd.as<int>(),
            t->limThreshold.as<double>(),
            t->closedBranchParameters.as<double>(), t->closedBranchIndices.as<int>(), t->openBranchParameters.as<double>(), t->openBranchIndices.as<int>(),
            t->entryStart.as<int>(), t->entryBranch.as<int>(), t->entryKind.as<int>(), t->entryRow.as<double>(),
            x, slotOf, t->records.as<int>(), t->counts.as<int>());
    if (t->nBuses > 0) {
        screenBuses<<<grid(t->nBuses * nScenarios), BLOCK>>>(
                t->nBuses, nScenarios, n, capacity, band, t->nLimits,
                t->busVRow.as<int>(), t->busHigh.as<double>(), t->busLow.as<double>(),
                x, slotOf, t->records.as<int>(), t->counts.as<int>());
        if (realistic) {
            screenRealistic<<<grid(t->nBuses * nScenarios), BLOCK>>>(
                    t->nBuses, nScenarios, n, capacity, minV, maxV, t->nLimits + 2 * t->nBuses,
                    t->busVRow.as<int>(), x, slotOf, t->records.as<int>(), t->counts.as<int>());
        }
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "olfgpu violations: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

/** Run the screen over one chunk of states, leaving counts and records on the device. False on any CUDA failure. */
static bool runChunk(JNIEnv* env, ViolationTable* t, jobject jStates, jint nScenarios, jint n,
                     jdouble band, jint capacity, jintArray jEntryStart, jintArray jEntryBranch, jintArray jEntryKind,
                     jdoubleArray jEntryRow) {
    size_t stateBytes = (size_t) nScenarios * n * sizeof(double);
    if (!grow(t->states, t->statesCap, stateBytes)
            || !prepareChunk(env, t, nScenarios, capacity, jEntryStart, jEntryBranch, jEntryKind, jEntryRow)) {
        return false;
    }
    // The chunk's states sit in a direct buffer: read in place, no JNI copy of them first.
    const void* states = env->GetDirectBufferAddress(jStates);
    if (states == nullptr || env->GetDirectBufferCapacity(jStates) < static_cast<jlong>(stateBytes)) {
        return false;
    }
    cudaError_t err = cudaMemcpy(t->states.p, states, stateBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "olfgpu violations: state upload: %s\n", cudaGetErrorString(err));
        return false;
    }
    return launchScreen(t, nScenarios, n, band, capacity, t->states.as<double>(), nullptr, false, 0, 0);
}

/** The chunk's counts then records, as one int array (see screenViolationsBatch), or null on failure. */
static jintArray readRecords(JNIEnv* env, ViolationTable* t, jint nScenarios, jint capacity) {
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
 * Screen a chunk of scenarios. Returns nScenarios counts followed by nScenarios * capacity records;
 * a count ABOVE the capacity means the scenario overflowed and the host must scan it in full.
 */
JNIEXPORT jintArray JNICALL
OLF_JNI(sa_gpu, ViolationScreenNative, screenViolationsBatch)(
        JNIEnv* env, jclass, jlong handle, jobject jStates, jint nScenarios, jint n,
        jdouble band, jint capacity, jintArray jEntryStart, jintArray jEntryBranch, jintArray jEntryKind,
        jdoubleArray jEntryRow) {
    auto* t = reinterpret_cast<ViolationTable*>(handle);
    if (t == nullptr || nScenarios <= 0
            || !runChunk(env, t, jStates, nScenarios, n, band, capacity, jEntryStart, jEntryBranch, jEntryKind, jEntryRow)) {
        return nullptr;
    }
    return readRecords(env, t, nScenarios, capacity);
}

/**
 * Screen the states a batched context's last solve call handed back, where they are (the context keeps its states on the
 * device: BatchSolverContext::statesToHost off), in the order of its result rows: the same records as
 * screenViolationsBatch, plus a record realisticRecordBase + bus (realisticRecordBase = limits + 2 x buses) for every bus
 * outside [minRealisticV, maxRealisticV] when realistic. Null on failure.
 */
JNIEXPORT jintArray JNICALL
OLF_JNI(sa_gpu, ViolationScreenNative, screenHandedBack)(
        JNIEnv* env, jclass, jlong handle, jlong batchHandle, jdouble band, jint capacity, jintArray jEntryStart,
        jintArray jEntryBranch, jintArray jEntryKind, jdoubleArray jEntryRow, jboolean realistic, jdouble minRealisticV,
        jdouble maxRealisticV) {
    auto* t = reinterpret_cast<ViolationTable*>(handle);
    olfgpu::HandedBackStates states = olfgpu::handedBackStates(batchHandle);
    const int nScenarios = static_cast<int>(states.slots.size());
    if (t == nullptr || states.dX == nullptr || nScenarios == 0
            || !grow(t->slots, t->slotsCap, (size_t) nScenarios * sizeof(int))
            || cudaMemcpy(t->slots.p, states.slots.data(), (size_t) nScenarios * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess
            || !prepareChunk(env, t, nScenarios, capacity, jEntryStart, jEntryBranch, jEntryKind, jEntryRow)
            || !launchScreen(t, nScenarios, states.n, band, capacity, states.dX, t->slots.as<int>(), realistic == JNI_TRUE,
                             minRealisticV, maxRealisticV)) {
        return nullptr;
    }
    return readRecords(env, t, nScenarios, capacity);
}

/**
 * Values of the handed-back states (see screenHandedBack): request k is row gatherRow[k] of the handed-back scenario
 * gatherScenario[k] (its index in the result rows). Null on failure.
 */
JNIEXPORT jdoubleArray JNICALL
OLF_JNI(sa_gpu, ViolationScreenNative, gatherHandedBack)(
        JNIEnv* env, jclass, jlong handle, jlong batchHandle, jintArray jGatherScenario, jintArray jGatherRow) {
    auto* t = reinterpret_cast<ViolationTable*>(handle);
    olfgpu::HandedBackStates states = olfgpu::handedBackStates(batchHandle);
    const int nRows = env->GetArrayLength(jGatherRow);
    if (t == nullptr || states.dX == nullptr || env->GetArrayLength(jGatherScenario) != nRows) {
        return nullptr;
    }
    jdoubleArray out = env->NewDoubleArray(nRows);
    if (out == nullptr || nRows == 0) {
        return out;
    }
    const int nScenarios = static_cast<int>(states.slots.size());
    if (!grow(t->slots, t->slotsCap, (size_t) nScenarios * sizeof(int))
            || !grow(t->gatherScenario, t->gatherScenarioCap, (size_t) nRows * sizeof(int))
            || !grow(t->gatherRow, t->gatherRowCap, (size_t) nRows * sizeof(int))
            || !grow(t->gathered, t->gatheredCap, (size_t) nRows * sizeof(double))
            || cudaMemcpy(t->slots.p, states.slots.data(), (size_t) nScenarios * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        return nullptr;
    }
    jint* scenario = env->GetIntArrayElements(jGatherScenario, nullptr);
    jint* row = env->GetIntArrayElements(jGatherRow, nullptr);
    bool ok = cudaMemcpy(t->gatherScenario.p, scenario, (size_t) nRows * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess
            && cudaMemcpy(t->gatherRow.p, row, (size_t) nRows * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess;
    env->ReleaseIntArrayElements(jGatherScenario, scenario, JNI_ABORT);
    env->ReleaseIntArrayElements(jGatherRow, row, JNI_ABORT);
    if (!ok) {
        return nullptr;
    }
    gatherRows<<<grid(nRows), BLOCK>>>(nRows, states.n, t->gatherScenario.as<int>(), t->gatherRow.as<int>(), states.dX,
                                       t->slots.as<int>(), t->gathered.as<double>());
    jdouble* host = env->GetDoubleArrayElements(out, nullptr);
    cudaError_t err = cudaMemcpy(host, t->gathered.p, (size_t) nRows * sizeof(double), cudaMemcpyDeviceToHost);
    env->ReleaseDoubleArrayElements(out, host, 0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "olfgpu violations: gather: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return out;
}

/**
 * The full states of the handed-back scenarios {@code which} (indices in the result rows), one after the other, n doubles
 * each: gathered on the device, then one copy. Null on failure.
 */
JNIEXPORT jdoubleArray JNICALL
OLF_JNI(sa_gpu, ViolationScreenNative, readHandedBackStates)(
        JNIEnv* env, jclass, jlong handle, jlong batchHandle, jintArray jWhich) {
    auto* t = reinterpret_cast<ViolationTable*>(handle);
    olfgpu::HandedBackStates states = olfgpu::handedBackStates(batchHandle);
    const int count = env->GetArrayLength(jWhich);
    const int nScenarios = static_cast<int>(states.slots.size());
    if (t == nullptr || states.dX == nullptr) {
        return nullptr;
    }
    const size_t total = (size_t) count * states.n;
    jdoubleArray out = env->NewDoubleArray((jsize) total);
    if (out == nullptr || count == 0) {
        return out;
    }
    jint* which = env->GetIntArrayElements(jWhich, nullptr);
    bool valid = true;
    for (int j = 0; j < count; j++) {
        valid = valid && which[j] >= 0 && which[j] < nScenarios;
    }
    bool ok = valid
            && grow(t->slots, t->slotsCap, (size_t) nScenarios * sizeof(int))
            && grow(t->gatherScenario, t->gatherScenarioCap, (size_t) count * sizeof(int))
            && grow(t->states, t->statesCap, total * sizeof(double))
            && cudaMemcpy(t->slots.p, states.slots.data(), (size_t) nScenarios * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess
            && cudaMemcpy(t->gatherScenario.p, which, (size_t) count * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess;
    env->ReleaseIntArrayElements(jWhich, which, JNI_ABORT);
    if (!ok) {
        return nullptr;
    }
    gatherStates<<<(unsigned int) ((total + BLOCK - 1) / BLOCK), BLOCK>>>(count, states.n, t->gatherScenario.as<int>(),
                                                                         t->slots.as<int>(), states.dX, t->states.as<double>());
    jdouble* host = env->GetDoubleArrayElements(out, nullptr);
    cudaError_t err = cudaMemcpy(host, t->states.p, total * sizeof(double), cudaMemcpyDeviceToHost);
    env->ReleaseDoubleArrayElements(out, host, 0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "olfgpu violations: state readback: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return out;
}

}  // extern "C"
