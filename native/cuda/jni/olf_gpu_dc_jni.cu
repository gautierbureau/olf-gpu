// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// DC on the device: one cuDSS factorization of OLF's DC matrix reused against many right-hand
// sides, with the per-column work done here so no n x N matrix crosses PCIe.
//
// Consumers (docs/dc-port-plan.md): the Woodbury DC security analysis (this file's first user),
// the time-series DC load flow and the DC sensitivity analysis. What the host hands over is OLF's
// own SparseMatrix arrays: the CSC of the transposed Jacobian OLF stores IS the CSR of the true
// matrix (rows = equations, columns = variables), so cuDSS solves J x = b directly — no transpose.
// Row indices everywhere are the equation system's own Variable.getRow()/Equation.getColumn().
//
// Two lessons the AC bridge and the core cuDSS binding already paid for, applied here:
//  - OLF numbers equations and variables independently, so the matrix has an (almost) empty
//    structural diagonal; cuDSS needs MATCHING or it returns NaN (applyAdjointMatching's comment
//    in olf_gpu_nr_jni.cu, and libmathcudss commit c36d2c4).
//  - a dense solve with zero columns SIGFPEs inside libcudss; guard it.
//
// The Woodbury chunk = one launch sequence per chunk of contingencies:
//   alphas   : per contingency, the k x k flow-transfer system (k <= MAX_K), solved in registers
//   states   : x_post = x0 + S alpha, one thread per (row, contingency) — downloaded to the host,
//              which confirms candidates on OLF's own evaluables and builds the results from it
//   flows    : p1 = -power (ph2 - ph1 + A2 - a1), the verbatim DC branch formula, all branches
//   screen   : the flat limit table (GpuLimitTable) compared with a RELAXED threshold, every
//              exceeded limit of a group emitted — the kernel SCREENS, the host DECIDES
//              (GpuLimitScanner.detectFromCandidates), exactly the AC contract.
#include <cuda_runtime.h>
#include <cudss.h>
#include <jni.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static_assert(CUDSS_VERSION >= 701, "olf_gpu_dc requires cuDSS >= 0.7.1");
#if CUDSS_VERSION >= 800
#  define DC_CUDSS_CSR_TYPES CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F
#  define DC_CUDSS_DN_TYPE   CUDSS_R_64F
#else
#  define DC_CUDSS_CSR_TYPES CUDA_R_32I, CUDA_R_64F
#  define DC_CUDSS_DN_TYPE   CUDA_R_64F
#endif

#define DC_CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) \
        + " at " __FILE__ ":" + std::to_string(__LINE__)); } } while (0)
#define DC_CUDSS_CHECK(x) do { cudssStatus_t s_ = (x); if (s_ != CUDSS_STATUS_SUCCESS) { \
    throw std::runtime_error(std::string("cuDSS error, status ") + std::to_string((int) s_) \
        + " at " __FILE__ ":" + std::to_string(__LINE__) + " [" #x "]"); } } while (0)

constexpr int BLOCK = 256;
constexpr int MAX_K = 8;                     // elements per contingency solved in-kernel
constexpr int QTY_I = 0;                     // GpuLimitTable.QTY_I
constexpr int QTY_P = 1;                     // GpuLimitTable.QTY_P
constexpr int QTY_S = 2;                     // apparent power: NaN in DC, never a candidate
constexpr double A2 = 0.0;                   // PiModel.A2 (a constant in OLF's DC branch term)
constexpr int S_SOLVE_COLS = 1024;           // columns per cuDSS solve when building S (scratch = 2 x n x cols doubles)

inline int grid(long nThreads) {
    return static_cast<int>((nThreads + BLOCK - 1) / BLOCK);
}

struct Buf {
    void* p = nullptr;
    size_t bytes = 0;
    void alloc(size_t n) {
        release();
        bytes = n;
        if (n > 0) {
            DC_CUDA_CHECK(cudaMalloc(&p, n));
        }
    }
    void ensure(size_t n) {
        if (n > bytes) {
            alloc(n);
        }
    }
    void upload(const void* host, size_t n) {
        alloc(n);
        if (n > 0) {
            DC_CUDA_CHECK(cudaMemcpy(p, host, n, cudaMemcpyHostToDevice));
        }
    }
    void release() {
        if (p != nullptr) {
            cudaFree(p);
            p = nullptr;
        }
        bytes = 0;
    }
    template <typename T> T* as() const { return static_cast<T*>(p); }
    ~Buf() { release(); }
};

struct DcContext {
    int n = 0;
    int nnz = 0;
    cudaStream_t stream = nullptr;
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matA = nullptr;
    cudssMatrix_t matB = nullptr;
    cudssMatrix_t matX = nullptr;
    Buf rowPtr, colIdx, val;                 // CSR of the true matrix
    Buf b, x;                                // dense RHS / solution scratch, n x rhsCols
    int rhsCols = 0;
    // element states S (n x nElem, column-major) and the elements' rows
    int nElem = 0;
    Buf s, elemRowPlus, elemRowMinus, elemBranch;
    // branches
    int nBranches = 0;
    Buf ph1Row, ph2Row, a1Row, a1Const, power;
    // base state
    Buf x0;
    // limit table (DC quantities only: I and P from |p|)
    int nGroups = 0;
    int nLimits = 0;
    Buf branchGroupStart, branchGroupEnd, groupQty, groupLimStart, groupLimEnd, limThreshold;
    double dcPowerFactor = 1.0;
    // chunk scratch
    Buf ctgPtr, ctgElem, ctgBase, alpha, states, flows, records, counts;
    // per-chunk modified-target solutions (n x nBases, column-major): the base state of the
    // contingencies that need one (islands, lost power, lost PST) — setChunkTargets
    Buf bases;
    int nBases = 0;

    ~DcContext() {
        if (matB) { cudssMatrixDestroy(matB); }
        if (matX) { cudssMatrixDestroy(matX); }
        if (matA) { cudssMatrixDestroy(matA); }
        if (data && handle) { cudssDataDestroy(handle, data); }
        if (config) { cudssConfigDestroy(config); }
        if (handle) { cudssDestroy(handle); }
        if (stream) { cudaStreamDestroy(stream); }
    }

    /** Dense RHS/solution matrices of the given column count (recreated when the count changes). */
    void ensureRhs(int cols) {
        if (cols == rhsCols && matB != nullptr) {
            return;
        }
        if (matB) { cudssMatrixDestroy(matB); matB = nullptr; }
        if (matX) { cudssMatrixDestroy(matX); matX = nullptr; }
        b.ensure(static_cast<size_t>(n) * cols * sizeof(double));
        x.ensure(static_cast<size_t>(n) * cols * sizeof(double));
        DC_CUDSS_CHECK(cudssMatrixCreateDn(&matB, n, cols, n, b.p, DC_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        DC_CUDSS_CHECK(cudssMatrixCreateDn(&matX, n, cols, n, x.p, DC_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        rhsCols = cols;
    }

    /** Solve J X = B for the current dense RHS (cols columns already in b). */
    void solveDense(int cols) {
        if (cols == 0) {
            return;                                          // cuDSS SIGFPEs on an n x 0 solve
        }
        ensureRhs(cols);
        DC_CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, matA, matX, matB));
        DC_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

// ---------------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------------

/** +1 / -1 at the two bus P rows of every element of the block (a row of -1 means "skip": the
 *  slack side, whose P equation is inactive — ComputedElement.fillRhs). */
__global__ void fillElementRhs(int n, int col0, int cols, const int* rowPlus, const int* rowMinus, double* b) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) {
        return;
    }
    int e = col0 + j;
    if (rowPlus[e] >= 0) {
        b[rowPlus[e] + (size_t) j * n] = 1.0;
    }
    if (rowMinus[e] >= 0) {
        b[rowMinus[e] + (size_t) j * n] = -1.0;
    }
}

/** dph1 - dph2 + da1 of a branch in column `col` of S — ClosedBranchDcFlowEquationTerm.calculateSensi
 *  without the `power` factor (eval(dph1,dph2,da1) - eval(0,0,0) = power * (dph1 - dph2 + da1)). */
__device__ inline double phaseSensi(const double* s, int n, int col, int ph1, int ph2, int a1) {
    const double* c = s + (size_t) col * n;
    double v = c[ph1] - c[ph2];
    if (a1 >= 0) {
        v += c[a1];
    }
    return v;
}

/** One thread per contingency: WoodburyEngine.setAlphas for k <= MAX_K elements — the k x k
 *  system A alpha = b with A_ij = delta_ij / power_i - (S[ph1_i, e_j] - S[ph2_i, e_j]) and
 *  b_i = x0[ph1_i] - x0[ph2_i] (no tap action: newAlpha = 0), Gaussian elimination with partial
 *  pivoting (OLF: DenseMatrix.decomposeLU). k == 1 is the plain division OLF also takes. */
__global__ void woodburyAlphas(int nCtg, int n, const int* ctgPtr, const int* ctgElem, const int* ctgBase,
                               const int* elemBranch, const int* ph1Row, const int* ph2Row,
                               const double* power, const double* s, const double* x0, const double* bases,
                               double* alpha, int* bad) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCtg) {
        return;
    }
    int start = ctgPtr[c];
    int k = ctgPtr[c + 1] - start;
    double* out = alpha + (size_t) c * MAX_K;
    if (k == 0) {
        return;                                              // every element reconnects: the base column IS the state
    }
    if (k < 0 || k > MAX_K) {
        bad[c] = 1;
        return;
    }
    const double* x = ctgBase[c] >= 0 ? bases + (size_t) ctgBase[c] * n : x0;
    double a[MAX_K][MAX_K];
    double rhs[MAX_K];
    for (int i = 0; i < k; i++) {
        int bi = elemBranch[ctgElem[start + i]];
        int p1 = ph1Row[bi];
        int p2 = ph2Row[bi];
        rhs[i] = x[p1] - x[p2];                              // getAlphaRhsValue: ph1 - ph2 (+ 0, no tap action)
        for (int j = 0; j < k; j++) {
            int ej = ctgElem[start + j];
            double deltaX = i == j ? 1.0 / power[bi] : 0.0;
            a[i][j] = deltaX - phaseSensi(s, n, ej, p1, p2, -1);
        }
    }
    if (k == 1) {
        out[0] = rhs[0] / a[0][0];
        return;
    }
    // Gaussian elimination, partial pivoting
    for (int col = 0; col < k; col++) {
        int piv = col;
        double best = fabs(a[col][col]);
        for (int r = col + 1; r < k; r++) {
            double v = fabs(a[r][col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (piv != col) {
            for (int j = 0; j < k; j++) {
                double t = a[col][j];
                a[col][j] = a[piv][j];
                a[piv][j] = t;
            }
            double t = rhs[col];
            rhs[col] = rhs[piv];
            rhs[piv] = t;
        }
        for (int r = col + 1; r < k; r++) {
            double f = a[r][col] / a[col][col];
            for (int j = col; j < k; j++) {
                a[r][j] -= f * a[col][j];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    for (int r = k - 1; r >= 0; r--) {
        double v = rhs[r];
        for (int j = r + 1; j < k; j++) {
            v -= a[r][j] * out[j];
        }
        out[r] = v / a[r][r];
    }
}

/** x_post[row, c] = x0[row] + sum_i alpha_i S[row, e_i] — WoodburyEngine.toPostContingencyStates. */
__global__ void woodburyStates(int nCtg, int n, const int* ctgPtr, const int* ctgElem, const int* ctgBase, const double* alpha,
                               const double* s, const double* x0, const double* bases, double* states) {
    long t = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long) n * nCtg) {
        return;
    }
    int c = (int) (t / n);
    int row = (int) (t % n);
    int start = ctgPtr[c];
    int k = ctgPtr[c + 1] - start;
    double v = ctgBase[c] >= 0 ? bases[row + (size_t) ctgBase[c] * n] : x0[row];
    const double* al = alpha + (size_t) c * MAX_K;
    for (int i = 0; i < k; i++) {
        v += al[i] * s[row + (size_t) ctgElem[start + i] * n];
    }
    states[row + (size_t) c * n] = v;
}

/** p1 of every branch in every state: ClosedBranchSide1DcFlowEquationTerm.eval, verbatim
 *  (DcNetworkVector.updateClosedBranches: 0 when a side has no angle variable). */
__global__ void dcFlows(int nCtg, int n, int nBranches, const int* ph1Row, const int* ph2Row, const int* a1Row,
                        const double* a1Const, const double* power, const double* states, double* flows) {
    long t = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long) nBranches * nCtg) {
        return;
    }
    int c = (int) (t / nBranches);
    int b = (int) (t % nBranches);
    const double* x = states + (size_t) c * n;
    double p = 0.0;
    if (ph1Row[b] >= 0 && ph2Row[b] >= 0) {
        double a1 = a1Row[b] >= 0 ? x[a1Row[b]] : a1Const[b];
        double deltaPhase = x[ph2Row[b]] - x[ph1Row[b]] + A2 - a1;
        p = -power[b] * deltaPhase;
    }
    flows[b + (size_t) c * nBranches] = p;
}

/** ConnectivityBreakAnalysis.isGroupOfElementsBreakingConnectivity, one thread per contingency:
 *  for every element e of the contingency, the sum over the contingency's elements e2 of
 *  |p1_{branch(e2)}.calculateSensi(S, column(e))| = |power_{b2} (dph1 - dph2 + da1)| read in column e;
 *  the maximum of those sums is returned (the host compares it with 1 - 1e-6). */
__global__ void connectivityCriterion(int nCtg, int n, const int* ctgPtr, const int* ctgElem, const int* elemBranch,
                                      const int* ph1Row, const int* ph2Row, const int* a1Row, const double* power,
                                      const double* s, double* maxSum) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCtg) {
        return;
    }
    double best = 0.0;
    for (int e = ctgPtr[c]; e < ctgPtr[c + 1]; e++) {
        int col = ctgElem[e];
        double sum = 0.0;
        for (int e2 = ctgPtr[c]; e2 < ctgPtr[c + 1]; e2++) {
            int b2 = elemBranch[ctgElem[e2]];
            sum += fabs(power[b2] * phaseSensi(s, n, col, ph1Row[b2], ph2Row[b2], a1Row[b2]));
        }
        best = fmax(best, sum);
    }
    maxSum[c] = best;
}

__device__ inline void emitRecord(int* records, int* counts, int c, int capacity, int record) {
    int slot = atomicAdd(counts + c, 1);
    if (slot < capacity) {
        records[(size_t) c * capacity + slot] = record;
    }
}

/** Screen every (branch, contingency) against the flat limit table. DC quantities: active power
 *  |p| (side 2 is -p1, same magnitude) and current |p| / dcPowerFactor (ClosedBranchDcCurrent);
 *  apparent power is NaN in DC and never fires. A branch the contingency outages is skipped, like
 *  the stock predicate. Relaxed by `band`; all exceeded limits of a group are emitted. */
__global__ void dcScreen(int nCtg, int nBranches, const int* ctgPtr, const int* ctgElem, const int* elemBranch,
                         const int* branchGroupStart, const int* branchGroupEnd, const int* groupQty,
                         const int* groupLimStart, const int* groupLimEnd, const double* limThreshold,
                         double dcPowerFactor, double band, int capacity,
                         const double* flows, int* records, int* counts) {
    long t = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long) nBranches * nCtg) {
        return;
    }
    int c = (int) (t / nBranches);
    int b = (int) (t % nBranches);
    int g0 = branchGroupStart[b];
    int g1 = branchGroupEnd[b];
    if (g0 == g1) {
        return;
    }
    for (int e = ctgPtr[c]; e < ctgPtr[c + 1]; e++) {
        if (elemBranch[ctgElem[e]] == b) {
            return;                                          // outaged by this contingency
        }
    }
    double absP = fabs(flows[b + (size_t) c * nBranches]);
    for (int g = g0; g < g1; g++) {
        int qty = groupQty[g];
        double value;
        if (qty == QTY_P) {
            value = absP;
        } else if (qty == QTY_I) {
            value = absP / dcPowerFactor;
        } else {
            continue;                                        // QTY_S: NaN in DC
        }
        for (int l = groupLimStart[g]; l < groupLimEnd[g]; l++) {
            if (value > limThreshold[l] * (1.0 - band)) {
                emitRecord(records, counts, c, capacity, l);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// JNI helpers
// ---------------------------------------------------------------------------------------------

DcContext* ctxOf(jlong h) {
    return reinterpret_cast<DcContext*>(h);
}

struct IntArray {
    JNIEnv* env;
    jintArray arr;
    jint* p;
    jsize len;
    IntArray(JNIEnv* e, jintArray a) : env(e), arr(a), p(a ? e->GetIntArrayElements(a, nullptr) : nullptr), len(a ? e->GetArrayLength(a) : 0) {}
    ~IntArray() { if (p) { env->ReleaseIntArrayElements(arr, p, JNI_ABORT); } }
};

struct DoubleArray {
    JNIEnv* env;
    jdoubleArray arr;
    jdouble* p;
    jsize len;
    DoubleArray(JNIEnv* e, jdoubleArray a) : env(e), arr(a), p(a ? e->GetDoubleArrayElements(a, nullptr) : nullptr), len(a ? e->GetArrayLength(a) : 0) {}
    ~DoubleArray() { if (p) { env->ReleaseDoubleArrayElements(arr, p, JNI_ABORT); } }
};

void throwJava(JNIEnv* env, const std::string& msg) {
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls != nullptr) {
        env->ThrowNew(cls, msg.c_str());
    }
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_createContext(JNIEnv* env, jclass, jint n, jintArray jRowPtr,
                                                               jintArray jColIdx, jdoubleArray jVal) {
    DcContext* ctx = new DcContext();
    try {
        IntArray rowPtr(env, jRowPtr);
        IntArray colIdx(env, jColIdx);
        DoubleArray val(env, jVal);
        if (rowPtr.len != n + 1) {
            throw std::runtime_error("rowPtr length must be n + 1");
        }
        ctx->n = n;
        ctx->nnz = rowPtr.p[n];
        if (val.len != ctx->nnz || colIdx.len < ctx->nnz) {
            throw std::runtime_error("CSR arrays inconsistent with rowPtr[n]");
        }
        DC_CUDA_CHECK(cudaStreamCreate(&ctx->stream));
        DC_CUDSS_CHECK(cudssCreate(&ctx->handle));
        DC_CUDSS_CHECK(cudssSetStream(ctx->handle, ctx->stream));
        DC_CUDSS_CHECK(cudssConfigCreate(&ctx->config));
        // Iterative refinement: OFF by default. KLU (the reference) refines nothing, the LU is backward
        // stable, and each refinement step costs a solve + a SpMV on every column — measured 2.1 s for
        // the 9042-column S with 2 steps. OLF_GPU_DC_IR=<n> overrides (A/B against the parity gate).
        int ir = 0;
        if (const char* e = std::getenv("OLF_GPU_DC_IR"); e != nullptr && *e != '\0') {
            ir = std::atoi(e);
        }
        DC_CUDSS_CHECK(cudssConfigSet(ctx->config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)));
#if CUDSS_VERSION >= 800
        cudssMatchingAlg_t matching = CUDSS_MATCHING_ALG_AUTO;   // empty structural diagonal, see header
        DC_CUDSS_CHECK(cudssConfigSet(ctx->config, CUDSS_CONFIG_MATCHING_ALG, &matching, sizeof(matching)));
        cudssReorderingAlg_t amd = CUDSS_REORDERING_ALG_AMD;       // the AC sweep winner
        DC_CUDSS_CHECK(cudssConfigSet(ctx->config, CUDSS_CONFIG_REORDERING_ALG, &amd, sizeof(amd)));
#endif
        DC_CUDSS_CHECK(cudssDataCreate(ctx->handle, &ctx->data));
        ctx->rowPtr.upload(rowPtr.p, (size_t) (n + 1) * sizeof(int));
        ctx->colIdx.upload(colIdx.p, (size_t) ctx->nnz * sizeof(int));
        ctx->val.upload(val.p, (size_t) ctx->nnz * sizeof(double));
        DC_CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->matA, n, n, ctx->nnz, ctx->rowPtr.p, nullptr, ctx->colIdx.p, ctx->val.p,
                                            DC_CUDSS_CSR_TYPES, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        ctx->ensureRhs(1);
        DC_CUDSS_CHECK(cudssExecute(ctx->handle, CUDSS_PHASE_ANALYSIS, ctx->config, ctx->data, ctx->matA, ctx->matX, ctx->matB));
        DC_CUDSS_CHECK(cudssExecute(ctx->handle, CUDSS_PHASE_FACTORIZATION, ctx->config, ctx->data, ctx->matA, ctx->matX, ctx->matB));
        DC_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        return reinterpret_cast<jlong>(ctx);
    } catch (const std::exception& e) {
        delete ctx;
        throwJava(env, std::string("GpuDcSolver.createContext: ") + e.what());
        return 0;
    }
}

JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_destroyContext(JNIEnv*, jclass, jlong handle) {
    delete ctxOf(handle);
}

/** Solve J x = rhs for one right-hand side; returns x. */
JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_solve(JNIEnv* env, jclass, jlong handle, jdoubleArray jRhs) {
    DcContext* ctx = ctxOf(handle);
    try {
        DoubleArray rhs(env, jRhs);
        if (rhs.len != ctx->n) {
            throw std::runtime_error("rhs length must be n");
        }
        ctx->ensureRhs(1);
        DC_CUDA_CHECK(cudaMemcpy(ctx->b.p, rhs.p, (size_t) ctx->n * sizeof(double), cudaMemcpyHostToDevice));
        ctx->solveDense(1);
        jdoubleArray out = env->NewDoubleArray(ctx->n);
        std::vector<double> host(ctx->n);
        DC_CUDA_CHECK(cudaMemcpy(host.data(), ctx->x.p, (size_t) ctx->n * sizeof(double), cudaMemcpyDeviceToHost));
        env->SetDoubleArrayRegion(out, 0, ctx->n, host.data());
        return out;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.solve: ") + e.what());
        return nullptr;
    }
}

/** The contingency elements: +1 at rowPlus, -1 at rowMinus (-1 = none), one S column each,
 *  solved in blocks against the factorization and kept on the device. Returns false when the
 *  device cannot hold S (the caller then routes the run to the CPU). */
JNIEXPORT jboolean JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setElements(JNIEnv* env, jclass, jlong handle, jintArray jRowPlus,
                                                             jintArray jRowMinus, jintArray jElemBranch) {
    DcContext* ctx = ctxOf(handle);
    try {
        IntArray rowPlus(env, jRowPlus);
        IntArray rowMinus(env, jRowMinus);
        IntArray elemBranch(env, jElemBranch);
        int k = rowPlus.len;
        if (rowMinus.len != k || elemBranch.len != k) {
            throw std::runtime_error("element arrays must have the same length");
        }
        ctx->nElem = k;
        ctx->elemRowPlus.upload(rowPlus.p, (size_t) k * sizeof(int));
        ctx->elemRowMinus.upload(rowMinus.p, (size_t) k * sizeof(int));
        ctx->elemBranch.upload(elemBranch.p, (size_t) k * sizeof(int));
        size_t sBytes = (size_t) ctx->n * k * sizeof(double);
        ctx->s.release();
        if (k > 0 && cudaMalloc(&ctx->s.p, sBytes) != cudaSuccess) {
            cudaGetLastError();
            ctx->s.p = nullptr;
            return JNI_FALSE;
        }
        ctx->s.bytes = sBytes;
        for (int col0 = 0; col0 < k; col0 += S_SOLVE_COLS) {
            int cols = std::min(S_SOLVE_COLS, k - col0);
            ctx->ensureRhs(cols);
            DC_CUDA_CHECK(cudaMemsetAsync(ctx->b.p, 0, (size_t) ctx->n * cols * sizeof(double), ctx->stream));
            fillElementRhs<<<grid(cols), BLOCK, 0, ctx->stream>>>(ctx->n, col0, cols, ctx->elemRowPlus.as<int>(),
                                                                 ctx->elemRowMinus.as<int>(), ctx->b.as<double>());
            DC_CUDA_CHECK(cudaGetLastError());
            ctx->solveDense(cols);
            DC_CUDA_CHECK(cudaMemcpy(ctx->s.as<double>() + (size_t) col0 * ctx->n, ctx->x.p,
                                     (size_t) ctx->n * cols * sizeof(double), cudaMemcpyDeviceToDevice));
        }
        return JNI_TRUE;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setElements: ") + e.what());
        return JNI_FALSE;
    }
}

/** One S column back to the host (tests: the LODF/PTDF oracle). */
JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_readElementStates(JNIEnv* env, jclass, jlong handle, jint elem) {
    DcContext* ctx = ctxOf(handle);
    try {
        if (elem < 0 || elem >= ctx->nElem) {
            throw std::runtime_error("element index out of range");
        }
        std::vector<double> host(ctx->n);
        DC_CUDA_CHECK(cudaMemcpy(host.data(), ctx->s.as<double>() + (size_t) elem * ctx->n,
                                 (size_t) ctx->n * sizeof(double), cudaMemcpyDeviceToHost));
        jdoubleArray out = env->NewDoubleArray(ctx->n);
        env->SetDoubleArrayRegion(out, 0, ctx->n, host.data());
        return out;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.readElementStates: ") + e.what());
        return nullptr;
    }
}

JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setBranches(JNIEnv* env, jclass, jlong handle, jintArray jPh1, jintArray jPh2,
                                                             jintArray jA1Row, jdoubleArray jA1Const, jdoubleArray jPower) {
    DcContext* ctx = ctxOf(handle);
    try {
        IntArray ph1(env, jPh1);
        IntArray ph2(env, jPh2);
        IntArray a1Row(env, jA1Row);
        DoubleArray a1Const(env, jA1Const);
        DoubleArray power(env, jPower);
        int nb = ph1.len;
        if (ph2.len != nb || a1Row.len != nb || a1Const.len != nb || power.len != nb) {
            throw std::runtime_error("branch arrays must have the same length");
        }
        ctx->nBranches = nb;
        ctx->ph1Row.upload(ph1.p, (size_t) nb * sizeof(int));
        ctx->ph2Row.upload(ph2.p, (size_t) nb * sizeof(int));
        ctx->a1Row.upload(a1Row.p, (size_t) nb * sizeof(int));
        ctx->a1Const.upload(a1Const.p, (size_t) nb * sizeof(double));
        ctx->power.upload(power.p, (size_t) nb * sizeof(double));
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setBranches: ") + e.what());
    }
}

JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setBase(JNIEnv* env, jclass, jlong handle, jdoubleArray jX0) {
    DcContext* ctx = ctxOf(handle);
    try {
        DoubleArray x0(env, jX0);
        if (x0.len != ctx->n) {
            throw std::runtime_error("x0 length must be n");
        }
        ctx->x0.upload(x0.p, (size_t) ctx->n * sizeof(double));
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setBase: ") + e.what());
    }
}

JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setLimitTable(JNIEnv* env, jclass, jlong handle,
                                                               jintArray jBranchGroupStart, jintArray jBranchGroupEnd,
                                                               jintArray jGroupQty, jintArray jGroupLimStart, jintArray jGroupLimEnd,
                                                               jdoubleArray jLimThreshold, jdouble dcPowerFactor) {
    DcContext* ctx = ctxOf(handle);
    try {
        IntArray bgs(env, jBranchGroupStart);
        IntArray bge(env, jBranchGroupEnd);
        IntArray gq(env, jGroupQty);
        IntArray gls(env, jGroupLimStart);
        IntArray gle(env, jGroupLimEnd);
        DoubleArray thr(env, jLimThreshold);
        if (bgs.len != ctx->nBranches || bge.len != ctx->nBranches) {
            throw std::runtime_error("limit table branches must match setBranches");
        }
        ctx->nGroups = gq.len;
        ctx->nLimits = thr.len;
        ctx->branchGroupStart.upload(bgs.p, (size_t) bgs.len * sizeof(int));
        ctx->branchGroupEnd.upload(bge.p, (size_t) bge.len * sizeof(int));
        ctx->groupQty.upload(gq.p, (size_t) gq.len * sizeof(int));
        ctx->groupLimStart.upload(gls.p, (size_t) gls.len * sizeof(int));
        ctx->groupLimEnd.upload(gle.p, (size_t) gle.len * sizeof(int));
        ctx->limThreshold.upload(thr.p, (size_t) thr.len * sizeof(double));
        ctx->dcPowerFactor = dcPowerFactor;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setLimitTable: ") + e.what());
    }
}

/** The sensitivity criterion of ConnectivityBreakAnalysis for every contingency (CSR over
 *  element indices): the max over its elements of the summed |sensitivities|; > 1 - 1e-6 means
 *  "potentially breaking" and goes to the host's graph confirmation. */
JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_connectivityCriterion(JNIEnv* env, jclass, jlong handle, jintArray jCtgPtr, jintArray jCtgElem) {
    DcContext* ctx = ctxOf(handle);
    try {
        IntArray ctgPtr(env, jCtgPtr);
        IntArray ctgElem(env, jCtgElem);
        int nCtg = ctgPtr.len - 1;
        if (nCtg <= 0) {
            return env->NewDoubleArray(0);
        }
        if (ctx->s.p == nullptr || ctx->nBranches == 0) {
            throw std::runtime_error("context not set up (elements, branches)");
        }
        ctx->ctgPtr.upload(ctgPtr.p, (size_t) ctgPtr.len * sizeof(int));
        ctx->ctgElem.upload(ctgElem.p, (size_t) std::max<jsize>(ctgElem.len, 1) * sizeof(int));
        Buf out;
        out.alloc((size_t) nCtg * sizeof(double));
        connectivityCriterion<<<grid(nCtg), BLOCK, 0, ctx->stream>>>(nCtg, ctx->n, ctx->ctgPtr.as<int>(), ctx->ctgElem.as<int>(),
                ctx->elemBranch.as<int>(), ctx->ph1Row.as<int>(), ctx->ph2Row.as<int>(), ctx->a1Row.as<int>(), ctx->power.as<double>(),
                ctx->s.as<double>(), out.as<double>());
        DC_CUDA_CHECK(cudaGetLastError());
        DC_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<double> host(nCtg);
        DC_CUDA_CHECK(cudaMemcpy(host.data(), out.p, (size_t) nCtg * sizeof(double), cudaMemcpyDeviceToHost));
        jdoubleArray result = env->NewDoubleArray(nCtg);
        env->SetDoubleArrayRegion(result, 0, nCtg, host.data());
        return result;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.connectivityCriterion: ") + e.what());
        return nullptr;
    }
}

/** The modified target columns of a chunk (n x nBases, column-major), solved against the
 *  factorization and kept on the device as the base states of the contingencies that need one
 *  (WoodburyEngine.runDcLoadFlowWithModifiedTargetVector, batched). */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setChunkTargets(JNIEnv* env, jclass, jlong handle, jdoubleArray jTargets, jint nBases) {
    DcContext* ctx = ctxOf(handle);
    try {
        ctx->nBases = nBases;
        if (nBases == 0) {
            return;
        }
        DoubleArray targets(env, jTargets);
        if (targets.len < (jsize) ((size_t) ctx->n * nBases)) {
            throw std::runtime_error("targets too small");
        }
        ctx->bases.ensure((size_t) ctx->n * nBases * sizeof(double));
        for (int col0 = 0; col0 < nBases; col0 += S_SOLVE_COLS) {
            int cols = std::min(S_SOLVE_COLS, nBases - col0);
            ctx->ensureRhs(cols);
            DC_CUDA_CHECK(cudaMemcpy(ctx->b.p, targets.p + (size_t) col0 * ctx->n, (size_t) ctx->n * cols * sizeof(double), cudaMemcpyHostToDevice));
            ctx->solveDense(cols);
            DC_CUDA_CHECK(cudaMemcpy(ctx->bases.as<double>() + (size_t) col0 * ctx->n, ctx->x.p,
                                     (size_t) ctx->n * cols * sizeof(double), cudaMemcpyDeviceToDevice));
        }
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setChunkTargets: ") + e.what());
    }
}

/**
 * One chunk of contingencies: alphas, states, flows, screen. Writes the n x nCtg post-contingency
 * states (column-major) into outStates and returns [counts(nCtg) | records(nCtg x capacity)]; a
 * count above capacity means "screen this one on the host". ctgBase[c] is the column of the
 * chunk's modified-target solutions (setChunkTargets) the contingency starts from, or -1 for
 * the pre-contingency state. A contingency with more than MAX_K elements makes the call throw —
 * the host routes those to the CPU beforehand.
 */
JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_woodburyChunk(JNIEnv* env, jclass, jlong handle, jintArray jCtgPtr,
                                                               jintArray jCtgElem, jintArray jCtgBase, jdouble band, jint capacity,
                                                               jdoubleArray jOutStates) {
    DcContext* ctx = ctxOf(handle);
    try {
        IntArray ctgPtr(env, jCtgPtr);
        IntArray ctgElem(env, jCtgElem);
        IntArray ctgBase(env, jCtgBase);
        int nCtg = ctgPtr.len - 1;
        if (ctgBase.len != nCtg) {
            throw std::runtime_error("ctgBase length must be nCtg");
        }
        for (int c = 0; c < nCtg; c++) {
            if (ctgBase.p[c] >= ctx->nBases) {
                throw std::runtime_error("ctgBase refers to a column beyond setChunkTargets");
            }
        }
        if (nCtg <= 0) {
            throw std::runtime_error("empty chunk");
        }
        if (ctx->x0.p == nullptr || ctx->s.p == nullptr || ctx->nBranches == 0) {
            throw std::runtime_error("context not set up (base, elements, branches)");
        }
        int n = ctx->n;
        int nb = ctx->nBranches;
        ctx->ctgPtr.upload(ctgPtr.p, (size_t) ctgPtr.len * sizeof(int));
        ctx->ctgElem.upload(ctgElem.p, (size_t) std::max<jsize>(ctgElem.len, 1) * sizeof(int));
        ctx->ctgBase.upload(ctgBase.p, (size_t) nCtg * sizeof(int));
        ctx->alpha.ensure((size_t) nCtg * MAX_K * sizeof(double));
        ctx->states.ensure((size_t) n * nCtg * sizeof(double));
        ctx->flows.ensure((size_t) nb * nCtg * sizeof(double));
        ctx->records.ensure((size_t) nCtg * capacity * sizeof(int));
        ctx->counts.ensure((size_t) nCtg * sizeof(int));
        Buf bad;
        bad.alloc((size_t) nCtg * sizeof(int));
        DC_CUDA_CHECK(cudaMemsetAsync(bad.p, 0, (size_t) nCtg * sizeof(int), ctx->stream));
        DC_CUDA_CHECK(cudaMemsetAsync(ctx->counts.p, 0, (size_t) nCtg * sizeof(int), ctx->stream));

        const double* bases = ctx->nBases > 0 ? ctx->bases.as<double>() : ctx->x0.as<double>();
        woodburyAlphas<<<grid(nCtg), BLOCK, 0, ctx->stream>>>(nCtg, n, ctx->ctgPtr.as<int>(), ctx->ctgElem.as<int>(), ctx->ctgBase.as<int>(),
                ctx->elemBranch.as<int>(), ctx->ph1Row.as<int>(), ctx->ph2Row.as<int>(),
                ctx->power.as<double>(), ctx->s.as<double>(), ctx->x0.as<double>(), bases, ctx->alpha.as<double>(), bad.as<int>());
        DC_CUDA_CHECK(cudaGetLastError());
        woodburyStates<<<grid((long) n * nCtg), BLOCK, 0, ctx->stream>>>(nCtg, n, ctx->ctgPtr.as<int>(), ctx->ctgElem.as<int>(), ctx->ctgBase.as<int>(),
                ctx->alpha.as<double>(), ctx->s.as<double>(), ctx->x0.as<double>(), bases, ctx->states.as<double>());
        DC_CUDA_CHECK(cudaGetLastError());
        dcFlows<<<grid((long) nb * nCtg), BLOCK, 0, ctx->stream>>>(nCtg, n, nb, ctx->ph1Row.as<int>(), ctx->ph2Row.as<int>(),
                ctx->a1Row.as<int>(), ctx->a1Const.as<double>(), ctx->power.as<double>(), ctx->states.as<double>(), ctx->flows.as<double>());
        DC_CUDA_CHECK(cudaGetLastError());
        if (ctx->nGroups > 0) {
            dcScreen<<<grid((long) nb * nCtg), BLOCK, 0, ctx->stream>>>(nCtg, nb, ctx->ctgPtr.as<int>(), ctx->ctgElem.as<int>(),
                    ctx->elemBranch.as<int>(), ctx->branchGroupStart.as<int>(), ctx->branchGroupEnd.as<int>(), ctx->groupQty.as<int>(),
                    ctx->groupLimStart.as<int>(), ctx->groupLimEnd.as<int>(), ctx->limThreshold.as<double>(),
                    ctx->dcPowerFactor, band, capacity, ctx->flows.as<double>(), ctx->records.as<int>(), ctx->counts.as<int>());
            DC_CUDA_CHECK(cudaGetLastError());
        }
        DC_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        std::vector<int> badHost(nCtg);
        DC_CUDA_CHECK(cudaMemcpy(badHost.data(), bad.p, (size_t) nCtg * sizeof(int), cudaMemcpyDeviceToHost));
        for (int c = 0; c < nCtg; c++) {
            if (badHost[c]) {
                throw std::runtime_error("contingency " + std::to_string(c) + " has more than " + std::to_string(MAX_K) + " elements");
            }
        }
        // states out
        if (env->GetArrayLength(jOutStates) < (jsize) ((size_t) n * nCtg)) {
            throw std::runtime_error("outStates too small");
        }
        std::vector<double> statesHost((size_t) n * nCtg);
        DC_CUDA_CHECK(cudaMemcpy(statesHost.data(), ctx->states.p, statesHost.size() * sizeof(double), cudaMemcpyDeviceToHost));
        env->SetDoubleArrayRegion(jOutStates, 0, (jsize) statesHost.size(), statesHost.data());
        // records out
        std::vector<int> out((size_t) nCtg + (size_t) nCtg * capacity, 0);
        DC_CUDA_CHECK(cudaMemcpy(out.data(), ctx->counts.p, (size_t) nCtg * sizeof(int), cudaMemcpyDeviceToHost));
        if (ctx->nGroups > 0) {
            DC_CUDA_CHECK(cudaMemcpy(out.data() + nCtg, ctx->records.p, (size_t) nCtg * capacity * sizeof(int), cudaMemcpyDeviceToHost));
        }
        jintArray result = env->NewIntArray((jsize) out.size());
        env->SetIntArrayRegion(result, 0, (jsize) out.size(), out.data());
        return result;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.woodburyChunk: ") + e.what());
        return nullptr;
    }
}

}  // extern "C"
