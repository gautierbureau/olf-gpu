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
constexpr int MAX_K = 32;                    // elements per contingency solved in-kernel (32x32 doubles of local memory per thread)
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
    // WoodburyEngine's per-element terms: the diagonal 1/(oldPower - newPower) (1/power for an outage,
    // -1/power for a branch enabled by an action, 1/(p_old - p_new) for a tap change) and the new
    // phase shift a tap action imposes (0 otherwise) — getAlphaMatrixValue / getAlphaRhsValue
    Buf elemDiag, elemNewAlpha;
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

    // --- device assembly (createAssembled, GpuDcAssembly): the structure the device fills itself
    bool assembled = false;
    bool analyzed = false;
    Buf contribPtr, contribKind, contribElem, contribCoef;         // matrix values from the branch powers
    Buf targetKind, targetElem, rhsPtr, rhsKind, rhsElem, rhsCoef; // the target vector formula
    Buf t;                                                         // the device target vector (n)
    int nBuses = 0;
    Buf busPColumn, busPhRow, busLoadP, busFlags, busSc, busGenStart, busGenP, busExcluded;
    int nSc = 0;
    Buf scBusPtr, scBusList, scSlackCount, scMismatch;
    int nGen = 0;
    Buf genInit, genTarget, genMinT, genMaxT, genMinP, genMaxP, genDroop, genPf, genPart;
    Buf dsElem, dsFactor, dsOut;                                   // distribution scratch and its 2 outputs
    Buf scalarOut;
    Buf lfState;                                                   // the last solveTarget solution (n), read by the loop kernels
    // --- DC outer loops on the device (GpuDcLoadFlowEngine)
    // incremental phase control: the controllers, their controlled branch/side/target, the tap a1 tables,
    // and the per-BRANCH direction state of IncrementalContextData.ControllerContext
    int nCtrl = 0;
    Buf pcBranch, pcAlphaCol, pcControlled, pcSide, pcTarget, pcHalfDb, pcTapIdx, pcTapPtr, pcTapA1, pcNewTap;
    Buf pcAllowed, pcDirCount, pcCurDir, pcSens;
    // HVDC AC emulation limits
    int nHvdc = 0;
    Buf hvPh1, hvPh2, hvK, hvP0, hvPmax12, hvPmax21, hvStatus, hvActive, hvNew;

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
                               const double* elemDiag, const double* elemNewAlpha, const double* s, const double* x0, const double* bases,
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
        int ei = ctgElem[start + i];
        int bi = elemBranch[ei];
        int p1 = ph1Row[bi];
        int p2 = ph2Row[bi];
        rhs[i] = x[p1] - x[p2] + elemNewAlpha[ei];           // getAlphaRhsValue: ph1 - ph2 + the tap action's new alpha
        for (int j = 0; j < k; j++) {
            int ej = ctgElem[start + j];
            double deltaX = i == j ? elemDiag[ei] : 0.0;     // getAlphaMatrixValue: 1/(oldPower - newPower) on the diagonal
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


// ---------------------------------------------------------------------------------------------
// device assembly: the DC system built on the device from GpuDcAssembly's structure
// ---------------------------------------------------------------------------------------------

constexpr int CONTRIB_CONST = 0;
constexpr int CONTRIB_BRANCH = 1;
constexpr int TARGET_ZERO = 0;
constexpr int TARGET_BUS_P = 1;
constexpr int TARGET_ALPHA1 = 2;
constexpr int TARGET_ZERO_PHI = 3;
constexpr int RHS_CONST = 0;
constexpr int RHS_BRANCH = 1;
constexpr int BUS_DISABLED = 1;
constexpr int BUS_PARTICIPATING = 2;
constexpr int BUS_FICTITIOUS = 4;
constexpr int BUS_SLACK = 8;
constexpr double P_RESIDUE_EPS = 1e-5;       // ActivePowerDistribution.P_RESIDUE_EPS = 10^-5

/** JacobianMatrix.initDer's element values: the sequential sum of the active terms' derivatives,
 *  each ±power[b] (closed-branch flow terms) or a constant (±1 variable terms, ±k HVDC emulation),
 *  in OLF's callback order — `value = 0; value += term.der(v)`. One thread per non-zero. */
__global__ void fillValues(int nnz, const int* contribPtr, const int* kind, const int* elem, const double* coef,
                           const double* power, double* val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }
    double v = 0.0;
    for (int c = contribPtr[i]; c < contribPtr[i + 1]; c++) {
        // coef is ±1 for a branch term: the product is exact, no contraction concern
        v = __dadd_rn(v, kind[c] == CONTRIB_BRANCH ? __dmul_rn(coef[c], power[elem[c]]) : coef[c]);
    }
    val[i] = v;
}

/** AbstractLfBus.getGenerationTargetP: the bus's generators' targetP summed in their order (from 0.0). */
__global__ void busGeneration(int nBuses, const int* genStart, const double* genTarget, double* genP) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBuses) {
        return;
    }
    double s = 0.0;
    for (int g = genStart[b]; g < genStart[b + 1]; g++) {
        s = __dadd_rn(s, genTarget[g]);
    }
    genP[b] = s;
}

/** DcLoadFlowEngine.getActivePowerMismatch over a synchronous component's buses (multi-slack share). */
__global__ void scMismatches(int nSc, const int* scBusPtr, const int* scBusList, const int* busFlags,
                             const double* genP, const double* loadP, double* out) {
    int sc = blockIdx.x * blockDim.x + threadIdx.x;
    if (sc >= nSc) {
        return;
    }
    double sum = 0.0;
    for (int i = scBusPtr[sc]; i < scBusPtr[sc + 1]; i++) {
        int b = scBusList[i];
        if ((busFlags[b] & BUS_DISABLED) == 0) {
            sum = __dadd_rn(sum, __dadd_rn(genP[b], -loadP[b]));
        }
    }
    out[sc] = -sum;
}

/** DcTargetVector.init, one thread per equation column: the base by kind (bus targetP with the
 *  multi-slack share, the phase shifter's a1, A2 - a1, or 0) minus the sequential sum of the active
 *  terms' rhs (-p (A2 - a1fixed) / +p (A2 - a1fixed) for the branch terms, ±p0 for HVDC). */
__global__ void fillTarget(int n, const int* tKind, const int* tElem, const int* rhsPtr, const int* rhsKind, const int* rhsElem,
                           const double* rhsCoef, const double* genP, const double* loadP, const int* busFlags, const int* busSc,
                           const int* scSlackCount, const double* scMismatch, const double* a1Const, const int* a1Row,
                           const double* power, double* t) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) {
        return;
    }
    double base;
    switch (tKind[c]) {
        case TARGET_BUS_P: {
            int b = tElem[c];
            base = __dadd_rn(genP[b], -loadP[b]);
            if (busFlags[b] & BUS_SLACK) {
                base = __dadd_rn(base, __ddiv_rn(scMismatch[busSc[b]], (double) scSlackCount[busSc[b]]));
            }
            break;
        }
        case TARGET_ALPHA1:
            base = a1Const[tElem[c]];
            break;
        case TARGET_ZERO_PHI:
            base = __dadd_rn(A2, -a1Const[tElem[c]]);
            break;
        default:
            base = 0.0;
    }
    double rhs = 0.0;
    for (int r = rhsPtr[c]; r < rhsPtr[c + 1]; r++) {
        if (rhsKind[r] == RHS_BRANCH) {
            int b = rhsElem[r];
            double a1Fixed = a1Row[b] >= 0 ? 0.0 : a1Const[b];
            // (coef * power) * (A2 - a1) as the Java: -getPower() * (A2 - a1()) / getPower() * (A2 - a1())
            rhs = __dadd_rn(rhs, __dmul_rn(__dmul_rn(rhsCoef[r], power[b]), __dadd_rn(A2, -a1Fixed)));
        } else {
            rhs = __dadd_rn(rhs, rhsCoef[r]);
        }
    }
    t[c] = __dadd_rn(base, -rhs);
}

/** DcLoadFlowEngine.getActivePowerMismatch(buses): -sum over the enabled, non-excluded buses in
 *  bus order of (generation - load). One thread, sequential like the CPU. */
__global__ void activePowerMismatch(int nBuses, const int* busFlags, const int* excluded, const double* genP, const double* loadP, double* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    double mismatch = 0.0;
    for (int b = 0; b < nBuses; b++) {
        if ((busFlags[b] & BUS_DISABLED) == 0 && !(excluded != nullptr && excluded[b])) {
            mismatch = __dadd_rn(mismatch, __dadd_rn(genP[b], -loadP[b]));
        }
    }
    out[0] = -mismatch;
}

/** Java's Math.max / Math.min on doubles (NaN wins, -0.0 < 0.0), not IEEE maxNum. */
__device__ inline double javaMax(double a, double b) {
    if (a != a) { return a; }
    if (b != b) { return b; }
    if (a == 0.0 && b == 0.0) { return signbit(a) ? b : a; }
    return a >= b ? a : b;
}

__device__ inline double javaMin(double a, double b) {
    if (a != a) { return a; }
    if (b != b) { return b; }
    if (a == 0.0 && b == 0.0) { return signbit(a) ? a : b; }
    return a <= b ? a : b;
}

/** DoubleStream.sum() as the JDK (17+) computes it: Kahan compensation, sequential, with the
 *  simple-sum fallback for an infinite total — ParticipatingElement.participationFactorNorm. */
__device__ inline double javaStreamSum(const double* v, int count) {
    double sum = 0.0;
    double comp = 0.0;
    double simple = 0.0;
    for (int k = 0; k < count; k++) {
        double tmp = __dadd_rn(v[k], -comp);
        double velvel = __dadd_rn(sum, tmp);
        comp = __dadd_rn(__dadd_rn(velvel, -sum), -tmp);
        sum = velvel;
        simple = __dadd_rn(simple, v[k]);
    }
    double tmp = __dadd_rn(sum, -comp);
    if (tmp != tmp && isinf(simple)) {
        return simple;
    }
    return tmp;
}

constexpr int PT_MAX = 0;                    // PROPORTIONAL_TO_GENERATION_P_MAX
constexpr int PT_TARGET = 1;                 // PROPORTIONAL_TO_GENERATION_P
constexpr int PT_PARTICIPATION_FACTOR = 2;   // PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR
constexpr int PT_REMAINING_MARGIN = 3;       // PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN

/** GenerationActivePowerDistributionStep.getParticipationFactor. */
__device__ inline double participationFactor(int type, double targetP, double maxP, double minP, double droop, double pf, double mismatch) {
    switch (type) {
        case PT_MAX:
            return __ddiv_rn(maxP, droop);
        case PT_TARGET:
            return fabs(targetP);
        case PT_PARTICIPATION_FACTOR:
            return pf;
        default: {
            if (targetP < 0) {
                maxP = javaMin(maxP, 0.0);
            } else {
                minP = javaMax(minP, 0.0);
            }
            return mismatch > 0. ? javaMax(0.0, __dadd_rn(maxP, -targetP)) : javaMax(0.0, __dadd_rn(targetP, -minP));
        }
    }
}

/** GenerationActivePowerDistributionStep.isParticipating. */
__device__ inline bool stepParticipating(int type, int participating, double droop, double pf) {
    if (!participating) {
        return false;
    }
    switch (type) {
        case PT_MAX:
            return droop != 0;
        case PT_PARTICIPATION_FACTOR:
            return pf > 0;
        default:
            return true;
    }
}

/**
 * ActivePowerDistribution.run with GenerationActivePowerDistributionStep, on the device generator
 * tables, one thread, sequential like the CPU (a few thousand generators, a handful of iterations):
 * the participating buses (participating, enabled, not fictitious, not excluded), the reset of the
 * reference generator then of every participating generator to its INITIAL target P (the undone
 * amount re-added to the mismatch), the participating elements with their factors, then the
 * water-filling loop with the JDK's compensated norm, normalized factors, sign-preserving limits and
 * iterator removal. Writes out[0] = remaining mismatch, out[1] = iterations.
 */
__global__ void distributeSlack(int nBuses, const int* busFlags, const int* excluded, const int* busGenStart,
                                const double* genInit, double* genTarget, const double* genMinT, const double* genMaxT,
                                const double* genMinP, const double* genMaxP, const double* genDroop, const double* genPf,
                                const int* genPart, int refGen, int type, int useActiveLimits, double mismatch,
                                int* elements, double* factors, double* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    // Step.resetToInitialState: the reference generator first (the interface default), then the
    // generation step's own loop over every participating generator of a participating bus, in bus
    // order — two separately accumulated amounts, added at the end, then added to the mismatch
    // (ActivePowerDistribution.run: remainingMismatch = activePowerMismatch + previousMismatch)
    double refPrevious = 0.0;
    if (refGen >= 0) {
        refPrevious = __dadd_rn(refPrevious, -__dadd_rn(genInit[refGen], -genTarget[refGen]));
        genTarget[refGen] = genInit[refGen];
    }
    double stepPrevious = 0.0;
    for (int b = 0; b < nBuses; b++) {
        bool busOk = (busFlags[b] & BUS_PARTICIPATING) && !(busFlags[b] & BUS_DISABLED) && !(busFlags[b] & BUS_FICTITIOUS)
                     && !(excluded != nullptr && excluded[b]);
        if (!busOk) {
            continue;
        }
        for (int g = busGenStart[b]; g < busGenStart[b + 1]; g++) {
            if (genPart[g]) {
                stepPrevious = __dadd_rn(stepPrevious, -__dadd_rn(genInit[g], -genTarget[g]));
                genTarget[g] = genInit[g];
            }
        }
    }
    double remaining = __dadd_rn(mismatch, __dadd_rn(stepPrevious, refPrevious));
    // getParticipatingElements, factors on the reset targets and the effective mismatch
    int count = 0;
    for (int b = 0; b < nBuses; b++) {
        bool busOk = (busFlags[b] & BUS_PARTICIPATING) && !(busFlags[b] & BUS_DISABLED) && !(busFlags[b] & BUS_FICTITIOUS)
                     && !(excluded != nullptr && excluded[b]);
        if (!busOk) {
            continue;
        }
        for (int g = busGenStart[b]; g < busGenStart[b + 1]; g++) {
            double f = participationFactor(type, genTarget[g], genMaxP[g], genMinP[g], genDroop[g], genPf[g], remaining);
            if (stepParticipating(type, genPart[g], genDroop[g], genPf[g]) && f != 0) {
                elements[count] = g;
                factors[count] = f;
                count++;
            }
        }
    }
    int iteration = 0;
    while (count > 0 && fabs(remaining) > P_RESIDUE_EPS) {
        double norm = javaStreamSum(factors, count);
        if (!(norm > 0.0)) {
            break;
        }
        // normalizeParticipationFactors
        for (int k = 0; k < count; k++) {
            factors[k] = __ddiv_rn(factors[k], norm);
        }
        double done = 0.0;
        int k = 0;
        while (k < count) {
            int g = elements[k];
            double factor = factors[k];
            double minTargetP = useActiveLimits ? genMinT[g] : -1.7976931348623157e308;
            double maxTargetP = useActiveLimits ? genMaxT[g] : 1.7976931348623157e308;
            double targetP = genTarget[g];
            if (targetP < 0) {
                maxTargetP = javaMin(maxTargetP, 0.0);
            } else {
                minTargetP = javaMax(minTargetP, 0.0);
            }
            double newTargetP = __dadd_rn(targetP, __dmul_rn(remaining, factor));
            bool drop = false;
            if (remaining > 0 && newTargetP > maxTargetP) {
                newTargetP = maxTargetP;
                drop = true;
            } else if (remaining < 0 && newTargetP < minTargetP) {
                newTargetP = minTargetP;
                drop = true;
            }
            if (newTargetP != targetP) {
                genTarget[g] = newTargetP;
                done = __dadd_rn(done, __dadd_rn(newTargetP, -targetP));
            }
            if (drop) {
                for (int j = k + 1; j < count; j++) {
                    elements[j - 1] = elements[j];
                    factors[j - 1] = factors[j];
                }
                count--;
            } else {
                k++;
            }
        }
        remaining = __dadd_rn(remaining, -done);
        iteration++;
    }
    out[0] = remaining;
    out[1] = (double) iteration;
}


// ---------------------------------------------------------------------------------------------
// DC outer loops on the device: the decisions of DcIncrementalPhaseControlOuterLoop and
// DcHvdcAcEmulationLimitsOuterLoop, read on the device state; the host applies them to the
// network with OLF's own mutators (setTapPosition, switchToSaturation.../switchToLinearMode)
// ---------------------------------------------------------------------------------------------

constexpr double DEGREES_TO_RADIANS = 0.017453292519943295;   // Math.toRadians(x) = x * DEGREES_TO_RADIANS
constexpr double SENSI_EPS = 1e-6;                            // AbstractIncrementalPhaseControlOuterLoop.SENSI_EPS
constexpr int MAX_DIRECTION_CHANGE = 3;                       // AbstractIncrementalPhaseControlOuterLoop.MAX_DIRECTION_CHANGE
constexpr int DIR_NONE = -1;
constexpr int DIR_INCREASE = 0;                               // Direction.INCREASE
constexpr int DIR_DECREASE = 1;                               // Direction.DECREASE
constexpr int ALLOWED_INCREASE = 0;                           // AllowedDirection.INCREASE
constexpr int ALLOWED_DECREASE = 1;
constexpr int ALLOWED_BOTH = 2;
constexpr int HVDC_LINEAR = 0;                                // AcEmulationStatus.LINEAR_MODE
constexpr int HVDC_SAT_12 = 1;                                // SATURATION_MODE_FROM_CS1_TO_CS2
constexpr int HVDC_SAT_21 = 2;                                // SATURATION_MODE_FROM_CS2_TO_CS1
constexpr int HVDC_FROZEN = 3;

/** DcSensitivityContext.calculateSensitivityValues' right-hand sides: toRadians(1) at the
 *  controller's BRANCH_TARGET_ALPHA1 column, one column per controller (zero elsewhere). */
__global__ void phaseRhs(int n, int nCtrl, const int* alphaCol, double* b) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCtrl) {
        return;
    }
    if (alphaCol[c] >= 0) {
        b[alphaCol[c] + (size_t) c * n] = DEGREES_TO_RADIANS;
    }
}

/** DcNetworkVector.updateClosedBranches' p1 / p2 of a branch on a state. */
__device__ inline double branchFlow(const double* x, int b, int side, const int* ph1Row, const int* ph2Row, const int* a1Row,
                                    const double* a1Const, const double* power) {
    if (ph1Row[b] < 0 || ph2Row[b] < 0) {
        return 0.0;
    }
    double a1 = a1Row[b] >= 0 ? x[a1Row[b]] : a1Const[b];
    double deltaPhase = __dadd_rn(__dadd_rn(__dadd_rn(x[ph2Row[b]], -x[ph1Row[b]]), A2), -a1);
    return side == 1 ? __dmul_rn(-power[b], deltaPhase) : __dmul_rn(power[b], deltaPhase);
}

/**
 * AbstractIncrementalPhaseControlOuterLoop.checkActivePowerControlPhaseControls, one thread per
 * controller: the controlled flow on the state, the deadband, the sensitivity from the solved
 * column (AbstractClosedBranchDcFlowEquationTermArrayEvaluator.calculateSensi: dp/dph1·S[ph1] +
 * dp/dph2·S[ph2] + dp/da1·S[a1]), da = toRadians(dp / a2p), then
 * PiModelArray.updateTapPositionToReachNewA1 with the ClosestTapPositionFinder over the allowed
 * range (MAX_TAP_SHIFT infinite), and ControllerContext.updateAllowedDirection. Writes the new tap
 * index, or -1 when the tap does not move; the direction state is per branch and persists.
 */
__global__ void phaseDecision(int n, int nCtrl, const double* x, const double* sens, const int* pcBranch, const int* pcControlled,
                              const int* pcSide, const double* pcTarget, const double* pcHalfDb, int* pcTapIdx,
                              const int* pcTapPtr, const double* pcTapA1, int* allowed, int* dirCount, int* curDir,
                              const int* ph1Row, const int* ph2Row, const int* a1Row, const double* a1Const, const double* power,
                              int* newTap) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCtrl) {
        return;
    }
    newTap[c] = -1;
    int controller = pcBranch[c];
    int controlled = pcControlled[c];
    int side = pcSide[c];
    double pValue = branchFlow(x, controlled, side, ph1Row, ph2Row, a1Row, a1Const, power);
    if (!(fabs(__dadd_rn(pValue, -pcTarget[c])) > pcHalfDb[c])) {
        return;
    }
    double dp = __dadd_rn(pcTarget[c], -pValue);
    const double* s = sens + (size_t) c * n;
    double dph1 = s[ph1Row[controlled]];
    double dph2 = s[ph2Row[controlled]];
    double da1 = a1Row[controlled] >= 0 ? s[a1Row[controlled]] : 0.0;
    double p = power[controlled];
    // side 1: [p, -p, p]; side 2: [-p, p, -p] — ((d1*dph1) + (d2*dph2)) + (d3*da1)
    double a2p = side == 1
        ? __dadd_rn(__dadd_rn(__dmul_rn(p, dph1), __dmul_rn(-p, dph2)), __dmul_rn(p, da1))
        : __dadd_rn(__dadd_rn(__dmul_rn(-p, dph1), __dmul_rn(p, dph2)), __dmul_rn(-p, da1));
    if (!(fabs(a2p) > SENSI_EPS)) {
        return;
    }
    double da = __dmul_rn(__ddiv_rn(dp, a2p), DEGREES_TO_RADIANS);
    // PiModelArray.getA1(): the continuous a1 the network holds after updateNetwork = the solved alpha
    double currentA1 = a1Row[controller] >= 0 ? x[a1Row[controller]] : a1Const[controller];
    double newA1 = __dadd_rn(currentA1, da);
    int tapStart = pcTapPtr[c];
    int tapCount = pcTapPtr[c + 1] - tapStart;
    int idx = pcTapIdx[c];
    int lo = 0;
    int hi = tapCount - 1;
    switch (allowed[controller]) {
        case ALLOWED_INCREASE: lo = idx; break;
        case ALLOWED_DECREASE: hi = idx; break;
        default: break;
    }
    int closest = idx;
    double smallest = fabs(__dadd_rn(newA1, -pcTapA1[tapStart + idx]));
    for (int i = lo; i <= hi; i++) {
        double distance = fabs(__dadd_rn(newA1, -pcTapA1[tapStart + i]));
        if (distance < smallest) {
            closest = i;
            smallest = distance;
        }
    }
    if (closest == idx) {
        return;
    }
    int direction = closest - idx > 0 ? DIR_INCREASE : DIR_DECREASE;
    // ControllerContext.updateAllowedDirection
    if (dirCount[controller] < MAX_DIRECTION_CHANGE) {
        if (curDir[controller] != DIR_NONE && curDir[controller] != direction) {
            dirCount[controller]++;
        }
        curDir[controller] = direction;
    } else {
        allowed[controller] = direction == DIR_INCREASE ? ALLOWED_INCREASE : ALLOWED_DECREASE;
    }
    pcTapIdx[c] = closest;
    newTap[c] = closest;
}

/**
 * AbstractHvdcAcEmulationLimitsOuterLoop.checkAcEmulationMode, one thread per HVDC, on the
 * emulation flows of the state (HvdcAcEmulationSide1/2DCFlowEquationTerm.eval: k (ph1 - ph2) + p0
 * and k (ph2 - ph1) - p0). Writes the new status, or -1 when the status stays.
 */
__global__ void hvdcDecision(int nHvdc, const double* x, const int* ph1, const int* ph2, const double* k, const double* p0,
                             const double* pmax12, const double* pmax21, const int* status, const int* active, int* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nHvdc) {
        return;
    }
    out[h] = -1;
    if (!active[h]) {
        return;
    }
    double p1 = __dadd_rn(__dmul_rn(k[h], __dadd_rn(x[ph1[h]], -x[ph2[h]])), p0[h]);
    double p2 = __dadd_rn(__dmul_rn(k[h], __dadd_rn(x[ph2[h]], -x[ph1[h]])), -p0[h]);
    switch (status[h]) {
        case HVDC_LINEAR:
            if (p1 >= 0) {
                if (p1 > pmax12[h]) {
                    out[h] = HVDC_SAT_12;
                }
            } else if (p2 > pmax21[h]) {
                out[h] = HVDC_SAT_21;
            }
            break;
        case HVDC_SAT_12:
        case HVDC_SAT_21:
            if (p1 > 0 && p1 < pmax12[h] || p2 > 0 && p2 < pmax21[h]) {
                out[h] = HVDC_LINEAR;
            } else if (p1 < 0 && status[h] == HVDC_SAT_12) {
                out[h] = HVDC_SAT_21;
            } else if (p2 < 0 && status[h] == HVDC_SAT_21) {
                out[h] = HVDC_SAT_12;
            }
            break;
        default:
            break;
    }
}

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

/** The cuDSS handle, stream and configuration shared by both context creations. */
void initCudss(DcContext* ctx) {
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
}

void requireBusesAndGenerators(DcContext* ctx) {
    if (!ctx->assembled || ctx->nBuses == 0 || ctx->genTarget.p == nullptr) {
        throw std::runtime_error("context not set up (assembled, buses, generators)");
    }
}

/** The per-bus generation (from the device generator targets), then the components' mismatches. */
void refreshBusGeneration(DcContext* ctx) {
    busGeneration<<<grid(ctx->nBuses), BLOCK, 0, ctx->stream>>>(ctx->nBuses, ctx->busGenStart.as<int>(), ctx->genTarget.as<double>(),
            ctx->busGenP.as<double>());
    DC_CUDA_CHECK(cudaGetLastError());
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
        initCudss(ctx);
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

/** Per-element Woodbury terms (see DcContext.elemDiag / elemNewAlpha), one entry per S column. */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setElementAttributes(JNIEnv* env, jclass, jlong handle, jdoubleArray jDiag, jdoubleArray jNewAlpha) {
    DcContext* ctx = ctxOf(handle);
    try {
        DoubleArray diag(env, jDiag);
        DoubleArray newAlpha(env, jNewAlpha);
        if (diag.len != ctx->nElem || newAlpha.len != ctx->nElem) {
            throw std::runtime_error("element attribute arrays must have one entry per element");
        }
        ctx->elemDiag.upload(diag.p, (size_t) ctx->nElem * sizeof(double));
        ctx->elemNewAlpha.upload(newAlpha.p, (size_t) ctx->nElem * sizeof(double));
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setElementAttributes: ") + e.what());
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
        if ((ctx->nElem > 0 && ctx->s.p == nullptr) || ctx->nBranches == 0) {
            throw std::runtime_error("context not set up (elements, branches)");
        }
        ctx->ctgPtr.upload(ctgPtr.p, (size_t) ctgPtr.len * sizeof(int));
        ctx->ctgElem.upload(ctgElem.p, (size_t) std::max<jsize>(ctgElem.len, 1) * sizeof(int));
        Buf out;
        out.alloc((size_t) nCtg * sizeof(double));
        const double* sPtr = ctx->nElem > 0 ? ctx->s.as<double>() : ctx->power.as<double>();   // never read when nElem == 0
        connectivityCriterion<<<grid(nCtg), BLOCK, 0, ctx->stream>>>(nCtg, ctx->n, ctx->ctgPtr.as<int>(), ctx->ctgElem.as<int>(),
                ctx->elemBranch.as<int>(), ctx->ph1Row.as<int>(), ctx->ph2Row.as<int>(), ctx->a1Row.as<int>(), ctx->power.as<double>(),
                sPtr, out.as<double>());
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
        if (ctx->x0.p == nullptr || (ctx->nElem > 0 && ctx->s.p == nullptr) || ctx->nBranches == 0) {
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
        const double* sPtr = ctx->nElem > 0 ? ctx->s.as<double>() : ctx->x0.as<double>();       // never read when nElem == 0
        if (ctx->nElem > 0 && ctx->elemDiag.p == nullptr) {
            throw std::runtime_error("element attributes not set (setElementAttributes)");
        }
        const double* diagPtr = ctx->nElem > 0 ? ctx->elemDiag.as<double>() : ctx->x0.as<double>();
        const double* newAlphaPtr = ctx->nElem > 0 ? ctx->elemNewAlpha.as<double>() : ctx->x0.as<double>();
        woodburyAlphas<<<grid(nCtg), BLOCK, 0, ctx->stream>>>(nCtg, n, ctx->ctgPtr.as<int>(), ctx->ctgElem.as<int>(), ctx->ctgBase.as<int>(),
                ctx->elemBranch.as<int>(), ctx->ph1Row.as<int>(), ctx->ph2Row.as<int>(),
                diagPtr, newAlphaPtr, sPtr, ctx->x0.as<double>(), bases, ctx->alpha.as<double>(), bad.as<int>());
        DC_CUDA_CHECK(cudaGetLastError());
        woodburyStates<<<grid((long) n * nCtg), BLOCK, 0, ctx->stream>>>(nCtg, n, ctx->ctgPtr.as<int>(), ctx->ctgElem.as<int>(), ctx->ctgBase.as<int>(),
                ctx->alpha.as<double>(), sPtr, ctx->x0.as<double>(), bases, ctx->states.as<double>());
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


// ---------------------------------------------------------------------------------------------
// device assembly entry points (GpuDcAssembly)
// ---------------------------------------------------------------------------------------------

/** A context on the STRUCTURE only: pattern, value and target contribution lists. No value, no
 *  factorization yet — setBranches / setBuses / setGenerators, then fillValues. */
JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_createAssembled(JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx,
                                                                 jintArray jContribPtr, jintArray jContribKind, jintArray jContribElem,
                                                                 jdoubleArray jContribCoef, jintArray jTargetKind, jintArray jTargetElem,
                                                                 jintArray jRhsPtr, jintArray jRhsKind, jintArray jRhsElem, jdoubleArray jRhsCoef) {
    DcContext* ctx = new DcContext();
    try {
        IntArray rowPtr(env, jRowPtr);
        IntArray colIdx(env, jColIdx);
        IntArray contribPtr(env, jContribPtr);
        IntArray contribKind(env, jContribKind);
        IntArray contribElem(env, jContribElem);
        DoubleArray contribCoef(env, jContribCoef);
        IntArray targetKind(env, jTargetKind);
        IntArray targetElem(env, jTargetElem);
        IntArray rhsPtr(env, jRhsPtr);
        IntArray rhsKind(env, jRhsKind);
        IntArray rhsElem(env, jRhsElem);
        DoubleArray rhsCoef(env, jRhsCoef);
        if (rowPtr.len != n + 1 || targetKind.len != n || targetElem.len != n || rhsPtr.len != n + 1) {
            throw std::runtime_error("per-column arrays must have length n (+1 for the pointers)");
        }
        ctx->n = n;
        ctx->nnz = rowPtr.p[n];
        if (colIdx.len < ctx->nnz || contribPtr.len != ctx->nnz + 1) {
            throw std::runtime_error("CSR arrays inconsistent with rowPtr[n]");
        }
        int nContrib = contribPtr.p[ctx->nnz];
        if (contribKind.len != nContrib || contribElem.len != nContrib || contribCoef.len != nContrib) {
            throw std::runtime_error("contribution arrays inconsistent with contribPtr");
        }
        int nRhs = rhsPtr.p[n];
        if (rhsKind.len != nRhs || rhsElem.len != nRhs || rhsCoef.len != nRhs) {
            throw std::runtime_error("rhs arrays inconsistent with rhsPtr");
        }
        initCudss(ctx);
        ctx->rowPtr.upload(rowPtr.p, (size_t) (n + 1) * sizeof(int));
        ctx->colIdx.upload(colIdx.p, (size_t) ctx->nnz * sizeof(int));
        ctx->val.alloc((size_t) ctx->nnz * sizeof(double));
        DC_CUDA_CHECK(cudaMemset(ctx->val.p, 0, (size_t) ctx->nnz * sizeof(double)));
        ctx->contribPtr.upload(contribPtr.p, (size_t) (ctx->nnz + 1) * sizeof(int));
        ctx->contribKind.upload(contribKind.p, (size_t) std::max(nContrib, 1) * sizeof(int));
        ctx->contribElem.upload(contribElem.p, (size_t) std::max(nContrib, 1) * sizeof(int));
        ctx->contribCoef.upload(contribCoef.p, (size_t) std::max(nContrib, 1) * sizeof(double));
        ctx->targetKind.upload(targetKind.p, (size_t) n * sizeof(int));
        ctx->targetElem.upload(targetElem.p, (size_t) n * sizeof(int));
        ctx->rhsPtr.upload(rhsPtr.p, (size_t) (n + 1) * sizeof(int));
        ctx->rhsKind.upload(rhsKind.p, (size_t) std::max(nRhs, 1) * sizeof(int));
        ctx->rhsElem.upload(rhsElem.p, (size_t) std::max(nRhs, 1) * sizeof(int));
        ctx->rhsCoef.upload(rhsCoef.p, (size_t) std::max(nRhs, 1) * sizeof(double));
        ctx->t.alloc((size_t) n * sizeof(double));
        ctx->scalarOut.alloc(sizeof(double));
        ctx->dsOut.alloc(2 * sizeof(double));
        DC_CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->matA, n, n, ctx->nnz, ctx->rowPtr.p, nullptr, ctx->colIdx.p, ctx->val.p,
                                            DC_CUDSS_CSR_TYPES, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        ctx->ensureRhs(1);
        ctx->assembled = true;
        return reinterpret_cast<jlong>(ctx);
    } catch (const std::exception& e) {
        delete ctx;
        throwJava(env, std::string("GpuDcSolver.createAssembled: ") + e.what());
        return 0;
    }
}

/** The bus tables: the active P column, the angle row, the load target P, the flags, the
 *  synchronous component and the generator range of every bus; the components' bus lists and
 *  slack counts. */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setBuses(JNIEnv* env, jclass, jlong handle, jintArray jPColumn, jintArray jPhRow,
                                                          jdoubleArray jLoadP, jintArray jFlags, jintArray jSc, jintArray jGenStart,
                                                          jintArray jScBusPtr, jintArray jScBusList, jintArray jScSlackCount) {
    DcContext* ctx = ctxOf(handle);
    try {
        IntArray pColumn(env, jPColumn);
        IntArray phRow(env, jPhRow);
        DoubleArray loadP(env, jLoadP);
        IntArray flags(env, jFlags);
        IntArray sc(env, jSc);
        IntArray genStart(env, jGenStart);
        IntArray scBusPtr(env, jScBusPtr);
        IntArray scBusList(env, jScBusList);
        IntArray scSlackCount(env, jScSlackCount);
        int nb = pColumn.len;
        if (phRow.len != nb || loadP.len != nb || flags.len != nb || sc.len != nb || genStart.len != nb + 1) {
            throw std::runtime_error("bus arrays must have the same length");
        }
        int nSc = scBusPtr.len - 1;
        if (nSc < 0 || scSlackCount.len != nSc || scBusList.len < scBusPtr.p[nSc]) {
            throw std::runtime_error("synchronous component arrays inconsistent");
        }
        ctx->nBuses = nb;
        ctx->nSc = nSc;
        ctx->busPColumn.upload(pColumn.p, (size_t) nb * sizeof(int));
        ctx->busPhRow.upload(phRow.p, (size_t) nb * sizeof(int));
        ctx->busLoadP.upload(loadP.p, (size_t) nb * sizeof(double));
        ctx->busFlags.upload(flags.p, (size_t) nb * sizeof(int));
        ctx->busSc.upload(sc.p, (size_t) nb * sizeof(int));
        ctx->busGenStart.upload(genStart.p, (size_t) (nb + 1) * sizeof(int));
        ctx->busGenP.alloc((size_t) nb * sizeof(double));
        ctx->busExcluded.alloc((size_t) nb * sizeof(int));
        DC_CUDA_CHECK(cudaMemset(ctx->busExcluded.p, 0, (size_t) nb * sizeof(int)));
        ctx->scBusPtr.upload(scBusPtr.p, (size_t) (nSc + 1) * sizeof(int));
        ctx->scBusList.upload(scBusList.p, (size_t) std::max<jsize>(scBusList.len, 1) * sizeof(int));
        ctx->scSlackCount.upload(scSlackCount.p, (size_t) std::max(nSc, 1) * sizeof(int));
        ctx->scMismatch.alloc((size_t) std::max(nSc, 1) * sizeof(double));
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setBuses: ") + e.what());
    }
}

/** The loads' target P per bus (a time-series step, a refreshed network). */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setBusLoads(JNIEnv* env, jclass, jlong handle, jdoubleArray jLoadP) {
    DcContext* ctx = ctxOf(handle);
    try {
        DoubleArray loadP(env, jLoadP);
        if (loadP.len != ctx->nBuses) {
            throw std::runtime_error("loadP length must be the bus count");
        }
        DC_CUDA_CHECK(cudaMemcpy(ctx->busLoadP.p, loadP.p, (size_t) ctx->nBuses * sizeof(double), cudaMemcpyHostToDevice));
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setBusLoads: ") + e.what());
    }
}

/** The buses excluded from the mismatch and the distribution (islanded by a contingency); null = none. */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setExcludedBuses(JNIEnv* env, jclass, jlong handle, jintArray jBuses) {
    DcContext* ctx = ctxOf(handle);
    try {
        std::vector<int> mask(ctx->nBuses, 0);
        if (jBuses != nullptr) {
            IntArray buses(env, jBuses);
            for (jsize i = 0; i < buses.len; i++) {
                if (buses.p[i] < 0 || buses.p[i] >= ctx->nBuses) {
                    throw std::runtime_error("excluded bus out of range");
                }
                mask[buses.p[i]] = 1;
            }
        }
        DC_CUDA_CHECK(cudaMemcpy(ctx->busExcluded.p, mask.data(), (size_t) ctx->nBuses * sizeof(int), cudaMemcpyHostToDevice));
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setExcludedBuses: ") + e.what());
    }
}

/** The generator tables (GenerationActivePowerDistributionStep's inputs), in bus-then-generator order. */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setGenerators(JNIEnv* env, jclass, jlong handle, jdoubleArray jInit, jdoubleArray jTarget,
                                                               jdoubleArray jMinT, jdoubleArray jMaxT, jdoubleArray jMinP, jdoubleArray jMaxP,
                                                               jdoubleArray jDroop, jdoubleArray jPf, jintArray jPart) {
    DcContext* ctx = ctxOf(handle);
    try {
        DoubleArray init(env, jInit);
        DoubleArray target(env, jTarget);
        DoubleArray minT(env, jMinT);
        DoubleArray maxT(env, jMaxT);
        DoubleArray minP(env, jMinP);
        DoubleArray maxP(env, jMaxP);
        DoubleArray droop(env, jDroop);
        DoubleArray pf(env, jPf);
        IntArray part(env, jPart);
        int m = init.len;
        if (target.len != m || minT.len != m || maxT.len != m || minP.len != m || maxP.len != m || droop.len != m || pf.len != m || part.len != m) {
            throw std::runtime_error("generator arrays must have the same length");
        }
        ctx->nGen = m;
        size_t bytes = (size_t) std::max(m, 1) * sizeof(double);
        ctx->genInit.upload(init.p, bytes);
        ctx->genTarget.upload(target.p, bytes);
        ctx->genMinT.upload(minT.p, bytes);
        ctx->genMaxT.upload(maxT.p, bytes);
        ctx->genMinP.upload(minP.p, bytes);
        ctx->genMaxP.upload(maxP.p, bytes);
        ctx->genDroop.upload(droop.p, bytes);
        ctx->genPf.upload(pf.p, bytes);
        ctx->genPart.upload(part.p, (size_t) std::max(m, 1) * sizeof(int));
        ctx->dsElem.alloc((size_t) std::max(m, 1) * sizeof(int));
        ctx->dsFactor.alloc(bytes);
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setGenerators: ") + e.what());
    }
}

/** The generators' current target P, overwritten (the host distributed, or restored, the network). */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setGeneratorTargets(JNIEnv* env, jclass, jlong handle, jdoubleArray jTarget) {
    DcContext* ctx = ctxOf(handle);
    try {
        DoubleArray target(env, jTarget);
        if (target.len != ctx->nGen) {
            throw std::runtime_error("target length must be the generator count");
        }
        if (ctx->nGen > 0) {
            DC_CUDA_CHECK(cudaMemcpy(ctx->genTarget.p, target.p, (size_t) ctx->nGen * sizeof(double), cudaMemcpyHostToDevice));
        }
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setGeneratorTargets: ") + e.what());
    }
}

JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_readGeneratorTargets(JNIEnv* env, jclass, jlong handle) {
    DcContext* ctx = ctxOf(handle);
    try {
        std::vector<double> host((size_t) ctx->nGen);
        if (ctx->nGen > 0) {
            DC_CUDA_CHECK(cudaMemcpy(host.data(), ctx->genTarget.p, (size_t) ctx->nGen * sizeof(double), cudaMemcpyDeviceToHost));
        }
        jdoubleArray out = env->NewDoubleArray(ctx->nGen);
        env->SetDoubleArrayRegion(out, 0, ctx->nGen, host.data());
        return out;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.readGeneratorTargets: ") + e.what());
        return nullptr;
    }
}

/** The matrix values from the branch powers (setBranches), then the factorization — the analysis
 *  once, on the first fill. */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_fillValues(JNIEnv* env, jclass, jlong handle) {
    DcContext* ctx = ctxOf(handle);
    try {
        if (!ctx->assembled) {
            throw std::runtime_error("not an assembled context");
        }
        if (ctx->power.p == nullptr) {
            throw std::runtime_error("branches not set (setBranches)");
        }
        fillValues<<<grid(ctx->nnz), BLOCK, 0, ctx->stream>>>(ctx->nnz, ctx->contribPtr.as<int>(), ctx->contribKind.as<int>(),
                ctx->contribElem.as<int>(), ctx->contribCoef.as<double>(), ctx->power.as<double>(), ctx->val.as<double>());
        DC_CUDA_CHECK(cudaGetLastError());
        ctx->ensureRhs(1);
        if (!ctx->analyzed) {
            DC_CUDSS_CHECK(cudssExecute(ctx->handle, CUDSS_PHASE_ANALYSIS, ctx->config, ctx->data, ctx->matA, ctx->matX, ctx->matB));
            ctx->analyzed = true;
        }
        DC_CUDSS_CHECK(cudssExecute(ctx->handle, CUDSS_PHASE_FACTORIZATION, ctx->config, ctx->data, ctx->matA, ctx->matX, ctx->matB));
        DC_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.fillValues: ") + e.what());
    }
}

JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_readValues(JNIEnv* env, jclass, jlong handle) {
    DcContext* ctx = ctxOf(handle);
    try {
        std::vector<double> host((size_t) ctx->nnz);
        DC_CUDA_CHECK(cudaMemcpy(host.data(), ctx->val.p, (size_t) ctx->nnz * sizeof(double), cudaMemcpyDeviceToHost));
        jdoubleArray out = env->NewDoubleArray(ctx->nnz);
        env->SetDoubleArrayRegion(out, 0, ctx->nnz, host.data());
        return out;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.readValues: ") + e.what());
        return nullptr;
    }
}


/** DcLoadFlowEngine.getActivePowerMismatch over the enabled, non-excluded buses. */
JNIEXPORT jdouble JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_activePowerMismatch(JNIEnv* env, jclass, jlong handle, jboolean withExcluded) {
    DcContext* ctx = ctxOf(handle);
    try {
        requireBusesAndGenerators(ctx);
        refreshBusGeneration(ctx);
        activePowerMismatch<<<1, 32, 0, ctx->stream>>>(ctx->nBuses, ctx->busFlags.as<int>(), withExcluded ? ctx->busExcluded.as<int>() : nullptr,
                ctx->busGenP.as<double>(), ctx->busLoadP.as<double>(), ctx->scalarOut.as<double>());
        DC_CUDA_CHECK(cudaGetLastError());
        double out = 0;
        DC_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        DC_CUDA_CHECK(cudaMemcpy(&out, ctx->scalarOut.p, sizeof(double), cudaMemcpyDeviceToHost));
        return out;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.activePowerMismatch: ") + e.what());
        return 0;
    }
}

/** ActivePowerDistribution.run on the device generator targets; returns [remaining, iterations]. */
JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_distributeSlack(JNIEnv* env, jclass, jlong handle, jint participationType,
                                                                 jboolean useActiveLimits, jint referenceGenerator, jboolean withExcluded,
                                                                 jdouble mismatch) {
    DcContext* ctx = ctxOf(handle);
    try {
        requireBusesAndGenerators(ctx);
        if (referenceGenerator >= ctx->nGen) {
            throw std::runtime_error("reference generator out of range");
        }
        distributeSlack<<<1, 32, 0, ctx->stream>>>(ctx->nBuses, ctx->busFlags.as<int>(), withExcluded ? ctx->busExcluded.as<int>() : nullptr,
                ctx->busGenStart.as<int>(), ctx->genInit.as<double>(), ctx->genTarget.as<double>(), ctx->genMinT.as<double>(),
                ctx->genMaxT.as<double>(), ctx->genMinP.as<double>(), ctx->genMaxP.as<double>(), ctx->genDroop.as<double>(),
                ctx->genPf.as<double>(), ctx->genPart.as<int>(), referenceGenerator, participationType, useActiveLimits ? 1 : 0, mismatch,
                ctx->dsElem.as<int>(), ctx->dsFactor.as<double>(), ctx->dsOut.as<double>());
        DC_CUDA_CHECK(cudaGetLastError());
        DC_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        double out[2];
        DC_CUDA_CHECK(cudaMemcpy(out, ctx->dsOut.p, 2 * sizeof(double), cudaMemcpyDeviceToHost));
        jdoubleArray result = env->NewDoubleArray(2);
        env->SetDoubleArrayRegion(result, 0, 2, out);
        return result;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.distributeSlack: ") + e.what());
        return nullptr;
    }
}

/** DcTargetVector on the device from the current generator targets, loads and branch model. */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_fillTarget(JNIEnv* env, jclass, jlong handle) {
    DcContext* ctx = ctxOf(handle);
    try {
        requireBusesAndGenerators(ctx);
        if (ctx->power.p == nullptr) {
            throw std::runtime_error("branches not set (setBranches)");
        }
        refreshBusGeneration(ctx);
        if (ctx->nSc > 0) {
            scMismatches<<<grid(ctx->nSc), BLOCK, 0, ctx->stream>>>(ctx->nSc, ctx->scBusPtr.as<int>(), ctx->scBusList.as<int>(),
                    ctx->busFlags.as<int>(), ctx->busGenP.as<double>(), ctx->busLoadP.as<double>(), ctx->scMismatch.as<double>());
            DC_CUDA_CHECK(cudaGetLastError());
        }
        fillTarget<<<grid(ctx->n), BLOCK, 0, ctx->stream>>>(ctx->n, ctx->targetKind.as<int>(), ctx->targetElem.as<int>(), ctx->rhsPtr.as<int>(),
                ctx->rhsKind.as<int>(), ctx->rhsElem.as<int>(), ctx->rhsCoef.as<double>(), ctx->busGenP.as<double>(), ctx->busLoadP.as<double>(),
                ctx->busFlags.as<int>(), ctx->busSc.as<int>(), ctx->scSlackCount.as<int>(), ctx->scMismatch.as<double>(),
                ctx->a1Const.as<double>(), ctx->a1Row.as<int>(), ctx->power.as<double>(), ctx->t.as<double>());
        DC_CUDA_CHECK(cudaGetLastError());
        DC_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.fillTarget: ") + e.what());
    }
}

JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_readTarget(JNIEnv* env, jclass, jlong handle) {
    DcContext* ctx = ctxOf(handle);
    try {
        if (ctx->t.p == nullptr) {
            throw std::runtime_error("not an assembled context");
        }
        std::vector<double> host((size_t) ctx->n);
        DC_CUDA_CHECK(cudaMemcpy(host.data(), ctx->t.p, (size_t) ctx->n * sizeof(double), cudaMemcpyDeviceToHost));
        jdoubleArray out = env->NewDoubleArray(ctx->n);
        env->SetDoubleArrayRegion(out, 0, ctx->n, host.data());
        return out;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.readTarget: ") + e.what());
        return nullptr;
    }
}

/** Solve J x = t for the device target vector (fillTarget); returns x. */
JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_solveTarget(JNIEnv* env, jclass, jlong handle) {
    DcContext* ctx = ctxOf(handle);
    try {
        if (!ctx->assembled || !ctx->analyzed) {
            throw std::runtime_error("not an assembled and factorized context (fillValues)");
        }
        ctx->ensureRhs(1);
        DC_CUDA_CHECK(cudaMemcpy(ctx->b.p, ctx->t.p, (size_t) ctx->n * sizeof(double), cudaMemcpyDeviceToDevice));
        ctx->solveDense(1);
        ctx->lfState.ensure((size_t) ctx->n * sizeof(double));
        DC_CUDA_CHECK(cudaMemcpy(ctx->lfState.p, ctx->x.p, (size_t) ctx->n * sizeof(double), cudaMemcpyDeviceToDevice));
        jdoubleArray out = env->NewDoubleArray(ctx->n);
        std::vector<double> host(ctx->n);
        DC_CUDA_CHECK(cudaMemcpy(host.data(), ctx->x.p, (size_t) ctx->n * sizeof(double), cudaMemcpyDeviceToHost));
        env->SetDoubleArrayRegion(out, 0, ctx->n, host.data());
        return out;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.solveTarget: ") + e.what());
        return nullptr;
    }
}


/** The value and rhs contribution lists refreshed on the SAME pattern (a term switched on or off:
 *  an HVDC saturation) — no new analysis; fillValues refactorizes afterwards. */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setContributions(JNIEnv* env, jclass, jlong handle, jintArray jContribPtr, jintArray jContribKind,
                                                                  jintArray jContribElem, jdoubleArray jContribCoef, jintArray jTargetKind,
                                                                  jintArray jTargetElem, jintArray jRhsPtr, jintArray jRhsKind, jintArray jRhsElem,
                                                                  jdoubleArray jRhsCoef) {
    DcContext* ctx = ctxOf(handle);
    try {
        IntArray contribPtr(env, jContribPtr);
        IntArray contribKind(env, jContribKind);
        IntArray contribElem(env, jContribElem);
        DoubleArray contribCoef(env, jContribCoef);
        IntArray targetKind(env, jTargetKind);
        IntArray targetElem(env, jTargetElem);
        IntArray rhsPtr(env, jRhsPtr);
        IntArray rhsKind(env, jRhsKind);
        IntArray rhsElem(env, jRhsElem);
        DoubleArray rhsCoef(env, jRhsCoef);
        int n = ctx->n;
        if (!ctx->assembled || contribPtr.len != ctx->nnz + 1 || targetKind.len != n || targetElem.len != n || rhsPtr.len != n + 1) {
            throw std::runtime_error("arrays inconsistent with the context");
        }
        int nContrib = contribPtr.p[ctx->nnz];
        int nRhs = rhsPtr.p[n];
        if (contribKind.len != nContrib || contribElem.len != nContrib || contribCoef.len != nContrib
                || rhsKind.len != nRhs || rhsElem.len != nRhs || rhsCoef.len != nRhs) {
            throw std::runtime_error("contribution arrays inconsistent with their pointers");
        }
        ctx->contribPtr.upload(contribPtr.p, (size_t) (ctx->nnz + 1) * sizeof(int));
        ctx->contribKind.upload(contribKind.p, (size_t) std::max(nContrib, 1) * sizeof(int));
        ctx->contribElem.upload(contribElem.p, (size_t) std::max(nContrib, 1) * sizeof(int));
        ctx->contribCoef.upload(contribCoef.p, (size_t) std::max(nContrib, 1) * sizeof(double));
        ctx->targetKind.upload(targetKind.p, (size_t) n * sizeof(int));
        ctx->targetElem.upload(targetElem.p, (size_t) n * sizeof(int));
        ctx->rhsPtr.upload(rhsPtr.p, (size_t) (n + 1) * sizeof(int));
        ctx->rhsKind.upload(rhsKind.p, (size_t) std::max(nRhs, 1) * sizeof(int));
        ctx->rhsElem.upload(rhsElem.p, (size_t) std::max(nRhs, 1) * sizeof(int));
        ctx->rhsCoef.upload(rhsCoef.p, (size_t) std::max(nRhs, 1) * sizeof(double));
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setContributions: ") + e.what());
    }
}

/** The phase controllers (active power control): controller branch, its BRANCH_TARGET_ALPHA1 column
 *  (-1 = none), controlled branch and side, target and half deadband, current tap index and the
 *  tap tables' a1 values (CSR). */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setPhaseControllers(JNIEnv* env, jclass, jlong handle, jintArray jBranch, jintArray jAlphaCol,
                                                                     jintArray jControlled, jintArray jSide, jdoubleArray jTarget, jdoubleArray jHalfDb,
                                                                     jintArray jTapIdx, jintArray jTapPtr, jdoubleArray jTapA1) {
    DcContext* ctx = ctxOf(handle);
    try {
        IntArray branch(env, jBranch);
        IntArray alphaCol(env, jAlphaCol);
        IntArray controlled(env, jControlled);
        IntArray side(env, jSide);
        DoubleArray target(env, jTarget);
        DoubleArray halfDb(env, jHalfDb);
        IntArray tapIdx(env, jTapIdx);
        IntArray tapPtr(env, jTapPtr);
        DoubleArray tapA1(env, jTapA1);
        int m = branch.len;
        if (alphaCol.len != m || controlled.len != m || side.len != m || target.len != m || halfDb.len != m || tapIdx.len != m || tapPtr.len != m + 1) {
            throw std::runtime_error("controller arrays must have the same length");
        }
        if (ctx->nBranches == 0) {
            throw std::runtime_error("branches not set (setBranches)");
        }
        ctx->nCtrl = m;
        size_t ints = (size_t) std::max(m, 1) * sizeof(int);
        size_t dbls = (size_t) std::max(m, 1) * sizeof(double);
        ctx->pcBranch.upload(branch.p, ints);
        ctx->pcAlphaCol.upload(alphaCol.p, ints);
        ctx->pcControlled.upload(controlled.p, ints);
        ctx->pcSide.upload(side.p, ints);
        ctx->pcTarget.upload(target.p, dbls);
        ctx->pcHalfDb.upload(halfDb.p, dbls);
        ctx->pcTapIdx.upload(tapIdx.p, ints);
        ctx->pcTapPtr.upload(tapPtr.p, (size_t) (m + 1) * sizeof(int));
        ctx->pcTapA1.upload(tapA1.p, (size_t) std::max<jsize>(tapA1.len, 1) * sizeof(double));
        ctx->pcNewTap.ensure(ints);
        if (ctx->pcAllowed.p == nullptr) {
            // the direction state is per branch and lives across prepare() calls; reset by resetPhaseControlDirections
            std::vector<int> both(ctx->nBranches, ALLOWED_BOTH);
            std::vector<int> zero(ctx->nBranches, 0);
            std::vector<int> none(ctx->nBranches, DIR_NONE);
            ctx->pcAllowed.upload(both.data(), (size_t) ctx->nBranches * sizeof(int));
            ctx->pcDirCount.upload(zero.data(), (size_t) ctx->nBranches * sizeof(int));
            ctx->pcCurDir.upload(none.data(), (size_t) ctx->nBranches * sizeof(int));
        }
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setPhaseControllers: ") + e.what());
    }
}

/** IncrementalContextData rebuilt: every controller back to BOTH directions, no change counted. */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_resetPhaseControlDirections(JNIEnv* env, jclass, jlong handle) {
    DcContext* ctx = ctxOf(handle);
    try {
        if (ctx->nBranches == 0) {
            return;
        }
        std::vector<int> both(ctx->nBranches, ALLOWED_BOTH);
        std::vector<int> zero(ctx->nBranches, 0);
        std::vector<int> none(ctx->nBranches, DIR_NONE);
        ctx->pcAllowed.upload(both.data(), (size_t) ctx->nBranches * sizeof(int));
        ctx->pcDirCount.upload(zero.data(), (size_t) ctx->nBranches * sizeof(int));
        ctx->pcCurDir.upload(none.data(), (size_t) ctx->nBranches * sizeof(int));
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.resetPhaseControlDirections: ") + e.what());
    }
}

/** DcIncrementalPhaseControlOuterLoop.check on the device state: the sensitivities (one solve,
 *  one column per controller, on the current factorization) and the tap decisions. Returns the
 *  new tap INDEX per controller, -1 where the tap does not move. */
JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_phaseControlCheck(JNIEnv* env, jclass, jlong handle) {
    DcContext* ctx = ctxOf(handle);
    try {
        int m = ctx->nCtrl;
        jintArray result = env->NewIntArray(m);
        if (m == 0) {
            return result;
        }
        if (ctx->lfState.p == nullptr || !ctx->analyzed) {
            throw std::runtime_error("no solved state (solveTarget)");
        }
        int n = ctx->n;
        ctx->pcSens.ensure((size_t) n * m * sizeof(double));
        for (int col0 = 0; col0 < m; col0 += S_SOLVE_COLS) {
            int cols = std::min(S_SOLVE_COLS, m - col0);
            ctx->ensureRhs(cols);
            DC_CUDA_CHECK(cudaMemsetAsync(ctx->b.p, 0, (size_t) n * cols * sizeof(double), ctx->stream));
            phaseRhs<<<grid(cols), BLOCK, 0, ctx->stream>>>(n, cols, ctx->pcAlphaCol.as<int>() + col0, ctx->b.as<double>());
            DC_CUDA_CHECK(cudaGetLastError());
            ctx->solveDense(cols);
            DC_CUDA_CHECK(cudaMemcpy(ctx->pcSens.as<double>() + (size_t) col0 * n, ctx->x.p, (size_t) n * cols * sizeof(double), cudaMemcpyDeviceToDevice));
        }
        phaseDecision<<<grid(m), BLOCK, 0, ctx->stream>>>(n, m, ctx->lfState.as<double>(), ctx->pcSens.as<double>(), ctx->pcBranch.as<int>(),
                ctx->pcControlled.as<int>(), ctx->pcSide.as<int>(), ctx->pcTarget.as<double>(), ctx->pcHalfDb.as<double>(), ctx->pcTapIdx.as<int>(),
                ctx->pcTapPtr.as<int>(), ctx->pcTapA1.as<double>(), ctx->pcAllowed.as<int>(), ctx->pcDirCount.as<int>(), ctx->pcCurDir.as<int>(),
                ctx->ph1Row.as<int>(), ctx->ph2Row.as<int>(), ctx->a1Row.as<int>(), ctx->a1Const.as<double>(), ctx->power.as<double>(),
                ctx->pcNewTap.as<int>());
        DC_CUDA_CHECK(cudaGetLastError());
        DC_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<int> host(m);
        DC_CUDA_CHECK(cudaMemcpy(host.data(), ctx->pcNewTap.p, (size_t) m * sizeof(int), cudaMemcpyDeviceToHost));
        env->SetIntArrayRegion(result, 0, m, host.data());
        return result;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.phaseControlCheck: ") + e.what());
        return nullptr;
    }
}

/** The HVDC links in AC emulation: angle rows of both ends, droop k (per unit per radian), p0,
 *  the two Pmax, the status (0 linear, 1 saturated CS1->CS2, 2 saturated CS2->CS1, 3 frozen) and
 *  whether the loop looks at it (emulation on, both buses and the link enabled). */
JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_setHvdcs(JNIEnv* env, jclass, jlong handle, jintArray jPh1, jintArray jPh2, jdoubleArray jK,
                                                          jdoubleArray jP0, jdoubleArray jPmax12, jdoubleArray jPmax21, jintArray jStatus, jintArray jActive) {
    DcContext* ctx = ctxOf(handle);
    try {
        IntArray ph1(env, jPh1);
        IntArray ph2(env, jPh2);
        DoubleArray k(env, jK);
        DoubleArray p0(env, jP0);
        DoubleArray pmax12(env, jPmax12);
        DoubleArray pmax21(env, jPmax21);
        IntArray status(env, jStatus);
        IntArray active(env, jActive);
        int m = ph1.len;
        if (ph2.len != m || k.len != m || p0.len != m || pmax12.len != m || pmax21.len != m || status.len != m || active.len != m) {
            throw std::runtime_error("hvdc arrays must have the same length");
        }
        ctx->nHvdc = m;
        size_t ints = (size_t) std::max(m, 1) * sizeof(int);
        size_t dbls = (size_t) std::max(m, 1) * sizeof(double);
        ctx->hvPh1.upload(ph1.p, ints);
        ctx->hvPh2.upload(ph2.p, ints);
        ctx->hvK.upload(k.p, dbls);
        ctx->hvP0.upload(p0.p, dbls);
        ctx->hvPmax12.upload(pmax12.p, dbls);
        ctx->hvPmax21.upload(pmax21.p, dbls);
        ctx->hvStatus.upload(status.p, ints);
        ctx->hvActive.upload(active.p, ints);
        ctx->hvNew.ensure(ints);
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.setHvdcs: ") + e.what());
    }
}

/** DcHvdcAcEmulationLimitsOuterLoop.check on the device state: the new status per HVDC, -1 where it stays. */
JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_dc_gpu_GpuDcSolver_hvdcCheck(JNIEnv* env, jclass, jlong handle) {
    DcContext* ctx = ctxOf(handle);
    try {
        int m = ctx->nHvdc;
        jintArray result = env->NewIntArray(m);
        if (m == 0) {
            return result;
        }
        if (ctx->lfState.p == nullptr) {
            throw std::runtime_error("no solved state (solveTarget)");
        }
        hvdcDecision<<<grid(m), BLOCK, 0, ctx->stream>>>(m, ctx->lfState.as<double>(), ctx->hvPh1.as<int>(), ctx->hvPh2.as<int>(),
                ctx->hvK.as<double>(), ctx->hvP0.as<double>(), ctx->hvPmax12.as<double>(), ctx->hvPmax21.as<double>(),
                ctx->hvStatus.as<int>(), ctx->hvActive.as<int>(), ctx->hvNew.as<int>());
        DC_CUDA_CHECK(cudaGetLastError());
        DC_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<int> host(m);
        DC_CUDA_CHECK(cudaMemcpy(host.data(), ctx->hvNew.p, (size_t) m * sizeof(int), cudaMemcpyDeviceToHost));
        env->SetIntArrayRegion(result, 0, m, host.data());
        return result;
    } catch (const std::exception& e) {
        throwJava(env, std::string("GpuDcSolver.hvdcCheck: ") + e.what());
        return nullptr;
    }
}

}  // extern "C"
