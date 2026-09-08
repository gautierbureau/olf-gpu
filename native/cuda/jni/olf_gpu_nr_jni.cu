// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// FULL Newton-Raphson AC power flow on the GPU, behind ONE JNI call. Java hands over
// the network data (per-family element parameters + the equation/variable indices the
// scatter tables resolve against), the initial state and the targets; the native side
// then keeps everything resident on the device:
//
//   - the fixed-pattern CSR Jacobian structure is built ONCE on the host from the
//     generated scatter tables (closed-branch, open-branch, shunt, HVDC AC emulation
//     + the slack/PV identity rows) — cuDSS analyses it once;
//   - each iteration, the four generated fused kernels (evalClosedBranch /
//     evalOpenBranch / evalShunt / evalHvdc) fill the CSR values AND the mismatch F
//     directly in device memory, reading the device state vector through the
//     precomputed index arrays (no host repack);
//   - cuDSS REFACTORIZATION + SOLVE produce dx on the device, a kernel applies
//     x += dx, and only the n-double mismatch crosses PCIe for the convergence check.
//
// Orientation: J is true CSR with row = equation column, col = variable row (OLF's
// indices), so cuDSS solves J dx = -F directly — no transposition games (contrast
// powsybl-math's CSC-as-CSR convention, which exists to serve solveTransposed).
//
// The host-built destination array folds the scatter ROLE in, so the kernels never
// index the constexpr scatter tables:  dest >= 0 -> J values slot;  dest == -1 ->
// skip;  dest <= -2 -> mismatch row (-2 - dest).
//
// Java: com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver
//   static native double[] solve(double[] x0, double[] target, int maxIter, double tol,
//       double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
//       double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx, int[] idIdx)
//     x0/target: length n (state by variable row / target by equation column)
//     cbIn nbr*8  (y g1 g2 b1 b2 ksi r1 a1)   cbIdx nbr*12 (P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col qAlt1 qAlt2)
//     obIn nob*7  (y ksi g1 b1 g2 b2 r1)      obIdx nob*6 (P1 Q1 P2 Q2 v1 v2)
//     shIn nsh*2  (g b)                       shIdx nsh*3 (P Q v)
//     hvIn nhv*5  (p0 k lf1 lf2 r)            hvIdx nhv*4 (P1 P2 ph1 ph2)
//     idIdx nid*2 (eqRow varCol)              -1 = absent equation / variable
//     returns double[n + 2] = converged state, iterations, final ||F||inf
//
// This library links NVIDIA cuDSS DYNAMICALLY (never bundled, same licensing posture
// as powsybl-math-native's libmathcudss); cudart is linked statically.
#include <jni.h>
#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "closed_branch.cuh"
#include "closed_branch_scatter.h"
#include "open_branch.cuh"
#include "open_branch_scatter.h"
#include "shunt.cuh"
#include "shunt_scatter.h"
#include "hvdc.cuh"
#include "hvdc_scatter.h"
#include "vdl.cuh"

using namespace powsybl::openloadflow;

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_)); } } while (0)
#define CUDSS_CHECK(x) do { cudssStatus_t s_ = (x); if (s_ != CUDSS_STATUS_SUCCESS) { \
    throw std::runtime_error(std::string("cuDSS error, status ") + std::to_string((int) s_) \
        + " at " __FILE__ ":" + std::to_string(__LINE__) + " [" #x "]"); } } while (0)

// Batched N-1 solve path. BOTH implementations of the per-iteration linear solve are
// compiled in; the batch context picks one at RUNTIME from its `mode` field (set by the
// Java caller via -Dolf.gpu.batchMode, default 0):
//
//   mode 1 : the UNIFORM BATCHED cuDSS API (cudssMatrixCreateBatchCsr/Dn) — one ANALYSIS on
//     the shared pattern, then ONE batched FACTORIZATION + SOLVE over every scenario's value
//     set. Fastest where it fits; holds ~one factorization per scenario, so chunk it on small
//     GPUs (olf.gpu.sa.maxBatch).
//
//   mode 2 : BLOCK-DIAGONAL single matrix — assemble blkdiag(J_1..J_nS) as ONE (n*nS) CSR and
//     ANALYZE+FACTORIZE once, then REFACTORIZE every iteration (reuse pivots — a
//     refactor-every-iteration strategy). Measured ~parity with mode 1 (cuDSS refactorization ≈ factorization;
//     unlike CPU KLU it is not pivot-dominated). Same memory profile as mode 1 (chunk it).
//
//   mode 3 : UBATCH — a SINGLE per-block CSR with value/RHS pointers aimed at the batched buffers
//     + CUDSS_CONFIG_UBATCH_SIZE=nS. cuDSS analyzes ONE block (cheap: ~74 ms vs ~1.6 s for mode 1's
//     all-blocks analysis — a cheap single-block analysis). The CHUNK SIZE (olf.gpu.sa.maxBatch)
//     is decisive on cuDSS 0.8: its UBATCH phases carry a high per-launch fixed cost that
//     amortizes with batch size — measured on pegase9241 (500 ctg, K=4, RTX A1000, warm
//     ms/scenario): maxBatch=32 17.8, 125 6.35, 500 4.87 — the FASTEST configuration measured
//     on any mode/version, and 0.8's memory handling holds 500 blocks where 0.7.1's mode 1
//     ALLOC_FAILs. On cuDSS 0.7.x UBATCH is batch-size-INSENSITIVE instead (~8.3 at any chunk).
//     So: 0.8 + large chunks -> best known; 0.8 + small chunks (32) is a ~2x NET LOSS vs
//     mode 1 — an earlier note here calling 0.8's UBATCH "~4x slower than the batch API"
//     measured only that small-chunk regime. DEFAULT mode.
//
//   mode 4 : UBATCH + DirectIter0Only — same setup as mode 3 but FACTORIZE only at each chunk's
//     iter 0, SOLVE-only for the remaining iters (the Jacobian is frozen at the AC-hot-started
//     point, the mismatch/RHS is still recomputed every iter). Skips ~75% of the factorizations
//     at K=4 — the factorization is ~39% of GPU time. EXPERIMENTAL (convergence bet: a single-branch
//     N-1 is a small perturbation so the frozen J still converges from the warm start).
//
//   mode 5 : UBATCH + DirectRefactorEvery — same setup as mode 3 but FACTORIZE once (ever) then
//     REFACTORIZE after (reuse pivot order, recompute values). The UBATCH analog of mode 2;
//     EXPERIMENTAL probe of whether cuDSS refactorization is cheaper than factorization on the
//     UBATCH path (mode 2 showed ~parity on the block-diagonal path).
//
//   mode 0 (DEFAULT) : the per-scenario loop — repoint a single matrix and refactorize+solve
//     each scenario. Analysis is still amortized over the set; memory-light, scales to large
//     cases on small GPUs.
//
// cuDSS >= 0.7.1 is required. The cudssMatrixCreate{Csr,Dn,BatchCsr,BatchDn} data-type
// arguments differ between 0.7.x and 0.8 (0.8 added a separate `offsetType` and renamed the
// enum cudaDataType_t -> cudssDataType_t with CUDSS_R_* aliasing CUDA_R_*), so the type-arg
// groups are shimmed by CUDSS_VERSION below and the same source builds against either.
static_assert(CUDSS_VERSION >= 701,
              "libolfgpu requires cuDSS >= 0.7.1 (CUDSS_VERSION >= 701)");
#if CUDSS_VERSION >= 800
#  define OLF_CUDSS_CSR_TYPES  CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F   // offsetType, indexType, valueType
#  define OLF_CUDSS_DN_TYPE    CUDSS_R_64F                             // valueType
#  define OLF_CUDSS_BDN_TYPES  CUDSS_R_32I, CUDSS_R_64F                // integerType, valueType
#else
#  define OLF_CUDSS_CSR_TYPES  CUDA_R_32I, CUDA_R_64F                  // indexType, valueType (no offsetType)
#  define OLF_CUDSS_DN_TYPE    CUDA_R_64F
#  define OLF_CUDSS_BDN_TYPES  CUDA_R_32I, CUDA_R_64F
#endif

// Auto-sizing the batch chunk: query cuDSS's reported LU fill (CUDSS_DATA_LU_NNZ) via a
// cheap single-matrix probe analysis on the shared pattern, instead of guessing the fill
// with a heuristic multiplier. CUDSS_DATA_LU_NNZ has been in cuDSS since early versions;
// set -DOLF_GPU_CUDSS_LU_NNZ=0 if the installed headers lack the enum (the heuristic is
// then used). Even with it on, a runtime query failure falls back to the heuristic.
#ifndef OLF_GPU_CUDSS_LU_NNZ
#  define OLF_GPU_CUDSS_LU_NNZ 1
#endif

namespace {

// cuDSS iterative-refinement steps. Default 0: the OUTER Newton loop self-corrects any
// linear-solve inexactness, so refining each solve is redundant for NR — measured ~30% faster
// on batched N-1 with IDENTICAL convergence (vs the old IR=2 inherited from powsybl-math-native).
// Override per run with OLF_GPU_IR_STEPS=<n> (e.g. 2) if a stiff/near-singular Jacobian needs it.
// NB reusing pivots via CUDSS_PHASE_REFACTORIZATION was also benchmarked and gives NO speedup in
// cuDSS batched mode (its factorization cost is the numeric fill, not pivoting), so it is not used.
constexpr int IR_N_STEPS = 0;
constexpr int BLOCK = 64;

inline int irSteps() {
    const char* e = std::getenv("OLF_GPU_IR_STEPS");
    return e ? std::atoi(e) : IR_N_STEPS;
}

// ---------------------------------------------------------------------------------
// cuDSS fill-reducing ordering knob (the reordering-alg SWEEP): factorize cost is
// LU-fill-bound, and the fill is decided by the ordering cuDSS picks at ANALYSIS.
// OLF_GPU_REORDER_ALG selects it explicitly on EVERY cuDSS config we create (base
// context, batch context, chunk-size probe — one run = one ordering everywhere):
//   default | btf_colamd | colamd | amd | nd | none
// Unset on cuDSS >= 0.8.0 = AMD, the 2026-07-16 sweep winner on case9241pegase:
// LU_NNZ 260076 vs 304772 for the library default (which measured identical to nd,
// so the default IS nested dissection there), warm chunk -3.4% and warm SA -11% in
// every interleaved A/B pair, convergence and the full suite unchanged; btf_colamd
// and colamd are NOT_SUPPORTED (status 4) at the UBATCH ANALYSIS. Unset on 0.7.1 =
// library default (unmeasured there). "none" is the natural order — expect
// pathological fill on grid patterns; sweep it only with a small OLF_GPU_MAXBATCH.
// Driver: native/cuda/sweep_reorder.sh, which compares the reported LU_NNZ (see
// traceLuNnz) and the warm bench line per algorithm. cuDSS 0.7 names the same
// ordinals CUDSS_ALG_1..5 (its docs map 1/2 to BTF_COLAMD/COLAMD).
// ---------------------------------------------------------------------------------

#if CUDSS_VERSION >= 800
using OlfReorderAlg = cudssReorderingAlg_t;
constexpr OlfReorderAlg REORDER_ALGS[] = {CUDSS_REORDERING_ALG_DEFAULT, CUDSS_REORDERING_ALG_BTF_COLAMD,
                                          CUDSS_REORDERING_ALG_COLAMD, CUDSS_REORDERING_ALG_AMD,
                                          CUDSS_REORDERING_ALG_NESTED_DISSECTION, CUDSS_REORDERING_ALG_NONE};
#else
using OlfReorderAlg = cudssAlgType_t;
constexpr OlfReorderAlg REORDER_ALGS[] = {CUDSS_ALG_DEFAULT, CUDSS_ALG_1, CUDSS_ALG_2,
                                          CUDSS_ALG_3, CUDSS_ALG_4, CUDSS_ALG_5};
#endif

// ---------------------------------------------------------------------------------
// Matching (MC64-style) for the STANDALONE adjoint solve — NOT used by the batched NR.
//
// The batched NR path can skip this because it builds its own superset pattern with a
// GUARANTEED nonzero diagonal (the diagonal-pairing row permutation), so cuDSS never
// meets a zero pivot even though we configure no pivoting. A raw OLF Jacobian handed
// over unpermuted has no such guarantee: measured on ieee14, 5 of 28 diagonal entries
// are EXACTLY zero and 0 of 28 rows are diagonally dominant. cuDSS then substitutes a
// static perturbation for the zero pivot and returns a wrong answer WITH A MATCHING
// RESIDUAL — which reads as a correctness bug and is really a pivoting one. Measured
// on that matrix, error vs KLU: AMD 7.2e-3, nested dissection 7.2e-2, natural order
// 6.2e-15. Iterative refinement does NOT rescue it (IR=0 and IR=3 are bit-identical).
//
// Natural order is not a fix — fill explodes on grid patterns at real network sizes —
// so we keep the fill-reducing ordering and give cuDSS a matching pass to permute
// large entries onto the diagonal first. Override with OLF_GPU_MATCHING_ALG=<0..6>
// (0 disables; see cudssMatchingAlg_t), mainly to reproduce the failure above.
// ---------------------------------------------------------------------------------
inline void applyAdjointMatching(cudssConfig_t config) {
#if CUDSS_VERSION >= 800
    const char* e = std::getenv("OLF_GPU_MATCHING_ALG");
    // Treat an EMPTY value as unset, like applyReorderAlg: atoi("") is 0, which is
    // CUDSS_MATCHING_ALG_NONE, so `VAR= cmd` would silently disable matching instead of
    // selecting the default — a trap that cost a whole misread sweep row.
    const bool unset = e == nullptr || *e == '\0';
    cudssMatchingAlg_t alg = unset ? CUDSS_MATCHING_ALG_AUTO
                                   : static_cast<cudssMatchingAlg_t>(std::atoi(e));
    CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_MATCHING_ALG, &alg, sizeof(alg)));
#else
    (void) config;                                           // 0.7.1 has no matching knob
#endif
}

inline void applyReorderAlg(cudssConfig_t config) {
    const char* e = std::getenv("OLF_GPU_REORDER_ALG");
    if (e == nullptr || *e == '\0') {
#if CUDSS_VERSION >= 800
        OlfReorderAlg amd = CUDSS_REORDERING_ALG_AMD;        // sweep winner (see block comment)
        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &amd, sizeof(amd)));
#endif
        return;                                              // 0.7.1: library default, untouched
    }
    static const char* const NAMES[] = {"default", "btf_colamd", "colamd", "amd", "nd", "none"};
    for (size_t i = 0; i < sizeof(NAMES) / sizeof(NAMES[0]); i++) {
        if (std::string(e) == NAMES[i]) {
            OlfReorderAlg alg = REORDER_ALGS[i];
            CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)));
            return;
        }
    }
    throw std::runtime_error(std::string("OLF_GPU_REORDER_ALG: unknown value '") + e
                             + "' (default|btf_colamd|colamd|amd|nd|none)");
}

// Report the LU fill after an ANALYSIS when sweeping (or tracing): one stderr line the
// sweep driver greps. Non-fatal — some (version, ordering) combinations may not expose
// CUDSS_DATA_LU_NNZ; the sweep then judges on the bench timing alone.
inline void traceLuNnz(cudssHandle_t handle, cudssData_t data, const char* where) {
    if (std::getenv("OLF_GPU_REORDER_ALG") == nullptr && std::getenv("OLF_GPU_TRACE") == nullptr) {
        return;
    }
#if OLF_GPU_CUDSS_LU_NNZ
    int64_t luNnz = 0;
    size_t written = 0;
    if (cudssDataGet(handle, data, CUDSS_DATA_LU_NNZ, &luNnz, sizeof(luNnz), &written) == CUDSS_STATUS_SUCCESS
            && luNnz > 0) {
        const char* alg = std::getenv("OLF_GPU_REORDER_ALG");
#if CUDSS_VERSION >= 800
        const char* unsetName = "amd(default)";
#else
        const char* unsetName = "default";
#endif
        std::fprintf(stderr, "OLF_GPU_CUDSS reorder=%s %s LU_NNZ=%lld\n",
                     alg != nullptr ? alg : unsetName, where, static_cast<long long>(luNnz));
        return;
    }
#endif
    std::fprintf(stderr, "OLF_GPU_CUDSS %s LU_NNZ=unavailable\n", where);
}

// ---------------------------------------------------------------------------------
// Host: resolve each scatter-table row to (equation row, variable col) from one
// element's packed index array. Role 2 (skip) resolves to absent.
// ---------------------------------------------------------------------------------

struct RC { int row = -1; int col = -1; };

RC resolveCb(const CbScatter& s, const int* ix) {            // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    RC rc;
    if (s.role == 2) return rc;
    rc.row = ix[s.rowEnd * 2 + s.rowEq];
    // col = 4 + endpoint*2 + var: endpoint 0/1 = bus1/bus2 (PHI/V cols), endpoint 2 = the branch's
    // own control cols (ALPHA1 at ix[8], RHO1 at ix[9]). A control col is -1 when not derived, so
    // the entry resolves to absent and buildDest skips it for that element.
    if (s.role == 1) rc.col = ix[4 + s.colEnd * 2 + s.colVar];
    return rc;
}

RC resolveOb(const ObScatter& s, const int* ix) {            // P1 Q1 P2 Q2 v1 v2
    RC rc;
    if (s.role == 2) return rc;
    rc.row = ix[s.rowEnd * 2 + s.rowEq];
    if (s.role == 1) rc.col = ix[4 + s.colEnd];              // BUS_V columns only
    return rc;
}

RC resolveSh(const ShScatter& s, const int* ix) {            // P Q v bcol
    RC rc;
    if (s.role == 2) return rc;
    rc.row = ix[s.rowEq];                                    // single bus: rowEnd == 0
    // col endpoint 0 = the bus (BUS_V at ix[2]); endpoint 2 = the shunt's own control col (SHUNT_B at
    // ix[3]) — -1 when the shunt derives no voltage control, so dq/db resolves to absent and is skipped.
    if (s.role == 1) rc.col = (s.colEnd == 2) ? ix[3] : ix[2];
    return rc;
}

RC resolveHv(const HvdcScatter& s, const int* ix) {          // P1 P2 ph1 ph2
    RC rc;
    if (s.role == 2) return rc;
    rc.row = ix[s.rowEnd];                                   // P rows only: rowEq == 0
    if (s.role == 1) rc.col = ix[2 + s.colEnd];
    return rc;
}

// One pass per family: collect the J pattern entries (role 1, both indices present).
template <typename S, typename F>
void collectPattern(int ne, const int* idx, int stride, const S* table, int nout, F resolve,
                    std::vector<std::pair<int, int>>& pat) {
    for (int e = 0; e < ne; e++) {
        for (int k = 0; k < nout; k++) {
            RC rc = resolve(table[k], idx + e * stride);
            if (table[k].role == 1 && rc.row >= 0 && rc.col >= 0) {
                pat.emplace_back(rc.row, rc.col);
            }
        }
    }
}

// Lookup of a (row, col) entry's slot in the SORTED CSR pattern. The pattern is built by a
// std::sort of (row, col) pairs, so colIdx within each row is ascending — a binary search over
// colIdx[rowPtr[row] .. rowPtr[row+1]) resolves (row, col) -> slot with no extra memory (replaces
// a std::map<pair,int> tree, which was pure overhead on an already-sorted pattern).
struct CsrPattern {
    const int* rowPtr;
    const int* colIdx;
    int at(int row, int col) const {
        int lo = rowPtr[row];
        int hi = rowPtr[row + 1];
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (colIdx[mid] < col) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo >= rowPtr[row + 1] || colIdx[lo] != col) {
            throw std::runtime_error("CSR pattern lookup miss");
        }
        return lo;
    }
};

// Second pass: the per-(element, output) destination with the role folded in:
//   role 1 -> CSR values slot (>= 0);  role 0 -> -2 - F row;  absent/skip -> -1.
template <typename S, typename F>
std::vector<int> buildDest(int ne, const int* idx, int stride, const S* table, int nout,
                           F resolve, const CsrPattern& pos) {
    std::vector<int> dest(static_cast<size_t>(ne) * nout, -1);
    for (int e = 0; e < ne; e++) {
        for (int k = 0; k < nout; k++) {
            RC rc = resolve(table[k], idx + e * stride);
            if (table[k].role == 1 && rc.row >= 0 && rc.col >= 0) {
                dest[static_cast<size_t>(e) * nout + k] = pos.at(rc.row, rc.col);
            } else if (table[k].role == 0 && rc.row >= 0) {
                dest[static_cast<size_t>(e) * nout + k] = -2 - rc.row;
            }
        }
    }
    return dest;
}

// ---- Dynamic Q-relocation (production-design doc §11) ---------------------------------------
// A transformer/shunt voltage-controlled bus has two possible reactive-balance rows across the
// outer loop: its own BUS_V row (control OFF) and the free controller's BRANCH_RHO1 / SHUNT_B row
// (control ON). The extractor freezes ONE into cbIdx; the alternate travels at cbIdx[10+rowEnd]
// (packed by GpuAcFamilyPacker as qRowAlt, remapped through rowMap like every other row index).
// When enabled, the closed-branch/shunt fill routes each controlled-bus q output to whichever of the
// two candidate rows is FREE this scenario (row-mode != 0), so one shared batch structure follows a
// per-scenario control toggle instead of solving a subtly wrong system.
//
// Auto-enabled per structure: a structure needs dynamic Q-relocation exactly when some element carries
// an alternate Q row (a transformer/shunt voltage-controlled bus). qAlt lives at cbIdx[e*12+10/11] and
// shIdx[e*5+4], set to a real row (>= 0) only for a controllable bus, -1 otherwise. When no element has
// one, the union pattern is empty and every fill uses the frozen dest — so plain PF keeps the fast
// path and only voltage-control structures pay the (tiny) per-output selector cost.
static bool hasAltEntries(const int* a, int len) {
    for (int i = 0; i < len; i++) {
        if (a[i] != -1) {
            return true;
        }
    }
    return false;
}

static bool hasRelocatable(int ncb, const int* cbIdx, int nsh, const int* shIdx) {
    for (int e = 0; e < ncb; e++) {
        if (cbIdx[static_cast<size_t>(e) * 12 + 10] >= 0 || cbIdx[static_cast<size_t>(e) * 12 + 10] <= -2
                || cbIdx[static_cast<size_t>(e) * 12 + 11] >= 0 || cbIdx[static_cast<size_t>(e) * 12 + 11] <= -2) {
            return true;                                     // plain alt row, or the -2-groupId encoding (plan-A6)
        }
    }
    for (int e = 0; e < nsh; e++) {
        if (shIdx[static_cast<size_t>(e) * 5 + 4] >= 0 || shIdx[static_cast<size_t>(e) * 5 + 4] <= -2) {
            return true;
        }
    }
    return false;
}

// The alternate candidate Q row for a closed-branch output, or -1 if this output does not relocate
// (only Q rows, rowEq==1) or the endpoint bus has no second candidate (qRowAlt was -1).
inline int cbAltRow(const CbScatter& s, const int* ix) {
    return s.rowEq == 1 ? ix[10 + s.rowEnd] : -1;
}

// CONTINUOUS PHASE CONTROL (BRANCH_TARGET_P): when a phase-controller branch's control is enabled
// per scenario (row mode 3 on the CONTROLLER's a1-paired row), the controlled side's P-flow outputs
// are emitted AGAIN at that row — the p-flow equation replacing the a1 identity. Union those
// (a1Row, col) positions into the pattern; cbPcSide[e] = -1 none, else (controllerA1Col << 1) | side
// — the entry sits on the CONTROLLED element, whose own a1 may be absent (REMOTE control).
void collectCbPhasePattern(int ncb, const int* cbIdx, const int* cbPcSide,
                           const std::vector<int>& rowMap, std::vector<std::pair<int, int>>& pat) {
    for (int e = 0; e < ncb; e++) {
        if (cbPcSide[e] < 0) {
            continue;
        }
        int side = cbPcSide[e] & 1;
        int a1Col = cbPcSide[e] >> 1;
        const int* ix = cbIdx + static_cast<size_t>(e) * 12;
        int pcRow = rowMap[a1Col];
        int k0 = side == 0 ? 0 : 14;                         // p1 block / p2 block of CB_SCATTER
        for (int k = k0 + 1; k <= k0 + 6; k++) {             // the 6 derivative outputs
            RC rc = resolveCb(CB_SCATTER[k], ix);
            if (rc.col >= 0) {
                pat.emplace_back(pcRow, rc.col);
            }
        }
    }
}

// Per (closed branch, output): the phase-row destination (J slot, or -2-row for the P residual) and
// the a1 row whose per-scenario mode gates the emission (mode 3 = control enabled). -1 = no emission.
void buildCbPhaseDest(int ncb, const int* cbIdx, const int* cbPcSide, const std::vector<int>& rowMap,
                      const CsrPattern& pos, std::vector<int>& pcDest, std::vector<int>& pcSel) {
    pcDest.assign(static_cast<size_t>(ncb) * CB_NOUT, -1);
    pcSel.assign(static_cast<size_t>(ncb) * CB_NOUT, -1);
    for (int e = 0; e < ncb; e++) {
        if (cbPcSide[e] < 0) {
            continue;
        }
        int side = cbPcSide[e] & 1;
        int a1Col = cbPcSide[e] >> 1;
        const int* ix = cbIdx + static_cast<size_t>(e) * 12;
        int pcRow = rowMap[a1Col];
        int k0 = side == 0 ? 0 : 14;
        pcDest[static_cast<size_t>(e) * CB_NOUT + k0] = -2 - pcRow;    // the P residual
        pcSel[static_cast<size_t>(e) * CB_NOUT + k0] = a1Col;          // mode-3 gate lives at the CONTROLLER's a1 row
        for (int k = k0 + 1; k <= k0 + 6; k++) {
            RC rc = resolveCb(CB_SCATTER[k], ix);
            if (rc.col >= 0) {
                pcDest[static_cast<size_t>(e) * CB_NOUT + k] = pos.at(pcRow, rc.col);
                pcSel[static_cast<size_t>(e) * CB_NOUT + k] = a1Col;
            }
        }
    }
}

// Union the alternate Q rows into the CSR pattern: for each closed-branch role-1 Q output emit an
// extra (altRow, col) entry so buildDest can resolve a slot there. Deduped by the caller's unique().
void collectCbAltPattern(int ncb, const int* cbIdx, std::vector<std::pair<int, int>>& pat,
                         int nQr, const int* qrPtr, const int* qrRows) {
    for (int e = 0; e < ncb; e++) {
        const int* ix = cbIdx + static_cast<size_t>(e) * 12;
        for (int k = 0; k < CB_NOUT; k++) {
            const CbScatter& s = CB_SCATTER[k];
            if (s.role != 1) {
                continue;
            }
            RC rc = resolveCb(s, ix);
            int altRow = cbAltRow(s, ix);
            if (rc.col < 0) {
                continue;
            }
            if (altRow <= -2) {                              // GROUP relocation (plan-A6): union EVERY
                int g = -2 - altRow;                         // candidate row's slot for this column
                for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                    pat.emplace_back(qrRows[c], rc.col);
                }
            } else if (altRow >= 0 && altRow != rc.row) {
                pat.emplace_back(altRow, rc.col);
            }
        }
    }
}

// Per (closed branch, output): the alternate destination (CSR slot for a role-1 J entry, -2-row for
// a role-0 F residual) and the FROZEN row to test the per-scenario row-mode against. selRow == -1
// marks an output that never relocates (the fill then always uses the frozen dest). Sized ncb*CB_NOUT
// like the frozen dest, so the fill indexes all three arrays identically.
void buildCbDynDest(int ncb, const int* cbIdx, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow,
                    int nQr, const int* qrPtr, const int* qrRows,
                    std::vector<int>& relocGroup, std::vector<int>& relocOff, std::vector<int>& relocPool) {
    altDest.assign(static_cast<size_t>(ncb) * CB_NOUT, -1);
    selRow.assign(static_cast<size_t>(ncb) * CB_NOUT, -1);
    relocGroup.assign(static_cast<size_t>(ncb) * CB_NOUT, -1);
    relocOff.assign(static_cast<size_t>(ncb) * CB_NOUT, -1);
    for (int e = 0; e < ncb; e++) {
        const int* ix = cbIdx + static_cast<size_t>(e) * 12;
        for (int k = 0; k < CB_NOUT; k++) {
            const CbScatter& s = CB_SCATTER[k];
            RC rc = resolveCb(s, ix);
            int altRow = cbAltRow(s, ix);
            size_t o = static_cast<size_t>(e) * CB_NOUT + k;
            if (altRow <= -2) {                              // GROUP relocation (plan-A6): one slot per
                int g = -2 - altRow;                         // candidate, selected per scenario by the
                if (s.role == 1 && rc.col >= 0) {            // first-valid-member mode scan
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(pos.at(qrRows[c], rc.col));
                    }
                } else if (s.role == 0) {
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(-2 - qrRows[c]);
                    }
                }
                continue;
            }
            if (altRow < 0 || altRow == rc.row) {
                continue;                                    // not a relocatable output
            }
            if (s.role == 1 && rc.col >= 0) {
                altDest[o] = pos.at(altRow, rc.col);
                selRow[o] = rc.row >= 0 ? rc.row : altRow;   // suppressed frozen dest (remote member):
            } else if (s.role == 0) {                        // select on the ALT row itself (mode 4)
                altDest[o] = -2 - altRow;
                selRow[o] = rc.row >= 0 ? rc.row : altRow;
            }
        }
    }
}

// Shunt analogue of cbAltRow / collectCbAltPattern / buildCbDynDest. The alternate Q row of a
// voltage-controlling shunt (its own bus's SHUNT_B row <-> bus-V row) travels at shIdx[4]. A fixed
// shunt on a transformer/shunt-controlled bus also relocates (qRowAlt was set for that bus). This is
// the one-terminal case dynamic relocation exists for: SHUNT_B has no second terminal to hold its
// column, so a frozen structure freezes it at warm-start.
inline int shAltRow(const ShScatter& s, const int* ix) {
    return s.rowEq == 1 ? ix[4] : -1;
}

void collectShAltPattern(int nsh, const int* shIdx, std::vector<std::pair<int, int>>& pat,
                         int nQr, const int* qrPtr, const int* qrRows) {
    for (int e = 0; e < nsh; e++) {
        const int* ix = shIdx + static_cast<size_t>(e) * 5;
        for (int k = 0; k < SH_NOUT; k++) {
            const ShScatter& s = SH_SCATTER[k];
            if (s.role != 1) {
                continue;
            }
            RC rc = resolveSh(s, ix);
            int altRow = shAltRow(s, ix);
            if (rc.col < 0) {
                continue;
            }
            if (altRow <= -2) {
                int g = -2 - altRow;
                for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                    pat.emplace_back(qrRows[c], rc.col);
                }
            } else if (altRow >= 0 && altRow != rc.row) {
                pat.emplace_back(altRow, rc.col);
            }
        }
    }
}

void buildShDynDest(int nsh, const int* shIdx, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow,
                    int nQr, const int* qrPtr, const int* qrRows,
                    std::vector<int>& relocGroup, std::vector<int>& relocOff, std::vector<int>& relocPool) {
    altDest.assign(static_cast<size_t>(nsh) * SH_NOUT, -1);
    selRow.assign(static_cast<size_t>(nsh) * SH_NOUT, -1);
    relocGroup.assign(static_cast<size_t>(nsh) * SH_NOUT, -1);
    relocOff.assign(static_cast<size_t>(nsh) * SH_NOUT, -1);
    for (int e = 0; e < nsh; e++) {
        const int* ix = shIdx + static_cast<size_t>(e) * 5;
        for (int k = 0; k < SH_NOUT; k++) {
            const ShScatter& s = SH_SCATTER[k];
            RC rc = resolveSh(s, ix);
            int altRow = shAltRow(s, ix);
            size_t o = static_cast<size_t>(e) * SH_NOUT + k;
            if (altRow <= -2) {
                int g = -2 - altRow;
                if (s.role == 1 && rc.col >= 0) {
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(pos.at(qrRows[c], rc.col));
                    }
                } else if (s.role == 0) {
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(-2 - qrRows[c]);
                    }
                }
                continue;
            }
            if (altRow < 0 || altRow == rc.row) {
                continue;
            }
            if (s.role == 1 && rc.col >= 0) {
                altDest[o] = pos.at(altRow, rc.col);
                selRow[o] = rc.row >= 0 ? rc.row : altRow;   // suppressed frozen dest: select on ALT row
            } else if (s.role == 0) {
                altDest[o] = -2 - altRow;
                selRow[o] = rc.row >= 0 ? rc.row : altRow;
            }
        }
    }
}

// Open-branch analogue of shAltRow / collectShAltPattern / buildShDynDest: the ALTERNATE Q row
// of each side's bus travels in a SEPARATE obQAlt array (nob*2) rather than inside obIdx —
// the connected side's half-open charging q follows a pinned remote member's mode-4 row or a
// relocation group's carrier exactly like its closed-branch terms.
inline int obAltRow(const ObScatter& s, const int* qAlt, int e) {
    return s.rowEq == 1 ? qAlt[e * 2 + s.rowEnd] : -1;
}

void collectObAltPattern(int nob, const int* obIdx, const int* obQAlt, std::vector<std::pair<int, int>>& pat,
                         int nQr, const int* qrPtr, const int* qrRows) {
    for (int e = 0; e < nob; e++) {
        const int* ix = obIdx + static_cast<size_t>(e) * 6;
        for (int k = 0; k < OB_NOUT; k++) {
            const ObScatter& s = OB_SCATTER[k];
            if (s.role != 1) {
                continue;
            }
            RC rc = resolveOb(s, ix);
            int altRow = obAltRow(s, obQAlt, e);
            if (rc.col < 0) {
                continue;
            }
            if (altRow <= -2) {
                int g = -2 - altRow;
                for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                    pat.emplace_back(qrRows[c], rc.col);
                }
            } else if (altRow >= 0 && altRow != rc.row) {
                pat.emplace_back(altRow, rc.col);
            }
        }
    }
}

void buildObDynDest(int nob, const int* obIdx, const int* obQAlt, const CsrPattern& pos,
                    std::vector<int>& altDest, std::vector<int>& selRow,
                    int nQr, const int* qrPtr, const int* qrRows,
                    std::vector<int>& relocGroup, std::vector<int>& relocOff, std::vector<int>& relocPool) {
    altDest.assign(static_cast<size_t>(nob) * OB_NOUT, -1);
    selRow.assign(static_cast<size_t>(nob) * OB_NOUT, -1);
    relocGroup.assign(static_cast<size_t>(nob) * OB_NOUT, -1);
    relocOff.assign(static_cast<size_t>(nob) * OB_NOUT, -1);
    for (int e = 0; e < nob; e++) {
        const int* ix = obIdx + static_cast<size_t>(e) * 6;
        for (int k = 0; k < OB_NOUT; k++) {
            const ObScatter& s = OB_SCATTER[k];
            RC rc = resolveOb(s, ix);
            int altRow = obAltRow(s, obQAlt, e);
            size_t o = static_cast<size_t>(e) * OB_NOUT + k;
            if (altRow <= -2) {
                int g = -2 - altRow;
                if (s.role == 1 && rc.col >= 0) {
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(pos.at(qrRows[c], rc.col));
                    }
                } else if (s.role == 0) {
                    relocGroup[o] = g;
                    relocOff[o] = static_cast<int>(relocPool.size());
                    for (int c = qrPtr[g]; c < qrPtr[g + 1]; c++) {
                        relocPool.push_back(-2 - qrRows[c]);
                    }
                }
                continue;
            }
            if (altRow < 0 || altRow == rc.row) {
                continue;
            }
            if (s.role == 1 && rc.col >= 0) {
                altDest[o] = pos.at(altRow, rc.col);
                selRow[o] = rc.row >= 0 ? rc.row : altRow;   // suppressed frozen dest: select on ALT row
            } else if (s.role == 0) {
                altDest[o] = -2 - altRow;
                selRow[o] = rc.row >= 0 ? rc.row : altRow;
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Device: the four family fill kernels + the loop helpers
// ---------------------------------------------------------------------------------

__device__ void scatterAdd(int nout, const int* dest, const double* o, double* vals, double* F) {
    for (int k = 0; k < nout; k++) {
        int d = dest[k];
        if (d >= 0) {
            atomicAdd(&vals[d], o[k]);
        } else if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

// Dynamic Q-relocation scatter: when an output's frozen Q row is an identity this scenario
// (mode[selRow] == 0), route it to the alternate candidate row instead. mode is this scenario's
// row-mode slice (length n). dest/altDest/selRow share the per-output layout of scatterAdd.
// Mode 4 ("pinned remote gen-VC member" — the member row hosts its OWN Q balance, plan-B2) also
// selects the alternate: those entries carry a SUPPRESSED frozen dest and their own row as the
// alternate, with selRow = the member row itself. TVC/shunt select rows never carry mode 4.
// GROUP pilot-Q relocation tables (plan-A6), bundled to keep the fill signatures stable. All
// nullptr when the structure has no relocation groups.
struct RelocTables {
    const int* group = nullptr;                              // per output: group id, -1 = not grouped
    const int* off = nullptr;                                // per output: offset into pool
    const int* pool = nullptr;                               // one slot (or -2-row) per (output, candidate)
    const int* qrPtr = nullptr;                              // group -> candidate range
    const int* qrRows = nullptr;                             // candidate DEVICE rows ([0] = fallback)
};

// GROUP pilot-Q relocation (plan-A6): candidates = [fallback row, member rows in merged order];
// pick the FIRST member row whose per-scenario mode is neither identity (0: outaged/clamped/off)
// nor pinned (4) — exactly OLF's "first enabled" free-role rule — else the fallback (control off).
__device__ inline int relocChoose(int g, const int* qrPtr, const int* qrRows, const int* mode) {
    int p0 = qrPtr[g];
    int cnt = qrPtr[g + 1] - p0;
    for (int c = 1; c < cnt; c++) {
        int m = mode[qrRows[p0 + c]];
        if (m != 0 && m != 4) {
            return c;
        }
    }
    return 0;
}

// zi q-slot dynamic pick (F/J slots {0..3} per element): group -> mode-scan candidate slot,
// two-way -> alt when the select row is identity/pinned this scenario, else the frozen dest.
__device__ inline int ziDynDest(int frozen, int slot, const int* alt, const int* sel,
                                const int* mode, RelocTables rt) {
    int rg = rt.group != nullptr ? rt.group[slot] : -1;
    if (rg >= 0) {
        return rt.pool[rt.off[slot] + relocChoose(rg, rt.qrPtr, rt.qrRows, mode)];
    }
    int sr = sel != nullptr ? sel[slot] : -1;
    return (sr >= 0 && (mode[sr] == 0 || mode[sr] == 4)) ? alt[slot] : frozen;
}

__device__ void scatterAddDyn(int nout, const int* dest, const int* altDest, const int* selRow,
                              const int* mode, const double* o, double* vals, double* F,
                              RelocTables rt, int eOff) {
    for (int k = 0; k < nout; k++) {
        int d;
        int rg = rt.group != nullptr ? rt.group[eOff + k] : -1;
        if (rg >= 0) {
            d = rt.pool[rt.off[eOff + k] + relocChoose(rg, rt.qrPtr, rt.qrRows, mode)];
        } else {
            int sr = selRow[k];
            d = (sr >= 0 && (mode[sr] == 0 || mode[sr] == 4)) ? altDest[k] : dest[k];
        }
        if (d >= 0) {
            atomicAdd(&vals[d], o[k]);
        } else if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

__global__ void fillClosedBranch(int ne, const double* in, const int* ix, const int* dest,
                                 const double* x, double* vals, double* F,
                                 const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                 RelocTables rt) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne) return;
    const double* p = in + e * 8;                            // y g1 g2 b1 b2 ksi r1 a1
    const int* q = ix + e * 12;                              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    double v1 = x[q[5]], v2 = x[q[7]], ph1 = x[q[4]], ph2 = x[q[6]];
    // a controlling transformer's ratio/shift are STATE variables — read them from x at their
    // column; otherwise the fixed pi-model value from the input pack (q[8]/q[9] are -1 then).
    double r1 = q[9] >= 0 ? x[q[9]] : p[6];
    double a1 = q[8] >= 0 ? x[q[8]] : p[7];
    double o[CB_NOUT];
    evalClosedBranch(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    if (dynReloc) {
        scatterAddDyn(CB_NOUT, dest + e * CB_NOUT, altDest + e * CB_NOUT, selRow + e * CB_NOUT, mode, o, vals, F,
                      rt, e * CB_NOUT);
    } else {
        scatterAdd(CB_NOUT, dest + e * CB_NOUT, o, vals, F);
    }
}

__global__ void fillOpenBranch(int ne, const double* in, const int* ix, const int* dest,
                               const double* x, double* vals, double* F,
                               const int* altDest, const int* selRow, const int* mode, int dynReloc,
                               RelocTables rt) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne) return;
    const double* p = in + e * 7;                            // y ksi g1 b1 g2 b2 r1
    const int* q = ix + e * 6;
    double v1 = q[4] >= 0 ? x[q[4]] : 1.0;                   // absent side: value discarded
    double v2 = q[5] >= 0 ? x[q[5]] : 1.0;                   // (its outputs have dest -1)
    double o[OB_NOUT];
    evalOpenBranch(p[0], p[1], p[2], p[3], p[4], p[5], v1, p[6], v2, 0,
                   o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7);
    if (dynReloc) {
        scatterAddDyn(OB_NOUT, dest + e * OB_NOUT, altDest + e * OB_NOUT, selRow + e * OB_NOUT, mode, o, vals, F,
                      rt, e * OB_NOUT);
    } else {
        scatterAdd(OB_NOUT, dest + e * OB_NOUT, o, vals, F);
    }
}

__global__ void fillVdlBatch(int ne, int nScenarios, int n, int nnz,
                             const double* in, const int* ix, const int* dest,
                             const double* x, double* vals, double* F);

__global__ void fillShunt(int ne, const double* in, const int* ix, const int* dest,
                          const double* x, double* vals, double* F,
                          const int* altDest, const int* selRow, const int* mode, int dynReloc,
                          RelocTables rt) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne) return;
    const double* p = in + e * 2;                            // g b
    // b is the SHUNT_B state variable when the shunt derives voltage control (bcol = ix[e*5+3] >= 0),
    // else the fixed packed susceptance — exactly like a closed branch reads r1 from x[r1col] or cbIn.
    double b = ix[e * 5 + 3] >= 0 ? x[ix[e * 5 + 3]] : p[1];
    double v = x[ix[e * 5 + 2]];
    double o[SH_NOUT];
    evalShunt(v, p[0], b, 0, o + 0, o + 1, o + 2, o + 3, o + 4);
    if (dynReloc) {
        scatterAddDyn(SH_NOUT, dest + e * SH_NOUT, altDest + e * SH_NOUT, selRow + e * SH_NOUT, mode, o, vals, F,
                      rt, e * SH_NOUT);
    } else {
        scatterAdd(SH_NOUT, dest + e * SH_NOUT, o, vals, F);
    }
}

__global__ void fillHvdc(int ne, const double* in, const int* ix, const int* dest,
                         const double* x, double* vals, double* F) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne) return;
    const double* p = in + e * 5;                            // p0 k lf1 lf2 r
    const int* q = ix + e * 4;
    double ph1 = x[q[2]], ph2 = x[q[3]];
    double o[HVDC_NOUT];
    evalHvdc(p[0], p[1], p[2], p[3], p[4], ph1, ph2, 0, o + 0, o + 1, o + 2, o + 3, o + 4, o + 5);
    scatterAdd(HVDC_NOUT, dest + e * HVDC_NOUT, o, vals, F);
}

// DISTR_Q (remote generator voltage control) reactive-distribution rows: one thread per weighted
// contribution. Each is a closed branch's side-s reactive flow (q1 or q2, value + the 5 derivatives
// dq/d{v1,v2,φ1,φ2,r1}) re-evaluated from the SAME closed-branch params/state and scattered, scaled
// by the controller weight, into a DISTR_Q row. dest[6] = {F-row, the 5 J slots} (built like buildDest;
// dq/dr1 slot is -1 when the branch derives no ratio). The DISTR_Q row is a power row (mode 1), so
// applyRowMode leaves it; the controllers' own per-bus Q scatter was suppressed (qRow=-1) on the host.
__global__ void fillDistrQ(int ndq, const int* dqElem, const int* dqSide, const int* dqKind,
                           const double* dqWeight, const int* dqDest,
                           const double* cbIn, const int* cbIx, const double* shIn, const int* shIx,
                           const double* obIn, const int* obIx,
                           const double* x, double* vals, double* F, const int* mode) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ndq) return;
    int e = dqElem[c];
    double w = dqWeight[c];
    const int* dst = dqDest + c * 6;
    // an RL-PINNED remote member's row (mode 4) hosts its OWN Q balance — its DISTR_Q emissions
    // stop (the CPU deactivates only that member's DISTR_Q equation; the others keep its q terms)
    if (mode != nullptr && dst[0] <= -2 && mode[-2 - dst[0]] == 4) return;
    if (dqKind[c] == 1) {                                    // shunt q: {q value -> F, dq/dv -> J(row, vcol)}
        const double* sp = shIn + e * 2;                     // g b (a DISTR_Q shunt contributor is always fixed)
        int vcol = shIx[e * 5 + 2];
        double so[SH_NOUT];
        evalShunt(x[vcol], sp[0], sp[1], 0, so + 0, so + 1, so + 2, so + 3, so + 4);
        if (dst[0] <= -2) atomicAdd(&F[-2 - dst[0]], w * so[2]);    // q
        if (dst[1] >= 0) atomicAdd(&vals[dst[1]], w * so[3]);       // dq/dv
        return;
    }
    if (dqKind[c] == 3) {                                    // zero-impedance DUMMY_Q: linear w·dummyQ term
        int col = e;                                         // dqElem holds the DUMMY_Q variable column directly
        if (dst[0] <= -2) atomicAdd(&F[-2 - dst[0]], w * x[col]);
        if (dst[1] >= 0) atomicAdd(&vals[dst[1]], w);        // d(w·dummyQ)/d(dummyQ) = w
        return;
    }
    if (dqKind[c] == 2) {                                    // half-open branch: connected-side q + dq/dv
        const double* op = obIn + e * 7;                     // y ksi g1 b1 g2 b2 r1
        const int* oq = obIx + e * 6;                        // P1 Q1 P2 Q2 v1col v2col
        int vcol = dqSide[c] == 0 ? oq[4] : oq[5];           // the controller's (connected) side V column
        double v1 = dqSide[c] == 0 ? x[vcol] : 1.0;
        double v2 = dqSide[c] == 1 ? x[vcol] : 1.0;          // absent side: value discarded
        double oo[OB_NOUT];
        evalOpenBranch(op[0], op[1], op[2], op[3], op[4], op[5], v1, op[6], v2, 0,
                       oo + 0, oo + 1, oo + 2, oo + 3, oo + 4, oo + 5, oo + 6, oo + 7);
        int qi = dqSide[c] == 0 ? 6 : 2;                     // side-1 q1[6]/dq1dv1[7], side-2 q2[2]/dq2dv2[3]
        if (dst[0] <= -2) atomicAdd(&F[-2 - dst[0]], w * oo[qi]);
        if (dst[1] >= 0) atomicAdd(&vals[dst[1]], w * oo[qi + 1]);
        return;
    }
    const double* p = cbIn + e * 8;                          // y g1 g2 b1 b2 ksi r1 a1
    const int* q = cbIx + e * 12;                            // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    double v1 = x[q[5]], v2 = x[q[7]], ph1 = x[q[4]], ph2 = x[q[6]];
    double r1 = q[9] >= 0 ? x[q[9]] : p[6];
    double a1 = q[8] >= 0 ? x[q[8]] : p[7];
    double o[CB_NOUT];
    evalClosedBranch(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    // side 1 q outputs: q1[7] dq1dv1[8] dq1dv2[9] dq1dph1[10] dq1dph2[11] dq1dr1[13];  side 2: +14 (dr1 +14)
    int qi = dqSide[c] == 0 ? 7 : 21;
    double out6[6] = {o[qi], o[qi + 1], o[qi + 2], o[qi + 3], o[qi + 4], o[qi + 6]};
    for (int m = 0; m < 6; m++) {
        int d = dst[m];
        double val = w * out6[m];
        if (d >= 0) {
            atomicAdd(&vals[d], val);
        } else if (d <= -2) {
            atomicAdd(&F[-2 - d], val);
        }
    }
}

// Scatter one (dest, value): dest >= 0 -> J slot, dest <= -2 -> F row, -1 -> skip.
__device__ inline void addAt(int dest, double val, double* vals, double* F) {
    if (dest >= 0) {
        atomicAdd(&vals[dest], val);
    } else if (dest <= -2) {
        atomicAdd(&F[-2 - dest], val);
    }
}

// Zero-impedance branch (bus coupler): a LINEAR family. ix (NOT row-mapped — it carries the variable
// COLUMNS the kernel reads; the mapped rows live in dest): dpCol dqCol v1 v2 ph1 ph2 tree at ix[4..10].
// dest[14] (built host-side through the CSR pattern): the dummy ±1 couplings into both buses' P/Q
// balances [0..7], then, for a spanning-tree edge, ZERO_V (v1−rho·v2) on the DUMMY_P row and ZERO_PHI
// (ph1−ph2) on the DUMMY_Q row [8..13] (−1/skip on a non-tree edge, whose dummy rows are mode-0 identities
// the row-mode mask handles). The Jacobian entries are constants; only F depends on the state.
__global__ void fillZeroImp(int nzi, const double* in, const int* ix, const int* dest,
                            const double* x, double* vals, double* F,
                            const int* ziAlt, const int* ziSel, const int* mode, int ziDyn,
                            RelocTables rt) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nzi) return;
    double rho = in[e];
    const int* q = ix + e * 11;
    double dummyP = x[q[4]], dummyQ = x[q[5]];
    const int* d = dest + e * 14;
    int d2 = d[2], d3 = d[3], d6 = d[6], d7 = d[7];
    if (ziDyn) {                                             // dynamic Q-routing (ob/zi channels)
        d2 = ziDynDest(d2, e * 4 + 0, ziAlt, ziSel, mode, rt);
        d3 = ziDynDest(d3, e * 4 + 1, ziAlt, ziSel, mode, rt);
        d6 = ziDynDest(d6, e * 4 + 2, ziAlt, ziSel, mode, rt);
        d7 = ziDynDest(d7, e * 4 + 3, ziAlt, ziSel, mode, rt);
    }
    addAt(d[0], dummyP, vals, F);                            // F[bus1 P] += dummyP
    addAt(d[1], -dummyP, vals, F);                           // F[bus2 P] -= dummyP
    addAt(d2, dummyQ, vals, F);                              // F[bus1 Q] += dummyQ
    addAt(d3, -dummyQ, vals, F);                             // F[bus2 Q] -= dummyQ
    addAt(d[4], 1.0, vals, F);                               // J(bus1 P, dummyP)
    addAt(d[5], -1.0, vals, F);                              // J(bus2 P, dummyP)
    addAt(d6, 1.0, vals, F);                                 // J(bus1 Q, dummyQ)
    addAt(d7, -1.0, vals, F);                                // J(bus2 Q, dummyQ)
    if (q[10]) {                                             // spanning-tree edge: voltage/angle coupling
        double v1 = x[q[6]], v2 = x[q[7]], ph1 = x[q[8]], ph2 = x[q[9]];
        addAt(d[8], v1 - rho * v2, vals, F);                 // F[ZERO_V row] += v1 - rho*v2
        addAt(d[9], 1.0, vals, F);                           // J(ZERO_V, v1)
        addAt(d[10], -rho, vals, F);                         // J(ZERO_V, v2)
        addAt(d[11], ph1 - ph2, vals, F);                    // F[ZERO_PHI row] += ph1 - ph2
        addAt(d[12], 1.0, vals, F);                          // J(ZERO_PHI, ph1)
        addAt(d[13], -1.0, vals, F);                         // J(ZERO_PHI, ph2)
    }
}

// DISTR_RHO (transformer ratio distribution): one thread per linear contribution. A controller's
// DISTR_RHO row = (1/count)·Σ r1 − r1_k, so each is coef·r1_j into the row's F + a constant coef in
// J(row, r1_j). dest[2] = {F row, J slot} (built host-side through the CSR pattern).
__global__ void fillDistrRho(int ndr, const int* drCol, const double* drCoef, const int* drDest,
                             const double* x, double* vals, double* F,
                             const int* drQrGroup, const int* mode, const int* qrPtr, const int* qrRows) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ndr) return;
    // the CURRENT pilot-Q carrier's row (the free role re-picked to this member after a clamp)
    // hosts the relocated pilot balance — its DISTR_RHO row is inactive (the clamp-in-the-BASE
    // sequential failure: the leftover emissions solved a MIXED equation 7.4e-3 off the CPU's)
    if (drQrGroup != nullptr && drQrGroup[c] >= 0) {
        int g = drQrGroup[c];
        if (-2 - drDest[c * 2] == qrRows[qrPtr[g] + relocChoose(g, qrPtr, qrRows, mode)]) return;
    }
    double coef = drCoef[c];
    addAt(drDest[c * 2 + 0], coef * x[drCol[c]], vals, F);   // F[DISTR_RHO row] += coef * r1
    addAt(drDest[c * 2 + 1], coef, vals, F);                 // J(DISTR_RHO row, r1) += coef
}

// DISTR_SHUNT_B (shunt susceptance distribution): the SHUNT_B analogue of fillDistrRho. A controller's
// DISTR_SHUNT_B row = (1/count)·Σ b − b_k, so each contribution is coef·b_j into the row's F + a constant
// coef in J(row, b_j). dest[2] = {F row, J slot} (built host-side through the CSR pattern).
__global__ void fillDistrShuntB(int ndb, const int* dbCol, const double* dbCoef, const int* dbDest,
                                const double* x, double* vals, double* F) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ndb) return;
    double coef = dbCoef[c];
    addAt(dbDest[c * 2 + 0], coef * x[dbCol[c]], vals, F);   // F[DISTR_SHUNT_B row] += coef * b
    addAt(dbDest[c * 2 + 1], coef, vals, F);                 // J(DISTR_SHUNT_B row, b) += coef
}

// Slack/PV target identity rows, applied as a PER-SOLVE mask over the fixed superset
// pattern: a row in identity mode (mode 0) is zeroed, its diagonal set to 1, and its
// mismatch overwritten with x - target. The fill kernels may have scattered into it —
// this runs after them and overwrites, which is what keeps the structure (and cuDSS's
// analysis) immutable across PV->PQ activation switching.
__global__ void applyRowMode(int n, const int* mode, const int* rowPtr, const int* diagPos,
                             const double* x, const double* target, double* vals, double* F) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n || mode[r] != 0) return;
    for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
        vals[k] = 0.0;
    }
    vals[diagPos[r]] = 1.0;
    F[r] = x[r] - target[r];                                 // row r is variable r's row
}

__global__ void initF(int n, const double* target, double* F) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) F[i] = -target[i];                            // F = eval - target
}

__global__ void negateInto(int n, const double* F, double* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = -F[i];                                 // rhs of J dx = -F
}

__global__ void axpy1(int n, const double* dx, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += dx[i];
}

// Line-search backtrack: x = xPrev + mu*dx (mu in (0,1]; mu==1 reproduces the full step).
__global__ void stepFromPrev(int n, const double* xPrev, const double* dx, double mu, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = xPrev[i] + mu * dx[i];
}

// Per-scenario masked backtrack step: x = xPrev + mu[s]*dx for scenarios with mu[s] >= 0;
// mu[s] < 0 leaves the scenario's state untouched (not backtracking this pass).
__global__ void stepFromPrevBatchMasked(int n, int total, const double* xPrev, const double* dx,
                                        const double* mu, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        double m = mu[i / n];
        if (m >= 0.0) x[i] = xPrev[i] + m * dx[i];
    }
}

inline int grid(int n) { return (n + BLOCK - 1) / BLOCK; }

// ---------------------------------------------------------------------------------
// The solver context: every device buffer + the cuDSS objects, freed by destructor.
// ---------------------------------------------------------------------------------

struct DeviceBuf {
    void* p = nullptr;
    ~DeviceBuf() { if (p) cudaFree(p); }
    template <typename T> T* as() { return static_cast<T*>(p); }
    void alloc(size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
    void upload(const void* host, size_t bytes) {
        alloc(bytes);
        CUDA_CHECK(cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice));
    }
};

struct CudssCtx {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matA = nullptr, matB = nullptr, matX = nullptr;
    ~CudssCtx() {
        if (matA) cudssMatrixDestroy(matA);
        if (matB) cudssMatrixDestroy(matB);
        if (matX) cudssMatrixDestroy(matX);
        if (data && handle) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
    }
};

struct Family {
    int ne = 0, instride = 0, ixstride = 0, nout = 0;
    DeviceBuf in, ix, dest;
    DeviceBuf altDest, selRow;                               // Dynamic Q-relocation (cb only); empty when disabled
    DeviceBuf relocGroup, relocOff, relocPool;               // GROUP relocation (plan-A6): per-output group id +
    bool dynReloc = false;                                   // slot-pool offset; pool = one slot per candidate
    DeviceBuf pcDest, pcSel;                                 // Continuous phase control (cb only): P-flow row at a1
    bool phaseCtl = false;                                   // pcDest/pcSel populated (mode-3 gated emission)
};

// Build one family's device-side data (inputs, indices, role-folded destinations).
template <typename S, typename R>
void setupFamily(Family& f, int ne, const double* in, int instride, const int* idx,
                 int ixstride, const S* table, int nout, R resolve,
                 const CsrPattern& pos) {
    f.ne = ne; f.instride = instride; f.ixstride = ixstride; f.nout = nout;
    if (ne == 0) return;
    f.in.upload(in, static_cast<size_t>(ne) * instride * sizeof(double));
    f.ix.upload(idx, static_cast<size_t>(ne) * ixstride * sizeof(int));
    std::vector<int> dest = buildDest(ne, idx, ixstride, table, nout, resolve, pos);
    f.dest.upload(dest.data(), dest.size() * sizeof(int));
}

struct Result {
    std::vector<double> x;
    int iters = 0;
    double norm = 0.0;
};

// Row permutation: pair each equation row with its dominant-diagonal variable column —
// P rows with their own bus's PHI column (dP/dθ self term), Q rows with the bus's V
// column (dQ/dV), slack/PV identities with their column — and USE THE PAIRED COLUMN as
// the row index. The CSR diagonal is then structurally strong regardless of OLF's
// equation numbering: the plain creator happens to interleave P/Q/V per bus (fine), but
// the VECTORIZED creator numbers array equations in per-type blocks, whose leading
// minors defeat cuDSS's static pivoting (KLU pivots dynamically and never noticed).
// Choosing the ordering ourselves — a bus-ordered J — fixes any
// numbering. The mismatch/targets are permuted the same way; columns are untouched, so
// the solution dx needs no back-permutation.
std::vector<int> buildRowMap(int n, int ncb, const int* cbIdx, int nob, const int* obIdx,
                             int nsh, const int* shIdx, int nhv, const int* hvIdx,
                             int ndq, const int* dqRow, int nzi, const int* ziIdx) {
    std::vector<int> rowMap(n, -1);
    auto pairRc = [&](int row, int col) {
        if (row < 0 || col < 0) return;
        if (rowMap[row] == -1) {
            rowMap[row] = col;
        } else if (rowMap[row] != col) {
            throw std::runtime_error("conflicting diagonal pairing for equation row " + std::to_string(row));
        }
    };
    // A bus whose reactive balance was ROUTED ELSEWHERE (Qrow != its own V col) leaves that V col
    // carrying the v−target identity (PV / transformer / generator-pilot) or another power row that
    // no Q self-pair claims — claim it here. For an ordinary bus Qrow == V col, so this is a no-op.
    auto claimMovedV = [&](int qrow, int vcol) {
        if (vcol >= 0 && qrow != vcol) {
            pairRc(vcol, vcol);
        }
    };
    for (int e = 0; e < ncb; e++) {                          // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
        const int* ix = cbIdx + e * 12;
        pairRc(ix[0], ix[4]);                                // P1 -> ph1
        pairRc(ix[2], ix[6]);                                // P2 -> ph2
        // Q rows pair with their own stable index — its V col on an ordinary bus (Qrow==v col), the
        // controlling transformer's RHO col or a controller's DISTR_Q row on a voltage-controlled bus
        // (the extractor set Qrow to it). So the strong diagonal is dq/dv, dq/dr1 or the DISTR_Q diag.
        pairRc(ix[1], ix[1]);
        pairRc(ix[3], ix[3]);
        claimMovedV(ix[1], ix[5]);                           // side 1 moved-V identity
        claimMovedV(ix[3], ix[7]);                           // side 2 moved-V identity
        if (ix[9] >= 0) {
            // claim the rho row: its diagonal is the rho col either way — dq/dr1 when the ratio is a
            // free control, or the unit derivative of the r1−target identity when it is held constant.
            pairRc(ix[9], ix[9]);
        }
        if (ix[8] >= 0) {
            // Claim the a1 row (phase control). Unlike rho — whose row IS a bus Q row relocated onto the
            // rho column, so the dq/dr1 scatter lands on its diagonal — NOTHING scatters into the a1 row:
            // dp1da1/dq1da1/dp2da1/dq2da1 all go to the BUS P/Q rows against the a1 COLUMN. The a1 row is
            // therefore an identity row only (a1−target, INCREMENTAL mode holding a1 at its tap value),
            // and its 1.0 comes from the per-solve row-mode mask. The (r,r) entry is already guaranteed
            // in the pattern below; all that is missing is this claim, without which rowMap has no
            // pairing for the row and buildStructure rejects the system.
            pairRc(ix[8], ix[8]);
        }
    }
    for (int e = 0; e < nob; e++) {                          // P1 Q1 P2 Q2 v1 v2 (no ph cols:
        const int* ix = obIdx + e * 6;                       // P rows pair via closed branches)
        pairRc(ix[1], ix[1]);                                // Q rows self-pair (v col, or rho col if controlled)
        pairRc(ix[3], ix[3]);
        claimMovedV(ix[1], ix[4]);
        claimMovedV(ix[3], ix[5]);
    }
    for (int e = 0; e < nsh; e++) {                          // P Q v bcol
        pairRc(shIdx[e * 5 + 1], shIdx[e * 5 + 1]);          // Q self-pairs (v col, or SHUNT_B col if controlled)
        claimMovedV(shIdx[e * 5 + 1], shIdx[e * 5 + 2]);
        if (shIdx[e * 5 + 3] >= 0) {
            // claim the SHUNT_B row: its diagonal is the b col either way — dq/db when the susceptance is a
            // free control, or the unit derivative of the b−target identity when it is held constant.
            pairRc(shIdx[e * 5 + 3], shIdx[e * 5 + 3]);
        }
    }
    for (int e = 0; e < nhv; e++) {                          // P1 P2 ph1 ph2
        const int* ix = hvIdx + e * 4;
        pairRc(ix[0], ix[2]);
        pairRc(ix[1], ix[3]);
    }
    // DISTR_Q reactive-distribution rows sit on a controller's BUS_V row and self-pair: their strong
    // diagonal is that controller's own weighted dq/dV (qPct_k·∂q_Gk/∂V_Gk − ∂q_Gk/∂V_Gk). Each
    // distinct dqRow appears for several contributions; pairRc is idempotent so re-pairing is fine.
    for (int c = 0; c < ndq; c++) {
        pairRc(dqRow[c], dqRow[c]);
    }
    // Zero-impedance branch: claim its DUMMY_P / DUMMY_Q rows (ZERO_V/ZERO_PHI or DUMMY_TARGET identities)
    // and, defensively, its bus P/Q rows (a coupler-only bus has no closed/open branch to claim them).
    for (int e = 0; e < nzi; e++) {
        const int* ix = ziIdx + e * 11;                      // b1P b1Q b2P b2Q dpRow dqRow ...
        for (int s = 0; s < 6; s++) {
            pairRc(ix[s], ix[s]);
        }
    }
    std::vector<char> colUsed(n, 0);
    for (int r = 0; r < n; r++) {
        if (rowMap[r] < 0) {                                 // every index in [0,n) is an active equation
            throw std::runtime_error("equation row " + std::to_string(r) + " has no diagonal pairing");
        }
        if (colUsed[rowMap[r]]) {
            throw std::runtime_error("diagonal pairing is not a bijection (column "
                                     + std::to_string(rowMap[r]) + " claimed twice)");
        }
        colUsed[rowMap[r]] = 1;
    }
    return rowMap;
}

// The DISTR_Q / DISTR_RHO / zero-impedance family contributions to the CSR pattern (the rows they add
// variable columns to). Shared by buildStructure for both solver contexts, so every consumer sees the
// SAME nnz the real context builds. cbIdx/obIdx/shIdx are the row-mapped packs; only the family ROWS are
// taken through rowMap here, the contributor columns (v/ph/dummy/r1) are variable columns left as-is.
void collectFamilyPattern(const std::vector<int>& rowMap,
                          const int* cbIdx, const int* obIdx, const int* shIdx,
                          int ndq, const int* dqElem, const int* dqSide, const int* dqKind, const int* dqRow,
                          int ndr, const int* drRow, const int* drCol,
                          int ndb, const int* dbRow, const int* dbCol,
                          int nzi, const int* ziIdx,
                          std::vector<std::pair<int, int>>& pat) {
    for (int c = 0; c < ndq; c++) {
        int mappedRow = rowMap[dqRow[c]];
        if (dqKind[c] == 1) {                                // shunt q: bus V column
            pat.emplace_back(mappedRow, shIdx[static_cast<size_t>(dqElem[c]) * 5 + 2]);
        } else if (dqKind[c] == 2) {                         // half-open: connected-side V column
            const int* oq = obIdx + static_cast<size_t>(dqElem[c]) * 6;
            pat.emplace_back(mappedRow, dqSide[c] == 0 ? oq[4] : oq[5]);
        } else if (dqKind[c] == 3) {                         // zero-imp DUMMY_Q column
            pat.emplace_back(mappedRow, dqElem[c]);
        } else {                                             // closed branch: v1 v2 ph1 ph2 r1
            const int* q = cbIdx + static_cast<size_t>(dqElem[c]) * 12;
            pat.emplace_back(mappedRow, q[5]);
            pat.emplace_back(mappedRow, q[7]);
            pat.emplace_back(mappedRow, q[4]);
            pat.emplace_back(mappedRow, q[6]);
            if (q[9] >= 0) pat.emplace_back(mappedRow, q[9]);
        }
    }
    for (int c = 0; c < ndr; c++) {                          // DISTR_RHO: controller r1 column in a DISTR_RHO row
        pat.emplace_back(rowMap[drRow[c]], drCol[c]);
    }
    for (int c = 0; c < ndb; c++) {                          // DISTR_SHUNT_B: controller b column in a DISTR_SHUNT_B row
        pat.emplace_back(rowMap[dbRow[c]], dbCol[c]);
    }
    for (int e = 0; e < nzi; e++) {                          // zero-impedance: dummy couplings + (tree) ZERO_V/PHI
        const int* z = ziIdx + static_cast<size_t>(e) * 11;
        int b1P = rowMap[z[0]], b2P = rowMap[z[2]];
        int b1Q = z[1] >= 0 ? rowMap[z[1]] : -1, b2Q = z[3] >= 0 ? rowMap[z[3]] : -1;
        pat.emplace_back(b1P, z[4]);
        pat.emplace_back(b2P, z[4]);
        if (b1Q >= 0) pat.emplace_back(b1Q, z[5]);
        if (b2Q >= 0) pat.emplace_back(b2Q, z[5]);
        if (z[10]) {
            int dpRow = rowMap[z[4]], dqRow2 = rowMap[z[5]];
            pat.emplace_back(dpRow, z[6]);
            pat.emplace_back(dpRow, z[7]);
            pat.emplace_back(dqRow2, z[8]);
            pat.emplace_back(dqRow2, z[9]);
        }
    }
}

// Per closed-branch STATIC trig (OLF's own CPU optimization, on device): ksi and the
// uncontrolled a1 are constant per element, so sin/cos(ksi) and sin/cos(ksi -/+ a1 -/+ A2)
// are computed ONCE per element at structure build; the batched F-only fill derives
// sinTheta1/2, cosTheta1/2 from ONE dynamic sincos(dph) via angle addition (see the
// closedBranchTrig helper next to the batched fill kernels).
__global__ void initClosedBranchTrig(int ne, const double* in, double* trig) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne) return;
    const double* p = in + e * 8;                            // y g1 g2 b1 b2 ksi r1 a1
    double ksi = p[5], a1 = p[7];
    double* t = trig + e * 6;
    sincos(ksi, t + 0, t + 1);                               // sinKsi, cosKsi
    sincos(ksi - a1 + A2, t + 2, t + 3);                     // sinC1, cosC1
    sincos(ksi + a1 - A2, t + 4, t + 5);                     // sinC2, cosC2
}

// ---------------------------------------------------------------------------------
// The SHARED STRUCTURE: everything the two solver contexts derive from ONE extraction
// (GpuAcData) — the diagonal-pairing row permutation, the CSR pattern (host + device),
// the per-family device data with role-folded destination arrays, the DISTR_Q /
// DISTR_RHO / zero-impedance packs, the per-element static trig. The base-case solver
// (SolverContext) and the batched N-1 solver (BatchSolverContext) previously each
// rebuilt all of this from the same extraction; it is now built ONCE (cached Java-side
// per GpuAcData by GpuAcStructures) and held by shared_ptr — a context outliving the
// Java-side owner keeps the structure alive.
// ---------------------------------------------------------------------------------

struct SharedStructure {
    int n = 0;
    int nnz = 0;
    int nElements = 0;                                       // ncb + nob + nsh + nhv (disabled-mask layout)
    int cbOffset = 0, obOffset = 0, shOffset = 0, hvOffset = 0;
    std::vector<int> rowMap;                                 // eq row -> paired variable col
    std::vector<int> rowPtr, colIdx;                         // host CSR pattern (mode-2 blkdiag build)
    Family cb, ob, sh, hv;
    DeviceBuf cbTrig;                                        // per closed-branch STATIC trig (6 doubles: sinKsi cosKsi sinC1 cosC1 sinC2 cosC2)
    int ndq = 0;                                             // DISTR_Q weighted reactive contributions
    DeviceBuf dDqElem, dDqSide, dDqKind, dDqWeight, dDqDest; // ndq each (dDqDest is ndq*6)
    int nzi = 0;                                             // zero-impedance branches
    DeviceBuf dZiIn, dZiIdx, dZiDest;                        // nzi (dZiIn) / nzi*11 (dZiIdx) / nzi*14 (dZiDest)
    DeviceBuf dDrQrGroup;                                    // per DISTR_RHO contribution: the qr relocation group
    int drQr = 0;                                            // whose candidates contain its row (-1 none) — the seq
                                                             // fill suppresses the CURRENT carrier's row in-kernel
    DeviceBuf dZiAltDest;                                    // zi dynamic Q-routing (ob/zi channels): per element,
    DeviceBuf dZiSelRow;                                     // 4 q slots {F1,F2,J1,J2} alt dests + select rows, and
    DeviceBuf dZiRelocGroup;                                 // the GROUP relocation tables (same encodings as the
    DeviceBuf dZiRelocOff;                                   // cb/sh families)
    DeviceBuf dZiRelocPool;
    bool ziDyn = false;
    int ndr = 0;                                             // DISTR_RHO linear contributions
    DeviceBuf dDrCol, dDrCoef, dDrDest;                      // ndr each (dDrDest is ndr*2)
    int ndb = 0;                                             // DISTR_SHUNT_B linear contributions
    DeviceBuf dDbCol, dDbCoef, dDbDest;                      // ndb each (dDbDest is ndb*2)
    int nvdl = 0;                                            // voltage-dependent load model terms (ZIP/exp)
    DeviceBuf dVdlIn, dVdlIdx, dVdlDest;                     // nvdl*3 (c, n, target — verbatim OLF) / nvdl*2 (row, vCol) / nvdl
    int nQr = 0;                                             // pilot-Q relocation GROUPS (plan-A6): candidate
    std::vector<int> hQrPtr, hQrRows;                        // tables [fallback row, member rows...] per group
    DeviceBuf dQrPtr, dQrRows;                               // (rows in DEVICE row space, rowMap applied)
    DeviceBuf dRowPtr, dColIdx, dDiagPos;                    // device CSR pattern + guaranteed-diagonal slots
    double luBytes = 0.0;                                    // cached LU-probe footprint (0 = not probed yet)
};

// The native handle Java holds (GpuAcStructures) is a heap-allocated shared_ptr; each
// context copies it, so destroyStructure only drops the Java-side reference.
using SharedStructurePtr = std::shared_ptr<SharedStructure>;

static void buildDistrQDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                            int ndq, const int* dqElem, const int* dqSide, const int* dqKind,
                            const double* dqWeight, const int* dqRow,
                            const int* cbIdx, const int* obIdx, const int* shIdx) {
    st->ndq = ndq;
    if (ndq > 0) {
        std::vector<int> dqDest(static_cast<size_t>(ndq) * 6, -1);
        for (int c = 0; c < ndq; c++) {
            int mappedRow = rowMap[dqRow[c]];
            dqDest[c * 6 + 0] = -2 - mappedRow;              // q value -> F row
            if (dqKind[c] == 1) {                            // shunt: dq/dv on the bus V column
                dqDest[c * 6 + 1] = pos.at(mappedRow, shIdx[static_cast<size_t>(dqElem[c]) * 5 + 2]);
            } else if (dqKind[c] == 2) {                     // half-open: dq/dv on the connected-side V col
                const int* oq = obIdx + static_cast<size_t>(dqElem[c]) * 6;
                dqDest[c * 6 + 1] = pos.at(mappedRow, dqSide[c] == 0 ? oq[4] : oq[5]);
            } else if (dqKind[c] == 3) {                     // zero-impedance: J slot on the dummyQ column
                dqDest[c * 6 + 1] = pos.at(mappedRow, dqElem[c]);
            } else {                                         // closed branch
                const int* q = cbIdx + static_cast<size_t>(dqElem[c]) * 12;
                dqDest[c * 6 + 1] = pos.at(mappedRow, q[5]);     // dq/dv1
                dqDest[c * 6 + 2] = pos.at(mappedRow, q[7]);     // dq/dv2
                dqDest[c * 6 + 3] = pos.at(mappedRow, q[4]);     // dq/dph1
                dqDest[c * 6 + 4] = pos.at(mappedRow, q[6]);     // dq/dph2
                dqDest[c * 6 + 5] = q[9] >= 0 ? pos.at(mappedRow, q[9]) : -1;  // dq/dr1
            }
        }
        st->dDqElem.upload(dqElem, ndq * sizeof(int));
        st->dDqSide.upload(dqSide, ndq * sizeof(int));
        st->dDqKind.upload(dqKind, ndq * sizeof(int));
        st->dDqWeight.upload(dqWeight, ndq * sizeof(double));
        st->dDqDest.upload(dqDest.data(), dqDest.size() * sizeof(int));
    }
}

static void buildZeroImpDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                             int nzi, const double* ziIn, const int* ziIdx, const int* ziQAlt) {
    st->nzi = nzi;
    if (nzi > 0) {
        std::vector<int> ziDest(static_cast<size_t>(nzi) * 14, -1);
        // zi dynamic Q-routing (ob/zi channels): 4 q slots per element {F side1, F side2, J side1,
        // J side2} with the same alt/select/group encodings as the cb/sh families. A GROUP side's
        // FROZEN F dest is re-pointed at the fallback row (the pilot's own row) so the qBuf
        // diagnostic never lands the pilot's dummy-Q on a member row.
        std::vector<int> ziAlt(static_cast<size_t>(nzi) * 4, -1);
        std::vector<int> ziSel(static_cast<size_t>(nzi) * 4, -1);
        std::vector<int> ziGrp(static_cast<size_t>(nzi) * 4, -1);
        std::vector<int> ziOff(static_cast<size_t>(nzi) * 4, -1);
        std::vector<int> ziPool;
        bool anyDyn = false;
        for (int e = 0; e < nzi; e++) {
            const int* z = ziIdx + static_cast<size_t>(e) * 11;
            // -1 Q row = a DISTR_Q controller whose reactive balance is suppressed (its dummyQ joins the
            // DISTR_Q rows via kind 3, not its own Q balance) — skip the coupling there.
            int b1P = rowMap[z[0]], b2P = rowMap[z[2]];
            int b1Q = z[1] >= 0 ? rowMap[z[1]] : -1, b2Q = z[3] >= 0 ? rowMap[z[3]] : -1;
            int* dd = ziDest.data() + static_cast<size_t>(e) * 14;
            dd[0] = -2 - b1P;                                // F bus1 P (+dummyP)
            dd[1] = -2 - b2P;                                // F bus2 P (-dummyP)
            dd[2] = b1Q >= 0 ? -2 - b1Q : -1;                // F bus1 Q (+dummyQ), skip if suppressed
            dd[3] = b2Q >= 0 ? -2 - b2Q : -1;                // F bus2 Q (-dummyQ), skip if suppressed
            dd[4] = pos.at(b1P, z[4]);                       // J(bus1 P, dummyP)
            dd[5] = pos.at(b2P, z[4]);                       // J(bus2 P, dummyP)
            dd[6] = b1Q >= 0 ? pos.at(b1Q, z[5]) : -1;       // J(bus1 Q, dummyQ)
            dd[7] = b2Q >= 0 ? pos.at(b2Q, z[5]) : -1;       // J(bus2 Q, dummyQ)
            int frozenQ[2] = {b1Q, b2Q};
            for (int side = 0; side < 2; side++) {
                int altRow = ziQAlt[e * 2 + side];
                int fSlot = static_cast<int>(e) * 4 + side;
                int jSlot = static_cast<int>(e) * 4 + 2 + side;
                if (altRow <= -2) {                          // GROUP relocation: one slot per candidate
                    int g = -2 - altRow;
                    ziGrp[fSlot] = g;
                    ziOff[fSlot] = static_cast<int>(ziPool.size());
                    for (int c = st->hQrPtr[g]; c < st->hQrPtr[g + 1]; c++) {
                        ziPool.push_back(-2 - st->hQrRows[c]);
                    }
                    ziGrp[jSlot] = g;
                    ziOff[jSlot] = static_cast<int>(ziPool.size());
                    for (int c = st->hQrPtr[g]; c < st->hQrPtr[g + 1]; c++) {
                        ziPool.push_back(pos.at(st->hQrRows[c], z[5]));
                    }
                    dd[2 + side] = -2 - st->hQrRows[st->hQrPtr[g]];   // frozen F -> fallback (qBuf)
                    anyDyn = true;
                } else if (altRow >= 0 && altRow != frozenQ[side]) {
                    ziAlt[fSlot] = -2 - altRow;
                    ziAlt[jSlot] = pos.at(altRow, z[5]);
                    int sel = frozenQ[side] >= 0 ? frozenQ[side] : altRow;
                    ziSel[fSlot] = sel;                      // suppressed frozen dest: select on ALT row
                    ziSel[jSlot] = sel;
                    anyDyn = true;
                }
            }
            if (z[10]) {                                     // spanning-tree edge
                int dpRow = rowMap[z[4]], dqRow2 = rowMap[z[5]];
                dd[8] = -2 - dpRow;                          // F ZERO_V row
                dd[9] = pos.at(dpRow, z[6]);                 // J(ZERO_V, v1)
                dd[10] = pos.at(dpRow, z[7]);                // J(ZERO_V, v2)
                dd[11] = -2 - dqRow2;                        // F ZERO_PHI row
                dd[12] = pos.at(dqRow2, z[8]);               // J(ZERO_PHI, ph1)
                dd[13] = pos.at(dqRow2, z[9]);               // J(ZERO_PHI, ph2)
            }
        }
        st->dZiIn.upload(ziIn, nzi * sizeof(double));
        st->dZiIdx.upload(ziIdx, static_cast<size_t>(nzi) * 11 * sizeof(int));
        st->dZiDest.upload(ziDest.data(), ziDest.size() * sizeof(int));
        if (anyDyn) {
            st->dZiAltDest.upload(ziAlt.data(), ziAlt.size() * sizeof(int));
            st->dZiSelRow.upload(ziSel.data(), ziSel.size() * sizeof(int));
            if (!ziPool.empty()) {
                st->dZiRelocGroup.upload(ziGrp.data(), ziGrp.size() * sizeof(int));
                st->dZiRelocOff.upload(ziOff.data(), ziOff.size() * sizeof(int));
                st->dZiRelocPool.upload(ziPool.data(), ziPool.size() * sizeof(int));
            }
            st->ziDyn = true;
        }
    }
}

static void buildDistrRhoDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                              int ndr, const int* drRow, const int* drCol, const double* drCoef) {
    st->ndr = ndr;
    if (ndr > 0) {
        std::vector<int> drDest(static_cast<size_t>(ndr) * 2);
        std::vector<int> drQrGroup(ndr, -1);
        for (int c = 0; c < ndr; c++) {
            int mappedRow = rowMap[drRow[c]];
            drDest[c * 2 + 0] = -2 - mappedRow;              // DISTR_RHO F row
            drDest[c * 2 + 1] = pos.at(mappedRow, drCol[c]); // J slot
            // link the row to its pilot-Q relocation group (a member rho row is a candidate): when
            // the free role RE-PICKS to this row per scenario, it hosts the relocated pilot Q and
            // its DISTR_RHO emissions must stop — the SEQUENTIAL fill suppresses in-kernel by the
            // same mode-scan the scatter uses (the batch machine instead pushes drOv zero entries).
            for (int g = 0; g < st->nQr && drQrGroup[c] < 0; g++) {
                for (int k = st->hQrPtr[g] + 1; k < st->hQrPtr[g + 1]; k++) {
                    if (st->hQrRows[k] == mappedRow) {
                        drQrGroup[c] = g;
                        st->drQr = 1;
                        break;
                    }
                }
            }
        }
        st->dDrCol.upload(drCol, ndr * sizeof(int));
        st->dDrCoef.upload(drCoef, ndr * sizeof(double));
        st->dDrDest.upload(drDest.data(), drDest.size() * sizeof(int));
        if (st->drQr) {
            st->dDrQrGroup.upload(drQrGroup.data(), drQrGroup.size() * sizeof(int));
        }
    }
}

static void buildDistrShuntBDest(SharedStructure* st, const CsrPattern& pos, const std::vector<int>& rowMap,
                                 int ndb, const int* dbRow, const int* dbCol, const double* dbCoef) {
    st->ndb = ndb;
    if (ndb > 0) {
        std::vector<int> dbDest(static_cast<size_t>(ndb) * 2);
        for (int c = 0; c < ndb; c++) {
            int mappedRow = rowMap[dbRow[c]];
            dbDest[c * 2 + 0] = -2 - mappedRow;              // DISTR_SHUNT_B F row
            dbDest[c * 2 + 1] = pos.at(mappedRow, dbCol[c]); // J slot
        }
        st->dDbCol.upload(dbCol, ndb * sizeof(int));
        st->dDbCoef.upload(dbCoef, ndb * sizeof(double));
        st->dDbDest.upload(dbDest.data(), dbDest.size() * sizeof(int));
    }
}

// plan-B4: the FROZEN role-0 dest of a GROUP-relocated output backs only the qBuf DIAGNOSTIC
// fill (dynReloc = 0; every Newton fill selects a pool slot instead) — point it at the group's
// FALLBACK row (the pilot's own row, read by no outer loop) so a remote member's diagnostic q
// row carries ONLY its own injection (fillRemoteMemberQBatch). Without this the pilot's whole
// balance lands on the base carrier's row and corrupts the free member's reactive-limit check.
static void patchFrozenGroupDests(DeviceBuf& dest, const std::vector<int>& relocGroup, int nout,
                                  const int* scatterRole, const int* qrPtr, const int* qrRows) {
    for (size_t o = 0; o < relocGroup.size(); o++) {
        int g = relocGroup[o];
        if (g >= 0 && scatterRole[o % nout] == 0) {
            int v = -2 - qrRows[qrPtr[g]];
            CUDA_CHECK(cudaMemcpy(dest.as<int>() + o, &v, sizeof(int), cudaMemcpyHostToDevice));
        }
    }
}

// The GROUP relocation tables of one family, or all-null when the structure has none (plan-A6).
static RelocTables relocOf(Family& f, SharedStructure* st) {
    RelocTables rt;
    if (f.relocPool.p != nullptr) {
        rt.group = f.relocGroup.as<int>();
        rt.off = f.relocOff.as<int>();
        rt.pool = f.relocPool.as<int>();
        rt.qrPtr = st->dQrPtr.as<int>();
        rt.qrRows = st->dQrRows.as<int>();
    }
    return rt;
}

// The zi family's relocation tables, or all-null when it has none.
static RelocTables ziRelocOf(SharedStructure* st) {
    RelocTables rt;
    if (st->dZiRelocPool.p != nullptr) {
        rt.group = st->dZiRelocGroup.as<int>();
        rt.off = st->dZiRelocOff.as<int>();
        rt.pool = st->dZiRelocPool.as<int>();
        rt.qrPtr = st->dQrPtr.as<int>();
        rt.qrRows = st->dQrRows.as<int>();
    }
    return rt;
}

SharedStructure* buildStructure(int n,
                                int ncb, const double* cbIn, const int* cbIdx,
                                int nob, const double* obIn, const int* obIdx,
                                int nsh, const double* shIn, const int* shIdx,
                                int nhv, const double* hvIn, const int* hvIdx,
                                int ndq, const int* dqElem, const int* dqSide, const double* dqWeight,
                                const int* dqRow, const int* dqKind,
                                int nzi, const double* ziIn, const int* ziIdx,
                                int ndr, const int* drRow, const int* drCol, const double* drCoef,
                                int ndb, const int* dbRow, const int* dbCol, const double* dbCoef,
                                const int* cbPcSide,
                                int nvdl, const double* vdlIn, const int* vdlIdx,
                                int nQr, const int* qrPtr, const int* qrRowsIn,
                                const int* obQAltIn, const int* ziQAltIn) {
    auto st = std::make_unique<SharedStructure>();
    st->n = n;
    st->nElements = ncb + nob + nsh + nhv;
    st->cbOffset = 0;
    st->obOffset = ncb;
    st->shOffset = ncb + nob;
    st->hvOffset = ncb + nob + nsh;

    // ---- diagonal-pairing row permutation + index packs rewritten through it ----
    st->rowMap = buildRowMap(n, ncb, cbIdx, nob, obIdx, nsh, shIdx, nhv, hvIdx, ndq, dqRow, nzi, ziIdx);
    const std::vector<int>& rowMap = st->rowMap;
    auto mapRows = [&](const int* idx, int ne, int stride, std::initializer_list<int> rowSlots) {
        std::vector<int> out(idx, idx + static_cast<size_t>(ne) * stride);
        for (int e = 0; e < ne; e++) {
            for (int s : rowSlots) {
                int& r = out[static_cast<size_t>(e) * stride + s];
                if (r >= 0) {
                    r = rowMap[r];
                }
            }
        }
        return out;
    };
    st->nQr = nQr;
    if (nQr > 0) {
        st->hQrPtr.assign(qrPtr, qrPtr + nQr + 1);
        st->hQrRows.assign(qrRowsIn, qrRowsIn + qrPtr[nQr]);
        for (int& r : st->hQrRows) {
            r = rowMap[r];                                   // device row space (self-paired rows: identity)
        }
        st->dQrPtr.upload(st->hQrPtr.data(), st->hQrPtr.size() * sizeof(int));
        st->dQrRows.upload(st->hQrRows.data(), st->hQrRows.size() * sizeof(int));
    }
    std::vector<int> cbIdxM = mapRows(cbIdx, ncb, 12, {0, 1, 2, 3, 10, 11});
    std::vector<int> obIdxM = mapRows(obIdx, nob, 6, {0, 1, 2, 3});
    std::vector<int> obQAltM = mapRows(obQAltIn, nob, 2, {0, 1});   // negative encodings pass through
    std::vector<int> ziQAltM = mapRows(ziQAltIn, nzi, 2, {0, 1});
    const int* obQAlt = obQAltM.data();
    const int* ziQAlt = ziQAltM.data();
    std::vector<int> shIdxM = mapRows(shIdx, nsh, 5, {0, 1, 4});
    std::vector<int> hvIdxM = mapRows(hvIdx, nhv, 4, {0, 1});
    std::vector<int> vdlIdxM = mapRows(vdlIdx, nvdl, 2, {0});
    vdlIdx = vdlIdxM.data();
    cbIdx = cbIdxM.data();
    obIdx = obIdxM.data();
    shIdx = shIdxM.data();
    hvIdx = hvIdxM.data();

    // ---- fixed CSR pattern (rows = permuted equations, cols = variables) ----
    std::vector<std::pair<int, int>> pat;
    collectPattern(ncb, cbIdx, 12, CB_SCATTER, CB_NOUT, resolveCb, pat);
    bool dynReloc = hasRelocatable(ncb, cbIdx, nsh, shIdx)
            || hasAltEntries(obQAlt, nob * 2) || hasAltEntries(ziQAlt, nzi * 2);
    if (dynReloc) {
        collectCbAltPattern(ncb, cbIdx, pat, st->nQr,        // union the alternate Q rows into the pattern
                st->hQrPtr.data(), st->hQrRows.data());      // (+ every GROUP candidate's slots, plan-A6)
    }
    bool phaseCtl = false;
    for (int e = 0; e < ncb; e++) {
        phaseCtl |= cbPcSide[e] >= 0;
    }
    if (phaseCtl) {
        collectCbPhasePattern(ncb, cbIdx, cbPcSide, rowMap, pat);   // union the a1-row P-flow positions
    }
    collectPattern(nob, obIdx, 6, OB_SCATTER, OB_NOUT, resolveOb, pat);
    if (dynReloc) {
        collectObAltPattern(nob, obIdx, obQAlt, pat, st->nQr, st->hQrPtr.data(), st->hQrRows.data());
    }
    collectPattern(nsh, shIdx, 5, SH_SCATTER, SH_NOUT, resolveSh, pat);
    if (dynReloc) {
        collectShAltPattern(nsh, shIdx, pat, st->nQr, st->hQrPtr.data(), st->hQrRows.data());
    }
    collectPattern(nhv, hvIdx, 4, HVDC_SCATTER, HVDC_NOUT, resolveHv, pat);
    collectFamilyPattern(rowMap, cbIdx, obIdx, shIdx, ndq, dqElem, dqSide, dqKind, dqRow,
                         ndr, drRow, drCol, ndb, dbRow, dbCol, nzi, ziIdx, pat);
    if (dynReloc) {
        for (int e = 0; e < nzi; e++) {                      // zi alt J slots: (alt/candidate row, dummyQ col)
            int dqCol = ziIdx[static_cast<size_t>(e) * 11 + 5];
            for (int side = 0; side < 2; side++) {
                int altRow = ziQAlt[e * 2 + side];
                if (altRow <= -2) {
                    int g = -2 - altRow;
                    for (int c = st->hQrPtr[g]; c < st->hQrPtr[g + 1]; c++) {
                        pat.emplace_back(st->hQrRows[c], dqCol);
                    }
                } else if (altRow >= 0) {
                    pat.emplace_back(altRow, dqCol);
                }
            }
        }
    }
    for (int e = 0; e < nvdl; e++) {                         // voltage-dependent load: (row, own-V) slot
        pat.emplace_back(vdlIdx[e * 2], vdlIdx[e * 2 + 1]);
    }
    for (int r = 0; r < n; r++) {
        pat.emplace_back(r, r);                              // guaranteed diagonal: the per-solve
    }                                                        // row-mode mask writes 1.0 there
    std::sort(pat.begin(), pat.end());
    pat.erase(std::unique(pat.begin(), pat.end()), pat.end());
    st->nnz = static_cast<int>(pat.size());
    if (st->nnz == 0) throw std::runtime_error("empty Jacobian pattern");

    st->rowPtr.assign(n + 1, 0);
    st->colIdx.resize(st->nnz);
    for (int i = 0; i < st->nnz; i++) {                      // pat is row-major sorted
        st->colIdx[i] = pat[i].second;
        st->rowPtr[pat[i].first + 1]++;
    }
    for (int r = 0; r < n; r++) st->rowPtr[r + 1] += st->rowPtr[r];
    CsrPattern pos{st->rowPtr.data(), st->colIdx.data()};    // (row,col) -> slot by binary search
    st->dRowPtr.upload(st->rowPtr.data(), (n + 1) * sizeof(int));
    st->dColIdx.upload(st->colIdx.data(), st->nnz * sizeof(int));

    // ---- family device data (inputs, indices, role-folded destinations) + static trig ----
    setupFamily(st->cb, ncb, cbIn, 8, cbIdx, 12, CB_SCATTER, CB_NOUT, resolveCb, pos);
    if (dynReloc && ncb) {                                   // Dynamic Q-relocation: alt-dest + frozen-row select map
        std::vector<int> altDest;
        std::vector<int> selRow;
        std::vector<int> relocGroup;
        std::vector<int> relocOff;
        std::vector<int> relocPool;
        buildCbDynDest(ncb, cbIdx, pos, altDest, selRow, st->nQr, st->hQrPtr.data(), st->hQrRows.data(),
                relocGroup, relocOff, relocPool);
        st->cb.altDest.upload(altDest.data(), altDest.size() * sizeof(int));
        st->cb.selRow.upload(selRow.data(), selRow.size() * sizeof(int));
        if (!relocPool.empty()) {
            st->cb.relocGroup.upload(relocGroup.data(), relocGroup.size() * sizeof(int));
            st->cb.relocOff.upload(relocOff.data(), relocOff.size() * sizeof(int));
            st->cb.relocPool.upload(relocPool.data(), relocPool.size() * sizeof(int));
            static int cbRoles[CB_NOUT];
            for (int k = 0; k < CB_NOUT; k++) {
                cbRoles[k] = CB_SCATTER[k].role;
            }
            patchFrozenGroupDests(st->cb.dest, relocGroup, CB_NOUT, cbRoles,
                    st->hQrPtr.data(), st->hQrRows.data());
        }
        st->cb.dynReloc = true;
    }
    if (phaseCtl) {                                          // continuous phase control: P-flow row at a1
        std::vector<int> pcDest;
        std::vector<int> pcSel;
        buildCbPhaseDest(ncb, cbIdx, cbPcSide, rowMap, pos, pcDest, pcSel);
        st->cb.pcDest.upload(pcDest.data(), pcDest.size() * sizeof(int));
        st->cb.pcSel.upload(pcSel.data(), pcSel.size() * sizeof(int));
        st->cb.phaseCtl = true;
    }
    if (ncb) {                                               // per-element static trig (OLF's precomputed-trig shape)
        st->cbTrig.alloc(static_cast<size_t>(ncb) * 6 * sizeof(double));
        initClosedBranchTrig<<<grid(ncb), BLOCK>>>(ncb, st->cb.in.as<double>(), st->cbTrig.as<double>());
    }
    setupFamily(st->ob, nob, obIn, 7, obIdx, 6, OB_SCATTER, OB_NOUT, resolveOb, pos);
    if (dynReloc && nob) {                                   // ob dynamic Q-routing: alt-dest + selector
        std::vector<int> obAltDest;
        std::vector<int> obSelRow;
        std::vector<int> obRelocGroup;
        std::vector<int> obRelocOff;
        std::vector<int> obRelocPool;
        buildObDynDest(nob, obIdx, obQAlt, pos, obAltDest, obSelRow, st->nQr, st->hQrPtr.data(), st->hQrRows.data(),
                obRelocGroup, obRelocOff, obRelocPool);
        st->ob.altDest.upload(obAltDest.data(), obAltDest.size() * sizeof(int));
        st->ob.selRow.upload(obSelRow.data(), obSelRow.size() * sizeof(int));
        if (!obRelocPool.empty()) {
            st->ob.relocGroup.upload(obRelocGroup.data(), obRelocGroup.size() * sizeof(int));
            st->ob.relocOff.upload(obRelocOff.data(), obRelocOff.size() * sizeof(int));
            st->ob.relocPool.upload(obRelocPool.data(), obRelocPool.size() * sizeof(int));
            static int obRoles[OB_NOUT];
            for (int k = 0; k < OB_NOUT; k++) {
                obRoles[k] = OB_SCATTER[k].role;
            }
            patchFrozenGroupDests(st->ob.dest, obRelocGroup, OB_NOUT, obRoles,
                    st->hQrPtr.data(), st->hQrRows.data());
        }
        st->ob.dynReloc = true;
    }
    setupFamily(st->sh, nsh, shIn, 2, shIdx, 5, SH_SCATTER, SH_NOUT, resolveSh, pos);
    if (dynReloc && nsh) {                                   // Dynamic Q-relocation: shunt alt-dest + selector
        std::vector<int> shAltDest;
        std::vector<int> shSelRow;
        std::vector<int> shRelocGroup;
        std::vector<int> shRelocOff;
        std::vector<int> shRelocPool;
        buildShDynDest(nsh, shIdx, pos, shAltDest, shSelRow, st->nQr, st->hQrPtr.data(), st->hQrRows.data(),
                shRelocGroup, shRelocOff, shRelocPool);
        st->sh.altDest.upload(shAltDest.data(), shAltDest.size() * sizeof(int));
        st->sh.selRow.upload(shSelRow.data(), shSelRow.size() * sizeof(int));
        if (!shRelocPool.empty()) {
            st->sh.relocGroup.upload(shRelocGroup.data(), shRelocGroup.size() * sizeof(int));
            st->sh.relocOff.upload(shRelocOff.data(), shRelocOff.size() * sizeof(int));
            st->sh.relocPool.upload(shRelocPool.data(), shRelocPool.size() * sizeof(int));
            static int shRoles[SH_NOUT];
            for (int k = 0; k < SH_NOUT; k++) {
                shRoles[k] = SH_SCATTER[k].role;
            }
            patchFrozenGroupDests(st->sh.dest, shRelocGroup, SH_NOUT, shRoles,
                    st->hQrPtr.data(), st->hQrRows.data());
        }
        st->sh.dynReloc = true;
    }
    setupFamily(st->hv, nhv, hvIn, 5, hvIdx, 4, HVDC_SCATTER, HVDC_NOUT, resolveHv, pos);

    // ---- DISTR_Q destinations: per contribution {F-row, dq/dv1, dq/dv2, dq/dph1, dq/dph2, dq/dr1}
    //      (-2-row encodes an F row, >=0 a J slot, -1 skip) ----
    buildDistrQDest(st.get(), pos, rowMap, ndq, dqElem, dqSide, dqKind, dqWeight, dqRow, cbIdx, obIdx, shIdx);

    // ---- zero-impedance destinations: per branch the 8 dummy couplings + (tree) the 6 ZERO_V/ZERO_PHI ----
    buildZeroImpDest(st.get(), pos, rowMap, nzi, ziIn, ziIdx, ziQAlt);

    // ---- DISTR_RHO destinations: per contribution {F row, J slot (row, r1 col)} ----
    buildDistrRhoDest(st.get(), pos, rowMap, ndr, drRow, drCol, drCoef);

    // ---- DISTR_SHUNT_B destinations: per contribution {F row, J slot (row, b col)} ----
    buildDistrShuntBDest(st.get(), pos, rowMap, ndb, dbRow, dbCol, dbCoef);

    // ---- voltage-dependent load (ZIP/exponential) destinations: one J slot (row, own-V) per term ----
    st->nvdl = nvdl;
    if (nvdl > 0) {
        std::vector<int> vdlDest(nvdl);
        for (int e = 0; e < nvdl; e++) {
            vdlDest[e] = pos.at(vdlIdx[e * 2], vdlIdx[e * 2 + 1]);
        }
        st->dVdlIn.upload(vdlIn, static_cast<size_t>(nvdl) * 3 * sizeof(double));
        st->dVdlIdx.upload(vdlIdx, static_cast<size_t>(nvdl) * 2 * sizeof(int));
        st->dVdlDest.upload(vdlDest.data(), static_cast<size_t>(nvdl) * sizeof(int));
    }

    std::vector<int> diagPos(n);
    for (int r = 0; r < n; r++) diagPos[r] = pos.at(r, r);
    st->dDiagPos.upload(diagPos.data(), n * sizeof(int));
    CUDA_CHECK(cudaDeviceSynchronize());                     // initClosedBranchTrig done before first use
    return st.release();
}

constexpr double LU_FILL = 12.0;     // heuristic LU-fill multiplier over nnz (fallback when LU_NNZ is unavailable)

// Per-scenario bytes the LU numeric factors occupy. The exact fill-in is only known after a
// symbolic ANALYSIS, so run a CHEAP one on a single matrix with the structure's device pattern
// and read cuDSS's reported CUDSS_DATA_LU_NNZ; fall back to the LU_FILL heuristic if that is
// unavailable (older cuDSS, query failure). The result is CACHED on the structure, so repeated
// chunk-size queries (one per SA run) probe only once per extraction.
double luBytesPerScenario(SharedStructure& st) {
    if (st.luBytes > 0) {
        return st.luBytes;
    }
    int n = st.n;
#if OLF_GPU_CUDSS_LU_NNZ
    try {
        DeviceBuf dVals;
        DeviceBuf dB;
        DeviceBuf dX;
        dVals.alloc(st.nnz * sizeof(double));                // ANALYSIS is symbolic — values unused
        CUDA_CHECK(cudaMemset(dVals.p, 0, st.nnz * sizeof(double)));
        dB.alloc(n * sizeof(double));
        dX.alloc(n * sizeof(double));
        CudssCtx ds;
        CUDSS_CHECK(cudssCreate(&ds.handle));
        CUDSS_CHECK(cudssConfigCreate(&ds.config));
        applyReorderAlg(ds.config);
        CUDSS_CHECK(cudssDataCreate(ds.handle, &ds.data));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ds.matA, n, n, st.nnz, st.dRowPtr.as<int>(), nullptr,
                                         st.dColIdx.as<int>(), dVals.as<double>(), OLF_CUDSS_CSR_TYPES,
                                         CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matB, n, 1, n, dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matX, n, 1, n, dX.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssExecute(ds.handle, CUDSS_PHASE_ANALYSIS, ds.config, ds.data, ds.matA, ds.matX, ds.matB));
        // cuDSS's OWN device-memory estimate for one matrix with this pattern — [0] permanent,
        // [1] peak (16 int64 entries). This is the number to size with: the LU numeric factors are
        // only part of what a factorization allocates, and the workspace is the part our model used
        // to miss. MEASURED on rte6515 (n=13040, nnz=91002, 2026-09-08): cuDSS reports 4.97 MB/
        // matrix while the LU factors alone are ~1.3 MB, and the batch's REAL device usage sampled
        // over chunk sizes 50 and 200 was 3.7 MiB/scenario — i.e. this estimate is slightly
        // conservative, which is exactly what a chunk budget wants.
        {
            int64_t est[16] = {0};
            size_t writtenEst = 0;
            if (cudssDataGet(ds.handle, ds.data, CUDSS_DATA_MEMORY_ESTIMATES,
                             est, sizeof(est), &writtenEst) == CUDSS_STATUS_SUCCESS
                    && writtenEst >= 2 * sizeof(int64_t)) {
                int64_t deviceBytes = std::max(est[0], est[1]);
                if (deviceBytes > 0) {
                    st.luBytes = static_cast<double>(deviceBytes);
                    return st.luBytes;
                }
            }
        }
        int64_t luNnz = 0;
        size_t written = 0;
        CUDSS_CHECK(cudssDataGet(ds.handle, ds.data, CUDSS_DATA_LU_NNZ, &luNnz, sizeof(luNnz), &written));
        if (luNnz > 0) {
            st.luBytes = static_cast<double>(luNnz) * sizeof(double);
            return st.luBytes;
        }
    } catch (...) {
        // fall through to the heuristic
    }
#endif
    st.luBytes = LU_FILL * static_cast<double>(st.nnz) * sizeof(double);
    return st.luBytes;
}


// ---------------------------------------------------------------------------------
// The CACHED solver context: the SHARED structure (pattern, families, destination
// arrays — see SharedStructure) plus this consumer's OWN state — the per-solve device
// buffers, the cuDSS objects and its ANALYSIS — built once and reused across solver
// runs (outer loops re-run the solver many times with only targets / state changing).
// Per solve, only x0 and the targets cross JNI; cuDSS keeps its symbolic factorization
// and REFACTORIZES on the in-place-filled values. Java owns the lifetime
// (GpuNewtonRaphson caches the handle, a Cleaner frees it).
// ---------------------------------------------------------------------------------

struct SolverContext {
    SharedStructurePtr st;                                   // shared with any batch context on the same extraction
    int n = 0;
    int nnz = 0;
    int ndq = 0, nzi = 0, ndr = 0, ndb = 0;                  // copies of the structure counts
    std::vector<double> hF;                                  // host mismatch buffer
    DeviceBuf dX, dTarget, dF, dB, dDx, dVals, dMode;
    DeviceBuf dXprev;                                        // state before the last full step (line search)
    CudssCtx ds;
    bool analyzed = false;                                   // ANALYSIS+first FACTORIZATION done
    // Dedicated BLOCKING stream for the solve (kernels/memsets/D2D + cuDSS via cudssSetStream),
    // same rationale as BatchSolverContext::stream: keep the iteration off the legacy default
    // stream's implicit device-wide serialization; missed ops still order correctly.
    cudaStream_t stream = nullptr;
    ~SolverContext() {
        if (stream) cudaStreamDestroy(stream);
    }
};

SolverContext* createContext(const SharedStructurePtr& st) {
    auto ctx = std::make_unique<SolverContext>();
    ctx->st = st;
    int n = st->n;
    ctx->n = n;
    ctx->nnz = st->nnz;
    ctx->ndq = st->ndq;
    ctx->nzi = st->nzi;
    ctx->ndr = st->ndr;
    ctx->ndb = st->ndb;
    ctx->hF.resize(n);

    // ---- device state (persists across solves) ----
    ctx->dX.alloc(n * sizeof(double));
    ctx->dTarget.alloc(n * sizeof(double));
    ctx->dF.alloc(n * sizeof(double));
    ctx->dB.alloc(n * sizeof(double));
    ctx->dDx.alloc(n * sizeof(double));
    ctx->dXprev.alloc(n * sizeof(double));
    ctx->dVals.alloc(static_cast<size_t>(ctx->nnz) * sizeof(double));
    ctx->dMode.alloc(n * sizeof(int));

    // ---- cuDSS objects: J as true CSR; kernels fill d_ax in place; ANALYSIS lazy ----
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    CUDSS_CHECK(cudssCreate(&ctx->ds.handle));
    CUDSS_CHECK(cudssSetStream(ctx->ds.handle, ctx->stream));
    CUDSS_CHECK(cudssConfigCreate(&ctx->ds.config));
    int ir = irSteps();
    CUDSS_CHECK(cudssConfigSet(ctx->ds.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)));
    applyReorderAlg(ctx->ds.config);
    CUDSS_CHECK(cudssDataCreate(ctx->ds.handle, &ctx->ds.data));
    CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->ds.matA, n, n, ctx->nnz,
                                     st->dRowPtr.as<int>(), nullptr, st->dColIdx.as<int>(), ctx->dVals.as<double>(),
                                     OLF_CUDSS_CSR_TYPES,
                                     CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&ctx->ds.matB, n, 1, n, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&ctx->ds.matX, n, 1, n, ctx->dDx.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    return ctx.release();
}

Result solveContext(SolverContext* ctx, const double* x0, const double* target, const int* rowMode,
                    int maxIter, double tol, double convEpsPerEq, int lsMode, int lsMaxIter, double lsStepFold,
                    double maxDv, double maxDphi, const int* varType) {
    int n = ctx->n;
    CUDA_CHECK(cudaMemcpy(ctx->dX.p, x0, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dTarget.p, target, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dMode.p, rowMode, n * sizeof(int), cudaMemcpyHostToDevice));

    // Fill J values + mismatch F at the current dX, read F back, and return its infnorm (for the
    // convergence test). Also sets curNorm2 to the EUCLIDEAN 2-norm — the quantity OLF's line search
    // compares (DefaultNewtonRaphsonStoppingCriteria uses Vectors.norm2): the Newton direction is a
    // descent direction for ‖F‖₂ but NOT for ‖F‖∞, so backtracking must judge on the 2-norm or it
    // stalls. Both norms are set to HUGE_VAL when any mismatch entry is non-finite.
    double curNorm2 = 0.0;
    auto fillAndNorm = [&]() -> double {
        CUDA_CHECK(cudaMemsetAsync(ctx->dVals.p, 0, ctx->nnz * sizeof(double), ctx->stream));
        initF<<<grid(n), BLOCK, 0, ctx->stream>>>(n, ctx->dTarget.as<double>(), ctx->dF.as<double>());
        if (ctx->st->cb.ne) fillClosedBranch<<<grid(ctx->st->cb.ne), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->st->cb.altDest.as<int>(), ctx->st->cb.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->cb.dynReloc ? 1 : 0, relocOf(ctx->st->cb, ctx->st.get()));
        if (ctx->st->ob.ne) fillOpenBranch<<<grid(ctx->st->ob.ne), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->st->ob.altDest.as<int>(), ctx->st->ob.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ob.dynReloc ? 1 : 0, relocOf(ctx->st->ob, ctx->st.get()));
        if (ctx->st->sh.ne) fillShunt<<<grid(ctx->st->sh.ne), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->st->sh.altDest.as<int>(), ctx->st->sh.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->sh.dynReloc ? 1 : 0, relocOf(ctx->st->sh, ctx->st.get()));
        if (ctx->st->nvdl) fillVdlBatch<<<grid(ctx->st->nvdl), BLOCK, 0, ctx->stream>>>(ctx->st->nvdl, 1, ctx->st->n, ctx->st->nnz, ctx->st->dVdlIn.as<double>(), ctx->st->dVdlIdx.as<int>(), ctx->st->dVdlDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>());
        if (ctx->st->hv.ne) fillHvdc<<<grid(ctx->st->hv.ne), BLOCK, 0, ctx->stream>>>(ctx->st->hv.ne, ctx->st->hv.in.as<double>(), ctx->st->hv.ix.as<int>(), ctx->st->hv.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>());
        if (ctx->ndq) fillDistrQ<<<grid(ctx->ndq), BLOCK, 0, ctx->stream>>>(ctx->ndq, ctx->st->dDqElem.as<int>(), ctx->st->dDqSide.as<int>(), ctx->st->dDqKind.as<int>(), ctx->st->dDqWeight.as<double>(), ctx->st->dDqDest.as<int>(), ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dMode.as<int>());
        if (ctx->nzi) fillZeroImp<<<grid(ctx->nzi), BLOCK, 0, ctx->stream>>>(ctx->nzi, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->st->dZiAltDest.as<int>(), ctx->st->dZiSelRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ziDyn ? 1 : 0, ziRelocOf(ctx->st.get()));
        if (ctx->ndr) fillDistrRho<<<grid(ctx->ndr), BLOCK, 0, ctx->stream>>>(ctx->ndr, ctx->st->dDrCol.as<int>(), ctx->st->dDrCoef.as<double>(), ctx->st->dDrDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->st->drQr ? ctx->st->dDrQrGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>());
        if (ctx->ndb) fillDistrShuntB<<<grid(ctx->ndb), BLOCK, 0, ctx->stream>>>(ctx->ndb, ctx->st->dDbCol.as<int>(), ctx->st->dDbCoef.as<double>(), ctx->st->dDbDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>());
        applyRowMode<<<grid(n), BLOCK, 0, ctx->stream>>>(n, ctx->dMode.as<int>(), ctx->st->dRowPtr.as<int>(), ctx->st->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>());
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        CUDA_CHECK(cudaMemcpy(ctx->hF.data(), ctx->dF.p, n * sizeof(double), cudaMemcpyDeviceToHost));
        double infn = 0.0;
        double sumsq = 0.0;
        for (double v : ctx->hF) {
            if (!std::isfinite(v)) {
                curNorm2 = HUGE_VAL;
                return HUGE_VAL;
            }
            infn = std::max(infn, std::fabs(v));
            sumsq += v * v;
        }
        curNorm2 = std::sqrt(sumsq);
        return infn;
    };

    Result res;
    res.x.resize(n);
    std::vector<double> hDx(lsMode == 2 ? n : 0);            // host step buffer for MAX_VOLTAGE_CHANGE clamp
    const bool trace = std::getenv("OLF_GPU_TRACE") != nullptr;   // per-iteration diagnostic
    const double olf2NormThreshold = convEpsPerEq > 0.0 ? std::sqrt(convEpsPerEq * convEpsPerEq * n) : 0.0;
    double lastNorm2 = 0.0;
    bool haveLast = false;                                   // a full step was applied at the previous iteration
    int it = 0;
    for (; it <= maxIter; it++) {
        // fill J + F; res.norm is the infnorm (convergence test), norm2 the 2-norm (line search)
        res.norm = fillAndNorm();
        double norm2 = curNorm2;

        // Line search (LINE_SEARCH mode): the full step applied at it-1 may have RAISED ‖F‖₂;
        // backtrack x = xPrev + mu*dx with mu = stepFold^-k (k=1..lsMaxIter) until the 2-norm
        // improves, mirroring OLF's LineSearchStateVectorScaling. mu==1 (step already improved)
        // means no backtrack — the undamped path stays bit-identical. The re-fill leaves J at the
        // accepted x, so the next solve uses the correct Jacobian.
        if (lsMode == 1 && haveLast && norm2 >= lastNorm2) {
            for (int k = 1; k <= lsMaxIter && norm2 >= lastNorm2; k++) {
                double mu = 1.0 / std::pow(lsStepFold, k);
                stepFromPrev<<<grid(n), BLOCK, 0, ctx->stream>>>(n, ctx->dXprev.as<double>(), ctx->dDx.as<double>(), mu, ctx->dX.as<double>());
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                res.norm = fillAndNorm();
                norm2 = curNorm2;
            }
        }
        if (!std::isfinite(res.norm)) {
            throw std::runtime_error("mismatch diverged to NaN/Inf at iteration " + std::to_string(it));
        }
        if (trace) {
            std::fprintf(stderr, "OLF_GPU_TRACE it=%d infnorm=%.17e norm2=%.17e\n", it, res.norm, norm2);
        }
        lastNorm2 = norm2;
        haveLast = true;
        bool converged = convEpsPerEq > 0.0 ? (norm2 < olf2NormThreshold) : (res.norm < tol);
        // Mirror CPU NewtonRaphson.run: x0 may already satisfy the stopping criterion after an outer-loop
        // retarget, but the CPU still performs one Newton step before it is allowed to stop. If we early-exit
        // here on it==0, distributed-slack target changes can be accepted without any state update and the
        // outer loop sees the same slack mismatch again.
        if ((converged && it > 0) || it == maxIter) break;

        // J dx = -F on the device (analyze ONCE per structure, refactorize after —
        // including across solver runs reusing this context)
        negateInto<<<grid(n), BLOCK, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dB.as<double>());
        if (!ctx->analyzed) {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB));
            traceLuNnz(ctx->ds.handle, ctx->ds.data, "base");
        }
        // ONE cudssExecute per iteration (phases are OR-able bit flags): first pass factorizes,
        // later ones refactorize reusing the pivot order — fused with the solve either way.
        CUDSS_CHECK(cudssExecute(ctx->ds.handle,
                (ctx->analyzed ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION) | CUDSS_PHASE_SOLVE,
                ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB));
        ctx->analyzed = true;
        CUDA_CHECK(cudaMemcpyAsync(ctx->dXprev.p, ctx->dX.p, n * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
        if (lsMode == 2) {
            // MAX_VOLTAGE_CHANGE: one global step factor so the largest |Δv| / |Δφ| stays within
            // maxDv / maxDphi (mirrors MaxVoltageChangeStateVectorScaling). Read dx back (same per-iter
            // cost as the mismatch readback) and reduce on the host; the whole step is then scaled —
            // including non-V/φ rows like BRANCH_RHO1 — via stepFromPrev(x = xPrev + s*dx).
            CUDA_CHECK(cudaMemcpy(hDx.data(), ctx->dDx.p, n * sizeof(double), cudaMemcpyDeviceToHost));
            double stepSize = 1.0;
            for (int r = 0; r < n; r++) {
                double a = std::fabs(hDx[r]);
                if (varType[r] == 1) {
                    if (a > maxDv) stepSize = std::min(stepSize, maxDv / a);
                } else if (varType[r] == 0) {
                    if (a > maxDphi) stepSize = std::min(stepSize, maxDphi / a);
                }
            }
            stepFromPrev<<<grid(n), BLOCK, 0, ctx->stream>>>(n, ctx->dXprev.as<double>(), ctx->dDx.as<double>(), stepSize, ctx->dX.as<double>());
        } else {
            axpy1<<<grid(n), BLOCK, 0, ctx->stream>>>(n, ctx->dDx.as<double>(), ctx->dX.as<double>());
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }
    res.iters = it;
    // Return the LAST iterate (res.norm is its ‖F‖∞, set by the final fillAndNorm), matching the CPU
    // NewtonRaphson, which returns whatever state it holds when its loop exits. The final
    // CONVERGED / MAX_ITERATION_REACHED verdict is decided by the configured stopping criteria on the
    // CPU from this written-back state (see GpuNewtonRaphson), so no on-device best-iterate selection.
    CUDA_CHECK(cudaMemcpy(res.x.data(), ctx->dX.p, n * sizeof(double), cudaMemcpyDeviceToHost));
    return res;
}

// VALUES-level refresh: re-upload only the per-element PARAMETER packs (cbIn/obIn/shIn/hvIn/ziIn)
// into the SHARED structure, in place. The CSR pattern, the diagonal pairing, the index/dest packs and
// cuDSS's symbolic ANALYSIS are UNCHANGED — they depend on the index packs and structure, not on these
// values (a transformer tap or shunt-b move changes r1/a1/b — or a coupler's rho, ziIn — only). The
// fill kernels read in[] every iteration, so the next solveContext refactorizes on the fresh values.
// Element COUNTS must match the cached structure (the Java side only takes this path when the index
// packs are identical), so we copy into the EXISTING device buffers (no realloc — DeviceBuf::alloc
// would leak the old pointer). The static trig is re-derived (a tap move can change ksi/a1). Mutating
// the shared structure is the correct semantic — one extraction has ONE value set; any context sharing
// it must see the new values (the Java side then re-keys the structure cache to the new extraction,
// GpuAcStructures.rebind, so content matching never aliases stale values).
void refreshStructureValues(SharedStructure& st, int ncb, const double* cbIn, int nob, const double* obIn,
                            int nsh, const double* shIn, int nhv, const double* hvIn,
                            int nzi, const double* ziIn) {
    auto refresh = [](Family& f, int ne, const double* in) {
        if (ne != f.ne) {
            throw std::runtime_error("refreshContextValues: element count changed (not a VALUES event)");
        }
        if (ne) {
            CUDA_CHECK(cudaMemcpy(f.in.p, in, static_cast<size_t>(ne) * f.instride * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
    };
    refresh(st.cb, ncb, cbIn);
    refresh(st.ob, nob, obIn);
    refresh(st.sh, nsh, shIn);
    refresh(st.hv, nhv, hvIn);
    if (nzi != st.nzi) {
        throw std::runtime_error("refreshContextValues: zero-impedance count changed (not a VALUES event)");
    }
    if (nzi) {
        CUDA_CHECK(cudaMemcpy(st.dZiIn.p, ziIn, static_cast<size_t>(nzi) * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (ncb) {
        initClosedBranchTrig<<<grid(ncb), BLOCK>>>(ncb, st.cb.in.as<double>(), st.cbTrig.as<double>());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// =================================================================================
// BATCHED N-1 security analysis: many post-contingency scenarios on the SAME network
// structure, solved in one batched Newton-Raphson. The CSR pattern, the family index
// packs, the destination arrays and cuDSS's SYMBOLIC ANALYSIS are SHARED across all
// scenarios (built once); only the value/state/mismatch arrays gain a scenario
// dimension. Per-scenario element disablement (a branch outage) is a device mask the
// fill kernels honor (a disabled element simply does not scatter — its entries stay
// structurally present but numerically zero, and the diagonal-pairing guarantee keeps
// every row factorizable). Converged scenarios retire from the batch (an active mask),
// so the per-iteration cost shrinks as scenarios finish.
//
// cuDSS (OLF_GPU_BATCHED_CUDSS, default on for cuDSS >= 0.3.0): the UNIFORM BATCHED API —
// one batch CSR matrix whose scenarios share the rowPtr/colIdx pattern with per-scenario
// value sets, batched dense B/X. ANALYSIS runs ONCE on the shared pattern; each Newton
// iteration is ONE batched FACTORIZATION + ONE batched SOLVE over all scenarios. Retired
// scenarios (converged/diverged) are frozen to an identity system so a singular sub-block
// never fails the batch. On older cuDSS the fallback path repoints a single matrix and
// refactorizes+solves per scenario. Either way the symbolic analysis — the dominant
// per-case cost in sequential SA — is amortized over the entire contingency set.
// =================================================================================

// Batched fill kernels: thread = element x scenario. Skip if the scenario has retired
// (converged) or the element is outaged in that scenario. Values/mismatch are per-scenario.
// ---------------------------------------------------------------------------------
// Closed-branch PRECOMPUTED TRIG (OLF's own CPU optimization, on device): ksi and the
// uncontrolled a1 are STATIC per element, so sin/cos(ksi) and sin/cos(ksi -/+ a1 -/+ A2)
// are computed ONCE per element (initClosedBranchTrig, at context build). Per
// (element, scenario) the fill then derives sinTheta1/2, cosTheta1/2 from ONE dynamic
// sincos(dph) via the angle-addition identities:
//   theta1 = (ksi - a1 + A2) + (ph2 - ph1) = C1 + d
//   theta2 = (ksi + a1 - A2) - (ph2 - ph1) = C2 - d
// — 1 dynamic sincos per branch-scenario instead of 3 (the fill kernel was
// compute-bound on FP64 trig). A branch whose a1 IS a state variable (phase control,
// a1col >= 0) falls back to direct sincos of theta1/theta2 (2 dynamic). The generated
// evalClosedBranchPre/PreF routines take the six values as parameters — the same
// precomputed-trig signature shape as OLF's Java methods. The static-trig buffer is
// filled at structure build (initClosedBranchTrig, defined next to SharedStructure).
// ---------------------------------------------------------------------------------

__device__ inline void closedBranchTrig(const double* t, bool a1FromState, double ksi, double a1,
                                        double ph1, double ph2,
                                        double& sinKsi, double& cosKsi,
                                        double& sinTheta1, double& cosTheta1,
                                        double& sinTheta2, double& cosTheta2) {
    sinKsi = t[0];
    cosKsi = t[1];
    if (a1FromState) {                                       // controlled phase: C1/C2 statics invalid
        sincos(ksi - a1 + A2 - ph1 + ph2, &sinTheta1, &cosTheta1);
        sincos(ksi + a1 - A2 + ph1 - ph2, &sinTheta2, &cosTheta2);
        return;
    }
    double sind;
    double cosd;
    sincos(ph2 - ph1, &sind, &cosd);
    sinTheta1 = t[2] * cosd + t[3] * sind;                   // sin(C1 + d)
    cosTheta1 = t[3] * cosd - t[2] * sind;
    sinTheta2 = t[4] * cosd - t[5] * sind;                   // sin(C2 - d)
    cosTheta2 = t[5] * cosd + t[4] * sind;
}

// Per-scenario closed-branch TAP override (tap/PST actions on FIXED-tap transformers): entries
// are (slot, cb elem) with the target tap's FULL static row (y g1 g2 b1 b2 ksi r1 a1)
// replacing the element's cbIn row — per-tap r/x/g/b changes are first-class (y/ksi move).
// Java guards that the branch's ratio/phase are NOT state variables (r1col/a1col == -1), so
// the state path never conflicts. An override invalidates the element's precomputed trig
// statics (ksi/a1 may differ) — the caller must take the raw-sincos path (same fallback as
// a state-variable a1). Returns true when out8 holds the replacement row.
__device__ inline bool findTapOverride(int nTapOv, const int* tapOvSlotElem, const double* tapOvVal,
                                       int s, int e, double* out8) {
    for (int k = 0; k < nTapOv; k++) {
        if (tapOvSlotElem[2 * k] == s && tapOvSlotElem[2 * k + 1] == e) {
            for (int c = 0; c < 8; c++) {
                out8[c] = tapOvVal[8 * k + c];
            }
            return true;
        }
    }
    return false;
}

__global__ void fillClosedBranchBatch(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                      const double* in, const int* ix, const int* dest,
                                      const double* x, double* vals, double* F,
                                      const char* disabled, const char* active,
                                      int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                      const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                      const int* pcDest, const int* pcSel, RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 8;                            // y g1 g2 b1 b2 ksi r1 a1
    const int* q = ix + e * 12;                              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) {
        p = pOv;
    }
    const double* xs = x + s * n;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    // The FUSED fill is atomic/memory-bound on its 28 scattered outputs (measured:
    // routing it through the precomputed-trig buffer made it SLOWER, 81 -> 100 ms —
    // the extra static reads cost more than the saved FP64 trig), so it keeps the
    // raw-angle assembler; the F-ONLY fill below IS compute-bound and uses the
    // precomputed-trig path.
    double o[CB_NOUT];
    evalClosedBranch(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    if (dynReloc) {
        scatterAddDyn(CB_NOUT, dest + e * CB_NOUT, altDest + e * CB_NOUT, selRow + e * CB_NOUT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n, rt, e * CB_NOUT);
    } else {
        scatterAdd(CB_NOUT, dest + e * CB_NOUT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
    if (pcDest != nullptr) {                                 // continuous phase control: emit the controlled
        const int* pd = pcDest + e * CB_NOUT;                // side's P outputs AGAIN at the a1 row when this
        const int* psl = pcSel + e * CB_NOUT;                // scenario has the control enabled (row mode 3)
        for (int k = 0; k < CB_NOUT; k++) {
            int d = pd[k];
            if (d == -1 || mode[(size_t) s * n + psl[k]] != 3) {
                continue;
            }
            if (d >= 0) {
                atomicAdd(vals + (size_t) s * nnz + d, o[k]);
            } else {
                atomicAdd(F + (size_t) s * n + (-2 - d), o[k]);
            }
        }
    }
}

__global__ void fillOpenBranchBatch(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                    const double* in, const int* ix, const int* dest,
                                    const double* x, double* vals, double* F,
                                    const char* disabled, const char* active,
                                    const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                    RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 7;                            // y ksi g1 b1 g2 b2 r1
    const int* q = ix + e * 6;
    const double* xs = x + s * n;
    double v1 = q[4] >= 0 ? xs[q[4]] : 1.0;
    double v2 = q[5] >= 0 ? xs[q[5]] : 1.0;
    double o[OB_NOUT];
    evalOpenBranch(p[0], p[1], p[2], p[3], p[4], p[5], v1, p[6], v2, 0,
                   o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7);
    if (dynReloc) {
        scatterAddDyn(OB_NOUT, dest + e * OB_NOUT, altDest + e * OB_NOUT, selRow + e * OB_NOUT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n, rt, e * OB_NOUT);
    } else {
        scatterAdd(OB_NOUT, dest + e * OB_NOUT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
}

// Per-scenario shunt admittance override (section shifts / reconnection): entries are
// (slot, PACK elem) pairs with (dG, dB) deltas added onto the static per-element g/b. Linear
// scan per thread — the list is a handful of entries per run.
__device__ inline void applyShuntOverride(int nShOv, const int* shOvSlotElem, const double* shOvGb,
                                          int s, int e, double& g, double& b) {
    for (int k = 0; k < nShOv; k++) {
        if (shOvSlotElem[2 * k] == s && shOvSlotElem[2 * k + 1] == e) {
            g += shOvGb[2 * k];
            b += shOvGb[2 * k + 1];
        }
    }
}

__global__ void fillShuntBatch(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                               const double* in, const int* ix, const int* dest,
                               const double* x, double* vals, double* F,
                               const char* disabled, const char* active,
                               int nShOv, const int* shOvSlotElem, const double* shOvGb,
                               const int* altDest, const int* selRow, const int* mode, int dynReloc,
                               RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 2;                            // g b
    double g = p[0];
    double b = p[1];
    applyShuntOverride(nShOv, shOvSlotElem, shOvGb, s, e, g, b);
    // A voltage-controlling shunt (bcol >= 0) reads its susceptance from the per-scenario state vector; a
    // fixed shunt keeps the packed value (+ any contingency shunt-shift override).
    int bcol = ix[e * 5 + 3];
    if (bcol >= 0) b = x[(size_t) s * n + bcol];
    double v = x[(size_t) s * n + ix[e * 5 + 2]];
    double o[SH_NOUT];
    evalShunt(v, g, b, 0, o + 0, o + 1, o + 2, o + 3, o + 4);
    if (dynReloc) {
        scatterAddDyn(SH_NOUT, dest + e * SH_NOUT, altDest + e * SH_NOUT, selRow + e * SH_NOUT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n, rt, e * SH_NOUT);
    } else {
        scatterAdd(SH_NOUT, dest + e * SH_NOUT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
}

__global__ void fillHvdcBatch(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                              const double* in, const int* ix, const int* dest,
                              const double* x, double* vals, double* F,
                              const char* disabled, const char* active) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 5;                            // p0 k lf1 lf2 r
    const int* q = ix + e * 4;
    const double* xs = x + (size_t) s * n;
    double ph1 = xs[q[2]], ph2 = xs[q[3]];
    double o[HVDC_NOUT];
    evalHvdc(p[0], p[1], p[2], p[3], p[4], ph1, ph2, 0, o + 0, o + 1, o + 2, o + 3, o + 4, o + 5);
    scatterAdd(HVDC_NOUT, dest + e * HVDC_NOUT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
}

// Batched DISTR_Q fill: one thread per (contribution, scenario). Mirrors the single-solve fillDistrQ but
// strides x/vals/F per scenario and DROPS a contribution whose underlying closed (kind 0) / open (kind 2)
// branch is outaged in this scenario — matching the CPU, which removes that branch's q from the controller's
// reactive sum (the controller weight qPercent is unchanged, so only the term vanishes). kinds 1 (shunt) and
// 3 (zero-imp dummy) cannot be a single-branch contingency in a qualified batched scenario, so no mask check.
// ---------------------------------------------------------------------------------
// F-ONLY batch fills: evaluate just the residual values (the generated eval<X>F
// routines) and scatter only the F rows — for iterations that never consume the
// Jacobian (the final residual/convergence fill, and mode 4's solve-only
// iterations where the factors are frozen at iter 0). The residual positions in
// each family's dest stride are pinned to the generated scatter tables by
// static_asserts, so a codegen output-order change fails the build loudly.
// ---------------------------------------------------------------------------------

static_assert(CB_SCATTER[0].role == 0 && CB_SCATTER[7].role == 0
           && CB_SCATTER[14].role == 0 && CB_SCATTER[21].role == 0,
              "closed-branch residual outputs moved — update fillClosedBranchBatchF");
static_assert(OB_SCATTER[0].role == 0 && OB_SCATTER[2].role == 0
           && OB_SCATTER[4].role == 0 && OB_SCATTER[6].role == 0,
              "open-branch residual outputs moved — update fillOpenBranchBatchF");
static_assert(SH_SCATTER[0].role == 0 && SH_SCATTER[2].role == 0,
              "shunt residual outputs moved — update fillShuntBatchF");
static_assert(HVDC_SCATTER[0].role == 0 && HVDC_SCATTER[3].role == 0,
              "hvdc residual outputs moved — update fillHvdcBatchF");

__device__ void scatterAddF(int nres, const int* dest, const int* pos, const double* o, double* F) {
    for (int k = 0; k < nres; k++) {
        int d = dest[pos[k]];
        if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

// Dynamic Q-relocation, residual-only variant: route each residual to its alternate F row when the
// frozen row is an identity this scenario. dest/altDest/selRow are the element's per-output slices;
// pos maps each of the nres residuals to its CB_NOUT output index.
__device__ void scatterAddFDyn(int nres, const int* dest, const int* altDest, const int* selRow,
                               const int* pos, const int* mode, const double* o, double* F,
                               RelocTables rt, int eOff) {
    for (int k = 0; k < nres; k++) {
        int idx = pos[k];
        int d;
        int rg = rt.group != nullptr ? rt.group[eOff + idx] : -1;
        if (rg >= 0) {
            d = rt.pool[rt.off[eOff + idx] + relocChoose(rg, rt.qrPtr, rt.qrRows, mode)];
        } else {
            int sr = selRow[idx];
            d = (sr >= 0 && (mode[sr] == 0 || mode[sr] == 4)) ? altDest[idx] : dest[idx];
        }
        if (d <= -2) {
            atomicAdd(&F[-2 - d], o[k]);
        }
    }
}

__global__ void fillClosedBranchBatchF(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                       const double* in, const int* ix, const int* dest, const double* trig,
                                       const double* x, double* F,
                                       const char* disabled, const char* active,
                                       int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                       const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                       const int* pcDest, const int* pcSel, RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 8;                            // y g1 g2 b1 b2 ksi r1 a1
    const int* q = ix + e * 12;                              // P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
    double pOv[8];
    bool tapOv = findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv);
    if (tapOv) {
        p = pOv;
    }
    const double* xs = x + s * n;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    double sinKsi;
    double cosKsi;
    double sinTheta1;
    double cosTheta1;
    double sinTheta2;
    double cosTheta2;
    closedBranchTrig(trig + e * 6, q[8] >= 0 || tapOv, p[5], a1, ph1, ph2,
                     sinKsi, cosKsi, sinTheta1, cosTheta1, sinTheta2, cosTheta2);
    double o[4];
    evalClosedBranchPreF(p[0], p[1], p[2], p[3], p[4], v1, r1, v2,
                         sinKsi, sinTheta1, cosTheta1, cosKsi, sinTheta2, cosTheta2, 0,
                         o + 0, o + 1, o + 2, o + 3);
    const int pos[4] = {0, 7, 14, 21};
    if (dynReloc) {
        scatterAddFDyn(4, dest + e * CB_NOUT, altDest + e * CB_NOUT, selRow + e * CB_NOUT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, rt, e * CB_NOUT);
    } else {
        scatterAddF(4, dest + e * CB_NOUT, pos, o, F + (size_t) s * n);
    }
    if (pcDest != nullptr) {                                 // continuous phase control: the P residual at a1
        for (int j = 0; j < 4; j += 2) {                     // outputs 0 (p1) / 2 (p2) at table pos 0 / 14
            int d = pcDest[e * CB_NOUT + pos[j]];
            if (d < -1 && mode[(size_t) s * n + pcSel[e * CB_NOUT + pos[j]]] == 3) {
                atomicAdd(F + (size_t) s * n + (-2 - d), o[j]);
            }
        }
    }
}

__global__ void fillOpenBranchBatchF(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                     const double* in, const int* ix, const int* dest,
                                     const double* x, double* F,
                                     const char* disabled, const char* active,
                                     const int* altDest, const int* selRow, const int* mode, int dynReloc,
                                     RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 7;                            // y ksi g1 b1 g2 b2 r1
    const int* q = ix + e * 6;
    const double* xs = x + s * n;
    double v1 = q[4] >= 0 ? xs[q[4]] : 1.0;
    double v2 = q[5] >= 0 ? xs[q[5]] : 1.0;
    double o[4];
    evalOpenBranchF(p[0], p[1], p[2], p[3], p[4], p[5], v1, p[6], v2, 0, o + 0, o + 1, o + 2, o + 3);
    const int pos[4] = {0, 2, 4, 6};
    if (dynReloc) {
        scatterAddFDyn(4, dest + e * OB_NOUT, altDest + e * OB_NOUT, selRow + e * OB_NOUT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, rt, e * OB_NOUT);
    } else {
        scatterAddF(4, dest + e * OB_NOUT, pos, o, F + (size_t) s * n);
    }
}

// Voltage-dependent load model terms (OLF AbstractLoadModelEquationTerm): per entry e,
// F[row] += coef*v^n and J[(row, vOwn)] += coef*n*v^(n-1), coef = c_n * load target (pu),
// exactly OLF's f()/dfdv() (the n = 0 constant part lives in the target vector, not here).
// Identity rows are wiped by applyRowMode afterwards, so a PV bus's Q-row term vanishes
// exactly like every other Q fill (and reappears when RL flips the bus PQ). vals == nullptr
// makes it the residual-only (line-search) variant.
__global__ void fillVdlBatch(int ne, int nScenarios, int n, int nnz,
                             const double* in, const int* ix, const int* dest,
                             const double* x, double* vals, double* F) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ne * nScenarios) return;
    int e = t % ne;
    int s = t / ne;
    // GENERATED evalVdl (codegen family "vdl", parsed from OLF's
    // AbstractLoadModelEquationTerm.f/dfdv): one packed element per exp term, the sum over a
    // load model's terms IS the per-entry atomicAdd — exactly the Java accumulation loop.
    double c = in[e * 3];
    double expn = in[e * 3 + 1];
    double target = in[e * 3 + 2];
    double v = x[(size_t) s * n + ix[e * 2 + 1]];
    double o[2];
    evalVdl(v, target, c, expn, 0, o + 0, o + 1);
    atomicAdd(&F[(size_t) s * n + ix[e * 2]], o[0]);
    if (vals != nullptr) {
        atomicAdd(&vals[(size_t) s * nnz + dest[e]], o[1]);
    }
}

__global__ void fillShuntBatchF(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                const double* in, const int* ix, const int* dest,
                                const double* x, double* F,
                                const char* disabled, const char* active,
                                int nShOv, const int* shOvSlotElem, const double* shOvGb,
                                const int* altDest, const int* selRow, const int* mode, int dynReloc,
                               RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 2;                            // g b
    double g = p[0];
    double b = p[1];
    applyShuntOverride(nShOv, shOvSlotElem, shOvGb, s, e, g, b);
    int bcol = ix[e * 5 + 3];
    if (bcol >= 0) b = x[(size_t) s * n + bcol];             // voltage-controlling shunt: b from state
    double v = x[(size_t) s * n + ix[e * 5 + 2]];
    double o[2];
    evalShuntF(v, g, b, 0, o + 0, o + 1);
    const int pos[2] = {0, 2};
    if (dynReloc) {
        scatterAddFDyn(2, dest + e * SH_NOUT, altDest + e * SH_NOUT, selRow + e * SH_NOUT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, rt, e * SH_NOUT);
    } else {
        scatterAddF(2, dest + e * SH_NOUT, pos, o, F + (size_t) s * n);
    }
}

__global__ void fillHvdcBatchF(int ne, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                               const double* in, const int* ix, const int* dest,
                               const double* x, double* F,
                               const char* disabled, const char* active) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ne * nScenarios) return;
    int e = tid % ne;
    int s = tid / ne;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 5;                            // p0 k lf1 lf2 r
    const int* q = ix + e * 4;
    const double* xs = x + (size_t) s * n;
    double ph1 = xs[q[2]], ph2 = xs[q[3]];
    double o[2];
    evalHvdcF(p[0], p[1], p[2], p[3], p[4], ph1, ph2, 0, o + 0, o + 1);
    const int pos[2] = {0, 3};
    scatterAddF(2, dest + e * HVDC_NOUT, pos, o, F + (size_t) s * n);
}

__global__ void fillDistrQBatch(int ndq, int nScenarios, int n, int nnz, int nElements,
                                int cbOffset, int obOffset,
                                const int* dqElem, const int* dqSide, const int* dqKind,
                                const double* dqWeight, const int* dqDest,
                                const double* cbIn, const int* cbIx, const double* shIn, const int* shIx,
                                const double* obIn, const int* obIx,
                                const double* x, double* vals, double* F,
                                const char* disabled, const char* active,
                                int nDqOv, const int* dqOvIdx, const double* dqOvW,
                                const int* mode,
                                const int* dqKMember, const int* dqJMember, const double* mKey,
                                const int* mRlEntry, const int* mGroupStart, const int* mGroupCount,
                                const char* limitType, int nrl,
                                const int* mQrGroup, const int* qrPtr, const int* qrRows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ndq * nScenarios) return;
    int c = tid % ndq;
    int s = tid / ndq;
    if (!active[s]) return;
    // an RL-PINNED remote member's row (mode 4) hosts its OWN Q balance — its DISTR_Q emissions
    // stop for this scenario (the CPU deactivates only that member's DISTR_Q equation)
    {
        int d0 = dqDest[c * 6];
        if (d0 <= -2 && mode[(size_t) s * n + (-2 - d0)] == 4) return;
        // plan-B4: the CURRENT pilot-Q carrier's row (a re-picked free role after the base free
        // member pinned) hosts the group's relocated pilot balance — its DISTR_Q row is inactive
        // this scenario, exactly like OLF deactivating the new free member's DISTR_Q equation.
        if (mQrGroup != nullptr && dqJMember[c] >= 0) {
            int g = mQrGroup[dqJMember[c]];
            if (g >= 0 && d0 <= -2
                    && -2 - d0 == qrRows[qrPtr[g] + relocChoose(g, qrPtr, qrRows, mode + (size_t) s * n)]) {
                return;
            }
        }
    }
    int e = dqElem[c];
    int kind = dqKind[c];
    if (kind == 0 && disabled[s * nElements + cbOffset + e]) return;   // outaged closed contributor
    if (kind == 2 && disabled[s * nElements + obOffset + e]) return;   // outaged open contributor
    double w = dqWeight[c];
    int kM = dqKMember != nullptr ? dqKMember[c] : -1;
    if (kM >= 0) {
        // DYNAMIC reactive keys (plan-B3): an RL pin disables the member's controller, which zeroes
        // its key and renormalizes qPct over the survivors (updateReactiveKeys). Replicate OLF's
        // zero-then-sum arithmetic in member order, driven by the per-(entry, scenario) limitType.
        int gs = mGroupStart[kM];
        int gc = mGroupCount[kM];
        double sum = 0.0;
        for (int m = gs; m < gs + gc; m++) {
            int re = mRlEntry[m];
            bool pinned = re >= 0 && limitType[(size_t) s * nrl + re] != 0;
            sum += pinned ? 0.0 : mKey[m];
        }
        // the PINNED member's q terms STAY in every survivor's row (weight = the renormalized
        // qPct of the ROW OWNER); the pin is balanced on the TARGET side instead — OLF's
        // getGeneratorReactivePowerDistributionTarget adds qPct_k * targetQ_j for each frozen j
        // (updateDistrQTargetsBatch keeps the per-scenario targets in step with the pin states)
        int reK = mRlEntry[kM];
        bool kPinned = reK >= 0 && limitType[(size_t) s * nrl + reK] != 0;
        double qpk = (sum == 0.0 || kPinned) ? 0.0 : mKey[kM] / sum;
        w = qpk - (dqJMember[c] == kM ? 1.0 : 0.0);
    }
    for (int k = 0; k < nDqOv; k++) {                        // per-scenario qPct renormalization (gen outage)
        if (dqOvIdx[k * 2] == s && dqOvIdx[k * 2 + 1] == c) {
            w = dqOvW[k];
        }
    }
    const int* dst = dqDest + c * 6;
    const double* xs = x + (size_t) s * n;
    double* valsS = vals + (size_t) s * nnz;
    double* FS = F + (size_t) s * n;

    if (kind == 1) {                             // shunt q: {q -> F, dq/dv -> J(row, vcol)}
        const double* sp = shIn + e * 2;
        int vcol = shIx[e * 5 + 2];
        double so[SH_NOUT];
        evalShunt(xs[vcol], sp[0], sp[1], 0, so + 0, so + 1, so + 2, so + 3, so + 4);
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * so[2]);
        if (dst[1] >= 0) atomicAdd(&valsS[dst[1]], w * so[3]);
        return;
    }
    if (kind == 3) {                             // zero-impedance DUMMY_Q: linear w*dummyQ
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * xs[e]);
        if (dst[1] >= 0) atomicAdd(&valsS[dst[1]], w);
        return;
    }
    if (kind == 2) {                             // half-open branch: connected-side q + dq/dv
        const double* op = obIn + e * 7;
        const int* oq = obIx + e * 6;
        int vcol = dqSide[c] == 0 ? oq[4] : oq[5];
        double v1 = dqSide[c] == 0 ? xs[vcol] : 1.0;
        double v2 = dqSide[c] == 1 ? xs[vcol] : 1.0;
        double oo[OB_NOUT];
        evalOpenBranch(op[0], op[1], op[2], op[3], op[4], op[5], v1, op[6], v2, 0,
                       oo + 0, oo + 1, oo + 2, oo + 3, oo + 4, oo + 5, oo + 6, oo + 7);
        int qi = dqSide[c] == 0 ? 6 : 2;
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * oo[qi]);
        if (dst[1] >= 0) atomicAdd(&valsS[dst[1]], w * oo[qi + 1]);
        return;
    }
    // kind 0: closed-branch q1/q2 + its derivatives
    const double* p = cbIn + e * 8;
    const int* q = cbIx + e * 12;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    double o[CB_NOUT];
    evalClosedBranch(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    int qi = dqSide[c] == 0 ? 7 : 21;            // side-1 q1[7]..dq1dr1[13]; side-2 q2[21]..dq2dr1[27]
    double out6[6] = {o[qi], o[qi + 1], o[qi + 2], o[qi + 3], o[qi + 4], o[qi + 6]};
    for (int m = 0; m < 6; m++) {
        int d = dst[m];
        double val = w * out6[m];
        if (d >= 0) atomicAdd(&valsS[d], val);
        else if (d <= -2) atomicAdd(&FS[-2 - d], val);
    }
}

// Batched DISTR_RHO fill: one thread per (contribution, scenario). Linear coef*r1 into the controller's
// BRANCH_RHO1 row — strided per scenario, active-gated, no mask (controller-outage contingencies are
// rejected to the CPU because the 1/count coefficients reform).
__global__ void fillDistrRhoBatch(int ndr, int nScenarios, int n, int nnz,
                                  const int* drCol, const double* drCoef, const int* drDest,
                                  const double* x, double* vals, double* F, const char* active,
                                  int nDrOv, const int* drOvIdx, const double* drOvC,
                                  const int* drQrGroup, const int* mode, const int* qrPtr, const int* qrRows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ndr * nScenarios) return;
    int c = tid % ndr;
    int s = tid / ndr;
    if (!active[s]) return;
    // the CURRENT pilot-Q carrier's row hosts the relocated pilot balance — its DISTR_RHO row is
    // inactive PER SCENARIO (same rule as the sequential fillDistrRho and fillDistrQBatch; the
    // machine's drOv zeroing covers machine steps, but a bare batch solve — the batch-of-1
    // sequential route on a clamp-in-the-base network — has no drOv and needs the in-kernel skip)
    if (drQrGroup != nullptr && drQrGroup[c] >= 0) {
        int g = drQrGroup[c];
        if (-2 - drDest[c * 2] == qrRows[qrPtr[g] + relocChoose(g, qrPtr, qrRows, mode + (size_t) s * n)]) return;
    }
    double coef = drCoef[c];
    for (int k = 0; k < nDrOv; k++) {                        // per-scenario 1/count' renormalization
        if (drOvIdx[k * 2] == s && drOvIdx[k * 2 + 1] == c) {
            coef = drOvC[k];
        }
    }
    int dF = drDest[c * 2 + 0];
    int dJ = drDest[c * 2 + 1];
    if (dF <= -2) atomicAdd(&F[(size_t) s * n + (-2 - dF)], coef * x[(size_t) s * n + drCol[c]]);
    if (dJ >= 0) atomicAdd(&vals[(size_t) s * nnz + dJ], coef);
}

// Batched DISTR_SHUNT_B fill: the SHUNT_B analogue of fillDistrRhoBatch. Linear coef*b into the controller's
// SHUNT_B row, strided per scenario, active-gated.
__global__ void fillDistrShuntBBatch(int ndb, int nScenarios, int n, int nnz,
                                     const int* dbCol, const double* dbCoef, const int* dbDest,
                                     const double* x, double* vals, double* F, const char* active) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ndb * nScenarios) return;
    int c = tid % ndb;
    int s = tid / ndb;
    if (!active[s]) return;
    double coef = dbCoef[c];
    int dF = dbDest[c * 2 + 0];
    int dJ = dbDest[c * 2 + 1];
    if (dF <= -2) atomicAdd(&F[(size_t) s * n + (-2 - dF)], coef * x[(size_t) s * n + dbCol[c]]);
    if (dJ >= 0) atomicAdd(&vals[(size_t) s * nnz + dJ], coef);
}

// Batched zero-impedance fill: one thread per (branch, scenario). Mirrors the single-solve fillZeroImp with
// per-scenario striding — DUMMY_P/Q ±1 couplings into both buses' P/Q balances and, on a spanning-tree edge,
// the ZERO_V (v1−rho·v2) / ZERO_PHI (ph1−ph2) coupling. Active-gated, no mask (bus-coupler outage is rejected).
__global__ void fillZeroImpBatch(int nzi, int nScenarios, int n, int nnz,
                                 const double* in, const int* ix, const int* dest,
                                 const double* x, double* vals, double* F, const char* active,
                                 int nziBase, int nZiOn, const int* ziOnSlotElem,
                                 int nZiOff, const int* ziOffSlotElem,
                                 const int* ziAlt, const int* ziSel, const int* mode, int ziDyn,
                                 RelocTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nzi * nScenarios) return;
    int e = tid % nzi;
    int s = tid / nzi;
    if (!active[s]) return;
    for (int k = 0; k < nZiOff; k++) {                       // per-scenario switch OUTAGE (node-breaker
        if (ziOffSlotElem[2 * k] == s && ziOffSlotElem[2 * k + 1] == e) {   // contingency propagation)
            return;
        }
    }
    if (e >= nziBase) {
        // BASE-OPEN closable switch element: contributes ONLY in scenarios that CLOSE it —
        // sparse (slot, elem) enable list, a handful of entries per run.
        bool on = false;
        for (int k = 0; k < nZiOn; k++) {
            if (ziOnSlotElem[2 * k] == s && ziOnSlotElem[2 * k + 1] == e) {
                on = true;
                break;
            }
        }
        if (!on) return;
    }
    double rho = in[e];
    const int* q = ix + e * 11;                              // b1P b1Q b2P b2Q dpCol dqCol v1 v2 ph1 ph2 tree
    const double* xs = x + (size_t) s * n;
    double* valsS = vals + (size_t) s * nnz;
    double* FS = F + (size_t) s * n;
    const int* d = dest + e * 14;
    double dummyP = xs[q[4]], dummyQ = xs[q[5]];
    int d2 = d[2], d3 = d[3], d6 = d[6], d7 = d[7];
    if (ziDyn) {                                             // dynamic Q-routing (ob/zi channels)
        const int* ms = mode + (size_t) s * n;
        d2 = ziDynDest(d2, e * 4 + 0, ziAlt, ziSel, ms, rt);
        d3 = ziDynDest(d3, e * 4 + 1, ziAlt, ziSel, ms, rt);
        d6 = ziDynDest(d6, e * 4 + 2, ziAlt, ziSel, ms, rt);
        d7 = ziDynDest(d7, e * 4 + 3, ziAlt, ziSel, ms, rt);
    }
    addAt(d[0], dummyP, valsS, FS);                          // F[bus1 P] += dummyP
    addAt(d[1], -dummyP, valsS, FS);                         // F[bus2 P] -= dummyP
    addAt(d2, dummyQ, valsS, FS);                            // F[bus1 Q] += dummyQ
    addAt(d3, -dummyQ, valsS, FS);                           // F[bus2 Q] -= dummyQ
    addAt(d[4], 1.0, valsS, FS);                             // J(bus1 P, dummyP)
    addAt(d[5], -1.0, valsS, FS);                            // J(bus2 P, dummyP)
    addAt(d6, 1.0, valsS, FS);                               // J(bus1 Q, dummyQ)
    addAt(d7, -1.0, valsS, FS);                              // J(bus2 Q, dummyQ)
    if (q[10]) {                                             // spanning-tree edge: voltage/angle coupling
        double v1 = xs[q[6]], v2 = xs[q[7]], ph1 = xs[q[8]], ph2 = xs[q[9]];
        addAt(d[8], v1 - rho * v2, valsS, FS);               // F[ZERO_V row]
        addAt(d[9], 1.0, valsS, FS);                         // J(ZERO_V, v1)
        addAt(d[10], -rho, valsS, FS);                       // J(ZERO_V, v2)
        addAt(d[11], ph1 - ph2, valsS, FS);                  // F[ZERO_PHI row]
        addAt(d[12], 1.0, valsS, FS);                        // J(ZERO_PHI, ph1)
        addAt(d[13], -1.0, valsS, FS);                       // J(ZERO_PHI, ph2)
    }
}

// Per-scenario slack/PV identity rows (rowMode is per scenario: a contingency can change
// which buses are PV/slack). Same mask semantics as the single-solve applyRowMode.
__global__ void applyRowModeBatch(int n, int nScenarios, int nnz, const int* mode, const int* rowPtr,
                                  const int* diagPos, const double* x, const double* target,
                                  double* vals, double* F, const char* active, const double* rowSlope) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int r = tid % n;
    int s = tid / n;
    if (!active[s]) return;
    int m = mode[(size_t) s * n + r];
    double* valsS = vals + (size_t) s * nnz;
    if (m == 0) {                                            // identity: v/phi/rho/a1/b pinned to target
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
            valsS[k] = 0.0;
        }
        valsS[diagPos[r]] = 1.0;
        F[(size_t) s * n + r] = x[(size_t) s * n + r] - target[(size_t) s * n + r];
    } else if (m == 2) {                                     // SLOPE-SVC: V + slope*sumQ = target — the natural
        double slope = rowSlope[r];                          // reactive fill stands, scaled, plus the v term
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
            valsS[k] *= slope;
        }
        valsS[diagPos[r]] += 1.0;
        size_t idx = (size_t) s * n + r;
        F[idx] = slope * (F[idx] + target[idx]) + x[idx] - target[idx];
    }
}

__global__ void initFBatch(int n, int nScenarios, const double* target, double* F, const char* active) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int s = tid / n;
    if (!active[s]) return;
    F[tid] = -target[tid];
}

// Seed the per-(controller, scenario) limitType from the BASE state, so a controller that OLF had already
// switched to PQ in the base case (packed by the extractor as a voltage-CAPABLE PQ bus, base row mode 1)
// starts pinned at the RIGHT limit and can be switched BACK to PV by a relieving contingency. A base-PV
// controller (mode 0) starts at limitType 0. For a base-PQ controller, v below its target ⇒ it is producing
// max reactive yet cannot reach target ⇒ maxQ-pinned (2); v above target ⇒ minQ-pinned (1). switchCount
// starts at 0 either way (a separate memset).
// Per-scenario RL V-target override (secondary voltage control moves generator-VC targets): entries are
// (slot, rl index) pairs with the REPLACEMENT target. Linear scan — the list is a handful of entries.
__device__ inline double rlTargetOf(int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    int s, int i, double baseTarget) {
    for (int k = 0; k < nRlVtOv; k++) {
        if (rlVtOvSlotIdx[2 * k] == s && rlVtOvSlotIdx[2 * k + 1] == i) {
            return rlVtOvVal[k];
        }
    }
    return baseTarget;
}

__global__ void initRlLimitTypeBatch(int nrl, int nScenarios, int n, const int* rlVrow,
                                     const double* rlVtarget, const char* rlMonitored,
                                     const int* baseMode, const double* x, char* limitType,
                                     int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                     const char* rlRemote, const int* rlCmpVrow,
                                     const double* target, const double* rlMinQb, const double* rlMaxQb,
                                     const double* rlLoadQb) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    int vrow = rlVrow[i];
    size_t idx = (size_t) s * n + vrow;
    size_t st = (size_t) s * nrl + i;
    if (rlRemote != nullptr && rlRemote[i]) {
        // REMOTE group member (plan-B3): base mode 1 at its row is the UNPINNED DISTR_Q share —
        // only the canonical mode 4 marks a base pin. The pilot's V equals the group target at the
        // base solution (the survivors keep controlling), so the v-vs-target heuristic is
        // DEGENERATE here — key the direction on WHICH limit the extracted BUS_TARGET_Q target
        // equals instead (target = limit − loadQ was written by the base activation).
        limitType[st] = 0;
        if (baseMode[vrow] == 4) {
            double tgt = target[idx];
            limitType[st] = fabs(tgt - (rlMaxQb[i] - rlLoadQb[i])) <= fabs(tgt - (rlMinQb[i] - rlLoadQb[i]))
                    ? 2 : 1;
        }
        return;
    }
    // Read the BASE row mode (length n, contingency-free) — NOT the per-scenario mode: a controller flipped to
    // PQ by a per-scenario gen-loss OVERRIDE (its generator is gone) has base mode 0 and must get limitType 0 so
    // the reactive-limits switch-back leaves it PQ. Only a controller that is PQ in the BASE (an actual Q-limit
    // switch during the base loadflow) gets a pinned limit type so a relieving contingency can switch it back.
    if (baseMode[vrow] == 1 && rlMonitored[i] == 0) {       // PQ in the base case (a real Q-limit switch —
        // NOT a monitoring SVC standing by: that one seeds UNPINNED so only the VoltageMonitoring
        // orchestrator may activate it, exactly like the CPU's null qLimitType)
        double vt = rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, s, i, rlVtarget[i]);
        limitType[st] = (x[idx] < vt) ? 2 : 1;              // v below target ⇒ maxQ pinned; above ⇒ minQ pinned
    } else {
        limitType[st] = 0;                                  // base PV (incl. a per-scenario gen-override flip)
    }
}

// Reactive-limits (PV↔PQ) pass: thread = (controller, scenario), state per (controller, scenario) in
// limitType (0 PV / 1 minQ / 2 maxQ) + switchCount (PV→PQ transitions, capped to block oscillation).
//  - PV (mode 0): generator q = (Σ branch q at the BUS_V row, in qBuf) + loadQ. If q < minQ−tol, pin to
//    minQ; if q > maxQ+tol, pin to maxQ → flip the row PV→PQ (mode 0→1, target = qLimit − loadQ), record
//    the limit type, increment the switch count.
//  - PQ (mode 1, pinned): if it has not yet switched MAX_SWITCH times, resume voltage control (PQ→PV) when
//    the voltage indicates the limit is no longer binding: a maxQ-pinned bus whose v rose ABOVE target, or
//    a minQ-pinned bus whose v fell BELOW target (mirrors ReactiveLimitsOuterLoop). Restore mode 1→0,
//    target → Vtarget. The count cap (matching OLF's maxPqPvSwitch) bounds oscillation. SLOPE-SVC buses
//    never back-switch (OLF: canSwitchToPv = !hasGeneratorsWithSlope()), and SVC limits are re-evaluated
//    at the live V (B*v^2 quadratic channel) at every check plus the pinned re-freeze.
// Either switch marks the scenario for another inner Newton solve.
// ReactiveCapabilityCurveImpl.getReactiveLimit, verbatim: exact point hit -> the point's value;
// inside [minP,maxP] -> floor/ceiling interpolation; outside -> CLAMP to the first/last point, or
// (extrapolate != 0, OLF's extrapolateReactiveLimits) the ReactiveCapabilityCurveUtil
// .extrapolateReactiveLimitsSlope math — the BOUNDARY two-point slope continued past the range,
// with the intersecting-limits corner (extrapolated minQ > maxQ) collapsing BOTH to their mean.
// sel 0 = minQ, 1 = maxQ. p in MW, result in MVAr.
__device__ double rlCurveEval(double pMW, const double* pts, int npts, int sel, int extrapolate) {
    for (int k = 0; k < npts; k++) {
        if (pts[k * 3] == pMW) {
            return pts[k * 3 + 1 + sel];                     // first case: exact curve point
        }
    }
    if (pMW >= pts[0] && pMW <= pts[(npts - 1) * 3]) {       // second case: interpolate floor..ceiling
        int lo = 0;
        while (lo + 1 < npts && pts[(lo + 1) * 3] < pMW) {
            lo++;
        }
        double p1 = pts[lo * 3];
        double q1 = pts[lo * 3 + 1 + sel];
        double p2 = pts[(lo + 1) * 3];
        double q2 = pts[(lo + 1) * 3 + 1 + sel];
        return q1 + (q2 - q1) / (p2 - p1) * (pMW - p1);
    }
    if (!extrapolate) {
        return pMW < pts[0] ? pts[1 + sel] : pts[(npts - 1) * 3 + 1 + sel];   // third case: clamp
    }
    int bi = pMW < pts[0] ? 0 : npts - 1;                    // pBound = the boundary point,
    int si = pMW < pts[0] ? 1 : npts - 2;                    // pbis = its inward neighbor
    double pB = pts[bi * 3];
    double pS = pts[si * 3];
    double slopeMin = (pts[si * 3 + 1] - pts[bi * 3 + 1]) / (pS - pB);
    double slopeMax = (pts[si * 3 + 2] - pts[bi * 3 + 2]) / (pS - pB);
    double minQ = pts[bi * 3 + 1] + slopeMin * (pMW - pB);
    double maxQ = pts[bi * 3 + 2] + slopeMax * (pMW - pB);
    if (minQ <= maxQ) {
        return sel ? maxQ : minQ;
    }
    return (minQ + maxQ) / 2;                                // intersecting-limits corner: the mean, BOTH
}

// Evaluate the per-scenario DYNAMIC reactive limits (OLF: bus.getMinQ()/getMaxQ() re-read each RL
// check, curve gens at their CURRENT targetP): start from the static pack values and add each
// participating curve gen's delta eval(p_now) - eval(p_base). p_now = basePpu + share * (the bus's
// DS-moved row target - its base) — exact while intra-bus proportionality holds (the pack's DS
// model). At the base point the delta is bit-zero (same formula both sides).
__global__ void rlCurveLimitsBatch(int nrl, int nScenarios, int ncv, int n, int cvExtrapolate,
                                   const int* cvIdx, const double* cvIn, const double* cvPts,
                                   const double* rlMinQ, const double* rlMaxQ, const double* target,
                                   double* minQDyn, double* maxQDyn) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    double mn = rlMinQ[i];
    double mx = rlMaxQ[i];
    for (int g = 0; g < ncv; g++) {
        if (cvIdx[g * 4] != i) continue;
        int pRow = cvIdx[g * 4 + 1];
        const double* pts = cvPts + cvIdx[g * 4 + 2] * 3;
        int npts = cvIdx[g * 4 + 3];
        double pNow = cvIn[g * 5] + cvIn[g * 5 + 1] * (target[(size_t) s * n + pRow] - cvIn[g * 5 + 2]);
        mn += rlCurveEval(pNow * 100.0, pts, npts, 0, cvExtrapolate) / 100.0 - cvIn[g * 5 + 3];
        mx += rlCurveEval(pNow * 100.0, pts, npts, 1, cvExtrapolate) / 100.0 - cvIn[g * 5 + 4];
    }
    minQDyn[(size_t) s * nrl + i] = mn;
    maxQDyn[(size_t) s * nrl + i] = mx;
}

// DISTR_Q target compensation (plan-B3): OLF's DISTR_Q row target is
// (qPct_k - 1)*targetQ_k + qPct_k*sum_{j!=k} targetQ_j, and targetQ_j is nonzero exactly for a
// FROZEN member (pinned at a limit). Recompute every UNPINNED member row's target from the
// per-scenario limitType so an in-flight pin/unpin keeps the targets consistent with the live
// renormalized weights. Idempotent per pass; one thread per (member, scenario).
__global__ void updateDistrQTargetsBatch(int nMembers, int nScenarios, int n, int nrl,
                                         const double* mKey, const int* mRlEntry,
                                         const int* mGroupStart, const int* mGroupCount,
                                         const int* mRow, const char* mHasRow,
                                         const int* rlVrowArr, const double* rlMinQ, const double* rlMaxQ,
                                         const double* rlLoadQ, const char* limitType, int cvStride,
                                         double* target, const char* rlActive,
                                         const int* mQrGroup, const int* qrPtr, const int* qrRows,
                                         const int* mode, const double* pilotQ, int nQr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nMembers * nScenarios) return;
    int k = tid % nMembers;
    int s = tid / nMembers;
    if (!rlActive[s]) return;
    int qg = mQrGroup != nullptr ? mQrGroup[k] : -1;
    if (!mHasRow[k] && qg < 0) return;                       // only members owning a packed DISTR_Q row
                                                             // (or linked to a relocation group, plan-B4)
    int reK = mRlEntry[k];
    if (reK >= 0 && limitType[(size_t) s * nrl + reK] != 0) return;   // pinned: the RL kernel owns its row
    if (qg >= 0) {
        // plan-B4: the member whose row CURRENTLY hosts the relocated pilot Q carries the pilot's
        // Q target (captured at solve init); its DISTR_Q compensation does not apply. A non-carrier
        // member without a packed DISTR_Q row (the base free member, superseded mid-scenario) has
        // no row to maintain here — the RL kernel owns it while pinned.
        int carrier = qrRows[qrPtr[qg] + relocChoose(qg, qrPtr, qrRows, mode + (size_t) s * n)];
        if (mRow[k] == carrier) {
            target[(size_t) s * n + mRow[k]] = pilotQ[(size_t) s * nQr + qg];
            return;
        }
        if (!mHasRow[k]) return;
    }
    int gs = mGroupStart[k];
    int gc = mGroupCount[k];
    double sum = 0.0;
    double frozen = 0.0;
    for (int m = gs; m < gs + gc; m++) {
        int re = mRlEntry[m];
        bool pinned = re >= 0 && limitType[(size_t) s * nrl + re] != 0;
        sum += pinned ? 0.0 : mKey[m];
        if (pinned) {
            size_t li = cvStride ? (size_t) s * nrl + re : (size_t) re;
            char lt = limitType[(size_t) s * nrl + re];
            frozen += (lt == 1 ? rlMinQ[li] : rlMaxQ[li]) - rlLoadQ[re];
        }
    }
    double qpk = sum == 0.0 ? 0.0 : mKey[k] / sum;
    target[(size_t) s * n + mRow[k]] = qpk * frozen;         // 0 when nothing is frozen (the base DISTR_Q target)
}

// plan-B4: capture, per (scenario, relocation group), the pilot's Q target from the BASE carrier's
// row — the base target upload and the per-scenario injection overrides both address the pilot's
// reactive balance at that row (structure-time qRow). updateDistrQTargetsBatch hands the value to
// whichever member row currently carries the relocated pilot Q. Runs BEFORE the RL mode seeding so
// a warm-continuation pin cannot overwrite the value first.
__global__ void capturePilotQBatch(int nQr, int nScenarios, int n, const int* qrPtr, const int* qrRows,
                                   const int* baseMode, const double* target, double* pilotQ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nQr * nScenarios) return;
    int g = tid % nQr;
    int s = tid / nQr;
    int row = qrRows[qrPtr[g] + relocChoose(g, qrPtr, qrRows, baseMode)];
    pilotQ[(size_t) s * nQr + g] = target[(size_t) s * n + row];
}

// DEVICE all-pinned corner (plan-B4 follow-up): when the LAST live member of a remote gen-VC
// relocation group pins (the mode-scan finds no candidate with mode not in {0, 4} and falls back
// to candidate 0 = the pilot's own row), OLF deactivates the pilot's V equation — control lost.
// Flip the pilot row to a POWER row carrying the group's captured pilot-Q target (the scatter
// already lands the pilot's balance there via the same fallback); when a member unpins again,
// restore the V identity and its saved target. Flag-guarded per (scenario, group): only rows
// THIS kernel flipped are ever flipped back — a base-lost-control pilot row is already a power
// row in the pushed base modes and stays untouched. A flip (either direction) marks the scenario
// switched: the row swap needs a re-solve even when it rides a no-switch warm-continuation pass.
__global__ void updatePilotVeqBatch(int nQr, int nScenarios, int n,
                                    const int* qrPtr, const int* qrRows, const char* qrGenVc,
                                    const double* pilotQ, int* mode, double* target,
                                    char* veqOff, double* savedVTarget,
                                    const char* rlActive, char* switched) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nQr * nScenarios) return;
    int g = tid % nQr;
    int s = tid / nQr;
    if (!rlActive[s] || !qrGenVc[g]) return;
    int pilotRow = qrRows[qrPtr[g]];
    size_t pi = (size_t) s * n + pilotRow;
    size_t fi = (size_t) s * nQr + g;
    bool lost = relocChoose(g, qrPtr, qrRows, mode + (size_t) s * n) == 0;
    if (lost && !veqOff[fi] && mode[pi] == 0) {
        savedVTarget[fi] = target[pi];
        mode[pi] = 1;
        target[pi] = pilotQ[fi];
        veqOff[fi] = 1;
        switched[s] = 1;
    } else if (!lost && veqOff[fi]) {
        mode[pi] = 0;
        target[pi] = savedVTarget[fi];
        veqOff[fi] = 0;
        switched[s] = 1;
    }
}

// Remote-member OWN reactive injection (plan-B3): accumulate each remote member's own bus q into
// the diagnostic Q buffer at its OWN row (suppressed by the main qBuf fill), so the RL kernel's
// q = qBuf[row] + loadQ works verbatim for remote entries. Same family evaluation as fillDistrQBatch
// with weight 1 (±1 for zero-impedance DUMMY_Q terms), residual value only (no Jacobian).
__global__ void fillRemoteMemberQBatch(int nrmq, int nScenarios, int n, int nElements,
                                       int cbOffset, int obOffset,
                                       const int* rmqRow, const int* rmqElem, const int* rmqSide,
                                       const int* rmqKind, const double* rmqWeight,
                                       const double* cbIn, const int* cbIx, const double* shIn, const int* shIx,
                                       const double* obIn, const int* obIx,
                                       const double* x, double* qBuf,
                                       const char* disabled, const char* rlActive,
                                       int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                       int nShOv, const int* shOvSlotElem, const double* shOvGb) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrmq * nScenarios) return;
    int c = tid % nrmq;
    int s = tid / nrmq;
    if (!rlActive[s]) return;
    int e = rmqElem[c];
    int kind = rmqKind[c];
    if (kind == 0 && disabled[s * nElements + cbOffset + e]) return;
    if (kind == 2 && disabled[s * nElements + obOffset + e]) return;
    double w = rmqWeight[c];
    const double* xs = x + (size_t) s * n;
    double* q = qBuf + (size_t) s * n + rmqRow[c];
    if (kind == 1) {                                         // fixed shunt q at the member bus
        const double* sp = shIn + e * 2;
        double g = sp[0];
        double b = sp[1];
        applyShuntOverride(nShOv, shOvSlotElem, shOvGb, s, e, g, b);
        int vcol = shIx[e * 5 + 2];
        double so[SH_NOUT];
        evalShunt(xs[vcol], g, b, 0, so + 0, so + 1, so + 2, so + 3, so + 4);
        atomicAdd(q, w * so[2]);
        return;
    }
    if (kind == 3) {                                         // zero-impedance DUMMY_Q linear term
        atomicAdd(q, w * xs[e]);
        return;
    }
    if (kind == 2) {                                         // half-open branch, connected side
        const double* op = obIn + e * 7;
        const int* oq = obIx + e * 6;
        int vcol = rmqSide[c] == 0 ? oq[4] : oq[5];
        double v1 = rmqSide[c] == 0 ? xs[vcol] : 1.0;
        double v2 = rmqSide[c] == 1 ? xs[vcol] : 1.0;
        double oo[OB_NOUT];
        evalOpenBranch(op[0], op[1], op[2], op[3], op[4], op[5], v1, op[6], v2, 0,
                       oo + 0, oo + 1, oo + 2, oo + 3, oo + 4, oo + 5, oo + 6, oo + 7);
        atomicAdd(q, w * oo[rmqSide[c] == 0 ? 6 : 2]);
        return;
    }
    // kind 0: closed branch q at the member's side (full pi row honors the tap override channel)
    const double* p0 = cbIn + e * 8;
    double p[8];
    for (int i = 0; i < 8; i++) {
        p[i] = p0[i];
    }
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) {
        for (int i = 0; i < 8; i++) {
            p[i] = pOv[i];
        }
    }
    const int* qx = cbIx + e * 12;
    double v1 = xs[qx[5]];
    double v2 = xs[qx[7]];
    double ph1 = xs[qx[4]];
    double ph2 = xs[qx[6]];
    double r1 = qx[9] >= 0 ? xs[qx[9]] : p[6];
    double a1 = qx[8] >= 0 ? xs[qx[8]] : p[7];
    double o[CB_NOUT];
    evalClosedBranch(p[0], p[1], p[2], p[3], p[4], v1, r1, v2, p[5], a1, ph1, ph2, 0,
                     o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                     o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                     o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
    atomicAdd(q, w * o[rmqSide[c] == 0 ? 7 : 21]);           // q1 / q2 residual outputs
    return;
}

__global__ void reactiveLimitsBatch(int nrl, int nScenarios, int n, double tol, int maxSwitch,
                                    const int* rlVrow, const double* rlMinQ, const double* rlMaxQ,
                                    const double* rlLoadQ, const double* rlVtarget, const double* qBuf,
                                    const double* x, const char* rlActive, int* mode, double* target,
                                    char* limitType, char* switchCount, char* switched,
                                    int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    const double* rowSlope, double* savedTarget, int cvStride,
                                    const char* rlRemote, const int* rlCmpVrow,
                                    const double* rlSvcBmin, const double* rlSvcBmax, const double* rlV2Base) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    if (!rlActive[s]) return;
    // SVC reactive limits are VOLTAGE-DEPENDENT: LfStaticVarCompensatorImpl evaluates
    // Bmin/Bmax*(v*nomV)^2 at the LIVE bus V on every CPU check (and a pinned bus re-freezes at
    // the moved limit via busesWithUpdatedQLimits). rlMinQ/rlMaxQ hold the base-V aggregates;
    // the per-entry svcB coefficients add the quadratic delta vs the packed base v^2 — exact for
    // the SVC part, identically zero on SVC-less buses.
    bool svcLim = rlSvcBmin[i] != 0.0 || rlSvcBmax[i] != 0.0;
    if (rlRemote != nullptr && rlRemote[i]) {
        // REMOTE group member (plan-B3). Unpinned = its DISTR_Q row (mode 1, limitType 0);
        // pinned = mode 4 ("row hosts its own Q balance"). BOTH switch conditions test the
        // PILOT's voltage (rlCmpVrow) against the GROUP target — B0-verified CPU semantics.
        // The FREE member pins the same way (plan-B4): the fill kernels re-pick the pilot-Q
        // carrier by mode-scan over the relocation candidates, and updateDistrQTargetsBatch
        // hands the pilot's Q target to the new carrier's row. When the LAST live member pins,
        // updatePilotVeqBatch flips the pilot's V identity to a power row (control lost) and
        // restores it on the first unpin — the all-pinned corner runs on-device too.
        size_t ridx = (size_t) s * n + rlVrow[i];
        size_t rst = (size_t) s * nrl + i;
        size_t rli = cvStride ? rst : (size_t) i;
        double rdv2 = x[ridx] * x[ridx] - rlV2Base[i];
        double rliMin = rlMinQ[rli] + rlSvcBmin[i] * rdv2;
        double rliMax = rlMaxQ[rli] + rlSvcBmax[i] * rdv2;
        if (limitType[rst] == 0) {
            if (mode[ridx] != 1) return;                     // islanded / frozen this scenario
            double q = qBuf[ridx] + rlLoadQ[i];              // own q via fillRemoteMemberQBatch
            if (q < rliMin - tol) {
                savedTarget[rst] = target[ridx];             // the DISTR_Q row target (0)
                mode[ridx] = 4;
                target[ridx] = rliMin - rlLoadQ[i];
                limitType[rst] = 1;
                switchCount[rst]++;
                switched[s] = 1;
            } else if (q > rliMax + tol) {
                savedTarget[rst] = target[ridx];
                mode[ridx] = 4;
                target[ridx] = rliMax - rlLoadQ[i];
                limitType[rst] = 2;
                switchCount[rst]++;
                switched[s] = 1;
            }
        } else if (switchCount[rst] < maxSwitch) {           // pinned — switch-back tests the PILOT
            double v = x[(size_t) s * n + rlCmpVrow[i]];
            double vt = rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, s, i, rlVtarget[i]);
            bool back = (limitType[rst] == 2 && v > vt)
                     || (limitType[rst] == 1 && v < vt);
            if (back) {
                mode[ridx] = 1;                              // back to the DISTR_Q share row
                target[ridx] = savedTarget[rst];
                limitType[rst] = 0;
                switched[s] = 1;
            } else if (cvStride || svcLim) {                 // curve/SVC limit drift while pinned
                double lim = limitType[rst] == 1 ? rliMin : rliMax;
                if (fabs(lim - (target[ridx] + rlLoadQ[i])) > tol) {
                    target[ridx] = lim - rlLoadQ[i];
                    switched[s] = 1;
                }
            }
        }
        return;
    }
    int vrow = rlVrow[i];
    size_t idx = (size_t) s * n + vrow;
    size_t st = (size_t) s * nrl + i;
    size_t li = cvStride ? st : (size_t) i;                 // curve limits are per (scenario, entry)
    bool slopeRow = rowSlope[vrow] != 0;                    // SLOPE-SVC bus: its PV form is row mode 2
    double dv2 = x[idx] * x[idx] - rlV2Base[i];
    double liMin = rlMinQ[li] + rlSvcBmin[i] * dv2;
    double liMax = rlMaxQ[li] + rlSvcBmax[i] * dv2;
    if ((mode[idx] == 0 || mode[idx] == 2) && limitType[st] == 0) {   // currently PV — check the reactive limits
        double q = qBuf[idx] + rlLoadQ[i];
        if (q < liMin - tol) {
            savedTarget[st] = target[idx];                   // remember the row target for the switch-back
            mode[idx] = 1;
            target[idx] = liMin - rlLoadQ[i];
            limitType[st] = 1;
            switchCount[st]++;
            switched[s] = 1;
        } else if (q > liMax + tol) {
            savedTarget[st] = target[idx];
            mode[idx] = 1;
            target[idx] = liMax - rlLoadQ[i];
            limitType[st] = 2;
            switchCount[st]++;
            switched[s] = 1;
        }
    } else if (limitType[st] != 0 && switchCount[st] < maxSwitch) {   // pinned at a limit — check switchback
        double v = x[idx];
        double vt = rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, s, i, rlVtarget[i]);
        // OLF never back-switches a slope-SVC bus: checkPqBus runs with
        // canSwitchToPv = !bus.hasGeneratorsWithSlope() ("we don't support switching PQ to PV
        // for bus with one controller with slope") — once pinned, a slope row stays PQ.
        bool back = !slopeRow
                && ((limitType[st] == 2 && v > vt)                   // maxQ-pinned but v above target
                 || (limitType[st] == 1 && v < vt));                 // minQ-pinned but v below target
        if (back) {
            mode[idx] = 0;                                  // PQ → PV: resume voltage control
            target[idx] = vt;
            limitType[st] = 0;
            switched[s] = 1;
        } else if (cvStride || svcLim) {
            // Q-limit UPDATE while pinned (OLF checkPqBusWithQLimitType): a CURVE limit that moved
            // (DS moved the gen's P) or an SVC limit that moved (B*v^2 at the live V) re-freezes
            // the pinned Q at the NEW limit when it drifted by more than the mismatch tol.
            // Unreachable with static limits — gated on the channels.
            double lim = limitType[st] == 1 ? liMin : liMax;
            if (fabs(lim - (target[idx] + rlLoadQ[i])) > tol) {
                target[idx] = lim - rlLoadQ[i];
                switched[s] = 1;
            }
        }
    }
}

// Distributed-slack pass: thread = (participating bus, scenario). The slack mismatch is
// qBuf[slackPhiRow] (= Σ branch p at the slack) − slackTargetP. If it exceeds the tolerance, each
// participating bus takes factor·mismatch into its BUS_TARGET_P target (factor normalized to sum 1, so
// the whole mismatch is absorbed), mirroring ActivePowerDistribution (one-shot, no Pmin/Pmax saturation
// — exact vs OLF when no generator saturates). A distribution marks its scenario for another inner solve.
// Freeze, on the FIRST distribution pass of a solve, the active-power share the slack/reference bus must
// RETAIN when its own generators participate: retain = (1 − Σfactor)·mismatch0. Σfactor over the emitted
// (non-slack) buses is < 1 exactly when the slack participates (its factor sits in the normalization
// denominator but emits no row); 1 − Σfactor is the slack's participation fraction ρ. Without this the
// geometric outer loop would drive the slack mismatch to 0 and leak the slack's whole share to the
// non-slack buses. ρ = 0 (non-participating slack) ⇒ retain = 0 ⇒ the slack keeps nothing extra.
// The slack/reference bus retains ρ = (1 − Σfactor) of the TOTAL active imbalance currently being shared,
// where total = the slack's residual mismatch (qBuf at slackPhiRow − slackTargetP) PLUS what the non-slack
// participating buses have already absorbed (Σ of their cumulative target deltas). Re-evaluated EVERY outer
// pass: freezing ρ·mismatch0 at the first pass ignores the AC loss change during redistribution, so the
// slack would keep ρ·mismatch0 and the non-slack side would over-absorb by ρ·Δlosses; tracking the current
// total instead makes the non-slack buses absorb exactly (1 − ρ)·total_final, matching OLF's per-pass
// distribution of the full mismatch. ρ = 0 (non-participating slack, incl. all batched gen-loss scenarios)
// ⇒ retain = 0, so this is a no-op there.
__global__ void setUnitRhsBatch(int n, int nScenarios, int unitRow, double* b) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nScenarios) return;
    b[t] = (t % n == unitRow) ? 1.0 : 0.0;
}

__global__ void gatherSensiBatch(int n, int nScenarios, int k, int m, int ki, const int* observeCols,
                                 const double* dx, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;                     // t = s*m + mi
    if (t >= nScenarios * m) return;
    int sIdx = t / m;
    int mi = t % m;
    out[(size_t) sIdx * k * m + (size_t) ki * m + mi] = dx[(size_t) sIdx * n + observeCols[mi]];
}

// Slack-only distributed slack: the slack generator is the ONLY participant, so OLF's distribution
// puts the whole mismatch into the slack gen's own target — no flow change, but the outer loop reports
// UNSTABLE and triggers a re-solve that polishes the state by one more Newton run. Mirror it exactly:
// track the absorbed total, flag a switch while the residual mismatch stays above the outer gate.
__global__ void slackOnlyDsCheck(int nScenarios, int n, int slackPhiRow, double slackTargetP,
                                 double slackDistTol, const double* qBuf, const char* rlActive,
                                 double* absorbed, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !rlActive[s]) return;
    double m = qBuf[(size_t) s * n + slackPhiRow] - slackTargetP - absorbed[s];
    if (fabs(m) > slackDistTol) {
        absorbed[s] += m;
        switched[s] = 1;
    }
}

__global__ void computeSlackRetain(int nScenarios, int n, int nds, int slackPhiRow, double slackTargetP,
                                   double oneMinusFactorSum, const int* dsPhiRow, const double* dsBaseTargetP,
                                   const double* target, const double* qBuf, const char* rlActive,
                                   double* retain,
                                   const int* dsOvBus, const double* dsOvBaseDelta,
                                   const double* dsAnchorShift, const double* dsSlackShift,
                                   int dsMode, const double* dsMarginUp, const double* dsMarginDown,
                                   double slackMarginUp, double slackMarginDown) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!rlActive[s]) return;
    double absorbed = 0.0;
    int ov = dsOvBus != nullptr ? dsOvBus[s] : -1;
    for (int i = 0; i < nds; i++) {
        // A composed injection override on a participant bus (redispatch action, load delta) shifts
        // the row target WITHOUT being distributed power — the effective base must shift with it,
        // or the retain math counts the override as absorbed slack and the waterfill CANCELS it
        // (measured: a +60 MW redispatch read back as a no-op, GeneratorAction drill 2026-07-19).
        double base = dsBaseTargetP[i];
        if (dsAnchorShift != nullptr) {
            base += dsAnchorShift[(size_t) s * nds + i];     // pair leg: cumulative reference re-anchored at the
        }                                                    // PARENT contingency's distributed targets (OLF's
        if (ov == i) {                                       // setGeneratorsInitialTargetPToTargetP before the
            base -= dsOvBaseDelta[s];                        // strategy re-solve)
        }
        absorbed += target[(size_t) s * n + dsPhiRow[i]] - base;
    }
    double slackTgt = slackTargetP + (dsSlackShift != nullptr ? dsSlackShift[s] : 0.0);
    double tot = (qBuf[(size_t) s * n + slackPhiRow] - slackTgt) + absorbed;   // total-from-anchor, OLF's
    double rho = oneMinusFactorSum;                                            // remainingMismatch at run() start
    if (dsMode == 1) {
        // REMAINING_MARGIN: the slack's retained share is its own margin share in the DIRECTION of the
        // total-from-anchor (OLF evaluates the direction-dependent factors once per run at the anchor).
        // Margins shift with a moved anchor: +delta of anchor target eats headroom, frees footroom.
        double slackM = tot > 0 ? slackMarginUp : slackMarginDown;
        if (dsSlackShift != nullptr) {
            slackM = fmax(0.0, slackM - (tot > 0 ? dsSlackShift[s] : -dsSlackShift[s]));
        }
        double sum = slackM;
        for (int i = 0; i < nds; i++) {
            double m = tot > 0 ? dsMarginUp[i] : dsMarginDown[i];
            if (dsAnchorShift != nullptr) {
                double sh = dsAnchorShift[(size_t) s * nds + i];
                m = fmax(0.0, m - (tot > 0 ? sh : -sh));
            }
            sum += m;
        }
        rho = sum > 0 ? slackM / sum : 0.0;
    }
    retain[s] = rho * tot;
}

// Distribute the slack mismatch by FULL WATER-FILLING per scenario (one thread = one scenario), matching OLF's
// ActivePowerDistribution.run + GenerationActivePowerDistributionStep.run: distribute the whole mismatch among
// the participating buses, re-normalizing the factors over the UNSATURATED buses each round and removing buses
// that hit a limit, iterating until the residual falls below residueEps (P_RESIDUE_EPS) or all buses saturate —
// all WITHOUT a power-flow re-solve. The full water-fill at a FIXED operating point is what matches OLF under
// saturation: a single clamp step per outer pass (letting the saturated residual reappear at the slack, to be
// redistributed only AFTER the next re-solve) would drift — the loss change between re-solves converges to a
// different in-tolerance point than OLF.
__global__ void distributeSlackBatch(int nds, int nScenarios, int n, double slackDistTol, double residueEps,
                                              int slackPhiRow, double slackTargetP, const int* dsPhiRow,
                                              const double* dsFactor, const double* dsBaseTargetP, const double* dsAnchorShift, const double* dsSlackShift,
                                              const double* dsMaxDelta, const double* dsMinDelta, const double* dsRetain,
                                              const double* qBuf, const char* rlActive,
                                              const int* dsOvBus, const double* dsOvFactorDelta, const double* dsOvBaseDelta,
                                              const double* dsOvMaxDelta, const double* dsOvMinDelta,
                                              int dsMode, const double* dsMarginUp, const double* dsMarginDown,
                                              double slackMarginUp, double slackMarginDown,
                                              double* target, char* switched, char* dsResidual) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!rlActive[s]) return;
    // The participating buses share the mismatch in EXCESS of the slack's retained share (rem); only distribute
    // past the slackBusPMaxMismatch outer gate, exactly like OLF's DistributedSlackOuterLoop.check.
    double slackTgt = slackTargetP + (dsSlackShift != nullptr ? dsSlackShift[s] : 0.0);
    double rem = (qBuf[(size_t) s * n + slackPhiRow] - slackTgt) - dsRetain[s];
    int ov = (dsOvBus != nullptr) ? dsOvBus[s] : -1;        // the bus losing a participating gen (gen contingency), -1 = none
    // REMAINING_MARGIN: factors are the per-bus margins in the DIRECTION of the pass's remaining mismatch,
    // picked ONCE at entry (OLF evaluates them once per run at the anchor); a moved anchor (+delta) eats
    // headroom and frees footroom symmetrically.
    bool dirUp = rem > 0;
    // rem is the NON-SLACK share of the CPU's slack mismatch: rem = (1 - rho) * m_cpu, with rho the slack's
    // GLOBAL retained share (the same constant the retain kernel uses — a per-scenario participant reweight
    // does not change it). OLF's outer gate tests |m_cpu| > slackBusPMaxMismatch, so the device gate scales
    // by (1 - rho) = the GLOBAL participant factor sum: unscaled it is 1/(1-rho) looser and lets the device
    // retire with up to (1/(1-rho) - 1) * tol still undistributed (measured 4.9e-4 kV on a 60 MW redispatch
    // pair with a participating slack, 2x); scaling by the per-scenario EFFECTIVE sum instead over-tightens
    // a gen-outage scenario with a non-participating slack and adds a round the CPU never runs (3.9e-4).
    double fsGlobal = 0.0;
    if (dsMode == 1) {
        double sumM = 0.0;
        for (int i = 0; i < nds; i++) {
            double m = dirUp ? dsMarginUp[i] : dsMarginDown[i];
            if (dsAnchorShift != nullptr) {
                double sh = dsAnchorShift[(size_t) s * nds + i];
                m = fmax(0.0, m - (dirUp ? sh : -sh));
            }
            sumM += m;
        }
        double slackM = dirUp ? slackMarginUp : slackMarginDown;
        if (dsSlackShift != nullptr) {
            slackM = fmax(0.0, slackM - (dirUp ? dsSlackShift[s] : -dsSlackShift[s]));
        }
        fsGlobal = sumM + slackM > 0 ? sumM / (sumM + slackM) : 0.0;
    } else {
        for (int i = 0; i < nds; i++) {
            fsGlobal += dsFactor[i];
        }
    }
    if (fsGlobal <= 0.0 || fabs(rem) <= fsGlobal * slackDistTol) {
        return;
    }
    bool changed = false;
    for (int iter = 0; iter < nds && fabs(rem) > residueEps; iter++) {
        double fsumActive = 0.0;                            // re-normalize factors over the still-unsaturated buses
        for (int i = 0; i < nds; i++) {
            double f = dsFactor[i], base = dsBaseTargetP[i], maxD = dsMaxDelta[i], minD = dsMinDelta[i];
            double sh = dsAnchorShift != nullptr ? dsAnchorShift[(size_t) s * nds + i] : 0.0;
            if (dsAnchorShift != nullptr) { base += sh; maxD -= sh; minD -= sh; }
            if (dsMode == 1) { f = fmax(0.0, (dirUp ? dsMarginUp[i] : dsMarginDown[i]) - (dirUp ? sh : -sh)); }
            if (ov == i) { f -= dsOvFactorDelta[s]; base -= dsOvBaseDelta[s]; maxD -= dsOvMaxDelta[s]; minD -= dsOvMinDelta[s]; }
            double cum = target[(size_t) s * n + dsPhiRow[i]] - base;
            bool sat = (rem > 0) ? (cum >= maxD) : (cum <= minD);
            if (!sat) {
                fsumActive += f;
            }
        }
        if (fsumActive <= 0.0) {
            break;                                          // every participating bus saturated — residual stays at the slack
        }
        double done = 0.0;
        for (int i = 0; i < nds; i++) {
            double f = dsFactor[i], base = dsBaseTargetP[i], maxD = dsMaxDelta[i], minD = dsMinDelta[i];
            double sh = dsAnchorShift != nullptr ? dsAnchorShift[(size_t) s * nds + i] : 0.0;
            if (dsAnchorShift != nullptr) { base += sh; maxD -= sh; minD -= sh; }
            if (dsMode == 1) { f = fmax(0.0, (dirUp ? dsMarginUp[i] : dsMarginDown[i]) - (dirUp ? sh : -sh)); }
            if (ov == i) { f -= dsOvFactorDelta[s]; base -= dsOvBaseDelta[s]; maxD -= dsOvMaxDelta[s]; minD -= dsOvMinDelta[s]; }
            size_t idx = (size_t) s * n + dsPhiRow[i];
            double cum = target[idx] - base;
            bool sat = (rem > 0) ? (cum >= maxD) : (cum <= minD);
            if (sat) {
                continue;
            }
            double newCum = fmin(fmax(cum + rem * (f / fsumActive), minD), maxD);
            if (newCum != cum) {
                target[idx] = base + newCum;
                done += newCum - cum;
                changed = true;
            }
        }
        rem -= done;
    }
    if (changed) {
        switched[s] = 1;
    }
    if (fabs(rem) > fsGlobal * slackDistTol) {
        dsResidual[s] = 1;                                   // saturated: residual stays at the slack — the host
    }                                                        // decides per the run's slackDistributionFailureBehavior
}

__global__ void negateIntoOne(int n, const double* F, double* b) {       // single scenario slice
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = -F[i];
}

__global__ void axpy1One(int n, const double* dx, double* x) {           // single scenario slice
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += dx[i];
}

__global__ void negateIntoBatch(int total, const double* F, double* b) {  // b = -F over all scenarios
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) b[i] = -F[i];
}

// Warm-continuation reconcile: a SEEDED pinned controller (limitType != 0) must also carry its
// consequences — the BUS_V row flips to the reactive equation (mode 1) with the pinned Q target —
// else the solve holds the PV setpoint while the RL bookkeeping thinks the bus is at a limit.
__global__ void seedRlModeFromState(int nrl, int nScenarios, int n, const int* rlVrow,
                                    const double* rlMinQ, const double* rlMaxQ, const double* rlLoadQ,
                                    const char* limitType, int* mode, double* target, int cvStride,
                                    const char* rlRemote) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    char lt = limitType[(size_t) s * nrl + i];
    if (lt != 0) {
        size_t idx = (size_t) s * n + rlVrow[i];
        size_t li = cvStride ? (size_t) s * nrl + i : (size_t) i;
        // a pinned REMOTE member's row is mode 4 (own Q balance), a pinned local one mode 1
        mode[idx] = (rlRemote != nullptr && rlRemote[i]) ? 4 : 1;
        target[idx] = (lt == 1 ? rlMinQ[li] : rlMaxQ[li]) - rlLoadQ[i];
    }
}

__global__ void axpy1Batch(int total, const double* dx, double* x) {      // x += dx over all scenarios
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += dx[i];
}

// Per-scenario MAX_VOLTAGE_CHANGE step factor (mirrors MaxVoltageChangeStateVectorScaling, exactly
// like the base solver's host-side clamp but one factor PER SCENARIO): the largest |Δv| (varType 1)
// stays within maxDv and the largest |Δφ| (varType 0) within maxDphi; the WHOLE step is scaled.
// One block per scenario, block-stride column loop, shared-memory max reduction.
__global__ void maxVoltageChangeFactorBatch(int n, const double* dx, const int* varType,
                                            const char* active, double maxDv, double maxDphi,
                                            double* factor) {
    int s = blockIdx.x;
    __shared__ double smDv[BLOCK];
    __shared__ double smDphi[BLOCK];
    double mdv = 0.0;
    double mdphi = 0.0;
    if (active == nullptr || active[s]) {
        const double* d = dx + (size_t) s * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double a = fabs(d[i]);
            if (varType[i] == 1) {
                mdv = fmax(mdv, a);
            } else if (varType[i] == 0) {
                mdphi = fmax(mdphi, a);
            }
        }
    }
    smDv[threadIdx.x] = mdv;
    smDphi[threadIdx.x] = mdphi;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            smDv[threadIdx.x] = fmax(smDv[threadIdx.x], smDv[threadIdx.x + off]);
            smDphi[threadIdx.x] = fmax(smDphi[threadIdx.x], smDphi[threadIdx.x + off]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double f = 1.0;
        if (smDv[0] > maxDv) f = fmin(f, maxDv / smDv[0]);
        if (smDphi[0] > maxDphi) f = fmin(f, maxDphi / smDphi[0]);
        factor[s] = f;
    }
}

__global__ void axpy1BatchScaled(int n, int total, const double* dx, const double* factor, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += factor[i / n] * dx[i];
}

// Per-scenario convergence norms on DEVICE: out[s*3] = ||F||inf, out[s*3+1] = sum(F^2),
// out[s*3+2] = finite flag — so the per-iteration readback is nS*3 doubles instead of the full
// nS*n mismatch (70 MB at 500 x 18k, pageable ~5.5 GB/s + a 9.2M-element host loop). infnorm is
// exact vs the host loop; the ssq PARALLEL summation order differs from the host's sequential
// sum by last-ulp amounts — borderline ||F||2 stopping decisions can flip at ulp level (same
// class of effect as any reduction-order change; the criterion itself is unchanged).
__global__ void residualNormsBatch(int n, const double* F, const char* active, double* out,
                                   const double* thrIdent, const double* thrPower, const int* mode) {
    int s = blockIdx.x;
    __shared__ double smInf[BLOCK];
    __shared__ double smSsq[BLOCK];
    __shared__ double smRat[BLOCK];
    __shared__ int smFin[BLOCK];
    double mInf = 0.0;
    double ssq = 0.0;
    double mRat = 0.0;
    int fin = 1;
    if (active == nullptr || active[s]) {
        const double* f = F + (size_t) s * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double v = f[i];
            if (!isfinite(v)) fin = 0;
            mInf = fmax(mInf, fabs(v));
            ssq += v * v;
            if (thrIdent != nullptr) {
                // PER_EQUATION_TYPE: identity/slope rows (mode 0/2) use the row's identity-type
                // threshold; power rows (mode 1), the phase P row (mode 3) and the pinned
                // remote-member Q row (mode 4) the power-type one
                int m = mode[(size_t) s * n + i];
                double thr = (m == 1 || m == 3 || m == 4) ? thrPower[i] : thrIdent[i];
                mRat = fmax(mRat, fabs(v) / thr);
            }
        }
    }
    smInf[threadIdx.x] = mInf;
    smSsq[threadIdx.x] = ssq;
    smRat[threadIdx.x] = mRat;
    smFin[threadIdx.x] = fin;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            smInf[threadIdx.x] = fmax(smInf[threadIdx.x], smInf[threadIdx.x + off]);
            smSsq[threadIdx.x] += smSsq[threadIdx.x + off];
            smRat[threadIdx.x] = fmax(smRat[threadIdx.x], smRat[threadIdx.x + off]);
            smFin[threadIdx.x] &= smFin[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[s * 4 + 0] = smInf[0];
        out[s * 4 + 1] = smSsq[0];
        out[s * 4 + 2] = smFin[0] ? 1.0 : 0.0;
        out[s * 4 + 3] = smRat[0];                           // max |F_i|/thr_i (0 without thresholds)
    }
}

// Batch COMPACTION (solve-only): gather the active scenarios' per-scenario blocks into a
// contiguous compact buffer (dst[c*per+i] = src[map[c]*per+i]) so cuDSS runs a NARROWER ubatch,
// then scatter the solution back to the full-width slots. Everything else (fills, norms,
// updates) stays full-width — retirement already makes those cheap; cuDSS width was the cost.
__global__ void gatherScenarios(long per, int nC, const int* map, const double* src, double* dst) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= per * nC) return;
    int c = (int) (i / per);
    dst[i] = src[(long) map[c] * per + i % per];
}

__global__ void scatterScenarios(long per, int nReal, const int* map, const double* srcC, double* dst) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= per * nReal) return;                             // pad slots (c >= nReal) are DISCARDED
    int c = (int) (i / per);
    dst[(long) map[c] * per + i % per] = srcC[i];
}

// Tile the base row (row 0, already in d[0..n)) across all nS scenarios: d[s*n+i] = d[i] for
// s in 1..nS-1. The base state/targets/row-mode are identical for every contingency, so the host
// uploads ONE n-row and this replicates it on-device — no 32x JNI marshalling / H2D of the base.
template <typename T>
__global__ void tileBaseRow(int total, int n, T* d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n && idx < total) {                       // rows 1..nS-1 copy from row 0; row 0 is the source
        d[idx] = d[idx % n];
    }
}

// Make a RETIRED scenario's system an identity (diagonal 1, mismatch 0) so it stays
// well-posed inside the uniform batched factorization (a singular sub-block would fail the
// whole batch) and its state is frozen (dx = 0). A scenario retired THIS iteration still
// holds its filled values, so the row is zeroed before the diagonal is set — this is
// correct whether the scenario was just filled or has been inactive (zero) for a while.
__global__ void writeIdentityForInactive(int n, int nScenarios, int nnz, const int* rowPtr,
                                         const int* diagPos, const char* active, double* vals, double* F) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * nScenarios) return;
    int r = tid % n;
    int s = tid / n;
    if (active[s]) return;
    double* valsS = vals + (size_t) s * nnz;
    for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
        valsS[k] = 0.0;
    }
    valsS[diagPos[r]] = 1.0;
    F[(size_t) s * n + r] = 0.0;
}

// The batched solver context: the SHARED structure (pattern, families, destination arrays,
// static trig — shared by shared_ptr with any single-solve context on the same extraction,
// see SharedStructure) + per-scenario value/state arrays and this consumer's own cuDSS
// objects/analysis. Structure notes that are batch-specific:
// - DISTR_Q: the controller structure is fixed across scenarios; the per-scenario disable
//   mask drops an outaged contributor's term (kinds 0/2 = closed/open).
// - DISTR_RHO: no per-scenario mask — a contingency that outages a controlling transformer
//   is rejected to the CPU (the 1/count coefficients would reform).
// - Zero-impedance: no per-scenario mask — a zero-imp outage is auto-rejected (the element
//   mapper maps it to -1), so the fill just replicates the base.
struct BatchSolverContext {
    SharedStructurePtr st;                                   // shared with any single-solve context on the same extraction
    int n = 0;
    int nnz = 0;
    int nScenarios = 0;
    int nElements = 0;
    int cbOffset = 0, obOffset = 0, shOffset = 0, hvOffset = 0;   // element-mask offsets per family
    int ndq = 0, ndr = 0, ndb = 0, nzi = 0;                  // copies of the structure counts
    DeviceBuf dX, dTarget, dF, dB, dDx, dVals, dMode;        // per-scenario (n or nnz strided)
    bool clampStep = false;                                  // MAX_VOLTAGE_CHANGE per-scenario step clamp (set per solve)
    double maxDv = 0.0, maxDphi = 0.0;
    DeviceBuf dVarType, dStepFactor;                         // n ints / nS factors (lazy, first clamped solve)
    DeviceBuf dXprevLs;                                      // n*nS pre-step iterates (lazy, LINE_SEARCH solves)
    int lsMaxIter = 10;                                      // LINE_SEARCH backtracks (setBatchLineSearch)
    double lsStepFold = 4.0 / 3.0;
    DeviceBuf dNorms;                                        // nS*4 device-side convergence norms (see residualNormsBatch)
    DeviceBuf dThrIdent, dThrPower;                          // n each: per-row PER_EQUATION_TYPE thresholds (empty = 2-norm/infnorm criterion)
    bool perRowCriteria = false;
    int nShOv = 0;                                           // per-scenario shunt admittance overrides (set per solve)
    DeviceBuf dShOvSlotElem, dShOvGb;                        // nShOv*2 ints (slot, PACK elem) / nShOv*2 doubles (dG, dB)
    int nziBase = 0;                                         // zi elements [0,nziBase) always enabled; [nziBase,nzi) are
                                                             // base-open closable switches gated by the ziOn list
    int nZiOn = 0;
    int nZiOff = 0;
    DeviceBuf dZiOffSlotElem;                                // nZiOff*2 ints (slot, zi elem) — scenarios that OPEN a switch
    DeviceBuf dZiOnSlotElem;                                 // nZiOn*2 ints (slot, zi elem) — scenarios that CLOSE a switch
    int nR1Ov = 0;                                           // per-scenario closed-branch tap overrides (tap/PST actions)
    DeviceBuf dR1OvSlotElem, dR1OvVal;                       // nR1Ov*2 ints / nR1Ov*8 doubles (full cbIn row per entry)
    // ---- solve-only batch compaction (mode 3, tol > 0): lazily allocated on first engage ----
    DeviceBuf dValsC, dBC, dDxC, dMapC;                      // nnz*cap / n*cap / n*cap / cap ints
    CudssCtx dsC;                                            // dedicated handle/config/data for the compact width
    cudssMatrix_t matUbC = nullptr, matUbCB = nullptr, matUbCX = nullptr;
    int compactCap = 0;                                      // buffer capacity (nS/2)
    int compactWidth = 0;                                    // currently ANALYZED ubatch width (0 = none)
    DeviceBuf dDisabled, dActive;                            // per-scenario masks
    // Reactive limits (full-loadflow PV→PQ outer loop): per local-PV controller bus, its (rowMap-mapped)
    // BUS_V row + reactive limits + load Q. dQbuf holds Σ branch q at every row (a fill into a zeroed
    // buffer); dRlActive marks which scenarios to RL-check this pass; dRlSwitched flags scenarios that switched.
    int nrl = 0;
    DeviceBuf dRlVrow, dRlMinQ, dRlMaxQ, dRlLoadQ, dRlVtarget;   // per controller (nrl)
    // Reactive capability CURVES at runtime P (setBatchRlCurves): per participating curve gen —
    // cvIdx {rlIdx, pRow, ptsOff, npts}, cvIn {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu},
    // cvPts {p_MW, minQ_MW, maxQ_MW} sorted by p; dRlMinQDyn/dRlMaxQDyn = nS*nrl evaluated limits.
    int ncv = 0;
    int cvExtrapolate = 0;                                 // OLF extrapolateReactiveLimits for curve gens
    DeviceBuf dCvIdx, dCvIn, dCvPts, dRlMinQDyn, dRlMaxQDyn;
    // DISTR_Q weight overrides (setBatchDqWeightOverrides): a gen outage at a remote-VC controller
    // bus renormalizes the group's qPct keys per scenario — {s, contrib} pairs + the new weight.
    int nDqOv = 0;
    DeviceBuf dDqOvIdx, dDqOvW;
    // DISTR_RHO coefficient overrides (setBatchDrCoefOverrides): a member outage renormalizes the
    // group's 1/count coefficients per scenario — {s, contrib} pairs + the new coefficient.
    int nDrOv = 0;
    DeviceBuf dDrOvIdx, dDrOvC;
    // Per-scenario ACTIVE equation counts (setBatchActiveRowCounts): OLF's L2 stopping threshold is
    // sqrt(eps^2 * nActive) and the CPU REMOVES equations for outaged elements carrying variables
    // (rho/a1 identities) and islanded buses — the padded device n over-counts by those rows, a ~%
    // threshold skew that lands stops up to the criterion apart. Empty = use n for every scenario.
    std::vector<int> hNActive;
    DeviceBuf dRlMonitored;                                // per controller: 1 = MONITORING_VOLTAGE SVC standing by
    std::vector<char> hRlMonitored;                        // host copy, for the forced-PV detector
    int allPvPinnable = 0;                                 // every non-monitored RL entry has finite limits — the
                                                           // ONLY networks where OLF's global "strongest stays PV"
                                                           // rule (remainingPvBusCount 0) is reachable
    DeviceBuf dRowSlope;                                   // per row: SLOPE-SVC slope (mode-2 V rows), 0 elsewhere
    int nRlRemote = 0;                                     // remote-member RL entries present (plan-B1/B3)
    DeviceBuf dRlRemote;                                   // per-entry remote kind (1 = remote group member)
    DeviceBuf dRlCmpVrow;                                  // per-entry compare row (PILOT v col for remote members)
    DeviceBuf dRlSvcBmin, dRlSvcBmax, dRlV2Base;           // per-entry SVC quadratic limit coefs + packed base v^2
                                                           // (limit(v) = static + b*(v^2 - v2Base)); zeroed default
    int nDqDyn = 0;                                        // dynamic reactive keys (plan-B3): an RL pin zeroes the
    DeviceBuf dDqKMember;                                  // pinned member's key and renormalizes the DISTR_Q
    DeviceBuf dDqJMember;                                  // weights over the survivors (updateReactiveKeys) — the
    DeviceBuf dDqMemberKey;                                // kernel recomputes each contribution's weight from the
    DeviceBuf dDqMemberRlEntry;                            // RAW keys + the per-(entry, scenario) limitType,
    DeviceBuf dDqMemberGroupStart;                         // zero-then-sum in member order (bit-exact vs OLF)
    DeviceBuf dDqMemberGroupCount;
    DeviceBuf dDqMemberRow;                                // per member: its own row (vCol) + whether it OWNS a
    DeviceBuf dDqMemberHasRow;                             // packed DISTR_Q row (non-free with existing equation)
    int nDqMembers = 0;
    int hasDqQrGroups = 0;                                 // plan-B4: any member linked to a pilot-Q relocation group
    DeviceBuf dDqMemberQrGroup;                            // per member: the pilot's qr candidate group id, -1 = none
    DeviceBuf dQrPilotQ;                                   // per (scenario, qr group): the pilot's Q target, captured
                                                           // at solve init from the BASE carrier's row and handed to
                                                           // the CURRENT carrier's row on every re-pick
    DeviceBuf dQrGenVc;                                    // per qr group: 1 = remote gen-VC group (vs TVC — only
                                                           // those get the all-pinned pilot-vEq flip)
    DeviceBuf dQrVeqOff;                                   // per (scenario, qr group): 1 = the pilot's V equation was
    DeviceBuf dQrSavedVT;                                  // flipped OFF by updatePilotVeqBatch (+ the saved V target)
    int nRmq = 0;                                          // remote-member OWN-q contributions (plan-B3): a twin
    DeviceBuf dRmqRow;                                     // of the DISTR_Q assembly accumulating each member's own
    DeviceBuf dRmqElem;                                    // bus q into dQbuf at its own row (that slot is free —
    DeviceBuf dRmqSide;                                    // the main qBuf fill suppresses member rows)
    DeviceBuf dRmqKind;
    DeviceBuf dRmqWeight;
    DeviceBuf dRlSavedTarget;                              // per (controller, scenario): the row target saved at a
                                                           // PV->PQ pin, restored on switch-back for mode-2 rows
    DeviceBuf dQbuf, dRlActive, dRlSwitched;               // dQbuf nS*n, dRlActive/dRlSwitched nS
    DeviceBuf dRlLimitType, dRlSwitchCount;               // per (controller, scenario): 0 PV / 1 minQ / 2 maxQ; PV→PQ count
    // Distributed slack: the (mapped) slack BUS_PHI row + its scheduled targetP give the per-scenario
    // mismatch (read from dQbuf); dDsPhiRow/dDsFactor are the participating buses' P-target rows + normalized
    // participation factors. Reuses dRlActive/dRlSwitched (same outer schedule as reactive limits).
    int nds = 0, dsSlackPhiRow = -1;
    double dsSlackTargetP = 0.0;
    DeviceBuf dDsPhiRow, dDsFactor;                        // per participating bus (nds)
    DeviceBuf dDsBaseTargetP, dDsMaxDelta, dDsMinDelta;   // per participating bus (nds): base P target + headroom/footroom
    double dsFactorSum = 1.0;                              // Σ emitted factors (< 1 ⇒ slack participates, keeps 1−Σ)
    int dsMode = 0;                                        // 0 = static factors; 1 = REMAINING_MARGIN (direction-
    DeviceBuf dDsMarginUp, dDsMarginDown;                  //   dependent per-bus margins, normalized in-kernel)
    double dsSlackMarginUp = 0.0, dsSlackMarginDown = 0.0; //   the slack's own margins (its retained share)
    DeviceBuf dDsRetain;
    DeviceBuf dDsResidual;                                   // nS chars: waterfill left an out-of-tolerance residual                                   // per scenario (nS): frozen slack-retained mismatch share
    bool dsSlackOnly = false;                                // DS enabled with the SLACK generator as the only participant:
    DeviceBuf dDsSlackAbsorbed;                              // nS doubles — mismatch absorbed into the slack's own target so far
                                                             // (OLF distributes into the slack gen and re-solves; no flow change,
                                                             // but the extra polish run must happen to match the CPU trajectory)
    CudssCtx ds;
    bool analyzed = false;
    bool factored = false;                                   // first FACTORIZATION done (mode 2: then REFACTORIZE)
    int mode = 0;                                            // 0 = per-scenario loop, 1 = batched API, 2 = block-diagonal, 3 = UBATCH
    // Uniform batched cuDSS matrices (mode 1 only): A (CSR) shares the rowPtr/colIdx pattern
    // across all scenarios with per-scenario value pointers; B/X are batched dense (n x 1).
    // The host pointer + size arrays must outlive the matrix objects, so they live here.
    cudssMatrix_t matABatch = nullptr, matBBatch = nullptr, matXBatch = nullptr;
    std::vector<void*> valPtrs, rowStartPtrs, rowEndPtrs, colPtrs, bPtrs, xPtrs;   // host: arrays of device pointers
    DeviceBuf dValPtrs, dRowStartPtrs, dRowEndPtrs, dColPtrs, dBPtrs, dXPtrs;       // device copies (cuDSS 0.8 wants device arrays)
    std::vector<int> nrowsArr, ncolsArr, ncols1Arr, nnzArr, ldArr;                  // host size arrays: int32 + CUDSS_R_32I
    // Block-diagonal single matrix (mode 2 only): one (n*nS) CSR = blkdiag of all scenarios,
    // values = dVals (per-scenario contiguous), B/X = dB/dDx. Factorize once, REFACTORIZE after
    // (reuse pivots — the refactor-every strategy; refactor IS cheap on a single matrix, unlike the batch API).
    cudssMatrix_t matBig = nullptr, matBigB = nullptr, matBigX = nullptr;
    DeviceBuf dBigRowPtr, dBigColIdx;
    // UBATCH single matrices (mode 3 only): a SINGLE per-block CSR (n, nnz, shared rowPtr/colIdx)
    // whose value/RHS pointers are the batched buffers (dVals = nS*nnz, dB/dDx = nS*n); cuDSS
    // analyzes ONE block (cheap) and factorizes/solves all nS via CUDSS_CONFIG_UBATCH_SIZE.
    cudssMatrix_t matUb = nullptr, matUbB = nullptr, matUbX = nullptr;
    // Dedicated stream for the WHOLE batch solve (every kernel/memset/async copy + cuDSS via
    // cudssSetStream): keeps the per-iteration launch sequence off the legacy default stream,
    // whose implicit device-wide serialization stitched the timeline with sync points (the
    // ~25 ms/iter host/launch gaps of the 2026-07-15 nsys attribution — a loop kept on one
    // explicit stream shows none). Created BLOCKING (default flags), so any
    // op left on the legacy stream still orders correctly against it — safety over speed for
    // whatever this conversion misses.
    cudaStream_t stream = nullptr;
    ~BatchSolverContext() {
        if (stream) cudaStreamDestroy(stream);
        if (matABatch) cudssMatrixDestroy(matABatch);
        if (matBBatch) cudssMatrixDestroy(matBBatch);
        if (matXBatch) cudssMatrixDestroy(matXBatch);
        if (matBig) cudssMatrixDestroy(matBig);
        if (matBigB) cudssMatrixDestroy(matBigB);
        if (matBigX) cudssMatrixDestroy(matBigX);
        if (matUb) cudssMatrixDestroy(matUb);
        if (matUbB) cudssMatrixDestroy(matUbB);
        if (matUbX) cudssMatrixDestroy(matUbX);
        if (matUbC) cudssMatrixDestroy(matUbC);
        if (matUbCB) cudssMatrixDestroy(matUbCB);
        if (matUbCX) cudssMatrixDestroy(matUbCX);
    }
};

static void setupBatchMatricesUniform(BatchSolverContext* ctx) {
    int n = ctx->n;
    int nScenarios = ctx->nScenarios;
    // ---- uniform batched matrices: one A (CSR) per scenario sharing the pattern, batched
    //      dense B/X. The host pointer + size arrays persist in the context. ----
        int* rowPtrDev = ctx->st->dRowPtr.as<int>();
        int* colDev = ctx->st->dColIdx.as<int>();
        double* valsDev = ctx->dVals.as<double>();
        double* bDev = ctx->dB.as<double>();
        double* xDev = ctx->dDx.as<double>();
        ctx->valPtrs.resize(nScenarios);
        ctx->rowStartPtrs.resize(nScenarios);
        ctx->rowEndPtrs.resize(nScenarios);
        ctx->colPtrs.resize(nScenarios);
        ctx->bPtrs.resize(nScenarios);
        ctx->xPtrs.resize(nScenarios);
        ctx->nrowsArr.assign(nScenarios, n);
        ctx->ncolsArr.assign(nScenarios, n);
        ctx->ncols1Arr.assign(nScenarios, 1);
        ctx->nnzArr.assign(nScenarios, ctx->nnz);
        ctx->ldArr.assign(nScenarios, n);
        for (int s = 0; s < nScenarios; s++) {
            ctx->valPtrs[s] = valsDev + (size_t) s * ctx->nnz;
            ctx->rowStartPtrs[s] = rowPtrDev;                // shared CSR row offsets
            ctx->rowEndPtrs[s] = rowPtrDev + 1;              // rowEnd = rowStart + 1
            ctx->colPtrs[s] = colDev;                        // shared column indices
            ctx->bPtrs[s] = bDev + (size_t) s * n;
            ctx->xPtrs[s] = xDev + (size_t) s * n;
        }
        // cuDSS 0.8 batch API wants the pointer arrays in DEVICE memory (arrays of device
        // pointers); the size arrays stay on the host. Upload the six host pointer arrays.
        size_t ptrBytes = static_cast<size_t>(nScenarios) * sizeof(void*);
        ctx->dValPtrs.upload(ctx->valPtrs.data(), ptrBytes);
        ctx->dRowStartPtrs.upload(ctx->rowStartPtrs.data(), ptrBytes);
        ctx->dRowEndPtrs.upload(ctx->rowEndPtrs.data(), ptrBytes);
        ctx->dColPtrs.upload(ctx->colPtrs.data(), ptrBytes);
        ctx->dBPtrs.upload(ctx->bPtrs.data(), ptrBytes);
        ctx->dXPtrs.upload(ctx->xPtrs.data(), ptrBytes);
        CUDSS_CHECK(cudssMatrixCreateBatchCsr(&ctx->matABatch, nScenarios,
                ctx->nrowsArr.data(), ctx->ncolsArr.data(), ctx->nnzArr.data(),
                ctx->dRowStartPtrs.as<void*>(), nullptr, ctx->dColPtrs.as<void*>(), ctx->dValPtrs.as<void*>(),
                OLF_CUDSS_CSR_TYPES,
                CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateBatchDn(&ctx->matBBatch, nScenarios,
                ctx->nrowsArr.data(), ctx->ncols1Arr.data(), ctx->ldArr.data(),
                ctx->dBPtrs.as<void*>(), OLF_CUDSS_BDN_TYPES, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateBatchDn(&ctx->matXBatch, nScenarios,
                ctx->nrowsArr.data(), ctx->ncols1Arr.data(), ctx->ldArr.data(),
                ctx->dXPtrs.as<void*>(), OLF_CUDSS_BDN_TYPES, CUDSS_LAYOUT_COL_MAJOR));
}

static void setupBatchMatricesBlockDiag(BatchSolverContext* ctx) {
    int n = ctx->n;
    int nScenarios = ctx->nScenarios;
    const std::vector<int>& rowPtr = ctx->st->rowPtr;
    const std::vector<int>& colIdx = ctx->st->colIdx;
    // ---- block-diagonal single matrix: one (n*nS) CSR = blkdiag(J_1..J_nS). Its values array
    //      IS dVals (per-scenario nnz contiguous); we only build the big rowPtr/colIdx (per-block
    //      pattern with column offsets s*n). One ANALYZE+FACTORIZE, then REFACTORIZE every solve. ----
    {
        long bigN = (long) n * nScenarios;
        long bigNnz = (long) ctx->nnz * nScenarios;
        std::vector<int> bigRowPtr(bigN + 1), bigColIdx(bigNnz);
        bigRowPtr[0] = 0;
        for (int s = 0; s < nScenarios; s++) {
            for (int r = 0; r < n; r++) {
                long gRow = (long) s * n + r;
                bigRowPtr[gRow + 1] = bigRowPtr[gRow] + (rowPtr[r + 1] - rowPtr[r]);
                for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
                    bigColIdx[(long) s * ctx->nnz + k] = colIdx[k] + s * n;   // shift cols into block s
                }
            }
        }
        ctx->dBigRowPtr.upload(bigRowPtr.data(), (bigN + 1) * sizeof(int));
        ctx->dBigColIdx.upload(bigColIdx.data(), bigNnz * sizeof(int));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->matBig, bigN, bigN, bigNnz,
                ctx->dBigRowPtr.as<int>(), nullptr, ctx->dBigColIdx.as<int>(), ctx->dVals.as<double>(),
                OLF_CUDSS_CSR_TYPES,
                CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matBigB, bigN, 1, bigN, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matBigX, bigN, 1, bigN, ctx->dDx.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    }
}

static void setupBatchMatricesUbatch(BatchSolverContext* ctx) {
    int n = ctx->n;
    int nScenarios = ctx->nScenarios;
    // ---- UBATCH: ONE per-block CSR (n, nnz, shared rowPtr/colIdx) with batched value/RHS pointers;
    //      cuDSS analyzes a single block (cheap) and factorizes/solves all nS via UBATCH_SIZE.
    //      Modes 4/5 share this exact setup and differ only in the per-iteration factorize policy
    //      (mode 4 = factor at chunk iter 0 only; mode 5 = factor once + refactorize after). ----
    {
        int ub = nScenarios;
        CUDSS_CHECK(cudssConfigSet(ctx->ds.config, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof(ub)));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->matUb, n, n, ctx->nnz,
                ctx->st->dRowPtr.as<int>(), nullptr, ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(),
                OLF_CUDSS_CSR_TYPES,
                CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbB, n, 1, n, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbX, n, 1, n, ctx->dDx.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    }
}

static void setupBatchMatricesFallback(BatchSolverContext* ctx) {
    int n = ctx->n;
    // ---- fallback: a single matrix repointed per scenario inside the solve loop ----
    CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->ds.matA, n, n, ctx->nnz,
                                     ctx->st->dRowPtr.as<int>(), nullptr, ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(),
                                     OLF_CUDSS_CSR_TYPES,
                                     CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&ctx->ds.matB, n, 1, n, ctx->dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&ctx->ds.matX, n, 1, n, ctx->dDx.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
}

BatchSolverContext* createBatchContext(const SharedStructurePtr& st, int nScenarios, int mode,
                                       int nrl, const int* rlVrow, const double* rlMinQ,
                                       const double* rlMaxQ, const double* rlLoadQ, const double* rlVtarget,
                                       const signed char* rlMonitored, const double* rowSlope,
                                       int nds, const int* dsPhiRow, const double* dsFactor,
                                       int dsSlackPhiRow, double dsSlackTargetP,
                                       const double* dsBaseTargetP, const double* dsMaxDelta, const double* dsMinDelta,
                                       int dsMode, const double* dsMarginUp, const double* dsMarginDown,
                                       double dsSlackMarginUp, double dsSlackMarginDown) {
    auto ctx = std::make_unique<BatchSolverContext>();
    ctx->st = st;
    int n = st->n;
    ctx->n = n;
    ctx->nnz = st->nnz;
    ctx->nScenarios = nScenarios;
    ctx->mode = mode;
    ctx->nElements = st->nElements;
    ctx->cbOffset = st->cbOffset;
    ctx->obOffset = st->obOffset;
    ctx->shOffset = st->shOffset;
    ctx->hvOffset = st->hvOffset;
    ctx->ndq = st->ndq;
    ctx->ndr = st->ndr;
    ctx->ndb = st->ndb;
    ctx->nzi = st->nzi;
    const std::vector<int>& rowMap = st->rowMap;
    const std::vector<int>& rowPtr = st->rowPtr;
    const std::vector<int>& colIdx = st->colIdx;

    // ---- per-scenario device state ----
    size_t nS = static_cast<size_t>(n) * nScenarios;
    size_t nnzS = static_cast<size_t>(ctx->nnz) * nScenarios;
    ctx->dX.alloc(nS * sizeof(double));
    ctx->dTarget.alloc(nS * sizeof(double));
    ctx->dF.alloc(nS * sizeof(double));
    ctx->dB.alloc(nS * sizeof(double));
    ctx->dDx.alloc(nS * sizeof(double));
    ctx->dVals.alloc(nnzS * sizeof(double));
    ctx->dNorms.alloc(static_cast<size_t>(nScenarios) * 4 * sizeof(double));
    ctx->dMode.alloc(nS * sizeof(int));
    ctx->dDisabled.alloc(static_cast<size_t>(ctx->nElements) * nScenarios * sizeof(char));
    ctx->dActive.alloc(static_cast<size_t>(nScenarios) * sizeof(char));

    // ---- reactive-limits pack: the controller BUS_V rows mapped through the diagonal-pairing
    //      permutation (= identity in the batch path, no families, but mapped for correctness) ----
    ctx->nrl = nrl;
    ctx->dQbuf.alloc(static_cast<size_t>(n) * nScenarios * sizeof(double));
    ctx->dRlActive.alloc(static_cast<size_t>(nScenarios) * sizeof(char));
    ctx->dRlSwitched.alloc(static_cast<size_t>(nScenarios) * sizeof(char));
    if (nrl > 0) {
        std::vector<int> rlVrowM(nrl);
        for (int i = 0; i < nrl; i++) {
            rlVrowM[i] = rowMap[rlVrow[i]];
        }
        ctx->dRlVrow.upload(rlVrowM.data(), nrl * sizeof(int));
        ctx->dRlMinQ.upload(rlMinQ, nrl * sizeof(double));
        ctx->dRlMaxQ.upload(rlMaxQ, nrl * sizeof(double));
        ctx->dRlLoadQ.upload(rlLoadQ, nrl * sizeof(double));
        ctx->dRlVtarget.upload(rlVtarget, nrl * sizeof(double));
        ctx->dRlMonitored.upload(rlMonitored, nrl * sizeof(char));
        ctx->hRlMonitored.assign(rlMonitored, rlMonitored + nrl);
        ctx->allPvPinnable = 1;
        for (int i = 0; i < nrl; i++) {
            if (!rlMonitored[i] && !((rlMinQ[i] > -1e7 || rlMaxQ[i] < 1e7) && rlMinQ[i] < rlMaxQ[i])) {
                ctx->allPvPinnable = 0;                      // an unlimited PV bus: the rule never fires
            }
        }
        ctx->dRlSavedTarget.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(double));
        ctx->dRlLimitType.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(char));
        ctx->dRlSwitchCount.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(char));
        std::vector<double> svcZero(nrl, 0.0);              // neutral until setBatchRlSvcLimits installs
        ctx->dRlSvcBmin.upload(svcZero.data(), nrl * sizeof(double));
        ctx->dRlSvcBmax.upload(svcZero.data(), nrl * sizeof(double));
        ctx->dRlV2Base.upload(svcZero.data(), nrl * sizeof(double));
    }
    ctx->dRowSlope.upload(rowSlope, n * sizeof(double));
    ctx->nds = nds;
    ctx->dsSlackTargetP = dsSlackTargetP;
    ctx->dsMode = dsMode;
    ctx->dsSlackMarginUp = dsSlackMarginUp;
    ctx->dsSlackMarginDown = dsSlackMarginDown;
    if (dsMode == 1 && nds > 0) {
        ctx->dDsMarginUp.upload(dsMarginUp, nds * sizeof(double));
        ctx->dDsMarginDown.upload(dsMarginDown, nds * sizeof(double));
    }
    ctx->dsSlackPhiRow = dsSlackPhiRow >= 0 ? rowMap[dsSlackPhiRow] : -1;
    ctx->dsSlackOnly = (nds == 0 && ctx->dsSlackPhiRow >= 0);
    if (ctx->dsSlackOnly) {
        ctx->dDsSlackAbsorbed.alloc(static_cast<size_t>(nScenarios) * sizeof(double));
    }
    if (nds > 0) {
        std::vector<int> dsPhiRowM(nds);
        for (int i = 0; i < nds; i++) {
            dsPhiRowM[i] = rowMap[dsPhiRow[i]];
        }
        ctx->dDsPhiRow.upload(dsPhiRowM.data(), nds * sizeof(int));
        ctx->dDsFactor.upload(dsFactor, nds * sizeof(double));
        ctx->dDsBaseTargetP.upload(dsBaseTargetP, nds * sizeof(double));
        ctx->dDsMaxDelta.upload(dsMaxDelta, nds * sizeof(double));
        ctx->dDsMinDelta.upload(dsMinDelta, nds * sizeof(double));
        double factorSum = 0.0;
        for (int i = 0; i < nds; i++) {
            factorSum += dsFactor[i];
        }
        ctx->dsFactorSum = factorSum > 0 ? factorSum : 1.0;
        ctx->dDsRetain.alloc(static_cast<size_t>(nScenarios) * sizeof(double));
    }

    // ---- cuDSS handle/config/data (shared by both solve paths) ----
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    CUDSS_CHECK(cudssCreate(&ctx->ds.handle));
    CUDSS_CHECK(cudssSetStream(ctx->ds.handle, ctx->stream));
    CUDSS_CHECK(cudssConfigCreate(&ctx->ds.config));
    int ir = irSteps();
    CUDSS_CHECK(cudssConfigSet(ctx->ds.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)));
    applyReorderAlg(ctx->ds.config);
    CUDSS_CHECK(cudssDataCreate(ctx->ds.handle, &ctx->ds.data));

    if (mode == 1) {
        setupBatchMatricesUniform(ctx.get());
    } else if (mode == 2) {
        setupBatchMatricesBlockDiag(ctx.get());
    } else if (mode == 3 || mode == 4 || mode == 5) {
        setupBatchMatricesUbatch(ctx.get());
    } else {
        setupBatchMatricesFallback(ctx.get());
    }
    return ctx.release();
}

struct BatchResult {
    std::vector<double> x;                                   // nScenarios * n
    std::vector<int> iters;                                  // nScenarios
    std::vector<double> norm;                                // nScenarios
    std::vector<int> status;                                 // nScenarios: 0 conv, 1 maxiter, -1 diverged
};

// Per-scenario target overrides: each entry writes target[ovIndex[k]] = ovValue[k], where ovIndex is the
// FLAT dTarget index (scenarioSlot*n + row). Used by generator contingencies to set a scenario's bus P/Q
// target (remove the lost generator's P, set BUS_TARGET_Q on a PV→PQ flip) after the base row is tiled.
__global__ void applyTargetOverrideBatch(int nOv, const int* ovIndex, const double* ovValue, double* target) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nOv) {
        target[ovIndex[k]] = ovValue[k];
    }
}

static void uploadBatchOverrides(BatchSolverContext* ctx, int lsMode, double maxDv, double maxDphi,
                                 const int* varType, int nShOv, const int* shOvSlotElem, const double* shOvGb,
                                 int nziBase, int nZiOff, const int* ziOffSlotElem,
                                 int nZiOn, const int* ziOnSlotElem,
                                 int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal) {
    int n = ctx->n;
    int nS = ctx->nScenarios;
    bool clampStep = lsMode == 2 && varType != nullptr;
    if (clampStep && !ctx->dVarType.p) {
        ctx->dVarType.upload(varType, static_cast<size_t>(n) * sizeof(int));
        ctx->dStepFactor.alloc(static_cast<size_t>(nS) * sizeof(double));
    }
    if (lsMode == 1) {                                       // LINE_SEARCH backtracking buffers
        if (!ctx->dXprevLs.p) ctx->dXprevLs.alloc(static_cast<size_t>(n) * nS * sizeof(double));
        if (!ctx->dStepFactor.p) ctx->dStepFactor.alloc(static_cast<size_t>(nS) * sizeof(double));
    }
    ctx->clampStep = clampStep;
    ctx->maxDv = maxDv;
    ctx->maxDphi = maxDphi;
    // Per-scenario shunt admittance overrides (section shifts / reconnection). PACK-indexed
    // elements (shunt family index, not the global element index). Empty = a 1-entry dummy
    // upload so the kernels' pointers are always valid.
    ctx->nShOv = nShOv;
    if (nShOv > 0) {
        ctx->dShOvSlotElem.upload(shOvSlotElem, static_cast<size_t>(nShOv) * 2 * sizeof(int));
        ctx->dShOvGb.upload(shOvGb, static_cast<size_t>(nShOv) * 2 * sizeof(double));
    } else if (!ctx->dShOvSlotElem.p) {
        int dummyI[2] = {-1, -1};
        double dummyD[2] = {0, 0};
        ctx->dShOvSlotElem.upload(dummyI, sizeof(dummyI));
        ctx->dShOvGb.upload(dummyD, sizeof(dummyD));
    }
    ctx->nziBase = nziBase > 0 ? nziBase : ctx->nzi;         // default: every zi element always enabled
    ctx->nZiOff = nZiOff;
    if (nZiOff > 0) {
        ctx->dZiOffSlotElem.upload(ziOffSlotElem, static_cast<size_t>(nZiOff) * 2 * sizeof(int));
    } else if (!ctx->dZiOffSlotElem.p) {
        int dummyZiOff[2] = {-1, -1};
        ctx->dZiOffSlotElem.upload(dummyZiOff, sizeof(dummyZiOff));
    }
    ctx->nZiOn = nZiOn;
    if (nZiOn > 0) {
        ctx->dZiOnSlotElem.upload(ziOnSlotElem, static_cast<size_t>(nZiOn) * 2 * sizeof(int));
    } else if (!ctx->dZiOnSlotElem.p) {
        int dummyZi[2] = {-1, -1};
        ctx->dZiOnSlotElem.upload(dummyZi, sizeof(dummyZi));
    }
    ctx->nR1Ov = nR1Ov;
    if (nR1Ov > 0) {
        ctx->dR1OvSlotElem.upload(r1OvSlotElem, static_cast<size_t>(nR1Ov) * 2 * sizeof(int));
        ctx->dR1OvVal.upload(r1OvVal, static_cast<size_t>(nR1Ov) * 8 * sizeof(double));
    } else if (!ctx->dR1OvSlotElem.p) {
        int dummyI[2] = {-1, -1};
        double dummyD[1] = {0};
        ctx->dR1OvSlotElem.upload(dummyI, sizeof(dummyI));
        ctx->dR1OvVal.upload(dummyD, sizeof(dummyD));
    }
}

BatchResult solveBatchContext(BatchSolverContext* ctx, const double* x0, const double* target,
                              const int* rowMode, const int* baseRowMode, const char* disabled,
                              int nOv, const int* ovIndex, const double* ovValue,
                              const int* dsOvBus, const double* dsOvFactorDelta, const double* dsOvBaseDelta,
                              const double* dsOvMaxDelta, const double* dsOvMinDelta, const double* dsAnchorShift,
                              const double* dsSlackShift,
                              int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                              int maxIter, double tol,
                              double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                              int lsMode, double maxDv, double maxDphi, const int* varType,
                              const double* x0Full, int nShOv, const int* shOvSlotElem, const double* shOvGb,
                              const int* rlStateIn, int* rlStateOut,
                              int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                              int maxOuterProductive, int rlMaxSwitch, int dsRejectResidual, int skipRlPhase,
                              int nziBase, int nZiOn, const int* ziOnSlotElem,
                              int nZiOff, const int* ziOffSlotElem) {
    int n = ctx->n;
    int nnz = ctx->nnz;
    int nS = ctx->nScenarios;
    // MAX_VOLTAGE_CHANGE (lsMode 2): per-scenario step clamp, same rule as the base solver's
    // (varType 1 = V vs maxDv, 0 = PHI vs maxDphi; whole step scaled). Buffers allocated lazily.
    uploadBatchOverrides(ctx, lsMode, maxDv, maxDphi, varType, nShOv, shOvSlotElem, shOvGb,
                         nziBase, nZiOff, ziOffSlotElem, nZiOn, ziOnSlotElem, nR1Ov, r1OvSlotElem, r1OvVal);
    size_t nTot = static_cast<size_t>(n) * nS;
    DeviceBuf dOvIndex, dOvValue;                            // function-scoped: outlive the async override kernel
    DeviceBuf dBaseMode;                                     // the contingency-free base row mode (length n)
    // Per-scenario distributed-slack reweight (generator loss at a DS-participant bus): one affected DS bus per
    // scenario (dsOvBus[s], -1 if none) whose factor/base/headroom/footroom shrink by the lost gen's share, and
    // a per-scenario factor-sum drop so the other buses pick up that share. nullptr when no DS scenarios.
    DeviceBuf dDsOvBus, dDsOvFactorDelta, dDsOvBaseDelta, dDsOvMaxDelta, dDsOvMinDelta;
    bool hasDsOverride = (dsOvBus != nullptr && ctx->nds > 0);
    // Per-scenario DS ANCHOR SHIFT (pair/strategy legs): shift_i = parent's final target at participant i
    // minus the extraction-time base — re-anchors the water-fill's cumulative reference at the PARENT
    // contingency's distribution, mirroring OLF's setGeneratorsInitialTargetPToTargetP before a strategy
    // re-solve. nullptr = anchored at the base (plain contingencies).
    DeviceBuf dRlVtOvSlotIdx, dRlVtOvVal;
    if (nRlVtOv > 0) {
        dRlVtOvSlotIdx.upload(rlVtOvSlotIdx, 2L * nRlVtOv * sizeof(int));
        dRlVtOvVal.upload(rlVtOvVal, (size_t) nRlVtOv * sizeof(double));
    }
    DeviceBuf dDsAnchorShift, dDsSlackShift;
    bool hasDsAnchor = (dsAnchorShift != nullptr && dsSlackShift != nullptr && ctx->nds > 0);
    if (hasDsAnchor) {
        dDsAnchorShift.upload(dsAnchorShift, static_cast<size_t>(nS) * ctx->nds * sizeof(double));
        dDsSlackShift.upload(dsSlackShift, static_cast<size_t>(nS) * sizeof(double));
    }

    // x0/target are the BASE row (length n), identical for every scenario: upload once and tile across
    // nS on-device. rowMode is PER-SCENARIO (length n*nS): a contingency can freeze islanded buses
    // (Phase 0b) or flip PV↔PQ via reactive limits (Phase 1), so it is NOT tiled — uploaded as-is and
    // mutated in place on-device by the outer-loop kernels. Only the disabled mask is also per-scenario.
    // x0: either the BASE row tiled across nS (default), or a FULL per-scenario n*nS state
    // (x0Full != null) — the warm-restart hook for multi-call outer sequencing (e.g. the
    // transformer voltage-control orchestrator re-solving scenarios from their previous states).
    if (x0Full != nullptr) {
        CUDA_CHECK(cudaMemcpy(ctx->dX.p, x0Full, (size_t) nTot * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(ctx->dX.p, x0, n * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(ctx->dTarget.p, target, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->dMode.p, rowMode, nTot * sizeof(int), cudaMemcpyHostToDevice));
    dBaseMode.upload(baseRowMode, n * sizeof(int));         // base (contingency-free) row mode for the RL limit-type seed
    if (x0Full == nullptr) {
        tileBaseRow<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, n, ctx->dX.as<double>());
    }
    tileBaseRow<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, n, ctx->dTarget.as<double>());
    if (nOv > 0) {                                           // per-scenario target overrides (generator loss)
        dOvIndex.upload(ovIndex, nOv * sizeof(int));
        dOvValue.upload(ovValue, nOv * sizeof(double));
        applyTargetOverrideBatch<<<grid(nOv), BLOCK, 0, ctx->stream>>>(nOv, dOvIndex.as<int>(), dOvValue.as<double>(), ctx->dTarget.as<double>());
    }
    if (hasDsOverride) {                                     // per-scenario generator-loss DS reweight
        dDsOvBus.upload(dsOvBus, nS * sizeof(int));
        dDsOvFactorDelta.upload(dsOvFactorDelta, nS * sizeof(double));
        dDsOvBaseDelta.upload(dsOvBaseDelta, nS * sizeof(double));
        dDsOvMaxDelta.upload(dsOvMaxDelta, nS * sizeof(double));
        dDsOvMinDelta.upload(dsOvMinDelta, nS * sizeof(double));
    }
    CUDA_CHECK(cudaMemcpy(ctx->dDisabled.p, disabled,
                          static_cast<size_t>(ctx->nElements) * nS * sizeof(char), cudaMemcpyHostToDevice));

    std::vector<char> active(nS, 1);

    BatchResult res;
    res.x.resize(nTot);
    res.iters.assign(nS, 0);
    res.norm.assign(nS, 0.0);
    res.status.assign(nS, 1);                                // default: max_iter (overwritten on converge/diverge)

    std::vector<double> hF(nTot);

    // Device-side fixed-iteration mode: tol <= 0 means "run exactly maxIter Newton steps
    // on the device with NO per-iteration host sync and NO convergence retirement" (the
    // a fixed-iteration NR loop). tol > 0 keeps the per-iteration retirement loop.
    bool deviceFixedIter = (tol <= 0.0);
    // STRAGGLER EARLY-EXIT (tol > 0 only): once the active count falls to <= frac*nS (min 1),
    // stop the batched loop and mark the remainder STATUS_REJECTED (-2) so the Java side routes
    // them to the per-contingency fallback. Rationale (2026-07-17 tol>0 profile): retirement does
    // NOT shrink cuDSS UBATCH cost — 499 retired identity systems still factorize at the full
    // ~233 ms/iteration — so one never-converging scenario makes the whole chunk pay maxIter
    // full-width iterations per outer pass. OLF_GPU_STRAGGLER_FRAC overrides (0 disables).
    int stragglerExit = 0;
    if (!deviceFixedIter) {
        double stragglerFrac = 0.02;
        if (const char* e = std::getenv("OLF_GPU_STRAGGLER_FRAC")) {
            stragglerFrac = std::atof(e);
        }
        if (stragglerFrac > 0.0) {
            stragglerExit = std::max(1, static_cast<int>(stragglerFrac * nS));
        }
    }

    // Full-loadflow mode (tol > 0) with local PV controllers runs the reactive-limits (PV→PQ) OUTER loop:
    // converge the inner Newton, switch any generator over its Q limit, re-converge the switched scenarios,
    // repeat until a pass switches nothing. Fixed-iter mode (tol <= 0, the throughput benchmark) and
    // family-free networks with no controllers do ONE pass (no outer loop) — byte-identical to before.
    bool fullLoadflow = !deviceFixedIter && (ctx->nrl > 0 || ctx->nds > 0 || ctx->dsSlackOnly);
    const int RL_MAX_OUTER = maxOuterProductive > 0 ? maxOuterProductive : 20;   // OLF maxOuterLoopIterations
    const int RL_MAX_SWITCH = rlMaxSwitch > 0 ? rlMaxSwitch : 3;   // per-bus PV→PQ count cap (OLF maxPqPvSwitch)
    // Outer-loop convergence tolerances (pu) — passed in so the device uses the SAME criteria as the CPU
    // loadflow that built the base case (maxReactivePowerMismatch / slackBusPMaxMismatch, per-unitized), NOT
    // hardcoded constants. With matching parameters and the same equations the GPU outer-loop fixed point
    // matches the CPU's to the inner-Newton precision; a loose tolerance (e.g. the 1 MW slack default) leaves
    // a redistribution band, so a test wanting machine-precision agreement passes tight values on both sides.
    const double REACTIVE_LIMIT_TOL = reactiveLimitTol;
    const double SLACK_DIST_TOL = slackDistTol;
    const double P_RESIDUE_EPS = 1e-5;                       // OLF ActivePowerDistribution.P_RESIDUE_EPS: the
                                                            // water-filling distributes the mismatch down to this
    // RL_MAX_OUTER bounds the PRODUCTIVE outer iterations (an outer loop that re-solved), matching OLF's
    // outerLoopTotalIterations budget. The raw pass loop gets a generous safety bound on top, because the
    // sequential driver also spends non-productive passes (phase transitions / stability re-checks) that OLF
    // does not count.
    int maxOuter = fullLoadflow ? 8 * RL_MAX_OUTER : 1;
    std::vector<char> rlActive(nS), rlSwitched(nS), wasActive(nS);
    if (fullLoadflow) {                                      // per-solve PV↔PQ state: 0 switches, limitType from base
        if (ctx->hasDqQrGroups && ctx->st->nQr > 0) {        // plan-B4: capture each group's pilot-Q target from
            size_t need = static_cast<size_t>(ctx->st->nQr) * nS * sizeof(double);   // the BASE carrier's row,
            if (!ctx->dQrPilotQ.p) ctx->dQrPilotQ.alloc(need);   // BEFORE any RL mode seeding rewrites it
            if (!ctx->dQrSavedVT.p) ctx->dQrSavedVT.alloc(need);
            if (!ctx->dQrVeqOff.p) ctx->dQrVeqOff.alloc(static_cast<size_t>(ctx->st->nQr) * nS * sizeof(char));
            CUDA_CHECK(cudaMemsetAsync(ctx->dQrVeqOff.p, 0,      // per-solve state, like limitType
                    static_cast<size_t>(ctx->st->nQr) * nS * sizeof(char), ctx->stream));
            capturePilotQBatch<<<grid(ctx->st->nQr * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nQr, nS, n,
                    ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>(), dBaseMode.as<int>(),
                    ctx->dTarget.as<double>(), ctx->dQrPilotQ.as<double>());
        }
        if (rlStateIn != nullptr && ctx->nrl > 0) {
            // WARM CONTINUATION (operator strategies stage 2): seed each (controller, scenario)'s
            // pinned limit type AND switch count from a previous solve's exported state, so the
            // strategy scenario continues the contingency scenario's outer-loop trajectory exactly
            // like the CPU's per-contingency action solve does.
            std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS), sc(static_cast<size_t>(ctx->nrl) * nS);
            for (size_t k = 0; k < lt.size(); k++) {
                lt[k] = static_cast<char>(rlStateIn[k] & 3);
                sc[k] = static_cast<char>(rlStateIn[k] >> 2);
            }
            CUDA_CHECK(cudaMemcpyAsync(ctx->dRlLimitType.p, lt.data(), lt.size(), cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpy(ctx->dRlSwitchCount.p, sc.data(), sc.size(), cudaMemcpyHostToDevice));
            if (ctx->ncv > 0) {                              // curve limits at the CURRENT (seeded) targets
                rlCurveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, ctx->ncv, n, ctx->cvExtrapolate,
                        ctx->dCvIdx.as<int>(), ctx->dCvIn.as<double>(), ctx->dCvPts.as<double>(),
                        ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(), ctx->dTarget.as<double>(),
                        ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>());
            }
            seedRlModeFromState<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n,
                    ctx->dRlVrow.as<int>(),
                    ctx->ncv > 0 ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                    ctx->ncv > 0 ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                    ctx->dRlLoadQ.as<double>(), ctx->dRlLimitType.as<char>(),
                    ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->ncv > 0 ? 1 : 0,
                    ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr);
        } else {
            CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitchCount.p, 0, static_cast<size_t>(ctx->nrl) * nS * sizeof(char), ctx->stream));
            if (ctx->nrl > 0) {                              // seed the pinned limit for base-PQ controllers (else all PV)
                initRlLimitTypeBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n, ctx->dRlVrow.as<int>(),
                        ctx->dRlVtarget.as<double>(), ctx->dRlMonitored.as<char>(), dBaseMode.as<int>(), ctx->dX.as<double>(),
                        ctx->dRlLimitType.as<char>(),
                        nRlVtOv, nRlVtOv > 0 ? dRlVtOvSlotIdx.as<int>() : nullptr,
                        nRlVtOv > 0 ? dRlVtOvVal.as<double>() : nullptr,
                        ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                        ctx->nRlRemote > 0 ? ctx->dRlCmpVrow.as<int>() : nullptr,
                        ctx->dTarget.as<double>(), ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(),
                        ctx->dRlLoadQ.as<double>());
            }
        }
        if (ctx->nds > 0) {                                  // slack-retained share: 0 until frozen at the first pass
            CUDA_CHECK(cudaMemsetAsync(ctx->dDsRetain.p, 0, nS * sizeof(double), ctx->stream));
        }
        if (ctx->dsSlackOnly) {
            if (dsSlackShift != nullptr) {                   // strategy pairs: seed with the parent contingency's
                CUDA_CHECK(cudaMemcpyAsync(ctx->dDsSlackAbsorbed.p, dsSlackShift,   // absorbed total (the device
                        nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream)); // analogue of the CPU re-anchor)
            } else {
                CUDA_CHECK(cudaMemsetAsync(ctx->dDsSlackAbsorbed.p, 0, nS * sizeof(double), ctx->stream));
            }
        }
    }

    // SEQUENTIAL outer loops, exactly like OLF's AcloadFlowEngine: fully stabilize the reactive-limits loop
    // (switch + re-solve until it switches nothing), THEN fully stabilize the distributed-slack loop
    // (distribute + re-solve until balanced), and repeat the whole RL->DS sequence until a full sequence
    // triggers no inner Newton re-solve. Running RL and DS in the SAME pass converges to a different (still
    // in-tolerance) point when switching interacts with slack distribution. phase 0 = reactive limits, 1 = slack.
    std::vector<char> live(active);                          // scenarios still iterating outer loops (dropped on failure)
    // OLF's DefaultAcOuterLoopConfig orders DistributedSlack BEFORE ReactiveLimits (innermost first), and the
    // engine stabilizes them in that order. Match it: phase 0 = distributed slack, phase 1 = reactive limits.
    // (Switching PV->PQ before the slack is balanced leaves a large active imbalance that the undamped device
    // Newton diverges on — DS must stabilize first.)
    int phase = (ctx->nds > 0 || ctx->dsSlackOnly) ? 0 : 1;  // start with distributed slack if present, else reactive limits
    bool seqChanged = false;                                 // did any outer loop re-solve during the current RL->DS sequence
    std::vector<int> prodIters(nS, 0);                       // PER-SCENARIO productive outer iterations: OLF's
                                                             // maxOuterLoopIterations is a per-RUN budget — a global
                                                             // counter capped busy chunks collectively (violation-diff
                                                             // root cause: a scenario solved solo matched the CPU
                                                             // exactly but diverged inside a full 500-chunk)
    // A phase transition after a NO-SWITCH pass must NOT re-solve: OLF's runOuterLoop re-solves only on an
    // UNSTABLE check — a STABLE loop is followed by checking the NEXT loop against the SAME state. The device
    // Newton forces >=1 iteration per solve (matching OLF's per-run behavior), so an extra no-op pass would
    // move an already-converged state off the CPU's by ~the stopping residual (measured 4e-4 kV on the
    // DS->RL transition with a distributed base). solvePass=false skips the inner Newton for such a pass.
    bool solvePass = true;
    for (int outer = 0; outer < maxOuter; outer++) {
    std::copy(active.begin(), active.end(), wasActive.begin());   // scenarios (re)solved in this inner pass
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    int nActive = 0;
    for (char a : active) {
        if (a) nActive++;
    }
    if (solvePass) {

    // ---- batched fill for all active scenarios (J + F, or F-ONLY: the final residual fill and
    //      mode 4's solve-only iterations evaluate just the residual and skip the vals memset —
    //      the J buffer is never read before the next full fill re-zeroes it). A lambda so the
    //      LINE_SEARCH backtrack below can re-evaluate at a reduced step, like the sequential
    //      solveContext's fillAndNorm. ----
    auto fillActiveBatch = [&](bool fOnly) {
        if (!fOnly) {
            CUDA_CHECK(cudaMemsetAsync(ctx->dVals.p, 0, static_cast<size_t>(nnz) * nS * sizeof(double), ctx->stream));
        }
        initFBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dTarget.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
        if (ctx->st->cb.ne) {
            if (fOnly) {
                fillClosedBranchBatchF<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->st->cbTrig.as<double>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->st->cb.altDest.as<int>(), ctx->st->cb.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->cb.dynReloc ? 1 : 0, ctx->st->cb.phaseCtl ? ctx->st->cb.pcDest.as<int>() : nullptr, ctx->st->cb.phaseCtl ? ctx->st->cb.pcSel.as<int>() : nullptr, relocOf(ctx->st->cb, ctx->st.get()));
            } else {
                fillClosedBranchBatch<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->st->cb.altDest.as<int>(), ctx->st->cb.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->cb.dynReloc ? 1 : 0, ctx->st->cb.phaseCtl ? ctx->st->cb.pcDest.as<int>() : nullptr, ctx->st->cb.phaseCtl ? ctx->st->cb.pcSel.as<int>() : nullptr, relocOf(ctx->st->cb, ctx->st.get()));
            }
        }
        if (ctx->st->ob.ne) {
            if (fOnly) {
                fillOpenBranchBatchF<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->ob.altDest.as<int>(), ctx->st->ob.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ob.dynReloc ? 1 : 0, relocOf(ctx->st->ob, ctx->st.get()));
            } else {
                fillOpenBranchBatch<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->ob.altDest.as<int>(), ctx->st->ob.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ob.dynReloc ? 1 : 0, relocOf(ctx->st->ob, ctx->st.get()));
            }
        }
        if (ctx->st->sh.ne) {
            if (fOnly) {
                fillShuntBatchF<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), ctx->st->sh.altDest.as<int>(), ctx->st->sh.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->sh.dynReloc ? 1 : 0, relocOf(ctx->st->sh, ctx->st.get()));
            } else {
                fillShuntBatch<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), ctx->st->sh.altDest.as<int>(), ctx->st->sh.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->sh.dynReloc ? 1 : 0, relocOf(ctx->st->sh, ctx->st.get()));
            }
        }
        // voltage-dependent load model: own guarded block (NOT inside the shunt guard — a modeled
        // load needs no shunt present); residual always, Jacobian slot unless the F-only pass
        if (ctx->st->nvdl) {
            fillVdlBatch<<<grid(ctx->st->nvdl * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nvdl, nS, n, nnz, ctx->st->dVdlIn.as<double>(), ctx->st->dVdlIdx.as<int>(), ctx->st->dVdlDest.as<int>(), ctx->dX.as<double>(), fOnly ? nullptr : ctx->dVals.as<double>(), ctx->dF.as<double>());
        }
        if (ctx->st->hv.ne) {
            if (fOnly) {
                fillHvdcBatchF<<<grid(ctx->st->hv.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->hv.ne, nS, n, nnz, ctx->nElements, ctx->hvOffset, ctx->st->hv.in.as<double>(), ctx->st->hv.ix.as<int>(), ctx->st->hv.dest.as<int>(), ctx->dX.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>());
            } else {
                fillHvdcBatch<<<grid(ctx->st->hv.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->hv.ne, nS, n, nnz, ctx->nElements, ctx->hvOffset, ctx->st->hv.in.as<double>(), ctx->st->hv.ix.as<int>(), ctx->st->hv.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>());
            }
        }
        if (ctx->ndq) fillDistrQBatch<<<grid(ctx->ndq * nS), BLOCK, 0, ctx->stream>>>(ctx->ndq, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->obOffset, ctx->st->dDqElem.as<int>(), ctx->st->dDqSide.as<int>(), ctx->st->dDqKind.as<int>(), ctx->st->dDqWeight.as<double>(), ctx->st->dDqDest.as<int>(), ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nDqOv, ctx->nDqOv ? ctx->dDqOvIdx.as<int>() : nullptr, ctx->nDqOv ? ctx->dDqOvW.as<double>() : nullptr, ctx->dMode.as<int>(), ctx->nDqDyn ? ctx->dDqKMember.as<int>() : nullptr, ctx->nDqDyn || ctx->hasDqQrGroups ? ctx->dDqJMember.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberKey.as<double>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberRlEntry.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupStart.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupCount.as<int>() : nullptr, ctx->dRlLimitType.as<char>(), ctx->nrl, ctx->hasDqQrGroups ? ctx->dDqMemberQrGroup.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr);
        if (ctx->ndr) fillDistrRhoBatch<<<grid(ctx->ndr * nS), BLOCK, 0, ctx->stream>>>(ctx->ndr, nS, n, nnz, ctx->st->dDrCol.as<int>(), ctx->st->dDrCoef.as<double>(), ctx->st->dDrDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nDrOv, ctx->nDrOv ? ctx->dDrOvIdx.as<int>() : nullptr, ctx->nDrOv ? ctx->dDrOvC.as<double>() : nullptr, ctx->st->drQr ? ctx->st->dDrQrGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>());
        if (ctx->ndb) fillDistrShuntBBatch<<<grid(ctx->ndb * nS), BLOCK, 0, ctx->stream>>>(ctx->ndb, nS, n, nnz, ctx->st->dDbCol.as<int>(), ctx->st->dDbCoef.as<double>(), ctx->st->dDbDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
        if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), ctx->st->dZiAltDest.as<int>(), ctx->st->dZiSelRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ziDyn ? 1 : 0, ziRelocOf(ctx->st.get()));
        applyRowModeBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->dMode.as<int>(), ctx->st->dRowPtr.as<int>(), ctx->st->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dRowSlope.as<double>());
        if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: stream-ordered into the solve, no host sync
    };
    auto readNormsBatch = [&](std::vector<double>& hn) {
        residualNormsBatch<<<nS, BLOCK, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dNorms.as<double>(),
                ctx->perRowCriteria ? ctx->dThrIdent.as<double>() : nullptr,
                ctx->perRowCriteria ? ctx->dThrPower.as<double>() : nullptr, ctx->dMode.as<int>());
        hn.resize(static_cast<size_t>(nS) * 4);
        CUDA_CHECK(cudaMemcpy(hn.data(), ctx->dNorms.p, hn.size() * sizeof(double), cudaMemcpyDeviceToHost));
    };
    // LINE_SEARCH per-scenario backtracking state (mirrors the sequential solveContext: lastNorm2 /
    // haveLast, per pass — a phase re-entry retargets the system, voiding the step relation)
    std::vector<double> lsLastSsq(lsMode == 1 ? nS : 0, 0.0);
    std::vector<char> lsHaveLast(lsMode == 1 ? nS : 0, 0);
    std::vector<double> hMu(lsMode == 1 ? nS : 0, -1.0);

    for (int it = 0; it <= maxIter && nActive > 0; it++) {
        bool fOnly = it == maxIter || (ctx->mode == 4 && it > 0);
        fillActiveBatch(fOnly);

        // ---- residual check + retirement ----
        // Normal mode (tol > 0): every iteration copies the batched mismatch to the host (the
        // only thing crossing PCIe) and retires converged/diverged scenarios.
        // Device fixed-iteration mode (tol <= 0): NO per-iteration host sync and NO retirement
        // — every scenario runs exactly maxIter steps on the device; the mismatch is read back
        // ONCE, on the final pass, only to report ||F||inf. This drops the per-iteration D->H
        // copy of the full batched mismatch. Tradeoff: a scenario going non-finite is not
        // isolated, so it can poison a uniform-batched factorization — use tol > 0 if
        // scenarios may diverge.
        if (!deviceFixedIter || it == maxIter) {
            std::vector<double> hNorms;
            if (!deviceFixedIter) {
                // Device-side reduction: read back nS*4 norms, not the 70-MB mismatch.
                readNormsBatch(hNorms);
                // ---- LINE_SEARCH backtrack (per scenario): the full step applied at it-1 may have
                //      RAISED that scenario's ||F||2 — walk it back to x = xPrev + mu*dx with
                //      mu = stepFold^-k until the 2-norm improves, mirroring the sequential
                //      solveContext / OLF's LineSearchStateVectorScaling. Judged on the 2-norm
                //      (ssq, monotonic-equivalent); a non-finite mismatch backtracks too. Scenarios
                //      not backtracking keep their state (mu sentinel -1 masks the step kernel). ----
                if (lsMode == 1) {
                    auto needsBt = [&](int s2) {
                        return active[s2] && lsHaveLast[s2]
                                && (hNorms[s2 * 4 + 2] == 0.0 || hNorms[s2 * 4 + 1] >= lsLastSsq[s2]);
                    };
                    bool anyBt = false;
                    for (int s2 = 0; s2 < nS; s2++) {
                        if (needsBt(s2)) anyBt = true;
                    }
                    for (int k = 1; anyBt && k <= ctx->lsMaxIter; k++) {
                        double mu = 1.0 / std::pow(ctx->lsStepFold, k);
                        for (int s2 = 0; s2 < nS; s2++) hMu[s2] = needsBt(s2) ? mu : -1.0;
                        CUDA_CHECK(cudaMemcpyAsync(ctx->dStepFactor.p, hMu.data(), nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
                        stepFromPrevBatchMasked<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS,
                                ctx->dXprevLs.as<double>(), ctx->dDx.as<double>(), ctx->dStepFactor.as<double>(), ctx->dX.as<double>());
                        fillActiveBatch(false);              // full fill: J must sit at the accepted x (like the seq path)
                        readNormsBatch(hNorms);
                        anyBt = false;
                        for (int s2 = 0; s2 < nS; s2++) {
                            if (needsBt(s2)) anyBt = true;
                        }
                    }
                    for (int s2 = 0; s2 < nS; s2++) {
                        if (active[s2]) {
                            lsLastSsq[s2] = hNorms[s2 * 4 + 1];
                            lsHaveLast[s2] = 1;
                        }
                    }
                }
            } else {
                CUDA_CHECK(cudaMemcpy(hF.data(), ctx->dF.p, nTot * sizeof(double), cudaMemcpyDeviceToHost));
            }
            bool anyChanged = false;
            // Inner-Newton convergence test. Default (convEpsPerEq <= 0): the device infinity-norm floor
            // ‖F‖∞ < tol (stricter than OLF's usual criterion). When convEpsPerEq > 0: OLF's EXACT criterion
            // — DefaultNewtonRaphsonStoppingCriteria stops at ‖F‖₂ < sqrt(convEpsPerEq² · n) — so a batched SA
            // can be run with the SAME stopping decision as the CPU loadflow for an apples-to-apples timing
            // comparison. ‖F‖∞ is still recorded in res.norm[s] for reporting either way. n counts the padded
            // system rows; for batched scenarios (no islanding — those go to the CPU) that equals OLF's active
            // equation count, since disabled-branch rows keep the bus equations and identity pads contribute 0.
            const double olf2NormThreshold = convEpsPerEq > 0.0 ? std::sqrt(convEpsPerEq * convEpsPerEq * n) : 0.0;
            auto thr2Of = [&](int sIdx) {
                return convEpsPerEq > 0.0 && !ctx->hNActive.empty()
                        ? std::sqrt(convEpsPerEq * convEpsPerEq * ctx->hNActive[sIdx]) : olf2NormThreshold;
            };
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                double norm = 0.0;
                double ssq = 0.0;
                bool finite = true;
                double maxRatio = 0.0;
                if (!deviceFixedIter) {
                    norm = hNorms[(size_t) s * 4];
                    ssq = hNorms[(size_t) s * 4 + 1];
                    finite = hNorms[(size_t) s * 4 + 2] != 0.0;
                    maxRatio = hNorms[(size_t) s * 4 + 3];
                } else {
                    for (int i = 0; i < n; i++) {
                        double v = hF[(size_t) s * n + i];
                        if (!std::isfinite(v)) {
                            finite = false;
                            break;
                        }
                        norm = std::max(norm, std::fabs(v));
                        ssq += v * v;
                    }
                }
                // PER_EQUATION_TYPE (per-row thresholds set on the context): OLF's
                // PerEquationTypeStoppingCriteria — stop when EVERY row clears its type threshold.
                bool converged = ctx->perRowCriteria ? (finite && maxRatio < 1.0)
                        : convEpsPerEq > 0.0 ? (std::sqrt(ssq) < thr2Of(s)) : (norm < tol);
                if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
                    std::fprintf(stderr, "BATCH_TRACE s=%d it=%d infnorm=%.17e norm2=%.17e thr=%.17e conv=%d\n",
                                 s, it, norm, std::sqrt(ssq), thr2Of(s), converged ? 1 : 0);
                }
                res.iters[s] = it;
                res.norm[s] = norm;
                if (deviceFixedIter) {
                    res.status[s] = finite ? 1 : -1;         // ran maxIter; no convergence test
                } else if (!finite) {                        // a diverged scenario must not poison the batch
                    res.status[s] = -1;
                    active[s] = 0;
                    live[s] = 0;                             // failed: stop iterating its outer loops
                    nActive--;
                    anyChanged = true;
                } else if (converged && it > 0) {            // force >=1 iteration, like CPU NewtonRaphson /
                    res.status[s] = 0;                       // solveContext (line ~1227): a warm-started state may
                    active[s] = 0;                           // already meet the global 2-norm threshold yet leave a
                    nActive--;                               // just-flipped identity row (PV target) unpinned — one
                    anyChanged = true;                       // Newton step pins it. Retiring at it==0 skips that solve.
                } else if (it == maxIter) {
                    res.status[s] = 1;                       // exhausted iterations
                    active[s] = 0;
                    live[s] = 0;                             // did not converge: stop iterating its outer loops
                    nActive--;
                    anyChanged = true;
                }
            }
            if (!deviceFixedIter && anyChanged) {
                CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
            }
        }
        if (deviceFixedIter) {
            if (it == maxIter) {
                break;                                       // final residual recorded; done
            }
        } else if (nActive == 0) {
            break;
        } else if (stragglerExit > 0 && nActive <= stragglerExit && it * 2 >= maxIter) {
            // The few remaining unconverged scenarios would keep the FULL-width batch iterating;
            // hand them to the sequential fallback instead (status -2 = STATUS_REJECTED on the
            // Java side, same routing as an extractor rejection). Their outer loops stop here.
            // BOTH gates matter: few actives AND deep into the iteration budget — someone is
            // always last to retire in a healthy pass (it ~5-8), only cap-bound stragglers
            // (it >= maxIter/2) get offloaded.
            for (int s = 0; s < nS; s++) {
                if (active[s]) {
                    res.status[s] = -2;
                    res.iters[s] = it;
                    active[s] = 0;
                    live[s] = 0;
                    nActive--;
                }
            }
            break;
        }

        if (lsMode == 1) {                                   // LINE_SEARCH: remember the pre-step iterates
            CUDA_CHECK(cudaMemcpyAsync(ctx->dXprevLs.p, ctx->dX.p, static_cast<size_t>(n) * nS * sizeof(double),
                                       cudaMemcpyDeviceToDevice, ctx->stream));
        }
        if (ctx->mode == 1) {
        // ---- UNIFORM BATCHED solve: one ANALYSIS on the shared pattern, then ONE batched
        //      FACTORIZATION + SOLVE over every scenario's value set. Retired scenarios are
        //      frozen to identity so the batch stays non-singular. ----
        writeIdentityForInactive<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                ctx->st->dDiagPos.as<int>(), ctx->dActive.as<char>(), ctx->dVals.as<double>(), ctx->dF.as<double>());
        negateIntoBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dF.as<double>(), ctx->dB.as<double>());
        if (!ctx->analyzed) {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matABatch, ctx->matXBatch, ctx->matBBatch));
            traceLuNnz(ctx->ds.handle, ctx->ds.data, "batch-api");
            ctx->analyzed = true;
        }
        CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                ctx->ds.config, ctx->ds.data, ctx->matABatch, ctx->matXBatch, ctx->matBBatch));
        if (ctx->clampStep) {
            maxVoltageChangeFactorBatch<<<nS, BLOCK, 0, ctx->stream>>>(n, ctx->dDx.as<double>(), ctx->dVarType.as<int>(), ctx->dActive.as<char>(), ctx->maxDv, ctx->maxDphi, ctx->dStepFactor.as<double>());
            axpy1BatchScaled<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS, ctx->dDx.as<double>(), ctx->dStepFactor.as<double>(), ctx->dX.as<double>());
        } else {
            axpy1Batch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dDx.as<double>(), ctx->dX.as<double>());
        }
        if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: no per-iter host sync (final readback syncs)
        } else if (ctx->mode == 2) {
        // ---- BLOCK-DIAGONAL solve: one big blkdiag matrix; ANALYZE+FACTORIZE once (ever, across
        //      chunks), then REFACTORIZE every iteration reusing the pivot order (the
        //      refactor-every strategy) — refactor IS cheap on a single matrix, unlike the batch API. ----
        writeIdentityForInactive<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                ctx->st->dDiagPos.as<int>(), ctx->dActive.as<char>(), ctx->dVals.as<double>(), ctx->dF.as<double>());
        negateIntoBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dF.as<double>(), ctx->dB.as<double>());
        if (!ctx->analyzed) {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matBig, ctx->matBigX, ctx->matBigB));
            traceLuNnz(ctx->ds.handle, ctx->ds.data, "blkdiag");
            ctx->analyzed = true;
        }
        CUDSS_CHECK(cudssExecute(ctx->ds.handle,
                (ctx->factored ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION) | CUDSS_PHASE_SOLVE,
                ctx->ds.config, ctx->ds.data, ctx->matBig, ctx->matBigX, ctx->matBigB));
        ctx->factored = true;
        if (ctx->clampStep) {
            maxVoltageChangeFactorBatch<<<nS, BLOCK, 0, ctx->stream>>>(n, ctx->dDx.as<double>(), ctx->dVarType.as<int>(), ctx->dActive.as<char>(), ctx->maxDv, ctx->maxDphi, ctx->dStepFactor.as<double>());
            axpy1BatchScaled<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS, ctx->dDx.as<double>(), ctx->dStepFactor.as<double>(), ctx->dX.as<double>());
        } else {
            axpy1Batch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dDx.as<double>(), ctx->dX.as<double>());
        }
        if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: no per-iter host sync (final readback syncs)
        } else if (ctx->mode == 3 || ctx->mode == 4 || ctx->mode == 5) {
        // ---- UBATCH solve: ANALYZE one block once (cheap), then a linear solve over all nS systems
        //      via CUDSS_CONFIG_UBATCH_SIZE on the single per-block matrix. The FACTORIZE policy varies:
        //        mode 3 = FACTORIZE every iteration (honest Newton, DirectRefactorEvery analog);
        //        mode 4 = FACTORIZE only at this chunk's iter 0, SOLVE-only after (DirectIter0Only:
        //                 the J is frozen at the AC-hot-started point, the RHS/F is still fresh each iter);
        //        mode 5 = FACTORIZE once (ever), REFACTORIZE after (DirectRefactorEvery on UBATCH:
        //                 reuse the pivot order, recompute numeric values each iter). ----
        writeIdentityForInactive<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                ctx->st->dDiagPos.as<int>(), ctx->dActive.as<char>(), ctx->dVals.as<double>(), ctx->dF.as<double>());
        negateIntoBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dF.as<double>(), ctx->dB.as<double>());
        if (!ctx->analyzed) {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
            traceLuNnz(ctx->ds.handle, ctx->ds.data, "ubatch");
            ctx->analyzed = true;
        }
        // ONE cudssExecute per iteration (phases are OR-able bit flags): halves the per-iteration
        // host-side phase overhead vs separate FACTORIZATION + SOLVE calls.
        int ubPhase = CUDSS_PHASE_SOLVE;
        if (ctx->mode == 5) {
            ubPhase |= ctx->factored ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
            ctx->factored = true;
        } else if (ctx->mode == 3 || it == 0) {              // mode 3 every iter; mode 4 only at chunk iter 0
            ubPhase |= CUDSS_PHASE_FACTORIZATION;
        }
        // ---- solve-only COMPACTION: when few scenarios remain active, gather their vals/B into
        //      compact buffers and solve a NARROWER ubatch on the dedicated dsC context, then
        //      scatter dx back. mode 3 only (factorizes every iteration — stateless across width
        //      changes); width quantized to power-of-2 tiers so the cheap single-block re-ANALYSIS
        //      runs only on tier change. Pad slots duplicate the first active scenario (keeps the
        //      batch non-singular) and are discarded by the scatter. dDx is zeroed first so
        //      inactive scenarios' frozen states are not perturbed by stale solutions (the
        //      full-width update kernel adds dx over ALL scenarios). ----
        bool compactSolve = !deviceFixedIter && ctx->mode == 3 && nActive > 0 && nActive * 2 <= nS;
        if (compactSolve) {
            if (ctx->compactCap == 0) {                       // lazy first-engage setup
                ctx->compactCap = nS / 2;
                ctx->dValsC.alloc(static_cast<size_t>(nnz) * ctx->compactCap * sizeof(double));
                ctx->dBC.alloc(static_cast<size_t>(n) * ctx->compactCap * sizeof(double));
                ctx->dDxC.alloc(static_cast<size_t>(n) * ctx->compactCap * sizeof(double));
                ctx->dMapC.alloc(static_cast<size_t>(ctx->compactCap) * sizeof(int));
                CUDSS_CHECK(cudssCreate(&ctx->dsC.handle));
                CUDSS_CHECK(cudssSetStream(ctx->dsC.handle, ctx->stream));
                CUDSS_CHECK(cudssConfigCreate(&ctx->dsC.config));
                applyReorderAlg(ctx->dsC.config);
                int irC = irSteps();
                CUDSS_CHECK(cudssConfigSet(ctx->dsC.config, CUDSS_CONFIG_IR_N_STEPS, &irC, sizeof(irC)));
                CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->matUbC, n, n, nnz,
                        ctx->st->dRowPtr.as<int>(), nullptr, ctx->st->dColIdx.as<int>(), ctx->dValsC.as<double>(),
                        OLF_CUDSS_CSR_TYPES, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbCB, n, 1, n, ctx->dBC.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbCX, n, 1, n, ctx->dDxC.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
            }
            std::vector<int> cmap;
            cmap.reserve(ctx->compactCap);
            for (int cs = 0; cs < nS; cs++) {
                if (active[cs]) {
                    cmap.push_back(cs);
                }
            }
            int nReal = static_cast<int>(cmap.size());
            int w = 1;
            while (w < nReal) {
                w <<= 1;
            }
            if (w > ctx->compactCap) {
                w = ctx->compactCap;
            }
            while (static_cast<int>(cmap.size()) < w) {
                cmap.push_back(cmap[0]);
            }
            CUDA_CHECK(cudaMemcpyAsync(ctx->dMapC.p, cmap.data(), w * sizeof(int), cudaMemcpyHostToDevice, ctx->stream));
            gatherScenarios<<<grid(nnz * w), BLOCK, 0, ctx->stream>>>(nnz, w, ctx->dMapC.as<int>(), ctx->dVals.as<double>(), ctx->dValsC.as<double>());
            gatherScenarios<<<grid(n * w), BLOCK, 0, ctx->stream>>>(n, w, ctx->dMapC.as<int>(), ctx->dB.as<double>(), ctx->dBC.as<double>());
            if (w != ctx->compactWidth) {                     // tier change: fresh data + single-block ANALYSIS
                if (ctx->dsC.data) {
                    CUDSS_CHECK(cudssDataDestroy(ctx->dsC.handle, ctx->dsC.data));
                    ctx->dsC.data = nullptr;
                }
                CUDSS_CHECK(cudssDataCreate(ctx->dsC.handle, &ctx->dsC.data));
                CUDSS_CHECK(cudssConfigSet(ctx->dsC.config, CUDSS_CONFIG_UBATCH_SIZE, &w, sizeof(w)));
                CUDSS_CHECK(cudssExecute(ctx->dsC.handle, CUDSS_PHASE_ANALYSIS, ctx->dsC.config, ctx->dsC.data, ctx->matUbC, ctx->matUbCX, ctx->matUbCB));
                ctx->compactWidth = w;
            }
            CUDSS_CHECK(cudssExecute(ctx->dsC.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, ctx->dsC.config, ctx->dsC.data, ctx->matUbC, ctx->matUbCX, ctx->matUbCB));
            CUDA_CHECK(cudaMemsetAsync(ctx->dDx.p, 0, static_cast<size_t>(n) * nS * sizeof(double), ctx->stream));
            scatterScenarios<<<grid(n * nReal), BLOCK, 0, ctx->stream>>>(n, nReal, ctx->dMapC.as<int>(), ctx->dDxC.as<double>(), ctx->dDx.as<double>());
        } else {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, ubPhase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
        }
        if (ctx->clampStep) {
            maxVoltageChangeFactorBatch<<<nS, BLOCK, 0, ctx->stream>>>(n, ctx->dDx.as<double>(), ctx->dVarType.as<int>(), ctx->dActive.as<char>(), ctx->maxDv, ctx->maxDphi, ctx->dStepFactor.as<double>());
            axpy1BatchScaled<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS, ctx->dDx.as<double>(), ctx->dStepFactor.as<double>(), ctx->dX.as<double>());
        } else {
            axpy1Batch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dDx.as<double>(), ctx->dX.as<double>());
        }
        if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: no per-iter host sync (final readback syncs)
        } else {
        // ---- fallback: analyze once (shared pattern), refactorize + solve per scenario ----
        for (int s = 0; s < nS; s++) {
            if (!active[s]) continue;
            double* valsS = ctx->dVals.as<double>() + (size_t) s * nnz;
            double* bS = ctx->dB.as<double>() + (size_t) s * n;
            double* dxS = ctx->dDx.as<double>() + (size_t) s * n;
            double* fS = ctx->dF.as<double>() + (size_t) s * n;
            negateIntoOne<<<grid(n), BLOCK, 0, ctx->stream>>>(n, fS, bS);
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matA, valsS));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matB, bS));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matX, dxS));
            if (!ctx->analyzed) {
                CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB));
                traceLuNnz(ctx->ds.handle, ctx->ds.data, "per-scenario");
                ctx->analyzed = true;
            }
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                    ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB));
            double* xS = ctx->dX.as<double>() + (size_t) s * n;
            axpy1One<<<grid(n), BLOCK, 0, ctx->stream>>>(n, dxS, xS);
        }
        if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: no per-iter host sync
        }
    }   // ---- end inner Newton loop ----
    }   // ---- end if (solvePass) ----

    if (!fullLoadflow) {
        break;                                              // fixed-iter / no controllers: one pass only
    }

    // ---- REACTIVE LIMITS pass: switch any local PV controller over its Q limit to PQ ----
    // Only scenarios (re)solved this pass that converged can have new violations.
    for (int s = 0; s < nS; s++) {
        rlActive[s] = (wasActive[s] && res.status[s] == 0) ? 1 : 0;
    }
    CUDA_CHECK(cudaMemcpy(ctx->dRlActive.p, rlActive.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    // generator Q per controller = (Σ branch q + shunt q) at its BUS_V row + load Q. Compute the first
    // part by a fill into a ZEROED buffer: no initF (−target), no applyRowMode, so qBuf[row] is the raw
    // bus reactive balance. Disabled branches are skipped (outage); dVals is scratch (the next inner
    // iteration memsets and refills it).
    CUDA_CHECK(cudaMemsetAsync(ctx->dQbuf.p, 0, nTot * sizeof(double), ctx->stream));
    // Reactive-limit Q buffer: physical bus-Q for PV-limit detection — keep the FROZEN routing
    // (dynReloc=0), the relocation only concerns the Newton residual, not this diagnostic Q.
    if (ctx->st->cb.ne) fillClosedBranchBatch<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), nullptr, nullptr, nullptr, 0, nullptr, nullptr, RelocTables{});
    if (ctx->st->ob.ne) fillOpenBranchBatch<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), nullptr, nullptr, nullptr, 0, RelocTables{});
    if (ctx->st->sh.ne) fillShuntBatch<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), nullptr, nullptr, nullptr, 0, RelocTables{});
    if (ctx->st->nvdl) fillVdlBatch<<<grid(ctx->st->nvdl * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nvdl, nS, n, nnz, ctx->st->dVdlIn.as<double>(), ctx->st->dVdlIdx.as<int>(), ctx->st->dVdlDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>());
    // Zero-impedance dummy transfers MUST be in the Q/P buffer too: the distributed-slack retain
    // reads the slack row's balance from dQbuf — without the coupler's dummy-P the DS sees a
    // phantom mismatch equal to the through-flow and distributes it FOREVER (measured: +392 MW
    // runaway on the first switch-close scenario until the per-scenario budget rejected it).
    if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), nullptr, nullptr, nullptr, 0, RelocTables{});
    // Remote-member OWN q (plan-B3): fill each member's own reactive injection into its (suppressed,
    // hence free) qBuf row so the RL kernel evaluates remote entries like local ones.
    if (ctx->nRmq > 0) {
        fillRemoteMemberQBatch<<<grid(ctx->nRmq * nS), BLOCK, 0, ctx->stream>>>(ctx->nRmq, nS, n,
                ctx->nElements, ctx->cbOffset, ctx->obOffset,
                ctx->dRmqRow.as<int>(), ctx->dRmqElem.as<int>(), ctx->dRmqSide.as<int>(),
                ctx->dRmqKind.as<int>(), ctx->dRmqWeight.as<double>(),
                ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(),
                ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(),
                ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(),
                ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>());
    }
    CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitched.p, 0, nS * sizeof(char), ctx->stream));
    // Apply ONLY the current phase's outer loop this pass (sequential, like OLF) — never both at once.
    bool inDsPhase = (phase == 0);
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        std::fprintf(stderr, "BATCH_TRACE pass done: outer=%d phase=%d (0=DS,1=RL)\n", outer, phase);
    }
    if (inDsPhase) {                                       // distributed-slack outer loop: spread the slack mismatch
        if (ctx->dsSlackOnly) {
            slackOnlyDsCheck<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, ctx->dsSlackPhiRow, ctx->dsSlackTargetP,
                    SLACK_DIST_TOL, ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(),
                    ctx->dDsSlackAbsorbed.as<double>(), ctx->dRlSwitched.as<char>());
        }
        if (ctx->nds > 0) {
            // Recompute the slack's retained share from the CURRENT total imbalance every pass (NOT frozen at
            // pass 0) so a participating slack tracks the loss change during redistribution.
            computeSlackRetain<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, ctx->nds, ctx->dsSlackPhiRow, ctx->dsSlackTargetP,
                    1.0 - ctx->dsFactorSum, ctx->dDsPhiRow.as<int>(), ctx->dDsBaseTargetP.as<double>(),
                    ctx->dTarget.as<double>(), ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(),
                    ctx->dDsRetain.as<double>(),
                    hasDsOverride ? dDsOvBus.as<int>() : nullptr, dDsOvBaseDelta.as<double>(),
                    hasDsAnchor ? dDsAnchorShift.as<double>() : nullptr,
                    hasDsAnchor ? dDsSlackShift.as<double>() : nullptr,
                    ctx->dsMode, ctx->dsMode == 1 ? ctx->dDsMarginUp.as<double>() : nullptr,
                    ctx->dsMode == 1 ? ctx->dDsMarginDown.as<double>() : nullptr,
                    ctx->dsSlackMarginUp, ctx->dsSlackMarginDown);
            if (!ctx->dDsResidual.p) ctx->dDsResidual.alloc(nS * sizeof(char));
            CUDA_CHECK(cudaMemsetAsync(ctx->dDsResidual.p, 0, nS * sizeof(char), ctx->stream));
            distributeSlackBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nds, nS, n, SLACK_DIST_TOL, P_RESIDUE_EPS,
                    ctx->dsSlackPhiRow, ctx->dsSlackTargetP, ctx->dDsPhiRow.as<int>(),
                    ctx->dDsFactor.as<double>(), ctx->dDsBaseTargetP.as<double>(),
                    hasDsAnchor ? dDsAnchorShift.as<double>() : nullptr,
                    hasDsAnchor ? dDsSlackShift.as<double>() : nullptr,
                    ctx->dDsMaxDelta.as<double>(), ctx->dDsMinDelta.as<double>(), ctx->dDsRetain.as<double>(),
                    ctx->dQbuf.as<double>(), ctx->dRlActive.as<char>(),
                    hasDsOverride ? dDsOvBus.as<int>() : nullptr, dDsOvFactorDelta.as<double>(),
                    dDsOvBaseDelta.as<double>(), dDsOvMaxDelta.as<double>(), dDsOvMinDelta.as<double>(),
                    ctx->dsMode, ctx->dsMode == 1 ? ctx->dDsMarginUp.as<double>() : nullptr,
                    ctx->dsMode == 1 ? ctx->dDsMarginDown.as<double>() : nullptr,
                    ctx->dsSlackMarginUp, ctx->dsSlackMarginDown,
                    ctx->dTarget.as<double>(), ctx->dRlSwitched.as<char>(), ctx->dDsResidual.as<char>());
        }
    } else {                                               // reactive-limits outer loop: switch PV->PQ over Q limit
        if (ctx->nrl > 0 && !skipRlPhase) {                  // skipRlPhase: a monitoring interleave solve — the
            if (ctx->ncv > 0) {                              // curve limits at the CURRENT (post-DS) targets
                rlCurveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, ctx->ncv, n, ctx->cvExtrapolate,
                        ctx->dCvIdx.as<int>(), ctx->dCvIn.as<double>(), ctx->dCvPts.as<double>(),
                        ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(), ctx->dTarget.as<double>(),
                        ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>());
            }
            reactiveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n, REACTIVE_LIMIT_TOL, RL_MAX_SWITCH,
                    ctx->dRlVrow.as<int>(),
                    ctx->ncv > 0 ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                    ctx->ncv > 0 ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                    ctx->dRlLoadQ.as<double>(),
                    ctx->dRlVtarget.as<double>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
                    ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                    ctx->dRlLimitType.as<char>(), ctx->dRlSwitchCount.as<char>(), ctx->dRlSwitched.as<char>(),
                    nRlVtOv, nRlVtOv > 0 ? dRlVtOvSlotIdx.as<int>() : nullptr,
                    nRlVtOv > 0 ? dRlVtOvVal.as<double>() : nullptr,
                    ctx->dRowSlope.as<double>(), ctx->dRlSavedTarget.as<double>(), ctx->ncv > 0 ? 1 : 0,
                    ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                    ctx->nRlRemote > 0 ? ctx->dRlCmpVrow.as<int>() : nullptr,
                    ctx->dRlSvcBmin.as<double>(), ctx->dRlSvcBmax.as<double>(), ctx->dRlV2Base.as<double>());
            if (ctx->nRlRemote > 0 && ctx->nDqMembers > 0) {
                // keep the survivors' DISTR_Q row targets in step with the pin states (plan-B3)
                updateDistrQTargetsBatch<<<grid(ctx->nDqMembers * nS), BLOCK, 0, ctx->stream>>>(
                        ctx->nDqMembers, nS, n, ctx->nrl,
                        ctx->dDqMemberKey.as<double>(), ctx->dDqMemberRlEntry.as<int>(),
                        ctx->dDqMemberGroupStart.as<int>(), ctx->dDqMemberGroupCount.as<int>(),
                        ctx->dDqMemberRow.as<int>(), ctx->dDqMemberHasRow.as<char>(),
                        ctx->dRlVrow.as<int>(),
                        ctx->ncv > 0 ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                        ctx->ncv > 0 ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                        ctx->dRlLoadQ.as<double>(), ctx->dRlLimitType.as<char>(), ctx->ncv > 0 ? 1 : 0,
                        ctx->dTarget.as<double>(), ctx->dRlActive.as<char>(),
                        ctx->hasDqQrGroups ? ctx->dDqMemberQrGroup.as<int>() : nullptr,
                        ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr,
                        ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr,
                        ctx->dMode.as<int>(),
                        ctx->hasDqQrGroups ? ctx->dQrPilotQ.as<double>() : nullptr, ctx->st->nQr);
            }
            if (ctx->hasDqQrGroups && ctx->st->nQr > 0) {    // all-pinned corner: pilot vEq flip/restore
                updatePilotVeqBatch<<<grid(ctx->st->nQr * nS), BLOCK, 0, ctx->stream>>>(
                        ctx->st->nQr, nS, n,
                        ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>(), ctx->dQrGenVc.as<char>(),
                        ctx->dQrPilotQ.as<double>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                        ctx->dQrVeqOff.as<char>(), ctx->dQrSavedVT.as<double>(),
                        ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>());
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(rlSwitched.data(), ctx->dRlSwitched.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
    // FORCED-PV detector: OLF's switchPvPq NEVER lets the network reach zero PV buses ("all PV
    // buses should switch PQ, strongest one will stay PV" — a network-wide election the device
    // does not run). The moment a scenario's RL state has EVERY non-monitored entry pinned, its
    // trajectory has already diverged from the CPU's (measured 1.9 kV) — reject it to the
    // per-contingency CPU fallback. Only reachable when every PV-capable entry has finite limits
    // (allPvPinnable, precomputed): one unlimited PV bus anywhere skips the check entirely.
    if (!inDsPhase && !skipRlPhase && ctx->nrl > 0 && ctx->allPvPinnable) {
        std::vector<char> ltHost(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(ltHost.data(), ctx->dRlLimitType.p, ltHost.size(), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            if (!rlActive[s]) {
                continue;
            }
            bool anyPv = false;
            for (int i = 0; i < ctx->nrl && !anyPv; i++) {
                anyPv = !ctx->hRlMonitored[i] && ltHost[(size_t) s * ctx->nrl + i] == 0;
            }
            if (!anyPv) {
                res.status[s] = -2;                          // STATUS_REJECTED -> CPU recomputes it
                live[s] = 0;
                active[s] = 0;
            }
        }
    }
    if (!inDsPhase && ctx->nRlRemote > 0 && std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {   // TEMP remote-RL trace
        std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS);
        std::vector<char> rem(ctx->nrl);
        std::vector<int> vr(ctx->nrl);
        std::vector<double> qb(static_cast<size_t>(n) * nS);
        std::vector<double> lq(ctx->nrl);
        CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rem.data(), ctx->dRlRemote.p, rem.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vr.data(), ctx->dRlVrow.p, vr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qb.data(), ctx->dQbuf.p, qb.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lq.data(), ctx->dRlLoadQ.p, lq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int s2 = 0; s2 < nS; s2++) {
            for (int i2 = 0; i2 < ctx->nrl; i2++) {
                if (rem[i2]) {
                    std::fprintf(stderr, "RLREMOTE_TRACE s=%d entry=%d lt=%d switched=%d q=%.6f\n",
                                 s2, i2, (int) lt[(size_t) s2 * ctx->nrl + i2], (int) rlSwitched[s2],
                                 qb[(size_t) s2 * n + vr[i2]] + lq[i2]);
                }
            }
        }
    }
    if (inDsPhase && ctx->nds > 0 && getenv("OLF_GPU_DS_DEBUG")) {
        double slackP = 0.0;
        double retain0 = 0.0;
        double tgt0 = 0.0;
        CUDA_CHECK(cudaMemcpy(&slackP, ctx->dQbuf.as<double>() + ctx->dsSlackPhiRow, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&retain0, ctx->dDsRetain.as<double>(), sizeof(double), cudaMemcpyDeviceToHost));
        int row0 = 0;
        CUDA_CHECK(cudaMemcpy(&row0, ctx->dDsPhiRow.as<int>(), sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&tgt0, ctx->dTarget.as<double>() + row0, sizeof(double), cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "[DS-DEV] scen0 pass: slackP=%.6f slackTargetP=%.6f retain=%.6f rem=%.6f tol=%.1e part0Target=%.6f switched=%d\n",
                slackP, ctx->dsSlackTargetP, retain0, (slackP - ctx->dsSlackTargetP) - retain0, SLACK_DIST_TOL,
                tgt0, (int) rlSwitched[0]);
    }
    if (inDsPhase && ctx->nds > 0 && dsRejectResidual) {
        // slackDistributionFailureBehavior != LEAVE_ON_SLACK_BUS: a scenario whose participants all
        // saturated must NOT be reported converged with the residual absorbed at the slack — route it
        // to the CPU fallback (status -2), which replays it with the run's real failure behavior.
        std::vector<char> dsResidual(nS);
        CUDA_CHECK(cudaMemcpy(dsResidual.data(), ctx->dDsResidual.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            if (dsResidual[s] && live[s]) {
                res.status[s] = -2;
                active[s] = 0;
                live[s] = 0;
                rlSwitched[s] = 0;
            }
        }
    }
    int nChanged = 0;
    for (int s = 0; s < nS; s++) {
        if (rlSwitched[s]) {
            nChanged++;
        }
    }
    if (nChanged > 0) {                                     // this outer loop is still unstable
        for (int s = 0; s < nS; s++) {
            if (rlSwitched[s] && ++prodIters[s] >= RL_MAX_OUTER) {
                // This scenario exhausted ITS outer-iteration budget mid-sequence. Never report it
                // converged on a half-finished RL/DS fixpoint — reject to the CPU fallback, which
                // replays it under OLF's own maxOuterLoopIterations/UNSTABLE handling.
                res.status[s] = -2;
                rlSwitched[s] = 0;
                live[s] = 0;
            }
            active[s] = rlSwitched[s];                      // re-solve only the scenarios it changed, stay in this phase
        }
        seqChanged = true;
        solvePass = true;                                   // mutated targets/modes -> a real re-solve is due
    } else {                                               // this outer loop stabilized
        if (inDsPhase && ctx->nrl > 0) {
            phase = 1;                                      // distributed slack done -> move to reactive limits
        } else {
            if (!seqChanged) {
                break;                                      // a full DS->RL sequence re-solved nothing -> converged
            }
            seqChanged = false;
            phase = (ctx->nds > 0 || ctx->dsSlackOnly) ? 0 : 1;   // restart the sequence at the first loop (distributed slack)
        }
        for (int s = 0; s < nS; s++) {
            active[s] = live[s];                            // re-CHECK every live scenario in the next phase...
        }
        solvePass = false;                                  // ...against the SAME state: STABLE never re-solves (OLF)
    }
    }   // ---- end sequential outer-loop driver ----

    if (getenv("OLF_GPU_DBG_DS") && ctx->nds > 0) {
        std::vector<double> tFinal(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(tFinal.data(), ctx->dTarget.p, tFinal.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<char> ltDump(static_cast<size_t>(std::max(ctx->nrl, 1)) * nS, 0);
        if (ctx->nrl > 0) {
            CUDA_CHECK(cudaMemcpy(ltDump.data(), ctx->dRlLimitType.p, static_cast<size_t>(ctx->nrl) * nS, cudaMemcpyDeviceToHost));
        }
        std::vector<int> dsRows(ctx->nds);
        std::vector<double> dsBase(ctx->nds);
        CUDA_CHECK(cudaMemcpy(dsRows.data(), ctx->dDsPhiRow.p, ctx->nds * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dsBase.data(), ctx->dDsBaseTargetP.p, ctx->nds * sizeof(double), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            double tot = 0;
            fprintf(stderr, "GPU-DS s=%d", s);
            for (int i = 0; i < ctx->nds; i++) {
                double d = tFinal[(size_t) s * n + dsRows[i]] - dsBase[i];
                tot += d;
                fprintf(stderr, " row%d=%+.4f", dsRows[i], d * 100.0);
            }
            fprintf(stderr, " | totMW=%+.4f | lt:", tot * 100.0);
            for (int i = 0; i < ctx->nrl; i++) fprintf(stderr, "%d", ltDump[(size_t) i * nS + s]);
            fprintf(stderr, " status=%d\n", res.status[s]);
        }
    }
    if (rlStateOut != nullptr && ctx->nrl > 0) {             // export per-(controller, scenario) RL state
        std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS), sc(static_cast<size_t>(ctx->nrl) * nS);
        CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sc.data(), ctx->dRlSwitchCount.p, sc.size(), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < lt.size(); k++) {
            rlStateOut[k] = (lt[k] & 3) | (sc[k] << 2);
        }
    }
    CUDA_CHECK(cudaMemcpy(res.x.data(), ctx->dX.p, nTot * sizeof(double), cudaMemcpyDeviceToHost));
    return res;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_createStructure(
        JNIEnv* env, jclass, jint n,
        jdoubleArray jcbIn, jintArray jcbIdx, jdoubleArray jobIn, jintArray jobIdx,
        jdoubleArray jshIn, jintArray jshIdx, jdoubleArray jhvIn, jintArray jhvIdx,
        jintArray jdqElem, jintArray jdqSide, jdoubleArray jdqWeight, jintArray jdqRow, jintArray jdqKind,
        jdoubleArray jziIn, jintArray jziIdx,
        jintArray jdrRow, jintArray jdrCol, jdoubleArray jdrCoef,
        jintArray jdbRow, jintArray jdbCol, jdoubleArray jdbCoef, jintArray jcbPcSide,
        jdoubleArray jvdlIn, jintArray jvdlIdx, jintArray jqrPtr, jintArray jqrRows,
        jintArray jobQAlt, jintArray jziQAlt) {
    jdouble* cbIn = env->GetDoubleArrayElements(jcbIn, nullptr);
    jint* cbIdx = env->GetIntArrayElements(jcbIdx, nullptr);
    jdouble* obIn = env->GetDoubleArrayElements(jobIn, nullptr);
    jint* obIdx = env->GetIntArrayElements(jobIdx, nullptr);
    jdouble* shIn = env->GetDoubleArrayElements(jshIn, nullptr);
    jint* shIdx = env->GetIntArrayElements(jshIdx, nullptr);
    jdouble* hvIn = env->GetDoubleArrayElements(jhvIn, nullptr);
    jint* hvIdx = env->GetIntArrayElements(jhvIdx, nullptr);
    jint* dqElem = env->GetIntArrayElements(jdqElem, nullptr);
    jint* dqSide = env->GetIntArrayElements(jdqSide, nullptr);
    jdouble* dqWeight = env->GetDoubleArrayElements(jdqWeight, nullptr);
    jint* dqRow = env->GetIntArrayElements(jdqRow, nullptr);
    jint* dqKind = env->GetIntArrayElements(jdqKind, nullptr);
    jdouble* ziIn = env->GetDoubleArrayElements(jziIn, nullptr);
    jint* ziIdx = env->GetIntArrayElements(jziIdx, nullptr);
    jint* drRow = env->GetIntArrayElements(jdrRow, nullptr);
    jint* drCol = env->GetIntArrayElements(jdrCol, nullptr);
    jdouble* drCoef = env->GetDoubleArrayElements(jdrCoef, nullptr);
    jint* dbRow = env->GetIntArrayElements(jdbRow, nullptr);
    jint* dbCol = env->GetIntArrayElements(jdbCol, nullptr);
    jdouble* dbCoef = env->GetDoubleArrayElements(jdbCoef, nullptr);
    jint* cbPcSide = env->GetIntArrayElements(jcbPcSide, nullptr);
    jint* qrPtr = env->GetIntArrayElements(jqrPtr, nullptr);
    jint* qrRows = env->GetIntArrayElements(jqrRows, nullptr);
    int nQr = env->GetArrayLength(jqrPtr) - 1;               // groups; qrPtr = [0] when none
    int nQrRows = env->GetArrayLength(jqrRows);
    jdouble* vdlIn = env->GetDoubleArrayElements(jvdlIn, nullptr);
    jint* vdlIdx = env->GetIntArrayElements(jvdlIdx, nullptr);
    jint* obQAlt = env->GetIntArrayElements(jobQAlt, nullptr);
    jint* ziQAlt = env->GetIntArrayElements(jziQAlt, nullptr);

    int ncb = env->GetArrayLength(jcbIdx) / 12;             // cbIdx stride 12 (…a1col r1col qAlt1 qAlt2)
    int nob = env->GetArrayLength(jobIdx) / 6;
    int nsh = env->GetArrayLength(jshIdx) / 5;             // shIdx stride 5 (P Q v bcol qAlt)
    int nhv = env->GetArrayLength(jhvIdx) / 4;
    int ndq = env->GetArrayLength(jdqElem);
    int nzi = env->GetArrayLength(jziIdx) / 11;
    int ndr = env->GetArrayLength(jdrRow);
    int ndb = env->GetArrayLength(jdbRow);
    int nvdl = env->GetArrayLength(jvdlIdx) / 2;             // vdlIdx stride 2 (row, vCol)

    std::string error;
    SharedStructurePtr* st = nullptr;
    try {
        // DEFENSIVE input validation (an intermittent SIGSEGV at buildStructure+0xe43 — three
        // hs_err logs 2026-09-06 — is a one-past-the-end host read that only faults at a page
        // boundary; a named exception routes to the CPU fallback and identifies the bad input).
        auto checkRange = [&](const char* what, const jint* a, size_t len, size_t stride,
                              std::initializer_list<size_t> slots, int lo, int hi) {
            for (size_t e = 0; e < len; e++) {
                for (size_t sl : slots) {
                    int v = a[e * stride + sl];
                    if (v < lo || v >= hi) {
                        throw std::runtime_error(std::string("bad ") + what + " at elem " + std::to_string(e)
                                + " slot " + std::to_string(sl) + ": " + std::to_string(v)
                                + " (allowed [" + std::to_string(lo) + ", " + std::to_string(hi) + "))");
                    }
                }
            }
        };
        // rows/cols may be -1 = absent; anything else must be a valid row/col index < n
        checkRange("cbIdx", cbIdx, ncb, 12, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, -1, n);
        // qAlt slots ([10]/[11] cb, [4] sh) also carry the GROUP encoding -2-groupId (plan-A6)
        checkRange("cbIdx.qAlt", cbIdx, ncb, 12, {10, 11}, -1 - nQr, n);
        checkRange("obIdx", obIdx, nob, 6, {0, 1, 2, 3, 4, 5}, -1, n);
        checkRange("obQAlt", obQAlt, nob, 2, {0, 1}, -1 - nQr, n);
        checkRange("shIdx", shIdx, nsh, 5, {0, 1, 2, 3}, -1, n);
        checkRange("shIdx.qAlt", shIdx, nsh, 5, {4}, -1 - nQr, n);
        checkRange("qrRows", qrRows, nQrRows, 1, {0}, 0, n);
        for (int g = 0; g < nQr; g++) {
            if (qrPtr[g] < 0 || qrPtr[g + 1] <= qrPtr[g] + 1 || qrPtr[g + 1] > nQrRows) {
                throw std::runtime_error("bad qrPtr group " + std::to_string(g));
            }
        }
        checkRange("hvIdx", hvIdx, nhv, 4, {0, 1, 2, 3}, -1, n);
        checkRange("dqRow", dqRow, ndq, 1, {0}, 0, n);
        checkRange("ziIdx", ziIdx, nzi, 11, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, -1, n);
        checkRange("ziQAlt", ziQAlt, nzi, 2, {0, 1}, -1 - nQr, n);
        checkRange("drRow", drRow, ndr, 1, {0}, 0, n);
        checkRange("drCol", drCol, ndr, 1, {0}, 0, n);
        checkRange("dbRow", dbRow, ndb, 1, {0}, 0, n);
        checkRange("dbCol", dbCol, ndb, 1, {0}, 0, n);
        checkRange("vdlIdx", vdlIdx, nvdl, 2, {0, 1}, 0, n);
        for (int e = 0; e < ncb; e++) {                      // encoded (a1Col << 1) | side, or -1
            if (cbPcSide[e] >= 0 && (cbPcSide[e] >> 1) >= n) {
                throw std::runtime_error("bad cbPcSide at elem " + std::to_string(e)
                        + ": a1Col " + std::to_string(cbPcSide[e] >> 1) + " >= n " + std::to_string(n));
            }
        }
        st = new SharedStructurePtr(buildStructure(n, ncb, cbIn, cbIdx, nob, obIn, obIdx,
                                                   nsh, shIn, shIdx, nhv, hvIn, hvIdx,
                                                   ndq, dqElem, dqSide, dqWeight, dqRow, dqKind,
                                                   nzi, ziIn, ziIdx,
                                                   ndr, drRow, drCol, drCoef,
                                                   ndb, dbRow, dbCol, dbCoef, cbPcSide,
                                                   nvdl, vdlIn, vdlIdx, nQr, qrPtr, qrRows,
                                                   obQAlt, ziQAlt));
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    env->ReleaseDoubleArrayElements(jcbIn, cbIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jcbIdx, cbIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jcbPcSide, cbPcSide, JNI_ABORT);
    env->ReleaseIntArrayElements(jqrPtr, qrPtr, JNI_ABORT);
    env->ReleaseIntArrayElements(jqrRows, qrRows, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jobIn, obIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jobIdx, obIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jshIn, shIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jshIdx, shIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jhvIn, hvIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jhvIdx, hvIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqElem, dqElem, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqSide, dqSide, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdqWeight, dqWeight, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqRow, dqRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqKind, dqKind, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jziIn, ziIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jziIdx, ziIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jdrRow, drRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdrCol, drCol, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdrCoef, drCoef, JNI_ABORT);
    env->ReleaseIntArrayElements(jdbRow, dbRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdbCol, dbCol, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdbCoef, dbCoef, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jvdlIn, vdlIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jvdlIdx, vdlIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jobQAlt, obQAlt, JNI_ABORT);
    env->ReleaseIntArrayElements(jziQAlt, ziQAlt, JNI_ABORT);

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU structure creation failed: " + error).c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(st);
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_destroyStructure(
        JNIEnv*, jclass, jlong handle) {
    // Drops the Java-side owner reference only: contexts hold their own shared_ptr copies,
    // so the device-resident structure lives until the last context is destroyed.
    delete reinterpret_cast<SharedStructurePtr*>(handle);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_createContext(
        JNIEnv* env, jclass, jlong structureHandle) {
    auto* st = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    std::string error;
    SolverContext* ctx = nullptr;
    try {
        if (st == nullptr || !*st) {
            throw std::runtime_error("invalid structure handle");
        }
        ctx = createContext(*st);
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }
    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU Newton-Raphson context creation failed: " + error).c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveContext(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jx0, jdoubleArray jtarget,
        jintArray jrowMode, jint maxIter, jdouble tol, jdouble convEpsPerEq, jint lsMode, jint lsMaxIter, jdouble lsStepFold,
        jdouble maxDv, jdouble maxDphi, jintArray jvarType) {
    auto* ctx = reinterpret_cast<SolverContext*>(handle);
    jdouble* x0 = env->GetDoubleArrayElements(jx0, nullptr);
    jdouble* target = env->GetDoubleArrayElements(jtarget, nullptr);
    jint* rowMode = env->GetIntArrayElements(jrowMode, nullptr);
    jint* varType = env->GetIntArrayElements(jvarType, nullptr);
    int n = env->GetArrayLength(jx0);

    std::string error;
    Result res;
    try {
        if (ctx == nullptr || n != ctx->n) {
            throw std::runtime_error("invalid solver context or state size");
        }
        res = solveContext(ctx, x0, target, rowMode, maxIter, tol, convEpsPerEq, lsMode, lsMaxIter, lsStepFold,
                           maxDv, maxDphi, varType);
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    env->ReleaseDoubleArrayElements(jx0, x0, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jtarget, target, JNI_ABORT);
    env->ReleaseIntArrayElements(jrowMode, rowMode, JNI_ABORT);
    env->ReleaseIntArrayElements(jvarType, varType, JNI_ABORT);

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU Newton-Raphson failed: " + error).c_str());
        return nullptr;
    }
    jdoubleArray out = env->NewDoubleArray(n + 2);
    env->SetDoubleArrayRegion(out, 0, n, res.x.data());
    double tail[2] = {static_cast<double>(res.iters), res.norm};
    env->SetDoubleArrayRegion(out, n, 2, tail);
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_destroyContext(
        JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<SolverContext*>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_refreshContextValues(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jcbIn, jdoubleArray jobIn,
        jdoubleArray jshIn, jdoubleArray jhvIn, jdoubleArray jziIn) {
    auto* ctx = reinterpret_cast<SolverContext*>(handle);
    jdouble* cbIn = env->GetDoubleArrayElements(jcbIn, nullptr);
    jdouble* obIn = env->GetDoubleArrayElements(jobIn, nullptr);
    jdouble* shIn = env->GetDoubleArrayElements(jshIn, nullptr);
    jdouble* hvIn = env->GetDoubleArrayElements(jhvIn, nullptr);
    jdouble* ziIn = env->GetDoubleArrayElements(jziIn, nullptr);
    std::string error;
    try {
        if (ctx == nullptr) {
            throw std::runtime_error("invalid solver context");
        }
        refreshStructureValues(*ctx->st, env->GetArrayLength(jcbIn) / 8, cbIn, env->GetArrayLength(jobIn) / 7, obIn,
                               env->GetArrayLength(jshIn) / 2, shIn, env->GetArrayLength(jhvIn) / 5, hvIn,
                               env->GetArrayLength(jziIn), ziIn);
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }
    env->ReleaseDoubleArrayElements(jcbIn, cbIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jobIn, obIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jshIn, shIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jhvIn, hvIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jziIn, ziIn, JNI_ABORT);
    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU values refresh failed: " + error).c_str());
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_createBatchContext(
        JNIEnv* env, jclass, jlong structureHandle, jint nScenarios, jint mode,
        jintArray jrlVrow, jdoubleArray jrlMinQ, jdoubleArray jrlMaxQ, jdoubleArray jrlLoadQ,
        jdoubleArray jrlVtarget, jbyteArray jrlMonitored, jdoubleArray jrowSlope,
        jintArray jdsPhiRow, jdoubleArray jdsFactor,
        jint jdsSlackPhiRow, jdouble jdsSlackTargetP,
        jdoubleArray jdsBaseTargetP, jdoubleArray jdsMaxDelta, jdoubleArray jdsMinDelta,
        jint jdsMode, jdoubleArray jdsMarginUp, jdoubleArray jdsMarginDown,
        jdouble jdsSlackMarginUp, jdouble jdsSlackMarginDown) {
    auto* st = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    jint* rlVrow = env->GetIntArrayElements(jrlVrow, nullptr);
    jdouble* rlMinQ = env->GetDoubleArrayElements(jrlMinQ, nullptr);
    jdouble* rlMaxQ = env->GetDoubleArrayElements(jrlMaxQ, nullptr);
    jdouble* rlLoadQ = env->GetDoubleArrayElements(jrlLoadQ, nullptr);
    jdouble* rlVtarget = env->GetDoubleArrayElements(jrlVtarget, nullptr);
    jbyte* rlMonitored = env->GetByteArrayElements(jrlMonitored, nullptr);
    jdouble* rowSlope = env->GetDoubleArrayElements(jrowSlope, nullptr);
    jint* dsPhiRow = env->GetIntArrayElements(jdsPhiRow, nullptr);
    jdouble* dsFactor = env->GetDoubleArrayElements(jdsFactor, nullptr);
    jdouble* dsBaseTargetP = env->GetDoubleArrayElements(jdsBaseTargetP, nullptr);
    jdouble* dsMaxDelta = env->GetDoubleArrayElements(jdsMaxDelta, nullptr);
    jdouble* dsMinDelta = env->GetDoubleArrayElements(jdsMinDelta, nullptr);
    jdouble* dsMarginUp = env->GetDoubleArrayElements(jdsMarginUp, nullptr);
    jdouble* dsMarginDown = env->GetDoubleArrayElements(jdsMarginDown, nullptr);

    int nrl = env->GetArrayLength(jrlVrow);
    int nds = env->GetArrayLength(jdsPhiRow);

    std::string error;
    BatchSolverContext* ctx = nullptr;
    try {
        if (st == nullptr || !*st) {
            throw std::runtime_error("invalid structure handle");
        }
        ctx = createBatchContext(*st, nScenarios, mode,
                                 nrl, rlVrow, rlMinQ, rlMaxQ, rlLoadQ, rlVtarget, rlMonitored, rowSlope,
                                 nds, dsPhiRow, dsFactor, jdsSlackPhiRow, jdsSlackTargetP,
                                 dsBaseTargetP, dsMaxDelta, dsMinDelta,
                                 jdsMode, dsMarginUp, dsMarginDown, jdsSlackMarginUp, jdsSlackMarginDown);
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    env->ReleaseIntArrayElements(jrlVrow, rlVrow, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlMinQ, rlMinQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlMaxQ, rlMaxQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlLoadQ, rlLoadQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlVtarget, rlVtarget, JNI_ABORT);
    env->ReleaseByteArrayElements(jrlMonitored, rlMonitored, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrowSlope, rowSlope, JNI_ABORT);
    env->ReleaseIntArrayElements(jdsPhiRow, dsPhiRow, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsFactor, dsFactor, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsBaseTargetP, dsBaseTargetP, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMaxDelta, dsMaxDelta, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMinDelta, dsMinDelta, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMarginUp, dsMarginUp, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMarginDown, dsMarginDown, JNI_ABORT);

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU batched context creation failed: " + error).c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveBatchContext(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jx0Base, jdoubleArray jtargetBase,
        jobjectArray jrowMode, jintArray jbaseRowMode, jobjectArray jdisabledMask,
        jintArray jtargetOverrideIndex, jdoubleArray jtargetOverrideValue,
        jintArray jdsOverrideBus, jdoubleArray jdsOverrideFactorDelta, jdoubleArray jdsOverrideBaseDelta,
        jdoubleArray jdsOverrideMaxDelta, jdoubleArray jdsOverrideMinDelta, jdoubleArray jdsAnchorShift,
        jdoubleArray jdsSlackShift, jintArray jrlVtOvSlotIdx, jdoubleArray jrlVtOvVal, jint maxIter, jdouble tol,
        jdouble reactiveLimitTol, jdouble slackDistTol, jdouble convEpsPerEq,
        jint lsMode, jdouble maxDv, jdouble maxDphi, jintArray jvarType, jdoubleArray jx0Full,
        jintArray jShuntOvSlotElem, jdoubleArray jShuntOvGb,
        jintArray jRlStateIn, jintArray jRlStateOut,
        jintArray jR1OvSlotElem, jdoubleArray jR1OvVal,
        jint maxOuterProductive, jint rlMaxSwitch, jint dsRejectResidual, jint skipRlPhase,
        jint nziBase, jintArray jZiOnSlotElem, jintArray jZiOffSlotElem) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    std::vector<int> r1OvSlotElemVec;
    std::vector<double> r1OvValVec;
    if (jR1OvSlotElem != nullptr) {
        jsize rn = env->GetArrayLength(jR1OvSlotElem);
        r1OvSlotElemVec.resize(rn);
        env->GetIntArrayRegion(jR1OvSlotElem, 0, rn, r1OvSlotElemVec.data());
        r1OvValVec.resize(static_cast<size_t>(rn) * 4);      // full 8-double cbIn row per (slot, elem) entry
        env->GetDoubleArrayRegion(jR1OvVal, 0, rn * 4, r1OvValVec.data());
    }
    std::vector<int> ziOffSlotElemVec;
    if (jZiOffSlotElem != nullptr) {
        jsize zn = env->GetArrayLength(jZiOffSlotElem);
        ziOffSlotElemVec.resize(zn);
        env->GetIntArrayRegion(jZiOffSlotElem, 0, zn, ziOffSlotElemVec.data());
    }
    std::vector<int> ziOnSlotElemVec;
    if (jZiOnSlotElem != nullptr) {
        jsize zn = env->GetArrayLength(jZiOnSlotElem);
        ziOnSlotElemVec.resize(zn);
        env->GetIntArrayRegion(jZiOnSlotElem, 0, zn, ziOnSlotElemVec.data());
    }
    std::vector<int> rlStateInVec;
    if (jRlStateIn != nullptr) {
        jsize rn = env->GetArrayLength(jRlStateIn);
        rlStateInVec.resize(rn);
        env->GetIntArrayRegion(jRlStateIn, 0, rn, rlStateInVec.data());
    }
    std::vector<int> rlStateOutVec;
    if (jRlStateOut != nullptr) {
        rlStateOutVec.resize(env->GetArrayLength(jRlStateOut));
    }
    std::vector<int> shOvSlotElemVec;
    std::vector<double> shOvGbVec;
    if (jShuntOvSlotElem != nullptr) {
        jsize sn = env->GetArrayLength(jShuntOvSlotElem);
        shOvSlotElemVec.resize(sn);
        env->GetIntArrayRegion(jShuntOvSlotElem, 0, sn, shOvSlotElemVec.data());
        shOvGbVec.resize(sn);
        env->GetDoubleArrayRegion(jShuntOvGb, 0, sn, shOvGbVec.data());
    }
    std::vector<double> x0FullVec;
    if (jx0Full != nullptr) {
        jsize fn = env->GetArrayLength(jx0Full);
        x0FullVec.resize(fn);
        env->GetDoubleArrayRegion(jx0Full, 0, fn, x0FullVec.data());
    }
    std::vector<int> varTypeVec;
    if (jvarType != nullptr) {
        jsize vn = env->GetArrayLength(jvarType);
        varTypeVec.resize(vn);
        env->GetIntArrayRegion(jvarType, 0, vn, varTypeVec.data());
    }

    std::string error;
    BatchResult res;
    try {
        if (ctx == nullptr) {
            throw std::runtime_error("invalid batch solver context");
        }
        int n = ctx->n;
        int nS = ctx->nScenarios;
        int nElem = ctx->nElements;
        if (env->GetArrayLength(jx0Base) != n) {
            throw std::runtime_error("base state length mismatch with batch context");
        }
        if (env->GetArrayLength(jdisabledMask) != nS) {
            throw std::runtime_error("scenario count mismatch with batch context");
        }

        if (env->GetArrayLength(jrowMode) != nS) {
            throw std::runtime_error("rowMode scenario count mismatch with batch context");
        }
        // x0/target are the BASE row (length n), identical for every scenario — read once and tiled
        // across nS on-device. rowMode is PER-SCENARIO ([nS][n]) — read all nS rows. Only the disabled
        // mask is also per-scenario.
        std::vector<double> x0(n);
        std::vector<double> target(n);
        env->GetDoubleArrayRegion(jx0Base, 0, n, x0.data());
        env->GetDoubleArrayRegion(jtargetBase, 0, n, target.data());
        std::vector<int> rowMode(static_cast<size_t>(n) * nS);
        for (int s = 0; s < nS; s++) {
            auto mRow = (jintArray) env->GetObjectArrayElement(jrowMode, s);
            if (env->GetArrayLength(mRow) != n) {
                env->DeleteLocalRef(mRow);
                throw std::runtime_error("rowMode row length mismatch with batch context");
            }
            env->GetIntArrayRegion(mRow, 0, n, rowMode.data() + (size_t) s * n);
            env->DeleteLocalRef(mRow);
        }
        std::vector<int> baseRowMode(n);                     // contingency-free base row mode (length n)
        env->GetIntArrayRegion(jbaseRowMode, 0, n, baseRowMode.data());
        std::vector<char> disabled(static_cast<size_t>(nElem) * nS, 0);
        for (int s = 0; s < nS; s++) {
            auto dRow = (jbooleanArray) env->GetObjectArrayElement(jdisabledMask, s);
            jboolean* d = env->GetBooleanArrayElements(dRow, nullptr);
            int dLen = env->GetArrayLength(dRow);
            for (int e = 0; e < dLen && e < nElem; e++) {
                disabled[(size_t) s * nElem + e] = d[e] ? 1 : 0;
            }
            env->ReleaseBooleanArrayElements(dRow, d, JNI_ABORT);
            env->DeleteLocalRef(dRow);
        }

        int nOv = env->GetArrayLength(jtargetOverrideIndex);
        std::vector<int> ovIndex(nOv);
        std::vector<double> ovValue(nOv);
        if (nOv > 0) {
            env->GetIntArrayRegion(jtargetOverrideIndex, 0, nOv, ovIndex.data());
            env->GetDoubleArrayRegion(jtargetOverrideValue, 0, nOv, ovValue.data());
        }

        // Per-scenario distributed-slack reweight: present (length nS) only when a generator-loss scenario
        // sits on a DS-participant bus; empty array → no DS override (pass nullptr to the solver).
        int nDsOv = env->GetArrayLength(jdsOverrideBus);
        std::vector<int> dsOvBus(nDsOv);
        std::vector<double> dsOvFactorDelta(nDsOv), dsOvBaseDelta(nDsOv), dsOvMaxDelta(nDsOv), dsOvMinDelta(nDsOv);
        if (nDsOv > 0) {
            env->GetIntArrayRegion(jdsOverrideBus, 0, nDsOv, dsOvBus.data());
            env->GetDoubleArrayRegion(jdsOverrideFactorDelta, 0, nDsOv, dsOvFactorDelta.data());
            env->GetDoubleArrayRegion(jdsOverrideBaseDelta, 0, nDsOv, dsOvBaseDelta.data());
            env->GetDoubleArrayRegion(jdsOverrideMaxDelta, 0, nDsOv, dsOvMaxDelta.data());
            env->GetDoubleArrayRegion(jdsOverrideMinDelta, 0, nDsOv, dsOvMinDelta.data());
        }
        std::vector<double> dsAnchorVec;
        if (jdsAnchorShift != nullptr) {
            dsAnchorVec.resize(env->GetArrayLength(jdsAnchorShift));
            env->GetDoubleArrayRegion(jdsAnchorShift, 0, static_cast<jsize>(dsAnchorVec.size()), dsAnchorVec.data());
        }
        std::vector<double> dsSlackShiftVec;
        if (jdsSlackShift != nullptr) {
            dsSlackShiftVec.resize(env->GetArrayLength(jdsSlackShift));
            env->GetDoubleArrayRegion(jdsSlackShift, 0, static_cast<jsize>(dsSlackShiftVec.size()), dsSlackShiftVec.data());
        }
        std::vector<int> rlVtOvSlotIdxVec;
        std::vector<double> rlVtOvValVec;
        if (jrlVtOvSlotIdx != nullptr && jrlVtOvVal != nullptr) {
            rlVtOvSlotIdxVec.resize(env->GetArrayLength(jrlVtOvSlotIdx));
            env->GetIntArrayRegion(jrlVtOvSlotIdx, 0, static_cast<jsize>(rlVtOvSlotIdxVec.size()), rlVtOvSlotIdxVec.data());
            rlVtOvValVec.resize(env->GetArrayLength(jrlVtOvVal));
            env->GetDoubleArrayRegion(jrlVtOvVal, 0, static_cast<jsize>(rlVtOvValVec.size()), rlVtOvValVec.data());
        }

        res = solveBatchContext(ctx, x0.data(), target.data(), rowMode.data(), baseRowMode.data(), disabled.data(),
                                nOv, ovIndex.data(), ovValue.data(),
                                nDsOv > 0 ? dsOvBus.data() : nullptr, dsOvFactorDelta.data(),
                                dsOvBaseDelta.data(), dsOvMaxDelta.data(), dsOvMinDelta.data(),
                                dsAnchorVec.empty() ? nullptr : dsAnchorVec.data(),
                                dsSlackShiftVec.empty() ? nullptr : dsSlackShiftVec.data(),
                                static_cast<int>(rlVtOvValVec.size()),
                                rlVtOvSlotIdxVec.empty() ? nullptr : rlVtOvSlotIdxVec.data(),
                                rlVtOvValVec.empty() ? nullptr : rlVtOvValVec.data(), maxIter, tol,
                                reactiveLimitTol, slackDistTol, convEpsPerEq,
                                lsMode, maxDv, maxDphi, varTypeVec.empty() ? nullptr : varTypeVec.data(),
                                x0FullVec.empty() ? nullptr : x0FullVec.data(),
                                static_cast<int>(shOvSlotElemVec.size() / 2),
                                shOvSlotElemVec.empty() ? nullptr : shOvSlotElemVec.data(),
                                shOvGbVec.empty() ? nullptr : shOvGbVec.data(),
                                rlStateInVec.empty() ? nullptr : rlStateInVec.data(),
                                rlStateOutVec.empty() ? nullptr : rlStateOutVec.data(),
                                static_cast<int>(r1OvSlotElemVec.size() / 2),
                                r1OvSlotElemVec.empty() ? nullptr : r1OvSlotElemVec.data(),
                                r1OvValVec.empty() ? nullptr : r1OvValVec.data(),
                                maxOuterProductive, rlMaxSwitch, dsRejectResidual, skipRlPhase,
                                nziBase, static_cast<int>(ziOnSlotElemVec.size() / 2),
                                ziOnSlotElemVec.empty() ? nullptr : ziOnSlotElemVec.data(),
                                static_cast<int>(ziOffSlotElemVec.size() / 2),
                                ziOffSlotElemVec.empty() ? nullptr : ziOffSlotElemVec.data());
        if (jRlStateOut != nullptr && !rlStateOutVec.empty()) {
            env->SetIntArrayRegion(jRlStateOut, 0, static_cast<jsize>(rlStateOutVec.size()), rlStateOutVec.data());
        }
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU batched Newton-Raphson failed: " + error).c_str());
        return nullptr;
    }

    // Pack per-scenario results: [state(n), iterations, finalMismatch, convergenceStatus].
    int n = ctx->n;
    int nS = ctx->nScenarios;
    jclass doubleArrayClass = env->FindClass("[D");
    jobjectArray out = env->NewObjectArray(nS, doubleArrayClass, nullptr);
    for (int s = 0; s < nS; s++) {
        jdoubleArray row = env->NewDoubleArray(n + 3);
        env->SetDoubleArrayRegion(row, 0, n, res.x.data() + (size_t) s * n);
        double tail[3] = {static_cast<double>(res.iters[s]), res.norm[s], static_cast<double>(res.status[s])};
        env->SetDoubleArrayRegion(row, n, 3, tail);
        env->SetObjectArrayElement(out, s, row);
        env->DeleteLocalRef(row);
    }
    return out;
}

// Read back the batch context's per-scenario TARGET buffer (n * nScenarios doubles) after a solve —
// the device outer loops (distributed slack) mutate targets in place, and the strategy/pair phase needs
// the PARENT contingency's final distributed targets to re-anchor its own distribution (OLF's
// setGeneratorsInitialTargetPToTargetP semantics).
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    size_t nTot = static_cast<size_t>(ctx->n) * ctx->nScenarios;
    std::vector<double> host(nTot);
    cudaMemcpy(host.data(), ctx->dTarget.p, nTot * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(nTot));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(nTot), host.data());
    return out;
}

// Batched SENSITIVITY solve for the incremental outer-loop family (device port of OLF's
// SensitivityContext): J at the given per-scenario states (the same forward system the Newton
// solves — OLF's "solveTransposed" IS the forward direction in powsybl-math's transposed
// storage), factorized once, solved for a unit RHS at each controller's pinned-variable
// identity row (BRANCH_TARGET_RHO1 & co, storage row = rowMap[eqRow]); the requested variable
// columns of each solution are gathered per scenario. MUST be called on a context that has just
// completed solveBatchContext with the SAME per-scenario overrides/row modes — the fill reads
// them from the live device buffers. UBATCH modes (3/4/5) only.
std::vector<double> solveBatchSensitivities(BatchSolverContext* ctx, const double* states,
                                            int k, const int* unitEqRows, int m, const int* observeVarCols) {
    if (ctx->mode != 3 && ctx->mode != 4 && ctx->mode != 5) {
        throw std::runtime_error("solveBatchSensitivities: UBATCH modes (3/4/5) only");
    }
    int n = ctx->st->n;
    int nS = ctx->nScenarios;
    int nnz = ctx->st->nnz;

    CUDA_CHECK(cudaMemcpyAsync(ctx->dX.p, states, (size_t) n * nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
    std::vector<char> allActive(nS, 1);
    CUDA_CHECK(cudaMemcpyAsync(ctx->dActive.p, allActive.data(), nS * sizeof(char), cudaMemcpyHostToDevice, ctx->stream));

    // ---- J fill at the converged states (same launches as the solve loop's full fill) ----
    CUDA_CHECK(cudaMemsetAsync(ctx->dVals.p, 0, (size_t) nnz * nS * sizeof(double), ctx->stream));
    initFBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dTarget.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->st->cb.ne) {
        fillClosedBranchBatch<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->st->cb.altDest.as<int>(), ctx->st->cb.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->cb.dynReloc ? 1 : 0, ctx->st->cb.phaseCtl ? ctx->st->cb.pcDest.as<int>() : nullptr, ctx->st->cb.phaseCtl ? ctx->st->cb.pcSel.as<int>() : nullptr, relocOf(ctx->st->cb, ctx->st.get()));
    }
    if (ctx->st->ob.ne) {
        fillOpenBranchBatch<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->ob.altDest.as<int>(), ctx->st->ob.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ob.dynReloc ? 1 : 0, relocOf(ctx->st->ob, ctx->st.get()));
    }
    if (ctx->st->sh.ne) {
        fillShuntBatch<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), ctx->st->sh.altDest.as<int>(), ctx->st->sh.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->sh.dynReloc ? 1 : 0, relocOf(ctx->st->sh, ctx->st.get()));
    }
    if (ctx->st->nvdl) {
        fillVdlBatch<<<grid(ctx->st->nvdl * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nvdl, nS, n, nnz, ctx->st->dVdlIn.as<double>(), ctx->st->dVdlIdx.as<int>(), ctx->st->dVdlDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>());
    }
    if (ctx->st->hv.ne) {
        fillHvdcBatch<<<grid(ctx->st->hv.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->hv.ne, nS, n, nnz, ctx->nElements, ctx->hvOffset, ctx->st->hv.in.as<double>(), ctx->st->hv.ix.as<int>(), ctx->st->hv.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>());
    }
    if (ctx->ndq) fillDistrQBatch<<<grid(ctx->ndq * nS), BLOCK, 0, ctx->stream>>>(ctx->ndq, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->obOffset, ctx->st->dDqElem.as<int>(), ctx->st->dDqSide.as<int>(), ctx->st->dDqKind.as<int>(), ctx->st->dDqWeight.as<double>(), ctx->st->dDqDest.as<int>(), ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nDqOv, ctx->nDqOv ? ctx->dDqOvIdx.as<int>() : nullptr, ctx->nDqOv ? ctx->dDqOvW.as<double>() : nullptr, ctx->dMode.as<int>(), ctx->nDqDyn ? ctx->dDqKMember.as<int>() : nullptr, ctx->nDqDyn || ctx->hasDqQrGroups ? ctx->dDqJMember.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberKey.as<double>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberRlEntry.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupStart.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupCount.as<int>() : nullptr, ctx->dRlLimitType.as<char>(), ctx->nrl, ctx->hasDqQrGroups ? ctx->dDqMemberQrGroup.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr);
    if (ctx->ndr) fillDistrRhoBatch<<<grid(ctx->ndr * nS), BLOCK, 0, ctx->stream>>>(ctx->ndr, nS, n, nnz, ctx->st->dDrCol.as<int>(), ctx->st->dDrCoef.as<double>(), ctx->st->dDrDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nDrOv, ctx->nDrOv ? ctx->dDrOvIdx.as<int>() : nullptr, ctx->nDrOv ? ctx->dDrOvC.as<double>() : nullptr, ctx->st->drQr ? ctx->st->dDrQrGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>());
    if (ctx->ndb) fillDistrShuntBBatch<<<grid(ctx->ndb * nS), BLOCK, 0, ctx->stream>>>(ctx->ndb, nS, n, nnz, ctx->st->dDbCol.as<int>(), ctx->st->dDbCoef.as<double>(), ctx->st->dDbDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), ctx->st->dZiAltDest.as<int>(), ctx->st->dZiSelRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ziDyn ? 1 : 0, ziRelocOf(ctx->st.get()));
    applyRowModeBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->dMode.as<int>(), ctx->st->dRowPtr.as<int>(), ctx->st->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dRowSlope.as<double>());

    if (!ctx->analyzed) {
        CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
        ctx->analyzed = true;
    }
    DeviceBuf dObs, dOut;
    dObs.upload(observeVarCols, m * sizeof(int));
    dOut.alloc((size_t) nS * k * m * sizeof(double));
    for (int ki = 0; ki < k; ki++) {
        int row = ctx->st->rowMap[unitEqRows[ki]];           // equation row -> CSR storage row
        if (row < 0) {
            throw std::runtime_error("solveBatchSensitivities: unit row not an active equation");
        }
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        int phase = CUDSS_PHASE_SOLVE | (ki == 0 ? CUDSS_PHASE_FACTORIZATION : 0);
        CUDSS_CHECK(cudssExecute(ctx->ds.handle, phase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
        gatherSensiBatch<<<grid(nS * m), BLOCK, 0, ctx->stream>>>(n, nS, k, m, ki, dObs.as<int>(),
                ctx->dDx.as<double>(), dOut.as<double>());
    }
    ctx->factored = true;                                    // keep mode-5 REFACTORIZATION bookkeeping honest
    std::vector<double> out((size_t) nS * k * m);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), dOut.p, out.size() * sizeof(double), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveBatchSensitivities(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jstates, jintArray junitRows, jintArray jobserveCols) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    std::vector<double> states(env->GetArrayLength(jstates));
    env->GetDoubleArrayRegion(jstates, 0, (jsize) states.size(), states.data());
    std::vector<int> unitRows(env->GetArrayLength(junitRows));
    env->GetIntArrayRegion(junitRows, 0, (jsize) unitRows.size(), unitRows.data());
    std::vector<int> observeCols(env->GetArrayLength(jobserveCols));
    env->GetIntArrayRegion(jobserveCols, 0, (jsize) observeCols.size(), observeCols.data());
    std::string error;
    std::vector<double> out;
    try {
        if (states.size() != (size_t) ctx->st->n * ctx->nScenarios) {
            throw std::runtime_error("states size mismatch");
        }
        out = solveBatchSensitivities(ctx, states.data(),
                (int) unitRows.size(), unitRows.data(), (int) observeCols.size(), observeCols.data());
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU batched sensitivity solve failed: " + error).c_str());
        return nullptr;
    }
    jdoubleArray jout = env->NewDoubleArray((jsize) out.size());
    env->SetDoubleArrayRegion(jout, 0, (jsize) out.size(), out.data());
    return jout;
}

// Per-scenario slack-only-DS absorbed totals after a solve (see slackOnlyDsCheck) — the pair phase
// seeds its scenarios with the parent contingency's value (the device analogue of OLF's
// setGeneratorsInitialTargetPToTargetP before a strategy re-solve). Null when not slack-only.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchSlackAbsorbed(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dsSlackOnly) {
        return nullptr;
    }
    std::vector<double> host(ctx->nScenarios);
    cudaMemcpy(host.data(), ctx->dDsSlackAbsorbed.p, ctx->nScenarios * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(ctx->nScenarios));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(ctx->nScenarios), host.data());
    return out;
}

// LINE_SEARCH backtracking parameters for a batch context (OLF LineSearchStateVectorScaling's
// maxIteration and stepFold): consumed by solves called with lsMode == 1. Context-scoped like
// setBatchRowCriteria, so the solveBatchContext signature stays unchanged.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchLineSearch(
        JNIEnv*, jclass, jlong handle, jint lsMaxIter, jdouble lsStepFold) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->lsMaxIter = lsMaxIter > 0 ? lsMaxIter : 10;
    ctx->lsStepFold = lsStepFold > 1.0 ? lsStepFold : 4.0 / 3.0;
}

// PER_EQUATION_TYPE stopping criterion: install (or clear, with nulls) the per-row threshold
// arrays on a batch context — thrIdent for identity/slope rows, thrPower for power rows, both
// length n. The convergence test then requires EVERY row's |F| under its threshold (OLF's
// PerEquationTypeStoppingCriteria), replacing the 2-norm/infnorm rule for this context.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRowCriteria(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jThrIdent, jdoubleArray jThrPower) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jThrIdent == nullptr || jThrPower == nullptr) {
        ctx->perRowCriteria = false;
        return;
    }
    int n = ctx->st->n;
    jdouble* thrIdent = env->GetDoubleArrayElements(jThrIdent, nullptr);
    jdouble* thrPower = env->GetDoubleArrayElements(jThrPower, nullptr);
    ctx->dThrIdent.upload(thrIdent, static_cast<size_t>(n) * sizeof(double));
    ctx->dThrPower.upload(thrPower, static_cast<size_t>(n) * sizeof(double));
    env->ReleaseDoubleArrayElements(jThrIdent, thrIdent, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jThrPower, thrPower, JNI_ABORT);
    ctx->perRowCriteria = true;
}

// Reactive-capability CURVES at runtime P: install (or clear, with nulls) the per-gen curve channel
// on a batch context — see rlCurveLimitsBatch. cvIdx stride 4 {rlIdx, pRow, ptsOff, npts}; cvIn
// stride 5 {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu}; cvPts stride 3
// {p_MW, minQ_MW, maxQ_MW} sorted by p.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlCurves(
        JNIEnv* env, jclass, jlong handle, jintArray jCvIdx, jdoubleArray jCvIn, jdoubleArray jCvPts,
        jint extrapolate) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jCvIdx == nullptr || jCvIn == nullptr || jCvPts == nullptr) {
        ctx->ncv = 0;
        return;
    }
    int ncv = env->GetArrayLength(jCvIdx) / 4;
    if (ncv == 0) {
        ctx->ncv = 0;
        return;
    }
    jint* cvIdx = env->GetIntArrayElements(jCvIdx, nullptr);
    jdouble* cvIn = env->GetDoubleArrayElements(jCvIn, nullptr);
    jdouble* cvPts = env->GetDoubleArrayElements(jCvPts, nullptr);
    ctx->dCvIdx.upload(cvIdx, static_cast<size_t>(ncv) * 4 * sizeof(int));
    ctx->dCvIn.upload(cvIn, static_cast<size_t>(ncv) * 5 * sizeof(double));
    ctx->dCvPts.upload(cvPts, static_cast<size_t>(env->GetArrayLength(jCvPts)) * sizeof(double));
    ctx->dRlMinQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    ctx->dRlMaxQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    env->ReleaseIntArrayElements(jCvIdx, cvIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCvIn, cvIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCvPts, cvPts, JNI_ABORT);
    ctx->ncv = ncv;
    ctx->cvExtrapolate = extrapolate;
}

// Remote-member RL entries (plan-B1): install (or clear) the per-entry remote kinds + compare rows
// on a batch context. The RL kernel SKIPS remote entries until the B3 transition machinery lands.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlSvcLimits(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jBmin, jdoubleArray jBmax, jdoubleArray jV2Base) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nrl == 0) {
        return;
    }
    if (env->GetArrayLength(jBmin) != ctx->nrl || env->GetArrayLength(jBmax) != ctx->nrl
            || env->GetArrayLength(jV2Base) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchRlSvcLimits: array length != nrl");
        return;
    }
    jdouble* bmin = env->GetDoubleArrayElements(jBmin, nullptr);
    jdouble* bmax = env->GetDoubleArrayElements(jBmax, nullptr);
    jdouble* v2b = env->GetDoubleArrayElements(jV2Base, nullptr);
    ctx->dRlSvcBmin.upload(bmin, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlSvcBmax.upload(bmax, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlV2Base.upload(v2b, static_cast<size_t>(ctx->nrl) * sizeof(double));
    env->ReleaseDoubleArrayElements(jBmin, bmin, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jBmax, bmax, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jV2Base, v2b, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlRemote(
        JNIEnv* env, jclass, jlong handle, jbyteArray jRemote, jintArray jCmpVrow,
        jintArray jRmqRow, jintArray jRmqElem, jintArray jRmqSide, jintArray jRmqKind, jdoubleArray jRmqWeight,
        jintArray jDqKMember, jintArray jDqJMember, jdoubleArray jMKey, jintArray jMRlEntry,
        jintArray jMGroupStart, jintArray jMGroupCount, jintArray jMRow, jbyteArray jMHasRow,
        jintArray jMQrGroup) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jRemote == nullptr || jCmpVrow == nullptr || env->GetArrayLength(jRemote) == 0) {
        ctx->nRlRemote = 0;
        ctx->nRmq = 0;
        ctx->nDqDyn = 0;
        ctx->hasDqQrGroups = 0;
        return;
    }
    int nr = env->GetArrayLength(jRemote);
    if (nr != ctx->nrl || env->GetArrayLength(jCmpVrow) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchRlRemote: array length != nrl");
        return;
    }
    jbyte* rem = env->GetByteArrayElements(jRemote, nullptr);
    jint* cmp = env->GetIntArrayElements(jCmpVrow, nullptr);
    int count = 0;
    for (int i = 0; i < nr; i++) {
        if (rem[i]) {
            count++;
        }
    }
    ctx->dRlRemote.upload(rem, static_cast<size_t>(nr) * sizeof(char));
    ctx->dRlCmpVrow.upload(cmp, static_cast<size_t>(nr) * sizeof(int));
    env->ReleaseByteArrayElements(jRemote, rem, JNI_ABORT);
    env->ReleaseIntArrayElements(jCmpVrow, cmp, JNI_ABORT);
    ctx->nRlRemote = count;
    ctx->nRmq = 0;
    if (jRmqRow != nullptr && env->GetArrayLength(jRmqRow) > 0) {
        int nq = env->GetArrayLength(jRmqRow);
        jint* qr = env->GetIntArrayElements(jRmqRow, nullptr);
        jint* qe = env->GetIntArrayElements(jRmqElem, nullptr);
        jint* qsd = env->GetIntArrayElements(jRmqSide, nullptr);
        jint* qk = env->GetIntArrayElements(jRmqKind, nullptr);
        jdouble* qw = env->GetDoubleArrayElements(jRmqWeight, nullptr);
        ctx->dRmqRow.upload(qr, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqElem.upload(qe, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqSide.upload(qsd, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqKind.upload(qk, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqWeight.upload(qw, static_cast<size_t>(nq) * sizeof(double));
        env->ReleaseIntArrayElements(jRmqRow, qr, JNI_ABORT);
        env->ReleaseIntArrayElements(jRmqElem, qe, JNI_ABORT);
        env->ReleaseIntArrayElements(jRmqSide, qsd, JNI_ABORT);
        env->ReleaseIntArrayElements(jRmqKind, qk, JNI_ABORT);
        env->ReleaseDoubleArrayElements(jRmqWeight, qw, JNI_ABORT);
        ctx->nRmq = nq;
    }
    ctx->nDqDyn = 0;
    ctx->hasDqQrGroups = 0;
    if (jDqKMember != nullptr && env->GetArrayLength(jDqKMember) > 0 && env->GetArrayLength(jMKey) > 0) {
        int ndqc = env->GetArrayLength(jDqKMember);
        int nm = env->GetArrayLength(jMKey);
        jint* km = env->GetIntArrayElements(jDqKMember, nullptr);
        jint* jm = env->GetIntArrayElements(jDqJMember, nullptr);
        jdouble* mk = env->GetDoubleArrayElements(jMKey, nullptr);
        jint* mre = env->GetIntArrayElements(jMRlEntry, nullptr);
        jint* mgs = env->GetIntArrayElements(jMGroupStart, nullptr);
        jint* mgc = env->GetIntArrayElements(jMGroupCount, nullptr);
        jint* mrw = env->GetIntArrayElements(jMRow, nullptr);
        jbyte* mhr = env->GetByteArrayElements(jMHasRow, nullptr);
        ctx->dDqMemberRow.upload(mrw, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberHasRow.upload(mhr, static_cast<size_t>(nm) * sizeof(char));
        env->ReleaseIntArrayElements(jMRow, mrw, JNI_ABORT);
        env->ReleaseByteArrayElements(jMHasRow, mhr, JNI_ABORT);
        ctx->nDqMembers = nm;
        ctx->dDqKMember.upload(km, static_cast<size_t>(ndqc) * sizeof(int));
        ctx->dDqJMember.upload(jm, static_cast<size_t>(ndqc) * sizeof(int));
        ctx->dDqMemberKey.upload(mk, static_cast<size_t>(nm) * sizeof(double));
        ctx->dDqMemberRlEntry.upload(mre, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberGroupStart.upload(mgs, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberGroupCount.upload(mgc, static_cast<size_t>(nm) * sizeof(int));
        env->ReleaseIntArrayElements(jDqKMember, km, JNI_ABORT);
        env->ReleaseIntArrayElements(jDqJMember, jm, JNI_ABORT);
        env->ReleaseDoubleArrayElements(jMKey, mk, JNI_ABORT);
        env->ReleaseIntArrayElements(jMRlEntry, mre, JNI_ABORT);
        env->ReleaseIntArrayElements(jMGroupStart, mgs, JNI_ABORT);
        env->ReleaseIntArrayElements(jMGroupCount, mgc, JNI_ABORT);
        ctx->nDqDyn = ndqc;
        ctx->hasDqQrGroups = 0;
        if (jMQrGroup != nullptr && env->GetArrayLength(jMQrGroup) == nm) {
            jint* mqg = env->GetIntArrayElements(jMQrGroup, nullptr);
            std::vector<char> genVc(ctx->st->nQr > 0 ? ctx->st->nQr : 1, 0);
            for (int i = 0; i < nm; i++) {
                if (mqg[i] >= 0) {
                    ctx->hasDqQrGroups = 1;                  // plan-B4: relocation-linked group present
                    genVc[mqg[i]] = 1;                       // this qr group is a remote gen-VC group
                }
            }
            ctx->dDqMemberQrGroup.upload(mqg, static_cast<size_t>(nm) * sizeof(int));
            if (ctx->hasDqQrGroups) {
                ctx->dQrGenVc.upload(genVc.data(), genVc.size() * sizeof(char));
            }
            env->ReleaseIntArrayElements(jMQrGroup, mqg, JNI_ABORT);
        }
  }
}

// plan-B4 (JAVA hybrid): install ONLY the carrier-row suppression tables — dqJMember (per DISTR_Q
// contribution: row-owner member) + dqMemberQrGroup (per member: pilot's relocation group) — so a
// pushed mode-4 free-member row gates the re-picked carrier's DISTR_Q row off in fillDistrQBatch.
// Does NOT touch nDqDyn: the hybrid's weights stay static-pack + dqOv (OLF renormalizes and the
// engine pushes live values). Nulls clear.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqQrSuppression(
        JNIEnv* env, jclass, jlong handle, jintArray jDqJMember, jintArray jMQrGroup) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jDqJMember == nullptr || jMQrGroup == nullptr
            || env->GetArrayLength(jDqJMember) == 0 || env->GetArrayLength(jMQrGroup) == 0) {
        ctx->hasDqQrGroups = 0;
        return;
    }
    int ndqc = env->GetArrayLength(jDqJMember);
    int nm = env->GetArrayLength(jMQrGroup);
    jint* jm = env->GetIntArrayElements(jDqJMember, nullptr);
    jint* mqg = env->GetIntArrayElements(jMQrGroup, nullptr);
    ctx->dDqJMember.upload(jm, static_cast<size_t>(ndqc) * sizeof(int));
    ctx->dDqMemberQrGroup.upload(mqg, static_cast<size_t>(nm) * sizeof(int));
    env->ReleaseIntArrayElements(jDqJMember, jm, JNI_ABORT);
    env->ReleaseIntArrayElements(jMQrGroup, mqg, JNI_ABORT);
    ctx->hasDqQrGroups = 1;
}

// DISTR_Q weight overrides: install (or clear, with nulls) the per-scenario qPct renormalization
// pairs {s, contrib} + new weights on a batch context (gen outage at a remote-VC controller bus).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqWeightOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jW) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jW == nullptr || env->GetArrayLength(jW) == 0) {
        ctx->nDqOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jW);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* w = env->GetDoubleArrayElements(jW, nullptr);
    ctx->dDqOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDqOvW.upload(w, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jW, w, JNI_ABORT);
    ctx->nDqOv = nOv;
}

// DISTR_RHO coefficient overrides: install (or clear, with nulls) the per-scenario 1/count'
// renormalization pairs {s, contrib} + new coefficients on a batch context (a shared-TVC group
// member outaged in that scenario — plan-group-member-dynamics A5).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDrCoefOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jC) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jC == nullptr || env->GetArrayLength(jC) == 0) {
        ctx->nDrOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jC);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* c = env->GetDoubleArrayElements(jC, nullptr);
    ctx->dDrOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDrOvC.upload(c, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jC, c, JNI_ABORT);
    ctx->nDrOv = nOv;
}

// Per-scenario ACTIVE equation counts for the L2 stopping threshold (see hNActive). Per-chunk state.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchActiveRowCounts(
        JNIEnv* env, jclass, jlong handle, jintArray jNActive) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hNActive.clear();
    if (jNActive != nullptr) {
        int len = env->GetArrayLength(jNActive);
        ctx->hNActive.resize(len);
        env->GetIntArrayRegion(jNActive, 0, len, ctx->hNActive.data());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_destroyBatchContext(
        JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<BatchSolverContext*>(handle);
}

// VALUES refresh addressed by the SHARED STRUCTURE handle (no solver context needed): the
// sequential batch-of-1 path reconciles tap/shunt moves the same way the seq context did —
// one extraction has ONE value set, and every context sharing the structure sees the update.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_refreshStructureValues(
        JNIEnv* env, jclass, jlong structureHandle, jdoubleArray jcbIn, jdoubleArray jobIn,
        jdoubleArray jshIn, jdoubleArray jhvIn, jdoubleArray jziIn) {
    auto* stp = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    jdouble* cbIn = env->GetDoubleArrayElements(jcbIn, nullptr);
    jdouble* obIn = env->GetDoubleArrayElements(jobIn, nullptr);
    jdouble* shIn = env->GetDoubleArrayElements(jshIn, nullptr);
    jdouble* hvIn = env->GetDoubleArrayElements(jhvIn, nullptr);
    jdouble* ziIn = env->GetDoubleArrayElements(jziIn, nullptr);
    std::string error;
    try {
        if (stp == nullptr || !*stp) {
            throw std::runtime_error("invalid structure handle");
        }
        refreshStructureValues(**stp, env->GetArrayLength(jcbIn) / 8, cbIn, env->GetArrayLength(jobIn) / 7, obIn,
                               env->GetArrayLength(jshIn) / 2, shIn, env->GetArrayLength(jhvIn) / 5, hvIn,
                               env->GetArrayLength(jziIn), ziIn);
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }
    env->ReleaseDoubleArrayElements(jcbIn, cbIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jobIn, obIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jshIn, shIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jhvIn, hvIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jziIn, ziIn, JNI_ABORT);
    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU structure values refresh failed: " + error).c_str());
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_recommendBatchSize(
        JNIEnv* env, jclass, jlong structureHandle, jint maxCap) {
    auto* stp = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    int rec = 1;
    try {
        if (stp == nullptr || !*stp) {
            throw std::runtime_error("invalid structure handle");
        }
        SharedStructure& st = **stp;
        // Per-scenario device footprint: the dominant term is cuDSS's LU numeric factors
        // (the exact fill from a probe analysis — cached on the structure — or the heuristic),
        // plus the value/state arrays (vals + x/target/F/B/dx). The LU index structure is
        // shared across the batch.
        constexpr double SAFETY = 0.7;                       // leave headroom for cuDSS workspace + fragmentation
        double luBytes = luBytesPerScenario(st);
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));  // after the probe freed its buffers
        // 7n doubles per scenario, not 5: x/target/F/B/dx PLUS dQbuf (the per-scenario reactive
        // accumulator) and dXprevLs (the line-search previous state, allocated under the RTE
        // parameters' MAX_VOLTAGE_CHANGE scaling).
        double perScenarioBytes = luBytes
                + (static_cast<double>(st.nnz) + 7.0 * st.n) * sizeof(double)
                + static_cast<double>(st.n) * sizeof(int)    // dMode
                + static_cast<double>(st.nElements);         // dDisabled (char)
        double budget = static_cast<double>(freeBytes) * SAFETY;
        long fit = perScenarioBytes > 0 ? static_cast<long>(budget / perScenarioBytes) : maxCap;
        rec = static_cast<int>(std::max<long>(1, std::min<long>(maxCap, fit)));
    } catch (...) {
        rec = 1;                                             // a query failure yields the safest chunk
    }
    return rec;
}

// ---------------------------------------------------------------------------------
// ADJOINT step 1a: solve Aᵀ x = b for an EXPLICIT CSR matrix, by building the transpose
// and factorizing it.
//
// Why this exists: cuDSS (through 0.8.0 — verified, its header has no transpose/OP_T
// symbol at all) offers only the non-transposed solve against its factors, so one GPU
// factorization yields ONE direction where KLU's klu_solve/klu_tsolve pair yields two.
// Our Jacobian is stored as TRUE CSR (row = equation, col = variable) so the plain solve
// serves the forward Newton; the adjoint (Jᵀλ = ...) is therefore the direction we cannot
// get from the forward's factors, and until cuDSS ships a transpose solve we buy it with a
// second factorization of an explicitly transposed matrix.
//
// This entry point is deliberately SELF-CONTAINED (its own handle/config/data, no reuse of
// the Newton context): it isolates exactly one question — is our CSR transpose + cuDSS
// factorize/solve of Aᵀ correct? — so a failure here cannot be confused with the adjoint's
// RHS assembly or contraction. It is a validation//measurement entry, not the production
// path; the production seam should reuse the forward's structure and swap to a native
// transpose solve the moment cuDSS provides one.
//
// CSR transpose is the standard counting sort: count entries per column of A (= rows of Aᵀ),
// prefix-sum into row pointers, then scatter. O(nnz + n), host-side, and for a FIXED pattern
// the resulting scatter is a permutation that can be computed once and reused per solve.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveTransposedExplicit(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx,
        jdoubleArray jVals, jdoubleArray jB) {
    try {
        const int nn = n;
        std::vector<int> rowPtr(nn + 1);
        env->GetIntArrayRegion(jRowPtr, 0, nn + 1, rowPtr.data());
        const int nnz = rowPtr[nn];
        std::vector<int> colIdx(nnz);
        std::vector<double> vals(nnz);
        std::vector<double> b(nn);
        env->GetIntArrayRegion(jColIdx, 0, nnz, colIdx.data());
        env->GetDoubleArrayRegion(jVals, 0, nnz, vals.data());
        env->GetDoubleArrayRegion(jB, 0, nn, b.data());

        // --- CSR(A) -> CSR(Aᵀ) by counting sort ---
        std::vector<int> tRowPtr(nn + 1, 0);
        std::vector<int> tColIdx(nnz);
        std::vector<double> tVals(nnz);
        for (int k = 0; k < nnz; k++) {
            tRowPtr[colIdx[k] + 1]++;                        // column of A becomes row of Aᵀ
        }
        for (int r = 0; r < nn; r++) {
            tRowPtr[r + 1] += tRowPtr[r];
        }
        std::vector<int> cursor(tRowPtr.begin(), tRowPtr.end() - 1);
        for (int r = 0; r < nn; r++) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
                const int dst = cursor[colIdx[k]]++;
                tColIdx[dst] = r;                            // row of A becomes column of Aᵀ
                tVals[dst] = vals[k];
            }
        }

        DeviceBuf dRowPtr;
        DeviceBuf dColIdx;
        DeviceBuf dVals;
        DeviceBuf dB;
        DeviceBuf dX;
        dRowPtr.upload(tRowPtr.data(), (nn + 1) * sizeof(int));
        dColIdx.upload(tColIdx.data(), nnz * sizeof(int));
        dVals.upload(tVals.data(), nnz * sizeof(double));
        dB.upload(b.data(), nn * sizeof(double));
        dX.alloc(nn * sizeof(double));

        CudssCtx ds;
        CUDSS_CHECK(cudssCreate(&ds.handle));
        CUDSS_CHECK(cudssConfigCreate(&ds.config));
        applyReorderAlg(ds.config);
        applyAdjointMatching(ds.config);
        CUDSS_CHECK(cudssDataCreate(ds.handle, &ds.data));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ds.matA, nn, nn, nnz, dRowPtr.as<int>(), nullptr,
                                         dColIdx.as<int>(), dVals.as<double>(), OLF_CUDSS_CSR_TYPES,
                                         CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matB, nn, 1, nn, dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matX, nn, 1, nn, dX.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssExecute(ds.handle, CUDSS_PHASE_ANALYSIS, ds.config, ds.data, ds.matA, ds.matX, ds.matB));
        CUDSS_CHECK(cudssExecute(ds.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                                 ds.config, ds.data, ds.matA, ds.matX, ds.matB));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> x(nn);
        CUDA_CHECK(cudaMemcpy(x.data(), dX.p, nn * sizeof(double), cudaMemcpyDeviceToHost));
        jdoubleArray out = env->NewDoubleArray(nn);
        env->SetDoubleArrayRegion(out, 0, nn, x.data());
        return out;
    } catch (const std::exception& e) {
        jclass ex = env->FindClass("java/lang/RuntimeException");
        if (ex) env->ThrowNew(ex, e.what());
        return nullptr;
    }
}
