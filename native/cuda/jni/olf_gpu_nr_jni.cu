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
#include <cublas_v2.h>

#include <algorithm>
#include <map>
#include <cstring>
#include <chrono>
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

// OLF_GPU_PHASE_TIME accounting: how much of a solveBatchContext call is device outer-loop passes
// vs Newton iterations. A pass that re-solves is the expensive one; a pass that only re-checks is not.
long long gOuterPasses = 0;
// The outer loops the device driver can run, in OLF's configured order.
enum DeviceLoopKind {
    LOOP_DS = 0,             // DistributedSlack                         (OLF order 0)
    LOOP_HVDC_FREEZE = 1,    // FreezingHvdcACEmulation                  (OLF order 1)
    LOOP_HVDC_LIMITS = 2,    // AcHvdcAcEmulationLimits                  (OLF order 2)
    LOOP_MONITORING = 3,     // MonitoringVoltage — the standby-SVC automaton (OLF order 5)
    LOOP_RL = 4,             // ReactiveLimits                           (OLF order 6)
    LOOP_HOST_SVC = 5,       // SecondaryVoltageControl — host pause     (OLF order 4, pre-RL)
    LOOP_HOST_PHASE = 6,     // PhaseControl — host pause                (OLF order 7, post-RL)
    LOOP_HOST_TVC = 7,       // TransformerVoltageControl — host pause   (OLF order 8, post-RL)
    LOOP_HOST_TRPC = 8,      // TransformerReactivePowerControl — host   (OLF order 9, post-RL)
    LOOP_HOST_SHUNT = 9,     // ShuntVoltageControl — host pause         (OLF order 10, post-RL)
    LOOP_TVC_KERNEL = 10,    // IncrementalTransformerVoltageControl — IN-KERNEL (OLF order 8)
    LOOP_SHUNT_KERNEL = 11,  // IncrementalShuntVoltageControl — IN-KERNEL (OLF order 10)
    LOOP_PHASE_KERNEL = 12,  // IncrementalPhaseControl (CONTROLLER + LIMITER) — IN-KERNEL (OLF order 7)
    LOOP_TRPC_KERNEL = 13,   // IncrementalTransformerReactivePowerControl — IN-KERNEL (OLF order 9)
    LOOP_TVD_KERNEL = 14,    // TransformerVoltageControl AFTER_GENERATOR / Simple (WITH_GENERATOR) — IN-KERNEL (OLF order 8)
    LOOP_SVD_KERNEL = 15,    // ShuntVoltageControl WITH_GENERATOR — IN-KERNEL (OLF order 10)
    LOOP_SVC_KERNEL = 16     // SecondaryVoltageControl — IN-KERNEL (OLF order 4)
};
// GpuBatchHvdcAcEmulation's status codes, shared with the host.
constexpr int HV_LINEAR = 0;
constexpr int HV_SAT_1_TO_2 = 1;
constexpr int HV_SAT_2_TO_1 = 2;
constexpr int HV_FROZEN = 3;
constexpr int MAX_DEVICE_LOOPS = 16;

long long gNewtonIters = 0;
int gLastPassIters = 0;   // debug: inner-Newton steps taken by the pass just finished
long long gSolvePasses = 0;
long long gActiveIterSum = 0;
long long gWidthIterSum = 0;
long long gCompactIters = 0;

using namespace powsybl::openloadflow;

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_)); } } while (0)
#define CUBLAS_CHECK(call)                                                                       \
    do {                                                                                         \
        cublasStatus_t st_ = (call);                                                             \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                      \
            throw std::runtime_error("cuBLAS error " + std::to_string(static_cast<int>(st_)));    \
        }                                                                                        \
    } while (0)

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

// IR IS REQUIRED WHEN THE SYSTEM HAS ZERO-IMPEDANCE BRANCHES. Each one puts ZERO_V / ZERO_PHI on
// the DUMMY_P / DUMMY_Q rows carrying v1-v2 and phi1-phi2, never their own dummy — two zero
// diagonals apiece — and AMD, a SYMMETRIC ordering that assumes a non-zero diagonal, then carries
// real FORWARD error there. LIN_RESID stays ~1e-14 throughout, because that is the BACKWARD
// residual; with kappa ~ 5e8 the forward error in dx is what moves, and two refinement steps
// recover it. Measured on rte6515's busbar outages: base 1.225e-07 -> 3.831e-11 and BUS-1017
// 2.302e-06 -> 2.454e-11, which is also exactly where nested dissection lands, two unrelated
// mechanisms agreeing on the accurate answer.
//
// NESTED DISSECTION WAS TRIED FIRST AND IS NOT USABLE: it reaches the same accuracy but COLLAPSES
// the batch at scale — 80 busbar outages give 0/80 scenarios batched against AMD's 60/80, most
// blowing up to ~1e14. That regression was invisible on the 20-contingency window it was validated
// on. IR keeps AMD's ordering, so batching share is unchanged (60/80 either way).
// RE-MEASURED 2026-09-21 AFTER THE ZERO-DIAGONAL ROW REPAIR (hazRepairVals): still needed. The repair
// makes every row nonsingular, but a borrowed +-1 diagonal sits among O(100) admittance entries and
// static pivoting is still inaccurate there. rte6515, 500 busbar outages, same code either way:
//   IR=2: worst ||J.dx-F||inf 1.5e-11 over every scenario and iteration; buses vs CPU worst 2.6e-9 kV,
//         3 contingencies above 1e-9 kV.
//   IR=0: worst ||J.dx-F||inf 3.8e-4 (ratio 1.5e-3 to ||F||), on the REPAIRED rows (VL-132_1,
//         VL-1113_1/2/3, a coupler DUMMY_Q); buses worst 2.6e-8 kV, 12 above 1e-9 kV.
// The open lever, if the two refinement solves per iteration ever matter for wall time, is to SCALE
// the borrowed row so the repaired diagonal is commensurate with the row's other entries rather than
// to drop the refinement.
inline int irSteps(bool hasZeroImpedance) {
    const char* e = std::getenv("OLF_GPU_IR_STEPS");
    if (e != nullptr && *e != '\0') {
        return std::atoi(e);
    }
    return hasZeroImpedance ? 2 : IR_N_STEPS;
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
// MATCHING IS NOT AVAILABLE TO THE BATCHED NR: cuDSS 0.8.0 rejects it at the uniform-batch analysis
// (status 4; the library string reads "Matching is not supported for uniform batches"), so the batched
// path repairs its zero diagonals itself — see hazRepairVals. What follows is the standalone case.
//
// MATCHING IS NOT OPTIONAL ON THESE JACOBIANS. A bus fed ONLY through a zero-impedance branch
// leaves four rows with a numerically ZERO diagonal — the branch's ZERO_V / ZERO_PHI (which carry
// v1-v2 and phi1-phi2, never their own dummy) and the dead-end bus's P / Q balances (which carry
// only -DUMMY_P / -DUMMY_Q). The matrix is non-singular, but factoring it needs the rows matched to
// other columns. Without matching cuDSS perturbs the zero pivot and the reciprocal propagates:
// measured on deviceRemoteZeroImpMemberPin as DUMMY_P = -1.220703125e+09 and, through the chained
// second pivot, phi = -1.220703125e+22 (1.220703125e-04 is 2^-13, so these are 1e13 and (1e13)^2
// scaled) where the exact answers are -0.05 and 0. The solve then "converges" somewhere else or
// diverges, and on the base load flow it was masked by the device-broken-solve safety net.
inline void applyMatchingAlg(cudssConfig_t config) {
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

inline void applyReorderAlg(cudssConfig_t config, bool hasZeroImpedance);   // defined below

// MODE 6 = the ROBUST single-matrix solve: mode 0's per-scenario loop with matching and pivoting
// on cuDSS's single-matrix API (the uniform batch supports none of the three). It is what a
// factorization needs on rte6515's full busbar structure (366 retained couplers, 1e-18 pivots under
// every unpivoted ordering): measured 2026-09-22, the batch-of-one base load flow that the uniform
// batch could not solve converges on the device in this mode. The batch-of-one falls into it
// automatically after a linear-solve reject (GpuNewtonRaphson); a context asks for it through
// olf.gpu.batchMode=6. Mode 0 itself is untouched (the JAVA hybrid builds mode-0 contexts and its
// 1e-12 fixtures sit on the last ulp of AMD's roundoff). OLF_GPU_REORDER_ALG still overrides the ordering.
inline void applyRobustSingleMatrixConfig(cudssConfig_t config) {
#if CUDSS_VERSION >= 800
    const char* e = std::getenv("OLF_GPU_REORDER_ALG");
    bool colamdFamily = true;                                // BTF_COLAMD / COLAMD: global column pivoting is defined
    if (e == nullptr || *e == '\0') {
        // cuDSS 0.8 REFUSES matching under BTF_COLAMD / COLAMD (status 4 at factorization, every
        // size); it accepts it under default / ND / AMD with automatic pivoting. AMD by default.
        OlfReorderAlg amd = CUDSS_REORDERING_ALG_AMD;
        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &amd, sizeof(amd)));
        colamdFamily = false;
    } else {
        applyReorderAlg(config, true);
        colamdFamily = std::string(e) == "btf_colamd" || std::string(e) == "colamd";
    }
    cudssPivotType_t pt = colamdFamily ? CUDSS_PIVOT_GLOBAL_COL : CUDSS_PIVOT_AUTO;
    CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_PIVOT_TYPE, &pt, sizeof(pt)));
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        std::fprintf(stderr, "ROBUST_CFG reorder=%s pivot=%s matching=auto\n", e == nullptr ? "amd" : e,
                     colamdFamily ? "global_col" : "auto");
    }
    // Matching is REQUIRED (measured: pivoting alone left the full-scope base at NaN). Its
    // permutation is value-dependent and cuDSS accepts ONE analysis per data object (a second
    // ANALYSIS returns status 4), so the analysis runs once on the first values of the run and the
    // later factorizations reuse it - which is what converged the full-scope base. A system cuDSS
    // refuses under this configuration (ieee14 in the cross-check test: status 4 at factorization)
    // throws, and the batch-of-one falls back to the CPU as before - never wrong.
    cudssMatchingAlg_t alg = CUDSS_MATCHING_ALG_AUTO;
    CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_MATCHING_ALG, &alg, sizeof(alg)));
#else
    (void) config;
#endif
}

// AMD IS NOT SAFE WHEN THE SYSTEM HAS ZERO-IMPEDANCE BRANCHES. AMD is a SYMMETRIC (approximate
// minimum degree) ordering and assumes a non-zero diagonal; a zero-impedance branch always breaks
// that, because ZERO_V / ZERO_PHI sit on the DUMMY_P / DUMMY_Q rows and carry v1-v2 and phi1-phi2,
// never their own dummy. The DEAD-END STUB case, where four such rows chain, is repaired
// structurally by the row relocation (GpuAcRowIndexing#ziDeadEndSide) — but the ordinary two per
// branch remain, and AMD loses accuracy on them: on rte6515's busbar outages it left BUS-1017 at
// 2.302e-06 kV and BUS-1003 at 1.142e-07 where nested dissection gives 4.074e-09 and 9.572e-11.
// That those are ALSO the numbers cuDSS iterative refinement converges to (OLF_GPU_IR_STEPS=2 and
// 4 both reproduce them, and the base likewise) is what identifies them as the accurate solve
// rather than merely a different one: two unrelated mechanisms agreeing on an answer AMD misses.
// AMD is kept where it is safe and was measured to pay — the 2026-07-16 sweep won on
// case9241pegase (LU_NNZ 260076 vs 304772, warm SA -11%), a network with no retained switches and
// therefore no zero-impedance branch, so it never exercised a zero diagonal.
inline void applyReorderAlg(cudssConfig_t config, bool hasZeroImpedance) {
    const char* e = std::getenv("OLF_GPU_REORDER_ALG");
    if (e == nullptr || *e == '\0') {
#if CUDSS_VERSION >= 800
            OlfReorderAlg amd = CUDSS_REORDERING_ALG_AMD;        // sweep winner; ND collapses the
        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &amd, sizeof(amd)));
#endif
        return;                                              // 0.7.1: library default, untouched
    }
    (void) hasZeroImpedance;
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
inline void traceLuNnz(cudssHandle_t handle, cudssData_t data, const char* where, int nScenarios = 0,
                       double ms = -1.0) {
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
        std::fprintf(stderr, "OLF_GPU_CUDSS reorder=%s %s nScenarios=%d LU_NNZ=%lld analysis=%.1f ms\n",
                     alg != nullptr ? alg : unsetName, where, nScenarios, static_cast<long long>(luNnz), ms);
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
    if (q[10] & 1) {                                         // spanning-tree edge: voltage/angle coupling
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
// OLF_GPU_LIN_RESID=1: how accurately did the LINEAR solve actually solve? Per row,
// r_i = (J.dx)_i -/+ F_i. This is the measurement that separates "the two arms assemble DIFFERENT
// Jacobians" from "they assemble the same one and the SOLVER lands elsewhere": if ||r|| is at
// round-off relative to ||F||, the solve is exact and any step difference must come from J itself;
// if ||r|| is large, the solve is inaccurate and iterative refinement is the remedy. It is also
// refinement's own first step, so the cost buys both answers.
__global__ void linResidRow(int n, const int* rowPtr, const int* colIdx, const double* vals,
                            const double* dx, const double* F, double* rPlus, double* rMinus) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        acc += vals[k] * dx[colIdx[k]];
    }
    rPlus[i] = acc - F[i];                                   // sign convention settled by taking the
    rMinus[i] = acc + F[i];                                  // smaller of the two on the host
}

// Batched twin of linResidRow: every scenario has its OWN values block (stride nnz) and its own
// dx / F block (stride n), over the SHARED sparsity pattern.
// ---------------------------------------------------------------------------------------------
// ZERO-DIAGONAL ROW REPAIR. The batched solve runs cuDSS without pivoting or matching (neither is
// supported for uniform batches), on the promise that the diagonal-pairing row permutation leaves
// every row a nonzero diagonal. A zero-impedance CLUSTER breaks that promise twice over: the couplers'
// ZERO_V / ZERO_PHI rows never carry their own dummy, and a section whose only branches are couplers
// has P and Q rows made of dummies alone — as a plain balance (the dead-end relocation covers that
// one shape) or, with a generator on every section, as the merged group's DISTR_Q rows, which the
// relocation must not touch. cuDSS then perturbs the zero pivot and returns a wrong STEP with a
// converging trajectory: measured on rte6515's 500-busbar structure at VL-1113, ||J.dx-F||inf = 82
// for ||F||inf = 1.24 on the first step of EVERY scenario, and the answer differs from OLF's wherever
// OLF stops one iteration short (BUS-1196_BBS, 5.157e-02 kV).
//
// The repair is numeric and per scenario, applied to the assembled values right before they are
// factorized: a row whose diagonal is exactly zero borrows a row that carries its diagonal column,
//     row_r += s_r * row_d          (values AND right-hand side)
// which leaves J dx = -F with the same solution (the combination matrix is nonsingular: P rows
// borrow with +1, Q rows and dummy rows with -1; on a spanning tree the borrowing closes only in
// 4-cycles P -> ZERO_PHI -> Q -> ZERO_V -> P, whose determinant 1 - (+1)(-1)(-1)(-1) is 2). Candidates are structural — a dummy row's two endpoint rows (and the pilot-Q relocation
// candidates when an endpoint's Q rides there), an endpoint row's incident couplers' dummy rows —
// and the choice is numeric, so a coupler switched off or a section pinned to identity this scenario
// simply is not a donor (an identity row has nothing but its own 1). Every right-hand side handed to
// those factors goes through the same combination (the unit vectors of the sensitivity solves too).
// The pattern carries the union of each row with its candidates so the addition lands in place.
// ---------------------------------------------------------------------------------------------
__device__ inline int csrFind(const int* rowPtr, const int* colIdx, int row, int col) {
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
    return (lo < rowPtr[row + 1] && colIdx[lo] == col) ? lo : -1;
}

__global__ void hazSnapshotVals(int nHaz, int nS, int nnz, int hazTot, const int* hazRow, const int* hazOff,
                                const int* rowPtr, const double* vals, double* snap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nHaz * nS) return;
    int i = tid % nHaz;
    int s = tid / nHaz;
    int r = hazRow[i];
    const double* v = vals + (size_t) s * nnz;
    double* o = snap + (size_t) s * hazTot + hazOff[i];
    for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
        o[k - rowPtr[r]] = v[k];
    }
}

__global__ void hazRepairVals(int nHaz, int nS, int nnz, int hazTot, const int* hazRow, const int* hazOff,
                              const int* hazDonorPtr, const int* hazDonor, const int* hazDonorHaz,
                              const char* hazSign, const int* rowPtr, const int* colIdx, const int* diagPos,
                              const double* snap, double* vals, int* choice, int traceNoDonor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nHaz * nS) return;
    int i = tid % nHaz;
    int s = tid / nHaz;
    int r = hazRow[i];
    double* v = vals + (size_t) s * nnz;
    choice[(size_t) s * nHaz + i] = -1;
    if (v[diagPos[r]] != 0.0) return;                        // a real (or identity) diagonal: nothing to do
    const double* sn = snap + (size_t) s * hazTot;
    double sgn = hazSign[i];
    for (int j = hazDonorPtr[i]; j < hazDonorPtr[i + 1]; j++) {
        int d = hazDonor[j];
        int hd = hazDonorHaz[j];                             // donor's own hazard slot (its ORIGINAL row is in snap)
        int p = csrFind(rowPtr, colIdx, d, r);
        if (p < 0) continue;
        double coef = hd >= 0 ? sn[hazOff[hd] + (p - rowPtr[d])] : v[p];
        if (coef == 0.0) continue;                           // this donor does not carry my column this scenario
        for (int k = rowPtr[d]; k < rowPtr[d + 1]; k++) {
            double dv = hd >= 0 ? sn[hazOff[hd] + (k - rowPtr[d])] : v[k];
            if (dv == 0.0) continue;
            int q = csrFind(rowPtr, colIdx, r, colIdx[k]);
            if (q >= 0) v[q] += sgn * dv;
        }
        choice[(size_t) s * nHaz + i] = j;
        return;
    }
    if (traceNoDonor) {                                      // a zero diagonal nobody could repair: the
        printf("HAZ_NODONOR s=%d row=%d donors=%d\n", s, r,   // pivot cuDSS will perturb (device printf)
               hazDonorPtr[i + 1] - hazDonorPtr[i]);
    }
}

__global__ void hazSnapshotB(int nHaz, int nS, int n, const int* hazRow, const double* B, double* bsnap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nHaz * nS) return;
    int i = tid % nHaz;
    int s = tid / nHaz;
    bsnap[tid] = B[(size_t) s * n + hazRow[i]];
}

__global__ void hazRepairRhs(int nHaz, int nS, int n, const int* hazRow, const int* hazDonor, const int* hazDonorHaz,
                             const char* hazSign, const int* choice, const double* bsnap, double* B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nHaz * nS) return;
    int i = tid % nHaz;
    int s = tid / nHaz;
    int j = choice[tid];
    if (j < 0) return;
    int d = hazDonor[j];
    int hd = hazDonorHaz[j];
    double bd = hd >= 0 ? bsnap[(size_t) s * nHaz + hd] : B[(size_t) s * n + d];
    B[(size_t) s * n + hazRow[i]] += hazSign[i] * bd;
}

template <typename Ctx>
static void hazRepairValues(Ctx* ctx, int nS, double* vals) {
    auto* st = ctx->st.get();
    if (st->nHaz == 0) return;
    if (ctx->hazCap < nS) {
        ctx->dHazSnap.alloc((size_t) nS * st->hazTot * sizeof(double));
        ctx->dHazBSnap.alloc((size_t) nS * st->nHaz * sizeof(double));
        ctx->dHazChoice.alloc((size_t) nS * st->nHaz * sizeof(int));
        ctx->hazCap = nS;
    }
    int nnz = st->nnz;
    hazSnapshotVals<<<((st->nHaz * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(st->nHaz, nS, nnz, st->hazTot,
            st->dHazRow.template as<int>(), st->dHazOff.template as<int>(), st->dRowPtr.template as<int>(), vals, ctx->dHazSnap.template as<double>());
    hazRepairVals<<<((st->nHaz * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(st->nHaz, nS, nnz, st->hazTot,
            st->dHazRow.template as<int>(), st->dHazOff.template as<int>(), st->dHazDonorPtr.template as<int>(), st->dHazDonor.template as<int>(),
            st->dHazDonorHaz.template as<int>(), st->dHazSign.template as<char>(), st->dRowPtr.template as<int>(), st->dColIdx.template as<int>(),
            st->dDiagPos.template as<int>(), ctx->dHazSnap.template as<double>(), vals, ctx->dHazChoice.template as<int>(),
            std::getenv("OLF_GPU_HAZ_TRACE") != nullptr ? 1 : 0);
}

/** The same combination on a right-hand side (nS scenarios from {@code B}, choices from the last
 *  hazRepairValues on this context; {@code sOff} selects the scenario block for a per-scenario call). */
template <typename Ctx>
static void hazRepairRightHandSide(Ctx* ctx, int nS, double* B, int sOff = 0) {
    auto* st = ctx->st.get();
    if (st->nHaz == 0 || ctx->hazCap < sOff + nS) return;
    int n = st->n;
    double* bsnap = ctx->dHazBSnap.template as<double>() + (size_t) sOff * st->nHaz;
    hazSnapshotB<<<((st->nHaz * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(st->nHaz, nS, n, st->dHazRow.template as<int>(), B, bsnap);
    hazRepairRhs<<<((st->nHaz * nS + BLOCK - 1) / BLOCK), BLOCK, 0, ctx->stream>>>(st->nHaz, nS, n, st->dHazRow.template as<int>(),
            st->dHazDonor.template as<int>(), st->dHazDonorHaz.template as<int>(), st->dHazSign.template as<char>(),
            ctx->dHazChoice.template as<int>() + (size_t) sOff * st->nHaz, bsnap, B);
}

// OLF_GPU_REJECT_TRACE=1: WHY a scenario left the batch for the CPU replay (status -2). The census
// counts them under one tag ("RUNTIME:status=-2"); this names the gate, per slot, so the Java side
// can pair it with the contingency id (GpuBatchPrecomputer prints the id under the same switch).
static inline void traceReject(int slot, const char* reason, int outer, int it) {
    static const bool on = std::getenv("OLF_GPU_REJECT_TRACE") != nullptr;
    if (on) {
        std::fprintf(stderr, "GPU_REJECT slot=%d reason=%s outer=%d it=%d\n", slot, reason, outer, it);
    }
}

// A linear solve whose residual ratio ||J.dx-B|| / ||F|| exceeds this did NOT solve the system: the
// factorization is wrong (unpivoted LU on a structure that needs pivoting), not merely inaccurate.
static const double LIN_SOLVE_FAIL_RATIO = 1e-2;

// Per-scenario score of the linear solve just done: max|J.dx-B| under both sign conventions and
// max|F|, one thread per scenario (n serial reads each; a few thousand per scenario, once per solve).
__global__ void linResidScoreBatch(int n, int nS, const double* rp, const double* rm, const double* F,
                                   double* score) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nS) return;
    double mp = 0.0;
    double mm = 0.0;
    double mf = 0.0;
    const double* rps = rp + (size_t) s * n;
    const double* rms = rm + (size_t) s * n;
    const double* fs = F + (size_t) s * n;
    for (int i = 0; i < n; i++) {
        mp = fmax(mp, fabs(rps[i]));
        mm = fmax(mm, fabs(rms[i]));
        mf = fmax(mf, fabs(fs[i]));
    }
    score[3 * s] = fmin(mp, mm);
    score[3 * s + 1] = mf;
    score[3 * s + 2] = (mf > 0.0 && isfinite(mp) && isfinite(mm)) ? fmin(mp, mm) / mf : 1e300;
}

__global__ void linResidRowBatch(int n, int nS, int nnz, const int* rowPtr, const int* colIdx,
                                 const double* vals, const double* dx, const double* F,
                                 double* rPlus, double* rMinus) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nS) return;
    int s = t / n;
    int i = t - s * n;
    const double* v = vals + (size_t) s * nnz;
    const double* d = dx + (size_t) s * n;
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
        acc += v[k] * d[colIdx[k]];
    }
    rPlus[t] = acc - F[t];
    rMinus[t] = acc + F[t];
}

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
    size_t cap = 0;                                          // bytes currently allocated
    ~DeviceBuf() { if (p) cudaFree(p); }
    template <typename T> T* as() { return static_cast<T*>(p); }
    /** Grow-only: keeps the buffer when it is already big enough, frees and reallocates when it is
     *  not. The old alloc() cudaMalloc'd over a live pointer on every call and leaked the previous
     *  allocation — any per-pass re-upload (the in-kernel TVC loop rebuilds the tap-override list
     *  every time a tap moves) needs a buffer that can be refilled without leaking. */
    void ensure(size_t bytes) {
        if (bytes > cap) {
            if (p) { cudaFree(p); p = nullptr; }
            CUDA_CHECK(cudaMalloc(&p, bytes));
            cap = bytes;
        }
    }
    void alloc(size_t bytes) { ensure(bytes); }
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
// ziIdx slot 10 packs two things: bit 0 = spanning-tree edge, bits 1-2 = which side is a DEAD-END
// STUB (0 = none, see GpuAcRowIndexing#ziDeadEndSide). On a stub the four rows around the branch
// would all have a ZERO DIAGONAL — ZERO_V and ZERO_PHI never depend on their own dummy, and the
// stub's P/Q balances carry only -DUMMY_P / -DUMMY_Q — and cuDSS's symmetric AMD ordering
// mis-factors that chain, perturbing the zero pivot and propagating its reciprocal. The Java packer
// therefore moves the stub's P/Q balances onto the dummy rows (slots 0..3 already carry them) and
// ZERO_V / ZERO_PHI onto the stub's own v / phi rows, which is what these two derive. Slots 4/5
// stay the dummy COLUMNS, because the kernel reads the dummy values through them.
inline int ziSpan(const int* z) { return z[10] & 1; }
inline int ziDeadSide(const int* z) { return (z[10] >> 1) & 3; }
inline int ziZeroVRow(const int* z) {
    int d = ziDeadSide(z);
    return d == 0 ? z[4] : (d == 1 ? z[6] : z[7]);           // the stub's own v row
}
inline int ziZeroPhiRow(const int* z) {
    int d = ziDeadSide(z);
    return d == 0 ? z[5] : (d == 1 ? z[8] : z[9]);           // the stub's own phi row
}

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
        const int* ix = ziIdx + e * 11;                      // b1P b1Q b2P b2Q dpRow dqRow v1 v2 ...
        for (int s = 0; s < 6; s++) {
            pairRc(ix[s], ix[s]);
        }
        // ...and the MOVED-V identities, exactly as the closed-branch, open-branch and shunt loops
        // already do. Slot 1/3 is the bus's Q ROW, which for an ordinary bus IS its V col — but a
        // remote gen-VC CONTROLLER has Qrow = -1 and a PILOT has Qrow = another bus's row, so its own
        // V col (slots 6/7) is a distinct row that nothing else claims. A zero-impedance branch
        // touching such a bus therefore left its V row unpaired and buildStructure rejected the whole
        // system. That is why busbar-section contingencies never reached the GPU: retaining switches
        // creates zero-impedance branches, and past ~20 retained sections one of them lands on a
        // remote-voltage-control bus (rte6515: row 2200 of 13048, appearing ONLY in zi[slot 6]).
        claimMovedV(ix[1], ix[6]);
        claimMovedV(ix[3], ix[7]);
        if (ziDeadSide(ix)) {
            // The stub's P/Q balances already sit on the dummy rows, where slots 0..3 self-pair and
            // -DUMMY_P / -DUMMY_Q is the diagonal. ZERO_V landed on the stub's v row, which
            // claimMovedV has just claimed (slot 1/3 no longer equals slot 6/7). ZERO_PHI's row is
            // the one nothing else names.
            pairRc(ziZeroPhiRow(ix), ziZeroPhiRow(ix));
        }
    }
    std::vector<char> colUsed(n, 0);
    for (int r = 0; r < n; r++) {
        if (rowMap[r] < 0) {                                 // every index in [0,n) is an active equation
            // Say WHERE the row appears. "row 2200 has no diagonal pairing" is a row index and nothing
            // else; resolving it back to an equation on the Java side is unreliable because the active
            // index shifts between solves. The builder, though, knows exactly which family arrays
            // mention the row and in which SLOT, and that is the whole diagnosis: a row no array
            // mentions was never packed, while a row some array mentions at a slot no pairing loop
            // reads is a GAP in that loop (e.g. the open-branch loop pairs only its Q slots, on the
            // assumption that "P rows pair via closed branches").
            auto where = [r](const char* name, const int* a, int count, int stride) {
                std::string out;
                for (int e = 0; e < count && a != nullptr; e++) {
                    for (int sl = 0; sl < stride; sl++) {
                        if (a[(size_t) e * stride + sl] == r) {
                            out += std::string(out.empty() ? "" : ",") + name + "[slot " + std::to_string(sl) + "]";
                            break;
                        }
                    }
                }
                return out;
            };
            std::string seen = where("cb", cbIdx, ncb, 12) + where("ob", obIdx, nob, 6)
                    + where("sh", shIdx, nsh, 5) + where("hv", hvIdx, nhv, 4)
                    + where("zi", ziIdx, nzi, 11);
            int dqHits = 0;
            for (int c = 0; c < ndq && dqRow != nullptr; c++) {
                if (dqRow[c] == r) {
                    dqHits++;
                }
            }
            throw std::runtime_error("equation row " + std::to_string(r) + " of " + std::to_string(n)
                                     + " has no diagonal pairing (appears in: "
                                     + (seen.empty() ? std::string("NOTHING") : seen)
                                     + ", dqRow hits " + std::to_string(dqHits) + ")");
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
        if (ziSpan(z)) {
            int zvRow = rowMap[ziZeroVRow(z)], zpRow = rowMap[ziZeroPhiRow(z)];
            pat.emplace_back(zvRow, z[6]);
            pat.emplace_back(zvRow, z[7]);
            pat.emplace_back(zpRow, z[8]);
            pat.emplace_back(zpRow, z[9]);
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
    int nHaz = 0, hazTot = 0;                                // zero-diagonal repair: hazard rows and their
    std::vector<int> hHazRow;                                // donor candidates (see hazRepairVals)
    DeviceBuf dHazRow, dHazOff, dHazDonorPtr, dHazDonor, dHazDonorHaz, dHazSign;
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
            static const char* ziTrace = std::getenv("OLF_GPU_ZI_DEST_TRACE");
            if (ziTrace != nullptr && z[5] == std::atoi(ziTrace)) {
                std::fprintf(stderr, "ZI_DEST e=%d dqCol=%d b1Q=%d b2Q=%d alt0=%d alt1=%d "
                             "dd2=%d dd3=%d dd6=%d dd7=%d\n",
                             e, z[5], b1Q, b2Q, ziQAlt[e * 2 + 0], ziQAlt[e * 2 + 1],
                             dd[2], dd[3], dd[6], dd[7]);
            }
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
            if (ziSpan(z)) {                                 // spanning-tree edge
                int zvRow = rowMap[ziZeroVRow(z)], zpRow = rowMap[ziZeroPhiRow(z)];
                dd[8] = -2 - zvRow;                          // F ZERO_V row
                dd[9] = pos.at(zvRow, z[6]);                 // J(ZERO_V, v1)
                dd[10] = pos.at(zvRow, z[7]);                // J(ZERO_V, v2)
                dd[11] = -2 - zpRow;                         // F ZERO_PHI row
                dd[12] = pos.at(zpRow, z[8]);                // J(ZERO_PHI, ph1)
                dd[13] = pos.at(zpRow, z[9]);                // J(ZERO_PHI, ph2)
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
    // ---- zero-diagonal repair tables (hazRepairVals): hazard rows + donor candidates, then the
    //      pattern union so a borrowed row lands in place ----
    std::vector<int> hazRows;
    std::vector<std::vector<int>> hazDonors;
    std::vector<char> hazSigns;
    {
        std::map<int, int> hazSlot;                          // physical row -> hazard slot
        auto slotOf = [&](int row, char sign) {
            auto it = hazSlot.find(row);
            if (it != hazSlot.end()) return it->second;
            int i = static_cast<int>(hazRows.size());
            hazSlot[row] = i;
            hazRows.push_back(row);
            hazDonors.emplace_back();
            hazSigns.push_back(sign);
            return i;
        };
        auto addDonor = [&](int slot, int donor) {
            if (donor < 0 || donor == hazRows[slot]) return;
            auto& v = hazDonors[slot];
            if (std::find(v.begin(), v.end(), donor) == v.end()) v.push_back(donor);
        };
        for (int e = 0; e < nzi; e++) {
            const int* z = ziIdx + static_cast<size_t>(e) * 11;
            if (!ziSpan(z) || ziDeadSide(z) != 0) continue;  // loop edge: identity dummies; stub: relocated
            // ZERO_V sits on the DUMMY_P row (its diagonal column is dummy-P, carried by the endpoint
            // P rows); ZERO_PHI on the DUMMY_Q row (dummy-Q, carried by the endpoint Q rows). The first
            // cut had these two swapped, so no dummy row ever found a donor: invisible at 28 couplers
            // (cuDSS still coped, with IR=2), fatal at rte6515's full 366 (the base solve's linear
            // residual was 1e9 and Newton never moved).
            int dP = rowMap[ziZeroVRow(z)];                  // hosts ZERO_V, diagonal column = dummy-P
            int dQ = rowMap[ziZeroPhiRow(z)];                // hosts ZERO_PHI, diagonal column = dummy-Q
            int pa = z[0] >= 0 ? rowMap[z[0]] : -1;
            int pb = z[2] >= 0 ? rowMap[z[2]] : -1;
            int qaRow = z[1] >= 0 ? z[1] : z[6];             // suppressed Q balance: the V slot hosts DISTR_Q
            int qbRow = z[3] >= 0 ? z[3] : z[7];
            int qa = qaRow >= 0 ? rowMap[qaRow] : -1;
            int qb = qbRow >= 0 ? rowMap[qbRow] : -1;
            int sP = slotOf(dP, -1);
            int sQ = slotOf(dQ, -1);
            addDonor(sP, pa);
            addDonor(sP, pb);
            addDonor(sQ, qa);
            addDonor(sQ, qb);
            for (int side = 0; side < 2; side++) {           // endpoint Q riding a relocation group
                int alt = ziQAlt[e * 2 + side];
                if (alt <= -2 && !st->hQrPtr.empty()) {
                    int g = -2 - alt;
                    for (int c = st->hQrPtr[g]; c < st->hQrPtr[g + 1]; c++) addDonor(sQ, st->hQrRows[c]);
                }
            }
            // Signs: the borrowing closes in 4-cycles P(a) -> ZERO_PHI -> Q(a) -> ZERO_V -> P(a) (a tree
            // allows no other cycle), and det(I + C) on such a cycle is 1 - (product of the four
            // signs): P rows +1, Q rows -1, dummy rows -1 gives det 2. All-bus-rows-+1 gave det 0 - a
            // singular, CONSISTENT system cuDSS solved to 1e-11 with the wrong answer (rte6515 500
            // busbars: 175 contingencies above 1e-9 kV where the tree-only borrowing had 3).
            if (pa >= 0) addDonor(slotOf(pa, 1), dQ);       // phi_a is carried by ZERO_PHI
            if (pb >= 0) addDonor(slotOf(pb, 1), dQ);
            if (qa >= 0) addDonor(slotOf(qa, -1), dP);      // v_a is carried by ZERO_V
            if (qb >= 0) addDonor(slotOf(qb, -1), dP);
        }
        if (!hazRows.empty()) {
            // DONOR ORDER decides LU stability without pivoting. The kernel takes the FIRST candidate
            // with a nonzero coefficient, and two rows borrowing the SAME donor come out near-identical
            // (a three-section bus: both couplers' ZERO_PHI rows plus the middle bus's Q row) -
            // nonsingular, but their elimination cancels to a pivot cuDSS's static pivoting cannot
            // survive (rte6515 full busbar structure: the repaired system solved to 9e-13 by a pivoted
            // LU, to 2.5e6 by cuDSS). So: a dummy row prefers the endpoint with the FEWEST incident
            // tree couplers (a leaf is unique to its coupler) and, among equals, the one no other
            // dummy row has taken; a bus row prefers a ZERO row no other bus row has taken.
            std::map<int, int> ziDeg;                        // bus row -> incident tree couplers
            for (int e = 0; e < nzi; e++) {
                const int* z = ziIdx + static_cast<size_t>(e) * 11;
                if (!ziSpan(z) || ziDeadSide(z) != 0) continue;
                for (int slot : {z[0], z[2], z[1] >= 0 ? z[1] : z[6], z[3] >= 0 ? z[3] : z[7]}) {
                    if (slot >= 0) ziDeg[rowMap[slot]]++;
                }
            }
            std::map<int, int> used;
            for (size_t i = 0; i < hazRows.size(); i++) {
                auto& v = hazDonors[i];
                std::stable_sort(v.begin(), v.end(), [&](int a, int b) {
                    int da = ziDeg.count(a) ? ziDeg[a] : 0;
                    int db = ziDeg.count(b) ? ziDeg[b] : 0;
                    if (da != db) return da < db;
                    return used[a] < used[b];
                });
                if (!v.empty()) used[v[0]]++;
            }
            std::vector<std::pair<int, int>> sorted(pat);
            std::sort(sorted.begin(), sorted.end());
            sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
            std::vector<int> rp(n + 1, 0);
            for (auto& rc : sorted) rp[rc.first + 1]++;
            for (int r = 0; r < n; r++) rp[r + 1] += rp[r];
            for (size_t i = 0; i < hazRows.size(); i++) {
                for (int d : hazDonors[i]) {
                    for (int k = rp[d]; k < rp[d + 1]; k++) {
                        pat.emplace_back(hazRows[i], sorted[k].second);
                    }
                    pat.emplace_back(d, hazRows[i]);         // the coefficient the numeric choice reads
                }
            }
        }
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
    st->nHaz = static_cast<int>(hazRows.size());
    if (st->nHaz > 0) {
        std::vector<int> hazOff(st->nHaz + 1, 0);
        std::vector<int> donorPtr(st->nHaz + 1, 0);
        std::vector<int> donor;
        std::vector<int> donorHaz;
        std::map<int, int> hazSlot;
        for (int i = 0; i < st->nHaz; i++) hazSlot[hazRows[i]] = i;
        for (int i = 0; i < st->nHaz; i++) {
            hazOff[i + 1] = hazOff[i] + (st->rowPtr[hazRows[i] + 1] - st->rowPtr[hazRows[i]]);
            for (int d : hazDonors[i]) {
                donor.push_back(d);
                auto it = hazSlot.find(d);
                donorHaz.push_back(it == hazSlot.end() ? -1 : it->second);
            }
            donorPtr[i + 1] = static_cast<int>(donor.size());
        }
        st->hazTot = hazOff[st->nHaz];
        st->hHazRow = hazRows;
        st->dHazRow.upload(hazRows.data(), hazRows.size() * sizeof(int));
        st->dHazOff.upload(hazOff.data(), hazOff.size() * sizeof(int));
        st->dHazDonorPtr.upload(donorPtr.data(), donorPtr.size() * sizeof(int));
        st->dHazDonor.upload(donor.data(), donor.size() * sizeof(int));
        st->dHazDonorHaz.upload(donorHaz.data(), donorHaz.size() * sizeof(int));
        st->dHazSign.upload(hazSigns.data(), hazSigns.size() * sizeof(char));
        if (std::getenv("OLF_GPU_TRACE") != nullptr) {
            std::fprintf(stderr, "zero-diagonal repair: %d hazard rows, %zu donor candidates\n", st->nHaz, donor.size());
        }
    }
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
        applyReorderAlg(ds.config, st.nzi > 0);
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
    DeviceBuf dHazSnap, dHazBSnap, dHazChoice;               // zero-diagonal repair scratch (lazy)
    int hazCap = 0;
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
    int ir = irSteps(ctx->st->nzi > 0);
    CUDSS_CHECK(cudssConfigSet(ctx->ds.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)));
    applyReorderAlg(ctx->ds.config, ctx->st->nzi > 0);
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
        hazRepairValues(ctx, 1, ctx->dVals.as<double>());
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
        hazRepairRightHandSide(ctx, 1, ctx->dB.as<double>());
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
        static const bool linResidSeqOn = std::getenv("OLF_GPU_LIN_RESID") != nullptr;
        if (linResidSeqOn) {
            DeviceBuf rp;
            DeviceBuf rm;
            std::vector<double> hrp(n);
            std::vector<double> hrm(n);
            std::vector<double> hf(n);
            rp.alloc(n * sizeof(double));
            rm.alloc(n * sizeof(double));
            linResidRow<<<grid(n), BLOCK, 0, ctx->stream>>>(n, ctx->st->dRowPtr.as<int>(),
                    ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), rp.as<double>(), rm.as<double>());   // B = the (repaired) RHS
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(hrp.data(), rp.p, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hrm.data(), rm.p, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hf.data(), ctx->dF.p, n * sizeof(double), cudaMemcpyDeviceToHost));
            double mp = 0;
            double mm = 0;
            double mf = 0;
            int argw = -1;
            for (int i = 0; i < n; i++) {
                mp = std::max(mp, std::fabs(hrp[i]));
                if (std::fabs(hrm[i]) > mm) {
                    mm = std::fabs(hrm[i]);
                    argw = i;
                }
                mf = std::max(mf, std::fabs(hf[i]));
            }
            double best = std::min(mp, mm);
            std::fprintf(stderr, "LIN_RESID n=%d ||J.dx-F||inf=%.6e ||F||inf=%.6e ratio=%.3e worstRow=%d\n",
                         n, best, mf, mf > 0 ? best / mf : 0.0, argw);
        }
        if (it == 1) {                                       // same guard as the batched loop, batch of one
            DeviceBuf rp;
            DeviceBuf rm;
            DeviceBuf sc;
            rp.alloc(n * sizeof(double));
            rm.alloc(n * sizeof(double));
            sc.alloc(3 * sizeof(double));
            linResidRow<<<grid(n), BLOCK, 0, ctx->stream>>>(n, ctx->st->dRowPtr.as<int>(),
                    ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), rp.as<double>(), rm.as<double>());
            linResidScoreBatch<<<1, BLOCK, 0, ctx->stream>>>(n, 1, rp.as<double>(), rm.as<double>(),
                    ctx->dF.as<double>(), sc.as<double>());
            double hsc[3];
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(hsc, sc.p, 3 * sizeof(double), cudaMemcpyDeviceToHost));
            if (!(hsc[2] <= LIN_SOLVE_FAIL_RATIO)) {
                throw std::runtime_error("device linear solve failed (||J.dx-B||/||F|| = " + std::to_string(hsc[2])
                                         + "): the structure does not factor unpivoted - CPU fallback");
            }
        }
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

// OLF_GPU_CB_PROBE=<cb element index>: the pi row the device ACTUALLY evaluates for that closed
// branch in EVERY slot, after findTapOverride, alongside its disabled/active bits. The host knows
// what it MEANT to send (GpuBatchSideOpen.danglingCbRow); this says what the kernel read. A
// side-open that reduces to a pure shunt must show y = 0 with the equivalent on the live side's
// g/b — anything else and the override never reached the fill.
// OLF_GPU_STATE_DUMP=<maxIt>: for slot 0 and iterations 0..maxIt, EVERY row's mode, target, state
// and residual. Only usable on a small system, which is exactly where a reproducer lives — on a
// 16-row network it shows at a glance which variable ran away and which equation let it.
__global__ void stateDumpBatch(int n, const double* F, const int* mode, const double* target,
                               const double* x, int s, int it) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < n; i++) {
        size_t k = (size_t) s * n + i;
        printf("[STATE] it=%d row=%d mode=%d target=%.9e x=%.9e F=%.9e\n",
               it, i, mode[k], target[k], x[k], F[k]);
    }
}

__global__ void cbRowProbe(int nS, int e, const double* cbIn, int nR1Ov,
                           const int* r1OvSlotElem, const double* r1OvVal,
                           int nElements, int elemOffset, const char* disabled, const char* active) {
    for (int s = 0; s < nS; s++) {
        const double* p = cbIn + (size_t) e * 8;
        double pOv[8];
        bool ov = findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv);
        if (ov) {
            p = pOv;
        }
        printf("[CB_ROW] s=%d e=%d ov=%d dis=%d act=%d y=%.12e g1=%.12e g2=%.12e b1=%.12e b2=%.12e "
               "ksi=%.12e r1=%.12f a1=%.12f\n",
               s, e, ov ? 1 : 0, (int) disabled[(size_t) s * nElements + elemOffset + e], (int) active[s],
               p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    }
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

// A DISTR_Q controller that stops controlling in ONE scenario (its bus lost its last voltage
// generator while the group's other controllers survive). OLF's updateReactiveKeys zeroes its key
// and renormalizes over the rest — the same treatment a PINNED member gets, except its target
// contribution stays its own targetQ instead of a reactive limit.
// A row an islanding outage FROZE to a flat identity state. Its target is not the outer loops' to
// write: OLF removes an islanded bus's equations altogether, so nothing there re-derives a target.
// Measured on rte6515 LINE-1525-590: the DISTR_Q target writer stamped 0.0 over a frozen voltage row
// still holding 1.0, which is an exact 1.0 residual on an identity row and the whole 1.192e-06 kV.
__device__ __forceinline__ bool rowIsFrozen(int nFrozen, const int* frozenSlotRow, int s, int row) {
    for (int k = 0; k < nFrozen; k++) {
        if (frozenSlotRow[2 * k] == s && frozenSlotRow[2 * k + 1] == row) return true;
    }
    return false;
}

__device__ __forceinline__ bool dqMemberInactive(int nInact, const int* inactSlotMember, int s, int m) {
    for (int k = 0; k < nInact; k++) {
        if (inactSlotMember[2 * k] == s && inactSlotMember[2 * k + 1] == m) {
            return true;
        }
    }
    return false;
}

static int gpuDqTraceRow() {
    // OLF_GPU_DQ_TRACE=<deviceRow>: every DISTR_Q residual contribution landing in that row.
    static const char* e = std::getenv("OLF_GPU_DQ_TRACE");
    static int v = e != nullptr ? std::atoi(e) : -1;
    return v;
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
                                int nDqInact, const int* dqInactSlotMember,
                                const int* dqKMember, const int* dqJMember, const double* mKey,
                                const int* mRlEntry, const int* mGroupStart, const int* mGroupCount,
                                const char* limitType, int nrl,
                                const int* mQrGroup, const int* qrPtr, const int* qrRows,
                                int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                int nShOv, const int* shOvSlotElem, const double* shOvGb,
                                int dqTraceRow) {
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
            sum += (pinned || dqMemberInactive(nDqInact, dqInactSlotMember, s, m)) ? 0.0 : mKey[m];
        }
        // the PINNED member's q terms STAY in every survivor's row (weight = the renormalized
        // qPct of the ROW OWNER); the pin is balanced on the TARGET side instead — OLF's
        // getGeneratorReactivePowerDistributionTarget adds qPct_k * targetQ_j for each frozen j
        // (updateDistrQTargetsBatch keeps the per-scenario targets in step with the pin states)
        int reK = mRlEntry[kM];
        bool kPinned = reK >= 0 && limitType[(size_t) s * nrl + reK] != 0;
        double qpk = (sum == 0.0 || kPinned || dqMemberInactive(nDqInact, dqInactSlotMember, s, kM))
                ? 0.0 : mKey[kM] / sum;
        w = qpk - (dqJMember[c] == kM ? 1.0 : 0.0);
        if (kind == 3 && dqSide[c] != 0) {
            // The host folds the SIDE SIGN into a kind-3 weight (+dummyQ where the controller is
            // bus1, -dummyQ where it is bus2 — exactly createReactiveTerms' term); this live
            // recomputation rebuilds qPct_k - delta_jk and must put that sign back. Without it a
            // coupler whose BOTH ends are controllers of the same group cancels to zero at
            // qPct = 0.5: measured on rte6515 BUS-1014_COUPLER_1, where DUMMY_Q then appeared in NO
            // equation at all and the Jacobian had a structurally zero column, against OLF's
            // coefficient of +1.0 in DISTR_Q of VL-1014_1.
            w = -w;
        }
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
        double g = sp[0];
        double b = sp[1];
        applyShuntOverride(nShOv, shOvSlotElem, shOvGb, s, e, g, b);
        double so[SH_NOUT];
        evalShunt(xs[vcol], g, b, 0, so + 0, so + 1, so + 2, so + 3, so + 4);
        if (dst[0] <= -2 && (-2 - dst[0]) == dqTraceRow) {
            printf("DQF kind=1(shunt) elem=%d w=%.6f contrib=%.9f\n", e, w, w * so[2]);
        }
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * so[2]);
        if (dst[1] >= 0) atomicAdd(&valsS[dst[1]], w * so[3]);
        return;
    }
    if (kind == 3) {                             // zero-impedance DUMMY_Q: linear w*dummyQ
        if (dst[0] <= -2 && (-2 - dst[0]) == dqTraceRow) {
            printf("DQF kind=3(ziDummyQ) col=%d w=%.6f contrib=%.9f\n", e, w, w * xs[e]);
        }
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
        if (dst[0] <= -2 && (-2 - dst[0]) == dqTraceRow) {
            printf("DQF kind=2(halfOpen) elem=%d w=%.6f contrib=%.9f\n", e, w, w * oo[qi]);
        }
        if (dst[0] <= -2) atomicAdd(&FS[-2 - dst[0]], w * oo[qi]);
        if (dst[1] >= 0) atomicAdd(&valsS[dst[1]], w * oo[qi + 1]);
        return;
    }
    // kind 0: closed-branch q1/q2 + its derivatives.
    // The per-scenario pi override MUST be honoured here, exactly as fillRemoteMemberQBatch does.
    // It is not a tap-only channel: an outaged branch on this path is nulled by zeroing its
    // admittance through it, NOT by the disabled mask. Reading the base pi instead kept three
    // transformers into an outaged busbar live against a far bus frozen at v=1.0 and injected
    // +0.813908 of phantom reactive into one DISTR_Q row (rte6515 BUS-1265_BBS, measured residual
    // -0.814738) - which moved the sharing solution 2.1 kV and then tripped a spurious RL pin.
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
        if (d <= -2 && (-2 - d) == dqTraceRow) {
            printf("DQF kind=0(closed) elem=%d w=%.6f v1=%.9f v2=%.9f y=%.6f contrib=%.9f\n",
                   e, w, v1, v2, p[0], val);
        }
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
    if (q[10] & 1) {                                         // spanning-tree edge: voltage/angle coupling
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
                                    const double* svcRlVt, int nrl, int s, int i, double baseTarget) {
    if (svcRlVt != nullptr) {                                // a zone moved this controller's target: the
        double v = svcRlVt[(size_t) s * nrl + i];            // switch-back must compare against the MOVED one
        if (!isnan(v)) {
            return v;
        }
    }
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
                                     int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal, const double* svcRlVt,
                                     const char* rlRemote, const int* rlCmpVrow,
                                     const double* target, const double* rlMinQb, const double* rlMaxQb,
                                     const double* rlLoadQb, const char* rlBaseLt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    int vrow = rlVrow[i];
    size_t idx = (size_t) s * n + vrow;
    size_t st = (size_t) s * nrl + i;
    // OLF's own reason for the base pin, when the host carried it: it distinguishes a
    // realistic-voltage pin (3/4, target = generation targetQ) from a Q-limit pin (1/2, target =
    // the limit). The nearest-limit guess below turns the former into the latter, and the
    // curve-limit drift then re-freezes it AT the limit (rte6515 VL-1251_1: -0.0035 -> -1.0000).
    char olfKind = rlBaseLt != nullptr ? rlBaseLt[i] : 0;
    if (rlRemote != nullptr && rlRemote[i]) {
        // REMOTE group member (plan-B3): base mode 1 at its row is the UNPINNED DISTR_Q share —
        // only the canonical mode 4 marks a base pin. The pilot's V equals the group target at the
        // base solution (the survivors keep controlling), so the v-vs-target heuristic is
        // DEGENERATE here — key the direction on WHICH limit the extracted BUS_TARGET_Q target
        // equals instead (target = limit − loadQ was written by the base activation).
        limitType[st] = 0;
        if (baseMode[vrow] == 4) {
            double tgt = target[idx];
            limitType[st] = olfKind != 0 ? olfKind
                    : (fabs(tgt - (rlMaxQb[i] - rlLoadQb[i])) <= fabs(tgt - (rlMinQb[i] - rlLoadQb[i])) ? 2 : 1);
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
        double vt = rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
        limitType[st] = olfKind != 0 ? olfKind
                : ((x[idx] < vt) ? 2 : 1);                  // v below target ⇒ maxQ pinned; above ⇒ minQ pinned
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
// rlCurveLimitsBatch reads each generator's OWN cumulative P from the distributed-slack water-fill
// (dGenCum), which clamps PER GENERATOR, instead of splitting its bus's movement by a share fixed at
// pack time. The share is exact only while the intra-bus split stays proportional, and a clamp breaks
// that: on rte6515 GEN-6172 at VL-6139_0 the 451 MW unit takes +0.0442 where its share says +0.0890,
// which puts the bus's Q box out by 2.67 MVAr and the bus 0.0379 kV off. The share reconstruction
// survives ONLY as the fallback for entries genCum cannot serve (no DS slot, the per-generator
// water-fill did not run, or the outaged bus's own entries) — there is no switch back to it.
__global__ void rlCurveLimitsBatch(int nrl, int nScenarios, int ncv, int n, int cvExtrapolate,
                                   const int* cvIdx, const double* cvIn, const double* cvPts,
                                   const double* rlMinQ, const double* rlMaxQ, const double* rlLoadQ,
                                   const double* target,
                                   int nQOv, const int* qOvSlotEntry, const double* qOvMinMax,
                                   const int* cvOvPtr, const int* cvOvEntry, const double* cvOvVal,
                                   const double* genCum, const char* genCumValid, int ngen,
                                   double* minQDyn, double* maxQDyn, double* loadQDyn) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    double mn = rlMinQ[i];
    double mx = rlMaxQ[i];
    double lq = rlLoadQ[i];
    int ovLo = cvOvPtr != nullptr ? cvOvPtr[s] : 0;           // this scenario's curve repairs, if any
    int ovHi = cvOvPtr != nullptr ? cvOvPtr[s + 1] : 0;       // (CSR by scenario: a handful of entries)
    for (int g = 0; g < ncv; g++) {
        if (cvIdx[g * 5] != i) continue;
        // A generator outage at this bus skews the p_now reconstruction below. pStep is the part of
        // the row target the OUTAGE moved (the lost P, and any shed load) — not slack power, so it
        // moves nobody's curve; shareScale renormalizes this generator's slack share over the
        // generators still there; shareScale < 0 marks the lost generator's OWN entry, whose whole
        // capability left with it (its static base share is taken off by the qOv channel below).
        double pStep = 0;
        double shareScale = 1;
        for (int k = ovLo; k < ovHi; k++) {
            if (cvOvEntry[k] == g) {
                pStep = cvOvVal[2 * k];
                shareScale = cvOvVal[2 * k + 1];
                break;
            }
        }
        if (shareScale < 0) continue;
        int pRow = cvIdx[g * 5 + 1];
        const double* pts = cvPts + cvIdx[g * 5 + 2] * 3;
        int npts = cvIdx[g * 5 + 3];
        int slot = cvIdx[g * 5 + 4];
        // This generator's OWN movement, straight from the distributed-slack water-fill, which clamps
        // PER GENERATOR. Splitting its bus's movement by a fixed share is exact only while the
        // intra-bus split stays proportional, and a clamp breaks that: measured on rte6515 GEN-6172 at
        // VL-6139_0, the 451 MW unit takes +0.0442 where its share says +0.0890 and the remainder goes
        // to its four siblings. The share reconstruction stays as the fallback for an entry with no
        // slot, and for the outaged bus's own entries, whose pStep/shareScale repair is expressed in
        // those terms.
        double pNow;
        // genCum already reflects the outage — the lost generator is marked out of the water-fill and
        // the bus's anchor carries its P — so pStep and shareScale, which exist only to repair the
        // share reconstruction on the outaged bus, are not needed when it is available. The lost
        // generator's OWN entry never reaches here (shareScale < 0 continues above).
        if (genCum != nullptr && genCumValid != nullptr && genCumValid[s] && slot >= 0) {
            pNow = cvIn[g * 5] + genCum[(size_t) s * ngen + slot];
        } else {
            pNow = cvIn[g * 5]
                    + cvIn[g * 5 + 1] * shareScale * (target[(size_t) s * n + pRow] - cvIn[g * 5 + 2] - pStep);
        }
        mn += rlCurveEval(pNow * 100.0, pts, npts, 0, cvExtrapolate) / 100.0 - cvIn[g * 5 + 3];
        mx += rlCurveEval(pNow * 100.0, pts, npts, 1, cvExtrapolate) / 100.0 - cvIn[g * 5 + 4];
    }
    // A generator lost at a bus that KEEPS voltage control takes its reactive capability with it. The
    // static aggregate is run-constant and still holds it, while OLF's bus.getMinQ()/getMaxQ() sum only
    // the generators still there, so the scenario's own delta comes in here. Without curves this is the
    // ONLY reason the dynamic limits exist, which is why the seeding above starts from the static pair.
    for (int k = 0; k < nQOv; k++) {
        if (qOvSlotEntry[2 * k] == s && qOvSlotEntry[2 * k + 1] == i) {
            mn += qOvMinMax[3 * k];
            mx += qOvMinMax[3 * k + 1];
            lq += qOvMinMax[3 * k + 2];                      // a SHED load's reactive power: the pinned
        }                                                    // target and the controller's own q both
    }                                                        // measure against the bus's load target Q
    minQDyn[(size_t) s * nrl + i] = mn;
    maxQDyn[(size_t) s * nrl + i] = mx;
    loadQDyn[(size_t) s * nrl + i] = lq;
}

// DISTR_Q target compensation (plan-B3): OLF's DISTR_Q row target is
// (qPct_k - 1)*targetQ_k + qPct_k*sum_{j!=k} targetQ_j, and targetQ_j is nonzero exactly for a
// FROZEN member (pinned at a limit). Recompute every UNPINNED member row's target from the
// per-scenario limitType so an in-flight pin/unpin keeps the targets consistent with the live
// renormalized weights. Idempotent per pass; one thread per (member, scenario).
// Per-scenario member targetQ. A member bus that loses reactive LOAD changes its own targetQ
// (targetQ = generation - load), and every DISTR_Q target in its group reads that value twice: once
// inside the shared sum and once as its own subtracted term. Seeding the row targets host-side is not
// enough, because this kernel RECOMPUTES them from mTargetQ on every reactive-limits pass and would
// put the base value straight back (measured on rte6515 LOAD-1263: the seeded row came back at
// final = base = 0.016, residual still 4.6e-02). The override list is tiny — one entry per outaged
// member bus — so a linear scan per lookup costs nothing.
__device__ inline double dqTargetQOf(int nTqOv, const int* tqOvKey, const double* tqOvVal,
                                     int s, int nMembers, int idx, const double* mTargetQ) {
    if (nTqOv > 0) {
        int want = s * nMembers + idx;
        for (int i = 0; i < nTqOv; i++) {
            if (tqOvKey[i] == want) {
                return tqOvVal[i];
            }
        }
    }
    return mTargetQ[idx];
}

__global__ void updateDistrQTargetsBatch(int nMembers, int nScenarios, int n, int nrl,
                                         const double* mKey, const int* mRlEntry,
                                         const int* mGroupStart, const int* mGroupCount,
                                         const int* mRow, const char* mHasRow,
                                         const int* rlVrowArr, const double* rlMinQ, const double* rlMaxQ,
                                         const double* rlLoadQ, const char* limitType, int cvStride,
                                         double* target, const char* rlActive, const double* mTargetQ,
                                         const int* mQrGroup, const int* qrPtr, const int* qrRows,
                                         const int* mode, const double* pilotQ, int nQr,
                                         int nDqInact, const int* dqInactSlotMember,
                                         int nFrozen, const int* frozenSlotRow,
                                         int nTqOv, const int* tqOvKey, const double* tqOvVal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nMembers * nScenarios) return;
    int k = tid % nMembers;
    int s = tid / nMembers;
    if (!rlActive[s]) return;
    int qg = mQrGroup != nullptr ? mQrGroup[k] : -1;
    if (!mHasRow[k] && qg < 0) return;                       // only members owning a packed DISTR_Q row
                                                             // (or linked to a relocation group, plan-B4)
    if (mRow[k] >= 0 && rowIsFrozen(nFrozen, frozenSlotRow, s, mRow[k])) {
        return;                                              // islanded: the row is frozen, not ours to write
    }
    if (mRow[k] >= 0 && mode[(size_t) s * n + mRow[k]] == 4) {
        // The row hosts its OWN reactive balance this scenario (mode 4): an RL pin (the RL kernel
        // owns the target) or a control the contingency KILLED (the injection override owns it,
        // OLF's BUS_TARGET_Q). A DISTR_Q target has no business here — fillDistrQBatch already
        // skips the row's DISTR_Q emissions for the same reason. Measured on rte6515 BUS-1055_BBS_1:
        // the dead control's survivor converged at target 0, then this pass put the base DISTR_Q
        // target (0.02 pu) back and the re-solve balanced 2 MVar the CPU never asked for (0.65 kV).
        return;
    }
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
    double sumTargetQ = 0.0;
    for (int m = gs; m < gs + gc; m++) {
        int re = mRlEntry[m];
        bool pinned = re >= 0 && limitType[(size_t) s * nrl + re] != 0;
        bool inactive = dqMemberInactive(nDqInact, dqInactSlotMember, s, m);
        sum += (pinned || inactive) ? 0.0 : mKey[m];
        if (pinned) {
            // a FROZEN member's targetQ IS its q limit: OLF's RL switch calls setGenerationTargetQ(limit),
            // so getTargetQ() = limit - loadTargetQ. Unpinned members keep their packed targetQ.
            size_t li = cvStride ? (size_t) s * nrl + re : (size_t) re;
            char lt = limitType[(size_t) s * nrl + re];
            sumTargetQ += (lt == 1 ? rlMinQ[li] : rlMaxQ[li]) - rlLoadQ[li];
        } else {
            sumTargetQ += dqTargetQOf(nTqOv, tqOvKey, tqOvVal, s, nMembers, m, mTargetQ);
        }
    }
    double qpk = (sum == 0.0 || dqMemberInactive(nDqInact, dqInactSlotMember, s, k))
            ? 0.0 : mKey[k] / sum;
    // AcTargetVector.getGeneratorReactivePowerDistributionTarget, verbatim:
    //   (qPct_k - 1) * targetQ_k + qPct_k * sum_{j != k} targetQ_j
    // which is qPct_k * sum_over_ALL_j targetQ_j - targetQ_k. k is never pinned here (the RL kernel
    // owns a pinned member's row and we returned above), so its own term is the packed targetQ.
    // This used to read `qpk * frozen` — the same expression with every UNPINNED targetQ assumed
    // zero. That assumption holds only for groups whose members carry no load and no generator
    // targetQ; anywhere else the device solved the row against a target short by exactly
    // qPct_k * sum_unpinned targetQ_j - targetQ_k, and drove the members to equal reactive
    // injections instead of equally-shared DEVIATIONS from their targets.
    target[(size_t) s * n + mRow[k]] = qpk * sumTargetQ
            - dqTargetQOf(nTqOv, tqOvKey, tqOvVal, s, nMembers, k, mTargetQ);
}

// plan-B4: capture, per (scenario, relocation group), the pilot's Q target from the BASE carrier's
// row — the base target upload and the per-scenario injection overrides both address the pilot's
// reactive balance at that row (structure-time qRow). updateDistrQTargetsBatch hands the value to
// whichever member row currently carries the relocated pilot Q. Runs BEFORE the RL mode seeding so
// a warm-continuation pin cannot overwrite the value first.
__global__ void capturePilotQBatch(int nQr, int nScenarios, int n, const int* qrPtr, const int* qrRows,
                                   const int* baseMode, const double* targetBase, double* pilotQ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nQr * nScenarios) return;
    int g = tid % nQr;
    int s = tid / nQr;
    int row = qrRows[qrPtr[g] + relocChoose(g, qrPtr, qrRows, baseMode)];
    pilotQ[(size_t) s * nQr + g] = targetBase[row];          // the BASE carrier's BASE target (n, untiled)
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
                                    const char* rlActive, char* switched,
                                    int nFrozen, const int* frozenSlotRow) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nQr * nScenarios) return;
    int g = tid % nQr;
    int s = tid / nQr;
    if (!rlActive[s] || !qrGenVc[g]) return;
    int pilotRow = qrRows[qrPtr[g]];
    // An ISLANDED pilot has no equation at all on the CPU: OLF removes the bus. Its row here is
    // frozen to a flat voltage identity, and every member row is frozen to identity too, so the
    // mode scan below finds no carrier and reads that as "control lost" — flipping the frozen
    // identity into a power row. The target override then re-stamps the frozen VOLTAGE value over
    // that power row, leaving a residual of exactly 1.0 that no Newton step can ever clear, so
    // every later re-solve buys iterations OLF does not run. Measured on LINE-3938-646: the whole
    // network moved 8.8e-08 kV off the CPU answer with identical solve counts.
    if (rowIsFrozen(nFrozen, frozenSlotRow, s, pilotRow)) return;
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

// A pilot that STOPPED being one of its OWN controllers (this scenario outaged its generators) keeps
// its BUS_TARGET_V while the surviving controllers still hold its voltage — the reactive balance it
// just acquired rides one of THEIR rows. When every survivor PINS, each of those rows flips to mode 4
// to host its own pinned balance, and the pilot's balance is evicted with nowhere left to go: the
// solve converges happily on a system one equation short (rte6515 GEN-1522: OLF's residual 0.601 pu
// on BUS_TARGET_Q of the pilot, every device equation converged). OLF drops the now-uncontrollable
// BUS_TARGET_V and activates the pilot's own BUS_TARGET_Q on the freed row; this makes the same move
// INSIDE the solve, so the re-solve sees it.
//
// Mode 4 is "row hosts its own Q balance" via the member ALTERNATE dest. It only reaches the pilot's
// q because the extraction now gives a pilot-that-is-its-own-controller that alternate; without it
// the row assembles EMPTY and the flip is a measured no-op (state unchanged to 10 digits).
// Reversible: if a later pass unpins a survivor, the V identity and its target come back.
__global__ void pilotRowAllPinnedBatch(int nEntries, int nScenarios, int n, int nrl,
                                       const int* entSlot, const int* entRow, const double* entQTarget,
                                       const int* entOff, const int* entCount, const int* entRlPool,
                                       const char* limitType, int* mode, double* target,
                                       char* flipped, double* savedVTarget,
                                       const char* rlActive, char* switched) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEntries) return;
    int s = entSlot[e];
    if (s < 0 || s >= nScenarios || !rlActive[s]) return;
    int row = entRow[e];
    if (row < 0 || row >= n) return;
    size_t ri = (size_t) s * n + row;
    bool allPinned = entCount[e] > 0;
    for (int k = 0; k < entCount[e]; k++) {
        int rl = entRlPool[entOff[e] + k];
        if (rl < 0 || rl >= nrl || limitType[(size_t) s * nrl + rl] == 0) {
            allPinned = false;                              // this one still carries its DISTR_Q row
            break;
        }
    }
    if (allPinned && !flipped[e] && mode[ri] == 0) {
        savedVTarget[e] = target[ri];
        mode[ri] = 4;
        target[ri] = entQTarget[e];
        flipped[e] = 1;
        switched[s] = 1;
    } else if (!allPinned && flipped[e]) {
        mode[ri] = 0;
        target[ri] = savedVTarget[e];
        flipped[e] = 0;
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
                                       int nShOv, const int* shOvSlotElem, const double* shOvGb,
                                       int traceRow) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrmq * nScenarios) return;
    int c = tid % nrmq;
    int s = tid / nrmq;
    if (!rlActive[s]) return;
    int e = rmqElem[c];
    int kind = rmqKind[c];
    if (kind == 0 && disabled[s * nElements + cbOffset + e]) {
        if (rmqRow[c] == traceRow) {
            printf("RMQ s=%d c=%d kind=0(closed) elem=%d SKIPPED disabled\n", s, c, e);
        }
        return;
    }
    if (kind == 2 && disabled[s * nElements + obOffset + e]) {
        if (rmqRow[c] == traceRow) {
            printf("RMQ s=%d c=%d kind=2(halfOpen) elem=%d SKIPPED disabled\n", s, c, e);
        }
        return;
    }
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
        if (rmqRow[c] == traceRow) {
            printf("RMQ s=%d c=%d kind=1(shunt) elem=%d w=%.6f contrib=%.9f\n", s, c, e, w, w * so[2]);
        }
        atomicAdd(q, w * so[2]);
        return;
    }
    if (kind == 3) {                                         // zero-impedance DUMMY_Q linear term
        if (rmqRow[c] == traceRow) {
            printf("RMQ s=%d c=%d kind=3(ziDummyQ) dqCol=%d w=%.6f dummyQ=%.9f contrib=%.9f\n",
                   s, c, e, w, xs[e], w * xs[e]);
        }
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
        if (rmqRow[c] == traceRow) {
            printf("RMQ s=%d c=%d kind=2(halfOpen) elem=%d side=%d w=%.6f contrib=%.9f\n",
                   s, c, e, rmqSide[c], w, w * oo[rmqSide[c] == 0 ? 6 : 2]);
        }
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
    if (rmqRow[c] == traceRow) {
        printf("RMQ s=%d c=%d kind=0(closed) elem=%d side=%d w=%.6f v1=%.9f v2=%.9f"
               " ph1=%.9f ph2=%.9f r1=%.6f y=%.6f b1=%.6f b2=%.6f contrib=%.9f\n",
               s, c, e, rmqSide[c], w, v1, v2, ph1, ph2, r1, p[0], p[3], p[4],
               w * o[rmqSide[c] == 0 ? 7 : 21]);
    }
    atomicAdd(q, w * o[rmqSide[c] == 0 ? 7 : 21]);           // q1 / q2 residual outputs
    return;
}

// OLF AbstractHvdcAcEmulationFlowEquationTerm.getAbsActivePowerWithLosses + HvdcUtils losses.
__device__ inline double hvdcAbsActivePowerWithLosses(double pController, double lossController,
                                                      double lossNonController, double r) {
    double lineInputPower = (1.0 - lossController) * fabs(pController);
    double losses = r * lineInputPower * lineInputPower;     // HvdcUtils.getHvdcLineLosses(p, 1, r)
    return (1.0 - lossNonController) * (lineInputPower - losses);
}

/** OLF's FreezingHvdcACEmulation (mode 0) and AcHvdcAcEmulationLimits (mode 1) as one driver phase.
 *  Transitions are applied INCREMENTALLY to the converter-bus target rows — target += newDelta -
 *  appliedDelta — so they compose with whatever else already overrides that row (a lost load on a
 *  converter bus, the DS carry) without rebuilding the whole override list, which is what the host
 *  orchestrator had to do on every round. */
__global__ void hvdcAcEmulationLoopBatch(int nh, int nScenarios, int n, int nElements, int hvOffset,
                                         int mode,
                                         const int* hvElem, const int* hvBus1Row, const int* hvBus2Row,
                                         const double* hvIn, const char* hvCtgOut,
                                         const double* x, double* target, char* disabled,
                                         int* status, double* appliedD1, double* appliedD2,
                                         const char* rlActive, char* switched) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nh * nScenarios) return;
    int h = tid % nh;
    int s = tid / nh;
    if (!rlActive[s] || hvCtgOut[(size_t) s * nh + h]) return;    // outaged: the contingency owns that bit
    const double* p = hvIn + (size_t) h * 9;                 // p0 k pMax12 pMax21 loss1 loss2 r baseD1 baseD2
    int st = status[(size_t) s * nh + h];
    int next = st;
    if (mode == 0) {                                         // FreezingHvdcACEmulation: unfreeze
        if (st != HV_FROZEN) return;
        next = HV_LINEAR;
    } else {                                                 // AcHvdcAcEmulationLimits
        size_t elemSlot = (size_t) s * nElements + hvElem[h];
        if (st == HV_LINEAR && disabled[elemSlot]) return;
        const double* xs = x + (size_t) s * n;
        double ph1 = xs[hvBus1Row[h] >= 0 ? hvBus1Row[h] : ~hvBus1Row[h]];   // ~row: P row inactive (slack)
        double ph2 = xs[hvBus2Row[h] >= 0 ? hvBus2Row[h] : ~hvBus2Row[h]];
        double pp1 = hvdc::p1(p[0], p[1], p[4], p[5], p[6], ph1, ph2);
        double pp2 = hvdc::p2(p[0], p[1], p[4], p[5], p[6], ph1, ph2);
        if (st == HV_LINEAR) {                               // OLF checkAcEmulationMode
            if (pp1 >= 0) {
                if (pp1 > p[2]) { next = HV_SAT_1_TO_2; }
            } else if (pp2 > p[3]) {
                next = HV_SAT_2_TO_1;
            }
        } else if (st == HV_SAT_1_TO_2 || st == HV_SAT_2_TO_1) {
            if ((pp1 > 0 && pp1 < p[2]) || (pp2 > 0 && pp2 < p[3])) {
                next = HV_LINEAR;
            } else if (pp1 < 0 && st == HV_SAT_1_TO_2) {
                next = HV_SAT_2_TO_1;
            } else if (pp2 < 0 && st == HV_SAT_2_TO_1) {
                next = HV_SAT_1_TO_2;
            }
        }
    }
    if (next == st) return;
    double nd1 = 0.0;
    double nd2 = 0.0;
    size_t elemSlot = (size_t) s * nElements + hvElem[h];
    if (next == HV_LINEAR) {
        disabled[elemSlot] = 0;                              // re-enabled: its own equation terms carry the flow
    } else {
        disabled[elemSlot] = 1;
        if (next == HV_SAT_1_TO_2) {
            nd1 = -p[2];
            nd2 = hvdcAbsActivePowerWithLosses(p[2], p[4], p[5], p[6]);
        } else {
            nd1 = hvdcAbsActivePowerWithLosses(p[3], p[5], p[4], p[6]);
            nd2 = -p[3];
        }
    }
    double want1 = nd1 - p[7];                               // composition: base - baseD + d
    double want2 = nd2 - p[8];
    if (hvBus1Row[h] >= 0) {                                 // an inactive P row (the slack) absorbs it, as OLF's does
        target[(size_t) s * n + hvBus1Row[h]] += want1 - appliedD1[(size_t) s * nh + h];
    }
    if (hvBus2Row[h] >= 0) {
        target[(size_t) s * n + hvBus2Row[h]] += want2 - appliedD2[(size_t) s * nh + h];
    }
    appliedD1[(size_t) s * nh + h] = want1;
    appliedD2[(size_t) s * nh + h] = want2;
    status[(size_t) s * nh + h] = next;
    switched[s] = 1;                                         // this outer loop is UNSTABLE for s
}

// Reactive-limit limit types: 0 unpinned, 1 MIN_Q, 2 MAX_Q, 3 MIN_REALISTIC_V, 4 MAX_REALISTIC_V, and
// 5 = frozen by OLF's GeneratorVoltageControlManager. The DISTR_Q key renormalization drops any member
// whose type is non-zero, which is why the freeze needs one; the reactive-limits pass must then LEAVE
// type 5 alone, because a bus whose voltage control is off is not one of its controllers at all.
#define RL_LIMIT_FROZEN 5

__global__ void reactiveLimitsBatch(const double* svcRlVt, int nrl, int nScenarios, int n, double tol, int maxSwitch,
                                    const int* rlVrow, const double* rlMinQ, const double* rlMaxQ,
                                    const double* rlLoadQ, const double* rlVtarget, const double* qBuf,
                                    const double* x, const char* rlActive, int* mode, double* target,
                                    char* limitType, char* switchCount, char* switched,
                                    char* passSwitched, char* savedMode,
                                    int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                                    const double* rowSlope, double* savedTarget, int cvStride,
                                    const char* rlRemote, const int* rlCmpVrow,
                                    const double* rlSvcBmin, const double* rlSvcBmax, const double* rlV2Base,
                                    const char* rlIslanded,
                                    const double* rlInitTgtQ, double vLoRobust, double vHiRobust,
                                    int robust,
                                    int noPinRow) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    if (!rlActive[s]) return;
    // A controller the contingency put in ANOTHER synchronous component is not a controller at all:
    // OLF disables that bus and its outer loops never see it. Its rows here are FROZEN at a flat
    // 1<0, so the q read out of qBuf is a placeholder, not a measurement — and switching on it pins
    // a PV bus that OLF leaves alone, which perturbs the MAIN component through any remote-control
    // or DISTR_Q group the controller belongs to. Measured on LINE-3320-350: the device pinned
    // VL-6432_0, an ISLANDED bus, giving it 20 pinned controllers against OLF's 19 and one extra
    // LOW_VOLTAGE violation at VL-1624.
    if (rlIslanded != nullptr && rlIslanded[(size_t) s * nrl + i]) return;
    // A controller FROZEN by the generator voltage control manager has its voltage control OFF, so it
    // is not one of this loop's controllers at all — OLF iterates buses with voltage control and never
    // reaches it. Leave the type (and the row) to the transformer phase that set it.
    if (limitType[(size_t) s * nrl + i] == RL_LIMIT_FROZEN) return;
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
        // The quadratic delta exists only for SVC entries: rlV2Base is the packed base v², NaN when
        // the extraction ran on a never-solved network (the base load flow's own context) — 0 × NaN
        // poisoned every entry's limits there and nothing ever pinned at base.
        double rdv2 = svcLim ? x[ridx] * x[ridx] - rlV2Base[i] : 0.0;
        double rliMin = rlMinQ[rli] + (svcLim ? rlSvcBmin[i] * rdv2 : 0.0);
        double rliMax = rlMaxQ[rli] + (svcLim ? rlSvcBmax[i] * rdv2 : 0.0);
        // rlRemote 2 = a PILOT that is one of its own controllers: unpinned, its row is the group's
        // voltage IDENTITY (mode 0), not a DISTR_Q share; pinned, its own Q balance (mode 4, the
        // alternate dest). Everything else (q via the remote q pack, the pilot's V as compare row)
        // is a remote member's. As a LOCAL entry its pin landed on mode 1, an EMPTY row.
        int pvForm = rlRemote[i] == 2 ? 0 : 1;
        if (limitType[rst] == 0) {
            if (mode[ridx] != pvForm) return;                // islanded / frozen this scenario
            // OLF_GPU_RL_NOPIN_ROW=<deviceRow>: never pin THIS row. Isolates one pin so the solve
            // it would have interrupted can be compared against the CPU on its own - the only way
            // to tell "the pin is wrong" from "the state the pin was taken at was already wrong".
            if (ridx % n == (size_t) noPinRow) return;
            double q = qBuf[ridx] + rlLoadQ[rli];            // own q via fillRemoteMemberQBatch
            // voltageRemoteControlRobustMode (ReactiveLimitsOuterLoop.checkControllerBus): the criteria
            // below test this REMOTE controller's OWN voltage — the same x row the quadratic SVC delta
            // reads — against the realistic band shrunk by the 1.02 margin.
            // OLF's robust criterion applies to REMOTE controllers only; a pilot that controls its
            // own bus (class 2) is a local controller in that respect
            bool unrealisticLo = robust && rlRemote[i] != 2 && x[ridx] < vLoRobust;
            bool unrealisticHi = robust && rlRemote[i] != 2 && x[ridx] > vHiRobust;
            if (q < rliMin - tol) {
                savedTarget[rst] = target[ridx];             // the DISTR_Q row target (NOT 0 in general:
                                                             // see updateDistrQTargetsBatch)
                mode[ridx] = 4;
                target[ridx] = rliMin - rlLoadQ[rli];
                limitType[rst] = 1;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            } else if (q > rliMax + tol) {
                savedTarget[rst] = target[ridx];
                mode[ridx] = 4;
                target[ridx] = rliMax - rlLoadQ[rli];
                limitType[rst] = 2;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            } else if (unrealisticLo || unrealisticHi) {
                // Q inside its limits but the controller's own voltage is already unrealistic: OLF
                // stops maintaining the REMOTE voltage and freezes the generation at its INITIAL
                // target Q, with a realistic-V limit type (3 = MIN_REALISTIC_V, 4 = MAX_REALISTIC_V)
                // that the switch-back reads like a min/max limit but the Q-limit drift ignores.
                savedTarget[rst] = target[ridx];
                mode[ridx] = 4;
                target[ridx] = rlInitTgtQ[i] - rlLoadQ[rli];
                limitType[rst] = unrealisticLo ? 3 : 4;
                switchCount[rst]++;
                switched[s] = 1;
                passSwitched[rst] = 1;
            }
            // OLF also sets the controller's V to a safe 1 pu for the next computation, under BOTH
            // robust criteria. That WRITES the state vector every other entry is still reading here,
            // so it runs as its own pass (rlRobustVResetBatch) once this one has finished.
        } else if (switchCount[rst] < maxSwitch) {           // pinned — switch-back tests the PILOT
            double v = x[(size_t) s * n + rlCmpVrow[i]];
            double vt = rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
            bool minPin = limitType[rst] == 1 || limitType[rst] == 3;   // LfBus.QLimitType.isMinLimit()
            bool back = (!minPin && v > vt) || (minPin && v < vt);
            if (back) {
                mode[ridx] = pvForm;                         // back to the DISTR_Q share row (the V identity for a pilot)
                target[ridx] = savedTarget[rst];
                limitType[rst] = 0;
                switched[s] = 1;
                passSwitched[rst] = 2;
            } else if ((cvStride || svcLim) && limitType[rst] <= 2) {   // curve/SVC limit drift while pinned
                // OLF re-freezes at a moved limit only for MIN_Q/MAX_Q (checkPqBusWithQLimitType tests
                // the type explicitly); a realistic-V freeze holds its initial target Q.
                double lim = limitType[rst] == 1 ? rliMin : rliMax;
                if (fabs(lim - (target[ridx] + rlLoadQ[rli])) > tol) {
                    target[ridx] = lim - rlLoadQ[rli];
                    switched[s] = 1;
                    passSwitched[rst] = 3;
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
    double dv2 = svcLim ? x[idx] * x[idx] - rlV2Base[i] : 0.0;   // see the remote branch: NaN base v² otherwise
    double liMin = rlMinQ[li] + (svcLim ? rlSvcBmin[i] * dv2 : 0.0);
    double liMax = rlMaxQ[li] + (svcLim ? rlSvcBmax[i] * dv2 : 0.0);
    if ((mode[idx] == 0 || mode[idx] == 2) && limitType[st] == 0) {   // currently PV — check the reactive limits
        double q = qBuf[idx] + rlLoadQ[li];
        if (q < liMin - tol) {
            savedTarget[st] = target[idx];                   // remember the row target for the switch-back
            savedMode[st] = static_cast<char>(mode[idx]);   // and the PV form (0, or 2 for a slope SVC)
            mode[idx] = 1;
            target[idx] = liMin - rlLoadQ[li];
            limitType[st] = 1;
            switchCount[st]++;
            switched[s] = 1;
            passSwitched[st] = 1;
        } else if (q > liMax + tol) {
            savedTarget[st] = target[idx];
            savedMode[st] = static_cast<char>(mode[idx]);
            mode[idx] = 1;
            target[idx] = liMax - rlLoadQ[li];
            limitType[st] = 2;
            switchCount[st]++;
            switched[s] = 1;
            passSwitched[st] = 1;
        }
    } else if (limitType[st] != 0 && switchCount[st] < maxSwitch) {   // pinned at a limit — check switchback
        double v = x[idx];
        double vt = rlTargetOf(nRlVtOv, rlVtOvSlotIdx, rlVtOvVal, svcRlVt, nrl, s, i, rlVtarget[i]);
        // OLF never back-switches a slope-SVC bus: checkPqBus runs with
        // canSwitchToPv = !bus.hasGeneratorsWithSlope() ("we don't support switching PQ to PV
        // for bus with one controller with slope") — once pinned, a slope row stays PQ.
        bool maxPin = limitType[st] == 2 || limitType[st] == 4;       // LfBus.QLimitType.isMaxLimit()
        bool minPin = limitType[st] == 1 || limitType[st] == 3;       // LfBus.QLimitType.isMinLimit()
        bool back = !slopeRow
                && ((maxPin && v > vt)                               // maxQ-pinned but v above target
                 || (minPin && v < vt));                             // minQ-pinned but v below target
        if (back) {
            mode[idx] = 0;                                  // PQ → PV: resume voltage control
            target[idx] = vt;
            limitType[st] = 0;
            switched[s] = 1;
            passSwitched[st] = 2;
        } else if ((cvStride || svcLim) && limitType[st] <= 2) {   // a realistic-V freeze keeps its target Q
            // Q-limit UPDATE while pinned (OLF checkPqBusWithQLimitType): a CURVE limit that moved
            // (DS moved the gen's P) or an SVC limit that moved (B*v^2 at the live V) re-freezes
            // the pinned Q at the NEW limit when it drifted by more than the mismatch tol.
            // Unreachable with static limits — gated on the channels.
            double lim = limitType[st] == 1 ? liMin : liMax;
            if (fabs(lim - (target[idx] + rlLoadQ[li])) > tol) {
                target[idx] = lim - rlLoadQ[li];
                switched[s] = 1;
                passSwitched[st] = 3;
            }
        }
    }
}

// AcloadFlowEngine.isStateUnrealistic, on the states INSIDE the sweep. OLF re-checks after every
// solver run once the last loop that can fix an unrealistic state has run, and a state outside
// [minRealisticVoltage, maxRealisticVoltage] on a non-fictitious bus at or above
// minNominalVoltageRealisticVoltageCheck ends the outer-loop cycle with UNREALISTIC_STATE — the run
// FAILS and reports no violations. The device only ever checked the FINAL restored state, so a
// scenario that passed through an unrealistic state and recovered came back CONVERGED where OLF
// reports FAILED. Flagged here after every inner solve and routed to the CPU replay, which applies
// OLF's own rule about WHICH solve the check belongs to (it depends on the configured loop order and
// on robust mode) instead of the device guessing at it. Rejecting is always sound: it changes the
// route, never the answer.
__global__ void realisticCheckBatch(int n, int nScenarios, const char* rowChecked, const double* x,
                                    const char* live, double minV, double maxV, char* bad) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nScenarios) return;
    int i = t % n;
    int s = t / n;
    if (!live[s] || !rowChecked[i]) return;
    double v = x[(size_t) s * n + i];
    if (v < minV || v > maxV) {
        bad[s] = 1;
    }
}

// voltageRemoteControlRobustMode, second half: "Set V to a safe 1 pu for the next computation".
// ReactiveLimitsOuterLoop does this to every REMOTE voltage controller it switched PV->PQ in this
// check whose OWN voltage is outside the realistic band — whether it switched on a reactive limit
// (the first robust block) or on the realistic-V criterion itself. It is a WRITE to the state
// vector the check kernel reads, so it gets its own pass: one thread per (entry, scenario), reading
// the same pre-pass x, which makes the result independent of the order the entries happen to run in.
// x is the next inner solve's warm start, so this is OLF's controllerBus.setV(1) verbatim.
__global__ void rlRobustVResetBatch(int nrl, int nScenarios, int n, const int* rlVrow, const char* rlRemote,
                                    const char* rlActive, const char* passSwitched, double* x,
                                    double vLoRobust, double vHiRobust) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrl * nScenarios) return;
    int i = tid % nrl;
    int s = tid / nrl;
    if (!rlActive[s]) return;
    if (rlRemote == nullptr || rlRemote[i] != 1) return;     // robust mode is remote-controller only (not a class-2 pilot)
    if (passSwitched[(size_t) s * nrl + i] != 1) return;     // switched PV -> PQ in THIS check
    size_t idx = (size_t) s * n + rlVrow[i];
    if (x[idx] < vLoRobust || x[idx] > vHiRobust) {
        x[idx] = 1.0;
    }
}

// OLF's GLOBAL forced-PV election (ReactiveLimitsOuterLoop.switchPvPq): when EVERY bus still PV
// would switch PV→PQ in this check (remainingPvBusCount == 0), the strongest candidate stays PV —
// highest controlled-bus nominal V, then highest bus target P (LIVE: the DistributedSlack may have
// moved it), then lowest id. One thread per scenario, after the pin kernel: it UNDOES the elected
// pin (row mode and target back, limit type and switch count back) and re-derives the scenario's
// "changed" flag from what the pass still did. Before this the device pinned every candidate and a
// per-scenario DETECTOR routed the scenario to the CPU (1.9 kV wrong on the forced-PV network).
__global__ void rlForcedPvElectBatch(int nrl, int nScenarios, int n,
                                     const int* rlVrow, const char* rlRemote, const char* rlIslanded,
                                     const double* nomV, const int* pRow, const double* staticP, const int* idRank,
                                     char* limitType, char* switchCount, char* passSwitched, const char* savedMode,
                                     const double* savedTarget, int* mode, double* target,
                                     const char* rlActive, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!rlActive[s]) return;
    int remaining = 0;
    int cand = 0;
    int best = -1;
    double bestV = -1.0;
    double bestP = -1e300;
    int bestId = 0x7fffffff;
    for (int i = 0; i < nrl; i++) {
        size_t st = (size_t) s * nrl + i;
        if (rlIslanded != nullptr && rlIslanded[st]) continue;   // not a controller in this component
        bool remote = rlRemote != nullptr && rlRemote[i];
        int m = mode[(size_t) s * n + rlVrow[i]];
        if (passSwitched[st] == 1) {
            cand++;
            double v = nomV[i];
            double p = pRow[i] >= 0 ? target[(size_t) s * n + pRow[i]] : staticP[i];
            bool better = best < 0 || v > bestV
                    || (v == bestV && (p > bestP || (p == bestP && idRank[i] < bestId)));
            if (better) {
                best = i;
                bestV = v;
                bestP = p;
                bestId = idRank[i];
            }
        } else if (limitType[st] == 0 && (rlRemote != nullptr && rlRemote[i] == 2 ? m == 0
                                          : remote ? m == 1 : (m == 0 || m == 2))) {
            remaining++;                                     // still PV after this pass
        }
    }
    if (cand == 0 || remaining > 0 || best < 0) return;
    size_t st = (size_t) s * nrl + best;
    size_t idx = (size_t) s * n + rlVrow[best];
    mode[idx] = (rlRemote != nullptr && rlRemote[best] == 2) ? 0
              : (rlRemote != nullptr && rlRemote[best]) ? 1 : savedMode[st];
    target[idx] = savedTarget[st];
    limitType[st] = 0;
    switchCount[st]--;
    passSwitched[st] = 0;
    char any = 0;
    for (int i = 0; i < nrl && !any; i++) {
        any = passSwitched[(size_t) s * nrl + i] != 0;
    }
    switched[s] = any;
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
/** MonitoringVoltage (OLF MonitoringVoltageOuterLoop): a standby SVC whose monitored bus leaves its
 *  band switches to voltage control at the corresponding setpoint. One-way — once a controller has
 *  fired for a scenario it is never re-checked, exactly as OLF's activatedTargetV bookkeeping. */
__global__ void monitoringLoopBatch(int nmon, int nScenarios, int n,
                                    const int* monVRow, const double* monBand,
                                    const int* deltaPtr, const int* deltaRow,
                                    const int* deltaMode, const double* deltaTarget,
                                    const double* x, double* target, int* mode,
                                    double* activated, const char* rlActive, char* switched) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nmon * nScenarios) return;
    int c = k % nmon;
    int s = k / nmon;
    if (!rlActive[s]) return;
    size_t slot = (size_t) s * nmon + c;
    if (!isnan(activated[slot])) return;                     // already switched — a normal PV bus now
    double v = x[(size_t) s * n + monVRow[c]];
    double lowThr = monBand[c * 4];
    double highThr = monBand[c * 4 + 1];
    double lowTgt = monBand[c * 4 + 2];
    double highTgt = monBand[c * 4 + 3];
    double chosen;
    if (v > highThr) {
        chosen = highTgt;
    } else if (v < lowThr) {
        chosen = lowTgt;
    } else {
        return;
    }
    activated[slot] = chosen;
    for (int d = deltaPtr[c]; d < deltaPtr[c + 1]; d++) {
        int row = deltaRow[d];
        mode[(size_t) s * n + row] = deltaMode[d];
        target[(size_t) s * n + row] = row == monVRow[c] ? chosen : deltaTarget[d];
    }
    switched[s] = 1;
}

// OLF_GPU_DS_DEBUG: the kernel-level water-fill prints ([DS-GATE]/[DS-WF]/[DS-SAT]) follow ONE slot,
// OLF_GPU_DS_DEBUG_SLOT (default 0); returns slot+1, 0 = off.
static int dsDebugTraceSlot() {
    if (std::getenv("OLF_GPU_DS_DEBUG") == nullptr) return 0;
    const char* e = std::getenv("OLF_GPU_DS_DEBUG_SLOT");
    return (e != nullptr ? std::atoi(e) : 0) + 1;
}

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
                                   double slackMarginUp, double slackMarginDown,
                                   const char* dsIslanded, const double* dsFactor) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    if (!rlActive[s]) return;
    double absorbed = 0.0;
    double fsActive = 0.0;                                   // Σ f over the participants still in the main component
    int ov = dsOvBus != nullptr ? dsOvBus[s] : -1;
    for (int i = 0; i < nds; i++) {
        if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
            continue;                                        // in another component: not a participant here
        }
        fsActive += dsFactor[i];
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
    if (dsIslanded != nullptr && dsMode != 1) {
        // The factors were normalized over ALL participants (W), so rho = w_slack/W. Dropping the
        // stranded ones renormalizes over W' = W - Σ_stranded w, and w_slack/W' = rho/(Σ_active f + rho).
        // Leaving rho global is exactly what made the first attempt at this fail: the participants
        // were masked out of the distribution loops while the slack kept the share it had when they
        // were still in, so the pass retired with most of the mismatch undistributed.
        rho = fsActive + rho > 0 ? rho / (fsActive + rho) : 0.0;
    }
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
            if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
                continue;                                    // REMAINING_MARGIN needs no rescale: dropping
            }                                                // the stranded margins IS the renormalization
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
                                              const char* dsIslanded,
                                              int ngen, const int* dgBus, const double* dgFactor,
                                              const double* dgMaxDelta, const double* dgMinDelta,
                                              const double* dgMarginUp, const double* dgMarginDown,
                                              const int* busGenStart, const int* busGenEnd,
                                              double* genCum, char* genCumValid, char* genRemoved, double* genFactorWork,
                                              int nDsLostGen, const int* dsLostGen,
                                              double* target, char* switched, char* dsResidual, double* retainApplied, int dbgTrace) {
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
            if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
                continue;
            }
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
        double fsActive = 0.0;
        double fsAll = 0.0;
        for (int i = 0; i < nds; i++) {
            fsAll += dsFactor[i];
            if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
                continue;
            }
            fsActive += dsFactor[i];
        }
        // With every participant present this is Σf, bit-for-bit what it was. With some stranded it
        // is the survivors' share of the RENORMALIZED set, 1 - rho' = fsActive/(fsActive + rho) with
        // rho = 1 - Σf the slack's original share — the same formula computeSlackRetain uses for
        // rho', so the two halves of the split still add up to the whole mismatch. The gate below
        // then scales the tolerance by the share these participants really carry, which is what
        // OLF's gate tests on its surviving participant set.
        double rho = 1.0 - fsAll;
        fsGlobal = fsActive == fsAll
                ? fsActive
                : (fsActive + rho > 0 ? fsActive / (fsActive + rho) : 0.0);
    }
    if (dbgTrace && s == dbgTrace - 1) {
        printf("[DS-GATE] slackMismatch=%.9f retain=%.9f rem=%.9f fsGlobal=%.9f tol=%.9f slackTgt=%.9f fire=%d\n",
               qBuf[(size_t) s * n + slackPhiRow] - slackTgt, dsRetain[s], rem, fsGlobal, slackDistTol, slackTgt,
               (fsGlobal > 0.0 && fabs(rem) > fsGlobal * slackDistTol) ? 1 : 0);
    }
    if (fsGlobal <= 0.0 || fabs(rem) <= fsGlobal * slackDistTol) {
        return;
    }
    if (retainApplied != nullptr) {
        retainApplied[s] = dsRetain[s];                      // the slack's share of THIS distribution (write-back)
    }
    // OLF's ActivePowerDistribution.run RESETS every participant to its INITIAL target
    // (step.resetToInitialState) and adds back what that undid (previousStateInfo.previousMismatch()),
    // then re-derives the WHOLE distribution from the anchor. Accumulating each pass on top of the
    // previous one is not the same function once anyone saturates: the re-normalisation over the
    // still-unsaturated participants is taken against a different remaining amount. Ported here for
    // FIDELITY — it is not what closes the residual rte6515 gap (that is the reactive-limits loop),
    // but the incremental form was not what OLF does.
    double absorbedNow = 0.0;
    for (int i = 0; i < nds; i++) {
        if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
            continue;                                        // its row is frozen; it never absorbed anything
        }
        double base = dsBaseTargetP[i];
        if (dsAnchorShift != nullptr) { base += dsAnchorShift[(size_t) s * nds + i]; }
        if (ov == i) { base -= dsOvBaseDelta[s]; }
        size_t idx = (size_t) s * n + dsPhiRow[i];
        absorbedNow += target[idx] - base;
        target[idx] = base;                                 // reset to the anchor
    }
    rem = fsGlobal * ((qBuf[(size_t) s * n + slackPhiRow] - slackTgt) + absorbedNow);
    if (dbgTrace && s == dbgTrace - 1) {
        printf("[DS-WF] entry slackMismatch=%.9f absorbedNow=%.9f fsGlobal=%.9f rem=%.9f nds=%d ngen=%d anchorShift=%d\n",
               qBuf[(size_t) s * n + slackPhiRow] - slackTgt, absorbedNow, fsGlobal, rem, nds, ngen,
               dsAnchorShift != nullptr ? 1 : 0);
    }
    bool changed = false;
    // PER-GENERATOR water-fill, the shape of OLF's GenerationActivePowerDistributionStep.run: factors
    // renormalized over the elements still in the list at each iteration start, one pass using the
    // mismatch as of that start, each element clamped at its OWN limit and REMOVED when it strictly
    // overshoots. A bus's row is then the sum of its generators' cumulative deltas.
    //
    // Used whenever the pack carries generators and THIS scenario is not re-anchored. A pair leg's
    // dsAnchorShift is a BUS-level displacement (the parent contingency's absorption, carried so the
    // leg does not undo it) and nothing in the pack says how to split it across a bus's generators, so
    // a re-anchored scenario keeps the aggregate loop below — unchanged, and no less faithful than it
    // was. The test is on the VALUE, not on the pointer: an orchestrator re-solve passes the carry
    // arrays for every scenario, all zeros for a plain N-1, and taking that as "re-anchored" quietly
    // sent every post-orchestrator pass down the aggregate path (measured: the first two DS rounds of
    // LINE-3320-350 matched the CPU digit for digit, the third did not).
    bool anchored = false;
    if (dsAnchorShift != nullptr) {
        for (int i = 0; i < nds; i++) {
            if (dsAnchorShift[(size_t) s * nds + i] != 0.0) {
                anchored = true;
                break;
            }
        }
    }
    bool perGen = ngen > 0 && !anchored;
    if (genCumValid != nullptr) {
        genCumValid[s] = perGen ? 1 : 0;
    }
    if (perGen) {
        // The outaged bus's SURVIVORS join the water-fill like any others, with the generators this
        // scenario removes marked out by identity below. The gate is the BUS, not the lost unit: a
        // lost generator outside the DS pack never participated and has nothing to remove, but its
        // participating siblings still need their own P — leaving them on the aggregate entry is what
        // keeps their genCum at zero and sends the curve channel back to the share model.
        bool ovPerGen = ov >= 0;
        for (int g = 0; g < ngen; g++) {                     // fresh list, fresh cumulative deltas
            genCum[(size_t) s * ngen + g] = 0.0;
            bool lost = false;
            for (int k = 0; k < nDsLostGen; k++) {
                if (dsLostGen[2 * k] == s && dsLostGen[2 * k + 1] == g) {
                    lost = true;
                    break;
                }
            }
            genRemoved[(size_t) s * ngen + g] = lost ? 1 : 0;
            genFactorWork[(size_t) s * ngen + g] = lost ? 0.0
                    : (dsMode == 1 ? (dirUp ? dgMarginUp[g] : dgMarginDown[g]) : dgFactor[g]);
        }
        for (int iter = 0; iter < ngen && fabs(rem) > residueEps; iter++) {
            double fsumActive = 0.0;                         // normalizeParticipationFactors, over the remaining
            for (int g = 0; g < ngen; g++) {
                int i = dgBus[g];
                if (genRemoved[(size_t) s * ngen + g] || (ov == i && !ovPerGen)
                        || (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i])) {
                    continue;
                }
                fsumActive += genFactorWork[(size_t) s * ngen + g];
            }
            // The bus that lost a generator stays on the AGGREGATE treatment: the override channel
            // carries {factor, base, headroom, footroom} deltas for the bus, not the identity of the
            // generator, so a per-generator entry cannot express it. Its share is added to the same
            // denominator, so the normalization is the one the old loop used.
            double ovF = 0.0;
            double ovBase = 0.0;
            double ovMaxD = 0.0;
            double ovMinD = 0.0;
            bool ovActive = false;
            if (ov >= 0 && !ovPerGen && (dsIslanded == nullptr || !dsIslanded[(size_t) s * nds + ov])) {
                ovBase = dsBaseTargetP[ov] - dsOvBaseDelta[s];
                ovMaxD = dsMaxDelta[ov] - dsOvMaxDelta[s];
                ovMinD = dsMinDelta[ov] - dsOvMinDelta[s];
                ovF = dsMode == 1 ? fmax(0.0, dirUp ? dsMarginUp[ov] : dsMarginDown[ov])
                                  : dsFactor[ov] - dsOvFactorDelta[s];
                double ovCum = target[(size_t) s * n + dsPhiRow[ov]] - ovBase;
                ovActive = ovF > 0 && !((rem > 0) ? (ovCum >= ovMaxD) : (ovCum <= ovMinD));
                if (ovActive) {
                    fsumActive += ovF;
                }
            }
            if (fsumActive <= 0.0) {
                break;                                       // every participant removed — residual stays at the slack
            }
            double done = 0.0;
            for (int g = 0; g < ngen; g++) {
                int i = dgBus[g];
                if (genRemoved[(size_t) s * ngen + g] || (ov == i && !ovPerGen)
                        || (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i])) {
                    continue;
                }
                // normalizeParticipationFactors: divide the STORED factor, then use it.
                double f = genFactorWork[(size_t) s * ngen + g] / fsumActive;
                genFactorWork[(size_t) s * ngen + g] = f;
                double cum = genCum[(size_t) s * ngen + g];
                double newCum = cum + rem * f;
                // OLF removes an element only on a STRICT overshoot: one that lands exactly on its
                // limit stays in the list another iteration, where it takes weight and moves nothing.
                if (rem > 0 && newCum > dgMaxDelta[g]) {
                    newCum = dgMaxDelta[g];
                    genRemoved[(size_t) s * ngen + g] = 1;
                } else if (rem < 0 && newCum < dgMinDelta[g]) {
                    newCum = dgMinDelta[g];
                    genRemoved[(size_t) s * ngen + g] = 1;
                }
                if (newCum != cum) {
                    genCum[(size_t) s * ngen + g] = newCum;
                    done += newCum - cum;
                    changed = true;
                }
                if (dbgTrace && s == dbgTrace - 1 && genRemoved[(size_t) s * ngen + g]
                        && (newCum == dgMaxDelta[g] || newCum == dgMinDelta[g])) {
                    printf("[DS-SAT] iter=%d gen=%d newCum=%.9f maxD=%.9f minD=%.9f\n",
                           iter, g, newCum, dgMaxDelta[g], dgMinDelta[g]);
                }
            }
            if (ovActive) {
                size_t idx = (size_t) s * n + dsPhiRow[ov];
                double cum = target[idx] - ovBase;
                double newCum = fmin(fmax(cum + rem * (ovF / fsumActive), ovMinD), ovMaxD);
                if (newCum != cum) {
                    target[idx] = ovBase + newCum;
                    done += newCum - cum;
                    changed = true;
                }
            }
            // The bus rows: anchor + Σ over the bus's generators, whose slice is contiguous.
            for (int i = 0; i < nds; i++) {
                if ((ov == i && !ovPerGen) || (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i])) {
                    continue;                                // ov keeps its own value; a frozen row is not written
                }
                double sum = 0.0;
                for (int g = busGenStart[i]; g < busGenEnd[i]; g++) {
                    sum += genCum[(size_t) s * ngen + g];
                }
                // The outage's own move of the row target is not slack power: it shifts the anchor.
                target[(size_t) s * n + dsPhiRow[i]] =
                        dsBaseTargetP[i] - (ov == i ? dsOvBaseDelta[s] : 0.0) + sum;
            }
            if (dbgTrace && s == dbgTrace - 1) {
                printf("[DS-WF] iter=%d fsumActive=%.9f rem=%.9f done=%.9f (per-generator)\n",
                       iter, fsumActive, rem, done);
            }
            rem -= done;
        }
    } else {
    for (int iter = 0; iter < nds && fabs(rem) > residueEps; iter++) {
        double fsumActive = 0.0;                            // re-normalize factors over the still-unsaturated buses
        for (int i = 0; i < nds; i++) {
            if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
                continue;
            }
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
            if (dsIslanded != nullptr && dsIslanded[(size_t) s * nds + i]) {
                continue;                                    // writing here would land on a frozen row
            }
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
        if (dbgTrace && s == dbgTrace - 1) {
            printf("[DS-WF] iter=%d fsumActive=%.9f rem=%.9f done=%.9f\n", iter, fsumActive, rem, done);
        }
        rem -= done;
    }
    }
    if (changed) {
        switched[s] = 1;
    }
    // OLF decides the distribution FAILED on ActivePowerDistribution.P_RESIDUE_EPS (1e-5 pu, 1 kW),
    // not on slackBusPMaxMismatch: handleDistributionFailureBehavior is entered whenever the water-fill
    // left more than P_RESIDUE_EPS unplaced. Flagging on the (much looser) entry band instead let a
    // saturated scenario with, say, 0.3 MW still unplaced come back CONVERGED where the CPU reports
    // FAILED. The flag only routes the scenario to the CPU fallback, so the tighter test can cost
    // batching share but can never make the device answer where OLF refuses to.
    if (fabs(rem) > residueEps) {
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
// A PINNED pilot that is one of its own controllers (rlRemote 2) while a member of the same group
// (same compare row = the pilot's V) still controls: OLF then keeps the pilot's V identity AND
// activates the pilot's own Q balance, hosting it on the reference member's row - a relocation this
// group has no table for on the device. Flag the scenario; the host rejects it to the CPU replay.
// The other mixed state (members pinned, pilot controlling) is an ordinary layout: pilot identity,
// members at their own Q. One thread per (scenario, class-2 entry).
__global__ void rlPilotSelfMixedGate(int nrl, int nScenarios, const char* rlRemote, const int* rlCmpVrow,
                                     const char* limitType, char* bad) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nrl * nScenarios) return;
    int i = t % nrl;
    int s = t / nrl;
    if (rlRemote[i] != 2 || limitType[(size_t) s * nrl + i] == 0) return;   // only a PINNED pilot
    for (int j = 0; j < nrl; j++) {                          // ... with a member still controlling
        if (j != i && rlCmpVrow[j] == rlCmpVrow[i] && rlRemote[j] != 0
                && limitType[(size_t) s * nrl + j] == 0) {
            bad[s] = 1;
            return;
        }
    }
}

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
        target[idx] = (lt == 1 ? rlMinQ[li] : rlMaxQ[li]) - rlLoadQ[li];
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
// OLF_GPU_WORST_ROW=1 (with OLF_GPU_BATCH_TRACE): name the row carrying the largest residual at a
// solve's start. A norm says HOW FAR the state is; only the row says WHY — an identity row sitting at
// an exact 1.0 is a different animal from a power row at 1.0.
__global__ void worstRowBatch(int n, const double* F, const int* mode, const double* target,
                              const double* x, int s) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int bi = -1;
    double bv = -1.0;
    for (int i = 0; i < n; i++) {
        double v = fabs(F[(size_t) s * n + i]);
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    if (bi >= 0) {
        printf("[WORST_ROW] s=%d row=%d |F|=%.12f mode=%d target=%.9f x=%.9f\n",
               s, bi, bv, mode[(size_t) s * n + bi], target[(size_t) s * n + bi], x[(size_t) s * n + bi]);
    }
}

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
    DeviceBuf dTargetBase;                                   // the base targets (n), before tiling/overrides
    DeviceBuf dHazSnap, dHazBSnap, dHazChoice;               // zero-diagonal repair scratch (lazy)
    int hazCap = 0;
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
    DeviceBuf dRlLoadQDyn;                                       // per (scenario, controller) — shed load Q
    int nDqInact = 0;                                            // {slot, dqMember} pairs: a controller
    DeviceBuf dDqInact;                                          // that stops controlling in ONE scenario
    // Reactive capability CURVES at runtime P (setBatchRlCurves): per participating curve gen —
    // cvIdx {rlIdx, pRow, ptsOff, npts, dsGenSlot}, cvIn {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu},
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
    int allPvPinnable = 0;                                 // every non-monitored RL entry has finite limits — the
                                                           // ONLY networks where OLF's global "strongest stays PV"
                                                           // rule (remainingPvBusCount 0) is reachable
    DeviceBuf dRowSlope;                                   // per row: SLOPE-SVC slope (mode-2 V rows), 0 elsewhere
    int nRlRemote = 0;                                     // remote-member RL entries present (plan-B1/B3)
    int nRlPilotSelf = 0;                                  // rlRemote == 2 entries (a pilot that is its own controller)
    DeviceBuf dRlMixedBad;                                 // nS chars: rlPilotSelfMixedGate flags
    DeviceBuf dRlRemote;                                   // per-entry remote kind (1 = remote group member)
    DeviceBuf dRlBaseLimitType;                            // OLF's base pin kind per entry (0..4); see hasRlBaseLt
    int hasRlBaseLt = 0;
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
    DeviceBuf dDqMemberTargetQ;                            // per member: its UNPINNED targetQ (generationTargetQ -
                                                           // loadTargetQ), the value AcTargetVector reads for every
                                                           // member the RL loop has not frozen
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
    // OLF's GLOBAL forced-PV election ("all PV buses should switch — the strongest stays PV"): what each
    // entry did THIS pass (1 = pinned PV→PQ, 2 = switched back, 3 = limit re-freeze), the row mode a pin
    // replaced (to undo the elected one), and the static election keys (controlled bus nominal V, the
    // bus's P row for its LIVE target P, the static target P when it has no row, the id rank).
    DeviceBuf dRlPassSwitched, dRlSavedMode;
    DeviceBuf dRlElectNomV, dRlElectPRow, dRlElectStaticP, dRlElectIdRank;
    int hasRlElection = 0;
    // voltageRemoteControlRobustMode (OLF default TRUE): a REMOTE voltage controller whose OWN bus
    // voltage leaves the realistic band is reseeded at 1 pu, and — when its Q is still inside its
    // limits — frozen at its initial generation target Q with a MIN/MAX_REALISTIC_V limit type.
    DeviceBuf dRlInitTgtQ;                                 // per entry: Σ generator targetQ (pu)
    int rlRobust = 0;
    double rlVLoRobust = 0.0, rlVHiRobust = 0.0;           // minRealisticV·1.02 and maxRealisticV/1.02
    // AcloadFlowEngine's realistic-voltage check, run on the INTERMEDIATE states of the driver's
    // sweep: per row, 1 = a non-fictitious bus's BUS_V at or above minNominalVoltageRealisticVoltageCheck.
    DeviceBuf dRealisticRow, dRealisticGate, dRealisticBad;
    int realisticCheck = 0;
    double realisticMinV = 0.0, realisticMaxV = 0.0;
    // AcloadFlowEngine DEFERS the check until the last outer loop that can FIX an unrealistic state
    // has run (voltageRemoteControlRobustMode; ReactiveLimits and the discrete TransformerVoltageControl
    // are the two loops that declare canFixUnrealisticState). -1 = no such loop, check every solve.
    // 0 = ReactiveLimits is the last one, 1 = the discrete TransformerVoltageControl.
    int realisticFixingLoop = -1;
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
    DeviceBuf dDsRetainApplied;                              // per scenario: dsRetain at the last check that FIRED,
                                                             // NaN while no distribution happened in this solve
    // Per-GENERATOR participants (setBatchDsGenerators). OLF's water-fill participates and saturates
    // per generator — ParticipatingElement wraps an LfGenerator and step.run clamps each at its OWN
    // limits and REMOVES it — so a bus-level aggregate only reproduces OLF while nothing saturates.
    // The bus arrays keep owning the equation row and the anchor; these own the factors and limits.
    int ngen = 0;
    DeviceBuf dgBus, dgFactor, dgMaxDelta, dgMinDelta, dgMarginUp, dgMarginDown;
    DeviceBuf dBusGenStart, dBusGenEnd;
    // Per (scenario, generator): its cumulative delta from the anchor, and whether the water-fill has
    // REMOVED it. Removal is not the same as "sitting at its bound": OLF removes an element only when
    // it strictly OVERSHOOTS, so one that lands exactly on its limit stays in the normalization for
    // another iteration — where it absorbs weight and moves nothing.
    // MonitoringVoltage (OLF order 6): the standby-SVC automaton, as a DRIVER PHASE. Per controller:
    // the monitored V row, its band and the two setpoints; plus a CSR list of (row, mode, target) the
    // activation writes. dMonActivated holds the chosen target per (controller, scenario), NaN until
    // it fires — OLF never re-checks a controller it has already switched.
    // DRIVER PAUSE/RESUME. Some of OLF's outer loops cannot run on the device without RE-DERIVING
    // what OLF already computes in Java (SecondaryVoltageControl's sensitivity observables; the
    // discrete tap/section tables of the transformer and shunt loops). Rather than port that maths —
    // a wrong target converges cleanly and nothing flags it — the driver PAUSES at their slot in
    // OLF's order, hands control back, and RESUMES where it stopped. The per-solve device state
    // (RL pins, monitoring activations, DS absorbed totals, HVDC statuses, x/targets/modes) simply
    // stays in the context across the pause; only the host-side driver locals need saving.
    int hostLoopMask = 0;                                    // bitmask of enabled host-owned loops
                                                             // bit 0 = LOOP_HOST_SVC, 1 = PHASE, 2 = TVC, 3 = TRPC, 4 = SHUNT
    int driverResume = 0;                                    // this call continues a paused driver
    int driverPaused = 0;                                    // the last call stopped at a LOOP_HOST_*
    int pausedLoopKind = 0;                                  // WHICH host loop the driver stopped at
    std::vector<int> pausedStatus;                           // per-scenario solve status at the pause: the resumed
                                                             // call's checks (rlActive = converged) need it back
    int startLoopKind = -1;                                  // a RESTARTED sequence resumes its sweep AT this
                                                             // loop (OLF re-checks the loop that went UNSTABLE
                                                             // after the re-solve, before restarting the sweep)
    int pausedLoopIdx = 0;
    int pausedSeqChanged = 0;
    std::vector<int> pausedProdIters;
    std::vector<char> pausedLive, pausedActive;
    std::vector<char> hostChanged;                           // per scenario: the host loop was UNSTABLE
    int nmon = 0;
    DeviceBuf dMonVRow, dMonBand, dMonDeltaPtr, dMonDeltaRow, dMonDeltaMode, dMonDeltaTarget, dMonActivated;
    // INCREMENTAL TRANSFORMER VOLTAGE CONTROL — in-kernel outer loop (LOOP_TVC_KERNEL).
    // Per-controller static data uploaded once; per-(scenario, controller) mutable state allocated
    // at first use. Tap tables are CSR-packed: tapData[tapPtr[c]..tapPtr[c+1]) holds nTaps rows of
    // 8 doubles [y, g1, g2, b1, b2, ksi, r1, a1] for controller c.
    int nTvc = 0;                                            // number of incremental TVC controllers
    DeviceBuf dTvcRhoRow, dTvcVRow;                          // [nTvc] int: rho identity row, controlled bus V row
    DeviceBuf dTvcTargetV, dTvcHalfDb;                       // [nTvc] double: target voltage, half deadband
    DeviceBuf dTvcElemIndex;                                 // [nTvc] int: element index for disabled-mask check
    DeviceBuf dTvcTapPtr;                                    // [nTvc+1] int: CSR pointer into flat tap table
    DeviceBuf dTvcTapData;                                   // [totalTaps * 8] double: flat tap rows
    DeviceBuf dTvcBaseTap;                                   // [nTvc] int: base (extraction-time) tap index
    // Per-(scenario, controller) mutable state:
    DeviceBuf dTvcCurTap;                                    // [nS * nTvc] int: current tap position
    DeviceBuf dTvcRatchetDir, dTvcRatchetCount, dTvcRatchetAllowed; // [nS * nTvc] int: ratchet state
    // Per-controller group structure (CSR): controllers sharing a controlled bus are grouped for
    // the one-tap-at-a-time round-robin (OLF's adjustWithSeveralControllers).
    DeviceBuf dTvcGroupPtr, dTvcGroupCtrl;                   // CSR: groupPtr[nGroups+1], groupCtrl[nTvc]
    int nTvcGroups = 0;
    int tvcMaxTapShift = 1;                                  // OLF incrementalTransformerRatioTapControlOuterLoopMaxTapShift
    // Per-pass scratch and the previous tap (to apply only the pairs that moved this pass).
    DeviceBuf dTvcOutOfDb, dTvcDiffV, dTvcSensi, dTvcPrevTap;
    // Host mirrors of the static tables, for the tap-override rebuild and the rho-row lookup.
    std::vector<int> hTvcRhoRow, hTvcElemIndex, hTvcBaseTap, hTvcTapPtr;
    std::vector<double> hTvcTapData;
    // Host copy of the CALLER's tap overrides (contingency tap actions) as uploaded at solve entry:
    // the TVC loop's moves are merged IN FRONT of them (findTapOverride takes the first match).
    std::vector<int> hR1OvBaseSlotElem;
    std::vector<double> hR1OvBaseVal;
    // INCREMENTAL SHUNT VOLTAGE CONTROL — in-kernel outer loop (LOOP_SHUNT_KERNEL). Controllers are
    // grouped per controlled bus (OLF adjustB round-robin, |b|-descending); each controller shunt has
    // sub-controllers (OLF LfShunt.Controller) with their own section B/G tables and ratchet.
    int nShc = 0, nShcGroups = 0, nShcSubs = 0, shcMaxSectionShift = 3;
    DeviceBuf dShcBRow, dShcVRow, dShcTargetV, dShcHalfDb, dShcElemAbs, dShcShOvElem;   // per controller
    DeviceBuf dShcGroupPtr, dShcGroupCtrl;                   // CSR: group -> controllers (sorted)
    DeviceBuf dShcSubPtr;                                    // CSR: controller -> subs
    DeviceBuf dShcSecPtr, dShcSecB, dShcSecG, dShcSubMin, dShcSubBase;   // per sub: section tables
    DeviceBuf dShcPos, dShcRatDir, dShcRatCount, dShcRatAllowed, dShcShift;   // per (scenario, sub)
    DeviceBuf dShcSensi, dShcBaseB, dShcAppliedDb;           // per (scenario, controller)
    DeviceBuf dShcOutOfDb, dShcDiffV;                        // per (scenario, group)
    std::vector<int> hShcBRow, hShcShOvElem, hShcSubPtr, hShcSecPtr, hShcSubBase;
    std::vector<double> hShcSecB, hShcSecG;
    // FLOW-OBSERVED incremental loops (LOOP_PHASE_KERNEL / LOOP_TRPC_KERNEL): a rho (RPC) or a1 (phase)
    // tap driving a CONTROLLED branch's Q, P or I. kind 0 = RPC (rho -> Q, closest-r1), 1 = phase
    // CONTROLLER (a1 -> P, closest-a1), 2 = phase LIMITER (a1 -> I, first-above-a1, one-sided).
    int nFlc = 0, nFlcPhase = 0, nFlcRpc = 0;
    DeviceBuf dFlcKind, dFlcPinRow, dFlcPinCol, dFlcSide, dFlcTargetValue, dFlcHalfDb, dFlcElem, dFlcCtlPack,
              dFlcTapPtr, dFlcTapData, dFlcBaseTap, dFlcMaxShift;
    DeviceBuf dFlcCurTap, dFlcPrevTap, dFlcRatDir, dFlcRatCount, dFlcRatAllowed, dFlcSensi, dFlcDiff, dFlcOutOfDb;
    std::vector<int> hFlcKind, hFlcPinRow, hFlcElem, hFlcTapPtr, hFlcBaseTap;
    std::vector<double> hFlcTapData;
    // CONTINUOUS phase control (OLF PhaseControlOuterLoop, CONTINUOUS_WITH_DISCRETISATION), sharing the
    // phase slot: kind 0 = CONTROLLER (a1 solved free under row mode 3 for the first solve, then switched
    // off and rounded to the closest tap at the loop's first check), kind 1 = LIMITER (from the second
    // check on, one tap per check against the sign of dI/da1 while I exceeds the limit — local only).
    int nCph = 0;
    DeviceBuf dCphKind, dCphA1Row, dCphSide, dCphLimit, dCphElem, dCphTapPtr, dCphTapData, dCphBaseTap;
    DeviceBuf dCphTap, dCphPrevTap, dCphIter;                // per (scenario, controller) tap; per scenario check count
    std::vector<int> hCphElem, hCphTapPtr, hCphBaseTap;
    std::vector<double> hCphTapData;
    std::vector<int> hShOvBaseSlotElem;                      // the caller's shunt overrides at solve entry
    std::vector<double> hShOvBaseGb;
    // DISCRETE (non-incremental) TRANSFORMER VOLTAGE CONTROL — LOOP_TVD_KERNEL (OLF order 8): OLF's
    // TransformerVoltageControlOuterLoop (tvdMode 0, AFTER_GENERATOR: INITIAL/CONTROL/COMPLETE) or
    // SimpleTransformerVoltageControlOuterLoop (tvdMode 1, WITH_GENERATOR). One group per controlled bus
    // (CSR over members); the group's ON/OFF activations are the toggle-diffed row deltas (CSR).
    int nTvd = 0, nTvdGroups = 0, nTvdDelta = 0, tvdMode = 0, tvdEntryOn = 0;
    DeviceBuf dTvdGrpPtr, dTvdGrpVRow, dTvdGrpTargetV, dTvdGrpHalfDb, dTvdGrpUseInit;
    DeviceBuf dTvdDeltaPtr, dTvdDeltaRow, dTvdDeltaModeOn, dTvdDeltaModeOff, dTvdDeltaTargetOn, dTvdDeltaTargetOff;
    DeviceBuf dTvdMemRhoRow, dTvdMemElem, dTvdMemBaseTap, dTvdMemTapPtr, dTvdMemMin, dTvdMemMax, dTvdMemInit, dTvdTapData;
    DeviceBuf dTvdScnStep, dTvdGrpStep, dTvdGrpShMin, dTvdGrpShMax, dTvdGrpShInit;   // per scenario / per (scenario, group)
    DeviceBuf dTvdMemState, dTvdMemTap, dTvdMemRoundFrom;                          // per (scenario, member)
    DeviceBuf dTvdSaved, dTvdReject;                                               // per (scenario, delta row) / per scenario
    // SECONDARY VOLTAGE CONTROL — LOOP_SVC_KERNEL (OLF order 4). Zones of controller buses regulating a
    // pilot point: run entirely on the device, the small dense zone system included.
    int nSvc = 0, nSvcZones = 0;
    DeviceBuf dSvcCtldVRow, dSvcBusVRow, dSvcRlIdx, dSvcBaseTargetV, dSvcMinQ, dSvcMaxQ, dSvcLoadQ;
    DeviceBuf dSvcQPtr, dSvcQKind, dSvcQElem, dSvcQBRow;
    DeviceBuf dSvcZonePtr, dSvcZoneCtl, dSvcZonePilotRow, dSvcZoneTargetV;
    DeviceBuf dSvcTargetV, dSvcK0, dSvcJk, dSvcJvpp, dSvcEnabled, dSvcNeed, dSvcReject, dSvcRlVt;
    // Batched dense LU workspaces: the zone systems live in DEVICE memory, one N*N column-major matrix
    // and one RHS per scenario, plus the pointer/pivot/info arrays cuBLAS's batched API takes.
    DeviceBuf dSvcMat, dSvcRhs, dSvcMatPtr, dSvcRhsPtr, dSvcPiv, dSvcInfo, dSvcCtlZone;
    cublasHandle_t svcBlas = nullptr;
    std::vector<int> hSvcCtldVRow;
    std::vector<double> hSvcBaseTargetV;
    double svcMinPlausibleV = 0.8, svcMaxPlausibleV = 1.2;

    std::vector<int> hTvdMemElem, hTvdMemBaseTap, hTvdMemTapPtr;
    std::vector<double> hTvdTapData;
    // GeneratorVoltageControlManager (AFTER_GENERATOR): generator VC controller buses under the max
    // transformer-controlled nominal voltage are FROZEN at their current Q while the transformers act
    // (freezeGenerationTargetQAndDisableGeneratorVoltageControl(q.eval()) — OLF's generation target is
    // the NET injection, so the row target is eval − loadQ), re-enabled at COMPLETE; plus
    // fixTransformerVoltageControls' PV-bus lists per member (the not-controlled side's component).
    int nTvdFz = 0;
    DeviceBuf dTvdFzVRow, dTvdFzLoadQ;                       // per candidate
    DeviceBuf dTvdFzFrozen, dTvdFzSavedMode, dTvdFzSavedTarget;   // per (scenario, candidate)
    DeviceBuf dTvdFzRlEntry;                                 // per candidate: RL entry of a REMOTE one, else -1
    DeviceBuf dTvdFixPvRow;                                  // the fix universe: every gen-VC bus's V row
    DeviceBuf dTvdFixMinMask, dTvdFixMaxMask;                // per (scenario, member): fixWords ints each
    int tvdFixWords = 0;
    bool tvdFixTables = false;
    // DISCRETE SHUNT VOLTAGE CONTROL — LOOP_SVD_KERNEL (OLF order 10): ShuntVoltageControlOuterLoop
    // (WITH_GENERATOR): control ON for the first solve, LfShuntImpl.dispatchB at the first check.
    int nSvd = 0, nSvdGroups = 0, nSvdDelta = 0, nSvdSubs = 0, svdEntryOn = 0;
    DeviceBuf dSvdGrpPtr, dSvdDeltaPtr, dSvdDeltaRow, dSvdDeltaModeOn, dSvdDeltaModeOff, dSvdDeltaTargetOn, dSvdDeltaTargetOff;
    DeviceBuf dSvdMemBRow, dSvdMemElem, dSvdMemSubPtr, dSvdSecPtr, dSvdSecB, dSvdSubMin, dSvdSubBase;
    DeviceBuf dSvdGrpStep, dSvdMemState, dSvdMemContB, dSvdSubPos, dSvdSaved;

    DeviceBuf dGenCum, dGenRemoved;
    DeviceBuf dFrozenRow;                                  // (scenario, row) pairs: rows an islanding
    int nFrozenRow = 0;                                    // outage froze to a flat identity state
    DeviceBuf dDsLostGen;                                  // (scenario, DS generator slot) pairs: the
    int nDsLostGen = 0;                                    // generators an outage removes, by identity
    DeviceBuf dGenCumValid;                                // per scenario: the PER-GENERATOR water-fill ran, so
                                                           // dGenCum holds each generator's own movement. The
                                                           // anchored path distributes per BUS and leaves it zero,
                                                           // which a reader cannot tell from "nothing moved".

    // The WORKING factors, normalized IN PLACE each iteration exactly as OLF's
    // normalizeParticipationFactors mutates ParticipatingElement.factor: by iteration k a survivor's
    // factor has been divided k times. f/Sigma_survivors(raw) is the same number algebraically and a
    // different one in floating point, which is enough to flip a clamp decision at a knife-edge.
    DeviceBuf dGenFactorWork;
    // Per (scenario, participant) chars, 1 = this participant is in ANOTHER synchronous component in
    // this scenario. OLF distributes over getSynchronousNetworks().getFirst(), so an islanded
    // participant is simply not a participant: its share must not be computed, and — decisively — the
    // set it is renormalized against must shrink with it. Empty = every participant participates.
    int nDsIslanded = 0;
    DeviceBuf dDsIslanded;
    // Same idea for the reactive-limits pass: per (scenario, controller), 1 = its bus is in another
    // synchronous component in this scenario, so the loop must not look at it.
    int nRlIslanded = 0;
    DeviceBuf dRlIslanded;
    int nDqTqOv = 0;                                       // per-scenario member targetQ overrides
    DeviceBuf dDqTqOvKey;
    DeviceBuf dDqTqOvVal;
    int nPilotFlip = 0;                                    // pilot-row flip entries (all-survivors-pinned)
    DeviceBuf dPfSlot;
    DeviceBuf dPfRow;
    DeviceBuf dPfQTarget;
    DeviceBuf dPfOff;
    DeviceBuf dPfCount;
    DeviceBuf dPfRlPool;
    DeviceBuf dPfFlipped;
    DeviceBuf dPfSavedVT;
    // HVDC AC-emulation outer loops (FreezingHvdcACEmulation + AcHvdcAcEmulationLimits) as DRIVER
    // PHASES rather than host orchestrators. OLF runs both BETWEEN DistributedSlack and
    // ReactiveLimits; run from the host after the whole solve they land after RL instead, which is
    // what put the device's last DS check on a pre-pin state.
    int nhv = 0;                                             // AC-emulation HVDCs in the loop tables
    int hvLimitsActive = 0;                                  // AcHvdcAcEmulationLimits configured
    DeviceBuf dHvElem, dHvBus1Row, dHvBus2Row;               // per hvdc
    DeviceBuf dHvIn;                                         // per hvdc x9: p0 k pMax12 pMax21 loss1 loss2 r baseD1 baseD2
    DeviceBuf dHvCtgOut;                                     // per (scenario, hvdc): outaged BY the contingency
    DeviceBuf dHvStatus;                                     // per (scenario, hvdc): LINEAR/SAT_1_2/SAT_2_1/FROZEN
    DeviceBuf dHvAppliedD1, dHvAppliedD2;                    // the converter-row delta currently APPLIED
    DeviceBuf dDsResidual;                                   // nS chars: waterfill left an out-of-tolerance residual                                   // per scenario (nS): frozen slack-retained mismatch share
    bool dsSlackOnly = false;                                // DS enabled with the SLACK generator as the only participant:
    DeviceBuf dDsSlackAbsorbed;                              // nS doubles — mismatch absorbed into the slack's own target so far
                                                             // (OLF distributes into the slack gen and re-solves; no flow change,
                                                             // but the extra polish run must happen to match the CPU trajectory)
    CudssCtx ds;
    bool analyzed = false;
    bool factored = false;                                   // first FACTORIZATION done (mode 2: then REFACTORIZE)
    int mode = 0;                                            // 0 = per-scenario loop, 1 = batched API, 2 = block-diagonal, 3 = UBATCH, 6 = per-scenario loop, robust (pivoted) config
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
        if (svcBlas != nullptr) {
            cublasDestroy(svcBlas);
            svcBlas = nullptr;
        }
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
        ctx->allPvPinnable = 1;
        for (int i = 0; i < nrl; i++) {
            if (!rlMonitored[i] && !((rlMinQ[i] > -1e7 || rlMaxQ[i] < 1e7) && rlMinQ[i] < rlMaxQ[i])) {
                ctx->allPvPinnable = 0;                      // an unlimited PV bus: the rule never fires
            }
        }
        ctx->dRlSavedTarget.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(double));
        ctx->dRlPassSwitched.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(char));
        ctx->dRlSavedMode.alloc(static_cast<size_t>(nrl) * nScenarios * sizeof(char));
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
        ctx->dDsRetainApplied.alloc(static_cast<size_t>(nScenarios) * sizeof(double));
    }

    // ---- cuDSS handle/config/data (shared by both solve paths) ----
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    CUDSS_CHECK(cudssCreate(&ctx->ds.handle));
    CUDSS_CHECK(cudssSetStream(ctx->ds.handle, ctx->stream));
    CUDSS_CHECK(cudssConfigCreate(&ctx->ds.config));
    int ir = irSteps(ctx->st->nzi > 0);
    CUDSS_CHECK(cudssConfigSet(ctx->ds.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)));
    if (mode == 6) {
        applyRobustSingleMatrixConfig(ctx->ds.config);       // its own ordering: never set the ordering twice
    } else {
        applyReorderAlg(ctx->ds.config, ctx->st->nzi > 0);
    }
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
    ctx->hShOvBaseSlotElem.assign(shOvSlotElem, shOvSlotElem + (size_t) nShOv * 2);
    ctx->hShOvBaseGb.assign(shOvGb, shOvGb + (size_t) nShOv * 2);
    if (nShOv > 0 || ctx->nShc > 0) {
        // room for the in-kernel shunt loop to append one (dG, dB) per (scenario, controller)
        size_t capEntries = std::max<size_t>(static_cast<size_t>(nShOv) + (ctx->nShc > 0 ? static_cast<size_t>(nS) * ctx->nShc : 0), 1);
        ctx->dShOvSlotElem.ensure(capEntries * 2 * sizeof(int));
        ctx->dShOvGb.ensure(capEntries * 2 * sizeof(double));
        if (nShOv > 0) {
            CUDA_CHECK(cudaMemcpy(ctx->dShOvSlotElem.p, shOvSlotElem, static_cast<size_t>(nShOv) * 2 * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dShOvGb.p, shOvGb, static_cast<size_t>(nShOv) * 2 * sizeof(double), cudaMemcpyHostToDevice));
        }
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
    ctx->hR1OvBaseSlotElem.assign(r1OvSlotElem, r1OvSlotElem + (size_t) nR1Ov * 2);
    ctx->hR1OvBaseVal.assign(r1OvVal, r1OvVal + (size_t) nR1Ov * 8);
    if (nR1Ov > 0 || ctx->nTvc > 0 || ctx->nFlc > 0 || ctx->nCph > 0) {
        // Room for the in-kernel TVC loop to merge one moved tap per (scenario, controller) in
        // front of the caller's list without reallocating mid-solve.
        size_t capEntries = static_cast<size_t>(nR1Ov) + static_cast<size_t>(nS) * (ctx->nTvc + ctx->nFlc + ctx->nCph + ctx->nTvd);
        capEntries = std::max<size_t>(capEntries, 1);
        ctx->dR1OvSlotElem.ensure(capEntries * 2 * sizeof(int));
        ctx->dR1OvVal.ensure(capEntries * 8 * sizeof(double));
        if (nR1Ov > 0) {
            CUDA_CHECK(cudaMemcpy(ctx->dR1OvSlotElem.p, r1OvSlotElem, static_cast<size_t>(nR1Ov) * 2 * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dR1OvVal.p, r1OvVal, static_cast<size_t>(nR1Ov) * 8 * sizeof(double), cudaMemcpyHostToDevice));
        }
    } else if (!ctx->dR1OvSlotElem.p) {
        int dummyI[2] = {-1, -1};
        double dummyD[1] = {0};
        ctx->dR1OvSlotElem.upload(dummyI, sizeof(dummyI));
        ctx->dR1OvVal.upload(dummyD, sizeof(dummyD));
    }
}

// ======================== INCREMENTAL TVC IN-KERNEL OUTER LOOP ========================
// Per (scenario, controller): check if |targetV - V| > halfDeadband. Writes 1 to outOfDb
// for each out-of-deadband pair, 0 otherwise. Also writes diffV per pair for the tap decision.
__global__ void tvcDeadbandCheckBatch(int nTvc, int nScenarios, int n, int nElements,
        const int* tvcVRow, const double* tvcTargetV, const double* tvcHalfDb,
        const int* tvcElemIndex, const char* disabled,
        const double* x, const char* active,
        char* outOfDb, double* diffV) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nTvc * nScenarios) return;
    int c = tid % nTvc;
    int s = tid / nTvc;
    if (!active[s]) {
        outOfDb[tid] = 0;
        return;
    }
    if (disabled[(size_t) s * nElements + tvcElemIndex[c]]) {    // outaged in this scenario
        outOfDb[tid] = 0;
        return;
    }
    double v = x[(size_t) s * n + tvcVRow[c]];
    double dv = tvcTargetV[c] - v;
    if (fabs(dv) > tvcHalfDb[c]) {
        outOfDb[tid] = 1;
        diffV[tid] = dv;
    } else {
        outOfDb[tid] = 0;
        diffV[tid] = 0.0;
    }
}

// Per (scenario, controller): given the sensitivity dV/dρ, find the closest tap and move there.
// Updates curTap, writes the new r1 to newR1, and sets moved[s] = 1 if any scenario moved.
// Single-controller-per-bus simplification for now; the multi-controller round-robin needs the
// group structure and is a follow-up.
// OLF checkAndAdjustControlledBus, one thread per (scenario, controlled-bus group). One controller:
// adjustWithOneController (full diffV, maxTapShift). Several: adjustWithSeveralControllers — the
// one-tap-at-a-time round-robin in merged order, remainingDiffV tracked through each controller's
// sensitivity, until a pass moves nothing. Both apply MIN_SENSI_FILTER and the ratchet
// (ControllerContext.updateAllowedDirection, MAX_DIRECTION_CHANGE 3); updateTapPositionToReachNewR1
// = the closest r1 within the shift bound and the allowed range, strictly closer than the current one.
__device__ inline int tvcMoveTap(const int* tapPtr, const double* tapData, size_t st, int c, double deltaR1, int maxShift,
                                 int* curTap, int* ratchetDir, int* ratchetCount, int* ratchetAllowed) {
    int cur = curTap[st];
    int nTaps = tapPtr[c + 1] - tapPtr[c];
    int base = tapPtr[c];
    double curR1 = tapData[(size_t) (base + cur) * 8 + 6];
    double targetR1 = curR1 + deltaR1;
    int lo = ratchetAllowed[st] > 0 ? cur : 0;
    int hi = ratchetAllowed[st] < 0 ? cur : nTaps - 1;
    int best = cur;
    double bestDist = fabs(targetR1 - curR1);
    for (int i = lo; i <= hi; i++) {
        if (abs(i - cur) > maxShift) continue;
        double dist = fabs(targetR1 - tapData[(size_t) (base + i) * 8 + 6]);
        if (dist < bestDist) { best = i; bestDist = dist; }
    }
    if (best == cur) return 0;
    curTap[st] = best;
    int direction = best > cur ? 1 : -1;
    if (ratchetCount[st] < 3) {
        if (ratchetDir[st] != 0 && ratchetDir[st] != direction) ratchetCount[st]++;
        ratchetDir[st] = direction;
    } else {
        ratchetAllowed[st] = direction;
    }
    return direction;
}

__global__ void tvcGroupDecisionBatch(int nGroups, int nScenarios, int nTvc,
        const int* groupPtr, const int* groupCtrl,
        const char* outOfDb, const double* diffV, const double* sensi, const double* halfDb,
        const int* tapPtr, const double* tapData,
        int* curTap, int* ratchetDir, int* ratchetCount, int* ratchetAllowed,
        int maxTapShift, double minSensiFilter, char* moved) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nGroups * nScenarios) return;
    int g = tid % nGroups;
    int s = tid / nGroups;
    int k0 = groupPtr[g];
    int k1 = groupPtr[g + 1];
    // the group's live controllers (outOfDb is per controller: 0 when outaged or inside the band)
    int nLive = 0;
    int lastLive = -1;
    for (int k = k0; k < k1; k++) {
        int c = groupCtrl[k];
        if (outOfDb[(size_t) s * nTvc + c]) { nLive++; lastLive = c; }
    }
    if (nLive == 0) return;
    if (nLive == 1 && k1 - k0 == 1) {                        // adjustWithOneController
        size_t st = (size_t) s * nTvc + lastLive;
        double sv = sensi[st];
        if (fabs(sv) < minSensiFilter) return;
        if (tvcMoveTap(tapPtr, tapData, st, lastLive, diffV[st] / sv, maxTapShift,
                       curTap, ratchetDir, ratchetCount, ratchetAllowed) != 0) moved[s] = 1;
        return;
    }
    // adjustWithSeveralControllers: one tap per controller per pass, in merged order
    size_t st0 = (size_t) s * nTvc + groupCtrl[k0];
    double remaining = diffV[st0];
    double hdb = halfDb[groupCtrl[k0]];
    bool changed = true;
    bool any = false;
    while (changed) {
        changed = false;
        for (int k = k0; k < k1; k++) {
            int c = groupCtrl[k];
            size_t st = (size_t) s * nTvc + c;
            if (!outOfDb[st]) continue;                      // outaged member
            if (fabs(remaining) <= hdb) continue;
            double sv = sensi[st];
            if (fabs(sv) < minSensiFilter) continue;
            double prevR1 = tapData[(size_t) (tapPtr[c] + curTap[st]) * 8 + 6];
            if (tvcMoveTap(tapPtr, tapData, st, c, remaining / sv, 1,
                           curTap, ratchetDir, ratchetCount, ratchetAllowed) != 0) {
                remaining -= (tapData[(size_t) (tapPtr[c] + curTap[st]) * 8 + 6] - prevR1) * sv;
                changed = true;
                any = true;
            }
        }
    }
    if (any) moved[s] = 1;
}

// Per scenario: read the controlled bus's dV out of the unit-RHS solution for controller c.
__global__ void tvcGatherSensiBatch(int nScenarios, int n, int c, int nTvc,
                                    const int* tvcVRow, const double* dx, double* sensi) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    sensi[(size_t) s * nTvc + c] = dx[(size_t) s * n + tvcVRow[c]];
}

// Per (scenario, controller) whose tap moved this pass: pin the rho identity row's TARGET at the
// new tap's r1 and seed the rho STATE there — exactly what the host orchestrator writes before its
// warm re-solve (ovVal.add(row[6]); x0Full[rhoRow] = row[6]). Then remember the tap as applied.
__global__ void tvcApplyRhoBatch(int nTvc, int nScenarios, int n,
                                 const int* tvcRhoRow, const int* tapPtr, const double* tapData,
                                 const int* curTap, int* prevTap, double* target, double* x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nTvc * nScenarios) return;
    int c = tid % nTvc;
    int s = tid / nTvc;
    int t = curTap[tid];
    if (t == prevTap[tid]) return;
    double r1 = tapData[(size_t) (tapPtr[c] + t) * 8 + 6];
    target[(size_t) s * n + tvcRhoRow[c]] = r1;
    x[(size_t) s * n + tvcRhoRow[c]] = r1;
    prevTap[tid] = t;
}

/** Rebuild the device tap-override list = [every (scenario, controller) whose current tap differs
 *  from its base tap, as the tap's full 8-double pi row] ++ [the caller's own overrides]. The TVC
 *  entries go FIRST because findTapOverride returns the first (slot, elem) match, so a contingency
 *  tap action on the same transformer must not shadow the loop's move. Runs after every pass that
 *  moved a tap, and on a driver RESUME (the caller's list is re-uploaded at solve entry and would
 *  otherwise erase the moves the paused sequence had already made). */
static void tvcRebuildOverrides(BatchSolverContext* ctx) {
    int nS = ctx->nScenarios;
    int nTvc = ctx->nTvc;
    std::vector<int> slot;
    std::vector<double> val;
    if (nTvc > 0) {
        std::vector<int> cur(static_cast<size_t>(nS) * nTvc);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dTvcCurTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int c = 0; c < nTvc; c++) {
                int t = cur[static_cast<size_t>(s) * nTvc + c];
                if (t == ctx->hTvcBaseTap[c]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hTvcElemIndex[c]);
                const double* row = &ctx->hTvcTapData[static_cast<size_t>(ctx->hTvcTapPtr[c] + t) * 8];
                val.insert(val.end(), row, row + 8);
            }
        }
    }
    if (ctx->nCph > 0) {                                     // the continuous phase loop's rounded/shifted taps
        std::vector<int> cur(static_cast<size_t>(nS) * ctx->nCph);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dCphTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int c = 0; c < ctx->nCph; c++) {
                int t = cur[static_cast<size_t>(s) * ctx->nCph + c];
                if (t == ctx->hCphBaseTap[c]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hCphElem[c]);
                const double* row = &ctx->hCphTapData[static_cast<size_t>(ctx->hCphTapPtr[c] + t) * 8];
                val.insert(val.end(), row, row + 8);
            }
        }
    }
    if (ctx->nFlc > 0) {                                     // the flow-observed loops' moved taps too
        std::vector<int> cur(static_cast<size_t>(nS) * ctx->nFlc);
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dFlcCurTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int c = 0; c < ctx->nFlc; c++) {
                int t = cur[static_cast<size_t>(s) * ctx->nFlc + c];
                if (t == ctx->hFlcBaseTap[c]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hFlcElem[c]);
                const double* row = &ctx->hFlcTapData[static_cast<size_t>(ctx->hFlcTapPtr[c] + t) * 8];
                val.insert(val.end(), row, row + 8);
            }
        }
    }
    if (ctx->nTvd > 0) {                                     // the discrete TVC loop's fixed (rounded/clamped) taps
        std::vector<int> st(static_cast<size_t>(nS) * ctx->nTvd), cur(static_cast<size_t>(nS) * ctx->nTvd);
        CUDA_CHECK(cudaMemcpy(st.data(), ctx->dTvdMemState.p, st.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dTvdMemTap.p, cur.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            for (int m = 0; m < ctx->nTvd; m++) {
                size_t sm = static_cast<size_t>(s) * ctx->nTvd + m;
                if (st[sm] != 2 || cur[sm] == ctx->hTvdMemBaseTap[m]) continue;
                slot.push_back(s);
                slot.push_back(ctx->hTvdMemElem[m]);
                const double* row = &ctx->hTvdTapData[static_cast<size_t>(ctx->hTvdMemTapPtr[m] + cur[sm]) * 8];
                val.insert(val.end(), row, row + 8);
            }
        }
    }
    slot.insert(slot.end(), ctx->hR1OvBaseSlotElem.begin(), ctx->hR1OvBaseSlotElem.end());
    val.insert(val.end(), ctx->hR1OvBaseVal.begin(), ctx->hR1OvBaseVal.end());
    int nOv = static_cast<int>(slot.size() / 2);
    if (nOv > 0) {
        ctx->dR1OvSlotElem.ensure(slot.size() * sizeof(int));
        ctx->dR1OvVal.ensure(val.size() * sizeof(double));
        CUDA_CHECK(cudaMemcpy(ctx->dR1OvSlotElem.p, slot.data(), slot.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dR1OvVal.p, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    ctx->nR1Ov = nOv;
}
// ======================== FLOW-OBSERVED INCREMENTAL LOOPS (phase / RPC) ========================
// The controlled branch's flow and its gradient at the current state: P/Q through the codegen'd
// evalClosedBranch (the very functions the fill scatters), I through OLF's own reI/imI decomposition
// (ClosedBranchSide{1,2}CurrentMagnitudeEquationTerm, incl. its one-sided rule at I = 0 and its
// dr1-less sensitivity). g[0..5] = d/d(v1, v2, ph1, ph2, a1, r1).
__device__ inline double flcFlowAndGrad(int side, int kind, const double* p, const double* xs, const int* q, double* g) {
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = q[9] >= 0 ? xs[q[9]] : p[6];
    double a1 = q[8] >= 0 ? xs[q[8]] : p[7];
    double y = p[0], g1 = p[1], g2 = p[2], b1 = p[3], b2 = p[4], ksi = p[5];
    if (kind != 2) {                                         // P or Q: the assembled outputs
        double o[CB_NOUT];
        evalClosedBranch(y, g1, g2, b1, b2, v1, r1, v2, ksi, a1, ph1, ph2, 0,
                         o + 0, o + 1, o + 2, o + 3, o + 4, o + 5, o + 6, o + 7, o + 8, o + 9,
                         o + 10, o + 11, o + 12, o + 13, o + 14, o + 15, o + 16, o + 17, o + 18,
                         o + 19, o + 20, o + 21, o + 22, o + 23, o + 24, o + 25, o + 26, o + 27);
        int blk = (kind == 0 ? 7 : 0) + (side == 1 ? 0 : 14);   // p1 0, q1 7, p2 14, q2 21: [val dv1 dv2 dph1 dph2 da1 dr1]
        for (int k = 0; k < 6; k++) g[k] = o[blk + 1 + k];
        return o[blk];
    }
    const double R2 = 1.0;
    if (side == 1) {
        double theta = ksi - a1 + ph2;                       // OLF: ksi - a1 + A2 + ph2, A2 = 0
        double ire = g1 * cos(ph1) - b1 * sin(ph1) + y * sin(ph1 + ksi);
        double iim = g1 * sin(ph1) + b1 * cos(ph1) - y * cos(ph1 + ksi);
        double re = r1 * (r1 * v1 * ire - y * R2 * v2 * sin(theta));
        double im = r1 * (r1 * v1 * iim + y * R2 * v2 * cos(theta));
        double dre[6] = { r1 * r1 * ire, r1 * (-y * R2 * sin(theta)),
                          r1 * r1 * v1 * (-g1 * sin(ph1) - b1 * cos(ph1) + y * cos(ph1 + ksi)),
                          r1 * (-y * R2 * v2 * cos(theta)), 0.0, 0.0 };
        double dim[6] = { r1 * r1 * iim, r1 * (y * R2 * cos(theta)),
                          r1 * r1 * v1 * ire, r1 * (-y * R2 * v2 * sin(theta)), 0.0, 0.0 };
        dre[4] = -dre[3];                                    // a1 enters theta with the opposite sign of ph2
        dim[4] = -dim[3];
        double i = hypot(re, im);
        if (i == 0.0) {                                      // one-sided limit: the gradient's MAGNITUDE — the
            for (int k = 0; k < 6; k++) g[k] = 0.0;          // dot with dx is done by the caller as hypot(dre·dx, dim·dx)
            g[0] = 1e300;                                    // flag (see flcGatherSensiBatch)
            for (int k = 0; k < 6; k++) { g[6 + k] = dre[k]; g[12 + k] = dim[k]; }
            return 0.0;
        }
        for (int k = 0; k < 6; k++) g[k] = (re * dre[k] + im * dim[k]) / i;
        return i;
    }
    double theta = ksi + a1 + ph1;                           // OLF: ksi + a1 - A2 + ph1
    double ire = g2 * cos(ph2) - b2 * sin(ph2) + y * sin(ph2 + ksi);
    double iim = g2 * sin(ph2) + b2 * cos(ph2) - y * cos(ph2 + ksi);
    double re = R2 * (R2 * v2 * ire - y * r1 * v1 * sin(theta));
    double im = R2 * (R2 * v2 * iim + y * r1 * v1 * cos(theta));
    double dre[6] = { R2 * (-y * r1 * sin(theta)), R2 * R2 * ire,
                      R2 * (-y * r1 * v1 * cos(theta)),
                      R2 * R2 * v2 * (-g2 * sin(ph2) - b2 * cos(ph2) + y * cos(ph2 + ksi)), 0.0, 0.0 };
    double dim[6] = { R2 * (y * r1 * cos(theta)), R2 * R2 * iim,
                      R2 * (-y * r1 * v1 * sin(theta)), R2 * R2 * v2 * ire, 0.0, 0.0 };
    dre[4] = dre[2];                                         // a1 enters theta like ph1
    dim[4] = dim[2];
    double i = hypot(re, im);
    if (i == 0.0) {
        for (int k = 0; k < 6; k++) g[k] = 0.0;
        g[0] = 1e300;
        for (int k = 0; k < 6; k++) { g[6 + k] = dre[k]; g[12 + k] = dim[k]; }
        return 0.0;
    }
    for (int k = 0; k < 6; k++) g[k] = (re * dre[k] + im * dim[k]) / i;
    return i;
}

// Per (scenario, controller): OLF's out-of-deadband test on the controlled flow (LIMITER: one-sided).
__global__ void flcCheckBatch(int nFlc, int nScenarios, int n, int nElements, int cbOffset, int kindMask,
                              const int* kind, const int* side, const double* targetValue, const double* halfDb,
                              const int* elem, const int* ctlPack,
                              const double* cbIn, const int* cbIx, int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                              const char* disabled, const double* x, const char* rlActive,
                              char* outOfDb, double* diff) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nFlc * nScenarios) return;
    int c = tid % nFlc;
    int s = tid / nFlc;
    outOfDb[tid] = 0;
    if (!rlActive[s] || !((kindMask >> kind[c]) & 1)) return;
    int e = ctlPack[c];
    if (disabled[(size_t) s * nElements + elem[c]] || disabled[(size_t) s * nElements + cbOffset + e]) return;
    const double* p = cbIn + e * 8;
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) p = pOv;
    double g[18];
    double flow = flcFlowAndGrad(side[c], kind[c], p, x + (size_t) s * n, cbIx + e * 12, g);
    double d = targetValue[c] - flow;
    diff[tid] = d;
    outOfDb[tid] = (kind[c] == 2 ? flow > targetValue[c] : fabs(d) > halfDb[c]) ? 1 : 0;
}

// Per scenario, for controller c: the controlled flow's gradient dotted with the unit-RHS solution at
// the controlled branch's variable columns (a non-variable r1/a1 column contributes nothing).
__global__ void flcGatherSensiBatch(int nScenarios, int n, int c, int nFlc, int cbOffset,
                                    const int* kind, const int* side, const int* ctlPack,
                                    const double* cbIn, const int* cbIx, int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                    const double* x, const double* dx, double* sensi) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    int e = ctlPack[c];
    const double* p = cbIn + e * 8;
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) p = pOv;
    const int* q = cbIx + e * 12;
    const double* dxs = dx + (size_t) s * n;
    double g[18];
    flcFlowAndGrad(side[c], kind[c], p, x + (size_t) s * n, q, g);
    double d[6] = { dxs[q[5]], dxs[q[7]], dxs[q[4]], dxs[q[6]], q[8] >= 0 ? dxs[q[8]] : 0.0, q[9] >= 0 ? dxs[q[9]] : 0.0 };
    double sv;
    if (kind[c] == 2 && g[0] == 1e300) {                     // I = 0: hypot(dre·dx, dim·dx) (OLF's one-sided rule)
        double a = 0.0, b = 0.0;
        for (int k = 0; k < 5; k++) { a += g[6 + k] * d[k]; b += g[12 + k] * d[k]; }
        sv = hypot(a, b);
    } else {
        sv = 0.0;
        int nk = kind[c] == 2 ? 5 : 6;                       // OLF's current sensitivity has no dr1 term
        for (int k = 0; k < nk; k++) sv += g[k] * d[k];
    }
    sensi[(size_t) s * nFlc + c] = sv;
}

// OLF's tap decisions per (scenario, controller): RPC — deltaR1 = diffQ / sensi, closest r1 within
// maxTapShift (no sensitivity gate); phase CONTROLLER — |sensi·toRadians(1)| > SENSI_EPS, da = dp / sensi,
// closest a1 (MAX_TAP_SHIFT unbounded); phase LIMITER — same gate, da = (target − i) / sensi, then
// FirstTapPositionAboveFinder: walk tap by tap in the shift's direction while the remaining shift still
// reaches past the next tap's a1. Ratchet as everywhere (updateAllowedDirection, MAX_DIRECTION_CHANGE 3).
__global__ void flcDecisionBatch(int nFlc, int nScenarios, int kindMask,
                                 const int* kind, const int* pinCol, const int* tapPtr, const double* tapData, const int* maxShift,
                                 const char* outOfDb, const double* diff, const double* sensi,
                                 int* curTap, int* ratDir, int* ratCount, int* ratAllowed, char* moved) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nFlc * nScenarios) return;
    int c = tid % nFlc;
    int s = tid / nFlc;
    if (!outOfDb[tid] || !((kindMask >> kind[c]) & 1)) return;
    double sv = sensi[tid];
    if (kind[c] != 0 && fabs(sv * 0.017453292519943295) <= 1e-6) return;   // SENSI_EPS on the per-degree value
    if (sv == 0.0) return;
    double delta = diff[tid] / sv;
    int cur = curTap[tid];
    int base = tapPtr[c];
    int nTaps = tapPtr[c + 1] - base;
    int col = pinCol[c];
    int lo = ratAllowed[tid] > 0 ? cur : 0;
    int hi = ratAllowed[tid] < 0 ? cur : nTaps - 1;
    int best = cur;
    if (kind[c] != 2) {                                      // ClosestTapPositionFinder
        double newV = tapData[(size_t) (base + cur) * 8 + col] + delta;
        double bestDist = fabs(delta);
        for (int i = lo; i <= hi; i++) {
            if (abs(i - cur) > maxShift[c]) continue;
            double dist = fabs(newV - tapData[(size_t) (base + i) * 8 + col]);
            if (dist < bestDist) { best = i; bestDist = dist; }
        }
    } else {                                                 // FirstTapPositionAboveFinder
        double remaining = delta;
        int last = nTaps - 1;
        while (true) {
            int next = -1;                                   // OLF nextTapPositionIndex
            if (remaining != 0.0) {
                double vc = tapData[(size_t) (base + best) * 8 + col];
                auto descending = [&](int n1, int n2) {
                    double a = tapData[(size_t) (base + n1) * 8 + col], b = tapData[(size_t) (base + n2) * 8 + col];
                    return remaining > 0 ? a < b : a > b;
                };
                (void) vc;
                if (best == 0 && last >= 1 && descending(1, 0)) next = -1;
                else if (best > 0 && descending(best, best - 1)) next = best - 1;
                else if (best < last && descending(best, best + 1)) next = best + 1;
            }
            if (next < 0 || next < lo || next > hi) break;
            double value = tapData[(size_t) (base + best) * 8 + col];
            double nextValue = tapData[(size_t) (base + next) * 8 + col];
            best = next;
            if ((remaining < 0 && value + remaining > nextValue) || (remaining > 0 && value + remaining < nextValue)) break;
            remaining -= nextValue - value;
        }
    }
    if (best == cur) return;
    curTap[tid] = best;
    int direction = best > cur ? 1 : -1;
    if (ratCount[tid] < 3) {
        if (ratDir[tid] != 0 && ratDir[tid] != direction) ratCount[tid]++;
        ratDir[tid] = direction;
    } else {
        ratAllowed[tid] = direction;
    }
    moved[s] = 1;
}

// Per (scenario, controller) whose tap moved: pin the identity row (rho or a1) at the tap's value.
__global__ void flcApplyBatch(int nFlc, int nScenarios, int n, const int* pinRow, const int* pinCol,
                              const int* tapPtr, const double* tapData, const int* curTap, int* prevTap,
                              double* target, double* x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nFlc * nScenarios) return;
    int c = tid % nFlc;
    int s = tid / nFlc;
    int t = curTap[tid];
    if (t == prevTap[tid]) return;
    double v = tapData[(size_t) (tapPtr[c] + t) * 8 + pinCol[c]];
    target[(size_t) s * n + pinRow[c]] = v;
    x[(size_t) s * n + pinRow[c]] = v;
    prevTap[tid] = t;
}
// ======================== CONTINUOUS PHASE CONTROL (OLF PhaseControlOuterLoop) ========================
// One thread per (scenario, controller). iter[s] = the loop's check count in this solve. First check:
// every live CONTROLLER-mode control is switched OFF (row mode 3 -> 0, the identity pinned at the tap
// closest to the solved a1) and the scenario is UNSTABLE when it has any live controller (OLF returns
// UNSTABLE on a non-empty controller list, LIMITER-only included — a no-change re-solve). Later checks:
// a live LIMITER whose current exceeds the limit shifts one tap against the sign of dI/da1
// (shiftOneTapPositionToChangeA1). The tap's pi row follows through the shared r1 override rebuild.
__global__ void cphCheckBatch(int nCph, int nScenarios, int n, int nElements, int cbOffset,
                              const int* kind, const int* a1Row, const int* side, const double* limit, const int* elem,
                              const int* tapPtr, const double* tapData,
                              const double* cbIn, const int* cbIx, int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                              const char* disabled, const char* rlActive, const int* iter,
                              int* tap, int* mode, double* target, double* x, char* switched) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nCph * nScenarios) return;
    int c = tid % nCph;
    int s = tid / nCph;
    if (!rlActive[s] || disabled[(size_t) s * nElements + elem[c]]) return;
    size_t ar = (size_t) s * n + a1Row[c];
    int base = tapPtr[c];
    int nTaps = tapPtr[c + 1] - base;
    if (iter[s] == 0) {
        if (kind[c] == 0) {                                  // switch off + round a1 to the closest tap
            double a1 = x[ar];
            int best = tap[tid];
            for (int t = 0; t < nTaps; t++) {
                if (fabs(tapData[(size_t) (base + t) * 8 + 7] - a1) < fabs(tapData[(size_t) (base + best) * 8 + 7] - a1)) best = t;
            }
            tap[tid] = best;
            mode[ar] = 0;
            target[ar] = tapData[(size_t) (base + best) * 8 + 7];
            x[ar] = target[ar];
        }
        switched[s] = 1;                                     // OLF: UNSTABLE whenever a controller exists
        return;
    }
    if (kind[c] != 1) return;
    int e = elem[c] - cbOffset;                              // LIMITER is local: the controlled branch is itself
    const double* p = cbIn + e * 8;
    double pOv[8];
    if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) p = pOv;
    double g[18];
    double i = flcFlowAndGrad(side[c], 2, p, x + (size_t) s * n, cbIx + e * 12, g);
    if (!(i > limit[c])) return;
    double dida1 = g[0] == 1e300 ? 0.0 : g[4];
    int dir = dida1 > 0 ? -1 : 1;                            // reduce I: move a1 against the slope
    int cur = tap[tid];
    double a1 = tapData[(size_t) (base + cur) * 8 + 7];
    int next = cur;
    if (cur + 1 < nTaps && ((dir > 0) == (tapData[(size_t) (base + cur + 1) * 8 + 7] > a1))) next = cur + 1;
    else if (cur - 1 >= 0 && ((dir > 0) == (tapData[(size_t) (base + cur - 1) * 8 + 7] > a1))) next = cur - 1;
    if (next == cur) return;
    tap[tid] = next;
    target[ar] = tapData[(size_t) (base + next) * 8 + 7];
    x[ar] = target[ar];
    switched[s] = 1;
}
// ======================== INCREMENTAL SHUNT VC KERNELS ========================
// Per (scenario, controlled-bus group): OLF isOutOfDeadband on the group's controlled bus, with at
// least one live (not outaged) controller shunt.
__global__ void shcDeadbandCheckBatch(int nGroups, int nScenarios, int n, int nElements,
                                      const int* groupPtr, const int* groupCtrl,
                                      const int* vRow, const double* targetV, const double* halfDb, const int* elemAbs,
                                      const char* disabled, const double* x, const char* rlActive,
                                      char* outOfDb, double* diffV) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nGroups * nScenarios) return;
    int g = tid % nGroups;
    int s = tid / nGroups;
    outOfDb[tid] = 0;
    if (!rlActive[s]) return;
    int c0 = groupCtrl[groupPtr[g]];
    bool live = false;
    for (int k = groupPtr[g]; k < groupPtr[g + 1] && !live; k++) {
        int c = groupCtrl[k];
        live = elemAbs[c] < 0 || !disabled[(size_t) s * nElements + elemAbs[c]];
    }
    double d = targetV[c0] - x[(size_t) s * n + vRow[c0]];
    diffV[tid] = d;
    outOfDb[tid] = (live && fabs(d) > halfDb[c0]) ? 1 : 0;
}

__global__ void shcGatherSensiBatch(int nScenarios, int n, int c, int nShc, const int* vRow,
                                    const double* dx, double* sensi) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    sensi[(size_t) s * nShc + c] = dx[(size_t) s * n + vRow[c]];
}

// OLF IncrementalShuntVoltageControlOuterLoop.adjustB, one thread per (scenario, group): round-robin
// over the |b|-sorted controller shunts, one section per sub-controller per pass, the per-round
// maxSectionShift cap, remainingDiffV tracked through the sensitivity; LfShunt.Controller.
// updateSectionB (closest section within one step and the ratchet's allowed range, strict <, only
// when closer than the current b); ControllerContext.updateAllowedDirection (MAX_DIRECTION_CHANGE 3).
__global__ void shcAdjustBatch(int nGroups, int nScenarios, int n, int nElements, int nShc, int nSubs, int maxSectionShift,
                               const int* groupPtr, const int* groupCtrl, const int* subPtr,
                               const int* secPtr, const double* secB, const int* subMin,
                               const double* halfDb, const int* elemAbs, const char* disabled,
                               const char* outOfDb, const double* diffV, const double* sensi,
                               int* pos, int* ratDir, int* ratCount, int* ratAllowed, int* shift,
                               char* switched) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nGroups * nScenarios) return;
    int g = tid % nGroups;
    int s = tid / nGroups;
    if (!outOfDb[tid]) return;
    int c0 = groupCtrl[groupPtr[g]];
    double hdb = halfDb[c0];
    double remaining = diffV[tid];
    for (int k = groupPtr[g]; k < groupPtr[g + 1]; k++) {   // per-round shift counters
        int c = groupCtrl[k];
        for (int u = subPtr[c]; u < subPtr[c + 1]; u++) shift[(size_t) s * nSubs + u] = 0;
    }
    bool changed = true;
    bool moved = false;
    while (changed) {
        changed = false;
        for (int k = groupPtr[g]; k < groupPtr[g + 1]; k++) {
            int c = groupCtrl[k];
            if (elemAbs[c] >= 0 && disabled[(size_t) s * nElements + elemAbs[c]]) continue;   // outaged
            double sv = sensi[(size_t) s * nShc + c];
            for (int u = subPtr[c]; u < subPtr[c + 1]; u++) {
                if (fabs(remaining) <= hdb) continue;
                size_t su = (size_t) s * nSubs + u;
                if (shift[su] > maxSectionShift) continue;   // reached max section shift this outer loop
                int cur = pos[su];
                int p0 = secPtr[u];
                int nSec = secPtr[u + 1] - p0;
                double prevB = secB[p0 + cur];
                double deltaB = remaining / sv;
                double newB = prevB + deltaB;
                int allowed = ratAllowed[su];
                int lo = allowed > 0 ? cur : subMin[u];
                int hi = allowed < 0 ? cur : nSec - 1;
                int best = cur;
                double smallest = fabs(deltaB);
                for (int p = lo; p <= hi; p++) {
                    if (abs(p - cur) > 1) continue;          // maxSectionShift = 1 per updateSectionB call
                    double dist = fabs(newB - secB[p0 + p]);
                    if (dist < smallest) { best = p; smallest = dist; }
                }
                if (best == cur) continue;
                pos[su] = best;
                int direction = best > cur ? 1 : -1;
                if (ratCount[su] < 3) {                      // ControllerContext.updateAllowedDirection
                    if (ratDir[su] != 0 && ratDir[su] != direction) ratCount[su]++;
                    ratDir[su] = direction;
                } else {
                    ratAllowed[su] = direction;
                }
                shift[su]++;
                remaining -= (secB[p0 + best] - prevB) * sv;
                changed = true;
                moved = true;
            }
        }
    }
    if (moved) switched[s] = 1;
}

// Per (scenario, controller): the section total B pins the SHUNT_B identity row (target + state
// seed), exactly the host orchestrator's ovVal.add(baseB + db) / x0Full[bRow] = newB.
__global__ void shcApplyBatch(int nShc, int nScenarios, int n, const int* bRow, const int* subPtr,
                              const int* secPtr, const double* secB, const int* subBase, const int* pos,
                              int nSubs, const double* baseB, double* appliedDb, double* target, double* x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nShc * nScenarios) return;
    int c = tid % nShc;
    int s = tid / nShc;
    double db = 0.0;
    for (int u = subPtr[c]; u < subPtr[c + 1]; u++) {
        db += secB[secPtr[u] + pos[(size_t) s * nSubs + u]] - secB[secPtr[u] + subBase[u]];
    }
    if (db == appliedDb[tid]) return;
    double nb = baseB[tid] + db;
    target[(size_t) s * n + bRow[c]] = nb;
    x[(size_t) s * n + bRow[c]] = nb;
    appliedDb[tid] = db;
}

/** Rebuild the device shunt-override list = the caller's entries ++ one (dG, dB) per (scenario,
 *  controller) whose sections moved off their base — additive deltas (applyShuntOverride SUMS every
 *  matching entry), so the loop's move composes on top of whatever the caller had. Re-done after
 *  every pass that moved a section and on a driver RESUME. */
static void shcRebuildOverrides(BatchSolverContext* ctx) {
    int nS = ctx->nScenarios;
    std::vector<int> pos(static_cast<size_t>(nS) * ctx->nShcSubs);
    CUDA_CHECK(cudaMemcpy(pos.data(), ctx->dShcPos.p, pos.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> slot(ctx->hShOvBaseSlotElem);
    std::vector<double> val(ctx->hShOvBaseGb);
    for (int s = 0; s < nS; s++) {
        for (int c = 0; c < ctx->nShc; c++) {
            if (ctx->hShcShOvElem[c] < 0) continue;          // no packed element: b rides SHUNT_B only
            double db = 0.0, dg = 0.0;
            for (int u = ctx->hShcSubPtr[c]; u < ctx->hShcSubPtr[c + 1]; u++) {
                int p0 = ctx->hShcSecPtr[u];
                int pc = pos[static_cast<size_t>(s) * ctx->nShcSubs + u];
                db += ctx->hShcSecB[p0 + pc] - ctx->hShcSecB[p0 + ctx->hShcSubBase[u]];
                dg += ctx->hShcSecG[p0 + pc] - ctx->hShcSecG[p0 + ctx->hShcSubBase[u]];
            }
            if (db == 0.0 && dg == 0.0) continue;
            slot.push_back(s);
            slot.push_back(ctx->hShcShOvElem[c]);
            val.push_back(dg);
            val.push_back(db);
        }
    }
    int nOv = static_cast<int>(slot.size() / 2);
    if (nOv > 0) {
        ctx->dShOvSlotElem.ensure(slot.size() * sizeof(int));
        ctx->dShOvGb.ensure(val.size() * sizeof(double));
        CUDA_CHECK(cudaMemcpy(ctx->dShOvSlotElem.p, slot.data(), slot.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dShOvGb.p, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    ctx->nShOv = nOv;
}
// ======================== END INCREMENTAL TVC KERNELS ========================
// SecondaryVoltageControl's tryToReEnableHelpfulControllerBuses, applied to the DEVICE pin state at a
// driver pause: a zone whose controllers are only PARTLY at a reactive limit un-pins the limited ones so
// they can help the zone realign. OLF does it with setGeneratorVoltageControlEnabledAndRecomputeTargetQ
// (back to PV) + setQLimitType(null); here that is the row's saved PV form, the zone's CURRENT voltage
// target, and limitType 0. The host orchestrator got this for free by re-seeding rlStateIn on its own
// re-solve — a RESUME skips the RL seeding, so without this channel the un-pin never reached the device.
__global__ void rlUnpinBatch(int nPairs, int nS, int n, int nrl, const int* pairs, const double* newTarget,
                             const int* rlVrow, char* limitType, const char* savedMode, int* mode, double* target) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nPairs) return;
    int s = pairs[2 * k];
    int i = pairs[2 * k + 1];
    if (s < 0 || s >= nS || i < 0 || i >= nrl) return;
    size_t st = (size_t) s * nrl + i;
    if (limitType[st] == 0) return;                          // not pinned: nothing to give back
    // A member FROZEN by the generator voltage control manager is not a reactive-limit pin: giving it
    // back would restore a V identity on a row that holds the group's DISTR_Q share, and would undo a
    // decision the transformer phase owns and will undo itself at COMPLETE.
    if (limitType[st] == RL_LIMIT_FROZEN) return;
    size_t idx = (size_t) s * n + rlVrow[i];
    char sm = savedMode[st];
    mode[idx] = (sm == 2) ? 2 : 0;                           // the PV form saved at pin time (2 = slope SVC)
    target[idx] = newTarget[k];
    limitType[st] = 0;
}

// ======================== DISCRETE TRANSFORMER / SHUNT VOLTAGE CONTROL KERNELS ========================
// OLF's non-incremental voltage-control loops as driver phases:
//  - TransformerVoltageControlOuterLoop (AFTER_GENERATOR, tvdMode 0): INITIAL (enable the out-of-deadband
//    controls, TransformerRatioManager bounds) -> CONTROL (clamp continuous ratios outside their range to
//    the extreme, round, disable; else updateContinuousRatio + roundVoltageRatios -> COMPLETE; UNSTABLE in
//    any case) -> COMPLETE (STABLE);
//  - SimpleTransformerVoltageControlOuterLoop (WITH_GENERATOR, tvdMode 1): control ON for the first solve,
//    every controller rounded to its closest tap and switched off at the first check (UNSTABLE);
//  - ShuntVoltageControlOuterLoop (WITH_GENERATOR): control ON for the first solve, LfShuntImpl.dispatchB
//    per controller shunt and switched off at the first check (UNSTABLE).
// A group's ON/OFF activations are the toggle-diffed row deltas the Java side captured; member rows are
// laid out by ROLE per scenario — the first alive continuous member hosts the relocated pilot Q (the
// fills' mode scan picks the same row), the others host DISTR_RHO (target 0), fixed/outaged members are
// identity pins — so a member outage or a clamp re-forms the group exactly as OLF's "first enabled" rule.
struct TvdTables {
    int nG, nM, nDelta, mode, rejectCarrierClamp;
    const int* grpPtr; const int* grpVRow; const double* grpTargetV; const double* grpHalfDb; const int* grpUseInit;
    const int* deltaPtr; const int* deltaRow; const int* deltaModeOn; const int* deltaModeOff;
    const double* deltaTargetOn; const double* deltaTargetOff;
    const int* memRhoRow; const int* memElem; const int* memBaseTap; const int* memTapPtr;
    const double* memMin; const double* memMax; const double* memInit; const double* tapData;
    int* scnStep; int* grpStep; double* grpShMin; double* grpShMax; double* grpShInit;
    int* memState; int* memTap; double* memRoundFrom; double* saved; char* reject;
    // generator freeze + connectivity fix (AFTER_GENERATOR only)
    int nFz, hasFix, fixWords, nrl;
    const int* fzVRow; const double* fzLoadQ; const int* fzRlEntry; const int* fixPvRow;
    char* rlLimitType;                                       // the freeze marks a REMOTE member here
    const int* fixMinMask; const int* fixMaxMask;               // per (scenario, member), fixWords ints
    int* fzFrozen; int* fzSavedMode; double* fzSavedTarget;
};

// Does any generator-voltage-CONTROLLED bus in this mask still carry an ENABLED control? Row mode 0 is
// a plain V identity and 2 a slope SVC — both are voltage control ON; a frozen or reactive-limit-pinned
// bus is a power row, which is exactly OLF's isGeneratorVoltageControlEnabled() == false.
__device__ inline bool tvdAnyPv(const TvdTables& T, size_t sn, const int* mode, const int* mask) {
    for (int w = 0; w < T.fixWords; w++) {
        unsigned int bits = (unsigned int) mask[w];
        while (bits) {
            int b = __ffs((int) bits) - 1;
            bits &= bits - 1;
            int mr = mode[sn + T.fixPvRow[w * 32 + b]];
            if (mr == 0 || mr == 2) {
                return true;
            }
        }
    }
    return false;
}

// OLF PiModelArray.roundR1ToClosestTap: ClosestTapPositionFinder starts at the CURRENT (base) tap and
// takes a strictly closer one only.
__device__ inline double tvdRoundToTap(const TvdTables& T, int m, double r1, int* tapOut) {
    int base = T.memTapPtr[m];
    int nTaps = T.memTapPtr[m + 1] - base;
    int best = T.memBaseTap[m];
    for (int t = 0; t < nTaps; t++) {
        if (fabs(T.tapData[(size_t) (base + t) * 8 + 6] - r1) < fabs(T.tapData[(size_t) (base + best) * 8 + 6] - r1)) best = t;
    }
    *tapOut = best;
    return T.tapData[(size_t) (base + best) * 8 + 6];
}

// Lay out one group's rows for one scenario from its members' states: ON (any alive continuous member)
// or OFF (none). Non-member delta rows (the pilot's V identity) apply as captured — the scenario's OFF
// target is saved on the way ON (a per-scenario override on the pilot's Q row must come back); member
// rows by role.
__device__ inline void tvdLayoutGroup(const TvdTables& T, int s, int n, int nElements, int g,
                                      const char* disabled, int* mode, double* target, double* x) {
    size_t sn = (size_t) s * n;
    bool anyCont = false;
    for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
        if (!disabled[(size_t) s * nElements + T.memElem[m]] && T.memState[(size_t) s * T.nM + m] == 1) anyCont = true;
    }
    int m0 = T.grpPtr[g];
    double pilotQ = 0.0;
    for (int k = T.deltaPtr[g]; k < T.deltaPtr[g + 1]; k++) {
        if (T.deltaRow[k] == T.memRhoRow[m0]) pilotQ = T.deltaTargetOn[k];   // the BASE carrier's captured pilot-Q target
    }
    for (int k = T.deltaPtr[g]; k < T.deltaPtr[g + 1]; k++) {
        int r = T.deltaRow[k];
        bool memberRow = false;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) if (T.memRhoRow[m] == r) memberRow = true;
        if (memberRow) continue;                             // member rows are built by ROLE below
        size_t sk = (size_t) s * T.nDelta + k;
        if (anyCont) {
            if (isnan(T.saved[sk])) T.saved[sk] = target[sn + r];
            mode[sn + r] = T.deltaModeOn[k];
            target[sn + r] = T.deltaTargetOn[k];
        } else {
            mode[sn + r] = T.deltaModeOff[k];
            target[sn + r] = isnan(T.saved[sk]) ? T.deltaTargetOff[k] : T.saved[sk];
            T.saved[sk] = nan("");
        }
    }
    bool carrier = false;
    for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
        if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;   // outaged: base identity as uploaded
        size_t mr = sn + T.memRhoRow[m];
        int st = T.memState[(size_t) s * T.nM + m];
        if (anyCont && st == 1) {                            // ALIVE continuous: carrier (pilot Q) or DISTR_RHO (0)
            mode[mr] = 1;
            target[mr] = carrier ? 0.0 : pilotQ;
            carrier = true;
        } else if (st == 2) {                                // FIXED: identity pin at the rounded tap's r1
            int tap = T.memTap[(size_t) s * T.nM + m];
            double r1 = T.tapData[(size_t) (T.memTapPtr[m] + tap) * 8 + 6];
            mode[mr] = 0;
            target[mr] = r1;
            x[mr] = r1;
        }
    }
}

// Solve entry (OLF initialize): WITH_GENERATOR enables every connected controller before the first solve —
// applied here for a chunk (entryOn = 0), or read off the uploaded activation for a base load flow whose
// network OLF's own initialize already switched on (entryOn = 1). AFTER_GENERATOR starts with every
// control off.
__global__ void tvdEntryBatch(TvdTables T, int nScenarios, int n, int nElements, int entryOn,
                              const char* disabled, int* mode, double* target, double* x) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    size_t sn = (size_t) s * n;
    for (int g = 0; g < T.nG; g++) {
        bool any = false;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;
            size_t sm = (size_t) s * T.nM + m;
            if (entryOn) {
                T.memState[sm] = mode[sn + T.memRhoRow[m]] == 1 ? 1 : 0;
            } else if (T.mode == 1) {
                T.memState[sm] = 1;
                any = true;
            }
        }
        if (any) tvdLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
    }
}

// One thread per scenario (OLF keeps ONE step for all controls of a run).
__global__ void tvdCheckBatch(TvdTables T, int nScenarios, int n, int nElements, const char* disabled, const char* rlActive,
                              const double* qBuf, int* mode, double* target, double* x, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !rlActive[s]) return;
    size_t sn = (size_t) s * n;
    if (T.mode == 1) {                                       // SimpleTransformerVoltageControl: iteration 0 rounds all
        bool anyCtl = false;
        for (int g = 0; g < T.nG; g++) {
            size_t sg = (size_t) s * T.nG + g;
            if (T.grpStep[sg] != 0) continue;
            bool touched = false;
            for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
                if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;
                anyCtl = true;
                size_t sm = (size_t) s * T.nM + m;
                if (T.memState[sm] != 1) continue;
                T.memRoundFrom[sm] = x[sn + T.memRhoRow[m]];
                int tap;
                tvdRoundToTap(T, m, T.memRoundFrom[sm], &tap);
                T.memTap[sm] = tap;
                T.memState[sm] = 2;
                touched = true;
            }
            T.grpStep[sg] = 2;
            if (touched) tvdLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
        }
        if (anyCtl) switched[s] = 1;                         // OLF: UNSTABLE whenever a live controller exists
        return;
    }
    int step = T.scnStep[s];
    if (step == 2) return;                                   // COMPLETE: STABLE
    if (step == 0) {                                         // INITIAL
        bool needRun = false;
        for (int g = 0; g < T.nG; g++) {
            int alive = 0;
            for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) if (!disabled[(size_t) s * nElements + T.memElem[m]]) alive++;
            if (alive == 0) continue;
            if (!(fabs(T.grpTargetV[g] - x[sn + T.grpVRow[g]]) > T.grpHalfDb[g])) continue;
            // TransformerRatioManager over the just-enabled members: group-averaged bounds and the
            // normalized-headroom initial (the exact identity for a single member)
            double sumMin = 0.0, sumMax = 0.0, a = 0.0, b = 0.0;
            for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
                if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;
                T.memState[(size_t) s * T.nM + m] = 1;
                sumMin += T.memMin[m];
                sumMax += T.memMax[m];
                if (T.memMax[m] != T.memMin[m]) {
                    a += (T.memMax[m] - T.memInit[m]) / (T.memMax[m] - T.memMin[m]);
                    b += (T.memInit[m] - T.memMin[m]) / (T.memMax[m] - T.memMin[m]);
                } else {
                    a += 1.0;
                    b += 1.0;
                }
            }
            size_t sg = (size_t) s * T.nG + g;
            T.grpShMin[sg] = sumMin / alive;
            T.grpShMax[sg] = sumMax / alive;
            T.grpShInit[sg] = (T.grpShMin[sg] * a + T.grpShMax[sg] * b) / alive;
            needRun = true;
        }
        if (!needRun) {
            T.scnStep[s] = 2;                                // every control in band: COMPLETE, STABLE
            return;
        }
        // GeneratorVoltageControlManager.disableGeneratorVoltageControlsUnderMaxControlledNominalVoltage:
        // every candidate whose control is ON right now (not RL-pinned) freezes at its current Q
        for (int c = 0; c < T.nFz; c++) {
            size_t sr = sn + T.fzVRow[c];
            size_t sc = (size_t) s * T.nFz + c;
            if (T.fzFrozen[sc]) continue;
            int re = T.fzRlEntry[c];
            if (re >= 0) {
                // REMOTE controller. OLF freezes it exactly like a local one, but its own row is the
                // group's DISTR_Q SHARE, not a V identity — so the freeze is the reactive-limit remote
                // PIN (mode 4, "the row hosts its own Q balance") with the CURRENT injection as the
                // target instead of a Q limit. The limit type is what the DISTR_Q key renormalization
                // reads to drop a member, so it has to be set, and RL_LIMIT_FROZEN keeps it
                // distinguishable from a real Q-limit pin (which the switch-back may undo; this must
                // not be, OLF's reactive-limits loop never sees a bus whose voltage control is off).
                size_t st = (size_t) s * T.nrl + re;
                if (mode[sr] != 1 || T.rlLimitType[st] != 0) continue;   // pinned already, or not live
                T.fzSavedMode[sc] = mode[sr];
                T.fzSavedTarget[sc] = target[sr];
                mode[sr] = 4;
                target[sr] = qBuf[sr] - T.fzLoadQ[c];
                T.rlLimitType[st] = RL_LIMIT_FROZEN;
                T.fzFrozen[sc] = 1;
                continue;
            }
            if (mode[sr] != 0 && mode[sr] != 2) continue;
            T.fzSavedMode[sc] = mode[sr];
            T.fzSavedTarget[sc] = target[sr];
            mode[sr] = 1;
            target[sr] = qBuf[sr] - T.fzLoadQ[c];            // generationTargetQ = q.eval(); row target = gen − load
            T.fzFrozen[sc] = 1;
        }
        // LfNetwork.fixTransformerVoltageControls: a just-enabled controller whose not-controlled side's
        // component holds no ENABLED generator voltage control (frozen and pinned buses count as off)
        // is disabled again. OLF cuts the CONTINGENCY's branches and the ENABLED controller branches;
        // the host knows the first and brackets the second with two masks (see FreezeTables), so a PV
        // bus in the MIN mask is one OLF sees too, and no PV bus in the MAX mask means OLF sees none.
        // Between the two the answer depends on which controllers THIS scenario enabled, which is not
        // something the host could precompute — that scenario goes to the CPU.
        if (T.hasFix) {
            for (int g = 0; g < T.nG; g++) {
                for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
                    size_t sm = (size_t) s * T.nM + m;
                    if (T.memState[sm] != 1) continue;
                    size_t base = ((size_t) s * T.nM + m) * T.fixWords;
                    bool pvMin = tvdAnyPv(T, sn, mode, T.fixMinMask + base);
                    if (pvMin) continue;                       // OLF keeps this control enabled
                    if (tvdAnyPv(T, sn, mode, T.fixMaxMask + base)) {
                        T.reject[s] = 1;                       // ambiguous without the enabled set
                        return;
                    }
                    T.memState[sm] = 0;                        // OLF disables it
                }
            }
        }
        for (int g = 0; g < T.nG; g++) {
            bool any = false;
            for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) any |= T.memState[(size_t) s * T.nM + m] == 1;
            if (any) tvdLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
        }
        T.scnStep[s] = 1;
        switched[s] = 1;
        return;
    }
    // CONTROL: roundR1ToExtremeTapPosition on every enabled controller
    bool outOfBound = false;
    for (int g = 0; g < T.nG; g++) {
        size_t sg = (size_t) s * T.nG + g;
        bool useInit = T.grpUseInit[g] != 0;
        bool touched = false;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;
            size_t sm = (size_t) s * T.nM + m;
            if (T.memState[sm] != 1) continue;
            double r1 = x[sn + T.memRhoRow[m]];
            double lo = useInit ? T.grpShMin[sg] : T.memMin[m];
            double hi = useInit ? T.grpShMax[sg] : T.memMax[m];
            if (!(r1 < lo || r1 > hi)) continue;
            if (!useInit) {
                // plan-A6 remaining: clamping the current CARRIER while another member stays continuous
                // re-picks the pilot-Q role mid-flight — the CPU's 1-MW DS knife edge; route the scenario
                bool isCarrier = true, otherAlive = false;
                for (int mm = T.grpPtr[g]; mm < T.grpPtr[g + 1]; mm++) {
                    if (mm == m || disabled[(size_t) s * nElements + T.memElem[mm]]
                            || T.memState[(size_t) s * T.nM + mm] != 1) continue;
                    if (mm < m) isCarrier = false;
                    otherAlive = true;
                }
                if (isCarrier && otherAlive && T.rejectCarrierClamp) {
                    T.reject[s] = 1;
                    return;
                }
            }
            double v = r1 > hi ? hi : lo;
            T.memRoundFrom[sm] = v;
            int tap;
            tvdRoundToTap(T, m, v, &tap);
            T.memTap[sm] = tap;
            T.memState[sm] = 2;
            touched = true;
        }
        if (touched) {
            tvdLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
            outOfBound = true;
        }
    }
    if (outOfBound) {
        switched[s] = 1;                                     // stays CONTROL; OLF: UNSTABLE
        return;
    }
    // none out of range: updateContinuousRatio (useInitialTapPosition remap) + roundVoltageRatios -> COMPLETE
    for (int g = 0; g < T.nG; g++) {
        size_t sg = (size_t) s * T.nG + g;
        bool useInit = T.grpUseInit[g] != 0;
        bool touched = false;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            if (disabled[(size_t) s * nElements + T.memElem[m]]) continue;
            size_t sm = (size_t) s * T.nM + m;
            if (T.memState[sm] != 1) continue;
            double v = x[sn + T.memRhoRow[m]];
            if (useInit) {
                double init = T.grpShInit[sg];
                v = v >= init
                        ? T.memInit[m] + (v - init) * (T.memMax[m] - T.memInit[m]) / (T.grpShMax[sg] - init)
                        : T.memInit[m] - (init - v) * (T.memInit[m] - T.memMin[m]) / (init - T.grpShMin[sg]);
            }
            T.memRoundFrom[sm] = v;
            int tap;
            tvdRoundToTap(T, m, v, &tap);
            T.memTap[sm] = tap;
            T.memState[sm] = 2;
            touched = true;
        }
        if (touched) tvdLayoutGroup(T, s, n, nElements, g, disabled, mode, target, x);
    }
    // enableGeneratorVoltageControlsUnderMaxControlledNominalVoltage: the frozen controllers resume
    for (int c = 0; c < T.nFz; c++) {
        size_t sc = (size_t) s * T.nFz + c;
        if (!T.fzFrozen[sc]) continue;
        size_t sr = sn + T.fzVRow[c];
        mode[sr] = T.fzSavedMode[sc];
        target[sr] = T.fzSavedTarget[sc];
        if (T.fzRlEntry[c] >= 0) {                           // a remote member rejoins the DISTR_Q shares
            T.rlLimitType[(size_t) s * T.nrl + T.fzRlEntry[c]] = 0;
        }
        T.fzFrozen[sc] = 0;
    }
    T.scnStep[s] = 2;
    switched[s] = 1;                                         // OLF controlStep: UNSTABLE in any case (the COMPLETE solve)
}

struct SvdTables {
    int nG, nM, nDelta, nSubs;
    const int* grpPtr;
    const int* deltaPtr; const int* deltaRow; const int* deltaModeOn; const int* deltaModeOff;
    const double* deltaTargetOn; const double* deltaTargetOff;
    const int* memBRow; const int* memElem; const int* memSubPtr;
    const int* secPtr; const double* secB; const int* subMin; const int* subBase;
    int* grpStep; int* memState; double* memContB; int* subPos; double* saved;
};

// A shunt group is live when none of its (packed) member shunts is outaged — such contingencies are
// CPU-routed at qualification anyway (a voltage-controlling shunt outage never qualifies).
__device__ inline bool svdGroupLive(const SvdTables& T, int s, int nElements, int g, const char* disabled) {
    for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
        if (T.memElem[m] >= 0 && disabled[(size_t) s * nElements + T.memElem[m]]) return false;
    }
    return true;
}

// ON: every captured delta (the pilot's V identity, the member SHUNT_B power rows — pilot Q relocation and
// DISTR_SHUNT_B — as the toggled equation system produced them); OFF: the pilot back to its saved/OFF
// target, every member's SHUNT_B identity pinned at its dispatched section total.
__device__ inline void svdLayoutGroup(const SvdTables& T, int s, int n, int g, bool on,
                                      int* mode, double* target, double* x) {
    size_t sn = (size_t) s * n;
    for (int k = T.deltaPtr[g]; k < T.deltaPtr[g + 1]; k++) {
        int r = T.deltaRow[k];
        int member = -1;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) if (T.memBRow[m] == r) member = m;
        size_t sk = (size_t) s * T.nDelta + k;
        if (on) {
            if (member < 0 && isnan(T.saved[sk])) T.saved[sk] = target[sn + r];
            mode[sn + r] = T.deltaModeOn[k];
            target[sn + r] = T.deltaTargetOn[k];
        } else if (member < 0) {
            mode[sn + r] = T.deltaModeOff[k];
            target[sn + r] = isnan(T.saved[sk]) ? T.deltaTargetOff[k] : T.saved[sk];
            T.saved[sk] = nan("");
        }
    }
    if (!on) {
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            if (T.memState[(size_t) s * T.nM + m] != 2) continue;
            double bSum = 0.0;
            for (int u = T.memSubPtr[m]; u < T.memSubPtr[m + 1]; u++) bSum += T.secB[T.secPtr[u] + T.subPos[(size_t) s * T.nSubs + u]];
            size_t br = sn + T.memBRow[m];
            mode[br] = 0;
            target[br] = bSum;
            x[br] = bSum;
        }
    }
}

__global__ void svdEntryBatch(SvdTables T, int nScenarios, int n, int nElements, int entryOn,
                              const char* disabled, int* mode, double* target, double* x) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    size_t sn = (size_t) s * n;
    for (int g = 0; g < T.nG; g++) {
        if (!svdGroupLive(T, s, nElements, g, disabled)) continue;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            T.memState[(size_t) s * T.nM + m] = entryOn ? (mode[sn + T.memBRow[m]] == 1 ? 1 : 0) : 1;
        }
        if (!entryOn) svdLayoutGroup(T, s, n, g, true, mode, target, x);
    }
}

// The loop's first check: OLF disables every controller shunt and dispatches its continuous b to the
// closest sections (LfShuntImpl.dispatchB: per sub-controller, residue / remaining to the closest section
// from the CURRENT position over [minPosition, size), strictly closer only), UNSTABLE.
__global__ void svdCheckBatch(SvdTables T, int nScenarios, int n, int nElements, const char* disabled, const char* rlActive,
                              int* mode, double* target, double* x, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !rlActive[s]) return;
    size_t sn = (size_t) s * n;
    bool anyCtl = false;
    for (int g = 0; g < T.nG; g++) {
        size_t sg = (size_t) s * T.nG + g;
        if (T.grpStep[sg] != 0) continue;
        T.grpStep[sg] = 2;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            if (T.memElem[m] < 0 || !disabled[(size_t) s * nElements + T.memElem[m]]) anyCtl = true;
        }
        if (!svdGroupLive(T, s, nElements, g, disabled)) continue;
        bool touched = false;
        for (int m = T.grpPtr[g]; m < T.grpPtr[g + 1]; m++) {
            size_t sm = (size_t) s * T.nM + m;
            if (T.memState[sm] != 1) continue;
            double residue = x[sn + T.memBRow[m]];
            T.memContB[sm] = residue;
            int remaining = T.memSubPtr[m + 1] - T.memSubPtr[m];
            for (int u = T.memSubPtr[m]; u < T.memSubPtr[m + 1]; u++) {
                double toDispatch = residue / remaining--;
                int p0 = T.secPtr[u];
                int nSec = T.secPtr[u + 1] - p0;
                int best = T.subBase[u];
                double smallest = fabs(toDispatch - T.secB[p0 + best]);
                for (int p = T.subMin[u]; p < nSec; p++) {
                    double dist = fabs(toDispatch - T.secB[p0 + p]);
                    if (dist < smallest) {
                        best = p;
                        smallest = dist;
                    }
                }
                T.subPos[(size_t) s * T.nSubs + u] = best;
                residue -= T.secB[p0 + best];
            }
            T.memState[sm] = 2;
            touched = true;
        }
        if (touched) svdLayoutGroup(T, s, n, g, false, mode, target, x);
    }
    if (anyCtl) switched[s] = 1;                             // OLF: UNSTABLE whenever a live controller shunt exists
}

static TvdTables tvdTables(BatchSolverContext* ctx) {
    TvdTables T;
    T.nG = ctx->nTvdGroups; T.nM = ctx->nTvd; T.nDelta = ctx->nTvdDelta; T.mode = ctx->tvdMode;
    T.rejectCarrierClamp = std::getenv("OLF_GPU_TVD_REJECT_CARRIER_CLAMP") != nullptr ? 1 : 0;
    T.grpPtr = ctx->dTvdGrpPtr.as<int>(); T.grpVRow = ctx->dTvdGrpVRow.as<int>();
    T.grpTargetV = ctx->dTvdGrpTargetV.as<double>(); T.grpHalfDb = ctx->dTvdGrpHalfDb.as<double>(); T.grpUseInit = ctx->dTvdGrpUseInit.as<int>();
    T.deltaPtr = ctx->dTvdDeltaPtr.as<int>(); T.deltaRow = ctx->dTvdDeltaRow.as<int>();
    T.deltaModeOn = ctx->dTvdDeltaModeOn.as<int>(); T.deltaModeOff = ctx->dTvdDeltaModeOff.as<int>();
    T.deltaTargetOn = ctx->dTvdDeltaTargetOn.as<double>(); T.deltaTargetOff = ctx->dTvdDeltaTargetOff.as<double>();
    T.memRhoRow = ctx->dTvdMemRhoRow.as<int>(); T.memElem = ctx->dTvdMemElem.as<int>();
    T.memBaseTap = ctx->dTvdMemBaseTap.as<int>(); T.memTapPtr = ctx->dTvdMemTapPtr.as<int>();
    T.memMin = ctx->dTvdMemMin.as<double>(); T.memMax = ctx->dTvdMemMax.as<double>(); T.memInit = ctx->dTvdMemInit.as<double>();
    T.tapData = ctx->dTvdTapData.as<double>();
    T.scnStep = ctx->dTvdScnStep.as<int>(); T.grpStep = ctx->dTvdGrpStep.as<int>();
    T.grpShMin = ctx->dTvdGrpShMin.as<double>(); T.grpShMax = ctx->dTvdGrpShMax.as<double>(); T.grpShInit = ctx->dTvdGrpShInit.as<double>();
    T.memState = ctx->dTvdMemState.as<int>(); T.memTap = ctx->dTvdMemTap.as<int>(); T.memRoundFrom = ctx->dTvdMemRoundFrom.as<double>();
    T.saved = ctx->dTvdSaved.as<double>(); T.reject = ctx->dTvdReject.as<char>();
    T.nFz = ctx->nTvdFz; T.hasFix = ctx->tvdFixTables ? 1 : 0; T.fixWords = ctx->tvdFixWords;
    T.fzVRow = ctx->dTvdFzVRow.as<int>(); T.fzLoadQ = ctx->dTvdFzLoadQ.as<double>();
    T.fzRlEntry = ctx->dTvdFzRlEntry.as<int>(); T.nrl = ctx->nrl; T.rlLimitType = ctx->dRlLimitType.as<char>();
    T.fixPvRow = ctx->dTvdFixPvRow.as<int>();
    T.fixMinMask = ctx->dTvdFixMinMask.as<int>(); T.fixMaxMask = ctx->dTvdFixMaxMask.as<int>();
    T.fzFrozen = ctx->dTvdFzFrozen.as<int>(); T.fzSavedMode = ctx->dTvdFzSavedMode.as<int>(); T.fzSavedTarget = ctx->dTvdFzSavedTarget.as<double>();
    return T;
}

static SvdTables svdTables(BatchSolverContext* ctx) {
    SvdTables T;
    T.nG = ctx->nSvdGroups; T.nM = ctx->nSvd; T.nDelta = ctx->nSvdDelta; T.nSubs = ctx->nSvdSubs;
    T.grpPtr = ctx->dSvdGrpPtr.as<int>();
    T.deltaPtr = ctx->dSvdDeltaPtr.as<int>(); T.deltaRow = ctx->dSvdDeltaRow.as<int>();
    T.deltaModeOn = ctx->dSvdDeltaModeOn.as<int>(); T.deltaModeOff = ctx->dSvdDeltaModeOff.as<int>();
    T.deltaTargetOn = ctx->dSvdDeltaTargetOn.as<double>(); T.deltaTargetOff = ctx->dSvdDeltaTargetOff.as<double>();
    T.memBRow = ctx->dSvdMemBRow.as<int>(); T.memElem = ctx->dSvdMemElem.as<int>(); T.memSubPtr = ctx->dSvdMemSubPtr.as<int>();
    T.secPtr = ctx->dSvdSecPtr.as<int>(); T.secB = ctx->dSvdSecB.as<double>(); T.subMin = ctx->dSvdSubMin.as<int>(); T.subBase = ctx->dSvdSubBase.as<int>();
    T.grpStep = ctx->dSvdGrpStep.as<int>(); T.memState = ctx->dSvdMemState.as<int>(); T.memContB = ctx->dSvdMemContB.as<double>();
    T.subPos = ctx->dSvdSubPos.as<int>(); T.saved = ctx->dSvdSaved.as<double>();
    return T;
}
// ======================== END DISCRETE VOLTAGE CONTROL KERNELS ========================

// ======================== SECONDARY VOLTAGE CONTROL KERNELS (LOOP_SVC_KERNEL, OLF order 4) ========================
// OLF's SecondaryVoltageControlOuterLoop, per (scenario, zone), with nothing left on the host. The
// observables are the SAME quantities OLF's EquationTerms compute, reached the device's way: a
// controller bus's reactive injection is the sum of its CLOSED-BRANCH side flows (gradient from the
// codegen'd evalClosedBranch, exactly as flcGatherSensiBatch does it) plus its shunts' q = -b*v^2
// (ShuntCompensatorReactiveFlowEquationTerm: dq/dv = -2bv, dq/db = -v^2), and the pilot's calculated V
// is the BUS_V variable itself, so its sensitivity is dx at the pilot row.
constexpr double SVC_MAT_BUDGET_BYTES = 512.0 * 1024 * 1024;   // zone-system workspace ceiling
constexpr double SVC_DV_EPS = 1e-4;                          // OLF DV_EPS (0.1 kV)
constexpr double SVC_DK_DIFF_MAX_EPS = 1e-3;                 // OLF DK_DIFF_MAX_EPS (1 MVar)

struct SvcTables {
    int nCtl, nZones, nrl;
    const int* ctldVRow; const int* busVRow; const int* rlIdx;
    const double* baseTargetV; const double* minQ; const double* maxQ; const double* loadQ;
    const int* qPtr; const int* qKind; const int* qElem; const int* qBRow;
    const int* zonePtr; const int* zoneCtl; const int* zonePilotRow; const double* zoneTargetV;
    double* targetV; double* k0; double* jk; double* jvpp; double* rlVt;
    char* enabled; char* need; char* reject;
    double minPlausibleV, maxPlausibleV;
};

__device__ inline double svcQToK(double q, double minQ, double maxQ) {
    return (2.0 * q - maxQ - minQ) / (maxQ - minQ);          // OLF qToK
}

// OLF check(): tryToReEnableHelpfulControllerBuses FIRST (a zone with SOME but not all controllers at a
// reactive limit gives every limited one back its voltage control), then the per-zone trigger on the
// pilot deadband and the K spread. The re-enable runs whether or not the zone then adjusts — OLF mutates
// the network before it decides.
__global__ void svcTriggerBatch(SvcTables T, int nScenarios, int n, const char* rlActive,
                                const double* qBuf, const double* x, char* limitType, const char* savedMode,
                                int* mode, double* target, int svcDebug) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nZones * nScenarios) return;
    int z = tid % T.nZones;
    int s = tid / T.nZones;
    if (!rlActive[s]) return;
    int lo = T.zonePtr[z];
    int hi = T.zonePtr[z + 1];
    // ---- tryToReEnableHelpfulControllerBuses ----
    int nPinned = 0;
    for (int k = lo; k < hi; k++) {
        if (limitType[(size_t) s * T.nrl + T.rlIdx[T.zoneCtl[k]]] != 0) nPinned++;
    }
    if (nPinned > 0 && nPinned < hi - lo) {
        for (int k = lo; k < hi; k++) {
            int c = T.zoneCtl[k];
            size_t st = (size_t) s * T.nrl + T.rlIdx[c];
            if (limitType[st] == 0) continue;
            size_t idx = (size_t) s * n + T.busVRow[c];
            char sm = savedMode[st];
            mode[idx] = (sm == 2) ? 2 : 0;                   // back to its PV form
            target[idx] = T.targetV[(size_t) s * T.nCtl + c];
            limitType[st] = 0;
        }
    }
    // ---- the zone's trigger: pilot deadband, or K spread while not all controllers are at a limit ----
    double pilotDv = T.zoneTargetV[z] - x[(size_t) s * n + T.zonePilotRow[z]];
    double kMin = 1e300;
    double kMax = -1e300;
    bool allAbove = true;
    bool allBelow = true;
    for (int k = lo; k < hi; k++) {
        int c = T.zoneCtl[k];
        double q = qBuf[(size_t) s * n + T.busVRow[c]] + T.loadQ[c];
        double kk = svcQToK(q, T.minQ[c], T.maxQ[c]);
        T.k0[(size_t) s * T.nCtl + c] = kk;
        kMin = fmin(kMin, kk);
        kMax = fmax(kMax, kk);
        allAbove &= kk > 1.0;
        allBelow &= kk < -1.0;
        T.enabled[(size_t) s * T.nCtl + c] = limitType[(size_t) s * T.nrl + T.rlIdx[c]] == 0 ? 1 : 0;
    }
    // OLF filters the zone OUT of the check when it has no controller whose voltage control is enabled
    // (findAnyControlledBusWithAtLeastOneControllerBusWithVoltageControlEnabled): every controller at a
    // reactive limit means the zone cannot act, so it is not even looked at. Without this the pilot
    // deadband still flags the scenario, and the sensitivity solve that follows adjusts nothing while
    // leaving the driver's factorization state disturbed — 1.0e-10 kV on ieee14's z1.
    int nEnabled = 0;
    for (int k = lo; k < hi; k++) {
        if (T.enabled[(size_t) s * T.nCtl + T.zoneCtl[k]]) nEnabled++;
    }
    bool allAtLimits = allAbove || allBelow;
    if (nEnabled > 0 && (fabs(pilotDv) > SVC_DV_EPS || (!allAtLimits && fabs(kMax - kMin) > SVC_DK_DIFF_MAX_EPS))) {
        T.need[s] = 1;
    }
}

// After the unit-RHS solve for CONTROLLED bus i: OLF's calculateSensiK for every controller j — the sum
// over j's own Q observables of (term gradient . dx_i), scaled by 2/(maxQ-minQ) — and calculateSensiVpp,
// which is dx_i at the zone's pilot row.
__global__ void svcGatherBatch(SvcTables T, int nScenarios, int n, int i, int nElements, int cbOffset, int shOffset,
                               const double* cbIn, const int* cbIx, const double* shIn, const int* shIx,
                               int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                               int nShOv, const int* shOvSlotElem, const double* shOvGb,
                               const char* disabled, const double* x, const double* dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nCtl * nScenarios) return;
    int j = tid % T.nCtl;
    int s = tid / T.nCtl;
    const double* xs = x + (size_t) s * n;
    const double* dxs = dx + (size_t) s * n;
    double sq = 0.0;
    for (int k = T.qPtr[j]; k < T.qPtr[j + 1]; k++) {
        int elem = T.qElem[k];
        // NO isDisabled test here, deliberately: OLF's calculateSensiQ sums the controller bus's branch
        // terms as they stand, an outaged one included. Replicating the quirk keeps us on its trajectory.
        if (T.qKind[k] == 2) {                               // shunt: q = -b v^2
            int e = elem - shOffset;
            if (e < 0) continue;
            double g = shIn[e * 2];
            double b = shIn[e * 2 + 1];
            applyShuntOverride(nShOv, shOvSlotElem, shOvGb, s, e, g, b);
            int vCol = shIx[e * 5 + 2];
            int bCol = shIx[e * 5 + 3];                      // SHUNT_B row when b is a solved variable
            if (bCol >= 0) b = xs[bCol];
            double v = xs[vCol];
            sq += (-2.0 * b * v) * dxs[vCol] + (bCol >= 0 ? (-v * v) * dxs[bCol] : 0.0);
        } else {
            int e = elem - cbOffset;
            if (e < 0) continue;
            const double* p = cbIn + e * 8;
            double pOv[8];
            if (findTapOverride(nR1Ov, r1OvSlotElem, r1OvVal, s, e, pOv)) p = pOv;
            const int* q = cbIx + e * 12;
            double g[18];
            flcFlowAndGrad(T.qKind[k] == 0 ? 1 : 2, 0, p, xs, q, g);   // kind 0 of flcFlowAndGrad = REACTIVE flow
            double d[6] = { dxs[q[5]], dxs[q[7]], dxs[q[4]], dxs[q[6]],
                            q[8] >= 0 ? dxs[q[8]] : 0.0, q[9] >= 0 ? dxs[q[9]] : 0.0 };
            for (int m = 0; m < 6; m++) sq += g[m] * d[m];
        }
    }
    T.jk[((size_t) s * T.nCtl + i) * T.nCtl + j] = 2.0 * sq / (T.maxQ[j] - T.minQ[j]);
    if (j == 0) {
        // the pilot sensitivity of controlled bus i, for whichever zone i belongs to
        for (int z = 0; z < T.nZones; z++) {
            for (int k = T.zonePtr[z]; k < T.zonePtr[z + 1]; k++) {
                if (T.zoneCtl[k] == i) {
                    T.jvpp[(size_t) s * T.nCtl + i] = dxs[T.zonePilotRow[z]];
                }
            }
        }
    }
}

// OLF processSecondaryVoltageControl's zone system, ASSEMBLED per scenario into device memory for a
// batched dense LU. OLF builds one system over every ENABLED controller of every zone — A is block
// diagonal by zone but Jk is full, since a controller in one zone moves a controlled bus in another
// through the network — so the size is the total controller count, not a per-zone one. A DISABLED
// controller (at a reactive limit) is padded out with an identity row and a zero right-hand side,
// which leaves the enabled block exactly the system OLF solves and gives the disabled one dv = 0.
// Column-major, because that is what cuBLAS reads.
__global__ void svcAssembleBatch(SvcTables T, int nScenarios, int n, const char* rlActive,
                                 const int* ctlZone, const double* x, double* mat, double* rhs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nCtl * nScenarios) return;
    int i = tid % T.nCtl;
    int s = tid / T.nCtl;
    int nC = T.nCtl;
    double* m = mat + (size_t) s * nC * nC;
    double* r = rhs + (size_t) s * nC;
    bool live = rlActive[s] && T.need[s];
    if (!live || !T.enabled[(size_t) s * nC + i]) {
        for (int j = 0; j < nC; j++) {
            m[(size_t) j * nC + i] = (i == j) ? 1.0 : 0.0;   // identity row: this controller does not move
        }
        r[i] = 0.0;
        return;
    }
    int z = ctlZone[i];
    int lo = T.zonePtr[z];
    int hi = T.zonePtr[z + 1];
    int cnt = 0;
    int last = -1;
    for (int k = lo; k < hi; k++) {
        int c = T.zoneCtl[k];
        if (T.enabled[(size_t) s * nC + c]) {
            cnt++;
            last = c;
        }
    }
    if (i == last) {                                         // the zone's LAST controlled bus carries the pilot row
        for (int j = 0; j < nC; j++) {
            m[(size_t) j * nC + i] = T.enabled[(size_t) s * nC + j] ? T.jvpp[(size_t) s * nC + j] : 0.0;
        }
        r[i] = T.zoneTargetV[z] - x[(size_t) s * n + T.zonePilotRow[z]];
        return;
    }
    for (int j = 0; j < nC; j++) {                           // b = A . Jk^T, restricted to enabled columns
        if (!T.enabled[(size_t) s * nC + j]) {
            m[(size_t) j * nC + i] = 0.0;
            continue;
        }
        double v = 0.0;
        for (int k = lo; k < hi; k++) {
            int c = T.zoneCtl[k];
            if (!T.enabled[(size_t) s * nC + c]) continue;
            double a = (c == i) ? 1.0 - 1.0 / cnt : -1.0 / cnt;
            v += a * T.jk[((size_t) s * nC + j) * nC + c];
        }
        m[(size_t) j * nC + i] = v;
    }
    double rv = 0.0;                                         // rhs = -A . k0
    for (int k = lo; k < hi; k++) {
        int c = T.zoneCtl[k];
        if (!T.enabled[(size_t) s * nC + c]) continue;
        double a = (c == i) ? 1.0 - 1.0 / cnt : -1.0 / cnt;
        rv += a * T.k0[(size_t) s * nC + c];
    }
    r[i] = -rv;
}

// The pointer arrays cuBLAS's batched API takes, one entry per scenario.
__global__ void svcPointersBatch(int nScenarios, int nC, double* mat, double* rhs,
                                 double** matPtr, double** rhsPtr) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios) return;
    matPtr[s] = mat + (size_t) s * nC * nC;
    rhsPtr[s] = rhs + (size_t) s * nC;
}

// After the batched solve: OLF's plausibility test on the new targets, then commit. A singular zone
// system (cuBLAS info != 0) or an implausible target sends the scenario to the CPU, which is where
// OLF's own MatrixException / FAILED verdict would land it.
__global__ void svcCommitBatch(SvcTables T, int nScenarios, const char* rlActive,
                               const int* info, const double* rhs, char* switched) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nScenarios || !rlActive[s] || !T.need[s]) return;
    int nC = T.nCtl;
    if (info[s] != 0) {
        T.reject[s] = 1;
        return;
    }
    const double* dv = rhs + (size_t) s * nC;
    for (int i = 0; i < nC; i++) {
        if (!T.enabled[(size_t) s * nC + i]) continue;
        double nt = T.targetV[(size_t) s * nC + i] + dv[i];
        if (!(nt >= T.minPlausibleV && nt <= T.maxPlausibleV)) {
            T.reject[s] = 1;
            return;
        }
    }
    bool moved = false;
    for (int i = 0; i < nC; i++) {
        if (!T.enabled[(size_t) s * nC + i]) continue;
        T.targetV[(size_t) s * nC + i] += dv[i];
        moved = true;
    }
    if (moved) {
        switched[s] = 1;                                     // OLF: UNSTABLE, the zone adjusted
    }
}

// Give the moved zone targets to the system: an UNPINNED controller's controlled-bus V identity takes the
// new target; a PINNED one keeps its Q pin but its switch-back must compare against the moved target.
__global__ void svcApplyBatch(SvcTables T, int nScenarios, int n, const char* rlActive,
                              const char* limitType, double* target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T.nCtl * nScenarios) return;
    int c = tid % T.nCtl;
    int s = tid / T.nCtl;
    if (!rlActive[s]) return;
    double tv = T.targetV[(size_t) s * T.nCtl + c];
    if (tv == T.baseTargetV[c]) return;
    T.rlVt[(size_t) s * T.nrl + T.rlIdx[c]] = tv;
    if (limitType[(size_t) s * T.nrl + T.rlIdx[c]] == 0) {
        target[(size_t) s * n + T.ctldVRow[c]] = tv;
    }
}

static SvcTables svcTables(BatchSolverContext* ctx) {
    SvcTables T;
    T.nCtl = ctx->nSvc; T.nZones = ctx->nSvcZones; T.nrl = ctx->nrl;
    T.ctldVRow = ctx->dSvcCtldVRow.as<int>(); T.busVRow = ctx->dSvcBusVRow.as<int>(); T.rlIdx = ctx->dSvcRlIdx.as<int>();
    T.baseTargetV = ctx->dSvcBaseTargetV.as<double>(); T.minQ = ctx->dSvcMinQ.as<double>();
    T.maxQ = ctx->dSvcMaxQ.as<double>(); T.loadQ = ctx->dSvcLoadQ.as<double>();
    T.qPtr = ctx->dSvcQPtr.as<int>(); T.qKind = ctx->dSvcQKind.as<int>();
    T.qElem = ctx->dSvcQElem.as<int>(); T.qBRow = ctx->dSvcQBRow.as<int>();
    T.zonePtr = ctx->dSvcZonePtr.as<int>(); T.zoneCtl = ctx->dSvcZoneCtl.as<int>();
    T.zonePilotRow = ctx->dSvcZonePilotRow.as<int>(); T.zoneTargetV = ctx->dSvcZoneTargetV.as<double>();
    T.targetV = ctx->dSvcTargetV.as<double>(); T.k0 = ctx->dSvcK0.as<double>();
    T.jk = ctx->dSvcJk.as<double>(); T.jvpp = ctx->dSvcJvpp.as<double>(); T.rlVt = ctx->dSvcRlVt.as<double>();
    T.enabled = ctx->dSvcEnabled.as<char>(); T.need = ctx->dSvcNeed.as<char>(); T.reject = ctx->dSvcReject.as<char>();
    T.minPlausibleV = ctx->svcMinPlausibleV; T.maxPlausibleV = ctx->svcMaxPlausibleV;
    return T;
}
// ======================== END SECONDARY VOLTAGE CONTROL KERNELS ========================


BatchResult solveBatchContext(BatchSolverContext* ctx, const double* x0, const double* target,
                              const int* rowMode, const int* baseRowMode, const char* disabled,
                              int nOv, const int* ovIndex, const double* ovValue,
                              const int* dsOvBus, const double* dsOvFactorDelta, const double* dsOvBaseDelta,
                              const double* dsOvMaxDelta, const double* dsOvMinDelta, const double* dsAnchorShift,
                              const double* dsSlackShift,
                              int nRlVtOv, const int* rlVtOvSlotIdx, const double* rlVtOvVal,
                              int nRlQOv, const int* rlQOvSlotEntry, const double* rlQOvMinMax,
                              int nCvOv, const int* cvOvSlotEntry, const double* cvOvStepScale,
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
    // RESUME: the driver is continuing from a LOOP_HOST pause, so x/targets/modes already hold the
    // sequence's current state on the device — re-uploading the base rows would throw away the
    // distributed slack, the HVDC deltas and the monitoring activations it has accumulated.
    const bool resume = ctx->driverResume != 0;
    ctx->driverResume = 0;
    ctx->driverPaused = 0;
    if (resume && (ctx->nTvc > 0 || ctx->nFlc > 0 || ctx->nCph > 0 || ctx->nTvd > 0)) {
        tvcRebuildOverrides(ctx);                            // uploadBatchOverrides just erased the moves
    }
    if (resume && ctx->nShc > 0) {
        shcRebuildOverrides(ctx);
    }
    dBaseMode.upload(baseRowMode, n * sizeof(int));         // base (contingency-free) row mode for the RL limit-type seed
    if (!resume) {
        if (x0Full != nullptr) {
            CUDA_CHECK(cudaMemcpy(ctx->dX.p, x0Full, (size_t) nTot * sizeof(double), cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy(ctx->dX.p, x0, n * sizeof(double), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(ctx->dTarget.p, target, n * sizeof(double), cudaMemcpyHostToDevice));
        // The BASE targets, kept aside before tiling and the per-scenario overrides: the pilot-Q
        // capture below reads the base carrier's row, and a scenario may have overridden that row
        // (a member islanded by a bus loss freezes it to a flat voltage identity). Capturing from the
        // base keeps the captured value the pilot's reactive target whatever the scenario did to the
        // row. (The carrier's target itself is stamped host-side when the base carrier is frozen -
        // GpuBatchChunkSolver - because the device stamp runs only inside the outer-loop passes.)
        if (!ctx->dTargetBase.p) ctx->dTargetBase.alloc((size_t) n * sizeof(double));
        CUDA_CHECK(cudaMemcpy(ctx->dTargetBase.p, target, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->dMode.p, rowMode, nTot * sizeof(int), cudaMemcpyHostToDevice));
        if (x0Full == nullptr) {
            tileBaseRow<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, n, ctx->dX.as<double>());
        }
        tileBaseRow<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, n, ctx->dTarget.as<double>());
        if (nOv > 0) {                                       // per-scenario target overrides (generator loss)
            dOvIndex.upload(ovIndex, nOv * sizeof(int));
            dOvValue.upload(ovValue, nOv * sizeof(double));
            applyTargetOverrideBatch<<<grid(nOv), BLOCK, 0, ctx->stream>>>(nOv, dOvIndex.as<int>(), dOvValue.as<double>(), ctx->dTarget.as<double>());
        }
    }
    if (hasDsOverride) {                                     // per-scenario generator-loss DS reweight
        dDsOvBus.upload(dsOvBus, nS * sizeof(int));
        dDsOvFactorDelta.upload(dsOvFactorDelta, nS * sizeof(double));
        dDsOvBaseDelta.upload(dsOvBaseDelta, nS * sizeof(double));
        dDsOvMaxDelta.upload(dsOvMaxDelta, nS * sizeof(double));
        dDsOvMinDelta.upload(dsOvMinDelta, nS * sizeof(double));
    }
    if (!resume) {
        // NOT on a resume: the HVDC AC-emulation driver phases MUTATE element bits on the device during
        // the paused sweep (a frozen or saturated emulation has its coupling deactivated), and the host's
        // mirror of the mask is the pre-solve one. Re-uploading it mid-sequence silently undoes those
        // loops' work — measured on a 400-contingency rte6515 slice: 5 differing violations against 1,
        // with NO host decision involved at all (the split alone did it). Contingency outages do not
        // change during a sweep, so the device copy stays authoritative across the pause.
        CUDA_CHECK(cudaMemcpy(ctx->dDisabled.p, disabled,
                              static_cast<size_t>(ctx->nElements) * nS * sizeof(char), cudaMemcpyHostToDevice));
    }

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
    // STRAGGLER EARLY-EXIT (tol > 0 only, OFF by default): once the active count falls to
    // <= frac*nS (min 1), stop the batched loop and mark the remainder STATUS_REJECTED (-2) so the
    // Java side routes them to the per-contingency fallback. Rationale (2026-07-17 tol>0 profile):
    // retirement does NOT shrink cuDSS UBATCH cost — 499 retired identity systems still factorize
    // at the full ~233 ms/iteration — so one never-converging scenario makes the whole chunk pay
    // maxIter full-width iterations per outer pass. Off since 2026-09-22: a scenario that exhausts
    // the budget is reported as MAX_ITERATION_REACHED by the device itself (the precomputer's
    // exhausted set), which is OLF's own answer, and the replay it replaced cost about what the
    // extra iterations do (rte6515, 500 busbar sections: +1.2 s of 16 warm, 5/5 statuses agree).
    // OLF_GPU_STRAGGLER_FRAC=<frac> re-enables the exit as a perf switch.
    int stragglerExit = 0;
    if (!deviceFixedIter) {
        double stragglerFrac = 0.0;
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
    bool fullLoadflow = !deviceFixedIter && (ctx->nrl > 0 || ctx->nds > 0 || ctx->dsSlackOnly
                                             || ctx->nmon > 0 || ctx->nTvc > 0 || ctx->nShc > 0 || ctx->nFlc > 0 || ctx->nCph > 0
                                             || ctx->nTvd > 0 || ctx->nSvd > 0 || ctx->nSvc > 0);
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
    // The reactive-limit limits become PER-SCENARIO as soon as either channel is live: curve limits at
    // the runtime P, or a generator lost at a bus that keeps voltage control taking its capability away.
    DeviceBuf dRlQOvSlotEntry, dRlQOvMinMax;
    if (nRlQOv > 0) {
        dRlQOvSlotEntry.upload(rlQOvSlotEntry, static_cast<size_t>(nRlQOv) * 2 * sizeof(int));
        dRlQOvMinMax.upload(rlQOvMinMax, static_cast<size_t>(nRlQOv) * 3 * sizeof(double));   // {dMinQ, dMaxQ, dLoadQ}
    }
    // Per-scenario curve repairs, CSR by scenario: the dynamic-limits kernel runs one thread per
    // (reactive-limit entry, scenario) and walks the curve entries inside, so a flat {slot, entry}
    // scan would be paid ncv times per thread. Bucketing by slot makes it a handful of comparisons.
    DeviceBuf dCvOvPtr, dCvOvEntry, dCvOvVal;
    if (nCvOv > 0) {
        std::vector<int> ptr(nS + 1, 0);
        for (int k = 0; k < nCvOv; k++) ptr[cvOvSlotEntry[2 * k] + 1]++;
        for (int t = 0; t < nS; t++) ptr[t + 1] += ptr[t];
        std::vector<int> cursor(ptr.begin(), ptr.end() - 1);
        std::vector<int> entry(nCvOv);
        std::vector<double> val(static_cast<size_t>(nCvOv) * 2);
        for (int k = 0; k < nCvOv; k++) {
            int at = cursor[cvOvSlotEntry[2 * k]]++;
            entry[at] = cvOvSlotEntry[2 * k + 1];
            val[2 * at] = cvOvStepScale[2 * k];
            val[2 * at + 1] = cvOvStepScale[2 * k + 1];
        }
        dCvOvPtr.upload(ptr.data(), ptr.size() * sizeof(int));
        dCvOvEntry.upload(entry.data(), entry.size() * sizeof(int));
        dCvOvVal.upload(val.data(), val.size() * sizeof(double));
    }
    bool useDynRlLimits = ctx->nrl > 0 && (ctx->ncv > 0 || nRlQOv > 0);
    if (useDynRlLimits && (!ctx->dRlMinQDyn.p || !ctx->dRlLoadQDyn.p)) {   // curves allocate these on their own
                                                            // table upload; a Q override alone must too, and a
                                                            // guard that names only ONE of them silently leaves
                                                            // the others null on a curve network (measured: the
                                                            // whole precompute fell back to the CPU)
        ctx->dRlLoadQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
        ctx->dRlMinQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
        ctx->dRlMaxQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    }
    int maxOuter = fullLoadflow ? 8 * RL_MAX_OUTER : 1;
    std::vector<char> rlActive(nS), rlSwitched(nS), wasActive(nS);
    // ...but NOT on a resume: these are per-SOLVE state, and a resumed driver is continuing the SAME
    // solve. Re-seeding here would discard the RL pins, monitoring activations and absorbed slack the
    // sequence has already established — which is exactly what makes a resume cheaper than the
    // orchestrator pattern it replaces: that one resets and re-runs the whole driver every round.
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr && nS == 1) {
        std::fprintf(stderr, "CTX_FLAGS nS=1 nrl=%d nRlRemote=%d hasDqQrGroups=%d nQr=%d ndq=%d nds=%d ngen=%d nzi=%d nziBase=%d nRmq=%d resume=%d fullLoadflow=%d\n",
                     ctx->nrl, ctx->nRlRemote, ctx->hasDqQrGroups ? 1 : 0, ctx->st->nQr, ctx->ndq, ctx->nds, ctx->ngen, ctx->nzi, ctx->nziBase, ctx->nRmq, resume ? 1 : 0, fullLoadflow ? 1 : 0);
    }
    if (fullLoadflow && !resume && ctx->nSvc > 0) {
        // BEFORE the reactive-limit seeding below, which consults it: no zone has moved a target yet.
        std::vector<double> nanv(static_cast<size_t>(nS) * std::max(ctx->nrl, 1), std::nan(""));
        CUDA_CHECK(cudaMemcpy(ctx->dSvcRlVt.p, nanv.data(), nanv.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (fullLoadflow && !resume) {                           // per-solve PV↔PQ state: 0 switches, limitType from base
        if (ctx->hasDqQrGroups && ctx->st->nQr > 0) {        // plan-B4: capture each group's pilot-Q target from
            size_t need = static_cast<size_t>(ctx->st->nQr) * nS * sizeof(double);   // the BASE carrier's row,
            if (!ctx->dQrPilotQ.p) ctx->dQrPilotQ.alloc(need);   // BEFORE any RL mode seeding rewrites it
            if (!ctx->dQrSavedVT.p) ctx->dQrSavedVT.alloc(need);
            if (!ctx->dQrVeqOff.p) ctx->dQrVeqOff.alloc(static_cast<size_t>(ctx->st->nQr) * nS * sizeof(char));
            CUDA_CHECK(cudaMemsetAsync(ctx->dQrVeqOff.p, 0,      // per-solve state, like limitType
                    static_cast<size_t>(ctx->st->nQr) * nS * sizeof(char), ctx->stream));
            capturePilotQBatch<<<grid(ctx->st->nQr * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nQr, nS, n,
                    ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>(), dBaseMode.as<int>(),
                    ctx->dTargetBase.as<double>(), ctx->dQrPilotQ.as<double>());
            if (nS == 1 && std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                int nQr = ctx->st->nQr;
                std::vector<int> ptr(nQr + 1);
                CUDA_CHECK(cudaMemcpy(ptr.data(), ctx->st->dQrPtr.p, ptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
                std::vector<int> rows(ptr[nQr]);
                CUDA_CHECK(cudaMemcpy(rows.data(), ctx->st->dQrRows.p, rows.size() * sizeof(int), cudaMemcpyDeviceToHost));
                std::vector<double> pq(nQr), tg(n);
                std::vector<int> md(n);
                CUDA_CHECK(cudaMemcpy(pq.data(), ctx->dQrPilotQ.p, nQr * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(tg.data(), ctx->dTarget.p, n * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(md.data(), dBaseMode.p, n * sizeof(int), cudaMemcpyDeviceToHost));
                for (int g = 0; g < nQr; g++) {
                    std::fprintf(stderr, "PILOTQ_CAPTURE group=%d captured=%.9f candidates:", g, pq[g]);
                    for (int c = ptr[g]; c < ptr[g + 1]; c++) {
                        std::fprintf(stderr, " row%d(mode=%d,target=%.9f)", rows[c], md[rows[c]], tg[rows[c]]);
                    }
                    std::fprintf(stderr, "\n");
                }
            }
        }
        if (rlStateIn != nullptr && ctx->nrl > 0) {
            // WARM CONTINUATION (operator strategies stage 2): seed each (controller, scenario)'s
            // pinned limit type AND switch count from a previous solve's exported state, so the
            // strategy scenario continues the contingency scenario's outer-loop trajectory exactly
            // like the CPU's per-contingency action solve does.
            std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS), sc(static_cast<size_t>(ctx->nrl) * nS);
            for (size_t k = 0; k < lt.size(); k++) {
                lt[k] = static_cast<char>(rlStateIn[k] & 7);
                sc[k] = static_cast<char>(rlStateIn[k] >> 3);
            }
            CUDA_CHECK(cudaMemcpyAsync(ctx->dRlLimitType.p, lt.data(), lt.size(), cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpy(ctx->dRlSwitchCount.p, sc.data(), sc.size(), cudaMemcpyHostToDevice));
            if (useDynRlLimits) {                            // curve limits at the CURRENT (seeded) targets,
                rlCurveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, ctx->ncv, n, ctx->cvExtrapolate,
                        ctx->dCvIdx.as<int>(), ctx->dCvIn.as<double>(), ctx->dCvPts.as<double>(),
                        ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(), ctx->dRlLoadQ.as<double>(),
                        ctx->dTarget.as<double>(),
                        nRlQOv, nRlQOv > 0 ? dRlQOvSlotEntry.as<int>() : nullptr,
                        nRlQOv > 0 ? dRlQOvMinMax.as<double>() : nullptr,
                        nCvOv > 0 ? dCvOvPtr.as<int>() : nullptr,
                        nCvOv > 0 ? dCvOvEntry.as<int>() : nullptr,
                        nCvOv > 0 ? dCvOvVal.as<double>() : nullptr,
                        ctx->ngen > 0 ? ctx->dGenCum.as<double>() : nullptr,
                        ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->ngen,
                        ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>(),
                        ctx->dRlLoadQDyn.as<double>());
            }
            seedRlModeFromState<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n,
                    ctx->dRlVrow.as<int>(),
                    useDynRlLimits ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                    useDynRlLimits ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                    useDynRlLimits ? ctx->dRlLoadQDyn.as<double>() : ctx->dRlLoadQ.as<double>(), ctx->dRlLimitType.as<char>(),
                    ctx->dMode.as<int>(), ctx->dTarget.as<double>(), useDynRlLimits ? 1 : 0,
                    ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr);
        } else {
            CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitchCount.p, 0, static_cast<size_t>(ctx->nrl) * nS * sizeof(char), ctx->stream));
            if (ctx->nrl > 0) {                              // seed the pinned limit for base-PQ controllers (else all PV)
                initRlLimitTypeBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n, ctx->dRlVrow.as<int>(),
                        ctx->dRlVtarget.as<double>(), ctx->dRlMonitored.as<char>(), dBaseMode.as<int>(), ctx->dX.as<double>(),
                        ctx->dRlLimitType.as<char>(),
                        nRlVtOv, nRlVtOv > 0 ? dRlVtOvSlotIdx.as<int>() : nullptr,
                        nRlVtOv > 0 ? dRlVtOvVal.as<double>() : nullptr,
                        ctx->nSvc > 0 ? ctx->dSvcRlVt.as<double>() : nullptr,
                        ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                        ctx->nRlRemote > 0 ? ctx->dRlCmpVrow.as<int>() : nullptr,
                        ctx->dTarget.as<double>(), ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(),
                        ctx->dRlLoadQ.as<double>(),
                        ctx->hasRlBaseLt ? ctx->dRlBaseLimitType.as<char>() : nullptr);
            }
        }
        if (ctx->nds > 0) {                                  // slack-retained share: 0 until frozen at the first pass
            CUDA_CHECK(cudaMemsetAsync(ctx->dDsRetain.p, 0, nS * sizeof(double), ctx->stream));
            {
                std::vector<double> nanv(nS, std::numeric_limits<double>::quiet_NaN());
                CUDA_CHECK(cudaMemcpyAsync(ctx->dDsRetainApplied.p, nanv.data(), nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            }
            if (ctx->ngen > 0 && ctx->dGenCum.p) {           // per-solve too: the write-back reads it even when
                CUDA_CHECK(cudaMemsetAsync(ctx->dGenCum.p, 0,   // the DS phase never fired (NaN targetP otherwise)
                        static_cast<size_t>(ctx->ngen) * nS * sizeof(double), ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(ctx->dGenCumValid.p, 0, (size_t) nS * sizeof(char), ctx->stream));
            }
        }
        if (ctx->nmon > 0) {                                 // per-SOLVE state, like the RL limit types:
            std::vector<double> nan(static_cast<size_t>(ctx->nmon) * nS, std::nan(""));   // nothing has
            CUDA_CHECK(cudaMemcpyAsync(ctx->dMonActivated.p, nan.data(),   // fired yet in this solve
                    nan.size() * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        }
        if (ctx->nTvc > 0) {                                 // per-SOLVE state: every scenario starts at the
            size_t nst = static_cast<size_t>(nS) * ctx->nTvc;   // base tap with a fresh ratchet
            std::vector<int> cur(nst);
            for (int s2 = 0; s2 < nS; s2++) {
                for (int c = 0; c < ctx->nTvc; c++) {
                    cur[static_cast<size_t>(s2) * ctx->nTvc + c] = ctx->hTvcBaseTap[c];
                }
            }
            ctx->dTvcPrevTap.ensure(nst * sizeof(int));
            CUDA_CHECK(cudaMemcpy(ctx->dTvcCurTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dTvcPrevTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(ctx->dTvcRatchetDir.p, 0, nst * sizeof(int)));
            CUDA_CHECK(cudaMemset(ctx->dTvcRatchetCount.p, 0, nst * sizeof(int)));
            CUDA_CHECK(cudaMemset(ctx->dTvcRatchetAllowed.p, 0, nst * sizeof(int)));
            ctx->dTvcOutOfDb.ensure(nst * sizeof(char));
            ctx->dTvcDiffV.ensure(nst * sizeof(double));
            ctx->dTvcSensi.ensure(nst * sizeof(double));
        }
        if (ctx->nCph > 0) {                                 // per-SOLVE state: base taps, check count 0
            size_t nst = static_cast<size_t>(nS) * ctx->nCph;
            std::vector<int> cur(nst);
            for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < ctx->nCph; c++) cur[static_cast<size_t>(s2) * ctx->nCph + c] = ctx->hCphBaseTap[c];
            CUDA_CHECK(cudaMemcpy(ctx->dCphTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dCphPrevTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(ctx->dCphIter.p, 0, nS * sizeof(int)));
        }
        if (ctx->nTvd > 0) {                                 // per-SOLVE state: every control at its entry state
            size_t nsm = static_cast<size_t>(nS) * ctx->nTvd;
            size_t nsg = static_cast<size_t>(nS) * ctx->nTvdGroups;
            size_t nsd = static_cast<size_t>(nS) * std::max(ctx->nTvdDelta, 1);
            std::vector<int> tap(nsm);
            for (int s2 = 0; s2 < nS; s2++) for (int m = 0; m < ctx->nTvd; m++) tap[static_cast<size_t>(s2) * ctx->nTvd + m] = ctx->hTvdMemBaseTap[m];
            std::vector<double> nanv(std::max(nsm, nsd), std::nan(""));
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(ctx->dTvdMemTap.p, tap.data(), nsm * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(ctx->dTvdMemState.p, 0, nsm * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(ctx->dTvdMemRoundFrom.p, nanv.data(), nsm * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dTvdSaved.p, nanv.data(), nsd * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(ctx->dTvdScnStep.p, 0, nS * sizeof(int)));
            CUDA_CHECK(cudaMemset(ctx->dTvdGrpStep.p, 0, nsg * sizeof(int)));
            CUDA_CHECK(cudaMemset(ctx->dTvdGrpShMin.p, 0, nsg * sizeof(double)));
            CUDA_CHECK(cudaMemset(ctx->dTvdGrpShMax.p, 0, nsg * sizeof(double)));
            CUDA_CHECK(cudaMemset(ctx->dTvdGrpShInit.p, 0, nsg * sizeof(double)));
            CUDA_CHECK(cudaMemset(ctx->dTvdReject.p, 0, nS * sizeof(char)));
            if (ctx->nTvdFz > 0) CUDA_CHECK(cudaMemset(ctx->dTvdFzFrozen.p, 0, static_cast<size_t>(nS) * ctx->nTvdFz * sizeof(int)));
            tvdEntryBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(tvdTables(ctx), nS, n, ctx->nElements, ctx->tvdEntryOn,
                    ctx->dDisabled.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        }
        if (ctx->nSvc > 0) {                                 // per-SOLVE state: zone targets back at the
            size_t nsc = static_cast<size_t>(nS) * ctx->nSvc;   // network's own, nothing adjusted yet
            std::vector<double> tv(nsc);
            for (int s2 = 0; s2 < nS; s2++) {
                for (int c = 0; c < ctx->nSvc; c++) tv[static_cast<size_t>(s2) * ctx->nSvc + c] = ctx->hSvcBaseTargetV[c];
            }
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(ctx->dSvcTargetV.p, tv.data(), nsc * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(ctx->dSvcNeed.p, 0, nS));
            CUDA_CHECK(cudaMemset(ctx->dSvcReject.p, 0, nS));
            CUDA_CHECK(cudaMemset(ctx->dSvcEnabled.p, 1, nsc));
        }
        if (ctx->nSvd > 0) {                                 // per-SOLVE state, same shape
            size_t nsm = static_cast<size_t>(nS) * ctx->nSvd;
            size_t nsg = static_cast<size_t>(nS) * ctx->nSvdGroups;
            size_t nsd = static_cast<size_t>(nS) * std::max(ctx->nSvdDelta, 1);
            size_t nsu = static_cast<size_t>(nS) * std::max(ctx->nSvdSubs, 1);
            std::vector<int> pos(nsu);
            for (int s2 = 0; s2 < nS; s2++) for (int u = 0; u < ctx->nSvdSubs; u++) pos[static_cast<size_t>(s2) * ctx->nSvdSubs + u] = 0;
            std::vector<double> nanv(std::max(nsm, nsd), std::nan(""));
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(ctx->dSvdSubPos.p, pos.data(), nsu * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(ctx->dSvdMemState.p, 0, nsm * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(ctx->dSvdMemContB.p, nanv.data(), nsm * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dSvdSaved.p, nanv.data(), nsd * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(ctx->dSvdGrpStep.p, 0, nsg * sizeof(int)));
            svdEntryBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(svdTables(ctx), nS, n, ctx->nElements, ctx->svdEntryOn,
                    ctx->dDisabled.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        }
        if (ctx->nFlc > 0) {                                 // per-SOLVE state: base taps, fresh ratchets
            size_t nst = static_cast<size_t>(nS) * ctx->nFlc;
            std::vector<int> cur(nst);
            for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < ctx->nFlc; c++) cur[static_cast<size_t>(s2) * ctx->nFlc + c] = ctx->hFlcBaseTap[c];
            CUDA_CHECK(cudaMemcpy(ctx->dFlcCurTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->dFlcPrevTap.p, cur.data(), nst * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(ctx->dFlcRatDir.p, 0, nst * sizeof(int)));
            CUDA_CHECK(cudaMemset(ctx->dFlcRatCount.p, 0, nst * sizeof(int)));
            CUDA_CHECK(cudaMemset(ctx->dFlcRatAllowed.p, 0, nst * sizeof(int)));
        }
        if (ctx->nShc > 0) {                                 // per-SOLVE state: base sections, fresh ratchets,
            size_t nsu = static_cast<size_t>(nS) * ctx->nShcSubs;   // the identity row's target as the base b
            std::vector<int> pos(nsu);
            for (int s2 = 0; s2 < nS; s2++) {
                for (int u = 0; u < ctx->nShcSubs; u++) pos[static_cast<size_t>(s2) * ctx->nShcSubs + u] = ctx->hShcSubBase[u];
            }
            CUDA_CHECK(cudaMemcpy(ctx->dShcPos.p, pos.data(), nsu * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(ctx->dShcRatDir.p, 0, nsu * sizeof(int)));
            CUDA_CHECK(cudaMemset(ctx->dShcRatCount.p, 0, nsu * sizeof(int)));
            CUDA_CHECK(cudaMemset(ctx->dShcRatAllowed.p, 0, nsu * sizeof(int)));
            size_t nsc = static_cast<size_t>(nS) * ctx->nShc;
            CUDA_CHECK(cudaMemset(ctx->dShcAppliedDb.p, 0, nsc * sizeof(double)));
            std::vector<double> tgt(static_cast<size_t>(n) * nS), bb(nsc);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(tgt.data(), ctx->dTarget.p, tgt.size() * sizeof(double), cudaMemcpyDeviceToHost));
            for (int s2 = 0; s2 < nS; s2++) {
                for (int c = 0; c < ctx->nShc; c++) bb[static_cast<size_t>(s2) * ctx->nShc + c] = tgt[static_cast<size_t>(s2) * n + ctx->hShcBRow[c]];
            }
            CUDA_CHECK(cudaMemcpy(ctx->dShcBaseB.p, bb.data(), nsc * sizeof(double), cudaMemcpyHostToDevice));
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
    // The outer loops this solve drives, IN OLF'S CONFIGURED ORDER. OLF's engine
    // (AcloadFlowEngine.runOuterLoops) walks its configured list in order, drives each loop to
    // STABLE — re-solving after every unstable check — then moves to the next, restarts from the
    // top, and stops when a full sweep changes nothing. The driver below already has exactly those
    // semantics; what it lacked was the LIST: it hard-coded two loops, so every other loop had to be
    // bolted on as a host orchestrator running AFTER the whole solve, landing out of order. Naming
    // the order explicitly is what lets the remaining loops move into their proper slot.
    //
    // OLF's contingency order: DistributedSlack(1), FreezingHvdcACEmulation(2),
    // AcHvdcAcEmulationLimits(3), SecondaryVoltageControl(5), MonitoringVoltage(6),
    // ReactiveLimits(7). All but SecondaryVoltageControl are driver phases here, in that order.
    // The loops OLF runs AFTER ReactiveLimits (PhaseControl, transformer and shunt voltage control)
    // are still host orchestrators: their position relative to RL is right, but OLF restarts the
    // WHOLE sweep whenever one of them changes something, so an earlier loop is re-checked and here
    // it is not. Folding them needs the discrete tap/section tables on the device.
    int loopOrder[MAX_DEVICE_LOOPS];
    int nLoops = 0;
    if (ctx->nds > 0 || ctx->dsSlackOnly) {
        loopOrder[nLoops++] = LOOP_DS;
    }
    if (ctx->nhv > 0) {
        loopOrder[nLoops++] = LOOP_HVDC_FREEZE;              // OLF: FreezingHvdcACEmulation
        if (ctx->hvLimitsActive) {
            loopOrder[nLoops++] = LOOP_HVDC_LIMITS;          // OLF: AcHvdcAcEmulationLimits
        }
    }
    if (ctx->nSvc > 0) {
        loopOrder[nLoops++] = LOOP_SVC_KERNEL;               // OLF order 4: SecondaryVoltageControl, IN-KERNEL
    } else if (ctx->hostLoopMask & (1 << 0)) {
        loopOrder[nLoops++] = LOOP_HOST_SVC;                 // OLF order 4: SecondaryVoltageControl, host pause
    }
    if (ctx->nmon > 0) {
        loopOrder[nLoops++] = LOOP_MONITORING;               // OLF order 5: MonitoringVoltage
    }
    if (ctx->nrl > 0) {
        loopOrder[nLoops++] = LOOP_RL;                       // OLF order 6: ReactiveLimits
    }
    if (ctx->nFlcPhase > 0 || ctx->nCph > 0) {
        loopOrder[nLoops++] = LOOP_PHASE_KERNEL;             // OLF order 7: (Incremental)PhaseControl, IN-KERNEL
    } else if (ctx->hostLoopMask & (1 << 1)) {
        loopOrder[nLoops++] = LOOP_HOST_PHASE;               // OLF order 7: PhaseControl
    }
    if (ctx->nTvc > 0) {
        loopOrder[nLoops++] = LOOP_TVC_KERNEL;               // OLF order 8: TransformerVoltageControl, IN-KERNEL
    } else if (ctx->nTvd > 0) {
        loopOrder[nLoops++] = LOOP_TVD_KERNEL;               // OLF order 8: AFTER_GENERATOR / Simple TVC, IN-KERNEL
    } else if (ctx->hostLoopMask & (1 << 2)) {
        loopOrder[nLoops++] = LOOP_HOST_TVC;                 // OLF order 8: TransformerVoltageControl, host pause
    }
    if (ctx->nFlcRpc > 0) {
        loopOrder[nLoops++] = LOOP_TRPC_KERNEL;              // OLF order 9: TransformerReactivePowerControl, IN-KERNEL
    } else if (ctx->hostLoopMask & (1 << 3)) {
        loopOrder[nLoops++] = LOOP_HOST_TRPC;                // OLF order 9: TransformerReactivePowerControl
    }
    if (ctx->nShc > 0) {
        loopOrder[nLoops++] = LOOP_SHUNT_KERNEL;             // OLF order 10: ShuntVoltageControl, IN-KERNEL
    } else if (ctx->nSvd > 0) {
        loopOrder[nLoops++] = LOOP_SVD_KERNEL;               // OLF order 10: WITH_GENERATOR shunt VC, IN-KERNEL
    } else if (ctx->hostLoopMask & (1 << 4)) {
        loopOrder[nLoops++] = LOOP_HOST_SHUNT;               // OLF order 10: ShuntVoltageControl
    }
    if (nLoops == 0) {
        loopOrder[nLoops++] = LOOP_RL;                       // nothing to drive: the old code's phase-1 fallback
    }
    // Where the realistic-voltage check STARTS applying in this sweep. AcloadFlowEngine keeps
    // checkUnrealisticStates false up to and including the last loop that can fix an unrealistic state
    // (that is the whole point of robust mode: give ReactiveLimits / the discrete
    // TransformerVoltageControl a chance to repair the state instead of failing the run on it), then
    // turns it on for the rest of the sweep and resets it at the top of the next one. A device pass at
    // index i applies loopOrder[i] to the state its solve produced, so "past the fixing loop" is
    // i > realisticFixingIdx. Not found, or no fixing loop at all, means check every solve — the
    // conservative direction, since a check that fires too early only routes to the CPU.
    int realisticFixingIdx = -1;
    for (int i = 0; i < nLoops; i++) {
        bool isFixing = ctx->realisticFixingLoop == 0
                ? loopOrder[i] == LOOP_RL
                : (ctx->realisticFixingLoop == 1
                        && (loopOrder[i] == LOOP_HOST_TVC
                            || (loopOrder[i] == LOOP_TVD_KERNEL && ctx->tvdMode != 1)));
        if (isFixing) {
            realisticFixingIdx = i;
        }
    }
    int loopIdx = 0;
    bool seqChanged = false;                                 // did any outer loop re-solve during the current RL->DS sequence
    if (!resume && ctx->startLoopKind >= 0) {
        // Restart after a host loop mutated the system: the first pass solves, then the sweep picks
        // up at THAT loop's slot (OLF's runOuterLoop re-checks it until STABLE, and only then does
        // the sweep restart from the first loop — seqChanged makes that restart happen).
        for (int i = 0; i < nLoops; i++) {
            if (loopOrder[i] == ctx->startLoopKind) {
                loopIdx = i;
                seqChanged = true;
            }
        }
        ctx->startLoopKind = -1;
    }
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
    if (resume) {
        // Continue the paused sequence: the loop the host owned has just run, so step PAST it and
        // let its verdict drive the pass exactly as a device loop's would — UNSTABLE re-solves only
        // the scenarios it changed, STABLE advances without a solve (OLF's rule).
        loopIdx = ctx->pausedLoopIdx;
        seqChanged = ctx->pausedSeqChanged != 0;
        prodIters = ctx->pausedProdIters;
        live = ctx->pausedLive;
        if (ctx->pausedStatus.size() == res.status.size()) {
            res.status = ctx->pausedStatus;                  // the state is the converged one the pause left: a
        }                                                    // STABLE host verdict then re-CHECKS RL on it (rlActive)
        bool anyChanged = false;
        for (int s = 0; s < nS; s++) {
            char ch = s < (int) ctx->hostChanged.size() ? ctx->hostChanged[s] : 0;
            active[s] = ch ? 1 : 0;
            anyChanged |= ch != 0;
        }
        if (anyChanged) {
            seqChanged = true;
            solvePass = true;                                // the host loop moved targets -> re-solve
            // ...and it CONSUMES the outer-iteration budget, exactly as a device loop's change does.
            // Without this a host loop that keeps going UNSTABLE never retires a scenario: each resume
            // starts a fresh `outer` budget, so the pause/resume cycle spins forever (measured: a
            // 2000-contingency rte6515 slice with SecondaryVoltageControl did not finish in 28 min).
            for (int s = 0; s < nS; s++) {
                if (active[s] && ++prodIters[s] >= RL_MAX_OUTER) {
                    traceReject(s, "host-loop-outer-budget", -1, 0);
                    res.status[s] = -2;                      // budget exhausted -> CPU replays it
                    live[s] = 0;
                    active[s] = 0;
                }
            }
        } else {
            solvePass = false;
            for (int s = 0; s < nS; s++) {
                active[s] = live[s];                         // stable: re-CHECK every live scenario
            }
            if (++loopIdx >= nLoops) {                       // ...and advance, as the stable branch does
                if (!seqChanged) {
                    loopIdx = nLoops;                        // a full sweep changed nothing -> converged
                } else {
                    seqChanged = false;
                    loopIdx = 0;
                }
            }
        }
    }
    for (int outer = 0; outer < maxOuter && loopIdx < nLoops; outer++) {
    gOuterPasses++;
    if (solvePass) {
        gSolvePasses++;
    }
    std::copy(active.begin(), active.end(), wasActive.begin());   // scenarios (re)solved in this inner pass
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    int nActive = 0;
    for (char a : active) {
        if (a) nActive++;
    }
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
        {
            static const char* cbProbe = std::getenv("OLF_GPU_CB_PROBE");
            static bool cbProbed = false;
            if (cbProbe != nullptr && !cbProbed && nS > 1 && ctx->st->cb.ne) {
                cbProbed = true;
                cbRowProbe<<<1, 1, 0, ctx->stream>>>(nS, std::atoi(cbProbe), ctx->st->cb.in.as<double>(),
                        ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                        ctx->nElements, ctx->cbOffset, ctx->dDisabled.as<char>(), ctx->dActive.as<char>());
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
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
    int dqTraceRow = gpuDqTraceRow();
    if (ctx->ndq) fillDistrQBatch<<<grid(ctx->ndq * nS), BLOCK, 0, ctx->stream>>>(ctx->ndq, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->obOffset, ctx->st->dDqElem.as<int>(), ctx->st->dDqSide.as<int>(), ctx->st->dDqKind.as<int>(), ctx->st->dDqWeight.as<double>(), ctx->st->dDqDest.as<int>(), ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nDqOv, ctx->nDqOv ? ctx->dDqOvIdx.as<int>() : nullptr, ctx->nDqOv ? ctx->dDqOvW.as<double>() : nullptr, ctx->dMode.as<int>(), ctx->nDqInact,
                ctx->nDqInact ? ctx->dDqInact.as<int>() : nullptr,
                ctx->nDqDyn ? ctx->dDqKMember.as<int>() : nullptr, ctx->nDqDyn || ctx->hasDqQrGroups ? ctx->dDqJMember.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberKey.as<double>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberRlEntry.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupStart.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupCount.as<int>() : nullptr, ctx->dRlLimitType.as<char>(), ctx->nrl, ctx->hasDqQrGroups ? ctx->dDqMemberQrGroup.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr, ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), dqTraceRow);
        if (ctx->ndr) fillDistrRhoBatch<<<grid(ctx->ndr * nS), BLOCK, 0, ctx->stream>>>(ctx->ndr, nS, n, nnz, ctx->st->dDrCol.as<int>(), ctx->st->dDrCoef.as<double>(), ctx->st->dDrDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nDrOv, ctx->nDrOv ? ctx->dDrOvIdx.as<int>() : nullptr, ctx->nDrOv ? ctx->dDrOvC.as<double>() : nullptr, ctx->st->drQr ? ctx->st->dDrQrGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>());
        if (ctx->ndb) fillDistrShuntBBatch<<<grid(ctx->ndb * nS), BLOCK, 0, ctx->stream>>>(ctx->ndb, nS, n, nnz, ctx->st->dDbCol.as<int>(), ctx->st->dDbCoef.as<double>(), ctx->st->dDbDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
        if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), ctx->st->dZiAltDest.as<int>(), ctx->st->dZiSelRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ziDyn ? 1 : 0, ziRelocOf(ctx->st.get()));
        applyRowModeBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->dMode.as<int>(), ctx->st->dRowPtr.as<int>(), ctx->st->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dRowSlope.as<double>());
        hazRepairValues(ctx, nS, ctx->dVals.as<double>());
        if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: stream-ordered into the solve, no host sync
    };
    // ---- IncrementalTransformerVoltageControl as a DRIVER PHASE (OLF order 8). OLF's own rule, per
    //      controlled bus: if |targetV - V| > halfDeadband, take dV/drho from the Jacobian at the converged
    //      state (unit RHS at the rho identity row, read the bus V entry), deltaR1 = diffV / sensi, move to
    //      the tap whose r1 is closest to r1 + deltaR1 within maxTapShift and the ratchet's allowed
    //      direction (MAX_DIRECTION_CHANGE 3), skip controllers with |sensi| < MIN_SENSI_FILTER (0.05).
    //      The sensitivity is the same forward solve GpuBatchIncrementalTvc used to make through
    //      solveBatchSensitivities — one host round trip per round — done here against the factors the
    //      driver already holds, with no host copy of the states. Single controller per controlled bus
    //      (the JNI upload refuses shared buses, which stay on the host orchestrator). ----
    std::vector<char> loopRejected(nS, 0);                   // a device loop routed the scenario to the CPU this pass
    auto tvcStep = [&]() {
        int nTvc = ctx->nTvc;
        if (nTvc <= 0) return;
        if (ctx->mode != 3 && ctx->mode != 4 && ctx->mode != 5) {
            static bool warned = false;
            if (!warned) { std::fprintf(stderr, "in-kernel TVC needs a UBATCH mode (3/4/5); mode %d runs it as stable\n", ctx->mode); warned = true; }
            return;
        }
        size_t nst = static_cast<size_t>(nS) * nTvc;
        tvcDeadbandCheckBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nTvc, nS, n, ctx->nElements,
                ctx->dTvcVRow.as<int>(), ctx->dTvcTargetV.as<double>(), ctx->dTvcHalfDb.as<double>(),
                ctx->dTvcElemIndex.as<int>(), ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
                ctx->dTvcOutOfDb.as<char>(), ctx->dTvcDiffV.as<double>());
        std::vector<char> outOfDb(nst);
        CUDA_CHECK(cudaMemcpyAsync(outOfDb.data(), ctx->dTvcOutOfDb.p, nst, cudaMemcpyDeviceToHost, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<char> needC(nTvc, 0);
        bool any = false;
        for (int s2 = 0; s2 < nS; s2++) {
            for (int c = 0; c < nTvc; c++) {
                if (outOfDb[static_cast<size_t>(s2) * nTvc + c]) { needC[c] = 1; any = true; }
            }
        }
        if (!any) return;                                    // every bus inside its deadband: STABLE
        // Sensitivities dV/drho at the CURRENT states: refill J for every scenario, factorize once,
        // then one unit-RHS solve per controller that has an out-of-deadband scenario.
        std::vector<char> ones(nS, 1);
        CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
        fillActiveBatch(false);
        if (!ctx->analyzed) {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
            ctx->analyzed = true;
        }
        CUDA_CHECK(cudaMemsetAsync(ctx->dTvcSensi.p, 0, nst * sizeof(double), ctx->stream));
        bool first = true;
        for (int c = 0; c < nTvc; c++) {
            if (!needC[c]) continue;
            int row = ctx->st->rowMap[ctx->hTvcRhoRow[c]];
            if (row < 0) continue;                           // rho row not an active equation: stays insensitive
            setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
            hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
            int phase = CUDSS_PHASE_SOLVE | (first ? CUDSS_PHASE_FACTORIZATION : 0);
            first = false;
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, phase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
            tvcGatherSensiBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, c, nTvc, ctx->dTvcVRow.as<int>(),
                    ctx->dDx.as<double>(), ctx->dTvcSensi.as<double>());
        }
        ctx->factored = true;
        CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));   // as it was
        // Tap decision; a moved scenario is an UNSTABLE verdict (dRlSwitched), like every device loop.
        tvcGroupDecisionBatch<<<grid(ctx->nTvcGroups * nS), BLOCK, 0, ctx->stream>>>(ctx->nTvcGroups, nS, nTvc,
                ctx->dTvcGroupPtr.as<int>(), ctx->dTvcGroupCtrl.as<int>(),
                ctx->dTvcOutOfDb.as<char>(), ctx->dTvcDiffV.as<double>(), ctx->dTvcSensi.as<double>(), ctx->dTvcHalfDb.as<double>(),
                ctx->dTvcTapPtr.as<int>(), ctx->dTvcTapData.as<double>(),
                ctx->dTvcCurTap.as<int>(), ctx->dTvcRatchetDir.as<int>(), ctx->dTvcRatchetCount.as<int>(),
                ctx->dTvcRatchetAllowed.as<int>(), ctx->tvcMaxTapShift, 0.05, ctx->dRlSwitched.as<char>());
        // Apply: rho target + state seed on the moved pairs, then the pi rows through the tap-override list.
        tvcApplyRhoBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nTvc, nS, n,
                ctx->dTvcRhoRow.as<int>(), ctx->dTvcTapPtr.as<int>(), ctx->dTvcTapData.as<double>(),
                ctx->dTvcCurTap.as<int>(), ctx->dTvcPrevTap.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        tvcRebuildOverrides(ctx);
        if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
            std::vector<int> cur(nst), base(ctx->hTvcBaseTap);
            CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dTvcCurTap.p, nst * sizeof(int), cudaMemcpyDeviceToHost));
            int nOut = 0, nMoved = 0;
            for (size_t k = 0; k < nst; k++) {
                nOut += outOfDb[k] ? 1 : 0;
            }
            for (int s2 = 0; s2 < nS; s2++) {
                for (int c = 0; c < nTvc; c++) {
                    if (cur[static_cast<size_t>(s2) * nTvc + c] != base[c]) nMoved++;
                }
            }
            std::fprintf(stderr, "TVC_KERNEL nS=%d nCtrl=%d outOfDeadband=%d pairsOffBaseTap=%d nR1Ov=%d\n",
                         nS, nTvc, nOut, nMoved, ctx->nR1Ov);
            std::vector<double> dv(nst), se(nst);
            CUDA_CHECK(cudaMemcpy(dv.data(), ctx->dTvcDiffV.p, nst * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(se.data(), ctx->dTvcSensi.p, nst * sizeof(double), cudaMemcpyDeviceToHost));
            for (size_t k = 0; k < nst; k++) {
                if (outOfDb[k]) {
                    std::fprintf(stderr, "TVC_KERNEL_PAIR s=%d c=%d diffV=%.9e sensi=%.9e tap=%d base=%d rhoRowMapped=%d\n",
                                 (int) (k / nTvc), (int) (k % nTvc), dv[k], se[k], cur[k], base[k % nTvc],
                                 ctx->st->rowMap[ctx->hTvcRhoRow[k % nTvc]]);
                }
            }
        }
    };
    // ---- IncrementalShuntVoltageControl as a DRIVER PHASE (OLF order 10): deadband per controlled bus,
    //      dV/dB from a unit RHS at each needed controller's SHUNT_B identity row (the driver's own
    //      factors), OLF's adjustB round-robin per group, the section total pinned at the identity row
    //      and the static (dG, dB) through the shunt-override list. Groups with several controller
    //      shunts ARE handled (the round-robin runs in the group kernel). ----
    auto shcStep = [&]() {
        int nShc = ctx->nShc;
        if (nShc <= 0) return;
        if (ctx->mode != 3 && ctx->mode != 4 && ctx->mode != 5) {
            static bool warned = false;
            if (!warned) { std::fprintf(stderr, "in-kernel shunt VC needs a UBATCH mode (3/4/5); mode %d runs it as stable\n", ctx->mode); warned = true; }
            return;
        }
        int nG = ctx->nShcGroups;
        size_t nsg = static_cast<size_t>(nS) * nG;
        shcDeadbandCheckBatch<<<grid(static_cast<int>(nsg)), BLOCK, 0, ctx->stream>>>(nG, nS, n, ctx->nElements,
                ctx->dShcGroupPtr.as<int>(), ctx->dShcGroupCtrl.as<int>(),
                ctx->dShcVRow.as<int>(), ctx->dShcTargetV.as<double>(), ctx->dShcHalfDb.as<double>(), ctx->dShcElemAbs.as<int>(),
                ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
                ctx->dShcOutOfDb.as<char>(), ctx->dShcDiffV.as<double>());
        std::vector<char> outOfDb(nsg);
        CUDA_CHECK(cudaMemcpyAsync(outOfDb.data(), ctx->dShcOutOfDb.p, nsg, cudaMemcpyDeviceToHost, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<int> gPtr(nG + 1), gCtrl(ctx->nShc);
        CUDA_CHECK(cudaMemcpy(gPtr.data(), ctx->dShcGroupPtr.p, gPtr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gCtrl.data(), ctx->dShcGroupCtrl.p, gCtrl.size() * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<char> needC(nShc, 0);
        bool any = false;
        for (int s2 = 0; s2 < nS; s2++) {
            for (int g = 0; g < nG; g++) {
                if (!outOfDb[static_cast<size_t>(s2) * nG + g]) continue;
                any = true;
                for (int k = gPtr[g]; k < gPtr[g + 1]; k++) needC[gCtrl[k]] = 1;
            }
        }
        if (!any) return;                                    // every controlled bus inside its deadband: STABLE
        std::vector<char> ones(nS, 1);
        CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
        fillActiveBatch(false);
        if (!ctx->analyzed) {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
            ctx->analyzed = true;
        }
        CUDA_CHECK(cudaMemsetAsync(ctx->dShcSensi.p, 0, static_cast<size_t>(nS) * nShc * sizeof(double), ctx->stream));
        bool first = true;
        for (int c = 0; c < nShc; c++) {
            if (!needC[c]) continue;
            int row = ctx->st->rowMap[ctx->hShcBRow[c]];
            if (row < 0) continue;
            setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
            hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
            int phase = CUDSS_PHASE_SOLVE | (first ? CUDSS_PHASE_FACTORIZATION : 0);
            first = false;
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, phase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
            shcGatherSensiBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, c, nShc, ctx->dShcVRow.as<int>(),
                    ctx->dDx.as<double>(), ctx->dShcSensi.as<double>());
        }
        ctx->factored = true;
        CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));
        shcAdjustBatch<<<grid(static_cast<int>(nsg)), BLOCK, 0, ctx->stream>>>(nG, nS, n, ctx->nElements, nShc, ctx->nShcSubs,
                ctx->shcMaxSectionShift, ctx->dShcGroupPtr.as<int>(), ctx->dShcGroupCtrl.as<int>(), ctx->dShcSubPtr.as<int>(),
                ctx->dShcSecPtr.as<int>(), ctx->dShcSecB.as<double>(), ctx->dShcSubMin.as<int>(),
                ctx->dShcHalfDb.as<double>(), ctx->dShcElemAbs.as<int>(), ctx->dDisabled.as<char>(),
                ctx->dShcOutOfDb.as<char>(), ctx->dShcDiffV.as<double>(), ctx->dShcSensi.as<double>(),
                ctx->dShcPos.as<int>(), ctx->dShcRatDir.as<int>(), ctx->dShcRatCount.as<int>(), ctx->dShcRatAllowed.as<int>(),
                ctx->dShcShift.as<int>(), ctx->dRlSwitched.as<char>());
        shcApplyBatch<<<grid(nShc * nS), BLOCK, 0, ctx->stream>>>(nShc, nS, n, ctx->dShcBRow.as<int>(), ctx->dShcSubPtr.as<int>(),
                ctx->dShcSecPtr.as<int>(), ctx->dShcSecB.as<double>(), ctx->dShcSubBase.as<int>(), ctx->dShcPos.as<int>(),
                ctx->nShcSubs, ctx->dShcBaseB.as<double>(), ctx->dShcAppliedDb.as<double>(), ctx->dTarget.as<double>(), ctx->dX.as<double>());
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        shcRebuildOverrides(ctx);
        if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
            std::vector<int> pos(static_cast<size_t>(nS) * ctx->nShcSubs);
            CUDA_CHECK(cudaMemcpy(pos.data(), ctx->dShcPos.p, pos.size() * sizeof(int), cudaMemcpyDeviceToHost));
            int nOut = 0, nMoved = 0;
            for (char c : outOfDb) nOut += c ? 1 : 0;
            for (int s2 = 0; s2 < nS; s2++) for (int u = 0; u < ctx->nShcSubs; u++) if (pos[static_cast<size_t>(s2) * ctx->nShcSubs + u] != ctx->hShcSubBase[u]) nMoved++;
            std::fprintf(stderr, "SHUNT_KERNEL nS=%d nCtrl=%d nGroups=%d outOfDeadband=%d subsOffBase=%d nShOv=%d\n", nS, nShc, nG, nOut, nMoved, ctx->nShOv);
        }
    };
    // ---- the FLOW-OBSERVED incremental loops as DRIVER PHASES: kindMask selects the phase controllers
    //      (OLF order 7: kinds 1+2) or the RPC controllers (OLF order 9: kind 0). Same skeleton as the
    //      voltage-observed phases; the observable is the controlled branch's flow gradient dotted with
    //      the unit-RHS solution (OLF's EquationTerm.calculateSensi), evaluated by the codegen'd branch
    //      functions on the device. ----
    auto flcStep = [&](int kindMask) {
        int nF = ctx->nFlc;
        if (nF <= 0) return;
        if (ctx->mode != 3 && ctx->mode != 4 && ctx->mode != 5) {
            static bool warned = false;
            if (!warned) { std::fprintf(stderr, "in-kernel flow control needs a UBATCH mode (3/4/5); mode %d runs it as stable\n", ctx->mode); warned = true; }
            return;
        }
        size_t nst = static_cast<size_t>(nS) * nF;
        flcCheckBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nF, nS, n, ctx->nElements, ctx->cbOffset, kindMask,
                ctx->dFlcKind.as<int>(), ctx->dFlcSide.as<int>(), ctx->dFlcTargetValue.as<double>(), ctx->dFlcHalfDb.as<double>(),
                ctx->dFlcElem.as<int>(), ctx->dFlcCtlPack.as<int>(),
                ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
                ctx->dFlcOutOfDb.as<char>(), ctx->dFlcDiff.as<double>());
        std::vector<char> outOfDb(nst);
        CUDA_CHECK(cudaMemcpyAsync(outOfDb.data(), ctx->dFlcOutOfDb.p, nst, cudaMemcpyDeviceToHost, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<char> needC(nF, 0);
        bool any = false;
        for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < nF; c++) if (outOfDb[static_cast<size_t>(s2) * nF + c]) { needC[c] = 1; any = true; }
        if (!any) return;
        std::vector<char> ones(nS, 1);
        CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
        fillActiveBatch(false);
        if (!ctx->analyzed) {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
            ctx->analyzed = true;
        }
        CUDA_CHECK(cudaMemsetAsync(ctx->dFlcSensi.p, 0, nst * sizeof(double), ctx->stream));
        bool first = true;
        for (int c = 0; c < nF; c++) {
            if (!needC[c]) continue;
            int row = ctx->st->rowMap[ctx->hFlcPinRow[c]];
            if (row < 0) continue;
            setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
            hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
            int phase = CUDSS_PHASE_SOLVE | (first ? CUDSS_PHASE_FACTORIZATION : 0);
            first = false;
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, phase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
            flcGatherSensiBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, n, c, nF, ctx->cbOffset,
                    ctx->dFlcKind.as<int>(), ctx->dFlcSide.as<int>(), ctx->dFlcCtlPack.as<int>(),
                    ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                    ctx->dX.as<double>(), ctx->dDx.as<double>(), ctx->dFlcSensi.as<double>());
        }
        ctx->factored = true;
        CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));
        flcDecisionBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nF, nS, kindMask,
                ctx->dFlcKind.as<int>(), ctx->dFlcPinCol.as<int>(), ctx->dFlcTapPtr.as<int>(), ctx->dFlcTapData.as<double>(), ctx->dFlcMaxShift.as<int>(),
                ctx->dFlcOutOfDb.as<char>(), ctx->dFlcDiff.as<double>(), ctx->dFlcSensi.as<double>(),
                ctx->dFlcCurTap.as<int>(), ctx->dFlcRatDir.as<int>(), ctx->dFlcRatCount.as<int>(), ctx->dFlcRatAllowed.as<int>(), ctx->dRlSwitched.as<char>());
        flcApplyBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nF, nS, n, ctx->dFlcPinRow.as<int>(), ctx->dFlcPinCol.as<int>(),
                ctx->dFlcTapPtr.as<int>(), ctx->dFlcTapData.as<double>(), ctx->dFlcCurTap.as<int>(), ctx->dFlcPrevTap.as<int>(),
                ctx->dTarget.as<double>(), ctx->dX.as<double>());
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        tvcRebuildOverrides(ctx);
        if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
            std::vector<int> cur(nst);
            CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dFlcCurTap.p, nst * sizeof(int), cudaMemcpyDeviceToHost));
            int nOut = 0, nMoved = 0;
            for (char c : outOfDb) nOut += c ? 1 : 0;
            for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < nF; c++) if (cur[static_cast<size_t>(s2) * nF + c] != ctx->hFlcBaseTap[c]) nMoved++;
            std::fprintf(stderr, "FLOW_KERNEL kindMask=%d nS=%d nCtrl=%d outOfDeadband=%d pairsOffBaseTap=%d nR1Ov=%d\n", kindMask, nS, nF, nOut, nMoved, ctx->nR1Ov);
        }
    };
    // ---- CONTINUOUS PhaseControl as a DRIVER PHASE (OLF order 7), see cphCheckBatch. ----
    auto cphStep = [&]() {
        int nC = ctx->nCph;
        if (nC <= 0) return;
        size_t nst = static_cast<size_t>(nS) * nC;
        cphCheckBatch<<<grid(static_cast<int>(nst)), BLOCK, 0, ctx->stream>>>(nC, nS, n, ctx->nElements, ctx->cbOffset,
                ctx->dCphKind.as<int>(), ctx->dCphA1Row.as<int>(), ctx->dCphSide.as<int>(), ctx->dCphLimit.as<double>(), ctx->dCphElem.as<int>(),
                ctx->dCphTapPtr.as<int>(), ctx->dCphTapData.as<double>(),
                ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(), ctx->dCphIter.as<int>(),
                ctx->dCphTap.as<int>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>(), ctx->dRlSwitched.as<char>());
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        {                                                    // the loop's check count, per checked scenario
            std::vector<int> it(nS);
            CUDA_CHECK(cudaMemcpy(it.data(), ctx->dCphIter.p, nS * sizeof(int), cudaMemcpyDeviceToHost));
            for (int s2 = 0; s2 < nS; s2++) if (rlActive[s2]) it[s2]++;
            CUDA_CHECK(cudaMemcpy(ctx->dCphIter.p, it.data(), nS * sizeof(int), cudaMemcpyHostToDevice));
        }
        tvcRebuildOverrides(ctx);
        if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
            std::vector<int> cur(nst);
            CUDA_CHECK(cudaMemcpy(cur.data(), ctx->dCphTap.p, nst * sizeof(int), cudaMemcpyDeviceToHost));
            int nMoved = 0;
            for (int s2 = 0; s2 < nS; s2++) for (int c = 0; c < nC; c++) if (cur[static_cast<size_t>(s2) * nC + c] != ctx->hCphBaseTap[c]) nMoved++;
            std::fprintf(stderr, "CPHASE_KERNEL nS=%d nCtrl=%d pairsOffBaseTap=%d nR1Ov=%d\n", nS, nC, nMoved, ctx->nR1Ov);
        }
    };
    // ---- the DISCRETE voltage-control loops as DRIVER PHASES: TransformerVoltageControl (AFTER_GENERATOR /
    //      WITH_GENERATOR, OLF order 8) and ShuntVoltageControl (WITH_GENERATOR, OLF order 10). No sensitivities:
    //      OLF's decisions read the converged state only (deadband, clamp bounds, closest tap / section). ----
    auto tvdStep = [&]() {
        if (ctx->nTvd <= 0) return;
        tvdCheckBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(tvdTables(ctx), nS, n, ctx->nElements, ctx->dDisabled.as<char>(),
                ctx->dRlActive.as<char>(), ctx->dQbuf.as<double>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>(),
                ctx->dRlSwitched.as<char>());
        // Freezing a REMOTE controller can take the LAST live member of its group, and then the group
        // controls nothing: OLF's freeze disables the voltage control, so the CONTROLLED bus stops
        // carrying a V identity. That flip lives in its own kernel, which the reactive-limits phase runs
        // after its pins; the freeze happens two loops later and would otherwise leave the pilot holding
        // a target for a control that no longer exists until the next sweep reached RL again (measured
        // 0.6 kV). Same kernel, run again here — it restores the identity at COMPLETE on the same rule.
        if (ctx->hasDqQrGroups && ctx->st->nQr > 0 && ctx->nTvdFz > 0) {
            updatePilotVeqBatch<<<grid(ctx->st->nQr * nS), BLOCK, 0, ctx->stream>>>(
                    ctx->st->nQr, nS, n,
                    ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>(), ctx->dQrGenVc.as<char>(),
                    ctx->dQrPilotQ.as<double>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                    ctx->dQrVeqOff.as<char>(), ctx->dQrSavedVT.as<double>(),
                    ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>(),
                    ctx->nFrozenRow, ctx->nFrozenRow ? ctx->dFrozenRow.as<int>() : nullptr);
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<char> rej(nS);
        CUDA_CHECK(cudaMemcpy(rej.data(), ctx->dTvdReject.p, nS, cudaMemcpyDeviceToHost));
        for (int s2 = 0; s2 < nS; s2++) if (rej[s2]) { loopRejected[s2] = 1; traceReject(s2, "tvd-kernel-reject", outer, 0); }
        tvcRebuildOverrides(ctx);
        if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
            size_t nsm = static_cast<size_t>(nS) * ctx->nTvd;
            std::vector<int> st(nsm), step(nS);
            CUDA_CHECK(cudaMemcpy(st.data(), ctx->dTvdMemState.p, nsm * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(step.data(), ctx->dTvdScnStep.p, nS * sizeof(int), cudaMemcpyDeviceToHost));
            int nCont = 0, nFixed = 0, nRej = 0;
            for (size_t k = 0; k < nsm; k++) { nCont += st[k] == 1; nFixed += st[k] == 2; }
            for (int s2 = 0; s2 < nS; s2++) nRej += rej[s2] ? 1 : 0;
            std::fprintf(stderr, "TVD_KERNEL mode=%d nS=%d nMembers=%d continuous=%d fixed=%d rejected=%d step0=%d nR1Ov=%d\n",
                         ctx->tvdMode, nS, ctx->nTvd, nCont, nFixed, nRej, step[0], ctx->nR1Ov);
        }
    };
    // ---- SecondaryVoltageControl as a DRIVER PHASE (OLF order 4). The whole loop: the re-enable step and
    //      the per-zone trigger on the paused state's own qBuf, one unit-RHS solve per controlled bus on the
    //      driver's own factors for the K and pilot sensitivities, the small dense zone system per scenario,
    //      and the moved targets applied to the controlled buses' V rows (plus the switch-back comparison
    //      target for a pinned controller). Nothing goes back to the host, so the loop sits in OLF's order
    //      instead of after the sweep. ----
    auto svcStep = [&]() {
        int nC = ctx->nSvc;
        if (nC <= 0) return;
        // DIAGNOSTIC: keep the slot in the sweep but do nothing in it, to separate "the extra slot moved
        // the sweep" from "the phase body did". That probe is how the residual below was pinned down.
        static const bool svcInert = std::getenv("OLF_GPU_SVC_INERT") != nullptr;
        if (svcInert) return;
        if (ctx->mode != 3 && ctx->mode != 4 && ctx->mode != 5) {
            static bool warned = false;
            if (!warned) { std::fprintf(stderr, "in-kernel SecondaryVc needs a UBATCH mode (3/4/5); mode %d runs it as stable\n", ctx->mode); warned = true; }
            return;
        }
        CUDA_CHECK(cudaMemsetAsync(ctx->dSvcNeed.p, 0, nS, ctx->stream));
        svcTriggerBatch<<<grid(ctx->nSvcZones * nS), BLOCK, 0, ctx->stream>>>(svcTables(ctx), nS, n,
                ctx->dRlActive.as<char>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(),
                ctx->dRlLimitType.as<char>(), ctx->dRlSavedMode.as<char>(),
                ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                std::getenv("OLF_GPU_SVC_DBG") != nullptr ? 1 : 0);
        std::vector<char> need(nS);
        CUDA_CHECK(cudaMemcpyAsync(need.data(), ctx->dSvcNeed.p, nS, cudaMemcpyDeviceToHost, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        bool any = false;
        for (char c : need) any |= c != 0;
        if (!any) return;                                    // every zone inside its bands: OLF's STABLE
        std::vector<char> ones(nS, 1);
        CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS, cudaMemcpyHostToDevice));
        fillActiveBatch(false);
        if (!ctx->analyzed) {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
            ctx->analyzed = true;
        }
        bool first = true;
        for (int i = 0; i < nC; i++) {
            int row = ctx->st->rowMap[ctx->hSvcCtldVRow[i]];
            if (row < 0) continue;                           // controlled bus not a live identity: insensitive
            setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
            hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
            int phase = CUDSS_PHASE_SOLVE | (first ? CUDSS_PHASE_FACTORIZATION : 0);
            first = false;
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, phase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
            svcGatherBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(svcTables(ctx), nS, n, i,
                    ctx->nElements, ctx->cbOffset, ctx->shOffset,
                    ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(),
                    ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(),
                    ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                    ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(),
                    ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dDx.as<double>());
        }
        ctx->factored = true;
        CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS, cudaMemcpyHostToDevice));   // as it was
        // ---- the zone systems: assembled into device memory, solved by cuBLAS's BATCHED dense LU ----
        if (ctx->svcBlas == nullptr) {
            CUBLAS_CHECK(cublasCreate(&ctx->svcBlas));
        }
        CUBLAS_CHECK(cublasSetStream(ctx->svcBlas, ctx->stream));
        svcAssembleBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(svcTables(ctx), nS, n,
                ctx->dRlActive.as<char>(), ctx->dSvcCtlZone.as<int>(), ctx->dX.as<double>(),
                ctx->dSvcMat.as<double>(), ctx->dSvcRhs.as<double>());
        svcPointersBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, nC,
                ctx->dSvcMat.as<double>(), ctx->dSvcRhs.as<double>(),
                reinterpret_cast<double**>(ctx->dSvcMatPtr.p), reinterpret_cast<double**>(ctx->dSvcRhsPtr.p));
        CUBLAS_CHECK(cublasDgetrfBatched(ctx->svcBlas, nC,
                reinterpret_cast<double**>(ctx->dSvcMatPtr.p), nC,
                ctx->dSvcPiv.as<int>(), ctx->dSvcInfo.as<int>(), nS));
        int trsInfo = 0;
        CUBLAS_CHECK(cublasDgetrsBatched(ctx->svcBlas, CUBLAS_OP_N, nC, 1,
                reinterpret_cast<const double* const*>(ctx->dSvcMatPtr.p), nC,
                ctx->dSvcPiv.as<int>(),
                reinterpret_cast<double**>(ctx->dSvcRhsPtr.p), nC, &trsInfo, nS));
        if (trsInfo != 0) {
            throw std::runtime_error("cublasDgetrsBatched rejected its arguments");
        }
        svcCommitBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(svcTables(ctx), nS,
                ctx->dRlActive.as<char>(), ctx->dSvcInfo.as<int>(), ctx->dSvcRhs.as<double>(),
                ctx->dRlSwitched.as<char>());
        svcApplyBatch<<<grid(nC * nS), BLOCK, 0, ctx->stream>>>(svcTables(ctx), nS, n,
                ctx->dRlActive.as<char>(), ctx->dRlLimitType.as<char>(), ctx->dTarget.as<double>());
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        std::vector<char> rej(nS);
        CUDA_CHECK(cudaMemcpy(rej.data(), ctx->dSvcReject.p, nS, cudaMemcpyDeviceToHost));
        int nRej = 0;
        for (int s2 = 0; s2 < nS; s2++) {
            if (rej[s2]) { loopRejected[s2] = 1; nRej++; traceReject(s2, "svc-kernel-reject", outer, 0); }
        }
        if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
            int nNeed = 0;
            for (char c : need) nNeed += c ? 1 : 0;
            std::fprintf(stderr, "SVC_KERNEL nS=%d nCtl=%d nZones=%d adjusting=%d rejected=%d\n",
                         nS, nC, ctx->nSvcZones, nNeed, nRej);
        }
    };
    auto svdStep = [&]() {
        if (ctx->nSvd <= 0) return;
        svdCheckBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(svdTables(ctx), nS, n, ctx->nElements, ctx->dDisabled.as<char>(),
                ctx->dRlActive.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dX.as<double>(), ctx->dRlSwitched.as<char>());
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
            size_t nsm = static_cast<size_t>(nS) * ctx->nSvd;
            std::vector<int> st(nsm);
            CUDA_CHECK(cudaMemcpy(st.data(), ctx->dSvdMemState.p, nsm * sizeof(int), cudaMemcpyDeviceToHost));
            int nCont = 0, nFixed = 0;
            for (size_t k = 0; k < nsm; k++) { nCont += st[k] == 1; nFixed += st[k] == 2; }
            std::fprintf(stderr, "SVD_KERNEL nS=%d nMembers=%d continuous=%d dispatched=%d\n", nS, ctx->nSvd, nCont, nFixed);
        }
    };
    if (solvePass) {

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

    // OLF_GPU_FORCE_FIRST_STEP=1 (diagnostic): mirror OLF's "always at least one Newton step".
    const bool forceFirstStep = std::getenv("OLF_GPU_FORCE_FIRST_STEP") != nullptr;
    for (int it = 0; it <= maxIter && nActive > 0; it++) {
        gLastPassIters = it;
        gNewtonIters++;
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
                // OLF's NewtonRaphson.run() computes the initial mismatch but only uses it to pick the
                // step scaling — it then enters runIteration UNCONDITIONALLY, and runIteration steps
                // FIRST and tests AFTER. So OLF always takes at least one step, even from a state that
                // already satisfies the criterion. We test before stepping, so a solve re-entered with
                // slightly moved targets (every outer-loop re-solve does exactly that) can retire
                // having moved nothing, leaving the state one step behind OLF's.
                if (forceFirstStep && it == 0) {
                    converged = false;
                }
                // OLF_GPU_CONV_TRACE=<slot>|all: the stopping DECISION itself - the 2-norm, the
                // threshold it is held against and the active row count behind that threshold. Two
                // arms that agree on every residual and still stop one Newton step apart differ
                // here and nowhere else, and the exit norms alone cannot show it.
                static const char* convTraceEnv = std::getenv("OLF_GPU_CONV_TRACE");
                static const int convTraceSlot = convTraceEnv == nullptr ? -1
                        : (std::string(convTraceEnv) == "all" ? -2 : std::atoi(convTraceEnv));
                if (convTraceSlot == -2 || convTraceSlot == s) {
                    std::fprintf(stderr, "CONV_TRACE s=%d it=%d norm2=%.17e thr2=%.17e nActive=%d inf=%.17e perRow=%d maxRatio=%.6e forceFirst=%d converged=%d\n",
                                 s, it, std::sqrt(ssq), thr2Of(s), ctx->hNActive.empty() ? n : ctx->hNActive[s], norm,
                                 ctx->perRowCriteria ? 1 : 0, maxRatio, (forceFirstStep && it == 0) ? 1 : 0, converged ? 1 : 0);
                    // the eight worst rows, by index: WHERE the residual sits, so a step that lands
                    // two orders of magnitude from the other arm's can be traced to its rows
                    std::vector<double> hFs(n);
                    CUDA_CHECK(cudaMemcpy(hFs.data(), ctx->dF.as<double>() + (size_t) s * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
                    std::vector<int> ord(n);
                    for (int i = 0; i < n; i++) ord[i] = i;
                    std::partial_sort(ord.begin(), ord.begin() + std::min(n, 8), ord.end(),
                                      [&](int a, int b) { return std::fabs(hFs[a]) > std::fabs(hFs[b]); });
                    std::fprintf(stderr, "CONV_TOP s=%d it=%d :", s, it);
                    for (int k = 0; k < std::min(n, 8); k++) {
                        std::fprintf(stderr, " [%d]=%.9e", ord[k], hFs[ord[k]]);
                    }
                    std::fprintf(stderr, "\n");
                }
                static const char* sdEnv = std::getenv("OLF_GPU_STATE_DUMP");
                if (sdEnv != nullptr && s == 0 && it <= std::atoi(sdEnv) && n <= 64) {
                    stateDumpBatch<<<1, 1, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dMode.as<int>(),
                            ctx->dTarget.as<double>(), ctx->dX.as<double>(), s, it);
                    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                }
                static const char* wrEnv = std::getenv("OLF_GPU_WORST_ROW");
                if (wrEnv != nullptr && s == 0 && it <= std::atoi(wrEnv)) {
                    worstRowBatch<<<1, 1, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dMode.as<int>(),
                            ctx->dTarget.as<double>(), ctx->dX.as<double>(), s);
                    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                }
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
                    traceReject(s, "straggler-exit", outer, it);
                    res.status[s] = -2;
                    res.iters[s] = it;
                    active[s] = 0;
                    live[s] = 0;
                    nActive--;
                }
            }
            break;
        }

        // Scenarios still being FACTORIZED this iteration. Summed over iterations this is the batch's
        // real work measure: the CPU pays each contingency its own iteration count, so a batch is only
        // as efficient as its retirement — a batch that keeps every scenario alive until the last
        // straggler converges pays the WORST case for all of them.
        gActiveIterSum += nActive;
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
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        if (!ctx->analyzed) {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matABatch, ctx->matXBatch, ctx->matBBatch));
            traceLuNnz(ctx->ds.handle, ctx->ds.data, "batch-api", ctx->nScenarios);
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
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        if (!ctx->analyzed) {
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matBig, ctx->matBigX, ctx->matBigB));
            traceLuNnz(ctx->ds.handle, ctx->ds.data, "blkdiag", ctx->nScenarios);
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
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        if (!ctx->analyzed) {
            bool timeIt = std::getenv("OLF_GPU_TRACE") != nullptr;
            long long t0 = 0;
            if (timeIt) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                t0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
            }
            CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB));
            double ms = -1.0;
            if (timeIt) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                ms = (std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count() - t0) / 1.0e6;
            }
            traceLuNnz(ctx->ds.handle, ctx->ds.data, "ubatch", ctx->nScenarios, ms);
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
        if (!compactSolve) {
            gWidthIterSum += nS;                             // full-width factorization this iteration
        }
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
                applyReorderAlg(ctx->dsC.config, ctx->st->nzi > 0);
                int irC = irSteps(ctx->st->nzi > 0);
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
            gWidthIterSum += w;                              // compacted width actually factorized
            gCompactIters++;
            CUDA_CHECK(cudaMemcpyAsync(ctx->dMapC.p, cmap.data(), w * sizeof(int), cudaMemcpyHostToDevice, ctx->stream));
            gatherScenarios<<<grid(nnz * w), BLOCK, 0, ctx->stream>>>(nnz, w, ctx->dMapC.as<int>(), ctx->dVals.as<double>(), ctx->dValsC.as<double>());
            gatherScenarios<<<grid(n * w), BLOCK, 0, ctx->stream>>>(n, w, ctx->dMapC.as<int>(), ctx->dB.as<double>(), ctx->dBC.as<double>());
            // A new batch size needs a FRESH data object and a fresh ANALYSIS. Setting
            // CUDSS_CONFIG_UBATCH_SIZE on the existing data and skipping the analysis is accepted by
            // cuDSS and even looks faster on a bench -- measured 2026-09-16 at -11% warm with exact
            // violation parity -- and then fails with an illegal memory access on a bigger network:
            // the data object's buffers are sized for the batch size it was analyzed at. The analysis
            // is cheap (~64 ms) and costs the same at any batch size, but its STATE belongs to one.
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
        // OLF_GPU_DX_DUMP=<path>: the FIRST solve's first Newton step, written out whole — the
        // assembled Jacobian (CSR triplets), the mismatch F it was built at, and the step dx it
        // produced. The CPU arm writes the same three under OLF_NR_DX_DUMP, and the two are
        // comparable WITHOUT a row map: the device's COLUMN index is OLF's own variable row
        // (GpuBatchPrecomputer stores v.getRow() verbatim), so dx compares term by term, and a
        // per-variable-column fingerprint of |J| localises any Jacobian difference to a variable.
        // Only F needs the row map, and its sorted multiset is permutation-invariant anyway.
        {
            static const char* dxPath = std::getenv("OLF_GPU_DX_DUMP");   // read once, not per iteration
            static bool dumped = false;
            if (dxPath != nullptr && !dumped && it == 0) {
                dumped = true;
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                std::vector<double> hv(nnz);
                std::vector<double> hd(n);
                std::vector<double> hff(n);
                std::vector<int> hrp(n + 1);
                std::vector<int> hci(nnz);
                std::vector<int> hmode(n);
                std::vector<double> hb(n);                   // the RHS as SOLVED (row-repaired, see hazRepairRhs)
                CUDA_CHECK(cudaMemcpy(hb.data(), ctx->dB.p, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hv.data(), ctx->dVals.p, (size_t) nnz * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hd.data(), ctx->dDx.p, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hff.data(), ctx->dF.p, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hrp.data(), ctx->st->dRowPtr.p, (size_t) (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hci.data(), ctx->st->dColIdx.p, (size_t) nnz * sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hmode.data(), ctx->dMode.p, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost));
                FILE* f = std::fopen(dxPath, "w");
                if (f != nullptr) {
                    std::fprintf(f, "N %d %d\n", n, nnz);
                    for (int i = 0; i < n; i++) {
                        std::fprintf(f, "F %d %.17e\n", i, hff[i]);
                        std::fprintf(f, "DX %d %.17e\n", i, hd[i]);
                        std::fprintf(f, "B %d %.17e\n", i, hb[i]);
                        std::fprintf(f, "MODE %d %d\n", i, hmode[i]);
                        for (int k = hrp[i]; k < hrp[i + 1]; k++) {
                            std::fprintf(f, "J %d %d %.17e\n", i, hci[k], hv[k]);
                        }
                    }
                    std::fclose(f);
                    std::fprintf(stderr, "OLF_GPU_DX_DUMP wrote %s (n=%d nnz=%d)\n", dxPath, n, nnz);
                }
            }
        }
        // OLF_GPU_ROW_PROBE=<row>: that row in EVERY scenario at EVERY iteration - its residual,
        // its own state, and its non-zero Jacobian entries by column. The DX dump answers "what is
        // the shared base structure" and the state dump only fits a toy system; neither says what
        // one row of a 13000-row SCENARIO looks like at convergence, which is the question when a
        // bus's voltage has not moved by a single bit from the base while the CPU's has.
        {
            static const char* rowProbeEnv = std::getenv("OLF_GPU_ROW_PROBE");
            static int rowProbe = rowProbeEnv != nullptr ? std::atoi(rowProbeEnv) : -1;
            if (rowProbe >= 0 && rowProbe < n) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                std::vector<int> hrp(n + 1);
                CUDA_CHECK(cudaMemcpy(hrp.data(), ctx->st->dRowPtr.p, (size_t) (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
                int k0 = hrp[rowProbe];
                int k1 = hrp[rowProbe + 1];
                std::vector<int> hci(k1 - k0);
                CUDA_CHECK(cudaMemcpy(hci.data(), ctx->st->dColIdx.as<int>() + k0, (size_t) (k1 - k0) * sizeof(int), cudaMemcpyDeviceToHost));
                std::vector<int> hmode(nS);
                for (int s2 = 0; s2 < nS; s2++) {
                    double fv = 0;
                    double xv = 0;
                    int mv = 0;
                    CUDA_CHECK(cudaMemcpy(&fv, ctx->dF.as<double>() + (size_t) s2 * n + rowProbe, sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(&xv, ctx->dX.as<double>() + (size_t) s2 * n + rowProbe, sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(&mv, ctx->dMode.as<int>() + (size_t) s2 * n + rowProbe, sizeof(int), cudaMemcpyDeviceToHost));
                    double tv = 0;
                    CUDA_CHECK(cudaMemcpy(&tv, ctx->dTarget.as<double>() + (size_t) s2 * n + rowProbe, sizeof(double), cudaMemcpyDeviceToHost));
                    std::vector<double> hv(k1 - k0);
                    CUDA_CHECK(cudaMemcpy(hv.data(), ctx->dVals.as<double>() + (size_t) s2 * nnz + k0, (size_t) (k1 - k0) * sizeof(double), cudaMemcpyDeviceToHost));
                    std::fprintf(stderr, "ROW_PROBE it=%d s=%d row=%d mode=%d F=%.12e x=%.12e target=%.12e nnz=%d :", it, s2, rowProbe, mv, fv, xv, tv, k1 - k0);
                    for (int k = 0; k < k1 - k0; k++) {
                        if (hv[k] != 0.0) {
                            std::fprintf(stderr, " [%d]=%.6e", hci[k], hv[k]);
                        }
                    }
                    std::fprintf(stderr, "\n");
                }
            }
        }
        static const bool linResidOn = std::getenv("OLF_GPU_LIN_RESID") != nullptr;
        if (linResidOn) {
            DeviceBuf rp;
            DeviceBuf rm;
            rp.alloc((size_t) n * nS * sizeof(double));
            rm.alloc((size_t) n * nS * sizeof(double));
            linResidRowBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                    ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), rp.as<double>(), rm.as<double>());   // B = the (repaired) RHS
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            std::vector<double> hrp((size_t) n * nS);
            std::vector<double> hrm((size_t) n * nS);
            std::vector<double> hf((size_t) n * nS);
            std::vector<double> hdx((size_t) n * nS);
            CUDA_CHECK(cudaMemcpy(hrp.data(), rp.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hrm.data(), rm.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hf.data(), ctx->dF.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hdx.data(), ctx->dDx.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                double mp = 0;
                double mm = 0;
                double mf = 0;
                double mdx = 0;
                int argp = -1;
                int argm = -1;
                for (int i = 0; i < n; i++) {
                    size_t t = (size_t) s * n + i;
                    if (std::fabs(hrp[t]) > mp) { mp = std::fabs(hrp[t]); argp = i; }
                    if (std::fabs(hrm[t]) > mm) { mm = std::fabs(hrm[t]); argm = i; }
                    mf = std::max(mf, std::fabs(hf[t]));
                    mdx = std::max(mdx, std::fabs(hdx[t]));
                }
                double best = std::min(mp, mm);
                int argw = mp < mm ? argp : argm;
                std::fprintf(stderr, "LIN_RESID it=%d slot=%d n=%d ||J.dx-F||inf=%.6e ||F||inf=%.6e "
                             "||dx||inf=%.6e ratio=%.3e worstRow=%d\n",
                             it, s, n, best, mf, mdx, mf > 0 ? best / mf : 0.0, argw);
            }
        }
        // LINEAR-SOLVE GUARD, first iteration of the first pass of every batched solve: a
        // factorization the UNPIVOTED uniform batch got wrong shows as a linear residual ratio of
        // 1e6..1e13 (rte6515's full busbar scope, 366 retained couplers: min pivots of 1e-18 under
        // any unpivoted ordering) while a sound one sits at 1e-9 (repaired rows, IR=2) and the hardest
        // measured case at 4e-4 (IR=0). Newton then never moves and the scenario surfaced as an
        // honest-looking MAX_ITERATION_REACHED; for the batch of ONE that is the BASE load flow of a
        // whole security analysis, which then reported NOTHING (6876 sections, GPU={}). Reject it
        // here (status -2): the base throws into the CPU fallback, a chunk scenario replays on the
        // CPU - never wrong, only un-accelerated. One SpMV per scenario, once per solve. Checked at
        // the SECOND iteration of the first pass: a first step from the flat start can be garbage
        // on a row that the first refactorization then heals (deviceRemoteSingletonMemberPin:
        // ratio 0.17, dx 2e13, converged to 1e-12 all the same), while a wrong factorization stays
        // at 1e6..1e13 on every iteration.
        if (outer == 0 && it == 1 && !deviceFixedIter) {   // it=1: a flat-start first step may be garbage and heal (singleton-pin fixture); a wrong factorization never moves
            DeviceBuf rp;
            DeviceBuf rm;
            DeviceBuf sc;
            rp.alloc((size_t) n * nS * sizeof(double));
            rm.alloc((size_t) n * nS * sizeof(double));
            sc.alloc((size_t) 3 * nS * sizeof(double));
            linResidRowBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                    ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), rp.as<double>(), rm.as<double>());
            linResidScoreBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(n, nS, rp.as<double>(), rm.as<double>(),
                    ctx->dF.as<double>(), sc.as<double>());
            std::vector<double> hsc((size_t) 3 * nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(hsc.data(), sc.p, (size_t) 3 * nS * sizeof(double), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                double ratio = hsc[3 * s + 2];
                if (!(ratio <= LIN_SOLVE_FAIL_RATIO)) {         // NaN counts as failed
                    if (std::getenv("OLF_GPU_REJECT_TRACE") != nullptr) {
                        std::fprintf(stderr, "GPU_REJECT slot=%d reason=linear-solve ratio=%.3e ||J.dx-B||inf=%.3e ||F||inf=%.3e\n",
                                     s, ratio, hsc[3 * s], hsc[3 * s + 1]);
                    }
                    traceReject(s, "linear-solve", outer, it);
                    res.status[s] = -3;                      // STATUS_LINEAR_SOLVE_FAILED: the one reject a pivoted retry can fix
                    res.iters[s] = it;
                    active[s] = 0;
                    live[s] = 0;
                    nActive--;
                }
            }
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
            hazRepairRightHandSide(ctx, 1, bS, s);
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matA, valsS));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matB, bS));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matX, dxS));
            if (!ctx->analyzed) {
                CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB));
                traceLuNnz(ctx->ds.handle, ctx->ds.data, "per-scenario", ctx->nScenarios);
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

    // ---- AcloadFlowEngine's realistic-voltage check, on this INTERMEDIATE state ----
    // OLF re-checks after every solver run once the last loop that can fix an unrealistic state has
    // run, and an unrealistic state ends the outer-loop cycle with UNREALISTIC_STATE: the run FAILS
    // and reports no violations. Which solve the check belongs to depends on the configured loop
    // order and on robust mode, so rather than reproduce that ordering the device routes any
    // scenario that is EVER unrealistic to the CPU replay, which applies OLF's rule itself. That is
    // stricter than OLF (a state it would have let a later loop fix also leaves the batch), so it
    // can cost batching share and can never change an answer.
    if (ctx->realisticCheck && solvePass && loopIdx > realisticFixingIdx) {
        std::vector<char> gate(nS);
        for (int s = 0; s < nS; s++) {
            gate[s] = (wasActive[s] && res.status[s] == 0 && live[s]) ? 1 : 0;
        }
        if (!ctx->dRealisticGate.p) ctx->dRealisticGate.alloc(nS * sizeof(char));
        if (!ctx->dRealisticBad.p) ctx->dRealisticBad.alloc(nS * sizeof(char));
        CUDA_CHECK(cudaMemcpyAsync(ctx->dRealisticGate.p, gate.data(), nS * sizeof(char),
                                   cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->dRealisticBad.p, 0, nS * sizeof(char), ctx->stream));
        realisticCheckBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dRealisticRow.as<char>(),
                ctx->dX.as<double>(), ctx->dRealisticGate.as<char>(),
                ctx->realisticMinV, ctx->realisticMaxV, ctx->dRealisticBad.as<char>());
        std::vector<char> bad(nS);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        CUDA_CHECK(cudaMemcpy(bad.data(), ctx->dRealisticBad.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
        static const bool traceUnrealistic = std::getenv("OLF_GPU_UNREALISTIC_TRACE") != nullptr;
        for (int s = 0; s < nS; s++) {
            if (bad[s]) {
                if (traceUnrealistic) {
                    std::fprintf(stderr, "GPU_UNREALISTIC_SWEEP slot=%d outer=%d loop=%d band=[%.4f, %.4f]\n",
                                 s, outer, loopOrder[loopIdx], ctx->realisticMinV, ctx->realisticMaxV);
                }
                traceReject(s, "unrealistic-state", outer, 0);
                res.status[s] = -2;                          // STATUS_REJECTED -> the CPU replays it
                live[s] = 0;
                active[s] = 0;
            }
        }
    }

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
        // OLF_GPU_RMQ_TRACE=<deviceRow>: every contribution this kernel adds to that member's
        // diagnostic q row, named by kind and element. The RL check pins on q = qBuf[row] + loadQ,
        // so a q that disagrees with OLF is answered here and nowhere else.
        static const char* rmqTraceEnv = std::getenv("OLF_GPU_RMQ_TRACE");
        int rmqTraceRow = rmqTraceEnv != nullptr ? std::atoi(rmqTraceEnv) : -1;
        fillRemoteMemberQBatch<<<grid(ctx->nRmq * nS), BLOCK, 0, ctx->stream>>>(ctx->nRmq, nS, n,
                ctx->nElements, ctx->cbOffset, ctx->obOffset,
                ctx->dRmqRow.as<int>(), ctx->dRmqElem.as<int>(), ctx->dRmqSide.as<int>(),
                ctx->dRmqKind.as<int>(), ctx->dRmqWeight.as<double>(),
                ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(),
                ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(),
                ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dRlActive.as<char>(),
                ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(),
                ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(),
                rmqTraceRow);
    }
    CUDA_CHECK(cudaMemsetAsync(ctx->dRlSwitched.p, 0, nS * sizeof(char), ctx->stream));
    // Apply ONLY the current phase's outer loop this pass (sequential, like OLF) — never both at once.
    bool inDsPhase = (loopOrder[loopIdx] == LOOP_DS);
    if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
        const char* kindName = "?";
        switch (loopOrder[loopIdx]) {
            case LOOP_DS:           kindName = "DistributedSlack"; break;
            case LOOP_HVDC_FREEZE:  kindName = "FreezingHvdcACEmulation"; break;
            case LOOP_HVDC_LIMITS:  kindName = "AcHvdcAcEmulationLimits"; break;
            case LOOP_HOST_SVC:     kindName = "HOST:SecondaryVoltageControl"; break;
            case LOOP_MONITORING:   kindName = "MonitoringVoltage"; break;
            case LOOP_RL:           kindName = "ReactiveLimits"; break;
            case LOOP_HOST_PHASE:   kindName = "HOST:PhaseControl"; break;
            case LOOP_HOST_TVC:     kindName = "HOST:TransformerVoltageControl"; break;
            case LOOP_HOST_TRPC:    kindName = "HOST:TransformerReactivePowerControl"; break;
            case LOOP_HOST_SHUNT:   kindName = "HOST:ShuntVoltageControl"; break;
            case LOOP_TVC_KERNEL:   kindName = "TransformerVoltageControl"; break;
            case LOOP_SHUNT_KERNEL: kindName = "ShuntVoltageControl"; break;
            case LOOP_PHASE_KERNEL: kindName = ctx->nCph > 0 ? "PhaseControl" : "IncrementalPhaseControl"; break;
            case LOOP_TRPC_KERNEL:  kindName = "TransformerReactivePowerControl"; break;
            case LOOP_TVD_KERNEL:   kindName = ctx->tvdMode == 1 ? "SimpleTransformerVoltageControl" : "TransformerVoltageControl"; break;
            case LOOP_SVD_KERNEL:   kindName = "ShuntVoltageControl"; break;
            case LOOP_SVC_KERNEL:   kindName = "SecondaryVoltageControl"; break;
        }
        int nSw = 0;
        {
            std::vector<char> sw(nS);
            CUDA_CHECK(cudaMemcpy(sw.data(), ctx->dRlSwitched.p, nS, cudaMemcpyDeviceToHost));
            for (char c : sw) nSw += c ? 1 : 0;
        }
        int nPinned = 0;
        if (ctx->nrl > 0) {
            std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS);
            CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
            for (char c : lt) nPinned += c ? 1 : 0;
        }
        std::fprintf(stderr, "BATCH_TRACE pass done: outer=%d loop=%d/%d kind=%s solved=%d changedScenarios=%d iters=%d pinned=%d\n",
                     outer, loopIdx, nLoops, kindName, solvePass ? 1 : 0, nSw, gLastPassIters, nPinned);
    }
    if (inDsPhase && getenv("OLF_GPU_DS_DEBUG") && nS > 0) {
        // Is scenario 0 actually CONVERGED at the moment the DS loop reads its mismatch? OLF checks
        // its slack mismatch on a converged state; if ours is only part-way there, the mismatch it
        // measures is not the same quantity, and a threshold at the same scale flips.
        std::vector<double> f0(n);
        CUDA_CHECK(cudaMemcpy(f0.data(), ctx->dF.as<double>(), n * sizeof(double), cudaMemcpyDeviceToHost));
        double ssq = 0.0;
        double inf = 0.0;
        for (int i = 0; i < n; i++) {
            ssq += f0[i] * f0[i];
            inf = fmax(inf, fabs(f0[i]));
        }
        double slackM = std::numeric_limits<double>::quiet_NaN();
        double retain0 = std::numeric_limits<double>::quiet_NaN();
        if (ctx->dsSlackPhiRow >= 0) {
            std::vector<double> q0(n);
            CUDA_CHECK(cudaMemcpy(q0.data(), ctx->dQbuf.as<double>(), n * sizeof(double), cudaMemcpyDeviceToHost));
            slackM = q0[ctx->dsSlackPhiRow] - ctx->dsSlackTargetP;
            if (ctx->dDsRetain.p) CUDA_CHECK(cudaMemcpy(&retain0, ctx->dDsRetain.p, sizeof(double), cudaMemcpyDeviceToHost));
        }
        std::fprintf(stderr, "[DS-CONV] outer=%d innerNewtonIters=%d ||F||2=%.9e ||F||inf=%.9e slackMismatchFromAnchor=%.9f retain=%.9f rlActive=%d\n",
                     outer, gLastPassIters, std::sqrt(ssq), inf, slackM, retain0, (int) rlActive[0]);
    }
    int loopKind = loopOrder[loopIdx];
    // DIAGNOSTIC (OLF_GPU_HOST_LOOP_NOOP=1): keep the host slot in the sweep but treat it as an
    // in-kernel STABLE loop — no break, no host round trip. Separates "the extra slot changed the
    // sweep" from "splitting the call changed the sweep".
    static const bool hostLoopNoop = std::getenv("OLF_GPU_HOST_LOOP_NOOP") != nullptr;
    if (hostLoopNoop && loopKind >= LOOP_HOST_SVC && loopKind <= LOOP_HOST_SHUNT) {
        loopKind = -1;                                       // falls through every branch: a STABLE check
    }
    if (loopKind >= LOOP_HOST_SVC && loopKind <= LOOP_HOST_SHUNT) {   // any LOOP_HOST_* kind: pause for the host
        // Hand the sequence back. Everything the device holds — x, targets, row modes, RL pins,
        // monitoring activations, absorbed slack, HVDC statuses — stays put in the context; only
        // these host-side locals need saving.
        ctx->pausedLoopIdx = loopIdx;
        ctx->pausedSeqChanged = seqChanged ? 1 : 0;
        ctx->pausedProdIters = prodIters;
        ctx->pausedLive = live;
        ctx->pausedActive = active;
        ctx->pausedStatus = res.status;
        ctx->driverPaused = 1;
        ctx->pausedLoopKind = loopKind;                      // WHICH host loop to run
        break;
    }
    if (loopKind == LOOP_TVC_KERNEL) {
        tvcStep();
    } else if (loopKind == LOOP_SHUNT_KERNEL) {
        shcStep();
    } else if (loopKind == LOOP_PHASE_KERNEL) {
        if (ctx->nCph > 0) cphStep(); else flcStep(6);       // continuous, or incremental kinds 1 + 2
    } else if (loopKind == LOOP_TRPC_KERNEL) {
        flcStep(1);                                          // kind 0 (RPC)
    } else if (loopKind == LOOP_TVD_KERNEL) {
        tvdStep();
    } else if (loopKind == LOOP_SVD_KERNEL) {
        svdStep();
    } else if (loopKind == LOOP_SVC_KERNEL) {
        svcStep();
    } else if (loopKind == LOOP_MONITORING) {
        monitoringLoopBatch<<<grid(ctx->nmon * nS), BLOCK, 0, ctx->stream>>>(
                ctx->nmon, nS, n, ctx->dMonVRow.as<int>(), ctx->dMonBand.as<double>(),
                ctx->dMonDeltaPtr.as<int>(), ctx->dMonDeltaRow.as<int>(),
                ctx->dMonDeltaMode.as<int>(), ctx->dMonDeltaTarget.as<double>(),
                ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dMode.as<int>(),
                ctx->dMonActivated.as<double>(), ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>());
    } else if (loopKind == LOOP_HVDC_FREEZE || loopKind == LOOP_HVDC_LIMITS) {
        hvdcAcEmulationLoopBatch<<<grid(ctx->nhv * nS), BLOCK, 0, ctx->stream>>>(
                ctx->nhv, nS, n, ctx->nElements, ctx->hvOffset,
                loopKind == LOOP_HVDC_FREEZE ? 0 : 1,
                ctx->dHvElem.as<int>(), ctx->dHvBus1Row.as<int>(), ctx->dHvBus2Row.as<int>(),
                ctx->dHvIn.as<double>(), ctx->dHvCtgOut.as<char>(),
                ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dDisabled.as<char>(),
                ctx->dHvStatus.as<int>(), ctx->dHvAppliedD1.as<double>(), ctx->dHvAppliedD2.as<double>(),
                ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>());
    } else if (inDsPhase) {                                // distributed-slack outer loop: spread the slack mismatch
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
                    ctx->dsSlackMarginUp, ctx->dsSlackMarginDown,
                    ctx->nDsIslanded > 0 ? ctx->dDsIslanded.as<char>() : nullptr,
                    ctx->dDsFactor.as<double>());
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
                    ctx->nDsIslanded > 0 ? ctx->dDsIslanded.as<char>() : nullptr,
                    ctx->ngen, ctx->dgBus.as<int>(), ctx->dgFactor.as<double>(),
                    ctx->dgMaxDelta.as<double>(), ctx->dgMinDelta.as<double>(),
                    ctx->dgMarginUp.as<double>(), ctx->dgMarginDown.as<double>(),
                    ctx->dBusGenStart.as<int>(), ctx->dBusGenEnd.as<int>(),
                    ctx->dGenCum.as<double>(), ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr,
                    ctx->dGenRemoved.as<char>(), ctx->dGenFactorWork.as<double>(),
                    ctx->nDsLostGen, ctx->nDsLostGen > 0 ? ctx->dDsLostGen.as<int>() : nullptr,
                    ctx->dTarget.as<double>(), ctx->dRlSwitched.as<char>(), ctx->dDsResidual.as<char>(),
                    ctx->dDsRetainApplied.as<double>(),
                    dsDebugTraceSlot());
        }
    } else {                                               // reactive-limits outer loop: switch PV->PQ over Q limit
        if (loopKind == LOOP_RL && ctx->nrl > 0 && !skipRlPhase) {                  // skipRlPhase: a monitoring interleave solve — the
            if (useDynRlLimits) {                            // curve limits at the CURRENT (post-DS) targets,
                rlCurveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, ctx->ncv, n, ctx->cvExtrapolate,
                        ctx->dCvIdx.as<int>(), ctx->dCvIn.as<double>(), ctx->dCvPts.as<double>(),
                        ctx->dRlMinQ.as<double>(), ctx->dRlMaxQ.as<double>(), ctx->dRlLoadQ.as<double>(),
                        ctx->dTarget.as<double>(),
                        nRlQOv, nRlQOv > 0 ? dRlQOvSlotEntry.as<int>() : nullptr,
                        nRlQOv > 0 ? dRlQOvMinMax.as<double>() : nullptr,
                        nCvOv > 0 ? dCvOvPtr.as<int>() : nullptr,
                        nCvOv > 0 ? dCvOvEntry.as<int>() : nullptr,
                        nCvOv > 0 ? dCvOvVal.as<double>() : nullptr,
                        ctx->ngen > 0 ? ctx->dGenCum.as<double>() : nullptr,
                        ctx->ngen > 0 ? ctx->dGenCumValid.as<char>() : nullptr, ctx->ngen,
                        ctx->dRlMinQDyn.as<double>(), ctx->dRlMaxQDyn.as<double>(),
                        ctx->dRlLoadQDyn.as<double>());
            }
            CUDA_CHECK(cudaMemsetAsync(ctx->dRlPassSwitched.p, 0, static_cast<size_t>(ctx->nrl) * nS, ctx->stream));
            static const char* noPinEnv = std::getenv("OLF_GPU_RL_NOPIN_ROW");
            int rlNoPinRow = noPinEnv != nullptr ? std::atoi(noPinEnv) : -1;
            reactiveLimitsBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(
                    ctx->nSvc > 0 ? ctx->dSvcRlVt.as<double>() : nullptr,
                    ctx->nrl, nS, n, REACTIVE_LIMIT_TOL, RL_MAX_SWITCH,
                    ctx->dRlVrow.as<int>(),
                    useDynRlLimits ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                    useDynRlLimits ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                    useDynRlLimits ? ctx->dRlLoadQDyn.as<double>() : ctx->dRlLoadQ.as<double>(),
                    ctx->dRlVtarget.as<double>(), ctx->dQbuf.as<double>(), ctx->dX.as<double>(), ctx->dRlActive.as<char>(),
                    ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                    ctx->dRlLimitType.as<char>(), ctx->dRlSwitchCount.as<char>(), ctx->dRlSwitched.as<char>(),
                    ctx->dRlPassSwitched.as<char>(), ctx->dRlSavedMode.as<char>(),
                    nRlVtOv, nRlVtOv > 0 ? dRlVtOvSlotIdx.as<int>() : nullptr,
                    nRlVtOv > 0 ? dRlVtOvVal.as<double>() : nullptr,
                    ctx->dRowSlope.as<double>(), ctx->dRlSavedTarget.as<double>(), useDynRlLimits ? 1 : 0,
                    ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                    ctx->nRlRemote > 0 ? ctx->dRlCmpVrow.as<int>() : nullptr,
                    ctx->dRlSvcBmin.as<double>(), ctx->dRlSvcBmax.as<double>(), ctx->dRlV2Base.as<double>(),
                    ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
                    ctx->rlRobust ? ctx->dRlInitTgtQ.as<double>() : nullptr,
                    ctx->rlVLoRobust, ctx->rlVHiRobust, ctx->rlRobust, rlNoPinRow);
            if (ctx->nRlPilotSelf > 0) {
                // a pilot that is its own controller pinned while a member of its group is not (or
                // the reverse): the device has no row for OLF's layout of that state - CPU replay
                if (!ctx->dRlMixedBad.p) ctx->dRlMixedBad.alloc(nS * sizeof(char));
                CUDA_CHECK(cudaMemsetAsync(ctx->dRlMixedBad.p, 0, nS * sizeof(char), ctx->stream));
                rlPilotSelfMixedGate<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS,
                        ctx->dRlRemote.as<char>(), ctx->dRlCmpVrow.as<int>(), ctx->dRlLimitType.as<char>(),
                        ctx->dRlMixedBad.as<char>());
                std::vector<char> mixed(nS);
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                CUDA_CHECK(cudaMemcpy(mixed.data(), ctx->dRlMixedBad.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
                for (int s = 0; s < nS; s++) {
                    if (mixed[s] && live[s]) {
                        traceReject(s, "pilot-self-mixed-pin", outer, 0);
                        res.status[s] = -2;
                        active[s] = 0;
                        live[s] = 0;
                    }
                }
            }
            if (ctx->rlRobust && ctx->nRlRemote > 0) {       // "set V to a safe 1 pu", as its own pass
                rlRobustVResetBatch<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(
                        ctx->nrl, nS, n, ctx->dRlVrow.as<int>(), ctx->dRlRemote.as<char>(),
                        ctx->dRlActive.as<char>(), ctx->dRlPassSwitched.as<char>(), ctx->dX.as<double>(),
                        ctx->rlVLoRobust, ctx->rlVHiRobust);
            }
            if (ctx->hasRlElection && ctx->allPvPinnable) {  // the rule can only fire when every PV is pinnable
                rlForcedPvElectBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n,
                        ctx->dRlVrow.as<int>(),
                        ctx->nRlRemote > 0 ? ctx->dRlRemote.as<char>() : nullptr,
                        ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
                        ctx->dRlElectNomV.as<double>(), ctx->dRlElectPRow.as<int>(),
                        ctx->dRlElectStaticP.as<double>(), ctx->dRlElectIdRank.as<int>(),
                        ctx->dRlLimitType.as<char>(), ctx->dRlSwitchCount.as<char>(), ctx->dRlPassSwitched.as<char>(),
                        ctx->dRlSavedMode.as<char>(), ctx->dRlSavedTarget.as<double>(),
                        ctx->dMode.as<int>(), ctx->dTarget.as<double>(), ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>());
            }
            if (ctx->nRlRemote > 0 && ctx->nDqMembers > 0) {
                // keep the survivors' DISTR_Q row targets in step with the pin states (plan-B3)
                updateDistrQTargetsBatch<<<grid(ctx->nDqMembers * nS), BLOCK, 0, ctx->stream>>>(
                        ctx->nDqMembers, nS, n, ctx->nrl,
                        ctx->dDqMemberKey.as<double>(), ctx->dDqMemberRlEntry.as<int>(),
                        ctx->dDqMemberGroupStart.as<int>(), ctx->dDqMemberGroupCount.as<int>(),
                        ctx->dDqMemberRow.as<int>(), ctx->dDqMemberHasRow.as<char>(),
                        ctx->dRlVrow.as<int>(),
                        useDynRlLimits ? ctx->dRlMinQDyn.as<double>() : ctx->dRlMinQ.as<double>(),
                        useDynRlLimits ? ctx->dRlMaxQDyn.as<double>() : ctx->dRlMaxQ.as<double>(),
                        useDynRlLimits ? ctx->dRlLoadQDyn.as<double>() : ctx->dRlLoadQ.as<double>(), ctx->dRlLimitType.as<char>(), useDynRlLimits ? 1 : 0,
                        ctx->dTarget.as<double>(), ctx->dRlActive.as<char>(),
                        ctx->dDqMemberTargetQ.as<double>(),
                        ctx->hasDqQrGroups ? ctx->dDqMemberQrGroup.as<int>() : nullptr,
                        ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr,
                        ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr,
                        ctx->dMode.as<int>(),
                        ctx->hasDqQrGroups ? ctx->dQrPilotQ.as<double>() : nullptr, ctx->st->nQr,
                        ctx->nDqInact, ctx->nDqInact ? ctx->dDqInact.as<int>() : nullptr,
                        ctx->nFrozenRow, ctx->nFrozenRow ? ctx->dFrozenRow.as<int>() : nullptr,
                        ctx->nDqTqOv, ctx->nDqTqOv ? ctx->dDqTqOvKey.as<int>() : nullptr,
                        ctx->nDqTqOv ? ctx->dDqTqOvVal.as<double>() : nullptr);
            }
            if (ctx->hasDqQrGroups && ctx->st->nQr > 0) {    // all-pinned corner: pilot vEq flip/restore
                updatePilotVeqBatch<<<grid(ctx->st->nQr * nS), BLOCK, 0, ctx->stream>>>(
                        ctx->st->nQr, nS, n,
                        ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>(), ctx->dQrGenVc.as<char>(),
                        ctx->dQrPilotQ.as<double>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                        ctx->dQrVeqOff.as<char>(), ctx->dQrSavedVT.as<double>(),
                        ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>(),
                        ctx->nFrozenRow, ctx->nFrozenRow ? ctx->dFrozenRow.as<int>() : nullptr);
            }
            if (ctx->nPilotFlip > 0) {                      // pilot stopped controlling AND every survivor pinned
                pilotRowAllPinnedBatch<<<grid(ctx->nPilotFlip), BLOCK, 0, ctx->stream>>>(
                        ctx->nPilotFlip, nS, n, ctx->nrl,
                        ctx->dPfSlot.as<int>(), ctx->dPfRow.as<int>(), ctx->dPfQTarget.as<double>(),
                        ctx->dPfOff.as<int>(), ctx->dPfCount.as<int>(), ctx->dPfRlPool.as<int>(),
                        ctx->dRlLimitType.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>(),
                        ctx->dPfFlipped.as<char>(), ctx->dPfSavedVT.as<double>(),
                        ctx->dRlActive.as<char>(), ctx->dRlSwitched.as<char>());
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(rlSwitched.data(), ctx->dRlSwitched.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
    for (int s = 0; s < nS; s++) {
        if (loopRejected[s]) {                              // a device loop routed the scenario to the CPU fallback
            traceReject(s, "device-loop-routed", outer, 0);
            res.status[s] = -2;
            live[s] = 0;
            active[s] = 0;
            rlSwitched[s] = 0;
            loopRejected[s] = 0;
        }
    }
    // OLF_GPU_RL_TRACE_VROW=<v row>: one LOCAL entry's pin state at every check (the trace below
    // otherwise names remote entries only) - with its live limits and row target, which is what a
    // curve-limit drift moves.
    static const char* rlTraceVrowEnv = std::getenv("OLF_GPU_RL_TRACE_VROW");
    static const int rlTraceVrow = rlTraceVrowEnv != nullptr ? std::atoi(rlTraceVrowEnv) : -1;
    if (!inDsPhase && (ctx->nRlRemote > 0 || nS == 1 || rlTraceVrow >= 0) && ctx->nrl > 0
            && (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr || rlTraceVrow >= 0)) {   // RL trace
        std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS);
        std::vector<char> rem(ctx->nrl);
        std::vector<int> vr(ctx->nrl);
        std::vector<double> qb(static_cast<size_t>(n) * nS);
        std::vector<double> lq(ctx->nrl);
        CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size(), cudaMemcpyDeviceToHost));
        if (ctx->nRlRemote > 0) {
        CUDA_CHECK(cudaMemcpy(rem.data(), ctx->dRlRemote.p, rem.size(), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaMemcpy(vr.data(), ctx->dRlVrow.p, vr.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qb.data(), ctx->dQbuf.p, qb.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lq.data(), ctx->dRlLoadQ.p, lq.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> hMode(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hMode.data(), ctx->dMode.p, hMode.size() * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<double> hMaxQ(static_cast<size_t>(ctx->nrl) * (ctx->ncv > 0 ? nS : 1));
        CUDA_CHECK(cudaMemcpy(hMaxQ.data(), ctx->ncv > 0 ? ctx->dRlMaxQDyn.p : ctx->dRlMaxQ.p, hMaxQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<char> hRlAct(nS);
        CUDA_CHECK(cudaMemcpy(hRlAct.data(), ctx->dRlActive.p, nS, cudaMemcpyDeviceToHost));
        // The STATE the pin decision is taken at. Without it, a q that disagrees with OLF cannot be
        // told apart from a q evaluated at a different (or not yet converged) operating point - the
        // same quantity at a different V is not evidence of a wrong formula.
        std::vector<double> hXrl(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hXrl.data(), ctx->dX.p, hXrl.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> hMinQ(hMaxQ.size());
        CUDA_CHECK(cudaMemcpy(hMinQ.data(), ctx->ncv > 0 ? ctx->dRlMinQDyn.p : ctx->dRlMinQ.p, hMinQ.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> hTgt(static_cast<size_t>(n) * nS);
        CUDA_CHECK(cudaMemcpy(hTgt.data(), ctx->dTarget.p, hTgt.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int s2 = 0; s2 < nS; s2++) {
            for (int i2 = 0; i2 < ctx->nrl; i2++) {
                if (rlTraceVrow >= 0 && vr[i2] == rlTraceVrow) {
                    std::fprintf(stderr, "RL_VROW s=%d entry=%d lt=%d q=%.9f minQ=%.9f maxQ=%.9f target=%.9f mode=%d v=%.9f loadQ=%.9f\n",
                                 s2, i2, (int) lt[(size_t) s2 * ctx->nrl + i2], qb[(size_t) s2 * n + vr[i2]] + lq[i2],
                                 hMinQ[ctx->ncv > 0 ? (size_t) s2 * ctx->nrl + i2 : (size_t) i2],
                                 hMaxQ[ctx->ncv > 0 ? (size_t) s2 * ctx->nrl + i2 : (size_t) i2],
                                 hTgt[(size_t) s2 * n + vr[i2]], hMode[(size_t) s2 * n + vr[i2]],
                                 hXrl[(size_t) s2 * n + vr[i2]], lq[i2]);
                }
                if ((rem[i2] || nS == 1) && std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
                    std::fprintf(stderr, "RL_TRACE s=%d entry=%d remote=%d lt=%d switched=%d q=%.9f mode=%d tol=%.3e maxQ=%.6f ncv=%d rlActive=%d v=%.9f qbuf=%.9f\n",
                                 s2, i2, (int) rem[i2], (int) lt[(size_t) s2 * ctx->nrl + i2], (int) rlSwitched[s2],
                                 qb[(size_t) s2 * n + vr[i2]] + lq[i2], hMode[(size_t) s2 * n + vr[i2]], reactiveLimitTol,
                                 hMaxQ[ctx->ncv > 0 ? (size_t) s2 * ctx->nrl + i2 : (size_t) i2], ctx->ncv, (int) hRlAct[s2],
                                 hXrl[(size_t) s2 * n + vr[i2]], qb[(size_t) s2 * n + vr[i2]]);
                }
            }
        }
    }
    if (inDsPhase && (ctx->nds > 0 || ctx->dsSlackOnly) && getenv("OLF_GPU_DS_DEBUG")) {
        // EVERY scenario, not just 0: the question "did OLF's DS fire on this check and did ours?"
        // is per scenario, and the scenario that disagrees is rarely slot 0.
        std::vector<double> qbufH(static_cast<size_t>(n) * nS);
        std::vector<double> retainH(nS, 0.0);
        std::vector<double> absH(nS, 0.0);
        CUDA_CHECK(cudaMemcpy(qbufH.data(), ctx->dQbuf.p, sizeof(double) * qbufH.size(), cudaMemcpyDeviceToHost));
        if (ctx->nds > 0 && ctx->dDsRetain.p) {
            CUDA_CHECK(cudaMemcpy(retainH.data(), ctx->dDsRetain.p, sizeof(double) * nS, cudaMemcpyDeviceToHost));
        }
        if (ctx->dsSlackOnly && ctx->dDsSlackAbsorbed.p) {
            CUDA_CHECK(cudaMemcpy(absH.data(), ctx->dDsSlackAbsorbed.p, sizeof(double) * nS, cudaMemcpyDeviceToHost));
        }
        for (int s = 0; s < nS; s++) {
            double slackP = qbufH[static_cast<size_t>(s) * n + ctx->dsSlackPhiRow];
            double rem = (slackP - ctx->dsSlackTargetP) - retainH[s] - absH[s];
            std::fprintf(stderr, "[DS-DEV] outer=%d s=%d live=%d slackP=%.9f slackTargetP=%.9f retain=%.9f rem=%.9f tol=%.3e fires=%d\n",
                    outer, s, (int) live[s], slackP, ctx->dsSlackTargetP, retainH[s], rem, SLACK_DIST_TOL,
                    std::fabs(rem) > SLACK_DIST_TOL ? 1 : 0);
        }
    }
    if (inDsPhase && ctx->nds > 0 && dsRejectResidual) {
        // slackDistributionFailureBehavior != LEAVE_ON_SLACK_BUS: a scenario whose participants all
        // saturated must NOT be reported converged with the residual absorbed at the slack — route it
        // to the CPU fallback (status -2), which replays it with the run's real failure behavior.
        std::vector<char> dsResidual(nS);
        CUDA_CHECK(cudaMemcpy(dsResidual.data(), ctx->dDsResidual.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            if (dsResidual[s] && live[s]) {
                traceReject(s, "ds-residual-saturated", outer, 0);
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
    // OLF_GPU_LOOP_TRACE=<slot>: the device side of OLF's OL_CHECK - which loop this pass ran, whether
    // it found the slot UNSTABLE (a re-solve follows) and whether the whole phase is still unstable.
    // Two arms that agree on every residual and still stop apart differ in this SEQUENCE.
    static const char* loopTraceEnv = std::getenv("OLF_GPU_LOOP_TRACE");
    static const int loopTraceSlot = loopTraceEnv != nullptr ? std::atoi(loopTraceEnv) : -1;
    if (loopTraceSlot >= 0 && loopTraceSlot < nS) {
        std::fprintf(stderr, "LOOP_PASS s=%d outer=%d loopIdx=%d kind=%d solved=%d switched=%d nChanged=%d live=%d status=%d\n",
                     loopTraceSlot, outer, loopIdx, loopOrder[loopIdx], solvePass ? 1 : 0,
                     (int) rlSwitched[loopTraceSlot], nChanged, (int) live[loopTraceSlot], res.status[loopTraceSlot]);
    }
    if (nChanged > 0) {                                     // this outer loop is still unstable
        for (int s = 0; s < nS; s++) {
            if (rlSwitched[s] && ++prodIters[s] >= RL_MAX_OUTER) {
                // This scenario exhausted ITS outer-iteration budget mid-sequence. Never report it
                // converged on a half-finished RL/DS fixpoint — reject to the CPU fallback, which
                // replays it under OLF's own maxOuterLoopIterations/UNSTABLE handling.
                traceReject(s, "outer-loop-budget", outer, 0);
                res.status[s] = -2;
                rlSwitched[s] = 0;
                live[s] = 0;
            }
            active[s] = rlSwitched[s];                      // re-solve only the scenarios it changed, stay in this phase
        }
        seqChanged = true;
        solvePass = true;                                   // mutated targets/modes -> a real re-solve is due
    } else {                                               // this outer loop stabilized
        if (++loopIdx >= nLoops) {                          // ...move to the next loop in OLF's order
            if (!seqChanged) {
                break;                                      // a full sweep re-solved nothing -> converged
            }
            seqChanged = false;
            loopIdx = 0;                                    // restart the sweep at the first loop
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
            rlStateOut[k] = (lt[k] & 7) | (sc[k] << 3);
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

// OLF_GPU_PHASE_TIME: split one solveBatchContext JNI call into the three things it does —
// marshalling the Java arrays in, the solve itself, and packing the per-scenario result arrays out.
// The Java-side timer sees only the total, and on a 642-scenario chunk the total carries roughly a
// second of fixed cost that the Newton iterations do not explain.
static bool jniPhaseTime() {
    static const bool on = std::getenv("OLF_GPU_PHASE_TIME") != nullptr;
    return on;
}

static long long nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

extern long long gOuterPasses;
extern long long gNewtonIters;
extern long long gSolvePasses;
extern long long gActiveIterSum;
extern long long gWidthIterSum;
extern long long gCompactIters;

static long long marshalNs = 0;
static long long solveNs = 0;
static long long packNs = 0;
static long long calls = 0;

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveBatchContext(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jx0Base, jdoubleArray jtargetBase,
        jobjectArray jrowMode, jintArray jbaseRowMode, jobjectArray jdisabledMask,
        jintArray jtargetOverrideIndex, jdoubleArray jtargetOverrideValue,
        jintArray jdsOverrideBus, jdoubleArray jdsOverrideFactorDelta, jdoubleArray jdsOverrideBaseDelta,
        jdoubleArray jdsOverrideMaxDelta, jdoubleArray jdsOverrideMinDelta, jdoubleArray jdsAnchorShift,
        jdoubleArray jdsSlackShift, jintArray jrlVtOvSlotIdx, jdoubleArray jrlVtOvVal,
        jintArray jrlQOvSlotEntry, jdoubleArray jrlQOvMinMax,
        jintArray jcvOvSlotEntry, jdoubleArray jcvOvStepScale, jint maxIter, jdouble tol,
        jdouble reactiveLimitTol, jdouble slackDistTol, jdouble convEpsPerEq,
        jint lsMode, jdouble maxDv, jdouble maxDphi, jintArray jvarType, jdoubleArray jx0Full,
        jintArray jShuntOvSlotElem, jdoubleArray jShuntOvGb,
        jintArray jRlStateIn, jintArray jRlStateOut,
        jintArray jR1OvSlotElem, jdoubleArray jR1OvVal,
        jint maxOuterProductive, jint rlMaxSwitch, jint dsRejectResidual, jint skipRlPhase,
        jint nziBase, jintArray jZiOnSlotElem, jintArray jZiOffSlotElem) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    // CONTRACT: reactive-limit entries require OLF's forced-PV election keys. switchPvPq never lets
    // a network reach zero PV buses ("all PV buses should switch PQ, strongest one will stay PV"),
    // and rlForcedPvElectBatch is the only thing on the device that reproduces it. A context that
    // installed RL entries without the keys would pin every candidate and silently diverge from the
    // CPU (measured 1.9 kV on the forced-PV network). GpuBatchOuterLoopTables.install always sets
    // them, so reaching this is a table-installation bug — say so instead of routing the scenario
    // away and hiding it.
    if (ctx != nullptr && ctx->nrl > 0 && !ctx->hasRlElection) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "solveBatchContext: reactive-limit entries installed without the forced-PV "
                      "election keys (setBatchRlElection) — OLF's strongest-bus-stays-PV rule "
                      "cannot be reproduced on the device");
        return nullptr;
    }
    long long tMarshal0 = jniPhaseTime() ? nowNs() : 0;
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
        std::vector<int> rlQOvSlotEntryVec;
        std::vector<double> rlQOvMinMaxVec;
        if (jrlQOvSlotEntry != nullptr && jrlQOvMinMax != nullptr) {
            rlQOvSlotEntryVec.resize(env->GetArrayLength(jrlQOvSlotEntry));
            env->GetIntArrayRegion(jrlQOvSlotEntry, 0, static_cast<jsize>(rlQOvSlotEntryVec.size()), rlQOvSlotEntryVec.data());
            rlQOvMinMaxVec.resize(env->GetArrayLength(jrlQOvMinMax));
            env->GetDoubleArrayRegion(jrlQOvMinMax, 0, static_cast<jsize>(rlQOvMinMaxVec.size()), rlQOvMinMaxVec.data());
        }
        std::vector<int> cvOvSlotEntryVec;
        std::vector<double> cvOvStepScaleVec;
        if (jcvOvSlotEntry != nullptr && jcvOvStepScale != nullptr) {
            cvOvSlotEntryVec.resize(env->GetArrayLength(jcvOvSlotEntry));
            env->GetIntArrayRegion(jcvOvSlotEntry, 0, static_cast<jsize>(cvOvSlotEntryVec.size()), cvOvSlotEntryVec.data());
            cvOvStepScaleVec.resize(env->GetArrayLength(jcvOvStepScale));
            env->GetDoubleArrayRegion(jcvOvStepScale, 0, static_cast<jsize>(cvOvStepScaleVec.size()), cvOvStepScaleVec.data());
        }
        if (jrlVtOvSlotIdx != nullptr && jrlVtOvVal != nullptr) {
            rlVtOvSlotIdxVec.resize(env->GetArrayLength(jrlVtOvSlotIdx));
            env->GetIntArrayRegion(jrlVtOvSlotIdx, 0, static_cast<jsize>(rlVtOvSlotIdxVec.size()), rlVtOvSlotIdxVec.data());
            rlVtOvValVec.resize(env->GetArrayLength(jrlVtOvVal));
            env->GetDoubleArrayRegion(jrlVtOvVal, 0, static_cast<jsize>(rlVtOvValVec.size()), rlVtOvValVec.data());
        }

        if (jniPhaseTime()) {
            marshalNs += nowNs() - tMarshal0;
        }
        long long tSolve0 = jniPhaseTime() ? nowNs() : 0;
        res = solveBatchContext(ctx, x0.data(), target.data(), rowMode.data(), baseRowMode.data(), disabled.data(),
                                nOv, ovIndex.data(), ovValue.data(),
                                nDsOv > 0 ? dsOvBus.data() : nullptr, dsOvFactorDelta.data(),
                                dsOvBaseDelta.data(), dsOvMaxDelta.data(), dsOvMinDelta.data(),
                                dsAnchorVec.empty() ? nullptr : dsAnchorVec.data(),
                                dsSlackShiftVec.empty() ? nullptr : dsSlackShiftVec.data(),
                                static_cast<int>(rlVtOvValVec.size()),
                                rlVtOvSlotIdxVec.empty() ? nullptr : rlVtOvSlotIdxVec.data(),
                                rlVtOvValVec.empty() ? nullptr : rlVtOvValVec.data(),
                                static_cast<int>(rlQOvMinMaxVec.size() / 3),
                                rlQOvSlotEntryVec.empty() ? nullptr : rlQOvSlotEntryVec.data(),
                                rlQOvMinMaxVec.empty() ? nullptr : rlQOvMinMaxVec.data(),
                                static_cast<int>(cvOvStepScaleVec.size() / 2),
                                cvOvSlotEntryVec.empty() ? nullptr : cvOvSlotEntryVec.data(),
                                cvOvStepScaleVec.empty() ? nullptr : cvOvStepScaleVec.data(), maxIter, tol,
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
        if (jniPhaseTime()) {
            solveNs += nowNs() - tSolve0;
        }
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
    long long tPack0 = jniPhaseTime() ? nowNs() : 0;
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
    if (jniPhaseTime()) {
        packNs += nowNs() - tPack0;
        calls++;
        fprintf(stderr, "JNI_PHASE_TIME calls=%lld marshal=%lld ms solve=%lld ms pack=%lld ms "
                "| outerPasses=%lld (solving=%lld) newtonIters=%lld activeIterSum=%lld widthIterSum=%lld compactIters=%lld\n",
                calls, marshalNs / 1000000, solveNs / 1000000, packNs / 1000000,
                gOuterPasses, gSolvePasses, gNewtonIters, gActiveIterSum, gWidthIterSum, gCompactIters);
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

// The per-scenario ROW MODES as the device ENDED with them, the companion to readBatchTargets.
// A row's equation is (mode, target) together, and the outer-loop kernels rewrite BOTH: the host's
// rowModePerScenario is only what SETUP asked for, so reading a target without its final mode can
// say "the target is right" about a row that is no longer the equation it was set up to be. That is
// the whole failure shape of the device-target-writer class, where a wrong row converges cleanly and
// nothing flags it.
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchRowModes(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    size_t nTot = static_cast<size_t>(ctx->n) * ctx->nScenarios;
    std::vector<int> host(nTot);
    cudaMemcpy(host.data(), ctx->dMode.p, nTot * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(nTot));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(nTot), host.data());
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
    int dqTraceRow = gpuDqTraceRow();
    if (ctx->ndq) fillDistrQBatch<<<grid(ctx->ndq * nS), BLOCK, 0, ctx->stream>>>(ctx->ndq, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->obOffset, ctx->st->dDqElem.as<int>(), ctx->st->dDqSide.as<int>(), ctx->st->dDqKind.as<int>(), ctx->st->dDqWeight.as<double>(), ctx->st->dDqDest.as<int>(), ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nDqOv, ctx->nDqOv ? ctx->dDqOvIdx.as<int>() : nullptr, ctx->nDqOv ? ctx->dDqOvW.as<double>() : nullptr, ctx->dMode.as<int>(), ctx->nDqInact,
                ctx->nDqInact ? ctx->dDqInact.as<int>() : nullptr,
                ctx->nDqDyn ? ctx->dDqKMember.as<int>() : nullptr, ctx->nDqDyn || ctx->hasDqQrGroups ? ctx->dDqJMember.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberKey.as<double>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberRlEntry.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupStart.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupCount.as<int>() : nullptr, ctx->dRlLimitType.as<char>(), ctx->nrl, ctx->hasDqQrGroups ? ctx->dDqMemberQrGroup.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr, ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), dqTraceRow);
    if (ctx->ndr) fillDistrRhoBatch<<<grid(ctx->ndr * nS), BLOCK, 0, ctx->stream>>>(ctx->ndr, nS, n, nnz, ctx->st->dDrCol.as<int>(), ctx->st->dDrCoef.as<double>(), ctx->st->dDrDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nDrOv, ctx->nDrOv ? ctx->dDrOvIdx.as<int>() : nullptr, ctx->nDrOv ? ctx->dDrOvC.as<double>() : nullptr, ctx->st->drQr ? ctx->st->dDrQrGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>());
    if (ctx->ndb) fillDistrShuntBBatch<<<grid(ctx->ndb * nS), BLOCK, 0, ctx->stream>>>(ctx->ndb, nS, n, nnz, ctx->st->dDbCol.as<int>(), ctx->st->dDbCoef.as<double>(), ctx->st->dDbDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), ctx->st->dZiAltDest.as<int>(), ctx->st->dZiSelRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ziDyn ? 1 : 0, ziRelocOf(ctx->st.get()));
    applyRowModeBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->dMode.as<int>(), ctx->st->dRowPtr.as<int>(), ctx->st->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dRowSlope.as<double>());
    hazRepairValues(ctx, nS, ctx->dVals.as<double>());

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
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
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

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDsGenCum(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->ngen <= 0 || !ctx->dGenCum.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nScenarios) * ctx->ngen;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dGenCum.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(cnt));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDsRetain(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dDsRetainApplied.p) {
        return nullptr;
    }
    std::vector<double> host(ctx->nScenarios);
    cudaMemcpy(host.data(), ctx->dDsRetainApplied.p, ctx->nScenarios * sizeof(double), cudaMemcpyDeviceToHost);
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
// on a batch context — see rlCurveLimitsBatch. cvIdx stride 5 {rlIdx, pRow, ptsOff, npts, dsGenSlot}; cvIn
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
    int ncv = env->GetArrayLength(jCvIdx) / 5;   // stride 5: {rlIdx, pRow, ptsOff, npts, dsGenSlot}
    if (ncv == 0) {
        ctx->ncv = 0;
        return;
    }
    jint* cvIdx = env->GetIntArrayElements(jCvIdx, nullptr);
    jdouble* cvIn = env->GetDoubleArrayElements(jCvIn, nullptr);
    jdouble* cvPts = env->GetDoubleArrayElements(jCvPts, nullptr);
    ctx->dCvIdx.upload(cvIdx, static_cast<size_t>(ncv) * 5 * sizeof(int));
    ctx->dCvIn.upload(cvIn, static_cast<size_t>(ncv) * 5 * sizeof(double));
    ctx->dCvPts.upload(cvPts, static_cast<size_t>(env->GetArrayLength(jCvPts)) * sizeof(double));
    ctx->dRlMinQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    ctx->dRlMaxQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    ctx->dRlLoadQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    env->ReleaseIntArrayElements(jCvIdx, cvIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCvIn, cvIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCvPts, cvPts, JNI_ABORT);
    ctx->ncv = ncv;
    ctx->cvExtrapolate = extrapolate;
}

// Remote-member RL entries (plan-B1): install (or clear) the per-entry remote kinds + compare rows
// on a batch context. The RL kernel SKIPS remote entries until the B3 transition machinery lands.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlElection(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jNomV, jintArray jPRow, jdoubleArray jStaticP, jintArray jIdRank) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hasRlElection = 0;
    if (ctx->nrl == 0 || jNomV == nullptr || jPRow == nullptr || jStaticP == nullptr || jIdRank == nullptr) {
        return;
    }
    if (env->GetArrayLength(jNomV) != ctx->nrl || env->GetArrayLength(jPRow) != ctx->nrl
            || env->GetArrayLength(jStaticP) != ctx->nrl || env->GetArrayLength(jIdRank) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRlElection: array length != nrl");
        return;
    }
    jdouble* nv = env->GetDoubleArrayElements(jNomV, nullptr);
    jint* pr = env->GetIntArrayElements(jPRow, nullptr);
    jdouble* sp = env->GetDoubleArrayElements(jStaticP, nullptr);
    jint* ir = env->GetIntArrayElements(jIdRank, nullptr);
    ctx->dRlElectNomV.upload(nv, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlElectPRow.upload(pr, static_cast<size_t>(ctx->nrl) * sizeof(int));
    ctx->dRlElectStaticP.upload(sp, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlElectIdRank.upload(ir, static_cast<size_t>(ctx->nrl) * sizeof(int));
    env->ReleaseDoubleArrayElements(jNomV, nv, JNI_ABORT);
    env->ReleaseIntArrayElements(jPRow, pr, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jStaticP, sp, JNI_ABORT);
    env->ReleaseIntArrayElements(jIdRank, ir, JNI_ABORT);
    ctx->hasRlElection = 1;
}

// voltageRemoteControlRobustMode (OLF's default): install the robust-mode channel on a batch context —
// the per-entry initial generation target Q that a realistic-V freeze pins at, and the two thresholds
// ReactiveLimitsOuterLoop tests the controller's OWN voltage against (minRealisticVoltage ·
// REALISTIC_VOLTAGE_MARGIN and maxRealisticVoltage / REALISTIC_VOLTAGE_MARGIN, margin 1.02). robust = 0
// clears the channel, so a cached context reused by a run with the mode off stops applying it.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlRobust(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jInitTgtQ, jdouble vLo, jdouble vHi, jint robust) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->rlRobust = 0;
    if (ctx->nrl == 0 || robust == 0 || jInitTgtQ == nullptr) {
        return;
    }
    if (env->GetArrayLength(jInitTgtQ) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRlRobust: array length != nrl");
        return;
    }
    jdouble* q0 = env->GetDoubleArrayElements(jInitTgtQ, nullptr);
    ctx->dRlInitTgtQ.upload(q0, static_cast<size_t>(ctx->nrl) * sizeof(double));
    env->ReleaseDoubleArrayElements(jInitTgtQ, q0, JNI_ABORT);
    ctx->rlVLoRobust = vLo;
    ctx->rlVHiRobust = vHi;
    ctx->rlRobust = 1;
}

// AcloadFlowEngine's realistic-voltage check: install (or clear) the per-row "this row is a
// checked bus voltage" mask and the two bounds. enabled = 0 clears it on a cached context.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRealisticCheck(
        JNIEnv* env, jclass, jlong handle, jbyteArray jRowChecked, jdouble minV, jdouble maxV, jint enabled,
        jint fixingLoop) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->realisticCheck = 0;
    if (enabled == 0 || jRowChecked == nullptr) {
        return;
    }
    if (env->GetArrayLength(jRowChecked) != ctx->n) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRealisticCheck: array length != n");
        return;
    }
    jbyte* rc = env->GetByteArrayElements(jRowChecked, nullptr);
    ctx->dRealisticRow.upload(rc, static_cast<size_t>(ctx->n) * sizeof(char));
    env->ReleaseByteArrayElements(jRowChecked, rc, JNI_ABORT);
    ctx->realisticMinV = minV;
    ctx->realisticMaxV = maxV;
    ctx->realisticFixingLoop = fixingLoop;
    ctx->realisticCheck = 1;
}

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
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlBaseLimitType(
        JNIEnv* env, jclass, jlong handle, jbyteArray jKinds) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKinds == nullptr || env->GetArrayLength(jKinds) == 0) {
        ctx->hasRlBaseLt = 0;
        return;
    }
    if (env->GetArrayLength(jKinds) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchRlBaseLimitType: array length != nrl");
        return;
    }
    jbyte* kinds = env->GetByteArrayElements(jKinds, nullptr);
    ctx->dRlBaseLimitType.alloc(static_cast<size_t>(ctx->nrl));
    CUDA_CHECK(cudaMemcpy(ctx->dRlBaseLimitType.p, kinds, static_cast<size_t>(ctx->nrl), cudaMemcpyHostToDevice));
    env->ReleaseByteArrayElements(jKinds, kinds, JNI_ABORT);
    ctx->hasRlBaseLt = 1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlRemote(
        JNIEnv* env, jclass, jlong handle, jbyteArray jRemote, jintArray jCmpVrow,
        jintArray jRmqRow, jintArray jRmqElem, jintArray jRmqSide, jintArray jRmqKind, jdoubleArray jRmqWeight,
        jintArray jDqKMember, jintArray jDqJMember, jdoubleArray jMKey, jintArray jMRlEntry,
        jintArray jMGroupStart, jintArray jMGroupCount, jintArray jMRow, jbyteArray jMHasRow,
        jintArray jMQrGroup, jdoubleArray jMTargetQ) {
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
    int selfCount = 0;
    for (int i = 0; i < nr; i++) {
        if (rem[i]) {
            count++;
        }
        if (rem[i] == 2) {
            selfCount++;
        }
    }
    ctx->nRlPilotSelf = selfCount;
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
    // Keyed on the MEMBER list, not on the DISTR_Q rows: a SINGLETON remote group has members (and a
    // pilot-Q relocation group, so its pilot's control-loss flip must be armed) but no DISTR_Q row at
    // all — gating on dqKMember left hasDqQrGroups at 0 for it, and the base load flow on the driver
    // kept the pilot's V equation after pinning its only member (2.4e-3 pu on the pilot's Q).
    if (jMKey != nullptr && env->GetArrayLength(jMKey) > 0) {
        int ndqc = jDqKMember != nullptr ? env->GetArrayLength(jDqKMember) : 0;
        int nm = env->GetArrayLength(jMKey);
        jint* km = ndqc > 0 ? env->GetIntArrayElements(jDqKMember, nullptr) : nullptr;
        jint* jm = ndqc > 0 ? env->GetIntArrayElements(jDqJMember, nullptr) : nullptr;
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
        if (jMTargetQ == nullptr || env->GetArrayLength(jMTargetQ) != nm) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchRlRemote: dqMemberTargetQ missing or length != members");
            return;
        }
        jdouble* mtq = env->GetDoubleArrayElements(jMTargetQ, nullptr);
        ctx->dDqMemberTargetQ.upload(mtq, static_cast<size_t>(nm) * sizeof(double));
        env->ReleaseDoubleArrayElements(jMTargetQ, mtq, JNI_ABORT);
        ctx->nDqMembers = nm;
        if (ndqc > 0) {
            ctx->dDqKMember.upload(km, static_cast<size_t>(ndqc) * sizeof(int));
            ctx->dDqJMember.upload(jm, static_cast<size_t>(ndqc) * sizeof(int));
        }
        ctx->dDqMemberKey.upload(mk, static_cast<size_t>(nm) * sizeof(double));
        ctx->dDqMemberRlEntry.upload(mre, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberGroupStart.upload(mgs, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberGroupCount.upload(mgc, static_cast<size_t>(nm) * sizeof(int));
        if (ndqc > 0) {
            env->ReleaseIntArrayElements(jDqKMember, km, JNI_ABORT);
            env->ReleaseIntArrayElements(jDqJMember, jm, JNI_ABORT);
        }
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
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqInactiveMembers(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotMember) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotMember == nullptr || env->GetArrayLength(jSlotMember) == 0) {
        ctx->nDqInact = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotMember);
    jint* sm = env->GetIntArrayElements(jSlotMember, nullptr);
    ctx->dDqInact.upload(sm, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotMember, sm, JNI_ABORT);
    ctx->nDqInact = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchFrozenRows(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotRow) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotRow == nullptr || env->GetArrayLength(jSlotRow) == 0) {
        ctx->nFrozenRow = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotRow);
    jint* sr = env->GetIntArrayElements(jSlotRow, nullptr);
    ctx->dFrozenRow.upload(sr, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotRow, sr, JNI_ABORT);
    ctx->nFrozenRow = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsLostGenerators(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotGen) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotGen == nullptr || env->GetArrayLength(jSlotGen) == 0) {
        ctx->nDsLostGen = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotGen);
    jint* sg = env->GetIntArrayElements(jSlotGen, nullptr);
    ctx->dDsLostGen.upload(sg, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotGen, sg, JNI_ABORT);
    ctx->nDsLostGen = n2 / 2;
}

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

// Per-GENERATOR distributed-slack participants: the factors and limits OLF's water-fill actually
// saturates on, grouped by bus (a bus's row value is the sum over its contiguous slice).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsGenerators(
        JNIEnv* env, jclass, jlong handle, jintArray jBus, jdoubleArray jFactor,
        jdoubleArray jMaxDelta, jdoubleArray jMinDelta, jdoubleArray jMarginUp, jdoubleArray jMarginDown,
        jintArray jBusGenStart, jintArray jBusGenEnd) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jBus == nullptr || env->GetArrayLength(jBus) == 0) {
        ctx->ngen = 0;
        return;
    }
    int ngen = env->GetArrayLength(jBus);
    auto upI = [&](jintArray a, DeviceBuf& buf) {
        jint* p = env->GetIntArrayElements(a, nullptr);
        buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, p, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& buf) {
        if (a == nullptr || env->GetArrayLength(a) == 0) {
            return;
        }
        jdouble* p = env->GetDoubleArrayElements(a, nullptr);
        buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, p, JNI_ABORT);
    };
    upI(jBus, ctx->dgBus);
    upD(jFactor, ctx->dgFactor);
    upD(jMaxDelta, ctx->dgMaxDelta);
    upD(jMinDelta, ctx->dgMinDelta);
    upD(jMarginUp, ctx->dgMarginUp);
    upD(jMarginDown, ctx->dgMarginDown);
    upI(jBusGenStart, ctx->dBusGenStart);
    upI(jBusGenEnd, ctx->dBusGenEnd);
    ctx->dGenCum.alloc((size_t) ctx->nScenarios * ngen * sizeof(double));
    ctx->dGenCumValid.alloc((size_t) ctx->nScenarios * sizeof(char));
    ctx->dGenFactorWork.alloc((size_t) ctx->nScenarios * ngen * sizeof(double));
    ctx->dGenRemoved.alloc((size_t) ctx->nScenarios * ngen * sizeof(char));
    ctx->ngen = ngen;
}

// The HVDC AC-emulation outer loops' tables + their per-scenario starting state (the statuses and
// applied converter-row deltas the host's pre-solve freeze produced). Install to run those loops as
// DRIVER PHASES in OLF's order; clear (null) to leave them to the host orchestrator.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchHvdcLoops(
        JNIEnv* env, jclass, jlong handle, jintArray jElem, jintArray jBus1Row, jintArray jBus2Row,
        jdoubleArray jIn, jbyteArray jCtgOut, jintArray jStatus,
        jdoubleArray jAppliedD1, jdoubleArray jAppliedD2, jint limitsActive) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jElem == nullptr || env->GetArrayLength(jElem) == 0) {
        ctx->nhv = 0;
        return;
    }
    int nh = env->GetArrayLength(jElem);
    auto upI = [&](jintArray a, DeviceBuf& b) {
        jint* q = env->GetIntArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, q, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& b) {
        jdouble* q = env->GetDoubleArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, q, JNI_ABORT);
    };
    upI(jElem, ctx->dHvElem);
    upI(jBus1Row, ctx->dHvBus1Row);
    upI(jBus2Row, ctx->dHvBus2Row);
    upD(jIn, ctx->dHvIn);
    upI(jStatus, ctx->dHvStatus);
    upD(jAppliedD1, ctx->dHvAppliedD1);
    upD(jAppliedD2, ctx->dHvAppliedD2);
    jbyte* co = env->GetByteArrayElements(jCtgOut, nullptr);
    ctx->dHvCtgOut.upload(co, (size_t) env->GetArrayLength(jCtgOut));
    env->ReleaseByteArrayElements(jCtgOut, co, JNI_ABORT);
    ctx->nhv = nh;
    ctx->hvLimitsActive = limitsActive;
}

/** Enable HOST-owned outer-loop slots via a bitmask.
 *  Bit 0 = SecondaryVoltageControl (OLF order 4, pre-RL),
 *  Bit 1 = PhaseControl (OLF order 7, post-RL),
 *  Bit 2 = TransformerVoltageControl (OLF order 8),
 *  Bit 3 = TransformerReactivePowerControl (OLF order 9),
 *  Bit 4 = ShuntVoltageControl (OLF order 10).
 *  0 disables all host loops; the driver behaves exactly as if the mechanism did not exist. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchHostLoop(
        JNIEnv*, jclass, jlong handle, jint mask) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->hostLoopMask = mask;
    }
}

/** Upload the incremental TVC controller data for the in-kernel outer loop.
 *  rhoRow/vRow/elemIndex: [nCtrl] int; targetV/halfDb: [nCtrl] double;
 *  tapPtr: [nCtrl+1] int (CSR into tapData); tapData: [totalTaps*8] double;
 *  baseTap: [nCtrl] int; groupPtr: [nGroups+1] int (CSR into groupCtrl);
 *  groupCtrl: [nCtrl] int (controller indices per group); maxTapShift: OLF's per-pass bound.
 *  Returns 1 when the kernel loop OWNS the controllers (shared controlled buses included — the
 *  one-tap-at-a-time round robin runs in tvcGroupDecisionBatch), 0 when there are none. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchTvcControllers(
        JNIEnv* env, jclass, jlong handle,
        jintArray jRhoRow, jintArray jVRow, jdoubleArray jTargetV, jdoubleArray jHalfDb,
        jintArray jElemIndex, jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap,
        jintArray jGroupPtr, jintArray jGroupCtrl, jint maxTapShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nTvc = 0;
    if (jRhoRow == nullptr || env->GetArrayLength(jRhoRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> groupPtr;
    readI(jGroupPtr, groupPtr);                              // shared controlled buses run in-kernel too
    std::vector<int> vRow, groupCtrl;
    std::vector<double> targetV, halfDb;
    readI(jRhoRow, ctx->hTvcRhoRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfDb, halfDb);
    readI(jElemIndex, ctx->hTvcElemIndex);
    readI(jTapPtr, ctx->hTvcTapPtr);
    readD(jTapData, ctx->hTvcTapData);
    readI(jBaseTap, ctx->hTvcBaseTap);
    readI(jGroupCtrl, groupCtrl);
    int nc = static_cast<int>(ctx->hTvcRhoRow.size());
    ctx->dTvcRhoRow.upload(ctx->hTvcRhoRow.data(), nc * sizeof(int));
    ctx->dTvcVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->dTvcTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->dTvcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dTvcElemIndex.upload(ctx->hTvcElemIndex.data(), nc * sizeof(int));
    ctx->dTvcTapPtr.upload(ctx->hTvcTapPtr.data(), ctx->hTvcTapPtr.size() * sizeof(int));
    ctx->dTvcTapData.upload(ctx->hTvcTapData.data(), ctx->hTvcTapData.size() * sizeof(double));
    ctx->dTvcBaseTap.upload(ctx->hTvcBaseTap.data(), nc * sizeof(int));
    ctx->nTvcGroups = static_cast<int>(groupPtr.size()) - 1;
    ctx->dTvcGroupPtr.upload(groupPtr.data(), groupPtr.size() * sizeof(int));
    ctx->dTvcGroupCtrl.upload(groupCtrl.data(), groupCtrl.size() * sizeof(int));
    ctx->tvcMaxTapShift = maxTapShift > 0 ? maxTapShift : 1;
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dTvcCurTap.ensure(nst * sizeof(int));
    ctx->dTvcPrevTap.ensure(nst * sizeof(int));
    ctx->dTvcRatchetDir.ensure(nst * sizeof(int));
    ctx->dTvcRatchetCount.ensure(nst * sizeof(int));
    ctx->dTvcRatchetAllowed.ensure(nst * sizeof(int));
    ctx->nTvc = nc;
    return 1;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchMonitoringTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nmon == 0 || !ctx->dMonActivated.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nmon) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dMonActivated.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(cnt));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

// The zone-regulated generator voltage control TARGETS the SVC phase settled on, per
// (scenario, controller) — what OLF's SecondaryVoltageControlOuterLoop leaves on the network. The
// batched SA restores per scenario and needs none of this; the base load flow has to give them back.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchSecondaryVcTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvc <= 0 || !ctx->dSvcTargetV.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvc) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dSvcTargetV.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(cnt));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchTvcTaps(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvc <= 0 || !ctx->dTvcCurTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nTvc) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dTvcCurTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Upload the incremental SHUNT voltage-control tables for the in-kernel loop. Per controller:
 *  bRow (SHUNT_B identity row), vRow (controlled bus), targetV, halfDb, elemAbs (global element, -1),
 *  shOvElem (shunt-pack element for the static (dG, dB) override, -1); groups: groupPtr[nGroups+1] ->
 *  groupCtrl (controllers, |b|-descending); subs: subPtr[nCtrl+1] -> per sub secPtr[nSubs+1] into
 *  secB/secG, subMin, subBase. Returns 1 when the kernel loop owns them, 0 when there are none. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchShuntControllers(
        JNIEnv* env, jclass, jlong handle,
        jintArray jBRow, jintArray jVRow, jdoubleArray jTargetV, jdoubleArray jHalfDb, jintArray jElemAbs, jintArray jShOvElem,
        jintArray jGroupPtr, jintArray jGroupCtrl, jintArray jSubPtr, jintArray jSecPtr, jdoubleArray jSecB, jdoubleArray jSecG,
        jintArray jSubMin, jintArray jSubBase, jint maxSectionShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nShc = 0;
    if (jBRow == nullptr || env->GetArrayLength(jBRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> vRow, elemAbs, groupPtr, groupCtrl, subMin;
    std::vector<double> targetV, halfDb;
    readI(jBRow, ctx->hShcBRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfDb, halfDb);
    readI(jElemAbs, elemAbs);
    readI(jShOvElem, ctx->hShcShOvElem);
    readI(jGroupPtr, groupPtr);
    readI(jGroupCtrl, groupCtrl);
    readI(jSubPtr, ctx->hShcSubPtr);
    readI(jSecPtr, ctx->hShcSecPtr);
    readD(jSecB, ctx->hShcSecB);
    readD(jSecG, ctx->hShcSecG);
    readI(jSubMin, subMin);
    readI(jSubBase, ctx->hShcSubBase);
    int nc = static_cast<int>(ctx->hShcBRow.size());
    int nSubs = static_cast<int>(ctx->hShcSubBase.size());
    ctx->dShcBRow.upload(ctx->hShcBRow.data(), nc * sizeof(int));
    ctx->dShcVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->dShcTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->dShcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dShcElemAbs.upload(elemAbs.data(), nc * sizeof(int));
    ctx->dShcShOvElem.upload(ctx->hShcShOvElem.data(), nc * sizeof(int));
    ctx->dShcGroupPtr.upload(groupPtr.data(), groupPtr.size() * sizeof(int));
    ctx->dShcGroupCtrl.upload(groupCtrl.data(), groupCtrl.size() * sizeof(int));
    ctx->dShcSubPtr.upload(ctx->hShcSubPtr.data(), ctx->hShcSubPtr.size() * sizeof(int));
    ctx->dShcSecPtr.upload(ctx->hShcSecPtr.data(), ctx->hShcSecPtr.size() * sizeof(int));
    ctx->dShcSecB.upload(ctx->hShcSecB.data(), ctx->hShcSecB.size() * sizeof(double));
    ctx->dShcSecG.upload(ctx->hShcSecG.data(), ctx->hShcSecG.size() * sizeof(double));
    ctx->dShcSubMin.upload(subMin.data(), nSubs * sizeof(int));
    ctx->dShcSubBase.upload(ctx->hShcSubBase.data(), nSubs * sizeof(int));
    ctx->nShcGroups = static_cast<int>(groupPtr.size()) - 1;
    ctx->nShcSubs = nSubs;
    ctx->shcMaxSectionShift = maxSectionShift >= 0 ? maxSectionShift : 3;
    size_t nsu = static_cast<size_t>(ctx->nScenarios) * nSubs;
    size_t nsc = static_cast<size_t>(ctx->nScenarios) * nc;
    size_t nsg = static_cast<size_t>(ctx->nScenarios) * ctx->nShcGroups;
    ctx->dShcPos.ensure(nsu * sizeof(int));
    ctx->dShcRatDir.ensure(nsu * sizeof(int));
    ctx->dShcRatCount.ensure(nsu * sizeof(int));
    ctx->dShcRatAllowed.ensure(nsu * sizeof(int));
    ctx->dShcShift.ensure(nsu * sizeof(int));
    ctx->dShcSensi.ensure(nsc * sizeof(double));
    ctx->dShcBaseB.ensure(nsc * sizeof(double));
    ctx->dShcAppliedDb.ensure(nsc * sizeof(double));
    ctx->dShcOutOfDb.ensure(nsg * sizeof(char));
    ctx->dShcDiffV.ensure(nsg * sizeof(double));
    ctx->nShc = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchShuntPositions(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nShc <= 0 || !ctx->dShcPos.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nShcSubs) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dShcPos.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Upload the FLOW-OBSERVED incremental controllers (RPC / incremental phase). Per controller: kind
 *  (0 RPC, 1 phase CONTROLLER, 2 phase LIMITER), pinRow (the rho/a1 identity row), pinCol (6 r1 / 7 a1),
 *  side (1/2) of the controlled flow, targetValue, halfDb, elem (controller's global element),
 *  ctlElem (controlled branch's global element — a closed branch), tapPtr/tapData ([y g1 g2 b1 b2 ksi r1
 *  a1] per tap), baseTap, maxShift. Returns 1 when the kernel loops own them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchFlowControllers(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jPinRow, jintArray jPinCol, jintArray jSide, jdoubleArray jTargetValue, jdoubleArray jHalfDb,
        jintArray jElem, jintArray jCtlElem, jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap, jintArray jMaxShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nFlc = 0;
    ctx->nFlcPhase = 0;
    ctx->nFlcRpc = 0;
    if (jKind == nullptr || env->GetArrayLength(jKind) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> pinCol, side, ctlElem, maxShift;
    std::vector<double> targetValue, halfDb;
    readI(jKind, ctx->hFlcKind);
    readI(jPinRow, ctx->hFlcPinRow);
    readI(jPinCol, pinCol);
    readI(jSide, side);
    readD(jTargetValue, targetValue);
    readD(jHalfDb, halfDb);
    readI(jElem, ctx->hFlcElem);
    readI(jCtlElem, ctlElem);
    readI(jTapPtr, ctx->hFlcTapPtr);
    readD(jTapData, ctx->hFlcTapData);
    readI(jBaseTap, ctx->hFlcBaseTap);
    readI(jMaxShift, maxShift);
    int nc = static_cast<int>(ctx->hFlcKind.size());
    std::vector<int> ctlPack(nc);
    for (int c = 0; c < nc; c++) {
        ctlPack[c] = ctlElem[c] - ctx->cbOffset;             // the controlled branch's index in the closed-branch pack
        if (ctlPack[c] < 0 || ctlPack[c] >= ctx->st->cb.ne) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
        if (ctx->hFlcKind[c] == 0) ctx->nFlcRpc++; else ctx->nFlcPhase++;
    }
    ctx->dFlcKind.upload(ctx->hFlcKind.data(), nc * sizeof(int));
    ctx->dFlcPinRow.upload(ctx->hFlcPinRow.data(), nc * sizeof(int));
    ctx->dFlcPinCol.upload(pinCol.data(), nc * sizeof(int));
    ctx->dFlcSide.upload(side.data(), nc * sizeof(int));
    ctx->dFlcTargetValue.upload(targetValue.data(), nc * sizeof(double));
    ctx->dFlcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dFlcElem.upload(ctx->hFlcElem.data(), nc * sizeof(int));
    ctx->dFlcCtlPack.upload(ctlPack.data(), nc * sizeof(int));
    ctx->dFlcTapPtr.upload(ctx->hFlcTapPtr.data(), ctx->hFlcTapPtr.size() * sizeof(int));
    ctx->dFlcTapData.upload(ctx->hFlcTapData.data(), ctx->hFlcTapData.size() * sizeof(double));
    ctx->dFlcBaseTap.upload(ctx->hFlcBaseTap.data(), nc * sizeof(int));
    ctx->dFlcMaxShift.upload(maxShift.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dFlcCurTap.ensure(nst * sizeof(int));
    ctx->dFlcPrevTap.ensure(nst * sizeof(int));
    ctx->dFlcRatDir.ensure(nst * sizeof(int));
    ctx->dFlcRatCount.ensure(nst * sizeof(int));
    ctx->dFlcRatAllowed.ensure(nst * sizeof(int));
    ctx->dFlcSensi.ensure(nst * sizeof(double));
    ctx->dFlcDiff.ensure(nst * sizeof(double));
    ctx->dFlcOutOfDb.ensure(nst * sizeof(char));
    ctx->nFlc = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchFlowTaps(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nFlc <= 0 || !ctx->dFlcCurTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nFlc) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dFlcCurTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Upload the CONTINUOUS phase controllers (kind 0 CONTROLLER / 1 LIMITER, a1 identity row, controlled
 *  side, current limit, global element, tap rows, base tap). Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchContinuousPhase(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jA1Row, jintArray jSide, jdoubleArray jLimit, jintArray jElem,
        jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nCph = 0;
    if (jKind == nullptr || env->GetArrayLength(jKind) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> kind, a1Row, side;
    std::vector<double> limit;
    readI(jKind, kind);
    readI(jA1Row, a1Row);
    readI(jSide, side);
    readD(jLimit, limit);
    readI(jElem, ctx->hCphElem);
    readI(jTapPtr, ctx->hCphTapPtr);
    readD(jTapData, ctx->hCphTapData);
    readI(jBaseTap, ctx->hCphBaseTap);
    int nc = static_cast<int>(kind.size());
    for (int c = 0; c < nc; c++) {
        int e = ctx->hCphElem[c] - ctx->cbOffset;
        if (e < 0 || e >= ctx->st->cb.ne) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
    }
    ctx->dCphKind.upload(kind.data(), nc * sizeof(int));
    ctx->dCphA1Row.upload(a1Row.data(), nc * sizeof(int));
    ctx->dCphSide.upload(side.data(), nc * sizeof(int));
    ctx->dCphLimit.upload(limit.data(), nc * sizeof(double));
    ctx->dCphElem.upload(ctx->hCphElem.data(), nc * sizeof(int));
    ctx->dCphTapPtr.upload(ctx->hCphTapPtr.data(), ctx->hCphTapPtr.size() * sizeof(int));
    ctx->dCphTapData.upload(ctx->hCphTapData.data(), ctx->hCphTapData.size() * sizeof(double));
    ctx->dCphBaseTap.upload(ctx->hCphBaseTap.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dCphTap.ensure(nst * sizeof(int));
    ctx->dCphPrevTap.ensure(nst * sizeof(int));
    ctx->dCphIter.ensure(static_cast<size_t>(ctx->nScenarios) * sizeof(int));
    ctx->nCph = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchContinuousPhaseTaps(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nCph <= 0 || !ctx->dCphTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nCph) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dCphTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Install the SecondaryVoltageControl zones for the in-kernel phase (OLF order 4). Controllers are
 *  addressed by their controlled bus's V row (the sensitivity's unit row and where a moved target lands)
 *  and their own bus V row (the reactive injection the K comes from); each carries a CSR slice of Q
 *  observables (kind 0/1 = a closed branch's side-1/side-2 reactive flow, 2 = a shunt) and the zones a
 *  CSR slice of controllers. Returns 1 when the kernel loop owns them — it refuses a zone set wider than
 *  the in-kernel dense system, or an observable on an element the batch does not pack. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchSecondaryVc(
        JNIEnv* env, jclass, jlong handle,
        jintArray jCtldVRow, jintArray jBusVRow, jintArray jRlIdx, jdoubleArray jBaseTargetV,
        jdoubleArray jMinQ, jdoubleArray jMaxQ, jdoubleArray jLoadQ,
        jintArray jQPtr, jintArray jQKind, jintArray jQElem, jintArray jQBRow,
        jintArray jZonePtr, jintArray jZoneCtl, jintArray jZonePilotRow, jdoubleArray jZoneTargetV,
        jdouble minPlausibleV, jdouble maxPlausibleV) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nSvc = 0;
    ctx->nSvcZones = 0;
    if (jCtldVRow == nullptr || env->GetArrayLength(jCtldVRow) == 0 || ctx->nrl <= 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> busVRow, rlIdx, qPtr, qKind, qElem, qBRow, zonePtr, zoneCtl, zonePilotRow;
    std::vector<double> minQ, maxQ, loadQ, zoneTargetV;
    readI(jCtldVRow, ctx->hSvcCtldVRow);
    readI(jBusVRow, busVRow);
    readI(jRlIdx, rlIdx);
    readD(jBaseTargetV, ctx->hSvcBaseTargetV);
    readD(jMinQ, minQ);
    readD(jMaxQ, maxQ);
    readD(jLoadQ, loadQ);
    readI(jQPtr, qPtr);
    readI(jQKind, qKind);
    readI(jQElem, qElem);
    readI(jQBRow, qBRow);
    readI(jZonePtr, zonePtr);
    readI(jZoneCtl, zoneCtl);
    readI(jZonePilotRow, zonePilotRow);
    readD(jZoneTargetV, zoneTargetV);
    int nC = static_cast<int>(ctx->hSvcCtldVRow.size());
    int nZ = static_cast<int>(zonePtr.size()) - 1;
    if (nZ <= 0) {
        return 0;
    }
    // OLF solves ONE system over every enabled controller of every zone, so the size is the total
    // controller count. The batched LU holds an N*N matrix per scenario, which is what actually bounds
    // this: refuse when that workspace would not fit rather than at an arbitrary controller count.
    double matBytes = static_cast<double>(ctx->nScenarios) * nC * nC * sizeof(double);
    if (matBytes > SVC_MAT_BUDGET_BYTES) {
        std::fprintf(stderr, "SVC in-kernel declined: %d controllers x %d scenarios needs %.1f MB of zone-system "
                     "workspace (budget %.0f MB)\n", nC, ctx->nScenarios, matBytes / 1048576.0,
                     SVC_MAT_BUDGET_BYTES / 1048576.0);
        return 0;
    }
    for (int e : qElem) {
        if (e < 0) {
            return 0;                                        // an observable the batch does not pack
        }
    }
    ctx->dSvcCtldVRow.upload(ctx->hSvcCtldVRow.data(), nC * sizeof(int));
    ctx->dSvcBusVRow.upload(busVRow.data(), nC * sizeof(int));
    ctx->dSvcRlIdx.upload(rlIdx.data(), nC * sizeof(int));
    ctx->dSvcBaseTargetV.upload(ctx->hSvcBaseTargetV.data(), nC * sizeof(double));
    ctx->dSvcMinQ.upload(minQ.data(), nC * sizeof(double));
    ctx->dSvcMaxQ.upload(maxQ.data(), nC * sizeof(double));
    ctx->dSvcLoadQ.upload(loadQ.data(), nC * sizeof(double));
    ctx->dSvcQPtr.upload(qPtr.data(), qPtr.size() * sizeof(int));
    ctx->dSvcQKind.upload(qKind.data(), std::max<size_t>(qKind.size(), 1) * sizeof(int));
    ctx->dSvcQElem.upload(qElem.data(), std::max<size_t>(qElem.size(), 1) * sizeof(int));
    ctx->dSvcQBRow.upload(qBRow.data(), std::max<size_t>(qBRow.size(), 1) * sizeof(int));
    ctx->dSvcZonePtr.upload(zonePtr.data(), zonePtr.size() * sizeof(int));
    ctx->dSvcZoneCtl.upload(zoneCtl.data(), zoneCtl.size() * sizeof(int));
    ctx->dSvcZonePilotRow.upload(zonePilotRow.data(), nZ * sizeof(int));
    ctx->dSvcZoneTargetV.upload(zoneTargetV.data(), nZ * sizeof(double));
    size_t nS = ctx->nScenarios;
    ctx->dSvcTargetV.ensure(nS * nC * sizeof(double));
    ctx->dSvcK0.ensure(nS * nC * sizeof(double));
    ctx->dSvcJk.ensure(nS * nC * nC * sizeof(double));
    ctx->dSvcJvpp.ensure(nS * nC * sizeof(double));
    ctx->dSvcEnabled.ensure(nS * nC * sizeof(char));
    ctx->dSvcNeed.ensure(nS * sizeof(char));
    ctx->dSvcReject.ensure(nS * sizeof(char));
    ctx->dSvcRlVt.ensure(nS * std::max(ctx->nrl, 1) * sizeof(double));
    ctx->dSvcMat.ensure(nS * static_cast<size_t>(nC) * nC * sizeof(double));
    ctx->dSvcRhs.ensure(nS * nC * sizeof(double));
    ctx->dSvcMatPtr.ensure(nS * sizeof(double*));
    ctx->dSvcRhsPtr.ensure(nS * sizeof(double*));
    ctx->dSvcPiv.ensure(nS * nC * sizeof(int));
    ctx->dSvcInfo.ensure(nS * sizeof(int));
    {
        std::vector<int> zoneOf(nC, 0);                      // controller -> its zone, for the assembly
        for (int z = 0; z < nZ; z++) {
            for (int k = zonePtr[z]; k < zonePtr[z + 1]; k++) {
                zoneOf[zoneCtl[k]] = z;
            }
        }
        ctx->dSvcCtlZone.upload(zoneOf.data(), zoneOf.size() * sizeof(int));
    }
    {
        // NaN = "this zone has not moved the controller's target", which is what rlTargetOf tests. The
        // reactive-limit SEEDING reads this buffer before the per-solve reset further down runs, so an
        // uninitialised one hands the switch-back a garbage comparison voltage and un-pins controllers
        // that should stay pinned (measured on ieee14's z1: pins 4 -> 2 instead of 4 -> 6, and 1.0e-10 kV).
        std::vector<double> nanv(nS * std::max(ctx->nrl, 1), std::nan(""));
        CUDA_CHECK(cudaMemcpy(ctx->dSvcRlVt.p, nanv.data(), nanv.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    ctx->svcMinPlausibleV = minPlausibleV;
    ctx->svcMaxPlausibleV = maxPlausibleV;
    ctx->nSvcZones = nZ;
    ctx->nSvc = nC;
    return 1;
}

/** Un-pin reactive-limit entries at a driver pause (SecondaryVoltageControl's re-enable step):
 *  {@code pairs} is (scenario, RL entry) flattened, {@code newTarget} the voltage target each row goes
 *  back to. No-op for an entry that is not pinned. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlUnpin(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jdoubleArray jTargets) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nrl <= 0 || jPairs == nullptr || jTargets == nullptr) {
        return;
    }
    int nPairs = env->GetArrayLength(jTargets);
    if (nPairs == 0 || env->GetArrayLength(jPairs) != 2 * nPairs) {
        return;
    }
    std::vector<int> pairs(2 * nPairs);
    std::vector<double> targets(nPairs);
    env->GetIntArrayRegion(jPairs, 0, 2 * nPairs, pairs.data());
    env->GetDoubleArrayRegion(jTargets, 0, nPairs, targets.data());
    DeviceBuf dPairs;
    DeviceBuf dTargets;
    dPairs.upload(pairs.data(), pairs.size() * sizeof(int));
    dTargets.upload(targets.data(), targets.size() * sizeof(double));
    rlUnpinBatch<<<grid(nPairs), BLOCK, 0, ctx->stream>>>(nPairs, ctx->nScenarios, ctx->n, ctx->nrl,
            dPairs.as<int>(), dTargets.as<double>(), ctx->dRlVrow.as<int>(), ctx->dRlLimitType.as<char>(),
            ctx->dRlSavedMode.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

/** Did the last solve stop at a host-owned slot? Returns 0 if not paused, or the
 *  DeviceLoopKind value (LOOP_HOST_SVC=5, LOOP_HOST_PHASE=6, ...) of the paused slot. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDriverPaused(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    return (ctx == nullptr || !ctx->driverPaused) ? 0 : ctx->pausedLoopKind;
}

/** Resume the paused driver: {@code hostChanged[s] != 0} is the host loop's UNSTABLE verdict for
 *  that scenario, which drives the next pass exactly as a device loop's verdict would. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDriverResume(
        JNIEnv* env, jclass, jlong handle, jbyteArray jHostChanged) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hostChanged.assign(ctx->nScenarios, 0);
    if (jHostChanged != nullptr) {
        jsize len = std::min<jsize>(env->GetArrayLength(jHostChanged), ctx->nScenarios);
        env->GetByteArrayRegion(jHostChanged, 0, len, reinterpret_cast<jbyte*>(ctx->hostChanged.data()));
    }
    ctx->driverResume = 1;
}

/** Drop a paused driver sequence without resuming it: the host loop changed the equation layout or
 *  the row modes, so the next solveBatchContext must START OVER on the re-extracted system. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_cancelBatchDriverPause(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->driverPaused = 0;
    ctx->driverResume = 0;
    ctx->pausedLoopKind = 0;
}

/** The next FRESH solve resumes its sweep at this loop kind after its first solve (a restarted
 *  sequence: OLF re-checks the loop that went UNSTABLE before restarting the sweep). The context may
 *  have been rebuilt since the pause, so the Java side carries the kind across. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDriverStartLoop(
        JNIEnv*, jclass, jlong handle, jint kind) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->startLoopKind = kind;
}

/** Write absolute target values at (scenario, row) positions — how a host-owned loop applies its
 *  decision without disturbing the device-side targets the sequence has already accumulated. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchTargetDeltas(
        JNIEnv* env, jclass, jlong handle, jintArray jRows, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || jRows == nullptr) {
        return;
    }
    jsize cnt = env->GetArrayLength(jRows);
    if (cnt == 0) {
        return;
    }
    std::vector<int> rows(cnt);
    std::vector<double> vals(cnt);
    env->GetIntArrayRegion(jRows, 0, cnt, rows.data());
    env->GetDoubleArrayRegion(jVals, 0, cnt, vals.data());
    DeviceBuf dR;
    DeviceBuf dV;
    dR.upload(rows.data(), (size_t) cnt * sizeof(int));
    dV.upload(vals.data(), (size_t) cnt * sizeof(double));
    applyTargetOverrideBatch<<<grid(cnt), BLOCK, 0, ctx->stream>>>(cnt, dR.as<int>(), dV.as<double>(),
                                                                  ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

/** MonitoringVoltage controller table (OLF order 6). Empty array clears it. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchMonitoringLoops(
        JNIEnv* env, jclass, jlong handle, jintArray jVRow, jdoubleArray jBand,
        jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaMode, jdoubleArray jDeltaTarget) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jVRow == nullptr || env->GetArrayLength(jVRow) == 0) {
        ctx->nmon = 0;
        return;
    }
    auto upI = [&](jintArray a, DeviceBuf& b) {
        jint* q = env->GetIntArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, q, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& b) {
        jdouble* q = env->GetDoubleArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, q, JNI_ABORT);
    };
    upI(jVRow, ctx->dMonVRow);
    upD(jBand, ctx->dMonBand);
    upI(jDeltaPtr, ctx->dMonDeltaPtr);
    upI(jDeltaRow, ctx->dMonDeltaRow);
    upI(jDeltaMode, ctx->dMonDeltaMode);
    upD(jDeltaTarget, ctx->dMonDeltaTarget);
    ctx->nmon = env->GetArrayLength(jVRow);
    ctx->dMonActivated.alloc(static_cast<size_t>(ctx->nmon) * ctx->nScenarios * sizeof(double));
}

/** How many (controller, scenario) monitoring automatons FIRED in the last solve. The host keeps
 *  the ACTIVATIONS counter meaningful now that the loop is a driver phase rather than an
 *  orchestrator — it is the coverage guard that a test case really exercises an activation. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchMonitoringActivations(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nmon == 0 || !ctx->dMonActivated.p) {
        return 0;
    }
    size_t cnt = static_cast<size_t>(ctx->nmon) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dMonActivated.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jint fired = 0;
    for (double v : host) {
        if (!std::isnan(v)) {
            fired++;
        }
    }
    return fired;
}

/** Read the HVDC statuses back after a solve, so the host keeps its own view in step. */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchHvdcStatus(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nhv == 0) {
        return nullptr;
    }
    size_t len = (size_t) ctx->nScenarios * ctx->nhv;
    std::vector<int> host(len);
    if (cudaMemcpy(host.data(), ctx->dHvStatus.p, len * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return nullptr;
    }
    jintArray out = env->NewIntArray((jsize) len);
    env->SetIntArrayRegion(out, 0, (jsize) len, host.data());
    return out;
}

// Reactive-limits controllers STRANDED in another synchronous component: install (or clear, with
// null) the per-scenario mask as {slot, controller} pairs.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlIslanded(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlIslanded = 0;
        return;
    }
    size_t maskLen = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<char> mask(maskLen, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            mask[(size_t) s * ctx->nrl + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dRlIslanded.upload(mask.data(), maskLen);
    ctx->nRlIslanded = nPairs;
}

// Per-scenario member targetQ overrides for the DISTR_Q target kernel: {key, value} where key is
// slot * nDqMembers + memberIndex. Set when an outaged bus is a packed group member whose own targetQ
// moved (a reactive LOAD loss). Nulls clear.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqMemberTargetQOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKeys == nullptr || env->GetArrayLength(jKeys) == 0) {
        ctx->nDqTqOv = 0;
        return;
    }
    int ne = env->GetArrayLength(jKeys);
    jint* keys = env->GetIntArrayElements(jKeys, nullptr);
    jdouble* vals = env->GetDoubleArrayElements(jVals, nullptr);
    ctx->dDqTqOvKey.upload(keys, static_cast<size_t>(ne) * sizeof(int));
    ctx->dDqTqOvVal.upload(vals, static_cast<size_t>(ne) * sizeof(double));
    ctx->nDqTqOv = ne;
    env->ReleaseIntArrayElements(jKeys, keys, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVals, vals, JNI_ABORT);
}

// Pilot-row flip entries (see pilotRowAllPinnedBatch): one per (scenario, pilot) where the outage
// made the pilot stop being one of its own controllers. slots/rows/qTargets are per entry; rlCounts
// says how many surviving-controller RL entries each one owns and rlPool concatenates them in order.
// Nulls clear. Flip state (flipped/savedVTarget) is allocated zeroed here, so each solve starts with
// every pilot row unflipped.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchPilotAllPinnedFlip(
        JNIEnv* env, jclass, jlong handle, jintArray jSlots, jintArray jRows, jdoubleArray jQTargets,
        jintArray jRlCounts, jintArray jRlPool) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlots == nullptr || env->GetArrayLength(jSlots) == 0) {
        ctx->nPilotFlip = 0;
        return;
    }
    int ne = env->GetArrayLength(jSlots);
    jint* slots = env->GetIntArrayElements(jSlots, nullptr);
    jint* rows = env->GetIntArrayElements(jRows, nullptr);
    jdouble* qt = env->GetDoubleArrayElements(jQTargets, nullptr);
    jint* cnt = env->GetIntArrayElements(jRlCounts, nullptr);
    jint* pool = env->GetIntArrayElements(jRlPool, nullptr);
    int poolLen = env->GetArrayLength(jRlPool);
    std::vector<int> off(ne);
    int acc = 0;
    for (int i = 0; i < ne; i++) {
        off[i] = acc;
        acc += cnt[i];
    }
    ctx->dPfSlot.upload(slots, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfRow.upload(rows, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfQTarget.upload(qt, static_cast<size_t>(ne) * sizeof(double));
    ctx->dPfCount.upload(cnt, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfOff.upload(off.data(), static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfRlPool.upload(pool, static_cast<size_t>(poolLen > 0 ? poolLen : 1) * sizeof(int));
    std::vector<char> zeroC(ne, 0);
    std::vector<double> zeroD(ne, 0.0);
    ctx->dPfFlipped.upload(zeroC.data(), static_cast<size_t>(ne) * sizeof(char));
    ctx->dPfSavedVT.upload(zeroD.data(), static_cast<size_t>(ne) * sizeof(double));
    ctx->nPilotFlip = ne;
    env->ReleaseIntArrayElements(jSlots, slots, JNI_ABORT);
    env->ReleaseIntArrayElements(jRows, rows, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jQTargets, qt, JNI_ABORT);
    env->ReleaseIntArrayElements(jRlCounts, cnt, JNI_ABORT);
    env->ReleaseIntArrayElements(jRlPool, pool, JNI_ABORT);
}

// Distributed-slack participants STRANDED in another synchronous component: install (or clear, with
// null) the per-scenario mask as {slot, participant} pairs. A contingency that fragments the network
// leaves some participating buses outside the main component; OLF drops them from the participant
// set, so the device must renormalize over the survivors — see distributeSlackBatch.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsIslanded(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nds == 0) {
        ctx->nDsIslanded = 0;
        return;
    }
    // Expanded to a dense (scenario, participant) mask on the host: the kernel reads it once per
    // participant inside loops it already walks, so a scan over a pair list would be the only cost
    // the mask adds to scenarios that have no islanded participant at all.
    size_t maskLen = static_cast<size_t>(ctx->nScenarios) * ctx->nds;
    std::vector<char> mask(maskLen, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nds) {
            mask[(size_t) s * ctx->nds + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dDsIslanded.upload(mask.data(), maskLen);
    ctx->nDsIslanded = nPairs;
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


/** Install the DISCRETE transformer voltage-control tables (OLF order 8): mode 0 = AFTER_GENERATOR
 *  (TransformerVoltageControlOuterLoop), 1 = WITH_GENERATOR (SimpleTransformerVoltageControlOuterLoop);
 *  entryOn = the uploaded activation already carries the control ON (a base load flow whose network
 *  OLF's initialize switched on). Groups = controlled buses (CSR over members), each with its toggle-
 *  diffed ON/OFF row deltas (CSR); members with rho row, element, base tap, tap table, r1 range and
 *  initial r1. Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteTvc(
        JNIEnv* env, jclass, jlong handle, jint mode, jint entryOn,
        jintArray jGrpPtr, jintArray jGrpVRow, jdoubleArray jGrpTargetV, jdoubleArray jGrpHalfDb, jintArray jGrpUseInit,
        jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaModeOn, jintArray jDeltaModeOff,
        jdoubleArray jDeltaTargetOn, jdoubleArray jDeltaTargetOff,
        jintArray jMemRhoRow, jintArray jMemElem, jintArray jMemBaseTap, jintArray jMemTapPtr,
        jdoubleArray jMemMin, jdoubleArray jMemMax, jdoubleArray jMemInit, jdoubleArray jTapData) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nTvd = 0;
    ctx->nTvdFz = 0;
    ctx->tvdFixTables = false;
    ctx->tvdFixWords = 0;
    if (jMemRhoRow == nullptr || env->GetArrayLength(jMemRhoRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> grpPtr, grpVRow, grpUseInit, deltaPtr, deltaRow, deltaModeOn, deltaModeOff, memRhoRow;
    std::vector<double> grpTargetV, grpHalfDb, deltaTargetOn, deltaTargetOff, memMin, memMax, memInit;
    readI(jGrpPtr, grpPtr);
    readI(jGrpVRow, grpVRow);
    readD(jGrpTargetV, grpTargetV);
    readD(jGrpHalfDb, grpHalfDb);
    readI(jGrpUseInit, grpUseInit);
    readI(jDeltaPtr, deltaPtr);
    readI(jDeltaRow, deltaRow);
    readI(jDeltaModeOn, deltaModeOn);
    readI(jDeltaModeOff, deltaModeOff);
    readD(jDeltaTargetOn, deltaTargetOn);
    readD(jDeltaTargetOff, deltaTargetOff);
    readI(jMemRhoRow, memRhoRow);
    readI(jMemElem, ctx->hTvdMemElem);
    readI(jMemBaseTap, ctx->hTvdMemBaseTap);
    readI(jMemTapPtr, ctx->hTvdMemTapPtr);
    readD(jMemMin, memMin);
    readD(jMemMax, memMax);
    readD(jMemInit, memInit);
    readD(jTapData, ctx->hTvdTapData);
    int nM = static_cast<int>(memRhoRow.size());
    int nG = static_cast<int>(grpPtr.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    for (int m = 0; m < nM; m++) {
        if (ctx->hTvdMemTapPtr[m + 1] - ctx->hTvdMemTapPtr[m] <= 0) {
            return 0;                                        // no tap table: the host orchestrator keeps the control
        }
    }
    ctx->dTvdGrpPtr.upload(grpPtr.data(), grpPtr.size() * sizeof(int));
    ctx->dTvdGrpVRow.upload(grpVRow.data(), nG * sizeof(int));
    ctx->dTvdGrpTargetV.upload(grpTargetV.data(), nG * sizeof(double));
    ctx->dTvdGrpHalfDb.upload(grpHalfDb.data(), nG * sizeof(double));
    ctx->dTvdGrpUseInit.upload(grpUseInit.data(), nG * sizeof(int));
    ctx->dTvdDeltaPtr.upload(deltaPtr.data(), deltaPtr.size() * sizeof(int));
    ctx->dTvdDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->dTvdDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->dTvdMemRhoRow.upload(memRhoRow.data(), nM * sizeof(int));
    ctx->dTvdMemElem.upload(ctx->hTvdMemElem.data(), nM * sizeof(int));
    ctx->dTvdMemBaseTap.upload(ctx->hTvdMemBaseTap.data(), nM * sizeof(int));
    ctx->dTvdMemTapPtr.upload(ctx->hTvdMemTapPtr.data(), ctx->hTvdMemTapPtr.size() * sizeof(int));
    ctx->dTvdMemMin.upload(memMin.data(), nM * sizeof(double));
    ctx->dTvdMemMax.upload(memMax.data(), nM * sizeof(double));
    ctx->dTvdMemInit.upload(memInit.data(), nM * sizeof(double));
    ctx->dTvdTapData.upload(ctx->hTvdTapData.data(), ctx->hTvdTapData.size() * sizeof(double));
    size_t nS = ctx->nScenarios;
    ctx->dTvdScnStep.ensure(nS * sizeof(int));
    ctx->dTvdGrpStep.ensure(nS * nG * sizeof(int));
    ctx->dTvdGrpShMin.ensure(nS * nG * sizeof(double));
    ctx->dTvdGrpShMax.ensure(nS * nG * sizeof(double));
    ctx->dTvdGrpShInit.ensure(nS * nG * sizeof(double));
    ctx->dTvdMemState.ensure(nS * nM * sizeof(int));
    ctx->dTvdMemTap.ensure(nS * nM * sizeof(int));
    ctx->dTvdMemRoundFrom.ensure(nS * nM * sizeof(double));
    ctx->dTvdSaved.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->dTvdReject.ensure(nS * sizeof(char));
    ctx->nTvdGroups = nG;
    ctx->nTvdDelta = nD;
    ctx->tvdMode = mode;
    ctx->tvdEntryOn = entryOn;
    ctx->nTvd = nM;
    return 1;
}

/** GeneratorVoltageControlManager tables for the AFTER_GENERATOR phase installed by setBatchDiscreteTvc:
 *  the freeze candidates (controller bus V row, load Q) and fixTransformerVoltageControls' per-member
 *  PV-candidate row lists (CSR, [nM+1]; null = no connectivity fix). Returns 1 when installed. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteTvcFreeze(
        JNIEnv* env, jclass, jlong handle, jintArray jFzVRow, jdoubleArray jFzLoadQ, jintArray jFzRlEntry,
        jintArray jFixPvRow) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> vRow, fixRow;
    std::vector<double> loadQ;
    if (jFzVRow != nullptr) {
        readI(jFzVRow, vRow);
        loadQ.resize(env->GetArrayLength(jFzLoadQ));
        env->GetDoubleArrayRegion(jFzLoadQ, 0, static_cast<jsize>(loadQ.size()), loadQ.data());
    }
    int nFz = static_cast<int>(vRow.size());
    if (nFz > 0) {
        std::vector<int> rlEntry;
        if (jFzRlEntry != nullptr) {
            readI(jFzRlEntry, rlEntry);
        }
        if (static_cast<int>(rlEntry.size()) != nFz) {
            rlEntry.assign(nFz, -1);                         // no remote candidate
        }
        ctx->dTvdFzVRow.upload(vRow.data(), nFz * sizeof(int));
        ctx->dTvdFzLoadQ.upload(loadQ.data(), nFz * sizeof(double));
        ctx->dTvdFzRlEntry.upload(rlEntry.data(), nFz * sizeof(int));
        size_t nsc = static_cast<size_t>(ctx->nScenarios) * nFz;
        ctx->dTvdFzFrozen.ensure(nsc * sizeof(int));
        ctx->dTvdFzSavedMode.ensure(nsc * sizeof(int));
        ctx->dTvdFzSavedTarget.ensure(nsc * sizeof(double));
    }
    ctx->nTvdFz = nFz;
    ctx->tvdFixTables = false;                               // armed by setBatchDiscreteTvcFixMasks
    ctx->tvdFixWords = 0;
    if (jFixPvRow != nullptr) {
        readI(jFixPvRow, fixRow);
        if (!fixRow.empty()) {
            ctx->dTvdFixPvRow.upload(fixRow.data(), fixRow.size() * sizeof(int));
        }
    }
    return 1;
}

// The per-(scenario, member) connectivity masks of fixTransformerVoltageControls. Uploaded per CHUNK,
// unlike the tables above: OLF re-runs the check on the POST-CONTINGENCY network, so the component of a
// not-controlled side is a property of the scenario, not of the run.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteTvcFixMasks(
        JNIEnv* env, jclass, jlong handle, jintArray jMin, jintArray jMax, jint words) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0) {
        return;
    }
    ctx->tvdFixTables = false;
    ctx->tvdFixWords = 0;
    if (jMin == nullptr || jMax == nullptr || words <= 0) {
        return;
    }
    size_t want = static_cast<size_t>(ctx->nScenarios) * ctx->nTvd * words;
    if (static_cast<size_t>(env->GetArrayLength(jMin)) != want
            || static_cast<size_t>(env->GetArrayLength(jMax)) != want) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchDiscreteTvcFixMasks: array length != nScenarios * nMembers * words");
        return;
    }
    std::vector<int> mn(want), mx(want);
    env->GetIntArrayRegion(jMin, 0, static_cast<jsize>(want), mn.data());
    env->GetIntArrayRegion(jMax, 0, static_cast<jsize>(want), mx.data());
    ctx->dTvdFixMinMask.upload(mn.data(), want * sizeof(int));
    ctx->dTvdFixMaxMask.upload(mx.data(), want * sizeof(int));
    ctx->tvdFixWords = words;
    ctx->tvdFixTables = true;
}

/** Per (scenario, freeze candidate): {frozen (0/1), the row target while frozen (gen − load)} — layout
 *  {@code (s * nFz + c) * 2}; null without candidates. */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteTvcFreeze(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0 || ctx->nTvdFz <= 0 || !ctx->dTvdFzFrozen.p) {
        return nullptr;
    }
    int n = ctx->n;
    size_t cnt = static_cast<size_t>(ctx->nTvdFz) * ctx->nScenarios;
    std::vector<int> fz(cnt), vrow(ctx->nTvdFz);
    std::vector<double> tgt(static_cast<size_t>(n) * ctx->nScenarios), out(cnt * 2);
    cudaMemcpy(fz.data(), ctx->dTvdFzFrozen.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vrow.data(), ctx->dTvdFzVRow.p, vrow.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tgt.data(), ctx->dTarget.p, tgt.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (int s = 0; s < ctx->nScenarios; s++) {
        for (int c = 0; c < ctx->nTvdFz; c++) {
            size_t i = static_cast<size_t>(s) * ctx->nTvdFz + c;
            out[i * 2] = fz[i];
            out[i * 2 + 1] = tgt[static_cast<size_t>(s) * n + vrow[c]];
        }
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Per (scenario, member): {state (0 off, 1 continuous, 2 fixed), tap index, the r1 fed to the
 *  rounding} — layout {@code (s * nM + m) * 3}; null without the table. */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteTvcState(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0 || !ctx->dTvdMemState.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nTvd) * ctx->nScenarios;
    std::vector<int> st(cnt), tap(cnt);
    std::vector<double> rf(cnt), out(cnt * 3);
    cudaMemcpy(st.data(), ctx->dTvdMemState.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tap.data(), ctx->dTvdMemTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(rf.data(), ctx->dTvdMemRoundFrom.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < cnt; i++) {
        out[i * 3] = st[i];
        out[i * 3 + 1] = tap[i];
        out[i * 3 + 2] = rf[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Install the DISCRETE (WITH_GENERATOR) shunt voltage-control tables (OLF order 10): groups =
 *  controlled buses (CSR over member shunts) with their toggle-diffed ON/OFF row deltas; members with
 *  SHUNT_B row, element (-1 = unpacked, never outaged) and sub-controllers (CSR) whose section-B tables,
 *  min and base positions drive LfShuntImpl.dispatchB. Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteShunt(
        JNIEnv* env, jclass, jlong handle, jint entryOn,
        jintArray jGrpPtr, jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaModeOn, jintArray jDeltaModeOff,
        jdoubleArray jDeltaTargetOn, jdoubleArray jDeltaTargetOff,
        jintArray jMemBRow, jintArray jMemElem, jintArray jMemSubPtr,
        jintArray jSecPtr, jdoubleArray jSecB, jintArray jSubMin, jintArray jSubBase) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nSvd = 0;
    if (jMemBRow == nullptr || env->GetArrayLength(jMemBRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> grpPtr, deltaPtr, deltaRow, deltaModeOn, deltaModeOff, memBRow, memElem, memSubPtr, secPtr, subMin, subBase;
    std::vector<double> deltaTargetOn, deltaTargetOff, secB;
    readI(jGrpPtr, grpPtr);
    readI(jDeltaPtr, deltaPtr);
    readI(jDeltaRow, deltaRow);
    readI(jDeltaModeOn, deltaModeOn);
    readI(jDeltaModeOff, deltaModeOff);
    readD(jDeltaTargetOn, deltaTargetOn);
    readD(jDeltaTargetOff, deltaTargetOff);
    readI(jMemBRow, memBRow);
    readI(jMemElem, memElem);
    readI(jMemSubPtr, memSubPtr);
    readI(jSecPtr, secPtr);
    readD(jSecB, secB);
    readI(jSubMin, subMin);
    readI(jSubBase, subBase);
    int nM = static_cast<int>(memBRow.size());
    int nG = static_cast<int>(grpPtr.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    int nSubs = static_cast<int>(subBase.size());
    ctx->dSvdGrpPtr.upload(grpPtr.data(), grpPtr.size() * sizeof(int));
    ctx->dSvdDeltaPtr.upload(deltaPtr.data(), deltaPtr.size() * sizeof(int));
    ctx->dSvdDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->dSvdDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->dSvdMemBRow.upload(memBRow.data(), nM * sizeof(int));
    ctx->dSvdMemElem.upload(memElem.data(), nM * sizeof(int));
    ctx->dSvdMemSubPtr.upload(memSubPtr.data(), memSubPtr.size() * sizeof(int));
    ctx->dSvdSecPtr.upload(secPtr.data(), secPtr.size() * sizeof(int));
    ctx->dSvdSecB.upload(secB.data(), std::max<size_t>(secB.size(), 1) * sizeof(double));
    ctx->dSvdSubMin.upload(subMin.data(), std::max(nSubs, 1) * sizeof(int));
    ctx->dSvdSubBase.upload(subBase.data(), std::max(nSubs, 1) * sizeof(int));
    size_t nS = ctx->nScenarios;
    ctx->dSvdGrpStep.ensure(nS * nG * sizeof(int));
    ctx->dSvdMemState.ensure(nS * nM * sizeof(int));
    ctx->dSvdMemContB.ensure(nS * nM * sizeof(double));
    ctx->dSvdSubPos.ensure(nS * std::max(nSubs, 1) * sizeof(int));
    ctx->dSvdSaved.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->nSvdGroups = nG;
    ctx->nSvdDelta = nD;
    ctx->nSvdSubs = nSubs;
    ctx->svdEntryOn = entryOn;
    ctx->nSvd = nM;
    return 1;
}

/** Per (scenario, member shunt): {state (0 off, 1 continuous, 2 dispatched), the continuous b the
 *  dispatch started from} — layout {@code (s * nM + m) * 2}; null without the table. */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteShuntState(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvd <= 0 || !ctx->dSvdMemState.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvd) * ctx->nScenarios;
    std::vector<int> st(cnt);
    std::vector<double> cb(cnt), out(cnt * 2);
    cudaMemcpy(st.data(), ctx->dSvdMemState.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cb.data(), ctx->dSvdMemContB.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < cnt; i++) {
        out[i * 2] = st[i];
        out[i * 2 + 1] = cb[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Per (scenario, sub-controller): the dispatched section — layout {@code s * nSubs + u}. */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteShuntPositions(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvd <= 0 || ctx->nSvdSubs <= 0 || !ctx->dSvdSubPos.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvdSubs) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dSvdSubPos.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
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
        applyReorderAlg(ds.config, false);   // adjoint: its own matrix, matching enabled
        applyMatchingAlg(ds.config);
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
