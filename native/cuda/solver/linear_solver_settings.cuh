// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The run's linear-solver settings (GpuLoadFlowParameters, installed by setLinearSolverSettings before the
// run's contexts are built) and the cuDSS configuration every context applies from them: ordering,
// matching, pivoting and iterative refinement.
#ifndef OLF_GPU_SOLVER_LINEAR_SOLVER_SETTINGS_CUH
#define OLF_GPU_SOLVER_LINEAR_SOLVER_SETTINGS_CUH

#include <cudss.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "../common/check.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

// cuDSS internal refinement steps of a system WITHOUT zero-impedance couplers. The outer Newton loop corrects
// any linear-solve inexactness there, so refining each solve is redundant (measured ~30% faster on batched N-1
// with identical convergence). Reusing pivots through CUDSS_PHASE_REFACTORIZATION gives no speedup in batched
// mode either (the factorization cost is the numeric fill, not pivoting), so it is not used.
constexpr int ITERATIVE_REFINEMENT_STEPS_WITHOUT_COUPLERS = 0;

// Refinement IS required on a system WITH zero-impedance couplers. Each coupler puts ZERO_V / ZERO_PHI on its
// DUMMY_P / DUMMY_Q rows, carrying v1-v2 and phi1-phi2 and never their own dummy, so two diagonals per coupler
// are structurally zero; the zero-diagonal repair (repairZeroDiagonalRowsBatch) makes every row nonsingular, but a borrowed
// +-1 diagonal among O(100) admittance entries is still factored inaccurately by static pivoting. The backward
// residual stays ~1e-14 while the forward error in dx, with kappa ~ 5e8, is what moves. Measured on rte6515's
// 500 busbar outages: with refinement the worst ||J.dx-F||inf is 1.5e-11 and buses match the CPU to 2.6e-9 kV;
// without, 3.8e-4 on the repaired rows and 2.6e-8 kV. Nested dissection reaches the same accuracy without
// refinement but collapses the batch at scale (0/80 busbar scenarios batched against AMD's 60/80).
//
// Refinement is ADAPTIVE: cuDSS runs configuredIterativeRefinementBaseSteps internally, then the batch loop measures every lane's backward
// residual ratio ||b - J dx||inf / ||b||inf and refines (r = b - J dx, J d = r, dx += d) only while an active
// lane sits above iterativeRefinementTolerance, up to configuredIterativeRefinementMaxSteps solves in all. The count a lane needs depends on which other matrices
// share its batch, so it is decided per iteration from the residual itself (network.xiidm: a fixed 4 steps cost
// 64.7 s of 126 s of kernels, a fixed 2 left two busbar scenarios at a linear-solve failure). The internal step
// matters on the busbar scopes: full network.xiidm busbar scope, 12869 of 12918 contingencies on device with it
// against 5121-5422 without.
//
// -1 = AUTO: the per-context choice each apply* function documents; a NaN threshold = the library's own.
struct LinearSolverSettings {
    int reorder = -1;                                        // index into REORDER_ALGS, -1 = AUTO (AMD on cuDSS 0.8)
    int matching = 6;                                        // cudssMatchingAlg_t, CUDSS_MATCHING_ALG_AUTO
    int pivotType = -1;                                      // cudssPivotType_t, -1 = derived from the ordering
    double pivotThreshold = NAN;
    int iterativeRefinementBaseSteps = 1;                                          // cuDSS internal refinement steps (coupler systems)
    int iterativeRefinementMaxSteps = 4;                                           // adaptive refinement: at most this many solves
    double iterativeRefinementTolerance = 1e-12;                                    // adaptive refinement: backward-residual target
    int batchedLuOrdering = 0;                               // BatchedLuOrdering: the elimination order our batched LU is planned on
};

// The elimination order of our batched LU (GpuBatchedLuOrdering): cuDSS's analysis order, the host's minimum degree of
// A + A^T (minimum_degree.cuh), or whichever of the two has the fewer elimination levels on the structure.
enum BatchedLuOrdering : int { BATCHED_LU_ORDERING_CUDSS = 0, BATCHED_LU_ORDERING_MINIMUM_DEGREE = 1,
                               BATCHED_LU_ORDERING_FEWEST_LEVELS = 2 };
extern LinearSolverSettings gLinearSolver;

inline int configuredIterativeRefinementBaseSteps() {
    return gLinearSolver.iterativeRefinementBaseSteps;
}
inline int configuredIterativeRefinementMaxSteps() {
    return gLinearSolver.iterativeRefinementMaxSteps;
}
inline double iterativeRefinementTolerance() {
    return gLinearSolver.iterativeRefinementTolerance;
}

// The fill-reducing orderings, indexed by GpuReorderingAlgorithm. Factorization cost is LU-fill-bound and the fill
// is decided by the ordering cuDSS picks at ANALYSIS; one run uses one ordering on every config it creates. AUTO on
// cuDSS >= 0.8 is AMD, the sweep winner on case9241pegase (LU_NNZ 260076 vs 304772 for the library default, warm
// SA -11%); BTF_COLAMD and COLAMD are NOT_SUPPORTED at the UBATCH analysis. AUTO on 0.7.1 is the library default.
// cuDSS 0.7 names the same ordinals CUDSS_ALG_1..5.
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

// Matching (MC64-style), for the robust single-matrix mode and the standalone transposed solve. The uniform
// batch cannot use it (cuDSS 0.8 rejects matching at the uniform-batch analysis), which is why the batched path
// repairs its zero diagonals itself (repairZeroDiagonalRowsBatch). A raw OLF Jacobian has no nonzero-diagonal guarantee: a bus
// fed only through a zero-impedance branch leaves four rows with a ZERO diagonal (the branch's ZERO_V / ZERO_PHI
// and the dead-end bus's P / Q balances). Without matching cuDSS perturbs the zero pivot and returns a wrong
// answer with a matching residual (ieee14, error vs KLU: AMD 7.2e-3, nested dissection 7.2e-2), so matching is
// never NONE here.
inline void applyMatchingAlg(cudssConfig_t config) {
#if CUDSS_VERSION >= 800
    // Never CUDSS_MATCHING_ALG_NONE: these Jacobians need matching (see above), so the parameter does not offer it.
    cudssMatchingAlg_t alg = static_cast<cudssMatchingAlg_t>(gLinearSolver.matching);
    CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_MATCHING_ALG, &alg, sizeof(alg)));
#else
    (void) config;                                           // 0.7.1 has no matching knob
#endif
}

inline void applyReorderAlg(cudssConfig_t config, bool hasZeroImpedance);   // defined below

// MODE 6 = the ROBUST single-matrix solve: a per-scenario loop with matching and pivoting on cuDSS's
// single-matrix API (the uniform batch supports neither). It is what a factorization needs on rte6515's full
// busbar structure (366 retained couplers, 1e-18 pivots under every unpivoted ordering). The batch-of-one falls
// into it after a linear-solve reject (GpuNewtonRaphson); tests force it through
// GpuBatchContextCache.forceModeForTests. Matching is REQUIRED (pivoting alone left the full-scope base at NaN);
// its permutation is value-dependent and cuDSS accepts ONE analysis per data object, so the analysis runs once
// on the run's first values and later factorizations reuse it. A system cuDSS refuses under this configuration
// throws, and the batch-of-one falls back to the CPU. cuDSS 0.8 refuses matching under BTF_COLAMD / COLAMD, so
// AUTO orders with AMD and AUTO pivoting; under the COLAMD family the default pivot is GLOBAL_COL.
inline void applyRobustSingleMatrixConfig(cudssConfig_t config) {
#if CUDSS_VERSION >= 800
    const int reorder = gLinearSolver.reorder;
    bool colamdFamily = true;                                // BTF_COLAMD / COLAMD: global column pivoting is defined
    if (reorder < 0) {
        OlfReorderAlg amd = CUDSS_REORDERING_ALG_AMD;
        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &amd, sizeof(amd)));
        colamdFamily = false;
    } else {
        applyReorderAlg(config, true);
        colamdFamily = reorder == 1 || reorder == 2;          // REORDER_ALGS: btf_colamd, colamd
    }
    cudssPivotType_t pt = colamdFamily ? CUDSS_PIVOT_GLOBAL_COL : CUDSS_PIVOT_AUTO;
    if (gLinearSolver.pivotType >= 0) {
        pt = static_cast<cudssPivotType_t>(gLinearSolver.pivotType);
    }
    CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_PIVOT_TYPE, &pt, sizeof(pt)));
    if (!std::isnan(gLinearSolver.pivotThreshold)) {
        double thr = gLinearSolver.pivotThreshold;
        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_PIVOT_THRESHOLD, &thr, sizeof(thr)));
    }
    if (traceOn(Trace::BATCH_TRACE)) {
        std::fprintf(stderr, "ROBUST_CFG reorder=%d pivot=%d matching=%d threshold=%g\n",
                     reorder, static_cast<int>(pt), gLinearSolver.matching, gLinearSolver.pivotThreshold);
    }
    applyMatchingAlg(config);
#else
    (void) config;
#endif
}

// The configured ordering, AMD under AUTO. AMD is a symmetric ordering that assumes a nonzero diagonal, which a
// zero-impedance coupler always breaks; the accuracy it loses there is what the iterative refinement above
// recovers, so the ordering itself does not depend on the couplers.
inline void applyReorderAlg(cudssConfig_t config, bool hasZeroImpedance) {
    const int reorder = gLinearSolver.reorder;
    if (reorder < 0) {
#if CUDSS_VERSION >= 800
            OlfReorderAlg amd = CUDSS_REORDERING_ALG_AMD;        // sweep winner; ND collapses the
        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &amd, sizeof(amd)));
#endif
        return;                                              // 0.7.1: library default, untouched
    }
    (void) hasZeroImpedance;
    if (reorder >= static_cast<int>(sizeof(REORDER_ALGS) / sizeof(REORDER_ALGS[0]))) {
        throw std::runtime_error("reordering algorithm index out of range: " + std::to_string(reorder));
    }
    OlfReorderAlg alg = REORDER_ALGS[reorder];
    CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)));
}

// OLF_GPU_TRACE: the LU fill after an ANALYSIS, one stderr line. Non-fatal: some (version, ordering)
// combinations do not expose CUDSS_DATA_LU_NNZ.
inline void traceLuNnz(cudssHandle_t handle, cudssData_t data, const char* where, int nScenarios = 0,
                       double ms = -1.0) {
    if (!traceOn(Trace::TRACE)) {
        return;
    }
#if OLF_GPU_CUDSS_LU_NNZ
    int64_t luNnz = 0;
    size_t written = 0;
    if (cudssDataGet(handle, data, CUDSS_DATA_LU_NNZ, &luNnz, sizeof(luNnz), &written) == CUDSS_STATUS_SUCCESS
            && luNnz > 0) {
        std::fprintf(stderr, "OLF_GPU_CUDSS reorder=%d %s nScenarios=%d LU_NNZ=%lld analysis=%.1f ms\n",
                     gLinearSolver.reorder, where, nScenarios, static_cast<long long>(luNnz), ms);
        return;
    }
#endif
    std::fprintf(stderr, "OLF_GPU_CUDSS %s LU_NNZ=unavailable\n", where);
}

}  // namespace olfgpu

#endif  // OLF_GPU_SOLVER_LINEAR_SOLVER_SETTINGS_CUH
