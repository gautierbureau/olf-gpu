// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Error checks for the CUDA runtime, cuBLAS and cuDSS calls, and the cuDSS version shim: the
// cudssMatrixCreate* type arguments differ between cuDSS 0.7.x and 0.8.
#ifndef OLF_GPU_COMMON_CHECK_CUH
#define OLF_GPU_COMMON_CHECK_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudss.h>

#include <stdexcept>
#include <string>

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

// The batched linear solve. The batch context's `mode` (GpuBatchContextCache) picks one of two setups:
//
//   GpuLinearSolveMode::UBATCH (the default): a SINGLE per-block CSR with value/RHS pointers aimed at the batched
//     buffers + CUDSS_CONFIG_UBATCH_SIZE = nScenarios. cuDSS analyzes ONE block, then factorizes and
//     solves every scenario in one call. The chunk size is decisive on cuDSS 0.8: its UBATCH phases carry
//     a high per-launch fixed cost that amortizes with batch size (pegase9241, 500 ctg, warm ms/scenario:
//     chunk 32 = 17.8, 125 = 6.35, 500 = 4.87), hence the memory auto-sizing of the chunk.
//
//   GpuLinearSolveMode::ROBUST: a per-scenario loop over a single matrix with BTF-COLAMD ordering, global column
//     pivoting and matching. What a batch of one retries in after a linear-solve reject (rte6515's full
//     busbar structure does not factor in the uniform batch).
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

#endif  // OLF_GPU_COMMON_CHECK_CUH
