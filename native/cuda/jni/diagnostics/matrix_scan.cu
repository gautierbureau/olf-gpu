// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Diagnostics of the assembled linear system, see matrix_scan.cuh.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/check.cuh"
#include "matrix_scan.cuh"

namespace olfgpu {

/** One scenario's linear system AS SOLVED, written whole: the row-repaired Jacobian (CSR triplets), the mismatch F it
 *  was built at, the repaired right-hand side B, the step dx the factorization returned and each row's mode. Read by
 *  the KLU re-solve probe and the structural-matching script (OLF_GPU_DX_DUMP_ON_REJECT). */
void dumpSlotSystem(BatchSolverContext* ctx, int n, int nnz, int scenario, const char* path) {
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<double> jacobianValues(nnz);
    std::vector<double> dx(n);
    std::vector<double> mismatch(n);
    std::vector<int> rowStart(n + 1);
    std::vector<int> columnIndex(nnz);
    std::vector<int> mode(n);
    std::vector<double> rightHandSide(n);                               // the RHS as SOLVED (row-repaired, see repairZeroDiagonalRightHandSideBatch)
    CUDA_CHECK(cudaMemcpy(rightHandSide.data(), ctx->dB.template as<double>() + (size_t) scenario * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(jacobianValues.data(), ctx->dVals.template as<double>() + (size_t) scenario * nnz, (size_t) nnz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dx.data(), ctx->dDx.template as<double>() + (size_t) scenario * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mismatch.data(), ctx->dF.template as<double>() + (size_t) scenario * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rowStart.data(), ctx->structure->dRowPtr.p, (size_t) (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(columnIndex.data(), ctx->structure->dColIdx.p, (size_t) nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mode.data(), ctx->dMode.template as<int>() + (size_t) scenario * n, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost));
    FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
        std::fprintf(stderr, "OLF_GPU_DX_DUMP_ON_REJECT cannot open %s\n", path);
        return;
    }
    std::fprintf(f, "N %d %d\n", n, nnz);
    for (int i = 0; i < n; i++) {
        std::fprintf(f, "F %d %.17e\n", i, mismatch[i]);
        std::fprintf(f, "DX %d %.17e\n", i, dx[i]);
        std::fprintf(f, "B %d %.17e\n", i, rightHandSide[i]);
        std::fprintf(f, "MODE %d %d\n", i, mode[i]);
        for (int k = rowStart[i]; k < rowStart[i + 1]; k++) {
            std::fprintf(f, "J %d %d %.17e\n", i, columnIndex[k], jacobianValues[k]);
        }
    }
    std::fclose(f);
    std::fprintf(stderr, "OLF_GPU_DX_DUMP_ON_REJECT wrote %s (n=%d nnz=%d slot=%d)\n", path, n, nnz, scenario);
}

}  // namespace olfgpu
