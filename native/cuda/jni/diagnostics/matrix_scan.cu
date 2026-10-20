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
void dumpSlotSystem(BatchSolverContext* ctx, int n, int nnz, int slot, const char* path) {
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<double> hv(nnz);
    std::vector<double> hd(n);
    std::vector<double> hff(n);
    std::vector<int> hrp(n + 1);
    std::vector<int> hci(nnz);
    std::vector<int> hmode(n);
    std::vector<double> hb(n);                               // the RHS as SOLVED (row-repaired, see hazRepairRhs)
    CUDA_CHECK(cudaMemcpy(hb.data(), ctx->dB.template as<double>() + (size_t) slot * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hv.data(), ctx->dVals.template as<double>() + (size_t) slot * nnz, (size_t) nnz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hd.data(), ctx->dDx.template as<double>() + (size_t) slot * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hff.data(), ctx->dF.template as<double>() + (size_t) slot * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hrp.data(), ctx->st->dRowPtr.p, (size_t) (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hci.data(), ctx->st->dColIdx.p, (size_t) nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hmode.data(), ctx->dMode.template as<int>() + (size_t) slot * n, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost));
    FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
        std::fprintf(stderr, "OLF_GPU_DX_DUMP_ON_REJECT cannot open %s\n", path);
        return;
    }
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
    std::fprintf(stderr, "OLF_GPU_DX_DUMP_ON_REJECT wrote %s (n=%d nnz=%d slot=%d)\n", path, n, nnz, slot);
}

}  // namespace olfgpu
