// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See cudss_order.cuh.
#include "cudss_order.cuh"

#include <cudss.h>

#include "../common/check.cuh"
#include "../common/device_buf.cuh"
#include "../common/stats.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "../structure/shared_structure.cuh"

namespace olfgpu {

void cudssEliminationOrder(int n, int nnz, const int* dRowPtr, const int* dColIdx, bool hasZeroImpedance,
                           std::vector<int>& rowOrder, std::vector<int>& colOrder, int64_t& luNnz) {
    DeviceBuf dVals;
    DeviceBuf dB;
    DeviceBuf dX;
    dVals.alloc(static_cast<size_t>(nnz) * sizeof(double));            // ANALYSIS is symbolic: values unused
    dB.alloc(static_cast<size_t>(n) * sizeof(double));
    dX.alloc(static_cast<size_t>(n) * sizeof(double));
    CudssCtx solver;
    CUDSS_CHECK(cudssCreate(&solver.handle));
    CUDSS_CHECK(cudssConfigCreate(&solver.config));
    applyReorderAlg(solver.config, hasZeroImpedance);
    CUDSS_CHECK(cudssDataCreate(solver.handle, &solver.data));
    CUDSS_CHECK(cudssMatrixCreateCsr(&solver.systemMatrix, n, n, nnz, const_cast<int*>(dRowPtr), nullptr,
                                     const_cast<int*>(dColIdx), dVals.as<double>(), OLF_CUDSS_CSR_TYPES,
                                     CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&solver.rightHandSideMatrix, n, 1, n, dB.as<double>(), OLF_CUDSS_DN_TYPE,
                                    CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&solver.solutionMatrix, n, 1, n, dX.as<double>(), OLF_CUDSS_DN_TYPE,
                                    CUDSS_LAYOUT_COL_MAJOR));
    gAnalysesMain++;
    gCudssOrders++;
    {
        AnalysisTimer at_(__LINE__);
        CUDSS_CHECK(cudssExecute(solver.handle, CUDSS_PHASE_ANALYSIS, solver.config, solver.data, solver.systemMatrix,
                                 solver.solutionMatrix, solver.rightHandSideMatrix));
    }
    rowOrder.resize(n);
    colOrder.resize(n);
    size_t written = 0;
    CUDSS_CHECK(cudssDataGet(solver.handle, solver.data, CUDSS_DATA_PERM_REORDER_ROW, rowOrder.data(),
                             rowOrder.size() * sizeof(int), &written));
    CUDSS_CHECK(cudssDataGet(solver.handle, solver.data, CUDSS_DATA_PERM_REORDER_COL, colOrder.data(),
                             colOrder.size() * sizeof(int), &written));
    luNnz = 0;
    CUDSS_CHECK(cudssDataGet(solver.handle, solver.data, CUDSS_DATA_LU_NNZ, &luNnz, sizeof(luNnz), &written));
}

}  // namespace olfgpu
