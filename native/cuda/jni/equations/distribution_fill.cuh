// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The distribution rows of the batched Jacobian and mismatch: DISTR_Q (remote generator voltage control),
// DISTR_RHO (shared transformer voltage control) and DISTR_SHUNT_B (shared shunt voltage control).
#ifndef OLF_GPU_EQUATIONS_DISTRIBUTION_FILL_CUH
#define OLF_GPU_EQUATIONS_DISTRIBUTION_FILL_CUH

#include "../solver/driver_pass.cuh"

namespace olfgpu {

int gpuDqTraceRow();

__global__ void fillDistrQBatch(int ndq, int nScenarios, int n, int nnz, int nElements,
                                int cbOffset, int obOffset,
                                const int* dqElem, const int* dqSide, const int* dqKind,
                                const double* dqWeight, const int* dqDest,
                                const int* dqAlt, const int* dqSel, const int* dqZiRow,
                                const double* cbIn, const int* cbIx, const double* shIn, const int* shIx,
                                const double* obIn, const int* obIx,
                                const double* x, double* vals, double* F,
                                const char* disabled, const char* active,
                                int nDqOv, const int* dqOvIdx, const double* dqOvW,
                                int nDqKeyOv, const int* dqKeyOvIdx, const double* dqKeyOvKey,
                                const int* mode,
                                int nDqInact, const int* dqInactSlotMember,
                                const int* dqKMember, const int* dqJMember, const double* mKey,
                                const int* mRlEntry, const int* mGroupStart, const int* mGroupCount,
                                const char* limitType, int nrl,
                                const char* psSkip, int psGroups,
                                const int* mQrGroup, const int* qrPtr, const int* qrRows,
                                int nR1Ov, const int* r1OvSlotElem, const double* r1OvVal,
                                int nShOv, const int* shOvSlotElem, const double* shOvGb,
                                int dqTraceRow,
                                const char* dqSub, const int* qrMemberStart, int nDqMembersAll);

__global__ void fillDistrRhoBatch(int ndr, int nScenarios, int n, int nnz,
                                  const int* drCol, const double* drCoef, const int* drDest,
                                  const double* x, double* vals, double* F, const char* active,
                                  int nDrOv, const int* drOvIdx, const double* drOvC,
                                  const int* drQrGroup, const int* mode, const int* qrPtr, const int* qrRows);

__global__ void fillDistrShuntBBatch(int ndb, int nScenarios, int n, int nnz,
                                     const int* dbCol, const double* dbCoef, const int* dbDest,
                                     const double* x, double* vals, double* F, const char* active);

}  // namespace olfgpu

#endif  // OLF_GPU_EQUATIONS_DISTRIBUTION_FILL_CUH
