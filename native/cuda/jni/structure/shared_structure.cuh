// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The SHARED STRUCTURE built once per extraction (GpuAcData) and every batched context reads: the CSR
// pattern, the per-family device data with role-folded destinations, the distribution and zero-impedance
// packs, plus the cuDSS object holder and the zero-impedance index readers.
#ifndef OLF_GPU_STRUCTURE_SHARED_STRUCTURE_CUH
#define OLF_GPU_STRUCTURE_SHARED_STRUCTURE_CUH

#include <cudss.h>

#include <memory>
#include <vector>

#include "../common/device_buf.cuh"
#include "csr_pattern.cuh"

namespace olfgpu {

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
    DeviceBuf relocDep;                                      // per output: 1 = dependent-pilot routing (depPick)
    bool dynReloc = false;                                   // slot-pool offset; pool = one slot per candidate
    DeviceBuf pcDest, pcSel;                                 // Continuous phase control (cb only): P-flow row at a1
    bool phaseCtl = false;                                   // pcDest/pcSel populated (mode-3 gated emission)
};

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
// STOCK placement (no child): ZERO_V on the DUMMY_Q row and ZERO_PHI on the DUMMY_P row - CROSSWISE. Neither
// equation holds a dummy, so either way both rows are zero-diagonal hazards the numeric repair fills from a
// donor row carrying that dummy: a Q row for DUMMY_Q, a P row for DUMMY_P. The repaired row is then
// row_h + s*row_d, and the 2x2 principal block of (row_d, row_h') on (donor diagonal, dummy) is singular
// unless row_h has an entry of its own in the donor's diagonal column: ZERO_V (v_a - rho*v_b) has one in a
// Q row's V diagonal, ZERO_PHI (phi_a - phi_b) in a P row's PHI diagonal. The other way round (ZERO_V on the
// DUMMY_P row, repaired from a P row) the block is {[955.8, 1], [-955.8, -1]} exactly - measured on
// network.xiidm's cut trees, 35 such pairs in one dump, and the unpivoted uniform-batch factorization lost
// them (rte6515: 37 stock couplers in every scenario's matrix). Crosswise, det = -+1 on every such block.
inline int ziZeroVRow(const int* z) {
    int d = ziDeadSide(z);
    return d == 0 ? z[5] : (d == 1 ? z[6] : z[7]);           // the child's own v row
}
inline int ziZeroPhiRow(const int* z) {
    int d = ziDeadSide(z);
    return d == 0 ? z[4] : (d == 1 ? z[8] : z[9]);           // the child's own phi row
}

// The SHARED STRUCTURE: everything a batched context derives from ONE extraction
// (GpuAcData) — the diagonal-pairing row permutation, the CSR pattern (host + device),
// the per-family device data with role-folded destination arrays, the DISTR_Q /
// DISTR_RHO / zero-impedance packs, the per-element static trig. It is built ONCE per
// extraction (cached Java-side per GpuAcData by GpuAcStructures) and held by shared_ptr
// — a batched context outliving the Java-side owner keeps the structure alive.

struct SharedStructure {
    int n = 0;
    int nnz = 0;
    int nElements = 0;                                       // ncb + nob + nsh + nhv (disabled-mask layout)
    int cbOffset = 0, obOffset = 0, shOffset = 0, hvOffset = 0;
    std::vector<int> rowMap;                                 // eq row -> paired variable col
    std::vector<int> rowPtr, colIdx;                         // host CSR pattern (mode-2 blkdiag build)
    Family cb, ob, sh, hv;
    int ndq = 0;                                             // DISTR_Q weighted reactive contributions
    DeviceBuf dDqElem, dDqSide, dDqKind, dDqWeight, dDqDest; // ndq each (dDqDest is ndq*6)
    DeviceBuf dRelocOwn;                                     // per row: the relocated child's OWN row for a
                                                             // coupler DUMMY row, -1 elsewhere (stock placement)
    DeviceBuf dDqZiRow;                                      // per contribution: a kind-3 coupler dummy
                                                             // column's device ROW, -1 otherwise
    DeviceBuf dDqAlt, dDqSel;                                // the STOCK-placement twin of dDqDest
                                                             // (ndq*6) and the dummy row it keys on (ndq)
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
    DeviceBuf dVdlAltRow, dVdlAltDest;                       // nvdl each: a relocated child's home row and slot (-1: none)
    bool vdlDyn = false;
    int nQr = 0;                                             // pilot-Q relocation GROUPS (plan-A6): candidate
    std::vector<int> hQrPtr, hQrRows;                        // tables [fallback row, member rows...] per group
    DeviceBuf dQrPtr, dQrRows;                               // (rows in DEVICE row space, rowMap applied)
    std::vector<int> hQrDepBase, hQrBaseTargetRow;          // DEPENDENT pilots (a merged remote group's dependent
    std::vector<int> hDepGroupOfRow;                         // controlled bus): per group its base group (-1 = a base
    DeviceBuf dQrDepBase, dQrBaseTargetRow;                  // group) and the base row of its reactive target; per
    DeviceBuf dZiRelocDep;                                   // DEVICE row the dependent group of its owner bus
    DeviceBuf dRowPtr, dColIdx, dDiagPos;                    // device CSR pattern + guaranteed-diagonal slots
    int nHaz = 0, hazTot = 0;                                // zero-diagonal repair: hazard rows and their
    std::vector<int> hHazRow;                                // donor candidates (see hazRepairVals)
    DeviceBuf dHazRow, dHazOff, dHazDonorPtr, dHazDonor, dHazDonorHaz, dHazSign;
    double luBytes = 0.0;                                    // cached LU-probe footprint (0 = not probed yet)
};

// The native handle Java holds (GpuAcStructures) is a heap-allocated shared_ptr; each
// context copies it, so destroyStructure only drops the Java-side reference.
using SharedStructurePtr = std::shared_ptr<SharedStructure>;

// Built once per extraction (the createStructure JNI entry), see structure_build.cu.
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
                                const int* obQAltIn, const int* ziQAltIn,
                                const int* qrDepBaseIn, const int* qrBaseTargetRowIn, const int* depGroupOfRowIn);

// The device bytes one scenario's LU factors take, from a cuDSS probe analysis (cached on the structure).
double luBytesPerScenario(SharedStructure& st);

// Re-upload the per-element parameter packs in place (a tap or shunt-susceptance move): same pattern, new values.
void refreshStructureValues(SharedStructure& st, int ncb, const double* cbIn, int nob, const double* obIn,
                            int nsh, const double* shIn, int nhv, const double* hvIn,
                            int nzi, const double* ziIn);

}  // namespace olfgpu

#endif  // OLF_GPU_STRUCTURE_SHARED_STRUCTURE_CUH
