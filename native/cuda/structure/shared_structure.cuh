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
#include "../common/codes/gpu_two_sides.cuh"
#include "../lu/lu_plan.cuh"
#include "csr_pattern.cuh"

namespace olfgpu {

struct CudssCtx {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t systemMatrix = nullptr, rightHandSideMatrix = nullptr, solutionMatrix = nullptr;
    ~CudssCtx() {
        if (systemMatrix) cudssMatrixDestroy(systemMatrix);
        if (rightHandSideMatrix) cudssMatrixDestroy(rightHandSideMatrix);
        if (solutionMatrix) cudssMatrixDestroy(solutionMatrix);
        if (data && handle) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
    }
};

// One family of equation terms evaluated per element (closed branch, open branch, shunt, HVDC AC emulation): the
// element parameters and indices GpuAcData packs (ElementPack), and where each term's outputs land in F and J.
struct Family {
    int elementCount = 0, parameterStride = 0, indexStride = 0, outputCount = 0;
    DeviceBuf parameters, indices, destination;
    DeviceBuf alternateDestination, selectorRow;             // Dynamic Q-relocation (cb only); empty when disabled
    DeviceBuf relocationGroup, relocationCandidateOffset, relocationCandidateDestinations;               // GROUP relocation (plan-A6): per-output group id +
    DeviceBuf relocationDependentPilot;                                      // per output: 1 = dependent-pilot routing (dependentPilotDestination)
    bool dynamicRelocation = false;                                   // slot-pool offset; pool = one slot per candidate
    // Per element (closed branches): 1 when one of its outputs has a destination the scenario decides (a relocation group
    // or a selector row), 0 when every output keeps its frozen one - the fill then scatters it as a static family.
    DeviceBuf relocatedElement;
    DeviceBuf relocatedElements;                             // ...and those elements, listed
    int relocatedCount = 0;
    DeviceBuf phaseControlDestination, phaseControlSelector; // Continuous phase control (cb only): P-flow row at a1
    bool hasPhaseControl = false;                            // phase control destinations populated (BRANCH_TARGET_P gated emission)
};

// zeroImpedanceBranchIndices slot 10 packs two things: bit 0 = spanning-tree edge, bits 1-2 = which side is a DEAD-END
// STUB, stored as side + 1 so 0 is none (CouplerTreeReHoming.encodeChildSide; zeroImpedanceDeadEndSideOf decodes it to a
// GpuTwoSides, NONE when none). On a stub the four rows around the branch
// would all have a ZERO DIAGONAL — ZERO_V and ZERO_PHI never depend on their own dummy, and the
// stub's P/Q balances carry only -DUMMY_P / -DUMMY_Q — and cuDSS's symmetric AMD ordering
// mis-factors that chain, perturbing the zero pivot and propagating its reciprocal. The Java packer
// therefore moves the stub's P/Q balances onto the dummy rows (slots 0..3 already carry them) and
// ZERO_V / ZERO_PHI onto the stub's own v / phi rows, which is what these two derive. Slots 4/5
// stay the dummy COLUMNS, because the kernel reads the dummy values through them.
inline int isSpanningTreeEdge(const int* z) { return z[10] & 1; }
inline int zeroImpedanceDeadEndSideOf(const int* z) { return ((z[10] >> 1) & 3) - 1; }
// STOCK placement (no child): ZERO_V on the DUMMY_Q row and ZERO_PHI on the DUMMY_P row - CROSSWISE. Neither
// equation holds a dummy, so either way both rows are zero-diagonal hazards the numeric repair fills from a
// donor row carrying that dummy: a Q row for DUMMY_Q, a P row for DUMMY_P. The repaired row is then
// row_h + s*row_d, and the 2x2 principal block of (row_d, row_h') on (donor diagonal, dummy) is singular
// unless row_h has an entry of its own in the donor's diagonal column: ZERO_V (v_a - rho*v_b) has one in a
// Q row's V diagonal, ZERO_PHI (phi_a - phi_b) in a P row's PHI diagonal. The other way round (ZERO_V on the
// DUMMY_P row, repaired from a P row) the block is {[955.8, 1], [-955.8, -1]} exactly - measured on
// network.xiidm's cut trees, 35 such pairs in one dump, and the unpivoted uniform-batch factorization lost
// them (rte6515: 37 stock couplers in every scenario's matrix). Crosswise, det = -+1 on every such block.
inline int zeroVRowOf(const int* z) {
    int d = zeroImpedanceDeadEndSideOf(z);
    return d == GpuTwoSides::NONE ? z[5] : (d == GpuTwoSides::ONE ? z[6] : z[7]);  // the child's own v row
}
inline int zeroPhiRowOf(const int* z) {
    int d = zeroImpedanceDeadEndSideOf(z);
    return d == GpuTwoSides::NONE ? z[4] : (d == GpuTwoSides::ONE ? z[8] : z[9]);  // the child's own phi row
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
    int nElements = 0;                                       // nClosedBranch + nOpenBranch + nShunt + nHvdcAcEmulation (disabled-mask layout)
    int closedBranchOffset = 0, openBranchOffset = 0, shuntOffset = 0, hvdcAcEmulationOffset = 0;
    std::vector<int> rowMap;                                 // eq row -> paired variable col
    std::vector<int> rowPtr, colIdx;                         // host CSR pattern
    Family closedBranch, openBranch, shunt, hvdcAcEmulation;
    int nDistrQ = 0;                                             // DISTR_Q weighted reactive contributions
    DeviceBuf dDistrQElement, dDistrQSide, dDistrQKind, dDistrQWeight, dDistrQDestination; // nDistrQ each (dDistrQDestination is nDistrQ*6)
    DeviceBuf dRelocatedChildOwnRow;                                     // per row: the relocated child's OWN row for a
                                                             // coupler DUMMY row, -1 elsewhere (stock placement)
    DeviceBuf dDistrQDummyQRow;                                      // per contribution: a kind-3 coupler dummy
                                                             // column's device ROW, -1 otherwise
    DeviceBuf dDistrQAlternateDestination, dDistrQSelectorRow;                                // the STOCK-placement twin of dDistrQDestination
                                                             // (nDistrQ*6) and the dummy row it keys on (nDistrQ)
    int nZeroImpedanceBranch = 0;                                             // zero-impedance branches
    DeviceBuf dZeroImpedanceBranchParameters, dZeroImpedanceBranchIndices, dZeroImpedanceBranchDestination;                        // nZeroImpedanceBranch (dZeroImpedanceBranchParameters) / nZeroImpedanceBranch*11 (dZeroImpedanceBranchIndices) / nZeroImpedanceBranch*14 (dZeroImpedanceBranchDestination)
    DeviceBuf dDistrRhoRelocationGroup;                                    // per DISTR_RHO contribution: the qr relocation group
    int hasDistrRhoRelocationGroups = 0;                                            // whose candidates contain its row (-1 none) — the seq
                                                             // fill suppresses the CURRENT carrier's row in-kernel
    DeviceBuf dZeroImpedanceAlternateDestination;                                    // zero-impedance dynamic Q-routing (ob/zero-impedance channels): per element,
    DeviceBuf dZeroImpedanceSelectorRow;                                     // 4 q slots {F1,F2,J1,J2} alt dests + select rows, and
    DeviceBuf dZeroImpedanceRelocationGroup;                                 // the GROUP relocation tables (same encodings as the
    DeviceBuf dZeroImpedanceRelocationCandidateOffset;                                   // cb/sh families)
    DeviceBuf dZeroImpedanceRelocationCandidateDestinations;
    bool zeroImpedanceDynamicRelocation = false;
    int nDistrRho = 0;                                             // DISTR_RHO linear contributions
    DeviceBuf dDistrRhoColumn, dDistrRhoCoefficient, dDistrRhoDestination;                      // nDistrRho each (dDistrRhoDestination is nDistrRho*2)
    int nDistrShuntB = 0;                                             // DISTR_SHUNT_B linear contributions
    DeviceBuf dDistrShuntBColumn, dDistrShuntBCoefficient, dDistrShuntBDestination;                      // nDistrShuntB each (dDistrShuntBDestination is nDistrShuntB*2)
    int nLoadModelTerms = 0;                                            // voltage-dependent load model terms (ZIP/exp)
    DeviceBuf dLoadModelParameters, dLoadModelIndices, dLoadModelDestination;                     // nLoadModelTerms*3 (c, n, target — verbatim OLF) / nLoadModelTerms*2 (row, vCol) / nLoadModelTerms
    DeviceBuf dLoadModelAlternateRow, dLoadModelAlternateDestination;                       // nLoadModelTerms each: a relocated child's home row and slot (-1: none)
    bool loadModelRelocated = false;
    int nReactiveRelocationGroups = 0;                                             // pilot-Q relocation GROUPS (plan-A6): candidate
    std::vector<int> hReactiveRelocationGroupStart, hReactiveRelocationCandidateRows;                        // tables [fallback row, member rows...] per group
    DeviceBuf dReactiveRelocationGroupStart, dReactiveRelocationCandidateRows;                               // (rows in DEVICE row space, rowMap applied)
    std::vector<int> hReactiveRelocationDependentBaseGroup, hReactiveRelocationBaseTargetRow;          // DEPENDENT pilots (a merged remote group's dependent
    std::vector<int> hReactiveRelocationDependentGroupOfRow;                         // controlled bus): per group its base group (-1 = a base
    DeviceBuf dReactiveRelocationDependentBaseGroup, dReactiveRelocationBaseTargetRow;                  // group) and the base row of its reactive target; per
    DeviceBuf dZeroImpedanceRelocationDependentPilot;                                   // DEVICE row the dependent group of its owner bus
    DeviceBuf dRowPtr, dColIdx, dDiagPos;                    // device CSR pattern + guaranteed-diagonal slots
    int nZeroDiagonalRows = 0, nZeroDiagonalRowValues = 0;                                // zero-diagonal repair: hazard rows and their
    std::vector<int> hZeroDiagonalRow;                                // donor candidates (see repairZeroDiagonalRowsBatch)
    DeviceBuf dZeroDiagonalRow, dZeroDiagonalRowValueStart, dZeroDiagonalDonorStart, dZeroDiagonalDonor, dZeroDiagonalDonorOwnIndex, dZeroDiagonalRepairSign;
    DeviceBuf dZeroDiagonalRowValuePosition;                 // per hazard-row value (in the snapshot's order): its slot in the values
    double luBytes = 0.0;                                    // cached LU-probe footprint (0 = not probed yet)
    std::shared_ptr<LuPlan> luPlan;                          // our batched LU's plan, built by the first context using it
};

// The native handle Java holds (GpuAcStructures) is a heap-allocated shared_ptr; each
// context copies it, so destroyStructure only drops the Java-side reference.
using SharedStructurePtr = std::shared_ptr<SharedStructure>;

// Built once per extraction (the createStructure JNI entry), see structure_build.cu.
SharedStructure* buildStructure(int n,
                                int nClosedBranch, const double* closedBranchParameters, const int* closedBranchIndices,
                                int nOpenBranch, const double* openBranchParameters, const int* openBranchIndices,
                                int nShunt, const double* shuntParameters, const int* shuntIndices,
                                int nHvdcAcEmulation, const double* hvdcAcEmulationParameters, const int* hvdcAcEmulationIndices,
                                int nDistrQ, const int* distrQElement, const int* distrQSide, const double* distrQWeight,
                                const int* distrQRow, const int* distrQKind,
                                int nZeroImpedanceBranch, const double* zeroImpedanceBranchParameters, const int* zeroImpedanceBranchIndices,
                                int nDistrRho, const int* distrRhoRow, const int* distrRhoColumn, const double* distrRhoCoefficient,
                                int nDistrShuntB, const int* distrShuntBRow, const int* distrShuntBColumn, const double* distrShuntBCoefficient,
                                const int* closedBranchPhaseControlSide,
                                int nLoadModelTerms, const double* loadModelParameters, const int* loadModelIndices,
                                int nReactiveRelocationGroups, const int* reactiveRelocationGroupStart, const int* reactiveRelocationCandidateRowsIn,
                                const int* openBranchAlternateReactiveDestinationIn, const int* zeroImpedanceBranchAlternateReactiveDestinationIn,
                                const int* reactiveRelocationDependentBaseGroupIn, const int* reactiveRelocationBaseTargetRowIn, const int* reactiveRelocationDependentGroupOfRowIn);

// The device bytes one scenario's LU factors take, from a cuDSS probe analysis (cached on the structure).
double luBytesPerScenario(SharedStructure& structure);

// Our batched LU's plan of the structure's pattern in cuDSS's elimination order, built on first use and kept.
LuPlan& luPlanOf(SharedStructure& structure);

}  // namespace olfgpu

#endif  // OLF_GPU_STRUCTURE_SHARED_STRUCTURE_CUH
