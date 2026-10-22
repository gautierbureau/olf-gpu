// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See bus_injection_fill.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "generated/closed_branch.cuh"
#include "generated/closed_branch_scatter.h"
#include "generated/open_branch.cuh"
#include "generated/open_branch_scatter.h"
#include "generated/shunt.cuh"
#include "generated/shunt_scatter.h"
#include "generated/hvdc_ac_emulation.cuh"
#include "generated/hvdc_ac_emulation_scatter.h"
#include "generated/load_model.cuh"
#include "scatter.cuh"
#include "../fixedmatrix/reactive_relocation.cuh"
#include "../fixedmatrix/relocated_scatter.cuh"
#include "scenario_overrides.cuh"
#include "bus_injection_fill.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

__global__ void fillShuntBatch(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                               const double* in, const int* ix, const int* dest,
                               const double* x, double* vals, double* F,
                               const char* disabled, const char* active,
                               int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                               const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                               ReactiveRelocationTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elementCount * nScenarios) return;
    int e = tid % elementCount;
    int s = tid / elementCount;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 2;                            // g b
    double g = p[0];
    double b = p[1];
    applyShuntOverride(nShuntAdmittanceOverrides, shuntAdmittanceOverrideScenarioElement, shuntAdmittanceOverrideDelta, s, e, g, b);
    // A voltage-controlling shunt (bcol >= 0) reads its susceptance from the per-scenario state vector; a
    // fixed shunt keeps the packed value (+ any contingency shunt-shift override).
    int bcol = ix[e * 5 + 3];
    if (bcol >= 0) b = x[(size_t) s * n + bcol];
    double v = x[(size_t) s * n + ix[e * 5 + 2]];
    double o[SHUNT_OUTPUT_COUNT];
    updateShunt(v, g, b, 0, o + 0, o + 1, o + 2, o + 3, o + 4);
    if (dynamicRelocation) {
        scatterAddRelocated(SHUNT_OUTPUT_COUNT, dest + e * SHUNT_OUTPUT_COUNT, alternateDestination + e * SHUNT_OUTPUT_COUNT, selectorRow + e * SHUNT_OUTPUT_COUNT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n,
                      relocationTablesForScenario(rt, s), e * SHUNT_OUTPUT_COUNT);
    } else {
        scatterAdd(SHUNT_OUTPUT_COUNT, dest + e * SHUNT_OUTPUT_COUNT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
}

__global__ void fillHvdcBatch(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                              const double* in, const int* ix, const int* dest,
                              const double* x, double* vals, double* F,
                              const char* disabled, const char* active,
                              const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elementCount * nScenarios) return;
    int e = tid % elementCount;
    int s = tid / elementCount;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 5;                            // p0 k lf1 lf2 r
    const int* q = ix + e * 4;
    const double* xs = x + (size_t) s * n;
    double ph1 = xs[q[2]], ph2 = xs[q[3]];
    double o[HVDC_AC_EMULATION_OUTPUT_COUNT];
    updateHvdcAcEmulation(p[0], p[1], p[2], p[3], p[4], ph1, ph2, 0, o + 0, o + 1, o + 2, o + 3, o + 4, o + 5);
    if (dynamicRelocation) {
        scatterAddRelocated(HVDC_AC_EMULATION_OUTPUT_COUNT, dest + e * HVDC_AC_EMULATION_OUTPUT_COUNT, alternateDestination + e * HVDC_AC_EMULATION_OUTPUT_COUNT, selectorRow + e * HVDC_AC_EMULATION_OUTPUT_COUNT,
                      mode + (size_t) s * n, o, vals + (size_t) s * nnz, F + (size_t) s * n, ReactiveRelocationTables{}, e * HVDC_AC_EMULATION_OUTPUT_COUNT);
    } else {
        scatterAdd(HVDC_AC_EMULATION_OUTPUT_COUNT, dest + e * HVDC_AC_EMULATION_OUTPUT_COUNT, o, vals + (size_t) s * nnz, F + (size_t) s * n);
    }
}

// Voltage-dependent load model terms (OLF AbstractLoadModelEquationTerm): per entry e,
// F[row] += coef*v^n and J[(row, vOwn)] += coef*n*v^(n-1), coef = c_n * load target (pu),
// exactly OLF's f()/dfdv() (the n = 0 constant part lives in the target vector, not here).
// Identity rows are wiped by applyRowModeBatch afterwards, so a PV bus's Q-row term vanishes
// exactly like every other Q fill (and reappears when RL flips the bus PQ). vals == nullptr
// makes it the residual-only (line-search) variant.
__global__ void fillLoadModelBatch(int elementCount, int nScenarios, int n, int nnz,
                             const double* in, const int* ix, const int* dest,
                             const double* x, double* vals, double* F,
                             const int* alternateRow, const int* alternateDestination, const int* mode) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= elementCount * nScenarios) return;
    int e = t % elementCount;
    int s = t / elementCount;
    // GENERATED updateLoadModel (codegen family "vdl", parsed from OLF's
    // AbstractLoadModelEquationTerm.f/dfdv): one packed element per exp term, the sum over a
    // load model's terms IS the per-entry atomicAdd — exactly the Java accumulation loop.
    double c = in[e * 3];
    double expn = in[e * 3 + 1];
    double target = in[e * 3 + 2];
    double v = x[(size_t) s * n + ix[e * 2 + 1]];
    double o[2];
    updateLoadModel(v, target, c, expn, 0, o + 0, o + 1);
    int row = ix[e * 2];
    int destination = dest[e];
    // A load on a relocated coupler-tree child: its term sits on the parent coupler's dummy row, and goes HOME
    // (the child's own row and its reserved slot) in a lane whose dummy row is an identity - the coupler is
    // open or out of the tree there. Mode 4 on that row is the balance of the child switched to PQ, which stays.
    if (mode != nullptr && alternateRow != nullptr && alternateRow[e] >= 0 && mode[(size_t) s * n + row] == GpuRowMode::FIXED_VARIABLE) {
        row = alternateRow[e];
        destination = alternateDestination[e];
    }
    atomicAdd(&F[(size_t) s * n + row], o[0]);
    if (vals != nullptr) {
        atomicAdd(&vals[(size_t) s * nnz + destination], o[1]);
    }
}

__global__ void fillShuntBatchF(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                                const double* in, const int* ix, const int* dest,
                                const double* x, double* F,
                                const char* disabled, const char* active,
                                int nShuntAdmittanceOverrides, const int* shuntAdmittanceOverrideScenarioElement, const double* shuntAdmittanceOverrideDelta,
                                const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation,
                               ReactiveRelocationTables rt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elementCount * nScenarios) return;
    int e = tid % elementCount;
    int s = tid / elementCount;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 2;                            // g b
    double g = p[0];
    double b = p[1];
    applyShuntOverride(nShuntAdmittanceOverrides, shuntAdmittanceOverrideScenarioElement, shuntAdmittanceOverrideDelta, s, e, g, b);
    int bcol = ix[e * 5 + 3];
    if (bcol >= 0) b = x[(size_t) s * n + bcol];             // voltage-controlling shunt: b from state
    double v = x[(size_t) s * n + ix[e * 5 + 2]];
    double o[2];
    updateShuntWithoutDerivatives(v, g, b, 0, o + 0, o + 1);
    const int pos[2] = {0, 2};
    if (dynamicRelocation) {
        scatterAddFRelocated(2, dest + e * SHUNT_OUTPUT_COUNT, alternateDestination + e * SHUNT_OUTPUT_COUNT, selectorRow + e * SHUNT_OUTPUT_COUNT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, rt, e * SHUNT_OUTPUT_COUNT);
    } else {
        scatterAddF(2, dest + e * SHUNT_OUTPUT_COUNT, pos, o, F + (size_t) s * n);
    }
}

__global__ void fillHvdcBatchF(int elementCount, int nScenarios, int n, int nnz, int nElements, int elemOffset,
                               const double* in, const int* ix, const int* dest,
                               const double* x, double* F,
                               const char* disabled, const char* active,
                               const int* alternateDestination, const int* selectorRow, const int* mode, int dynamicRelocation) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elementCount * nScenarios) return;
    int e = tid % elementCount;
    int s = tid / elementCount;
    if (!active[s] || disabled[s * nElements + elemOffset + e]) return;
    const double* p = in + e * 5;                            // p0 k lf1 lf2 r
    const int* q = ix + e * 4;
    const double* xs = x + (size_t) s * n;
    double ph1 = xs[q[2]], ph2 = xs[q[3]];
    double o[2];
    updateHvdcAcEmulationWithoutDerivatives(p[0], p[1], p[2], p[3], p[4], ph1, ph2, 0, o + 0, o + 1);
    const int pos[2] = {0, 3};
    if (dynamicRelocation) {
        scatterAddFRelocated(2, dest + e * HVDC_AC_EMULATION_OUTPUT_COUNT, alternateDestination + e * HVDC_AC_EMULATION_OUTPUT_COUNT, selectorRow + e * HVDC_AC_EMULATION_OUTPUT_COUNT,
                       pos, mode + (size_t) s * n, o, F + (size_t) s * n, ReactiveRelocationTables{}, e * HVDC_AC_EMULATION_OUTPUT_COUNT);
    } else {
        scatterAddF(2, dest + e * HVDC_AC_EMULATION_OUTPUT_COUNT, pos, o, F + (size_t) s * n);
    }
}

}  // namespace olfgpu
