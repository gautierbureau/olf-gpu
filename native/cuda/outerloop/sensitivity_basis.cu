// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See sensitivity_basis.cuh.
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/codes/gpu_controlled_flow.cuh"
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../common/codes/gpu_pi_model_column.cuh"
#include "../common/codes/gpu_row_mode.cuh"
#include "../common/codes/gpu_sensitivity_basis_kind.cuh"
#include "../common/codes/gpu_sensitivity_function.cuh"
#include "../common/codes/gpu_two_sides.cuh"
#include "../equations/scenario_overrides.cuh"
#include "../solver/batch_context.cuh"
#include "../solver/driver_steps.cuh"
#include "../solver/scenario_override_tables.cuh"
#include "flow_observation.cuh"
#include "sensitivity.cuh"
#include "sensitivity_basis.cuh"

namespace olfgpu {

__global__ void broadcastColumnBatch(int n, int nScenarios, const double* column, double* b) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nScenarios) return;
    b[t] = column[t % n];
}

__global__ void gatherSensitivityBasisBatch(int nScenarios, int n, int nGather, const int* gatherRows, int column, int nColumns,
                                            const double* dx, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nScenarios * nGather) return;
    int s = t / nGather;
    int f = t % nGather;
    out[((size_t) s * nColumns + column) * nGather + f] = dx[(size_t) s * n + gatherRows[f]];
}

__global__ void fillSensitivityBasisColumnBatch(int nScenarios, int nGather, int column, int nColumns, double value, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nScenarios * nGather) return;
    int s = t / nGather;
    int f = t % nGather;
    out[((size_t) s * nColumns + column) * nGather + f] = value;
}

__global__ void fillSensitivityValuesBatch(size_t count, double value, double* out) {
    size_t t = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    out[t] = value;
}

// SensitivityBasisRequest.Contraction.coefficient, which mirrors this: a row mode FIXED_VARIABLE is an identity (a voltage
// held, an angle reference, an islanded bus's frozen rows), any other a balance.
__device__ inline double sensitivityBasisCoefficient(int kind, int rowMode, int busPRowMode, bool slackBus) {
    switch (kind) {
        case GpuSensitivityBasisKind::P:
        case GpuSensitivityBasisKind::Q:
            return rowMode != GpuRowMode::FIXED_VARIABLE ? 1.0 : 0.0;
        case GpuSensitivityBasisKind::V:
            return rowMode == GpuRowMode::FIXED_VARIABLE && (busPRowMode != GpuRowMode::FIXED_VARIABLE || slackBus) ? 1.0 : 0.0;
        default:
            return 1.0;
    }
}

__global__ void sensitivityRowModesBatch(int nScenarios, int n, int nFactors, int nUnits, const int* unitRows, const int* unitBusPRows,
                                         const int* mode, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nScenarios * nUnits) return;
    int s = t / nUnits;
    int u = t % nUnits;
    const int* modeS = mode + (size_t) s * n;
    double* modesOut = out + (size_t) s * (nFactors + 2 * nUnits) + nFactors;
    modesOut[2 * u] = unitRows[u] >= 0 ? modeS[unitRows[u]] : nan("");
    modesOut[2 * u + 1] = unitBusPRows[u] >= 0 ? modeS[unitBusPRows[u]] : GpuRowMode::FIXED_VARIABLE;
}

__global__ void contractSensitivitiesBatch(int nScenarios, int n, int nFactors, int nUnits, int nColumns, int nGather,
                                           const int* unitRows, const int* unitBusPRows, const char* unitSlack,
                                           const int* groupColumnStart, const int* groupColumns, const int* groupColumnKind,
                                           const int* factorGroup, const int* factorFunction, const int* factorElement,
                                           const int* factorGather, const double* closedBranchParameters, const int* closedBranchIndices,
                                           ClosedBranchParameterOverrides closedBranchParameterOverrides, const int* mode,
                                           const double* x, const double* basisOut, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nScenarios * nFactors) return;
    int s = t / nFactors;
    int i = t % nFactors;
    const int* modeS = mode + (size_t) s * n;
    const double* solutions = basisOut + (size_t) s * nColumns * nGather;
    const int* gather = factorGather + 4 * i;
    double* value = out + (size_t) s * (nFactors + 2 * nUnits) + i;
    // the factor's group's solved state at the four gather positions: its basis columns combined by the slot's row modes
    double dx[4] = {0.0, 0.0, 0.0, 0.0};
    int g = factorGroup[i];
    for (int k = groupColumnStart[g]; k < groupColumnStart[g + 1]; k++) {
        int column = groupColumns[k];
        double coefficient = 1.0;
        if (column < nUnits) {
            if (unitRows[column] < 0) {                       // an equation the structure has no row for: not solved
                *value = nan("");
                return;
            }
            int busPRow = unitBusPRows[column];
            coefficient = sensitivityBasisCoefficient(groupColumnKind[k], modeS[unitRows[column]],
                    busPRow >= 0 ? modeS[busPRow] : GpuRowMode::FIXED_VARIABLE, unitSlack[column] != 0);
        }
        if (coefficient == 0.0) continue;
        for (int a = 0; a < 4; a++) {
            if (gather[a] >= 0) dx[a] += coefficient * solutions[(size_t) column * nGather + gather[a]];
        }
    }
    int function = factorFunction[i];
    if (function == GpuSensitivityFunction::BUS_VOLTAGE) {        // BusVoltageEquationTerm.calculateSensi: dv
        *value = dx[0];
        return;
    }
    int e = factorElement[i];
    const int* q = closedBranchIndices + (size_t) e * 12;
    if (q[8] >= 0 || q[9] >= 0) {                                 // a1 or r1 a variable: rows the gather does not read
        *value = nan("");
        return;
    }
    const double* p = closedBranchParameters + (size_t) e * GpuPiModelColumn::COUNT;
    double pOv[GpuPiModelColumn::COUNT];
    if (findTapOverride(closedBranchParameterOverrides, s, e, pOv)) p = pOv;
    const double* xs = x + (size_t) s * n;
    double v1 = xs[q[5]], v2 = xs[q[7]], ph1 = xs[q[4]], ph2 = xs[q[6]];
    double r1 = p[GpuPiModelColumn::R1], a1 = p[GpuPiModelColumn::A1];
    double dph1 = dx[0], dv1 = dx[1], dph2 = dx[2], dv2 = dx[3];
    double y = p[GpuPiModelColumn::Y], ksi = p[GpuPiModelColumn::KSI];
    switch (function) {                                           // the function term's calculateSensi (flow_observation.cuh)
        case GpuSensitivityFunction::ACTIVE_POWER_1:
            *value = closedBranchFlowSensitivity(GpuTwoSides::ONE, GpuControlledFlow::ACTIVE_POWER, p, v1, v2, ph1, ph2, r1, a1,
                                                 dph1, dph2, dv1, dv2, 0.0, 0.0);
            break;
        case GpuSensitivityFunction::REACTIVE_POWER_1:
            *value = closedBranchFlowSensitivity(GpuTwoSides::ONE, GpuControlledFlow::REACTIVE_POWER, p, v1, v2, ph1, ph2, r1, a1,
                                                 dph1, dph2, dv1, dv2, 0.0, 0.0);
            break;
        case GpuSensitivityFunction::ACTIVE_POWER_2:
            *value = closedBranchFlowSensitivity(GpuTwoSides::TWO, GpuControlledFlow::ACTIVE_POWER, p, v1, v2, ph1, ph2, r1, a1,
                                                 dph1, dph2, dv1, dv2, 0.0, 0.0);
            break;
        case GpuSensitivityFunction::REACTIVE_POWER_2:
            *value = closedBranchFlowSensitivity(GpuTwoSides::TWO, GpuControlledFlow::REACTIVE_POWER, p, v1, v2, ph1, ph2, r1, a1,
                                                 dph1, dph2, dv1, dv2, 0.0, 0.0);
            break;
        case GpuSensitivityFunction::CURRENT_1:
            *value = closedBranchSide1CurrentSensitivity(y, ksi, p[GpuPiModelColumn::G1], p[GpuPiModelColumn::B1], v1, ph1, r1, a1,
                                                         v2, ph2, dph1, dph2, dv1, dv2, 0.0);
            break;
        case GpuSensitivityFunction::CURRENT_2:
            *value = closedBranchSide2CurrentSensitivity(y, ksi, p[GpuPiModelColumn::G2], p[GpuPiModelColumn::B2], v1, ph1, r1, a1,
                                                         v2, ph2, dph1, dph2, dv1, dv2, 0.0);
            break;
        default:
            *value = nan("");
    }
}

namespace {

// The context's basis solutions contracted into its factors' sensitivities, with the row modes they were decided on.
void contractSensitivityBasis(BatchSolverContext* ctx, int n, int nS) {
    SensitivityBasisTables& basis = ctx->sensitivityBasis;
    double* values = ctx->dSensitivityValuesOut.as<double>();
    if (basis.nUnits > 0) {
        sensitivityRowModesBatch<<<grid(nS * basis.nUnits), BLOCK, 0, ctx->stream>>>(nS, n, basis.nFactors, basis.nUnits,
                basis.dUnitRows.as<int>(), basis.dUnitBusPRows.as<int>(), ctx->dMode.as<int>(), values);
    }
    contractSensitivitiesBatch<<<grid(nS * basis.nFactors), BLOCK, 0, ctx->stream>>>(nS, n, basis.nFactors, basis.nUnits,
            basis.nColumns, basis.nGather, basis.dUnitRows.as<int>(), basis.dUnitBusPRows.as<int>(), basis.dUnitSlack.as<char>(),
            basis.dGroupColumnStart.as<int>(), basis.dGroupColumns.as<int>(), basis.dGroupColumnKind.as<int>(),
            basis.dFactorGroup.as<int>(), basis.dFactorFunction.as<int>(), basis.dFactorElement.as<int>(), basis.dFactorGather.as<int>(),
            ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(),
            closedBranchParameterOverrides(ctx), ctx->dMode.as<int>(), ctx->dX.as<double>(), ctx->dSensitivityBasisOut.as<double>(),
            values);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void sensitivityBasisStep(BatchSolverContext* ctx, DriverPass& drv) {
    SensitivityBasisTables& basis = ctx->sensitivityBasis;
    if (basis.nColumns <= 0 || basis.nGather <= 0) return;
    const int n = drv.n;
    const int nS = drv.nS;
    const int nColumns = basis.nColumns;
    const int nGather = basis.nGather;
    const int gatherThreads = nS * nGather;
    ctx->dSensitivityBasisOut.ensure(static_cast<size_t>(nS) * nColumns * nGather * sizeof(double));
    double* out = ctx->dSensitivityBasisOut.as<double>();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const size_t valueCount = static_cast<size_t>(nS) * (basis.nFactors + 2 * basis.nUnits);
    if (basis.nFactors > 0) {
        ctx->dSensitivityValuesOut.ensure(valueCount * sizeof(double));
    }
    // Every scenario of the call ended (no refill pause: the caller asks for none with a basis), in a mode that solves
    // the whole batch on factors it keeps: otherwise no column is solved, and no sensitivity contracted.
    bool solvable = drv.nActive == 0 && (ctx->mode == GpuLinearSolveMode::UBATCH || ctx->mode == GpuLinearSolveMode::BATCHED_LU);
    if (!solvable) {
        for (int j = 0; j < nColumns; j++) {
            fillSensitivityBasisColumnBatch<<<grid(gatherThreads), BLOCK, 0, ctx->stream>>>(nS, nGather, j, nColumns, nan, out);
        }
        if (basis.nFactors > 0) {
            fillSensitivityValuesBatch<<<grid(static_cast<int>(valueCount)), BLOCK, 0, ctx->stream>>>(valueCount, nan,
                    ctx->dSensitivityValuesOut.as<double>());
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // The Jacobian of every slot at its converged state, factorized with the first column's solve, as the incremental
    // loops take their sensitivities: every slot active for the fill, the call's activity restored after.
    std::vector<char> ones(nS, 1);
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, ones.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    fillActiveBatch(ctx, drv, false);
    double* b = ctx->dB.as<double>();
    const size_t rhsBytes = static_cast<size_t>(nS) * n * sizeof(double);
    const bool participation = nColumns > basis.nUnits + basis.nWeighted;
    bool first = true;
    for (int j = 0; j < nColumns; j++) {
        if (j < basis.nUnits) {
            int row = basis.hUnitDeviceRows[j];
            if (row < 0) {                                       // an equation the structure has no row for: unsolved
                fillSensitivityBasisColumnBatch<<<grid(gatherThreads), BLOCK, 0, ctx->stream>>>(nS, nGather, j, nColumns, nan, out);
                continue;
            }
            setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, b);
        } else if (j < basis.nUnits + basis.nWeighted) {
            broadcastColumnBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS,
                    basis.dWeighted.as<double>() + static_cast<size_t>(j - basis.nUnits) * n, b);
        } else if (participation && ctx->sensitivityParticipation.device() != nullptr
                   && ctx->sensitivityParticipation.width() == static_cast<size_t>(n)) {
            // each slot's own column, on device rows (setBatchSensitivityParticipation)
            CUDA_CHECK(cudaMemcpyAsync(b, ctx->sensitivityParticipation.device(), rhsBytes, cudaMemcpyDeviceToDevice, ctx->stream));
        } else {
            CUDA_CHECK(cudaMemsetAsync(b, 0, rhsBytes, ctx->stream));
        }
        CUDA_CHECK(cudaGetLastError());
        repairZeroDiagonalRightHandSide(ctx, nS, b);
        sensitivitySolve(ctx, drv, first);
        gatherSensitivityBasisBatch<<<grid(gatherThreads), BLOCK, 0, ctx->stream>>>(nS, n, nGather, basis.dGatherRows.as<int>(), j,
                nColumns, ctx->dDx.as<double>(), out);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(ctx->dActive.p, drv.active.data(), nS * sizeof(char), cudaMemcpyHostToDevice));   // as it was
    if (basis.nFactors > 0) {
        contractSensitivityBasis(ctx, n, nS);
    }
}

}  // namespace olfgpu
