// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Host check of our batched LU's transposed solve (native/cuda/lu/transposed_solve.cuh), no GPU or CUDA toolkit
// needed: tools/run_lu_host_check.sh builds and runs it.
//
// What is real and what is a port: the plans are the real ones (scalar_lu_plan.cu, supernodal_lu_plan.cu,
// minimum_degree.cu, pivot_delays.cu compiled as C++), and so are the transposed solve's bodies (transposed_solve.cuh
// with LU_TRANSPOSED_FN defined empty: the very functions the kernels call per row and per supernode). The
// factorizations and the PLAIN solves are host ports of the kernels of scalar_lu.cu and supernodal_lu.cu on one lane
// (capacity 1): the plain solve giving A x = b proves the port reads the factors the kernels write, and the transposed
// solve on the same factors must then give A^T y = c. A change of the factor layout in the kernels has to be made here
// too. The schedules the kernels launch level by level are checked for races apart: every value a row (fast engine)
// or a supernode (pivoting engine) reads in the transposed solve comes from a strictly earlier level.
//
// The systems: Jacobian-shaped random patterns on several orders (the minimum degree among them), the same with their
// rows moved so the row and column orders differ (the only systems that tell the plain solve's permutations from the
// transposed solve's), weak diagonals, dense blocks with zero diagonals (the pivoting engine's complete pivoting), the
// Java tests' coupler systems and line groups (the fast engine learning pivot delays, the transposed solve then on the
// delayed order), a hub joined to a chain (long rows and columns), and any CSR system given on the command line ("n
// nnz", then the n + 1 row pointers, the nnz columns, the nnz values) - converged AC Jacobians with their rows paired
// as the GPU structure pairs them (a bus's P equation on its angle row, its Q or voltage target on its voltage row).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "lu_pivot.cuh"
#include "minimum_degree.cuh"
#include "pivot_delays.cuh"
#include "scalar_lu_plan.cuh"
#include "supernodal_lu_plan.cuh"
#include "transposed_solve.cuh"

using namespace olfgpu;

namespace {

struct Csr {
    int n = 0;
    std::vector<int> rowPtr, colIdx;
    int nnz() const { return rowPtr[n]; }
};

int failures = 0;
bool verbose = false;                                        // per-system lines (the systems from files)
int checks = 0;

void expect(bool ok, const std::string& what) {
    checks++;
    if (!ok) {
        failures++;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

std::vector<int> inverse(const std::vector<int>& order) {
    std::vector<int> inv(order.size());
    for (size_t i = 0; i < order.size(); i++) {
        inv[order[i]] = static_cast<int>(i);
    }
    return inv;
}

// ---------------------------------------------------------------- fast engine (scalar_lu.cu, one lane)

// factorizeScalarLanes: A values to their LU entries, fill zeroed, the steps in order (each step's items are
// independent: computed one after the other here, all at once on the device), finishEntry's division and clamps.
std::vector<double> factorFast(const ScalarLuPlan& plan, const std::vector<double>& a, int& trouble) {
    std::vector<double> lu(plan.nnzLu, 0.0);
    for (int e = 0; e < plan.nnzLu; e++) {
        if (plan.luSource[e] >= 0) {
            lu[e] = a[plan.luSource[e]];
        }
    }
    trouble = 0;
    for (int s = 0; s < plan.factorSteps(); s++) {
        for (int k = plan.stepStart[s]; k < plan.stepStart[s + 1]; k++) {
            const int e = plan.stepEntry[k];
            double acc = lu[e];
            for (int c = plan.contributionStart[e]; c < plan.contributionStart[e + 1]; c++) {
                acc = fma(-lu[plan.contributionL[c]], lu[plan.contributionU[c]], acc);
            }
            const int d = plan.entryDivisor[e];
            if (d >= 0) {
                acc /= lu[d];
                if (!(std::fabs(acc) <= LU_MULTIPLIER_LIMIT)) {
                    trouble++;
                }
            } else if (d == SCALAR_ITEM_PIVOT && !(std::fabs(acc) >= LU_PIVOT_EPSILON)) {
                if (!std::isnan(acc)) {
                    acc = acc < 0.0 ? -LU_PIVOT_EPSILON : LU_PIVOT_EPSILON;
                }
                trouble++;
            }
            lu[e] = acc;
        }
    }
    return lu;
}

// solveScalar: in through the row positions, forwardLevel then backwardLevel bodies, out through the column positions.
std::vector<double> solveFast(const ScalarLuPlan& plan, const std::vector<double>& lu, const std::vector<double>& b) {
    const int n = plan.n;
    const std::vector<int> rowPosition = inverse(plan.rowOrder);
    const std::vector<int> colPosition = inverse(plan.colOrder);
    std::vector<double> z(n);
    for (int r = 0; r < n; r++) {
        z[rowPosition[r]] = b[r];
    }
    for (int l = 0; l < plan.forwardLevels(); l++) {
        for (int k = plan.forwardLevelStart[l]; k < plan.forwardLevelStart[l + 1]; k++) {
            const int i = plan.forwardRow[k];
            double y = z[i];
            for (int q = plan.luRowPtr[i]; q < plan.luDiag[i]; q++) {
                y = fma(-lu[q], z[plan.luCol[q]], y);
            }
            z[i] = y;
        }
    }
    for (int l = 0; l < plan.backwardLevels(); l++) {
        for (int k = plan.backwardLevelStart[l]; k < plan.backwardLevelStart[l + 1]; k++) {
            const int i = plan.backwardRow[k];
            double x = z[i];
            for (int q = plan.luDiag[i] + 1; q < plan.luRowPtr[i + 1]; q++) {
                x = fma(-lu[q], z[plan.luCol[q]], x);
            }
            z[i] = x / lu[plan.luDiag[i]];
        }
    }
    std::vector<double> x(n);
    for (int c = 0; c < n; c++) {
        x[c] = z[colPosition[c]];
    }
    return x;
}

// solveScalarTransposed: in through the column positions, the transposed bodies on the reversed levels, out through
// the row positions - the very functions the kernels call.
std::vector<double> solveFastTransposed(const ScalarLuPlan& plan, const ScalarLuColumns& cols,
                                        const std::vector<double>& lu, const std::vector<double>& b) {
    const int n = plan.n;
    const std::vector<int> rowPosition = inverse(plan.rowOrder);
    const std::vector<int> colPosition = inverse(plan.colOrder);
    std::vector<double> z(n);
    for (int c = 0; c < n; c++) {
        z[colPosition[c]] = b[c];
    }
    for (int l = plan.backwardLevels() - 1; l >= 0; l--) {
        for (int k = plan.backwardLevelStart[l]; k < plan.backwardLevelStart[l + 1]; k++) {
            upperTransposedRow(plan.backwardRow[k], 0, 1, cols.columnStart.data(), cols.columnSplit.data(),
                               cols.columnRow.data(), cols.columnEntry.data(), plan.luDiag.data(), lu.data(), z.data());
        }
    }
    for (int l = plan.forwardLevels() - 1; l >= 0; l--) {
        for (int k = plan.forwardLevelStart[l]; k < plan.forwardLevelStart[l + 1]; k++) {
            lowerTransposedRow(plan.forwardRow[k], 0, 1, cols.columnStart.data(), cols.columnSplit.data(),
                               cols.columnRow.data(), cols.columnEntry.data(), lu.data(), z.data());
        }
    }
    std::vector<double> x(n);
    for (int r = 0; r < n; r++) {
        x[r] = z[rowPosition[r]];
    }
    return x;
}

// The transposed schedules are race-free: a row reads only rows of earlier levels (strictly), and the column lists
// hold exactly the off-diagonal LU entries, U above, L below, rows ascending.
void checkFastSchedule(const ScalarLuPlan& plan, const ScalarLuColumns& cols, const std::string& name) {
    const int n = plan.n;
    std::vector<int> upperStep(n, -1);                       // processing step of each row, U^T forward
    std::vector<int> lowerStep(n, -1);
    int step = 0;
    for (int l = plan.backwardLevels() - 1; l >= 0; l--, step++) {
        for (int k = plan.backwardLevelStart[l]; k < plan.backwardLevelStart[l + 1]; k++) {
            upperStep[plan.backwardRow[k]] = step;
        }
    }
    step = 0;
    for (int l = plan.forwardLevels() - 1; l >= 0; l--, step++) {
        for (int k = plan.forwardLevelStart[l]; k < plan.forwardLevelStart[l + 1]; k++) {
            lowerStep[plan.forwardRow[k]] = step;
        }
    }
    bool ok = true;
    long long listed = 0;
    for (int j = 0; j < n; j++) {
        ok = ok && upperStep[j] >= 0 && lowerStep[j] >= 0;
        for (int p = cols.columnStart[j]; p < cols.columnStart[j + 1]; p++) {
            const int k = cols.columnRow[p];
            const int e = cols.columnEntry[p];
            const bool above = p < cols.columnSplit[j];
            ok = ok && plan.luCol[e] == j && (above ? k < j : k > j);
            ok = ok && e >= plan.luRowPtr[k] && e < plan.luRowPtr[k + 1];
            ok = ok && (p == cols.columnStart[j] || p == cols.columnSplit[j] || cols.columnRow[p - 1] < k);
            ok = ok && (above ? upperStep[k] < upperStep[j] : lowerStep[k] < lowerStep[j]);
            listed++;
        }
    }
    expect(ok && listed == plan.nnzLu - n, name + ": fast transposed schedule reads only earlier levels");
}

// ---------------------------------------------------------------- pivoting engine (supernodal_lu.cu, one lane)

struct Supernodal {
    std::vector<double> lu;
    std::vector<unsigned char> rowPermutation, columnPermutation;
    int clamps = 0;
};

// factorBlocks for one supernode (capacity 1): singleton-row first, otherwise complete pivoting; clamps.
void factorBlock(const SupernodalLuPlan& plan, int s, Supernodal& f) {
    constexpr int B = LU_MAX_SUPERNODE;
    const int first = plan.supernodeStart[s];
    const int b = plan.supernodeStart[s + 1] - first;
    double* block = f.lu.data() + plan.diagonalBase[s];
    double d[B * B];
    unsigned char rows[B];
    unsigned char columns[B];
    for (int q = 0; q < b * b; q++) {
        d[q] = block[q];
    }
    for (int q = 0; q < b; q++) {
        rows[q] = static_cast<unsigned char>(q);
        columns[q] = static_cast<unsigned char>(q);
    }
    bool panelEmpty[B];
    const int count = plan.rowStart[s + 1] - plan.rowStart[s];
    const double* upper = f.lu.data() + plan.upperBase[s];
    for (int q = 0; q < b; q++) {
        bool empty = true;
        for (int column = 0; column < count && empty; column++) {
            empty = upper[column * b + q] == 0.0;
        }
        panelEmpty[q] = empty;
    }
    for (int k = 0; k < b; k++) {
        int pr = -1;
        int pc = -1;
        for (int r = k; r < b && pr < 0; r++) {
            if (!panelEmpty[rows[r]]) {
                continue;
            }
            int nonZeros = 0;
            int column = -1;
            for (int c = k; c < b; c++) {
                if (d[r * b + c] != 0.0) {
                    nonZeros++;
                    column = c;
                }
            }
            if (nonZeros == 1) {
                pr = r;
                pc = column;
            }
        }
        if (pr < 0) {
            pr = k;
            pc = k;
            double best = -1.0;
            for (int r = k; r < b; r++) {
                for (int c = k; c < b; c++) {
                    const double m = std::fabs(d[r * b + c]);
                    if (m > best) {
                        best = m;
                        pr = r;
                        pc = c;
                    }
                }
            }
        }
        if (pr != k) {
            for (int c = 0; c < b; c++) {
                std::swap(d[k * b + c], d[pr * b + c]);
            }
            std::swap(rows[k], rows[pr]);
        }
        if (pc != k) {
            for (int r = 0; r < b; r++) {
                std::swap(d[r * b + k], d[r * b + pc]);
            }
            std::swap(columns[k], columns[pc]);
        }
        double pivot = d[k * b + k];
        if (!(std::fabs(pivot) >= LU_PIVOT_EPSILON)) {
            if (!std::isnan(pivot)) {
                pivot = pivot < 0.0 ? -LU_PIVOT_EPSILON : LU_PIVOT_EPSILON;
                d[k * b + k] = pivot;
            }
            f.clamps++;
        }
        for (int r = k + 1; r < b; r++) {
            const double multiplier = d[r * b + k] / pivot;
            d[r * b + k] = multiplier;
            for (int c = k + 1; c < b; c++) {
                d[r * b + c] = fma(-multiplier, d[k * b + c], d[r * b + c]);
            }
        }
    }
    for (int q = 0; q < b * b; q++) {
        block[q] = d[q];
    }
    for (int q = 0; q < b; q++) {
        f.rowPermutation[first + q] = rows[q];
        f.columnPermutation[first + q] = columns[q];
    }
}

// applyPanels for one (supernode, panel row).
void applyPanel(const SupernodalLuPlan& plan, int s, int item, Supernodal& f) {
    constexpr int B = LU_MAX_SUPERNODE;
    const int first = plan.supernodeStart[s];
    const int b = plan.supernodeStart[s + 1] - first;
    const int count = plan.rowStart[s + 1] - plan.rowStart[s];
    const double* d = f.lu.data() + plan.diagonalBase[s];
    double v[B];
    double w[B];
    if (item < count) {
        double* column = f.lu.data() + plan.upperBase[s] + item * b;
        for (int k = 0; k < b; k++) {
            v[k] = column[k];
        }
        for (int k = 0; k < b; k++) {
            double x = v[f.rowPermutation[first + k]];
            for (int q = 0; q < k; q++) {
                x = fma(-d[k * b + q], w[q], x);
            }
            w[k] = x;
        }
        for (int k = 0; k < b; k++) {
            column[k] = w[k];
        }
    } else {
        double* row = f.lu.data() + plan.lowerBase[s] + (item - count) * b;
        for (int c = 0; c < b; c++) {
            v[c] = row[c];
        }
        for (int c = 0; c < b; c++) {
            double x = v[f.columnPermutation[first + c]];
            for (int q = 0; q < c; q++) {
                x = fma(-w[q], d[q * b + c], x);
            }
            w[c] = x / d[c * b + c];
        }
        for (int c = 0; c < b; c++) {
            row[c] = w[c];
        }
    }
}

// factorizeSupernodalLanes: A values to their stored entries (fill zero), then per level the gathers, the blocks, the
// panels.
Supernodal factorSupernodal(const SupernodalLuPlan& plan, const std::vector<double>& a) {
    Supernodal f;
    f.lu.assign(plan.nStored, 0.0);
    f.rowPermutation.assign(plan.n, 0);
    f.columnPermutation.assign(plan.n, 0);
    for (int p = 0; p < plan.nnzA; p++) {
        f.lu[plan.storedOfA[p]] = a[p];
    }
    for (int l = 0; l < plan.levels(); l++) {
        for (int k = plan.levelEntryStart[l]; k < plan.levelEntryStart[l + 1]; k++) {
            const int e = plan.levelEntry[k];
            if (plan.runStart[e] == plan.runStart[e + 1]) {
                continue;
            }
            double acc = f.lu[e];
            for (int r = plan.runStart[e]; r < plan.runStart[e + 1]; r++) {
                for (int q = 0; q < plan.runLength[r]; q++) {
                    acc = fma(-f.lu[plan.runLower[r] + q], f.lu[plan.runUpper[r] + q], acc);
                }
            }
            f.lu[e] = acc;
        }
        for (int q = plan.levelStart[l]; q < plan.levelStart[l + 1]; q++) {
            factorBlock(plan, plan.levelSupernode[q], f);
        }
        for (int q = plan.levelPanelStart[l]; q < plan.levelPanelStart[l + 1]; q++) {
            applyPanel(plan, plan.panelSupernode[q], plan.panelRow[q], f);
        }
    }
    return f;
}

// solveSupernodal: forwardLevel / backwardLevel bodies (capacity 1).
std::vector<double> solveSupernodal(const SupernodalLuPlan& plan, const Supernodal& f, const std::vector<double>& b) {
    constexpr int B = LU_MAX_SUPERNODE;
    const int n = plan.n;
    const std::vector<int> rowPosition = inverse(plan.rowOrder);
    const std::vector<int> colPosition = inverse(plan.colOrder);
    std::vector<double> z(n);
    for (int r = 0; r < n; r++) {
        z[rowPosition[r]] = b[r];
    }
    for (int l = 0; l < plan.levels(); l++) {
        for (int t = plan.levelStart[l]; t < plan.levelStart[l + 1]; t++) {
            const int s = plan.levelSupernode[t];
            const int first = plan.supernodeStart[s];
            const int bs = plan.supernodeStart[s + 1] - first;
            double v[B];
            double w[B];
            for (int r = 0; r < bs; r++) {
                const int i = first + r;
                double x = z[i];
                for (int run = plan.forwardRunStart[i]; run < plan.forwardRunStart[i + 1]; run++) {
                    for (int k = 0; k < plan.forwardRunLength[run]; k++) {
                        x = fma(-f.lu[plan.forwardRunLower[run] + k], z[plan.forwardRunY[run] + k], x);
                    }
                }
                v[r] = x;
            }
            const double* d = f.lu.data() + plan.diagonalBase[s];
            for (int k = 0; k < bs; k++) {
                double x = v[f.rowPermutation[first + k]];
                for (int q = 0; q < k; q++) {
                    x = fma(-d[k * bs + q], w[q], x);
                }
                w[k] = x;
            }
            for (int k = 0; k < bs; k++) {
                z[first + k] = w[k];
            }
        }
    }
    for (int l = plan.levels() - 1; l >= 0; l--) {
        for (int t = plan.levelStart[l]; t < plan.levelStart[l + 1]; t++) {
            const int s = plan.levelSupernode[t];
            const int first = plan.supernodeStart[s];
            const int bs = plan.supernodeStart[s + 1] - first;
            double v[B];
            for (int k = 0; k < bs; k++) {
                v[k] = z[first + k];
            }
            const double* upper = f.lu.data() + plan.upperBase[s];
            for (int q = plan.rowStart[s]; q < plan.rowStart[s + 1]; q++) {
                const double x = z[plan.rows[q]];
                const double* column = upper + (q - plan.rowStart[s]) * bs;
                for (int k = 0; k < bs; k++) {
                    v[k] = fma(-column[k], x, v[k]);
                }
            }
            const double* d = f.lu.data() + plan.diagonalBase[s];
            for (int k = bs - 1; k >= 0; k--) {
                double x = v[k];
                for (int q = k + 1; q < bs; q++) {
                    x = fma(-d[k * bs + q], v[q], x);
                }
                v[k] = x / d[k * bs + k];
            }
            for (int c = 0; c < bs; c++) {
                z[first + f.columnPermutation[first + c]] = v[c];
            }
        }
    }
    std::vector<double> x(n);
    for (int c = 0; c < n; c++) {
        x[c] = z[colPosition[c]];
    }
    return x;
}

// solveSupernodalTransposed: the transposed bodies, up pass from the leaves, down pass from the root.
std::vector<double> solveSupernodalTransposed(const SupernodalLuPlan& plan, const Supernodal& f,
                                              const std::vector<double>& b) {
    const int n = plan.n;
    const std::vector<int> rowPosition = inverse(plan.rowOrder);
    const std::vector<int> colPosition = inverse(plan.colOrder);
    std::vector<double> z(n);
    for (int c = 0; c < n; c++) {
        z[colPosition[c]] = b[c];
    }
    for (int l = 0; l < plan.levels(); l++) {
        for (int t = plan.levelStart[l]; t < plan.levelStart[l + 1]; t++) {
            supernodeUpTransposed(plan.levelSupernode[t], 0, 1, plan.supernodeStart.data(), plan.diagonalBase.data(),
                                  plan.forwardRunStart.data(), plan.forwardRunUpper.data(), plan.forwardRunY.data(),
                                  plan.forwardRunLength.data(), f.columnPermutation.data(), f.lu.data(), z.data());
        }
    }
    for (int l = plan.levels() - 1; l >= 0; l--) {
        for (int t = plan.levelStart[l]; t < plan.levelStart[l + 1]; t++) {
            supernodeDownTransposed(plan.levelSupernode[t], 0, 1, plan.supernodeStart.data(), plan.rowStart.data(),
                                    plan.rows.data(), plan.diagonalBase.data(), plan.lowerBase.data(),
                                    f.rowPermutation.data(), f.lu.data(), z.data());
        }
    }
    std::vector<double> x(n);
    for (int r = 0; r < n; r++) {
        x[r] = z[rowPosition[r]];
    }
    return x;
}

// The up pass reads the slots of descendants only (strictly lower levels), the down pass the rows R_s of ancestors only
// (strictly higher levels); forwardRunUpper is the U panel column at the forward run's L panel row.
void checkSupernodalSchedule(const SupernodalLuPlan& plan, const std::string& name) {
    std::vector<int> levelOf(plan.nSupernodes, -1);
    for (int l = 0; l < plan.levels(); l++) {
        for (int t = plan.levelStart[l]; t < plan.levelStart[l + 1]; t++) {
            levelOf[plan.levelSupernode[t]] = l;
        }
    }
    std::vector<int> supernodeOf(plan.n);
    for (int s = 0; s < plan.nSupernodes; s++) {
        for (int j = plan.supernodeStart[s]; j < plan.supernodeStart[s + 1]; j++) {
            supernodeOf[j] = s;
        }
    }
    bool ok = static_cast<int>(plan.forwardRunUpper.size()) == plan.forwardRunStart[plan.n];
    for (int s = 0; s < plan.nSupernodes && ok; s++) {
        for (int j = plan.supernodeStart[s]; j < plan.supernodeStart[s + 1]; j++) {
            for (int run = plan.forwardRunStart[j]; run < plan.forwardRunStart[j + 1]; run++) {
                const int t = supernodeOf[plan.forwardRunY[run]];
                ok = ok && plan.forwardRunY[run] == plan.supernodeStart[t] && levelOf[t] < levelOf[s];
                ok = ok && plan.forwardRunLength[run] == plan.supernodeSize(t);
                const int lowerOffset = plan.forwardRunLower[run] - plan.lowerBase[t];
                const int upperOffset = plan.forwardRunUpper[run] - plan.upperBase[t];
                ok = ok && lowerOffset == upperOffset && lowerOffset % plan.supernodeSize(t) == 0;
                ok = ok && plan.rows[plan.rowStart[t] + lowerOffset / plan.supernodeSize(t)] == j;
            }
        }
        for (int q = plan.rowStart[s]; q < plan.rowStart[s + 1]; q++) {
            ok = ok && levelOf[supernodeOf[plan.rows[q]]] > levelOf[s];
        }
    }
    expect(ok, name + ": pivoting transposed schedule reads only finished supernodes");
}

// ---------------------------------------------------------------- references

double normInf(const std::vector<double>& v) {
    double m = 0;
    for (double x : v) {
        m = std::max(m, std::fabs(x));
    }
    return m;
}

// ||A x - b||_inf / (||A||_inf ||x||_inf + ||b||_inf), A or A^T
double backwardError(const Csr& a, const std::vector<double>& vals, const std::vector<double>& x,
                     const std::vector<double>& b, bool transposed) {
    std::vector<double> r(b);
    std::vector<double> rowSum(a.n, 0.0);
    for (int i = 0; i < a.n; i++) {
        for (int p = a.rowPtr[i]; p < a.rowPtr[i + 1]; p++) {
            const int j = a.colIdx[p];
            if (transposed) {
                r[j] -= vals[p] * x[i];
                rowSum[j] += std::fabs(vals[p]);
            } else {
                r[i] -= vals[p] * x[j];
                rowSum[i] += std::fabs(vals[p]);
            }
        }
    }
    return normInf(r) / (normInf(rowSum) * normInf(x) + normInf(b));
}

// Dense partial-pivoting solve of A (transposed: A^T), the reference the Java tests use.
std::vector<double> denseSolve(const Csr& a, const std::vector<double>& vals, const std::vector<double>& b,
                               bool transposed) {
    const int n = a.n;
    std::vector<std::vector<double>> m(n, std::vector<double>(n + 1, 0.0));
    for (int i = 0; i < n; i++) {
        for (int p = a.rowPtr[i]; p < a.rowPtr[i + 1]; p++) {
            if (transposed) {
                m[a.colIdx[p]][i] = vals[p];
            } else {
                m[i][a.colIdx[p]] = vals[p];
            }
        }
        m[i][n] = b[i];
    }
    for (int col = 0; col < n; col++) {
        int pivot = col;
        for (int r = col + 1; r < n; r++) {
            if (std::fabs(m[r][col]) > std::fabs(m[pivot][col])) {
                pivot = r;
            }
        }
        std::swap(m[col], m[pivot]);
        for (int r = col + 1; r < n; r++) {
            const double f = m[r][col] / m[col][col];
            if (f != 0.0) {
                for (int c = col; c <= n; c++) {
                    m[r][c] -= f * m[col][c];
                }
            }
        }
    }
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        double s = m[i][n];
        for (int j = i + 1; j < n; j++) {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    return x;
}

double relativeDifference(const std::vector<double>& x, const std::vector<double>& y) {
    double m = 0;
    for (size_t i = 0; i < x.size(); i++) {
        m = std::max(m, std::fabs(x[i] - y[i]));
    }
    return m / std::max(normInf(y), 1e-300);
}

// ---------------------------------------------------------------- systems

// The Java tests' jacobianLikePattern: random couplings, mostly two-way, every fifth row reduced to its diagonal.
Csr jacobianLike(int n, double density, unsigned seed) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::vector<char>> pattern(n, std::vector<char>(n, 0));
    for (int i = 0; i < n; i++) {
        pattern[i][i] = 1;
        for (int j = 0; j < n; j++) {
            if (j != i && u(random) < density) {
                pattern[i][j] = 1;
                pattern[j][i] = u(random) < 0.8;
            }
        }
    }
    for (int i = 0; i < n; i += 5) {
        std::fill(pattern[i].begin(), pattern[i].end(), 0);
        pattern[i][i] = 1;
    }
    Csr a;
    a.n = n;
    a.rowPtr.assign(n + 1, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (pattern[i][j]) {
                a.colIdx.push_back(j);
            }
        }
        a.rowPtr[i + 1] = static_cast<int>(a.colIdx.size());
    }
    return a;
}

Csr denseCsr(int n) {
    Csr a;
    a.n = n;
    a.rowPtr.assign(n + 1, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a.colIdx.push_back(j);
        }
        a.rowPtr[i + 1] = static_cast<int>(a.colIdx.size());
    }
    return a;
}

// Values: a strong diagonal and signed couplings (dominant), or couplings as large as the diagonal (weak): every
// entry different, so A and A^T differ.
std::vector<double> values(const Csr& a, unsigned seed, double diagonal) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> v(a.nnz());
    for (int i = 0; i < a.n; i++) {
        for (int p = a.rowPtr[i]; p < a.rowPtr[i + 1]; p++) {
            v[p] = a.colIdx[p] == i ? diagonal * (1.0 + 0.25 * (u(random) + 1.0)) : u(random);
        }
    }
    return v;
}

std::vector<int> identityOrder(int n) {
    std::vector<int> o(n);
    for (int i = 0; i < n; i++) {
        o[i] = i;
    }
    return o;
}

std::vector<int> shuffledOrder(int n, unsigned seed) {
    std::vector<int> o = identityOrder(n);
    std::mt19937 random(seed);
    std::shuffle(o.begin(), o.end(), random);
    return o;
}

// A CSR system read from a file: "n nnz", then n + 1 row pointers, nnz columns, nnz values (whitespace separated).
bool readSystem(const std::string& path, Csr& a, std::vector<double>& vals) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    int nnz = 0;
    in >> a.n >> nnz;
    a.rowPtr.resize(a.n + 1);
    a.colIdx.resize(nnz);
    vals.resize(nnz);
    for (int& v : a.rowPtr) {
        in >> v;
    }
    for (int& v : a.colIdx) {
        in >> v;
    }
    for (double& v : vals) {
        in >> v;
    }
    return static_cast<bool>(in);
}

// ---------------------------------------------------------------- the check of one system

struct Worst {
    double fastForward = 0, fastTransposed = 0, pivotingForward = 0, pivotingTransposed = 0;
    double fastVsDense = 0, pivotingVsDense = 0;
    int fastSystems = 0, fastTroubled = 0, pivotingSystems = 0, pivotingClamped = 0, unsymmetricOrders = 0;
} worst;

struct Rhs {
    std::vector<double> b, c;
};

Rhs rightHandSides(int n, unsigned seed) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Rhs r{std::vector<double>(n), std::vector<double>(n)};
    for (int i = 0; i < n; i++) {
        r.b[i] = u(random);
        r.c[i] = u(random);
    }
    return r;
}

// The fast engine on plan: the schedule, then - when it trusts its factorization, as BatchedLu does - both solves.
// Returns its trouble count.
int checkFast(const std::string& name, const Csr& a, const std::vector<double>& vals, const ScalarLuPlan& plan,
              unsigned seed) {
    const Rhs r = rightHandSides(a.n, seed);
    const ScalarLuColumns columns = buildScalarLuColumns(plan);
    checkFastSchedule(plan, columns, name);
    int trouble = 0;
    const std::vector<double> lu = factorFast(plan, vals, trouble);
    if (trouble > 0) {
        worst.fastTroubled++;
        return trouble;
    }
    worst.fastSystems++;
    worst.unsymmetricOrders += plan.rowOrder != plan.colOrder ? 1 : 0;
    const double forward = backwardError(a, vals, solveFast(plan, lu, r.b), r.b, false);
    const std::vector<double> y = solveFastTransposed(plan, columns, lu, r.c);
    const double transposed = backwardError(a, vals, y, r.c, true);
    worst.fastForward = std::max(worst.fastForward, forward);
    worst.fastTransposed = std::max(worst.fastTransposed, transposed);
    // no pivoting: a weak diagonal grows the factors (the plain solve's backward error goes to ~1e-13 and beyond)
    if (verbose) {
        std::printf("  %-48s fast engine:     plain %.1e  transposed %.1e  (%d LU entries, %d + %d levels)\n",
                    name.c_str(), forward, transposed, plan.nnzLu, plan.forwardLevels(), plan.backwardLevels());
    }
    expect(forward < 1e-11, name + ": fast plain solve, backward error " + std::to_string(forward));
    expect(transposed < 1e-11, name + ": fast transposed solve, backward error " + std::to_string(transposed));
    if (a.n <= 400) {
        worst.fastVsDense = std::max(worst.fastVsDense, relativeDifference(y, denseSolve(a, vals, r.c, true)));
    }
    return 0;
}

void checkPivoting(const std::string& name, const Csr& a, const std::vector<double>& vals,
                   const SupernodalLuPlan& plan, unsigned seed) {
    const Rhs r = rightHandSides(a.n, seed);
    checkSupernodalSchedule(plan, name);
    const Supernodal f = factorSupernodal(plan, vals);
    worst.pivotingSystems++;
    if (f.clamps > 0) {
        worst.pivotingClamped++;
        std::printf("  %s: %d pivots clamped (a block singular in this order): no accuracy expected\n", name.c_str(),
                    f.clamps);
        return;
    }
    worst.unsymmetricOrders += plan.rowOrder != plan.colOrder ? 1 : 0;
    const double forward = backwardError(a, vals, solveSupernodal(plan, f, r.b), r.b, false);
    const std::vector<double> y = solveSupernodalTransposed(plan, f, r.c);
    const double transposed = backwardError(a, vals, y, r.c, true);
    worst.pivotingForward = std::max(worst.pivotingForward, forward);
    worst.pivotingTransposed = std::max(worst.pivotingTransposed, transposed);
    if (verbose) {
        std::printf("  %-48s pivoting engine: plain %.1e  transposed %.1e  (%d supernodes, %d levels)\n", name.c_str(),
                    forward, transposed, plan.nSupernodes, plan.levels());
    }
    expect(forward < 1e-12, name + ": pivoting plain solve, backward error " + std::to_string(forward));
    expect(transposed < 1e-12, name + ": pivoting transposed solve, backward error " + std::to_string(transposed));
    if (a.n <= 400) {
        worst.pivotingVsDense = std::max(worst.pivotingVsDense, relativeDifference(y, denseSolve(a, vals, r.c, true)));
    }
}

void checkSystem(const std::string& name, const Csr& a, const std::vector<double>& vals, const std::vector<int>& rowOrder,
                 const std::vector<int>& colOrder, unsigned seed, bool reportTrouble = false) {
    const ScalarLuPlan fast = buildScalarLuPlan(a.n, a.rowPtr.data(), a.colIdx.data(), rowOrder.data(), colOrder.data());
    const int trouble = checkFast(name, a, vals, fast, seed);
    if (trouble > 0 && reportTrouble) {
        std::printf("  %s: the fast engine does not trust it (%d troubles): pivoting engine only\n", name.c_str(),
                    trouble);
    }
    checkPivoting(name, a, vals, buildSupernodalLuPlan(a.n, a.rowPtr.data(), a.colIdx.data(), rowOrder.data(),
                                                       colOrder.data()), seed);
}

// The fast engine as BatchedLu runs it on a structure whose order cancels pivots: factorize, learn the delays from the
// untrusted factorization, rebuild the plan with them, until it trusts the lane or learns nothing more. The transposed
// solve then runs on the delayed order.
void checkLearnedDelays(const std::string& name, const Csr& a, const std::vector<double>& vals,
                        const std::vector<int>& order, unsigned seed) {
    PivotDelays delays;
    std::shared_ptr<const FastLuPlan> plan = buildFastLuPlan(a.n, a.rowPtr.data(), a.colIdx.data(), order.data(),
                                                             order.data(), delays);
    int rounds = 0;
    for (; rounds < PIVOT_DELAY_MAX_MOVES * 4; rounds++) {
        int trouble = 0;
        const std::vector<double> lu = factorFast(plan->scalar, vals, trouble);
        if (trouble == 0 || !learnPivotDelays(delays, *plan, a.rowPtr.data(), a.colIdx.data(), lu.data(), vals.data())) {
            break;
        }
        plan = buildFastLuPlan(a.n, a.rowPtr.data(), a.colIdx.data(), order.data(), order.data(), delays);
    }
    const int trouble = checkFast(name + " (" + std::to_string(delays.delayed) + " delays)", a, vals, plan->scalar,
                                  seed);
    std::printf("  %s: %d pivot delays learned in %d rounds, the fast engine %s\n", name.c_str(), delays.delayed,
                rounds, trouble == 0 ? "trusts the delayed order" : "still does not trust it");
    checkPivoting(name, a, vals, buildSupernodalLuPlan(a.n, a.rowPtr.data(), a.colIdx.data(), order.data(),
                                                       order.data()), seed);
}

// Row i of a0 moved to row pi[i]: with rowOrder = pi o order and colOrder = order the plans eliminate a0(order, order)
// - a well-conditioned elimination on orders that differ, so the plain and transposed solves' permutations are
// told apart.
Csr permuteRows(const Csr& a0, const std::vector<double>& v0, const std::vector<int>& pi, std::vector<double>& v) {
    const std::vector<int> from = inverse(pi);               // row of A -> row of a0
    Csr a;
    a.n = a0.n;
    a.rowPtr.assign(a.n + 1, 0);
    v.clear();
    for (int r = 0; r < a.n; r++) {
        const int i = from[r];
        for (int p = a0.rowPtr[i]; p < a0.rowPtr[i + 1]; p++) {
            a.colIdx.push_back(a0.colIdx[p]);
            v.push_back(v0[p]);
        }
        a.rowPtr[r + 1] = static_cast<int>(a.colIdx.size());
    }
    return a;
}

// The Java tests' couplerSystem: a strong diagonal, then coupler rows (+x and -x on two later columns, no diagonal)
// repaired by adding a donor row carrying the coupler's column: eliminating the donor first cancels the coupler's pivot.
Csr couplerSystem(unsigned seed, int n, int couplers, double density, std::vector<double>& v) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::vector<std::pair<int, double>>> rows(n);
    auto put = [](std::vector<std::pair<int, double>>& row, int j, double x, bool add) {
        for (auto& e : row) {
            if (e.first == j) {
                e.second = add ? e.second + x : x;
                return;
            }
        }
        row.emplace_back(j, x);
    };
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            if (j != i && u(random) < density) {
                const double x = u(random) * 2 - 1;
                rows[i].emplace_back(j, x);
                sum += std::fabs(x);
            }
        }
        rows[i].emplace_back(i, (sum + 1) * (0.5 + u(random)));
    }
    for (int k = 0; k < couplers; k++) {
        const int r = n / 4 + static_cast<int>(u(random) * (n / 2));
        const int d = static_cast<int>(u(random) * r);
        const int c1 = r + 1 + static_cast<int>(u(random) * (n - r - 1));
        const int c2 = r + 1 + static_cast<int>(u(random) * (n - r - 1));
        if (c1 == c2 || d == r) {
            continue;
        }
        for (int i = 0; i < n; i++) {
            if (i != d && i != r) {
                rows[i].erase(std::remove_if(rows[i].begin(), rows[i].end(),
                                             [r](const std::pair<int, double>& e) { return e.first == r; }),
                              rows[i].end());
            }
        }
        put(rows[d], r, 0.5 + u(random), false);
        const double x = 1 + u(random);
        std::vector<std::pair<int, double>> repaired;
        put(repaired, c1, x, false);
        put(repaired, c2, -x, false);
        for (const auto& e : rows[d]) {
            put(repaired, e.first, e.second, true);
        }
        rows[r] = repaired;
    }
    Csr a;
    a.n = n;
    a.rowPtr.assign(n + 1, 0);
    v.clear();
    for (int i = 0; i < n; i++) {
        put(rows[i], i, 0.0, true);                          // the diagonal in the pattern, zero if absent
        std::sort(rows[i].begin(), rows[i].end());
        for (const auto& e : rows[i]) {
            a.colIdx.push_back(e.first);
            v.push_back(e.second);
        }
        a.rowPtr[i + 1] = static_cast<int>(a.colIdx.size());
    }
    return a;
}

// The Java tests' hubAndChain: a chain with a hub joined to every row, the hub last or first - long L and U rows, and
// long columns: the transposed solve's long runs.
Csr hubAndChain(int n, bool hubLast) {
    std::vector<std::vector<char>> pattern(n, std::vector<char>(n, 0));
    const int hub = hubLast ? n - 1 : 0;
    for (int i = 0; i < n; i++) {
        pattern[i][i] = 1;
        pattern[i][hub] = 1;
        pattern[hub][i] = 1;
        if (i + 1 < n) {
            pattern[i][i + 1] = 1;
            pattern[i + 1][i] = 1;
        }
    }
    Csr a;
    a.n = n;
    a.rowPtr.assign(n + 1, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (pattern[i][j]) {
                a.colIdx.push_back(j);
            }
        }
        a.rowPtr[i + 1] = static_cast<int>(a.colIdx.size());
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    // 1. Jacobian-like patterns, dominant values: symmetric orders (the fast engine factorizes them), unsymmetric ones
    //    (most pivots structural zeros: the pivoting engine), the minimum degree the plans really use.
    for (int n : {7, 40, 120, 300, 1000}) {
        const Csr a = jacobianLike(n, 3.0 / n, 100 + n);
        const std::vector<double> v = values(a, 200 + n, 20.0);
        const std::vector<int> md = minimumDegreeOrder(n, a.rowPtr.data(), a.colIdx.data());
        const std::vector<std::vector<int>> orders = {identityOrder(n), shuffledOrder(n, 1), md};
        const char* names[] = {"identity", "shuffled", "minimum degree"};
        for (size_t o = 0; o < orders.size(); o++) {
            checkSystem("jacobian-like n=" + std::to_string(n) + " " + names[o], a, v, orders[o], orders[o],
                        300 + n + o);
            // the same elimination on rows moved by pi: rowOrder = pi o order differs from colOrder = order
            const std::vector<int> pi = shuffledOrder(n, 7 + o);
            std::vector<double> moved;
            const Csr permuted = permuteRows(a, v, pi, moved);
            std::vector<int> rowOrder(n);
            for (int k = 0; k < n; k++) {
                rowOrder[k] = pi[orders[o][k]];
            }
            checkSystem("jacobian-like n=" + std::to_string(n) + " " + names[o] + ", rows moved", permuted, moved,
                        rowOrder, orders[o], 350 + n + o);
        }
    }
    // 2. weak diagonals: the fast engine's multipliers grow, the pivoting engine pivots inside its supernodes
    for (int n : {40, 120, 300}) {
        const Csr a = jacobianLike(n, 4.0 / n, 400 + n);
        const std::vector<double> v = values(a, 500 + n, 0.05);
        const std::vector<int> md = minimumDegreeOrder(n, a.rowPtr.data(), a.colIdx.data());
        checkSystem("weak diagonal n=" + std::to_string(n) + " minimum degree", a, v, md, md, 600 + n, true);
    }
    // 3. a dense block with zero diagonals: the pivoting engine's complete pivoting inside supernodes of 16
    for (int n : {12, 16, 40}) {
        const Csr a = denseCsr(n);
        std::vector<double> v = values(a, 700 + n, 20.0);
        for (int i = 0; i < n; i++) {
            v[a.rowPtr[i] + i] = 0.0;
        }
        checkSystem("dense zero-diagonal n=" + std::to_string(n), a, v, identityOrder(n), identityOrder(n), 800 + n,
                    true);
    }
    // 4. couplers: the fast engine learns pivot delays (BatchedLu's loop), the transposed solve runs on the delayed order
    for (int n : {60, 200, 600}) {
        std::vector<double> v;
        const Csr a = couplerSystem(1000 + n, n, n / 10, 3.0 / n, v);
        checkLearnedDelays("couplers n=" + std::to_string(n) + " natural order", a, v, identityOrder(n), 1100 + n);
        checkLearnedDelays("couplers n=" + std::to_string(n) + " minimum degree", a, v,
                           minimumDegreeOrder(n, a.rowPtr.data(), a.colIdx.data()), 1200 + n);
    }
    // 5. a hub joined to every row of a chain: long rows, long columns
    for (bool hubLast : {true, false}) {
        const Csr a = hubAndChain(300, hubLast);
        checkSystem(std::string("hub and chain, hub ") + (hubLast ? "last" : "first"), a, values(a, 1300, 20.0),
                    identityOrder(300), identityOrder(300), 1301);
    }
    // 6. the Java tests' line group (two buses joined by a line, and by a coupler only to the rest, breaker mode): in the
    //    natural order the line's second balance cancels on every lane, and the fast engine learns to delay it; alone
    //    and as 50 blocks with their own values
    for (int blocks : {1, 50}) {
        const int lgRowPtr[] = {0, 2, 5, 7, 9};
        const int lgColIdx[] = {0, 1, 0, 1, 3, 1, 2, 2, 3};
        std::mt19937_64 random(1400 + blocks);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        Csr a;
        a.n = 4 * blocks;
        a.rowPtr.assign(a.n + 1, 0);
        std::vector<double> v;
        for (int k = 0; k < blocks; k++) {
            const double g1 = 1 + 9 * u(random);
            const double g2 = blocks == 1 ? g1 : 1 + 9 * u(random);   // lossless, or a lossy line
            const double sh = 1 + 9 * u(random);
            const double lg[] = {g1, -g1, -g2, g2, 1, 1, -1, sh, -1};
            for (int r = 0; r < 4; r++) {
                for (int p = lgRowPtr[r]; p < lgRowPtr[r + 1]; p++) {
                    a.colIdx.push_back(4 * k + lgColIdx[p]);
                    v.push_back(lg[p]);
                }
                a.rowPtr[4 * k + r + 1] = static_cast<int>(a.colIdx.size());
            }
        }
        checkLearnedDelays("line groups x" + std::to_string(blocks), a, v, identityOrder(a.n), 1500 + blocks);
    }
    // 7. systems from files (real Jacobians): their own minimum-degree order
    verbose = true;
    for (int k = 1; k < argc; k++) {
        Csr a;
        std::vector<double> v;
        if (!readSystem(argv[k], a, v)) {
            std::printf("cannot read %s\n", argv[k]);
            failures++;
            continue;
        }
        const std::vector<int> md = minimumDegreeOrder(a.n, a.rowPtr.data(), a.colIdx.data());
        const std::string name = std::string(argv[k]).substr(std::string(argv[k]).find_last_of('/') + 1);
        checkSystem(name + " (n=" + std::to_string(a.n) + ", nnz=" + std::to_string(a.nnz()) + ") minimum degree", a,
                    v, md, md, 900 + k, true);
        const std::vector<int> pi = shuffledOrder(a.n, 950 + k);
        std::vector<double> moved;
        const Csr permuted = permuteRows(a, v, pi, moved);
        std::vector<int> rowOrder(a.n);
        for (int q = 0; q < a.n; q++) {
            rowOrder[q] = pi[md[q]];
        }
        checkSystem(name + " minimum degree, rows moved", permuted, moved, rowOrder, md, 970 + k, true);
    }
    std::printf("fast engine: %d systems factorized (%d left to the pivoting engine); worst backward error plain %.2e,"
                " transposed %.2e; transposed vs dense pivoted solve %.2e\n", worst.fastSystems, worst.fastTroubled,
                worst.fastForward, worst.fastTransposed, worst.fastVsDense);
    std::printf("pivoting engine: %d systems (%d clamped); worst backward error plain %.2e, transposed %.2e; transposed"
                " vs dense pivoted solve %.2e\n", worst.pivotingSystems, worst.pivotingClamped, worst.pivotingForward,
                worst.pivotingTransposed, worst.pivotingVsDense);
    std::printf("systems solved on orders whose rows and columns differ: %d\n", worst.unsymmetricOrders);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
