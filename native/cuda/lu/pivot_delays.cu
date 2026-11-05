// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See pivot_delays.cuh.
#include "pivot_delays.cuh"

#include <algorithm>
#include <cmath>

#include "lu_pivot.cuh"

namespace olfgpu {

namespace {

// The analyzed positions in the delays' order: each undelayed one in turn, each followed by the pivots delayed after
// it (and theirs), in analyzed order; then the pivots delayed last, or after one another in a cycle.
std::vector<int> delayedOrder(const PivotDelays& delays, int n) {
    std::vector<int> order;
    order.reserve(n);
    if (delays.delayed == 0) {
        for (int p = 0; p < n; p++) {
            order.push_back(p);
        }
        return order;
    }
    std::vector<std::vector<int>> after(n);
    for (int p = 0; p < n; p++) {
        if (delays.partner[p] >= 0) {
            after[delays.partner[p]].push_back(p);
        }
    }
    std::vector<char> placed(n, 0);
    std::vector<int> pending;
    auto place = [&](int first) {
        pending.push_back(first);
        while (!pending.empty()) {
            const int p = pending.back();
            pending.pop_back();
            if (placed[p]) {
                continue;
            }
            placed[p] = 1;
            order.push_back(p);
            for (auto next = after[p].rbegin(); next != after[p].rend(); ++next) {
                pending.push_back(*next);
            }
        }
    };
    for (int p = 0; p < n; p++) {
        if (delays.partner[p] == PIVOT_NOT_DELAYED) {
            place(p);
        }
    }
    for (int p = 0; p < n; p++) {
        if (!placed[p]) {
            place(p);
        }
    }
    return order;
}

}  // namespace

std::shared_ptr<const FastLuPlan> buildFastLuPlan(int n, const int* rowPtr, const int* colIdx, const int* rowOrder,
                                                  const int* colOrder, const PivotDelays& delays,
                                                  const ScalarLuPlanLimits& limits) {
    auto plan = std::make_shared<FastLuPlan>();
    plan->analyzedPosition = delayedOrder(delays, n);
    plan->delayed = delays.delayed;
    std::vector<int> rows(n);
    std::vector<int> cols(n);
    for (int i = 0; i < n; i++) {
        rows[i] = rowOrder[plan->analyzedPosition[i]];
        cols[i] = colOrder[plan->analyzedPosition[i]];
    }
    plan->scalar = buildScalarLuPlan(n, rowPtr, colIdx, rows.data(), cols.data(), limits);
    const ScalarLuPlan& sp = plan->scalar;
    plan->lowerStart.assign(n + 1, 0);
    for (int i = 0; i < n; i++) {
        for (int q = sp.luRowPtr[i]; q < sp.luDiag[i]; q++) {
            plan->lowerStart[sp.luCol[q] + 1]++;
        }
    }
    for (int k = 0; k < n; k++) {
        plan->lowerStart[k + 1] += plan->lowerStart[k];
    }
    plan->lowerRow.resize(plan->lowerStart[n]);
    plan->lowerEntry.resize(plan->lowerStart[n]);
    std::vector<int> cursor(plan->lowerStart.begin(), plan->lowerStart.end() - 1);
    for (int i = 0; i < n; i++) {
        for (int q = sp.luRowPtr[i]; q < sp.luDiag[i]; q++) {
            const int c = cursor[sp.luCol[q]]++;
            plan->lowerRow[c] = i;
            plan->lowerEntry[c] = q;
        }
    }
    return plan;
}

bool learnPivotDelays(PivotDelays& delays, const FastLuPlan& plan, const int* rowPtr, const int* colIdx,
                      const double* lu, const double* a) {
    const ScalarLuPlan& sp = plan.scalar;
    const int n = sp.n;
    std::vector<double> rowMax(n, 0.0);
    std::vector<double> colMax(n, 0.0);
    for (int r = 0; r < n; r++) {
        for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) {
            const double v = std::fabs(a[p]);
            if (!std::isfinite(v)) {
                return false;
            }
            rowMax[r] = std::max(rowMax[r], v);
            colMax[colIdx[p]] = std::max(colMax[colIdx[p]], v);
        }
    }
    if (delays.partner.empty()) {
        delays.partner.assign(n, PIVOT_NOT_DELAYED);
        delays.moves.assign(n, 0);
    }
    bool changed = false;
    for (int k = 0; k < n; k++) {
        const double pivot = std::fabs(lu[sp.luDiag[k]]);
        const double scale = std::max(rowMax[sp.rowOrder[k]], colMax[sp.colOrder[k]]);
        if (pivot != LU_PIVOT_EPSILON && pivot >= LU_PIVOT_RELATIVE_LIMIT * scale) {
            continue;                                        // not cancelled (a NaN pivot is)
        }
        int rescue = -1;
        double largest = 0.0;
        for (int c = plan.lowerStart[k]; c < plan.lowerStart[k + 1]; c++) {
            const double v = std::fabs(lu[plan.lowerEntry[c]]);
            if (v > largest) {
                largest = v;
                rescue = plan.lowerRow[c];
            }
        }
        if (rescue < 0) {
            for (int q = sp.luDiag[k] + 1; q < sp.luRowPtr[k + 1]; q++) {
                const double v = std::fabs(lu[q]);
                if (v > largest) {
                    largest = v;
                    rescue = sp.luCol[q];
                }
            }
        }
        const int p = plan.analyzedPosition[k];
        const int partner = rescue >= 0 ? plan.analyzedPosition[rescue] : PIVOT_DELAYED_LAST;
        if (delays.partner[p] == PIVOT_NOT_DELAYED) {
            delays.partner[p] = partner;
            delays.delayed++;
            changed = true;
        } else if (delays.partner[p] != partner && delays.moves[p] < PIVOT_DELAY_MAX_MOVES) {
            delays.partner[p] = partner;
            delays.moves[p]++;
            changed = true;
        }
    }
    return changed;
}

}  // namespace olfgpu
