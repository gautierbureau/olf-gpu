// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See scalar_lu_plan.cuh.
#include "scalar_lu_plan.cuh"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>

namespace olfgpu {

namespace {

std::vector<int> inverseOf(int n, const int* order, const char* what) {
    std::vector<int> inverse(n, -1);
    for (int i = 0; i < n; i++) {
        int o = order[i];
        if (o < 0 || o >= n || inverse[o] != -1) {
            throw std::runtime_error(std::string(what) + " is not a permutation of 0.." + std::to_string(n - 1));
        }
        inverse[o] = i;
    }
    return inverse;
}

// Counting sort of 0..key.size()-1 by key (stable): start[b] .. start[b + 1] lists the indices with key b.
void bucketByKey(const std::vector<int>& key, int nKeys, std::vector<int>& start, std::vector<int>& list) {
    start.assign(nKeys + 1, 0);
    for (int k : key) {
        start[k + 1]++;
    }
    for (int b = 0; b < nKeys; b++) {
        start[b + 1] += start[b];
    }
    std::vector<int> cursor(start.begin(), start.end() - 1);
    list.resize(key.size());
    for (int e = 0; e < static_cast<int>(key.size()); e++) {
        list[cursor[key[e]]++] = e;
    }
}

}  // namespace

ScalarLuPlan buildScalarLuPlan(int n, const int* rowPtr, const int* colIdx, const int* rowOrder, const int* colOrder) {
    ScalarLuPlan plan;
    plan.n = n;
    plan.nnzA = rowPtr[n];
    plan.rowOrder.assign(rowOrder, rowOrder + n);
    plan.colOrder.assign(colOrder, colOrder + n);
    inverseOf(n, rowOrder, "the row order");
    const std::vector<int> colPosition = inverseOf(n, colOrder, "the column order");

    // ---- symbolic factorization, row by row: row i of L+U is B's row i closed under the U rows of its L entries,
    //      taken in ascending pivot order (a pending pivot only ever adds columns after itself) ----
    plan.luRowPtr.assign(1, 0);
    plan.luDiag.assign(n, -1);
    std::vector<int> mark(n, -1);
    std::vector<int> lower;
    std::vector<int> upper;
    std::priority_queue<int, std::vector<int>, std::greater<int>> pending;
    for (int i = 0; i < n; i++) {
        lower.clear();
        upper.clear();
        const int r = rowOrder[i];
        for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) {
            const int j = colPosition[colIdx[p]];
            if (mark[j] == i) {
                throw std::runtime_error("duplicate column in row " + std::to_string(r) + " of the CSR pattern");
            }
            mark[j] = i;
            if (j < i) {
                pending.push(j);
            } else {
                upper.push_back(j);
            }
        }
        if (mark[i] != i) {                                  // structurally zero pivot: kept as a fill entry
            mark[i] = i;
            upper.push_back(i);
        }
        while (!pending.empty()) {
            const int k = pending.top();
            pending.pop();
            lower.push_back(k);
            for (int q = plan.luDiag[k] + 1; q < plan.luRowPtr[k + 1]; q++) {
                const int j = plan.luCol[q];
                if (mark[j] != i) {
                    mark[j] = i;
                    if (j < i) {
                        pending.push(j);
                    } else {
                        upper.push_back(j);
                    }
                }
            }
        }
        std::sort(upper.begin(), upper.end());
        plan.luCol.insert(plan.luCol.end(), lower.begin(), lower.end());
        plan.luDiag[i] = static_cast<int>(plan.luCol.size());
        plan.luCol.insert(plan.luCol.end(), upper.begin(), upper.end());
        plan.luRowPtr.push_back(static_cast<int>(plan.luCol.size()));
    }
    plan.nnzLu = static_cast<int>(plan.luCol.size());

    plan.luSource.assign(plan.nnzLu, -1);
    for (int i = 0; i < n; i++) {
        const int r = rowOrder[i];
        const auto rowBegin = plan.luCol.begin() + plan.luRowPtr[i];
        const auto rowEnd = plan.luCol.begin() + plan.luRowPtr[i + 1];
        for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) {
            plan.luSource[std::lower_bound(rowBegin, rowEnd, colPosition[colIdx[p]]) - plan.luCol.begin()] = p;
        }
    }

    // ---- contributions: row i's L entry L[i][k] pairs with every U[k][j], j > k, and the pair lands on (i, j),
    //      which the closure above put in row i. Generated in ascending k, so each entry's list is in pivot order.
    //      Two passes (count, then fill) ----
    std::vector<int> entryOfColumn(n, -1);                   // the current row's LU entry of each column
    plan.contributionStart.assign(plan.nnzLu + 1, 0);
    for (int pass = 0; pass < 2; pass++) {
        std::vector<int> cursor;
        if (pass == 1) {
            for (int e = 0; e < plan.nnzLu; e++) {
                plan.contributionStart[e + 1] += plan.contributionStart[e];
            }
            plan.contributionL.resize(plan.contributionStart[plan.nnzLu]);
            plan.contributionU.resize(plan.contributionStart[plan.nnzLu]);
            cursor.assign(plan.contributionStart.begin(), plan.contributionStart.end() - 1);
        }
        for (int i = 0; i < n; i++) {
            for (int q = plan.luRowPtr[i]; q < plan.luRowPtr[i + 1]; q++) {
                entryOfColumn[plan.luCol[q]] = q;
            }
            for (int q = plan.luRowPtr[i]; q < plan.luDiag[i]; q++) {
                const int k = plan.luCol[q];
                for (int u = plan.luDiag[k] + 1; u < plan.luRowPtr[k + 1]; u++) {
                    const int target = entryOfColumn[plan.luCol[u]];
                    if (pass == 0) {
                        if (target < plan.luRowPtr[i] || target >= plan.luRowPtr[i + 1]
                                || plan.luCol[target] != plan.luCol[u]) {
                            throw std::runtime_error("LU plan: contribution outside the closed pattern of row "
                                                     + std::to_string(i));
                        }
                        plan.contributionStart[target + 1]++;
                    } else {
                        const int c = cursor[target]++;
                        plan.contributionL[c] = q;
                        plan.contributionU[c] = u;
                    }
                }
            }
        }
    }

    // ---- levels: a pivot comes after every pivot its row (L) or its column (U) depends on ----
    std::vector<int> level(n, 0);
    int maxLevel = 0;
    for (int p = 0; p < n; p++) {
        for (int q = plan.luRowPtr[p]; q < plan.luDiag[p]; q++) {
            level[p] = std::max(level[p], level[plan.luCol[q]] + 1);
        }
        for (int u = plan.luDiag[p] + 1; u < plan.luRowPtr[p + 1]; u++) {
            level[plan.luCol[u]] = std::max(level[plan.luCol[u]], level[p] + 1);
        }
        maxLevel = std::max(maxLevel, level[p]);
    }
    std::vector<int> stepOfEntry(plan.nnzLu);
    plan.entryDivisor.resize(plan.nnzLu);
    for (int i = 0; i < n; i++) {
        for (int q = plan.luRowPtr[i]; q < plan.luRowPtr[i + 1]; q++) {
            const int j = plan.luCol[q];
            if (j < i) {
                stepOfEntry[q] = 2 * level[j] + 1;
                plan.entryDivisor[q] = plan.luDiag[j];
            } else {
                stepOfEntry[q] = 2 * level[i];
                plan.entryDivisor[q] = j == i ? SCALAR_ITEM_PIVOT : SCALAR_ITEM_UPPER;
            }
        }
    }
    bucketByKey(stepOfEntry, 2 * (maxLevel + 1), plan.stepStart, plan.stepEntry);

    // ---- triangular-solve levels ----
    std::vector<int> forwardLevel(n, 0);
    int maxForward = 0;
    for (int i = 0; i < n; i++) {
        for (int q = plan.luRowPtr[i]; q < plan.luDiag[i]; q++) {
            forwardLevel[i] = std::max(forwardLevel[i], forwardLevel[plan.luCol[q]] + 1);
        }
        maxForward = std::max(maxForward, forwardLevel[i]);
    }
    bucketByKey(forwardLevel, maxForward + 1, plan.forwardLevelStart, plan.forwardRow);
    std::vector<int> backwardLevel(n, 0);
    int maxBackward = 0;
    for (int i = n - 1; i >= 0; i--) {
        for (int u = plan.luDiag[i] + 1; u < plan.luRowPtr[i + 1]; u++) {
            backwardLevel[i] = std::max(backwardLevel[i], backwardLevel[plan.luCol[u]] + 1);
        }
        maxBackward = std::max(maxBackward, backwardLevel[i]);
    }
    bucketByKey(backwardLevel, maxBackward + 1, plan.backwardLevelStart, plan.backwardRow);
    return plan;
}

std::string describeScalarLuPlan(const ScalarLuPlan& plan) {
    const int steps = plan.factorSteps();
    int narrowSteps = 0;
    int widestStep = 0;
    int longestList = 0;
    long long tail = 0;
    for (int s = 0; s < steps; s++) {
        const int items = plan.stepStart[s + 1] - plan.stepStart[s];
        narrowSteps += items > 0 && items < 32 ? 1 : 0;
        widestStep = std::max(widestStep, items);
        for (int t = plan.stepStart[s]; t < plan.stepStart[s + 1]; t++) {
            const int e = plan.stepEntry[t];
            const int length = plan.contributionStart[e + 1] - plan.contributionStart[e];
            longestList = std::max(longestList, length);
            if (s >= steps - steps / 10) {
                tail += length;
            }
        }
    }
    char line[256];
    std::snprintf(line, sizeof(line), "narrowSteps=%d widestStep=%d longestList=%d tailContributions=%lld (%.1f%%)",
                  narrowSteps, widestStep, longestList, tail,
                  plan.contributions() > 0 ? 100.0 * static_cast<double>(tail) / static_cast<double>(plan.contributions()) : 0.0);
    return line;
}

}  // namespace olfgpu
