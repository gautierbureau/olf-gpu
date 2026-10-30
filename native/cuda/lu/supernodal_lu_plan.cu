// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See supernodal_lu_plan.cuh.
#include "supernodal_lu_plan.cuh"

#include <algorithm>
#include <cstdio>
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

// The lower part of the symmetrized pattern of B = A(rowOrder, colOrder): per column j, the rows i > j where B[i][j]
// or B[j][i] is an entry, ascending.
std::vector<std::vector<int>> symmetrizedLower(int n, const int* rowPtr, const int* colIdx, const int* rowOrder,
                                               const std::vector<int>& colPosition) {
    std::vector<std::vector<int>> lower(n);
    for (int i = 0; i < n; i++) {
        const int r = rowOrder[i];
        for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) {
            const int j = colPosition[colIdx[p]];
            if (i > j) {
                lower[j].push_back(i);
            } else if (j > i) {
                lower[i].push_back(j);
            }
        }
    }
    for (auto& rows : lower) {
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    return lower;
}

// The elimination tree of a symmetric pattern given by its lower part (Liu's algorithm, path compression).
std::vector<int> eliminationTree(int n, const std::vector<std::vector<int>>& lower) {
    std::vector<std::vector<int>> upperOfRow(n);                 // row i: the columns k < i with S[i][k]
    for (int k = 0; k < n; k++) {
        for (int i : lower[k]) {
            upperOfRow[i].push_back(k);
        }
    }
    std::vector<int> parent(n, -1);
    std::vector<int> ancestor(n, -1);
    for (int i = 0; i < n; i++) {
        for (int k : upperOfRow[i]) {
            int r = k;
            while (ancestor[r] != -1 && ancestor[r] != i) {
                const int next = ancestor[r];
                ancestor[r] = i;
                r = next;
            }
            if (ancestor[r] == -1) {
                ancestor[r] = i;
                parent[r] = i;
            }
        }
    }
    return parent;
}

// Postorder of a forest given by parent links (children visited in ascending order): post[new] = old.
std::vector<int> postorder(int n, const std::vector<int>& parent) {
    std::vector<int> head(n, -1);
    std::vector<int> next(n, -1);
    for (int j = n - 1; j >= 0; j--) {                           // children lists in ascending order
        if (parent[j] >= 0) {
            next[j] = head[parent[j]];
            head[parent[j]] = j;
        }
    }
    std::vector<int> post;
    post.reserve(n);
    std::vector<int> stack;
    for (int root = 0; root < n; root++) {
        if (parent[root] != -1) {
            continue;
        }
        stack.push_back(root);
        while (!stack.empty()) {
            const int j = stack.back();
            if (head[j] != -1) {                                 // descend into the next unvisited child
                const int child = head[j];
                head[j] = next[child];
                stack.push_back(child);
            } else {
                post.push_back(j);
                stack.pop_back();
            }
        }
    }
    return post;
}

}  // namespace

long long SupernodalLuPlan::gatherMultiplyAdds() const {
    long long total = 0;
    for (size_t r = 0; r < runLength.size(); r++) {
        total += runLength[r];
    }
    return total;
}

SupernodalLuPlan buildSupernodalLuPlan(int n, const int* rowPtr, const int* colIdx, const int* rowOrder, const int* colOrder) {
    SupernodalLuPlan plan;
    plan.n = n;
    plan.nnzA = rowPtr[n];
    inverseOf(n, rowOrder, "the row order");
    const std::vector<int> givenColPosition = inverseOf(n, colOrder, "the column order");

    // ---- postorder the elimination tree of the symmetrized pattern: same fill, consecutive supernodes ----
    std::vector<int> post;
    {
        const auto lower = symmetrizedLower(n, rowPtr, colIdx, rowOrder, givenColPosition);
        post = postorder(n, eliminationTree(n, lower));
    }
    plan.rowOrder.resize(n);
    plan.colOrder.resize(n);
    for (int i = 0; i < n; i++) {
        plan.rowOrder[i] = rowOrder[post[i]];
        plan.colOrder[i] = colOrder[post[i]];
    }
    const std::vector<int> rowPosition = inverseOf(n, plan.rowOrder.data(), "the row order");
    const std::vector<int> colPosition = inverseOf(n, plan.colOrder.data(), "the column order");
    const auto lower = symmetrizedLower(n, rowPtr, colIdx, plan.rowOrder.data(), colPosition);
    const std::vector<int> parent = eliminationTree(n, lower);

    // ---- the Cholesky structure: struct(j) = lower(j) U (struct(c) \ {j}) over the children c of j (rows below j) ----
    std::vector<std::vector<int>> structure(n);
    std::vector<int> childCount(n, 0);
    {
        std::vector<std::vector<int>> children(n);
        for (int j = 0; j < n; j++) {
            if (parent[j] >= 0) {
                children[parent[j]].push_back(j);
                childCount[parent[j]]++;
            }
        }
        std::vector<int> merged;
        for (int j = 0; j < n; j++) {
            merged.assign(lower[j].begin(), lower[j].end());
            for (int c : children[j]) {
                for (int i : structure[c]) {
                    if (i != j) {
                        merged.push_back(i);
                    }
                }
            }
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            structure[j] = merged;
            plan.choleskyEntries += 2 * static_cast<long long>(merged.size()) + 1;
        }
    }

    // ---- fundamental supernodes: a chain of the tree with nested structure, at most LU_MAX_SUPERNODE columns ----
    std::vector<int> fundamentalStart(1, 0);
    for (int j = 1; j < n; j++) {
        const int start = fundamentalStart.back();
        const bool chained = parent[j - 1] == j && childCount[j] == 1
                && structure[j - 1].size() == structure[j].size() + 1;
        if (!chained || j - start >= LU_MAX_SUPERNODE) {
            fundamentalStart.push_back(j);
        }
    }
    fundamentalStart.push_back(n);

    // ---- relaxed amalgamation (CHOLMOD's rule within LU_MAX_SUPERNODE columns): a supernode joins its parent when the
    //      parent comes right after it and the merged block has at most 4 columns or less than 80 % stored zeros. A
    //      bigger diagonal block is more room for a scenario to pivot in: cuDSS solves a controller's renormalized row
    //      (a zero diagonal after a generator outage) that 2-column fundamental supernodes clamp. ----
    {
        const int nFundamental = static_cast<int>(fundamentalStart.size()) - 1;
        std::vector<int> columns(nFundamental);
        std::vector<int> top(nFundamental);
        std::vector<long long> nonZeros(nFundamental, 0);
        std::vector<char> head(nFundamental, 1);
        std::vector<int> fundamentalOf(n);
        for (int s = 0; s < nFundamental; s++) {
            columns[s] = fundamentalStart[s + 1] - fundamentalStart[s];
            top[s] = s;
            for (int j = fundamentalStart[s]; j < fundamentalStart[s + 1]; j++) {
                nonZeros[s] += static_cast<long long>(structure[j].size()) + 1;
                fundamentalOf[j] = s;
            }
        }
        for (int s = nFundamental - 2; s >= 0; s--) {
            const int last = fundamentalStart[s + 1] - 1;
            if (parent[last] < 0 || fundamentalOf[parent[last]] != s + 1) {
                continue;                                        // the parent does not come right after
            }
            const long long merged = columns[s] + columns[s + 1];
            if (merged > LU_MAX_SUPERNODE) {
                continue;
            }
            const long long below = static_cast<long long>(structure[fundamentalStart[top[s + 1] + 1] - 1].size());
            const long long stored = merged * (merged + 1) / 2 + merged * below;
            const long long nonZero = nonZeros[s] + nonZeros[s + 1];
            if (merged <= 4 || static_cast<double>(stored - nonZero) < 0.8 * static_cast<double>(stored)) {
                columns[s] = static_cast<int>(merged);
                nonZeros[s] = nonZero;
                top[s] = top[s + 1];
                head[s + 1] = 0;
            }
        }
        for (int s = 0; s < nFundamental; s++) {
            if (head[s]) {
                plan.supernodeStart.push_back(fundamentalStart[s]);
            }
        }
        plan.supernodeStart.push_back(n);
    }
    plan.nSupernodes = static_cast<int>(plan.supernodeStart.size()) - 1;
    std::vector<int> supernodeOf(n);
    for (int s = 0; s < plan.nSupernodes; s++) {
        for (int j = plan.supernodeStart[s]; j < plan.supernodeStart[s + 1]; j++) {
            supernodeOf[j] = s;
        }
    }

    // ---- per supernode: R = the structure of its last column; storage [diagonal | L panel | U panel] ----
    plan.rowStart.assign(1, 0);
    plan.diagonalBase.resize(plan.nSupernodes);
    plan.lowerBase.resize(plan.nSupernodes);
    plan.upperBase.resize(plan.nSupernodes);
    long long stored = 0;
    for (int s = 0; s < plan.nSupernodes; s++) {
        const int last = plan.supernodeStart[s + 1] - 1;
        plan.rows.insert(plan.rows.end(), structure[last].begin(), structure[last].end());
        plan.rowStart.push_back(static_cast<int>(plan.rows.size()));
        const long long b = plan.supernodeSize(s);
        const long long r = plan.rowCount(s);
        plan.diagonalBase[s] = static_cast<int>(stored);
        plan.lowerBase[s] = static_cast<int>(stored + b * b);
        plan.upperBase[s] = static_cast<int>(stored + b * b + r * b);
        stored += b * b + 2 * r * b;
        if (stored > 2000000000LL) {
            throw std::runtime_error("LU plan: more than 2e9 stored entries");
        }
    }
    plan.nStored = static_cast<int>(stored);
    std::vector<std::vector<int>>().swap(structure);
    auto storedEntry = [&](int i, int j) {                       // (row, column) of B -> stored entry, both in a panel
        const int s = supernodeOf[std::min(i, j)];
        const int f = plan.supernodeStart[s];
        const int b = plan.supernodeSize(s);
        if (supernodeOf[i] == s && supernodeOf[j] == s) {
            return plan.diagonalBase[s] + (i - f) * b + (j - f);
        }
        const int* rowsBegin = plan.rows.data() + plan.rowStart[s];
        const int* rowsEnd = plan.rows.data() + plan.rowStart[s + 1];
        const int other = std::max(i, j);
        const int* found = std::lower_bound(rowsBegin, rowsEnd, other);
        if (found == rowsEnd || *found != other) {
            throw std::runtime_error("LU plan: entry outside the supernodal structure");
        }
        const int index = static_cast<int>(found - rowsBegin);
        return i > j ? plan.lowerBase[s] + index * b + (j - f)          // L panel, row-major
                     : plan.upperBase[s] + index * b + (i - f);          // U panel, column-major
    };

    plan.storedOfA.resize(plan.nnzA);
    for (int r = 0; r < n; r++) {
        for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) {
            plan.storedOfA[p] = storedEntry(rowPosition[r], colPosition[colIdx[p]]);
        }
    }

    // ---- gather runs: every pair (i, j) of R_t receives L_t[i][.] . U_t[.][j], supernodes t ascending ----
    plan.runStart.assign(plan.nStored + 1, 0);
    for (int pass = 0; pass < 2; pass++) {
        std::vector<int> cursor;
        if (pass == 1) {
            for (int e = 0; e < plan.nStored; e++) {
                plan.runStart[e + 1] += plan.runStart[e];
            }
            const size_t runs = static_cast<size_t>(plan.runStart[plan.nStored]);
            plan.runLower.resize(runs);
            plan.runUpper.resize(runs);
            plan.runLength.resize(runs);
            cursor.assign(plan.runStart.begin(), plan.runStart.end() - 1);
        }
        for (int t = 0; t < plan.nSupernodes; t++) {
            const int b = plan.supernodeSize(t);
            const int count = plan.rowCount(t);
            const int* rowsT = plan.rows.data() + plan.rowStart[t];
            for (int a = 0; a < count; a++) {
                for (int c = 0; c < count; c++) {
                    const int target = storedEntry(rowsT[a], rowsT[c]);
                    if (pass == 0) {
                        plan.runStart[target + 1]++;
                    } else {
                        const int run = cursor[target]++;
                        plan.runLower[run] = plan.lowerBase[t] + a * b;
                        plan.runUpper[run] = plan.upperBase[t] + c * b;
                        plan.runLength[run] = b;
                    }
                }
            }
        }
        if (pass == 0) {
            long long total = 0;
            for (int e = 0; e < plan.nStored; e++) {
                total += plan.runStart[e + 1];
            }
            if (total > 2000000000LL) {
                throw std::runtime_error("LU plan: more than 2e9 gather runs");
            }
        }
    }

    // ---- levels: a supernode after its whole subtree (postorder: children come first) ----
    std::vector<int> level(plan.nSupernodes, 0);
    int maxLevel = 0;
    for (int s = 0; s < plan.nSupernodes; s++) {
        const int last = plan.supernodeStart[s + 1] - 1;
        if (parent[last] >= 0) {
            const int up = supernodeOf[parent[last]];
            level[up] = std::max(level[up], level[s] + 1);
        }
        maxLevel = std::max(maxLevel, level[s]);
    }
    plan.levelStart.assign(maxLevel + 2, 0);
    for (int s = 0; s < plan.nSupernodes; s++) {
        plan.levelStart[level[s] + 1]++;
    }
    for (int l = 0; l <= maxLevel; l++) {
        plan.levelStart[l + 1] += plan.levelStart[l];
    }
    plan.levelSupernode.resize(plan.nSupernodes);
    {
        std::vector<int> cursor(plan.levelStart.begin(), plan.levelStart.end() - 1);
        for (int s = 0; s < plan.nSupernodes; s++) {
            plan.levelSupernode[cursor[level[s]]++] = s;
        }
    }
    plan.levelEntryStart.assign(1, 0);
    plan.levelPanelStart.assign(1, 0);
    for (int l = 0; l <= maxLevel; l++) {
        for (int q = plan.levelStart[l]; q < plan.levelStart[l + 1]; q++) {
            const int s = plan.levelSupernode[q];
            const int end = s + 1 < plan.nSupernodes ? plan.diagonalBase[s + 1] : plan.nStored;
            for (int e = plan.diagonalBase[s]; e < end; e++) {
                plan.levelEntry.push_back(e);
            }
            for (int item = 0; item < 2 * plan.rowCount(s); item++) {
                plan.panelSupernode.push_back(s);
                plan.panelRow.push_back(item);
            }
        }
        plan.levelEntryStart.push_back(static_cast<int>(plan.levelEntry.size()));
        plan.levelPanelStart.push_back(static_cast<int>(plan.panelSupernode.size()));
    }

    // ---- forward solve: row i takes L_t[i][.] . y_t from every supernode t holding i in R_t ----
    plan.forwardRunStart.assign(n + 1, 0);
    for (int t = 0; t < plan.nSupernodes; t++) {
        for (int a = plan.rowStart[t]; a < plan.rowStart[t + 1]; a++) {
            plan.forwardRunStart[plan.rows[a] + 1]++;
        }
    }
    for (int i = 0; i < n; i++) {
        plan.forwardRunStart[i + 1] += plan.forwardRunStart[i];
    }
    plan.forwardRunLower.resize(plan.forwardRunStart[n]);
    plan.forwardRunY.resize(plan.forwardRunStart[n]);
    plan.forwardRunLength.resize(plan.forwardRunStart[n]);
    {
        std::vector<int> cursor(plan.forwardRunStart.begin(), plan.forwardRunStart.end() - 1);
        for (int t = 0; t < plan.nSupernodes; t++) {
            const int b = plan.supernodeSize(t);
            for (int a = plan.rowStart[t]; a < plan.rowStart[t + 1]; a++) {
                const int run = cursor[plan.rows[a]]++;
                plan.forwardRunLower[run] = plan.lowerBase[t] + (a - plan.rowStart[t]) * b;
                plan.forwardRunY[run] = plan.supernodeStart[t];
                plan.forwardRunLength[run] = b;
            }
        }
    }
    return plan;
}

std::string describeSupernodalLuPlan(const SupernodalLuPlan& plan) {
    int largest = 0;
    int singles = 0;
    for (int s = 0; s < plan.nSupernodes; s++) {
        largest = std::max(largest, plan.supernodeSize(s));
        singles += plan.supernodeSize(s) == 1 ? 1 : 0;
    }
    char line[256];
    std::snprintf(line, sizeof(line), "supernodes=%d singleColumn=%d largest=%d levels=%d stored=%d gatherMultiplyAdds=%lld runs=%zu",
                  plan.nSupernodes, singles, largest, plan.levels(), plan.nStored, plan.gatherMultiplyAdds(),
                  plan.runLength.size());
    return line;
}

}  // namespace olfgpu
