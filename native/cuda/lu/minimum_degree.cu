// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See minimum_degree.cuh.
#include "minimum_degree.cuh"

#include <algorithm>
#include <functional>
#include <iterator>
#include <queue>
#include <utility>

namespace olfgpu {

namespace {

// A + A^T without the diagonal, each row sorted.
std::vector<std::vector<int>> symmetricAdjacency(int n, const int* rowPtr, const int* colIdx) {
    std::vector<std::vector<int>> adjacency(n);
    for (int i = 0; i < n; i++) {
        for (int e = rowPtr[i]; e < rowPtr[i + 1]; e++) {
            const int j = colIdx[e];
            if (j != i) {
                adjacency[i].push_back(j);
                adjacency[j].push_back(i);
            }
        }
    }
    for (std::vector<int>& row : adjacency) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return adjacency;
}

}  // namespace

std::vector<int> minimumDegreeOrder(int n, const int* rowPtr, const int* colIdx) {
    std::vector<std::vector<int>> adjacency = symmetricAdjacency(n, rowPtr, colIdx);
    using Entry = std::pair<int, int>;                       // (degree, vertex): the least degree, then the lowest index
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    for (int v = 0; v < n; v++) {
        queue.emplace(static_cast<int>(adjacency[v].size()), v);
    }
    std::vector<char> eliminated(n, 0);
    std::vector<int> order;
    order.reserve(n);
    std::vector<int> merged;
    while (!queue.empty()) {
        const int degree = queue.top().first;
        const int v = queue.top().second;
        queue.pop();
        if (eliminated[v] || degree != static_cast<int>(adjacency[v].size())) {
            continue;                                        // an entry its vertex's later degree superseded
        }
        eliminated[v] = 1;
        order.push_back(v);
        const std::vector<int> neighbours = std::move(adjacency[v]);
        adjacency[v].clear();
        for (int u : neighbours) {
            // u loses v and gains the other neighbours of v: the eliminated vertex's neighbourhood becomes a clique
            merged.clear();
            std::set_union(adjacency[u].begin(), adjacency[u].end(), neighbours.begin(), neighbours.end(),
                           std::back_inserter(merged));
            merged.erase(std::remove_if(merged.begin(), merged.end(), [&](int w) { return w == u || w == v; }),
                         merged.end());
            adjacency[u].swap(merged);
        }
        for (int u : neighbours) {
            queue.emplace(static_cast<int>(adjacency[u].size()), u);
        }
    }
    return order;
}

int eliminationLevels(int n, const int* rowPtr, const int* colIdx, const std::vector<int>& order) {
    const std::vector<std::vector<int>> adjacency = symmetricAdjacency(n, rowPtr, colIdx);
    std::vector<int> position(n);
    for (int k = 0; k < n; k++) {
        position[order[k]] = k;
    }
    // Liu's elimination tree of the permuted pattern, with path compression
    std::vector<int> parent(n, -1);
    std::vector<int> ancestor(n, -1);
    for (int i = 0; i < n; i++) {
        for (int original : adjacency[order[i]]) {
            int r = position[original];
            if (r >= i) {
                continue;
            }
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
    std::vector<int> level(n, 0);                            // a parent comes after its children
    int height = n > 0 ? 1 : 0;
    for (int i = 0; i < n; i++) {
        if (parent[i] != -1) {
            level[parent[i]] = std::max(level[parent[i]], level[i] + 1);
        }
        height = std::max(height, level[i] + 1);
    }
    return height;
}

}  // namespace olfgpu
