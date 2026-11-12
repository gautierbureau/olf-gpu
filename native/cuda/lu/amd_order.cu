// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See amd_order.cuh.
#include "amd_order.cuh"

#include <amd.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace olfgpu {

std::vector<int> approximateMinimumDegreeOrder(int n, const int* rowPtr, const int* colIdx) {
    std::vector<int> order(n);
    if (n == 0) {
        return order;
    }
    // AMD breaks its ties in the order of its input entries: it is given what cuDSS orders, the symmetrized pattern
    // A + A^T with the diagonal, each row sorted and without repeats
    std::vector<std::vector<int>> rows(n);
    for (int i = 0; i < n; i++) {
        rows[i].push_back(i);
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
            rows[i].push_back(colIdx[k]);
            rows[colIdx[k]].push_back(i);
        }
    }
    std::vector<int> symmetricRowPtr(n + 1, 0);
    std::vector<int> symmetricColIdx;
    for (int i = 0; i < n; i++) {
        std::sort(rows[i].begin(), rows[i].end());
        rows[i].erase(std::unique(rows[i].begin(), rows[i].end()), rows[i].end());
        symmetricColIdx.insert(symmetricColIdx.end(), rows[i].begin(), rows[i].end());
        symmetricRowPtr[i + 1] = static_cast<int>(symmetricColIdx.size());
    }
    const int status = amd_order(n, symmetricRowPtr.data(), symmetricColIdx.data(), order.data(), nullptr, nullptr);
    if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED) {
        throw std::runtime_error("AMD ordering failed: status " + std::to_string(status));
    }
    return order;
}

}  // namespace olfgpu
