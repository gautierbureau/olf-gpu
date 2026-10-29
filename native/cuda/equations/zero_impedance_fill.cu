// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See zero_impedance_fill.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "scatter.cuh"
#include "../fixedmatrix/reactive_relocation.cuh"
#include "../fixedmatrix/relocated_scatter.cuh"
#include "scenario_overrides.cuh"
#include "zero_impedance_fill.cuh"


namespace olfgpu {

// Batched zero-impedance fill: one thread per (branch, scenario). The Jacobian entries are
// constants, only F depends on the state — DUMMY_P/Q ±1 couplings into both buses' P/Q balances and, on a spanning-tree edge,
// the ZERO_V (v1−rho·v2) / ZERO_PHI (ph1−ph2) coupling. Active-gated, no mask (bus-coupler outage is rejected).
__global__ void fillZeroImpBatch(int nZeroImpedanceBranch, int nScenarios, int n, int nnz,
                                 const double* in, const int* ix, const int* dest,
                                 const double* x, double* vals, double* F, const char* active,
                                 int nBaseClosedZeroImpedanceBranches,
                                 const int* zeroImpedanceAlternateDestination, const int* zeroImpedanceSelectorRow, const int* mode, int zeroImpedanceDynamicRelocation,
                                 ReactiveRelocationTables rt,
                                 const char* zeroImpedanceBranchOpenMask, const char* zeroImpedanceBranchCloseMask, const char* spanningTreeOverrideMask, BatchLayout layout) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nZeroImpedanceBranch * nScenarios) return;
    int e, s;
    itemAndScenario(tid, nZeroImpedanceBranch, layout, e, s);
    if (!active[s]) return;
    size_t me = (size_t) s * nZeroImpedanceBranch + e;                        // the dense (scenario, element) masks, when built
    if (zeroImpedanceBranchOpenMask[me]) return;                               // per-scenario switch OUTAGE (node-breaker contingency propagation)
    if (e >= nBaseClosedZeroImpedanceBranches) {
        // BASE-OPEN closable switch element: contributes ONLY in scenarios that CLOSE it —
        // sparse (slot, elem) enable list, a handful of entries per run.
        if (zeroImpedanceBranchCloseMask[me] == 0) return;
    }
    double rho = in[e];
    const int* q = ix + e * 11;                              // b1P b1Q b2P b2Q dummyPColumn dummyQColumn v1 v2 ph1 ph2 tree
    const double* xs = x + (size_t) s * n;
    const ScenarioValues valsS = scenarioEntries(vals, s, nnz, layout);
    const ScenarioValues FS = scenarioEntries(F, s, n, layout);
    const int* d = dest + e * 14;
    double dummyP = xs[q[4]], dummyQ = xs[q[5]];
    int d2 = d[2], d3 = d[3], d6 = d[6], d7 = d[7];
    int d0 = d[0], d1 = d[1], d4 = d[4], d5 = d[5];
    if (zeroImpedanceDynamicRelocation) {                                             // dynamic Q-routing (ob/zero-impedance channels)
        const int* ms = mode + (size_t) s * n;
        ReactiveRelocationTables rtS = relocationTablesForScenario(rt, s);
        d2 = zeroImpedanceRelocationDestination(d2, e * 8 + 0, zeroImpedanceAlternateDestination, zeroImpedanceSelectorRow, ms, rtS);
        d3 = zeroImpedanceRelocationDestination(d3, e * 8 + 1, zeroImpedanceAlternateDestination, zeroImpedanceSelectorRow, ms, rtS);
        d6 = zeroImpedanceRelocationDestination(d6, e * 8 + 2, zeroImpedanceAlternateDestination, zeroImpedanceSelectorRow, ms, rtS);
        d7 = zeroImpedanceRelocationDestination(d7, e * 8 + 3, zeroImpedanceAlternateDestination, zeroImpedanceSelectorRow, ms, rtS);
        d0 = zeroImpedanceRelocationDestination(d0, e * 8 + 4, zeroImpedanceAlternateDestination, zeroImpedanceSelectorRow, ms, rtS);   // +-dummyP: back to the bus's own PHI row
        d1 = zeroImpedanceRelocationDestination(d1, e * 8 + 5, zeroImpedanceAlternateDestination, zeroImpedanceSelectorRow, ms, rtS);   // when its parent coupler opens
        d4 = zeroImpedanceRelocationDestination(d4, e * 8 + 6, zeroImpedanceAlternateDestination, zeroImpedanceSelectorRow, ms, rtS);
        d5 = zeroImpedanceRelocationDestination(d5, e * 8 + 7, zeroImpedanceAlternateDestination, zeroImpedanceSelectorRow, ms, rtS);
    }
    addAt(d0, dummyP, valsS, FS);                          // F[bus1 P] += dummyP
    addAt(d1, -dummyP, valsS, FS);                         // F[bus2 P] -= dummyP
    addAt(d2, dummyQ, valsS, FS);                            // F[bus1 Q] += dummyQ
    addAt(d3, -dummyQ, valsS, FS);                           // F[bus2 Q] -= dummyQ
    addAt(d4, 1.0, valsS, FS);                             // J(bus1 P, dummyP)
    addAt(d5, -1.0, valsS, FS);                            // J(bus2 P, dummyP)
    addAt(d6, 1.0, valsS, FS);                               // J(bus1 Q, dummyQ)
    addAt(d7, -1.0, valsS, FS);                              // J(bus2 Q, dummyQ)
    int isTree = q[10] & 1;                                  // the tree at EXTRACTION...
    char t = spanningTreeOverrideMask[me];                                 // ...overridden where OLF's has moved by then
    if (t >= 0) isTree = t;
    if (isTree) {                                            // spanning-tree edge: voltage/angle coupling
        double v1 = xs[q[6]], v2 = xs[q[7]], ph1 = xs[q[8]], ph2 = xs[q[9]];
        addAt(d[8], v1 - rho * v2, valsS, FS);               // F[ZERO_V row]
        addAt(d[9], 1.0, valsS, FS);                         // J(ZERO_V, v1)
        addAt(d[10], -rho, valsS, FS);                       // J(ZERO_V, v2)
        addAt(d[11], ph1 - ph2, valsS, FS);                  // F[ZERO_PHI row]
        addAt(d[12], 1.0, valsS, FS);                        // J(ZERO_PHI, ph1)
        addAt(d[13], -1.0, valsS, FS);                       // J(ZERO_PHI, ph2)
    }
}

}  // namespace olfgpu
