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
#include "scenario_overrides.cuh"
#include "zero_impedance_fill.cuh"


namespace olfgpu {

// Batched zero-impedance fill: one thread per (branch, scenario). The Jacobian entries are
// constants, only F depends on the state — DUMMY_P/Q ±1 couplings into both buses' P/Q balances and, on a spanning-tree edge,
// the ZERO_V (v1−rho·v2) / ZERO_PHI (ph1−ph2) coupling. Active-gated, no mask (bus-coupler outage is rejected).
__global__ void fillZeroImpBatch(int nzi, int nScenarios, int n, int nnz,
                                 const double* in, const int* ix, const int* dest,
                                 const double* x, double* vals, double* F, const char* active,
                                 int nziBase, int nZiOn, const int* ziOnSlotElem,
                                 int nZiOff, const int* ziOffSlotElem,
                                 const int* ziAlt, const int* ziSel, const int* mode, int ziDyn,
                                 RelocTables rt, int nZiTree, const int* ziTreeSlotElemVal,
                                 const char* ziOffMask, const char* ziOnMask, const char* ziTreeMask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nzi * nScenarios) return;
    int e = tid % nzi;
    int s = tid / nzi;
    if (!active[s]) return;
    size_t me = (size_t) s * nzi + e;                        // the dense (scenario, element) masks, when built
    if (ziOffMask != nullptr) {
        if (ziOffMask[me]) return;                           // per-scenario switch OUTAGE
    } else {
        for (int k = 0; k < nZiOff; k++) {                   // per-scenario switch OUTAGE (node-breaker
            if (ziOffSlotElem[2 * k] == s && ziOffSlotElem[2 * k + 1] == e) {   // contingency propagation)
                return;
            }
        }
    }
    if (e >= nziBase) {
        // BASE-OPEN closable switch element: contributes ONLY in scenarios that CLOSE it —
        // sparse (slot, elem) enable list, a handful of entries per run.
        bool on = false;
        if (ziOnMask != nullptr) {
            on = ziOnMask[me] != 0;
        } else {
            for (int k = 0; k < nZiOn; k++) {
                if (ziOnSlotElem[2 * k] == s && ziOnSlotElem[2 * k + 1] == e) {
                    on = true;
                    break;
                }
            }
        }
        if (!on) return;
    }
    double rho = in[e];
    const int* q = ix + e * 11;                              // b1P b1Q b2P b2Q dpCol dqCol v1 v2 ph1 ph2 tree
    const double* xs = x + (size_t) s * n;
    double* valsS = vals + (size_t) s * nnz;
    double* FS = F + (size_t) s * n;
    const int* d = dest + e * 14;
    double dummyP = xs[q[4]], dummyQ = xs[q[5]];
    int d2 = d[2], d3 = d[3], d6 = d[6], d7 = d[7];
    int d0 = d[0], d1 = d[1], d4 = d[4], d5 = d[5];
    if (ziDyn) {                                             // dynamic Q-routing (ob/zi channels)
        const int* ms = mode + (size_t) s * n;
        RelocTables rtS = relocForScenario(rt, s);
        d2 = ziDynDest(d2, e * 8 + 0, ziAlt, ziSel, ms, rtS);
        d3 = ziDynDest(d3, e * 8 + 1, ziAlt, ziSel, ms, rtS);
        d6 = ziDynDest(d6, e * 8 + 2, ziAlt, ziSel, ms, rtS);
        d7 = ziDynDest(d7, e * 8 + 3, ziAlt, ziSel, ms, rtS);
        d0 = ziDynDest(d0, e * 8 + 4, ziAlt, ziSel, ms, rtS);   // +-dummyP: back to the bus's own PHI row
        d1 = ziDynDest(d1, e * 8 + 5, ziAlt, ziSel, ms, rtS);   // when its parent coupler opens
        d4 = ziDynDest(d4, e * 8 + 6, ziAlt, ziSel, ms, rtS);
        d5 = ziDynDest(d5, e * 8 + 7, ziAlt, ziSel, ms, rtS);
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
    if (ziTreeMask != nullptr) {
        char t = ziTreeMask[me];                             // ...overridden where OLF's has moved by then
        if (t >= 0) isTree = t;
    } else {
        for (int k = 0; k < nZiTree; k++) {
            if (ziTreeSlotElemVal[3 * k] == s && ziTreeSlotElemVal[3 * k + 1] == e) {
                isTree = ziTreeSlotElemVal[3 * k + 2];
                break;
            }
        }
    }
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
