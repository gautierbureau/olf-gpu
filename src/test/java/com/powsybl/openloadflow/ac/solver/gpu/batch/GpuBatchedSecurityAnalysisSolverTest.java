/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

/**
 * CPU-only checks of the GPU batch chunk-size resolution, which bounds the per-chunk device-memory
 * footprint. A size forced by the test hook is honoured; otherwise the chunk is auto-sized from free device
 * memory by a native query,
 * falling back to the default when no GPU is present. The auto path (and the native nnz /
 * cudaMemGetInfo estimate) is exercised by the device tests.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchedSecurityAnalysisSolverTest {

    private static final GpuAcData EMPTY = new GpuAcData(new double[0], new int[0], new int[0], new double[0], new int[0],
            new double[0], new int[0], new double[0], new int[0],
            new int[0], new int[0], new double[0], new int[0], new int[0],
            new int[0], new int[0], new double[0], new int[0], new int[0], new int[0],
            new int[0], new byte[0], new int[0], new double[0], new int[0],
            new int[0], new int[0],
            new double[0], new int[0],
            new int[0], new int[0], new double[0],
            new int[0], new int[0], new double[0],
            new int[0], new double[0], new double[0], new double[0], new double[0],
            new double[0], new double[0], new double[0], new byte[0],
            new byte[0], new int[0], new int[0],
            new double[0], new int[0], new double[0], new int[0], new double[0],
            new int[0], new int[0], new int[0], new int[0], new double[0], new double[0], new double[0],
            new int[0], new double[0], -1, 0.0,
            new double[0], new double[0], new double[0],
            0, new double[0], new double[0], 0.0, 0.0, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
            new int[0], new double[0], new double[0], new double[0], new double[0], new double[0],
            new int[0], new int[0],
            new int[0], new String[0],
            new int[0], new int[0], new int[0], new int[0], new double[0], new int[0], new int[] {0}, new int[0],
            new int[0], new int[0], new byte[0], new int[0], new int[0], new int[0]);

    @Test
    void explicitCapIsHonouredAndClampedToCount() {
        GpuBatchContextCache.forceChunkSizeForTests(64);
        try {
            assertEquals(64, GpuBatchContextCache.resolveChunkSize(1000, 10, EMPTY));
            assertEquals(10, GpuBatchContextCache.resolveChunkSize(10, 10, EMPTY),
                    "chunk is clamped to the scenario count");
        } finally {
            GpuBatchContextCache.forceChunkSizeForTests(null);
        }
    }

    @Test
    void autoSizeFallsBackToDefaultWhenNoGpu() {
        assumeFalse(GpuAcNewtonSolver.isAvailable(), "auto-size queries the device when a GPU is present");
        // No GPU: recommendBatchSize is unlinked, so resolveChunkSize falls back to the default.
        GpuBatchContextCache.forceChunkSizeForTests(null);
        assertEquals(GpuBatchContextCache.DEFAULT_MAX_BATCH,
                GpuBatchContextCache.resolveChunkSize(1000, 10, EMPTY));
    }

    @Test
    void chunkSizeAlwaysClampedToScenarioCount() {
        // Holds for every path (explicit cap, auto-size, or fallback): 1 <= chunk <= nb.
        GpuBatchContextCache.forceChunkSizeForTests(null);
        int chunk = GpuBatchContextCache.resolveChunkSize(5, 10, EMPTY);
        assertTrue(chunk >= 1 && chunk <= 5, "chunk must be in [1, nb], was " + chunk);

        GpuBatchContextCache.forceChunkSizeForTests(0);   // invalid -> auto/fallback
        try {
            int c2 = GpuBatchContextCache.resolveChunkSize(5, 10, EMPTY);
            assertTrue(c2 >= 1 && c2 <= 5, "chunk must be in [1, nb], was " + c2);
        } finally {
            GpuBatchContextCache.forceChunkSizeForTests(null);
        }
    }
}
