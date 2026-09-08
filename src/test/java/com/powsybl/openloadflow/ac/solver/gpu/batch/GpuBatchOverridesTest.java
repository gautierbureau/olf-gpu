/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The invariants the batched SA's per-chunk override list must hold. These are the ones a
 * network-level test cannot pin: the batch solver reuses ONE list across chunks, and no case small
 * enough for the suite spans a chunk boundary with chunk shapes different enough to expose a
 * desynchronised row index — the real failure took 4 chunks of 643 scenarios on a 6.5k-bus network
 * (IndexOutOfBounds on the second chunk, or, in range, a silently rewritten override).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchOverridesTest {

    @Test
    void composeReplacesInPlaceAndNeverDuplicatesARow() {
        GpuBatchOverrides ov = new GpuBatchOverrides();
        ov.add(10, 1.0);
        ov.add(20, 2.0);
        ov.compose(10, 99.0, 0.5);                           // existing row: add to it, ignore base
        ov.compose(30, 7.0, 0.25);                           // new row: base + delta

        assertArrayEquals(new int[] {10, 20, 30}, ov.indexArray(),
                "a row must appear exactly once — the device scatter has no ordering");
        assertArrayEquals(new double[] {1.5, 2.0, 7.25}, ov.valueArray(), 0.0);
    }

    @Test
    void putReplacesTheRowsValue() {
        GpuBatchOverrides ov = new GpuBatchOverrides();
        ov.add(10, 1.0);
        ov.put(10, 4.0);
        ov.put(11, 5.0);
        assertArrayEquals(new int[] {10, 11}, ov.indexArray());
        assertArrayEquals(new double[] {4.0, 5.0}, ov.valueArray(), 0.0);
    }

    /** THE regression: the index must die with the entries. A chunk-1 position resolved against a
     *  shorter chunk-2 list threw, or silently rewrote a different row. */
    @Test
    void clearDropsTheRowIndexWithTheEntries() {
        GpuBatchOverrides ov = new GpuBatchOverrides();
        for (int row = 0; row < 40; row++) {                 // a "long" first chunk
            ov.add(row, row);
        }
        ov.clear();

        ov.add(500, 0.5);                                    // a "short" second chunk...
        ov.compose(7, 3.0, 0.5);                             // ...composing a row chunk 1 had held

        assertArrayEquals(new int[] {500, 7}, ov.indexArray(),
                "a row from the cleared chunk must be a NEW entry, not a stale position");
        assertArrayEquals(new double[] {0.5, 3.5}, ov.valueArray(), 0.0);
    }

    /** Appends made through the raw lists (the orchestrators' pre-solve hooks take plain lists)
     *  must not leave the index believing stale positions. */
    @Test
    void appendsThroughTheRawListsInvalidateTheIndex() {
        GpuBatchOverrides ov = new GpuBatchOverrides();
        ov.add(10, 1.0);

        ov.mutableIndexList().add(20);                       // bypasses add()
        ov.mutableValueList().add(2.0);

        ov.compose(20, 100.0, 1.0);                          // must find the row appended behind our back
        assertArrayEquals(new int[] {10, 20}, ov.indexArray());
        assertArrayEquals(new double[] {1.0, 3.0}, ov.valueArray(), 0.0,
                "composing onto an externally appended row must add to it, not re-create it");
    }

    @Test
    void readOnlyViewsCannotBeMutated() {
        GpuBatchOverrides ov = new GpuBatchOverrides();
        ov.add(1, 1.0);
        assertThrows(UnsupportedOperationException.class, () -> ov.indexList().add(2));
        assertThrows(UnsupportedOperationException.class, () -> ov.valueList().add(2.0));
        assertEquals(1, ov.indexArray().length);
    }
}
