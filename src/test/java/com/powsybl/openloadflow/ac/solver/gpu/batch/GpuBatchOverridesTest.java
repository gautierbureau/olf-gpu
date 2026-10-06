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

    /** A restored snapshot is the whole list again, with its index: a later composition finds its rows. */
    @Test
    void restoreRebuildsTheIndex() {
        GpuBatchOverrides ov = new GpuBatchOverrides();
        ov.add(10, 1.0);
        ov.add(20, 2.0);
        GpuBatchOverrides.Snapshot before = ov.snapshot();
        ov.compose(10, 0.0, 5.0);
        ov.add(30, 3.0);

        ov.restore(before);
        assertArrayEquals(new int[] {10, 20}, ov.indexArray());
        assertArrayEquals(new double[] {1.0, 2.0}, ov.valueArray(), 0.0);
        ov.compose(20, 100.0, 1.0);
        assertArrayEquals(new double[] {1.0, 3.0}, ov.valueArray(), 0.0, "composing onto a restored row adds to it");
    }

    /** The fast restore (value updates and appends of new rows since the snapshot) and the rebuild (a row appended
     *  twice since) must both give back exactly the snapshot, with a working index. */
    @Test
    void restoreAfterUpdatesAppendsAndDuplicates() {
        GpuBatchOverrides ov = new GpuBatchOverrides();
        ov.add(1, 1.0);
        ov.add(2, 2.0);
        GpuBatchOverrides.Snapshot s = ov.snapshot();
        ov.merge(1, 0.0, 0.5, 0.25);                         // update an existing row
        ov.merge(3, 3.0, 0.0, 1.0);                          // append a new row
        ov.restore(s);                                       // fast path
        assertArrayEquals(new int[] {1, 2}, ov.indexArray());
        assertArrayEquals(new double[] {1.0, 2.0}, ov.valueArray(), 0.0);
        ov.compose(3, 10.0, 1.0);                            // row 3 must be gone from the index
        assertArrayEquals(new int[] {1, 2, 3}, ov.indexArray());
        assertArrayEquals(new double[] {1.0, 2.0, 11.0}, ov.valueArray(), 0.0);

        GpuBatchOverrides.Snapshot t = ov.snapshot();
        ov.add(2, 9.0);                                      // row 2 added twice: replaced in place, counted as a duplicate
        ov.restore(t);                                       // rebuild path (a duplicate since the snapshot)
        ov.compose(2, 0.0, 1.0);
        assertArrayEquals(new int[] {1, 2, 3}, ov.indexArray());
        assertArrayEquals(new double[] {1.0, 3.0, 11.0}, ov.valueArray(), 0.0, "row 2 resolves to its original entry");
    }

    /** The HVDC freeze's merge: (existing override, else the base target) - baseD + d, in that order. */
    @Test
    void mergeUsesTheExistingOverrideOrTheBaseTarget() {
        GpuBatchOverrides ov = new GpuBatchOverrides();
        ov.add(7, 4.0);
        ov.merge(7, 100.0, 1.0, 0.5);                        // existing: 4 - 1 + 0.5
        ov.merge(8, 10.0, 2.0, 0.25);                        // new: 10 - 2 + 0.25
        assertArrayEquals(new int[] {7, 8}, ov.indexArray());
        assertArrayEquals(new double[] {4.0 - 1.0 + 0.5, 10.0 - 2.0 + 0.25}, ov.valueArray(), 0.0);
    }

    /** A row added twice keeps ONE entry holding the LAST value: the device must never see a row twice
     *  (applyTargetOverrideBatch would pick the survivor by scheduling — rte6515 BUS-2640_BBS_1, 193 MW), and
     *  the disagreement is counted so the chunk solver can name it. */
    @Test
    void aDuplicatedRowResolvesToItsLastEntry() {
        GpuBatchOverrides ov = new GpuBatchOverrides();
        ov.add(5, 1.0);
        ov.add(5, 2.0);
        ov.compose(5, 0.0, 1.0);
        assertArrayEquals(new int[] {5}, ov.indexArray());
        assertArrayEquals(new double[] {3.0}, ov.valueArray(), 0.0);
        assertEquals(1, ov.size());
        assertEquals(1, ov.conflictingAddCount(), "1.0 then 2.0 on one row is a conflict");
        ov.add(5, 3.0);                                      // same value again: a benign duplicate
        assertEquals(1, ov.conflictingAddCount());
        ov.clear();
        assertEquals(0, ov.conflictingAddCount());
    }
}
