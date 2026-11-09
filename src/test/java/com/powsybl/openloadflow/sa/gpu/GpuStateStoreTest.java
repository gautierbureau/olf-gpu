/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.PowsyblException;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The disk-backed state map behaves as the heap map it replaces: every state comes back bit for bit, a key written
 * twice holds its last state, the keys keep their insertion order, and the entries can be walked once in that
 * order (the violation screen's pass over every state).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuStateStoreTest {

    private static double[] state(int n, int seed) {
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = Math.sin(seed + 0.37 * i) * 1e3 + seed;
        }
        return s;
    }

    @Test
    void statesComeBackBitForBitInInsertionOrder() {
        int n = 1001;
        try (GpuStateStore store = GpuStateStore.create(n)) {
            assertTrue(store.isEmpty());
            for (int k = 0; k < 7; k++) {
                assertNull(store.put("c" + k, state(n, k)));
            }
            assertEquals(7, store.size());
            assertTrue(store.containsKey("c3"));
            assertFalse(store.containsKey("c9"));
            assertNull(store.get("c9"));
            for (int k = 0; k < 7; k++) {
                assertArrayEquals(state(n, k), store.get("c" + k), 0.0, "state c" + k);
            }
            // a key written twice keeps its slot and its last state
            double[] previous = store.put("c3", state(n, 42));
            assertArrayEquals(state(n, 3), previous, 0.0);
            assertArrayEquals(state(n, 42), store.get("c3"), 0.0);
            assertEquals(7, store.size());
            assertEquals(List.of("c0", "c1", "c2", "c3", "c4", "c5", "c6"), new ArrayList<>(store.keySet()));
            List<String> walked = new ArrayList<>();
            for (Map.Entry<String, double[]> e : store.entrySet()) {
                walked.add(e.getKey());
                assertArrayEquals(state(n, e.getKey().equals("c3") ? 42 : Integer.parseInt(e.getKey().substring(1))),
                        e.getValue(), 0.0);
            }
            assertEquals(new ArrayList<>(store.keySet()), walked);
            assertThrows(PowsyblException.class, () -> store.put("bad", new double[n - 1]));
        }
    }

    @Test
    void statesReadIntoABufferLandAtTheirOffsetBitForBit() {
        int n = 1001;
        try (GpuStateStore store = GpuStateStore.create(n)) {
            for (int k = 0; k < 5; k++) {
                store.put("c" + k, state(n, k));
            }
            // the device screen's chunk: scenario i at byte i * n * 8, native order, read in place
            ByteBuffer chunk = ByteBuffer.allocateDirect(3 * n * Double.BYTES).order(ByteOrder.nativeOrder());
            String[] order = {"c4", "c1", "c3"};
            for (int i = 0; i < order.length; i++) {
                assertTrue(store.readInto(order[i], chunk, i * n * Double.BYTES));
            }
            assertFalse(store.readInto("c9", chunk, 0));
            DoubleBuffer doubles = chunk.asDoubleBuffer();
            for (int i = 0; i < order.length; i++) {
                double[] read = new double[n];
                doubles.get(i * n, read);
                assertArrayEquals(store.get(order[i]), read, 0.0, "scenario " + i);
            }
        }
    }

    @Test
    void thresholdDecidesTheBacking() {
        assertFalse(GpuStateStore.wanted(100, 1000, 2L << 30), "100 states of 8 KB stay on the heap");
        assertTrue(GpuStateStore.wanted(40_000, 35_000, 2L << 30), "37k states of 280 KB go to the file");
        GpuStateStore store = GpuStateStore.create(10);
        store.put("a", new double[10]);
        store.close();
        assertThrows(PowsyblException.class, () -> store.get("a"));
    }
}
