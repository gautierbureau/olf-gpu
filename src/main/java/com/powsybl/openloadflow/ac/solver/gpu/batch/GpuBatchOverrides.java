/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A per-scenario target-override list that OWNS its flat-row → position index.
 *
 * <p>Two properties matter here and neither survives keeping the pieces as loose locals:
 *
 * <ul>
 *   <li><b>A row must carry at most ONE override.</b> The device scatter has no ordering, so two
 *       entries for the same row race. Composition therefore has to find an existing entry, which
 *       needs a lookup — and the obvious {@code lastIndexOf} scan is quadratic when it runs once
 *       per (scenario, DS participant) over a list that grows to the same size (measured: 97 s of
 *       a 101 s batch on a 6.5k-bus network).</li>
 *   <li><b>The index must never outlive the list it indexes.</b> A hand-maintained map beside the
 *       list desynchronises the moment someone clears the list, appends without updating it, or
 *       reuses it for the next chunk — resolving stale positions into a shorter list, which either
 *       throws or, worse, silently rewrites another row's override.</li>
 * </ul>
 *
 * <p>So the index lives here, {@link #clear()} clears it with the entries by construction, and any
 * append made through the {@link #mutableIndexList() raw lists} (the orchestrator APIs take plain
 * lists) only marks the index DIRTY — it is rebuilt on the next lookup rather than trusted. The
 * failure mode is then unreachable rather than merely untested: the batch solver reuses one
 * instance across chunks and the small networks in the test suite never span a chunk boundary
 * deeply enough to expose a desync.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchOverrides {

    private final List<Integer> index = new ArrayList<>();
    private final List<Double> value = new ArrayList<>();
    /** flat row -> position; primitive (trove) because a chunk holds millions of overrides and the boxed
     *  HashMap was ~1 s of rte6515's precompute. -1 = absent. */
    private final gnu.trove.map.hash.TIntIntHashMap positionOf = new gnu.trove.map.hash.TIntIntHashMap(64, 0.5f, -1, -1);
    private boolean indexDirty;

    /** Drop every entry — and the index with it, which is the point of this class. */
    void clear() {
        index.clear();
        value.clear();
        positionOf.clear();
        indexDirty = false;
    }

    /** Append an override for {@code flat} (the caller guarantees the row is not already there,
     *  or accepts that a later {@link #compose} resolves to this newer entry — the same
     *  "last one wins" a {@code lastIndexOf} scan gave). */
    void add(int flat, double v) {
        positionOf.put(flat, index.size());
        index.add(flat);
        value.add(v);
    }

    /** Add {@code delta} to the row's existing override, or create it at {@code base + delta}. */
    void compose(int flat, double base, double delta) {
        ensureIndex();
        int at = positionOf.get(flat);
        if (at >= 0) {
            value.set(at, value.get(at) + delta);
        } else {
            add(flat, base + delta);
        }
    }

    /** Set the row's override to {@code v} outright, replacing any existing entry. */
    void put(int flat, double v) {
        ensureIndex();
        int at = positionOf.get(flat);
        if (at >= 0) {
            value.set(at, v);
        } else {
            add(flat, v);
        }
    }

    /** The rows, read-only. */
    List<Integer> indexList() {
        return Collections.unmodifiableList(index);
    }

    /** The values, read-only. */
    List<Double> valueList() {
        return Collections.unmodifiableList(value);
    }

    /** The raw rows for an API that appends to a plain list (the orchestrators' pre-solve hooks).
     *  Any such append invalidates the index, which is rebuilt on the next lookup. */
    List<Integer> mutableIndexList() {
        indexDirty = true;
        return index;
    }

    /** The raw values for the same APIs; see {@link #mutableIndexList()}. */
    List<Double> mutableValueList() {
        indexDirty = true;
        return value;
    }

    int[] indexArray() {
        int[] out = new int[index.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = index.get(i);
        }
        return out;
    }

    double[] valueArray() {
        double[] out = new double[value.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = value.get(i);
        }
        return out;
    }

    private void ensureIndex() {
        if (!indexDirty && positionOf.size() >= index.size()) {
            return;
        }
        positionOf.clear();
        for (int k = 0; k < index.size(); k++) {
            positionOf.put(index.get(k), k);                 // later duplicates win
        }
        indexDirty = false;
    }
}
