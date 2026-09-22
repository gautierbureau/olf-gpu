/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntIntHashMap;

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
 * <p>So the index lives here and every write goes through this class: {@link #clear()} clears it with the
 * entries, {@link #add} keeps it current (a later entry for the same row wins, as a rebuild would decide),
 * and {@link #restore} puts a snapshot back — rebuilding the index only when it cannot prove it need not. Storage is primitive (trove): a chunk holds millions of overrides, and
 * the boxed {@code List<Integer>} / {@code List<Double>} it replaced — with full boxed copies for snapshots
 * and an index rebuilt on EVERY lookup once a row appeared twice — cost ~0.6 s of a cold rte6515 run.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchOverrides {

    private final TIntArrayList index = new TIntArrayList();
    private final TDoubleArrayList value = new TDoubleArrayList();
    /** flat row -> position of its LAST entry. -1 = absent. */
    private final TIntIntHashMap positionOf = new TIntIntHashMap(64, 0.5f, -1, -1);

    /** Bumped by {@link #clear} and {@link #restore}: a snapshot taken in an earlier generation cannot assume the
     *  entries before it are still in place. */
    private int generation;
    /** Appends of a row that already had an entry (the index then forgets the earlier position). */
    private int duplicateAppends;

    /** A copy of the entries, to put back with {@link #restore}. */
    record Snapshot(int[] index, double[] value, int generation, int duplicateAppends) {
    }

    /** Drop every entry — and the index with it, which is the point of this class. */
    void clear() {
        index.resetQuick();
        value.resetQuick();
        positionOf.clear();
        generation++;
    }

    /** Append an override for {@code flat} (the caller guarantees the row is not already there,
     *  or accepts that a later {@link #compose} resolves to this newer entry — the same
     *  "last one wins" a {@code lastIndexOf} scan gave). */
    void add(int flat, double v) {
        if (positionOf.put(flat, index.size()) >= 0) {
            duplicateAppends++;
        }
        index.add(flat);
        value.add(v);
    }

    /** Add {@code delta} to the row's existing override, or create it at {@code base + delta}. */
    void compose(int flat, double base, double delta) {
        int at = positionOf.get(flat);
        if (at >= 0) {
            value.setQuick(at, value.getQuick(at) + delta);
        } else {
            add(flat, base + delta);
        }
    }

    /** Set the row's override to {@code v} outright, replacing any existing entry. */
    void put(int flat, double v) {
        int at = positionOf.get(flat);
        if (at >= 0) {
            value.setQuick(at, v);
        } else {
            add(flat, v);
        }
    }

    /** The HVDC freeze's update: the row's current override (or {@code baseTarget}) minus {@code baseD} plus
     *  {@code d}, in that order — the arithmetic its list-based form used. */
    void merge(int flat, double baseTarget, double baseD, double d) {
        int at = positionOf.get(flat);
        double from = at >= 0 ? value.getQuick(at) : baseTarget;
        if (at >= 0) {
            value.setQuick(at, from - baseD + d);
        } else {
            add(flat, from - baseD + d);
        }
    }

    int size() {
        return index.size();
    }

    int indexAt(int k) {
        return index.getQuick(k);
    }

    double valueAt(int k) {
        return value.getQuick(k);
    }

    Snapshot snapshot() {
        return new Snapshot(index.toArray(), value.toArray(), generation, duplicateAppends);
    }

    /**
     * Replace the entries with a snapshot's. When nothing but value updates and appends of NEW rows happened since
     * the snapshot — the HVDC freeze between its snapshot and its restore — the entries before it are still in
     * place: drop the appended rows from the index, truncate, and put the values back, instead of re-inserting
     * millions of entries into the index. Otherwise (a clear, a restore, or a row appended twice since) rebuild.
     */
    void restore(Snapshot snapshot) {
        int size = snapshot.index().length;
        if (snapshot.generation() == generation && snapshot.duplicateAppends() == duplicateAppends && index.size() >= size) {
            for (int k = size; k < index.size(); k++) {
                positionOf.remove(index.getQuick(k));
            }
            index.remove(size, index.size() - size);
            value.remove(size, value.size() - size);
            value.set(0, snapshot.value());
            generation++;
            return;
        }
        clear();
        for (int k = 0; k < size; k++) {
            add(snapshot.index()[k], snapshot.value()[k]);
        }
    }

    int[] indexArray() {
        return index.toArray();
    }

    double[] valueArray() {
        return value.toArray();
    }
}
