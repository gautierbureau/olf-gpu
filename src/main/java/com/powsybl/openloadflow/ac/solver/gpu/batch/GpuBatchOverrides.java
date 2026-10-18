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
public final class GpuBatchOverrides {

    private final TIntArrayList index = new TIntArrayList();
    private final TDoubleArrayList value = new TDoubleArrayList();
    /** flat row -> position of its LAST entry. -1 = absent. */
    private final TIntIntHashMap positionOf = new TIntIntHashMap(64, 0.5f, -1, -1);

    /** Bumped by {@link #clear} and {@link #restore}: a snapshot taken in an earlier generation cannot assume the
     *  entries before it are still in place. */
    private int generation;
    /** Adds of a row that already had an entry (its value was replaced in place). */
    private int duplicateAppends;
    /** Of those, the adds whose value DIFFERED from the entry they replaced: two producers disagreeing on one
     *  row's target. Counted on every run — the chunk solver names the chunk when it is non-zero. */
    private int conflictCount;

    /** OLF_GPU_OV_DUP_TRACE: measure the invariant this class states in its own javadoc — a row must
     *  carry at most ONE override, because applyTargetOverrideBatch is one thread per entry writing
     *  target[ovIndex[k]] and two entries for one row RACE. A duplicate whose value EQUALS the entry
     *  it shadows is benign (either write gives the same target); one with a different value is a
     *  nondeterministic target. Off by default: add() runs millions of times per chunk. */
    private static final boolean DUP_TRACE = System.getenv("OLF_GPU_OV_DUP_TRACE") != null;
    /** Distinct flat rows that got a second entry, and those where it CONFLICTED (different value). */
    private final gnu.trove.set.hash.TIntHashSet dupRows = new gnu.trove.set.hash.TIntHashSet();
    private final gnu.trove.set.hash.TIntHashSet conflictRows = new gnu.trove.set.hash.TIntHashSet();
    /** The first few conflicts, as {flat, shadowedValue, newValue} — enough to name them. */
    private final java.util.List<double[]> conflictSamples = new java.util.ArrayList<>();
    /** ...and, per sample, the call site of the SHADOWING write. Only one of the two producers, but
     *  naming one is what lets the pair be read off the code instead of guessed. Captured only for a
     *  conflict (at most 40 per chunk), never on the 3.4M-entry hot path. */
    private final java.util.List<String> conflictSites = new java.util.ArrayList<>();

    /** A copy of the entries, to put back with {@link #restore}. */
    record Snapshot(int[] index, double[] value, int generation, int duplicateAppends) {
    }

    /** Drop every entry — and the index with it, which is the point of this class. */
    void clear() {
        index.resetQuick();
        value.resetQuick();
        positionOf.clear();
        generation++;
        conflictCount = 0;
        if (DUP_TRACE) {
            dupRows.clear();
            conflictRows.clear();
            conflictSamples.clear();
            conflictSites.clear();
        }
    }

    private static final java.util.Set<Integer> WATCH = System.getenv("OLF_GPU_OV_WATCH") == null ? java.util.Set.of()
            : java.util.Arrays.stream(System.getenv("OLF_GPU_OV_WATCH").split(",")).map(String::trim)
                    .map(Integer::valueOf).collect(java.util.stream.Collectors.toUnmodifiableSet());

    /** Adds since the last {@link #clear} that replaced an existing entry with a DIFFERENT value — a target two
     *  producers disagreed on, resolved deterministically to the later one. Zero on a sound plan. */
    int conflictingAddCount() {
        return conflictCount;
    }

    /** Add an override for {@code flat}. A row that already has an entry gets its VALUE REPLACED in place —
     *  "last one wins", decided HERE and not by the device. This used to append a second entry, and
     *  applyTargetOverrideBatch (one thread per entry writing {@code target[ovIndex[k]]}) then picked the
     *  survivor by scheduling: rte6515 BUS-2640_BBS_1's re-homed voltage identity (1.034) on row 5956 lost to a
     *  DISTR_Q seed of 0.0 in the 6667-contingency batch (slot 1844, 193.85 MW of distributed power and
     *  +197 violations against OLF) and won in a 3-contingency batch (slot 0, bit-identical to OLF). The
     *  entries handed to the device now never hold a row twice; a conflicting duplicate is still COUNTED
     *  (always, one compare on a rare path) so the chunk solver can name it. */
    public void add(int flat, double v) {
        if (WATCH.contains(flat)) {                          // OLF_GPU_OV_WATCH=<flat>[,<flat>]: who writes it
            System.err.println("OV_WATCH flat=" + flat + " v=" + v + " at " + java.util.Arrays.stream(new Throwable().getStackTrace())
                    .filter(f -> f.getClassName().startsWith("com.powsybl.openloadflow")).limit(4)
                    .map(f -> f.getFileName() + ":" + f.getLineNumber()).reduce((a, b) -> a + "<" + b).orElse("?"));
        }
        int prev = positionOf.get(flat);
        if (prev >= 0) {
            duplicateAppends++;
            double shadowed = value.getQuick(prev);
            boolean conflict = Double.compare(shadowed, v) != 0;
            if (conflict) {
                conflictCount++;
            }
            if (DUP_TRACE) {
                dupRows.add(flat);
                if (conflict && conflictRows.add(flat) && conflictSamples.size() < 40) {
                    conflictSamples.add(new double[] {flat, shadowed, v});
                    conflictSites.add(java.util.Arrays.stream(new Throwable().getStackTrace())
                            .filter(f -> f.getClassName().startsWith("com.powsybl.openloadflow"))
                            .limit(4)
                            .map(f -> f.getFileName() + ":" + f.getLineNumber())
                            .reduce((a, b) -> a + "<" + b).orElse("?"));
                }
            }
            value.setQuick(prev, v);
            return;
        }
        positionOf.put(flat, index.size());
        index.add(flat);
        value.add(v);
    }

    /** Does {@code flat} already carry an override? */
    boolean has(int flat) {
        return positionOf.get(flat) >= 0;
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

    /** Is the duplicate measurement on? (OLF_GPU_OV_DUP_TRACE) */
    static boolean dupTrace() {
        return DUP_TRACE;
    }

    /** Distinct rows that carry more than one override entry in the CURRENT chunk. */
    int duplicateRowCount() {
        return dupRows.size();
    }

    /** Of those, the rows whose duplicate entries disagree — a target the device picks by a race. */
    int conflictingRowCount() {
        return conflictRows.size();
    }

    /** Up to 40 samples of the disagreements, as {flat, shadowedValue, newValue}. */
    java.util.List<double[]> conflictSamples() {
        return conflictSamples;
    }

    /** Per sample, the call site of the shadowing write (innermost openloadflow frames). */
    java.util.List<String> conflictSites() {
        return conflictSites;
    }

    int[] indexArray() {
        return index.toArray();
    }

    double[] valueArray() {
        return value.toArray();
    }
}
