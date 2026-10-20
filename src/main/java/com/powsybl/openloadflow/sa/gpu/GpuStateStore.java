/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.PowsyblException;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The per-contingency device states of one security analysis, kept in a temporary file instead of the heap.
 *
 * <p>The precomputer holds every batched contingency's post-contingency state (one {@code double[n]}) until the
 * analysis ends, because the violation screen decides on all of them at once and each contingency's result is
 * injected from its state later in OLF's own loop. On network.xiidm (n about 35k) that is 280 KB per
 * contingency: 3.6 GB for the 12918 busbar sections and 10.4 GB for the all-kinds-plus-busbar list, which the
 * heap of this machine cannot hold next to the rest of the run. This map writes each state once, at the slot
 * its key was first seen at, and reads it back on demand; the file lives in the page cache as far as memory
 * allows and on disk beyond that. Keys are insertion-ordered. Single-writer use, as the GPU security analysis
 * is host-single-threaded; reads and writes are serialized all the same.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuStateStore extends AbstractMap<String, double[]> implements AutoCloseable {

    private final int n;
    private final Path file;
    private final FileChannel channel;
    private final Map<String, Integer> slotOf = new LinkedHashMap<>();
    private final ByteBuffer stateBuffer;
    private boolean closed;

    private GpuStateStore(int n) throws IOException {
        this.n = n;
        this.file = Files.createTempFile("olf-gpu-states-", ".bin");
        this.file.toFile().deleteOnExit();
        this.channel = FileChannel.open(file, StandardOpenOption.READ, StandardOpenOption.WRITE);
        this.stateBuffer = ByteBuffer.allocateDirect(n * Double.BYTES).order(ByteOrder.nativeOrder());
    }

    /** A store for states of {@code n} entries each. */
    public static GpuStateStore create(int n) {
        try {
            return new GpuStateStore(n);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Above this many bytes of states the precomputer stores them here rather than in a heap map: an eighth of
     *  the heap the JVM may grow to (2 GiB at the 16 GB heap of the scale runs). */
    public static long thresholdBytes() {
        return Runtime.getRuntime().maxMemory() / 8;
    }

    /** Whether {@code count} states of {@code n} entries belong here rather than in a heap map. */
    public static boolean wanted(int count, int n) {
        return wanted(count, n, thresholdBytes());
    }

    static boolean wanted(int count, int n, long thresholdBytes) {
        return (long) count * n * Double.BYTES > thresholdBytes;
    }

    public int stateLength() {
        return n;
    }

    @Override
    public synchronized double[] put(String key, double[] state) {
        Objects.requireNonNull(key);
        if (state.length != n) {
            throw new PowsyblException("GPU state store: a state of " + state.length + " entries for a store of " + n);
        }
        checkOpen();
        Integer slot = slotOf.get(key);
        double[] previous = null;
        if (slot == null) {
            slot = slotOf.size();
            slotOf.put(key, slot);
        } else {
            previous = read(slot);
        }
        stateBuffer.clear();
        stateBuffer.asDoubleBuffer().put(state);
        stateBuffer.limit(n * Double.BYTES);
        try {
            long offset = (long) slot * n * Double.BYTES;
            while (stateBuffer.hasRemaining()) {
                offset += channel.write(stateBuffer, offset);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return previous;
    }

    @Override
    public synchronized double[] get(Object key) {
        checkOpen();
        Integer slot = slotOf.get(key);
        return slot == null ? null : read(slot);
    }

    @Override
    public synchronized boolean containsKey(Object key) {
        return slotOf.containsKey(key);
    }

    @Override
    public synchronized int size() {
        return slotOf.size();
    }

    @Override
    public synchronized Set<String> keySet() {
        return Collections.unmodifiableSet(slotOf.keySet());
    }

    /** The entries in insertion order, each state read when the iterator reaches it. */
    @Override
    public Set<Map.Entry<String, double[]>> entrySet() {
        return new AbstractSet<>() {
            @Override
            public Iterator<Map.Entry<String, double[]>> iterator() {
                Iterator<String> keys = keySet().iterator();
                return new Iterator<>() {
                    @Override
                    public boolean hasNext() {
                        return keys.hasNext();
                    }

                    @Override
                    public Map.Entry<String, double[]> next() {
                        String key = keys.next();
                        return new SimpleImmutableEntry<>(key, get(key));
                    }
                };
            }

            @Override
            public int size() {
                return GpuStateStore.this.size();
            }
        };
    }

    private double[] read(int slot) {
        stateBuffer.clear();
        stateBuffer.limit(n * Double.BYTES);
        try {
            long offset = (long) slot * n * Double.BYTES;
            while (stateBuffer.hasRemaining()) {
                int r = channel.read(stateBuffer, offset);
                if (r < 0) {
                    throw new PowsyblException("GPU state store: state " + slot + " is missing from " + file);
                }
                offset += r;
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        stateBuffer.flip();
        double[] state = new double[n];
        stateBuffer.asDoubleBuffer().get(state);
        return state;
    }

    private void checkOpen() {
        if (closed) {
            throw new PowsyblException("GPU state store: closed");
        }
    }

    /** Drops the file. The map is unusable afterwards. */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            channel.close();
            Files.deleteIfExists(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
