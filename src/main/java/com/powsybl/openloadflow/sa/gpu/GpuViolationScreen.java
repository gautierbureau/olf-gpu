/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;

import java.util.Map;

/**
 * Screens a post-contingency state against the run's limit table and returns the CANDIDATES —
 * the limit entries a scenario may be violating.
 *
 * <p>It SCREENS, it does not decide. Two properties make that the right contract, and both exist so
 * that the same code can run on the device without the arithmetic deciding anything:
 * <ul>
 *   <li>the comparison uses a RELAXED threshold, so the screen can only ever emit MORE candidates
 *       than OLF would report, never fewer;</li>
 *   <li>it does NOT stop at the first exceeded limit of a group while that one is AMBIGUOUS — within the
 *       band, where the exact re-test may reject it — so the consumer is free to reject it and move on. It
 *       stops at the first limit exceeded BEYOND the band: that one is certain to confirm, and the consumer
 *       reports the first confirmed limit of a group and never looks further (this cut the candidates from
 *       ~415 to about the number of violations per contingency on rte6515).</li>
 * </ul>
 * The consumer ({@link GpuLimitScanner#confirm}) then re-tests each candidate with OLF's exact
 * {@code >} on OLF's own value and takes the first that confirms. So a candidate list computed in
 * double precision on a GPU, whose {@code sin}/{@code cos} and FMA contraction differ from the
 * host's in the last bits, cannot change a single reported violation as long as the band is wider
 * than that discrepancy — and the discrepancy is ~1e-16 relative while the band is 1e-9.
 *
 * <p>This Java implementation is not scaffolding: it is the fallback whenever the device path is
 * off or a branch is not in the device model, and it is the ORACLE the kernel is validated against —
 * the kernel must emit the same candidates, and the same reported violations must come out the far
 * end.
 *
 * <p>The flows are computed exactly as OLF's vectorized network computes them, from the same packed
 * element data the solver uploads: {@code p1/q1/p2/q2} from the closed/open branch formulas, then
 * {@code i1 = hypot(p1, q1) / v1} and {@code s1 = sqrt(p1² + q1²)} — note that OLF uses
 * {@code hypot} for the current and a plain {@code sqrt} for the apparent power, and the two are not
 * the same function in floating point.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuViolationScreen {

    /** Relative slack on the threshold, so the screen over-emits rather than ever missing a
     *  violation the host would report. Chosen ~7 orders of magnitude above the device-vs-host
     *  discrepancy it covers, and it costs only a handful of extra candidates per contingency. */
    static final double BAND = 1e-9;

    /** Where a branch's flows come from: the closed-branch pack, the open-branch pack, or nowhere
     *  (zero-impedance branches take their flows from dummy variables, and anything the extraction
     *  did not pack is left to the host scan). */
    private static final byte FAMILY_NONE = -1;
    private static final byte FAMILY_CLOSED = 0;
    private static final byte FAMILY_OPEN = 1;

    private final GpuLimitTable table;
    private final byte[] branchFamily;
    private final int[] branchElem;
    private final int[] busVRow;

    private final double[] closedBranchParameters;
    private final int[] closedBranchIndices;
    private final double[] openBranchParameters;
    private final int[] openBranchIndices;

    private GpuViolationScreen(GpuLimitTable table, byte[] branchFamily, int[] branchElem, int[] busVRow,
                               double[] closedBranchParameters, int[] closedBranchIndices, double[] openBranchParameters, int[] openBranchIndices) {
        this.table = table;
        this.branchFamily = branchFamily;
        this.branchElem = branchElem;
        this.busVRow = busVRow;
        this.closedBranchParameters = closedBranchParameters;
        this.closedBranchIndices = closedBranchIndices;
        this.openBranchParameters = openBranchParameters;
        this.openBranchIndices = openBranchIndices;
    }

    /**
     * Wire the table's branches and buses to their place in the packed element data.
     *
     * <p>{@code baseVariableRows} is the BASE row of every variable — the row space the precomputed
     * states are stored in (the injector remaps them into the contingency-mutated system), and the
     * same space the extraction's column indices live in. Mixing the two would read a bus's voltage
     * out of another bus's row, so both sides must come from the base snapshot.
     */
    static GpuViolationScreen build(LfNetwork network, GpuLimitTable table, GpuAcData data,
                                    GpuAcElementMapper mapper, EquationSystem<AcVariableType, ?> equationSystem,
                                    Map<Variable<AcVariableType>, Integer> baseVariableRows) {
        LfBranch[] branches = table.branches();
        byte[] family = new byte[branches.length];
        int[] elem = new int[branches.length];
        int closedCount = mapper.getClosedBranchCount();
        for (int b = 0; b < branches.length; b++) {
            int index = mapper.getElementIndex(branches[b]);
            if (index < 0) {
                family[b] = FAMILY_NONE;                     // zero-impedance / not packed — host scan
                elem[b] = -1;
            } else if (index < closedCount) {
                family[b] = FAMILY_CLOSED;
                elem[b] = index;
            } else {
                family[b] = FAMILY_OPEN;
                elem[b] = index - closedCount;
            }
            // Coverage is decided HERE, not at screen time. A branch whose state columns are absent
            // cannot be evaluated, and a screen that discovered that mid-flight would silently emit
            // no candidate for it — which reads as "no violation", not as "not screened". Marking it
            // uncovered instead sends it down the host scan, where it is detected normally.
            if (family[b] != FAMILY_NONE && !hasColumns(family[b], elem[b], data)) {
                family[b] = FAMILY_NONE;
                elem[b] = -1;
            }
        }

        LfBus[] buses = table.buses();
        int[] vRow = new int[buses.length];
        for (int i = 0; i < buses.length; i++) {
            Variable<AcVariableType> v = equationSystem.getVariableSet()
                    .getVariable(buses[i].getNum(), AcVariableType.BUS_V);
            Integer row = baseVariableRows.get(v);
            vRow[i] = row != null ? row : -1;                // absent at base (never a candidate)
        }
        return new GpuViolationScreen(table, family, elem, vRow, data.closedBranch().parameters(), data.closedBranch().indices(), data.openBranch().parameters(), data.openBranch().indices());
    }

    /** Every state column the flow evaluation reads is present at base. */
    private static boolean hasColumns(byte family, int elem, GpuAcData data) {
        if (family == FAMILY_CLOSED) {
            int[] indices = data.closedBranch().indices();
            int c = elem * 12;
            return indices[c + 4] >= 0 && indices[c + 5] >= 0 && indices[c + 6] >= 0 && indices[c + 7] >= 0;
        }
        int[] indices = data.openBranch().indices();
        int c = elem * 6;
        return indices[c + 4] >= 0 || indices[c + 5] >= 0;        // an open branch carries one side only
    }

    /** True when this branch's flows can be screened at all — a branch that cannot is scanned on
     *  the host, so coverage is a performance property, never a correctness one. */
    boolean covers(int branchIndex) {
        return branchFamily[branchIndex] != FAMILY_NONE;
    }

    /** Records per scenario the device buffer holds. A contingency on rte6515 raises ~160 candidates;
     *  a scenario that exceeds this is not wrong, it is simply scanned on the host instead. */
    private static final int DEVICE_CAPACITY = 1024;

    /** Scenarios per device call — bounds the state upload ({@code chunk * n} doubles) and the
     *  record buffer, and is well past the point where the kernel saturates. */
    private static final int DEVICE_CHUNK = 512;

    /**
     * Screen every scenario, on the device when it is available and on the host otherwise.
     *
     * <p>The device path is a pure accelerator of {@link #screen}: same table, same band, same
     * records. It uploads the states the host is about to inject anyway, so the screen and the
     * confirm see the same numbers, and it needs nothing from the solver's context — which keeps it
     * out of the batch's memory ceiling and independent of the orchestration that produced the state.
     */
    java.util.Map<String, int[]> screenAll(java.util.Map<String, double[]> states, int n) {
        java.util.Map<String, int[]> out = new java.util.HashMap<>();
        java.util.List<String> ids = new java.util.ArrayList<>(states.keySet());
        long table = deviceTable();
        if (table == 0) {
            for (String id : ids) {
                out.put(id, screen(states.get(id)));
            }
            return out;
        }
        onDevice = 0;
        // One direct buffer for every chunk, which the device reads in place: each state is written into it once (read
        // straight from the state store's file when the states are there), with no chunk array and no JNI copy of it.
        java.nio.ByteBuffer flat = java.nio.ByteBuffer.allocateDirect(Math.min(DEVICE_CHUNK, ids.size()) * n * Double.BYTES)
                .order(java.nio.ByteOrder.nativeOrder());
        try {
            for (int from = 0; from < ids.size(); from += DEVICE_CHUNK) {
                int to = Math.min(from + DEVICE_CHUNK, ids.size());
                int count = to - from;
                for (int i = 0; i < count; i++) {
                    writeState(states, ids.get(from + i), flat, i, n);
                }
                int[] result = ViolationScreenNative.screenViolationsBatch(table, flat, count, n, BAND, DEVICE_CAPACITY);
                if (result == null) {
                    for (int i = from; i < to; i++) {        // device refused the chunk — host screen
                        out.put(ids.get(i), screen(states.get(ids.get(i))));
                    }
                    continue;
                }
                onDevice += count;
                for (int i = 0; i < count; i++) {
                    String id = ids.get(from + i);
                    int emitted = result[i];
                    if (emitted > DEVICE_CAPACITY) {
                        overflows++;
                        out.put(id, screen(states.get(id)));  // more candidates than the buffer holds
                        continue;
                    }
                    int[] records = java.util.Arrays.copyOfRange(result, count + i * DEVICE_CAPACITY,
                            count + i * DEVICE_CAPACITY + emitted);
                    // The kernel's threads emit in whatever order they finish; sorting restores the
                    // table order the consumer relies on for "first exceeded of the group wins", and
                    // leaves the bus records (encoded above every limit index) at the end.
                    java.util.Arrays.sort(records);
                    out.put(id, records);
                }
            }
        } finally {
            ViolationScreenNative.destroyViolationTable(table);
            deviceTable = 0;
        }
        return out;
    }

    /** The state of {@code id} into scenario {@code i}'s {@code n} doubles of {@code flat}, zero past a shorter state. */
    private static void writeState(java.util.Map<String, double[]> states, String id, java.nio.ByteBuffer flat, int i, int n) {
        int at = i * n * Double.BYTES;
        if (states instanceof GpuStateStore store && store.stateLength() == n && store.readInto(id, flat, at)) {
            return;
        }
        double[] state = states.get(id);
        int length = Math.min(n, state.length);
        java.nio.DoubleBuffer into = flat.slice(at, n * Double.BYTES).order(java.nio.ByteOrder.nativeOrder()).asDoubleBuffer();
        into.put(state, 0, length);
        while (into.hasRemaining()) {
            into.put(0.0);
        }
    }

    /** Scenarios whose candidates did not fit the device buffer and were screened on the host. */
    int overflows() {
        return overflows;
    }

    /** Scenarios the DEVICE screened. Reported so that "no failures" can never be read as "the
     *  kernel never ran" — every fallback here is silent by design. */
    int onDevice() {
        return onDevice;
    }

    private long deviceTable;
    private int overflows;
    private int onDevice;

    /** Upload the table once; 0 means "no device path", which is a fallback, not a failure. */
    private long deviceTable() {
        if (deviceTable != 0) {
            return deviceTable;
        }
        if (!GpuAcNewtonSolver.isAvailable()) {
            return 0;
        }
        int nGroups = table.groupCount();
        int[] groupSide = new int[nGroups];
        int[] groupQuantity = new int[nGroups];
        int[] groupLimitStart = new int[nGroups];
        int[] groupLimitEnd = new int[nGroups];
        for (int g = 0; g < nGroups; g++) {
            groupSide[g] = table.side(g);
            groupQuantity[g] = table.quantity(g);
            groupLimitStart[g] = table.limitStart(g);
            groupLimitEnd[g] = table.limitEnd(g);
        }
        int nBranches = table.branches().length;
        int[] family = new int[nBranches];
        int[] groupStart = new int[nBranches];
        int[] groupEnd = new int[nBranches];
        for (int b = 0; b < nBranches; b++) {
            family[b] = branchFamily[b];
            groupStart[b] = table.groupStart(b);
            groupEnd[b] = table.groupEnd(b);
        }
        double[] thresholds = new double[table.limitCount()];
        for (int l = 0; l < thresholds.length; l++) {
            thresholds[l] = table.threshold(l);
        }
        int nBuses = table.buses().length;
        double[] busHigh = new double[nBuses];
        double[] busLow = new double[nBuses];
        for (int i = 0; i < nBuses; i++) {
            busHigh[i] = table.busHighLimit(i);
            busLow[i] = table.busLowLimit(i);
        }
        deviceTable = ViolationScreenNative.uploadViolationTable(family, branchElem, groupStart, groupEnd,
                groupSide, groupQuantity, groupLimitStart, groupLimitEnd, thresholds,
                busVRow, busHigh, busLow, closedBranchParameters, closedBranchIndices, openBranchParameters, openBranchIndices);
        return deviceTable;
    }

    /**
     * The limit entries this state may be violating, as limit indices in table order (so a
     * consumer walking them in order meets each group's entries most-severe-first, the way OLF's
     * own loop does), followed by the bus candidates.
     *
     * <p>A bus candidate is encoded ABOVE the limit indices, as
     * {@code limitCount + busIndex * 2 + (high ? 0 : 1)}, so that simply SORTING a record array
     * ascending restores table order and leaves the bus block at the end — which is what lets the
     * device kernel emit records in whatever order its threads finish.
     */
    int[] screen(double[] state) {
        int[] out = new int[64];
        int count = 0;
        LfBranch[] branches = table.branches();
        for (int b = 0; b < branches.length; b++) {
            int groupStart = table.groupStart(b);
            int groupEnd = table.groupEnd(b);
            if (groupStart == groupEnd || branchFamily[b] == FAMILY_NONE) {
                continue;
            }
            double p1 = Double.NaN;
            double q1 = Double.NaN;
            double p2 = Double.NaN;
            double q2 = Double.NaN;
            double v1 = Double.NaN;
            double v2 = Double.NaN;
            boolean computed = false;
            for (int g = groupStart; g < groupEnd; g++) {
                if (!computed) {
                    double[] flows = flows(b, state);
                    if (flows == null) {
                        break;                               // a side's variable is absent at base
                    }
                    p1 = flows[0];
                    q1 = flows[1];
                    p2 = flows[2];
                    q2 = flows[3];
                    v1 = flows[4];
                    v2 = flows[5];
                    computed = true;
                }
                byte side = table.side(g);
                byte quantity = table.quantity(g);
                double p = side == GpuTwoSides.ONE ? p1 : p2;
                double q = side == GpuTwoSides.ONE ? q1 : q2;
                double v = side == GpuTwoSides.ONE ? v1 : v2;
                double value = switch (quantity) {
                    case GpuLimitTable.QTY_I -> Math.hypot(p, q) / v;
                    case GpuLimitTable.QTY_P -> Math.abs(p);
                    default -> Math.sqrt(p * p + q * q);
                };
                if (Double.isNaN(value)) {
                    continue;
                }
                for (int l = table.limitStart(g); l < table.limitEnd(g); l++) {
                    double threshold = table.threshold(l);
                    if (value > threshold - Math.abs(threshold) * BAND) {
                        if (count == out.length) {
                            out = java.util.Arrays.copyOf(out, out.length * 2);
                        }
                        out[count++] = l;
                        if (value > threshold + Math.abs(threshold) * BAND) {
                            break;                           // certain to confirm: nothing after it is reported
                        }
                    }
                }
            }
        }
        for (int i = 0; i < busVRow.length; i++) {
            if (busVRow[i] < 0) {
                continue;
            }
            double v = state[busVRow[i]];
            double high = table.busHighLimit(i);
            double low = table.busLowLimit(i);
            boolean overHigh = v > high - Math.abs(high) * BAND;
            boolean underLow = v < low + Math.abs(low) * BAND;
            if (overHigh || underLow) {
                if (count + 2 > out.length) {
                    out = java.util.Arrays.copyOf(out, out.length * 2 + 2);
                }
                if (overHigh) {
                    out[count++] = table.limitCount() + i * 2;
                }
                if (underLow) {
                    out[count++] = table.limitCount() + i * 2 + 1;
                }
            }
        }
        return java.util.Arrays.copyOf(out, count);
    }

    /** The bus a bus-candidate record refers to. Only call on a record {@code >= limitCount}. */
    static int busOf(GpuLimitTable table, int record) {
        return (record - table.limitCount()) / 2;
    }

    /** True when a bus candidate is the HIGH voltage limit (else the low one). */
    static boolean isHighVoltage(GpuLimitTable table, int record) {
        return (record - table.limitCount()) % 2 == 0;
    }

    /** {p1, q1, p2, q2, v1, v2} at this state, or null when a needed variable is absent at base. */
    private double[] flows(int branchIndex, double[] state) {
        int e = branchElem[branchIndex];
        if (branchFamily[branchIndex] == FAMILY_CLOSED) {
            double[] parameters = closedBranchParameters;
            int[] indices = closedBranchIndices;
            int o = e * 8;
            int c = e * 12;
            int ph1Col = indices[c + 4];
            int v1Col = indices[c + 5];
            int ph2Col = indices[c + 6];
            int v2Col = indices[c + 7];
            if (ph1Col < 0 || v1Col < 0 || ph2Col < 0 || v2Col < 0) {
                return null;
            }
            double y = parameters[o];
            double g1 = parameters[o + 1];
            double g2 = parameters[o + 2];
            double b1 = parameters[o + 3];
            double b2 = parameters[o + 4];
            double ksi = parameters[o + 5];
            // A controlled ratio / phase shift is a STATE variable, so the scenario's own value is in
            // the state vector, not in the pack (which carries the base tap) — the fill kernels make
            // exactly this choice, and a screen that took the pack value would evaluate the wrong tap.
            double r1 = indices[c + 9] >= 0 ? state[indices[c + 9]] : parameters[o + 6];
            double a1 = indices[c + 8] >= 0 ? state[indices[c + 8]] : parameters[o + 7];
            double v1 = state[v1Col];
            double ph1 = state[ph1Col];
            double v2 = state[v2Col];
            double ph2 = state[ph2Col];
            double theta1 = ksi - a1 - ph1 + ph2;
            double theta2 = ksi + a1 + ph1 - ph2;
            double sinKsi = Math.sin(ksi);
            double cosKsi = Math.cos(ksi);
            double sinTheta1 = Math.sin(theta1);
            double cosTheta1 = Math.cos(theta1);
            double sinTheta2 = Math.sin(theta2);
            double cosTheta2 = Math.cos(theta2);
            return new double[] {
                r1 * v1 * (g1 * r1 * v1 + y * r1 * v1 * sinKsi - y * v2 * sinTheta1),
                r1 * v1 * (-b1 * r1 * v1 + y * r1 * v1 * cosKsi - y * v2 * cosTheta1),
                v2 * (g2 * v2 - y * r1 * v1 * sinTheta2 + y * v2 * sinKsi),
                v2 * (-b2 * v2 - y * r1 * v1 * cosTheta2 + y * v2 * cosKsi),
                v1, v2,
            };
        }
        double[] parameters = openBranchParameters;
        int[] indices = openBranchIndices;
        int o = e * 7;
        int c = e * 6;
        int v1Col = indices[c + 4];
        int v2Col = indices[c + 5];
        double y = parameters[o];
        double ksi = parameters[o + 1];
        double g1 = parameters[o + 2];
        double b1 = parameters[o + 3];
        double g2 = parameters[o + 4];
        double b2 = parameters[o + 5];
        double r1 = parameters[o + 6];
        double sinKsi = Math.sin(ksi);
        double cosKsi = Math.cos(ksi);
        // Each side's flow sees the OTHER side's shunt through the series admittance, so side 1's
        // denominator is built from g2/b2 and side 2's from g1/b1 — the open-branch terms' own shape.
        double den2 = (g1 + y * sinKsi) * (g1 + y * sinKsi) + (-b1 + y * cosKsi) * (-b1 + y * cosKsi);
        double den1 = (g2 + y * sinKsi) * (g2 + y * sinKsi) + (-b2 + y * cosKsi) * (-b2 + y * cosKsi);
        double p1 = Double.NaN;
        double q1 = Double.NaN;
        double p2 = Double.NaN;
        double q2 = Double.NaN;
        double v1 = Double.NaN;
        double v2 = Double.NaN;
        if (v1Col >= 0) {                                    // side 2 open: only side 1 carries a flow
            v1 = state[v1Col];
            p1 = r1 * r1 * v1 * v1 * (g1 + y * y * g2 / den1 + (b2 * b2 + g2 * g2) * y * sinKsi / den1);
            q1 = -r1 * r1 * v1 * v1 * (b1 + y * y * b2 / den1 - (b2 * b2 + g2 * g2) * y * cosKsi / den1);
        }
        if (v2Col >= 0) {                                    // side 1 open
            v2 = state[v2Col];
            p2 = v2 * v2 * (g2 + y * y * g1 / den2 + (b1 * b1 + g1 * g1) * y * sinKsi / den2);
            q2 = -v2 * v2 * (b2 + y * y * b1 / den2 - (b1 * b1 + g1 * g1) * y * cosKsi / den2);
        }
        return new double[] {p1, q1, p2, q2, v1, v2};
    }
}
