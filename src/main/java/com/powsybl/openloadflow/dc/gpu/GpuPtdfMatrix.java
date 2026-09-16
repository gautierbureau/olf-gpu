/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.equations.DcEquationType;
import com.powsybl.openloadflow.dc.equations.DcVariableType;
import com.powsybl.openloadflow.equations.Equation;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The full PTDF matrix, assembled on the device and kept there: one row per monitored branch, one
 * column per injection bus, the sensitivity of that branch's active power flow to an injection at
 * that bus, the slack bus absorbing.
 *
 * <p>Producing PTDF factors as a PRODUCT does not pay on a GPU - three loads and three flops each,
 * then the whole matrix over PCIe - and {@code docs/numbers.md} has the measurements. This class is
 * for the other case: when the factors are an INPUT to something that also runs on the device, and
 * the matrix should never cross the bus at all. It is assembled column-block by column-block and
 * stays resident; {@link #readBlock} and {@link #toDenseMatrix} exist for tests and for callers that
 * really do want it back.
 *
 * <p>Convention. A column is the response to a unit injection at one bus, in per unit. The slack
 * bus carries no active power equation, so it absorbs, which makes these the ordinary PTDF factors
 * against that slack. The buses given to {@link #assemble} are taken in the order given, and a bus
 * without an active power equation - the slack - is rejected rather than silently dropped. A branch
 * without a closed DC flow equation gets a zero row, the same value {@code calculateSensi} gives it.
 *
 * <p>Size. Rows times columns doubles on the device, on top of the states that produced them, which
 * are the bus count times the equation count. For a 6500-bus network with every branch monitored
 * that is about 450 MB of factors on top of 340 MB of states, so a caller with a smaller card should
 * pass a subset of buses and assemble more than once.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuPtdfMatrix implements AutoCloseable {

    private final GpuDcLoadFlowEngine engine;
    private final boolean ownsEngine;
    private final List<LfBranch> branches;
    private final List<LfBus> buses;

    private GpuPtdfMatrix(GpuDcLoadFlowEngine engine, boolean ownsEngine, List<LfBranch> branches, List<LfBus> buses) {
        this.engine = engine;
        this.ownsEngine = ownsEngine;
        this.branches = branches;
        this.buses = buses;
    }

    public static boolean isAvailable() {
        return GpuDcSolver.isAvailable();
    }

    /**
     * Assembles the PTDF of these branches against these buses on a context of its own.
     *
     * @throws IllegalStateException when the device is not available
     */
    public static GpuPtdfMatrix assemble(DcLoadFlowContext loadFlowContext, List<LfBranch> monitoredBranches,
                                         List<LfBus> injectionBuses) {
        Objects.requireNonNull(loadFlowContext);
        if (!isAvailable()) {
            throw new IllegalStateException("GPU native library not available");
        }
        GpuDcLoadFlowEngine engine = new GpuDcLoadFlowEngine(loadFlowContext);
        try {
            engine.prepare();
            return assemble(engine, monitoredBranches, injectionBuses, true);
        } catch (RuntimeException e) {
            engine.close();
            throw e;
        }
    }

    /** Same, on an engine the caller already prepared and keeps ownership of. */
    public static GpuPtdfMatrix assemble(GpuDcLoadFlowEngine engine, List<LfBranch> monitoredBranches,
                                         List<LfBus> injectionBuses) {
        return assemble(engine, monitoredBranches, injectionBuses, false);
    }

    private static GpuPtdfMatrix assemble(GpuDcLoadFlowEngine engine, List<LfBranch> monitoredBranches,
                                          List<LfBus> injectionBuses, boolean ownsEngine) {
        Objects.requireNonNull(engine);
        List<LfBranch> branches = List.copyOf(monitoredBranches);
        List<LfBus> buses = List.copyOf(injectionBuses);
        if (branches.isEmpty() || buses.isEmpty()) {
            throw new IllegalArgumentException("a PTDF needs at least one branch and one bus");
        }
        EquationSystem<DcVariableType, DcEquationType> equationSystem = engine.getContext().getEquationSystem();

        // one column per bus, one entry per column: the unit injection on that bus's active power row.
        // This is the right-hand side OLF builds for an injection factor, minus the participation of a
        // distributed slack, which a caller wanting those can apply to the columns afterwards.
        int[] columnPointers = new int[buses.size() + 1];
        int[] rows = new int[buses.size()];
        double[] values = new double[buses.size()];
        for (int i = 0; i < buses.size(); i++) {
            LfBus bus = buses.get(i);
            // the slack bus HAS an active power equation, deactivated - OLF's way of making it absorb -
            // and an inactive equation carries no column, so the check is on the column, not on presence
            rows[i] = equationSystem.getEquation(bus.getNum(), DcEquationType.BUS_TARGET_P)
                    .filter(Equation::isActive)
                    .map(Equation::getColumn)
                    .filter(column -> column >= 0)
                    .orElseThrow(() -> new IllegalArgumentException("bus '" + bus.getId()
                            + "' carries no active power row, so it cannot take an injection factor"
                            + " (the slack bus is the usual reason: it absorbs)"));
            values[i] = 1d;
            columnPointers[i + 1] = i + 1;
        }
        GpuDcSolver solver = engine.getSolver();
        solver.sensiSetFactorColumns(columnPointers, rows, values, buses.size(), false);
        solver.ptdfAssemble(branches.stream().mapToInt(LfBranch::getNum).toArray());
        return new GpuPtdfMatrix(engine, ownsEngine, branches, buses);
    }

    public int getRowCount() {
        return branches.size();
    }

    public int getColumnCount() {
        return buses.size();
    }

    public List<LfBranch> getBranches() {
        return branches;
    }

    public List<LfBus> getBuses() {
        return buses;
    }

    /** Bytes the resident matrix occupies on the device. */
    public long getDeviceBytes() {
        return (long) getRowCount() * getColumnCount() * Double.BYTES;
    }

    /**
     * Copies {@code cols} columns starting at {@code col0} into {@code out}, column-major: the value
     * for branch {@code row} of column {@code col0 + j} is at {@code out[row + j * rowCount]}.
     */
    public void readBlock(int col0, int cols, double[] out) {
        if (col0 < 0 || cols < 0 || col0 + cols > getColumnCount()) {
            throw new IndexOutOfBoundsException("columns " + col0 + " to " + (col0 + cols) + " outside " + getColumnCount());
        }
        if ((long) out.length < (long) getRowCount() * cols) {
            throw new IllegalArgumentException("out holds " + out.length + " values, the block needs "
                    + (long) getRowCount() * cols);
        }
        engine.getSolver().ptdfBlock(col0, cols, out);
    }

    /** The whole matrix as a dense one. Only for a matrix that fits: see {@link #getDeviceBytes}. */
    public DenseMatrix toDenseMatrix() {
        long elements = (long) getRowCount() * getColumnCount();
        if (elements > DenseMatrix.MAX_ELEMENT_COUNT) {
            throw new IllegalStateException("PTDF is too large for a DenseMatrix (" + getRowCount() + " x "
                    + getColumnCount() + "): read it in blocks");
        }
        DenseMatrix matrix = new DenseMatrix(getRowCount(), getColumnCount());
        double[] block = new double[getRowCount()];
        for (int column = 0; column < getColumnCount(); column++) {
            readBlock(column, 1, block);
            for (int row = 0; row < getRowCount(); row++) {
                matrix.set(row, column, block[row]);
            }
        }
        return matrix;
    }

    /** The branches a PTDF can be built for: those with a closed DC flow equation. */
    public static List<LfBranch> flowingBranches(LfNetwork network) {
        List<LfBranch> branches = new ArrayList<>();
        for (LfBranch branch : network.getBranches()) {
            if (branch.getBus1() != null && branch.getBus2() != null && !branch.isZeroImpedance(LoadFlowModel.DC)) {
                branches.add(branch);
            }
        }
        return branches;
    }

    /** The buses an injection factor can be taken at: every bus but the slack, which absorbs. */
    public static List<LfBus> injectionBuses(LfNetwork network) {
        List<LfBus> buses = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            if (!bus.isDisabled() && !bus.isSlack()) {
                buses.add(bus);
            }
        }
        return buses;
    }

    @Override
    public void close() {
        if (ownsEngine) {
            engine.close();
        }
    }
}
