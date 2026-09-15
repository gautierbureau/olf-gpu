/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.equations.DcEquationType;
import com.powsybl.openloadflow.dc.equations.DcVariableType;
import com.powsybl.openloadflow.dc.fastdc.LodfResultWriter;
import com.powsybl.openloadflow.equations.Equation;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * {@code LodfCalculator} on the device: the LODF factors of a set of monitored branches for the
 * single outages of a set of branches, {@code LODF(l, k) = PTDF(l, k) / (1 - PTDF(k, k))}.
 *
 * <p>Nothing new is needed on the device for this: the "injection states" the CPU calculator solves
 * — the angle response to a {@code +1/-1} injection at the ends of each outaged branch — are
 * exactly the element states the Woodbury security analysis already keeps device-resident, and the
 * PTDF of a branch on a column is the same {@code calculateSensi} expression the connectivity
 * criterion already evaluates. So the whole calculator is: the same {@code +1/-1} right-hand sides,
 * one multi-column solve on one factorization, and a kernel over (monitored branch, outaged branch).
 *
 * <p>Conventions are the CPU calculator's, value for value: {@code -1} for a branch monitored under
 * its own outage, {@code NaN} for an outage that breaks connectivity (self-PTDF within
 * {@code 1e-6} of 1) and for a monitored branch without a closed DC flow equation.
 *
 * <p>What bounds the run is the result, not the algebra: a full N-1 of a 9000-branch network is
 * 82 million factors, 650 MB. The factors are therefore streamed — the outages in batches whose
 * states fit a device-memory budget, each batch in column blocks downloaded into a bounded host
 * buffer — and {@link #computeLodfMatrix} only materializes a matrix when the caller asks for one.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuLodfCalculator {

    /** Device memory budget for one batch of element states, in bytes (the CPU calculator's own cap). */
    private static long maxStateBytesPerBatch() {
        return Long.getLong("olf.gpu.dc.lodf.stateBytes", 256L * 1024 * 1024);
    }

    /** Host buffer budget for one downloaded block of factors, in bytes. */
    private static long maxBlockBytes() {
        return Long.getLong("olf.gpu.dc.lodf.blockBytes", 64L * 1024 * 1024);
    }

    private GpuLodfCalculator() {
    }

    public static boolean isAvailable() {
        return GpuDcSolver.isAvailable();
    }

    /**
     * The LODF matrix of the monitored branches for the single outages of the outaged branches, row
     * {@code l} for the {@code l}-th monitored branch and column {@code k} for the {@code k}-th
     * outaged branch.
     */
    public static DenseMatrix computeLodfMatrix(DcLoadFlowContext loadFlowContext, List<LfBranch> monitoredBranches,
                                                List<LfBranch> outagedBranches) {
        Objects.requireNonNull(monitoredBranches);
        Objects.requireNonNull(outagedBranches);
        long resultSize = (long) monitoredBranches.size() * outagedBranches.size();
        if (resultSize > DenseMatrix.MAX_ELEMENT_COUNT) {
            throw new PowsyblException("LODF matrix is too large (" + monitoredBranches.size() + " monitored branches x "
                    + outagedBranches.size() + " outaged branches = " + resultSize + " elements, maximum is "
                    + DenseMatrix.MAX_ELEMENT_COUNT + "): use computeLodf with a streaming LodfResultWriter, or split the "
                    + "monitored or the outaged branches into groups");
        }
        DenseMatrix lodfMatrix = new DenseMatrix(monitoredBranches.size(), outagedBranches.size());
        computeLodf(loadFlowContext, monitoredBranches, outagedBranches, lodfMatrix::set);
        return lodfMatrix;
    }

    /**
     * Streams the LODF factors to the writer, one per (monitored branch, outaged branch) pair,
     * without ever materializing the full matrix.
     *
     * @throws IllegalStateException when the device is not available, or cannot hold one batch of states
     */
    public static void computeLodf(DcLoadFlowContext loadFlowContext, List<LfBranch> monitoredBranches,
                                   List<LfBranch> outagedBranches, LodfResultWriter writer) {
        Objects.requireNonNull(loadFlowContext);
        if (!isAvailable()) {
            throw new IllegalStateException("GPU native library not available");
        }
        try (GpuDcLoadFlowEngine engine = new GpuDcLoadFlowEngine(loadFlowContext)) {
            engine.prepare();
            computeLodf(engine, monitoredBranches, outagedBranches, writer);
        }
    }

    /**
     * Same, on an engine that is already prepared — the factorization is the one the load flow (or
     * the security analysis) built, so the factors cost only their own solves and kernels. The
     * device analysis and factorization are a fixed cost of a second or two on a large network,
     * which is worth paying once.
     */
    public static void computeLodf(GpuDcLoadFlowEngine engine, List<LfBranch> monitoredBranches,
                                   List<LfBranch> outagedBranches, LodfResultWriter writer) {
        Objects.requireNonNull(engine);
        Objects.requireNonNull(monitoredBranches);
        Objects.requireNonNull(outagedBranches);
        Objects.requireNonNull(writer);
        outagedBranches.forEach(GpuLodfCalculator::checkOutagedBranch);
        if (monitoredBranches.isEmpty() || outagedBranches.isEmpty()) {
            return;
        }
        EquationSystem<DcVariableType, DcEquationType> equationSystem = engine.getContext().getEquationSystem();
        GpuDcSolver solver = engine.getSolver();
        int[] monitored = monitoredBranches.stream().mapToInt(LfBranch::getNum).toArray();
        int n = engine.getAssembly().size();
        int batchSize = Math.max(1, (int) Math.min(outagedBranches.size(), maxStateBytesPerBatch() / ((long) n * Double.BYTES)));
        int blockColumns = Math.max(1, (int) (maxBlockBytes() / ((long) monitored.length * Double.BYTES)));
        double[] block = null;
        for (int batchStart = 0; batchStart < outagedBranches.size(); batchStart += batchSize) {
            int batchEnd = Math.min(batchStart + batchSize, outagedBranches.size());
            List<LfBranch> batch = outagedBranches.subList(batchStart, batchEnd);
            int[] columnOfOutage = uploadElements(solver, equationSystem, batch);
            int columnCount = 1 + java.util.Arrays.stream(columnOfOutage).max().orElse(-1);
            int blockSize = Math.min(blockColumns, columnCount);
            if (block == null || block.length < (long) monitored.length * blockSize) {
                block = new double[monitored.length * blockSize];
            }
            for (int col0 = 0; col0 < columnCount; col0 += blockSize) {
                int cols = Math.min(blockSize, columnCount - col0);
                solver.lodfBlock(monitored, col0, cols, block);
                for (int o = 0; o < columnOfOutage.length; o++) {
                    int column = columnOfOutage[o] - col0;
                    if (column < 0 || column >= cols) {
                        continue;
                    }
                    int outagedIndex = batchStart + o;
                    int offset = column * monitored.length;
                    for (int row = 0; row < monitored.length; row++) {
                        writer.writeLodf(row, outagedIndex, block[offset + row]);
                    }
                }
            }
        }
    }

    /**
     * The {@code +1/-1} right-hand sides of one batch, {@code ComputedElement.fillRhs} row for row
     * (the slack side carries no entry), with {@code setComputedElementIndexes}' column sharing
     * between outages of the same branch.
     *
     * @return the state column of every outage of the batch
     */
    private static int[] uploadElements(GpuDcSolver solver, EquationSystem<DcVariableType, DcEquationType> equationSystem,
                                        List<LfBranch> batch) {
        Map<LfBranch, Integer> columnByBranch = new LinkedHashMap<>();
        int[] columnOfOutage = new int[batch.size()];
        for (int i = 0; i < batch.size(); i++) {
            columnOfOutage[i] = columnByBranch.computeIfAbsent(batch.get(i), b -> columnByBranch.size());
        }
        int columnCount = columnByBranch.size();
        int[] rowPlus = new int[columnCount];
        int[] rowMinus = new int[columnCount];
        int[] elemBranch = new int[columnCount];
        java.util.Arrays.fill(rowPlus, -1);
        java.util.Arrays.fill(rowMinus, -1);
        for (Map.Entry<LfBranch, Integer> e : columnByBranch.entrySet()) {
            LfBranch branch = e.getKey();
            int column = e.getValue();
            elemBranch[column] = branch.getNum();
            LfBus bus1 = branch.getBus1();
            LfBus bus2 = branch.getBus2();
            if (bus1.isSlack()) {
                rowMinus[column] = pColumn(equationSystem, bus2);
            } else if (bus2.isSlack()) {
                rowPlus[column] = pColumn(equationSystem, bus1);
            } else {
                rowPlus[column] = pColumn(equationSystem, bus1);
                rowMinus[column] = pColumn(equationSystem, bus2);
            }
        }
        if (!solver.setElements(rowPlus, rowMinus, elemBranch)) {
            throw new IllegalStateException("the device cannot hold the element states of " + columnCount + " outages");
        }
        return columnOfOutage;
    }

    private static int pColumn(EquationSystem<DcVariableType, DcEquationType> equationSystem, LfBus bus) {
        return equationSystem.getEquation(bus.getNum(), DcEquationType.BUS_TARGET_P)
                .map(Equation::getColumn)
                .orElseThrow(() -> new IllegalStateException("no P equation for bus " + bus.getId()));
    }

    /** {@code LodfCalculator.checkOutagedBranch}, message for message. */
    private static void checkOutagedBranch(LfBranch branch) {
        if (branch.getBus1() == null || branch.getBus2() == null) {
            throw new PowsyblException("Branch '" + branch.getId() + "' must be connected on both sides to compute LODF factors for its outage");
        }
        if (branch.isZeroImpedance(LoadFlowModel.DC)) {
            throw new PowsyblException("LODF factors cannot be computed for the outage of zero impedance branch '" + branch.getId() + "'");
        }
    }

    /** For tests and callers that already hold a list: the branches of a network, in its own order. */
    public static List<LfBranch> flowingBranches(LfNetwork network) {
        List<LfBranch> branches = new ArrayList<>();
        for (LfBranch branch : network.getBranches()) {
            if (branch.getBus1() != null && branch.getBus2() != null && !branch.isZeroImpedance(LoadFlowModel.DC)) {
                branches.add(branch);
            }
        }
        return branches;
    }
}
