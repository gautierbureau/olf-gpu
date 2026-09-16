/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.DcLoadFlowParameters;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.util.PerUnit;
import com.powsybl.sensitivity.SensitivityAnalysis;
import com.powsybl.sensitivity.SensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityFactor;
import com.powsybl.sensitivity.SensitivityFunctionType;
import com.powsybl.sensitivity.SensitivityVariableType;
import com.powsybl.sensitivity.SensitivityAnalysisResult;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The device PTDF against OLF's own DC sensitivity analysis, which is the oracle: the same factors
 * computed by a completely different route, on the host, through the stock code.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuPtdfMatrixTest {

    /** Two direct solvers on one matrix: the gap is LU ordering roundoff, nothing else. */
    private static final double TOL = 1e-10;

    private static LoadFlowParameters dcParameters() {
        LoadFlowParameters parameters = new LoadFlowParameters().setDc(true).setDistributedSlack(false);
        OpenLoadFlowParameters.create(parameters);
        return parameters;
    }

    private record Context(LfNetwork lfNetwork, DcLoadFlowContext context) {
    }

    private static Context dcContext(Network network, LoadFlowParameters parameters) {
        OpenLoadFlowParameters parametersExt = OpenLoadFlowParameters.get(parameters);
        DcLoadFlowParameters dcParameters = OpenLoadFlowParameters.createDcParameters(network, parameters, parametersExt,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false);
        LfNetwork lfNetwork = Networks.load(network, dcParameters.getNetworkParameters()).get(0);
        return new Context(lfNetwork, new DcLoadFlowContext(lfNetwork, dcParameters));
    }

    /**
     * Every branch against every non-slack bus, compared with the value OLF's sensitivity analysis
     * gives for the same pair. The device works in per unit and OLF reports MW per MW, which for an
     * active power flow against an active power injection is the same number.
     */
    private void matchesStockSensitivity(Network network) {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        LoadFlowParameters parameters = dcParameters();
        Context context = dcContext(network, parameters);
        List<LfBranch> branches;
        List<LfBus> buses;
        DenseMatrix ptdf;
        try (DcLoadFlowContext loadFlowContext = context.context()) {
            branches = GpuPtdfMatrix.flowingBranches(context.lfNetwork());
            buses = GpuPtdfMatrix.injectionBuses(context.lfNetwork());
            try (GpuPtdfMatrix matrix = GpuPtdfMatrix.assemble(loadFlowContext, branches, buses)) {
                assertEquals(branches.size(), matrix.getRowCount());
                assertEquals(buses.size(), matrix.getColumnCount());
                ptdf = matrix.toDenseMatrix();
            }
        }

        // the oracle: OLF's own DC sensitivity analysis, injection factors on the same pairs
        List<SensitivityFactor> factors = new ArrayList<>();
        for (LfBranch branch : branches) {
            for (LfBus bus : buses) {
                bus.getGenerators().stream().findFirst().ifPresent(generator ->
                        factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_ACTIVE_POWER_1, branch.getId(),
                                SensitivityVariableType.INJECTION_ACTIVE_POWER, generator.getOriginalId(),
                                false, com.powsybl.contingency.ContingencyContext.none())));
            }
        }
        assumeTrue(!factors.isEmpty(), "no generator to take an injection factor at");
        SensitivityAnalysisParameters sensitivityParameters = new SensitivityAnalysisParameters();
        sensitivityParameters.setLoadFlowParameters(parameters);
        SensitivityAnalysisResult result = SensitivityAnalysis.find("OpenLoadFlow")
                .run(network, factors, List.of(), List.of(), sensitivityParameters);

        // looked up by identifier, not by position: the result is not promised in factor order
        int checked = 0;
        double maxDiff = 0;
        String worst = "";
        for (int row = 0; row < branches.size(); row++) {
            for (int column = 0; column < buses.size(); column++) {
                LfBus bus = buses.get(column);
                var generator = bus.getGenerators().stream().findFirst();
                if (generator.isEmpty()) {
                    continue;
                }
                double expected = result.getBranchFlow1SensitivityValue(generator.get().getOriginalId(),
                        branches.get(row).getId(), SensitivityVariableType.INJECTION_ACTIVE_POWER);
                double actual = ptdf.get(row, column);
                double diff = Math.abs(expected - actual);
                if (diff > maxDiff) {
                    maxDiff = diff;
                    worst = branches.get(row).getId() + " / " + bus.getId() + ": stock " + expected + " device " + actual;
                }
                checked++;
            }
        }
        assertTrue(checked > 0, "nothing was compared");
        assertTrue(maxDiff < TOL, "max |device - stock| over " + checked + " factors: " + maxDiff + " at " + worst);
    }

    @Test
    void matchesStockSensitivityOnIeee14() {
        matchesStockSensitivity(IeeeCdfNetworkFactory.create14());
    }

    @Test
    void matchesStockSensitivityOnIeee118() {
        matchesStockSensitivity(IeeeCdfNetworkFactory.create118());
    }

    /** A block read is the same values as the whole matrix, and the bounds are checked. */
    @Test
    void readsBlocks() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create14();
        Context context = dcContext(network, dcParameters());
        try (DcLoadFlowContext loadFlowContext = context.context()) {
            List<LfBranch> branches = GpuPtdfMatrix.flowingBranches(context.lfNetwork());
            List<LfBus> buses = GpuPtdfMatrix.injectionBuses(context.lfNetwork());
            try (GpuPtdfMatrix matrix = GpuPtdfMatrix.assemble(loadFlowContext, branches, buses)) {
                DenseMatrix whole = matrix.toDenseMatrix();
                int cols = Math.min(3, matrix.getColumnCount());
                double[] block = new double[matrix.getRowCount() * cols];
                matrix.readBlock(1, cols, block);
                for (int j = 0; j < cols; j++) {
                    for (int row = 0; row < matrix.getRowCount(); row++) {
                        assertEquals(whole.get(row, 1 + j), block[row + j * matrix.getRowCount()], 0d,
                                "row " + row + " column " + (1 + j));
                    }
                }
                assertEquals(matrix.getDeviceBytes(), (long) matrix.getRowCount() * matrix.getColumnCount() * Double.BYTES);
                assertThrows(IndexOutOfBoundsException.class, () -> matrix.readBlock(matrix.getColumnCount(), 1, block));
                assertThrows(IllegalArgumentException.class, () -> matrix.readBlock(0, cols, new double[1]));
            }
        }
    }

    /** The slack has no active power row, so asking for an injection there is refused, not ignored. */
    @Test
    void refusesTheSlackBus() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create14();
        Context context = dcContext(network, dcParameters());
        try (DcLoadFlowContext loadFlowContext = context.context()) {
            List<LfBranch> branches = GpuPtdfMatrix.flowingBranches(context.lfNetwork());
            LfBus slack = context.lfNetwork().getBuses().stream().filter(LfBus::isSlack).findFirst().orElseThrow();
            assertThrows(IllegalArgumentException.class,
                () -> GpuPtdfMatrix.assemble(loadFlowContext, branches, List.of(slack)));
            assertThrows(IllegalArgumentException.class,
                () -> GpuPtdfMatrix.assemble(loadFlowContext, branches, List.of()));
        }
    }

    /** Per unit is what the device works in; this pins the scale the class documents. */
    @Test
    void isPerUnit() {
        assertEquals(100.0, PerUnit.SB, 0d);
    }
}
