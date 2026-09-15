/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.iidm.network.test.PhaseShifterTestCaseFactory;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.DcLoadFlowParameters;
import com.powsybl.openloadflow.dc.fastdc.LodfCalculator;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The device LODF factors against OLF's own {@code LodfCalculator} on the same network: every cell,
 * every NaN, every {@code -1} diagonal.
 *
 * <p>Bound 1e-10, the project's rule for two direct solvers on one matrix — the factors are ratios
 * of quantities of order one, so this is four orders of magnitude tighter than anything physical.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuLodfCalculatorTest {

    private static final double TOL = 1e-10;

    record Case(String name, Network network, Consumer<DcLoadFlowParameters> tweak) {
        Case(String name, Network network) {
            this(name, network, p -> { });
        }
    }

    static List<Case> cases() {
        List<Case> cases = new ArrayList<>();
        cases.add(new Case("ieee14", IeeeCdfNetworkFactory.create14()));
        cases.add(new Case("ieee14 scalar equations", IeeeCdfNetworkFactory.create14(), p -> p.setVectorized(false)));
        cases.add(new Case("ieee30", IeeeCdfNetworkFactory.create30()));
        cases.add(new Case("ieee118", IeeeCdfNetworkFactory.create118()));
        cases.add(new Case("eurostag", EurostagFactory.fix(EurostagTutorialExample1Factory.create())));
        cases.add(new Case("phase shifter with alpha variable", PhaseShifterTestCaseFactory.create(),
                p -> p.getEquationSystemCreationParameters().setForcePhaseControlOffAndAddAngle1Var(true)));
        Network zero = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        zero.getLine("NHV1_NHV2_1").setR(0).setX(0);
        cases.add(new Case("zero impedance line among the monitored", zero));
        return cases;
    }

    private static DcLoadFlowParameters dcParameters(Case c) {
        LoadFlowParameters p = new LoadFlowParameters().setDc(true);
        OpenLoadFlowParameters.create(p);
        DcLoadFlowParameters dcParameters = OpenLoadFlowParameters.createDcParameters(c.network(), p, OpenLoadFlowParameters.get(p),
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false);
        c.tweak().accept(dcParameters);
        return dcParameters;
    }

    @TestFactory
    Stream<DynamicTest> matchesStockCalculator() {
        return cases().stream().map(c -> DynamicTest.dynamicTest(c.name(), () -> check(c)));
    }

    private static void check(Case c) {
        assumeTrue(GpuLodfCalculator.isAvailable(), "GPU native library not available");
        DcLoadFlowParameters dcParameters = dcParameters(c);
        LfNetwork lfNetwork = LfNetwork.load(c.network(), new LfNetworkLoaderImpl(), dcParameters.getNetworkParameters(), ReportNode.NO_OP).getFirst();
        try (DcLoadFlowContext context = new DcLoadFlowContext(lfNetwork, dcParameters)) {
            // monitored: every branch, including those without a DC flow equation (they must come back NaN)
            List<LfBranch> monitored = lfNetwork.getBranches();
            List<LfBranch> outaged = GpuLodfCalculator.flowingBranches(lfNetwork);
            assertFalse(outaged.isEmpty(), c.name() + ": no outageable branch");
            DenseMatrix cpu = LodfCalculator.computeLodfMatrix(context, monitored, outaged);
            DenseMatrix gpu = GpuLodfCalculator.computeLodfMatrix(context, monitored, outaged);
            int nan = 0;
            int diagonal = 0;
            int noFlowRows = 0;
            double maxDiff = 0;
            for (int row = 0; row < monitored.size(); row++) {
                boolean noFlow = !outaged.contains(monitored.get(row));
                for (int col = 0; col < outaged.size(); col++) {
                    double a = cpu.get(row, col);
                    double b = gpu.get(row, col);
                    if (Double.isNaN(a) || Double.isNaN(b)) {
                        assertEquals(Double.isNaN(a), Double.isNaN(b), c.name() + ": NaN at (" + row + ", " + col + "): " + a + " vs " + b);
                        nan++;
                        continue;
                    }
                    // a monitored branch without a closed DC flow equation is NaN for every outage
                    assertFalse(noFlow, c.name() + ": branch without a DC flow got a value at (" + row + ", " + col + ")");
                    if (monitored.get(row) == outaged.get(col)) {
                        assertEquals(-1d, a, c.name() + ": the stock diagonal is -1");
                        assertEquals(-1d, b, c.name() + ": the device diagonal is -1");
                        diagonal++;
                    }
                    maxDiff = Math.max(maxDiff, Math.abs(a - b));
                }
                if (noFlow) {
                    noFlowRows++;
                }
            }
            assertTrue(maxDiff < TOL, c.name() + ": max |diff| " + maxDiff);
            assertTrue(diagonal > 0, c.name() + ": the -1 convention is exercised");
            assertTrue(nan >= noFlowRows * outaged.size(), c.name() + ": every branch without a DC flow is NaN everywhere");
        }
    }

    /** The streamed factors are the matrix's, and the batching does not change a value. */
    @Test
    void streamingAndBatchingAgree() {
        assumeTrue(GpuLodfCalculator.isAvailable(), "GPU native library not available");
        Case c = new Case("ieee118", IeeeCdfNetworkFactory.create118());
        DcLoadFlowParameters dcParameters = dcParameters(c);
        LfNetwork lfNetwork = LfNetwork.load(c.network(), new LfNetworkLoaderImpl(), dcParameters.getNetworkParameters(), ReportNode.NO_OP).getFirst();
        try (DcLoadFlowContext context = new DcLoadFlowContext(lfNetwork, dcParameters)) {
            List<LfBranch> branches = GpuLodfCalculator.flowingBranches(lfNetwork);
            DenseMatrix reference = GpuLodfCalculator.computeLodfMatrix(context, branches, branches);
            // one outage batch of ~4 columns and one downloaded block of ~2 columns
            System.setProperty("olf.gpu.dc.lodf.stateBytes", Long.toString(4L * context.getEquationSystem().getIndex().getColumnCount() * Double.BYTES));
            System.setProperty("olf.gpu.dc.lodf.blockBytes", Long.toString(2L * branches.size() * Double.BYTES));
            try {
                double[][] streamed = new double[branches.size()][branches.size()];
                boolean[][] written = new boolean[branches.size()][branches.size()];
                GpuLodfCalculator.computeLodf(context, branches, branches, (row, col, value) -> {
                    assertFalse(written[row][col], "each factor written once");
                    written[row][col] = true;
                    streamed[row][col] = value;
                });
                for (int row = 0; row < branches.size(); row++) {
                    for (int col = 0; col < branches.size(); col++) {
                        assertTrue(written[row][col], "every factor written");
                        double a = reference.get(row, col);
                        double b = streamed[row][col];
                        if (Double.isNaN(a) || Double.isNaN(b)) {
                            assertEquals(Double.isNaN(a), Double.isNaN(b), "NaN at (" + row + ", " + col + ")");
                        } else {
                            // not bit-identical across batchings: a cuDSS multi-right-hand-side solve blocks its
                            // columns, so the same column solved in a block of 4 and in a block of 186 differs by
                            // an ulp or two. The factors are within the 1e-10 bound either way.
                            assertEquals(a, b, TOL, "batched value at (" + row + ", " + col + ")");
                        }
                    }
                }
            } finally {
                System.clearProperty("olf.gpu.dc.lodf.stateBytes");
                System.clearProperty("olf.gpu.dc.lodf.blockBytes");
            }
        }
    }

    @Test
    void rejectsTheSameOutagesAsStock() {
        assumeTrue(GpuLodfCalculator.isAvailable(), "GPU native library not available");
        Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        network.getLine("NHV1_NHV2_1").setR(0).setX(0);
        Case c = new Case("eurostag", network);
        DcLoadFlowParameters dcParameters = dcParameters(c);
        LfNetwork lfNetwork = LfNetwork.load(network, new LfNetworkLoaderImpl(), dcParameters.getNetworkParameters(), ReportNode.NO_OP).getFirst();
        try (DcLoadFlowContext context = new DcLoadFlowContext(lfNetwork, dcParameters)) {
            LfBranch zeroImpedance = lfNetwork.getBranches().stream()
                    .filter(b -> b.getId().equals("NHV1_NHV2_1")).findFirst().orElseThrow();
            List<LfBranch> monitored = lfNetwork.getBranches();
            PowsyblException stock = assertThrows(PowsyblException.class,
                    () -> LodfCalculator.computeLodfMatrix(context, monitored, List.of(zeroImpedance)));
            PowsyblException device = assertThrows(PowsyblException.class,
                    () -> GpuLodfCalculator.computeLodfMatrix(context, monitored, List.of(zeroImpedance)));
            assertEquals(stock.getMessage(), device.getMessage());
        }
    }
}
