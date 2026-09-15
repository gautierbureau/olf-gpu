/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.PhaseTapChanger;
import com.powsybl.iidm.network.TwoWindingsTransformer;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.iidm.network.test.PhaseShifterTestCaseFactory;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrix;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.DcLoadFlowEngine;
import com.powsybl.openloadflow.dc.DcLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcTargetVector;
import com.powsybl.openloadflow.dc.equations.DcApproximationType;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.HvdcNetworkFactory;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import com.powsybl.openloadflow.network.util.ActivePowerDistribution;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The device-assembled DC system against OLF's own: the matrix values bit for bit
 * ({@code SparseMatrix.getValues()} in OLF's storage order), the target vector bit for bit
 * ({@code TargetVector.getArray()}), the solve within two direct solvers' roundoff (1e-10 pu), and
 * the slack distribution kernel against {@code ActivePowerDistribution} bit for bit — on every DC
 * equation shape: plain, distributed slack, phase shifters fixed and regulating, multi-slack, HVDC
 * AC emulation, zero-impedance branches, the IGNORE_G approximation, scalar and vectorized systems.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuDcAssemblyTest {

    private static final double SOLVE_TOL = 1e-10;

    record Case(String name, Network network, LoadFlowParameters parameters, Consumer<DcLoadFlowParameters> tweak) {
        Case(String name, Network network, LoadFlowParameters parameters) {
            this(name, network, parameters, p -> { });
        }
    }

    private static LoadFlowParameters dc() {
        LoadFlowParameters p = new LoadFlowParameters().setDc(true);
        OpenLoadFlowParameters.create(p);
        return p;
    }

    private static Network phaseShifter(boolean regulating) {
        Network network = PhaseShifterTestCaseFactory.create();
        TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().getStep(0).setAlpha(-5);
        ps1.getPhaseTapChanger().getStep(2).setAlpha(5);
        ps1.getPhaseTapChanger().setTapPosition(2);                    // a non-zero fixed shift: the rhs is exercised
        if (regulating) {
            ps1.getPhaseTapChanger().setTargetDeadband(10);
            ps1.getPhaseTapChanger().setRegulationMode(PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL);
            ps1.getPhaseTapChanger().setRegulationValue(-80);
            ps1.getPhaseTapChanger().setRegulating(true);
        }
        return network;
    }

    static List<Case> cases() {
        List<Case> cases = new ArrayList<>();
        cases.add(new Case("ieee14", IeeeCdfNetworkFactory.create14(), dc()));
        cases.add(new Case("ieee14 scalar equations", IeeeCdfNetworkFactory.create14(), dc(), p -> p.setVectorized(false)));
        cases.add(new Case("ieee14 distributed P_MAX", IeeeCdfNetworkFactory.create14(),
                dc().setDistributedSlack(true).setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX)));
        cases.add(new Case("ieee14 distributed P", IeeeCdfNetworkFactory.create14(),
                dc().setDistributedSlack(true).setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P)));
        cases.add(new Case("ieee14 distributed LOAD", IeeeCdfNetworkFactory.create14(),
                dc().setDistributedSlack(true).setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_LOAD)));
        LoadFlowParameters multiSlack = dc();
        OpenLoadFlowParameters.get(multiSlack).setMaxSlackBusCount(3);
        cases.add(new Case("ieee14 three slack buses", IeeeCdfNetworkFactory.create14(), multiSlack));
        cases.add(new Case("eurostag", EurostagFactory.fix(EurostagTutorialExample1Factory.create()), dc()));
        LoadFlowParameters ignoreG = dc();
        OpenLoadFlowParameters.get(ignoreG).setDcApproximationType(DcApproximationType.IGNORE_G);
        cases.add(new Case("eurostag IGNORE_G", EurostagFactory.fix(EurostagTutorialExample1Factory.create()), ignoreG));
        cases.add(new Case("phase shifter fixed", phaseShifter(false), dc()));
        cases.add(new Case("phase shifter regulating", phaseShifter(true), dc().setPhaseShifterRegulationOn(true)));
        cases.add(new Case("phase shifter with alpha variable", phaseShifter(false), dc(), p -> p.getEquationSystemCreationParameters().setForcePhaseControlOffAndAddAngle1Var(true)));
        LoadFlowParameters hvdc = dc();
        hvdc.setHvdcAcEmulation(true);
        cases.add(new Case("hvdc ac emulation", HvdcNetworkFactory.createWithHvdcInAcEmulation(), hvdc));
        Network zero = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        zero.getLine("NHV1_NHV2_1").setR(0).setX(0);
        cases.add(new Case("zero impedance line", zero, dc()));
        cases.add(new Case("ieee118 distributed P_MAX", IeeeCdfNetworkFactory.create118(),
                dc().setDistributedSlack(true).setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX)));
        return cases;
    }

    private static DcLoadFlowParameters dcParameters(Case c) {
        DcLoadFlowParameters p = OpenLoadFlowParameters.createDcParameters(c.network(), c.parameters(), OpenLoadFlowParameters.get(c.parameters()),
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false);
        c.tweak().accept(p);
        return p;
    }

    @TestFactory
    Stream<DynamicTest> matrixAndTargetBitIdentical() {
        return cases().stream().map(c -> DynamicTest.dynamicTest(c.name(), () -> checkAssembly(c)));
    }

    private static void checkAssembly(Case c) {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        DcLoadFlowParameters dcParameters = dcParameters(c);
        int checked = 0;
        for (LfNetwork lfNetwork : LfNetwork.load(c.network(), new LfNetworkLoaderImpl(), dcParameters.getNetworkParameters(), ReportNode.NO_OP)) {
            if (lfNetwork.getValidity() != LfNetwork.Validity.VALID) {
                continue;
            }
            try (DcLoadFlowContext context = new DcLoadFlowContext(lfNetwork, dcParameters)) {
                // the stock engine: the network in its final state (slack distributed, taps moved)
                assertTrue(new DcLoadFlowEngine(context).run().isSuccess(), c.name() + ": stock DC load flow");
                JacobianMatrix<?, ?> jacobian = context.getJacobianMatrix();
                SparseMatrix matrix = (SparseMatrix) jacobian.getMatrix();
                // the target vector as OLF builds it from the network's CURRENT state: the context's own copy is
                // stale by construction after a run (updateNetwork writes the SOLVED alpha back into the pi model,
                // one ulp from the alpha the target was built from, and nothing invalidates the target on setA1)
                double[] target;
                try (var fresh = new DcTargetVector(lfNetwork, context.getEquationSystem())) {
                    target = fresh.getArray().clone();
                }

                GpuDcAssembly a = GpuDcAssembly.capture(lfNetwork, context.getEquationSystem(), dcParameters.getEquationSystemCreationParameters());
                int n = a.size();
                assertEquals(matrix.getRowCount(), n, "size");
                int[] columnStart = Arrays.copyOf(matrix.getColumnStart(), n + 1);
                assertArrayEquals(columnStart, a.rowPtr, "pattern: column starts");
                int nnz = columnStart[n];
                assertArrayEquals(Arrays.copyOf(matrix.getRowIndices(), nnz), a.colIdx, "pattern: row indices");
                assertBitIdentical(Arrays.copyOf(matrix.getValues(), nnz), a.hostValues(), "host model of the values");
                assertBitIdentical(target, a.hostTarget(a.genTargetP), "host model of the target");

                try (GpuDcSolver solver = GpuDcSolver.createAssembled(a)) {
                    assertBitIdentical(Arrays.copyOf(matrix.getValues(), nnz), solver.readValues(), "device values");
                    solver.fillTarget();
                    assertBitIdentical(target, solver.readTarget(), "device target");
                    double[] x = solver.solveTarget();
                    double[] ref = target.clone();
                    jacobian.solveTransposed(ref);
                    double maxDiff = 0;
                    for (int i = 0; i < n; i++) {
                        maxDiff = Math.max(maxDiff, Math.abs(x[i] - ref[i]));
                    }
                    assertTrue(maxDiff < SOLVE_TOL, c.name() + ": device solve vs KLU, max diff " + maxDiff);
                    checkDistribution(c, lfNetwork, a, solver);
                }
                checked++;
            }
        }
        assertTrue(checked > 0, "at least one valid component");
    }

    /**
     * The device distribution kernel against OLF's {@code ActivePowerDistribution} from the same
     * network state, for every generation balance type with and without active limits, on mismatches
     * of both signs — generator targets and remaining mismatch bit for bit.
     */
    private static void checkDistribution(Case c, LfNetwork lfNetwork, GpuDcAssembly a, GpuDcSolver solver) {
        List<LfGenerator> generators = a.generators();
        double[] snapshot = new double[generators.size()];
        for (int i = 0; i < snapshot.length; i++) {
            snapshot[i] = generators.get(i).getTargetP();
        }
        LfGenerator reference = lfNetwork.getSynchronousNetworks().getFirst().getReferenceGenerator();
        for (LoadFlowParameters.BalanceType balanceType : List.of(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX,
                LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P, LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR,
                LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN)) {
            for (boolean limits : new boolean[] {true, false}) {
                for (double mismatch : new double[] {0.37, -0.81, 12.5, -1e-4}) {
                    // host
                    ActivePowerDistribution.Result host = ActivePowerDistribution.create(balanceType, false, limits)
                            .run(reference, lfNetwork.getBuses(), mismatch);
                    double[] hostTargets = new double[generators.size()];
                    for (int i = 0; i < hostTargets.length; i++) {
                        hostTargets[i] = generators.get(i).getTargetP();
                        generators.get(i).setTargetP(snapshot[i]);
                    }
                    // device, from the same snapshot
                    solver.setGeneratorTargets(snapshot);
                    GpuDcSolver.Distribution device = solver.distributeSlack(GpuDcLoadFlowEngine.participationType(balanceType), limits,
                            a.referenceGenerator, false, mismatch);
                    String where = c.name() + " " + balanceType + (limits ? " limits" : "") + " mismatch " + mismatch;
                    assertBitIdentical(hostTargets, solver.readGeneratorTargets(), where + ": generator targets");
                    assertEquals(host.remainingMismatch(), device.remainingMismatch(), 0.0, where + ": remaining mismatch");
                    assertEquals(host.iteration(), device.iterations(), where + ": iterations");
                }
            }
        }
        solver.setGeneratorTargets(snapshot);
    }

    private static void assertBitIdentical(double[] expected, double[] actual, String what) {
        assertEquals(expected.length, actual.length, what + ": length");
        for (int i = 0; i < expected.length; i++) {
            if (Double.doubleToLongBits(expected[i]) != Double.doubleToLongBits(actual[i])) {
                fail(what + ": entry " + i + " expected " + expected[i] + " got " + actual[i] + " (diff " + (actual[i] - expected[i]) + ")");
            }
        }
    }
}
