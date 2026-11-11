/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.PhaseTapChanger;
import com.powsybl.iidm.network.TwoWindingsTransformer;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.iidm.network.test.ThreeWindingsTransformerNetworkFactory;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.AcSolverStatus;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.PhaseControlFactory;
import com.powsybl.openloadflow.network.SlackBusSelectionMode;
import com.powsybl.openloadflow.network.VoltageControlNetworkFactory;
import com.powsybl.openloadflow.network.ZeroImpedanceNetworkFactory;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link GpuStateView} against OLF's own objects, with no device: OLF's CPU load flow solves a fixture, and every
 * quantity the violation detection reads (I, P and apparent power at both sides of every enabled branch, every enabled
 * bus's voltage and angle) must come out of the view, fed with the solved state vector, equal bit for bit to what the
 * branch's and the bus's own accessors return. Each fixture exercises one input path: closed branches (the vectorized
 * network's arrays and the scalar current terms), a side opened after the base solve (the open terms, at each side), a
 * transformer with a RHO1 variable and a phase shifter with an ALPHA1 variable (r1 and a1 from the state), a tap changed
 * after the equation system was built (the refreshed parameters), three-windings transformer legs, and zero-impedance
 * branches (no flow terms).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuStateViewTest {

    private record Solved(LfNetwork network, AcLoadFlowContext context) {
    }

    private static Solved solve(Network iidm, BiConsumer<LoadFlowParameters, OpenLoadFlowParameters> configure,
                                Consumer<LfNetwork> beforeContext) {
        LoadFlowParameters lfp = new LoadFlowParameters();
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.create(lfp);
        configure.accept(lfp, olfp);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);
        beforeContext.accept(network);
        AcLoadFlowContext context = new AcLoadFlowContext(network, acp);
        assertEquals(AcSolverStatus.CONVERGED, new AcloadFlowEngine(context).run().getSolverStatus());
        return new Solved(network, context);
    }

    private static void resolve(Solved solved) {
        assertEquals(AcSolverStatus.CONVERGED, new AcloadFlowEngine(solved.context()).run().getSolverStatus());
    }

    /** The state vector and its rows, as the device hands a state back (here the rows are the index's own). */
    private static Map<Variable<AcVariableType>, Integer> rows(Solved solved) {
        Map<Variable<AcVariableType>, Integer> rows = new HashMap<>();
        for (Variable<AcVariableType> v : solved.context().getEquationSystem().getIndex().getSortedVariablesToFind()) {
            rows.put(v, v.getRow());
        }
        return rows;
    }

    private static double olf(LfBranch branch, int side, byte quantity) {
        if (side == GpuTwoSides.ONE) {
            return quantity == GpuLimitTable.QTY_I ? branch.getI1().eval()
                    : quantity == GpuLimitTable.QTY_P ? branch.getP1().eval() : branch.computeApparentPower1();
        }
        return quantity == GpuLimitTable.QTY_I ? branch.getI2().eval()
                : quantity == GpuLimitTable.QTY_P ? branch.getP2().eval() : branch.computeApparentPower2();
    }

    /** Every quantity of every enabled branch and bus, bit for bit; returns the bus voltages whose round trip differs
     *  from the state's value. */
    private static int assertViewIsOlf(Solved solved) {
        Map<Variable<AcVariableType>, Integer> rows = rows(solved);
        double[] state = solved.context().getEquationSystem().getStateVector().get().clone();
        GpuStateView view = new GpuStateView(solved.network(), rows);
        int compared = 0;
        for (LfBranch branch : solved.network().getBranches()) {
            if (branch.isDisabled()) {
                continue;
            }
            for (int side = GpuTwoSides.ONE; side <= GpuTwoSides.TWO; side++) {
                for (byte quantity = GpuLimitTable.QTY_I; quantity <= GpuLimitTable.QTY_S; quantity++) {
                    double expected = olf(branch, side, quantity);
                    double actual = view.branch(state, branch, side, quantity);
                    int s = side;
                    byte q = quantity;
                    assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(actual),
                        () -> branch.getId() + " side " + (s + 1) + " quantity " + q + ": OLF " + expected + ", view " + actual);
                    compared++;
                }
            }
        }
        int roundTripDiffers = 0;
        for (LfBus bus : solved.network().getBuses()) {
            if (bus.isDisabled()) {
                continue;
            }
            assertEquals(Double.doubleToLongBits(bus.getV()), Double.doubleToLongBits(view.busV(state, bus)), bus.getId() + " V");
            assertEquals(Double.doubleToLongBits(bus.getAngle()), Double.doubleToLongBits(view.busAngle(state, bus)),
                    bus.getId() + " angle");
            if (bus.getV() != view.stateV(state, bus)) {
                roundTripDiffers++;
            }
            compared += 2;
        }
        assertTrue(compared > 0);
        return roundTripDiffers;
    }

    private static boolean hasVariable(Solved solved, AcVariableType type) {
        return rows(solved).keySet().stream().anyMatch(v -> v.getType() == type);
    }

    @Test
    void closedBranchesAndBusesOnIeee118() {
        int roundTripDiffers = assertViewIsOlf(solve(IeeeCdfNetworkFactory.create118(), (p, o) -> { }, n -> { }));
        // the view's (x * nominalV) / nominalV is not x everywhere: without it some buses would differ by one ulp
        assertTrue(roundTripDiffers > 0, "no bus voltage differs from its round trip: the test would not see a view reading x");
    }

    @Test
    void sideOpenedAfterTheBaseSolve() {
        for (boolean side1 : new boolean[] {true, false}) {
            Solved solved = solve(EurostagTutorialExample1Factory.createWithFixedCurrentLimits(), (p, o) -> { }, n -> {
                LfBranch line = n.getBranchById("NHV1_NHV2_1");
                if (side1) {
                    line.setDisconnectionAllowedSide1(true);
                } else {
                    line.setDisconnectionAllowedSide2(true);
                }
            });
            assertViewIsOlf(solved);
            LfBranch line = solved.network().getBranchById("NHV1_NHV2_1");
            if (side1) {
                line.setConnectedSide1(false);
            } else {
                line.setConnectedSide2(false);
            }
            resolve(solved);
            assertViewIsOlf(solved);
        }
    }

    @Test
    void transformerWithARho1Variable() {
        Network iidm = VoltageControlNetworkFactory.createNetworkWithT2wt();
        TwoWindingsTransformer t2wt = iidm.getTwoWindingsTransformer("T2wT");
        t2wt.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                .setRegulationTerminal(t2wt.getTerminal2()).setTargetV(34.0);
        Solved solved = solve(iidm, (p, o) -> p.setTransformerVoltageControlOn(true), n -> { });
        assertTrue(hasVariable(solved, AcVariableType.BRANCH_RHO1), "the fixture must hold a RHO1 variable");
        assertViewIsOlf(solved);
    }

    @Test
    void phaseShifterWithAnAlpha1Variable() {
        Network iidm = PhaseControlFactory.createNetworkWithT2wt();
        TwoWindingsTransformer ps1 = iidm.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().setRegulationMode(PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL).setTargetDeadband(1)
                .setRegulating(true).setTapPosition(1).setRegulationTerminal(ps1.getTerminal1()).setRegulationValue(83);
        Solved solved = solve(iidm, (p, o) -> p.setPhaseShifterRegulationOn(true), n -> { });
        assertTrue(hasVariable(solved, AcVariableType.BRANCH_ALPHA1), "the fixture must hold an ALPHA1 variable");
        assertViewIsOlf(solved);
    }

    @Test
    void tapChangedAfterTheEquationSystemWasBuilt() {
        Solved solved = solve(EurostagTutorialExample1Factory.create(), (p, o) -> { }, n -> { });
        LfBranch transformer = solved.network().getBranchById("NHV2_NLOAD");
        int tap = transformer.getPiModel().getTapPosition();
        transformer.getPiModel().setTapPosition(tap == 0 ? 1 : tap - 1);
        resolve(solved);
        assertViewIsOlf(solved);
    }

    @Test
    void threeWindingsTransformerLegs() {
        // the slack OLF's own leg-limit tests take on this network: with the default selection it does not converge
        assertViewIsOlf(solve(ThreeWindingsTransformerNetworkFactory.create(), (p, o) -> o
                .setSlackBusSelectionMode(SlackBusSelectionMode.NAME).setSlackBusesIds(List.of("VL_33_0")), n -> { }));
    }

    @Test
    void zeroImpedanceBranches() {
        Solved solved = solve(ZeroImpedanceNetworkFactory.createWithVoltageControl(), (p, o) -> { }, n -> { });
        assertTrue(solved.network().getBranches().stream()
                .anyMatch(b -> b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)),
                "the fixture must hold a zero-impedance branch");
        assertViewIsOlf(solved);
    }
}
