/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.ac.equations.AbstractClosedBranchAcFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.GpuBranchTermValues;
import com.powsybl.openloadflow.ac.equations.vector.AcBranchVector;
import com.powsybl.openloadflow.ac.equations.vector.GpuBranchVectorValues;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.equations.EquationTerm;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.util.Evaluable;
import net.jafama.DoubleWrapper;
import net.jafama.FastMath;

import java.util.Arrays;
import java.util.Map;

/**
 * The quantities OLF's violation detection reads after a post-contingency load flow, computed from a device state array
 * (base row layout) instead of read from OLF's objects once the state is written into them.
 *
 * <p>Each value is the one the branch's or the bus's own accessor returns after today's write-back
 * ({@code GpuStateInjector.injectState} then {@code GpuPostContingencyState.updateBusesFromState}), bit for bit: OLF's
 * own static formula functions, fed with the parameters the accessor reads and the state's doubles.
 * <ul>
 *   <li>A closed branch's P and Q: the vectorized network's arrays ({@link GpuBranchVectorValues}), from the vector's
 *       cached parameters; r1 and a1 from the state when the branch has a RHO1 or ALPHA1 variable, else the vector's.</li>
 *   <li>A closed branch's current: the scalar current term ({@link GpuBranchTermValues}), from its own parameter
 *       snapshot; r1 and a1 from the state when the term reads a variable, else the live pi model.</li>
 *   <li>A branch open at one side: the open terms at the live side, from their snapshots, with the pi model's r1 for P
 *       and Q and the term's own r1 for the current; 0 at the open side; NaN with both sides open, as OLF's evaluables
 *       answer.</li>
 *   <li>Apparent power: {@code FastMath.sqrt(p * p + q * q)} of the side's P and Q, as
 *       {@code AbstractLfBranch.computeApparentPower1/2} computes it.</li>
 *   <li>A bus voltage: {@code (x * nominalV) / nominalV}, the round trip {@code LfBus.setV} then {@code getV} makes (it
 *       differs from {@code x} by one ulp on about one bus in ten); the angle: {@code x} itself.</li>
 * </ul>
 * A branch without flow terms (a zero-impedance branch, a switch) answers its evaluables' constants, which read no
 * state.
 *
 * <p>The branch's connection status is read from the branch, as the contingency's {@code apply} left it. The view never
 * writes an OLF object. It is not thread-safe (one scratch wrapper): one view per partition's run state.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuStateView {

    private final Map<Variable<AcVariableType>, Integer> baseRows;
    private final int[] vRow;
    private final int[] phiRow;
    private final int[] r1Row;
    private final int[] a1Row;
    private final boolean usable;
    private final DoubleWrapper w = new DoubleWrapper();

    GpuStateView(LfNetwork network, Map<Variable<AcVariableType>, Integer> baseRows) {
        this.baseRows = baseRows;
        int nBuses = network.getBuses().size();
        int nBranches = network.getBranches().size();
        vRow = new int[nBuses];
        phiRow = new int[nBuses];
        r1Row = new int[nBranches];
        a1Row = new int[nBranches];
        Arrays.fill(vRow, -1);
        Arrays.fill(phiRow, -1);
        Arrays.fill(r1Row, -1);
        Arrays.fill(a1Row, -1);
        boolean kindsKnown = true;
        for (Map.Entry<Variable<AcVariableType>, Integer> e : baseRows.entrySet()) {
            Variable<AcVariableType> v = e.getKey();
            int num = v.getElementNum();
            switch (v.getType()) {
                case BUS_V -> vRow[num] = e.getValue();
                case BUS_PHI -> phiRow[num] = e.getValue();
                case BRANCH_RHO1 -> r1Row[num] = e.getValue();
                case BRANCH_ALPHA1 -> a1Row[num] = e.getValue();
                case DUMMY_P, DUMMY_Q, SHUNT_B -> {
                    // feed no quantity the detection reads: a coupler's dummy flows, a shunt's susceptance
                }
                default -> kindsKnown = false;
            }
        }
        boolean rowsComplete = true;
        for (int b = 0; b < nBuses && rowsComplete; b++) {
            rowsComplete = vRow[b] >= 0 && phiRow[b] >= 0;
        }
        boolean vectorized = true;
        for (LfBranch branch : network.getBranches()) {
            if (branch.getClosedP1() instanceof EquationTerm<?, ?> && GpuBranchVectorValues.of(branch.getClosedP1()) == null) {
                vectorized = false;                          // a scalar closed flow term: not the vectorized network
                break;
            }
        }
        usable = kindsKnown && rowsComplete && vectorized;
    }

    /**
     * True when the view can stand for the written-back objects on this run: every bus has its voltage and angle rows,
     * the system holds no variable kind the view does not know (BUS_V, BUS_PHI, BRANCH_RHO1, BRANCH_ALPHA1, and the
     * DUMMY_P/Q and SHUNT_B it ignores), and the closed branches' flows come from the vectorized network.
     */
    boolean usable() {
        return usable;
    }

    /** True when this view was built from these base rows. */
    boolean isFor(Map<Variable<AcVariableType>, Integer> rows) {
        return baseRows == rows;
    }

    /** {@code bus.getV()} once the state is written back: the {@code setV}/{@code getV} round trip through kV. */
    double busV(double[] state, LfBus bus) {
        double nominalV = bus.getNominalV();
        return state[vRow[bus.getNum()]] * nominalV / nominalV;
    }

    /** The bus's voltage as the state holds it, before the round trip. */
    double stateV(double[] state, LfBus bus) {
        return state[vRow[bus.getNum()]];
    }

    /** {@code bus.getAngle()} once the state is written back. */
    double busAngle(double[] state, LfBus bus) {
        return state[phiRow[bus.getNum()]];
    }

    /**
     * What {@code branch.getI1/2().eval()}, {@code getP1/2().eval()} or {@code computeApparentPower1/2()} returns once
     * the state is written back, for {@code side} ({@link GpuTwoSides}) and {@code quantity}
     * ({@link GpuLimitTable#QTY_I}, {@code QTY_P}, {@code QTY_S}).
     */
    double branch(double[] state, LfBranch branch, int side, byte quantity) {
        if (quantity == GpuLimitTable.QTY_S) {
            double p = flow(state, branch, side, true);
            double q = flow(state, branch, side, false);
            return FastMath.sqrt(p * p + q * q);
        }
        if (quantity == GpuLimitTable.QTY_P) {
            return flow(state, branch, side, true);
        }
        return current(state, branch, side);
    }

    /** True when the branch has no flow terms (a zero-impedance branch, a switch): OLF's evaluables for it are
     *  constants (NaN, or null where never set), which read no state. */
    private static boolean withoutFlowTerms(LfBranch branch) {
        return !(branch.getClosedP1() instanceof EquationTerm<?, ?>) && !(branch.getOpenP1() instanceof EquationTerm<?, ?>)
                && !(branch.getOpenP2() instanceof EquationTerm<?, ?>);
    }

    /** The branch's active ({@code active}) or reactive flow at {@code side}. */
    private double flow(double[] state, LfBranch branch, int side, boolean active) {
        if (withoutFlowTerms(branch)) {
            return side == GpuTwoSides.ONE ? (active ? branch.getP1() : branch.getQ1()).eval()
                    : (active ? branch.getP2() : branch.getQ2()).eval();
        }
        LfBus bus1 = branch.getBus1();
        LfBus bus2 = branch.getBus2();
        boolean live1 = bus1 != null && branch.isConnectedSide1();
        boolean live2 = bus2 != null && branch.isConnectedSide2();
        int num = branch.getNum();
        if (live1 && live2) {
            AcBranchVector vector = GpuBranchVectorValues.of(branch.getClosedP1());
            double v1 = state[vRow[bus1.getNum()]];
            double ph1 = state[phiRow[bus1.getNum()]];
            double v2 = state[vRow[bus2.getNum()]];
            double ph2 = state[phiRow[bus2.getNum()]];
            double r1 = r1Row[num] >= 0 ? state[r1Row[num]] : GpuBranchVectorValues.r1(vector, num);
            double a1 = a1Row[num] >= 0 ? state[a1Row[num]] : GpuBranchVectorValues.a1(vector, num);
            if (side == GpuTwoSides.ONE) {
                return active ? GpuBranchVectorValues.p1(vector, num, v1, ph1, r1, a1, v2, ph2, w)
                        : GpuBranchVectorValues.q1(vector, num, v1, ph1, r1, a1, v2, ph2, w);
            }
            return active ? GpuBranchVectorValues.p2(vector, num, v1, ph1, r1, a1, v2, ph2, w)
                    : GpuBranchVectorValues.q2(vector, num, v1, ph1, r1, a1, v2, ph2, w);
        }
        if (live1) {                                         // side 2 open: OLF's open terms at side 1, ZERO at side 2
            if (side != GpuTwoSides.ONE) {
                return 0;
            }
            double v1 = state[vRow[bus1.getNum()]];
            double r1 = branch.getPiModel().getR1();
            return active ? GpuBranchTermValues.openP1(branch.getOpenP1(), v1, r1)
                    : GpuBranchTermValues.openQ1(branch.getOpenQ1(), v1, r1);
        }
        if (live2) {                                         // side 1 open: ZERO at side 1, the open terms at side 2
            if (side != GpuTwoSides.TWO) {
                return 0;
            }
            double v2 = state[vRow[bus2.getNum()]];
            return active ? GpuBranchTermValues.openP2(branch.getOpenP2(), v2)
                    : GpuBranchTermValues.openQ2(branch.getOpenQ2(), v2);
        }
        return Double.NaN;
    }

    /** The branch's current magnitude at {@code side}. */
    private double current(double[] state, LfBranch branch, int side) {
        if (withoutFlowTerms(branch)) {
            return (side == GpuTwoSides.ONE ? branch.getI1() : branch.getI2()).eval();
        }
        LfBus bus1 = branch.getBus1();
        LfBus bus2 = branch.getBus2();
        boolean live1 = bus1 != null && branch.isConnectedSide1();
        boolean live2 = bus2 != null && branch.isConnectedSide2();
        int num = branch.getNum();
        if (live1 && live2) {
            Evaluable term = side == GpuTwoSides.ONE ? branch.getClosedI1() : branch.getClosedI2();
            AbstractClosedBranchAcFlowEquationTerm closed = (AbstractClosedBranchAcFlowEquationTerm) term;
            double v1 = state[vRow[bus1.getNum()]];
            double ph1 = state[phiRow[bus1.getNum()]];
            double v2 = state[vRow[bus2.getNum()]];
            double ph2 = state[phiRow[bus2.getNum()]];
            double r1 = closed.getR1Var() != null ? state[r1Row[num]] : branch.getPiModel().getR1();
            double a1 = closed.getA1Var() != null ? state[a1Row[num]] : branch.getPiModel().getA1();
            return side == GpuTwoSides.ONE ? GpuBranchTermValues.closedI1(term, v1, ph1, r1, a1, v2, ph2)
                    : GpuBranchTermValues.closedI2(term, v1, ph1, r1, a1, v2, ph2);
        }
        if (live1) {
            if (side != GpuTwoSides.ONE) {
                return 0;
            }
            // the open current term reads its RHO1 variable when the branch was created deriving r1, as the closed
            // terms of the same branch do
            Evaluable closedI1 = branch.getClosedI1();
            boolean derivesR1 = closedI1 instanceof AbstractClosedBranchAcFlowEquationTerm closed && closed.getR1Var() != null;
            double r1 = derivesR1 ? state[r1Row[num]] : branch.getPiModel().getR1();
            return GpuBranchTermValues.openI1(branch.getOpenI1(), state[vRow[bus1.getNum()]], state[phiRow[bus1.getNum()]], r1);
        }
        if (live2) {
            if (side != GpuTwoSides.TWO) {
                return 0;
            }
            return GpuBranchTermValues.openI2(branch.getOpenI2(), state[vRow[bus2.getNum()]], state[phiRow[bus2.getNum()]]);
        }
        return Double.NaN;
    }
}
