/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector;

import com.powsybl.openloadflow.ac.equations.ClosedBranchSide1ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.ClosedBranchSide1ReactiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.ClosedBranchSide2ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.ClosedBranchSide2ReactiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.OpenBranchSide1ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.OpenBranchSide1ReactiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.OpenBranchSide2ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.OpenBranchSide2ReactiveFlowEquationTerm;
import com.powsybl.openloadflow.equations.EquationTermArray;
import com.powsybl.openloadflow.util.Evaluable;
import net.jafama.DoubleWrapper;
import net.jafama.FastMath;

import static com.powsybl.openloadflow.network.PiModel.A2;

/**
 * A closed branch's active and reactive flows as the vectorized network computes them, at a state given explicitly.
 *
 * <p>With a vectorized equation system, a closed branch's P and Q evaluables return the {@link AcBranchVector}'s
 * {@code p1}, {@code q1}, {@code p2}, {@code q2} arrays, which {@code AcNetworkVector.updateClosedBranch} fills from the
 * vector's own cached parameters (refreshed on tap changes), and the one {@code FastMath.sinAndCos} of the angle
 * difference that the vector combines with the sine and cosine of ksi. These methods make the same calls on the same doubles, so the value is the array's, bit
 * for bit, at that state. When the branch has no RHO1 or ALPHA1 variable, its {@code r1} and {@code a1} are the
 * vector's cached ones ({@link #r1}, {@link #a1}), not the pi model's.
 *
 * <p>This class sits in OLF's package (a classpath split package, like {@code util.MatrixFactoryUtil}) to read the
 * vector's package-private arrays; it never writes them.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBranchVectorValues {

    private GpuBranchVectorValues() {
    }

    /** The branch vector a vectorized closed term evaluates from, or null when the term is not an element of one of the
     *  vectorized network's term arrays (a scalar equation system). */
    public static AcBranchVector of(Evaluable closedTerm) {
        if (closedTerm instanceof EquationTermArray.EquationTermArrayElementImpl<?, ?> element
                && element.getEvaluator() instanceof AbstractBranchEquationTermArrayEvaluator evaluator) {
            return evaluator.branchVector;
        }
        return null;
    }

    /** The ratio the vector uses when the branch has no RHO1 variable. */
    public static double r1(AcBranchVector vector, int branchNum) {
        return vector.r1[branchNum];
    }

    /** The phase the vector uses when the branch has no ALPHA1 variable. */
    public static double a1(AcBranchVector vector, int branchNum) {
        return vector.a1[branchNum];
    }

    /**
     * The sine of the angle difference d = ph2 - ph1 - a1 + A2 in {@code w}'s return, its cosine in {@code w.value}: the
     * one sine and cosine {@code AcNetworkVector.updateClosedBranch} computes for a closed branch, combined there with
     * those of ksi (theta1 = ksi + d, theta2 = ksi - d).
     */
    private static double sinD(double ph1, double a1, double ph2, DoubleWrapper w) {
        return FastMath.sinAndCos(ph2 - ph1 - a1 + A2, w);
    }

    /** {@code p1[branchNum]} after a state update at this state. */
    public static double p1(AcBranchVector vector, int branchNum, double v1, double ph1, double r1, double a1, double v2,
                            double ph2, DoubleWrapper w) {
        double sinD = sinD(ph1, a1, ph2, w);
        double cosD = w.value;
        double sinTheta1 = vector.sinKsi[branchNum] * cosD + vector.cosKsi[branchNum] * sinD;
        return ClosedBranchSide1ActiveFlowEquationTerm.p1(vector.y[branchNum], vector.sinKsi[branchNum], vector.g1[branchNum],
                v1, r1, v2, sinTheta1);
    }

    /** {@code q1[branchNum]} after a state update at this state. */
    public static double q1(AcBranchVector vector, int branchNum, double v1, double ph1, double r1, double a1, double v2,
                            double ph2, DoubleWrapper w) {
        double sinD = sinD(ph1, a1, ph2, w);
        double cosD = w.value;
        double cosTheta1 = vector.cosKsi[branchNum] * cosD - vector.sinKsi[branchNum] * sinD;
        return ClosedBranchSide1ReactiveFlowEquationTerm.q1(vector.y[branchNum], vector.cosKsi[branchNum], vector.b1[branchNum],
                v1, r1, v2, cosTheta1);
    }

    /** {@code p2[branchNum]} after a state update at this state. */
    public static double p2(AcBranchVector vector, int branchNum, double v1, double ph1, double r1, double a1, double v2,
                            double ph2, DoubleWrapper w) {
        double sinD = sinD(ph1, a1, ph2, w);
        double cosD = w.value;
        double sinTheta2 = vector.sinKsi[branchNum] * cosD - vector.cosKsi[branchNum] * sinD;
        return ClosedBranchSide2ActiveFlowEquationTerm.p2(vector.y[branchNum], vector.sinKsi[branchNum], vector.g2[branchNum],
                v1, r1, v2, sinTheta2);
    }

    /** {@code q2[branchNum]} after a state update at this state. */
    public static double q2(AcBranchVector vector, int branchNum, double v1, double ph1, double r1, double a1, double v2,
                            double ph2, DoubleWrapper w) {
        double sinD = sinD(ph1, a1, ph2, w);
        double cosD = w.value;
        double cosTheta2 = vector.cosKsi[branchNum] * cosD + vector.sinKsi[branchNum] * sinD;
        return ClosedBranchSide2ReactiveFlowEquationTerm.q2(vector.y[branchNum], vector.cosKsi[branchNum], vector.b2[branchNum],
                v1, r1, v2, cosTheta2);
    }

    /** {@code openP1[branchNum]} (side 2 open, bus 1 live) after a state update at this state, with the pi model's r1. */
    public static double openP1(AcBranchVector vector, int branchNum, double v1, double r1) {
        return OpenBranchSide2ActiveFlowEquationTerm.p1(vector.y[branchNum], vector.cosKsi[branchNum], vector.sinKsi[branchNum],
                vector.g1[branchNum], vector.g2[branchNum], vector.b2[branchNum], v1, r1);
    }

    /** {@code openQ1[branchNum]} (side 2 open, bus 1 live) after a state update at this state, with the pi model's r1. */
    public static double openQ1(AcBranchVector vector, int branchNum, double v1, double r1) {
        return OpenBranchSide2ReactiveFlowEquationTerm.q1(vector.y[branchNum], vector.cosKsi[branchNum], vector.sinKsi[branchNum],
                vector.b1[branchNum], vector.g2[branchNum], vector.b2[branchNum], v1, r1);
    }

    /** {@code openP2[branchNum]} (side 1 open, bus 2 live) after a state update at this state. */
    public static double openP2(AcBranchVector vector, int branchNum, double v2) {
        return OpenBranchSide1ActiveFlowEquationTerm.p2(vector.y[branchNum], vector.cosKsi[branchNum], vector.sinKsi[branchNum],
                vector.g1[branchNum], vector.b1[branchNum], vector.g2[branchNum], v2);
    }

    /** {@code openQ2[branchNum]} (side 1 open, bus 2 live) after a state update at this state. */
    public static double openQ2(AcBranchVector vector, int branchNum, double v2) {
        return OpenBranchSide1ReactiveFlowEquationTerm.q2(vector.y[branchNum], vector.cosKsi[branchNum], vector.sinKsi[branchNum],
                vector.g1[branchNum], vector.b1[branchNum], vector.b2[branchNum], v2);
    }
}
