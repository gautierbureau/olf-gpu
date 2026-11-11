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
import com.powsybl.openloadflow.equations.EquationTermArray;
import com.powsybl.openloadflow.util.Evaluable;
import net.jafama.DoubleWrapper;
import net.jafama.FastMath;

/**
 * A closed branch's active and reactive flows as the vectorized network computes them, at a state given explicitly.
 *
 * <p>With a vectorized equation system, a closed branch's P and Q evaluables return the {@link AcBranchVector}'s
 * {@code p1}, {@code q1}, {@code p2}, {@code q2} arrays, which {@code AcNetworkVector.updateClosedBranch} fills from the
 * vector's own cached parameters (refreshed on tap changes), {@code AcNetworkVector.theta1/theta2} and
 * {@code FastMath.sinAndCos}. These methods make the same calls on the same doubles, so the value is the array's, bit
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

    /** {@code p1[branchNum]} after a state update at this state. */
    public static double p1(AcBranchVector vector, int branchNum, double v1, double ph1, double r1, double a1, double v2,
                            double ph2, DoubleWrapper w) {
        double sinTheta1 = FastMath.sinAndCos(AcNetworkVector.theta1(vector.ksi[branchNum], ph1, a1, ph2), w);
        return ClosedBranchSide1ActiveFlowEquationTerm.p1(vector.y[branchNum], vector.sinKsi[branchNum], vector.g1[branchNum],
                v1, r1, v2, sinTheta1);
    }

    /** {@code q1[branchNum]} after a state update at this state. */
    public static double q1(AcBranchVector vector, int branchNum, double v1, double ph1, double r1, double a1, double v2,
                            double ph2, DoubleWrapper w) {
        FastMath.sinAndCos(AcNetworkVector.theta1(vector.ksi[branchNum], ph1, a1, ph2), w);
        return ClosedBranchSide1ReactiveFlowEquationTerm.q1(vector.y[branchNum], vector.cosKsi[branchNum], vector.b1[branchNum],
                v1, r1, v2, w.value);
    }

    /** {@code p2[branchNum]} after a state update at this state. */
    public static double p2(AcBranchVector vector, int branchNum, double v1, double ph1, double r1, double a1, double v2,
                            double ph2, DoubleWrapper w) {
        double sinTheta2 = FastMath.sinAndCos(AcNetworkVector.theta2(vector.ksi[branchNum], ph1, a1, ph2), w);
        return ClosedBranchSide2ActiveFlowEquationTerm.p2(vector.y[branchNum], vector.sinKsi[branchNum], vector.g2[branchNum],
                v1, r1, v2, sinTheta2);
    }

    /** {@code q2[branchNum]} after a state update at this state. */
    public static double q2(AcBranchVector vector, int branchNum, double v1, double ph1, double r1, double a1, double v2,
                            double ph2, DoubleWrapper w) {
        FastMath.sinAndCos(AcNetworkVector.theta2(vector.ksi[branchNum], ph1, a1, ph2), w);
        return ClosedBranchSide2ReactiveFlowEquationTerm.q2(vector.y[branchNum], vector.cosKsi[branchNum], vector.b2[branchNum],
                v1, r1, v2, w.value);
    }
}
