/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations;

import com.powsybl.openloadflow.util.Evaluable;
import net.jafama.FastMath;

/**
 * A branch flow term's value at a state given explicitly, instead of read from the equation system's state vector.
 *
 * <p>Each method calls the static function the term's own {@code eval()} calls, with the parameters the term holds
 * (its construction snapshot, which OLF refreshes on tap changes) and the state values passed in. The same doubles go
 * through the same calls, so the value is the term's, bit for bit, at that state. The open terms' {@code eval()} takes
 * {@code FastMath.cos(ksi)} and {@code FastMath.sin(ksi)}, and so do these.
 *
 * <p>This class sits in OLF's package (a classpath split package, like {@code util.MatrixFactoryUtil}) to read the
 * terms' protected parameters; it never writes them.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBranchTermValues {

    private GpuBranchTermValues() {
    }

    /** {@link ClosedBranchSide1CurrentMagnitudeEquationTerm#eval()} at the given state. */
    public static double closedI1(Evaluable term, double v1, double ph1, double r1,
                                  double a1, double v2, double ph2) {
        ClosedBranchSide1CurrentMagnitudeEquationTerm t = (ClosedBranchSide1CurrentMagnitudeEquationTerm) term;
        return ClosedBranchSide1CurrentMagnitudeEquationTerm.i1(t.y, t.ksi, t.g1, t.b1, v1, ph1, r1, a1, v2, ph2);
    }

    /** {@link ClosedBranchSide2CurrentMagnitudeEquationTerm#eval()} at the given state. */
    public static double closedI2(Evaluable term, double v1, double ph1, double r1,
                                  double a1, double v2, double ph2) {
        ClosedBranchSide2CurrentMagnitudeEquationTerm t = (ClosedBranchSide2CurrentMagnitudeEquationTerm) term;
        return ClosedBranchSide2CurrentMagnitudeEquationTerm.i2(t.y, t.ksi, t.g2, t.b2, v1, ph1, r1, a1, v2, ph2);
    }

    /** {@link OpenBranchSide2ActiveFlowEquationTerm#eval()} (side 2 open, bus 1 live) at the given state. */
    public static double openP1(Evaluable term, double v1, double r1) {
        OpenBranchSide2ActiveFlowEquationTerm t = (OpenBranchSide2ActiveFlowEquationTerm) term;
        return OpenBranchSide2ActiveFlowEquationTerm.p1(t.y, FastMath.cos(t.ksi), FastMath.sin(t.ksi), t.g1, t.g2, t.b2, v1, r1);
    }

    /** {@link OpenBranchSide2ReactiveFlowEquationTerm#eval()} (side 2 open, bus 1 live) at the given state. */
    public static double openQ1(Evaluable term, double v1, double r1) {
        OpenBranchSide2ReactiveFlowEquationTerm t = (OpenBranchSide2ReactiveFlowEquationTerm) term;
        return OpenBranchSide2ReactiveFlowEquationTerm.q1(t.y, FastMath.cos(t.ksi), FastMath.sin(t.ksi), t.b1, t.g2, t.b2, v1, r1);
    }

    /** {@link OpenBranchSide2CurrentMagnitudeEquationTerm#eval()} (side 2 open, bus 1 live) at the given state. */
    public static double openI1(Evaluable term, double v1, double ph1, double r1) {
        OpenBranchSide2CurrentMagnitudeEquationTerm t = (OpenBranchSide2CurrentMagnitudeEquationTerm) term;
        return OpenBranchSide2CurrentMagnitudeEquationTerm.i1(t.y, FastMath.cos(t.ksi), FastMath.sin(t.ksi), t.g1, t.b1, t.g2,
                t.b2, v1, ph1, r1);
    }

    /** {@link OpenBranchSide1ActiveFlowEquationTerm#eval()} (side 1 open, bus 2 live) at the given state. */
    public static double openP2(Evaluable term, double v2) {
        OpenBranchSide1ActiveFlowEquationTerm t = (OpenBranchSide1ActiveFlowEquationTerm) term;
        return OpenBranchSide1ActiveFlowEquationTerm.p2(t.y, FastMath.cos(t.ksi), FastMath.sin(t.ksi), t.g1, t.b1, t.g2, v2);
    }

    /** {@link OpenBranchSide1ReactiveFlowEquationTerm#eval()} (side 1 open, bus 2 live) at the given state. */
    public static double openQ2(Evaluable term, double v2) {
        OpenBranchSide1ReactiveFlowEquationTerm t = (OpenBranchSide1ReactiveFlowEquationTerm) term;
        return OpenBranchSide1ReactiveFlowEquationTerm.q2(t.y, FastMath.cos(t.ksi), FastMath.sin(t.ksi), t.g1, t.b1, t.b2, v2);
    }

    /** {@link OpenBranchSide1CurrentMagnitudeEquationTerm#eval()} (side 1 open, bus 2 live) at the given state. */
    public static double openI2(Evaluable term, double v2, double ph2) {
        OpenBranchSide1CurrentMagnitudeEquationTerm t = (OpenBranchSide1CurrentMagnitudeEquationTerm) term;
        return OpenBranchSide1CurrentMagnitudeEquationTerm.i2(t.y, FastMath.cos(t.ksi), FastMath.sin(t.ksi), t.g1, t.b1, t.g2,
                t.b2, v2, ph2);
    }
}
