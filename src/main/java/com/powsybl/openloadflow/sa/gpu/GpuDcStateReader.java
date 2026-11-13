/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.dc.equations.ClosedBranchDcCurrent;
import com.powsybl.openloadflow.equations.EquationTerm;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.PiModel;
import com.powsybl.openloadflow.util.Evaluable;
import net.jafama.FastMath;

/**
 * A DC branch's flows, currents and apparent powers read from a state array instead of from OLF's model: what the
 * branch's evaluables give once that state is loaded into the equation system, computed without loading it.
 *
 * <p>OLF's DC model, operation for operation: the closed flow {@code DcNetworkVector.updateClosedBranches} computes
 * ({@code deltaPhase = phi2 - phi1 + A2 - a1}, {@code p1 = -power * deltaPhase}, {@code p2 = power * deltaPhase}, the
 * shift from the state when it is a variable), zero for a branch without both angles or of zero impedance; the
 * current of {@link ClosedBranchDcCurrent} ({@code |p| / dcPowerFactor}); the apparent power of
 * {@code AbstractLfBranch.computeApparentPower1/2} ({@code sqrt(p * p + q * q)}, q a constant in DC). A branch whose
 * evaluable is not its closed flow term (open at one side or both: OLF swaps in a constant), or no equation term at all
 * (a switch's NaN), gives that evaluable's value. The branch arrays are the device's (the same rows, shifts and powers the Woodbury kernels read).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuDcStateReader {

    private final int[] ph1Row;
    private final int[] ph2Row;
    private final int[] a1Row;
    private final double[] a1Const;
    private final double[] power;

    GpuDcStateReader(int[] ph1Row, int[] ph2Row, int[] a1Row, double[] a1Const, double[] power) {
        this.ph1Row = ph1Row;
        this.ph2Row = ph2Row;
        this.a1Row = a1Row;
        this.a1Const = a1Const;
        this.power = power;
    }

    /** {@code branch.getP1().eval()} (side 1) or {@code getP2()} (side 2) with {@code state} loaded. */
    double p(double[] state, LfBranch branch, int side) {
        Evaluable p = side == GpuTwoSides.ONE ? branch.getP1() : branch.getP2();
        Evaluable closed = side == GpuTwoSides.ONE ? branch.getClosedP1() : branch.getClosedP2();
        if (p != closed || !(p instanceof EquationTerm)) {
            return p.eval();                                 // a constant: an open side OLF swapped in, or a switch's NaN
        }
        int b = branch.getNum();
        if (ph1Row[b] < 0 || ph2Row[b] < 0) {
            return 0;
        }
        double a1 = a1Row[b] >= 0 ? state[a1Row[b]] : a1Const[b];
        double deltaPhase = state[ph2Row[b]] - state[ph1Row[b]] + PiModel.A2 - a1;
        return side == GpuTwoSides.ONE ? -power[b] * deltaPhase : power[b] * deltaPhase;
    }

    /** {@code branch.getI1().eval()} or {@code getI2()} with {@code state} loaded. */
    double i(double[] state, LfBranch branch, int side, double dcPowerFactor) {
        Evaluable i = side == GpuTwoSides.ONE ? branch.getI1() : branch.getI2();
        if (!(i instanceof ClosedBranchDcCurrent)) {
            return i.eval();
        }
        return Math.abs(p(state, branch, side)) / dcPowerFactor;
    }

    /** {@code branch.computeApparentPower1()} or {@code 2} with {@code state} loaded. */
    double s(double[] state, LfBranch branch, int side) {
        double p = p(state, branch, side);
        double q = side == GpuTwoSides.ONE ? branch.getQ1().eval() : branch.getQ2().eval();
        return FastMath.sqrt(p * p + q * q);
    }

    /** The limit scanner's quantity of a branch side ({@link GpuLimitTable#QTY_I} / {@code QTY_P} / {@code QTY_S}). */
    double branch(double[] state, LfBranch branch, int side, byte quantity, double dcPowerFactor) {
        return switch (quantity) {
            case GpuLimitTable.QTY_I -> i(state, branch, side, dcPowerFactor);
            case GpuLimitTable.QTY_P -> p(state, branch, side);
            default -> s(state, branch, side);
        };
    }
}
