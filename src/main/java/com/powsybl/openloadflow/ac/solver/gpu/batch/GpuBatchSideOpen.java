/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.PiModel;

/**
 * A branch open at ONE end, expressed as a per-scenario pi-model row.
 *
 * <p>Seen from the bus that is still connected, a dangling branch is a PURE SHUNT: the open side's
 * charging sits behind the series impedance, and the two reduce to a single equivalent admittance
 *
 * <pre>    Y_eq = Y_shunt(live side) + Y_series (+) Y_shunt(open side)</pre>
 *
 * <p>so the branch can stay LIVE with its pi row overridden to {@code y = 0} (no series coupling, so
 * the far bus neither receives nor contributes anything) and the live side's g/b carrying Y_eq. That
 * needs no new kernel — it is the same per-scenario override the tap channel uses.
 *
 * <p>The alternative the batch used to take, masking the element outright, DELETES that injection.
 * OLF keeps it: its own {@code testLineOpenOneSideContingency} measures -55.873 MVAr on this shape.
 * The error is exactly the equivalent shunt, which for a symmetric line is about twice the half
 * charging — measured on rte6515 BUS-1003_BBS as -2 x b_side x v^2 at every bus bordering an opened
 * side, and on the 9-bus reproducer as 5.23 kV at VL2.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchSideOpen {

    private GpuBatchSideOpen() {
    }

    /**
     * The cb pi row {@code {y, g1, g2, b1, b2, ksi, r1, a1}} that makes {@code branch} behave as if
     * {@code openSide} (a GpuTwoSides) were disconnected, or {@code null} when the shape is not expressible this way
     * and the caller should fall back to masking the element.
     *
     * <p>The RATIO needs no special handling, which is worth stating because it looks like it should.
     * OLF's own open-side-2 terms are
     *
     * <pre>    p1 =  r1^2 v1^2 (g1 + y^2 g2/shunt + (b2^2+g2^2) y sinKsi/shunt)
     *     q1 = -r1^2 v1^2 (b1 + y^2 b2/shunt - (b2^2+g2^2) y cosKsi/shunt)</pre>
     *
     * with {@code shunt = (g2 + y sinKsi)^2 + (-b2 + y cosKsi)^2}, and expanding
     * {@code Y_series (+) Y_open} reproduces those brackets TERM FOR TERM — the reduction below IS
     * OLF's formula. r1^2 multiplies the WHOLE bracket, b1 included, so carrying the branch's own r1
     * into the override row and letting the closed-branch evaluation apply r1^2 is exact. a1 drops out
     * entirely: with y = 0 there are no cross terms for a phase shift to rotate.
     */
    static double[] danglingClosedBranchParameterRow(LfBranch branch, int openSide) {
        PiModel pi = branch.getPiModel();
        // OLF's series admittance from (y, ksi): G = y sin(ksi), B = -y cos(ksi).
        double gs = pi.getY() * Math.sin(pi.getKsi());
        double bs = -pi.getY() * Math.cos(pi.getKsi());
        double gOpen = openSide == GpuTwoSides.ONE ? pi.getG1() : pi.getG2();
        double bOpen = openSide == GpuTwoSides.ONE ? pi.getB1() : pi.getB2();
        double gLive = openSide == GpuTwoSides.ONE ? pi.getG2() : pi.getG1();
        double bLive = openSide == GpuTwoSides.ONE ? pi.getB2() : pi.getB1();
        // Y_series (+) Y_open, i.e. (Ys * Yo) / (Ys + Yo) in complex arithmetic.
        double sumG = gs + gOpen;
        double sumB = bs + bOpen;
        double den = sumG * sumG + sumB * sumB;
        double combG = 0;
        double combB = 0;
        if (den != 0) {
            double numG = gs * gOpen - bs * bOpen;
            double numB = gs * bOpen + bs * gOpen;
            combG = (numG * sumG + numB * sumB) / den;
            combB = (numB * sumG - numG * sumB) / den;
        }
        double gEq = gLive + combG;
        double bEq = bLive + combB;
        //     y    g1                    g2                    b1                    b2                 ksi r1 a1
        // r1/a1 carried through: the cb evaluation applies r1^2 to the live side exactly as OLF does.
        return openSide == GpuTwoSides.ONE
                ? new double[] {0, 0, gEq, 0, bEq, 0, pi.getR1(), pi.getA1()}
                : new double[] {0, gEq, 0, bEq, 0, 0, pi.getR1(), pi.getA1()};
    }
}
