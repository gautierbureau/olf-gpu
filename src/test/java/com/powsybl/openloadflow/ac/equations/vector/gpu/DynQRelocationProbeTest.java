/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.VoltageControl;
import com.powsybl.openloadflow.network.VoltageControlNetworkFactory;
import com.powsybl.openloadflow.network.impl.Networks;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Empirical feasibility probe (no native build) for Dynamic Q-relocation (§11):
 *  - REGIME check (Q on bus-V row vs. control-var row per control state),
 *  - PRECONDITION check (control-var column persists when control is rounded OFF),
 *  - BIJECTION GATE: a faithful Java port of the native buildRowMap, run on the extracted packs in
 *    BOTH the control-off and control-on layouts of the same network. If rowMap is identical the
 *    shared static pivot assignment serves both layouts (the union only needs to supply the
 *    off-diagonal entries) and buildRowMap needs no change. Prints only.
 */
class DynQRelocationProbeTest {

    private static int row(EquationSystem<AcVariableType, AcEquationType> es, int bus, AcEquationType t) {
        return GpuAcRowIndexing.row(es, bus, t);
    }

    /** Faithful Java port of native buildRowMap. Returns rowMap or null if it would throw. */
    private static int[] buildRowMap(int n, GpuAcData d) {
        int[] rowMap = new int[n];
        Arrays.fill(rowMap, -1);
        boolean[] ok = {true};
        java.util.function.BiConsumer<Integer, Integer> pairRc = (rowI, colI) -> {
            int r = rowI;
            int c = colI;
            if (r < 0 || c < 0) {
                return;
            }
            if (rowMap[r] == -1) {
                rowMap[r] = c;
            } else if (rowMap[r] != c) {
                System.out.printf("    CONFLICT: row %d already -> %d, now -> %d%n", r, rowMap[r], c);
                ok[0] = false;
            }
        };
        java.util.function.BiConsumer<Integer, Integer> claimMovedV = (qrow, vcol) -> {
            if (vcol >= 0 && !qrow.equals(vcol)) {
                pairRc.accept(vcol, vcol);
            }
        };
        int[] cb = d.cbIdx();
        for (int e = 0; e < cb.length / 12; e++) {          // cbIdx stride 12 (…a1col r1col qAlt1 qAlt2)
            int b = e * 12;
            pairRc.accept(cb[b], cb[b + 4]);
            pairRc.accept(cb[b + 2], cb[b + 6]);
            pairRc.accept(cb[b + 1], cb[b + 1]);
            pairRc.accept(cb[b + 3], cb[b + 3]);
            claimMovedV.accept(cb[b + 1], cb[b + 5]);
            claimMovedV.accept(cb[b + 3], cb[b + 7]);
            if (cb[b + 9] >= 0) {
                pairRc.accept(cb[b + 9], cb[b + 9]);
            }
            if (cb[b + 8] >= 0) {
                pairRc.accept(cb[b + 8], cb[b + 8]);
            }
        }
        int[] ob = d.obIdx();
        for (int e = 0; e < ob.length / 6; e++) {
            int b = e * 6;
            pairRc.accept(ob[b + 1], ob[b + 1]);
            pairRc.accept(ob[b + 3], ob[b + 3]);
            claimMovedV.accept(ob[b + 1], ob[b + 4]);
            claimMovedV.accept(ob[b + 3], ob[b + 5]);
        }
        int[] sh = d.shIdx();
        for (int e = 0; e < sh.length / 4; e++) {
            int b = e * 4;
            pairRc.accept(sh[b + 1], sh[b + 1]);
            claimMovedV.accept(sh[b + 1], sh[b + 2]);
            if (sh[b + 3] >= 0) {
                pairRc.accept(sh[b + 3], sh[b + 3]);
            }
        }
        int[] hv = d.hvIdx();
        for (int e = 0; e < hv.length / 4; e++) {
            int b = e * 4;
            pairRc.accept(hv[b], hv[b + 2]);
            pairRc.accept(hv[b + 1], hv[b + 3]);
        }
        int[] dqRow = d.dqRow();
        for (int c = 0; c < dqRow.length; c++) {
            pairRc.accept(dqRow[c], dqRow[c]);
        }
        int[] zi = d.ziIdx();
        for (int e = 0; e < zi.length / 11; e++) {
            for (int s = 0; s < 6; s++) {
                pairRc.accept(zi[e * 11 + s], zi[e * 11 + s]);
            }
        }
        boolean[] colUsed = new boolean[n];
        for (int r = 0; r < n; r++) {
            if (rowMap[r] < 0) {
                System.out.printf("    UNPAIRED row %d%n", r);
                ok[0] = false;
            } else if (colUsed[rowMap[r]]) {
                System.out.printf("    NON-BIJECTION: col %d claimed twice%n", rowMap[r]);
                ok[0] = false;
            } else {
                colUsed[rowMap[r]] = true;
            }
        }
        return ok[0] ? rowMap : null;
    }

    private int[] snapshot(String label, LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es) {
        int n = es.getIndex().getSortedVariablesToFind().size();      // force row/col assignment
        ExtractionColumns c = ExtractionColumns.of(network, es);
        System.out.println("==== " + label + "  (n=" + n + ") ====");
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            boolean vA = row(es, bn, AcEquationType.BUS_TARGET_V) >= 0;
            boolean qA = row(es, bn, AcEquationType.BUS_TARGET_Q) >= 0;
            if (bus.getTransformerVoltageControl().isPresent() && c.vCol()[bn] != c.qRow()[bn]) {
                System.out.printf("  controlled bus %d (%s): vCol=%d qRow=%d  V=%b Q=%b%n",
                        bn, bus.getId(), c.vCol()[bn], c.qRow()[bn], vA, qA);
            }
        }
        for (LfBranch br : network.getBranches()) {
            int rhoCol = c.branchRhoCol()[br.getNum()];
            if (rhoCol >= 0) {
                System.out.printf("  controller %d (%s): branchRhoCol=%d  TARGET_RHO1=%b DISTR_RHO=%b%n",
                        br.getNum(), br.getId(), rhoCol,
                        row(es, br.getNum(), AcEquationType.BRANCH_TARGET_RHO1) >= 0,
                        row(es, br.getNum(), AcEquationType.DISTR_RHO) >= 0);
            }
        }
        GpuAcData d = GpuAcDataExtractor.extract(network, es);
        int[] rowMap = buildRowMap(n, d);
        System.out.println("  buildRowMap -> " + (rowMap == null ? "INVALID (throws)" : "valid bijection"));
        return rowMap;
    }

    private void runCase(String name, Network iidm) {
        LfNetworkParameters params = new LfNetworkParameters()
                .setSlackBusSelector(new FirstSlackBusSelector())
                .setTransformerVoltageControl(true);
        LfNetwork network = Networks.load(iidm, params).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();

        System.out.println("\n######## " + name + " ########");
        int[] off = snapshot("CONTROL OFF (rounded base - regime B)", network, es);

        for (LfBranch b : network.<LfBranch>getControllerElements(VoltageControl.Type.TRANSFORMER)) {
            if (b.isConnectedAtBothSides()) {
                b.setVoltageControlEnabled(true);
            }
        }
        network.fixTransformerVoltageControls();
        int[] on = snapshot("CONTROL ON (continuous)", network, es);

        boolean identical = off != null && on != null && Arrays.equals(off, on);
        System.out.println("  rowMap identical across layouts? " + identical);
        // Feasibility gate for Dynamic Q-relocation (§11.3.1): the shared static pivot assignment must
        // be a valid bijection in BOTH control layouts and be the SAME in both — otherwise buildRowMap
        // would need per-layout regrouping and the dynamic-relocation path would not be viable.
        assertNotNull(off, name + ": buildRowMap invalid in CONTROL-OFF layout");
        assertNotNull(on, name + ": buildRowMap invalid in CONTROL-ON layout");
        assertArrayEquals(off, on, name + ": rowMap differs across control layouts");
    }

    @Test
    void bijectionGateSingleAndMultiController() {
        runCase("SINGLE local transformer VC (T2wt)", tuneT2wt(VoltageControlNetworkFactory.createNetworkWithT2wt()));
        runCase("MULTI-controller shared remote TVC (DISTR_RHO)",
                VoltageControlNetworkFactory.createWithTransformerSharedRemoteControl());
    }

    private static Network tuneT2wt(Network iidm) {
        var t2wt = iidm.getTwoWindingsTransformer("T2wT");
        t2wt.getRatioTapChanger()
                .setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                .setRegulationTerminal(t2wt.getTerminal2()).setTargetV(34.0);
        return iidm;
    }
}
