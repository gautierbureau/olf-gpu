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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Stable rows (docs/gpu-vs-olf.md section 2.1) hold across transformer voltage control layouts: a Java port of the
 * native buildRowMap, run on the packs extracted with the control OFF and with it ON, must give the same row
 * pairing. The device keeps one fixed matrix for both, so the union pattern only has to supply the off-diagonal
 * entries and buildRowMap needs no layout-specific case. Also checks the regime (Q on the bus V row vs the control
 * variable's row) and that the control variable's column persists when the control is rounded off.
 */
class StableRowsAcrossControlLayoutsTest {

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
        int[] closedBranchIndices = d.closedBranch().indices();
        for (int e = 0; e < closedBranchIndices.length / 12; e++) {          // closedBranchIndices stride 12 (…a1col r1col alternateQRow1 alternateQRow2)
            int b = e * 12;
            pairRc.accept(closedBranchIndices[b], closedBranchIndices[b + 4]);
            pairRc.accept(closedBranchIndices[b + 2], closedBranchIndices[b + 6]);
            pairRc.accept(closedBranchIndices[b + 1], closedBranchIndices[b + 1]);
            pairRc.accept(closedBranchIndices[b + 3], closedBranchIndices[b + 3]);
            claimMovedV.accept(closedBranchIndices[b + 1], closedBranchIndices[b + 5]);
            claimMovedV.accept(closedBranchIndices[b + 3], closedBranchIndices[b + 7]);
            if (closedBranchIndices[b + 9] >= 0) {
                pairRc.accept(closedBranchIndices[b + 9], closedBranchIndices[b + 9]);
            }
            if (closedBranchIndices[b + 8] >= 0) {
                pairRc.accept(closedBranchIndices[b + 8], closedBranchIndices[b + 8]);
            }
        }
        int[] openBranchIndices = d.openBranch().indices();
        for (int e = 0; e < openBranchIndices.length / 6; e++) {
            int b = e * 6;
            pairRc.accept(openBranchIndices[b + 1], openBranchIndices[b + 1]);
            pairRc.accept(openBranchIndices[b + 3], openBranchIndices[b + 3]);
            claimMovedV.accept(openBranchIndices[b + 1], openBranchIndices[b + 4]);
            claimMovedV.accept(openBranchIndices[b + 3], openBranchIndices[b + 5]);
        }
        int[] shuntIndices = d.shunt().indices();
        for (int e = 0; e < shuntIndices.length / 4; e++) {
            int b = e * 4;
            pairRc.accept(shuntIndices[b + 1], shuntIndices[b + 1]);
            claimMovedV.accept(shuntIndices[b + 1], shuntIndices[b + 2]);
            if (shuntIndices[b + 3] >= 0) {
                pairRc.accept(shuntIndices[b + 3], shuntIndices[b + 3]);
            }
        }
        int[] hvdcAcEmulationIndices = d.hvdcAcEmulation().indices();
        for (int e = 0; e < hvdcAcEmulationIndices.length / 4; e++) {
            int b = e * 4;
            pairRc.accept(hvdcAcEmulationIndices[b], hvdcAcEmulationIndices[b + 2]);
            pairRc.accept(hvdcAcEmulationIndices[b + 1], hvdcAcEmulationIndices[b + 3]);
        }
        int[] distrQRow = d.distrQ().row();
        for (int c = 0; c < distrQRow.length; c++) {
            pairRc.accept(distrQRow[c], distrQRow[c]);
        }
        int[] zeroImpedanceBranchIndices = d.zeroImpedanceBranch().indices();
        for (int e = 0; e < zeroImpedanceBranchIndices.length / 11; e++) {
            for (int s = 0; s < 6; s++) {
                pairRc.accept(zeroImpedanceBranchIndices[e * 11 + s], zeroImpedanceBranchIndices[e * 11 + s]);
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
