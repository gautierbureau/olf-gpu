/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CouplerTreeReHoming;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;

import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

/**
 * The GPU equation system AS TEXT, the counterpart of {@link EquationSystem#write} on the CPU side.
 *
 * <p>OLF can print its equations ({@code P1 = ac_p_closed_1(...) + ...}), which is how a wrong equation is
 * normally caught. The GPU system has no such reading: it is a set of index PACKS that say which row each
 * element's output is scattered into, so "our equations" only exist implicitly, spread across
 * {@code closedBranchIndices}/{@code openBranchIndices}/{@code shuntIndices}/{@code hvdcAcEmulationIndices}/{@code zeroImpedanceBranchIndices} and the DISTR_Q / DISTR_RHO tables.
 * This writes them out the same way round as OLF: one line per DEVICE ROW, naming the variable that row
 * belongs to and every element output that lands there. A row with no contribution that is not an identity,
 * or a coupling that appears with the same sign twice, is then as visible as it is in OLF's own dump.
 *
 * <p>The ZERO-IMPEDANCE section is printed per branch with ALL its row slots, {@code -1}s included, because
 * a dropped term is exactly a {@code -1} where a row was expected: a {@code DUMMY_Q} needs an OPPOSING pair
 * (+1 in one endpoint's Q row, -1 in the other's) to be determined at all, so a suppressed endpoint row
 * turns the column into a free variable and the matrix loses rank.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuEquationDump {

    private GpuEquationDump() {
    }

    private static String elementId(LfNetwork network, AcVariableType type, int num) {
        try {
            return switch (type) {
                case BUS_V, BUS_PHI -> network.getBus(num).getId();
                case BRANCH_RHO1, BRANCH_ALPHA1, DUMMY_P, DUMMY_Q -> network.getBranch(num).getId();
                case SHUNT_B -> network.getShunts().get(num).getId();
                default -> "#" + num;
            };
        } catch (RuntimeException e) {
            return "#" + num;                                // a diagnostic never fails the run it describes
        }
    }

    private static void add(List<String>[] into, int row, String what) {
        if (row >= 0 && row < into.length) {
            into[row].add(what);
        }
    }

    /**
     * Writes the dump. {@code rowMode} may be null (then no mode column); pass the activation's row modes to
     * see which rows are identities, power equations (1), slope (2), phase (3) or RL-pinned members (4).
     */
    @SuppressWarnings("unchecked")
    public static void write(Writer w, LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                             GpuAcData d, int[] rowMode) throws IOException {
        List<Variable<AcVariableType>> vars = es.getIndex().getSortedVariablesToFind();
        int n = vars.size();
        Variable<AcVariableType>[] varOfRow = new Variable[n];
        for (Variable<AcVariableType> v : vars) {
            if (v.getRow() >= 0 && v.getRow() < n) {
                varOfRow[v.getRow()] = v;
            }
        }
        List<String>[] terms = new List[n];
        for (int r = 0; r < n; r++) {
            terms[r] = new ArrayList<>();
        }

        int[] closedBranchIndices = d.closedBranch().indices();
        for (int e = 0; e < closedBranchIndices.length / 12; e++) {
            int o = e * 12;
            add(terms, closedBranchIndices[o], "cb#" + e + ".p1");
            add(terms, closedBranchIndices[o + 1], "cb#" + e + ".q1");
            add(terms, closedBranchIndices[o + 2], "cb#" + e + ".p2");
            add(terms, closedBranchIndices[o + 3], "cb#" + e + ".q2");
            add(terms, closedBranchIndices[o + 10], "cb#" + e + ".q1@alt");
            add(terms, closedBranchIndices[o + 11], "cb#" + e + ".q2@alt");
        }
        int[] openBranchIndices = d.openBranch().indices();
        for (int e = 0; e < openBranchIndices.length / 6; e++) {
            int o = e * 6;
            add(terms, openBranchIndices[o], "ob#" + e + ".p1");
            add(terms, openBranchIndices[o + 1], "ob#" + e + ".q1");
            add(terms, openBranchIndices[o + 2], "ob#" + e + ".p2");
            add(terms, openBranchIndices[o + 3], "ob#" + e + ".q2");
        }
        int[] shuntIndices = d.shunt().indices();
        for (int e = 0; e < shuntIndices.length / 5; e++) {
            int o = e * 5;
            add(terms, shuntIndices[o], "sh#" + e + ".p");
            add(terms, shuntIndices[o + 1], "sh#" + e + ".q");
            add(terms, shuntIndices[o + 4], "sh#" + e + ".q@alt");
        }
        int[] hvdcAcEmulationIndices = d.hvdcAcEmulation().indices();
        for (int e = 0; e < hvdcAcEmulationIndices.length / 4; e++) {
            add(terms, hvdcAcEmulationIndices[e * 4], "hv#" + e + ".p1");
            add(terms, hvdcAcEmulationIndices[e * 4 + 1], "hv#" + e + ".p2");
        }
        int[] zeroImpedanceBranchIndices = d.zeroImpedanceBranch().indices();
        int[] zeroImpedanceBranchNums = d.zeroImpedanceBranch().branchNum();
        for (int e = 0; e < zeroImpedanceBranchIndices.length / 11; e++) {
            int o = e * 11;
            String tag = "zi#" + e;
            add(terms, zeroImpedanceBranchIndices[o], tag + ".p1(+dP)");
            add(terms, zeroImpedanceBranchIndices[o + 1], tag + ".q1(+dQ)");
            add(terms, zeroImpedanceBranchIndices[o + 2], tag + ".p2(-dP)");
            add(terms, zeroImpedanceBranchIndices[o + 3], tag + ".q2(-dQ)");
            if ((zeroImpedanceBranchIndices[o + 10] & 1) != 0) {
                int childSide = CouplerTreeReHoming.childSide(zeroImpedanceBranchIndices[o + 10]);
                // stock: crosswise, ZERO_V on the DUMMY_Q row and ZERO_PHI on the DUMMY_P row (native zeroVRowOf)
                int zv = childSide == 0 ? zeroImpedanceBranchIndices[o + 5] : (childSide == 1 ? zeroImpedanceBranchIndices[o + 6] : zeroImpedanceBranchIndices[o + 7]);
                int zp = childSide == 0 ? zeroImpedanceBranchIndices[o + 4] : (childSide == 1 ? zeroImpedanceBranchIndices[o + 8] : zeroImpedanceBranchIndices[o + 9]);
                add(terms, zv, tag + ".ZERO_V");
                add(terms, zp, tag + ".ZERO_PHI");
            }
        }
        int[] distrQRow = d.distrQ().row();
        for (int k = 0; k < distrQRow.length; k++) {
            add(terms, distrQRow[k], String.format("dq[k=%d elem=%d side=%d kind=%d w=%.6g]",
                    k, d.distrQ().element()[k], d.distrQ().side()[k], d.distrQ().kind()[k], d.distrQ().weight()[k]));
        }
        for (int k = 0; k < d.distrRho().row().length; k++) {
            add(terms, d.distrRho().row()[k], "dr[col=" + d.distrRho().column()[k] + "]");
        }
        for (int k = 0; k < d.distrShuntB().row().length; k++) {
            add(terms, d.distrShuntB().row()[k], "db[col=" + d.distrShuntB().column()[k] + "]");
        }
        int[] loadModel = d.loadModel().indices();
        for (int e = 0; e < loadModel.length / 4; e++) {
            add(terms, loadModel[e * 4], "loadModel#" + e + ".p");
            add(terms, loadModel[e * 4 + 1], "loadModel#" + e + ".q");
        }

        w.write("# GPU equation system: one line per device row (row = the variable it is paired with)\n");
        w.write("# n=" + n + "\n");
        for (int r = 0; r < n; r++) {
            Variable<AcVariableType> v = varOfRow[r];
            String name = v == null ? "?" : v.getType() + "(" + elementId(network, v.getType(), v.getElementNum()) + ")";
            w.write(String.format("row=%d %s mode=%s terms=%s%n", r, name,
                    rowMode == null ? "-" : Integer.toString(rowMode[r]), terms[r]));
        }

        w.write("\n# ZERO-IMPEDANCE branches, every row slot including -1 (a -1 is a DROPPED term)\n");
        int[] zeroImpedanceBranchAlternateReactiveDestination = d.zeroImpedanceBranch().alternateReactiveDestination();
        for (int e = 0; e < zeroImpedanceBranchIndices.length / 11; e++) {
            int o = e * 11;
            LfBranch br = e < zeroImpedanceBranchNums.length && zeroImpedanceBranchNums[e] >= 0 ? network.getBranch(zeroImpedanceBranchNums[e]) : null;
            LfBus b1 = br == null ? null : br.getBus1();
            LfBus b2 = br == null ? null : br.getBus2();
            w.write(String.format("zi#%d branch=%s b1=%s b2=%s pRow1=%d qRow1=%d pRow2=%d qRow2=%d"
                            + " dummyP=%d dummyQ=%d spanningTree=%d childSide=%d alternateQRows=[%d,%d]%n",
                    e, br == null ? "?" : br.getId(), b1 == null ? "-" : b1.getId(), b2 == null ? "-" : b2.getId(),
                    zeroImpedanceBranchIndices[o], zeroImpedanceBranchIndices[o + 1], zeroImpedanceBranchIndices[o + 2],
                    zeroImpedanceBranchIndices[o + 3], zeroImpedanceBranchIndices[o + 4], zeroImpedanceBranchIndices[o + 5],
                    zeroImpedanceBranchIndices[o + 10] & 1, CouplerTreeReHoming.childSide(zeroImpedanceBranchIndices[o + 10]),
                    e * 2 + 1 < zeroImpedanceBranchAlternateReactiveDestination.length ? zeroImpedanceBranchAlternateReactiveDestination[e * 2] : -1,
                    e * 2 + 1 < zeroImpedanceBranchAlternateReactiveDestination.length ? zeroImpedanceBranchAlternateReactiveDestination[e * 2 + 1] : -1));
        }

        // The invariant a rank-deficient assembly breaks: a DUMMY_Q (or DUMMY_P) is determined only by an
        // OPPOSING pair of couplings. One side suppressed (-1) leaves the column free.
        w.write("\n# couplers whose dummy coupling is INCOMPLETE (one endpoint row suppressed)\n");
        for (int e = 0; e < zeroImpedanceBranchIndices.length / 11; e++) {
            int o = e * 11;
            boolean pIncomplete = zeroImpedanceBranchIndices[o] < 0 ^ zeroImpedanceBranchIndices[o + 2] < 0;
            boolean qIncomplete = zeroImpedanceBranchIndices[o + 1] < 0 ^ zeroImpedanceBranchIndices[o + 3] < 0;
            if (pIncomplete || qIncomplete) {
                LfBranch br = e < zeroImpedanceBranchNums.length && zeroImpedanceBranchNums[e] >= 0 ? network.getBranch(zeroImpedanceBranchNums[e]) : null;
                w.write(String.format("zi#%d branch=%s %s%s pRows=[%d,%d] qRows=[%d,%d] alternateQRows=[%d,%d]%n",
                        e, br == null ? "?" : br.getId(), pIncomplete ? "P " : "", qIncomplete ? "Q " : "",
                        zeroImpedanceBranchIndices[o], zeroImpedanceBranchIndices[o + 2], zeroImpedanceBranchIndices[o + 1], zeroImpedanceBranchIndices[o + 3],
                        e * 2 + 1 < zeroImpedanceBranchAlternateReactiveDestination.length ? zeroImpedanceBranchAlternateReactiveDestination[e * 2] : -1,
                        e * 2 + 1 < zeroImpedanceBranchAlternateReactiveDestination.length ? zeroImpedanceBranchAlternateReactiveDestination[e * 2 + 1] : -1));
            }
        }

        // The GROUPS behind the suppressed rows: a member's Q row is suppressed because its balance lives in
        // the group's DISTR_Q row, and the pilot's own balance rides the FREE member's row. Whether that
        // arrangement survives an RL pin depends on the candidate table (alternateQRow = -2-group) and on each
        // member's alternate, so print them together with the reasons a table is skipped.
        w.write("\n# remote generator voltage-control groups (pilot, members, rows, candidate table)\n");
        ExtractionColumns cols = ExtractionColumns.of(network, es);
        int[] qRow = cols.qRow();
        int[] alternateQRow = cols.alternateQRow();
        int[] vCol = cols.vCol();
        for (LfBus bus : network.getBuses()) {
            var voltageControl = bus.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == com.powsybl.openloadflow.network.VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (voltageControl == null) {
                continue;
            }
            List<LfBus> members = voltageControl.getMergedControllerElements();
            StringBuilder sb = new StringBuilder();
            for (LfBus m : members) {
                sb.append(String.format(" %s[vCol=%d qRow=%d alternateQRow=%d enabled=%b]", m.getId(), vCol[m.getNum()],
                        qRow[m.getNum()], alternateQRow[m.getNum()], voltageControl.isControllerEnabled(m)));
            }
            w.write(String.format("gvc pilot=%s controlled=%s local=%b pilotIsController=%b vCol=%d qRow=%d"
                            + " alternateQRow=%d members=%d:%s%n",
                    bus.getId(), voltageControl.getControlledBus().getId(), voltageControl.isLocalControl(), members.contains(bus),
                    vCol[bus.getNum()], qRow[bus.getNum()], alternateQRow[bus.getNum()], members.size(), sb));
        }
        w.flush();
    }
}
