/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.PiModel;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * The single per-family packing pass of the GPU extraction: one walk of the network that packs
 * every closed / open / zero-impedance branch, then the batch-SA base-open closable branches and
 * zero-impedance switches, then shunts and HVDC AC-emulation links, into the fixed-stride device
 * arrays — reading the precomputed {@link ExtractionColumns}. It also builds the per-family element
 * index maps ({@code closedBranchElementOf} / {@code openBranchElementOf} / {@code shuntElementOf}) the DISTR_Q outer-loop pack
 * later wires reactive contributions through. Kept as ONE pass (not per-family classes): the
 * families share the branch walk and jointly fill the element maps.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuAcFamilyPacker {

    private GpuAcFamilyPacker() {
    }

    /**
     * The packed families and the per-family element-index maps. The {@code *In} / {@code *Idx}
     * lists are fixed-stride rows flattened at assembly; {@code closedBranchElementOf} / {@code openBranchElementOf} /
     * {@code shuntElementOf} map a branch/bus number to its element index (-1 when absent); the
     * {@code baseOpen*} / {@code zeroImpedanceBranchNums} lists carry the batch-SA per-scenario mask metadata.
     */
    record Packs(List<double[]> closedBranchParameters, List<int[]> closedBranchIndices, List<double[]> openBranchParameters, List<int[]> openBranchIndices,
                 List<double[]> shuntParameters, List<int[]> shuntIndices, List<double[]> hvdcAcEmulationParameters, List<int[]> hvdcAcEmulationIndices,
                 List<double[]> zeroImpedanceBranchParameters, List<int[]> zeroImpedanceBranchIndices, List<Integer> zeroImpedanceBranchNums,
                 List<Integer> baseOpenCb, List<Integer> baseOpenZeroImpedanceBranches, List<Integer> baseOpenZeroImpedanceBranchNums,
                 int[] closedBranchElementOf, int[] openBranchElementOf, int[] shuntElementOf, List<Integer> closedBranchPhaseControlSide,
                 List<double[]> loadModelParameters, List<int[]> loadModelIndices,
                 List<int[]> openBranchAlternateReactiveDestination, List<int[]> zeroImpedanceBranchAlternateReactiveDestination) {
    }

    static Packs pack(LfNetwork network, ExtractionColumns columns,
                      Set<String> baseOpenBranchIds, Set<String> baseOpenZeroImpedanceBranchIdList) {
        int[] phiCol = columns.phiCol();
        int[] pRow = columns.pRow();       // a bus's P-balance row: its PHI column, or its parent coupler's DUMMY_P
        int[] vCol = columns.vCol();
        int[] branchRhoCol = columns.branchRhoCol();
        int[] branchAlphaCol = columns.branchAlphaCol();
        int[] dummyPCol = columns.dummyPCol();
        int[] dummyQCol = columns.dummyQCol();
        int[] shuntBCol = columns.shuntBCol();
        int[] qRow = columns.qRow();
        int[] alternateQRow = columns.alternateQRow();                   // Dynamic Q-relocation: 2nd candidate Q row per bus (§11)
        int nBuses = network.getBuses().size();
        int nBranches = network.getBranches().size();

        List<double[]> zeroImpedanceBranchParameters = new ArrayList<>();
        List<int[]> zeroImpedanceBranchIndices = new ArrayList<>();
        List<Integer> zeroImpedanceBranchNums = new ArrayList<>();            // LfBranch num per zero-impedance element (per-scenario masks)

        // closed-branch element index per branch number (-1 = not a closed branch), used to wire the
        // DISTR_Q reactive contributions to their closed-branch q1/q2 outputs.
        int[] closedBranchElementOf = new int[nBranches];
        java.util.Arrays.fill(closedBranchElementOf, -1);

        List<double[]> closedBranchParameters = new ArrayList<>();
        List<int[]> closedBranchIndices = new ArrayList<>();
        List<Integer> closedBranchPhaseControlSide = new ArrayList<>();
        List<double[]> openBranchParameters = new ArrayList<>();
        List<int[]> openBranchIndices = new ArrayList<>();
        // Dynamic Q-relocation for the ob/zero-impedance families (the cb/sh channels' twin): per element, the
        // ALTERNATE Q row of each side's bus — a two-way row (>= 0), a -2-groupId relocation-group
        // encoding, or -1 (no second candidate). Lets a pinned remote member's / relocated pilot's
        // half-open-charging and dummy-Q terms follow the per-scenario routing instead of the
        // frozen dests (the guarded 3.3e-4 kV gap).
        List<int[]> openBranchAlternateReactiveDestination = new ArrayList<>();
        List<int[]> zeroImpedanceBranchAlternateReactiveDestination = new ArrayList<>();
        int[] openBranchElementOf = new int[nBranches];                 // open-branch element index per branch (-1 = none)
        java.util.Arrays.fill(openBranchElementOf, -1);
        for (LfBranch br : network.getBranches()) {
            if (br.isDisabled()) {
                continue;
            }
            LfBus b1 = br.getBus1();
            LfBus b2 = br.getBus2();
            PiModel pi = br.getPiModel();
            if (br.isZeroImpedance(LoadFlowModel.AC)) {
                // bus coupler / non-impedant branch: no flow terms — DUMMY_P/Q variables couple the buses'
                // P/Q balances, and a spanning-tree edge adds the ZERO_V/ZERO_PHI voltage/angle coupling.
                if (b1 == null || b2 == null) {
                    throw new PowsyblException("GPU AC solver: zero-impedance branch " + br.getId()
                            + " is open on one side (not supported)");
                }
                int b1n = b1.getNum();
                int b2n = b2.getNum();
                int dpRow = dummyPCol[br.getNum()];
                int distrQRow = dummyQCol[br.getNum()];
                if (dpRow < 0 || distrQRow < 0) {
                    throw new PowsyblException("GPU AC solver: zero-impedance branch " + br.getId()
                            + " has no DUMMY_P/DUMMY_Q variables (unsupported non-impedant shape, e.g. both "
                            + "ends voltage-controlled or a 3-winding star)");
                }
                zeroImpedanceBranchNums.add(br.getNum());
                zeroImpedanceBranchParameters.add(new double[] {PiModel.R2 / pi.getR1()});
                // COUPLER-TREE relocation (the dead-end stub is its k=1 case): the CHILD bus's P/Q balances move
                // onto this coupler's DUMMY_P/DUMMY_Q rows, where ∓DUMMY_P / ∓DUMMY_Q is their diagonal, and
                // ZERO_V/ZERO_PHI move onto the child's own v/phi rows. Slots 4/5 stay the dummy COLUMNS (the kernel
                // reads the dummy values through them), so the relocated ZERO_V/ZERO_PHI ROWS are derived natively
                // from the child side encoded in slot 10 — the same encoding the stub case used.
                int child = columns.spanningTreeChildBus()[br.getNum()];
                int childSide = child == b1n ? 1 : (child == b2n ? 2 : 0);
                zeroImpedanceBranchIndices.add(new int[] {pRow[b1n], qRow[b1n], pRow[b2n], qRow[b2n], dpRow, distrQRow,
                    vCol[b1n], vCol[b2n], phiCol[b1n], phiCol[b2n],
                    (br.isSpanningTreeEdge(LoadFlowModel.AC) ? 1 : 0) | (childSide << 1)
                        // bit 3: the child side came from the COUPLER-TREE walk; these need the per-scenario
                        // stock-placement fallback (native stockPlacementAlternateRow)
                        | (childSide != 0 ? 8 : 0)});
                zeroImpedanceBranchAlternateReactiveDestination.add(new int[] {alternateQRow[b1n], alternateQRow[b2n]});
                continue;
            }
            if (b1 != null && b2 != null) {
                int ph1 = phiCol[b1.getNum()];
                int v1 = vCol[b1.getNum()];
                int ph2 = phiCol[b2.getNum()];
                int v2 = vCol[b2.getNum()];
                int a1col = branchAlphaCol[br.getNum()];
                int r1col = branchRhoCol[br.getNum()];
                closedBranchElementOf[br.getNum()] = closedBranchParameters.size();
                closedBranchParameters.add(new double[] {pi.getY(), pi.getG1(), pi.getG2(), pi.getB1(), pi.getB2(),
                    pi.getKsi(), pi.getR1(), pi.getA1()});
                // Prow = bus PHI col; Qrow = qRow[bus] (its V col normally, the controlling
                // transformer's rho col when this bus is voltage-controlled). a1col/r1col carry the
                // branch's own control columns, -1 when not derived (then those scatter rows skip).
                closedBranchIndices.add(new int[] {pRow[b1.getNum()], qRow[b1.getNum()], pRow[b2.getNum()], qRow[b2.getNum()],
                    ph1, v1, ph2, v2, a1col, r1col,
                    alternateQRow[b1.getNum()], alternateQRow[b2.getNum()]});    // [10],[11]: alternate Q rows (dyn reloc)
                // CONTINUOUS phase control (CONTROLLER mode, LOCAL): the controlled side whose P-flow is
                // re-emitted at the CONTROLLER's a1 row when the control is enabled per scenario
                // (row mode 3). Encoding: -1 = none, else (controllerA1Col << 1) | controlledSide —
                // the entry sits on the CONTROLLED element (which, for REMOTE control, has no a1 of
                // its own), and the emission row is the controller's a1-paired row.
                int pcCode = -1;
                var pcOpt = br.getPhaseControl();
                if (pcOpt.isPresent()
                        && pcOpt.orElseThrow().getMode() == com.powsybl.openloadflow.network.TransformerPhaseControl.Mode.CONTROLLER
                        && pcOpt.orElseThrow().getControlledBranch() == br) {
                    int ctlA1Col = branchAlphaCol[pcOpt.orElseThrow().getControllerBranch().getNum()];
                    if (ctlA1Col >= 0) {
                        pcCode = (ctlA1Col << 1)
                                | (pcOpt.orElseThrow().getControlledSide() == com.powsybl.iidm.network.TwoSides.ONE ? 0 : 1);
                    }
                }
                closedBranchPhaseControlSide.add(pcCode);
            } else if (b2 != null) {                       // side 1 open: p2/q2 at bus2, variable v2
                openBranchElementOf[br.getNum()] = openBranchParameters.size();
                openBranchParameters.add(new double[] {pi.getY(), pi.getKsi(), pi.getG1(), pi.getB1(), pi.getG2(), pi.getB2(),
                         pi.getR1()});
                openBranchIndices.add(new int[] {-1, -1,
                    pRow[b2.getNum()], qRow[b2.getNum()],
                    -1, vCol[b2.getNum()]});
                openBranchAlternateReactiveDestination.add(new int[] {-1, alternateQRow[b2.getNum()]});
            } else if (b1 != null) {                       // side 2 open: p1/q1 at bus1, variable v1
                openBranchElementOf[br.getNum()] = openBranchParameters.size();
                openBranchParameters.add(new double[] {pi.getY(), pi.getKsi(), pi.getG1(), pi.getB1(), pi.getG2(), pi.getB2(),
                         pi.getR1()});
                openBranchIndices.add(new int[] {
                    pRow[b1.getNum()], qRow[b1.getNum()],
                    -1, -1,
                    vCol[b1.getNum()], -1});
                openBranchAlternateReactiveDestination.add(new int[] {alternateQRow[b1.getNum()], -1});
            }
        }

        // BASE-OPEN closable branches (batch SA only): appended AFTER the normal closed branches so
        // every normal element index is unchanged; the element mapper mirrors this enumeration.
        List<Integer> baseOpenCb = new ArrayList<>();
        packBaseOpenClosedBranches(baseOpenBranchIds, network, phiCol, vCol, baseOpenCb, closedBranchParameters, pRow, qRow, alternateQRow,
                                   closedBranchIndices, closedBranchPhaseControlSide);

        // BASE-OPEN closable ZERO-IMPEDANCE branches (retained-open switches, batch SA only): a
        // DISABLED zero-impedance branch keeps its DUMMY_TARGET_P/Q identity rows ACTIVE (dummies pinned 0) and
        // its dummy variables therefore HAVE rows — packing it adds the coupling slots to the
        // superset pattern; the solver enables the element and flips the two dummy rows to the
        // ZERO_V/ZERO_PHI coupling only in scenarios that CLOSE the switch (always a spanning-tree
        // merge by qualification scope, so the tree flag is baked to 1). Both coupling targets are 0
        // for a switch (rho = 1, phase shift A = 0), matching the identity rows' targets — no
        // per-scenario target override needed.
        List<Integer> baseOpenZeroImpedanceBranches = new ArrayList<>();
        List<Integer> baseOpenZeroImpedanceBranchNums = new ArrayList<>();
        packBaseOpenZeroImpedanceBranches(baseOpenZeroImpedanceBranchIdList, network, dummyPCol, dummyQCol, baseOpenZeroImpedanceBranches, zeroImpedanceBranchParameters,
                                          baseOpenZeroImpedanceBranchNums, zeroImpedanceBranchNums, phiCol, qRow, vCol,
                                          zeroImpedanceBranchIndices, alternateQRow,
                                          zeroImpedanceBranchAlternateReactiveDestination);

        List<double[]> shuntParameters = new ArrayList<>();
        List<int[]> shuntIndices = new ArrayList<>();
        List<double[]> loadModelParameters = new ArrayList<>();
        List<int[]> loadModelIndices = new ArrayList<>();
        int[] shuntElementOf = new int[nBuses];                    // shunt element index per bus (-1 = none)
        java.util.Arrays.fill(shuntElementOf, -1);
        for (LfBus bus : network.getBuses()) {
            // Fixed (non-controlling) shunt: b is a constant (bRow = -1, idx 3). shuntElementOf points DISTR_Q's
            // reactive contribution at the fixed shunt.
            bus.getShunt().ifPresent(sh -> {
                shuntElementOf[bus.getNum()] = shuntParameters.size();
                shuntParameters.add(new double[] {sh.getG(), sh.getB()});
                shuntIndices.add(new int[] {
                    pRow[bus.getNum()], qRow[bus.getNum()], vCol[bus.getNum()], -1,
                    alternateQRow[bus.getNum()]});                 // [4]: alternate Q row (dyn reloc)
            });
            // Voltage-CONTROLLING shunt: its susceptance b is a SHUNT_B state variable (bRow >= 0, idx 3), so the
            // device reads b from x and scatters dq/db onto the b column — the exact analogue of a transformer's
            // BRANCH_RHO1. bRow = -1 degrades it to a fixed shunt (shunt voltage control off in the equation system).
            bus.getControllerShunt().ifPresent(sh -> {
                shuntParameters.add(new double[] {sh.getG(), sh.getB()});
                shuntIndices.add(new int[] {
                    pRow[bus.getNum()], qRow[bus.getNum()], vCol[bus.getNum()], shuntBCol[sh.getNum()],
                    alternateQRow[bus.getNum()]});                 // [4]: alternate Q row (dyn reloc)
            });
            // SVC stand-by shunt (StandbyAutomaton b0): a permanent fixed susceptance the SVC injects
            // whether standing by or active — a plain fixed-shunt element (silently dropping it was a
            // 1.17 kV wrong answer on the monitoring network, found 2026-09-02).
            bus.getSvcShunt().ifPresent(sh -> {
                shuntParameters.add(new double[] {sh.getG(), sh.getB()});
                shuntIndices.add(new int[] {
                    pRow[bus.getNum()], qRow[bus.getNum()], vCol[bus.getNum()], -1,
                    alternateQRow[bus.getNum()]});                 // [4]: alternate Q row (dyn reloc)
            });
            packLoadModels(bus, pRow, qRow, vCol, loadModelParameters, loadModelIndices);
        }

        List<double[]> hvdcAcEmulationParameters = new ArrayList<>();
        List<int[]> hvdcAcEmulationIndices = new ArrayList<>();
        for (LfHvdc hvdc : network.getHvdcs()) {
            if (hvdc.getBus1() == null || hvdc.getBus2() == null || !hvdc.isAcEmulation()) {
                continue;
            }
            double k = hvdc.getAcEmulationControl().getDroop() * 180 / Math.PI;
            int ph1 = phiCol[hvdc.getBus1().getNum()];
            int ph2 = phiCol[hvdc.getBus2().getNum()];
            int p1 = pRow[hvdc.getBus1().getNum()];
            int p2 = pRow[hvdc.getBus2().getNum()];
            hvdcAcEmulationParameters.add(new double[] {hvdc.getAcEmulationControl().getP0(), k,
                hvdc.getConverterStation1().getLossFactor() / 100, hvdc.getConverterStation2().getLossFactor() / 100,
                hvdc.getR()});
            hvdcAcEmulationIndices.add(new int[] {p1, p2, ph1, ph2});      // [0],[1] = the two P ROWS; [2],[3] = the PHI columns
        }

        return new Packs(closedBranchParameters, closedBranchIndices, openBranchParameters, openBranchIndices,
                shuntParameters, shuntIndices, hvdcAcEmulationParameters, hvdcAcEmulationIndices, zeroImpedanceBranchParameters, zeroImpedanceBranchIndices, zeroImpedanceBranchNums,
                baseOpenCb, baseOpenZeroImpedanceBranches, baseOpenZeroImpedanceBranchNums, closedBranchElementOf, openBranchElementOf, shuntElementOf, closedBranchPhaseControlSide,
                loadModelParameters, loadModelIndices, openBranchAlternateReactiveDestination, zeroImpedanceBranchAlternateReactiveDestination);
    }

    /** BASE-OPEN closable branches (batch SA only), appended after the normal closed branches so every normal element index is unchanged. */
    private static void packBaseOpenClosedBranches(final Set<String> baseOpenBranchIds, final LfNetwork network,
                                                   final int[] phiCol, final int[] vCol,
                                                   final List<Integer> baseOpenCb, final List<double[]> closedBranchParameters,
                                                   final int[] pRow, final int[] qRow, final int[] alternateQRow,
                                                   final List<int[]> closedBranchIndices, final List<Integer> closedBranchPhaseControlSide) {
        if (!baseOpenBranchIds.isEmpty()) {
            for (LfBranch br : network.getBranches()) {
                if (!br.isDisabled() || !baseOpenBranchIds.contains(br.getId())
                        || br.isZeroImpedance(LoadFlowModel.AC)) {
                    continue;
                }
                LfBus b1 = br.getBus1();
                LfBus b2 = br.getBus2();
                if (b1 == null || b2 == null || b1.isDisabled() || b2.isDisabled()) {
                    continue;                                // dead-island reconnection — CPU scope
                }
                PiModel pi = br.getPiModel();
                int ph1 = phiCol[b1.getNum()];
                int v1 = vCol[b1.getNum()];
                int ph2 = phiCol[b2.getNum()];
                int v2 = vCol[b2.getNum()];
                baseOpenCb.add(closedBranchParameters.size());
                closedBranchParameters.add(new double[] {pi.getY(), pi.getG1(), pi.getG2(), pi.getB1(), pi.getB2(),
                    pi.getKsi(), pi.getR1(), pi.getA1()});
                closedBranchIndices.add(new int[] {pRow[b1.getNum()], qRow[b1.getNum()], pRow[b2.getNum()], qRow[b2.getNum()],
                    ph1, v1, ph2, v2, -1, -1,                // no control columns on a disabled branch
                    alternateQRow[b1.getNum()], alternateQRow[b2.getNum()]});    // [10],[11]: alternate Q rows (dyn reloc)
                closedBranchPhaseControlSide.add(-1);                            // EVERY cb element needs its entry: this list
                // was NOT appended here when closedBranchPhaseControlSide landed (d131ec7), so base-open runs read PAST
                // the array in native code — the 4x SIGSEGV-at-buildStructure+0xe43 root cause,
                // caught 2026-09-06 by the JNI input validation ("bad closedBranchPhaseControlSide at elem 4")
            }
        }
    }

    /** BASE-OPEN closable zero-impedance branches (retained-open switches, batch SA only), appended after the base couplers. */
    private static void packBaseOpenZeroImpedanceBranches(final Set<String> baseOpenZeroImpedanceBranchIdList,
                                                          final LfNetwork network, final int[] dummyPCol,
                                                          final int[] dummyQCol, final List<Integer> baseOpenZeroImpedanceBranches,
                                                          final List<double[]> zeroImpedanceBranchParameters,
                                                          final List<Integer> baseOpenZeroImpedanceBranchNums,
                                                          final List<Integer> zeroImpedanceBranchNums, final int[] phiCol,
                                                          final int[] qRow, final int[] vCol,
                                                          final List<int[]> zeroImpedanceBranchIndices, final int[] alternateQRow,
                                                          final List<int[]> zeroImpedanceBranchAlternateReactiveDestination) {
        if (!baseOpenZeroImpedanceBranchIdList.isEmpty()) {
            for (LfBranch br : network.getBranches()) {
                if (!br.isDisabled() || !baseOpenZeroImpedanceBranchIdList.contains(br.getId())
                        || !br.isZeroImpedance(LoadFlowModel.AC)) {
                    continue;
                }
                LfBus b1 = br.getBus1();
                LfBus b2 = br.getBus2();
                if (b1 == null || b2 == null || b1.isDisabled() || b2.isDisabled()) {
                    continue;                                // dead-island reconnection — CPU scope
                }
                int dpRow = dummyPCol[br.getNum()];
                int distrQRow = dummyQCol[br.getNum()];
                if (dpRow < 0 || distrQRow < 0) {
                    continue;                                // no dummy rows — CPU scope
                }
                int b1n = b1.getNum();
                int b2n = b2.getNum();
                baseOpenZeroImpedanceBranches.add(zeroImpedanceBranchParameters.size());
                baseOpenZeroImpedanceBranchNums.add(br.getNum());
                zeroImpedanceBranchNums.add(br.getNum());
                zeroImpedanceBranchParameters.add(new double[] {PiModel.R2 / br.getPiModel().getR1()});
                zeroImpedanceBranchIndices.add(new int[] {phiCol[b1n], qRow[b1n], phiCol[b2n], qRow[b2n], dpRow, distrQRow,
                    vCol[b1n], vCol[b2n], phiCol[b1n], phiCol[b2n], 1});
                zeroImpedanceBranchAlternateReactiveDestination.add(new int[] {alternateQRow[b1n], alternateQRow[b2n]});
            }
        }
    }

    /** Voltage-dependent load model (ZIP/exponential): one load model entry per NON-CONSTANT exp term,
     *  f = c_n * target * V^n added to the bus P/Q row with the derivative on the own-V column —
     *  exactly OLF's AbstractLoadModelEquationTerm (the n = 0 part lives in the target vector). */
    private static void packLoadModels(com.powsybl.openloadflow.network.LfBus bus, int[] pRow, int[] qRow, int[] vCol,
                                       List<double[]> loadModelParameters, List<int[]> loadModelIndices) {
        for (com.powsybl.openloadflow.network.LfLoad load : bus.getLoads()) {
            var lm = load.getLoadModel().orElse(null);
            if (lm == null) {
                continue;
            }
            if (bus.getControllerShunt().isPresent()) {
                // dynamic Q-relocation can move this bus's Q row per scenario; the load model Q slot is
                // static — reject the combo to the CPU rather than emit at a stale row
                throw new com.powsybl.commons.PowsyblException("GPU AC solver: load model on controller-shunt bus "
                        + bus.getId() + " (dynamic Q-relocation) not supported");
            }
            if (qRow[bus.getNum()] < 0 || pRow[bus.getNum()] < 0 || vCol[bus.getNum()] < 0) {
                // a suppressed/relocated Q row (e.g. a REMOTE-VC controller bus: qRow == -1) has no
                // static slot for the load model term — a -1 row would be an OOB read in buildStructure
                throw new com.powsybl.commons.PowsyblException("GPU AC solver: load model on bus "
                        + bus.getId() + " with a suppressed/absent P/Q/V row not supported");
            }
            // VERBATIM OLF parameters (c, n, target) — consumed by the GENERATED evalVdl
            // (codegen family "vdl", parsed from AbstractLoadModelEquationTerm.f/dfdv); the
            // n != 0 skip mirrors the Java loop's guard (constant terms live in the targets)
            for (var t : lm.getExpTermsP()) {
                if (t.n() != 0) {
                    loadModelParameters.add(new double[] {t.c(), t.n(), load.getTargetP()});
                    loadModelIndices.add(new int[] {pRow[bus.getNum()], vCol[bus.getNum()]});
                }
            }
            for (var t : lm.getExpTermsQ()) {
                if (t.n() != 0) {
                    loadModelParameters.add(new double[] {t.c(), t.n(), load.getTargetQ()});
                    loadModelIndices.add(new int[] {qRow[bus.getNum()], vCol[bus.getNum()]});
                }
            }
        }
    }

}
