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
 * index maps ({@code cbElemOf} / {@code obElemOf} / {@code shElemOf}) the DISTR_Q outer-loop pack
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
     * lists are fixed-stride rows flattened at assembly; {@code cbElemOf} / {@code obElemOf} /
     * {@code shElemOf} map a branch/bus number to its element index (-1 when absent); the
     * {@code baseOpen*} / {@code ziNums} lists carry the batch-SA per-scenario mask metadata.
     */
    record Packs(List<double[]> cbIn, List<int[]> cbIdx, List<double[]> obIn, List<int[]> obIdx,
                 List<double[]> shIn, List<int[]> shIdx, List<double[]> hvIn, List<int[]> hvIdx,
                 List<double[]> ziIn, List<int[]> ziIdx, List<Integer> ziNums,
                 List<Integer> baseOpenCb, List<Integer> baseOpenZi, List<Integer> baseOpenZiNums,
                 int[] cbElemOf, int[] obElemOf, int[] shElemOf) {
    }

    static Packs pack(LfNetwork network, ExtractionColumns columns,
                      Set<String> baseOpenBranchIds, Set<String> baseOpenZiBranchIds) {
        int[] phiCol = columns.phiCol();
        int[] vCol = columns.vCol();
        int[] branchRhoCol = columns.branchRhoCol();
        int[] branchAlphaCol = columns.branchAlphaCol();
        int[] dummyPCol = columns.dummyPCol();
        int[] dummyQCol = columns.dummyQCol();
        int[] qRow = columns.qRow();
        int nBuses = network.getBuses().size();
        int nBranches = network.getBranches().size();

        List<double[]> ziIn = new ArrayList<>();
        List<int[]> ziIdx = new ArrayList<>();
        List<Integer> ziNums = new ArrayList<>();            // LfBranch num per zi element (per-scenario masks)

        // closed-branch element index per branch number (-1 = not a closed branch), used to wire the
        // DISTR_Q reactive contributions to their closed-branch q1/q2 outputs.
        int[] cbElemOf = new int[nBranches];
        java.util.Arrays.fill(cbElemOf, -1);

        List<double[]> cbIn = new ArrayList<>();
        List<int[]> cbIdx = new ArrayList<>();
        List<double[]> obIn = new ArrayList<>();
        List<int[]> obIdx = new ArrayList<>();
        int[] obElemOf = new int[nBranches];                 // open-branch element index per branch (-1 = none)
        java.util.Arrays.fill(obElemOf, -1);
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
                int dqRow = dummyQCol[br.getNum()];
                if (dpRow < 0 || dqRow < 0) {
                    throw new PowsyblException("GPU AC solver: zero-impedance branch " + br.getId()
                            + " has no DUMMY_P/DUMMY_Q variables (unsupported non-impedant shape, e.g. both "
                            + "ends voltage-controlled or a 3-winding star)");
                }
                ziNums.add(br.getNum());
                ziIn.add(new double[] {PiModel.R2 / pi.getR1()});
                ziIdx.add(new int[] {phiCol[b1n], qRow[b1n], phiCol[b2n], qRow[b2n], dpRow, dqRow,
                    vCol[b1n], vCol[b2n], phiCol[b1n], phiCol[b2n],
                    br.isSpanningTreeEdge(LoadFlowModel.AC) ? 1 : 0});
                continue;
            }
            if (b1 != null && b2 != null) {
                int ph1 = phiCol[b1.getNum()];
                int v1 = vCol[b1.getNum()];
                int ph2 = phiCol[b2.getNum()];
                int v2 = vCol[b2.getNum()];
                int a1col = branchAlphaCol[br.getNum()];
                int r1col = branchRhoCol[br.getNum()];
                cbElemOf[br.getNum()] = cbIn.size();
                cbIn.add(new double[] {pi.getY(), pi.getG1(), pi.getG2(), pi.getB1(), pi.getB2(),
                    pi.getKsi(), pi.getR1(), pi.getA1()});
                // Prow = bus PHI col; Qrow = qRow[bus] (its V col normally, the controlling
                // transformer's rho col when this bus is voltage-controlled). a1col/r1col carry the
                // branch's own control columns, -1 when not derived (then those scatter rows skip).
                cbIdx.add(new int[] {ph1, qRow[b1.getNum()], ph2, qRow[b2.getNum()],
                    ph1, v1, ph2, v2, a1col, r1col});
            } else if (b2 != null) {                       // side 1 open: p2/q2 at bus2, variable v2
                obElemOf[br.getNum()] = obIn.size();
                obIn.add(new double[] {pi.getY(), pi.getKsi(), pi.getG1(), pi.getB1(), pi.getG2(), pi.getB2(), pi.getR1()});
                obIdx.add(new int[] {-1, -1,
                    phiCol[b2.getNum()], qRow[b2.getNum()],
                    -1, vCol[b2.getNum()]});
            } else if (b1 != null) {                       // side 2 open: p1/q1 at bus1, variable v1
                obElemOf[br.getNum()] = obIn.size();
                obIn.add(new double[] {pi.getY(), pi.getKsi(), pi.getG1(), pi.getB1(), pi.getG2(), pi.getB2(), pi.getR1()});
                obIdx.add(new int[] {
                    phiCol[b1.getNum()], qRow[b1.getNum()],
                    -1, -1,
                    vCol[b1.getNum()], -1});
            }
        }

        // BASE-OPEN closable branches (batch SA only): appended AFTER the normal closed branches so
        // every normal element index is unchanged; the element mapper mirrors this enumeration.
        List<Integer> baseOpenCb = new ArrayList<>();
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
                baseOpenCb.add(cbIn.size());
                cbIn.add(new double[] {pi.getY(), pi.getG1(), pi.getG2(), pi.getB1(), pi.getB2(),
                    pi.getKsi(), pi.getR1(), pi.getA1()});
                cbIdx.add(new int[] {ph1, qRow[b1.getNum()], ph2, qRow[b2.getNum()],
                    ph1, v1, ph2, v2, -1, -1});              // no control columns on a disabled branch
            }
        }

        // BASE-OPEN closable ZERO-IMPEDANCE branches (retained-open switches, batch SA only): a
        // DISABLED zi branch keeps its DUMMY_TARGET_P/Q identity rows ACTIVE (dummies pinned 0) and
        // its dummy variables therefore HAVE rows — packing it adds the coupling slots to the
        // superset pattern; the solver enables the element and flips the two dummy rows to the
        // ZERO_V/ZERO_PHI coupling only in scenarios that CLOSE the switch (always a spanning-tree
        // merge by qualification scope, so the tree flag is baked to 1). Both coupling targets are 0
        // for a switch (rho = 1, phase shift A = 0), matching the identity rows' targets — no
        // per-scenario target override needed.
        List<Integer> baseOpenZi = new ArrayList<>();
        List<Integer> baseOpenZiNums = new ArrayList<>();
        if (!baseOpenZiBranchIds.isEmpty()) {
            for (LfBranch br : network.getBranches()) {
                if (!br.isDisabled() || !baseOpenZiBranchIds.contains(br.getId())
                        || !br.isZeroImpedance(LoadFlowModel.AC)) {
                    continue;
                }
                LfBus b1 = br.getBus1();
                LfBus b2 = br.getBus2();
                if (b1 == null || b2 == null || b1.isDisabled() || b2.isDisabled()) {
                    continue;                                // dead-island reconnection — CPU scope
                }
                int dpRow = dummyPCol[br.getNum()];
                int dqRow = dummyQCol[br.getNum()];
                if (dpRow < 0 || dqRow < 0) {
                    continue;                                // no dummy rows — CPU scope
                }
                int b1n = b1.getNum();
                int b2n = b2.getNum();
                baseOpenZi.add(ziIn.size());
                baseOpenZiNums.add(br.getNum());
                ziNums.add(br.getNum());
                ziIn.add(new double[] {PiModel.R2 / br.getPiModel().getR1()});
                ziIdx.add(new int[] {phiCol[b1n], qRow[b1n], phiCol[b2n], qRow[b2n], dpRow, dqRow,
                    vCol[b1n], vCol[b2n], phiCol[b1n], phiCol[b2n], 1});
            }
        }

        List<double[]> shIn = new ArrayList<>();
        List<int[]> shIdx = new ArrayList<>();
        int[] shElemOf = new int[nBuses];                    // shunt element index per bus (-1 = none)
        java.util.Arrays.fill(shElemOf, -1);
        for (LfBus bus : network.getBuses()) {
            bus.getShunt().ifPresent(sh -> {
                shElemOf[bus.getNum()] = shIn.size();
                shIn.add(new double[] {sh.getG(), sh.getB()});
                shIdx.add(new int[] {
                    phiCol[bus.getNum()], qRow[bus.getNum()], vCol[bus.getNum()]});
            });
        }

        List<double[]> hvIn = new ArrayList<>();
        List<int[]> hvIdx = new ArrayList<>();
        for (LfHvdc hvdc : network.getHvdcs()) {
            if (hvdc.getBus1() == null || hvdc.getBus2() == null || !hvdc.isAcEmulation()) {
                continue;
            }
            double k = hvdc.getAcEmulationControl().getDroop() * 180 / Math.PI;
            int ph1 = phiCol[hvdc.getBus1().getNum()];
            int ph2 = phiCol[hvdc.getBus2().getNum()];
            hvIn.add(new double[] {hvdc.getAcEmulationControl().getP0(), k,
                hvdc.getConverterStation1().getLossFactor() / 100, hvdc.getConverterStation2().getLossFactor() / 100,
                hvdc.getR()});
            hvIdx.add(new int[] {ph1, ph2, ph1, ph2});
        }

        return new Packs(cbIn, cbIdx, obIn, obIdx, shIn, shIdx, hvIn, hvIdx, ziIn, ziIdx, ziNums,
                baseOpenCb, baseOpenZi, baseOpenZiNums, cbElemOf, obElemOf, shElemOf);
    }
}
