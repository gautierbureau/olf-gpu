/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfShunt;

import java.util.List;

/**
 * The seeding of the elements one slot's contingency takes out: its branches, fixed shunts and AC-emulation HVDCs
 * masked, and its side-opened branches kept live with their dangling-side equivalent as a per-scenario pi row.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class OutageMaskSeed {

    private OutageMaskSeed() {
    }

    /** Seed slot {@code s} (contingency index {@code ci}) with the elements its contingency takes out. */
    static void seed(SlotSeeding ctx, int s, int ci, List<LfBranch> branches, List<LfShunt> shunts,
                     List<com.powsybl.openloadflow.network.LfHvdc> hvdcs, List<BatchContingency.SideOpen> sideOpens) {
        final List<BatchContingency> contingencies = ctx.contingencies();
        final GpuAcElementMapper elementMapper = ctx.elementMapper();
        final GpuAcData gpuData = ctx.gpuData();
        final java.util.Map<com.powsybl.openloadflow.network.LfHvdc, Integer> hvdcElemIndex = ctx.hvdcElemIndex();
        final RunScratch rs = ctx.rs();
        int[] eis = new int[branches.size() + shunts.size() + hvdcs.size() + sideOpens.size()];
        int ei = 0;
        for (BatchContingency.SideOpen so : sideOpens) {
            // A side-opened branch is NOT fully out. OLF keeps it DANGLING, and the
            // dangling half's charging still injects at the connected bus — OLF's own
            // testLineOpenOneSideContingency measures -55.873 MVAr on exactly this
            // shape. Masking the element deletes that injection: measured on rte6515
            // BUS-1003_BBS as EXACTLY -2 x b_side x v^2 at every bus bordering the
            // opened side (six buses, ratios -1.994..-2.004), and on the 9-bus
            // reproducer as VL2 374.62 kV against the CPU's 379.85.
            //
            // A branch open at one end is, seen from the live bus, a PURE SHUNT: the
            // far side's series+shunt reduces to one equivalent admittance. So keep
            // the element LIVE and override its pi row to that shunt — y = 0 kills the
            // series coupling (so the frozen far bus contributes nothing and receives
            // nothing), and the connected side's g/b carry the equivalent. No new
            // kernel: this is the per-scenario pi-model override the tap channel
            // already uses, and the cb evaluation reads it through findTapOverride.
            double[] closedBranchParameterRow = GpuBatchSideOpen.danglingClosedBranchParameterRow(so.branch(), so.side());
            if (closedBranchParameterRow == null) {         // not expressible -> keep the old behaviour
                int closedBranchElementIndex = elementMapper.getElementIndex(so.branch());
                rs.disabledMask[s][closedBranchElementIndex] = true;
                eis[ei++] = closedBranchElementIndex;
            } else {
                int soElem = elementMapper.getElementIndex(so.branch());
                if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                    // Does this branch also feed a DISTR_Q row? fillDistrQBatch reads
                    // closedBranchParameters with NO closedBranchParameterOverrides parameters, so it cannot see the per-scenario
                    // pi override: a side-opened branch that contributes to a remote
                    // voltage-control group's reactive distribution would be counted
                    // there with its FULL series admittance instead of the dangling
                    // equivalent.
                    int hits = 0;
                    for (int k = 0; k < gpuData.distrQ().element().length; k++) {
                        if (gpuData.distrQ().element()[k] == soElem) {
                            hits++;
                        }
                    }
                    var sopi = so.branch().getPiModel();
                    System.err.printf("SIDEOPEN_DQ ctg=%s branch=%s side=%d elem=%d"
                                    + " distrQTerms=%d base(g1=%.9e b1=%.9e g2=%.9e b2=%.9e)"
                                    + " eq(g1=%.9e b1=%.9e g2=%.9e b2=%.9e)%n",
                            contingencies.get(ci).contingencyId(), so.branch().getId(),
                            so.side(), soElem, hits,
                            sopi.getG1(), sopi.getB1(), sopi.getG2(), sopi.getB2(),
                            closedBranchParameterRow[1], closedBranchParameterRow[3], closedBranchParameterRow[2], closedBranchParameterRow[4]);
                }
                rs.closedBranchParameterOverrideScenarioElementList.add(s);
                rs.closedBranchParameterOverrideScenarioElementList.add(soElem);
                for (double v : closedBranchParameterRow) {
                    rs.closedBranchParameterOverrideValueList.add(v);
                }
            }
        }
        for (LfBranch branch : branches) {
            int idx = elementMapper.getElementIndex(branch);          // qualification guarantees >= 0
            rs.disabledMask[s][idx] = true;
            eis[ei++] = idx;
        }
        for (LfShunt shunt : shunts) {                                // fixed-shunt disconnection
            int idx = elementMapper.getElementIndex(shunt);          // qualification guarantees >= 0
            rs.disabledMask[s][idx] = true;
            eis[ei++] = idx;
        }
        for (com.powsybl.openloadflow.network.LfHvdc hvdc : hvdcs) { // AC-emulation HVDC outage
            int idx = hvdcElemIndex.get(hvdc);                       // composition guarantees packed
            rs.disabledMask[s][idx] = true;
            eis[ei++] = idx;
        }
        rs.disabledIdx[s] = eis;
    }
}
