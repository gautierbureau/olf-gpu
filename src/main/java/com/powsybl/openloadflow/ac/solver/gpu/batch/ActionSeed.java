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
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.ShuntShift;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.TapR1Override;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.network.LfBranch;

import java.util.List;

/**
 * The seeding of what an operator strategy's actions change in one slot: the branches it recloses (a coupler back
 * into its zero-impedance coupling, any other branch back into the element mask), its fixed-ratio tap overrides and
 * its shunt section shifts.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class ActionSeed {

    private ActionSeed() {
    }

    /** Seed slot {@code s} (contingency index {@code ci}) with its recloses, tap overrides and shunt shifts. */
    static void seed(SlotSeeding ctx, int s, int ci) {
        final int[] baseRowMode = ctx.baseRowMode();
        final List<BatchContingency> contingencies = ctx.contingencies();
        final GpuAcElementMapper elementMapper = ctx.elementMapper();
        final GpuAcData gpuData = ctx.gpuData();
        final RunScratch rs = ctx.rs();
        for (LfBranch reclosed : contingencies.get(ci).reclosedBranches()) {
            if (reclosed.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                Integer ze = rs.ziElemByBranchNumChunk.get(reclosed.getNum());
                rs.ziOnSlotElemList.add(s);
                rs.ziOnSlotElemList.add(ze);
                // flip the two dummy rows from identity (pinned 0) to the ZERO_V/ZERO_PHI
                // coupling; targets stay 0 (switch: rho=1, phase shift 0)
                int dpRow = gpuData.ziIdx()[ze * 11 + 4];
                int dqRow = gpuData.ziIdx()[ze * 11 + 5];
                int[] row = rs.rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rs.rowModePerScenario[s];
                row[dpRow] = GpuRowMode.EQUATION;
                row[dqRow] = GpuRowMode.EQUATION;
                rs.rowModePerScenario[s] = row;
            } else {
                rs.disabledMask[s][elementMapper.getElementIndex(reclosed)] = false;
            }
        }
        for (TapR1Override tap : contingencies.get(ci).tapOverrides()) {
            rs.r1OvSlotElemList.add(s);
            rs.r1OvSlotElemList.add(elementMapper.getElementIndex(tap.branch()));   // cb pack index == element index
            for (double v : tap.cbRow()) {
                rs.r1OvValList.add(v);
            }
        }
        for (ShuntShift sh : contingencies.get(ci).shuntShifts()) {
            rs.shOvSlotElemList.add(s);
            rs.shOvSlotElemList.add(elementMapper.getElementIndex(sh.shunt()) - rs.shPackOffset);
            rs.shOvGbList.add(sh.dg());
            rs.shOvGbList.add(sh.db());
        }
    }
}
