/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchQualification;

/**
 * The seeding of one slot's entries of the scenario plan: its scheduled active power loss, its DISTR_Q weight and key
 * overrides, its DISTR_RHO coefficients, the distributed-slack participants and reactive-limit controllers it
 * strands in another component, and the subgroup labels of a voltage-control group it splits.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class PlanOverrideSeed {

    private PlanOverrideSeed() {
    }

    /** Seed slot {@code s} (contingency index {@code ci}) with its plan entries. */
    static void seed(SlotSeeding ctx, int s, int ci) {
        final ChunkLists cl = ctx.cl();
        final boolean pairPhase = ctx.pairPhase();
        final GpuBatchQualification.ScenarioPlan plan = ctx.plan();
        final double[] activePowerLossByIndex = plan.activePowerLossByIndex();
        final double[][] dqKeyOvByIndex = plan.dqKeyOvByIndex();
        final int[][] dqKeyOvMemberByIndex = plan.dqKeyOvMemberByIndex();
        final int[][] dqOvContribByIndex = plan.dqOvContribByIndex();
        final double[][] dqOvWByIndex = plan.dqOvWByIndex();
        final int[][] dqSubByIndex = plan.dqSubByIndex();
        final double[][] drOvCByIndex = plan.drOvCByIndex();
        final int[][] drOvContribByIndex = plan.drOvContribByIndex();
        final int[][] islandedDsByIndex = plan.islandedDsByIndex();
        final int[][] islandedRlByIndex = plan.islandedRlByIndex();
        if (!pairPhase && activePowerLossByIndex != null) {
            cl.preLossBySlot[s] = activePowerLossByIndex[ci];
        }
        if (dqOvContribByIndex != null && dqOvContribByIndex[ci] != null) {
            for (int k = 0; k < dqOvContribByIndex[ci].length; k++) {
                cl.dqOvIdxList.add(s);
                cl.dqOvIdxList.add(dqOvContribByIndex[ci][k]);
                cl.dqOvWList.add(dqOvWByIndex[ci][k]);
            }
        }
        if (dqKeyOvMemberByIndex != null && dqKeyOvMemberByIndex[ci] != null) {
            for (int k = 0; k < dqKeyOvMemberByIndex[ci].length; k++) {
                cl.dqKeyOvIdxList.add(s);
                cl.dqKeyOvIdxList.add(dqKeyOvMemberByIndex[ci][k]);
                cl.dqKeyOvList.add(dqKeyOvByIndex[ci][k]);
            }
        }
        if (drOvContribByIndex != null && drOvContribByIndex[ci] != null) {
            for (int k = 0; k < drOvContribByIndex[ci].length; k++) {
                cl.drOvIdxList.add(s);
                cl.drOvIdxList.add(drOvContribByIndex[ci][k]);
                cl.drOvCList.add(drOvCByIndex[ci][k]);
            }
        }
        if (islandedDsByIndex != null && islandedDsByIndex[ci] != null) {
            for (int participant : islandedDsByIndex[ci]) {
                cl.dsIslandedList.add(s);
                cl.dsIslandedList.add(participant);
            }
        }
        if (islandedRlByIndex != null && islandedRlByIndex[ci] != null) {
            for (int controller : islandedRlByIndex[ci]) {
                cl.rlIslandedList.add(s);
                cl.rlIslandedList.add(controller);
            }
        }
        if (dqSubByIndex != null && dqSubByIndex[ci] != null && cl.nDqMembersAll > 0) {
            for (int p = 0; p < dqSubByIndex[ci].length; p += 2) {
                cl.dqSubDense[s * cl.nDqMembersAll + dqSubByIndex[ci][p]] = (byte) dqSubByIndex[ci][p + 1];
            }
            cl.anyDqSub = true;
        }
    }
}
