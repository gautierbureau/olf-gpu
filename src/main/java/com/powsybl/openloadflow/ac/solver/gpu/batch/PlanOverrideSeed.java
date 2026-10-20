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
        final double[][] distrQKeyOverrideKeyByIndex = plan.distrQKeyOverrideKeyByIndex();
        final int[][] distrQKeyOverrideMemberByIndex = plan.distrQKeyOverrideMemberByIndex();
        final int[][] distrQWeightOverrideTermByIndex = plan.distrQWeightOverrideTermByIndex();
        final double[][] distrQWeightOverrideWeightByIndex = plan.distrQWeightOverrideWeightByIndex();
        final int[][] distrQSubgroupByIndex = plan.distrQSubgroupByIndex();
        final double[][] distrRhoOverrideCoefficientByIndex = plan.distrRhoOverrideCoefficientByIndex();
        final int[][] distrRhoOverrideTermByIndex = plan.distrRhoOverrideTermByIndex();
        final int[][] islandedDsByIndex = plan.islandedDsByIndex();
        final int[][] islandedRlByIndex = plan.islandedRlByIndex();
        if (!pairPhase && activePowerLossByIndex != null) {
            cl.preLossBySlot[s] = activePowerLossByIndex[ci];
        }
        if (distrQWeightOverrideTermByIndex != null && distrQWeightOverrideTermByIndex[ci] != null) {
            for (int k = 0; k < distrQWeightOverrideTermByIndex[ci].length; k++) {
                cl.distrQWeightOverrideScenarioTermList.add(s);
                cl.distrQWeightOverrideScenarioTermList.add(distrQWeightOverrideTermByIndex[ci][k]);
                cl.distrQWeightOverrideWeightList.add(distrQWeightOverrideWeightByIndex[ci][k]);
            }
        }
        if (distrQKeyOverrideMemberByIndex != null && distrQKeyOverrideMemberByIndex[ci] != null) {
            for (int k = 0; k < distrQKeyOverrideMemberByIndex[ci].length; k++) {
                cl.distrQKeyOverrideScenarioMemberList.add(s);
                cl.distrQKeyOverrideScenarioMemberList.add(distrQKeyOverrideMemberByIndex[ci][k]);
                cl.distrQKeyOverrideKeyList.add(distrQKeyOverrideKeyByIndex[ci][k]);
            }
        }
        if (distrRhoOverrideTermByIndex != null && distrRhoOverrideTermByIndex[ci] != null) {
            for (int k = 0; k < distrRhoOverrideTermByIndex[ci].length; k++) {
                cl.distrRhoOverrideScenarioTermList.add(s);
                cl.distrRhoOverrideScenarioTermList.add(distrRhoOverrideTermByIndex[ci][k]);
                cl.distrRhoOverrideCoefficientList.add(distrRhoOverrideCoefficientByIndex[ci][k]);
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
        if (distrQSubgroupByIndex != null && distrQSubgroupByIndex[ci] != null && cl.nDistrQMembersAll > 0) {
            for (int p = 0; p < distrQSubgroupByIndex[ci].length; p += 2) {
                cl.distrQSubgroupDense[s * cl.nDistrQMembersAll + distrQSubgroupByIndex[ci][p]] = (byte) distrQSubgroupByIndex[ci][p + 1];
            }
            cl.anyDistrQSubgroup = true;
        }
    }
}
