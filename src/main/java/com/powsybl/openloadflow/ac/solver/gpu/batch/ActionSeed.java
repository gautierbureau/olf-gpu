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
import com.powsybl.openloadflow.network.LfBus;

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
        final List<BatchContingency> contingencies = ctx.contingencies();
        final GpuAcElementMapper elementMapper = ctx.elementMapper();
        final GpuAcData gpuData = ctx.gpuData();
        final RunScratch rs = ctx.rs();
        // OLF's equation count takes the balances of every bus the actions bring back from out of the network, merged into
        // the bus its switch joins or with rows past OLF's (a switch's ZERO_PHI / ZERO_V take the place of its DUMMY_TARGET
        // identities: same count)
        int broughtBackEquations = 2 * contingencies.get(ci).broughtBackBuses().size();
        ctx.cl().nActive[s] += broughtBackEquations;
        if (broughtBackEquations > 0) {
            GpuBatchedSecurityAnalysisSolver.LAST_BROUGHT_BACK_EQUATIONS.set(broughtBackEquations);
        }
        java.util.Set<LfBus> broughtBack = new java.util.HashSet<>();
        java.util.List<LfBranch> outOfTree = contingencies.get(ci).stickyNonSpanningTreeBranches();
        int switchClosesOutOfTree = 0;
        int switchCloses = 0;
        for (LfBranch reclosed : contingencies.get(ci).reclosedBranches()) {
            if (reclosed.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                switchCloses++;
            }
            if (reclosed.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC) && outOfTree.contains(reclosed)) {
                // a switch close OLF's tree for the pair leaves out (another close of the pair already joins its two
                // buses): its DUMMY_TARGET identities stay, holding its dummies at 0, and it couples nothing more
                switchClosesOutOfTree++;
                continue;
            }
            if (reclosed.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                Integer ze = rs.zeroImpedanceElementByBranchNumChunk.get(reclosed.getNum());
                rs.zeroImpedanceBranchCloseScenarioElementList.add(s);
                rs.zeroImpedanceBranchCloseScenarioElementList.add(ze);
                // flip the two dummy rows from identity (held at 0) to the ZERO_V/ZERO_PHI
                // coupling; targets stay 0 (switch: rho=1, phase shift 0)
                int dpRow = gpuData.zeroImpedanceBranch().indices()[ze * 11 + 4];
                int distrQRow = gpuData.zeroImpedanceBranch().indices()[ze * 11 + 5];
                rs.rowModes.set(s, dpRow, GpuRowMode.EQUATION);
                rs.rowModes.set(s, distrQRow, GpuRowMode.EQUATION);
            } else {
                rs.disabledMask[s][elementMapper.getElementIndex(reclosed)] = false;
            }
            for (LfBus end : java.util.Arrays.asList(reclosed.getBus1(), reclosed.getBus2())) {
                if (end != null && ctx.deadBusRows().phiRow(end) >= 0 && broughtBack.add(end)) {
                    bringBack(ctx, s, end);
                }
            }
        }
        if (switchCloses > 0) {
            GpuBatchedSecurityAnalysisSolver.LAST_SWITCH_CLOSES_OUT_OF_TREE.set(switchClosesOutOfTree);
        }
        shiftCurveLimits(ctx, s, contingencies.get(ci).generatorActions());
        for (TapR1Override tap : contingencies.get(ci).tapOverrides()) {
            rs.closedBranchParameterOverrideScenarioElementList.add(s);
            rs.closedBranchParameterOverrideScenarioElementList.add(elementMapper.getElementIndex(tap.branch()));   // cb pack index == element index
            for (double v : tap.closedBranchParameterRow()) {
                rs.closedBranchParameterOverrideValueList.add(v);
            }
        }
        for (ShuntShift shuntShift : contingencies.get(ci).shuntShifts()) {
            rs.shuntAdmittanceOverrideScenarioElementList.add(s);
            rs.shuntAdmittanceOverrideScenarioElementList.add(elementMapper.getElementIndex(shuntShift.shunt()) - rs.shuntPackOffset);
            rs.shuntAdmittanceOverrideDeltaList.add(shuntShift.deltaG());
            rs.shuntAdmittanceOverrideDeltaList.add(shuntShift.deltaB());
        }
    }

    /**
     * A bus out of the network at base that the slot's actions bring back across the branches they close (DeadBusRows): its
     * fixed shunt is enabled, its BUS_PHI and BUS_V rows past OLF's become equations (OLF's equation count takes its balances in {@link #seed}). The
     * bus's balances are at
     * OLF's bus targets on its own rows, or on its closing switch's DUMMY_P / DUMMY_Q rows when it is that switch's child,
     * its own rows then holding the switch's ZERO_PHI / ZERO_V at 0 (a switch: no ratio, no phase shift). The rows start
     * where every lane holds them, at OLF's start values.
     */
    private static void bringBack(SlotSeeding ctx, int s, LfBus bus) {
        final RunScratch rs = ctx.rs();
        final int n = ctx.n();
        int phiRow = ctx.deadBusRows().phiRow(bus);
        int vRow = ctx.deadBusRows().vRow(bus);
        // its fixed shunt comes back with it (packed on its rows, masked while it is out: deadShuntElements); the composer
        // refuses a shunt with a voltage control and an SVC's stand-by one
        bus.getShunt().ifPresent(shunt -> rs.disabledMask[s][ctx.elementMapper().getElementIndex(shunt)] = false);
        rs.rowModes.set(s, phiRow, GpuRowMode.EQUATION);
        rs.rowModes.set(s, vRow, GpuRowMode.EQUATION);
        LfBranch closing = ctx.deadBusRows().switchOf(bus);
        if (closing != null) {
            Integer ze = rs.zeroImpedanceElementByBranchNumChunk.get(closing.getNum());
            int dpRow = ctx.gpuData().zeroImpedanceBranch().indices()[ze * 11 + 4];
            int distrQRow = ctx.gpuData().zeroImpedanceBranch().indices()[ze * 11 + 5];
            rs.overrides.add(s * n + dpRow, bus.getTargetP());
            rs.overrides.add(s * n + distrQRow, bus.getTargetQ());
            rs.overrides.add(s * n + phiRow, 0);
            rs.overrides.add(s * n + vRow, 0);
        } else {
            rs.overrides.add(s * n + phiRow, bus.getTargetP());
            rs.overrides.add(s * n + vRow, bus.getTargetQ());
        }
    }

    /**
     * The reactive limits of the curve generators the slot's actions move and the curve channel does not follow (outside
     * the distributed-slack pack, or no distribution): their P is the action's for the whole solve, so their bus's limits
     * move by the curve's change between their base P and that P, the last action on a generator deciding, through the
     * scenario's reactive-limit deltas (summed with the plan's, as a generator lost on the bus).
     */
    private static void shiftCurveLimits(SlotSeeding ctx, int s, java.util.List<BatchContingency.GeneratorMove> moves) {
        java.util.Map<com.powsybl.openloadflow.network.LfGenerator, BatchContingency.GeneratorMove> last = new java.util.LinkedHashMap<>();
        moves.forEach(m -> last.put(m.generator(), m));
        java.util.List<String> followed = ctx.reactiveCapabilityCurves() != null
                ? java.util.Arrays.asList(ctx.reactiveCapabilityCurves().reactiveCapabilityCurveGeneratorId()) : java.util.List.of();
        for (BatchContingency.GeneratorMove m : last.values()) {
            if (m.minQShift() == 0 && m.maxQShift() == 0 || followed.contains(m.generator().getId())) {
                continue;
            }
            int entry = ctx.gpuData().reactiveLimits().entryOf(m.generator().getBus().getNum());
            if (entry < 0) {
                continue;                                    // its bus has no reactive-limit entry: nothing reads its limits
            }
            ctx.cl().reactiveLimitDeltaOverrideScenarioEntryList.add(s);
            ctx.cl().reactiveLimitDeltaOverrideScenarioEntryList.add(entry);
            ctx.cl().reactiveLimitDeltaOverrideValueList.add(m.minQShift());
            ctx.cl().reactiveLimitDeltaOverrideValueList.add(m.maxQShift());
            ctx.cl().reactiveLimitDeltaOverrideValueList.add(0.0);      // the bus's load Q
            ctx.cl().reactiveLimitDeltaOverrideValueList.add(0.0);      // an SVC's B range
            ctx.cl().reactiveLimitDeltaOverrideValueList.add(0.0);
        }
    }
}
