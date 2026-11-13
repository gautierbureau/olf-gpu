/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CouplerTreeReHoming;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GenOutage;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchInjectionOutages;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.util.gpu.GpuTrace;

import java.util.List;

/**
 * The seeding of one slot's zero-impedance topology: the couplers the contingency opens (their coupling rows back to
 * DUMMY_TARGET identities, the relocated child buses' balances brought home, the child that keeps its cluster's
 * voltage target) and OLF's spanning tree replayed where the contingency moved it (the base's non-tree couplers
 * that stay out, the ones that join the tree).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class CouplerSeed {

    private CouplerSeed() {
    }

    /** Seed slot {@code s} (contingency index {@code ci}); returns its plan entries with the rows of every bus whose
     *  balances come home here rewritten to the bus's own rows. */
    static GenOutage[] seed(SlotSeeding ctx, int s, int ci, GenOutage[] slotGosIn) {
        GenOutage[] slotGos = slotGosIn;
        final int[] baseNonSpanningTreeBranches = ctx.baseNonSpanningTreeBranches();
        final int[] baseRowMode = ctx.baseRowMode();
        final double[] baseTargets = ctx.baseTargets();
        final ChunkLists cl = ctx.cl();
        final List<BatchContingency> contingencies = ctx.contingencies();
        final int[][] frozenRowsByIndex = ctx.plan().frozenRowsByIndex();
        final GpuAcData gpuData = ctx.gpuData();
        final int n = ctx.n();
        final int nDistributedSlackBuses = ctx.nDistributedSlackBuses();
        final RunScratch rs = ctx.rs();
        for (LfBranch outagedZeroImpedanceBranchOfContingency : contingencies.get(ci).outagedZeroImpedanceBranches()) {
            Integer ze = rs.zeroImpedanceElementByNumAll.get(outagedZeroImpedanceBranchOfContingency.getNum());
            // OLF_GPU_ZITREE_TRACE: does this scenario OPEN a coupler the tree relocation
            // moved rows onto? Such a scenario deactivates the coupler's ZERO_V/ZERO_PHI,
            // which the relocation put on the CHILD bus's rows, and holds its dummies with
            // DUMMY_TARGET identities those rows no longer host — the per-scenario gap.
            // childSide (zeroImpedanceBranchIndices slot 10, bits 1-2) is not NONE exactly when rows moved.
            if (GpuTrace.ZITREE_TRACE.isOn()
                    && CouplerTreeReHoming.childSide(gpuData.zeroImpedanceBranch().indices()[ze * 11 + 10]) != GpuTwoSides.NONE) {
                System.err.println("ZITREE_OPEN ctg=" + contingencies.get(ci).contingencyId()
                        + " coupler=" + outagedZeroImpedanceBranchOfContingency.getId()
                        + " childSide=" + CouplerTreeReHoming.childSide(gpuData.zeroImpedanceBranch().indices()[ze * 11 + 10]));
            }
            rs.zeroImpedanceBranchOpenScenarioElementList.add(s);
            rs.zeroImpedanceBranchOpenScenarioElementList.add(ze);
            // the decoupled branch's ZERO_V/ZERO_PHI rows become identities again
            // (dummies held at 0) — the inverse of a close
            int dpRow = gpuData.zeroImpedanceBranch().indices()[ze * 11 + 4];
            int distrQRow = gpuData.zeroImpedanceBranch().indices()[ze * 11 + 5];
            rs.rowModes.set(s, dpRow, GpuRowMode.FIXED_VARIABLE);
            rs.rowModes.set(s, distrQRow, GpuRowMode.FIXED_VARIABLE);
            rs.overrides.add(s * n + dpRow, 0.0);
            rs.overrides.add(s * n + distrQRow, 0.0);
            // The RELOCATION gave this coupler's ZERO_V the CHILD bus's own V row. With the
            // coupler open that equation is gone, and OLF's post-contingency system has
            // BUS_TARGET_V active on that bus instead (read off the CPU dump for
            // BUS-6014_BBS_2: bus_target_v6283 ACTIVE, zero_v9043 inactive). The device
            // must make the same move, or the row is left EMPTY - the P balance comes back
            // through the stock-placement fallback, but nothing restores the V identity.
            // Only for a bus that is still VOLTAGE-CONTROLLED: a PQ child's reactive
            // balance returns to this row through that same fallback instead.
            int childSide = CouplerTreeReHoming.childSide(gpuData.zeroImpedanceBranch().indices()[ze * 11 + 10]);
            if (childSide != GpuTwoSides.NONE) {
                LfBus child = childSide == GpuTwoSides.ONE ? outagedZeroImpedanceBranchOfContingency.getBus1() : outagedZeroImpedanceBranchOfContingency.getBus2();
                int childVRow = gpuData.zeroImpedanceBranch().indices()[ze * 11 + (childSide == GpuTwoSides.ONE ? 6 : 7)];
                var childVc = child == null ? java.util.Optional.<GeneratorVoltageControl>empty()
                        : child.getGeneratorVoltageControl();
                // The TARGET must travel with the equation. The child's P balance lives on
                // this coupler's DUMMY_P while it is closed, so its target (the bus's net
                // scheduled injection, and whatever the distributed slack has shifted it
                // to) sits there too. The stock-placement fallback moves the EQUATION back
                // to the child's own PHI row when the coupler opens; without moving the
                // target with it the bus's whole injection is left on a row that is now a
                // DUMMY_TARGET identity, and the slack absorbs it — measured on
                // BUS-6014_BBS_2 as 7.386 MW of distributed active power against the CPU
                // and every bus shifted 2-4e-2 kV in one direction.
                int childPhiRow = childSide == GpuTwoSides.ONE ? gpuData.zeroImpedanceBranch().indices()[ze * 11 + 8]
                        : gpuData.zeroImpedanceBranch().indices()[ze * 11 + 9];
                // ...unless the child is in the ISLAND this outage created. Then the
                // freeze already holds its rows at a clean flat state (V=1, phi=0) and
                // OLF has removed the bus, so moving an injection target onto them would
                // overwrite the freeze - these overrides are applied after it.
                int[] frozenForCtg = frozenRowsByIndex[ci];
                boolean childFrozen = false;
                for (int fr : frozenForCtg) {
                    if (fr == childPhiRow || fr == childVRow) {
                        childFrozen = true;
                        break;
                    }
                }
                if (childFrozen) {
                    continue;
                }
                // The child keeps controlling its own voltage once the coupler that carried
                // its ZERO_V is gone, so its own V row takes the BUS_TARGET_V identity.
                // This USED TO RUN ABOVE the frozen guard, so an ISLANDED child got the
                // identity too -- a second override on a row the island freeze holds flat,
                // and applyTargetOverrideBatch resolves two entries on one row by a RACE.
                // Measured on rte6515 BUS-6194_BBS_2: VL-6194_1 is islanded and its V row
                // carried both 1.000 (the freeze) and 1.123 (this identity). The guard's own
                // comment already said the freeze must not be overwritten -- it just sat
                // below the write it was guarding.
                // ...and not when the PLAN flips this row itself (an injection lost on the child while
                // its coupler opens - the section survives its own loss, both sides energized - has its
                // P target and its reactive flip moved home by rowsHome below), while the default
                // targets yield to any plan entry writing the row.
                slotGos = GpuBatchInjectionOutages.rowsHome(slotGos, dpRow, childPhiRow, distrQRow, childVRow);
                boolean planHasV = GenOutage.planSeedsRow(slotGos, childVRow);     // a target: the defaults yield
                boolean planHasPhi = GenOutage.planSeedsRow(slotGos, childPhiRow);
                boolean applied = !GenOutage.planFlipsRow(slotGos, childVRow) && childVRow >= 0 && child != null && !child.isDisabled()
                        && child.isGeneratorVoltageControlled() && childVc.isPresent()
                        && ZeroImpedanceClusters.holdsClusterVoltage(contingencies.get(ci).contingencyId(), child,
                                contingencies.get(ci).outagedZeroImpedanceBranches(), rs.zeroImpedanceElementByNumAll, gpuData.zeroImpedanceBranch().indices(),
                                frozenForCtg);
                int childReactiveLimitEntry = gpuData.reactiveLimits().entryOf(child.getNum());
                boolean childSwitchedToPq = applied
                        && gpuData.switchedPvToPqInBase(childReactiveLimitEntry);
                if (applied && childSwitchedToPq) {
                    // The child is PQ in the base (reactive limit): its balance comes home at
                    // the frozen target Q, no identity (the qualification refused such a child
                    // until the entry channels below existed).
                    rs.rowModes.set(s, childVRow, GpuRowMode.BUS_TARGET_Q);
                    rs.overrides.add(s * n + childVRow, child.getTargetQ());
                } else if (applied) {
                    rs.rowModes.set(s, childVRow, GpuRowMode.FIXED_VARIABLE);
                    rs.overrides.add(s * n + childVRow, childVc.orElseThrow().getTargetValue());
                }
                if (childPhiRow >= 0 && dpRow >= 0) {
                    if (!planHasPhi) {
                        rs.overrides.add(s * n + childPhiRow, baseTargets[dpRow]);
                    }
                    // The DS participant on that bus has to follow its equation. Its
                    // structural row is this coupler's DUMMY_P, now an identity.
                    for (int i = 0; i < nDistributedSlackBuses; i++) {
                        if (gpuData.distributedSlack().busPhiRow()[i] != dpRow) {
                            continue;
                        }
                        cl.distributedSlackBusRowOverrideList.add(s);
                        cl.distributedSlackBusRowOverrideList.add(i);
                        cl.distributedSlackBusRowOverrideList.add(childPhiRow);
                        if (cl.distributedSlackBusRowOverrideByScenario[s] == null) {
                            cl.distributedSlackBusRowOverrideByScenario[s] = new int[nDistributedSlackBuses];
                            java.util.Arrays.fill(cl.distributedSlackBusRowOverrideByScenario[s], -1);
                        }
                        cl.distributedSlackBusRowOverrideByScenario[s][i] = childPhiRow;
                    }
                }
                // A child that is NOT voltage-controlled has its reactive balance on the
                // coupler's DUMMY_Q instead, so its Q target follows the same way.
                //
                // ...unless that row is a DISTR_Q row whose target the plan RENORMALIZES for
                // this scenario (an island took a controller out of the group). Then
                // baseTargets[distrQRow] is the BASE-key target and the plan's entry is OLF's,
                // and writing both left TWO override entries on one row — which the device
                // resolves by a race (applyTargetOverrideBatch is one thread per entry
                // writing target[ovIndex[k]]). Measured on the middle-section fixture: the
                // base -0.133333333 won over the renormalized -0.2, so the row kept a target
                // built from three keys while its weights had been renormalized over two —
                // 13.33 MVAr on l2 and 0.32 kV on every bus behind it. Yielding here (rather
                // than letting the plan's entry overwrite) keeps every scenario whose plan
                // seeds no target for the row bit-identical.
                if (!applied && !planHasV && childVRow >= 0 && distrQRow >= 0) {
                    rs.overrides.add(s * n + childVRow, baseTargets[distrQRow]);
                }
                if (GpuTrace.ZITREE_TRACE.isOn()) {
                    System.err.println("ZITREE_VID coupler=" + outagedZeroImpedanceBranchOfContingency.getId()
                            + " child=" + (child == null ? "-" : child.getId())
                            + " vRow=" + childVRow
                            + " disabled=" + (child != null && child.isDisabled())
                            + " vc=" + (child != null && child.isGeneratorVoltageControlled())
                            + " baseMode=" + (childVRow >= 0 ? baseRowMode[childVRow] : -1)
                            + " target=" + (childVc.isPresent()
                                    ? childVc.orElseThrow().getTargetValue() : Double.NaN)
                            + " applied=" + applied);
                }
            }
        }
        // OLF's spanning tree, replayed by GpuBatchPrecomputer. The device extracted
        // its structure on the BASE tree; where OLF has moved on by the time it reaches
        // this contingency, the coupler that is out of the tree there takes the identity
        // rows (target 0) and the one it replaced takes the ZERO_V/ZERO_PHI coupling
        // back. The split is physically equivalent either way - this is about solving
        // the system OLF solves, from the state OLF solves it at.
        if (GpuTrace.ZITREE_TRACE.isOn() && contingencies.get(ci).stickyNonSpanningTreeBranches().size() > 40) {
            System.err.println("ZITREE_LEAVE_WIDE slot=" + s + " ctg=" + contingencies.get(ci).contingencyId()
                    + " nonTree=" + contingencies.get(ci).stickyNonSpanningTreeBranches().size());
        }
        for (LfBranch nonTree : contingencies.get(ci).stickyNonSpanningTreeBranches()) {
            // The contingency's OWN opened couplers are out of the tree too (the replay
            // reports the tree OLF solves on), but the opening path above already owns
            // them - identities, target migration, the child's BUS_TARGET_V. Handling them
            // again here overwrote that voltage target with the reactive one: network.xiidm
            // IKFR (opens IKFH) converged on the CPU and came back non-converged, 66.9 MW
            // against -2.42. They stay in stillOut below, so they are not re-coupled either.
            if (contingencies.get(ci).outagedZeroImpedanceBranches().contains(nonTree)) {
                continue;
            }
            Integer ze = rs.zeroImpedanceElementByNumAll.get(nonTree.getNum());
            if (ze == null || gpuData.zeroImpedanceBranch().indices()[ze * 11 + 10] == 0) {
                continue;                // absent, or already out of the base tree
            }
            rs.spanningTreeOverrideScenarioElementValueList.add(s);
            rs.spanningTreeOverrideScenarioElementValueList.add(ze);
            rs.spanningTreeOverrideScenarioElementValueList.add(0);
            int dpRow = gpuData.zeroImpedanceBranch().indices()[ze * 11 + 4];
            int distrQRow = gpuData.zeroImpedanceBranch().indices()[ze * 11 + 5];
            rs.rowModes.set(s, dpRow, GpuRowMode.FIXED_VARIABLE);         // identity rows, target 0
            rs.rowModes.set(s, distrQRow, GpuRowMode.FIXED_VARIABLE);
            rs.overrides.add(s * n + dpRow, 0.0);
            rs.overrides.add(s * n + distrQRow, 0.0);
            // ...and if the RELOCATION put a child bus's power balance on those dummy rows,
            // its TARGETS have to travel with it, exactly as on the coupler-OPENING path
            // above. Holding the row with an identity is what makes the kernel's
            // stock-placement fallback move the EQUATIONS back to the child's own PHI / V
            // rows; leaving the targets behind erases the bus's injection. This path is the
            // one the 2026-09-28 tree overrides never covered, because they were written
            // while the relocation was off and a non-tree coupler's dummies carried nothing.
            // Measured on network.xiidm: LJIN re-picks OLF's tree from LMQ to LMK, and
            // LMQ's DUMMY_P (row 14822) went from -1.029778 to a fixed 0 while LLJ_2's own
            // row got nothing — the bus's 102.98 MW vanished and the slack placed 87.8 MW of
            // it, on 100 LJ* contingencies (cpu=0.0 against gpu=-87.8 MW each).
            //
            // NB no BUS_TARGET_V identity here, unlike the opening path: the coupler is only
            // leaving the TREE, not opening, so the replacement edge still couples the child
            // and its voltage is not its own to set.
            int treeChildSide = CouplerTreeReHoming.childSide(gpuData.zeroImpedanceBranch().indices()[ze * 11 + 10]);
            if (treeChildSide == GpuTwoSides.NONE) {
                continue;                // not a relocated coupler: nothing moved onto it
            }
            int childPhiRow = treeChildSide == GpuTwoSides.ONE ? gpuData.zeroImpedanceBranch().indices()[ze * 11 + 8]
                    : gpuData.zeroImpedanceBranch().indices()[ze * 11 + 9];
            int childVRow = gpuData.zeroImpedanceBranch().indices()[ze * 11 + (treeChildSide == GpuTwoSides.ONE ? 6 : 7)];
            boolean childFrozen = false;
            for (int fr : frozenRowsByIndex[ci]) {
                if (fr == childPhiRow || fr == childVRow) {
                    childFrozen = true;  // islanded: the freeze holds these rows flat
                    break;
                }
            }
            if (childFrozen) {
                continue;
            }
            slotGos = GpuBatchInjectionOutages.rowsHome(slotGos, dpRow, childPhiRow, distrQRow, childVRow);
            if (childPhiRow >= 0 && dpRow >= 0) {
                if (!GenOutage.planSeedsRow(slotGos, childPhiRow)) {   // the plan's own target wins
                    rs.overrides.put(s * n + childPhiRow, baseTargets[dpRow]);
                }
                for (int i = 0; i < nDistributedSlackBuses; i++) {   // the DS participant follows its equation
                    if (gpuData.distributedSlack().busPhiRow()[i] != dpRow) {
                        continue;
                    }
                    cl.distributedSlackBusRowOverrideList.add(s);
                    cl.distributedSlackBusRowOverrideList.add(i);
                    cl.distributedSlackBusRowOverrideList.add(childPhiRow);
                    if (cl.distributedSlackBusRowOverrideByScenario[s] == null) {
                        cl.distributedSlackBusRowOverrideByScenario[s] = new int[nDistributedSlackBuses];
                        java.util.Arrays.fill(cl.distributedSlackBusRowOverrideByScenario[s], -1);
                    }
                    cl.distributedSlackBusRowOverrideByScenario[s][i] = childPhiRow;
                }
            }
            if (childVRow >= 0 && distrQRow >= 0 && !GenOutage.planSeedsRow(slotGos, childVRow)) {
                rs.overrides.put(s * n + childVRow, baseTargets[distrQRow]);
            }
        }
        if (!contingencies.get(ci).stickyNonSpanningTreeBranches().isEmpty()) {
            int[] stillOut = contingencies.get(ci).stickyNonSpanningTreeBranches().stream()
                    .mapToInt(LfBranch::getNum).sorted().toArray();
            int[] zeroImpedanceBranchNums = gpuData.zeroImpedanceBranch().branchNum();
            for (int zb : baseNonSpanningTreeBranches) {
                if (java.util.Arrays.binarySearch(stillOut, zeroImpedanceBranchNums[zb]) >= 0) {
                    continue;            // still out (the base-tree ones are not in baseNonSpanningTreeBranches)
                }
                rs.spanningTreeOverrideScenarioElementValueList.add(s);
                rs.spanningTreeOverrideScenarioElementValueList.add(zb);
                rs.spanningTreeOverrideScenarioElementValueList.add(1);
                int bdp = gpuData.zeroImpedanceBranch().indices()[zb * 11 + 4];
                int bdq = gpuData.zeroImpedanceBranch().indices()[zb * 11 + 5];
                rs.rowModes.set(s, bdp, GpuRowMode.EQUATION);             // back into the coupling
                rs.rowModes.set(s, bdq, GpuRowMode.EQUATION);
            }
        }
        return slotGos;
    }
}
