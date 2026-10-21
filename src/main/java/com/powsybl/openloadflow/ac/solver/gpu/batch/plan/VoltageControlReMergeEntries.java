/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CrossIdentity;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocationTables;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.util.gpu.GpuTrace;

import java.util.ArrayList;
import java.util.List;

/**
 * The per-scenario reactive-limit and DISTR_Q entries of re-merged voltage controls: the split sub-groups, the
 * remote mains turned identities, and the PV / pinned PV / cross entries of the new controllers.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class VoltageControlReMergeEntries {

    private VoltageControlReMergeEntries() {
    }

    /** tasks #71: the {dq member, label} pairs of a re-merge with subgroups, or null (and a reject tag) when a
     *  labelled bus's controller is not a packed member or the candidate table's member order cannot be trusted. */
    static int[] subgroupPairs(VoltageControlReMerge.ReMerged reMerged, GpuAcData gpuData, String[] reject) {
        if (reMerged.subgroupOf().isEmpty()) {
            return null;
        }
        if (!ReactiveRelocationTables.reactiveRelocationMemberOrderConsistent(gpuData.reactiveRelocation(), gpuData.distrQMembers().row(),
                gpuData.distrQMembers().hasRow(), gpuData.distrQMembers().relocationGroup())) {
            reject[0] = "PLAN:VC_REMERGE_SUBGROUP_TABLE";
            return null;
        }
        List<Integer> out = new ArrayList<>();
        for (var e : reMerged.subgroupOf().entrySet()) {
            var voltageControl = e.getKey().getGeneratorVoltageControl().orElse(null);
            List<LfBus> controllers = voltageControl != null ? voltageControl.getControllerElements() : List.of(e.getKey());
            for (LfBus controller : controllers) {
                int slot = GpuBatchInjectionOutages.distrQMemberSlot(gpuData, controller);
                if (slot < 0) {
                    reject[0] = "PLAN:VC_REMERGE_SUBGROUP_UNPACKED";
                    return null;
                }
                if (gpuData.distrQMembers().relocationGroup() == null || gpuData.distrQMembers().relocationGroup()[slot] < 0) {
                    reject[0] = "PLAN:VC_REMERGE_SUBGROUP_TABLE";    // no candidate table: no per-scenario carrier pick
                    return null;
                }
                out.add(slot);
                out.add(e.getValue());
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Drop the host-replayed weight overrides of a group the scenario SPLIT: the kernel's dynamic keys, now
     *  subgroup-aware, own those rows (a host weight computed over the whole base group would overwrite them). */
    static double[][] withoutSplitGroups(double[][] ov, int[] subPairs, GpuAcData gpuData) {
        if (ov == null || subPairs == null) {
            return ov;
        }
        java.util.Set<Integer> labelled = new java.util.HashSet<>();
        for (int p = 0; p < subPairs.length; p += 2) {
            labelled.add(subPairs[p]);
        }
        java.util.Set<Integer> groups = new java.util.HashSet<>();
        for (int m : labelled) {
            groups.add(gpuData.distrQMembers().groupStart()[m]);
        }
        List<Integer> keep = new ArrayList<>();
        for (int k = 0; k < ov[0].length; k++) {
            int c = (int) ov[0][k];
            int kM = gpuData.distrQ().rowMember() != null ? gpuData.distrQ().rowMember()[c] : -1;
            if (kM >= 0 && groups.contains(gpuData.distrQMembers().groupStart()[kM])) {
                continue;
            }
            keep.add(k);
        }
        List<Integer> keepKey = new ArrayList<>();
        for (int k = 0; ov.length > 3 && k < ov[2].length; k++) {
            int m = (int) ov[2][k];
            if (!groups.contains(gpuData.distrQMembers().groupStart()[m])) {
                keepKey.add(k);
            }
        }
        double[][] out = new double[4][];
        out[0] = new double[keep.size()];
        out[1] = new double[keep.size()];
        for (int i = 0; i < keep.size(); i++) {
            out[0][i] = ov[0][keep.get(i)];
            out[1][i] = ov[1][keep.get(i)];
        }
        out[2] = new double[keepKey.size()];
        out[3] = new double[keepKey.size()];
        for (int i = 0; i < keepKey.size(); i++) {
            out[2][i] = ov[2][keepKey.get(i)];
            out[3][i] = ov[3][keepKey.get(i)];
        }
        return out;
    }

    /**
     * The voltage identities a surviving REMOTE cluster's elected main still needs EXPLICITLY. The stock
     * placement (GpuBatchChunkSolver's killed-coupler block) already installs one on the CHILD of every
     * coupler the contingency opens — on rte6515 the elected main is that child in most accepted re-merges
     * ({@code blockCovers}); a main whose own row is NOT relocated (the tree's root, BUS-2640_BBS_1's
     * VL-2640_1) gets an explicit ROW_V_IDENTITY here. Emitting it in BOTH places would be a duplicate
     * override on one row (#62's race), so a main whose own row is the relocated child row of an OUTAGED
     * coupler is skipped. A main whose own row carries a LIVE coupler's ZERO_V (BUS-6029_BBS_1: the lost
     * pilot's neighbour has no control, the elected main is two couplers away) is refused: its identity
     * would displace that equation. Returns null when an identity cannot be hosted (no V row, frozen, that
     * relocation, or the bus is PINNED at a reactive limit — a pinned bus cannot also hold its voltage, see
     * the pvTargets loop above), with the reject tag in {@code reject[0]}.
     */
    static List<GenOutage> remoteMainIdentityEntries(VoltageControlReMerge.ReMerged reMerged, EquationSystem<AcVariableType, AcEquationType> es,
                                                             GpuAcData gpuData, java.util.Map<Integer, int[]> relocatedChildRows,
                                                             java.util.Set<Integer> outagedZeroImpedanceBranchNums, java.util.Set<Integer> frozenSet,
                                                             String[] reject) {
        List<GenOutage> out = new ArrayList<>();
        for (var e : reMerged.remoteMainTargets().entrySet()) {
            var vVar = es.getVariableSet().getVariable(e.getKey().getNum(), AcVariableType.BUS_V);
            if (vVar == null || vVar.getRow() < 0 || frozenSet.contains(vVar.getRow())) {
                reject[0] = "PLAN:VC_REMERGE_NO_VROW";
                return null;
            }
            int[] reloc = relocatedChildRows.get(vVar.getRow());
            boolean blockCovers = reloc != null && outagedZeroImpedanceBranchNums.contains(reloc[0]);
            int reactiveLimitEntry = gpuData.reactiveLimits().entryOf(e.getKey().getNum());
            boolean switchedPvToPqInBase = gpuData.switchedPvToPqInBase(reactiveLimitEntry);
            if (GpuTrace.QUALIFY_TRACE.isOn()) {
                System.err.printf("QUALIFY_TRACE remoteMain bus=%s vRow=%d target=%.6f blockCovers=%b switchedPvToPqInBase=%b%n",
                        e.getKey().getId(), vVar.getRow(), e.getValue(), blockCovers, switchedPvToPqInBase);
            }
            // A main PINNED at its reactive limit in the base cannot hold a voltage identity in EITHER placement.
            // This test used to sit below the blockCovers exit, so the killed-coupler block installed the identity on
            // a pinned child and the RL kernel's "limit drift while pinned" branch then wrote the reactive limit as
            // that row's target: rte6515 BUS-6194_BBS_1, VL-6194_1 (base MAX_Q) held V = 0.6251 = maxQ - loadQ,
            // FAILED where OLF converges with 60 violations; BUS-6003_BBS_1 (VL-6003_1, base MAX_Q) 0.27 kV and
            // 0.457 MW. OLF has the pinned main PQ with BUS_TARGET_V off until it unpins — a per-scenario pilot the
            // device does not host yet (tasks #70), so the scenario is the CPU's.
            if (switchedPvToPqInBase) {
                // The elected main is PQ in OLF. If EVERY controller of its cluster is pinned too, OLF's re-merged
                // group has no enabled controller: BUS_TARGET_V stays off and the cluster is plain PQ at the
                // limits — the stock placement's layout already (mode 4 rows at the pinned targets), so no
                // identity is installed; an UNPIN would re-activate BUS_TARGET_V on the main, which the device
                // cannot host for a lost pilot's table, so each member's entry forbids it (kernel reject, CPU
                // replay). rte6515 BUS-6194_BBS_1 (VL-6194_1/_2 both MAX_Q) and BUS-6003_BBS_1: OLF never
                // unpins them. A pinned main beside an ENABLED sibling (BUS_TARGET_V kept through the sibling)
                // is still refused.
                int mainLabel = reMerged.subgroupOf().getOrDefault(e.getKey(), 0);
                List<GenOutage> guards = new ArrayList<>();
                for (LfBus clusterBus : reMerged.remoteClusterBuses()) {
                    if (reMerged.subgroupOf().getOrDefault(clusterBus, 0) != mainLabel) {
                        continue;
                    }
                    var clusterVoltageControl = clusterBus.getGeneratorVoltageControl().orElse(null);
                    List<LfBus> controllers = clusterVoltageControl != null ? clusterVoltageControl.getControllerElements() : List.of(clusterBus);
                    for (LfBus controller : controllers) {
                        int controllerElementReactiveLimitEntry = gpuData.reactiveLimits().entryOf(controller.getNum());
                        if (!gpuData.switchedPvToPqInBase(controllerElementReactiveLimitEntry)) {
                            reject[0] = "PLAN:VC_REMERGE_PINNED_MAIN";
                            return null;
                        }
                        guards.add(noUnpinEntry(controllerElementReactiveLimitEntry));
                    }
                }
                out.addAll(guards);
                continue;
            }
            if (blockCovers) {
                continue;
            }
            if (reloc != null) {
                // its own row carries a LIVE coupler's ZERO_V: the identity goes on the parent's DUMMY_Q row as
                // a cross identity (rte6515 BUS-6029_BBS_1/2: the elected main two couplers from the island)
                out.add(reMergedCrossEntry(reloc[1], e.getValue(), reactiveLimitEntry, false, 0));
                continue;
            }
            out.add(reMergedPvEntry(vVar.getRow(), e.getValue(), reactiveLimitEntry));
        }
        return out;
    }

    /** The reactive-limit entries of the REMOTE controllers OLF re-merges onto a new pilot ({@code reMergedControls}):
     *  each is checked against the new pilot's voltage and target this scenario. No row of theirs changes. Null (and a
     *  reject tag) when the new pilot has no V row to read. A controller without a reactive-limit entry (limits off)
     *  needs nothing: its row carries the new pilot's balance through the dependent-pilot pool regardless. */
    static List<GenOutage> reMergedRemoteControllerEntries(VoltageControlReMerge.ReMerged reMerged, EquationSystem<AcVariableType, AcEquationType> es,
                                                                  GpuAcData gpuData, String[] reject) {
        List<GenOutage> out = new ArrayList<>();
        for (var e : reMerged.reMergedRemoteControllers().entrySet()) {
            int reactiveLimitEntry = gpuData.reactiveLimits().entryOf(e.getKey().getNum());
            if (reactiveLimitEntry < 0) {
                continue;
            }
            var pilotV = es.getVariableSet().getVariable(e.getValue().getNum(), AcVariableType.BUS_V);
            Double target = reMerged.remoteMainTargets().get(e.getValue());
            if (pilotV == null || pilotV.getRow() < 0 || target == null) {
                reject[0] = "PLAN:VC_REMERGE_NO_VROW";
                return null;
            }
            if (GpuTrace.QUALIFY_TRACE.isOn()) {
                System.err.printf("QUALIFY_TRACE reMergeRemoteController controller=%s reactiveLimitEntry=%d pilot=%s pilotVRow=%d target=%.6f%n",
                        e.getKey().getId(), reactiveLimitEntry, e.getValue().getId(), pilotV.getRow(), target);
            }
            out.add(new GenOutage(-1, 0, -1, target, false, false,
                    -1, 0, 0, 0, 0, reactiveLimitEntry, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                    new int[0], new double[0], GenOutage.ROW_REMOTE_REMERGE, -1, 0, -1, 0, -1, 0, -1, -1, 0, false,
                    pilotV.getRow(), 0, 0));
        }
        return out;
    }

    /** Row-mode-only GenOutage: the bus's V row becomes a VOLTAGE IDENTITY at {@code vTarget} (a re-merged control). */
    static GenOutage reMergedPvEntry(int vRow, double vTarget, int reactiveLimitEntry) {
        return new GenOutage(-1, 0, vRow, vTarget, true, true,
                -1, 0, 0, 0, 0, reactiveLimitEntry, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                new int[0], new double[0], GenOutage.ROW_V_IDENTITY, -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0);
    }

    /** A re-merged control whose bus the BASE pinned: its own row starts as its reactive balance at the pinned
     *  value (mode 4, {@code qTarget} = LfBus.getTargetQ(), the frozen generation minus the load) and its entry is a
     *  LOCAL PV bus at {@code vTarget}, so an unpin restores the identity as OLF re-activates BUS_TARGET_V. */
    static GenOutage reMergedPinnedPvEntry(int vRow, double qTarget, double vTarget, int reactiveLimitEntry) {
        return new GenOutage(-1, 0, vRow, qTarget, true, true,
                -1, 0, 0, 0, 0, reactiveLimitEntry, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                new int[0], new double[0], GenOutage.ROW_PINNED_V, -1, vTarget, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0);
    }

    /** A re-merged control whose own V row hosts a LIVE coupler's ZERO_V (tasks #70, VC_REMERGE_RELOC_VROW): its
     *  identity is a CROSS identity on the parent coupler's DUMMY_Q row (mode 6, the bus's V column), or — base
     *  pinned — that row starts as the balance at the pinned value and the entry unpins back to the cross form. */
    static GenOutage reMergedCrossEntry(int dummyQRow, double vTarget, int reactiveLimitEntry, boolean pinned, double qTarget) {
        return pinned ? reMergedPinnedPvEntry(dummyQRow, qTarget, vTarget, reactiveLimitEntry)
                : new GenOutage(-1, 0, dummyQRow, vTarget, true, true,
                    -1, 0, 0, 0, 0, reactiveLimitEntry, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                    new int[0], new double[0], GenOutage.ROW_CROSS_V_IDENTITY, -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0);
    }

    /** The PV entry of a re-merged control (the bus whose own {@code BUS_V} row is {@code vRow}): a cross identity
     *  when {@code crossRow >= 0} ({@link CrossIdentity#row}: its own row carries a LIVE coupler's ZERO_V), else the
     *  base-pinned form when the base pinned it ({@code switchedPvToPqInBase}), else the plain voltage identity. */
    static GenOutage pvEntry(int vRow, int crossRow, double vTarget, int reactiveLimitEntry, boolean switchedPvToPqInBase, double qTarget) {
        if (crossRow >= 0) {
            return reMergedCrossEntry(crossRow, vTarget, reactiveLimitEntry, switchedPvToPqInBase, qTarget);
        }
        if (switchedPvToPqInBase) {
            return reMergedPinnedPvEntry(vRow, qTarget, vTarget, reactiveLimitEntry);
        }
        return reMergedPvEntry(vRow, vTarget, reactiveLimitEntry);
    }

    /** A base-pinned member of a re-merged cluster with no enabled controller: no row change, unpin forbidden. */
    static GenOutage noUnpinEntry(int reactiveLimitEntry) {
        return new GenOutage(-1, 0, -1, 0, false, false,
                -1, 0, 0, 0, 0, reactiveLimitEntry, 0, 0, false, java.util.Set.of(), 0, 0, 0, -1, new int[0],
                new int[0], new double[0], GenOutage.ROW_PINNED_NO_UNPIN, -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0);
    }
}
