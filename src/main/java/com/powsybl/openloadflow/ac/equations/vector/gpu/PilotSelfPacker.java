/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.VoltageControl;

import java.util.ArrayList;
import java.util.List;

/**
 * The pilot-self groups of remote voltage control: a group whose pilot is also one of its controllers, whose
 * reactive balance is relocated to a carrier once a reactive limit pins it.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class PilotSelfPacker {

    private PilotSelfPacker() {
    }

    /**
     * PILOT-SELF voltage-control groups, keyed by RL ENTRY so the free-role rotation can be computed from the
     * pin state rather than from row modes. One group per MAIN remote generator voltage control whose merged
     * controllers CONTAIN the controlled bus, with at least two controllers.
     *
     * <p>OLF's rule ({@code AcEquationSystemCreator.updateRemoteVoltageControlEquations}) is a rotation over
     * the LIVE ENABLED controllers: {@code BUS_TARGET_V} is active while any is enabled, every PINNED
     * controller gets its own {@code BUS_TARGET_Q}, and {@code DISTR_Q} is active for every enabled
     * controller except the FIRST — so which controller is "free" changes each time one pins. In this layout
     * (rows are VARIABLES) that means: {@code BUS_TARGET_V} stays on the controlled bus's own row, because a
     * mode-0 identity constrains its OWN row's variable and moving it would pin a member's v to the pilot's
     * target; and a PINNED pilot's reactive balance has to ride the first ENABLED member's row instead.
     * {@code computeQRow} already places it there, but statically, frozen at extraction — while the device's
     * reactive-limits loop changes pin states mid-solve. That mismatch is the rank-deficient base on
     * rte6515's busbar windows.
     *
     * <p>The choice cannot be read off the row modes: the row to pick is mode 4 BECAUSE it carries the
     * relocated balance, which is circular. {@code limitType} per RL entry is the ground truth OLF itself
     * uses, hence the entry keying here. {@code memberEntry} is {@code -1} for a controller with no RL entry
     * (it can never pin), and a group is dropped when the pilot has no entry or no row.
     *
     * @param ptr group boundaries into {@code memberEntry}/{@code memberRow} ({@code nGroups + 1} entries)
     * @param memberEntry RL entry per controller, in merged order, or -1
     * @param memberRow the controller's own BUS_V row, in merged order
     * @param pilotEntry the pilot's own RL entry (it is one of its controllers, so it always has one)
     * @param pilotRow the controlled bus's BUS_V row — where BUS_TARGET_V stays
     * @param vTarget the group's voltage target, for the identity the pilot's row keeps
     */
    public record PilotSelfPack(int[] ptr, int[] memberEntry, int[] memberRow, int[] pilotEntry,
                                int[] pilotRow, double[] vTarget) {

        public int groupCount() {
            return ptr.length == 0 ? 0 : ptr.length - 1;
        }
    }

    /** Builds {@link PilotSelfPack} from an RL pack's entries (bus num -> entry index). */
    public static PilotSelfPack pilotSelfGroups(LfNetwork network,
                                                ExtractionColumns columns, int[] reactiveLimitBusNum) {
        java.util.Map<Integer, Integer> entryOfBus = new java.util.HashMap<>();
        for (int i = 0; i < reactiveLimitBusNum.length; i++) {
            entryOfBus.putIfAbsent(reactiveLimitBusNum[i], i);
        }
        int[] vCol = columns.vCol();
        List<Integer> ptr = new ArrayList<>();
        List<Integer> memberEntry = new ArrayList<>();
        List<Integer> memberRow = new ArrayList<>();
        List<Integer> pilotEntry = new ArrayList<>();
        List<Integer> pilotRow = new ArrayList<>();
        List<Double> vTarget = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            var voltageControl = bus.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> !c.isLocalControl())
                    .orElse(null);
            if (voltageControl == null || voltageControl.getControlledBus() != bus || !voltageControl.getMergedControllerElements().contains(bus)
                    || voltageControl.getMergedControllerElements().size() < 2) {
                continue;                                    // not a pilot-self group with a rotation to make
            }
            Integer pe = entryOfBus.get(bus.getNum());
            if (pe == null || vCol[bus.getNum()] < 0) {
                continue;                                    // the pilot cannot pin: nothing rotates
            }
            List<LfBus> members = voltageControl.getMergedControllerElements();
            boolean rowsResolvable = members.stream().allMatch(m -> vCol[m.getNum()] >= 0);
            if (!rowsResolvable) {
                continue;
            }
            ptr.add(memberEntry.size());
            for (LfBus m : members) {
                memberEntry.add(entryOfBus.getOrDefault(m.getNum(), -1));
                memberRow.add(vCol[m.getNum()]);
            }
            pilotEntry.add(pe);
            pilotRow.add(vCol[bus.getNum()]);
            vTarget.add(voltageControl.getTargetValue());
        }
        ptr.add(memberEntry.size());
        if (pilotEntry.isEmpty()) {
            return new PilotSelfPack(new int[0], new int[0], new int[0], new int[0], new int[0], new double[0]);
        }
        return new PilotSelfPack(GpuAcArrays.toIntArray(ptr), GpuAcArrays.toIntArray(memberEntry),
                GpuAcArrays.toIntArray(memberRow), GpuAcArrays.toIntArray(pilotEntry),
                GpuAcArrays.toIntArray(pilotRow), GpuAcArrays.toDoubleArray(vTarget));
    }

    /** A class-2 entry: a pilot that is one of its own controllers (packed as a remote entry). */
    public static boolean isPilotSelfEntry(LfBus bus) {
        GeneratorVoltageControl voltageControl = RemoteMemberQPacker.mainGeneratorVoltageControl(bus);
        return voltageControl != null && GpuAcRowIndexing.isRemoteGeneratorVoltageControlControllerElement(bus) && voltageControl.getControlledBus() == bus;
    }
}
