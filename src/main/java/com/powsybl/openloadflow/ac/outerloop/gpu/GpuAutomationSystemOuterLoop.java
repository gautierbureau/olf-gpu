/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.iidm.network.TwoSides;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.graph.GraphConnectivity;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfOverloadManagementSystem;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.action.AbstractLfBranchAction;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;

/**
 * AutomationSystem as a device driver phase (native/cuda/outerloop/automation_system.cu): OLF's AutomationSystemOuterLoop,
 * whose overload management systems open or close a branch when the current at a monitored branch's side passes a
 * threshold. The device switches a branch with its bit in the scenario's disabled-element mask, so it owns the loop where
 * every switch keeps the network's connectivity and its equation structure: every operated branch impedant, in the
 * device's closed-branch pack (a branch to close as a base-open element) and controlling nothing, and the network still
 * connected with every branch an opening tripping operates open at once (then no sequence of switches leaves an island
 * or brings one back). Elsewhere, a switch or a branch whose opening islands buses, OLF's loop stays on the host.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAutomationSystemOuterLoop {

    /** Branches the device switched and gave back to the network (how a test tells the device drove the loop: OLF's
     *  own loop on the host switches the same branches). */
    public static final LongAdder DEVICE_SWITCHED_BRANCHES = new LongAdder();

    private GpuAutomationSystemOuterLoop() {
    }

    /** The phase's tables (see AutomationSystemNative.setBatchAutomationSystem) and the operated branch of each tripping. */
    public record Tables(int[] monitoredElement, int[] monitoredSide, int[] trippingStart, int[] operatedMaskSlot,
                         byte[] openAction, double[] threshold, List<LfBranch> operatedBranches) {

        public int systemCount() {
            return monitoredElement.length;
        }
    }

    /** The branches the network's trippings close that are open in the base: the device packs them as base-open closed
     *  branches, disabled until a tripping closes them. */
    public static Set<String> baseOpenOperatedBranchIds(LfNetwork network) {
        Set<String> ids = new LinkedHashSet<>();
        for (LfOverloadManagementSystem system : network.getOverloadManagementSystems()) {
            for (LfOverloadManagementSystem.LfBranchTripping tripping : system.getBranchTrippingList()) {
                if (tripping.branchToOperate().isDisabled()) {
                    ids.add(tripping.branchToOperate().getId());
                }
            }
        }
        return ids;
    }

    /** Whether the device can own the network's automation systems (see the class comment). */
    public static boolean representable(LfNetwork network) {
        List<LfOverloadManagementSystem> systems = network.getOverloadManagementSystems();
        if (systems.isEmpty()) {
            return false;
        }
        Set<LfBranch> opened = new LinkedHashSet<>();
        for (LfOverloadManagementSystem system : systems) {
            LfBranch monitored = system.getMonitoredBranch();
            if (monitored.isZeroImpedance(LoadFlowModel.AC) || monitored.getBus1() == null || monitored.getBus2() == null) {
                return false;
            }
            for (LfOverloadManagementSystem.LfBranchTripping tripping : system.getBranchTrippingList()) {
                LfBranch branch = tripping.branchToOperate();
                if (branch.isZeroImpedance(LoadFlowModel.AC) || branch.getBus1() == null || branch.getBus2() == null
                        || branch.getBus1().isDisabled() || branch.getBus2().isDisabled()
                        || branch.isVoltageController() || branch.isPhaseController() || branch.isPhaseControlled()
                        || branch.isTransformerReactivePowerController() || branch.isTransformerReactivePowerControlled()) {
                    return false;
                }
                if (tripping.branchOpen() && !branch.isDisabled()) {
                    opened.add(branch);
                }
            }
        }
        // every branch an opening tripping operates open at once: the main component must keep every bus
        GraphConnectivity<LfBus, LfBranch> connectivity = network.getConnectivity();
        connectivity.startTemporaryChanges();
        try {
            opened.forEach(connectivity::removeEdge);
            return connectivity.getVerticesRemovedFromMainComponent().isEmpty();
        } finally {
            connectivity.undoTemporaryChanges();
        }
    }

    /** The tables of the network's automation systems, with the element numbering of {@code mapper} (which must pack the
     *  base-open operated branches); null when an operated or monitored branch is not a closed-branch element. */
    public static Tables build(LfNetwork network, GpuAcElementMapper mapper) {
        List<LfOverloadManagementSystem> systems = network.getOverloadManagementSystems();
        int closedBranchCount = mapper.getClosedBranchCount();
        int[] monitoredElement = new int[systems.size()];
        int[] monitoredSide = new int[systems.size()];
        int[] trippingStart = new int[systems.size() + 1];
        List<Integer> operatedMaskSlot = new ArrayList<>();
        List<Byte> openAction = new ArrayList<>();
        List<Double> threshold = new ArrayList<>();
        List<LfBranch> operatedBranches = new ArrayList<>();
        for (int m = 0; m < systems.size(); m++) {
            LfOverloadManagementSystem system = systems.get(m);
            int monitored = mapper.getElementIndex(system.getMonitoredBranch());
            if (monitored < 0 || monitored >= closedBranchCount) {
                return null;
            }
            monitoredElement[m] = monitored;
            monitoredSide[m] = system.getMonitoredSide() == TwoSides.ONE ? GpuTwoSides.ONE : GpuTwoSides.TWO;
            trippingStart[m] = operatedMaskSlot.size();
            for (LfOverloadManagementSystem.LfBranchTripping tripping : system.getBranchTrippingList()) {
                int operated = mapper.getElementIndex(tripping.branchToOperate());
                if (operated < 0 || operated >= closedBranchCount) {
                    return null;
                }
                operatedMaskSlot.add(operated);
                openAction.add(tripping.branchOpen() ? (byte) 1 : (byte) 0);
                threshold.add(tripping.threshold());
                operatedBranches.add(tripping.branchToOperate());
            }
        }
        trippingStart[systems.size()] = operatedMaskSlot.size();
        byte[] openActions = new byte[openAction.size()];
        for (int t = 0; t < openActions.length; t++) {
            openActions[t] = openAction.get(t);
        }
        return new Tables(monitoredElement, monitoredSide, trippingStart, operatedMaskSlot.stream().mapToInt(Integer::intValue).toArray(),
                openActions, threshold.stream().mapToDouble(Double::doubleValue).toArray(), operatedBranches);
    }

    /** Install the run's tables, or clear the phase with null. */
    public static void uploadToDevice(long handle, Tables tables) {
        if (tables == null) {
            AutomationSystemNative.setBatchAutomationSystem(handle, null, null, null, null, null, null);
            return;
        }
        AutomationSystemNative.setBatchAutomationSystem(handle, tables.monitoredElement(), tables.monitoredSide(), tables.trippingStart(),
                tables.operatedMaskSlot(), tables.openAction(), tables.threshold());
    }

    /**
     * Give the device's switches to the network: each operated branch the solve left in the other state is opened or closed
     * as AutomationSystemOuterLoop does, on the connectivity, which disables or enables the branch (and nothing else, the
     * device having switched only where the connectivity holds).
     */
    public static void applyToNetwork(LfNetwork network, Tables tables, byte[] operatedDisabled) {
        if (tables == null || operatedDisabled == null) {
            return;
        }
        Set<LfBranch> branchesToOpen = new LinkedHashSet<>();
        Set<LfBranch> branchesToClose = new LinkedHashSet<>();
        for (int t = 0; t < tables.operatedBranches().size() && t < operatedDisabled.length; t++) {
            LfBranch branch = tables.operatedBranches().get(t);
            boolean disabled = operatedDisabled[t] != 0;
            if (disabled && !branch.isDisabled()) {
                branchesToOpen.add(branch);
            } else if (!disabled && branch.isDisabled()) {
                branchesToClose.add(branch);
            }
        }
        if (branchesToOpen.isEmpty() && branchesToClose.isEmpty()) {
            return;
        }
        GraphConnectivity<LfBus, LfBranch> connectivity = network.getConnectivity();
        connectivity.startTemporaryChanges();
        branchesToOpen.forEach(connectivity::removeEdge);
        branchesToClose.forEach(branch -> connectivity.addEdge(branch.getBus1(), branch.getBus2(), branch));
        AbstractLfBranchAction.updateBusesAndBranchStatus(connectivity);
        connectivity.undoTemporaryChanges();
        branchesToOpen.forEach(connectivity::removeEdge);
        branchesToClose.forEach(branch -> connectivity.addEdge(branch.getBus1(), branch.getBus2(), branch));
        DEVICE_SWITCHED_BRANCHES.add(branchesToOpen.size() + branchesToClose.size());
    }
}
