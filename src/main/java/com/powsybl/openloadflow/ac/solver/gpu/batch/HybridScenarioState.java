/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.network.BranchState;
import com.powsybl.openloadflow.network.BusState;
import com.powsybl.openloadflow.network.HvdcState;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfHvdc;

import java.util.List;

/**
 * Compact per-scenario outer-loop state snapshot for the hybrid (Java-outer-loop) batch security analysis — the
 * lightweight replacement for a full {@link com.powsybl.openloadflow.network.NetworkState} snapshot.
 *
 * <p>The hybrid drives one shared {@link com.powsybl.openloadflow.network.LfNetwork} through OLF's Java outer loops
 * for every scenario in sequence, so each scenario's state must be saved/restored between scenarios. A full
 * {@code NetworkState} allocates {@link BusState} for <em>every</em> bus plus {@code BranchState} for
 * <em>every</em> branch (~25000 objects per save). This captures only what the active outer loops mutate:
 *
 * <ul>
 *   <li>OLF {@link BusState} over the GENERATOR voltage-control CONTROLLER buses only — the sole buses reactive
 *       limits can switch PV&harr;PQ (VC-enabled, Q-freeze, {@code qLimitType}, control types);</li>
 *   <li>{@link HvdcState} for the HVDC AC-emulation loop;</li>
 *   <li>OLF {@link BranchState} over the TRANSFORMER voltage-control CONTROLLER branches — the tap position,
 *       {@code r1}/{@code a1}, and VC-enabled flag the transformer voltage-control loop mutates (empty when the
 *       run has no transformer voltage control).</li>
 * </ul>
 *
 * <p>Generator active-power {@code targetP} (distributed slack) is managed separately (a flat array — {@code
 * BusState} does not capture it); bus V/&phi; come from the device solve each round; and outage branch-disable is
 * managed incrementally. Measured ~3.9&times; cheaper than the full {@code NetworkState}. See
 * {@code docs/hybrid-outer-loop-production-design.md} §5.3.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class HybridScenarioState {

    private final List<BusState> busStates;
    private final List<HvdcState> hvdcStates;
    private final List<BranchState> branchStates;

    /**
     * Capture the compact state now.
     *
     * @param controllerBuses    the GENERATOR voltage-control controller buses
     *                           ({@code network.getControllerElements(VoltageControl.Type.GENERATOR)})
     * @param hvdcs              the network HVDCs ({@code network.getHvdcs()})
     * @param transformerBranches the TRANSFORMER voltage-control controller branches
     *                           ({@code network.getControllerElements(VoltageControl.Type.TRANSFORMER)}) — empty
     *                           when the run has no transformer voltage control
     */
    public HybridScenarioState(List<LfBus> controllerBuses, List<LfHvdc> hvdcs, List<LfBranch> transformerBranches) {
        this.busStates = controllerBuses.stream().map(BusState::save).toList();
        this.hvdcStates = hvdcs.stream().map(HvdcState::save).toList();
        this.branchStates = transformerBranches.stream().map(BranchState::save).toList();
    }

    /** Restore ONE captured controller bus, by its index in the controllerBuses list this was built from.
     *  Dirty-tracking restores a handful of buses instead of all of them. */
    public void restoreBus(int index) {
        busStates.get(index).restore();
    }

    /** Restore only the HVDC + transformer-branch part. Those lists are small (single digits here) and are
     *  always restored whole; only the bus part is dirty-tracked. */
    public void restoreHvdcsAndBranches() {
        hvdcStates.forEach(HvdcState::restore);
        branchStates.forEach(BranchState::restore);
    }

    /** Restore the captured controller-bus + HVDC + transformer-branch state onto their live network elements. */
    public void restore() {
        busStates.forEach(BusState::restore);
        hvdcStates.forEach(HvdcState::restore);
        branchStates.forEach(BranchState::restore);
    }
}
