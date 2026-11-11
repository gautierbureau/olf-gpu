/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.network.DisabledBranchStatus;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfContingency;

import java.util.Map;
import java.util.Set;

/**
 * Which branches, branch sides and buses a post-contingency result reads as out of service.
 *
 * <p>{@link #NETWORK} reads the network's own flags, as {@code LfContingency.apply} leaves them. {@link #of} reads them
 * for a contingency that was not applied: the network's flags (its base state) plus the contingency's disabled network,
 * as {@code apply} would set them: a {@code BOTH_SIDES} branch disabled, a {@code SIDE_1} or {@code SIDE_2} branch
 * opened at that side, a lost bus disabled.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
record GpuContingencyTopology(Map<LfBranch, DisabledBranchStatus> branches, Set<LfBus> buses) {

    /** The network's own flags. */
    static final GpuContingencyTopology NETWORK = new GpuContingencyTopology(Map.of(), Set.of());

    /** The topology {@code lfContingency.apply} would leave, on a network where it was not applied. */
    static GpuContingencyTopology of(LfContingency lfContingency) {
        return new GpuContingencyTopology(lfContingency.getDisabledNetwork().getBranchesStatus(),
                lfContingency.getDisabledNetwork().getBuses());
    }

    boolean branchOut(LfBranch branch) {
        return branch.isDisabled() || branches.get(branch) == DisabledBranchStatus.BOTH_SIDES;
    }

    boolean connectedSide1(LfBranch branch) {
        DisabledBranchStatus status = branches.get(branch);
        return branch.isConnectedSide1() && status != DisabledBranchStatus.SIDE_1 && status != DisabledBranchStatus.BOTH_SIDES;
    }

    boolean connectedSide2(LfBranch branch) {
        DisabledBranchStatus status = branches.get(branch);
        return branch.isConnectedSide2() && status != DisabledBranchStatus.SIDE_2 && status != DisabledBranchStatus.BOTH_SIDES;
    }

    boolean busOut(LfBus bus) {
        return bus.isDisabled() || buses.contains(bus);
    }
}
