/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;

import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.SlackBusSelector;

import java.util.List;
import java.util.Set;

/**
 * How OLF re-picks the slack when a contingency isolates it (PropagatedContingency: every bus outside the
 * largest connected component becomes ineligible, LfSynchronousNetworkImpl re-runs the RUN's slack-bus
 * selector over the rest). The selector is the run's own instance (the network wrapper OLF hands the
 * network), so the device's pick is OLF's. Null when the device cannot follow a relocation: more than one
 * slack bus, or a reference bus that is not the first slack.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record SlackRelocation(SlackBusSelector selector, int maxSlackBusCount) {

    /** The bus OLF makes the slack once only {@code eligible} may hold it, or null when the selector does not
     *  pick exactly one (OLF then falls back to its own default selector, which is not followed here). */
    public LfBus select(LfNetwork network, Set<LfBus> eligible) {
        List<LfBus> selectable = network.getSynchronousNetworks().getFirst()
                .getBuses().stream().filter(eligible::contains).toList();
        List<LfBus> picked = selector.select(selectable, maxSlackBusCount).getBuses();
        return picked.size() == 1 ? picked.get(0) : null;
    }
}
