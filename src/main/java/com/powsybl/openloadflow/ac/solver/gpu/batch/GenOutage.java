/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

/**
 * The per-scenario target/row-mode changes a generator loss applies on the device: remove the lost
 * generator's P from its bus's P target ({@code pRow}→{@code pTarget}); if the loss un-controls the bus
 * ({@code toPq}) flip its BUS_V row to the reactive-power equation and set {@code vRow}→{@code qTarget}.
 * Mirrors {@code LfContingency.processLostGenerators}.
 *
 * <p>{@code pRow < 0} means "no P change": the entry only carries the row-mode flip and its Q
 * target. That is how an island that KILLS a remote voltage control is expressed — every surviving
 * member bus of the control has to solve its own reactive balance again, and nothing about P moves.
 *
 * <p>When the lost generator sits on a distributed-slack participant bus, {@code dsBus} is that bus's
 * DS-pack index (else -1) and the {@code ds*Delta} fields carry the reweight the device applies: the
 * bus's participation factor, base target P, headroom and footroom each shrink by the lost gen's share,
 * and the per-scenario factor sum drops by {@code dsFactorDelta} so the other buses absorb that share.
 * This is only valid when the slack itself does not participate (otherwise the loss is rejected).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record GenOutage(int pRow, double pTarget, int vRow, double qTarget, boolean toPq, boolean qOverride,
                        int dsBus, double dsFactorDelta, double dsBaseDelta, double dsMaxDelta, double dsMinDelta,
                        int rlEntry, double minQDelta, double maxQDelta) {

    /**
     * A bus that KEEPS voltage control after losing a generator still has that generator's reactive
     * capability in the device's aggregate: {@code rlMinQ}/{@code rlMaxQ} are run-constant while OLF's
     * {@code bus.getMinQ()} sums the generators that are still there. {@code rlEntry} names the
     * reactive-limit entry to correct and the two deltas are what to add to it (negative: the lost
     * generator's own share). {@code rlEntry < 0} when there is nothing to correct — the bus goes to PQ,
     * or it is not a reactive-limit controller at all.
     */
    public boolean shrinksReactiveLimits() {
        return rlEntry >= 0 && (minQDelta != 0 || maxQDelta != 0);
    }
}
