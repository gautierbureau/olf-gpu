/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.PiModel;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Gives the network the taps the device's outer loops left on a scenario's transformers, before its result is built:
 * OLF's own loops leave the post-contingency taps on the branch models, and its results — the reported flows and the
 * values the violations are confirmed with — are evaluated with them. A tap position change notifies the network's
 * listeners, which refresh the equation terms and the vectorized network exactly as OLF's loops do.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuDeviceTaps {

    /** Tap changes given to the network since the counter was last reset (read by the tests: a scenario whose tap
     *  moved, not a fixture where nothing moves). */
    static final java.util.concurrent.atomic.AtomicInteger CHANGES = new java.util.concurrent.atomic.AtomicInteger();

    private GpuDeviceTaps() {
    }

    /**
     * Puts each transformer of {@code moves} (DeviceTapMoves' {branchNum, tap index, rounding r1} triples, relative to the
     * base; null or empty when none moved) on its device tap, and every branch moved earlier for the same contingency
     * ({@code baseTaps}: its base tap, filled here) and not in {@code moves} back on its base tap — a strategy pair's moves
     * are from the base too. A rounded tap goes through OLF's rounding, which also keeps the continuous ratio it rounded
     * from ({@code getContinuousR1}); the device's tap wins a tie. Returns the branches whose tap changed.
     *
     * <p>For a strategy pair, {@code knownBase} holds the base tap of the branches its tap actions retapped (read before
     * the actions: the current tap is the action's) and {@code keep} those branches, which keep the action's tap unless
     * the pair's loops moved them. Both are empty for a contingency.
     */
    static List<LfBranch> apply(LfNetwork network, double[] moves, Map<LfBranch, Integer> baseTaps,
                                Map<LfBranch, Integer> knownBase, Set<LfBranch> keep) {
        if ((moves == null || moves.length == 0) && baseTaps.isEmpty()) {
            return List.of();
        }
        List<LfBranch> changed = new ArrayList<>();
        Set<LfBranch> listed = new HashSet<>();
        if (moves != null) {
            for (int k = 0; k + 2 < moves.length; k += 3) {
                LfBranch branch = network.getBranch((int) moves[k]);
                PiModel piModel = branch.getPiModel();
                baseTaps.putIfAbsent(branch, knownBase.getOrDefault(branch, piModel.getTapPosition()));
                listed.add(branch);
                int position = piModel.getTapPositionRange().getMinimum() + (int) moves[k + 1];
                if (piModel.getTapPosition() != position) {
                    double roundingR1 = moves[k + 2];
                    if (!Double.isNaN(roundingR1)) {          // OLF's discrete transformer voltage control: roundVoltageRatios
                        piModel.setR1(roundingR1);
                        piModel.roundR1ToClosestTap();
                    }
                    if (piModel.getTapPosition() != position) {
                        piModel.setTapPosition(position);
                    }
                    changed.add(branch);
                }
            }
        }
        for (Map.Entry<LfBranch, Integer> e : baseTaps.entrySet()) {
            PiModel piModel = e.getKey().getPiModel();
            if (!listed.contains(e.getKey()) && !keep.contains(e.getKey()) && piModel.getTapPosition() != e.getValue()) {
                piModel.setTapPosition(e.getValue());
                changed.add(e.getKey());
            }
        }
        CHANGES.addAndGet(changed.size());
        return changed;
    }
}
