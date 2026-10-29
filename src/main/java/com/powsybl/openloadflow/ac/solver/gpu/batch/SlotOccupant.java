/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

/**
 * What the harvest of a slot needs from the scenario installed into it, kept from its install until the solve hands it
 * back: the scenario, the distributed-slack values its seeding derived (what the device distributed is measured from
 * them), and whether its device answer goes to the CPU whatever it is.
 *
 * @param index the scenario's index in the run's contingency list
 * @param preLoss the active power its loss pre-distribution was asked to place (pu, 0: none)
 * @param preAbsorbed what that pre-distribution placed on the participants (pu)
 * @param distributedSlackBusRowOverride per distributed-slack bus, the row its balance is solved on here (-1: its own), or null
 * @param distributedSlackOverrideBus the distributed-slack bus a generator loss reweights (-1: none)
 * @param distributedSlackOverrideBaseTargetPDelta that bus's base target P delta
 * @param distributedSlackOverrideFactorDelta that bus's factor delta
 * @param genPreMoves each distributed-slack generator's pre-distribution move, or null
 * @param distributedSlackBusAnchorShift a pair's per-bus anchor on its parent's absorption, or null
 * @param pairSlackAbsorbedSeed a slack-only pair's parent absorbed total (0: none)
 * @param preSlackSeed a contingency's pre-distribution share on the slack generators (0: none)
 * @param pairDistribution a pair whose distribution is re-anchored on its parent's
 * @param generatorAnchored a pair with per-generator anchors
 * @param rejectAfterSolve its device answer goes to the CPU
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
record SlotOccupant(int index, double preLoss, double preAbsorbed, int[] distributedSlackBusRowOverride,
                    int distributedSlackOverrideBus, double distributedSlackOverrideBaseTargetPDelta,
                    double distributedSlackOverrideFactorDelta, double[] genPreMoves, double[] distributedSlackBusAnchorShift,
                    double pairSlackAbsorbedSeed, double preSlackSeed, boolean pairDistribution, boolean generatorAnchored,
                    boolean rejectAfterSolve) {

    /** The row a scenario solves distributed-slack bus {@code i}'s balance on: its own override, or the bus's phi row. */
    int distributedSlackBusRow(int i, int[] busPhiRow) {
        return distributedSlackBusRowOverride != null && distributedSlackBusRowOverride[i] >= 0
                ? distributedSlackBusRowOverride[i] : busPhiRow[i];
    }
}
