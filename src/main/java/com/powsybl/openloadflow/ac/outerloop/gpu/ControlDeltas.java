/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import java.util.ArrayList;
import java.util.List;

/**
 * What switching a voltage control ON changes in the equation system against OFF: the rows whose row mode or target
 * differs between the two activations, with each one's mode and target in either state. The transformer and shunt
 * voltage-control phases diff their two states with it, against each other rather than against a caller's base: a base
 * extracted with the control already ON yields the same rows as a control-off one.
 */
record ControlDeltas(int[] rows, int[] modesOn, double[] targetsOn, int[] modesOff, double[] targetsOff) {

    /** The rows among the first {@code rowCount} whose mode or target differs between the ON and OFF activations. */
    static ControlDeltas between(int rowCount, int[] onModes, double[] onTargets, int[] offModes, double[] offTargets) {
        List<Integer> changed = new ArrayList<>();
        for (int r = 0; r < rowCount; r++) {
            if (onModes[r] != offModes[r] || onTargets[r] != offTargets[r]) {
                changed.add(r);
            }
        }
        int[] rows = changed.stream().mapToInt(Integer::intValue).toArray();
        int[] modesOn = new int[rows.length];
        double[] targetsOn = new double[rows.length];
        int[] modesOff = new int[rows.length];
        double[] targetsOff = new double[rows.length];
        for (int k = 0; k < rows.length; k++) {
            modesOn[k] = onModes[rows[k]];
            targetsOn[k] = onTargets[rows[k]];
            modesOff[k] = offModes[rows[k]];
            targetsOff[k] = offTargets[rows[k]];
        }
        return new ControlDeltas(rows, modesOn, targetsOn, modesOff, targetsOff);
    }
}
