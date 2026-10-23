/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

/**
 * The native side of the AreaInterchangeControl driver phase (native/cuda/outerloop/area_interchange_control.cu).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class AreaInterchangeControlNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private AreaInterchangeControlNative() {
    }

    /**
     * Install (or clear, with {@code outerLoop} false) the run's AreaInterchangeControl. Per area: its interchange target
     * and its share of the slack bus mismatch (pu), and the slice of its boundaries ({@code boundaryStart}, nAreas + 1).
     * Per boundary: its {@link com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAreaBoundaryKind}, its element in that
     * kind's branch family, its slot in the disabled mask, its side (GpuTwoSides) and, for a tie line, its half's
     * {r, x, g, b, nominalV, SB} ({@code boundaryTieLineHalf}, 6 per boundary). Per distributed-slack bus and generator:
     * its area, -1 for a bus without area. No area at all makes the phase OLF's DistributedSlackOuterLoop.
     */
    public static native void setBatchAreaInterchangeControl(long handle, boolean outerLoop, double areaInterchangePMaxMismatch,
                                                             double[] interchangeTarget, double[] slackDistributionFactor,
                                                             int[] boundaryStart, int[] boundaryKind, int[] boundaryElement,
                                                             int[] boundaryMaskSlot, int[] boundarySide, double[] boundaryTieLineHalf,
                                                             int[] busArea, int[] generatorArea);

    /** Per scenario, the active power the AreaInterchangeControl checks of the last solve distributed (pu), as OLF's
     *  context data accumulates it; null before any solve. */
    public static native double[] readBatchAreaInterchangeControlDistributedActivePower(long handle);
}
