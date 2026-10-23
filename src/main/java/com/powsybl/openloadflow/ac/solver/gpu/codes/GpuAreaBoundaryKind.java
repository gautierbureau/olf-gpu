/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * Which flow an area boundary reads, as OLF's {@code LfAreaImpl.BoundaryImpl.getP} does. Mirrored by
 * native/cuda/common/codes/gpu_area_boundary_kind.cuh (GpuCodesMirrorTest checks they agree).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAreaBoundaryKind {

    /** A branch connected at both sides: its p1 or p2. */
    public static final int CLOSED_BRANCH = 0;
    /** A branch connected at one side: that side's p, 0 on the other. */
    public static final int OPEN_BRANCH = 1;
    /** A tie line: the flow at the boundary point of the half on the boundary's side ({@code SV.otherSideP}). */
    public static final int TIE_LINE = 2;

    private GpuAreaBoundaryKind() {
    }
}
