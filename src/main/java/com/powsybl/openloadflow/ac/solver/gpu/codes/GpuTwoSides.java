/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

import com.powsybl.iidm.network.TwoSides;

/**
 * A branch side as the device reads it: powsybl's {@link TwoSides}, by ordinal (ONE 0, TWO 1; not {@code getNum()}).
 * Every pack that carries a side uses it: DISTR_Q and remote-member reactive terms, controlled flows, limits and
 * violations, side-open branches, contingencies, HVDC curve stations and the zero-impedance coupler packs. Mirrored by
 * native/cuda/jni/common/codes/gpu_two_sides.cuh (GpuCodesMirrorTest checks it equals OLF's ordinals).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuTwoSides {

    /** No side: the child side of a coupler with no relocated child, a shunt's reactive term, a plain generator's
     *  curve entry. */
    public static final int NONE = -1;
    /** Side 1. */
    public static final int ONE = TwoSides.ONE.ordinal();
    /** Side 2. */
    public static final int TWO = TwoSides.TWO.ordinal();

    private GpuTwoSides() {
    }
}
