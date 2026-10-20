/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * The kind of a reactive injection term in a DISTR_Q contribution (distrQKind) or a remote member's own Q balance
 * (RemoteMemberReactiveTerms.kind): the terms OLF's {@code AcEquationSystemCreator.createReactiveTerms} sums for a bus. Mirrored by
 * native/cuda/common/codes/gpu_reactive_term_kind.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuReactiveTermKind {

    /** ClosedBranchSide1/2ReactiveFlowEquationTerm: a closed branch's q1 / q2. */
    public static final int CLOSED_BRANCH_REACTIVE_FLOW = 0;
    /** ShuntCompensatorReactiveFlowEquationTerm: a fixed shunt's q. */
    public static final int SHUNT_COMPENSATOR_REACTIVE_FLOW = 1;
    /** OpenBranchSide1/2ReactiveFlowEquationTerm: a half-open branch's connected-side q. */
    public static final int OPEN_BRANCH_REACTIVE_FLOW = 2;
    /** The DUMMY_Q variable of a zero-impedance spanning-tree branch (linear). */
    public static final int DUMMY_Q = 3;

    private GpuReactiveTermKind() {
    }
}
