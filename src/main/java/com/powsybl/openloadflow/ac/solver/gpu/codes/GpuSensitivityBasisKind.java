/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * How a factor group combines a basis column of a batched sensitivity analysis's contraction
 * ({@code SensitivityBasisRequest.Contraction#groupColumnKind}): a unit column by the activity, in the scenario, of the
 * equation a bus writes on that row,
 * <ul>
 *   <li>P: its active power balance (active on a balance row);</li>
 *   <li>Q: its reactive power balance (active on a balance row);</li>
 *   <li>V: its voltage target (active on an identity voltage row of a bus that is not islanded);</li>
 * </ul>
 * and ONE: a weighted or participation column, as it is. Mirrored by native/cuda/common/codes/gpu_sensitivity_basis_kind.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuSensitivityBasisKind {

    public static final int P = 0;
    public static final int Q = 1;
    public static final int V = 2;
    public static final int ONE = 3;

    private GpuSensitivityBasisKind() {
    }
}
