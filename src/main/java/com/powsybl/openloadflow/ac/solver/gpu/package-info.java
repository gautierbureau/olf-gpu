/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
/**
 * The base-case GPU AC solver: {@link com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson}
 * (a full Newton-Raphson run entirely on the device behind one JNI call, plugged in through the
 * {@code AcSolver} SPI by {@code GpuNewtonRaphsonFactory}) plus its native-context lifecycle
 * ({@code GpuNativeContext}) and event-driven cache invalidation ({@code GpuSolverInvalidation}).
 * The batched N-1 solver lives in the {@code .batch} sub-package.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
package com.powsybl.openloadflow.ac.solver.gpu;
