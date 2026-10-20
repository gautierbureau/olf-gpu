/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
/**
 * Device-only: exists because the device keeps one fixed matrix for every contingency and factors without pivoting
 * (docs/gpu-vs-olf.md section 2); OLF behaviour lives in the OLF-shaped packages. Here: the stable rows (section 2.1,
 * {@link com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.StableRowIndexing}) and the equations placed on
 * another bus's row (section 2.3: {@link com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocation},
 * {@link com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocationCandidates},
 * {@link com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CouplerTreeRelocation}).
 *
 * <p>A leaf: it imports OLF core, {@code network.gpu} and {@code ac.solver.gpu.codes} only, never another GPU package.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix;
