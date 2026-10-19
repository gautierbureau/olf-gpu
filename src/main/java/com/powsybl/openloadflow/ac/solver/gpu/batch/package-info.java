/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
/**
 * The batched N-1 GPU security-analysis solver: {@link com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver}
 * runs hundreds to thousands of post-contingency AC power flows in one batched Newton-Raphson. This package holds
 * the solve side: the chunk driver ({@code GpuBatchChunkSolver}) with its per-scenario seeds, the context cache, the
 * run configuration ({@code BatchConfig}), the results and the {@code GpuBatchStatus} constants. What each scenario
 * changes and whether it can be batched is decided in {@link com.powsybl.openloadflow.ac.solver.gpu.batch.plan},
 * which this package reads and which never reads this one.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;
