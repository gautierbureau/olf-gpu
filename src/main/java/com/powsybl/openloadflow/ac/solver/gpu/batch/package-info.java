/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
/**
 * The batched N-1 GPU security-analysis solver: {@link com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver}
 * runs hundreds to thousands of post-contingency AC power flows in one batched Newton-Raphson. The
 * orchestrator delegates to the qualification ({@code GpuBatchQualification} / {@code GpuBatchContingencyValidator}),
 * the injection-outage and transformer-control mappers, the context cache and the chunk driver
 * ({@code GpuBatchChunkSolver}); the DTO records ({@code BatchContingency}, {@code BatchScenarioResult},
 * {@code BatchConfig}, {@code GenOutage}, …) and {@code GpuBatchStatus} constants are the shared vocabulary.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;
