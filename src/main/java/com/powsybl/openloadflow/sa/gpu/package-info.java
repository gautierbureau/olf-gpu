/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
/**
 * The GPU-accelerated security-analysis provider: {@link com.powsybl.openloadflow.sa.gpu.GpuBatchedAcSecurityAnalysis}
 * intercepts OLF's AC security analysis to precompute the contingency batch on the device
 * ({@code GpuBatchPrecomputer} -> {@code PrecomputedBatch}) once the base load flow converges, then
 * injects each precomputed state instead of re-solving ({@code GpuStateInjector}). The run-scope
 * decision ({@code GpuSaRunScope}), contingency/strategy mapping ({@code GpuContingencyMapper},
 * {@code GpuStrategyComposer}) and the {@code @AutoService} provider complete the integration.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
package com.powsybl.openloadflow.sa.gpu;
