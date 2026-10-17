/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;

import java.util.List;
import java.util.Map;

/**
 * What the per-slot seeding of one chunk reads besides the slot itself: the base case, the run's lookups, the
 * run-scope buffers and this chunk's lists. Built once per chunk, next to its {@link ChunkLists}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
record SlotSeeding(int chunkSize, int n, GpuAcData gpuData, List<BatchContingency> contingencies,
                   double[] baseTargets, int[] baseRowMode, GpuBatchRlCurves.Channel rlCurves,
                   Map<Integer, int[]> curveEntriesByPRow, RunScratch rs, ChunkLists cl,
                   GpuBatchElementMapper elementMapper) {
}
