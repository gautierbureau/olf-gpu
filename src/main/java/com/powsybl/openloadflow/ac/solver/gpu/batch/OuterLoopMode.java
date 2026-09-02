/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

/**
 * How the batched N-1 security analysis runs its OUTER loops (distributed slack, reactive limits, …).
 *
 * <ul>
 *   <li>{@link #DEVICE} — the outer loops are ported to CUDA kernels and folded into the batched device
 *       solve ({@code solveBatchContext} + the DS/RL packs). Fastest; a second outer-loop implementation
 *       to keep in sync with OLF.</li>
 *   <li>{@link #JAVA} — the batched device solve runs the inner Newton ONLY (empty packs), and OLF's own
 *       validated Java outer loops make every switch/redistribution DECISION per scenario, via
 *       {@link GpuHybridBatchOuterLoopEngine}. Matches the CPU by construction (no second implementation);
 *       a throughput cost that grows with the round count.</li>
 * </ul>
 *
 * <p>Selected per run through the {@code olf.gpu.sa.outerLoopMode} system property (default {@link #DEVICE}).
 * See {@code docs/hybrid-outer-loop-production-design.md}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public enum OuterLoopMode {
    DEVICE,
    JAVA
}
