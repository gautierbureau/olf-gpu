/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

/**
 * The read-back of a batched solve's state, targets and row modes.
 *
 * <p>JNI glue: {@code native/cuda/jni/batch_readback_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class BatchReadbackNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private BatchReadbackNative() {
    }

    /** The driver's DistributedSlack result per (scenario, participating generator): the cumulative
     *  target delta (pu) the water-fill gave each generator — null when the context has no per-generator
     *  list. Layout {@code s * nDistributedSlackGenerators + g}, g in the extraction's generator order. */
    public static native double[] readBatchDistributedSlackGeneratorMove(long handle);

    /** Per scenario, the slack's RETAINED share of the distributed mismatch (pu) — what OLF's own loop
     *  gives the slack bus's participating generators, which have no device row of their own. */
    public static native double[] readBatchSlackRetainedShare(long handle);

    /** One scenario's state at its last CONVERGED inner solve of the last solve, or its starting state when none
     *  converged: what OLF's network holds after a run that failed, where it starts that contingency's strategies. */
    public static native double[] readBatchLastConvergedState(long handle, int slot);

    /** Per scenario of the last batch solve: {solve passes, first pass's Newton iterations, total, outer-loop checks},
     *  flattened. */
    public static native int[] readBatchOuterLoopStats(long handle);

    /** The per-scenario ROW MODES the device ended with — the companion to {@link #readBatchTargets}.
     *  A row's equation is (mode, target) TOGETHER, and the outer-loop kernels rewrite both; the row modes the
     *  host installed are only what setup asked for. Reading a target without its final mode
     *  can say "the target is right" about a row that is no longer the equation it was set up to be. */
    public static native int[] readBatchRowModes(long handle);

    /** Per-scenario slack-only-DS absorbed totals after a solve, or null when the context is not
     *  slack-only — the pair phase seeds its scenarios with the parent contingency's value. */
    public static native double[] readBatchSlackAbsorbed(long handle);

    /** How many {@code solveBatchContext} calls the context has run. The distributed-slack read-backs
     *  ({@link #readBatchSlackRetainedShare}, {@link #readBatchDistributedSlackGeneratorMove}, {@link #readBatchSlackAbsorbed}) are reset at
     *  every solve: they describe the whole distribution only when one solve covered it. */
    public static native long readBatchSolveCount(long handle);

    /** The batched Newton iterations of the last {@code solveBatchContext} call: every scenario's solves together. */
    public static native int readBatchNewtonLoopIterations(long handle);

    /** Read back the batch context's per-scenario TARGET buffer (n * nScenarios) after a solve — the device
     *  outer loops mutate targets in place; the strategy/pair phase re-anchors its slack distribution at the
     *  PARENT contingency's final distributed targets (OLF's setGeneratorsInitialTargetPToTargetP). */
    public static native double[] readBatchTargets(long handle);
}
