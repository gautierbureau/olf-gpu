/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.util.gpu;

/**
 * The diagnostic channels of the GPU extension's Java side: each one an environment variable, read once when this
 * class loads. A channel only prints or writes files; none changes a result. The native side has its own list
 * ({@code native/cuda/common/trace.cuh}); a variable both sides read is listed on both.
 *
 * <p>Channels whose value names a contingency ({@link #ITER_TRACE_CTG}, {@link #LOOP_TRACE_CTG},
 * {@link #EQ_DUMP_CTG}, {@link #TRACE_CTG}) scope their output to it, so one contingency can be traced inside the
 * full run that produces its answer.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public enum GpuTrace {

    /** Per-pass and per-iteration convergence of the batched solves, and what every outer loop decided. */
    BATCH_TRACE("OLF_GPU_BATCH_TRACE"),
    /** {@code <file>}: every chunk's contingencies in slot order, one line per chunk. */
    CHUNK_DUMP("OLF_GPU_CHUNK_DUMP"),
    /** The contingencies the device hands back to the CPU, and why. */
    CPU_ROUTED("OLF_GPU_CPU_ROUTED"),
    /** {@code 1}: the buses, rows and islands a branch outage touches, and the bus values injected back. */
    DBG_BUSDUMP("OLF_GPU_DBG_BUSDUMP"),
    /** The distributed active power booked when a load flow's distributed slack check falls back to OLF's own. */
    DS_ACCOUNT("OLF_GPU_DS_ACCOUNT"),
    /** {@code <file>}: the GPU equation system of the load flow as text, once (the first pinned pass). */
    EQ_DUMP("OLF_GPU_EQ_DUMP"),
    /** {@code any}: {@link #EQ_DUMP} writes the first pass, pinned or not. */
    EQ_DUMP_WHEN("OLF_GPU_EQ_DUMP_WHEN"),
    /** {@code <contingency id>}: OLF's post-contingency equation system of that contingency, as text. */
    EQ_DUMP_CTG("OLF_GPU_EQ_DUMP_CTG"),
    /** {@code <file>}: where {@link #EQ_DUMP_CTG} writes (default: the contingency id + {@code .olf.post}). */
    EQ_DUMP_CTG_FILE("OLF_GPU_EQ_DUMP_CTG_FILE"),
    /** {@code <phiRow>} or {@code all}: the AC emulation HVDC links the pack carries and their loop decisions. */
    HVDC_TRACE("OLF_GPU_HVDC_TRACE"),
    /** {@code <id>[,<id>]}: iteration count, final mismatch and distributed active power of those contingencies. */
    ITER_TRACE_CTG("OLF_GPU_ITER_TRACE_CTG"),
    /** {@code <contingency id>}: the native outer-loop trace of that contingency's slot, in every chunk. */
    LOOP_TRACE_CTG("OLF_GPU_LOOP_TRACE_CTG"),
    /** Each contingency's outer-loop passes and Newton iterations, in the shape of OLF's own {@code CPU_OL_STATS}. */
    OL_STATS("OLF_OL_STATS"),
    /** Where the run's time goes: phases of the security analysis, chunk sizing, context reuse. */
    PHASE_TIME("OLF_GPU_PHASE_TIME"),
    /** The replayed loss pre-distribution: what each contingency's generators take before the first solve. */
    PRELOSS_TRACE("OLF_GPU_PRELOSS_TRACE"),
    /** Why each contingency is batched, or which rule sends it to the CPU. */
    QUALIFY_TRACE("OLF_GPU_QUALIFY_TRACE"),
    /** The scenarios the device rejects, with the reason. */
    REJECT_TRACE("OLF_GPU_REJECT_TRACE"),
    /** The reactive-limit entries the pack builds, and the device's pin decisions per scenario. */
    RL_TRACE("OLF_GPU_RL_TRACE"),
    /** {@code <file>}: the security analysis extraction's equation system as text. */
    SA_EQ_DUMP("OLF_GPU_SA_EQ_DUMP"),
    /** {@code <n>}: the n worst rows of OLF's residual at the device's post-contingency state, with their equation. */
    SA_RESID("OLF_GPU_SA_RESID"),
    /** {@code <file>}: the device's post-contingency state per OLF variable (scoped by {@link #TRACE_CTG}). */
    SA_VAR_DUMP("OLF_GPU_SA_VAR_DUMP"),
    /** {@code <file>}: the per-generator seed of the loss pre-distribution of every contingency. */
    SEED_DUMP("OLF_GPU_SEED_DUMP"),
    /** The incremental shunt voltage control's sensitivities and decisions. */
    SHUNT_TRACE("OLF_GPU_SHUNT_TRACE"),
    /** The post-contingency realistic-voltage check: which buses make a state unrealistic. */
    UNREALISTIC_TRACE("OLF_GPU_UNREALISTIC_TRACE"),
    /** Every contingency's violations compared with OLF's stock detection over every branch. */
    VIOL_CROSSCHECK("OLF_GPU_VIOL_CROSSCHECK"),
    /** The zero-impedance coupler trees relocated, and the seeds of their couplers. */
    ZITREE_TRACE("OLF_GPU_ZITREE_TRACE"),
    /** {@code <contingency id>}: OLF's own filter for its debug dumps, which {@link #SA_VAR_DUMP} follows. */
    TRACE_CTG("OLF_TRACE_CTG");

    private final String variable;

    private final String value;

    GpuTrace(String variable) {
        this.variable = variable;
        this.value = System.getenv(variable);
    }

    /** The environment variable of this channel. */
    public String variable() {
        return variable;
    }

    /** True when the variable is set, to anything. */
    public boolean isOn() {
        return value != null;
    }

    /** The variable's value, or null when it is not set. */
    public String value() {
        return value;
    }

    /** The variable's value, or {@code otherwise} when it is not set. */
    public String valueOr(String otherwise) {
        return value != null ? value : otherwise;
    }
}
