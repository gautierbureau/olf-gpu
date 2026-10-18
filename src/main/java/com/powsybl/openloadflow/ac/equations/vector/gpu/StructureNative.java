/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

/**
 * The shared device structure built from one extraction.
 *
 * <p>JNI glue: {@code native/cuda/jni/jni/structure_jni.cu}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class StructureNative {

    static {
        GpuAcNewtonSolver.isAvailable();   // loads libolfgpu once
    }

    private StructureNative() {
    }

    public static native long createStructure(int n,
                                              double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
                                              double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx,
                                              int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                              double[] ziIn, int[] ziIdx,
                                              int[] drRow, int[] drCol, double[] drCoef,
                                              int[] dbRow, int[] dbCol, double[] dbCoef, int[] cbPcSide,
                                              double[] vdlIn, int[] vdlIdx, int[] qrPtr, int[] qrRows,
                                              int[] obQAlt, int[] ziQAlt,
                                              int[] qrDepBase, int[] qrBaseTargetRow, int[] depGroupOfRow);

    /**
     * Release a structure handle from {@link #createStructure} (idempotent for handle 0). Contexts
     * built from it keep the device-resident structure alive natively (shared ownership) — this only
     * drops the Java-side reference, so it is always safe to call once the handle is no longer used
     * to create new contexts.
     */
    public static native void destroyStructure(long handle);

    /**
     * Recommend how many scenarios fit in one GPU batch chunk for this structure, from the
     * free device memory ({@code cudaMemGetInfo}) and the per-scenario footprint. The
     * dominant term, cuDSS's LU numeric factors, is sized from a cheap single-matrix probe
     * ANALYSIS on the structure's device pattern that reads cuDSS's reported
     * {@code CUDSS_DATA_LU_NNZ} (the exact fill-in), falling back to a heuristic multiplier
     * over the Jacobian nnz only if that query is unavailable — and CACHED on the structure,
     * so only the first query per extraction pays the probe. The value/state arrays are added
     * and a safety margin applied; the result is clamped to {@code [1, maxCap]}.
     *
     * @param structureHandle a structure from {@link #createStructure}
     * @param maxCap          absolute upper bound on the returned chunk size
     * @return a recommended chunk size in {@code [1, maxCap]}
     */
    public static native int recommendBatchSize(long structureHandle, int maxCap);

    /** VALUES refresh addressed by the SHARED STRUCTURE handle: same semantics as
     *  {@link #refreshStructureValues} (tap r1/a1 or shunt-b moves with an unchanged pattern), but
     *  usable when the caller holds a BATCH context — one extraction has ONE value set, every
     *  context sharing the structure sees the update. */
    public static native void refreshStructureValues(long structureHandle, double[] cbIn, double[] obIn,
                                                     double[] shIn, double[] hvIn, double[] ziIn);
}
