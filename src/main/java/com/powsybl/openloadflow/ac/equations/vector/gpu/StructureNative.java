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

    /**
     * Build the SHARED STRUCTURE for one equation-system extraction: the diagonal-pairing row
     * permutation, the fixed SUPERSET CSR pattern (every row carries its full power-equation
     * entries + a guaranteed diagonal), the per-family device data with role-folded destination
     * arrays, the DISTR_Q / DISTR_RHO / zero-impedance packs and the per-element static trig.
     * This is everything a batched context derives from one {@link GpuAcData} — build it once
     * and pass the handle to {@code BatchSolveNative.createBatchContext}, so the base load flow and the batched
     * security analysis sharing one extraction share one device-resident structure (see {@link GpuAcStructures}
     * for the per-extraction cache). Rows use the STABLE indexing (the diagonal-paired variable
     * row, see GpuAcDataExtractor), so the structure survives activation switching; element
     * counts are implied by the index array lengths and -1 marks an absent branch side.
     *
     * @param n     number of variables / equations
     * @param cbIn  closed branches, {@code ncb*8}: (y g1 g2 b1 b2 ksi r1 a1)
     * @param cbIdx closed branches, {@code ncb*12}: (Prow1 Qrow1 Prow2 Qrow2 ph1col v1col ph2col v2col a1col r1col qAlt1 qAlt2)
     * @param obIn  open branches, {@code nob*7}: (y ksi g1 b1 g2 b2 r1)
     * @param obIdx open branches, {@code nob*6}: (Prow1 Qrow1 Prow2 Qrow2 v1col v2col)
     * @param shIn  shunts, {@code nsh*2}: (g b)
     * @param shIdx shunts, {@code nsh*3}: (Prow Qrow vcol)
     * @param hvIn  HVDC AC emulation, {@code nhv*5}: (p0 k lossFactor1 lossFactor2 r)
     * @param hvIdx HVDC AC emulation, {@code nhv*4}: (Prow1 Prow2 ph1col ph2col)
     * @return an opaque native handle; release it with {@link #destroyStructure}
     */
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
