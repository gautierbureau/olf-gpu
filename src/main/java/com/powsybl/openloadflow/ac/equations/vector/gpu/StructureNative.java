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
 * <p>JNI glue: {@code native/cuda/jni/structure_jni.cu}.
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
     * @param closedBranchParameters  closed branches, {@code nClosedBranch*8}: (y g1 g2 b1 b2 ksi r1 a1)
     * @param closedBranchIndices closed branches, {@code nClosedBranch*12}: (Prow1 Qrow1 Prow2 Qrow2 ph1col v1col ph2col v2col a1col r1col alternateQRow1 alternateQRow2)
     * @param openBranchParameters  open branches, {@code nOpenBranch*7}: (y ksi g1 b1 g2 b2 r1)
     * @param openBranchIndices open branches, {@code nOpenBranch*6}: (Prow1 Qrow1 Prow2 Qrow2 v1col v2col)
     * @param shuntParameters  shunts, {@code nShunt*2}: (g b)
     * @param shuntIndices shunts, {@code nShunt*3}: (Prow Qrow vcol)
     * @param hvdcAcEmulationParameters  HVDC AC emulation, {@code nHvdcAcEmulation*5}: (p0 k lossFactor1 lossFactor2 r)
     * @param hvdcAcEmulationIndices HVDC AC emulation, {@code nHvdcAcEmulation*4}: (Prow1 Prow2 ph1col ph2col)
     * @return an opaque native handle; release it with {@link #destroyStructure}
     */
    public static native long createStructure(int n,
                                              double[] closedBranchParameters, int[] closedBranchIndices, double[] openBranchParameters, int[] openBranchIndices,
                                              double[] shuntParameters, int[] shuntIndices, double[] hvdcAcEmulationParameters, int[] hvdcAcEmulationIndices,
                                              int[] distrQElement, int[] distrQSide, double[] distrQWeight, int[] distrQRow, int[] distrQKind,
                                              double[] zeroImpedanceBranchParameters, int[] zeroImpedanceBranchIndices,
                                              int[] distrRhoRow, int[] distrRhoColumn, double[] distrRhoCoefficient,
                                              int[] distrShuntBRow, int[] distrShuntBColumn, double[] distrShuntBCoefficient, int[] closedBranchPhaseControlSide,
                                              double[] loadModelParameters, int[] loadModelIndices, int[] reactiveRelocationGroupStart, int[] reactiveRelocationCandidateRows,
                                              int[] openBranchAlternateReactiveDestination, int[] zeroImpedanceBranchAlternateReactiveDestination,
                                              int[] reactiveRelocationDependentBaseGroup, int[] reactiveRelocationBaseTargetRow, int[] reactiveRelocationDependentGroupOfRow);

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
}
