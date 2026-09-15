/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.powsybl.math.matrix.SparseMatrix;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

import java.util.Objects;

/**
 * The DC device context: one cuDSS factorization of OLF's DC matrix, reused against many
 * right-hand sides, with the per-column work (Woodbury combination, branch flows, limit screen)
 * done on the device — {@code native/cuda/jni/olf_gpu_dc_jni.cu}, in the same {@code libolfgpu}
 * as the AC Newton-Raphson.
 *
 * <p>The matrix is handed over as OLF's own {@link SparseMatrix} arrays: the column-compressed
 * storage of the transposed Jacobian OLF keeps IS the row-compressed storage of the true matrix
 * (rows = equations, columns = variables), so the device solves {@code J x = b} directly. Every
 * row index is the equation system's own {@code Variable.getRow()} / {@code Equation.getColumn()}.
 *
 * <p>Not thread-safe: one context serves one security analysis on one thread, like the AC batch
 * contexts. {@link #close()} frees the device memory; the {@link java.lang.ref.Cleaner} is a
 * safety net for a forgotten close.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuDcSolver implements AutoCloseable {

    /** Elements per contingency the alpha kernel solves in registers (MAX_K in the native). */
    public static final int MAX_ELEMENTS = 32;

    private static final java.lang.ref.Cleaner CLEANER = java.lang.ref.Cleaner.create();

    private final int n;
    private final State state;
    private final java.lang.ref.Cleaner.Cleanable cleanable;

    /** The native handle, owned by the cleaner action so a forgotten close still frees the device. */
    private static final class State implements Runnable {
        private long handle;

        private State(long handle) {
            this.handle = handle;
        }

        @Override
        public void run() {
            if (handle != 0) {
                destroyContext(handle);
                handle = 0;
            }
        }
    }

    private GpuDcSolver(int n, long handle) {
        this.n = n;
        this.state = new State(handle);
        this.cleanable = CLEANER.register(this, state);
    }

    /** True when {@code libolfgpu} (and its cuDSS/CUDA chain) loaded — the AC loader's verdict. */
    public static boolean isAvailable() {
        return GpuAcNewtonSolver.isAvailable();
    }

    /**
     * Factorize the DC matrix on the device.
     *
     * @param matrix OLF's Jacobian matrix (rows = variables, columns = equations, square)
     */
    public static GpuDcSolver create(SparseMatrix matrix) {
        Objects.requireNonNull(matrix);
        if (!isAvailable()) {
            throw new IllegalStateException("GPU native library not available");
        }
        int n = matrix.getRowCount();
        if (matrix.getColumnCount() != n) {
            throw new IllegalArgumentException("DC matrix is not square: " + n + " x " + matrix.getColumnCount());
        }
        int[] columnStart = matrix.getColumnStart();
        int nnz = columnStart[n];
        int[] rowIndices = matrix.getRowIndices();
        double[] values = matrix.getValues();
        if (rowIndices.length != nnz || values.length != nnz) {
            // the backing arrays may be larger than the filled part
            rowIndices = java.util.Arrays.copyOf(rowIndices, nnz);
            values = java.util.Arrays.copyOf(values, nnz);
        }
        long handle = createContext(n, columnStart, rowIndices, values);
        return new GpuDcSolver(n, handle);
    }

    /**
     * A context the DEVICE assembles: the structure {@link GpuDcAssembly} captured, the values from
     * the branch powers, the target vector from the generator and load targets, the slack
     * distribution on the device generator tables. Uploads every table of the assembly and fills +
     * factorizes once; {@link #fillTarget()} / {@link #solveTarget()} then run a load flow with no
     * host-side matrix at all.
     */
    public static GpuDcSolver createAssembled(GpuDcAssembly a) {
        Objects.requireNonNull(a);
        if (!isAvailable()) {
            throw new IllegalStateException("GPU native library not available");
        }
        long handle = createAssembled(a.n, a.rowPtr, a.colIdx, a.contribPtr, a.contribKind, a.contribElem, a.contribCoef,
                a.targetKind, a.targetElem, a.rhsPtr, a.rhsKind, a.rhsElem, a.rhsCoef);
        GpuDcSolver solver = new GpuDcSolver(a.n, handle);
        try {
            solver.setBranches(a.ph1Row, a.ph2Row, a.a1Row, a.a1, a.power);
            solver.setBuses(a);
            solver.setHvdcOffsets(a.hvdcP0);
            solver.setGenerators(a);
            solver.fillValues();
        } catch (RuntimeException e) {
            solver.close();
            throw e;
        }
        return solver;
    }

    /** Re-upload the branch model (powers, fixed shifts) after a tap change, then refill and refactorize. */
    public void refreshBranches(GpuDcAssembly a) {
        setBranches(a.ph1Row, a.ph2Row, a.a1Row, a.a1, a.power);
        setHvdcOffsets(a.hvdcP0);
        fillValues();
    }

    private void setBuses(GpuDcAssembly a) {
        setBuses(handle(), a.busPColumn, a.busPhRow, a.busLoadTargetP, a.busFlags, a.busSc, a.busGenStart, a.scBusPtr, a.scBusList, a.scSlackCount);
    }

    private void setGenerators(GpuDcAssembly a) {
        setGenerators(handle(), a.genInitialTargetP, a.genTargetP, a.genMinTargetP, a.genMaxTargetP, a.genMinP, a.genMaxP,
                a.genDroop, a.genParticipationFactor, a.genParticipating);
    }

    /** The AC-emulation offsets the target right-hand side reads (LfNetwork.getHvdcs() order). */
    public void setHvdcOffsets(double[] p0) {
        setHvdcOffsets(handle(), p0);
    }

    /** The loads' target P per bus (LfNetwork.getBuses() order). */
    public void setBusLoads(double[] loadTargetP) {
        setBusLoads(handle(), loadTargetP);
    }

    /** The buses left out of the mismatch and the distribution (islanded); {@code null} clears. */
    public void setExcludedBuses(int[] buses) {
        setExcludedBuses(handle(), buses);
    }

    /** The generators' current target P (the assembly's generator order). */
    public void setGeneratorTargets(double[] targetP) {
        setGeneratorTargets(handle(), targetP);
    }

    public double[] readGeneratorTargets() {
        return readGeneratorTargets(handle());
    }

    /** The matrix values from the branch powers, then the factorization (analysis on the first call). */
    public void fillValues() {
        fillValues(handle());
    }

    /** The device matrix values (pattern order) — the assembly tests' probe. */
    public double[] readValues() {
        return readValues(handle());
    }

    /** {@code DcLoadFlowEngine.getActivePowerMismatch} over the enabled buses (minus the excluded ones when asked). */
    public double activePowerMismatch(boolean withExcluded) {
        return activePowerMismatch(handle(), withExcluded);
    }

    /** The result of a device slack distribution: what {@code ActivePowerDistribution.Result} carries. */
    public record Distribution(double remainingMismatch, int iterations) {
    }

    /**
     * {@code ActivePowerDistribution.run} with the generation step on the device generator targets.
     *
     * @param participationType 0 = P_MAX, 1 = P, 2 = PARTICIPATION_FACTOR, 3 = REMAINING_MARGIN (the generation balance types)
     * @param referenceGenerator the assembly's reference generator index, or -1
     */
    public Distribution distributeSlack(int participationType, boolean useActiveLimits, int referenceGenerator, boolean withExcluded, double mismatch) {
        double[] out = distributeSlack(handle(), participationType, useActiveLimits, referenceGenerator, withExcluded, mismatch);
        return new Distribution(out[0], (int) out[1]);
    }

    /** The target vector on the device from the current generator targets, loads and branch model. */
    public void fillTarget() {
        fillTarget(handle());
    }

    public double[] readTarget() {
        return readTarget(handle());
    }

    /** Solve against the device target vector; returns the state (kept on the device for the loop kernels). */
    public double[] solveTarget() {
        return solveTarget(handle());
    }

    /** The value and rhs contribution lists refreshed on the SAME pattern (a term switched on or off). */
    public void setContributions(GpuDcAssembly a) {
        setContributions(handle(), a.contribPtr, a.contribKind, a.contribElem, a.contribCoef,
                a.targetKind, a.targetElem, a.rhsPtr, a.rhsKind, a.rhsElem, a.rhsCoef);
    }

    /** The phase controllers the device outer loop decides for ({@link GpuDcOuterLoops}). */
    void setPhaseControllers(int[] branch, int[] alphaColumn, int[] controlled, int[] side, double[] target,
                             double[] halfDeadband, int[] tapIndex, int[] tapPtr, double[] tapA1) {
        setPhaseControllers(handle(), branch, alphaColumn, controlled, side, target, halfDeadband, tapIndex, tapPtr, tapA1);
    }

    void resetPhaseControlDirections() {
        resetPhaseControlDirections(handle());
    }

    /** The incremental phase control decision per controller: the new tap index, or -1. */
    int[] phaseControlCheck() {
        return phaseControlCheck(handle());
    }

    /** The HVDC links in AC emulation the device outer loop decides for. */
    public void setHvdcs(int[] ph1Row, int[] ph2Row, double[] k, double[] p0, double[] pMax12, double[] pMax21, int[] status, int[] active) {
        setHvdcs(handle(), ph1Row, ph2Row, k, p0, pMax12, pMax21, status, active);
    }

    /** The AC-emulation limits decision per HVDC: the new status, or -1. */
    int[] hvdcCheck() {
        return hvdcCheck(handle());
    }

    /**
     * One block of LODF columns from the element states: {@code out[row + j * monitored.length]} is
     * the factor of {@code monitored[row]} for the outage held in element column {@code col0 + j}.
     */
    void lodfBlock(int[] monitored, int col0, int cols, double[] out) {
        lodfBlock(handle(), monitored, col0, cols, out);
    }

    /** One block of raw sensitivities of the function branches to the element states (the PTDF block). */
    void sensiBlock(int[] functions, int col0, int cols, double[] out) {
        sensiBlock(handle(), functions, col0, cols, out);
    }

    // --- DC sensitivity: the factor and flow states live on the device (GpuDcSensitivityAnalysis)

    /** State buffer selector. */
    public static final int SRC_BASE = 0;
    public static final int SRC_ALT = 1;
    public static final int SRC_POST = 2;

    /** Element kinds of the Woodbury alpha system. */
    public static final int EL_CONTINGENCY = 0;
    public static final int EL_ACTION = 1;

    /** The factor-group right-hand sides (sparse CSC) solved into the base or the alternate factor states. */
    public void sensiSetFactorColumns(int[] colPtr, int[] rows, double[] values, int cols, boolean alt) {
        sensiSetFactorColumns(handle(), colPtr, rows, values, cols, alt);
    }

    /** The already-solved flow-states column (OLF builds its modified target on the network). */
    public void sensiSetFlowStates(double[] states, boolean alt) {
        sensiSetFlowStates(handle(), states, alt);
    }

    /** The {@code +1/-1} element states OLF computed for the contingencies and the actions, column-major. */
    public void sensiSetElementStates(double[] contingencyStates, int contingencyColumns, double[] actionStates, int actionColumns) {
        sensiSetElementStates(handle(), contingencyStates, contingencyColumns, actionStates, actionColumns);
    }

    /** The elements of the contingency being processed. */
    public void sensiSetElements(int[] ph1Row, int[] ph2Row, int[] stateColumn, int[] kind, double[] diag, double[] newAlpha) {
        sensiSetElements(handle(), ph1Row, ph2Row, stateColumn, kind, diag, newAlpha);
    }

    /** The Woodbury update of the flow column and of every factor column, into the post buffers. */
    public void sensiWoodbury(int factorSource, int flowSource) {
        sensiWoodbury(handle(), factorSource, flowSource);
    }

    /**
     * One chunk of time-series steps: per step the slack distribution and the target vector, then
     * one multi-right-hand-side solve. {@code participationType} below zero means no distribution.
     */
    public void timeSeriesChunk(int steps, double[] generatorTargets, int generatorCount, double[] loadTargetP, int busCount,
                                double[] generatorInitialTargets, int[] generatorParticipating, double[] generatorMinTargetP,
                                double[] generatorMaxTargetP, double[] generatorMinP, double[] generatorMaxP,
                                double[] generatorDroop, double[] generatorParticipationFactor, double[] hvdcP0PerStep,
                                int participationType, boolean useActiveLimits, int referenceGenerator, double[] outStates,
                                double[] outGeneratorTargets, double[] outDistributed, double[] outMismatch) {
        timeSeriesChunk(handle(), steps, generatorTargets, generatorCount, loadTargetP, busCount, generatorInitialTargets,
                generatorParticipating, generatorMinTargetP, generatorMaxTargetP, generatorMinP, generatorMaxP, generatorDroop,
                generatorParticipationFactor, hvdcP0PerStep, participationType, useActiveLimits, referenceGenerator, outStates,
                outGeneratorTargets, outDistributed, outMismatch);
    }

    /** The AC-emulation limits decision of every step of a chunk, on its own state and status. */
    public int[] timeSeriesHvdcCheck(int steps, int[] statusPerStep, double[] p0PerStep) {
        return timeSeriesHvdcCheck(handle(), steps, statusPerStep, p0PerStep);
    }

    /** The incremental phase control decision for a subset of steps; the ratchet arrays are updated in place. */
    public int[] timeSeriesPhaseCheck(int[] stepIndices, int[] tapIndex, int[] allowedDirection, int[] directionChangeCount,
                                      int[] currentDirection) {
        return timeSeriesPhaseCheck(handle(), stepIndices, tapIndex, allowedDirection, directionChangeCount, currentDirection);
    }

    /** The phase controllers of the run ({@link GpuDcOuterLoops} builds the same tables). */
    public void setPhaseControllerTables(int[] branch, int[] alphaColumn, int[] controlled, int[] side, double[] target,
                                         double[] halfDeadband, int[] tapIndex, int[] tapPtr, double[] tapA1) {
        setPhaseControllers(handle(), branch, alphaColumn, controlled, side, target, halfDeadband, tapIndex, tapPtr, tapA1);
    }

    /** Re-solve a subset of the chunk's steps against the current factorization. */
    public void timeSeriesResolve(int[] stepIndices, double[] generatorTargets, int generatorCount, double[] loadTargetP,
                                  int busCount, double[] outStates) {
        timeSeriesResolve(handle(), stepIndices, generatorTargets, generatorCount, loadTargetP, busCount, outStates);
    }

    /** The slack mismatch of every step, from the injections as they now stand. */
    public void timeSeriesMismatch(int steps, double[] outMismatch) {
        timeSeriesMismatch(handle(), steps, outMismatch);
    }

    /** One contingency in one call: its elements, the Woodbury update and the factor evaluations. */
    public void sensiContingency(int[] ph1Row, int[] ph2Row, int[] stateColumn, int[] kind, double[] diag, double[] newAlpha,
                                 int factorSource, int flowSource, int[] pairFunction, int[] pairSide, int[] pairColumn,
                                 int[] functionBranch, int[] functionSide, double[] outSensi, double[] outFunction) {
        sensiContingency(handle(), ph1Row, ph2Row, stateColumn, kind, diag, newAlpha, factorSource, flowSource,
                pairFunction, pairSide, pairColumn, functionBranch, functionSide, outSensi, outFunction);
    }

    /** The sensitivity values and the function references — the only thing that comes back. */
    public void sensiValues(int factorSource, int flowSource, int[] pairFunction, int[] pairSide, int[] pairColumn,
                            int[] functionBranch, int[] functionSide, double[] outSensi, double[] outFunction) {
        sensiValues(handle(), factorSource, flowSource, pairFunction, pairSide, pairColumn, functionBranch, functionSide,
                outSensi, outFunction);
    }

    public int size() {
        return n;
    }

    private long handle() {
        if (state.handle == 0) {
            throw new IllegalStateException("GpuDcSolver is closed");
        }
        return state.handle;
    }

    /** Solve {@code J x = rhs} for one right-hand side (the pre-contingency and modified-target solves). */
    public double[] solve(double[] rhs) {
        return solve(handle(), rhs);
    }

    /**
     * The contingency elements: {@code +1} at {@code rowPlus[e]}, {@code -1} at {@code rowMinus[e]}
     * ({@code -1} = no entry, the slack side), one device-resident S column per element.
     *
     * @return false when the device cannot hold S — route the run to the CPU
     */
    public boolean setElements(int[] rowPlus, int[] rowMinus, int[] elemBranch) {
        return setElements(handle(), rowPlus, rowMinus, elemBranch);
    }

    /**
     * The per-element Woodbury terms, one per S column: {@code diag} is
     * {@code WoodburyEngine.getAlphaMatrixValue}'s diagonal ({@code 1/power} for an outage,
     * {@code 1/(oldPower - newPower)} for an action element) and {@code newAlpha} the phase shift a tap
     * action imposes ({@code 0} otherwise, {@code getAlphaRhsValue}).
     */
    public void setElementAttributes(double[] diag, double[] newAlpha) {
        setElementAttributes(handle(), diag, newAlpha);
    }

    /** One S column back on the host (the LODF/PTDF oracle in tests). */
    public double[] readElementStates(int elem) {
        return readElementStates(handle(), elem);
    }

    /**
     * The branch model in {@code LfNetwork.getBranches()} order: the angle rows of the two ends
     * ({@code -1} when a side has no angle variable — open or zero-impedance branch: flow 0), the
     * phase-shift variable row or {@code -1} with the constant shift, and OLF's {@code power}.
     */
    public void setBranches(int[] ph1Row, int[] ph2Row, int[] a1Row, double[] a1Const, double[] power) {
        setBranches(handle(), ph1Row, ph2Row, a1Row, a1Const, power);
    }

    /** The pre-contingency state the Woodbury correction is applied to. */
    public void setBase(double[] x0) {
        setBase(handle(), x0);
    }

    /** The flattened limit table (GpuLimitTable's arrays) the screen compares against. */
    public void setLimitTable(int[] branchGroupStart, int[] branchGroupEnd, int[] groupQuantity,
                              int[] groupLimitStart, int[] groupLimitEnd, double[] limitThreshold, double dcPowerFactor) {
        setLimitTable(handle(), branchGroupStart, branchGroupEnd, groupQuantity, groupLimitStart, groupLimitEnd, limitThreshold, dcPowerFactor);
    }

    /**
     * One chunk of contingencies (CSR over element indices). Fills {@code outStates} with the
     * {@code n x nCtg} post-contingency states (column-major) and returns
     * {@code [counts(nCtg) | records(nCtg x capacity)]} — the limit-table candidates per
     * contingency; a count above {@code capacity} means "screen this one on the host".
     * {@code ctgBase[c]} is the column of {@link #setChunkTargets} the contingency starts from, or
     * {@code -1} for the pre-contingency state.
     */
    public int[] woodburyChunk(int[] ctgPtr, int[] ctgElem, int[] ctgBase, double band, int capacity, double[] outStates) {
        return woodburyChunk(handle(), ctgPtr, ctgElem, ctgBase, band, capacity, outStates);
    }

    /**
     * {@code ConnectivityBreakAnalysis}'s sensitivity criterion for every contingency (CSR over
     * element indices) evaluated on the device S: the max over a contingency's elements of the
     * summed |sensitivities| on its own branches. Above {@code 1 - 1e-6} the contingency
     * potentially breaks connectivity and the graph confirmation decides.
     */
    public double[] connectivityCriterion(int[] ctgPtr, int[] ctgElem) {
        return connectivityCriterion(handle(), ctgPtr, ctgElem);
    }

    /**
     * The modified target columns of the coming chunk ({@code n x count}, column-major), solved
     * against the factorization and kept as the base states {@code ctgBase} refers to — the
     * batched form of {@code WoodburyEngine.runDcLoadFlowWithModifiedTargetVector}.
     */
    public void setChunkTargets(double[] targets, int count) {
        setChunkTargets(handle(), targets, count);
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private static native long createContext(int n, int[] rowPtr, int[] colIdx, double[] val);

    private static native void destroyContext(long handle);

    private static native double[] solve(long handle, double[] rhs);

    private static native boolean setElements(long handle, int[] rowPlus, int[] rowMinus, int[] elemBranch);

    private static native double[] readElementStates(long handle, int elem);

    private static native void setElementAttributes(long handle, double[] diag, double[] newAlpha);

    private static native void setBranches(long handle, int[] ph1Row, int[] ph2Row, int[] a1Row, double[] a1Const, double[] power);

    private static native void setBase(long handle, double[] x0);

    private static native void setLimitTable(long handle, int[] branchGroupStart, int[] branchGroupEnd, int[] groupQuantity,
                                             int[] groupLimitStart, int[] groupLimitEnd, double[] limitThreshold, double dcPowerFactor);

    private static native int[] woodburyChunk(long handle, int[] ctgPtr, int[] ctgElem, int[] ctgBase, double band, int capacity, double[] outStates);

    private static native void setChunkTargets(long handle, double[] targets, int count);

    private static native double[] connectivityCriterion(long handle, int[] ctgPtr, int[] ctgElem);

    private static native long createAssembled(int n, int[] rowPtr, int[] colIdx, int[] contribPtr, int[] contribKind, int[] contribElem,
                                               double[] contribCoef, int[] targetKind, int[] targetElem, int[] rhsPtr, int[] rhsKind,
                                               int[] rhsElem, double[] rhsCoef);

    private static native void setBuses(long handle, int[] pColumn, int[] phRow, double[] loadP, int[] flags, int[] sc, int[] genStart,
                                        int[] scBusPtr, int[] scBusList, int[] scSlackCount);

    private static native void setHvdcOffsets(long handle, double[] p0);

    private static native void setBusLoads(long handle, double[] loadP);

    private static native void setExcludedBuses(long handle, int[] buses);

    private static native void setGenerators(long handle, double[] init, double[] target, double[] minT, double[] maxT, double[] minP,
                                             double[] maxP, double[] droop, double[] pf, int[] participating);

    private static native void setGeneratorTargets(long handle, double[] target);

    private static native double[] readGeneratorTargets(long handle);

    private static native void fillValues(long handle);

    private static native double[] readValues(long handle);

    private static native double activePowerMismatch(long handle, boolean withExcluded);

    private static native double[] distributeSlack(long handle, int participationType, boolean useActiveLimits, int referenceGenerator,
                                                   boolean withExcluded, double mismatch);

    private static native void fillTarget(long handle);

    private static native double[] readTarget(long handle);

    private static native double[] solveTarget(long handle);

    private static native void setContributions(long handle, int[] contribPtr, int[] contribKind, int[] contribElem, double[] contribCoef,
                                                int[] targetKind, int[] targetElem, int[] rhsPtr, int[] rhsKind, int[] rhsElem, double[] rhsCoef);

    private static native void setPhaseControllers(long handle, int[] branch, int[] alphaColumn, int[] controlled, int[] side,
                                                   double[] target, double[] halfDeadband, int[] tapIndex, int[] tapPtr, double[] tapA1);

    private static native void resetPhaseControlDirections(long handle);

    private static native int[] phaseControlCheck(long handle);

    private static native void setHvdcs(long handle, int[] ph1Row, int[] ph2Row, double[] k, double[] p0, double[] pMax12,
                                        double[] pMax21, int[] status, int[] active);

    private static native int[] hvdcCheck(long handle);

    private static native void lodfBlock(long handle, int[] monitored, int col0, int cols, double[] out);

    private static native void sensiBlock(long handle, int[] functions, int col0, int cols, double[] out);

    private static native void sensiSetFactorColumns(long handle, int[] colPtr, int[] rows, double[] values, int cols, boolean alt);

    private static native void sensiSetFlowStates(long handle, double[] states, boolean alt);

    private static native void sensiSetElementStates(long handle, double[] contingencyStates, int contingencyColumns,
                                                     double[] actionStates, int actionColumns);

    private static native void sensiSetElements(long handle, int[] ph1Row, int[] ph2Row, int[] stateColumn, int[] kind,
                                                double[] diag, double[] newAlpha);

    private static native void sensiWoodbury(long handle, int factorSource, int flowSource);

    private static native void timeSeriesChunk(long handle, int steps, double[] generatorTargets, int generatorCount,
                                               double[] loadTargetP, int busCount, double[] generatorInitialTargets,
                                               int[] generatorParticipating, double[] generatorMinTargetP, double[] generatorMaxTargetP,
                                               double[] generatorMinP, double[] generatorMaxP, double[] generatorDroop,
                                               double[] generatorParticipationFactor, double[] hvdcP0PerStep, int participationType,
                                               boolean useActiveLimits,
                                               int referenceGenerator, double[] outStates, double[] outGeneratorTargets,
                                               double[] outDistributed, double[] outMismatch);

    private static native int[] timeSeriesPhaseCheck(long handle, int[] stepIndices, int[] tapIndex, int[] allowedDirection,
                                                     int[] directionChangeCount, int[] currentDirection);

    private static native int[] timeSeriesHvdcCheck(long handle, int steps, int[] statusPerStep, double[] p0PerStep);

    private static native void timeSeriesResolve(long handle, int[] stepIndices, double[] generatorTargets, int generatorCount,
                                                 double[] loadTargetP, int busCount, double[] outStates);

    private static native void timeSeriesMismatch(long handle, int steps, double[] outMismatch);

    private static native void sensiContingency(long handle, int[] ph1Row, int[] ph2Row, int[] stateColumn, int[] kind,
                                                double[] diag, double[] newAlpha, int factorSource, int flowSource,
                                                int[] pairFunction, int[] pairSide, int[] pairColumn, int[] functionBranch,
                                                int[] functionSide, double[] outSensi, double[] outFunction);

    private static native void sensiValues(long handle, int factorSource, int flowSource, int[] pairFunction, int[] pairSide,
                                           int[] pairColumn, int[] functionBranch, int[] functionSide, double[] outSensi,
                                           double[] outFunction);
}
