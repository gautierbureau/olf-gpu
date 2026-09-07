/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

/**
 * JNI binding: a FULL Newton-Raphson AC power flow run on the GPU behind one native call.
 * The native library libolfgpu (native/cuda/jni/olf_gpu_nr_jni.cu, built by the root
 * CMakeLists — see native/build-gpu.sh) keeps the fixed-pattern CSR Jacobian, the
 * mismatch and the state resident on the device: the four generated fused kernels
 * (closed-branch / open-branch / shunt / HVDC AC emulation) fill J and F each
 * iteration, cuDSS refactorizes and solves, a kernel applies the state update.
 *
 * <p>Distribution: libolfgpu is OUR wrapper (MPL) and is bundled in the jar, built
 * location-independent (no RPATH); what is NOT redistributable is NVIDIA's cuDSS, which
 * the wrapper only references dynamically ({@code NEEDED libcudss.so.0}). At load time
 * this class therefore PRELOADS the NVIDIA chain (libcublasLt → libcublas → libcudss)
 * from the user's install — {@code olf.cudss.path} system property or {@code CUDSS_ROOT}
 * env for cuDSS, {@code olf.cuda.path} property or {@code CUDA_HOME}/{@code CUDA_PATH}/
 * {@code CUDA_TOOLKIT} env for the toolkit libs — so the dynamic linker resolves the
 * bundled wrapper's dependency from the already-loaded sonames, with no RPATH and no
 * LD_LIBRARY_PATH. If cuDSS is already resolvable (ldconfig, LD_LIBRARY_PATH, or a local
 * RPATH-embedded build), the preload is simply unnecessary and load succeeds anyway.
 *
 * <p>The {@code olf.gpu.lib} system property overrides the classpath lookup with an
 * explicit path (dev builds). {@link #isAvailable()} is false when the library or its
 * cuDSS dependency cannot be loaded, so CUDA-free builds are unaffected.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcNewtonSolver {

    private static final boolean LOADED = GpuNativeLoader.load();

    private GpuAcNewtonSolver() {
    }

    public static boolean isAvailable() {
        return LOADED;
    }

    /**
     * Build the SHARED STRUCTURE for one equation-system extraction: the diagonal-pairing row
     * permutation, the fixed SUPERSET CSR pattern (every row carries its full power-equation
     * entries + a guaranteed diagonal), the per-family device data with role-folded destination
     * arrays, the DISTR_Q / DISTR_RHO / zero-impedance packs and the per-element static trig.
     * This is everything BOTH solver contexts (single-solve and batched N-1) derive from one
     * {@link GpuAcData} — build it once and pass the handle to {@link #createContext} and
     * {@link #createBatchContext}, so the base load flow and the batched security analysis
     * sharing one extraction share one device-resident structure (see {@link GpuAcStructures}
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
    private static int[] noPcSide(int ncb) {
        int[] a = new int[ncb];
        java.util.Arrays.fill(a, -1);
        return a;
    }

    public static native long createStructure(int n,
                                              double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
                                              double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx,
                                              int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                              double[] ziIn, int[] ziIdx,
                                              int[] drRow, int[] drCol, double[] drCoef,
                                              int[] dbRow, int[] dbCol, double[] dbCoef, int[] cbPcSide,
                                              double[] vdlIn, int[] vdlIdx, int[] qrPtr, int[] qrRows);

    /**
     * Release a structure handle from {@link #createStructure} (idempotent for handle 0). Contexts
     * built from it keep the device-resident structure alive natively (shared ownership) — this only
     * drops the Java-side reference, so it is always safe to call once the handle is no longer used
     * to create new contexts.
     */
    public static native void destroyStructure(long handle);

    /**
     * Build a CACHED solver context on a shared structure: the per-solve device buffers and the
     * cuDSS objects persist in it, and cuDSS's symbolic ANALYSIS is done once — subsequent
     * {@link #solveContext} calls only refactorize. The pattern / families / destination arrays
     * are NOT duplicated: they live in the structure, shared with any batched context built from
     * the same handle.
     *
     * @param structureHandle a structure from {@link #createStructure}
     * @return an opaque native handle; release it with {@link #destroyContext}
     */
    public static native long createContext(long structureHandle);

    /**
     * Run a full Newton-Raphson AC power flow on the GPU against a cached context — only
     * the initial state, the targets and the activation mask cross JNI.
     *
     * @param handle  a context from {@link #createContext}
     * @param x0      initial state, indexed by variable row (length n)
     * @param target  targets in the STABLE row order (GpuAcDataExtractor#extractTargets)
     * @param rowMode per row: 1 = power equation, 0 = slack/PV identity (the device mask
     *                zeroes the row, sets the diagonal to 1 and F = x - target)
     * @param maxIter maximum Newton iterations
     * @param tol     convergence tolerance on the mismatch infinity norm
     * @param convEpsPerEq if {@code > 0}, the device inner-Newton uses OLF's exact
     *                     {@code DefaultNewtonRaphsonStoppingCriteria} rule
     *                     ({@code ‖F‖₂ < sqrt(convEpsPerEq² · n)}) instead of the
     *                     infinity-norm floor
     * @param lsMode  device step scaling: 0 = none, 1 = line search, 2 = max voltage change
     * @param maxDv   MAX_VOLTAGE_CHANGE (lsMode 2): clamp on |Δv| per iteration (BUS_V rows)
     * @param maxDphi MAX_VOLTAGE_CHANGE (lsMode 2): clamp on |Δφ| per iteration (BUS_PHI rows, radians)
     * @param varType per row, the variable kind for the maxVoltageChange clamp (0 = BUS_PHI,
     *                1 = BUS_V, 2 = other); see GpuAcActivationExtractor#extractVarType. Unused for lsMode 0/1.
     * @return {@code double[n + 2]}: the converged state, then iterations, then the final
     *         mismatch infinity norm
     */
    public static native double[] solveContext(long handle, double[] x0, double[] target,
                                               int[] rowMode, int maxIter, double tol, double convEpsPerEq,
                                               int lsMode, int lsMaxIter, double lsStepFold,
                                               double maxDv, double maxDphi, int[] varType);

    /**
     * VALUES-level refresh: re-upload only the per-element parameter packs (families + zero-impedance
     * rho) into the context's SHARED structure, keeping its CSR pattern and the context's cuDSS
     * symbolic analysis; the per-element static trig is re-derived on device. Valid only when the
     * element counts / index packs are unchanged (a transformer tap or shunt-susceptance move) — the
     * caller guarantees this by taking this path only when {@link GpuAcData#sameStructureAs} holds,
     * and must then re-key the structure cache to the new extraction ({@link GpuAcStructures#rebind})
     * so content matching never aliases the refreshed structure under its old values. Far cheaper
     * than a rebuild.
     */
    public static native void refreshContextValues(long handle, double[] cbIn, double[] obIn,
                                                   double[] shIn, double[] hvIn, double[] ziIn);

    /** Release a context from {@link #createContext} (idempotent for handle 0). */
    public static native void destroyContext(long handle);

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

    /**
     * One-shot convenience: {@link #createStructure} + {@link #createContext} + {@link #solveContext}
     * + destroy both. For repeated solves on the same structure (outer loops), cache the context
     * instead — see GpuNewtonRaphson.
     */
    public static double[] solve(double[] x0, double[] target, int[] rowMode, int maxIter, double tol,
                                 double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
                                 double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx,
                                 int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                 double[] ziIn, int[] ziIdx,
                                 int[] drRow, int[] drCol, double[] drCoef,
                                 int[] dbRow, int[] dbCol, double[] dbCoef) {
        return solve(x0, target, rowMode, maxIter, tol, cbIn, cbIdx, obIn, obIdx, shIn, shIdx, hvIn, hvIdx,
                dqElem, dqSide, dqWeight, dqRow, dqKind, ziIn, ziIdx, drRow, drCol, drCoef, dbRow, dbCol, dbCoef,
                new int[] {0}, new int[0]);
    }

    /** {@link #solve} with the GROUP pilot-Q relocation tables (plan-A6) — required whenever the
     *  index packs carry the {@code -2-groupId} qAlt encoding (shared transformer-VC groups). */
    @SuppressWarnings("java:S107")
    public static double[] solve(double[] x0, double[] target, int[] rowMode, int maxIter, double tol,
                                 double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
                                 double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx,
                                 int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                 double[] ziIn, int[] ziIdx,
                                 int[] drRow, int[] drCol, double[] drCoef,
                                 int[] dbRow, int[] dbCol, double[] dbCoef, int[] qrPtr, int[] qrRows) {
        long structure = createStructure(x0.length, cbIn, cbIdx, obIn, obIdx, shIn, shIdx, hvIn, hvIdx,
                dqElem, dqSide, dqWeight, dqRow, dqKind, ziIn, ziIdx, drRow, drCol, drCoef, dbRow, dbCol, dbCoef,
                noPcSide(cbIdx.length / 12), new double[0], new int[0], qrPtr, qrRows);
        try {
            long handle = createContext(structure);
            try {
                // undamped (no device step scaling): lsMode 0 ignores maxDv/maxDphi/varType
                return solveContext(handle, x0, target, rowMode, maxIter, tol, 0.0, 0, 0, 0.0, 0.0, 0.0, new int[x0.length]);
            } finally {
                destroyContext(handle);
            }
        } finally {
            destroyStructure(structure);
        }
    }

    /**
     * Build a CACHED solver context for BATCHED N-1 security analysis on a shared structure:
     * multiple scenarios (post-contingency states) against the same network STRUCTURE and
     * superset pattern. Per-scenario element disablement is encoded as device masks; converged
     * scenarios can stop early. cuDSS's symbolic analysis is done once over the superset
     * pattern. The pattern / families / destination arrays are NOT duplicated: they live in
     * the structure, shared with any single-solve context built from the same handle (the
     * base-case load flow) — only the per-scenario state and the reactive-limits /
     * distributed-slack outer-loop packs are owned here.
     *
     * @param structureHandle a structure from {@link #createStructure}
     * @param nScenarios      number of contingency scenarios to solve in parallel
     * @return an opaque native handle for batched solves; release with {@link #destroyBatchContext}
     */
    public static native long createBatchContext(long structureHandle, int nScenarios, int mode,
                                                  int[] rlVrow, double[] rlMinQ, double[] rlMaxQ, double[] rlLoadQ,
                                                  double[] rlVtarget, byte[] rlMonitored, double[] rowSlope,
                                                  int[] dsPhiRow, double[] dsFactor,
                                                  int dsSlackPhiRow, double dsSlackTargetP,
                                                  double[] dsBaseTargetP, double[] dsMaxDelta, double[] dsMinDelta,
                                                  int dsMode, double[] dsMarginUp, double[] dsMarginDown,
                                                  double dsSlackMarginUp, double dsSlackMarginDown);

    /**
     * Run batched Newton-Raphson AC power flow on the GPU: one iteration loop over all
     * scenarios, with per-scenario convergence retirement (stop filling/solving once
     * converged). Scenarios remain in the batch until they converge or hit maxIter.
     *
     * <p>The base state/targets are SHARED by every scenario (a single-branch N-1 starts from the same
     * converged base operating point): they are passed as single length-{@code n} rows and tiled across
     * the batch on-device. The row-mode is PER-SCENARIO ({@code [nScenarios][n]}) so a contingency can
     * freeze islanded buses or flip PV↔PQ (reactive limits) independently; it is uploaded as-is and
     * mutated in place on-device by the outer-loop kernels. The disabled-element mask is also
     * per-scenario. {@code nScenarios} is {@code disabledElementMask.length}.
     *
     * @param handle              a batch context from {@link #createBatchContext}
     * @param x0Base              shared initial state, {@code [n]}: tiled to every scenario on-device
     * @param targetBase          shared targets, {@code [n]}: in STABLE row order, tiled per scenario
     * @param rowMode             per-scenario row modes, {@code [nScenarios][n]}: 1 = equation, 0 = identity
     *                            (islanded buses frozen, slack/PV identity rows)
     * @param disabledElementMask per-scenario disabled elements, {@code [nScenarios][nElements]}: bit-packed
     *                            (true if that element is outaged in this scenario; elements enumerated
     *                            as: first ncb closed branches, then nob open branches, then nsh shunts, then nhv HVDCs)
     * @param maxIter             maximum Newton iterations per scenario
     * @param tol                 convergence tolerance on mismatch infinity norm (the device floor)
     * @param convEpsPerEq        if {@code > 0}, the inner-Newton convergence test uses OLF's EXACT criterion
     *                            ({@code DefaultNewtonRaphsonStoppingCriteria}: {@code ‖F‖₂ < sqrt(convEpsPerEq² · n)})
     *                            instead of the infinity-norm floor — so a batched SA can stop on the SAME
     *                            decision as the CPU loadflow (apples-to-apples timing). {@code <= 0} keeps the
     *                            {@code tol} infinity-norm floor.
     * @return {@code [nScenarios][n + 3]}: per-scenario [state(n), iterations, finalMismatch, convergenceStatus]
     *         where convergenceStatus is: 0 = converged, 1 = max_iter, -1 = diverged/NaN
     */
    public static native double[][] solveBatchContext(long handle, double[] x0Base, double[] targetBase,
                                                       int[][] rowMode, int[] baseRowMode, boolean[][] disabledElementMask,
                                                       int[] targetOverrideIndex, double[] targetOverrideValue,
                                                       int[] dsOverrideBus, double[] dsOverrideFactorDelta,
                                                       double[] dsOverrideBaseDelta, double[] dsOverrideMaxDelta,
                                                       double[] dsOverrideMinDelta, double[] dsAnchorShift,
                                                       double[] dsSlackShift,
                                                       int[] rlVtOvSlotIdx, double[] rlVtOvVal,
                                                       int maxIter, double tol,
                                                       double reactiveLimitTol, double slackDistTol, double convEpsPerEq,
                                                       int lsMode, double maxDv, double maxDphi, int[] varType,
                                                       double[] x0PerScenario,
                                                       int[] shuntOvSlotElem, double[] shuntOvGb,
                                                       int[] rlStateIn, int[] rlStateOut,
                                                       int[] r1OvSlotElem, double[] r1OvVal,
                                                       int maxOuterIterations, int maxPqPvSwitch, int dsRejectResidual, int skipRlPhase,
                                                       int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem);

    /** Read back the batch context's per-scenario TARGET buffer (n * nScenarios) after a solve — the device
     *  outer loops mutate targets in place; the strategy/pair phase re-anchors its slack distribution at the
     *  PARENT contingency's final distributed targets (OLF's setGeneratorsInitialTargetPToTargetP). */
    public static native double[] readBatchTargets(long handle);

    /** Batched SENSITIVITY solve for the incremental outer-loop family: J at the given per-scenario
     *  states (same forward system as the Newton), unit RHS at each {@code unitRows} pinned-variable
     *  identity row, the {@code observeCols} variable entries of each solution gathered per scenario.
     *  Returns {@code [nScenarios * unitRows.length * observeCols.length]} laid out scenario-major.
     *  MUST follow a solveBatchContext call on the same handle with the same per-scenario overrides. */
    public static native double[] solveBatchSensitivities(long handle, double[] states,
                                                          int[] unitRows, int[] observeCols);

    /** Per-scenario slack-only-DS absorbed totals after a solve, or null when the context is not
     *  slack-only — the pair phase seeds its scenarios with the parent contingency's value. */
    public static native double[] readBatchSlackAbsorbed(long handle);

    /** Release a batch context from {@link #createBatchContext} (idempotent for handle 0). */
    /** Install (or clear, with nulls) the PER_EQUATION_TYPE per-row stopping thresholds on a batch
     *  context: {@code thrIdent[r]} for identity/slope rows, {@code thrPower[r]} for power rows —
     *  the convergence test then requires EVERY row's |F| under its threshold (OLF's
     *  PerEquationTypeStoppingCriteria) instead of the 2-norm/infnorm rule. Contexts are CACHED, so
     *  a run without the criterion must clear (null, null). */
    public static native void setBatchRowCriteria(long handle, double[] thrIdent, double[] thrPower);

    /** Install (or clear, with nulls) the reactive-capability-CURVE channel on a batch context —
     *  per-scenario dynamic RL limits evaluated at the current targets (see GpuBatchRlCurves). */
    public static native void setBatchRlCurves(long handle, int[] cvIdx, double[] cvIn, double[] cvPts);

    /** Remote-member RL entries (plan-B1): per-entry remote kinds + compare rows (PILOT v column for
     *  remote group members — both RL switch conditions test the pilot's V against the GROUP target).
     *  {@code dqMemberQrGroup} (plan-B4) links each group member to its pilot's relocation candidate
     *  group (−1 = none) so the kernels suppress the CURRENT carrier's DISTR_Q row and hand its
     *  target the pilot's Q when the free role re-picks. Nulls clear. */
    public static native void setBatchRlRemote(long handle, byte[] rlRemote, int[] rlCmpVrow,
                                               int[] rmqRow, int[] rmqElem, int[] rmqSide, int[] rmqKind,
                                               double[] rmqWeight,
                                               int[] dqKMember, int[] dqJMember, double[] dqMemberKey,
                                               int[] dqMemberRlEntry, int[] dqMemberGroupStart,
                                               int[] dqMemberGroupCount, int[] dqMemberRow, byte[] dqMemberHasRow,
                                               int[] dqMemberQrGroup);

    /** Install (or clear, with nulls) per-scenario DISTR_Q weight overrides on a batch context —
     *  {@code idx} = flat {scenario, contribution} pairs, {@code w} = the renormalized weight (a gen
     *  outage at a remote-VC controller bus re-runs updateReactiveKeys on the CPU). Per-chunk state. */
    public static native void setBatchDqWeightOverrides(long handle, int[] idx, double[] w);

    /** Install (or clear, with nulls) per-scenario DISTR_RHO coefficient overrides on a batch
     *  context — {@code idx} = flat {scenario, contribution} pairs, {@code c} = the renormalized
     *  1/count' coefficient (a shared-TVC group member outaged in that scenario). Per-chunk state. */
    public static native void setBatchDrCoefOverrides(long handle, int[] idx, double[] c);

    /** Install (or clear, with null) per-scenario ACTIVE equation counts for the L2 stopping
     *  threshold sqrt(eps^2 * nActive) — the CPU removes equations for outaged variable-carrying
     *  elements (rho/a1 identities) and islanded buses; the padded device n over-counts those rows
     *  and skews the stop decision by the sqrt ratio. Per-chunk state. */
    public static native void setBatchActiveRowCounts(long handle, int[] nActive);

    public static native void destroyBatchContext(long handle);

    /**
     * Solves {@code Aᵀ x = b} for an explicit CSR matrix {@code A}, by building the transpose and
     * factorizing it. Validation / measurement entry for the batched-adjoint work (docs/tasks.md #40
     * step 1a) — NOT the production path.
     *
     * <p>cuDSS through 0.8.0 has no transpose solve (verified: its header carries no transpose/OP_T
     * symbol), so one GPU factorization yields ONE direction, where KLU's {@code klu_solve} /
     * {@code klu_tsolve} pair yields two from the same factors. Our Jacobian is stored as TRUE CSR
     * (row = equation, col = variable) so the plain solve serves the forward Newton; the adjoint
     * ({@code Jᵀλ = …}) is therefore the direction the forward's factors cannot provide, and until
     * cuDSS ships a transpose solve we buy it with a second factorization of an explicit transpose.
     *
     * <p>Deliberately self-contained (own handle/config/data, no Newton context) so a failure here
     * can only mean the CSR transpose or the cuDSS factorize/solve of {@code Aᵀ} is wrong — never the
     * adjoint's RHS assembly or contraction.
     *
     * @param n       matrix dimension
     * @param rowPtr  CSR row pointers of {@code A}, length {@code n+1}
     * @param colIdx  CSR column indices of {@code A}, length {@code rowPtr[n]}
     * @param vals    CSR values of {@code A}, length {@code rowPtr[n]}
     * @param b       right-hand side, length {@code n}
     * @return {@code x} such that {@code Aᵀ x = b}
     */
    public static native double[] solveTransposedExplicit(int n, int[] rowPtr, int[] colIdx,
                                                          double[] vals, double[] b);

    /**
     * One-shot batch convenience: {@link #createStructure} + {@link #createBatchContext} +
     * {@link #solveBatchContext} + destroy both. For repeated batches on the same structure,
     * cache the context instead. The base state/targets/row-mode are shared by every scenario
     * (tiled on-device); {@code nScenarios} is {@code disabledElementMask.length}.
     */
    public static double[][] solveBatch(double[] x0Base, double[] targetBase, int[][] rowMode,
                                         boolean[][] disabledElementMask, int mode, int maxIter, double tol,
                                         double reactiveLimitTol, double slackDistTol,
                                         double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
                                         double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx,
                                         int[] rlVrow, double[] rlMinQ, double[] rlMaxQ, double[] rlLoadQ,
                                         double[] rlVtarget, int[] dsPhiRow, double[] dsFactor,
                                         int dsSlackPhiRow, double dsSlackTargetP,
                                         double[] dsBaseTargetP, double[] dsMaxDelta, double[] dsMinDelta,
                                         int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                         int[] drRow, int[] drCol, double[] drCoef,
                                         int[] dbRow, int[] dbCol, double[] dbCoef, double[] ziIn, int[] ziIdx) {
        int n = x0Base.length;
        int nScenarios = disabledElementMask.length;
        long structure = createStructure(n, cbIn, cbIdx, obIn, obIdx, shIn, shIdx, hvIn, hvIdx,
                dqElem, dqSide, dqWeight, dqRow, dqKind, ziIn, ziIdx, drRow, drCol, drCoef, dbRow, dbCol, dbCoef,
                noPcSide(cbIdx.length / 12), new double[0], new int[0], new int[] {0}, new int[0]);
        try {
            long handle = createBatchContext(structure, nScenarios, mode,
                    rlVrow, rlMinQ, rlMaxQ, rlLoadQ, rlVtarget, new byte[rlVrow.length], new double[x0Base.length],
                    dsPhiRow, dsFactor, dsSlackPhiRow, dsSlackTargetP,
                    dsBaseTargetP, dsMaxDelta, dsMinDelta, 0, new double[0], new double[0], 0, 0);
            try {
                // one-shot: no per-scenario gen-override flips, so any scenario's row mode IS the base row mode
                return solveBatchContext(handle, x0Base, targetBase, rowMode, rowMode[0], disabledElementMask,
                        new int[0], new double[0], new int[0], new double[0], new double[0], new double[0], new double[0], null, null, null, null,
                        maxIter, tol, reactiveLimitTol, slackDistTol, 0.0, 0, 0.0, 0.0, null, null, null, null, null, null, null, null, 0, 0, 0, 0, 0, null, null);
            } finally {
                destroyBatchContext(handle);
            }
        } finally {
            destroyStructure(structure);
        }
    }
}
