/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.solver.gpu.batch.ControlReHoming;

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
 * the wrapper only references dynamically ({@code NEEDED libcudss.so.0}): the system loader
 * resolves it ({@code LD_LIBRARY_PATH} or ldconfig, see GpuNativeLoader). {@link #isAvailable()}
 * is false when the library or its cuDSS dependency cannot be loaded, so CUDA-free builds are
 * unaffected.
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
     * This is everything a batched context derives from one {@link GpuAcData} — build it once
     * and pass the handle to {@link #createBatchContext}, so the base load flow and the batched
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
    /** No dependent-pilot group for any candidate group (the raw-row entry points have no merged controls). */
    private static int[] noDepGroups(int[] qrPtr) {
        int[] a = new int[Math.max(0, qrPtr.length - 1)];
        java.util.Arrays.fill(a, -1);
        return a;
    }

    /** No dependent-pilot group for any device row. */
    private static int[] noDepRows(int n) {
        int[] a = new int[n];
        java.util.Arrays.fill(a, -1);
        return a;
    }

    private static int[] noQAlt(int ne) {
        int[] a = new int[ne * 2];
        java.util.Arrays.fill(a, -1);
        return a;
    }

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

    /**
     * One-shot undamped pure Newton-Raphson: {@link #createStructure} + {@link #solveOnStructure}, then the
     * structure is destroyed.
     *
     * @return {@code double[n + 3]}: the state, then iterations, the final mismatch infinity norm and the
     *         convergence status (see {@link #solveBatchContext})
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
                new int[] {0}, new int[0], noQAlt(obIdx.length / 6), noQAlt(ziIdx.length / 11));
    }

    /** {@link #solve} with the GROUP pilot-Q relocation tables (plan-A6) and the ob/zi alternate
     *  Q rows — required whenever the index packs carry qAlt encodings (shared groups / remote
     *  members). Pass all-(-1) ob/zi alt arrays when only cb/sh relocate. */
    @SuppressWarnings("java:S107")
    public static double[] solve(double[] x0, double[] target, int[] rowMode, int maxIter, double tol,
                                 double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
                                 double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx,
                                 int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                                 double[] ziIn, int[] ziIdx,
                                 int[] drRow, int[] drCol, double[] drCoef,
                                 int[] dbRow, int[] dbCol, double[] dbCoef, int[] qrPtr, int[] qrRows,
                                 int[] obQAlt, int[] ziQAlt) {
        long structure = createStructure(x0.length, cbIn, cbIdx, obIn, obIdx, shIn, shIdx, hvIn, hvIdx,
                dqElem, dqSide, dqWeight, dqRow, dqKind, ziIn, ziIdx, drRow, drCol, drCoef, dbRow, dbCol, dbCoef,
                noPcSide(cbIdx.length / 12), new double[0], new int[0], qrPtr, qrRows, obQAlt, ziQAlt,
                noDepGroups(qrPtr), noDepGroups(qrPtr), noDepRows(x0.length));
        try {
            return solveOnStructure(structure, x0, target, rowMode, maxIter, tol, 0, 0.0, 0.0, new int[x0.length]);
        } finally {
            destroyStructure(structure);
        }
    }

    /**
     * One-shot pure Newton-Raphson on an existing structure: a batch-of-one {@link #createBatchContext}
     * without any outer-loop pack, one {@link #solveBatchContext}, then the context is destroyed.
     *
     * @param lsMode  device step scaling: 0 = none, 1 = line search, 2 = max voltage change
     * @param maxDv   MAX_VOLTAGE_CHANGE (lsMode 2): clamp on |dv| per iteration (BUS_V rows)
     * @param maxDphi MAX_VOLTAGE_CHANGE (lsMode 2): clamp on |dphi| per iteration (BUS_PHI rows, radians)
     * @param varType per row, the variable kind for the clamp, see GpuAcActivationExtractor#extractVarType
     * @return {@code double[n + 3]}: the state, then iterations, the final mismatch infinity norm and the
     *         convergence status (see {@link #solveBatchContext})
     */
    @SuppressWarnings("java:S107")
    public static double[] solveOnStructure(long structure, double[] x0, double[] target, int[] rowMode,
                                            int maxIter, double tol, int lsMode, double maxDv, double maxDphi,
                                            int[] varType) {
        long handle = createBatchContext(structure, 1, com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache.MODE_UBATCH,
                new int[0], new double[0], new double[0], new double[0], new double[0], new byte[0], new double[x0.length],
                new int[0], new double[0], -1, 0.0, new double[0], new double[0], new double[0],
                0, new double[0], new double[0], 0, 0);
        try {
            return solveBatchContext(handle, x0, target, new int[][] {rowMode}, rowMode, new boolean[1][0],
                    new int[0], new double[0], new int[0], new double[0], new double[0], new double[0], new double[0], null, null, null, null, null, null, null, null,
                    maxIter, tol, 0.0, 0.0, 0.0, lsMode, maxDv, maxDphi, varType, null, null, null, null, null, null, null, 0,
                    com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop.MAX_SWITCH_PQ_PV_DEFAULT_VALUE,
                    0, 0, 0, null, null)[0];
        } finally {
            destroyBatchContext(handle);
        }
    }

    /**
     * Build a CACHED solver context for BATCHED N-1 security analysis on a shared structure:
     * multiple scenarios (post-contingency states) against the same network STRUCTURE and
     * superset pattern. Per-scenario element disablement is encoded as device masks; converged
     * scenarios can stop early. cuDSS's symbolic analysis is done once over the superset
     * pattern. The pattern / families / destination arrays are NOT duplicated: they live in
     * the structure, built once per extraction — only the per-scenario state and the reactive-limits /
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
    /** Starting values for the next batch solve that is given no full per-scenario state: {@code s * n + row} ->
     *  value, written over the base state the device tiles; consumed by that solve. Null clears. */
    public static native void setBatchX0Seeds(long handle, int[] flatIndex, double[] value);

    /** Per (scenario, coupler) spanning-tree flag, as flat {slot, ziElement, isTree} triples, for the
     *  scenarios where OLF's tree has moved away from the one the structure was extracted on. Empty
     *  clears. */
    public static native void setBatchZiTreeOverrides(long handle, int[] slotElemVal);

    /** OLF_GPU_LOOP_TRACE by contingency: the slot to trace in the next solve of this context, -1 = none. */
    public static native void setBatchLoopTraceSlot(long handle, int slot);

    public static native double[][] solveBatchContext(long handle, double[] x0Base, double[] targetBase,
                                                       int[][] rowMode, int[] baseRowMode, boolean[][] disabledElementMask,
                                                       int[] targetOverrideIndex, double[] targetOverrideValue,
                                                       int[] dsOverrideBus, double[] dsOverrideFactorDelta,
                                                       double[] dsOverrideBaseDelta, double[] dsOverrideMaxDelta,
                                                       double[] dsOverrideMinDelta, double[] dsAnchorShift,
                                                       double[] dsSlackShift,
                                                       int[] rlVtOvSlotIdx, double[] rlVtOvVal,
                                                       int[] rlQOvSlotEntry, double[] rlQOvMinMax,
                                                       int[] cvOvSlotEntry, double[] cvOvStepScale,
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

    /** The per-scenario ROW MODES the device ended with — the companion to {@link #readBatchTargets}.
     *  A row's equation is (mode, target) TOGETHER, and the outer-loop kernels rewrite both; the host's
     *  {@code rowModePerScenario} is only what setup asked for. Reading a target without its final mode
     *  can say "the target is right" about a row that is no longer the equation it was set up to be. */
    public static native int[] readBatchRowModes(long handle);

    /** VALUES refresh addressed by the SHARED STRUCTURE handle: same semantics as
     *  {@link #refreshStructureValues} (tap r1/a1 or shunt-b moves with an unchanged pattern), but
     *  usable when the caller holds a BATCH context — one extraction has ONE value set, every
     *  context sharing the structure sees the update. */
    public static native void refreshStructureValues(long structureHandle, double[] cbIn, double[] obIn,
                                                     double[] shIn, double[] hvIn, double[] ziIn);

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

    /** The driver's DistributedSlack result per (scenario, participating generator): the cumulative
     *  target delta (pu) the water-fill gave each generator — null when the context has no per-generator
     *  list. Layout {@code s * ngen + g}, g in the extraction's generator order. */
    public static native double[] readBatchDsGenCum(long handle);

    /** Per scenario, the slack's RETAINED share of the distributed mismatch (pu) — what OLF's own loop
     *  gives the slack bus's participating generators, which have no device row of their own. */
    public static native double[] readBatchDsRetain(long handle);

    /** One scenario's state at its last CONVERGED inner solve of the last solve, or its starting state when none
     *  converged: what OLF's network holds after a run that failed, where it starts that contingency's strategies. */
    public static native double[] readBatchLastConvergedState(long handle, int slot);

    /** How many {@code solveBatchContext} calls the context has run. The distributed-slack read-backs
     *  ({@link #readBatchDsRetain}, {@link #readBatchDsGenCum}, {@link #readBatchSlackAbsorbed}) are reset at
     *  every solve: they describe the whole distribution only when one solve covered it. */
    public static native long readBatchSolveCount(long handle);

    /** Per scenario of the last batch solve: {solve passes, first pass's Newton iterations, total}, flattened. */
    public static native int[] readBatchOlStats(long handle);

    /**
     * Upload the run's limit thresholds and the branch model they are compared against, once per
     * run: a contingency cannot change a threshold, so this outlives every chunk. Returns 0 when the
     * upload failed (the caller then screens on the host).
     *
     * <p>Groups and limits are flat, in the host table's order, so a record is simply the limit's
     * index. See {@code GpuViolationScreen} for the layout and {@code olf_gpu_violations.cu} for
     * why the kernel screens rather than decides.
     */
    public static native long uploadViolationTable(
            int[] branchFamily, int[] branchElem, int[] branchGroupStart, int[] branchGroupEnd,
            int[] groupSide, int[] groupQuantity, int[] groupLimitStart, int[] groupLimitEnd,
            double[] limitThreshold,
            int[] busVRow, double[] busHighLimit, double[] busLowLimit,
            double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx);

    /**
     * Screen a chunk of post-contingency states against an uploaded table.
     *
     * @param states {@code nScenarios * n} states, scenario-major
     * @param band relative slack on every threshold, so the screen can only over-emit
     * @param capacity records per scenario
     * @return {@code nScenarios} counts, then {@code nScenarios * capacity} records; a count ABOVE
     *         the capacity means that scenario overflowed and must be scanned on the host
     */
    public static native int[] screenViolationsBatch(long tableHandle, double[] states, int nScenarios,
                                                     int n, double band, int capacity);

    /**
     * Install (or clear, with null) the per-scenario distributed-slack participants that a
     * fragmenting contingency strands in another synchronous component, as {@code {slot, participant}}
     * pairs indexing {@code dsPhiRow}.
     *
     * <p>OLF distributes over {@code getSynchronousNetworks().getFirst()}, so a stranded participant
     * is not a participant: the device drops it from the distribution AND renormalizes the survivors'
     * shares — including the slack's retained share and the stopping tolerance — over the set that
     * remains. Doing only the first half leaves the pass retiring with most of the mismatch
     * undistributed.
     */
    public static native void setBatchDsIslanded(long handle, int[] slotParticipantPairs);

    /**
     * Install (or clear, with null) the per-scenario P-balance row of a distributed-slack participant
     * whose row THIS scenario moves, as {@code {slot, participant, row}} triples ({@code participant}
     * indexes {@code dsPhiRow}, {@code row} is an equation row in the same space as {@code dsPhiRow}).
     *
     * <p>{@code dsPhiRow} is structural, and under the coupler-tree relocation a child bus's P balance
     * sits on its parent coupler's {@code DUMMY_P} row. A contingency that OPENS that coupler pins the
     * dummy row to a {@code DUMMY_TARGET} identity and the stock-placement fallback returns the balance
     * to the child's own {@code BUS_PHI} row — so for that scenario the participant's row is the
     * child's, and the distribution must read its anchor and write its share there. Writing to the
     * pinned row instead loses the share AND feeds the retain/water-fill totals a spurious
     * {@code (0 - anchor)} of absorbed power, which makes the slack loop redistribute what OLF placed
     * once.
     */
    public static native void setBatchDsRowOverrides(long handle, int[] slotParticipantRowTriples);

    /**
     * Install (or clear, with null) the per-scenario reactive-limits controllers a fragmenting
     * contingency strands in another synchronous component, as {@code {slot, controller}} pairs.
     *
     * <p>OLF disables those buses, so its ReactiveLimits loop never sees them. The device holds their
     * rows frozen at a flat 1&#8736;0, so the q it would read is a placeholder — switching on it pins a
     * PV bus OLF leaves alone and perturbs the main component through the controller's group.
     */
    public static native void setBatchRlIslanded(long handle, int[] slotControllerPairs);

    /** (scenario, reactive-limit entry) pairs whose UNPIN the device cannot host (a pinned member of a re-homed
     *  cluster with no enabled controller): the kernel rejects the scenario instead of switching (CPU replay). */
    public static native void setBatchRlNoUnpin(long handle, int[] slotEntryPairs);

    /** (scenario, reactive-limit entry) pairs whose bus lost EVERY generator this scenario: no controller there
     *  (OLF has no voltage control at the bus), so the entry is seeded FROZEN — skipped by the limit check, read
     *  as "not enabled" by the pilot-self layout, and its reactive balance emitted through the pilot-self gate. */
    public static native void setBatchRlGenLost(long handle, int[] slotEntryPairs);

    /** Re-homed voltage controls (ControlReHoming.reHomedControls): per (slot, entry) pair the entry is
     *  checked by the reactive-limits pass as a LOCAL controller at {@code vTargets[k]} — the bus is a plain PV
     *  bus this scenario, whatever its base packing. Null clears the context's list. */
    public static native void setBatchRlLocal(long handle, int[] slotEntryPairs, double[] vTargets);
    /** Remote controllers OLF re-merges onto a NEW pilot after a zero-impedance split (GpuBatchQualification
     *  .reHomedControls): per (slot, entry) pair the V row the entry's reactive-limit check compares against the
     *  group target instead of the base pilot's. The target itself rides the per-scenario V-target overrides of
     *  the solve. Null clears the context's list. */
    public static native void setBatchRlCmpOverrides(long handle, int[] slotEntryPairs, int[] cmpVRows);

    /** Per-scenario member targetQ overrides for the DISTR_Q target kernel, as {@code key = slot *
     *  nDqMembers + memberIndex} with the member's post-contingency targetQ. Needed because that kernel
     *  RECOMPUTES the group's row targets from the base member targetQ on every reactive-limits pass, so
     *  a host-side row seed alone is put straight back. Nulls clear. */
    public static native void setBatchDqMemberTargetQOverrides(long handle, int[] keys, double[] values);

    /** Per-scenario moves of a remote-control pilot's own reactive target ({key = slot * nQr + group, delta pu}, one
     *  entry per key): added to the value captured from the base carrier and to the first fill's carrier target.
     *  Nulls clear. */
    public static native void setBatchPilotQDeltas(long handle, int[] keys, double[] deltas);

    /** Per slot, the row the distributed-slack kernels read the slack's balance on: the angle row of the bus OLF
     *  re-picks as the slack when a contingency isolates the base one, -1 = the base slack row. Null clears. */
    public static native void setBatchDsSlackRows(long handle, int[] rows);

    /** Pilot-row flip entries: one per (scenario, pilot) where the outage made the pilot stop being one
     *  of its OWN controllers. While the survivors hold its voltage the pilot keeps BUS_TARGET_V and its
     *  acquired reactive balance rides one of THEIR rows; when every survivor pins, those rows take their
     *  own pinned balances and the pilot's is evicted. The kernel then puts it on the pilot's freed row,
     *  as OLF does. {@code rlCounts} sizes each entry's slice of {@code rlPool} (the surviving
     *  controllers' reactive-limit entries). Nulls clear. */
    public static native void setBatchPilotAllPinnedFlip(long handle, int[] slots, int[] rows,
                                                         double[] qTargets, int[] rlCounts, int[] rlPool);

    /**
     * Install the HVDC AC-emulation outer loops' tables and their per-scenario starting state, so
     * FreezingHvdcACEmulation and AcHvdcAcEmulationLimits run as DRIVER PHASES in OLF's order —
     * between DistributedSlack and ReactiveLimits — instead of as host orchestrators after the whole
     * solve. Pass a null/empty {@code elem} to leave them to the host orchestrator.
     *
     * @param in per hvdc x9: {@code p0 k pMax12 pMax21 loss1 loss2 r baseD1 baseD2}
     * @param ctgOut per (scenario, hvdc): the contingency itself outaged it, so the loops leave it alone
     * @param status per (scenario, hvdc) starting status, and {@code appliedD1/D2} the converter-row
     *        deltas the pre-solve freeze already applied — transitions are applied incrementally
     */
    public static native void setBatchHvdcLoops(long handle, int[] elem, int[] bus1Row, int[] bus2Row,
                                                int[] bus1Phi, int[] bus2Phi,
                                                double[] in, byte[] ctgOut, int[] status,
                                                double[] appliedD1, double[] appliedD2, int limitsActive);

    /**
     * MonitoringVoltage controller table (OLF outer-loop order 6) for the DRIVER phase: per
     * controller its monitored V row and {@code band = {lowThreshold, highThreshold, lowTargetV,
     * highTargetV}}, plus the CSR list ({@code deltaPtr}) of (row, mode, target) writes its
     * activation applies. A null/empty {@code vRow} clears the table.
     */
    public static native void setBatchMonitoringLoops(long handle, int[] vRow, double[] band,
                                                      int[] deltaPtr, int[] deltaRow,
                                                      int[] deltaMode, double[] deltaTarget);

    /** Enable HOST-owned outer-loop slots via a bitmask.
     *  Bit 0 = SecondaryVoltageControl (OLF order 4, pre-RL),
     *  Bit 1 = PhaseControl (OLF order 7, post-RL),
     *  Bit 2 = TransformerVoltageControl (OLF order 8),
     *  Bit 3 = TransformerReactivePowerControl (OLF order 9),
     *  Bit 4 = ShuntVoltageControl (OLF order 10).
     *  0 disables all; the driver behaves exactly as before. */
    public static native void setBatchHostLoop(long handle, int mask);

    /** Host loop kind constants matching the native {@code DeviceLoopKind} enum. */
    public static final int HOST_LOOP_SVC = 5;
    public static final int HOST_LOOP_PHASE = 6;
    public static final int HOST_LOOP_TVC = 7;
    public static final int HOST_LOOP_TRPC = 8;
    public static final int HOST_LOOP_SHUNT = 9;

    /** Upload incremental TVC controller data for the in-kernel outer loop.
     *  rhoRow/vRow/elemIndex: per-controller int arrays; targetV/halfDb: per-controller doubles;
     *  tapPtr: CSR pointers [nCtrl+1] into tapData; tapData: flat [totalTaps*8] (y,g1,g2,b1,b2,ksi,r1,a1);
     *  baseTap: per-controller initial tap index; groupPtr/groupCtrl: CSR grouping of controllers
     *  sharing a controlled bus (for the one-tap-at-a-time round-robin). */
    /** Returns 1 when the in-kernel TVC loop owns these controllers, 0 when it refused them (a
     *  controlled bus shared by several transformers stays on the host orchestrator). */
    public static native int setBatchTvcControllers(long handle,
            int[] rhoRow, int[] vRow, double[] targetV, double[] halfDb, int[] elemIndex,
            int[] tapPtr, double[] tapData, int[] baseTap, int[] groupPtr, int[] groupCtrl,
            int maxTapShift);

    /** Host loop enable mask bits — use with {@link #setBatchHostLoop}. */
    public static final int HOST_MASK_SVC = 1 << 0;
    public static final int HOST_MASK_PHASE = 1 << 1;
    public static final int HOST_MASK_TVC = 1 << 2;
    public static final int HOST_MASK_TRPC = 1 << 3;
    public static final int HOST_MASK_SHUNT = 1 << 4;

    /** Did the last solve stop at a host-owned slot? Returns 0 if not paused, or the
     *  {@code HOST_LOOP_*} constant identifying WHICH host loop was reached. */
    public static native int readBatchDriverPaused(long handle);

    /** Resume the paused driver; {@code hostChanged[s] != 0} is the host loop's UNSTABLE verdict. */
    public static native void setBatchDriverResume(long handle, byte[] hostChanged);

    /** Drop a paused driver sequence without resuming it (the host loop changed the equation layout
     *  or the row modes): the next solveBatchContext starts over on the re-extracted system. */
    public static native void cancelBatchDriverPause(long handle);

    /** The next fresh solve resumes its sweep at this LOOP_HOST_* kind after its first solve — the
     *  restarted sequence re-checks the loop that went UNSTABLE, as OLF's runOuterLoop does. */
    public static native void setBatchDriverStartLoop(long handle, int kind);

    /** Absolute target writes at flat {@code (scenario * n + row)} positions, for a host-owned loop. */
    public static native void setBatchTargetDeltas(long handle, int[] rows, double[] values);

    /** How many (controller, scenario) monitoring automatons fired in the last solve. */
    public static native int readBatchMonitoringActivations(long handle);

    /** The HVDC statuses after a solve, {@code nScenarios * nHvdc}, or null when none are installed. */
    public static native int[] readBatchHvdcStatus(long handle);

    /** Per (scenario, monitoring controller): the target the driver's MonitoringVoltage phase activated
     *  (pu), NaN while standing by — null without the table. Layout {@code s * nmon + c}. */
    public static native double[] readBatchMonitoringTargets(long handle);

    /** The zone-regulated generator voltage control targets the SecondaryVoltageControl phase settled
     *  on, per {@code (scenario, controller)} — what OLF's own loop leaves on the network. */
    public static native double[] readBatchSecondaryVcTargets(long handle);

    /** Per (scenario, incremental TVC controller): the tap index the driver's TransformerVoltageControl
     *  phase left — null without the table. Layout {@code s * nTvc + c}. */
    public static native int[] readBatchTvcTaps(long handle);

    /** Install the incremental SHUNT voltage-control tables for the in-kernel loop (OLF order 10);
     *  see GpuBatchIncrementalShuntVc.uploadToDevice. Returns 1 when the kernel loop owns them. */
    public static native int setBatchShuntControllers(long handle,
            int[] bRow, int[] vRow, double[] targetV, double[] halfDb, int[] elemAbs, int[] shOvElem,
            int[] groupPtr, int[] groupCtrl, int[] subPtr, int[] secPtr, double[] secB, double[] secG,
            int[] subMin, int[] subBase, int maxSectionShift);

    /** Per (scenario, sub-controller): the section the driver's ShuntVoltageControl phase left —
     *  null without the table. Layout {@code s * nSubs + u}, subs in upload order. */
    public static native int[] readBatchShuntPositions(long handle);

    /** Install the FLOW-OBSERVED incremental controllers (RPC / incremental phase) for the in-kernel
     *  loops (OLF orders 7 and 9); see GpuBatchIncrementalFlowControl.uploadToDevice. Returns 1 when
     *  the kernel loops own them. */
    public static native int setBatchFlowControllers(long handle, int[] kind, int[] pinRow, int[] pinCol, int[] side,
            double[] targetValue, double[] halfDb, int[] elem, int[] ctlElem, int[] tapPtr, double[] tapData,
            int[] baseTap, int[] maxShift);

    /** Per (scenario, flow controller): the tap the driver left — null without the table. */
    public static native int[] readBatchFlowTaps(long handle);

    /** Install the CONTINUOUS phase controllers for the in-kernel PhaseControl phase (OLF order 7);
     *  see GpuBatchContinuousPhase.uploadToDevice. Returns 1 when the kernel loop owns them. */
    public static native int setBatchContinuousPhase(long handle, int[] kind, int[] a1Row, int[] side, double[] limit,
            int[] elem, int[] tapPtr, double[] tapData, int[] baseTap);

    /** Per (scenario, continuous phase controller): the tap the driver left — null without the table. */
    public static native int[] readBatchContinuousPhaseTaps(long handle);

    /** Install the DISCRETE transformer voltage-control tables for the in-kernel TransformerVoltageControl
     *  phase (OLF order 8): {@code mode} 0 = AFTER_GENERATOR (TransformerVoltageControlOuterLoop), 1 =
     *  WITH_GENERATOR (SimpleTransformerVoltageControlOuterLoop); {@code entryOn} = the uploaded activation
     *  already carries the control ON (a base load flow whose network OLF's initialize switched on). Groups
     *  are controlled buses (CSR over members) with their toggle-diffed ON/OFF row deltas (CSR); see
     *  GpuBatchTransformerControl.uploadToDevice. Returns 1 when the kernel loop owns them. */
    @SuppressWarnings("java:S107")
    public static native int setBatchDiscreteTvc(long handle, int mode, int entryOn,
            int[] grpPtr, int[] grpVRow, double[] grpTargetV, double[] grpHalfDb, int[] grpUseInit,
            int[] deltaPtr, int[] deltaRow, int[] deltaModeOn, int[] deltaModeOff, double[] deltaTargetOn, double[] deltaTargetOff,
            int[] memRhoRow, int[] memElem, int[] memBaseTap, int[] memTapPtr, double[] memMin, double[] memMax, double[] memInit,
            double[] tapData);

    /** Per (scenario, member): {state (0 off, 1 continuous, 2 fixed), tap index, the r1 fed to the rounding},
     *  layout {@code (s * nMembers + m) * 3} — null without the table. */
    public static native double[] readBatchDiscreteTvcState(long handle);

    /** GeneratorVoltageControlManager tables for the AFTER_GENERATOR phase (after setBatchDiscreteTvc): the
     *  freeze candidates (controller-bus V row, load Q, and the reactive-limit entry of a REMOTE
     *  controller or −1) and the UNIVERSE of
     *  fixTransformerVoltageControls — the {@code BUS_V} row of every generator-voltage-controlled bus,
     *  which the per-scenario masks index. Returns 1 when installed. */
    public static native int setBatchDiscreteTvcFreeze(long handle, int[] fzVRow, double[] fzLoadQ, int[] fzRlEntry,
                                                      int[] fixPvRow);

    /** fixTransformerVoltageControls' per-(scenario, member) connectivity masks over the
     *  {@code fixPvRow} universe, {@code words} ints each: {@code minMask} cuts every controller branch
     *  (a subset of OLF's component), {@code maxMask} cuts none (a superset). A PV bus in min means keep
     *  the control, no PV bus in max means disable it, anything between routes the scenario to the CPU.
     *  Null clears the channel. */
    public static native void setBatchDiscreteTvcFixMasks(long handle, int[] minMask, int[] maxMask, int words);

    /** Per (scenario, freeze candidate): {frozen (0/1), the row target while frozen (generation − load Q)},
     *  layout {@code (s * nFz + c) * 2} — null without candidates. */
    public static native double[] readBatchDiscreteTvcFreeze(long handle);

    /** Install the WITH_GENERATOR shunt voltage-control tables for the in-kernel ShuntVoltageControl phase
     *  (OLF order 10); see GpuBatchSimpleShuntVc.uploadToDevice. Returns 1 when the kernel loop owns them. */
    @SuppressWarnings("java:S107")
    public static native int setBatchDiscreteShunt(long handle, int entryOn,
            int[] grpPtr, int[] deltaPtr, int[] deltaRow, int[] deltaModeOn, int[] deltaModeOff,
            double[] deltaTargetOn, double[] deltaTargetOff,
            int[] memBRow, int[] memElem, int[] memSubPtr, int[] secPtr, double[] secB, int[] subMin, int[] subBase);

    /** Per (scenario, member shunt): {state (0 off, 1 continuous, 2 dispatched), the continuous b the
     *  dispatch started from}, layout {@code (s * nMembers + m) * 2} — null without the table. */
    public static native double[] readBatchDiscreteShuntState(long handle);

    /** Per (scenario, sub-controller): the dispatched section, layout {@code s * nSubs + u}. */
    public static native int[] readBatchDiscreteShuntPositions(long handle);

    /** Un-pin reactive-limit entries at a driver pause (SecondaryVoltageControl's re-enable step):
     *  {@code pairs} is (scenario, RL entry) flattened, {@code newTargets} the voltage target each row
     *  returns to. A resume skips the RL seeding, so this is the only way a host-side un-pin reaches
     *  the device. No-op for an entry that is not pinned. */
    public static native void setBatchRlUnpin(long handle, int[] pairs, double[] newTargets);

    /** Install the SecondaryVoltageControl zones for the in-kernel phase (OLF order 4); see
     *  GpuBatchSecondaryVc.uploadToDevice. Returns 1 when the kernel loop owns them — it refuses a zone
     *  set wider than the in-kernel dense system, or an observable on an element the batch does not pack. */
    @SuppressWarnings("java:S107")
    public static native int setBatchSecondaryVc(long handle,
            int[] ctldVRow, int[] busVRow, int[] rlIdx, double[] baseTargetV,
            double[] minQ, double[] maxQ, double[] loadQ,
            int[] qPtr, int[] qKind, int[] qElem, int[] qBRow,
            int[] zonePtr, int[] zoneCtl, int[] zonePilotRow, double[] zoneTargetV,
            double minPlausibleV, double maxPlausibleV);

    /**
     * Install the PER-GENERATOR distributed-slack participants: the factors and limits OLF's
     * water-fill actually saturates on, grouped by bus ({@code busGenStart}/{@code busGenEnd} slice
     * the generator arrays, so a bus's row value is the sum over its slice).
     *
     * <p>OLF's {@code ParticipatingElement} wraps an {@code LfGenerator} and {@code step.run} clamps
     * each at its own {@code minTargetP}/{@code maxTargetP} and removes it, so a bus-level aggregate
     * reproduces OLF only while nothing saturates.
     */
    /** The slack bus's participating generators' total rooms (GpuAcData.dsSlackRoomUp/Down): the bound of a
     *  slack-only distribution, beyond which the device flags OLF's remaining mismatch. */
    public static native void setBatchDsSlackRooms(long handle, double roomUp, double roomDown);

    /** The run's linear-solver settings (GpuLoadFlowParameters.install): -1 reorder / pivot type = AUTO, NaN threshold =
     *  cuDSS's own. Read by every context built afterwards and by every solve's iterative refinement. */
    public static native void setLinearSolverSettings(int reorder, int matching, int pivotType, double pivotThreshold,
                                                      int irBase, int irMax, double irTol);

    /** The installed linear-solver settings, in {@link #setLinearSolverSettings}'s order. */
    public static native double[] readLinearSolverSettings();

    public static native void setBatchDsGenerators(long handle, int[] genBus, double[] genFactor,
                                                   double[] genMaxDelta, double[] genMinDelta,
                                                   double[] genMarginUp, double[] genMarginDown,
                                                   int[] busGenStart, int[] busGenEnd);

    /** Release a violation table (idempotent for handle 0). */
    public static native void destroyViolationTable(long tableHandle);

    /** Release a batch context from {@link #createBatchContext} (idempotent for handle 0). */
    /** Install (or clear, with nulls) the PER_EQUATION_TYPE per-row stopping thresholds on a batch
     *  context: {@code thrIdent[r]} for identity/slope rows, {@code thrPower[r]} for power rows —
     *  the convergence test then requires EVERY row's |F| under its threshold (OLF's
     *  PerEquationTypeStoppingCriteria) instead of the 2-norm/infnorm rule. Contexts are CACHED, so
     *  a run without the criterion must clear (null, null). */
    public static native void setBatchRowCriteria(long handle, double[] thrIdent, double[] thrPower);

    /** The alternate per-row threshold and its selector row (-1: none) of the PER_EQUATION_TYPE criterion (see
     *  GpuAcActivationExtractor.extractRowCriteria); nulls clear them. Call after {@link #setBatchRowCriteria}. */
    public static native void setBatchRowCriteriaAlt(long handle, double[] thrAlt, int[] thrSel);

    /** Installs the four arrays of GpuAcActivationExtractor.extractRowCriteria, or clears every threshold. */
    public static void installRowCriteria(long handle, double[][] rowCriteria) {
        setBatchRowCriteria(handle, rowCriteria != null ? rowCriteria[0] : null, rowCriteria != null ? rowCriteria[1] : null);
        if (rowCriteria != null && rowCriteria.length >= 4) {
            int[] sel = new int[rowCriteria[3].length];
            for (int i = 0; i < sel.length; i++) {
                sel[i] = (int) rowCriteria[3][i];
            }
            setBatchRowCriteriaAlt(handle, rowCriteria[2], sel);
        }
    }

    /** LINE_SEARCH backtracking parameters for a batch context (OLF LineSearchStateVectorScaling's
     *  maxIteration and stepFold) — consumed by solves called with {@code lsMode == 1}. */
    public static native void setBatchLineSearch(long handle, int lsMaxIter, double lsStepFold);

    /** Install (or clear, with nulls) the reactive-capability-CURVE channel on a batch context —
     *  per-scenario dynamic RL limits evaluated at the current targets (see GpuBatchRlCurves). */
    public static native void setBatchRlCurves(long handle, int[] cvIdx, double[] cvIn, double[] cvPts,
                                               int[] cvHv, int extrapolate);

    /** Install (or clear, with nulls) the forceTargetQInReactiveLimits channel on a batch context: the Q
     *  targets of generators OLF clamps into their curve limits at their current P (see GpuBatchForcedTargetQ). */
    public static native void setBatchForcedTargetQ(long handle, int[] fqIdx, double[] fqIn, double[] fqPts,
                                                    int extrapolate);

    /** Per-chunk loss pre-distribution moves of the forced-target-Q generators ([nS * nfq] pu), or null. */
    public static native void setBatchForcedTargetQPre(long handle, double[] fqPre);

    /** Slack-only distribution: per scenario, the slack generators' initial target relative to the slack's target P
     *  for the following solves - a strategy pair's parent total, which OLF re-anchors on; null = 0 (a contingency,
     *  whose loss pre-distribution OLF undoes at its first pass). */
    public static native void setBatchSlackOnlyAnchor(long handle, double[] anchor);

    /** Per scenario and participating generator ({@code s * ngen + g}, pu) for the following solves: {@code anchor},
     *  its move from its base target P to where the slack's water-fill starts it (a strategy pair: the parent's final
     *  target, or a GeneratorAction's value; null = the base); {@code start}, its move from that anchor when the solve
     *  starts (a contingency's loss pre-distribution; null = none). The curve and forced-target-Q channels evaluate a
     *  generator at anchor + its water-fill move once a slack pass ran, at anchor + start before. {@code seedAnchor}
     *  (strategy pairs): the anchor without the strategy's GeneratorActions, the P the pins continued from the parent
     *  are stamped at (OLF keeps their restored target Q until its reactive-limit check); null = the anchor. */
    public static native void setBatchGenAnchor(long handle, double[] anchor, double[] start, double[] seedAnchor);

    /** Remote-member RL entries (plan-B1): per-entry remote kinds + compare rows (PILOT v column for
     *  remote group members — both RL switch conditions test the pilot's V against the GROUP target).
     *  {@code dqMemberQrGroup} (plan-B4) links each group member to its pilot's relocation candidate
     *  group (−1 = none) so the kernels suppress the CURRENT carrier's DISTR_Q row and hand its
     *  target the pilot's Q when the free role re-picks. Nulls clear. */
    /** Install the voltage-dependent SVC reactive-limit coefficients on a batch context:
     *  per RL entry, limit(v) = packed base limit + svcB*(v^2 - v2Base) — exactly OLF's live
     *  Bmin/Bmax*(v*nomV)^2 evaluation (and the pinned re-freeze via busesWithUpdatedQLimits). */
    /** OLF's GLOBAL forced-PV election keys per RL entry (ReactiveLimitsOuterLoop.switchPvPq: when every
     *  PV bus would switch, the strongest stays PV — highest controlled-bus nominal V, then highest bus
     *  target P, then lowest id): the nominal V, the bus's BUS_TARGET_P row (−1 = none, use staticP)
     *  and the id rank. Installed on every batch context; the RL pass then runs the election in-kernel. */
    public static native void setBatchRlElection(long handle, double[] nominalV, int[] pRow, double[] staticTargetP, int[] idRank);

    public static native void setBatchRlSvcLimits(long handle, double[] svcBmin, double[] svcBmax,
                                                  double[] v2Base);

    /** {@code voltageRemoteControlRobustMode} (OLF's default): per RL entry, the initial generation
     *  target Q a realistic-V freeze pins at, plus the two thresholds the controller's OWN voltage is
     *  tested against ({@code minRealisticVoltage * 1.02} and {@code maxRealisticVoltage / 1.02}).
     *  {@code robust = 0} clears the channel on a cached context. */
    public static native void setBatchRlRobust(long handle, double[] rlInitTargetQ, double vLo, double vHi, int robust);

    /** {@code AcloadFlowEngine.isStateUnrealistic} on the driver's INTERMEDIATE states: per row, 1 = a
     *  non-fictitious bus's {@code BUS_V} at or above {@code minNominalVoltageRealisticVoltageCheck},
     *  plus the realistic band. A scenario that ever leaves the band goes to the CPU replay.
     *  {@code enabled = 0} clears the channel on a cached context. */
    public static native void setBatchRealisticCheck(long handle, byte[] rowChecked, double minV, double maxV,
                                                     int enabled, int fixingLoop);

    /** OLF's own base pin kind per RL entry (0 none, 1 MIN_Q, 2 MAX_Q, 3 MIN_REALISTIC_V,
     *  4 MAX_REALISTIC_V); null clears it on a cached context and the device falls back to its
     *  nearest-limit guess. */
    public static native void setBatchRlBaseLimitType(long handle, byte[] rlBaseQLimitType);

    public static native void setBatchRlRemote(long handle, byte[] rlRemote, int[] rlCmpVrow,
                                               int[] rmqRow, int[] rmqElem, int[] rmqSide, int[] rmqKind,
                                               double[] rmqWeight,
                                               int[] dqKMember, int[] dqJMember, double[] dqMemberKey,
                                               int[] dqMemberRlEntry, int[] dqMemberGroupStart,
                                               int[] dqMemberGroupCount, int[] dqMemberRow, byte[] dqMemberHasRow,
                                               int[] dqMemberQrGroup, double[] dqMemberTargetQ,
                                               int[] dqMemberTgtPtr, int[] dqMemberTgtPeer);

    /** Install (or clear, with nulls) per-scenario DISTR_Q weight overrides on a batch context —
     *  {@code idx} = flat {scenario, contribution} pairs, {@code w} = the renormalized weight (a gen
     *  outage at a remote-VC controller bus re-runs updateReactiveKeys on the CPU). Per-chunk state. */
    /** {@code {slot, dqMember}} pairs: DISTR_Q controllers a scenario DEACTIVATES (their bus lost its
     *  last voltage generator while the group survives). Context state, not a solve parameter — a
     *  monitoring or secondary-VC re-solve on the same context must not silently drop it, which is
     *  exactly what happened while this was a parameter: the re-solve passed null and
     *  updateDistrQTargetsBatch recomputed the DISTR_Q target with the dead controller's key back in
     *  the denominator, clobbering the right target with the base one. Cleared when the chunk has none. */
    public static native void setBatchDqInactiveMembers(long handle, int[] slotMember);

    /** tasks #71: per (scenario, dq member) SUBGROUP labels of a group a contingency split behind its surviving
     *  main (dense {@code chunkSize x nDqMembers} bytes, 0 = the base pilot's group) and the qr group -> first
     *  dq member map the kernels use to pair candidates with members. Null clears the channel. */
    public static native void setBatchDqSubgroups(long handle, byte[] dense, int[] qrMemberStart);

    /** The generators an outage REMOVES, as (scenario, DS generator slot) pairs. With the identity the
     *  outaged bus's survivors take part in the per-generator slack water-fill like any others, instead
     *  of the bus riding on one aggregate entry — which leaves their own P unknown to everything that
     *  reads it, the reactive-capability curve channel above all. */
    public static native void setBatchDsLostGenerators(long handle, int[] slotGen);

    /** The participants a strategy's GeneratorAction takes out of the slack distribution (OLF's
     *  reApplyActivePowerControlChecks), as (scenario, DS generator slot) pairs: they keep the action's target and
     *  take no share. Null clears the channel. */
    public static native void setBatchDsOutGenerators(long handle, int[] slotGen);

    /** Per (scenario, reactive-limit entry): the pinned target a continued solve restores (a strategy pair takes its
     *  parent's, as OLF restores the parent's target Q), NaN = stamp the pin at its limit. Null clears it. */
    public static native void setBatchRlPinTargets(long handle, double[] targets);

    /** Per (scenario, reactive-limit entry): the pinned target the last solve ended with, NaN when not pinned. */
    public static native double[] readBatchRlPinTargets(long handle);

    /** Rows an islanding outage froze to a flat identity state, as (scenario, row) pairs. Their targets
     *  are not the outer loops' to write — OLF removes an islanded bus's equations, so nothing there
     *  re-derives a target for them. */
    public static native void setBatchFrozenRows(long handle, int[] slotRow);

    public static native void setBatchDqWeightOverrides(long handle, int[] idx, double[] w);

    /** Per-scenario member reactive keys: {@code idx} = flat {scenario, member} pairs, {@code key} = OLF's
     *  qPct after the scenario's generator losses (updateReactiveKeys); the DISTR_Q fill and the target kernel
     *  renormalize them over the members a pin leaves enabled. Per-chunk state; nulls clear it. */
    public static native void setBatchDqKeyOverrides(long handle, int[] idx, double[] key);

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
                noPcSide(cbIdx.length / 12), new double[0], new int[0], new int[] {0}, new int[0],
                noQAlt(obIdx.length / 6), noQAlt(ziIdx.length / 11), new int[0], new int[0], noDepRows(n));
        try {
            long handle = createBatchContext(structure, nScenarios, mode,
                    rlVrow, rlMinQ, rlMaxQ, rlLoadQ, rlVtarget, new byte[rlVrow.length], new double[x0Base.length],
                    dsPhiRow, dsFactor, dsSlackPhiRow, dsSlackTargetP,
                    dsBaseTargetP, dsMaxDelta, dsMinDelta, 0, new double[0], new double[0], 0, 0);
            try {
                // OLF's forced-PV election keys. This one-shot entry point has no network behind it
                // (the caller hands raw rows), so the keys are NEUTRAL: equal nominal voltage, no
                // live target-P row, and the entry order as the id rank — which reduces the rule to
                // OLF's last tie-break, "lowest id stays PV". A context that reaches a solve without
                // any keys is refused, and the rule can only fire at all when every PV bus is
                // pinnable, so this keeps the low-level entry usable without inventing a ranking.
                int[] idRank = new int[rlVrow.length];
                int[] pRow = new int[rlVrow.length];
                for (int i = 0; i < idRank.length; i++) {
                    idRank[i] = i;
                    pRow[i] = -1;
                }
                setBatchRlElection(handle, new double[rlVrow.length], pRow, new double[rlVrow.length], idRank);
                // one-shot: no per-scenario gen-override flips, so any scenario's row mode IS the base row mode
                return solveBatchContext(handle, x0Base, targetBase, rowMode, rowMode[0], disabledElementMask,
                        new int[0], new double[0], new int[0], new double[0], new double[0], new double[0], new double[0], null, null, null, null, null, null, null, null,
                        maxIter, tol, reactiveLimitTol, slackDistTol, 0.0, 0, 0.0, 0.0, null, null, null, null, null, null, null, null, 0,
                        com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop.MAX_SWITCH_PQ_PV_DEFAULT_VALUE,   // OLF's cap: 0 means never switch back
                        0, 0, 0, null, null);
            } finally {
                destroyBatchContext(handle);
            }
        } finally {
            destroyStructure(structure);
        }
    }
}
