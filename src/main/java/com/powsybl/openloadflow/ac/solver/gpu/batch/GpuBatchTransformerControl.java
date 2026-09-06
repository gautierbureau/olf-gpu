/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * The per-scenario transformer voltage-control orchestration for the batched N-1 path: the stage-1
 * scope check, the controller pack built by toggling each control on the shared equation system and
 * diffing the activation, the discrete tap-table snapshot (reflection on PiModelArray), and the
 * Java orchestrator that ports TransformerVoltageControlOuterLoop (AFTER_GENERATOR mode) as repeated
 * batched device solves over one chunk (INITIAL deadband enable, CONTROL clamp, round-to-tap, final).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchTransformerControl {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchTransformerControl.class);

    private GpuBatchTransformerControl() {
    }

    /** One member transformer of a (possibly shared) voltage control: rho row (== r1 col,
     *  self-paired), continuous r1 range, discrete tap r1 values, element index (per-scenario
     *  outage check), and the row-mode/target deltas that turn THIS member's control OFF while the
     *  rest of the group stays ON (a clamp: OLF's roundR1ToExtremeTapPosition disables the member;
     *  coefficients do NOT renormalize — the DISTR_RHO cross terms to a control-disabled member
     *  stay). {@code clampNeedsCpu}: clamping this member re-picks the group's FREE role (its
     *  off-delta touches other members' rows) — per-scenario CPU route, out of stage-1 scope. */
    record Member(int rhoRow, double minR1, double maxR1, double[] taps, double[][] tapRows, int elemIndex,
                  int[] offDeltaRows, int[] offDeltaModes, double[] offDeltaTargets, boolean clampNeedsCpu) {
        double closestTap(double r1) {
            if (taps.length == 0) {
                return r1;                                   // no tap table: keep the continuous ratio
            }
            double best = taps[0];
            for (double t : taps) {
                if (Math.abs(t - r1) < Math.abs(best - r1)) {
                    best = t;
                }
            }
            return best;
        }

        /** The FULL 8-double pi row of the tap whose r1 is closest to {@code r1}, or null without a
         *  table. Rounding a tap on the CPU switches the branch's per-step R/X/G/B, not just rho —
         *  pinning rho alone left the base impedance in the fills (a DIFFERENT model: 2.4e-4 kV on a
         *  non-base tap, found via the shared-group member-outage test). */
        double[] closestTapRow(double r1) {
            if (tapRows.length == 0) {
                return null;
            }
            double[] best = tapRows[0];
            for (double[] t : tapRows) {
                if (Math.abs(t[6] - r1) < Math.abs(best[6] - r1)) {
                    best = t;
                }
            }
            return best;
        }
    }

    /** One voltage CONTROL (the group), precomputed for the per-scenario orchestrator:
     *  controlled-bus V col, target/deadband, the row-mode/target DELTAS between control-off (base)
     *  and ALL-members-on activations (plan-group-member-dynamics A3 — captured by toggling the
     *  WHOLE group, never one member alone), and the members. */
    record TransformerController(int vCol, double targetV, double halfDeadband,
                                 int[] deltaRows, int[] deltaModesOn, double[] deltaTargetsOn,
                                 Member[] members) {
    }

    /** Stage-1 scope check: the CPU loop's GeneratorVoltageControlManager disables generator
     *  voltage controls at buses under the max transformer-controlled nominal voltage. Porting
     *  the frozen-Q flip is stage 2 — until then a network that would NEED it must stay on the
     *  CPU path (the provider's guard calls this). */
    /** True when any transformer voltage control has more than one merged controller (a SHARED
     *  group, DISTR_RHO coupling). The AFTER_GENERATOR/WITH_GENERATOR orchestrators handle groups
     *  since plan-group-member-dynamics A3 (group-level toggle-diff controllers); INCREMENTAL and
     *  the useInitialTapPosition shared-ratio manager do not — the scope gates those combinations. */
    public static boolean sharedGroupPresent(LfNetwork network) {
        return network.getControlledBuses(com.powsybl.openloadflow.network.VoltageControl.Type.TRANSFORMER).stream()
                .map(b -> b.getTransformerVoltageControl().orElse(null))
                .filter(java.util.Objects::nonNull)
                .anyMatch(vc -> vc.getMergedControllerElements().size() > 1);
    }

    public static boolean transformerControlSupported(LfNetwork network) {
        double maxControlledNominalV = network.<LfBranch>getControllerElements(com.powsybl.openloadflow.network.VoltageControl.Type.TRANSFORMER).stream()
                .flatMap(b -> b.getVoltageControl().stream())
                .map(vc -> vc.getControlledBus().getNominalV())
                .mapToDouble(Double::doubleValue).max().orElse(-1);
        if (maxControlledNominalV < 0) {
            return true;                                     // no transformer controls at all: loop is a no-op
        }
        boolean genUnderThreshold = network.getControlledBuses(com.powsybl.openloadflow.network.VoltageControl.Type.GENERATOR).stream()
                .anyMatch(bus -> bus.getNominalV() <= maxControlledNominalV);
        if (genUnderThreshold) {
            LOGGER.info("Transformer control orchestration out of stage-1 scope: generator voltage controls "
                    + "exist under the transformer-controlled nominal voltage ({} kV) — CPU fallback", maxControlledNominalV);
        }
        return !genUnderThreshold;
    }

    /** Build the orchestrator's controller pack by TOGGLING each control on the shared equation
     *  system and diffing the activation — the ON recipe is exactly what the engine itself would
     *  produce, no hand-crafted modes. The network is restored before returning. */
    static List<TransformerController> buildTransformerControllers(
            LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
            TargetVector<AcVariableType, AcEquationType> targetVector,
            GpuAcActivation baseActivation, double[] baseTargets, GpuBatchElementMapper elementMapper) {
        List<TransformerController> out = new ArrayList<>();
        java.util.Map<Integer, Integer> busVCols = new java.util.HashMap<>();
        java.util.Map<Integer, Integer> branchR1Cols = new java.util.HashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVCols.put(v.getElementNum(), v.getRow());
            } else if (v.getType() == AcVariableType.BRANCH_RHO1) {
                branchR1Cols.put(v.getElementNum(), v.getRow());
            }
        }
        for (LfBus controlled : network.getControlledBuses(com.powsybl.openloadflow.network.VoltageControl.Type.TRANSFORMER)) {
            var vc = controlled.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == com.powsybl.openloadflow.network.VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (vc == null) {
                continue;
            }
            List<LfBranch> memberBranches = vc.getMergedControllerElements().stream()
                    .filter(LfBranch::isConnectedAtBothSides)
                    .filter(b -> !b.isDisabled())
                    .toList();
            Integer vCol = busVCols.get(controlled.getNum());
            boolean representable = vCol != null && !memberBranches.isEmpty()
                    && memberBranches.stream().allMatch(b -> branchR1Cols.get(b.getNum()) != null
                            && elementMapper.getElementIndex(b) >= 0);
            if (!representable) {
                LOGGER.info("Transformer control of {} not representable on the GPU batch — skipped (results follow the base taps)",
                        controlled.getId());
                continue;
            }
            double halfDeadband = vc.getTargetDeadband().orElse(0.1 / controlled.getNominalV()) / 2;
            boolean[] wasEnabled = new boolean[memberBranches.size()];
            for (int m = 0; m < memberBranches.size(); m++) {
                wasEnabled[m] = memberBranches.get(m).isVoltageControlEnabled();
                memberBranches.get(m).setVoltageControlEnabled(true);    // the WHOLE group ON (A3)
            }
            GpuAcActivation onAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] onTargets = onAct.applyTargets(targetVector.getArray());
            List<Integer> dRows = new ArrayList<>();
            for (int r = 0; r < baseTargets.length; r++) {
                if (onAct.rowMode()[r] != baseActivation.rowMode()[r] || onTargets[r] != baseTargets[r]) {
                    dRows.add(r);
                }
            }
            int[] deltaRows = dRows.stream().mapToInt(Integer::intValue).toArray();
            int[] deltaModesOn = new int[deltaRows.length];
            double[] deltaTargetsOn = new double[deltaRows.length];
            for (int k = 0; k < deltaRows.length; k++) {
                deltaModesOn[k] = onAct.rowMode()[deltaRows[k]];
                deltaTargetsOn[k] = onTargets[deltaRows[k]];
            }
            Member[] members = new Member[memberBranches.size()];
            for (int m = 0; m < memberBranches.size(); m++) {
                LfBranch branch = memberBranches.get(m);
                // this member's control OFF while the rest stays ON — the CLAMP activation (OLF's
                // roundR1ToExtremeTapPosition disables the member; the group keeps distributing)
                branch.setVoltageControlEnabled(false);
                GpuAcActivation mOff = GpuAcActivation.fromEquationSystem(network, es);
                double[] mOffTargets = mOff.applyTargets(targetVector.getArray());
                branch.setVoltageControlEnabled(true);
                List<Integer> oRows = new ArrayList<>();
                for (int r = 0; r < baseTargets.length; r++) {
                    if (mOff.rowMode()[r] != onAct.rowMode()[r] || mOffTargets[r] != onTargets[r]) {
                        oRows.add(r);
                    }
                }
                int rhoRow = branchR1Cols.get(branch.getNum());
                // a clamp that only flips THIS member's rho row is expressible; anything wider means
                // the FREE role re-picked (or a single-member group turning fully off, which reverts
                // the whole group delta — also expressible)
                boolean cpuOnly = memberBranches.size() > 1
                        && oRows.stream().anyMatch(r -> r != rhoRow);
                int[] offDeltaRows = oRows.stream().mapToInt(Integer::intValue).toArray();
                int[] offDeltaModes = new int[offDeltaRows.length];
                double[] offDeltaTargets = new double[offDeltaRows.length];
                for (int k = 0; k < offDeltaRows.length; k++) {
                    offDeltaModes[k] = mOff.rowMode()[offDeltaRows[k]];
                    offDeltaTargets[k] = mOffTargets[offDeltaRows[k]];
                }
                members[m] = new Member(rhoRow, branch.getPiModel().getMinR1(), branch.getPiModel().getMaxR1(),
                        tapR1Values(branch.getPiModel()), tapModels(branch.getPiModel()),
                        elementMapper.getElementIndex(branch),
                        offDeltaRows, offDeltaModes, offDeltaTargets, cpuOnly);
            }
            for (int m = 0; m < memberBranches.size(); m++) {
                memberBranches.get(m).setVoltageControlEnabled(wasEnabled[m]);
            }
            out.add(new TransformerController(vCol, vc.getTargetValue(), halfDeadband,
                    deltaRows, deltaModesOn, deltaTargetsOn, members));
        }
        return out;
    }

    /** Discrete tap r1 values, snapshotted read-only (reflection on PiModelArray.models; an
     *  inaccessible or non-array model yields an empty table = keep the continuous ratio). */
    public static double[] tapR1Values(Object piModel) {
        double[][] models = tapModels(piModel);
        double[] taps = new double[models.length];
        for (int i = 0; i < models.length; i++) {
            taps[i] = models[i][6];
        }
        return taps;
    }

    /** Full per-tap static branch rows, snapshotted read-only via reflection on
     *  PiModelArray.models: one row [y, g1, g2, b1, b2, ksi, r1, a1] per tap position — the
     *  EXACT cbIn packing, so a tap/PST action's override is the target row verbatim (empty
     *  when not a tap array or inaccessible). */
    public static double[][] tapModels(Object piModel) {
        try {
            if (!"PiModelArray".equals(piModel.getClass().getSimpleName())) {
                return new double[0][];
            }
            java.lang.reflect.Field f = piModel.getClass().getDeclaredField("models");
            f.setAccessible(true);
            List<?> models = (List<?>) f.get(piModel);
            String[] getters = {"getY", "getG1", "getG2", "getB1", "getB2", "getKsi", "getR1", "getA1"};
            double[][] taps = new double[models.size()][getters.length];
            for (int i = 0; i < models.size(); i++) {
                for (int g = 0; g < getters.length; g++) {
                    taps[i][g] = (double) models.get(i).getClass().getMethod(getters[g]).invoke(models.get(i));
                }
            }
            return taps;
        } catch (ReflectiveOperationException | RuntimeException e) {
            LOGGER.info("Transformer tap table not accessible ({}) — continuous ratios will not be rounded", e.toString());
            return new double[0][];
        }
    }

    /**
     * Per-scenario port of TransformerVoltageControlOuterLoop (AFTER_GENERATOR mode), run as a
     * Java orchestrator around repeated batched solves on THIS chunk, mirroring the CPU steps:
     * INITIAL (deadband check on the solved states -> enable out-of-band controls via the
     * activation-diffed row/target deltas), CONTROL (clamp continuous r1 outside [minR1,maxR1]
     * to the extreme and disable, iterate), then round to the closest discrete tap, final solve.
     * Scenarios re-solve WARM from their previous states (x0PerScenario). Returns the merged
     * chunk results (unaffected slots keep the incoming ones). Stage-1 scope: no generator
     * voltage controls under the transformer-controlled level (checked by the provider guard).
     */
    static double[][] orchestrateTransformerControl(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<TransformerController> tcs,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            int[] targetOverrideIndex, double[] targetOverrideValue,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            int[] dsPhiRow, double[] dsFactor,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        // per-member flat indexing across all controls: memberBase[c] + m
        int[] memberBase = new int[tcs.size()];
        int totalMembers = 0;
        for (int c = 0; c < tcs.size(); c++) {
            memberBase[c] = totalMembers;
            totalMembers += tcs.get(c).members().length;
        }
        // INITIAL: which (slot, CONTROL) pairs are out of deadband on the converged first solve?
        List<int[]> enabled = new ArrayList<>();             // {slot, controlIdx}
        for (int slot = 0; slot < realCount; slot++) {
            double[] r = chunkResults[slot];
            if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                continue;
            }
            for (int c = 0; c < tcs.size(); c++) {
                TransformerController tc = tcs.get(c);
                if (disabledMask[slot][tc.members()[0].elemIndex()]) {
                    continue;                                // FREE member outaged: guard-rejected — safety net
                }
                if (Math.abs(tc.targetV() - r[tc.vCol()]) > tc.halfDeadband()) {
                    enabled.add(new int[] {slot, c});
                }
            }
        }
        if (enabled.isEmpty()) {
            return chunkResults;
        }
        java.util.Set<Integer> affected = new java.util.TreeSet<>();
        boolean[][] on = new boolean[chunkSize][tcs.size()];
        for (int[] e : enabled) {
            affected.add(e[0]);
            on[e[0]][e[1]] = true;
        }
        LOGGER.info("Transformer control: {} out-of-deadband controls across {} scenario(s) of the chunk", enabled.size(), affected.size());

        // Base per-scenario overrides (islanding/gen-outage) are kept and extended per call.
        List<Integer> ovIdx = new ArrayList<>();
        List<Double> ovVal = new ArrayList<>();
        double[] rounded = new double[chunkSize * totalMembers];   // fixed r1 once clamped/rounded (NaN = still continuous)
        java.util.Arrays.fill(rounded, Double.NaN);

        // DS target carry: OLF's targets are STATEFUL across outer-loop re-solves, but every device
        // call re-uploads baseTargets — so the slack distribution applied by PREVIOUS solves on this
        // handle must be carried into each machine step: water-fill anchors (pure absorption per
        // participant), the slack's own absorbed share, and absolute DS-row target seeds (the same
        // three coupled pieces as the strategy-leg re-anchoring in GpuBatchChunkSolver). Without it
        // the VC-on solve re-derived the whole distribution from scratch and took an extra Newton
        // step the CPU never takes (1.7e-4 kV drift on the shared-TVC member-outage scenario).
        int nds = dsPhiRow != null ? dsPhiRow.length : 0;
        double[] dsAnchorCarry = nds > 0 ? new double[chunkSize * nds] : null;
        double[] dsSlackCarry = nds > 0 ? new double[chunkSize] : null;
        double[] dsSeed = nds > 0 ? new double[chunkSize * nds] : null;
        if (nds > 0) {
            captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed);
        }
        int[][] rowMode = new int[chunkSize][];
        for (int step = 0; step < 8; step++) {               // INITIAL solve, clamp iterations, final rounded solve
            ovIdx.clear();
            ovVal.clear();
            for (int k = 0; k < targetOverrideIndex.length; k++) {
                ovIdx.add(targetOverrideIndex[k]);
                ovVal.add(targetOverrideValue[k]);
            }
            if (nds > 0) {                                   // DS-row target seeds at the carried values
                for (int slot = 0; slot < realCount; slot++) {
                    double[] r = chunkResults[slot];
                    if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                        continue;
                    }
                    for (int i = 0; i < nds; i++) {
                        int flat = slot * n + dsPhiRow[i];
                        int at = ovIdx.lastIndexOf(Integer.valueOf(flat));
                        if (at >= 0) {
                            ovVal.set(at, dsSeed[slot * nds + i]);
                        } else {
                            ovIdx.add(flat);
                            ovVal.add(dsSeed[slot * nds + i]);
                        }
                    }
                }
            }
            double[] x0Full = new double[chunkSize * n];
            List<Integer> r1OvSlotElem = new ArrayList<>();  // fixed members' FULL tap pi rows (per-step
            List<Double> r1OvVal = new ArrayList<>();        // R/X/G/B follow the rounded tap, like the CPU)
            for (int slot = 0; slot < chunkSize; slot++) {
                double[] r = slot < realCount ? chunkResults[slot] : null;
                for (int i = 0; i < n; i++) {
                    x0Full[slot * n + i] = r != null ? r[i] : baseState[i];
                }
                rowMode[slot] = rowModePerScenario[slot];
                if (!affected.contains(slot)) {
                    continue;
                }
                int[] row = rowModePerScenario[slot].clone();
                for (int c = 0; c < tcs.size(); c++) {
                    if (on[slot][c]) {
                        applyControlDeltas(slot, n, tcs.get(c), disabledMask, rounded, memberBase[c],
                                totalMembers, baseState, row, ovIdx, ovVal, r1OvSlotElem, r1OvVal);
                    }
                }
                rowMode[slot] = row;
            }
            // OLF-faithful re-solve initialization (same fix as the JAVA hybrid engine): OLF's re-solves
            // re-read every variable from the NETWORK (PreviousValueVoltageInitializer) — which equals the
            // previous state for everything EXCEPT what the outer loop just mutated. Here the only mutation is
            // a clamped/rounded ratio, so re-initialize EXACTLY that: the rho variable starts the next Newton
            // AT its pinned fixedR1 (the pi-model value the CPU would re-read). Snapping OTHER identity rows to
            // their targets would deviate from OLF — the CPU keeps a long-pinned row at its previous value
            // (target ± the stopping residual), and that difference alone leaves ~1e-6 vs the CPU's ~1e-13.
            for (int slot : affected) {
                for (int c = 0; c < tcs.size(); c++) {
                    Member[] members = tcs.get(c).members();
                    for (int m = 0; m < members.length; m++) {
                        double fixedR1 = rounded[slot * totalMembers + memberBase[c] + m];
                        if (!Double.isNaN(fixedR1)) {
                            x0Full[slot * n + members[m].rhoRow()] = fixedR1;
                        }
                    }
                }
            }
            double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                    rowMode, baseRowMode, disabledMask,
                    ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                    ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                    dsAnchorCarry, dsSlackCarry, null, null,
                    parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                    config.lsMode(), config.maxDv(), config.maxDphi(),
                    config.lsMode() == 2 ? varType : null, x0Full, null, null, null, null,
                    r1OvSlotElem.isEmpty() ? null : r1OvSlotElem.stream().mapToInt(Integer::intValue).toArray(),
                    r1OvVal.isEmpty() ? null : r1OvVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0, 0, null, null);
            for (int slot : affected) {
                chunkResults[slot] = r2[slot];
            }
            if (nds > 0) {                                   // this solve may have distributed further
                captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed);
            }
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                traceStep(step, affected, chunkResults, tcs, n, rounded, memberBase, totalMembers, disabledMask);
            }
            // CONTROL: clamp out-of-range continuous ratios; when none, ROUND the remaining ones.
            boolean anyClamped = false;
            boolean anyContinuous = false;
            for (int slot : affected) {
                double[] r = chunkResults[slot];
                if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                    continue;
                }
                for (int c = 0; c < tcs.size(); c++) {
                    TransformerController tc = tcs.get(c);
                    if (!on[slot][c]) {
                        continue;
                    }
                    Member[] members = tc.members();
                    for (int m = 0; m < members.length; m++) {
                        Member mem = members[m];
                        if (!Double.isNaN(rounded[slot * totalMembers + memberBase[c] + m])
                                || disabledMask[slot][mem.elemIndex()]) {
                            continue;                        // fixed already, or outaged (pinned at base)
                        }
                        double r1 = r[mem.rhoRow()];
                        if (r1 < mem.minR1() || r1 > mem.maxR1()) {
                            if (mem.clampNeedsCpu()) {
                                // clamping this member re-picks the group's FREE role — per-scenario
                                // CPU route (never wrong, only un-accelerated)
                                chunkResults[slot][n + 2] = GpuBatchStatus.STATUS_REJECTED;
                                continue;
                            }
                            rounded[slot * totalMembers + memberBase[c] + m] = Math.max(mem.minR1(), Math.min(mem.maxR1(), r1));
                            anyClamped = true;
                        } else {
                            anyContinuous = true;
                        }
                    }
                }
            }
            if (anyClamped) {
                continue;                                    // re-solve with the clamped ones fixed
            }
            if (anyContinuous) {                             // round ALL remaining continuous ratios, one final solve
                for (int slot : affected) {
                    double[] r = chunkResults[slot];
                    if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                        continue;
                    }
                    for (int c = 0; c < tcs.size(); c++) {
                        TransformerController tc = tcs.get(c);
                        if (!on[slot][c]) {
                            continue;
                        }
                        Member[] members = tc.members();
                        for (int m = 0; m < members.length; m++) {
                            if (Double.isNaN(rounded[slot * totalMembers + memberBase[c] + m])
                                    && !disabledMask[slot][members[m].elemIndex()]) {
                                rounded[slot * totalMembers + memberBase[c] + m] = members[m].closestTap(r[members[m].rhoRow()]);
                            }
                        }
                    }
                }
                continue;                                    // the final solve runs with every control fixed
            }
            break;                                           // everything fixed and re-solved: done
        }
        return chunkResults;
    }

    /** Applies one control's per-step row/target deltas for one scenario: the group's ON deltas while
     *  any member is still continuous (fixed/outaged members swapped to their OFF-deltas + rho pins),
     *  or plain pins at the rounded taps once every member is fixed. Fixed members also emit their
     *  rounded tap's FULL pi row as an r1 override — rounding to a non-base tap switches the CPU
     *  branch's per-step R/X/G/B, not just rho. */
    @SuppressWarnings("java:S107")
    private static void applyControlDeltas(int slot, int n, TransformerController tc,
            boolean[][] disabledMask, double[] rounded, int base, int totalMembers, double[] baseState,
            int[] row, List<Integer> ovIdx, List<Double> ovVal, List<Integer> r1OvSlotElem, List<Double> r1OvVal) {
        Member[] members = tc.members();
        boolean allFixed = true;
        for (int m = 0; m < members.length; m++) {
            allFixed &= !Double.isNaN(rounded[slot * totalMembers + base + m])
                    || disabledMask[slot][members[m].elemIndex()];
        }
        if (!allFixed) {                                     // control ACTIVE: the whole group's ON deltas
            // rows owned by a FIXED (clamped) or OUTAGED member get their own override below —
            // the same row must NEVER carry two target overrides (parallel scatter, no ordering)
            java.util.Set<Integer> ownRows = new java.util.HashSet<>();
            for (int m = 0; m < members.length; m++) {
                if (!Double.isNaN(rounded[slot * totalMembers + base + m])
                        || disabledMask[slot][members[m].elemIndex()]) {
                    ownRows.add(members[m].rhoRow());
                }
            }
            for (int k = 0; k < tc.deltaRows().length; k++) {
                if (ownRows.contains(tc.deltaRows()[k])) {
                    continue;
                }
                row[tc.deltaRows()[k]] = tc.deltaModesOn()[k];
                ovIdx.add(slot * n + tc.deltaRows()[k]);
                ovVal.add(tc.deltaTargetsOn()[k]);
            }
            for (int m = 0; m < members.length; m++) {       // clamped OR outaged members: OFF-delta
                Member mem = members[m];
                boolean outaged = disabledMask[slot][mem.elemIndex()];
                double fixedR1 = rounded[slot * totalMembers + base + m];
                if (Double.isNaN(fixedR1) && !outaged) {
                    continue;
                }
                for (int k = 0; k < mem.offDeltaRows().length; k++) {
                    row[mem.offDeltaRows()[k]] = mem.offDeltaModes()[k];
                    if (mem.offDeltaRows()[k] != mem.rhoRow()) {
                        ovIdx.add(slot * n + mem.offDeltaRows()[k]);
                        ovVal.add(mem.offDeltaTargets()[k]);
                    }
                }
                ovIdx.add(slot * n + mem.rhoRow());
                // outaged: pin at the BASE-STATE rho — matching the warm start, so the row's
                // residual is exactly 0 (the CPU REMOVES the equation; a captured stale target
                // injected a unit residual that cost an extra Newton iterate — 2.4e-4 kV drift).
                // clamped: pinned at the CLAMP value, not the base tap.
                ovVal.add(outaged ? baseState[mem.rhoRow()] : fixedR1);
                if (!outaged) {
                    emitTapRow(slot, mem, fixedR1, r1OvSlotElem, r1OvVal);
                }
            }
        } else {                                             // every member fixed: back to base rows + pins
            for (int m = 0; m < members.length; m++) {
                Member mem = members[m];
                if (disabledMask[slot][mem.elemIndex()]) {
                    continue;                                // outaged: base identity + base target already
                }
                double fixedR1 = rounded[slot * totalMembers + base + m];
                ovIdx.add(slot * n + mem.rhoRow());
                ovVal.add(fixedR1);
                emitTapRow(slot, mem, fixedR1, r1OvSlotElem, r1OvVal);
            }
        }
    }

    @SuppressWarnings("java:S106")
    private static void traceStep(int step, java.util.Set<Integer> affected, double[][] chunkResults,
            List<TransformerController> tcs, int n, double[] rounded, int[] memberBase, int totalMembers,
            boolean[][] disabledMask) {
        for (int slot : affected) {
            double[] r = chunkResults[slot];
            StringBuilder sb = new StringBuilder("TVC_TRACE step=" + step + " slot=" + slot
                    + " status=" + (r != null ? (int) r[n + 2] : -9));
            for (int c = 0; c < tcs.size(); c++) {
                for (int m = 0; m < tcs.get(c).members().length; m++) {
                    Member mem = tcs.get(c).members()[m];
                    sb.append(String.format(" m%d[r1=%.6f fx=%.6f out=%b]", m,
                            r != null ? r[mem.rhoRow()] : Double.NaN,
                            rounded[slot * totalMembers + memberBase[c] + m],
                            disabledMask[slot][mem.elemIndex()]));
                }
            }
            System.err.println(sb);
        }
    }

    private static void emitTapRow(int slot, Member mem, double fixedR1,
            List<Integer> r1OvSlotElem, List<Double> r1OvVal) {
        double[] tapRow = mem.closestTapRow(fixedR1);
        if (tapRow != null) {                                // the fixed tap's FULL pi row
            r1OvSlotElem.add(slot);
            r1OvSlotElem.add(mem.elemIndex());
            for (double d : tapRow) {
                r1OvVal.add(d);
            }
        }
    }

    /** Reads the LAST solve's final per-scenario targets off the context and derives the carried
     *  distributed-slack state: per-participant pure absorption (final minus the scenario's own
     *  effective base), the slack's own absorbed share at the scenario's effective factor sum, and
     *  the absolute DS-row targets for seeding the next call. Dead or non-converged slots carry 0. */
    @SuppressWarnings("java:S107")
    private static void captureDsCarry(long handle, int chunkSize, int realCount, int n,
            int[] dsPhiRow, double[] dsFactor, double[] baseTargets,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta,
            double[][] chunkResults, double[] anchor, double[] slack, double[] seed) {
        int nds = dsPhiRow.length;
        double fsBase = 0;
        for (double fv : dsFactor) {
            fsBase += fv;
        }
        double[] finalTargets = GpuAcNewtonSolver.readBatchTargets(handle);
        for (int s = 0; s < chunkSize; s++) {
            boolean liveSlot = s < realCount && chunkResults[s] != null
                    && (int) chunkResults[s][n + 2] == GpuBatchStatus.STATUS_CONVERGED;
            double absorbedSum = 0;
            for (int i = 0; i < nds; i++) {
                int row = dsPhiRow[i];
                double ovDelta = dsOvBus != null && dsOvBus[s] == i ? dsOvBaseDelta[s] : 0.0;
                double effBase = baseTargets[row] - ovDelta;
                double shift = liveSlot ? finalTargets[s * n + row] - effBase : 0.0;
                anchor[s * nds + i] = shift;
                absorbedSum += shift;
                seed[s * nds + i] = effBase + shift;
            }
            double fs = fsBase - (dsOvBus != null && dsOvBus[s] >= 0 ? dsOvFactorDelta[s] : 0.0);
            slack[s] = fs > 0 ? (1.0 - fs) / fs * absorbedSum : 0.0;
        }
    }
}
