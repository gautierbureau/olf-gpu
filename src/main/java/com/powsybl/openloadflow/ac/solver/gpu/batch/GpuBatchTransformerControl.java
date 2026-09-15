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
    public record Member(int rhoRow, double minR1, double maxR1, double initialR1,
                  double[] taps, double[][] tapRows, int elemIndex,
                  int[] offDeltaRows, int[] offDeltaModes, double[] offDeltaTargets, boolean clampNeedsCpu,
                  int branchNum, int baseTapIdx) {
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
    public record TransformerController(int vCol, double targetV, double halfDeadband,
                                 int[] deltaRows, int[] deltaModesOn, double[] deltaTargetsOn,
                                 Member[] members, boolean useInitialTapPosition,
                                 int[] deltaModesOff, double[] deltaTargetsOff) {
    }

    /** OLF's {@code TransformerRatioManager} SHARED behavior ({@code useInitialTapPosition=true}),
     *  per scenario over the group's SURVIVING members (the CPU builds it at the INITIAL step over
     *  the just-enabled controllers; outaged members never enable): group-AVERAGED clamp bounds and
     *  the normalized-headroom initial, plus the spread-preserving remap each member's continuous
     *  ratio goes through before rounding. A single-member group degenerates to the exact identity
     *  (initialR1 = own r1, bounds = own bounds), so applying it uniformly is behavior-neutral. */
    record SharedRatio(double minR1, double maxR1, double initialR1) {
        static SharedRatio of(Member[] members, boolean[] disabledByElem) {
            double sumMin = 0;
            double sumMax = 0;
            double a = 0;
            double b = 0;
            int count = 0;
            for (Member m : members) {
                if (disabledByElem[m.elemIndex()]) {
                    continue;
                }
                sumMin += m.minR1();
                sumMax += m.maxR1();
                if (m.maxR1() != m.minR1()) {
                    a += (m.maxR1() - m.initialR1()) / (m.maxR1() - m.minR1());
                    b += (m.initialR1() - m.minR1()) / (m.maxR1() - m.minR1());
                } else {
                    a += 1;
                    b += 1;
                }
                count++;
            }
            double avgMin = sumMin / count;
            double avgMax = sumMax / count;
            return new SharedRatio(avgMin, avgMax, (avgMin * a + avgMax * b) / count);
        }

        /** OLF's {@code updateContinuousRatio}, verbatim: map the COMMON computed ratio to this
         *  member's individual one, preserving the initial spread (two-slope piecewise linear). */
        double remap(Member m, double computedR1) {
            return computedR1 >= initialR1
                    ? m.initialR1() + (computedR1 - initialR1) * (m.maxR1() - m.initialR1()) / (maxR1 - initialR1)
                    : m.initialR1() - (initialR1 - computedR1) * (m.initialR1() - m.minR1()) / (initialR1 - minR1);
        }
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
    public static List<TransformerController> buildTransformerControllers(
            LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
            TargetVector<AcVariableType, AcEquationType> targetVector,
            GpuAcActivation baseActivation, double[] baseTargets, GpuBatchElementMapper elementMapper,
            boolean useInitialTapPosition) {
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
                memberBranches.get(m).setVoltageControlEnabled(false);   // the WHOLE group OFF: the control-off activation
            }
            GpuAcActivation offAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] offTargets = offAct.applyTargets(targetVector.getArray());
            for (int m = 0; m < memberBranches.size(); m++) {
                memberBranches.get(m).setVoltageControlEnabled(true);    // the WHOLE group ON (A3)
            }
            GpuAcActivation onAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] onTargets = onAct.applyTargets(targetVector.getArray());
            // the ON/OFF deltas are diffed against each other, not against the caller's base: a base
            // extracted with the control already ON (the WITH_GENERATOR base load flow, whose network
            // OLF's initialize switched on) yields the same rows as a control-off base
            List<Integer> dRows = new ArrayList<>();
            for (int r = 0; r < baseTargets.length; r++) {
                if (onAct.rowMode()[r] != offAct.rowMode()[r] || onTargets[r] != offTargets[r]) {
                    dRows.add(r);
                }
            }
            int[] deltaRows = dRows.stream().mapToInt(Integer::intValue).toArray();
            int[] deltaModesOn = new int[deltaRows.length];
            double[] deltaTargetsOn = new double[deltaRows.length];
            int[] deltaModesOff = new int[deltaRows.length];
            double[] deltaTargetsOff = new double[deltaRows.length];
            for (int k = 0; k < deltaRows.length; k++) {
                deltaModesOn[k] = onAct.rowMode()[deltaRows[k]];
                deltaTargetsOn[k] = onTargets[deltaRows[k]];
                deltaModesOff[k] = offAct.rowMode()[deltaRows[k]];
                deltaTargetsOff[k] = offTargets[deltaRows[k]];
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
                double[][] tapRows = tapModels(branch.getPiModel());
                int baseTap = 0;
                for (int t = 1; t < tapRows.length; t++) {   // the CURRENT tap: OLF's rounding starts there
                    if (Math.abs(tapRows[t][6] - branch.getPiModel().getR1()) < Math.abs(tapRows[baseTap][6] - branch.getPiModel().getR1())) {
                        baseTap = t;
                    }
                }
                members[m] = new Member(rhoRow, branch.getPiModel().getMinR1(), branch.getPiModel().getMaxR1(),
                        branch.getPiModel().getR1(),             // individual initial (base) ratio
                        tapR1Values(branch.getPiModel()), tapRows,
                        elementMapper.getElementIndex(branch),
                        offDeltaRows, offDeltaModes, offDeltaTargets, cpuOnly, branch.getNum(), baseTap);
            }
            for (int m = 0; m < memberBranches.size(); m++) {
                memberBranches.get(m).setVoltageControlEnabled(wasEnabled[m]);
            }
            out.add(new TransformerController(vCol, vc.getTargetValue(), halfDeadband,
                    deltaRows, deltaModesOn, deltaTargetsOn, members, useInitialTapPosition,
                    deltaModesOff, deltaTargetsOff));
        }
        return out;
    }

    /** Install the controls for the in-kernel DISCRETE TransformerVoltageControl phase (OLF order 8):
     *  {@code mode} 0 = AFTER_GENERATOR (TransformerVoltageControlOuterLoop), 1 = WITH_GENERATOR
     *  (SimpleTransformerVoltageControlOuterLoop); {@code entryOn} = the activation the solve starts from
     *  already carries the control ON (the WITH_GENERATOR base load flow — OLF's initialize ran on the
     *  network). Returns true when the kernel loop owns them (every member has a tap table). */
    public static boolean uploadToDevice(long handle, List<TransformerController> tcs, int mode, boolean entryOn) {
        if (tcs == null || tcs.isEmpty()) {
            GpuAcNewtonSolver.setBatchDiscreteTvc(handle, 0, 0, null, null, null, null, null, null, null, null, null,
                    null, null, null, null, null, null, null, null, null, null);
            return false;
        }
        int nG = tcs.size();
        int[] grpPtr = new int[nG + 1];
        int[] grpVRow = new int[nG];
        double[] grpTargetV = new double[nG];
        double[] grpHalfDb = new double[nG];
        int[] grpUseInit = new int[nG];
        int[] deltaPtr = new int[nG + 1];
        List<Integer> dRow = new ArrayList<>();
        List<Integer> dModeOn = new ArrayList<>();
        List<Integer> dModeOff = new ArrayList<>();
        List<Double> dTgtOn = new ArrayList<>();
        List<Double> dTgtOff = new ArrayList<>();
        List<Integer> memRhoRow = new ArrayList<>();
        List<Integer> memElem = new ArrayList<>();
        List<Integer> memBaseTap = new ArrayList<>();
        List<Integer> memTapPtr = new ArrayList<>();
        List<Double> memMin = new ArrayList<>();
        List<Double> memMax = new ArrayList<>();
        List<Double> memInit = new ArrayList<>();
        List<Double> tapData = new ArrayList<>();
        boolean tables = true;
        for (int g = 0; g < nG; g++) {
            TransformerController tc = tcs.get(g);
            grpPtr[g] = memRhoRow.size();
            grpVRow[g] = tc.vCol();
            grpTargetV[g] = tc.targetV();
            grpHalfDb[g] = tc.halfDeadband();
            grpUseInit[g] = tc.useInitialTapPosition() ? 1 : 0;
            deltaPtr[g] = dRow.size();
            for (int k = 0; k < tc.deltaRows().length; k++) {
                dRow.add(tc.deltaRows()[k]);
                dModeOn.add(tc.deltaModesOn()[k]);
                dModeOff.add(tc.deltaModesOff()[k]);
                dTgtOn.add(tc.deltaTargetsOn()[k]);
                dTgtOff.add(tc.deltaTargetsOff()[k]);
            }
            for (Member m : tc.members()) {
                memRhoRow.add(m.rhoRow());
                memElem.add(m.elemIndex());
                memBaseTap.add(m.baseTapIdx());
                memTapPtr.add(tapData.size() / 8);
                memMin.add(m.minR1());
                memMax.add(m.maxR1());
                memInit.add(m.initialR1());
                tables &= m.tapRows().length > 0;
                for (double[] row : m.tapRows()) {
                    for (double d : row) {
                        tapData.add(d);
                    }
                }
            }
        }
        grpPtr[nG] = memRhoRow.size();
        deltaPtr[nG] = dRow.size();
        memTapPtr.add(tapData.size() / 8);
        boolean owned = tables && GpuAcNewtonSolver.setBatchDiscreteTvc(handle, mode, entryOn ? 1 : 0,
                grpPtr, grpVRow, grpTargetV, grpHalfDb, grpUseInit,
                deltaPtr, dRow.stream().mapToInt(Integer::intValue).toArray(),
                dModeOn.stream().mapToInt(Integer::intValue).toArray(), dModeOff.stream().mapToInt(Integer::intValue).toArray(),
                dTgtOn.stream().mapToDouble(Double::doubleValue).toArray(), dTgtOff.stream().mapToDouble(Double::doubleValue).toArray(),
                memRhoRow.stream().mapToInt(Integer::intValue).toArray(), memElem.stream().mapToInt(Integer::intValue).toArray(),
                memBaseTap.stream().mapToInt(Integer::intValue).toArray(), memTapPtr.stream().mapToInt(Integer::intValue).toArray(),
                memMin.stream().mapToDouble(Double::doubleValue).toArray(), memMax.stream().mapToDouble(Double::doubleValue).toArray(),
                memInit.stream().mapToDouble(Double::doubleValue).toArray(),
                tapData.stream().mapToDouble(Double::doubleValue).toArray()) != 0;
        if (!owned) {
            GpuAcNewtonSolver.setBatchDiscreteTvc(handle, 0, 0, null, null, null, null, null, null, null, null, null,
                    null, null, null, null, null, null, null, null, null, null);
        }
        LOGGER.info("Uploaded {} discrete transformer voltage controls ({} members, mode {}) to device: {}",
                nG, memRhoRow.size(), mode == 1 ? "WITH_GENERATOR" : "AFTER_GENERATOR", owned ? "in-kernel loop" : "host orchestrator");
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("TVD_UPLOAD mode=%d entryOn=%b nGroups=%d nMembers=%d kernel=%b%n", mode, entryOn, nG, memRhoRow.size(), owned);
        }
        return owned;
    }

    /** Give the in-kernel phase's decisions back to the network (batch-of-one base load flow), exactly as
     *  OLF's own loop leaves it: a FIXED member had {@code setR1(roundFrom); roundR1ToClosestTap()} and
     *  its control disabled ({@code roundR1ToExtremeTapPosition} / {@code roundVoltageRatios}); a member
     *  never enabled keeps its base tap. {@code state} is readBatchDiscreteTvcState's layout. */
    public static void writeBack(LfNetwork network, List<TransformerController> tcs, double[] state) {
        if (tcs == null || tcs.isEmpty() || state == null) {
            return;
        }
        int m = 0;
        for (TransformerController tc : tcs) {
            for (Member mem : tc.members()) {
                int k = m++ * 3;
                if (k + 2 >= state.length) {
                    return;
                }
                LfBranch branch = network.getBranch(mem.branchNum());
                if (branch == null || (int) state[k] != 2) {
                    continue;
                }
                var pm = branch.getPiModel();
                int pos = pm.getTapPositionRange().getMinimum() + (int) state[k + 1];
                if (pm.getTapPosition() != pos) {
                    pm.setR1(state[k + 2]);
                    pm.roundR1ToClosestTap();
                    if (pm.getTapPosition() != pos) {        // tie on a different start tap: take the device's
                        pm.setTapPosition(pos);
                    }
                }
                if (branch.isVoltageControlEnabled()) {
                    branch.setVoltageControlEnabled(false);
                }
            }
        }
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
            int[] chunkDrOvIdx, double[] chunkDrOvC, int[] drRowByContrib,
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
                boolean anySurvivor = false;                 // plan-A6: a FREE-member outage re-picks the
                for (Member m : tc.members()) {              // carrier (mode-scan scatter + offDelta encode
                    anySurvivor |= !disabledMask[slot][m.elemIndex()];   // it); only a fully-outaged group skips
                }
                if (!anySurvivor) {
                    continue;
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
        // useInitialTapPosition: the CPU builds its TransformerRatioManager at the INITIAL step
        // over the just-enabled controllers — per scenario that is the ON control's non-outaged
        // members, and the stats stay frozen for the run (later clamps do not rebuild them).
        SharedRatio[][] shared = new SharedRatio[chunkSize][tcs.size()];
        for (int[] e : enabled) {
            affected.add(e[0]);
            on[e[0]][e[1]] = true;
            if (tcs.get(e[1]).useInitialTapPosition()) {
                shared[e[0]][e[1]] = SharedRatio.of(tcs.get(e[1]).members(), disabledMask[e[0]]);
            }
        }
        LOGGER.info("Transformer control: {} out-of-deadband controls across {} scenario(s) of the chunk", enabled.size(), affected.size());

        // Base per-scenario overrides (islanding/gen-outage) are kept and extended per call.
        List<Integer> ovIdx = new ArrayList<>();
        List<Double> ovVal = new ArrayList<>();
        double[] rounded = new double[chunkSize * totalMembers];   // fixed r1 once clamped/rounded (NaN = still continuous)
        boolean[] polish = new boolean[chunkSize];           // slot owes OLF's post-clamp COMPLETE solve
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
            // DS-row target seeds at the carried values (shared index-backed merge — the inline
            // lastIndexOf scan this replaced was quadratic in realCount * nds)
            seedDsTargets(nds, dsPhiRow, dsSeed, chunkResults, realCount, n, ovIdx, ovVal);
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
            // plan-A6: the CURRENT carrier's packed DISTR_RHO contributions must gate OFF per
            // scenario (its row hosts the pilot's relocated Q). Merge zero-entries for each slot's
            // carrier row AFTER the chunk-level drOv (the kernel's scan is last-match-wins) and
            // (re)install the channel — the carrier can move between steps as clamps accumulate.
            if (drRowByContrib != null && drRowByContrib.length > 0) {
                List<Integer> drIdx = new ArrayList<>();
                List<Double> drC = new ArrayList<>();
                if (chunkDrOvIdx != null) {
                    for (int k = 0; k < chunkDrOvIdx.length; k++) {
                        drIdx.add(chunkDrOvIdx[k]);
                    }
                    for (double v : chunkDrOvC) {
                        drC.add(v);
                    }
                }
                for (int slot : affected) {
                    for (int c = 0; c < tcs.size(); c++) {
                        if (!on[slot][c]) {
                            continue;
                        }
                        Member[] mems = tcs.get(c).members();
                        int carrier = -1;
                        for (int m = 0; m < mems.length; m++) {
                            if (!disabledMask[slot][mems[m].elemIndex()]
                                    && Double.isNaN(rounded[slot * totalMembers + memberBase[c] + m])) {
                                carrier = m;
                                break;
                            }
                        }
                        if (carrier <= 0) {
                            continue;                        // base carrier's row has no packed contributions
                        }
                        int carrierRow = mems[carrier].rhoRow();
                        for (int k = 0; k < drRowByContrib.length; k++) {
                            if (drRowByContrib[k] == carrierRow) {
                                drIdx.add(slot);
                                drIdx.add(k);
                                drC.add(0.0);
                            }
                        }
                    }
                }
                GpuAcNewtonSolver.setBatchDrCoefOverrides(handle,
                        drIdx.isEmpty() ? null : drIdx.stream().mapToInt(Integer::intValue).toArray(),
                        drC.isEmpty() ? null : drC.stream().mapToDouble(Double::doubleValue).toArray());
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
            boolean[] taken = new boolean[chunkSize];
            for (int slot : affected) {
                chunkResults[slot] = r2[slot];
                taken[slot] = true;
            }
            if (nds > 0) {                                   // this solve may have distributed further
                captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed, taken);
            }
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                traceStep(step, affected, chunkResults, tcs, n, rounded, memberBase, totalMembers, disabledMask);
            }
            int action = clampOrRoundStep(affected, chunkResults, n, tcs, on, rounded, memberBase, totalMembers, disabledMask, shared, polish);
            if (action == 0) {
                // A slot whose LAST continuous member was fixed by a CLAMP (not a round) owes one
                // more solve: OLF's controlStep returns UNSTABLE "in any case", so the COMPLETE
                // transition triggers one more NR — and OLF's NewtonRaphson ALWAYS takes at least
                // one Newton step before testing (no early exit on a converged start), so that
                // "no-op" solve perturbs the state by ~1e-6 pu (measured 9.2e-4 kV without it).
                // The round path needs none: rounding and COMPLETE share the same controlStep.
                java.util.Set<Integer> needPolish = new java.util.TreeSet<>();
                for (int slot : affected) {
                    if (polish[slot]) {
                        polish[slot] = false;
                        needPolish.add(slot);
                    }
                }
                if (needPolish.isEmpty()) {
                    break;                                   // everything fixed and re-solved: done
                }
                affected = needPolish;                       // one extra pinned solve for those slots
            }
        }
        return chunkResults;
    }

    /** One CONTROL decision per machine step: clamp out-of-range continuous ratios (returns 1 —
     *  re-solve with them fixed); when none, round ALL remaining continuous ones (returns 2 — one
     *  final solve); when nothing is continuous anymore, returns 0 (done). Sets {@code polish} for
     *  a slot whose clamp fixed its last continuous member (it still owes OLF's COMPLETE solve). */
    private static int clampOrRoundStep(java.util.Set<Integer> affected, double[][] chunkResults, int n,
            List<TransformerController> tcs, boolean[][] on, double[] rounded, int[] memberBase, int totalMembers,
            boolean[][] disabledMask, SharedRatio[][] shared, boolean[] polish) {
        boolean anyClamped = false;
        boolean anyContinuous = false;
        for (int slot : affected) {
            double[] r = chunkResults[slot];
            if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                continue;
            }
            boolean slotClamped = false;
            boolean slotContinuous = false;
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
                    SharedRatio sr = tc.useInitialTapPosition() ? shared[slot][c] : null;
                    if (sr != null && (r1 < sr.minR1() || r1 > sr.maxR1())) {
                        // SHARED-ratio clamp (useInitialTapPosition): the COMMON continuous ratio
                        // trips the GROUP-AVERAGED bounds, so every alive member clamps in the
                        // SAME pass — OLF sets each to the group extreme, rounds to its OWN
                        // closest tap and disables it; the whole control ends together, so no
                        // mid-flight carrier re-pick can arise (the plan-A6 rejection below is
                        // individual-mode only).
                        rounded[slot * totalMembers + memberBase[c] + m] =
                                mem.closestTap(r1 > sr.maxR1() ? sr.maxR1() : sr.minR1());
                        slotClamped = true;
                        continue;
                    }
                    if (sr == null && (r1 < mem.minR1() || r1 > mem.maxR1())) {
                        // plan-A6 remaining: clamping the CURRENT CARRIER while other members
                        // stay continuous re-picks the pilot-Q role MID-FLIGHT; the re-picked
                        // system sits on the CPU's 1-MW DS-threshold knife edge (measured
                        // 1.7e-9 trajectory offset flipping a 1.0045-MW distribution decision)
                        // — per-scenario CPU route until the trajectory is exactly replicated.
                        boolean isCarrier = true;
                        for (int mm = 0; mm < m; mm++) {
                            if (!disabledMask[slot][members[mm].elemIndex()]
                                    && Double.isNaN(rounded[slot * totalMembers + memberBase[c] + mm])) {
                                isCarrier = false;       // an earlier alive member carries the pilot Q
                                break;
                            }
                        }
                        boolean otherAlive = false;
                        for (int mm = 0; mm < members.length; mm++) {
                            if (mm != m && !disabledMask[slot][members[mm].elemIndex()]
                                    && Double.isNaN(rounded[slot * totalMembers + memberBase[c] + mm])) {
                                otherAlive = true;
                            }
                        }
                        if (isCarrier && otherAlive) {
                            chunkResults[slot][n + 2] = GpuBatchStatus.STATUS_REJECTED;
                            continue;
                        }
                        rounded[slot * totalMembers + memberBase[c] + m] = Math.max(mem.minR1(), Math.min(mem.maxR1(), r1));
                        slotClamped = true;
                    } else {
                        slotContinuous = true;
                    }
                }
            }
            if ((int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                continue;                                    // rejected mid-slot: no further solves for it
            }
            anyClamped |= slotClamped;
            anyContinuous |= slotContinuous;
            if (slotClamped && !slotContinuous) {
                polish[slot] = true;                         // clamp ended the whole control: COMPLETE solve owed
            }
        }
        if (anyClamped) {
            return 1;                                        // re-solve with the clamped ones fixed
        }
        if (anyContinuous) {                                 // round ALL remaining continuous ratios, one final solve
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
                    SharedRatio srr = tc.useInitialTapPosition() ? shared[slot][c] : null;
                    for (int m = 0; m < members.length; m++) {
                        if (Double.isNaN(rounded[slot * totalMembers + memberBase[c] + m])
                                && !disabledMask[slot][members[m].elemIndex()]) {
                            // useInitialTapPosition: OLF's updateContinuousRatio remaps the COMMON
                            // solved ratio to each member's spread-preserving individual value
                            // right before roundVoltageRatios rounds it (same check, one re-solve
                            // after) — compose remap + round here.
                            double cont = r[members[m].rhoRow()];
                            rounded[slot * totalMembers + memberBase[c] + m] =
                                    members[m].closestTap(srr != null ? srr.remap(members[m], cont) : cont);
                        }
                    }
                }
            }
            return 2;                                        // the final solve runs with every control fixed
        }
        return 0;                                            // everything fixed and re-solved: done
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
        if (!allFixed) {                                     // control ACTIVE: build the group's rows for the
            // EFFECTIVE role set (plan-A6). Pairwise offDeltas do NOT compose when a clamp and an
            // outage combine (the free member's offDelta encodes a re-pick to a member that may
            // itself be outaged this scenario), so member rho rows are constructed DIRECTLY:
            //  - the CARRIER (first alive member — OLF's free-role rule) hosts the pilot's
            //    relocated Q (the group mode-scan scatter routes the flows; the target is the one
            //    captured for the BASE carrier's row);
            //  - other alive members host DISTR_RHO (target 0);
            //  - clamped / outaged members get identity pins (clamp value / base-state rho).
            // Non-member delta rows (the pilot's v identity) apply as captured.
            int carrier = -1;
            for (int m = 0; m < members.length; m++) {
                if (!disabledMask[slot][members[m].elemIndex()]
                        && Double.isNaN(rounded[slot * totalMembers + base + m])) {
                    carrier = m;
                    break;
                }
            }
            double pilotQTarget = 0;
            for (int k = 0; k < tc.deltaRows().length; k++) {
                if (tc.deltaRows()[k] == members[0].rhoRow()) {
                    pilotQTarget = tc.deltaTargetsOn()[k];   // the BASE carrier's captured pilot-Q target
                }
            }
            java.util.Set<Integer> memberRows = new java.util.HashSet<>();
            for (Member mem : members) {
                memberRows.add(mem.rhoRow());
            }
            for (int k = 0; k < tc.deltaRows().length; k++) {
                if (memberRows.contains(tc.deltaRows()[k])) {
                    continue;                                // member rho rows are built by ROLE below
                }
                row[tc.deltaRows()[k]] = tc.deltaModesOn()[k];
                ovIdx.add(slot * n + tc.deltaRows()[k]);
                ovVal.add(tc.deltaTargetsOn()[k]);
            }
            for (int m = 0; m < members.length; m++) {
                Member mem = members[m];
                boolean outaged = disabledMask[slot][mem.elemIndex()];
                double fixedR1 = rounded[slot * totalMembers + base + m];
                if (!outaged && Double.isNaN(fixedR1)) {     // ALIVE: carrier or DISTR_RHO
                    row[mem.rhoRow()] = 1;
                    ovIdx.add(slot * n + mem.rhoRow());
                    ovVal.add(m == carrier ? pilotQTarget : 0.0);
                    continue;
                }
                row[mem.rhoRow()] = 0;                       // identity pin
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
    /** Merge the carried absolute DS-row target seeds into an override list — replace an existing
     *  entry for the row, never duplicate it (the device scatter has no ordering). Shared by every
     *  orchestrator that re-solves after the main chunk solve.
     *
     *  <p>The lookup goes through a flat-row → position INDEX, not a scan: this runs
     *  {@code realCount * nds} times over a list that ends up the same size, so the obvious
     *  {@code lastIndexOf} made it quadratic — measured on a 6.5k-bus network with 554 DS
     *  participants and 500 scenarios, the two orchestrators that re-solve (HVDC AC-emulation and
     *  VoltageMonitoring) spent 97 s of a 101 s batch in here, against 1.5 s of actual device
     *  solving. */
    static void seedDsTargets(int nds, int[] dsPhiRow, double[] dsSeed, double[][] chunkResults,
            int realCount, int n, List<Integer> ovIdx, List<Double> ovVal) {
        if (nds <= 0) {
            return;
        }
        java.util.Map<Integer, Integer> posOf = new java.util.HashMap<>(2 * ovIdx.size() + 16);
        for (int k = 0; k < ovIdx.size(); k++) {
            posOf.put(ovIdx.get(k), k);                      // later duplicates win, as lastIndexOf did
        }
        for (int s2 = 0; s2 < realCount; s2++) {
            double[] r = chunkResults[s2];
            if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                continue;
            }
            for (int i = 0; i < nds; i++) {
                int flat = s2 * n + dsPhiRow[i];
                Integer at = posOf.get(flat);
                if (at != null) {
                    ovVal.set(at, dsSeed[s2 * nds + i]);
                } else {
                    posOf.put(flat, ovIdx.size());
                    ovIdx.add(flat);
                    ovVal.add(dsSeed[s2 * nds + i]);
                }
            }
        }
    }

    static void captureDsCarry(long handle, int chunkSize, int realCount, int n,
            int[] dsPhiRow, double[] dsFactor, double[] baseTargets,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta,
            double[][] chunkResults, double[] anchor, double[] slack, double[] seed) {
        captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, anchor, slack, seed, null);
    }

    /**
     * As above, but capturing ONLY the scenarios in {@code taken} (null = all of them).
     *
     * <p>The distribution is read from the DEVICE's target buffer, which a batched solve rewrites for
     * every scenario in the chunk — including the ones an orchestrator solved but whose result it then
     * threw away, because they were already stable and the CPU would never have re-run Newton on them
     * ("untouched scenarios KEEP their converged state"). Capturing those slots too hands the next
     * solve a slack distribution belonging to a trajectory the scenario never took, while its state
     * says otherwise. So the mask that decides which results are kept must also decide which carries
     * are captured; an untouched slot keeps the carry that matches the state it kept.
     */
    @SuppressWarnings("java:S107")
    static void captureDsCarry(long handle, int chunkSize, int realCount, int n,
            int[] dsPhiRow, double[] dsFactor, double[] baseTargets,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta,
            double[][] chunkResults, double[] anchor, double[] slack, double[] seed, boolean[] taken) {
        int nds = dsPhiRow.length;
        double fsBase = 0;
        for (double fv : dsFactor) {
            fsBase += fv;
        }
        double[] finalTargets = GpuAcNewtonSolver.readBatchTargets(handle);
        for (int s = 0; s < chunkSize; s++) {
            if (taken != null && s < realCount && !taken[s]) {
                continue;                                    // result discarded — its carry stays as it was
            }
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
