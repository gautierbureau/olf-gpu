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

    /** One voltage-controlling transformer, precomputed for the per-scenario orchestrator:
     *  rho row (== r1 col, self-paired), controlled-bus V col, target/deadband, continuous r1
     *  range, discrete tap r1 values, its element index (per-scenario outage check), and the
     *  row-mode/target DELTAS between control-off (base) and control-on activations. */
    record TransformerController(int rhoRow, int vCol, double targetV, double halfDeadband,
                                 double minR1, double maxR1, double[] taps, int elemIndex,
                                 int[] deltaRows, int[] deltaModesOn, double[] deltaTargetsOn) {
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
    }

    /** Stage-1 scope check: the CPU loop's GeneratorVoltageControlManager disables generator
     *  voltage controls at buses under the max transformer-controlled nominal voltage. Porting
     *  the frozen-Q flip is stage 2 — until then a network that would NEED it must stay on the
     *  CPU path (the provider's guard calls this). */
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
        for (LfBranch branch : network.<LfBranch>getControllerElements(com.powsybl.openloadflow.network.VoltageControl.Type.TRANSFORMER)) {
            var vcOpt = branch.getVoltageControl();
            if (vcOpt.isEmpty() || !branch.isConnectedAtBothSides()) {
                continue;
            }
            var vc = vcOpt.orElseThrow();
            Integer r1Col = branchR1Cols.get(branch.getNum());
            Integer vCol = busVCols.get(vc.getControlledBus().getNum());
            int elemIndex = elementMapper.getElementIndex(branch);
            if (r1Col == null || vCol == null || elemIndex < 0) {
                LOGGER.info("Transformer controller {} not representable on the GPU batch — skipped (results follow the base tap)", branch.getId());
                continue;
            }
            double halfDeadband = vc.getTargetDeadband().orElse(0.1 / vc.getControlledBus().getNominalV()) / 2;
            var piModel = branch.getPiModel();
            double[] taps = tapR1Values(piModel);
            boolean wasEnabled = branch.isVoltageControlEnabled();
            branch.setVoltageControlEnabled(true);
            GpuAcActivation onAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] onTargets = onAct.applyTargets(targetVector.getArray());
            branch.setVoltageControlEnabled(wasEnabled);
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
            out.add(new TransformerController(r1Col, vCol, vc.getTargetValue(), halfDeadband,
                    piModel.getMinR1(), piModel.getMaxR1(), taps, elemIndex,
                    deltaRows, deltaModesOn, deltaTargetsOn));
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
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        // INITIAL: which (slot, controller) pairs are out of deadband on the converged first solve?
        List<int[]> enabled = new ArrayList<>();             // {slot, controllerIdx}
        for (int slot = 0; slot < realCount; slot++) {
            double[] r = chunkResults[slot];
            if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                continue;
            }
            for (int c = 0; c < tcs.size(); c++) {
                TransformerController tc = tcs.get(c);
                if (disabledMask[slot][tc.elemIndex()]) {
                    continue;                                // the controller transformer itself is outaged
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
        double[] rounded = new double[chunkSize * tcs.size()];   // fixed r1 once clamped/rounded (NaN = still continuous)
        java.util.Arrays.fill(rounded, Double.NaN);

        int[][] rowMode = new int[chunkSize][];
        for (int step = 0; step < 8; step++) {               // INITIAL solve, clamp iterations, final rounded solve
            ovIdx.clear();
            ovVal.clear();
            for (int k = 0; k < targetOverrideIndex.length; k++) {
                ovIdx.add(targetOverrideIndex[k]);
                ovVal.add(targetOverrideValue[k]);
            }
            double[] x0Full = new double[chunkSize * n];
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
                    TransformerController tc = tcs.get(c);
                    double fixedR1 = rounded[slot * tcs.size() + c];
                    if (on[slot][c] && Double.isNaN(fixedR1)) {          // control ACTIVE: apply the ON deltas
                        for (int k = 0; k < tc.deltaRows().length; k++) {
                            row[tc.deltaRows()[k]] = tc.deltaModesOn()[k];
                            ovIdx.add(slot * n + tc.deltaRows()[k]);
                            ovVal.add(tc.deltaTargetsOn()[k]);
                        }
                    } else if (!Double.isNaN(fixedR1)) {                 // clamped/rounded: identity at the fixed r1
                        ovIdx.add(slot * n + tc.rhoRow());
                        ovVal.add(fixedR1);
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
                    double fixedR1 = rounded[slot * tcs.size() + c];
                    if (!Double.isNaN(fixedR1)) {
                        x0Full[slot * n + tcs.get(c).rhoRow()] = fixedR1;
                    }
                }
            }
            double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                    rowMode, baseRowMode, disabledMask,
                    ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                    ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, null, null,
                    parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                    config.lsMode(), config.maxDv(), config.maxDphi(),
                    config.lsMode() == 2 ? varType : null, x0Full, null, null, null, null, null, null,
                    config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0, null, null);
            for (int slot : affected) {
                chunkResults[slot] = r2[slot];
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
                    if (!on[slot][c] || !Double.isNaN(rounded[slot * tcs.size() + c])) {
                        continue;
                    }
                    double r1 = r[tc.rhoRow()];
                    if (r1 < tc.minR1() || r1 > tc.maxR1()) {
                        rounded[slot * tcs.size() + c] = Math.max(tc.minR1(), Math.min(tc.maxR1(), r1));
                        anyClamped = true;
                    } else {
                        anyContinuous = true;
                    }
                }
            }
            if (anyClamped) {
                continue;                                    // re-solve with the clamped ones fixed
            }
            if (anyContinuous) {                             // round ALL remaining continuous ratios, one final solve
                for (int slot : affected) {
                    double[] r = chunkResults[slot];
                    if (r == null) {
                        continue;
                    }
                    for (int c = 0; c < tcs.size(); c++) {
                        TransformerController tc = tcs.get(c);
                        if (on[slot][c] && Double.isNaN(rounded[slot * tcs.size() + c])) {
                            rounded[slot * tcs.size() + c] = tc.closestTap(r[tc.rhoRow()]);
                        }
                    }
                }
                continue;                                    // the final solve runs with every control fixed
            }
            break;                                           // everything fixed and re-solved: done
        }
        return chunkResults;
    }
}
