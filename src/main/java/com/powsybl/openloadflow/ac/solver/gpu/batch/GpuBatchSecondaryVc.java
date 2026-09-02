/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.math.matrix.LUDecomposition;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationTerm;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfSecondaryVoltageControl;
import com.powsybl.openloadflow.network.VoltageControl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of OLF's {@code SecondaryVoltageControlOuterLoop} (pilot-point zone regulation).
 * The architecture follows the session's established split:
 *
 * <ul>
 *   <li>the BIG Jacobian sensitivity solve (unit RHS at each controlled bus's pinned BUS_TARGET_V
 *       row) runs BATCHED ON THE DEVICE via {@link GpuAcNewtonSolver#solveBatchSensitivities} — the
 *       same forward factorization as the Newton, no CPU Jacobian is ever formed;</li>
 *   <li>the SMALL zone system (A, K0, Jk, Jvpp, the pilot-row replacement and the tiny dense LU) is
 *       OLF's own math replayed per scenario on the host with powsybl's DenseMatrix — at
 *       nControllers×nControllers size a GPU port would be pure overhead;</li>
 *   <li>the observables (controller-bus branch/shunt Q sensitivities, pilot V sensitivity, the
 *       controllers' K) reuse OLF's OWN EquationTerms via es-state loading + sparse DenseMatrix
 *       injection ({@code calculateSensi}/{@code eval}), so nothing is re-derived.</li>
 * </ul>
 *
 * <p>Per-scenario mechanics: new generator-VC targets land as V-row target overrides (PV rows) and
 * per-scenario RL V-target overrides (the {@code rlVtOv} channel — the RL switch-back must compare
 * against the MOVED target); OLF's {@code tryToReEnableHelpfulControllerBuses} maps to clearing the
 * some-but-not-all pinned controllers' RL state before the re-solve. Stage-1 scope: LOCAL generator
 * voltage control only (controller bus == controlled bus), every controller in the RL pack.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchSecondaryVc {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchSecondaryVc.class);

    private static final double DV_EPS = 1E-4;               // OLF SecondaryVoltageControlOuterLoop
    private static final double DK_DIFF_MAX_EPS = 1E-3;

    private GpuBatchSecondaryVc() {
    }

    /** One (local) controller/controlled bus: the pinned V row (sensitivity unit row, target index,
     *  RL vtarget anchor), its RL pack index, limits, and OLF's own Q terms for K and dQ/dVtarget. */
    record CtlBus(int vRow, int rlIdx, double baseTargetV, double minQ, double maxQ, double loadTargetQ,
                  com.powsybl.openloadflow.util.Evaluable busQ,
                  List<EquationTerm<AcVariableType, AcEquationType>> qTerms) {
    }

    record Zone(int pilotVRow, double pilotTargetV, int[] ctls) {
    }

    record Controllers(List<CtlBus> ctls, List<Zone> zones,
                       GpuBatchIncrementalFlowControl.EsContext esCtx,
                       double minPlausibleTargetV, double maxPlausibleTargetV) {
    }

    /** True if every enabled secondary-voltage-control zone is representable (LOCAL generator VC,
     *  every controller in the device RL pack) — else the run must stay on the CPU. */
    public static boolean supported(LfNetwork network) {
        for (LfSecondaryVoltageControl svc : network.getEnabledSecondaryVoltageControls()) {
            for (LfBus controller : svc.getControllerBuses()) {
                var vc = controller.getGeneratorVoltageControl().orElse(null);
                if (vc == null || vc.getControlledBus() != controller) {
                    return false;                            // remote control — stage-1 scope is local only
                }
            }
        }
        return true;
    }

    @SuppressWarnings("unchecked")
    static Controllers build(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                             int[] rlVrow, GpuBatchIncrementalFlowControl.EsContext esCtx,
                             double minPlausibleTargetV, double maxPlausibleTargetV) {
        Map<Integer, Integer> busVRows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVRows.put(v.getElementNum(), v.getRow());
            }
        }
        Map<Integer, Integer> rlIdxOfVRow = new LinkedHashMap<>();
        for (int i = 0; i < rlVrow.length; i++) {
            rlIdxOfVRow.put(rlVrow[i], i);
        }
        List<CtlBus> ctls = new ArrayList<>();
        List<Zone> zones = new ArrayList<>();
        for (LfSecondaryVoltageControl svc : network.getEnabledSecondaryVoltageControls()) {
            List<Integer> zoneCtls = new ArrayList<>();
            for (LfBus controller : svc.getControllerBuses()) {
                var vc = controller.getGeneratorVoltageControl().orElseThrow();
                Integer vRow = busVRows.get(controller.getNum());
                Integer rlIdx = vRow != null ? rlIdxOfVRow.get(vRow) : null;
                if (vRow == null || rlIdx == null) {
                    LOGGER.info("Secondary VC controller {} not representable on the GPU batch — skipped zone {}",
                            controller.getId(), svc.getZoneName());
                    return new Controllers(List.of(), List.of(), esCtx, minPlausibleTargetV, maxPlausibleTargetV);
                }
                // OLF calculateSensiQ: the controller bus's branch Q flows (side by attachment) + shunts
                List<EquationTerm<AcVariableType, AcEquationType>> qTerms = new ArrayList<>();
                for (LfBranch branch : controller.getBranches()) {
                    if (!branch.isDisabled() && branch.isConnectedAtBothSides()) {
                        if (branch.getBus1() == controller) {
                            qTerms.add((EquationTerm<AcVariableType, AcEquationType>) branch.getQ1());
                        }
                        if (branch.getBus2() == controller) {
                            qTerms.add((EquationTerm<AcVariableType, AcEquationType>) branch.getQ2());
                        }
                    }
                }
                controller.getShunt().ifPresent(sh -> qTerms.add((EquationTerm<AcVariableType, AcEquationType>) sh.getQ()));
                controller.getControllerShunt().ifPresent(sh -> qTerms.add((EquationTerm<AcVariableType, AcEquationType>) sh.getQ()));
                controller.getSvcShunt().ifPresent(sh -> qTerms.add((EquationTerm<AcVariableType, AcEquationType>) sh.getQ()));
                zoneCtls.add(ctls.size());
                ctls.add(new CtlBus(vRow, rlIdx, vc.getTargetValue(), controller.getMinQ(), controller.getMaxQ(),
                        controller.getLoadTargetQ(), controller.getQ(), qTerms));
            }
            Integer pilotVRow = busVRows.get(svc.getPilotBus().getNum());
            if (pilotVRow == null || zoneCtls.isEmpty()) {
                return new Controllers(List.of(), List.of(), esCtx, minPlausibleTargetV, maxPlausibleTargetV);
            }
            zones.add(new Zone(pilotVRow, svc.getTargetValue(),
                    zoneCtls.stream().mapToInt(Integer::intValue).toArray()));
        }
        return new Controllers(ctls, zones, esCtx, minPlausibleTargetV, maxPlausibleTargetV);
    }

    private static double qToK(double q, CtlBus c) {         // OLF qToK
        return (2 * q - c.maxQ() - c.minQ()) / (c.maxQ() - c.minQ());
    }

    @SuppressWarnings({"java:S107", "java:S3776"})
    static double[][] orchestrate(long handle, int chunkSize, int realCount, int n, int nrl,
            double[][] chunkResults, Controllers controllers, int[] rlState,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            int[] targetOverrideIndex, double[] targetOverrideValue,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            double[] dsAnchorShift, double[] dsSlackShift,
            int[] r1OvSlotElem, double[] r1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        List<CtlBus> ctls = controllers.ctls();
        List<Zone> zones = controllers.zones();
        int nc = ctls.size();
        if (nc == 0 || rlState == null) {
            return chunkResults;
        }
        var esCtx = controllers.esCtx();
        double[][] targetV = new double[chunkSize][nc];      // per-scenario CURRENT generator-VC targets
        for (int s = 0; s < chunkSize; s++) {
            for (int c = 0; c < nc; c++) {
                targetV[s][c] = ctls.get(c).baseTargetV();
            }
        }
        // sensitivity geometry: unit rows = every controlled bus's V row; observed columns = the union
        // of the q-terms' variable rows + the pilot V rows (OLF extracts through these same terms)
        int[] unit = new int[nc];
        for (int c = 0; c < nc; c++) {
            unit[c] = ctls.get(c).vRow();
        }
        List<Integer> obsList = new ArrayList<>();
        Map<Integer, Integer> obsPos = new LinkedHashMap<>();
        for (CtlBus c : ctls) {
            for (EquationTerm<AcVariableType, AcEquationType> t : c.qTerms()) {
                for (Variable<AcVariableType> v : t.getVariables()) {
                    obsPos.computeIfAbsent(v.getRow(), r -> {
                        obsList.add(r);
                        return obsList.size() - 1;
                    });
                }
            }
        }
        for (Zone z : zones) {
            obsPos.computeIfAbsent(z.pilotVRow(), r -> {
                obsList.add(r);
                return obsList.size() - 1;
            });
        }
        int[] obs = obsList.stream().mapToInt(Integer::intValue).toArray();
        int m = obs.length;
        int maxRounds = config.maxOuterIterations() > 0 ? config.maxOuterIterations() : 20;

        for (int round = 0; round < maxRounds; round++) {
            // ---- per-scenario OLF check(): re-enable pass, zone triggers ----
            boolean[] scenAdjusts = new boolean[chunkSize];
            boolean[][] reEnabled = new boolean[chunkSize][nc];
            boolean anyAdjusts = false;
            for (int s = 0; s < realCount; s++) {
                double[] r = chunkResults[s];
                if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                    continue;
                }
                esCtx.stateVector().set(java.util.Arrays.copyOf(r, n));
                // tryToReEnableHelpfulControllerBuses: some-but-not-all pinned -> unpin them
                for (Zone z : zones) {
                    int pinned = 0;
                    for (int c : z.ctls()) {
                        if ((rlState[s * nrl + ctls.get(c).rlIdx()] & 3) != 0) {
                            pinned++;
                        }
                    }
                    if (pinned > 0 && pinned < z.ctls().length) {
                        for (int c : z.ctls()) {
                            if ((rlState[s * nrl + ctls.get(c).rlIdx()] & 3) != 0) {
                                reEnabled[s][c] = true;
                            }
                        }
                    }
                }
                // zone triggers: pilot deadband, K alignment spread
                for (Zone z : zones) {
                    double pilotDv = z.pilotTargetV() - r[z.pilotVRow()];
                    double kMin = Double.POSITIVE_INFINITY;
                    double kMax = Double.NEGATIVE_INFINITY;
                    boolean allAbove = true;
                    boolean allBelow = true;
                    for (int c : z.ctls()) {
                        CtlBus ctl = ctls.get(c);
                        double k = qToK(ctl.busQ().eval() + ctl.loadTargetQ(), ctl);
                        kMin = Math.min(kMin, k);
                        kMax = Math.max(kMax, k);
                        allAbove &= k > 1;
                        allBelow &= k < -1;
                    }
                    boolean allAtLimits = allAbove || allBelow;
                    if (Math.abs(pilotDv) > DV_EPS || !allAtLimits && Math.abs(kMax - kMin) > DK_DIFF_MAX_EPS) {
                        scenAdjusts[s] = true;
                        anyAdjusts = true;
                    }
                }
            }
            if (!anyAdjusts) {
                break;
            }
            // ---- batched device sensitivities dState/dVtarget for every controlled bus ----
            double[] statesFlat = new double[chunkSize * n];
            for (int s = 0; s < chunkSize; s++) {
                double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
                System.arraycopy(src, 0, statesFlat, s * n, n);
            }
            double[] dxAtObs = GpuAcNewtonSolver.solveBatchSensitivities(handle, statesFlat, unit, obs);

            // ---- OLF processSecondaryVoltageControl per adjusting scenario ----
            boolean[] affected = new boolean[chunkSize];
            boolean anyMoved = false;
            DenseMatrix dx = new DenseMatrix(n, nc);
            for (int s = 0; s < realCount; s++) {
                if (!scenAdjusts[s]) {
                    continue;
                }
                esCtx.stateVector().set(java.util.Arrays.copyOf(chunkResults[s], n));
                for (int c = 0; c < nc; c++) {
                    for (int mi = 0; mi < m; mi++) {
                        dx.set(obs[mi], c, dxAtObs[s * nc * m + c * m + mi]);
                    }
                }
                // enabled controllers this scenario: unpinned or just re-enabled
                List<Integer> enabled = new ArrayList<>();
                for (int c = 0; c < nc; c++) {
                    if ((rlState[s * nrl + ctls.get(c).rlIdx()] & 3) == 0 || reEnabled[s][c]) {
                        enabled.add(c);
                    }
                }
                if (enabled.isEmpty()) {
                    continue;
                }
                Map<Integer, Integer> col = new LinkedHashMap<>();   // ctl -> dense index
                for (int i = 0; i < enabled.size(); i++) {
                    col.put(enabled.get(i), i);
                }
                int ne = enabled.size();
                // A (zone alignment blocks over ENABLED controllers), K0, rhs = -A*K0
                DenseMatrix a = new DenseMatrix(ne, ne);
                for (Zone z : zones) {
                    List<Integer> zc = new ArrayList<>();
                    for (int c : z.ctls()) {
                        if (col.containsKey(c)) {
                            zc.add(c);
                        }
                    }
                    int nControl = zc.size();
                    for (int ci : zc) {
                        for (int cj : zc) {
                            int i = col.get(ci);
                            int j = col.get(cj);
                            a.set(i, j, i == j ? 1d - 1d / nControl : -1d / nControl);
                        }
                    }
                }
                DenseMatrix k0 = new DenseMatrix(ne, 1);
                for (int c : enabled) {
                    CtlBus ctl = ctls.get(c);
                    k0.set(col.get(c), 0, qToK(ctl.busQ().eval() + ctl.loadTargetQ(), ctl));
                }
                DenseMatrix rhs = a.times(k0, -1);
                // Jk(i=controlled, j=controller) = 2*sensiQ/(maxQ-minQ); local VC: controlled == controller set
                DenseMatrix jk = new DenseMatrix(ne, ne);
                for (int ci : enabled) {                     // controlled bus index (rows of jk pre-transpose)
                    for (int cj : enabled) {                 // controller
                        CtlBus ctl = ctls.get(cj);
                        double sq = 0;
                        for (EquationTerm<AcVariableType, AcEquationType> t : ctl.qTerms()) {
                            sq += t.calculateSensi(dx, ci);
                        }
                        jk.set(col.get(ci), col.get(cj), 2 * sq / (ctl.maxQ() - ctl.minQ()));
                    }
                }
                DenseMatrix b = a.times(jk.transpose());
                // per zone: replace the LAST controlled bus's row with the pilot sensitivities + pilot dv
                for (Zone z : zones) {
                    List<Integer> zc = new ArrayList<>();
                    for (int c : z.ctls()) {
                        if (col.containsKey(c)) {
                            zc.add(c);
                        }
                    }
                    if (zc.isEmpty()) {
                        continue;
                    }
                    int last = col.get(zc.get(zc.size() - 1));
                    for (int c : enabled) {
                        // dVpilot/dVtarget(controlled c) = the pilot V entry of column c
                        b.set(last, col.get(c), dx.get(z.pilotVRow(), c));
                    }
                    rhs.set(last, 0, z.pilotTargetV() - chunkResults[s][z.pilotVRow()]);
                }
                try (LUDecomposition lu = b.decomposeLU()) {
                    lu.solve(rhs);
                } catch (RuntimeException e) {
                    LOGGER.warn("Secondary VC zone system solve failed for a scenario — left unadjusted ({})", e.toString());
                    continue;
                }
                // new targets + plausibility gate (OLF skips the WHOLE adjustment when any is implausible)
                double[] newT = new double[nc];
                boolean plausible = true;
                for (int c = 0; c < nc; c++) {
                    if (col.containsKey(c)) {
                        newT[c] = targetV[s][c] + rhs.get(col.get(c), 0);
                        if (newT[c] < controllers.minPlausibleTargetV() || newT[c] > controllers.maxPlausibleTargetV()) {
                            plausible = false;
                        }
                    } else {
                        newT[c] = targetV[s][c];
                    }
                }
                if (!plausible) {
                    LOGGER.warn("Secondary VC: implausible new target voltages — scenario left unadjusted");
                    continue;
                }
                targetV[s] = newT;
                affected[s] = true;
                anyMoved = true;
                // apply the re-enables: clear the RL pins so the next solve resumes voltage control there
                for (int c = 0; c < nc; c++) {
                    if (reEnabled[s][c]) {
                        int st = s * nrl + ctls.get(c).rlIdx();
                        rlState[st] = rlState[st] & ~3;
                    }
                }
            }
            if (!anyMoved) {
                break;
            }
            // ---- apply: V-row targets (PV rows), RL vtarget overrides, warm re-solve ----
            List<Integer> ovIdx = new ArrayList<>();
            List<Double> ovVal = new ArrayList<>();
            for (int i = 0; i < targetOverrideIndex.length; i++) {
                ovIdx.add(targetOverrideIndex[i]);
                ovVal.add(targetOverrideValue[i]);
            }
            List<Integer> vtSlotIdx = new ArrayList<>();
            List<Double> vtVal = new ArrayList<>();
            for (int s = 0; s < realCount; s++) {
                for (int c = 0; c < nc; c++) {
                    if (targetV[s][c] != ctls.get(c).baseTargetV()) {
                        CtlBus ctl = ctls.get(c);
                        if ((rlState[s * nrl + ctl.rlIdx()] & 3) == 0) {
                            ovIdx.add(s * n + ctl.vRow());   // PV row: the V target moves now
                            ovVal.add(targetV[s][c]);
                        }                                    // pinned row: the Q pin stays; the override below
                        vtSlotIdx.add(s);                    // steers the RL switch-back and its restored target
                        vtSlotIdx.add(ctl.rlIdx());
                        vtVal.add(targetV[s][c]);
                    }
                }
            }
            int[] rlStateOut = new int[chunkSize * nrl];
            double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                    rowModePerScenario, baseRowMode, disabledMask,
                    ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                    ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorShift, dsSlackShift,
                    vtSlotIdx.stream().mapToInt(Integer::intValue).toArray(),
                    vtVal.stream().mapToDouble(Double::doubleValue).toArray(),
                    parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                    config.lsMode(), config.maxDv(), config.maxDphi(),
                    config.lsMode() == 2 ? varType : null, statesFlat,
                    shOvSlotElem, shOvGb, rlState, rlStateOut,
                    r1OvSlotElem, r1OvVal,
                    config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0,
                    nziBase, ziOnSlotElem, ziOffSlotElem);
            System.arraycopy(rlStateOut, 0, rlState, 0, rlStateOut.length);
            for (int s = 0; s < realCount; s++) {
                if (affected[s]) {                           // untouched scenarios KEEP their converged state
                    chunkResults[s] = r2[s];
                }
            }
        }
        return chunkResults;
    }
}
