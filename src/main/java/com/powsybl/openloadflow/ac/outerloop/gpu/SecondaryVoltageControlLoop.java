/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.math.matrix.LUDecomposition;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSkips;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchStatus;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationTerm;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfSecondaryVoltageControl;
import com.powsybl.openloadflow.network.LfShunt;
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
 * some-but-not-all pinned controllers' RL state before the re-solve. Scope: LOCAL and REMOTE
 * generator voltage control with DISTINCT controlled buses (a controller<->controlled bijection —
 * see {@link #supported}); a remote controller's RL entry is the rlRemote member entry, its moved
 * zone target lands on the CONTROLLED bus's V row and steers the pilot-referenced switch-back via
 * the rlVtOv channel.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class SecondaryVoltageControlLoop {

    private static final Logger LOGGER = LoggerFactory.getLogger(SecondaryVoltageControlLoop.class);

    private static final double DV_EPS = 1E-4;               // OLF SecondaryVoltageControlOuterLoop
    private static final double DK_DIFF_MAX_EPS = 1E-3;

    private SecondaryVoltageControlLoop() {
    }

    /** One Q observable of a controller bus, as the DEVICE sees it: {@code kind} 0/1 = the CLOSED
     *  BRANCH's side-1/side-2 reactive flow, whose gradient comes from the codegen'd evalClosedBranch
     *  (the very function the fill scatters); 2 = a SHUNT's q = -b*v^2, i.e. OLF's
     *  ShuntCompensatorReactiveFlowEquationTerm (dq/dv = -2bv, dq/db = -v^2). {@code elem} is the
     *  PACK-relative element index; {@code bRow} a derived shunt's SHUNT_B row, -1 when b is fixed. */
    public record QObs(int kind, int elem, int bRow) {
    }

    /** One (controller, controlled) PAIR — a bijection under {@link #supported}: {@code vRow} is
     *  the CONTROLLER bus's V row (keys its RL entry — for a REMOTE controller that is the
     *  rlRemote member entry), {@code ctldVRow} the CONTROLLED bus's pinned BUS_TARGET_V row (the
     *  sensitivity unit row and the row the moved zone target lands on — OLF adjusts the
     *  CONTROLLED bus's VC target). Local control collapses the two. */
    public record CtlBus(int vRow, int ctldVRow, int ctldBusNum, int rlIdx, double baseTargetV, double minQ, double maxQ,
                  double loadTargetQ,
                  com.powsybl.openloadflow.util.Evaluable busQ,
                  List<EquationTerm<AcVariableType, AcEquationType>> qTerms,
                  List<QObs> qObs) {
    }

    public record Zone(int pilotVRow, double pilotTargetV, int[] ctls) {
    }

    public record Controllers(List<CtlBus> ctls, List<Zone> zones,
                       IncrementalFlowControl.EsContext esCtx,
                       double minPlausibleTargetV, double maxPlausibleTargetV) {
    }

    /** True if every enabled secondary-voltage-control zone is representable: LOCAL or REMOTE
     *  generator VC with DISTINCT controlled buses (controllers <-> controlled is a BIJECTION —
     *  OLF's own zone system requires it: b = A*Jk^T is nControllers x nControlled and its LU
     *  needs a square matrix, so a shared controlled bus is outside OLF's effective scope too),
     *  and every REMOTE controller carries a real Q range (the RL pack's remote-entry condition —
     *  without an entry the pin state is invisible to the orchestrator). */
    public static boolean supported(LfNetwork network) {
        java.util.Set<LfBus> controlled = new java.util.HashSet<>();
        for (LfSecondaryVoltageControl svc : network.getEnabledSecondaryVoltageControls()) {
            for (LfBus controller : svc.getControllerBuses()) {
                var vc = controller.getGeneratorVoltageControl().orElse(null);
                if (vc == null) {
                    return false;
                }
                if (!controlled.add(vc.getControlledBus())) {
                    return false;                            // shared controlled bus — not bijective
                }
                if (vc.getControlledBus() != controller && !(controller.getMinQ() < controller.getMaxQ())) {
                    return false;                            // remote controller without an RL entry
                }
            }
        }
        return true;
    }

    @SuppressWarnings("unchecked")
    public static Controllers build(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                             int[] rlVrow, IncrementalFlowControl.EsContext esCtx,
                             double minPlausibleTargetV, double maxPlausibleTargetV,
                             GpuBatchElementMapper elementMapper) {
        Map<Integer, Integer> busVRows = new LinkedHashMap<>();
        Map<Integer, Integer> shuntBRows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVRows.put(v.getElementNum(), v.getRow());
            } else if (v.getType() == AcVariableType.SHUNT_B) {
                shuntBRows.put(v.getElementNum(), v.getRow());
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
                Integer ctldVRow = busVRows.get(vc.getControlledBus().getNum());
                Integer rlIdx = vRow != null ? rlIdxOfVRow.get(vRow) : null;
                if (vRow == null || ctldVRow == null || rlIdx == null) {
                    GpuBatchSkips.record(GpuBatchSkips.Slot.SVC, LOGGER,
                            "Secondary VC controller {} not representable on the GPU batch — skipped zone {}",
                            controller.getId(), svc.getZoneName());
                    return new Controllers(List.of(), List.of(), esCtx, minPlausibleTargetV, maxPlausibleTargetV);
                }
                // OLF calculateSensiQ: the controller bus's branch Q flows (side by attachment) + shunts.
                // qTerms drives the HOST decision (OLF's own EquationTerm.calculateSensi); qObs is the
                // same list addressed the way the DEVICE reaches it — pack-relative element indices, so
                // the in-kernel phase evaluates the identical quantities without re-deriving them.
                List<EquationTerm<AcVariableType, AcEquationType>> qTerms = new ArrayList<>();
                List<QObs> qObs = new ArrayList<>();
                for (LfBranch branch : controller.getBranches()) {
                    if (!branch.isDisabled() && branch.isConnectedAtBothSides()) {
                        int elem = elementMapper.getElementIndex(branch);
                        if (branch.getBus1() == controller) {
                            qTerms.add((EquationTerm<AcVariableType, AcEquationType>) branch.getQ1());
                            qObs.add(new QObs(0, elem, -1));
                        }
                        if (branch.getBus2() == controller) {
                            qTerms.add((EquationTerm<AcVariableType, AcEquationType>) branch.getQ2());
                            qObs.add(new QObs(1, elem, -1));
                        }
                    }
                }
                for (var shOpt : List.of(controller.getShunt(), controller.getControllerShunt(), controller.getSvcShunt())) {
                    if (shOpt.isPresent()) {
                        LfShunt sh = shOpt.get();
                        qTerms.add((EquationTerm<AcVariableType, AcEquationType>) sh.getQ());
                        Integer bRow = shuntBRows.get(sh.getNum());
                        qObs.add(new QObs(2, elementMapper.getElementIndex(sh), bRow != null ? bRow : -1));
                    }
                }
                zoneCtls.add(ctls.size());
                ctls.add(new CtlBus(vRow, ctldVRow, vc.getControlledBus().getNum(), rlIdx, vc.getTargetValue(),
                        controller.getMinQ(), controller.getMaxQ(),
                        controller.getLoadTargetQ(), controller.getQ(), qTerms, qObs));
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

    /**
     * Give the in-kernel zone regulation back to the network — the batch-of-one's analogue of what
     * OLF's {@code SecondaryVoltageControlOuterLoop} does as a side effect of running: it moves each
     * controller's generator voltage control TARGET, and without this OLF's world keeps the
     * pre-regulation targets while the device solved at the moved ones.
     *
     * <p>The moved target comes from the phase's OWN state, not from the controlled bus's row. Reading
     * the row looks right and is not: that row carries a Q target the moment the bus is pinned or is a
     * shared-distribution member, and writing 0.165 pu in as a target voltage puts the control outside
     * the plausible band and OLF drops it.
     */
    public static void writeBack(LfNetwork network, Controllers controllers, double[] movedTargetV) {
        if (controllers == null || movedTargetV == null) {
            return;
        }
        List<CtlBus> ctls = controllers.ctls();
        for (int c = 0; c < ctls.size() && c < movedTargetV.length; c++) {
            double moved = movedTargetV[c];                  // batch of one: scenario 0
            if (Double.isNaN(moved) || moved <= 0) {
                continue;                                    // the phase never adjusted this controller
            }
            LfBus ctld = network.getBus(ctls.get(c).ctldBusNum());
            if (ctld == null) {
                continue;
            }
            final double v = moved;
            if (System.getenv("OLF_GPU_SVC_WB_TRACE") != null) {
                System.err.printf("SVC_WB bus=%s base=%.6f moved=%.6f%n", ctld.getId(), ctls.get(c).baseTargetV(), v);
            }
            ctld.getGeneratorVoltageControl().ifPresent(vc -> vc.setTargetValue(v));
        }
    }

    private static double qToK(double q, CtlBus c) {         // OLF qToK
        return (2 * q - c.maxQ() - c.minQ()) / (c.maxQ() - c.minQ());
    }

    @SuppressWarnings({"java:S107", "java:S3776"})
    /** Install the zones for the IN-KERNEL SecondaryVoltageControl phase (OLF order 4). Returns true when
     *  the kernel owns the loop: the whole check — the re-enable step, the per-zone trigger, the
     *  sensitivities on the driver's own factors, the dense zone system and the moved targets — then runs
     *  inside the sweep at OLF's position, with no host round trip and no orchestrator after it. */
    public static boolean uploadToDevice(long handle, Controllers controllers) {
        if (controllers == null || controllers.ctls().isEmpty() || controllers.zones().isEmpty()) {
            SecondaryVoltageControlNative.setBatchSecondaryVc(handle, null, null, null, null, null, null, null,
                    null, null, null, null, null, null, null, null, 0, 0);
            return false;
        }
        List<CtlBus> ctls = controllers.ctls();
        int nc = ctls.size();
        int[] ctldVRow = new int[nc];
        int[] busVRow = new int[nc];
        int[] rlIdx = new int[nc];
        double[] baseTargetV = new double[nc];
        double[] minQ = new double[nc];
        double[] maxQ = new double[nc];
        double[] loadQ = new double[nc];
        int[] qPtr = new int[nc + 1];
        List<Integer> qKind = new ArrayList<>();
        List<Integer> qElem = new ArrayList<>();
        List<Integer> qBRow = new ArrayList<>();
        for (int c = 0; c < nc; c++) {
            CtlBus ctl = ctls.get(c);
            ctldVRow[c] = ctl.ctldVRow();
            busVRow[c] = ctl.vRow();
            rlIdx[c] = ctl.rlIdx();
            baseTargetV[c] = ctl.baseTargetV();
            minQ[c] = ctl.minQ();
            maxQ[c] = ctl.maxQ();
            loadQ[c] = ctl.loadTargetQ();
            qPtr[c] = qKind.size();
            for (QObs o : ctl.qObs()) {
                qKind.add(o.kind());
                qElem.add(o.elem());
                qBRow.add(o.bRow());
            }
        }
        qPtr[nc] = qKind.size();
        int nz = controllers.zones().size();
        int[] zonePtr = new int[nz + 1];
        List<Integer> zoneCtl = new ArrayList<>();
        int[] zonePilotRow = new int[nz];
        double[] zoneTargetV = new double[nz];
        for (int z = 0; z < nz; z++) {
            Zone zone = controllers.zones().get(z);
            zonePtr[z] = zoneCtl.size();
            for (int c : zone.ctls()) {
                zoneCtl.add(c);
            }
            zonePilotRow[z] = zone.pilotVRow();
            zoneTargetV[z] = zone.pilotTargetV();
        }
        zonePtr[nz] = zoneCtl.size();
        boolean owned = SecondaryVoltageControlNative.setBatchSecondaryVc(handle, ctldVRow, busVRow, rlIdx, baseTargetV,
                minQ, maxQ, loadQ, qPtr,
                qKind.stream().mapToInt(Integer::intValue).toArray(),
                qElem.stream().mapToInt(Integer::intValue).toArray(),
                qBRow.stream().mapToInt(Integer::intValue).toArray(),
                zonePtr, zoneCtl.stream().mapToInt(Integer::intValue).toArray(), zonePilotRow, zoneTargetV,
                controllers.minPlausibleTargetV(), controllers.maxPlausibleTargetV()) != 0;
        LOGGER.info("Uploaded {} secondary voltage-control zones ({} controllers) to device: {}",
                nz, nc, owned ? "in-kernel loop" : "host path");
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("SVC_UPLOAD nZones=%d nCtl=%d kernel=%b%n", nz, nc, owned);
        }
        return owned;
    }

    /** The sensitivity unit rows (each controlled bus's pinned V row) and the observed columns (every
     *  variable the controller-bus Q terms depend on, plus the pilot V rows). */
    static int[][] observables(Controllers controllers) {
        List<Integer> obsList = new ArrayList<>();
        Map<Integer, Integer> seen = new LinkedHashMap<>();
        int[] unit = new int[controllers.ctls().size()];
        for (int c = 0; c < controllers.ctls().size(); c++) {
            unit[c] = controllers.ctls().get(c).ctldVRow();
        }
        for (CtlBus c : controllers.ctls()) {
            for (EquationTerm<AcVariableType, AcEquationType> t : c.qTerms()) {
                for (Variable<AcVariableType> v : t.getVariables()) {
                    seen.computeIfAbsent(v.getRow(), r -> {
                        obsList.add(r);
                        return obsList.size() - 1;
                    });
                }
            }
        }
        for (Zone z : controllers.zones()) {
            seen.computeIfAbsent(z.pilotVRow(), r -> {
                obsList.add(r);
                return obsList.size() - 1;
            });
        }
        return new int[][] {unit, obsList.stream().mapToInt(Integer::intValue).toArray()};
    }

    /** What the driver pause hands back to the device: the moved V targets as FLAT target deltas
     *  ({@code scenario * n + row}), the per-(scenario, RL entry) voltage targets the switch-back must
     *  compare against, and the per-scenario "this loop changed something" flags OLF's UNSTABLE verdict
     *  maps to. */
    public record PauseDecision(int[] targetRows, double[] targetValues, int[] rlVtSlotIdx, double[] rlVtValues,
                                int[] unpinPairs, double[] unpinTargets, byte[] hostChanged, boolean anyMoved) {
    }

    /** The SecondaryVoltageControl check at the driver's OLF-order-4 pause: decide on the states the
     *  paused sweep produced and return the mutations, without re-solving — the driver's own resume does
     *  the solve, in OLF's order, with the distributed slack and the reactive-limit state it already
     *  holds. {@code targetV} carries the zone targets across pauses (OLF keeps them on the network). */
    @SuppressWarnings("java:S107")
    public static PauseDecision decideAtPause(long handle, int chunkSize, int realCount, int n, int nrl,
            double[][] states, Controllers controllers, int[] rlState, double[][] targetV, double[] baseState) {
        List<CtlBus> ctls = controllers.ctls();
        int nc = ctls.size();
        byte[] hostChanged = new byte[chunkSize];
        if (nc == 0 || rlState == null) {
            return new PauseDecision(null, null, null, null, null, null, hostChanged, false);
        }
        int[][] uo = observables(controllers);
        // the pin state BEFORE the decision: what decideRound clears is OLF's re-enable, and the device
        // only learns about it through the un-pin channel (a resume re-seeds no RL state)
        int[] pinnedBefore = rlState.clone();
        final double[] esStateOnEntry = controllers.esCtx().stateVector().get().clone();
        boolean[] affected;
        try {
            affected = decideRound(handle, chunkSize, realCount, n, nrl, states, controllers, rlState,
                    targetV, baseState, uo[0], uo[1], 0);
        } finally {
            controllers.esCtx().stateVector().set(esStateOnEntry);
        }
        List<Integer> rows = new ArrayList<>();
        List<Double> vals = new ArrayList<>();
        List<Integer> vtSlotIdx = new ArrayList<>();
        List<Double> vtVal = new ArrayList<>();
        List<Integer> unpin = new ArrayList<>();
        List<Double> unpinT = new ArrayList<>();
        for (int s = 0; s < realCount; s++) {
            hostChanged[s] = (byte) (affected != null && affected[s] ? 1 : 0);
            for (int c = 0; c < nc; c++) {
                CtlBus ctl = ctls.get(c);
                int e = s * nrl + ctl.rlIdx();
                if ((pinnedBefore[e] & 7) != 0 && (rlState[e] & 7) == 0) {
                    unpin.add(s);                            // OLF's re-enable: back to PV at the zone's target
                    unpin.add(ctl.rlIdx());
                    unpinT.add(targetV[s][c]);
                }
                // The switch-back comparison target is a SOLVE ARGUMENT, re-uploaded per call: it must be
                // rebuilt from the ACCUMULATED targets on every resume, not from this round's moves — a
                // round that moves nothing would otherwise drop the channel and the kernel would compare
                // against the packed base target (323 differing violations on rte6515).
                if (targetV[s][c] == ctl.baseTargetV()) {
                    continue;
                }
                if ((rlState[e] & 7) == 0) {
                    rows.add(s * n + ctl.ctldVRow());        // the CONTROLLED bus's V target moves now
                    vals.add(targetV[s][c]);
                }                                            // pinned row: the Q pin stays; the override below
                vtSlotIdx.add(s);                            // steers the RL switch-back and its restored target
                vtSlotIdx.add(ctl.rlIdx());
                vtVal.add(targetV[s][c]);
            }
        }
        return new PauseDecision(rows.stream().mapToInt(Integer::intValue).toArray(),
                vals.stream().mapToDouble(Double::doubleValue).toArray(),
                vtSlotIdx.stream().mapToInt(Integer::intValue).toArray(),
                vtVal.stream().mapToDouble(Double::doubleValue).toArray(),
                unpin.stream().mapToInt(Integer::intValue).toArray(),
                unpinT.stream().mapToDouble(Double::doubleValue).toArray(),
                hostChanged, affected != null);
    }

    /** ONE decision round of OLF's processSecondaryVoltageControl over the chunk: the per-zone trigger
     *  (pilot deadband, K spread), tryToReEnableHelpfulControllerBuses, the batched device sensitivities
     *  and the per-scenario zone system. Mutates {@code targetV} (the moved generator-VC targets) and
     *  {@code rlState} (re-enabled controllers lose their pin); returns the scenarios that moved, or
     *  {@code null} when nothing did. Shared by the host orchestrator and the driver-pause path so the
     *  two can never drift. */
    // a zone system that does not solve leaves that scenario unadjusted
    @SuppressWarnings({"java:S107", "java:S3776", "checkstyle:IllegalCatchError"})
    private static boolean[] decideRound(long handle, int chunkSize, int realCount, int n, int nrl,
            double[][] chunkResults, Controllers controllers, int[] rlState, double[][] targetV,
            double[] baseState, int[] unit, int[] obs, int round) {
        List<CtlBus> ctls = controllers.ctls();
        List<Zone> zones = controllers.zones();
        int nc = ctls.size();
        int m = obs.length;
        var esCtx = controllers.esCtx();
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
                    if ((rlState[s * nrl + ctls.get(c).rlIdx()] & 7) != 0) {
                        pinned++;
                    }
                }
                if (pinned > 0 && pinned < z.ctls().length) {
                    for (int c : z.ctls()) {
                        if ((rlState[s * nrl + ctls.get(c).rlIdx()] & 7) != 0) {
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
                if (System.getenv("OLF_GPU_SVC_TRACE") != null) {
                    System.err.println("SVC_TRACE round=" + round + " s=" + s + " pilotDv=" + pilotDv
                            + " kMin=" + kMin + " kMax=" + kMax);
                }
                if (Math.abs(pilotDv) > DV_EPS || !allAtLimits && Math.abs(kMax - kMin) > DK_DIFF_MAX_EPS) {
                    scenAdjusts[s] = true;
                    anyAdjusts = true;
                }
            }
        }
        if (System.getenv("OLF_GPU_SVC_TRACE") != null) {
            System.err.printf("SVC_ROUND round=%d anyAdjusts=%b%n", round, anyAdjusts);
        }
        if (!anyAdjusts) {
            return null;                                     // every zone inside its bands: OLF's STABLE
        }
        // ---- batched device sensitivities dState/dVtarget for every controlled bus ----
        double[] statesFlat = new double[chunkSize * n];
        for (int s = 0; s < chunkSize; s++) {
            double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
            System.arraycopy(src, 0, statesFlat, s * n, n);
        }
        double[] dxAtObs = SensitivityNative.solveBatchSensitivities(handle, statesFlat, unit, obs);

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
                if ((rlState[s * nrl + ctls.get(c).rlIdx()] & 7) == 0 || reEnabled[s][c]) {
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
            if (System.getenv("OLF_GPU_SVC_TRACE") != null) {
                System.err.printf("SVC_MOVED round=%d scenario=%d%n", round, s);
            }
            // apply the re-enables: clear the RL pins so the next solve resumes voltage control there
            for (int c = 0; c < nc; c++) {
                if (reEnabled[s][c]) {
                    int st = s * nrl + ctls.get(c).rlIdx();
                    rlState[st] = rlState[st] & ~3;
                }
            }
        }
        return anyMoved ? affected : null;
    }

}
