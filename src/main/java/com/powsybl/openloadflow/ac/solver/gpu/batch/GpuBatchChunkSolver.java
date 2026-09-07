/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfShunt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Drives the batched GPU solve of the qualified N-1 scenarios in device-memory-bounded chunks:
 * one cached native context (sized to the chunk, cuDSS analyzed once) reused across all chunks and
 * two phases (plain contingencies, then warm-continued (ctg, strategy) pairs). Assembles each
 * scenario's per-slot element mask, row-mode overrides (islanding freeze / PV->PQ flip / zi switch),
 * target/shunt/tap/distributed-slack overrides, runs the batched Newton, then the per-scenario
 * transformer-control orchestration, and writes results back by original contingency index.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchChunkSolver {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchChunkSolver.class);

    private GpuBatchChunkSolver() {
    }

    /**
     * Solve the qualified scenarios on the GPU in chunks sized by {@link GpuBatchContextCache#resolveChunkSize}
     * (auto from device memory, or a manual cap). ONE native context, sized to the chunk, is created and reused across all
     * chunks: cuDSS's symbolic ANALYSIS (on the shared pattern, identical for every chunk) is
     * done once on the first chunk and reused, so chunking costs nothing in amortization — it
     * only bounds the resident per-scenario LU factors. The last (partial) chunk is padded
     * with base-case scenarios (no element disabled): they converge at iteration 0 and their
     * results are discarded.
     */

    static void solveQualifiedInChunks(List<Integer> batchedIndices, List<BatchContingency> contingencies,
                                               GpuBatchElementMapper elementMapper, GpuAcData gpuData,
                                               double[] baseState, double[] baseTargets, int[] baseRowMode,
                                               int[][] frozenRowsByIndex, double[][] frozenTargetsByIndex,
                                               GenOutage[][] genOutagesByIndex, double[] activePowerLossByIndex,
                                               java.util.Map<com.powsybl.openloadflow.network.LfHvdc, Integer> hvdcElemIndex,
                                               double[][] rowCriteria, GpuBatchRlCurves.Channel rlCurves,
                                               int[][] dqOvContribByIndex, double[][] dqOvWByIndex,
                                               int[][] drOvContribByIndex, double[][] drOvCByIndex,
                                               int n, int nElements, NewtonRaphsonParameters parameters,
                                               double tol, double reactiveLimitTol, double slackDistTol,
                                               double convEpsPerEq, double[][] resultsByIndex, int[] varType,
                                               List<GpuBatchTransformerControl.TransformerController> transformerControllers,
                                               List<GpuBatchIncrementalTvc.Controller> incrementalTvcControllers,
                                               List<GpuBatchIncrementalShuntVc.Controller> incrementalShuntControllers,
                                               List<GpuBatchIncrementalFlowControl.Controller> incrementalFlowControllers,
                                               GpuBatchIncrementalFlowControl.EsContext esContext,
                                               List<GpuBatchHvdcAcEmulation.Ctl> hvdcControllers,
                                               GpuBatchSecondaryVc.Controllers svcControllers,
                                               List<GpuBatchVoltageMonitoring.Ctl> monitoringControllers,
                                               List<GpuBatchContinuousPhase.Ctl> continuousPhaseControllers,
                                               BatchConfig config) {
        int nb = batchedIndices.size();
        int chunkSize = GpuBatchContextCache.resolveChunkSize(nb, n, gpuData);
        int nChunks = (nb + chunkSize - 1) / chunkSize;

        // Reuse a cached analyzed context when the pattern/admittances/chunk/mode match (across
        // SA runs on the same network) — the context is NOT destroyed here, it stays cached.
        long handle = GpuBatchContextCache.acquireBatchContext(gpuData, chunkSize, GpuBatchContextCache.resolveBatchMode(), n);
        // PER_EQUATION_TYPE criterion: install the per-row thresholds on the (possibly cached)
        // context — or CLEAR a previous run's thresholds under UNIFORM_CRITERIA.
        GpuAcNewtonSolver.setBatchRowCriteria(handle,
                rowCriteria != null ? rowCriteria[0] : null, rowCriteria != null ? rowCriteria[1] : null);
        // Curve limits at runtime P: install (or clear, for cached contexts) the per-gen curve channel.
        GpuAcNewtonSolver.setBatchRlCurves(handle,
                rlCurves != null ? rlCurves.cvIdx() : null,
                rlCurves != null ? rlCurves.cvIn() : null,
                rlCurves != null ? rlCurves.cvPts() : null);
        // Remote-member RL entries (plan-B1): install (or clear, on a cached context) the kinds.
        boolean anyRlRemote = false;
        for (byte b : gpuData.rlRemote()) {
            anyRlRemote |= b != 0;
        }
        GpuAcNewtonSolver.setBatchRlRemote(handle,
                anyRlRemote ? gpuData.rlRemote() : null,
                anyRlRemote ? gpuData.rlCmpVrow() : null,
                anyRlRemote ? gpuData.rmqRow() : null,
                anyRlRemote ? gpuData.rmqElem() : null,
                anyRlRemote ? gpuData.rmqSide() : null,
                anyRlRemote ? gpuData.rmqKind() : null,
                anyRlRemote ? gpuData.rmqWeight() : null,
                anyRlRemote ? gpuData.dqKMember() : null,
                anyRlRemote ? gpuData.dqJMember() : null,
                anyRlRemote ? gpuData.dqMemberKey() : null,
                anyRlRemote ? gpuData.dqMemberRlEntry() : null,
                anyRlRemote ? gpuData.dqMemberGroupStart() : null,
                anyRlRemote ? gpuData.dqMemberGroupCount() : null,
                anyRlRemote ? gpuData.dqMemberRow() : null,
                anyRlRemote ? gpuData.dqMemberHasRow() : null,
                anyRlRemote ? gpuData.dqMemberQrGroup() : null);

        // The base state / targets are IDENTICAL for every scenario (a single-branch N-1 starts from the
        // same converged base operating point): passed as single length-n rows and tiled on-device. The
        // disabled mask is per-scenario (each chunk clears the one flag each slot set, sets the new one).
        // The ROW MODE is per-scenario too: a non-fragmenting outage shares the base row (same PV/slack
        // layout); a fragmenting one gets a cloned base row with its islanded buses frozen to identity.
        boolean[][] disabledMask = new boolean[chunkSize][nElements];
        int[][] disabledIdx = new int[chunkSize][];      // element indices each slot currently disables (1-2), null = none
        int[][] rowModePerScenario = new int[chunkSize][];   // per-scenario row mode (re-assigned every chunk)
        // Per-scenario target overrides (generator contingencies): flat (dTargetIndex = slot*n + row, value)
        // lists rebuilt per chunk. Empty when no generator outage is in the chunk.
        List<Integer> ovIndexList = new ArrayList<>();
        List<Double> ovValueList = new ArrayList<>();
        java.util.Map<Integer, Integer> ziElemByBranchNumChunk = new java.util.HashMap<>();
        for (int z = 0; z < gpuData.baseOpenZiElements().length; z++) {
            ziElemByBranchNumChunk.put(gpuData.baseOpenZiBranchNums()[z], gpuData.baseOpenZiElements()[z]);
        }
        int nziBase = gpuData.ziIdx().length / 11 - gpuData.baseOpenZiElements().length;
        java.util.Map<Integer, Integer> ziElemByNumAll = new java.util.HashMap<>();
        for (int z = 0; z < gpuData.ziIdx().length / 11; z++) {
            ziElemByNumAll.put(gpuData.ziBranchNums()[z], z);
        }
        List<Integer> ziOffSlotElemList = new ArrayList<>(); // per-scenario zi-element outages (switch opens)
        List<Integer> ziOnSlotElemList = new ArrayList<>(); // per-scenario zi-element enables (switch closes)
        List<Integer> shOvSlotElemList = new ArrayList<>();  // per-scenario shunt admittance overrides
        List<Double> shOvGbList = new ArrayList<>();
        List<Integer> r1OvSlotElemList = new ArrayList<>();  // per-scenario fixed-ratio tap overrides
        List<Double> r1OvValList = new ArrayList<>();
        int shPackOffset = elementMapper.getClosedBranchCount() + elementMapper.getOpenBranchCount();
        // Per-scenario distributed-slack reweight (a generator loss at a DS-participant bus): one affected DS
        // bus per slot (dsOvBus[s] = its DS-pack index, -1 = none) whose factor/base/headroom/footroom and the
        // factor sum shrink by the lost gen's share. Only allocated when the network has distributed slack.
        boolean hasDs = gpuData.dsPhiRow().length > 0;
        int[] dsOvBus = hasDs ? new int[chunkSize] : new int[0];
        double[] dsOvFactorDelta = hasDs ? new double[chunkSize] : new double[0];
        double[] dsOvBaseDelta = hasDs ? new double[chunkSize] : new double[0];
        double[] dsOvMaxDelta = hasDs ? new double[chunkSize] : new double[0];
        double[] dsOvMinDelta = hasDs ? new double[chunkSize] : new double[0];
        // TWO-PHASE ordering (operator strategies, warm continuation): plain contingencies first,
        // then (ctg, strategy) PAIRS — each pair chunk is seeded with its PARENT contingency's
        // solved state (x0PerScenario) and reactive-limits pinning (rlStateIn), so the strategy
        // solve continues the contingency's outer-loop trajectory exactly like the CPU's
        // per-contingency action solve. Pair whose parent did not converge is dropped (CPU path).
        List<Integer> ctgIdxList = new ArrayList<>();
        List<Integer> pairIdxList = new ArrayList<>();
        for (int idx : batchedIndices) {
            (contingencies.get(idx).contingencyId().indexOf(GpuBatchStatus.STRATEGY_KEY_SEP) >= 0 ? pairIdxList : ctgIdxList).add(idx);
        }
        java.util.Map<String, Integer> indexById = new java.util.HashMap<>();
        for (int i = 0; i < contingencies.size(); i++) {
            indexById.put(contingencies.get(i).contingencyId(), i);
        }
        int nrl = gpuData.rlVrow().length;
        java.util.Map<Integer, int[]> rlByIndex = new java.util.HashMap<>();    // ctg index -> its nrl RL-state slice
        // ctg index -> its FINAL distributed-slack targets at the DS participant rows (the device DS mutates
        // targets in place; read back after each ctg chunk). The pair phase re-anchors its distribution here,
        // mirroring OLF's setGeneratorsInitialTargetPToTargetP before the strategy re-solve.
        int nds = gpuData.dsPhiRow().length;
        java.util.Map<Integer, double[]> dsTargetsByIndex = new java.util.HashMap<>();
        java.util.Map<Integer, Double> slackAbsorbedByIndex = new java.util.HashMap<>();   // slack-only DS parents
        boolean dsSlackOnly = gpuData.dsPhiRow().length == 0 && gpuData.dsSlackPhiRow() >= 0;
        // Host-side base RL seed (mirrors initRlLimitTypeBatch) for PADDING slots of pair chunks.
        int[] baseRlSeed = new int[nrl];
        for (int i = 0; i < nrl; i++) {
            int vrow = gpuData.rlVrow()[i];
            // remote entries (plan-B3): base mode 1 = UNPINNED DISTR_Q share; only mode 4 is a pin.
            // The pilot's V equals the group target at the base (degenerate) — key the direction on
            // WHICH limit the extracted BUS_TARGET_Q target equals instead.
            boolean remote = gpuData.rlRemote().length > i && gpuData.rlRemote()[i] != 0;
            if (remote) {
                double tgt = baseTargets[vrow];
                baseRlSeed[i] = baseRowMode[vrow] == 4
                        ? (Math.abs(tgt - (gpuData.rlMaxQ()[i] - gpuData.rlLoadQ()[i]))
                           <= Math.abs(tgt - (gpuData.rlMinQ()[i] - gpuData.rlLoadQ()[i])) ? 2 : 1)
                        : 0;
            } else {
                baseRlSeed[i] = baseRowMode[vrow] == 1 ? (baseState[vrow] < gpuData.rlVtarget()[i] ? 2 : 1) : 0;
            }
        }
        try {
            long startTime = System.currentTimeMillis();
            for (List<Integer> group : List.of(ctgIdxList, pairIdxList)) {
                boolean pairPhase = group == pairIdxList;
                int nbg = group.size();
                for (int chunkStart = 0; chunkStart < nbg; chunkStart += chunkSize) {
                    int realCount = Math.min(chunkSize, nbg - chunkStart);
                    ovIndexList.clear();
                    ovValueList.clear();
                    List<Integer> dqOvIdxList = new ArrayList<>();   // {s, contrib} pairs (DISTR_Q reweight)
                    List<Double> dqOvWList = new ArrayList<>();
                    List<Integer> drOvIdxList = new ArrayList<>();   // {s, contrib} pairs (DISTR_RHO renorm, plan A5)
                    List<Double> drOvCList = new ArrayList<>();
                    // per-scenario ACTIVE equation counts (L2 threshold parity — the CPU removes an
                    // equation per frozen islanded row and per outaged element carrying a variable)
                    int[] nActive = new int[chunkSize];
                    java.util.Arrays.fill(nActive, n);
                    ziOffSlotElemList.clear();
                    ziOnSlotElemList.clear();
                    shOvSlotElemList.clear();
                    shOvGbList.clear();
                    r1OvSlotElemList.clear();
                    r1OvValList.clear();
                    if (hasDs) {
                        java.util.Arrays.fill(dsOvBus, -1);
                        java.util.Arrays.fill(dsOvFactorDelta, 0);
                        java.util.Arrays.fill(dsOvBaseDelta, 0);
                        java.util.Arrays.fill(dsOvMaxDelta, 0);
                        java.util.Arrays.fill(dsOvMinDelta, 0);
                    }
                    double[] preLossBySlot = new double[chunkSize];   // scheduled active power loss per slot (plain phase)
                    for (int s = 0; s < chunkSize; s++) {
                        if (disabledIdx[s] != null) {        // clear this slot's outage(s) from the previous chunk
                            for (int ei : disabledIdx[s]) {
                                disabledMask[s][ei] = false;
                            }
                            disabledIdx[s] = null;
                        }
                        for (int e : gpuData.baseOpenCbElements()) {
                            disabledMask[s][e] = true;           // base-open elements stay OUT unless this scenario recloses them
                        }
                        rowModePerScenario[s] = baseRowMode;     // default: base layout (padding + non-fragmenting)
                        if (s < realCount) {                 // real scenario: disable its outaged branch(es)
                            int ci = group.get(chunkStart + s);
                            if (!pairPhase && activePowerLossByIndex != null) {
                                preLossBySlot[s] = activePowerLossByIndex[ci];
                            }
                            if (dqOvContribByIndex != null && dqOvContribByIndex[ci] != null) {
                                for (int k = 0; k < dqOvContribByIndex[ci].length; k++) {
                                    dqOvIdxList.add(s);
                                    dqOvIdxList.add(dqOvContribByIndex[ci][k]);
                                    dqOvWList.add(dqOvWByIndex[ci][k]);
                                }
                            }
                            if (drOvContribByIndex != null && drOvContribByIndex[ci] != null) {
                                for (int k = 0; k < drOvContribByIndex[ci].length; k++) {
                                    drOvIdxList.add(s);
                                    drOvIdxList.add(drOvContribByIndex[ci][k]);
                                    drOvCList.add(drOvCByIndex[ci][k]);
                                }
                            }
                            List<LfBranch> branches = contingencies.get(ci).outagedBranches();
                            List<LfShunt> shunts = contingencies.get(ci).outagedShunts();
                            List<com.powsybl.openloadflow.network.LfHvdc> hvdcs = contingencies.get(ci).outagedHvdcs();
                            int[] eis = new int[branches.size() + shunts.size() + hvdcs.size()];
                            int ei = 0;
                            for (LfBranch branch : branches) {
                                int idx = elementMapper.getElementIndex(branch);          // qualification guarantees >= 0
                                disabledMask[s][idx] = true;
                                eis[ei++] = idx;
                            }
                            for (LfShunt shunt : shunts) {                                // fixed-shunt disconnection
                                int idx = elementMapper.getElementIndex(shunt);          // qualification guarantees >= 0
                                disabledMask[s][idx] = true;
                                eis[ei++] = idx;
                            }
                            for (com.powsybl.openloadflow.network.LfHvdc hvdc : hvdcs) { // AC-emulation HVDC outage
                                int idx = hvdcElemIndex.get(hvdc);                       // composition guarantees packed
                                disabledMask[s][idx] = true;
                                eis[ei++] = idx;
                            }
                            disabledIdx[s] = eis;
                            nActive[s] -= frozenRowsByIndex[ci].length;  // islanded buses' equations leave the CPU index
                            for (LfBranch branch : branches) {           // var-carrying outaged branches: the CPU drops
                                int eIdx = elementMapper.getElementIndex(branch);   // the rho/a1 identity equations too
                                if (eIdx >= 0 && eIdx < elementMapper.getClosedBranchCount()) {
                                    if (gpuData.cbIdx()[eIdx * 12 + 9] >= 0) {
                                        nActive[s]--;        // BRANCH_RHO1 variable + its identity leave the index
                                    }
                                    if (gpuData.cbIdx()[eIdx * 12 + 8] >= 0) {
                                        nActive[s]--;        // BRANCH_ALPHA1 variable + its identity leave the index
                                    }
                                }
                            }
                            int[] frozen = frozenRowsByIndex[ci];
                            if (frozen.length > 0) {         // fragmenting outage: freeze the islanded buses' rows
                                int[] row = baseRowMode.clone();
                                double[] frozenTargets = frozenTargetsByIndex[ci];
                                for (int k = 0; k < frozen.length; k++) {
                                    row[frozen[k]] = 0;      // identity row — bus decoupled from the main component
                                    // ...held at a CLEAN flat state (V=1, phi=0) via the target override, NOT the base
                                    // equation target (a P/Q value that gives a nonsensical islanded voltage/angle).
                                    ovIndexList.add(s * n + frozen[k]);
                                    ovValueList.add(frozenTargets[k]);
                                }
                                rowModePerScenario[s] = row;
                            }
                            for (LfBranch ziOut : contingencies.get(ci).outagedZiBranches()) {
                                Integer ze = ziElemByNumAll.get(ziOut.getNum());
                                ziOffSlotElemList.add(s);
                                ziOffSlotElemList.add(ze);
                                // the decoupled branch's ZERO_V/ZERO_PHI rows become identities again
                                // (dummies pinned to 0) — the inverse of a close
                                int dpRow = gpuData.ziIdx()[ze * 11 + 4];
                                int dqRow = gpuData.ziIdx()[ze * 11 + 5];
                                int[] rowZi = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                rowZi[dpRow] = 0;
                                rowZi[dqRow] = 0;
                                ovIndexList.add(s * n + dpRow);
                                ovValueList.add(0.0);
                                ovIndexList.add(s * n + dqRow);
                                ovValueList.add(0.0);
                                rowModePerScenario[s] = rowZi;
                            }
                            for (LfBranch reclosed : contingencies.get(ci).reclosedBranches()) {
                                if (reclosed.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC)) {
                                    Integer ze = ziElemByBranchNumChunk.get(reclosed.getNum());
                                    ziOnSlotElemList.add(s);
                                    ziOnSlotElemList.add(ze);
                                    // flip the two dummy rows from identity (pinned 0) to the ZERO_V/ZERO_PHI
                                    // coupling; targets stay 0 (switch: rho=1, phase shift 0)
                                    int dpRow = gpuData.ziIdx()[ze * 11 + 4];
                                    int dqRow = gpuData.ziIdx()[ze * 11 + 5];
                                    int[] row = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                    row[dpRow] = 1;
                                    row[dqRow] = 1;
                                    rowModePerScenario[s] = row;
                                } else {
                                    disabledMask[s][elementMapper.getElementIndex(reclosed)] = false;
                                }
                            }
                            for (TapR1Override tap : contingencies.get(ci).tapOverrides()) {
                                r1OvSlotElemList.add(s);
                                r1OvSlotElemList.add(elementMapper.getElementIndex(tap.branch()));   // cb pack index == element index
                                for (double v : tap.cbRow()) {
                                    r1OvValList.add(v);
                                }
                            }
                            for (ShuntShift sh : contingencies.get(ci).shuntShifts()) {
                                shOvSlotElemList.add(s);
                                shOvSlotElemList.add(elementMapper.getElementIndex(sh.shunt()) - shPackOffset);
                                shOvGbList.add(sh.dg());
                                shOvGbList.add(sh.db());
                            }
                            GenOutage[] gos = genOutagesByIndex[ci];
                            if (gos != null) {               // generator loss(es): per-BUS target overrides (merged by bus;
                                for (GenOutage go : gos) {   // at most one DS-participant bus per contingency by construction)
                                    ovIndexList.add(s * n + go.pRow());
                                    ovValueList.add(go.pTarget());
                                    if (go.qOverride()) {    // Q target sheds the lost gens' Q: on the PV->PQ flip AND
                                        ovIndexList.add(s * n + go.vRow());   // on an ALREADY-PQ bus losing fixed-Q gens (#21)
                                        ovValueList.add(go.qTarget());
                                    }
                                    if (go.toPq()) {
                                        int[] row = rowModePerScenario[s] == baseRowMode ? baseRowMode.clone() : rowModePerScenario[s];
                                        row[go.vRow()] = 1;  // BUS_V row becomes the reactive-power equation (PV→PQ)
                                        rowModePerScenario[s] = row;
                                    }
                                    if (hasDs && go.dsBus() >= 0) {   // lost gen(s) on a DS-participant bus: per-scenario reweight
                                        dsOvBus[s] = go.dsBus();
                                        dsOvFactorDelta[s] = go.dsFactorDelta();
                                        dsOvBaseDelta[s] = go.dsBaseDelta();
                                        dsOvMaxDelta[s] = go.dsMaxDelta();
                                        dsOvMinDelta[s] = go.dsMinDelta();
                                    }
                                }
                            }
                        }
                        // else: padding scenario = base case (no element disabled), discarded below
                    }
                    double[] dsAnchorShift = null;
                    double[] dsSlackShift = null;
                    if (!pairPhase && hasDs && config.contingencyLossPreDistribution()) {
                        // CPU parity (DefaultContingencyActivePowerLossDistribution): with DS on, the
                        // contingency's SCHEDULED lost active power is distributed ONCE over the
                        // participating generators BEFORE the post-contingency solve — the CPU's DS loop
                        // then only measures the loss-nonlinearity residual (usually inside the band, so
                        // it never redistributes). Replay the same one-shot water-fill host-side
                        // (GpuBatchLossPreDistribution) and hand it to the device as the strategy-pair
                        // trio: participant target overrides + anchor shifts + the slack's own share.
                        for (int s = 0; s < realCount; s++) {
                            if (preLossBySlot[s] == 0) {
                                continue;
                            }
                            boolean participantFrozen = false;   // islanded DS participant: CPU excludes a
                            int ci = group.get(chunkStart + s);  // disabled bus — out of stage-1 scope, skip
                            int[] frozen = frozenRowsByIndex[ci];
                            for (int i = 0; i < nds && !participantFrozen; i++) {
                                for (int fr : frozen) {
                                    if (fr == gpuData.dsPhiRow()[i]) {
                                        participantFrozen = true;
                                    }
                                }
                            }
                            if (participantFrozen) {
                                continue;
                            }
                            double[] shares = GpuBatchLossPreDistribution.distribute(gpuData, preLossBySlot[s],
                                    dsOvBus.length > 0 ? dsOvBus[s] : -1,
                                    dsOvFactorDelta.length > 0 ? dsOvFactorDelta[s] : 0,
                                    dsOvBaseDelta.length > 0 ? dsOvBaseDelta[s] : 0,
                                    dsOvMaxDelta.length > 0 ? dsOvMaxDelta[s] : 0,
                                    dsOvMinDelta.length > 0 ? dsOvMinDelta[s] : 0);
                            if (shares == null) {
                                continue;
                            }
                            if (dsAnchorShift == null) {
                                dsAnchorShift = new double[chunkSize * nds];
                                dsSlackShift = new double[chunkSize];
                            }
                            for (int i = 0; i < nds; i++) {
                                if (shares[i] == 0) {
                                    continue;
                                }
                                dsAnchorShift[s * nds + i] = shares[i];
                                int flat = s * n + gpuData.dsPhiRow()[i];
                                int ovAt = ovIndexList.lastIndexOf(flat);   // compose with a gen-outage override
                                if (ovAt >= 0) {
                                    ovValueList.set(ovAt, ovValueList.get(ovAt) + shares[i]);
                                } else {
                                    ovIndexList.add(flat);
                                    ovValueList.add(baseTargets[gpuData.dsPhiRow()[i]] + shares[i]);
                                }
                            }
                            dsSlackShift[s] = shares[nds];
                        }
                    }
                    // SIMPLE (WITH_GENERATOR) transformer VC: OLF's initialize() enables continuous VC
                    // BEFORE the first solve — apply the VC-ON activation deltas now, and keep the
                    // pre-TVC override lists for the round-to-tap re-solve.
                    // CONTINUOUS phase control: OLF's initialize() enables the CONTROLLER-mode controls
                    // BEFORE the first solve — keep the pre-enable override lists for the pinned re-solves.
                    List<Integer> prePcOvIdx = null;
                    List<Double> prePcOvVal = null;
                    if (continuousPhaseControllers != null && !continuousPhaseControllers.isEmpty()) {
                        prePcOvIdx = new ArrayList<>(ovIndexList);
                        prePcOvVal = new ArrayList<>(ovValueList);
                        GpuBatchContinuousPhase.applyPhaseOn(realCount, n, continuousPhaseControllers,
                                rowModePerScenario, baseRowMode, disabledMask, ovIndexList, ovValueList);
                    }
                    List<Integer> preTvcOvIdx = null;
                    List<Double> preTvcOvVal = null;
                    if (config.simpleTransformerControl() && transformerControllers != null && !transformerControllers.isEmpty()) {
                        preTvcOvIdx = new ArrayList<>(ovIndexList);
                        preTvcOvVal = new ArrayList<>(ovValueList);
                        GpuBatchSimpleTvc.applyVcOn(realCount, n, transformerControllers,
                                rowModePerScenario, baseRowMode, disabledMask, baseState, ovIndexList, ovValueList);
                    }
                    int[] targetOverrideIndex = ovIndexList.stream().mapToInt(Integer::intValue).toArray();
                    double[] targetOverrideValue = ovValueList.stream().mapToDouble(Double::doubleValue).toArray();
                    if (pairPhase && "1".equals(System.getenv("OLF_GPU_DBG_PAIR_OV"))) {
                        for (int s2 = 0; s2 < realCount; s2++) {
                            StringBuilder sb = new StringBuilder("GPU-PAIR-OV "
                                    + contingencies.get(group.get(chunkStart + s2)).contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|'));
                            for (int k2 = 0; k2 < targetOverrideIndex.length; k2++) {
                                if (targetOverrideIndex[k2] / n == s2) {
                                    int row = targetOverrideIndex[k2] % n;
                                    sb.append(String.format(" row%d=%.4f(base %.4f)", row, targetOverrideValue[k2], baseTargets[row]));
                                }
                            }
                            System.err.println(sb);
                        }
                        System.err.println("GPU-PAIR-SHOV slotElem=" + shOvSlotElemList + " gb=" + shOvGbList
                                + " shPackOffset=" + shPackOffset);
                    }
                    // Pair phase: warm continuation from each parent contingency's state + RL pinning.
                    double[] x0Full = null;
                    int[] rlStateIn = null;
                    // Strategy-leg DS re-anchoring: OLF's processContingency calls
                    // setGeneratorsInitialTargetPToTargetP() before the strategy re-solve, so the CPU's strategy
                    // leg distributes only the INCREMENT from the parent contingency's already-distributed
                    // targets. The device mirrors this with three coupled pieces (each alone is wrong —
                    // participant shifts without the slack shift leave the band check measured against the
                    // pre-distribution slack target, which kills the distribution):
                    //  - dsAnchorShift: participant water-fill anchors moved to the parent's final DS targets;
                    //  - dsSlackShift: the slack bus's own absorbed share of the parent's distribution
                    //    ((1-fs)/fs x the participants' absorbed sum), so the mismatch/band is measured against
                    //    the parent's slack target exactly as the CPU leg measures it;
                    //  - DS-row target seeds at the parent values, so the first inner solve runs at the CPU
                    //    leg's starting operating point (losses are nonlinear along the path).
                    if (pairPhase) {
                        x0Full = new double[chunkSize * n];
                        rlStateIn = nrl > 0 ? new int[chunkSize * nrl] : null;
                        if (nds > 0) {
                            dsAnchorShift = new double[chunkSize * nds];
                            dsSlackShift = new double[chunkSize];
                        } else if (dsSlackOnly) {
                            dsSlackShift = new double[chunkSize];    // absorbed-total seeds (no participant anchors)
                        }
                        double dsFactorSum = 0;
                        for (double fv : gpuData.dsFactor()) {
                            dsFactorSum += fv;
                        }
                        for (int s = 0; s < chunkSize; s++) {
                            double[] parent = null;
                            int[] parentRl = null;
                            double[] parentDs = null;
                            Double parentSlackAbs = null;
                            if (s < realCount) {
                                String pairId = contingencies.get(group.get(chunkStart + s)).contingencyId();
                                Integer pIdx = indexById.get(pairId.substring(0, pairId.indexOf(GpuBatchStatus.STRATEGY_KEY_SEP)));
                                if (pIdx != null && resultsByIndex[pIdx] != null
                                        && (int) resultsByIndex[pIdx][n + 2] == GpuBatchStatus.STATUS_CONVERGED) {
                                    parent = resultsByIndex[pIdx];
                                    parentRl = rlByIndex.get(pIdx);
                                    parentDs = dsTargetsByIndex.get(pIdx);
                                    parentSlackAbs = slackAbsorbedByIndex.get(pIdx);
                                }
                            }
                            for (int i = 0; i < n; i++) {
                                x0Full[s * n + i] = parent != null ? parent[i] : baseState[i];
                            }
                            if (rlStateIn != null) {
                                System.arraycopy(parentRl != null ? parentRl : baseRlSeed, 0, rlStateIn, s * nrl, nrl);
                            }
                            if (dsSlackOnly && dsSlackShift != null && parentSlackAbs != null) {
                                dsSlackShift[s] = parentSlackAbs;    // seed the pair with the parent's absorbed total
                            }
                            if (parentDs != null && dsAnchorShift != null && dsFactorSum > 0) {
                                // parentDs = [absorption per DS row..., parent factor sum] (captured after the
                                // parent's solve). The anchor shift is the parent's PURE ABSORPTION — the part
                                // of the parent's final DS target that its distribution added on top of the
                                // parent's own effective base — which composes with the pair's own dsOvBaseDelta
                                // in the kernel (base_eff = base - ovDelta + shift), so gen-outage parents and
                                // action deltas are both handled without double counting.
                                double absorbedSum = 0;
                                for (int i = 0; i < nds; i++) {
                                    int row = gpuData.dsPhiRow()[i];
                                    double shift = parentDs[i];
                                    dsAnchorShift[s * nds + i] = shift;
                                    absorbedSum += shift;
                                    // Seed the DS-row target at the CPU strategy leg's starting value — the
                                    // parent-absorbed target plus the pair's own composed delta — so the first
                                    // inner solve runs at the CPU leg's starting operating point (losses are
                                    // nonlinear along the path). A pair override on the row (gen action /
                                    // gen-outage pTarget) is base-referenced: add the parent absorption to it.
                                    int flat = s * n + row;
                                    int ovAt = ovIndexList.lastIndexOf(flat);
                                    if (ovAt >= 0) {
                                        ovValueList.set(ovAt, ovValueList.get(ovAt) + shift);
                                    } else {
                                        ovIndexList.add(flat);
                                        ovValueList.add(baseTargets[row] + shift);
                                    }
                                }
                                // the slack's own absorption in the parent's distribution, proportional to the
                                // participants' share at the PARENT's factor sum (reweighted if the parent lost
                                // a participating generator)
                                double parentFs = parentDs[nds];
                                dsSlackShift[s] = parentFs > 0 ? (1.0 - parentFs) / parentFs * absorbedSum : 0.0;
                            }
                        }
                        targetOverrideIndex = ovIndexList.stream().mapToInt(Integer::intValue).toArray();
                        targetOverrideValue = ovValueList.stream().mapToDouble(Double::doubleValue).toArray();
                    }
                    // HVDC AC-emulation: snapshot the pre-freeze override lists (the round rebuild base),
                    // then OLF's FreezingHvdcACEmulation freezes every live hvdc at its scenario's
                    // WARM-START flows BEFORE the first solve.
                    List<Integer> preHvdcOvIdx = null;
                    List<Double> preHvdcOvVal = null;
                    int[][] hvdcStatus = null;
                    if (hvdcControllers != null && !hvdcControllers.isEmpty()) {
                        preHvdcOvIdx = new ArrayList<>(ovIndexList);
                        preHvdcOvVal = new ArrayList<>(ovValueList);
                        if (config.hvdcFreezing()) {
                            hvdcStatus = GpuBatchHvdcAcEmulation.applyFreeze(chunkSize, realCount, n, hvdcControllers,
                                    esContext, baseState, x0Full, disabledMask, baseTargets, ovIndexList, ovValueList);
                            targetOverrideIndex = ovIndexList.stream().mapToInt(Integer::intValue).toArray();
                            targetOverrideValue = ovValueList.stream().mapToDouble(Double::doubleValue).toArray();
                        } else {
                            hvdcStatus = new int[chunkSize][hvdcControllers.size()];
                        }
                    }
                    boolean dbgPairRl = pairPhase && nrl > 0 && "1".equals(System.getenv("OLF_GPU_DBG_PAIR_RL"));
                    boolean svcActive = svcControllers != null && !svcControllers.ctls().isEmpty();
                    boolean monActive = monitoringControllers != null && !monitoringControllers.isEmpty();
                    int[] rlStateOut = (!pairPhase && !pairIdxList.isEmpty() || dbgPairRl || svcActive || monActive) && nrl > 0 ? new int[chunkSize * nrl] : null;
                    // DISTR_Q weight overrides: per-chunk context state (orchestrator re-solves keep
                    // the slot assignment); cleared when the chunk has none.
                    GpuAcNewtonSolver.setBatchDqWeightOverrides(handle,
                            dqOvIdxList.isEmpty() ? null : dqOvIdxList.stream().mapToInt(Integer::intValue).toArray(),
                            dqOvWList.isEmpty() ? null : dqOvWList.stream().mapToDouble(Double::doubleValue).toArray());
                    GpuAcNewtonSolver.setBatchDrCoefOverrides(handle,
                            drOvIdxList.isEmpty() ? null : drOvIdxList.stream().mapToInt(Integer::intValue).toArray(),
                            drOvCList.isEmpty() ? null : drOvCList.stream().mapToDouble(Double::doubleValue).toArray());
                    GpuAcNewtonSolver.setBatchActiveRowCounts(handle, nActive);
                    double[][] chunkResults = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                            rowModePerScenario, baseRowMode, disabledMask, targetOverrideIndex, targetOverrideValue,
                            dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorShift, dsSlackShift,
                            null, null,
                            parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                            config.lsMode(), config.maxDv(), config.maxDphi(),
                            config.lsMode() == 2 ? varType : null, x0Full,
                            shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                            rlStateIn, rlStateOut,
                            r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                            config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(),
                            // monitoring active: OLF checks VoltageMonitoring BEFORE ReactiveLimits — the main
                            // solve surfaces the DS-stable / pre-RL state; the orchestrator then interleaves
                            // and finishes with the full RL-enabled solves.
                            monActive ? 1 : 0,
                            nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                            ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray());
                    if (dbgPairRl) {
                        for (int s2 = 0; s2 < realCount; s2++) {
                            StringBuilder sb = new StringBuilder("GPU-PAIR-RL " + contingencies.get(group.get(chunkStart + s2)).contingencyId().replace(GpuBatchStatus.STRATEGY_KEY_SEP, '|'));
                            for (int i2 = 0; i2 < nrl; i2++) {
                                int v = rlStateOut[s2 * nrl + i2];
                                if ((v & 3) != 0) {
                                    sb.append(" ctl").append(i2).append("=lt").append(v & 3).append("/sc").append(v >> 2);
                                }
                            }
                            System.err.println(sb);
                        }
                    }
                    if (rlStateOut != null && !pairPhase) {      // capture each ctg's final RL pinning for its pairs
                        for (int s = 0; s < realCount; s++) {
                            int[] slice = new int[nrl];
                            System.arraycopy(rlStateOut, s * nrl, slice, 0, nrl);
                            rlByIndex.put(group.get(chunkStart + s), slice);
                        }
                    }
                    if (!pairPhase && dsSlackOnly && !pairIdxList.isEmpty()) {
                        // capture each ctg's slack-only-DS absorbed total for its pairs' seeding
                        double[] absorbed = GpuAcNewtonSolver.readBatchSlackAbsorbed(handle);
                        if (absorbed != null) {
                            for (int s = 0; s < realCount; s++) {
                                slackAbsorbedByIndex.put(group.get(chunkStart + s), absorbed[s]);
                            }
                        }
                    }
                    if (!pairPhase && nds > 0 && !pairIdxList.isEmpty()) {
                        // Capture each ctg's DS ABSORPTION (final distributed target minus the ctg's own
                        // effective base per DS row) plus its factor sum, for its pairs' re-anchoring —
                        // the device analogue of OLF's setGeneratorsInitialTargetPToTargetP before a
                        // strategy re-solve.
                        double[] finalTargets = GpuAcNewtonSolver.readBatchTargets(handle);
                        for (int s = 0; s < realCount; s++) {
                            double[] slice = new double[nds + 1];
                            double fs = 0;
                            for (int i = 0; i < nds; i++) {
                                int row = gpuData.dsPhiRow()[i];
                                double effBase = baseTargets[row]
                                        - (dsOvBus[s] == i ? dsOvBaseDelta[s] : 0.0);
                                slice[i] = finalTargets[s * n + row] - effBase;
                                fs += gpuData.dsFactor()[i]
                                        - (dsOvBus[s] == i ? dsOvFactorDelta[s] : 0.0);
                            }
                            slice[nds] = fs;
                            dsTargetsByIndex.put(group.get(chunkStart + s), slice);
                        }
                    }
                    if (incrementalTvcControllers != null && !incrementalTvcControllers.isEmpty()) {
                        chunkResults = GpuBatchIncrementalTvc.orchestrate(handle, chunkSize, realCount, n,
                                chunkResults, incrementalTvcControllers, rowModePerScenario, baseRowMode, disabledMask,
                                targetOverrideIndex, targetOverrideValue,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                dsAnchorShift, dsSlackShift,
                                r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                config.lsMode() == 2 ? varType : null, config);
                    }
                    if (hvdcControllers != null && !hvdcControllers.isEmpty()) {
                        chunkResults = GpuBatchHvdcAcEmulation.orchestrate(handle, chunkSize, realCount, n,
                                chunkResults, hvdcControllers, hvdcStatus, config.hvdcLimits(), esContext,
                                rowModePerScenario, baseRowMode, disabledMask,
                                preHvdcOvIdx, preHvdcOvVal,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                dsAnchorShift, dsSlackShift,
                                r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                config.lsMode() == 2 ? varType : null, config);
                    }
                    if (hvdcControllers != null && !hvdcControllers.isEmpty()) {
                        chunkResults = GpuBatchHvdcAcEmulation.orchestrate(handle, chunkSize, realCount, n,
                                chunkResults, hvdcControllers, hvdcStatus, config.hvdcLimits(), esContext,
                                rowModePerScenario, baseRowMode, disabledMask,
                                preHvdcOvIdx, preHvdcOvVal,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                dsAnchorShift, dsSlackShift,
                                r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                config.lsMode() == 2 ? varType : null, config);
                    }
                    if (monActive && rlStateOut != null) {
                        chunkResults = GpuBatchVoltageMonitoring.orchestrate(handle, chunkSize, realCount, n, nrl,
                                chunkResults, monitoringControllers, rlStateOut,
                                rowModePerScenario, baseRowMode, disabledMask,
                                targetOverrideIndex, targetOverrideValue,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                dsAnchorShift, dsSlackShift,
                                r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                config.lsMode() == 2 ? varType : null, config);
                    }
                    if (svcActive && rlStateOut != null) {
                        chunkResults = GpuBatchSecondaryVc.orchestrate(handle, chunkSize, realCount, n, nrl,
                                chunkResults, svcControllers, rlStateOut,
                                rowModePerScenario, baseRowMode, disabledMask,
                                targetOverrideIndex, targetOverrideValue,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                dsAnchorShift, dsSlackShift,
                                r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                config.lsMode() == 2 ? varType : null, config);
                    }
                    if (continuousPhaseControllers != null && !continuousPhaseControllers.isEmpty()) {
                        chunkResults = GpuBatchContinuousPhase.orchestrate(handle, chunkSize, realCount, n,
                                chunkResults, continuousPhaseControllers, esContext,
                                rowModePerScenario, baseRowMode, disabledMask,
                                prePcOvIdx, prePcOvVal,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                dsAnchorShift, dsSlackShift,
                                r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                config.lsMode() == 2 ? varType : null, config);
                    }
                    if (config.simpleTransformerControl() && transformerControllers != null && !transformerControllers.isEmpty()) {
                        chunkResults = GpuBatchSimpleTvc.roundAndResolve(handle, chunkSize, realCount, n,
                                chunkResults, transformerControllers, rowModePerScenario, baseRowMode, disabledMask,
                                preTvcOvIdx, preTvcOvVal,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                dsAnchorShift, dsSlackShift,
                                nds > 0 ? gpuData.dsPhiRow() : null, nds > 0 ? gpuData.dsFactor() : null,
                                r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                config.lsMode() == 2 ? varType : null, config);
                    }
                    if (incrementalFlowControllers != null && !incrementalFlowControllers.isEmpty()) {
                        chunkResults = GpuBatchIncrementalFlowControl.orchestrate(handle, chunkSize, realCount, n,
                                chunkResults, incrementalFlowControllers, esContext, rowModePerScenario, baseRowMode, disabledMask,
                                targetOverrideIndex, targetOverrideValue,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                dsAnchorShift, dsSlackShift,
                                r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                config.lsMode() == 2 ? varType : null, config);
                    }
                    if (incrementalShuntControllers != null && !incrementalShuntControllers.isEmpty()) {
                        chunkResults = GpuBatchIncrementalShuntVc.orchestrate(handle, chunkSize, realCount, n,
                                chunkResults, incrementalShuntControllers, rowModePerScenario, baseRowMode, disabledMask,
                                targetOverrideIndex, targetOverrideValue,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                dsAnchorShift, dsSlackShift,
                                r1OvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                r1OvValList.stream().mapToDouble(Double::doubleValue).toArray(),
                                shOvSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                shOvGbList.stream().mapToDouble(Double::doubleValue).toArray(),
                                nziBase, ziOnSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                ziOffSlotElemList.stream().mapToInt(Integer::intValue).toArray(),
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                                config.lsMode() == 2 ? varType : null, config);
                    }
                    if (config.transformerControl() && transformerControllers != null && !transformerControllers.isEmpty()) {
                        chunkResults = GpuBatchTransformerControl.orchestrateTransformerControl(handle, chunkSize, realCount, n,
                                chunkResults, transformerControllers, rowModePerScenario, baseRowMode, disabledMask,
                                targetOverrideIndex, targetOverrideValue,
                                dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta,
                                nds > 0 ? gpuData.dsPhiRow() : null, nds > 0 ? gpuData.dsFactor() : null,
                                drOvIdxList.isEmpty() ? null : drOvIdxList.stream().mapToInt(Integer::intValue).toArray(),
                                drOvCList.isEmpty() ? null : drOvCList.stream().mapToDouble(Double::doubleValue).toArray(),
                                gpuData.drRow(),
                                baseState, baseTargets, parameters, tol, reactiveLimitTol, slackDistTol, convEpsPerEq, varType,
                                config);
                    }
                    for (int s = 0; s < realCount; s++) {
                        resultsByIndex[group.get(chunkStart + s)] = chunkResults[s];
                    }
                }
            }
            long elapsed = System.currentTimeMillis() - startTime;
            LOGGER.info("Batched N-1: {} GPU scenarios in {} chunk(s) of <= {} solved in {} ms ({} ms/scenario)",
                    nb, nChunks, chunkSize, elapsed, (double) elapsed / nb);
        } catch (RuntimeException e) {
            GpuBatchContextCache.clearContextCache();   // a failed solve may leave the context inconsistent — rebuild next time
            throw e;
        }
    }
}
