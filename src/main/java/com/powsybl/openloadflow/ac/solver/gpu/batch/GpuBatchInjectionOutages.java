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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Computes the per-scenario {@link GenOutage} device changes a generator/load contingency applies
 * (mirrors {@code LfContingency.processLostGenerators}), or signals a CPU fallback by returning
 * {@code null}. Groups the lost generators/loads by bus, sums each bus's target and distributed-slack
 * reweight deltas, and rejects (null) the shapes the on-device retain math cannot hold: the
 * slack/reference bus, a remote / distributed (DISTR_Q) voltage controller, more than one
 * distributed-slack-participant bus, or a factor reweight when the slack itself participates.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchInjectionOutages {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchInjectionOutages.class);

    private GpuBatchInjectionOutages() {
    }

    /** Merged per-BUS outages for a MULTI-generator contingency: gens grouped by bus, each bus's
     *  targets/DS deltas summed. Stage-1 constraint: at most ONE distributed-slack-participant bus
     *  per contingency (the native DS override carries one entry per scenario) — more, or any
     *  per-bus disqualification, returns null and the contingency routes to the CPU. */
    static GenOutage[] computeInjectionOutages(EquationSystem<AcVariableType, AcEquationType> es, GpuAcData gpuData,
                                               double dsSum, boolean slackParticipates,
                                               com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams dsParams,
                                               List<LfGenerator> gens, List<LoadLoss> loads) {
        java.util.Map<LfBus, List<LfGenerator>> byBus = new java.util.LinkedHashMap<>();
        java.util.Map<LfBus, List<LoadLoss>> loadsByBus = new java.util.LinkedHashMap<>();
        for (LfGenerator g : gens) {
            if (g.getBus() == null) {
                return null;
            }
            byBus.computeIfAbsent(g.getBus(), b -> new ArrayList<>()).add(g);
        }
        for (LoadLoss l : loads) {
            loadsByBus.computeIfAbsent(l.bus(), b -> new ArrayList<>()).add(l);
            byBus.computeIfAbsent(l.bus(), b -> new ArrayList<>());   // load-only bus: empty gen list
        }
        GenOutage[] out = new GenOutage[byBus.size()];
        int k = 0;
        int dsBuses = 0;
        for (java.util.Map.Entry<LfBus, List<LfGenerator>> e : byBus.entrySet()) {
            GenOutage go = computeBusGenOutage(es, gpuData, dsSum, slackParticipates, dsParams, e.getKey(), e.getValue(),
                    loadsByBus.getOrDefault(e.getKey(), List.of()));
            if (go == null) {
                return null;
            }
            if (go.dsBus() >= 0 && ++dsBuses > 1) {
                LOGGER.debug("Multi-injection contingency touches several DS-participant buses — CPU fallback");
                return null;
            }
            out[k++] = go;
        }
        return out;
    }

    private static GenOutage computeBusGenOutage(EquationSystem<AcVariableType, AcEquationType> es, GpuAcData gpuData,
                                                 double dsSum, boolean slackParticipates,
                                                 com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams dsParams, LfBus bus,
                                                 List<LfGenerator> lost, List<LoadLoss> lostLoads) {
        int busNum = bus.getNum();
        if (es.getEquation(busNum, AcEquationType.BUS_TARGET_PHI).isPresent()        // slack/reference: P is free
                || es.getEquation(busNum, AcEquationType.DISTR_Q).isPresent()) {     // remote/distributed VC controller
            return null;
        }
        if (dsParams.isLoadBased() && !lostLoads.isEmpty()) {
            return null;                                     // a lost/shifted load changes the LOAD participation
        }                                                    // factors — per-scenario load reweight not modeled: CPU
        java.util.Set<LfGenerator> lostSet = new java.util.HashSet<>(lost);
        int pRow = es.getVariableSet().getVariable(busNum, AcVariableType.BUS_PHI).getRow();
        int vRow = es.getVariableSet().getVariable(busNum, AcVariableType.BUS_V).getRow();
        if (pRow < 0 || vRow < 0) {
            return null;
        }
        // Is this bus a distributed-slack participant? Matters for BOTH shapes: lost generators
        // leave the participation (factor/headroom reweight), AND pure injection deltas (load
        // losses, redispatch actions) shift the row target — which the retain/waterfill accounting
        // would misread as already-distributed slack power unless dsBaseTargetP shifts with it
        // (measured: a +60 MW redispatch cancelled to a no-op before the base-shift fix).
        double preLostLoadP = lostLoads.stream().mapToDouble(LoadLoss::p).sum();
        int dsBus = -1;
        if (!lost.isEmpty() || preLostLoadP != 0) {
            int[] dsPhiRow = gpuData.dsPhiRow();
            for (int i = 0; i < dsPhiRow.length; i++) {
                if (dsPhiRow[i] == pRow) {
                    dsBus = i;
                    break;
                }
            }
        }
        boolean lostParticipation = lost.stream().anyMatch(g -> dsParams.factor(g) != 0);
        if (dsParams.isRemainingMargin() && dsBus >= 0 && lostParticipation) {
            return null;                                     // RM margins reweight per scenario is not modeled — CPU
        }
        if (dsBus >= 0 && lostParticipation && slackParticipates) {
            return null;                            // FACTOR reweight + participating slack: retain math doesn't hold
        }                                           // (a pure base shift keeps the factors — retain stays exact)
        if (bus.getGeneratorVoltageControl().map(vc -> vc.getControlledBus() != null
                && vc.getControlledBus().getNum() != busNum).orElse(false)) {
            return null;                                                            // remote voltage control
        }
        double lostP = lost.stream().mapToDouble(LfGenerator::getTargetP).sum();
        double lostLoadP = lostLoads.stream().mapToDouble(LoadLoss::p).sum();
        double lostLoadQ = lostLoads.stream().mapToDouble(LoadLoss::q).sum();
        double pTarget = bus.getTargetP() - lostP + lostLoadP;   // shedding load RAISES the P target
        boolean busPv = bus.isGeneratorVoltageControlEnabled();
        // A Q-carrying load delta at a voltage-controlling bus would make the RL pack's static
        // loadQ go stale — CPU fallback. P-only deltas (e.g. redispatch actions) are safe: PV Q is free.
        if (busPv && lostLoads.stream().anyMatch(l -> l.q() != 0)) {
            return null;
        }
        boolean lostControls = lost.stream()
                .anyMatch(g -> g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
        boolean otherController = bus.getGenerators().stream()
                .anyMatch(g -> !lostSet.contains(g) && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
        boolean toPq = busPv && lostControls && !otherController;
        // The bus Q TARGET must shed the lost gens' targetQ whenever the V row solves the reactive
        // equation in the scenario: on the PV->PQ flip AND on an ALREADY-PQ bus losing fixed-Q
        // generation (task #21 — the flip-only version silently kept the lost Q injected; measured
        // 0.27 kV voltage divergence on ieee14 for a 15-MVAr fixed-Q gen loss).
        boolean qOverride = toPq || !busPv;
        double qTarget = 0;
        if (qOverride) {
            double remainingGenQ = bus.getGenerators().stream().filter(g -> !lostSet.contains(g))
                    .mapToDouble(LfGenerator::getTargetQ).sum();
            qTarget = remainingGenQ - (bus.getLoadTargetQ() - lostLoadQ);
        }
        // Distributed-slack reweight for the affected bus. The bus's base target P always drops by the lost
        // gen's P. The participation factor / headroom / footroom drop only if the LOST gen itself participated
        // (mirrors the extractor, which sums these over participating gens with a non-zero droop).
        double dsFactorDelta = 0;
        double dsBaseDelta = 0;
        double dsMaxDelta = 0;
        double dsMinDelta = 0;
        if (dsBus >= 0) {
            dsBaseDelta = lostP - lostLoadP;                 // net DROP of the row target (load shed RAISES it)
            for (LfGenerator gen : lost) {                   // per-gen deltas compose additively
                double gf = dsParams.factor(gen);
                if (gf != 0) {
                    dsFactorDelta += gf / dsSum;
                    if (dsParams.useActiveLimits()) {        // rooms are +-infinity otherwise — a lost gen
                        double minTp = gen.getMinTargetP();  // cannot change them (and inf - inf = NaN)
                        double maxTp = gen.getMaxTargetP();
                        double tp = gen.getTargetP();
                        if (tp < 0) {
                            maxTp = Math.min(maxTp, 0);
                        } else {
                            minTp = Math.max(minTp, 0);
                        }
                        dsMaxDelta += maxTp - tp;
                        dsMinDelta += minTp - tp;
                    }
                }
            }
        }
        return new GenOutage(pRow, pTarget, vRow, qTarget, toPq, qOverride, dsBus, dsFactorDelta, dsBaseDelta, dsMaxDelta, dsMinDelta);
    }
}
