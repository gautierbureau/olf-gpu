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

    /** OLF's AbstractLfBus.getLimitQ: a bus's reactive aggregate takes each generator's LIMIT when it
     *  controls voltage (or remote reactive power) and its TARGET Q otherwise — the aggregate is NOT a
     *  plain sum of getMinQ()/getMaxQ(). A fixed-Q generator lost from a bus its siblings keep in
     *  voltage control therefore takes its targetQ off the aggregate, not its limit; charging the limit
     *  instead moved rte6515 GEN-2899 by 0.32 kV and three violations, on a bus whose own limit is
     *  1.0 pu against a bus aggregate of 0.997. */
    private static double qContrib(LfGenerator g, java.util.function.ToDoubleFunction<LfGenerator> limit) {
        return g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE
                || g.getGeneratorControlType() == LfGenerator.GeneratorControlType.REMOTE_REACTIVE_POWER
                ? limit.applyAsDouble(g) : g.getTargetQ();
    }

    /** WHY one bus's injection outage could not be represented. All twelve exits share the
     *  PLAN:INJECTION_OUTAGE_SHAPE census tag, which is too coarse to pick the next thing to fix;
     *  OLF_GPU_GENOUT_REJECT=1 names the exit and the bus. */
    private static <T> T reject(String why, LfBus bus) {
        if (System.getenv("OLF_GPU_GENOUT_REJECT") != null) {
            System.err.println("GENOUT_REJECT " + why + " bus=" + (bus == null ? "?" : bus.getId()));
        }
        return null;
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
                return reject("GEN_WITHOUT_BUS", null);
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
                return null;                                 // the per-bus exit already named itself
            }
            if (go.dsBus() >= 0 && ++dsBuses > 1) {
                LOGGER.debug("Multi-injection contingency touches several DS-participant buses — CPU fallback");
                return reject("MULTI_DS_BUS", e.getKey());
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
        if (es.getEquation(busNum, AcEquationType.BUS_TARGET_PHI).isPresent()) {     // slack/reference: P is free
            return reject("SLACK_BUS", bus);
        }
        // DISTR_Q controller bus (remote/distributed VC): the qPct weights are BAKED into the
        // equation system at creation — the CPU does not rebuild them on a gen loss either. When the
        // bus KEEPS another voltage-controlling generator the group is unchanged on both arms and the
        // outage is a plain P-target drop (no PV->PQ flip, no Q override — the distribution absorbs);
        // losing the bus's LAST controller restructures the group -> CPU.
        boolean distrQBus = es.getEquation(busNum, AcEquationType.DISTR_Q).isPresent();
        if (distrQBus) {
            java.util.Set<LfGenerator> lostHere = new java.util.HashSet<>(lost);
            boolean keepsController = bus.getGenerators().stream()
                    .anyMatch(g -> !lostHere.contains(g)
                            && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
            if (!keepsController || !lostLoads.isEmpty()) {   // group restructure / Q-carrying load shape — CPU
                // The load-loss half is NOT closed by the per-scenario load target Q. Measured on
                // rte6515 LOAD-1263 with the gate lifted: the channel fires (the load carries
                // -0.092 pu and the delta is applied), every violation still matches, and the bus is
                // 0.105 kV out — local to the outage bus. That bus is a DISTR_Q group's pilot AND one
                // of its controllers, so its reactive accounting runs through the relocation path
                // (qBuf / pilotQ), which reads a load Q the per-scenario channel does not reach.
                // Violation parity alone would have called this fixed; the 1e-12 bus diff is what
                // caught it.
                return reject(keepsController ? "DISTR_Q_BUS_WITH_LOAD_LOSS" : "DISTR_Q_LAST_CONTROLLER", bus);
            }
        }
        if (bus.hasGeneratorsWithSlope()) {               // slope-SVC bus outage: the mode-2 row's fate on a
            return reject("SLOPE_SVC_BUS", bus);             // gen loss is not modeled — CPU
        }
        if (dsParams.isLoadBased() && !lostLoads.isEmpty()) {   // a lost/shifted load changes the LOAD
            return reject("LOAD_BASED_DS_LOAD_LOSS", bus);   // participation factors — not modeled: CPU
        }
        java.util.Set<LfGenerator> lostSet = new java.util.HashSet<>(lost);
        int pRow = es.getVariableSet().getVariable(busNum, AcVariableType.BUS_PHI).getRow();
        int vRow = es.getVariableSet().getVariable(busNum, AcVariableType.BUS_V).getRow();
        if (pRow < 0 || vRow < 0) {
            return reject("NO_P_OR_V_ROW", bus);
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
            return reject("REMAINING_MARGIN_DS", bus);       // RM margins reweight per scenario: not modeled — CPU
        }
        if (dsBus >= 0 && lostParticipation && slackParticipates) {
            return reject("PARTICIPATING_SLACK", bus);   // FACTOR reweight + participating slack: the retain math
        }                                           // doesn't hold (a pure base shift keeps the factors — exact)
        if (!distrQBus && bus.getGeneratorVoltageControl().map(vc -> vc.getControlledBus() != null
                && vc.getControlledBus().getNum() != busNum).orElse(false)) {
            return reject("REMOTE_SINGLE_CONTROLLER", bus);   // (pilot-Q remap, no DISTR_Q row at this bus):
        }                                                        // its q accounting is not modeled — CPU
        double lostP = lost.stream().mapToDouble(LfGenerator::getTargetP).sum();
        double lostLoadP = lostLoads.stream().mapToDouble(LoadLoss::p).sum();
        double lostLoadQ = lostLoads.stream().mapToDouble(LoadLoss::q).sum();
        double pTarget = bus.getTargetP() - lostP + lostLoadP;   // shedding load RAISES the P target
        boolean busPv = bus.isGeneratorVoltageControlEnabled();
        boolean lostControls = lost.stream()
                .anyMatch(g -> g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
        boolean otherController = bus.getGenerators().stream()
                .anyMatch(g -> !lostSet.contains(g) && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
        boolean toPq = busPv && lostControls && !otherController;
        // distrQBus with keepsController: otherController is true by the guard above, so toPq stays
        // false and qOverride below stays false (busPv true) — the group's distribution re-balances
        // the lost reactive share on both arms; only the P target drop and DS reweight apply.
        // The bus Q TARGET must shed the lost gens' targetQ whenever the V row solves the reactive
        // equation in the scenario: on the PV->PQ flip AND on an ALREADY-PQ bus losing fixed-Q
        // generation (task #21 — the flip-only version silently kept the lost Q injected; measured
        // 0.27 kV voltage divergence on ieee14 for a 15-MVAr fixed-Q gen loss).
        boolean qOverride = toPq || !busPv;
        double qTarget = 0;
        if (qOverride) {
            // A bus the BASE loadflow pinned at a reactive limit has its generation target Q FROZEN at
            // that limit, and OLF keeps that value as long as a VOLTAGE-type generator survives:
            // processLostGenerators calls setGeneratorVoltageControlEnabledAndRecomputeTargetQ only when
            // none does, and invalidateGenerationTargetP leaves a frozen bus frozen. Summing the
            // survivors' own targetQ instead hands the first inner solve a target 1.7 pu off the one
            // OLF solves (measured at rte6515 VL-6003_0: 1.03 vs 2.69 pu) — the reactive-limit pass
            // then makes its pin/unpin decisions along a trajectory the CPU never takes. When the last
            // VOLTAGE generator goes, OLF unfreezes and recomputes over what is left, which is what the
            // survivor sum gives (rte6515 GEN-6095/6130/6226/6239, one generator on the bus: keeping the
            // frozen value there injects the dead generator's reactive limit, ~1 kV).
            double genQ = bus.isGenerationTargetQFrozen() && otherController
                    ? bus.getGenerationTargetQ()
                    : bus.getGenerators().stream().filter(g -> !lostSet.contains(g))
                            .mapToDouble(LfGenerator::getTargetQ).sum();
            qTarget = genQ - (bus.getLoadTargetQ() - lostLoadQ);
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
        // A bus that KEEPS voltage control still carries the lost generators' reactive capability in the
        // device's run-constant aggregate, while OLF's bus.getMinQ()/getMaxQ() sum only the generators
        // that are still there. Name the reactive-limit entry and how much to take off it. (A bus the
        // outage takes to PQ needs nothing: its entry stops being a controller and the limits are never
        // read.) A bus the BASE loadflow already PINNED at a limit is NOT that case, though it reads
        // busPv=false: its entry is still a controller, its pinned target Q IS the limit, and it can
        // switch back to PV. Measured on rte6515 GEN-6003 — one generator of three lost at a pinned
        // bus, the stale capability injected as Q, and 1.9 kV network-wide.
        int rlEntry = -1;
        double minQDelta = 0;
        double maxQDelta = 0;
        int[] rlBusNum = gpuData.rlBusNum();
        for (int i = 0; rlBusNum != null && i < rlBusNum.length; i++) {
            if (rlBusNum[i] == bus.getNum()) {
                rlEntry = i;
                break;
            }
        }
        double loadQDelta = 0;
        if (rlEntry >= 0 && !toPq) {
            minQDelta = -lost.stream().mapToDouble(g -> qContrib(g, LfGenerator::getMinQ)).sum();
            maxQDelta = -lost.stream().mapToDouble(g -> qContrib(g, LfGenerator::getMaxQ)).sum();
            // The pack's loadQ is the bus's load target Q as EXTRACTED, and the reactive-limit
            // machinery reads it in two paired forms: the pinned target is limit - loadQ, and the
            // controller's own q is the net injection + loadQ. Shed a Q-carrying load and both go
            // stale. This used to be a CPU gate for a PV bus; the bus the base loadflow had already
            // PINNED fell through it and was silently wrong instead (rte6515 LOAD-6157, 0.37 kV).
            loadQDelta = -lostLoadQ;
        }
        // No generator left at all. The reactive-limit entry must stop being a controller for this
        // scenario: a bus the BASE loadflow had already pinned PQ keeps a seeded limit type so a
        // relieving contingency can switch it back to PV — and with its only generator gone there is
        // nothing to switch back to. OLF simply has no controller there. (Measured on rte6515 GEN-6130:
        // 0.995 kV at that node against 2.2e-4 everywhere else, and one apparent-power violation.)
        boolean allGensLost = bus.getGenerators().stream().allMatch(lostSet::contains);
        if (System.getenv("OLF_GPU_GENOUT_TRACE") != null) {
            System.err.printf("GENOUT bus=%s busPv=%b frozenQ=%b toPq=%b qOverride=%b lost=%s busGens=%d rlEntry=%d "
                    + "dMinQ=%.6f dMaxQ=%.6f busMinQ=%.6f busMaxQ=%.6f distrQ=%b ctrlBus=%s dsBus=%d dFactor=%.9f dBase=%.9f lostLoadQ=%.9f dLoadQ=%.9f%n",
                    bus.getId(), busPv, bus.isGenerationTargetQFrozen(), toPq, qOverride,
                    lost.stream().map(LfGenerator::getId).toList(), bus.getGenerators().size(),
                    rlEntry, minQDelta, maxQDelta, bus.getMinQ(), bus.getMaxQ(), distrQBus,
                    bus.getGeneratorVoltageControl().map(vc -> vc.getControlledBus().getId()).orElse("-"),
                    dsBus, dsFactorDelta, dsBaseDelta, lostLoadQ, loadQDelta);
        }
        // Identity of what was lost, for the reactive-capability CURVE channel: it evaluates each
        // generator's curve at its OWN post-DS active power, which it reconstructs from the bus's row
        // target by participation share. The lost generator's entry has to go, the survivors' shares
        // have to renormalize over what is left, and the step the outage itself puts in the row target
        // is not DS power and must not move any curve. All three need per-generator identity.
        java.util.Set<String> lostGenIds = lost.stream().map(LfGenerator::getId)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
        double lostFactorSum = lost.stream().mapToDouble(dsParams::factor).sum();
        return new GenOutage(pRow, pTarget, vRow, qTarget, toPq, qOverride, dsBus, dsFactorDelta, dsBaseDelta,
                dsMaxDelta, dsMinDelta, rlEntry, minQDelta, maxQDelta, allGensLost, lostGenIds, lostFactorSum,
                lostLoadP - lostP, loadQDelta);
    }
}
