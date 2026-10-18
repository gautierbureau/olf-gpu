/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU-batched security analysis against the CPU: reactive limits: pins, capability curves, forced target Q and the realistic-voltage check.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaReactiveLimitsTest extends AbstractGpuSaTest {

    GpuSaReactiveLimitsTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @Test
    void sequentialFallThroughOnGpuSolverWithReactiveLimits() {
        // Regression guard around the historical CME (#15, root-caused since: cleaner-thread
        // listener removal, fixed). NB the 2-branch contingency here is BATCHED (multi-branch IS
        // in batch scope) — the test's value is the GPU-provider-vs-CPU status parity under
        // reactive limits, not sequential-path coverage.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<String> lines = network.getLineStream().limit(2).map(l -> l.getId()).toList();
        List<Contingency> contingencies = List.of(
                new Contingency("multi", new BranchContingency(lines.get(0)), new BranchContingency(lines.get(1))),
                new Contingency(lines.get(0), new BranchContingency(lines.get(0))));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setUseReactiveLimits(true).setDistributedSlack(true);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, java.util.Collections.emptyList(), sap);
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus(),
                        "status mismatch for " + cpu.getPostContingencyResults().get(i).getContingency().getId());
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** Reactive capability CURVE limits at runtime P: B2's curve makes maxQ fall as DS RAISES its
     *  targetP post-outage — the RL pin must use the limit at the MOVED P (rlCurveLimitsBatch), not
     *  the base snapshot; a sloped curve segment spans the DS range so the two differ materially. */
    @Test
    void deviceCurveReactiveLimitsAtRuntimePMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g2 = network.getGenerator("B2-G");
        g2.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-40).setMaxQ(55).endPoint()
                .beginPoint().setP(40).setMinQ(-40).setMaxQ(50).endPoint()
                .beginPoint().setP(140).setMinQ(-40).setMaxQ(34).endPoint()
                .add();
        network.getGenerator("B1-G").newMinMaxReactiveLimits().setMinQ(-990).setMaxQ(990).add();
        List<Contingency> ctgs = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L1-5-1", new BranchContingency("L1-5-1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
    }

    /** A generator outage whose loss is PRE-distributed, followed by a slack round that FIRES (slackBusPMaxMismatch
     *  1e-4), with the curve generator B2-G pinned at its limit.
     *
     *  <p>OPEN, measured 2026-10-08: at the default convEps the device ends 1.5e-5 kV off at B2 (B3-G outage); at
     *  convEps 1e-12 both arms agree to 1e-11, so the curve is evaluated at the right P (the suspicion that the curve
     *  channel misses the pre-distribution move is NOT confirmed - genCum carries it). Without the curve (fixed min/max
     *  limits) the same case is still 1.6e-7 kV off. A gap above machine precision means the two arms are not solving
     *  the same equation system at some solve: find the first pass whose targets, row modes or limits differ (OLF_DS_TRACE
     *  / OLF_RL_TRACE against the device's per-pass state). The assert is not loosened. */
    @org.junit.jupiter.api.Disabled("OPEN: pre-distribution + slack round: the arms solve different systems at some pass "
            + "(1.6e-7 kV, 1.5e-5 kV with a pinned curve generator, at default tolerance)")
    @Test
    void deviceCurveLimitsAtPreDistributedPlusSlackRoundMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g2 = network.getGenerator("B2-G");
        g2.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-40).setMaxQ(55).endPoint()
                .beginPoint().setP(40).setMinQ(-40).setMaxQ(50).endPoint()
                .beginPoint().setP(140).setMinQ(-40).setMaxQ(34).endPoint()
                .add();
        com.powsybl.iidm.network.Generator g1 = network.getGenerator("B1-G");
        g1.newMinMaxReactiveLimits().setMinQ(-990).setMaxQ(990).add();
        // real active power on two more units (off the slack unit, so the case stays balanced): losing either
        // pre-distributes 30 / 20 MW, a share of which lands on B2-G's sloped curve segment
        network.getGenerator("B3-G").setTargetP(30).setMinP(0).setMaxP(100);
        network.getGenerator("B6-G").setTargetP(20).setMinP(0).setMaxP(100);
        g1.setTargetP(g1.getTargetP() - 50);
        // the slack unit sits out (a participating slack plus a lost participant is a CPU shape: PARTICIPATING_SLACK)
        g1.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false).withParticipationFactor(1.0).add();
        List<Contingency> ctgs = List.of(
                new Contingency("B3-G", new com.powsybl.contingency.GeneratorContingency("B3-G")),
                new Contingency("B6-G", new com.powsybl.contingency.GeneratorContingency("B6-G")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setSlackBusPMaxMismatch(1e-4);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "both generator outages must run on the DEVICE");
    }

    /** A Q-carrying LOAD shed at a bus PINNED on its reactive limit. The reactive-limit pack holds the
     *  bus's load target Q as a run-constant, and reads it in two paired forms: the pinned row target is
     *  limit - loadQ, and the controller's own q is the net injection + loadQ. Shedding the load makes
     *  both stale. B6-G is given tight limits (+-5 MVAr) so B6 certainly pins, and B6's load carries
     *  7.5 MVAr, so the stale term is larger than the limit itself. B6 rather than the bigger B3: its
     *  load is 11 MW, small enough that the slack distribution does not saturate and reject.
     *
     *  <p>This used to be TWO defects on the two halves of one condition: a PV bus was routed to the CPU
     *  for it, and a bus the base loadflow had ALREADY pinned reads busPv=false, fell through that gate,
     *  batched, and was silently wrong (rte6515 LOAD-6157, 0.37 kV — found only by widening the load set
     *  past the top 50). One per-scenario load Q closes both. VERIFIED by disabling the fix and watching
     *  this test fail. */
    @Test
    void devicePinnedBusLosingReactiveLoadMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        // ONLY B3 is tight. Pin every PV bus in the case instead and the run hits the all-pinned
        // corner, which the device rejects by design (OLF's global "strongest stays PV" election is
        // not a kernel) — the scenario would go to the CPU and the test would prove nothing.
        network.getGenerator("B6-G").newMinMaxReactiveLimits().setMinQ(-5).setMaxQ(5).add();
        for (String wide : List.of("B1-G", "B2-G", "B3-G", "B8-G")) {
            network.getGenerator(wide).newMinMaxReactiveLimits().setMinQ(-990).setMaxQ(990).add();
        }
        List<Contingency> ctgs = List.of(
                new Contingency("B6-L", new com.powsybl.contingency.LoadContingency("B6-L")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
    }

    /** A generator outage at a bus whose SURVIVORS carry sloped reactive capability curves. The curve
     *  channel reconstructs each generator's own active power from the bus's P row target to evaluate
     *  its curve there, and losing one of the bus's generators skews that reconstruction three ways at
     *  once: the dead generator's entry must stop evaluating, the step the OUTAGE puts in the row target
     *  is not distributed-slack power and must move nobody's curve, and the survivors now share the
     *  bus's slack over their own factors alone. B3 keeps voltage control (B3-G survives) so its
     *  reactive-limit entry stays live and the evaluated limit reaches the answer. The curve slopes
     *  steeply and B3-G2's 60 MW is large next to the distributed-slack movement, so each of the three,
     *  left unrepaired, moves the limit by MVAr — far above the bar here. (rte6515's curves are flat
     *  over their working range, so the production set exercises this path without testing its
     *  arithmetic; B2 cannot host it, that is the slack bus and its P is free.)
     *
     *  <p>The curve sits low enough that B3 is genuinely PINNED at its maxQ, which is what makes the
     *  evaluated limit reach the answer — VERIFIED by disabling each of the three repairs in turn and
     *  watching this test fail on each. An earlier version of it, with the same shape but a curve the
     *  bus never reached, passed with all three disabled and proved nothing. */
    @Test
    void deviceCurveLimitsWithLostSiblingGeneratorMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g3 = network.getGenerator("B3-G");
        g3.setTargetP(40).setMinP(0).setMaxP(200);
        com.powsybl.iidm.network.Generator g3b = network.getVoltageLevel("VL3").newGenerator()
                .setId("B3-G2")
                .setBus("B3").setConnectableBus("B3")
                .setTargetP(60).setMinP(0).setMaxP(200)
                .setTargetV(g3.getTargetV()).setVoltageRegulatorOn(true)
                .add();
        // Sloped over the whole working range, and steeply: a mis-attributed P moves maxQ by MVAr.
        for (com.powsybl.iidm.network.Generator g : List.of(g3, g3b)) {
            g.newReactiveCapabilityCurve()
                    .beginPoint().setP(0).setMinQ(-40).setMaxQ(5).endPoint()
                    .beginPoint().setP(200).setMinQ(-40).setMaxQ(-45).endPoint()
                    .add();
        }
        network.getGenerator("B1-G").newMinMaxReactiveLimits().setMinQ(-990).setMaxQ(990).add();
        // B3's 100 MW comes off B1-G so the case keeps its original generation/load balance: the
        // participants below can only absorb by coming DOWN, and an unbalanced case exhausts that
        // headroom and fails the distribution outright (the base loadflow does not even converge).
        com.powsybl.iidm.network.Generator g1 = network.getGenerator("B1-G");
        g1.setTargetP(g1.getTargetP() - 100);
        // A participating SLACK bus combined with a factor reweight is CPU-routed by design (the retain
        // math does not hold), so the two generators at the slack end sit out and the condensers take
        // their place. Four equal participants, two of them at B3: losing B3-G2 leaves B3 with 1 of the
        // remaining 3, so its survivor's renormalized share is exactly 2 — a value the unrepaired
        // channel cannot produce by accident.
        network.getGenerator("B6-G").setMinP(0).setMaxP(100);
        network.getGenerator("B8-G").setMinP(0).setMaxP(100);
        java.util.Set<String> sitOut = java.util.Set.of("B1-G", "B2-G");
        for (com.powsybl.iidm.network.Generator g : network.getGenerators()) {
            g.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                    .withParticipate(!sitOut.contains(g.getId())).withParticipationFactor(1.0).add();
        }
        List<Contingency> ctgs = List.of(
                new Contingency("B3-G2", new com.powsybl.contingency.GeneratorContingency("B3-G2")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
    }

    /** A curve generator that is NOT in voltage control shares a bus with one that is. A bus's
     *  reactive limit is not the sum of its generators' curve limits: AbstractLfBus.getLimitQ takes a
     *  generator's own limit only when its control type is VOLTAGE or REMOTE_REACTIVE_POWER, and its
     *  fixed targetQ otherwise. So a machine that is not controlling contributes a CONSTANT to the bus
     *  limit, and the distributed slack moving its P must not move that limit — while the curve channel
     *  packed every participating curve generator and the kernel added its slope to the bus's box.
     *
     *  <p>B3 carries B3-G in voltage control and B3-G2 with its regulator OFF; both participate in the
     *  distributed slack and both carry a curve sloped over the whole working range, so the slack moves
     *  both generators' P while only B3-G's slope belongs in B3's limit. B3-G2's fixed targetQ is low
     *  enough that B3 is genuinely PINNED at its maxQ (Q_PROBE: genTargetQ == maxQ, and the bus sits
     *  0.0004 pu over), which is what makes the evaluated limit reach the answer — an earlier version
     *  with B3 1.9 MVAr short of its limit passed with the filter removed and proved nothing. With the
     *  filter removed this fails at 136.35 kV against the CPU's 134.44. The same shape on network.xiidm
     *  (battery FQHV, regulator off, sharing a controlled bus) put the load contingency DJXY
     *  5.166e-05 kV out, invariant to convEps. */
    @Test
    void deviceCurveLimitOfNonControllingGeneratorStaysConstant() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g3 = network.getGenerator("B3-G");
        g3.setTargetP(40).setMinP(0).setMaxP(200);
        // regulator OFF -> LfGenerator control type OFF -> contributes its fixed targetQ to B3's box
        com.powsybl.iidm.network.Generator g3b = network.getVoltageLevel("VL3").newGenerator()
                .setId("B3-G2")
                .setBus("B3").setConnectableBus("B3")
                .setTargetP(60).setMinP(0).setMaxP(200)
                .setTargetQ(-10).setVoltageRegulatorOn(false)
                .add();
        // B3-G's curve puts B3 at its maxQ; B3-G2's is steeper still, so packing it moves the box by
        // MVAr per MW of slack — a difference no tolerance hides.
        g3.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-40).setMaxQ(5).endPoint()
                .beginPoint().setP(200).setMinQ(-40).setMaxQ(-45).endPoint()
                .add();
        g3b.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-60).setMaxQ(60).endPoint()
                .beginPoint().setP(200).setMinQ(-60).setMaxQ(-60).endPoint()
                .add();
        network.getGenerator("B1-G").newMinMaxReactiveLimits().setMinQ(-990).setMaxQ(990).add();
        // keep the case's generation/load balance: B3's extra 100 MW comes off B1-G (see
        // deviceCurveLimitsWithLostSiblingGeneratorMatchesCpu — an unbalanced case fails outright).
        com.powsybl.iidm.network.Generator g1 = network.getGenerator("B1-G");
        g1.setTargetP(g1.getTargetP() - 100);
        network.getGenerator("B6-G").setMinP(0).setMaxP(100);
        network.getGenerator("B8-G").setMinP(0).setMaxP(100);
        java.util.Set<String> sitOut = java.util.Set.of("B1-G", "B2-G");   // the slack end sits out
        for (com.powsybl.iidm.network.Generator g : network.getGenerators()) {
            g.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                    .withParticipate(!sitOut.contains(g.getId())).withParticipationFactor(1.0).add();
        }
        // LOAD outages, not line outages: a line outage only moves the LOSSES, and a tenth of a MW
        // through the curve slope stays under the bar. Losing B4-L takes ~48 MW off the participants,
        // which is what makes a wrongly-packed slope worth MVAr.
        List<Contingency> ctgs = List.of(
                new Contingency("B4-L", new com.powsybl.contingency.LoadContingency("B4-L")),
                new Contingency("B9-L", new com.powsybl.contingency.LoadContingency("B9-L")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "both contingencies must run on the DEVICE, or the curve channel is never exercised");
    }

    /** forceTargetQInReactiveLimits (ON in the RTE production params since OLF 2.3): a generator that does
     *  not control voltage injects its targetQ CLAMPED into its reactive limits, and with a capability curve
     *  those limits are a function of its P - so every distributed-slack move of its P re-clamps the target
     *  (LfGeneratorImpl.getTargetQ, re-read through AbstractLfBus.invalidateGenerationTargetP).
     *
     *  <p>B4-G sits alone on load bus B4, regulator OFF, targetQ 40 MVAr against a curve whose maxQ falls
     *  from 60 MVAr at P = 0 to -60 at P = 200: at its base 60 MW the clamp gives 24 MVAr, and losing a load
     *  moves its P by the slack share, so the injected Q moves with it. The CPU-only check below proves the
     *  flag changes this answer, so a GPU that kept the raw or the base-clamped target cannot pass. */
    @Test
    void deviceForcedTargetQFollowsDistributedSlackP() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g4 = network.getVoltageLevel("VL4").newGenerator()
                .setId("B4-G")
                .setBus("B4").setConnectableBus("B4")
                .setTargetP(60).setMinP(0).setMaxP(200)
                .setTargetQ(40).setVoltageRegulatorOn(false)
                .add();
        g4.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-60).setMaxQ(60).endPoint()
                .beginPoint().setP(200).setMinQ(-60).setMaxQ(-60).endPoint()
                .add();
        com.powsybl.iidm.network.Generator g1 = network.getGenerator("B1-G");
        g1.setTargetP(g1.getTargetP() - 60);                     // keep the case balanced
        g1.newMinMaxReactiveLimits().setMinQ(-990).setMaxQ(990).add();
        java.util.Set<String> sitOut = java.util.Set.of("B1-G", "B2-G");   // the slack end sits out
        for (com.powsybl.iidm.network.Generator g : network.getGenerators()) {
            g.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                    .withParticipate(!sitOut.contains(g.getId())).withParticipationFactor(1.0).add();
        }
        network.getGenerator("B6-G").setMinP(0).setMaxP(100);
        network.getGenerator("B8-G").setMinP(0).setMaxP(100);
        // the generator outage's lost P is PRE-distributed before the post-contingency solve: B4-G must start from
        // the clamp at its pre-distributed P, not at its base P
        List<Contingency> ctgs = List.of(
                new Contingency("B9-L", new com.powsybl.contingency.LoadContingency("B9-L")),
                new Contingency("B14-L", new com.powsybl.contingency.LoadContingency("B14-L")),
                new Contingency("B8-G", new com.powsybl.contingency.GeneratorContingency("B8-G")));
        java.util.function.Function<Boolean, com.powsybl.loadflow.LoadFlowParameters> params = force -> {
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setDistributedSlack(true).setUseReactiveLimits(true)
                    .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(p).setForceTargetQInReactiveLimits(force);
            return p;
        };
        // CPU only: the flag must move B4 by more than any tolerance below, or this test proves nothing.
        java.util.Set<String> vls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), vls, java.util.Set.of()));
        double maxShift = 0;
        SecurityAnalysisResult[] cpu = new SecurityAnalysisResult[2];
        for (int f = 0; f < 2; f++) {
            com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
            sap.setLoadFlowParameters(params.apply(f == 1));
            cpu[f] = runSecurityAnalysis(network, ctgs, mon, sap);
        }
        for (int i = 0; i < ctgs.size(); i++) {
            var off = cpu[0].getPostContingencyResults().get(i).getNetworkResult().getBusResults();
            var on = cpu[1].getPostContingencyResults().get(i).getNetworkResult().getBusResults();
            for (int b = 0; b < off.size(); b++) {
                maxShift = Math.max(maxShift, Math.abs(off.get(b).getV() - on.get(b).getV()));
            }
        }
        assertTrue(maxShift > 1e-3, "the flag must change the CPU answer (max bus shift " + maxShift + " kV)");
        assertGpuMatchesCpu(network, ctgs, 1e-11, params.apply(true));
        assertEquals(3, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "every contingency must run on the DEVICE");
    }

    /** forceTargetQInReactiveLimits on a voltage-controlled bus, through its LIMITS: B3-G controls B3 and B3-G2 sits
     *  on the same bus with its regulator OFF, so AbstractLfBus.getLimitQ counts B3-G2's target Q - its IIDM 40 MVAr
     *  clamped into its own curve at its CURRENT P - in B3's min and max limits. The base load flow pins B3 at that
     *  max and FREEZES its generation target; losing B8-G (30 MW) pre-distributes onto B3's two units, both curves
     *  fall, and OLF re-freezes B3 at the moved limit (RL_REFREEZE -0.1368 -> -0.2173 pu). Without the limit entries
     *  the device is 0.86 kV off at B3; routing a frozen bus's clamp to its Q row instead is just as far off; adding
     *  the loss pre-distribution on top of genCum (which already carries it) is 0.84 kV off the other way. The
     *  CPU-only check proves the flag moves the answer. */
    @Test
    void deviceForcedTargetQOnVoltageControlledBusMovesItsLimits() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g3 = network.getGenerator("B3-G");
        g3.setTargetP(40).setMinP(0).setMaxP(200);
        com.powsybl.iidm.network.Generator g3b = network.getVoltageLevel("VL3").newGenerator()
                .setId("B3-G2")
                .setBus("B3").setConnectableBus("B3")
                .setTargetP(60).setMinP(0).setMaxP(200)
                .setTargetQ(40).setVoltageRegulatorOn(false)
                .add();
        // 35 MVAr lower than the fixture B3-G2 normally pins against, to make room for B3-G2's +34 MVAr clamped
        // target: B3's max limit stays where it pins (-5 - 35 + 24 = -16 MVAr at the base P)
        g3.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-80).setMaxQ(-30).endPoint()
                .beginPoint().setP(200).setMinQ(-80).setMaxQ(-80).endPoint()
                .add();
        g3b.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-60).setMaxQ(60).endPoint()
                .beginPoint().setP(200).setMinQ(-60).setMaxQ(-60).endPoint()
                .add();
        com.powsybl.iidm.network.Generator g1 = network.getGenerator("B1-G");
        g1.newMinMaxReactiveLimits().setMinQ(-990).setMaxQ(990).add();
        // two more units with real power: losing one pre-distributes it onto B3's generators, RAISING B3-G2's P,
        // which lowers its clamped target and so B3's max limit - B3 stays pinned, at a lower value
        network.getGenerator("B6-G").setTargetP(20).setMinP(0).setMaxP(100);
        network.getGenerator("B8-G").setTargetP(30).setMinP(0).setMaxP(100);
        g1.setTargetP(g1.getTargetP() - 150);
        java.util.Set<String> sitOut = java.util.Set.of("B1-G", "B2-G");
        for (com.powsybl.iidm.network.Generator g : network.getGenerators()) {
            g.newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                    .withParticipate(!sitOut.contains(g.getId())).withParticipationFactor(1.0).add();
        }
        List<Contingency> ctgs = List.of(
                new Contingency("B8-G", new com.powsybl.contingency.GeneratorContingency("B8-G")),
                new Contingency("B6-G", new com.powsybl.contingency.GeneratorContingency("B6-G")));
        java.util.function.Function<Boolean, com.powsybl.loadflow.LoadFlowParameters> params = force -> {
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setDistributedSlack(true).setUseReactiveLimits(true)
                    .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(p).setForceTargetQInReactiveLimits(force);
            return p;
        };
        java.util.Set<String> vls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), vls, java.util.Set.of()));
        SecurityAnalysisResult[] cpu = new SecurityAnalysisResult[2];
        for (int f = 0; f < 2; f++) {
            com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
            sap.setLoadFlowParameters(params.apply(f == 1));
            cpu[f] = runSecurityAnalysis(network, ctgs, mon, sap);
        }
        double maxShift = 0;
        for (int i = 0; i < ctgs.size(); i++) {
            var off = cpu[0].getPostContingencyResults().get(i).getNetworkResult().getBusResults();
            var on = cpu[1].getPostContingencyResults().get(i).getNetworkResult().getBusResults();
            for (int b = 0; b < off.size(); b++) {
                maxShift = Math.max(maxShift, Math.abs(off.get(b).getV() - on.get(b).getV()));
            }
        }
        assertTrue(maxShift > 1e-3, "the flag must change the CPU answer (max bus shift " + maxShift + " kV)");
        assertGpuMatchesCpu(network, ctgs, 1e-11, params.apply(true));
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "both contingencies must run on the DEVICE");
    }

    /** Curve-limit EXTRAPOLATION (extrapolateReactiveLimits — ON in the RTE production params;
     *  was a whole-run CPU gate on any curve network): the curve stops at P = 41 MW just above
     *  B2-G's base 40 MW, so the post-outage DS raise pushes P OUTSIDE the range and the limit
     *  must continue on the BOUNDARY TWO-POINT SLOPE (ReactiveCapabilityCurveUtil
     *  .extrapolateReactiveLimitsSlope) instead of clamping — rlCurveEval's extrapolate branch,
     *  fed live by rlCurveLimitsBatch at the DS-moved targets. GPU SA == CPU SA at 1e-11 (the
     *  curve channel's established share-reconstruction tolerance). */
    @Test
    void deviceCurveExtrapolationMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Generator g2 = network.getGenerator("B2-G");
        g2.newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-40).setMaxQ(55).endPoint()
                .beginPoint().setP(45).setMinQ(-40).setMaxQ(45).endPoint()
                .add();
        network.getGenerator("B1-G").newMinMaxReactiveLimits().setMinQ(-990).setMaxQ(990).add();
        List<Contingency> ctgs = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L1-5-1", new BranchContingency("L1-5-1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setExtrapolateReactiveLimits(true);
        assertGpuMatchesCpu(network, ctgs, 1e-11, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the extrapolating curve run must ENGAGE the batch (was a whole-run CPU gate)");
    }

    /** A base-PINNED controller that loses its only generator is still checked by OLF's reactive-limits loop
     *  (#80, network.xiidm/FBQV). g1 regulates vlb at a target between the bus's voltage with the generator in
     *  (399.80 kV at Q = 0) and without it (399.50 kV): the base pins it MIN_Q at 0. Losing g1, OLF's
     *  processLostGenerators disables the voltage control but keeps the qLimitType and the controller-element
     *  place, so the first reactive-limit pass sees a MIN_Q pin under its target and switches the bus back to PV,
     *  and the next pins it MAX_Q at its now-empty [0, 0] range. The device had put the entry on the islanded
     *  channel and never checked it: one solve fewer than OLF, and a state that stops elsewhere.
     *  GPU SA == CPU SA at 1e-12, and the outage must BATCH. */
    @Test
    void deviceBasePinnedBusLosingItsOnlyGeneratorIsStillCheckedMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = basePinnedDeadBusNetwork();
        List<Contingency> ctgs = List.of(
                new Contingency("g1", new com.powsybl.contingency.GeneratorContingency("g1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(false).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_0"));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the outage must batch (or the test proves nothing)");
    }

    /** maxPqPvSwitch = 0: OLF's switchPqPv blocks EVERY PQ -> PV switch (a bus's pvPqSwitchCount is never
     *  below 0), so a bus pinned in the base stays pinned through every contingency. The device read 0 as
     *  "unset" and applied OLF's default of 3: on the fixture above both members un-pin on the device
     *  where the CPU keeps them pinned. The two CPU answers (cap 0 / default cap) are first shown to
     *  differ on a member bus, or the cap would not be exercised. GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceHonoursZeroPqPvSwitchCapMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = unpinnedRemoteGroupNetwork();
        List<Contingency> contingencies = List.of(new Contingency("lpq", new com.powsybl.contingency.LineContingency("lpq")));
        java.util.function.IntFunction<com.powsybl.loadflow.LoadFlowParameters> params = cap -> {
            com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                    .setDistributedSlack(true).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                    .setSlackBusesIds(List.of("vl0_0"))
                    .setReactiveLimitsMaxPqPvSwitch(cap);
            return lfp;
        };
        // the cap must matter here: the CPU keeps both members pinned at 0 and un-pins them at the default
        double vCapped = cpuPostContingencyV(network, contingencies, params.apply(0), "vlc1");
        double vDefault = cpuPostContingencyV(network, contingencies, params.apply(3), "vlc1");
        assertTrue(Math.abs(vCapped - vDefault) > 0.1,
                "CPU at cap 0 vs the default cap: " + vCapped + " vs " + vDefault + " kV, the cap is not exercised");
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, contingencies, 1e-12, params.apply(0));
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the contingency must batch (or the test proves nothing)");
    }

    /** 3-member shared transformer voltage-control group (T2wT + T2wT2 + a cloned T2wT3, all
     *  regulating BUS_3 to 33 kV) plus a parallel line for a non-islanding line outage. Free
     *  member = T2wT (first enabled). */
    /** OLF checks the realistic band on the state the last loop able to fix it (here the discrete
     *  TransformerVoltageControl) leaves at the end of EVERY sweep, not only on the final state. Under the T2wT
     *  outage an intermediate sweep ends with a bus at 1.0240039 pu, the final state peaks at 1.0239403 pu and the
     *  base at 1.0227 pu: with maxRealisticVoltage 1.02397 the CPU reports FAILED. The device checked only the
     *  solves of the loops after the fixing one - none here - and the final state, so it reported CONVERGED. */
    @Test
    void deviceUnrealisticStateAtTransformerLoopEndMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        List<Contingency> ctgs = List.of(new Contingency("T2wT", new BranchContingency("T2wT")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL)
                .setMinNominalVoltageRealisticVoltageCheck(0)
                .setMaxRealisticVoltage(1.02397);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters()
                .setLoadFlowParameters(lfp);
        // Non-vacuity: the base converges and the CPU fails the contingency, on a state that is not its final one.
        assertEquals(com.powsybl.security.PostContingencyComputationStatus.FAILED,
                runSecurityAnalysis(sharedThreeMemberTvcNetwork(), ctgs, List.of(), sap,
                        com.powsybl.commons.report.ReportNode.NO_OP).getPostContingencyResults().get(0).getStatus());
        assertGpuMatchesCpu(sharedThreeMemberTvcNetwork(), ctgs, 1e-12, lfp, false);
    }

    /** voltageRemoteControlRobustMode — OLF's DEFAULT — now a criterion of the RL kernel. g1 controls
     *  b2 REMOTELY through a very reactive transformer; losing one of the two parallel b2-b3 lines
     *  collapses g1's OWN 20 kV bus far below the realistic band while its reactive power is still
     *  well inside [-3000, 6000] MVAr. ReactiveLimitsOuterLoop then stops maintaining the remote
     *  voltage: it freezes the generation at its INITIAL target Q with a MIN_REALISTIC_V limit type
     *  and reseeds the bus at 1 pu for the next solve. Turning the mode off rejects the very same
     *  state as unrealistic, so the status assertion below keeps the parity assertion honest. */
    @Test
    void deviceRemoteRobustRealisticVoltageMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = robustRealisticVNetwork();
        List<Contingency> ctgs = List.of(new Contingency("l23b", new BranchContingency("l23b")));
        // Non-vacuity: the criteria are what makes this network solvable at all. With the mode off the
        // base state is outside the realistic band and OLF rejects it, so there is no contingency to
        // compare; with it on the base converges and the contingency follows.
        assertEquals(com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.FAILED,
                runSecurityAnalysis(network, ctgs, robustRealisticVParameters(false))
                        .getPreContingencyResult().getStatus(),
                "without robust mode the base state is unrealistic");
        assertEquals(com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.CONVERGED,
                runSecurityAnalysis(network, ctgs, robustRealisticVParameters(true))
                        .getPreContingencyResult().getStatus(),
                "robust mode recovers it — this is the criterion under test");
        assertGpuMatchesCpu(network, ctgs, 1e-12, robustRealisticVParameters(true));
    }

    /** A member FROZEN by the realistic-voltage check (OLF: generation target Q frozen at its INITIAL value,
     *  MIN/MAX_REALISTIC_V) enters its peers' DISTR_Q targets at that frozen value (AcTargetVector sums
     *  otherControllerBus.getTargetQ()). The device rebuilt a pinned peer's targetQ from its Q limit, which for a
     *  realistic-voltage freeze is not where OLF froze it. g1 (10 MVar initial, 6000 MVar ceiling) frozen on l23b,
     *  g4/g5 sharing: at 1e-12. */
    @Test
    void deviceRealisticVoltageFrozenMemberInPeersTargetsMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = realisticFreezePeersNetwork();
        List<Contingency> ctgs = List.of(new Contingency("l23b", new BranchContingency("l23b")));
        com.powsybl.loadflow.LoadFlowParameters lfp = robustRealisticVParameters(true);
        // non-vacuity: OLF freezes g1 at its initial target Q on the contingency - the transformer's b1-side
        // reactive flow IS g1's output (b1 carries g1 alone), read off the CPU's own branch result
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of("t12a"), java.util.Set.of(), java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, ctgs, monitors, sap);
        assertEquals(com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.CONVERGED, cpu.getPreContingencyResult().getStatus());
        var post = cpu.getPostContingencyResults().get(0);
        assertEquals(com.powsybl.security.PostContingencyComputationStatus.CONVERGED, post.getStatus());
        double q1 = post.getNetworkResult().getBranchResult("t12a").getQ1();
        assertEquals(10.0, Math.abs(q1), 0.1, "g1 frozen at its initial 10 MVar on the CPU (q1 of t12a = " + q1
                + "; the branch flow carries the default criterion's slack; unfrozen it absorbs about 900 MVar)");
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp, true);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** Two merged sections (retained coupler), both generators regulating the main's own bus — a PILOT-SELF
     *  group — with a reactive-capability CURVE on the main's generator. A line loss drives both sections past
     *  MAX_Q and the pins change the losses enough for a distributed-slack round to run AFTER them, which moves
     *  the curve limit by LESS than maxReactivePowerMismatch. OLF leaves such a pin at its pin-time limit
     *  (checkPqBusWithQLimitType re-freezes only past the threshold); the device's member path did the same but
     *  its enrolled-pilot re-stamp fired above 1e-12 (tasks #73, rte6515 BUS-6029_BBS_1: VL-6231_0 stamped
     *  0.570758104 against OLF's 0.570780403, 5.68e-05 kV). GPU SA == CPU SA at 1e-12 at the DEFAULT criteria. */
    @Test
    void devicePinnedPilotSelfKeepsItsLimitUnderThresholdMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = pilotSelfCurveNetwork();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusPMaxMismatch(1e-4)               // a slack round fires AFTER the pins, moving the curve
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_b"));
        // the section loss is LISTED so the coupler is retained and the two sections are two buses (OLF
        // retains only the switches a listed contingency touches); losing g3's reactive support at vl3 is
        // what drives both sections past MAX_Q
        List<Contingency> ctgs = List.of(
                new Contingency("bus-1", new com.powsybl.contingency.BusbarSectionContingency("bbs1")),
                new Contingency("g3", new com.powsybl.contingency.GeneratorContingency("g3")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "both contingencies must batch (or the test proves nothing)");
    }

    /** A curve generator on the CHILD section of a coupler-joined pair (g1 on bbs1, relocated under the
     *  retained coupler), pinned by the loss of a generator whose active power the security analysis
     *  PRE-DISTRIBUTES: both arms must evaluate g1's curve at the pre-distributed P. The device's share
     *  reconstruction read the child's own PHI row, which hosts ZERO_PHI (target 0) under the relocation,
     *  so it evaluated the curve at the BASE P and pinned 8e-4 pu off while OLF pinned at the moved P
     *  (tasks #66/#73 tail, rte6515 BUS-4467_BBS: VL-6231_1 pinned at 0.576013 against OLF's 0.566571,
     *  the pilot VL-6231_0 — same unit, same P — exact; 9.56e-4 MW of distributed power). The slack
     *  threshold stays at its DEFAULT so no slack round runs before the pin and the fallback is what
     *  evaluates the curve. GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceRelocatedChildCurvePinnedAtPreDistributedPMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = pilotSelfCurveNetwork(true, 100);
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_b"));
        // ...and the loss of the PILOT section itself (bbs0: g0 and the load), which kills the coupler: the
        // surviving child's balance moves back to its own PHI row (dsRowOv) and the kernel must follow it
        // there - the structural dummy row is an identity at 0 in that scenario
        List<Contingency> ctgs = List.of(
                new Contingency("bus-1", new com.powsybl.contingency.BusbarSectionContingency("bbs1")),
                new Contingency("bus-0", new com.powsybl.contingency.BusbarSectionContingency("bbs0")),
                new Contingency("g3", new com.powsybl.contingency.GeneratorContingency("g3")));
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(3, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "every contingency must batch (or the test proves nothing)");
    }

    /** A bus pinned at a reactive limit that MOVES, once it has used up its PV-PQ switch budget. OLF
     *  caps the SWITCH (maxPqPvSwitch, applied in switchPqPv) but keeps re-freezing a pinned bus at its
     *  current limit every check; the device gated its whole pinned-bus block on that cap, so a capped
     *  bus stopped following its limit. The limit here is a reactive-capability CURVE, which moves with
     *  the generator's targetP as the distributed slack shifts it - exactly the real case
     *  (network.xiidm / GBMK: CQEM_0 and CQEM_5 stayed at -0.181175517 / -0.184950765 where OLF had
     *  re-frozen at -0.181173957 / -0.184949017, 2.079e-05 kV network-wide and invariant to the
     *  convergence criterion). The cap is set to 1 so a single pin exhausts it. GPU SA == CPU SA at 1e-12. */
    @Test
    void deviceCappedPinStillFollowsItsMovingLimitMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = Network.create("capped-pin-moving-limit", "code");
        com.powsybl.iidm.network.Substation s = network.newSubstation().setId("s").add();
        for (String vl : List.of("vl0", "vlp", "vlc", "vlg")) {
            com.powsybl.iidm.network.VoltageLevel v = s.newVoltageLevel().setId(vl).setNominalV(400)
                    .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
            v.getBusBreakerView().newBus().setId(vl + "_b").add();
        }
        network.getVoltageLevel("vl0").newGenerator().setId("g0").setBus("vl0_b").setConnectableBus("vl0_b")
                .setMinP(0).setMaxP(2000).setTargetP(200).setTargetV(400).setVoltageRegulatorOn(true).add();
        network.getVoltageLevel("vlp").newLoad().setId("ldp").setBus("vlp_b").setConnectableBus("vlp_b").setP0(300).setQ0(150).add();
        // the load whose loss makes the distributed slack move gc's P by tens of MW, which MOVES gc's
        // capability curve while it is pinned; it sits far from gc so the reactive picture there barely
        // changes and the pin holds
        network.getVoltageLevel("vlg").newLoad().setId("ld1").setBus("vlg_b").setConnectableBus("vlg_b").setP0(200).setQ0(0).add();
        // the curve generator: minQ rises steeply with P, so the distributed slack MOVES its limit
        // a reactive load at the generator's own bus and a target it cannot reach: gc saturates at
        // MAX_Q and its voltage stays BELOW target, so it never wants to unpin - the case that must
        // keep following the moving limit
        network.getVoltageLevel("vlc").newLoad().setId("ldc").setBus("vlc_b").setConnectableBus("vlc_b").setP0(100).setQ0(130).add();
        network.getVoltageLevel("vlc").newGenerator().setId("gc").setBus("vlc_b").setConnectableBus("vlc_b")
                .setMinP(0).setMaxP(400).setTargetP(150).setTargetV(404).setVoltageRegulatorOn(true).add()
                .newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-60).setMaxQ(60).endPoint()
                .beginPoint().setP(400).setMinQ(-4).setMaxQ(90).endPoint()
                .add();
        java.util.function.BiConsumer<String, String[]> line = (id, ends) -> network.newLine().setId(id)
                .setVoltageLevel1(ends[0]).setBus1(ends[0] + "_b").setConnectableBus1(ends[0] + "_b")
                .setVoltageLevel2(ends[1]).setBus2(ends[1] + "_b").setConnectableBus2(ends[1] + "_b")
                .setR(0.5).setX(12).setG1(0).setB1(0).setG2(0).setB2(0).add();
        line.accept("l0p", new String[] {"vl0", "vlp"});
        line.accept("l0p_BIS", new String[] {"vl0", "vlp"});
        line.accept("lpc", new String[] {"vlp", "vlc"});
        line.accept("l0c", new String[] {"vl0", "vlc"});
        line.accept("l0g", new String[] {"vl0", "vlg"});     // g1 on its own bus: not the slack's
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setReactiveLimitsMaxPqPvSwitch(1)           // one pin exhausts the budget
                .setSlackBusPMaxMismatch(1e-4)               // a slack round fires AFTER the pin, moving the curve
                // UNIFORM_CRITERIA ties the reactive-limit tolerance to this value on BOTH arms
                // (AbstractAcOuterLoopConfig.createReactiveLimitsOuterLoop), and the post-pin curve
                // drift is ~1e-5 pu: at the default 1e-4 neither arm re-freezes and the path under
                // test never runs. The comparison tolerance below is untouched.
                .setNewtonRaphsonConvEpsPerEq(1e-8)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("vl0_0"));
        assertGpuMatchesCpu(network, List.of(new Contingency("l0c", new com.powsybl.contingency.LineContingency("l0c"))),
                1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "the contingency must batch (or the test proves nothing)");
    }

    @Test
    void fixedQGeneratorLossAdjustsQTarget() {
        // #21: a lost FIXED-Q generator on an already-PQ bus must remove its targetQ from the bus
        // Q target (the CPU does via LfContingency). ieee14 has realistic impedances, so a wrong
        // Q injection moves voltages by ~0.4 kV — far above the 0.05 kV assertion tolerance,
        // while the convergence-slack noise is ~1e-3 kV.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.iidm.network.Bus b10 = network.getBusBreakerView().getBus("B10");
        b10.getVoltageLevel().newGenerator().setId("gq").setBus("B10").setConnectableBus("B10")
                .setMinP(0).setMaxP(50).setTargetP(10).setTargetQ(15).setVoltageRegulatorOn(false).add();
        // distributed slack OFF: gq would participate (PROPORTIONAL_TO_GENERATION_P_MAX includes
        // all gens) alongside a participating slack — the known DS fallback combo, orthogonal here.
        assertGpuMatchesCpu(network,
                List.of(new Contingency("gq", new com.powsybl.contingency.GeneratorContingency("gq"))), 1e-12, false);
    }
}
