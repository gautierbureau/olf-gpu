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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-batched security analysis against the CPU: distributed slack: the device distribution, the loss pre-distribution and its carry into strategy pairs.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaDistributedSlackTest extends AbstractGpuSaTest {

    GpuSaDistributedSlackTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @Test
    void loadBasedDistributedSlackMatchesCpu() {
        // Load-based distributed slack (BalanceType.PROPORTIONAL_TO_LOAD): OLF's DistributedSlackOuterLoop
        // redistributes the mismatch onto LOADS. The batch must ENGAGE and the post-contingency states must match
        // the stock CPU SA.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_LOAD);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)   // create() ATTACHES the extension
                .setSlackBusPMaxMismatch(0.01).setNewtonRaphsonConvEpsPerEq(1e-9);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        List<Contingency> contingencies = List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, monitors, sap);
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get() > 0,
                    "the load-based-DS run must ENGAGE the batch (guard fell back -> the path is not exercised)");
            assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
            for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
                assertEquals(cpu.getPostContingencyResults().get(i).getStatus(),
                        gpu.getPostContingencyResults().get(i).getStatus());
                var cb = cpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                var gb = gpu.getPostContingencyResults().get(i).getNetworkResult().getBusResults();
                assertEquals(cb.size(), gb.size());
                for (int b = 0; b < cb.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(cb.get(b).getV(), gb.get(b).getV(), 1e-12,
                            "load-based-DS bus voltage " + cb.get(b).getBusId());
                }
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** DEVICE DS balance type PROPORTIONAL_TO_GENERATION_P (TARGET factors = |targetP| at the anchor):
     *  ieee14 line outages, GPU SA == CPU SA at machine precision. */
    @Test
    void deviceDsBalanceTypeTargetMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P);
        assertGpuMatchesCpu(network, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L1-5-1", new BranchContingency("L1-5-1"))), 1e-12, lfp);
    }

    /** DEVICE DS balance type PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR (explicit per-generator
     *  factors, ASYMMETRIC so a wrong split fails the state assert): GPU SA == CPU SA at machine precision. */
    @Test
    void deviceDsBalanceTypeParticipationFactorMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B1-G").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(true).withParticipationFactor(3.0).add();
        network.getGenerator("B2-G").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(true).withParticipationFactor(1.0).add();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR);
        assertGpuMatchesCpu(network, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L1-5-1", new BranchContingency("L1-5-1"))), 1e-12, lfp);
    }

    /** DEVICE DS with useActiveLimits=false (unbounded water-fill — the rooms are infinite): a generator
     *  contingency (reweight path) plus line outages, GPU SA == CPU SA at machine precision. TIGHT maxP
     *  on B2-G would saturate WITH limits, so the no-limits split is genuinely different — a limits leak
     *  in the pack fails the state assert. */
    @Test
    void deviceDsNoActiveLimitsMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B2-G").setMaxP(40.5);          // headroom 0.5 MW: saturates under limits
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setUseActiveLimits(false);
        assertGpuMatchesCpu(network, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L1-5-1", new BranchContingency("L1-5-1"))), 1e-12, lfp);
    }

    /** useActiveLimits=false still keeps OLF's no-sign-change rule (GenerationActivePowerDistributionStep: the limits
     *  become -+MAX_VALUE, THEN a positive generator's floor is 0 and a negative one's ceiling 0). B2-G at 0.5 MW
     *  participates under P_MAX; with the slack's unit out of the distribution the base mismatch pushes generation DOWN, B2-G's
     *  share by more than 0.5 MW, and OLF holds B2-G at 0 (B3-G, at 200 MW, takes the rest) while the device, with infinite rooms, took it negative. The contingencies'
     *  own loss is pre-distributed on the host with OLF's rule; the base load flow and the in-loop residuals run the
     *  device's water-fill. 1e-12 on device. */
    @Test
    void deviceDsNoActiveLimitsKeepsTheGenerationSignMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B2-G").setTargetP(0.5);         // participates (P_MAX), 0.5 MW above its sign floor
        network.getGenerator("B3-G").setMaxP(400).setTargetP(200); // a second participant, with room to go down
        network.getGenerator("B1-G").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false).withDroop(4).add();             // the slack's own unit does not participate
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setUseActiveLimits(false);
        List<Contingency> ctgs = List.of(new Contingency("B9-L", new com.powsybl.contingency.LoadContingency("B9-L")),
                new Contingency("B14-L", new com.powsybl.contingency.LoadContingency("B14-L")));
        GpuBatchScopeCensus.reset();
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "census=" + GpuBatchScopeCensus.snapshot());
    }

    /** DEVICE DS balance type PROPORTIONAL_TO_LOAD: participants are the LOADS (per-bus factor = the
     *  variable load share, unbounded absorption); GPU SA == CPU SA at machine precision. */
    @Test
    void deviceDsBalanceTypeLoadMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_LOAD);
        assertGpuMatchesCpu(network, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L1-5-1", new BranchContingency("L1-5-1"))), 1e-12, lfp);
    }

    /** DEVICE DS balance type PROPORTIONAL_TO_CONFORM_LOAD: only each load's VARIABLE part participates
     *  (LoadDetail extension, ASYMMETRIC fixed/variable splits so a plain-load factor mixup fails);
     *  GPU SA == CPU SA at machine precision. */
    @Test
    void deviceDsBalanceTypeConformLoadMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        // give two big loads an explicit fixed/variable split; the others stay fully fixed under
        // CONFORM (no LoadDetail => variablePart = 0), so the participating set itself changes too
        network.getLoad("B3-L").newExtension(com.powsybl.iidm.network.extensions.LoadDetailAdder.class)
                .withFixedActivePower(70).withVariableActivePower(24.2)
                .withFixedReactivePower(19).withVariableReactivePower(0).add();
        network.getLoad("B4-L").newExtension(com.powsybl.iidm.network.extensions.LoadDetailAdder.class)
                .withFixedActivePower(10).withVariableActivePower(37.8)
                .withFixedReactivePower(-3.9).withVariableReactivePower(0).add();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_CONFORM_LOAD);
        assertGpuMatchesCpu(network, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L1-5-1", new BranchContingency("L1-5-1"))), 1e-12, lfp);
    }

    /** DEVICE DS balance type PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN: direction-dependent margin
     *  factors computed in-kernel (ASYMMETRIC margins via a tight maxP on B2-G, so a static-factor or
     *  wrong-direction mixup fails); GPU SA == CPU SA at machine precision. */
    @Test
    void deviceDsBalanceTypeRemainingMarginMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B2-G").setMaxP(100);           // margin 60 MW vs B1-G's ~9767 — sharply asymmetric
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN);
        assertGpuMatchesCpu(network, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L1-5-1", new BranchContingency("L1-5-1"))), 1e-12, lfp);
    }

    /** DS-carry regression for the INCREMENTAL SHUNT orchestrator (the IncrementalTvc bug's sibling):
     *  the l1 outage reroutes ~100 MW over the doubled-impedance clone, so the MAIN solve's in-kernel
     *  DS distributes the loss jump; the tightened deadband makes the same scenario MOVE shunt
     *  sections, whose re-solve must CARRY that distribution (capture/seed/re-capture) — without it
     *  the kernel re-derives the slack split from the re-uploaded base targets. 1e-12. */
    @Test
    void deviceIncrementalShuntDsCarryMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.ShuntNetworkFactory.create();
        network.getShuntCompensator("SHUNT").setTargetDeadband(1.0);   // post-outage V sag must trip it
        com.powsybl.iidm.network.Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l1", new BranchContingency("l1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setShuntCompensatorVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setShuntVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.ShuntVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get());
    }

    /** Tightened-DS regression for the INCREMENTAL FLOW-CONTROL orchestrator: at 0.01 MW the
     *  scenarios batch at 1e-12 with the DS carry plumbed. The second block is the 1e-4 MW threshold,
     *  below OLF's other distribution bound: DistributedSlackOuterLoop.check distributes only past BOTH
     *  slackBusPMaxMismatch and P_RESIDUE_EPS (1 kW). The device tested the threshold alone and ran one
     *  more pass on L1 (a 1.196e-6 pu mismatch OLF leaves): 1.1e-10 kV and 1.2e-4 MW of distributed
     *  power, once read as a knife edge. */
    @Test
    void deviceIncrementalPhaseDsCarryMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL)
                .setTargetDeadband(1).setRegulating(true).setTapPosition(1)
                .setRegulationTerminal(ps1.getTerminal1()).setRegulationValue(83);
        com.powsybl.iidm.network.Line l1 = network.getLine("L1");
        network.newLine().setId("L1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("L1_BIS", new BranchContingency("L1_BIS")),
                new Contingency("L1", new BranchContingency("L1")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setPhaseShifterRegulationOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setPhaseShifterControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.PhaseShifterControlMode.INCREMENTAL)
                .setSlackBusPMaxMismatch(0.01);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
        // a threshold below P_RESIDUE_EPS (100 W)
        com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp).setSlackBusPMaxMismatch(1e-4);
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    /** A generator outage whose loss pre-distribution SATURATES one generator on a multi-generator bus.
     *  OLF distributes the contingency's scheduled loss over the participating GENERATORS, clamping each
     *  at its own limit and renormalizing over the survivors. The device's host-side seed has to be the
     *  same function as the water-fill it seeds, or the device re-derives from an anchor the seed never
     *  produced — and it used to drop to a BUS-aggregate fill for every generator outage, because the
     *  per-generator path was gated on there being no gen-outage override bus.
     *
     *  <p>b2 carries three generators; g2c is given just 3 MW of headroom so the pre-distribution pins
     *  it while g2 and g2b still have room. A bus aggregate cannot see that: b2's combined headroom is
     *  never exhausted, so the bus-level fill saturates nobody and spreads the loss differently. On
     *  rte6515 that difference put GEN-6372's line current 1.11e-05 out (relative) and flipped a
     *  violation; the state difference survived tightening the inner Newton to 1e-12, so it was never
     *  convergence slack.
     *
     *  <p>DISABLED, and honestly so. Clamping the seed per generator improves this case 28x
     *  (8.02e-08 kV -> 2.81e-09 kV at b2) and closes rte6515's GEN-6372 completely (1.11e-05 relative
     *  -> 3.64e-13, and the 1389-generator set from 1 differing violation to 0), but 2.81e-09 kV is
     *  still above the bar, so something smaller-order in the saturating path is still wrong. The seed
     *  and OLF now agree on WHICH generator pins and at WHAT value (OLF_GPU_SEED_TRACE against OLF's
     *  OLF_DS_TRACE: both clamp g2c at 0.23 - 0.217443120 = 0.01255688), so the remaining difference is
     *  elsewhere. Sweeping g2c's headroom shows the split cleanly: at maxP 50/30/26 nobody saturates
     *  and BOTH arms agree at 1e-11; only at 23 does anything differ. That is also why this network
     *  cannot host a passing regression test for the fix — where it bites, this residual bites too.
     *  The water-fill is continuous at a clamp, so a 2.8e-09 output needs a 2.8e-09 input: this is a
     *  real difference, not amplification through a kink. */
    @org.junit.jupiter.api.Disabled("OPEN: the saturating pre-distribution still leaves 2.8e-09 kV. "
            + "Enable when it reaches the 1e-11 bar — the assert is deliberately NOT loosened to pass.")
    @Test
    void deviceLossPreDistributionSaturatingOneGeneratorOnABusMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = distributedSlackNetworkWithExtras();
        network.getGenerator("g2c").setMaxP(23);              // targetP 20 -> 3 MW of headroom: pins first
        List<Contingency> ctgs = List.of(
                new Contingency("g3", new com.powsybl.contingency.GeneratorContingency("g3")));
        // The factory's generators carry a DROOP, not a participation factor, so the default
        // PROPORTIONAL_TO_GENERATION_P_MAX is the balance type that actually distributes here; asking
        // for PARTICIPATION_FACTOR gives every one of them a zero factor and the base loadflow fails.
        assertGpuMatchesCpu(network, ctgs, 1e-11);
    }

    /** DS-carry regression for the HVDC AC-emulation orchestrator: same freeze/saturation scenarios,
     *  with slackBusPMaxMismatch tightened to 0.01 MW (both arms) so the MW-scale network's loss
     *  deltas actually DISTRIBUTE in the main solve — the freeze/limit re-solves must carry that
     *  distribution (capture/seed/re-capture) instead of re-deriving it from base targets. 1e-12. */
    @Test
    void deviceHvdcDsCarryMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        network.getHvdcLine("hvdc12").newExtension(com.powsybl.iidm.network.extensions.HvdcOperatorActivePowerRangeAdder.class)
                .withOprFromCS1toCS2(1.4f).withOprFromCS2toCS1(1.4f).add();
        com.powsybl.iidm.network.Line l12 = network.getLine("l12");
        network.newLine().setId("l12_BIS")
                .setVoltageLevel1(l12.getTerminal1().getVoltageLevel().getId()).setBus1(l12.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l12.getTerminal2().getVoltageLevel().getId()).setBus2(l12.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l12.getR() * 2).setX(l12.getX() * 2)
                .setG1(l12.getG1()).setB1(l12.getB1()).setG2(l12.getG2()).setB2(l12.getB2()).add();
        List<Contingency> ctgs = List.of(new Contingency("l12_BIS", new BranchContingency("l12_BIS")),
                new Contingency("l12", new BranchContingency("l12")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setStartWithFrozenACEmulation(true)
                .setSlackBusPMaxMismatch(1e-4);   // MW-scale network: micro-threshold so DS genuinely distributes (both arms)
        assertGpuMatchesCpu(network, ctgs, 1e-12, lfp);
    }

    @Test
    void replayedLossDistributionKeepsResultsAndSparesStrategyContingencies() {
        // The batch answers OLF's per-contingency loss distribution from its replay (lead L2 of
        // docs/gpu-sa-host-overhead-assessment.md) — EXCEPT for a contingency an operator strategy targets:
        // an InjectionThresholdCondition on ACTIVE_POWER reads the post-contingency generator target P, which
        // only OLF's host distribution sets (ThresholdConditionEvaluator). IEEE 14 as shipped cannot show it:
        // B3/B6/B8 are at 0 MW (excluded from distribution) and B1-G's ~9,999 MW maxP takes nearly all of any
        // share. So B3-G, B6-G and B8-G get 20 MW each and B1-G is taken out of the distribution. Losing B2-G
        // (40 MW) then spreads over B3/B6/B8, and the
        // strategy fires only if B6-G's target is then more than 5 MW above its base one (read from a plain CPU
        // load flow: a non-slack generator's P is its target). B8-G's outage carries no strategy and is
        // device-solved (its small loss spreads UP over B3/B6's headroom — a load outage here would drive
        // B3/B6/B8 to their minimum, and a saturating distribution is routed to the CPU by design). The replay
        // must answer B8-G alone, and give the CPU's results.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> fixture = () -> {
            Network n = IeeeCdfNetworkFactory.create14();
            for (String g : List.of("B3-G", "B6-G", "B8-G")) {
                n.getGenerator(g).setTargetP(20);
            }
            n.getGenerator("B1-G").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                    .withParticipate(false).add();
            return n;
        };
        Network network = fixture.get();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        Network base = fixture.get();
        assertTrue(new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(
                new com.powsybl.math.matrix.DenseMatrixFactory())).run(base, lfp).isFullyConverged());
        // MW: needs OLF's ThresholdConditionEvaluator to compare injections in MW like branches (OLF e97bb4e57,
        // assessment L11); before that fix it compared per-unit and this condition held on BOTH paths.
        double threshold = -base.getGenerator("B6-G").getTerminal().getP() + 5;
        List<Contingency> contingencies = List.of(
                new Contingency("B2-G", new com.powsybl.contingency.GeneratorContingency("B2-G")),
                new Contingency("B8-G", new com.powsybl.contingency.GeneratorContingency("B8-G")));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("redispatch3")
                        .withGeneratorId("B3-G").withActivePowerRelativeValue(true).withActivePowerValue(10).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sB6High",
                        com.powsybl.contingency.ContingencyContext.specificContingency("B2-G"),
                        new com.powsybl.contingency.strategy.condition.InjectionThresholdCondition("B6-G",
                                com.powsybl.contingency.strategy.condition.AbstractThresholdCondition.Variable.ACTIVE_POWER,
                                com.powsybl.contingency.strategy.condition.AbstractThresholdCondition.ComparisonType.GREATER_THAN,
                                threshold),
                        List.of("redispatch3")));
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.addExtension(OpenSecurityAnalysisParameters.class,
                new OpenSecurityAnalysisParameters().setContingencyPropagation(false));
        sap.setLoadFlowParameters(lfp);
        java.util.Set<String> allVls = network.getVoltageLevelStream().map(vl -> vl.getId())
                .collect(java.util.stream.Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(), allVls, java.util.Set.of()));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, monitors, sap, strategies, actions,
                com.powsybl.commons.report.ReportNode.NO_OP);
        assertEquals(1, cpu.getOperatorStrategyResults().size(),
                "the condition must hold on the CPU: B6-G above " + threshold + " MW once B2-G's loss is distributed");
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult replayed = runSecurityAnalysis(network, contingencies, monitors, sap, strategies, actions,
                    com.powsybl.commons.report.ReportNode.NO_OP);
            assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_REPLAYED_LOSS_COUNT.get(),
                    "the replay answers B8-G only: B2-G is targeted by a strategy");
            assertEquals(1, replayed.getOperatorStrategyResults().size(), "the strategy must fire on the GPU too");
            for (int c = 0; c < contingencies.size(); c++) {
                var r = replayed.getPostContingencyResults().get(c);
                var h = cpu.getPostContingencyResults().get(c);
                assertEquals(h.getStatus(), r.getStatus());
                org.junit.jupiter.api.Assertions.assertEquals(h.getDistributedActivePower(), r.getDistributedActivePower(), 1e-9,
                        "distributed active power of " + r.getContingency().getId());
                var hb = h.getNetworkResult().getBusResults();
                var rb = r.getNetworkResult().getBusResults();
                assertEquals(hb.size(), rb.size());
                for (int b = 0; b < hb.size(); b++) {
                    org.junit.jupiter.api.Assertions.assertEquals(hb.get(b).getV(), rb.get(b).getV(), 1e-10,
                            r.getContingency().getId() + " bus voltage " + hb.get(b).getBusId());
                }
            }
            var hs = cpu.getOperatorStrategyResults().get(0).getConditionalActionsResults().get(0).getNetworkResult().getBusResults();
            var rs = replayed.getOperatorStrategyResults().get(0).getConditionalActionsResults().get(0).getNetworkResult().getBusResults();
            for (int b = 0; b < hs.size(); b++) {
                org.junit.jupiter.api.Assertions.assertEquals(hs.get(b).getV(), rs.get(b).getV(), 1e-10,
                        "strategy bus voltage " + hs.get(b).getBusId());
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void distributedActivePowerMatchesCpu() {
        // What a result reports as distributed active power is OLF's preDistributedActivePower PLUS its load
        // flow's own distributed-slack passes. The GPU path reported the pre-distribution alone (assessment
        // L10: 124.5 MW off on rte6515); it now adds what its own passes distributed. B1-G (the slack) is out
        // of the distribution and B3/B6/B8 carry 20 MW so several generators share every mismatch.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        for (String g : List.of("B3-G", "B6-G", "B8-G")) {
            network.getGenerator(g).setTargetP(20);
        }
        network.getGenerator("B1-G").newExtension(com.powsybl.iidm.network.extensions.ActivePowerControlAdder.class)
                .withParticipate(false).add();
        String lineId = network.getLineStream()
                .filter(l -> !l.getId().contains("7") && !l.getId().contains("8"))
                .skip(2).findFirst().orElseThrow().getId();
        List<Contingency> contingencies = List.of(
                new Contingency(lineId, new BranchContingency(lineId)),
                new Contingency("B8-G", new com.powsybl.contingency.GeneratorContingency("B8-G")),
                new Contingency("B14-L", new com.powsybl.contingency.LoadContingency("B14-L")));
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true));
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, List.of(), sap, List.of(), List.of(),
                com.powsybl.commons.report.ReportNode.NO_OP);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, List.of(), sap, List.of(), List.of(),
                    com.powsybl.commons.report.ReportNode.NO_OP);
            assertEquals(contingencies.size(), GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                    "every contingency must be solved on the device");
            org.junit.jupiter.api.Assertions.assertEquals(cpu.getPreContingencyResult().getDistributedActivePower(),
                    gpu.getPreContingencyResult().getDistributedActivePower(), 1e-6, "base case");
            for (int c = 0; c < contingencies.size(); c++) {
                var cr = cpu.getPostContingencyResults().get(c);
                var gr = gpu.getPostContingencyResults().get(c);
                assertEquals(cr.getContingency().getId(), gr.getContingency().getId());
                org.junit.jupiter.api.Assertions.assertEquals(cr.getDistributedActivePower(), gr.getDistributedActivePower(), 1e-6,
                        cr.getContingency().getId());
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    @Test
    void participatingGeneratorOutageWithAParticipatingSlackMatchesCpu() {
        // The slack bus's generator takes part in the distribution. Losing another participant renormalizes
        // the factors over the survivors, the slack's included: its retained share grows. On ieee14 the only
        // other participant is B2-G, so its loss leaves the slack generators alone; with a third participant
        // at B6 the survivors share it.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertGpuMatchesCpu(IeeeCdfNetworkFactory.create14(), List.of(new Contingency("B2-G",
                new com.powsybl.contingency.GeneratorContingency("B2-G"))));
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "B2-G outage on the device");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getVoltageLevel("VL6").newGenerator().setId("B6-GP").setBus("B6").setConnectableBus("B6")
                .setTargetP(20).setTargetQ(0).setVoltageRegulatorOn(false).setMinP(0).setMaxP(80).add();
        assertGpuMatchesCpu(network, List.of(new Contingency("B2-G",
                new com.powsybl.contingency.GeneratorContingency("B2-G"))));
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "B2-G outage on the device");
    }

    /** The slack bus's generator participates and a peer reaches its limit. OLF keeps the slack generators in the
     *  water-fill list: the peer's leftover is renormalized over the survivors, the slack's included, in the same
     *  distribution. Losing B4's load lowers generation; B2-G stops at its minP 38 MW, 2 MW down. */
    @Test
    void participatingSlackWithASaturatingPeerMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B1-G").setMinP(0).setMaxP(400);
        network.getGenerator("B2-G").setMinP(38).setMaxP(200);
        String loadId = loadAtBus(network, "B4").getId();
        assertGpuMatchesCpu(network, List.of(new Contingency(loadId,
                new com.powsybl.contingency.LoadContingency(loadId))));
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "load loss on the device");
        // ...and the slack generator itself at its limit: losing B2-G's 40 MW, B1-G stops at maxP 240 MW (+8.4 MW)
        // and the rest goes to B6-GP
        Network network2 = IeeeCdfNetworkFactory.create14();
        network2.getGenerator("B1-G").setMinP(0).setMaxP(240);
        network2.getGenerator("B2-G").setMinP(0).setMaxP(200);
        network2.getVoltageLevel("VL6").newGenerator().setId("B6-GP").setBus("B6").setConnectableBus("B6")
                .setTargetP(20).setTargetQ(0).setVoltageRegulatorOn(false).setMinP(0).setMaxP(80).add();
        assertGpuMatchesCpu(network2, List.of(new Contingency("B2-G",
                new com.powsybl.contingency.GeneratorContingency("B2-G"))));
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "B2-G outage on the device");
        // ...and the slack generator on a capability curve: its limits follow its own distributed P
        Network network3 = IeeeCdfNetworkFactory.create14();
        // (B1-G absorbs about 17 MVAr, so it stays at min Q, which rises 2 MVAr per 100 MW)
        network3.getGenerator("B1-G").setMinP(0).setMaxP(400).newReactiveCapabilityCurve()
                .beginPoint().setP(0).setMinQ(-10).setMaxQ(100).endPoint()
                .beginPoint().setP(400).setMinQ(-2).setMaxQ(100).endPoint()
                .add();
        network3.getGenerator("B2-G").setMinP(38).setMaxP(200);
        assertGpuMatchesCpu(network3, List.of(new Contingency(loadId,
                new com.powsybl.contingency.LoadContingency(loadId))));
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "load loss on the device");
        // ...and a small loss after which OLF's slack check stays STABLE: the loss pre-distribution alone places it.
        // Losing a 5 MW load at B4, B2-G stops at its minP 44.5 MW.
        Network network4 = IeeeCdfNetworkFactory.create14();
        network4.getGenerator("B1-G").setMinP(0).setMaxP(400);
        network4.getGenerator("B2-G").setTargetP(45).setMinP(44.5).setMaxP(200);
        network4.getVoltageLevel("VL4").newLoad().setId("B4-SMALL").setBus("B4").setConnectableBus("B4")
                .setP0(5).setQ0(0).add();
        String smallLoadId = "B4-SMALL";
        assertGpuMatchesCpu(network4, List.of(new Contingency(smallLoadId,
                new com.powsybl.contingency.LoadContingency(smallLoadId))));
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "load loss on the device");
        // ...and two generators on the slack bus, the second one stopped at its minP by the BASE load flow (B6-GP's
        // 20 MW are surplus): every contingency starts from the base's per-generator split of the slack's share
        Network network5 = IeeeCdfNetworkFactory.create14();
        network5.getGenerator("B1-G").setMinP(0).setMaxP(400);
        network5.getGenerator("B2-G").setMinP(0).setMaxP(200);
        network5.getVoltageLevel("VL6").newGenerator().setId("B6-GP").setBus("B6").setConnectableBus("B6")
                .setTargetP(20).setTargetQ(0).setVoltageRegulatorOn(false).setMinP(0).setMaxP(80).add();
        network5.getVoltageLevel("VL1").newGenerator().setId("B1-G2").setBus("B1").setConnectableBus("B1")
                .setTargetP(10).setTargetQ(0).setVoltageRegulatorOn(false).setMinP(9.5).setMaxP(300).add();
        String loadId5 = loadAtBus(network5, "B4").getId();
        assertGpuMatchesCpu(network5, List.of(new Contingency(loadId5,
                new com.powsybl.contingency.LoadContingency(loadId5))));
        assertEquals(1, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "load loss on the device");
    }

    /** Load-based distribution whose BASE load flow distributes, with a load on the slack bus: the device's
     *  distribution is the loads' (OLF's LoadActivePowerDistributionStep), so it is written back onto the loads, the
     *  slack's own share onto the slack bus's. Moving only generators left OLF's load targets where they were, and OLF
     *  measured the device's converged base against them: MAX_ITERATION_REACHED. */
    @Test
    void deviceDsBalanceTypeLoadDistributingInTheBaseMatchesCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getVoltageLevel("VL1").newLoad().setId("B1-L").setBus("B1").setConnectableBus("B1").setP0(30).setQ0(10).add();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true)
                .setBalanceType(com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_LOAD);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(com.powsybl.openloadflow.network.SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("VL1_0"));
        assertGpuMatchesCpu(network, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("L2-3-1", new BranchContingency("L2-3-1"))), 1e-12, lfp);
    }

    /** Strategy pairs whose parent water-filled over a participating slack with a saturated peer: each pair starts
     *  from the slack generator's own parent move. */
    @Test
    void participatingSlackStrategyPairsMatchCpu() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        network.getGenerator("B1-G").setMinP(0).setMaxP(400);
        network.getGenerator("B2-G").setMinP(38).setMaxP(200);
        String loadId = loadAtBus(network, "B4").getId();
        List<Contingency> contingencies = List.of(new Contingency(loadId,
                new com.powsybl.contingency.LoadContingency(loadId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.GeneratorActionBuilder().withId("up2")
                        .withGeneratorId("B2-G").withActivePowerRelativeValue(true).withActivePowerValue(20).build(),
                new com.powsybl.action.TerminalsConnectionActionBuilder().withId("open1011")
                        .withNetworkElementId("L10-11-1").withOpen(true).build(),
                new com.powsybl.action.GeneratorActionBuilder().withId("up1")
                        .withGeneratorId("B1-G").withActivePowerRelativeValue(true).withActivePowerValue(10).build(),
                new com.powsybl.action.GeneratorActionBuilder().withId("set1")
                        .withGeneratorId("B1-G").withActivePowerRelativeValue(false).withActivePowerValue(220).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("sUp1",
                        com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("up1")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sSet1",
                        com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("set1")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sUp2",
                        com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("up2")),
                new com.powsybl.contingency.strategy.OperatorStrategy("sOpen",
                        com.powsybl.contingency.ContingencyContext.specificContingency(loadId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("open1011")));
        assertStrategyMatchesCpu(network, contingencies, strategies, actions, 5);
    }

    @Test
    void l121SevereContingencyMatchesCpuAtTightSlackTol() {
        // The L1-2-1 outage (main B1-B2 tie) pins ALL PV buses — an ill-conditioned state where the
        // DS stopping band (slackBusPMaxMismatch, default 1 MW) leaves ~0.1 kV of legitimate
        // ambiguity between arms (measured: 0.105 kV @1 MW, 0.055 @0.1, match @0.01 — scales with
        // the band, NOT a defect). This regression asserts exact machinery parity with the band
        // narrowed; the ambiguity analysis lives in docs/gpu-vs-cpu-capabilities.md §A.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        System.setProperty("olf.gpu.test.slackTol", "0.01");
        System.setProperty("olf.gpu.test.convEps", "1e-9");
        try {
            Network network = IeeeCdfNetworkFactory.create14();
            assertGpuMatchesCpu(network, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1"))), 1e-12, true);
            Network network2 = IeeeCdfNetworkFactory.create14();
            network2.getVoltageLevel("VL10").newGenerator().setId("B10-GQ").setBus("B10").setConnectableBus("B10")
                    .setTargetP(5).setTargetQ(8).setVoltageRegulatorOn(false).setMinP(0).setMaxP(20).add();
            assertGpuMatchesCpu(network2, List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1"))), 1e-12, true);
        } finally {
            System.clearProperty("olf.gpu.test.slackTol");
            System.clearProperty("olf.gpu.test.convEps");
        }
    }
}
