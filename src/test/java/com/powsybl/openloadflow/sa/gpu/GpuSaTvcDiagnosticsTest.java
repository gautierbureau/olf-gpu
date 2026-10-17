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
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.powsybl.openloadflow.sa.gpu.GpuSaTestNetworks.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-batched security analysis against the CPU: transformer voltage control diagnostics.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaTvcDiagnosticsTest extends AbstractGpuSaTest {

    GpuSaTvcDiagnosticsTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @Test
    void tvcSaVsFreshLfIterationCounts() {
        // CPU-ONLY: understand why the CPU sequential SA does not reproduce a fresh CPU LF for this
        // contingency (~6e-3 kV) BEFORE looking at the GPU path. Counts the number of Newton SOLVES
        // (mismatchInitial = 1 + outer-loop re-solves) — NOT olf.OuterLoopIteration, which the AFTER_GENERATOR
        // TransformerVoltageControlOuterLoop does not emit (only the incremental/RL loops do). Result:
        // fresh LF = 5 solves / 13 Newton (tap 3); CPU-SA post-contingency = 3 solves / 6 Newton. The TVC
        // DOES run in the post-contingency (initStep -> controlStep -> round), it just does fewer re-solves
        // from the warm start and lands ~6e-3 short (both at tap 3, since 6e-3 << the ~1.7 kV tap step). So
        // the gap is a warm-start convergence difference in the CPU SA, not a skipped outer loop. Reported at
        // IDENTICAL default config. No GPU involved.
        String[] bis = new String[1];
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            bis[0] = l0.getId() + "_BIS";
            network.newLine().setId(bis[0])
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        java.util.function.BiConsumer<com.powsybl.loadflow.LoadFlowParameters, com.powsybl.openloadflow.OpenLoadFlowParameters.ReportedFeatures> cfg = (lfp, feat) ->
                com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                        .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL)
                        .setNewtonRaphsonConvEpsPerEq(1e-4)
                        .setReportedFeatures(java.util.Set.of(feat));

        // FRESH LF at the post-contingency operating point (l0_BIS disconnected)
        Network pc = baseNet.get();
        pc.getLine(bis[0]).getTerminal1().disconnect();
        pc.getLine(bis[0]).getTerminal2().disconnect();
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        cfg.accept(lfp, com.powsybl.openloadflow.OpenLoadFlowParameters.ReportedFeatures.NEWTON_RAPHSON_LOAD_FLOW);
        com.powsybl.commons.report.ReportNode rLf = com.powsybl.commons.report.ReportNode.newRootReportNode()
                .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                .withMessageTemplate("olf.securityAnalysis").build();
        loadFlowProvider.run(pc, computationManager, pc.getVariantManager().getWorkingVariantId(), lfp, rLf).join();
        java.util.Map<String, Integer> tLf = new java.util.TreeMap<>();
        countReportKeys(rLf, tLf);
        double lfV3 = pc.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        System.err.printf("[ITER] FRESH-LF  V(BUS_3)=%.6f  newtonSolves=%d  newtonIter=%d  tap=%d%n", lfV3,
                tLf.getOrDefault("olf.mismatchInitial", 0),
                tLf.getOrDefault("olf.mismatchInitial", 0) + tLf.getOrDefault("olf.mismatchIteration", 0),
                pc.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition());

        // CPU sequential SA on the contingency
        Network saNet = baseNet.get();
        List<Contingency> ctgs = List.of(new Contingency(bis[0], new BranchContingency(bis[0])));
        com.powsybl.loadflow.LoadFlowParameters saLfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
        cfg.accept(saLfp, com.powsybl.openloadflow.OpenLoadFlowParameters.ReportedFeatures.NEWTON_RAPHSON_SECURITY_ANALYSIS);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(saLfp);
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                saNet.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        com.powsybl.commons.report.ReportNode rSa = com.powsybl.commons.report.ReportNode.newRootReportNode()
                .withResourceBundles(com.powsybl.openloadflow.util.report.PowsyblOpenLoadFlowReportResourceBundle.BASE_NAME)
                .withMessageTemplate("olf.securityAnalysis").build();
        SecurityAnalysisResult saCpu = runSecurityAnalysis(saNet, ctgs, mon, sap, rSa);
        double saV3 = saCpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
        // isolate the POST-contingency subtree (exclude the base N run)
        var preNode = findSubtree(rSa, "olf.preContingencySimulation");
        var postNode = findSubtree(rSa, "olf.postContingencySimulation");
        java.util.Map<String, Integer> tPre = new java.util.TreeMap<>();
        java.util.Map<String, Integer> tPost = new java.util.TreeMap<>();
        if (preNode != null) {
            countReportKeys(preNode, tPre);
        }
        if (postNode != null) {
            countReportKeys(postNode, tPost);
        }
        System.err.printf("[ITER] CPU-SA base N       newtonSolves=%d  newtonIter=%d%n",
                tPre.getOrDefault("olf.mismatchInitial", 0),
                tPre.getOrDefault("olf.mismatchInitial", 0) + tPre.getOrDefault("olf.mismatchIteration", 0));
        System.err.printf("[ITER] CPU-SA post-ctg     V(BUS_3)=%.6f  newtonSolves=%d  newtonIter=%d%n", saV3,
                tPost.getOrDefault("olf.mismatchInitial", 0),
                tPost.getOrDefault("olf.mismatchInitial", 0) + tPost.getOrDefault("olf.mismatchIteration", 0));
        System.err.println("[ITER] CPU-SA post-ctg keys: " + tPost);
    }

    @Test
    void tvcWarmVsFlatPostContingencyDiagnostic() {
        // CPU-ONLY: does a WARM-started post-contingency LF (base LF -> apply contingency -> LF from previous
        // values, like the CPU SA) converge to a DIFFERENT point than a FRESH flat LF of the same operating
        // point? If yes (and different tap or ~6e-3 V), the CPU SA's ~6e-3 is a warm-start non-robustness, and
        // the GPU (which reproduces the flat LF) is the more accurate. Reads the SOLVED TAP on each.
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mk = () -> {
            com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            return lfp;
        };
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));

        // FRESH FLAT LF at the post-contingency point (uniform init)
        Network flat = baseNet.get();
        flat.getLine(bis).getTerminal1().disconnect();
        flat.getLine(bis).getTerminal2().disconnect();
        runner.run(flat, mk.get());
        double flatV = flat.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        int flatTap = flat.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition();

        // WARM: base LF (N), then apply contingency, then LF from PREVIOUS_VALUES (like the CPU SA warm-start)
        Network warm = baseNet.get();
        runner.run(warm, mk.get());                                   // base N
        int baseTap = warm.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition();
        warm.getLine(bis).getTerminal1().disconnect();
        warm.getLine(bis).getTerminal2().disconnect();
        com.powsybl.loadflow.LoadFlowParameters warmLfp = mk.get();
        warmLfp.setVoltageInitMode(com.powsybl.loadflow.LoadFlowParameters.VoltageInitMode.PREVIOUS_VALUES);
        runner.run(warm, warmLfp);                                    // warm post-contingency
        double warmV = warm.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        int warmTap = warm.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition();

        System.err.printf("[WARM-VS-FLAT] baseN tap=%d | FLAT V=%.6f tap=%d | WARM V=%.6f tap=%d | d(warm,flat)=%.3e%n",
                baseTap, flatV, flatTap, warmV, warmTap, Math.abs(warmV - flatV));
    }

    /** CPU-ONLY isolation: which outer loop makes the CPU SA post-contingency diverge from a plain LF of the
     *  SAME operating point? Toggles TVC / distributed slack / reactive limits independently, at DEFAULT
     *  tolerance (never tighten). For each combination prints plainLF vs CPU-SA V(BUS_3). */
    @Test
    void cpuSaVsPlainLfFeatureIsolation() {
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        boolean[][] combos = {{true, true, true}, {true, false, true}, {true, true, false}, {false, true, true}, {true, false, false}};
        for (boolean[] c : combos) {
            boolean tvc = c[0];
            boolean ds = c[1];
            boolean rl = c[2];
            java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mk = () -> {
                com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                        .setTransformerVoltageControlOn(tvc).setDistributedSlack(ds).setUseReactiveLimits(rl);
                com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                        .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
                return p;   // DEFAULT convEps / slack mismatch
            };
            Network flat = baseNet.get();
            flat.getLine(bis).getTerminal1().disconnect();
            flat.getLine(bis).getTerminal2().disconnect();
            runner.run(flat, mk.get());
            double flatV = flat.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
            Network saNet = baseNet.get();
            com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
            sap.setLoadFlowParameters(mk.get());
            List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                    saNet.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
            double saV = runSecurityAnalysis(saNet, List.of(new Contingency(bis, new BranchContingency(bis))), mon, sap)
                    .getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                    .filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            System.err.printf("[ISO] tvc=%-5b ds=%-5b rl=%-5b  plainLF=%.6f  CPU-SA=%.6f  d=%.3e%n",
                    tvc, ds, rl, flatV, saV, Math.abs(flatV - saV));
        }
    }

    /** CPU-ONLY: the DS mechanism. Compare the DISTRIBUTED ACTIVE POWER (and the resulting V) of a plain
     *  post-contingency LF vs the CPU SA's post-contingency. Distributed slack stops inside a band
     *  (slackBusPMaxMismatch, default 1 MW), so two paths reaching the contingency from DIFFERENT generation
     *  splits (fresh vs base-distributed) legitimately stop at different points inside that band. */
    @Test
    void cpuSaDistributedPowerDiagnostic() {
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mk = () -> {
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(true).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            return p;
        };
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        // plain LF at the post-contingency point, from a FRESH network
        Network flat = baseNet.get();
        flat.getLine(bis).getTerminal1().disconnect();
        flat.getLine(bis).getTerminal2().disconnect();
        com.powsybl.loadflow.LoadFlowResult flatRes = runner.run(flat, mk.get());
        System.err.printf("[DS] plain post-ctg LF: distributedActivePower=%.6f MW  V(BUS_3)=%.6f  genP=%s%n",
                flatRes.getComponentResults().get(0).getDistributedActivePower(),
                flat.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV(),
                flat.getGeneratorStream().map(g -> String.format("%s=%.4f", g.getId(), g.getTerminal().getP())).toList());
        // CPU SA: base N + post-contingency
        Network saNet = baseNet.get();
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(mk.get());
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                saNet.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        SecurityAnalysisResult sa = runSecurityAnalysis(saNet, List.of(new Contingency(bis, new BranchContingency(bis))), mon, sap);
        System.err.printf("[DS] CPU-SA preContingency: distributedActivePower=%.6f MW%n",
                sa.getPreContingencyResult().getDistributedActivePower());
        System.err.printf("[DS] CPU-SA postContingency: distributedActivePower=%.6f MW  V(BUS_3)=%.6f%n",
                sa.getPostContingencyResults().get(0).getDistributedActivePower(),
                sa.getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                        .filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV());
        // Same measurement on the GPU SA paths — does the GPU distribute slack where the CPU SA distributes 0?
        if (com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable()) {
            securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            SecurityAnalysisResult gpu;
            try {
                gpu = runSecurityAnalysis(baseNet.get(), List.of(new Contingency(bis, new BranchContingency(bis))), mon, sap);
            } finally {
                securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            }
            System.err.printf("[DS] GPU-SA postContingency: distributedActivePower=%.6f MW  V(BUS_3)=%.6f%n",
                    gpu.getPostContingencyResults().get(0).getDistributedActivePower(),
                    gpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                            .filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV());
            // CROSS-CHECK with distributed slack OFF: if GPU-SA == CPU-SA there, the whole GPU-vs-CPU SA
            // discrepancy is DS-orchestration, and DS is the single thing to align.
            com.powsybl.loadflow.LoadFlowParameters noDs = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(noDs)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            com.powsybl.security.SecurityAnalysisParameters sapNoDs = new com.powsybl.security.SecurityAnalysisParameters();
            sapNoDs.setLoadFlowParameters(noDs);
            double cpuNoDs = runSecurityAnalysis(baseNet.get(), List.of(new Contingency(bis, new BranchContingency(bis))), mon, sapNoDs)
                    .getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                    .filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            double gpuNoDs;
            try {
                gpuNoDs = runSecurityAnalysis(baseNet.get(), List.of(new Contingency(bis, new BranchContingency(bis))), mon, sapNoDs)
                        .getPostContingencyResults().get(0).getNetworkResult().getBusResults().stream()
                        .filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            } finally {
                securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            }
            System.err.printf("[DS-OFF] CPU-SA=%.6f  GPU-SA=%.6f  d=%.3e%n", cpuNoDs, gpuNoDs, Math.abs(cpuNoDs - gpuNoDs));
        }
    }

    /** GPU-vs-CPU SA isolation at DEFAULT tolerance: which outer-loop feature drives the GPU's ~5.5e-3?
     *  Compares CPU-SA / GPU-SA (and the plain LF reference) with TVC / DS / RL toggled. */
    @Test
    void gpuVsCpuSaFeatureIsolation() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)));
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        boolean[][] combos = {{true, true, true}, {true, false, true}, {false, true, true}, {false, true, false}, {false, false, true}, {false, false, false}};
        for (boolean[] c : combos) {
            boolean tvc = c[0];
            boolean ds = c[1];
            boolean rl = c[2];
            java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mk = () -> {
                com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                        .setTransformerVoltageControlOn(tvc).setDistributedSlack(ds).setUseReactiveLimits(rl);
                com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                        .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
                return p;
            };
            Network flat = baseNet.get();
            flat.getLine(bis).getTerminal1().disconnect();
            flat.getLine(bis).getTerminal2().disconnect();
            runner.run(flat, mk.get());
            double flatV = flat.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
            com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
            sap.setLoadFlowParameters(mk.get());
            Network probe = baseNet.get();
            List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                    probe.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
            java.util.function.Function<SecurityAnalysisResult, Double> v3 = r -> r.getPostContingencyResults().get(0)
                    .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            double cpu = v3.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
            securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            double gpu;
            int engaged;
            try {
                gpu = v3.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
                engaged = GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get();
            } finally {
                securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            }
            System.err.printf("[GPU-ISO] tvc=%-5b ds=%-5b rl=%-5b | plainLF=%.6f CPU=%.6f | GPU=%.6f (d=%.1e eng=%d)%n",
                    tvc, ds, rl, flatV, cpu, gpu, Math.abs(cpu - gpu), engaged);
        }
    }

    /** Is the GPU TVC error already present in the PRE-contingency (N) result? tvc=on, ds=off (clean case).
     *  If N already differs, it is the base LF / TVC handling, not the contingency orchestration. */
    @Test
    void gpuTvcPreContingencyDiagnostic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
        // plain N load flow (no contingency), CPU vs GPU solver
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        Network nc = baseNet.get();
        runner.run(nc, lfp);
        com.powsybl.loadflow.LoadFlowParameters gl = new com.powsybl.loadflow.LoadFlowParameters()
                .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(gl)
                .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL)
                .setAcSolverType(com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory.NAME);
        Network ng = baseNet.get();
        runner.run(ng, gl);
        System.err.printf("[N-LF] plain N: CPU=%.9f tap=%d | GPU=%.9f tap=%d | d=%.3e%n",
                nc.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV(),
                nc.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition(),
                ng.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV(),
                ng.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition(),
                Math.abs(nc.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV()
                        - ng.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV()));
        // SA pre-contingency (N) result, CPU vs GPU providers
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        Network probe = baseNet.get();
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                probe.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)));
        java.util.function.Function<SecurityAnalysisResult, Double> preV = r -> r.getPreContingencyResult()
                .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
        double cpuPre = preV.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        double gpuPre;
        try {
            gpuPre = preV.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        System.err.printf("[N-SA] SA preContingency: CPU=%.9f  GPU=%.9f  d=%.3e%n", cpuPre, gpuPre, Math.abs(cpuPre - gpuPre));
    }

    /** The GPU batch's ROUND-0 equation system (post-contingency, TVC control OFF, tap pinned at 3) solved as a
     *  plain LF on CPU vs GPU. The batch's round 0 reported V=34.425712 (ds=off); if the CPU gives 34.4311 here,
     *  the GPU's very first post-contingency solve is already wrong, before any TVC action. */
    @Test
    void tvcRound0EquivalentDiagnostic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> mkNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            // tap 3 (rho 1.1) = where the base LF leaves it; control OFF so r1 stays pinned there
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(false)
                    .setTapPosition(3).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            network.getLine(l0.getId() + "_BIS").getTerminal1().disconnect();
            network.getLine(l0.getId() + "_BIS").getTerminal2().disconnect();
            return network;
        };
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        for (boolean gpu : new boolean[] {false, true}) {
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
            var ext = com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            if (gpu) {
                ext.setAcSolverType(com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory.NAME);
            }
            Network net = mkNet.get();
            runner.run(net, p);
            System.err.printf("[ROUND0] %s: V(BUS_3)=%.9f  tap=%d%n", gpu ? "GPU" : "CPU",
                    net.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV(),
                    net.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getTapPosition());
        }
    }

    /** DECISIVE test of the stale-admittance hypothesis: OLF's PiModelArray returns R/X/Y/ksi from the CURRENT
     *  TAP, but the GPU bakes the admittance into cbIn at extraction. Build the SAME case with every tap step
     *  sharing IDENTICAL R/X (so a tap change cannot alter the admittance) and compare GPU vs CPU SA. If the
     *  ~5.5e-3 collapses to machine precision, the stale device admittance on tap change is the bug. */
    @Test
    void tvcUniformTapImpedanceDiagnostic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        for (boolean uniform : new boolean[] {false, true}) {
            java.util.function.Supplier<Network> mkNet = () -> {
                Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
                com.powsybl.iidm.network.TwoWindingsTransformer t = network.getTwoWindingsTransformer("T2wT");
                if (uniform) {
                    // force every tap step to the SAME R/X (only rho differs)
                    double r0 = t.getRatioTapChanger().getStep(0).getR();
                    double x0 = t.getRatioTapChanger().getStep(0).getX();
                    for (int i = t.getRatioTapChanger().getLowTapPosition(); i <= t.getRatioTapChanger().getHighTapPosition(); i++) {
                        t.getRatioTapChanger().getStep(i).setR(r0).setX(x0);
                    }
                }
                t.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                        .setRegulationTerminal(t.getTerminal2()).setTargetV(34.0);
                com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
                network.newLine().setId(l0.getId() + "_BIS")
                        .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                        .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                        .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
                return network;
            };
            String bis = mkNet.get().getLines().iterator().next().getId() + "_BIS";
            com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
            sap.setLoadFlowParameters(lfp);
            List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)));
            List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                    mkNet.get().getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
            java.util.function.Function<SecurityAnalysisResult, Double> v3 = r -> r.getPostContingencyResults().get(0)
                    .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            double cpu = v3.apply(runSecurityAnalysis(mkNet.get(), ctgs, mon, sap));
            securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            double gpu;
            try {
                gpu = v3.apply(runSecurityAnalysis(mkNet.get(), ctgs, mon, sap));
            } finally {
                securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            }
            System.err.printf("[TAP-Z] uniformTapImpedance=%-5b  CPU=%.9f  GPU=%.9f  d=%.3e%n", uniform, cpu, gpu, Math.abs(cpu - gpu));
        }
    }

    /** Is the CPU TVC answer a FIXED POINT of the tap it lands on? Run the plain post-contingency LF with TVC
     *  (lands at tap 3, V=34.4311), then FREEZE that tap (regulating=false) and re-solve. If V moves to the
     *  fixed-tap solution (~34.4254 — which is where the GPU batch lands), the TVC path's reported state is not
     *  a converged solution of its own final tap system. CPU-only, default tolerance. */
    @Test
    void tvcResultIsFixedPointDiagnostic() {
        java.util.function.Supplier<Network> mkNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            com.powsybl.iidm.network.TwoWindingsTransformer t = network.getTwoWindingsTransformer("T2wT");
            t.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                    .setRegulationTerminal(t.getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            network.getLine(l0.getId() + "_BIS").getTerminal1().disconnect();
            network.getLine(l0.getId() + "_BIS").getTerminal2().disconnect();
            return network;
        };
        java.util.function.Supplier<com.powsybl.loadflow.LoadFlowParameters> mk = () -> {
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            return p;
        };
        com.powsybl.loadflow.LoadFlow.Runner runner = new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory()));
        Network net = mkNet.get();
        runner.run(net, mk.get());
        int solvedTap = net.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition();
        double vTvc = net.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        System.err.printf("[FIXPOINT] TVC LF: V=%.9f solvedTap=%d (tapPosition=%d)%n", vTvc, solvedTap,
                net.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getTapPosition());
        // freeze that tap and re-solve
        net.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setRegulating(false).setTapPosition(solvedTap);
        runner.run(net, mk.get());
        double vFrozen = net.getVoltageLevel("VL_3").getBusView().getBusStream().findFirst().orElseThrow().getV();
        System.err.printf("[FIXPOINT] re-solved with tap FROZEN at %d: V=%.9f  (moved by %.3e)%n",
                solvedTap, vFrozen, Math.abs(vFrozen - vTvc));
    }

    /** Find a targetV where the CONTINGENCY MOVES THE TAP (base tap != post-contingency tap) — the case that
     *  exercises the device pi-model refresh (a tap change alters R/X/Y/ksi in OLF's PiModelArray). */
    @Test
    void findTapMovingContingencyDiagnostic() {
        java.util.function.BiFunction<Double, Boolean, int[]> run = (targetV, outage) -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            com.powsybl.iidm.network.TwoWindingsTransformer t = network.getTwoWindingsTransformer("T2wT");
            t.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0).setRegulationTerminal(t.getTerminal2()).setTargetV(targetV);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            if (outage) {
                network.getLine(l0.getId()).getTerminal1().disconnect();   // outage the ORIGINAL line (bigger change)
                network.getLine(l0.getId()).getTerminal2().disconnect();
            }
            com.powsybl.loadflow.LoadFlowParameters p = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(true).setDistributedSlack(false).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(p)
                    .setTransformerVoltageControlMode(com.powsybl.openloadflow.OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
            new com.powsybl.loadflow.LoadFlow.Runner(new com.powsybl.openloadflow.OpenLoadFlowProvider(commonTestConfig.matrixFactory())).run(network, p);
            return new int[] {network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().getSolvedTapPosition()};
        };
        for (double tv : new double[] {30.0, 31.0, 32.0, 32.5, 33.0, 33.5, 34.0, 35.0, 36.0, 37.0, 38.0}) {
            int baseTap = run.apply(tv, false)[0];
            int ctgTap = run.apply(tv, true)[0];
            System.err.printf("[TAPMOVE] targetV=%.1f  baseTap=%d  postCtgTap=%d %s%n",
                    tv, baseTap, ctgTap, baseTap != ctgTap ? "  <== TAP MOVES" : "");
        }
    }

    /** DIAGNOSTIC (isolation only, not validation): does the device DS loop distribute when the CPU's band says
     *  "do nothing"? Run the DS-only case with the DEFAULT band (1 MW) and with an effectively-infinite band
     *  (1e6 MW, both sides identically). CPU distributes 0 in both -> identical CPU results. If DEVICE matches
     *  CPU with the infinite band but not with the default band, its gate/distribution fires when OLF's does not. */
    @Test
    void deviceDsBandDiagnostic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        java.util.function.Supplier<Network> baseNet = () -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createNetworkWithT2wt();
            network.getTwoWindingsTransformer("T2wT").getRatioTapChanger().setTargetDeadband(0).setRegulating(true)
                    .setTapPosition(0).setRegulationTerminal(network.getTwoWindingsTransformer("T2wT").getTerminal2()).setTargetV(34.0);
            com.powsybl.iidm.network.Line l0 = network.getLines().iterator().next();
            network.newLine().setId(l0.getId() + "_BIS")
                    .setVoltageLevel1(l0.getTerminal1().getVoltageLevel().getId()).setBus1(l0.getTerminal1().getBusBreakerView().getBus().getId())
                    .setVoltageLevel2(l0.getTerminal2().getVoltageLevel().getId()).setBus2(l0.getTerminal2().getBusBreakerView().getBus().getId())
                    .setR(l0.getR() * 2).setX(l0.getX() * 2).setG1(l0.getG1()).setB1(l0.getB1()).setG2(l0.getG2()).setB2(l0.getB2()).add();
            return network;
        };
        String bis = baseNet.get().getLines().iterator().next().getId() + "_BIS";
        for (double band : new double[] {1.0, 1e6}) {
            com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                    .setTransformerVoltageControlOn(false).setDistributedSlack(true).setUseReactiveLimits(true);
            com.powsybl.openloadflow.OpenLoadFlowParameters.create(lfp).setSlackBusPMaxMismatch(band);
            com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
            sap.setLoadFlowParameters(lfp);
            List<Contingency> ctgs = List.of(new Contingency(bis, new BranchContingency(bis)));
            List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.all(), java.util.Set.of(),
                    baseNet.get().getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
            java.util.function.Function<SecurityAnalysisResult, Double> v3 = r -> r.getPostContingencyResults().get(0)
                    .getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals("BUS_3")).findFirst().orElseThrow().getV();
            double cpu = v3.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
            securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            double dev;
            try {
                dev = v3.apply(runSecurityAnalysis(baseNet.get(), ctgs, mon, sap));
            } finally {
                securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
            }
            System.err.printf("[DS-BAND] band=%.0e MW  CPU=%.9f  DEV=%.9f  d=%.3e%n", band, cpu, dev, Math.abs(cpu - dev));
        }
    }

    /** DIAGNOSTIC for the DEVICE operator-strategy TEMPORARY tolerance: which leg diverges — the
     *  post-contingency solve (should be machine-exact after the fidelity fixes) or the STRATEGY leg
     *  (warm-continued pair scenario)? Prints per-leg CPU-vs-GPU max dV. Default tolerance. */
    @Test
    void deviceStrategyLegIsolationDiagnostic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        String lineId = network.getLineStream()
                .filter(l -> !l.getId().contains("7") && !l.getId().contains("8"))
                .skip(1).findFirst().orElseThrow().getId();
        com.powsybl.iidm.network.Load load12 = network.getLoadStream()
                .filter(l -> "B12".equals(l.getTerminal().getBusBreakerView().getBus().getId()))
                .findFirst().orElseThrow();
        List<Contingency> contingencies = List.of(new Contingency(lineId, new BranchContingency(lineId)));
        List<com.powsybl.action.Action> actions = List.of(
                new com.powsybl.action.LoadActionBuilder().withId("shed12").withLoadId(load12.getId())
                        .withRelativeValue(false).withActivePowerValue(load12.getP0() / 2).build());
        List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of(
                new com.powsybl.contingency.strategy.OperatorStrategy("s1",
                        com.powsybl.contingency.ContingencyContext.specificContingency(lineId),
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("shed12")));
        com.powsybl.loadflow.LoadFlowParameters lfp = new com.powsybl.loadflow.LoadFlowParameters()
                .setDistributedSlack(true).setUseReactiveLimits(true);
        com.powsybl.security.SecurityAnalysisParameters sap = new com.powsybl.security.SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        List<com.powsybl.security.monitor.StateMonitor> mon = List.of(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.all(),
                network.getBranchStream().map(br -> br.getId()).collect(java.util.stream.Collectors.toSet()),
                network.getVoltageLevelStream().map(vl -> vl.getId()).collect(java.util.stream.Collectors.toSet()), java.util.Set.of()));
        if (System.getenv("OLF_GPU_DBG_RL") != null) {
            ch.qos.logback.classic.Logger rlLog = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger("com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop");
            rlLog.setLevel(ch.qos.logback.classic.Level.TRACE);
            ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
            if (!root.iteratorForAppenders().hasNext()) {
                ch.qos.logback.classic.encoder.PatternLayoutEncoder enc = new ch.qos.logback.classic.encoder.PatternLayoutEncoder();
                enc.setContext(root.getLoggerContext());
                enc.setPattern("%msg%n");
                enc.start();
                ch.qos.logback.core.ConsoleAppender<ch.qos.logback.classic.spi.ILoggingEvent> ca = new ch.qos.logback.core.ConsoleAppender<>();
                ca.setContext(root.getLoggerContext());
                ca.setEncoder(enc);
                ca.start();
                rlLog.addAppender(ca);
            }
        }
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, mon, sap, strategies, actions,
                com.powsybl.commons.report.ReportNode.NO_OP);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        SecurityAnalysisResult gpu;
        try {
            gpu = runSecurityAnalysis(network, contingencies, mon, sap, strategies, actions,
                    com.powsybl.commons.report.ReportNode.NO_OP);
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        double worstCtg = 0;
        var cb = cpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults();
        var gb = gpu.getPostContingencyResults().get(0).getNetworkResult().getBusResults();
        for (int b = 0; b < cb.size(); b++) {
            worstCtg = Math.max(worstCtg, Math.abs(cb.get(b).getV() - gb.get(b).getV()));
        }
        double worstStrat = 0;
        var cs = cpu.getOperatorStrategyResults().get(0).getNetworkResult().getBusResults();
        var gs = gpu.getOperatorStrategyResults().get(0).getNetworkResult().getBusResults();
        for (int b = 0; b < cs.size(); b++) {
            worstStrat = Math.max(worstStrat, Math.abs(cs.get(b).getV() - gs.get(b).getV()));
        }
        System.err.printf("[STRAT-ISO] post-ctg maxDV=%.3e | STRATEGY leg maxDV=%.3e%n", worstCtg, worstStrat);
        for (var cbr : cpu.getOperatorStrategyResults().get(0).getNetworkResult().getBranchResults()) {
            var gbr = gpu.getOperatorStrategyResults().get(0).getNetworkResult().getBranchResults().stream()
                    .filter(x -> x.getBranchId().equals(cbr.getBranchId())).findFirst().orElse(null);
            if (gbr != null && Math.abs(cbr.getP1() - gbr.getP1()) > 0.1) {
                System.err.printf("[STRAT-ISO] branch %s P1 CPU=%.4f GPU=%.4f dP=%.4f MW%n",
                        cbr.getBranchId(), cbr.getP1(), gbr.getP1(), cbr.getP1() - gbr.getP1());
            }
        }
        System.err.printf("[STRAT-ISO] distributedP post-ctg CPU=%.6f GPU=%.6f MW%n",
                cpu.getPostContingencyResults().get(0).getDistributedActivePower(),
                gpu.getPostContingencyResults().get(0).getDistributedActivePower());
        for (int b = 0; b < cs.size(); b++) {
            double d = Math.abs(cs.get(b).getV() - gs.get(b).getV());
            if (d > worstStrat * 0.3) {
                System.err.printf("[STRAT-ISO] strategy %s CPU=%.6f GPU=%.6f d=%.3e%n",
                        cs.get(b).getBusId(), cs.get(b).getV(), gs.get(b).getV(), d);
            }
        }
    }
}
