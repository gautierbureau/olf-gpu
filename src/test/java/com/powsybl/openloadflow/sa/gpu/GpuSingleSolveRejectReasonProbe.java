/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The experiment: single GPU LF of a disconnected branch vs CPU, ON DEVICE (fallback DISABLED so a
 * rejection throws instead of silently faking CPU-identical results). Reports worst per-bus |v| gap
 * and the top divergent buses. eps/slackTol overridable via OLF_GPU_DISC_EPS / OLF_GPU_DISC_SLACKTOL.
 */
class GpuSingleSolveRejectReasonProbe {

    private static final Path CASES = GpuTestPaths.caseDir();
    private static final String CASE = "case9241pegase.xiidm.gz";

    static {
        GpuTestPaths.init();
    }

    @BeforeEach
    void check() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available");
        assumeTrue(Files.exists(CASES.resolve(CASE)), CASE + " not found");
        assumeTrue(Files.exists(Path.of("rte-loadflow-default-parameters.json")), "rte params not found");
    }

    private static LoadFlowParameters params() throws Exception {
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(Path.of("rte-loadflow-default-parameters.json"));
        String eps = System.getenv("OLF_GPU_DISC_EPS");
        if (eps != null && !eps.isBlank()) {
            OpenLoadFlowParameters.create(lfp).setNewtonRaphsonConvEpsPerEq(Double.parseDouble(eps));
        }
        String st = System.getenv("OLF_GPU_DISC_SLACKTOL");
        if (st != null && !st.isBlank()) {
            OpenLoadFlowParameters.create(lfp).setSlackBusPMaxMismatch(Double.parseDouble(st) * 100.0);
        }
        return lfp;
    }

    private static Map<String, Double> solve(boolean gpu, String branchId) throws Exception {
        Network net = Network.read(CASES.resolve(CASE));
        if (!"NONE".equals(branchId)) {
            Branch<?> br = net.getBranch(branchId);
            br.getTerminal1().disconnect();
            br.getTerminal2().disconnect();
        }
        LoadFlowParameters lfp = params();
        if (gpu) {
            OpenLoadFlowParameters.create(lfp).setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        }
        LoadFlowResult r = new LoadFlow.Runner(new OpenLoadFlowProvider(new SparseMatrixFactory())).run(net, lfp);
        System.out.printf("  %s converged=%s%n", gpu ? "GPU" : "CPU", r.isFullyConverged());
        Map<String, Double> v = new HashMap<>();
        for (Bus b : net.getBusView().getBuses()) {
            if (!Double.isNaN(b.getV())) {
                v.put(b.getId(), b.getV() / b.getVoltageLevel().getNominalV());
            }
        }
        return v;
    }


    @Test
    void probe() throws Exception {
        String branchId = System.getenv().getOrDefault("OLF_GPU_DISC_CTG", "LINE-4985-165");
        System.out.printf("%n=== SINGLE-LF GPU-vs-CPU (on device) %s | eps=%s slackTol=%s ===%n", branchId,
                System.getenv().getOrDefault("OLF_GPU_DISC_EPS", "prod"),
                System.getenv().getOrDefault("OLF_GPU_DISC_SLACKTOL", "prod-1MW"));
        Map<String, Double> cpu = solve(false, branchId);
        Map<String, Double> gpu = solve(true, branchId);
        List<Map.Entry<String, Double>> diffs = new ArrayList<>();
        double sum = 0;
        int nn = 0;
        for (Map.Entry<String, Double> e : cpu.entrySet()) {
            Double g = gpu.get(e.getKey());
            if (g != null) {
                double d = Math.abs(e.getValue() - g);
                diffs.add(Map.entry(e.getKey(), d));
                sum += d;
                nn++;
            }
        }
        diffs.sort(Comparator.comparingDouble((Map.Entry<String, Double> e) -> e.getValue()).reversed());
        System.out.printf("  ALL-BUS |v_gpu - v_cpu| (pu): worst=%.3e mean=%.3e over %d buses%n",
                diffs.isEmpty() ? 0 : diffs.get(0).getValue(), sum / nn, nn);
        System.out.println("  top-8 divergent buses:");
        diffs.stream().limit(8).forEach(e ->
                System.out.printf("    %-12s d=%.3e  CPU=%.6f GPU=%.6f%n",
                        e.getKey(), e.getValue(), cpu.get(e.getKey()), gpu.get(e.getKey())));
    }
}
