/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi.gpu;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.ContingencyContext;
import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Generator;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.sensi.OpenSensitivityAnalysisParameters;
import com.powsybl.openloadflow.sensi.OpenSensitivityAnalysisProvider;
import com.powsybl.sensitivity.*;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU against CPU sensitivity analysis on a real network: OLF's {@link OpenSensitivityAnalysisProvider} and the GPU
 * provider on the same network, contingencies, factors and parameters (the RTE production file at the module root),
 * SPARSE matrix factory on both. Factors: the active flow of 200 branches against the injections of the 20 largest
 * generators, and the voltage of 50 buses against the target voltages of the 10 largest. Prints the wall time of each
 * run (GPU cold then twice warm, CPU twice on 4 threads and once on 1), how many contingencies the device batched,
 * solved the basis of and contracted, and the largest difference against OLF's values per function type.
 *
 * <p>Machine-local measurement, not a gate: runs only with {@code OLF_GPU_SENSI_BENCH} set, the GPU library, the case
 * and {@code OLF_GPU_CONT_FILE}. Scope selectors (docs/env-vars.md): {@code OLF_GPU_SENSI_CASE} (default
 * case9241pegase), {@code OLF_GPU_SENSI_NCTG} (the first contingencies of the file, default all of them).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuVsCpuSensitivityBenchmarkTest {

    private static int env(String name, int defaultValue) {
        String value = System.getenv(name);
        return value != null && !value.isBlank() ? Integer.parseInt(value.trim()) : defaultValue;
    }

    private record Run(Map<List<Integer>, SensitivityValue> values, List<SensitivityAnalysisResult.SensitivityStateStatus> statuses,
                       double seconds) {
    }

    private static Run run(SensitivityAnalysisProvider provider, Network network, List<SensitivityFactor> factors,
                           List<Contingency> contingencies, SensitivityAnalysisParameters parameters) {
        SensitivityResultModelWriter writer = new SensitivityResultModelWriter(contingencies, List.of());
        long start = System.nanoTime();
        provider.run(network, network.getVariantManager().getWorkingVariantId(), new SensitivityFactorModelReader(factors, network),
                writer, new SensitivityAnalysisRunParameters().setParameters(parameters).setContingencies(contingencies)).join();
        double seconds = (System.nanoTime() - start) / 1e9;
        Map<List<Integer>, SensitivityValue> values = new HashMap<>();
        for (SensitivityValue value : writer.getValues()) {
            values.put(List.of(value.getFactorIndex(), value.getContingencyIndex()), value);
        }
        return new Run(values, writer.getStateStatuses(), seconds);
    }

    private static SensitivityAnalysisParameters parameters(Path file, int threads) {
        SensitivityAnalysisParameters parameters = new SensitivityAnalysisParameters();
        parameters.setLoadFlowParameters(JsonLoadFlowParameters.read(file));
        parameters.addExtension(OpenSensitivityAnalysisParameters.class, new OpenSensitivityAnalysisParameters().setThreadCount(threads));
        return parameters;
    }

    private static List<SensitivityFactor> factors(Network network) {
        int nBranch = 200;
        int nGen = 20;
        int nBus = 50;
        List<Generator> generators = network.getGeneratorStream()
                .filter(g -> g.getTerminal().getBusView().getBus() != null)
                .sorted(Comparator.comparingDouble(Generator::getTargetP).reversed().thenComparing(Generator::getId))
                .limit(nGen).toList();
        List<Branch> branches = network.getBranchStream()
                .filter(b -> b.getTerminal1().getBusView().getBus() != null && b.getTerminal2().getBusView().getBus() != null)
                .sorted(Comparator.comparing(Branch::getId)).limit(nBranch).toList();
        List<SensitivityFactor> factors = new ArrayList<>();
        for (Branch<?> branch : branches) {
            for (Generator generator : generators) {
                factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_ACTIVE_POWER_1, branch.getId(),
                        SensitivityVariableType.INJECTION_ACTIVE_POWER, generator.getId(), false, ContingencyContext.all()));
            }
        }
        List<String> buses = network.getBusView().getBusStream().map(b -> b.getId()).sorted().limit(nBus).toList();
        for (String bus : buses) {
            for (Generator generator : generators.subList(0, Math.min(generators.size(), nGen / 2))) {
                factors.add(new SensitivityFactor(SensitivityFunctionType.BUS_VOLTAGE, bus,
                        SensitivityVariableType.BUS_TARGET_VOLTAGE, generator.getId(), false, ContingencyContext.all()));
            }
        }
        return factors;
    }

    /** Largest absolute difference of the values and of the function references, per function type; and the keys one
     *  side has and the other has not. */
    private static void compare(String label, Run reference, Run other, List<SensitivityFactor> factors) {
        Map<SensitivityFunctionType, double[]> worst = new EnumMap<>(SensitivityFunctionType.class);
        int missing = 0;
        int nanMismatch = 0;
        for (Map.Entry<List<Integer>, SensitivityValue> entry : reference.values().entrySet()) {
            SensitivityValue o = other.values().get(entry.getKey());
            if (o == null) {
                missing++;
                continue;
            }
            SensitivityValue r = entry.getValue();
            double[] w = worst.computeIfAbsent(factors.get(entry.getKey().get(0)).getFunctionType(), t -> new double[4]);
            if (Double.isNaN(r.getValue()) != Double.isNaN(o.getValue())
                    || Double.isNaN(r.getFunctionReference()) != Double.isNaN(o.getFunctionReference())) {
                nanMismatch++;
                continue;
            }
            if (!Double.isNaN(r.getValue())) {
                w[0] = Math.max(w[0], Math.abs(r.getValue() - o.getValue()));
                w[1] = Math.max(w[1], Math.abs(r.getValue() - o.getValue()) / Math.max(1, Math.abs(r.getValue())));
            }
            if (!Double.isNaN(r.getFunctionReference())) {
                w[2] = Math.max(w[2], Math.abs(r.getFunctionReference() - o.getFunctionReference()));
                w[3] = Math.max(w[3], Math.abs(r.getFunctionReference() - o.getFunctionReference())
                        / Math.max(1, Math.abs(r.getFunctionReference())));
            }
        }
        long extra = other.values().keySet().stream().filter(k -> !reference.values().containsKey(k)).count();
        long statusDiffs = 0;
        Map<String, SensitivityAnalysisResult.Status> referenceStatus = new HashMap<>();
        reference.statuses().forEach(s -> referenceStatus.put(s.getState().toString(), s.getStatus()));
        Map<SensitivityAnalysisResult.Status, Long> counts = new EnumMap<>(SensitivityAnalysisResult.Status.class);
        for (var s : other.statuses()) {
            counts.merge(s.getStatus(), 1L, Long::sum);
            if (referenceStatus.get(s.getState().toString()) != s.getStatus()) {
                statusDiffs++;
            }
        }
        System.out.printf("SENSI_BENCH diff %s: values %d (reference %d), missing %d, extra %d, NaN mismatches %d, "
                + "status differences %d, statuses %s%n", label, other.values().size(), reference.values().size(), missing, extra,
                nanMismatch, statusDiffs, counts);
        worst.forEach((type, w) -> System.out.printf("SENSI_BENCH diff %s %s: value abs %.3e rel %.3e, reference abs %.3e rel %.3e%n",
                label, type, w[0], w[1], w[2], w[3]));
    }

    @Test
    void sensitivityGpuVsCpu() throws Exception {
        assumeTrue(System.getenv("OLF_GPU_SENSI_BENCH") != null, "OLF_GPU_SENSI_BENCH not set");
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available");
        assumeTrue(System.getenv("OLF_GPU_CONT_FILE") != null, "OLF_GPU_CONT_FILE not set");
        Path caseFile = GpuTestPaths.caseDir().resolve(Optional.ofNullable(System.getenv("OLF_GPU_SENSI_CASE")).orElse("case9241pegase.xiidm.gz"));
        assumeTrue(Files.exists(caseFile), caseFile + " not found");
        Path parametersFile = Path.of("rte-loadflow-default-parameters.json");
        assumeTrue(Files.exists(parametersFile), "rte-loadflow-default-parameters.json not found");

        Network network = Network.read(caseFile);
        List<Contingency> contingencies = Files.readAllLines(Path.of(System.getenv("OLF_GPU_CONT_FILE"))).stream()
                .map(String::trim).filter(id -> !id.isEmpty() && network.getBranch(id) != null)
                .map(id -> new Contingency(id, new BranchContingency(id))).toList();
        contingencies = contingencies.subList(0, Math.min(contingencies.size(), env("OLF_GPU_SENSI_NCTG", contingencies.size())));
        List<SensitivityFactor> factors = factors(network);
        int cpuThreads = 4;
        System.out.printf("SENSI_BENCH case %s, %d contingencies, %d factors, %d buses%n", caseFile.getFileName(),
                contingencies.size(), factors.size(), network.getBusView().getBusStream().count());

        OpenSensitivityAnalysisProvider cpuProvider = new OpenSensitivityAnalysisProvider(new SparseMatrixFactory());
        GpuSensitivityAnalysisProvider gpuProvider = new GpuSensitivityAnalysisProvider(new SparseMatrixFactory());

        long batched = GpuBatchedAcSensitivityAnalysis.BATCHED_CONTINGENCIES.sum();
        long basis = GpuBatchedAcSensitivityAnalysis.BASIS_SOLVED_CONTINGENCIES.sum();
        long valued = GpuBatchedAcSensitivityAnalysis.DEVICE_VALUED_CONTINGENCIES.sum();
        Run gpuCold = run(gpuProvider, network, factors, contingencies, parameters(parametersFile, 1));
        System.out.printf("SENSI_BENCH gpu cold %.2f s: batched %d, basis solved %d, contracted on the device %d%n", gpuCold.seconds(),
                GpuBatchedAcSensitivityAnalysis.BATCHED_CONTINGENCIES.sum() - batched,
                GpuBatchedAcSensitivityAnalysis.BASIS_SOLVED_CONTINGENCIES.sum() - basis,
                GpuBatchedAcSensitivityAnalysis.DEVICE_VALUED_CONTINGENCIES.sum() - valued);
        int warmRuns = 2;
        Run gpuWarm = gpuCold;
        for (int i = 0; i < warmRuns; i++) {
            gpuWarm = run(gpuProvider, network, factors, contingencies, parameters(parametersFile, 1));
            System.out.printf("SENSI_BENCH gpu warm %.2f s%n", gpuWarm.seconds());
        }
        Run cpuMulti = run(cpuProvider, network, factors, contingencies, parameters(parametersFile, cpuThreads));
        System.out.printf("SENSI_BENCH cpu %d threads %.2f s%n", cpuThreads, cpuMulti.seconds());
        Run cpuMulti2 = run(cpuProvider, network, factors, contingencies, parameters(parametersFile, cpuThreads));
        System.out.printf("SENSI_BENCH cpu %d threads (second run) %.2f s%n", cpuThreads, cpuMulti2.seconds());
        Run cpuSingle = run(cpuProvider, network, factors, contingencies, parameters(parametersFile, 1));
        System.out.printf("SENSI_BENCH cpu 1 thread %.2f s%n", cpuSingle.seconds());
        compare("gpu vs cpu", cpuMulti, gpuWarm, factors);
        compare("cpu vs cpu", cpuMulti, cpuMulti2, factors);
    }
}
