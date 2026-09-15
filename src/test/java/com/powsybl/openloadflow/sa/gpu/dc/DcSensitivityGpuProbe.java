/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu.dc;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.ContingencyContext;
import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Generator;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.sensitivity.*;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * DC sensitivity analysis on rte6515, stock engine against the device one, with contingencies.
 *
 * <p>This is the shape the LODF measurement pointed at: the states never leave the device, and per
 * contingency only the factor VALUES come back. Knobs: {@code OLF_SENSI_GROUPS} injection variables
 * (default 50), {@code OLF_SENSI_FUNCTIONS} monitored branches (default 200),
 * {@code OLF_SENSI_CTG} contingencies (default 100).
 *
 * <p>Note on the CPU arm: OLF's DC sensitivity analysis ignores the executor it is given, so the
 * reference is single-threaded whatever the thread count — unlike the security analysis, there is
 * no 8-thread arm to compare against.
 */
class DcSensitivityGpuProbe {

    private static final Path CASE_FILE = Path.of("rte6515_full.xiidm.gz");
    private static final Path PARAMS_FILE = Path.of("rte-loadflow-default-parameters.json");

    @Test
    void stockVersusGpu() {
        assumeTrue(Files.exists(CASE_FILE), "rte6515_full.xiidm.gz not found");
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        int groups = Integer.parseInt(System.getenv().getOrDefault("OLF_SENSI_GROUPS", "50"));
        int functions = Integer.parseInt(System.getenv().getOrDefault("OLF_SENSI_FUNCTIONS", "200"));
        int contingencyCount = Integer.parseInt(System.getenv().getOrDefault("OLF_SENSI_CTG", "100"));

        Network network = Network.read(CASE_FILE);
        LoadFlowParameters lfParameters = JsonLoadFlowParameters.read(PARAMS_FILE).setDc(true);
        SensitivityAnalysisParameters parameters = new SensitivityAnalysisParameters();
        parameters.setLoadFlowParameters(lfParameters);

        List<String> variables = network.getGeneratorStream().limit(groups).map(Generator::getId).toList();
        List<String> monitored = network.getBranchStream().limit(functions).map(Branch::getId).toList();
        List<SensitivityFactor> factors = new ArrayList<>();
        for (String branchId : monitored) {
            for (String generatorId : variables) {
                factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_ACTIVE_POWER_1, branchId,
                        SensitivityVariableType.INJECTION_ACTIVE_POWER, generatorId, false, ContingencyContext.all()));
            }
        }
        List<Contingency> contingencies = network.getBranchStream().limit(contingencyCount)
                .map(b -> new Contingency(b.getId(), new BranchContingency(b.getId())))
                .toList();
        System.out.printf("DC-SENSI rte6515_full | %d variables x %d functions = %,d factors | %d contingencies | %,d values%n",
                variables.size(), monitored.size(), factors.size(), contingencies.size(),
                (long) factors.size() * (contingencies.size() + 1));

        long t0 = System.nanoTime();
        SensitivityAnalysisResult cpu = new SensitivityAnalysis.Runner(new com.powsybl.openloadflow.sensi.OpenSensitivityAnalysisProvider(new SparseMatrixFactory()))
                .run(network, factors, contingencies, parameters);
        long t1 = System.nanoTime();
        com.powsybl.openloadflow.sensi.GpuDcSensitivityAnalysis.LAST_DEVICE_FACTOR_COUNT.set(-1);
        SensitivityAnalysisResult gpu = new SensitivityAnalysis.Runner(new com.powsybl.openloadflow.sensi.GpuSensitivityAnalysisProvider(new SparseMatrixFactory()))
                .run(network, factors, contingencies, parameters);
        long t2 = System.nanoTime();

        double cpuMs = (t1 - t0) / 1e6;
        double gpuMs = (t2 - t1) / 1e6;
        System.out.printf("DC-SENSI CPU %8.0f ms (1 thread, OLF ignores the executor) | GPU %8.0f ms | x%.2f | %,d factor evaluations on the device%n",
                cpuMs, gpuMs, cpuMs / gpuMs, com.powsybl.openloadflow.sensi.GpuDcSensitivityAnalysis.LAST_DEVICE_FACTOR_COUNT.get());

        assertEquals(cpu.getValues().size(), gpu.getValues().size(), "same number of values");
        double maxSensi = 0;
        double maxFunction = 0;
        for (int i = 0; i < cpu.getValues().size(); i++) {
            SensitivityValue a = cpu.getValues().get(i);
            SensitivityValue b = gpu.getValues().get(i);
            if (!Double.isNaN(a.getValue()) && !Double.isNaN(b.getValue())) {
                maxSensi = Math.max(maxSensi, Math.abs(a.getValue() - b.getValue()));
            }
            if (!Double.isNaN(a.getFunctionReference()) && !Double.isNaN(b.getFunctionReference())) {
                maxFunction = Math.max(maxFunction, Math.abs(a.getFunctionReference() - b.getFunctionReference()));
            }
        }
        System.out.printf("DC-SENSI parity: max |dSensi| %.3e | max |dFunction| %.3e MW%n", maxSensi, maxFunction);
        assertTrue(maxSensi < 1e-9, "sensitivities differ by " + maxSensi);
        assertTrue(maxFunction < 1e-6, "function references differ by " + maxFunction);
    }
}
