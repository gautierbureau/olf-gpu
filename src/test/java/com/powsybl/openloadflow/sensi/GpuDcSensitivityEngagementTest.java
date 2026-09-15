/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.ContingencyContext;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Generator;
import com.powsybl.iidm.network.Network;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.sensitivity.*;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The device sensitivity path engaged, and its values against the stock engine's.
 *
 * <p>OLF's own suites (rerun in the sibling classes) check the values against expectations written
 * for the CPU engine, but a test that only compares cannot tell that the device never ran. This one
 * asserts the engagement counter as well — the AC lesson, where a hollow test passed for a session.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuDcSensitivityEngagementTest {

    private static List<SensitivityFactor> factors(Network network) {
        List<SensitivityFactor> factors = new ArrayList<>();
        for (Branch<?> branch : network.getBranches()) {
            for (Generator generator : network.getGenerators()) {
                factors.add(new SensitivityFactor(SensitivityFunctionType.BRANCH_ACTIVE_POWER_1, branch.getId(),
                        SensitivityVariableType.INJECTION_ACTIVE_POWER, generator.getId(), false, ContingencyContext.all()));
            }
        }
        return factors;
    }

    private static SensitivityAnalysisResult run(Network network, List<SensitivityFactor> factors,
                                                 List<Contingency> contingencies, boolean gpu) {
        SensitivityAnalysisParameters parameters = new SensitivityAnalysisParameters();
        parameters.getLoadFlowParameters().setDc(true).setDistributedSlack(true);
        var provider = gpu ? new GpuSensitivityAnalysisProvider(new SparseMatrixFactory())
                : new OpenSensitivityAnalysisProvider(new SparseMatrixFactory());
        return new SensitivityAnalysis.Runner(provider).run(network, factors, contingencies, parameters);
    }

    /** A value undefined on one arm must be undefined on the other; otherwise the gap. */
    private static double difference(double cpu, double gpu, String what) {
        if (Double.isNaN(cpu) || Double.isNaN(gpu)) {
            assertEquals(Double.isNaN(cpu), Double.isNaN(gpu), what + ": " + cpu + " vs " + gpu);
            return 0;
        }
        return Math.abs(cpu - gpu);
    }

    @Test
    void valuesMatchStockAndTheDeviceRan() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network network = IeeeCdfNetworkFactory.create118();
        List<SensitivityFactor> factors = factors(network);
        List<Contingency> contingencies = network.getBranchStream().limit(20)
                .map(b -> new Contingency(b.getId(), new BranchContingency(b.getId())))
                .toList();

        SensitivityAnalysisResult cpu = run(network, factors, contingencies, false);
        GpuDcSensitivityAnalysis.LAST_DEVICE_FACTOR_COUNT.set(-1);
        SensitivityAnalysisResult gpu = run(network, factors, contingencies, true);

        int evaluated = GpuDcSensitivityAnalysis.LAST_DEVICE_FACTOR_COUNT.get();
        assertTrue(evaluated > 0, "the device evaluated the factors: " + evaluated);
        assertEquals(cpu.getValues().size(), gpu.getValues().size(), "same number of sensitivity values");
        assertTrue(cpu.getValues().size() > factors.size(), "the contingencies produced values too");

        double maxSensi = 0;
        double maxFunction = 0;
        for (int i = 0; i < cpu.getValues().size(); i++) {
            SensitivityValue a = cpu.getValues().get(i);
            SensitivityValue b = gpu.getValues().get(i);
            assertEquals(a.getFactorIndex(), b.getFactorIndex(), "same factor order");
            assertEquals(a.getContingencyIndex(), b.getContingencyIndex(), "same contingency order");
            maxSensi = Math.max(maxSensi, difference(a.getValue(), b.getValue(), "sensitivity of factor " + a.getFactorIndex()));
            maxFunction = Math.max(maxFunction, difference(a.getFunctionReference(), b.getFunctionReference(),
                    "function reference of factor " + a.getFactorIndex()));
        }
        // two direct solvers on one matrix: the project's 1e-10 bound, here on values of order 1 (the
        // sensitivities) and of order 100 MW (the function references)
        assertTrue(maxSensi < 1e-10, "max sensitivity difference " + maxSensi);
        assertTrue(maxFunction < 1e-8, "max function reference difference " + maxFunction);
    }
}
