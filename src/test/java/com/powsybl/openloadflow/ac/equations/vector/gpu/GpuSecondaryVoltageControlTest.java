/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.extensions.SecondaryVoltageControlAdder;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.solver.GpuNewtonRaphson;
import com.powsybl.openloadflow.ac.solver.GpuNewtonRaphsonFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * SECONDARY voltage control on the sequential GPU inner solver.
 *
 * <p>Unlike phase control and shunt voltage control, this loop owns NO solver variable and NO equation
 * type: it builds its own dense side matrices, solves a small standalone LU OUTSIDE the Newton system,
 * and writes back only {@code GeneratorVoltageControl.setTargetValue(..)} — i.e. it mutates the RHS of
 * already-existing BUS_TARGET_V rows and their active flags. So on the sequential path (CPU outer loops
 * around a GPU inner solve) it should work with NO packing support at all. This test exists to prove
 * that claim rather than assume it.
 *
 * <p>Setup mirrors OLF's own SecondaryVoltageControlTest#testNoReactiveLimits (ieee14, zone z1, pilot
 * point B10 at 13 kV, control units B6-G and B8-G), so the CPU arm is a known-good reference.
 *
 * <p>The {@link GpuNewtonRaphson#CPU_FALLBACK_COUNT} assertion is not optional decoration: a GPU test
 * that only compares against CPU CANNOT detect that the GPU never ran, because CPU-identical results are
 * exactly what the fallback contract guarantees.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSecondaryVoltageControlTest {

    static {
        GpuTestPaths.init();
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    private static Network zonedIeee14() {
        Network network = IeeeCdfNetworkFactory.create14();
        // same reactive-limit widening as the OLF reference test, so the controllers do not sit at limit
        network.getGenerator("B8-G").newMinMaxReactiveLimits().setMinQ(-6).setMaxQ(200).add();
        network.newExtension(SecondaryVoltageControlAdder.class)
                .newControlZone()
                    .withName("z1")
                    .newPilotPoint().withTargetV(13).withBusbarSectionsOrBusesIds(List.of("B10")).add()
                    .newControlUnit().withId("B6-G").add()
                    .newControlUnit().withId("B8-G").add()
                .add()
                .add();
        return network;
    }

    @Test
    void secondaryVoltageControlRunsOnGpuAndMatchesCpu() {
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));

        Network cpuNetwork = zonedIeee14();
        LoadFlowParameters cpuParameters = new LoadFlowParameters().setUseReactiveLimits(false);
        OpenLoadFlowParameters.create(cpuParameters)
                .setSecondaryVoltageControl(true)
                .setNewtonRaphsonConvEpsPerEq(1e-8);
        LoadFlowResult cpuResult = runner.run(cpuNetwork, cpuParameters);
        assertTrue(cpuResult.isFullyConverged(), "CPU reference must converge");

        Network gpuNetwork = zonedIeee14();
        LoadFlowParameters gpuParameters = new LoadFlowParameters().setUseReactiveLimits(false);
        OpenLoadFlowParameters.create(gpuParameters)
                .setSecondaryVoltageControl(true)
                .setNewtonRaphsonConvEpsPerEq(1e-8)
                .setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        long fallbacksBefore = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        LoadFlowResult gpuResult = runner.run(gpuNetwork, gpuParameters);
        assertTrue(gpuResult.isFullyConverged(), "GPU load flow must converge");
        assertEquals(0, GpuNewtonRaphson.CPU_FALLBACK_COUNT.get() - fallbacksBefore,
                "the GPU must actually SOLVE this — a silent fallback leaves every other assertion here "
                        + "passing on CPU-vs-CPU results");

        double worstV = 0;
        for (Bus cpuBus : cpuNetwork.getBusView().getBuses()) {
            Bus gpuBus = gpuNetwork.getBusView().getBus(cpuBus.getId());
            if (gpuBus != null && !Double.isNaN(cpuBus.getV())) {
                worstV = Math.max(worstV, Math.abs(cpuBus.getV() - gpuBus.getV()));
            }
        }
        // The pilot point is what secondary voltage control actually steers: if the loop were dropped,
        // B10 would sit where the plain run leaves it (12.611 kV in the OLF reference) instead of at
        // its 13 kV target, which is orders of magnitude outside this tolerance.
        double pilotV = gpuNetwork.getBusBreakerView().getBus("B10").getV();
        System.out.printf("SVC GPU vs CPU: max dV = %.2e kV, pilot B10 = %.4f kV (target 13)%n", worstV, pilotV);
        assertEquals(13, pilotV, 1e-3, "the pilot point must be driven to its target — a dropped "
                + "secondary voltage control would leave it near the uncontrolled 12.611 kV");
        assertTrue(worstV < 1e-6, "bus voltages must match CPU (max dV = " + worstV + " kV)");
    }
}
