/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ts;

import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Generator;
import com.powsybl.iidm.network.Load;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.resultswriter.NetworkResultWriter;
import com.powsybl.loadflow.resultswriter.NetworkResultWriterFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.dc.gpu.GpuTimeSeriesDcEngine;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.timeseries.DoubleTimeSeries;
import com.powsybl.timeseries.RegularTimeSeriesIndex;
import com.powsybl.timeseries.TimeSeries;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The device time-series DC engine against OLF's own, row for row.
 *
 * <p>Stronger than rerunning OLF's assertions: both engines stream into a recording writer and
 * every emitted row is compared — branch flows, bus angles, generator dispatch, in order — plus the
 * per-step statuses, distributed power and slack mismatch. A device engine that silently solved
 * nothing would differ on the first row; the engagement counter is asserted too.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuTimeSeriesDcEngineTest {

    private static final double TOL_MW = 1e-8;

    /** One streamed row: its identity (compared exactly) and its values (compared numerically). */
    private record Row(String key, double[] values) {
    }

    /** Keeps every streamed row, in order. */
    private static final class Recorder implements NetworkResultWriter {
        private final List<Row> rows = new ArrayList<>();

        @Override
        public void writeBranchResult(String stateId, String subStateId, String status, String branchId,
                                      double p1, double q1, double i1, double p2, double q2, double i2, double flowTransfer) {
            rows.add(new Row(String.join("|", "B", stateId, status, branchId), new double[] {p1, q1, i1, p2, q2, i2}));
        }

        @Override
        public void writeBusResult(String stateId, String subStateId, String status, String busId, double v, double angle) {
            rows.add(new Row(String.join("|", "U", stateId, status, busId), new double[] {v, angle}));
        }

        @Override
        public void writeGeneratorResult(String stateId, String subStateId, String status, String generatorId,
                                         double targetP, double p) {
            rows.add(new Row(String.join("|", "G", stateId, status, generatorId), new double[] {targetP, p}));
        }

        @Override
        public void close() {
            // nothing
        }
    }

    record Case(String name, Supplier<Network> network, Supplier<List<DoubleTimeSeries>> plan, Supplier<LoadFlowParameters> parameters) {
    }

    private static RegularTimeSeriesIndex index(int steps) {
        return RegularTimeSeriesIndex.create(Instant.parse("2025-01-01T00:00:00Z"),
                Instant.parse("2025-01-01T00:00:00Z").plus(Duration.ofHours(steps - 1L)), Duration.ofHours(1));
    }

    private static LoadFlowParameters dc() {
        LoadFlowParameters p = new LoadFlowParameters().setDc(true);
        OpenLoadFlowParameters.create(p);
        return p;
    }

    /** A plan over the given generators, wandering around their base target. */
    private static List<DoubleTimeSeries> generatorPlan(Network network, int steps, int count) {
        RegularTimeSeriesIndex index = index(steps);
        List<DoubleTimeSeries> plan = new ArrayList<>();
        List<Generator> generators = network.getGeneratorStream().filter(g -> g.getTargetP() > 0).limit(count).toList();
        for (int g = 0; g < generators.size(); g++) {
            double[] values = new double[steps];
            for (int s = 0; s < steps; s++) {
                values[s] = generators.get(g).getTargetP() * (1 + 0.2 * Math.sin(2 * Math.PI * (s + g) / steps));
            }
            plan.add(TimeSeries.createDouble(generators.get(g).getId(), index, values));
        }
        return plan;
    }

    private static List<DoubleTimeSeries> loadPlan(Network network, int steps, int count) {
        RegularTimeSeriesIndex index = index(steps);
        List<DoubleTimeSeries> plan = new ArrayList<>();
        List<Load> loads = network.getLoadStream().filter(l -> l.getP0() > 0).limit(count).toList();
        for (int l = 0; l < loads.size(); l++) {
            double[] values = new double[steps];
            for (int s = 0; s < steps; s++) {
                values[s] = loads.get(l).getP0() * (1 + 0.3 * Math.cos(2 * Math.PI * (s + l) / steps));
            }
            plan.add(TimeSeries.createDouble(loads.get(l).getId(), index, values));
        }
        return plan;
    }

    static List<Case> cases() {
        List<Case> cases = new ArrayList<>();
        cases.add(new Case("eurostag generators", () -> EurostagFactory.fix(EurostagTutorialExample1Factory.create()),
                () -> generatorPlan(EurostagFactory.fix(EurostagTutorialExample1Factory.create()), 7, 1), GpuTimeSeriesDcEngineTest::dc));
        cases.add(new Case("ieee14 generators, distributed slack", IeeeCdfNetworkFactory::create14,
                () -> generatorPlan(IeeeCdfNetworkFactory.create14(), 11, 3),
                () -> dc().setDistributedSlack(true).setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX)));
        cases.add(new Case("ieee14 loads", IeeeCdfNetworkFactory::create14,
                () -> loadPlan(IeeeCdfNetworkFactory.create14(), 9, 4), GpuTimeSeriesDcEngineTest::dc));
        cases.add(new Case("ieee14 loads, distributed slack on P", IeeeCdfNetworkFactory::create14,
                () -> loadPlan(IeeeCdfNetworkFactory.create14(), 9, 4),
                () -> dc().setDistributedSlack(true).setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P)));
        cases.add(new Case("ieee14 three slack buses", IeeeCdfNetworkFactory::create14,
                () -> generatorPlan(IeeeCdfNetworkFactory.create14(), 5, 3), () -> {
                    LoadFlowParameters p = dc().setDistributedSlack(true);
                    OpenLoadFlowParameters.get(p).setMaxSlackBusCount(3);
                    return p;
                }));
        cases.add(new Case("ieee118 generators, more steps than a chunk", IeeeCdfNetworkFactory::create118,
                () -> generatorPlan(IeeeCdfNetworkFactory.create118(), 300, 5),
                () -> dc().setDistributedSlack(true).setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX)));
        // the AC-emulation limits loop, driven over the chunk: the plan walks the link's offset
        // across its limit so that steps of one chunk end in different saturation states
        cases.add(new Case("hvdc ac emulation saturating across steps", () -> {
            Network n = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcLinkedByTwoLinesAndSwitch(
                    com.powsybl.iidm.network.HvdcConverterStation.HvdcType.VSC);
            n.getHvdcLine("hvdc23").setMaxP(170);
            return n;
        }, () -> {
            Network n = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcLinkedByTwoLinesAndSwitch(
                    com.powsybl.iidm.network.HvdcConverterStation.HvdcType.VSC);
            return generatorPlan(n, 12, 2);
        }, () -> dc().setHvdcAcEmulation(true)));
        cases.add(new Case("hvdc ac emulation, planned offset", () -> {
            Network n = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcInAcEmulationAndPSTInSymetricNetwork();
            n.getHvdcLine("hvdc12").setMaxP(20);
            return n;
        }, () -> {
            int steps = 10;
            RegularTimeSeriesIndex index = index(steps);
            double[] values = new double[steps];
            for (int s = 0; s < steps; s++) {
                values[s] = -40 + 80.0 * s / (steps - 1);       // sweeps both saturation sides
            }
            return List.of(TimeSeries.createDouble("hvdc12", index, values));
        }, () -> dc().setHvdcAcEmulation(true)));
        cases.add(new Case("ieee300 generators", IeeeCdfNetworkFactory::create300,
                () -> generatorPlan(IeeeCdfNetworkFactory.create300(), 13, 8),
                () -> dc().setDistributedSlack(true).setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX)));
        return cases;
    }

    @TestFactory
    Stream<DynamicTest> matchesStockEngine() {
        return cases().stream().map(c -> DynamicTest.dynamicTest(c.name(), () -> check(c)));
    }

    private static void check(Case c) {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        TimeSeriesLoadFlowParameters parameters = new TimeSeriesLoadFlowParameters()
                .setLoadFlowParameters(c.parameters().get())
                .setStreamBranchResults(true)
                .setStreamBusResults(true)
                .setStreamGeneratorResults(true);

        Recorder cpuRows = new Recorder();
        TimeSeriesLoadFlowResult cpu = TimeSeriesLoadFlow.run(c.network().get(), c.plan().get(), parameters,
                partition -> cpuRows, TimeSeriesLoadFlow.EnginePlanProvider.DEFAULT);

        GpuTimeSeriesDcEngine.LAST_DEVICE_STEP_COUNT.set(-1);
        Recorder gpuRows = new Recorder();
        TimeSeriesLoadFlowResult gpu = TimeSeriesLoadFlow.run(c.network().get(), c.plan().get(), parameters,
                partition -> gpuRows, GpuTimeSeriesDcEngine.planProvider());

        int onDevice = GpuTimeSeriesDcEngine.LAST_DEVICE_STEP_COUNT.get();
        assertTrue(onDevice >= cpu.getStepResults().size(), c.name() + ": steps solved on the device: " + onDevice);

        assertEquals(cpu.getStepResults().size(), gpu.getStepResults().size(), c.name() + ": step count");
        for (int s = 0; s < cpu.getStepResults().size(); s++) {
            StepResult a = cpu.getStepResults().get(s);
            StepResult b = gpu.getStepResults().get(s);
            assertEquals(a.stepIndex(), b.stepIndex(), c.name() + ": step index");
            assertEquals(a.status(), b.status(), c.name() + ": status of step " + s);
            assertEquals(a.distributedActivePower(), b.distributedActivePower(), TOL_MW,
                    c.name() + ": distributed power of step " + s);
            assertEquals(a.slackBusActivePowerMismatch(), b.slackBusActivePowerMismatch(), TOL_MW,
                    c.name() + ": slack mismatch of step " + s);
        }
        assertEquals(cpuRows.rows.size(), gpuRows.rows.size(), c.name() + ": streamed row count");
        assertFalse(cpuRows.rows.isEmpty(), c.name() + ": nothing was streamed");
        double maxDiff = 0;
        for (int i = 0; i < cpuRows.rows.size(); i++) {
            Row a = cpuRows.rows.get(i);
            Row b = gpuRows.rows.get(i);
            // the identity of the row - dataset, step, status, element - must match exactly; the
            // values within the bound two direct solvers on one matrix leave
            assertEquals(a.key(), b.key(), c.name() + ": row " + i);
            for (int v = 0; v < a.values().length; v++) {
                double x = a.values()[v];
                double y = b.values()[v];
                if (Double.isNaN(x) || Double.isNaN(y)) {
                    assertEquals(Double.isNaN(x), Double.isNaN(y), c.name() + ": row " + i + " value " + v + ": " + x + " vs " + y);
                } else {
                    maxDiff = Math.max(maxDiff, Math.abs(x - y));
                }
            }
        }
        assertTrue(maxDiff < TOL_MW, c.name() + ": max streamed value difference " + maxDiff);
    }
}
