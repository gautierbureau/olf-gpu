/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu.dc;

import com.powsybl.iidm.network.Generator;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.loadflow.resultswriter.NetworkResultWriter;
import com.powsybl.loadflow.resultswriter.NetworkResultWriterFactory;
import com.powsybl.openloadflow.ts.TimeSeriesLoadFlow;
import com.powsybl.openloadflow.ts.TimeSeriesLoadFlowParameters;
import com.powsybl.openloadflow.ts.TimeSeriesLoadFlowResult;
import com.powsybl.timeseries.DoubleTimeSeries;
import com.powsybl.timeseries.RegularTimeSeriesIndex;
import com.powsybl.timeseries.TimeSeries;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Phase 3.0: where the CPU time-series DC load flow spends its per-step time on rte6515.
 *
 * <p>The design question this answers, before any device engine is written: a step is
 * {@code restore -> applySetpoints -> DcLoadFlowEngine.run -> emit}, and the device can only take
 * the solve. The variants below switch the emitted datasets off and on against a no-op writer, so
 * the per-step cost splits into the part a device engine could remove and the part it could not.
 *
 * <p>Knobs: {@code OLF_TS_STEPS} (default 50), {@code OLF_TS_SERIES} planned generators (default 200),
 * {@code OLF_TS_THREADS} (default 1).
 */
class DcTimeSeriesBaselineProbe {

    /** Hands one {@link CountingWriter} to each partition and adds their counts up at the end. */
    private static final class RowCounter implements NetworkResultWriterFactory {

        private final List<CountingWriter> writers = java.util.Collections.synchronizedList(new java.util.ArrayList<>());

        @Override
        public NetworkResultWriter create(int partitionIndex) {
            CountingWriter writer = new CountingWriter();
            writers.add(writer);
            return writer;
        }

        long rows() {
            long total = 0;
            synchronized (writers) {
                for (CountingWriter writer : writers) {
                    total += writer.rows;
                }
            }
            return total;
        }
    }

    private static final Path CASE_FILE = Path.of("rte6515_full.xiidm.gz");
    private static final Path PARAMS_FILE = Path.of("rte-loadflow-default-parameters.json");

    /**
     * Counts rows without storing them: the emit walk without a writer's own cost.
     *
     * <p>One instance per PARTITION, handed out by {@link RowCounter}, so the count is a plain field
     * add on a thread-confined object. A single shared writer with a {@code LongAdder} was measured
     * at 11% of all execution samples, which inflated the emission figure on both arms.
     */
    private static final class CountingWriter implements NetworkResultWriter {
        private long rows;

        @Override
        public void writeBranchResult(String stateId, String subStateId, String status, String branchId,
                                      double p1, double q1, double i1, double p2, double q2, double i2, double flowTransfer) {
            rows++;
        }

        @Override
        public void writeBusResult(String stateId, String subStateId, String status, String busId, double v, double angle) {
            rows++;
        }

        @Override
        public void writeGeneratorResult(String stateId, String subStateId, String status, String generatorId,
                                         double targetP, double p) {
            rows++;
        }

        @Override
        public void close() {
            // nothing to close
        }
    }

    @Test
    void perStepSplit() {
        assumeTrue(Files.exists(CASE_FILE), "rte6515_full.xiidm.gz not found");
        int steps = Integer.parseInt(System.getenv().getOrDefault("OLF_TS_STEPS", "50"));
        int seriesCount = Integer.parseInt(System.getenv().getOrDefault("OLF_TS_SERIES", "200"));
        int threads = Integer.parseInt(System.getenv().getOrDefault("OLF_TS_THREADS", "1"));

        Network network = Network.read(CASE_FILE);
        // rte6515 carries AC-emulation HVDC links. OLF_TS_HVDC=1 leaves them on, so the AC-emulation
        // limits outer loop runs — driven over the chunk on the device arm.
        boolean hvdcAcEmulation = "1".equals(System.getenv("OLF_TS_HVDC"));
        LoadFlowParameters lfParameters = JsonLoadFlowParameters.read(PARAMS_FILE).setDc(true).setHvdcAcEmulation(hvdcAcEmulation);
        RegularTimeSeriesIndex index = RegularTimeSeriesIndex.create(Instant.parse("2025-01-01T00:00:00Z"),
                Instant.parse("2025-01-01T00:00:00Z").plus(Duration.ofHours(steps - 1L)), Duration.ofHours(1));

        List<Generator> planned = network.getGeneratorStream()
                .filter(g -> !Double.isNaN(g.getTargetP()) && g.getTargetP() > 0)
                .limit(seriesCount)
                .toList();
        List<DoubleTimeSeries> plan = new ArrayList<>();
        for (Generator generator : planned) {
            double[] values = new double[steps];
            for (int s = 0; s < steps; s++) {
                values[s] = generator.getTargetP() * (1 + 0.1 * Math.sin(2 * Math.PI * s / Math.max(steps, 2)));
            }
            plan.add(TimeSeries.createDouble(generator.getId(), index, values));
        }
        System.out.printf("DC-TS rte6515_full | %d steps x %d planned generators | %d thread(s) | hvdc ac emulation %s%n",
                steps, plan.size(), threads, hvdcAcEmulation ? "ON (outer loop runs)" : "off");

        // warm up the JIT and the network caches with a couple of steps
        run(network, plan.subList(0, Math.min(plan.size(), 5)), lfParameters, threads, false, false, false, null);

        // a no-op engine: restore + set points only, no solve at all. The gap to the full run is
        // what a device engine can actually take.
        double restoreOnly = run(network, plan, lfParameters, threads, false, false, false, null, NO_SOLVE);
        double solveOnly = run(network, plan, lfParameters, threads, false, false, false, null);
        double withBranches = run(network, plan, lfParameters, threads, true, false, false, null);
        RowCounter counter = new RowCounter();
        double withAll = run(network, plan, lfParameters, threads, true, true, true, counter);

        System.out.printf("DC-TS restore+setpoints (no solve) %8.1f ms total | %7.2f ms/step%n", restoreOnly, restoreOnly / steps);
        System.out.printf("DC-TS restore+setpoints+solve      %8.1f ms total | %7.2f ms/step | solve alone %6.2f ms/step%n",
                solveOnly, solveOnly / steps, (solveOnly - restoreOnly) / steps);
        System.out.printf("DC-TS + branch rows                %8.1f ms total | %7.2f ms/step | branch emit %6.2f ms/step%n",
                withBranches, withBranches / steps, (withBranches - solveOnly) / steps);
        System.out.printf("DC-TS + bus and generator rows     %8.1f ms total | %7.2f ms/step | all emit    %6.2f ms/step | %,d rows%n",
                withAll, withAll / steps, (withAll - solveOnly) / steps, counter.rows());
        System.out.printf("DC-TS share removable by a device solve: %.0f%%%n", 100 * solveOnly / withAll);

        // the device engine: same streams, same writer
        if (com.powsybl.openloadflow.dc.gpu.GpuDcSolver.isAvailable()) {
            var provider = com.powsybl.openloadflow.dc.gpu.GpuTimeSeriesDcEngine.planProvider();
            run(network, plan.subList(0, Math.min(plan.size(), 5)), lfParameters, threads, false, false, false, null, provider);
            com.powsybl.openloadflow.dc.gpu.GpuTimeSeriesDcEngine.LAST_DEVICE_STEP_COUNT.set(-1);
            com.powsybl.openloadflow.dc.gpu.GpuTimeSeriesDcEngine.LAST_DEVICE_LOOP_REGROUPINGS.set(0);
            double gpuSolveOnly = run(network, plan, lfParameters, threads, false, false, false, null, provider);
            RowCounter gpuCounter = new RowCounter();
            double gpuAll = run(network, plan, lfParameters, threads, true, true, true, gpuCounter, provider);
            System.out.printf("DC-TS GPU restore+setpoints+solve   %8.1f ms total | %7.2f ms/step | x%.2f%n",
                    gpuSolveOnly, gpuSolveOnly / steps, solveOnly / gpuSolveOnly);
            System.out.printf("DC-TS GPU with all rows            %8.1f ms total | %7.2f ms/step | x%.2f | %,d rows | %d steps on the device%n",
                    gpuAll, gpuAll / steps, withAll / gpuAll, gpuCounter.rows(),
                    com.powsybl.openloadflow.dc.gpu.GpuTimeSeriesDcEngine.LAST_DEVICE_STEP_COUNT.get());
            System.out.printf("DC-TS outer loop: %d AC-emulation configuration(s) applied and re-solved over the run%n",
                    com.powsybl.openloadflow.dc.gpu.GpuTimeSeriesDcEngine.LAST_DEVICE_LOOP_REGROUPINGS.get());
        }
    }

    /** An engine plan that applies the set points and skips the solve entirely. */
    private static final TimeSeriesLoadFlow.EnginePlanProvider NO_SOLVE = (network, parameters, matrixFactory, connectivityFactory) -> {
        TimeSeriesLoadFlow.EnginePlan delegate = TimeSeriesLoadFlow.EnginePlanProvider.DEFAULT
                .create(network, parameters, matrixFactory, connectivityFactory);
        return new TimeSeriesLoadFlow.EnginePlan() {
            @Override
            public com.powsybl.openloadflow.network.LfNetworkParameters networkParameters() {
                return delegate.networkParameters();
            }

            @Override
            public com.powsybl.openloadflow.network.LoadFlowModel loadFlowModel() {
                return delegate.loadFlowModel();
            }

            @Override
            public TimeSeriesLoadFlow.StepEngine createEngine(com.powsybl.openloadflow.network.LfNetwork lfNetwork) {
                return new TimeSeriesLoadFlow.StepEngine() {
                    @Override
                    public TimeSeriesLoadFlow.SolveResult solve() {
                        return new TimeSeriesLoadFlow.SolveResult(
                                com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.CONVERGED, 0, 0);
                    }

                    @Override
                    public void close() {
                        // nothing
                    }
                };
            }
        };
    };

    private static double run(Network network, List<DoubleTimeSeries> plan, LoadFlowParameters lfParameters, int threads,
                              boolean branches, boolean buses, boolean generators, RowCounter writer) {
        return run(network, plan, lfParameters, threads, branches, buses, generators, writer,
                TimeSeriesLoadFlow.EnginePlanProvider.DEFAULT);
    }

    private static double run(Network network, List<DoubleTimeSeries> plan, LoadFlowParameters lfParameters, int threads,
                              boolean branches, boolean buses, boolean generators, RowCounter writer,
                              TimeSeriesLoadFlow.EnginePlanProvider provider) {
        TimeSeriesLoadFlowParameters parameters = new TimeSeriesLoadFlowParameters()
                .setLoadFlowParameters(lfParameters)
                .setThreadCount(threads)
                .setStreamBranchResults(branches)
                .setStreamBusResults(buses)
                .setStreamGeneratorResults(generators);
        NetworkResultWriterFactory factory = writer != null ? writer : NetworkResultWriterFactory.NO_OP;
        long t0 = System.nanoTime();
        TimeSeriesLoadFlowResult result = TimeSeriesLoadFlow.run(network, plan, parameters, factory, provider);
        long t1 = System.nanoTime();
        assertEquals(plan.get(0).getMetadata().getIndex().getPointCount(), result.getStepResults().size());
        return (t1 - t0) / 1e6;
    }
}
