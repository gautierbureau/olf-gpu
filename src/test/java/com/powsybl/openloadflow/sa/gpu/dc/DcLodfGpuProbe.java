/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu.dc;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.DcLoadFlowParameters;
import com.powsybl.openloadflow.dc.fastdc.LodfCalculator;
import com.powsybl.openloadflow.dc.fastdc.LodfResultWriter;
import com.powsybl.openloadflow.dc.gpu.GpuLodfCalculator;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The LODF factors of rte6515, CPU against device, on the same {@code DcLoadFlowContext}.
 *
 * <p>Both arms stream into the same cheap accumulating writer, so what is compared is the
 * computation and not a per-factor lambda into a 650 MB matrix. The CPU arm runs with
 * {@code OLF_DC_THREADS} threads (default 8, the project's comparison rule); the device arm is
 * single-threaded by construction. {@code OLF_DC_MONITORED} caps the monitored branches (0 = all,
 * the full N-1 LODF matrix).
 */
class DcLodfGpuProbe {

    private static final Path CASE_FILE = Path.of("rte6515_full.xiidm.gz");
    private static final Path PARAMS_FILE = Path.of("rte-loadflow-default-parameters.json");

    /** Cheap, order-independent digest. THREAD-SAFE: the CPU calculator calls the writer from its
     *  whole pool, and plain fields silently lose writes (this probe measured 661k of 1.8M factors
     *  before the adders went in). */
    private static final class Digest implements LodfResultWriter {
        private final java.util.concurrent.atomic.LongAdder count = new java.util.concurrent.atomic.LongAdder();
        private final java.util.concurrent.atomic.LongAdder nan = new java.util.concurrent.atomic.LongAdder();
        private final java.util.concurrent.atomic.DoubleAdder sum = new java.util.concurrent.atomic.DoubleAdder();
        private final java.util.concurrent.atomic.DoubleAccumulator max =
                new java.util.concurrent.atomic.DoubleAccumulator(Math::max, 0.0);

        @Override
        public void writeLodf(int monitoredBranchIndex, int outagedBranchIndex, double lodf) {
            if (Double.isNaN(lodf)) {
                nan.increment();
                return;
            }
            count.increment();
            sum.add(lodf);
            max.accumulate(Math.abs(lodf));
        }
    }

    @Test
    void stockVersusGpu() {
        assumeTrue(Files.exists(CASE_FILE), "rte6515_full.xiidm.gz not found");
        assumeTrue(GpuLodfCalculator.isAvailable(), "GPU native library not available");
        Network network = Network.read(CASE_FILE);
        LoadFlowParameters parameters = JsonLoadFlowParameters.read(PARAMS_FILE).setDc(true);
        DcLoadFlowParameters dcParameters = OpenLoadFlowParameters.createDcParameters(network, parameters,
                OpenLoadFlowParameters.get(parameters), new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false);
        LfNetwork lfNetwork = LfNetwork.load(network, new LfNetworkLoaderImpl(), dcParameters.getNetworkParameters(), ReportNode.NO_OP).getFirst();
        int threads = Integer.parseInt(System.getenv().getOrDefault("OLF_DC_THREADS", "8"));
        int monitoredCap = Integer.parseInt(System.getenv().getOrDefault("OLF_DC_MONITORED", "0"));
        try (DcLoadFlowContext context = new DcLoadFlowContext(lfNetwork, dcParameters)) {
            List<LfBranch> outaged = GpuLodfCalculator.flowingBranches(lfNetwork);
            List<LfBranch> monitored = monitoredCap > 0 ? outaged.subList(0, Math.min(monitoredCap, outaged.size())) : lfNetwork.getBranches();
            System.out.printf("DC-LODF rte6515_full | %d monitored x %d outaged = %,d factors | CPU threads %d, GPU 1%n",
                    monitored.size(), outaged.size(), (long) monitored.size() * outaged.size(), threads);

            // OLF_DC_NULL_WRITER=1 drops the per-factor consumption on BOTH arms, to see what the
            // computation costs without the 82-million-call streaming that dominates it
            boolean nullWriter = "1".equals(System.getenv("OLF_DC_NULL_WRITER"));
            Digest cpu = new Digest();
            LodfResultWriter cpuWriter = nullWriter ? (r, c, v) -> { } : cpu;
            long t0 = System.nanoTime();
            LodfCalculator.computeLodf(context, monitored, outaged, cpuWriter, threads);
            long t1 = System.nanoTime();
            Digest gpu = new Digest();
            long t2 = System.nanoTime();
            try (com.powsybl.openloadflow.dc.gpu.GpuDcLoadFlowEngine engine =
                         new com.powsybl.openloadflow.dc.gpu.GpuDcLoadFlowEngine(context)) {
                engine.prepare();
                long t3 = System.nanoTime();
                GpuLodfCalculator.computeLodf(engine, monitored, outaged, nullWriter ? (r, c, v) -> { } : gpu);
                long t4 = System.nanoTime();
                double cpuMs = (t1 - t0) / 1e6;
                double setupMs = (t3 - t2) / 1e6;
                double factorMs = (t4 - t3) / 1e6;
                System.out.printf("DC-LODF CPU %8.0f ms (%d threads) | GPU setup %6.0f ms + factors %8.0f ms = %8.0f ms | x%.2f on the factors, x%.2f all in%n",
                        cpuMs, threads, setupMs, factorMs, setupMs + factorMs, cpuMs / factorMs, cpuMs / (setupMs + factorMs));
            }
            System.out.printf("DC-LODF parity: finite %,d vs %,d | NaN %,d vs %,d | sum %.9e vs %.9e | max %.9f vs %.9f%n",
                    cpu.count.sum(), gpu.count.sum(), cpu.nan.sum(), gpu.nan.sum(), cpu.sum.sum(), gpu.sum.sum(),
                    cpu.max.get(), gpu.max.get());
            if (nullWriter) {
                return;
            }
            assertEquals(cpu.count.sum(), gpu.count.sum(), "same number of finite factors");
            assertEquals(cpu.nan.sum(), gpu.nan.sum(), "same number of undefined factors");
            assertTrue(Math.abs(cpu.sum.sum() - gpu.sum.sum()) < 1e-6 * Math.max(1, Math.abs(cpu.sum.sum())), "same factor sum");
            assertEquals(cpu.max.get(), gpu.max.get(), 1e-10, "same largest factor");
        }
    }
}
