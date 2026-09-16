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
import com.powsybl.openloadflow.dc.gpu.GpuDcLoadFlowEngine;
import com.powsybl.openloadflow.dc.gpu.GpuPtdfMatrix;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * What a full PTDF costs on the device, and what it costs to get it back: the assemble is the point
 * of {@link GpuPtdfMatrix}, the read-back is the part a caller should avoid.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class DcPtdfGpuProbe {

    private static final Path CASE_FILE = Path.of("rte6515_full.xiidm.gz");
    private static final Path PARAMS_FILE = Path.of("rte-loadflow-default-parameters.json");

    @Test
    void assembleFullPtdf() {
        assumeTrue(Files.exists(CASE_FILE), "rte6515_full.xiidm.gz not found");
        assumeTrue(Files.exists(PARAMS_FILE), "params json not found");
        assumeTrue(GpuPtdfMatrix.isAvailable(), "GPU native library not available");
        Network network = Network.read(CASE_FILE);
        LoadFlowParameters parameters = JsonLoadFlowParameters.read(PARAMS_FILE).setDc(true);
        DcLoadFlowParameters dcParameters = OpenLoadFlowParameters.createDcParameters(network, parameters,
                OpenLoadFlowParameters.get(parameters), new SparseMatrixFactory(),
                new EvenShiloachGraphDecrementalConnectivityFactory<>(), false);
        LfNetwork lfNetwork = LfNetwork.load(network, new LfNetworkLoaderImpl(), dcParameters.getNetworkParameters(),
                ReportNode.NO_OP).getFirst();
        int busCap = Integer.parseInt(System.getenv().getOrDefault("OLF_DC_PTDF_BUSES", "0"));
        try (DcLoadFlowContext context = new DcLoadFlowContext(lfNetwork, dcParameters)) {
            List<LfBranch> branches = GpuPtdfMatrix.flowingBranches(lfNetwork);
            List<LfBus> buses = GpuPtdfMatrix.injectionBuses(lfNetwork);
            if (busCap > 0 && busCap < buses.size()) {
                buses = buses.subList(0, busCap);
            }
            System.out.printf("DC-PTDF rte6515_full | %d branches x %d buses = %,d factors%n",
                    branches.size(), buses.size(), (long) branches.size() * buses.size());
            try (GpuDcLoadFlowEngine engine = new GpuDcLoadFlowEngine(context)) {
                long t0 = System.nanoTime();
                engine.prepare();
                long t1 = System.nanoTime();
                try (GpuPtdfMatrix ptdf = GpuPtdfMatrix.assemble(engine, branches, buses)) {
                    long t2 = System.nanoTime();
                    System.out.printf("DC-PTDF prepare %,.0f ms | assemble %,.0f ms | resident %,.0f MB%n",
                            (t1 - t0) / 1e6, (t2 - t1) / 1e6, ptdf.getDeviceBytes() / (1024.0 * 1024.0));
                    // the read-back, which is the part the verdict says not to pay: one column block at a time
                    int block = Math.max(1, 64 * 1024 * 1024 / (ptdf.getRowCount() * Double.BYTES));
                    double[] host = new double[ptdf.getRowCount() * block];
                    long t3 = System.nanoTime();
                    double maxAbs = 0;
                    long nonZero = 0;
                    long nan = 0;
                    for (int col0 = 0; col0 < ptdf.getColumnCount(); col0 += block) {
                        int cols = Math.min(block, ptdf.getColumnCount() - col0);
                        ptdf.readBlock(col0, cols, host);
                        int used = ptdf.getRowCount() * cols;
                        for (int i = 0; i < used; i++) {
                            double v = host[i];
                            if (Double.isNaN(v)) {
                                nan++;
                            } else {
                                maxAbs = Math.max(maxAbs, Math.abs(v));
                                if (v != 0) {
                                    nonZero++;
                                }
                            }
                        }
                    }
                    long t4 = System.nanoTime();
                    System.out.printf("DC-PTDF read back %,.0f ms (%,.0f MB over PCIe, the scan included)%n",
                            (t4 - t3) / 1e6, ptdf.getDeviceBytes() / (1024.0 * 1024.0));
                    System.out.printf("DC-PTDF values: max |ptdf| %.6f | non-zero %,d of %,d | NaN %,d%n",
                            maxAbs, nonZero, (long) ptdf.getRowCount() * ptdf.getColumnCount(), nan);
                }
            }
        }
    }
}
