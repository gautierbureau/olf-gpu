/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.gpu;

import com.google.auto.service.AutoService;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowProvider;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.DcLoadFlowEngine;
import com.powsybl.openloadflow.dc.DcLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcLoadFlowResult;
import com.powsybl.openloadflow.dc.gpu.GpuDcLoadFlowEngine;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * OLF's load flow provider with the DC load flow assembled, distributed and solved on the device
 * ({@link GpuDcLoadFlowEngine}); the AC load flow is the stock one. Registered as
 * {@code "OpenLoadFlowGpu"}, the same name as the GPU security analysis provider. Without a
 * device, or for a network whose DC equations have no device model, the stock DC engine runs.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
@AutoService(LoadFlowProvider.class)
public class GpuLoadFlowProvider extends OpenLoadFlowProvider {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuLoadFlowProvider.class);

    public static final String NAME = "OpenLoadFlowGpu";

    /** The last run's DC components solved on the device (test/validation probe; the provider runs on a
     *  pool thread, hence not thread-local), -1 when the whole run took the stock engine. */
    public static final java.util.concurrent.atomic.AtomicInteger LAST_GPU_COMPONENT_COUNT = new java.util.concurrent.atomic.AtomicInteger(-1);

    public GpuLoadFlowProvider(MatrixFactory matrixFactory, GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory) {
        super(matrixFactory, connectivityFactory);
    }

    public GpuLoadFlowProvider(MatrixFactory matrixFactory) {
        this(matrixFactory, new EvenShiloachGraphDecrementalConnectivityFactory<>());
    }

    public GpuLoadFlowProvider() {
        this(new SparseMatrixFactory());
    }

    @Override
    public String getName() {
        return NAME;
    }

    /**
     * The OLF provider already registers the {@code open-load-flow-parameters} JSON serializer;
     * {@code JsonLoadFlowParameters} keys the serializers by extension name and refuses a duplicate.
     * The GPU provider reads and writes the same extension through OLF's own serializer.
     */
    @Override
    public java.util.Optional<com.powsybl.commons.extensions.ExtensionJsonSerializer> getSpecificParametersSerializer() {
        return java.util.Optional.empty();
    }

    @Override
    protected List<DcLoadFlowResult> runDcEngine(Network network, DcLoadFlowParameters dcParameters, ReportNode reportNode) {
        if (!GpuDcSolver.isAvailable()) {
            LOGGER.info("GPU DC load flow: no device, stock engine");
            LAST_GPU_COMPONENT_COUNT.set(-1);
            return super.runDcEngine(network, dcParameters, reportNode);
        }
        int onDevice = 0;
        List<DcLoadFlowResult> results = new java.util.ArrayList<>();
        for (LfNetwork lfNetwork : LfNetwork.load(network, new LfNetworkLoaderImpl(), dcParameters.getNetworkParameters(), reportNode)) {
            if (lfNetwork.getValidity() != LfNetwork.Validity.VALID) {
                results.add(DcLoadFlowResult.createNoCalculationResult(lfNetwork));
                continue;
            }
            try (DcLoadFlowContext context = new DcLoadFlowContext(lfNetwork, dcParameters)) {
                boolean device = true;
                try (GpuDcLoadFlowEngine engine = new GpuDcLoadFlowEngine(context)) {
                    engine.prepare();
                    results.add(engine.run());
                } catch (UnsupportedOperationException e) {
                    LOGGER.warn("GPU DC load flow: {} — component {} on the stock engine", e.getMessage(), lfNetwork.getNumCC());
                    device = false;
                }
                if (!device) {
                    results.add(new DcLoadFlowEngine(context).run());
                } else {
                    onDevice++;
                }
            }
        }
        LAST_GPU_COMPONENT_COUNT.set(onDevice);
        return results;
    }
}
