/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi;

import com.google.auto.service.AutoService;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.sensitivity.SensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityAnalysisProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * OLF's sensitivity analysis provider with the DC engine on the device
 * ({@link GpuDcSensitivityAnalysis}); AC is the stock one. Registered as {@code "OpenLoadFlowGpu"},
 * the same name as the GPU load flow and security analysis providers. Without a device the stock DC
 * engine runs.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
@AutoService(SensitivityAnalysisProvider.class)
public class GpuSensitivityAnalysisProvider extends OpenSensitivityAnalysisProvider {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuSensitivityAnalysisProvider.class);

    public static final String NAME = "OpenLoadFlowGpu";

    private final MatrixFactory matrixFactory;

    public GpuSensitivityAnalysisProvider(MatrixFactory matrixFactory, GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory) {
        super(matrixFactory, connectivityFactory);
        this.matrixFactory = matrixFactory;
    }

    public GpuSensitivityAnalysisProvider(MatrixFactory matrixFactory) {
        this(matrixFactory, new EvenShiloachGraphDecrementalConnectivityFactory<>());
    }

    public GpuSensitivityAnalysisProvider() {
        this(new SparseMatrixFactory());
    }

    @Override
    public String getName() {
        return NAME;
    }

    /**
     * The OLF provider already registers the {@code open-sensitivity-parameters} JSON serializer, and
     * the serializers are keyed by extension name, so this provider must not register it twice.
     */
    @Override
    public java.util.Optional<com.powsybl.commons.extensions.ExtensionJsonSerializer> getSpecificParametersSerializer() {
        return java.util.Optional.empty();
    }

    @Override
    protected AbstractSensitivityAnalysis<?, ?> createDcSensitivityAnalysis(GraphConnectivityFactory<LfBus, LfBranch> selectedConnectivityFactory,
                                                                            SensitivityAnalysisParameters sensitivityAnalysisParameters) {
        if (!GpuDcSolver.isAvailable()) {
            LOGGER.info("GPU DC sensitivity analysis: no device, stock engine");
            return super.createDcSensitivityAnalysis(selectedConnectivityFactory, sensitivityAnalysisParameters);
        }
        return new GpuDcSensitivityAnalysis(matrixFactory, selectedConnectivityFactory, sensitivityAnalysisParameters);
    }
}
