/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi.gpu;

import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.solver.gpu.batch.SensitivityBasisRequest;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.sa.gpu.CpuPostContingencyBatches;
import com.powsybl.openloadflow.sa.gpu.GpuPostContingencyBatch;
import com.powsybl.openloadflow.sensi.AcSensitivityAnalysis;
import com.powsybl.openloadflow.sensi.OpenSensitivityAnalysisProvider;
import com.powsybl.sensitivity.SensitivityAnalysisParameters;

import java.util.List;
import java.util.Set;

/**
 * Test support: OLF's provider whose AC analysis is {@link GpuBatchedAcSensitivityAnalysis} with its batch made by OLF
 * on the CPU ({@link CpuPostContingencyBatches}), the device's states, decisions and, with {@code withBasis}, the
 * sensitivity basis it is asked for, without a device.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class CpuBatchedProvider extends OpenSensitivityAnalysisProvider {

    private final Set<String> leftOut;
    private final boolean baseDecisions;
    private final boolean withBasis;
    private CpuPostContingencyBatches.Made made;

    /** {@code leftOut}: contingencies the batch does not bring; {@code baseDecisions}: every contingency handed the base's
     *  PV/PQ states instead of its own. */
    CpuBatchedProvider(Set<String> leftOut, boolean baseDecisions) {
        this(leftOut, baseDecisions, false);
    }

    /** {@code withBasis}: the batch also solves the sensitivity basis the analysis asks for, so the contingencies'
     *  sensitivities are solved from it. */
    CpuBatchedProvider(Set<String> leftOut, boolean baseDecisions, boolean withBasis) {
        this.leftOut = leftOut;
        this.baseDecisions = baseDecisions;
        this.withBasis = withBasis;
    }

    /** What the last run's batch was made of. */
    CpuPostContingencyBatches.Made made() {
        return made;
    }

    @Override
    protected AcSensitivityAnalysis createAcSensitivityAnalysis(MatrixFactory matrixFactory,
                                                                GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory,
                                                                SensitivityAnalysisParameters sensitivityAnalysisParameters) {
        return new GpuBatchedAcSensitivityAnalysis(matrixFactory, connectivityFactory, sensitivityAnalysisParameters) {
            @Override
            protected boolean contractsOnDevice() {
                return true;                                 // the emulated batch contracts as the device does
            }

            @Override
            protected GpuPostContingencyBatch solveBatch(Network network, LfNetwork lfNetwork, AcLoadFlowContext context,
                                                         List<PropagatedContingency> contingencies, LoadFlowParameters lfParameters,
                                                         OpenLoadFlowParameters contingencyLfParametersExt,
                                                         SensitivityBasisRequest sensitivityBasis) {
                made = CpuPostContingencyBatches.solve(lfNetwork, context, contingencies, lfParameters, contingencyLfParametersExt,
                        leftOut, baseDecisions, withBasis ? sensitivityBasis : null);
                return made.batch();
            }
        };
    }
}
