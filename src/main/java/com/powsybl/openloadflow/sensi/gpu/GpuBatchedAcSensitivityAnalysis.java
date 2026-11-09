/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi.gpu;

import com.powsybl.action.Action;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfContingency;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.network.impl.PropagatedContingencyCreationParameters;
import com.powsybl.openloadflow.sa.gpu.GpuPostContingencyBatch;
import com.powsybl.openloadflow.sensi.AcSensitivityAnalysis;
import com.powsybl.openloadflow.sensi.OpenSensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityFactorReader;
import com.powsybl.sensitivity.SensitivityResultWriter;
import com.powsybl.sensitivity.SensitivityVariableSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.LongAdder;

/**
 * OLF's AC sensitivity analysis with its post-contingency load flows solved together on the device: once the base case
 * is solved, the load flows of all the contingencies run in one batch ({@link GpuPostContingencyBatch}), and OLF then
 * computes each contingency's sensitivities on the Jacobian at the state the batch converged, the reactive-limit switches
 * of that load flow handed to the network as OLF's own load flow leaves them.
 *
 * <p>A contingency the batch does not bring (not converged on the device, out of the batch's scope) runs its own load
 * flow, as it does without the batch; so does every contingency of a run whose outer loops the batch does not cover.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuBatchedAcSensitivityAnalysis extends AcSensitivityAnalysis {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchedAcSensitivityAnalysis.class);

    /** How many contingencies' load flows a batch brought, over the JVM: a test reads its change over a run. */
    static final LongAdder BATCHED_CONTINGENCIES = new LongAdder();

    /** The run's network, which the batch maps the contingencies on: set for the duration of {@link #analyse}. */
    private volatile Network network;

    /** One batch per network partition, keyed by that partition's load flow context, closed with the run. */
    private final Map<AcLoadFlowContext, GpuPostContingencyBatch> batches = new ConcurrentHashMap<>();

    public GpuBatchedAcSensitivityAnalysis(MatrixFactory matrixFactory, GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory,
                                           SensitivityAnalysisParameters parameters) {
        super(matrixFactory, connectivityFactory, parameters);
    }

    @Override
    public void analyse(Network network, String workingVariantId, List<Contingency> contingencies, List<OperatorStrategy> operatorStrategies,
                        List<Action> actions, PropagatedContingencyCreationParameters creationParameters,
                        List<SensitivityVariableSet> variableSets, SensitivityFactorReader factorReader,
                        SensitivityResultWriter resultWriter, ReportNode sensiReportNode,
                        OpenSensitivityAnalysisParameters sensitivityAnalysisParametersExt,
                        Executor executor) throws ExecutionException {
        this.network = network;
        try {
            super.analyse(network, workingVariantId, contingencies, operatorStrategies, actions, creationParameters, variableSets,
                    factorReader, resultWriter, sensiReportNode, sensitivityAnalysisParametersExt, executor);
        } finally {
            batches.values().forEach(GpuPostContingencyBatch::close);
            batches.clear();
            this.network = null;
        }
    }

    /** Solves the contingencies' load flows together on the device, from the base state the context now holds. */
    @Override
    protected void beforeContingencySimulations(LfNetwork lfNetwork, AcLoadFlowContext context, List<PropagatedContingency> contingencies,
                                                LoadFlowParameters lfParameters, OpenLoadFlowParameters contingencyLfParametersExt) {
        super.beforeContingencySimulations(lfNetwork, context, contingencies, lfParameters, contingencyLfParametersExt);
        Network runNetwork = network;
        if (contingencies.isEmpty() || runNetwork == null) {
            return;
        }
        GpuPostContingencyBatch batch = solveBatch(runNetwork, lfNetwork, context, contingencies, lfParameters, contingencyLfParametersExt);
        batch.notBatchedReason().ifPresentOrElse(
            reason -> LOGGER.info("GPU sensitivity analysis: the contingencies' load flows run one by one, {}", reason),
            () -> LOGGER.info("GPU sensitivity analysis: {} of {} contingency load flows converged together on the device",
                    batch.size(), contingencies.size()));
        GpuPostContingencyBatch previous = batches.put(context, batch);
        if (previous != null) {
            previous.close();
        }
    }

    /**
     * The batch of the contingencies' load flows, from the base state the context holds: solved on the device
     * ({@link GpuPostContingencyBatch#solve}). The one place a subclass brings states from elsewhere.
     */
    protected GpuPostContingencyBatch solveBatch(Network network, LfNetwork lfNetwork, AcLoadFlowContext context,
                                                 List<PropagatedContingency> contingencies, LoadFlowParameters lfParameters,
                                                 OpenLoadFlowParameters contingencyLfParametersExt) {
        return GpuPostContingencyBatch.solve(network, lfNetwork, context, contingencies, lfParameters,
                contingencyLfParametersExt.isStartWithFrozenACEmulation(), lfNetwork.getReportNode());
    }

    /** The contingency's state from the batch when it converged there, its own load flow otherwise. */
    @Override
    protected boolean runPostContingencyLoadFlow(AcLoadFlowContext context, LfContingency lfContingency) {
        GpuPostContingencyBatch batch = batches.get(context);
        if (batch != null && batch.bringToConvergedState(context, lfContingency)) {
            BATCHED_CONTINGENCIES.increment();
            return true;
        }
        return super.runPostContingencyLoadFlow(context, lfContingency);
    }
}
