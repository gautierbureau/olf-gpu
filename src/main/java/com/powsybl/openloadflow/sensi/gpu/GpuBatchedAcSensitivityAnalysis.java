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
import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.gpu.batch.SensitivityBasisRequest;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfContingency;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.NetworkState;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.network.impl.PropagatedContingencyCreationParameters;
import com.powsybl.openloadflow.network.util.ParticipatingElement;
import com.powsybl.openloadflow.sa.gpu.GpuPostContingencyBatch;
import com.powsybl.openloadflow.sensi.AcSensitivityAnalysis;
import com.powsybl.openloadflow.sensi.GpuSensitivityFactorGroups;
import com.powsybl.openloadflow.sensi.OpenSensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityFactorReader;
import com.powsybl.sensitivity.SensitivityResultWriter;
import com.powsybl.sensitivity.SensitivityVariableSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

/**
 * OLF's AC sensitivity analysis with its post-contingency load flows solved together on the device: once the base case
 * is solved, the load flows of all the contingencies run in one batch ({@link GpuPostContingencyBatch}), and OLF then
 * computes each contingency's sensitivities on the Jacobian at the state the batch converged, the reactive-limit switches
 * of that load flow handed to the network as OLF's own load flow leaves them.
 *
 * <p>The sensitivity solves are batched too, when the factors allow it ({@link GpuSensitivityBasis}): the device solves a
 * basis of right-hand sides on each scenario's Jacobian at its converged state, and a contingency brought from the batch
 * has its states rebuilt from it at the {@code solveFactorsStates} seam, OLF's own right-hand sides decomposed exactly on
 * the basis; one that is not in it is solved on the Jacobian, as OLF does.
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

    /** How many contingencies' sensitivities were solved from the device's basis, over the JVM. */
    static final LongAdder BASIS_SOLVED_CONTINGENCIES = new LongAdder();

    /** The run's network, which the batch maps the contingencies on: set for the duration of {@link #analyse}. */
    private volatile Network network;

    /** The factor groups and factors of the partition this thread analyses, from OLF's createFactorGroups to the batch. */
    private final ThreadLocal<CapturedFactors> capturedFactors = new ThreadLocal<>();

    private record CapturedFactors(SensitivityFactorGroupList<AcVariableType, AcEquationType> groups,
                                   List<LfSensitivityFactor<AcVariableType, AcEquationType>> factors) {
    }

    /** What a network partition's contingencies are brought and solved from: its batch, its sensitivity basis (null
     *  without one), and the contingencies brought from the batch (their states are the device's). */
    private record Partition(GpuPostContingencyBatch batch, GpuSensitivityBasis basis, Set<String> brought) {
    }

    /** One per network partition, keyed by that partition's load flow context, closed with the run. */
    private final Map<AcLoadFlowContext, Partition> partitions = new ConcurrentHashMap<>();

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
            partitions.values().forEach(partition -> partition.batch().close());
            partitions.clear();
            capturedFactors.remove();
            this.network = null;
        }
    }

    /** OLF's factor groups, kept for the batch of the same partition: their right-hand sides make its sensitivity basis. */
    @Override
    protected SensitivityFactorGroupList<AcVariableType, AcEquationType> createFactorGroups(List<LfSensitivityFactor<AcVariableType, AcEquationType>> factors) {
        SensitivityFactorGroupList<AcVariableType, AcEquationType> groups = super.createFactorGroups(factors);
        capturedFactors.set(new CapturedFactors(groups, List.copyOf(factors)));
        return groups;
    }

    /** Solves the contingencies' load flows together on the device, from the base state the context now holds, and their
     *  sensitivity basis at each converged state when the factors allow one. */
    @Override
    protected void beforeContingencySimulations(LfNetwork lfNetwork, AcLoadFlowContext context, List<PropagatedContingency> contingencies,
                                                LoadFlowParameters lfParameters, OpenLoadFlowParameters contingencyLfParametersExt) {
        super.beforeContingencySimulations(lfNetwork, context, contingencies, lfParameters, contingencyLfParametersExt);
        CapturedFactors captured = capturedFactors.get();
        capturedFactors.remove();
        Network runNetwork = network;
        if (contingencies.isEmpty() || runNetwork == null) {
            return;
        }
        GpuSensitivityBasis basis = null;
        if (captured != null) {
            List<GpuSensitivityFactorGroups.Group> groups = GpuSensitivityFactorGroups.of(captured.groups());
            boolean participation = lfParameters.isDistributedSlack() && groups.stream().anyMatch(GpuSensitivityFactorGroups.Group::participation);
            GpuSensitivityBasis.Built built = GpuSensitivityBasis.build(lfNetwork, context.getEquationSystem(), groups,
                    GpuSensitivityFactorGroups.functionVariables(captured.factors()),
                    GpuSensitivityFactorGroups.functionTerms(captured.factors()),
                    participation ? participationByContingency(lfNetwork, contingencies, lfParameters, contingencyLfParametersExt) : null);
            basis = built.basis();
            if (basis == null) {
                LOGGER.info("GPU sensitivity analysis: contingency sensitivities solved on the CPU, {}", built.notBuiltReason());
            }
        }
        GpuPostContingencyBatch batch = solveBatch(runNetwork, lfNetwork, context, contingencies, lfParameters, contingencyLfParametersExt,
                basis != null ? basis.request() : null);
        batch.notBatchedReason().ifPresentOrElse(
            reason -> LOGGER.info("GPU sensitivity analysis: the contingencies' load flows run one by one, {}", reason),
            () -> LOGGER.info("GPU sensitivity analysis: {} of {} contingency load flows converged together on the device",
                    batch.size(), contingencies.size()));
        if (basis != null && batch.notBatchedReason().isEmpty()) {
            batch.sensitivityBasisNotSolvedReason().ifPresent(reason ->
                    LOGGER.info("GPU sensitivity analysis: contingency sensitivities solved on the CPU, {}", reason));
        }
        Partition previous = partitions.put(context, new Partition(batch,
                batch.sensitivityBasisNotSolvedReason().isEmpty() ? basis : null, ConcurrentHashMap.newKeySet()));
        if (previous != null) {
            previous.batch().close();
        }
    }

    /**
     * OLF's participation of each contingency, as {@code computeLfContingency} computes it: the contingency applied, the
     * participating elements of its slack component, bus to minus the factor. The network is restored after each.
     */
    private Map<String, Map<LfBus, Double>> participationByContingency(LfNetwork lfNetwork, List<PropagatedContingency> contingencies,
                                                                      LoadFlowParameters lfParameters, OpenLoadFlowParameters contingencyLfParametersExt) {
        Map<String, Map<LfBus, Double>> participation = new HashMap<>();
        NetworkState networkState = NetworkState.save(lfNetwork);
        for (PropagatedContingency contingency : contingencies) {
            Optional<LfContingency> optionalLfContingency = contingency.toLfContingency(lfNetwork);
            if (optionalLfContingency.isEmpty()) {
                continue;
            }
            LfContingency lfContingency = optionalLfContingency.get();
            lfContingency.apply(lfParameters.getBalanceType());
            Set<LfBus> disabled = lfContingency.getDisabledNetwork().getBuses();
            Set<LfBus> slackConnectedComponent = disabled.isEmpty() ? new HashSet<>(lfNetwork.getBuses())
                    : lfNetwork.getBuses().stream().filter(bus -> !disabled.contains(bus)).collect(Collectors.toSet());
            participation.put(lfContingency.getId(), getParticipatingElements(slackConnectedComponent, lfParameters.getBalanceType(),
                    contingencyLfParametersExt).stream()
                    .collect(Collectors.toMap(ParticipatingElement::getLfBus, element -> -element.getFactor(), Double::sum)));
            networkState.restore();
        }
        return participation;
    }

    /**
     * The batch of the contingencies' load flows, from the base state the context holds: solved on the device
     * ({@link GpuPostContingencyBatch#solve}), with the {@code sensitivityBasis} solved at each converged state when not
     * null. The one place a subclass brings states from elsewhere.
     */
    @SuppressWarnings("java:S107")
    protected GpuPostContingencyBatch solveBatch(Network network, LfNetwork lfNetwork, AcLoadFlowContext context,
                                                 List<PropagatedContingency> contingencies, LoadFlowParameters lfParameters,
                                                 OpenLoadFlowParameters contingencyLfParametersExt, SensitivityBasisRequest sensitivityBasis) {
        return GpuPostContingencyBatch.solve(network, lfNetwork, context, contingencies, lfParameters,
                contingencyLfParametersExt.isStartWithFrozenACEmulation(), sensitivityBasis, lfNetwork.getReportNode());
    }

    /** The contingency's state from the batch when it converged there, its own load flow otherwise. */
    @Override
    protected boolean runPostContingencyLoadFlow(AcLoadFlowContext context, LfContingency lfContingency) {
        Partition partition = partitions.get(context);
        if (partition != null && partition.batch().bringToConvergedState(context, lfContingency)) {
            partition.brought().add(lfContingency.getId());
            BATCHED_CONTINGENCIES.increment();
            return true;
        }
        return super.runPostContingencyLoadFlow(context, lfContingency);
    }

    /** A contingency brought from the batch with its basis solved there: its states rebuilt from the basis, when OLF's
     *  right-hand sides are in it; otherwise solved on the Jacobian, as OLF does. */
    @Override
    protected void solveFactorsStates(AcLoadFlowContext context, DenseMatrix factorsStates, LfContingency lfContingency) {
        Partition partition = lfContingency != null ? partitions.get(context) : null;
        if (partition != null && partition.basis() != null && partition.brought().contains(lfContingency.getId())) {
            double[] solutions = partition.batch().sensitivityBasis(lfContingency.getId());
            EquationSystem<AcVariableType, AcEquationType> es = context.getEquationSystem();
            if (solutions != null && partition.basis().solve(factorsStates, es, lfContingency.getId(), solutions,
                    partition.batch().currentRows(es, partition.basis().gatherRows()))) {
                BASIS_SOLVED_CONTINGENCIES.increment();
                return;
            }
        }
        super.solveFactorsStates(context, factorsStates, lfContingency);
    }
}
