/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataCache;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.outerloop.AcHvdcAcEmulationLimitsOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.FreezingHvdcACEmulationOuterloop;
import com.powsybl.openloadflow.ac.outerloop.MonitoringVoltageOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuReactiveLimitsOuterLoop;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.gpu.batch.BatchScenarioResult;
import com.powsybl.openloadflow.ac.solver.gpu.batch.SensitivityBasisRequest;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfContingency;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * The post-contingency load flows of a list of contingencies, solved together on the device, for an analysis that
 * brings each contingency's converged state into OLF's network itself instead of solving it there: the AC sensitivity
 * analysis, whose sensitivities OLF then computes on the Jacobian at that state.
 *
 * <p>A contingency is brought to what OLF's own load flow leaves for that analysis to read: the reactive-limit switches
 * its solve made, handed to the buses as OLF's loop makes them (so the equation system is the one the device converged
 * on), then its state set as a full state (every derivative refreshed, the Jacobian read next) and the buses' voltages
 * updated. A run is batched only when every outer loop of the contingencies' load flows is one whose decisions are
 * handed back (reactive limits), one that changes no equation (distributed slack moves targets only, and the
 * sensitivity analysis computes its participation before the load flow), or one with nothing to decide on this network
 * (voltage monitoring without a monitoring SVC, the HVDC AC emulation loops without an HVDC); the rest is the device
 * batch's own scope ({@link GpuSaRunScope}). A contingency the device did not converge, islanded otherwise than OLF, or
 * whose decisions cannot all be handed back is not brought here: its analysis solves it as it would without the batch.
 *
 * <p>Used from the analysis's thread, one batch per network partition; {@link #close} drops a file-backed state store.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuPostContingencyBatch implements AutoCloseable {

    private final PrecomputedBatch precomputed;

    private final String notBatchedReason;

    private final String basisNotSolvedReason;

    private final GpuStateInjector.Remap remap = new GpuStateInjector.Remap();

    private GpuPostContingencyBatch(PrecomputedBatch precomputed, String notBatchedReason, String basisNotSolvedReason) {
        this.precomputed = Objects.requireNonNull(precomputed);
        this.notBatchedReason = notBatchedReason;
        this.basisNotSolvedReason = basisNotSolvedReason;
    }

    private static GpuPostContingencyBatch notBatched(String reason) {
        return new GpuPostContingencyBatch(PrecomputedBatch.empty(), reason, reason);
    }

    /** A batch bringing what {@code precomputed} holds: the device's precompute, or states made otherwise in its layout. */
    static GpuPostContingencyBatch of(PrecomputedBatch precomputed) {
        return of(precomputed, null);
    }

    private static GpuPostContingencyBatch of(PrecomputedBatch precomputed, String basisNotSolvedReason) {
        return new GpuPostContingencyBatch(precomputed, null, basisNotSolvedReason);
    }

    /**
     * Solves the contingencies' load flows on the device, from the converged base state the context holds, with the
     * contingency load flow parameters the context's parameters carry: {@code loadFlowParameters} the run's,
     * {@code startWithFrozenACEmulation} whether the contingencies start with frozen AC emulation. The contingency's
     * active power loss is distributed as the sensitivity analysis distributes it (OLF's default distribution, when the
     * slack is distributed). An empty batch, with its reason, when the run is not batched or the device is not there.
     */
    public static GpuPostContingencyBatch solve(Network network, LfNetwork lfNetwork, AcLoadFlowContext context,
                                                List<PropagatedContingency> contingencies, LoadFlowParameters loadFlowParameters,
                                                boolean startWithFrozenACEmulation, ReportNode reportNode) {
        return solve(network, lfNetwork, context, contingencies, loadFlowParameters, startWithFrozenACEmulation, null, reportNode);
    }

    /**
     * As above; with a {@code sensitivityBasis}, every converged contingency also has that basis solved on its Jacobian at
     * its converged state ({@link #sensitivityBasis}), on a network the basis's stable rows describe: no zero-impedance
     * coupler and no remote voltage control group (both move equations off a bus's own rows). On any other network the
     * load flows are batched without it ({@link #sensitivityBasisNotSolvedReason}).
     */
    @SuppressWarnings("java:S107")
    public static GpuPostContingencyBatch solve(Network network, LfNetwork lfNetwork, AcLoadFlowContext context,
                                                List<PropagatedContingency> contingencies, LoadFlowParameters loadFlowParameters,
                                                boolean startWithFrozenACEmulation, SensitivityBasisRequest sensitivityBasis,
                                                ReportNode reportNode) {
        Objects.requireNonNull(network);
        Objects.requireNonNull(lfNetwork);
        Objects.requireNonNull(context);
        Objects.requireNonNull(contingencies);
        Objects.requireNonNull(loadFlowParameters);
        AcLoadFlowParameters acParameters = context.getParameters();
        List<String> notHandedBack = loopsNotHandedBack(lfNetwork, acParameters);
        if (!notHandedBack.isEmpty()) {
            return notBatched("the decisions of the outer loops " + notHandedBack + " are not handed back to the network");
        }
        GpuSaRunScope.Scope scope = GpuSaRunScope.evaluate(lfNetwork, acParameters, loadFlowParameters, startWithFrozenACEmulation,
                OpenSecurityAnalysisParameters.CONTINGENCY_ACTIVE_POWER_LOSS_DISTRIBUTION_DEFAULT_VALUE);
        if (scope.batchDisabled()) {
            return notBatched("the device batch does not cover the outer loops " + scope.unsupportedOuterLoops());
        }
        if (scope.distributedSlack() != loadFlowParameters.isDistributedSlack()) {
            // the device pre-distributes a contingency's loss when its loops distribute active power, the sensitivity
            // analysis when the slack is distributed
            return notBatched("the contingencies' active power loss would not be distributed as the analysis distributes it");
        }
        SensitivityBasisRequest basis = sensitivityBasis;
        String basisNotSolvedReason = basis == null ? "no sensitivity basis was asked for" : null;
        if (basis != null && GpuAcNewtonSolver.isAvailable()) {
            GpuAcData data = GpuAcDataCache.extract(lfNetwork, context.getEquationSystem(), scope.batchConfig().distributedSlackParameters());
            if (data.getZeroImpedanceCount() > 0 || data.getDistrQContributionCount() > 0) {
                basisNotSolvedReason = "the network has zero-impedance couplers or remote voltage control groups";
                basis = null;
            }
        }
        PrecomputedBatch precomputed = GpuBatchPrecomputer.precompute(network, lfNetwork, context, contingencies, List.of(), List.of(),
                scope.distributedSlack(), scope.reactiveLimits(), scope.reactiveLimitTol(), scope.slackDistTol(), scope.convEpsPerEq(),
                scope.batchConfig(), true, basis, reportNode);
        if (!precomputed.isReady()) {
            return notBatched("the device batch did not run (no GPU, or a network or contingency it does not cover)");
        }
        return of(precomputed, basisNotSolvedReason);
    }

    /**
     * The outer loops of the contingencies' load flows whose decisions this batch does not hand back to the network: any
     * loop but reactive limits and distributed slack, unless it has nothing to decide on this network.
     */
    static List<String> loopsNotHandedBack(LfNetwork lfNetwork, AcLoadFlowParameters acParameters) {
        List<String> notHandedBack = new ArrayList<>();
        for (AcOuterLoop outerLoop : acParameters.getOuterLoops()) {
            String name = outerLoop.getName();
            boolean handedBack = switch (name) {
                case ReactiveLimitsOuterLoop.NAME, DistributedSlackOuterLoop.NAME -> true;
                case MonitoringVoltageOuterLoop.NAME -> !hasMonitoringVoltageSvc(lfNetwork);
                case AcHvdcAcEmulationLimitsOuterLoop.NAME, FreezingHvdcACEmulationOuterloop.NAME -> lfNetwork.getHvdcs().isEmpty();
                default -> false;
            };
            if (!handedBack) {
                notHandedBack.add(name);
            }
        }
        return notHandedBack;
    }

    private static boolean hasMonitoringVoltageSvc(LfNetwork lfNetwork) {
        for (LfBus bus : lfNetwork.getBuses()) {
            for (LfGenerator generator : bus.getGenerators()) {
                if (generator.getGeneratorControlType() == LfGenerator.GeneratorControlType.MONITORING_VOLTAGE) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Brings the contingency's state (the contingency applied to the network, its loss distributed) to the load flow the
     * device converged, as described above. False, with the network as it was, when this batch has no converged state
     * to bring for it: its load flow is then the caller's to run.
     */
    public boolean bringToConvergedState(AcLoadFlowContext context, LfContingency lfContingency) {
        EquationSystem<AcVariableType, AcEquationType> es = context.getEquationSystem();
        if (!precomputed.isReady() || precomputed.equationSystem() != es) {
            return false;
        }
        String id = lfContingency.getId();
        double[] state = precomputed.states().get(id);
        if (state == null) {
            return false;                                     // not converged on the device, or never batched
        }
        LfNetwork network = context.getNetwork();
        int[] islanded = precomputed.islandedBuses().get(id);
        if (islanded != null) {
            for (int num : islanded) {
                LfBus bus = network.getBus(num);
                if (bus == null || !bus.isDisabled()) {
                    return false;                             // the device froze a bus OLF solves: not the same network
                }
            }
        }
        GpuAcData data = precomputed.gpuData();
        BatchScenarioResult.ReactiveLimitDecisions decisions = precomputed.reactiveLimitDecisions().get(id);
        if (decisions == null && data.reactiveLimits().busNum() != null && data.reactiveLimits().busNum().length > 0) {
            return false;                                     // its switches were not read back
        }
        if (!GpuStateInjector.coversCurrentVariables(es, precomputed.baseVariableRows(), remap)) {
            return false;                                     // a variable the device has no value for
        }
        if (decisions != null && !GpuReactiveLimitsOuterLoop.applyScenarioToNetwork(network, data, decisions.finalState(),
                decisions.frozenRowTargetQ())) {
            return false;
        }
        if (!GpuStateInjector.injectFullState(es, precomputed.baseVariableRows(), state, remap)) {
            // a PV/PQ switch changes equations, never the variables to find: the check above holds here too
            throw new IllegalStateException("GPU post-contingency batch: the reactive-limit switches of contingency '" + id
                    + "' changed the variables to find");
        }
        AcSolverUtil.updateNetwork(network, es);
        return true;
    }

    /** The device's solutions of the sensitivity basis at a contingency's converged state, null when it has none. */
    public double[] sensitivityBasis(String contingencyId) {
        return precomputed.sensitivityBases().get(contingencyId);
    }

    /** The current variable row of each of {@code baseRows} (variable rows of the base state, the device's layout), -1 for
     *  a variable the contingency's system no longer finds. */
    public int[] currentRows(EquationSystem<AcVariableType, AcEquationType> es, int[] baseRows) {
        return GpuStateInjector.currentRows(es, precomputed.baseVariableRows(), remap, baseRows);
    }

    /** Why no sensitivity basis was solved, empty when it was. */
    public Optional<String> sensitivityBasisNotSolvedReason() {
        return Optional.ofNullable(basisNotSolvedReason);
    }

    /** How many contingencies the device converged (whose states this batch can bring). */
    public int size() {
        return precomputed.states().size();
    }

    /** Why the run is not batched, empty when it is. */
    public Optional<String> notBatchedReason() {
        return Optional.ofNullable(notBatchedReason);
    }

    @Override
    public void close() {
        if (precomputed.states() instanceof GpuStateStore store) {
            store.close();
        }
    }
}
