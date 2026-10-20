/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSkips;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.StateVector;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.network.LfNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * The phase controller tables, built ONCE for both routes.
 *
 * <p>The base load flow and the batched security analysis solve the same equations with the same
 * kernels, and they used to build these tables in two places — 150 lines against 60, calling the same
 * nine builders with the same arguments in two orders. The duplication was not free: SecondaryVoltageControl
 * was wired into one and not the other for a whole release, so every load flow with a zone paused the
 * driver while the same network ran it on device under the analysis. A new phase added here reaches both
 * routes or neither.
 *
 * <p>Building is shared; INSTALLING is not, and cannot be. The base load flow uploads straight onto its
 * batch-of-one context, while the analysis builds before its chunk context exists and uploads per chunk.
 * What both then apply to the result is {@link DevicePhaseOwnership}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DevicePhases {

    private DevicePhases() {
    }

    /**
     * Which loops this run configures, and the parameters the builders need from them. The analysis
     * fills it from its {@code BatchConfig}, the base load flow from the run's configured outer loops;
     * both then get the same tables out.
     *
     * <p>{@code rpcMaxTapShift} belongs to the incremental transformer REACTIVE POWER loop, not to the
     * transformer voltage one — the config field it comes from is named after the ratio tap control and
     * reads like the latter. Feeding it the wrong loop's shift moved a bus by 1.3e-4 kV.
     */
    public record Request(boolean monitoring,
                          boolean incrementalTvc,
                          boolean incrementalShunt,
                          boolean simpleShunt,
                          boolean rpc, boolean incrementalPhase, int rpcMaxTapShift,
                          boolean continuousPhase,
                          boolean discreteTvc, boolean tvcUseInitialTapPosition, double genVcMinNominalV,
                          boolean secondaryVc, double minPlausibleTargetV, double maxPlausibleTargetV) {
    }

    /** Every phase's controllers, null where the run does not configure the loop. */
    public record Built(List<GpuMonitoringVoltageOuterLoop.Ctl> monitoring,
                        List<GpuIncrementalTransformerVoltageControlOuterLoop.Controller> incrementalTvc,
                        List<GpuIncrementalShuntVoltageControlOuterLoop.Controller> incrementalShunt,
                        List<GpuShuntVoltageControlOuterLoop.Group> simpleShunt,
                        List<DeviceFlowControllers.Controller> flow,
                        List<GpuPhaseControlOuterLoop.Ctl> continuousPhase,
                        List<GpuTransformerVoltageControlOuterLoop.TransformerController> discreteTvc,
                        GpuTransformerVoltageControlOuterLoop.FreezeTables freeze,
                        GpuSecondaryVoltageControlOuterLoop.Controllers secondaryVc,
                        EsContext esContext) {
    }

    /** The observable-evaluation context captured at build time: the shared es state vector (scenario
     *  states are loaded into it so OLF's own terms evaluate/differentiate at the right point). */
    public record EsContext(StateVector stateVector, int n) {
    }

    /**
     * Build every configured phase's tables. Resets {@link GpuBatchSkips} first, so the caller can ask
     * {@link DevicePhaseOwnership} afterwards whether each loop came out COMPLETE.
     *
     * <p>{@code reactiveLimitVoltageRow} is the reactive-limit row pack the monitoring and secondary-voltage phases index
     * against, {@code reactiveLimitBusNum} its bus numbers (the generator freeze needs them to find a REMOTE
     * candidate's entry) and {@code n} the system size. One element mapper is built here and shared: the
     * base load flow used to construct a fresh one per builder, three times over.
     */
    public static Built build(Request request, LfNetwork network,
                              EquationSystem<AcVariableType, AcEquationType> es,
                              TargetVector<AcVariableType, AcEquationType> targetVector,
                              GpuAcActivation activation, double[] targets, int[] reactiveLimitVoltageRow, int[] reactiveLimitBusNum, int n) {
        GpuBatchSkips.reset();
        GpuAcElementMapper mapper = new GpuAcElementMapper(network);
        int shuntOffset = mapper.getClosedBranchCount() + mapper.getOpenBranchCount();
        List<GpuTransformerVoltageControlOuterLoop.TransformerController> discreteTvc = request.discreteTvc()
                ? GpuTransformerVoltageControlOuterLoop.buildTransformerControllers(network, es, targetVector, activation,
                        targets, mapper, request.tvcUseInitialTapPosition())
                : null;
        return new Built(
                request.monitoring()
                        ? GpuMonitoringVoltageOuterLoop.buildControllers(network, es, targetVector, activation, targets, reactiveLimitVoltageRow)
                        : null,
                request.incrementalTvc()
                        ? GpuIncrementalTransformerVoltageControlOuterLoop.buildControllers(network, es, mapper)
                        : null,
                request.incrementalShunt()
                        ? GpuIncrementalShuntVoltageControlOuterLoop.buildControllers(network, es, mapper, shuntOffset)
                        : null,
                request.simpleShunt()
                        ? GpuShuntVoltageControlOuterLoop.buildGroups(network, es, targetVector, activation, targets, mapper, shuntOffset)
                        : null,
                request.rpc() || request.incrementalPhase() ? flowControllers(request, network, es, mapper) : null,
                request.continuousPhase()
                        ? GpuPhaseControlOuterLoop.buildControllers(network, es, targetVector, activation, targets, mapper)
                        : null,
                discreteTvc,
                discreteTvc != null
                        ? GpuTransformerVoltageControlOuterLoop.buildFreezeTables(network, es, discreteTvc,
                                request.genVcMinNominalV(), reactiveLimitBusNum)
                        : null,
                request.secondaryVc() && GpuSecondaryVoltageControlOuterLoop.supported(network)
                        ? GpuSecondaryVoltageControlOuterLoop.build(network, es, reactiveLimitVoltageRow,
                                new EsContext(es.getStateVector(), n),
                                request.minPlausibleTargetV(), request.maxPlausibleTargetV(), mapper)
                        : null,
                new EsContext(es.getStateVector(), n));
    }

    /** One table for the two flow-observed loops, reactive power controllers first: the kernel indexes it. */
    private static List<DeviceFlowControllers.Controller> flowControllers(Request request, LfNetwork network,
                                                                          EquationSystem<AcVariableType, AcEquationType> es,
                                                                          GpuAcElementMapper mapper) {
        List<DeviceFlowControllers.Controller> flow = new ArrayList<>();
        if (request.rpc()) {
            flow.addAll(GpuIncrementalTransformerReactivePowerControlOuterLoop.buildControllers(network, es, mapper,
                    request.rpcMaxTapShift()));
        }
        if (request.incrementalPhase()) {
            flow.addAll(GpuIncrementalPhaseControlOuterLoop.buildControllers(network, es, mapper));
        }
        return flow;
    }
}
