/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.iidm.network.TwoSides;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSkips;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of OLF's {@code IncrementalTransformerReactivePowerControlOuterLoop}: rho taps driving a
 * branch Q flow. Builds the controllers; the loop runs in the kernel (incremental_flow_control.cu, OLF
 * order 9) on the table {@link DeviceFlowControllers} uploads, shared with
 * {@link GpuIncrementalPhaseControlOuterLoop}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuIncrementalTransformerReactivePowerControlOuterLoop {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuIncrementalTransformerReactivePowerControlOuterLoop.class);

    private static final double MIN_TARGET_DEADBAND_MVAR = 0.1;      // per-unit, OLF literal

    private GpuIncrementalTransformerReactivePowerControlOuterLoop() {
    }

    public static List<DeviceFlowControllers.Controller> buildControllers(LfNetwork network,
                                                                          EquationSystem<AcVariableType, AcEquationType> es,
                                                                          GpuBatchElementMapper elementMapper, int rpcMaxTapShift) {
        List<DeviceFlowControllers.Controller> out = new ArrayList<>();
        Map<Integer, Integer> rhoRows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BRANCH_RHO1) {
                rhoRows.put(v.getElementNum(), v.getRow());
            }
        }
        for (LfBranch controlled : network.getBranches()) {
            if (!controlled.isTransformerReactivePowerControlled() || controlled.isDisabled()) {
                continue;
            }
            var rc = controlled.getTransformerReactivePowerControl().orElseThrow();
            LfBranch controller = rc.getControllerBranch();
            Integer pinRow = rhoRows.get(controller.getNum());
            int elemIndex = elementMapper.getElementIndex(controller);
            int ctlElemIndex = elementMapper.getElementIndex(controlled);
            double[][] tapRows = GpuTransformerVoltageControlOuterLoop.tapModels(controller.getPiModel());
            if (pinRow == null || elemIndex < 0 || ctlElemIndex < 0 || tapRows.length == 0) {
                GpuBatchSkips.record(GpuBatchSkips.Slot.RPC, LOGGER,
                        "Incremental RPC controller {} not representable on the GPU batch — skipped", controller.getId());
                continue;
            }
            // OLF getHighestPriorityReactivePowerTarget: a generator reactive power control wins
            double target = controlled.getGeneratorReactivePowerControl()
                    .map(com.powsybl.openloadflow.network.GeneratorReactivePowerControl::getTargetValue)
                    .orElse(rc.getTargetValue());
            double halfDb = rc.getTargetDeadband().orElse(MIN_TARGET_DEADBAND_MVAR) / 2;
            out.add(new DeviceFlowControllers.Controller(DeviceFlowControllers.KIND_RPC, pinRow, 6, target, halfDb,
                    tapRows, DeviceFlowControllers.closestTap(tapRows, 6, controller.getPiModel().getR1()),
                    elemIndex, ctlElemIndex, rpcMaxTapShift, rc.getControlledSide() == TwoSides.ONE ? 1 : 2, controller.getNum()));
        }
        return out;
    }
}
