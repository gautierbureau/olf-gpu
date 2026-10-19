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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
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
 * DEVICE port of OLF's {@code AcIncrementalPhaseControlOuterLoop}: a1 taps driving a branch P flow in
 * CONTROLLER mode, or capping a branch current in LIMITER mode. Builds the controllers; the loop runs in
 * the kernel (incremental_flow_control.cu, OLF order 7) on the table {@link DeviceFlowControllers}
 * uploads, shared with {@link GpuIncrementalTransformerReactivePowerControlOuterLoop}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuIncrementalPhaseControlOuterLoop {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuIncrementalPhaseControlOuterLoop.class);

    private static final double MIN_TARGET_DEADBAND_P = 1.0 / 100.0; // 1 MW in per-unit

    private GpuIncrementalPhaseControlOuterLoop() {
    }

    public static List<DeviceFlowControllers.Controller> buildControllers(LfNetwork network,
                                                                          EquationSystem<AcVariableType, AcEquationType> es,
                                                                          GpuAcElementMapper elementMapper) {
        List<DeviceFlowControllers.Controller> out = new ArrayList<>();
        Map<Integer, Integer> a1Rows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BRANCH_ALPHA1) {
                a1Rows.put(v.getElementNum(), v.getRow());
            }
        }
        for (LfBranch controller : network.getBranches()) {
            var pcOpt = controller.getPhaseControl();
            if (pcOpt.isEmpty() || controller.isDisabled() || !controller.isPhaseController()) {
                continue;
            }
            var pc = pcOpt.orElseThrow();
            int kind = switch (pc.getMode()) {
                case CONTROLLER -> DeviceFlowControllers.KIND_PHASE_P;
                case LIMITER -> DeviceFlowControllers.KIND_PHASE_I;
                default -> -1;
            };
            if (kind < 0) {
                continue;
            }
            LfBranch controlled = pc.getControlledBranch();
            Integer pinRow = a1Rows.get(controller.getNum());
            int elemIndex = elementMapper.getElementIndex(controller);
            int ctlElemIndex = elementMapper.getElementIndex(controlled);
            double[][] tapRows = GpuTransformerVoltageControlOuterLoop.tapModels(controller.getPiModel());
            if (pinRow == null || elemIndex < 0 || ctlElemIndex < 0 || tapRows.length == 0) {
                GpuBatchSkips.record(GpuBatchSkips.Slot.PHASE, LOGGER,
                        "Incremental phase controller {} not representable on the GPU batch — skipped", controller.getId());
                continue;
            }
            double halfDb = Math.max(pc.getTargetDeadband(), MIN_TARGET_DEADBAND_P) / 2;
            out.add(new DeviceFlowControllers.Controller(kind, pinRow, 7, pc.getTargetValue(), halfDb,
                    tapRows, DeviceFlowControllers.closestTap(tapRows, 7, controller.getPiModel().getA1()),
                    elemIndex, ctlElemIndex, Integer.MAX_VALUE, pc.getControlledSide() == TwoSides.ONE ? 1 : 2, controller.getNum()));
        }
        return out;
    }
}
