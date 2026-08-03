/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.VoltageControlNetworkFactory;
import com.powsybl.openloadflow.network.impl.Networks;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Remote generator voltage control with one controller switched PQ by reactive limits: the disabled controller
 * must recover its own BUS_V-backed BUS_TARGET_Q row, while the pilot bus reactive balance moves to the next
 * enabled controller's BUS_V row. This is a structural change for the GPU extractor, not just a target remap.
 */
class GpuRemoteControllerReactiveLimitsTest {

    @Test
    void disabledRemoteControllerRecoversOwnQRow() {
        LfNetwork network = Networks.load(VoltageControlNetworkFactory.createWithGeneratorRemoteControl(),
                new LfNetworkParameters()
                        .setSlackBusSelector(new FirstSlackBusSelector())
                        .setGeneratorVoltageRemoteControl(true))
                .get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();

        LfBus controller = network.getBuses().stream()
                .filter(bus -> bus.getGenerators().stream().anyMatch(generator -> generator.getId().equals("g1")))
                .findFirst()
                .orElse(null);
        assertNotNull(controller);
        controller.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(false);
        AcEquationSystemCreator.updateGeneratorVoltageControl(controller.getGeneratorVoltageControl().orElseThrow(), es);

        GpuAcDataExtractor.extract(network, es);
        GpuAcActivation activation = GpuAcActivation.fromEquationSystem(network, es);

        LfBus freeController = network.getBuses().stream()
                .filter(bus -> bus != controller)
                .filter(bus -> bus.getGeneratorVoltageControl().isPresent())
                .filter(bus -> !bus.isGeneratorVoltageControlled())
                .filter(LfBus::isGeneratorVoltageControlEnabled)
                .filter(bus -> es.getEquation(bus.getNum(), AcEquationType.DISTR_Q)
                        .map(eq -> !eq.isActive())
                        .orElse(true))
                .findFirst()
                .orElse(null);
        LfBus pilot = network.getBuses().stream()
                .filter(LfBus::isGeneratorVoltageControlled)
                .findFirst()
                .orElse(null);
        assertNotNull(freeController);
        assertNotNull(pilot);
        int controllerVRow = es.getVariableSet().getVariable(controller.getNum(), AcVariableType.BUS_V).getRow();
        int freeControllerVRow = es.getVariableSet().getVariable(freeController.getNum(), AcVariableType.BUS_V).getRow();
        int controllerQCol = es.getEquation(controller.getNum(), AcEquationType.BUS_TARGET_Q).orElseThrow().getColumn();
        int pilotQCol = es.getEquation(pilot.getNum(), AcEquationType.BUS_TARGET_Q).orElseThrow().getColumn();

        assertEquals(controllerQCol, activation.targetMap()[controllerVRow],
                "a remote controller switched to PQ must map its BUS_TARGET_Q onto its own BUS_V row");
        assertEquals(pilotQCol, activation.targetMap()[freeControllerVRow],
                "the pilot Q balance must move to the remaining free controller BUS_V row");
        assertNotEquals(-1, controllerVRow);
    }
}
