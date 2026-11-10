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

    /** Plan-B1 pack acceptance: every remote group member with finite limits is packed as an
     *  reactiveLimitVoltageControlRole=1 entry whose COMPARE row is the PILOT's v column (both RL switch conditions test
     *  the pilot's V against the GROUP target — B0), the write row stays the member's own bus row,
     *  and a remote member SWITCHED TO PQ in the base must NOT leak into the local basePqCapable shape. */
    @org.junit.jupiter.api.Test
    void remoteMembersPackedAsRemoteReactiveLimitEntries() {
        LfNetwork network = Networks.load(VoltageControlNetworkFactory.createWithGeneratorRemoteControl(),
                new LfNetworkParameters()
                        .setSlackBusSelector(new FirstSlackBusSelector())
                        .setGeneratorVoltageRemoteControl(true))
                .get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        // switch one member to PQ in the base (RL-disabled controller), like the base loadflow would
        LfBus memberSwitchedToPq = network.getBuses().stream()
                .filter(bus -> bus.getGenerators().stream().anyMatch(g -> g.getId().equals("g1")))
                .findFirst().orElseThrow();
        memberSwitchedToPq.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(false);
        AcEquationSystemCreator.updateGeneratorVoltageControl(memberSwitchedToPq.getGeneratorVoltageControl().orElseThrow(), es);

        GpuAcData d = GpuAcDataExtractor.extract(network, es);
        LfBus pilot = memberSwitchedToPq.getGeneratorVoltageControl().orElseThrow().getControlledBus();
        long members = network.getBuses().stream()
                .filter(bus -> bus.getGeneratorVoltageControl()
                        .filter(c -> c.getControlledBus() != bus).isPresent())
                .filter(bus -> bus.getMinQ() < bus.getMaxQ())
                .count();
        int remoteEntries = 0;
        for (int i = 0; i < d.reactiveLimits().voltageControlRole().length; i++) {
            if (d.reactiveLimits().voltageControlRole()[i] != 0) {
                remoteEntries++;
                assertNotEquals(d.reactiveLimits().voltageRow()[i], d.reactiveLimits().comparedVoltageRow()[i],
                        "a remote entry's compare row must be the pilot's v column, not its own row");
                assertEquals(d.reactiveLimits().targetV()[i],
                        memberSwitchedToPq.getGeneratorVoltageControl().orElseThrow().getTargetValue(), 0,
                        "remote entries carry the GROUP target");
            } else {
                assertEquals(d.reactiveLimits().voltageRow()[i], d.reactiveLimits().comparedVoltageRow()[i], "local entries compare at their own row");
            }
        }
        assertEquals(members, remoteEntries,
                "every remote member with finite limits must be packed as reactiveLimitVoltageControlRole=1 (pilot " + pilot.getId() + ")");
    }

    /** Plan-B2 engagement proof (the deadband-vacuous lesson: never trust a pass without showing
     *  the machinery FIRED): on the remote group with one member switched to PQ in the base, the canonical
     *  encoding must be in force — that member's frozen q dest SUPPRESSED, its own row the
     *  ALTERNATE, and the activation extractor emitting row mode BUS_TARGET_Q there (not EQUATION). */
    @org.junit.jupiter.api.Test
    void canonicalRemoteMemberEncodingEngages() {
        LfNetwork network = Networks.load(VoltageControlNetworkFactory.createWithGeneratorRemoteControl(),
                new LfNetworkParameters()
                        .setSlackBusSelector(new FirstSlackBusSelector())
                        .setGeneratorVoltageRemoteControl(true))
                .get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        LfBus memberSwitchedToPq = network.getBuses().stream()
                .filter(bus -> bus.getGenerators().stream().anyMatch(g -> g.getId().equals("g2")))
                .findFirst().orElseThrow();
        memberSwitchedToPq.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(false);
        AcEquationSystemCreator.updateGeneratorVoltageControl(memberSwitchedToPq.getGeneratorVoltageControl().orElseThrow(), es);

        int n = es.getIndex().getSortedVariablesToFind().size();     // force the index (row assignment)
        ExtractionColumns cols = ExtractionColumns.of(network, es);
        int bn = memberSwitchedToPq.getNum();
        int vColSwitchedToPq = cols.vCol()[bn];
        assertEquals(-1, cols.qRow()[bn], "canonical: the frozen q dest of the member switched to PQ is SUPPRESSED");
        assertEquals(vColSwitchedToPq, cols.alternateQRow()[bn], "canonical: the member's own row is the ALTERNATE");

        GpuAcActivation act = GpuAcActivationExtractor.extractActivation(network, es, n);
        assertEquals(4, act.rowMode()[vColSwitchedToPq],
                "an active BUS_TARGET_Q at a non-free remote member must extract as row mode BUS_TARGET_Q");
    }
}
