/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.AbstractLoadFlowNetworkFactory;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.impl.Networks;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The PILOT-SELF voltage-control group: a generator voltage control whose controlled bus is ALSO one of its
 * controllers, with other controller buses alongside it. OLF's rule
 * ({@code AcEquationSystemCreator.updateRemoteVoltageControlEquations}) is a ROTATION over the live enabled
 * controllers — {@code BUS_TARGET_V} active while any is enabled, {@code BUS_TARGET_Q} for every one switched to PQ,
 * {@code DISTR_Q} for every enabled one except the FIRST — so which controller is "free" changes every time
 * one switches to PQ.
 *
 * <p>In this layout rows are VARIABLES, so the rotation cannot move {@code BUS_TARGET_V} (a mode-0 identity
 * constrains its OWN row's variable; on a member's row it would hold that member's v at the pilot's target).
 * What moves is the reactive balance of a pilot SWITCHED TO PQ, onto the first ENABLED member's row. {@code computeQRow}
 * already places it there, but frozen at extraction, while the device's reactive-limits loop switches buses
 * between PV and PQ mid-solve — the mismatch that left rte6515's busbar base rank-deficient.
 *
 * <p>This fixture exists because rte6515's instances of the shape are buried in a 13k-variable system: here
 * the group is three buses and the table can be asserted directly. The pack is keyed by RL ENTRY rather than
 * by row precisely because the row to pick is mode 4 BECAUSE it carries the relocated balance, which is
 * circular — {@code limitType} per entry is the ground truth OLF itself uses.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuPilotSelfGroupTest extends AbstractLoadFlowNetworkFactory {

    /**
     * b1 carries a generator controlling its OWN voltage, and b2/b3 carry generators controlling b1
     * REMOTELY, so OLF merges them into one control: controlled bus b1, controllers {b1, b2, b3}.
     */
    private static Network pilotSelfNetwork() {
        Network network = Network.create("pilot-self", "code");
        Bus b1 = createBus(network, "s", "b1");
        Bus b2 = createBus(network, "s", "b2");
        Bus b3 = createBus(network, "s", "b3");
        Bus b4 = createBus(network, "s", "b4");
        createLoad(b4, "ld4", 6, 3);
        createLine(network, b1, b4, "l14", 0.1);
        createLine(network, b2, b4, "l24", 0.1);
        createLine(network, b3, b4, "l34", 0.1);
        createGenerator(b1, "g1", 2, 1.0);                   // the PILOT: controls its own bus
        var g2 = b2.getVoltageLevel().newGenerator()
                .setId("g2").setBus("b2").setConnectableBus("b2")
                .setTargetP(2).setTargetV(1.0 * b1.getVoltageLevel().getNominalV())
                .setMinP(0).setMaxP(10).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(network.getGenerator("g1").getTerminal())
                .add();
        g2.newMinMaxReactiveLimits().setMinQ(-5).setMaxQ(5).add();
        var g3 = b3.getVoltageLevel().newGenerator()
                .setId("g3").setBus("b3").setConnectableBus("b3")
                .setTargetP(2).setTargetV(1.0 * b1.getVoltageLevel().getNominalV())
                .setMinP(0).setMaxP(10).setVoltageRegulatorOn(true)
                .setRegulatingTerminal(network.getGenerator("g1").getTerminal())
                .add();
        g3.newMinMaxReactiveLimits().setMinQ(-5).setMaxQ(5).add();
        network.getGenerator("g1").newMinMaxReactiveLimits().setMinQ(-5).setMaxQ(5).add();
        return network;
    }

    @Test
    void tableKeysTheGroupByReactiveLimitEntryAndKeepsMergedOrder() {
        LfNetwork network = Networks.load(pilotSelfNetwork(),
                new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        com.powsybl.openloadflow.ac.solver.AcSolverUtil.initStateVector(network, es,
                new com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer());

        LfBus pilot = network.getBuses().stream()
                .filter(b -> b.getGeneratorVoltageControl()
                        .filter(c -> c.getMergedControllerElements().size() > 1
                                && c.getControlledBus() == b && c.getMergedControllerElements().contains(b))
                        .isPresent())
                .findFirst().orElse(null);
        org.junit.jupiter.api.Assumptions.assumeTrue(pilot != null,
                "this OLF build did not merge the generators into a pilot-self group");
        List<LfBus> members = pilot.getGeneratorVoltageControl().orElseThrow().getMergedControllerElements();

        ExtractionColumns columns = ExtractionColumns.of(network, es);
        // Every controller bus is its own RL entry here, in bus order, which is what the real pack builds.
        int[] reactiveLimitBusNum = network.getBuses().stream().mapToInt(LfBus::getNum).toArray();
        PilotSelfPacker.PilotSelfPack pack = PilotSelfPacker.pilotSelfGroups(network, columns, reactiveLimitBusNum);

        assertEquals(1, pack.groupCount(), "the one pilot-self group must be packed");
        assertEquals(members.size(), pack.ptr()[1] - pack.ptr()[0], "every controller must be a member entry");
        assertEquals(columns.vCol()[pilot.getNum()], pack.pilotRow()[0],
                "BUS_TARGET_V stays on the controlled bus's own row, so that is the pilot row");
        assertEquals(pilot.getNum(), reactiveLimitBusNum[pack.pilotEntry()[0]], "the pilot entry must be the pilot's");
        for (int k = 0; k < members.size(); k++) {            // MERGED ORDER matters: OLF's free role is index 0
            LfBus m = members.get(k);
            int at = pack.ptr()[0] + k;
            assertEquals(columns.vCol()[m.getNum()], pack.memberRow()[at],
                    "member " + m.getId() + " must keep its merged position");
            assertEquals(m.getNum(), reactiveLimitBusNum[pack.memberEntry()[at]], "member entry must resolve to the bus");
        }
        assertTrue(pack.vTarget()[0] > 0, "the group's voltage target must travel with it");
    }

    /** A group whose controlled bus is NOT one of its controllers keeps the existing candidate-table path. */
    @Test
    void aPlainRemoteGroupIsNotPackedHere() {
        Network network = pilotSelfNetwork();
        network.getGenerator("g1").setTargetQ(0).setVoltageRegulatorOn(false);   // pilot stops controlling
        LfNetwork lf = Networks.load(network,
                new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(lf).create();
        com.powsybl.openloadflow.ac.solver.AcSolverUtil.initStateVector(lf, es,
                new com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer());
        ExtractionColumns columns = ExtractionColumns.of(lf, es);
        int[] reactiveLimitBusNum = lf.getBuses().stream().mapToInt(LfBus::getNum).toArray();
        PilotSelfPacker.PilotSelfPack pack = PilotSelfPacker.pilotSelfGroups(lf, columns, reactiveLimitBusNum);
        assertEquals(0, pack.groupCount(), "a pilot that is not one of its own controllers must not be packed");
    }
}
