/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.network;

import com.powsybl.iidm.network.*;

import java.time.ZonedDateTime;

/**
 * The meshed variants of OLF's {@code AutomationSystemNetworkFactory} network that the GPU automation system tests use.
 * OLF's factory does not ship them (they come with fork PRs about automation systems), so they live here, on the same
 * common network, copied from OLF's private {@code createCommonNetwork}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAutomationSystemNetworkFactory extends AbstractLoadFlowNetworkFactory {

    private GpuAutomationSystemNetworkFactory() {
    }

    /** OLF's {@code AutomationSystemNetworkFactory.createCommonNetwork}, unchanged. */
    private static Network createCommonNetwork() {
        Network network = Network.create("OverloadManagementSystemTestCase", "code");
        network.setCaseDate(ZonedDateTime.parse("2020-04-05T14:11:00.000+01:00"));
        Bus b1 = createBus(network, "s1", "b1", 225);
        Bus b2 = createBus(network, "s2", "b2", 225);
        Bus b3 = createBus(network, "s1", "b3", 63);
        Bus b3p = b3.getVoltageLevel().getBusBreakerView().newBus()
                .setId("b3p")
                .add();
        Bus b4 = createBus(network, "s2", "b4", 63);
        createGenerator(b1, "g1", 100, 230);
        createLoad(b3p, "ld3", 3, 2);
        createLoad(b4, "ld4", 90, 60);
        createLine(network, b1, b2, "l12", 0.1, 3);
        createLine(network, b3p, b4, "l34", 0.05, 3.2);
        createTransformer(network, "s1", b1, b3, "tr1", 0.2, 2, 1);
        createTransformer(network, "s2", b2, b4, "tr2", 0.3, 3, 1);
        return network;
    }

    /** The common network meshed: b3-b3p, b1-b2, b3p-b4 and b3-b4 are doubled or linked by a line. */
    public static Network createMeshed() {
        Network network = createCommonNetwork();
        Bus b1 = network.getBusBreakerView().getBus("b1");
        Bus b2 = network.getBusBreakerView().getBus("b2");
        Bus b3 = network.getBusBreakerView().getBus("b3");
        Bus b3p = network.getBusBreakerView().getBus("b3p");
        Bus b4 = network.getBusBreakerView().getBus("b4");
        createLine(network, b3, b3p, "l33p", 0.1, 3);
        createLine(network, b1, b2, "l12b", 0.1, 3);
        createLine(network, b3p, b4, "l34b", 0.05, 3.2);
        createLine(network, b3, b4, "l3b4", 0.05, 6);
        return network;
    }

    /** {@link #createMeshed()} with an overload management system on l34 that trips l33p (side 2) above 150 A. */
    public static Network createMeshedWithBranchTripping() {
        Network network = createMeshed();
        Substation s1 = network.getSubstation("s1");
        s1.newOverloadManagementSystem()
                .setId("l34_opens_l33p")
                .setEnabled(true)
                .setMonitoredElementId("l34")
                .setMonitoredElementSide(ThreeSides.ONE)
                .newBranchTripping()
                .setKey("l33p key")
                .setBranchToOperateId("l33p")
                .setSideToOperate(TwoSides.TWO)
                .setCurrentLimit(150.)
                .add()
                .add();
        return network;
    }
}
