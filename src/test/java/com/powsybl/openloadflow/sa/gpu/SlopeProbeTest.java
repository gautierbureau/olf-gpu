/**
 * TEMPORARY scenario probe — deleted before commit.
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.iidm.network.Line;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.StaticVarCompensator;
import com.powsybl.iidm.network.extensions.VoltagePerReactivePowerControlAdder;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.network.VoltageControlNetworkFactory;
import org.junit.jupiter.api.Test;

class SlopeProbeTest {

    private static Network net(double bmin, double bmax, double setpoint, double q0) {
        Network network = VoltageControlNetworkFactory.createWithStaticVarCompensator();
        network.getLoad("ld1").setQ0(q0);
        StaticVarCompensator svc1 = network.getStaticVarCompensator("svc1");
        svc1.setBmin(bmin).setBmax(bmax)
                .setVoltageSetpoint(setpoint).setRegulationMode(StaticVarCompensator.RegulationMode.VOLTAGE)
                .setRegulating(true);
        svc1.newExtension(VoltagePerReactivePowerControlAdder.class).withSlope(0.0001).add();
        Line l1 = network.getLine("l1");
        network.newLine().setId("l1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR()).setX(l1.getX()).add();
        return network;
    }

    @Test
    void probe() {
        double bmin = -0.004;
        double bmax = 0.004;
        double setpoint = 385;
        double q0 = 50;
        LoadFlowParameters lfp = new LoadFlowParameters().setDistributedSlack(true).setUseReactiveLimits(true);
        OpenLoadFlowParameters.create(lfp).setVoltagePerReactivePowerControl(true);
        LoadFlowParameters cpu = lfp.copy();
        OpenLoadFlowParameters.get(cpu);

        Network base = net(bmin, bmax, setpoint, q0);
        LoadFlow.find("OpenLoadFlow").run(base, cpu);
        System.err.println("PROBE base:   Vb2=" + base.getBusBreakerView().getBus("b2").getV()
                + " svcQ=" + base.getStaticVarCompensator("svc1").getTerminal().getQ());

        Network post = net(bmin, bmax, setpoint, q0);
        post.getLine("l1_BIS").getTerminal1().disconnect();
        post.getLine("l1_BIS").getTerminal2().disconnect();
        LoadFlow.find("OpenLoadFlow").run(post, cpu);
        System.err.println("PROBE l1_BIS out: Vb2=" + post.getBusBreakerView().getBus("b2").getV()
                + " svcQ=" + post.getStaticVarCompensator("svc1").getTerminal().getQ());
    }
}
