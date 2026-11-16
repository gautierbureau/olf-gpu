/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.Equation;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The injection of a device state (base row layout) into a contingency-mutated equation system: every variable of the
 * current index must read the device value of ITS base row, whichever way the row mapping was brought up to date - from
 * the index's published delta (a bus's V and PHI removed or back) or by the full rebuild (the first injection, a
 * variable the base lacks, two index updates between injections). A wrong mapping hands a bus its neighbour's voltage.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuStateInjectorTest {

    @Test
    void everyVariableReadsItsOwnBaseRowWhicheverWayTheMappingIsUpdated() {
        Network iidm = IeeeCdfNetworkFactory.create14();
        LoadFlowParameters lfp = new LoadFlowParameters();
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.create(lfp);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);
        AcLoadFlowContext context = new AcLoadFlowContext(network, acp);
        new AcloadFlowEngine(context).run();
        EquationSystem<AcVariableType, AcEquationType> es = context.getEquationSystem();

        Map<Variable<AcVariableType>, Integer> baseRows = new HashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            baseRows.put(v, v.getRow());
        }
        double[] device = new double[baseRows.size()];
        for (int row = 0; row < device.length; row++) {
            device[row] = 1000 + row;                        // a value that names its base row
        }
        // two buses whose isolation removes their V and PHI: the one-branch bus and a two-branch one away from it
        LfBus a = network.getBuses().stream().filter(b -> !b.isSlack() && !b.isReference() && b.getBranches().size() == 1)
                .findFirst().orElseThrow();
        LfBus b = network.getBuses().stream().filter(x -> x != a && !x.isSlack() && !x.isReference()
                        && x.getBranches().size() == 2 && x.getBranches().stream().noneMatch(br -> touches(br, a)))
                .findFirst().orElseThrow();
        GpuStateInjector.Remap remap = new GpuStateInjector.Remap();

        injectAndCheck(es, baseRows, device, remap, 0, 1);  // the first one: full
        isolate(a, true);
        injectAndCheck(es, baseRows, device, remap, 1, 1);  // a's V and PHI removed: delta
        isolate(a, false);
        isolate(b, true);
        injectAndCheck(es, baseRows, device, remap, 2, 1);  // a back, b removed: delta
        isolate(b, false);
        injectAndCheck(es, baseRows, device, remap, 3, 1);  // back to the base: delta

        // a variable the base lacks (a strategy closing a coupler adds its dummy flows): the list is no longer the
        // base minus some rows, the sizes say so, and the mapping is rebuilt; the new variable reads 0
        Variable<AcVariableType> dummy = es.getVariable(network.getBranches().get(0).getNum(), AcVariableType.DUMMY_P);
        Equation<AcVariableType, AcEquationType> dummyEquation = es.createEquation(network.getBranches().get(0).getNum(),
                AcEquationType.DUMMY_TARGET_P);
        dummyEquation.addTerm(dummy.createTerm());
        injectAndCheck(es, baseRows, device, remap, 3, 2);
        isolate(a, true);
        injectAndCheck(es, baseRows, device, remap, 3, 3);  // still not a subsequence: full
        dummyEquation.setActive(false);
        isolate(a, false);
        injectAndCheck(es, baseRows, device, remap, 3, 4);  // the base again, found by the full merge

        // two index updates between injections: the last delta does not describe the change from the mapped list
        isolate(a, true);
        es.getIndex().getSortedVariablesToFindOnly();
        isolate(a, false);
        isolate(b, true);
        injectAndCheck(es, baseRows, device, remap, 3, 5);
        isolate(b, false);
        injectAndCheck(es, baseRows, device, remap, 4, 5);  // and the chain resumes
    }

    private static boolean touches(LfBranch branch, LfBus bus) {
        return branch.getBus1() == bus || branch.getBus2() == bus;
    }

    /** Disables (or enables back) a bus and its branches, as a contingency's apply (or restore) does. */
    private static void isolate(LfBus bus, boolean disabled) {
        for (LfBranch branch : bus.getBranches()) {
            branch.setDisabled(disabled);
        }
        bus.setDisabled(disabled);
    }

    private static void injectAndCheck(EquationSystem<AcVariableType, AcEquationType> es,
                                       Map<Variable<AcVariableType>, Integer> baseRows, double[] device,
                                       GpuStateInjector.Remap remap, int deltaUpdates, int fullRebuilds) {
        GpuStateInjector.injectState(es, baseRows, device, remap);
        List<Variable<AcVariableType>> vars = es.getIndex().getSortedVariablesToFindOnly();
        double[] state = es.getStateVector().get();
        assertEquals(vars.size(), state.length);
        for (Variable<AcVariableType> v : vars) {
            Integer baseRow = baseRows.get(v);
            assertEquals(baseRow == null ? 0 : device[baseRow], state[v.getRow()], v.toString());
        }
        assertEquals(deltaUpdates, remap.deltaUpdates, "mapping changes taken from the index's delta");
        assertEquals(fullRebuilds, remap.fullRebuilds, "full rebuilds of the mapping");
    }
}
