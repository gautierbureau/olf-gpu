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
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.AbstractLoadFlowNetworkFactory;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.VoltageControl;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Two generators, each regulating its OWN bus, on buses joined by a ZERO-IMPEDANCE branch. OLF merges
 * the two generator voltage controls into one group: the first control stays MAIN and the second
 * becomes DEPENDENT, with {@code mainMergedVoltageControl} pointing at the first. Both buses are then
 * controller elements of the merged group and the second one carries a DISTR_Q equation.
 *
 * <p>This is the smallest reproducer of a defect found on rte6515 at 2000 busbar-section contingencies.
 * A merge-status FILTER ({@code getMergeStatus() == MAIN}) answers the wrong question on the DEPENDENT
 * object: its own {@code controllerElements} list is just itself regulating itself, so it looks like a
 * purely LOCAL control and its merged element list is unreachable. Every predicate written that way
 * reported "not a controller" for a bus that is one, and {@link GpuAcRowIndexing#computeQRow} left the
 * bus's reactive balance on its own BUS_V row — the very row that already hosts its DISTR_Q equation.
 * The zero-impedance family then added its direct +-DUMMY_Q coupling there next to the DISTR_Q
 * derivative and the two CANCELLED EXACTLY, leaving the coupler's DUMMY_Q in no equation at all.
 *
 * <p>A structurally zero COLUMN is the worst kind of wrong: the free variable perturbs nothing, so the
 * mismatch norm stays healthy and no residual check ever fires, while the Newton step it belongs to is
 * junk. It is asserted here directly, on the row indexing, rather than waited for downstream.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuMergedGenVoltageControlTest extends AbstractLoadFlowNetworkFactory {

    /**
     * <pre>
     *   ld0            g1          g2
     *    |             |           |
     *   b0 --- l01 --- b1 === l12 === b2
     *        (x=0.1)      (r=x=0)
     * </pre>
     * g1 and g2 each regulate their own bus at the same target, so the zero-impedance l12 merges their
     * two voltage controls and the group shares reactive power through a DISTR_Q equation.
     */
    private static Network mergedAcrossCoupler() {
        Network network = Network.create("merged-gen-vc", "code");
        Bus b0 = createBus(network, "s", "b0");
        Bus b1 = createBus(network, "s", "b1");
        Bus b2 = createBus(network, "s", "b2");
        createLoad(b0, "ld0", 4);
        createGenerator(b1, "g1", 2, 1.0);
        createGenerator(b2, "g2", 2, 1.0);
        createLine(network, b0, b1, "l01", 0.1);
        createLine(network, b1, b2, "l12", 0.0, 0.0);
        return network;
    }

    @Test
    void dependentControllerKeepsNoOwnQRow() {
        LfNetwork network = Networks.load(mergedAcrossCoupler(),
                new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        // Until the state vector is initialised the index is not built and EVERY getRow() answers -1,
        // which makes a "must be -1" assertion pass for the wrong reason.
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());

        // The fixture must really produce a merge, or the rest of the test proves nothing.
        List<LfBus> dependent = network.getBuses().stream()
                .filter(b -> b.getGeneratorVoltageControl()
                        .map(c -> c.getMergeStatus() == VoltageControl.MergeStatus.DEPENDENT).orElse(false))
                .toList();
        assertEquals(1, dependent.size(), "the zero-impedance branch must merge the two voltage controls");
        LfBus dep = dependent.get(0);

        // On the DEPENDENT object itself both tests a MAIN-filter would rely on read the wrong thing:
        // it looks local, and its element list is only itself. That is exactly the trap.
        var own = dep.getGeneratorVoltageControl().orElseThrow();
        assertTrue(own.isLocalControl(), "the dependent control looks LOCAL in isolation - the trap");
        assertEquals(dep, own.getControlledBus(), "...regulating its own bus, in isolation");

        // What is actually true of this bus, resolved on the MAIN control.
        assertTrue(GpuAcRowIndexing.isRemoteGenVcControllerElement(dep),
                "a merged controller element must be recognised through its MAIN control");

        // Its reactive injection feeds the group's DISTR_Q row, so it must NOT also scatter to a Q row
        // of its own. With the merge-status filter this came back as the bus's own BUS_V row, which is
        // where its DISTR_Q equation already lives - and the zero-impedance coupling cancelled against
        // it, emptying the coupler's DUMMY_Q column.
        assertTrue(GpuAcRowIndexing.row(es, dep.getNum(), AcEquationType.DISTR_Q) >= 0,
                "the dependent controller must carry the group's DISTR_Q equation");
        assertEquals(-1, GpuAcRowIndexing.row(es, dep.getNum(), AcEquationType.BUS_TARGET_Q),
                "...and no own reactive target while it controls");
        assertEquals(-1, ExtractionColumns.of(network, es).qRow()[dep.getNum()],
                "a merged controller element must have NO own Q-balance row");
    }

    /**
     * The consequence the row indexing exists to prevent, asserted on the assembled structure: every
     * variable must appear in at least one equation. {@link #dependentControllerKeepsNoOwnQRow} pins the
     * cause; this pins the effect, so a future routing change that empties a column some other way is
     * caught too.
     */
    @Test
    void couplerDummyQIsConstrained() {
        LfNetwork network = Networks.load(mergedAcrossCoupler(),
                new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        // Until the state vector is initialised the index is not built and EVERY getRow() answers -1,
        // which makes a "must be -1" assertion pass for the wrong reason.
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        ExtractionColumns columns = ExtractionColumns.of(network, es);

        int coupler = network.getBranchById("l12").getNum();
        int dummyQ = columns.dummyQCol()[coupler];
        assertTrue(dummyQ >= 0, "the zero-impedance branch must have a DUMMY_Q variable");

        // The DISTR_Q derivative is the ONLY thing that may write the coupler's dummy-Q into the merged
        // controller's row; the direct zero-impedance q coupling must stay away from it. Both buses of
        // the coupler are controller elements of the merged group, so BOTH must be suppressed.
        LfBus b1 = network.getBranchById("l12").getBus1();
        LfBus b2 = network.getBranchById("l12").getBus2();
        assertNotNull(b1);
        assertNotNull(b2);
        assertEquals(-1, columns.qRow()[b1.getNum()], "coupler side 1 is a merged controller element");
        assertEquals(-1, columns.qRow()[b2.getNum()], "coupler side 2 is a merged controller element");
        assertFalse(columns.qRow()[b1.getNum()] == columns.qRow()[b2.getNum()]
                && columns.qRow()[b1.getNum()] >= 0,
                "the two sides must never share a live Q row - their couplings would cancel");
    }
}
