/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocation;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveTermKind;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.gpu.GeneratorVoltageControlRoles;

import java.util.ArrayList;
import java.util.List;

/**
 * The reactive power of remote voltage control members, read by the reactive-limits check: which buses are
 * remote members, and the branch-flow terms that add up to each member's generation Q.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class RemoteMemberQPacker {

    private RemoteMemberQPacker() {
    }

    /**
     * OLF's OWN test for "is this a remote controller", which gates robust mode's unrealistic-voltage
     * PV→PQ freeze — copied verbatim from {@code ReactiveLimitsOuterLoop.isGeneratorRemoteController}:
     *
     * <pre>return controllerBus.getGeneratorVoltageControl().map(c -&gt; c.getControlledBus() != controllerBus).orElse(false);</pre>
     *
     * <p>Note which control it asks: the bus's OWN {@code getGeneratorVoltageControl()}, NOT
     * {@link GeneratorVoltageControlRoles#mainGeneratorVoltageControl}. For a MERGED group every member keeps its own DEPENDENT control whose
     * controlled bus is ITSELF, so OLF reads every such member as a LOCAL controller and skips the
     * whole {@code robustMode && generatorRemoteController && …} block for it — while the device's
     * {@code reactiveLimitVoltageControlRole} class, built from the MAIN control, calls that member "remote" and applies the
     * rule. That gap is [[#66]]: on rte6515 the device froze VL-2669_1..4 with
     * {@code MIN_REALISTIC_V} (their shared target, 12.4 kV on a 20 kV level = 0.62 pu, is below the
     * 0.816 pu trip point) where OLF froze nothing, so mode-4 own-Q balances replaced the DISTR_Q
     * sharing in the STRUCTURE and DUMMY_Q ended 8.410878e-03 pu off. Entries that answer false here
     * are packed as class 3 and the kernel's robust gate exempts them.
     */
    public static boolean isOlfRobustRemoteController(LfBus bus) {
        return bus.getGeneratorVoltageControl().map(c -> c.getControlledBus() != bus).orElse(false);
    }

    static RemoteMemberReactiveTerms remoteMemberQPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                     ExtractionColumns columns, GpuAcFamilyPacker.Packs family) {
        List<int[]> out = new ArrayList<>();
        List<Double> w = new ArrayList<>();
        int[] vCol = columns.vCol();
        for (LfBus bus : network.getBuses()) {
            boolean member = GeneratorVoltageControlRoles.isRemoteGeneratorVoltageControlControllerElement(bus);   // incl. a pilot that is a controller
            if (!member || vCol[bus.getNum()] < 0) {
                continue;                                // mirror the RL pack's remote-entry condition
            }
            // A pilot-self pilot whose Q is RE-ROUTED through a candidate table (alternateQRow = -2-g) already gets
            // its own injection at its own row from the diagnostic fill: patchGroupFrozenDestinations points the
            // group's frozen role-0 dest at the table's FALLBACK row, which for such a pilot IS its own row.
            // Adding the remoteMemberReactiveTerms twin on top counted it TWICE. Measured on rte6515 VL-6112_0 with the rotation
            // property on: q read 0.603080467 where OLF has 0.910540234, and
            // 1.218 (loadQ) + 2 * (0.910540234 - 1.218) = 0.603080 — the doubled injection, to six digits.
            // The reactive-limit check then never saw the pilot beyond its limit and never switched it to PQ, which
            // is what kept the rotation path from ever reaching the PQ arrangement it exists for.
            if (PilotSelfPacker.isPilotSelfEntry(bus) && columns.alternateQRow()[bus.getNum()] <= -2) {
                continue;
            }
            var voltageControl = bus.getGeneratorVoltageControl().orElseThrow();
            if (vCol[voltageControl.getControlledBus().getNum()] < 0) {
                continue;
            }
            // The row the reactive-limit check READS (native reactiveLimitReactivePowerRow): for a relocated child that is the
            // parent coupler's DUMMY_Q row, not its own V row, which now holds the coupler's ZERO_V. Filling
            // the own row left the check reading ZERO for every relocated member, so it could never switch one to PQ —
            // measured on rte6515's full busbar scope: the device's base stopped after its first
            // ReactiveLimits pass where OLF takes three, and came back MAX_ITERATION_REACHED.
            int row = ReactiveRelocation.relocatedQRow(bus.getNum(), columns.spanningTreeParentBranch(),
                    columns.dummyQCol(), vCol);
            if (bus.getShunt().isPresent()) {
                out.add(new int[] {row, family.shuntElementOf()[bus.getNum()], GpuTwoSides.NONE, GpuReactiveTermKind.SHUNT_COMPENSATOR_REACTIVE_FLOW});
                w.add(1.0);
            }
            for (LfBranch br : bus.getBranches()) {
                if (br.isDisabled()) {
                    continue;
                }
                int side = br.getBus1() == bus ? GpuTwoSides.ONE : GpuTwoSides.TWO;
                int e = family.closedBranchElementOf()[br.getNum()];
                if (e >= 0) {
                    out.add(new int[] {row, e, side, GpuReactiveTermKind.CLOSED_BRANCH_REACTIVE_FLOW});
                    w.add(1.0);
                    continue;
                }
                int oe = family.openBranchElementOf()[br.getNum()];
                if (oe >= 0) {
                    out.add(new int[] {row, oe, side, GpuReactiveTermKind.OPEN_BRANCH_REACTIVE_FLOW});
                    w.add(1.0);
                    continue;
                }
                // every coupler, not just the base tree's: a coupler out of it is held at dummyQ = 0 and adds
                // nothing, and one a contingency brings into OLF's tree carries this member's reactive flow
                if (br.isZeroImpedance(LoadFlowModel.AC)) {
                    int dummyQColumn = columns.dummyQCol()[br.getNum()];
                    if (dummyQColumn >= 0) {
                        out.add(new int[] {row, dummyQColumn, side, GpuReactiveTermKind.DUMMY_Q});
                        w.add(side == GpuTwoSides.ONE ? 1.0 : -1.0);
                    }
                }
            }
        }
        int[] row = new int[out.size()];
        int[] elem = new int[out.size()];
        int[] side = new int[out.size()];
        int[] kind = new int[out.size()];
        for (int i = 0; i < out.size(); i++) {
            row[i] = out.get(i)[0];
            elem[i] = out.get(i)[1];
            side[i] = out.get(i)[2];
            kind[i] = out.get(i)[3];
        }
        return new RemoteMemberReactiveTerms(row, elem, side, kind, GpuAcArrays.toDoubleArray(w));
    }
}
