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
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;

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

    /** Remote-member OWN reactive-injection contributors (plan-B3): per contribution, the member's
     *  own row ({@code row}), the family element + side + kind (same encoding as the DISTR_Q pack)
     *  and a weight (1, or ±1 for a zero-impedance DUMMY_Q term). A twin of the DISTR_Q assembly
     *  accumulates each member's own bus q into the diagnostic Q buffer at its own row — that slot
     *  is free because the main qBuf fill suppresses member rows — so the RL kernel's
     *  {@code q = qBuf[row] + loadQ} works verbatim for remote entries. */
    record RmqPack(int[] row, int[] elem, int[] side, int[] kind, double[] weight) {
    }

    /** A REMOTE member entry of the reactive-limit pack: a controller of a MAIN remote group that is
     *  not the group's controlled bus. Shared with ReactiveCapabilityCurves, which REPLAYS this enumeration
     *  and must count exactly the same buses (a drift throws and the run falls to the CPU). */
    public static boolean isRemoteMemberEntry(LfBus bus) {
        GeneratorVoltageControl vc = mainGenVc(bus);
        return vc != null && GpuAcRowIndexing.isRemoteGenVcControllerElement(bus) && vc.getControlledBus() != bus;
    }

    /**
     * OLF's OWN test for "is this a remote controller", which gates robust mode's unrealistic-voltage
     * PV→PQ freeze — copied verbatim from {@code ReactiveLimitsOuterLoop.isGeneratorRemoteController}:
     *
     * <pre>return controllerBus.getGeneratorVoltageControl().map(c -&gt; c.getControlledBus() != controllerBus).orElse(false);</pre>
     *
     * <p>Note which control it asks: the bus's OWN {@code getGeneratorVoltageControl()}, NOT
     * {@link #mainGenVc}. For a MERGED group every member keeps its own DEPENDENT control whose
     * controlled bus is ITSELF, so OLF reads every such member as a LOCAL controller and skips the
     * whole {@code robustMode && generatorRemoteController && …} block for it — while the device's
     * {@code rlRemote} class, built from the MAIN control, calls that member "remote" and applies the
     * rule. That gap is [[#66]]: on rte6515 the device froze VL-2669_1..4 with
     * {@code MIN_REALISTIC_V} (their shared target, 12.4 kV on a 20 kV level = 0.62 pu, is below the
     * 0.816 pu trip point) where OLF froze nothing, so mode-4 own-Q balances replaced the DISTR_Q
     * sharing in the STRUCTURE and DUMMY_Q ended 8.410878e-03 pu off. Entries that answer false here
     * are packed as class 3 and the kernel's robust gate exempts them.
     */
    public static boolean isOlfRobustRemoteController(LfBus bus) {
        return bus.getGeneratorVoltageControl().map(c -> c.getControlledBus() != bus).orElse(false);
    }

    /**
     * The bus's MAIN generator voltage control, which is the only one that can answer a question about the
     * GROUP: who the controlled bus is, what the target is, which nominal V it refers to.
     *
     * <p>A merged group keeps ONE main control (on the bus OLF picked) and leaves every other member holding
     * a DEPENDENT control that regulates ITSELF. Asking {@code bus.getGeneratorVoltageControl()} therefore
     * answered "I control my own bus" for every member, so each one was classified as the pilot
     * ({@code isPilotSelfEntry}) with its own row as the pilot row. Since class 2 means "unpinned, my row is
     * the group's voltage IDENTITY (mode 0)" while a member's row actually carries DISTR_Q (mode 1), the
     * reactive-limit kernel's {@code if (mode[ridx] != pvForm) return;} dropped every member before it ever
     * looked at q: rte6515's base pinned 16 where OLF pinned 19 (VL-5997_1, VL-6003_1, VL-6003_2 missing,
     * each with a correct q of 1.921012507 pu against a 0.7536 limit), then diverged at outer pass 7.
     */
    public static GeneratorVoltageControl mainGenVc(LfBus bus) {
        return GpuAcRowIndexing.mainGenVc(bus);              // ONE definition, shared with the row indexing
    }

    static RmqPack remoteMemberQPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                     ExtractionColumns columns, GpuAcFamilyPacker.Packs family, boolean reactiveLimits) {
        List<int[]> out = new ArrayList<>();
        List<Double> w = new ArrayList<>();
        int[] vCol = columns.vCol();
        if (reactiveLimits) {
            for (LfBus bus : network.getBuses()) {
                boolean member = GpuAcRowIndexing.isRemoteGenVcControllerElement(bus);   // incl. a pilot that is a controller
                // OLF_GPU_RMQ_WHY=1: one line per REMOTE CONTROLLER bus saying whether it gets a diagnostic q
                // and, when it does not, which gate dropped it. A member with no q source is invisible to the
                // reactive-limit check — it can never pin, whatever its real q — which is how rte6515's base
                // pinned 16 where OLF pinned 19 and then diverged.
                if (member && System.getenv("OLF_GPU_RMQ_WHY") != null) {
                    System.err.printf("RMQ_WHY bus=%s vCol=%d minQ=%.6f maxQ=%.6f qRange=%b skipped=%s%n",
                            bus.getId(), vCol[bus.getNum()], bus.getMinQ(), bus.getMaxQ(),
                            bus.getMinQ() < bus.getMaxQ(),
                            vCol[bus.getNum()] < 0 ? "no-vCol" : !(bus.getMinQ() < bus.getMaxQ()) ? "empty-q-range" : "-");
                }
                if (!member || vCol[bus.getNum()] < 0 || !(bus.getMinQ() < bus.getMaxQ())) {
                    continue;                                // mirror the RL pack's remote-entry condition
                }
                // A pilot-self pilot whose Q is RE-ROUTED through a candidate table (qRowAlt = -2-g) already gets
                // its own injection at its own row from the diagnostic fill: patchFrozenGroupDests points the
                // group's frozen role-0 dest at the table's FALLBACK row, which for such a pilot IS its own row.
                // Adding the rmq twin on top counted it TWICE. Measured on rte6515 VL-6112_0 with the rotation
                // property on: q read 0.603080467 where OLF has 0.910540234, and
                // 1.218 (loadQ) + 2 * (0.910540234 - 1.218) = 0.603080 — the doubled injection, to six digits.
                // The reactive-limit check then never saw the pilot beyond its limit and never pinned it, which
                // is what kept the rotation path from ever reaching the pinned arrangement it exists for.
                if (PilotSelfPacker.isPilotSelfEntry(bus) && columns.qRowAlt()[bus.getNum()] <= -2) {
                    continue;
                }
                var vc = bus.getGeneratorVoltageControl().orElseThrow();
                if (vCol[vc.getControlledBus().getNum()] < 0) {
                    continue;
                }
                // The row the reactive-limit check READS (native rlQrow): for a relocated child that is the
                // parent coupler's DUMMY_Q row, not its own V row, which now holds the coupler's ZERO_V. Filling
                // the own row left the check reading ZERO for every relocated member, so it could never pin one —
                // measured on rte6515's full busbar scope: the device's base stopped after its first
                // ReactiveLimits pass where OLF takes three, and came back MAX_ITERATION_REACHED.
                int row = GpuAcRowIndexing.relocatedQRow(bus.getNum(), columns.ziParentEdge(),
                        columns.dummyQCol(), vCol);
                if (bus.getShunt().isPresent()) {
                    out.add(new int[] {row, family.shElemOf()[bus.getNum()], 0, 1});
                    w.add(1.0);
                }
                for (LfBranch br : bus.getBranches()) {
                    if (br.isDisabled()) {
                        continue;
                    }
                    int side = br.getBus1() == bus ? 0 : 1;
                    int e = family.cbElemOf()[br.getNum()];
                    if (e >= 0) {
                        out.add(new int[] {row, e, side, 0});
                        w.add(1.0);
                        continue;
                    }
                    int oe = family.obElemOf()[br.getNum()];
                    if (oe >= 0) {
                        out.add(new int[] {row, oe, side, 2});
                        w.add(1.0);
                        continue;
                    }
                    // every coupler, not just the base tree's: a coupler out of it is pinned at dummyQ = 0 and adds
                    // nothing, and one a contingency brings into OLF's tree carries this member's reactive flow
                    if (br.isZeroImpedance(LoadFlowModel.AC)) {
                        int dqCol = columns.dummyQCol()[br.getNum()];
                        if (dqCol >= 0) {
                            out.add(new int[] {row, dqCol, side, 3});
                            w.add(side == 0 ? 1.0 : -1.0);
                        }
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
        return new RmqPack(row, elem, side, kind, GpuAcArrays.toDoubleArray(w));
    }
}
