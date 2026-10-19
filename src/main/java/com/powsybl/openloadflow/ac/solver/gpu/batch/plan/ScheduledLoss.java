/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;

import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;

import java.util.ArrayList;
import java.util.List;

/**
 * The active power a contingency takes out of the balance (the scheduled loss OLF distributes before the
 * post-contingency solve), and the HVDC stations an island resets.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class ScheduledLoss {

    private ScheduledLoss() {
    }

    /**
     * {@code OLF_GPU_LOSS_TRACE=<contingency id>}: the device's SCHEDULED active power loss and the islanded
     * bus set it was computed from. The formula mirrors {@code LfContingency}'s
     * {@code disconnectedGenerationActivePower - disconnectedLoadActivePower} term for term, but only if this
     * islanded set equals OLF's {@code disabledNetwork.getBuses()} — an islanded bus contributes its WHOLE
     * injection and is skipped in the named lost load/generator sums, so a set that differs by one bus moves
     * the loss by that bus's (generation - load). Compare against the {@code disabled buses:} line that
     * {@code OLF_GPU_EQ_DUMP_CTG} prints.
     */
    static void traceScheduledLoss(BatchContingency c, List<LfBus> islandedBuses, double loss) {
        String want = System.getenv("OLF_GPU_LOSS_TRACE");
        if (want == null || !want.equals(c.contingencyId())) {
            return;
        }
        System.err.printf("LOSS_TRACE ctg=%s deviceScheduledLoss=%.9f pu islanded=%s lostGens=%s lostLoads=%s%n",
                c.contingencyId(), loss,
                islandedBuses.stream().map(LfBus::getId).sorted().toList(),
                c.lostGenerators().stream().map(com.powsybl.openloadflow.network.LfGenerator::getId).sorted()
                        .toList(),
                c.lostLoads().stream().map(ll -> ll.bus() == null ? "?" : ll.bus().getId()).sorted().toList());
    }

    /** The surviving converter station's injection when this contingency islands the other endpoint of its link.
     *  AC emulation: OLF's station reports targetP 0 once a bus of its link is disabled, so a saturated or frozen
     *  link's survivor always loses its injection (a linear one has none). SETPOINT links follow
     *  {@code LfContingency.processHvdcsWithoutPower}: when the islanded converter bus is ISOLATED in OLF's sense
     *  ({@code Networks.isIsolatedBusForHvdc}: a one-bus component after the cut, no non-fictitious load, no
     *  generator other than converter stations) and the other endpoint survives, OLF zeroes BOTH stations' targetP.
     *  The islanded station leaves with its frozen bus (its scheduled P is already out of the loss,
     *  {@link #scheduledActivePowerLoss} mirrors {@code getBusLostGenerationTargetP}); the SURVIVING station's
     *  injection is what changes on the live side, and it rides the plain injection channel as generation
     *  removed - a negative load loss on its bus, composed with whatever else that bus loses. A link whose islanded
     *  bus is NOT isolated (the island also holds a load or a generator, or more than one bus) keeps its power in
     *  OLF: the survivor goes on injecting with its counterpart gone, and the distributed slack absorbs it on both
     *  arms - nothing to do here. network.xiidm EWVZ/EWWB/ZJI/ZJJ: a section loss that opens the station's own
     *  breaker (a converter station is not an end node for the tripping), leaving EWVX_5 / EWVX_4 / ZJH_8 / ZJH_9
     *  alone with their station - refused as ISLANDS_REMOTE_VC_MEMBER until this. */
    static List<LoadLoss> hvdcStationResets(List<LfBus> islandedBuses, BatchContingency c) {
        List<LoadLoss> out = new ArrayList<>();
        java.util.Set<LfBus> islanded = new java.util.HashSet<>(islandedBuses);
        for (LfBus bus : islandedBuses) {
            for (com.powsybl.openloadflow.network.LfHvdc h : bus.getHvdcs()) {
                // AC emulation: a SATURATED or FROZEN link carries its power as station targetP (a LINEAR one as an
                // equation term, masked with the link), and LfVscConverterStationImpl.getTargetP returns 0 for an
                // AC-emulation station as soon as either endpoint bus is disabled - isolated or not. A setpoint link
                // loses its power only through processHvdcsWithoutPower, i.e. when the islanded bus is isolated.
                LfBus other = h.getBus1() == bus ? h.getBus2() : h.getBus1();
                if (other == null || islanded.contains(other) || other.isDisabled()
                        || !h.isAcEmulation() && !isolatedForHvdc(bus, c)) {
                    continue;
                }
                com.powsybl.openloadflow.network.LfGenerator survivor = h.getBus1() == other
                        ? h.getConverterStation1() : h.getConverterStation2();
                double p = survivor.getTargetP();
                if (p != 0) {
                    out.add(new LoadLoss(other, -p, 0));   // generation removed: the row target drops by p
                }
                if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                    System.err.printf("QUALIFY_TRACE hvdc %s without power in %s: station %s on %s targetP %.6f -> 0%n",
                            h.getId(), c.contingencyId(), survivor.getId(), other.getId(), p);
                }
            }
        }
        return out;
    }

    /** {@code Networks.isIsolatedBusForHvdc} on the post-contingency connectivity: OLF removes every branch of the
     *  contingency from the graph (closed outages, opened couplers, side-opened branches alike) and asks whether the
     *  bus is alone in its component. */
    static boolean isolatedForHvdc(LfBus bus, BatchContingency c) {
        if (bus.getNonFictitiousLoadTargetP() != 0.0
                || bus.getGenerators().stream().anyMatch(com.powsybl.openloadflow.network.impl.LfGeneratorImpl.class::isInstance)) {
            return false;
        }
        java.util.Set<LfBranch> cut = new java.util.HashSet<>(c.outagedBranches());
        cut.addAll(c.outagedZiBranches());
        for (BatchContingency.SideOpen so : c.sideOpenBranches()) {
            cut.add(so.branch());
        }
        for (LfBranch br : bus.getBranches()) {
            if (!br.isDisabled() && br.isConnectedAtBothSides() && !cut.contains(br)) {
                return false;                                // a live branch: the component is bigger than the bus
            }
        }
        return true;
    }

    /** The contingency's SCHEDULED active power loss (pu): mirror of LfContingency's
     *  disconnectedGenerationActivePower - disconnectedLoadActivePower, incl. islanded buses'
     *  injections (with the same non-AC-emulation HVDC generation exclusion). */
    static double scheduledActivePowerLoss(BatchContingency c, List<LfBus> islandedBuses) {
        double loss = 0;
        // An ISLANDED bus contributes its WHOLE injection below (generation minus load), so anything
        // the contingency also names individually on that bus must not be added again. A busbar
        // outage does exactly that: the propagation names the bus's load as a lost load AND the
        // connectivity pass finds the bus islanded, and charging both doubled the scheduled loss —
        // rte6515 BUS-1004_BBS pre-distributed -1.110 pu for a 0.555 pu load. The device's DS loop
        // then claws most of it back during the solve and lands at a different point inside the
        // slack band, which is why the error was a fixed ~0.13% of the lost load rather than the
        // whole of it. A plain load contingency islands nothing, so loads alone were never wrong.
        java.util.Set<Integer> islandedNums = new java.util.HashSet<>();
        for (LfBus bus : islandedBuses) {
            islandedNums.add(bus.getNum());
        }
        for (com.powsybl.openloadflow.network.LfGenerator g : c.lostGenerators()) {
            if (g.getBus() != null && islandedNums.contains(g.getBus().getNum())) {
                continue;
            }
            loss += g.getTargetP();
        }
        for (LoadLoss ll : c.lostLoads()) {
            if (ll.bus() != null && islandedNums.contains(ll.bus().getNum())) {
                continue;
            }
            loss -= ll.p();
        }
        for (LfBus bus : islandedBuses) {
            double gen = bus.getGenerationTargetP();
            gen -= bus.getHvdcs().stream()
                    .filter(h -> !h.isAcEmulation())
                    .map(h -> h.getBus1() == bus ? h.getConverterStation1() : h.getConverterStation2())
                    .mapToDouble(com.powsybl.openloadflow.network.LfGenerator::getTargetP)
                    .sum();
            loss += gen - bus.getLoadTargetP();
        }
        return loss;
    }
}
