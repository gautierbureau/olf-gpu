/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.graph.GraphConnectivity;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Qualifies single-branch contingencies for the batched GPU N-1 path. The batched solver
 * runs plain Newton-Raphson over the FIXED superset pattern with the outaged branch's
 * contributions zeroed — this is only valid when the outage keeps the main synchronous
 * component intact (same buses, same slack/PV roles). A contingency that FRAGMENTS the
 * network (its branch is a bridge, so removing it islands buses from the main component)
 * cannot be represented as a mask ALONE: the islanded buses would be left with a
 * structurally singular sub-block. With {@code handleIslanding} (the full-loadflow SA
 * path) such an outage still QUALIFIES: the qualification carries its
 * {@code islandedBuses}, whose variable rows the solver FREEZES to identity at a clean
 * flat 1&#8736;0 (V=1 pu, phi=0) in that scenario's per-scenario row mode — the sub-block
 * stays non-singular and the main component solves correctly (see
 * GpuBatchedIslandingTest). Without it (the fixed-iteration benchmark mode) fragmenting
 * outages are rejected (loudly) to the sequential CPU path instead of being silently
 * mis-solved.
 *
 * <p>Qualification uses {@link LfNetwork#getConnectivity()}: for each closed-branch
 * contingency, a temporary edge removal reveals whether any vertex leaves the main
 * component. Open-branch contingencies (one side already disconnected) carry no flow on
 * the open side and cannot fragment the network, so they qualify trivially; a branch the
 * GPU element model does not contain is rejected (it cannot be masked).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchContingencyValidator {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchContingencyValidator.class);

    public enum Disqualification {
        FRAGMENTS_NETWORK,      // removing the branch islands buses (only when islanding handling is OFF)
        ISLANDS_SLACK,          // the outage islands the slack/reference bus (no on-GPU reference left)
        NOT_IN_GPU_MODEL        // the branch is not a closed/open branch the GPU pack contains
    }

    /**
     * The outcome of qualifying one contingency: either qualified (batchable on the GPU), or
     * disqualified with a reason (route to the CPU path). A qualified contingency that fragments the
     * network carries its {@code islandedBuses} (empty for a non-fragmenting outage) — those buses are
     * frozen to identity (held at base) in that scenario's per-scenario row mode, so the islanded
     * sub-block stays non-singular and the main component solves correctly.
     */
    public record Qualification(BatchContingency contingency,
                                boolean qualified, Disqualification reason, List<LfBus> islandedBuses) {

        static Qualification ok(BatchContingency c) {
            return new Qualification(c, true, null, List.of());
        }

        static Qualification okIslanding(BatchContingency c, List<LfBus> islandedBuses) {
            return new Qualification(c, true, null, List.copyOf(islandedBuses));
        }

        static Qualification rejected(BatchContingency c, Disqualification reason) {
            return rejected(c, reason, "");
        }

        static Qualification rejected(BatchContingency c, Disqualification reason, String detail) {
            com.powsybl.openloadflow.sa.gpu.GpuBatchScopeCensus.record("VALIDATOR:" + reason);
            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {   // test logback is broken — stderr lever
                System.err.println("QUALIFY_TRACE rejected " + c.contingencyId() + " reason=" + reason
                        + (detail.isEmpty() ? "" : " (" + detail + ")"));
            }
            return new Qualification(c, false, reason, List.of());
        }
    }

    private GpuBatchContingencyValidator() {
    }

    /**
     * Qualify each contingency for the batched GPU path.
     *
     * @param network          the base-case AC network
     * @param contingencies    single-branch contingencies to qualify
     * @param elementMapper    maps an {@link LfBranch} to its GPU element index (-1 = absent)
     * @param handleIslanding  when true (full-loadflow mode), a fragmenting outage is QUALIFIED and
     *                         carries its islanded-bus set (frozen per scenario); when false
     *                         (fixed-iter benchmark mode) it is rejected to CPU
     * @return one {@link Qualification} per input contingency, in order
     */
    public static List<Qualification> qualify(LfNetwork network,
                                              List<BatchContingency> contingencies,
                                              GpuBatchElementMapper elementMapper,
                                              boolean handleIslanding) {
        Objects.requireNonNull(network);
        Objects.requireNonNull(contingencies);
        Objects.requireNonNull(elementMapper);

        GraphConnectivity<LfBus, LfBranch> connectivity = network.getConnectivity();
        Set<LfBus> slackBuses = network.getSynchronousNetworks().stream()
                .flatMap(sn -> sn.getSlackBuses().stream())
                .collect(Collectors.toSet());
        return contingencies.stream()
                .map(c -> qualifyOne(network, connectivity, slackBuses, c, elementMapper, handleIslanding))
                .toList();
    }

    private static Qualification qualifyOne(LfNetwork network, GraphConnectivity<LfBus, LfBranch> connectivity,
                                           Set<LfBus> slackBuses,
                                           BatchContingency contingency,
                                           GpuBatchElementMapper elementMapper,
                                           boolean handleIslanding) {
        List<LfBranch> branches = contingency.outagedBranches();

        // The GPU disabled mask can only mask branches the element pack contains — the SIDE-OPENED
        // ones included (their closed element is masked; the pre-packed view is enabled by index).
        for (LfBranch branch : branches) {
            if (elementMapper.getElementIndex(branch) < 0) {
                LOGGER.debug("Contingency {} rejected: branch not in GPU element model", contingency.contingencyId());
                return Qualification.rejected(contingency, Disqualification.NOT_IN_GPU_MODEL, "branch " + branch.getId());
            }
        }
        for (BatchContingency.SideOpen so : contingency.sideOpenBranches()) {
            if (elementMapper.getElementIndex(so.branch()) < 0) {
                LOGGER.debug("Contingency {} rejected: side-open branch not in GPU element model", contingency.contingencyId());
                return Qualification.rejected(contingency, Disqualification.NOT_IN_GPU_MODEL, "side-open " + so.branch().getId());
            }
        }

        // Only the closed branches (both sides) are connectivity edges; an open-ended branch carries no flow
        // on its open side and cannot fragment the network. Remove every closed outaged branch as a temporary
        // edge (N-2 removes both at once) and check whether any vertex leaves the main component.
        List<LfBranch> closed = new java.util.ArrayList<>(branches.stream()
                .filter(b -> b.getBus1() != null && b.getBus2() != null).toList());
        // A branch opened on ONE side carries no through-flow either: its edge leaves the graph the
        // same as a full outage (the surviving terminal keeps only the equivalent-shunt injection).
        contingency.sideOpenBranches().forEach(so -> closed.add(so.branch()));
        // Outaged ZERO-IMPEDANCE branches (node-breaker propagation opens the target's breakers
        // alongside it) are connectivity edges too: they must take part in the islanding analysis,
        // and the DEAD-STUB shape (one side left islanded) is the one the frozen-rows machinery
        // reproduces exactly. A switch whose two sides BOTH stay in the main component would
        // require re-picking the zero-impedance group's spanning tree per scenario — CPU.
        closed.addAll(contingency.outagedZiBranches());
        if (closed.isEmpty()) {
            return Qualification.ok(contingency);
        }
        connectivity.startTemporaryChanges();
        try {
            for (LfBranch branch : closed) {
                connectivity.removeEdge(branch);
            }
            // When the outage isolates the slack itself, CPU does NOT keep the slack's component as the
            // main one: PropagatedContingency RELOCATES the slack ("leads to an isolated slack bus:
            // relocate slack bus inside main component"), excluding every bus outside the largest
            // connected component from slack eligibility and re-picking a slack there. The batched pack
            // anchors its targets and the distributed-slack participation on the BASE slack row, so a
            // relocated slack is not representable as a per-scenario mask — route it to CPU. Detecting it
            // here is what stops the dead-stub case from being silently mis-solved: anchoring on the
            // isolated slack reports the LIVE side as removed (measured: {BBS2, BBS3}), which would
            // freeze the whole energized network instead of the stub.
            LfBus slack = network.getSynchronousNetworks().getFirst().getSlackBuses().getFirst();
            // Component 0 IS the largest one (GraphConnectivity numbering, and what getLargestConnectedComponent
            // returns), and asking for the slack's number looks only through the small components split off by
            // the outage, where getLargestConnectedComponent() copies every bus of the network into a new set.
            if (connectivity.getComponentNumber(slack) != 0) {
                // Isolating the slack necessarily FRAGMENTS the network, so in fixed-iter mode (islanding
                // handling off) that is the reason to report — it is the mode's primary rejection and what
                // its contract documents. Only when islanding IS handled does the distinction matter: there
                // a fragmenting outage would normally qualify with frozen islanded buses, and it is
                // specifically the relocated slack that disqualifies it.
                Disqualification reason = handleIslanding
                        ? Disqualification.ISLANDS_SLACK : Disqualification.FRAGMENTS_NETWORK;
                LOGGER.debug("Contingency {} rejected ({}): outage isolates the slack bus (CPU relocates the slack)",
                        contingency.contingencyId(), reason);
                return Qualification.rejected(contingency, reason);
            }
            // The main component must be ANCHORED on the slack before querying, exactly as
            // PropagatedContingency does (setMainComponentVertex + getVerticesRemovedFromMainComponent).
            connectivity.setMainComponentVertex(slack);
            Set<LfBus> islanded = connectivity.getVerticesRemovedFromMainComponent();
            if (System.getenv("OLF_GPU_QUALIFY_TRACE") != null) {
                // What this pass CUT and what it FOUND. The CPU derives its lost buses from the same
                // kind of pass on its own per-contingency topology; when the two disagree on a bus,
                // the edge lists are the only way to see whether the graphs differ or the cuts do.
                System.err.printf("QUALIFY_TRACE islanding %s: cut=%s -> islanded=%s%n",
                        contingency.contingencyId(),
                        closed.stream().map(b -> b.getId() + "(" + (b.getBus1() == null ? "-" : b.getBus1().getId())
                                + "," + (b.getBus2() == null ? "-" : b.getBus2().getId()) + ")").toList(),
                        islanded.stream().map(LfBus::getId).toList());
            }
            for (LfBranch zi : contingency.outagedZiBranches()) {
                if (!islanded.contains(zi.getBus1()) && !islanded.contains(zi.getBus2())) {
                    LOGGER.debug("Contingency {} rejected: switch outage {} leaves both sides energized "
                            + "(zero-impedance spanning tree would change)", contingency.contingencyId(), zi.getId());
                    return Qualification.rejected(contingency, Disqualification.NOT_IN_GPU_MODEL, "zi both-sides-energized " + zi.getId());
                }
            }
            if (islanded.isEmpty()) {
                return Qualification.ok(contingency);
            }
            if (!handleIslanding) {                          // fixed-iter mode: route to CPU
                LOGGER.debug("Contingency {} rejected: outage islands {} bus(es) (islanding handling off)",
                        contingency.contingencyId(), islanded.size());
                return Qualification.rejected(contingency, Disqualification.FRAGMENTS_NETWORK);
            }
            // The main component must keep a slack/reference: if the outage islands the slack itself,
            // the remaining system has no on-GPU reference — route that one to the CPU path.
            for (LfBus b : islanded) {
                if (slackBuses.contains(b)) {
                    LOGGER.debug("Contingency {} rejected: outage islands the slack bus", contingency.contingencyId());
                    return Qualification.rejected(contingency, Disqualification.ISLANDS_SLACK);
                }
            }
            LOGGER.debug("Contingency {} qualified with {} islanded bus(es) (frozen per scenario)",
                    contingency.contingencyId(), islanded.size());
            return Qualification.okIslanding(contingency, List.copyOf(islanded));
        } finally {
            connectivity.undoTemporaryChanges();
        }
    }
}
