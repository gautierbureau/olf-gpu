/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;

import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfShunt;

import java.util.List;
import java.util.Objects;

/**
 * A contingency to analyze on the device: exactly one of — 1–2 branch outages (N-1 / N-2, disabled in
 * the per-scenario element mask), a single generator loss ({@code lostGenerators}, applied as a
 * per-scenario target override), or one+ fixed-shunt disconnections ({@code outagedShunts}, disabled in
 * the element mask like branches).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record BatchContingency(String contingencyId, List<LfBranch> outagedBranches, List<LfGenerator> lostGenerators,
                               List<LoadLoss> lostLoads, List<ShuntShift> shuntShifts, List<TapR1Override> tapOverrides,
                               List<LfShunt> outagedShunts, List<LfBranch> reclosedBranches,
                               List<LfBranch> outagedZeroImpedanceBranches, List<LfHvdc> outagedHvdcs,
                               List<SideOpen> sideOpenBranches,
                               /** The zero-impedance branches OLF has OUT of its spanning tree when it reaches
                                *  this contingency. OLF's tree is history-dependent within one security
                                *  analysis: removeBranchAndTryToSplit marks a disabled tree edge non-tree and
                                *  re-elects, and addBranch never restores it, so the non-tree edge of each
                                *  zero-impedance network is simply the last one disabled - and it sticks for
                                *  every later contingency. The device extracts its structure before any
                                *  contingency runs, so without this it solves every scenario on the BASE tree
                                *  while OLF has moved on (network.xiidm: LMK at extraction, LMO, then LMQ for
                                *  the remaining 2938 of 3528). Empty = the base tree. */
                               List<LfBranch> stickyNonSpanningTreeBranches,
                               /** Strategy pairs: the generators a GeneratorAction moves. OLF's LfGeneratorAction sets
                                *  their targetP AND initial targetP, so the pair's slack distribution starts each of
                                *  them from the action's value: the parent's target plus the change (relative), or the
                                *  value itself over whatever the parent's distribution gave it (absolute). */
                               List<GeneratorMove> generatorActions,
                               /** Strategy pairs: the AC-emulation links an HvdcAction switches to fixed set points (OLF's
                                *  LfHvdcAction: each station's target becomes the flow at the strategy's starting state). */
                               List<LfHvdc> acEmulationOffHvdcs) {

    /** Separator inside a (contingency, strategy) pair's BatchContingency id — pairs are solved in a
     *  SECOND phase, WARM-CONTINUED from their parent contingency's state and reactive-limits pinning. */
    public static final char STRATEGY_KEY_SEP = '\u0001';

    @SuppressWarnings("java:S107")
    public BatchContingency(String contingencyId, List<LfBranch> outagedBranches, List<LfGenerator> lostGenerators,
                            List<LoadLoss> lostLoads, List<ShuntShift> shuntShifts, List<TapR1Override> tapOverrides,
                            List<LfShunt> outagedShunts, List<LfBranch> reclosedBranches,
                            List<LfBranch> outagedZeroImpedanceBranches, List<LfHvdc> outagedHvdcs, List<SideOpen> sideOpenBranches,
                            List<LfBranch> stickyNonSpanningTreeBranches, List<GeneratorMove> generatorActions) {
        this(contingencyId, outagedBranches, lostGenerators, lostLoads, shuntShifts, tapOverrides, outagedShunts,
                reclosedBranches, outagedZeroImpedanceBranches, outagedHvdcs, sideOpenBranches, stickyNonSpanningTreeBranches, generatorActions, List.of());
    }

    /** 12-arg compatibility constructor (no absolute generator target). */
    @SuppressWarnings("java:S107")
    public BatchContingency(String contingencyId, List<LfBranch> outagedBranches, List<LfGenerator> lostGenerators,
                            List<LoadLoss> lostLoads, List<ShuntShift> shuntShifts, List<TapR1Override> tapOverrides,
                            List<LfShunt> outagedShunts, List<LfBranch> reclosedBranches,
                            List<LfBranch> outagedZeroImpedanceBranches, List<LfHvdc> outagedHvdcs,
                            List<SideOpen> sideOpenBranches, List<LfBranch> stickyNonSpanningTreeBranches) {
        this(contingencyId, outagedBranches, lostGenerators, lostLoads, shuntShifts, tapOverrides,
                outagedShunts, reclosedBranches, outagedZeroImpedanceBranches, outagedHvdcs, sideOpenBranches, stickyNonSpanningTreeBranches,
                List.of());
    }

    /** 11-arg compatibility constructor (base spanning tree). */
    @SuppressWarnings("java:S107")
    public BatchContingency(String contingencyId, List<LfBranch> outagedBranches, List<LfGenerator> lostGenerators,
                            List<LoadLoss> lostLoads, List<ShuntShift> shuntShifts, List<TapR1Override> tapOverrides,
                            List<LfShunt> outagedShunts, List<LfBranch> reclosedBranches,
                            List<LfBranch> outagedZeroImpedanceBranches, List<LfHvdc> outagedHvdcs,
                            List<SideOpen> sideOpenBranches) {
        this(contingencyId, outagedBranches, lostGenerators, lostLoads, shuntShifts, tapOverrides,
                outagedShunts, reclosedBranches, outagedZeroImpedanceBranches, outagedHvdcs, sideOpenBranches, List.of());
    }

    /** 10-arg compatibility constructor (no side-open branches). */
    @SuppressWarnings("java:S107")
    public BatchContingency(String contingencyId, List<LfBranch> outagedBranches, List<LfGenerator> lostGenerators,
                            List<LoadLoss> lostLoads, List<ShuntShift> shuntShifts, List<TapR1Override> tapOverrides,
                            List<LfShunt> outagedShunts, List<LfBranch> reclosedBranches,
                            List<LfBranch> outagedZeroImpedanceBranches, List<LfHvdc> outagedHvdcs) {
        this(contingencyId, outagedBranches, lostGenerators, lostLoads, shuntShifts, tapOverrides,
                outagedShunts, reclosedBranches, outagedZeroImpedanceBranches, outagedHvdcs, List.of());
    }

    public BatchContingency {
        Objects.requireNonNull(contingencyId);
        Objects.requireNonNull(outagedBranches);
        Objects.requireNonNull(lostGenerators);
        Objects.requireNonNull(lostLoads);
        Objects.requireNonNull(shuntShifts);
        Objects.requireNonNull(tapOverrides);
        Objects.requireNonNull(outagedShunts);
        Objects.requireNonNull(sideOpenBranches);
        Objects.requireNonNull(generatorActions);
        Objects.requireNonNull(acEmulationOffHvdcs);
        boolean isInjection = !lostGenerators.isEmpty() || !lostLoads.isEmpty() || !shuntShifts.isEmpty()
                || !tapOverrides.isEmpty() || !outagedHvdcs.isEmpty();
        boolean isShunt = !outagedShunts.isEmpty();
        // a PURE switch (zero-impedance) outage is a valid shape (it used to THROW and silently kill the precompute)
        boolean isBranch = !outagedBranches.isEmpty() || !sideOpenBranches.isEmpty()
                || !outagedZeroImpedanceBranches.isEmpty();
        // Batchable shapes: N branches (the element mask is per-element; the validator rejects
        // fragmenting sets), N generators, N loads, and branch+injection COMBOS (independent
        // mechanisms: element mask vs target/row-mode overrides); fixed shunts stay exclusive
        // (their DISTR_Q interplay is validated separately). The former <=2-branch cap was
        // validated-scope only — lifted 2026-09-08 (N-3/N-4 at 1e-12).
        // A strategy pair may compose to NOTHING: an action reclosing the contingency's only branch leaves the base
        // topology, solved from where the contingency left the network. It used to throw here and take the whole
        // precompute down with it (every contingency of the run then went to the CPU).
        boolean emptyPair = !isInjection && !isShunt && !isBranch && contingencyId.indexOf(BatchContingency.STRATEGY_KEY_SEP) >= 0;
        if (!emptyPair && !isInjection && !isShunt && !isBranch
                || isShunt && (isInjection || isBranch)) {
            throw new IllegalArgumentException(
                    "a batched contingency is branch(es) and/or injection loss(es), or fixed shunt(s) alone");
        }
        outagedBranches = List.copyOf(outagedBranches);
        lostGenerators = List.copyOf(lostGenerators);
        lostLoads = List.copyOf(lostLoads);
        shuntShifts = List.copyOf(shuntShifts);
        tapOverrides = List.copyOf(tapOverrides);
        outagedShunts = List.copyOf(outagedShunts);
        outagedHvdcs = List.copyOf(outagedHvdcs);
        sideOpenBranches = List.copyOf(sideOpenBranches);
        generatorActions = List.copyOf(generatorActions);
    }

    /** A branch outage (N-1 .. N-k). */
    public BatchContingency(String contingencyId, List<LfBranch> outagedBranches) {
        this(contingencyId, outagedBranches, List.of(), List.of(), List.of(), List.of(), List.of(), List.of(), List.of(), List.of());
    }

    /** Convenience for the common single-branch (N-1) case. */
    public BatchContingency(String contingencyId, LfBranch outagedBranch) {
        this(contingencyId, List.of(Objects.requireNonNull(outagedBranch)), List.of(), List.of(), List.of(), List.of(), List.of(), List.of(), List.of(), List.of());
    }

    /** One GeneratorAction of a strategy pair: {@code deltaP} (pu) is the change of target P from the BASE
     *  target, which is the whole move for an absolute action and the change added to the parent's target
     *  for a relative one. {@code change} is OLF's own GeneratorChange (pu): the new target of an absolute action,
     *  the amount added to the current target of a relative one. {@code participatesAt} is OLF's participation
     *  verdict at a target P (pu) once the action re-checks it (GpuGeneratorParticipation), null when the re-check
     *  leaves it as it was. */
    public record GeneratorMove(LfGenerator generator, double deltaP, boolean absolute, double change,
                                java.util.function.DoublePredicate participatesAt) {
        public GeneratorMove {
            Objects.requireNonNull(generator);
        }

        /** OLF's new target P (pu) from the generator's {@code currentTargetP} (pu) before the action. */
        public double newTargetP(double currentTargetP) {
            return absolute ? change : currentTargetP + change;
        }
    }

    /** The generators an absolute GeneratorAction sets: the pair does not re-anchor their bus on the parent. */
    public List<LfGenerator> absoluteTargetGenerators() {
        return generatorActions.stream().filter(GeneratorMove::absolute).map(GeneratorMove::generator).toList();
    }

    /** A branch opened on ONE side by the contingency (node-breaker propagation): the closed-branch
     *  element is masked and the pre-packed side-open OB view enabled — {@code side} is the OPEN
     *  side (a GpuTwoSides), i.e. the remaining flow terms sit at the OTHER bus. */
    public record SideOpen(LfBranch branch, int side) {
        public SideOpen {
            Objects.requireNonNull(branch);
            if (side != GpuTwoSides.ONE && side != GpuTwoSides.TWO) {
                throw new IllegalArgumentException("side must be GpuTwoSides.ONE or GpuTwoSides.TWO");
            }
        }
    }

    /** A combined branch + generator/load contingency. */
    public static BatchContingency branchesAndInjections(String contingencyId, List<LfBranch> branches,
                                                        List<LfGenerator> gens, List<LoadLoss> loads,
                                                        List<ShuntShift> shifts, List<TapR1Override> taps,
                                                        List<LfBranch> outagedZeroImpedanceBranch) {
        return new BatchContingency(contingencyId, branches, gens, loads, shifts, taps, List.of(), List.of(), outagedZeroImpedanceBranch, List.of());
    }

    /** A combined branch + injection + AC-emulation-HVDC contingency. */
    public static BatchContingency branchesInjectionsAndHvdcs(String contingencyId, List<LfBranch> branches,
                                                              List<LfGenerator> gens, List<LoadLoss> loads,
                                                              List<ShuntShift> shifts, List<TapR1Override> taps,
                                                              List<LfBranch> outagedZeroImpedanceBranch, List<LfHvdc> hvdcs) {
        return new BatchContingency(contingencyId, branches, gens, loads, shifts, taps, List.of(), List.of(), outagedZeroImpedanceBranch, hvdcs);
    }

    public static BatchContingency branchesAndInjections(String contingencyId, List<LfBranch> branches,
                                                         List<LfGenerator> gens, List<LoadLoss> loads,
                                                         List<ShuntShift> shifts, List<TapR1Override> taps) {
        return new BatchContingency(contingencyId, branches, gens, loads, shifts, taps, List.of(), List.of(), List.of(), List.of());
    }

    /** A generator loss. */
    public static BatchContingency generator(String contingencyId, LfGenerator lostGenerator) {
        return new BatchContingency(contingencyId, List.of(), List.of(Objects.requireNonNull(lostGenerator)), List.of(), List.of(), List.of(), List.of(), List.of(), List.of(), List.of());
    }

    /** A fixed-shunt disconnection (the shunt element is masked off, like a branch outage). */
    public static BatchContingency shunt(String contingencyId, LfShunt outagedShunt) {
        return new BatchContingency(contingencyId, List.of(), List.of(), List.of(), List.of(), List.of(), List.of(Objects.requireNonNull(outagedShunt)), List.of(), List.of(), List.of());
    }
}
