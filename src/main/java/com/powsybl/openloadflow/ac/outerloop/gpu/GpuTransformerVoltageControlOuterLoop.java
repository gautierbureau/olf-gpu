/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSkips;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuDiscreteControlMemberState;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuPiModelColumn;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTransformerVoltageControlMode;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfVscConverterStation;
import com.powsybl.openloadflow.network.PiModel;
import com.powsybl.openloadflow.network.PiModelArray;
import com.powsybl.openloadflow.network.TransformerVoltageControl;
import com.powsybl.openloadflow.network.VoltageControl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * The per-scenario transformer voltage-control orchestration for the batched N-1 path: the stage-1
 * scope check, the controller pack built by toggling each control on the shared equation system and
 * diffing the activation, the discrete tap-table snapshot (reflection on PiModelArray), and the
 * Java orchestrator that ports TransformerVoltageControlOuterLoop (AFTER_GENERATOR mode) as repeated
 * batched device solves over one chunk (INITIAL deadband enable, CONTROL clamp, round-to-tap, final).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuTransformerVoltageControlOuterLoop {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuTransformerVoltageControlOuterLoop.class);

    private GpuTransformerVoltageControlOuterLoop() {
    }

    /** One member transformer of a (possibly shared) voltage control: rho row (== r1 col,
     *  self-paired), continuous r1 range, discrete tap r1 values, element index (per-scenario
     *  outage check), and the row-mode/target deltas that turn THIS member's control OFF while the
     *  rest of the group stays ON (a clamp: OLF's roundR1ToExtremeTapPosition disables the member;
     *  coefficients do NOT renormalize — the DISTR_RHO cross terms to a control-disabled member
     *  stay). {@code clampNeedsCpu}: clamping this member re-picks the group's FREE role (its
     *  off-delta touches other members' rows) — per-scenario CPU route, out of stage-1 scope. */
    public record Member(int rhoRow, double minR1, double maxR1, double initialR1,
                  double[] taps, double[][] tapRows, int elemIndex,
                  int[] offDeltaRows, int[] offDeltaModes, double[] offDeltaTargets, boolean clampNeedsCpu,
                  int branchNum, int baseTapIdx) {
        double closestTap(double r1) {
            if (taps.length == 0) {
                return r1;                                   // no tap table: keep the continuous ratio
            }
            double best = taps[0];
            for (double t : taps) {
                if (Math.abs(t - r1) < Math.abs(best - r1)) {
                    best = t;
                }
            }
            return best;
        }

        /** The FULL 8-double pi row of the tap whose r1 is closest to {@code r1}, or null without a
         *  table. Rounding a tap on the CPU switches the branch's per-step R/X/G/B, not just rho —
         *  pinning rho alone left the base impedance in the fills (a DIFFERENT model: 2.4e-4 kV on a
         *  non-base tap, found via the shared-group member-outage test). */
        double[] closestTapRow(double r1) {
            if (tapRows.length == 0) {
                return null;
            }
            double[] best = tapRows[0];
            for (double[] t : tapRows) {
                if (Math.abs(t[6] - r1) < Math.abs(best[6] - r1)) {
                    best = t;
                }
            }
            return best;
        }
    }

    /** One voltage CONTROL (the group), precomputed for the per-scenario orchestrator:
     *  controlled-bus V col, target/deadband, the row-mode/target DELTAS between control-off (base)
     *  and ALL-members-on activations (plan-group-member-dynamics A3 — captured by toggling the
     *  WHOLE group, never one member alone), and the members. */
    public record TransformerController(int vCol, double targetV, double halfDeadband,
                                 int[] deltaRows, int[] deltaModesOn, double[] deltaTargetsOn,
                                 Member[] members, boolean useInitialTapPosition,
                                 int[] deltaModesOff, double[] deltaTargetsOff) {
    }

    /** OLF's {@code TransformerRatioManager} SHARED behavior ({@code useInitialTapPosition=true}),
     *  per scenario over the group's SURVIVING members (the CPU builds it at the INITIAL step over
     *  the just-enabled controllers; outaged members never enable): group-AVERAGED clamp bounds and
     *  the normalized-headroom initial, plus the spread-preserving remap each member's continuous
     *  ratio goes through before rounding. A single-member group degenerates to the exact identity
     *  (initialR1 = own r1, bounds = own bounds), so applying it uniformly is behavior-neutral. */
    record SharedRatio(double minR1, double maxR1, double initialR1) {
        static SharedRatio of(Member[] members, boolean[] disabledByElem) {
            double sumMin = 0;
            double sumMax = 0;
            double a = 0;
            double b = 0;
            int count = 0;
            for (Member m : members) {
                if (disabledByElem[m.elemIndex()]) {
                    continue;
                }
                sumMin += m.minR1();
                sumMax += m.maxR1();
                if (m.maxR1() != m.minR1()) {
                    a += (m.maxR1() - m.initialR1()) / (m.maxR1() - m.minR1());
                    b += (m.initialR1() - m.minR1()) / (m.maxR1() - m.minR1());
                } else {
                    a += 1;
                    b += 1;
                }
                count++;
            }
            double avgMin = sumMin / count;
            double avgMax = sumMax / count;
            return new SharedRatio(avgMin, avgMax, (avgMin * a + avgMax * b) / count);
        }

        /** OLF's {@code updateContinuousRatio}, verbatim: map the COMMON computed ratio to this
         *  member's individual one, preserving the initial spread (two-slope piecewise linear). */
        double remap(Member m, double computedR1) {
            return computedR1 >= initialR1
                    ? m.initialR1() + (computedR1 - initialR1) * (m.maxR1() - m.initialR1()) / (maxR1 - initialR1)
                    : m.initialR1() - (initialR1 - computedR1) * (m.initialR1() - m.minR1()) / (initialR1 - minR1);
        }
    }

    /** Stage-1 scope check: the CPU loop's GeneratorVoltageControlManager disables generator
     *  voltage controls at buses under the max transformer-controlled nominal voltage. Porting
     *  the frozen-Q flip is stage 2 — until then a network that would NEED it must stay on the
     *  CPU path (the provider's guard calls this). */

    public static boolean transformerControlSupported(LfNetwork network) {
        double maxControlledNominalV = network.<LfBranch>getControllerElements(com.powsybl.openloadflow.network.VoltageControl.Type.TRANSFORMER).stream()
                .flatMap(b -> b.getVoltageControl().stream())
                .map(vc -> vc.getControlledBus().getNominalV())
                .mapToDouble(Double::doubleValue).max().orElse(-1);
        if (maxControlledNominalV < 0) {
            return true;                                     // no transformer controls at all: loop is a no-op
        }
        boolean genUnderThreshold = network.getControlledBuses(com.powsybl.openloadflow.network.VoltageControl.Type.GENERATOR).stream()
                .anyMatch(bus -> bus.getNominalV() <= maxControlledNominalV);
        if (genUnderThreshold) {
            LOGGER.info("Transformer control orchestration out of stage-1 scope: generator voltage controls "
                    + "exist under the transformer-controlled nominal voltage ({} kV) — CPU fallback", maxControlledNominalV);
        }
        return !genUnderThreshold;
    }

    /** OLF's GeneratorVoltageControlManager, static part: the min nominal voltage limit (the override,
     *  or the max nominal voltage over the transformer-controlled buses whose controllers all connect two
     *  DIFFERENT voltage levels — Double.MIN_VALUE when none). */
    static double minNominalVoltageLimit(LfNetwork network, double override) {
        if (override >= 0) {
            return override;
        }
        double limit = Double.MIN_VALUE;
        for (LfBus bus : network.getBuses()) {
            if (bus.isDisabled() || !bus.isTransformerVoltageControlled()) {
                continue;
            }
            TransformerVoltageControl tvc = bus.getTransformerVoltageControl().orElse(null);
            if (tvc == null) {
                continue;
            }
            boolean valid = true;
            for (LfBranch branch : tvc.getControllerElements()) {
                if (!branch.isConnectedAtBothSides() || branch.getBus1().getNominalV() == branch.getBus2().getNominalV()) {
                    valid = false;
                }
            }
            if (valid) {
                limit = Math.max(limit, bus.getNominalV());
            }
        }
        return limit;
    }

    /** OLF's GeneratorVoltageControlManager.hasStepUpTransformers, verbatim. */
    static boolean hasStepUpTransformers(LfBus bus, double limit) {
        if (bus.getGenerators().stream().anyMatch(LfVscConverterStation.class::isInstance)) {
            return true;
        }
        if (bus.getBranches().stream().anyMatch(b -> b.getVoltageControl().isPresent())) {
            return false;
        }
        double startingNominalVoltage = bus.getNominalV();
        double minConnectedVoltageLevel = bus.getBranches().stream()
                .filter(LfBranch::isConnectedAtBothSides)
                .mapToDouble(b -> Math.max(b.getBus1().getNominalV(), b.getBus2().getNominalV()))
                .min()
                .orElse(-1);
        return minConnectedVoltageLevel > startingNominalVoltage && minConnectedVoltageLevel > limit;
    }

    /** The generator voltage controls OLF's manager would freeze under the transformer-controlled level:
     *  the controller buses (in merged order) of every generator-controlled bus at or under the limit that
     *  have no step-up exemption. OLF freezes a REMOTE controller exactly like a local one, and so does the
     *  kernel — through the reactive-limit remote PIN (row mode 4) rather than a V identity flip — as long
     *  as the member has a reactive-limit ENTRY, which is what carries the limit type the DISTR_Q key
     *  renormalization reads. A remote candidate without one (equal Q limits, no pilot column) has no slot
     *  to mark and still gates the run: null. */
    private static List<LfBus> freezeCandidates(LfNetwork network, double limit) {
        List<LfBus> out = new ArrayList<>();
        for (LfBus bus : network.getControlledBuses(VoltageControl.Type.GENERATOR)) {
            if (bus.getNominalV() > limit) {
                continue;
            }
            var vc = bus.getGeneratorVoltageControl().orElse(null);
            if (vc == null) {
                continue;
            }
            for (LfBus controller : vc.getMergedControllerElements()) {
                if (hasStepUpTransformers(controller, limit)) {
                    continue;
                }
                if (controller != bus && !remoteFreezeRepresentable(controller)) {
                    return null;                             // no reactive-limit entry to carry the type
                }
                out.add(controller);
            }
        }
        return out;
    }

    /** Can a REMOTE controller be frozen in-kernel? It is frozen as a reactive-limit remote PIN, and the
     *  limit type that marks it lives on its reactive-limit ENTRY — which the pack creates for a MAIN
     *  generator voltage control with distinct Q limits. This is the network-only half of the condition,
     *  so the whole-run gate can be decided before the extraction exists; {@link #buildFreezeTables}
     *  checks the entry really is there and fails loudly if the two ever disagree. */
    private static boolean remoteFreezeRepresentable(LfBus controller) {
        return controller.getGeneratorVoltageControl()
                .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                .isPresent() && controller.getMinQ() < controller.getMaxQ();
    }

    /** The reactive-limit pack entry of a bus, or −1. */
    private static int rlEntryOf(int[] rlBusNum, int busNum) {
        if (rlBusNum == null) {
            return -1;
        }
        for (int i = 0; i < rlBusNum.length; i++) {
            if (rlBusNum[i] == busNum) {
                return i;
            }
        }
        return -1;
    }

    /** True when the AFTER_GENERATOR loop's generator freeze can run IN-KERNEL on this network: every
     *  candidate is a LOCAL controller and every transformer controller has a tap table (the kernel
     *  phase owns the control). Complements {@link #transformerControlSupported} (no candidates at all). */
    public static boolean generatorFreezeSupported(LfNetwork network, double minNominalVOverride) {
        double limit = minNominalVoltageLimit(network, minNominalVOverride);
        if (freezeCandidates(network, limit) == null) {
            return false;
        }
        return network.<LfBranch>getControllerElements(VoltageControl.Type.TRANSFORMER).stream()
                .allMatch(b -> tapModels(b.getPiModel()).length > 0);
    }

    /**
     * The GeneratorVoltageControlManager + fixTransformerVoltageControls tables of the in-kernel phase.
     *
     * <p>{@code fixPvRow} is the UNIVERSE the connectivity fix looks at: the {@code BUS_V} row of every
     * generator-voltage-controlled bus, in a fixed order. A member's component is then a BITMASK over
     * that universe, {@code fixWords} ints per member, and the kernel asks which of those buses is
     * still PV in its own scenario.
     *
     * <p>Two masks, because OLF's component is not computable from one. It cuts the CONTINGENCY's
     * branches and the ENABLED controller branches; the device knows the contingency host-side but
     * learns the enabled set only inside the solve. So the host brackets it: {@code minMask} cuts every
     * controller branch (a SUBSET of OLF's component, since OLF cuts only the enabled ones) and
     * {@code maxMask} cuts none (a SUPERSET). A PV bus in the min mask means OLF sees one too — keep the
     * control. No PV bus in the max mask means OLF sees none either — disable it. In between, the answer
     * depends on which controllers this scenario enabled, and the scenario goes to the CPU.
     */
    public record FreezeTables(int[] vRow, double[] loadQ, int[] busNum, int[] rlEntry,
                               int[] fixPvRow, int fixWords, int[] minMask, int[] maxMask, int nMembers,
                               FixMaskBuilder builder) {
        public boolean hasCandidates() {
            return vRow.length > 0;
        }

        public boolean hasFix() {
            return fixPvRow.length > 0 && nMembers > 0;
        }
    }

    /** Build the freeze tables: per candidate its controlled-bus V row and load Q (OLF freezes the
     *  generation target at {@code q.eval()}, the NET injection — the row target is that minus the load);
     *  per member the V rows of the generator-controlled buses in its not-controlled side's component with
     *  every transformer controller edge removed (OLF removes the ENABLED ones and the contingency's
     *  branches — at the base neither exists yet; the kernel then asks which of those buses is still ON).
     *  Null when a candidate is a remote controller. */
    public static FreezeTables buildFreezeTables(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                 List<TransformerController> tcs, double minNominalVOverride,
                                                 int[] rlBusNum) {
        double limit = minNominalVoltageLimit(network, minNominalVOverride);
        List<LfBus> candidates = freezeCandidates(network, limit);
        if (candidates == null) {
            return null;
        }
        java.util.Map<Integer, Integer> busVRows = new java.util.HashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVRows.put(v.getElementNum(), v.getRow());
            }
        }
        int[] vRow = new int[candidates.size()];
        double[] loadQ = new double[candidates.size()];
        int[] busNum = new int[candidates.size()];
        int[] rlEntry = new int[candidates.size()];
        for (int c = 0; c < candidates.size(); c++) {
            LfBus bus = candidates.get(c);
            Integer row = busVRows.get(bus.getNum());
            if (row == null) {
                return null;
            }
            vRow[c] = row;
            loadQ[c] = bus.getLoadTargetQ();
            busNum[c] = bus.getNum();
            // REMOTE controller (the controlled bus is not this one): the kernel freezes it as a
            // reactive-limit remote pin and marks its ENTRY, so the DISTR_Q shares renormalize.
            boolean remote = bus.getGeneratorVoltageControl().map(c2 -> c2.getControlledBus() != bus).orElse(false);
            rlEntry[c] = remote ? rlEntryOf(rlBusNum, bus.getNum()) : -1;
            if (remote && rlEntry[c] < 0) {
                // The gate admitted it on the network-only condition but the pack has no entry, so
                // nothing would carry the frozen limit type and the group's DISTR_Q shares would not
                // renormalize. Silently skipping the freeze is the one outcome worth refusing.
                throw new com.powsybl.commons.PowsyblException("GPU batch: remote generator voltage controller '"
                        + bus.getId() + "' is a freeze candidate with no reactive-limit entry");
            }
        }
        int nMembers = 0;
        for (TransformerController tc : tcs) {
            nMembers += tc.members().length;
        }
        // The universe of the connectivity fix: every generator-voltage-CONTROLLED bus, in bus order.
        // OLF's test is isGeneratorVoltageControlled() && isGeneratorVoltageControlEnabled(); the first
        // half is structural and lives here, the second is the per-scenario row mode the kernel reads.
        List<LfBus> pvBuses = new ArrayList<>();
        List<Integer> pvRows = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            Integer row = bus.isGeneratorVoltageControlled() ? busVRows.get(bus.getNum()) : null;
            if (row != null) {
                pvBuses.add(bus);
                pvRows.add(row);
            }
        }
        int words = (pvBuses.size() + 31) / 32;
        int[] fixPvRow = pvRows.stream().mapToInt(Integer::intValue).toArray();
        int[] memberBranchNum = new int[nMembers];
        int mi = 0;
        for (TransformerController tc : tcs) {
            for (Member mem : tc.members()) {
                memberBranchNum[mi++] = mem.branchNum();
            }
        }
        FixMaskBuilder builder = new FixMaskBuilder(network, memberBranchNum, pvBuses, words);
        int[] minMask = new int[nMembers * words];
        int[] maxMask = new int[nMembers * words];
        builder.fill(List.of(), minMask, 0, maxMask, 0);       // base topology: no contingency branches
        return new FreezeTables(vRow, loadQ, busNum, rlEntry, fixPvRow, words, minMask, maxMask, nMembers, builder);
    }

    /**
     * Builds the two connectivity masks of {@link FreezeTables} for one scenario's topology.
     *
     * <p>Held per run and called per scenario, because OLF re-runs {@code fixTransformerVoltageControls}
     * on the post-contingency network: the base-topology masks are wrong the moment a contingency splits
     * a not-controlled side's component. The cost is two connectivity sessions per scenario, each
     * {@code O(pv buses + members)} — the component NUMBER of every PV bus is read once and members look
     * their own component's mask up, rather than walking a component per member.
     */
    public static final class FixMaskBuilder {

        private final LfNetwork network;
        private final int[] memberBranchNum;
        private final List<LfBus> pvBuses;
        private final int words;

        /** Members are named by BRANCH NUMBER, which is all the connectivity question needs — and lets a
         *  test build one from a bare network without the whole controller pack. */
        FixMaskBuilder(LfNetwork network, int[] memberBranchNum, List<LfBus> pvBuses, int words) {
            this.network = network;
            this.memberBranchNum = memberBranchNum;
            this.pvBuses = pvBuses;
            this.words = words;
        }

        public int words() {
            return words;
        }

        public int nMembers() {
            return memberBranchNum.length;
        }

        /** Fill {@code minMask} (every controller branch cut) and {@code maxMask} (none cut) for a
         *  scenario whose contingency opens {@code outaged}, each at its own offset. */
        public void fill(List<LfBranch> outaged, int[] minMask, int minOff, int[] maxMask, int maxOff) {
            session(outaged, true, minMask, minOff);
            session(outaged, false, maxMask, maxOff);
        }

        private void session(List<LfBranch> outaged, boolean cutControllers, int[] mask, int off) {
            var connectivity = network.getConnectivity();
            connectivity.startTemporaryChanges();
            try {
                java.util.Set<LfBranch> removed = new java.util.HashSet<>();
                for (LfBranch b : outaged) {
                    if (b.getBus1() != null && b.getBus2() != null && removed.add(b)) {
                        connectivity.removeEdge(b);
                    }
                }
                if (cutControllers) {
                    for (LfBranch b : network.getBranches()) {
                        if (!b.isDisabled() && b.isVoltageController()
                                && b.getBus1() != null && b.getBus2() != null && removed.add(b)) {
                            connectivity.removeEdge(b);
                        }
                    }
                }
                // component number -> the PV buses it holds, built once per session
                java.util.Map<Integer, int[]> byComponent = new java.util.HashMap<>();
                for (int i = 0; i < pvBuses.size(); i++) {
                    Integer comp = componentOf(connectivity, pvBuses.get(i));
                    if (comp != null) {
                        byComponent.computeIfAbsent(comp, k -> new int[words])[i >> 5] |= 1 << (i & 31);
                    }
                }
                int[] all = new int[words];                    // "every PV bus": OLF skips the check entirely
                for (int i = 0; i < pvBuses.size(); i++) {
                    all[i >> 5] |= 1 << (i & 31);
                }
                for (int m = 0; m < memberBranchNum.length; m++) {
                    LfBranch branch = network.getBranch(memberBranchNum[m]);
                    var vc = branch != null ? branch.getVoltageControl().orElse(null) : null;
                    int[] src;
                    if (vc == null) {
                        src = all;                             // defensive: OLF never disables such a member
                    } else {
                        LfBus notControlled = vc.getControlledBus() == branch.getBus1() ? branch.getBus2()
                                : vc.getControlledBus() == branch.getBus2() ? branch.getBus1() : null;
                        Integer comp = notControlled != null ? componentOf(connectivity, notControlled) : null;
                        src = comp == null ? all : byComponent.getOrDefault(comp, new int[words]);
                    }
                    System.arraycopy(src, 0, mask, off + m * words, words);
                }
            } finally {
                connectivity.undoTemporaryChanges();
            }
        }

        @SuppressWarnings("checkstyle:IllegalCatchError") // a bus outside the graph (disabled) has no component
        private static Integer componentOf(com.powsybl.openloadflow.graph.GraphConnectivity<LfBus, LfBranch> c, LfBus bus) {
            try {
                return c.getComponentNumber(bus);
            } catch (RuntimeException e) {
                return null;                                   // a bus outside the graph (disabled): no component
            }
        }
    }

    /** Install the freeze tables on a context whose discrete TVC tables {@link #uploadToDevice} just
     *  installed (AFTER_GENERATOR only). */
    public static void uploadFreeze(long handle, FreezeTables fz) {
        if (fz == null) {
            TransformerVoltageControlNative.setBatchDiscreteTvcFreeze(handle, null, null, null, null);
            return;
        }
        TransformerVoltageControlNative.setBatchDiscreteTvcFreeze(handle, fz.vRow(), fz.loadQ(), fz.rlEntry(), fz.fixPvRow());
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("TVD_FREEZE_UPLOAD nCandidates=%d fixPvBuses=%d members=%d%n",
                    fz.vRow().length, fz.fixPvRow().length, fz.nMembers());
        }
    }

    /**
     * Per-scenario connectivity masks for the chunk about to solve: OLF re-runs
     * {@code fixTransformerVoltageControls} on the POST-CONTINGENCY network, so each scenario gets its
     * own pair. Padding slots repeat the base-topology masks, which is what an unused slot would have
     * used anyway.
     */
    public static void uploadFixMasks(long handle, FreezeTables fz, List<BatchContingency> scenarios, int chunkSize) {
        if (fz == null || !fz.hasFix()) {
            TransformerVoltageControlNative.setBatchDiscreteTvcFixMasks(handle, null, null, 0);
            return;
        }
        int stride = fz.nMembers() * fz.fixWords();
        int[] minMask = new int[chunkSize * stride];
        int[] maxMask = new int[chunkSize * stride];
        for (int s = 0; s < chunkSize; s++) {
            if (s < scenarios.size()) {
                List<LfBranch> outaged = new ArrayList<>(scenarios.get(s).outagedBranches());
                outaged.addAll(scenarios.get(s).outagedZeroImpedanceBranches());
                fz.builder().fill(outaged, minMask, s * stride, maxMask, s * stride);
            } else {
                System.arraycopy(fz.minMask(), 0, minMask, s * stride, stride);
                System.arraycopy(fz.maxMask(), 0, maxMask, s * stride, stride);
            }
        }
        TransformerVoltageControlNative.setBatchDiscreteTvcFixMasks(handle, minMask, maxMask, fz.fixWords());
    }

    /** Give the freeze state back to the network (batch-of-one): a frozen candidate is
     *  {@code freezeGenerationTargetQAndDisableGeneratorVoltageControl(rowTarget + loadQ)} (OLF's
     *  generation target is the net injection), an unfrozen one whose network state still shows OUR
     *  freeze (control off, target frozen, no reactive-limit type) is re-enabled like
     *  {@code enableGeneratorVoltageControlsUnderMaxControlledNominalVoltage}. */
    public static void applyFreezeToNetwork(LfNetwork network, FreezeTables fz, double[] state) {
        if (fz == null || state == null) {
            return;
        }
        for (int c = 0; c < fz.vRow().length && c * 2 + 1 < state.length; c++) {
            LfBus bus = network.getBus(fz.busNum()[c]);
            if (bus == null) {
                continue;
            }
            boolean frozen = (int) state[c * 2] != 0;
            if (frozen) {
                if (bus.isGeneratorVoltageControlEnabled()) {
                    bus.freezeGenerationTargetQAndDisableGeneratorVoltageControl(state[c * 2 + 1] + fz.loadQ()[c]);
                }
            } else if (!bus.isGeneratorVoltageControlEnabled() && bus.isGenerationTargetQFrozen()
                    && bus.getQLimitType().isEmpty()) {
                bus.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(true);
            }
        }
    }

    /** Build the orchestrator's controller pack by TOGGLING each control on the shared equation
     *  system and diffing the activation — the ON recipe is exactly what the engine itself would
     *  produce, no hand-crafted modes. The network is restored before returning. */
    public static List<TransformerController> buildTransformerControllers(
            LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
            TargetVector<AcVariableType, AcEquationType> targetVector,
            GpuAcActivation baseActivation, double[] baseTargets, GpuAcElementMapper elementMapper,
            boolean useInitialTapPosition) {
        List<TransformerController> out = new ArrayList<>();
        java.util.Map<Integer, Integer> busVCols = new java.util.HashMap<>();
        java.util.Map<Integer, Integer> branchR1Cols = new java.util.HashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVCols.put(v.getElementNum(), v.getRow());
            } else if (v.getType() == AcVariableType.BRANCH_RHO1) {
                branchR1Cols.put(v.getElementNum(), v.getRow());
            }
        }
        for (LfBus controlled : network.getControlledBuses(com.powsybl.openloadflow.network.VoltageControl.Type.TRANSFORMER)) {
            var vc = controlled.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == com.powsybl.openloadflow.network.VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (vc == null) {
                continue;
            }
            List<LfBranch> memberBranches = vc.getMergedControllerElements().stream()
                    .filter(LfBranch::isConnectedAtBothSides)
                    .filter(b -> !b.isDisabled())
                    .toList();
            Integer vCol = busVCols.get(controlled.getNum());
            boolean representable = vCol != null && !memberBranches.isEmpty()
                    && memberBranches.stream().allMatch(b -> branchR1Cols.get(b.getNum()) != null
                            && elementMapper.getElementIndex(b) >= 0);
            if (!representable) {
                GpuBatchSkips.record(GpuBatchSkips.Slot.TVC, LOGGER,
                        "Transformer control of {} not representable on the GPU batch — skipped (results follow the base taps)",
                        controlled.getId());
                continue;
            }
            double halfDeadband = vc.getTargetDeadband().orElse(0.1 / controlled.getNominalV()) / 2;
            boolean[] wasEnabled = new boolean[memberBranches.size()];
            for (int m = 0; m < memberBranches.size(); m++) {
                wasEnabled[m] = memberBranches.get(m).isVoltageControlEnabled();
                memberBranches.get(m).setVoltageControlEnabled(false);   // the WHOLE group OFF: the control-off activation
            }
            GpuAcActivation offAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] offTargets = offAct.applyTargets(targetVector.getArray());
            for (int m = 0; m < memberBranches.size(); m++) {
                memberBranches.get(m).setVoltageControlEnabled(true);    // the WHOLE group ON (A3)
            }
            GpuAcActivation onAct = GpuAcActivation.fromEquationSystem(network, es);
            double[] onTargets = onAct.applyTargets(targetVector.getArray());
            // the ON/OFF deltas are diffed against each other, not against the caller's base: a base
            // extracted with the control already ON (the WITH_GENERATOR base load flow, whose network
            // OLF's initialize switched on) yields the same rows as a control-off base
            List<Integer> dRows = new ArrayList<>();
            for (int r = 0; r < baseTargets.length; r++) {
                if (onAct.rowMode()[r] != offAct.rowMode()[r] || onTargets[r] != offTargets[r]) {
                    dRows.add(r);
                }
            }
            int[] deltaRows = dRows.stream().mapToInt(Integer::intValue).toArray();
            int[] deltaModesOn = new int[deltaRows.length];
            double[] deltaTargetsOn = new double[deltaRows.length];
            int[] deltaModesOff = new int[deltaRows.length];
            double[] deltaTargetsOff = new double[deltaRows.length];
            for (int k = 0; k < deltaRows.length; k++) {
                deltaModesOn[k] = onAct.rowMode()[deltaRows[k]];
                deltaTargetsOn[k] = onTargets[deltaRows[k]];
                deltaModesOff[k] = offAct.rowMode()[deltaRows[k]];
                deltaTargetsOff[k] = offTargets[deltaRows[k]];
            }
            Member[] members = new Member[memberBranches.size()];
            for (int m = 0; m < memberBranches.size(); m++) {
                LfBranch branch = memberBranches.get(m);
                // this member's control OFF while the rest stays ON — the CLAMP activation (OLF's
                // roundR1ToExtremeTapPosition disables the member; the group keeps distributing)
                branch.setVoltageControlEnabled(false);
                GpuAcActivation mOff = GpuAcActivation.fromEquationSystem(network, es);
                double[] mOffTargets = mOff.applyTargets(targetVector.getArray());
                branch.setVoltageControlEnabled(true);
                List<Integer> oRows = new ArrayList<>();
                for (int r = 0; r < baseTargets.length; r++) {
                    if (mOff.rowMode()[r] != onAct.rowMode()[r] || mOffTargets[r] != onTargets[r]) {
                        oRows.add(r);
                    }
                }
                int rhoRow = branchR1Cols.get(branch.getNum());
                // a clamp that only flips THIS member's rho row is expressible; anything wider means
                // the FREE role re-picked (or a single-member group turning fully off, which reverts
                // the whole group delta — also expressible)
                boolean cpuOnly = memberBranches.size() > 1
                        && oRows.stream().anyMatch(r -> r != rhoRow);
                int[] offDeltaRows = oRows.stream().mapToInt(Integer::intValue).toArray();
                int[] offDeltaModes = new int[offDeltaRows.length];
                double[] offDeltaTargets = new double[offDeltaRows.length];
                for (int k = 0; k < offDeltaRows.length; k++) {
                    offDeltaModes[k] = mOff.rowMode()[offDeltaRows[k]];
                    offDeltaTargets[k] = mOffTargets[offDeltaRows[k]];
                }
                double[][] tapRows = tapModels(branch.getPiModel());
                int baseTap = 0;
                for (int t = 1; t < tapRows.length; t++) {   // the CURRENT tap: OLF's rounding starts there
                    if (Math.abs(tapRows[t][GpuPiModelColumn.R1] - branch.getPiModel().getR1()) < Math.abs(tapRows[baseTap][GpuPiModelColumn.R1] - branch.getPiModel().getR1())) {
                        baseTap = t;
                    }
                }
                members[m] = new Member(rhoRow, branch.getPiModel().getMinR1(), branch.getPiModel().getMaxR1(),
                        branch.getPiModel().getR1(),             // individual initial (base) ratio
                        tapR1Values(branch.getPiModel()), tapRows,
                        elementMapper.getElementIndex(branch),
                        offDeltaRows, offDeltaModes, offDeltaTargets, cpuOnly, branch.getNum(), baseTap);
            }
            for (int m = 0; m < memberBranches.size(); m++) {
                memberBranches.get(m).setVoltageControlEnabled(wasEnabled[m]);
            }
            out.add(new TransformerController(vCol, vc.getTargetValue(), halfDeadband,
                    deltaRows, deltaModesOn, deltaTargetsOn, members, useInitialTapPosition,
                    deltaModesOff, deltaTargetsOff));
        }
        return out;
    }

    /** Install the controls for the in-kernel DISCRETE TransformerVoltageControl phase (OLF order 8):
     *  {@code mode} a GpuTransformerVoltageControlMode; {@code entryOn} = the activation the solve starts from
     *  already carries the control ON (the WITH_GENERATOR base load flow — OLF's initialize ran on the
     *  network). Returns true when the kernel loop owns them (every member has a tap table). */
    public static boolean uploadToDevice(long handle, List<TransformerController> tcs, int mode, boolean entryOn) {
        if (tcs == null || tcs.isEmpty()) {
            TransformerVoltageControlNative.setBatchDiscreteTvc(handle, GpuTransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL, 0, null, null, null, null, null, null, null, null, null,
                    null, null, null, null, null, null, null, null, null, null);
            return false;
        }
        int nG = tcs.size();
        int[] grpPtr = new int[nG + 1];
        int[] grpVRow = new int[nG];
        double[] grpTargetV = new double[nG];
        double[] grpHalfDb = new double[nG];
        int[] grpUseInit = new int[nG];
        int[] deltaPtr = new int[nG + 1];
        List<Integer> dRow = new ArrayList<>();
        List<Integer> dModeOn = new ArrayList<>();
        List<Integer> dModeOff = new ArrayList<>();
        List<Double> dTgtOn = new ArrayList<>();
        List<Double> dTgtOff = new ArrayList<>();
        List<Integer> memRhoRow = new ArrayList<>();
        List<Integer> memElem = new ArrayList<>();
        List<Integer> memBaseTap = new ArrayList<>();
        List<Integer> memTapPtr = new ArrayList<>();
        List<Double> memMin = new ArrayList<>();
        List<Double> memMax = new ArrayList<>();
        List<Double> memInit = new ArrayList<>();
        List<Double> tapData = new ArrayList<>();
        boolean tables = true;
        for (int g = 0; g < nG; g++) {
            TransformerController tc = tcs.get(g);
            grpPtr[g] = memRhoRow.size();
            grpVRow[g] = tc.vCol();
            grpTargetV[g] = tc.targetV();
            grpHalfDb[g] = tc.halfDeadband();
            grpUseInit[g] = tc.useInitialTapPosition() ? 1 : 0;
            deltaPtr[g] = dRow.size();
            for (int k = 0; k < tc.deltaRows().length; k++) {
                dRow.add(tc.deltaRows()[k]);
                dModeOn.add(tc.deltaModesOn()[k]);
                dModeOff.add(tc.deltaModesOff()[k]);
                dTgtOn.add(tc.deltaTargetsOn()[k]);
                dTgtOff.add(tc.deltaTargetsOff()[k]);
            }
            for (Member m : tc.members()) {
                memRhoRow.add(m.rhoRow());
                memElem.add(m.elemIndex());
                memBaseTap.add(m.baseTapIdx());
                memTapPtr.add(tapData.size() / GpuPiModelColumn.COUNT);
                memMin.add(m.minR1());
                memMax.add(m.maxR1());
                memInit.add(m.initialR1());
                tables &= m.tapRows().length > 0;
                for (double[] row : m.tapRows()) {
                    for (double d : row) {
                        tapData.add(d);
                    }
                }
            }
        }
        grpPtr[nG] = memRhoRow.size();
        deltaPtr[nG] = dRow.size();
        memTapPtr.add(tapData.size() / GpuPiModelColumn.COUNT);
        boolean owned = tables && TransformerVoltageControlNative.setBatchDiscreteTvc(handle, mode, entryOn ? 1 : 0,
                grpPtr, grpVRow, grpTargetV, grpHalfDb, grpUseInit,
                deltaPtr, dRow.stream().mapToInt(Integer::intValue).toArray(),
                dModeOn.stream().mapToInt(Integer::intValue).toArray(), dModeOff.stream().mapToInt(Integer::intValue).toArray(),
                dTgtOn.stream().mapToDouble(Double::doubleValue).toArray(), dTgtOff.stream().mapToDouble(Double::doubleValue).toArray(),
                memRhoRow.stream().mapToInt(Integer::intValue).toArray(), memElem.stream().mapToInt(Integer::intValue).toArray(),
                memBaseTap.stream().mapToInt(Integer::intValue).toArray(), memTapPtr.stream().mapToInt(Integer::intValue).toArray(),
                memMin.stream().mapToDouble(Double::doubleValue).toArray(), memMax.stream().mapToDouble(Double::doubleValue).toArray(),
                memInit.stream().mapToDouble(Double::doubleValue).toArray(),
                tapData.stream().mapToDouble(Double::doubleValue).toArray()) != 0;
        if (!owned) {
            TransformerVoltageControlNative.setBatchDiscreteTvc(handle, GpuTransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL, 0, null, null, null, null, null, null, null, null, null,
                    null, null, null, null, null, null, null, null, null, null);
        }
        LOGGER.info("Uploaded {} discrete transformer voltage controls ({} members, mode {}) to device: {}",
                nG, memRhoRow.size(), mode == GpuTransformerVoltageControlMode.WITH_GENERATOR_VOLTAGE_CONTROL ? "WITH_GENERATOR" : "AFTER_GENERATOR", owned ? "in-kernel loop" : "host orchestrator");
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("TVD_UPLOAD mode=%d entryOn=%b nGroups=%d nMembers=%d kernel=%b%n", mode, entryOn, nG, memRhoRow.size(), owned);
        }
        return owned;
    }

    /** Give the in-kernel phase's decisions back to the network (batch-of-one base load flow), exactly as
     *  OLF's own loop leaves it: a FIXED member had {@code setR1(roundFrom); roundR1ToClosestTap()} and
     *  its control disabled ({@code roundR1ToExtremeTapPosition} / {@code roundVoltageRatios}); a member
     *  never enabled keeps its base tap. {@code state} is readBatchDiscreteTvcState's layout. */
    public static void applyToNetwork(LfNetwork network, List<TransformerController> tcs, double[] state) {
        if (tcs == null || tcs.isEmpty() || state == null) {
            return;
        }
        int m = 0;
        for (TransformerController tc : tcs) {
            for (Member mem : tc.members()) {
                int k = m++ * 3;
                if (k + 2 >= state.length) {
                    return;
                }
                LfBranch branch = network.getBranch(mem.branchNum());
                if (branch == null || (int) state[k] != GpuDiscreteControlMemberState.ROUNDED) {
                    continue;
                }
                var pm = branch.getPiModel();
                int pos = pm.getTapPositionRange().getMinimum() + (int) state[k + 1];
                if (pm.getTapPosition() != pos) {
                    pm.setR1(state[k + 2]);
                    pm.roundR1ToClosestTap();
                    if (pm.getTapPosition() != pos) {        // tie on a different start tap: take the device's
                        pm.setTapPosition(pos);
                    }
                }
                if (branch.isVoltageControlEnabled()) {
                    branch.setVoltageControlEnabled(false);
                }
            }
        }
    }

    /** Discrete tap r1 values, snapshotted read-only (reflection on PiModelArray.models; an
     *  inaccessible or non-array model yields an empty table = keep the continuous ratio). */
    public static double[] tapR1Values(Object piModel) {
        double[][] models = tapModels(piModel);
        double[] taps = new double[models.length];
        for (int i = 0; i < models.length; i++) {
            taps[i] = models[i][GpuPiModelColumn.R1];
        }
        return taps;
    }

    /** Full per-tap static branch rows, snapshotted read-only from {@code PiModelArray.getModels()}:
     *  one row [y, g1, g2, b1, b2, ksi, r1, a1] per tap position — the EXACT closedBranchParameters packing, so a
     *  tap/PST action's override is the target row verbatim.
     *
     *  <p>Empty means the branch carries a continuous {@code SimplePiModel} and has no taps at all.
     *  It used to also mean "reflection on the private field did not work", which turned a JVM
     *  configuration into silently unrounded ratios and into whole-run CPU fallbacks that no gate
     *  explained; the accessor is public on the patched OLF, so that failure mode is gone. */
    public static double[][] tapModels(Object piModel) {
        if (!(piModel instanceof PiModelArray array)) {
            return new double[0][];
        }
        List<PiModel> models = array.getModels();
        double[][] taps = new double[models.size()][GpuPiModelColumn.COUNT];
        for (int i = 0; i < models.size(); i++) {
            PiModel m = models.get(i);
            taps[i] = new double[] {m.getY(), m.getG1(), m.getG2(), m.getB1(), m.getB2(),
                                    m.getKsi(), m.getR1(), m.getA1()};
        }
        return taps;
    }

}
