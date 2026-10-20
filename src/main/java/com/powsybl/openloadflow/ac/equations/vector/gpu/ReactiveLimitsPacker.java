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
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.StableRowIndexing;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuQLimitType;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveLimitsEntryFlag;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuVoltageControlRole;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * The reactive-limits pack (ReactiveLimitsOuterLoop): one entry per voltage-controlling bus with its limits, its
 * voltage target and the rows it pins to, plus the static var compensators' quadratic limits.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class ReactiveLimitsPacker {

    private ReactiveLimitsPacker() {
    }

    /** Rank of each id in lexicographic order — OLF's BY_ID_COMPARATOR tie-break, as an int the kernel can compare. */
    private static int[] idRanks(List<String> ids) {
        List<String> sorted = new ArrayList<>(ids);
        java.util.Collections.sort(sorted);
        java.util.Map<String, Integer> rank = new java.util.HashMap<>();
        for (int i = 0; i < sorted.size(); i++) {
            rank.putIfAbsent(sorted.get(i), i);
        }
        int[] out = new int[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            out[i] = rank.get(ids.get(i));
        }
        return out;
    }

    /** SVC reactive limits are VOLTAGE-DEPENDENT: {@code LfStaticVarCompensatorImpl} evaluates
     *  Bmin/Bmax*(v*nomV)^2 at the LIVE bus V on every CPU RL check, and a pinned bus re-freezes at
     *  the moved limit ({@code busesWithUpdatedQLimits}). Per RL entry, capture the quadratic
     *  coefficients (pu: SVC limit contribution = b*v^2) and the base v^2, so the kernel evaluates
     *  limit(v) = packedLimit + b*(v^2 - v2Base) — exact for the SVC part, identically zero on
     *  SVC-less buses. Must run under the SAME control-type view as the min/max capture (the
     *  monitoring flip). */
    private static void addSvcQuadratic(LfBus bus, List<Double> reactiveLimitSvcMinB, List<Double> reactiveLimitSvcMaxB,
                                        List<Double> reactiveLimitBaseVSquared) {
        double v2 = bus.getV() * bus.getV();
        double bmin = 0;
        double bmax = 0;
        for (LfGenerator g : bus.getGenerators()) {
            if (g instanceof com.powsybl.openloadflow.network.LfStaticVarCompensator
                    && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE) {
                bmin += g.getMinQ() / v2;
                bmax += g.getMaxQ() / v2;
            }
        }
        reactiveLimitSvcMinB.add(bmin);
        reactiveLimitSvcMaxB.add(bmax);
        reactiveLimitBaseVSquared.add(v2);
    }

    /** A bus whose generators include a standby-automaton MONITORING_VOLTAGE one: the VoltageMonitoring phase's
     *  controllers, which need a reactive-limit entry to be activated through even when limits are disabled. */
    private static boolean isMonitoringSvcBus(LfBus bus) {
        return bus.getGenerators().stream()
                .anyMatch(g -> g.getGeneratorControlType() == LfGenerator.GeneratorControlType.MONITORING_VOLTAGE);
    }

    private static boolean anyMonitoringSvc(LfNetwork network) {
        return network.getBuses().stream().anyMatch(ReactiveLimitsPacker::isMonitoringSvcBus);
    }

    static ReactiveLimitEntries reactiveLimitsPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                     ExtractionColumns columns, boolean reactiveLimits) {
        int[] vCol = columns.vCol();
        // Reactive-limits (PV↔PQ) pack: every LOCAL voltage-capable controller bus. Two kinds are packed so the
        // device outer loop can move a bus in EITHER direction (matching OLF, which re-derives the PV/PQ set for
        // each contingency from scratch):
        //  - base PV (BUS_TARGET_V active, BUS_TARGET_Q inactive): its branch q scatters to its own BUS_V row;
        //    the outer loop computes its generator Q and flips the row PV→PQ on a limit violation.
        //  - base PQ (BUS_TARGET_Q active, BUS_TARGET_V inactive) but voltage-CAPABLE (a generator voltage
        //    control that OLF disabled during the base loadflow because it hit a Q limit): MUST be packed too, so
        //    a contingency that RELIEVES its reactive loading can switch it BACK to PV. Omitting these froze such
        //    a bus at its base-case limit for every scenario — a real ~1e-3 error vs OLF when a contingency eased
        //    it (e.g. losing a nearby transformer). The device seeds the pinned limit type from the base state
        //    (v below target ⇒ maxQ-pinned, above ⇒ minQ-pinned) so the switch-back fires.
        // REMOTE group members (a MAIN generator voltage control whose controlled bus is NOT the member —
        // plan-group-member-dynamics Part B): packed as reactiveLimitVoltageControlRole=1 entries with the PILOT's v column as the
        // compare row (both RL switch conditions test the pilot's V against the GROUP target — verified B0).
        List<Integer> reactiveLimitVoltageRow = new ArrayList<>();
        List<Double> reactiveLimitMinQ = new ArrayList<>();
        List<Double> reactiveLimitMaxQ = new ArrayList<>();
        List<Double> reactiveLimitLoadQ = new ArrayList<>();
        List<Double> reactiveLimitTargetV = new ArrayList<>();
        List<Double> reactiveLimitSvcMinB = new ArrayList<>();
        List<Double> reactiveLimitSvcMaxB = new ArrayList<>();
        List<Double> reactiveLimitBaseVSquared = new ArrayList<>();
        List<Byte> reactiveLimitFlags = new ArrayList<>();
        List<Byte> reactiveLimitVoltageControlRole = new ArrayList<>();
        List<Integer> reactiveLimitComparedVoltageRow = new ArrayList<>();
        List<Integer> reactiveLimitBusNum = new ArrayList<>();
        // forced-PV election keys (OLF switchPvPq: controlled bus nominal V, live target P, id)
        List<Double> reactiveLimitNominalV = new ArrayList<>();
        List<Integer> reactiveLimitActivePowerRow = new ArrayList<>();
        List<Double> reactiveLimitTargetP = new ArrayList<>();
        List<String> reactiveLimitBusId = new ArrayList<>();
        // Robust-mode pin target (ReactiveLimitsOuterLoop.getInitialGenerationTargetQ): the sum of the
        // bus generators' targetQ, which is what OLF freezes a REMOTE controller at when its own voltage
        // leaves the realistic band while its Q is still inside its limits.
        List<Double> reactiveLimitInitialTargetQ = new ArrayList<>();
        // OLF's OWN reason for a base pin (LfBus.getQLimitType): 1 MIN_Q, 2 MAX_Q, 3 MIN_REALISTIC_V,
        // 4 MAX_REALISTIC_V, 0 none. The device used to guess the KIND of a base pin from which Q
        // limit the row's target was nearest to - which turns a realistic-voltage pin (target = the
        // generation targetQ, nowhere near a limit) into a MIN_Q/MAX_Q pin, and the curve-limit drift
        // then legitimately re-freezes that "limit pin" AT the limit. rte6515 VL-1251_1: OLF pins it
        // at -0.0035 (V < 16 kV), the device at -1.0000, in every one of 366 contingencies.
        List<Byte> reactiveLimitBaseQLimitType = new ArrayList<>();
        // A never-solved network has NaN bus voltages, and an SVC's reactive limits are Bmin/Bmax·V²:
        // packing them (and the quadratic reference v²) at NaN left the entry's limits NaN and the
        // device never pinned it (the base load flow on the driver; a chunk extracts after a solve).
        // Evaluate at a flat 1 pu instead — the kernel's limit(v) = base + B·(v² − v²ref) is exact at
        // any finite reference — and put the NaN back.
        List<LfBus> flatVBuses = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            if (Double.isNaN(bus.getV())) {
                flatVBuses.add(bus);
                bus.setV(1.0);
            }
        }
        try {
            // Reactive limits OFF: the pack is empty EXCEPT for standby-automaton MONITORING SVC buses. Such a bus
            // needs an entry not for its limits (no RL loop runs, so they are dormant) but for the pin machinery the
            // VoltageMonitoring phase activates it through: VoltageMonitoring maps a bus to its rl entry and,
            // finding none, recorded a SKIP for every monitored controller — which the all-or-nothing phase gate
            // (DevicePhaseOwnership.batchable) turns into "route the whole run to the CPU". Measured with
            // OLF_GPU_SA_NO_RL=1 on rte6515's busbar window: 976 of 976 batchable contingencies CPU-replayed
            // (RUNTIME:status=-2=976, 0/1000 on device) while the run's own loop list HAS GpuMonitoringVoltageOuterLoop.
            if (reactiveLimits || anyMonitoringSvc(network)) {
                // This enumeration is WIDER than OLF's, on purpose but knowingly:
                // ReactiveLimitsOuterLoop walks getControllerElements(GENERATOR) and nothing else, while
                // getGeneratorVoltageControl().isPresent() also answers yes on the CONTROLLED (pilot)
                // bus of a remote group. A pilot with no generators of its own therefore enters as a
                // local entry whose limits are a sum over ZERO generators - minQ = maxQ = 0 - and whose
                // q is the bus's own reactive balance, zero up to rounding on a transit node. Its PIN is
                // refused in the kernel (reactiveLimitsBatch, degenerate range) because at convEps 1e-12
                // it fired on the last bits; its entry is kept because a pilot row that came in PINNED
                // from the base still needs the switch-back to hand the group's voltage identity back.
                // 7 such buses here: CQHT_0/1/3/5, HMDR_0, IZUM_0, KUNQ_3.
                for (LfBus bus : network.getBuses()) {
                    if (!reactiveLimits && !isMonitoringSvcBus(bus)) {
                        continue;                                  // limits off: monitored buses only (see above)
                    }
                    int bn = bus.getNum();
                    boolean vActive = StableRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_V) >= 0;
                    boolean qActive = StableRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_Q) >= 0;
                    boolean remoteMember = RemoteMemberQPacker.isRemoteMemberEntry(bus);
                    // A pilot that is one of its OWN controllers is packed as a remote entry of class 2: its
                    // q feeds the group's DISTR_Q rows like a remote member's (read through the remote q
                    // pack, not its row), a pin puts its own Q balance on its row (mode 4, alternate dest),
                    // and an UNPIN gives the row back its voltage IDENTITY (mode 0), not a DISTR_Q form. As
                    // a LOCAL entry its pin used to land on mode 1, an EMPTY row: measured on the
                    // all-pinned-group fixture, 0.68 kV at the pilot.
                    boolean pilotSelf = PilotSelfPacker.isPilotSelfEntry(bus);
                    if ((remoteMember || pilotSelf) && vCol[bn] >= 0) {
                        var groupVoltageControl = RemoteMemberQPacker.mainGeneratorVoltageControl(bus);                   // the GROUP's control: the group identity, the pilot's V row
                        // ...but the switch-back test is the member's OWN control's: OLF's ReactiveLimitsOuterLoop reads
                        // getBusV / getBusTargetV off bus.getGeneratorVoltageControl(), which for a control MERGED into
                        // the group (a dependent section) keeps its own controlled bus and its own target while the
                        // cluster sits at the main's. Compared against the main's target a base-pinned dependent at
                        // 405 kV in a 406 kV cluster never saw its "voltage above target" and stayed pinned where OLF
                        // unpins it (deviceMergedDependentSwitchesBackAgainstItsOwnTargetMatchesCpu).
                        var own = bus.getGeneratorVoltageControl().orElse(groupVoltageControl);
                        int pilotV = vCol[own.getControlledBus().getNum()];
                        if (pilotV >= 0 && bus.getMinQ() < bus.getMaxQ()) {
                            reactiveLimitVoltageRow.add(vCol[bn]);                  // transition WRITE row: the member's own bus row
                            reactiveLimitMinQ.add(bus.getMinQ());
                            reactiveLimitMaxQ.add(bus.getMaxQ());
                            reactiveLimitLoadQ.add(bus.getLoadTargetQ());
                            reactiveLimitTargetV.add(own.getTargetValue());   // the member's own control's target
                            addSvcQuadratic(bus, reactiveLimitSvcMinB, reactiveLimitSvcMaxB, reactiveLimitBaseVSquared);
                            reactiveLimitFlags.add((byte) 0);
                            // 2 = pilot-self; 1 = a remote member OLF's robust rule APPLIES to;
                            // 3 = a remote member OLF's robust rule EXEMPTS, because the bus's OWN control is
                            // local (see isOlfRobustRemoteController). Class 3 is truthy and not 2, so every
                            // other reactiveLimitVoltageControlRole site treats it exactly like class 1 — only the kernel's robust
                            // unrealistic-voltage gate and reactiveLimitsRobustVoltageResetBatch (which tests == 1) single it out.
                            byte remoteClass = (byte) (pilotSelf ? GpuVoltageControlRole.PILOT_SELF
                                    : RemoteMemberQPacker.isOlfRobustRemoteController(bus) ? GpuVoltageControlRole.REMOTE
                                    : GpuVoltageControlRole.REMOTE_ROBUST_MODE_EXEMPT);
                            reactiveLimitVoltageControlRole.add(remoteClass);
                            reactiveLimitComparedVoltageRow.add(pilotV);                 // switch conditions compare the PILOT's V
                            reactiveLimitBusNum.add(bn);
                            if (System.getenv("OLF_GPU_RL_TRACE") != null) {
                                System.err.println("GPU_RL_ENTRY idx=" + (reactiveLimitVoltageRow.size() - 1) + " vRow=" + vCol[bn]
                                        + " bus=" + bus.getId() + " remote=" + remoteClass + " pilotVRow=" + pilotV);
                            }
                            reactiveLimitNominalV.add(own.getControlledBus().getNominalV());
                            reactiveLimitActivePowerRow.add(StableRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_P) >= 0 ? columns.pRow()[bn] : -1);
                            reactiveLimitTargetP.add(bus.getTargetP());
                            reactiveLimitBusId.add(bus.getId());
                            reactiveLimitInitialTargetQ.add(initialGenerationTargetQ(bus));
                            reactiveLimitBaseQLimitType.add(GpuQLimitType.of(bus));
                        }
                        continue;                                  // never let a base-pinned remote member
                    }                                              // masquerade as a LOCAL basePqCapable entry
                    boolean basePv = vActive && !qActive;          // voltage-controlling in the base case
                    boolean basePqCapable = qActive && !vActive    // switched to PQ in the base case but still capable
                            && !bus.isGeneratorVoltageControlEnabled();
                    if (vCol[bn] >= 0 && bus.getGeneratorVoltageControl().isPresent() && (basePv || basePqCapable)) {
                        LfGenerator monitoringSvc = bus.getGenerators().stream()
                                .filter(g -> g.getGeneratorControlType() == LfGenerator.GeneratorControlType.MONITORING_VOLTAGE)
                                .findFirst().orElse(null);
                        double minQ;
                        double maxQ;
                        if (monitoringSvc != null) {
                            // A standing-by monitored bus never reads its RL limits (mode 1, seeded unpinned) —
                            // it reads them only AFTER the VoltageMonitoring activation turns it PV, when the
                            // CPU evaluates them with the SVC in VOLTAGE control. Capture the ACTIVATED limits
                            // (a standby capture of ~[0,0] would pin the bus right back, undoing the activation).
                            monitoringSvc.setGeneratorControlType(LfGenerator.GeneratorControlType.VOLTAGE);
                            minQ = bus.getMinQ();
                            maxQ = bus.getMaxQ();
                            addSvcQuadratic(bus, reactiveLimitSvcMinB, reactiveLimitSvcMaxB, reactiveLimitBaseVSquared);
                            monitoringSvc.setGeneratorControlType(LfGenerator.GeneratorControlType.MONITORING_VOLTAGE);
                        } else {
                            minQ = bus.getMinQ();
                            maxQ = bus.getMaxQ();
                            addSvcQuadratic(bus, reactiveLimitSvcMinB, reactiveLimitSvcMaxB, reactiveLimitBaseVSquared);
                        }
                        reactiveLimitVoltageRow.add(vCol[bn]);
                        if (System.getenv("OLF_GPU_RL_TRACE") != null) {
                            // entry -> bus, so a device pin can be named and lined up against OLF's RL_SWITCH
                            System.err.println("GPU_RL_ENTRY idx=" + (reactiveLimitVoltageRow.size() - 1) + " vRow=" + vCol[bn]
                                    + " bus=" + bus.getId() + " minQ=" + minQ + " maxQ=" + maxQ
                                    + " loadQ=" + bus.getLoadTargetQ() + " basePv=" + basePv
                                    + " basePqCapable=" + basePqCapable + " vcEnabled=" + bus.isGeneratorVoltageControlEnabled()
                                    + " gens=" + bus.getGenerators().size());
                        }
                        reactiveLimitMinQ.add(minQ);
                        reactiveLimitMaxQ.add(maxQ);
                        reactiveLimitLoadQ.add(bus.getLoadTargetQ());
                        reactiveLimitTargetV.add(bus.getGeneratorVoltageControl().orElseThrow().getTargetValue());
                        // bit 1: a MONITORING_VOLTAGE SVC standing by; bit 2: this bus is NOT one of its voltage control's
                        // controller elements (a controlled pilot with no generators of its own) - the only entries the
                        // device's degenerate-range guard may skip, because OLF's loop walks controller elements and a real
                        // generator bus with a [0, 0] curve at its current P is still one of them (network.xiidm/FBQV).
                        com.powsybl.openloadflow.network.VoltageControl<LfBus> mainVc =
                                bus.getGeneratorVoltageControl().orElseThrow().getMainVoltageControl();
                        boolean ownController = mainVc.getMergedControllerElements().contains(bus)
                                || bus.getGeneratorVoltageControl().orElseThrow().getControllerElements().contains(bus);
                        reactiveLimitFlags.add((byte) ((monitoringSvc != null ? GpuReactiveLimitsEntryFlag.STAND_BY_AUTOMATON : 0)
                                | (ownController ? 0 : GpuReactiveLimitsEntryFlag.NOT_CONTROLLER_ELEMENT)));
                        reactiveLimitVoltageControlRole.add((byte) GpuVoltageControlRole.LOCAL);
                        reactiveLimitComparedVoltageRow.add(vCol[bn]);                   // local: compare row = own row
                        reactiveLimitBusNum.add(bn);
                        reactiveLimitNominalV.add(bus.getGeneratorVoltageControl().map(c -> c.getControlledBus().getNominalV()).orElse(bus.getNominalV()));
                        reactiveLimitActivePowerRow.add(StableRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_P) >= 0 ? columns.pRow()[bn] : -1);
                        reactiveLimitTargetP.add(bus.getTargetP());
                        reactiveLimitBusId.add(bus.getId());
                        reactiveLimitInitialTargetQ.add(initialGenerationTargetQ(bus));
                        reactiveLimitBaseQLimitType.add(GpuQLimitType.of(bus));
                    }
                }
            }
        } finally {
            for (LfBus bus : flatVBuses) {
                bus.setV(Double.NaN);
            }
        }
        byte[] flags = new byte[reactiveLimitFlags.size()];
        byte[] voltageControlRoles = new byte[reactiveLimitVoltageControlRole.size()];
        for (int i = 0; i < flags.length; i++) {
            flags[i] = reactiveLimitFlags.get(i);
            voltageControlRoles[i] = reactiveLimitVoltageControlRole.get(i);
        }
        if (System.getenv("OLF_GPU_RL_TRACE") != null) {
            // The pilot-self groups this extraction sees, with their member ENTRIES: the rotation the device
            // still owes OLF is computed from those entries' pin states (see PilotSelfPacker.PilotSelfPack).
            PilotSelfPacker.PilotSelfPack pilotSelfPack = PilotSelfPacker.pilotSelfGroups(network, columns, GpuAcArrays.toIntArray(reactiveLimitBusNum));
            java.util.Map<Integer, Integer> bySize = new java.util.TreeMap<>();
            for (int g = 0; g < pilotSelfPack.groupCount(); g++) {
                bySize.merge(pilotSelfPack.ptr()[g + 1] - pilotSelfPack.ptr()[g], 1, Integer::sum);
            }
            System.err.println("GPU_PILOTSELF groups=" + pilotSelfPack.groupCount() + " bySize=" + bySize);
        }

        byte[] baseQLimitTypes = new byte[reactiveLimitBaseQLimitType.size()];
        for (int i = 0; i < baseQLimitTypes.length; i++) {
            baseQLimitTypes[i] = reactiveLimitBaseQLimitType.get(i);
        }
        return new ReactiveLimitEntries(GpuAcArrays.toIntArray(reactiveLimitVoltageRow), GpuAcArrays.toDoubleArray(reactiveLimitMinQ),
                GpuAcArrays.toDoubleArray(reactiveLimitMaxQ), GpuAcArrays.toDoubleArray(reactiveLimitLoadQ),
                GpuAcArrays.toDoubleArray(reactiveLimitTargetV), GpuAcArrays.toDoubleArray(reactiveLimitSvcMinB),
                GpuAcArrays.toDoubleArray(reactiveLimitSvcMaxB), GpuAcArrays.toDoubleArray(reactiveLimitBaseVSquared),
                flags, voltageControlRoles, GpuAcArrays.toIntArray(reactiveLimitComparedVoltageRow),
                GpuAcArrays.toIntArray(reactiveLimitBusNum),
                GpuAcArrays.toDoubleArray(reactiveLimitNominalV), GpuAcArrays.toIntArray(reactiveLimitActivePowerRow), GpuAcArrays.toDoubleArray(reactiveLimitTargetP),
                idRanks(reactiveLimitBusId), GpuAcArrays.toDoubleArray(reactiveLimitInitialTargetQ), baseQLimitTypes);
    }

    /** The targetQ OLF gives a member once the reactive-limits loop UNPINS it. A member pinned in the
     *  base has its generation targetQ FROZEN at the limit (freezeGenerationTargetQAndDisableGenerator
     *  VoltageControl), so {@code getTargetQ()} returns the pin, not what the kernel needs for an
     *  unpinned member. OLF's unpin (setGeneratorVoltageControlEnabledAndRecomputeTargetQ) invalidates
     *  the frozen value and getGenerationTargetQ recomputes it as the sum over the non-voltage-
     *  controlling, enabled generators - replayed here verbatim. Measured on network.xiidm / LJYK: a
     *  three-member group pinned in the base and unpinned by the contingency kept sharing its
     *  deviation from the PINNED targets, every member landed 1.4 MVar inside its own limit, and the
     *  CPU's re-pin of two of them never happened (6e-3 kV at convEps 1e-12 on both arms). */
    static double unpinnedTargetQ(LfBus m) {
        if (!m.isGenerationTargetQFrozen()) {
            return m.getTargetQ();
        }
        double generationTargetQ = m.getGenerators().stream()
                .filter(g -> g.getGeneratorControlType() != LfGenerator.GeneratorControlType.VOLTAGE && !g.isDisabled())
                .mapToDouble(LfGenerator::getTargetQ)
                .filter(d -> !Double.isNaN(d))
                .sum();
        return generationTargetQ - m.getLoadTargetQ();
    }

    /** {@code ReactiveLimitsOuterLoop.getInitialGenerationTargetQ}, verbatim. */
    private static double initialGenerationTargetQ(LfBus bus) {
        return bus.getGenerators().stream().mapToDouble(LfGenerator::getTargetQ).sum();
    }
}
