/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.ShuntVoltageControl;
import com.powsybl.openloadflow.network.TransformerVoltageControl;
import com.powsybl.openloadflow.network.VoltageControl;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * The outer-loop packs of the GPU extraction — the parts the batched device solver gates on a
 * non-empty pack (an empty pack means "do not run that outer loop", mirroring OLF's configured
 * outer-loop list): the DISTR_Q remote-reactive-distribution contributions, the DISTR_RHO ratio-
 * distribution contributions, the reactive-limits (PV↔PQ) pack and the distributed-slack pack.
 * All read the precomputed {@link ExtractionColumns} (and DISTR_Q the family element maps).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcOuterLoopPacks {

    private GpuAcOuterLoopPacks() {
    }

    /** DISTR_Q contributions in the flat device layout: {@code {elem, side, dqRow, kind}} per row + weight. */
    record DqPack(int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                  int[] dqKMember, int[] dqJMember,
                  double[] dqMemberKey, int[] dqMemberRlEntry, int[] dqMemberGroupStart, int[] dqMemberGroupCount,
                  int[] dqMemberRow, byte[] dqMemberHasRow, int[] dqMemberQrGroup, double[] dqMemberTargetQ,
                  int[] dqMemberBusNum) {
    }

    /** Per-group member table for the IN-KERNEL reactive-key renormalization on a remote RL pin
     *  (plan-B3): OLF's updateReactiveKeys zeroes a controller-DISABLED member's key and
     *  renormalizes over the survivors — so the DISTR_Q weights are PIN-STATE-DEPENDENT and the
     *  kernel recomputes them per scenario from the RAW keys (Control.createReactiveKeys) and the
     *  per-(entry, scenario) limitType, replicating the CPU's zero-then-sum-in-member-order
     *  arithmetic bit-for-bit. dqKMember/dqJMember are −1 for non-group contributions. */
    private record MemberTables(java.util.Map<Integer, Integer> memberIdxByBusNum,
                                List<Double> key, List<Integer> rlEntry,
                                List<Integer> groupStart, List<Integer> groupCount,
                                List<Integer> row, List<Byte> hasRow, List<Double> targetQ,
                                List<Integer> busNum) {
    }

    /** DISTR_RHO contributions in the flat device layout: {@code {drRow, drCol}} per row + coefficient. */
    record DrPack(int[] drRow, int[] drCol, double[] drCoef) {
    }

    /** DISTR_SHUNT_B contributions in the flat device layout: {@code {dbRow, dbCol}} per row + coefficient —
     *  the shunt-susceptance analogue of {@link DrPack} (structurally identical linear distribution rows). */
    record DsbPack(int[] dbRow, int[] dbCol, double[] dbCoef) {
    }

    /** Reactive-limits (PV↔PQ) pack: one entry per voltage-capable controller bus. {@code rlMonitored}
     *  flags a bus whose base-PQ state is a MONITORING_VOLTAGE SVC standing by — NOT a Q-limit switch:
     *  the device RL must seed it UNPINNED (no spontaneous switch-back; only the VoltageMonitoring
     *  orchestrator may activate it). */
    record RlPack(int[] rlVrow, double[] rlMinQ, double[] rlMaxQ, double[] rlLoadQ, double[] rlVtarget,
                  double[] rlSvcBmin, double[] rlSvcBmax, double[] rlV2Base,
                  byte[] rlMonitored, byte[] rlRemote, int[] rlCmpVrow, int[] rlBusNum,
                  double[] rlNominalV, int[] rlPRow, double[] rlTargetP, int[] rlIdRank,
                  double[] rlInitTargetQ, byte[] rlBaseQLimitType) {
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
     *  not the group's controlled bus. Shared with GpuBatchRlCurves, which REPLAYS this enumeration
     *  and must count exactly the same buses (a drift throws and the run falls to the CPU). */
    public static boolean isRemoteMemberEntry(LfBus bus) {
        return bus.getGeneratorVoltageControl()
                .filter(c -> c.getMergeStatus() == com.powsybl.openloadflow.network.VoltageControl.MergeStatus.MAIN)
                .map(c -> c.getControlledBus() != bus).orElse(false);
    }

    /** A class-2 entry: a pilot that is one of its own controllers (packed as a remote entry). */
    public static boolean isPilotSelfEntry(LfBus bus) {
        return !isRemoteMemberEntry(bus) && GpuAcRowIndexing.isRemoteGenVcControllerElement(bus)
                && bus.getGeneratorVoltageControl().map(c -> c.getControlledBus() == bus).orElse(false);
    }

    static RmqPack remoteMemberQPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                     ExtractionColumns columns, GpuAcFamilyPacker.Packs family, boolean reactiveLimits) {
        List<int[]> out = new ArrayList<>();
        List<Double> w = new ArrayList<>();
        int[] vCol = columns.vCol();
        if (reactiveLimits) {
            for (LfBus bus : network.getBuses()) {
                boolean member = GpuAcRowIndexing.isRemoteGenVcControllerElement(bus);   // incl. a pilot that is a controller
                if (!member || vCol[bus.getNum()] < 0 || !(bus.getMinQ() < bus.getMaxQ())) {
                    continue;                                // mirror the RL pack's remote-entry condition
                }
                var vc = bus.getGeneratorVoltageControl().orElseThrow();
                if (vCol[vc.getControlledBus().getNum()] < 0) {
                    continue;
                }
                int row = vCol[bus.getNum()];
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
                    if (br.isZeroImpedance(LoadFlowModel.AC) && br.isSpanningTreeEdge(LoadFlowModel.AC)) {
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

    /** Distributed-slack pack: per-participating-bus factor/headroom + the slack bus's own share.
     *  {@code dsMode} 0 = static factors ({@code dsFactor}); 1 = REMAINING_MARGIN (factors are the
     *  direction-dependent per-bus margins {@code dsMarginUp}/{@code dsMarginDown} + the slack's own
     *  {@code dsSlackMarginUp}/{@code Down}, normalized in-kernel per scenario and pass). */
    /**
     * @param dgBus       per participating GENERATOR, the index of its bus in {@code dsPhiRow}
     * @param dgFactor    per generator, its RAW participation factor (same normalization as dsFactor)
     * @param dgMaxDelta  per generator, {@code maxTargetP - targetP} under OLF's no-sign-change rule
     * @param dgMinDelta  per generator, {@code minTargetP - targetP}
     * @param dgMarginUp  per generator, REMAINING_MARGIN's own direction-dependent factor (up)
     * @param dgMarginDown same, down
     * @param busGenStart per bus, the first index into the generator arrays (generators are grouped
     *                    by bus and contiguous, so a bus's row value is the sum over its slice)
     * @param busGenEnd   per bus, one past its last generator
     */
    record DsPack(int[] dsPhiRow, double[] dsFactor, int dsSlackPhiRow, double dsSlackTargetP,
                  double[] dsBaseTargetP, double[] dsMaxDelta, double[] dsMinDelta,
                  int dsMode, double[] dsMarginUp, double[] dsMarginDown,
                  double dsSlackMarginUp, double dsSlackMarginDown,
                  int[] dgBus, double[] dgFactor, double[] dgMaxDelta, double[] dgMinDelta,
                  double[] dgMarginUp, double[] dgMarginDown, int[] busGenStart, int[] busGenEnd,
                  int[] dsBusNum, String[] dgGenId) {
    }

    static DqPack distrQPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                             ExtractionColumns columns, GpuAcFamilyPacker.Packs family, RlPack rl) {
        // member tables (plan-B3): raw keys + RL-entry mapping + group ranges, in group member order
        MemberTables mt = new MemberTables(new java.util.HashMap<>(), new ArrayList<>(), new ArrayList<>(),
                new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),
                new ArrayList<>());
        java.util.Map<Integer, Integer> rlEntryByBusNum = new java.util.HashMap<>();
        for (int i = 0; i < rl.rlBusNum().length; i++) {
            rlEntryByBusNum.put(rl.rlBusNum()[i], i);
        }
        List<Integer> qrGroupList = new ArrayList<>();
        for (LfBus pilot : network.getBuses()) {
            GeneratorVoltageControl gvc = pilot.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> pilot.isGeneratorVoltageControlled() && !c.isLocalControl())
                    .orElse(null);
            if (gvc == null) {
                continue;
            }
            List<LfBus> controllers = gvc.getMergedControllerElements();
            double[] rawKeys = com.powsybl.openloadflow.network.Control.createReactiveKeys(
                    controllers, com.powsybl.openloadflow.network.LfGenerator.GeneratorControlType.VOLTAGE);
            int start = mt.key().size();
            LfBus freeMember = controllers.stream().filter(gvc::isControllerEnabled).findFirst().orElse(null);
            // the pilot's relocation group: the kernels re-pick the pilot-Q carrier by mode-scan, so
            // the CURRENT carrier's DISTR_Q row is suppressed and its target carries the pilot's Q
            // instead. Read from qrGroupOfPilot, NOT decoded from qRowAlt: table membership and Q
            // re-routing are separate facts that happened to share one encoding.
            int qrGroup = columns.qrGroupOfPilot()[pilot.getNum()];
            for (int i = 0; i < controllers.size(); i++) {
                LfBus m = controllers.get(i);
                mt.memberIdxByBusNum().put(m.getNum(), mt.key().size());
                mt.key().add(rawKeys[i]);
                mt.rlEntry().add(rlEntryByBusNum.getOrDefault(m.getNum(), -1));
                mt.row().add(columns.vCol()[m.getNum()]);
                // a member OWNS a DISTR_Q row when the equation EXISTS and it is not the FREE
                // member (whose row carries the pilot's relocated Q)
                mt.hasRow().add((byte) (m != freeMember
                        && es.getEquation(m.getNum(), AcEquationType.DISTR_Q).isPresent() ? 1 : 0));
                // the member's UNPINNED targetQ (generationTargetQ - loadTargetQ), which is what
                // AcTargetVector reads for every member the RL loop has not frozen; a frozen
                // member's targetQ is its q limit and the kernel derives that from the RL tables.
                mt.targetQ().add(unpinnedTargetQ(m));
                mt.busNum().add(m.getNum());
                qrGroupList.add(qrGroup);
            }
            for (int i = 0; i < controllers.size(); i++) {
                mt.groupStart().add(start);
                mt.groupCount().add(controllers.size());
            }
        }
        List<int[]> dq = new ArrayList<>();      // {elem, side, dqRow, kind, kMember, jMember} per DISTR_Q contribution
        List<Double> dqW = new ArrayList<>();    // weight per contribution
        buildDistrQContributions(network, es, columns.vCol(), family.cbElemOf(), family.shElemOf(),
                family.obElemOf(), columns.dummyQCol(), dq, dqW, mt.memberIdxByBusNum());
        // dqKind: 0 = closed-branch q1/q2, 1 = shunt q, 2 = open-branch q,
        // 3 = zero-impedance DUMMY_Q (linear, elem holds the var column)
        int[] dqKind = new int[dq.size()];
        int[] dqKMember = new int[dq.size()];
        int[] dqJMember = new int[dq.size()];
        int[] dqElem = new int[dq.size()];
        int[] dqSide = new int[dq.size()];
        int[] dqRow = new int[dq.size()];
        double[] dqWeight = new double[dq.size()];
        for (int i = 0; i < dq.size(); i++) {
            dqElem[i] = dq.get(i)[0];
            dqSide[i] = dq.get(i)[1];
            dqRow[i] = dq.get(i)[2];
            dqKind[i] = dq.get(i)[3];
            dqKMember[i] = dq.get(i)[4];
            dqJMember[i] = dq.get(i)[5];
            dqWeight[i] = dqW.get(i);
        }
        byte[] hasRow = new byte[mt.hasRow().size()];
        for (int i = 0; i < hasRow.length; i++) {
            hasRow[i] = mt.hasRow().get(i);
        }
        return new DqPack(dqElem, dqSide, dqWeight, dqRow, dqKind, dqKMember, dqJMember,
                GpuAcArrays.toDoubleArray(mt.key()), GpuAcArrays.toIntArray(mt.rlEntry()),
                GpuAcArrays.toIntArray(mt.groupStart()), GpuAcArrays.toIntArray(mt.groupCount()),
                GpuAcArrays.toIntArray(mt.row()), hasRow, GpuAcArrays.toIntArray(qrGroupList),
                GpuAcArrays.toDoubleArray(mt.targetQ()), GpuAcArrays.toIntArray(mt.busNum()));
    }

    static DrPack distrRhoPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                               ExtractionColumns columns) {
        List<int[]> dr = new ArrayList<>();      // {drRow, drCol} per DISTR_RHO contribution
        List<Double> drC = new ArrayList<>();    // coefficient per contribution
        buildDistrRhoContributions(network, es, columns.branchRhoCol(), dr, drC);
        int[] drRow = new int[dr.size()];
        int[] drCol = new int[dr.size()];
        double[] drCoef = new double[dr.size()];
        for (int i = 0; i < dr.size(); i++) {
            drRow[i] = dr.get(i)[0];
            drCol[i] = dr.get(i)[1];
            drCoef[i] = drC.get(i);
        }
        return new DrPack(drRow, drCol, drCoef);
    }

    static DsbPack distrShuntBPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                   ExtractionColumns columns) {
        List<int[]> db = new ArrayList<>();      // {dbRow, dbCol} per DISTR_SHUNT_B contribution
        List<Double> dbC = new ArrayList<>();    // coefficient per contribution
        buildDistrShuntBContributions(network, es, columns.shuntBCol(), db, dbC);
        int[] dbRow = new int[db.size()];
        int[] dbCol = new int[db.size()];
        double[] dbCoef = new double[db.size()];
        for (int i = 0; i < db.size(); i++) {
            dbRow[i] = db.get(i)[0];
            dbCol[i] = db.get(i)[1];
            dbCoef[i] = dbC.get(i);
        }
        return new DsbPack(dbRow, dbCol, dbCoef);
    }

    /** SVC reactive limits are VOLTAGE-DEPENDENT: {@code LfStaticVarCompensatorImpl} evaluates
     *  Bmin/Bmax*(v*nomV)^2 at the LIVE bus V on every CPU RL check, and a pinned bus re-freezes at
     *  the moved limit ({@code busesWithUpdatedQLimits}). Per RL entry, capture the quadratic
     *  coefficients (pu: SVC limit contribution = b*v^2) and the base v^2, so the kernel evaluates
     *  limit(v) = packedLimit + b*(v^2 - v2Base) — exact for the SVC part, identically zero on
     *  SVC-less buses. Must run under the SAME control-type view as the min/max capture (the
     *  monitoring flip). */
    /** Σ over the bus's participating generators of (current − anchor target P): what OLF's
     *  resetToInitialState undoes and adds back as previousMismatch. 0 when anchoring at the current
     *  targets. The anchor row target = the row's current target minus this. */
    static double anchorOffset(LfBus bus, GpuDsParams dsParams) {
        double off = 0;
        for (LfGenerator g : bus.getGenerators()) {
            if (dsParams.factor(g) != 0) {
                off += g.getTargetP() - dsParams.anchorTargetP(g);
            }
        }
        return off;
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

    private static void addSvcQuadratic(LfBus bus, List<Double> rlSvcBmin, List<Double> rlSvcBmax,
                                        List<Double> rlV2Base) {
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
        rlSvcBmin.add(bmin);
        rlSvcBmax.add(bmax);
        rlV2Base.add(v2);
    }

    static RlPack reactiveLimitsPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
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
        // plan-group-member-dynamics Part B): packed as rlRemote=1 entries with the PILOT's v column as the
        // compare row (both RL switch conditions test the pilot's V against the GROUP target — verified B0).
        // The kernel SKIPS remote entries until the B3 transition machinery lands; the whole-run scope gate
        // still routes remote-pinnable runs to the CPU, so these entries are structurally inert today.
        List<Integer> rlVrow = new ArrayList<>();
        List<Double> rlMinQ = new ArrayList<>();
        List<Double> rlMaxQ = new ArrayList<>();
        List<Double> rlLoadQ = new ArrayList<>();
        List<Double> rlVtarget = new ArrayList<>();
        List<Double> rlSvcBmin = new ArrayList<>();
        List<Double> rlSvcBmax = new ArrayList<>();
        List<Double> rlV2Base = new ArrayList<>();
        List<Byte> rlMonitored = new ArrayList<>();
        List<Byte> rlRemote = new ArrayList<>();
        List<Integer> rlCmpVrow = new ArrayList<>();
        List<Integer> rlBusNum = new ArrayList<>();
        // forced-PV election keys (OLF switchPvPq: controlled bus nominal V, live target P, id)
        List<Double> rlNominalV = new ArrayList<>();
        List<Integer> rlPRow = new ArrayList<>();
        List<Double> rlTargetP = new ArrayList<>();
        List<String> rlBusId = new ArrayList<>();
        // Robust-mode pin target (ReactiveLimitsOuterLoop.getInitialGenerationTargetQ): the sum of the
        // bus generators' targetQ, which is what OLF freezes a REMOTE controller at when its own voltage
        // leaves the realistic band while its Q is still inside its limits.
        List<Double> rlInitTargetQ = new ArrayList<>();
        // OLF's OWN reason for a base pin (LfBus.getQLimitType): 1 MIN_Q, 2 MAX_Q, 3 MIN_REALISTIC_V,
        // 4 MAX_REALISTIC_V, 0 none. The device used to guess the KIND of a base pin from which Q
        // limit the row's target was nearest to - which turns a realistic-voltage pin (target = the
        // generation targetQ, nowhere near a limit) into a MIN_Q/MAX_Q pin, and the curve-limit drift
        // then legitimately re-freezes that "limit pin" AT the limit. rte6515 VL-1251_1: OLF pins it
        // at -0.0035 (V < 16 kV), the device at -1.0000, in every one of 366 contingencies.
        List<Byte> rlBaseQLimitType = new ArrayList<>();
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
        if (reactiveLimits) {                                   // empty pack when the RL outer loop is disabled
            // KNOWN DEFECT, measured 2026-09-23, not yet fixed - read this before touching the
            // predicate below. This enumeration is WIDER than OLF's: ReactiveLimitsOuterLoop walks
            // getControllerElements(GENERATOR) and nothing else, while
            // getGeneratorVoltageControl().isPresent() also answers yes on the CONTROLLED (pilot)
            // bus of a remote group. A pilot with no generators of its own therefore enters as a
            // local entry whose limits are a sum over ZERO generators - minQ = maxQ = 0 - and whose
            // q is the bus's own reactive balance, which for a pure transit node is zero up to
            // rounding. Against the reactive-limit tolerance (convEps under UNIFORM_CRITERIA, 1e-12
            // in validation runs) such an entry pins MIN_Q or MAX_Q on the LAST BITS: two runs of
            // the same 2000-contingency window of network.xiidm differed on 430 reactive-limit
            // decisions and every single one was such an entry, q at +-1.2e-12 against limits of
            // exactly 0 (7 buses: CQHT_0/1/3/5, HMDR_0, IZUM_0, KUNQ_3). That is where the device's
            // run-to-run non-determinism comes from, and once per ~2000 contingencies it changes a
            // real downstream pin (8.178e-05 kV on the KUMB group).
            // Excluding them is NOT enough on its own: for a group whose voltage control is already
            // OFF in the base case, this entry's pin/unpin is currently the ONLY thing that takes
            // the pilot's row between its own Q balance and the group's voltage identity
            // (updatePilotVeqBatch restores only rows IT flipped, and pilotRowAllPinnedBatch covers
            // a pilot that is its own controller). Dropping the entry, or refusing to pin a
            // degenerate range, both move LKPH by 0.32 kV - measured. The fix is to drive the
            // pilot's row from the GROUP's control state (any member unpinned = voltage identity),
            // and only then take these buses out of the pack.
            for (LfBus bus : network.getBuses()) {
                int bn = bus.getNum();
                boolean vActive = GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_V) >= 0;
                boolean qActive = GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_Q) >= 0;
                boolean remoteMember = isRemoteMemberEntry(bus);
                // A pilot that is one of its OWN controllers is packed as a remote entry of class 2: its
                // q feeds the group's DISTR_Q rows like a remote member's (read through the remote q
                // pack, not its row), a pin puts its own Q balance on its row (mode 4, alternate dest),
                // and an UNPIN gives the row back its voltage IDENTITY (mode 0), not a DISTR_Q form. As
                // a LOCAL entry its pin used to land on mode 1, an EMPTY row: measured on the
                // all-pinned-group fixture, 0.68 kV at the pilot.
                boolean pilotSelf = isPilotSelfEntry(bus);
                if ((remoteMember || pilotSelf) && vCol[bn] >= 0) {
                    var vc = bus.getGeneratorVoltageControl().orElseThrow();
                    int pilotV = vCol[vc.getControlledBus().getNum()];
                    if (pilotV >= 0 && bus.getMinQ() < bus.getMaxQ()) {
                        rlVrow.add(vCol[bn]);                  // transition WRITE row: the member's own bus row
                        rlMinQ.add(bus.getMinQ());
                        rlMaxQ.add(bus.getMaxQ());
                        rlLoadQ.add(bus.getLoadTargetQ());
                        rlVtarget.add(vc.getTargetValue());    // the GROUP target (pilot-referenced)
                        addSvcQuadratic(bus, rlSvcBmin, rlSvcBmax, rlV2Base);
                        rlMonitored.add((byte) 0);
                        rlRemote.add((byte) (pilotSelf ? 2 : 1));
                        rlCmpVrow.add(pilotV);                 // switch conditions compare the PILOT's V
                        rlBusNum.add(bn);
                        if (System.getenv("OLF_GPU_RL_TRACE") != null) {
                            System.err.println("GPU_RL_ENTRY idx=" + (rlVrow.size() - 1) + " vRow=" + vCol[bn]
                                    + " bus=" + bus.getId() + " remote=" + (pilotSelf ? 2 : 1) + " pilotVRow=" + pilotV);
                        }
                        rlNominalV.add(vc.getControlledBus().getNominalV());
                        rlPRow.add(GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_P) >= 0 ? columns.phiCol()[bn] : -1);
                        rlTargetP.add(bus.getTargetP());
                        rlBusId.add(bus.getId());
                        rlInitTargetQ.add(initialGenerationTargetQ(bus));
                        rlBaseQLimitType.add(qLimitTypeCode(bus));
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
                        addSvcQuadratic(bus, rlSvcBmin, rlSvcBmax, rlV2Base);
                        monitoringSvc.setGeneratorControlType(LfGenerator.GeneratorControlType.MONITORING_VOLTAGE);
                    } else {
                        minQ = bus.getMinQ();
                        maxQ = bus.getMaxQ();
                        addSvcQuadratic(bus, rlSvcBmin, rlSvcBmax, rlV2Base);
                    }
                    rlVrow.add(vCol[bn]);
                    if (System.getenv("OLF_GPU_RL_TRACE") != null) {
                        // entry -> bus, so a device pin can be named and lined up against OLF's RL_SWITCH
                        System.err.println("GPU_RL_ENTRY idx=" + (rlVrow.size() - 1) + " vRow=" + vCol[bn]
                                + " bus=" + bus.getId() + " minQ=" + minQ + " maxQ=" + maxQ
                                + " loadQ=" + bus.getLoadTargetQ() + " basePv=" + basePv
                                + " basePqCapable=" + basePqCapable + " vcEnabled=" + bus.isGeneratorVoltageControlEnabled()
                                + " gens=" + bus.getGenerators().size());
                    }
                    rlMinQ.add(minQ);
                    rlMaxQ.add(maxQ);
                    rlLoadQ.add(bus.getLoadTargetQ());
                    rlVtarget.add(bus.getGeneratorVoltageControl().orElseThrow().getTargetValue());
                    rlMonitored.add(monitoringSvc != null ? (byte) 1 : (byte) 0);
                    rlRemote.add((byte) 0);
                    rlCmpVrow.add(vCol[bn]);                   // local: compare row = own row
                    rlBusNum.add(bn);
                    rlNominalV.add(bus.getGeneratorVoltageControl().map(c -> c.getControlledBus().getNominalV()).orElse(bus.getNominalV()));
                    rlPRow.add(GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_P) >= 0 ? columns.phiCol()[bn] : -1);
                    rlTargetP.add(bus.getTargetP());
                    rlBusId.add(bus.getId());
                    rlInitTargetQ.add(initialGenerationTargetQ(bus));
                        rlBaseQLimitType.add(qLimitTypeCode(bus));
                }
            }
        }
        } finally {
            for (LfBus bus : flatVBuses) {
                bus.setV(Double.NaN);
            }
        }
        byte[] mon = new byte[rlMonitored.size()];
        byte[] rem = new byte[rlRemote.size()];
        for (int i = 0; i < mon.length; i++) {
            mon[i] = rlMonitored.get(i);
            rem[i] = rlRemote.get(i);
        }
        if (System.getenv("OLF_GPU_RL_PACK_DUMP") != null) {
            for (int i = 0; i < rlBusNum.size(); i++) {
                // cmpVrow is the row whose V the switch condition reads - for a REMOTE member that is
                // the PILOT's row, not the member's own, so a pin that looks unexplainable from the
                // member alone is often the pilot's V. loadQ and vTarget complete the pin target
                // (OLF pins at limit - loadTargetQ), which is the other half of any pin mismatch.
                System.err.printf("RL_PACK entry=%d bus=%s vRow=%d minQ=%.9f maxQ=%.9f remote=%d"
                        + " monitored=%d cmpVrow=%d loadQ=%.9f vTarget=%.9f baseQLimitType=%d%n",
                        i, network.getBus(rlBusNum.get(i)).getId(), rlVrow.get(i), rlMinQ.get(i), rlMaxQ.get(i),
                        rlRemote.get(i), rlMonitored.get(i), rlCmpVrow.get(i), rlLoadQ.get(i),
                        rlVtarget.get(i), rlBaseQLimitType.get(i));
            }
        }

        byte[] baseLt = new byte[rlBaseQLimitType.size()];
        for (int i = 0; i < baseLt.length; i++) {
            baseLt[i] = rlBaseQLimitType.get(i);
        }
        return new RlPack(GpuAcArrays.toIntArray(rlVrow), GpuAcArrays.toDoubleArray(rlMinQ),
                GpuAcArrays.toDoubleArray(rlMaxQ), GpuAcArrays.toDoubleArray(rlLoadQ),
                GpuAcArrays.toDoubleArray(rlVtarget), GpuAcArrays.toDoubleArray(rlSvcBmin),
                GpuAcArrays.toDoubleArray(rlSvcBmax), GpuAcArrays.toDoubleArray(rlV2Base),
                mon, rem, GpuAcArrays.toIntArray(rlCmpVrow),
                GpuAcArrays.toIntArray(rlBusNum),
                GpuAcArrays.toDoubleArray(rlNominalV), GpuAcArrays.toIntArray(rlPRow), GpuAcArrays.toDoubleArray(rlTargetP),
                idRanks(rlBusId), GpuAcArrays.toDoubleArray(rlInitTargetQ), baseLt);
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
    static byte qLimitTypeCode(LfBus bus) {
        return bus.getQLimitType().map(t -> switch (t) {
            case MIN_Q -> (byte) 1;
            case MAX_Q -> (byte) 2;
            case MIN_REALISTIC_V -> (byte) 3;
            case MAX_REALISTIC_V -> (byte) 4;
        }).orElse((byte) 0);
    }

    private static double initialGenerationTargetQ(LfBus bus) {
        return bus.getGenerators().stream().mapToDouble(LfGenerator::getTargetQ).sum();
    }

    static DsPack distributedSlackPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                       ExtractionColumns columns, boolean distributedSlack, GpuDsParams dsParams) {
        int[] phiCol = columns.phiCol();
        // Distributed-slack pack: the single slack/reference bus (BUS_TARGET_PHI active, BUS_TARGET_P
        // inactive — its P is free) provides the mismatch; every OTHER bus with participating generators
        // takes a share of it into its BUS_TARGET_P target. Default balance type PROPORTIONAL_TO_GENERATION_
        // P_MAX: per-bus factor = Σ (maxP / droop) over its participating generators, normalized to sum 1.
        //
        // PARTICIPATING SLACK: if the slack/reference bus ITSELF has participating generators, OLF gives them
        // a share too (it resets the reference generator and redistributes the full mismatch over ALL
        // participants incl. the slack). The GPU has no BUS_TARGET_P at the slack to retarget, but it does not
        // need one: by including the slack's factor in the NORMALIZATION DENOMINATOR (without emitting a row),
        // the non-slack shares shrink to f_i/Σ_all and the slack bus naturally retains the remaining
        // f_slack/Σ_all of the mismatch — exactly OLF's allocation.
        //
        // SATURATION: per-bus headroom/footroom (Σ over participating gens of maxTargetP−targetP / minTargetP−
        // targetP, with OLF's no-sign-change rule, useActiveLimits assumed true = the default) bound how much a
        // bus may absorb. The batch kernel clamps each bus's CUMULATIVE delta to [footroom, headroom]; an
        // over-allocation is clamped and its residual reappears at the slack, redistributed next outer pass to
        // unsaturated buses ∝ their fixed factors — converging to OLF's water-filling fixed point.
        int dsSlackPhiRow = -1;
        double dsSlackTargetP = 0;
        double dsSlackFactor = 0;                                       // slack's own participation (denominator only)
        double dsSlackMarginUp = 0;                                     // REMAINING_MARGIN: the slack's own margins
        double dsSlackMarginDown = 0;
        List<Integer> dsPhiRow = new ArrayList<>();
        List<Integer> dgBus = new ArrayList<>();
        List<Double> dgFactorRaw = new ArrayList<>();
        List<Double> dgMaxDelta = new ArrayList<>();
        List<Double> dgMinDelta = new ArrayList<>();
        List<Double> dgMarginUp = new ArrayList<>();
        List<Double> dgMarginDown = new ArrayList<>();
        List<Integer> busGenStart = new ArrayList<>();
        List<Integer> busGenEnd = new ArrayList<>();
        List<Integer> dsBusNum = new ArrayList<>();                     // WHICH bus each row is (write-back)
        List<String> dgGenId = new ArrayList<>();                       // WHICH generator each slot is
        List<Double> dsFactorRaw = new ArrayList<>();
        List<Double> dsBaseTargetP = new ArrayList<>();                 // each bus's initial BUS_TARGET_P
        List<Double> dsMaxDelta = new ArrayList<>();                    // headroom ≥ 0
        List<Double> dsMinDelta = new ArrayList<>();                    // footroom ≤ 0
        List<Double> dsMarginUp = new ArrayList<>();                    // REMAINING_MARGIN per-bus margins
        List<Double> dsMarginDown = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            if (!distributedSlack) {                                    // empty pack when the DS outer loop is disabled
                break;
            }
            int bn = bus.getNum();
            if (GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_PHI) >= 0) {       // the slack/reference bus
                dsSlackPhiRow = phiCol[bn];
                dsSlackTargetP = bus.getTargetP() - anchorOffset(bus, dsParams);   // the slack's ANCHOR target
                dsSlackFactor = dsParams.busFactor(bus);
                if (System.getenv("OLF_GPU_DS_GEN_TRACE") != null) {
                    System.err.println("GPU_DS_SLACK bus=" + bus.getId() + " busFactor=" + dsSlackFactor
                            + " gens=" + bus.getGenerators().stream().map(g -> g.getId() + ":" + dsParams.factor(g)).toList());
                }
                if (dsParams.isRemainingMargin()) {
                    double[] m = dsParams.busMargins(bus);
                    dsSlackMarginUp = m[0];
                    dsSlackMarginDown = m[1];
                }
                continue;
            }
            if (GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_P) < 0) {
                if (System.getenv("OLF_GPU_DS_GEN_TRACE") != null) {
                    System.err.println("GPU_DS_SKIP bus=" + bus.getId() + " busFactor=" + dsParams.busFactor(bus)
                            + " gens=" + bus.getGenerators().stream().map(g -> g.getId() + ":" + dsParams.factor(g)).toList()
                            + " (no active BUS_TARGET_P row)");
                }
                continue;                                               // no active P balance to retarget
            }
            double factor = dsParams.busFactor(bus);
            double headroom = 0;
            double footroom = 0;
            for (LfGenerator g : bus.getGenerators()) {
                if (dsParams.factor(g) != 0) {
                    double minTp = g.getMinTargetP();
                    double maxTp = g.getMaxTargetP();
                    double tp = dsParams.anchorTargetP(g);
                    if (tp < 0) {                                       // OLF: distribution never flips a gen's sign
                        maxTp = Math.min(maxTp, 0);
                    } else {
                        minTp = Math.max(minTp, 0);
                    }
                    headroom += maxTp - tp;
                    footroom += minTp - tp;
                }
            }
            if (!dsParams.useActiveLimits() || dsParams.isLoadBased()) {
                headroom = Double.POSITIVE_INFINITY;                    // OLF: +-Double.MAX_VALUE bounds (or loads,
                footroom = Double.NEGATIVE_INFINITY;                    // which OLF rescales unbounded) — the
            }                                                           // water-fill clamp never saturates
            if (factor > 0 && System.getenv("OLF_GPU_DS_PACK_TRACE") != null
                    && (footroom > 0 || headroom < 0)) {
                // The base P is OUTSIDE the bus's own active limits. OLF's water-fill still SETS such a
                // generator to the bound (newTargetP = minTargetP), which MOVES it -- in the opposite
                // direction to the mismatch when it starts beyond the bound.
                System.err.printf("DS_PACK_OUTSIDE bus=%s baseTargetP=%.9f headroom=%.9f footroom=%.9f factor=%.9f%n",
                        bus.getId(), bus.getTargetP(), headroom, footroom, factor);
            }
            if (factor > 0) {
                if (System.getenv("OLF_GPU_DS_TARGET_DUMP") != null) {
                    System.err.println("GPU_DS_BUS idx=" + dsPhiRow.size() + " bus=" + bus.getId());
                }
                dsPhiRow.add(phiCol[bn]);
                dsBusNum.add(bn);
                dsFactorRaw.add(factor);
                dsBaseTargetP.add(bus.getTargetP() - anchorOffset(bus, dsParams));   // the row's ANCHOR target
                dsMaxDelta.add(headroom);
                dsMinDelta.add(footroom);
                if (dsParams.isRemainingMargin()) {
                    double[] m = dsParams.busMargins(bus);
                    dsMarginUp.add(m[0]);
                    dsMarginDown.add(m[1]);
                }
                // Per GENERATOR as well, contiguous within the bus. OLF's water-fill participates and
                // SATURATES per generator (ParticipatingElement wraps an LfGenerator, and step.run
                // clamps each one at its OWN minTargetP/maxTargetP and removes it), so a bus-level
                // aggregate only reproduces it while nothing saturates. Both groupings are kept: the
                // bus one owns the equation row and the anchor, the generator one owns the factors
                // and the limits.
                busGenStart.add(dgFactorRaw.size());
                for (LfGenerator g : bus.getGenerators()) {
                    double gf = dsParams.factor(g);
                    if (gf == 0) {
                        continue;                            // OLF: isParticipating(gen) && factor != 0
                    }
                    double minTp = g.getMinTargetP();
                    double maxTp = g.getMaxTargetP();
                    double tp = dsParams.anchorTargetP(g);
                    if (tp < 0) {                            // OLF: distribution never flips a gen's sign
                        maxTp = Math.min(maxTp, 0);
                    } else {
                        minTp = Math.max(minTp, 0);
                    }
                    if (System.getenv("OLF_GPU_DS_GEN_TRACE") != null) {
                        System.err.println("GPU_DS_GEN idx=" + dgFactorRaw.size() + " bus=" + bus.getId()
                                + " gen=" + g.getId() + " factor=" + gf
                                + " targetP=" + g.getTargetP()
                                + " maxTargetP=" + g.getMaxTargetP() + " minTargetP=" + g.getMinTargetP());
                    }
                    dgBus.add(dsPhiRow.size() - 1);
                    dgGenId.add(g.getId());
                    dgFactorRaw.add(gf);
                    boolean unbounded = !dsParams.useActiveLimits() || dsParams.isLoadBased();
                    dgMaxDelta.add(unbounded ? Double.POSITIVE_INFINITY : maxTp - tp);
                    dgMinDelta.add(unbounded ? Double.NEGATIVE_INFINITY : minTp - tp);
                    if (dsParams.isRemainingMargin()) {
                        double minP = g.getMinP();
                        double maxP = g.getMaxP();
                        if (tp < 0) {
                            maxP = Math.min(maxP, 0);
                        } else {
                            minP = Math.max(minP, 0);
                        }
                        dgMarginUp.add(Math.max(0.0, maxP - tp));
                        dgMarginDown.add(Math.max(0.0, tp - minP));
                    }
                }
                busGenEnd.add(dgFactorRaw.size());
            }
        }
        double dsSum = dsFactorRaw.stream().mapToDouble(Double::doubleValue).sum() + dsSlackFactor;
        double[] dsFactor = new double[dsPhiRow.size()];
        for (int i = 0; i < dsFactor.length; i++) {
            dsFactor[i] = dsFactorRaw.get(i) / dsSum;
        }
        double[] dgFactor = new double[dgFactorRaw.size()];     // SAME denominator, so Σ over a bus's
        for (int i = 0; i < dgFactor.length; i++) {             // generators is exactly its dsFactor
            dgFactor[i] = dgFactorRaw.get(i) / dsSum;
        }
        // A pack with participants but no slack row is unusable; one with NEITHER participants nor a
        // participating slack means DS never fires. But a PARTICIPATING SLACK ALONE is kept (nds == 0,
        // dsSlackPhiRow >= 0): OLF then distributes the whole mismatch into the slack generator's own
        // target — no flow change, but the outer loop reports UNSTABLE and re-solves, and the device
        // must take that same extra polish run (slackOnlyDsCheck) to match the CPU trajectory.
        if (dsSlackPhiRow < 0 || dsPhiRow.isEmpty() && dsSlackFactor <= 0) {
            dsPhiRow.clear();
            dsFactor = new double[0];
            dsBaseTargetP.clear();
            dsMaxDelta.clear();
            dsMinDelta.clear();
            dsMarginUp.clear();
            dsMarginDown.clear();
            dsSlackPhiRow = -1;
            dgBus.clear();
            dgFactor = new double[0];
            dgMaxDelta.clear();
            dgMinDelta.clear();
            dgMarginUp.clear();
            dgMarginDown.clear();
            busGenStart.clear();
            busGenEnd.clear();
            dsBusNum.clear();
            dgGenId.clear();
        }
        return new DsPack(GpuAcArrays.toIntArray(dsPhiRow), dsFactor, dsSlackPhiRow, dsSlackTargetP,
                GpuAcArrays.toDoubleArray(dsBaseTargetP), GpuAcArrays.toDoubleArray(dsMaxDelta),
                GpuAcArrays.toDoubleArray(dsMinDelta),
                dsParams.isRemainingMargin() ? 1 : 0,
                GpuAcArrays.toDoubleArray(dsMarginUp), GpuAcArrays.toDoubleArray(dsMarginDown),
                dsSlackMarginUp, dsSlackMarginDown,
                GpuAcArrays.toIntArray(dgBus), dgFactor,
                GpuAcArrays.toDoubleArray(dgMaxDelta), GpuAcArrays.toDoubleArray(dgMinDelta),
                GpuAcArrays.toDoubleArray(dgMarginUp), GpuAcArrays.toDoubleArray(dgMarginDown),
                GpuAcArrays.toIntArray(busGenStart), GpuAcArrays.toIntArray(busGenEnd),
                GpuAcArrays.toIntArray(dsBusNum), dgGenId.toArray(new String[0]));
    }

    /**
     * Build the weighted closed-branch reactive contributions of every active DISTR_Q row (remote
     * generator voltage control). For a control group, the row at controller {@code Gk}'s BUS_V row is
     * {@code DISTR_Q_k = qPct_k·Σ_j q_Gj − q_Gk}, so each controller {@code Gj}'s reactive injection
     * (the q1/q2 flow of each branch incident to {@code Gj}) contributes with weight
     * {@code qPct_k − (j==k ? 1 : 0)}. A controller's reactive injection is its closed-branch q1/q2 (kind 0),
     * a fixed shunt's q (kind 1), a half-open incident branch's connected-side q (kind 2), or a zero-impedance
     * spanning-tree branch's linear DUMMY_Q variable term (kind 3) — mirroring the terms {@code createReactiveTerms}
     * sums. The pilot bus's own reactive balance is handled by qRow (routed to the free controller's BUS_V row);
     * the controllers' own per-bus Q scatter is suppressed there too.
     */
    private static void buildDistrQContributions(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                 int[] vCol, int[] cbElemOf, int[] shElemOf, int[] obElemOf,
                                                 int[] dummyQCol, List<int[]> out, List<Double> weights,
                                                 java.util.Map<Integer, Integer> memberIdxByBusNum) {
        for (LfBus pilot : network.getBuses()) {
            GeneratorVoltageControl gvc = pilot.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> pilot.isGeneratorVoltageControlled() && !c.isLocalControl())
                    .orElse(null);
            if (gvc == null) {
                continue;
            }
            List<LfBus> controllers = gvc.getMergedControllerElements();
            LfBus freeCtl = controllers.stream()
                    .filter(b -> gvc.isControllerEnabled(b)).findFirst().orElse(null);
            if (System.getenv("OLF_GPU_FREECTL_TRACE") != null) {
                System.err.printf("FREECTL pilot=%s members=%s freeCtl=%s olfHasDistrQ=%s%n",
                        gvc.getControlledBus().getId(),
                        controllers.stream().map(LfBus::getId).toList(),
                        freeCtl == null ? "-" : freeCtl.getId(),
                        controllers.stream().map(b -> b.getId() + "="
                                + !es.getEquation(b.getNum(), AcEquationType.DISTR_Q).isEmpty()).toList());
            }
            for (LfBus gk : controllers) {
                // EXISTENCE-keyed (the A1 lesson): pack every non-free member whose DISTR_Q equation
                // EXISTS — a base-PINNED member's row is inactive (mode 4, own Q) but an in-flight
                // UNPIN re-activates it, and the union pattern must already hold its slots. The row-
                // mode/fill gates decide per scenario which face the row shows.
                if (gk == freeCtl || es.getEquation(gk.getNum(), AcEquationType.DISTR_Q).isEmpty()) {
                    continue;                                // the free controller carries the pilot's Q
                }
                int dqRow = GpuAcRowIndexing.distrQRow(es, gk);   // free-controller row when gk IS the pilot
                double qpk = gk.getRemoteControlReactivePercent();
                int kIdx = memberIdxByBusNum.getOrDefault(gk.getNum(), -1);
                for (LfBus gj : controllers) {
                    double weight = qpk - (gj == gk ? 1.0 : 0.0);
                    int jIdx = memberIdxByBusNum.getOrDefault(gj.getNum(), -1);
                    if (gj.getControllerShunt().isPresent()) {
                        throw new PowsyblException("GPU AC solver: DISTR_Q controller " + gj.getId()
                                + " has a voltage-controlling shunt (SHUNT_B) reactive contribution (not supported)");
                    }
                    if (gj.getShunt().isPresent()) {         // a fixed shunt's reactive joins the q sum
                        out.add(new int[] {shElemOf[gj.getNum()], 0, dqRow, 1, kIdx, jIdx});
                        weights.add(weight);
                    }
                    for (LfBranch brj : gj.getBranches()) {
                        if (brj.isDisabled()) {
                            continue;
                        }
                        int side = brj.getBus1() == gj ? 0 : 1;
                        int e = cbElemOf[brj.getNum()];
                        if (e >= 0) {                        // closed branch: weighted q1/q2 (kind 0)
                            out.add(new int[] {e, side, dqRow, 0, kIdx, jIdx});
                            weights.add(weight);
                            continue;
                        }
                        int oe = obElemOf[brj.getNum()];
                        if (oe >= 0) {                       // half-open branch incident on the controller:
                            // its connected-side reactive flow (shunt + series) joins the q sum (kind 2)
                            out.add(new int[] {oe, side, dqRow, 2, kIdx, jIdx});
                            weights.add(weight);
                            continue;
                        }
                        if (brj.isZeroImpedance(LoadFlowModel.AC)) {
                            // a zero-impedance coupler: only a spanning-tree edge carries reactive flow, as the
                            // linear DUMMY_Q variable (+1 if the controller is bus1, −1 if bus2) — exactly the
                            // term createReactiveTerms adds. Non-tree edges contribute nothing (matches OLF).
                            if (!brj.isSpanningTreeEdge(LoadFlowModel.AC)) {
                                continue;
                            }
                            int dqCol = dummyQCol[brj.getNum()];
                            if (dqCol < 0) {
                                throw new PowsyblException("GPU AC solver: DISTR_Q controller " + gj.getId()
                                        + " has a zero-impedance branch " + brj.getId() + " without a DUMMY_Q variable");
                            }
                            // Keyed by dummy-Q column OR, with a leading 'r', by dqROW - when the
                            // question is "what lands in THIS equation" the column is not known yet.
                            String dq3 = System.getenv("OLF_GPU_DQ3_TRACE");
                            if (dq3 != null && (dq3.startsWith("r")
                                    ? dqRow == Integer.parseInt(dq3.substring(1))
                                    : dqCol == Integer.parseInt(dq3))) {
                                System.err.printf("DQ3 gk=%s qpk=%.9f gj=%s side=%d dqRow=%d weight=%.9f"
                                        + " signed=%.9f%n", gk.getId(), qpk, gj.getId(), side, dqRow,
                                        weight, weight * (side == 0 ? 1.0 : -1.0));
                            }
                            out.add(new int[] {dqCol, side, dqRow, 3, kIdx, jIdx});  // elem = the DUMMY_Q var column
                            weights.add(weight * (side == 0 ? 1.0 : -1.0));
                            continue;
                        }
                        throw new PowsyblException("GPU AC solver: DISTR_Q controller " + gj.getId()
                                + " has an unsupported branch " + brj.getId() + " reactive contribution");
                    }
                }
            }
        }
    }

    /**
     * Build the LINEAR DISTR_RHO contributions (ratio distribution when several transformers control one
     * bus). A controller branch {@code Gk} with an active DISTR_RHO equation gets, on its BRANCH_RHO1 row,
     * {@code DISTR_RHO_k = (1/count)·Σ_j r1_j − r1_k}, i.e. each controller {@code Gj}'s ratio column
     * contributes with coefficient {@code 1/count − (j==k ? 1 : 0)}, where {@code count} is the number of
     * enabled controllers. The free controller (no DISTR_RHO) carries the pilot's Q balance via qRow.
     */
    private static void buildDistrRhoContributions(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                   int[] branchRhoCol, List<int[]> out, List<Double> coefs) {
        for (LfBus pilot : network.getBuses()) {
            // ACTIVATION-INDEPENDENT (plan-group-member-dynamics A1): pack the ALL-ENABLED
            // configuration of every group whose DISTR_RHO equations EXIST — never key on
            // isTransformerVoltageControlled()/getColumn(), which are false/-1 at a control-OFF
            // base (AFTER_GENERATOR) and silently emptied the pack (the e82290f audit). The FREE
            // member is the FIRST enabled one — OLF's updateRemoteVoltageControlEquations pick and
            // computeQRowAlt's, so its row carries the relocated pilot Q, never DISTR_RHO. Rows
            // inactive in a scenario are identity-wiped by the row-mode pass (superset principle).
            TransformerVoltageControl tvc = pilot.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (tvc == null) {
                continue;
            }
            List<LfBranch> controllers = tvc.getMergedControllerElements().stream()
                    .filter(c -> !c.isDisabled())
                    .filter(c -> es.getEquation(c.getNum(), AcEquationType.DISTR_RHO).isPresent())
                    .toList();
            long count = controllers.size();
            for (int k = 1; k < controllers.size(); k++) {   // k = 0 is the FREE member: no DISTR_RHO row
                LfBranch gk = controllers.get(k);
                int drRow = branchRhoCol[gk.getNum()];
                for (LfBranch gj : controllers) {
                    double coef = 1.0 / count - (gj == gk ? 1.0 : 0.0);
                    out.add(new int[] {drRow, branchRhoCol[gj.getNum()]});
                    coefs.add(coef);
                }
            }
        }
    }

    /**
     * DISTR_SHUNT_B linear-distribution rows — the shunt-susceptance analogue of
     * {@link #buildDistrRhoContributions}. For each shunt-voltage-controlled pilot bus, every controller shunt
     * with an ACTIVE DISTR_SHUNT_B equation gets a row {@code b_k = (1/count)·Σ_j b_j} rewritten as
     * {@code 0 = (1/count − 1)·b_k + Σ_{j≠k} b_j/count} over the {@code count} enabled controllers. The free
     * controller (no DISTR_SHUNT_B) carries the pilot's Q balance via qRow.
     */
    private static void buildDistrShuntBContributions(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                      int[] shuntBCol, List<int[]> out, List<Double> coefs) {
        for (LfBus pilot : network.getBuses()) {
            ShuntVoltageControl svc = pilot.getShuntVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> pilot.isShuntVoltageControlled())
                    .orElse(null);
            if (svc == null) {
                continue;
            }
            List<LfShunt> controllers = svc.getMergedControllerElements();
            long count = controllers.stream().filter(c -> !c.isDisabled()).count();
            for (LfShunt gk : controllers) {
                if (GpuAcRowIndexing.row(es, gk.getNum(), AcEquationType.DISTR_SHUNT_B) < 0) {
                    continue;                                // the free controller carries the pilot's Q, no DISTR_SHUNT_B
                }
                int dbRow = shuntBCol[gk.getNum()];
                for (LfShunt gj : controllers) {
                    if (gj.isDisabled()) {
                        continue;
                    }
                    double coef = 1.0 / count - (gj == gk ? 1.0 : 0.0);
                    out.add(new int[] {dbRow, shuntBCol[gj.getNum()]});
                    coefs.add(coef);
                }
            }
        }
    }

    /**
     * The transformers that participate in a ratio-distribution (DISTR_RHO) group. Outaging one of these
     * REFORMS the {@code 1/count} coefficients (OLF redistributes the ratio across the remaining controllers),
     * which the fixed batch pattern cannot represent — so the batched N-1 path rejects such a contingency to
     * the CPU. Returns the controllers of every transformer-voltage-controlled pilot whose group has more than
     * one enabled member (a single local controller is item 4, not DISTR_RHO, and is mask-safe).
     */
    public static Set<LfBranch> distrRhoControllerBranches(LfNetwork network,
                                                           EquationSystem<AcVariableType, AcEquationType> es) {
        Set<LfBranch> controllers = new java.util.HashSet<>();
        for (LfBus pilot : network.getBuses()) {
            // NOTE: keyed on the DISTR_RHO equations EXISTING, NOT on isTransformerVoltageControlled()
            // — at an AFTER_GENERATOR base the control is DISABLED and that predicate is false, which
            // silently emptied this set and let a member outage batch with the coupling missing
            // (measured 3.7e-3 kV wrong, 2026-09-06).
            TransformerVoltageControl tvc = pilot.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (tvc == null) {
                continue;
            }
            List<LfBranch> group = tvc.getMergedControllerElements();
            // EXISTENCE-keyed (row()/getColumn() is -1 for INACTIVE equations — the same trap):
            // a shared group's equations exist whether or not the control is on at base.
            boolean distrRho = group.stream().filter(c -> !c.isDisabled())
                    .filter(c -> es.getEquation(c.getNum(), AcEquationType.DISTR_RHO).isPresent())
                    .count() > 1;
            if (distrRho) {
                controllers.addAll(group);
            }
        }
        return controllers;
    }
}
