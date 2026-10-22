/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CouplerTreeReHoming;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.gpu.GeneratorVoltageControlRoles;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Computes the per-scenario {@link GenOutage} device changes a generator/load contingency applies
 * (mirrors {@code LfContingency.processLostGenerators}), or signals a CPU fallback by returning
 * {@code null}. Groups the lost generators/loads by bus, sums each bus's target and distributed-slack
 * reweight deltas, and rejects (null) the shapes the on-device retain math cannot hold: the
 * slack/reference bus, a remote / distributed (DISTR_Q) voltage controller, more than one
 * distributed-slack-participant bus, or a factor reweight when the slack itself participates.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchInjectionOutages {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchInjectionOutages.class);

    private GpuBatchInjectionOutages() {
    }

    /** OLF's AbstractLfBus.getLimitQ: a bus's reactive aggregate takes each generator's LIMIT when it
     *  controls voltage (or remote reactive power) and its TARGET Q otherwise — the aggregate is NOT a
     *  plain sum of getMinQ()/getMaxQ(). A fixed-Q generator lost from a bus its siblings keep in
     *  voltage control therefore takes its targetQ off the aggregate, not its limit; charging the limit
     *  instead moved rte6515 GEN-2899 by 0.32 kV and three violations, on a bus whose own limit is
     *  1.0 pu against a bus aggregate of 0.997. */
    private static double qContrib(LfGenerator g, java.util.function.ToDoubleFunction<LfGenerator> limit) {
        return g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE
                || g.getGeneratorControlType() == LfGenerator.GeneratorControlType.REMOTE_REACTIVE_POWER
                ? limit.applyAsDouble(g) : g.getTargetQ();
    }

    /** The bus's own slot in the device's packed DISTR_Q member tables, or -1 when it is not a member
     *  of any group. Used both to locate the group for target seeding and to decide whether a load
     *  loss on it is expressible as a per-scenario member targetQ override. */
    static int distrQMemberSlot(GpuAcData gpuData, LfBus bus) {
        int[] busNums = gpuData.distrQMembers().busNum();
        if (busNums == null) {
            return -1;
        }
        for (int m = 0; m < busNums.length; m++) {
            if (busNums[m] == bus.getNum()) {
                return m;
            }
        }
        return -1;
    }

    /** The row mode a surviving member of a DEAD voltage control gets on its BUS_V row. A DISTR_Q
     *  member's row must be mode 4 ("hosts its own reactive balance"): fillDistrQBatch skips every
     *  contribution whose F row is mode 4 and the member's own q rides its alternate dest, which is
     *  keyed on mode 4. Mode 1 left the DISTR_Q family writing into the row on top of nothing that
     *  balanced the bus: rte6515 BUS-1055_BBS_1 (pilot VL-1055_0 islanded, merged section VL-1055_1
     *  surviving) - OLF's BUS_TARGET_Q at the survivor off by 0.039 pu, 1.3 kV. A survivor that is
     *  not a packed DISTR_Q member keeps the plain reactive row. */
    static int deadControlRowMode(GpuAcData gpuData, LfBus survivor) {
        int slot = distrQMemberSlot(gpuData, survivor);
        if (GpuTrace.QUALIFY_TRACE.isOn()) {
            System.err.println("QUALIFY_TRACE deadControlRowMode survivor=" + survivor.getId() + " distrQMemberSlot=" + slot
                    + " -> mode " + (slot >= 0 ? 4 : 1));
        }
        return slot >= 0 ? 4 : 1;
    }

    /** RENORMALIZE through an ISLAND: the group's DISTR_Q controllers the contingency puts in another
     *  synchronous component stop controlling, and their keys leave the renormalization exactly as a
     *  lost generator's do. The qPct override channel (distrQWeightOverrides) carries the survivors'
     *  WEIGHTS; this carries the two things a generator outage also seeds and this path never did:
     *  the survivors' DISTR_Q row TARGETS renormalized over the members keeping their keys (OLF's
     *  qPct_k * sum_j targetQ_j - targetQ_k, summed over every merged controller as AcTargetVector
     *  does), and one inactive-member entry per islanded packed member so updateDistrQTargetsBatch
     *  keeps it out of the denominator in the reactive-limits passes. Without the target the device
     *  solved the renormalized weights against the BASE target: invisible while the pilot carries no
     *  reactive load (target 0 either way), 8.5e-2 kV on rte6515 BUS-1359_BBS_2 whose pilot has one.
     *  Empty when no islanded controller is a packed member. */
    static List<GenOutage> islandedDistrQEntries(GpuAcData gpuData, List<LfBus> islandedControllers,
                                                java.util.Map<Integer, int[]> relocatedDummyQRows,
                                                java.util.Set<Integer> outagedZeroImpedanceBranchNums) {
        return islandedDistrQEntries(gpuData, islandedControllers, relocatedDummyQRows, outagedZeroImpedanceBranchNums, null);
    }

    /** {dq member -> subgroup label} from the plan's flat {member, label} pairs; null when there are none. */
    static java.util.Map<Integer, Integer> labelsOf(int[] subPairs) {
        if (subPairs == null || subPairs.length == 0) {
            return null;
        }
        java.util.Map<Integer, Integer> out = new java.util.HashMap<>();
        for (int p = 0; p < subPairs.length; p += 2) {
            out.put(subPairs[p], subPairs[p + 1]);
        }
        return out;
    }

    /** @param labelOf tasks #71: a split group's subgroup labels — a member's renormalized target sums the keys
     *  of ITS subgroup only, exactly as the kernel's updateDistrQTargetsBatch will maintain it. */
    static List<GenOutage> islandedDistrQEntries(GpuAcData gpuData, List<LfBus> islandedControllers,
                                                java.util.Map<Integer, int[]> relocatedDummyQRows,
                                                java.util.Set<Integer> outagedZeroImpedanceBranchNums,
                                                java.util.Map<Integer, Integer> labelOf) {
        List<GenOutage> out = new ArrayList<>();
        int[] mBusNum = gpuData.distrQMembers().busNum();
        if (mBusNum == null || islandedControllers.isEmpty()) {
            return out;
        }
        java.util.Set<Integer> inactive = new java.util.LinkedHashSet<>();
        for (LfBus b : islandedControllers) {
            int m = distrQMemberSlot(gpuData, b);
            if (m >= 0) {
                inactive.add(m);
            }
        }
        int[] groupStart = gpuData.distrQMembers().groupStart();
        int[] groupCount = gpuData.distrQMembers().groupCount();
        double[] mKey = gpuData.distrQMembers().key();
        byte[] mHasRow = gpuData.distrQMembers().hasRow();
        int[] mRow = gpuData.distrQMembers().row();
        LfBus any = islandedControllers.get(0);
        java.util.Set<Integer> groupsDone = new java.util.HashSet<>();
        for (int im : inactive) {
            int gs = groupStart[im];
            int gc = groupCount[im];
            int[] rows = new int[0];
            double[] values = new double[0];
            if (groupsDone.add(gs)) {                        // the group's targets once, on its first entry
                java.util.Map<Integer, Double> keySumByLabel = new java.util.HashMap<>();
                for (int m = gs; m < gs + gc; m++) {         // live keys over the whole merged group, per subgroup
                    int lbl = labelOf != null ? labelOf.getOrDefault(m, 0) : 0;
                    /** OLF's {@code updateReactiveKeys} rule, plus THIS scenario's loss: a controller that is
                     *  disabled, has its voltage control off, or is the member the outage deactivates takes no share
                     *  of the group's reactive dispatch, and its key leaves the renormalization denominator. */
                    double kk = !inactive.contains(m) && memberKeepsKey(any, mBusNum[m], m, -1) ? mKey[m] : 0;
                    keySumByLabel.merge(lbl, kk, Double::sum);
                }
                List<Integer> r = new ArrayList<>();
                List<Double> v = new ArrayList<>();
                for (int m = gs; m < gs + gc; m++) {
                    if (mHasRow[m] == 0 || mRow[m] < 0 || inactive.contains(m)) {
                        continue;                            // an islanded member's own row is frozen
                    }
                    // ...and a member SWITCHED TO PQ in the base has no DISTR_Q equation: its row hosts its
                    // own reactive balance at the frozen limit, which OLF never rewrites (see the
                    // generator-outage loop below for the measurement - this is the loop a BRANCH
                    // outage that islands a member reaches, network.xiidm/IZUZ included).
                    if (!any.getNetwork().getBus(mBusNum[m]).isGeneratorVoltageControlEnabled()) {
                        continue;
                    }
                    double keySum = keySumByLabel.getOrDefault(labelOf != null ? labelOf.getOrDefault(m, 0) : 0, 0.0);
                    double qpk = keySum == 0 || !memberKeepsKey(any, mBusNum[m], m, -1) ? 0 : mKey[m] / keySum;
                    // distrQMemberRow is STRUCTURAL: under the coupler-tree relocation a child controller's
                    // reactive row is its parent coupler's DUMMY_Q. A scenario that OPENS that coupler fixes
                    // the dummy with a DUMMY_TARGET identity and OLF's DISTR_Q sits on the bus's own row
                    // instead (read off OLF's post-contingency dump for the middle-section fixture:
                    // distr_q5 ACTIVE with keys 0.5/0.5, bus_target_q[5] inactive). Seeding the
                    // renormalized target on the dummy left the live row at its BASE target, so the device
                    // solved the renormalized WEIGHTS against a target built from the base keys: measured
                    // -13.3334 MVAr where OLF has -20.0001, one third of the pilot's 40 MVAr load instead
                    // of one half, and 0.32 kV on every bus behind it.
                    r.add(CouplerTreeReHoming.scenarioReactiveRowOfStructuralRow(mRow[m], relocatedDummyQRows, outagedZeroImpedanceBranchNums));
                    // targetQ over THIS member's peers — its own control's elements, as AcTargetVector sums them
                    double sumTargetQ = 0;
                    for (int t = gpuData.distrQMembers().targetPeerStart()[m]; t < gpuData.distrQMembers().targetPeerStart()[m + 1]; t++) {
                        sumTargetQ += baseTargetQ(any, mBusNum[gpuData.distrQMembers().targetPeer()[t]]);
                    }
                    v.add(qpk * sumTargetQ - baseTargetQ(any, mBusNum[m]));
                }
                rows = r.stream().mapToInt(Integer::intValue).toArray();
                values = v.stream().mapToDouble(Double::doubleValue).toArray();
            }
            out.add(new GenOutage(-1, 0, -1, 0, false, false, -1, 0, 0, 0, 0, -1, 0, 0, false,
                    java.util.Set.of(), 0, 0, 0, im, new int[0], rows, values, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, false, -1, 0, 0));
        }
        return out;
    }

    /** A member's targetQ in the BASE state these seeds describe: a member switched to PQ in the base IS frozen
     *  at its limit there. The device table holds the RECOMPUTED value instead (what the kernel needs once the
     *  member switches back to PV), so the seeds read the bus. */
    private static double baseTargetQ(LfBus anyMember, int memberBusNum) {
        return anyMember.getNetwork().getBus(memberBusNum).getTargetQ();
    }

    private static boolean memberKeepsKey(LfBus anyMember, int memberBusNum, int member, int inactiveMember) {
        if (member == inactiveMember) {
            return false;
        }
        LfBus mb = anyMember.getNetwork().getBus(memberBusNum);
        return mb != null && !mb.isDisabled() && mb.isGeneratorVoltageControlEnabled();
    }

    /** Merged per-BUS outages for a MULTI-generator contingency: gens grouped by bus, each bus's
     *  targets/DS deltas summed. Stage-1 constraint: at most ONE distributed-slack-participant bus
     *  per contingency (the native DS override carries one entry per scenario) — more, or any
     *  per-bus disqualification, returns null and the contingency routes to the CPU. */
    static GenOutage[] computeInjectionOutages(EquationSystem<AcVariableType, AcEquationType> es, GpuAcData gpuData, boolean distributedSlack,
                                               double distributedSlackRawFactorSum, boolean slackParticipates,
                                               com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters distributedSlackParameters,
                                               List<LfGenerator> gens, List<LoadLoss> loads, int[][] busRows) {
        java.util.Map<LfBus, List<LfGenerator>> byBus = new java.util.LinkedHashMap<>();
        java.util.Map<LfBus, List<LoadLoss>> loadsByBus = new java.util.LinkedHashMap<>();
        for (LfGenerator g : gens) {
            if (g.getBus() == null) {
                return null;
            }
            byBus.computeIfAbsent(g.getBus(), b -> new ArrayList<>()).add(g);
        }
        for (LoadLoss l : loads) {
            loadsByBus.computeIfAbsent(l.bus(), b -> new ArrayList<>()).add(l);
            byBus.computeIfAbsent(l.bus(), b -> new ArrayList<>());   // load-only bus: empty gen list
        }
        GenOutage[] out = new GenOutage[byBus.size()];
        int k = 0;
        int nDistributedSlackOverrideBuses = 0;
        for (java.util.Map.Entry<LfBus, List<LfGenerator>> e : byBus.entrySet()) {
            GenOutage go = computeBusGenOutage(es, gpuData, distributedSlack, distributedSlackRawFactorSum, slackParticipates, distributedSlackParameters, e.getKey(), e.getValue(),
                    loadsByBus.getOrDefault(e.getKey(), List.of()), busRows);
            if (go == null) {
                return null;
            }
            if (go.distributedSlackOverrideBusIndex() >= 0 && ++nDistributedSlackOverrideBuses > 1) {
                LOGGER.debug("Multi-injection contingency touches several DS-participant buses — CPU fallback");
                return null;
            }
            out[k++] = go;
        }
        // A pilot whose balance target moves while the SAME contingency changes a control (a member to PQ, a member
        // deactivated, the group's control killed or its pilot row flipped): the move rides the carrier the fill
        // picks, and how it composes with a re-picked carrier or a pilot row that becomes a balance has not been
        // measured. Route it.
        boolean movesPilotQ = false;
        boolean changesControl = false;
        for (GenOutage go : out) {
            movesPilotQ |= go.pilotQGroup() >= 0;
            changesControl |= go.toPq() || go.deactivatesDistrQMember() || go.killsPilotVoltageIdentity()
                    || go.flipsPilotRowWithoutEnabledController();
        }
        if (movesPilotQ && changesControl) {
            return null;
        }
        // Two buses of the SAME remote group deactivated by one contingency: each bus's seeded
        // DISTR_Q targets renormalize over its own loss alone, so neither accounts for the other and
        // both are wrong. The kernel's mask handles the matrix side for any number of them; only the
        // plan-time seed is per-bus. Rather than make the seed a group-wide computation for a shape
        // rte6515 never produces (its generator contingencies are single-generator), route it.
        for (int a = 0; a < out.length; a++) {
            if (!out[a].deactivatesDistrQMember()) {
                continue;
            }
            int ga = gpuData.distrQMembers().groupStart()[out[a].inactiveDistrQMember()];
            for (int b = a + 1; b < out.length; b++) {
                if (out[b].deactivatesDistrQMember()
                        && gpuData.distrQMembers().groupStart()[out[b].inactiveDistrQMember()] == ga) {
                    return null;
                }
            }
        }
        return out;
    }

    /** The entries whose balance rows the scenario's structure moves HOME: a bus whose balances the coupler-tree
     *  relocation put on a coupler's DUMMY rows gets them back on its own rows when that coupler opens or leaves the
     *  tree (the chunk solver fixes the dummies with identities and the kernel's stock-placement fallback brings the
     *  equations home), so a target the plan wrote at the relocated row would land on an identity and be lost.
     *  Measured on network.xiidm DBIG: a section loss that opens the coupler DBIS while both sides stay energized and
     *  takes a 20 MW load off the child DBIE_1 - the override sat on the DUMMY_P row, the load stayed in the balance,
     *  20.47 MW of distributed active power and 0.374 kV against the CPU, while the root-side twin DBIF (the same load
     *  size on DBIE_0, the tree's root) matched at 2e-10 kV. The DS participant index ({@code distributedSlackOverrideBusIndex}) keeps naming
     *  the structural row: the chunk solver re-points that participant's row the same way. {@code toQ} equal to the
     *  entry's own V row becomes "no relocated Q row" (-1), the entry's Q target then goes to {@code vRow}. */
    public static GenOutage[] rowsHome(GenOutage[] gos, int fromP, int toP, int fromQ, int toQ) {
        if (gos == null || gos.length == 0) {
            return gos;
        }
        GenOutage[] out = null;
        for (int k = 0; k < gos.length; k++) {
            GenOutage go = gos[k];
            boolean p = fromP >= 0 && toP >= 0 && go.pRow() == fromP;
            boolean q = fromQ >= 0 && toQ >= 0 && go.qTargetRow() == fromQ;
            if (!p && !q) {
                continue;
            }
            if (out == null) {
                out = gos.clone();
            }
            int newP = p ? toP : go.pRow();
            int newQ = q ? (toQ == go.vRow() ? -1 : toQ) : go.qTargetRow();
            out[k] = go.withBalanceRows(newP, newQ);
            if (GpuTrace.QUALIFY_TRACE.isOn()) {
                System.err.printf("QUALIFY_TRACE injection rows home: pRow %d -> %d, qTargetRow %d -> %d%n",
                        go.pRow(), newP, go.qTargetRow(), newQ);
            }
        }
        return out == null ? gos : out;
    }

    /** The bus's generator voltage control as the GROUP sees it: the MAIN control of a zero-impedance merge.
     *  A section's own control is DEPENDENT there, and asking it reads a one-member group (its own controller)
     *  with no survivors - which refused 285 of rte6515's full-list generator outages as the last controller
     *  (members=1 alive=0). See the merged-voltage-control note on GeneratorVoltageControlRoles.mainGeneratorVoltageControl. */
    private static java.util.Optional<com.powsybl.openloadflow.network.GeneratorVoltageControl> mainVcOf(LfBus bus) {
        return java.util.Optional.ofNullable(
                GeneratorVoltageControlRoles.mainGeneratorVoltageControl(bus));
    }

    private static GenOutage computeBusGenOutage(EquationSystem<AcVariableType, AcEquationType> es, GpuAcData gpuData, boolean distributedSlack,
                                                 double distributedSlackRawFactorSum, boolean slackParticipates,
                                                 com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters distributedSlackParameters, LfBus bus,
                                                 List<LfGenerator> lost, List<LoadLoss> lostLoads, int[][] busRows) {
        int busNum = bus.getNum();
        // The slack (reference) bus solves without a P balance: OLF drops its BUS_TARGET_P, and its scheduled P is read
        // only by the distributed-slack check (the slack's mismatch = its P minus that target). A LOAD lost there moves
        // that target and nothing else on the P side, which the device carries per scenario in the slack shift its
        // distributed-slack kernels already add to the slack target. A generator lost there changes the slack's own
        // participation, and a slack-only distribution reports what it absorbed through that same accumulator: both
        // stay on the CPU.
        boolean slackBus = es.getEquation(busNum, AcEquationType.BUS_TARGET_PHI).isPresent();
        if (slackBus && (!lost.isEmpty() || distributedSlack && gpuData.distributedSlack().busPhiRow().length == 0 && gpuData.distributedSlack().slackPhiRow() >= 0)) {
            return null;
        }
        // DISTR_Q controller bus (remote/distributed VC): the qPct weights are BAKED into the
        // equation system at creation — the CPU does not rebuild them on a gen loss either. When the
        // bus KEEPS another voltage-controlling generator the group is unchanged on both arms and the
        // outage is a plain P-target drop (no PV->PQ flip, no Q override — the distribution absorbs);
        // losing the bus's LAST controller restructures the group -> CPU.
        boolean distrQBus = es.getEquation(busNum, AcEquationType.DISTR_Q).isPresent();
        DistrQBusFate fate = distrQBus ? distrQBusFate(es, gpuData, bus, lost, lostLoads) : NOT_DISTR_Q;
        if (fate == null) {
            return null;
        }
        boolean vcSurvives = fate.vcSurvives();
        boolean deadRemoteMember = fate.deadRemoteMember();
        int deadPilotVRow = fate.deadPilotVRow();
        double deadPilotQTarget = fate.deadPilotQTarget();
        if (bus.hasGeneratorsWithSlope()) {               // slope-SVC bus outage: the mode-2 row's fate on a
            return null;             // gen loss is not modeled — CPU
        }
        if (distributedSlackParameters.isLoadBased() && !lostLoads.isEmpty()) {   // a lost/shifted load changes the LOAD
            return null;   // participation factors — not modeled: CPU
        }
        java.util.Set<LfGenerator> lostSet = new java.util.HashSet<>(lost);
        // The bus's P-BALANCE row, which under a coupler tree is its parent coupler's DUMMY_P column,
        // NOT its own BUS_PHI column (GpuAcDataExtractor#busPowerRows). Resolving it from the variable
        // set instead put every injection override of a relocated bus on a coupling row whose target is
        // 0, so the absolute override was a no-op and the lost injection stayed in the balance — and it
        // also made the distributedSlackBusPhiRow lookup below miss, dropping the anchor shift. network.xiidm's loads:
        // 4183 of 7494 wrong totals, worst 148 MW, 12640 of 13338 buses off by up to 1.03 kV.
        int pRow = slackBus ? -1 : busRows[0][busNum];       // the slack's row is its angle reference
        int vRow = es.getVariableSet().getVariable(busNum, AcVariableType.BUS_V).getRow();
        if (pRow < 0 && !slackBus || vRow < 0) {
            return null;
        }
        // Is this bus a distributed-slack participant? Matters for BOTH shapes: lost generators
        // leave the participation (factor/headroom reweight), AND pure injection deltas (load
        // losses, redispatch actions) shift the row target — which the retain/waterfill accounting
        // would misread as already-distributed slack power unless distributedSlackBusBaseTargetP shifts with it
        // (measured: a +60 MW redispatch cancelled to a no-op before the base-shift fix).
        double preLostLoadP = lostLoads.stream().mapToDouble(LoadLoss::p).sum();
        int distributedSlackOverrideBusIndex = -1;
        if (distributedSlack && (!lost.isEmpty() || preLostLoadP != 0)) {      // the run distributes: its participant reweights
            int[] distributedSlackBusPhiRow = gpuData.distributedSlack().busPhiRow();
            for (int i = 0; i < distributedSlackBusPhiRow.length; i++) {
                if (distributedSlackBusPhiRow[i] == pRow) {
                    distributedSlackOverrideBusIndex = i;
                    break;
                }
            }
        }
        boolean lostParticipation = lost.stream().anyMatch(g -> distributedSlackParameters.factor(g) != 0);
        if (distributedSlackParameters.isRemainingMargin() && distributedSlackOverrideBusIndex >= 0 && lostParticipation) {
            return null;       // RM margins reweight per scenario: not modeled — CPU
        }
        if (!distrQBus && bus.getGeneratorVoltageControl().map(voltageControl -> voltageControl.getControlledBus() != null
                && voltageControl.getControlledBus().getNum() != busNum).orElse(false)) {
            return null;   // (pilot-Q remap, no DISTR_Q row at this bus):
        }                                                        // its q accounting is not modeled — CPU
        double lostP = lost.stream().mapToDouble(LfGenerator::getTargetP).sum();
        double lostLoadP = lostLoads.stream().mapToDouble(LoadLoss::p).sum();
        double lostLoadQ = lostLoads.stream().mapToDouble(LoadLoss::q).sum();
        double pTarget = bus.getTargetP() - lostP + lostLoadP;   // shedding load RAISES the P target
        boolean busPv = bus.isGeneratorVoltageControlEnabled();
        // The bus is the CONTROLLED bus of an enabled, non-local generator voltage control: its V row
        // holds BUS_TARGET_V that some OTHER bus's generators maintain. Same predicate the extraction
        // uses to decide a pilot's row layout.
        boolean remotelyControlled = bus.getGeneratorVoltageControl()
                .filter(c -> c.getMergeStatus() == com.powsybl.openloadflow.network.VoltageControl.MergeStatus.MAIN)
                .filter(c -> bus.isGeneratorVoltageControlled() && !c.isLocalControl())
                .isPresent();
        boolean lostControls = lost.stream()
                .anyMatch(g -> g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
        boolean otherController = bus.getGenerators().stream()
                .anyMatch(g -> !lostSet.contains(g) && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
        // vcSurvives: the group's other controllers still regulate this bus, so it keeps its voltage
        // target and must NOT be flipped to its own reactive balance (measured: doing so moves the
        // SURVIVING controllers — rte6515 GEN-1522, 3.157 kV at VL-1525).
        boolean toPq = busPv && lostControls && !otherController && !vcSurvives;
        // distrQBus with keepsController: otherController is true by the guard above, so toPq stays
        // false and qOverride below stays false (busPv true) — the group's distribution re-balances
        // the lost reactive share on both arms; only the P target drop and DS reweight apply.
        // The bus Q TARGET must shed the lost gens' targetQ whenever the V row solves the reactive
        // equation in the scenario: on the PV->PQ flip AND on an ALREADY-PQ bus losing fixed-Q
        // generation (task #21 — the flip-only version silently kept the lost Q injected; measured
        // 0.27 kV voltage divergence on ieee14 for a 15-MVAr fixed-Q gen loss).
        // ...and `!busPv` alone is the WRONG proxy for that. busPv asks "is this bus a CONTROLLER",
        // but a bus REMOTELY controlled by a generator group elsewhere (a pilot) owns no generators, so
        // busPv is false while its V row still holds BUS_TARGET_V — it does not solve a reactive
        // equation at all. Stamping a Q target over that identity leaves mode 0 with target 0 against a
        // state of 1.111 pu, and the Newton then drives the bus's VOLTAGE toward zero: measured on
        // rte6515 LOAD-5958, the solve oscillates (1.111 -> 0.995 -> 0.880 -> 0.765 -> 1.598 -> 1.410
        // ...) and exhausts its iterations instead of converging.
        // REMOTELY is the operative word, and the narrower test is load-bearing: a LOCALLY controlled
        // bus that the base switched PV->PQ also has busPv false, and its V row DOES solve the reactive
        // equation, so it must keep shedding the lost Q. Suppressing it there moves
        // deviceBusSwitchedToPqLosingReactiveLoadMatchesCpu's B4 by 1.0e-08 kV.
        boolean qOverride = toPq || !busPv && !remotelyControlled;
        // ...but a remotely controlled bus still HAS a reactive balance - OLF keeps its BUS_TARGET_Q, generation
        // minus load like any PQ bus - and the extraction moved it onto the relocation group's CARRIER (alternateQRow =
        // -2-g: the free member's row), whose target the device captures from the BASE targets at solve start
        // (capturePilotQBatch) and hands to whichever member carries the balance later. Losing reactive load (or a
        // fixed-Q generator) at the pilot moves that target, and neither channel could say so: measured on the
        // remote-control fixture, a 20 MVar load lost at the pilot stayed in its balance and every controller bus
        // ended 3.7e-2 kV off OLF. The group and the move travel to the carrier's first-solve target and to the
        // device's captured value (setBatchPilotQDeltas).
        int pilotQGroup = -1;
        double pilotQDelta = 0;
        boolean purePilot = remotelyControlled
                && mainVcOf(bus).map(voltageControl -> !voltageControl.getMergedControllerElements().contains(bus)).orElse(false);
        if (purePilot) {                                     // a pilot among its controllers has no balance of its
            double lostGenQ = lost.stream().mapToDouble(LfGenerator::getTargetQ).sum();   // own (DISTR_Q member)
            if (lostLoadQ != 0 || lostGenQ != 0) {
                int alt = busRows.length > 3 ? busRows[3][busNum] : -1;
                if (alt > -2) {                              // the balance is not on a carrier: no channel for it
                    return null;
                }
                pilotQGroup = -2 - alt;
                pilotQDelta = lostLoadQ - lostGenQ;          // targetQ = generation - load
            }
        }
        double qTarget = 0;
        if (qOverride) {
            // A bus the BASE loadflow switched to PQ at a reactive limit has its generation target Q FROZEN at
            // that limit, and OLF keeps that value as long as a VOLTAGE-type generator survives:
            // processLostGenerators calls setGeneratorVoltageControlEnabledAndRecomputeTargetQ only when
            // none does, and invalidateGenerationTargetP leaves a frozen bus frozen. Summing the
            // survivors' own targetQ instead hands the first inner solve a target 1.7 pu off the one
            // OLF solves (measured at rte6515 VL-6003_0: 1.03 vs 2.69 pu) — the reactive-limit pass
            // then makes its switch decisions along a trajectory the CPU never takes. When the last
            // VOLTAGE generator goes, OLF unfreezes and recomputes over what is left, which is what the
            // survivor sum gives (rte6515 GEN-6095/6130/6226/6239, one generator on the bus: keeping the
            // frozen value there injects the dead generator's reactive limit, ~1 kV).
            double genQ = bus.isGenerationTargetQFrozen() && otherController
                    ? bus.getGenerationTargetQ()
                    : bus.getGenerators().stream().filter(g -> !lostSet.contains(g))
                            .mapToDouble(LfGenerator::getTargetQ).sum();
            qTarget = genQ - (bus.getLoadTargetQ() - lostLoadQ);
        }
        // Distributed-slack reweight for the affected bus. The bus's base target P always drops by the lost
        // gen's P. The participation factor / headroom / footroom drop only if the LOST gen itself participated
        // (mirrors the extractor, which sums these over participating gens with a non-zero droop).
        DistributedSlackReweight distributedSlackReweightOut = distributedSlackReweight(distributedSlackOverrideBusIndex, lostP, lostLoadP, lost, distributedSlackParameters,
                distributedSlackRawFactorSum, bus, lostSet);
        if (distributedSlackReweightOut == null) {
            return null;
        }
        double distributedSlackFactorDelta = distributedSlackReweightOut.distributedSlackFactorDelta();
        double distributedSlackBaseTargetPDelta = distributedSlackReweightOut.distributedSlackBaseTargetPDelta();
        double distributedSlackOverrideMaxDelta = distributedSlackReweightOut.distributedSlackOverrideMaxDelta();
        double distributedSlackOverrideMinDelta = distributedSlackReweightOut.distributedSlackOverrideMinDelta();
        // A bus that KEEPS voltage control still carries the lost generators' reactive capability in the
        // device's run-constant aggregate, while OLF's bus.getMinQ()/getMaxQ() sum only the generators
        // that are still there. Name the reactive-limit entry and how much to take off it. (A bus the
        // outage takes to PQ needs nothing: its entry stops being a controller and the limits are never
        // read.) A bus the BASE loadflow already SWITCHED TO PQ at a limit is NOT that case, though it
        // reads busPv=false: its entry is still a controller, its frozen target Q IS the limit, and it can
        // switch back to PV. Measured on rte6515 GEN-6003 — one generator of three lost at a bus switched
        // to PQ, the stale capability injected as Q, and 1.9 kV network-wide.
        int[] reactiveLimitBusNum = gpuData.reactiveLimits().busNum();
        ReactiveLimitDeltas reactiveLimitDeltasOut = reactiveLimitDeltas(reactiveLimitBusNum, bus, toPq, lost, lostLoadQ);
        int reactiveLimitEntry = reactiveLimitDeltasOut.reactiveLimitEntry();
        double minQDelta = reactiveLimitDeltasOut.minQDelta();
        double maxQDelta = reactiveLimitDeltasOut.maxQDelta();
        double loadQDelta = reactiveLimitDeltasOut.loadQDelta();
        double svcBminDelta = reactiveLimitDeltasOut.svcBminDelta();
        double svcBmaxDelta = reactiveLimitDeltasOut.svcBmaxDelta();
        // No generator left at all. The reactive-limit entry must stop being a controller for this
        // scenario: a bus the BASE loadflow had already switched to PQ keeps a seeded limit type so a
        // relieving contingency can switch it back to PV — and with its only generator gone there is
        // nothing to switch back to. OLF simply has no controller there. (Measured on rte6515 GEN-6130:
        // 0.995 kV at that node against 2.2e-4 everywhere else, and one apparent-power violation.)
        boolean allGensLost = bus.getGenerators().stream().allMatch(lostSet::contains);
        // Identity of what was lost, for the reactive-capability CURVE channel: it evaluates each
        // generator's curve at its OWN post-DS active power, which it reconstructs from the bus's row
        // target by participation share. The lost generator's entry has to go, the survivors' shares
        // have to renormalize over what is left, and the step the outage itself puts in the row target
        // is not DS power and must not move any curve. All three need per-generator identity.
        int inactiveDistrQMember = -1;
        if (vcSurvives) {
            int[] mBus = gpuData.distrQMembers().busNum();
            for (int m = 0; mBus != null && m < mBus.length; m++) {
                if (mBus[m] == busNum) {
                    inactiveDistrQMember = m;
                    break;
                }
            }
            if (inactiveDistrQMember < 0) {
                return null;
            }
        }
        // The group's DISTR_Q row targets, renormalized over the survivors exactly as OLF does
        // (updateReactiveKeys zeroes a non-enabled controller's key and renormalizes; AcTargetVector
        // then reads qPct_k * sum_j targetQ_j - targetQ_k). Seeded per scenario because the kernel that
        // maintains these targets runs only inside the reactive-limits phase — see GenOutage.
        int[] distrQTargetRows = new int[0];
        double[] distrQTargetValues = new double[0];
        // The outaged bus's OWN member slot in the device's DISTR_Q tables, so the group can be located
        // even when no controller is deactivated (a pure reactive-load loss on a member bus).
        int ownDistrQMember = distrQMemberSlot(gpuData, bus);
        // targetQ = generation - load, so losing reactive load MOVES this member's own targetQ, and
        // every DISTR_Q target in the group reads it twice (shared sum + own subtracted term).
        // Measured on rte6515 LOAD-1263: OLF's residual is EXACTLY 4.600000000e-02 on the DISTR_Q row
        // of VL-1263_1 = qPct (0.5, two equal-key members) x the pilot's 0.092 pu of lost load Q.
        int distrQTargetQOverrideMember = -1;
        double distrQTargetQOverrideValue = 0;
        if (ownDistrQMember >= 0 && lostLoadQ != 0 && gpuData.distrQMembers().targetQ() != null) {
            distrQTargetQOverrideMember = ownDistrQMember;
            distrQTargetQOverrideValue = gpuData.distrQMembers().targetQ()[ownDistrQMember] + lostLoadQ;
        }
        int distrQAnchor = vcSurvives ? inactiveDistrQMember : ownDistrQMember;
        double distrQOwnTargetQDelta = vcSurvives ? 0 : lostLoadQ;
        DistrQTargets distrQSeed = seededDistrQTargets(gpuData, bus, vcSurvives, inactiveDistrQMember, distrQAnchor, distrQOwnTargetQDelta,
                lostLoadQ);
        distrQTargetRows = distrQSeed.rows();
        distrQTargetValues = distrQSeed.values();
        int[] pilotSurvivorReactiveLimitEntries = vcSurvives ? survivorReactiveLimitEntriesOf(bus, lostSet, reactiveLimitBusNum) : new int[0];
        // The pilot's OWN row, for the runtime flip when every survivor switches to PQ. Only when the outaged
        // bus IS the pilot: that is the case where losing its generators makes the pilot stop being
        // one of its own controllers and ACQUIRE a reactive balance. See
        // GenOutage.flipsPilotRowWithoutEnabledController.
        int pilotVRowWithoutEnabledController = -1;
        double pilotQTargetWithoutEnabledController = 0;
        if (vcSurvives && pilotSurvivorReactiveLimitEntries.length > 0
                && mainVcOf(bus).orElseThrow().getControlledBus() == bus) {
            var pilotV = es.getVariableSet().getVariable(bus.getNum(), AcVariableType.BUS_V);
            if (pilotV != null && pilotV.getRow() >= 0) {
                pilotVRowWithoutEnabledController = pilotV.getRow();
                // generationTargetQ - loadTargetQ. A VOLTAGE generator contributes 0 to
                // generationTargetQ, so losing one leaves this unchanged.
                pilotQTargetWithoutEnabledController = bus.getTargetQ();
            }
        }
        java.util.Set<String> lostGenIds = lost.stream().map(LfGenerator::getId)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
        double lostFactorSum = lost.stream().mapToDouble(distributedSlackParameters::factor).sum();
        // Same relocation as pRow, on the reactive side: under a coupler the bus's Q BALANCE sits on the
        // parent's DUMMY_Q column, so an absolute Q target written at the BUS_V row is a no-op and the
        // lost load's REACTIVE power stays in the system. Only when the extraction actually moved it —
        // a DISTR_Q bus keeps its Q on its own group row and is never relocated (ExtractionColumns).
        int qTargetRow = busRows[2][busNum] >= 0 && busRows[1][busNum] >= 0 && busRows[1][busNum] != vRow
                ? busRows[1][busNum] : -1;
        // A relocated DISTR_Q member the BASE SWITCHED TO PQ has no static Q row (its balance rides the group's rows)
        // but a mode-4 ALTERNATE row - its coupler's DUMMY_Q - that hosts its own balance from the start, with
        // the frozen limit as target. That is the row the override belongs on: OLF recomputes the target over the
        // generators left the moment the last one goes, and nothing on the device rewrites a base-set target
        // otherwise. Written at the bus's own V row (the coupler's ZERO_V, target 0) the override was a no-op:
        // measured on the merged-control fixture as 0.36 kV at b1, row 7 ending at 0.05 against OLF's 0.
        if (qTargetRow < 0 && busRows[2][busNum] >= 0 && busRows.length > 3 && busRows[3][busNum] >= 0
                && busRows[3][busNum] != vRow && allGensLost && reactiveLimitEntry >= 0
                && gpuData.switchedPvToPqInBase(reactiveLimitEntry)) {
            qTargetRow = busRows[3][busNum];
        }
        return new GenOutage(pRow, pTarget, vRow, qTarget, toPq, qOverride, distributedSlackOverrideBusIndex, distributedSlackFactorDelta, distributedSlackBaseTargetPDelta,
                distributedSlackOverrideMaxDelta, distributedSlackOverrideMinDelta, reactiveLimitEntry, minQDelta, maxQDelta, allGensLost, lostGenIds, lostFactorSum,
                lostLoadP - lostP, loadQDelta, inactiveDistrQMember, pilotSurvivorReactiveLimitEntries,
                distrQTargetRows, distrQTargetValues, deadRemoteMember ? GpuRowMode.BUS_TARGET_Q : GpuRowMode.EQUATION,
                deadPilotVRow, deadPilotQTarget, pilotVRowWithoutEnabledController, pilotQTargetWithoutEnabledController,
                distrQTargetQOverrideMember, distrQTargetQOverrideValue, qTargetRow, pilotQGroup, pilotQDelta, slackBus, -1, svcBminDelta, svcBmaxDelta);
    }

    /** Does another controller of the bus's remote voltage-control group keep a VOLTAGE generator after the loss of
     *  {@code lostHere}? Then the group still holds this bus's voltage. */
    private static boolean remoteGroupSurvives(com.powsybl.openloadflow.network.GeneratorVoltageControl voltageControl,
                                               LfBus bus, java.util.Set<LfGenerator> lostHere) {
        List<LfBus> members = voltageControl.getMergedControllerElements();
        long survivors = members.stream()
                .filter(member -> member != bus && member.getGenerators().stream()
                        .anyMatch(g -> !lostHere.contains(g)
                                && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE))
                .count();
        // The group's FREE member — the first enabled controller, whose row does NOT
        // hold a DISTR_Q: it carries the pilot's relocated reactive balance instead,
        // or, for a pilot that is one of its own controllers, nothing at all, since
        // while it controls it has no reactive balance to relocate.
        boolean isFree = members.stream().filter(voltageControl::isControllerEnabled)
                .findFirst().map(m -> m == bus).orElse(false);
        boolean pilotIsController = members.contains(voltageControl.getControlledBus());
        // Losing the FREE member moves the free role, and with it the row OLF puts the
        // pilot's reactive balance on; the mode scan re-picks the carrier, so any number of
        // survivors is fine (measured: rte6515 GEN-5958, 3 members and 2 survivors,
        // 3.8e-11; reproducer deviceRemoteGroupLosingFreeMemberMatchesCpu). That holds for a
        // pilot that IS one of its controllers too: its gen-lost entry opens the pilot-self
        // gate, its raw q lands on the carrier with its own balance as target and the
        // carrier's DISTR_Q is suppressed. Two or more survivors used to route to the CPU
        // on a 0.149 kV measurement, which was the seed-time stamps reading the dynamic
        // limit arrays before anything had filled them on the cold path (carrier target
        // 0 - 0, a PQ peer's target 0 - 0): devicePilotControllerLosingItsOwnGenerator
        // MatchesCpu is exact at 1e-11 since, and rte6515's 50 such outages batch.
        return survivors > 0;
    }

    /** A DISTR_Q controller bus's outage: does its group still hold its voltage, does the group's control die with
     *  it (and then the pilot's V row and reactive target), or null when the shape refuses the contingency. */
    private static DistrQBusFate distrQBusFate(final EquationSystem<AcVariableType, AcEquationType> es, final GpuAcData gpuData,
                                               final LfBus bus, final List<LfGenerator> lost, final List<LoadLoss> lostLoads) {
        boolean vcSurvives = false;
        boolean deadRemoteMember = false;
        int deadPilotVRow = -1;
        double deadPilotQTarget = 0;
        java.util.Set<LfGenerator> lostHere = new java.util.HashSet<>(lost);
        boolean keepsController = bus.getGenerators().stream()
                .anyMatch(g -> !lostHere.contains(g)
                        && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
        // A controller bus that loses its last VOLTAGE generator while the group's OTHER
        // controllers survive does NOT go to PQ: they still hold its voltage, so its BUS_TARGET_V
        // stays. What changes is that it stops participating in the reactive sharing, which the
        // qPct override already expresses on the matrix side — and now on the target side too
        // (inactiveDistrQMember). When NO controller survives the control itself dies, which is a
        // different shape (every member re-solves its own reactive balance) and still routes to
        // the CPU, as does a Q-carrying load loss here.
        // Admit exactly the shape the degeneration argument covers, and no more:
        //  (a) another controller survives, so the group still holds this bus's voltage;
        //  (b) the survivors MAY all switch to PQ: the row of a survivor switched to PQ takes its own
        //      balance (mode 4), and a pilot's switched to PQ too, so nothing is evicted - a runtime reject
        //      used to replay such a scenario on the CPU, for the case where the outaged bus
        //      IS the pilot and its balance rode a survivor's row (GEN-1522: q 0.601 vs OLF's 0);
        //      that case flips the pilot's own row in the solve since (flipsPilotRowWithoutEnabledController),
        //      and the member case was measured exact without the reject (the fixture whose survivor
        //      switches to PQ at runtime, rte6515 GEN-6231#0). Where the survivors keep headroom the degeneration
        //      holds exactly — GEN-1523 at overMaxBy -0.149 and GEN-5958 at -0.900 both land at
        //      ~4e-11 kV;
        //  (c) the pilot carries no reactive load. Lifting this leaves GEN-1264 at 0.134 kV and
        //      GEN-1263 at 0.247 kV, and every INPUT to that row has been verified correct:
        //        weights  - distrQWeightOverrideTable never fires here (nDistrQWeightOverrides=0); the inactive mask supplies them and
        //                   they are exactly right (w=0 from the survivor, w=-1 from the dead bus);
        //        target   - computed AND persisted: final -0.060 against a base of 0.016, mode 1.
        //      The converged state still violates that row by 0.076 (the probe's distrQResid), so
        //      the row as ASSEMBLED is not the equation those inputs describe. The next layer is
        //      what F contributions actually land on a mode-1 row here — not the inputs, which are
        //      settled. Same unidentified cause as LOAD-1263 (0.105 kV), a pure load loss on this
        //      same group, where both buses carry reactive load (-0.092 and -0.060).
        // With qPct renormalized to 1 on the survivors, DISTR_Q (qPct_k·Σq − q_k) on this bus's own
        // row degenerates to q = targetQ: precisely the reactive balance OLF re-activates here.
        vcSurvives = !keepsController && mainVcOf(bus)
                .map(voltageControl -> remoteGroupSurvives(voltageControl, bus, lostHere))
                .orElse(false);
        // The group's control DIES with this bus (no controller anywhere keeps a VOLTAGE
        // generator). OLF then deactivates the pilot's BUS_TARGET_V and re-activates a reactive
        // balance at BOTH the pilot and this member. The device reaches the same layout through
        // the relocation it already has, provided this row goes to mode 4 rather than mode 1:
        // mode 4 means "this row hosts its own Q balance", and it is also what makes the group's
        // mode scan skip the row, so the pilot's balance falls back to the pilot's own row and
        // updatePilotTargetVEquationBatch drops the voltage identity. Needs a pilot that is NOT one of its own
        // controllers — that shape has no relocation table for the fallback to land in.
        boolean controlDies = !keepsController && mainVcOf(bus)
                .map(voltageControl -> voltageControl.getMergedControllerElements().stream()
                                .noneMatch(member -> member.getGenerators().stream()
                                        .anyMatch(g -> !lostHere.contains(g)
                                                && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE))
                        && !voltageControl.getMergedControllerElements().contains(voltageControl.getControlledBus())
                        && voltageControl.getControlledBus() != bus)
                .orElse(false);
        if (controlDies && lostLoads.isEmpty()) {
            deadRemoteMember = true;
            LfBus pilot = mainVcOf(bus).orElseThrow().getControlledBus();
            var pilotV = es.getVariableSet().getVariable(pilot.getNum(), AcVariableType.BUS_V);
            if (pilotV == null || pilotV.getRow() < 0) {
                return null;
            }
            deadPilotVRow = pilotV.getRow();
            deadPilotQTarget = pilot.getTargetQ();
        } else if (!keepsController && !vcSurvives || !lostLoads.isEmpty()) {
            // BOTH halves are load-bearing, and 2026-09-17 they turned out to be ONE root cause.
            // Lifting them on rte6515's full sets: the 9 generator cases land wrong (0.067 up to
            // 3.157 kV at GEN-1522) and LOAD-1263 lands 0.105 kV out.
            //
            // EVERY one of those buses is a PILOT THAT IS ALSO ONE OF ITS CONTROLLERS
            // — the shape ExtractionColumns deliberately
            // excludes from the Q-relocation table, because while it controls it has NO Q balance
            // of its own to relocate (its q feeds the group's DISTR_Q rows, and a table entry
            // would double-count it). ReactiveRelocation.distrQRow already parks its DISTR_Q on the
            // first enabled member's BUS_V row while its own v row holds BUS_TARGET_V.
            //
            // The moment such a pilot stops being a controller it ACQUIRES a Q balance, which has
            // to take that relocated row — and there is no table entry to carry it. The qualifier's
            // qPct rediff renormalizes the weights but cannot express that. Note the bus does NOT
            // become PQ when other controllers survive (vcCtlAlive > 0): they still hold its
            // voltage, so its BUS_TARGET_V must stay — which is why toPq, computed from the bus's
            // OWN generators, is the wrong instrument here.
            //
            // Closing this means giving pilot-is-controller buses a relocation table that is INERT
            // in the base and activates per scenario. That is plan B4 extended into the case it
            // explicitly excludes, not a gate tweak.
            //
            // If you come back to lift either, diff BUSES (OLF_GPU_SA_BUSDIFF=1). Violation parity
            // is a threshold test: of the 8 wrong generator cases it caught exactly ONE.
            // A PURE reactive-LOAD loss on a packed DISTR_Q member, with the group's control
            // untouched, IS expressible: the member's own targetQ moves (targetQ = generation -
            // load) and every DISTR_Q target in the group follows it through the per-scenario
            // member-targetQ override (setBatchDistrQMemberTargetQOverrides), which the device's
            // target kernel reads instead of the base value. Deliberately narrow — no generator
            // may be lost on the bus in the same scenario, because a frozen generation target Q
            // makes the surviving-generator sum the wrong post-contingency value (see the
            // qOverride note above), and that case has not been measured.
            boolean pureLoadLossOnPackedMember = keepsController && lostHere.isEmpty()
                    && !lostLoads.isEmpty() && distrQMemberSlot(gpuData, bus) >= 0;
            if (!pureLoadLossOnPackedMember) {
                return null;
            }
        }
        return new DistrQBusFate(vcSurvives, deadRemoteMember, deadPilotVRow, deadPilotQTarget);
    }

    /** The fate of a DISTR_Q controller bus's group under an outage (see distrQBusFate). */
    private record DistrQBusFate(boolean vcSurvives, boolean deadRemoteMember, int deadPilotVRow,
                                 double deadPilotQTarget) {
    }

    private static final DistrQBusFate NOT_DISTR_Q = new DistrQBusFate(false, false, -1, 0);

    /** The DISTR_Q row targets an outage seeds in its group: renormalized over the surviving controllers when the
     *  group still holds the bus, or moved by the reactive load the bus loses; empty when neither applies. */
    private static DistrQTargets seededDistrQTargets(final GpuAcData gpuData, final LfBus bus, final boolean vcSurvives,
                                             final int inactiveDistrQMember, final int distrQAnchor, final double distrQOwnTargetQDelta,
                                             final double lostLoadQ) {
        int[] distrQTargetRows = new int[0];
        double[] distrQTargetValues = new double[0];
        if (distrQAnchor >= 0 && (vcSurvives || lostLoadQ != 0)) {
            // Read the DEVICE's own member tables rather than recomputing the group: which member
            // OWNS a DISTR_Q row (the base free member's row carries the pilot's relocated balance
            // instead) and which device row it is were both decided at extraction, and a predicate
            // re-derived here would seed the wrong rows the moment the two disagreed.
            int gs = gpuData.distrQMembers().groupStart()[distrQAnchor];
            int gc = gpuData.distrQMembers().groupCount()[distrQAnchor];
            int[] mBusNum = gpuData.distrQMembers().busNum();
            double[] mKey = gpuData.distrQMembers().key();
            byte[] mHasRow = gpuData.distrQMembers().hasRow();
            int[] mRow = gpuData.distrQMembers().row();
            double keySum = 0;
            for (int m = gs; m < gs + gc; m++) {             // live keys over the whole merged group
                keySum += memberKeepsKey(bus, mBusNum[m], m, inactiveDistrQMember) ? mKey[m] : 0;
            }
            List<Integer> rows = new ArrayList<>();
            List<Double> values = new ArrayList<>();
            for (int m = gs; m < gs + gc; m++) {
                if (mHasRow[m] == 0 || mRow[m] < 0) {
                    continue;
                }
                // A member SWITCHED TO PQ in the base has no DISTR_Q equation: its row hosts its own reactive
                // balance at the frozen limit (targetQ_frozen - loadQ), and OLF's updateReactiveKeys /
                // AcTargetVector never write a share there. Seeding the share formula on it anyway
                // handed the row -targetQ_m (no live key, so qpk = 0): the frozen Q with its sign
                // flipped. Measured on network.xiidm/IZUZ (IZSV_0 islanded, IZTH_0 switched to PQ at MIN_Q in the base):
                // +0.276956953 against OLF's -0.276956953, a 0.5539 pu entry residual, one Newton
                // iteration more than OLF on the first solve, and the slack pass read 0.000845 pu
                // less - the 0.0845 MW the arms reported apart. If the member switches back to PV later, the
                // kernel that maintains these targets (updateDistrQTargetsBatch) writes its share then.
                if (!bus.getNetwork().getBus(mBusNum[m]).isGeneratorVoltageControlEnabled()) {
                    continue;
                }
                double qpk = keySum == 0 || !memberKeepsKey(bus, mBusNum[m], m, inactiveDistrQMember)
                        ? 0 : mKey[m] / keySum;
                rows.add(mRow[m]);
                // targetQ over THIS member's peers — the elements of its own control (see the peer CSR):
                // a member whose control is merged INTO the group sums only its own, as AcTargetVector does.
                double sumTargetQ = 0;
                for (int t = gpuData.distrQMembers().targetPeerStart()[m]; t < gpuData.distrQMembers().targetPeerStart()[m + 1]; t++) {
                    int peer = gpuData.distrQMembers().targetPeer()[t];
                    sumTargetQ += baseTargetQ(bus, mBusNum[peer])
                            + (mBusNum[peer] == bus.getNum() ? distrQOwnTargetQDelta : 0);
                }
                values.add(qpk * sumTargetQ
                        - (baseTargetQ(bus, mBusNum[m]) + (mBusNum[m] == bus.getNum() ? distrQOwnTargetQDelta : 0)));
            }
            distrQTargetRows = rows.stream().mapToInt(Integer::intValue).toArray();
            distrQTargetValues = values.stream().mapToDouble(Double::doubleValue).toArray();
            if (GpuTrace.RL_TRACE.isOn()) {
                StringBuilder sb = new StringBuilder();
                for (int m = gs; m < gs + gc; m++) {
                    sb.append(" m").append(m).append("[bus=").append(mBusNum[m]).append(" hasRow=").append(mHasRow[m])
                            .append(" row=").append(mRow[m]).append(" key=").append(mKey[m])
                            .append(" keeps=").append(memberKeepsKey(bus, mBusNum[m], m, inactiveDistrQMember)).append(']');
                }
                System.err.println("DQSEED bus=" + bus.getId() + " anchor=" + distrQAnchor + " inactive=" + inactiveDistrQMember
                        + " vcSurvives=" + vcSurvives + " keySum=" + keySum + " rows=" + rows + " values=" + values + sb);
            }
        }
        return new DistrQTargets(distrQTargetRows, distrQTargetValues);
    }

    /** DISTR_Q rows and the targets an outage seeds on them. */
    private record DistrQTargets(int[] rows, double[] values) {
    }

    /** The reactive-limit entry of the outaged bus and the reactive capability the outage takes from it: the
     *  generators' Q limits, the lost load's Q, and an SVC's susceptance limits. */
    private static ReactiveLimitDeltas reactiveLimitDeltas(final int[] reactiveLimitBusNum, final LfBus bus, final boolean toPq,
                                                final List<LfGenerator> lost, final double lostLoadQ) {
        int reactiveLimitEntry = -1;
        double minQDelta = 0;
        double maxQDelta = 0;
        for (int i = 0; reactiveLimitBusNum != null && i < reactiveLimitBusNum.length; i++) {
            if (reactiveLimitBusNum[i] == bus.getNum()) {
                reactiveLimitEntry = i;
                break;
            }
        }
        double loadQDelta = 0;
        double svcBminDelta = 0;
        double svcBmaxDelta = 0;
        if (reactiveLimitEntry >= 0 && !toPq) {
            minQDelta = -lost.stream().mapToDouble(g -> qContrib(g, LfGenerator::getMinQ)).sum();
            maxQDelta = -lost.stream().mapToDouble(g -> qContrib(g, LfGenerator::getMaxQ)).sum();
            // the same SVCs ReactiveLimitsPacker.addSvcQuadratic folds into the entry's coefficients
            double v2 = bus.getV() * bus.getV();
            for (LfGenerator g : lost) {
                if (g instanceof com.powsybl.openloadflow.network.LfStaticVarCompensator
                        && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE) {
                    svcBminDelta -= g.getMinQ() / v2;
                    svcBmaxDelta -= g.getMaxQ() / v2;
                }
            }
            // The pack's loadQ is the bus's load target Q as EXTRACTED, and the reactive-limit
            // machinery reads it in two paired forms: the frozen target is limit - loadQ, and the
            // controller's own q is the net injection + loadQ. Shed a Q-carrying load and both go
            // stale. This used to be a CPU gate for a PV bus; the bus the base loadflow had already
            // SWITCHED TO PQ fell through it and was silently wrong instead (rte6515 LOAD-6157, 0.37 kV).
            loadQDelta = -lostLoadQ;
        }
        return new ReactiveLimitDeltas(reactiveLimitEntry, minQDelta, maxQDelta, loadQDelta, svcBminDelta, svcBmaxDelta);
    }

    private record ReactiveLimitDeltas(int reactiveLimitEntry, double minQDelta, double maxQDelta, double loadQDelta, double svcBminDelta,
                            double svcBmaxDelta) {
    }

    /** How the outage reweights the bus's distributed-slack participation: the factor, row-target base and
     *  headroom/footroom the lost generators take with them, or null when the shape refuses the contingency. */
    private static DistributedSlackReweight distributedSlackReweight(final int distributedSlackOverrideBusIndex, final double lostP, final double lostLoadP,
                                                       final List<LfGenerator> lost,
                                                       final com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDistributedSlackParameters distributedSlackParameters,
                                                       final double distributedSlackRawFactorSum, final LfBus bus,
                                                       final java.util.Set<LfGenerator> lostSet) {
        double distributedSlackFactorDelta = 0;
        double distributedSlackBaseTargetPDelta = 0;
        double distributedSlackOverrideMaxDelta = 0;
        double distributedSlackOverrideMinDelta = 0;
        if (distributedSlackOverrideBusIndex >= 0) {
            distributedSlackBaseTargetPDelta = lostP - lostLoadP;                 // net DROP of the row target (load shed RAISES it)
            for (LfGenerator gen : lost) {                   // per-gen deltas compose additively
                double gf = distributedSlackParameters.factor(gen);
                if (gf != 0) {
                    distributedSlackFactorDelta += gf / distributedSlackRawFactorSum;
                    double[] r = distributedSlackParameters.rooms(gen, gen.getTargetP());
                    // An infinite room (active limits off, OLF's sign rule leaves one side open) cannot be taken
                    // off the bus's aggregate as a delta: it is 0 while another participant on the bus keeps that
                    // side open, and inexpressible (inf - inf) when the survivors all close it.
                    for (int side = 0; side < 2; side++) {
                        if (Double.isInfinite(r[side])) {
                            boolean stillOpen = false;
                            for (LfGenerator other : bus.getGenerators()) {
                                if (!lostSet.contains(other) && distributedSlackParameters.factor(other) != 0
                                        && Double.isInfinite(distributedSlackParameters.rooms(other, other.getTargetP())[side])) {
                                    stillOpen = true;
                                    break;
                                }
                            }
                            if (!stillOpen && !distributedSlackParameters.isLoadBased()) {
                                return null;
                            }
                            r[side] = 0;
                        }
                    }
                    distributedSlackOverrideMaxDelta += r[0];
                    distributedSlackOverrideMinDelta += r[1];
                }
            }
        }
        return new DistributedSlackReweight(distributedSlackFactorDelta, distributedSlackBaseTargetPDelta, distributedSlackOverrideMaxDelta, distributedSlackOverrideMinDelta);
    }

    private record DistributedSlackReweight(double distributedSlackFactorDelta, double distributedSlackBaseTargetPDelta,
            double distributedSlackOverrideMaxDelta, double distributedSlackOverrideMinDelta) {
    }

    /** The reactive-limit entries of the group's other controllers that keep a VOLTAGE generator after the loss. */
    private static int[] survivorReactiveLimitEntriesOf(LfBus bus, java.util.Set<LfGenerator> lostSet, int[] reactiveLimitBusNum) {
        java.util.List<Integer> entries = new ArrayList<>();
        for (LfBus controller : mainVcOf(bus).orElseThrow().getMergedControllerElements()) {
            if (controller == bus || controller.getGenerators().stream().noneMatch(g -> !lostSet.contains(g)
                    && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE)) {
                continue;
            }
            for (int i = 0; reactiveLimitBusNum != null && i < reactiveLimitBusNum.length; i++) {
                if (reactiveLimitBusNum[i] == controller.getNum()) {
                    entries.add(i);
                    break;
                }
            }
        }
        return entries.stream().mapToInt(Integer::intValue).toArray();
    }
}
