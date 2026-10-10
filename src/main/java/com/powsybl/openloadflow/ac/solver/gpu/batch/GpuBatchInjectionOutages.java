/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
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
final class GpuBatchInjectionOutages {

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

    /** OLF's {@code updateReactiveKeys} rule, plus THIS scenario's loss: a controller that is
     *  disabled, has its voltage control off, or is the member the outage deactivates takes no share
     *  of the group's reactive dispatch, and its key leaves the renormalization denominator. */
    /** The bus's own slot in the device's packed DISTR_Q member tables, or -1 when it is not a member
     *  of any group. Used both to locate the group for target seeding and to decide whether a load
     *  loss on it is expressible as a per-scenario member targetQ override. */
    static int dqMemberSlot(GpuAcData gpuData, LfBus bus) {
        int[] busNums = gpuData.dqMemberBusNum();
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
                                                java.util.Map<Integer, int[]> relocDummyQRows,
                                                java.util.Set<Integer> outagedZiNums) {
        return islandedDistrQEntries(gpuData, islandedControllers, relocDummyQRows, outagedZiNums, null);
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
                                                java.util.Map<Integer, int[]> relocDummyQRows,
                                                java.util.Set<Integer> outagedZiNums,
                                                java.util.Map<Integer, Integer> labelOf) {
        List<GenOutage> out = new ArrayList<>();
        int[] mBusNum = gpuData.dqMemberBusNum();
        if (mBusNum == null || islandedControllers.isEmpty()) {
            return out;
        }
        java.util.Set<Integer> inactive = new java.util.LinkedHashSet<>();
        for (LfBus b : islandedControllers) {
            int m = dqMemberSlot(gpuData, b);
            if (m >= 0) {
                inactive.add(m);
            }
        }
        int[] groupStart = gpuData.dqMemberGroupStart();
        int[] groupCount = gpuData.dqMemberGroupCount();
        double[] mKey = gpuData.dqMemberKey();
        byte[] mHasRow = gpuData.dqMemberHasRow();
        int[] mRow = gpuData.dqMemberRow();
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
                    double kk = !inactive.contains(m) && memberKeepsKey(any, mBusNum[m], m, -1) ? mKey[m] : 0;
                    keySumByLabel.merge(lbl, kk, Double::sum);
                }
                List<Integer> r = new ArrayList<>();
                List<Double> v = new ArrayList<>();
                for (int m = gs; m < gs + gc; m++) {
                    if (mHasRow[m] == 0 || mRow[m] < 0 || inactive.contains(m)) {
                        continue;                            // an islanded member's own row is frozen
                    }
                    // ...and a member PINNED in the base has no DISTR_Q equation: its row hosts its
                    // own reactive balance at the frozen limit, which OLF never rewrites (see the
                    // generator-outage loop below for the measurement - this is the loop a BRANCH
                    // outage that islands a member reaches, network.xiidm/IZUZ included).
                    if (!any.getNetwork().getBus(mBusNum[m]).isGeneratorVoltageControlEnabled()) {
                        continue;
                    }
                    double keySum = keySumByLabel.getOrDefault(labelOf != null ? labelOf.getOrDefault(m, 0) : 0, 0.0);
                    double qpk = keySum == 0 || !memberKeepsKey(any, mBusNum[m], m, -1) ? 0 : mKey[m] / keySum;
                    // dqMemberRow is STRUCTURAL: under the coupler-tree relocation a child controller's
                    // reactive row is its parent coupler's DUMMY_Q. A scenario that OPENS that coupler pins
                    // the dummy to a DUMMY_TARGET identity and OLF's DISTR_Q sits on the bus's own row
                    // instead (read off OLF's post-contingency dump for the middle-section fixture:
                    // distr_q5 ACTIVE with keys 0.5/0.5, bus_target_q[5] inactive). Seeding the
                    // renormalized target on the dummy left the live row at its BASE target, so the device
                    // solved the renormalized WEIGHTS against a target built from the base keys: measured
                    // -13.3334 MVAr where OLF has -20.0001, one third of the pilot's 40 MVAr load instead
                    // of one half, and 0.32 kV on every bus behind it.
                    r.add(GpuBatchQualification.unrelocQRow(mRow[m], relocDummyQRows, outagedZiNums));
                    // targetQ over THIS member's peers — its own control's elements, as AcTargetVector sums them
                    double sumTargetQ = 0;
                    for (int t = gpuData.dqMemberTgtPtr()[m]; t < gpuData.dqMemberTgtPtr()[m + 1]; t++) {
                        sumTargetQ += baseTargetQ(any, mBusNum[gpuData.dqMemberTgtPeer()[t]]);
                    }
                    v.add(qpk * sumTargetQ - baseTargetQ(any, mBusNum[m]));
                }
                rows = r.stream().mapToInt(Integer::intValue).toArray();
                values = v.stream().mapToDouble(Double::doubleValue).toArray();
                if (System.getenv("OLF_GPU_DQISL_TRACE") != null) {
                    // The counterpart of the generator path's DQSEED. Its absence cost a measurement:
                    // DQSEED staying silent says nothing about THIS path, and reading that as "the
                    // target is never seeded" sent one fix at an inert code path.
                    StringBuilder sb = new StringBuilder();
                    for (int m = gs; m < gs + gc; m++) {
                        sb.append(" m").append(m).append("[bus=").append(mBusNum[m])
                                .append(" hasRow=").append(mHasRow[m]).append(" row=").append(mRow[m])
                                .append(" reloc=").append(GpuBatchQualification.unrelocQRow(mRow[m],
                                        relocDummyQRows, outagedZiNums))
                                .append(" key=").append(mKey[m])
                                .append(" targetQ=").append(baseTargetQ(any, mBusNum[m]))
                                .append(" inactive=").append(inactive.contains(m))
                                .append(" keeps=").append(memberKeepsKey(any, mBusNum[m], m, -1)).append(']');
                    }
                    System.err.println("DQISL group=" + gs + " count=" + gc + " keySumByLabel=" + keySumByLabel
                            + " rows=" + r + " values=" + v + sb);
                }
            }
            out.add(new GenOutage(-1, 0, -1, 0, false, false, -1, 0, 0, 0, 0, -1, 0, 0, false,
                    java.util.Set.of(), 0, 0, 0, im, new int[0], rows, values, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, false));
        }
        return out;
    }

    /** A member's targetQ in the BASE state these seeds describe: a base-pinned member IS frozen at its
     *  limit there. The device table holds the UNPINNED value instead (what the kernel needs once a pin
     *  lifts), so the seeds read the bus. */
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

    /** WHY one bus's injection outage could not be represented. All twelve exits share the
     *  PLAN:INJECTION_OUTAGE_SHAPE census tag, which is too coarse to pick the next thing to fix;
     *  OLF_GPU_GENOUT_REJECT=1 names the exit and the bus. */
    private static <T> T reject(String why, LfBus bus) {
        if (System.getenv("OLF_GPU_GENOUT_REJECT") != null) {
            System.err.println("GENOUT_REJECT " + why + " bus=" + (bus == null ? "?" : bus.getId()));
        }
        return null;
    }

    /** Merged per-BUS outages for a MULTI-generator contingency: gens grouped by bus, each bus's
     *  targets/DS deltas summed. Stage-1 constraint: at most ONE distributed-slack-participant bus
     *  per contingency (the native DS override carries one entry per scenario) — more, or any
     *  per-bus disqualification, returns null and the contingency routes to the CPU. */
    static GenOutage[] computeInjectionOutages(EquationSystem<AcVariableType, AcEquationType> es, GpuAcData gpuData,
                                               double dsSum, boolean slackParticipates,
                                               com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams dsParams,
                                               List<LfGenerator> gens, List<LoadLoss> loads, int[][] busRows) {
        java.util.Map<LfBus, List<LfGenerator>> byBus = new java.util.LinkedHashMap<>();
        java.util.Map<LfBus, List<LoadLoss>> loadsByBus = new java.util.LinkedHashMap<>();
        for (LfGenerator g : gens) {
            if (g.getBus() == null) {
                return reject("GEN_WITHOUT_BUS", null);
            }
            byBus.computeIfAbsent(g.getBus(), b -> new ArrayList<>()).add(g);
        }
        for (LoadLoss l : loads) {
            loadsByBus.computeIfAbsent(l.bus(), b -> new ArrayList<>()).add(l);
            byBus.computeIfAbsent(l.bus(), b -> new ArrayList<>());   // load-only bus: empty gen list
        }
        GenOutage[] out = new GenOutage[byBus.size()];
        int k = 0;
        int dsBuses = 0;
        for (java.util.Map.Entry<LfBus, List<LfGenerator>> e : byBus.entrySet()) {
            GenOutage go = computeBusGenOutage(es, gpuData, dsSum, slackParticipates, dsParams, e.getKey(), e.getValue(),
                    loadsByBus.getOrDefault(e.getKey(), List.of()), busRows);
            if (go == null) {
                return null;                                 // the per-bus exit already named itself
            }
            if (go.dsBus() >= 0 && ++dsBuses > 1) {
                LOGGER.debug("Multi-injection contingency touches several DS-participant buses — CPU fallback");
                return reject("MULTI_DS_BUS", e.getKey());
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
            changesControl |= go.toPq() || go.deactivatesDqMember() || go.killsPilotVoltageIdentity()
                    || go.flipsPilotRowWhenAllPinned();
        }
        if (movesPilotQ && changesControl) {
            return reject("PILOT_Q_WITH_CONTROL_CHANGE", null);
        }
        // Two buses of the SAME remote group deactivated by one contingency: each bus's seeded
        // DISTR_Q targets renormalize over its own loss alone, so neither accounts for the other and
        // both are wrong. The kernel's mask handles the matrix side for any number of them; only the
        // plan-time seed is per-bus. Rather than make the seed a group-wide computation for a shape
        // rte6515 never produces (its generator contingencies are single-generator), route it.
        for (int a = 0; a < out.length; a++) {
            if (!out[a].deactivatesDqMember()) {
                continue;
            }
            int ga = gpuData.dqMemberGroupStart()[out[a].inactiveDqMember()];
            for (int b = a + 1; b < out.length; b++) {
                if (out[b].deactivatesDqMember()
                        && gpuData.dqMemberGroupStart()[out[b].inactiveDqMember()] == ga) {
                    return reject("DISTR_Q_TWO_MEMBERS_LOST", null);
                }
            }
        }
        return out;
    }

    /** The bus's generator voltage control as the GROUP sees it: the MAIN control of a zero-impedance merge.
     *  A section's own control is DEPENDENT there, and asking it reads a one-member group (its own controller)
     *  with no survivors - which refused 285 of rte6515's full-list generator outages as DISTR_Q_LAST_CONTROLLER
     *  (OLF_GPU_GENOUT_SHAPE: members=1 alive=0). See the merged-voltage-control note on GpuAcRowIndexing.mainGenVc. */
    private static java.util.Optional<com.powsybl.openloadflow.network.GeneratorVoltageControl> mainVcOf(LfBus bus) {
        return java.util.Optional.ofNullable(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcOuterLoopPacks.mainGenVc(bus));
    }

    private static GenOutage computeBusGenOutage(EquationSystem<AcVariableType, AcEquationType> es, GpuAcData gpuData,
                                                 double dsSum, boolean slackParticipates,
                                                 com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams dsParams, LfBus bus,
                                                 List<LfGenerator> lost, List<LoadLoss> lostLoads, int[][] busRows) {
        int busNum = bus.getNum();
        // The slack (reference) bus solves without a P balance: OLF drops its BUS_TARGET_P, and its scheduled P is read
        // only by the distributed-slack check (the slack's mismatch = its P minus that target). A LOAD lost there moves
        // that target and nothing else on the P side, which the device carries per scenario in the slack shift its
        // distributed-slack kernels already add to the slack target. A generator lost there changes the slack's own
        // participation, and a slack-only distribution reports what it absorbed through that same accumulator: both
        // stay on the CPU.
        boolean slackBus = es.getEquation(busNum, AcEquationType.BUS_TARGET_PHI).isPresent();
        if (slackBus && (!lost.isEmpty() || (gpuData.dsPhiRow().length == 0 && gpuData.dsSlackPhiRow() >= 0))) {
            return reject("SLACK_BUS", bus);
        }
        // DISTR_Q controller bus (remote/distributed VC): the qPct weights are BAKED into the
        // equation system at creation — the CPU does not rebuild them on a gen loss either. When the
        // bus KEEPS another voltage-controlling generator the group is unchanged on both arms and the
        // outage is a plain P-target drop (no PV->PQ flip, no Q override — the distribution absorbs);
        // losing the bus's LAST controller restructures the group -> CPU.
        boolean distrQBus = es.getEquation(busNum, AcEquationType.DISTR_Q).isPresent();
        boolean vcSurvives = false;
        boolean deadRemoteMember = false;
        int deadPilotVRow = -1;
        double deadPilotQTarget = 0;
        if (distrQBus) {
            java.util.Set<LfGenerator> lostHere = new java.util.HashSet<>(lost);
            boolean keepsController = bus.getGenerators().stream()
                    .anyMatch(g -> !lostHere.contains(g)
                            && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE);
            // A controller bus that loses its last VOLTAGE generator while the group's OTHER
            // controllers survive does NOT go to PQ: they still hold its voltage, so its BUS_TARGET_V
            // stays. What changes is that it stops participating in the reactive sharing, which the
            // qPct override already expresses on the matrix side — and now on the target side too
            // (inactiveDqMember). When NO controller survives the control itself dies, which is a
            // different shape (every member re-solves its own reactive balance) and still routes to
            // the CPU, as does a Q-carrying load loss here.
            // Admit exactly the shape the degeneration argument covers, and no more:
            //  (a) another controller survives, so the group still holds this bus's voltage;
            //  (b) the survivors MAY all pin: a pinned survivor's row takes its own balance (mode 4), and
            //      a pinned pilot's too, so nothing is evicted - a runtime reject used to replay such a
            //      scenario on the CPU (DISTR_Q_PILOT_ALL_PINNED), for the case where the outaged bus
            //      IS the pilot and its balance rode a survivor's row (GEN-1522: q 0.601 vs OLF's 0);
            //      that case flips the pilot's own row in the solve since (flipsPilotRowWhenAllPinned),
            //      and the member case was measured exact without the reject (runtime-pin fixture,
            //      rte6515 GEN-6231#0). Where the survivors keep headroom the degeneration
            //      holds exactly — GEN-1523 at overMaxBy -0.149 and GEN-5958 at -0.900 both land at
            //      ~4e-11 kV;
            //  (c) the pilot carries no reactive load. Lifting this leaves GEN-1264 at 0.134 kV and
            //      GEN-1263 at 0.247 kV, and every INPUT to that row has been verified correct:
            //        weights  - dqOv never fires here (nDqOv=0); the inactive mask supplies them and
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
                    .map(vc -> {
                        List<LfBus> members = vc.getMergedControllerElements();
                        long survivors = members.stream()
                                .filter(cb -> cb != bus && cb.getGenerators().stream()
                                        .anyMatch(g -> !lostHere.contains(g)
                                                && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE))
                                .count();
                        // The group's FREE member — the first enabled controller, whose row does NOT
                        // hold a DISTR_Q: it carries the pilot's relocated reactive balance instead,
                        // or, for a pilot that is one of its own controllers, nothing at all, since
                        // while it controls it has no reactive balance to relocate.
                        boolean isFree = members.stream().filter(vc::isControllerEnabled)
                                .findFirst().map(m -> m == bus).orElse(false);
                        boolean pilotIsController = members.contains(vc.getControlledBus());
                        // Losing the FREE member moves the free role, and with it the row OLF puts the
                        // pilot's reactive balance on; the mode scan re-picks the carrier, so any number of
                        // survivors is fine (measured: rte6515 GEN-5958, 3 members and 2 survivors,
                        // 3.8e-11; reproducer deviceRemoteGroupLosingFreeMemberMatchesCpu). That holds for a
                        // pilot that IS one of its controllers too: its gen-lost entry opens the pilot-self
                        // gate, its raw q lands on the carrier with its own balance as target and the
                        // carrier's DISTR_Q is suppressed. Two or more survivors used to route to the CPU
                        // on a 0.149 kV measurement, which was the seed-time stamps reading the dynamic
                        // limit arrays before anything had filled them on the cold path (carrier target
                        // 0 - 0, a pinned peer's target 0 - 0): devicePilotControllerLosingItsOwnGenerator
                        // MatchesCpu is exact at 1e-11 since, and rte6515's 50 such outages batch.
                        return survivors > 0;
                    })
                    .orElse(false);
            // The group's control DIES with this bus (no controller anywhere keeps a VOLTAGE
            // generator). OLF then deactivates the pilot's BUS_TARGET_V and re-activates a reactive
            // balance at BOTH the pilot and this member. The device reaches the same layout through
            // the relocation it already has, provided this row goes to mode 4 rather than mode 1:
            // mode 4 means "this row hosts its own Q balance", and it is also what makes the group's
            // mode scan skip the row, so the pilot's balance falls back to the pilot's own row and
            // updatePilotVeqBatch drops the voltage identity. Needs a pilot that is NOT one of its own
            // controllers — that shape has no relocation table for the fallback to land in.
            boolean controlDies = !keepsController && mainVcOf(bus)
                    .map(vc -> vc.getMergedControllerElements().stream()
                                    .noneMatch(cb -> cb.getGenerators().stream()
                                            .anyMatch(g -> !lostHere.contains(g)
                                                    && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE))
                            && !vc.getMergedControllerElements().contains(vc.getControlledBus())
                            && vc.getControlledBus() != bus)
                    .orElse(false);
            if (controlDies && lostLoads.isEmpty()) {
                deadRemoteMember = true;
                LfBus pilot = mainVcOf(bus).orElseThrow().getControlledBus();
                var pilotV = es.getVariableSet().getVariable(pilot.getNum(), AcVariableType.BUS_V);
                if (pilotV == null || pilotV.getRow() < 0) {
                    return reject("DEAD_CONTROL_PILOT_NO_V_ROW", bus);
                }
                deadPilotVRow = pilotV.getRow();
                deadPilotQTarget = pilot.getTargetQ();
            } else if ((!keepsController && !vcSurvives) || !lostLoads.isEmpty()) {
                // BOTH halves are load-bearing, and 2026-09-17 they turned out to be ONE root cause.
                // Lifting them on rte6515's full sets: the 9 generator cases land wrong (0.067 up to
                // 3.157 kV at GEN-1522) and LOAD-1263 lands 0.105 kV out.
                //
                // EVERY one of those buses is a PILOT THAT IS ALSO ONE OF ITS CONTROLLERS
                // (OLF_GPU_GENOUT_TRACE prints pilotIsCtl) — the shape ExtractionColumns deliberately
                // excludes from the Q-relocation table, because while it controls it has NO Q balance
                // of its own to relocate (its q feeds the group's DISTR_Q rows, and a table entry
                // would double-count it). GpuAcRowIndexing.distrQRow already parks its DISTR_Q on the
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
                if (System.getenv("OLF_GPU_GENOUT_SHAPE") != null) {
                    var vcO = mainVcOf(bus);
                    LfBus pilot = vcO.map(vc -> vc.getControlledBus()).orElse(null);
                    long alive = vcO.map(vc -> vc.getMergedControllerElements().stream()
                            .filter(cb -> cb.getGenerators().stream().anyMatch(g -> !lostHere.contains(g)
                                    && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE))
                            .count()).orElse(0L);
                    System.err.printf("GENOUT_SHAPE bus=%s pilot=%s pilotIsCtl=%b members=%d alive=%d "
                            + "pilotLoadQ=%.4f keepsCtl=%b lostLoads=%d%n",
                            bus.getId(), pilot == null ? "?" : pilot.getId(),
                            vcO.map(vc -> vc.getMergedControllerElements().contains(vc.getControlledBus())).orElse(false),
                            vcO.map(vc -> vc.getMergedControllerElements().size()).orElse(0), alive,
                            pilot == null ? 0.0 : pilot.getLoadTargetQ(), keepsController, lostLoads.size());
                    // The DEVICE rows of the pilot and of every member, so a post-solve readback
                    // (OLF_GPU_FINAL_TARGET_ROW) can be pointed at this group without guessing which
                    // row index the equation landed on.
                    if (pilot != null) {
                        StringBuilder mr = new StringBuilder();
                        vcO.ifPresent(vc -> vc.getMergedControllerElements().forEach(m -> {
                            var mv = es.getVariableSet().getVariable(m.getNum(), AcVariableType.BUS_V);
                            mr.append(m.getId()).append('=').append(mv == null ? -1 : mv.getRow()).append(' ');
                        }));
                        var pv = es.getVariableSet().getVariable(pilot.getNum(), AcVariableType.BUS_V);
                        System.err.printf("GENOUT_SHAPE   rows pilot=%s:%d members: %s%n",
                                pilot.getId(), pv == null ? -1 : pv.getRow(), mr.toString().trim());
                    }
                    double postGenQ = bus.getGenerators().stream().filter(g -> !lostHere.contains(g))
                            .mapToDouble(LfGenerator::getTargetQ).sum();
                    System.err.printf("GENOUT_SHAPE   bus=%s baseTargetQ=%.9f postTargetQ=%.9f delta=%.9f "
                            + "frozen=%b genTargetQ=%.9f loadQ=%.9f%n",
                            bus.getId(), bus.getTargetQ(), postGenQ - bus.getLoadTargetQ(),
                            (postGenQ - bus.getLoadTargetQ()) - bus.getTargetQ(),
                            bus.isGenerationTargetQFrozen(), bus.getGenerationTargetQ(), bus.getLoadTargetQ());
                }
                // A PURE reactive-LOAD loss on a packed DISTR_Q member, with the group's control
                // untouched, IS expressible: the member's own targetQ moves (targetQ = generation -
                // load) and every DISTR_Q target in the group follows it through the per-scenario
                // member-targetQ override (setBatchDqMemberTargetQOverrides), which the device's
                // target kernel reads instead of the base value. Deliberately narrow — no generator
                // may be lost on the bus in the same scenario, because a frozen generation target Q
                // makes the surviving-generator sum the wrong post-contingency value (see the
                // qOverride note above), and that case has not been measured.
                boolean pureLoadLossOnPackedMember = keepsController && lostHere.isEmpty()
                        && !lostLoads.isEmpty() && dqMemberSlot(gpuData, bus) >= 0;
                if (!pureLoadLossOnPackedMember) {
                    return reject(keepsController ? "DISTR_Q_BUS_WITH_LOAD_LOSS" : "DISTR_Q_LAST_CONTROLLER", bus);
                }
            }
        }
        if (bus.hasGeneratorsWithSlope()) {               // slope-SVC bus outage: the mode-2 row's fate on a
            return reject("SLOPE_SVC_BUS", bus);             // gen loss is not modeled — CPU
        }
        if (dsParams.isLoadBased() && !lostLoads.isEmpty()) {   // a lost/shifted load changes the LOAD
            return reject("LOAD_BASED_DS_LOAD_LOSS", bus);   // participation factors — not modeled: CPU
        }
        java.util.Set<LfGenerator> lostSet = new java.util.HashSet<>(lost);
        // The bus's P-BALANCE row, which under a coupler tree is its parent coupler's DUMMY_P column,
        // NOT its own BUS_PHI column (GpuAcDataExtractor#busPowerRows). Resolving it from the variable
        // set instead put every injection override of a relocated bus on a coupling row whose target is
        // 0, so the absolute override was a no-op and the lost injection stayed in the balance — and it
        // also made the dsPhiRow lookup below miss, dropping the anchor shift. network.xiidm's loads:
        // 4183 of 7494 wrong totals, worst 148 MW, 12640 of 13338 buses off by up to 1.03 kV.
        int pRow = slackBus ? -1 : busRows[0][busNum];       // the slack's row is its angle reference
        int vRow = es.getVariableSet().getVariable(busNum, AcVariableType.BUS_V).getRow();
        if ((pRow < 0 && !slackBus) || vRow < 0) {
            return reject("NO_P_OR_V_ROW", bus);
        }
        // Is this bus a distributed-slack participant? Matters for BOTH shapes: lost generators
        // leave the participation (factor/headroom reweight), AND pure injection deltas (load
        // losses, redispatch actions) shift the row target — which the retain/waterfill accounting
        // would misread as already-distributed slack power unless dsBaseTargetP shifts with it
        // (measured: a +60 MW redispatch cancelled to a no-op before the base-shift fix).
        double preLostLoadP = lostLoads.stream().mapToDouble(LoadLoss::p).sum();
        int dsBus = -1;
        if (!lost.isEmpty() || preLostLoadP != 0) {
            int[] dsPhiRow = gpuData.dsPhiRow();
            for (int i = 0; i < dsPhiRow.length; i++) {
                if (dsPhiRow[i] == pRow) {
                    dsBus = i;
                    break;
                }
            }
        }
        boolean lostParticipation = lost.stream().anyMatch(g -> dsParams.factor(g) != 0);
        if (dsParams.isRemainingMargin() && dsBus >= 0 && lostParticipation) {
            return reject("REMAINING_MARGIN_DS", bus);       // RM margins reweight per scenario: not modeled — CPU
        }
        if (dsBus >= 0 && lostParticipation && slackParticipates) {
            return reject("PARTICIPATING_SLACK", bus);   // FACTOR reweight + participating slack: the retain math
        }                                           // doesn't hold (a pure base shift keeps the factors — exact)
        if (!distrQBus && bus.getGeneratorVoltageControl().map(vc -> vc.getControlledBus() != null
                && vc.getControlledBus().getNum() != busNum).orElse(false)) {
            return reject("REMOTE_SINGLE_CONTROLLER", bus);   // (pilot-Q remap, no DISTR_Q row at this bus):
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
        // bus that the base pinned PV->PQ also has busPv false, and its V row DOES solve the reactive
        // equation, so it must keep shedding the lost Q. Suppressing it there moves
        // devicePinnedBusLosingReactiveLoadMatchesCpu's B4 by 1.0e-08 kV.
        boolean qOverride = toPq || (!busPv && !remotelyControlled);
        // ...but a remotely controlled bus still HAS a reactive balance - OLF keeps its BUS_TARGET_Q, generation
        // minus load like any PQ bus - and the extraction moved it onto the relocation group's CARRIER (qRowAlt =
        // -2-g: the free member's row), whose target the device captures from the BASE targets at solve start
        // (capturePilotQBatch) and hands to whichever member carries the balance later. Losing reactive load (or a
        // fixed-Q generator) at the pilot moves that target, and neither channel could say so: measured on the
        // remote-control fixture, a 20 MVar load lost at the pilot stayed in its balance and every controller bus
        // ended 3.7e-2 kV off OLF. The group and the move travel to the carrier's first-solve target and to the
        // device's captured value (setBatchPilotQDeltas).
        int pilotQGroup = -1;
        double pilotQDelta = 0;
        boolean purePilot = remotelyControlled
                && mainVcOf(bus).map(vc -> !vc.getMergedControllerElements().contains(bus)).orElse(false);
        if (purePilot) {                                     // a pilot among its controllers has no balance of its
            double lostGenQ = lost.stream().mapToDouble(LfGenerator::getTargetQ).sum();   // own (DISTR_Q member)
            if (lostLoadQ != 0 || lostGenQ != 0) {
                int alt = busRows.length > 3 ? busRows[3][busNum] : -1;
                if (alt > -2) {                              // the balance is not on a carrier: no channel for it
                    return reject("PILOT_Q_NOT_RELOCATED", bus);
                }
                pilotQGroup = -2 - alt;
                pilotQDelta = lostLoadQ - lostGenQ;          // targetQ = generation - load
            }
        }
        double qTarget = 0;
        if (qOverride) {
            // A bus the BASE loadflow pinned at a reactive limit has its generation target Q FROZEN at
            // that limit, and OLF keeps that value as long as a VOLTAGE-type generator survives:
            // processLostGenerators calls setGeneratorVoltageControlEnabledAndRecomputeTargetQ only when
            // none does, and invalidateGenerationTargetP leaves a frozen bus frozen. Summing the
            // survivors' own targetQ instead hands the first inner solve a target 1.7 pu off the one
            // OLF solves (measured at rte6515 VL-6003_0: 1.03 vs 2.69 pu) — the reactive-limit pass
            // then makes its pin/unpin decisions along a trajectory the CPU never takes. When the last
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
        double dsFactorDelta = 0;
        double dsBaseDelta = 0;
        double dsMaxDelta = 0;
        double dsMinDelta = 0;
        if (dsBus >= 0) {
            dsBaseDelta = lostP - lostLoadP;                 // net DROP of the row target (load shed RAISES it)
            for (LfGenerator gen : lost) {                   // per-gen deltas compose additively
                double gf = dsParams.factor(gen);
                if (gf != 0) {
                    dsFactorDelta += gf / dsSum;
                    if (dsParams.useActiveLimits()) {        // rooms are +-infinity otherwise — a lost gen
                        double minTp = gen.getMinTargetP();  // cannot change them (and inf - inf = NaN)
                        double maxTp = gen.getMaxTargetP();
                        double tp = gen.getTargetP();
                        if (tp < 0) {
                            maxTp = Math.min(maxTp, 0);
                        } else {
                            minTp = Math.max(minTp, 0);
                        }
                        dsMaxDelta += maxTp - tp;
                        dsMinDelta += minTp - tp;
                    }
                }
            }
        }
        // A bus that KEEPS voltage control still carries the lost generators' reactive capability in the
        // device's run-constant aggregate, while OLF's bus.getMinQ()/getMaxQ() sum only the generators
        // that are still there. Name the reactive-limit entry and how much to take off it. (A bus the
        // outage takes to PQ needs nothing: its entry stops being a controller and the limits are never
        // read.) A bus the BASE loadflow already PINNED at a limit is NOT that case, though it reads
        // busPv=false: its entry is still a controller, its pinned target Q IS the limit, and it can
        // switch back to PV. Measured on rte6515 GEN-6003 — one generator of three lost at a pinned
        // bus, the stale capability injected as Q, and 1.9 kV network-wide.
        int rlEntry = -1;
        double minQDelta = 0;
        double maxQDelta = 0;
        int[] rlBusNum = gpuData.rlBusNum();
        for (int i = 0; rlBusNum != null && i < rlBusNum.length; i++) {
            if (rlBusNum[i] == bus.getNum()) {
                rlEntry = i;
                break;
            }
        }
        double loadQDelta = 0;
        if (rlEntry >= 0 && !toPq) {
            minQDelta = -lost.stream().mapToDouble(g -> qContrib(g, LfGenerator::getMinQ)).sum();
            maxQDelta = -lost.stream().mapToDouble(g -> qContrib(g, LfGenerator::getMaxQ)).sum();
            // The pack's loadQ is the bus's load target Q as EXTRACTED, and the reactive-limit
            // machinery reads it in two paired forms: the pinned target is limit - loadQ, and the
            // controller's own q is the net injection + loadQ. Shed a Q-carrying load and both go
            // stale. This used to be a CPU gate for a PV bus; the bus the base loadflow had already
            // PINNED fell through it and was silently wrong instead (rte6515 LOAD-6157, 0.37 kV).
            loadQDelta = -lostLoadQ;
        }
        // No generator left at all. The reactive-limit entry must stop being a controller for this
        // scenario: a bus the BASE loadflow had already pinned PQ keeps a seeded limit type so a
        // relieving contingency can switch it back to PV — and with its only generator gone there is
        // nothing to switch back to. OLF simply has no controller there. (Measured on rte6515 GEN-6130:
        // 0.995 kV at that node against 2.2e-4 everywhere else, and one apparent-power violation.)
        boolean allGensLost = bus.getGenerators().stream().allMatch(lostSet::contains);
        if (System.getenv("OLF_GPU_GENOUT_TRACE") != null) {
            System.err.printf("GENOUT bus=%s busPv=%b frozenQ=%b toPq=%b qOverride=%b lost=%s busGens=%d rlEntry=%d "
                    + "dMinQ=%.6f dMaxQ=%.6f busMinQ=%.6f busMaxQ=%.6f distrQ=%b ctrlBus=%s dsBus=%d dFactor=%.9f dBase=%.9f lostLoadQ=%.9f dLoadQ=%.9f vcCtl=%d vcCtlAlive=%d local=%b pilotIsCtl=%b vRow=%d members=%s free=%s pilotLoadQ=%.9f deadOwnsRow=%b%n",
                    bus.getId(), busPv, bus.isGenerationTargetQFrozen(), toPq, qOverride,
                    lost.stream().map(LfGenerator::getId).toList(), bus.getGenerators().size(),
                    rlEntry, minQDelta, maxQDelta, bus.getMinQ(), bus.getMaxQ(), distrQBus,
                    bus.getGeneratorVoltageControl().map(vc -> vc.getControlledBus().getId()).orElse("-"),
                    dsBus, dsFactorDelta, dsBaseDelta, lostLoadQ, loadQDelta,
                    bus.getGeneratorVoltageControl().map(vc -> vc.getMergedControllerElements().size()).orElse(0),
                    bus.getGeneratorVoltageControl().map(vc -> (int) vc.getMergedControllerElements().stream()
                            .filter(cb -> cb.getGenerators().stream()
                                    .anyMatch(g -> !lostSet.contains(g)
                                            && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE))
                            .count()).orElse(0),
                    bus.getGeneratorVoltageControl().map(vc -> vc.isLocalControl()).orElse(false),
                    bus.getGeneratorVoltageControl().map(vc -> vc.getMergedControllerElements().contains(bus))
                            .orElse(false),
                    vRow,
                    bus.getGeneratorVoltageControl().map(vc -> vc.getMergedControllerElements().stream()
                            .map(m -> m.getId() + "#" + es.getVariableSet()
                                    .getVariable(m.getNum(), AcVariableType.BUS_V).getRow()).toList().toString())
                            .orElse("-"),
                    bus.getGeneratorVoltageControl().map(vc -> vc.getMergedControllerElements().stream()
                            .filter(vc::isControllerEnabled).findFirst().map(LfBus::getId).orElse("none"))
                            .orElse("-"),
                    bus.getGeneratorVoltageControl().map(vc -> vc.getControlledBus().getLoadTargetQ()).orElse(0.0),
                    bus.getGeneratorVoltageControl().map(vc -> vc.getMergedControllerElements().stream()
                            .filter(vc::isControllerEnabled).findFirst().map(m -> m != bus).orElse(false))
                            .orElse(false));
        }
        // Identity of what was lost, for the reactive-capability CURVE channel: it evaluates each
        // generator's curve at its OWN post-DS active power, which it reconstructs from the bus's row
        // target by participation share. The lost generator's entry has to go, the survivors' shares
        // have to renormalize over what is left, and the step the outage itself puts in the row target
        // is not DS power and must not move any curve. All three need per-generator identity.
        int inactiveDqMember = -1;
        if (vcSurvives) {
            int[] mBus = gpuData.dqMemberBusNum();
            for (int m = 0; mBus != null && m < mBus.length; m++) {
                if (mBus[m] == busNum) {
                    inactiveDqMember = m;
                    break;
                }
            }
            if (inactiveDqMember < 0) {
                return reject("DISTR_Q_MEMBER_NOT_PACKED", bus);
            }
        }
        // The group's DISTR_Q row targets, renormalized over the survivors exactly as OLF does
        // (updateReactiveKeys zeroes a non-enabled controller's key and renormalizes; AcTargetVector
        // then reads qPct_k * sum_j targetQ_j - targetQ_k). Seeded per scenario because the kernel that
        // maintains these targets runs only inside the reactive-limits phase — see GenOutage.
        int[] dqTargetRows = new int[0];
        double[] dqTargetValues = new double[0];
        // The outaged bus's OWN member slot in the device's DISTR_Q tables, so the group can be located
        // even when no controller is deactivated (a pure reactive-load loss on a member bus).
        int ownDqMember = dqMemberSlot(gpuData, bus);
        // targetQ = generation - load, so losing reactive load MOVES this member's own targetQ, and
        // every DISTR_Q target in the group reads it twice (shared sum + own subtracted term).
        // Measured on rte6515 LOAD-1263: OLF's residual is EXACTLY 4.600000000e-02 on the DISTR_Q row
        // of VL-1263_1 = qPct (0.5, two equal-key members) x the pilot's 0.092 pu of lost load Q.
        int dqTargetQOvMember = -1;
        double dqTargetQOvValue = 0;
        if (ownDqMember >= 0 && lostLoadQ != 0 && gpuData.dqMemberTargetQ() != null) {
            dqTargetQOvMember = ownDqMember;
            dqTargetQOvValue = gpuData.dqMemberTargetQ()[ownDqMember] + lostLoadQ;
        }
        int dqAnchor = vcSurvives ? inactiveDqMember : ownDqMember;
        double dqOwnTargetQDelta = vcSurvives ? 0 : lostLoadQ;
        if (dqAnchor >= 0 && (vcSurvives || lostLoadQ != 0)) {
            // Read the DEVICE's own member tables rather than recomputing the group: which member
            // OWNS a DISTR_Q row (the base free member's row carries the pilot's relocated balance
            // instead) and which device row it is were both decided at extraction, and a predicate
            // re-derived here would seed the wrong rows the moment the two disagreed.
            int gs = gpuData.dqMemberGroupStart()[dqAnchor];
            int gc = gpuData.dqMemberGroupCount()[dqAnchor];
            int[] mBusNum = gpuData.dqMemberBusNum();
            double[] mKey = gpuData.dqMemberKey();
            byte[] mHasRow = gpuData.dqMemberHasRow();
            int[] mRow = gpuData.dqMemberRow();
            double keySum = 0;
            for (int m = gs; m < gs + gc; m++) {             // live keys over the whole merged group
                keySum += memberKeepsKey(bus, mBusNum[m], m, inactiveDqMember) ? mKey[m] : 0;
            }
            List<Integer> rows = new ArrayList<>();
            List<Double> values = new ArrayList<>();
            for (int m = gs; m < gs + gc; m++) {
                if (mHasRow[m] == 0 || mRow[m] < 0) {
                    continue;
                }
                // A member PINNED in the base has no DISTR_Q equation: its row hosts its own reactive
                // balance at the frozen limit (targetQ_pinned - loadQ), and OLF's updateReactiveKeys /
                // AcTargetVector never write a share there. Seeding the share formula on it anyway
                // handed the row -targetQ_m (no live key, so qpk = 0): the pinned Q with its sign
                // flipped. Measured on network.xiidm/IZUZ (IZSV_0 islanded, IZTH_0 base-pinned MIN_Q):
                // +0.276956953 against OLF's -0.276956953, a 0.5539 pu entry residual, one Newton
                // iteration more than OLF on the first solve, and the slack pass read 0.000845 pu
                // less - the 0.0845 MW the arms reported apart. If the member unpins later, the
                // kernel that maintains these targets (updateDistrQTargetsBatch) writes its share then.
                if (!bus.getNetwork().getBus(mBusNum[m]).isGeneratorVoltageControlEnabled()) {
                    continue;
                }
                double qpk = keySum == 0 || !memberKeepsKey(bus, mBusNum[m], m, inactiveDqMember)
                        ? 0 : mKey[m] / keySum;
                rows.add(mRow[m]);
                // targetQ over THIS member's peers — the elements of its own control (see the peer CSR):
                // a member whose control is merged INTO the group sums only its own, as AcTargetVector does.
                double sumTargetQ = 0;
                for (int t = gpuData.dqMemberTgtPtr()[m]; t < gpuData.dqMemberTgtPtr()[m + 1]; t++) {
                    int peer = gpuData.dqMemberTgtPeer()[t];
                    sumTargetQ += baseTargetQ(bus, mBusNum[peer])
                            + (mBusNum[peer] == bus.getNum() ? dqOwnTargetQDelta : 0);
                }
                values.add(qpk * sumTargetQ
                        - (baseTargetQ(bus, mBusNum[m]) + (mBusNum[m] == bus.getNum() ? dqOwnTargetQDelta : 0)));
            }
            dqTargetRows = rows.stream().mapToInt(Integer::intValue).toArray();
            dqTargetValues = values.stream().mapToDouble(Double::doubleValue).toArray();
            if (System.getenv("OLF_GPU_RL_TRACE") != null) {
                StringBuilder sb = new StringBuilder();
                for (int m = gs; m < gs + gc; m++) {
                    sb.append(" m").append(m).append("[bus=").append(mBusNum[m]).append(" hasRow=").append(mHasRow[m])
                            .append(" row=").append(mRow[m]).append(" key=").append(mKey[m])
                            .append(" keeps=").append(memberKeepsKey(bus, mBusNum[m], m, inactiveDqMember)).append(']');
                }
                System.err.println("DQSEED bus=" + bus.getId() + " anchor=" + dqAnchor + " inactive=" + inactiveDqMember
                        + " vcSurvives=" + vcSurvives + " keySum=" + keySum + " rows=" + rows + " values=" + values + sb);
            }
        }
        int[] pilotSurvivorRlEntries = new int[0];
        if (vcSurvives) {
            java.util.List<Integer> entries = new ArrayList<>();
            for (LfBus cb : mainVcOf(bus).orElseThrow().getMergedControllerElements()) {
                if (cb == bus || cb.getGenerators().stream().noneMatch(g -> !lostSet.contains(g)
                        && g.getGeneratorControlType() == LfGenerator.GeneratorControlType.VOLTAGE)) {
                    continue;
                }
                for (int i = 0; rlBusNum != null && i < rlBusNum.length; i++) {
                    if (rlBusNum[i] == cb.getNum()) {
                        entries.add(i);
                        break;
                    }
                }
            }
            pilotSurvivorRlEntries = entries.stream().mapToInt(Integer::intValue).toArray();
        }
        // The pilot's OWN row, for the runtime flip when every survivor pins. Only when the outaged
        // bus IS the pilot: that is the case where losing its generators makes the pilot stop being
        // one of its own controllers and ACQUIRE a reactive balance. See
        // GenOutage.flipsPilotRowWhenAllPinned.
        int pinnedPilotVRow = -1;
        double pinnedPilotQTarget = 0;
        if (vcSurvives && pilotSurvivorRlEntries.length > 0
                && mainVcOf(bus).orElseThrow().getControlledBus() == bus) {
            var pilotV = es.getVariableSet().getVariable(bus.getNum(), AcVariableType.BUS_V);
            if (pilotV != null && pilotV.getRow() >= 0) {
                pinnedPilotVRow = pilotV.getRow();
                // generationTargetQ - loadTargetQ. A VOLTAGE generator contributes 0 to
                // generationTargetQ, so losing one leaves this unchanged.
                pinnedPilotQTarget = bus.getTargetQ();
            }
        }
        java.util.Set<String> lostGenIds = lost.stream().map(LfGenerator::getId)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
        double lostFactorSum = lost.stream().mapToDouble(dsParams::factor).sum();
        // Same relocation as pRow, on the reactive side: under a coupler the bus's Q BALANCE sits on the
        // parent's DUMMY_Q column, so an absolute Q target written at the BUS_V row is a no-op and the
        // lost load's REACTIVE power stays in the system. Only when the extraction actually moved it —
        // a DISTR_Q bus keeps its Q on its own group row and is never relocated (ExtractionColumns).
        int qTargetRow = busRows[2][busNum] >= 0 && busRows[1][busNum] >= 0 && busRows[1][busNum] != vRow
                ? busRows[1][busNum] : -1;
        // A relocated DISTR_Q member the BASE PINNED has no static Q row (its balance rides the group's rows)
        // but a mode-4 ALTERNATE row - its coupler's DUMMY_Q - that hosts its own balance from the start, with
        // the pinned limit as target. That is the row the override belongs on: OLF recomputes the target over the
        // generators left the moment the last one goes, and nothing on the device rewrites a base-set target
        // otherwise. Written at the bus's own V row (the coupler's ZERO_V, target 0) the override was a no-op:
        // measured on the merged-control fixture as 0.36 kV at b1, row 7 ending at 0.05 against OLF's 0.
        if (qTargetRow < 0 && busRows[2][busNum] >= 0 && busRows.length > 3 && busRows[3][busNum] >= 0
                && busRows[3][busNum] != vRow && allGensLost && rlEntry >= 0
                && gpuData.rlBaseQLimitType() != null && rlEntry < gpuData.rlBaseQLimitType().length
                && gpuData.rlBaseQLimitType()[rlEntry] >= 1 && gpuData.rlBaseQLimitType()[rlEntry] <= 4) {
            qTargetRow = busRows[3][busNum];
        }
        return new GenOutage(pRow, pTarget, vRow, qTarget, toPq, qOverride, dsBus, dsFactorDelta, dsBaseDelta,
                dsMaxDelta, dsMinDelta, rlEntry, minQDelta, maxQDelta, allGensLost, lostGenIds, lostFactorSum,
                lostLoadP - lostP, loadQDelta, inactiveDqMember, pilotSurvivorRlEntries,
                dqTargetRows, dqTargetValues, deadRemoteMember ? 4 : 1,
                deadPilotVRow, deadPilotQTarget, pinnedPilotVRow, pinnedPilotQTarget,
                dqTargetQOvMember, dqTargetQOvValue, qTargetRow, pilotQGroup, pilotQDelta, slackBus);
    }
}
