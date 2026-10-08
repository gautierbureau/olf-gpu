/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

/**
 * The per-scenario target/row-mode changes a generator loss applies on the device: remove the lost
 * generator's P from its bus's P target ({@code pRow}→{@code pTarget}); if the loss un-controls the bus
 * ({@code toPq}) flip its BUS_V row to the reactive-power equation and set {@code vRow}→{@code qTarget}.
 * Mirrors {@code LfContingency.processLostGenerators}.
 *
 * <p>{@code pRow < 0} means "no P change": the entry only carries the row-mode flip and its Q
 * target. That is how an island that KILLS a remote voltage control is expressed — every surviving
 * member bus of the control has to solve its own reactive balance again, and nothing about P moves.
 *
 * <p>When the lost generator sits on a distributed-slack participant bus, {@code dsBus} is that bus's
 * DS-pack index (else -1) and the {@code ds*Delta} fields carry the reweight the device applies: the
 * bus's participation factor, base target P, headroom and footroom each shrink by the lost gen's share,
 * and the per-scenario factor sum drops by {@code dsFactorDelta} so the other buses absorb that share.
 * This is only valid when the slack itself does not participate (otherwise the loss is rejected).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record GenOutage(int pRow, double pTarget, int vRow, double qTarget, boolean toPq, boolean qOverride,
                        int dsBus, double dsFactorDelta, double dsBaseDelta, double dsMaxDelta, double dsMinDelta,
                        int rlEntry, double minQDelta, double maxQDelta, boolean allGensLost,
                        java.util.Set<String> lostGenIds, double lostFactorSum, double pStep, double loadQDelta,
                        int inactiveDqMember, int[] pilotGroupSurvivorRlEntries,
                        int[] dqTargetRows, double[] dqTargetValues, int pqRowMode,
                        int deadPilotVRow, double deadPilotQTarget,
                        int pinnedPilotVRow, double pinnedPilotQTarget,
                        int dqTargetQOvMember, double dqTargetQOvValue,
                        /** The row carrying this bus's Q BALANCE, when the coupler-tree relocation moved it off the
                         *  bus's own BUS_V column onto the parent coupler's DUMMY_Q column; -1 = it is {@code vRow}.
                         *  The two part company only under a coupler: {@code vRow} stays the BUS_V row because that
                         *  is what the PV->PQ mode flip rewrites, while the Q TARGET has to follow the balance. */
                        int qTargetRow) {

    /** Where to write this bus's Q target: the relocated Q-balance row, else the BUS_V row. */
    public int qTargetRowOrV() {
        return qTargetRow >= 0 ? qTargetRow : vRow;
    }


    /** This entry without the DISTR_Q row targets that fall on a FROZEN (islanded) row: such a row is an identity
     *  held at a flat value, and a second override on it (a survivor's renormalized target written onto an islanded
     *  member's own row) won over the flat one and drove the dead bus's V to 0 in clamped steps. */
    public GenOutage withoutFrozenTargetRows(java.util.Set<Integer> frozenRows) {
        if (dqTargetRows == null || dqTargetRows.length == 0 || frozenRows.isEmpty()) {
            return this;
        }
        java.util.List<Integer> rows = new java.util.ArrayList<>();
        java.util.List<Double> values = new java.util.ArrayList<>();
        for (int t = 0; t < dqTargetRows.length; t++) {
            if (!frozenRows.contains(dqTargetRows[t])) {
                rows.add(dqTargetRows[t]);
                values.add(dqTargetValues[t]);
            }
        }
        if (rows.size() == dqTargetRows.length) {
            return this;
        }
        return new GenOutage(pRow, pTarget, vRow, qTarget, toPq, qOverride, dsBus, dsFactorDelta, dsBaseDelta,
                dsMaxDelta, dsMinDelta, rlEntry, minQDelta, maxQDelta, allGensLost, lostGenIds, lostFactorSum, pStep,
                loadQDelta, inactiveDqMember, pilotGroupSurvivorRlEntries,
                rows.stream().mapToInt(Integer::intValue).toArray(), values.stream().mapToDouble(Double::doubleValue).toArray(),
                pqRowMode, deadPilotVRow, deadPilotQTarget, pinnedPilotVRow, pinnedPilotQTarget, dqTargetQOvMember,
                dqTargetQOvValue, qTargetRow);
    }
    // lostGenIds / lostFactorSum exist for the reactive-capability CURVE channel, which evaluates each
    // generator's curve at its OWN post-distribution active power, reconstructed from the bus's row
    // target by participation share. Losing a generator skews that reconstruction three ways — the lost
    // entry, the step the outage puts in the row target, and the survivors' shares — and the chunk
    // solver repairs all three per scenario from this identity. pStep is the outage's OWN move of the
    // bus row target (-lostP +lostLoadP): not reconstructible from the base targets, which already carry
    // the base loadflow's slack distribution.

    /**
     * A bus that KEEPS voltage control after losing a generator still has that generator's reactive
     * capability in the device's aggregate: {@code rlMinQ}/{@code rlMaxQ} are run-constant while OLF's
     * {@code bus.getMinQ()} sums the generators that are still there. {@code rlEntry} names the
     * reactive-limit entry to correct and the two deltas are what to add to it (negative: the lost
     * generator's own share). {@code rlEntry < 0} when there is nothing to correct — the bus goes to PQ,
     * or it is not a reactive-limit controller at all.
     */
    /**
     * This bus was a DISTR_Q controller and stops being one in this scenario, while the group's OTHER
     * controllers survive. Its reactive key must leave the renormalization denominator on the TARGET
     * side too — {@code fillDistrQBatch} gets that from the qPct override channel, but
     * {@code updateDistrQTargetsBatch} had no equivalent and kept using the dead controller's key.
     * -1 when nothing changes. See the DISTR_Q gate in GpuBatchInjectionOutages for why the bus keeps
     * its BUS_TARGET_V while this happens.
     */
    public boolean deactivatesDqMember() {
        return inactiveDqMember >= 0;
    }

    /**
     * Deactivating a DISTR_Q member renormalizes qPct over the survivors, and the row's TARGET
     * (qPct_k * sum_j targetQ_j - targetQ_k, AcTargetVector.getGeneratorReactivePowerDistributionTarget)
     * moves with it. {@code fillDistrQBatch} applies the mask to the MATRIX unconditionally, but the
     * kernel that maintains the targets runs only inside the reactive-limits phase and only for a
     * scenario the limits are live on — so a scenario with no pin left the renormalized row solving
     * against the BASE target. Measured on rte6515 GEN-1263/GEN-1264: the surviving row solved against
     * 0.016 where OLF's is -0.060 and 0.092, and OLF's own residual at the device's state was 0.076 on
     * exactly one equation, the deactivated member's reactive balance. These (row, target) pairs seed
     * every member row of the group with the renormalized value; when the reactive-limits kernel does
     * run it recomputes the same expression from the live pin states and supersedes them.
     */
    /**
     * The row mode the PV-&gt;PQ flip installs on this bus's BUS_V row: 1 (a plain reactive-power
     * equation) for an ordinary bus, 4 for a REMOTE GROUP MEMBER whose control dies with it. Mode 4
     * means "this row hosts its own Q balance", which is what OLF re-activates there — and it is also
     * what makes the group's relocation mode-scan skip the row, so the pilot's own reactive balance
     * falls back to the pilot's row and {@code updatePilotVeqBatch} drops its now-meaningless voltage
     * identity. Installing mode 1 instead would leave the row looking like a live carrier, and the
     * pilot's balance would land on top of the member's own.
     */
    /**
     * When this outage kills the group's control, the PILOT loses its voltage identity and goes back
     * to solving its own reactive balance — {@code deadPilotVRow} is its BUS_V row and
     * {@code deadPilotQTarget} that balance's target. It has to be installed at SETUP, not left to
     * {@code updatePilotVeqBatch} in the reactive-limits phase: this bus's row is mode 4 from the
     * first fill, so the group's mode scan already sends the pilot's q to the pilot's own row, and a
     * row that still carries the voltage identity would receive both.
     */
    public boolean killsPilotVoltageIdentity() {
        return deadPilotVRow >= 0;
    }

    /** {@code pqRowMode} value meaning "the V row becomes a VOLTAGE IDENTITY (mode 0)": a merged voltage
     *  control re-homed to this bus by a zero-impedance split (see GpuBatchQualification.reHomedControls),
     *  with {@code qTarget} then carrying the bus's own voltage target. 0 is the "default" (1). */
    public static final int ROW_V_IDENTITY = -1;
    /** A re-homed control whose bus is PINNED at a reactive limit in the base: the row starts as its own
     *  reactive balance (mode 4) at {@code qTarget} (the pinned generation minus the load), and its reactive-limit
     *  entry is checked as a LOCAL PV bus at {@code deadPilotQTarget}, which here carries the VOLTAGE target —
     *  an unpin restores the voltage identity exactly as OLF re-activates BUS_TARGET_V (tasks #70,
     *  VC_REMERGE_PINNED_MAIN; rte6515 BUS-6157_BBS_1 unpins and re-pins at the other limit). */
    public static final int ROW_PINNED_V = -5;
    /** A base-pinned member of a re-homed cluster with NO enabled controller left (every controller pinned):
     *  no row change — the stock placement already keeps its balance at the limit — but an UNPIN would need a
     *  voltage identity the device cannot host, so the kernel rejects the scenario instead (CPU replay). */
    public static final int ROW_PINNED_NO_UNPIN = -6;
    /** A re-homed control whose own V row hosts a LIVE coupler's ZERO_V: the voltage identity goes on the parent
     *  coupler's DUMMY_Q row (where the bus's reactive equation lives) as a CROSS identity on the bus's own V
     *  column — device row mode 6 — with {@code vRow} = that dummy row and {@code qTarget} the voltage target
     *  (tasks #70, VC_REMERGE_RELOC_VROW). */
    public static final int ROW_CROSS_V_IDENTITY = -7;

    public int pqRowModeOrDefault() {
        return pqRowMode == 0 ? 1 : pqRowMode == ROW_PINNED_V ? 4 : pqRowMode == ROW_CROSS_V_IDENTITY ? 6
                : pqRowMode < 0 ? 0 : pqRowMode;
    }

    /** The voltage target a re-homed control is checked against ({@link #ROW_V_IDENTITY}: the row target itself;
     *  {@link #ROW_PINNED_V}: carried in deadPilotQTarget, the row target being the pinned Q). */
    public double reHomeVTarget() {
        return pqRowMode == ROW_PINNED_V ? deadPilotQTarget : qTarget;
    }

    public boolean forbidsUnpin() {
        return pqRowMode == ROW_PINNED_NO_UNPIN && rlEntry >= 0;
    }

    public boolean seedsDqTargets() {
        return dqTargetRows.length > 0;
    }

    /**
     * The pilot's own BUS_V row and the reactive
     * balance it must carry once every surviving controller has pinned. While the group still holds
     * the pilot's voltage, setup correctly leaves this row a mode-0 BUS_TARGET_V identity and the
     * balance rides a survivor's row. When the last survivor pins, that row takes its own pinned
     * balance and the pilot's is evicted; OLF drops the now-uncontrollable BUS_TARGET_V and activates
     * the pilot's own BUS_TARGET_Q on the freed row. The device makes the same move IN the solve, by
     * putting the row in mode 4 — which reaches the pilot's q only because the extraction now gives a
     * pilot-that-is-its-own-controller the member ALTERNATE dest.
     */
    public boolean flipsPilotRowWhenAllPinned() {
        return pinnedPilotVRow >= 0 && pilotGroupSurvivorRlEntries.length > 0;
    }

    /** This outage moved a packed DISTR_Q member's OWN targetQ (a reactive LOAD loss on a group member).
     *  Every target in the group reads that value twice — inside the shared sum and as its own
     *  subtracted term — and the device's target kernel recomputes them from the base value on every
     *  reactive-limits pass, so the new value has to travel as an override rather than a row seed. */
    public boolean overridesDqMemberTargetQ() {
        return dqTargetQOvMember >= 0;
    }

    /** The bus keeps NO generator: its reactive-limit entry is not a controller in this scenario. */
    /** A re-homed voltage control (see {@link #ROW_V_IDENTITY}): its bus is a plain PV bus this scenario, so its
     *  reactive-limit entry — packed as a remote member of the base group — is checked as a LOCAL controller
     *  (pin on its own row, switch back on its own voltage) at the target {@code qTarget} carries. */
    public boolean reHomesRlEntry() {
        return (pqRowMode == ROW_V_IDENTITY || pqRowMode == ROW_PINNED_V || pqRowMode == ROW_CROSS_V_IDENTITY) && rlEntry >= 0;
    }

    public boolean silencesReactiveLimitEntry() {
        return rlEntry >= 0 && allGensLost;
    }

    /** Anything this outage changes about the entry's run-constant reactive inputs: the capability the
     *  lost generators take with them, and the load target Q the pinned target and the controller's own
     *  q are both measured against. */
    public boolean shrinksReactiveLimits() {
        return rlEntry >= 0 && (minQDelta != 0 || maxQDelta != 0 || loadQDelta != 0);
    }
}
