/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

/**
 * The flat per-family element data the GPU Newton-Raphson consumes, extracted from an
 * OLF equation system by {@link GpuAcDataExtractor}. Rows use the STABLE indexing (a
 * row is its diagonal-paired variable row — P rows are BUS_PHI rows, Q rows BUS_V
 * rows), so this structure is the fixed superset, identical across activation
 * switching. Packings (an index of -1 marks an absent branch side):
 * <ul>
 *   <li>closed branches — in {@code ncb*8} (y g1 g2 b1 b2 ksi r1 a1),
 *       idx {@code ncb*12} (Prow1 Qrow1 Prow2 Qrow2 ph1col v1col ph2col v2col a1col r1col qAlt1 qAlt2).
 *       {@code a1col}/{@code r1col} are the BRANCH_ALPHA1/BRANCH_RHO1 variable columns when the
 *       branch derives that control (transformer phase/voltage control), else -1; a controlling
 *       transformer also routes the controlled bus's Q row to its {@code r1col} (see GpuAcDataExtractor)</li>
 *   <li>open branches — in {@code nob*7} (y ksi g1 b1 g2 b2 r1),
 *       idx {@code nob*6} (Prow1 Qrow1 Prow2 Qrow2 v1col v2col)</li>
 *   <li>shunts — in {@code nsh*2} (g b), idx {@code nsh*4} (Prow Qrow vcol bcol) — {@code bcol} is the
 *       shunt's SHUNT_B variable row when it derives voltage control (b read from the state vector; dq/db
 *       scattered onto that column), else -1 (fixed shunt, b constant); a controlling shunt also routes the
 *       controlled bus's Q row to its {@code bcol}, the analogue of the transformer rho pairing</li>
 *   <li>HVDC AC emulation — in {@code nhv*5} (p0 k lossFactor1 lossFactor2 r),
 *       idx {@code nhv*4} (Prow1 Prow2 ph1col ph2col)</li>
 * </ul>
 *
 * <p>DISTR_Q (remote generator voltage control) reactive-distribution rows are assembled from a list
 * of weighted closed-branch reactive contributions, one per (controller branch side, DISTR_Q row):
 * {@code dqElem} the closed-branch element index, {@code dqSide} 0/1 (side 1/2 reactive q), {@code dqRow}
 * the stable DISTR_Q row (a controller's BUS_V row), {@code dqWeight} the coefficient
 * ({@code qPercent_k} for a foreign controller, {@code qPercent_k - 1} for the row's own controller).
 * Empty when no remote generator voltage control is active.
 *
 * <p>Zero-impedance branches (bus couplers) are a LINEAR family: {@code ziIn} {@code nzi*1} (rho = R2/r1),
 * idx {@code nzi*11} (b1Prow b1Qrow b2Prow b2Qrow dpRow dqRow v1col v2col ph1col ph2col enabledTree). Each
 * adds DUMMY_P/DUMMY_Q variables whose columns ({@code dpRow}/{@code dqRow}) enter both buses' P/Q balances
 * (±1); a spanning-tree edge ({@code enabledTree}=1) also carries the voltage/angle coupling
 * ZERO_V (v1−rho·v2) on the DUMMY_P row and ZERO_PHI (ph1−ph2) on the DUMMY_Q row.
 *
 * <p>DISTR_RHO (ratio distribution across several transformers controlling one bus) is a LINEAR family of
 * contributions: {@code drRow} the stable DISTR_RHO row (a controller's BRANCH_RHO1 row), {@code drCol} a
 * controller's BRANCH_RHO1 column, {@code drCoef} the constant coefficient ({@code 1/count} for a foreign
 * controller, {@code 1/count - 1} for the row's own). Empty unless several transformers share one bus.
 *
 * <p>Reactive limits (the batched full-loadflow PV↔PQ outer loop): per LOCAL generator-voltage-controller
 * (PV) bus, {@code rlVrow} its stable BUS_V row (where a PV→PQ switch flips the row mode 0→1 and the target
 * Vtarget→qLimit−loadQ; a PQ→PV switchback restores mode 1→0 and target→{@code rlVtarget}), {@code rlMinQ}/
 * {@code rlMaxQ} its reactive limits (pu/SB), {@code rlLoadQ} its load target Q and {@code rlVtarget} its
 * controlled voltage target (pu). The calculated generator Q is {@code Σ branch q + shunt q + rlLoadQ}, the
 * first part read from the bus's BUS_V row after a fill into a zeroed buffer. Empty when no PV control (or
 * consumed only by the batched full-loadflow path; the single-solve path runs outer loops on the CPU).
 *
 * <p>Distributed slack (the batched full-loadflow outer loop): {@code dsSlackPhiRow} the slack bus's stable
 * BUS_PHI row (-1 if none) and {@code dsSlackTargetP} its scheduled active target (pu) give the mismatch
 * {@code Σ branch p at the slack − dsSlackTargetP}; each participating non-slack bus {@code dsPhiRow} (its
 * BUS_TARGET_P row) takes {@code dsFactor × mismatch} into its P target, with {@code dsFactor} the NORMALIZED
 * participation factor (PROPORTIONAL_TO_GENERATION_P_MAX: {@code Σ maxP/droop} over the bus's participating
 * generators) normalized by {@code Σ maxP/droop over ALL participants INCLUDING any at the slack bus} — so a
 * participating slack generator's share is left at the slack (no row emitted) instead of over-distributed.
 * {@code dsBaseTargetP} is each emitted bus's initial P target and {@code dsMaxDelta}/{@code dsMinDelta} its
 * headroom/footroom ({@code Σ (maxTargetP−targetP)} / {@code Σ (minTargetP−targetP)} over participating gens,
 * OLF's no-sign-change rule, useActiveLimits assumed true): the batch clamps each bus's CUMULATIVE delta to
 * [footroom, headroom], so saturation drop-out + redistribution converge to OLF's water-filling.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record GpuAcData(double[] cbIn, int[] cbIdx, int[] cbPcSide, double[] obIn, int[] obIdx,
                        double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx,
                        int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind,
                        int[] dqKMember, int[] dqJMember, double[] dqMemberKey, int[] dqMemberRlEntry,
                        int[] dqMemberGroupStart, int[] dqMemberGroupCount, int[] dqMemberRow, byte[] dqMemberHasRow,
                        int[] dqMemberQrGroup, double[] dqMemberTargetQ, int[] dqMemberBusNum,
                        int[] dqMemberTgtPtr, int[] dqMemberTgtPeer,
                        double[] ziIn, int[] ziIdx,
                        int[] drRow, int[] drCol, double[] drCoef,
                        int[] dbRow, int[] dbCol, double[] dbCoef,
                        int[] rlVrow, double[] rlMinQ, double[] rlMaxQ, double[] rlLoadQ, double[] rlVtarget,
                        double[] rlSvcBmin, double[] rlSvcBmax, double[] rlV2Base,
                        byte[] rlMonitored, byte[] rlRemote, int[] rlCmpVrow, int[] rlBusNum,
                        double[] rlNominalV, int[] rlPRow, double[] rlTargetP, int[] rlIdRank,
                        double[] rlInitTargetQ,
                        int[] rmqRow, int[] rmqElem, int[] rmqSide, int[] rmqKind, double[] rmqWeight,
                        double[] rowSlope, double[] rowNominalV,
                        int[] dsPhiRow, double[] dsFactor, int dsSlackPhiRow, double dsSlackTargetP,
                        double[] dsBaseTargetP, double[] dsMaxDelta, double[] dsMinDelta,
                        int dsMode, double[] dsMarginUp, double[] dsMarginDown,
                        double dsSlackMarginUp, double dsSlackMarginDown,
                        int[] dgBus, double[] dgFactor, double[] dgMaxDelta, double[] dgMinDelta,
                        double[] dgMarginUp, double[] dgMarginDown, int[] busGenStart, int[] busGenEnd,
                        int[] dsBusNum, String[] dgGenId,
                        int[] baseOpenCbElements, int[] baseOpenZiElements, int[] baseOpenZiBranchNums,
                        int[] ziBranchNums, double[] vdlIn, int[] vdlIdx, int[] qrPtr, int[] qrRows,
                        int[] obQAlt, int[] ziQAlt, byte[] rlBaseQLimitType) {

    /**
     * Content equality over every array — true means a cached solver context built from
     * {@code other} (structure AND element parameters) is still valid for this data. With
     * the stable row indexing this holds across activation switching (PV→PQ), so outer
     * loops virtually always reuse; it goes false only on real topology/parameter changes.
     */
    public boolean sameAs(GpuAcData other) {
        return sameStructureAs(other)
                && java.util.Arrays.equals(cbIn, other.cbIn) && java.util.Arrays.equals(obIn, other.obIn)
                && java.util.Arrays.equals(shIn, other.shIn) && java.util.Arrays.equals(hvIn, other.hvIn)
                && java.util.Arrays.equals(ziIn, other.ziIn) && java.util.Arrays.equals(vdlIn, other.vdlIn);
        // drCoef is structural (qPct/count-derived); covered by sameStructureAs below.
    }

    /**
     * Content equality over the SHARED-STRUCTURE inputs only — the family packs (values AND indices),
     * DISTR_Q, zero-impedance and DISTR_RHO — i.e. exactly what {@code GpuAcNewtonSolver.createStructure}
     * consumes. The reactive-limits / distributed-slack packs are EXCLUDED: they are batch-CONTEXT
     * inputs, not structure inputs, so two extraction FLAVORS of one equation-system state (the base
     * load flow always extracts DS_RL; an SA run may extract with DS or RL disabled) compare equal here
     * and can share one device-resident structure (see {@code GpuAcStructures}).
     */
    public boolean sameStructureInputsAs(GpuAcData other) {
        if (other != null && (!java.util.Arrays.equals(baseOpenCbElements, other.baseOpenCbElements)
                || !java.util.Arrays.equals(baseOpenZiElements, other.baseOpenZiElements))) {
            return false;
        }
        return other != null
                && java.util.Arrays.equals(cbIn, other.cbIn) && java.util.Arrays.equals(cbIdx, other.cbIdx)
                && java.util.Arrays.equals(obIn, other.obIn) && java.util.Arrays.equals(obIdx, other.obIdx)
                && java.util.Arrays.equals(shIn, other.shIn) && java.util.Arrays.equals(shIdx, other.shIdx)
                && java.util.Arrays.equals(hvIn, other.hvIn) && java.util.Arrays.equals(hvIdx, other.hvIdx)
                && java.util.Arrays.equals(dqElem, other.dqElem) && java.util.Arrays.equals(dqSide, other.dqSide)
                && java.util.Arrays.equals(dqWeight, other.dqWeight) && java.util.Arrays.equals(dqRow, other.dqRow)
                && java.util.Arrays.equals(dqKind, other.dqKind)
                && java.util.Arrays.equals(ziIn, other.ziIn) && java.util.Arrays.equals(ziIdx, other.ziIdx)
                && java.util.Arrays.equals(drRow, other.drRow) && java.util.Arrays.equals(drCol, other.drCol)
                && java.util.Arrays.equals(drCoef, other.drCoef)
                && java.util.Arrays.equals(dbRow, other.dbRow) && java.util.Arrays.equals(dbCol, other.dbCol)
                && java.util.Arrays.equals(dbCoef, other.dbCoef)
                && java.util.Arrays.equals(vdlIn, other.vdlIn) && java.util.Arrays.equals(vdlIdx, other.vdlIdx)
                // structure input too: the phase-control alt-dest tables (pcDest/pcSel) are built from it
                && java.util.Arrays.equals(cbPcSide, other.cbPcSide)
                && java.util.Arrays.equals(obQAlt, other.obQAlt) && java.util.Arrays.equals(ziQAlt, other.ziQAlt)
                // the candidate tables build the group relocation pools (buildCbDynDest & co.)
                && java.util.Arrays.equals(qrPtr, other.qrPtr) && java.util.Arrays.equals(qrRows, other.qrRows);
    }

    /**
     * Structural equality over the INDEX packs ONLY (the variable columns each element touches, hence
     * the CSR pattern + diagonal pairing). True means a cached context's pattern and cuDSS symbolic
     * analysis are still valid and only the element PARAMETER packs may differ — the VALUES invalidation
     * level: a cheap in-place re-upload (refreshContextValues) instead of a full rebuild + re-analysis.
     * A transformer tap or shunt-susceptance move changes {@code cbIn}/{@code shIn} values but not the
     * indices, so it is {@code sameStructureAs} but not {@code sameAs}.
     */
    public boolean sameStructureAs(GpuAcData other) {
        return other != null
                && java.util.Arrays.equals(cbIdx, other.cbIdx) && java.util.Arrays.equals(obIdx, other.obIdx)
                && java.util.Arrays.equals(shIdx, other.shIdx) && java.util.Arrays.equals(hvIdx, other.hvIdx)
                && java.util.Arrays.equals(dqElem, other.dqElem) && java.util.Arrays.equals(dqSide, other.dqSide)
                && java.util.Arrays.equals(dqRow, other.dqRow) && java.util.Arrays.equals(dqWeight, other.dqWeight)
                && java.util.Arrays.equals(dqKind, other.dqKind)
                && java.util.Arrays.equals(ziIdx, other.ziIdx)
                && java.util.Arrays.equals(drRow, other.drRow) && java.util.Arrays.equals(drCol, other.drCol)
                && java.util.Arrays.equals(drCoef, other.drCoef)
                && java.util.Arrays.equals(dbRow, other.dbRow) && java.util.Arrays.equals(dbCol, other.dbCol)
                && java.util.Arrays.equals(dbCoef, other.dbCoef)
                && java.util.Arrays.equals(rlVrow, other.rlVrow) && java.util.Arrays.equals(rlMinQ, other.rlMinQ)
                && java.util.Arrays.equals(rlMaxQ, other.rlMaxQ) && java.util.Arrays.equals(rlLoadQ, other.rlLoadQ)
                && java.util.Arrays.equals(rlVtarget, other.rlVtarget)
                && java.util.Arrays.equals(rlSvcBmin, other.rlSvcBmin)
                && java.util.Arrays.equals(rlSvcBmax, other.rlSvcBmax)
                && java.util.Arrays.equals(rlV2Base, other.rlV2Base)
                && java.util.Arrays.equals(dsPhiRow, other.dsPhiRow) && java.util.Arrays.equals(dsFactor, other.dsFactor)
                && dsSlackPhiRow == other.dsSlackPhiRow && dsSlackTargetP == other.dsSlackTargetP
                && java.util.Arrays.equals(dsBaseTargetP, other.dsBaseTargetP)
                && java.util.Arrays.equals(dsMaxDelta, other.dsMaxDelta) && java.util.Arrays.equals(dsMinDelta, other.dsMinDelta)
                && java.util.Arrays.equals(qrPtr, other.qrPtr) && java.util.Arrays.equals(qrRows, other.qrRows)
                && java.util.Arrays.equals(obQAlt, other.obQAlt) && java.util.Arrays.equals(ziQAlt, other.ziQAlt)
                && java.util.Arrays.equals(rlMonitored, other.rlMonitored)
                && java.util.Arrays.equals(rlRemote, other.rlRemote)
                && java.util.Arrays.equals(rlBaseQLimitType, other.rlBaseQLimitType)
                && java.util.Arrays.equals(rlCmpVrow, other.rlCmpVrow)
                && java.util.Arrays.equals(rlInitTargetQ, other.rlInitTargetQ)
                && java.util.Arrays.equals(dqKMember, other.dqKMember) && java.util.Arrays.equals(dqJMember, other.dqJMember)
                && java.util.Arrays.equals(dqMemberKey, other.dqMemberKey)
                && java.util.Arrays.equals(dqMemberRlEntry, other.dqMemberRlEntry)
                && java.util.Arrays.equals(dqMemberGroupStart, other.dqMemberGroupStart)
                && java.util.Arrays.equals(dqMemberGroupCount, other.dqMemberGroupCount)
                && java.util.Arrays.equals(dqMemberRow, other.dqMemberRow)
                && java.util.Arrays.equals(dqMemberHasRow, other.dqMemberHasRow)
                && java.util.Arrays.equals(dqMemberQrGroup, other.dqMemberQrGroup)
                && java.util.Arrays.equals(dqMemberTargetQ, other.dqMemberTargetQ)
                && java.util.Arrays.equals(dqMemberBusNum, other.dqMemberBusNum)
                && java.util.Arrays.equals(rmqRow, other.rmqRow) && java.util.Arrays.equals(rmqElem, other.rmqElem)
                && java.util.Arrays.equals(rmqSide, other.rmqSide) && java.util.Arrays.equals(rmqKind, other.rmqKind)
                && java.util.Arrays.equals(rmqWeight, other.rmqWeight)
                && java.util.Arrays.equals(rowSlope, other.rowSlope)
                && java.util.Arrays.equals(rowNominalV, other.rowNominalV)
                && java.util.Arrays.equals(cbPcSide, other.cbPcSide)
                && dsMode == other.dsMode
                && java.util.Arrays.equals(dsMarginUp, other.dsMarginUp) && java.util.Arrays.equals(dsMarginDown, other.dsMarginDown)
                && dsSlackMarginUp == other.dsSlackMarginUp && dsSlackMarginDown == other.dsSlackMarginDown;
    }

    public int getClosedBranchCount() {
        return cbIdx.length / 12;
    }

    public int getOpenBranchCount() {
        return obIdx.length / 6;
    }

    public int getShuntCount() {
        return shIdx.length / 5;
    }

    public int getHvdcCount() {
        return hvIdx.length / 4;
    }

    public int getDistrQContributionCount() {
        return dqElem.length;
    }

    public int getZeroImpedanceCount() {
        return ziIdx.length / 11;
    }

    /**
     * The relocated {@code DUMMY_P} row that hosts the ACTIVE-power balance of the bus whose own
     * {@code BUS_PHI} column is {@code phiCol}, or {@code phiCol} itself when that bus is not a relocated
     * coupler child.
     *
     * <p>Every consumer of "where does this bus's P equation live" must ask this, exactly as the reactive
     * side must ask for the relocated Q row. Under the coupler-tree relocation a child's P balance moves to
     * its parent coupler's {@code DUMMY_P} row while its own {@code BUS_PHI} row hosts the coupler's
     * {@code ZERO_PHI} — so a row index that is correct as an ANGLE COLUMN is the WRONG place to write a P
     * target. Writing there puts the target on the angle identity: measured on rte6515 (tasks #68), the HVDC
     * AC-emulation freeze setpoints landed on all ten VSC-station breakers' {@code ZERO_PHI} rows, whose
     * target must be 0, while the {@code DUMMY_P} rows that actually carry the HVDC p1/p2 terms had target 0.
     *
     * <p>Derived from the same zero-impedance pack as the reactive twin, so the two cannot drift.
     *
     * <p><b>UNUSED, and kept as a WARNING.</b> The one caller it was written for — the HVDC
     * AC-emulation P target in {@code GpuBatchHvdcAcEmulation} — was measured CATASTROPHICALLY worse
     * with it: {@code max-iter-bad-solve} went from 108 to 13194 on rte6515's full busbar scope. The
     * premise was that a relocated child's P target must follow its P balance to the parent's
     * {@code DUMMY_P} row, inferred from the equation dump's {@code terms=} listing; that inference is
     * wrong, because a spanning-tree edge carries {@code ZERO_V} on the {@code DUMMY_P} row and
     * {@code ZERO_PHI} on the {@code DUMMY_Q} row (see the class contract above), not on the buses' own
     * rows. Do not wire this into a target writer without re-deriving the arrangement first.
     */
    public int relocatedPRow(int phiCol) {
        if (ziIdx == null || phiCol < 0) {
            return phiCol;
        }
        for (int e = 0; e < ziIdx.length / 11; e++) {
            int o = e * 11;
            int childSide = (ziIdx[o + 10] >> 1) & 3;
            int ownPhi = childSide == 1 ? ziIdx[o + 8] : (childSide == 2 ? ziIdx[o + 9] : -1);
            if (ownPhi == phiCol && ziIdx[o + 4] >= 0) {
                return ziIdx[o + 4];
            }
        }
        return phiCol;
    }

    public int getDistrRhoContributionCount() {
        return drRow.length;
    }

    /** Number of local PV controller buses subject to the batched reactive-limits (PV→PQ) outer loop. */
    public int getReactiveLimitCount() {
        return rlVrow.length;
    }

    /** Number of participating buses in the batched distributed-slack outer loop (0 = no distribution). */
    public int getDistributedSlackBusCount() {
        return dsPhiRow.length;
    }

    public int getTotalElementCount() {
        return getClosedBranchCount() + getOpenBranchCount() + getShuntCount() + getHvdcCount();
    }
}
