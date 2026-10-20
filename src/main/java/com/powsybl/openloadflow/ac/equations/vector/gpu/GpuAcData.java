/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuQLimitType;

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
 * <p>DISTR_Q (remote generator voltage control): {@link DistrQTerms} and {@link DistrQMembers}. Empty when no
 * remote generator voltage control is active.
 *
 * <p>Zero-impedance branches (bus couplers) are a LINEAR family: {@code ziIn} {@code nzi*1} (rho = R2/r1),
 * idx {@code nzi*11} (b1Prow b1Qrow b2Prow b2Qrow dpRow dqRow v1col v2col ph1col ph2col enabledTree). Each
 * adds DUMMY_P/DUMMY_Q variables whose columns ({@code dpRow}/{@code dqRow}) enter both buses' P/Q balances
 * (±1); a spanning-tree edge ({@code enabledTree}=1) also carries the voltage/angle coupling
 * ZERO_V (v1−rho·v2) on the DUMMY_P row and ZERO_PHI (ph1−ph2) on the DUMMY_Q row.
 *
 * <p>DISTR_RHO and DISTR_SHUNT_B are {@link LinearDistributionTerms}.
 *
 * <p>Reactive limits: {@link ReactiveLimitEntries}. The calculated generator Q of an entry is
 * {@code sum branch q + shunt q + loadQ}, the first part read from its BUS_V row after a fill into a zeroed buffer.
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
public record GpuAcData(ClosedBranchPack closedBranch, OpenBranchPack openBranch, ElementPack shunt,
                        ElementPack hvdcAcEmulation,
                        DistrQTerms distrQ, DistrQMembers distrQMembers,
                        ZeroImpedanceBranchPack zeroImpedanceBranch,
                        LinearDistributionTerms distrRho, LinearDistributionTerms distrShuntB,
                        ReactiveLimitEntries reactiveLimits,
                        RemoteMemberReactiveTerms remoteMemberReactiveTerms,
                        VoltageRowParameters voltageRows,
                        DistributedSlackParticipants distributedSlack,
                        BaseOpenBranches baseOpenBranches,
                        ElementPack loadModel, ReactiveRelocationCandidates reactiveRelocation) {

    /**
     * Content equality over every array — true means a cached solver context built from
     * {@code other} (structure AND element parameters) is still valid for this data. With
     * the stable row indexing this holds across activation switching (PV→PQ), so outer
     * loops virtually always reuse; it goes false only on real topology/parameter changes.
     */
    public boolean sameAs(GpuAcData other) {
        return sameStructureAs(other)
                && closedBranch.sameParametersAs(other.closedBranch) && openBranch.sameParametersAs(other.openBranch)
                && shunt.sameParametersAs(other.shunt) && hvdcAcEmulation.sameParametersAs(other.hvdcAcEmulation)
                && zeroImpedanceBranch.sameParametersAs(other.zeroImpedanceBranch) && loadModel.sameParametersAs(other.loadModel);
        // distrRho's coefficients are structural (qPct/count-derived); covered by sameStructureAs below.
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
        if (other != null && !baseOpenBranches.sameElementsAs(other.baseOpenBranches)) {
            return false;
        }
        return other != null
                && closedBranch.sameParametersAs(other.closedBranch) && closedBranch.sameIndicesAs(other.closedBranch)
                && openBranch.sameParametersAs(other.openBranch) && openBranch.sameIndicesAs(other.openBranch)
                && shunt.sameParametersAs(other.shunt) && shunt.sameIndicesAs(other.shunt)
                && hvdcAcEmulation.sameParametersAs(other.hvdcAcEmulation) && hvdcAcEmulation.sameIndicesAs(other.hvdcAcEmulation)
                && distrQ.sameTermsAs(other.distrQ)
                && zeroImpedanceBranch.sameParametersAs(other.zeroImpedanceBranch)
                && zeroImpedanceBranch.sameIndicesAs(other.zeroImpedanceBranch)
                && distrRho.sameAs(other.distrRho) && distrShuntB.sameAs(other.distrShuntB)
                && loadModel.sameParametersAs(other.loadModel) && loadModel.sameIndicesAs(other.loadModel)
                // the candidate tables build the group relocation pools (buildCbDynDest & co.)
                && reactiveRelocation.sameAs(other.reactiveRelocation);
    }

    /**
     * Structural equality over the INDEX packs ONLY (the variable columns each element touches, hence
     * the CSR pattern + diagonal pairing). True means a cached context's pattern and cuDSS symbolic
     * analysis are still valid and only the element PARAMETER packs may differ — the VALUES invalidation
     * level: a cheap in-place re-upload (refreshStructureValues) instead of a full rebuild + re-analysis.
     * A transformer tap or shunt-susceptance move changes {@code cbIn}/{@code shIn} values but not the
     * indices, so it is {@code sameStructureAs} but not {@code sameAs}.
     */
    public boolean sameStructureAs(GpuAcData other) {
        return other != null
                && closedBranch.sameIndicesAs(other.closedBranch) && openBranch.sameIndicesAs(other.openBranch)
                && shunt.sameIndicesAs(other.shunt) && hvdcAcEmulation.sameIndicesAs(other.hvdcAcEmulation)
                && distrQ.sameAs(other.distrQ)
                && zeroImpedanceBranch.sameIndicesAs(other.zeroImpedanceBranch)
                && distrRho.sameAs(other.distrRho) && distrShuntB.sameAs(other.distrShuntB)
                && reactiveLimits.sameAs(other.reactiveLimits)
                && distributedSlack.sameAs(other.distributedSlack)
                && reactiveRelocation.sameGroupsAs(other.reactiveRelocation)
                && distrQMembers.sameAs(other.distrQMembers)
                && remoteMemberReactiveTerms.sameAs(other.remoteMemberReactiveTerms)
                && voltageRows.sameAs(other.voltageRows);
    }

    public int getClosedBranchCount() {
        return closedBranch.indices().length / 12;
    }

    public int getOpenBranchCount() {
        return openBranch.indices().length / 6;
    }

    public int getShuntCount() {
        return shunt.indices().length / 5;
    }

    public int getHvdcCount() {
        return hvdcAcEmulation.indices().length / 4;
    }

    public int getDistrQContributionCount() {
        return distrQ.size();
    }

    public int getZeroImpedanceCount() {
        return zeroImpedanceBranch.indices().length / 11;
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
     * <p>Used by {@code GpuHvdcAcEmulationLimitsOuterLoop} for the converter buses' P-target rows since 2026-10-05. The
     * first attempt (2026-10-02) was measured "catastrophic" (max-iter-bad-solve 108 -> 13194) and blamed the
     * premise; the premise was right — a relocated child's P balance DOES live on the parent's {@code DUMMY_P}
     * row, the child's own PHI row hosting the coupler's {@code ZERO_PHI} — and the failure was the kernel
     * reading the bus ANGLE at the same index it wrote the target to (a {@code DUMMY_P} variable read as an
     * angle). The angle column and the target row are separate arrays now, and a scenario that opens the parent
     * coupler diverts the write to the own row (relocOwn). Measured on rte6515 BUS-1219_BBS: the saturation
     * delta of SYN_HVDC_4 landing on a ZERO_PHI row reversed the emulated flow and the limits loop toggled
     * between the two saturation sides to its budget — the one device failure on an OLF-converging busbar
     * contingency (tasks #68 / #69).
     */
    public int relocatedPRow(int phiCol) {
        int[] ziIdx = zeroImpedanceBranch.indices();
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
        return distrRho.row().length;
    }

    /** Number of local PV controller buses subject to the batched reactive-limits (PV→PQ) outer loop. */
    public int getReactiveLimitCount() {
        return reactiveLimits.size();
    }

    /**
     * Did OLF's {@code ReactiveLimitsOuterLoop} switch this entry's bus PV -> PQ in the base load flow, at any of
     * the four {@code LfBus.QLimitType}s? False for no entry ({@code rlEntry < 0}).
     */
    public boolean switchedPvToPqInBase(int rlEntry) {
        byte[] baseQLimitType = reactiveLimits.baseQLimitType();
        return baseQLimitType != null && rlEntry >= 0 && rlEntry < baseQLimitType.length && baseQLimitType[rlEntry] != GpuQLimitType.NONE;
    }

    /** Number of participating buses in the batched distributed-slack outer loop (0 = no distribution). */
    public int getDistributedSlackBusCount() {
        return distributedSlack.size();
    }

    public int getTotalElementCount() {
        return getClosedBranchCount() + getOpenBranchCount() + getShuntCount() + getHvdcCount();
    }
}
