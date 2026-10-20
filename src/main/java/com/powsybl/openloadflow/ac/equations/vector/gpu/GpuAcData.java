/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocationCandidates;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuQLimitType;

/**
 * The flat per-family element data the GPU Newton-Raphson consumes, extracted from an
 * OLF equation system by {@link GpuAcDataExtractor}. Rows use the STABLE indexing (a
 * row is its diagonal-paired variable row — P rows are BUS_PHI rows, Q rows BUS_V
 * rows), so this structure is the fixed superset, identical across activation
 * switching. Packings (an index of -1 marks an absent branch side):
 * <ul>
 *   <li>closed branches — in {@code nClosedBranch*8} (y g1 g2 b1 b2 ksi r1 a1),
 *       idx {@code nClosedBranch*12} (Prow1 Qrow1 Prow2 Qrow2 ph1col v1col ph2col v2col a1col r1col alternateQRow1 alternateQRow2).
 *       {@code a1col}/{@code r1col} are the BRANCH_ALPHA1/BRANCH_RHO1 variable columns when the
 *       branch derives that control (transformer phase/voltage control), else -1; a controlling
 *       transformer also routes the controlled bus's Q row to its {@code r1col} (see GpuAcDataExtractor)</li>
 *   <li>open branches — in {@code nOpenBranch*7} (y ksi g1 b1 g2 b2 r1),
 *       idx {@code nOpenBranch*6} (Prow1 Qrow1 Prow2 Qrow2 v1col v2col)</li>
 *   <li>shunts — in {@code nShunt*2} (g b), idx {@code nShunt*4} (Prow Qrow vcol bcol) — {@code bcol} is the
 *       shunt's SHUNT_B variable row when it derives voltage control (b read from the state vector; dq/db
 *       scattered onto that column), else -1 (fixed shunt, b constant); a controlling shunt also routes the
 *       controlled bus's Q row to its {@code bcol}, the analogue of the transformer rho pairing</li>
 *   <li>HVDC AC emulation — in {@code nHvdcAcEmulation*5} (p0 k lossFactor1 lossFactor2 r),
 *       idx {@code nHvdcAcEmulation*4} (Prow1 Prow2 ph1col ph2col)</li>
 * </ul>
 *
 * <p>DISTR_Q (remote generator voltage control): {@link DistrQTerms} and {@link DistrQMembers}. Empty when no
 * remote generator voltage control is active.
 *
 * <p>Zero-impedance branches (bus couplers) are a LINEAR family: {@code zeroImpedanceBranchParameters} {@code nZeroImpedanceBranch*1} (rho = R2/r1),
 * idx {@code nZeroImpedanceBranch*11} (b1Prow b1Qrow b2Prow b2Qrow dpRow distrQRow v1col v2col ph1col ph2col enabledTree). Each
 * adds DUMMY_P/DUMMY_Q variables whose columns ({@code dpRow}/{@code distrQRow}) enter both buses' P/Q balances
 * (±1); a spanning-tree edge ({@code enabledTree}=1) also carries the voltage/angle coupling
 * ZERO_V (v1−rho·v2) on the DUMMY_P row and ZERO_PHI (ph1−ph2) on the DUMMY_Q row.
 *
 * <p>DISTR_RHO and DISTR_SHUNT_B are {@link LinearDistributionTerms}.
 *
 * <p>Reactive limits: {@link ReactiveLimitEntries}. The calculated generator Q of an entry is
 * {@code sum branch q + shunt q + loadQ}, the first part read from its BUS_V row after a fill into a zeroed buffer.
 *
 * <p>Distributed slack (the batched full-loadflow outer loop): {@code slackBusPhiRow} the slack bus's stable
 * BUS_PHI row (-1 if none) and {@code slackBusTargetP} its scheduled active target (pu) give the mismatch
 * {@code Σ branch p at the slack − slackBusTargetP}; each participating non-slack bus {@code distributedSlackBusPhiRow} (its
 * BUS_TARGET_P row) takes {@code distributedSlackBusFactor × mismatch} into its P target, with {@code distributedSlackBusFactor} the NORMALIZED
 * participation factor (PROPORTIONAL_TO_GENERATION_P_MAX: {@code Σ maxP/droop} over the bus's participating
 * generators) normalized by {@code Σ maxP/droop over ALL participants INCLUDING any at the slack bus} — so a
 * participating slack generator's share is left at the slack (no row emitted) instead of over-distributed.
 * {@code distributedSlackBusBaseTargetP} is each emitted bus's initial P target and {@code distributedSlackBusMaxDelta}/{@code distributedSlackBusMinDelta} its
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
     * inputs, not structure inputs, so two extractions of one equation-system state (the base load flow always
     * extracts with the distributed slack and reactive limits tables; an SA run may extract without one) compare
     * equal here and can share one device-resident structure (see {@code GpuAcStructures}).
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
                // the candidate tables build the group relocation pools (buildClosedBranchRelocationDestination & co.)
                && reactiveRelocation.sameAs(other.reactiveRelocation);
    }

    /**
     * Structural equality over the INDEX packs ONLY (the variable columns each element touches, hence
     * the CSR pattern + diagonal pairing). True means a cached context's pattern and cuDSS symbolic
     * analysis are still valid and only the element PARAMETER packs may differ — the VALUES invalidation
     * level: a cheap in-place re-upload (refreshStructureValues) instead of a full rebuild + re-analysis.
     * A transformer tap or shunt-susceptance move changes {@code closedBranchParameters}/{@code shuntParameters} values but not the
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

    public int getDistrRhoContributionCount() {
        return distrRho.row().length;
    }

    /** Number of local PV controller buses subject to the batched reactive-limits (PV→PQ) outer loop. */
    public int getReactiveLimitCount() {
        return reactiveLimits.size();
    }

    /**
     * Did OLF's {@code ReactiveLimitsOuterLoop} switch this entry's bus PV -> PQ in the base load flow, at any of
     * the four {@code LfBus.QLimitType}s? False for no entry ({@code reactiveLimitEntry < 0}).
     */
    public boolean switchedPvToPqInBase(int reactiveLimitEntry) {
        byte[] baseQLimitType = reactiveLimits.baseQLimitType();
        return baseQLimitType != null && reactiveLimitEntry >= 0 && reactiveLimitEntry < baseQLimitType.length && baseQLimitType[reactiveLimitEntry] != GpuQLimitType.NONE;
    }

    /** Number of participating buses in the batched distributed-slack outer loop (0 = no distribution). */
    public int getDistributedSlackBusCount() {
        return distributedSlack.size();
    }

    public int getTotalElementCount() {
        return getClosedBranchCount() + getOpenBranchCount() + getShuntCount() + getHvdcCount();
    }
}
