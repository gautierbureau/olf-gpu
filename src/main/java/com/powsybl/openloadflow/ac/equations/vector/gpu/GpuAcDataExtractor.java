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
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;

/**
 * Extracts the {@link GpuAcData} the GPU Newton-Raphson consumes from an OLF equation
 * system, mirroring exactly the wiring AcEquationSystemCreator produced (closed/open
 * branch, shunt and HVDC AC-emulation terms).
 *
 * <p>STABLE row indexing: every equation row is identified by its diagonal-paired
 * VARIABLE row — a bus's P row is its BUS_PHI variable row, its Q row its BUS_V row —
 * never by OLF's equation column, which is renumbered on every activation change
 * (PV→PQ switching). The extracted structure is therefore the FIXED SUPERSET pattern,
 * identical across outer-loop activation switching, so a cached GPU context survives
 * it. What activation actually decides is carried per solve instead:
 * {@link GpuAcActivationExtractor#extractActivation} (power equation vs slack/PV identity per row, applied as a
 * device-side mask, and the targets re-ordered to the stable rows by {@link GpuAcActivation#applyTargets}).
 *
 * <p>VALIDATES first, and throws {@link PowsyblException} on anything the GPU path does
 * not cover — unsupported variable types (e.g. transformer controls), unsupported
 * equation types (e.g. zero-impedance branches), unsupported active equation terms
 * (e.g. ZIP load models), and equation-array (vectorized) systems — so an unsupported
 * network fails loudly instead of converging to a wrong state. Disconnection-allowed
 * branches (security-analysis contingency candidates) are NOT rejected: each fixed
 * configuration is solved correctly and a structure change (outage) triggers a context
 * rebuild, so the closed/open switch is a between-solve event the extractor handles.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcDataExtractor {

    private GpuAcDataExtractor() {
    }

    public static GpuAcData extract(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es) {
        // Default to OLF's own defaults (slack distribution + reactive limits ON). The batched SA path passes
        // the RUN's actual outer-loop set so a disabled outer loop is never run on the device (an outer loop
        // must not run unconditionally).
        return extract(network, es, true, true);
    }

    /**
     * Extract the GPU data, building the distributed-slack (DS) and reactive-limits (RL) outer-loop packs
     * ONLY when the corresponding outer loop is enabled for this run. The batched device solver gates each
     * outer-loop pass on its pack being non-empty, so an empty pack means "do not run that outer loop" —
     * mirroring OLF, which only runs an outer loop when it is in the configured outer-loop list. Building a
     * pack from network capability alone (participating gens / PV buses) would make the device distribute
     * slack or switch PV→PQ even when the run disabled it, diverging from the CPU under any non-trivial
     * mismatch (latent on balanced networks, where the retarget is negligible).
     */
    public static GpuAcData extract(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                    boolean distributedSlack, boolean reactiveLimits) {
        return extract(network, es, distributedSlack, reactiveLimits, java.util.Set.of(), java.util.Set.of());
    }

    public static GpuAcData extract(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                    boolean distributedSlack, boolean reactiveLimits,
                                    java.util.Set<String> baseOpenBranchIds) {
        return extract(network, es, distributedSlack, reactiveLimits, baseOpenBranchIds, java.util.Set.of());
    }

    /**
     * Batch-SA flavor: additionally packs the given BASE-OPEN branches (disabled at base, both buses
     * present and energized — the shape {@code Networks.loadWithReconnectableElements} produces for
     * branches closable by an action: loaded CLOSED in a temp variant, then re-opened by connectivity,
     * so the pi model and both buses are valid). They are appended to the closed-branch pack and
     * reported in {@code baseOpenClosedBranchElements}; the batch solver DISABLES them in every scenario's mask
     * by default — a reclose action clears the bit. The plain-LF path never packs them (empty set).
     */
    public static GpuAcData extract(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                    boolean distributedSlack, boolean reactiveLimits,
                                    java.util.Set<String> baseOpenBranchIds,
                                    java.util.Set<String> baseOpenZiBranchIds) {
        return extract(network, es, distributedSlack, reactiveLimits, baseOpenBranchIds, baseOpenZiBranchIds,
                GpuDsParams.DEFAULT);
    }

    /**
     * Each bus's P-balance and Q-balance ROW, indexed by {@code LfBus.getNum()}: its own BUS_PHI /
     * BUS_V column, or its parent coupler's DUMMY_P / DUMMY_Q column under the coupler-tree
     * relocation ({@code ziTreeRelocation}), which is where the extraction actually writes that bus's
     * targets.
     *
     * <p>A consumer that resolves a bus's target row from the variable set instead lands on the
     * UNRELOCATED row for every bus under a coupler. That row is a coupling row whose target is 0, so
     * an absolute target override written there is a NO-OP and the injection it was meant to remove
     * stays in the balance. Measured on network.xiidm's full load scope: 4183 of 7494 contingencies
     * reported the wrong distributed active power (worst 148 MW) because the lost load never left the
     * system — the distributed-slack loop then read the surplus as a mismatch and undid the whole
     * loss pre-distribution. rte6515 has no couplers, so it never showed it.
     *
     * @return {@code {pRow, qRow, ziParentEdge, qRowAlt}} — the third names the coupler hosting each bus's power
     *         equations, or -1 when the bus was not relocated; the fourth is the ALTERNATE reactive row, where a
     *         reactive-limit pin puts the bus's own balance (mode 4): a relocated child's is its coupler's DUMMY_Q
     *         row, -1 or a negative group code otherwise.
     */
    public static int[][] busPowerRows(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es) {
        ExtractionColumns columns = ExtractionColumns.of(network, es);
        return new int[][] {columns.pRow(), columns.qRow(), columns.ziParentEdge(), columns.qRowAlt()};
    }

    public static GpuAcData extract(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                    boolean distributedSlack, boolean reactiveLimits,
                                    java.util.Set<String> baseOpenBranchIds,
                                    java.util.Set<String> baseOpenZiBranchIds, GpuDsParams dsParams) {
        GpuAcValidation.validate(es);
        ExtractionColumns columns = ExtractionColumns.of(network, es);
        GpuAcFamilyPacker.Packs family = GpuAcFamilyPacker.pack(network, columns, baseOpenBranchIds, baseOpenZiBranchIds);
        ReactiveLimitEntries rl = ReactiveLimitsPacker.reactiveLimitsPack(network, es, columns, reactiveLimits);
        DistrQPacker.Pack dq = DistrQPacker.distrQPack(network, es, columns, family, rl);
        LinearDistributionTerms distrRho = DistrRhoPacker.distrRhoPack(network, es, columns);
        LinearDistributionTerms distrShuntB = DistrShuntBPacker.distrShuntBPack(network, es, columns);

        RemoteMemberReactiveTerms rmq = RemoteMemberQPacker.remoteMemberQPack(network, es, columns, family, reactiveLimits);
        DistributedSlackParticipants ds = DistributedSlackPacker.distributedSlackPack(network, es, columns, distributedSlack, dsParams);
        // SLOPE-SVC rows (V + slope*sumQ = target, row mode 2): per-row slope, 0 everywhere else
        double[] rowSlope = new double[es.getIndex().getSortedVariablesToFind().size()];
        for (com.powsybl.openloadflow.network.LfBus bus : network.getBuses()) {
            if (bus.hasGeneratorsWithSlope()) {
                int vRow = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcRowIndexing.col(
                        es, bus.getNum(), com.powsybl.openloadflow.ac.equations.AcVariableType.BUS_V);
                if (vRow >= 0) {
                    rowSlope[vRow] = bus.getGeneratorsControllingVoltageWithSlope().get(0).getSlope();
                }
            }
        }
        // Realistic-voltage check (AcloadFlowEngine.isStateUnrealistic): it walks the BUS_V variables
        // of NON-FICTITIOUS buses and compares the state against [minRealisticVoltage,
        // maxRealisticVoltage], but only where the bus's nominal voltage reaches
        // minNominalVoltageRealisticVoltageCheck. The nominal voltage is network data and the
        // thresholds are run parameters, so the row carries the nominal voltage and the install site
        // applies the parameters. 0 marks a row the check never looks at.
        double[] rowNominalV = new double[es.getIndex().getSortedVariablesToFind().size()];
        for (com.powsybl.openloadflow.network.LfBus bus : network.getBuses()) {
            if (bus.isFictitious()) {
                continue;
            }
            int vRow = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcRowIndexing.col(
                    es, bus.getNum(), com.powsybl.openloadflow.ac.equations.AcVariableType.BUS_V);
            if (vRow >= 0) {
                rowNominalV[vRow] = bus.getNominalV();
            }
        }
        return new GpuAcData(
                new ClosedBranchPack(GpuAcArrays.flatD(family.closedBranchParameters(), 8), GpuAcArrays.flatI(family.closedBranchIndices(), 12),
                        family.closedBranchPhaseControlSide().stream().mapToInt(Integer::intValue).toArray()),
                new OpenBranchPack(GpuAcArrays.flatD(family.openBranchParameters(), 7), GpuAcArrays.flatI(family.openBranchIndices(), 6),
                        GpuAcArrays.flatI(family.openBranchAlternateReactiveDestination(), 2)),
                new ElementPack(GpuAcArrays.flatD(family.shuntParameters(), 2), GpuAcArrays.flatI(family.shuntIndices(), 5)),
                new ElementPack(GpuAcArrays.flatD(family.hvdcAcEmulationParameters(), 5), GpuAcArrays.flatI(family.hvdcAcEmulationIndices(), 4)),
                dq.terms(), dq.members(),
                new ZeroImpedanceBranchPack(GpuAcArrays.flatD(family.ziIn(), 1), GpuAcArrays.flatI(family.ziIdx(), 11),
                        family.ziNums().stream().mapToInt(Integer::intValue).toArray(), GpuAcArrays.flatI(family.ziQAlt(), 2)),
                distrRho, distrShuntB,
                rl,
                rmq, new VoltageRowParameters(rowSlope, rowNominalV),
                ds,
                new BaseOpenBranches(family.baseOpenCb().stream().mapToInt(Integer::intValue).toArray(),
                        family.baseOpenZi().stream().mapToInt(Integer::intValue).toArray(),
                        family.baseOpenZiNums().stream().mapToInt(Integer::intValue).toArray()),
                new ElementPack(GpuAcArrays.flatD(family.loadModelParameters(), 3), GpuAcArrays.flatI(family.loadModelIndices(), 2)),
                new ReactiveRelocationCandidates(columns.qrPtr(), columns.qrRows(), columns.qrDepBase(),
                        columns.qrBaseTargetRow(), columns.depGroupOfRow()));
    }

    /** The couplers whose {@code DUMMY_P} / {@code DUMMY_Q} rows host a bus's power equations under the
     *  coupler-tree relocation ({@link GpuAcRowIndexing#ziTreeRelocation}). A batched scenario has to know them:
     *  inside an island every row must become an identity, and under the relocation two of those rows belong to
     *  the coupler, not to the bus. */
    public static java.util.Set<LfBranch> relocatedCouplerBranches(LfNetwork network,
                                                                  EquationSystem<AcVariableType, AcEquationType> es) {
        int[] childBusOfEdge = ExtractionColumns.of(network, es).ziChildBus();
        java.util.Set<LfBranch> relocated = new java.util.HashSet<>();
        for (LfBranch br : network.getBranches()) {
            if (childBusOfEdge[br.getNum()] >= 0) {
                relocated.add(br);
            }
        }
        return relocated;
    }
}
