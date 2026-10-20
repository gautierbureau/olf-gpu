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
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CouplerTreeRelocation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocationCandidates;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocationTables;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.StableRowIndexing;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;

/**
 * The per-extraction column arrays every family and outer-loop packer reads: each bus's
 * BUS_PHI / BUS_V variable rows ({@code phiCol} / {@code vCol}, indexed by bus number), the
 * per-branch transformer-control columns ({@code branchRhoCol} / {@code branchAlphaCol}) and
 * zero-impedance dummy columns ({@code dummyPCol} / {@code dummyQCol}, indexed by branch
 * number), and the stable Q-balance row of each bus ({@code qRow}). Computed once at the top of
 * {@link GpuAcDataExtractor#extract} — a bus appears in several branches, so caching these
 * lookups avoids resolving the same variable ~2x its degree. {@code -1} marks an absent column.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
record ExtractionColumns(int[] phiCol, int[] vCol, int[] branchRhoCol, int[] branchAlphaCol,
                         int[] dummyPCol, int[] dummyQCol, int[] shuntBCol, int[] qRow, int[] alternateQRow,
                         int[] reactiveRelocationGroupStart, int[] reactiveRelocationCandidateRows, int[] reactiveRelocationGroupOfPilot, int[] zeroImpedanceDeadEndSides,
                         /** Per candidate group: the BASE group a DEPENDENT pilot's group belongs to, -1 for a base
                          *  group. A merged remote group's dependent controlled bus becomes the pilot of OLF's re-merged
                          *  group when a contingency cuts it from the main: its reactive balance then rides a member row
                          *  of the same table, which this group's pool reserves (see the dependent-pilot rule in the
                          *  native pick functions). */
                         int[] reactiveRelocationDependentBaseGroup,
                         /** Per candidate group: the row holding the pilot's reactive target in the BASE structure for a
                          *  dependent pilot (its relocated reactive row), -1 for a base group (the base carrier's). */
                         int[] reactiveRelocationBaseTargetRow,
                         /** Per DEVICE row: the dependent-pilot group of the bus whose own V row this is, -1 elsewhere. */
                         int[] reactiveRelocationDependentGroupOfRow,
                         /** Each bus's P-balance ROW: its own BUS_PHI column, or its parent coupler's DUMMY_P under
                          *  the coupler-tree relocation (CouplerTreeRelocation#spanningTreeRelocation). */
                         int[] pRow,
                         /** Per bus, the coupler whose dummy columns host its power equations (-1 = root/none), and
                          *  per branch, the bus whose PHI/V host its zero equations (-1 = not relocated). */
                         int[] spanningTreeParentBranch, int[] spanningTreeChildBus) {

    static ExtractionColumns of(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es) {
        // Cache each bus's BUS_PHI/BUS_V column once: a bus appears in several branches (degree ~3),
        // so resolving these per branch-endpoint repeats the same variable lookups ~2x its degree.
        // Indexed by LfBus.getNum() (contiguous 0..nBuses-1); -1 means absent.
        int nBuses = network.getBuses().size();
        int[] phiCol = new int[nBuses];
        int[] vCol = new int[nBuses];
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            phiCol[bn] = StableRowIndexing.col(es, bn, AcVariableType.BUS_PHI);
            vCol[bn] = StableRowIndexing.col(es, bn, AcVariableType.BUS_V);
        }

        // Per-branch transformer control columns (BRANCH_RHO1 ratio, BRANCH_ALPHA1 phase), -1 when
        // the branch does not derive that control. A controlling transformer's rho column also
        // becomes the controlled bus's Q-balance row (qRow below) — the strong-diagonal pairing.
        int nBranches = network.getBranches().size();
        int[] branchRhoCol = StableRowIndexing.branchVarCols(es, nBranches, AcVariableType.BRANCH_RHO1);
        int[] branchAlphaCol = StableRowIndexing.branchVarCols(es, nBranches, AcVariableType.BRANCH_ALPHA1);
        int[] dummyPCol = StableRowIndexing.branchVarCols(es, nBranches, AcVariableType.DUMMY_P);
        int[] dummyQCol = StableRowIndexing.branchVarCols(es, nBranches, AcVariableType.DUMMY_Q);
        // Per-shunt SHUNT_B variable column (indexed by LfShunt.getNum()), -1 when the shunt derives no voltage
        // control. A controlling shunt's SHUNT_B column also becomes the controlled bus's Q-balance row (qRow),
        // the exact analogue of the transformer rho pairing.
        int nShunts = network.getShunts().size();
        int[] shuntBCol = StableRowIndexing.branchVarCols(es, nShunts, AcVariableType.SHUNT_B);
        int[] qRow = ReactiveRelocation.computeQRow(network, es, vCol, branchRhoCol, shuntBCol);
        // Dynamic Q-relocation: the OTHER candidate Q row of each transformer/shunt voltage-controlled
        // bus, so the fill can follow a per-scenario control toggle on the shared structure (§11).
        int[] alternateQRow = ReactiveRelocation.computeAlternateQRow(network, es, vCol, branchRhoCol, shuntBCol, qRow);
        // The coupler-tree relocation, computed HERE rather than after the candidate tables: a child bus's power
        // equations move onto its parent coupler's dummy columns, so every row of a coupler chain keeps a
        // structurally non-zero diagonal — and the tables below must name the row that ends up hosting each
        // member's reactive balance, which for a relocated member is that dummy column and no longer its own V row.
        int[][] spanningTreeEdges = CouplerTreeRelocation.spanningTreeRelocation(network, es, phiCol, vCol, qRow, dummyPCol, dummyQCol);
        // The candidate tables of the reactive relocation (ReactiveRelocationTables#build): mutates qRow / alternateQRow.
        ReactiveRelocationTables reactiveRelocationTables = ReactiveRelocationTables.build(network, es, vCol, branchRhoCol, dummyQCol,
                spanningTreeEdges[0], qRow, alternateQRow);
        // The coupler-tree relocation applied to the rows (CouplerTreeRelocation#applyToRows): dead-end sides, then
        // pRow / qRow / alternateQRow onto the parent coupler; mutates qRow / alternateQRow / spanningTreeEdges[1].
        CouplerTreeRelocation.AppliedRows relocatedRows = CouplerTreeRelocation.applyToRows(network, es, phiCol, vCol, dummyPCol, dummyQCol,
                qRow, alternateQRow, spanningTreeEdges);
        int[] spanningTreeParentBranch = spanningTreeEdges[0];
        int[] spanningTreeChildBus = spanningTreeEdges[1];
        ReactiveRelocationCandidates candidates = reactiveRelocationTables.candidates();
        return new ExtractionColumns(phiCol, vCol, branchRhoCol, branchAlphaCol, dummyPCol, dummyQCol, shuntBCol, qRow, alternateQRow,
                candidates.groupStart(), candidates.candidateRows(), reactiveRelocationTables.groupOfPilot(), relocatedRows.zeroImpedanceDeadEndSides(),
                candidates.dependentBaseGroup(), candidates.baseTargetRow(), candidates.dependentGroupOfRow(),
                relocatedRows.pRow(), spanningTreeParentBranch, spanningTreeChildBus);
    }
}
