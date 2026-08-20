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
                         int[] dummyPCol, int[] dummyQCol, int[] shuntBCol, int[] qRow, int[] qRowAlt) {

    static ExtractionColumns of(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es) {
        // Cache each bus's BUS_PHI/BUS_V column once: a bus appears in several branches (degree ~3),
        // so resolving these per branch-endpoint repeats the same variable lookups ~2x its degree.
        // Indexed by LfBus.getNum() (contiguous 0..nBuses-1); -1 means absent.
        int nBuses = network.getBuses().size();
        int[] phiCol = new int[nBuses];
        int[] vCol = new int[nBuses];
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            phiCol[bn] = GpuAcRowIndexing.col(es, bn, AcVariableType.BUS_PHI);
            vCol[bn] = GpuAcRowIndexing.col(es, bn, AcVariableType.BUS_V);
        }

        // Per-branch transformer control columns (BRANCH_RHO1 ratio, BRANCH_ALPHA1 phase), -1 when
        // the branch does not derive that control. A controlling transformer's rho column also
        // becomes the controlled bus's Q-balance row (qRow below) — the strong-diagonal pairing.
        int nBranches = network.getBranches().size();
        int[] branchRhoCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.BRANCH_RHO1);
        int[] branchAlphaCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.BRANCH_ALPHA1);
        int[] dummyPCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.DUMMY_P);
        int[] dummyQCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.DUMMY_Q);
        // Per-shunt SHUNT_B variable column (indexed by LfShunt.getNum()), -1 when the shunt derives no voltage
        // control. A controlling shunt's SHUNT_B column also becomes the controlled bus's Q-balance row (qRow),
        // the exact analogue of the transformer rho pairing.
        int nShunts = network.getShunts().size();
        int[] shuntBCol = GpuAcRowIndexing.branchVarCols(es, nShunts, AcVariableType.SHUNT_B);
        int[] qRow = GpuAcRowIndexing.computeQRow(network, es, vCol, branchRhoCol, shuntBCol);
        // Dynamic Q-relocation: the OTHER candidate Q row of each transformer/shunt voltage-controlled
        // bus, so the fill can follow a per-scenario control toggle on the shared structure (§11).
        int[] qRowAlt = GpuAcRowIndexing.computeQRowAlt(network, es, vCol, branchRhoCol, shuntBCol, qRow);
        return new ExtractionColumns(phiCol, vCol, branchRhoCol, branchAlphaCol, dummyPCol, dummyQCol, shuntBCol, qRow, qRowAlt);
    }
}
