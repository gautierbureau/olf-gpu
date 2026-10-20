/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;

/**
 * docs/gpu-vs-olf.md section 2.1 (stable rows): the STABLE row-indexing policy shared by the whole GPU extraction,
 * the variable/equation row lookups. Every equation row is identified by its diagonal-paired VARIABLE row (a bus's P
 * row is its BUS_PHI variable row, its Q row its BUS_V row — or the controlling transformer's BRANCH_RHO1 row when the
 * bus is transformer-voltage-controlled, see {@link ReactiveRelocation}), never by OLF's equation column, so the
 * extracted structure survives activation switching. {@code GpuAcDataExtractor}, {@code GpuAcActivationExtractor}
 * and the family/outer-loop packers all lean on this.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class StableRowIndexing {

    private StableRowIndexing() {
    }

    public static int col(EquationSystem<AcVariableType, AcEquationType> es, int bus, AcVariableType t) {
        return es.getVariableSet().getVariable(bus, t).getRow();
    }

    public static int row(EquationSystem<AcVariableType, AcEquationType> es, int bus, AcEquationType t) {
        return es.getEquation(bus, t).map(e -> e.getColumn()).orElse(-1);
    }

    /** Variable rows of every {@code type} variable, indexed by its element (branch) number; -1 when absent. */
    public static int[] branchVarCols(EquationSystem<AcVariableType, AcEquationType> es, int nBranches,
                               AcVariableType type) {
        int[] cols = new int[nBranches];
        java.util.Arrays.fill(cols, -1);
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == type) {
                cols[v.getElementNum()] = v.getRow();
            }
        }
        return cols;
    }
}
