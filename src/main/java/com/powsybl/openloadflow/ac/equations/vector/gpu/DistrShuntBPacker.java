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
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.ShuntVoltageControl;
import com.powsybl.openloadflow.network.VoltageControl;

import java.util.ArrayList;
import java.util.List;

/**
 * The DISTR_SHUNT_B rows of shared shunt voltage control: each controller shunt's susceptance share, as
 * linear (row, column, coefficient) contributions (OLF ShuntDistributionEquationTerm).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class DistrShuntBPacker {

    private DistrShuntBPacker() {
    }

    static LinearDistributionTerms distrShuntBPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                   ExtractionColumns columns) {
        List<int[]> db = new ArrayList<>();      // {dbRow, dbCol} per DISTR_SHUNT_B contribution
        List<Double> dbC = new ArrayList<>();    // coefficient per contribution
        buildDistrShuntBContributions(network, es, columns.shuntBCol(), db, dbC);
        int[] dbRow = new int[db.size()];
        int[] dbCol = new int[db.size()];
        double[] dbCoef = new double[db.size()];
        for (int i = 0; i < db.size(); i++) {
            dbRow[i] = db.get(i)[0];
            dbCol[i] = db.get(i)[1];
            dbCoef[i] = dbC.get(i);
        }
        return new LinearDistributionTerms(dbRow, dbCol, dbCoef);
    }

    /**
     * DISTR_SHUNT_B linear-distribution rows — the shunt-susceptance analogue of
     * {@link DistrRhoPacker#buildDistrRhoContributions}. For each shunt-voltage-controlled pilot bus, every controller shunt
     * with an ACTIVE DISTR_SHUNT_B equation gets a row {@code b_k = (1/count)·Σ_j b_j} rewritten as
     * {@code 0 = (1/count − 1)·b_k + Σ_{j≠k} b_j/count} over the {@code count} enabled controllers. The free
     * controller (no DISTR_SHUNT_B) carries the pilot's Q balance via qRow.
     */
    private static void buildDistrShuntBContributions(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                      int[] shuntBCol, List<int[]> out, List<Double> coefs) {
        for (LfBus pilot : network.getBuses()) {
            ShuntVoltageControl svc = pilot.getShuntVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> pilot.isShuntVoltageControlled())
                    .orElse(null);
            if (svc == null) {
                continue;
            }
            List<LfShunt> controllers = svc.getMergedControllerElements();
            long count = controllers.stream().filter(c -> !c.isDisabled()).count();
            for (LfShunt gk : controllers) {
                if (GpuAcRowIndexing.row(es, gk.getNum(), AcEquationType.DISTR_SHUNT_B) < 0) {
                    continue;                                // the free controller carries the pilot's Q, no DISTR_SHUNT_B
                }
                int dbRow = shuntBCol[gk.getNum()];
                for (LfShunt gj : controllers) {
                    if (gj.isDisabled()) {
                        continue;
                    }
                    double coef = 1.0 / count - (gj == gk ? 1.0 : 0.0);
                    out.add(new int[] {dbRow, shuntBCol[gj.getNum()]});
                    coefs.add(coef);
                }
            }
        }
    }
}
