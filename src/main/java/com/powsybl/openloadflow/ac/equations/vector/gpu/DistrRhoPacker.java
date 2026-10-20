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
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.TransformerVoltageControl;
import com.powsybl.openloadflow.network.VoltageControl;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * The DISTR_RHO rows of shared transformer voltage control: each controller transformer's ratio share, as
 * linear (row, column, coefficient) contributions (OLF TransformerDistributionEquationTerm).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DistrRhoPacker {

    private DistrRhoPacker() {
    }

    static LinearDistributionTerms distrRhoPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                               ExtractionColumns columns) {
        List<int[]> terms = new ArrayList<>();   // {distrRhoRow, distrRhoColumn} per DISTR_RHO contribution
        List<Double> coefficients = new ArrayList<>();    // coefficient per contribution
        buildDistrRhoContributions(network, es, columns.branchRhoCol(), terms, coefficients);
        int[] distrRhoRow = new int[terms.size()];
        int[] distrRhoColumn = new int[terms.size()];
        double[] distrRhoCoefficient = new double[terms.size()];
        for (int i = 0; i < terms.size(); i++) {
            distrRhoRow[i] = terms.get(i)[0];
            distrRhoColumn[i] = terms.get(i)[1];
            distrRhoCoefficient[i] = coefficients.get(i);
        }
        return new LinearDistributionTerms(distrRhoRow, distrRhoColumn, distrRhoCoefficient);
    }

    /**
     * Build the LINEAR DISTR_RHO contributions (ratio distribution when several transformers control one
     * bus). A controller branch {@code Gk} with an active DISTR_RHO equation gets, on its BRANCH_RHO1 row,
     * {@code DISTR_RHO_k = (1/count)·Σ_j r1_j − r1_k}, i.e. each controller {@code Gj}'s ratio column
     * contributes with coefficient {@code 1/count − (j==k ? 1 : 0)}, where {@code count} is the number of
     * enabled controllers. The free controller (no DISTR_RHO) carries the pilot's Q balance via qRow.
     */
    private static void buildDistrRhoContributions(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                   int[] branchRhoCol, List<int[]> out, List<Double> coefs) {
        for (LfBus pilot : network.getBuses()) {
            // ACTIVATION-INDEPENDENT (plan-group-member-dynamics A1): pack the ALL-ENABLED
            // configuration of every group whose DISTR_RHO equations EXIST — never key on
            // isTransformerVoltageControlled()/getColumn(), which are false/-1 at a control-OFF
            // base (AFTER_GENERATOR) and silently emptied the pack (the e82290f audit). The FREE
            // member is the FIRST enabled one — OLF's updateRemoteVoltageControlEquations pick and
            // computeAlternateQRow's, so its row carries the relocated pilot Q, never DISTR_RHO. Rows
            // inactive in a scenario are identity-wiped by the row-mode pass (superset principle).
            TransformerVoltageControl voltageControl = pilot.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (voltageControl == null) {
                continue;
            }
            List<LfBranch> controllers = voltageControl.getMergedControllerElements().stream()
                    .filter(c -> !c.isDisabled())
                    .filter(c -> es.getEquation(c.getNum(), AcEquationType.DISTR_RHO).isPresent())
                    .toList();
            long count = controllers.size();
            for (int k = 1; k < controllers.size(); k++) {   // k = 0 is the FREE member: no DISTR_RHO row
                LfBranch gk = controllers.get(k);
                int distrRhoRow = branchRhoCol[gk.getNum()];
                for (LfBranch gj : controllers) {
                    double coef = 1.0 / count - (gj == gk ? 1.0 : 0.0);
                    out.add(new int[] {distrRhoRow, branchRhoCol[gj.getNum()]});
                    coefs.add(coef);
                }
            }
        }
    }

    /**
     * The transformers that participate in a ratio-distribution (DISTR_RHO) group. Outaging one of these
     * REFORMS the {@code 1/count} coefficients (OLF redistributes the ratio across the remaining controllers),
     * which the fixed batch pattern cannot represent — so the batched N-1 path rejects such a contingency to
     * the CPU. Returns the controllers of every transformer-voltage-controlled pilot whose group has more than
     * one enabled member (a single local controller is item 4, not DISTR_RHO, and is mask-safe).
     */
    public static Set<LfBranch> distrRhoControllerBranches(LfNetwork network,
                                                           EquationSystem<AcVariableType, AcEquationType> es) {
        Set<LfBranch> controllers = new java.util.HashSet<>();
        for (LfBus pilot : network.getBuses()) {
            // NOTE: keyed on the DISTR_RHO equations EXISTING, NOT on isTransformerVoltageControlled()
            // — at an AFTER_GENERATOR base the control is DISABLED and that predicate is false, which
            // silently emptied this set and let a member outage batch with the coupling missing
            // (measured 3.7e-3 kV wrong, 2026-09-06).
            TransformerVoltageControl voltageControl = pilot.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (voltageControl == null) {
                continue;
            }
            List<LfBranch> group = voltageControl.getMergedControllerElements();
            // EXISTENCE-keyed (row()/getColumn() is -1 for INACTIVE equations — the same trap):
            // a shared group's equations exist whether or not the control is on at base.
            boolean distrRho = group.stream().filter(c -> !c.isDisabled())
                    .filter(c -> es.getEquation(c.getNum(), AcEquationType.DISTR_RHO).isPresent())
                    .count() > 1;
            if (distrRho) {
                controllers.addAll(group);
            }
        }
        return controllers;
    }
}
