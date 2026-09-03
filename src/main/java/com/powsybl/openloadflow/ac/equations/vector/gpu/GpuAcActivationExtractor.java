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
import com.powsybl.openloadflow.equations.SingleEquation;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;

/**
 * Extracts what ACTIVATION decides — the per-row power-equation/identity mode, the
 * OLF-column→stable-row target mapping ({@link GpuAcActivation}) and the per-row variable
 * kind used by the MAX_VOLTAGE_CHANGE device step scaling. Rebuilt only when an activation
 * event fired; the extracted {@link GpuAcData} structure stays valid across PV→PQ switching
 * (see {@link GpuAcRowIndexing} for the stable-row policy this reuses).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcActivationExtractor {

    private GpuAcActivationExtractor() {
    }

    /**
     * Extract what ACTIVATION decides — the per-row power-equation/identity mode and the
     * OLF-column→stable-row target mapping. Rebuilt only when an activation event fired
     * (the structure stays valid across PV→PQ switching). Walks every active equation, so
     * an unsupported equation type ACTIVATED at runtime (e.g. zero-impedance) fails loudly
     * here even though the full structural validation does not re-run.
     */
    public static GpuAcActivation extractActivation(LfNetwork network,
                                                    EquationSystem<AcVariableType, AcEquationType> es, int n) {
        int[] mode = new int[n];
        java.util.Arrays.fill(mode, 1);
        int[] targetMap = new int[n];
        java.util.Arrays.fill(targetMap, -1);
        int nBranches = network.getBranches().size();
        int[] branchRhoCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.BRANCH_RHO1);
        int[] shuntBCol = GpuAcRowIndexing.branchVarCols(es, network.getShunts().size(), AcVariableType.SHUNT_B);
        int[] vCol = new int[network.getBuses().size()];
        for (LfBus bus : network.getBuses()) {
            vCol[bus.getNum()] = GpuAcRowIndexing.col(es, bus.getNum(), AcVariableType.BUS_V);
        }
        int[] qRow = GpuAcRowIndexing.computeQRow(network, es, vCol, branchRhoCol, shuntBCol);
        for (LfBus bus : network.getBuses()) {
            int p = GpuAcRowIndexing.row(es, bus.getNum(), AcEquationType.BUS_TARGET_P);
            if (p >= 0) {
                targetMap[GpuAcRowIndexing.col(es, bus.getNum(), AcVariableType.BUS_PHI)] = p;
            }
            int q = GpuAcRowIndexing.row(es, bus.getNum(), AcEquationType.BUS_TARGET_Q);
            if (q >= 0) {
                targetMap[qRow[bus.getNum()]] = q;          // Q row = V row, or the controlling rho row
            }
        }
        for (SingleEquation<AcVariableType, AcEquationType> e : es.getIndex().getSortedSingleEquationsToSolve()) {
            // POWER-equation rows (mode stays 1) whose stable row is a non-bus-P/Q variable — assembled by
            // the GPU DISTR_Q / zero-impedance passes; just map the target onto that variable's row.
            AcVariableType powerRowVar = switch (e.getType()) {
                case DISTR_Q -> AcVariableType.BUS_V;        // controller's BUS_V row
                case DISTR_RHO -> AcVariableType.BRANCH_RHO1; // controller's BRANCH_RHO1 row
                case DISTR_SHUNT_B -> AcVariableType.SHUNT_B; // controller's SHUNT_B row
                case ZERO_V -> AcVariableType.DUMMY_P;       // zero-imp v1−rho·v2 sits on the DUMMY_P row
                case ZERO_PHI -> AcVariableType.DUMMY_Q;     // zero-imp phi1−phi2 sits on the DUMMY_Q row
                default -> null;
            };
            if (powerRowVar != null) {
                targetMap[GpuAcRowIndexing.col(es, e.getElementNum(), powerRowVar)] = e.getColumn();
                continue;
            }
            // IDENTITY rows (mode 0): the device row-mode mask writes diag 1 and F = x − target.
            AcVariableType vt = switch (e.getType()) {
                case BUS_TARGET_V -> AcVariableType.BUS_V;
                case BUS_TARGET_PHI -> AcVariableType.BUS_PHI;
                case BRANCH_TARGET_RHO1 -> AcVariableType.BRANCH_RHO1;   // ratio held constant: r1−target identity
                // Phase shift held constant: a1−target identity. This is the ONLY phase-control row the
                // GPU packs, and it is the one INCREMENTAL mode keeps active for the whole run (that loop
                // never calls setPhaseControlEnabled, so a1 stays a variable PINNED to its tap value and
                // the taps move between solves). CONTINUOUS mode with control ON activates BRANCH_TARGET_P
                // on the CONTROLLED branch instead — a branch-owned power row the GPU has no family for,
                // and it is NOT in this switch, so it hits the default throw below and the solver falls
                // back to CPU for that solve. Safe by construction, never silently dropped.
                case BRANCH_TARGET_ALPHA1 -> AcVariableType.BRANCH_ALPHA1;
                case SHUNT_TARGET_B -> AcVariableType.SHUNT_B;           // susceptance held constant: b−target identity
                case DUMMY_TARGET_P -> AcVariableType.DUMMY_P;           // non-tree zero-imp: dummyP held = 0
                case DUMMY_TARGET_Q -> AcVariableType.DUMMY_Q;           // non-tree zero-imp: dummyQ held = 0
                case BUS_TARGET_P, BUS_TARGET_Q -> null;     // power rows, mapped above
                default -> throw new PowsyblException("GPU AC solver: unsupported equation type "
                        + e.getType() + " activated");
            };
            if (vt != null) {
                int r = GpuAcRowIndexing.col(es, e.getElementNum(), vt);
                // SLOPE-SVC V row (V + slope*sumQ = target): the natural reactive fill STAYS in the row,
                // scaled by the slope and completed with the +v diagonal (row mode 2) — not an identity.
                boolean slopeRow = e.getType() == AcEquationType.BUS_TARGET_V
                        && network.getBus(e.getElementNum()).hasGeneratorsWithSlope();
                mode[r] = slopeRow ? 2 : 0;
                targetMap[r] = e.getColumn();
            }
        }
        return new GpuAcActivation(mode, targetMap);
    }

    /**
     * Per-row variable kind for the MAX_VOLTAGE_CHANGE device step scaling: {@code 0} = BUS_PHI (clamp
     * the angle step to maxDphi), {@code 1} = BUS_V (clamp the magnitude step to maxDv), {@code 2} = any
     * other variable (BRANCH_RHO1, …) — scaled by the global step factor but never itself a clamp source,
     * mirroring {@code MaxVoltageChangeStateVectorScaling} (which only inspects BUS_V / BUS_PHI). Indexed by
     * the variable row, which is the column index of the Newton step.
     */
    public static int[] extractVarType(EquationSystem<AcVariableType, AcEquationType> es, int n) {
        int[] varType = new int[n];
        java.util.Arrays.fill(varType, 2);
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            varType[v.getRow()] = switch (v.getType()) {
                case BUS_PHI -> 0;
                case BUS_V -> 1;
                default -> 2;
            };
        }
        return varType;
    }
}
