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
import com.powsybl.openloadflow.ac.equations.ClosedBranchSide1ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.ClosedBranchSide1ReactiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.ClosedBranchSide2ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.ClosedBranchSide2ReactiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.HvdcAcEmulationSide1ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.HvdcAcEmulationSide2ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.OpenBranchSide1ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.OpenBranchSide1ReactiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.OpenBranchSide2ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.OpenBranchSide2ReactiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.ShuntCompensatorActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.ShuntCompensatorReactiveFlowEquationTerm;
import com.powsybl.openloadflow.equations.EquationArray;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationTermArray;
import com.powsybl.openloadflow.equations.SingleEquation;
import com.powsybl.openloadflow.equations.SingleEquationTerm;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.equations.VariableEquationTerm;

import java.util.Set;

/**
 * Structural validation of the OLF equation system against what the GPU AC path covers.
 * Runs FIRST in {@link GpuAcDataExtractor#extract} and throws {@link PowsyblException} on
 * anything unsupported — unsupported variable types (e.g. transformer controls beyond the
 * covered shapes), unsupported equation types (e.g. shunt voltage control), unsupported
 * active equation terms (e.g. ZIP load models), and unsupported equation-array (vectorized)
 * evaluators — so an unsupported network fails loudly instead of converging to a wrong
 * state. Only reads {@code es.getIndex()}; a disconnection-allowed branch (a security-
 * analysis contingency candidate) is NOT rejected (see the note at the end of validate).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuAcValidation {

    private static final Set<Class<?>> SUPPORTED_PQ_TERMS = Set.of(
            ClosedBranchSide1ActiveFlowEquationTerm.class, ClosedBranchSide1ReactiveFlowEquationTerm.class,
            ClosedBranchSide2ActiveFlowEquationTerm.class, ClosedBranchSide2ReactiveFlowEquationTerm.class,
            OpenBranchSide1ActiveFlowEquationTerm.class, OpenBranchSide1ReactiveFlowEquationTerm.class,
            OpenBranchSide2ActiveFlowEquationTerm.class, OpenBranchSide2ReactiveFlowEquationTerm.class,
            ShuntCompensatorActiveFlowEquationTerm.class, ShuntCompensatorReactiveFlowEquationTerm.class,
            HvdcAcEmulationSide1ActiveFlowEquationTerm.class, HvdcAcEmulationSide2ActiveFlowEquationTerm.class,
            // voltage-dependent load model (ZIP/exponential): assembled by the GPU vdl family
            com.powsybl.openloadflow.ac.equations.LoadModelActiveFlowEquationTerm.class,
            com.powsybl.openloadflow.ac.equations.LoadModelReactiveFlowEquationTerm.class);

    // the closed-branch equation-term arrays of the VECTORIZED equation system
    // (AcVectorizedEquationSystemCreator) — the GPU covers exactly the same physics
    private static final Set<String> SUPPORTED_ARRAY_EVALUATORS = Set.of(
            "ac_p_array_closed_1", "ac_p_array_closed_2", "ac_q_array_closed_1", "ac_q_array_closed_2");

    private GpuAcValidation() {
    }

    static void validate(EquationSystem<AcVariableType, AcEquationType> es) {
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            // BRANCH_RHO1 (continuous transformer voltage control) is supported for a single
            // transformer locally controlling one of its own terminal buses; computeQRow throws
            // on the unsupported shapes (remote / multi-transformer DISTR_RHO).
            // BRANCH_ALPHA1 (phase control) is packed as a COLUMN only: the closed-branch kernel already
            // reads a1 from the state vector when the branch derives it (cbIdx[8]) and already emits
            // dp1da1/dq1da1/dp2da1/dq2da1. Which ROW pairs with that column is decided in
            // extractActivation: BRANCH_TARGET_ALPHA1 (identity, INCREMENTAL mode) is packed,
            // BRANCH_TARGET_P (CONTINUOUS mode with control enabled) throws there.
            AcVariableType vtype = v.getType();
            if (vtype != AcVariableType.BUS_V && vtype != AcVariableType.BUS_PHI
                    && vtype != AcVariableType.BRANCH_RHO1
                    && vtype != AcVariableType.BRANCH_ALPHA1
                    && vtype != AcVariableType.SHUNT_B
                    && vtype != AcVariableType.DUMMY_P && vtype != AcVariableType.DUMMY_Q) {
                throw new PowsyblException("GPU AC solver: unsupported variable type " + vtype
                        + " (only plain AC power flow + transformer/shunt voltage control + zero-impedance "
                        + "branches are supported)");
            }
        }
        // vectorized (equation-array) systems: the arrays carry the closed-branch terms
        // (same physics as the GPU closed-branch kernel); per-element scalar terms
        // (open-branch, shunt, HVDC) attach to the array elements
        for (EquationArray<AcVariableType, AcEquationType> arr : es.getIndex().getSortedEquationArraysToSolve()) {
            if (arr.getType() != AcEquationType.BUS_TARGET_P && arr.getType() != AcEquationType.BUS_TARGET_Q) {
                throw new PowsyblException("GPU AC solver: unsupported equation-array type " + arr.getType());
            }
            for (EquationTermArray<AcVariableType, AcEquationType> ta : arr.getTermArrays()) {
                String name = ((EquationTermArray.EquationTermArrayElementImpl<AcVariableType, AcEquationType>) ta.getElement(0))
                        .getEvaluator().getName();
                if (!SUPPORTED_ARRAY_EVALUATORS.contains(name)) {
                    throw new PowsyblException("GPU AC solver: unsupported equation-term array '" + name
                            + "' in " + arr.getType());
                }
            }
            for (int elementNum = 0; elementNum < arr.getElementCount(); elementNum++) {
                for (SingleEquationTerm<AcVariableType, AcEquationType> t : arr.getSingleEquationTerms(elementNum)) {
                    // a VariableEquationTerm is a zero-impedance DUMMY_P/Q coupling term (GPU zi family)
                    if (t.isActive() && !SUPPORTED_PQ_TERMS.contains(t.getClass())
                            && !isDummyCouplingTerm(t)) {
                        throw new PowsyblException("GPU AC solver: unsupported equation term "
                                + t.getClass().getSimpleName() + " in " + arr.getType());
                    }
                }
            }
        }
        for (SingleEquation<AcVariableType, AcEquationType> e : es.getIndex().getSortedSingleEquationsToSolve()) {
            switch (e.getType()) {
                case BUS_TARGET_P, BUS_TARGET_Q -> {
                    for (SingleEquationTerm<AcVariableType, AcEquationType> t : e.getTerms()) {
                        // a VariableEquationTerm here is a zero-impedance branch's DUMMY_P/DUMMY_Q coupling
                        // term, assembled by the GPU zero-impedance family — accept it.
                        if (t.isActive() && !SUPPORTED_PQ_TERMS.contains(t.getClass())
                                && !isDummyCouplingTerm(t)) {
                            throw new PowsyblException("GPU AC solver: unsupported equation term "
                                    + t.getClass().getSimpleName() + " in " + e.getType());
                        }
                    }
                }
                case DISTR_Q -> {
                    // reactive-distribution row: OLF sums per-controller reactive terms wrapped in
                    // MultiplyByScalarEquationTerm. The GPU re-assembles it from network topology, so the
                    // real coverage guard is buildDistrQContributions (it throws on shunt / open-branch /
                    // remote shapes it cannot wire). Accept the equation here.
                }
                case ZERO_V, ZERO_PHI, DUMMY_TARGET_P, DUMMY_TARGET_Q -> {
                    // zero-impedance branch coupling (v1−rho·v2 / phi1−phi2) and the held-dummy identities —
                    // assembled by the GPU zero-impedance family from network topology. Accept.
                }
                case DISTR_RHO -> {
                    // transformer ratio distribution ((1/count)·Σ r1 − r1_k) — a linear coupling of the
                    // controller BRANCH_RHO1 variables, re-assembled by the GPU DISTR_RHO family. Accept.
                }
                case DISTR_SHUNT_B -> {
                    // shunt susceptance distribution ((1/count)·Σ b − b_k) — a linear coupling of the
                    // controller SHUNT_B variables, re-assembled by the GPU DISTR_SHUNT_B family. Accept.
                }
                case BUS_TARGET_V, BUS_TARGET_PHI, BRANCH_TARGET_RHO1, BRANCH_TARGET_ALPHA1, SHUNT_TARGET_B -> {
                    // identity rows on a single variable (v−target, phi−target, r1−target when a
                    // transformer ratio is held constant, a1−target when a phase shift is, b−target when a
                    // shunt susceptance is held constant) — applied device-side by the row-mode mask.
                    // EXCEPTION: a SLOPE-SVC BUS_TARGET_V additionally carries slope-scaled copies of the
                    // bus's reactive flow terms (V + slope*sumQ = target) — re-assembled device-side by
                    // scaling the natural reactive fill (row mode 2), so those terms are accepted.
                    for (SingleEquationTerm<AcVariableType, AcEquationType> t : e.getTerms()) {
                        if (t.isActive() && !(t instanceof VariableEquationTerm)
                                && !(e.getType() == AcEquationType.BUS_TARGET_V
                                        && t.getClass().getSimpleName().contains("Multiply"))) {
                            throw new PowsyblException("GPU AC solver: unsupported equation term "
                                    + t.getClass().getSimpleName() + " in " + e.getType());
                        }
                    }
                }
                case BRANCH_TARGET_P -> {
                    // CONTINUOUS phase control: the controlled branch's P-flow as its own equation row
                    // (re-emitted by the cb family at the a1 row, row mode 3). Accept the flow term.
                }
                default -> throw new PowsyblException("GPU AC solver: unsupported equation type " + e.getType());
            }
        }
        // NOTE: a branch flagged disconnection-allowed (a security-analysis contingency candidate) is NOT
        // rejected. Disconnection is a discrete event BETWEEN solves (a contingency), never a switch WITHIN
        // a single Newton/outer-loop sequence — so for any one fixed configuration the extracted pattern is
        // correct. A currently-closed candidate is modelled as a normal closed branch; when a contingency
        // outages it, its terms deactivate and it drops out of the index packs, so the structure changes and
        // GpuNewtonRaphson rebuilds its context (sameStructureAs is false) while the batch masks it
        // per-scenario. The throw that used to live here was therefore over-conservative.
    }

    /** A P/Q-balance term over only DUMMY_P/DUMMY_Q variables — a zero-impedance coupling term the GPU
     *  zero-impedance family assembles (its wrapper class may be a Variable or a MultiplyByScalar term). */
    private static boolean isDummyCouplingTerm(SingleEquationTerm<AcVariableType, AcEquationType> t) {
        return !t.getVariables().isEmpty() && t.getVariables().stream()
                .allMatch(v -> v.getType() == AcVariableType.DUMMY_P || v.getType() == AcVariableType.DUMMY_Q);
    }
}
