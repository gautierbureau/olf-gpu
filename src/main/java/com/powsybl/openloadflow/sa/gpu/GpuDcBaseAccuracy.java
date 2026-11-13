/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.math.matrix.SparseMatrix;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.equations.AbstractClosedBranchDcFlowEquationTerm;
import com.powsybl.openloadflow.dc.equations.DcEquationSystemCreationParameters;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LoadFlowModel;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * How accurate the device base state is, measured against OLF's OWN DC system (its Jacobian and target vector on the
 * same context): the device target against OLF's, the device state and the state OLF's LU gives (what the stock
 * security analysis solves) each against a reference refined with exact residuals, and each one's exact residual.
 * A diagnostic of the device LU: printed, and returned for a comparison to size its tolerance on (the probe). It
 * changes nothing.
 *
 * <p>OLF stores the DC Jacobian transposed (row = variable, column = equation) and solves with
 * {@code solveTransposed}, so the residual of equation j is column j of that matrix against x, minus b_j: summed
 * exactly here (BigDecimal), then rounded once.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuDcBaseAccuracy {

    /** |1/x| above which a branch makes the DC matrix stiff (OLF models a retained switch as x = 1e-8). */
    public static final double STIFF_ADMITTANCE = 1e6;

    /**
     * What the measurement found: the device state's and OLF's LU state's distances to the refined state (the
     * largest over the state vector, rad for the angles), whether the network has stiff branches, and the DC power
     * (per unit) of every flowing branch by id, the factor between an angle error and a flow error.
     */
    public record Result(double deviceError, double luError, int stiffBranches, Map<String, Double> branchPower) {

        /** How far apart two states that each solve this system can be: the sum of the two measured errors. */
        public double stateUncertainty() {
            return deviceError + luError;
        }
    }

    private GpuDcBaseAccuracy() {
    }

    static Result report(DcLoadFlowContext context, GpuDcSolver solver, double[] deviceState) {
        SparseMatrix m = (SparseMatrix) context.getJacobianMatrix().getMatrix();
        double[] b = context.getTargetVector().getArray().clone();
        double[] deviceTarget = solver.readTarget();
        int n = b.length;
        double targetDiff = 0;
        for (int j = 0; j < n; j++) {
            targetDiff = Math.max(targetDiff, Math.abs(deviceTarget[j] - b[j]));
        }
        Map<String, Integer> stiffByKind = new TreeMap<>();
        java.util.List<String> examples = new java.util.ArrayList<>();
        Map<String, Double> branchPower = new HashMap<>();
        DcEquationSystemCreationParameters creation = context.getParameters().getEquationSystemCreationParameters();
        int stiff = 0;
        for (LfBranch branch : context.getNetwork().getBranches()) {
            if (branch.isDisabled() || branch.isZeroImpedance(LoadFlowModel.DC)) {
                continue;
            }
            double power = AbstractClosedBranchDcFlowEquationTerm.computePower(creation.isUseTransformerRatio(),
                    creation.getDcApproximationType(), branch.getPiModel());
            branchPower.put(branch.getId(), Math.abs(power));
            if (Math.abs(power) > STIFF_ADMITTANCE) {
                stiff++;
                stiffByKind.merge(branch.getBranchType() + " x=" + branch.getPiModel().getX(), 1, Integer::sum);
                if (examples.size() < 5) {
                    examples.add(branch.getId());
                }
            }
        }
        System.err.printf("DC_BASE_ACCURACY branches with |power| > %.0e: %s, e.g. %s%n", STIFF_ADMITTANCE, stiffByKind, examples);
        double[] lu = b.clone();
        context.getJacobianMatrix().solveTransposed(lu);
        double[] reference = lu.clone();
        double[] step = new double[n];
        for (int it = 0; it < 4; it++) {
            residual(m, reference, b, step);
            for (int j = 0; j < n; j++) {
                step[j] = -step[j];
            }
            context.getJacobianMatrix().solveTransposed(step);
            for (int j = 0; j < n; j++) {
                reference[j] += step[j];
            }
        }
        double[] r = new double[n];
        double deviceError = 0;
        double luError = 0;
        double maxValue = 0;
        double minValue = Double.MAX_VALUE;
        for (double v : m.getValues()) {
            if (v != 0) {
                maxValue = Math.max(maxValue, Math.abs(v));
                minValue = Math.min(minValue, Math.abs(v));
            }
        }
        System.err.printf("DC_BASE_ACCURACY n=%d | max |target device - OLF| %.3e pu | |J| entries from %.3e to %.3e%n", n, targetDiff,
                minValue, maxValue);
        for (String arm : new String[] {"device", "OLF LU", "refined"}) {
            double[] x = switch (arm) {
                case "device" -> deviceState;
                case "OLF LU" -> lu;
                default -> reference;
            };
            residual(m, x, b, r);
            double maxResidual = 0;
            double maxError = 0;
            double maxX = 0;
            for (int j = 0; j < n; j++) {
                maxResidual = Math.max(maxResidual, Math.abs(r[j]));
                maxError = Math.max(maxError, Math.abs(x[j] - reference[j]));
                maxX = Math.max(maxX, Math.abs(x[j]));
            }
            if (x == deviceState) {
                deviceError = maxError;
            } else if (x == lu) {
                luError = maxError;
            }
            System.err.printf("DC_BASE_ACCURACY %-8s | max |residual| %.3e pu (%.3e MW) | max |x - refined| %.3e (max |x| %.3e)%n",
                    arm, maxResidual, maxResidual * 100, maxError, maxX);
        }
        double maxDiff = 0;
        for (int j = 0; j < n; j++) {
            maxDiff = Math.max(maxDiff, Math.abs(deviceState[j] - lu[j]));
        }
        System.err.printf("DC_BASE_ACCURACY max |x device - x OLF LU| %.3e%n", maxDiff);
        return new Result(deviceError, luError, stiff, branchPower);
    }

    /** out[j] = (column j of m) . x - b[j], summed exactly and rounded once. */
    private static void residual(SparseMatrix m, double[] x, double[] b, double[] out) {
        int[] start = m.getColumnStart();
        int[] rows = m.getRowIndices();
        double[] values = m.getValues();
        for (int j = 0; j < b.length; j++) {
            BigDecimal sum = new BigDecimal(b[j]).negate();
            for (int k = start[j]; k < start[j + 1]; k++) {
                sum = sum.add(new BigDecimal(values[k]).multiply(new BigDecimal(x[rows[k]])));
            }
            out[j] = sum.doubleValue();
        }
    }
}
