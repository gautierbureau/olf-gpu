/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.powsybl.openloadflow.dc.equations.AbstractClosedBranchDcFlowEquationTerm;
import com.powsybl.openloadflow.dc.equations.ClosedBranchSide1DcFlowEquationTerm;
import com.powsybl.openloadflow.dc.equations.ClosedBranchSide2DcFlowEquationTerm;
import com.powsybl.openloadflow.dc.equations.DcEquationSystemCreationParameters;
import com.powsybl.openloadflow.dc.equations.DcEquationType;
import com.powsybl.openloadflow.dc.equations.DcVariableType;
import com.powsybl.openloadflow.dc.equations.HvdcAcEmulationSide1DCFlowEquationTerm;
import com.powsybl.openloadflow.dc.equations.HvdcAcEmulationSide2DCFlowEquationTerm;
import com.powsybl.openloadflow.dc.equations.vector.ClosedBranchSide1DcFlowEquationTermArrayEvaluator;
import com.powsybl.openloadflow.dc.equations.vector.ClosedBranchSide2DcFlowEquationTermArrayEvaluator;
import com.powsybl.openloadflow.equations.Derivative;
import com.powsybl.openloadflow.equations.EquationArray;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationTerm;
import com.powsybl.openloadflow.equations.EquationTermArray;
import com.powsybl.openloadflow.equations.SingleEquation;
import com.powsybl.openloadflow.equations.SingleEquationTerm;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.equations.VariableEquationTerm;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfSynchronousNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Everything the device needs to ASSEMBLE OLF's DC system itself — the matrix values, the target
 * vector and the slack distribution — captured once from the equation system's STRUCTURE, with no
 * value in it that the host would have to recompute per solve.
 *
 * <p>The matrix. OLF's {@code JacobianMatrix.initDer} walks the equations in column order and, for
 * each, the derivative callbacks of {@code SingleEquation.der} / {@code EquationArray.der}: every
 * callback is one matrix element whose value is the SEQUENTIAL sum of the active terms' derivatives.
 * The DC derivatives are constants of the branch model: {@code ±power[b]} for a closed-branch flow
 * term, {@code ±1} for a variable term (reference angle, dummy P, zero-impedance coupling),
 * {@code ±k} for an HVDC AC-emulation term. So an element is a list of contributions
 * {@code (kind, element, coef)} in OLF's callback order, and the device recomputes it as
 * {@code v = 0; v += coef * power[elem]} (or {@code v += coef}) — the same operations in the same
 * order as the CPU: bit-identical values, verified by the tests against {@code SparseMatrix.getValues()}.
 *
 * <p>The pattern is the device's CSR: row = equation column (OLF's {@code Equation.getColumn()}),
 * entries = variable rows, in callback order — OLF's column-compressed transposed storage read as
 * row-compressed, exactly what {@link GpuDcSolver#create} uploads.
 *
 * <p>The target vector. {@code DcTargetVector.init}: per equation a base value by type (the bus
 * {@code targetP} — generation minus load — with the multi-slack share, the phase shifter's
 * {@code a1}, the zero-impedance coupling's {@code A2 - a1}, or 0) minus the sequential sum of the
 * active terms' {@code rhs()} (the fixed phase shift moved to the right-hand side,
 * {@code -power * (A2 - a1)} / {@code +power * (A2 - a1)}, and the HVDC {@code ±p0}). Captured
 * the same way: a base kind per column and an ordered rhs contribution list.
 *
 * <p>The buses and generators. The bus targets come from the generators' {@code targetP} — the
 * quantity the slack distribution moves — summed on the device in {@code bus.getGenerators()}
 * order like {@code AbstractLfBus.getGenerationTargetP}, and the loads' target P, uploaded. The
 * generator tables are {@code GenerationActivePowerDistributionStep}'s inputs in OLF's iteration
 * order (buses, then each bus's generators): initial and current target P, limits, droop,
 * participation factor and flag.
 *
 * <p>Not supported (the capture throws, the caller keeps the host path): an equation term class
 * outside the DC set above, or a bus P equation whose OLF callbacks the capture cannot reproduce.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuDcAssembly {

    /** A constant contribution: {@code coef}. */
    public static final int CONTRIB_CONST = 0;
    /** A branch-model contribution: {@code coef * power[elem]}. */
    public static final int CONTRIB_BRANCH = 1;

    /** Target base kinds (DcTargetVector.init's switch). */
    public static final int TARGET_ZERO = 0;
    public static final int TARGET_BUS_P = 1;
    public static final int TARGET_ALPHA1 = 2;
    public static final int TARGET_ZERO_PHI = 3;

    /** Rhs contribution kinds: a constant, or {@code coef * power[elem] * (A2 - a1fixed[elem])}. */
    public static final int RHS_CONST = 0;
    public static final int RHS_BRANCH = 1;

    // --- the pattern and the value contributions
    final int n;
    final int[] rowPtr;                 // n + 1
    final int[] colIdx;                 // nnz: variable rows
    final int[] contribPtr;             // nnz + 1
    final int[] contribKind;
    final int[] contribElem;
    final double[] contribCoef;

    // --- the target
    final int[] targetKind;             // n
    final int[] targetElem;             // n: bus or branch
    final int[] rhsPtr;                 // n + 1
    final int[] rhsKind;
    final int[] rhsElem;
    final double[] rhsCoef;

    // --- branches (LfNetwork.getBranches() order): the flow model and the fixed shift
    final int[] ph1Row;
    final int[] ph2Row;
    final int[] a1Row;
    final double[] a1;                  // the pi model's a1 (fixed shift when a1Row < 0, the alpha target otherwise)
    final double[] power;

    // --- buses (LfNetwork.getBuses() order)
    final int[] busPColumn;             // the active P equation's column, or -1
    final int[] busPhRow;               // the angle variable's row, or -1
    final double[] busLoadTargetP;
    final int[] busFlags;               // bit 0 disabled, bit 1 participating, bit 2 fictitious, bit 3 slack
    final int[] busSc;                  // synchronous component index
    final int[] busGenStart;            // buses + 1: the bus's generators in the generator tables

    // --- synchronous components: their buses (in LfSynchronousNetwork.getBuses() order) and slack count
    final int[] scBusPtr;
    final int[] scBusList;
    final int[] scSlackCount;

    // --- generators (buses, then bus.getGenerators() order)
    final List<LfGenerator> generators;
    final double[] genInitialTargetP;
    final double[] genTargetP;
    final double[] genMinTargetP;
    final double[] genMaxTargetP;
    final double[] genMinP;
    final double[] genMaxP;
    final double[] genDroop;
    final double[] genParticipationFactor;
    final int[] genParticipating;
    final int referenceGenerator;       // index, or -1

    public static final int BUS_DISABLED = 1;
    public static final int BUS_PARTICIPATING = 2;
    public static final int BUS_FICTITIOUS = 4;
    public static final int BUS_SLACK = 8;

    private GpuDcAssembly(Builder b) {
        n = b.n;
        rowPtr = b.rowPtr;
        colIdx = b.colIdx.toArray();
        contribPtr = b.contribPtr.toArray();
        contribKind = b.contribKind.toArray();
        contribElem = b.contribElem.toArray();
        contribCoef = b.contribCoef.toArray();
        targetKind = b.targetKind;
        targetElem = b.targetElem;
        rhsPtr = b.rhsPtr;
        rhsKind = b.rhsKind.toArray();
        rhsElem = b.rhsElem.toArray();
        rhsCoef = b.rhsCoef.toArray();
        ph1Row = b.ph1Row;
        ph2Row = b.ph2Row;
        a1Row = b.a1Row;
        a1 = b.a1;
        power = b.power;
        busPColumn = b.busPColumn;
        busPhRow = b.busPhRow;
        busLoadTargetP = b.busLoadTargetP;
        busFlags = b.busFlags;
        busSc = b.busSc;
        busGenStart = b.busGenStart;
        scBusPtr = b.scBusPtr;
        scBusList = b.scBusList;
        scSlackCount = b.scSlackCount;
        generators = List.copyOf(b.generators);
        genInitialTargetP = b.genInitialTargetP;
        genTargetP = b.genTargetP;
        genMinTargetP = b.genMinTargetP;
        genMaxTargetP = b.genMaxTargetP;
        genMinP = b.genMinP;
        genMaxP = b.genMaxP;
        genDroop = b.genDroop;
        genParticipationFactor = b.genParticipationFactor;
        genParticipating = b.genParticipating;
        referenceGenerator = b.referenceGenerator;
    }

    public int size() {
        return n;
    }

    public int nonZeroCount() {
        return colIdx.length;
    }

    public int busCount() {
        return busPColumn.length;
    }

    public int branchCount() {
        return power.length;
    }

    public int generatorCount() {
        return generators.size();
    }

    public List<LfGenerator> generators() {
        return generators;
    }

    /** The bus P column ({@code -1} when the bus has no active P equation — the first slack bus). */
    public int busPColumn(int bus) {
        return busPColumn[bus];
    }

    /**
     * Same non-zero PATTERN and same element identity: the device context can be kept and refilled,
     * the contribution lists re-uploaded ({@link GpuDcSolver#setContributions}). A term switched on
     * or off — an HVDC leaving AC emulation for saturation — keeps the pattern (OLF emits a matrix
     * element for every variable of every term, active or not) and only changes the contributions.
     */
    public boolean samePattern(GpuDcAssembly other) {
        return n == other.n && Arrays.equals(rowPtr, other.rowPtr) && Arrays.equals(colIdx, other.colIdx)
                && Arrays.equals(targetKind, other.targetKind) && Arrays.equals(targetElem, other.targetElem)
                && Arrays.equals(ph1Row, other.ph1Row) && Arrays.equals(ph2Row, other.ph2Row) && Arrays.equals(a1Row, other.a1Row)
                && Arrays.equals(busPColumn, other.busPColumn) && Arrays.equals(busPhRow, other.busPhRow)
                && Arrays.equals(busGenStart, other.busGenStart) && generators.equals(other.generators);
    }

    /** Same equation structure (pattern, contributions, target kinds and rhs lists): values may be refilled in place. */
    public boolean sameStructure(GpuDcAssembly other) {
        return n == other.n && Arrays.equals(rowPtr, other.rowPtr) && Arrays.equals(colIdx, other.colIdx)
                && Arrays.equals(contribPtr, other.contribPtr) && Arrays.equals(contribKind, other.contribKind)
                && Arrays.equals(contribElem, other.contribElem) && Arrays.equals(contribCoef, other.contribCoef)
                && Arrays.equals(targetKind, other.targetKind) && Arrays.equals(targetElem, other.targetElem)
                && Arrays.equals(rhsPtr, other.rhsPtr) && Arrays.equals(rhsKind, other.rhsKind)
                && Arrays.equals(rhsElem, other.rhsElem) && Arrays.equals(rhsCoef, other.rhsCoef)
                && Arrays.equals(ph1Row, other.ph1Row) && Arrays.equals(ph2Row, other.ph2Row) && Arrays.equals(a1Row, other.a1Row)
                && Arrays.equals(busPColumn, other.busPColumn) && Arrays.equals(busPhRow, other.busPhRow)
                && Arrays.equals(busGenStart, other.busGenStart) && generators.equals(other.generators);
    }

    /**
     * The host-side check of the value model: OLF's own matrix values recomputed from the
     * contribution lists, in OLF's order. Equal bit for bit to {@code SparseMatrix.getValues()} when
     * the capture is faithful — the assembly tests' oracle, and a cheap self-check for a caller.
     */
    public double[] hostValues() {
        double[] values = new double[colIdx.length];
        for (int i = 0; i < values.length; i++) {
            double v = 0;
            for (int c = contribPtr[i]; c < contribPtr[i + 1]; c++) {
                v += contribKind[c] == CONTRIB_BRANCH ? contribCoef[c] * power[contribElem[c]] : contribCoef[c];
            }
            values[i] = v;
        }
        return values;
    }

    /**
     * The device's target vector formula on the host, from the captured tables and the given
     * generator targets: {@code DcTargetVector.init} step for step (base by kind, the multi-slack
     * share, minus the sequential rhs sum). The assembly tests' oracle against
     * {@code TargetVector.getArray()}.
     */
    public double[] hostTarget(double[] generatorTargetP) {
        int nb = busCount();
        double[] genP = new double[nb];
        for (int b = 0; b < nb; b++) {
            double s = 0.0;
            for (int g = busGenStart[b]; g < busGenStart[b + 1]; g++) {
                s += generatorTargetP[g];
            }
            genP[b] = s;
        }
        double[] scMismatch = new double[scBusPtr.length - 1];
        for (int sc = 0; sc < scMismatch.length; sc++) {
            double sum = 0;
            for (int i = scBusPtr[sc]; i < scBusPtr[sc + 1]; i++) {
                int b = scBusList[i];
                if ((busFlags[b] & BUS_DISABLED) == 0) {
                    sum += genP[b] - busLoadTargetP[b];
                }
            }
            scMismatch[sc] = -sum;
        }
        double[] t = new double[n];
        for (int c = 0; c < n; c++) {
            double base;
            switch (targetKind[c]) {
                case TARGET_BUS_P -> {
                    int b = targetElem[c];
                    base = genP[b] - busLoadTargetP[b];
                    if ((busFlags[b] & BUS_SLACK) != 0) {
                        base += scMismatch[busSc[b]] / scSlackCount[busSc[b]];
                    }
                }
                case TARGET_ALPHA1 -> base = a1[targetElem[c]];
                case TARGET_ZERO_PHI -> base = A2 - a1[targetElem[c]];
                default -> base = 0;
            }
            double rhs = 0;
            for (int r = rhsPtr[c]; r < rhsPtr[c + 1]; r++) {
                if (rhsKind[r] == RHS_BRANCH) {
                    int b = rhsElem[r];
                    double a1Fixed = a1Row[b] >= 0 ? 0.0 : a1[b];
                    rhs += rhsCoef[r] * power[b] * (A2 - a1Fixed);
                } else {
                    rhs += rhsCoef[r];
                }
            }
            t[c] = base;
            t[c] -= rhs;
        }
        return t;
    }

    private static final double A2 = com.powsybl.openloadflow.network.PiModel.A2;

    /** Capture the structure and the current values of the network's DC system. */
    public static GpuDcAssembly capture(LfNetwork network, EquationSystem<DcVariableType, DcEquationType> equationSystem,
                                        DcEquationSystemCreationParameters creationParameters) {
        return new Builder(network, equationSystem, creationParameters).build();
    }

    private static final class Builder {
        private final LfNetwork network;
        private final EquationSystem<DcVariableType, DcEquationType> equationSystem;
        private final DcEquationSystemCreationParameters creationParameters;

        int n;
        int[] rowPtr;
        final TIntArrayList colIdx = new TIntArrayList();
        final TIntArrayList contribPtr = new TIntArrayList();
        final TIntArrayList contribKind = new TIntArrayList();
        final TIntArrayList contribElem = new TIntArrayList();
        final TDoubleArrayList contribCoef = new TDoubleArrayList();
        int[] targetKind;
        int[] targetElem;
        int[] rhsPtr;
        final TIntArrayList rhsKind = new TIntArrayList();
        final TIntArrayList rhsElem = new TIntArrayList();
        final TDoubleArrayList rhsCoef = new TDoubleArrayList();
        int[] ph1Row;
        int[] ph2Row;
        int[] a1Row;
        double[] a1;
        double[] power;
        int[] busPColumn;
        int[] busPhRow;
        double[] busLoadTargetP;
        int[] busFlags;
        int[] busSc;
        int[] busGenStart;
        int[] scBusPtr;
        int[] scBusList;
        int[] scSlackCount;
        final List<LfGenerator> generators = new ArrayList<>();
        double[] genInitialTargetP;
        double[] genTargetP;
        double[] genMinTargetP;
        double[] genMaxTargetP;
        double[] genMinP;
        double[] genMaxP;
        double[] genDroop;
        double[] genParticipationFactor;
        int[] genParticipating;
        int referenceGenerator = -1;

        // per-bus rows of BUS_PHI / per-branch rows of BRANCH_ALPHA1, from the index
        private int[] phRowOfBus;
        private int[] a1RowOfBranch;

        Builder(LfNetwork network, EquationSystem<DcVariableType, DcEquationType> equationSystem, DcEquationSystemCreationParameters creationParameters) {
            this.network = Objects.requireNonNull(network);
            this.equationSystem = Objects.requireNonNull(equationSystem);
            this.creationParameters = Objects.requireNonNull(creationParameters);
        }

        GpuDcAssembly build() {
            n = equationSystem.getIndex().getColumnCount();
            if (equationSystem.getIndex().getRowCount() != n) {
                throw new IllegalStateException("Expected to have same number of equations (" + n + ") and variables ("
                        + equationSystem.getIndex().getRowCount() + ")");
            }
            variables();
            branches();
            buses();
            equations();
            return new GpuDcAssembly(this);
        }

        private void variables() {
            phRowOfBus = new int[network.getBuses().size()];
            a1RowOfBranch = new int[network.getBranches().size()];
            Arrays.fill(phRowOfBus, -1);
            Arrays.fill(a1RowOfBranch, -1);
            for (Variable<DcVariableType> v : equationSystem.getIndex().getSortedVariablesToFind()) {
                switch (v.getType()) {
                    case BUS_PHI -> phRowOfBus[v.getElementNum()] = v.getRow();
                    case BRANCH_ALPHA1 -> a1RowOfBranch[v.getElementNum()] = v.getRow();
                    default -> { }
                }
            }
        }

        private void branches() {
            List<LfBranch> branches = network.getBranches();
            int nb = branches.size();
            ph1Row = new int[nb];
            ph2Row = new int[nb];
            a1Row = new int[nb];
            a1 = new double[nb];
            power = new double[nb];
            for (int b = 0; b < nb; b++) {
                LfBranch branch = branches.get(b);
                if (branch.getNum() != b) {
                    throw new IllegalStateException("branch numbering is not positional");
                }
                LfBus bus1 = branch.getBus1();
                LfBus bus2 = branch.getBus2();
                boolean flowing = bus1 != null && bus2 != null && !branch.isZeroImpedance(LoadFlowModel.DC);
                ph1Row[b] = flowing ? phRowOfBus[bus1.getNum()] : -1;
                ph2Row[b] = flowing ? phRowOfBus[bus2.getNum()] : -1;
                a1Row[b] = flowing ? a1RowOfBranch[b] : -1;
                a1[b] = branch.getPiModel().getA1();
                // AbstractClosedBranchDcFlowEquationTerm.getPower / DcBranchVector.updateBranchModel
                power[b] = flowing ? AbstractClosedBranchDcFlowEquationTerm.computePower(creationParameters.isUseTransformerRatio(),
                        creationParameters.getDcApproximationType(), branch.getPiModel()) : 0;
            }
        }

        private void buses() {
            List<LfBus> buses = network.getBuses();
            int nb = buses.size();
            busPColumn = new int[nb];
            busPhRow = new int[nb];
            busLoadTargetP = new double[nb];
            busFlags = new int[nb];
            busSc = new int[nb];
            busGenStart = new int[nb + 1];
            List<LfSynchronousNetwork> scs = network.getSynchronousNetworks();
            LfGenerator reference = scs.getFirst().getReferenceGenerator();
            for (int b = 0; b < nb; b++) {
                LfBus bus = buses.get(b);
                if (bus.getNum() != b) {
                    throw new IllegalStateException("bus numbering is not positional");
                }
                busPColumn[b] = equationSystem.getEquation(b, DcEquationType.BUS_TARGET_P)
                        .filter(com.powsybl.openloadflow.equations.Equation::isActive)
                        .map(com.powsybl.openloadflow.equations.Equation::getColumn)
                        .orElse(-1);
                busPhRow[b] = phRowOfBus[b];
                busLoadTargetP[b] = bus.getLoadTargetP();
                busFlags[b] = (bus.isDisabled() ? BUS_DISABLED : 0) | (bus.isParticipating() ? BUS_PARTICIPATING : 0)
                        | (bus.isFictitious() ? BUS_FICTITIOUS : 0) | (bus.isSlack() ? BUS_SLACK : 0);
                busSc[b] = -1;
                for (int sc = 0; sc < scs.size(); sc++) {
                    if (scs.get(sc).getNumSC() == bus.getNumSC()) {
                        busSc[b] = sc;
                    }
                }
                if (busSc[b] < 0) {
                    throw new IllegalStateException("bus " + bus.getId() + " belongs to no synchronous component");
                }
                busGenStart[b] = generators.size();
                for (LfGenerator g : bus.getGenerators()) {
                    if (g == reference) {
                        referenceGenerator = generators.size();
                    }
                    generators.add(g);
                }
            }
            busGenStart[nb] = generators.size();
            if (reference != null && referenceGenerator < 0) {
                throw new IllegalStateException("reference generator " + reference.getId() + " is on no bus of the network");
            }
            int m = generators.size();
            genInitialTargetP = new double[m];
            genTargetP = new double[m];
            genMinTargetP = new double[m];
            genMaxTargetP = new double[m];
            genMinP = new double[m];
            genMaxP = new double[m];
            genDroop = new double[m];
            genParticipationFactor = new double[m];
            genParticipating = new int[m];
            for (int i = 0; i < m; i++) {
                LfGenerator g = generators.get(i);
                genInitialTargetP[i] = g.getInitialTargetP();
                genTargetP[i] = g.getTargetP();
                genMinTargetP[i] = g.getMinTargetP();
                genMaxTargetP[i] = g.getMaxTargetP();
                genMinP[i] = g.getMinP();
                genMaxP[i] = g.getMaxP();
                genDroop[i] = g.getDroop();
                genParticipationFactor[i] = g.getParticipationFactor();
                genParticipating[i] = g.isParticipating() ? 1 : 0;
            }
            scBusPtr = new int[scs.size() + 1];
            TIntArrayList list = new TIntArrayList();
            scSlackCount = new int[scs.size()];
            for (int sc = 0; sc < scs.size(); sc++) {
                scBusPtr[sc] = list.size();
                for (LfBus bus : scs.get(sc).getBuses()) {
                    list.add(bus.getNum());
                }
                scSlackCount[sc] = scs.get(sc).getSlackBuses().size();
            }
            scBusPtr[scs.size()] = list.size();
            scBusList = list.toArray();
        }

        /** JacobianMatrix.initDer's walk: single equations and equation arrays interleaved by column. */
        private void equations() {
            rowPtr = new int[n + 1];
            targetKind = new int[n];
            targetElem = new int[n];
            rhsPtr = new int[n + 1];
            contribPtr.add(0);
            var itSingle = equationSystem.getIndex().getSortedSingleEquationsToSolve().iterator();
            var itArray = equationSystem.getIndex().getSortedEquationArraysToSolve().iterator();
            SingleEquation<DcVariableType, DcEquationType> eq = itSingle.hasNext() ? itSingle.next() : null;
            EquationArray<DcVariableType, DcEquationType> eqArray = itArray.hasNext() ? itArray.next() : null;
            int index = 0;
            while (eq != null || eqArray != null) {
                while (eq != null && index == eq.getColumn()) {
                    singleEquation(eq, index);
                    index++;
                    eq = itSingle.hasNext() ? itSingle.next() : null;
                }
                while (eqArray != null && index == eqArray.getFirstColumn()) {
                    equationArray(eqArray);
                    index += eqArray.getLength();
                    eqArray = itArray.hasNext() ? itArray.next() : null;
                }
                if (eq != null && eqArray == null && index != eq.getColumn() || eqArray != null && eq == null && index != eqArray.getFirstColumn()) {
                    throw new IllegalStateException("equation columns are not contiguous at " + index);
                }
            }
            if (index != n) {
                throw new IllegalStateException("walked " + index + " columns of " + n);
            }
        }

        private void beginColumn(int column) {
            rowPtr[column] = colIdx.size();
            rhsPtr[column] = rhsKind.size();
        }

        private void endColumn(int column) {
            rowPtr[column + 1] = colIdx.size();
            rhsPtr[column + 1] = rhsKind.size();
        }

        private void endElement() {
            contribPtr.add(contribKind.size());
        }

        private void singleEquation(SingleEquation<DcVariableType, DcEquationType> eq, int column) {
            beginColumn(column);
            // SingleEquation.der: variables in TreeMap order, the active terms' derivatives summed in list order
            for (Map.Entry<Variable<DcVariableType>, List<EquationTerm<DcVariableType, DcEquationType>>> e : eq.getTermsByVariable().entrySet()) {
                Variable<DcVariableType> variable = e.getKey();
                if (variable.getRow() == -1) {
                    continue;
                }
                colIdx.add(variable.getRow());
                for (EquationTerm<DcVariableType, DcEquationType> term : e.getValue()) {
                    if (term.isActive()) {
                        derivative(term, variable);
                    }
                }
                endElement();
            }
            // DcTargetVector.init(SingleEquation)
            switch (eq.getType()) {
                case BUS_TARGET_P -> {
                    targetKind[column] = TARGET_BUS_P;
                    targetElem[column] = eq.getElementNum();
                }
                case BUS_TARGET_PHI, DUMMY_TARGET_P -> targetKind[column] = TARGET_ZERO;
                case BRANCH_TARGET_ALPHA1 -> {
                    targetKind[column] = TARGET_ALPHA1;
                    targetElem[column] = eq.getElementNum();
                }
                case ZERO_PHI -> {
                    targetKind[column] = TARGET_ZERO_PHI;
                    targetElem[column] = eq.getElementNum();
                }
                default -> throw new IllegalStateException("Unknown state variable type: " + eq.getType());
            }
            // SingleEquation.rhs: the active terms with an rhs, in list order
            for (SingleEquationTerm<DcVariableType, DcEquationType> term : eq.getTerms()) {
                if (term.isActive() && term.hasRhs()) {
                    rhs(term);
                }
            }
            endColumn(column);
        }

        private record DerivativeEntry(int termArrayNum, int termNum, Derivative<DcVariableType> derivative) {
        }

        private void equationArray(EquationArray<DcVariableType, DcEquationType> array) {
            if (array.getType() != DcEquationType.BUS_TARGET_P) {
                throw new IllegalStateException("Unexpected DC equation array type: " + array.getType());
            }
            List<EquationTermArray<DcVariableType, DcEquationType>> termArrays = array.getTermArrays();
            for (int elementNum = 0; elementNum < array.getElementCount(); elementNum++) {
                if (!array.isElementActive(elementNum)) {
                    continue;
                }
                int column = array.getElementNumToColumn(elementNum);
                beginColumn(column);
                // EquationArray.addEquationDerivativeVectorSortedTerms: every term derivative of the element,
                // stable-sorted by variable
                List<DerivativeEntry> entries = new ArrayList<>();
                for (int termArrayNum = 0; termArrayNum < termArrays.size(); termArrayNum++) {
                    EquationTermArray<DcVariableType, DcEquationType> termArray = termArrays.get(termArrayNum);
                    int[] starts = termArray.getTermNumsConcatenatedStartIndices();
                    TIntArrayList termNums = termArray.getTermNumsConcatenated();
                    for (int i = starts[elementNum]; i < starts[elementNum + 1]; i++) {
                        int termNum = termNums.getQuick(i);
                        for (Derivative<DcVariableType> d : termArray.getTermDerivatives(termNum)) {
                            entries.add(new DerivativeEntry(termArrayNum, termNum, d));
                        }
                    }
                }
                entries.sort(Comparator.comparing(o -> o.derivative().getVariable()));
                // the additional single terms (dummy P of zero-impedance branches), by variable
                TreeMap<Variable<DcVariableType>, List<SingleEquationTerm<DcVariableType, DcEquationType>>> singleByVariable = new TreeMap<>();
                for (SingleEquationTerm<DcVariableType, DcEquationType> term : array.getSingleEquationTerms(elementNum)) {
                    for (Variable<DcVariableType> v : term.getVariables()) {
                        singleByVariable.computeIfAbsent(v, k -> new ArrayList<>()).add(term);
                    }
                }
                // EquationArray.der: one element per run of equal rows; the single terms of that row close it
                TIntArrayList computedRows = new TIntArrayList();
                int i = 0;
                while (i < entries.size()) {
                    int row = entries.get(i).derivative().getVariable().getRow();
                    if (row == -1) {
                        throw new IllegalStateException("array term derivative on a variable without a row");
                    }
                    colIdx.add(row);
                    while (i < entries.size() && entries.get(i).derivative().getVariable().getRow() == row) {
                        DerivativeEntry en = entries.get(i);
                        EquationTermArray<DcVariableType, DcEquationType> termArray = termArrays.get(en.termArrayNum());
                        if (termArray.isTermActive(en.termNum())) {
                            arrayDerivative(termArray, termArray.getTermElementNum(en.termNum()), en.derivative().getLocalIndex());
                        }
                        i++;
                    }
                    for (var e : singleByVariable.entrySet()) {
                        if (e.getKey().getRow() == row) {
                            for (SingleEquationTerm<DcVariableType, DcEquationType> term : e.getValue()) {
                                if (term.isActive()) {
                                    derivative(term, e.getKey());
                                }
                            }
                        }
                    }
                    computedRows.add(row);
                    endElement();
                }
                for (var e : singleByVariable.entrySet()) {
                    int row = e.getKey().getRow();
                    if (row != -1 && !computedRows.contains(row)) {
                        colIdx.add(row);
                        for (SingleEquationTerm<DcVariableType, DcEquationType> term : e.getValue()) {
                            if (term.isActive()) {
                                derivative(term, e.getKey());
                            }
                        }
                        endElement();
                    }
                }
                // DcTargetVector.init(EquationArray): bus target, then minus EquationArray.getRhs
                targetKind[column] = TARGET_BUS_P;
                targetElem[column] = elementNum;
                for (EquationTermArray<DcVariableType, DcEquationType> termArray : termArrays) {
                    int[] starts = termArray.getTermNumsConcatenatedStartIndices();
                    TIntArrayList termNums = termArray.getTermNumsConcatenated();
                    for (int k = starts[elementNum]; k < starts[elementNum + 1]; k++) {
                        int termNum = termNums.getQuick(k);
                        if (termArray.isTermActive(termNum)) {
                            arrayRhs(termArray, termArray.getTermElementNum(termNum));
                        }
                    }
                }
                for (SingleEquationTerm<DcVariableType, DcEquationType> term : array.getSingleEquationTerms(elementNum)) {
                    if (term.isActive() && term.hasRhs()) {
                        rhs(term);
                    }
                }
                endColumn(column);
            }
        }

        /** One active term's derivative with respect to {@code variable}, as a contribution. */
        private void derivative(EquationTerm<DcVariableType, DcEquationType> term, Variable<DcVariableType> variable) {
            if (term instanceof ClosedBranchSide1DcFlowEquationTerm || term instanceof ClosedBranchSide2DcFlowEquationTerm) {
                int branch = term.getElementNum();
                branchDerivative(branch, term instanceof ClosedBranchSide1DcFlowEquationTerm, variable, term.der(variable));
            } else if (term instanceof EquationTermArray.EquationTermArrayElementImpl<DcVariableType, DcEquationType> element) {
                var evaluator = element.getEvaluator();
                if (evaluator instanceof ClosedBranchSide1DcFlowEquationTermArrayEvaluator
                        || evaluator instanceof ClosedBranchSide2DcFlowEquationTermArrayEvaluator) {
                    branchDerivative(term.getElementNum(), evaluator instanceof ClosedBranchSide1DcFlowEquationTermArrayEvaluator, variable, term.der(variable));
                } else {
                    throw new UnsupportedOperationException("equation term array evaluator not supported on the device: " + evaluator.getClass().getName());
                }
            } else if (isConstantDerivative(term)) {
                constant(term.der(variable));
            } else {
                throw new UnsupportedOperationException("equation term not supported on the device: " + term.getClass().getName());
            }
        }

        private void arrayDerivative(EquationTermArray<DcVariableType, DcEquationType> termArray, int branch, int localIndex) {
            var evaluator = termArray.getEvaluator();
            boolean side1;
            if (evaluator instanceof ClosedBranchSide1DcFlowEquationTermArrayEvaluator) {
                side1 = true;
            } else if (evaluator instanceof ClosedBranchSide2DcFlowEquationTermArrayEvaluator) {
                side1 = false;
            } else {
                throw new UnsupportedOperationException("equation term array evaluator not supported on the device: " + evaluator.getClass().getName());
            }
            // AbstractClosedBranchDcFlowEquationTermArrayEvaluator.getDerivatives: [dph1, dph2, da1] = side 1: [p, -p, p]; side 2: [-p, p, -p]
            double coef = localIndex == 1 ? -1.0 : 1.0;
            if (!side1) {
                coef = -coef;
            }
            double expected = termArray.getEvaluator().evalDer()[localIndex][branch];
            if (Double.compare(expected, coef * power[branch]) != 0) {
                throw new IllegalStateException("branch " + branch + ": derivative " + expected + " is not " + coef + " x power " + power[branch]);
            }
            contribKind.add(CONTRIB_BRANCH);
            contribElem.add(branch);
            contribCoef.add(coef);
        }

        private void branchDerivative(int branch, boolean side1, Variable<DcVariableType> variable, double olfValue) {
            LfBranch lfBranch = network.getBranch(branch);
            double coef;
            if (variable.getType() == DcVariableType.BUS_PHI && variable.getElementNum() == lfBranch.getBus1().getNum()) {
                coef = 1.0;                                  // d p1 / d ph1 = power
            } else if (variable.getType() == DcVariableType.BUS_PHI && variable.getElementNum() == lfBranch.getBus2().getNum()) {
                coef = -1.0;                                 // d p1 / d ph2 = -power
            } else if (variable.getType() == DcVariableType.BRANCH_ALPHA1 && variable.getElementNum() == branch) {
                coef = 1.0;                                  // d p1 / d a1 = power
            } else {
                throw new IllegalStateException("branch " + lfBranch.getId() + " term derived on a foreign variable " + variable);
            }
            if (!side1) {
                coef = -coef;
            }
            if (Double.compare(olfValue, coef * power[branch]) != 0) {
                throw new IllegalStateException("branch " + lfBranch.getId() + ": derivative " + olfValue + " is not " + coef + " x power " + power[branch]);
            }
            contribKind.add(CONTRIB_BRANCH);
            contribElem.add(branch);
            contribCoef.add(coef);
        }

        private void constant(double value) {
            contribKind.add(CONTRIB_CONST);
            contribElem.add(-1);
            contribCoef.add(value);
        }

        private static boolean isConstantDerivative(EquationTerm<?, ?> term) {
            if (term instanceof VariableEquationTerm<?, ?>
                    || term instanceof HvdcAcEmulationSide1DCFlowEquationTerm
                    || term instanceof HvdcAcEmulationSide2DCFlowEquationTerm) {
                return true;
            }
            if (term instanceof SingleEquationTerm.MultiplyByScalarEquationTerm<?, ?> multiply) {
                return isConstantDerivative(multiply.getChildren().getFirst());
            }
            return false;
        }

        /** One active single term's rhs() as a contribution (SingleEquation.rhs / EquationArray.getRhs's single terms). */
        private void rhs(SingleEquationTerm<DcVariableType, DcEquationType> term) {
            if (term instanceof ClosedBranchSide1DcFlowEquationTerm || term instanceof ClosedBranchSide2DcFlowEquationTerm) {
                branchRhs(term.getElementNum(), term instanceof ClosedBranchSide1DcFlowEquationTerm, term.rhs());
            } else if (term instanceof HvdcAcEmulationSide1DCFlowEquationTerm || term instanceof HvdcAcEmulationSide2DCFlowEquationTerm) {
                rhsKind.add(RHS_CONST);
                rhsElem.add(-1);
                rhsCoef.add(term.rhs());                     // ±p0 of the AC emulation
            } else {
                throw new UnsupportedOperationException("equation term with rhs not supported on the device: " + term.getClass().getName());
            }
        }

        private void arrayRhs(EquationTermArray<DcVariableType, DcEquationType> termArray, int branch) {
            var evaluator = termArray.getEvaluator();
            boolean side1;
            if (evaluator instanceof ClosedBranchSide1DcFlowEquationTermArrayEvaluator) {
                side1 = true;
            } else if (evaluator instanceof ClosedBranchSide2DcFlowEquationTermArrayEvaluator) {
                side1 = false;
            } else {
                throw new UnsupportedOperationException("equation term array evaluator not supported on the device: " + evaluator.getClass().getName());
            }
            branchRhs(branch, side1, evaluator.rhs(branch));
        }

        /** rhs1 = -p * (A2 - a1fixed), rhs2 = p * (A2 - a1fixed); a1fixed = 0 when a1 is a variable. */
        private void branchRhs(int branch, boolean side1, double olfValue) {
            double coef = side1 ? -1.0 : 1.0;
            double a1Fixed = a1Row[branch] >= 0 ? 0.0 : a1[branch];
            if (Double.compare(olfValue, coef * power[branch] * (A2 - a1Fixed)) != 0) {
                throw new IllegalStateException("branch " + network.getBranch(branch).getId() + ": rhs " + olfValue + " is not "
                        + coef + " x power x (A2 - a1) = " + coef * power[branch] * (A2 - a1Fixed));
            }
            rhsKind.add(RHS_BRANCH);
            rhsElem.add(branch);
            rhsCoef.add(coef);
        }
    }
}
