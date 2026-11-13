/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.equations.AbstractClosedBranchDcFlowEquationTerm;
import com.powsybl.openloadflow.dc.equations.DcEquationSystemCreationParameters;
import com.powsybl.openloadflow.dc.equations.DcEquationType;
import com.powsybl.openloadflow.dc.equations.DcVariableType;
import com.powsybl.openloadflow.dc.fastdc.ComputedContingencyElement;
import com.powsybl.openloadflow.dc.fastdc.ComputedElement;
import com.powsybl.openloadflow.dc.fastdc.ComputedSwitchBranchElement;
import com.powsybl.openloadflow.dc.fastdc.ComputedTapPositionChangeElement;
import com.powsybl.openloadflow.dc.fastdc.WoodburyEngine;
import com.powsybl.openloadflow.dc.gpu.GpuDcAssembly;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.graph.GraphConnectivityFactory;
import com.powsybl.openloadflow.network.DisabledNetwork;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.TapPositionChange;
import com.powsybl.openloadflow.network.action.LfAction;
import com.powsybl.openloadflow.network.action.LfOperatorStrategy;
import com.powsybl.openloadflow.network.impl.LfLegBranch;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.network.util.ParticipatingElement;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
import com.powsybl.sensitivity.SensitivityAnalysisParameters;
import com.powsybl.sensitivity.SensitivityFunctionType;
import com.powsybl.sensitivity.SensitivityResultWriter;
import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * {@code DcSensitivityAnalysis} with the states on the device.
 *
 * <p>The whole point, and the thing the LODF measurement taught: a device sensitivity kernel only
 * pays when the numbers it produces are CONSUMED on the device instead of shipped. So here the
 * factor states ({@code equations x factor groups}) and the flow states ({@code equations}) are
 * solved into device memory once and stay there for the whole contingency loop; per contingency the
 * Woodbury update rebuilds them in place on the device, the per-factor {@code calculateSensi}
 * evaluations run in a kernel, and the only thing that crosses back is the VALUES — one double per
 * factor, orders of magnitude smaller than the states they come from.
 *
 * <p>Everything else is OLF's, inherited untouched: reading and checking the factors, grouping them,
 * the participation factors, the connectivity break analysis, the per-contingency network state
 * bookkeeping, GLSK rescaling, statuses, unscaling, filtering and writing. This class overrides four
 * seams:
 * <ul>
 *   <li>{@code calculateFactorStates} — builds OLF's own right-hand sides through a recording matrix
 *       ({@link SparseColumnRecorder}) and solves them on the device;</li>
 *   <li>{@code calculateFlowStates} — keeps OLF's modified-target build and its single host solve,
 *       then uploads the one resulting column;</li>
 *   <li>{@code createWoodburyEngine} — installs an engine whose post-contingency update runs on the
 *       device instead of on a host matrix;</li>
 *   <li>{@code calculateSensitivityValues} — evaluates the factors in a kernel.</li>
 * </ul>
 *
 * <p>The state matrices OLF passes between those seams are PLACEHOLDERS: tiny {@code DenseMatrix}
 * instances whose row count tags which device buffer they stand for. Nothing outside the four seams
 * reads them (OLF only ever copies them and passes them along), and this way none of OLF's
 * orchestration had to be duplicated.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuDcSensitivityAnalysis extends DcSensitivityAnalysis {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuDcSensitivityAnalysis.class);

    /** The last run's factor evaluations done on the device (-1 when the device never engaged). */
    public static final AtomicInteger LAST_DEVICE_FACTOR_COUNT = new AtomicInteger(-1);

    /** With {@code OLF_GPU_PHASE_TIME}: nanoseconds in the device call, in OLF's per-factor
     *  predefined-result scan, and in OLF's per-factor unscale/filter/write tail. */
    private long deviceNanos;
    private long predefinedNanos;
    private long writeNanos;
    private int stateCount;

    // placeholder tags, carried by the row count so that OLF's `new DenseMatrix(rows, cols)` copies keep them
    private static final int TAG_BASE_FLOW = 1;
    private static final int TAG_BASE_FACTOR = 2;
    private static final int TAG_ALT_FLOW = 3;
    private static final int TAG_ALT_FACTOR = 4;

    private GpuDcSolver solver;
    private GpuDcAssembly assembly;
    private DcLoadFlowContext deviceContext;
    private boolean baseFlowBuilt;
    private boolean baseFactorBuilt;
    private int factorColumnCount;

    /** Set by the device Woodbury engine, consumed by the next {@code calculateSensitivityValues}. */
    private boolean woodburyPending;
    private int pendingFactorSource = GpuDcSolver.SRC_BASE;
    private int pendingFlowSource = GpuDcSolver.SRC_BASE;

    public GpuDcSensitivityAnalysis(MatrixFactory matrixFactory, GraphConnectivityFactory<LfBus, LfBranch> connectivityFactory,
                                    SensitivityAnalysisParameters parameters) {
        super(matrixFactory, connectivityFactory, parameters);
    }

    private static DenseMatrix placeholder(int tag) {
        return new DenseMatrix(tag, 1);
    }

    private static boolean isFlow(DenseMatrix states) {
        return states.getRowCount() == TAG_BASE_FLOW || states.getRowCount() == TAG_ALT_FLOW;
    }

    private static int sourceOf(DenseMatrix states) {
        int tag = states.getRowCount();
        return tag == TAG_ALT_FLOW || tag == TAG_ALT_FACTOR ? GpuDcSolver.SRC_ALT : GpuDcSolver.SRC_BASE;
    }

    private void ensureDevice(DcLoadFlowContext loadFlowContext) {
        if (solver != null) {
            return;
        }
        deviceContext = loadFlowContext;
        assembly = GpuDcAssembly.capture(loadFlowContext.getNetwork(), loadFlowContext.getEquationSystem(),
                loadFlowContext.getParameters().getEquationSystemCreationParameters());
        solver = GpuDcSolver.createAssembled(assembly, GpuLoadFlowParameters.get(parameters.getLoadFlowParameters()).getLinearSolver());
        LAST_DEVICE_FACTOR_COUNT.set(0);
    }

    @Override
    protected DenseMatrix calculateFlowStates(DcLoadFlowContext loadFlowContext, List<ParticipatingElement> participatingElements,
                                              DisabledNetwork disabledNetwork, List<LfAction> actions, ReportNode reportNode) {
        // OLF's own modified-target build and its single solve: one column, on the host, exactly as
        // the CPU engine does it. Only the result is handed to the device.
        DenseMatrix host = super.calculateFlowStates(loadFlowContext, participatingElements, disabledNetwork, actions, reportNode);
        ensureDevice(loadFlowContext);
        double[] column = new double[host.getRowCount()];
        for (int i = 0; i < column.length; i++) {
            column[i] = host.get(i, 0);
        }
        boolean alt = baseFlowBuilt;
        solver.sensiSetFlowStates(column, alt);
        baseFlowBuilt = true;
        return placeholder(alt ? TAG_ALT_FLOW : TAG_BASE_FLOW);
    }

    @Override
    protected DenseMatrix calculateFactorStates(DcLoadFlowContext loadFlowContext,
                                                SensitivityFactorGroupList<DcVariableType, DcEquationType> factorGroups,
                                                List<ParticipatingElement> participatingElements) {
        ensureDevice(loadFlowContext);
        // DcSensitivityAnalysis.calculateFactorStates' participation map, then OLF's own fillRhs
        Map<LfBus, Double> slackParticipationByBus;
        if (participatingElements.isEmpty()) {
            slackParticipationByBus = Map.of(loadFlowContext.getNetwork().getSynchronousNetworks().getFirst().getSlackBuses().getFirst(), -1d);
        } else {
            slackParticipationByBus = participatingElements.stream().collect(Collectors.toMap(
                    ParticipatingElement::getLfBus,
                    element -> -element.getFactor(),
                    Double::sum));
        }
        int equationCount = loadFlowContext.getEquationSystem().getIndex().getColumnCount();
        int groupCount = factorGroups.getList().size();
        SparseColumnRecorder recorder = new SparseColumnRecorder(equationCount, groupCount);
        for (SensitivityFactorGroup<DcVariableType, DcEquationType> factorGroup : factorGroups.getList()) {
            factorGroup.fillRhs(recorder, slackParticipationByBus);
        }
        boolean alt = baseFactorBuilt;
        factorColumnCount = groupCount;
        solver.sensiSetFactorColumns(recorder.columnPointers(), recorder.rows(), recorder.values(), groupCount, alt);
        baseFactorBuilt = true;
        return placeholder(alt ? TAG_ALT_FACTOR : TAG_BASE_FACTOR);
    }

    @Override
    protected WoodburyEngine createWoodburyEngine(DcEquationSystemCreationParameters creationParameters,
                                                  List<ComputedContingencyElement> contingencyElements,
                                                  DenseMatrix contingenciesStates,
                                                  List<ComputedElement> actionElements,
                                                  DenseMatrix actionsStates) {
        uploadElementStates(contingenciesStates, actionsStates);
        return new DeviceWoodburyEngine(creationParameters, contingencyElements, contingenciesStates, actionElements, actionsStates);
    }

    private DenseMatrix lastContingenciesStates;
    private DenseMatrix lastActionsStates;

    /** The {@code +1/-1} element states OLF solved on the host, uploaded once (they do not change). */
    private void uploadElementStates(DenseMatrix contingenciesStates, DenseMatrix actionsStates) {
        if (contingenciesStates == lastContingenciesStates && actionsStates == lastActionsStates) {
            return;
        }
        lastContingenciesStates = contingenciesStates;
        lastActionsStates = actionsStates;
        solver.sensiSetElementStates(columnMajor(contingenciesStates), contingenciesStates.getColumnCount(),
                columnMajor(actionsStates), actionsStates.getColumnCount());
    }

    private static double[] columnMajor(DenseMatrix matrix) {
        int rows = matrix.getRowCount();
        int columns = matrix.getColumnCount();
        double[] values = new double[Math.max(rows * columns, 1)];
        for (int j = 0; j < columns; j++) {
            for (int i = 0; i < rows; i++) {
                values[i + j * rows] = matrix.get(i, j);
            }
        }
        return values;
    }

    /**
     * {@code WoodburyEngine} whose post-contingency update happens on the device. The call only
     * records which buffer is being updated; the kernels run once for both buffers when the values
     * are asked for, which is the very next thing OLF does.
     */
    private final class DeviceWoodburyEngine extends WoodburyEngine {

        private final List<ComputedContingencyElement> contingencyElements;
        private final List<ComputedElement> actionElements;
        private final DcEquationSystemCreationParameters creationParameters;

        private DeviceWoodburyEngine(DcEquationSystemCreationParameters creationParameters,
                                     List<ComputedContingencyElement> contingencyElements, DenseMatrix contingenciesStates,
                                     List<ComputedElement> actionElements, DenseMatrix actionsStates) {
            super(creationParameters, contingencyElements, contingenciesStates, actionElements, actionsStates);
            this.creationParameters = creationParameters;
            this.contingencyElements = contingencyElements;
            this.actionElements = actionElements;
        }

        @Override
        public void toPostContingencyAndOperatorStrategyStates(DenseMatrix states) {
            if (!woodburyPending) {
                pendingElements = elementTables(creationParameters, contingencyElements, actionElements);
                pendingFactorSource = GpuDcSolver.SRC_BASE;
                pendingFlowSource = GpuDcSolver.SRC_BASE;
            }
            if (isFlow(states)) {
                pendingFlowSource = sourceOf(states);
            } else {
                pendingFactorSource = sourceOf(states);
            }
            woodburyPending = true;
        }
    }

    /** The current contingency's element tables, as {@code WoodburyEngine}'s alpha system needs them. */
    private record ElementTables(int[] ph1Row, int[] ph2Row, int[] stateColumn, int[] kind, double[] diag, double[] newAlpha) {
    }

    private ElementTables pendingElements;

    private ElementTables elementTables(DcEquationSystemCreationParameters creationParameters,
                                        List<ComputedContingencyElement> contingencyElements, List<ComputedElement> actionElements) {
        List<ComputedElement> all = new ArrayList<>(contingencyElements);
        all.addAll(actionElements);
        int k = all.size();
        int[] ph1Row = new int[k];
        int[] ph2Row = new int[k];
        int[] stateColumn = new int[k];
        int[] kind = new int[k];
        double[] diag = new double[k];
        double[] newAlpha = new double[k];
        for (int e = 0; e < k; e++) {
            ComputedElement element = all.get(e);
            LfBranch branch = element.getLfBranch();
            var p1 = element.getLfBranchEquation();
            ph1Row[e] = p1.getPh1Var().getRow();
            ph2Row[e] = p1.getPh2Var().getRow();
            stateColumn[e] = element.getComputedElementIndex();
            kind[e] = element instanceof ComputedContingencyElement ? GpuDcSolver.EL_CONTINGENCY : GpuDcSolver.EL_ACTION;
            double power = AbstractClosedBranchDcFlowEquationTerm.computePower(creationParameters.isUseTransformerRatio(),
                    creationParameters.getDcApproximationType(), branch.getPiModel());
            // WoodburyEngine.getAlphaMatrixValue's diagonal and getAlphaRhsValue's new phase shift
            if (element instanceof ComputedContingencyElement) {
                diag[e] = 1d / power;
            } else if (element instanceof ComputedTapPositionChangeElement tap) {
                TapPositionChange change = tap.getTapPositionChange();
                double newPower = AbstractClosedBranchDcFlowEquationTerm.computePower(creationParameters.isUseTransformerRatio(),
                        creationParameters.getDcApproximationType(), change.getNewPiModel());
                diag[e] = 1d / (power - newPower);
                newAlpha[e] = change.getNewPiModel().getA1();
            } else if (element instanceof ComputedSwitchBranchElement sw) {
                diag[e] = sw.isEnabled() ? 1d / (0 - power) : 1d / power;
            } else {
                throw new IllegalStateException("Unexpected computed element type " + element.getClass().getSimpleName());
            }
        }
        return new ElementTables(ph1Row, ph2Row, stateColumn, kind, diag, newAlpha);
    }

    /** The side of the branch flow a function type observes ({@code getFunctionEquationTerm}). */
    private static int functionSide(LfSensitivityFactor<DcVariableType, DcEquationType> factor) {
        if (factor.getFunctionType() == SensitivityFunctionType.BRANCH_ACTIVE_POWER_2
                && !(factor.getFunctionElement() instanceof LfLegBranch)) {
            return 2;
        }
        return 1;
    }

    @Override
    protected void calculateSensitivityValues(List<LfSensitivityFactor<DcVariableType, DcEquationType>> lfFactors,
                                              DenseMatrix factorStates, DenseMatrix flowStates, PropagatedContingency contingency,
                                              LfOperatorStrategy operatorStrategy, SensitivityResultWriter resultWriter,
                                              DisabledNetwork disabledNetwork) {
        if (lfFactors.isEmpty()) {
            consumeWoodbury();
            return;
        }
        boolean contingencyUpdate = woodburyPending;
        int factorSource = contingencyUpdate ? pendingFactorSource : sourceOf(factorStates);
        int flowSource = contingencyUpdate ? pendingFlowSource : sourceOf(flowStates);
        woodburyPending = false;

        // OLF's own order: the function-only factors first, then the valid ones grouped by factor group
        List<LfSensitivityFactor<DcVariableType, DcEquationType>> ordered = new ArrayList<>();
        List<SensitivityFactorGroup<DcVariableType, DcEquationType>> groups = new ArrayList<>();
        for (LfSensitivityFactor<DcVariableType, DcEquationType> factor : lfFactors) {
            if (factor.getStatus() == LfSensitivityFactor.Status.VALID_ONLY_FOR_FUNCTION) {
                ordered.add(factor);
                groups.add(null);
            }
        }
        Map<SensitivityFactorGroup<DcVariableType, DcEquationType>, List<LfSensitivityFactor<DcVariableType, DcEquationType>>> factorsByGroup =
                lfFactors.stream()
                        .filter(factor -> factor.getStatus() == LfSensitivityFactor.Status.VALID)
                        .collect(Collectors.groupingBy(LfSensitivityFactor::getGroup, LinkedHashMap::new, Collectors.toList()));
        for (var e : factorsByGroup.entrySet()) {
            for (LfSensitivityFactor<DcVariableType, DcEquationType> factor : e.getValue()) {
                ordered.add(factor);
                groups.add(e.getKey());
            }
        }

        // what the device has to evaluate: the sensitivities that have no predefined result, and the
        // function references that have none either
        int count = ordered.size();
        long tPredefined = System.nanoTime();
        Pair<Optional<Double>, Optional<Double>>[] predefined = new Pair[count];
        List<Integer> pairFunction = new ArrayList<>();
        List<Integer> pairSide = new ArrayList<>();
        List<Integer> pairColumn = new ArrayList<>();
        int[] pairOf = new int[count];
        List<Integer> functionBranch = new ArrayList<>();
        List<Integer> functionSideList = new ArrayList<>();
        int[] functionOf = new int[count];
        for (int i = 0; i < count; i++) {
            LfSensitivityFactor<DcVariableType, DcEquationType> factor = ordered.get(i);
            predefined[i] = getPredefinedResults(factor, disabledNetwork, contingency);
            pairOf[i] = -1;
            functionOf[i] = -1;
            int branch = ((LfBranch) factor.getFunctionElement()).getNum();
            int side = functionSide(factor);
            if (predefined[i].getRight().isEmpty()) {
                functionOf[i] = functionBranch.size();
                functionBranch.add(branch);
                functionSideList.add(side);
            }
            if (predefined[i].getLeft().isEmpty() && groups.get(i) != null) {
                pairOf[i] = pairFunction.size();
                pairFunction.add(branch);
                pairSide.add(side);
                pairColumn.add(groups.get(i).getIndex());
            }
        }
        long tDevice = System.nanoTime();
        predefinedNanos += tDevice - tPredefined;
        double[] sensiValues = new double[Math.max(pairFunction.size(), 1)];
        double[] functionValues = new double[Math.max(functionBranch.size(), 1)];
        if (contingencyUpdate) {
            // elements, Woodbury and evaluations in ONE device call: split up, the launch and
            // allocation overhead cost more than the arithmetic
            solver.sensiContingency(pendingElements.ph1Row(), pendingElements.ph2Row(), pendingElements.stateColumn(),
                    pendingElements.kind(), pendingElements.diag(), pendingElements.newAlpha(), factorSource, flowSource,
                    toArray(pairFunction), toArray(pairSide), toArray(pairColumn), toArray(functionBranch),
                    toArray(functionSideList), sensiValues, functionValues);
        } else {
            solver.sensiValues(factorSource, flowSource, toArray(pairFunction), toArray(pairSide), toArray(pairColumn),
                    toArray(functionBranch), toArray(functionSideList), sensiValues, functionValues);
        }
        long tWrite = System.nanoTime();
        deviceNanos += tWrite - tDevice;
        LAST_DEVICE_FACTOR_COUNT.addAndGet(pairFunction.size());

        // then OLF's own per-factor tail: zero-reference fix, unscaling, filtering, writing
        for (int i = 0; i < count; i++) {
            LfSensitivityFactor<DcVariableType, DcEquationType> factor = ordered.get(i);
            double sensitivityValue = predefined[i].getLeft().orElse(0d);
            double functionValue = predefined[i].getRight().orElse(0d);
            if (functionOf[i] >= 0) {
                functionValue = functionValues[functionOf[i]];
            }
            if (pairOf[i] >= 0) {
                sensitivityValue = sensiValues[pairOf[i]];
            }
            functionValue = fixZeroFunctionReference(contingency, functionValue);
            double unscaledSensi = unscaleSensitivity(factor, sensitivityValue);
            if (!filterSensitivityValue(unscaledSensi, factor.getVariableType(), factor.getFunctionType(), parameters)) {
                resultWriter.writeSensitivityValue(factor.getIndex(), contingency != null ? contingency.getIndex() : -1,
                        operatorStrategy != null ? operatorStrategy.getIndex() : -1, unscaledSensi, unscaleFunction(factor, functionValue));
            }
        }
        writeNanos += System.nanoTime() - tWrite;
        stateCount++;
        if (GpuTrace.PHASE_TIME.isOn() && stateCount % 100 == 0) {
            System.err.printf("DC_SENSI_PHASE states=%d factors/state=%d device=%d ms predefined=%d ms unscale+filter+write=%d ms%n",
                    stateCount, count, deviceNanos / 1_000_000, predefinedNanos / 1_000_000, writeNanos / 1_000_000);
        }
    }

    /**
     * {@code DcSensitivityAnalysis.fixZeroFunctionReference}: a post-contingency reference flow that
     * should be strictly zero cannot be, since it is shifted from a non-zero pre-contingency flow.
     */
    private static double fixZeroFunctionReference(PropagatedContingency contingency, double functionValue) {
        if (contingency != null) {
            return Math.abs(functionValue) < 1e-13 ? 0 : functionValue;
        }
        return functionValue;
    }

    /** A Woodbury update whose values are never asked for still has to be dropped. */
    private void consumeWoodbury() {
        woodburyPending = false;
    }

    private static int[] toArray(List<Integer> list) {
        int[] array = new int[list.size()];
        for (int i = 0; i < array.length; i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    /** Closes the device context; the analysis object is not reused after a run. */
    public void close() {
        if (solver != null) {
            solver.close();
            solver = null;
        }
    }
}
