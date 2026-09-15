/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.dc.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.openloadflow.dc.equations.DcEquationType;
import com.powsybl.openloadflow.dc.equations.DcVariableType;
import com.powsybl.openloadflow.equations.Equation;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.lf.outerloop.OuterLoopStatus;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.PiModel;
import com.powsybl.openloadflow.network.PiModelArray;
import com.powsybl.openloadflow.network.TransformerPhaseControl;
import com.powsybl.openloadflow.util.PerUnit;
import com.powsybl.openloadflow.util.Reports;
import com.powsybl.iidm.network.TwoSides;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The two DC outer loops that decide from the solved state, as device kernels:
 * {@code DcIncrementalPhaseControlOuterLoop} and {@code DcHvdcAcEmulationLimitsOuterLoop}.
 *
 * <p>Split of work, the same one the AC device driver uses: the kernel READS the state and
 * produces the loop's DECISION (a new tap index per phase controller, a new AC-emulation status
 * per HVDC link), the host APPLIES it with OLF's own mutators so that everything downstream — the
 * equation updater, the listeners, the reports, the network result — is OLF's. Nothing about the
 * criteria is re-implemented on the host, and nothing about the application is re-implemented on
 * the device.
 *
 * <p>Phase control. The sensitivities {@code dP/dα} are one multi-column solve on the SAME
 * factorization the load flow uses ({@code DcSensitivityContext.calculateSensitivityValues}: a
 * right-hand side per controller with {@code toRadians(1)} at its {@code BRANCH_TARGET_ALPHA1}
 * row). They are recomputed at every check, like OLF's (the context is rebuilt per check), which
 * on the device costs one solve of k columns. The decision kernel then reproduces
 * {@code checkActivePowerControlPhaseControls} per controller: the controlled flow from the state,
 * the half deadband, {@code calculateSensi} on the solved column, {@code da = toRadians(dp/a2p)},
 * and {@code PiModelArray.updateTapPositionToReachNewA1}'s closest-tap search over the allowed
 * range, with {@code ControllerContext.updateAllowedDirection}'s ratchet held on the device.
 *
 * <p>HVDC AC emulation. The kernel evaluates the two emulation flows on the state and applies
 * {@code checkAcEmulationMode}'s transition table (linear → saturated either way, saturated →
 * linear, saturated → opposite saturation); the host calls
 * {@code AcEmulationControl.switchTo...} so the VSC target P and the equation deactivation are
 * OLF's own, and emits OLF's reports.
 *
 * <p>Not ported: {@code DcAreaInterchangeControl}. It is a distribution loop, not a criterion on
 * the state: it re-runs {@code ActivePowerDistribution} over area bus sets with its own iteration
 * and reporting. It stays a host-owned loop slot; the engine runs it through OLF unchanged.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuDcOuterLoops {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuDcOuterLoops.class);

    /** {@code AbstractIncrementalPhaseControlOuterLoop.MIN_TARGET_DEADBAND} (1 MW). */
    private static final double MIN_TARGET_DEADBAND = 1 / PerUnit.SB;

    private final LfNetwork network;
    private final GpuDcSolver solver;
    private final GpuDcAssembly assembly;

    // phase controllers, in OLF's controller-branch order
    private final List<TransformerPhaseControl> phaseControls = new ArrayList<>();
    private final List<PiModelArray> phaseModels = new ArrayList<>();
    private final int[] phaseLowTapPosition;

    // HVDC links in AC emulation, in LfNetwork.getHvdcs() order (every one of them: the ACTIVE
    // mask says which the loop looks at, and it is refreshed per check)
    private final List<LfHvdc> hvdcs = new ArrayList<>();

    private GpuDcOuterLoops(LfNetwork network, GpuDcSolver solver, GpuDcAssembly assembly, List<TransformerPhaseControl> controls,
                            List<PiModelArray> models, int[] lowTapPositions, List<LfHvdc> hvdcs) {
        this.network = network;
        this.solver = solver;
        this.assembly = assembly;
        this.phaseControls.addAll(controls);
        this.phaseModels.addAll(models);
        this.phaseLowTapPosition = lowTapPositions;
        this.hvdcs.addAll(hvdcs);
    }

    boolean hasPhaseControllers() {
        return !phaseControls.isEmpty();
    }

    boolean hasHvdcs() {
        return !hvdcs.isEmpty();
    }

    /**
     * Build the loop tables and upload them. Call after the outer loops' {@code initialize} (the
     * phase control one can switch a controller off for connectivity) and after
     * {@link GpuDcSolver#createAssembled}.
     *
     * @throws UnsupportedOperationException when a phase controller has no tap table (the caller keeps the host loop)
     */
    static GpuDcOuterLoops capture(LfNetwork network, EquationSystem<DcVariableType, DcEquationType> equationSystem, GpuDcSolver solver,
                                   GpuDcAssembly assembly) {
        // AbstractPhaseControlOuterLoop.getControllerBranches, then the CONTROLLER-mode controls
        List<TransformerPhaseControl> controls = new ArrayList<>();
        List<PiModelArray> models = new ArrayList<>();
        List<Integer> lowTapPositions = new ArrayList<>();
        for (LfBranch branch : network.getBranches()) {
            if (branch.isDisabled() || !branch.isPhaseController() || !branch.isConnectedAtBothSides()) {
                continue;
            }
            var phaseControl = branch.getPhaseControl().orElse(null);
            if (phaseControl == null || phaseControl.getMode() != TransformerPhaseControl.Mode.CONTROLLER) {
                continue;
            }
            PiModel piModel = branch.getPiModel();
            if (!(piModel instanceof PiModelArray piModelArray)) {
                throw new UnsupportedOperationException("phase controller '" + branch.getId() + "' has no tap table");
            }
            controls.add(phaseControl);
            models.add(piModelArray);
            lowTapPositions.add(piModelArray.getTapPositionRange().getMinimum());
        }
        List<LfHvdc> hvdcs = new ArrayList<>(network.getHvdcs());
        int[] lows = new int[lowTapPositions.size()];
        for (int i = 0; i < lows.length; i++) {
            lows[i] = lowTapPositions.get(i);
        }
        GpuDcOuterLoops loops = new GpuDcOuterLoops(network, solver, assembly, controls, models, lows, hvdcs);
        loops.upload(equationSystem);
        return loops;
    }

    /** Re-read the tables from the network (a tap moved, a status changed) and re-upload them. */
    void upload(EquationSystem<DcVariableType, DcEquationType> equationSystem) {
        uploadPhaseControllers(equationSystem);
        uploadHvdcs();
    }

    private void uploadPhaseControllers(EquationSystem<DcVariableType, DcEquationType> equationSystem) {
        int m = phaseControls.size();
        if (m == 0) {
            return;
        }
        int[] branch = new int[m];
        int[] alphaColumn = new int[m];
        int[] controlled = new int[m];
        int[] side = new int[m];
        double[] target = new double[m];
        double[] halfDeadband = new double[m];
        int[] tapIndex = new int[m];
        int[] tapPtr = new int[m + 1];
        List<Double> tapA1 = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            TransformerPhaseControl control = phaseControls.get(i);
            PiModelArray piModel = phaseModels.get(i);
            branch[i] = control.getControllerBranch().getNum();
            alphaColumn[i] = equationSystem.getEquation(branch[i], DcEquationType.BRANCH_TARGET_ALPHA1)
                    .map(Equation::getColumn)
                    .orElse(-1);
            controlled[i] = control.getControlledBranch().getNum();
            side[i] = control.getControlledSide() == TwoSides.ONE ? 1 : 2;
            target[i] = control.getTargetValue();
            // AbstractIncrementalPhaseControlOuterLoop.getHalfTargetDeadband
            halfDeadband[i] = Math.max(control.getTargetDeadband(), MIN_TARGET_DEADBAND) / 2;
            tapPtr[i] = tapA1.size();
            var range = piModel.getTapPositionRange();
            for (int tap = range.getMinimum(); tap <= range.getMaximum(); tap++) {
                tapA1.add(piModel.getModel(tap).getA1());
            }
            tapIndex[i] = piModel.getTapPosition() - phaseLowTapPosition[i];
        }
        tapPtr[m] = tapA1.size();
        double[] a1 = new double[tapA1.size()];
        for (int i = 0; i < a1.length; i++) {
            a1[i] = tapA1.get(i);
        }
        solver.setPhaseControllers(branch, alphaColumn, controlled, side, target, halfDeadband, tapIndex, tapPtr, a1);
    }

    private void uploadHvdcs() {
        int m = hvdcs.size();
        if (m == 0) {
            return;
        }
        int[] ph1 = new int[m];
        int[] ph2 = new int[m];
        double[] k = new double[m];
        double[] p0 = new double[m];
        double[] pMax12 = new double[m];
        double[] pMax21 = new double[m];
        int[] status = new int[m];
        int[] active = new int[m];
        for (int i = 0; i < m; i++) {
            LfHvdc hvdc = hvdcs.get(i);
            var control = hvdc.getAcEmulationControl();
            // DcHvdcAcEmulationLimitsOuterLoop.check's own filter
            boolean on = hvdc.isAcEmulation() && !hvdc.isDisabled()
                    && hvdc.getBus1() != null && hvdc.getBus2() != null
                    && !hvdc.getBus1().isDisabled() && !hvdc.getBus2().isDisabled();
            ph1[i] = on ? assembly.busPhRow[hvdc.getBus1().getNum()] : -1;
            ph2[i] = on ? assembly.busPhRow[hvdc.getBus2().getNum()] : -1;
            active[i] = on && ph1[i] >= 0 && ph2[i] >= 0 ? 1 : 0;
            // AbstractHvdcAcEmulationDcFlowEquationTerm: k = droop * 180 / PI
            k[i] = control.getDroop() * 180 / Math.PI;
            p0[i] = control.getP0();
            pMax12[i] = control.getPMaxFromCS1toCS2();
            pMax21[i] = control.getPMaxFromCS2toCS1();
            status[i] = statusCode(control.getAcEmulationStatus());
        }
        solver.setHvdcs(ph1, ph2, k, p0, pMax12, pMax21, status, active);
    }

    private static int statusCode(LfHvdc.AcEmulationControl.AcEmulationStatus status) {
        return switch (status) {
            case LINEAR_MODE -> 0;
            case SATURATION_MODE_FROM_CS1_TO_CS2 -> 1;
            case SATURATION_MODE_FROM_CS2_TO_CS1 -> 2;
            default -> 3;                                    // FROZEN: the loop leaves it alone
        };
    }

    /** {@code IncrementalContextData} rebuilt: the direction ratchets go back to BOTH. */
    void resetPhaseControlDirections() {
        solver.resetPhaseControlDirections();
    }

    /**
     * {@code DcIncrementalPhaseControlOuterLoop.check}: the device decides, the host moves the taps
     * with {@code PiModel.setTapPosition} so the listeners and the equation updater see OLF's own
     * change.
     */
    OuterLoopStatus checkPhaseControl() {
        int[] newTap = solver.phaseControlCheck();
        boolean changed = false;
        for (int i = 0; i < newTap.length; i++) {
            if (newTap[i] < 0) {
                continue;
            }
            PiModelArray piModel = phaseModels.get(i);
            int oldTapPosition = piModel.getTapPosition();
            int tapPosition = phaseLowTapPosition[i] + newTap[i];
            piModel.setTapPosition(tapPosition);
            LOGGER.debug("Controller branch '{}' change tap from {} to {} to reach active power target",
                    phaseControls.get(i).getControllerBranch().getId(), oldTapPosition, tapPosition);
            GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.incrementAndGet();
            changed = true;
        }
        return changed ? OuterLoopStatus.UNSTABLE : OuterLoopStatus.STABLE;
    }

    /**
     * {@code DcHvdcAcEmulationLimitsOuterLoop.check}: the device decides, the host switches the
     * mode with {@code AcEmulationControl}'s own methods (VSC target P, status, listeners) and
     * emits OLF's reports.
     */
    OuterLoopStatus checkHvdc(ReportNode reportNode) {
        int[] newStatus = solver.hvdcCheck();
        boolean changed = false;
        for (int i = 0; i < newStatus.length; i++) {
            if (newStatus[i] < 0) {
                continue;
            }
            LfHvdc hvdc = hvdcs.get(i);
            var control = hvdc.getAcEmulationControl();
            boolean fromLinear = control.getAcEmulationStatus() == LfHvdc.AcEmulationControl.AcEmulationStatus.LINEAR_MODE;
            switch (newStatus[i]) {
                case 0 -> {
                    control.switchToLinearMode();
                    Reports.reportAcEmulationBackToLinear(reportNode, hvdc.getId(), LOGGER);
                }
                case 1 -> {
                    control.switchToSaturationFromCS1toCS2(false);
                    report(reportNode, hvdc, fromLinear, hvdc.getConverterStation1().getId(), hvdc.getConverterStation2().getId(),
                            control.getPMaxFromCS1toCS2());
                }
                case 2 -> {
                    control.switchToSaturationFromCS2toCS1(false);
                    report(reportNode, hvdc, fromLinear, hvdc.getConverterStation2().getId(), hvdc.getConverterStation1().getId(),
                            control.getPMaxFromCS2toCS1());
                }
                default -> throw new IllegalStateException("unexpected AC emulation status " + newStatus[i]);
            }
            LOGGER.trace("Hvdc '{}' AC emulation state is changed to {}", hvdc.getId(), control.getAcEmulationStatus());
            GpuDcLoadFlowEngine.LAST_DEVICE_HVDC_SWITCHES.incrementAndGet();
            changed = true;
        }
        return changed ? OuterLoopStatus.UNSTABLE : OuterLoopStatus.STABLE;
    }

    private static void report(ReportNode reportNode, LfHvdc hvdc, boolean fromLinear, String controllerStationId,
                               String nonControllerStationId, double pMax) {
        if (fromLinear) {
            Reports.reportAcEmulationFromLinearToSaturated(reportNode, hvdc.getId(), controllerStationId, nonControllerStationId,
                    pMax * PerUnit.SB, LOGGER);
        } else {
            Reports.reportAcEmulationSaturationSideSwitch(reportNode, hvdc.getId(), controllerStationId, nonControllerStationId,
                    pMax * PerUnit.SB, LOGGER);
        }
    }

    LfNetwork getNetwork() {
        return Objects.requireNonNull(network);
    }
}
