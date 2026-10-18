/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;


import com.powsybl.openloadflow.ac.outerloop.gpu.ContinuousPhaseControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteShuntVoltageControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteTransformerVoltageControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.IncrementalFlowControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.IncrementalShuntVoltageControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.IncrementalTransformerVoltageControl;

import java.util.ArrayList;
import java.util.List;

/**
 * The run-constant tables of the control loops the device runs as driver phases (OLF orders 7 to 10: incremental
 * and discrete transformer voltage control, incremental flow controls, continuous phase control, incremental and
 * discrete shunt voltage control), installed on a batch context: the same installation for the base load flow's
 * batch of one and for every security analysis chunk.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DevicePhaseInstaller {

    private DevicePhaseInstaller() {
    }

    /** One run's control loops: per loop its controllers, or null when the run does not have that loop. OLF runs at
     *  most one loop of each type (ExplicitAcOuterLoopConfig.checkTypeUnicity), so an incremental loop and its
     *  discrete twin are never both present. {@code entryOn}: the activation the solve starts from already carries
     *  the controls OLF's initialize switched ON (the base load flow); a chunk applies them at solve entry itself. */
    public record Request(List<IncrementalTransformerVoltageControl.Controller> incrementalTvc, int incTvcMaxTapShift,
                          List<IncrementalFlowControl.Controller> flow,
                          List<ContinuousPhaseControl.Ctl> continuousPhase,
                          List<IncrementalShuntVoltageControl.Controller> incrementalShunt, int incShuntMaxSectionShift,
                          List<DiscreteTransformerVoltageControl.TransformerController> discreteTvc, boolean simpleTvc,
                          boolean afterGenerator, DiscreteTransformerVoltageControl.FreezeTables freeze,
                          List<DiscreteShuntVoltageControl.Group> simpleShunt, boolean entryOn) {
    }

    /** Which of the run's discrete and incremental control loops the device owns, and the ones it refused. */
    public record InstalledPhases(boolean incrementalTvc, boolean flow, boolean continuousPhase, boolean incrementalShunt,
                                  boolean discreteTvc, boolean simpleShunt, List<String> refused) {
    }

    /** Install every control-loop table of the run on the (possibly cached) context, or clear it when the run has
     *  none of that loop. Returns which loops the device owns; a loop the run has and the device does not own is
     *  listed in {@link InstalledPhases#refused()}, and the caller decides what runs it (the security analysis
     *  sends the whole run to the CPU, the base load flow leaves OLF's own loop at its pause slot). */
    public static InstalledPhases install(long handle, Request r) {
        // IncrementalTransformerVoltageControl as a DRIVER PHASE (OLF order 8), shared controlled
        // buses included (the one-tap-at-a-time round robin runs in the group kernel).
        // Every table upload also CLEARS when the run has none: the context is cached across runs on the
        // same network, and a stale table from a previous run selects the wrong driver phase (a Simple
        // TVC run after an INCREMENTAL one ran LOOP_TVC_KERNEL on its controls — 3.6e-7 kV).
        boolean tvcInKernel = IncrementalTransformerVoltageControl.uploadToDevice(handle, r.incrementalTvc(),
                r.incrementalTvc() != null ? r.incTvcMaxTapShift() : 1);
        // The flow-observed incremental loops (RPC at OLF order 9, incremental phase at order 7) as DRIVER PHASES.
        boolean flowInKernel = IncrementalFlowControl.uploadToDevice(handle, r.flow());
        // CONTINUOUS PhaseControl as a DRIVER PHASE (OLF order 7); the pre-solve control-ON activation stays host-side.
        boolean cphInKernel = ContinuousPhaseControl.uploadToDevice(handle, r.continuousPhase());
        // IncrementalShuntVoltageControl as a DRIVER PHASE (OLF order 10).
        boolean shuntInKernel = IncrementalShuntVoltageControl.uploadToDevice(handle, r.incrementalShunt(),
                r.incrementalShunt() != null ? r.incShuntMaxSectionShift() : 3);
        // The DISCRETE TransformerVoltageControl (AFTER_GENERATOR / WITH_GENERATOR, OLF order 8) and the
        // WITH_GENERATOR ShuntVoltageControl (order 10) as DRIVER PHASES: the control-ON entry (OLF's
        // initialize) and the rounding/dispatch checks run in-kernel. A control the upload refuses (a member
        // without a tap table) is not owned.
        boolean tvdInKernel = false;
        if (r.discreteTvc() != null) {
            tvdInKernel = DiscreteTransformerVoltageControl.uploadToDevice(handle, r.discreteTvc(), r.simpleTvc() ? 1 : 0,
                    r.entryOn() && r.simpleTvc());
            if (tvdInKernel && r.afterGenerator()) {
                // AFTER_GENERATOR: OLF's GeneratorVoltageControlManager freezes the generator controls under the
                // transformer-controlled level while the transformers act. Without the freeze tables (a candidate
                // the kernel cannot represent) the device does not own the loop.
                if (r.freeze() == null) {
                    DiscreteTransformerVoltageControl.uploadToDevice(handle, null, 0, false);
                    tvdInKernel = false;
                } else {
                    DiscreteTransformerVoltageControl.uploadFreeze(handle, r.freeze());
                }
            } else if (tvdInKernel) {
                DiscreteTransformerVoltageControl.uploadFreeze(handle, null);   // WITH_GENERATOR freezes nothing
            }
        } else {
            DiscreteTransformerVoltageControl.uploadToDevice(handle, null, 0, false);   // clears a cached context's table
        }
        boolean svdInKernel = DiscreteShuntVoltageControl.uploadToDevice(handle, r.simpleShunt(), r.entryOn());

        List<String> refusedLoops = new ArrayList<>();
        if (present(r.incrementalTvc()) && !tvcInKernel) {
            refusedLoops.add("IncrementalTransformerVoltageControl");
        }
        if (present(r.flow()) && !flowInKernel) {
            refusedLoops.add("IncrementalTransformerReactivePowerControl/IncrementalPhaseControl");
        }
        if (present(r.continuousPhase()) && !cphInKernel) {
            refusedLoops.add("PhaseControl");
        }
        if (present(r.incrementalShunt()) && !shuntInKernel) {
            refusedLoops.add("IncrementalShuntVoltageControl");
        }
        if (present(r.discreteTvc()) && !tvdInKernel) {
            refusedLoops.add("TransformerVoltageControl");
        }
        if (present(r.simpleShunt()) && !svdInKernel) {
            refusedLoops.add("ShuntVoltageControl");
        }
        return new InstalledPhases(tvcInKernel, flowInKernel, cphInKernel, shuntInKernel, tvdInKernel, svdInKernel,
                refusedLoops);
    }

    private static boolean present(List<?> controllers) {
        return controllers != null && !controllers.isEmpty();
    }
}
