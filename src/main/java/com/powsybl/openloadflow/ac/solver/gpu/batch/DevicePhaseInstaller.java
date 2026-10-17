/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import java.util.ArrayList;
import java.util.List;

/**
 * The run-constant tables of the control loops the device runs as driver phases (OLF orders 7 to 10: incremental
 * and discrete transformer voltage control, incremental flow controls, continuous phase control, incremental and
 * discrete shunt voltage control), installed on a batch context.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class DevicePhaseInstaller {

    private DevicePhaseInstaller() {
    }

    /** Which of the run's discrete and incremental control loops the device owns, and the ones it refused. */
    record InstalledPhases(boolean incrementalTvc, boolean flow, boolean continuousPhase, boolean incrementalShunt,
                           boolean discreteTvc, boolean simpleShunt, List<String> refused) {
    }

    /** Install every control-loop table of the run on the (possibly cached) context, or clear it when the run has
     *  none of that loop. Returns which loops the device owns; a loop the run has and the device does not own is
     *  listed in {@link InstalledPhases#refused()}. */
    static InstalledPhases install(long handle, GpuBatchPhases.Built phases, BatchConfig config) {
        // IncrementalTransformerVoltageControl as a DRIVER PHASE (OLF order 8), shared controlled
        // buses included (the one-tap-at-a-time round robin runs in the group kernel).
        // Every table upload also CLEARS when the run has none: the context is cached across runs on the
        // same network, and a stale table from a previous run selects the wrong driver phase (a Simple
        // TVC run after an INCREMENTAL one ran LOOP_TVC_KERNEL on its controls — 3.6e-7 kV).
        boolean tvcInKernel = false;
        if (phases.incrementalTvc() != null && !phases.incrementalTvc().isEmpty()) {
            tvcInKernel = GpuBatchIncrementalTvc.uploadToDevice(handle, phases.incrementalTvc(),
                    config.incTvcMaxTapShift());
        } else {
            GpuBatchIncrementalTvc.uploadToDevice(handle, null, 1);
        }
        // The flow-observed incremental loops (RPC at OLF order 9, incremental phase at order 7) as DRIVER PHASES.
        boolean flowInKernel = false;
        if (phases.flow() != null && !phases.flow().isEmpty()) {
            flowInKernel = GpuBatchIncrementalFlowControl.uploadToDevice(handle, phases.flow());
        } else {
            GpuBatchIncrementalFlowControl.uploadToDevice(handle, null);
        }
        // CONTINUOUS PhaseControl as a DRIVER PHASE (OLF order 7); the pre-solve control-ON activation stays host-side.
        boolean cphInKernel = false;
        if (phases.continuousPhase() != null && !phases.continuousPhase().isEmpty()) {
            cphInKernel = GpuBatchContinuousPhase.uploadToDevice(handle, phases.continuousPhase());
        } else {
            GpuBatchContinuousPhase.uploadToDevice(handle, null);
        }
        // IncrementalShuntVoltageControl as a DRIVER PHASE (OLF order 10).
        boolean shuntInKernel = false;
        if (phases.incrementalShunt() != null && !phases.incrementalShunt().isEmpty()) {
            shuntInKernel = GpuBatchIncrementalShuntVc.uploadToDevice(handle, phases.incrementalShunt(),
                    config.incShuntMaxSectionShift());
        } else {
            GpuBatchIncrementalShuntVc.uploadToDevice(handle, null, 3);
        }
        // The DISCRETE TransformerVoltageControl (AFTER_GENERATOR / WITH_GENERATOR, OLF order 8) and the
        // WITH_GENERATOR ShuntVoltageControl (order 10) as DRIVER PHASES: the control-ON entry (OLF's
        // initialize) and the rounding/dispatch checks run in-kernel; the host machines stay for a
        // control the upload refuses (a member without a tap table).
        boolean tvdInKernel = false;
        if ((config.transformerControl() || config.simpleTransformerControl())
                && phases.discreteTvc() != null && !phases.discreteTvc().isEmpty()) {
            tvdInKernel = GpuBatchTransformerControl.uploadToDevice(handle, phases.discreteTvc(),
                    config.simpleTransformerControl() ? 1 : 0, false);
            if (tvdInKernel) {
                // AFTER_GENERATOR: OLF's GeneratorVoltageControlManager freeze + fixTransformerVoltageControls
                GpuBatchTransformerControl.uploadFreeze(handle, config.transformerControl() ? phases.freeze() : null);
            } else if (phases.freeze() != null && phases.freeze().hasCandidates()) {
                throw new com.powsybl.commons.PowsyblException("GPU batch: the transformer voltage control needs the "
                        + "in-kernel phase (generator freeze under the controlled level) but a control has no tap table");
            }
        } else {
            GpuBatchTransformerControl.uploadToDevice(handle, null, 0, false);   // clears a cached context's table
        }
        boolean svdInKernel = false;
        if (config.simpleShuntControl() && phases.simpleShunt() != null && !phases.simpleShunt().isEmpty()) {
            svdInKernel = GpuBatchSimpleShuntVc.uploadToDevice(handle, phases.simpleShunt(), false);
        } else {
            GpuBatchSimpleShuntVc.uploadToDevice(handle, null, false);
        }

        List<String> refusedLoops = new ArrayList<>();
        if (phases.incrementalTvc() != null && !phases.incrementalTvc().isEmpty() && !tvcInKernel) {
            refusedLoops.add("IncrementalTransformerVoltageControl");
        }
        if (phases.flow() != null && !phases.flow().isEmpty() && !flowInKernel) {
            refusedLoops.add("IncrementalTransformerReactivePowerControl/IncrementalPhaseControl");
        }
        if (phases.continuousPhase() != null && !phases.continuousPhase().isEmpty() && !cphInKernel) {
            refusedLoops.add("PhaseControl");
        }
        if (phases.incrementalShunt() != null && !phases.incrementalShunt().isEmpty() && !shuntInKernel) {
            refusedLoops.add("IncrementalShuntVoltageControl");
        }
        if ((config.transformerControl() || config.simpleTransformerControl())
                && phases.discreteTvc() != null && !phases.discreteTvc().isEmpty() && !tvdInKernel) {
            refusedLoops.add("TransformerVoltageControl");
        }
        if (config.simpleShuntControl() && phases.simpleShunt() != null && !phases.simpleShunt().isEmpty() && !svdInKernel) {
            refusedLoops.add("ShuntVoltageControl");
        }
        return new InstalledPhases(tvcInKernel, flowInKernel, cphInKernel, shuntInKernel, tvdInKernel, svdInKernel,
                refusedLoops);
    }
}
