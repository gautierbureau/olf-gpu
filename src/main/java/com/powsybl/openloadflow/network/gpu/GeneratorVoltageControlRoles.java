/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.network.gpu;

import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.VoltageControl;

/**
 * docs/gpu-vs-olf.md section 2.3 (remote generator voltage control): OLF reads on a bus's generator voltage
 * control, made on the MAIN control of a zero-impedance merge. Not device-only: the device-only row placement
 * (the carrier, the coupler-tree relocation) asks them which role a bus plays in its group.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GeneratorVoltageControlRoles {

    private GeneratorVoltageControlRoles() {
    }

    /** True when {@code bus} is a member of a MAIN remote generator voltage-control group (its
     *  controlled bus is another bus). The FREE member (first controller-ENABLED, matching
     *  updateRemoteVoltageControlEquations) is the group's pilot-Q carrier and keeps the
     *  structure-frozen encoding; the others take the canonical suppressed-dest/mode-4 one. The reactive-limit
     *  pack enumerates its remote member entries with it and ReactiveCapabilityCurves REPLAYS that enumeration,
     *  so both must count exactly the same buses (a drift throws and the run falls to the CPU). */
    public static boolean isRemoteGenVcMember(LfBus bus) {
        GeneratorVoltageControl voltageControl = mainGeneratorVoltageControl(bus);
        return voltageControl != null && isRemoteGeneratorVoltageControlControllerElement(bus) && voltageControl.getControlledBus() != bus;
    }

    /**
     * The bus's MAIN generator voltage control, resolved the way {@link #isRemoteGeneratorVoltageControlControllerElement}
     * already resolves it and for the same reason: a control MERGED into another is DEPENDENT, and on the
     * dependent object the controlled bus is the member ITSELF, so a {@code getMergeStatus() == MAIN} filter
     * answers "not a member" for every member of a merged group but one.
     *
     * <p>Both predicates here did filter that way, so {@code ExtractionColumns}'s own-row alternate loop
     * skipped those members and left them with {@code alternateQRow = -1}. A row then put in mode 4 by a
     * reactive-limit switch to PQ assembles EMPTY: measured on rte6515's base, where the pass with that switch gave F = -target
     * exactly on both members of a pilot-self group, two identical rows and a dead DUMMY_Q column — a
     * singular matrix KLU refuses to factor.
     *
     * <p>A merged group keeps ONE main control (on the bus OLF picked) and leaves every other member holding
     * a DEPENDENT control that regulates ITSELF. Asking {@code bus.getGeneratorVoltageControl()} therefore
     * answered "I control my own bus" for every member, so each one was classified as the pilot
     * ({@code isPilotSelfEntry}) with its own row as the pilot row. Since class 2 means "still PV, my row is
     * the group's voltage IDENTITY (mode 0)" while a member's row actually carries DISTR_Q (mode 1), the
     * reactive-limit kernel's {@code if (mode[ridx] != pvForm) return;} dropped every member before it ever
     * looked at q: rte6515's base switched 16 to PQ where OLF switched 19 (VL-5997_1, VL-6003_1, VL-6003_2 missing,
     * each with a correct q of 1.921012507 pu against a 0.7536 limit), then diverged at outer pass 7.
     */
    public static GeneratorVoltageControl mainGeneratorVoltageControl(LfBus bus) {
        return bus.getGeneratorVoltageControl()
                .<GeneratorVoltageControl>map(VoltageControl::getMainVoltageControl)
                .orElse(null);
    }

    /** True when the bus is a CONTROLLER ELEMENT of a remote (non-local) generator voltage control
     *  — INCLUDING the case where it is also that control's CONTROLLED bus (a generator regulating
     *  its own bus, merged with remote controllers). Such a bus's reactive injection feeds the
     *  group's DISTR_Q rows, never a plain per-bus Q scatter.
     *
     *  <p>Resolved on the MAIN control via {@link VoltageControl#getMainVoltageControl()}, NOT by
     *  filtering {@code getMergeStatus() == MAIN}. A control that has been MERGED into another is
     *  DEPENDENT, and on the dependent object both tests below read the wrong thing: its
     *  {@code controllerElements} is just itself, so {@code isLocalControl()} is TRUE (it appears to
     *  be a generator regulating its own bus alone) and the merged element list is not reachable.
     *  Filtering DEPENDENT away therefore answered FALSE for a bus that really is a controller, and
     *  {@code ReactiveRelocation.computeQRow} then left its Q balance on its own BUS_V row — a row that already hosts
     *  the bus's DISTR_Q equation. The zero-impedance family's direct +-DUMMY_Q coupling landed
     *  there alongside the DISTR_Q derivative and the two CANCELLED: measured on rte6515
     *  BUS-24_COUPLER_1 as kind-3 terms summing +1.0 against the zero-impedance coupling's -1.0, leaving the
     *  coupler's DUMMY_Q in NO equation at all — a STRUCTURALLY ZERO COLUMN, which is invisible in
     *  every residual (the free variable perturbs nothing, so |F| looks healthy while the whole
     *  Newton step is polluted). 45 such columns at 2000 busbar sections.
     *
     *  <p>Generator voltage controls merge exactly when their buses are joined by ZERO-IMPEDANCE
     *  branches, which is why the whole population is busbar couplers and why the defect only
     *  appears once enough busbar-section contingencies retain enough switches. */
    public static boolean isRemoteGeneratorVoltageControlControllerElement(LfBus bus) {
        return bus.getGeneratorVoltageControl()
                .<GeneratorVoltageControl>map(VoltageControl::getMainVoltageControl)
                .filter(c -> !c.isLocalControl())
                .map(c -> c.getMergedControllerElements().contains(bus))
                .orElse(false);
    }

    /** The FREE member: the first ENABLED merged controller. None when no controller is enabled. */
    public static boolean isFreeRemoteGeneratorVoltageControlMember(LfBus bus) {
        GeneratorVoltageControl voltageControl = mainGeneratorVoltageControl(bus);
        return voltageControl != null && isRemoteGeneratorVoltageControlControllerElement(bus)
                && voltageControl.getMergedControllerElements().stream()
                        .filter(voltageControl::isControllerEnabled).findFirst().map(m -> m == bus).orElse(false);
    }
}
