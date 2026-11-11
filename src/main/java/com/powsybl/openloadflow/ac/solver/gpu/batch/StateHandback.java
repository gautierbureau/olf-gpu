/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

/**
 * What the host keeps of the scenarios a solve call hands back, decided while their states are still on the device (P4 of
 * {@code docs/device-final-results-plan.md}). With one on the run's {@link BatchConfig}, the chunk solver's contexts keep
 * their states on the device ({@code BatchSolverContext::statesToHost} off): after each solve call the handed-back
 * scenarios go to {@link #afterSolve}, which screens them there and keeps what the host will answer each with, and the full
 * state is read back ({@link #fullStates}) only for those that need one.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public interface StateHandback {

    /**
     * The scenarios the last solve call on {@code batchHandle} handed back, in the order of its result rows: their ids,
     * whether each converged, the branches each one's automation system switched (branch nums) and the taps its loops
     * moved (DeviceTapMoves' triples), null for none. Returns, per scenario, whether its full state must be read back.
     */
    boolean[] afterSolve(long batchHandle, String[] ids, boolean[] converged, int[][] automationSwitched, double[][] tapMoves);

    /** The full states ({@code n} doubles each) of the last call's handed-back scenarios {@code which}, indices in its
     *  result rows. */
    double[][] fullStates(long batchHandle, int[] which, int n);
}
