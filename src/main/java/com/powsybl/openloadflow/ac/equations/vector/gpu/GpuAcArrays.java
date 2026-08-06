/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import java.util.List;

/**
 * Small flattening helpers shared by the GPU extraction: pack a list of fixed-stride rows into
 * one contiguous primitive array (the layout the JNI structure expects), and unbox
 * {@code List<Integer>}/{@code List<Double>} accumulators to {@code int[]}/{@code double[]}.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuAcArrays {

    private GpuAcArrays() {
    }

    static double[] flatD(List<double[]> rows, int stride) {
        double[] out = new double[rows.size() * stride];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, out, i * stride, stride);
        }
        return out;
    }

    static int[] flatI(List<int[]> rows, int stride) {
        int[] out = new int[rows.size() * stride];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, out, i * stride, stride);
        }
        return out;
    }

    static int[] toIntArray(List<Integer> list) {
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }

    static double[] toDoubleArray(List<Double> list) {
        double[] out = new double[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }
}
