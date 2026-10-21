/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.util.gpu;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The environment is read only through the trace channels: {@link GpuTrace} on the Java side and
 * {@code native/cuda/common/trace.cuh} on the native side. A direct read elsewhere would be a switch nobody lists,
 * read at any time. No device needed.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuTraceTest {

    @Test
    void javaReadsTheEnvironmentOnlyThroughGpuTrace() throws IOException {
        assertEquals(List.of(), filesContaining(Path.of("src/main/java"), ".java", "System.getenv",
                Path.of("src/main/java/com/powsybl/openloadflow/util/gpu/GpuTrace.java")));
    }

    @Test
    void nativeCodeReadsTheEnvironmentOnlyThroughTheTraceHeader() throws IOException {
        assertEquals(List.of(), filesContaining(Path.of("native/cuda"), ".cu", "getenv(",
                Path.of("native/cuda/common/trace.cuh")));
    }

    /** The files under {@code root} whose name contains {@code extension} and whose text contains {@code text}. */
    private static List<Path> filesContaining(Path root, String extension, String text, Path allowed) throws IOException {
        try (Stream<Path> files = Files.walk(root)) {
            return files.filter(p -> p.getFileName().toString().contains(extension))
                    .filter(p -> !p.equals(allowed))
                    .filter(p -> {
                        try {
                            return Files.readString(p).contains(text);
                        } catch (IOException e) {
                            throw new java.io.UncheckedIOException(e);
                        }
                    })
                    .sorted()
                    .toList();
        }
    }
}
