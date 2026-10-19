/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Every code family in this package has ONE definition per side: the Java class here and a header of the same name
 * in native/cuda/jni/common/codes. This test parses each header and checks it holds exactly the Java constants, so
 * the two sides cannot drift apart. No device needed.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuCodesMirrorTest {

    private static final Path JAVA_DIR = Path.of("src/main/java/com/powsybl/openloadflow/ac/solver/gpu/codes");
    private static final Path NATIVE_DIR = Path.of("native/cuda/jni/common/codes");
    private static final Pattern NAMESPACE = Pattern.compile("namespace (\\w+) \\{(.*?)\\n}", Pattern.DOTALL);
    private static final Pattern CONSTANT = Pattern.compile("constexpr int (\\w+) = (-?\\d+);");

    @Test
    void everyJavaFamilyHasTheSameNativeConstants() throws Exception {
        try (Stream<Path> javaFiles = Files.list(JAVA_DIR)) {
            for (Path javaFile : javaFiles.filter(p -> p.toString().endsWith(".java")).toList()) {
                String family = javaFile.getFileName().toString().replace(".java", "");
                assertEquals(javaConstants(family), nativeConstants(family), family);
            }
        }
    }

    @Test
    void everyNativeFamilyHasAJavaClass() throws IOException {
        try (Stream<Path> headers = Files.list(NATIVE_DIR)) {
            for (Path header : headers.toList()) {
                Matcher m = NAMESPACE.matcher(Files.readString(header));
                assertTrue(m.find(), header + " declares no namespace");
                assertTrue(Files.exists(JAVA_DIR.resolve(m.group(1) + ".java")), header + " has no Java class " + m.group(1));
            }
        }
    }

    private static Map<String, Integer> javaConstants(String family) throws Exception {
        Map<String, Integer> out = new TreeMap<>();
        for (Field f : Class.forName(GpuCodesMirrorTest.class.getPackageName() + "." + family).getDeclaredFields()) {
            int mod = f.getModifiers();
            if (Modifier.isPublic(mod) && Modifier.isStatic(mod) && Modifier.isFinal(mod) && f.getType() == int.class) {
                out.put(f.getName(), f.getInt(null));
            }
        }
        return out;
    }

    private static Map<String, Integer> nativeConstants(String family) throws IOException {
        String snake = family.replaceAll("([a-z0-9])([A-Z])", "$1_$2").toLowerCase();
        Matcher ns = NAMESPACE.matcher(Files.readString(NATIVE_DIR.resolve(snake + ".cuh")));
        assertTrue(ns.find(), family + ": no namespace in " + snake + ".cuh");
        assertEquals(family, ns.group(1), snake + ".cuh");
        Map<String, Integer> out = new TreeMap<>();
        Matcher c = CONSTANT.matcher(ns.group(2));
        while (c.find()) {
            out.put(c.group(1), Integer.parseInt(c.group(2)));
        }
        return out;
    }
}
