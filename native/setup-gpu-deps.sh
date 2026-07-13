#!/usr/bin/env bash
# Copyright (c) 2026, RTE (http://www.rte-france.com) - SPDX-License-Identifier: MPL-2.0
#
# Bootstrap the GPU build dependencies for OLF on a fresh machine. Two artifacts the
# public powsybl releases don't carry must be installed into ~/.m2 before the GPU code
# compiles AND runs (this is the "MatrixFactory issue" — DenseMatrix/CuDssMatrixFactory
# both come from here):
#
#   1. powsybl-math 7.3.0-SNAPSHOT WITH com.powsybl.math.matrix.CuDssMatrixFactory (the
#      optional cuDSS matrix factory test6's MatrixFactoryUtil/OpenLoadFlowProvider use).
#      It is committed in the powsybl-core checkout's math module (branch
#      claude/cudss-lu-decomposition) — we just build+install that one module.
#   2. powsybl-math-native 1.5.0-SNAPSHOT whose jar bundles BOTH natives:
#        natives/<arch>/libmath.so       base KLU/dense JNI (DenseMatrix etc.; upstream)
#        natives/<arch>/libmathcudss.so  the cuDSS LU JNI (built from the math-native repo)
#      The math-native repo only builds libmathcudss.so, so we merge libmath.so in from
#      the newest RELEASED math-native jar already in ~/.m2 (its base JNI is ABI-stable).
#      Without libmath.so the math-native jar is incomplete and DenseMatrix fails to load
#      ("Couldn't load library library math") — which breaks essentially every test.
#
# After this: native/build-gpu.sh to build libolfgpu, then run the GPU tests.
#
#   POWSYBL_CORE  default ../powsybl-core  (math module has CuDssMatrixFactory)
#   MATH_NATIVE   default ../test7         (the cuDSS LU JNI math-native)
#   MVN_OFFLINE   set to 1 to pass -o (deps must already be cached)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CORE="${POWSYBL_CORE:-$ROOT/../powsybl-core}"
MATHNATIVE="${MATH_NATIVE:-$ROOT/../test7}"
M2="$HOME/.m2/repository/com/powsybl"
ARCH="linux_64"
OFF="$([ "${MVN_OFFLINE:-0}" = "1" ] && echo -o || true)"

for d in "$CORE/math/pom.xml" "$MATHNATIVE/pom.xml"; do
    [ -e "$d" ] || { echo "error: not found: $d (set POWSYBL_CORE / MATH_NATIVE)" >&2; exit 1; }
done

echo "[1/3] install powsybl-math (with CuDssMatrixFactory) from $CORE"
( cd "$CORE" && mvn $OFF -q install -pl math -DskipTests )

echo "[2/3] install powsybl-math-native (libmathcudss) from $MATHNATIVE, then ensure libmath.so is bundled"
( cd "$MATHNATIVE" && mvn $OFF -q install -DskipTests )
MN_JAR="$M2/powsybl-math-native/1.5.0-SNAPSHOT/powsybl-math-native-1.5.0-SNAPSHOT.jar"
if ! unzip -l "$MN_JAR" | grep -q "natives/$ARCH/libmath.so"; then
    echo "  libmath.so missing from the snapshot jar; merging it from the newest released math-native"
    BASE_JAR="$(for j in "$M2"/powsybl-math-native/*/powsybl-math-native-*.jar; do
        case "$j" in *-SNAPSHOT.jar) continue;; esac
        unzip -l "$j" 2>/dev/null | grep -q "natives/$ARCH/libmath.so" && echo "$j"
    done | tail -1)"
    [ -n "$BASE_JAR" ] || { echo "error: no released math-native with libmath.so in ~/.m2" >&2; exit 1; }
    TMP="$(mktemp -d)"
    ( cd "$TMP" && unzip -o "$BASE_JAR" "natives/$ARCH/libmath.so" >/dev/null && jar uf "$MN_JAR" "natives/$ARCH/libmath.so" )
    rm -rf "$TMP"
fi
echo "  math-native jar natives:" && unzip -l "$MN_JAR" | grep "natives/$ARCH/" || true

echo "[3/3] dependencies ready. Build the GPU library:  native/build-gpu.sh   then run the GPU tests."
