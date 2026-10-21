#!/usr/bin/env bash
# LOCAL dev tool (see tools/README.md): build ONCE per cuDSS version and
# swap the whole native stack between versions for A/B testing.
#
#   source tools/cudss-swap.sh 0.7.1     # or 0.8.0
#
# What one swap does:
#   1. libolfgpu.so     — built against the version if not cached
#                         (native/cudss-cache/<ver>/), then installed into
#                         target/classes/natives/linux_64/ (what the tests load).
#   2. libmathcudss.so  — same, from ~/Projects/powsybl/powsybl-math-native-1
#                         (per-version cmake build dir, EMBED_RPATH=ON so it also
#                         loads standalone), installed into the tmp/m2
#                         powsybl-math-native jar.
#   3. exports CUDSS_ROOT and puts its lib directory first on LD_LIBRARY_PATH, so the
#      JVM's dynamic loader resolves the SAME version — that is why this must be
#      SOURCED, not executed.
#
# BOTH natives must always match the loaded libcudss.so.0 (same soname across
# versions, first-load wins process-wide) — that is exactly what this enforces.
# Both sources carry a CUDSS_VERSION shim (OLF_CUDSS_* / MATH_CUDSS_*), so either
# version compiles.

_cudss_swap() {
    local ver="$1"
    local repo; repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    local mathrepo="$HOME/Projects/powsybl/powsybl-math-native-1"
    local jar="$HOME/Projects/powsybl/tmp/m2/com/powsybl/powsybl-math-native/1.6.0-SNAPSHOT/powsybl-math-native-1.6.0-SNAPSHOT.jar"
    local root
    case "$ver" in
        0.7.1) root="$HOME/Projects/powsybl/cuda/cudss-0.7.1" ;;
        0.8.0) root="$HOME/Projects/powsybl/cuda/cudss" ;;
        *) echo "usage: source tools/cudss-swap.sh 0.7.1|0.8.0" >&2; return 1 ;;
    esac
    [ -e "$root/include/cudss.h" ] || { echo "error: no cudss.h under $root" >&2; return 1; }

    local cache="$repo/native/cudss-cache/$ver"
    mkdir -p "$cache"

    # 1. libolfgpu (build-gpu.sh wipes its cmake cache on a CUDSS_ROOT change)
    if [ ! -f "$cache/libolfgpu.so" ]; then
        echo ">> building libolfgpu vs cuDSS $ver"
        CUDSS_ROOT="$root" "$repo/native/build-gpu.sh" >/dev/null || return 1
        /bin/cp -f "$repo/target/classes/natives/linux_64/libolfgpu.so" "$cache/"
    fi
    mkdir -p "$repo/target/classes/natives/linux_64"
    /bin/cp -f "$cache/libolfgpu.so" "$repo/target/classes/natives/linux_64/libolfgpu.so"

    # 2. libmathcudss (per-version build dir; RPATH bakes this version's lib dir)
    if [ ! -f "$cache/libmathcudss.so" ]; then
        echo ">> building libmathcudss vs cuDSS $ver"
        local bdir="$mathrepo/build-cudss-${ver//./}"
        cmake -S "$mathrepo" -B "$bdir" -DWITH_CUDSS=ON -DCUDSS_ROOT="$root" \
              -DCUDAToolkit_ROOT="${CUDA_HOME:-$HOME/Projects/Software/cuda-toolkit}" \
              -DCUDSS_EMBED_RPATH=ON -DCMAKE_BUILD_TYPE=Release >/dev/null || return 1
        cmake --build "$bdir" --target mathcudss -j"$(nproc)" >/dev/null || return 1
        /bin/cp -f "$mathrepo/target/classes/natives/linux_64/libmathcudss.so" "$cache/"
    fi
    local stage; stage="$(mktemp -d)"
    mkdir -p "$stage/natives/linux_64"
    /bin/cp -f "$cache/libmathcudss.so" "$stage/natives/linux_64/"
    (cd "$stage" && jar uf "$jar" natives/linux_64/libmathcudss.so) || return 1
    rm -rf "$stage"

    # 3. the dynamic loader must resolve the same version
    export CUDSS_ROOT="$root"
    local cleaned
    cleaned="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/cuda/cudss' | paste -sd: -)"
    export LD_LIBRARY_PATH="$root/lib${cleaned:+:$cleaned}"
    echo "cuDSS $ver ACTIVE: libolfgpu + libmathcudss installed, CUDSS_ROOT=$root (first on LD_LIBRARY_PATH)"
    echo "  (libolfgpu -> target/classes/natives/linux_64, libmathcudss -> tmp/m2 jar)"
}

_cudss_swap "${1:-}"
