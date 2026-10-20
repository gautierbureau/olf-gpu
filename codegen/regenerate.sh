#!/usr/bin/env bash
# Copyright (c) 2026, RTE (http://www.rte-france.com) - SPDX-License-Identifier: MPL-2.0
#
# Regenerate the committed fused CUDA kernels + scatter tables from the
# hand-written OLF equation Java (the single source of truth). The parser
# (extract_golden.py flattening) and the proof are pure stdlib; Jinja renders
# the kernel template (pip install -r codegen/requirements.txt). Run after an
# upstream change to a formula method, then commit the regenerated files.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[1/3] prove (parse==golden, FD audit of the hand Jacobian, rendered==committed, rendered==golden)"
python3 codegen/java2cuda/prove.py

echo "[2/3] check scatter stamps vs OLF equation-system wiring"
python3 codegen/check_scatter.py

echo "[3/3] render: OLF Java -> native/cuda/equations/generated/*.cuh + scatter tables"
python3 codegen/java2cuda/java2cuda.py

echo
echo "Done. Review with: git diff"
