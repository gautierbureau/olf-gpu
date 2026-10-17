# Behaviour switches — the cleaning list

`docs/env-vars.md` catalogues everything readable from the environment, most of it TRACING. This file
lists only the switches that **change what the device computes**, so each one can be dispositioned.
The rule this exists to enforce: *once a path matches the CPU better, DELETE the other* — a switch
whose other mode is simply wrong is not a feature, it is an unexploded measurement.

Status legend: **DELETE** = one mode is known wrong, remove it and the dead path · **OPT-OUT** = a
correct fallback worth keeping for triage · **TUNING** = both modes legitimate (library/perf knob) ·
**VERIFY** = disposition not yet established by measurement.

## System properties (`-Dolf.gpu.*`)

| switch | default | gates | disposition |
|---|---|---|---|

## Environment variables that change behaviour

| switch | gates | disposition |
|---|---|---|
| `OLF_GPU_REORDER_ALG` | cuDSS fill-reducing ordering | **TUNING** — library knob, both modes legitimate. |
| `OLF_GPU_MATCHING_ALG` | cuDSS matching (standalone solve only) | **TUNING** — and `0` is a known-wrong mode kept deliberately to reproduce the failure. Document, do not delete. |
| `OLF_GPU_PIVOT_TYPE`, `OLF_GPU_PIVOT_THRESHOLD` | cuDSS pivoting on the robust path | **TUNING** |
| `OLF_GPU_PS_MASK`, `OLF_GPU_PSG_STEP` | pilot-self masks / step gating | **VERIFY** — bisection leftovers from the pilot-self work; likely DELETE now that the rotation is default-on. |
| `OLF_GPU_CPU_ROUTED` | force named contingencies to the CPU | **OPT-OUT** — the escape hatch for a bad contingency in production. |

## How to use this list

1. Anything marked **DELETE** should go with its dead path in one commit each, suite + the full branch
   scope as the gate — deleting a switch is a behaviour change like any other.
2. **VERIFY** rows need one measurement each, not a discussion: run both modes on the full branch
   scope and keep the one that matches the CPU.
3. The two `ziTreeRelocation` rows are the exception and should be left alone until the relocation is
   correct for control-capable buses — see `gpu-coupler-tree-relocation-enable` in the session memory.
   That one is a switch because *both* modes are wrong for some input, which is the only honest reason
   to have one.
