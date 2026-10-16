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
| `olf.gpu.sa.fixedInnerIters` | `0` | K fixed Newton steps instead of OLF's budget | **TUNING** — the fixed-iter bench needs it; never a production mode. |
| `olf.gpu.lib` | — | native library path | **TUNING** (plumbing, not behaviour). |

## Environment variables that change behaviour

| switch | gates | disposition |
|---|---|---|
| `OLF_GPU_IR_STEPS` | iterative-refinement step count (default 4 when zero-impedance) | **TUNING** — the default was decided by measurement (`gpu-ir2-default-after-diag-repair`); keep as an override, it is how that decision gets re-checked. |
| `OLF_GPU_REORDER_ALG` | cuDSS fill-reducing ordering | **TUNING** — library knob, both modes legitimate. |
| `OLF_GPU_MATCHING_ALG` | cuDSS matching (standalone solve only) | **TUNING** — and `0` is a known-wrong mode kept deliberately to reproduce the failure. Document, do not delete. |
| `OLF_GPU_PIVOT_TYPE`, `OLF_GPU_PIVOT_THRESHOLD` | cuDSS pivoting on the robust path | **TUNING** |
| `OLF_GPU_STRAGGLER_FRAC` | straggler-exit fraction | **VERIFY** — off by default; the user's 2026-09-22 call was that the device should report MAX_ITER itself. Decide and delete the loser. |
| `OLF_GPU_COMPACT_CACHE` | structure-cache compaction | **VERIFY** |
| `OLF_GPU_RL_SKIP` | skip named reactive-limit entries | **DELETE** — a debugging amputation, not a mode. |
| `OLF_GPU_FORCE_FIRST_STEP` | force a Newton step OLF's check said was unnecessary | **DELETE** — this is the spurious-step class (`gpu-sa-spurious-newton-step`); the correct behaviour is settled. |
| `OLF_GPU_HOST_LOOP_NOOP` | make a host outer loop do nothing | **DELETE** — bisection scaffolding. |
| `OLF_GPU_SA_BASE_ONLY` | solve the base and stop | **OPT-OUT** — genuinely useful for triage, keep. |
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
