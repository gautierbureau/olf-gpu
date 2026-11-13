# GPU security analysis on network.xiidm: where the warm time goes, and the leads

**Status**: measured 2026-11-10 on `main` at 031d7b31 with our batched LU (`linearSolver: BATCHED_LU`), GPU arm, warm
run, one host thread. This document keeps every number of that profile and the leads that are **not** in the state
copy-back plan (`docs/device-final-results-plan.md`, branch `device-final-results`, phases P0–P5): that plan removes the
write-back of the device's states into OLF's objects and the states' round trip through the host. Everything else
is here.

The costs below are measured. **No gain is measured**: each lever has to be measured A/B (alternating, cold and warm)
and, when it touches the solve, pass the parity gates (`GpuSa*Test` on both solvers, rte6515's generator and branch
arms, the 1-thread network.xiidm run).

## 1. Setup

| | |
|---|---|
| Machine | NVIDIA H100 PCIe 80 GB (sm_90), 30 vCPU AMD EPYC 9354; JVM max heap 30,688 MB |
| Code | GPU repo `main` 031d7b31, native library rebuilt for sm_90; OLF `olf-local-patches` |
| Linear solver | our batched LU, selected through the config module: `-Dpowsybl.config.dirs=<dir>` whose `config.yml` holds `open-loadflow-gpu-default-parameters: linearSolver: BATCHED_LU` |
| Harness | `GpuVsCpuRte6515SaTest#gpuSaMatchesCpuOnRte6515Full` with `OLF_GPU_SA_CASE_FILE=network.xiidm OLF_GPU_SA_GPU_ONLY=1` and the RTE parameters. The CPU arm runs on an empty list, so the test fails by design; the GPU arms run cold, then warm, in one JVM |

The three scopes:

| | every branch | the JSON list | the full list |
|---|---|---|---|
| How | `OLF_GPU_SA_CTG_FILE=<every branch, N-1>.json` (kept outside the repo) | `OLF_GPU_SA_CTG_FILE=network-contingencies.json` | `OLF_GPU_SA_CTG_KINDS=branch,generator,load,hvdc,svc,shunt,battery,busbar OLF_GPU_SA_GEN_MIN_KV=0 OLF_GPU_SA_LOAD_TOP=1000000 OLF_GPU_SA_BUSBAR_TOP=1000000` |
| Contingencies | 10,540 | 14,266 (1,066 busbar sections) | 38,441: 10,540 branches, 6,071 generators, 7,495 loads, 6 HVDC, 6 SVC, 240 shunts, 21 batteries, 14,062 busbar sections (866 outside the solved network) |
| Batched / answered on the device | 10,540 / 10,538 | 12,732 / 12,726 | 37,297 / 37,282 |
| GPU statuses | CONVERGED 10,538, MAX_ITERATION_REACHED 2 | CONVERGED 12,726, FAILED 3, MAX_ITERATION_REACHED 3 | CONVERGED 37,282, FAILED 6, MAX_ITERATION_REACHED 9 |
| LfNetwork (buses / branches of the limit table) | 7,745 / 10,923 | 8,907 / 12,119 | 12,448 / 15,779 |
| State length n | 16,256 | 20,972 | 35,374 |
| Bank (slots) / native solve calls in the warm run | 3,514 / 8 | 3,183 / 17 | 1,777 / 109 |

The contingency list sets the retained switches, so the three scopes solve three different networks: the full list
retains every busbar section's switches.

Instruments (the recipes are in §8):
- **Phase timers** (`OLF_GPU_PHASE_TIME=1 OLF_SA_PROFILE=1`): warm 12.2 / 22.1 / 129.9 s against 12.7 / 21.4 / 135.0 s
  untraced (machine noise is of that order). The native site timers synchronize around our LU's calls.
- **Nsight Systems** 2026.1.3, CUDA trace only: device time per kernel and copy, idle gaps, CUDA API calls. It inflates
  the host's API costs (warm 13.7 / 25.7 / 131.7 s).
- **Java Flight Recorder** on cold-only runs of the full and JSON lists, 10 ms sampling: Java CPU by phase and method.

## 2. Headline

| | every branch | the JSON list | the full list |
|---|---|---|---|
| GPU SA cold / warm (untraced) | 14.8 / 12.7 s | 26.7 / 21.4 s | 147.6 / 135.0 s |
| Warm end to end (+ ~1.6 s network read) | 14.3 s | 23.0 s | 136.6 s |
| GPU warm per contingency | 1.20 ms | 1.50 ms | 3.51 ms |
| CPU OLF on 20 threads, cold / warm (timing only) | 174.0 / 170.8 s | 215.8 / 214.2 s | 684.1 / 667.2 s |
| CPU warm per contingency | 16.2 ms | 15.0 ms | 17.4 ms |
| GPU warm in the same JVM as the 20-thread arm, and the ratio | 13.1 s, ×13.1 | 21.1 s, ×10.2 | 135.2 s, ×4.9 |

A 20-thread CPU arm retains other switches per partition, so it is a timing reference only; correctness is checked at
one thread.

## 3. Where the warm time goes

Phase timers, warm arm, milliseconds.

| | every branch | the JSON list | the full list |
|---|---|---|---|
| **Base load flow + setup** | 668 | 834 | 1,367 |
| **Planning before the batch** (precompute minus its batch calls; device idle) | 878 | 1,475 | 8,044 |
| **Batch calls** (main batch + robust retry) | 5,937 | 11,595 | 85,979 |
| · device Newton loops | 3,518 | 8,779 | 68,253 |
| · native, outside the Newton loops (installs, handback) | 914 | 1,229 | 5,939 |
| · marshal (Java inputs into native) | 494 | 705 | 2,798 |
| · pack (states into Java rows) | 435 | 399 | 1,361 |
| · Java between the native calls | 576 | 483 | 7,628 |
| **Per-contingency loop** | 3,776 | 7,411 | 31,942 |
| · toLfContingency | 295 | 620 | 1,903 |
| · apply | 362 | 514 | 1,788 |
| · loss distribution | 17 | 30 | 98 |
| · post-contingency simulation | 2,343 | 5,024 | 23,790 |
| ·· state injection + network update (injectState + updateNetwork) | 1,896 (1,363 + 532) | 3,755 (2,981 + 770) | 15,019 (12,083 + 2,933) |
| ·· violation screen, every scenario at once | 229 | 333 | 2,729 |
| ·· violation detection, host confirm | 170 | 201 | 601 |
| ·· CPU replay of the device's rejects | — | 660 (3 contingencies) | 2,994 (6) |
| ·· rest (full list: the file-backed state store's reads) | 48 | 75 | 2,447 |
| · networkState.restore | 759 | 1,223 | 4,363 |
| **Outside the provider's timers** (LfNetwork load, propagation, results, GC; full list: the state file's delete) | 938 | 746 | 2,607 |
| **Warm SA** | 12,197 | 22,061 | 129,939 |

What the state copy-back plan removes, from these numbers:

| | every branch | the JSON list | the full list |
|---|---|---|---|
| State injection + network update (P1) | 1.90 s | 3.76 s | 15.02 s |
| Apply + restore (P2) | 1.12 s | 1.74 s | 6.15 s |
| The screen's state upload and run (P4) | 0.23 s | 0.33 s | 2.73 s |
| State handback in the batch: host buffer zero-fill, gather, copy, pack (P4; §4.3) | ~1.1 s | ~1.05 s | ~6.3 s |
| File-backed state store: writes, reads, delete (P4; full list only) | — | — | ~2.0 + 2.45 + 0.8 s |
| **About** | **4.4 s of 12.2** | **6.9 s of 22.1** | **35 s of 130** |

## 4. The batch

### 4.1 Newton work (phase counters, warm)

| | every branch | the JSON list | the full list |
|---|---|---|---|
| Batched Newton iterations | 53 | 83 | 261 |
| Lane iterations (factorized width) | 100,454 | 122,020 | 345,137 |
| Mean factorized width | 1,895 | 1,470 | 1,322 |
| Solves restarted by a check | 87,943 | 106,958 | 303,219 |
| Factorizations of our LU | 51 | 74 | 228 |
| Lane factorizations on the pivoting engine | 2 | 155 | 345,055 |
| Pivot-delay orders built | 2 | 14 | 4 |
| Refinement checks / extra solves | 168 / 117 | 289 / 215 | 1,004 / 776 |
| Outer-loop check rounds | 261 | 355 | 1,482 |

Our LU's calls, timed with a synchronization around each (`CUDSS_PHASE_SITE` lines under `OLF_GPU_PHASE_TIME`):

| | every branch | the JSON list | the full list |
|---|---|---|---|
| Factorization | 51 × 14.5 ms = 742 ms | 74 × 33.9 = 2,510 ms | 228 × 112.2 = 25,575 ms |
| Solve calls: Newton solves + first corrections | 168 × 4.03 = 678 ms | 289 × 6.68 = 1,930 ms | 1,004 × 18.44 = 18,510 ms |
| Internal-step solves of the corrections | 77 × 2.69 = 207 ms | 148 × 5.34 = 791 ms | 554 × 14.53 = 8,050 ms |
| Singleton substitution | 51 × 1.76 = 90 ms | 74 × 4.14 = 306 ms | 228 × 10.52 = 2,398 ms |
| Solve calls per factorization | 4.8 | 5.9 | 6.8 |
| Refinement's share of the solve time | 77 % | 82 % | 84 % |
| Robust retry on cuDSS (factorizations) | — | 18 × 6.4 = 115 ms | 67 × 12.7 = 853 ms |

The outer loops (`GPU_OL_TIME`, warm): scenario solves / lane iterations.

| | every branch | the JSON list | the full list |
|---|---|---|---|
| First solves | 10,541 / 12,407 | 12,736 / 14,907 | 37,304 / 41,663 |
| Re-solves after DistributedSlack | 1,949 / 1,949 | 6,680 / 6,680 | 22,096 / 22,096 |
| Re-solves after FreezingHvdcACEmulation | 10,538 / 10,546 | 12,732 / 12,749 | 37,293 / 37,334 |
| Re-solves after ReactiveLimits | 75,456 / 75,552 | 87,546 / 87,684 | 243,830 / 244,044 |
| **ReactiveLimits' share of the lane iterations** | **75.2 %** | **71.9 %** | **70.7 %** |
| Checks, ms: DistributedSlack / HVDC freeze / AC-emulation limits / ReactiveLimits | 59 / 28 / 24 / 82 | 100 / 49 / 42 / 127 | 455 / 249 / 208 / 571 |

### 4.2 Device time (Nsight, warm batch window)

The window runs from the batch's first device activity to the screen; on the branch list it also holds the warm base
load flow and the planning gap (less than 1 s apart there). Seconds.

| | every branch | the JSON list | the full list |
|---|---|---|---|
| **Window / device busy** | **7.99 / 3.54 (44 %)** | **13.78 / 8.25 (60 %)** | **89.51 / 67.91 (76 %)** |
| LU factorization | 0.68 | 1.90 (pivoting engine 0.09) | 22.56 (pivoting engine 21.05) |
| LU triangular solves | 0.74 | 2.29 | 25.59 (pivoting engine 23.21) |
| Refinement: residual and correction kernels | 0.40 | 0.90 | 4.40 |
| Singleton substitution | 0.08 | 0.28 | 2.33 |
| Jacobian + mismatch fill | 0.78 | 1.16 | 4.86 |
| Newton bookkeeping (row modes, identity lanes, norms, step) | 0.19 | 0.40 | 1.63 |
| Outer-loop checks and other kernels | 0.09 | 0.14 | 0.58 |
| Copies DtoD / DtoH / HtoD | 0.19 / 0.17 / 0.12 | 0.24 / 0.43 / 0.19 | 0.71 / 3.13 / 0.75 |
| Memsets | 0.06 | 0.14 | 0.34 |
| cuDSS (the robust retry) | 0.05 | 0.17 | 0.95 |

The heaviest kernels (launches, seconds, mean µs per launch):

| every branch | the JSON list | the full list |
|---|---|---|
| factorStep 7,136, 0.554, 78 | factorStep 17,136, 1.623, 95 | pivoting backwardLevel 55,722, 13.666, 245 |
| levelSplit 77,123, 0.474, 6.1 | levelSplit 179,699, 1.305, 7.3 | factorBlocks 8,029, 10.106, 1,259 |
| fillClosedBranchBatch 51, 0.276, 5,404 | backwardLevel 13,569, 0.555, 41 | forwardLevelSplit 52,378, 8.149, 156 |
| residualInterleavedBatch 237, 0.210, 887 | residualInterleavedBatch 406, 0.461, 1,134 | gatherLevel 8,029, 6.144, 765 |
| axpyLaneMaskBatch 241, 0.143, 592 | axpyLaneMaskBatch 474, 0.329, 694 | applyPanels 7,812, 4.800, 614 |
| backwardLevel 8,786, 0.139, 16 | fillClosedBranchBatch 69, 0.318, 4,610 | residualInterleavedBatch 1,506, 2.522, 1,674 |
| factorStepSplit 11,581, 0.119, 10 | factorStepSplit 13,167, 0.191, 15 | levelSplit 219,234, 1.512, 6.9 |
| fillClosedBranchBatchF 51, 0.114, 2,238 | forwardLevelSplit 1,276, 0.165, 129 | pivoting forwardLevel 3,344, 1.395, 417 |
| zeroActiveInterleavedValues 51, 0.106, 2,080 | forwardLevel 2,598, 0.163, 63 | axpyLaneMaskBatch 1,782, 1.188, 667 |

On the full list the fast engine still costs 1.51 s of factorization (factorStep 0.76, factorStepSplit 0.75) and 1.65 s
of solves (levelSplit 1.51, its forwardLevel and backwardLevel 0.13), for the 82 lane factorizations it keeps.

Triangular-solve kernels under 10 µs: 79,682 of 87,764 (every branch), 171,155 of 198,089 (the JSON list), 199,064 of
342,138 (the full list). A factorization launches two kernels per level of the elimination tree, a triangular solve one
per level and direction: about 370–410 launches per factorization and 350–450 per solve.

### 4.3 The device idle inside the batch

Idle gaps of the warm window by length:

| | every branch | the JSON list | the full list |
|---|---|---|---|
| Under 10 µs (between dependent kernels) | 270,817 gaps, 0.68 s | 459,228, 1.02 s | 1,204,402, 2.40 s |
| 10 µs to 1 ms | 7,792, 0.21 s | 7,266, 0.26 s | 18,578, 0.97 s |
| 1 to 100 ms | 82, 0.99 s | 173, 2.64 s | 516, 16.78 s |
| Over 100 ms | 12, 2.57 s (planning 0.69 s included) | 8, 1.61 s | 8, 1.47 s |

The idle gaps of 1 ms or more from the warm base load flow on, by the device activity around each (the host makes
almost no CUDA call during them: it computes):

| | every branch | the JSON list | the full list |
|---|---|---|---|
| Planning before the batch | 0.69 s | 2.16 s | 8.20 s |
| State handback: next to a state gather or a DtoH of 10 MiB or more (P4) | 1.24 s (12 gaps) | 1.67 s (29) | 6.57 s (212) |
| Refill, Java side: small readbacks before, install uploads after | 1.07 s (49) | 2.08 s (122) | 8.94 s (211) |
| Refill, native install: between install uploads | 0.41 s (21) | 0.56 s (28) | 2.66 s (108) |
| Other | — | 0.16 s (14) | 0.36 s (8) |

On the full list each refill idles the device ~110–140 ms: ~30–42 ms zero-filling the host buffer before the state
gather (`res.x.resize` in the driver's handback), ~16–22 ms of JNI pack after the copy, ~40–55 ms of Java (results,
state-store writes, the next scenarios' inputs, marshal), ~18–22 ms of native install between `scatterSlotRows`
uploads. On the branch list the whole-bank handback shows as ~220 ms before each 436 MB copy and 100–180 ms after it.

### 4.4 Copies and CUDA calls (warm batch window)

| | every branch | the JSON list | the full list |
|---|---|---|---|
| DtoD copies (each a scenario's state, n doubles) | 105,514, 13.2 GiB, 0.19 s | 129,336, 20.7 GiB, 0.24 s | 376,386, 99.7 GiB, 0.71 s |
| DtoH copies | 22,213, 1.60 GiB, 0.17 s | 27,161, 2.19 GiB, 0.43 s | 81,077, 11.16 GiB, 3.13 s |
| · of which strided 2D lane reads (pivot-delay learning) | 2 | 109 | 549, 2.36 s at 0.2 GB/s |
| · of which per-slot synchronous reads of 8 and 1,664 bytes | 21,342 | 25,954 | 76,182 |
| HtoD copies | 1,757, 1.97 GiB, 0.12 s | 2,724, 3.26 GiB, 0.19 s | 13,468, 15.06 GiB, 0.75 s |
| `cudaMemcpyAsync` calls, host time | 106,112, 1.36 s | 130,313, 2.97 s | 380,172, 6.11 s |
| `cudaMemcpy` calls, host time | 23,374, 0.54 s | 28,799, 0.69 s | 90,210, 2.17 s |
| `cudaMemcpy2DAsync` calls, host time | 2, 0.02 s | 109, 0.23 s | 549, 2.44 s |
| `cudaMemsetAsync` calls, host time | 36,208, 0.07 s | 68,558, 0.15 s | 261,355, 0.41 s |
| `cudaLaunchKernel` calls, host time | 112,373, 0.39 s | 238,173, 0.78 s | 489,850, 1.38 s |
| `cudaStreamSynchronize` calls, host waiting | 1,114, 1.88 s | 1,829, 4.41 s | 7,733, 57.76 s |

## 5. The host side (Java Flight Recorder, cold arms)

Java CPU (samples × 10 ms), each sample charged to its outermost phase frame.

| | the full list (cold 134.4 s) | the JSON list (cold 27.2 s) |
|---|---|---|
| Per-contingency loop | 15.73 s: state injector 7.82, `updateBusesFromState` 2.69, restore 2.63, CPU replays 1.34, apply 0.64, detection 0.43 | 4.35 s: state injector 2.09, `updateBusesFromState` 0.71, restore 0.65, CPU replays 0.40, apply 0.27 |
| Precompute (planning + the batch's Java) | 7.41 s: **`ZeroImpedanceTreeReplay` 6.09** (`HashMap.getNode` 4.88 under `nonTree()`'s filter), chunk solver 0.63, mapper and the rest 0.69 | 0.93 s: **`ZeroImpedanceTreeReplay` 0.41**, `GpuBatchPrecomputer` 0.23, chunk solver 0.17, retry 0.05, extraction 0.04 |
| The loop's toLfContingency and the rest | 2.36 s | 0.36 s (+ 0.52 s other, the base load flow's Java 0.31) |
| LfNetwork load / propagation | 0.30 / 0.05 s | 0.28 / 0.02 s |

Native samples of the full list's SA thread: the batch's native solve call, then the file-backed state store (≈ 2.5 s
of `pread`, ≈ 2.0 s of `pwrite`, ≈ 0.8 s of `unlink` at the end of the run); GC paused 1.03 s over 34 collections. The
JSON list's states stay in the heap: no file I/O.

## 6. The leads not in the state copy-back plan

Costs are warm, per scope (every branch / the JSON list / the full list).

### L1 — Refinement takes about half of the LU — 1.0 / 2.8 / 25.9 s
- 4.8 / 5.9 / 6.8 solve calls per factorization; the refinement's solves are 77 / 82 / 84 % of the triangular-solve
  time (§4.1). A solve on the refining lanes costs nearly as much as a full one (5.34 against 6.68 ms per call on the
  JSON list): the levels are latency-bound (L2).
- The residual and correction kernels (`residualInterleavedBatch`, `axpyLaneMaskBatch`) are launched over n × all
  slots (54–67 million threads) with a lane mask: 0.40 / 0.90 / 4.40 s.
- Levers: launch the residual and the correction on the refining lanes only; find why so many lanes stay above 1e-12
  after the first solve. The refinement is there for exactness on coupler systems: every change goes through the parity
  gates.

### L2 — Launch-bound levels — device idle 0.7 / 1.0 / 2.4 s between kernels, plus the latency in every LU call
- One kernel per elimination step (two per level) and per triangular-solve level and direction; most solve kernels run
  under 10 µs (§4.2), and 270,817 / 459,228 / 1,204,402 gaps under 10 µs separate them.
- Levers: fuse the narrow top of the elimination tree into one kernel per solve and per factorization (the lanes are
  independent, so a block can walk the levels with block-level synchronization), or capture the level sequence in a
  CUDA graph. This also makes the refinement's few-lane solves cheap (L1).

### L3 — The pivoting engine on the full list — 44.3 s
- 345,055 of 345,137 lane factorizations. Factorization 21.05 s (`factorBlocks` 10.11 s at 1.26 ms per launch,
  `gatherLevel` 6.14, `applyPanels` 4.80); solves 23.21 s, of which the backward solve 13.67 s at 245 µs per launch, one
  thread per (supernode, lane) walking its rows — 6de554ae split the forward solve's long rows over warps, not the
  backward solve's.
- Quick: split the backward solve, and `factorBlocks`, over warps as the forward solve is.
- Structural: an order the fast engine keeps on busbar systems. The learned pivot delays do not hold there
  (f5f04199's probe: 83,333 pivoted lanes pivoted again, 360 went back to the fast engine).

### L4 — The fast engine's work on lanes it flags, and learning that never stops — about 3.2 + 2.4 s on the full list
- The fast engine factorizes every new lane before flagging it, and `solveLanes` / `solveSelectedLanes` still solve the
  flagged lanes, whose x the pivoting engine then overwrites: 1.51 s of factorization and 1.65 s of solves on the full
  list, for 82 lane factorizations it keeps. Leaving the flagged lanes out of the fast solves changes no result.
- `learnFromUntrustedLanes` reads two flagged lanes back at every factorization that flags one: their factors, and their
  Jacobian values through a strided `cudaMemcpy2DAsync` (one double per row, 0.2 GB/s) — 549 copies, 2.36 s of copy
  time on the full list; 109 copies, 0.23 s on the JSON list. `PIVOT_DELAY_MAX_ORDERS` counts published orders only, and
  a round that learns nothing publishes nothing, so the learning never ends. Stop after a round that changes nothing;
  gather the lane on the device before copying it.

### L5 — Planning before the batch — 0.88 / 1.48 / 8.04 s, the device idle
- On the full list 6.09 s of Java CPU is `ZeroImpedanceTreeReplay` (0.41 s on the JSON list): `nonTree()` streams over
  every zero-impedance branch of the network and looks each up in a `HashSet`, once per contingency (and per strategy
  pair), including the contingencies that touch no coupler and those OLF never applies. The replay also builds its own `LfContingency`
  per contingency (0.77 s of the full list's cold Java CPU).
- Levers: keep the non-tree list and recompute it only when the replay changes the tree; index by branch number
  instead of hashing. Quick and contained.

### L6 — The refill pauses beyond the handback — 1.5 / 2.6 / 11.6 s of device idle (Nsight)
- Java side 1.07 / 2.08 / 8.94 s and native install 0.41 / 0.56 / 2.66 s (§4.3); 8 / 17 / 109 native solve calls per
  warm run. Inside: marshal 0.49 / 0.71 / 2.80 s, Java between the calls 0.58 / 0.48 / 7.63 s (of which ~2 s of
  state-store writes on the full list, P4), the per-slot calls of L7, GC.
- Levers: build the next scenarios' inputs while the device solves, and install them with one upload and one scatter;
  measure `REFILL_SHARE = 8` again now that a refill's cost is known.

### L7 — Per-scenario CUDA calls — their host time under Nsight 2.0 / 3.8 / 8.7 s (less untraced)
- One `cudaMemcpyAsync` per converged solve (saving the state to `dXLastConv`) plus the per-slot install copies:
  105,514 / 129,336 / 376,386 device-to-device copies per warm batch; per-slot synchronous reads (`readInstalledRows`):
  21,342 / 25,954 / 76,182; memsets 36,850 / 69,283 / 262,724.
- Lever: one gather or scatter kernel per iteration or install, and readbacks in one copy.

### L8 — Fill and Newton bookkeeping — 1.0 / 1.6 / 6.5 s of device time
- `zeroActiveInterleavedValues` (every Jacobian value zeroed each iteration) 2.1–2.7 ms per iteration, the closed-branch
  fill 2.6–5.4 ms per iteration; bookkeeping kernels (`writeIdentityForInactive`, `applyRowModeBatch`,
  `negateIntoBatch`, ...) launched over n × all slots.
- Levers: the parked gather-style fill (each Jacobian entry sums its own terms: no zeroing, no scatter), launches over
  the active lanes only.

### L9 — The device's rejects — 0 / 0.87 / 4.01 s
- The robust retry batch on cuDSS (0.21 / 1.02 s) recovers none of them (`RETRY:robust-recovered=0`), then OLF replays
  them on the CPU (0.66 s for 3, 2.99 s for 6) and reports `FAILED`. Already listed under "Not in this plan" in the plan:
  answer `FAILED` from the device once the reason is shown to be OLF's (`OLF_GPU_REJECT_TRACE`).

### L10 — The loop's toLfContingency — 0.30 / 0.62 / 1.90 s
- The tree replay (L5) already builds an `LfContingency` per contingency during planning: reuse it, or the
  precompute's propagation.

### L11 — Common to both arms
- Outside the provider's timers 0.94 / 0.75 / 2.61 s (LfNetwork load 0.3 s, propagation under 0.1 s, results, GC; the
  full list's state-file delete belongs to P4) and the ~1.6 s network read, both arms alike.

### L12 — The Newton work itself — 75.2 / 71.9 / 70.7 % of the lane iterations are ReactiveLimits re-solves
- #84: the three generators of one remote voltage control pin and release in turn up to `maxPqPvSwitch`, because OLF's
  switch-back test is a strict comparison 1.05e-7 pu away. OLF is the reference, so the device cannot skip them; the
  upstream issue #84 prepared would cut most of this work on both arms.

## 7. Where to start

- **The full list**: L5, L4 and the backward split of L3 are quick and contained and target over 11 s of measured
  cost (6.1 + 3.2 + 2.4 + part of 13.7 s); then L1–L2 (refinement and latency, ~26 s of LU) and L6 (~12 s of idle).
- **The JSON list and the branches**: L1–L2 (2.8 and 1.0 s of LU in refinement, plus the latency) and L6 (2.6 and
  1.5 s of idle) are the largest; L8 next.

## 8. Method

- **Phase timers.** Run the harness with `OLF_GPU_PHASE_TIME=1 OLF_SA_PROFILE=1`. `SA_PHASE_TIME`, `SA_PROFILE`,
  `JNI_PHASE_TIME`, `CUDSS_PHASE_SITE` and `GPU_OL_*` are cumulative over the JVM: the warm arm is the last print minus
  the last print before the harness's `GPU CENSUS (cold)` line. `BATCH_PHASE_TIME total` is per batch call. The planning
  is the precompute minus its batch calls; "outside the provider's timers" is the arm's wall time minus base, precompute
  and the five `OLF_SA_PROFILE` timers.
- **Nsight.** `nsys profile --trace=cuda --sample=none --cpuctxsw=none -o <out> mvn test ...` (no phase timers), then
  `nsys stats --force-export=true` (export to sqlite; force it when a stale sqlite is there). Device activity separated
  by more than 1 s of idle splits the run into windows: cold base load flow, cold batch, warm base load flow, warm batch
  (on the branch list the base load flow and the batch are less than 1 s apart). Idle gaps are classified by the device
  activity on either side: a state gather or a DtoH of 10 MiB or more → handback; small readbacks before and install
  uploads after → refill, Java side; between `scatterSlotRows` uploads → install. The two LU engines share kernel names
  (`forwardLevel`, `backwardLevel`, `toInterleaved`, `fromInterleaved`): tell them apart by their signature in
  `demangledName` (the pivoting engine's take 13–14 parameters, the fast engine's 10).
- **Java Flight Recorder.** `OLF_GPU_SA_COLD_ONLY=1 OLF_GPU_SA_JFR=<file> OLF_GPU_SA_JFR_PERIOD_MS=10`; `jfr print --json
  --events jdk.ExecutionSample` (and `jdk.NativeMethodSample`). Java samples × 10 ms is host CPU. Charge each sample to
  its **outermost** phase frame (`GpuBatchPrecomputer.precompute`, `AbstractSecurityAnalysis.processContingency`, ...):
  charging the innermost files the tree replay's own `toLfContingency` under the loop's.
- **Logs.** The harness's comparison lines name network elements: read the logs through the timing prefixes above only.
