# Global number tracker

One row per bench-affecting commit — the single place to see where performance and
equivalence stand over time. Cold numbers first (the honest end-to-end story), warm second.
Append a row with `native/track-numbers.sh` after every change touching the batched GPU
path; never edit old rows (history is the point). Context/analysis for each row lives in
`codegen/gpu_benchmark.md` under the matching date; capability state in
`docs/gpu-vs-cpu-capabilities.md`.

Production bench = `GpuVsCpuProductionSaBenchmarkTest`: case9241pegase, 500 matched
contingencies, RTE JSON parameters, CPU arm 4 threads, sparse forced.
Fixed-iter bench = `native/cuda/run_bench.sh`: same case/contingencies, K=4, tol=0,
UBATCH chunk 500 (the gpusim2grid-comparable configuration).
"viol diff" = onlyCPU/onlyGPU violation-report differences (ctg = contingencies affected);
statuses have been identical on every run since 2026-07-17. The diff target is ZERO:
single-modelisation parts were validated at ~1e-13, so a nonzero diff means some
contingencies take a different outer-loop decision path on the device — a defect to
root-cause, not noise (the 22→20 shift from wiring maxOuterLoopIterations proved this).
"Caps" = GPU-covered capabilities / total tracked in docs/gpu-vs-cpu-capabilities.md
(counter: `python3 codegen/capability_count.py`; full+partial+delta over non-N/A rows).

## Headline — total run wall-clock (case9241pegase, 500 contingencies, seconds)

Total time to run all 500 contingencies (= the ms/ctg columns below × 0.5). GPU-arm
seconds match the raw millisecond totals the bench prints (16725 / 13926 / 11761 ms).

| Measurement (2026-07-21) | Arm & criterion | Cold | Warm | ×CPU (warm) |
|---|---|---|---|---|
| Equal-tol `856767c` | CPU @1e-7 | — | 16.66 | — |
| | GPU @1e-7 | 12.50 | **11.35** | ×1.47 |
| **Per-arm split `7ac0627`** | CPU @1e-4 *(production criterion)* | — | 16.73 | — |
| | GPU @1e-7 *(fast operating point)* | 13.93 | **11.76** | **×1.42** |

The CPU takes ~16.7 s at its production criterion; the GPU takes ~11.8 s warm to produce
a tighter-or-equal answer — ~5 s faster per full run. The two CPU rows (16.66 vs 16.73 s)
are the eps-insensitivity finding: loosening 1e-7 → 1e-4 buys the CPU essentially nothing,
so it cannot close the gap by relaxing its criterion. Equivalence (bit-exact 0/0/0
violations) is proven at equal tolerance; the split run is a timing comparison only.

| Date | Commit | Prod COLD ms/ctg | Prod warm ms/ctg | CPU-4t ms/ctg | Warm ×CPU | Viol diff (cpu/gpu/ctg) | FxIter COLD E2E ms/ctg | FxIter warm ms/scn | @1e-6 | Caps | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-17 | (perf day) | — | 24.4–25.2 | 27.0 | ~×1.10 | 15/7/6 | 11.56 | 4.31 | 499/500 | — | straggler exit + device norms + compaction landed; violation saga closed |
| 2026-07-18 | cc41597 | 25.8 | 23.1 | 25.8 | ×1.12 | 15/7/6 | 11.71 | 4.23 | 499/500 | — | full-row tap override + PST/shunt actions |
| 2026-07-18 | a082092 | 26.7 | 23.6 | 24.6 | ×1.04 | 15/7/6 | — | 4.27 | 499/500 | — | bare-run = production defaults (harness fix) |
| 2026-07-18 | 955017a | 30.4–30.6 | 26.9–27.8 | 23.1–23.5 | ×0.85 | 13/7/5 | — | 4.25 | 499/500 | — | outer-loop param wiring: RTE maxOuterLoopIterations=30 on device — diff 22→20, warm +1.5 s |
| 2026-07-18 | 26fb178 | 29.92 | 27.24 | 26.72 | x0.98 | 13/7/5 | 7.31 | 4.30 | 499/500 | — | tracker validation run |
| 2026-07-18 | (per-scn budget) | 43.83 | 41.22 | 27.19 | x0.66 | 0/5/2 | — | 4.26 | 499/500 | 91/124 | per-scenario outer budget fix: diff 20→5, onlyCPU=0; warm cost = open perf item |
| 2026-07-18 | (eps=1e-7) | 23.32 | 20.92 | 28.78 | x1.38 | **0/0/0** | — | — | — | 91/124 | ZERO diff, statuses identical — full-path equivalence proven; GPU faster at tight eps (outer churn gone) |
| 2026-07-19 | (#28 GenAction) | 27.01 | 22.04 | 30.27 | x1.37 | **0/0/0** | — | 4.32 | 499/500 | — | eps=1e-7 arms; GeneratorAction batched + DS base-shift + islanding injection remap |
| 2026-07-20 | 3044d14+dirty | _55.46_ | _53.76_ | _39.64_ | _x0.74_ | **0/0/0** | _18.04_ | _14.53_ | 499/500 | 98/129 (76%) | #33 node-breaker contingencies under propagation (zi outage + ISLANDS_SLACK reject). TIMINGS INVALID — box contended (two python sweeps + tar/zstd, load 3.8) and the GPU thermally throttled to 1297/2100 MHz; CPU arm degraded 30→40 ms/ctg too. Only the correctness columns (zero diff, 499/500) count on this row; re-time on an idle box. |
| 2026-07-21 | 856767c | 25.00 | 22.70 | 33.32 | x1.47 | 0/0/0 | 7.47 | 4.53 | 499/500 | 100/127 (79%) | re-time on idle box (supersedes the 2026-07-20 throttled row); +phase-control a1, SVC plain-LF, action-table corrections |
| 2026-07-21 | c3fd042 | 27.85 | 23.52 | 33.45 | x1.42 | 305=305 | — | — | — | 100/127 (79%) | **PER-ARM eps split** (OLF_GPU_SA_GPU_CONV_EPS=1e-7): CPU at the file's PRODUCTION 1e-4, GPU at its fast 1e-7 — same run, no cross-run drift. GPU delivers a production-or-better answer ×1.42 faster than CPU delivers its production answer, violation counts identical (305=305). KEY FINDING: CPU eps barely matters (1e-4 → 33.45 ≈ 1e-7 → 33.32), so the CPU cannot recover by loosening; the historical "CPU@1e-4 ≈ 25" was cross-run machine-state noise, not an eps effect. Viol column here is corroboration, NOT the equivalence proof (that stays the equal-tol 1e-7 row above). |
| 2026-09-06 | 62b6820 | 22.91 | 19.84 | 26.69 | x1.35 | 0/0/0 | 7.81 | 4.49 | 499/500 | 106/129 (82%) | shared-TVC member outage on-device (A5: drOv + nActive + r1Ov full tap rows + DS carry, machine + SimpleTvc); box semi-loaded (user IDEs indexing) — treat timing drift with care, correctness columns authoritative |
| 2026-09-06 | 13eb259 | 22.22 | 19.26 | 27.12 | x1.41 | 0/0/0 | 7.58 | 4.47 | 499/500 | 106/129 (82%) | remote-member RL pin dynamics on-device (B1-B5: rlRemote entries, mode-4 encoding, dynamic DISTR_Q keys + target compensation, member own-q pass); pegase has no remote groups so the new channels are inert there |
| 2026-09-07 | 6bb1fd3 | 22.99 | 19.43 | 26.66 | x1.37 | 0/0/0 | 7.76 | 4.46 | 499/500 | 106/129 (82%) | A6 free-member TVC outage relocation on-device (qrPtr/qrRows candidate tables + in-kernel mode-scan select in all 6 fill kernels, role-built machine rows, carrier drOv zeroing); pegase has no shared RTC groups so the reloc channel is inert there — this row measures the RelocTables param cost on the hot fill path |
| 2026-09-07 | d2f3150 | 23.19 | 20.03 | 27.63 | x1.38 | 0/0/0 | 7.76 | 4.45 | 499/500 | 106/129 (82%) | B4 free-member RL pin relocation on-device (qr groups on DISTR_Q pilots, carrier-row DISTR_Q suppression, pilot-Q capture/handoff, frozen qBuf group dests re-pointed); pegase has no remote gen-VC groups so the new channels are inert there — this row measures the extra fillDistrQBatch/updateDistrQTargets params on the hot path |
| 2026-09-07 | 1942084 | 21.73 | 19.38 | 27.32 | x1.41 | 0/0/0 | 7.57 | 4.43 | 499/500 | 106/129 (82%) | B4 JAVA lift (setBatchDqQrSuppression carrier-suppression tables in the hybrid; fillDistrQBatch dqJMember launch-arg condition) — inert on pegase (no remote gen-VC groups) |
| 2026-09-07 | e7699d6 | 24.47 | 20.13 | 26.95 | x1.34 | 0/0/0 | 7.50 | 4.38 | 499/500 | 106/129 (82%) | unclean-pinnable-member guard (extraction frozen-encoding condition + shared scope gate) — Java-side only, inert on pegase |
| 2026-09-07 | 8cc4233 | 21.64 | 19.19 | 27.10 | x1.41 | 0/0/0 | 7.56 | 4.44 | 499/500 | 106/129 (82%) | JAVA all-pinned corner + shared-TVC member outages (live/base row map, nActive follow, disabled-member free-pick fix) — Java-side, inert on pegase |
| 2026-09-07 | 6a05679 | 21.58 | 19.49 | 27.15 | x1.39 | 0/0/0 | 7.78 | 4.46 | 499/500 | 106/129 (82%) | DEVICE all-pinned corner (updatePilotVeqBatch pilot vEq flip) + forced-PV detector (per-RL-pass allPvPinnable-guarded rejection) — NOTE pegase is all-gens-limited so the detector's per-pass limitType copy IS active there |
| 2026-09-07 | faa9a69 | 21.36 | 19.03 | 28.11 | x1.48 | 0/0/0 | 7.71 | 4.51 | 499/500 | 106/129 (82%) | ob/zero-impedance alternate-dest channels (obQAlt/ziQAlt, dyn params on all ob/zi kernels) — pegase HAS ob+zi elements, so this row measures the added per-element branch on those hot fills (reloc itself inert: no remote groups) |
| 2026-09-07 | c9d1821 | 20.67 | 19.31 | 27.72 | x1.44 | 0/0/0 | 7.67 | 4.45 | 499/500 | 106/129 (82%) | sequential-base fixes (free-pick control-ON filter, seq fillDistrRho carrier suppression + drQrGroup param, device-broken safety net) — fillDistrRho param change is the only batch-path touch, inert on pegase (no shared groups) |
| 2026-09-07 | 6341937 | 20.81 | 18.55 | 26.64 | x1.44 | 0/0/0 | 7.58 | 4.55 | 499/500 | 106/129 (82%) | useInitialTapPosition shared-ratio port (SharedRatio remap + group clamp + COMPLETE polish solve; Java-side machine change, inert on pegase) |
| 2026-09-07 | 8cbd15f | 21.39 | 18.38 | 27.29 | x1.48 | 0/0/0 | 7.78 | 4.44 | 499/500 | 107/129 (83%) | INCREMENTAL shared TVC groups (MAIN-bus grouping + incremental-machine DS carry; Java-side, inert on pegase — no shared groups there) |
| 2026-09-07 | 6659254 | 21.10 | 19.06 | 27.55 | x1.45 | 0/0/0 | 7.64 | 4.45 | 499/500 | 107/129 (83%) | threadCount>1 data race fix (per-run SA state -> ThreadLocal RunState; both partitions now batch) |
| 2026-09-07 | c502347 | 20.43 | 18.77 | 28.22 | x1.50 | 0/0/0 | 7.72 | 4.46 | 499/500 | 113/129 (88%) | AUDIT CORRECTION, no code change: six stale capability marks flipped (SecondaryVc, DISTR_RHO outages, DS balance types, useActiveLimits, Simple TVC — all landed 2026-08/09 with 1e-12 tests); CAPS steps 107→113 |
| 2026-09-08 | edf53ab | 21.72 | 18.72 | 26.68 | x1.43 | 0/0/0 | 7.60 | 4.45 | 499/500 | 113/129 (88%) | DS-carry closed in ALL orchestrators + incremental-shunt mapper gap (loop had never run) + micro-threshold knife-edge characterized (1e-10 accepted) |
| 2026-09-08 | 4dbc014 | 22.15 | 19.35 | 26.93 | x1.39 | 0/0/0 | 7.55 | 4.43 | 499/500 | 117/129 (91%) | sequential solve = BATCH-OF-1 (per-eq criteria, ZIP, SHUNT_B, modes 2/3 free; base LF now on batch kernels; fillDistrRhoBatch in-kernel carrier suppression) |
| 2026-09-08 | 608a923 | 21.51 | 18.92 | 26.97 | x1.43 | 0/0/0 | 7.48 | 4.45 | 499/500 | 117/129 (91%) | LINE_SEARCH ported to the batch loop (per-scenario backtracking, fillActiveBatch lambda extraction); legacy seq dispatch retired from production |
| 2026-09-08 | 1c1e31c | 21.04 | 19.14 | 26.71 | x1.40 | 0/0/0 | 7.69 | 4.47 | 499/500 | 118/130 (91%) | N-k branch outages (caps lifted, N-3/N-4 at 1e-12) + monitoring-SVC base activation (cpuBaseOnly = frozen-HVDC only) |
| 2026-09-08 | 3e5bec0 | 21.98 | 19.57 | 26.64 | x1.36 | 0/0/0 | 7.57 | 4.44 | 499/500 | 119/130 (92%) | single-side openings + bus/busbar loss (SIDE_x = full-out per OLF semantics, bus loss = incident side-opens + islanding; pure-zi shape fix) |
| 2026-09-08 | 78c796f | 22.38 | 19.27 | 26.75 | x1.39 | 0/0/0 | 7.41 | 4.44 | 499/500 | 119/130 (92%) | vdl codegen family: generated evalVdl consumed (verbatim c,n,target stride 3) |
| 2026-09-08 | 51e4395 | 21.10 | 19.15 | 26.05 | x1.36 | 0/0/0 | 7.76 | 4.45 | 499/500 | 122/133 (92%) | RL parity: slope pinned-stays-PQ + live SVC B*v^2 limits (setBatchRlSvcLimits) |
| 2026-09-08 | 26d3dfa | 20.76 | 18.74 | 26.74 | x1.43 | 0/0/0 | 7.58 | 4.44 | 499/500 | 122/133 (92%) | continuous shunt VC batched (GpuBatchSimpleShuntVc, WITH_GENERATOR default mode) |
