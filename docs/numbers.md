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
| 2026-09-08 | c4430d2 | 21.52 | 18.76 | 26.50 | x1.41 | 0/0/0 | 7.63 | 4.44 | 499/500 | 122/133 (92%) | saturated-base HVDC batched + last cpu-base-only condition retired |
| 2026-09-08 | 02534ac | 22.17 | 19.02 | 26.76 | x1.41 | 0/0/0 | 7.60 | 4.46 | 499/500 | 122/133 (92%) | SecondaryVc remote gen VC + remote pin-dynamics gate retired (tier A 3+4) |
| 2026-09-08 | f76daae+dirty | 23.56 | 18.93 | 27.49 | x1.45 | 0/0/0 | 7.79 | 4.46 | 499/500 | 122/133 (92%) | curve-limit extrapolation batched (tier B item 7) |
| 2026-09-08 | 24b7ecf | 21.28 | 18.97 | 26.13 | x1.38 | 0/0/0 | 7.45 | 4.47 | 499/500 | 122/133 (92%) | pilot-is-controller DISTR_Q fix + scope census + chunk shrink-retry |
| 2026-09-08 | ae58f89 | 24.45 | 19.69 | 26.14 | x1.33 | 0/0/0 | 7.56 | 4.44 | 499/500 | 122/133 (92%) | chunk sizing from cuDSS memory estimates (no shrink-retry on rte6515 @2000) |
| 2026-09-08 | b927cc2 | 24.75 | 20.10 | 28.39 | x1.41 | 0/0/0 | 7.57 | 4.47 | 499/500 | 122/133 (92%) | quadratic DS-seed merge fixed (rte6515 SA 563s -> 49.6s) |
| 2026-09-08 | 6ddfebf+dirty | 25.47 | 19.83 | 27.01 | x1.36 | 0/0/0 | 7.64 | 4.45 | 499/500 | 122/133 (92%) | perf: HVDC flows from OLF statics (no whole-network StateVector reload) + RL pinning carried across orchestrator solves |
| 2026-09-08 | bd8c69d+dirty | 24.52 | 19.82 | 26.12 | x1.32 | 0/0/0 | 7.71 | 4.45 | 499/500 | 122/133 (92%) | monitoring phase B follows the untouched-scenarios rule (parity fix + skips the confirm solve); DS carry masked by the copy-back |
| 2026-09-09 | 4f7790c+dirty | 21.72 | 19.47 | 26.27 | x1.35 | 0/0/0 | 7.72 | 4.43 | 499/500 | 122/133 (92%) | balanced chunks (no ragged padding-heavy tail chunk) + JNI/outer-pass instrumentation |
| 2026-09-09 | 3e64cc7+dirty | 22.06 | 19.73 | 27.81 | x1.41 | 0/0/0 | 7.57 | 4.43 | 499/500 | 122/133 (92%) | injectState: cached row permutation + dense base-row table (2.02 -> 1.55 s per 2000 ctg) |
| 2026-09-09 | bf2cfe7+dirty | 21.68 | 19.06 | 26.76 | x1.40 | 0/0/0 | 7.69 | 4.43 | 499/500 | 122/133 (92%) | chunk sizing priced ONCE per extraction (was racing free device memory across partitions) |
| 2026-09-09 | c77c936+dirty | 22.31 | 19.01 | 27.12 | x1.43 | 0/0/0 | 7.62 | 4.42 | 499/500 | 122/133 (92%) | values-only state injection (OLF patch: StateVector.setValuesOnly skips the derivative refresh) |
| 2026-09-09 | b59aeae+dirty | 22.28 | 18.64 | 27.78 | x1.49 | 0/0/0 | 7.48 | 4.48 | 499/500 | 122/133 (92%) | GPU SA forced single-partition so the BASE load flow always runs on the device |
| 2026-09-09 | a82b381+dirty | 22.98 | 19.66 | 26.73 | x1.36 | 0/0/0 | 7.45 | 4.46 | 499/500 | 122/133 (92%) | loss pre-distribution runs for fragmenting contingencies (islanded DS participants excluded, not the whole thing skipped) |
| 2026-09-09 | deec3e1+dirty | 23.81 | 20.20 | 28.62 | x1.42 | 0/0/0 | 8.64 | 5.07 | 499/500 | 122/133 (92%) | CPU-route contingencies that strand a DS participant (device water-fill renormalizes over the wrong set) |
| 2026-09-09 | 881a533+dirty | 24.61 | 21.88 | 31.00 | x1.42 | 0/0/0 | 8.35 | 4.45 | 499/500 | 122/133 (92%) | DS water-fill re-derives from the anchor each pass like OLF's resetToInitialState (fidelity; does not close the rte6515 gap) |
| 2026-09-09 | d461c05+dirty | 22.28 | 19.08 | 27.51 | x1.44 | 0/0/0 | 7.56 | 4.42 | 499/500 | 122/133 (92%) | DISTR_Q target: port AcTargetVector instead of assuming 0 |
| 2026-09-09 | 12686fa+dirty | 23.40 | 20.00 | 27.44 | x1.37 | 0/0/0 | 7.66 | 4.43 | 499/500 | 122/133 (92%) | post-contingency realistic-voltage check (port of AcloadFlowEngine.isStateUnrealistic) |
| 2026-09-09 | 0ccc0bc+dirty | 22.47 | 19.41 | 26.41 | x1.36 | 0/0/0 | 7.75 | 4.44 | 499/500 | 122/133 (92%) | probes only: generator targetP + curve limits in the Q probe |
| 2026-09-09 | cb69a35+dirty | 22.36 | 19.23 | 27.31 | x1.42 | 0/0/0 | 7.62 | 4.47 | 499/500 | 122/133 (92%) | loss pre-distribution is a displacement, not a new DS anchor |
| 2026-09-09 | 85513c4+dirty | 22.23 | 19.85 | 27.92 | x1.41 | 0/0/0 | 7.50 | 4.45 | 499/500 | 122/133 (92%) | precompute sees the CONTINGENCY outer-loop list (JAVA hybrid was missing FreezingHvdcACEmulation) |
| 2026-09-10 | 23738a1+dirty | 22.73 | 19.23 | 27.78 | x1.44 | 0/0/0 | 7.56 | 4.43 | 499/500 | 122/133 (92%) | hybrid chunking: chunk like DEVICE, restore the network per chunk, drop diagnostic retention |
| 2026-09-10 | f60263a+dirty | 22.02 | 18.99 | 27.16 | x1.43 | 0/0/0 | 7.64 | 4.41 | 499/500 | 122/133 (92%) | cut per-save allocations in OLF state snapshots (maps to parallel arrays) + allocation-free row-collision guard |
| 2026-09-10 | 1612d2e+dirty | 22.63 | 19.14 | 27.65 | x1.44 | 0/0/0 | 7.72 | 4.44 | 499/500 | 122/133 (92%) | hybrid dirty-tracking: restore only what a scenario moved (correct, save side 6-7x, total a wash) |
| 2026-09-10 | e370e78+dirty | 22.83 | 19.56 | 27.29 | x1.40 | 0/0/0 | 7.74 | 4.44 | 499/500 | 122/133 (92%) | hybrid dirty-tracking: revert only the difference (parity 0, still no measurable gain) |
| 2026-09-10 | 749c0a3+dirty | 20.95 | 17.38 | 30.76 | x1.77 | 0/0/0 | 7.71 | 4.42 | 499/500 | 122/133 (92%) | flat limit table for post-contingency violation detection (detectViolations 10.8 -> 6.9 s at rte6515/2000, 0 differing lines) |
| 2026-09-10 | 13b0a76+dirty | 19.38 | 16.06 | 26.70 | x1.66 | 0/0/0 | 7.55 | 4.48 | 499/500 | 122/133 (92%) | on-device violation screening: kernel proposes against a relaxed threshold, host confirms with OLF's exact > (rte6515/2000: detectViolations 6.0 -> 0.42 s, wall 33.6 -> 25.1 s, 0 differing lines) |
| 2026-09-10 | 3f8ba88+dirty | 19.21 | 15.97 | 27.43 | x1.72 | 0/0/0 | 7.69 | 4.48 | 499/500 | 122/133 (92%) | per-generator DS water-fill (OLF saturates per generator, not per bus) + drop the orchestrator's DS anchor carry (OLF re-derives from the base anchor, never increments); stranding contingencies batchable behind olf.gpu.sa.batchStrandedDs |
| 2026-09-10 | 1570f28+dirty | 19.19 | 15.98 | 27.23 | x1.70 | 0/0/0 | 7.72 | 4.47 | 499/500 | 122/133 (92%) | reactive-limits islanded mask (the device pinned a controller in another synchronous component) + OLF's in-place factor normalization |
| 2026-09-10 | 1f98b71+dirty | 19.07 | 15.54 | 26.33 | x1.69 | 0/0/0 | 7.71 | 4.42 | 499/500 | 122/133 (92%) | HVDC loops as driver phases: RL runs in the driver so no orchestrator owes a spurious extra Newton step (2000 rte6515: 1 -> 0 differing lines, 20.7 -> 19.1 s) |
| 2026-09-10 | 439fbc6+dirty | 19.66 | 16.36 | 28.62 | x1.75 | 0/0/0 | 7.70 | 4.44 | 499/500 | 122/133 (92%) | Slack-only loss pre-distribution: seed the slack's own target displacement so its DS check measures OLF's mismatch; HVDC loops in the kernel now DEFAULT (2000 rte6515 = 0 differing lines) |
| 2026-09-10 | 3b4ef0a+dirty | 19.08 | 16.44 | 28.26 | x1.72 | 0/0/0 | 7.96 | 4.47 | 499/500 | 122/133 (92%) | Stranded-DS batching ON by default (99.9% batched, 0 differing over 2000); remove the superseded islanded-DS override workaround, the dead hv index fill and this session's debug scaffolding |
| 2026-09-10 | f466d5c+dirty | 19.89 | 16.06 | 26.56 | x1.65 | 0/0/0 | 7.44 | 4.42 | 499/500 | 122/133 (92%) | ISLANDS_REMOTE_VC_MEMBER: reject only a VC group the island SPLITS, not one it takes whole -> 2000 rte6515 is 100% batched, 0 CPU-routed, 0 differing |
| 2026-09-10 | 0dc1b64+dirty | 18.63 | 16.07 | 26.58 | x1.65 | 0/0/0 | 7.88 | 4.51 | 499/500 | 122/133 (92%) | Delete the hvdcLoopsInKernel and batchStrandedDs off-paths (both were bugs): -292 lines, incl. the whole host HVDC orchestrator |
| 2026-09-10 | 4670b00+dirty | 18.53 | 16.04 | 27.86 | x1.74 | 0/0/0 | 7.67 | 4.46 | 499/500 | 122/133 (92%) | Clear the WHOLE disabledMask per chunk: stale HVDC hv bits leaked into the next chunk's slot -> rte6515 9042 goes 18 differing lines to ZERO, and the result is now chunk-size independent |
| 2026-09-10 | 257834a+dirty | 19.15 | 17.61 | 30.02 | x1.70 | 0/0/0 | 8.71 | 4.48 | 499/500 | 122/133 (92%) | Batch the islands that merely COST a remote-VC group a controller: renormalize its reactive keys through the generator-outage channel (9042: 10 -> 6 CPU-routed, still 0 differing) |
| 2026-09-10 | 3ffb684+dirty | 18.78 | 15.94 | 27.76 | x1.74 | 0/0/0 | 7.58 | 4.44 | 499/500 | 122/133 (92%) | Batch the islands that KILL a remote voltage control: every surviving member bus solves its own reactive balance again (9042: 100% batched, 6 -> 3 CPU-routed, 0 differing) |
| 2026-09-10 | bdd788c+dirty | 19.15 | 15.96 | 26.78 | x1.68 | 0/0/0 | 7.60 | 4.54 | 499/500 | 122/133 (92%) | Batch the islanded AC-emulation HVDC endpoint (masking the link IS OLF's processHvdcsWithoutPower there): 9042 qualification now rejects NOTHING, 2 CPU-routed are OLF's own non-convergent pair |
| 2026-09-10 | 4dd1357+dirty | 18.92 | 16.38 | 28.27 | x1.73 | 0/0/0 | 7.64 | 4.53 | 499/500 | 122/133 (92%) | Hand the device arm the RUN's Newton parameters, not new defaults: it was solving with maxIterations=15 while the RTE parameters configure 30 |
| 2026-09-10 | 32ae932+dirty | 18.98 | 16.27 | 27.20 | x1.67 | 0/0/0 | 7.53 | 4.45 | 499/500 | 122/133 (92%) | Fold MonitoringVoltage (OLF order 6) into the device driver: the orchestrator's phase-B confirmation solve is retired, not worked around |
| 2026-09-11 | f22b257+dirty | 19.73 | 16.25 | 27.21 | x1.67 | 0/0/0 | 7.62 | 4.47 | 499/500 | 122/133 (92%) | Driver pause/resume mechanism (inert until a host loop is enabled): the driver can stop at a host-owned slot in OLF's order and continue where it stopped |
| 2026-09-11 | 97a9fba+dirty | 20.00 | 16.07 | 26.85 | x1.67 | 0/0/0 | 7.41 | 4.40 | 499/500 | 122/133 (92%) | SecondaryVoltageControl left the SHARED equation-system state vector holding the last scenario's state: base LF 7.2 MW out, 16 differing lines with SVC on -> 0 |
| 2026-09-11 | e36a2ad+dirty | 19.15 | 16.08 | 27.10 | x1.69 | 0/0/0 | 7.53 | 4.52 | 499/500 | 122/133 (92%) | Base LF batch-of-one can now drive the device outer loops (OLF_GPU_DRIVER_OUTER_LOOPS=1); default path unchanged |
| 2026-09-11 | 1a12a91+dirty | 19.38 | 16.30 | 26.75 | x1.64 | 0/0/0 | 7.66 | 4.64 | 499/500 | 122/133 (92%) | Base LF on the device driver: Java loops now suppressed at the engine layer; blocker is the driver's DS<->RL cycling on a cold-start base |
| 2026-09-14 | 37f446b+dirty | 19.25 | 16.07 | 27.71 | x1.72 | 0/0/0 | 7.65 | 4.46 | 499/500 | 122/133 (92%) | in-kernel incremental TVC + base LF on the full driver (opt-in), shared outer-loop tables |
| 2026-09-15 | daff57d+dirty | 18.95 | 15.48 | 28.06 | x1.81 | 0/0/0 | 7.52 | 4.44 | 499/500 | 122/133 (92%) | forced-PV election in-kernel (rlForcedPvElectBatch): the detector's CPU rejection is gone |
| 2026-09-15 | 58ca046+dirty | 19.19 | 16.35 | 28.16 | x1.72 | 0/0/0 | 7.56 | 4.45 | 499/500 | 122/133 (92%) | driver-on base: restart on host mutation, RL after resume, HVDC slack row, SVC NaN-V pack, slack share split (39/136 under the flag, none >4e-4 kV) |
| 2026-09-15 | e417e0f+dirty | 19.12 | 16.00 | 28.10 | x1.76 | 0/0/0 | 7.38 | 4.44 | 499/500 | 122/133 (92%) | base LF DS anchored on initialTargetP (OLF resetToInitialState), retain at fire, restart from network state, decisions to network before host checks: 136/136 under the flag |
| 2026-09-15 | cbcdcf7+dirty | 19.70 | 16.05 | 27.23 | x1.70 | 0/0/0 | 7.44 | 4.42 | 499/500 | 122/133 (92%) | in-kernel IncrementalShuntVoltageControl (LOOP_SHUNT_KERNEL, groups round-robin), chunks + base |
| 2026-09-15 | 2094783+dirty | 18.80 | 15.92 | 28.62 | x1.80 | 0/0/0 | 7.60 | 4.59 | 499/500 | 122/133 (92%) | shared-group incremental TVC round-robin in the TVC phase (tvcGroupDecisionBatch); host TVC orchestrator off for every incremental group |
| 2026-09-15 | 2094783+dirty | 19.25 | 16.08 | 27.65 | x1.72 | 0/0/0 | 7.59 | 4.46 | 499/500 | 122/133 (92%) | shared-group TVC + flow-observed (RPC, incremental phase) loops in-kernel: every sensitivity-dependent incremental loop is a driver phase |
| 2026-09-15 | 2094783+dirty | 19.36 | 16.19 | 27.01 | x1.67 | 0/0/0 | 8.12 | 4.46 | 499/500 | 122/133 (92%) | shared-group TVC, flow-observed (RPC, incremental phase) and continuous PhaseControl in-kernel: only SecondaryVc + non-incremental TVC/shunt rounding stay on the host |
| 2026-09-15 | 94f9241+dirty | 18.75 | 16.21 | 27.17 | x1.68 | 0/0/0 | 7.68 | 4.49 | 499/500 | 122/133 (92%) | AFTER_GENERATOR + Simple TransformerVoltageControl and WITH_GENERATOR ShuntVoltageControl in-kernel (LOOP_TVD_KERNEL / LOOP_SVD_KERNEL): only SecondaryVoltageControl stays on the host |
| 2026-09-15 | 70fe7ad+dirty | 18.94 | 15.48 | 27.33 | x1.77 | 0/0/0 | 7.68 | 4.46 | 499/500 | 122/133 (92%) | GeneratorVoltageControlManager freeze + fixTransformerVoltageControls in the in-kernel AFTER_GENERATOR phase (no more gen-under-threshold CPU gate) |
| 2026-09-15 | 48ed502+dirty | 20.39 | 15.75 | 33.49 | x2.13 | 0/0/0 | 8.19 | 4.54 | 499/500 | 122/133 (92%) | obsolete gates lifted (INCREMENTAL+Simple TVC gen-under-level, shunt VC co-presence) + SecondaryVoltageControl order gate |
| 2026-09-15 | d33f9e3+dirty | 21.28 | 16.54 | 31.81 | x1.92 | 0/0/0 | 7.70 | 4.54 | 499/500 | 122/133 (92%) | SecondaryVoltageControl as a DRIVER PAUSE on the chunk path (OLF order 4): SVC co-present with the control loops batches again |
| 2026-09-15 | d33f9e3+dirty | 18.77 | 16.17 | 27.64 | x1.71 | 0/0/0 | 7.73 | 4.56 | 499/500 | 122/133 (92%) | SecondaryVoltageControl driver pause (opt-in) + all-zones pilot shift; default path unchanged |
| 2026-09-15 | ae5dd28+dirty | 19.21 | 16.23 | 28.40 | x1.75 | 0/0/0 | 7.51 | 4.50 | 499/500 | 122/133 (92%) | SecondaryVoltageControl driver pause fixed: resume must not re-upload the disabled mask, and a host-UNSTABLE verdict consumes the outer-loop budget |
| 2026-09-15 | ae5dd28+dirty | 19.19 | 15.57 | 28.72 | x1.84 | 0/0/0 | 7.68 | 4.46 | 499/500 | 122/133 (92%) | SecondaryVoltageControl driver pause is the DEFAULT: matches the host orchestrator at 27 percent less wall time |
| 2026-09-15 | a50c98d+dirty | 19.16 | 16.05 | 26.26 | x1.64 | 0/0/0 | 7.49 | 4.45 | 499/500 | 122/133 (92%) | SecondaryVoltageControl in-kernel phase (opt-in): the whole loop on the device, residual = the extra sweep slot |
| 2026-09-16 | 03d37d8+dirty | 19.20 | 15.55 | 27.37 | x1.76 | 0/0/0 | 7.69 | 4.52 | 499/500 | 122/133 (92%) | SecondaryVoltageControl in-kernel by default: every OLF outer loop is now a device phase |
| 2026-09-16 | 82a36d6+dirty | 19.27 | 15.97 | 28.22 | x1.77 | 0/0/0 | 7.56 | 4.46 | 499/500 | 122/133 (92%) | host orchestrators deleted; device driver is the default for the base load flow |
| 2026-09-16 | 91a44ac+dirty | 20.26 | 17.42 | 31.37 | x1.80 | 0/0/0 | 7.94 | 4.47 | 499/500 | 122/133 (92%) | SecondaryVoltageControl zone systems solved by cuBLAS batched LU; no hand-written linear algebra |
| 2026-09-16 | abd078c+dirty | 18.90 | 15.47 | 26.54 | x1.72 | 0/0/0 | 7.48 | 4.43 | 499/500 | 122/133 (92%) | Robust-mode realistic-V criteria in the RL kernel; forced-PV election installed at context creation |
| 2026-09-16 | df4062e+dirty | 19.72 | 16.49 | 29.55 | x1.79 | 0/0/0 | 7.88 | 4.45 | 499/500 | 122/133 (92%) | Distributed-slack residual flagged on P_RESIDUE_EPS, OLF's own failure threshold |
| 2026-09-16 | 9fc0767+dirty | 19.80 | 16.77 | 28.75 | x1.71 | 0/0/0 | 7.75 | 4.46 | 499/500 | 122/133 (92%) | Realistic-voltage check on the driver's intermediate states, routed to CPU |
| 2026-09-16 | d83be41+dirty | 18.71 | 15.87 | 27.98 | x1.76 | 0/0/0 | 7.46 | 4.51 | 499/500 | 122/133 (92%) | Realistic-voltage check DEFERRED to OLF's fixing loop: rte6515 back to 9040/9042 |
| 2026-09-16 | a195ade+dirty | 18.37 | 15.86 | 28.83 | x1.82 | 0/0/0 | 7.76 | 4.46 | 499/500 | 122/133 (92%) | Remote LIMITER phase control batches; tap table via the OLF getter, no reflection |
| 2026-09-16 | 8d514f8+dirty | 19.35 | 15.97 | 27.72 | x1.74 | 0/0/0 | 7.61 | 4.50 | 499/500 | 122/133 (92%) | fixTransformerVoltageControls connectivity is per scenario, bracketed by two masks |
| 2026-09-16 | f97e006+dirty | 19.17 | 16.52 | 27.16 | x1.64 | 0/0/0 | 7.51 | 4.45 | 499/500 | 122/133 (92%) | Remote generator-VC freeze candidates run in-kernel; no whole-run gate left in AFTER_GENERATOR |
| 2026-09-16 | d3ace41+dirty | 18.83 | 15.65 | 27.70 | x1.77 | 0/0/0 | 7.60 | 4.45 | 499/500 | 122/133 (92%) | SecondaryVoltageControl in-kernel on the BASE load flow too, same route as the SA |
| 2026-09-16 | bbe028b+dirty | 18.51 | 15.78 | 27.46 | x1.74 | 0/0/0 | 7.53 | 4.46 | 499/500 | 122/133 (92%) | One all-or-nothing ownership rule for both routes; partial representation no longer silent |
| 2026-09-16 | b187bd5+dirty | 19.09 | 15.80 | 26.66 | x1.69 | 0/0/0 | 7.64 | 4.48 | 499/500 | 122/133 (92%) | Both routes build their phase tables with ONE function |
| 2026-09-16 | 37b361c+dirty | 18.95 | 16.05 | 27.59 | x1.72 | 0/0/0 | 7.75 | 4.52 | 499/500 | 122/133 (92%) | Phase tables travel as one record, not ten positional arguments |
| 2026-09-16 | 0c3eaf4+dirty | 19.08 | 15.94 | 27.33 | x1.71 | 0/0/0 | 7.77 | 4.48 | 499/500 | 122/133 (92%) | Bus reactive limits shrink when a generator is lost and the bus keeps V control |
| 2026-09-16 | e5390df+dirty | 18.71 | 16.24 | 27.90 | x1.72 | 0/0/0 | 7.56 | 4.47 | 499/500 | 122/133 (92%) | Generator N-1 on rte6515 agrees exactly: 0 differing violations |
| 2026-09-16 | e077b73+dirty | 18.86 | 15.74 | 26.91 | x1.71 | 0/0/0 | 7.78 | 4.49 | 499/500 | 122/133 (92%) | Generator outages batch through the reactive-capability curve channel |
| 2026-09-16 | 82d733b+dirty | 19.02 | 15.32 | 29.67 | x1.94 | 0/0/0 | 7.56 | 4.44 | 499/500 | 122/133 (92%) | Per-scenario load target Q: every generator and load contingency batches |
| 2026-09-16 | 1c4c40f+dirty | 19.14 | 15.85 | 27.61 | x1.74 | 0/0/0 | 7.56 | 4.42 | 499/500 | 122/133 (92%) | Reactive aggregate follows OLF's per-generator rule, not a sum of limits |
| 2026-09-17 | 6db188d+dirty | 19.29 | 15.66 | 28.11 | x1.80 | 0/0/0 | 7.42 | 4.41 | 499/500 | 122/133 (92%) | Loss pre-distribution clamps per generator when a generator is lost |
| 2026-09-17 | b4ff206+dirty | 18.52 | 15.65 | 26.36 | x1.68 | 0/0/0 | 7.72 | 4.44 | 499/500 | 122/133 (92%) | DISTR_Q target renormalizes over a controller a scenario deactivates |
| 2026-09-17 | 530d93f+dirty | 18.98 | 15.69 | 26.97 | x1.72 | 0/0/0 | 7.64 | 4.44 | 499/500 | 122/133 (92%) | A deactivated pilot whose controllers all pin is caught at runtime |
| 2026-09-17 | e42e2bd+dirty | 18.97 | 15.55 | 26.76 | x1.72 | 0/0/0 | 7.70 | 4.43 | 499/500 | 122/133 (92%) | DISTR_Q inactive mask becomes context state; pilot-load cause narrowed to assembly |
| 2026-09-17 | f5817b0+dirty | 18.96 | 15.96 | 28.36 | x1.78 | 0/0/0 | 7.58 | 4.60 | 499/500 | 122/133 (92%) | DISTR_Q row targets renormalize when a scenario deactivates a controller |
| 2026-09-17 | 5a35c75+dirty | 19.06 | 15.73 | 28.40 | x1.81 | 0/0/0 | 7.59 | 4.45 | 499/500 | 122/133 (92%) | Curve limits read each generator's own P from the slack water-fill |
| 2026-09-17 | 7ab448b+dirty | 18.81 | 15.83 | 28.84 | x1.82 | 0/0/0 | 7.63 | 4.45 | 499/500 | 122/133 (92%) | Outage identity in the water-fill + seed: rte6515 generators ALL at machine precision |
| 2026-09-17 | 11ce77e+dirty | 18.94 | 15.69 | 27.08 | x1.73 | 0/0/0 | 7.84 | 4.44 | 499/500 | 122/133 (92%) | Dead remote control batches: rte6515 generators 1388/1389, all at machine precision |
| 2026-09-17 | 7d64f6f+dirty | 18.84 | 15.95 | 27.55 | x1.73 | 0/0/0 | 7.67 | 4.52 | 499/500 | 122/133 (92%) | Islanded frozen rows fixed: rte6515 branch violation diff 2 -> 0, the two long-standing lines gone |
| 2026-09-18 | 8f14a4c+dirty | 18.94 | 15.98 | 27.32 | x1.71 | 0/0/0 | 7.70 | 4.48 | 499/500 | 122/133 (92%) | Frozen pilot row no longer flipped by updatePilotVeqBatch |
| 2026-09-18 | 78f5081+dirty | 18.83 | 15.92 | 28.18 | x1.77 | 0/0/0 | 7.51 | 4.43 | 499/500 | 122/133 (92%) | GEN-1522 batches: pilot-is-own-controller gets the member alternate dest |
| 2026-09-18 | 5332863+dirty | 19.30 | 16.24 | 27.02 | x1.66 | 0/0/0 | 7.59 | 4.50 | 499/500 | 122/133 (92%) | LOAD-1263/1264 batch: DISTR_Q targets follow a member's own load-Q change |
