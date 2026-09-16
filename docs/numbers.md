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

## DC — Woodbury security analysis (the `dc` branch)

rte6515_full, ALL-branch N-1 of the main synchronous component (9042), RTE production parameters
switched to DC, `dcFastMode`; probe `DcWoodburyBaselineProbe` (`OLF_DC_GPU=1`). Rule: GPU
single-thread vs CPU 8 threads (user, 2026-09-15). Parity = violation sets keyed
subject|limitType|side|limitName + statuses + monitored flows; the two arms share one JVM (KLU first).

| date | commit | monitored | CPU 8t (s) | GPU 1t (s) | GPU ms/ctg | ratio | on device | parity | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-15 | phase 0 | 100 | 65.3 (1 thread) | 66.2 (cuDSS drop-in) | 7.32 | x0.99 | — | 0 lines, 1.2e-9 MW | zero-code baseline: the LU is <5% of the wall |
| 2026-09-15 | 9fed038 | 100 | 66.7 (1 thread) | 29.3 | 3.24 | x2.28 | 6458/9042 | 0 lines | phase 1: breaking outages still on the stock path |
| 2026-09-15 | f0e981e | 100 | 17.3 | 13.4 | 1.49 | x1.29 | 9042/9042 | 0 lines, 2.5e-10 MW | phase 2a: modified-target columns |
| 2026-09-15 | 660867b | 0 | 16.5 | 3.65 | 0.40 | x4.52 | 9042/9042 | 0 lines | phase 2b: device criterion, IR off — target met |
| 2026-09-15 | 660867b | 100 | 18.4 | 8.3 | 0.92 | x2.22 | 9042/9042 | 0 lines, 2.5e-10 MW | the extra 5.7 s is OLF's monitored-result walk, on both arms |
| 2026-09-15 | full LF on device | 0 | 15.2 / 16.7 | 5.30 / 5.59 | 0.59 / 0.62 | x2.87 / x2.99 | 9042/9042 | 0 lines, 1.7e-9 MW | matrix values, target, slack distribution and solve assembled on the device (GpuDcAssembly); capture 39 ms; the 2.1 s once attributed here to cuDSS is wrong, see the correction below |

### DC — full load flow (GpuLoadFlowProvider vs OpenLoadFlowProvider)

rte6515_full, RTE production parameters with `dc=true`, one `LoadFlow.run` per arm in one JVM
(`DcLoadFlowGpuProbe`). Cold = first run of each arm (network load excluded, CUDA/cuDSS init
included on the GPU arm); warm = second run.

| date | commit | CPU cold (ms) | GPU cold (ms) | CPU warm (ms) | GPU warm (ms) | parity | note |
|---|---|---|---|---|---|---|---|
| 2026-09-15 | full LF on device | 664 | 2529 | 154 | 225 | flows 1.7e-10 MW, angles 1e-11°, dispatch bit-identical | a single DC solve did not pay on the device; the reason given here (cuDSS analysis outweighing KLU's whole run) is wrong, see the correction below |

### DC — contingency chunk size (the AC batch-size question, asked 2026-09-15)

The DC Woodbury path chunks contingencies like the AC path (`DEVICE_CHUNK`, default 256,
`-Dolf.gpu.dc.chunk`), but the chunk is NOT decisive here: there is no cuDSS batch in it. The
matrix is factorized once and a chunk only sizes the Woodbury/flow/screen kernel launches and the
state download; the modified-target columns are solved in fixed 1024-column blocks regardless.
rte6515 × 9042, GPU 1-thread, 0 monitored:

| chunk | wall (s) | ms/ctg |
|---|---|---|
| 64 | 5.27 | 0.58 |
| 256 (default) | 5.80 | 0.64 |
| 1024 | 5.45 | 0.60 |

The spread is run-to-run noise (the same knob on the AC batch path swings ×2 on cuDSS 0.8).

### DC — LODF factors (GpuLodfCalculator vs OLF's LodfCalculator)

rte6515_full, all 9042 outageable branches, one shared `DcLoadFlowContext`, both arms streaming
into the same thread-safe digest (`DcLodfGpuProbe`). CPU 8 threads, GPU single-threaded.

| date | monitored | factors | CPU 8t | GPU factors | GPU setup | ratio on factors | parity |
|---|---|---|---|---|---|---|---|
| 2026-09-15 | 200 | 1.8 M | 671 ms | 597 ms | 2200 ms | x1.13 | exact counts, sum to 9 digits |
| 2026-09-15 | 9042 (full N-1) | 81.8 M | 880 ms | 2006 ms | 2165 ms | x0.44 | exact counts, sum to 9 digits |

Stage breakdown of the full case, which is the verdict: element states alone 512 ms CPU vs 561 ms
GPU (the 9042-column solve is a wash); computing all 81.8 M factors without consuming them 818 vs
969 ms (the GPU's extra is the 650 MB crossing PCIe); streaming them to a writer 880 vs 2006 ms
(one host thread walking 650 MB against eight). A LODF factor is three loads and three flops, so
producing factors for the host is bandwidth-bound and the device cannot win it. It wins when the
factors are consumed on the device — which is what the Woodbury SA already does, at 0.53 ms/ctg.

### DC — sensitivity analysis with contingencies (GpuDcSensitivityAnalysis vs DcSensitivityAnalysis)

rte6515_full, RTE parameters in DC, 50 injection variables x 200 monitored branches = 10 000
factors, 200 contingencies (2 M factor evaluations), `DcSensitivityGpuProbe`. The CPU arm is
single-threaded: OLF's DC sensitivity analysis ignores the executor it is given, so unlike the
security analysis there is no 8-thread arm.

| date | CPU 1t | GPU | ratio | parity |
|---|---|---|---|---|
| 2026-09-15 | 2876 ms | 2030 ms | x1.42 | sensitivities 4.9e-13, function references 8.3e-11 MW |
| 2026-09-15 | 2740 ms | 2012 ms | x1.36 | same |
| 2026-09-15 (contended) | 2437 ms | 6664 ms | x0.37 | same |

The third row is the AC session computing on the same card (1.7 GB resident): the device calls took
4480 ms instead of 65-68 ms. Phase split of a clean run: device calls 68 ms, OLF's per-factor
predefined-result scan 216 ms, OLF's unscale/filter/write 96 ms, the rest OLF's per-contingency
orchestration. The linear algebra is off the critical path; what remains is host bookkeeping,
identical on both arms.

### DC — time-series load flow (GpuTimeSeriesDcEngine vs OLF's step engine)

rte6515_full, RTE parameters in DC with the AC-emulation limits loop off (as the other DC probes
run it), 200 planned generators, one thread, all three datasets streamed into a counting writer
(16,990 rows a step). `DcTimeSeriesBaselineProbe`.

| date | steps | phase | CPU | GPU | ratio |
|---|---|---|---|---|---|
| 2026-09-15 | 1000 | restore + set points + solve | 2.95 ms/step | 1.07 ms/step | x2.76 |
| 2026-09-15 | 1000 | whole run, all rows | 6.12 ms/step | 3.16 ms/step | x1.94 |
| 2026-09-15 | 200 | restore + set points + solve | 5.04 ms/step | 1.50 ms/step | x3.36 |
| 2026-09-15 | 200 | whole run, all rows | 8.73 ms/step | 4.64 ms/step | x1.88 |

CPU per-step split (1000 steps): restore and set points 1.21, solve 1.74, branch rows 2.42, bus and
generator rows 0.75. The device takes the first two; the row emission is host work on both arms and
is what caps the end-to-end ratio at about two.

With the AC-emulation limits outer loop left ON (rte6515's own configuration), driven over the
chunk instead of sending the run to the stock engine:

| date | steps | phase | CPU | GPU | ratio |
|---|---|---|---|---|---|
| 2026-09-15 | 1000 | restore + set points + solve | 3.37 ms/step | 0.71 ms/step | x4.73 |
| 2026-09-15 | 1000 | whole run, all rows | 6.87 ms/step | 3.36 ms/step | x2.04 |

1000/1000 steps on the device, and zero AC-emulation configuration changes on this plan: the loop
costs one check kernel per chunk here. A refill plus refactorization, the price of one configuration
change, is 11.6 ms on rte6515. The 583 ms quoted here for the initial analysis and
factorization is wrong, see the correction below.

With BOTH DC outer loops configured (AC-emulation limits and incremental phase control), driven over
the chunk. Two runs, to show the machine's noise:

| date | steps | phase | CPU | GPU | ratio |
|---|---|---|---|---|---|
| 2026-09-15 | 1000 | restore + set points + solve | 4.49 ms/step | 0.98 ms/step | x4.60 |
| 2026-09-15 | 1000 | whole run, all rows | 14.69 ms/step | 5.54 ms/step | x2.65 |
| 2026-09-15 (repeat) | 1000 | restore + set points + solve | ~6.8 ms/step | 1.12 ms/step | x6.09 |
| 2026-09-15 (repeat) | 1000 | whole run, all rows | 6.79 ms/step | 2.94 ms/step | x2.31 |

1000/1000 steps on the device. This plan never saturates a link nor moves a tap, so the loops cost
one decision kernel per chunk; what they buy is that rte6515 runs on the device with its own
configuration at all.

#### Time-series chunk size — how it is determined (user, 2026-09-15: "how do we determine chunks like AC?")

Three different answers in this port, because three different things sit behind the chunk.

On the AC batch path the chunk IS the cuDSS `UBATCH` batch: one call factorizes and solves that many
systems, so the batch size decides whether the batched kernel is used at all, and it swings the run
by x2 on cuDSS 0.8. On the DC Woodbury path there is no batch: the matrix is factorized once and the
chunk only sizes kernel launches, so it is flat (the table above). On the time-series path the chunk
is the number of STEPS whose right-hand sides are assembled, solved and downloaded together.

The factorization is NOT what the chunk buys here, and an earlier version of this section implied it
was. A time-series run factorizes once for the whole plan: `refreshDeviceMatrix` is guarded by
`deviceMatrixDirty`, so it refactorizes only when an outer loop or a step moves a tap or saturates a
link. Traced with `OLF_TS_TRACE` on the plan below, the refresh count is ZERO at chunk 32 and ZERO
at chunk 1024 - one factorization either way. What changes is the number of chunks, and each chunk
carries a fixed cost: base table uploads, solve setup, the fixed kernel launches, and the outer-loop
sweep rounds (96 rounds at chunk 32 against 3 at chunk 1024, for the same 1000 steps).

rte6515_full, both loops configured, GPU 1 thread, restore + set points + solve:

| chunk | 1000 steps | 4096 steps |
|---|---|---|
| 32 | 1.81 ms/step | |
| 128 | 0.95 ms/step | |
| 256 | 0.91 ms/step | |
| 1024 | 0.76 ms/step | 0.62 ms/step (2048) |
| 4096 | | 0.66 ms/step |

Fitting `total = nChunks x (fixed + marginal x chunk)` to those four points gives a fixed cost of
**34.8 ms per chunk** and a marginal cost of **0.756 ms per step**, which reproduces the measured
points to within 8%. So the whole shape of the sweep is one number amortized:

| chunk | fixed share | measured |
|---|---|---|
| 32 | 1.09 ms/step | 1.84 ms/step |
| 128 | 0.27 ms/step | 0.95 ms/step |
| 256 | 0.14 ms/step | 0.91 ms/step |
| 1024 | 0.03 ms/step | 0.76 ms/step |

Two things follow. The marginal cost per step is flat across the sweep, so widening the
multi-right-hand-side solve does NOT itself pay - a 1024-column solve costs about 32 times a
32-column one. And the curve must flatten once the fixed share is small against 0.756, which is why
1024 and 2048 measure the same and there is nothing to chase above that.

`DEFAULT_CHUNK` is therefore 1024 (`-Dolf.gpu.dc.ts.chunk`). The ceiling is memory: a chunk holds one
right-hand side and one state column per step. A second, smaller argument points the same way on
plans this one does not exercise: the chunk is also the outer-loop grouping unit, steps being grouped
by configuration WITHIN a chunk, so on a plan that DOES move taps or saturate links a larger chunk
shares one refactorization between more steps. On this plan that term is zero.

#### Time-series row emission over several host threads (2026-09-15)

Once the device takes restore, set points and solve, the row emission is the whole remaining cost,
and it is host work identical on both arms. OLF's own step-range partitioning parallelizes it: each
partition thread walks its own steps and writes its own rows, and the device section serializes
in the native layer (cuDSS analysis is not thread-safe across host threads - concurrent
analysis calls segfault the JVM). rte6515_full, 4096 steps, 200 planned generators, chunk 1024, both
loops configured, 69,591,040 rows, end-to-end ms/step:

| threads | CPU | GPU | ratio | CPU "all emit" |
|---|---|---|---|---|
| 1 | 6.01 | 2.61 | x2.30 | 3.10 |
| 2 | 3.95 | 2.06 | x1.92 | 1.94 |
| 4 | 2.39 | 1.44 | x1.66 | 0.97 |
| 8 | 1.97 | 1.39 | x1.42 | 0.74 |

The GPU arm is ahead at every thread count, but the margin NARROWS as threads are added, and the
4-to-8 step buys it almost nothing: 1.44 to 1.39 while the CPU arm goes 2.39 to 1.97. That is the
signature of a serialization floor, and the floor was the device lock: a chunk's whole device
section sat in one global critical section, about 0.6 ms/step against a 1.39 ms/step wall.

#### The same sweep with the lock narrowed to the analysis phase (2026-09-15)

Every context owns its cuDSS handle and its CUDA stream, and factorization and solve on distinct
handles run concurrently. Only the analysis phase and the handle lifecycle share state between
handles, so the exclusion moved into the native layer around exactly those calls, and the chunk
solve now runs unlocked. Same plan, same machine, but a slower session for the host arm, so read the
ratios and the scaling rather than comparing absolute ms against the table above:

| threads | CPU | GPU | ratio | device section |
|---|---|---|---|---|
| 1 | 6.54 to 6.75 | 3.23 to 3.35 | x2.0 | 0.49 |
| 4 | 2.33 | 1.28 | x1.82 | 0.30 |
| 8 | 1.89 to 2.28 | 0.99 to 1.06 | x1.9 to x2.15 | 0.35 to 0.40 |

Two things changed. The GPU arm now scales 3.2x from 1 to 8 threads where it scaled 1.9x before, and
the 4-to-8 step is worth 24% instead of 3%. And the ratio no longer collapses as threads are added:
it holds near x1.9 across the sweep instead of decaying from x2.30 to x1.42. The device section per
step also FALLS as threads are added, from 0.49 to about 0.37, which is the concurrency that the
lock used to forbid.

Per the standing comparison rule the honest headline is the mono-thread GPU against the 8-thread CPU:
3.29 against 1.89 ms/step. The GPU arm still needs about two host threads to match a fully threaded
CPU on this plan. What the narrowing bought is that adding threads now helps it as much as it helps
the CPU.

#### The emission: the per-row network lookups removed (2026-09-15)

With the device section down to a third of a millisecond, the branch rows were the run. Each emitted
row recomputed the two per-unit-to-ampere scales of its branch from the network - terminal, voltage
level, nominal voltage, twice - although those are functions of the nominal voltages alone, which the
branch's pi model already baked in when the network was built. Computing them once per branch in
`LfBranchImpl`, `LfTieLineBranch` and `LfBoundaryLineBranch` halves the branch emission.

A/B on the same machine minutes apart, rte6515_full, 4096 steps, 1 thread:

| | branch emit | all emit | CPU total | GPU total |
|---|---|---|---|---|
| without the cache | 2.49 | 4.14 | 6.89 | 2.94 |
| with the cache | 1.13 to 1.18 | 1.54 to 1.89 | 4.47 to 4.80 | 1.52 to 1.56 |

The saving is about 235 ns per branch per step, more than two divisions and a few field reads
explain on their own; removing the deep call from the hot method most likely lets the rest of the
emission path inline. The control run above is the attribution: reverting only this change puts the
branch emission back at 2.49.

#### Where the time-series run stands after both changes

Same plan and machine, two sweeps, and the box is shared with the AC session so the CPU arm is noisy
while the GPU arm is stable:

| threads | CPU | GPU | device section |
|---|---|---|---|
| 1 | 4.28 to 4.66 | 1.49 to 1.64 | 0.53 to 0.65 |
| 4 | 2.16 to 3.52 | 0.94 to 1.02 | 0.31 to 0.41 |
| 8 | 1.71 to 2.31 | 1.16 to 1.19 | 0.49 to 0.77 |

The standing comparison has flipped. Mono-thread GPU is 1.49 to 1.64 ms/step against 1.71 to 2.31 for
the 8-thread CPU, so on this plan one GPU thread now beats a fully threaded CPU, where at the start
of the day it needed two threads to draw level. The emission is still the largest single term on the
GPU arm, and the per-branch identifier resolution and the per-step zero-impedance flow map are the
next two items in it.

#### After the counter and load-lookup fixes (2026-09-16)

rte6515_full, 4096 steps, 1 thread, two runs:

| | run a | run b |
|---|---|---|
| CPU total | 5.38 | 4.97 |
| GPU total | 1.46 | 1.44 |
| GPU device section | 0.53 | 0.52 |
| branch emit | 0.90 | 1.17 |

Against 1.49 to 1.64 for the GPU arm before them, so both fixes together sit inside the machine's own
noise. The row count is unchanged at 69,591,040, which is the check that the per-partition counters
add up.

#### The load state made positional (2026-09-16)

`LfLoadImpl` kept four maps keyed by the same original load identifier - the reference, p0, q0 and the
absolute variable target - so applying one set point hashed that string four times, and `BusDcState`
copied two of those maps into fresh `HashMap`s on every state save. They are now one record per load
in a list, with an index by identifier for the callers that arrive by name, and the state save and
restore go by position into `double[]`. Same move, for the same reason, as the generator arrays in the
same class.

This box is shared with the AC session and noisy enough to swallow a single run, so the targeted phase
was measured four times on each side of the change, at 1000 steps:

| | runs (ms/step) | min |
|---|---|---|
| before | 1.22, 1.43, 1.71, 2.17 | 1.22 |
| after | 0.84, 0.94, 0.96, 1.50 | 0.84 |

A 31% cut on restore plus set points, with the two distributions barely overlapping.

**The security analysis gain was noise, and is reported as such.** A first pair of runs suggested the
DC security analysis dropped from 6.57 to 3.71 s, which would have been a third. A confirming run on
the same code gave 6.09 s. Two runs of identical code thus spanned 3.71 to 6.09, so the effect on that
path is not separable from the machine here. Parity is unchanged, 0 differing lines on all 9042
contingencies.

### Correction: what the cuDSS analysis actually costs (2026-09-16)

Several rows above carried figures of 583 ms and 2.1 s for the cuDSS analysis, and a verdict that a
single DC solve cannot pay on the device because the analysis alone outweighs KLU's whole run. Both
are wrong. Nothing had ever timed the phase; the numbers were inferred from wall-clock gaps and from
native profiler samples landing in `createAssembled`, which is where CUDA initializes its context and
where the assembly tables upload - not where the analysis runs.

Timed directly, with `OLF_GPU_DC_TIMING=1`:

| run | n | nnz | analysis | factorization |
|---|---|---|---|---|
| security analysis | 6536 | 22774 | 47 ms | 15 ms |
| single load flow | 6520 | 22736 | 124 to 158 ms | 11 to 21 ms |

So the analysis is one to three percent of a multi-second security analysis or time series, not a
third. A permutation cache across contexts, which is what those wrong numbers were about to justify,
would buy about 130 ms on a REPEATED single load flow and nothing measurable anywhere else. It was
not built.

The single load flow has also moved a long way since that row was written, on a machine measured the
same day: cold CPU 662 ms against GPU 906 ms, warm CPU 163 ms against GPU 482 ms, versus 2529 ms cold
before. The verdict stands - a single DC solve still does not pay on the device - but the reason is
CUDA and library start-up plus the assembly capture, not the symbolic analysis.

### DC - the full PTDF assembled on the device (GpuPtdfMatrix, 2026-09-16)

One row per branch, one column per non-slack bus, the sensitivity of that branch's flow to a unit
injection at that bus with the slack absorbing. Assembled in column blocks and LEFT on the device.
rte6515_full, RTE parameters in DC, `DcPtdfGpuProbe`:

| | |
|---|---|
| shape | 9042 branches x 6519 buses = 58,944,798 factors |
| engine prepare (capture, upload, analysis, factorization) | 470 ms |
| assemble (6519-column solve plus the sensitivity kernel) | 412 ms |
| resident | 450 MB |
| read back over PCIe, with a full scan of the values | 576 ms |
| values | max abs 1.20, 49,069,731 non-zero, no NaN |

So about 143 million factors a second to produce, and the read-back costs more than half the assemble
again. That is the whole verdict in one table: assembling is cheap, shipping is not. A caller that
consumes the factors on the device gets them for 412 ms; a caller that wants them in host memory pays
the transfer, and at that point the earlier finding stands that producing factors is not where a GPU
wins.

Correctness is gated against OLF's own DC sensitivity analysis, which computes the same factors by a
different route on the host: `GpuPtdfMatrixTest` compares every (branch, bus) pair on IEEE 14 and
IEEE 118, 9858 factors on the latter, at 1e-10.
