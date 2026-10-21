# Equation codegen — java2cuda

**The hand-written OLF equation Java is the single source of truth.** The fused
CUDA kernels the GPU Newton-Raphson fills its Jacobian with — and the scatter
tables describing where each output lands — are PARSED and TRANSPILED from the
term classes in `src/main/java/.../ac/equations/`, which are byte-identical to
upstream powsybl-open-loadflow. No formula is authored twice, no Java is
generated, and no symbolic math runs anywhere: OLF hand-writes both the values
AND the Jacobian, so there is nothing left to derive.

```
src/main/java (untouched, upstream)          codegen/java2cuda/spec_<family>.py
  term classes: values + hand Jacobian         METADATA only: which files/methods,
  |                                            trig params, hoisted helpers, CUDA
  |  extract_golden.py's parser:               identity, scatter stamp
  |  flatten locals + recursively inline               |
  |  private helpers AND sibling calls                 v
  +--> flat Java expressions --(rewrite precomputed-trig params to shared kernel
        (valid C for doubles)  locals, hoist components)--> native/cuda/equations/generated/<family>.cuh
                                                          + <family>_scatter.h, codegen/scatter/<family>.scatter.json
```

The emitted kernel bodies are the VERBATIM OLF Java expressions — reviewable
line by line against the source — with the shared trig / component factoring
made explicit as kernel locals (`sinTheta1`, `shunt2`, `reI1`, `rawP`), the same
factoring OLF's own Java does with its locals and helpers.

## Files

- `java2cuda/spec_<family>.py` — one thin per-family spec, metadata only
  (declared exception: HVDC's `extern_methods` carries powsybl-iidm's
  `getHvdcLineLosses` closed form, which lives in another repo).
  Families: closed_branch, open_branch, shunt, hvdc_ac_emulation (the GPU fill kernels, with
  scatter stamps) + closed_branch_current_magnitude (current magnitude, read by the current-limiter
  loops; per-method functions only, no scatter)
  + load_model (ZIP/exponential load models: OLF's accumulation loop in
  AbstractLoadModelEquationTerm is parsed to a per-exp-term body, the sum over a
  load model's terms is the batch kernel's per-entry atomicAdd; no scatter —
  landing rows are dynamic vdlIdx, wired by the packer + parity tests).
- `java2cuda/java2cuda.py` — the transpiler + renderer (stdlib + one Jinja
  template, `templates/fused.cuh.j2`) + scatter emission.
- `java2cuda/prove.py` — the proof (stdlib): parse == the frozen golden manifest
  (`golden/`, extracted once from the original Java), a finite-difference audit
  of OLF's own hand-written Jacobian (skipped for HVDC's intentional
  approximation), rendered kernels == committed (the drift guard), rendered ==
  golden. See `java2cuda/README.md` for the mechanism details.
- `extract_golden.py` — the Java-method parser (flattening + helper inlining);
  also the tool that produced the golden manifests.
- `check_scatter.py` — validates each spec's scatter stamp against OLF's real
  equation-system wiring (term-class variable declarations +
  AcEquationSystemCreator's addTerm calls).
- `.github/workflows/codegen-check.yml` — CI: `render-up-to-date` (re-parse +
  re-render, diff — the alert that an upstream formula change needs a
  re-render) + `formulas-proven` (the numeric proof + scatter check).

## Workflow

An upstream change to a formula method lands in the Java with NO conflicts
(the files are upstream's own). Then:

```
codegen/regenerate.sh          # prove -> check scatter -> render
git diff                       # review the kernel change against the Java change
mvn -o validate                # checkstyle (the Java itself did not change here)
native/build-gpu.sh && native/run_gpu_tests.sh   # the end-to-end gate
```

A formula change makes `prove.py`'s golden check fail loudly (the goldens
freeze the ORIGINAL math) — update the golden manifest deliberately alongside
the kernel re-render, the same "intended change reviewed twice" discipline the
old pipeline had.

## History

The first pipeline (residual specs in Python + SymPy autodiff, generating BOTH
a scalar Java API and the CUDA kernels) guaranteed CPU/GPU consistency but
edited the OLF Java — which made every upstream merge of the equation files a
conflict. It was replaced by this direction (parse the untouched Java) once the
kernels were proven drop-in: rendered == the SymPy pipeline's committed kernels
to <=7e-15 over 4M compiled points, full GPU suite green with either. The
equation Java was then reverted byte-identical to upstream and the SymPy
pipeline retired.
