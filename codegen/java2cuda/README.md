# java2cuda — parse the OLF Java, transpile to CUDA (no SymPy)

THE equation codegen (see ../README.md for the overview): the untouched,
hand-written OLF Java is the single source of truth; the fused CUDA kernels +
scatter tables in `native/cuda/equations/generated/` are parsed and transpiled from it. Covers six
families: closed-branch, open-branch P/Q, shunt, HVDC AC emulation and the
ZIP/exponential load models (`load_model` — the GPU fill kernels) plus the
closed-branch current magnitude read by the current-limiter loops (its derivatives
divide by hypot(re, im) UNGUARDED at I=0, exactly like the CPU der() they
replicate; `outerloop/flow_observation.cuh` mirrors calculateSensi, which guards it).

## The idea

The first pipeline authored residuals in Python, derived the Jacobian with SymPy,
and generated BOTH a scalar Java API (the term classes were edited to delegate to
it) and the fused CUDA kernel. That guaranteed no CPU/GPU drift — but it **touched
the OLF Java**, which made every upstream merge of the equation files a conflict
(a rebase once deleted private helpers that new upstream code still called).

java2cuda inverts the direction: **the untouched, hand-written OLF Java is the
single source of truth** — it already contains the values AND the hand-derived
Jacobian, production-validated by `EquationsTest`. There is nothing left to derive,
so there is no SymPy anywhere. The pipeline is a pure text transpiler (stdlib +
jinja2 for the shared template):

```
OLF Java (src/main, upstream)                spec_closed_branch.py (METADATA only:
  |                                            which files/methods, sides, trig
  |  extract_golden's parser:                  params, kernel identity, scatter
  |  flatten locals + recursively inline       stamp — no formulas)
  |  private helpers AND sibling calls                 |
  +--> flat Java expressions  --(rewrite sinTheta -> sinTheta<side> etc., the
        (valid C for doubles)   shared per-element trig locals computed from the
                                PARSED theta1/theta2 bodies)--> native/cuda/equations/generated/closed_branch.cuh
```

The emitted kernel bodies are the **verbatim OLF Java text** — reviewable line by
line against the source. The trig sharing that `sp.cse` used to rediscover is now
explicit: `theta1/theta2` + the six sin/cos values are kernel locals computed once,
exactly the factoring `AcNetworkVector` does on the CPU.

## Files

- `spec_<family>.py` — the thin per-family specs: metadata only, zero formulas
  (one declared exception below). What each family exercised:
  - **closed_branch** — the worked example: per-side branch angles (`theta1/theta2`
    parsed from the Java, precomputed `sinTheta`→ shared `sinTheta<side>` locals),
    sibling-call inlining (`dp1dph2 = -dp1dph1(...)`).
  - **open_branch** — no branch angle (`side=None`), zero-derivative dropping
    (`dp2dv1` has no method in the Java → no kernel output, matching the pipeline),
    the cross-file `shunt` helper (defined in BOTH abstract bases with positionally
    renamed params — alpha-equivalent, the parser keeps one), and **helper
    hoisting** (`hoist_helpers`): one kernel local per distinct argument tuple
    (`shunt1` = g1/b1, `shunt2` = g2/b2), the same factoring OLF's own
    `double shunt = shunt(...)` local does.
  - **shunt** — the degenerate case: no trig, no helpers, zero kernel locals.
  - **hvdc_ac_emulation** — the old pipeline's EXPLICIT-family machinery disappears: the Java
    `?:` ternary is ALSO valid C and passes straight through; each side's private
    `isController` (`>= 0` vs `< 0`, same name/arity) resolves by PER-FILE helper
    scope (defining file's locals win over the global pool); `rawP` is hoisted
    (matching the committed kernel's CSE temp `x0`); the FD audit is skipped
    (`approx_jacobian` — OLF neglects the cable-loss derivative on purpose); and
    `getHvdcLineLosses` bottoms out in powsybl-iidm, so its closed form
    `r*p*p/(v*v)` is DECLARED in the spec (`extern_methods`) — the one formula in
    any spec, guarded by the golden equivalence (the residual pipeline's proof
    supplied the identical closed form).
  - **closed_branch_current_magnitude** — the current magnitude the current-limiter loops read.
    The value `i1` is OVERLOADED (public over a private `(..., theta)` core) — the
    parser picks the public one by visibility and inlines the private overload
    through it; each side's own `theta` resolves per-file. No assembler routine
    (`cuda_routines: []`): `outerloop/flow_observation.cuh` calls the methods one
    by one, mirroring `calculateSensi` (control flow the parser does not
    transpile) line for line; `class_methods` emits the private `dreI/dimI`
    components its one-sided rule at I < ZERO_CURRENT_THRESHOLD combines, in a
    namespace named after their Java class. No `dr1` output, matching OLF
    (`calculateSensi` throws on `dr1`), and the FD audit verifies the chain rule
    to 1.9e-9.
  - **load_model** (ZIP/exponential load models) — the parser's REDUCTION shape:
    `AbstractLoadModelEquationTerm.f/dfdv` are ACCUMULATION LOOPS
    (`Σ c·vⁿ · target` over a model's exp terms, `n == 0` skipped), extracted as
    the per-term loop BODY with the term accessors (`expTerm.c()/n()`) promoted
    to params — the sum IS the batch kernel's per-entry atomicAdd (one packed
    element per exp term), and the `n == 0` guard is mirrored at pack time (the
    constant part lives in the target vector). No scatter stamp: landing rows
    are dynamic (`vdlIdx`), so the wiring guarantee stays with the packer +
    parity tests; codegen adds the FORMULA guarantee.
- `java2cuda.py` — parser front end (reuses `codegen/extract_golden.py`) + the
  transpiler + the renderer (reuses `codegen/templates/fused.cuh.j2`).
- `prove.py` — the proof, stdlib-only, per family:
  - **A. parse fidelity** — parsed exprs == the committed golden manifest, 1e-12
    (measured: exact 0.0 — same text).
  - **B. FD audit** — OLF's hand Jacobian == central finite differences of OLF's
    own values, ~1e-6 (replaces the autodiff guarantee; audits the production
    Jacobian itself; also confirms the DROPPED zero derivatives FD to ~0).
    Skipped for HVDC (`approx_jacobian`), like the pipeline's autodiff skip.
  - **C. drop-in** — the rendered kernel == the COMMITTED `native/cuda/equations/generated/<family>.cuh`,
    output-for-output in the same order, 1e-12. Since adoption this is the drift
    guard (rendered vs committed render); at adoption time it proved the swap
    against the SymPy pipeline's kernels (≤7e-15).
  - **D. rendered** — the rendered kernel == the golden manifest, 1e-12.
- Kernels are emitted into `native/cuda/equations/generated/` (all seven host-compile clean in ONE
  translation unit with `g++ -Wall -Wextra` — the shared `R2` constexpr guard is
  exercised); the four fill-kernel families also emit their scatter
  `_scatter.h` + `codegen/scatter/<family>.scatter.json` from the spec's stamp, validated by
  `../check_scatter.py`.

## Rule families: an outer loop's decisions (`rules.py`)

A second shape, for the decisions of an OLF outer loop once OLF holds them as static functions of
primitives that read no network (`ReactiveLimitsDecisions`: a controlling bus switches to PQ, a bus
switched to PQ switches back or follows a limit that moved). Same direction — the OLF Java is the
source of truth — but nothing of the equation machinery applies:

| | equation family | rule family |
|---|---|---|
| types | `double` only | `double`, `boolean`, `int`, in signatures and locals |
| body | locals and helpers inlined into ONE expression | the Java statements one for one: typed locals, one `return` |
| calls | inlined | calls between the family's functions stay calls (a function is listed before its callers) |
| constants | spec-declared model constants | the class's `public static final int` constants, emitted as `constexpr int` |
| output | fused routines + scatter tables | one device function per method, nothing else |

- `spec_reactive_limits_decisions.py` — metadata only: the class, the methods, the reference vectors,
  the CUDA identity. Emits `native/cuda/outerloop/generated/reactive_limits_decisions.cuh`.
- The transpile is the Java text with `boolean` → `bool` and `Math.abs` → `fabs`. Anything else — a
  loop, a branch, an object, a member access, an unknown identifier — fails the parse. The methods
  that walk a `ReactiveLimitsVector` (the election of the bus kept PV, the cap on switches back)
  have loops and are NOT transpiled.
- Proof (`rules.prove_family`, run by `prove.py`; no finite differences — a decision has no
  derivative):
  - **A. parse == golden** — `codegen/golden/<family>.golden.json`, statement for statement,
    whitespace collapsed. A rule changed in OLF fails here; `python3 codegen/java2cuda/rules.py
    --write-golden` freezes the new one, deliberately, with the re-render.
  - **C. render == committed** header, as text.
  - **V. vectors** — the COMMITTED header is compiled for the host (`g++ -O2 -ffp-contract=off`, the
    function macro defined empty) into a small generated driver that replays OLF's reference vectors
    (`src/test/resources/reactiveLimitsDecisionsVectors.txt`, kept by OLF's
    `ReactiveLimitsDecisionsVectorsTest`: one call per line, doubles as the hexadecimal digits of
    their bits) and compares every result with the Java's, bit for bit. 2,644 calls: on each
    comparison of the rules, one bit under, one bit over, far from it, undefined, then 300 seeded
    random points. Mutating a comparison of the header (`<` to `<=`, `>` to `>=`) is caught.
- Not proven here: that a KERNEL calls these functions with the right arguments. No kernel calls
  them yet (the reactive-limit check still has its comparisons inline).
- A rule that multiplies then adds would need care on the device: the host check compiles without
  contraction, the CUDA compiler contracts to a fused multiply-add by default. None of today's rules
  does (`minQ - tol`, `maxQ + tol`, `limit - frozenTargetQ`).

## Run

```
codegen/regenerate.sh                    # prove -> check scatter -> render
```

## Adoption status

DONE. Step 1 reverted the equation Java byte-identical to upstream main and
deleted the old pipeline's generated `*Formulas.java` (commit ac3eef3ea); step 2
re-pointed the specs at `src/main/java`, made this directory emit into
`native/cuda/` (kernels + scatter), and retired the SymPy pipeline
(`codegen/equations/`, `generate.py`, `prove_equivalence.py`). Verified at swap
time: 4M-point compiled differential vs the old kernels ≤2.1e-14 (including the
voltage-collapse regime and the HVDC sign boundary), full GPU suite green with
either kernel set, full CPU+GPU suite 1563/0 after the Java revert.

## Known limits (checked loudly)

- The parser handles the expression-only Java subset OLF uses (locals + single
  return, private helper calls, boolean predicate helpers, ternaries,
  `Math`/`FastMath` calls). A future upstream loop/mutation would fail the parse
  with an explicit error.
- `abs` is mapped to `fabs`; other math calls are C `math.h` intrinsics as-is.
- A method name+arity defined ALPHA-differently in two files is excluded from the
  GLOBAL pool but stays in its file's scope (per-file resolution, defining file
  wins) — consuming a globally-ambiguous name from outside errors.
- A formula living outside this repo (powsybl-iidm's `HvdcUtils`) must be declared
  in the spec (`extern_methods`) and is guarded by the golden equivalence.
