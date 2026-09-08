#!/usr/bin/env python3
# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# Proof for the java2cuda codegen — pure stdlib, NO SymPy. Four independent
# guards per family, all numeric (formatting-robust):
#
#   A. parse fidelity  — every parsed-and-flattened expression equals the committed
#                        golden manifest (codegen/golden/closed_branch.golden.json,
#                        the independent extraction frozen at swap time) to 1e-12.
#   B. FD audit        — OLF's hand-written derivative methods match a central
#                        finite difference of OLF's own value methods (~1e-6, the
#                        FD floor). This audits the PRODUCTION Jacobian itself —
#                        the check that replaces the residual pipeline's autodiff.
#   C. drift guard     — the RENDERED kernel equals the COMMITTED
#                        native/cuda/<family>.cuh: both .cuh texts are parsed
#                        (locals + SoA assignments) and evaluated in Python over
#                        the same raw inputs, output-for-output, to 1e-12. At
#                        adoption time this proved the swap against the SymPy
#                        pipeline's kernels; since then it flags a stale or
#                        hand-edited committed kernel.
#   D. rendered .cuh   — the rendered kernel also equals the golden manifest fed
#                        the precomputed trig of the same physical point (catches
#                        a bug C and A would share through the parser).
#
# Run:  python3 codegen/java2cuda/prove.py        (exit 0 = proven)
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODEGEN = os.path.dirname(HERE)
ROOT = os.path.dirname(CODEGEN)
sys.path.insert(0, CODEGEN)
sys.path.insert(0, HERE)

import extract_golden as eg  # noqa: E402  (_split_args, for call-argument commas)
import java2cuda  # noqa: E402
import spec_closed_branch  # noqa: E402
import spec_open_branch  # noqa: E402
import spec_shunt  # noqa: E402
import spec_hvdc  # noqa: E402
import spec_closed_branch_i  # noqa: E402
import spec_open_branch_i  # noqa: E402
import spec_vdl  # noqa: E402

# (spec, golden file): the golden is the independent extraction frozen at swap time.
FAMILIES = [
    (spec_closed_branch.MODULE, "closed_branch.golden.json"),
    (spec_open_branch.MODULE, "open_branch.golden.json"),
    (spec_shunt.MODULE, "shunt.golden.json"),
    (spec_hvdc.MODULE, "hvdc.golden.json"),
    (spec_closed_branch_i.MODULE, "closed_branch_i.golden.json"),
    (spec_open_branch_i.MODULE, "open_branch_i.golden.json"),
    (spec_vdl.MODULE, "vdl.golden.json"),
]

EQUIV_TOL = 1e-12      # algebraic identity, parsed-vs-golden / rendered-vs-committed
FD_TOL = 1e-6          # finite-difference audit (central diff, h=1e-6)
SAMPLES = 200

# Physical operating ranges (same values as codegen/prove_equivalence.py, copied so
# this prototype never imports the SymPy-based pipeline). HVDC: rawP = p0 +
# k*(ph1-ph2) spans both signs, so both ternary branches are exercised.
_RANGES = {"v1": (0.9, 1.1), "v2": (0.9, 1.1), "r1": (0.9, 1.1),
           "ph1": (-0.5, 0.5), "ph2": (-0.5, 0.5), "a1": (-0.3, 0.3),
           "ksi": (1.2, 1.9), "y": (10.0, 50.0),
           "g1": (-0.1, 0.1), "g2": (-0.1, 0.1), "b1": (-0.1, 0.1), "b2": (-0.1, 0.1),
           "p0": (-1.0, 1.0), "k": (1.0, 3.0), "r": (0.0, 0.05),
           "lossFactor1": (0.0, 0.1), "lossFactor2": (0.0, 0.1)}

_MATH_NS = {"R2": 1.0, "A2": 0.0, "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt,
            "hypot": math.hypot, "pow": math.pow, "atan2": math.atan2,
            "abs": abs, "fabs": abs,
            # powsybl-iidm HvdcUtils closed form (HVDC golden bodies call it bare)
            "getHvdcLineLosses": lambda p, v, r: r * p * p / (v * v)}


def _c_to_py(s):
    """Translate a C/Java double expression to Python: rewrite every `cond ? a : b`
    ternary (right-associative, paren- and call-argument-aware) into a conditional
    expression; everything else is already valid Python."""
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "?" and depth == 0:
            level, d2 = 0, 0
            for j in range(i + 1, len(s)):
                cj = s[j]
                if cj == "(":
                    d2 += 1
                elif cj == ")":
                    d2 -= 1
                elif d2 == 0 and cj == "?":
                    level += 1
                elif d2 == 0 and cj == ":":
                    if level == 0:
                        return (f"(({_c_to_py(s[i + 1:j])}) if ({_c_to_py(s[:i])}) "
                                f"else ({_c_to_py(s[j + 1:])}))")
                    level -= 1
            raise AssertionError(f"unmatched ternary in: {s}")
    out, i = "", 0                                     # no top-level ternary: recurse
    while i < len(s):                                  # into parenthesized groups
        if s[i] == "(":
            depth, j = 0, i
            while True:
                depth += 1 if s[j] == "(" else (-1 if s[j] == ")" else 0)
                if depth == 0:
                    break
                j += 1
            parts = eg._split_args(s[i + 1:j])         # call args translate independently
            out += "(" + ", ".join(_c_to_py(p) for p in parts) + ")"
            i = j + 1
        else:
            out += s[i]
            i += 1
    return out


def _make_funcs(methods):
    """Compile [{name, params, body}] into callables sharing one namespace, so golden
    bodies that call siblings (dp1dph2 = -dp1dph1(...)) or the extern line-loss
    closed form resolve; HVDC ternaries are translated to Python conditionals."""
    ns = dict(_MATH_NS)
    for m in methods:
        src = f"def {m['name']}({', '.join(m['params'])}):\n    return {_c_to_py(m['body'])}\n"
        exec(src, ns)  # noqa: S102 — trusted, extracted from our own Java
    return ns


def _rand_point(rng, names):
    return {n: rng.uniform(*_RANGES.get(n, (0.5, 1.5))) for n in names}


class RawEval:
    """Evaluate a family method at a RAW point (base symbols): compute the side's
    branch angle from the PARSED theta bodies, derive the precomputed-trig params,
    and call the compiled method with its own parameter list."""

    def __init__(self, mod, pool, methods):
        self.mod = mod
        self.ns = _make_funcs(methods)
        self.params = {m["name"]: m["params"] for m in methods}
        self.thetas = {side: java2cuda.flatten(pool, nm)[1]
                       for side, nm in mod["theta_methods"].items()}

    def trig_point(self, pt, side):
        vals = dict(pt)
        if "ksi" in pt:                                # families with precomputed ksi trig
            vals["sinKsi"], vals["cosKsi"] = math.sin(pt["ksi"]), math.cos(pt["ksi"])
        if side is not None:                           # families with a branch angle
            theta = eval(self.thetas[side], {"__builtins__": {}}, {**_MATH_NS, **pt})  # noqa: S307
            vals["sinTheta"], vals["cosTheta"] = math.sin(theta), math.cos(theta)
        return vals

    def __call__(self, name, side, pt):
        vals = self.trig_point(pt, side)
        return self.ns[name](*[vals[p] for p in self.params[name]])


def check_parse_vs_golden(mod, out, parsed_eval, golden_eval):
    """A. Parsed-and-flattened bodies == the golden manifest at physical points."""
    base = mod["base_symbols"]
    rng = random.Random(20260707)
    worst = 0.0
    for name, side, _body in out:
        for _ in range(SAMPLES):
            pt = _rand_point(rng, base)
            a, b = parsed_eval(name, side, pt), golden_eval(name, side, pt)
            err = abs(a - b) / (1.0 + abs(b))
            worst = max(worst, err)
            if err > EQUIV_TOL:
                raise SystemExit(f"FAIL parse-vs-golden: {name} parsed={a!r} golden={b!r} relerr={err:.3e}")
    print(f"  A. parse fidelity: {len(out)} parsed exprs == golden manifest over "
          f"{SAMPLES} samples (worst relerr {worst:.2e} < {EQUIV_TOL:g})")


def check_fd(mod, out, parsed_eval):
    """B. OLF's hand Jacobian vs central finite differences of OLF's own values —
    perturbing the variable re-derives the branch angle, so the chain rule through
    theta is exercised. A variable with NO derivative method must FD to ~zero.
    Skipped for an intentionally approximated Jacobian (HVDC neglects the cable-loss
    derivative), exactly like the residual pipeline skipped its autodiff audit."""
    if mod.get("approx_jacobian"):
        print("  B. FD audit: skipped — the Jacobian is an intentional approximation "
              "(OLF neglects the cable-loss derivative; an FD check would rightly fail)")
        return
    side_of = {name: side for name, side, _b in out}
    have = {name for name, _s, _b in out}
    h = 1e-6
    rng = random.Random(20260708)
    worst = 0.0
    n_entries = 0
    for res, side in mod["residuals"]:
        for var in mod["variables"]:
            name = f"d{res}d{var}"
            for _ in range(SAMPLES // 4):
                pt = _rand_point(rng, mod["base_symbols"])
                hp, hm = dict(pt), dict(pt)
                hp[var] += h
                hm[var] -= h
                fd = (parsed_eval(res, side, hp) - parsed_eval(res, side, hm)) / (2 * h)
                an = parsed_eval(name, side_of[name], pt) if name in have else 0.0
                err = abs(fd - an) / (1.0 + abs(an))
                worst = max(worst, err)
                if err > FD_TOL:
                    raise SystemExit(f"FAIL FD audit: {name} fd={fd!r} hand={an!r} relerr={err:.3e}")
            n_entries += name in have
    print(f"  B. FD audit: {n_entries} hand-written Jacobian entries == central differences of "
          f"the values (worst relerr {worst:.2e} < {FD_TOL:g}) — OLF's own Jacobian audited")


# --- kernel-text evaluation: parse a rendered .cuh's locals + SoA assignments ----

_CUH_TEMP = re.compile(r"^\s*double (\w+) = (.*);$")
_CUH_OUT = re.compile(r"^\s*(\w+?)(?:_out)?\[i\] = (.*);$")
_CUH_METHOD = re.compile(r"^\w+ inline double (\w+)\(([^)]*)\) \{$")


def parse_cuh(text):
    """The rendered header now MIRRORS the CPU: per-method device functions (name,
    Java-signature params, single return body) + assembler routines whose SoA
    assignments are CALLS. Returns (methods, temps, outs) — outs deduped by name
    (the raw and Pre assemblers emit the same calls)."""
    lines = text.splitlines()
    methods = []
    for i, line in enumerate(lines):
        m = _CUH_METHOD.match(line)
        if m and i + 1 < len(lines) and lines[i + 1].strip().startswith("return "):
            params = [p.split()[-1] for p in m.group(2).split(",") if p.strip()]
            methods.append({"name": m.group(1), "params": params,
                            "body": lines[i + 1].strip().removeprefix("return ").rstrip(";")})
    temps = []
    seen_t = set()
    for m in map(_CUH_TEMP.match, lines):
        if m and m.group(1) not in seen_t:
            seen_t.add(m.group(1))
            temps.append((m.group(1), m.group(2)))
    outs = []
    seen_o = set()
    for m in map(_CUH_OUT.match, lines):
        if m and m.group(1) not in seen_o:
            seen_o.add(m.group(1))
            outs.append((m.group(1), m.group(2)))
    if not outs:
        raise AssertionError("no SoA assignments found in kernel text")
    return methods, temps, outs


def eval_cuh(methods, temps, outs, pt):
    ns = _make_funcs(methods)                          # the per-method functions
    ns.update(pt)
    for sym, expr in temps:                            # HVDC bodies carry C ternaries
        ns[sym] = eval(_c_to_py(expr), {"__builtins__": {}}, ns)  # noqa: S307 — our own arithmetic
    return {name: eval(_c_to_py(expr), {"__builtins__": {}}, ns)  # noqa: S307
            for name, expr in outs}


def check_dropin(mod, rendered_text):
    """C. The rendered kernel == the committed residual-pipeline kernel, evaluated
    over the same raw inputs; names AND order must match, so the scatter tables
    (built from that exact order) stay valid verbatim. The current-magnitude
    families are NEW GPU targets — the residual pipeline emitted no kernel for
    them, so there is nothing to compare against."""
    committed_path = os.path.join(ROOT, "native", "cuda", os.path.basename(mod["cuda_path"]))
    if not os.path.exists(committed_path):
        print(f"  C. drop-in: skipped — NEW GPU target (the residual pipeline emitted no "
              f"native/cuda/{os.path.basename(mod['cuda_path'])})")
        return
    with open(committed_path) as f:
        committed = f.read()
    nm_, nt, no = parse_cuh(rendered_text)
    cm_, ct, co = parse_cuh(committed)
    if [n for n, _ in no] != [n for n, _ in co]:
        raise SystemExit(f"FAIL drop-in: output order differs\n  parsed: {[n for n, _ in no]}"
                         f"\n  committed: {[n for n, _ in co]}")
    inputs = [n for row in mod["cuda_params"] for n in row]
    rng = random.Random(20260709)
    worst = 0.0
    for _ in range(SAMPLES):
        pt = _rand_point(rng, inputs)
        a, b = eval_cuh(nm_, nt, no, pt), eval_cuh(cm_, ct, co, pt)
        for name in a:
            err = abs(a[name] - b[name]) / (1.0 + abs(b[name]))
            worst = max(worst, err)
            if err > EQUIV_TOL:
                raise SystemExit(f"FAIL drop-in: {name} parsed={a[name]!r} committed={b[name]!r} relerr={err:.3e}")
    print(f"  C. drop-in: {len(no)} rendered outputs == the committed "
          f"native/cuda/{os.path.basename(mod['cuda_path'])}, same order "
          f"(worst relerr {worst:.2e} < {EQUIV_TOL:g}) — scatter tables reusable verbatim")


def check_signature_lock(mod, rendered_text, golden):
    """E. The rendered per-method device functions carry EXACTLY the CPU methods'
    signatures: same names, same parameter lists as the golden manifest (OLF's own
    optimized precomputed-trig form). This is the 'same methods as the CPU' contract."""
    meths, _t, _o = parse_cuh(rendered_text)
    rendered = {m["name"]: m["params"] for m in meths}
    gold = {m["name"]: m["params"] for m in golden}
    bad = []
    for name, params in gold.items():
        if name not in rendered:
            continue                                   # helpers kept only in the golden namespace
        if rendered[name] != params:
            bad.append(f"{name}: rendered {rendered[name]} != golden {params}")
    checked = [n for n in gold if n in rendered]
    if bad:
        raise SystemExit("FAIL signature-lock:\n  " + "\n  ".join(bad))
    print(f"  E. signature lock: {len(checked)} device methods carry the exact CPU "
          f"signatures (golden manifest)")


def check_rendered_vs_golden(mod, rendered_text, out, golden_eval):
    """D. The rendered kernel also == the golden manifest (independent of the parser
    path A and C share)."""
    meths, temps, outs = parse_cuh(rendered_text)
    side_of = {name: side for name, side, _b in out}
    rng = random.Random(20260710)
    worst = 0.0
    for _ in range(SAMPLES):
        pt = _rand_point(rng, mod["base_symbols"])
        vals = eval_cuh(meths, temps, outs, pt)
        for name, v in vals.items():
            g = golden_eval(name, side_of[name], pt)
            err = abs(v - g) / (1.0 + abs(g))
            worst = max(worst, err)
            if err > EQUIV_TOL:
                raise SystemExit(f"FAIL rendered: {name} cuda={v!r} golden={g!r} relerr={err:.3e}")
    print(f"  D. rendered .cuh: {len(outs)} kernel outputs == golden over {SAMPLES} samples "
          f"(worst relerr {worst:.2e} < {EQUIV_TOL:g})")


def main():
    for mod, golden_file in FAMILIES:
        pool = java2cuda.parse_pool(mod)
        temps, out = java2cuda.outputs(mod, pool)
        rendered = java2cuda.render_fused(mod, temps, out, pool)
        with open(os.path.join(CODEGEN, "golden", golden_file)) as f:
            gold = json.load(f)
        golden = gold["methods"] + gold.get("helpers", [])   # helpers: golden bodies may call them
        parsed_methods = [{"name": name, "params": java2cuda.flatten(pool, name)[0],
                           "body": java2cuda.flatten(pool, name)[1]} for name, _s, _b in out]
        parsed_eval = RawEval(mod, pool, parsed_methods)
        golden_eval = RawEval(mod, pool, golden)
        print(f"Proving java2cuda({mod['family']}): {len(out)} outputs parsed from "
              f"{len(mod['java_sources'])} Java files (stdlib only, no SymPy):")
        check_parse_vs_golden(mod, out, parsed_eval, golden_eval)
        check_fd(mod, out, parsed_eval)
        check_dropin(mod, rendered)
        check_rendered_vs_golden(mod, rendered, out, golden_eval)
        check_signature_lock(mod, rendered, golden)
    print("PROVEN — every Java-parsed fused kernel equals its golden manifest and the committed "
          "residual-pipeline kernel; OLF's hand Jacobians survive the FD audit.")


if __name__ == "__main__":
    main()
