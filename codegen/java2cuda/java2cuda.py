#!/usr/bin/env python3
# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda — generate the fused CUDA kernels + scatter tables by PARSING the
# hand-written OLF Java formula methods (src/main/java, byte-identical to
# upstream). NO SymPy: OLF already hand-writes BOTH the values and the Jacobian,
# so there is nothing left to derive — this is a pure TEXT transpiler:
#
#   OLF Java (untouched, source of truth: values AND hand Jacobian)
#     --(extract_golden's parser: flatten locals + inline helper/sibling calls)-->
#   flat Java expressions (valid C for doubles once Math./FastMath. is stripped)
#     --(rewrite the precomputed-trig parameters to shared kernel locals:
#        sinKsi -> sinKsi, sinTheta -> sinTheta<side>, computed ONCE per element
#        from the parsed theta1/theta2 bodies — explicit trig sharing, no CSE)-->
#   native/cuda/equations/generated/<family>.cuh  (bodies are the VERBATIM OLF Java text)
#   + codegen/scatter/<family>.scatter.json / native/cuda/equations/generated/<family>_scatter.h  (from the spec's stamp)
#
# The spec (spec_closed_branch.py) is metadata only; nothing is authored twice.
# Derivative correctness is OLF's own (EquationsTest / production); prove.py adds a
# finite-difference audit + numeric equivalence vs the golden manifest and vs the
# committed residual-pipeline .cuh — all stdlib, no SymPy anywhere.
#
# Run:  codegen/regenerate.sh   (prove -> check scatter -> render)
import json
import os
import re
import sys

from jinja2 import Environment, FileSystemLoader

HERE = os.path.dirname(os.path.abspath(__file__))
CODEGEN = os.path.dirname(HERE)
ROOT = os.path.dirname(CODEGEN)
# The parsed OLF Java lives in the powsybl-open-loadflow DEPENDENCY, not this repo:
# point OLF_SRC at an OLF checkout (or unpacked sources jar) matching the dependency
# version. The equation files are byte-identical across recent versions unless a
# formula changed — in which case prove.py's golden check flags it loudly anyway.
OLF_SRC = os.environ.get("OLF_SRC", os.path.join(os.path.dirname(ROOT), "powsybl-open-loadflow-3"))
if not os.path.isdir(os.path.join(OLF_SRC, "src", "main", "java")):
    sys.exit(f"error: OLF sources not found at {OLF_SRC} — set OLF_SRC to a "
             "powsybl-open-loadflow checkout matching the dependency version")
sys.path.insert(0, CODEGEN)
sys.path.insert(0, HERE)

import extract_golden as eg  # noqa: E402  (the proven Java-method parser, stdlib-only)
import spec_closed_branch  # noqa: E402
import spec_open_branch  # noqa: E402
import spec_shunt  # noqa: E402
import spec_hvdc  # noqa: E402
import spec_closed_branch_i  # noqa: E402
import spec_open_branch_i  # noqa: E402
import spec_vdl  # noqa: E402

MODULES = [spec_closed_branch.MODULE, spec_open_branch.MODULE,
           spec_shunt.MODULE, spec_hvdc.MODULE,
           spec_closed_branch_i.MODULE, spec_open_branch_i.MODULE,
           spec_vdl.MODULE]

# Math calls a flattened body may legitimately keep (extract_golden strips the
# Math./FastMath. prefix); anything else left as a call is an unresolved helper.
# All are C math.h intrinsics for doubles — except abs, which must become fabs.
KNOWN_CALLS = {"sin", "cos", "sqrt", "hypot", "atan2", "pow", "fabs"}
JAVA_TO_C = [(re.compile(r"\babs\("), "fabs(")]

REGEN = "Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java)."


# --------------------------------------------------------------------------------
# Front end: Java -> one flat C expression per method
# --------------------------------------------------------------------------------

def _alpha(params, body):
    """Positional normal form: rename each parameter to its position, so two
    definitions differing only in parameter names (the two abstract bases' `shunt`,
    g1/b1 vs g2/b2) compare equal — inlining substitutes by position, so either
    definition is correct for every call site."""
    for i, p in enumerate(params):
        body = re.sub(rf"\b{re.escape(p)}\b", f"@{i}", body)
    return body


def parse_pool(mod):
    """{"files": [per-file {(name, arity): (params, body, vis)}], "global": merged}.
    Bodies are flattened by extract_golden.body_of (locals inlined); helper/sibling
    CALLS are still present (flatten() resolves them). The GLOBAL pool merges all
    files first-wins; a name+arity defined ALPHA-DIFFERENTLY in two files (each
    class's calculateSensi, each HVDC side's isController) is dropped from it — but
    stays in its OWN file's scope, and flatten() resolves with the defining file's
    locals winning over the global pool (extract_golden's per-file-scope rule).
    Spec-declared extern methods (powsybl-iidm's HvdcUtils closed form) are appended
    as a synthetic globally-visible file."""
    files = []
    for path in mod["java_sources"]:
        with open(os.path.join(OLF_SRC, path)) as f:
            text = f.read()
        files.append({(m["name"], len(m["params"])): (m["params"], m["body"], m["mod"])
                      for m in eg.extract(text)})
    extern = {(m["name"], len(m["params"])): (m["params"], m["body"], "public")
              for m in mod.get("extern_methods", [])}
    if extern:
        files.append(extern)
    global_pool, conflicted = {}, set()
    for fmethods in files:
        for key, (params, body, _vis) in fmethods.items():
            if key in global_pool:
                op, ob, _ov = global_pool[key]
                if _alpha(op, ob) != _alpha(params, body):
                    conflicted.add(key)
                continue
            global_pool[key] = (params, body, _vis)
    for key in conflicted:
        del global_pool[key]
    return {"files": files, "global": global_pool}


def _hmap(entries, exclude_key, keep):
    """inline_helpers map from pool entries: (params, body) values, minus the method
    being flattened (by KEY, so its own other-arity overload — the private
    i1(..., theta) under the public i1(..., ph2) — still inlines) and `keep` names."""
    return {k: v[:2] for k, v in entries.items() if k != exclude_key and k[0] not in keep}


def _select(pool, name):
    """Resolve `name` to ONE method: across files and overload arities, prefer the
    PUBLIC one (closed-branch current magnitude overloads i1: private (..., theta)
    + public (..., a1, ..., ph2) — the public one is the output); definitions of the
    chosen key must agree across files (alpha). Returns (key, params, body, defining
    file's methods)."""
    entries = [(k, v, fm) for fm in pool["files"] for k, v in fm.items() if k[0] == name]
    if not entries:
        raise AssertionError(f"no method named {name}")
    keys = {k for k, _v, _f in entries}
    if len(keys) > 1:
        pub = {k for k, v, _f in entries if v[2] == "public"}
        if len(pub) != 1:
            raise AssertionError(f"ambiguous overloads of {name}: {sorted(keys)}")
        keys = pub
    key = next(iter(keys))
    chosen = [(v, fm) for k, v, fm in entries if k == key]
    (params, body, _vis), defining = chosen[0]
    for (p2, b2, _v2), _f in chosen[1:]:
        if _alpha(p2, b2) != _alpha(params, body):
            raise AssertionError(f"ambiguous method {name}: defined differently in two files")
    return key, params, body, defining


def flatten(pool, name, keep=()):
    """The named method's body with EVERY call to another pool method recursively
    inlined (private helpers like theta1/theta2 AND sibling outputs like dp1dph2's
    -dp1dph1(...)), leaving only arithmetic + math.h intrinsics — except calls to
    `keep` helpers, which stay as calls (to be hoisted into kernel locals). Helpers
    resolve with the DEFINING file's own methods winning over the global pool, so
    each HVDC side inlines its own isController."""
    key, params, body, defining = _select(pool, name)
    flat = eg.inline_helpers(body, _hmap({**pool["global"], **defining}, key, keep))
    for pat, repl in JAVA_TO_C:
        flat = pat.sub(repl, flat)
    flat = _unwrap_idents(flat)
    bad = {c for c in re.findall(r"\b([A-Za-z_]\w*)\s*\(", flat)
           if c not in KNOWN_CALLS and c not in keep}
    if bad:
        raise AssertionError(f"unresolved call(s) {sorted(bad)} in {name}: {flat}")
    return params, flat


def _unwrap_idents(s):
    """Drop redundant parens the recursive inliner piles up: around a bare
    identifier/number — (v1) is v1 — and directly-nested groups — ((x)) is (x)."""
    prev = None
    while prev != s:
        prev, s = s, re.sub(r"(?<![\w.])\(\s*([A-Za-z_]\w*|[0-9.]+)\s*\)", r"\1", s)
        i = 0
        while i < len(s) - 1:                          # ((...)) -> (...)
            if s[i] == "(" and s[i + 1] == "(":
                depth, j = 0, i + 1                    # match the INNER group
                while True:
                    depth += 1 if s[j] == "(" else (-1 if s[j] == ")" else 0)
                    if depth == 0:
                        break
                    j += 1
                if j + 1 < len(s) and s[j + 1] == ")":   # outer closes right after inner
                    s = s[:i] + s[i + 1:j + 1] + s[j + 2:]
                    continue
            i += 1
    return s


def hoist_calls(body, helper, pool, instances, keep):
    """Replace each call to `helper` in body by a kernel local — one local per DISTINCT
    argument tuple (open-branch: shunt(...g1,b1) -> shunt, shunt(...g2,b2) -> shunt2),
    reproducing the factoring OLF's own Java does with its `double shunt = ...` local.
    The local's expression inlines the helper's remaining internal calls (current
    magnitude: reI1's interReI1) so it is pure arithmetic. `instances` maps the
    canonical argument tuple -> (local name, expression) and accumulates across the
    family's outputs (insertion-ordered); the first instance keeps the bare helper
    name, later ones get a numeric suffix."""
    while True:
        m = re.search(rf"\b{re.escape(helper)}\s*\(", body)
        if not m:
            return body
        op = m.end() - 1
        depth, k = 0, op
        while k < len(body):
            depth += 1 if body[k] == "(" else (-1 if body[k] == ")" else 0)
            if depth == 0:
                break
            k += 1
        args = tuple(a.strip() for a in eg._split_args(body[op + 1:k]))
        if args not in instances:
            key = (helper, len(args))
            entries = [(v, fm) for fm in pool["files"] for k2, v in fm.items() if k2 == key]
            if not entries:
                raise AssertionError(f"hoist: no {helper}/{len(args)} in the pool")
            (hp, hb, _vis), defining = entries[0]
            for (p2, b2, _v2), _f in entries[1:]:
                if _alpha(p2, b2) != _alpha(hp, hb):
                    raise AssertionError(f"hoist: ambiguous {helper}/{len(args)} across files")
            # inline the helper's internal calls with ITS file's scope winning (the
            # open-branch-i sides define different gres/bres of the same name/arity)
            expr = eg._subst(hb, hp, list(args))
            expr = _unwrap_idents(eg.inline_helpers(
                expr, _hmap({**pool["global"], **defining}, key, keep)))
            local = helper if not instances else f"{helper}{len(instances) + 1}"
            instances[args] = (local, expr)
        body = body[:m.start()] + instances[args][0] + body[k + 1:]


# --------------------------------------------------------------------------------
# Transpile: rewrite precomputed-trig params to shared kernel locals
# --------------------------------------------------------------------------------

def outputs(mod, pool):
    """(temps, out): the kernel's shared locals (theta<side> + each used trig value,
    computed once per element) and the fused outputs [(name, side, c_body)] in scatter
    order — per residual its value method then d<res>d<var> in variables order (a
    variable with no method in the Java is a zero derivative and is skipped). Each
    body is the VERBATIM flattened OLF Java text, with precomputed-trig parameters
    (sinKsi, sinTheta, ...) rewritten to the shared locals; a side-dependent local
    (its spec expr references `theta`) gets the side suffix: sinTheta -> sinTheta1."""
    have = {k[0] for fm in pool["files"] for k in fm}
    hoist = mod.get("hoist_helpers", [])
    allowed = set(mod["base_symbols"]) | set(mod["constants"]) | KNOWN_CALLS
    locals_ = {}                                       # local name -> C expr (insertion-ordered)
    hoisted = {h: {} for h in hoist}                   # helper -> {arg tuple: (local, expr)}
    theta_sides = set()
    out = []
    for res, side in mod["residuals"]:
        for name in [res] + [f"d{res}d{v}" for v in mod["variables"]]:
            if name not in have:
                continue                               # zero derivative: no method in OLF
            _params, body = flatten(pool, name, keep=hoist)
            for t, expr in mod["precomputed"].items():
                if not re.search(rf"\b{t}\b", body):
                    continue
                if "theta" in expr:                    # side-dependent: bind to this side's angle
                    local = f"{t}{side}"
                    locals_.setdefault(local, expr.replace("theta", f"theta{side}"))
                    theta_sides.add(side)
                else:
                    local = t
                    locals_.setdefault(local, expr)
                body = re.sub(rf"\b{t}\b", local, body)
            for h in hoist:                            # helper calls -> shared kernel locals
                body = hoist_calls(body, h, pool, hoisted[h], hoist)
            body = _unwrap_idents(body)                # a hoisted (shunt2) is just shunt2
            hoist_names = {sym for inst in hoisted.values() for sym, _e in inst.values()}
            unknown = (set(re.findall(r"\b[A-Za-z_]\w*\b", body))
                       - allowed - set(locals_) - hoist_names)
            if unknown:
                raise AssertionError(f"unknown identifier(s) {sorted(unknown)} in {name}: {body}")
            out.append((name, side, body))
    # resolve keep-helper calls INSIDE hoisted expressions: a later-listed helper may
    # call an earlier-listed one (open-branch-i reI/imI call shunt), so its instance
    # expression is rewritten against the earlier helper's shared locals
    for hi, h in enumerate(hoist):
        for dep in hoist[:hi]:
            for args, (sym, expr) in list(hoisted[h].items()):
                expr = _unwrap_idents(hoist_calls(expr, dep, pool, hoisted[dep], hoist))
                hoisted[h][args] = (sym, expr)
    temps = [{"sym": f"theta{s}", "expr": flatten(pool, mod["theta_methods"][s])[1]}
             for s in sorted(theta_sides)]
    temps += [{"sym": n, "expr": e} for n, e in locals_.items()]
    temps += [{"sym": sym, "expr": expr}               # after the trig locals they reference
              for inst in hoisted.values() for sym, expr in inst.values()]
    temp_syms = {t["sym"] for t in temps}
    for t in temps:                                    # a hoisted expr must be pure arithmetic
        bad = {c for c in re.findall(r"\b([A-Za-z_]\w*)\s*\(", t["expr"])
               if c not in KNOWN_CALLS}
        unknown = (set(re.findall(r"\b[A-Za-z_]\w*\b", t["expr"]))
                   - allowed - temp_syms)
        if bad or unknown:
            raise AssertionError(f"unresolved {sorted(bad | unknown)} in local {t['sym']}: {t['expr']}")
    return temps, out


# --------------------------------------------------------------------------------
# Back end: same fused.cuh.j2 template as the residual pipeline (the trig locals
# take the place of the CSE temps)
# --------------------------------------------------------------------------------

def _guarded_consts(consts):
    """Families share model constants (R2 in the closed- AND open-branch headers, same
    namespace) and the fill kernel includes several fused headers in one TU — guard each
    constexpr so the second definition drops out instead of redefining."""
    out = []
    for c in consts:
        name = re.match(r"constexpr double (\w+)", c).group(1)
        out.append(f"#ifndef OLF_CONST_{name}\n#define OLF_CONST_{name}\n{c}\n#endif")
    return "\n".join(out)


def _prune_temps(temps, out):
    """The subset of kernel locals a reduced output set actually references, keeping
    declaration order and transitive references (a component local may use a hoisted
    denominator declared before it)."""
    needed = set()
    for _name, _side, body in out:
        needed.update(re.findall(r"\b[A-Za-z_]\w*\b", body))
    kept = []
    for t in reversed(temps):                          # reverse: keep deps of kept temps
        if t["sym"] in needed:
            kept.append(t)
            needed.update(re.findall(r"\b[A-Za-z_]\w*\b", t["expr"]))
    kept.reverse()
    return kept


def render_fused(mod, temps, out, pool):
    """Render the family header MIRRORING OLF's CPU structure: one device function per
    Java method — SAME name, SAME signature (the golden-locked precomputed-trig form),
    SAME body (the flattened parse) — inside a per-family namespace (the Java class
    scoping), plus four thin ASSEMBLER routines that only call them, exactly like
    AcNetworkVector on the CPU:
      eval<X>       raw angles: computes theta/trig locals, calls the methods
      eval<X>F      raw angles, residual values only
      eval<X>Pre    precomputed-trig parameters (the caller supplies sinKsi/…)
      eval<X>PreF   precomputed-trig, residual values only
    Pre variants are emitted only when the family HAS precomputed-trig parameters."""
    env = Environment(loader=FileSystemLoader(os.path.join(CODEGEN, "templates")),
                      trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True)
    parts = mod["cuda_namespace"].split("::")
    ns_open = " ".join(f"namespace {p} {{" for p in parts)
    ns_close = "}" * len(parts) + "  // namespace " + mod["cuda_namespace"]
    param_rows = ["double " + ", double ".join(row) + "," for row in mod["cuda_params"]]
    consts_block = _guarded_consts(mod["cuda_consts"]) + "\n" if mod["cuda_consts"] else ""

    # The per-method functions: golden signatures + fully-inlined bodies (helpers and
    # sibling calls resolved), UN-substituted — they speak sinKsi/sinTheta natively.
    methods = []
    for name, _side, _b in out:
        params, flat = flatten(pool, name)
        methods.append({"name": name, "params": params,
                        "sig": ", ".join("double " + p for p in params), "body": flat})

    # Call-argument mapping: a precomputed-trig parameter binds to the SIDE's value
    # (sinTheta -> sinTheta1 for a side-1 method) — like the Java call sites, which
    # pass FastMath.sin(theta1) into the sinTheta parameter.
    pre = mod.get("precomputed", {})

    def call_of(name, side, params):
        args = [f"{p}{side}" if p in pre and "theta" in pre[p] else p for p in params]
        return f"{name}({', '.join(args)})"

    calls = {}
    for m, (name, side, _b) in zip(methods, out):
        calls[name] = call_of(name, side, m["params"])

    trig_syms = [t["sym"] for t in temps
                 if any(t["sym"] == p or t["sym"].startswith(p) for p in pre)]
    theta_syms = [t["sym"] for t in temps if re.fullmatch(r"theta\d+", t["sym"])]
    raw_temps = [t for t in temps if t["sym"] in set(trig_syms) | set(theta_syms)]

    resid_names = {res for res, _side in mod["residuals"]}
    all_names = [n for n, _s, _e in out]
    f_names = [n for n in all_names if n in resid_names]

    def used_trig(names):
        u = set()
        for n in names:
            u.update(re.findall(r"\b[A-Za-z_]\w*\b", calls[n]))
        return u

    routines = [
        {"name": mod["cuda_routine"], "doc": mod["cuda_fused_doc"],
         "outputs": all_names, "temps": raw_temps, "calls": calls,
         "param_rows": param_rows},
        {"name": mod["cuda_routine"] + "F", "doc":
         "// F-ONLY variant: the residual values alone (no Jacobian outputs) — for fill\n"
         "// sites that never factorize: residual/convergence checks, line-search\n"
         "// backtracks, frozen-Jacobian iterations.",
         "outputs": f_names, "temps": _prune_temps(raw_temps, [(n, None, calls[n]) for n in f_names]),
         "calls": calls, "param_rows": param_rows},
    ]
    if trig_syms:
        for names, suffix, doc in (
                (all_names, "Pre",
                 "// PRECOMPUTED-TRIG variant — the caller supplies the trig (per-element\n"
                 "// statics + one sincos via angle addition in the fill kernel), matching\n"
                 "// how AcNetworkVector feeds the same methods on the CPU."),
                (f_names, "PreF",
                 "// PRECOMPUTED-TRIG + F-ONLY variant: residual values alone, trig supplied\n"
                 "// by the caller.")):
            u = used_trig(names)
            pre_rows = []
            for row in mod["cuda_params"]:
                kept = [a for a in row if a in u]
                if kept:
                    pre_rows.append(kept)
            pre_rows.append([tsym for tsym in trig_syms if tsym in u])
            routines.append({"name": mod["cuda_routine"] + suffix, "doc": doc,
                             "outputs": names, "temps": [], "calls": calls,
                             "param_rows": ["double " + ", double ".join(r) + "," for r in pre_rows]})
    return env.get_template("fused.cuh.j2").render(
        doc=mod["doc"], regen=REGEN, ns_open=ns_open, ns_close=ns_close,
        fn_macro=mod["cuda_fn_macro"], family=mod["family"],
        consts_block=consts_block, methods=methods, routines=routines)


# --------------------------------------------------------------------------------
# SCATTER descriptor: where each output lands in OLF's fixed-pattern Jacobian.
# Driven by the PARSED output order (value then non-zero derivatives per residual)
# and the spec's stamp; validated against the real term-class wiring by
# codegen/check_scatter.py. Packing convention shared with the GPU fill kernels
# (native/cuda/jni/olf_gpu_nr_jni.cu): a bus owns two equation rows (P, Q) and two
# variable columns (PHI, V); endpoints index which element. Codes: 0/1 = the
# element's first/only and second bus, 2 = the element itself (branch, shunt —
# only ever the endpoint of skip columns like BRANCH_ALPHA1/RHO1, SHUNT_B, which
# are not state variables in plain AC PF).
# --------------------------------------------------------------------------------

EQ_OFFSET = {"BUS_TARGET_P": 0, "BUS_TARGET_Q": 1}
VAR_OFFSET = {"BUS_PHI": 0, "BUS_V": 1, "BRANCH_ALPHA1": 0, "BRANCH_RHO1": 1, "SHUNT_B": 0}
ENDPOINT_CODE = {"bus1": 0, "bus2": 1, "branch": 2, "bus": 0, "shunt": 2}
ROLE_RESIDUAL, ROLE_JACOBIAN, ROLE_SKIP = 0, 1, 2


def _scatter_outputs(mod, out):
    """(residual, var-or-None) per fused-routine output, in the EXACT order the
    routine writes them — straight from the parsed output list."""
    resid_names = {res for res, _side in mod["residuals"]}
    rows = []
    for name, _side, _body in out:
        if name in resid_names:
            rows.append((name, None))
        else:
            res, var = name[1:].split("d", 1)          # d<res>d<var>
            rows.append((res, var))
    return rows


def scatter_descriptor(mod, out):
    """The fixed-pattern-J structural data: per residual the bus equation it feeds
    (the F row), and per Jacobian entry the (equation row, variable column) it
    scatters into. Endpoints are symbolic (bus1/bus2/branch) — the host resolves
    them per element to flat indices."""
    stamp = mod["stamp"]
    req, vcol = stamp["residual_equation"], stamp["variable_column"]
    family = mod["family"]
    residuals, jacobian = [], []
    for res, var in _scatter_outputs(mod, out):
        eq, at = req[res]
        if var is None:
            residuals.append({"name": res, "equation": eq, "at": at})
        else:
            vtype, vat = vcol[var]
            jacobian.append({"name": f"d{res}d{var}",
                             "row": {"equation": eq, "at": at},
                             "col": {"variable": vtype, "at": vat}})
    return {"family": family,
            "doc": f"Scatter map: how each {family.replace('_', '-')} output lands in OLF's "
                   "fixed-pattern AC Jacobian. GENERATED — DO NOT EDIT. " + REGEN,
            "endpoints": stamp.get("endpoints", ["bus1", "bus2", "branch"]),
            "residuals": residuals, "jacobian": jacobian}


def render_scatter_header(mod, out):
    """C++ scatter table for the fill kernel: per fused-routine output (in the exact
    order the routine writes them), where the value lands. role 0=residual(->F),
    1=jacobian(->J), 2=skip (column variable not a state var in plain AC PF)."""
    stamp = mod["stamp"]
    req, vcol = stamp["residual_equation"], stamp["variable_column"]
    family_dash = mod["family"].replace("_", "-")
    prefix = mod["scatter_prefix"]                     # e.g. "CB" -> CbScatter / CB_NOUT / CB_SCATTER
    struct = prefix.capitalize() + "Scatter"
    rows = []
    for res, var in _scatter_outputs(mod, out):
        eq, at = req[res]
        if var is None:
            rows.append((ROLE_RESIDUAL, ENDPOINT_CODE[at], EQ_OFFSET[eq], -1, -1, res))
        else:
            vtype, vat = vcol[var]
            if vtype in VAR_OFFSET:
                rows.append((ROLE_JACOBIAN, ENDPOINT_CODE[at], EQ_OFFSET[eq],
                             ENDPOINT_CODE[vat], VAR_OFFSET[vtype], f"d{res}d{var}"))
            else:
                rows.append((ROLE_SKIP, -1, -1, -1, -1, f"d{res}d{var}"))
    ns = mod["cuda_namespace"].split("::")
    ns_open = " ".join(f"namespace {p} {{" for p in ns)
    ns_close = "}" * len(ns) + "  // namespace " + mod["cuda_namespace"]
    body = "\n".join(
        f"    {{{r}, {re_}, {rq}, {ce}, {cv}}},  // [{i}] {nm}"
        for i, (r, re_, rq, ce, cv, nm) in enumerate(rows))
    endpoints = stamp.get("endpoints", ["bus1", "bus2", "branch"])
    ep_text = ", ".join(f"{ENDPOINT_CODE[e]}={e}" for e in endpoints)
    return f"""// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Scatter table for the fused {family_dash} fill kernel. GENERATED — DO NOT EDIT. {REGEN}
//
// One row per output of {mod["cuda_routine"]}, IN ORDER. role: 0=residual (add to F at its
// equation row), 1=jacobian (add to J at (row,col)), 2=skip (column variable is never a state
// variable in plain AC PF, e.g. SHUNT_B). Branch control columns (BRANCH_ALPHA1/RHO1) are
// emitted as jacobian (role 1) on the branch endpoint: they resolve to a real column only when
// the element derives that control (transformer phase/voltage control) and to absent (-1)
// otherwise, so the host marks the entry skipped per element.
// Endpoint codes: {ep_text}. A bus owns rows [P=+0, Q=+1] and cols [PHI=+0, V=+1]; the branch
// (endpoint 2) owns control cols [ALPHA1=+0, RHO1=+1]:
//   row = busOf(rowEnd) * 2 + rowEq ;  col = busOf(colEnd) * 2 + colVar
#pragma once

{ns_open}

struct {struct} {{ int role; int rowEnd; int rowEq; int colEnd; int colVar; }};

constexpr int {prefix}_NOUT = {len(rows)};
constexpr {struct} {prefix}_SCATTER[{prefix}_NOUT] = {{
{body}
}};

{ns_close}
"""


def _write(path, text):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    print(f"  wrote {path}")


def main():
    for mod in MODULES:
        pool = parse_pool(mod)
        temps, out = outputs(mod, pool)
        n_methods = sum(len(fm) for fm in pool["files"])
        print(f"{mod['family']}: parsed {n_methods} static methods -> {len(out)} fused outputs "
              f"+ {len(temps)} shared locals, straight from the OLF Java")
        _write(mod["cuda_path"], render_fused(mod, temps, out, pool))
        if "stamp" in mod:                             # GPU scatter descriptor (fill-kernel families)
            _write(mod["scatter_path"],
                   json.dumps(scatter_descriptor(mod, out), indent=2, ensure_ascii=False))
            _write(mod["scatter_header_path"], render_scatter_header(mod, out))


if __name__ == "__main__":
    main()
