#!/usr/bin/env python3
# Copyright (c) 2026, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
#
# java2cuda for RULE families: the decisions of an OLF outer loop, written in OLF as static
# functions of primitives (double, boolean, int) that read no network. Same direction as the
# equation families — the OLF Java is the source of truth, the device functions are transpiled
# from it — with a different shape:
#
#   equation family   double-only expressions, locals and helpers inlined into one expression,
#                     assembled into fused routines with a scatter stamp
#   rule family       typed signatures, the Java statements kept one for one (typed locals, one
#                     return), calls between the family's own functions kept as calls, the
#                     class's integer constants emitted with them; no routine, no scatter
#
# The emitted body is the Java text statement for statement: boolean -> bool, Math.abs -> fabs,
# nothing else. Anything outside that subset (a loop, an object, an unknown call) fails the parse.
#
# Proof (prove_family), none of it needs a GPU:
#   A. parse == the frozen golden manifest (codegen/golden/<family>.golden.json)
#   C. render == the committed header
#   V. the committed header, compiled for the host, gives the Java's result on every reference
#      vector OLF's own test maintains (src/test/resources/<vectors>): same inputs bit for bit,
#      same result bit for bit
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODEGEN = os.path.dirname(HERE)
ROOT = os.path.dirname(CODEGEN)

REGEN = "Regenerate with: python3 codegen/java2cuda/java2cuda.py (parsed from the OLF Java)."

C_TYPE = {"double": "double", "boolean": "bool", "int": "int"}

_CONSTANT = re.compile(r"public\s+static\s+final\s+int\s+(\w+)\s*=\s*(-?\d+)\s*;")
# a static method whose body holds no brace: typed locals and one return (a method with a loop
# or a branch is not matched, and asking for it by name fails below)
_METHOD = re.compile(
    r"(?:public|protected|private)\s+static\s+(int|boolean|double)\s+(\w+)\s*\(([^)]*)\)\s*\{([^{}]*)\}",
    re.DOTALL)
_LOCAL = re.compile(r"^(double|boolean|int)\s+(\w+)\s*=\s*(.+)$", re.DOTALL)
_RETURN = re.compile(r"^return\s+(.+)$", re.DOTALL)
# what an expression may be made of: identifiers, numbers, arithmetic, comparisons, && || !, ?:
_EXPRESSION = re.compile(r"^[\w\s+\-*/<>=!&|?:().,]+$")
_WORD = re.compile(r"\b[A-Za-z_]\w*\b")
JAVA_TO_C = [(re.compile(r"\bMath\.abs\("), "fabs(")]
C_CALLS = {"fabs"}
LITERALS = {"true", "false"}


def _olf_src():
    olf_src = os.environ.get("OLF_SRC", os.path.join(os.path.dirname(ROOT), "powsybl-open-loadflow-3"))
    if not os.path.isdir(os.path.join(olf_src, "src", "main", "java")):
        sys.exit(f"error: OLF sources not found at {olf_src} — set OLF_SRC to a "
                 "powsybl-open-loadflow checkout matching the dependency version")
    return olf_src


def _collapse(text):
    return " ".join(text.split())


def _expression(java, where):
    """One Java expression as C: the text itself, with Math.abs as fabs."""
    if not _EXPRESSION.match(java) or ">>>" in java:
        raise AssertionError(f"{where}: outside the rule subset: {_collapse(java)}")
    c = java
    for pattern, replacement in JAVA_TO_C:
        c = pattern.sub(replacement, c)
    if "." in re.sub(r"\d\.\d|\d\.|\.\d", "", c):      # a member access left: an object, or another Math call
        raise AssertionError(f"{where}: member access outside the rule subset: {_collapse(java)}")
    return c


def parse(mod):
    """{"constants": [(name, value)], "methods": [{name, returns, params, statements}]}: the class's
    integer constants in source order, and the spec's methods in the spec's order. A statement is
    ("local", type, name, expression) or ("return", expression); expressions are C text with the
    Java's own line breaks."""
    with open(os.path.join(_olf_src(), mod["java_source"])) as f:
        text = f.read()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)          # javadoc and block comments
    constants = [(name, int(value)) for name, value in _CONSTANT.findall(text)]
    found = {}
    for returns, name, params, body in _METHOD.findall(text):
        if name in found:
            raise AssertionError(f"{mod['family']}: {name} is overloaded")
        found[name] = (returns, params, body)
    known = {name for name, _value in constants} | set(mod["methods"]) | C_CALLS | LITERALS
    methods = []
    for name in mod["methods"]:
        if name not in found:
            raise AssertionError(f"{mod['family']}: no static method {name} with typed locals and one return "
                                 f"in {mod['java_source']}")
        returns, param_list, body = found[name]
        params = []
        for p in param_list.split(","):
            words = p.split()
            if len(words) != 2 or words[0] not in C_TYPE:
                raise AssertionError(f"{mod['family']}.{name}: parameter outside the rule subset: {p.strip()}")
            params.append((words[0], words[1]))
        names = {p for _t, p in params}
        statements = []
        body = re.sub(r"//[^\n]*", "", body)                        # line comments
        parts = [s.strip() for s in body.split(";") if s.strip()]
        for i, statement in enumerate(parts):
            where = f"{mod['family']}.{name}"
            local = _LOCAL.match(statement)
            ret = _RETURN.match(statement)
            if local and i < len(parts) - 1:
                expression = _expression(local.group(3), where)
                statements.append(("local", local.group(1), local.group(2), expression))
            elif ret and i == len(parts) - 1:
                expression = _expression(ret.group(1), where)
                statements.append(("return", expression))
            else:
                raise AssertionError(f"{where}: statement outside the rule subset: {_collapse(statement)}")
            unknown = set(_WORD.findall(expression)) - names - known
            if unknown:
                raise AssertionError(f"{where}: unknown identifier(s) {sorted(unknown)}: {_collapse(expression)}")
            later = set(_WORD.findall(expression)) & set(mod["methods"][mod["methods"].index(name):])
            if later:
                raise AssertionError(f"{where}: calls {sorted(later)}, listed after it in the spec (or itself): "
                                     "a function is emitted before the ones that call it")
            if local:
                names.add(local.group(2))
        methods.append({"name": name, "returns": returns, "params": params, "statements": statements})
    return {"constants": constants, "methods": methods}


def manifest(mod, parsed):
    """The parse as the golden manifest stores it: whitespace collapsed, so that re-indenting the
    Java does not move it."""
    return {
        "java_class": mod["java_class"],
        "constants": [[name, value] for name, value in parsed["constants"]],
        "methods": [{"name": m["name"], "returns": m["returns"],
                     "params": [[t, p] for t, p in m["params"]],
                     "statements": [[_collapse(str(part)) for part in s] for s in m["statements"]]}
                    for m in parsed["methods"]],
    }


def _lines(expression, indent, continuation):
    """The expression on the Java's own lines, re-indented: a multi-line ternary chain stays one
    case per line."""
    lines = [line.strip() for line in expression.splitlines() if line.strip()]
    return ("\n" + " " * (indent + continuation)).join(lines)


def render(mod, parsed):
    macro = mod["cuda_fn_macro"]
    parts = mod["cuda_namespace"].split("::")
    out = [
        "// Copyright (c) 2026, RTE (http://www.rte-france.com)",
        "// SPDX-License-Identifier: MPL-2.0",
        "//",
        f"// {mod['doc']}",
        "//",
        f"// GENERATED — DO NOT EDIT. {REGEN}",
        "#pragma once",
        "#include <math.h>",
        "",
        f"#ifndef {macro}",
        f"#define {macro} __host__ __device__   // also compiles host-side for the vectors check",
        "#endif",
        "",
        " ".join(f"namespace {p} {{" for p in parts),
        "",
        f"// {mod['java_class']}'s constants.",
    ]
    out += [f"constexpr int {name} = {value};" for name, value in parsed["constants"]]
    out += ["",
            "// One device function per OLF Java method — SAME name, SAME signature, SAME statements."]
    for m in parsed["methods"]:
        signature = ", ".join(f"{C_TYPE[t]} {p}" for t, p in m["params"])
        out += ["", f"{macro} inline {C_TYPE[m['returns']]} {m['name']}({signature}) {{"]
        for statement in m["statements"]:
            if statement[0] == "local":
                _kind, ctype, name, expression = statement
                out.append(f"    {C_TYPE[ctype]} {name} = {_lines(expression, 4, 8)};")
            else:
                out.append(f"    return {_lines(statement[1], 4, 5)};")
        out.append("}")
    out += ["", "}" * len(parts) + "  // namespace " + mod["cuda_namespace"], ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------
# Proof
# --------------------------------------------------------------------------------

def _read_argument(ctype, token):
    return {"double": f"bits({token})", "boolean": f"({token} == \"1\")", "int": f"std::stoi({token})"}[ctype]


def _write_result(ctype, call):
    return {"double": f"hex({call})", "boolean": f"std::string({call} ? \"1\" : \"0\")",
            "int": f"std::to_string({call})"}[ctype]


def _driver(mod, parsed, header):
    """A host program that reads the reference vectors, calls the function each line names with the
    line's arguments and compares the result with the line's: doubles by their bits."""
    cases = []
    for m in parsed["methods"]:
        args = ", ".join(_read_argument(t, f"t[{i + 1}]") for i, (t, _p) in enumerate(m["params"]))
        cases.append(f"        else if (t[0] == \"{m['name']}\" && t.size() == {len(m['params']) + 3}) "
                     f"got = {_write_result(m['returns'], m['name'] + '(' + args + ')')};")
    return f"""#define {mod['cuda_fn_macro']}
#include "{header}"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace {mod['cuda_namespace']};
static double bits(const std::string& h) {{ uint64_t u = std::stoull(h, nullptr, 16); double x; std::memcpy(&x, &u, 8); return x; }}
static std::string hex(double x) {{ uint64_t u; std::memcpy(&u, &x, 8); char b[17]; std::snprintf(b, 17, "%016llx", (unsigned long long) u); return b; }}
int main(int argc, char** argv) {{
    if (argc != 2) return 2;
    std::ifstream in(argv[1]);
    std::string line;
    long checked = 0, wrong = 0, lineNumber = 0;
    while (std::getline(in, line)) {{
        lineNumber++;
        std::istringstream ss(line);
        std::vector<std::string> t;
        for (std::string w; ss >> w;) t.push_back(w);
        if (t.empty()) continue;
        std::string got;
        if (t.size() < 3 || t[t.size() - 2] != "=") got = "<malformed>";
{chr(10).join(cases)}
        else got = "<unknown function or arity>";
        checked++;
        if (got != t.back()) {{
            if (wrong++ < 10) std::printf("line %ld: %s\\n  device function gives %s\\n", lineNumber, line.c_str(), got.c_str());
        }}
    }}
    std::printf("%ld %ld\\n", checked, wrong);
    return wrong == 0 && checked > 0 ? 0 : 1;
}}
"""


def check_vectors(mod, parsed):
    """V: the COMMITTED header compiled for the host against OLF's reference vectors."""
    vectors = os.path.join(_olf_src(), mod["vectors"])
    if not os.path.isfile(vectors):
        raise SystemExit(f"FAIL vectors: {vectors} not found (OLF's reference vectors of {mod['java_class']})")
    with open(vectors) as f:
        called = {line.split(" ", 1)[0] for line in f if line.strip()}
    missing = [m["name"] for m in parsed["methods"] if m["name"] not in called]
    if missing:
        raise SystemExit(f"FAIL vectors: no reference vector calls {missing}")
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if compiler is None:
        raise SystemExit("FAIL vectors: no C++ compiler (g++, c++ or clang++; or set CXX) to compile the rule "
                         "functions for the host")
    header = os.path.join(ROOT, mod["cuda_path"])
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "vectors_check.cpp")
        binary = os.path.join(tmp, "vectors_check")
        with open(source, "w") as f:
            f.write(_driver(mod, parsed, header))
        # no contraction: a rule that multiplies then adds must not become a fused multiply-add here
        compiled = subprocess.run([compiler, "-std=c++14", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                                   source, "-o", binary], capture_output=True, text=True)
        if compiled.returncode != 0:
            raise SystemExit(f"FAIL vectors: the committed {mod['cuda_path']} does not compile for the host:\n"
                             + compiled.stderr)
        ran = subprocess.run([binary, vectors], capture_output=True, text=True)
    if ran.returncode != 0:
        raise SystemExit(f"FAIL vectors: {mod['cuda_path']} differs from the Java on OLF's reference vectors:\n"
                         + ran.stdout + ran.stderr)
    checked = ran.stdout.split()[-2]
    print(f"  V. host-compiled {os.path.basename(mod['cuda_path'])}: {checked} reference vectors, every result the "
          f"Java's, bit for bit ({len(parsed['methods'])} functions)")


def prove_family(mod):
    parsed = parse(mod)
    print(f"Proving java2cuda({mod['family']}): {len(parsed['methods'])} rule functions and "
          f"{len(parsed['constants'])} constants parsed from {os.path.basename(mod['java_source'])}:")
    golden_path = os.path.join(CODEGEN, "golden", mod["family"] + ".golden.json")
    with open(golden_path) as f:
        golden = json.load(f)
    now = manifest(mod, parsed)
    if now != golden:
        for key in ("constants", "methods"):
            for g, n in zip(golden[key], now[key]):
                if g != n:
                    raise SystemExit(f"FAIL parse: {mod['family']} differs from its golden manifest:\n"
                                     f"  golden {json.dumps(g)}\n  parsed {json.dumps(n)}\n"
                                     "A rule changed in the Java: update the golden deliberately, with the re-render.")
        raise SystemExit(f"FAIL parse: {mod['family']} differs from its golden manifest (constants or methods "
                         "added or removed)")
    print(f"  A. parse == golden: {len(now['methods'])} functions, {len(now['constants'])} constants, statement "
          "for statement")
    with open(os.path.join(ROOT, mod["cuda_path"])) as f:
        committed = f.read()
    if render(mod, parsed) != committed:
        raise SystemExit(f"FAIL drift: the committed {mod['cuda_path']} is not the render of the Java "
                         "(stale or hand-edited): run codegen/regenerate.sh")
    print(f"  C. render == committed {mod['cuda_path']}")
    check_vectors(mod, parsed)


def write_golden(mod):
    """Deliberately: freeze the current parse as the family's golden manifest."""
    path = os.path.join(CODEGEN, "golden", mod["family"] + ".golden.json")
    with open(path, "w") as f:
        f.write(json.dumps(manifest(mod, parse(mod)), indent=2) + "\n")
    print(f"  wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import spec_reactive_limits_decisions  # noqa: E402
    if sys.argv[1:] == ["--write-golden"]:
        write_golden(spec_reactive_limits_decisions.MODULE)
    else:
        sys.exit("usage: rules.py --write-golden   (freezes the golden manifests; the render and the proof run "
                 "from java2cuda.py and prove.py)")
