"""Repository conventions, measured from the repository's own code (design
doc, Module 3.4: "auto-extract docstring style, naming conventions, and
import patterns from existing code as 'repo rules'").

Every rule is a measurement with its evidence attached, emitted only when a
clear majority (`MAJORITY`) of enough samples (`MIN_SAMPLES`) supports it.
Nothing is hard-coded per language: a repository that mixes styles gets no
rule for that aspect rather than a guess.
"""

from __future__ import annotations

import collections
import re
from collections.abc import Iterable

MAJORITY = 0.8
MIN_SAMPLES = 20
MAX_RULES = 6

_PY_DEF = re.compile(r"^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\((.*)")
_JS_FUNC = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)")
_STR_SINGLE = re.compile(r"(?<![\w\"])'(?:[^'\\\n]|\\.)*'")
_STR_DOUBLE = re.compile(r"(?<![\w'])\"(?:[^\"\\\n]|\\.)*\"")


def _pct(a: int, n: int) -> str:
    return f"{a} of {n}"


def _majority(counter: collections.Counter, min_samples: int = MIN_SAMPLES):
    n = sum(counter.values())
    if n < min_samples:
        return None, 0, n
    top, k = counter.most_common(1)[0]
    return (top, k, n) if k / n >= MAJORITY else (None, k, n)


def _indent_rule(texts: list[str]) -> str | None:
    unit: collections.Counter = collections.Counter()
    for t in texts:
        prev = 0
        for ln in t.splitlines():
            if not ln.strip():
                continue
            if ln.startswith("\t"):
                unit["tabs"] += 1
                continue
            ind = len(ln) - len(ln.lstrip(" "))
            step = ind - prev
            if step in (2, 4):
                unit[f"{step} spaces"] += 1
            prev = ind
    top, k, n = _majority(unit, 200)
    return f"Indentation: {top} ({_pct(k, n)} indented blocks)." if top else None


def _quote_rule(texts: list[str], language: str) -> str | None:
    c: collections.Counter = collections.Counter()
    for t in texts:
        for ln in t.splitlines():
            s = ln.strip()
            if s.startswith(("#", "//", "*", '"""', "'''")):
                continue
            c["single"] += len(_STR_SINGLE.findall(s))
            c["double"] += len(_STR_DOUBLE.findall(s))
    top, k, n = _majority(c, 100)
    return (f"Strings use {top} quotes ({_pct(k, n)} literals)." if top else None)


def _python_rules(texts: list[str], paths: list[str]) -> list[str]:
    rules = []
    defs = annotated = documented = 0
    styles: collections.Counter = collections.Counter()
    naming: collections.Counter = collections.Counter()
    imports: collections.Counter = collections.Counter()
    out: collections.Counter = collections.Counter()
    for t, p in zip(texts, paths, strict=True):
        lines = t.splitlines()
        is_test = "test" in p.lower()
        for i, ln in enumerate(lines):
            m = _PY_DEF.match(ln)
            if m:
                name = m.group(2)
                if not name.startswith("_"):
                    naming["snake_case" if name == name.lower() else "camelCase"] += 1
                if name.startswith("__") or is_test:
                    continue
                defs += 1
                sig = " ".join(lines[i:i + 6])
                annotated += "->" in sig.split(":\n")[0] or bool(re.search(r"\w:\s*\w", m.group(3)))
                nxt = next((x.strip() for x in lines[i + 1:i + 8] if x.strip()
                            and not x.strip().endswith((",", "(")) and ")" not in x[:2]), "")
                if nxt.startswith(('"""', "'''", 'r"""')):
                    documented += 1
                    block = "\n".join(lines[i + 1:i + 30])
                    styles["Google (Args:/Returns:)"] += bool(re.search(r"\n\s*(Args|Returns):", block))
                    styles["NumPy (Parameters / ----------)"] += bool(
                        re.search(r"\n\s*Parameters\n\s*-{3,}", block))
                    styles["reST (:param x:)"] += ":param" in block
            s = ln.strip()
            if s.startswith("from ."):
                imports["relative"] += 1
            elif s.startswith("from ") and " import " in s:
                imports["absolute"] += 1
            if not is_test:
                if re.match(r"print\(", s):
                    out["print()"] += 1
                elif re.match(r"(logger|log|logging)\.(debug|info|warning|error|exception)\(", s):
                    out["logging"] += 1
    if naming:
        top, k, n = _majority(naming)
        if top:
            rules.append(f"Function names are {top} ({_pct(k, n)} public functions).")
    if defs >= MIN_SAMPLES:
        ratio = annotated / defs
        if ratio >= MAJORITY:
            rules.append(f"Functions have type hints ({_pct(annotated, defs)} functions).")
        elif ratio <= 1 - MAJORITY:
            rules.append(f"Type hints are rarely used ({_pct(annotated, defs)} functions).")
        if documented / defs >= 0.5:
            line = f"Public functions have docstrings ({_pct(documented, defs)})"
            top, k, n = _majority(styles, 10)
            rules.append(line + (f", {top} style ({_pct(k, n)})." if top else "."))
    top, k, n = _majority(imports)
    if top:
        rules.append(f"Package imports are {top} ({_pct(k, n)} from-imports).")
    top, k, n = _majority(out)
    if top:
        rules.append(f"Non-test code reports through {top}, not "
                     f"{'print()' if top == 'logging' else 'logging'} ({_pct(k, n)} calls).")
    return rules


def _js_rules(texts: list[str], language: str) -> list[str]:
    rules = []
    semi: collections.Counter = collections.Counter()
    naming: collections.Counter = collections.Counter()
    modules: collections.Counter = collections.Counter()
    funcs = jsdoc = any_count = loc = 0
    for t in texts:
        lines = t.splitlines()
        loc += len(lines)
        for i, ln in enumerate(lines):
            s = ln.rstrip()
            st = s.strip()
            if re.match(r"^(const|let|var|return|import|export\s+(const|default)|throw)\b", st) \
                    and not st.endswith(("{", "(", "[", ",", "=>", "`")):
                semi["with semicolons" if st.endswith(";") else "without semicolons"] += 1
            m = _JS_FUNC.match(s)
            if m:
                funcs += 1
                name = m.group(1)
                naming["camelCase" if name[:1].islower() and "_" not in name else "other"] += 1
                prev = next((x.strip() for x in reversed(lines[max(0, i - 3):i]) if x.strip()), "")
                jsdoc += prev.endswith("*/")
            if st.startswith("import ") or st.startswith("export ") and " from " in st:
                modules["ES modules (import)"] += 1
            elif "require(" in st:
                modules["CommonJS (require)"] += 1
            if language == "typescript":
                any_count += len(re.findall(r":\s*any\b|<any>|as any\b", s))
    top, k, n = _majority(semi, 100)
    if top:
        rules.append(f"Statements are written {top} ({_pct(k, n)} statements).")
    top, k, n = _majority(modules)
    if top:
        rules.append(f"Modules use {top} ({_pct(k, n)} module lines).")
    top, k, n = _majority(naming)
    if top == "camelCase":
        rules.append(f"Function names are camelCase ({_pct(k, n)} functions).")
    if funcs >= MIN_SAMPLES and jsdoc / funcs >= 0.5:
        rules.append(f"Functions carry JSDoc comments ({_pct(jsdoc, funcs)} functions).")
    if language == "typescript" and loc >= 2000:
        per_k = any_count / (loc / 1000)
        if per_k < 0.5:
            rules.append(f"`any` is almost never used ({any_count} uses in {loc:,} lines).")
    return rules


def _go_rules(texts: list[str]) -> list[str]:
    rules = []
    wrap: collections.Counter = collections.Counter()
    doc = exported = 0
    for t in texts:
        lines = t.splitlines()
        for i, ln in enumerate(lines):
            for m in re.finditer(r"fmt\.Errorf\(\"[^\"]*\"", ln):
                wrap["%w" if "%w" in m.group(0) else "no %w"] += 1
            m = re.match(r"^func\s+(?:\([^)]*\)\s*)?([A-Z]\w*)", ln)
            if m:
                exported += 1
                doc += i > 0 and lines[i - 1].startswith(f"// {m.group(1)}")
    top, k, n = _majority(wrap)
    if top == "%w":
        rules.append(f'Errors are wrapped with fmt.Errorf("...: %w", err) ({_pct(k, n)} Errorf calls).')
    if exported >= MIN_SAMPLES and doc / exported >= 0.5:
        rules.append(f"Exported functions have a doc comment starting with their name "
                     f"({_pct(doc, exported)}).")
    return rules


def _rust_rules(texts: list[str]) -> list[str]:
    rules = []
    unwrap = question = doc = pubs = loc = 0
    for t in texts:
        lines = t.splitlines()
        loc += len(lines)
        in_tests = False
        for i, ln in enumerate(lines):
            if "#[cfg(test)]" in ln:
                in_tests = True
            if not in_tests:
                unwrap += len(re.findall(r"\.unwrap\(\)", ln))
                question += len(re.findall(r"\)\?", ln))
            if re.match(r"^\s*pub\s+fn\b", ln):
                pubs += 1
                doc += i > 0 and lines[i - 1].strip().startswith(("///", "#["))
    if unwrap + question >= MIN_SAMPLES:
        if question / (unwrap + question) >= MAJORITY:
            rules.append(f"Non-test code propagates errors with `?` rather than `.unwrap()` "
                         f"({question} `?` vs {unwrap} unwrap).")
    if pubs >= MIN_SAMPLES and doc / pubs >= 0.5:
        rules.append(f"Public functions have `///` doc comments ({_pct(doc, pubs)}).")
    return rules


def _test_rule(paths: list[str], texts: list[str], language: str) -> str | None:
    c: collections.Counter = collections.Counter()
    for p, t in zip(paths, texts, strict=True):
        if "test" not in p.lower():
            continue
        if language == "python":
            c["pytest"] += "import pytest" in t or ("def test_" in t and "unittest" not in t)
            c["unittest"] += "unittest" in t
        elif language in ("typescript", "javascript"):
            c["vitest"] += "from 'vitest'" in t or 'from "vitest"' in t
            c["jest"] += "jest." in t or "@jest" in t
            c["mocha"] += "from 'mocha'" in t or "require('mocha')" in t
    top, k, n = _majority(c, 5)
    return f"Tests use {top} ({_pct(k, n)} test files)." if top else None


def extract(files: Iterable[tuple[str, str]], language: str) -> list[str]:
    """Rules for `language` from (path, text) pairs of that language's files."""
    pairs = list(files)
    if not pairs:
        return []
    paths = [p for p, _ in pairs]
    texts = [t for _, t in pairs]
    rules: list[str] = []
    if language == "python":
        rules += _python_rules(texts, paths)
    elif language in ("typescript", "javascript"):
        rules += _js_rules(texts, language)
    elif language == "go":
        rules += _go_rules(texts)
    elif language == "rust":
        rules += _rust_rules(texts)
    if language in ("python", "typescript", "javascript"):
        q = _quote_rule(texts, language)
        if q:
            rules.append(q)
    if language not in ("go",):             # gofmt fixes Go's indentation
        ind = _indent_rule(texts)
        if ind:
            rules.append(ind)
    t = _test_rule(paths, texts, language)
    if t:
        rules.append(t)
    return rules[:MAX_RULES]
