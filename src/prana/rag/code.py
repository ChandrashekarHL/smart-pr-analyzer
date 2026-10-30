"""Code context for one diff hunk, read from the repository at the PR's base
commit. This is what a human reviewer scrolls to before commenting.

Two sources, in priority order:

1. **Enclosing definition.** The class and function the hunk sits in, with
   the full signature and the body lines above the hunk that the diff does
   not show. The hunk header gives git's one-line guess; this gives the
   parameters, types and early returns the change depends on.
2. **Definitions the change uses.** Functions, methods and types called or
   named on the added lines, looked up in the same file, then the same
   directory, then the parent directory. Each is shown as its signature plus
   the doc comment above or docstring below it.

What it deliberately does not do:

- No review text from any pull request, so nothing from this PR (or any
  other) can leak into the prompt.
- No repository-wide search: the lookup is local (file, directory, parent
  directory), which keeps it fast on any repository and lets the backend
  check out only those directories.
- No index or persistent state. Everything is read on the call.

Base-commit code never contains this PR's changes, so the context cannot
contain the reviewed change itself or a later fix of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

EXTENSIONS = {
    "python": (".py",),
    "go": (".go",),
    "rust": (".rs",),
    "typescript": (".ts", ".tsx", ".mts", ".cts"),
    "javascript": (".js", ".jsx", ".mjs", ".cjs"),
}

DEFAULT_BUDGET_CHARS = 1500       # ~750 tokens of code at ~2 chars/token
ENCLOSING_MAX_CHARS = 700
MAX_FILE_BYTES = 400_000
MAX_SCAN_FILES = 60
MAX_SCAN_BYTES = 3_000_000
MAX_DEFINITIONS = 6
SIGNATURE_MAX_LINES = 12          # one parameter per line is common in typed code
BODY_BEFORE_HUNK_LINES = 10

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# One regex per language; group "name" is the defined identifier.
_DEF = {
    "python": re.compile(r"^\s*(?:async\s+def|def|class)\s+(?P<name>[A-Za-z_]\w*)"),
    "go": re.compile(
        r"^(?:func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_]\w*)"
        r"|type\s+(?P<tname>[A-Za-z_]\w*)\s)"),
    "rust": re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|async\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*"
        r"(?:fn|struct|enum|trait|type|union)\s+(?P<name>[A-Za-z_]\w*)"),
    "typescript": None,   # filled below; shared with javascript
    "javascript": None,
}
_JS_DEF = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?"
    r"(?:(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)"
    r"|(?:class|interface|enum|type)\s+(?P<tname>[A-Za-z_$][\w$]*)"
    r"|(?:const|let|var)\s+(?P<vname>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?"
    r"(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|[A-Za-z_$][\w$]*\s*=>))")
# Methods inside a class body: `  async name(args) {` / `  public name<T>(...)`.
_JS_METHOD = re.compile(
    r"^\s+(?:(?:public|private|protected|static|readonly|async|override|get|set)\s+)*"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\([^;]*$")
_DEF["typescript"] = _JS_DEF
_DEF["javascript"] = _JS_DEF

# Rust `impl Foo` / `impl Trait for Foo` encloses methods but defines nothing new.
_RUST_IMPL = re.compile(r"^\s*impl\b")

_KEYWORDS = frozenset("""
if else elif for while return def class import from as with try except finally raise
pass break continue lambda yield async await not and or is in None True False self cls
super print len str int float bool list dict set tuple range open isinstance type
func go defer chan map struct interface package var const select switch case default
fallthrough goto nil make new append cap copy delete panic recover error string byte rune
fn let mut impl trait enum pub use mod crate where match loop move ref static unsafe
Some None Ok Err Box Vec String Option Result Self dyn as
function var this typeof instanceof void null undefined true false throw catch export
require console Array Object Promise Map Set JSON Math Number String Boolean Error Date
""".split())

# Method names so common across standard libraries that a same-named local
# definition is more likely a coincidence than the callee.
_COMMON_METHODS = frozenset("""
constructor init new get set has add put pop push shift append extend insert remove
clear update copy keys values items entries map filter reduce forEach find some every
join split strip trim replace format includes indexOf slice splice concat sort sorted
reverse toString valueOf then catch finally resolve reject call apply bind emit
assert equal equals hash next iter collect into unwrap expect clone to_string as_str
len_ sum min max abs round sqrt floor ceil parse stringify log info warn debug error
read write close open send recv lock unlock run start stop wait Errorf Sprintf Printf
Println Fprintf
""".split())

_CALL = re.compile(r"(?<![\w$])([A-Za-z_$][\w$]*)\s*(?:<[^<>()]*>)?\s*\(")
_TYPE = re.compile(r"(?<![\w$.])([A-Z][A-Za-z0-9_]{2,})\b")


@dataclass(frozen=True)
class Hunk:
    old_start: int          # 1-based line in the base file where the hunk begins
    old_lines: list[str]    # context and removed lines, in order (base side)
    added: list[str]        # added lines
    first_indent: int       # indentation of the first non-blank base-side line


def parse_hunk(diff_hunk: str) -> Hunk | None:
    lines = diff_hunk.splitlines()
    if not lines:
        return None
    m = _HUNK_HEADER.match(lines[0])
    if not m:
        return None
    old, added = [], []
    for ln in lines[1:]:
        if ln.startswith("\\"):            # "\ No newline at end of file"
            continue
        tag, text = (ln[:1], ln[1:]) if ln else (" ", "")
        if tag == "+":
            added.append(text)
        elif tag in (" ", "-"):
            old.append(text)
    first = next((t for t in old if t.strip()), next((t for t in added if t.strip()), ""))
    indent = len(first) - len(first.lstrip())
    return Hunk(int(m.group(1)), old, added, indent)


# ------------------------------------------------------------------ files

def safe_path(repo_dir: Path, rel: str) -> Path | None:
    """`rel` resolved inside `repo_dir`, or None if it escapes (`../`, absolute)."""
    try:
        root = repo_dir.resolve()
        p = (root / rel).resolve()
        p.relative_to(root)
    except (ValueError, OSError):
        return None
    return p


def read_lines(path: Path) -> list[str] | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:              # binary
        return None
    return data.decode("utf-8", "replace").splitlines()


def locate(base: list[str], hunk: Hunk) -> int | None:
    """0-based index in `base` where the hunk's base side starts.

    Trusts the header line number when the lines there match, else searches
    for the first base-side lines (the base branch may have moved since the
    review). Pure additions with no base side use the header number."""
    probe = [t.strip() for t in hunk.old_lines if t.strip()][:3]
    if not probe:
        return min(max(hunk.old_start - 1, 0), len(base)) if base else None

    def matches_at(i: int) -> bool:
        j = 0
        for k in range(i, min(len(base), i + len(hunk.old_lines) + 5)):
            if base[k].strip() == probe[j]:
                j += 1
                if j == len(probe):
                    return True
            elif base[k].strip():
                return False
        return False

    hinted = hunk.old_start - 1
    if 0 <= hinted < len(base) and matches_at(hinted):
        return hinted
    hits = [i for i, ln in enumerate(base) if ln.strip() == probe[0] and matches_at(i)]
    if not hits:
        return None
    return min(hits, key=lambda i: abs(i - hinted))


def _indent(s: str) -> int:
    return len(s) - len(s.lstrip())


def _is_def(line: str, language: str) -> bool:
    if _DEF[language].match(line):
        return True
    if language == "rust" and _RUST_IMPL.match(line):
        return True
    return language in ("typescript", "javascript") and bool(_JS_METHOD.match(line)) \
        and not line.lstrip().startswith(("if", "for", "while", "switch", "catch", "return"))


def signature(lines: list[str], i: int, language: str) -> list[str]:
    """The definition line at `i` plus continuation lines until the header
    closes (`:` for Python, `{` / `;` / `=>` otherwise)."""
    out = []
    for k in range(i, min(len(lines), i + SIGNATURE_MAX_LINES)):
        out.append(lines[k].rstrip())
        s = lines[k].rstrip()
        if language == "python":
            if s.endswith(":"):
                break
        elif s.endswith(("{", ";", "=>")) or ("{" in s and s.count("{") > s.count("}")):
            break
    return out


def doc_for(lines: list[str], i: int, sig_len: int, language: str) -> list[str]:
    """One or two lines of documentation: the docstring under a Python
    definition, or the comment block directly above any other."""
    if language == "python":
        k = i + sig_len
        if k < len(lines):
            s = lines[k].strip()
            if s.startswith(('"""', "'''", 'r"""')):
                text = s.strip("r").strip("\"'").strip()
                return [lines[k].rstrip()] if text else (
                    [lines[k + 1].rstrip()] if k + 1 < len(lines) else [])
        return []
    above = []
    k = i - 1
    while k >= 0 and len(above) < 2:
        s = lines[k].strip()
        if s.startswith(("//", "/*", "*", "#[")) and not s.startswith("#[derive"):
            above.insert(0, lines[k].rstrip())
            k -= 1
        else:
            break
    return above


def _docstring_end(lines: list[str], k: int) -> int:
    """Index of the first line after a Python docstring starting at `k`
    (or `k` itself when there is none)."""
    if k >= len(lines):
        return k
    s = lines[k].strip().lstrip("rRbBuU")
    for q in ('"""', "'''"):
        if s.startswith(q):
            if s.count(q) >= 2 and len(s) > 3:
                return k + 1
            for j in range(k + 1, min(len(lines), k + 200)):
                if q in lines[j]:
                    return j + 1
            return k + 1
    return k


def enclosing(base: list[str], at: int, indent: int, language: str) -> list[int]:
    """Definition lines enclosing base line `at`, outermost first (at most two:
    e.g. a class and its method)."""
    found = []
    limit = indent
    for k in range(at - 1, -1, -1):
        line = base[k]
        if not line.strip():
            continue
        ind = _indent(line)
        if ind < limit and _is_def(line, language):
            found.insert(0, k)
            limit = ind
            if ind == 0 or len(found) == 2:
                break
    return found


def render_enclosing(base: list[str], at: int, hunk: Hunk, language: str,
                     max_chars: int = ENCLOSING_MAX_CHARS) -> str:
    defs = enclosing(base, at, hunk.first_indent, language)
    if not defs:
        return ""
    out: list[str] = []
    for d in defs[:-1]:                       # outer container: signature line only
        out.append(base[d].rstrip())
        out.append(" " * (_indent(base[defs[-1]])) + "...")
    inner = defs[-1]
    sig = signature(base, inner, language)
    doc = doc_for(base, inner, len(sig), language)
    body_start = inner + len(sig)
    if language == "python":
        out.extend(sig + doc)
        body_start = _docstring_end(base, body_start)
    else:
        out.extend(doc + sig)
    # The hunk shows up to 3 context lines itself; show what sits above them.
    body = [ln.rstrip() for ln in base[body_start:at]]
    head = "\n".join(out).rstrip()
    if len(head) > max_chars:                  # even the signature is too long
        return head[:max_chars].rsplit("\n", 1)[0]
    # Keep the body lines nearest the hunk: at most BODY_BEFORE_HUNK_LINES,
    # and only as many as fit; one marker line says how many were dropped.
    marker_room = 40
    kept: list[str] = []
    size = len(head)
    for ln in reversed(body):
        if len(kept) == BODY_BEFORE_HUNK_LINES or size + len(ln) + 1 > max_chars - marker_room:
            break
        kept.insert(0, ln)
        size += len(ln) + 1
    if len(kept) < len(body):
        indent = " " * (_indent(base[inner]) + 4)
        kept.insert(0, f"{indent}... ({len(body) - len(kept)} lines)")
    return "\n".join([head, *kept]).rstrip()


# ------------------------------------------------------------ definitions

def used_names(hunk: Hunk, language: str) -> list[str]:
    """Names called or used as types on the added lines (removed lines too
    when nothing was added), most frequent first, minus keywords, very common
    method names, dunders, and names the hunk defines itself."""
    src = hunk.added or hunk.old_lines
    text = "\n".join(t for t in src if not t.lstrip().startswith(("#", "//", "*", "/*")))
    counts: dict[str, int] = {}
    for rx in (_CALL, _TYPE):
        for m in rx.finditer(text):
            n = m.group(1)
            if (len(n) < 3 or n in _KEYWORDS or n in _COMMON_METHODS
                    or (n.startswith("__") and n.endswith("__"))):
                continue
            counts[n] = counts.get(n, 0) + 1
    defined_here = {d for t in hunk.added + hunk.old_lines
                    if (d := _def_name(t, language)) is not None}
    order = sorted(counts, key=lambda n: (-counts[n], text.find(n)))
    return [n for n in order if n not in defined_here]


def _def_name(line: str, language: str) -> str | None:
    m = _DEF[language].match(line)
    if m:
        return next((g for g in m.groupdict().values() if g), None)
    if language in ("typescript", "javascript"):
        m = _JS_METHOD.match(line)
        if m and not line.lstrip().startswith(("if", "for", "while", "switch", "catch", "return")):
            return m.group("name")
    return None


def candidate_files(repo_dir: Path, file_path: str, language: str) -> list[tuple[str, Path]]:
    """Same file, then the same directory, then the parent directory; only
    this language's source files, bounded in count and bytes."""
    exts = EXTENSIONS[language]
    out: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    own = safe_path(repo_dir, file_path)
    if own is not None and own.is_file():
        out.append((file_path, own))
        seen.add(own)
    rel_dir = Path(file_path).parent
    budget = MAX_SCAN_BYTES
    for d in (rel_dir, rel_dir.parent) if str(rel_dir) not in ("", ".") else (rel_dir,):
        p = safe_path(repo_dir, str(d))
        if p is None or not p.is_dir():
            continue
        try:
            entries = sorted(p.iterdir())
        except OSError:
            continue
        for f in entries:
            if len(out) >= MAX_SCAN_FILES or budget <= 0:
                return out
            if f in seen or f.suffix not in exts or not f.is_file() or f.is_symlink():
                continue
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES:
                continue
            budget -= size
            seen.add(f)
            out.append((f.relative_to(repo_dir.resolve()).as_posix(), f))
    return out


def find_definitions(repo_dir: Path, file_path: str, language: str, names: list[str],
                     base: list[str] | None, skip_lines: set[int],
                     limit: int = MAX_DEFINITIONS) -> list[tuple[str, int, list[str]]]:
    """(relative path, 1-based line, rendered lines) for each name found,
    nearest scope first; `skip_lines` excludes the enclosing definitions
    already shown."""
    wanted = list(dict.fromkeys(names))
    found: dict[str, tuple[str, int, list[str]]] = {}
    for rel, path in candidate_files(repo_dir, file_path, language):
        lines = base if rel == file_path and base is not None else read_lines(path)
        if not lines:
            continue
        for i, line in enumerate(lines):
            n = _def_name(line, language)
            if n is None or n not in wanted or n in found:
                continue
            if rel == file_path and i in skip_lines:
                continue
            sig = signature(lines, i, language)
            doc = doc_for(lines, i, len(sig), language)
            block = (doc + sig) if language != "python" else (sig + doc)
            found[n] = (rel, i + 1, block)
        if len(found) >= limit or all(n in found for n in wanted):
            break
    return [found[n] for n in wanted if n in found][:limit]


# ------------------------------------------------------------------ build

Section = tuple[str, list[str]]       # (heading, whole items)


def code_sections(repo_dir: str | Path, file_path: str, diff_hunk: str,
                  language: str) -> dict[str, Section]:
    """{"enclosing": ..., "definitions": ...} for one hunk; absent keys mean
    nothing was found. Pure function of the files on disk."""
    language = (language or "").strip().lower()
    root = Path(repo_dir)
    hunk = parse_hunk(diff_hunk) if isinstance(diff_hunk, str) else None
    if language not in EXTENSIONS or hunk is None or not root.is_dir():
        return {}

    own = safe_path(root, file_path)
    base = read_lines(own) if own is not None else None
    out: dict[str, Section] = {}
    skip: set[int] = set()
    if base:
        at = locate(base, hunk)
        if at is not None:
            enc = render_enclosing(base, at, hunk, language)
            if enc:
                out["enclosing"] = (f"## Enclosing code in {file_path} (before this change)",
                                    [enc])
                skip = set(enclosing(base, at, hunk.first_indent, language))

    names = used_names(hunk, language)
    if names:
        items = []
        for rel, line_no, block in find_definitions(root, file_path, language, names,
                                                    base, skip):
            where = "this file" if rel == file_path else rel
            items.append(f"- {where}, line {line_no}:\n"
                         + "\n".join("    " + b.strip() for b in block))
        if items:
            out["definitions"] = ("## Definitions used by this change", items)
    return out


def code_context(repo_dir: str | Path, file_path: str, diff_hunk: str, language: str,
                 budget_chars: int = DEFAULT_BUDGET_CHARS) -> str:
    """Code sections only, assembled: plain text, or "" when nothing was found."""
    secs = code_sections(repo_dir, file_path, diff_hunk, language)
    return assemble([secs[k] for k in ("enclosing", "definitions") if k in secs],
                    budget_chars)


def assemble(sections: list[tuple[str, list[str]]], budget_chars: int) -> str:
    """Whole sections and whole items only, in priority order, under budget."""
    out: list[str] = []
    size = 0
    for heading, items in sections:
        kept: list[str] = []
        cost = len(heading) + 2
        for it in items:
            if size + cost + len(it) + 1 > budget_chars:
                break
            kept.append(it)
            cost += len(it) + 1
        if kept:
            out.append("\n".join([heading, *kept]))
            size += cost
    return "\n\n".join(out)
