"""The retrieval contract: code context from the repository at the base commit.

Every guarantee in `prana.rag.contract` has a test here: never raises, bounded
time, bounded size, no hunk or review text, local scope, determinism."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from prana.rag import code, contract
from prana.rag.contract import retrieve


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("utf-8"))


def _call(repo_dir, file_path, hunk, language, **kw):
    return retrieve("o/r", "a" * 40, file_path, hunk, language, 7,
                    str(repo_dir) if repo_dir is not None else None, **kw)


PY_STORE = '''import json


class Store:
    """Key-value store backed by a JSON file."""

    def __init__(self, path):
        self.path = path

    def load(self, key, default=None):
        """Return the value for key."""
        data = self._read()
        if key not in data:
            return default
        value = data[key]
        return value

    def _read(self):
        with open(self.path) as f:
            return json.load(f)
'''

PY_UTIL = '''def normalise_key(key: str) -> str:
    """Lower-case and strip a key."""
    return key.strip().lower()
'''

PY_HUNK = """@@ -15,2 +15,3 @@ class Store:
         value = data[key]
-        return value
+        key = normalise_key(key)
+        return validate_value(value)
"""


@pytest.fixture
def pyrepo(tmp_path):
    _write(tmp_path, "pkg/store.py", PY_STORE)
    _write(tmp_path, "pkg/util.py", PY_UTIL)
    _write(tmp_path, "validate.py",
           "def validate_value(v):\n    # reject None\n    return v\n")
    return tmp_path


# ------------------------------------------------------------- content

def test_python_enclosing_class_and_method_with_body_above_the_hunk(pyrepo):
    out = _call(pyrepo, "pkg/store.py", PY_HUNK, "python")
    assert out.startswith("## Enclosing code in pkg/store.py (before this change)")
    assert "class Store:" in out
    assert "def load(self, key, default=None):" in out
    assert out.count('"""Return the value for key."""') == 1
    assert "if key not in data:" in out          # body above the hunk, not in the diff
    assert "value = data[key]" not in out        # hunk context lines are not repeated


def test_multiline_docstring_is_not_shown_as_body(tmp_path):
    _write(tmp_path, "m.py", 'def run(job):\n    """\n    Run a job.\n\n    Long text.\n'
                             '    """\n    prepare(job)\n    start(job)\n    wait(job)\n')
    hunk = "@@ -8,2 +8,3 @@ def run(job):\n     start(job)\n     wait(job)\n+    log(job)\n"
    out = _call(tmp_path, "m.py", hunk, "python")
    assert "def run(job):" in out and "prepare(job)" in out
    assert "Long text." not in out


def test_definitions_found_in_same_dir_and_parent_dir(pyrepo):
    out = _call(pyrepo, "pkg/store.py", PY_HUNK, "python")
    assert "## Definitions used by this change" in out
    assert "pkg/util.py, line 1:" in out and "def normalise_key(key: str) -> str:" in out
    assert '"""Lower-case and strip a key."""' in out
    assert "validate.py, line 1:" in out          # parent directory


def test_output_never_contains_the_added_lines(pyrepo):
    out = _call(pyrepo, "pkg/store.py", PY_HUNK, "python")
    assert "key = normalise_key(key)" not in out
    assert "return validate_value(value)" not in out


def test_header_line_number_wrong_is_located_by_content(pyrepo):
    moved = PY_HUNK.replace("@@ -15,2 +15,3 @@", "@@ -40,2 +40,3 @@")
    assert "def load(self, key, default=None):" in _call(pyrepo, "pkg/store.py", moved,
                                                          "python")


def test_top_level_hunk_has_no_enclosing_definition(tmp_path):
    _write(tmp_path, "m.py", "def fetch():\n    return 1\n\nX = 1\nY = 2\n")
    hunk = "@@ -4,2 +4,3 @@\n X = 1\n Y = 2\n+Z = fetch()\n"
    out = _call(tmp_path, "m.py", hunk, "python")
    assert "Enclosing" not in out
    assert "this file, line 1:" in out and "def fetch():" in out


def test_go_method_and_function_in_sibling_file(tmp_path):
    _write(tmp_path, "db/conn.go",
           "package db\n\n// Conn wraps a pool.\ntype Conn struct {\n\tpool *Pool\n}\n\n"
           "// Query runs q.\nfunc (c *Conn) Query(ctx context.Context, q string) (*Rows, error) {\n"
           "\tif q == \"\" {\n\t\treturn nil, ErrEmpty\n\t}\n\trows, err := c.pool.Run(ctx, q)\n"
           "\treturn rows, err\n}\n")
    _write(tmp_path, "db/retry.go",
           "package db\n\n// withRetry retries f up to n times.\nfunc withRetry(n int, f func() error) error {\n"
           "\treturn f()\n}\n")
    hunk = ("@@ -13,2 +13,3 @@ func (c *Conn) Query(ctx context.Context, q string) (*Rows, error) {\n"
            " \trows, err := c.pool.Run(ctx, q)\n+\terr = withRetry(3, func() error { return err })\n"
            " \treturn rows, err\n")
    out = _call(tmp_path, "db/conn.go", hunk, "go")
    assert "func (c *Conn) Query(ctx context.Context, q string) (*Rows, error) {" in out
    assert "// Query runs q." in out
    assert "db/retry.go, line 4:" in out and "// withRetry retries f up to n times." in out


def test_typescript_class_method_and_exported_function(tmp_path):
    _write(tmp_path, "src/api/client.ts",
           "export class Client {\n  constructor(private base: string) {}\n\n"
           "  async get(path: string): Promise<Response> {\n    const url = this.base + path;\n"
           "    const res = await fetch(url);\n    return res;\n  }\n}\n")
    _write(tmp_path, "src/api/retry.ts",
           "/** Retry a promise-returning function. */\n"
           "export async function withBackoff<T>(fn: () => Promise<T>, tries = 3): Promise<T> {\n"
           "  return fn();\n}\n")
    hunk = ("@@ -6,2 +6,2 @@ export class Client {\n-    const res = await fetch(url);\n"
            "+    const res = await withBackoff(() => fetch(url));\n     return res;\n")
    out = _call(tmp_path, "src/api/client.ts", hunk, "typescript")
    assert "export class Client {" in out
    assert "async get(path: string): Promise<Response> {" in out
    assert "src/api/retry.ts, line 2:" in out
    assert "/** Retry a promise-returning function. */" in out


def test_rust_fn_inside_impl(tmp_path):
    _write(tmp_path, "src/cache.rs",
           "pub struct Cache { map: HashMap<String, Entry> }\n\nimpl Cache {\n"
           "    /// Fetch an entry.\n    pub fn get(&self, k: &str) -> Option<&Entry> {\n"
           "        let e = self.map.get(k)?;\n        Some(e)\n    }\n}\n\n"
           "/// True when the entry has expired.\npub fn is_stale(e: &Entry) -> bool {\n    false\n}\n")
    hunk = ("@@ -6,2 +6,3 @@ impl Cache {\n         let e = self.map.get(k)?;\n"
            "+        if is_stale(e) { return None; }\n         Some(e)\n")
    out = _call(tmp_path, "src/cache.rs", hunk, "rust")
    assert "impl Cache {" in out and "pub fn get(&self, k: &str) -> Option<&Entry> {" in out
    assert "/// Fetch an entry." in out
    assert "this file, line 12:" in out and "/// True when the entry has expired." in out


def test_new_file_gets_definitions_from_its_directory(tmp_path):
    _write(tmp_path, "pkg/util.py", PY_UTIL)
    hunk = "@@ -0,0 +1,2 @@\n+def go(k):\n+    return normalise_key(k)\n"
    out = _call(tmp_path, "pkg/new.py", hunk, "python")
    assert "Enclosing" not in out and "pkg/util.py, line 1:" in out


def test_names_defined_in_the_hunk_are_not_looked_up(tmp_path):
    _write(tmp_path, "a.py", "def helper():\n    pass\n")
    hunk = "@@ -0,0 +1,3 @@\n+def helper():\n+    return 2\n+helper()\n"
    assert _call(tmp_path, "b.py", hunk, "python") == ""


def test_long_python_signature_then_docstring_then_body(tmp_path):
    """Regression (test_ood sample): a 9-line signature was cut at 5 lines and
    the docstring then showed up as body."""
    params = "".join(f"    p{i}: int,\n" for i in range(7))
    _write(tmp_path, "m.py", "def choose(\n" + params + ') -> int:\n    """Pick one.\n\n'
                             '    :param p0: secret detail\n    """\n    total = p0\n    return total\n')
    hunk = "@@ -14,1 +14,2 @@ def choose(\n     return total\n+    log(total)\n"
    out = _call(tmp_path, "m.py", hunk, "python")
    assert ") -> int:" in out and '"""Pick one.' in out and "total = p0" in out
    assert "secret detail" not in out


def test_js_methods_defined_in_hunk_and_common_names_are_not_looked_up(tmp_path):
    """Regression (test_ood sample): a hunk defining its own `constructor`
    pulled in an unrelated class's constructor from a sibling file."""
    _write(tmp_path, "geo/circle.js", "class Circle {\n  constructor(radius) {\n  }\n"
                                      "  distance(a, b) {\n  }\n}\n")
    hunk = ("@@ -0,0 +1,6 @@\n+class Pair {\n+  constructor(points) {\n+    this.p = points.map(x => x)\n"
            "+  }\n+  distance(a, b) {\n+  }\n")
    assert _call(tmp_path, "geo/pair.js", hunk, "javascript") == ""


def test_oversized_enclosing_keeps_body_lines_nearest_the_hunk(tmp_path):
    body = "".join(f"    value_{i} = compute_something_long(argument_{i}, other_{i})  # {'x' * 40}\n"
                   for i in range(8))
    _write(tmp_path, "m.py", "def big(a, b):\n" + body + "    return a\n")
    hunk = ("@@ -9,1 +9,2 @@ def big(a, b):\n     value_7 = compute_something_long(argument_7, other_7)"
            f"  # {'x' * 40}\n+    y = 1\n")
    out = _call(tmp_path, "m.py", hunk, "python")
    assert "def big(a, b):" in out and "value_6 =" in out    # nearest line above the hunk
    assert "value_0 =" not in out and "lines)" in out
    assert len(out.split("\n\n")[0]) <= code.ENCLOSING_MAX_CHARS + 80


# ------------------------------------------------------------- bounds

def test_budget_is_respected_with_whole_items_only(pyrepo):
    full = _call(pyrepo, "pkg/store.py", PY_HUNK, "python")
    small = _call(pyrepo, "pkg/store.py", PY_HUNK, "python", budget_chars=400)
    assert len(small) <= 400 < len(full)
    assert small and not small.endswith(("(", ","))


def test_long_body_is_elided_and_capped(tmp_path):
    body = "".join(f"    x{i} = {i}\n" for i in range(200))
    _write(tmp_path, "m.py", "def big(a, b):\n" + body + "    return a\n")
    hunk = "@@ -201,1 +201,2 @@ def big(a, b):\n     x199 = 199\n+    y = 1\n"
    out = _call(tmp_path, "m.py", hunk, "python")
    assert "def big(a, b):" in out and "lines)" in out
    assert len(out) <= code.DEFAULT_BUDGET_CHARS


def test_deterministic(pyrepo):
    assert _call(pyrepo, "pkg/store.py", PY_HUNK, "python") == \
        _call(pyrepo, "pkg/store.py", PY_HUNK, "python")


# ------------------------------------------------------------- never raises

def test_no_checkout_returns_empty_silently(caplog):
    with caplog.at_level(logging.WARNING, logger="prana.rag"):
        assert _call(None, "a.py", PY_HUNK, "python") == ""
    assert caplog.records == []


def test_empty_repo_returns_empty(tmp_path):
    assert _call(tmp_path, "pkg/store.py", PY_HUNK, "python") == ""


def test_missing_repo_dir_returns_empty(tmp_path):
    assert _call(tmp_path / "nope", "pkg/store.py", PY_HUNK, "python") == ""


@pytest.mark.parametrize("lang", ["java", "", None, "PYTHON "])
def test_unsupported_or_odd_language(pyrepo, lang):
    out = _call(pyrepo, "pkg/store.py", PY_HUNK, lang)
    assert out == "" if lang != "PYTHON " else out.startswith("## Enclosing")


@pytest.mark.parametrize("hunk", ["", "not a hunk", "@@ broken @@\n+x", None, 42])
def test_garbage_hunk(pyrepo, hunk):
    assert _call(pyrepo, "pkg/store.py", hunk, "python") == ""


def test_path_traversal_never_reads_outside_the_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(tmp_path, "secret.py", "def leaked_secret():\n    pass\n")
    hunk = "@@ -1,1 +1,2 @@ def leaked_secret():\n     pass\n+leaked_secret()\n"
    for fp in ("../secret.py", str(tmp_path / "secret.py"), "a/../../secret.py"):
        assert "leaked_secret" not in _call(repo, fp, hunk, "python")


def test_binary_and_oversized_files_are_skipped(tmp_path):
    _write(tmp_path, "pkg/blob.py", "def normalise_key():\n")
    (tmp_path / "pkg/blob.py").write_bytes(b"def normalise_key():\x00\x01")
    _write(tmp_path, "pkg/huge.py",
           "def normalise_key():\n" + "#" * (code.MAX_FILE_BYTES + 10) + "\n")
    hunk = "@@ -0,0 +1,1 @@\n+normalise_key(k)\n"
    assert _call(tmp_path, "pkg/new.py", hunk, "python") == ""


def test_timeout_returns_empty_and_logs_one_line(pyrepo, monkeypatch, caplog):
    monkeypatch.setattr(contract, "build_sections", lambda *a, **k: time.sleep(1.0) or {})
    t = time.perf_counter()
    with caplog.at_level(logging.WARNING, logger="prana.rag"):
        out = _call(pyrepo, "pkg/store.py", PY_HUNK, "python", timeout_s=0.1)
    assert out == "" and time.perf_counter() - t < 0.6
    assert len(caplog.records) == 1 and '"event": "timeout"' in caplog.records[0].message


def test_internal_error_returns_empty_and_logs(pyrepo, monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(contract, "build_sections", boom)
    with caplog.at_level(logging.WARNING, logger="prana.rag"):
        assert _call(pyrepo, "pkg/store.py", PY_HUNK, "python") == ""
    assert '"event": "error"' in caplog.records[0].message
    assert "disk on fire" in caplog.records[0].message


def test_requirements_are_declared():
    r = contract.REQUIREMENTS
    assert "base_sha" in r["checkout"] and "ChromaDB" in r["index"]
    assert "commits" in r["history"]
    assert set(r["languages"]) == {"python", "go", "rust", "typescript", "javascript"}


# ------------------------------------------------- history, index, ablations

from prana.rag.embed import Hit  # noqa: E402  (pure dataclass; no heavy imports)
from prana.rag.history import Commit, FileHistory, PastReview  # noqa: E402

HIST = FileHistory(
    reviews=[PastReview("3", "2026-01-01T00:00:00Z", "Use a context manager for the file."),
             PastReview("5", "2026-03-01T00:00:00Z", "Normalise keys before lookup."),
             PastReview("7", "2026-04-01T00:00:00Z", "THIS PR - must never appear"),
             PastReview("4", "2026-02-01T00:00:00Z", "Avoid   bare\nexcept."),
             PastReview("2", "2025-12-01T00:00:00Z", "oldest, dropped by the limit")],
    commits=[Commit("a1", "2026-02-10T00:00:00Z", "Add Store.load\n\nlong body"),
             Commit("b2", "2026-03-15T00:00:00Z", "Fix key lookup")])


class FakeIndex:
    rules = {"python": ["Function names are snake_case (40 of 41 public functions)."]}

    def __init__(self):
        self.queries = []

    def query(self, text, language, k=5, exclude_path=None, max_distance=0.6):
        self.queries.append((text, language, exclude_path))
        return [Hit("pkg/cache.py", 10, 20, "def load_cached(key):\n    return CACHE[key]", 0.2)]


def test_past_reviews_newest_first_never_this_pr_at_most_three(tmp_path):
    out = retrieve("o/r", "a" * 40, "pkg/store.py", PY_HUNK, "python", 7, None, history=HIST)
    sec = out.split("## Recent commits")[0]
    assert sec.startswith("## Past reviews on pkg/store.py")
    assert "THIS PR" not in out and "oldest" not in out
    assert sec.index("PR #5") < sec.index("PR #4") < sec.index("PR #3")
    assert "Avoid bare except." in sec                    # whitespace collapsed


def test_before_drops_reviews_from_the_future():
    out = retrieve("o/r", "a" * 40, "pkg/store.py", PY_HUNK, "python", 9, None,
                   history=HIST, before="2026-02-15T00:00:00Z")
    assert "PR #4" in out and "PR #3" in out and "PR #5" not in out and "PR #7" not in out


def test_git_history_newest_first_first_line_only():
    out = retrieve("o/r", "a" * 40, "pkg/store.py", PY_HUNK, "python", 7, None, history=HIST)
    hist = out.split("## Recent commits to pkg/store.py\n")[1]
    assert hist.splitlines() == ["- 2026-03-15 Fix key lookup", "- 2026-02-10 Add Store.load"]


def test_similar_code_and_conventions_from_the_index(pyrepo):
    idx = FakeIndex()
    out = _call(pyrepo, "pkg/store.py", PY_HUNK, "python", index=idx)
    assert "## Similar code elsewhere in this repository" in out
    assert "pkg/cache.py, lines 10-20:" in out and "def load_cached(key):" in out
    assert "## Conventions in this repository (measured)" in out and "snake_case" in out
    text, lang, excluded = idx.queries[0]
    assert "normalise_key(key)" in text and lang == "python" and excluded == "pkg/store.py"


def test_sections_follow_the_design_doc_priority(pyrepo):
    out = _call(pyrepo, "pkg/store.py", PY_HUNK, "python", history=HIST, index=FakeIndex())
    heads = [ln for ln in out.splitlines() if ln.startswith("## ")]
    order = ["## Enclosing", "## Past reviews", "## Definitions", "## Similar code",
             "## Conventions", "## Recent commits"]
    assert [next(i for i, o in enumerate(order) if h.startswith(o)) for h in heads] == \
        sorted(range(len(heads)))
    assert len(heads) == 6


def test_budget_drops_lowest_priority_sources_first(pyrepo):
    full = _call(pyrepo, "pkg/store.py", PY_HUNK, "python", history=HIST, index=FakeIndex())
    small = _call(pyrepo, "pkg/store.py", PY_HUNK, "python", history=HIST, index=FakeIndex(),
                  budget_chars=700)
    assert len(small) <= 700 < len(full)
    # Higher-priority sections claim the budget first; a large lower-priority
    # section loses out (small items may still fill what is left).
    assert small.startswith("## Enclosing") and "## Similar code" not in small
    assert "## Similar code" in full


@pytest.mark.parametrize("off", sorted(contract.ALL_SOURCES))
def test_each_source_can_be_switched_off_for_ablations(pyrepo, off):
    heading = {"enclosing": "## Enclosing", "past_reviews": "## Past reviews",
               "definitions": "## Definitions", "similar_code": "## Similar code",
               "conventions": "## Conventions", "git_history": "## Recent commits"}
    out = _call(pyrepo, "pkg/store.py", PY_HUNK, "python", history=HIST, index=FakeIndex(),
                sources=contract.ALL_SOURCES - {off}, budget_chars=10_000)
    assert heading[off] not in out
    assert all(h in out for k, h in heading.items() if k != off)


def test_index_path_without_a_built_index_is_ignored(pyrepo, tmp_path):
    out = _call(pyrepo, "pkg/store.py", PY_HUNK, "python", index=str(tmp_path / "none"))
    assert out.startswith("## Enclosing") and "## Similar" not in out


def test_index_failure_keeps_the_other_sources_and_logs(pyrepo, caplog):
    class Broken(FakeIndex):
        def query(self, *a, **k):
            raise RuntimeError("chroma down")
    with caplog.at_level(logging.WARNING, logger="prana.rag"):
        out = _call(pyrepo, "pkg/store.py", PY_HUNK, "python", index=Broken(), history=HIST)
    assert out.startswith("## Enclosing") and "## Past reviews" in out
    assert "## Similar code" not in out
    assert '"source": "index"' in caplog.records[0].message
    assert "chroma down" in caplog.records[0].message
