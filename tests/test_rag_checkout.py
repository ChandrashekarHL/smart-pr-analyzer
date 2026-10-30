"""Sparse checkouts and the annotator, against a real local git origin."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from prana.rag import checkout as co
from prana.rag.code_annotate import annotate


def test_sparse_patterns_cover_dir_and_parent_direct_files_only():
    pats = co.sparse_patterns(["a/b/c.py", "a/b/d.py", "top.py"])
    assert pats == ["/*", "!/*/", "/a/*", "!/a/*/", "/a/b/*", "!/a/b/*/"]


def test_plan_groups_rows_by_repo_and_base_sha():
    rows = [{"repo_name": "o/r", "pr_number": 1, "file_path": "x/a.py"},
            {"repo_name": "o/r", "pr_number": 1, "file_path": "x/b.py"},
            {"repo_name": "o/r", "pr_number": 2, "file_path": "y.py"},
            {"repo_name": "o/q", "pr_number": 9, "file_path": "z.py"}]   # no sha
    need = co.plan(rows, {("o/r", "1"): "s1", ("o/r", "2"): "s2"})
    assert need == {("o/r", "s1"): ["x/a.py", "x/b.py"], ("o/r", "s2"): ["y.py"]}


def _g(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def origin(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    src = tmp_path / "src"
    src.mkdir()
    _g(src, "init", "-q")
    _g(src, "config", "user.email", "t@example.com")
    _g(src, "config", "user.name", "t")
    files = {
        "pkg/store.py": "class Store:\n    def load(self, key):\n        data = {}\n"
                        "        value = data[key]\n        return value\n",
        "pkg/util.py": "def normalise_key(key):\n    return key\n",
        "pkg/deep/skip.py": "def not_needed():\n    pass\n",
        "other/far.py": "def far_away():\n    pass\n",
        "root.py": "def at_root():\n    pass\n",
    }
    for rel, text in files.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
    _g(src, "add", "-A")
    _g(src, "commit", "-q", "-m", "base")
    sha = _g(src, "rev-parse", "HEAD")
    _g(src, "config", "uploadpack.allowFilter", "true")
    _g(src, "config", "uploadpack.allowAnySHA1InWant", "true")
    return src, sha


def test_checkout_is_sparse_and_idempotent(origin, tmp_path):
    src, sha = origin
    dest = co.dest_for(tmp_path / "cache", "o/r", sha)
    rec = co.checkout("o/r", sha, ["pkg/store.py"], dest, url=src.as_uri())
    assert (dest / "pkg/store.py").is_file() and (dest / "pkg/util.py").is_file()
    assert (dest / "root.py").is_file()                  # parent dir of pkg/
    assert not (dest / "pkg/deep/skip.py").exists()      # subdirectories excluded
    assert not (dest / "other/far.py").exists()
    assert rec["files"] == 3 and (dest / co.MARKER).exists()
    assert co.checkout("o/r", sha, ["pkg/store.py"], dest, url="file:///nowhere") == rec


def test_annotate_end_to_end(origin, tmp_path):
    src, sha = origin
    root = tmp_path / "cache"
    co.checkout("o/r", sha, ["pkg/store.py"], co.dest_for(root, "o/r", sha), url=src.as_uri())
    hunk = ("@@ -4,2 +4,3 @@ class Store:\n         value = data[key]\n"
            "+        key = normalise_key(key)\n         return value\n")
    rows = [{"id": "a", "repo_name": "o/r", "pr_number": 1, "file_path": "pkg/store.py",
             "diff_hunk": hunk, "language": "python"},
            {"id": "b", "repo_name": "o/r", "pr_number": 2, "file_path": "pkg/store.py",
             "diff_hunk": hunk, "language": "python"}]            # no sha for PR 2
    out, stats = annotate(rows, {("o/r", "1"): sha}, root)
    assert "def load(self, key):" in out[0]["rag_context"]
    assert "pkg/util.py, line 1:" in out[0]["rag_context"]
    assert out[1]["rag_context"] == ""
    assert stats["outcome"] == {"context": 1, "no_base_sha": 1}
    assert stats["rows_per_source"]["enclosing"] == 1
    assert stats["rows_per_source"]["definitions"] == 1
    assert stats["rows_per_source"]["past_reviews"] == 0


def test_annotate_with_history_respects_pr_and_time(tmp_path):
    from prana.rag.code_annotate import load_commits, load_review_store
    from prana.rag.contract import ALL_SOURCES
    from prana.data import jsonl as jl

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [  # (id, pr, file, comment)
        ("o/r_1-11", 1, "a.py", "Old advice on a.py from PR 1."),
        ("o/r_2-21", 2, "a.py", "Advice on a.py from PR 2."),
        ("o/r_3-31", 3, "a.py", "Target comment; PR 3 itself."),
        ("o/r_3-32", 3, "a.py", "Sibling comment in PR 3 - must not leak."),
        ("o/r_4-41", 4, "a.py", "Later PR 4 - from the future."),
        ("o/r_5-51", 5, "b.py", "Other file."),
    ]
    when = {"o/r_1-11": "2026-01-01T00:00:00Z", "o/r_2-21": "2026-02-01T00:00:00Z",
            "o/r_3-31": "2026-03-01T00:00:00Z", "o/r_3-32": "2026-03-02T00:00:00Z",
            "o/r_4-41": "2026-04-01T00:00:00Z", "o/r_5-51": "2026-01-15T00:00:00Z"}
    jl.write(corpus / "test_ood.jsonl", [
        {"id": i, "repo_name": "o/r", "pr_number": pr, "file_path": fp, "review_comment": c,
         "diff_hunk": "@@ -1,1 +1,2 @@\n x = 1\n+y = 2\n", "language": "python"}
        for i, pr, fp, c in rows])
    jl.write(tmp_path / "when.jsonl", [{"id": k, "created_at": v} for k, v in when.items()])
    jl.write(tmp_path / "commits.jsonl", [
        {"repo": "o/r", "sha": "s3", "path": "a.py",
         "commits": [{"sha": "c1", "date": "2026-02-20T00:00:00Z", "message": "Tidy a.py"}]}])
    store = load_review_store(corpus, tmp_path / "when.jsonl")
    target = [r for r in jl.read(corpus / "test_ood.jsonl") if r["id"] == "o/r_3-31"]
    out, stats = annotate(target, {("o/r", "3"): "s3"}, None, store=store,
                          commits=load_commits(tmp_path / "commits.jsonl"),
                          sources=ALL_SOURCES)
    ctx = out[0]["rag_context"]
    assert "Advice on a.py from PR 2." in ctx and "Old advice on a.py from PR 1." in ctx
    assert "PR 3" not in ctx and "future" not in ctx and "Other file" not in ctx
    assert "- 2026-02-20 Tidy a.py" in ctx
    assert stats["rows_per_source"] == {"enclosing": 0, "past_reviews": 1, "definitions": 0,
                                        "similar_code": 0, "conventions": 0, "git_history": 1}


def test_sha_lookup_survives_a_hung_gh_call(monkeypatch):
    """Regression: one `gh api` call hung past its timeout and the
    TimeoutExpired killed the whole lookup run."""
    import subprocess as sp

    from prana.rag import shas

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            raise sp.TimeoutExpired(cmd, 60)
        return sp.CompletedProcess(cmd, 0, stdout='{"base_sha": "abc"}', stderr="")

    monkeypatch.setattr(shas.subprocess, "run", fake_run)
    monkeypatch.setattr(shas.time, "sleep", lambda s: None)
    assert shas.lookup("o/r", 5) == {"repo": "o/r", "pr_number": "5", "base_sha": "abc"}
    monkeypatch.setattr(shas.subprocess, "run",
                        lambda cmd, **kw: (_ for _ in ()).throw(sp.TimeoutExpired(cmd, 60)))
    assert shas.lookup("o/r", 6)["error"] == "gh timed out after 60 s"


def test_source_checkout_takes_code_anywhere_but_skips_vendored(origin, tmp_path):
    src, sha = origin
    (src / "node_modules/lib").mkdir(parents=True)
    (src / "node_modules/lib/x.py").write_bytes(b"def vendored():\n    pass\n")
    _g(src, "add", "-A")
    _g(src, "commit", "-q", "-m", "vendored")
    sha2 = _g(src, "rev-parse", "HEAD")
    dest = tmp_path / "full"
    rec = co.checkout_source("o/r", sha2, [".py"], dest, url=src.as_uri())
    got = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*.py")
                 if ".git" not in p.parts)
    assert got == ["other/far.py", "pkg/deep/skip.py", "pkg/store.py", "pkg/util.py", "root.py"]
    assert rec["files"] == 5


def test_earliest_base_is_chosen_by_commit_time(origin, tmp_path):
    from prana.rag.index_repos import earliest_base
    src, sha1 = origin
    (src / "later.py").write_bytes(b"x = 1\n")
    _g(src, "add", "-A")
    env_date = "2030-01-01T00:00:00+05:30"
    subprocess.run(["git", "commit", "-q", "-m", "later"], cwd=src, check=True,
                   env={**__import__("os").environ, "GIT_COMMITTER_DATE": env_date,
                        "GIT_AUTHOR_DATE": env_date})
    sha2 = _g(src, "rev-parse", "HEAD")
    root = tmp_path / "cache"
    for s in (sha2, sha1):
        co.checkout("o/r", s, ["pkg/store.py"], co.dest_for(root, "o/r", s), url=src.as_uri())
    assert co.commit_time(co.dest_for(root, "o/r", sha1)) < \
        co.commit_time(co.dest_for(root, "o/r", sha2))
    assert earliest_base("o/r", {sha1, sha2}, root) == sha1
    assert earliest_base("o/r", {"f" * 40}, root) is None
