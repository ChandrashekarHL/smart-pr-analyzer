"""Chunking and the ChromaDB code index. Chunking runs anywhere; the index
test needs requirements-rag.txt (run it in the RAG environment)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prana.rag import embed


def test_chunks_start_at_definitions_and_respect_limits():
    src = "import os\n\n" + "".join(
        f"def f{i}(x):\n" + "".join(f"    y{j} = x\n" for j in range(12)) + "    return x\n\n"
        for i in range(4))
    chunks = embed.chunk_file("m.py", src, "python")
    assert [c.text.splitlines()[0] for c in chunks[1:]] == ["def f1(x):", "def f2(x):",
                                                             "def f3(x):"]
    assert chunks[0].start == 1 and "def f0(x):" in chunks[0].text   # short header merged
    assert all(c.end - c.start + 1 <= embed.CHUNK_MAX_LINES for c in chunks)


def test_long_definition_is_split():
    src = "def big():\n" + "".join(f"    v{i} = {i}\n" for i in range(100))
    chunks = embed.chunk_file("m.py", src, "python")
    assert len(chunks) == 3 and chunks[1].start == embed.CHUNK_MAX_LINES + 1


def test_repo_files_skip_vendored_minified_and_rank_focus_first(tmp_path):
    for rel in ("a/x.py", "b/y.py", "b/sub/z.py", "node_modules/m.js", "web/app.min.js",
                "web/app.js", "README.md"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x = 1\n")
    files = embed.repo_files(tmp_path, focus=["b/y.py"])
    paths = [p for p, _ in files]
    assert paths[0] == "b/y.py" and paths[1] == "b/sub/z.py"
    assert "node_modules/m.js" not in paths and "web/app.min.js" not in paths
    assert "README.md" not in paths and ("web/app.js", "javascript") in files


def test_build_and_query_real_index(tmp_path):
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg/db.py").write_bytes(
        b"def connect(url):\n    return Engine(url)\n\n"
        b"def run_query(conn, sql, params):\n    cur = conn.cursor()\n"
        b"    cur.execute(sql, params)\n    return cur.fetchall()\n")
    (repo / "pkg/text.py").write_bytes(
        b"def slugify(title):\n    return title.lower().replace(' ', '-')\n")
    meta = embed.build(repo, tmp_path / "idx", repo="o/r", sha="abc")
    assert meta["chunks"] >= 2 and meta["model"] == embed.MODEL_NAME
    assert json.loads((tmp_path / "idx/meta.json").read_text())["sha"] == "abc"
    idx = embed.CodeIndex(tmp_path / "idx")
    hits = idx.query("cursor.execute(sql, params)\nrows = cursor.fetchall()", "python", k=2,
                     max_distance=1.0)
    assert hits and hits[0].path == "pkg/db.py"
    assert idx.query("x", "python", exclude_path="pkg/db.py", k=5, max_distance=1.0)[0].path \
        == "pkg/text.py"
    assert idx.query("cursor.execute(sql)", "go", max_distance=1.0) == []   # language filter
    assert Path(tmp_path / "idx/conventions.json").exists()
