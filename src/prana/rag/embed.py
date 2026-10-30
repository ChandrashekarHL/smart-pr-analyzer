"""Semantic code index for one repository (design doc, Module 3.1-3.2):
files chunked, embedded with all-MiniLM-L6-v2 and stored in a persistent
ChromaDB collection; at review time the new code is the query and the top-k
most similar chunks come back.

    python -m prana.rag.embed build --repo-dir cache/full/o__r/<sha> --out cache/index/o__r

Build once per repository (and again when the default branch moves);
`meta.json` records the commit, file and chunk counts, the model and how long
the build took, and `conventions.json` holds the measured repo rules
(`prana.rag.conventions`) so review-time calls never rescan the repository.

Optional dependencies (`requirements-rag.txt`): sentence-transformers, chromadb,
torch (CPU). Imported lazily: everything else in `prana.rag` works without them.
The model loads once per process (~10 s); call `warm()` at worker start-up so
the first review does not pay for it inside its time budget.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from prana.rag import conventions
from prana.rag.code import EXTENSIONS, _is_def

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "code"
CHUNK_MIN_LINES = 8
CHUNK_MAX_LINES = 40
CHUNK_MAX_CHARS = 1600
MAX_CHUNKS = 4000
MAX_FILE_BYTES = 200_000
SKIP_DIRS = frozenset({".git", "node_modules", "vendor", "third_party", "third-party",
                       "dist", "build", "out", "target", "coverage", "__pycache__",
                       ".venv", "venv", "site-packages", ".next", ".nuxt", "bower_components"})
EXT_TO_LANG = {ext: lang for lang, exts in EXTENSIONS.items() for ext in exts}

_model = None
_model_lock = threading.Lock()


def model():
    """The embedding model, loaded once per process."""
    global _model
    with _model_lock:
        if _model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(int(os.environ.get("PRANA_EMBED_THREADS", "8")))
            _model = SentenceTransformer(MODEL_NAME, device="cpu")
        return _model


def warm() -> None:
    model().encode(["warm up"])


@dataclass(frozen=True)
class Chunk:
    path: str
    language: str
    start: int           # 1-based, inclusive
    end: int
    text: str


def chunk_file(path: str, text: str, language: str) -> list[Chunk]:
    """Split at definition lines near the left margin; merge pieces shorter
    than CHUNK_MIN_LINES, split pieces longer than CHUNK_MAX_LINES."""
    lines = text.splitlines()
    if not lines:
        return []
    starts = [0] + [i for i, ln in enumerate(lines)
                    if i and len(ln) - len(ln.lstrip()) <= 4 and _is_def(ln, language)]
    pieces: list[tuple[int, int]] = []
    for a, b in zip(starts, starts[1:] + [len(lines)], strict=True):
        while b - a > CHUNK_MAX_LINES:
            pieces.append((a, a + CHUNK_MAX_LINES))
            a += CHUNK_MAX_LINES
        if pieces and (b - a < CHUNK_MIN_LINES or pieces[-1][1] - pieces[-1][0] < CHUNK_MIN_LINES) \
                and b - pieces[-1][0] <= CHUNK_MAX_LINES:
            pieces[-1] = (pieces[-1][0], b)
        else:
            pieces.append((a, b))
    out = []
    for a, b in pieces:
        body = "\n".join(lines[a:b]).strip("\n")
        if body.strip():
            out.append(Chunk(path, language, a + 1, b, body[:CHUNK_MAX_CHARS]))
    return out


def repo_files(repo_dir: Path, focus: list[str] | None = None) -> list[tuple[str, str]]:
    """(relative path, language) of source files, nearest `focus` paths first."""
    found = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for f in sorted(files):
            ext = os.path.splitext(f)[1]
            if ext not in EXT_TO_LANG or ".min." in f:
                continue
            p = Path(root) / f
            try:
                if p.is_symlink() or p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append((p.relative_to(repo_dir).as_posix(), EXT_TO_LANG[ext]))
    if focus:
        dirs = [Path(x).parent.as_posix() for x in focus]
        tops = {d.split("/")[0] for d in dirs}

        def rank(item: tuple[str, str]) -> tuple[int, str]:
            d = Path(item[0]).parent.as_posix()
            if d in dirs:
                return (0, item[0])
            if any(d.startswith(x + "/") or x.startswith(d + "/") for x in dirs):
                return (1, item[0])
            return (2 if d.split("/")[0] in tops else 3, item[0])
        found.sort(key=rank)
    return found


def build(repo_dir: str | Path, out_dir: str | Path, repo: str = "", sha: str = "",
          focus: list[str] | None = None, max_chunks: int = MAX_CHUNKS) -> dict:
    """Index `repo_dir` into `out_dir`. Deterministic for the same files."""
    import chromadb

    repo_dir, out_dir = Path(repo_dir), Path(out_dir)
    started = time.time()
    chunks: list[Chunk] = []
    by_lang: dict[str, list[tuple[str, str]]] = {}
    files = repo_files(repo_dir, focus)
    used_files = 0
    for rel, lang in files:
        try:
            text = (repo_dir / rel).read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if "\x00" in text[:4096]:
            continue
        by_lang.setdefault(lang, []).append((rel, text))
        if len(chunks) < max_chunks:
            chunks.extend(chunk_file(rel, text, lang)[: max_chunks - len(chunks)])
            used_files += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(out_dir / "chroma"))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001 - absent on a fresh build
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    m = model()
    for i in range(0, len(chunks), 256):
        batch = chunks[i:i + 256]
        emb = m.encode([f"{c.path}\n{c.text}" for c in batch], batch_size=64,
                       normalize_embeddings=True).tolist()
        col.add(ids=[f"{c.path}:{c.start}" for c in batch], embeddings=emb,
                documents=[c.text for c in batch],
                metadatas=[{"path": c.path, "language": c.language, "start": c.start,
                            "end": c.end} for c in batch])
    rules = {lang: conventions.extract(pairs, lang) for lang, pairs in by_lang.items()}
    (out_dir / "conventions.json").write_bytes(
        (json.dumps(rules, indent=2, ensure_ascii=False) + "\n").encode())
    meta = {"repo": repo, "sha": sha, "model": MODEL_NAME, "files_seen": len(files),
            "files_indexed": used_files, "chunks": len(chunks),
            "truncated": len(chunks) >= max_chunks,
            "seconds": round(time.time() - started, 1)}
    (out_dir / "meta.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode())
    return meta


@dataclass(frozen=True)
class Hit:
    path: str
    start: int
    end: int
    text: str
    distance: float


class CodeIndex:
    """A built index, opened read-only. Cheap to open; cache one per repo."""

    def __init__(self, out_dir: str | Path):
        import chromadb
        self.dir = Path(out_dir)
        self._col = chromadb.PersistentClient(path=str(self.dir / "chroma")).get_collection(
            COLLECTION)
        conv = self.dir / "conventions.json"
        self.rules: dict[str, list[str]] = json.loads(conv.read_text("utf-8")) \
            if conv.exists() else {}

    def query(self, text: str, language: str, k: int = 5, exclude_path: str | None = None,
              max_distance: float = 0.6) -> list[Hit]:
        emb = model().encode([text], normalize_embeddings=True).tolist()
        res = self._col.query(query_embeddings=emb, n_results=k + 10,
                              where={"language": language})
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                   res["distances"][0], strict=True):
            if exclude_path and meta["path"] == exclude_path:
                continue
            if dist > max_distance:
                continue
            hits.append(Hit(meta["path"], int(meta["start"]), int(meta["end"]), doc, dist))
            if len(hits) == k:
                break
        return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--repo-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--repo", default="")
    b.add_argument("--sha", default="")
    b.add_argument("--max-chunks", type=int, default=MAX_CHUNKS)
    args = ap.parse_args()
    print(json.dumps(build(args.repo_dir, args.out, args.repo, args.sha,
                           max_chunks=args.max_chunks), indent=2))


if __name__ == "__main__":
    main()
