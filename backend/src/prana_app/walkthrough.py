"""PR walkthrough: one conversation comment saying what the pull request does,
file by file, and what deserves a careful look - the summary a reviewer reads
before the inline comments.

    status = post_walkthrough(gh, generate, repo, number, head_sha, title, files)

Written by the *base* DeepSeek-Coder (the LoRA adapter off, same weights and
GPU, so no code leaves the machine): the fine-tuned reviewer only knows how to
emit one finding per hunk. The model writes the prose; everything checkable
comes from GitHub's data, not from the model:
- the file list and the +/- line counts are GitHub's;
- a model row for a file the PR does not touch is dropped;
- if nothing usable comes back, nothing is posted.

One comment per PR, found by its marker and edited on every new commit, so a
long-running PR does not collect a walkthrough per push.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger("prana_app.walkthrough")

GenerateFn = Callable[[list[dict]], str]     # chat messages -> generated text

MARKER = "<!-- prana:walkthrough -->"
MAX_DIFF_CHARS = 6000          # ~2k tokens of diff keeps the prompt inside the window
MAX_RISKS = 4
MAX_SUMMARY_CHARS = 700
MAX_ROW_CHARS = 220

SYSTEM = ("You are a senior engineer writing the walkthrough of a pull request for its "
          "reviewers. Describe only what the diff shows; never invent files, functions or "
          "behaviour.")
INSTRUCTIONS = ("Write, in Markdown:\n"
                "## Walkthrough\nTwo or three sentences on what this PR changes and why it "
                "matters.\n"
                "## Changes\nA table with columns File | Change summary, one row per file.\n"
                "## Risk areas\nUp to three bullets on what a reviewer should look at closely.")


@dataclass
class Walkthrough:
    summary: str = ""
    per_file: dict[str, str] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.summary or self.per_file)


def build_messages(title: str, files: list[dict]) -> list[dict]:
    """The PR as a unified diff, files in GitHub's order, cut at MAX_DIFF_CHARS
    with a note saying which files did not fit."""
    parts, used, left_out = [], 0, []
    for f in files:
        name, patch = f.get("filename", ""), f.get("patch") or ""
        block = f"diff --git a/{name} b/{name}\n{patch or '(binary or too large to show)'}\n"
        if used + len(block) > MAX_DIFF_CHARS and parts:
            left_out.append(name)
            continue
        parts.append(block[:MAX_DIFF_CHARS])
        used += len(block)
    diff = "".join(parts)
    if left_out:
        diff += f"\n(also changed, diff not shown: {', '.join(left_out)})\n"
    user = f"Pull request: {title.strip() or '(untitled)'}\n\n```diff\n{diff}```\n\n{INSTRUCTIONS}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _sections(text: str) -> dict[str, str]:
    """`## Heading` -> body, headings lower-cased."""
    out: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"\s*#{1,4}\s*(.+?)\s*:?\s*$", line)
        if m:
            current = m.group(1).lower()
            out[current] = ""
        elif current is not None:
            out[current] += line + "\n"
    return out


def _find(sections: dict[str, str], *keys: str) -> str:
    for name, body in sections.items():
        if any(k in name for k in keys):
            return body
    return ""


def _clean(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def parse(text: str, paths: list[str]) -> Walkthrough:
    """Pull the three sections out of the model's Markdown. Table rows are
    matched to the PR's real paths (exact, else a unique basename)."""
    sections = _sections(text or "")
    w = Walkthrough(summary=_clean(_find(sections, "walkthrough", "summary"), MAX_SUMMARY_CHARS))

    by_base: dict[str, list[str]] = {}
    for p in paths:
        by_base.setdefault(p.rsplit("/", 1)[-1], []).append(p)
    for line in _find(sections, "change").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: "):
            continue
        name = cells[0].strip("`* ")
        if name in paths:
            path = name
        else:
            same_base = by_base.get(name.rsplit("/", 1)[-1], [])
            path = same_base[0] if len(same_base) == 1 else None
        summary = _clean(" ".join(cells[1:]), MAX_ROW_CHARS)
        if path and summary and path not in w.per_file:
            w.per_file[path] = summary

    for line in _find(sections, "risk", "check", "concern").splitlines():
        m = re.match(r"\s*(?:[-*•]|\d+[.)])\s+(.+)", line)
        if m and len(w.risks) < MAX_RISKS:
            w.risks.append(_clean(m.group(1), 400))
    return w


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def render(w: Walkthrough, files: list[dict], head_sha: str) -> str:
    lines = ["## Walkthrough", "", w.summary or "_No summary._", "", "## Changes", "",
             "| File | Change | Lines |", "|---|---|---|"]
    for f in files:
        name = f.get("filename", "")
        status = f.get("status", "modified")
        what = w.per_file.get(name) or ("_Removed._" if status == "removed" else "—")
        lines.append(f"| `{_cell(name)}` | {_cell(what)} | "
                     f"+{f.get('additions', 0)} / −{f.get('deletions', 0)} |")
    if w.risks:
        lines += ["", "## Things to check", ""] + [f"- {r}" for r in w.risks]
    lines += ["", f"<sub>prana walkthrough for `{head_sha[:7]}` · written by DeepSeek-Coder 6.7B "
                  "from the diff alone and not verified; the inline review comments are the "
                  "findings. Updated on every new commit.</sub>", MARKER]
    return "\n".join(lines)


def upsert(gh, repo: str, number: int, body: str, bot_login: str) -> str:
    """Edit our existing walkthrough comment, or create one."""
    for c in gh.issue_comments(repo, number):
        mine = (c.get("user") or {}).get("login") == bot_login
        if mine and MARKER in (c.get("body") or ""):
            gh.update_issue_comment(repo, c["id"], body)
            return "updated"
    gh.issue_comment(repo, number, body)
    return "created"


def post_walkthrough(gh, generate: GenerateFn, repo: str, number: int, head_sha: str,
                     title: str, files: list[dict],
                     bot_login: str = "prana-reviewer[bot]") -> str:
    """created | updated | unusable | failed. Never raises: the walkthrough is
    an extra, and the review is already on GitHub when this runs. When no
    walkthrough can be written, the progress comment still turns into a
    "review posted" note, so it never stays at "reviewing"."""
    try:
        text = generate(build_messages(title, files))
        w = parse(text, [f.get("filename", "") for f in files])
        if w.usable:
            return upsert(gh, repo, number, render(w, files, head_sha), bot_login)
        log.warning("walkthrough for %s#%s unusable: %r", repo, number, (text or "")[:200])
        status = "unusable"
    except Exception as e:  # noqa: BLE001 - see docstring
        log.warning("walkthrough for %s#%s failed: %s", repo, number, e)
        status = "failed"
    set_status(gh, repo, number, done_body(head_sha), bot_login)
    return status


# ---------------------------------------------------------------- progress
# The same comment shows progress while the review runs, so a pull request
# never looks idle for the ~20 s the model needs: "reviewing" right after the
# webhook, then the walkthrough (or a short note) when the review is posted.

def progress_body(head_sha: str, hunks: int, files: int, semgrep: bool) -> str:
    engines = "the fine-tuned DeepSeek-Coder model"
    if semgrep:
        engines += " and Semgrep security rules"
    return "\n".join([
        "## ⏳ prana is reviewing this pull request", "",
        f"Reviewing **{hunks}** changed {'hunk' if hunks == 1 else 'hunks'} in **{files}** "
        f"{'file' if files == 1 else 'files'} at commit `{head_sha[:7]}` with {engines}. "
        "The review and a walkthrough will appear in about 20–30 seconds.", "",
        "<sub>This comment updates itself when the review is done.</sub>", MARKER])


def done_body(head_sha: str) -> str:
    return "\n".join([
        "## ✅ prana review posted", "",
        f"The review of commit `{head_sha[:7]}` is above. A walkthrough could not be written "
        "for this commit.", MARKER])


def interrupted_body(head_sha: str, reason: str) -> str:
    return "\n".join([
        "## ⚠️ prana could not finish this review yet", "",
        f"The review of commit `{head_sha[:7]}` was interrupted: {reason}. It is retried "
        "automatically; if it keeps failing, a comment below explains why.", MARKER])


def set_status(gh, repo: str, number: int, body: str, bot_login: str) -> bool:
    """Write the progress comment. Never raises: a status update must not
    block or fail the review it describes."""
    try:
        upsert(gh, repo, number, body, bot_login)
        return True
    except Exception as e:  # noqa: BLE001 - see docstring
        log.warning("status comment for %s#%s failed: %s", repo, number, e)
        return False
