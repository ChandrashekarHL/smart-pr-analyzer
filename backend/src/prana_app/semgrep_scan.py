"""Rule-based security scanning with Semgrep, the deterministic half of the
security pass. Semgrep is reliable exactly where a language model is not -
known dangerous calls (shell=True, pickle.loads, InsecureSkipVerify, eval) -
and the model covers what rules cannot express.

    findings = scan({"app/x.py": code, ...})   # {path: [SemgrepFinding]}

Runs the Semgrep CLI on a temporary directory with the given rule packs,
metrics off, a timeout, and returns findings grouped by file. A missing
binary, a timeout or bad output returns {} and logs - never an exception.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("prana_app.semgrep")

# Semgrep's recommended default ruleset plus secrets. `p/security-audit` (the
# first choice) lacks the per-language rules: it missed Go's
# InsecureSkipVerify, which p/default catches (checked 2026-10-31).
DEFAULT_CONFIGS = ("p/default", "p/secrets")
TIMEOUT_S = 120
# Snippets are fragments; a package line lets Go parse them.
PREAMBLE = {".go": "package snippet\n\n"}


@dataclass(frozen=True)
class SemgrepFinding:
    rule: str
    line: int
    severity: str
    message: str


def scan(files: dict[str, str], configs: tuple[str, ...] = DEFAULT_CONFIGS,
         semgrep: str | None = None) -> dict[str, list[SemgrepFinding]]:
    exe = semgrep or shutil.which("semgrep")
    if not exe:
        log.warning("semgrep not installed; rule-based scan skipped")
        return {}
    with tempfile.TemporaryDirectory(prefix="prana-semgrep-") as tmp:
        offsets: dict[str, int] = {}
        for rel, text in files.items():
            p = Path(tmp) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            # Only fragments need the preamble; a real Go file already has its
            # own package clause, and a second one would not parse.
            pre = "" if text.lstrip().startswith("package ") else PREAMBLE.get(p.suffix, "")
            offsets[p.resolve().as_posix()] = pre.count("\n")
            p.write_text(pre + text, encoding="utf-8")
        cmd = [exe, "scan", "--json", "--quiet", "--metrics", "off", "--disable-version-check",
               "--timeout", "30"]
        for c in configs:
            cmd += ["--config", c]
        cmd.append(tmp)
        try:
            # Argument list, no shell; the executable, rule packs and target
            # directory are all ours.
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S,  # noqa: S603
                               env={**os.environ, "SEMGREP_SEND_METRICS": "off"})
            data = json.loads(p.stdout or "{}")
        except (subprocess.TimeoutExpired, OSError, ValueError) as e:
            log.warning("semgrep failed: %s", e)
            return {}
        out: dict[str, list[SemgrepFinding]] = {}
        root = Path(tmp).resolve()
        for r in data.get("results", []):
            path = Path(r.get("path", "")).resolve()
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            line = int(r.get("start", {}).get("line", 0)) - offsets.get(path.as_posix(), 0)
            extra = r.get("extra", {})
            out.setdefault(rel, []).append(SemgrepFinding(
                rule=r.get("check_id", "?"), line=max(line, 1),
                severity=str(extra.get("severity", "WARNING")).lower(),
                message=" ".join(str(extra.get("message", "")).split())[:300]))
        return out
