"""Guards against the file-level mistakes that have already cost a run.

The repo is edited on Windows and executed on an aarch64 Linux box. A shell
script that picks up CRLF endings fails there with `set: pipefail: invalid
option name` - which is exactly how the base-model eval died on 2026-10-19.
The same day a regex acquired a literal backspace where `\\b` was meant,
because a shell heredoc collapsed the escape; the pattern silently matched
nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEXT = sorted(
    [*ROOT.glob("scripts/*.sh"), *ROOT.glob("src/prana/**/*.py"),
     *ROOT.glob("tests/*.py"), *ROOT.glob("configs/*.yaml")]
)
CRLF = b"\r\n"
ALLOWED_CONTROL = {9, 10}  # tab, newline


@pytest.mark.parametrize("path", TEXT, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_crlf_line_endings(path):
    assert CRLF not in path.read_bytes(), (
        f"{path.name} has CRLF; bash on the Spark will not run it"
    )


@pytest.mark.parametrize("path", TEXT, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_stray_control_characters(path):
    bad = sorted({c for c in path.read_bytes() if c < 32 and c not in ALLOWED_CONTROL})
    assert not bad, (
        f"{path.name} contains control bytes {bad}; a collapsed escape such as "
        "\\b -> backspace makes a regex match nothing"
    )


@pytest.mark.parametrize("path", sorted(ROOT.glob("scripts/*.sh")), ids=lambda p: p.name)
def test_shell_scripts_start_with_bash_shebang(path):
    assert path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash"), (
        f"{path.name}: run-job and setsid invoke `bash <script>`; a wrong shebang "
        "is harmless there but misleading everywhere else"
    )
