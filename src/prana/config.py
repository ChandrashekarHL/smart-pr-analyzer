"""Single source of truth for every threshold in the data pipeline.

Nothing downstream hard-codes a number. The config is hashed into the dataset
card, so a corpus can always be traced back to the exact settings that built
it - if two runs disagree, compare fingerprints before comparing outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

# --------------------------------------------------------------------- scope

# Ordered by share of the corpus. Go leads, which is not what the original
# design document assumed; see docs/dataset-card.md.
LANGUAGES: tuple[str, ...] = ("go", "typescript", "rust", "python", "javascript")

EXT_LANG: dict[str, str] = {
    "go": "go",
    "rs": "rust",
    "py": "python", "pyi": "python", "pyx": "python",
    "ts": "typescript", "tsx": "typescript", "mts": "typescript", "cts": "typescript",
    "js": "javascript", "jsx": "javascript", "mjs": "javascript", "cjs": "javascript",
    "java": "java", "kt": "kotlin", "scala": "scala",
    "c": "c", "h": "c",
    "cc": "cpp", "cpp": "cpp", "cxx": "cpp", "hpp": "cpp", "hh": "cpp",
    "cs": "csharp", "rb": "ruby", "php": "php", "swift": "swift",
    "sh": "shell", "bash": "shell", "zsh": "shell", "fish": "shell",
    "vue": "vue", "svelte": "svelte", "sql": "sql", "proto": "protobuf",
    "lua": "lua", "zig": "zig",
    "yml": "yaml", "yaml": "yaml", "json": "json", "toml": "toml",
    "md": "markdown", "mdx": "markdown", "rst": "prose", "txt": "prose", "adoc": "prose",
}

PROSE_LANGUAGES: frozenset[str] = frozenset({"markdown", "prose"})


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Every knob that changes the contents of the built corpus."""

    languages: tuple[str, ...] = LANGUAGES
    drop_prose: bool = True

    # Quality floor. The original design document specified 30 words; we keep
    # it. Relaxing to 20 yields ~40k rows instead of ~30k, at visibly lower
    # comment quality - measured, not assumed.
    min_comment_words: int = 30
    max_diff_chars: int = 12_000
    drop_bare_questions: bool = True
    bare_question_max_words: int = 40

    # Licence policy for anything we publish. See data/licenses.py.
    require_permissive_license: bool = True

    # Secret redaction. Off means the raw value is kept, which is only ever
    # acceptable for a corpus that never leaves the machine.
    redact_secrets: bool = True

    # Decontamination between train and the held-out sets.
    decontaminate: bool = True
    ngram_size: int = 13
    max_test_overlap_ratio: float = 0.02

    # Split geometry. test_ood holds out whole repositories; validation and
    # test hold out whole pull requests from repositories seen in training.
    ood_fraction: float = 0.08
    val_fraction: float = 0.06
    test_fraction: float = 0.06
    seed: int = 13

    extra_bot_reviewers: tuple[str, ...] = field(default=())

    def fingerprint(self) -> str:
        """Stable 12-hex-char digest of the configuration."""
        blob = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)
