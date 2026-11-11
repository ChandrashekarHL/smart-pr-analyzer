"""Turn a benchmark snippet into the minimal source file it would live in -
the snippet plus the imports it uses - so file-level scanners such as
Semgrep see what they would see in a real repository. Many Semgrep rules
only fire when, say, `import subprocess` or `"crypto/tls"` is present; a bare
snippet would understate them. Evaluation only: production scans real files.
"""

from __future__ import annotations

import re

_PY = [("subprocess.", "import subprocess"), ("pickle.", "import pickle"),
       ("yaml.", "import yaml"), ("hashlib.", "import hashlib"), ("requests.", "import requests"),
       ("os.", "import os"), ("json.", "import json"), ("base64.", "import base64"),
       ("bcrypt.", "import bcrypt"), ("urllib.parse", "import urllib.parse"),
       ("Path(", "from pathlib import Path")]
_GO = [("fmt.", "fmt"), ("os.", "os"), ("exec.", "os/exec"), ("http.", "net/http"),
       ("tls.", "crypto/tls"), ("rand.Intn", "math/rand"), ("rand.Read", "crypto/rand"),
       ("sql.", "database/sql"), ("filepath.", "path/filepath"), ("hex.", "encoding/hex"),
       ("hmac.", "crypto/hmac"), ("sha256.", "crypto/sha256"), ("time.", "time"),
       ("errors.", "errors"), ("strings.", "strings"), ("sync.", "sync")]
_JS = [(re.compile(r"\bexec(?:File)?\("), 'import { exec, execFile } from "child_process";'),
       (re.compile(r"\bcrypto\."), 'import crypto from "crypto";'),
       (re.compile(r"\bjwt\."), 'import jwt from "jsonwebtoken";'),
       (re.compile(r"\bpath\."), 'import path from "path";'),
       (re.compile(r"\bnew Pool\("), 'import { Pool } from "pg";')]
_RS = [("Command::", "use std::process::Command;"), ("Path", "use std::path::{Path, PathBuf};"),
       ("HashMap", "use std::collections::HashMap;"), ("Connection", "use rusqlite::Connection;")]


def as_file(code: str, language: str) -> str:
    if language == "python":
        imports = [imp for key, imp in _PY if key in code]
        return "\n".join(imports) + ("\n\n\n" if imports else "") + code + "\n"
    if language == "go":
        pkgs = sorted({pkg for key, pkg in _GO if key in code})
        block = ("import (\n" + "".join(f'\t"{p}"\n' for p in pkgs) + ")\n\n") if pkgs else ""
        return "package snippet\n\n" + block + code + "\n"
    if language in ("javascript", "typescript"):
        imports = [imp for rx, imp in _JS if rx.search(code)]
        return "\n".join(imports) + ("\n\n" if imports else "") + code + "\n"
    if language == "rust":
        imports = [imp for key, imp in _RS if key in code]
        return "\n".join(imports) + ("\n\n" if imports else "") + code + "\n"
    return code + "\n"
