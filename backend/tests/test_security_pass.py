"""The security pass: LangChain chain, the API-backed chat model, Semgrep."""

from __future__ import annotations

import json
import shutil

import pytest

pytest.importorskip("langchain_core")
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from prana_app.security_pass import (  # noqa: E402
    PranaChatModel,
    build_chain,
    review_security,
)

HUNK = "@@ -0,0 +1,2 @@\n+def q(c, n):\n+    return c.execute(f\"SELECT * FROM t WHERE n='{n}'\")"
FINDING = {"vulnerability": "SQL injection", "line": 2, "severity": "critical",
           "explanation": "The query interpolates user input into SQL.",
           "fix": "Use a parameterised query."}


def _run(reply: str):
    return review_security(build_chain(FakeListChatModel(responses=[reply])), HUNK, "python",
                           "app/db.py")


def test_parses_findings_from_json_and_code_fences():
    assert _run(json.dumps({"findings": [FINDING]}))[0].vulnerability == "SQL injection"
    fenced = "Here you go:\n```json\n" + json.dumps({"findings": [FINDING]}) + "\n```"
    assert _run(fenced)[0].severity == "critical"
    assert _run(json.dumps([FINDING]))[0].line == 2               # bare list accepted


def test_empty_findings_and_garbage_never_raise():
    assert _run('{"findings": []}') == []
    assert _run("I think this code is fine.") == []
    bad = dict(FINDING, severity="catastrophic")                  # not in the allowed set
    assert _run(json.dumps({"findings": [bad]})) == []


def test_chain_failure_returns_no_findings():
    class Boom(FakeListChatModel):
        def _call(self, *a, **k):
            raise RuntimeError("model API down")
    assert review_security(build_chain(Boom(responses=["x"])), HUNK, "python", "a.py") == []


def test_prana_chat_model_calls_the_api_with_the_adapter_off():
    calls = []

    class FakeClient:
        def generate(self, messages, max_new_tokens, adapter):
            calls.append((messages, max_new_tokens, adapter))
            return {"text": json.dumps({"findings": [FINDING]})}
    chain = build_chain(PranaChatModel(client=FakeClient()))
    out = review_security(chain, HUNK, "python", "app/db.py")
    assert out[0].vulnerability == "SQL injection"
    (messages, max_new, adapter), = calls
    assert adapter is False and max_new == 384
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "app/db.py" in messages[1]["content"] and "SELECT" in messages[1]["content"]
    assert "security" in messages[0]["content"].lower()


@pytest.mark.skipif(shutil.which("semgrep") is None, reason="semgrep not installed")
def test_semgrep_flags_known_dangerous_calls_and_not_safe_code():
    from prana_app.semgrep_scan import scan
    out = scan({
        "bad.py": "import subprocess\n\ndef run(name):\n"
                  "    subprocess.run('wc ' + name, shell=True)\n",
        "good.py": "import subprocess\n\ndef run(name):\n"
                   "    subprocess.run(['wc', name], check=True)\n",
    })
    assert out.get("bad.py"), out
    assert not out.get("good.py")
    assert out["bad.py"][0].line == 4


def test_security_eval_combines_systems_honestly():
    from prana.eval.bench_security import CASES

    from prana_app.security_eval import outcomes, summarize
    alarm = json.dumps({"explanation": "x", "suggestion": "", "severity": "warning",
                        "issue_type": "BUG", "file_path": None, "line_number": None})
    data, verdicts = {}, {}
    for i, c in enumerate(CASES):
        data[f"{c['id']}:bad"] = {"reviewer": alarm, "llm_pass": [FINDING], "semgrep": []}
        # fixed code: llm pass raises a (rejected) finding on the first 4 cases
        data[f"{c['id']}:good"] = {"reviewer": alarm.replace("warning", "info"),
                                   "llm_pass": [FINDING] if i < 4 else [], "semgrep": []}
        verdicts[f"{c['id']}:bad:llm_pass"] = {"verdict": "same" if i % 2 == 0 else "different"}
        verdicts[f"{c['id']}:bad:reviewer"] = {"verdict": "same" if i % 2 == 1 and i < 10 else "x"}
        verdicts[f"{c['id']}:good:llm_pass"] = {"model_valid": "no"}
    s = summarize(outcomes(data, verdicts))
    n = len(CASES)
    assert s["llm_pass"]["detection"]["rate"] == round(len(range(0, n, 2)) / n, 3)
    assert s["llm_pass"]["false_positive"]["rate"] == round(4 / n, 3)
    assert s["reviewer"]["false_positive"]["rate"] == 0.0           # info on fixed code
    both = len(range(0, n, 2)) + len([i for i in range(10) if i % 2 == 1])
    assert s["all_three"]["detection"]["rate"] == round(both / n, 3)
    assert s["all_three"]["false_positive"]["rate"] == round(4 / n, 3)


def test_prose_reply_falls_back_to_the_strict_json_prompt():
    llm = FakeListChatModel(responses=["The code looks like it counts words.",
                                       json.dumps({"findings": [FINDING]})])
    out = review_security(build_chain(llm), HUNK, "python", "app/db.py")
    assert [f.vulnerability for f in out] == ["SQL injection"]       # second, strict call won


def test_snippets_become_files_with_their_imports():
    from prana_app.snippet_files import as_file
    py = as_file("def f(n):\n    subprocess.run('wc ' + n, shell=True)", "python")
    assert py.startswith("import subprocess\n")
    go = as_file('func F() {\n\trand.Intn(5)\n\texec.Command("ls")\n}', "go")
    assert go.startswith("package snippet") and '"math/rand"' in go and '"os/exec"' in go
    assert '"crypto/rand"' not in go
    js = as_file("export function t(f, cb) { exec(`convert ${f}`, cb); }", "javascript")
    assert js.startswith('import { exec, execFile } from "child_process";')


@pytest.mark.skipif(shutil.which("semgrep") is None, reason="semgrep not installed")
def test_go_file_with_its_own_package_clause_is_not_given_a_second_one():
    from prana_app.semgrep_scan import scan
    go = ('package main\n\nimport "crypto/tls"\n\n'
          "func c() *tls.Config {\n\treturn &tls.Config{InsecureSkipVerify: true}\n}\n")
    assert scan({"c.go": go}).get("c.go"), "a parse error would hide the TLS finding"


def test_semgrep_metadata_reaches_the_finding(tmp_path):
    import sys

    from prana_app.semgrep_scan import scan
    # A stand-in for the CLI: answers like `semgrep scan --json` for the
    # temporary directory it is given (the last argument).
    result = {"check_id": "py.pickle", "start": {"line": 3}, "extra": {
        "severity": "ERROR", "message": "Avoid pickle.",
        "metadata": {"cwe": ["CWE-502: Deserialization of Untrusted Data"],
                     "owasp": "A08:2021 - Software and Data Integrity Failures",
                     "shortlink": "https://sg.run/abc"}}}
    fake = tmp_path / "semgrep"
    fake.write_text("\n".join([
        f"#!{sys.executable}",
        "import json, sys",
        f"r = {result!r}",
        "r['path'] = sys.argv[-1] + '/a.py'",
        "print(json.dumps({'results': [r]}))",
    ]) + "\n", encoding="utf-8")
    fake.chmod(0o755)
    (hit,) = scan({"a.py": "import pickle\n\npickle.loads(x)\n"}, semgrep=str(fake))["a.py"]
    assert (hit.rule, hit.line, hit.severity) == ("py.pickle", 3, "error")
    assert hit.cwe == ("CWE-502: Deserialization of Untrusted Data",)
    assert hit.owasp == ("A08:2021 - Software and Data Integrity Failures",)
    assert hit.url == "https://sg.run/abc"
