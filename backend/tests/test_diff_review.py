from __future__ import annotations

from conftest import PATCH, model_result

from prana_app.diff import language_for, parse_patch, reviewable_hunks
from prana_app.review import Finding, finding_from, is_duplicate, summary
from prana_app.security import sign, verify_signature


def test_signature_round_trip_and_rejections():
    body = b'{"a":1}'
    good = sign("s3cret", body)
    assert verify_signature("s3cret", body, good)
    assert not verify_signature("s3cret", body + b" ", good)
    assert not verify_signature("other", body, good)
    assert not verify_signature("s3cret", body, None)
    assert not verify_signature("s3cret", body, good.replace("sha256=", "sha1="))
    assert not verify_signature("", body, sign("", body))     # unset secret never verifies


def test_parse_patch_numbers_new_file_lines():
    h1, h2 = parse_patch("app/api.py", PATCH, "python")
    assert h1.added == [11, 12] and h1.context == [10, 13]
    assert h2.added == [43] and h2.context == [42, 44]
    assert h1.text.startswith("@@ -10,4 +10,6 @@") and "+    rows = db.execute(q)" in h1.text


def test_anchor_keeps_valid_lines_and_moves_others_to_nearest_added():
    h1, _ = parse_patch("app/api.py", PATCH, "python")
    assert h1.anchor(12) == (12, True)
    assert h1.anchor(13) == (13, True)             # context lines are valid targets
    assert h1.anchor(30) == (12, False)
    assert h1.anchor(None) == (12, False)
    assert h1.anchor(-5) == (11, False)


def test_language_and_generated_file_filtering():
    assert language_for("src/a.ts") == "typescript" and language_for("x/y.rs") == "rust"
    assert language_for("README.md") is None
    assert language_for("web/node_modules/x.js") is None
    assert language_for("api/types.d.ts") is None and language_for("p/x_pb2.py") is None


def test_reviewable_hunks_largest_first_and_capped():
    files = [{"filename": "app/api.py", "status": "modified", "patch": PATCH},
             {"filename": "gone.py", "status": "removed", "patch": PATCH},
             {"filename": "README.md", "status": "modified", "patch": PATCH},
             {"filename": "only_del.py", "status": "modified",
              "patch": "@@ -1,2 +1,1 @@\n a\n-b"}]
    hunks, total = reviewable_hunks(files, max_hunks=1)
    assert total == 2 and len(hunks) == 1 and hunks[0].added == [11, 12]


def test_finding_and_comment_body():
    h1, _ = parse_patch("app/api.py", PATCH, "python")
    f = finding_from(h1, model_result(line=99))
    assert (f.path, f.line, f.line_exact, f.severity) == ("app/api.py", 12, False, "critical")
    c = f.as_github_comment()
    assert c["side"] == "RIGHT" and c["line"] == 12
    assert "critical · SECURITY" in c["body"] and "line approximate" in c["body"]
    assert f"<!-- prana:{f.fingerprint} -->" in c["body"]
    assert finding_from(h1, {"parsed": None, "text": "garbage"}) is None


def _f(**kw):
    base = dict(path="a.py", line=10, line_exact=True, severity="warning", issue_type="BUG",
                explanation="Guard against None before indexing.", suggestion="", confidence=0.7)
    return Finding(**{**base, **kw})


def test_duplicates_by_marker_and_by_near_identical_text():
    f = _f()
    assert is_duplicate(f, [{"path": "x.py", "line": 1, "body": f.body()}])          # marker
    assert is_duplicate(f, [{"path": "a.py", "line": 12,
                             "body": "Guard against None before indexing!"}])
    assert not is_duplicate(f, [{"path": "a.py", "line": 40,
                                 "body": "Guard against None before indexing."}])   # far away
    assert not is_duplicate(f, [{"path": "a.py", "line": 10, "body": "Rename this variable."}])


def test_summary_reports_counts_and_what_was_left_out():
    s = summary([_f(severity="critical"), _f(severity="warning")], [_f(confidence=0.1)],
                reviewed=5, total=9, failed=1, skipped_for_time=2, duplicates=1)
    assert "Reviewed **5** of 9" in s and "🔴 1 critical" in s and "🟠 1 warning" in s
    assert "2 smaller hunks were not reviewed" in s and "2 hunks skipped" in s
    assert "1 hunks had no usable model output" in s and "1 comments suppressed" in s
    assert "Low-confidence suggestions (1)" in s
