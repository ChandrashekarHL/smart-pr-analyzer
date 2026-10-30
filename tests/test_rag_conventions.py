"""Convention extraction: rules only with a clear, measured majority."""

from __future__ import annotations

from prana.rag.conventions import MAX_RULES, extract


def _py_module(i: int, typed: bool = True, doc: bool = True, quote: str = '"') -> str:
    sig = f"def load_item_{i}(key: str) -> int:" if typed else f"def load_item_{i}(key):"
    body = (f'    """Load item {i}.\n\n    Args:\n        key: the key.\n    """\n' if doc else "")
    return (f"from .store import Store\n\n\n{sig}\n{body}"
            f"    name = {quote}item{quote}\n    logger.info({quote}x{quote})\n"
            f"    if key:\n        return 1\n    return 0\n")


def test_python_rules_from_a_consistent_repo():
    files = [(f"pkg/m{i}.py", _py_module(i)) for i in range(30)]
    files += [("tests/test_m.py", "import pytest\n\ndef test_x():\n    assert True\n")] * 6
    rules = extract(files, "python")
    text = "\n".join(rules)
    assert "Function names are snake_case" in text
    assert "Functions have type hints (30 of 30 functions)." in text
    assert "docstrings (30 of 30), Google (Args:/Returns:) style" in text
    assert "Package imports are relative" in text
    assert "logging, not print()" in text
    assert len(rules) <= MAX_RULES


def test_mixed_styles_produce_no_rule():
    files = [(f"m{i}.py", _py_module(i, typed=i % 2 == 0, quote="'" if i % 2 else '"'))
             for i in range(40)]
    text = "\n".join(extract(files, "python"))
    assert "type hints" not in text and "quotes" not in text


def test_too_few_samples_produce_no_rules():
    assert extract([("a.py", _py_module(1))], "python") == []
    assert extract([], "go") == []


def test_go_error_wrapping_and_doc_comments():
    src = "".join(f'// Get{i} returns item {i}.\nfunc Get{i}() error {{\n'
                  f'\treturn fmt.Errorf("get {i}: %w", err)\n}}\n\n' for i in range(25))
    rules = extract([("pkg/get.go", src)], "go")
    assert any("%w" in r for r in rules)
    assert any("doc comment starting with their name (25 of 25)" in r for r in rules)


def test_rust_question_mark_over_unwrap_ignores_test_code():
    src = "".join(f"/// Do {i}.\npub fn f{i}() -> Result<()> {{\n    g()?;\n    Ok(())\n}}\n"
                  for i in range(25))
    src += "#[cfg(test)]\nmod tests {\n" + "    fn t() { h().unwrap(); }\n" * 50 + "}\n"
    rules = extract([("src/lib.rs", src)], "rust")
    assert any("`?` rather than `.unwrap()` (25 `?` vs 0 unwrap)" in r for r in rules)
    assert any("`///` doc comments (25 of 25)" in r for r in rules)


def test_typescript_semicolons_modules_and_any():
    src = "".join(f"import {{ a{i} }} from './a{i}';\n/** Doc. */\nexport function fn{i}(x: number): number {{\n"
                  f"  const y = x + 1;\n  return y;\n}}\n" for i in range(400))   # > 2,000 lines
    rules = extract([("src/a.ts", src)], "typescript")
    text = "\n".join(rules)
    assert "with semicolons" in text and "ES modules" in text
    assert "`any` is almost never used" in text
