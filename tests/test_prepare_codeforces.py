"""CPU-only tests for Codeforces dataset preparation helpers.

Run with either:
    python -m pytest tests/test_prepare_codeforces.py
    python tests/test_prepare_codeforces.py        # no pytest needed
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.prepare_codeforces import build_question, convert, keep_row


def _base_row(**overrides):
    row = {
        "id": "1000/A",
        "title": "A. Sample",
        "description": "Compute $$$x$$$.",
        "input_format": "One integer.",
        "output_format": "The answer.",
        "examples": [{"input": "1\r\n", "output": "1\r\n"}],
        "note": None,
        "input_mode": "stdio",
        "executable": True,
        "interaction_format": None,
        "generated_checker": None,
        "official_tests": [{"input": "1\r\n", "output": "1\r\n"}],
        "time_limit": 1.0,
        "rating": 800,
        "tags": ["implementation"],
    }
    row.update(overrides)
    return row


def test_keep_row_rejects_missing_description():
    assert keep_row(_base_row(description=None)) is False


def test_build_question_handles_nullable_optional_text_and_examples():
    question = build_question(_base_row(
        input_format=None,
        output_format=None,
        examples=[{"input": None, "output": None}, {"input": "2\n", "output": "4\n"}],
        note="Use $$$x^2$$$.",
    ))

    assert "A. Sample" in question
    assert "Compute $x$." in question
    assert "Input\n2\nOutput\n4" in question
    assert "Use $x^2$." in question


def test_convert_normalizes_official_tests_and_metadata():
    converted = convert(_base_row())

    assert converted["task_id"] == "1000/A"
    assert converted["test_input"] == ["1\n"]
    assert converted["test_output"] == ["1\n"]
    assert converted["test_time_limit"] == 4
    assert converted["test_method"] == "stdio"


if __name__ == "__main__":
    tests = [
        ("keep_row_rejects_missing_description", test_keep_row_rejects_missing_description),
        ("build_question_nullable_text", test_build_question_handles_nullable_optional_text_and_examples),
        ("convert_normalizes_tests", test_convert_normalizes_official_tests_and_metadata),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
