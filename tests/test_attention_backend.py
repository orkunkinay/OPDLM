"""CPU-only tests for shared attention-backend resolution."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_backend import resolve_attn_implementation


def test_resolve_attn_implementation_flash():
    assert resolve_attn_implementation("flash") == "flash_attention_2"


def test_resolve_attn_implementation_sdpa():
    assert resolve_attn_implementation("sdpa") == "sdpa"


def test_resolve_attn_implementation_invalid_value():
    try:
        resolve_attn_implementation("not-a-backend")
    except ValueError as exc:
        assert "Unsupported attention backend" in str(exc)
    else:
        raise AssertionError("invalid attention backend did not raise ValueError")


if __name__ == "__main__":
    tests = [
        ("resolve_flash", test_resolve_attn_implementation_flash),
        ("resolve_sdpa", test_resolve_attn_implementation_sdpa),
        ("invalid_value", test_resolve_attn_implementation_invalid_value),
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
