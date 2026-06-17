"""Shared attention-backend resolution for model loading."""

from __future__ import annotations

import importlib.util
from typing import Any


FLASH_ATTN_MISSING_ERROR = (
    "FlashAttention was requested but flash_attn is not installed. "
    "Either install flash-attn or rerun with model.attn_backend=sdpa."
)

_ATTN_BACKEND_MAP = {
    "flash": "flash_attention_2",
    "flash_attention_2": "flash_attention_2",
    "sdpa": "sdpa",
    "eager": "eager",
}


def resolve_attn_implementation(value: str | None) -> str:
    """Map the user-facing backend name to Transformers' implementation name."""
    if value is None:
        value = "flash"
    key = str(value).strip().lower()
    try:
        return _ATTN_BACKEND_MAP[key]
    except KeyError as exc:
        valid = ", ".join(("flash", "sdpa", "eager"))
        raise ValueError(
            f"Unsupported attention backend {value!r}. Expected one of: {valid}."
        ) from exc


def ensure_attn_implementation_available(attn_implementation: str) -> str:
    """Fail early with an actionable message for unavailable optional backends."""
    resolved = resolve_attn_implementation(attn_implementation)
    if resolved == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        raise ImportError(FLASH_ATTN_MISSING_ERROR)
    return resolved


def get_model_attn_implementation(config: Any) -> str:
    """Read model.attn_backend / model.attn_implementation and validate it."""
    model_cfg = getattr(config, "model", None)
    value = None
    if model_cfg is not None:
        if hasattr(model_cfg, "get"):
            value = model_cfg.get("attn_backend", None)
            if value is None:
                value = model_cfg.get("attn_implementation", None)
        else:
            value = getattr(model_cfg, "attn_backend", None)
            if value is None:
                value = getattr(model_cfg, "attn_implementation", None)
    return ensure_attn_implementation_available(value)


def get_model_attn_kwargs(config: Any) -> dict[str, str]:
    """Return kwargs suitable for Hugging Face from_pretrained calls."""
    return {"attn_implementation": get_model_attn_implementation(config)}
