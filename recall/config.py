"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("PORT", 8000))
    kb_dir: Path = field(default_factory=lambda: Path(os.getenv("KB_DIR", ROOT / "data" / "knowledge_base")))
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "auto"))  # auto | hashing | minilm
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    # gemini-1.5-flash (used in the original spec) was shut down on 2025-09-29.
    # "gemini-flash-latest" is an alias; pin a concrete model in production.
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-latest"))
    cache_threshold: float | None = field(
        default_factory=lambda: float(os.environ["CACHE_THRESHOLD"]) if os.getenv("CACHE_THRESHOLD") else None
    )
    cache_ttl_seconds: int = field(default_factory=lambda: _int("CACHE_TTL_SECONDS", 86_400))
    cache_max_entries: int = field(default_factory=lambda: _int("CACHE_MAX_ENTRIES", 1_000))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 3))
    rate_limit_per_min: int = field(default_factory=lambda: _int("RATE_LIMIT_PER_MIN", 60))
