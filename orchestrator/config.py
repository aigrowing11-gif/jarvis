"""Explicit, validated configuration; no implicit paid endpoint fallback."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    model: str = "qwen/qwen3-next-80b-a3b-instruct"
    request_interval: float = 3.0
    request_timeout: float = 45.0
    max_retries: int = 2
    max_retry_wait: float = 30.0
    max_tokens: int = 2048

    def __post_init__(self):
        for name in ("request_interval", "request_timeout", "max_retry_wait"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or value > 300:
                raise ValueError(f"{name} must be finite and between 0 and 300")
        if self.request_timeout == 0:
            raise ValueError("request_timeout must be positive")
        if not 0 <= self.max_retries <= 5 or not 1 <= self.max_tokens <= 4096:
            raise ValueError("max_retries must be 0..5 and max_tokens must be 1..4096")
        if not self.model.strip() or len(self.model) > 200:
            raise ValueError("NVIDIA_MODEL must be a nonempty catalog model ID")

    @classmethod
    def load(cls, env_file: Path = Path(".env")) -> "Settings":
        # Only the explicitly selected file; never search parent directories.
        values = {**dotenv_values(env_file), **os.environ}
        try:
            return cls(
                api_key=(values.get("NVIDIA_API_KEY") or "").strip(),
                model=values.get("NVIDIA_MODEL") or cls.model,
                request_interval=float(values.get("JARVIS_REQUEST_INTERVAL", 3)),
                request_timeout=float(values.get("JARVIS_REQUEST_TIMEOUT", 45)),
                max_retries=int(values.get("JARVIS_MAX_RETRIES", 2)),
                max_retry_wait=float(values.get("JARVIS_MAX_RETRY_WAIT", 30)),
                max_tokens=int(values.get("JARVIS_MAX_TOKENS", 2048)),
            )
        except (TypeError, ValueError):
            raise ValueError("Invalid Jarvis configuration; check .env.example ranges") from None
