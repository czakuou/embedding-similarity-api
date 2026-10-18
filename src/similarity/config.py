from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SharedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    redis_url: str = "redis://redis:6379/0"
    log_level: str = "INFO"
    result_ttl_s: int = 86400
    queue_name: str = "arq:queue"


class ApiSettings(SharedSettings):
    max_file_size_bytes: int = 102_400
    max_doc_chars: int = 4_096
    max_query_length: int = 1_000
    rate_limit: str = "20/second"
    rate_limit_strategy: str = "fixed-window"
    max_queue_depth: int = 1_000
    retry_after_base_s: int = 5
    poll_interval_hint_s: int = 1
    api_concurrency_limit: int = 200

    @model_validator(mode="after")
    def _check_invariants(self) -> ApiSettings:
        if self.max_doc_chars > self.max_file_size_bytes:
            raise ValueError("MAX_DOC_CHARS must be <= MAX_FILE_SIZE_BYTES")
        return self


class WorkerSettings(SharedSettings):
    batch_max_size: int = 32
    batch_linger_ms: int = 10
    poll_delay_s: float = 0.1
    visibility_timeout_s: int = 30
    shutdown_grace_s: int = 5
    max_attempts: int = 3
    torch_num_threads: int = 2
    metrics_port: int = 9_100

    @model_validator(mode="after")
    def _check_invariants(self) -> WorkerSettings:
        if self.max_attempts < 1:
            raise ValueError("MAX_ATTEMPTS must be >= 1")
        return self
