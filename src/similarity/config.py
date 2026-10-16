from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SharedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    redis_url: str = "redis://redis:6379/0"
    log_level: str = "INFO"
    result_ttl_s: int = 86400
    stream_key: str = "analyses:jobs"
    consumer_group: str = "workers"


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
    read_block_ms: int = 5_000
    visibility_timeout_s: int = 30
    reclaim_interval_s: int = 10
    max_attempts: int = 3
    torch_num_threads: int = 2
    heartbeat_path: str = "/tmp/worker-heartbeat"  # noqa: S108 -- container-local tmpfs, not shared with host
    heartbeat_max_age_s: int = 30
    metrics_port: int = 9_100

    @model_validator(mode="after")
    def _check_invariants(self) -> WorkerSettings:
        if self.heartbeat_max_age_s <= 2 * self.read_block_ms / 1000:
            raise ValueError("HEARTBEAT_MAX_AGE_S must be > 2 * READ_BLOCK_MS / 1000")
        if self.max_attempts < 1:
            raise ValueError("MAX_ATTEMPTS must be >= 1")
        return self
