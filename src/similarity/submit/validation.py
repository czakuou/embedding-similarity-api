"""Pure validation and pre-cut functions for the POST slice.

validate_client_id lives in analysis.py (shared kernel), not here: both this
slice and results.py need the same X-Client-Id contract.
"""

from __future__ import annotations

from dataclasses import dataclass


def validate_filename(filename: str | None) -> bool:
    return filename is not None and filename.lower().endswith(".txt")


def decode_document(raw: bytes) -> str | None:
    """Strict UTF-8 decode; SPEC A4: headers can lie, decoding is the real validation."""
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None


def validate_query(query: str) -> str | None:
    """SPEC A5: 1-1000 chars after trimming. Returns the trimmed query, or None if invalid."""
    trimmed = query.strip()
    if not (1 <= len(trimmed) <= 1_000):
        return None
    return trimmed


@dataclass(frozen=True, slots=True)
class PreCutDocument:
    text: str
    truncated: bool


def pre_cut_document(text: str, max_doc_chars: int) -> PreCutDocument:
    """SPEC A3: pre-cut to MAX_DOC_CHARS; the model truncates further at 256 tokens."""
    if len(text) <= max_doc_chars:
        return PreCutDocument(text=text, truncated=False)
    return PreCutDocument(text=text[:max_doc_chars], truncated=True)
