from __future__ import annotations

from similarity.submit.validation import (
    decode_document,
    pre_cut_document,
    validate_filename,
    validate_query,
)


def test_given_txt_filename_when_validating_then_accepted() -> None:
    assert validate_filename("doc.txt") is True
    assert validate_filename("DOC.TXT") is True


def test_given_non_txt_or_missing_filename_when_validating_then_rejected() -> None:
    assert validate_filename("data.csv") is False
    assert validate_filename(None) is False


def test_given_valid_utf8_when_decoding_then_text_returned() -> None:
    assert decode_document(b"hello") == "hello"


def test_given_invalid_utf8_when_decoding_then_none() -> None:
    assert decode_document(b"\xff\xfe\x00invalid") is None


def test_given_query_within_bounds_when_validating_then_trimmed_and_returned() -> None:
    assert validate_query("  fox over dog  ") == "fox over dog"


def test_given_empty_or_too_long_query_when_validating_then_rejected() -> None:
    assert validate_query("") is None
    assert validate_query("   ") is None
    assert validate_query("x" * 1_001) is None


def test_given_query_at_max_length_when_validating_then_accepted() -> None:
    assert validate_query("x" * 1_000) == "x" * 1_000


def test_given_short_document_when_pre_cutting_then_unchanged_and_not_truncated() -> None:
    result = pre_cut_document("short doc", max_doc_chars=4_096)
    assert result.text == "short doc"
    assert result.truncated is False


def test_given_long_document_when_pre_cutting_then_cut_and_marked_truncated() -> None:
    result = pre_cut_document("x" * 5_000, max_doc_chars=4_096)
    assert len(result.text) == 4_096
    assert result.truncated is True
