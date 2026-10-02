"""T18 — converter logs every malformed turn drop with full coordinates.

Silent drops are a correctness hazard: if 5% of conversations have
malformed turns and we don't notice, BEAM scores reflect ingestion loss
not memory quality. We assert that ``extract_sessions`` emits a WARNING
log line including conv id, session index, turn id, and a drop reason
for both supported failure modes (non-dict entry, missing content).
"""

from __future__ import annotations

import logging

import pytest

import convert_beam


def _make_chat() -> list[list]:
    """Build a synthetic ``chat`` shaped like the parquet's
    ``list<list<turn>>`` with three malformed turns:

      session 0: ["not a dict", {good turn}, {missing content}]
      session 1: [{good turn}]
    """
    good_turn = {
        "id": 100,
        "index": "1,1",
        "role": "user",
        "content": "real content here",
        "time_anchor": "March-15-2024",
        "question_type": "main_question",
    }
    missing_content_turn = {
        "id": 101,
        "index": "1,2",
        "role": "user",
        "content": "",
        "time_anchor": "",
        "question_type": None,
    }
    second_session_turn = {
        "id": 200,
        "index": "2,1",
        "role": "user",
        "content": "session 2 content",
        "time_anchor": "April-05-2024",
        "question_type": "main_question",
    }
    return [
        ["not a dict here", good_turn, missing_content_turn],
        [second_session_turn],
    ]


def test_logs_dropped_non_dict_turn(caplog: pytest.LogCaptureFixture) -> None:
    chat = _make_chat()
    with caplog.at_level(logging.WARNING, logger="convert_beam"):
        sessions = convert_beam.extract_sessions(chat, conv_id="conv-fixture")

    matches = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "not a dict" in r.getMessage()
    ]
    assert matches, "expected at least one WARNING for the non-dict turn"
    msg = matches[0].getMessage()
    assert "conv_id=conv-fixture" in msg
    assert "session_index=0" in msg
    assert "reason=not a dict" in msg
    # And the surviving turns are still present.
    assert len(sessions) == 2
    assert len(sessions[0]["turns"]) == 1  # only the good turn
    assert len(sessions[1]["turns"]) == 1


def test_logs_dropped_missing_content_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chat = _make_chat()
    with caplog.at_level(logging.WARNING, logger="convert_beam"):
        convert_beam.extract_sessions(chat, conv_id="conv-fixture")

    matches = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "missing content" in r.getMessage()
    ]
    assert matches, (
        "expected at least one WARNING for the missing-content turn"
    )
    msg = matches[0].getMessage()
    assert "conv_id=conv-fixture" in msg
    assert "session_index=0" in msg
    # The malformed turn carried id=101 — the log line must include it so
    # an operator can find the offending row in the parquet.
    assert "turn_id=101" in msg
    assert "reason=missing content" in msg


def test_no_warnings_on_clean_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Clean input should not emit drop warnings — guards against false
    positives flooding the log on normal data."""
    clean_chat = [
        [
            {
                "id": 1,
                "index": "1,1",
                "role": "user",
                "content": "hi",
                "time_anchor": "March-15-2024",
                "question_type": "main_question",
            },
            {
                "id": 2,
                "index": None,
                "role": "assistant",
                "content": "hello",
                "time_anchor": None,
                "question_type": None,
            },
        ]
    ]
    with caplog.at_level(logging.WARNING, logger="convert_beam"):
        sessions = convert_beam.extract_sessions(
            clean_chat, conv_id="conv-clean"
        )
    drop_warnings = [
        r for r in caplog.records if "dropping malformed" in r.getMessage()
    ]
    assert not drop_warnings
    assert len(sessions) == 1
    assert len(sessions[0]["turns"]) == 2
