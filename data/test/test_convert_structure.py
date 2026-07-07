"""T10 — structural invariants on the converter output.

We deliberately do not golden-compare the JSON byte-for-byte (key order,
dict ordering, whitespace would all be brittle). Instead we lock down the
semantic invariants the rest of the pipeline relies on:

  * Conversation count + per-conv shape match the parquet.
  * Sessions preserve the parquet's nested ``chat: list<list<turn>>``
    structure (no flattening).
  * Both ``user`` and ``assistant`` turns are emitted (we stopped
    filtering).
  * ``session_anchor`` and ``effective_reference_time`` are ISO-8601 UTC
    strings or ``null``.
  * ``total_user_turns`` matches a manual recount.
  * Within a session, turn order matches ``raw_index`` order.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import convert_beam

DATA_DIR = Path(__file__).resolve().parent.parent
PARQUET = DATA_DIR / "beam-100k.parquet"

# ISO-8601 UTC with a literal Z suffix — what the converter emits.
ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@pytest.fixture(scope="module")
def output(tmp_path_factory) -> dict:
    """Run the converter against the real parquet once per module."""
    if not PARQUET.exists():
        pytest.skip(f"missing parquet fixture: {PARQUET}")
    out_path = tmp_path_factory.mktemp("convert_out") / "beam.json"
    return convert_beam.convert(str(PARQUET), str(out_path))


@pytest.fixture(scope="module")
def conv1(output: dict) -> dict:
    for c in output["conversations"]:
        if c["id"] == "1":
            return c
    pytest.fail("conversation id=1 missing from converter output")


def test_conversation_count(output: dict) -> None:
    assert output["num_conversations"] == 20
    assert len(output["conversations"]) == 20


def test_conv1_session_shape(conv1: dict) -> None:
    """Conv 1 has 3 sessions with parquet-true totals."""
    assert len(conv1["sessions"]) == 3
    assert conv1["total_turns"] == 188
    assert conv1["total_user_turns"] == 94


def test_session_anchor_is_iso_or_null(output: dict) -> None:
    for conv in output["conversations"]:
        for sess in conv["sessions"]:
            anchor = sess["session_anchor"]
            assert anchor is None or ISO_UTC_RE.match(anchor), (
                f"conv {conv['id']} session {sess['session_index']}: "
                f"bad session_anchor={anchor!r}"
            )


def test_effective_reference_time_is_iso_or_null(output: dict) -> None:
    for conv in output["conversations"]:
        for sess in conv["sessions"]:
            for turn in sess["turns"]:
                ert = turn["effective_reference_time"]
                assert ert is None or ISO_UTC_RE.match(ert), (
                    f"conv {conv['id']} session {sess['session_index']} "
                    f"turn {turn['turn_index']}: bad effective_reference_time"
                    f"={ert!r}"
                )


def test_total_user_turns_matches_recount(output: dict) -> None:
    """The denormalized counter must equal the actual user-role count."""
    for conv in output["conversations"]:
        recount = sum(
            1
            for s in conv["sessions"]
            for t in s["turns"]
            if t["role"] == "user"
        )
        assert conv["total_user_turns"] == recount, (
            f"conv {conv['id']}: total_user_turns={conv['total_user_turns']} "
            f"but recount={recount}"
        )


def test_assistant_turns_preserved(conv1: dict) -> None:
    """Proves we stopped user-only filtering."""
    has_assistant = any(
        t["role"] == "assistant"
        for s in conv1["sessions"]
        for t in s["turns"]
    )
    assert has_assistant


def test_total_turns_equals_sum_of_session_turns(output: dict) -> None:
    for conv in output["conversations"]:
        s = sum(len(sess["turns"]) for sess in conv["sessions"])
        assert conv["total_turns"] == s


def test_raw_index_ordering_preserved(conv1: dict) -> None:
    """Within conv 1 session 0, turn order must match raw_index lex order
    on the user turns (the only ones that carry raw_index in parquet).

    BEAM's ``index`` is ``"<session>,<turn-in-session>"`` (e.g. ``1,1``,
    ``1,2``, …). Sorting numerically by the second component on the
    user-only subset must equal the emitted user-only order.
    """
    session0 = conv1["sessions"][0]
    user_turns = [
        t for t in session0["turns"] if t["raw_index"] is not None
    ]
    assert user_turns, "expected at least one turn with raw_index in conv1 s0"

    def sort_key(t: dict) -> tuple[int, int]:
        sess_str, turn_str = t["raw_index"].split(",")
        return (int(sess_str), int(turn_str))

    sorted_user = sorted(user_turns, key=sort_key)
    assert [t["raw_index"] for t in sorted_user] == [
        t["raw_index"] for t in user_turns
    ]


def test_top_level_passthroughs_present(conv1: dict) -> None:
    """The new top-level fields actually carry data (not empty strings)."""
    assert isinstance(conv1["subtopics"], list) and conv1["subtopics"]
    assert conv1["theme"]
    assert isinstance(conv1["user_profile"], dict) and conv1["user_profile"]
    assert conv1["conversation_plan"]
    assert conv1["narratives"]


def test_no_time_anchor_prefix_in_content(conv1: dict) -> None:
    """The ``[<time_anchor>]`` content prefix was a Rust harness behavior;
    the converter must not introduce it."""
    bad = []
    for sess in conv1["sessions"]:
        for turn in sess["turns"]:
            if turn["content"].startswith("["):
                # Allow content that legitimately starts with a square
                # bracket — but not a time_anchor-shaped one like
                # "[March-15-2024]".
                head = turn["content"][:32]
                if re.match(
                    r"^\[[A-Z][a-z]+-\d{1,2}-\d{4}\]", head
                ):
                    bad.append((sess["session_index"], turn["turn_index"]))
    assert not bad, f"time-anchor prefix leaked into content: {bad}"
