#!/usr/bin/env python3
"""Convert BEAM 100K parquet to JSON for Rust consumption.

This converter emits the post-STEP1 (`ClockContext`) shape:

  conversations[].sessions[].turns[]

instead of the pre-STEP1 user-only flat list. Both `user` and `assistant`
turns are preserved. Each turn carries a pre-resolved
`effective_reference_time` (ISO-8601 UTC) — the harness performs no
session-level fallback logic, the converter is the single source of truth
for carry-forward.

Naive BEAM date strings (e.g. ``"March-15-2024"``) are normalized to UTC
midnight (``"2024-03-15T00:00:00Z"``). If a future dataset provides
offset-aware timestamps, the parser must be extended to respect the offset
and convert to UTC; today's parser only knows the BEAM format.

Malformed turns (non-dict entries, missing/empty content) are LOGGED with
conv id + session index + turn id + reason and then dropped. Silent drops
are a correctness hazard — a missing turn skews downstream BEAM scores.
"""

import ast
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("convert_beam")

# BEAM ships time anchors as `<Month>-<DD>-<YYYY>` (e.g. "March-15-2024").
# Naive — no timezone. Per STEP1 codex #9 we treat these as UTC midnight.
BEAM_TIME_ANCHOR_FORMAT = "%B-%d-%Y"


def parse_time_anchor(raw: Optional[str]) -> Optional[datetime]:
    """Parse a BEAM ``time_anchor`` string into a tz-aware UTC datetime.

    Returns ``None`` for empty / falsy / unparseable inputs (the caller
    decides whether that's a warning or just a missing-anchor turn).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        naive = datetime.strptime(s, BEAM_TIME_ANCHOR_FORMAT)
    except ValueError:
        logger.warning("unparseable time_anchor: %r", s)
        return None
    return naive.replace(tzinfo=timezone.utc)


def to_iso_utc(dt: Optional[datetime]) -> Optional[str]:
    """Render a tz-aware datetime as ISO-8601 with a literal ``Z`` suffix."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_probing_questions(pq_str: str) -> dict:
    """Parse the Python literal string into a proper dict (best-effort)."""
    try:
        return ast.literal_eval(pq_str)
    except (ValueError, SyntaxError):
        return {}


def clean_content(content: str) -> str:
    """Strip the BEAM ``->-> N,N`` index suffix from message content.

    Note: we do NOT add a ``[time_anchor]`` prefix — that was a Rust
    harness behavior and lives in `effective_reference_time` only now.
    """
    if "->->" in content:
        return content.split("->->")[0].strip()
    return content


def _coerce_to_dict(value: Any) -> Optional[dict]:
    """Best-effort coerce parquet metadata fields (dict / str repr / numpy)
    into a plain Python dict. Returns None if the value isn't shaped like
    a mapping at all.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _coerce_to_list(value: Any) -> list:
    """Best-effort coerce parquet list-ish fields (numpy array, tuple,
    list, ``None``) into a plain Python list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def extract_sessions(
    chat_data: Optional[Iterable], conv_id: str
) -> list[dict]:
    """Build the ``sessions[].turns[]`` structure for one conversation.

    Returns a list of ``{session_index, session_anchor, turns}`` dicts,
    where each turn carries ``turn_index``, ``role``, ``content``,
    ``time_anchor``, ``effective_reference_time``, ``question_type``,
    ``raw_index``.

    Malformed turns are logged + skipped. The carry-forward rule:

    - ``session_anchor`` = first non-empty parsed ``time_anchor`` in the
      session's surviving turns.
    - For each turn: ``effective_reference_time`` = the turn's own
      anchor if set, else the session anchor (carry-forward).
    - If a session has zero anchored turns, ``session_anchor`` and every
      turn's ``effective_reference_time`` are ``null``, and we emit a
      WARNING (this should not happen on real BEAM data).
    """
    sessions: list[dict] = []
    if chat_data is None:
        return sessions

    for session_index, raw_session in enumerate(chat_data):
        if raw_session is None:
            logger.warning(
                "dropping malformed session — conv_id=%s, session_index=%d, "
                "reason=session is None",
                conv_id,
                session_index,
            )
            continue

        # First pass: keep only well-formed turns (with cleaned content).
        # Drops are logged with full coordinates.
        kept: list[dict] = []
        for raw_turn in raw_session:
            if not isinstance(raw_turn, dict):
                logger.warning(
                    "dropping malformed turn — conv_id=%s, session_index=%d, "
                    "turn_id=?, reason=not a dict",
                    conv_id,
                    session_index,
                )
                continue
            turn_id = raw_turn.get("id", "?")
            raw_content = raw_turn.get("content", "") or ""
            content = clean_content(raw_content)
            if not content:
                logger.warning(
                    "dropping malformed turn — conv_id=%s, session_index=%d, "
                    "turn_id=%s, reason=missing content",
                    conv_id,
                    session_index,
                    turn_id,
                )
                continue
            kept.append((raw_turn, content))

        # Resolve session anchor from the first turn that has one.
        session_anchor_dt: Optional[datetime] = None
        for raw_turn, _ in kept:
            ta = raw_turn.get("time_anchor")
            parsed = parse_time_anchor(ta)
            if parsed is not None:
                session_anchor_dt = parsed
                break

        if session_anchor_dt is None and kept:
            logger.warning(
                "session has no time_anchor — conv_id=%s, session_index=%d "
                "(emitting null effective_reference_time for all turns)",
                conv_id,
                session_index,
            )

        session_anchor_iso = to_iso_utc(session_anchor_dt)

        turns: list[dict] = []
        for turn_index, (raw_turn, content) in enumerate(kept):
            ta_raw = raw_turn.get("time_anchor")
            turn_anchor_dt = parse_time_anchor(ta_raw)
            effective_dt = turn_anchor_dt or session_anchor_dt

            turns.append(
                {
                    "turn_index": turn_index,
                    "role": raw_turn.get("role", ""),
                    "content": content,
                    # Preserve the original (possibly empty) anchor string
                    # so consumers can tell "this turn was anchored
                    # explicitly" vs "carried forward".
                    "time_anchor": (str(ta_raw).strip() or None)
                    if ta_raw
                    else None,
                    "effective_reference_time": to_iso_utc(effective_dt),
                    "question_type": raw_turn.get("question_type"),
                    # Parquet uses `index` ("1,1") for the BEAM internal
                    # coord; the plan calls this `raw_index`.
                    "raw_index": raw_turn.get("index"),
                }
            )

        sessions.append(
            {
                "session_index": session_index,
                "session_anchor": session_anchor_iso,
                "turns": turns,
            }
        )

    return sessions


def build_conversation(idx: int, row: pd.Series) -> tuple[dict, int]:
    """Produce one conversation dict and return (conv, total_questions)."""
    conv_id = str(row.get("conversation_id", f"conv_{idx}"))
    seed = _coerce_to_dict(row.get("conversation_seed")) or {}
    user_profile = _coerce_to_dict(row.get("user_profile")) or {}
    conversation_plan = row.get("conversation_plan")
    narratives = row.get("narratives")

    sessions = extract_sessions(row.get("chat"), conv_id)

    total_turns = sum(len(s["turns"]) for s in sessions)
    total_user_turns = sum(
        1 for s in sessions for t in s["turns"] if t["role"] == "user"
    )

    # Probing questions.
    pq_raw = row.get("probing_questions", "{}")
    if isinstance(pq_raw, dict):
        pq = pq_raw
    elif isinstance(pq_raw, str):
        pq = parse_probing_questions(pq_raw)
    else:
        pq = {}

    questions: list[dict] = []
    if isinstance(pq, dict):
        for ability, q_list in pq.items():
            if not isinstance(q_list, list):
                continue
            for q in q_list:
                if not isinstance(q, dict):
                    continue
                questions.append(
                    {
                        "ability": ability.lower().replace(" ", "_"),
                        "question": q.get(
                            "question", q.get("probing_question", "")
                        ),
                        "reference_answer": q.get(
                            "reference_answer", q.get("expected_answer", "")
                        ),
                        "rubric": q.get("rubric", q.get("rubric_items", [])),
                    }
                )

    conv = {
        "id": conv_id,
        "category": seed.get("category", "general"),
        "title": seed.get("title", ""),
        "subtopics": _coerce_to_list(seed.get("subtopics")),
        "theme": seed.get("theme", ""),
        "user_profile": user_profile,
        "conversation_plan": (
            str(conversation_plan) if conversation_plan is not None else ""
        ),
        "narratives": str(narratives) if narratives is not None else "",
        "sessions": sessions,
        "total_turns": total_turns,
        "total_user_turns": total_user_turns,
        "questions": questions,
    }
    return conv, len(questions)


def convert(input_file: str, output_file: str) -> dict:
    """Read a BEAM parquet file and write the post-STEP1 JSON shape.

    Returns the in-memory ``output`` dict so callers (tests) can inspect
    it directly without re-reading the file.
    """
    logger.info("Reading %s", input_file)
    df = pd.read_parquet(input_file)
    logger.info("Found %d conversations", len(df))

    conversations: list[dict] = []
    total_questions = 0
    total_user_msgs = 0

    for idx, row in df.iterrows():
        conv, n_q = build_conversation(idx, row)
        conversations.append(conv)
        total_questions += n_q
        total_user_msgs += conv["total_user_turns"]

    output = {
        "split": "100K",
        "num_conversations": len(conversations),
        "total_questions": total_questions,
        "conversations": conversations,
    }

    logger.info("Writing %s", output_file)
    logger.info("  Conversations: %d", len(conversations))
    logger.info("  Total questions: %d", total_questions)
    logger.info("  Total user messages: %d", total_user_msgs)
    if conversations:
        logger.info(
            "  Avg user messages/conv: %.0f",
            total_user_msgs / len(conversations),
        )
        logger.info(
            "  Avg questions/conv: %.0f",
            total_questions / len(conversations),
        )

    def default_handler(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=default_handler)

    return output


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    input_file = sys.argv[1] if len(sys.argv) > 1 else "beam-100k.parquet"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "beam-100k.json"
    output = convert(input_file, output_file)

    # Quick stats so the operator can eyeball success.
    for c in output["conversations"][:3]:
        n_user = c["total_user_turns"]
        n_total = c["total_turns"]
        n_sessions = len(c["sessions"])
        print(
            f"\n  Conv '{c['id']}': {n_sessions} sessions, "
            f"{n_total} turns ({n_user} user), "
            f"{len(c['questions'])} questions"
        )
        if c["questions"]:
            abilities = sorted({q["ability"] for q in c["questions"]})
            print(f"    Abilities: {', '.join(abilities)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
