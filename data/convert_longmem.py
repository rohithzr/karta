#!/usr/bin/env python3
"""Convert LongMemEval dataset to JSON for Rust consumption.

LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory (ICLR 2025)
Paper: https://arxiv.org/abs/2410.10813
Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Usage:
    python3 data/convert_longmem.py data/longmemeval_oracle_raw.json data/longmemeval.json
"""

import json
import sys


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/longmemeval_oracle_raw.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/longmemeval.json"

    print(f"Reading {input_file}...")
    with open(input_file) as f:
        entries = json.load(f)

    print(f"Found {len(entries)} questions")

    questions = []
    type_counts = {}

    for entry in entries:
        qtype = entry.get("question_type", "unknown")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

        # Extract sessions as user messages
        sessions = entry.get("haystack_sessions", [])
        dates = entry.get("haystack_dates", [])
        session_ids = entry.get("haystack_session_ids", [])

        user_messages = []
        for si, session in enumerate(sessions):
            date = dates[si] if si < len(dates) else ""
            sid = session_ids[si] if si < len(session_ids) else f"s{si}"

            if isinstance(session, list):
                for turn in session:
                    if isinstance(turn, dict):
                        role = turn.get("role", "")
                        content = turn.get("content", "")
                        if role == "user" and content.strip():
                            user_messages.append({
                                "role": "user",
                                "content": content,
                                "date": date,
                                "session_id": str(sid),
                            })
                    elif isinstance(turn, str):
                        user_messages.append({
                            "role": "user",
                            "content": turn,
                            "date": date,
                            "session_id": str(sid),
                        })
            elif isinstance(session, str):
                # Some entries have sessions as plain text
                user_messages.append({
                    "role": "user",
                    "content": session,
                    "date": date,
                    "session_id": str(sid),
                })

        questions.append({
            "question_id": entry.get("question_id", ""),
            "question_type": qtype,
            "question": entry.get("question", ""),
            "answer": entry.get("answer", ""),
            "question_date": entry.get("question_date", ""),
            "user_messages": user_messages,
            "num_sessions": len(sessions),
        })

    output = {
        "benchmark": "longmemeval",
        "split": "oracle",
        "total_questions": len(questions),
        "questions": questions,
    }

    print(f"Writing {output_file}...")
    print(f"  Questions: {len(questions)}")
    print(f"  Types: {json.dumps(type_counts, indent=2)}")
    print(f"  Avg user messages/question: {sum(len(q['user_messages']) for q in questions) / max(len(questions),1):.1f}")

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("Done.")


if __name__ == "__main__":
    main()
