#!/usr/bin/env python3
"""Convert LOCOMO dataset JSON to a flat format for Rust consumption.

LOCOMO (Maharana et al., ACL 2024) is a benchmark for long-term conversational memory.
Dataset: https://github.com/snap-research/locomo

Download the raw dataset first:
  curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o data/locomo10_raw.json

Then run:
  python3 data/convert_locomo.py data/locomo10_raw.json data/locomo.json
"""

import json
import sys

# LOCOMO QA category mapping (from the paper, Maharana et al. ACL 2024)
CATEGORY_MAP = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


def extract_sessions(conversation):
    """Extract ordered sessions from a conversation object.

    Sessions are stored as session_1, session_2, etc. with corresponding
    session_1_date_time timestamps.
    """
    sessions = []
    idx = 1
    while True:
        key = f"session_{idx}"
        if key not in conversation:
            break
        date_key = f"{key}_date_time"
        timestamp = conversation.get(date_key, "")
        turns = conversation[key]
        sessions.append({
            "session_index": idx,
            "timestamp": timestamp,
            "turns": turns,
        })
        idx += 1
    return sessions


def flatten_turns(sessions, speaker_a, speaker_b):
    """Flatten all session turns into a list of user messages for ingestion.

    Each message includes the session timestamp, speaker name, and text.
    We treat all dialogue turns as note-worthy content.
    """
    messages = []
    for session in sessions:
        ts = session["timestamp"]
        for turn in session["turns"]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")
            if not text.strip():
                continue
            messages.append({
                "speaker": speaker,
                "text": text,
                "dia_id": dia_id,
                "session_index": session["session_index"],
                "timestamp": ts,
            })
    return messages


def convert_qa(qa_list):
    """Convert QA items to a normalized format with string category labels."""
    questions = []
    for qa in qa_list:
        cat_num = qa.get("category", 0)
        category = CATEGORY_MAP.get(cat_num, f"unknown_{cat_num}")

        question = {
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "category": category,
            "category_id": cat_num,
            "evidence": qa.get("evidence", []),
        }

        # Adversarial questions have an additional field
        if "adversarial_answer" in qa:
            question["adversarial_answer"] = qa["adversarial_answer"]

        questions.append(question)
    return questions


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/locomo10_raw.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/locomo.json"

    print(f"Reading {input_file}...")
    with open(input_file) as f:
        raw = json.load(f)

    # locomo10.json is an array of conversation objects
    if not isinstance(raw, list):
        raw = [raw]

    print(f"Found {len(raw)} conversation(s)")

    conversations = []
    total_questions = 0
    total_messages = 0
    category_counts = {}

    for idx, sample in enumerate(raw):
        conv_data = sample.get("conversation", {})
        qa_data = sample.get("qa", [])

        speaker_a = conv_data.get("speaker_a", "Speaker A")
        speaker_b = conv_data.get("speaker_b", "Speaker B")

        # Extract sessions and flatten turns
        sessions = extract_sessions(conv_data)
        messages = flatten_turns(sessions, speaker_a, speaker_b)
        total_messages += len(messages)

        # Convert QA
        questions = convert_qa(qa_data)
        total_questions += len(questions)

        for q in questions:
            cat = q["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        conv_id = sample.get("sample_id", f"conv_{idx}")

        conversations.append({
            "id": str(conv_id),
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "num_sessions": len(sessions),
            "messages": messages,
            "questions": questions,
        })

    output = {
        "benchmark": "LOCOMO",
        "source": "https://github.com/snap-research/locomo",
        "num_conversations": len(conversations),
        "total_questions": total_questions,
        "total_messages": total_messages,
        "category_map": CATEGORY_MAP,
        "conversations": conversations,
    }

    print(f"Writing {output_file}...")
    print(f"  Conversations: {len(conversations)}")
    print(f"  Total messages: {total_messages}")
    print(f"  Total questions: {total_questions}")

    if total_messages > 0:
        print(f"  Avg messages/conv: {total_messages / len(conversations):.0f}")
    if total_questions > 0:
        print(f"  Avg questions/conv: {total_questions / len(conversations):.1f}")

    print(f"\n  Questions by category:")
    for cat in sorted(category_counts.keys()):
        print(f"    {cat:20s} {category_counts[cat]}")

    # Preview
    for c in conversations[:3]:
        print(f"\n  Conv '{c['id']}' ({c['speaker_a']} & {c['speaker_b']}): "
              f"{c['num_sessions']} sessions, {len(c['messages'])} msgs, "
              f"{len(c['questions'])} questions")

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
