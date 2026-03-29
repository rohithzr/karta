#!/usr/bin/env python3
"""Convert BEAM 100K parquet to JSON for Rust consumption."""

import ast
import json
import numpy as np
import pandas as pd
import sys

def parse_probing_questions(pq_str):
    """Parse the Python literal string into a proper dict."""
    try:
        return ast.literal_eval(pq_str)
    except:
        return {}

def clean_content(content):
    """Remove the ->-> timestamp suffix from message content."""
    if "->->" in content:
        return content.split("->->")[0].strip()
    return content

def extract_chat_turns(chat_data):
    """Extract turns from the chat structure (array of sessions of turn dicts)."""
    turns = []
    if chat_data is None:
        return turns
    for session in chat_data:
        if session is None:
            continue
        for turn in session:
            if isinstance(turn, dict):
                content = clean_content(turn.get("content", ""))
                if content:
                    turns.append({
                        "role": turn.get("role", ""),
                        "content": content,
                        "time_anchor": turn.get("time_anchor", "") or "",
                    })
    return turns

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "beam-100k.parquet"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "beam-100k.json"

    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Found {len(df)} conversations")

    conversations = []
    total_questions = 0
    total_user_msgs = 0

    for idx, row in df.iterrows():
        conv_id = row.get("conversation_id", f"conv_{idx}")

        # Parse chat turns
        chat_turns = extract_chat_turns(row.get("chat"))
        user_messages = [t for t in chat_turns if t["role"] == "user"]
        total_user_msgs += len(user_messages)

        # Parse probing questions
        pq_raw = row.get("probing_questions", "{}")
        pq = parse_probing_questions(pq_raw) if isinstance(pq_raw, str) else {}
        if isinstance(pq_raw, dict):
            pq = pq_raw

        # Flatten probing questions into a list
        questions = []
        if isinstance(pq, dict):
            for ability, q_list in pq.items():
                if isinstance(q_list, list):
                    for q in q_list:
                        if isinstance(q, dict):
                            questions.append({
                                "ability": ability.lower().replace(" ", "_"),
                                "question": q.get("question", q.get("probing_question", "")),
                                "reference_answer": q.get("reference_answer", q.get("expected_answer", "")),
                                "rubric": q.get("rubric", q.get("rubric_items", [])),
                            })

        total_questions += len(questions)

        # Get metadata
        seed = row.get("conversation_seed", {})
        if isinstance(seed, str):
            try:
                seed = ast.literal_eval(seed)
            except:
                seed = {}

        conversations.append({
            "id": str(conv_id),
            "category": seed.get("category", "general") if isinstance(seed, dict) else "general",
            "title": seed.get("title", "") if isinstance(seed, dict) else "",
            "user_messages": user_messages,
            "total_turns": len(chat_turns),
            "questions": questions,
        })

    output = {
        "split": "100K",
        "num_conversations": len(conversations),
        "total_questions": total_questions,
        "conversations": conversations,
    }

    print(f"Writing {output_file}...")
    print(f"  Conversations: {len(conversations)}")
    print(f"  Total questions: {total_questions}")
    print(f"  Total user messages: {total_user_msgs}")
    print(f"  Avg user messages/conv: {total_user_msgs / max(len(conversations),1):.0f}")
    print(f"  Avg questions/conv: {total_questions / max(len(conversations),1):.0f}")

    # Convert numpy types to native Python for JSON serialization
    def default_handler(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=default_handler)

    # Quick stats
    for c in conversations[:3]:
        print(f"\n  Conv '{c['id']}': {len(c['user_messages'])} user msgs, {len(c['questions'])} questions")
        if c['questions']:
            abilities = set(q['ability'] for q in c['questions'])
            print(f"    Abilities: {', '.join(sorted(abilities))}")

    print("\nDone.")

if __name__ == "__main__":
    main()
