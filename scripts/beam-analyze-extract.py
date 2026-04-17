#!/usr/bin/env python3
"""Extract unified BEAM failure analysis dataset.

Joins debug JSONLs + BEAM dataset + SQLite stored notes into a single
analysis-ready JSON file. Each entry is one rubric item with full pipeline
traceability.
"""
import json
import sqlite3
import glob
import os
import sys


def find_debug_jsonl(conv_num):
    pattern = f'.results/beam-debug-{conv_num}-2026041[56]*.jsonl'
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def find_data_dir(conv_num):
    pattern = f'/tmp/karta-beam100k-{conv_num}-*'
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for d in dirs:
        if 'e4c28004' not in d:
            return d
    return dirs[0] if dirs else None


def get_stored_notes(data_dir):
    db = os.path.join(data_dir, 'karta.db')
    notes = {}
    try:
        c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        for row in c.execute('SELECT id, source_note_id, subject, ordinal FROM atomic_facts'):
            notes.setdefault('facts', []).append({
                'id': row[0], 'source_note_id': row[1],
                'subject': row[2], 'ordinal': row[3]
            })
        for row in c.execute('''SELECT id, episode_id, entities_json, date_range_json,
                               aggregations_json, events_json, digest_text FROM episode_digests'''):
            notes.setdefault('digests', []).append({
                'id': row[0], 'episode_id': row[1],
                'entities': row[2], 'date_range': row[3],
                'aggregations': row[4], 'events': row[5],
                'digest_text': row[6]
            })
        for row in c.execute('''SELECT id, scope_id, entity_timeline_json,
                               cross_aggregations_json, events_json, digest_text
                               FROM cross_episode_digests'''):
            notes.setdefault('cross_digests', []).append({
                'id': row[0], 'scope_id': row[1],
                'entity_timeline': row[2], 'cross_aggregations': row[3],
                'events': row[4], 'digest_text': row[5]
            })
        for row in c.execute('SELECT from_id, to_id, reason FROM links'):
            notes.setdefault('links', []).append({
                'from_id': row[0], 'to_id': row[1], 'reason': row[2]
            })
        for row in c.execute('SELECT id, session_id, topic_tags_json FROM episodes'):
            notes.setdefault('episodes', []).append({
                'id': row[0], 'session_id': row[1], 'topic_tags': row[2]
            })
        c.close()
    except Exception as e:
        print(f'  WARN: SQLite error for {data_dir}: {e}', file=sys.stderr)
    return notes


def main():
    with open('data/beam-100k.json') as f:
        beam = json.load(f)

    analysis = {'convs': [], 'summary': {}}

    for conv_num in range(1, 21):
        debug_path = find_debug_jsonl(conv_num)
        data_dir = find_data_dir(conv_num)
        if not debug_path:
            print(f'  SKIP conv {conv_num}: no debug JSONL', file=sys.stderr)
            continue

        debug_rows = [json.loads(l) for l in open(debug_path) if l.strip()]
        beam_conv = beam['conversations'][conv_num - 1]
        stored = get_stored_notes(data_dir) if data_dir else {}

        conv_entry = {
            'conv_num': conv_num,
            'conv_id': beam_conv['id'],
            'category': beam_conv['category'],
            'total_messages': len(beam_conv['user_messages']),
            'data_dir': data_dir,
            'debug_path': debug_path,
            'stored_note_counts': {
                'facts': len(stored.get('facts', [])),
                'digests': len(stored.get('digests', [])),
                'cross_digests': len(stored.get('cross_digests', [])),
                'links': len(stored.get('links', [])),
                'episodes': len(stored.get('episodes', [])),
            },
            'stored': stored,
            'questions': [],
        }

        for qi, row in enumerate(debug_rows):
            beam_q = beam_conv['questions'][qi]
            rubric_items = beam_q.get('rubric', [])
            if isinstance(rubric_items, dict):
                rubric_items = [f"{k}: {v}" for k, v in rubric_items.items()]

            scored_rubrics = row.get('rubric_scores', [])
            q_entry = {
                'q_index': qi,
                'ability': row['ability'],
                'question': row['question'],
                'reference_answer': row.get('reference_answer', ''),
                'beam_score': row['beam_score'],
                'query_mode': row['query_mode'],
                'notes_used': row['notes_used'],
                'note_ids': row['note_ids'],
                'reranker_best_score': row.get('reranker_best_score'),
                'has_contradiction': row.get('has_contradiction', False),
                'system_answer': row['system_answer'],
                'rubrics': [],
            }

            for ri, scored in enumerate(scored_rubrics):
                rubric_text = scored.get('item', rubric_items[ri] if ri < len(rubric_items) else '')
                score = scored.get('score')
                grade = scored.get('grade', '')
                q_entry['rubrics'].append({
                    'index': ri,
                    'item': rubric_text,
                    'score': score,
                    'grade': grade,
                    'passed': score is not None and score >= 0.5,
                })

            conv_entry['questions'].append(q_entry)
        analysis['convs'].append(conv_entry)

    total_rubrics = 0
    passed_rubrics = 0
    failed_rubrics = 0
    ability_stats = {}
    for conv in analysis['convs']:
        for q in conv['questions']:
            ab = q['ability']
            for r in q['rubrics']:
                if r['score'] is None:
                    continue
                total_rubrics += 1
                ability_stats.setdefault(ab, {'passed': 0, 'total': 0, 'failed_items': []})
                ability_stats[ab]['total'] += 1
                if r['passed']:
                    passed_rubrics += 1
                    ability_stats[ab]['passed'] += 1
                else:
                    failed_rubrics += 1
                    ability_stats[ab]['failed_items'].append({
                        'conv_num': conv['conv_num'],
                        'category': conv['category'],
                        'q_index': q['q_index'],
                        'rubric_index': r['index'],
                        'rubric_item': r['item'],
                        'score': r['score'],
                        'question': q['question'][:200],
                        'ability': ab,
                    })

    analysis['summary'] = {
        'total_rubrics': total_rubrics,
        'passed_rubrics': passed_rubrics,
        'failed_rubrics': failed_rubrics,
        'pass_rate': passed_rubrics / total_rubrics if total_rubrics else 0,
        'ability_stats': {k: {'passed': v['passed'], 'total': v['total'],
                              'failure_count': len(v['failed_items'])}
                          for k, v in ability_stats.items()},
    }

    out_path = '.results/beam-analysis.json'
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f'Wrote {out_path}: {len(analysis["convs"])} convs, {total_rubrics} rubric items ({failed_rubrics} failures to analyze)')

    for ab in sorted(ability_stats, key=lambda k: ability_stats[k]['passed']/max(ability_stats[k]['total'],1)):
        s = ability_stats[ab]
        print(f'  {ab:<28} {s["passed"]}/{s["total"]} ({s["passed"]/s["total"]*100:.0f}%) — {len(s["failed_items"])} failures')


if __name__ == '__main__':
    main()
