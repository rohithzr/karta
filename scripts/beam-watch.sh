#!/bin/bash
# BEAM benchmark watcher — live progress, scores, and error tracking
# Usage: ./scripts/beam-watch.sh [logfile]
# If no logfile, watches the most recent .results/beam-full-*.log

set -euo pipefail

LOG="${1:-$(ls -t .results/beam-full-*.log 2>/dev/null | head -1)}"

if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    echo "No log file found. Start benchmark first."
    echo "Usage: ./scripts/beam-watch.sh [logfile]"
    exit 1
fi

ERROR_LOG=".results/beam-errors-$(date +%Y%m%d-%H%M%S).log"

while true; do
    clear

    echo "=== BEAM Benchmark Watcher ==="
    echo "Log: $LOG"
    echo "Size: $(wc -c < "$LOG" | tr -d ' ') bytes"
    echo ""

    # Progress: count completed questions and conversations
    QUESTIONS=$(grep -c "BEAM score:" "$LOG" 2>/dev/null || true)
    QUESTIONS=${QUESTIONS:-0}
    QUESTIONS=$(echo "$QUESTIONS" | tr -d '[:space:]')
    CONVS_DONE=$(grep -c "BEAM 100K" "$LOG" 2>/dev/null || true)
    CONVS_DONE=${CONVS_DONE:-0}
    CONVS_DONE=$(echo "$CONVS_DONE" | tr -d '[:space:]')
    echo "Progress: $QUESTIONS/400 questions scored ($CONVS_DONE convs started)"
    echo ""

    # Running score
    if [ "$QUESTIONS" -gt 0 ]; then
        python3 - "$LOG" <<'PYEOF'
import re, sys
scores, abilities = [], {}
ability = None
with open(sys.argv[1]) as f:
    for line in f:
        am = re.search(r'\[(\w+)\] Q\d+:', line)
        if am: ability = am.group(1)
        m = re.search(r'Q\d+ BEAM score: ([\d.]+)', line)
        if m and ability:
            s = float(m.group(1))
            scores.append(s)
            abilities.setdefault(ability, []).append(s)
            ability = None
if scores:
    passed = sum(1 for s in scores if s >= 0.5)
    avg = sum(scores)/len(scores)
    print(f'Overall: {avg:.1%} avg | {passed}/{len(scores)} passed (>= 0.5)')
    print()
    print('Per-ability:')
    for name in sorted(abilities):
        sc = abilities[name]
        a = sum(sc)/len(sc)
        p = sum(1 for s in sc if s >= 0.5)
        bar = '#' * int(a * 20)
        print(f'  {name:30s} {a:5.0%} {bar:20s} ({p}/{len(sc)})')
PYEOF
    fi

    echo ""
    echo "--- Last 20 lines ---"
    tail -20 "$LOG"

    # Extract errors to error log and show last 15
    grep -iE "(panic|thread.*panicked|WARN: Note|failed:)" "$LOG" | grep -v "^warning:" > "$ERROR_LOG" 2>/dev/null || true
    ERROR_COUNT=$(wc -l < "$ERROR_LOG" 2>/dev/null | tr -d ' ')
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo ""
        echo "=== ERRORS ($ERROR_COUNT total, last 15) ==="
        tail -15 "$ERROR_LOG"
    fi

    sleep 10
done
