#!/bin/bash
# Block until a BEAM benchmark tmux session finishes (or dies).
# Usage: scripts/beam-wait.sh <logfile> [session-name]
#   e.g. scripts/beam-wait.sh .results/beam-single-20260412-080021.log beam
#
# Prints the log tail on exit. Exit codes:
#   0 = summary block seen (normal completion)
#   1 = tmux session died without summary
#   2 = panic / compile error detected
#   3 = bad args

set -u

LOG="${1:-}"
SESSION="${2:-beam}"

if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    echo "usage: $0 <logfile> [session-name]" >&2
    exit 3
fi

while true; do
    if grep -qE "thread.*panicked|error\[E[0-9]|error: test failed" "$LOG" 2>/dev/null; then
        echo "WAIT: panic/compile error detected"
        tail -60 "$LOG"
        exit 2
    fi
    if grep -qE "SINGLE CONVERSATION RESULT|BEAM 100K Full Benchmark.*Complete|BEAM 100K FULL RESULTS|^EXIT=" "$LOG" 2>/dev/null; then
        echo "WAIT: summary / EXIT line detected"
        tail -80 "$LOG"
        exit 0
    fi
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "WAIT: tmux session '$SESSION' gone without summary"
        tail -60 "$LOG"
        exit 1
    fi
    sleep 30
done
