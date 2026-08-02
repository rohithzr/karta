#!/bin/bash
# karta-capture.sh — Droid capture hook for the six non-orient lifecycle events.
#
# Usage: karta-capture.sh <event_type>
#   where <event_type> is one of:
#     user_prompt, observation, turn_summary, subagent_result, session_end, pre_compact
#
# The hook payload is read from stdin, merged with an explicit `event` field,
# and POSTed to the karta-mcp /capture endpoint. The explicit event override
# guarantees the mapping matches the server-side event types used by
# crates/karta-mcp/src/capture.rs.
#
# The capture port is read from KARTA_CAPTURE_PORT (default 3137); it is NOT
# hardcoded.

set -uo pipefail

EVENT_TYPE="$1"
KARTA_CAPTURE_PORT="${KARTA_CAPTURE_PORT:-3137}"

case "$EVENT_TYPE" in
  user_prompt|observation|turn_summary|subagent_result|session_end|pre_compact)
    ;;
  *)
    echo "karta-capture.sh: unknown event type '$EVENT_TYPE'" >&2
    exit 1
    ;;
esac

# Add the server-side event type to the payload. The original hook_event_name
# is preserved, but the explicit `event` field takes precedence on the server.
jq --arg event "$EVENT_TYPE" '. + {event: $event}' \
  | curl -s -S -f -X POST "http://127.0.0.1:${KARTA_CAPTURE_PORT}/capture" \
      -H "Content-Type: application/json" \
      -d @- \
  > /dev/null
