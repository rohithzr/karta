#!/bin/bash
# karta-orient.sh — Droid SessionStart hook.
#
# Receives the Droid SessionStart JSON on stdin, forwards it to the karta-mcp
# /orient endpoint, and echoes the response on stdout so Droid can append it
# as additionalContext.
#
# The capture port is read from KARTA_CAPTURE_PORT (default 3137); it is NOT
# hardcoded, so the same script works when karta-mcp is run on a custom port.

set -uo pipefail

KARTA_CAPTURE_PORT="${KARTA_CAPTURE_PORT:-3137}"

# Forward the payload unchanged; /orient derives its query from agent/project/cwd.
curl -s -S -f -X POST "http://127.0.0.1:${KARTA_CAPTURE_PORT}/orient" \
  -H "Content-Type: application/json" \
  -d @- || true
