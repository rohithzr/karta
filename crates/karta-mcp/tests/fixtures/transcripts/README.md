# Transcript fixtures

These golden fixtures represent the JSONL transcript formats that
`karta-mcp`'s fallback sweep can parse.

## `droid-session.jsonl`

A synthetic Droid session transcript based on attested real Droid transcript
shapes. Each line is a JSON object with a top-level `type` field. The relevant
shapes are:

- `session_start`: session metadata (`sessionId`, `agent`, `project`, `cwd`).
- `message` with `message.role = "user"`:
  - `content` is an array of blocks.
  - `text` blocks carry the user prompt. On real Droid transcripts these
    messages do **not** have a `hookEventName`.
  - `tool_result` blocks carry the result of a previous tool call
    (`tool_use_id`, `content`, `is_error`). On real Droid transcripts these
    messages also do **not** have a `hookEventName`.
  - `hookEventName` appears only on separate hook-execution messages, which
    usually have an empty `content` array and carry metadata such as
    `hookCommands`, `hookResults`, and `hookToolCallId`. The only attested
    value on this machine is `PostToolUse`.
  - `SubagentStop` messages carry `taskName`/`taskResult` fields and may have
    an empty `content` array.
  - `PreCompact` messages carry `reason` and `hookToolCallId` and have an empty
    `content` array.
- `message` with `message.role = "assistant":
  - `content` may contain `thinking`, `text`, and `tool_use` blocks.
  - `tool_use` blocks have `id`, `name`, and `input`.
- `message` with `stop = true` or `hookEventName = "SessionEnd"` marks the end
  of the session.

## `claude-code-session.jsonl`

A synthetic Claude Code transcript. Claude Code's internal transcript format is
versioned and not publicly stable, so this fixture documents the expected hook-
input shapes that the parser recognizes:

- `hook_event_name = "SessionStart"`: context fields (`session_id`,
  `transcript_path`, `cwd`).
- `hook_event_name = "UserPromptSubmit"`: `prompt` is the user message.
- `hook_event_name = "PostToolUse"`: `tool_name`, `tool_input`, and
  `tool_response` describe a tool execution.
- `hook_event_name = "Stop"`: `last_assistant_message` is the assistant's
  final message in the turn.
- `hook_event_name = "SubagentStop"`: `task_name`, `task_result`, and
  `task_error` describe a subagent result.
- `hook_event_name = "SessionEnd"`: `reason`, `session_duration_ms`, and
  `message_count`.

The parser is intentionally lenient: unknown event types and missing optional
fields are skipped rather than treated as fatal errors.
