# karta-mcp

MCP server + full-hook auto-capture for Karta.

`karta-mcp serve` runs a single process that exposes:

- a stdio MCP server with the `karta_*` tools,
- a localhost HTTP capture endpoint for Droid/Claude Code lifecycle hooks,
- a durable SQLite-backed capture queue that drains into `karta_core`.

## Install

`karta-mcp` is a workspace member of the Karta repository. Build it from source with Cargo:

```bash
cd /path/to/karta
cargo build -p karta-mcp
```

The binary is produced at `target/debug/karta-mcp`. For release builds:

```bash
cargo build --release -p karta-mcp
```

No external services are required for the test/validation path (`serve --mock` uses deterministic mock providers). For live operation, see the environment variables below and the live-validation notes at the end of this README.

## Configuration

All configuration is env-only (12-factor). The wrapper never reads `.env` files.

| Variable                | Default          | Purpose                                      |
| ----------------------- | ---------------- | -------------------------------------------- |
| `KARTA_STORE_DIR`       | `~/.karta/store` | Directory holding `karta.db`                 |
| `KARTA_CAPTURE_PORT`    | `3137`           | Port for the HTTP capture endpoint           |
| `KARTA_PRECOMPACT`      | unset            | Set to `1` to opt in to `PreCompact` capture |
| `OPENAI_API_BASE`       | —                | OpenAI-compatible endpoint for live mode     |
| `OPENAI_API_KEY`        | —                | API key for the above                        |
| `KARTA_CORE_MODEL`      | —                | Chat model for live mode                     |
| `KARTA_EMBEDDING_MODEL` | —                | Embedding model for live mode                |

`HOME` is required when `KARTA_STORE_DIR` is unset so the default `~/.karta/store` path can be resolved.

## MCP client configuration

Register `karta-mcp` as a stdio MCP server in your client. The server speaks MCP over stdin/stdout; all logs are written to stderr.

### Droid

Use the pre-built binary (replace the path with your own):

```bash
droid mcp add karta-mcp -- /path/to/karta/target/debug/karta-mcp serve --mock
```

To pass environment variables (for example a custom store directory or capture port):

```bash
droid mcp add karta-mcp -- /path/to/karta/target/debug/karta-mcp serve --mock \
  --env KARTA_STORE_DIR=/Users/aj/.karta/store \
  --env KARTA_CAPTURE_PORT=3137
```

Verify the connection:

```bash
droid mcp list
```

You can also commit a project-level `.factory/mcp.json`:

```json
{
  "mcpServers": {
    "karta-mcp": {
      "type": "stdio",
      "command": "/path/to/karta/target/debug/karta-mcp",
      "args": ["serve", "--mock"],
      "env": {
        "KARTA_STORE_DIR": "/Users/aj/.karta/store",
        "KARTA_CAPTURE_PORT": "3137"
      }
    }
  }
}
```

### Claude Code

Register the server from the shell:

```bash
claude mcp add karta-mcp -- /path/to/karta/target/debug/karta-mcp serve --mock
```

Or add a project-scoped `.mcp.json` in the repository root:

```json
{
  "mcpServers": {
    "karta-mcp": {
      "type": "stdio",
      "command": "/path/to/karta/target/debug/karta-mcp",
      "args": ["serve", "--mock"]
    }
  }
}
```

Check the connection:

```bash
claude mcp list
```

For live mode, remove `--mock` from the command and ensure the LLM environment variables are set.

## Running

```bash
# Live mode (requires a local LLM endpoint such as LM Studio on localhost:1234)
karta-mcp serve

# Test / validation mode with deterministic mock providers
karta-mcp serve --mock
```

## Capture / orient contract

The HTTP endpoint is bound to `127.0.0.1:$KARTA_CAPTURE_PORT` (default `3137`) and is intended only for lifecycle hooks running on the same machine.

### `POST /orient`

Called once at session start. The body is forwarded JSON; `/orient` derives a query and returns relevant context synchronously.

- Request body: any JSON. Preferred fields:
  - `query` — used directly as the retrieval query.
  - `agent`, `project`, `cwd` — combined into a query when `query` is absent.
- Response: `200 OK` with `{"context": "...", "note_ids": ["..."]}`.
- `SessionStart` hook events must be sent here, not to `/capture`.

### `POST /capture`

Called for every other wired lifecycle event. The endpoint returns `202 Accepted` as soon as a durable row is inserted into the SQLite `capture_queue`; the background worker drains the row to `karta_core` asynchronously.

- Request body: JSON with a `hook_event_name` field (Droid/Claude Code naming) or an explicit `event` override.
- Response: `202 Accepted` with `{"status":"queued"}`.
- Body limit: 10 MiB.
- Invalid JSON returns `400`. Unknown `hook_event_name` returns `400`.

### Event mapping table

| Hook event name (`hook_event_name`) | Server-side event type | Destination | karta_core action                                  |
| ----------------------------------- | ---------------------- | ----------- | -------------------------------------------------- |
| `SessionStart`                      | —                      | `/orient`   | `fetch_memories` → orientation context             |
| `UserPromptSubmit`                  | `user_prompt`          | `/capture`  | `add_note_with_clock`                              |
| `PostToolUse`                       | `observation`          | `/capture`  | `add_note_with_clock`                              |
| `Stop`                              | `turn_summary`         | `/capture`  | `add_note_with_clock`                              |
| `SubagentStop`                      | `subagent_result`      | `/capture`  | `add_note_with_clock`                              |
| `SessionEnd`                        | `session_end`          | `/capture`  | marker note + rule-based consolidation             |
| `PreCompact`                        | `pre_compact`          | `/capture`  | marker note + rule-based consolidation (if enabled) |

`session_end` and `pre_compact` run a rule-based consolidator (no LLM, no dream) that promotes high-confidence `Observed` notes to `Fact` notes. `PreCompact` is disabled unless `KARTA_PRECOMPACT=1` is set.

## Operator commands

### status

Prints the current store state without starting the MCP server or HTTP endpoint:

```bash
karta-mcp status
```

Output:

```text
note_count: 42
store_dir: /Users/aj/.karta/store
embedding_model: text-embedding-nomic-embed-text-v1.5
capture_port: 3137
queue_depth: 0
```

`status` opens the store read-only and reads the embedding dimension from the
on-disk `notes_vec` schema, so it works correctly against real stores created
with non-1536-dim embedding models.

### backup

Create a consistent online snapshot while `serve` is running:

```bash
karta-mcp backup --dest /path/to/karta-backup.db
```

The snapshot includes vectors, graph, slot ledger, and the capture queue. It is
safe to run while `serve` is actively ingesting captures.

### export

Export every note to a markdown file under the destination directory:

```bash
karta-mcp export --dest /path/to/export-dir
```

Each file contains the note content, provenance label (`FACT`/`INFERRED`),
confidence, and source back-pointers.

### restore

Replace the current `karta.db` with a backup file. **You must stop `serve`
before restoring**; the command refuses to overwrite a locked store.

```bash
karta-mcp restore --from /path/to/karta-backup.db
```

After restore, restart `serve` to use the restored store.

## Hook templates (Droid and Claude Code)

Example hook configurations are shipped as `.example` files so they are **not**
activated automatically. Copy them into the live config locations and edit as
needed (manual installation). `jq` and `curl` are required for the Droid helper
scripts.

### Droid

From the repository root:

```bash
cp crates/karta-mcp/.factory/hooks.example.json .factory/hooks.json
cp crates/karta-mcp/.factory/hooks/karta-*.sh .factory/hooks/
chmod +x .factory/hooks/karta-*.sh
```

The Droid templates use `command`-type hooks. `SessionStart` calls
`karta-orient.sh` (which POSTs to `/orient`), and the six capture events call
`karta-capture.sh` (which POSTs to `/capture` with the correct server-side event
mapping).

### Claude Code

From the repository root:

```bash
cp crates/karta-mcp/.claude/settings.example.json .claude/settings.json
```

The Claude Code template uses `http`-type hooks. `SessionStart` points to
`/orient`; all other wired events point to `/capture`.

### Port configuration

The Droid helper scripts read `KARTA_CAPTURE_PORT` from the environment and
default to `3137`, so changing the port does not require editing the scripts.
The Claude Code example uses the default port in each URL; replace `3137` with
your configured port if you run karta-mcp on a non-default port.

## Testing

### Automated tests

All automated tests use `MockLlmProvider` and require no external LLM service:

```bash
cargo test -p karta-mcp
cargo clippy -p karta-mcp --no-deps --all-targets -- -D warnings
cargo fmt --check -p karta-mcp
```

### Manual smoke test with `serve --mock`

Start the server on a temporary store and port:

```bash
KARTA_STORE_DIR=/tmp/karta-smoke-store KARTA_CAPTURE_PORT=31501 \
  ./target/debug/karta-mcp serve --mock &
PID=$!
trap 'kill $PID' EXIT
sleep 1
```

Check `/orient`:

```bash
curl -s -X POST http://127.0.0.1:31501/orient \
  -H 'Content-Type: application/json' \
  -d '{"agent":"droid","project":"karta-mcp","cwd":"/tmp"}'
```

Send a capture:

```bash
curl -s -X POST http://127.0.0.1:31501/capture \
  -H 'Content-Type: application/json' \
  -d '{"hook_event_name":"UserPromptSubmit","prompt":"hello karta"}'
```

Poll until the queue drains:

```bash
sqlite3 /tmp/karta-smoke-store/karta.db "SELECT status FROM capture_queue;"
```

Stop the server:

```bash
kill -TERM $PID
```

## Scheduled backups with launchd (macOS)

Save the following as `~/Library/LaunchAgents/ai.karta.backup.plist` to run a
daily backup at 02:00, keeping a week of snapshots:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.karta.backup</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>mkdir -p ~/.karta/backups &amp;&amp; find ~/.karta/backups -name 'karta-*.db' -mtime +7 -delete &amp;&amp; /path/to/karta-mcp backup --dest ~/.karta/backups/karta-$(date +%Y%m%d-%H%M%S).db</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>~/.karta/backups/backup.log</string>
    <key>StandardErrorPath</key>
    <string>~/.karta/backups/backup.log</string>
</dict>
</plist>
```

Load and start the job:

```bash
launchctl load ~/Library/LaunchAgents/ai.karta.backup.plist
launchctl start ai.karta.backup
```

## Scheduled backups with cron (Linux / macOS)

Add a crontab entry that keeps seven days of snapshots:

```cron
0 2 * * * mkdir -p ~/.karta/backups && find ~/.karta/backups -name 'karta-*.db' -mtime +7 -delete && /path/to/karta-mcp backup --dest ~/.karta/backups/karta-$(date +\%Y\%m\%d-\%H\%M\%S).db >> ~/.karta/backups/backup.log 2>&1
```

## Known v1 limitations

- **Serve-down capture loss**: if `karta-mcp serve` is not running when a Droid or Claude Code hook fires, the capture is lost. There is no client-side spool in v1; the hook payload is dropped if the `/capture` or `/orient` endpoint is unreachable.
- **No client-side spool**: the hook scripts call curl with a short timeout and do not retry or persist failed captures locally.
- **Live LLM required for non-mock serve**: `serve` without `--mock` needs a running OpenAI-compatible endpoint (for example LM Studio on `localhost:1234`) and the matching model/embedding environment variables.
- **Dream output correctness**: `karta_run_dreaming` is exercised by tests for shape and reachability, but the semantic quality of generated dreams is intended for manual review.

## License

`karta-mcp` is Copyright (c) 2026 AJ Alon, licensed under the MIT License
(see `LICENSE` in this directory).

`karta-core` and `karta-cli` are upstream code from
[rohithzr/karta](https://github.com/rohithzr/karta), Copyright (c) Rohit Hazra,
licensed under MIT (see the repository root `LICENSE`).

## Fork pin

Built against karta fork HEAD `866a664`.
