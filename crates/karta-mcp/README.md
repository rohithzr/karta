# karta-mcp

MCP server + full-hook auto-capture for Karta.

`karta-mcp serve` runs a single process that exposes:

- a stdio MCP server with the `karta_*` tools,
- a localhost HTTP capture endpoint for Droid/Claude Code lifecycle hooks,
- a durable SQLite-backed capture queue that drains into `karta_core`.

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

## Running

```bash
# Live mode (requires a local LLM endpoint such as LM Studio on localhost:1234)
karta-mcp serve

# Test / validation mode with deterministic mock providers
karta-mcp serve --mock
```

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
activated automatically. copy them into the live config locations and edit as
needed (manual installation).

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

## License

`karta-mcp` is Copyright (c) 2026 AJ Alon, licensed under the MIT License
(see `LICENSE` in this directory).

`karta-core` and `karta-cli` are upstream code from
[rohithzr/karta](https://github.com/rohithzr/karta), Copyright (c) Rohit Hazra,
licensed under MIT (see the repository root `LICENSE`).

## Fork pin

Built against karta fork HEAD `866a664`.
