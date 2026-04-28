import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  type ExtensionAPI,
  truncateHead,
} from "@mariozechner/pi-coding-agent";
import { Type } from "typebox";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const KARTA_USAGE_GUIDELINES = [
  "Use Karta memory only for durable project facts, user preferences, architecture decisions, constraints, bug root causes, and important findings.",
  "Do not store secrets, raw logs, transient scratch work, large code blocks, speculative guesses, or every assistant response in Karta.",
  "Search Karta before answering questions that may depend on prior project context or decisions.",
];

type JsonObject = Record<string, unknown>;
type ExecFileFailure = Error & { stdout?: string; stderr?: string; code?: unknown; signal?: unknown };

function timeoutMs(): number {
  const raw = process.env.KARTA_TIMEOUT_MS;
  if (!raw) return 120_000;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 120_000;
}

function topK(value: unknown): string {
  const parsed = Number(value ?? 5);
  const clamped = Math.max(1, Math.min(100, Number.isFinite(parsed) ? Math.trunc(parsed) : 5));
  return String(clamped);
}

async function runKarta(args: string[], signal?: AbortSignal): Promise<JsonObject> {
  const fullArgs = ["--json", ...args];
  const options = {
    env: process.env,
    maxBuffer: 10 * 1024 * 1024,
    timeout: timeoutMs(),
    signal,
  };

  try {
    // If KARTA_BIN is set, use that binary. Otherwise, default to the workspace
    // crate so the extension works immediately from a Karta checkout.
    const command = process.env.KARTA_BIN;
    const result = command
      ? await execFileAsync(command, fullArgs, options)
      : await execFileAsync("cargo", ["run", "-q", "-p", "karta-cli", "--", ...fullArgs], options);

    return parseKartaJson(result.stdout, "stdout");
  } catch (error) {
    throw normalizeKartaError(error);
  }
}

function parseKartaJson(output: string | undefined, streamName: string): JsonObject {
  const text = output?.trim();
  if (!text) throw new Error(`karta returned empty ${streamName}`);

  try {
    return JSON.parse(text) as JsonObject;
  } catch (error) {
    throw new Error(`failed to parse karta JSON from ${streamName}: ${(error as Error).message}\n${streamName}:\n${text}`);
  }
}

function normalizeKartaError(error: unknown): Error {
  const failure = error as ExecFileFailure;

  if (failure.name === "AbortError") {
    return new Error("karta command cancelled");
  }

  for (const [streamName, output] of [
    ["stderr", failure.stderr],
    ["stdout", failure.stdout],
  ] as const) {
    const text = output?.trim();
    if (!text) continue;

    try {
      const payload = JSON.parse(text) as JsonObject;
      if (payload && payload.ok === false && typeof payload.error === "string") {
        return new Error(payload.error);
      }
      return new Error(`karta failed with JSON payload on ${streamName}: ${JSON.stringify(payload)}`);
    } catch {
      // Not JSON; fall through to stderr/stdout text below.
    }
  }

  const stderr = failure.stderr?.trim();
  if (stderr) return new Error(stderr);

  const stdout = failure.stdout?.trim();
  if (stdout) return new Error(stdout);

  return failure instanceof Error ? failure : new Error(String(error));
}

function toolResult(result: JsonObject) {
  const fullText = JSON.stringify(result, null, 2);
  const truncated = truncateHead(fullText, {
    maxBytes: DEFAULT_MAX_BYTES,
    maxLines: DEFAULT_MAX_LINES,
  });
  const text = truncated.truncated
    ? `${truncated.content}\n\n[Output truncated: showing ${truncated.outputLines}/${truncated.totalLines} lines, ${truncated.outputBytes}/${truncated.totalBytes} bytes. Full JSON is available in tool details.]`
    : truncated.content;

  return {
    content: [{ type: "text" as const, text }],
    details: {
      result,
      outputTruncated: truncated.truncated,
      truncation: truncated,
    },
  };
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.setStatus("karta", process.env.KARTA_BIN ? "Karta memory" : "Karta memory (cargo)");
  });

  pi.registerTool({
    name: "karta_add_note",
    label: "Karta: Add Note",
    description:
      "Store a durable memory note in Karta. Use for stable project facts, user preferences, architecture decisions, constraints, bug root causes, and important findings. Do not store secrets, raw logs, transient scratch work, or large code blocks.",
    promptSnippet: "Store durable project memory in Karta.",
    promptGuidelines: KARTA_USAGE_GUIDELINES,
    parameters: Type.Object({
      content: Type.String({ description: "The durable memory content to store." }),
      session_id: Type.Optional(Type.String({ description: "Optional session/workspace grouping ID." })),
      turn_index: Type.Optional(Type.Number({ description: "Optional source conversation turn index. Requires session_id." })),
      source_timestamp: Type.Optional(Type.String({ description: "Optional RFC3339 source timestamp. Requires session_id." })),
    }),
    async execute(_toolCallId, params, signal) {
      const args = ["add-note", "--content", params.content];
      if (params.session_id) args.push("--session-id", params.session_id);
      if (params.turn_index !== undefined) args.push("--turn-index", String(params.turn_index));
      if (params.source_timestamp) args.push("--source-timestamp", params.source_timestamp);
      return toolResult(await runKarta(args, signal));
    },
  });

  pi.registerTool({
    name: "karta_search",
    label: "Karta: Search",
    description:
      "Search Karta memories semantically. Use before answering questions that may depend on prior project decisions, maintainer preferences, recurring bugs, or architectural context.",
    promptSnippet: "Search Karta durable project memory.",
    promptGuidelines: KARTA_USAGE_GUIDELINES,
    parameters: Type.Object({
      query: Type.String({ description: "Search query." }),
      top_k: Type.Optional(Type.Number({ description: "Number of memories to return, 1-100. Defaults to 5." })),
    }),
    async execute(_toolCallId, params, signal) {
      return toolResult(await runKarta(["search", "--query", params.query, "--top-k", topK(params.top_k)], signal));
    },
  });

  pi.registerTool({
    name: "karta_ask",
    label: "Karta: Ask",
    description: "Ask Karta a question against stored memories and get a synthesized answer with retrieval metadata.",
    promptSnippet: "Ask Karta for a synthesized answer from durable memory.",
    promptGuidelines: KARTA_USAGE_GUIDELINES,
    parameters: Type.Object({
      query: Type.String({ description: "Question to ask Karta." }),
      top_k: Type.Optional(Type.Number({ description: "Number of context notes to consider, 1-100. Defaults to 5." })),
    }),
    async execute(_toolCallId, params, signal) {
      return toolResult(await runKarta(["ask", "--query", params.query, "--top-k", topK(params.top_k)], signal));
    },
  });

  pi.registerTool({
    name: "karta_get_note",
    label: "Karta: Get Note",
    description: "Retrieve a specific Karta memory note by ID.",
    promptSnippet: "Retrieve a specific Karta memory note by ID.",
    parameters: Type.Object({
      id: Type.String({ description: "Note ID." }),
    }),
    async execute(_toolCallId, params, signal) {
      return toolResult(await runKarta(["get-note", "--id", params.id], signal));
    },
  });

  pi.registerTool({
    name: "karta_note_count",
    label: "Karta: Note Count",
    description: "Get the total count of stored Karta memory notes.",
    promptSnippet: "Count stored Karta memory notes.",
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, signal) {
      return toolResult(await runKarta(["note-count"], signal));
    },
  });

  pi.registerTool({
    name: "karta_health",
    label: "Karta: Health",
    description: "Check Karta embedded store health and migration status.",
    promptSnippet: "Check Karta memory store health.",
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, signal) {
      return toolResult(await runKarta(["health"], signal));
    },
  });

  pi.registerTool({
    name: "karta_dream",
    label: "Karta: Dream",
    description:
      "Run Karta background reasoning over the memory graph. Produces inferred notes via deduction, induction, abduction, consolidation, contradiction detection, and episode digests.",
    promptSnippet: "Run Karta background reasoning over the memory graph.",
    promptGuidelines: KARTA_USAGE_GUIDELINES,
    parameters: Type.Object({
      scope_type: Type.Optional(Type.String({ description: "Dream scope type. Defaults to workspace." })),
      scope_id: Type.Optional(Type.String({ description: "Dream scope identifier. Defaults to default." })),
    }),
    async execute(_toolCallId, params, signal) {
      return toolResult(
        await runKarta(
          ["dream", "--scope-type", params.scope_type ?? "workspace", "--scope-id", params.scope_id ?? "default"],
          signal,
        ),
      );
    },
  });

  pi.registerCommand("karta-health", {
    description: "Check Karta CLI/store health",
    handler: async (_args, ctx) => {
      try {
        const result = await runKarta(["health"]);
        ctx.ui.notify(`Karta health: ${JSON.stringify(result)}`, "success");
      } catch (error) {
        ctx.ui.notify(`Karta health failed: ${(error as Error).message}`, "error");
      }
    },
  });
}
