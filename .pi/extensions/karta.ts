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

const AUTO_CONTEXT_DEFAULT_TOP_K = 5;
const AUTO_CONTEXT_DEFAULT_MAX_CHARS = 4_000;

const HARD_TOKEN_PATTERNS = [
  /`([^`]+)`/g,
  /[A-Za-z0-9_.-]+\/[A-Za-z0-9_./-]+/g,
  /\b[A-Z_][A-Z0-9_]{2,}\b/g,
  /\b[A-Z][A-Za-z0-9_]{2,}\b/g,
  /#[0-9]+\b/g,
  /\b[A-Za-z0-9_.-]+@[0-9][A-Za-z0-9_.-]*\b/g,
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

function envFlag(name: string, defaultValue: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return defaultValue;
  return !["0", "false", "no", "off"].includes(raw.toLowerCase());
}

function envInt(name: string, defaultValue: number, min: number, max: number): number {
  const parsed = Number(process.env[name]);
  if (!Number.isFinite(parsed)) return defaultValue;
  return Math.max(min, Math.min(max, Math.trunc(parsed)));
}

function truncateText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, Math.max(0, maxChars - 24)).trimEnd()}\n…[truncated]`;
}

function extractHardTokens(text: string): string[] {
  const tokens = new Set<string>();

  for (const pattern of HARD_TOKEN_PATTERNS) {
    pattern.lastIndex = 0;
    for (const match of text.matchAll(pattern)) {
      const token = (match[1] ?? match[0]).trim();
      if (token.length >= 2 && token.length <= 160) tokens.add(token);
    }
  }

  return [...tokens].slice(0, 24);
}

function buildAutoContextQuery(prompt: string, cwd: string | undefined): string {
  const hardTokens = extractHardTokens(prompt);
  return [
    prompt,
    cwd ? `cwd:${cwd}` : undefined,
    hardTokens.length ? `exact tokens: ${hardTokens.join(" ")}` : undefined,
  ]
    .filter(Boolean)
    .join("\n");
}

function getStringField(value: unknown, field: string): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const fieldValue = (value as JsonObject)[field];
  return typeof fieldValue === "string" ? fieldValue : undefined;
}

function getNumberField(value: unknown, field: string): number | undefined {
  if (!value || typeof value !== "object") return undefined;
  const fieldValue = (value as JsonObject)[field];
  return typeof fieldValue === "number" && Number.isFinite(fieldValue) ? fieldValue : undefined;
}

function formatAutoContext(result: JsonObject, maxChars: number): string | undefined {
  const hits = Array.isArray(result.results) ? result.results : [];
  const blocks: string[] = [];

  for (const hit of hits) {
    if (!hit || typeof hit !== "object") continue;
    const hitObject = hit as JsonObject;
    const note = hitObject.note;
    if (!note || typeof note !== "object") continue;

    const id = getStringField(note, "id") ?? "unknown";
    const content = getStringField(note, "content");
    if (!content?.trim()) continue;

    const score = getNumberField(hitObject, "score");
    const updatedAt = getStringField(note, "updated_at") ?? getStringField(note, "created_at");
    const provenance = getStringField(note, "provenance");
    const keywords = Array.isArray((note as JsonObject).keywords)
      ? ((note as JsonObject).keywords as unknown[]).filter((item): item is string => typeof item === "string").slice(0, 6)
      : [];

    const metadata = [
      `id=${id}`,
      score === undefined ? undefined : `score=${score.toFixed(3)}`,
      updatedAt ? `updated=${updatedAt}` : undefined,
      provenance ? `provenance=${provenance}` : undefined,
      keywords.length ? `keywords=${keywords.join(", ")}` : undefined,
    ]
      .filter(Boolean)
      .join("; ");

    blocks.push(`- ${metadata}\n  ${truncateText(content.replace(/\s+/g, " ").trim(), 700)}`);
  }

  if (!blocks.length) return undefined;

  return truncateText(
    [
      "Relevant durable Karta memories were retrieved automatically for this turn.",
      "Use them as background context only when relevant; prefer current user instructions and repository state if they conflict.",
      "Do not expose private memory details unless the user asks about memory.",
      "",
      ...blocks,
    ].join("\n"),
    maxChars,
  );
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
    const mode = process.env.KARTA_BIN ? "Karta memory" : "Karta memory (cargo)";
    const auto = envFlag("KARTA_AUTO_CONTEXT", true) ? ", auto-context" : "";
    ctx.ui.setStatus("karta", `${mode}${auto}`);
  });

  pi.on("before_agent_start", async (event, ctx) => {
    if (!envFlag("KARTA_AUTO_CONTEXT", true)) return;

    const topKValue = envInt("KARTA_AUTO_CONTEXT_TOP_K", AUTO_CONTEXT_DEFAULT_TOP_K, 1, 20);
    const maxChars = envInt("KARTA_AUTO_CONTEXT_MAX_CHARS", AUTO_CONTEXT_DEFAULT_MAX_CHARS, 500, 20_000);
    const query = buildAutoContextQuery(event.prompt, event.systemPromptOptions.cwd);

    try {
      const result = await runKarta(["search", "--query", query, "--top-k", String(topKValue)], ctx.signal);
      const content = formatAutoContext(result, maxChars);
      if (!content) return;

      return {
        message: {
          customType: "karta-memory",
          content,
          display: envFlag("KARTA_AUTO_CONTEXT_DISPLAY", false),
        },
      };
    } catch (error) {
      ctx.ui.notify(`Karta auto-context failed: ${(error as Error).message}`, "warning");
    }
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
      turn_index: Type.Optional(Type.Integer({ minimum: 0, description: "Optional source conversation turn index. Requires session_id." })),
      source_timestamp: Type.Optional(Type.String({ description: "Optional RFC3339 source timestamp. Requires session_id." })),
    }),
    async execute(_toolCallId, params, signal) {
      if ((params.turn_index !== undefined || params.source_timestamp) && !params.session_id) {
        throw new Error("session_id is required when turn_index or source_timestamp is provided");
      }
      if (params.turn_index !== undefined && (!Number.isFinite(params.turn_index) || !Number.isInteger(params.turn_index) || params.turn_index < 0)) {
        throw new Error("turn_index must be a non-negative integer");
      }

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
