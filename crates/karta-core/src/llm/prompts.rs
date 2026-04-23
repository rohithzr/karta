/// All LLM prompts used by Karta, centralized for easy tuning.
pub struct Prompts;

impl Prompts {
    pub fn note_attributes_system() -> &'static str {
        r#"You are a memory indexing system. Read one user message and write database rows that will be useful for FUTURE retrieval. Output JSON only, matching the provided schema exactly.

============================================================
SECTION 1 — ADMISSION: SHOULD THIS BECOME MEMORY?
============================================================

Your job is not to summarize the message.
Your job is to write durable retrieval rows.

For each candidate fact, classify it with `memory_kind`:

  Admit (durable kinds):
    durable_fact      — generic claim about the user's world
    future_commitment — deadline, scheduled event, plan with a date
    preference        — user-stated preference
    decision          — chosen path
    constraint        — hard requirement

  Reject (the validator drops these):
    ephemeral_request — "I want help with X" (the request, not a fact)
    speech_act        — "thanks", "ok", "got it"
    echo              — user restating the assistant's prior message

Pure questions, greetings, and help-seeking turns produce ZERO facts.
Returning an empty `atomic_facts: []` is a valid and often correct answer.

  Bad: "The user wants help creating a schedule."
       (memory_kind = ephemeral_request — the validator will drop this anyway,
        so don't bother emitting it. Emit nothing for this turn.)

  Bad: "The user has a project with a Time Anchor of March 15, 2024."
       (benchmark jargon, vague entity, no concrete value)

  Good: { entity_text: "budget tracker", facet: "deadline",
          value_date: "2024-03-15T00:00:00Z", memory_kind: "future_commitment" }

WRAPPER-STRIP RULE — "I want help with X" is two things: the request
(ephemeral, drop it) AND any concrete claim embedded in X (durable, keep).
Strip the request framing, then ask: does the remainder name a typed entity
+ facet + value? If yes, emit the embedded fact.

  Source: "I want help configuring Chart.js 4.3.0 for tooltips and 60fps animation."
  Drop:   the help-request itself.
  Keep:   { entity_type: project, facet: tech_stack,
            value_text: "Chart.js 4.3.0", memory_kind: durable_fact }

PAST EVENTS WITH CONCRETE DATES are durable facts — emit them with
`memory_kind=durable_fact` and `facet=event`. Do NOT classify them as
ephemeral just because they describe completed work.

  Source: "Sprint 1 ended on March 29."
  Keep:   { entity_text: "Sprint 1", facet: event, memory_kind: durable_fact,
            occurred_start: "2024-03-29T00:00:00Z", occurred_confidence: 0.8,
            supporting_spans: ["Sprint 1 ended on March 29", "March 29"] }

============================================================
SECTION 2 — ATOMIZATION
============================================================

Each fact is one entity + one facet + one value.

Prefer one precise fact over several paraphrases. If two candidates share the
same entity, facet, and value, keep only the most direct one — the validator
will dedup on `(entity_text, facet, value_*)` anyway.

Use ordinary-world language. Do NOT store benchmark or conversation jargon
("time anchor", "assistant", "memory"). Replace with the underlying claim
("the project's target date is...").

============================================================
SECTION 3 — GROUNDING (supporting_spans)
============================================================

EVERY fact must include 1-3 `supporting_spans`. Each span is a verbatim
substring of the source MESSAGE (NOT the fact text you wrote). The validator
will reject your fact if any span is not a real substring of the message.

  Source: "I have an April 15 deadline for the budget tracker."
  Fact:   { content: "The budget tracker has an April 15 deadline.",
            supporting_spans: ["April 15 deadline", "budget tracker"] }

Do not summarize, do not paraphrase, copy substrings. Each span ≥4 characters.

TEMPORAL GROUNDING — if the fact has any `occurred_*` populated, ONE of
your supporting_spans MUST contain the literal temporal phrase
("yesterday", "last week", "March 29", "2024-03-15"). The validator
strips occurred_* otherwise — the bounds become null and the fact loses
its temporal anchor.

  Source: "Yesterday I closed the auth ticket."
  Fact:   { facet: event, occurred_start: <ref-1d>, occurred_confidence: 0.7,
            supporting_spans: ["Yesterday I closed the auth ticket", "auth ticket"] }
                                ^^^^^^^^^ MUST appear in some span

============================================================
SECTION 4 — NORMALIZATION (entity / facet / value slots)
============================================================

  entity_type: user | project | person | org | task | unknown
    Coarse type of the thing the fact is about. Use `unknown` only when no
    typed entity exists in the message.

  entity_text: surface form ("budget tracker", "Coco", "v1") or null.
    Prefer the most specific name available in the message. Avoid generic
    nouns like "project" or "user" if a more specific name appears.

  facet: deadline | target_date | preference | tech_stack | location |
         ownership | constraint | event | unknown
    What aspect of the entity this fact describes.

  value_text: string-shaped value ("Flask 2.3.1", "vegetarian") or null.
  value_date: date-shaped value (deadline, target date) — RFC3339 UTC. null otherwise.

The validator REJECTS facts where BOTH `entity_type = unknown` AND
`facet = unknown`. One generic dimension is fine; both means the fact
carries no information.

============================================================
SECTION 5 — TEMPORAL SLOTS
============================================================

Two distinct slot families. Both have STRUCTURAL validators —
ungrounded or missing values get stripped, so populating them
correctly is the only way to surface temporal info downstream.

---- A) value_date — when the FACT IS A DATE ----

REQUIRED whenever facet ∈ { deadline, target_date }. The validator
STRIPS the fact entirely if facet is date-shaped and value_date is null.

Extract value_date from the source message's date phrase. If the
year is implicit ("by April 19" with no year), use the year from
reference_time.

  "deadline of 2024-04-15"          → facet=deadline,    value_date=2024-04-15
  "deadline of March 15, 2024"      → facet=deadline,    value_date=2024-03-15
  "Sprint 2 ... by April 19"        → facet=target_date, value_date=<ref-year>-04-19
  "targeting March 15 for v1"       → facet=target_date, value_date=2024-03-15

If you classify a fact as facet=deadline / target_date but cannot
extract a concrete date, you should not be using that facet — pick
a different facet (event, constraint, preference) instead.

---- B) occurred_* — when the FACT REFERENCES PAST/FUTURE EVENT TIME ----

Default for non-event facts: all three null + occurred_confidence=0.0.

REQUIRED grounding: at least one supporting_span MUST contain the
temporal phrase (literal date OR relative phrase like "yesterday",
"last week", "ago"). The validator STRIPS occurred_* when no span
carries a temporal marker — inferred bounds from sentence vibes get
dropped.

Band rules:
  1.0 — explicit ISO date in the source span ("2024-03-15")
  0.8 — NL absolute date in the source span ("March 15", "March 29")
  0.7 — relative reference in the source span ("yesterday", "last week")
        — resolve against reference_time
  0.5 — vague temporal word ("recently", "around March")

Examples:
  "Sprint 1 ended on March 29"
    → facet=event, occurred_start=2024-03-29, conf=0.8,
      supporting_spans includes "March 29"

  "yesterday I closed the auth ticket" (ref_time=2024-04-22)
    → facet=event, occurred_start=2024-04-21, conf=0.7,
      supporting_spans includes "yesterday"

The reference_time in the user message preamble is for resolving
relative phrases. It is NOT a default timestamp — do not stamp
present-tense facts with `[ref_time-1d, ref_time)`.

============================================================
ANTI-PATTERNS (real failure modes from production traces)
============================================================

A) Conversation-date contamination
   Don't put bounds on "Project uses Flask" just because the conversation is dated.

B) Bleeding one fact's date into siblings
   "Meet April 15 deadline. MVP includes login." → only the deadline fact gets
   `value_date`. The MVP-content fact gets null.

C) Inferring "yesterday" from present-tense verbs
   0.7 occurred_confidence requires the literal word "yesterday" / "today" /
   "last week" in the fact text.

D) Vague-by-default
   When uncertain about temporal, emit nulls. 0.5 is for explicit vague
   temporal language only.

E) Generic entity + generic facet
   `entity_type = unknown` AND `facet = unknown` together is a noise fact.
   Either type the entity, type the facet, or skip the fact.

F) Speech-act extraction
   The help-request wrapper is ephemeral and gets dropped. But X may itself
   contain a durable claim — extract that, drop the wrapper. See the
   WRAPPER-STRIP RULE in Section 1.

G) Jargon leakage
   "Time anchor", "assistant", "memory" are conversation/benchmark jargon.
   Translate to the underlying claim or drop the fact. The "[March-15-2024]"
   prefix on a benchmark message is jargon — do not bleed it into per-fact
   value_date or occurred_* unless the fact text itself contains the date.

H) Date-shaped facet without value_date
   facet=deadline / target_date with value_date=null is the v1-STEP2 failure
   mode. The validator strips these facts entirely. Either extract the date
   from the source phrase or pick a different facet.

I) Inferred occurred_* without span grounding
   occurred_confidence > 0 with no temporal phrase in supporting_spans is
   the v1-STEP2 failure mode for past events. The validator strips bounds
   that aren't cited. Quote the date / "yesterday" / "ago" phrase.

============================================================
OTHER FIELDS (note-level, not per-fact)
============================================================

- context: 1-2 sentences capturing implications the message does not literally
           state. Do not restate the input.
- keywords: 5-8 specific search terms.
- tags: 3-5 from this CLOSED SET — do not invent others:
    {preference, decision, constraint, workflow, entity, pattern, temporal,
     code, deadline, planning}
- foresight_signals: forward-looking statements with explicit expiry dates.

============================================================
PROPERTIES OF GOOD OUTPUT
============================================================

The validator will mechanically drop facts that don't have these
properties. Don't try to self-validate; just produce facts that look
like this:

  - memory_kind is one of the durable five
    (durable_fact, future_commitment, preference, decision, constraint).
    Ephemeral kinds get dropped, so emitting them wastes a slot.

  - 1-3 supporting_spans, each a verbatim substring of the source
    message, each ≥4 characters.

  - At least one of (entity_type, facet) is typed. Both `unknown` is
    the noise-fact shape that gets dropped.

  - Date-shaped facets (deadline, target_date) populate `value_date`.
    String-shaped facets (tech_stack, preference) populate `value_text`.

  - One slot per claim — no two facts in the same response with the
    same (entity_text, facet, value). Pick the most direct phrasing.

Returning `atomic_facts: []` when nothing meets these properties is
the correct output for many turns. Do not pad with low-quality facts
to fill the array.
"#
    }

    pub fn note_attributes_user(content: &str, reference_time: chrono::DateTime<chrono::Utc>) -> String {
        format!(
            "reference_time: {ref_time}\n\nMessage:\n{content}",
            ref_time = reference_time.to_rfc3339(),
            content = content,
        )
    }

    pub fn linking_system() -> &'static str {
        "You decide which existing memories should be linked to a new memory.\n\
         Link only when there is a MEANINGFUL relationship: same entity, causal connection, \
         shared constraint, complementary context, or directly related decision.\n\
         Do NOT link just because topics overlap loosely.\n\
         Respond with JSON: { \"links\": [{ \"noteId\": \"...\", \"reason\": \"one sentence why\" }] }"
    }

    pub fn linking_user(new_content: &str, new_context: &str, candidates: &str) -> String {
        format!(
            "New memory:\nContent: {}\nContext: {}\n\nCandidates:\n{}",
            new_content, new_context, candidates
        )
    }

    pub fn evolve_system() -> &'static str {
        "A new related memory has been added. Update the existing memory's context to reflect \
         what this new connection reveals.\n\
         The updated context should be richer — capturing implications that only become clear \
         when both pieces of information are known together.\n\
         Keep it to 1-2 sentences.\n\
         Respond with JSON: { \"updatedContext\": \"...\" }"
    }

    pub fn evolve_user(
        existing_content: &str,
        existing_context: &str,
        new_content: &str,
        link_reason: &str,
    ) -> String {
        format!(
            "Existing memory: {}\nCurrent context: {}\n\nNew related memory: {}\nLink reason: {}",
            existing_content, existing_context, new_content, link_reason
        )
    }

    pub fn synthesize_system() -> &'static str {
        "You answer questions using ONLY the provided memory notes. \
         Be thorough — include all relevant issues, risks, constraints, and technical details \
         from the notes that are relevant to the query. \
         Cite which notes informed each part of your answer.\n\n\
         CRITICAL RULES:\n\n\
         1. COMPLETENESS: Answer the question using ONLY the provided notes. \
         If the notes contain partial information, provide what you can and indicate what is missing. \
         Do NOT fabricate, guess, or infer answers from general knowledge. \
         Do NOT answer a different question than what was asked just because you have related notes. \
         If the notes are about a completely different topic than the question, say so naturally in your answer.\n\n\
         2. CONTRADICTIONS: If notes contain contradictory information, explicitly flag the contradiction. \
         State both sides and note which is more recent or authoritative. \
         Do NOT silently resolve contradictions — surface them. \
         Notes marked [CONTRADICTION SOURCE] are both sides of a detected contradiction. \
         Present BOTH sides explicitly with specific quotes and ask which is correct. \
         Do NOT silently choose one side.\n\n\
         3. TEMPORAL REASONING: Pay close attention to dates and time references. \
         If the context begins with an \"EVENTS IN THIS CONVERSATION\" block, \
         treat its ISO dates as authoritative — they were extracted by a dream-time pass \
         over the full conversation and are more reliable than dates you may infer from \
         scattered note text. When computing durations or ordering events, show your work: \
         state the specific dates from the EVENTS block (or from note content if no block \
         is present), then calculate. If an EVENTS block is present and the two events you \
         need are BOTH listed, you MUST compute the answer rather than saying the data is \
         missing.\n\n\
         4. PROVENANCE: Each note is tagged with provenance and age:\n\
         - FACT = directly observed information\n\
         - INFERRED:{type} = derived by reasoning (deduction, induction, abduction, etc.)\n\
         - FACT:from-{id} = atomic fact extracted from a note; highly precise for specific values\n\
         - DIGEST:{id} = episode summary with pre-computed counts and entity tracking; treat aggregation counts as authoritative\n\
         Treat FACT notes as authoritative. Treat INFERRED notes as supporting evidence \
         but flag them as inferences when they are central to your answer. \
         When FACT and INFERRED notes conflict, prioritize the FACT. \
         DIGEST notes contain pre-computed aggregations — use their counts rather than re-counting from individual notes.\n\n\
         5. SUPERSESSION: When multiple notes report different values for the SAME fact \
         (e.g., a metric, count, status, or preference), the MOST RECENT note wins — period. \
         State the latest value as the answer. Do NOT present old and new values as a \
         \"contradiction\" — a value changing over time is an UPDATE, not a conflict. \
         Only flag a contradiction when two notes from the SAME time period assert \
         incompatible claims about the same fact.\n\n\
         6. USER PREFERENCES AND INSTRUCTIONS: If any retrieved notes contain explicit user \
         instructions (\"always do X\", \"I prefer Y\", \"never Z\") or stated preferences, \
         you MUST apply them to your answer. User instructions override default formatting, \
         scope, or presentation choices. If a preference conflicts with another preference, \
         apply the more recent one (see rule 5).\n\n\
         7. EVENT ENUMERATION: When the user asks about ORDER, SEQUENCE, TIMELINE, or STEPS, \
         list SPECIFIC events with concrete details (names, dates, actions, outcomes), not \
         broad themes or categories. Each list item should describe ONE specific thing that \
         happened, not a topic area. Scan ALL provided notes — do not stop after the first \
         few. If 20 notes are provided, your answer should reflect content from most of them, \
         not just the top 3.\n\n\
         8. FORMAT: Match the format the user expects. \
         If the user asks for code, examples, or implementation details, and the notes contain \
         code snippets, function names, API calls, or technical patterns, include them as \
         properly formatted code blocks in your answer. \
         If the user asks for a list or timeline, format as a list or timeline. \
         The memory notes are the source — present them in the format that best answers the question."
    }

    pub fn synthesize_user(query: &str, notes_text: &str) -> String {
        format!("Query: {}\n\nRelevant memories:\n{}", query, notes_text)
    }

    // --- Dream prompts ---

    pub fn dream_deduction(notes_text: &str) -> String {
        format!(
            "You are a reasoning agent performing deductive analysis on a set of linked memory notes.\n\n\
             Your job:\n\
             1. Show your reasoning step by step (chain-of-thought).\n\
             2. Derive a conclusion that is LOGICALLY NECESSARY from the facts — not just plausible, but entailed.\n\
             3. Only produce a conclusion if you can actually derive it. If nothing can be deduced, say so.\n\n\
             Notes:\n{}\n\n\
             Respond with JSON:\n\
             {{\n\
               \"reasoning\": \"step-by-step chain of thought\",\n\
               \"conclusion\": \"the deduced fact, or null if nothing can be deduced\",\n\
               \"confidence\": 0.0\n\
             }}",
            notes_text
        )
    }

    pub fn dream_induction(notes_text: &str) -> String {
        format!(
            "You are a reasoning agent performing inductive analysis across a set of memory notes.\n\n\
             Your job:\n\
             1. Look for REPEATED patterns, recurring themes, or structural similarities across multiple notes.\n\
             2. Generalise them into a principle or rule that applies beyond any single note.\n\
             3. Show your reasoning. If no meaningful pattern exists, say so.\n\n\
             Notes:\n{}\n\n\
             Respond with JSON:\n\
             {{\n\
               \"reasoning\": \"what patterns you observed and why they support a generalisation\",\n\
               \"generalisation\": \"the induced principle or rule, or null if no pattern found\",\n\
               \"confidence\": 0.0,\n\
               \"supportingNoteCount\": 0\n\
             }}",
            notes_text
        )
    }

    pub fn dream_abduction(notes_text: &str) -> String {
        format!(
            "You are a reasoning agent looking for gaps in a set of memory notes.\n\n\
             Your job:\n\
             1. Examine what is known.\n\
             2. Identify what context, information, or facts are CONSPICUOUSLY ABSENT — \
             things that would normally be known if you had the full picture.\n\
             3. Hypothesise the most plausible explanation for the gap.\n\
             4. Be explicit: this is a hypothesis, not a fact.\n\n\
             Notes:\n{}\n\n\
             Respond with JSON:\n\
             {{\n\
               \"reasoning\": \"what gaps you noticed and why they are suspicious\",\n\
               \"hypothesis\": \"the abductive hypothesis about missing context, or null if no meaningful gap\",\n\
               \"confidence\": 0.0\n\
             }}",
            notes_text
        )
    }

    pub fn dream_consolidation(notes_text: &str) -> String {
        format!(
            "You are creating a \"peer card\" — a concise structured summary of everything known \
             about an entity or topic from a cluster of linked memory notes.\n\n\
             The peer card should capture:\n\
             - Who/what the entity is\n\
             - Key facts, constraints, and decisions\n\
             - Ongoing context and stakes\n\
             - Any risks or open questions\n\n\
             Notes (all about the same entity or project):\n{}\n\n\
             Respond with JSON:\n\
             {{\n\
               \"reasoning\": \"why these notes cluster around the same entity or topic\",\n\
               \"entityId\": \"the primary entity or topic name (e.g. person name, company name, project name)\",\n\
               \"peerCard\": \"2-4 sentence summary capturing the complete picture\",\n\
               \"confidence\": 0.0\n\
             }}",
            notes_text
        )
    }

    pub fn profile_merge(existing_profile: &str, new_peer_card: &str) -> String {
        format!(
            "You are updating an entity profile. Merge the new information into the existing profile.\n\n\
             Rules:\n\
             - Keep all still-valid facts from the existing profile\n\
             - Add new facts from the peer card\n\
             - When facts conflict, prefer the newer information (from the peer card)\n\
             - Keep the result to 2-4 sentences maximum\n\n\
             Existing profile:\n{}\n\n\
             New information:\n{}\n\n\
             Respond with JSON:\n\
             {{\n\
               \"updatedProfile\": \"the merged 2-4 sentence profile\"\n\
             }}",
            existing_profile, new_peer_card
        )
    }

    // --- Episode prompts ---

    pub fn episode_boundary_system() -> &'static str {
        "You decide whether a new message belongs to the same conversational episode \
         or starts a new one. An episode is a thematically coherent segment of conversation.\n\n\
         Consider: topic shift, entity change, temporal gap, intent change.\n\
         Respond with JSON: { \"sameEpisode\": true/false, \"reason\": \"brief explanation\" }"
    }

    pub fn episode_boundary_user(
        previous_content: &str,
        new_content: &str,
        time_gap_secs: i64,
    ) -> String {
        let gap_desc = if time_gap_secs < 60 {
            "seconds ago".to_string()
        } else if time_gap_secs < 3600 {
            format!("{} minutes ago", time_gap_secs / 60)
        } else if time_gap_secs < 86400 {
            format!("{} hours ago", time_gap_secs / 3600)
        } else {
            format!("{} days ago", time_gap_secs / 86400)
        };

        format!(
            "Previous message (from {}):\n{}\n\nNew message:\n{}",
            gap_desc, previous_content, new_content
        )
    }

    pub fn episode_narrative_system() -> &'static str {
        "Synthesize a concise 2-3 sentence narrative summary of this conversational episode.\n\
         Capture the key topic, decisions, and outcomes — not a transcript.\n\
         Also provide a chronological list of the main topics or events discussed, \
         in the exact order they appear in the notes. This ordering is critical.\n\
         Respond with JSON: { \"narrative\": \"...\", \"topicTags\": [\"2-4 topic labels\"], \
         \"topicOrder\": [\"first topic\", \"second topic\", \"third topic\"] }"
    }

    pub fn episode_narrative_user(notes_text: &str) -> String {
        format!("Episode notes:\n{}", notes_text)
    }

    // --- Episode Digest prompts (Phase Next) ---

    pub fn episode_digest(notes_text: &str) -> String {
        format!(
            "Analyze this episode's notes and produce a structured digest.\n\n\
             Extract:\n\
             1. entities: every named entity (person, tool, framework, project, date, number) \
                with type and mention count. If an entity's value was updated during the episode, \
                record the LATEST value.\n\
             2. date_range: earliest and latest dates mentioned IN THE CONTENT (not timestamps).\n\
             3. aggregations: countable collections (e.g., '5 movies discussed: [list]').\n\
             4. topic_sequence: topics in the ORDER they appeared.\n\
             5. timed_events: EVERY specific action, meeting, milestone, deadline, decision, \
                commitment, plan, or state change mentioned in the notes. BE VERY GENEROUS — \
                any concrete past/future happening counts, even planned or hypothetical ones. \
                If the episode has more than zero notes, it should have more than zero events. \
                Each event has:\n\
                - description: a concrete action phrased as a short clause \
                  (e.g. \"finished transaction management features\", \"met with Wyatt expressing skepticism\", \
                  \"planned deployment for Friday\")\n\
                - date: ISO YYYY-MM-DD. Determine it in this order:\n\
                  (a) If the note content contains an explicit date for the event (e.g. \"on March 10\", \
                      \"April 15 deadline\", \"2024-03-29\"), USE THAT DATE.\n\
                  (b) Otherwise, if the source note begins with a bracketed date prefix like \
                      [March-15-2024] or [2024-03-15] (this is the conversation timestamp of that message), \
                      USE THE NOTE'S PREFIX DATE. This is the most common case — ALMOST EVERY NOTE \
                      IN THIS DATASET HAS A [DATE] PREFIX, so almost every event should end up with a date.\n\
                  (c) Only set date to null if neither (a) nor (b) applies.\n\
                - source_turn: the [N] index of the note the event came from, if known, else null\n\
                IMPORTANT: downstream code uses this to answer \"how many days between X and Y\" \
                questions. Prefer MANY dated events over a few vague undated ones. Include the same event \
                only once even if it is mentioned across several notes.\n\
             6. digest_text: A 2-4 sentence summary that embeds well for retrieval. \
                Include specific names, numbers, and dates.\n\n\
             Notes:\n{}\n\n\
             Respond with JSON:\n\
             {{\n\
               \"entities\": [{{\"name\": \"...\", \"type\": \"person|tool|framework|project|date|number|other\", \
                 \"count\": 1, \"latest_value\": \"...or null\"}}],\n\
               \"date_range\": {{\"earliest\": \"YYYY-MM-DD\", \"latest\": \"YYYY-MM-DD\"}} or null,\n\
               \"aggregations\": [{{\"label\": \"movies discussed\", \"count\": 5, \"items\": [\"...\"]}}],\n\
               \"topic_sequence\": [\"first topic\", \"second topic\"],\n\
               \"timed_events\": [{{\"description\": \"finished transaction management features\", \
                 \"date\": \"2024-03-15\", \"source_turn\": 12}}],\n\
               \"digest_text\": \"retrieval-optimized summary\",\n\
               \"confidence\": 0.8\n\
             }}",
            notes_text
        )
    }

    pub fn cross_episode_digest(episode_digests_text: &str) -> String {
        format!(
            "You are analyzing digests from multiple episodes to find cross-episode patterns.\n\n\
             For each entity that appears in 2+ episodes, track how its value changed over time.\n\
             Identify aggregations that span episodes (total unique items across all episodes).\n\
             Merge all timed events into one deduped chronological list across episodes.\n\
             Find the overall topic progression across episodes.\n\n\
             Episode digests:\n{}\n\n\
             Respond with JSON:\n\
             {{\n\
               \"entity_timeline\": [{{\"name\": \"...\", \"type\": \"...\", \
                 \"changes\": [{{\"episode_id\": \"...\", \"value\": \"...\"}}]}}],\n\
               \"cross_aggregations\": [{{\"label\": \"...\", \"count\": 0, \"items\": [\"...\"]}}],\n\
               \"timed_events\": [{{\"description\": \"...\", \"date\": \"YYYY-MM-DD or null\", \
                 \"source_turn\": null}}],\n\
               \"topic_progression\": [\"...\"],\n\
               \"digest_text\": \"cross-episode summary optimized for retrieval\",\n\
               \"confidence\": 0.7\n\
             }}",
            episode_digests_text
        )
    }

    pub fn dream_contradiction(notes_text: &str) -> String {
        format!(
            "You are a consistency checker for a set of linked memory notes.\n\n\
             Your job:\n\
             1. Look for facts that CONTRADICT or TENSION with each other.\n\
             2. Contradictions: things that cannot both be true simultaneously.\n\
             3. Tensions: not logically impossible but in conflict.\n\
             4. If you find one, describe it precisely.\n\n\
             Notes:\n{}\n\n\
             Respond with JSON:\n\
             {{\n\
               \"reasoning\": \"which notes conflict and exactly how\",\n\
               \"contradiction\": \"precise description of the conflict, or null if notes are consistent\",\n\
               \"severity\": \"critical | tension | none\",\n\
               \"confidence\": 0.0\n\
             }}",
            notes_text
        )
    }
}
