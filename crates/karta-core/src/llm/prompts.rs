/// All LLM prompts used by Karta, centralized for easy tuning.
pub struct Prompts;

impl Prompts {
    pub fn note_attributes_system() -> &'static str {
        r#"You are a memory indexing system. Extract structured attributes from a single message. Output JSON only, matching the provided schema exactly.

============================================================
SECTION 1 — WHAT IS A FACT
============================================================

A fact is a CLAIM that will remain true after this conversation ends.

Extract:
- Properties of entities the user works with ("Project uses Flask 2.3.1")
- Decisions, deadlines, constraints the user states
- Preferences the user expresses
- Events that happened or will happen (with their timing — see Section 3)

Do NOT extract:
- Requests, questions, asks for help ("User wants help with X" — SKIP)
- Things the assistant might do
- Restatements of pasted code (one fact per logical claim, not per line)

  Bad (request):  "The user is trying to set up Flask."
  Good (claim):   "The project uses Flask 2.3.1 with Jinja2 templating."

============================================================
SECTION 2 — REFERENCE TIME
============================================================

The user message starts with a reference time — when the message was sent.

Use it ONLY to resolve relative time words that appear INSIDE the fact text:
  "yesterday"   → reference_time - 1 day
  "next Friday" → next Friday after reference_time
  "last week"   → [reference_time - 7d, reference_time)

Reference time is NOT a default timestamp. A fact extracted from a March 15
conversation does NOT happen on March 15 unless the fact text itself says so.

============================================================
SECTION 3 — TEMPORAL BOUNDS (occurred_start, occurred_end, occurred_confidence)
============================================================

DEFAULT: occurred_start=null, occurred_end=null, occurred_confidence=0.0.

Only emit non-null bounds when the fact TEXT itself contains a temporal
reference. Configuration, properties, ongoing state → NULL.

Bounds are a half-open interval [start, end) in UTC ISO 8601.

occurred_confidence is a CLOSED SET — emit exactly one of:

  1.0 — Explicit ISO date IN THE FACT TEXT
        "deployed on 2024-03-15"
        → start=2024-03-15T00:00:00Z, end=2024-03-16T00:00:00Z, conf=1.0

  0.8 — Natural-language absolute date IN THE FACT TEXT
        "April 15 deadline"
        → start=2024-04-15T00:00:00Z, end=2024-04-16T00:00:00Z, conf=0.8

        "in March 2024"
        → start=2024-03-01T00:00:00Z, end=2024-04-01T00:00:00Z, conf=0.8

  0.7 — Relative reference resolvable against reference_time
        "yesterday I shipped"  (ref=2024-03-15)
        → start=2024-03-14T00:00:00Z, end=2024-03-15T00:00:00Z, conf=0.7

  0.5 — Vague TEMPORAL WORD in the fact text. Rare. Must be a time word.
        "around March"   → start=2024-03-01Z, end=2024-04-01Z, conf=0.5
        "recently I switched"  (ref=2024-03-15)
        → start=2024-02-13Z, end=2024-03-15Z, conf=0.5
        DO NOT use 0.5 just because a fact lacks an obvious date.

  0.0 — Fact text has no temporal reference. NULL bounds.
        "Flask 2.3.1 is the framework"          → null, null, 0.0
        "User has Python 3.11 installed"         → null, null, 0.0
        "MVP scope includes user login"          → null, null, 0.0
        "Database is SQLite at /var/db/app.db"   → null, null, 0.0

INVARIANTS — violations are rejected:
  - Both bounds null AND confidence=0.0, OR
  - Both bounds present AND confidence in {0.5, 0.7, 0.8, 1.0}
  - end > start (for true instants like "at 14:30Z", end = start + 1 nanosecond)
  - confidence MUST be exactly one of {0.0, 0.5, 0.7, 0.8, 1.0}

============================================================
SECTION 4 — OTHER FIELDS
============================================================

- context: 1-2 sentences capturing implications the message does not literally state. Do not restate the input.
- keywords: 5-8 specific search terms.
- tags: 3-5 from this CLOSED SET — do not invent others:
    {preference, decision, constraint, workflow, entity, pattern, temporal, code, deadline, planning}
- foresight_signals: forward-looking statements with explicit expiry dates, if any.

============================================================
ANTI-PATTERNS (real failure modes — do not repeat)
============================================================

A) Conversation-date contamination
  Input: ref_time=2024-03-15, message="I'm setting up Flask 2.3.1..."
  WRONG:  fact "Project uses Flask 2.3.1" with [2024-03-15, 2024-03-16), conf=0.5
  RIGHT:  fact "Project uses Flask 2.3.1" with null, null, conf=0.0
  Why: Flask being installed is an ongoing property, not an event on March 15.

B) Bleeding one fact's date into sibling facts
  Input: "Meet April 15 deadline. MVP includes login, expense tracking, analytics."
  WRONG:  every fact gets [2024-04-15, 2024-04-16), conf=0.8
  RIGHT:  - "Project has April 15 deadline"     → [2024-04-15, 2024-04-16), 0.8
          - "MVP scope includes user login"     → null, null, 0.0
          - "MVP scope includes expense tracking" → null, null, 0.0
          - "MVP scope includes analytics"      → null, null, 0.0
  Why: The deadline has a date. The scope contents do not.

C) Vague-by-default
  WRONG: when uncertain, pick conf=0.5 with the conversation window as bounds.
  RIGHT: when the fact text has no temporal word, conf=0.0 with NULL bounds.
  Why: 0.5 is reserved for explicit vague TEMPORAL LANGUAGE ("recently", "around March"). It is not a fallback for "I don't know when."
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
