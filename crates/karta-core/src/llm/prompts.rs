/// All LLM prompts used by Karta, centralized for easy tuning.
pub struct Prompts;

impl Prompts {
    pub fn note_attributes_system() -> &'static str {
        "You are a memory indexing system. Given a piece of information, extract structured attributes.\n\n\
         Also extract 1-5 atomic facts. Each fact should be:\n\
         - A single, self-contained statement that makes sense without context\n\
         - Independently verifiable (not \"he said\" but \"John said\")\n\
         - Include specific values, dates, numbers when present\n\
         - Each fact about ONE thing\n\
         Example: \"I'm using Flask 2.3.1 on Python 3.11 and my budget is $500\" becomes:\n\
           1. \"The user is using Flask version 2.3.1\" (subject: \"Flask\")\n\
           2. \"The user is using Python version 3.11\" (subject: \"Python\")\n\
           3. \"The user's budget is $500\" (subject: \"budget\")\n\n\
         Respond with JSON only in this exact shape:\n\
         {\n\
           \"context\": \"A rich 1-2 sentence description capturing deeper meaning, implications, and why this matters — not just a restatement of the content. Include any specific dates or deadlines mentioned.\",\n\
           \"keywords\": [\"5 to 8 specific terms that would help find this note\"],\n\
           \"tags\": [\"3 to 5 categorical labels like: preference, decision, constraint, workflow, entity, pattern\"],\n\
           \"foresightSignals\": [{\"content\": \"forward-looking statement with time reference\", \"valid_until\": \"YYYY-MM-DD or null\"}],\n\
           \"atomic_facts\": [{\"content\": \"single atomic statement\", \"subject\": \"primary entity or null\"}]\n\
         }"
    }

    pub fn note_attributes_user(content: &str) -> String {
        format!("Index this memory:\n\n{}", content)
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
         3. TEMPORAL REASONING: Pay close attention to dates and time references in notes. \
         When computing durations or ordering events, show your work: \
         state the specific dates, then calculate. \
         Use note timestamps and date references as the source of truth.\n\n\
         4. PROVENANCE: Each note is tagged with provenance and age:\n\
         - FACT = directly observed information\n\
         - INFERRED:{type} = derived by reasoning (deduction, induction, abduction, etc.)\n\
         - FACT:from-{id} = atomic fact extracted from a note; highly precise for specific values\n\
         - DIGEST:{id} = episode summary with pre-computed counts and entity tracking; treat aggregation counts as authoritative\n\
         Treat FACT notes as authoritative. Treat INFERRED notes as supporting evidence \
         but flag them as inferences when they are central to your answer. \
         When FACT and INFERRED notes conflict, prioritize the FACT. \
         DIGEST notes contain pre-computed aggregations — use their counts rather than re-counting from individual notes.\n\n\
         5. RECENCY: More recent notes generally supersede older ones on the same topic. \
         When answering about current state, prefer the most recent note.\n\n\
         6. FORMAT: Match the format the user expects. \
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
             5. digest_text: A 2-4 sentence summary that embeds well for retrieval. \
                Include specific names, numbers, and dates.\n\n\
             Notes:\n{}\n\n\
             Respond with JSON:\n\
             {{\n\
               \"entities\": [{{\"name\": \"...\", \"type\": \"person|tool|framework|project|date|number|other\", \
                 \"count\": 1, \"latest_value\": \"...or null\"}}],\n\
               \"date_range\": {{\"earliest\": \"YYYY-MM-DD\", \"latest\": \"YYYY-MM-DD\"}} or null,\n\
               \"aggregations\": [{{\"label\": \"movies discussed\", \"count\": 5, \"items\": [\"...\"]}}],\n\
               \"topic_sequence\": [\"first topic\", \"second topic\"],\n\
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
             Find the overall topic progression across episodes.\n\n\
             Episode digests:\n{}\n\n\
             Respond with JSON:\n\
             {{\n\
               \"entity_timeline\": [{{\"name\": \"...\", \"type\": \"...\", \
                 \"changes\": [{{\"episode_id\": \"...\", \"value\": \"...\"}}]}}],\n\
               \"cross_aggregations\": [{{\"label\": \"...\", \"count\": 0, \"items\": [\"...\"]}}],\n\
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
