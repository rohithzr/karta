//! Live smoke + mini-eval for the Jev decider against hand-labeled queries.
//!
//! Skipped unless `JEV_API_KEY` is set. Optional: `JEV_BASE_URL`, `JEV_MODEL`.
//!
//!   JEV_API_KEY=... cargo test -p karta-core --test decider_jev_live -- --nocapture
//!
//! Purpose: (1) confirm the unverified wire format in `decide/jev.rs` works
//! against the real API, (2) print Jev vs today's keyword classifier on
//! query mode and ledger predicate so the first A/B has a starting point.
//! No accuracy floor is asserted until a baseline exists.

use karta_core::decide::questions::*;
use karta_core::decide::{Decider, JevDecider};
use karta_core::extract::slots::Predicate;
use karta_core::read::{classify_query, QueryMode};

/// (query, expected mode, expected predicate)
const LABELED: &[(&str, QueryMode, Option<Predicate>)] = &[
    ("In what order did I bring up the database migration, the auth bug and the launch plan?", QueryMode::Temporal, None),
    ("What did I work on first after the offsite, and what came next?", QueryMode::Temporal, None),
    ("Where am I working these days?", QueryMode::Recency, Some(Predicate::Employer)),
    ("What's the deadline for the grant application now that it moved?", QueryMode::Recency, Some(Predicate::Deadline)),
    ("Which database are we on after the switch?", QueryMode::Recency, Some(Predicate::TechChoice)),
    ("Who owns the billing service at the moment?", QueryMode::Recency, Some(Predicate::Ownership)),
    ("Give me an overview of how my thesis project developed.", QueryMode::Breadth, None),
    ("Summarize everything I told you about the kitchen renovation.", QueryMode::Breadth, None),
    ("How many days passed between my first interview and the offer?", QueryMode::Computation, None),
    ("How much more did I spend on rent in June than in May?", QueryMode::Computation, Some(Predicate::Amount)),
    ("Did I ever say I was learning Spanish?", QueryMode::Existence, None),
    ("Did I contradict myself about whether I like remote work?", QueryMode::Existence, Some(Predicate::Preference)),
    ("What testing framework did I pick for the frontend?", QueryMode::Standard, Some(Predicate::TechChoice)),
    ("What did my manager say about my presentation?", QueryMode::Standard, None),
    ("What's my current job title?", QueryMode::Recency, Some(Predicate::RoleTitle)),
    ("What was the model's accuracy after the last fine-tune?", QueryMode::Recency, Some(Predicate::MetricValue)),
    ("I'm known for being punctual, right?", QueryMode::Existence, None),
    ("When is the dentist appointment scheduled?", QueryMode::Standard, Some(Predicate::ScheduledDate)),
];

#[tokio::test]
async fn decider_jev_live() {
    let Some(jev) = JevDecider::from_env() else {
        eprintln!("JEV_API_KEY not set — skipping");
        return;
    };
    let questions = [query_mode_question(), query_predicate_question(), query_temporal_question()];

    let (mut mode_jev, mut mode_kw, mut pred_jev) = (0, 0, 0);
    for (query, want_mode, want_pred) in LABELED {
        let started = std::time::Instant::now();
        let out = jev
            .decide(query, &questions)
            .await
            .unwrap_or_else(|e| panic!("Jev failed on {:?}: {}", query, e));
        let ms = started.elapsed().as_millis();

        let got_mode = query_mode_from(&out[QUERY_MODE]);
        let got_pred = query_predicate_from(&out[QUERY_PREDICATE]);
        let kw_mode = classify_query(query).mode;
        mode_jev += (got_mode == Some(*want_mode)) as usize;
        mode_kw += (kw_mode == *want_mode) as usize;
        pred_jev += (got_pred == *want_pred) as usize;

        eprintln!(
            "{:>5}ms | mode jev={:?} ({:.2}) kw={:?} want={:?} | pred jev={:?} want={:?} | temporal p={:.2} | {}",
            ms,
            got_mode,
            out[QUERY_MODE].confidence(),
            kw_mode,
            want_mode,
            got_pred,
            want_pred,
            out[QUERY_TEMPORAL].confidence(),
            query
        );
    }
    let n = LABELED.len();
    eprintln!(
        "\nquery_mode: jev {}/{}  keyword {}/{}   predicate: jev {}/{}",
        mode_jev, n, mode_kw, n, pred_jev, n
    );
}
