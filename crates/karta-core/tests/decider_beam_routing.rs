//! Offline routing eval: Jev vs the keyword classifier on BEAM 100K's 400
//! probing questions. Only question text is used — no ingest, no answer LLM.
//!
//! Skipped unless `JEV_API_KEY` is set and the dataset exists
//! (`BEAM_DATASET_PATH`, default `data/beam-100k.json`; build it with
//! `python3 data/convert_beam.py data/beam-100k.parquet data/beam-100k.json`).
//!
//!   JEV_API_KEY=... cargo test -p karta-core --test decider_beam_routing -- --nocapture
//!
//! BEAM has no "correct mode" label, so five abilities with a natural target
//! mode are scored; the other five only report the routed distribution.
//! Optional: `DECIDER_OUT=path.jsonl` writes every routing decision.

use std::collections::BTreeMap;
use std::io::Write;

use futures::stream::{self, StreamExt};
use karta_core::decide::questions::*;
use karta_core::decide::{Decider, JevDecider};
use karta_core::read::{classify_query, QueryMode};

fn target_mode(ability: &str) -> Option<QueryMode> {
    match ability {
        "event_ordering" => Some(QueryMode::Temporal),
        "summarization" => Some(QueryMode::Breadth),
        "knowledge_update" => Some(QueryMode::Recency),
        "temporal_reasoning" => Some(QueryMode::Computation),
        "contradiction_resolution" => Some(QueryMode::Existence),
        _ => None,
    }
}

fn load_questions() -> Option<Vec<(String, String)>> {
    let path = std::env::var("BEAM_DATASET_PATH")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/beam-100k.json").into());
    let raw = std::fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let mut out = Vec::new();
    for conv in v["conversations"].as_array()? {
        for q in conv["questions"].as_array()? {
            out.push((
                q["ability"].as_str()?.to_string(),
                q["question"].as_str()?.to_string(),
            ));
        }
    }
    Some(out)
}

#[tokio::test]
async fn decider_beam_routing() {
    let Some(jev) = JevDecider::from_env() else {
        eprintln!("JEV_API_KEY not set — skipping");
        return;
    };
    let Some(questions) = load_questions() else {
        eprintln!("BEAM dataset not found — skipping");
        return;
    };
    let min_conf: f32 = std::env::var("K_DECIDER_MIN_CONF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7);
    let asks = [query_mode_question(), query_predicate_question()];

    let results: Vec<_> = stream::iter(questions.iter())
        .map(|(ability, q)| {
            let jev = &jev;
            let asks = &asks;
            async move { (ability, q, jev.decide(q, asks).await) }
        })
        .buffered(8)
        .collect()
        .await;

    // ability -> (n, keyword hits, jev hits, routed hits, mode histogram)
    let mut per: BTreeMap<String, (usize, usize, usize, usize, BTreeMap<String, usize>)> =
        BTreeMap::new();
    let mut out = std::env::var("DECIDER_OUT")
        .ok()
        .map(|p| std::fs::File::create(p).expect("DECIDER_OUT"));
    let mut errors = 0;
    let mut confidences: Vec<f32> = Vec::new();

    for (ability, q, res) in results {
        let kw = classify_query(q).mode;
        let entry = per.entry(ability.clone()).or_default();
        entry.0 += 1;
        let Ok(d) = res else {
            errors += 1;
            continue;
        };
        let jev_mode = query_mode_from(&d[QUERY_MODE]);
        let conf = d[QUERY_MODE].confidence();
        confidences.push(conf);
        // What the read path would actually use (keyword stands in for the
        // embedding centroid, which needs a real embedding model).
        let routed = if conf >= min_conf { jev_mode.unwrap_or(kw) } else { kw };
        if let Some(t) = target_mode(ability) {
            entry.1 += (kw == t) as usize;
            entry.2 += (jev_mode == Some(t)) as usize;
            entry.3 += (routed == t) as usize;
        }
        *entry.4.entry(format!("{:?}", routed)).or_default() += 1;
        if let Some(f) = out.as_mut() {
            let line = serde_json::json!({
                "ability": ability, "question": q,
                "keyword": format!("{:?}", kw),
                "jev": jev_mode.map(|m| format!("{:?}", m)), "jev_conf": conf,
                "routed": format!("{:?}", routed),
                "predicate": d[QUERY_PREDICATE].label(),
                "predicate_conf": d[QUERY_PREDICATE].confidence(),
            });
            writeln!(f, "{}", line).unwrap();
        }
    }

    eprintln!("\nmin_conf={}  errors={}/{}", min_conf, errors, questions.len());
    eprintln!("{:<26} {:>8} {:>8} {:>8}   routed distribution", "ability", "keyword", "jev", "routed");
    let (mut tk, mut tj, mut tr, mut tn) = (0, 0, 0, 0);
    for (ability, (n, k, j, r, hist)) in &per {
        let scored = target_mode(ability).is_some();
        let pct = |x: usize| if scored { format!("{:.0}%", 100.0 * x as f32 / *n as f32) } else { "-".into() };
        eprintln!("{:<26} {:>8} {:>8} {:>8}   {:?}", ability, pct(*k), pct(*j), pct(*r), hist);
        if scored {
            tk += k;
            tj += j;
            tr += r;
            tn += n;
        }
    }
    let pct = |x: usize| 100.0 * x as f32 / tn.max(1) as f32;
    eprintln!("{:<26} {:>7.0}% {:>7.0}% {:>7.0}%", "SCORED (5 abilities)", pct(tk), pct(tj), pct(tr));
    confidences.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let below = confidences.iter().filter(|c| **c < min_conf).count();
    eprintln!("jev mode confidence: {} of {} below {}", below, confidences.len(), min_conf);

    assert!(errors * 10 < questions.len(), "too many Jev errors: {}", errors);
}
