//! Real LLM eval — uses actual Azure OpenAI credentials from .env
//!
//! Run: cargo test --test real_eval -- --nocapture --ignored
//!
//! This test is #[ignore]d by default so `cargo test` won't hit your API.

use karta_core::Karta;
use karta_core::config::KartaConfig;

async fn create_real_karta(name: &str) -> Karta {
    let data_dir = format!("/tmp/karta-real-{}", name.replace(' ', "-").to_lowercase());
    let _ = std::fs::remove_dir_all(&data_dir);

    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir;

    Karta::with_defaults(config)
        .await
        .expect("Failed to create Karta — check .env credentials")
}

fn check_contains(answer: &str, pattern: &str) -> bool {
    let norm = answer
        .to_lowercase()
        .replace(['\u{2010}', '\u{2011}', '\u{2013}', '\u{2014}'], "-")
        .replace(['\u{2018}', '\u{2019}'], "'");
    pattern
        .split('|')
        .any(|alt| norm.contains(&alt.to_lowercase()))
}

#[tokio::test]
#[ignore]
async fn real_eval_user_preference_linking() {
    let karta = create_real_karta("user-pref").await;

    let notes = [
        "Sarah Chen, head of Marketing Ops at Brightline, prefers all workflow notifications to go via Slack rather than email — she says her team ignores email alerts.",
        "Brightline's IT policy requires all third-party integrations to go through an approved vendor list. Slack is on the list; most other tools are not.",
        "Sarah asked Fay to build a lead routing workflow. She wants it to assign leads based on territory, not round-robin.",
        "Sarah mentioned she's trying to reduce her team's tool fragmentation — they currently use 6 different platforms for ops.",
    ];

    for (i, note) in notes.iter().enumerate() {
        let result = karta.add_note(note).await.unwrap();
        println!(
            "Note {}: {} links, id={}",
            i + 1,
            result.links.len(),
            &result.id[..8]
        );
    }

    println!("\n--- Query 1 ---");
    let answer = karta
        .ask(
            "What should I know before building a notification system for Sarah?",
            5,
        )
        .await
        .unwrap()
        .answer;
    println!("A: {}\n", answer);
    assert!(check_contains(&answer, "Slack"), "Should mention Slack");
    assert!(check_contains(&answer, "Sarah"), "Should mention Sarah");

    println!("--- Query 2 ---");
    let answer = karta
        .ask("What constraints exist for Brightline integrations?", 5)
        .await
        .unwrap()
        .answer;
    println!("A: {}\n", answer);
    assert!(
        check_contains(&answer, "vendor|approved"),
        "Should mention vendor list"
    );

    println!("--- Stats ---");
    println!("Notes: {}", karta.note_count().await.unwrap());
}

#[tokio::test]
#[ignore]
async fn real_eval_retroactive_evolution() {
    let karta = create_real_karta("retro-evo").await;

    let notes = [
        "The CISO at TechCorp requires all automations to produce audit logs stored in their internal S3 bucket, not in Fay's cloud.",
        "TechCorp is running a SOC 2 Type II audit in Q3 2025 and needs all vendor tooling to be compliant.",
        "The ops team at TechCorp wants to automate their contractor onboarding — 50+ contractors per month.",
    ];

    for (i, note) in notes.iter().enumerate() {
        let result = karta.add_note(note).await.unwrap();
        println!("Note {}: {} links", i + 1, result.links.len());
    }

    println!("\n--- Query ---");
    let answer = karta
        .ask(
            "What compliance requirements affect the TechCorp contractor onboarding automation?",
            5,
        )
        .await
        .unwrap()
        .answer;
    println!("A: {}\n", answer);
    assert!(
        check_contains(&answer, "audit|S3"),
        "Should mention audit/S3"
    );
    assert!(
        check_contains(&answer, "SOC|compliance"),
        "Should mention SOC 2/compliance"
    );
}

#[tokio::test]
#[ignore]
async fn real_eval_cross_entity() {
    let karta = create_real_karta("cross-entity").await;

    let notes = [
        "HubSpot-to-Salesforce sync workflows consistently fail on contact deduplication when both systems have the same email with different capitalisation.",
        "Acme Corp is building a CRM sync between HubSpot and Salesforce. Their ops lead, Marcus, wants bidirectional sync.",
        "A bidirectional CRM sync requires a conflict resolution strategy — last-write-wins is the default but causes data loss in practice.",
        "Marcus at Acme has a hard deadline: the workflow needs to be live before their Q4 sales push starts Oct 1.",
        "The deduplication issue in HubSpot/Salesforce syncs can be solved with a normalisation step that lowercases emails before comparison.",
    ];

    for (i, note) in notes.iter().enumerate() {
        let result = karta.add_note(note).await.unwrap();
        println!("Note {}: {} links", i + 1, result.links.len());
    }

    println!("\n--- Query ---");
    let answer = karta
        .ask(
            "What technical issues should I warn Marcus about before building the Acme CRM sync?",
            5,
        )
        .await
        .unwrap()
        .answer;
    println!("A: {}\n", answer);
    assert!(
        check_contains(&answer, "dedup|deduplication"),
        "Should mention deduplication"
    );
    assert!(
        check_contains(&answer, "conflict"),
        "Should mention conflict resolution"
    );
}

#[tokio::test]
#[ignore]
async fn real_eval_contradiction() {
    let karta = create_real_karta("contradiction").await;

    let notes = [
        "Vanguard Logistics confirmed in writing on Feb 10 that their data must stay in the EU region (Frankfurt). They cited GDPR as the reason.",
        "Vanguard Logistics' new CTO, hired Mar 1, emailed saying they want all Fay workloads to run in US-East for latency reasons — their main operations center moved to Virginia.",
        "Vanguard Logistics processes 200K shipment records daily. Their current pipeline runs on AWS eu-central-1.",
        "Vanguard Logistics has a DPA (Data Processing Agreement) that specifies EU-only data residency. The DPA was signed Jan 2025 and is valid for 2 years.",
        "The Vanguard Logistics engineering team started provisioning resources in us-east-1 on Mar 15 without informing the compliance team.",
    ];

    for (i, note) in notes.iter().enumerate() {
        let result = karta.add_note(note).await.unwrap();
        println!("Note {}: {} links", i + 1, result.links.len());
    }

    println!("\n--- Query ---");
    let answer = karta
        .ask("Where should Vanguard Logistics data be processed?", 5)
        .await
        .unwrap()
        .answer;
    println!("A: {}\n", answer);
    assert!(
        check_contains(&answer, "EU|Frankfurt|eu-central"),
        "Should mention EU"
    );
    assert!(
        check_contains(&answer, "US-East|Virginia|us-east"),
        "Should mention US-East"
    );

    println!("--- Dreaming ---");
    let dream_run = karta.run_dreaming("workspace", "vanguard").await.unwrap();
    println!("Dreams attempted: {}", dream_run.dreams_attempted);
    println!("Dreams written:   {}", dream_run.dreams_written);
    for dream in &dream_run.dreams {
        let icon = if dream.would_write { "W" } else { "." };
        println!(
            "  [{}] {} conf={:.2}: {}",
            icon,
            dream.dream_type.as_str(),
            dream.confidence,
            &dream.dream_content.chars().take(120).collect::<String>()
        );
    }
}

#[tokio::test]
#[ignore]
async fn real_eval_long_chain() {
    let karta = create_real_karta("long-chain").await;

    let notes = [
        "ClearView Insurance wants to automate claims adjudication. Target: process a claim decision within 2 minutes of submission.",
        "ClearView's claims data lives in a legacy IBM AS/400 system. The only extraction method is a nightly batch SFTP export — no real-time API.",
        "The claims adjudication model (ML) requires the claimant's full policy history, which is only in the AS/400 system.",
        "ClearView's compliance team mandates that all automated claim decisions must include a human-reviewable audit trail with the model's reasoning.",
        "ClearView's IT team has budget approval to modernize one system per quarter. The AS/400 migration is scheduled for Q1 2026.",
        "The 2-minute SLA was promised to ClearView's board by the COO and is considered non-negotiable for the current quarter.",
    ];

    for (i, note) in notes.iter().enumerate() {
        let result = karta.add_note(note).await.unwrap();
        println!("Note {}: {} links", i + 1, result.links.len());
    }

    println!("\n--- Query ---");
    let answer = karta
        .ask(
            "Can ClearView meet their 2-minute claims processing target? What's blocking it?",
            5,
        )
        .await
        .unwrap()
        .answer;
    println!("A: {}\n", answer);
    assert!(
        check_contains(&answer, "AS/400|AS400|legacy"),
        "Should mention AS/400"
    );
    assert!(
        check_contains(&answer, "nightly|batch|SFTP"),
        "Should mention batch/SFTP constraint"
    );
}
