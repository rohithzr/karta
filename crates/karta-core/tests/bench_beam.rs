//! BEAM-style benchmark harness for Karta memory system.
//!
//! BEAM ("BEyond A Million tokens") is a benchmark from ICLR 2026 that evaluates
//! long-term memory in LLMs across 10 distinct memory abilities:
//!
//!   1. Contradiction Resolution — detect and reconcile inconsistent statements
//!   2. Event Ordering — reconstruct sequences of evolving information
//!   3. Information Extraction — recall entities and factual details
//!   4. Instruction Following — sustained adherence to user-specified constraints
//!   5. Information Update — revise stored facts as new ones appear
//!   6. Multi-hop Reasoning — inference across multiple non-adjacent segments
//!   7. Preference Following — adapt to evolving user preferences
//!   8. Summarization — abstract and compress dialogue content
//!   9. Temporal Reasoning — reason about time relations
//!  10. Abstention — recognize unanswerable questions
//!
//! BEAM comes in 4 scales: 100K, 500K, 1M, and 10M tokens.
//! "BEAM 100K" means running against the 128K-token conversation set.
//!
//! The real dataset is on HuggingFace: https://huggingface.co/datasets/Mohammadta/BEAM
//! Paper: https://arxiv.org/abs/2510.27246
//! Code: https://github.com/mohammadtavakoli78/BEAM
//!
//! This file provides:
//!   - Data structures matching BEAM's format (sessions, messages, questions)
//!   - Hand-crafted mini scenarios testing the same 10 memory abilities
//!   - Metrics: accuracy (must_contain), token usage, latency, notes stored
//!   - A placeholder for loading the real BEAM dataset
//!
//! LOCOMO is the other major memory benchmark (Maharana et al., ACL 2024).
//! It provides 10 conversations (~300 turns, ~9K tokens each, up to 35 sessions)
//! with questions in 5 categories: single-hop, multi-hop, temporal, commonsense,
//! and adversarial. Dataset: https://github.com/snap-research/locomo
//!
//! Run: cargo test --test bench_beam -- --ignored --nocapture

use std::sync::Arc;
use std::time::Instant;

use karta_core::Karta;
use karta_core::config::KartaConfig;
use karta_core::llm::LlmProvider;
use karta_core::llm::MockLlmProvider;
use karta_core::store::lance::LanceVectorStore;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::{GraphStore, VectorStore};

// ---------------------------------------------------------------------------
// Data structures (aligned with BEAM's format)
// ---------------------------------------------------------------------------

/// A single conversation turn within a session.
struct Message {
    role: &'static str, // "user" or "assistant"
    content: &'static str,
}

/// A session groups messages that occurred in one sitting.
/// BEAM conversations span multiple sessions; cross-session reasoning
/// requires connecting facts across session boundaries.
#[allow(dead_code)]
struct Session {
    session_id: usize,
    /// Optional label, e.g. "2025-01-15" for temporal scenarios.
    label: &'static str,
    messages: Vec<Message>,
}

/// A BEAM-style probing question.
struct BeamQuestion {
    /// The question to ask the memory system.
    question: &'static str,
    /// Which BEAM memory ability this tests.
    ability: BeamAbility,
    /// Patterns the answer must contain (pipe-separated alternatives).
    must_contain: Vec<&'static str>,
    /// If true, the correct answer is "I don't know" / abstention.
    expects_abstention: bool,
}

/// The 10 BEAM memory abilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BeamAbility {
    ContradictionResolution,
    EventOrdering,
    InformationExtraction,
    InstructionFollowing,
    InformationUpdate,
    MultiHopReasoning,
    PreferenceFollowing,
    Summarization,
    TemporalReasoning,
    Abstention,
}

impl BeamAbility {
    fn as_str(&self) -> &'static str {
        match self {
            Self::ContradictionResolution => "contradiction_resolution",
            Self::EventOrdering => "event_ordering",
            Self::InformationExtraction => "information_extraction",
            Self::InstructionFollowing => "instruction_following",
            Self::InformationUpdate => "information_update",
            Self::MultiHopReasoning => "multi_hop_reasoning",
            Self::PreferenceFollowing => "preference_following",
            Self::Summarization => "summarization",
            Self::TemporalReasoning => "temporal_reasoning",
            Self::Abstention => "abstention",
        }
    }
}

/// A complete BEAM-style scenario: multiple sessions + probing questions.
struct BeamScenario {
    name: &'static str,
    sessions: Vec<Session>,
    questions: Vec<BeamQuestion>,
}

/// Metrics collected per scenario.
#[derive(Debug, Default)]
struct ScenarioMetrics {
    name: String,
    notes_stored: usize,
    ingest_ms: u128,
    questions_total: usize,
    questions_passed: usize,
    checks_total: usize,
    checks_passed: usize,
    query_latencies_ms: Vec<u128>,
    /// Maps ability -> (passed, total)
    ability_scores: Vec<(BeamAbility, usize, usize)>,
}

/// Aggregate metrics across all scenarios.
#[derive(Debug, Default)]
struct BenchmarkReport {
    scenario_metrics: Vec<ScenarioMetrics>,
    total_ingest_ms: u128,
    total_query_ms: u128,
    total_notes: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Truncate a string to at most `max_bytes`, respecting UTF-8 char boundaries.
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Normalize unicode dashes/quotes to ASCII equivalents for matching.
fn normalize_for_matching(s: &str) -> String {
    s.to_lowercase()
        .replace(
            [
                '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2015}',
            ],
            "-",
        ) // dash and hyphen-like characters
        .replace(['\u{2018}', '\u{2019}'], "'") // left and right single quotes
        .replace(['\u{201C}', '\u{201D}'], "\"") // left and right double quotes
}

/// Check if `answer` contains at least one alternative from a `|`-separated pattern.
fn check_must_contain(answer: &str, pattern: &str) -> bool {
    let answer_norm = normalize_for_matching(answer);
    pattern
        .split('|')
        .any(|alt| answer_norm.contains(&normalize_for_matching(alt)))
}

/// Ingest all user messages from sessions into Karta with session-aware episode creation.
async fn ingest_sessions(karta: &Karta, sessions: &[Session]) -> usize {
    let mut count = 0usize;
    let mut global_turn: u32 = 0;
    for session in sessions {
        let session_id = format!("session-{}", session.session_id);

        // Parse session label as date for source_timestamp
        let source_timestamp = if session.label.is_empty() {
            None
        } else {
            chrono::NaiveDate::parse_from_str(session.label, "%Y-%m-%d")
                .ok()
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .map(|dt| dt.and_utc())
        };

        for msg in &session.messages {
            if msg.role == "user" {
                let note = if session.label.is_empty() {
                    msg.content.to_string()
                } else {
                    format!("[{}] {}", session.label, msg.content)
                };
                let result = karta
                    .add_note_with_metadata(&note, &session_id, Some(global_turn), source_timestamp)
                    .await
                    .unwrap();
                count += 1;
                global_turn += 1;
                println!("  Note {}: {} links", count, result.links.len());
            }
        }
    }
    count
}

fn data_dir(prefix: &str, scenario_name: &str) -> String {
    let dir_name = format!(
        "karta-{}-{}",
        prefix,
        scenario_name.replace([' ', '/'], "-").to_lowercase()
    );
    std::env::temp_dir()
        .join(dir_name)
        .to_string_lossy()
        .into_owned()
}

async fn create_karta_mock(scenario_name: &str) -> Karta {
    let dir = data_dir("beam-mock", scenario_name);
    let _ = std::fs::remove_dir_all(&dir);

    let vector_store = Arc::new(LanceVectorStore::new(&dir).await.unwrap()) as Arc<dyn VectorStore>;
    let graph_store = Arc::new(SqliteGraphStore::new(&dir).unwrap()) as Arc<dyn GraphStore>;
    let llm = Arc::new(MockLlmProvider::new()) as Arc<dyn LlmProvider>;

    let mut config = KartaConfig::default();
    config.episode.enabled = true;
    Karta::new(vector_store, graph_store, llm, config)
        .await
        .unwrap()
}

async fn create_karta_real(scenario_name: &str) -> Karta {
    // Include a random suffix to avoid collisions when tests run in parallel
    let suffix = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let dir = data_dir(&format!("beam-real-{}", suffix), scenario_name);
    let _ = std::fs::remove_dir_all(&dir);

    let mut config = KartaConfig::default();
    config.storage.data_dir = dir;
    config.episode.enabled = true;

    Karta::with_defaults(config)
        .await
        .expect("Failed to create Karta — check .env credentials")
}

// ---------------------------------------------------------------------------
// Hand-crafted BEAM-style scenarios
// ---------------------------------------------------------------------------

fn beam_scenarios() -> Vec<BeamScenario> {
    vec![
        // --- 1. Single-session recall + Information Extraction ---
        BeamScenario {
            name: "Single-session entity recall",
            sessions: vec![Session {
                session_id: 1,
                label: "2025-03-10",
                messages: vec![
                    Message {
                        role: "user",
                        content: "I just hired Elena Vasquez as our new VP of Engineering. She's coming from Stripe where she led the payments infrastructure team.",
                    },
                    Message {
                        role: "assistant",
                        content: "Great hire! What's her start date?",
                    },
                    Message {
                        role: "user",
                        content: "She starts April 1st. Her first project will be migrating our monolith to microservices.",
                    },
                    Message {
                        role: "assistant",
                        content: "Makes sense given her infrastructure background.",
                    },
                    Message {
                        role: "user",
                        content: "Elena's team will be 12 engineers. She wants to use Kubernetes on GCP, not AWS — she had bad experiences with EKS at Stripe.",
                    },
                    Message {
                        role: "assistant",
                        content: "Noted. GCP + GKE it is.",
                    },
                    Message {
                        role: "user",
                        content: "Budget for the migration is $2.4M over 18 months. Elena has full authority on technical decisions.",
                    },
                ],
            }],
            questions: vec![
                BeamQuestion {
                    question: "Who is the new VP of Engineering and where did they come from?",
                    ability: BeamAbility::InformationExtraction,
                    must_contain: vec!["Elena|Vasquez", "Stripe"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "What cloud platform was chosen for the migration and why?",
                    ability: BeamAbility::InformationExtraction,
                    must_contain: vec!["GCP|GKE|Kubernetes", "AWS|EKS|bad experience"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "What is the budget for the migration project?",
                    ability: BeamAbility::InformationExtraction,
                    must_contain: vec!["2.4|2,400,000|2.4M"],
                    expects_abstention: false,
                },
            ],
        },
        // --- 2. Cross-session reasoning + Multi-hop ---
        BeamScenario {
            name: "Cross-session multi-hop reasoning",
            sessions: vec![
                Session {
                    session_id: 1,
                    label: "2025-01-10",
                    messages: vec![
                        Message {
                            role: "user",
                            content: "Our data warehouse is on Snowflake. We're paying $8K/month and growing 20% quarter over quarter.",
                        },
                        Message {
                            role: "assistant",
                            content: "That's solid growth. Any cost concerns?",
                        },
                        Message {
                            role: "user",
                            content: "Not yet, but our CFO Rachel wants a cost projection for the next 12 months.",
                        },
                    ],
                },
                Session {
                    session_id: 2,
                    label: "2025-02-15",
                    messages: vec![
                        Message {
                            role: "user",
                            content: "We just signed a deal with MegaCorp — they'll send us 50GB of event data daily starting March 1.",
                        },
                        Message {
                            role: "assistant",
                            content: "That's a lot of data. Will your Snowflake setup handle it?",
                        },
                        Message {
                            role: "user",
                            content: "That's exactly what I'm worried about. The MegaCorp data needs to be queryable within 4 hours of arrival.",
                        },
                    ],
                },
                Session {
                    session_id: 3,
                    label: "2025-03-05",
                    messages: vec![
                        Message {
                            role: "user",
                            content: "Snowflake costs jumped to $18K this month after the MegaCorp data started flowing. Rachel is not happy.",
                        },
                        Message {
                            role: "assistant",
                            content: "That's more than double. What changed?",
                        },
                        Message {
                            role: "user",
                            content: "The auto-scaling warehouses are running hot processing the MegaCorp ingestion. We need a cheaper approach.",
                        },
                    ],
                },
            ],
            questions: vec![
                BeamQuestion {
                    question: "Why did Snowflake costs increase and who is concerned about it?",
                    ability: BeamAbility::MultiHopReasoning,
                    must_contain: vec!["MegaCorp|50GB|event data", "Rachel|CFO"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "What SLA exists for the MegaCorp data ingestion?",
                    ability: BeamAbility::InformationExtraction,
                    must_contain: vec!["4 hour|queryable"],
                    expects_abstention: false,
                },
            ],
        },
        // --- 3. Temporal reasoning ---
        BeamScenario {
            name: "Temporal ordering and awareness",
            sessions: vec![
                Session {
                    session_id: 1,
                    label: "2025-01-05",
                    messages: vec![Message {
                        role: "user",
                        content: "We decided to use PostgreSQL as our primary database. The team evaluated MongoDB and Postgres and chose Postgres for ACID compliance.",
                    }],
                },
                Session {
                    session_id: 2,
                    label: "2025-02-20",
                    messages: vec![Message {
                        role: "user",
                        content: "We're hitting performance issues with PostgreSQL on our analytics queries. Looking at adding a read replica.",
                    }],
                },
                Session {
                    session_id: 3,
                    label: "2025-03-15",
                    messages: vec![Message {
                        role: "user",
                        content: "We've decided to move analytics workloads off Postgres entirely. We'll use ClickHouse for analytics and keep Postgres for OLTP only.",
                    }],
                },
            ],
            questions: vec![
                BeamQuestion {
                    question: "What was the first database decision made and when?",
                    ability: BeamAbility::TemporalReasoning,
                    must_contain: vec!["PostgreSQL|Postgres", "January|2025-01"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "What is the current database architecture?",
                    ability: BeamAbility::InformationUpdate,
                    must_contain: vec!["ClickHouse|analytics", "Postgres|OLTP"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "What happened between the initial Postgres decision and the ClickHouse migration?",
                    ability: BeamAbility::EventOrdering,
                    must_contain: vec!["performance|analytics", "replica"],
                    expects_abstention: false,
                },
            ],
        },
        // --- 4. Contradiction resolution ---
        BeamScenario {
            name: "Contradiction detection and resolution",
            sessions: vec![
                Session {
                    session_id: 1,
                    label: "2025-01-20",
                    messages: vec![
                        Message {
                            role: "user",
                            content: "Our security policy requires all data to be encrypted at rest using AES-256. No exceptions — the CISO signed off on this.",
                        },
                        Message {
                            role: "assistant",
                            content: "Clear. AES-256 at rest, no exceptions.",
                        },
                        Message {
                            role: "user",
                            content: "We also require all API traffic to go through our API gateway. Direct service-to-service calls over the public internet are banned.",
                        },
                    ],
                },
                Session {
                    session_id: 2,
                    label: "2025-03-01",
                    messages: vec![
                        Message {
                            role: "user",
                            content: "The new CTO says we can relax encryption to AES-128 for non-PII data to save on compute costs. PII data stays AES-256.",
                        },
                        Message {
                            role: "assistant",
                            content: "That contradicts the earlier blanket AES-256 policy.",
                        },
                        Message {
                            role: "user",
                            content: "Yes, the CTO overrode the CISO on this. The updated policy is AES-128 for non-PII, AES-256 for PII.",
                        },
                    ],
                },
            ],
            questions: vec![
                BeamQuestion {
                    question: "What is the current encryption policy?",
                    ability: BeamAbility::ContradictionResolution,
                    must_contain: vec!["AES-128|non-PII", "AES-256|PII"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "How did the encryption policy change over time?",
                    ability: BeamAbility::EventOrdering,
                    must_contain: vec!["AES-256", "CTO|override|relax"],
                    expects_abstention: false,
                },
            ],
        },
        // --- 5. Preference following ---
        BeamScenario {
            name: "Evolving user preferences",
            sessions: vec![
                Session {
                    session_id: 1,
                    label: "2025-01-10",
                    messages: vec![Message {
                        role: "user",
                        content: "I prefer getting status updates via email. Send me a daily digest every morning at 9am.",
                    }],
                },
                Session {
                    session_id: 2,
                    label: "2025-02-05",
                    messages: vec![Message {
                        role: "user",
                        content: "Actually, email is too slow. Switch all my notifications to Slack. Use the #ops-alerts channel.",
                    }],
                },
                Session {
                    session_id: 3,
                    label: "2025-03-01",
                    messages: vec![Message {
                        role: "user",
                        content: "Keep Slack for urgent alerts only. For routine status updates, go back to email but make it a weekly digest on Mondays.",
                    }],
                },
            ],
            questions: vec![
                BeamQuestion {
                    question: "How should routine status updates be delivered?",
                    ability: BeamAbility::PreferenceFollowing,
                    must_contain: vec!["email|weekly", "Monday"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "How should urgent alerts be delivered?",
                    ability: BeamAbility::PreferenceFollowing,
                    must_contain: vec!["Slack"],
                    expects_abstention: false,
                },
            ],
        },
        // --- 6. Instruction following ---
        BeamScenario {
            name: "Sustained instruction adherence",
            sessions: vec![
                Session {
                    session_id: 1,
                    label: "",
                    messages: vec![
                        Message {
                            role: "user",
                            content: "Important rule: never suggest deploying to production on Fridays. Our change freeze runs Friday 3pm through Monday 8am.",
                        },
                        Message {
                            role: "assistant",
                            content: "Understood. No Friday deploys.",
                        },
                        Message {
                            role: "user",
                            content: "Also, always recommend staging environment testing before any production deployment. This is non-negotiable.",
                        },
                    ],
                },
                Session {
                    session_id: 2,
                    label: "",
                    messages: vec![Message {
                        role: "user",
                        content: "We have a critical bug fix for the payment system. It's Thursday evening and the fix is ready. When should we deploy?",
                    }],
                },
            ],
            questions: vec![
                BeamQuestion {
                    question: "When is the change freeze window?",
                    ability: BeamAbility::InstructionFollowing,
                    must_contain: vec!["Friday", "Monday"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "What must happen before any production deployment?",
                    ability: BeamAbility::InstructionFollowing,
                    must_contain: vec!["staging|test"],
                    expects_abstention: false,
                },
            ],
        },
        // --- 7. Entity tracking across sessions ---
        BeamScenario {
            name: "Entity tracking across sessions",
            sessions: vec![
                Session {
                    session_id: 1,
                    label: "2025-01-15",
                    messages: vec![
                        Message {
                            role: "user",
                            content: "Our client Nexus Corp has 3 active projects with us: Project Alpha (data pipeline), Project Beta (ML platform), and Project Gamma (dashboard).",
                        },
                        Message {
                            role: "assistant",
                            content: "Got it, 3 Nexus projects tracked.",
                        },
                        Message {
                            role: "user",
                            content: "Project Alpha's lead is Tom Chen. Budget: $500K. Deadline: March 30.",
                        },
                    ],
                },
                Session {
                    session_id: 2,
                    label: "2025-02-10",
                    messages: vec![
                        Message {
                            role: "user",
                            content: "Project Beta at Nexus Corp just got cancelled. Budget was reallocated to Project Alpha, which now has $750K.",
                        },
                        Message {
                            role: "assistant",
                            content: "Noted. Beta cancelled, Alpha budget increased.",
                        },
                        Message {
                            role: "user",
                            content: "Also, Tom Chen moved to Project Gamma. The new Alpha lead is Priya Sharma.",
                        },
                    ],
                },
                Session {
                    session_id: 3,
                    label: "2025-03-01",
                    messages: vec![Message {
                        role: "user",
                        content: "Project Alpha at Nexus Corp was delivered on time. Nexus is very happy. They want to expand Gamma's scope.",
                    }],
                },
            ],
            questions: vec![
                BeamQuestion {
                    question: "How many active projects does Nexus Corp currently have?",
                    ability: BeamAbility::InformationUpdate,
                    must_contain: vec!["two|2|Alpha|Gamma"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "Who is currently leading Project Alpha at Nexus?",
                    ability: BeamAbility::InformationUpdate,
                    must_contain: vec!["Priya|Sharma"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "What happened to Project Beta?",
                    ability: BeamAbility::InformationExtraction,
                    must_contain: vec!["cancel"],
                    expects_abstention: false,
                },
            ],
        },
        // --- 8. Abstention (unanswerable questions) ---
        BeamScenario {
            name: "Abstention on unknown information",
            sessions: vec![Session {
                session_id: 1,
                label: "",
                messages: vec![
                    Message {
                        role: "user",
                        content: "Our vendor for cloud hosting is AWS. We pay them about $45K/month.",
                    },
                    Message {
                        role: "assistant",
                        content: "Got it.",
                    },
                    Message {
                        role: "user",
                        content: "The DevOps team lead is Kai Nakamura. He manages 5 engineers.",
                    },
                ],
            }],
            questions: vec![
                BeamQuestion {
                    question: "What is the company's revenue?",
                    ability: BeamAbility::Abstention,
                    must_contain: vec![],
                    expects_abstention: true,
                },
                BeamQuestion {
                    question: "Who is the CEO?",
                    ability: BeamAbility::Abstention,
                    must_contain: vec![],
                    expects_abstention: true,
                },
                // Control: this one IS answerable
                BeamQuestion {
                    question: "Who leads the DevOps team?",
                    ability: BeamAbility::InformationExtraction,
                    must_contain: vec!["Kai|Nakamura"],
                    expects_abstention: false,
                },
            ],
        },
        // --- 9. Summarization ---
        BeamScenario {
            name: "Summarization across sessions",
            sessions: vec![
                Session {
                    session_id: 1,
                    label: "2025-01-05",
                    messages: vec![Message {
                        role: "user",
                        content: "We launched the customer portal v2 today. Key features: SSO integration, real-time usage dashboard, and self-service billing.",
                    }],
                },
                Session {
                    session_id: 2,
                    label: "2025-01-20",
                    messages: vec![Message {
                        role: "user",
                        content: "Portal v2 bug reports are coming in. The SSO flow breaks on Safari, and the billing page shows wrong currency for EU customers.",
                    }],
                },
                Session {
                    session_id: 3,
                    label: "2025-02-10",
                    messages: vec![Message {
                        role: "user",
                        content: "Fixed the Safari SSO bug. The billing currency issue is still open — waiting on the payments team to expose a locale API.",
                    }],
                },
                Session {
                    session_id: 4,
                    label: "2025-03-01",
                    messages: vec![Message {
                        role: "user",
                        content: "Portal v2 is now stable. All bugs resolved. Customer satisfaction score went from 3.2 to 4.5 after the fixes.",
                    }],
                },
            ],
            questions: vec![BeamQuestion {
                question: "Summarize the customer portal v2 journey from launch to stabilization.",
                ability: BeamAbility::Summarization,
                must_contain: vec!["SSO|portal", "bug|issue", "stable|fixed|resolved"],
                expects_abstention: false,
            }],
        },
        // --- 10. Token efficiency scenario (many messages, few relevant) ---
        BeamScenario {
            name: "Token efficiency under noise",
            sessions: vec![Session {
                session_id: 1,
                label: "",
                messages: vec![
                    Message {
                        role: "user",
                        content: "The marketing team wants to try TikTok ads for Q2. Budget: $15K.",
                    },
                    Message {
                        role: "user",
                        content: "Lunch order for the team meeting: 10 pizzas from Mario's.",
                    },
                    Message {
                        role: "user",
                        content: "Reminder: office party next Friday at 5pm.",
                    },
                    Message {
                        role: "user",
                        content: "CRITICAL: Our database backup job has been failing silently for 3 weeks. Last successful backup was February 28.",
                    },
                    Message {
                        role: "user",
                        content: "New coffee machine in the break room. It's a Breville Barista Express.",
                    },
                    Message {
                        role: "user",
                        content: "IT ticket #4521: Replace the projector in conference room B.",
                    },
                    Message {
                        role: "user",
                        content: "CRITICAL: The backup failure means we have no point-in-time recovery capability. RTO is currently infinite.",
                    },
                    Message {
                        role: "user",
                        content: "Company retreat is scheduled for June 15-17 in Lake Tahoe.",
                    },
                    Message {
                        role: "user",
                        content: "The backup runs on a cron job at 2am. The job's S3 credential expired and nobody rotated it.",
                    },
                    Message {
                        role: "user",
                        content: "Fantasy football draft is Thursday. Don't forget to set your lineups.",
                    },
                ],
            }],
            questions: vec![
                BeamQuestion {
                    question: "What is the critical infrastructure issue and what caused it?",
                    ability: BeamAbility::InformationExtraction,
                    must_contain: vec!["backup", "S3|credential|expired"],
                    expects_abstention: false,
                },
                BeamQuestion {
                    question: "When was the last successful database backup?",
                    ability: BeamAbility::InformationExtraction,
                    must_contain: vec!["February 28|Feb 28"],
                    expects_abstention: false,
                },
            ],
        },
    ]
}

// ---------------------------------------------------------------------------
// Main benchmark runner
// ---------------------------------------------------------------------------

async fn run_beam_benchmark(scenarios: &[BeamScenario], use_real_llm: bool) -> BenchmarkReport {
    let mut report = BenchmarkReport::default();

    for scenario in scenarios {
        println!("\n{}", "=".repeat(70));
        println!("BEAM SCENARIO: {}", scenario.name);
        println!("{}", "=".repeat(70));

        let karta = if use_real_llm {
            create_karta_real(scenario.name).await
        } else {
            create_karta_mock(scenario.name).await
        };

        // --- Ingest phase (session-aware for episode creation) ---
        let ingest_start = Instant::now();
        ingest_sessions(&karta, &scenario.sessions).await;

        let ingest_ms = ingest_start.elapsed().as_millis();
        let note_count = karta.note_count().await.unwrap();
        println!(
            "  Ingested {} notes in {}ms ({:.1}ms/note)",
            note_count,
            ingest_ms,
            ingest_ms as f64 / note_count.max(1) as f64
        );

        report.total_ingest_ms += ingest_ms;
        report.total_notes += note_count;

        // --- Query phase ---
        let mut metrics = ScenarioMetrics {
            name: scenario.name.to_string(),
            notes_stored: note_count,
            ingest_ms,
            ..Default::default()
        };

        // Track per-ability scores
        let mut ability_map: std::collections::HashMap<BeamAbility, (usize, usize)> =
            std::collections::HashMap::new();

        for q in &scenario.questions {
            metrics.questions_total += 1;

            let query_start = Instant::now();
            let answer = karta.ask(q.question, 5).await.unwrap().answer;
            let query_ms = query_start.elapsed().as_millis();
            metrics.query_latencies_ms.push(query_ms);
            report.total_query_ms += query_ms;

            println!("\n  [{}] Q: {}", q.ability.as_str(), q.question);
            println!("  A ({}ms): {}...", query_ms, safe_truncate(&answer, 200));

            let mut question_passed = true;

            if q.expects_abstention {
                // For abstention, check that the answer does NOT confidently
                // assert facts that aren't in the memory.
                // With MockLlm this is hard to test precisely, so we just
                // track it as a pass if the answer is short or hedging.
                metrics.checks_total += 1;
                let entry = ability_map.entry(q.ability).or_insert((0, 0));
                entry.1 += 1;

                let lower = answer.to_lowercase();
                let hedges = lower.contains("not")
                    || lower.contains("no information")
                    || lower.contains("don't")
                    || lower.contains("unknown")
                    || lower.contains("unclear")
                    || answer.len() < 100;
                if hedges {
                    metrics.checks_passed += 1;
                    entry.0 += 1;
                    println!("    [PASS] Abstention detected");
                } else {
                    question_passed = false;
                    println!("    [FAIL] Expected abstention but got confident answer");
                }
            } else {
                for pattern in &q.must_contain {
                    metrics.checks_total += 1;
                    let entry = ability_map.entry(q.ability).or_insert((0, 0));
                    entry.1 += 1;

                    if check_must_contain(&answer, pattern) {
                        metrics.checks_passed += 1;
                        entry.0 += 1;
                        println!("    [PASS] Contains: {}", pattern);
                    } else {
                        question_passed = false;
                        println!("    [FAIL] Missing: {}", pattern);
                    }
                }
            }

            if question_passed {
                metrics.questions_passed += 1;
            }
        }

        // Collect ability scores
        for (ability, (passed, total)) in &ability_map {
            metrics.ability_scores.push((*ability, *passed, *total));
        }

        report.scenario_metrics.push(metrics);
    }

    report
}

fn print_report(report: &BenchmarkReport) {
    println!("\n{}", "=".repeat(70));
    println!("BEAM BENCHMARK REPORT");
    println!("{}", "=".repeat(70));

    let mut total_q = 0;
    let mut passed_q = 0;
    let mut total_c = 0;
    let mut passed_c = 0;

    for m in &report.scenario_metrics {
        let status = if m.questions_passed == m.questions_total {
            "PASS"
        } else {
            "FAIL"
        };
        let avg_query_ms = if m.query_latencies_ms.is_empty() {
            0
        } else {
            m.query_latencies_ms.iter().sum::<u128>() / m.query_latencies_ms.len() as u128
        };

        println!(
            "  [{}] {} — {}/{} questions, {}/{} checks, {}notes, ingest={}ms, avg_query={}ms",
            status,
            m.name,
            m.questions_passed,
            m.questions_total,
            m.checks_passed,
            m.checks_total,
            m.notes_stored,
            m.ingest_ms,
            avg_query_ms,
        );

        total_q += m.questions_total;
        passed_q += m.questions_passed;
        total_c += m.checks_total;
        passed_c += m.checks_passed;
    }

    println!();
    println!("  TOTALS:");
    println!("    Scenarios:  {}", report.scenario_metrics.len());
    println!("    Questions:  {}/{} passed", passed_q, total_q);
    println!("    Checks:     {}/{} passed", passed_c, total_c);
    println!("    Notes:      {}", report.total_notes);
    println!("    Ingest:     {}ms total", report.total_ingest_ms);
    println!("    Queries:    {}ms total", report.total_query_ms);

    if total_c > 0 {
        let pass_rate = passed_c as f64 / total_c as f64;
        println!("    Pass rate:  {:.1}%", pass_rate * 100.0);
    }

    // Per-ability breakdown
    println!();
    println!("  PER-ABILITY BREAKDOWN:");
    let mut ability_agg: std::collections::HashMap<&str, (usize, usize)> =
        std::collections::HashMap::new();
    for m in &report.scenario_metrics {
        for (ability, passed, total) in &m.ability_scores {
            let entry = ability_agg.entry(ability.as_str()).or_insert((0, 0));
            entry.0 += passed;
            entry.1 += total;
        }
    }
    let mut abilities: Vec<_> = ability_agg.into_iter().collect();
    abilities.sort_by_key(|(name, _)| *name);
    for (name, (passed, total)) in &abilities {
        let rate = if *total > 0 {
            *passed as f64 / *total as f64 * 100.0
        } else {
            0.0
        };
        println!("    {:30} {}/{} ({:.0}%)", name, passed, total, rate);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Mock sanity check — quick build validation that the harness works.
/// This is the ONLY mock test. All real evals use the real LLM.
///
/// Run: cargo test --test bench_beam beam_mock_sanity -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn beam_mock_sanity() {
    let scenarios = beam_scenarios();
    let report = run_beam_benchmark(&scenarios, false).await;
    print_report(&report);

    let total_checks: usize = report.scenario_metrics.iter().map(|m| m.checks_total).sum();
    let passed_checks: usize = report
        .scenario_metrics
        .iter()
        .map(|m| m.checks_passed)
        .sum();
    if total_checks > 0 {
        let rate = passed_checks as f64 / total_checks as f64;
        assert!(
            rate >= 0.40,
            "Mock sanity pass rate {:.1}% below 40%",
            rate * 100.0
        );
    }
}

/// BEAM benchmark with real LLM — the main evaluation.
///
/// Run: cargo test --test bench_beam beam_real -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn beam_real() {
    let scenarios = beam_scenarios();
    let report = run_beam_benchmark(&scenarios, true).await;
    print_report(&report);

    let total_checks: usize = report.scenario_metrics.iter().map(|m| m.checks_total).sum();
    let passed_checks: usize = report
        .scenario_metrics
        .iter()
        .map(|m| m.checks_passed)
        .sum();
    if total_checks > 0 {
        let rate = passed_checks as f64 / total_checks as f64;
        println!("\n  REAL LLM pass rate: {:.1}%", rate * 100.0);
        assert!(
            rate >= 0.70,
            "Real LLM pass rate {:.1}% below 70%",
            rate * 100.0
        );
    }
}

/// BEAM + dreaming with real LLM — tests whether dream engine improves
/// cross-session reasoning via background inference.
///
/// Run: cargo test --test bench_beam beam_real_with_dreaming -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn beam_real_with_dreaming() {
    let all = beam_scenarios();
    let dream_relevant: Vec<&BeamScenario> = all
        .iter()
        .filter(|s| {
            s.name.contains("Cross-session")
                || s.name.contains("Contradiction")
                || s.name.contains("Entity tracking")
        })
        .collect();

    println!(
        "Running {} dream-relevant scenarios with real LLM",
        dream_relevant.len()
    );

    for scenario in &dream_relevant {
        println!("\n{}", "=".repeat(70));
        println!("BEAM+DREAM: {}", scenario.name);
        println!("{}", "=".repeat(70));

        let karta = create_karta_real(&format!("dream-{}", scenario.name)).await;

        // Ingest (session-aware for episode creation)
        ingest_sessions(&karta, &scenario.sessions).await;

        let pre_dream_count = karta.note_count().await.unwrap();
        println!("  Notes before dreaming: {}", pre_dream_count);

        // Dream
        let dream_start = Instant::now();
        let dream_run = karta.run_dreaming("benchmark", "beam").await.unwrap();
        let dream_ms = dream_start.elapsed().as_millis();

        let post_dream_count = karta.note_count().await.unwrap();
        println!("  Notes after dreaming:  {}", post_dream_count);
        println!(
            "  Dreams: {} attempted, {} written ({}ms)",
            dream_run.dreams_attempted, dream_run.dreams_written, dream_ms
        );

        for d in &dream_run.dreams {
            let icon = if d.would_write { "W" } else { "." };
            println!(
                "    [{}] {} conf={:.2}: {}",
                icon,
                d.dream_type.as_str(),
                d.confidence,
                safe_truncate(&d.dream_content, 120)
            );
        }

        // Query (post-dream)
        for q in &scenario.questions {
            let answer = karta.ask(q.question, 5).await.unwrap().answer;
            println!("\n  [{}] Q: {}", q.ability.as_str(), q.question);
            println!("  A: {}...", safe_truncate(&answer, 300));

            for pattern in &q.must_contain {
                if check_must_contain(&answer, pattern) {
                    println!("    [PASS] {}", pattern);
                } else {
                    println!("    [FAIL] {}", pattern);
                }
            }
        }
    }
}

/// Placeholder for loading and running the real BEAM dataset.
///
/// The BEAM dataset is available at:
///   - HuggingFace: https://huggingface.co/datasets/Mohammadta/BEAM
///   - GitHub: https://github.com/mohammadtavakoli78/BEAM
///
/// TODO:
///   1. Download BEAM dataset (JSON format with conversations + questions)
///   2. Parse into Session/BeamQuestion structs
///   3. Run at each scale: 100K (128K tokens), 500K, 1M
///   4. Compare against published baselines
///
/// BEAM dataset format (from the paper):
///   - Each entry has: conversation (list of turns), questions (list of QA pairs)
///   - Questions include: question text, reference answer, ability category
///   - Evaluation uses LLM-as-judge (GPT-4) for answer correctness scoring
///
/// LOCOMO dataset format (from snap-research/locomo):
///   - 10 conversations, each ~300 turns across ~35 sessions
///   - Questions categorized: single-hop, multi-hop, temporal, commonsense, adversarial
///   - Evaluation: exact match, F1, BERTScore, ROUGE-L
///
/// Run: cargo test --test bench_beam -- --ignored --nocapture beam_real_dataset
#[tokio::test]
#[ignore]
async fn beam_real_dataset() {
    // TODO: Implement real dataset loading
    //
    // Sketch of the implementation:
    //
    // 1. Check for dataset file:
    //    let dataset_path = std::env::var("BEAM_DATASET_PATH")
    //        .unwrap_or_else(|_| "data/beam_100k.json".to_string());
    //
    // 2. Parse JSON structure:
    //    #[derive(Deserialize)]
    //    struct BeamDataset {
    //        conversations: Vec<BeamConversation>,
    //    }
    //    #[derive(Deserialize)]
    //    struct BeamConversation {
    //        id: String,
    //        turns: Vec<BeamTurn>,
    //        questions: Vec<BeamDatasetQuestion>,
    //    }
    //    #[derive(Deserialize)]
    //    struct BeamTurn {
    //        role: String,
    //        content: String,
    //        turn_id: usize,
    //    }
    //    #[derive(Deserialize)]
    //    struct BeamDatasetQuestion {
    //        question: String,
    //        reference_answer: String,
    //        ability: String,  // one of the 10 BEAM abilities
    //    }
    //
    // 3. For each conversation:
    //    a. Ingest all user turns as notes
    //    b. Optionally run dreaming
    //    c. Ask each question
    //    d. Score answer against reference (LLM-as-judge or must_contain)
    //
    // 4. Report per-ability accuracy and overall score

    eprintln!(
        "BEAM real dataset test is a placeholder. Set BEAM_DATASET_PATH to run.\n\
         Download from: https://huggingface.co/datasets/Mohammadta/BEAM"
    );
}

/// Placeholder for LOCOMO benchmark.
///
/// LOCOMO (Maharana et al., ACL 2024) evaluates long-term conversational memory:
///   - 10 conversations, ~300 turns each, ~35 sessions
///   - 5 question types: single-hop, multi-hop, temporal, commonsense, adversarial
///   - Metrics: exact match, F1, BERTScore, ROUGE-L
///   - Dataset: https://github.com/snap-research/locomo
///
/// Run: cargo test --test bench_beam -- --ignored --nocapture locomo_real_dataset
#[tokio::test]
#[ignore]
async fn locomo_real_dataset() {
    eprintln!(
        "LOCOMO dataset test is a placeholder.\n\
         Download from: https://github.com/snap-research/locomo"
    );
}
