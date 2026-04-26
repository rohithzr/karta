//! Inspection test — ingest a few BEAM notes and examine the graph state.
//! Used to debug retrieval quality, linking, evolution, and temporal context.
//!
//! Run: cargo test --test inspect -- --ignored --nocapture

use karta_core::Karta;
use karta_core::config::KartaConfig;

async fn create_karta() -> Karta {
    let data_dir = "/tmp/karta-inspect";
    let _ = std::fs::remove_dir_all(data_dir);

    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir.to_string();
    // Enable reranker for abstention testing
    config.reranker.enabled = true;
    config.reranker.abstention_threshold = 0.1; // Jina raw scores: <0.1 = abstain
    config.reranker.max_rerank = 5;

    Karta::with_defaults(config)
        .await
        .expect("Failed to create Karta — check .env credentials")
}

#[tokio::test]
#[ignore]
async fn inspect_ingestion() {
    let karta = create_karta().await;

    // First 15 user messages from BEAM Conv 1 (coding/Flask budget tracker)
    let notes = vec![
        "[March-15-2024] I'm working on a project with a Time Anchor of March 15, 2024, and I need to plan my tasks accordingly, can you help me create a schedule to ensure I meet my deadlines by then?",
        "[March-15-2024] You know what, let's start from scratch. I'm building a personal budget tracker app. Here's the tech stack: Flask 2.3.1, Jinja2, Bootstrap 5.3 for the frontend, SQLite 3.39 database with SQLAlchemy ORM, running on local dev server port 5000, and Python 3.11.",
        "[March-15-2024] Let me organize my project with a monolithic architecture using Flask Blueprints for auth, transactions, and analytics modules. Let's go with the MVC pattern.",
        "[March-15-2024] I need a database schema for my budget tracker. Here's what I need: a users table with basic fields plus a role column for future RBAC, a transactions table for income/expenses with category and notes columns, and a budgets table linked to users.",
        "[March-15-2024] I want to set up my Flask project structure: app/__init__.py as the factory, separate blueprints in app/auth, app/transactions, and app/analytics, a config.py with development/production configs, and a requirements.txt.",
        "[March-15-2024] Let me be specific about my MVP deadline. My first sprint ends March 29, and I need user auth (registration, login, logout with hashed passwords), basic transaction CRUD, and a simple dashboard showing monthly totals.",
        "[March-15-2024] For authentication, I want to use Flask-Login for session management, Werkzeug for password hashing (generate_password_hash and check_password_hash), and I'll add a 'remember me' checkbox on the login form.",
        "[March-15-2024] I've never actually integrated Flask-Login into this project. I've been handling sessions manually with Flask's built-in session. But Flask-Login v0.6.3 is compatible with our Flask version.",
        "[March-15-2024] Wait, actually I have been writing Flask routes and handling HTTP requests. I defined @app.route('/') with methods=['GET', 'POST'] for the homepage, and I've been implementing the registration and login views.",
        "[March-15-2024] Now for transaction management: I need CRUD operations for income and expenses. The form should have fields for amount, category (dropdown), date, description, and type (income/expense). Add validation for positive amounts and required fields.",
        "[March-15-2024] For the dashboard analytics, I want: monthly income vs expenses bar chart, spending by category pie chart, running balance line graph. Use Chart.js for visualizations.",
        "[January-15-2024] My transaction management features need to be done by January 15, 2024. After that, I'll focus on analytics and the final deployment deadline is March 15, 2024.",
        "[March-15-2024] Let me update my sprint schedule. Sprint 1 (now extended to March 31 for extra testing): auth + transactions. Sprint 2 (April 1-19): analytics + dashboard + charts. Sprint 3 (April 20 - May 5): testing, deployment, documentation.",
        "[March-15-2024] Security review: I'm worried about XSS in the transaction notes field, CSRF protection (need Flask-WTF), SQL injection (using SQLAlchemy parameterized queries), and session fixation after login.",
        "[March-15-2024] I want to add input sanitization using bleach library for the notes field, implement CSRF tokens on all forms, add rate limiting on login attempts (max 5 per minute), and set secure cookie flags.",
    ];

    println!("=== INGESTING {} NOTES ===\n", notes.len());

    for (i, note) in notes.iter().enumerate() {
        let result = karta.add_note(note).await.unwrap();
        println!(
            "Note {}: id={}, links={}, keywords=[{}]",
            i + 1,
            &result.id[..8],
            result.links.len(),
            result.keywords.join(", ")
        );
        println!(
            "  Context: {}",
            &result.context[..result.context.len().min(120)]
        );
        println!();
    }

    // Inspect the graph
    println!("\n=== GRAPH STATE ===\n");

    let all_notes = karta.get_all_notes().await.unwrap();
    println!("Total notes: {}", all_notes.len());

    let mut total_links = 0;
    for note in &all_notes {
        let links = karta.get_links(&note.id).await.unwrap();
        total_links += links.len();
        if links.len() > 2 {
            println!(
                "  Hub note ({}): {} links — {}",
                &note.id[..8],
                links.len(),
                &note.content[..note.content.len().min(60)]
            );
        }
    }
    println!(
        "Total links: {} (avg {:.1}/note)",
        total_links,
        total_links as f64 / all_notes.len() as f64
    );

    // Test retrieval for the abstention question
    println!("\n=== RETRIEVAL TESTS ===\n");

    let test_queries = vec![
        (
            "ABSTENTION TEST",
            "Can you tell me about my background and previous development projects?",
        ),
        ("TEMPORAL TEST", "When does my first sprint end?"),
        (
            "CONTRADICTION TEST",
            "Have I integrated Flask-Login for session management?",
        ),
        (
            "SUMMARIZATION TEST",
            "Give me a comprehensive summary of my project progress",
        ),
        (
            "INSTRUCTION TEST",
            "Could you show me how to implement a login feature?",
        ),
    ];

    for (label, query) in &test_queries {
        println!("--- {} ---", label);
        println!("Q: {}", query);

        // Raw search results with scores
        let results = karta.search(query, 5).await.unwrap();
        println!("Top-5 search results:");
        for (i, r) in results.iter().enumerate() {
            println!(
                "  [{}] score={:.3} — {}",
                i + 1,
                r.score,
                &r.note.content[..r.note.content.len().min(80)]
            );
        }

        // Full ask
        let result = karta.ask(query, 5).await.unwrap();
        let preview: String = result.answer.chars().take(300).collect();
        println!("Answer: {}...", preview);
        println!();
    }
}
