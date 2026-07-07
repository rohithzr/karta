use chrono::{TimeZone, Utc};
use karta_core::llm::MockLlmProvider;
use karta_core::read::resolve_llm::{LlmResolver, ResolverContext};
use karta_core::read::temporal::ConfidenceBand;
use std::sync::Arc;

#[tokio::test]
async fn f7_t19_tier2_resolves_last_spring_via_llm() {
    let llm = Arc::new(MockLlmProvider::new());
    let resolver = LlmResolver::new(llm);
    let ref_time = Utc.with_ymd_and_hms(2024, 4, 22, 0, 0, 0).unwrap();
    let ctx = ResolverContext { recent_turns: vec![] };

    let result = resolver
        .resolve("what happened last spring", ref_time, &ctx)
        .await;
    let (iv, band) = result.expect("mock LLM should return a valid resolution for 'last spring'");
    assert_eq!(band, ConfidenceBand::Vague);
    assert_eq!(iv.start, Utc.with_ymd_and_hms(2024, 3, 1, 0, 0, 0).unwrap());
    assert_eq!(iv.end, Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap());
}

#[tokio::test]
async fn f7_t19_tier2_returns_none_when_no_anchor() {
    let llm = Arc::new(MockLlmProvider::new());
    let resolver = LlmResolver::new(llm);
    let ref_time = Utc.with_ymd_and_hms(2024, 4, 22, 0, 0, 0).unwrap();
    let ctx = ResolverContext { recent_turns: vec![] };

    // Mock returns null bounds for any query without "last spring".
    let result = resolver.resolve("something vague", ref_time, &ctx).await;
    assert!(result.is_none());
}
