//! Structural validation for the Droid and Claude Code hook template examples.
//!
//! These tests ensure the `.example` files ship with the correct event wiring,
//! handler types, and localhost URLs, and that the helper scripts use the
//! configured capture port rather than a hardcoded default.

use std::collections::HashSet;
use std::path::PathBuf;

use serde_json::Value;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_json(path: &str) -> Value {
    let full = manifest_dir().join(path);
    let text = std::fs::read_to_string(&full)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", full.display(), e));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("{} is not valid JSON: {}", full.display(), e))
}

fn read_file(path: &str) -> String {
    let full = manifest_dir().join(path);
    std::fs::read_to_string(&full)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", full.display(), e))
}

#[test]
fn droid_hooks_example_has_all_wired_events_and_command_type() {
    let value = read_json(".factory/hooks.example.json");
    let hooks = value
        .get("hooks")
        .and_then(|v| v.as_object())
        .expect("top-level 'hooks' object missing");

    let expected: HashSet<&str> = [
        "SessionStart",
        "UserPromptSubmit",
        "PostToolUse",
        "Stop",
        "SubagentStop",
        "SessionEnd",
        "PreCompact",
    ]
    .iter()
    .cloned()
    .collect();
    let actual: HashSet<&str> = hooks.keys().map(|s| s.as_str()).collect();
    assert_eq!(
        expected, actual,
        "Droid hooks.example.json is missing or has extra events"
    );

    for (event, groups) in hooks {
        let groups = groups
            .as_array()
            .unwrap_or_else(|| panic!("event {} should contain an array of matcher groups", event));
        assert!(!groups.is_empty(), "event {} has no matcher groups", event);

        for (i, group) in groups.iter().enumerate() {
            let handlers = group
                .get("hooks")
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| {
                    panic!("event {} group {} is missing the 'hooks' array", event, i)
                });
            assert!(
                !handlers.is_empty(),
                "event {} group {} has no handlers",
                event,
                i
            );

            for (j, handler) in handlers.iter().enumerate() {
                let ty = handler
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| {
                        panic!("event {} group {} handler {} missing 'type'", event, i, j)
                    });
                assert_eq!(
                    ty, "command",
                    "event {} group {} handler {} must be type 'command'",
                    event, i, j
                );
            }
        }
    }
}

#[test]
fn droid_hooks_example_references_correct_scripts() {
    let value = read_json(".factory/hooks.example.json");
    let hooks = value.get("hooks").unwrap().as_object().unwrap();

    let orient = hooks["SessionStart"][0]["hooks"][0]["command"]
        .as_str()
        .expect("SessionStart command missing");
    assert!(
        orient.contains("karta-orient.sh"),
        "SessionStart should call karta-orient.sh: {}",
        orient
    );

    let capture_events = [
        ("UserPromptSubmit", "user_prompt"),
        ("PostToolUse", "observation"),
        ("Stop", "turn_summary"),
        ("SubagentStop", "subagent_result"),
        ("SessionEnd", "session_end"),
        ("PreCompact", "pre_compact"),
    ];
    for (event, expected_arg) in capture_events.iter() {
        let command = hooks[*event][0]["hooks"][0]["command"]
            .as_str()
            .unwrap_or_else(|| panic!("{} command missing", event));
        assert!(
            command.contains("karta-capture.sh"),
            "{} should call karta-capture.sh: {}",
            event,
            command
        );
        assert!(
            command.contains(expected_arg),
            "{} command should pass the '{}' argument: {}",
            event,
            expected_arg,
            command
        );
    }
}

#[test]
fn droid_post_tool_use_matcher_is_file_edit_family() {
    let value = read_json(".factory/hooks.example.json");
    let post = value["hooks"]["PostToolUse"][0]
        .as_object()
        .expect("PostToolUse matcher group missing");
    let matcher = post.get("matcher").and_then(|v| v.as_str()).unwrap_or("*");
    assert!(
        matcher.contains("Create") || matcher.contains("Edit") || matcher.contains("ApplyPatch"),
        "PostToolUse matcher should target file edits, got: {}",
        matcher
    );
}

#[test]
fn droid_orient_script_posts_to_orient_and_uses_configured_port() {
    let script = read_file(".factory/hooks/karta-orient.sh");
    assert!(
        script.contains("/orient"),
        "karta-orient.sh must POST to /orient"
    );
    assert!(
        script.contains("${KARTA_CAPTURE_PORT:-3137}") || script.contains("$KARTA_CAPTURE_PORT"),
        "karta-orient.sh must use the configured KARTA_CAPTURE_PORT"
    );
    assert!(
        script.contains("127.0.0.1"),
        "karta-orient.sh must target 127.0.0.1"
    );
    assert!(script.contains("curl"), "karta-orient.sh must use curl");
}

#[test]
fn droid_capture_script_posts_to_capture_and_maps_events() {
    let script = read_file(".factory/hooks/karta-capture.sh");
    assert!(
        script.contains("/capture"),
        "karta-capture.sh must POST to /capture"
    );
    assert!(
        script.contains("${KARTA_CAPTURE_PORT:-3137}") || script.contains("$KARTA_CAPTURE_PORT"),
        "karta-capture.sh must use the configured KARTA_CAPTURE_PORT"
    );
    assert!(
        script.contains("127.0.0.1"),
        "karta-capture.sh must target 127.0.0.1"
    );

    for event in &[
        "user_prompt",
        "observation",
        "turn_summary",
        "subagent_result",
        "session_end",
        "pre_compact",
    ] {
        assert!(
            script.contains(event),
            "karta-capture.sh must handle event type {}",
            event
        );
    }
}

#[test]
fn claude_settings_example_has_all_wired_events_and_http_type() {
    let value = read_json(".claude/settings.example.json");
    let hooks = value
        .get("hooks")
        .and_then(|v| v.as_object())
        .expect("top-level 'hooks' object missing");

    let expected: HashSet<&str> = [
        "SessionStart",
        "UserPromptSubmit",
        "PostToolUse",
        "Stop",
        "SubagentStop",
        "SessionEnd",
        "PreCompact",
    ]
    .iter()
    .cloned()
    .collect();
    let actual: HashSet<&str> = hooks.keys().map(|s| s.as_str()).collect();
    assert_eq!(
        expected, actual,
        "Claude settings.example.json is missing or has extra events"
    );

    for (event, groups) in hooks {
        let groups = groups
            .as_array()
            .unwrap_or_else(|| panic!("event {} should contain an array of matcher groups", event));
        assert!(!groups.is_empty(), "event {} has no matcher groups", event);

        for (i, group) in groups.iter().enumerate() {
            let handlers = group
                .get("hooks")
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| {
                    panic!("event {} group {} is missing the 'hooks' array", event, i)
                });
            assert!(
                !handlers.is_empty(),
                "event {} group {} has no handlers",
                event,
                i
            );

            for (j, handler) in handlers.iter().enumerate() {
                let ty = handler
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| {
                        panic!("event {} group {} handler {} missing 'type'", event, i, j)
                    });
                assert_eq!(
                    ty, "http",
                    "event {} group {} handler {} must be type 'http'",
                    event, i, j
                );

                let url = handler
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| {
                        panic!("event {} group {} handler {} missing 'url'", event, i, j)
                    });
                assert!(
                    url.contains("127.0.0.1:3137"),
                    "event {} URL must point to 127.0.0.1:3137: {}",
                    event,
                    url
                );

                if event == "SessionStart" {
                    assert!(
                        url.ends_with("/orient"),
                        "SessionStart URL must end with /orient: {}",
                        url
                    );
                } else {
                    assert!(
                        url.ends_with("/capture"),
                        "{} URL must end with /capture: {}",
                        event,
                        url
                    );
                }
            }
        }
    }
}

#[test]
fn hook_templates_are_examples_not_live_configs() {
    let crate_root = manifest_dir();
    let live_droid = crate_root.join(".factory/hooks.json");
    let live_claude = crate_root.join(".claude/settings.json");
    assert!(
        !live_droid.exists(),
        "live Droid config {} must not be committed",
        live_droid.display()
    );
    assert!(
        !live_claude.exists(),
        "live Claude Code config {} must not be committed",
        live_claude.display()
    );

    let readme = read_file("README.md");
    assert!(
        readme.contains(".example"),
        "README must mention the .example template files"
    );
    assert!(
        readme.contains("copy") || readme.contains("install") || readme.contains("manual"),
        "README must explain that the templates require manual installation"
    );
}
