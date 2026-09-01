//! Scrub caller-owned secrets out of a request before it reaches the provider.
//!
//! One four-rule policy — a known literal key, an `sk-` prefixed token, an
//! address, and a card-shaped digit run — is compiled into a `RedactionPlugin`
//! and registered on the client. A prompt deliberately seeded with all four
//! shapes is then dispatched: the hook rewrites it in `on_request`, so what
//! crosses the wire carries `[REDACTED:{kind}]` tokens instead. The same
//! matcher is also called directly through `scrub_str`, on a clean and on a
//! dirty string, to show the borrowed-versus-owned contract.
//!
//! Every secret in this file is fake.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
//!   (`llama-server`) on port 1234 with the demo model loaded.
//!
//! # Run
//!
//! ```sh
//! cargo run --example redaction --features provider-llamacpp,plugin-redaction
//! ```
//!
//! # Configuration
//!
//! Both values default to a local llama.cpp server; override them to target
//! any OpenAI-compatible server:
//!
//! - `CUCA_BASE_URL`: server base URL, defaults to `http://127.0.0.1:1234/v1`.
//! - `CUCA_MODEL`: upstream model id, defaults to `google/gemma-4-e4b`.
//!
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example redaction --features provider-llamacpp,plugin-redaction`
//!
//! # Output
//!
//! The policy prints first, then the two direct `scrub_str` calls, then the
//! before/after prompt, the reply, and the counters. The shape, with the
//! numbers one run produced:
//!
//! ```text
//! Policy: 4 rules, match cap 1024 matches per string
//!   rule 0  literal     deploy-key exactly 19 bytes
//!   rule 1  prefixed    api-key    "sk-" + 8..=48 token bytes
//!   rule 2  email-like  email      local@domain.tld
//!   rule 3  digit-run   card       13..=19 digits, `-`/` ` separated
//!
//! Calling scrub_str directly, before anything enters a request
//!   clean input  0 redactions, Cow::Borrowed (nothing allocated)
//!     build finished in 4.2s with 0 warnings
//!   dirty input  2 redactions, Cow::Owned (one rewritten String)
//!     cat .env -> API_TOKEN=[REDACTED:api-key] and OWNER=[REDACTED:email]
//!
//! Dispatching a seeded prompt through the registered hook
//!   before: Rotate the runner: key DEPLOY-KEY-a1b2c3d4, provider token sk-live-51H9x0AbCdEfGh, notify ops@example.com, billing card 4111 1111 1111 1111. Reply with exactly: understood
//!   wire:   Rotate the runner: key [REDACTED:deploy-key], provider token [REDACTED:api-key], notify [REDACTED:email], billing card [REDACTED:card]. Reply with exactly: understood
//!   reply:  understood
//!
//! Counters after the turn
//!   last_request_redactions  4
//!   total_redactions         4
//!   last_redaction_event     kind=card field=message_text count=4
//! ```
//!
//! The `wire:` line is the request as the hook left it, and it is what the
//! provider received. `last_redaction_event` names the *class* of the last
//! value replaced in that field and how many were replaced there — never the
//! matched bytes, which is the whole point. The reply text depends on the
//! server and the model; the counters do not.
//!
//! With no server on the base URL, everything up to and including the `wire:`
//! line still prints — none of it needs a provider — and the program then names
//! the unreachable address, says how to fix it, and exits successfully.
//!
//! # Why an outbound hook and not an inbound filter?
//!
//! The leak this plugin exists to stop is caller-side data crossing the wire,
//! so it implements `on_request` and leaves `on_stream_chunk` at the trait
//! default. An inbound block was produced by the provider, which has already
//! seen whatever it emitted, and a per-chunk matcher would silently miss a
//! secret straddling two streamed blocks. Assistant turns re-enter the
//! conversation as `messages` on the next request, where the same hook scrubs
//! them; scrubbing is idempotent, because a `Literal` value containing
//! `[REDACTED:` is rejected at construction, so a replacement token cannot
//! re-match. The plugin also requires no registration position: register it
//! before a session store and the trajectory holds scrubbed content, after and
//! the store holds the raw value. Both are legal; the caller chooses.

use std::borrow::Cow;
use std::io::{Write, stdout};
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, RedactionConfig, RedactionPlugin, RedactionRule, UnifiedRequest};
use tokio_stream::StreamExt;

/// A fake deployment key: the kind of value a caller already knows verbatim and
/// can therefore match literally.
const DEPLOY_KEY: &str = "DEPLOY-KEY-a1b2c3d4";

/// The seeded prompt, carrying one of each shape the policy looks for.
const SEEDED_PROMPT: &str = "Rotate the runner: key DEPLOY-KEY-a1b2c3d4, provider token \
     sk-live-51H9x0AbCdEfGh, notify ops@example.com, billing card 4111 1111 1111 1111. \
     Reply with exactly: understood";

/// A tool-output line with nothing to redact.
const CLEAN_SAMPLE: &str = "build finished in 4.2s with 0 warnings";

/// A tool-output line carrying two of the four shapes.
const DIRTY_SAMPLE: &str = "cat .env -> API_TOKEN=sk-live-9f8e7d6c5b4a and OWNER=root@example.org";

/// One policy line: the variant, its `kind`, and the shape it looks for.
///
/// A `Literal` is described by byte length rather than by value: nothing in this
/// plugin's own reporting path ever prints a secret.
fn describe(rule: &RedactionRule) -> String {
    match rule {
        RedactionRule::Literal { kind, value } => {
            format!("literal     {kind:<10} exactly {} bytes", value.len())
        }
        RedactionRule::Prefixed {
            kind,
            prefix,
            min_len,
            max_len,
        } => format!("prefixed    {kind:<10} {prefix:?} + {min_len}..={max_len} token bytes"),
        RedactionRule::EmailLike { kind } => {
            format!("email-like  {kind:<10} local@domain.tld")
        }
        RedactionRule::DigitRun {
            kind,
            min_digits,
            max_digits,
        } => {
            format!("digit-run   {kind:<10} {min_digits}..={max_digits} digits, `-`/` ` separated")
        }
    }
}

/// Scrub `sample` through the public matcher and report both the result and the
/// allocation it did or did not make.
fn show_scrub(
    label: &str,
    plugin: &RedactionPlugin,
    sample: &str,
) -> Result<(), cuca::PluginError> {
    let redacted = plugin.scrub_str(sample)?;
    // The `Cow` discriminant *is* the allocation story: `Borrowed` means the
    // clean path never reached the allocator.
    let allocation = match &redacted.text {
        Cow::Borrowed(_) => "Cow::Borrowed (nothing allocated)",
        Cow::Owned(_) => "Cow::Owned (one rewritten String)",
    };
    println!("  {label:<12} {} redactions, {allocation}", redacted.count);
    println!("    {}", redacted.text);
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // Stage 1: the policy. Every rule is caller-authored — there is no built-in
    // rule set and no "looks like a secret" heuristic, so a false positive is
    // always a policy someone wrote and can see. The bounds (rule count,
    // pattern length, `kind` slug, match cap) are validated here, so an
    // unusable policy fails before the plugin exists rather than scrubbing
    // nothing at runtime.
    let rules = vec![
        RedactionRule::Literal {
            kind: "deploy-key".to_string(),
            value: DEPLOY_KEY.to_string(),
        },
        RedactionRule::Prefixed {
            kind: "api-key".to_string(),
            prefix: "sk-".to_string(),
            min_len: 8,
            max_len: 48,
        },
        RedactionRule::EmailLike {
            kind: "email".to_string(),
        },
        // 13..=19 keeps this off order numbers and years: `min_digits` is the
        // knob that decides whether the rule is useful or a nuisance.
        RedactionRule::DigitRun {
            kind: "card".to_string(),
            min_digits: 13,
            max_digits: 19,
        },
    ];
    let plugin = Arc::new(RedactionPlugin::new(RedactionConfig::new(rules.clone())?)?);
    println!(
        "Policy: {} rules, match cap {} matches per string",
        plugin.rule_count(),
        plugin.match_cap()
    );
    for (index, rule) in rules.iter().enumerate() {
        println!("  rule {index}  {}", describe(rule));
    }

    // Stage 2: the matcher on its own. `scrub_str` is public so a caller can
    // scrub a value before it ever enters a request, or reject on `count > 0`
    // instead of rewriting. A clean string comes back borrowed: the clean path
    // allocates nothing at all. It is a pure function of the policy and the
    // input, so it moves none of the counters read in stage 5.
    println!("\nCalling scrub_str directly, before anything enters a request");
    show_scrub("clean input", &plugin, CLEAN_SAMPLE)?;
    show_scrub("dirty input", &plugin, DIRTY_SAMPLE)?;

    // Stage 3: the client, with the plugin registered as an ordinary
    // `Arc<dyn CucaPlugin>`. The rewrite now happens inside the pipeline.
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&plugin) as Arc<dyn CucaPlugin>)
        .build()?;

    println!("\nDispatching a seeded prompt through the registered hook");
    println!("  before: {SEEDED_PROMPT}");
    // The hook applies exactly this rewrite to `req.messages` in `on_request`,
    // so printing it here is what the provider is about to receive. The
    // request itself is dispatched unscrubbed and the hook does the work; this
    // line is the preview, not the mutation.
    println!("  wire:   {}", plugin.scrub_str(SEEDED_PROMPT)?.text);

    // Stage 4: dispatch and drain. `on_request` runs before the provider
    // adapter translates anything onto the wire.
    let request = UnifiedRequest::new(&model)
        .add_system_message("You are concise.")
        .add_user_message(SEEDED_PROMPT)
        .set_max_tokens(64);
    let mut stream = match client.generate_stream(request).await {
        Ok(stream) => stream,
        Err(error) => {
            println!("\nNo server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    print!("  reply:  ");
    stdout().flush()?;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(text)) => {
                print!("{text}");
                stdout().flush()?;
            }
            Ok(_) => {}
            Err(error) => {
                print!("[the stream ended early: {error}]");
                break;
            }
        }
    }
    println!();

    // Stage 5: the counters. Fixed-size by construction: two totals and one
    // most-recent event tuple, replaced rather than appended, so nothing here
    // grows with traffic. Per-`kind` aggregation belongs on an OTel meter, fed
    // by the `kind` field of the `tracing::warn!` event the hook emits per
    // redaction (target `cuca::redaction`).
    println!("\nCounters after the turn");
    println!(
        "  {:<24} {}",
        "last_request_redactions",
        plugin.last_request_redactions()
    );
    println!("  {:<24} {}", "total_redactions", plugin.total_redactions());
    match plugin.last_redaction_event() {
        Some((kind, field, count)) => println!(
            "  {:<24} kind={kind} field={field} count={count}",
            "last_redaction_event"
        ),
        None => println!("  {:<24} none: nothing matched", "last_redaction_event"),
    }
    Ok(())
}
