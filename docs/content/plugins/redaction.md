+++
title = "Redaction"
description = "Outbound PII and secret scrubbing over every text-bearing request field: the rule types, the scrubbed-field map, and why on_stream_chunk stays a no-op."
template = "page.html"
weight = 16
+++

# Redaction

<dl class="page-facts">
<dt>In one line</dt>
<dd>Scrubs matched secrets and PII out of every text-bearing field of the outbound <code>UnifiedRequest</code> in <code>on_request</code>, replacing each hit with a <code>[REDACTED:&lt;kind&gt;]</code> marker.</dd>
<dt>You need</dt>
<dd>The <code>plugin-redaction</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are writing a redaction policy, registering <code>RedactionPlugin</code>, or tracing why a field was or was not scrubbed.</dd>
</dl>

`RedactionPlugin` scrubs matched secrets and PII out of every text-bearing field of the outbound `UnifiedRequest` in `on_request`, replacing each hit with a `[REDACTED:<kind>]` marker before the request reaches a provider. Rules are entirely caller-authored: `Literal`, `Prefixed`, `EmailLike`, and `DigitRun` cover common leak shapes, and the plugin ships no built-in rule set or heuristic detector. Reach for it to keep API keys, emails, and other caller-known secrets out of a provider's logs and a session-log trajectory.

```rust,name=Scrub four secret shapes out of one live turn
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

    // Stage 1: the policy. Every rule is caller-authored: there is no built-in
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
    // adapter translates anything onto the wire. A reasoning model spends most
    // of its budget on `Thinking` blocks before the first `Text` block, so
    // `max_tokens` has to leave room for both or the reply comes back empty.
    let request = UnifiedRequest::new(&model)
        .add_system_message("You are concise.")
        .add_user_message(SEEDED_PROMPT)
        .set_max_tokens(512);
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
    let mut thinking_blocks = 0usize;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(text)) => {
                print!("{text}");
                stdout().flush()?;
            }
            // Counted rather than printed: one block per reasoning token would
            // bury the one line this stage exists to show.
            Ok(MessageContentBlock::Thinking { .. }) => thinking_blocks += 1,
            Ok(_) => {}
            Err(error) => {
                print!("[the stream ended early: {error}]");
                break;
            }
        }
    }
    println!();
    println!("  blocks: {thinking_blocks} thinking, not printed");

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
```

```text,name=Expected output
Policy: 4 rules, match cap 1024 matches per string
  rule 0  literal     deploy-key exactly 19 bytes
  rule 1  prefixed    api-key    "sk-" + 8..=48 token bytes
  rule 2  email-like  email      local@domain.tld
  rule 3  digit-run   card       13..=19 digits, `-`/` ` separated

Calling scrub_str directly, before anything enters a request
  clean input  0 redactions, Cow::Borrowed (nothing allocated)
    build finished in 4.2s with 0 warnings
  dirty input  2 redactions, Cow::Owned (one rewritten String)
    cat .env -> API_TOKEN=[REDACTED:api-key] and OWNER=[REDACTED:email]

Dispatching a seeded prompt through the registered hook
  before: Rotate the runner: key DEPLOY-KEY-a1b2c3d4, provider token sk-live-51H9x0AbCdEfGh, notify ops@example.com, billing card 4111 1111 1111 1111. Reply with exactly: understood
  wire:   Rotate the runner: key [REDACTED:deploy-key], provider token [REDACTED:api-key], notify [REDACTED:email], billing card [REDACTED:card]. Reply with exactly: understood
  reply:  understood
  blocks: 290 thinking, not printed

Counters after the turn
  last_request_redactions  4
  total_redactions         4
  last_redaction_event     kind=card field=message_text count=4
```

## Try it

`examples/redaction.rs` is the program above. It prints the compiled policy, two direct `scrub_str` calls on a clean and a dirty sample, the seeded prompt before and after the hook rewrote it, the streamed reply, and the counters. It needs a `llama-server` on port 1234 with the demo model loaded; `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server. The reply text and the thinking-block count come from `google/gemma-4-12b-qat` and change with the model; the four redactions and the counters do not.

```bash,name=Runs the same on all three platforms
cargo run --example redaction --features "provider-llamacpp plugin-redaction"
```

## Entry types

`RedactionPlugin`, `RedactionConfig`, `RedactionRule`, `Redacted`.

## `CucaPlugin`

`RedactionPlugin` implements `CucaPlugin` with the plugin name `"redaction"` and attaches via `register_plugin`, like any other hook plugin. It overrides `on_request` only; `execute_local_tool`, `on_stream_chunk`, and `on_response_complete` take the trait defaults: see *Inbound deferral* below for why `on_stream_chunk` in particular stays a no-op rather than a partial implementation.

The single matcher entry point is `RedactionPlugin::scrub_str`, `pub fn scrub_str<'a>(&self, text: &'a str) -> Result<Redacted<'a>, PluginError>`. `on_request` calls it once per scrubbed field, and it is public so a caller can scrub a value before it ever enters a request, or reuse the same compiled policy outside the pipeline entirely. A clean string returns `Redacted { text: Cow::Borrowed(text), count: 0 }` and allocates nothing.

## Rules

`RedactionRule` has four variants. `kind` labels the replacement token and the tracing event on every variant; it is never the matched value.

| Variant | Matches | Fields |
|---|---|---|
| `Literal` | An exact, byte-for-byte occurrence of a known secret value | `kind`, `value` |
| `Prefixed` | `prefix` followed by `min_len..=max_len` bytes from the token alphabet (ASCII alphanumeric, `-`, `_`), e.g. `prefix = "sk-"` | `kind`, `prefix`, `min_len`, `max_len` |
| `EmailLike` | A `local@domain.tld`-shaped run: ASCII local part, at least one dot in the domain, no trailing separator | `kind` |
| `DigitRun` | A run of `min_digits..=max_digits` ASCII digits, optionally interrupted by single `-` or ` ` separators (card / phone / national-ID shapes) | `kind`, `min_digits`, `max_digits` |

Overlapping matches resolve leftmost-longest, breaking ties by policy order: the order `rules` are given in `RedactionConfig::new`.

## Config

`RedactionConfig::new(rules)` is the rejecting constructor; `validate()` re-checks the same bounds, since the fields are public and a struct literal can bypass `new`.

| Field | Meaning | Default |
|---|---|---|
| `rules` | The policy, applied in this order for overlap tie-breaks; must be non-empty | required |
| `max_matches_per_text` | Per-string cap on matches; exceeding it fails the hook instead of partially redacting | `1024` |
| `scrub_tool_definitions` | Whether `ToolDefinition::description` is scrubbed | `true` |

`RedactionConfig::default()` is deliberately unusable: `rules` is empty, so `RedactionPlugin::new` rejects it. It exists only for `..Default::default()` struct-update ergonomics, the same shape as `MemoryConfig`.

### Validation bounds

`RedactionConfig::new` and `validate()` reject an unusable policy with `PluginError::Validation { schema: "redaction-config", .. }`:

| Bound | Constant | Rejects |
|---|---|---|
| Rule count | `RedactionConfig::MAX_RULES` (256) | zero rules, or more than the cap |
| Pattern length | `RedactionConfig::MAX_PATTERN_BYTES` (512) | an empty or over-long `Literal::value` / `Prefixed::prefix` |
| `kind` slug | `RedactionConfig::MAX_KIND_BYTES` (32) | a `kind` that is not a non-empty ASCII `[a-z0-9_-]` slug, or one over the byte cap |
| Idempotency marker | `RedactionConfig::REDACTION_MARKER` (`"[REDACTED:"`) | a `Literal::value` that contains the marker itself |
| `Prefixed` range | n/a | `min_len > max_len` |
| `DigitRun` range | n/a | `min_digits == 0`, or `min_digits > max_digits` |
| Match cap | `RedactionConfig::MAX_MATCHES_PER_TEXT` (4096) | `max_matches_per_text == 0`, or over the ceiling |

Rejecting a `Literal::value` that contains `REDACTION_MARKER` is what makes scrubbing idempotent: a caller cannot author a rule that matches the plugin's own replacement token, so a second pass over already-scrubbed text adds no further redactions.

## Scrubbed fields

| Location | `field` label | Action |
|---|---|---|
| `MessageContentBlock::Text(t)` | `"message_text"` | scrub `t` |
| `Thinking { reasoning, .. }` | `"message_thinking"` | scrub `reasoning` |
| `ToolCall { arguments, .. }` | `"tool_call_arguments"` | scrub every JSON string leaf of the `serde_json::Value`, recursively, depth-bounded by `RedactionConfig::MAX_JSON_DEPTH` |
| `ToolResult { output, .. }` | `"tool_result_output"` | scrub `output` |
| `UnifiedMessage::name` | `"message_name"` | scrub the `Some` payload |
| `ToolDefinition::description` | `"tool_description"` | scrub when `scrub_tool_definitions` is set |

Each redaction emits one `tracing::warn!` under target `cuca::redaction` carrying `kind`, the `field` label, and the message index: never the matched bytes or the surrounding text. The message index is the position in `req.messages`, except for the `tool_description` field, where it is the position in `req.tools` instead.

## Not scrubbed, on purpose

Each omission is a stated decision, not a gap:

- `Thinking::signature`: an optional provider signature authenticating the reasoning; rewriting it invalidates it.
- `ToolCall::id`, `ToolResult::tool_call_id`, `UnifiedMessage::tool_call_id`: correlation ids; scrubbing them breaks call/result matching.
- `ToolCall::name`, `ToolDefinition::name`: tool identity; a renamed tool is an unroutable tool.
- `ToolDefinition::input_schema`: JSON Schema keywords (`const`, `enum`, `pattern`) are the tool's contract; rewriting them silently changes tool semantics. [Guardrails](@/plugins/guardrails.md) validates against these same schemas.
- `ImageBase64 { media_type, data }`: a literal hit inside base64 is a coincidence, and rewriting the payload corrupts the image.
- JSON object keys inside `arguments`: part of the tool's input contract, not user data.
- `req.model`, `req.provider`, `temperature`, `max_tokens`, `stream`, `thinking`, `prompt_cache`: routing/sampling knobs, not text.

## Why no regex

`Literal` and `Prefixed` rules compile to a `memchr::memmem::Finder`, built once at construction; `EmailLike` and `DigitRun` are hand-rolled ASCII scans. There is no `dep:regex`: it would let a rule express arbitrary detector shapes, but it also pulls `regex-syntax`, `regex-automata`, and `aho-corasick` into a crate whose core carries five dependencies, and it would put an untrusted, caller-supplied pattern language on the outbound hot path, where a pathological pattern becomes a per-request latency problem. The four matcher shapes cover the concrete leak classes this plugin exists for, and `RedactionRule` is the extension seam: a `Regex` variant behind a second feature is additive, not a rewrite.

## Inbound deferral

`on_stream_chunk` is *not* implemented; the trait default applies, so an inbound block carrying a matching secret comes back unchanged. Three reasons:

1. **Threat model.** The leak this plugin exists to stop is caller-side data crossing the wire to a provider. An inbound block has already been produced by the provider; scrubbing it protects nothing on the wire.
2. **A per-chunk matcher would be silently unreliable.** Providers stream text incrementally, so `on_stream_chunk` sees fragments; a secret straddling two `Text` blocks would be missed. A matcher that only works when token boundaries happen to align is worse than none.
3. **Inbound text is scrubbed anyway, on its next outbound pass.** Assistant turns and tool results re-enter the conversation as `req.messages` on the following request, where `on_request` scrubs them. Because scrubbing is idempotent (the replacement token cannot itself re-match), a multi-turn conversation converges.

## Order observability

The plugin does not require a registration position relative to any other plugin, but its effect is order-observable at two consuming sites, and both orders are legal:

- **`plugin-session-log`.** `on_request` hooks run in registration order over one shared request. Registering `RedactionPlugin` before `SessionLogPlugin` means the trajectory stores scrubbed content; registering it after means the log persists the raw value, to disk, if `FileBackend` is in use, where the append-only format never rewrites an existing frame. Model-output records are never scrubbed at any order, because `on_stream_chunk` is deferred.
- **`service-prompt-cache`.** The digest is computed from the request after every `on_request` hook, so enabling redaction changes every cache key. A snapshot imported from a pre-redaction run is rejected as a digest mismatch by `PromptCache::replace_snapshot`, not silently mismatched.

## Bounds

Nothing in this plugin grows with traffic:

| Structure | Cap | At-cap policy | Usage reading |
|---|---|---|---|
| `rules` | `RedactionConfig::MAX_RULES` (256), each pattern `MAX_PATTERN_BYTES` (512), each `kind` `MAX_KIND_BYTES` (32) | fixed at construction; `RedactionConfig::new` / `validate` reject an over-cap or empty policy before any allocation | `rule_count()` |
| per-call match buffer | `max_matches_per_text`, itself `<= MAX_MATCHES_PER_TEXT` (4096) | checked while pushing, so the buffer never exceeds the cap; over the cap the call returns `PluginError::HookFailure` and `on_request` refuses the whole turn before dispatch. This refusal is not atomic at the request level: fields scrubbed earlier in the same pass stay scrubbed in memory, but the mutated request never reaches a provider either way | `match_cap()` |
| stats | fixed: two `u64` counters plus one most-recent `(kind, field, count)` tuple | the event slot is replaced, never appended; the total uses `saturating_add` | `last_request_redactions()`, `total_redactions()`, `last_redaction_event()` |
| `ToolCall::arguments` recursion | `RedactionConfig::MAX_JSON_DEPTH` (64) stack frames | a depth pre-pass walks the value before any rewrite, so exceeding the cap returns `PluginError::HookFailure` atomically, no partially-rewritten `arguments`, with no unbounded recursion on an adversarially nested `serde_json::Value` | n/a (per-call) |

Searchers and replacement tokens are built once in `RedactionPlugin::new`; there is no per-request searcher construction. A clean string allocates nothing: `scrub_str` hands back a `Cow::Borrowed` and neither per-call buffer is ever pushed to. A dirty string allocates one output `String`, sized exactly from the input length and the net replacement delta, plus the bounded match buffer and one `rules.len()`-entry candidate cache that keeps the scan linear in the input instead of re-searching every rule after every replacement. Both cap-adjacent conditions are hard, loud errors rather than truncation: the request is always refused before dispatch rather than crossing the wire partially scrubbed, whether or not the in-memory `UnifiedRequest` was partially mutated first.

## Non-guarantees

- **It is a byte matcher, not a classifier.** An unlisted secret shape crosses the wire; there is no heuristic "looks like a secret" detector.
- **No built-in rule set.** `rules` must be non-empty and every rule is caller-authored, so a false positive (`DigitRun` eating an order number, for example) is a policy the caller wrote and can see in the `tracing` event.
- **It does not encrypt, and it does not redact anything already persisted or exported.** Redaction happens upstream of storage; it is not an export-time guarantee.
- **The tracing event never carries the matched value**: only `kind`, the field label, and the message index.
