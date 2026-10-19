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

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-redaction` (enables `tracing`; the matchers use the core `memchr` dependency) |
| Entry types | `RedactionPlugin`, `RedactionConfig`, `RedactionRule`, `Redacted` |
| Plugin name | `"redaction"` |
| Attachment | `register_plugin` |
| Needs | A non-empty, caller-written rule list; the plugin ships no built-in rules |
| Direct call | `RedactionPlugin::scrub_str(&self, text: &str) -> Result<Redacted<'_>, PluginError>` |

`scrub_str` is the same matcher the hook runs. A caller can scrub a value before it enters a request, reject on `count > 0` instead of rewriting, or reuse the compiled policy outside the pipeline. A clean string returns `Redacted { text: Cow::Borrowed(text), count: 0 }` and allocates nothing. `scrub_str` emits no tracing event and moves no counter.

## Hooks

`RedactionPlugin` overrides `on_request` only. `execute_local_tool`, `on_stream_chunk`, and `on_response_complete` use the trait defaults.

`on_request` runs the matcher over each field listed below, in message order, then over tool descriptions. It replaces each match with `[REDACTED:{kind}]`, so the provider adapter never sees the original value.

### Scrubbed fields

| Location | `field` label | Action |
|---|---|---|
| `MessageContentBlock::Text` | `message_text` | Scrub the text |
| `MessageContentBlock::Thinking`, `reasoning` | `message_thinking` | Scrub the reasoning |
| `MessageContentBlock::ToolCall`, `arguments` | `tool_call_arguments` | Scrub every JSON string leaf, recursively, to depth `MAX_JSON_DEPTH` |
| `MessageContentBlock::ToolResult`, `output` | `tool_result_output` | Scrub the output |
| `UnifiedMessage::name` | `message_name` | Scrub the `Some` payload |
| `ToolDefinition::description` | `tool_description` | Scrub when `scrub_tool_definitions` is `true` |

### Fields left unchanged

Each omission is a decision, not a gap.

- `Thinking::signature` authenticates the reasoning, and rewriting it invalidates it.
- `ToolCall::id`, `ToolResult::tool_call_id`, and `UnifiedMessage::tool_call_id` pair calls with results, and scrubbing them breaks the match.
- `ToolCall::name` and `ToolDefinition::name` identify the tool, and a renamed tool cannot be routed.
- `ToolDefinition::input_schema` holds the tool's contract in keywords such as `const`, `enum`, and `pattern`. [Guardrails](@/plugins/guardrails.md) validates against the same schemas.
- `ImageBase64 { media_type, data }`: a hit inside base64 is a coincidence, and rewriting it corrupts the image.
- JSON object keys inside `arguments` belong to the tool's input contract.
- `req.model`, `req.provider`, `temperature`, `max_tokens`, `stream`, `thinking`, and `prompt_cache` are routing and sampling settings, not text.

### Inbound text

`on_stream_chunk` is not implemented, so an inbound block carrying a matching secret comes back unchanged.

- The plugin stops caller-side data crossing the wire to a provider. A provider has already seen whatever it emitted.
- Providers stream text in fragments, so a secret split across two `Text` blocks would be missed. Reliable matching needs cross-chunk buffering, which conflicts with streaming latency and the bounded-memory rules.
- Assistant turns and tool results re-enter the conversation as `req.messages` on the next request, where `on_request` scrubs them. Scrubbing is idempotent, so a multi-turn conversation converges.

### Order with other plugins

The plugin requires no registration position relative to any other plugin. `on_request` hooks run in registration order over one shared request, so its effect is visible to plugins registered after it.

- With `plugin-session-log`, registering `RedactionPlugin` first means the trajectory stores scrubbed content. Registering it second means the log persists the raw value, to disk when `FileBackend` is in use, where the format never rewrites an existing frame. Model output records are never scrubbed in either order.
- With `service-prompt-cache`, the digest is computed after every `on_request` hook, so enabling redaction changes every cache key. `PromptCache::replace_snapshot` rejects a snapshot from before redaction as a digest mismatch. Two requests that differ only inside a redacted secret share one key.

### Tracing event and counters

Each replacement emits one `tracing::warn!` with the message `outbound value redacted`, under the target `cuca::redaction`. Its fields are `kind`, `field`, and `message_index`. It never carries the matched bytes or their surroundings.

`message_index` is the position in `req.messages`, except for `tool_description`, where it is the position in `req.tools`.

- `last_request_redactions()` returns the `u64` count from the most recent `on_request`, `0` before the first.
- `total_redactions()` returns the cumulative count across requests, using `saturating_add`.
- `last_redaction_event()` returns `Option<(String, &'static str, usize)>`: the `kind`, the field label, and the match count of the last field that matched. It mirrors the tracing event and never holds a matched value.

Per-kind aggregation belongs on a caller's OpenTelemetry meter, fed by the `kind` field of the event.

### Non-guarantees

- The plugin is a byte matcher, not a classifier. A secret with no rule crosses the wire, and there is no "looks like a secret" detector.
- A false positive, such as a `DigitRun` rule matching an order number, is a policy the caller wrote and can see in the tracing event.
- It does not encrypt, and it does not redact anything already persisted or exported. Redaction runs upstream of storage and is not an export-time guarantee.

## Configuration

`RedactionConfig::new(rules)` is the rejecting constructor. `validate()` re-checks the same bounds, because the fields are public and a struct literal bypasses `new`. `RedactionPlugin::new` validates again.

| Key | Type | Default |
|---|---|---|
| `rules` | `Vec<RedactionRule>` | none, required, non-empty |
| `max_matches_per_text` | `usize` | `1024` |
| `scrub_tool_definitions` | `bool` | `true` |

- `rules` is the policy. Its order breaks ties between overlapping matches.
- `max_matches_per_text` is the per-string match cap. Exceeding it fails the hook instead of partially redacting.
- `scrub_tool_definitions` controls whether `ToolDefinition::description` is scrubbed.

`RedactionConfig::default()` has empty `rules`, so `RedactionPlugin::new` rejects it. It exists for `..Default::default()` struct updates.

### Rules

`RedactionRule` has four variants. Every variant carries a `kind`, which labels the replacement token and the tracing event and is never the matched value.

- `Literal { kind, value }` matches an exact, byte-for-byte occurrence of a known secret.
- `Prefixed { kind, prefix, min_len, max_len }` matches `prefix` followed by `min_len..=max_len` bytes from the token alphabet: ASCII alphanumerics, `-`, and `_`. Longer runs truncate at `max_len`. Example: `prefix = "sk-"`.
- `EmailLike { kind }` matches a `local@domain.tld` run in ASCII. The local part opens on an alphanumeric and may hold `.`, `_`, `%`, `+`, and `-`. The domain holds at least one dot that is not its first byte, and a trailing `.` or `-` stays outside the match.
- `DigitRun { kind, min_digits, max_digits }` matches a run of `min_digits..=max_digits` ASCII digits, optionally split by single `-` or space separators. It fits card, phone, and national ID shapes. The run ends on a digit and truncates at `max_digits`.

Overlapping matches resolve leftmost-longest, with ties broken by policy order, so the output is a function of the policy and the input.

### Matchers and dependencies

`Literal` and `Prefixed` rules compile once, in `RedactionPlugin::new`, to a `memchr::memmem::Finder`. `EmailLike` and `DigitRun` are hand-written ASCII scans. The plugin has no `regex` dependency: it would add `regex-syntax`, `regex-automata`, and `aho-corasick`, and put a caller-supplied pattern language on the outbound hot path. `RedactionRule` is the extension seam for a future variant behind its own feature.

### Validation bounds

`RedactionConfig::new`, `validate()`, and `RedactionPlugin::new` reject an unusable policy with `PluginError::Validation { schema: "redaction-config", .. }`.

| Bound | Constant | Rejects |
|---|---|---|
| Rule count | `MAX_RULES` (256) | Zero rules, or more than the cap |
| Pattern length | `MAX_PATTERN_BYTES` (512) | An empty or over-long `Literal::value` or `Prefixed::prefix` |
| `kind` slug | `MAX_KIND_BYTES` (32) | A `kind` that is empty, over the cap, or not made of `[a-z0-9_-]` |
| Replacement marker | `REDACTION_MARKER` (`"[REDACTED:"`) | A `Literal::value` that contains the marker |
| `Prefixed` range | none | `min_len > max_len` |
| `DigitRun` range | none | `min_digits == 0`, or `min_digits > max_digits` |
| Match cap | `MAX_MATCHES_PER_TEXT` (4096) | `max_matches_per_text == 0`, or over the ceiling |

The marker check keeps scrubbing idempotent. No rule can match the plugin's own replacement token, so a second pass over scrubbed text adds no redactions. `MAX_JSON_DEPTH` (64) bounds `ToolCall::arguments` recursion. All constants are associated constants on `RedactionConfig`.

## Caps and at-cap policy

Nothing in this plugin grows with traffic. There is no near-cap warning, because no structure fills over time.

| Structure | Bound | At-cap policy | Usage reading |
|---|---|---|---|
| `rules` | `MAX_RULES` (256), each pattern `MAX_PATTERN_BYTES` (512), each `kind` `MAX_KIND_BYTES` (32) | Fixed at construction. An empty or over-cap policy is rejected before any allocation | `rule_count()` |
| Per-string match buffer | `max_matches_per_text`, at most `MAX_MATCHES_PER_TEXT` (4096) | Checked while pushing, so the buffer never exceeds the cap. Past the cap the call returns `PluginError::HookFailure` | `match_cap()` |
| Stats | Two `u64` counters and one `(kind, field, count)` tuple | The tuple is replaced, never appended | `last_request_redactions()`, `total_redactions()`, `last_redaction_event()` |
| `ToolCall::arguments` recursion | `MAX_JSON_DEPTH` (64) | A depth pre-pass walks every tool call before the first rewrite and returns `PluginError::HookFailure` past the bound | None, per call |

A cap refusal makes `on_request` fail, so the request never reaches a provider and later plugins' hooks never run. The depth refusal leaves the request untouched. A match-cap refusal can leave fields that were scrubbed earlier in the pass scrubbed in memory. The request is refused whole rather than sent partly scrubbed.

Allocation on the scan path:

- Searchers and replacement tokens are built once in `RedactionPlugin::new`.
- A clean string allocates nothing: `scrub_str` returns `Cow::Borrowed` and no per-call buffer is pushed to.
- A dirty string allocates one output `String`, sized from the input length and the net replacement delta. It also allocates the bounded match buffer and one candidate cache of `rules.len()` entries, which keeps the scan linear in the input.

## Errors

| Variant | When |
|---|---|
| `Validation`, schema `redaction-config` | `RedactionConfig::new`, `validate()`, or `RedactionPlugin::new` rejects the policy. The message names the field, such as `rules[3].kind` or `max_matches_per_text` |
| `HookFailure`, plugin `"redaction"`, stage `"request"` | A string holds more than `match_cap()` matches, from `scrub_str` or from `on_request` |
| `HookFailure`, same plugin and stage | A `ToolCall` `arguments` value nests deeper than `MAX_JSON_DEPTH`, from `on_request` |

## Example

`examples/redaction.rs` builds a four-rule policy, calls `scrub_str` on a clean and a dirty sample, sends a seeded prompt through a live turn, and prints the counters.

```rust,name=A four-rule policy
let rules = vec![
    RedactionRule::Literal {
        kind: "deploy-key".to_string(),
        value: "DEPLOY-KEY-a1b2c3d4".to_string(),
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
    RedactionRule::DigitRun {
        kind: "card".to_string(),
        min_digits: 13,
        max_digits: 19,
    },
];
let plugin = Arc::new(RedactionPlugin::new(RedactionConfig::new(rules)?)?);
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url)
    .register_plugin(Arc::clone(&plugin) as Arc<dyn CucaPlugin>)
    .build()?;
```

```bash,name=Runs the same on all three platforms
cargo run --example redaction --features "provider-llamacpp plugin-redaction"
```

It needs a `llama-server` on port 1234. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at another OpenAI-compatible server.

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

The reply text and the thinking-block count come from `google/gemma-4-12b-qat` and change with the model. The four redactions and the counters do not.

## Related pages

- [Session log](@/plugins/session-log.md): registration order decides whether the trajectory stores scrubbed or raw content.
- [Prompt cache](@/services/prompt-cache.md): redaction changes every cache digest.
- [Guardrails](@/plugins/guardrails.md): validates tool calls against the schemas redaction leaves unchanged.
- [Memory discipline](@/concepts/memory-discipline.md): why every cap here is a hard refusal.
