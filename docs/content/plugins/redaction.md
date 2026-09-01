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

## Entry types

`RedactionPlugin`, `RedactionConfig`, `RedactionRule`, `Redacted`.

## `CucaPlugin`

`RedactionPlugin` implements `CucaPlugin` with the plugin name `"redaction"` and attaches via `register_plugin`, like any other hook plugin. It overrides `on_request` only; `execute_local_tool`, `on_stream_chunk`, and `on_response_complete` take the trait defaults — see *Inbound deferral* below for why `on_stream_chunk` in particular stays a no-op rather than a partial implementation.

The single matcher entry point is `RedactionPlugin::scrub_str`, `pub fn scrub_str<'a>(&self, text: &'a str) -> Result<Redacted<'a>, PluginError>`. `on_request` calls it once per scrubbed field, and it is public so a caller can scrub a value before it ever enters a request, or reuse the same compiled policy outside the pipeline entirely. A clean string returns `Redacted { text: Cow::Borrowed(text), count: 0 }` and allocates nothing.

## Rules

`RedactionRule` has four variants. `kind` labels the replacement token and the tracing event on every variant; it is never the matched value.

| Variant | Matches | Fields |
|---|---|---|
| `Literal` | An exact, byte-for-byte occurrence of a known secret value | `kind`, `value` |
| `Prefixed` | `prefix` followed by `min_len..=max_len` bytes from the token alphabet (ASCII alphanumeric, `-`, `_`), e.g. `prefix = "sk-"` | `kind`, `prefix`, `min_len`, `max_len` |
| `EmailLike` | A `local@domain.tld`-shaped run: ASCII local part, at least one dot in the domain, no trailing separator | `kind` |
| `DigitRun` | A run of `min_digits..=max_digits` ASCII digits, optionally interrupted by single `-` or ` ` separators (card / phone / national-ID shapes) | `kind`, `min_digits`, `max_digits` |

Overlapping matches resolve leftmost-longest, breaking ties by policy order — the order `rules` are given in `RedactionConfig::new`.

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
| `Prefixed` range | — | `min_len > max_len` |
| `DigitRun` range | — | `min_digits == 0`, or `min_digits > max_digits` |
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

Each redaction emits one `tracing::warn!` under target `cuca::redaction` carrying `kind`, the `field` label, and the message index — never the matched bytes or the surrounding text. The message index is the position in `req.messages`, except for the `tool_description` field, where it is the position in `req.tools` instead.

## Not scrubbed, on purpose

Each omission is a stated decision, not a gap:

- `Thinking::signature` — an optional provider signature authenticating the reasoning; rewriting it invalidates it.
- `ToolCall::id`, `ToolResult::tool_call_id`, `UnifiedMessage::tool_call_id` — correlation ids; scrubbing them breaks call/result matching.
- `ToolCall::name`, `ToolDefinition::name` — tool identity; a renamed tool is an unroutable tool.
- `ToolDefinition::input_schema` — JSON Schema keywords (`const`, `enum`, `pattern`) are the tool's contract; rewriting them silently changes tool semantics. [Guardrails](@/plugins/guardrails.md) validates against these same schemas.
- `ImageBase64 { media_type, data }` — a literal hit inside base64 is a coincidence, and rewriting the payload corrupts the image.
- JSON object keys inside `arguments` — part of the tool's input contract, not user data.
- `req.model`, `req.provider`, `temperature`, `max_tokens`, `stream`, `thinking`, `prompt_cache` — routing/sampling knobs, not text.

## Why no regex

`Literal` and `Prefixed` rules compile to a `memchr::memmem::Finder`, built once at construction; `EmailLike` and `DigitRun` are hand-rolled ASCII scans. There is no `dep:regex`: it would let a rule express arbitrary detector shapes, but it also pulls `regex-syntax`, `regex-automata`, and `aho-corasick` into a crate whose core carries five dependencies, and it would put an untrusted, caller-supplied pattern language on the outbound hot path, where a pathological pattern becomes a per-request latency problem. The four matcher shapes cover the concrete leak classes this plugin exists for, and `RedactionRule` is the extension seam: a `Regex` variant behind a second feature is additive, not a rewrite.

## Inbound deferral

`on_stream_chunk` is *not* implemented; the trait default applies, so an inbound block carrying a matching secret comes back unchanged. Three reasons:

1. **Threat model.** The leak this plugin exists to stop is caller-side data crossing the wire to a provider. An inbound block has already been produced by the provider; scrubbing it protects nothing on the wire.
2. **A per-chunk matcher would be silently unreliable.** Providers stream text incrementally, so `on_stream_chunk` sees fragments; a secret straddling two `Text` blocks would be missed. A matcher that only works when token boundaries happen to align is worse than none.
3. **Inbound text is scrubbed anyway, on its next outbound pass.** Assistant turns and tool results re-enter the conversation as `req.messages` on the following request, where `on_request` scrubs them. Because scrubbing is idempotent (the replacement token cannot itself re-match), a multi-turn conversation converges.

## Order observability

The plugin does not require a registration position relative to any other plugin — but its effect is order-observable at two consuming sites, and both orders are legal:

- **`plugin-session-log`.** `on_request` hooks run in registration order over one shared request. Registering `RedactionPlugin` before `SessionLogPlugin` means the trajectory stores scrubbed content; registering it after means the log persists the raw value — to disk, if `FileBackend` is in use, where the append-only format never rewrites an existing frame. Model-output records are never scrubbed at any order, because `on_stream_chunk` is deferred.
- **`plugin-prompt-cache`.** The digest is computed from the request after every `on_request` hook, so enabling redaction changes every cache key. A snapshot imported from a pre-redaction run is rejected as a digest mismatch by `PromptCache::replace_snapshot`, not silently mismatched.

## Bounds

Nothing in this plugin grows with traffic:

| Structure | Cap | At-cap policy | Usage reading |
|---|---|---|---|
| `rules` | `RedactionConfig::MAX_RULES` (256), each pattern `MAX_PATTERN_BYTES` (512), each `kind` `MAX_KIND_BYTES` (32) | fixed at construction; `RedactionConfig::new` / `validate` reject an over-cap or empty policy before any allocation | `rule_count()` |
| per-call match buffer | `max_matches_per_text`, itself `<= MAX_MATCHES_PER_TEXT` (4096) | checked while pushing, so the buffer never exceeds the cap; over the cap the call returns `PluginError::HookFailure` and `on_request` refuses the whole turn before dispatch. This refusal is not atomic at the request level — fields scrubbed earlier in the same pass stay scrubbed in memory — but the mutated request never reaches a provider either way | `match_cap()` |
| stats | fixed: two `u64` counters plus one most-recent `(kind, field, count)` tuple | the event slot is replaced, never appended; the total uses `saturating_add` | `last_request_redactions()`, `total_redactions()`, `last_redaction_event()` |
| `ToolCall::arguments` recursion | `RedactionConfig::MAX_JSON_DEPTH` (64) stack frames | a depth pre-pass walks the value before any rewrite, so exceeding the cap returns `PluginError::HookFailure` atomically — no partially-rewritten `arguments` — with no unbounded recursion on an adversarially nested `serde_json::Value` | n/a (per-call) |

Searchers and replacement tokens are built once in `RedactionPlugin::new`; there is no per-request searcher construction. A clean string allocates nothing — `scrub_str` hands back a `Cow::Borrowed` and neither per-call buffer is ever pushed to. A dirty string allocates one output `String`, sized exactly from the input length and the net replacement delta, plus the bounded match buffer and one `rules.len()`-entry candidate cache that keeps the scan linear in the input instead of re-searching every rule after every replacement. Both cap-adjacent conditions are hard, loud errors rather than truncation: the request is always refused before dispatch rather than crossing the wire partially scrubbed, whether or not the in-memory `UnifiedRequest` was partially mutated first.

## Non-guarantees

- **It is a byte matcher, not a classifier.** An unlisted secret shape crosses the wire; there is no heuristic "looks like a secret" detector.
- **No built-in rule set.** `rules` must be non-empty and every rule is caller-authored, so a false positive (`DigitRun` eating an order number, for example) is a policy the caller wrote and can see in the `tracing` event.
- **It does not encrypt, and it does not redact anything already persisted or exported.** Redaction happens upstream of storage; it is not an export-time guarantee.
- **The tracing event never carries the matched value** — only `kind`, the field label, and the message index.

## Registering it

```rust,name=A literal secret plus an email-shaped rule
let config = RedactionConfig::new(vec![
    RedactionRule::Prefixed {
        kind: "api-key".into(),
        prefix: "sk-".into(),
        min_len: 20,
        max_len: 64,
    },
    RedactionRule::EmailLike { kind: "email".into() },
])?;
let redaction = RedactionPlugin::new(config)?;

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url)
    .register_plugin(Arc::new(redaction))
    .build()?;
```

## Try it

`examples/redaction.rs` runs one client through one turn against a four-rule caller-authored policy: a `Literal` deploy key, a `Prefixed` `sk-` API key, an `EmailLike` email, and a `DigitRun` card number, all fake. Five stages print in order: the compiled policy (`rule_count()`, `match_cap()`); two direct `scrub_str` calls on a clean and a dirty sample, each reporting its match count and whether the result was `Cow::Borrowed` (nothing allocated) or `Cow::Owned` (one rewritten `String`); the seeded prompt before and after, with a `wire:` line showing all four values replaced by their `[REDACTED:<kind>]` markers; the streamed reply; and the counters afterward — `last_request_redactions()` and `total_redactions()` both reach `4`, and `last_redaction_event()` names the last kind matched. The example never prints a matched value from the plugin's own reporting path; the `Literal` rule is described by byte length, not value. With no server reachable, the policy and both `scrub_str` results still print before the run exits.

```bash,name=Runs the same on all three platforms
cargo run --example redaction --features "provider-llamacpp plugin-redaction"
```
