+++
title = "Redaction"
description = "Hide secrets and personal data in the text of every outgoing request, before it leaves your process."
template = "page.html"
weight = 16
+++

# Redaction

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>RedactionPlugin</code> replaces secrets and personal data in the text of each outgoing request with a <code>[REDACTED:&lt;kind&gt;]</code> marker, using rules you write.</dd>
<dt>You need</dt>
<dd>The <code>plugin-redaction</code> feature and a list of rules. The plugin has no built-in rules.</dd>
<dt>Read this if</dt>
<dd>Prompts or tool output may contain API keys, emails, card numbers or other data that must not reach a provider.</dd>
</dl>

PII means personally identifiable information, such as an email address or a card number. The plugin looks for the patterns your rules describe. It is a text matcher, not a detector: data that no rule matches is sent unchanged.

## Register it

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

Expected effect: the text `notify ops@example.com` leaves your process as `notify [REDACTED:email]`.

## What gets scrubbed

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 190" role="img" aria-labelledby="red-title red-desc">
  <title id="red-title">Redaction scrubs the text of an outgoing request and leaves ids and names alone.</title>
  <desc id="red-desc">The request enters on_request. The plugin scrubs message text, reasoning, tool call arguments, tool results, message names and optionally tool descriptions. It leaves ids, tool names, input schemas, images and settings unchanged. The scrubbed request goes to the provider. Text that comes back from the provider is not scrubbed by this plugin.</desc>
  <defs>
    <marker id="red-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">RedactionPlugin on_request</text>
  <rect class="blk blk-data" x="5" y="40" width="120" height="50" rx="6"/>
  <text class="t-lbl t-mid" x="65" y="62">request</text>
  <text class="t-sm t-mid" x="65" y="78">with secrets</text>
  <rect class="blk blk-ctl" x="190" y="40" width="270" height="130" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="325" y="62">scrub with your rules</text>
  <text class="t-sm t-mid" x="325" y="84">text, reasoning, tool arguments,</text>
  <text class="t-sm t-mid" x="325" y="98">tool results, message name,</text>
  <text class="t-sm t-mid" x="325" y="112">tool descriptions (optional)</text>
  <text class="t-sm t-mid" x="325" y="136">not changed: ids, tool names,</text>
  <text class="t-sm t-mid" x="325" y="150">schemas, images, settings</text>
  <rect class="blk blk-bnd" x="525" y="40" width="130" height="50" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="590" y="62">provider</text>
  <text class="t-sm t-mid" x="590" y="78">sees markers only</text>
  <path class="arw arw-data" d="M126 65 H188" marker-end="url(#red-arrow)"/>
  <path class="arw arw-bnd" d="M461 65 H523" marker-end="url(#red-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>Redaction acts on the way out.</b> It does not touch text that comes back. The model's reply is scrubbed on the next turn, when it returns as a message in the request.</p>
</div>

## Run the example

`examples/redaction.rs` builds the four-rule policy, scrubs a clean and a dirty sample directly, sends a seeded prompt through a live turn, and prints the counters. It needs a `llama-server` on port 1234. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example redaction --features "provider-llamacpp plugin-redaction"
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

The reply text and the thinking block count depend on the model. The four redactions and the counters do not.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-redaction` |
| Entry types | `RedactionPlugin`, `RedactionConfig`, `RedactionRule`, `Redacted` |
| Plugin name | `"redaction"` |
| Attachment | `register_plugin` |
| Needs | A non-empty rule list that you write |
| Direct call | `RedactionPlugin::scrub_str(&self, text: &str) -> Result<Redacted<'_>, PluginError>` |

`scrub_str` runs the same matcher as the hook. Use it to clean a value before it enters a request, or to reject text when `count > 0`. A clean string returns `Redacted { text: Cow::Borrowed(text), count: 0 }` and allocates nothing. `scrub_str` writes no log event and changes no counter.

## Hooks

`RedactionPlugin` acts on `on_request` only. It runs the matcher over each field below, in message order, then over tool descriptions. It replaces each match with `[REDACTED:{kind}]`, so the provider never sees the original value.

### Scrubbed fields

| Location | `field` label | Action |
|---|---|---|
| `MessageContentBlock::Text` | `message_text` | Scrub the text. |
| `MessageContentBlock::Thinking`, `reasoning` | `message_thinking` | Scrub the reasoning. |
| `MessageContentBlock::ToolCall`, `arguments` | `tool_call_arguments` | Scrub every JSON string value, to depth `MAX_JSON_DEPTH`. |
| `MessageContentBlock::ToolResult`, `output` | `tool_result_output` | Scrub the output. |
| `UnifiedMessage::name` | `message_name` | Scrub the name, if there is one. |
| `ToolDefinition::description` | `tool_description` | Scrub when `scrub_tool_definitions` is `true`. |

### Fields left unchanged

- `Thinking::signature`: it vouches for the reasoning, and changing it makes it invalid.
- `ToolCall::id`, `ToolResult::tool_call_id` and `UnifiedMessage::tool_call_id`: they pair calls with results.
- `ToolCall::name` and `ToolDefinition::name`: a renamed tool cannot be routed.
- `ToolDefinition::input_schema`: it is the tool's contract. [Guardrails](@/plugins/guardrails.md) checks against the same schemas.
- `ImageBase64`: a match inside base64 is a coincidence, and changing it breaks the image.
- JSON object keys inside `arguments`.
- Settings such as `model`, `provider`, `temperature`, `max_tokens`, `top_p`, `top_k`, `seed`, `stop`, `stream`, `thinking` and `prompt_cache`.

### Text that comes back

`on_stream_chunk` is not implemented, so a secret in a streamed reply comes back unchanged. Providers stream text in fragments, so a secret split across two blocks would be missed. Reliable matching would need buffering across blocks, which conflicts with streaming latency.

The reply is scrubbed on the next turn, because it re-enters the conversation as a message in the request. Scrubbing the same text twice changes nothing more, so long conversations settle.

### Order with other plugins

The plugin needs no particular position. `on_request` hooks run in registration order on one shared request, so plugins registered after it see the scrubbed request.

- With [Session log](@/plugins/session-log.md), register `RedactionPlugin` first to store scrubbed content. Registered second, the log stores the raw value, on disk if you use `FileBackend`. Model output is never scrubbed in either order.
- With `service-prompt-cache`, the cache key is computed after every `on_request` hook, so redaction changes every key. Two requests that differ only inside a redacted secret share one key.

### Log events and counters

Each replacement logs one `tracing::warn!` with the message `outbound value redacted`, under the target `cuca::redaction`. Its fields are `kind`, `field` and `message_index`. It never holds the matched text.

`message_index` is the position in `req.messages`, except for `tool_description`, where it is the position in `req.tools`.

- `last_request_redactions()` returns the `u64` count from the latest `on_request`, or `0` before the first.
- `total_redactions()` returns the running count across requests.
- `last_redaction_event()` returns `Option<(String, &'static str, usize)>`: the `kind`, the field label and the match count of the last field that matched.

### What it does not do

- It does not detect secrets. A secret with no rule is sent.
- A false positive, such as a `DigitRun` rule matching an order number, comes from your policy. You can see it in the log event.
- It does not encrypt, and it does not clean data that is already stored or exported.

## Configuration

`RedactionConfig::new(rules)` checks the policy and rejects an unusable one. `validate()` checks again, because the fields are public and a struct literal skips `new`. `RedactionPlugin::new` also checks.

| Key | Type | Default |
|---|---|---|
| `rules` | `Vec<RedactionRule>` | none, required, non-empty |
| `max_matches_per_text` | `usize` | `1024` |
| `scrub_tool_definitions` | `bool` | `true` |

- `rules` is the policy. Its order breaks ties between overlapping matches.
- `max_matches_per_text` limits matches in one string. Going over it fails the hook instead of redacting part of the text.
- `scrub_tool_definitions` sets whether `ToolDefinition::description` is scrubbed.

`RedactionConfig::default()` has no rules, so `RedactionPlugin::new` rejects it. It exists for `..Default::default()` struct updates.

### Rules

`RedactionRule` has four variants. Each has a `kind`, which names the replacement marker and the log event. It is never the matched value.

| Variant | Matches |
|---|---|
| `Literal { kind, value }` | An exact, byte-for-byte copy of a known secret. |
| `Prefixed { kind, prefix, min_len, max_len }` | `prefix`, then `min_len` to `max_len` characters from letters, digits, `-` and `_`. Longer runs stop at `max_len`. Example: `prefix = "sk-"`. |
| `EmailLike { kind }` | An ASCII `local@domain.tld` address. The local part starts with a letter or digit and may hold `.`, `_`, `%`, `+` and `-`. The domain needs a dot that is not its first character. A trailing `.` or `-` stays outside the match. |
| `DigitRun { kind, min_digits, max_digits }` | `min_digits` to `max_digits` ASCII digits, optionally split by single `-` or space characters. It fits card, phone and national ID numbers. It ends on a digit and stops at `max_digits`. |

When matches overlap, the leftmost and longest wins. Ties go to the rule that comes first in the policy, so the output depends only on the policy and the input.

### Validation limits

`RedactionConfig::new`, `validate()` and `RedactionPlugin::new` reject a bad policy with `PluginError::Validation { schema: "redaction-config", .. }`.

| Limit | Constant | Rejects |
|---|---|---|
| Rule count | `MAX_RULES` (256) | Zero rules, or more than the limit. |
| Pattern length | `MAX_PATTERN_BYTES` (512) | An empty or too long `Literal::value` or `Prefixed::prefix`. |
| `kind` name | `MAX_KIND_BYTES` (32) | A `kind` that is empty, too long, or not made of `[a-z0-9_-]`. |
| Replacement marker | `REDACTION_MARKER` (`"[REDACTED:"`) | A `Literal::value` that contains the marker. |
| `Prefixed` range | none | `min_len > max_len`. |
| `DigitRun` range | none | `min_digits == 0`, or `min_digits > max_digits`. |
| Match limit | `MAX_MATCHES_PER_TEXT` (4096) | `max_matches_per_text == 0`, or above the limit. |

The marker rule keeps scrubbing repeatable: no rule can match the plugin's own marker. `MAX_JSON_DEPTH` (64) limits how deep `ToolCall::arguments` may nest. All constants are associated constants on `RedactionConfig`.

## Caps and at-cap policy

Nothing in this plugin grows with traffic.

- `rules`: `MAX_RULES` (256), each pattern `MAX_PATTERN_BYTES` (512), each `kind` `MAX_KIND_BYTES` (32). The set is fixed when the plugin is built, and an empty or too large policy is rejected. Read it with `rule_count()`.
- Matches in one string: `max_matches_per_text`, at most `MAX_MATCHES_PER_TEXT` (4096). At the limit the call returns `PluginError::HookFailure`. Read it with `match_cap()`.
- Statistics: two `u64` counters and one `(kind, field, count)` tuple. The tuple is replaced, never added to. Read them with `last_request_redactions()`, `total_redactions()` and `last_redaction_event()`.
- `ToolCall::arguments` depth: `MAX_JSON_DEPTH` (64). Past it the hook returns `PluginError::HookFailure`. Every tool call is checked before anything is rewritten. No usage reading.

A refusal makes `on_request` fail, so the request never reaches a provider and later plugins' hooks do not run. A refusal for too many matches may leave fields scrubbed in memory that were scrubbed earlier in the same pass. The request is refused as a whole, not sent half scrubbed.

## Errors

| Variant | When |
|---|---|
| `Validation`, schema `redaction-config` | `RedactionConfig::new`, `validate()` or `RedactionPlugin::new` rejects the policy. The message names the field, such as `rules[3].kind` or `max_matches_per_text`. |
| `HookFailure`, plugin `"redaction"`, stage `"request"` | A string has more than `match_cap()` matches, from `scrub_str` or `on_request`. |
| `HookFailure`, same plugin and stage | A `ToolCall` `arguments` value nests deeper than `MAX_JSON_DEPTH`, from `on_request`. |

## Related pages

- [Session log](@/plugins/session-log.md): registration order decides whether the log stores scrubbed or raw content.
- [Prompt cache](@/services/prompt-cache.md): redaction changes every cache key.
- [Guardrails](@/plugins/guardrails.md): checks tool calls against the schemas redaction leaves unchanged.
- [Memory discipline](@/concepts/memory-discipline.md): why every limit here is a hard refusal.
