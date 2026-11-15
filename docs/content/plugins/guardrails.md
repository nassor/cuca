+++
title = "Guardrails"
description = "Check tool arguments and JSON replies against a JSON Schema, and tell the model what to fix."
template = "page.html"
weight = 5
+++

# Guardrails

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>JsonGuardrailPlugin</code> checks a tool call's arguments, and JSON text replies, against JSON Schemas you register. When a check fails, it tells the model what was wrong so it can retry.</dd>
<dt>You need</dt>
<dd>The <code>plugin-guardrails</code> feature. Nothing else runs.</dd>
<dt>Read this if</dt>
<dd>You want to be sure a tool call or structured reply has the shape your code expects.</dd>
</dl>

A JSON Schema is a JSON document that describes which JSON values are valid: required fields, types, patterns. Models sometimes send tool arguments that break the schema. This plugin catches that before your code runs the tool.

## Register it

```rust,name=Register a guardrail with a retry budget of three
use std::collections::HashMap;

let guardrails = Arc::new(JsonGuardrailPlugin::with_schemas(
    HashMap::from([("book_flight".to_string(), schema)]),
    3,
)?);
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url)
    .register_plugin(Arc::clone(&guardrails) as Arc<dyn CucaPlugin>)
    .build()?;
```

Expected effect: a `book_flight` call that breaks `schema` is replaced in the stream by a `ToolResult` such as `{"error":"schema_validation_failed","issues":[...],"tool":"book_flight"}`. The model reads it and can call the tool again. A valid call passes through unchanged.

## How a failed check is handled

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 204" role="img" aria-labelledby="grd-title grd-desc">
  <title id="grd-title">How a tool call is checked and retried.</title>
  <desc id="grd-desc">A tool call from the model is checked against its schema. A valid call passes through and its attempt count is cleared. An invalid call is replaced by a ToolResult with the issues. While attempts are at or below max_attempts the error is schema_validation_failed, and the model can retry. Past max_attempts the error is guardrail_exhausted.</desc>
  <defs>
    <marker id="grd-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">on_stream_chunk for a ToolCall</text>
  <rect class="blk blk-data" x="5" y="40" width="130" height="50" rx="6"/>
  <text class="t-lbl t-mid" x="70" y="62">ToolCall</text>
  <text class="t-sm t-mid" x="70" y="78">from the model</text>
  <rect class="blk blk-ctl" x="190" y="40" width="130" height="50" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="255" y="62">check schema</text>
  <text class="t-sm t-mid" x="255" y="78">by tool name</text>
  <rect class="blk blk-data" x="400" y="40" width="130" height="50" rx="6"/>
  <text class="t-lbl t-mid" x="465" y="62">valid</text>
  <text class="t-sm t-mid" x="465" y="78">pass, clear count</text>
  <rect class="blk blk-ctl" x="190" y="126" width="130" height="50" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="255" y="148">invalid</text>
  <text class="t-sm t-mid" x="255" y="164">count attempt</text>
  <rect class="blk blk-data" x="400" y="100" width="255" height="42" rx="6"/>
  <text class="t-sm t-mid" x="527" y="117">attempts up to max_attempts</text>
  <text class="t-sm t-mid t-data" x="527" y="132">schema_validation_failed</text>
  <rect class="blk blk-data" x="400" y="152" width="255" height="42" rx="6"/>
  <text class="t-sm t-mid" x="527" y="169">attempts over max_attempts</text>
  <text class="t-sm t-mid t-data" x="527" y="184">guardrail_exhausted</text>
  <path class="arw arw-data" d="M136 65 H188" marker-end="url(#grd-arrow)"/>
  <path class="arw arw-data" d="M321 65 H398" marker-end="url(#grd-arrow)"/>
  <path class="arw arw-ctl" d="M255 92 V124" marker-end="url(#grd-arrow)"/>
  <path class="arw arw-data" d="M321 142 H350 V121 H398" marker-end="url(#grd-arrow)"/>
  <path class="arw arw-data" d="M321 160 H350 V173 H398" marker-end="url(#grd-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>A failed check does not fail the stream.</b> A bad call becomes a ToolResult with the issues, and the model can send fixed arguments. After more than <code>max_attempts</code> failures for one call id, the error text changes to <code>guardrail_exhausted</code>.</p>
</div>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-guardrails` |
| Plugin name | `"json-guardrails"` |
| Entry type | `JsonGuardrailPlugin` |
| Attachment | `CucaClientBuilder::register_plugin` |
| Needs | Schemas from a JSON file or from code |

## Hooks

`JsonGuardrailPlugin` acts on `on_stream_chunk` only. `on_request` and `on_response_complete` do nothing.

- A `ToolCall` is checked against the schema registered under its tool name. A tool with no schema passes through unchecked.
- A failing `ToolCall` becomes a `ToolResult` holding `{"error": ..., "tool": ..., "issues": [...]}`.
- A `Text` block that starts with `{` (after any spaces) is checked against the schema registered as `"response"`, if there is one. A failing block is replaced by the same error JSON, with `error` always `"schema_validation_failed"` and `tool` set to `"response"`. Text blocks are not counted as attempts.
- Other blocks pass through.
- `JsonGuardrailPlugin::validate(tool_name, value)` checks a value directly. It returns `Err` with one issue string per violation.

### Retries

- The plugin counts failed attempts per tool call id. A valid block for that id clears the count.
- `error` is `"schema_validation_failed"` while the count is at or below `max_attempts`, and `"guardrail_exhausted"` above it. With `max_attempts` of `0`, the first failure is already `"guardrail_exhausted"`.
- Each injected `ToolResult` logs a `tracing::warn!` event under target `cuca::guardrails`, with fields `schema_name`, `error_type` and `attempt_count`.
- `last_retry_event()` returns the latest `(schema_name, error_type, attempt_count)`, or `None` before the first one.

## Configuration

`JsonGuardrailPlugin` has no config struct. Two constructors set the schemas and the retry limit:

- `JsonGuardrailPlugin::new(schema_path)` reads the schemas from a JSON file, an object of tool name to JSON Schema. `max_attempts` is `3`.
- `JsonGuardrailPlugin::with_schemas(schemas, max_attempts)` takes a `HashMap<String, serde_json::Value>`. You pass `max_attempts` as a `u32`.
- Schemas are keyed by tool name. The key `"response"` is reserved for JSON text replies.
- Every schema is compiled when you construct the plugin. The set cannot change afterward.

## Caps and at-cap policy

| Item | Value |
|---|---|
| Limit | `JsonGuardrailPlugin::MAX_TRACKED_CALLS`, 4096 tool call ids tracked for attempt counts. |
| At the limit | The oldest tracked id is evicted. If that id comes back, it starts again at attempt 1. |
| Usage gauge | `JsonGuardrailPlugin::tracked_calls()` |

Only failing calls are tracked, and a valid block removes its id. A model that passes its schema leaves `tracked_calls()` at `0`.

## Errors

Validation failures are not returned as errors. They go back to the model as blocks, and the hook returns `Ok(())`.

| Variant | When |
|---|---|
| `Io` | `new` cannot open or read `schema_path`. |
| `Validation` | `new` finds a document that is not a JSON object, or a constructor finds a schema that does not compile. The `schema` field names the file path or the tool name. |
| `Internal` | `on_stream_chunk` finds its attempts lock or its retry-event lock poisoned. |

## Run the example

`examples/guardrails.rs` runs one `book_flight` tool through three policies: one equal to the tool's schema, one that also requires a `passenger`, and the same with `max_attempts` of `0`. It needs a `llama-server` on port 1234 with a tool-calling model loaded. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example guardrails --features "provider-llamacpp plugin-guardrails"
```

```text,name=Expected output
Tool book_flight advertises: {"additionalProperties":false,"properties":{"date":{"pattern":"^\\d{4}-\\d{2}-\\d{2}$","type":"string"},"destination":{"type":"string"}},"required":["destination","date"],"type":"object"}
Guardrail policy for the last two turns: {"properties":{"date":{"pattern":"^\\d{4}-\\d{2}-\\d{2}$","type":"string"},"destination":{"type":"string"},"passenger":{"type":"string"}},"required":["destination","date","passenger"],"type":"object"}

Policy matches the tool, retry budget 3: "Book a flight to Lisbon for 2026-03-14 with the book_flight tool."
  thinking blocks: 117
  ToolCall call="0UNMlIsYtMPi2o1N6r910nE645OFrWuD" name="book_flight" passed the schema
    {"date":"2026-03-14","destination":"Lisbon"}
  tracked_calls: 0
  last_retry_event: none, nothing was rewritten

Policy demands a passenger, retry budget 3: "Book a flight to Porto for 2026-04-02 with the book_flight tool."
  thinking blocks: 87
  ToolResult call="bgzkJZOykk62A9zcSTMnE3X7YHgXQEhe"
    {"error":"schema_validation_failed","issues":["\"passenger\" is a required property"],"tool":"book_flight"}
  tracked_calls: 1
  last_retry_event: schema "book_flight", error "schema_validation_failed", attempt 1

Same policy, retry budget 0: "Book a flight to Faro for 2026-05-20 with the book_flight tool."
  thinking blocks: 107
  ToolResult call="g1zw5RS7PUVPuVM5o4xM7or2ZerArLPu"
    {"error":"guardrail_exhausted","issues":["\"passenger\" is a required property"],"tool":"book_flight"}
  tracked_calls: 1
  last_retry_event: schema "book_flight", error "guardrail_exhausted", attempt 1
```

Tool call ids and thinking counts depend on the model. The error shapes and attempt counts do not.

## Related pages

- [Plugins](@/plugins/_index.md): the hooks, their order and the error rules.
- [Error types](@/reference/errors.md): `PluginError` and `CucaError`.
- [Write a custom plugin](@/guides/custom-plugin.md): the hook contract the guardrails follow.
