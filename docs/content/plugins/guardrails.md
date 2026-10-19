+++
title = "Guardrails"
description = "The JSON Schema output guardrail plugin: schema keys, the retry protocol, and the tracked-call cap."
template = "page.html"
weight = 5
+++

# Guardrails

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>JsonGuardrailPlugin</code> validates tool call arguments and JSON text responses against caller-registered JSON Schemas, and re-injects a diagnostic on failure instead of erroring the stream.</dd>
<dt>You need</dt>
<dd>The <code>plugin-guardrails</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>JsonGuardrailPlugin</code> or reading its retry behavior.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-guardrails` (enables `dep:jsonschema` and `dep:tracing`) |
| Plugin name | `"json-guardrails"` |
| Entry type | `JsonGuardrailPlugin` |
| Attachment | `CucaClientBuilder::register_plugin` |
| Needs | Nothing running; schemas come from a JSON file or from code |

## Hooks

`JsonGuardrailPlugin` acts on `on_stream_chunk`. It does not override `on_request` or `execute_local_tool`, and its `on_response_complete` is a no-op.

- `ToolCall` blocks validate against the schema keyed by the tool name. A tool with no registered schema passes through unvalidated.
- A failing `ToolCall` becomes a `ToolResult` carrying `{"error": ..., "tool": ..., "issues": [...]}`, so the model sees the diagnostic and can retry.
- `Text` blocks that start with `{` (after leading whitespace) validate against the reserved `"response"` schema when one is registered. A failing text block is replaced with the same error JSON, with `error` always `"schema_validation_failed"` and `tool` set to `"response"`. Text blocks are not attempt-counted and emit no tracing event.
- All other blocks pass through.
- `JsonGuardrailPlugin::validate(tool_name, value)` is the direct check. It returns `Err` with one issue string per violation.

### Retry protocol

- The plugin counts attempts per tool call id. Each failing block increments the count, and a valid block for that id clears it.
- `error` is `"schema_validation_failed"` while the count is at or below `max_attempts`, and `"guardrail_exhausted"` once it exceeds it. With `max_attempts` of `0`, the first failure is already `"guardrail_exhausted"`.
- Each injected `ToolResult` emits a `tracing::warn!` event under target `cuca::guardrails` with fields `schema_name`, `error_type` and `attempt_count`.
- `last_retry_event()` returns the most recent `(schema_name, error_type, attempt_count)`, or `None` before the first injection.

## Configuration

`JsonGuardrailPlugin` has no config struct. Two constructors set the schemas and the retry bound.

| Constructor | Schemas | `max_attempts` |
|---|---|---|
| `JsonGuardrailPlugin::new(schema_path)` | A JSON object of tool name to JSON Schema, read from `schema_path`. | `3` |
| `JsonGuardrailPlugin::with_schemas(schemas, max_attempts)` | A `HashMap<String, serde_json::Value>`. | caller-supplied (`u32`) |

- Schemas are keyed by tool name. The reserved key `"response"` guards JSON text responses.
- Every schema compiles at construction. The set is fixed afterward.

## Caps and at-cap policy

| Item | Value |
|---|---|
| Bound | `JsonGuardrailPlugin::MAX_TRACKED_CALLS`, 4096 tool call ids tracked for their attempt count. |
| At the bound | The oldest tracked id is evicted, in insertion order. A reused evicted id starts again at attempt 1. |
| Usage gauge | `JsonGuardrailPlugin::tracked_calls()`. |

Only failing calls are tracked, and a valid block removes its id, so a model that passes its schema leaves `tracked_calls()` at `0`.

## Errors

The hook does not return validation failures. It re-injects them as `ToolResult` or `Text` blocks and returns `Ok(())`.

| Variant | When |
|---|---|
| `Io` | `new` cannot open or read `schema_path`. |
| `Validation` | `new` finds a document that is not a JSON object, or either constructor finds a schema that fails to compile. The `schema` field names the file path or the tool name. |
| `Internal` | `on_stream_chunk` finds the attempts mutex or the retry-event mutex poisoned. |

## Example

`examples/guardrails.rs` runs one `book_flight` tool through three guardrails: a policy equal to the tool's schema, a policy that also requires a `passenger`, and that same policy with `max_attempts` of `0`.

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

The example needs a `llama-server` on port 1234 with a tool-calling model loaded. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server.

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

Both diagnostics carry the same `issues`, so the retry budget is the only change between the last two turns. The tool-call ids and thinking counts depend on the model, as does whether it emits a valid ISO date; the diagnostic shapes and attempt counts do not. The output above came from `google/gemma-4-12b-qat`.

## Related pages

- [Plugins](@/plugins/_index.md): hook order, error handling and the plugin tier rules.
- [Error types](@/reference/errors.md): `PluginError` and `CucaError`.
- [Write a custom plugin](@/guides/custom-plugin.md): the hook contract the guardrails follow.
