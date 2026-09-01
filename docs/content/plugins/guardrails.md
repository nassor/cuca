+++
title = "Guardrails"
description = "The JSON Schema output guardrail plugin: schema keys, the retry protocol, and the tracked-call cap."
template = "page.html"
weight = 5
+++

# Guardrails

<dl class="page-facts">
<dt>In one line</dt>
<dd>Validates tool call arguments and text responses against caller-registered JSON Schemas, re-injecting a diagnostic on failure instead of erroring the stream.</dd>
<dt>You need</dt>
<dd>The <code>plugin-guardrails</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>JsonGuardrailPlugin</code> or reading its retry behavior.</dd>
</dl>

`JsonGuardrailPlugin` validates tool call arguments, and optionally model text responses, against caller-registered JSON Schemas as blocks stream in. A failing value is not an error: the plugin replaces the block with a `ToolResult` diagnostic and re-injects it so the model can retry, bounded by `max_attempts` before it emits `"guardrail_exhausted"`. Reach for it to keep a model's structured tool calls honest against a schema without hand-writing retry logic.

```rust,name=Reject a tool call missing a required field
use std::collections::HashMap;
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, JsonGuardrailPlugin};
use serde_json::json;

let schemas = HashMap::from([(
    "make_reservation".to_string(),
    json!({
        "type": "object",
        "required": ["date"],
        "properties": { "date": { "type": "string" } }
    }),
)]);
let guardrails = Arc::new(JsonGuardrailPlugin::with_schemas(schemas, 3)?);

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .register_plugin(Arc::clone(&guardrails) as Arc<dyn CucaPlugin>)
    .build()?;

let mut call = MessageContentBlock::ToolCall {
    id: "call-1".into(),
    name: "make_reservation".into(),
    arguments: json!({}),
};
guardrails.on_stream_chunk(&mut call)?;
```

```text,name=The missing date field re-injects a diagnostic
ToolResult { tool_call_id: "call-1", output: "{\"error\":\"schema_validation_failed\",\"issues\":[\"\\\"date\\\" is a required property\"],\"tool\":\"make_reservation\"}" }
```

## Entry types

`JsonGuardrailPlugin`.

## `CucaPlugin`

`JsonGuardrailPlugin` implements `CucaPlugin` with the plugin name `"json-guardrails"`. It overrides `on_stream_chunk` only.

## Construction

| Constructor | `max_attempts` |
|---|---|
| `JsonGuardrailPlugin::new(schema_path)` | 3, loading a JSON object of tool name to JSON Schema from `schema_path` |
| `JsonGuardrailPlugin::with_schemas(schemas, max_attempts)` | caller-supplied |

Schemas are keyed by tool name. The reserved key `"response"`, when registered, guards model text responses that look like a JSON object (text starting with `{`). A tool with no registered schema passes through unvalidated.

## Retry behavior

An invalid `ToolCall` is replaced with a `ToolResult` carrying `{"error": ..., "tool": ..., "issues": [...]}`. The `error` field is `"schema_validation_failed"` while the tracked attempt count for that call id is at or below `max_attempts`, and `"guardrail_exhausted"` once it is exceeded. Each retry injection also emits a `tracing::warn!` event under target `cuca::guardrails` with `schema_name`, `error_type`, and `attempt_count`.

## Capacity

| | |
|---|---|
| Bound | `JsonGuardrailPlugin::MAX_TRACKED_CALLS`, 4096 tool call ids tracked for their attempt count |
| At-cap policy | The oldest tracked id is evicted, in insertion order |
| Usage gauge | `JsonGuardrailPlugin::tracked_calls()` |
