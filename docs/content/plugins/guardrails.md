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
