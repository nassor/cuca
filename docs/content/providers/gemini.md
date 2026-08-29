+++
title = "Google Gemini"
description = "The Gemini streamGenerateContent adapter: endpoint, required auth, thinking levels, and its no-DONE-marker stream ending."
template = "page.html"
weight = 3
+++

# Google Gemini

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to Google's streamGenerateContent endpoint.</dd>
<dt>You need</dt>
<dd>The <code>provider-gemini</code> feature and an API key. The key is required; there is no keyless mode.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::GoogleGemini</code> or handling tool calls against Gemini.</dd>
</dl>

## Endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-gemini` |
| `ProviderEndpoint` variant | `GoogleGemini` |
| Default base URL | `https://generativelanguage.googleapis.com`, used when the client's base URL is empty |
| Route | `POST {base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse` |

The base URL carries no `/v1` suffix. The `/v1beta` API version segment is part of the route, appended by the adapter, not the base URL.

## Authentication

`x-goog-api-key` header, required on every request. Dispatch fails with `CucaError::Config("gemini requires an api key")` when no key is configured.

## Request body

- `system_instruction: { parts: [{ text }] }` from every `Text` block of every `System` message, joined with a newline; omitted with no system messages. Non-text blocks in system messages are dropped.
- `generationConfig` carries `temperature` and `maxOutputTokens` only when set; the key is omitted entirely when neither is set.
- `contents` holds one entry per non-system message: role `"user"` for `User` and `Tool` messages, `"model"` for `Assistant` messages.

## Tool calls

Gemini's `functionCall` parts carry no call id, so the unified `ToolCall::id` is dropped on the wire. A `Tool`-role `ToolResult` message becomes a `"user"`-role turn carrying a `functionResponse` part; the function name comes from the message's `name` annotation, falling back to the block's `tool_call_id` when `name` is unset.

## Thinking

A `thinkingConfig` key is emitted when `req.thinking` is set: `includeThoughts: false` when disabled; otherwise `thinkingBudget` (params override), else `thinkingLevel` (params override, then the unified effort map below), else `includeThoughts: true`.

| Unified `ThinkingEffort` | `thinkingLevel` |
|---|---|
| `Minimal` | `"LOW"` |
| `Low` | `"LOW"` |
| `Medium` | `"MEDIUM"` |
| `High` | `"HIGH"` |
| `XHigh` | `"HIGH"` |

## Streaming end

Gemini sends no `[DONE]` marker. The final frame carries only `usageMetadata`, which translates to no blocks, and the stream ends when the underlying byte stream ends.

## Out of scope

Non-streaming `generateContent` batching, and Google auth beyond the `x-goog-api-key` header. OAuth for GCP service accounts is not implemented.
