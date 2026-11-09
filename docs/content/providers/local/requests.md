+++
title = "Local requests and responses"
description = "How a UnifiedRequest becomes a prompt for the local engine, what the stream yields, the exact token usage, and how thinking is switched."
template = "page.html"
weight = 2
+++

# Local requests and responses

<dl class="page-facts">
<dt>In one line</dt>
<dd>The request is rendered through the model's chat template; the stream carries <code>Text</code>, <code>Thinking</code> and <code>ToolCall</code> blocks, and usage counts are exact.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature and a loaded <code>LocalModel</code>.</dd>
<dt>Read this if</dt>
<dd>You need to know how a request field reaches the template or the sampler, or what the stream and the usage numbers mean.</dd>
</dl>

## Request translation

`UnifiedRequest` fields map to the model's chat template and the sampler:

- `messages`: `System` and `User` messages carry their `Text` blocks joined with a newline as `content`. An `Assistant` message adds its `Thinking` blocks, concatenated, as `reasoning_content`, and its `ToolCall` blocks as `tool_calls` entries `{id, type: "function", function: {name, arguments}}`. A `Tool` message carries the `tool_call_id` (from the message, else from its first `ToolResult`), an optional `name`, and its `ToolResult` outputs joined with a newline as `content`.
- Tool call `arguments` stay JSON objects and keep their key order, which a DeepSeek-V4.1 prompt writes parameter by parameter. An object, `null` or a string holding an object is accepted; anything else fails the request.
- `tools` reaches the template as `{"type": "function", "function": {name, description, parameters}}` with `parameters` from `input_schema`. The template sees the `tools` variable only when the request declares at least one tool.
- `temperature`, `top_p`, `top_k`, `seed`, `max_tokens` and `stop` apply per request. A sampling field the request leaves unset takes the value [`SamplingConfig` resolves](@/providers/local/configuration.md#samplingconfig). A request `seed` restarts the generator for that request only.
- `stop` strings end the reply before the first occurrence of any of them. The stop text is not emitted, text that might still begin a stop string is held back until it cannot, empty entries are ignored, and `Thinking` and `ToolCall` blocks pass through unchanged.
- `thinking` sets template variables; see [Thinking](#thinking).
- A message with an `ImageBase64` block is rejected: the engine is text only.
- The prompt is rendered with `add_generation_prompt` true. The variables and helpers a template sees are on [Chat templates](@/providers/local/chat-templates.md#variables-and-helpers).

Errors a request can raise are on [Local limits and errors](@/providers/local/limits-and-errors.md#errors).

## Response stream

- The stream yields `Result<MessageContentBlock, CucaError>` items: `Text`, `Thinking` and `ToolCall` blocks. The engine generates a `ToolCall` id as `call_` plus 16 hexadecimal digits, unless the model writes its own (Mistral Small 3.2 writes `[CALL_ID]`).
- How reasoning and tool calls are recognized follows the model's template; see [Reasoning](@/providers/local/chat-templates.md#reasoning) and [Tool calls](@/providers/local/chat-templates.md#tool-calls).
- A malformed tool call becomes a `CucaError::Provider` item with the text `malformed tool call: <body>`, and the stream continues.
- Generation stops at an end-of-generation token, at the first `stop` string, at `max_tokens`, or when the context is full. Token usage counts every token generated, including the one that completed the stop string.
- Dropping the stream stops generation. The work already done stays in the K/V cache and is reusable.

## Token usage

Read the usage per call with `CucaClient::generate_stream_with_response`.

- Usage is exact, not estimated. `prompt_tokens` equals the `read_tokens` plus `write_tokens` of `prompt_cache_usage`, and a `max_tokens` cap is met to the token.
- `prompt_cache_usage.read_tokens` counts the prompt tokens served from the prefix store, `write_tokens` the ones prefilled. K/V is kept in blocks of 16 positions named by the token ids they hold, so `read_tokens` is a multiple of 16 and at most the prompt length minus one, rounded down to a block. The trailing partial block (at most 15 tokens) and the last prompt token are always prefilled.
- Blocks are evicted least recently used when the pool is full, so an unrelated request does not evict a prefix that fits. A request that arrives while another with the same prompt runs reuses its blocks as soon as they are filled.
- A hybrid model (Qwen 3.5) also needs the snapshot of its recurrent state at the position it resumes from. The engine takes one at the block boundary below the last end-of-turn token of every prompt it prefills, so `read_tokens` is that boundary at most.
- `reasoning_tokens` is set only when thinking was on for the rendered prompt: the request enabled it, or the template thinks by default and the request did not disable it. It counts the tokens inside reasoning blocks, markers included.
- On the CPU, a request that reuses a prefix generates what a cold request generates, bit for bit, because prefill and verification run in a row-exact mode. The reason is on [Local models and formats](@/providers/local/models.md#weight-formats).

## Thinking

`thinking` on the request switches the template variables below. Which of them a model's template reads decides what the switch does.

- `enable_thinking` and `thinking` are true when `thinking` is enabled and false when it is disabled or missing, except for a template that reads `reasoning_budget`, below. The effort level is ignored.
- A template that reads `reasoning_effort` (Mistral Small 4) gets `high` when thinking is enabled and `none` otherwise, whatever the effort level.
- A template that reads `reasoning_strength` (Muse Glimmer, which cannot switch reasoning off) gets the effort level as `low` (minimal, low, and a disabled request), `medium`, `high` or `xhigh`. With no effort named, the variable stays undefined and the template's own default applies.
- A template that reads `reasoning_budget` (DeepSeek-V4.1) thinks unless the request disables thinking, so a request with no `thinking` setting reasons. An effort of minimal or low gives `low`, medium, high or xhigh gives `high`, and with no effort the variable stays undefined and the template applies its own default, `high`. The template's `max` has no unified effort.
- `preserve_thinking` is always false.
- Reasoning arrives as `Thinking` blocks. If the rendered prompt ends inside an open reasoning block, generation starts in thinking.
- A model whose template reads none of these variables, such as the DeepSeek-R1 distills, reasons on every request.
- `LocalModel::capabilities()` reports what the model supports, derived from its template and output conventions: `reasoning` (it writes a reasoning block), `can_disable_thinking` (it reasons and the template reads `enable_thinking`, `thinking`, `reasoning_effort` or `reasoning_budget`) and `tools` (it has a tool syntax the output parser understands).
