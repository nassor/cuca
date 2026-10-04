# Changelog

## 0.4.0

### Breaking

- `UnifiedResponse` has a new public field, `usage: Option<TokenUsage>`. Code
  that builds a `UnifiedResponse` with a struct literal must add `usage: None`.
  The field is `#[serde(default)]`, so serialized 0.3 responses still
  deserialize.
- `UnifiedResponse::completion_tokens` and `prompt_tokens` change meaning on
  the OpenAI-compatible adapters (OpenAI, DeepSeek, vLLM, LM Studio and the
  llama.cpp chat route). When the server reports usage, they are the provider's
  own token counts: `completion_tokens` is no longer one per `Text`, `Thinking`
  or `ToolCall` block, and `prompt_tokens` is no longer `0`. Without a
  reported usage, and on Gemini, Anthropic, the llama.cpp native `/completion`
  route and the speculative orchestrator, the block-count fallback stays and
  `usage` is `None`. `plugin-telemetry` and `plugin-session-log` now record the
  real counts; `CostPlugin` still uses its tiktoken estimates.

### Added

- Every OpenAI-compatible request body carries
  `"stream_options": {"include_usage": true}`, and `ChatCompletionTranslator`
  reads `usage` (including `completion_tokens_details.reasoning_tokens`) from
  whichever frame carries it. `ChatCompletionTranslator::take_usage` exposes
  it.
- `TokenUsage { prompt_tokens, completion_tokens, reasoning_tokens }`,
  exported from the crate root.
- `CucaClient::generate_stream_with_response` returns the block stream plus a
  `ResponseHandle`; `ResponseHandle::take` yields the call's terminal
  `UnifiedResponse` once the stream has ended. Unlike a plugin's
  `on_response_complete`, the handle belongs to one call, so concurrent calls
  stay apart.
- `UnifiedRequest::tools` reaches the wire in every provider adapter
  (OpenAI-compatible `tools`, Anthropic `tools`, Gemini
  `functionDeclarations`), without a tool-choice override.

### Fixed

- `ChatCompletionTranslator` no longer flushes a tool call on a
  `"finish_reason": null` frame, which emitted calls with `null` arguments.
- A usage frame that also drains a flushed tool call still records its usage.
