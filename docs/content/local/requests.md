+++
title = "Requests, thinking and tool calls"
description = "What each request field does on the local engine, what the reply stream and the token counts mean, how thinking is switched, which tool-call formats are read, and how to replace a chat template."
template = "page.html"
aliases = ["providers/local/requests/"]
weight = 4
+++

# Requests, thinking and tool calls

<dl class="page-facts">
<dt>In one line</dt>
<dd>The request becomes a prompt through the model's chat template, and the reply comes back as <code>Text</code>, <code>Thinking</code> and <code>ToolCall</code> blocks.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature and a loaded <code>LocalModel</code>; see <a href="@/local/_index.md">Local (in-process)</a>.</dd>
<dt>Read this if</dt>
<dd>You want to send tools or thinking to a local model, read the token counts, or replace a model's chat template.</dd>
</dl>

## Send a request with a tool

A request is a normal `UnifiedRequest`. This one declares a tool, turns thinking off and prints what arrives. Run it inside an async function that returns a `Result`, as `examples/local_model.rs` does.

```rust,name=Ask a local model to call a tool
use cuca::types::{MessageContentBlock, ToolDefinition};
use cuca::{LocalModel, LocalModelConfig, UnifiedRequest};
use tokio_stream::StreamExt;

let model = LocalModel::load(LocalModelConfig::default()).await?;
let request = UnifiedRequest::new("local")
    .add_user_message("What is the weather in Paris? Use the tool.")
    .add_tool(ToolDefinition {
        name: "get_weather".into(),
        description: "Current weather for a city".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    })
    .disable_thinking()
    .set_max_tokens(256);

let mut stream = model.generate(request).await?;
while let Some(block) = stream.next().await {
    match block? {
        MessageContentBlock::Text(text) => print!("{text}"),
        MessageContentBlock::ToolCall { name, arguments, .. } => println!("call {name} {arguments}"),
        _ => {}
    }
}
```

```text,name=Expected output (the exact text depends on the model)
call get_weather {"city":"Paris"}
```

## From request to blocks

A model reads and writes plain text. Two pieces translate. The chat template (a Jinja text that ships with the model) turns your messages into the exact prompt format the model was trained on. The output parser reads the generated text and splits it into blocks.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 204" role="img" aria-labelledby="req-title req-desc">
  <title id="req-title">A request becomes a prompt, and generated text becomes content blocks</title>
  <desc id="req-desc">Top row, left to right: a UnifiedRequest holding messages, tools, thinking and sampling goes through the chat template, which renders prompt text, into the engine, which tokenizes, prefills and decodes. The generated text returns to a second row, where the output parser reads the template's reasoning markers and tool-call syntax and emits Text, Thinking and ToolCall blocks.</desc>
  <defs>
    <marker id="req-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">One request through the local engine</text>
  <rect class="blk blk-ctl" x="6" y="36" width="164" height="58" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="88" y="56">UnifiedRequest</text>
  <text class="t-sm t-mid" x="88" y="72">messages, tools,</text>
  <text class="t-sm t-mid" x="88" y="85">thinking, sampling</text>
  <path class="arw arw-ctl" d="M172 68 H220" marker-end="url(#req-arrow)"/>
  <text class="t-ax t-mid t-ctl" x="196" y="60">render</text>
  <rect class="blk blk-ctl" x="222" y="36" width="164" height="58" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="304" y="56">Chat template</text>
  <text class="t-sm t-mid" x="304" y="72">Jinja in the model file</text>
  <text class="t-sm t-mid" x="304" y="85">or your override</text>
  <path class="arw arw-data" d="M388 68 H436" marker-end="url(#req-arrow)"/>
  <text class="t-ax t-mid t-data" x="412" y="60">prompt</text>
  <rect class="blk" x="438" y="36" width="216" height="58" rx="6"/>
  <text class="t-lbl t-mid" x="546" y="56">Engine</text>
  <text class="t-sm t-mid" x="546" y="72">tokenize, prefill, decode</text>
  <text class="t-sm t-mid" x="546" y="85">sampling picks each token</text>
  <path class="arw arw-data" d="M546 96 V114 H88 V132" marker-end="url(#req-arrow)"/>
  <text class="t-ax t-mid t-data" x="317" y="109">generated text</text>
  <text class="t-ax" x="224" y="128">stream items: MessageContentBlock</text>
  <rect class="blk blk-ctl" x="6" y="134" width="164" height="58" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="88" y="154">Output parser</text>
  <text class="t-sm t-mid" x="88" y="170">reads the template's</text>
  <text class="t-sm t-mid" x="88" y="183">markers and tool syntax</text>
  <path class="arw arw-data" d="M172 163 H220" marker-end="url(#req-arrow)"/>
  <rect class="blk blk-data" x="222" y="134" width="134" height="58" rx="6"/>
  <text class="t-lbl t-mid t-data" x="289" y="160">Text</text>
  <text class="t-sm t-mid" x="289" y="176">the answer</text>
  <rect class="blk blk-data" x="364" y="134" width="134" height="58" rx="6"/>
  <text class="t-lbl t-mid t-data" x="431" y="160">Thinking</text>
  <text class="t-sm t-mid" x="431" y="176">the reasoning</text>
  <rect class="blk blk-data" x="506" y="134" width="134" height="58" rx="6"/>
  <text class="t-lbl t-mid t-data" x="573" y="160">ToolCall</text>
  <text class="t-sm t-mid" x="573" y="176">name and arguments</text>
</svg>
</div>
<p class="dgm-cap"><b>The template decides both directions.</b> It writes the prompt, and the parser reads reasoning markers and tool-call syntax from the same template, so the engine holds no model names.</p>
</div>

## Request fields

- `messages`: `System` and `User` messages send their `Text` blocks joined with a newline. An `Assistant` message adds its `Thinking` blocks as `reasoning_content` and its `ToolCall` blocks as `tool_calls`. A `Tool` message carries the `tool_call_id` and its `ToolResult` outputs joined with a newline.
- Tool call `arguments` stay JSON objects with their key order. An object, `null` or a string holding an object is accepted; anything else fails the request with ``tool call `NAME` arguments must be a JSON object``.
- `tools`: each tool reaches the template as a function with `name`, `description` and `parameters` (from `input_schema`). The template sees the `tools` variable only when the request declares at least one tool.
- `temperature`, `top_p`, `top_k`, `seed` and `max_tokens` apply to this request. A sampling field you leave unset takes the value [`SamplingConfig` resolves](@/local/configuration.md#samplingconfig). A request `seed` restarts the generator for that request only. A `temperature` below 0 or not finite, or a `top_p` outside `(0, 1]`, is a `CucaError::Config`.
- `stop`: the reply ends before the first occurrence of any stop string, and the stop text is not emitted. Text that might still begin a stop string is held back until it cannot. Empty strings are ignored. `Thinking` and `ToolCall` blocks pass through unchanged.
- `thinking`: sets template variables; see [Thinking](#thinking).
- An `ImageBase64` block is rejected: the engine reads text only.
- The prompt always ends with the generation prompt, so the model continues as the assistant.

Errors a request can raise are on [Limits and errors](@/local/limits-and-errors.md#errors).

## The reply stream

- The stream yields `Result<MessageContentBlock, CucaError>`: `Text`, `Thinking` and `ToolCall` blocks.
- A `ToolCall` gets the id `call_` plus 16 hexadecimal digits, unless the model writes its own id (Mistral Small 3.2 does).
- Generation stops at the model's end-of-turn token, at the first `stop` string, at `max_tokens`, or when the context is full.
- A malformed tool call is a `CucaError::Provider` item with the text `malformed tool call: <body>`. The stream continues after it.
- Dropping the stream stops generation. The work already done stays in the K/V cache (the engine's saved work for the prompt so far) and later requests reuse it.

## Token usage

Read the counts per call with `CucaClient::generate_stream_with_response`, or with `stream.report()` on a stream from `LocalModel::generate`.

- The counts are exact, not estimated. A `max_tokens` cap is met to the token, and `completion_tokens` includes the token that completed a stop string.
- `prompt_cache_usage` splits the prompt: `read_tokens` were served from the prefix cache, `write_tokens` were computed. `prompt_tokens` equals the two added.
- The prefix cache works in blocks of 16 tokens. `read_tokens` is always a multiple of 16 and at most the prompt length minus one, rounded down to a block. The last prompt token and the trailing partial block (up to 15 tokens) are always computed.
- When the cache is full, the least recently used block goes first, so an unrelated request does not evict a prefix that still fits. A request that arrives while another with the same prompt runs reuses its blocks as soon as they are filled.
- A hybrid model (Qwen 3.5) resumes only at the block boundary below the last end-of-turn token of an earlier prompt, so its `read_tokens` is at most that boundary.
- `reasoning_tokens` is set only when thinking was on for the rendered prompt. It counts the tokens inside reasoning blocks, markers included.
- On the CPU, a request that reuses a prefix generates the same tokens as a cold request. `batching prefix_cache=#false` turns the cache off; see [Configuration](@/local/configuration.md#batchingconfig).

## Thinking

Some models write their reasoning before the answer. It arrives as `Thinking` blocks. The `thinking` field of the request sets the template variable below, and which variable a model's template reads decides what the switch does.

| Model family | Thinking | Variable the template reads |
|---|---|---|
| Gemma 4, Qwen 3, Qwen 3.5 and 3.6, SmolLM 3 | switchable, off unless the request enables it | `enable_thinking` |
| Granite 3.3 | switchable, off unless the request enables it | `thinking` |
| Mistral Small 4 | switchable | `reasoning_effort`: `high` when enabled, `none` otherwise |
| DeepSeek-V4.1 | on unless the request disables it | `reasoning_budget`: `low` or `high` |
| Muse Glimmer | always, cannot be switched off | `reasoning_strength`: `low`, `medium`, `high` or `xhigh` |
| Ministral 3 Reasoning, the DeepSeek-R1 distills | always | none |
| Gemma 3, Phi 4 mini, OLMo 2, OLMoE, Ministral 3 Instruct | never | none |

- The effort level (`minimal`, `low`, `medium`, `high`, `xhigh`) matters only for `reasoning_strength` and `reasoning_budget`.
- `reasoning_strength` gets `low` for minimal, low or a disabled request, and `medium`, `high` or `xhigh` for the same-named level. With no effort named, the variable stays undefined and the template's own default applies (`high` for Muse Glimmer).
- `reasoning_budget` gets `low` for minimal or low, and `high` for medium, high or xhigh. With no effort named, the template's default `high` applies. The template's `max` level cannot be requested.
- `enable_thinking` and `thinking` are both true when thinking is enabled and both false when it is disabled or missing. For a DeepSeek-V4.1 template they are true unless the request disables thinking.
- `preserve_thinking` is always false.
- If the rendered prompt ends inside an open reasoning block, generation starts in thinking.
- Give a reasoning model a generous `max_tokens`, or it can spend the whole budget before it answers. See [Known limits](@/local/verification.md#known-limits).
- `LocalModel::capabilities()` reports what a loaded model supports: `reasoning`, `can_disable_thinking` and `tools`.

The parser finds a model's reasoning markers from its template and vocabulary: `<think>` and `</think>`, `[THINK]` and `[/THINK]`, Gemma 4's channel markers, or Muse Glimmer's messages addressed to `self`.

## Tool calls

A model writes a tool call as text, in the format it was trained on. The parser reads it and returns one `ToolCall` block per call. It detects the format from the chat template and the vocabulary when the model loads.

| Model family | Format | Notes |
|---|---|---|
| Mistral Small 4, Mistral Small 3.2, Devstral, Ministral 3 | `[TOOL_CALLS]` | one marker per call, no closing marker; Mistral Small 3.2 adds `[CALL_ID]` and its own id |
| Gemma 4 | control tokens | `call:name{key:value}` between two control tokens |
| Qwen 3.5 and 3.6 | XML | `<function=name>` with one `<parameter=key>` per argument; a value that is valid JSON is read as JSON |
| Qwen 3, SmolLM 3 | tagged JSON | `<tool_call>` around one call object or a list; with no closing tag the call runs to the end of the reply |
| Granite 3.3 | tagged JSON | a `tool_call` marker, then a JSON list to the end of the reply |
| DeepSeek-V4.1 | DSML | a block after the answer; a `string` attribute says whether a value is text or JSON; a repeated parameter name is a malformed call |
| Muse Glimmer | channel protocol | a message addressed to the tool name, with `atem:invoke` and `atem:parameter` elements |
| Gemma 3, Phi 4 mini, OLMo 2, OLMoE, ERNIE 4.5, the DeepSeek-R1 distills | none | plain text; `capabilities().tools` is false |

A model whose template has no `tools` variable never sees the tools you declare. This is what each format looks like for one call, `get_weather(city="Paris")`:

```text,name=The raw text a model writes for one call
Mistral:       [TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}
Mistral 3.2:   [TOOL_CALLS]get_weather[CALL_ID]abc123xyz[ARGS]{"city": "Paris"}
Gemma 4:       <|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>
Qwen 3, SmolLM 3:
               <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
Granite 3.3:   <|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]
Qwen 3.5 and 3.6:
               <tool_call>
               <function=get_weather>
               <parameter=city>
               Paris
               </parameter>
               </function>
               </tool_call>
DeepSeek-V4.1:
               <｜DSML｜ calls>
               <｜DSML｜ invoke name="get_weather">
               <｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
               </｜DSML｜ invoke>
               </｜DSML｜ calls>
Muse Glimmer:  to=get_weather<|message|><atem:function_calls>
               <atem:invoke name="get_weather">
               <atem:parameter name="city">Paris</atem:parameter>
               </atem:invoke>
               </atem:function_calls>
```

A JSON call object is read in the shapes models write:

- The name comes from `name`, from a string `function`, or from the nested OpenAI shape (`{"type": "function", "function": {"name": ..., "arguments": ...}}`).
- The arguments come from `arguments` or `parameters`, as an object, `null` or a string holding an object. A missing `arguments` means no arguments.
- An empty list, a missing name, or arguments that are not an object make the call malformed, and the stream carries `malformed tool call: <body>`.

## Replace the chat template

The template is the Jinja text in the GGUF key `tokenizer.chat_template`. The converter writes it there from a checkpoint's `chat_template.jinja`, `chat_template.json` or `tokenizer_config.json`. A GGUF without it fails the load with ``GGUF metadata `tokenizer.chat_template`: missing``. The engine runs it in minijinja, set up like Hugging Face's `apply_chat_template`, so Hugging Face templates work as they are. A template sees `messages`, `tools`, `add_generation_prompt` (always true), `bos_token`, `eos_token`, the thinking variables above, and the helpers `raise_exception` and `strftime_now`.

You can replace the template. The first source that is set wins:

- `chat_template`: the template text.
- `chat_template_file`: a file holding the template, read at load. An unreadable file fails the load with `CucaError::Io` naming it.
- A template the preset pins. Three presets pin the template of the model's own repository, because the published GGUF embeds an earlier one: `ministral3-3b-reasoning-q4_0`, `ministral3-3b-instruct-q4_0` and `muse-glimmer-30b-q4_0`. The file downloads once into the cache and is checked against its SHA-256.
- The template inside the model file, or the one converted from the checkpoint.

```kdl,name=Use your own template file
source "qwen3-0.6b-q8_0"
chat_template_file "templates/qwen3.jinja"
```

A replacement drives everything derived from the template: the reasoning markers, whether thinking can be switched off, the tool-call format, the end-of-turn tokens and the thinking variables. A replacement that does not compile, or cannot render a one-message conversation, fails `LocalModel::load` with `CucaError::Config` (`chat_template override does not compile: ...` or `chat_template override does not render a one-message conversation: ...`). A GGUF's own template that does not compile fails the load with a `CucaError::Provider` that reads `chat template: ...`.

### Named templates

Some GGUFs carry a template name instead of Jinja, as llama.cpp's converter writes them. A `tokenizer.chat_template` or `chat_template` that is exactly one of these names is expanded to the equivalent Jinja at load.

| Name | Used by | Writes |
|---|---|---|
| `mistral-v7-tekken` | Mistral Small 3.0 and 3.1, Devstral Small 2505 | `[SYSTEM_PROMPT]` system `[/SYSTEM_PROMPT]`, `[INST]` user `[/INST]`, assistant, then `</s>` |
| `mistral-v7` | the SentencePiece variant | the same, with one space after `[SYSTEM_PROMPT]` and `[INST]` and before each assistant turn |
| `deepseek-v41` | DeepSeek-V4.1 | DeepSeek's own prompt format: thinking and chat modes, tools in the first system message, DSML tool calls |

```kdl,name=Name a built-in template
source file="/models/mistral-small.gguf"
chat_template "mistral-v7-tekken"
```

- The two Mistral names write a system message where it stands, write every role other than system and user (assistant, tool) as an assistant turn, add no BOS (the tokenizer adds it) and ignore declared tools, so the model never sees them.
- Any other bare name (letters, digits, `.`, `-` and `_` only) fails the load with a message naming the three supported names.
- DeepSeek-V4.1 ships no Jinja, so its converted GGUF carries the name `deepseek-v41`. See [Run DeepSeek-V4.1](@/local/deepseek-v41.md).
