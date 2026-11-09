+++
title = "Local chat templates, reasoning and tool calls"
description = "Where the local engine gets its chat template, the variables and helpers a template sees, and how reasoning blocks, tool calls and end-of-generation tokens are recognized from the template and the vocabulary."
template = "page.html"
weight = 8
+++

# Local chat templates, reasoning and tool calls

<dl class="page-facts">
<dt>In one line</dt>
<dd>The prompt format comes from a Jinja template; the template and the vocabulary also decide how the output is parsed into <code>Text</code>, <code>Thinking</code> and <code>ToolCall</code> blocks, with no model names in the engine.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature.</dd>
<dt>Read this if</dt>
<dd>A model's reply is parsed in a way you did not expect, a template needs replacing, or you are checking which tool-call syntax a model gets.</dd>
</dl>

## Where the template comes from

The template is the Jinja text of `tokenizer.chat_template`, which a GGUF must carry: a file without it fails the load with ``GGUF metadata `tokenizer.chat_template`: missing``. Two sources replace it, as llama.cpp's `--chat-template-file` and vLLM's `--chat-template` do:

- `LocalModelConfig::chat_template: Option<String>`, the template text. It wins over everything else. The live tests and `examples/local_model.rs` read it from the file `CUCA_LOCAL_CHAT_TEMPLATE` names.
- A template a source pins: `ModelSource::HuggingFace::template`, which a preset sets through `ModelPreset::template`. The file is downloaded once into the cache like a model file (`<cache>/huggingface/<repo>/<revision>/<file>`), verified against its SHA-256, and read at load.

A replacement is used for everything derived from the template: the reasoning markers and whether a request can switch reasoning off, the tool-call syntax, the end-of-turn tokens, and the `enable_thinking`, `reasoning_effort` and `reasoning_strength` variables. The replacement may also be a llama.cpp template name; see [Named templates](#named-templates).

A replacement that does not compile, or cannot render a one-message conversation, fails `LocalModel::load` with `CucaError::Config` (`chat_template override does not compile: …` or `chat_template override does not render a one-message conversation: …`), not the first request. A GGUF's own template that does not compile fails with a `CucaError::Provider` item `chat template: …`.

Three presets pin the repository's current template because the published GGUFs of those models embed an earlier one:

- `ministral3-3b-reasoning-q4_0` pins `chat_template.jinja` of `mistralai/Ministral-3-3B-Reasoning-2512` at `4a36357c`. The repository's template merges consecutive messages, writes `reasoning` and `reasoning_content` back as `[THINK]` blocks and accepts system content lists. The embedded one is the December 2025 template. The two render the same prompt for a plain turn, tools and a tool round trip, and differ for reasoning in the history.
- `ministral3-3b-instruct-q4_0` pins the file of `mistralai/Ministral-3-3B-Instruct-2512` at `b35d4dfe`. The embedded template has an earlier default system prompt without the current tool-calling paragraph.
- `muse-glimmer-30b-q4_0` pins `chat_template.jinja` of `meta-models/Muse-Glimmer-30B` at `cec94fcf`. The embedded template renders the same bytes except when the caller's system prompt itself mentions "Reasoning effort": the current template rewrites that to "Reasoning strength", the embedded one keeps the line and appends its own directive.

### Named templates

llama.cpp's converter writes a template name instead of Jinja for some Mistral checkpoints, and llama.cpp renders it in C++ (`llm_chat_apply_template`). A `tokenizer.chat_template` or a `chat_template` override that is exactly one of the names below is expanded to equivalent Jinja at load, before anything is derived from the template (`src/provider/local/chat/builtin.rs`). `deepseek-v41` is the engine's own name: DeepSeek-V4.1 ships no Jinja, and the converter writes this name into its GGUF.

| Name | Used by | Renders |
|---|---|---|
| `mistral-v7-tekken` | Mistral Small 3.0 and 3.1, Devstral Small 2505 | `[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]`, `[INST]{user}[/INST]`, `{assistant}</s>` |
| `mistral-v7` | the SentencePiece variant | the same, with one space after `[SYSTEM_PROMPT]` and `[INST]` and before each assistant turn |
| `deepseek-v41` | DeepSeek-V4.1 | the prompt format of DeepSeek's `encoding.py`: thinking and chat modes, tools in the first system message, DSML tool calls |

The Mistral names follow llama.cpp's branch: a system message is written where it stands, every role other than system and user (assistant, tool) is written as an assistant turn, no BOS is written (the tokenizer adds it), the generation prompt adds nothing, and declared tools are ignored, so the model never sees them. Any other bare name (letters, digits, `.`, `-`, `_` only) fails the load with a message naming the three supported ones, instead of rendering the name as the prompt. The table holds only names a 2025 or later model uses.

## Variables and helpers

The template is compiled with minijinja in the environment Hugging Face's `apply_chat_template` uses: blocks trimmed, leading block whitespace stripped, and Python string methods such as `.strip()`, `.startswith()` and `.split()` available. The text is not modified, except that Hugging Face's generation and end-generation tags are rewritten to a no-op block.

Variables the template sees:

- `messages`: the request's messages in OpenAI shape; see [Request translation](@/providers/local/requests.md#request-translation).
- `tools`: defined only when the request declares at least one tool, because templates test `tools is not none`. Each tool is `{"type": "function", "function": {"name", "description", "parameters"}}`.
- `add_generation_prompt`: always true.
- `enable_thinking` and `thinking`: both true when the request enables thinking, otherwise both false; for a template that reads `reasoning_budget` both are true unless the request disables thinking.
- `preserve_thinking`: always false.
- `reasoning_strength`, `reasoning_effort` and `reasoning_budget`: defined only for a template that reads them; see [Thinking](@/providers/local/requests.md#thinking).
- `bos_token` and `eos_token`: the token texts, or empty.

Helpers:

- `raise_exception(message)` surfaces as a render error.
- `strftime_now(format)` returns the current UTC time.
- The `tojson` filter lays JSON out like Python's `json.dumps` (`", "` and `": "` separators, with `indent`, `sort_keys` and `ensure_ascii`) and keeps the order of an object's keys, as Python does.
- The `string` filter renders like Python's `str()`, so a dict prints as `{'a': 1}`.

Tool call arguments stay JSON objects, as templates expect. `provider-local` enables the `preserve_order` feature of `serde_json`, so the key order inside a tool's JSON schema (`parameters`) and inside a call's arguments is the order the caller wrote, which the `deepseek-v41` template writes out parameter by parameter.

## Reasoning

When the template writes a reasoning block, the engine finds its markers from the template text and the vocabulary. In detection order:

- `<|channel>thought` ... `<channel|>`: Gemma 4's channel. The template must contain `<|channel>` and the vocabulary both markers. The header word `thought` and the whitespace after the opening marker are dropped.
- `[THINK]` ... `[/THINK]`: Mistral Small 4 and Ministral 3. Both are single control tokens and the template must mention `[THINK]`. The prompt ends after `[/INST]`, so the model opens the block itself, and a `[TOOL_CALLS]` written inside it stays reasoning text.
- `<think>` ... `</think>`: Qwen 3, Qwen 3.5, SmolLM 3, Granite 3.3, the R1 distills and DeepSeek-V4.1; the template must contain either tag.
- Messages addressed to `self`: Muse Glimmer's channel protocol, below.

The text between the markers arrives as `Thinking` blocks. If the rendered prompt ends inside an open block, generation starts in thinking. Markers are matched on text, so a marker that the vocabulary spells as several tokens works like a single control token. A close marker with no open marker is dropped.

Whether a request can switch thinking off follows the template:

- Gemma 4, Qwen 3, Qwen 3.5 and SmolLM 3 read `enable_thinking`, and Granite 3.3 reads `thinking`, which the request sets to the same value.
- Muse Glimmer reads `reasoning_strength` (`low`, `medium`, `high` or `xhigh`, default `high`). It cannot switch reasoning off: a request that disables thinking gets `low`, the least reasoning the model has.
- A template that reads `reasoning_effort` (Mistral Small 4: `none` or `high`, and it raises on any other value) is a switch like `enable_thinking`.
- A template that reads `reasoning_budget` (DeepSeek-V4.1: `low`, `high` or `max`, default `high`) is a switch that is on by default: it reasons unless the request disables thinking.
- The DeepSeek-R1 distills read none of these and reason on every request.

`LocalModel::capabilities()` reports `reasoning` and `can_disable_thinking` from the same detection. `reasoning_tokens` in the usage counts the tokens inside reasoning blocks, markers included.

## Tool calls

The template says how the model was taught to call tools, and the output parser follows it. The parser buffers a call whole and parses it into a `ToolCall` block. Declared tools reach a model only through its template, so a model whose template has no `tools` variable never sees them.

Syntaxes, in detection order:

- Channel protocol ("ATEM"), Muse Glimmer: a turn is a sequence of messages `<|start|>assistant to=RECIPIENT<|message|>BODY END`, where `END` is `<|eom|>` when more messages follow and `<|eot|>`, an end-of-generation token, at the end of the turn. The prompt ends with `<|start|>assistant`, so generation starts at ` to=...`. A message addressed to `self` is reasoning (`Thinking` blocks), one to `user` (or without a recipient) is answer text, and any other recipient is a tool call. Its body is `<atem:function_calls>`, one `<atem:invoke name="NAME">` per call and one `<atem:parameter name="KEY">VALUE</atem:parameter>` per argument. `<|eom|>` ends a message and generation continues. Markup quoted inside an answer or inside the reasoning stays text. A value is read as JSON when it is a JSON number, boolean, `null`, list or object without surrounding whitespace and as the raw string otherwise, so a string that looks like one of these reads back as it. Detected when the template contains `<atem:function_calls>` and `<|eom|>` and the vocabulary has `<|start|>`, `<|message|>`, `<|eom|>` and `<|eot|>`.
- DSML, DeepSeek-V4.1: a block `<｜DSML｜ calls>` ... `</｜DSML｜ calls>` that the model writes after its answer and a blank line, holding one `<｜DSML｜ invoke name="NAME">` per call and one `<｜DSML｜ parameter name="KEY" string="true|false">VALUE</｜DSML｜ parameter>` per argument. `string="true"` makes the value a JSON string of the exact text; `string="false"` parses it as JSON. A repeated parameter name is a malformed call. The opener inside reasoning starts the call. Detected when the template contains `｜DSML｜` followed by ` calls` or `tool_calls` and the vocabulary has the `｜DSML｜` token. `parse_dsml_calls` in `output/calls.rs` parses the body.
- Mistral Small 4, Devstral, Ministral 3 and Mistral Small 3.2: `[TOOL_CALLS]name[ARGS]{json}` once per call, with no closing marker. A call ends at the next `[TOOL_CALLS]` or at the end of the reply, and the arguments are one JSON object. Mistral Small 3.2 writes `[TOOL_CALLS]name[CALL_ID]id[ARGS]{json}` and its template requires 9-character ids, so the id the model wrote becomes the call's id. Detected when the template contains `[TOOL_CALLS]` and `[ARGS]` and the vocabulary has both tokens (`[CALL_ID]` likewise).
- Gemma 4: `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>` between control tokens. Detected when the template contains `<|tool_call>` and the vocabulary has both markers.
- XML, Qwen 3.5: `<tool_call>` ... `</tool_call>` around `<function=name>` and one `<parameter=key>` element per argument, with the value between a newline after the opening tag and a newline before the closing one. Detected when the template contains `<tool_call>`, `</tool_call>`, `<function=` and `<parameter=`. A value is read as JSON when it is valid JSON and as a string otherwise, so a string argument that is itself valid JSON, such as `42` or `true`, comes back as that JSON value.
- Tagged JSON, Qwen 3 and SmolLM 3: `<tool_call>` ... `</tool_call>` around one call object or a list of them. Detected when the template contains `<tool_call>`; without `</tool_call>` the call runs to the end of the reply.
- Tagged JSON with an alternate opener, Granite 3.3: `<|tool_call|>`, or the plain text `<tool_call>` that the model writes instead, followed by a JSON list to the end of the reply. A closing `<|/tool_call|>` is read only when the template contains it.
- None: Gemma 3, Phi 4 mini, OLMo 2, ERNIE 4.5 and the R1 distills. The output is plain text and `LocalCapabilities::tools` is false.

JSON call objects are read in the shapes models write them:

- The name comes from `name`, from a string `function` (`{"function": "get_weather", "arguments": {...}}`), or from the nested OpenAI shape (`{"type": "function", "function": {"name": ..., "arguments": ...}}`).
- The arguments come from `arguments` or `parameters`, in the nested object first and then beside it, as an object, `null`, or a string holding an object. A missing `arguments` means no arguments.
- An empty list, a missing name, or arguments that are not an object are a malformed call.

A malformed call is a `CucaError::Provider` stream item, `malformed tool call: <body>`, and the stream continues. The engine generates tool call ids as `call_` plus 16 hexadecimal digits.

## End of generation

Generation stops at an end-of-generation token: the EOS, EOT and EOM ids of the metadata plus these vocabulary tokens when present: `<|eot_id|>`, `<|eom_id|>`, `<|eot|>`, `<|im_end|>`, `<|end|>`, `<end_of_turn>`, `<turn|>`, `<|tool_response>`, `<|endoftext|>`, `<|end_of_text|>`, `<EOT>`, `_<EOT>`, `[EOT]`, the DeepSeek end-of-sentence token and `<end_of_utterance>`. A model with none of these and no metadata id fails to load.

The parser holds back bytes until they form complete UTF-8, drops leading whitespace after a reasoning edge or a tool call, and drops one leading space of the first text for SentencePiece models with `add_space_prefix`.

## What was checked

- `renders_match_the_reference_jinja_environment` in `src/provider/local/chat/golden.rs` renders 9 templates (Gemma 3 1B and 270M, Gemma 4, Granite 3.3, Phi 4 mini, Qwen 3, Qwen 3.5, the R1 Qwen distill, SmolLM 3) over 7 cases (a plain turn, a system prompt, a multi-turn history, thinking on and off, tools declared, a tool call with its result). The text equals what Python `jinja2` renders with `trim_blocks`, `lstrip_blocks`, a `json.dumps(ensure_ascii=False)` `tojson` and `strftime_now` pinned to 2026-10-18 UTC. The fixtures are in `src/provider/local/testdata/`.
- `chat/golden_new.rs` runs the same test for Muse Glimmer (25 cases, every reasoning strength among them), Mistral Small 4 (42 cases: tool calls with mapping, string and empty-string arguments, reasoning as a top-level field and as content blocks, consecutive messages of one role, list contents, `reasoning_effort` `none` and `high`, and five the template rejects), Mistral Small 3.2 (29 cases, with its call ids and alternation rule) and both Ministral 3 templates (44 cases each: system prompt or none, tools, tool calls and results, reasoning in the history, multi-turn, content lists, and the cases that must raise). Every render equals what `transformers`' `apply_chat_template` gives in Python, errors included.
- The two Mistral names equal llama.cpp's expected output for the v7 case of `tests/test-chat-template.cpp` and `mistral_common`'s Tekken v7 rendering of six conversations.
- `chat/golden_deepseek.rs` renders the `deepseek-v41` template over 400 cases (the official test pairs, about 80 hand-written conversations and seeded random ones: system and tool messages, thinking and chat modes, efforts, tool loops, per-message tools) and requires the bytes of DeepSeek's reference `encoding.py`. `output/golden_deepseek.rs` parses 20 reference completions, cut into tokens one character at a time, in groups and with the special tokens whole, and requires the blocks the reference parser returns.
- The unit tests of `output/atem.rs` hold the fixtures of llama.cpp's `tests/test-chat.cpp` for the channel protocol (a plain answer, a first-message call, inform-then-act, quoted markup, reasoning containing markup), and `mistral_think_then_call` in `output/mod.rs` covers the Mistral shape that llama.cpp's Ministral 3 parser also reads.
