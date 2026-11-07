+++
title = "Local tokenizers"
description = "The tokenizer schemes and pre-tokenizers provider-local builds from a GGUF file, how special tokens and BOS are handled, and how the ids were checked against reference tokenizers."
template = "page.html"
weight = 7
+++

# Local tokenizers

<dl class="page-facts">
<dt>In one line</dt>
<dd>The tokenizer is built once at load from <code>tokenizer.ggml.*</code> metadata in pure Rust: SentencePiece, byte-level BPE with hand-written pre-tokenizers, or Gemma 4's escaped scheme.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature.</dd>
<dt>Read this if</dt>
<dd>A GGUF fails with an unsupported tokenizer error, or prompt token counts differ from a reference tokenizer.</dd>
</dl>

## Schemes

`tokenizer.ggml.model` selects the scheme; any other value fails the load with ``unsupported tokenizer model `NAME` ``.

- `llama` is SentencePiece: score-ordered merging of UTF-8 characters, `<0xHH>` byte fallback, and the optional leading space of `tokenizer.ggml.add_space_prefix` (true when the key is absent).
- `gpt2` is byte-level BPE. The text is cut into words by the pre-tokenizer that `tokenizer.ggml.pre` names, each word is mapped through the GPT-2 byte alphabet and merged by rank. The pre-tokenizers are hand-written linear splitters over Unicode general categories, not regular expressions, so their cost grows linearly with the text.
- `gemma4`, or `gpt2` with `tokenizer.ggml.pre` set to `gemma4`, replaces spaces with `▁`, never merges across a newline boundary, and falls back to `<0xHH>` byte tokens. BOS is always on for it, whatever the metadata says.

Preset families and what their published files carry:

| Family | `tokenizer.ggml.model` | `tokenizer.ggml.pre` |
|---|---|---|
| Gemma 4 | `gemma4` | none |
| Gemma 3, ERNIE 4.5 | `llama` | `default` |
| Qwen 3 | `gpt2` | `qwen2` |
| Qwen 3.5 | `gpt2` | `qwen35` |
| DeepSeek-R1 distill (Qwen) | `gpt2` | `deepseek-r1-qwen` |
| DeepSeek-R1 distill (Llama) | `gpt2` | `llama-bpe` |
| SmolLM 3 | `gpt2` | `smaug-bpe` |
| Phi 4 mini | `gpt2` | `gpt-4o` |
| Granite 3.3 | `gpt2` | `refact` |
| OLMo 2 | `gpt2` | `dbrx` |
| OLMoE | `gpt2` | `olmo` |
| Muse Glimmer | `gpt2` | `llama4` |

## Pre-tokenizers

A `gpt2` model with no `pre`, or `default`, uses the default chain. SentencePiece models use no pre-tokenizer. The implemented `tokenizer.ggml.pre` values, by expression:

- `llama3`, `llama-v3`, `llama-bpe`, `falcon3`, `falcon-h1`, `midm-2.0`, `lfm2`, `jina-v5-nano`: the Llama 3 expression. A whole word found in the vocabulary skips the merges, and a BOS token is added by default.
- `smaug-bpe`, `dbrx`: the Llama 3 expression without that shortcut.
- `qwen2`, `deepseek-r1-qwen`, `kormo`, `f2llmv2`, `hunyuan`, `solar-open`: the Qwen 2 expression, with single digits.
- `qwen35`: the Qwen 2 expression with combining marks in letter runs.
- `gpt-2`, `olmo`, `trillion`, `granite-docling`, `exaone4`, `mellum`, `modern-bert`, `gigachat`, `a.x-4.0`: the GPT-2 expression.
- `starcoder`, `refact`, `command-r`, `exaone`, `mellum2`: single digits are split first, then the GPT-2 expression runs on the fragments between them.
- `gpt-4o`, `llama4`, `kanana2`, `talkie`, `minimax-m2`: the GPT-4o expression, with case classes from real Unicode categories. llama.cpp approximates them with ASCII case only, so the two can differ on non-ASCII mixed-case runs such as `éÉ`.
- `tekken`, `pixtral`: the Tekken expression of Mistral Nemo, Pixtral, Mistral Small 3 and Mistral Small 4, as llama.cpp maps both names (`LLAMA_VOCAB_PRE_TYPE_TEKKEN`). It is the GPT-4o expression without the contraction suffix on words and with single digits. A whole word found in the vocabulary skips the merges, and a BOS token is added by default.
- `gemma4`: the space-escaped scheme above.

Any other value fails the load with ``unsupported pre-tokenizer `NAME` ``. The stages run in sequence on the fragments the previous stage left, as in llama.cpp, so the starcoder family keeps a whitespace run before a digit whole where a single GPT-2 regular expression would give its last space to the digit.

The `tekken` and `pixtral` names are the Tekken expression, which is what Mistral Nemo, Pixtral and Mistral Small 3 and 4 were trained with (their `tekken.json`). The Hugging Face `tokenizer.json` of Pixtral and of Mistral Small 3.x declares the Llama 3 expression instead. On a corpus of 1,536 texts the two differ on 13, mostly non-ASCII letters. A checkpoint whose `tokenizer.json` carries the Tekken regular expression and `ignore_merges` converts to `tekken`.

## Special tokens and BOS

- Special tokens (control, user-defined and unknown types) are cut out of the text longest first before the scheme runs, so `<|im_start|>` in a rendered prompt is one token.
- The prompt gets a BOS token when `tokenizer.ggml.add_bos_token`, or the scheme's default, says so, and never a second one when the rendered template already starts with it. EOS is appended under `tokenizer.ggml.add_eos_token` the same way.
- For a SentencePiece model with `add_space_prefix`, text after a special token such as `</s>` gets a leading space token, as in llama.cpp. Hugging Face's tokenizer does not add it, so such a model's prompt has one more token than the reference tokenizer's. Some GGUF files of ERNIE 4.5 omit the key and so take the default; current llama.cpp conversions write it as `false`.

## What was checked

- `tokenizer/golden.rs` compares the ids of 16 texts and 3 chat prompts for 7 models (Qwen 3, Gemma 3 270M and 1B, SmolLM 3, Phi 4 mini, the R1 Qwen distill, Qwen 3.5) with the ids of each model's Hugging Face `tokenizer.json`. It reads cached model files and prints `SKIP:` for a missing one. Where llama.cpp and Hugging Face differ, the expected ids follow llama.cpp, except for the GPT-4o case classes, which follow Hugging Face.
- `tokenizer_corpus_matches_reference_ids` (driven by `CUCA_TOK_GGUF` and `CUCA_TOK_CASES`, skipped without them) gave identical ids for:
  - 4,537 texts, those and every one of Mistral Small 4's 1,000 special tokens alone and in context, against its `tokenizer.json`;
  - 1,536 texts against Mistral Small 3.2's `tekken.json` read by `mistral_common`, through the headers of the published GGUFs (pre-tokenizers `pixtral` and `tekken`);
  - 744 texts for Muse Glimmer (ASCII, digits, contractions, mixed-case non-ASCII, CJK, emoji, special tokens, fuzzed strings), through a tiny GGUF and through the real preset file's tokenizer.
- Muse Glimmer's `tokenizer.json` sets `ignore_merges`, which the engine's `llama4` rule does not. All 400,000 texts made of one vocabulary entry, alone and after a space, encode identically with the flag on and off and equal the engine's ids.
- `tekken_words_match_the_reference_regex` in `tokenizer/pretok.rs` compares the Tekken pieces of 336 texts (ASCII, digits, CJK, emoji, accented mixed case, code, whitespace runs, 300 seeded random ones) with those of Hugging Face `tokenizers`' regular expression splitter with Mistral Small 4's `tokenizer.json` pattern.
