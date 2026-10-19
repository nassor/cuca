//! The engine thread's state: model, K/V cache bookkeeping and the generation
//! loop behind one request.
//!
//! [`Engine::run`] executes one [`Job`] at a time on the dedicated engine
//! thread (jobs queue FIFO in the channel owned by `LocalModel`). A job is a
//! pre-tokenized prompt plus the channel its output blocks go to:
//!
//! 1. **Prefix reuse.** The engine remembers the ids whose K/V fill the model
//!    cache. A new prompt keeps the longest common prefix (always leaving the
//!    last prompt token to prefill, since its logits start generation) and the
//!    cache is truncated to it, so a conversation that appends a turn pays only
//!    for the new tokens. The reuse is reported as the response's
//!    [`PromptCacheUsage`] (`read_tokens` reused, `write_tokens` prefilled).
//! 2. **Prefill** in chunks of [`prefill_chunk`] tokens; only the last chunk
//!    computes logits. The loop notices a dropped stream between chunks.
//! 3. **Decode**: sample, stop on an end-of-generation id, feed the id through
//!    the [`OutputParser`] and forward every resulting block to the stream with
//!    a blocking send (a slow consumer backpressures generation), then run the
//!    model one token further. Generation stops at the end-of-generation id, at
//!    `max_tokens`, or when the context is full.
//! 4. **Usage**: exact prompt and completion token counts are recorded in the
//!    response metadata before the stream ends.
//!
//! A forward-pass error resets the cache and is delivered as the stream's last
//! item; a panic inside a job is caught, reported the same way, and the engine
//! keeps serving later jobs. A job whose consumer has gone away leaves the
//! cache consistent with the ids it processed, so its work is reusable.

use std::fs::File;
use std::io::BufReader;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::gguf_file;
use tokio::sync::mpsc;

use super::chat::ChatTemplate;
use super::gemma4::{Gemma4, Model, prefill_chunk};
use super::output::{Markers, OutputParser};
use super::sampler::{SplitMix64, random_seed, sample};
use super::tokenizer::Tokenizer;
use super::{LocalDevice, LocalModelConfig, SamplingConfig};
use crate::client::ResponseMetadataHandle;
use crate::error::CucaError;
use crate::request::{PromptCacheUsage, TokenUsage};
use crate::types::{MessageContentBlock, ProviderEndpoint};

/// One generation request for the engine thread.
pub(crate) struct Job {
    /// The tokenized prompt, `0 < len < context_length`.
    pub(crate) prompt_ids: Vec<u32>,
    /// Whether the prompt ends inside an open reasoning channel.
    pub(crate) start_in_thinking: bool,
    /// Whether the request enabled thinking (reasoning tokens are reported).
    pub(crate) enable_thinking: bool,
    /// Per-request temperature override.
    pub(crate) temperature: Option<f32>,
    /// Per-request cap on completion tokens.
    pub(crate) max_tokens: Option<u32>,
    /// Where output blocks (or the failure) go; dropping it ends the stream.
    pub(crate) events: mpsc::Sender<Result<MessageContentBlock, CucaError>>,
    /// Receives the prompt-cache and token usage of this job.
    pub(crate) metadata: ResponseMetadataHandle,
}

/// The model plus the state that survives between jobs.
pub(crate) struct Engine {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    markers: Markers,
    sampling: SamplingConfig,
    /// Ids whose K/V fill the model cache, in order; its length always equals
    /// the model's cache length.
    cached: Vec<u32>,
    context_length: usize,
    /// Sampling generator when no fixed seed is configured.
    rng: SplitMix64,
    /// Generator of tool-call ids.
    id_rng: SplitMix64,
}

fn local_err(message: impl Into<String>) -> CucaError {
    CucaError::provider(ProviderEndpoint::Local, message)
}

fn md_u32(
    md: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Result<u32, CucaError> {
    md.get(key)
        .ok_or_else(|| local_err(format!("GGUF metadata `{key}`: missing")))?
        .to_u32()
        .map_err(|e| local_err(format!("GGUF metadata `{key}`: {e}")))
}

impl Engine {
    /// Read the GGUF file at `path` and build the engine, the tokenizer and
    /// the chat template from it.
    ///
    /// The model lands on `config.device`: candle on the CPU, or burn on a
    /// GPU. Either way it is owned by the calling (engine) thread for its
    /// whole life; burn's device streams are per thread, so every forward
    /// pass runs on the thread that loaded the weights.
    ///
    /// # Errors
    ///
    /// [`CucaError::Io`] when the file cannot be opened or read,
    /// [`CucaError::Config`] when `config.context_length` exceeds the model
    /// maximum or `config.device` is compiled out, and
    /// [`CucaError::Provider`] for a file that is not a GGUF file candle can
    /// parse (for example an unsupported tensor dtype such as an IQ
    /// quantization), an unsupported checkpoint (wrong architecture, MoE, a
    /// weight dtype the GPU engine does not read), a missing tensor or
    /// metadata key, or an unusable tokenizer.
    pub(crate) fn open(
        path: &Path,
        config: &LocalModelConfig,
    ) -> Result<(Self, Arc<Tokenizer>, ChatTemplate), CucaError> {
        let io_err = |e: &dyn std::fmt::Display| CucaError::Io {
            message: format!("reading GGUF `{}`: {e}", path.display()),
        };
        let mut reader = BufReader::new(File::open(path).map_err(|e| io_err(&e))?);
        let content = gguf_file::Content::read(&mut reader).map_err(|e| match e {
            // A read failure of the file itself; anything else is a header
            // candle cannot interpret (a tensor dtype it does not implement,
            // such as an IQ quantization, is the usual cause).
            candle_core::Error::Io(_) => io_err(&e),
            other => local_err(format!(
                "`{}` is not a supported GGUF file: {other}",
                path.display()
            )),
        })?;
        drop(reader);

        let architecture = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok());
        if architecture.map(String::as_str) != Some("gemma4") {
            return Err(local_err(format!(
                "unsupported GGUF architecture {architecture:?}; expected \"gemma4\""
            )));
        }

        let model = match config.device {
            LocalDevice::Cpu => Model::Cpu(Gemma4::load(&content, path, config.context_length)?),
            #[cfg(feature = "provider-local-gpu")]
            device => Model::Gpu(super::gemma4::gpu::GpuGemma4::load(
                &content,
                path,
                config.context_length,
                device,
            )?),
            #[cfg(not(feature = "provider-local-gpu"))]
            device => {
                return Err(super::device_unavailable(device).unwrap_or_else(|| {
                    CucaError::Config(format!("device {device:?} is not available"))
                }));
            }
        };
        let tokenizer = Arc::new(Tokenizer::from_gguf(&content.metadata)?);
        let eos_id = md_u32(&content.metadata, "tokenizer.ggml.eos_token_id")?;
        let bos_id = md_u32(&content.metadata, "tokenizer.ggml.bos_token_id")?;
        let markers = Markers::from_tokenizer(&tokenizer, eos_id)?;

        const TEMPLATE_KEY: &str = "tokenizer.chat_template";
        let source = content
            .metadata
            .get(TEMPLATE_KEY)
            .ok_or_else(|| local_err(format!("GGUF metadata `{TEMPLATE_KEY}`: missing")))?
            .to_string()
            .map_err(|e| local_err(format!("GGUF metadata `{TEMPLATE_KEY}`: {e}")))?
            .clone();
        let template = ChatTemplate::new(source, tokenizer.token_text(bos_id).to_string())?;

        let engine = Self {
            model,
            tokenizer: Arc::clone(&tokenizer),
            markers,
            sampling: config.sampling,
            cached: Vec::new(),
            context_length: config.context_length,
            rng: SplitMix64(random_seed()),
            id_rng: SplitMix64(random_seed()),
        };
        Ok((engine, tokenizer, template))
    }

    /// Run one job to completion on the calling (engine) thread.
    ///
    /// A panic inside the job is caught: the consumer gets a provider error as
    /// the last stream item and the cache is reset, so later jobs still work.
    pub(crate) fn run(&mut self, job: Job) {
        let events = job.events.clone();
        if catch_unwind(AssertUnwindSafe(|| self.generate(job))).is_err() {
            self.reset();
            let _ = events.blocking_send(Err(local_err("local engine panicked; KV cache reset")));
        }
    }

    /// Forget every cached position.
    fn reset(&mut self) {
        self.model.truncate_cache(0);
        self.cached.clear();
    }

    fn generate(&mut self, job: Job) {
        let Job {
            prompt_ids,
            start_in_thinking,
            enable_thinking,
            temperature,
            max_tokens,
            events,
            metadata,
        } = job;
        let prompt_len = prompt_ids.len();
        if prompt_len == 0 || prompt_len >= self.context_length {
            let _ = events.blocking_send(Err(local_err(format!(
                "prompt has {prompt_len} tokens; context_length is {}",
                self.context_length
            ))));
            return;
        }

        // 1. Prefix reuse: the last prompt token is always prefilled, because
        // its logits start generation.
        let mut reuse = self
            .cached
            .iter()
            .zip(&prompt_ids)
            .take_while(|(a, b)| a == b)
            .count();
        if reuse == prompt_len {
            reuse -= 1;
        }
        self.model.truncate_cache(reuse);
        self.cached.truncate(reuse);

        // 2. Prefill in bounded chunks; only the last one computes logits.
        let chunk = prefill_chunk(self.context_length);
        let mut logits = None;
        let mut pos = reuse;
        while pos < prompt_len {
            let end = (pos + chunk).min(prompt_len);
            let last = end == prompt_len;
            match self.model.forward(&prompt_ids[pos..end], pos, last) {
                Ok(l) => logits = l,
                Err(e) => {
                    let _ = events.blocking_send(Err(e));
                    self.reset();
                    return;
                }
            }
            self.cached.extend_from_slice(&prompt_ids[pos..end]);
            pos = end;
            if !last && events.is_closed() {
                return;
            }
        }
        let Some(mut logits) = logits else {
            let _ = events.blocking_send(Err(local_err("prefill produced no logits")));
            self.reset();
            return;
        };

        // 3. Prompt-cache usage: what was reused and what was prefilled.
        metadata.set_prompt_cache_usage(PromptCacheUsage {
            read_tokens: saturating_u32(reuse),
            write_tokens: saturating_u32(prompt_len - reuse),
        });

        // 4. Generation setup.
        let temperature = temperature.unwrap_or(self.sampling.temperature);
        let (top_k, top_p) = (self.sampling.top_k, self.sampling.top_p);
        let mut seeded = self.sampling.seed.map(SplitMix64);
        let rng = seeded.as_mut().unwrap_or(&mut self.rng);
        let budget = max_tokens
            .map_or(usize::MAX, |n| n as usize)
            .min(self.context_length - prompt_len);
        let tokenizer = Arc::clone(&self.tokenizer);
        let mut parser = OutputParser::new(self.markers, start_in_thinking);
        let id_rng = &mut self.id_rng;
        let mut next_call_id = || format!("call_{:016x}", id_rng.next_u64());
        let mut items = Vec::new();
        let mut completion = 0usize;

        // 5. Decode loop.
        while completion < budget {
            let id = sample(&mut logits, temperature, top_k, top_p, rng);
            if self.markers.is_eog(id) {
                break;
            }
            completion += 1;
            parser.push(id, &tokenizer, &mut next_call_id, &mut items);
            for item in items.drain(..) {
                if events.blocking_send(item).is_err() {
                    return;
                }
            }
            if completion == budget || self.cached.len() == self.context_length {
                break;
            }
            match self.model.forward(&[id], self.cached.len(), true) {
                Ok(Some(l)) => logits = l,
                outcome => {
                    let e = outcome
                        .err()
                        .unwrap_or_else(|| local_err("decode step produced no logits"));
                    let _ = events.blocking_send(Err(e));
                    self.reset();
                    return;
                }
            }
            self.cached.push(id);
        }

        // 6. Flush whatever the parser still holds.
        parser.finish(&mut next_call_id, &mut items);
        for item in items.drain(..) {
            if events.blocking_send(item).is_err() {
                return;
            }
        }

        // 7. Usage, then drop `events` to end the stream.
        metadata.set_token_usage(TokenUsage {
            prompt_tokens: saturating_u32(prompt_len),
            completion_tokens: saturating_u32(completion),
            reasoning_tokens: enable_thinking.then(|| parser.reasoning_tokens()),
        });
    }
}

fn saturating_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Fixtures shared by the engine tests and the `LocalModel` tests.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::provider::local::gemma4::tiny::{TinyModel, VOCAB, load};

    /// Vocabulary of the tiny model: ids 1 (`<eos>`), 3 (`<turn|>`) and 4
    /// (`<|tool_response>`) end generation; 9..16 are one-letter pieces.
    pub(crate) fn tiny_tokenizer() -> Tokenizer {
        let mut pieces: Vec<String> = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<turn|>",
            "<|tool_response>",
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        pieces.extend(["a", "b", "c", "d", "e", "f", "g"].map(String::from));
        assert_eq!(pieces.len(), VOCAB);
        let types = (0..VOCAB).map(|i| if i < 9 { 3 } else { 1 }).collect();
        Tokenizer::new(pieces, types, &[]).unwrap()
    }

    /// An engine over the tiny model, with greedy or seeded sampling.
    pub(crate) fn engine(
        model: &TinyModel,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
    ) -> Engine {
        let tokenizer = Arc::new(tiny_tokenizer());
        Engine {
            model: Model::Cpu(load(model, context_length)),
            markers: Markers::from_tokenizer(&tokenizer, 1).unwrap(),
            tokenizer,
            sampling: SamplingConfig {
                temperature,
                top_k: 64,
                top_p: 0.95,
                seed,
            },
            cached: Vec::new(),
            context_length,
            rng: SplitMix64(1),
            id_rng: SplitMix64(2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::engine;
    use super::*;
    use crate::provider::local::gemma4::tiny::write_tiny_model;

    struct Outcome {
        items: Vec<Result<MessageContentBlock, CucaError>>,
        metadata: crate::client::ResponseMetadata,
    }

    fn run(e: &mut Engine, prompt: &[u32], max_tokens: Option<u32>) -> Outcome {
        let (tx, mut rx) = mpsc::channel(1024);
        let metadata = ResponseMetadataHandle::empty();
        e.run(Job {
            prompt_ids: prompt.to_vec(),
            start_in_thinking: false,
            enable_thinking: false,
            temperature: None,
            max_tokens,
            events: tx,
            metadata: metadata.clone(),
        });
        let mut items = Vec::new();
        while let Some(item) = rx.blocking_recv() {
            items.push(item);
        }
        Outcome {
            items,
            metadata: metadata.take(),
        }
    }

    /// The text of every `Text` block, concatenated.
    fn text(outcome: &Outcome) -> String {
        outcome
            .items
            .iter()
            .filter_map(|i| match i {
                Ok(MessageContentBlock::Text(t)) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    const PROMPT: [u32; 6] = [2, 9, 10, 11, 12, 13];

    #[test]
    fn reports_exact_usage_and_respects_max_tokens() {
        let model = write_tiny_model();
        let mut e = engine(&model, 32, 0.0, None);
        let out = run(&mut e, &PROMPT, Some(5));
        // Random weights may emit a malformed tool call; that is a parse
        // error item, not an engine failure, and usage is still reported.
        let usage = out.metadata.token_usage.expect("usage reported");
        assert_eq!(usage.prompt_tokens, 6);
        assert!(usage.completion_tokens <= 5);
        assert_eq!(usage.reasoning_tokens, None);
        assert_eq!(
            out.metadata.prompt_cache_usage,
            Some(PromptCacheUsage {
                read_tokens: 0,
                write_tokens: 6
            })
        );
    }

    #[test]
    fn context_limit_caps_the_completion() {
        let model = write_tiny_model();
        let mut e = engine(&model, 8, 0.0, None);
        let out = run(&mut e, &PROMPT, None);
        let usage = out.metadata.token_usage.expect("usage reported");
        // 8 positions, 6 of them prompt.
        assert!(usage.completion_tokens <= 2, "{usage:?}");
    }

    #[test]
    fn prefix_reuse_reads_the_shared_prefix() {
        let model = write_tiny_model();
        let mut e = engine(&model, 32, 0.0, None);
        run(&mut e, &PROMPT, Some(3));

        // Same prompt again: everything but the last token is reused.
        let again = run(&mut e, &PROMPT, Some(3));
        assert_eq!(
            again.metadata.prompt_cache_usage,
            Some(PromptCacheUsage {
                read_tokens: 5,
                write_tokens: 1
            })
        );

        // Longer prompt: at least the old prompt is reused.
        let mut longer = PROMPT.to_vec();
        longer.extend([14, 15]);
        let out = run(&mut e, &longer, Some(3));
        let cache = out.metadata.prompt_cache_usage.unwrap();
        assert!(cache.read_tokens >= 6, "{cache:?}");
        assert_eq!(cache.read_tokens + cache.write_tokens, 8);

        // Diverging prompt: only the common head is reused.
        let out = run(&mut e, &[2, 9, 10, 15, 14], Some(3));
        let cache = out.metadata.prompt_cache_usage.unwrap();
        assert_eq!(cache.read_tokens, 3);
        assert_eq!(cache.write_tokens, 2);
    }

    #[test]
    fn warm_cache_generates_what_a_cold_engine_generates() {
        let model = write_tiny_model();
        let mut warm = engine(&model, 32, 0.0, None);
        let cold = run(&mut engine(&model, 32, 0.0, None), &PROMPT, Some(4));
        // An unrelated job in between, then the same prompt: the cache is
        // truncated to the shared head and re-prefilled.
        run(&mut warm, &[2, 15, 14, 13, 12, 11, 10], Some(4));
        let rerun = run(&mut warm, &PROMPT, Some(4));
        assert_eq!(text(&cold), text(&rerun));
        assert_eq!(
            cold.metadata.token_usage.unwrap().completion_tokens,
            rerun.metadata.token_usage.unwrap().completion_tokens
        );
    }

    #[test]
    fn fixed_seed_reproduces_sampling() {
        let model = write_tiny_model();
        let a = run(&mut engine(&model, 32, 1.0, Some(5)), &PROMPT, Some(6));
        let b = run(&mut engine(&model, 32, 1.0, Some(5)), &PROMPT, Some(6));
        assert_eq!(text(&a), text(&b));
        assert_eq!(
            a.metadata.token_usage.unwrap().completion_tokens,
            b.metadata.token_usage.unwrap().completion_tokens
        );
    }

    #[test]
    fn forward_failure_is_reported_and_resets_the_cache() {
        let model = write_tiny_model();
        let mut e = engine(&model, 32, 0.0, None);
        run(&mut e, &PROMPT, Some(2));
        // Token 99 is outside the 16-entry vocabulary.
        let out = run(&mut e, &[2, 9, 99], Some(2));
        assert_eq!(out.items.len(), 1);
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
        assert!(e.cached.is_empty());
        // The engine still serves the next job from a cold cache.
        let ok = run(&mut e, &PROMPT, Some(2));
        assert!(ok.metadata.token_usage.is_some());
        assert_eq!(ok.metadata.prompt_cache_usage.unwrap().read_tokens, 0);
    }

    #[test]
    fn dropped_consumer_stops_generation_without_corrupting_state() {
        let model = write_tiny_model();
        let mut e = engine(&model, 32, 0.0, None);
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        e.run(Job {
            prompt_ids: PROMPT.to_vec(),
            start_in_thinking: false,
            enable_thinking: false,
            temperature: None,
            max_tokens: Some(10),
            events: tx,
            metadata: ResponseMetadataHandle::empty(),
        });
        // `cached` mirrors the model cache, so the next job reuses it.
        assert_eq!(e.cached.len(), e.model.cache_len());
        let out = run(&mut e, &PROMPT, Some(2));
        assert!(out.metadata.token_usage.is_some());
        assert_eq!(out.metadata.prompt_cache_usage.unwrap().read_tokens, 5);
    }

    #[test]
    fn rejects_prompts_that_do_not_fit() {
        let model = write_tiny_model();
        let mut e = engine(&model, 8, 0.0, None);
        let out = run(&mut e, &[2, 9, 10, 11, 12, 13, 14, 15], Some(1));
        assert_eq!(out.items.len(), 1);
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
        let out = run(&mut e, &[], Some(1));
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
    }

    /// A GGUF header declaring one tensor of dtype 20 (IQ4_NL), which candle
    /// cannot read: the load must name an unsupported file, not an I/O error.
    #[test]
    fn open_reports_an_unreadable_tensor_dtype_as_unsupported() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes()); // version
        bytes.extend(1u64.to_le_bytes()); // tensors
        bytes.extend(0u64.to_le_bytes()); // metadata entries
        bytes.extend(1u64.to_le_bytes()); // name length
        bytes.push(b'x');
        bytes.extend(1u32.to_le_bytes()); // dimensions
        bytes.extend(32u64.to_le_bytes());
        bytes.extend(20u32.to_le_bytes()); // IQ4_NL
        bytes.extend(0u64.to_le_bytes()); // offset
        bytes.resize(bytes.len().next_multiple_of(32) + 64, 0);
        let path = std::env::temp_dir().join(format!("cuca-iq4-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();

        let config = LocalModelConfig::default();
        let result = Engine::open(&path, &config);
        let _ = std::fs::remove_file(&path);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("an IQ4_NL tensor must not load"),
        };
        assert!(
            matches!(&err, CucaError::Provider { message, .. } if message.contains("not a supported GGUF file")),
            "{err:?}"
        );
    }
}
