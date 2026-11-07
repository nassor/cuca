//! In-process GGUF language-model inference (`provider-local`).
//!
//! The engine is pure Rust: candle 0.11 on the CPU (the default), or burn
//! 0.22 (CubeCL) on a GPU selected with [`LocalModelConfig::device`] and a
//! `provider-local-{vulkan,metal,cuda,rocm}` feature. No llama.cpp, no FFI
//! written by this crate, and no HTTP endpoint. (candle's dependency tree
//! compiles a C crate, `onig_sys`, via `tokenizers`, and CubeCL's compiles
//! `simsimd` and `aegis`; cuca calls none of them. The CUDA and ROCm
//! backends link a prebuilt LLVM.) The model runs inside the
//! application and is consumed through the library API only.
//!
//! `LocalModel::load` resolves and loads the GGUF file and starts one engine
//! thread; `CucaClientBuilder::with_local_model` attaches it to a client for
//! `ProviderEndpoint::Local`. Usage, behavior and build notes live on
//! `LocalModel`.
//!
//! # Choosing a model
//!
//! Any GGUF whose `general.architecture` is one of `gemma4`, `gemma3`,
//! `llama` (Llama 3.x and compatible), `qwen2`, `qwen3`, `phi3`
//! (Phi 4-mini and compatible), `smollm3`, `granite`, `qwen3moe`,
//! `olmoe`, `olmo2`, `ernie4_5`, `muse-glimmer`, `mistral3`, `mistral4`, `qwen35` or `qwen35moe` runs: the loader reads every
//! dimension (layers, widths, per-layer head counts and head dimensions,
//! FFN widths or experts, vocabulary, RoPE variants) from the file's
//! metadata and tensor shapes, handles biases, QK-norm, fused QKV and
//! gate/up tensors, tied or separate output heads, sliding windows, NoPE
//! layers, mixtures of experts, and rejects anything else (an
//! unknown architecture, a tensor dtype candle cannot read) with an error
//! naming the cause. [`ModelSource::PRESETS`] lists the verified models,
//! each pinned to a repository commit and SHA-256 (by name through
//! `str::parse`, e.g. `qwen3-0.6b-q8_0`), next to
//! [`ModelSource::hugging_face`] and [`ModelSource::File`] for any other
//! file. The default is Gemma 4 E2B Q4_0 (2.84 GB). Chat template, tool-call
//! syntax and reasoning markers come from the file ([`LocalModel::capabilities`]
//! reports what a model supports).
//!
//! # Checkpoints
//!
//! A Hugging Face safetensors checkpoint loads without a GGUF:
//! [`ModelSource::Checkpoint`] (`"repo:revision".parse()`, downloaded pinned to
//! `revision`) or [`ModelSource::Dir`] (a directory with `config.json`, the
//! `*.safetensors` shards and the tokenizer files) is converted once, in
//! process and in pure Rust, to a GGUF under `<cache>/converted/`, and that
//! file then loads exactly like any other. The conversion is a port of
//! llama.cpp's `convert_hf_to_gguf.py` for the architectures above
//! (`LlamaForCausalLM`, `MistralForCausalLM`, `SmolLM3ForCausalLM`, `GraniteForCausalLM`,
//! `Qwen2ForCausalLM`, `Qwen3ForCausalLM`,
//! `Qwen3MoeForCausalLM`, `Qwen3_5*`, `Qwen3_5Moe*`, `Phi3ForCausalLM`,
//! `Gemma3*`, `Gemma4*`, `Gemma4AssistantForCausalLM`,
//! `Olmo2ForCausalLM`, `Olmo3ForCausalLM` without sliding layers,
//! `OlmoeForCausalLM`, `Ernie4_5ForCausalLM`,
//! `MuseGlimmerForConditionalGeneration`, `MuseGlimmerAssistantModel` (the
//! DFlash drafter of Muse Glimmer, below),
//! `Mistral3ForConditionalGeneration` with a `mistral4` text model (FP8
//! weights are dequantized); any other architecture fails
//! before a weight is downloaded) and writes the file llama.cpp would write
//! (compared with its output tensor by tensor, see the local-provider docs). [`WeightQuant`]
//! (`LocalModelConfig::convert`, `f16`, `q8_0` by default, or `q4_0`) sets
//! the matrix precision; each precision is converted once and cached.
//! Conversion streams tensor by tensor (peak memory is one block of one
//! tensor), and reports its progress through
//! `LocalModelConfig::on_download_progress` as tensors converted out of the
//! tensor count.
//!
//! # Split GGUF files
//!
//! A model published as llama.cpp splits (`<base>-00001-of-0000N.gguf`) is
//! merged once into one cached GGUF: [`ModelSource::File`] naming any part
//! merges the set into `<cache>/merged/`, and a Hugging Face source whose file
//! is a split name downloads and verifies every part
//! ([`ModelSource::HuggingFace`]'s `parts`, [`ModelPreset::parts`]), merges
//! them into `<base>.gguf` next to where they were downloaded and deletes the
//! parts. The merge needs about twice the model's size free until the parts
//! are deleted, streams the tensor data, and is reported through
//! `LocalModelConfig::on_download_progress` as bytes copied.
//!
//! # Bounds
//!
//!
//! - Continuous batching ([`BatchingConfig`]): up to `max_sequences` requests
//!   (4 by default) run at once, each decode step one forward pass over a row
//!   of every running sequence, prompts prefilled in chunks next to them; at
//!   most 16 more queue behind (further callers wait), 64 output blocks are
//!   buffered per request, and a consumer that does not keep up stalls only
//!   its own request.
//! - K/V cache memory: blocks of 16 positions, `2 * 4 * 16 * n_kv_head *
//!   head_dim` bytes per block and layer that owns K/V ([`LocalModel::kv_cache_bytes`]),
//!   in RAM or device memory. The pool is shared by every running sequence and
//!   the cached prefixes: a full-attention group holds [`BatchingConfig::kv_tokens`]
//!   positions (twice `context_length` by default, whatever `max_sequences`
//!   is), a sliding-window group `ceil((window - 1 + rows) / 16) + 2` blocks
//!   per running sequence plus cached prefixes. When the pool is full a
//!   running request that needs a block preempts the youngest one, which
//!   resumes later from the blocks it left in the prefix store. At 8192
//!   positions: 422 MB for Gemma 4 E2B, 1.27 GB for E4B and 3.76 GB for
//!   Qwen3-0.6B (a full-attention model with many KV heads needs a smaller
//!   `context_length`). A prefix store keeps the
//!   blocks of finished requests, named by their token ids, and a prompt
//!   reuses whole blocks of its cached prefix (a hybrid model also resumes
//!   from a recurrent-state snapshot of that position; about 20 MB each for
//!   Qwen3.5-0.8B, [`ServingStats::snapshot_bytes`]).
//! - Gemma 4's per-layer embedding rows are read from disk per input token
//!   instead of being held in RAM (5,040 bytes each and 1.32 GB saved in E2B
//!   Q4_0). A mixture-of-experts model keeps every expert resident by default
//!   (file size; a pass only computes the experts its tokens route to);
//!   [`ExpertPlacement::Tiered`] keeps the experts in the file instead and
//!   streams them through RAM (and, on a GPU, device-memory) caches of a
//!   configured byte size, with the same output.
//! - [`LocalModelConfig::shards`] splits a model over several GPUs (tensor
//!   parallelism): attention heads and FFN widths are split evenly, each
//!   device holds its slice of the weights and of the K/V pools, and the
//!   partial outputs are summed twice per layer, by NCCL on CUDA and in a
//!   fixed host-staged order elsewhere. See the GPU page of the docs.
//! - Prefill runs in chunks of at most 256 tokens, bounding the score matrix,
//!   and a step carries at most [`BatchingConfig::max_step_tokens`] rows in
//!   all. An in-trunk drafter observes every position of its sequence, so
//!   with one the engine serves one request at a time; a file drafter
//!   speculates only while one request runs alone.
//!
//! # Determinism
//!
//! On the CPU a token's logits do not depend on how the prompt was split:
//! prefill and speculative verification run in a row-exact mode (float
//! matmuls per row, paged attention per query row on candle's worker pool,
//! and a safe Q4_0/Q8_0 matmul kernel whose rows are independent of their
//! batch) that equals single-token decoding bit for bit, at no cost in
//! prefill throughput. A request that reuses a cached prefix therefore
//! generates what a cold request generates. (Quantized matmuls round
//! activations to Q8_0, which otherwise amplifies the batch-shape rounding of
//! a float matmul into visible logit jumps.) The attention itself is an
//! online-softmax kernel over the K/V blocks, whose float rounding differs
//! from candle's matmul attention in 0.6.x: where two logits are within about
//! 0.1 of each other a greedy reply can differ from 0.6.x's.
//!
//! # Speculative decoding (opt-in)
//!
//! [`LocalModelConfig::drafter`] attaches Gemma 4's multi-token-prediction
//! assistant (preset `gemma4-e2b-mtp-f16`) to a Gemma 4 E2B target. The
//! drafter proposes up to [`LocalModelConfig::draft_tokens`] tokens from the
//! target's last hidden state, sharing the target's K/V cache; the target
//! verifies them in one batched pass and keeps the longest prefix it agrees
//! with. A speculative round always verifies its first draft and adds each
//! further one only while it pays: the engine predicts a draft's acceptance
//! from the drafter's own chained token probabilities (calibrated against
//! the acceptance it observes) and keeps it while it raises the round's
//! expected tokens per second under the measured verification cost per row
//! count, up to `draft_tokens`.
//! At temperature 0 the output equals plain greedy decoding; above 0 the
//! standard speculative-sampling acceptance rule (accept with
//! `min(1, p/q)`, resample the rejection from `max(0, p - q)`) preserves the
//! target's sampling distribution. An engine-side governor measures the
//! tokens per second of plain and speculative rounds and drafts only while
//! speculation is faster, re-probing with exponential back-off, so the worst
//! case stays near plain decoding (a seeded sampling run always drafts the
//! full `draft_tokens`, so equal prompts still reproduce equal tokens).
//! Measured on Gemma 4 E2B Q4_0 with the E2B assistant at the default
//! `draft_tokens` 5 (128-token replies, quiet development machine, speed
//! against plain decoding on the same device): CPU 1.32x on a code prompt,
//! 1.31x on a list, 1.01x on prose; Vulkan (RX 9070 XT) 0.97x, 0.97x and
//! 1.00x: GPU single-token decoding is host-bound (38 tok/s), so the
//! drafter's own launches cost as much as the verification saves, and the
//! governor keeps the run within about 3 % of plain decoding. Drafting is
//! off by default; it pays on the CPU.
//!
//! `DrafterSource::InTrunk` drafts with the prediction head embedded in a
//! Qwen 3.5 `*-MTP` file (for example `unsloth/Qwen3.5-0.8B-MTP-GGUF`): no
//! second file, CPU only. The head is one gated-attention block with its own
//! K/V cache over every position (4 KB per position, 21.7 MB of weights for
//! the 0.8B model), fed the trunk's final-norm hidden row of the last position
//! and the next token; a chain of drafts feeds the head's own normed output
//! back. The target verifies and rolls back exactly as above (a hybrid model
//! rolls back at most 16 drafts, one verification call), so greedy output
//! equals plain decoding bit for bit. Measured on Qwen3.5-0.8B Q8_0 (128-token
//! replies, quiet CPU, plain decoding 55 tok/s): at the default five drafts
//! per round a code prompt 1.59x, a list 1.40x, creative prose 0.98x (its
//! text is hard to predict, so the governor keeps drafting mostly off), the
//! three together 1.26x (67 % of drafted tokens accepted); at three drafts
//! 1.86x, 1.37x, 0.99x and 1.32x. A draft step costs about 1 ms against a
//! 19-22 ms trunk step because it ranks only the first 65,536 of the 248,320
//! token ids: 5x cheaper than the whole vocabulary for 4 % fewer accepted
//! tokens per round.
//!
//! A DFlash block drafter (`DrafterSource::File` on a GGUF of architecture
//! `dflash`, the file llama.cpp's converter writes
//! (`ggml-org/Muse-Glimmer-30B-GGUF`'s `dflash-*.gguf`), or the
//! `meta-models/Muse-Glimmer-30B-assistant` checkpoint as a
//! [`ModelSource::Checkpoint`], which the converter writes in the same layout
//! minus the target's vocabulary) proposes a whole block in one pass instead
//! of chaining one pass per token. It reads the residual stream of five
//! target layers (the outputs of layers 1, 13, 25, 37 and 49 of Muse
//! Glimmer's 52) at every accepted position, keeps its own K/V of those
//! features (one ring of the last 2,048 positions, 82 MB at the Muse size),
//! runs five sliding-window layers over a block of `1 + draft_tokens` rows (at
//! most 16: the pending token and mask tokens, the target's raw embeddings,
//! attending to each other in both directions and to the context) and ranks the
//! block's rows with the target's own LM head, so one pass yields every draft
//! of the round. Its cost is flat per round, not per draft: the engine's cost
//! model prices it that way (`take_block_draft`), and the governor measures
//! whole rounds as before. Greedy output equals plain decoding bit for bit and
//! sampling keeps the target's distribution, as for every drafter. CPU only
//! (the target's per-layer features come from the CPU forward pass); like the
//! in-trunk head it observes every position of its one sequence, so the
//! engine runs one request at a time. Published numbers for the method (not
//! measured here): average accepted length 4.35 to 7.84 tokens on Qwen3-4B at
//! blocks of 16 (the DFlash paper, arXiv 2602.06036, Table 1), 2.4 to 5.0 for
//! `RedHatAI/Mistral-Small-4-119B-2603.dflash` at seven drafts, and for Muse
//! Glimmer 30B a 3.1x speedup on an RTX 5090 with llama.cpp and 1.5x/1.8x on
//! Apple M4/M5 Max (the Muse Glimmer assistant's model card).

mod chat;
mod convert;
mod download;
mod engine;
mod model;
mod output;
mod sampler;
mod tokenizer;

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures_core::Stream;
use tokio::sync::{mpsc, oneshot};

use self::chat::{ChatFormat, ChatTemplate, chat_prompt};
pub use self::model::tier::{ExpertHeat, ExpertTierStats};
use self::tokenizer::Tokenizer;
use crate::client::{CucaClient, ProviderDispatch, ResponseMetadataHandle};
use crate::error::CucaError;
use crate::request::UnifiedRequest;
use crate::types::{MessageContentBlock, ProviderEndpoint};

/// Message shared by `CucaClientBuilder::build` and `dispatch_local` when the
/// selected provider is `Local` but no model was attached.
pub(crate) const MODEL_REQUIRED: &str =
    "ProviderEndpoint::Local requires CucaClientBuilder::with_local_model";

/// Jobs that may wait behind the running one.
const JOB_QUEUE: usize = 16;
/// Output blocks buffered between the engine thread and the stream.
const EVENT_BUFFER: usize = 64;
/// Context length a [`LocalModelConfig`] without an explicit one aims for.
const DEFAULT_CONTEXT_LENGTH: usize = 8192;

/// The context length to run with: the configured value, or the default
/// capped at the model's own maximum `model_max`.
pub(crate) fn resolve_context_length(configured: Option<usize>, model_max: usize) -> usize {
    configured.unwrap_or_else(|| DEFAULT_CONTEXT_LENGTH.min(model_max))
}

/// Default [`LocalModelConfig::draft_tokens`].
const DEFAULT_DRAFT_TOKENS: usize = 5;
/// Largest accepted [`LocalModelConfig::draft_tokens`].
pub(crate) const MAX_DRAFT_TOKENS: usize = 16;
/// Connect timeout of the one-off download client.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// One entry of the verified preset table, see [`ModelSource::PRESETS`].
///
/// Every entry pins an exact repository commit and the SHA-256 of its file,
/// and was loaded and checked by this crate's live test suite
/// (`tests/provider_local.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelPreset {
    /// Name accepted by `str::parse::<ModelSource>` (ASCII case-insensitive):
    /// `family-size-quant`, for example `qwen3-0.6b-q8_0`.
    pub name: &'static str,
    /// Hugging Face repository id.
    pub repo: &'static str,
    /// Commit hash of the repository the download is pinned to.
    pub revision: &'static str,
    /// File name inside the repository.
    pub file: &'static str,
    /// Lowercase hex SHA-256 of the file (the `lfs.oid` of that revision).
    pub sha256: &'static str,
    /// Size of the file in bytes.
    pub size_bytes: u64,
    /// The file's `general.architecture`.
    pub architecture: &'static str,
    /// `true` for a multi-token-prediction drafter, which is used through
    /// [`LocalModelConfig::drafter`] next to its target model and cannot be
    /// loaded as the main model.
    pub drafter: bool,
    /// The parts of a model published as a split GGUF (`llama-gguf-split`):
    /// every `<base>-0000i-of-0000N.gguf` file with its digest and size, in
    /// part order. Empty for a single-file preset. For a split preset `file`
    /// and `sha256` are those of the first part and `size_bytes` is the sum of
    /// the parts; loading downloads and verifies every part, merges them into
    /// one cached `<base>.gguf` and deletes the parts.
    pub parts: &'static [PresetPart],
}

/// One file of a split preset, see [`ModelPreset::parts`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresetPart {
    /// File name inside the repository.
    pub file: &'static str,
    /// Lowercase hex SHA-256 of the file (the `lfs.oid` of that revision).
    pub sha256: &'static str,
    /// Size of the file in bytes.
    pub size_bytes: u64,
}

impl ModelPreset {
    /// The pinned, checksummed [`ModelSource`] of this preset.
    pub fn source(&self) -> ModelSource {
        ModelSource::HuggingFace {
            repo: self.repo.to_string(),
            revision: self.revision.to_string(),
            file: self.file.to_string(),
            sha256: Some(self.sha256.to_string()),
            parts: self
                .parts
                .iter()
                .map(|p| ModelPart {
                    file: p.file.to_string(),
                    sha256: Some(p.sha256.to_string()),
                })
                .collect(),
        }
    }
}

// Revisions are the repositories' commit hashes and the digests the `lfs.oid`
// of each file, both read from the Hugging Face API.
const fn preset(
    name: &'static str,
    repo: &'static str,
    revision: &'static str,
    file: &'static str,
    sha256: &'static str,
    size_bytes: u64,
    architecture: &'static str,
) -> ModelPreset {
    ModelPreset {
        name,
        repo,
        revision,
        file,
        sha256,
        size_bytes,
        architecture,
        drafter: false,
        parts: &[],
    }
}

const PRESET_TABLE: [ModelPreset; 24] = [
    preset(
        "gemma4-e2b-q4_0",
        "ggml-org/gemma-4-E2B-it-GGUF",
        "b4243c156154b6dca9324415f8c7ccc098b4aed1",
        "gemma-4-E2B-it-Q4_0.gguf",
        "8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52",
        2_841_481_184,
        "gemma4",
    ),
    preset(
        "gemma4-e2b-q8_0",
        "ggml-org/gemma-4-E2B-it-GGUF",
        "b4243c156154b6dca9324415f8c7ccc098b4aed1",
        "gemma-4-E2B-it-Q8_0.gguf",
        "996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a",
        4_967_497_152,
        "gemma4",
    ),
    preset(
        "gemma4-e4b-q4_0",
        "ggml-org/gemma-4-E4B-it-GGUF",
        "b8093469224f83f5c38f691eb906c380e9e63114",
        "gemma-4-E4B-it-Q4_0.gguf",
        "a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2",
        4_590_807_392,
        "gemma4",
    ),
    preset(
        "gemma4-e4b-q8_0",
        "ggml-org/gemma-4-E4B-it-GGUF",
        "b8093469224f83f5c38f691eb906c380e9e63114",
        "gemma-4-E4B-it-Q8_0.gguf",
        "34be82b17b4942d389b9b527170c4b058027abdd32531fda063d3d97dd8ce80a",
        8_031_242_688,
        "gemma4",
    ),
    preset(
        "gemma3-270m-q8_0",
        "ggml-org/gemma-3-270m-it-GGUF",
        "e7647be17ae1108f2f605ed061ca0608b171afff",
        "gemma-3-270m-it-Q8_0.gguf",
        "0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3",
        291_545_600,
        "gemma3",
    ),
    preset(
        "gemma3-1b-q4_0",
        "unsloth/gemma-3-1b-it-GGUF",
        "f0b45be0aac41bd6a100a4b5734cad5f67255bfb",
        "gemma-3-1b-it-Q4_0.gguf",
        "27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e",
        721_918_496,
        "gemma3",
    ),
    preset(
        "qwen3-0.6b-q8_0",
        "Qwen/Qwen3-0.6B-GGUF",
        "23749fefcc72300e3a2ad315e1317431b06b590a",
        "Qwen3-0.6B-Q8_0.gguf",
        "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
        639_446_688,
        "qwen3",
    ),
    preset(
        "qwen3-1.7b-q8_0",
        "Qwen/Qwen3-1.7B-GGUF",
        "90862c4b9d2787eaed51d12237eafdfe7c5f6077",
        "Qwen3-1.7B-Q8_0.gguf",
        "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
        1_834_426_016,
        "qwen3",
    ),
    preset(
        "r1-distill-qwen-1.5b-q4_0",
        "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "9cc28b17e86fa2415fcb070f8ee5ec27c965aa61",
        "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
        "b3af887d0a015b39fab2395e4faf682c1a81a6a3fd09a43f0d4292f7d94bf4d0",
        1_068_807_776,
        "qwen2",
    ),
    preset(
        "r1-distill-llama-8b-q4_0",
        "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "615f8936e16dfde29dcc00be71145d4d5ce8ed53",
        "DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf",
        "4b041b222221e8e37d6faffe7b8184392a239c18c2d9a22e89e485b9a5a321e0",
        4_675_894_720,
        "llama",
    ),
    preset(
        "smollm3-3b-q4_0",
        "bartowski/HuggingFaceTB_SmolLM3-3B-GGUF",
        "86b3536ed1ca0dcb4716745642db4e6804bf3d32",
        "HuggingFaceTB_SmolLM3-3B-Q4_0.gguf",
        "1e0e56d458e6b6d5ea18d0e5fe598cca252450308f1648afab49984fceae47da",
        1_811_455_808,
        "smollm3",
    ),
    preset(
        "phi4-mini-q4_0",
        "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "7ff82c2aaa4dde30121698a973765f39be5288c0",
        "microsoft_Phi-4-mini-instruct-Q4_0.gguf",
        "2124412a2d3410dd05c5d01796457283812210633165a2a86c909f815971518e",
        2_331_442_560,
        "phi3",
    ),
    preset(
        "phi4-mini-q4_k_m",
        "unsloth/Phi-4-mini-instruct-GGUF",
        "78eb92a46fc37e6b524df991ed9aca9bc6aa7b80",
        "Phi-4-mini-instruct-Q4_K_M.gguf",
        "88c00229914083cd112853aab84ed51b87bdf6b9ce42f532d8c85c7c63b1730a",
        2_491_874_272,
        "phi3",
    ),
    preset(
        "granite3.3-2b-q4_0",
        "ibm-granite/granite-3.3-2b-instruct-GGUF",
        "7cdf86ccd1f1bb3491c9b7017b033f2e51367397",
        "granite-3.3-2b-instruct-Q4_0.gguf",
        "d5da3f5bb5a6ae442f830bdde0df4fcccb7e8bd395dc9ee5d28e704db4065592",
        1_453_389_088,
        "granite",
    ),
    preset(
        "olmo2-1b-q4_0",
        "allenai/OLMo-2-0425-1B-Instruct-GGUF",
        "62f8c199538474c3e33ed5d7e0580abd66686a27",
        "OLMo-2-0425-1B-Instruct-Q4_0.gguf",
        "c5efa891c82203620b4ad12f2676340b330641bc9790a5eda4f6586e855e222c",
        892_261_536,
        "olmo2",
    ),
    preset(
        "qwen3-30b-a3b-q4_0",
        "unsloth/Qwen3-30B-A3B-GGUF",
        "d5b1d57bd0b504ac62ae6c725904e96ef228dc74",
        "Qwen3-30B-A3B-Q4_0.gguf",
        "8b8e9febb6ed326d91a016f5d465cf5b55a0b328d157a43e2242fda71fde9c40",
        17_379_988_032,
        "qwen3moe",
    ),
    preset(
        "olmoe-1b-7b-q4_0",
        "allenai/OLMoE-1B-7B-0125-Instruct-GGUF",
        "2ac1d27317927518c9ef7bd99f91f3f1e4ee288d",
        "OLMoE-1B-7B-0125-Instruct-Q4_0.gguf",
        "df43bd738dfdcbe889c5f854120c271e9e53e2c624f212ac2523f8bc11ae9bf3",
        3_928_038_048,
        "olmoe",
    ),
    preset(
        "qwen3.5-0.8b-q8_0",
        "unsloth/Qwen3.5-0.8B-MTP-GGUF",
        "cf8a611f6ed2c2060046219a19f12cd3d5ecd67c",
        "Qwen3.5-0.8B-Q8_0.gguf",
        "c54f8b67069c70085b98440de696b44da8250250ac69a961b41133def876e262",
        833_592_736,
        "qwen35",
    ),
    preset(
        "qwen3.6-35b-a3b-q4_0",
        "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
        "5c2410d71524f4f72b023ce8daf7a80528226d5f",
        "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf",
        "52312daa5b2190c1f5723d33c3315c01c55af4206f6c6e6eb63f3d8dd52bb85e",
        20_836_243_072,
        "qwen35moe",
    ),
    // imatrix Q4_0 with a Q6_K `output` and `attn_q` in 26 layers and a Q4_1
    // `ffn_down` in 6 (all served by the row-batched kernels). It embeds the earlier Muse chat template, which renders the same
    // bytes as the current one except when the caller's system prompt itself
    // mentions "Reasoning effort": the current template rewrites that to
    // "Reasoning strength", this one appends a second directive.
    preset(
        "muse-glimmer-30b-q4_0",
        "bartowski/Muse-Glimmer-30B-GGUF",
        "01e1b7c30e19e1823ee9a73dec51d12f59ec4857",
        "Muse-Glimmer-30B-Q4_0.gguf",
        "95fd818c08f2030a51805548269690e1915fc84088e4322273c5f317e2696999",
        16_265_565_216,
        "muse-glimmer",
    ),
    // imatrix Q4_0 (Q4_1 and Q6_K in places) with the earlier chat template of
    // Ministral 3 Reasoning: identical to the current one except that the
    // reasoning of earlier assistant turns is not written back into the
    // history. mistralai publishes no Q4_0 of this model.
    preset(
        "ministral3-3b-reasoning-q4_0",
        "bartowski/mistralai_Ministral-3-3B-Reasoning-2512-GGUF",
        "36cf98fc920c42a4c89c78880723a8ec4aabe931",
        "mistralai_Ministral-3-3B-Reasoning-2512-Q4_0.gguf",
        "72ba5a30221864a7355c92336e639214e122dad4d11345139d2669d7c8bf7c3c",
        2_046_374_368,
        "mistral3",
    ),
    // imatrix Q4_0 of Ministral 3 Instruct. Its embedded chat template is the
    // earlier one: the same except for the wording of the default system
    // prompt (the current template adds a tool-calling paragraph there).
    preset(
        "ministral3-3b-instruct-q4_0",
        "bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF",
        "0a9035302dea19872c96076896304552890a73c2",
        "mistralai_Ministral-3-3B-Instruct-2512-Q4_0.gguf",
        "c2993f4b86ddcae94031b2e384169889ceecd51b148ceda30c0f826fcce721c8",
        2_046_375_904,
        "mistral3",
    ),
    // Two parts of `llama-gguf-split`: merged into one cached file at load.
    // imatrix Q4_0 (Q4_1 in the layers' `ffn_down_exps`, Q5_0 in places, Q8_0
    // for the shared experts and the latent projection, Q6_K `output`).
    ModelPreset {
        parts: &[
            PresetPart {
                file: "mistralai_Mistral-Small-4-119B-2603-Q4_0/mistralai_Mistral-Small-4-119B-2603-Q4_0-00001-of-00002.gguf",
                sha256: "db43a5e13e55b0b75de52e66cc4e2ccfa2863a940742df880e15c4b99f9199c8",
                size_bytes: 39_924_112_032,
            },
            PresetPart {
                file: "mistralai_Mistral-Small-4-119B-2603-Q4_0/mistralai_Mistral-Small-4-119B-2603-Q4_0-00002-of-00002.gguf",
                sha256: "3e325c6e7c0bddbbd536955983672717edbe25413cde0e317b61531ca6887112",
                size_bytes: 27_785_630_592,
            },
        ],
        ..preset(
            "mistral-small-4-119b-q4_0",
            "bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF",
            "f569d04efea9546de10ec040819908fa4e8f1c92",
            "mistralai_Mistral-Small-4-119B-2603-Q4_0/mistralai_Mistral-Small-4-119B-2603-Q4_0-00001-of-00002.gguf",
            "db43a5e13e55b0b75de52e66cc4e2ccfa2863a940742df880e15c4b99f9199c8",
            67_709_742_624,
            "mistral4",
        )
    },
    ModelPreset {
        drafter: true,
        ..preset(
            "gemma4-e2b-mtp-f16",
            "unsloth/gemma-4-E2B-it-GGUF",
            "0314792d7f1f7e229411f620751375812bb9faf2",
            "MTP/mtp-gemma-4-E2B-it-F16.gguf",
            "cf045574920f9889cebd21db2e5019535d01155947ae00f9e498714954c0fd74",
            170_193_984,
            "gemma4-assistant",
        )
    },
];

/// Where [`LocalModel::load`] gets the model from.
///
/// Any GGUF whose architecture the engine supports works (see the module
/// docs): the loader derives every dimension from the file. The verified
/// presets are in [`ModelSource::PRESETS`]; other files are described with
/// [`ModelSource::hugging_face`] or [`ModelSource::File`]. A Hugging Face
/// safetensors checkpoint ([`ModelSource::Checkpoint`], or a local checkpoint
/// directory, [`ModelSource::Dir`]) is converted once to a cached GGUF at the
/// weight precision of [`LocalModelConfig::convert`], then loaded like any
/// other file. A source also parses from a string (`str::parse`): a preset
/// name (ASCII case-insensitive), `repo:revision:file` (a GGUF) or
/// `repo:revision` (a checkpoint).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// A file hosted on Hugging Face, downloaded once into the cache and
    /// pinned to an exact `revision`.
    HuggingFace {
        /// Repository id, e.g. `ggml-org/gemma-4-E2B-it-GGUF`.
        repo: String,
        /// Commit hash the download is pinned to.
        revision: String,
        /// File name inside the repository.
        file: String,
        /// Expected lowercase hex SHA-256 of the file; `None` skips the check.
        sha256: Option<String>,
        /// The parts of a split GGUF with their digests, in part order; empty
        /// for a single file. When `file` is a split name
        /// (`<base>-00001-of-0000N.gguf`), loading downloads every part
        /// (named by `parts`, or derived from `file` and unverified when
        /// `parts` is empty), merges them into `<base>.gguf` next to where
        /// the parts were downloaded and deletes the parts. `parts[0]` must
        /// be `file`.
        parts: Vec<ModelPart>,
    },
    /// A GGUF file already on disk; nothing is downloaded. A file named like a
    /// split part (`<base>-00001-of-0000N.gguf`; any part names the set) is
    /// merged with its siblings in the same directory into one GGUF under
    /// `<cache>/merged/`, and that file is loaded; the parts stay in place.
    File(PathBuf),
    /// A Hugging Face safetensors checkpoint (`config.json`, the weight
    /// shards, the tokenizer files), downloaded into the cache pinned to
    /// `revision` and converted once to a GGUF there.
    Checkpoint {
        /// Repository id, e.g. `Qwen/Qwen3-0.6B`.
        repo: String,
        /// Commit hash the download is pinned to.
        revision: String,
    },
    /// A checkpoint directory already on disk (the same files as
    /// [`ModelSource::Checkpoint`]); converted once to a GGUF in the cache.
    /// The conversion is keyed by the directory's path and the size and
    /// modification time of its `config.json` and weight shards.
    Dir(PathBuf),
}

/// One file of a split GGUF set on Hugging Face, see
/// [`ModelSource::HuggingFace::parts`](ModelSource).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPart {
    /// File name inside the repository.
    pub file: String,
    /// Expected lowercase hex SHA-256 of the file; `None` skips the check.
    pub sha256: Option<String>,
}

impl ModelSource {
    /// Every verified preset: the models this crate's live suite runs, each
    /// pinned to a repository commit and a SHA-256, plus the multi-token
    /// prediction drafter. Sorted by family; the first entry is the default
    /// ([`ModelSource::default_model`]).
    pub const PRESETS: &'static [ModelPreset] = &PRESET_TABLE;

    /// The preset called `name` (ASCII case-insensitive), if any.
    pub fn preset(name: &str) -> Option<&'static ModelPreset> {
        let name = name.trim();
        Self::PRESETS
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
    }

    /// The default model: preset `gemma4-e2b-q4_0`, Gemma 4 E2B
    /// instruction-tuned at Q4_0 from `ggml-org/gemma-4-E2B-it-GGUF` at a
    /// pinned revision with its SHA-256 (2.84 GB). The default source of
    /// [`LocalModelConfig`].
    pub fn default_model() -> Self {
        Self::PRESETS[0].source()
    }

    /// An arbitrary GGUF file on Hugging Face, without a checksum.
    ///
    /// `revision` should be a commit hash: the cache is keyed by it, so a
    /// moving ref such as `main` would pin a stale file forever. The download
    /// is not verified (`sha256` is `None`); build
    /// [`ModelSource::HuggingFace`] directly to pin a digest.
    pub fn hugging_face(
        repo: impl Into<String>,
        revision: impl Into<String>,
        file: impl Into<String>,
    ) -> Self {
        ModelSource::HuggingFace {
            repo: repo.into(),
            revision: revision.into(),
            file: file.into(),
            sha256: None,
            parts: Vec::new(),
        }
    }
}

/// Parses a preset name from [`ModelSource::PRESETS`] (ASCII
/// case-insensitive) or `repo:revision:file` (an unverified
/// [`ModelSource::hugging_face`] source; none of the three parts contains
/// `:`).
impl std::str::FromStr for ModelSource {
    type Err = CucaError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if let Some(preset) = Self::preset(s) {
            return Ok(preset.source());
        }
        let parts: Vec<&str> = s.split(':').collect();
        match parts.as_slice() {
            [repo, revision, file]
                if !repo.is_empty() && !revision.is_empty() && !file.is_empty() =>
            {
                Ok(Self::hugging_face(*repo, *revision, *file))
            }
            [repo, revision]
                if !revision.is_empty()
                    && repo.split_once('/').is_some_and(|(owner, name)| {
                        !owner.is_empty() && !name.is_empty() && !name.contains('/')
                    }) =>
            {
                Ok(Self::Checkpoint {
                    repo: (*repo).to_string(),
                    revision: (*revision).to_string(),
                })
            }
            _ => Err(CucaError::Config(format!(
                "unknown model `{s}`: expected a preset ({}), `repo:revision:file` (a GGUF), or `repo:revision` (a checkpoint)",
                Self::PRESETS
                    .iter()
                    .map(|p| p.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }
}

/// Default sampling parameters of a [`LocalModel`].
///
/// Each value is overridden per request by the matching [`UnifiedRequest`]
/// field (`set_temperature`, `set_top_k`, `set_top_p`, `set_seed`) and holds
/// for requests that set none. A request `seed` restarts the generator for
/// that request only; the load-time `seed` restarts it for every request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    /// Softmax temperature; `0.0` selects the greedy argmax.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens; `0` disables the filter.
    pub top_k: usize,
    /// Nucleus threshold in `(0, 1]`; `1.0` disables the filter.
    pub top_p: f32,
    /// Fixed RNG seed: with `Some`, every request restarts the generator from
    /// the seed, so equal prompts sample equal tokens. `None` seeds the
    /// generator from the system once at load.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
            seed: None,
        }
    }
}

/// Which GPU adapter a wgpu-backed device ([`LocalDevice::Vulkan`],
/// [`LocalDevice::Metal`]) uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuAdapter {
    /// The adapter the graphics API reports first (wgpu's default choice).
    #[default]
    Default,
    /// The `n`-th discrete GPU, counting from 0.
    Discrete(usize),
    /// The `n`-th integrated GPU, counting from 0.
    Integrated(usize),
}

/// Where a [`LocalModel`] runs its forward pass.
///
/// Every variant exists in every build; selecting a GPU backend whose cargo
/// feature is off fails [`LocalModel::load`] with [`CucaError::Config`]
/// naming the feature ([`LocalDevice::required_feature`]). The CPU runs on
/// candle; the GPU backends run on burn (CubeCL). A device also parses from a
/// string (`str::parse`, ASCII case-insensitive): `cpu`; `vulkan`,
/// `vulkan:<n>` (the `n`-th discrete GPU) or `vulkan:integrated:<n>`;
/// `metal` in the same three forms; `cuda` or `cuda:<n>`; `rocm` or
/// `rocm:<n>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LocalDevice {
    /// candle on the CPU (always available).
    #[default]
    Cpu,
    /// burn's wgpu backend on Vulkan (`provider-local-vulkan`).
    Vulkan(GpuAdapter),
    /// burn's wgpu backend on Metal, Apple only at runtime
    /// (`provider-local-metal`).
    Metal(GpuAdapter),
    /// burn's CUDA backend on the GPU with this index (`provider-local-cuda`).
    Cuda(usize),
    /// burn's ROCm (HIP) backend on the GPU with this index, Linux only
    /// (`provider-local-rocm`).
    Rocm(usize),
}

impl LocalDevice {
    /// The cargo feature this device needs, `None` for [`LocalDevice::Cpu`].
    pub fn required_feature(self) -> Option<&'static str> {
        match self {
            LocalDevice::Cpu => None,
            LocalDevice::Vulkan(_) => Some("provider-local-vulkan"),
            LocalDevice::Metal(_) => Some("provider-local-metal"),
            LocalDevice::Cuda(_) => Some("provider-local-cuda"),
            LocalDevice::Rocm(_) => Some("provider-local-rocm"),
        }
    }

    /// Whether this build includes the backend the device needs.
    fn is_compiled(self) -> bool {
        match self {
            LocalDevice::Cpu => true,
            LocalDevice::Vulkan(_) => cfg!(feature = "provider-local-vulkan"),
            LocalDevice::Metal(_) => cfg!(feature = "provider-local-metal"),
            LocalDevice::Cuda(_) => cfg!(feature = "provider-local-cuda"),
            LocalDevice::Rocm(_) => cfg!(feature = "provider-local-rocm"),
        }
    }
}

/// The [`CucaError::Config`] for a device whose backend is compiled out, or
/// `None` when the device is available in this build.
pub(crate) fn device_unavailable(device: LocalDevice) -> Option<CucaError> {
    match device.required_feature() {
        Some(feature) if !device.is_compiled() => Some(CucaError::Config(format!(
            "device {device:?} requires the `{feature}` cargo feature"
        ))),
        _ => None,
    }
}

/// Parses `cpu`, `vulkan[:<n>|:integrated:<n>]`, `metal[:<n>|:integrated:<n>]`,
/// `cuda[:<n>]` or `rocm[:<n>]`, ASCII case-insensitive; see [`LocalDevice`].
impl std::str::FromStr for LocalDevice {
    type Err = CucaError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let lower = s.to_ascii_lowercase();
        let parts: Vec<&str> = lower.split(':').collect();
        let index = |n: &str| n.parse::<usize>().ok();
        let adapter = |rest: &[&str]| match rest {
            [] => Some(GpuAdapter::Default),
            [n] => index(n).map(GpuAdapter::Discrete),
            ["integrated", n] => index(n).map(GpuAdapter::Integrated),
            _ => None,
        };
        let device = match parts.split_first() {
            Some((&"cpu", [])) => Some(LocalDevice::Cpu),
            Some((&"vulkan", rest)) => adapter(rest).map(LocalDevice::Vulkan),
            Some((&"metal", rest)) => adapter(rest).map(LocalDevice::Metal),
            Some((&"cuda", [])) => Some(LocalDevice::Cuda(0)),
            Some((&"cuda", [n])) => index(n).map(LocalDevice::Cuda),
            Some((&"rocm", [])) => Some(LocalDevice::Rocm(0)),
            Some((&"rocm", [n])) => index(n).map(LocalDevice::Rocm),
            _ => None,
        };
        device.ok_or_else(|| {
            CucaError::Config(format!(
                "unknown local device `{s}`; expected cpu, vulkan[:<n>|:integrated:<n>], metal[...], cuda[:<n>] or rocm[:<n>]"
            ))
        })
    }
}

/// Model download progress, reported to a [`DownloadProgressCallback`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadProgress {
    /// Bytes of the file present on disk so far, resumed bytes included.
    pub downloaded_bytes: u64,
    /// Total file size when the server reported it.
    pub total_bytes: Option<u64>,
}

/// Shared callback invoked with [`DownloadProgress`] while a model downloads.
///
/// It runs on the engine thread during [`LocalModel::load`]; keep it cheap.
#[derive(Clone)]
pub struct DownloadProgressCallback(Arc<dyn Fn(DownloadProgress) + Send + Sync>);

impl DownloadProgressCallback {
    /// Wrap `f` as a progress callback.
    pub fn new(f: impl Fn(DownloadProgress) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
}

impl fmt::Debug for DownloadProgressCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DownloadProgressCallback(..)")
    }
}

/// Where the multi-token-prediction head of [`LocalModelConfig::drafter`]
/// comes from.
///
/// A [`ModelSource`] converts with `.into()` (it becomes
/// [`DrafterSource::File`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrafterSource {
    /// A separate drafter GGUF file, resolved and downloaded like any
    /// [`ModelSource`]: Gemma 4's assistant (preset `gemma4-e2b-mtp-f16`),
    /// which pairs with a Gemma 4 E2B target, shares the target's K/V cache
    /// and runs on the target's device.
    File(ModelSource),
    /// The prediction head embedded in the main model's own GGUF file:
    /// Qwen 3.5 `*-MTP` files (for example `unsloth/Qwen3.5-0.8B-MTP-GGUF`)
    /// carry one extra block after their trunk that predicts the token after
    /// the next. It loads from the same file (nothing more to download),
    /// keeps a K/V cache of its own (4 KB per context position for the 0.8B
    /// model) and runs on the CPU only. A model whose file has no such block
    /// fails to load with an error that says so.
    InTrunk,
}

impl From<ModelSource> for DrafterSource {
    fn from(source: ModelSource) -> Self {
        Self::File(source)
    }
}

/// Weight precision of a checkpoint converted to GGUF
/// ([`ConvertConfig::quant`]).
///
/// Every 2-D matrix whose row length is a multiple of 32 (attention and FFN
/// projections, expert stacks, the embeddings and the output head) is stored at
/// this precision (at `Q4_0` the token embedding and the output head stay at
/// Q8_0 and Gemma 4's `per_layer_model_proj` at F16, as in llama.cpp's Q4_0
/// files); norms, biases, scales, RoPE factors, the router and the
/// recurrent-layer parameters stay `f32`. Parses from a string (`str::parse`,
/// ASCII case-insensitive): `f16`, `q8_0` or `q4_0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightQuant {
    /// IEEE half precision: 2 bytes per weight, the closest to the checkpoint.
    F16,
    /// 8-bit blocks of 32 (8.5 bits per weight): the default, indistinguishable
    /// from `f16` in practice at half its size.
    #[default]
    Q8_0,
    /// 4-bit blocks of 32 (4.5 bits per weight): smallest and fastest, with
    /// visible quantization noise on small models.
    Q4_0,
}

impl WeightQuant {
    /// The lowercase name: `f16`, `q8_0` or `q4_0`; also the stem suffix of the
    /// converted file (`model-q8_0.gguf`).
    pub fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }
}

impl std::str::FromStr for WeightQuant {
    type Err = CucaError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        [Self::F16, Self::Q8_0, Self::Q4_0]
            .into_iter()
            .find(|q| q.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| {
                CucaError::Config(format!(
                    "unknown weight quantization `{s}`; expected f16, q8_0 or q4_0"
                ))
            })
    }
}

/// How a safetensors checkpoint ([`ModelSource::Checkpoint`],
/// [`ModelSource::Dir`]) is converted to GGUF; ignored for GGUF sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConvertConfig {
    /// Precision of the weight matrices; [`WeightQuant::Q8_0`] by default.
    /// Each precision is converted once and cached next to the others.
    pub quant: WeightQuant,
}

/// Where a mixture-of-experts model's routed experts live
/// ([`LocalModelConfig::experts`]); see [`ExpertTiers`].
///
/// Also parses from a string (`str::parse`, ASCII case-insensitive):
/// `resident`, or `tiered:` followed by comma-separated `ram=<size>`,
/// `device=<size>` and `threads=<n>` in any order (each at most once), where
/// `<size>` is a byte count with an optional `k`, `m` or `g` suffix (powers
/// of 1024): `tiered:ram=4g`, `tiered:device=6g,ram=12g,threads=8`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ExpertPlacement {
    /// Every expert is loaded at startup and stays in RAM (CPU) or device
    /// memory (GPU): memory is the file size.
    #[default]
    Resident,
    /// The experts stay in the GGUF file and are loaded on demand through
    /// bounded caches; mixture-of-experts models only.
    Tiered(ExpertTiers),
}

/// The caches of [`ExpertPlacement::Tiered`].
///
/// The dense weights (attention, norms, routers, embeddings, head) stay
/// where [`ExpertPlacement::Resident`] puts them. Each routed expert (the
/// gate, up and down matrices of one expert of one layer) is read from the
/// GGUF file the first time a token routes to it, by a pool of
/// `loader_threads` reader threads, and kept in a cache:
///
/// - on [`LocalDevice::Cpu`], one RAM tier of `ram_bytes`, holding the
///   quantized matmul weights (an expert costs its file bytes);
/// - on a GPU device, a device tier of `device_bytes` holding the uploaded
///   weights, in front of an optional RAM tier of `ram_bytes` holding
///   upload-ready host copies (`0` reads every device miss from the file).
///
/// A budget becomes the same number of slots in every MoE layer: `budget /
/// (bytes of one expert summed over the MoE layers)`, at most the expert
/// count. A tier must hold the model's experts-per-token (`n_used`) in
/// every layer; a smaller budget fails [`LocalModel::load`] with
/// [`CucaError::Config`] naming the minimum. A full tier evicts the expert
/// with the lowest learned heat (decaying pick counts) that the running layer
/// pass no longer needs, then the least recently used. The tiers never hold
/// more than their budget; an expert a layer pass is computing may outlive
/// its eviction for that pass (one on the CPU; on a GPU up to `n_used` in a
/// single-token pass, whose experts are all uploaded before it computes).
///
/// Placement only decides speed: an expert is built from the same file
/// bytes and computed in the same order as in the resident mode, so the
/// output is the resident mode's on the same device (bit for bit on the
/// CPU, to float rounding on a GPU). [`LocalModel::expert_tier_stats`] reports hits, reads and resident
/// bytes; [`LocalModel::expert_heat`] exports the learned heat, which
/// `warm_from` feeds back in a later session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertTiers {
    /// Bytes of RAM for cached experts. Required on [`LocalDevice::Cpu`]
    /// (at least `n_used` experts per layer); on a GPU device `0` disables
    /// the RAM tier, any other value must hold `n_used` experts per layer.
    pub ram_bytes: u64,
    /// Bytes of device memory for cached experts on a GPU device (required
    /// there, at least `n_used` experts per layer); must be `0` on
    /// [`LocalDevice::Cpu`].
    pub device_bytes: u64,
    /// Disk reader threads, `1..=64` (default 4).
    pub loader_threads: usize,
    /// Heat from an earlier session of the same model
    /// ([`LocalModel::expert_heat`]): it seeds the learned heat, and the
    /// hottest experts of every layer are loaded at startup (device tier
    /// first, then RAM). `None` (the default) starts cold.
    pub warm_from: Option<ExpertHeat>,
}

/// Default [`ExpertTiers::loader_threads`].
const DEFAULT_LOADER_THREADS: usize = 4;
/// Largest accepted [`ExpertTiers::loader_threads`].
const MAX_LOADER_THREADS: usize = 64;

impl Default for ExpertTiers {
    fn default() -> Self {
        Self {
            ram_bytes: 0,
            device_bytes: 0,
            loader_threads: DEFAULT_LOADER_THREADS,
            warm_from: None,
        }
    }
}

/// Parses `resident` or `tiered:<key>=<value>,...`; see [`ExpertPlacement`].
impl std::str::FromStr for ExpertPlacement {
    type Err = CucaError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let bad = || {
            CucaError::Config(format!(
                "unknown expert placement `{s}`; expected resident or tiered:ram=<size>[,device=<size>][,threads=<n>]"
            ))
        };
        let lower = s.to_ascii_lowercase();
        if lower == "resident" {
            return Ok(Self::Resident);
        }
        let rest = lower.strip_prefix("tiered:").ok_or_else(bad)?;
        let size = |v: &str| -> Option<u64> {
            let (digits, shift) = match v.as_bytes().last()? {
                b'k' => (&v[..v.len() - 1], 10),
                b'm' => (&v[..v.len() - 1], 20),
                b'g' => (&v[..v.len() - 1], 30),
                _ => (v, 0),
            };
            digits.parse::<u64>().ok()?.checked_mul(1u64 << shift)
        };
        let mut tiers = ExpertTiers::default();
        let (mut ram, mut device, mut threads) = (false, false, false);
        for part in rest.split(',') {
            let (key, value) = part.split_once('=').ok_or_else(bad)?;
            match key {
                "ram" if !ram => {
                    ram = true;
                    tiers.ram_bytes = size(value).ok_or_else(bad)?;
                }
                "device" if !device => {
                    device = true;
                    tiers.device_bytes = size(value).ok_or_else(bad)?;
                }
                "threads" if !threads => {
                    threads = true;
                    tiers.loader_threads = value.parse().map_err(|_| bad())?;
                }
                _ => return Err(bad()),
            }
        }
        if !(ram || device) {
            return Err(bad());
        }
        Ok(Self::Tiered(tiers))
    }
}

/// Configuration for [`LocalModel::load`].
///
/// `Debug` redacts `hf_token`.
#[derive(Clone)]
pub struct LocalModelConfig {
    /// Where the GGUF file comes from.
    pub source: ModelSource,
    /// Cache directory for downloads; `None` resolves `CUCA_CACHE_DIR`, then
    /// the platform cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Hugging Face access token; `None` falls back to the `HF_TOKEN`
    /// environment variable, then to anonymous access.
    pub hf_token: Option<String>,
    /// Maximum tokens of prompt plus completion per request. The K/V block
    /// pools hold twice this many positions per full-attention layer group
    /// (shared by every running request; [`BatchingConfig::kv_tokens`] sets
    /// the pool), so a layer costs `2 * context_length * 2 * 4 * n_kv_head * head_dim` bytes
    /// of RAM (Gemma 4 E2B 422 MB at 8192, Qwen3-0.6B 458,752 bytes per
    /// context position, i.e. 3.76 GB at 8192: size it
    /// for the model; [`LocalModel::kv_cache_bytes`] reports the result; see
    /// [`LocalModel`]).
    /// `None` (the default)
    /// resolves to the smaller of 8192 and the model's own maximum, so a
    /// model trained for 2048 or 4096 positions loads without configuration.
    /// An explicit value must be at least 1 and at most the model's maximum,
    /// or [`LocalModel::load`] fails with [`CucaError::Config`]. On a GPU
    /// device the cache lives in device memory with the same size.
    pub context_length: Option<usize>,
    /// Default sampling parameters.
    pub sampling: SamplingConfig,
    /// Where the forward pass runs; [`LocalDevice::Cpu`] by default.
    pub device: LocalDevice,
    /// Optional multi-token-prediction drafter for self-speculative decoding:
    /// a separate file (the `gemma4-e2b-mtp-f16` preset pairs with the Gemma 4
    /// E2B presets, [`DrafterSource::File`]) or the head inside the main
    /// model's own file ([`DrafterSource::InTrunk`], Qwen 3.5 `*-MTP`). It
    /// needs a matching target; see the module docs for the exactness
    /// guarantee and measured speedups. `None` (the default) keeps plain
    /// decoding.
    pub drafter: Option<DrafterSource>,
    /// Most tokens the drafter proposes per verification round, `1..=16`
    /// (default 5); ignored without a drafter. The engine drafts fewer when
    /// the drafter's confidence makes another draft not worth its
    /// verification cost, measured on the running device; a run with a fixed
    /// `seed` and a temperature above 0 always drafts exactly `draft_tokens`.
    pub draft_tokens: usize,
    /// Optional download progress observer.
    pub on_download_progress: Option<DownloadProgressCallback>,
    /// Where a mixture-of-experts model's experts live:
    /// [`ExpertPlacement::Resident`] (the default) keeps every expert in
    /// memory; [`ExpertPlacement::Tiered`] streams them from the file
    /// through bounded caches ([`ExpertTiers`]). Tiered placement on a model
    /// without experts fails [`LocalModel::load`] with [`CucaError::Config`].
    pub experts: ExpertPlacement,
    /// How a checkpoint source is converted to GGUF; unused for GGUF sources.
    pub convert: ConvertConfig,
    /// How concurrent requests share the engine: see [`BatchingConfig`].
    pub batching: BatchingConfig,
    /// More devices to split the model over (tensor parallelism); empty (the
    /// default) runs it on [`device`](Self::device) alone. The model runs on
    /// `1 + shards.len()` devices, `device` being the first: every layer's
    /// attention heads and FFN width are split evenly, each device holds its
    /// slice of the weights and of the K/V pools, and the partial outputs are
    /// summed twice per layer (device collectives on CUDA, a host-staged sum
    /// in fixed order elsewhere, so a run is deterministic). One device may
    /// appear several times (`vulkan:0` as `device` and as the only shard
    /// splits the model in two on one GPU: no speedup, but the same code
    /// path). [`LocalModel::load`] fails with [`CucaError::Config`] unless
    /// `device` and every shard are GPU devices of the same backend and
    /// there is no `drafter` and `experts` is [`ExpertPlacement::Resident`];
    /// at load it also rejects a model that cannot be split: the head count
    /// and the FFN widths must divide by the device count (a model with
    /// fewer K/V heads than devices, such as Gemma 4 E2B with one, keeps a
    /// copy of each K/V head on every device), a delta-net (hybrid) layer or
    /// a full-width Q/K norm cannot be split. See [`LocalModelConfig::set_devices`].
    pub shards: Vec<LocalDevice>,
}

impl Default for LocalModelConfig {
    fn default() -> Self {
        Self {
            source: ModelSource::default_model(),
            cache_dir: None,
            hf_token: None,
            context_length: None,
            sampling: SamplingConfig::default(),
            device: LocalDevice::Cpu,
            drafter: None,
            draft_tokens: DEFAULT_DRAFT_TOKENS,
            on_download_progress: None,
            experts: ExpertPlacement::Resident,
            convert: ConvertConfig::default(),
            batching: BatchingConfig::default(),
            shards: Vec::new(),
        }
    }
}

impl LocalModelConfig {
    /// Set [`device`](Self::device) and [`shards`](Self::shards) from a
    /// `+`-separated list of devices (the syntax of `CUCA_LOCAL_DEVICE` in the
    /// tests and the example): `vulkan:0` is one device, `vulkan:0+vulkan:1`
    /// splits the model over two. The first device is `device`, the rest are
    /// the shards; each parses as a [`LocalDevice`].
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] for an empty list or an entry that is not a
    /// device; the configuration is left unchanged.
    pub fn set_devices(&mut self, list: &str) -> Result<(), CucaError> {
        let mut devices = list.split('+').map(|d| d.trim().parse::<LocalDevice>());
        let first = devices
            .next()
            .ok_or_else(|| CucaError::Config("empty device list".into()))??;
        let rest = devices.collect::<Result<Vec<_>, _>>()?;
        self.device = first;
        self.shards = rest;
        Ok(())
    }
}

impl fmt::Debug for LocalModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalModelConfig")
            .field("source", &self.source)
            .field("cache_dir", &self.cache_dir)
            .field("hf_token", &self.hf_token.as_ref().map(|_| "<redacted>"))
            .field("context_length", &self.context_length)
            .field("sampling", &self.sampling)
            .field("device", &self.device)
            .field("drafter", &self.drafter)
            .field("draft_tokens", &self.draft_tokens)
            .field("on_download_progress", &self.on_download_progress)
            .field("experts", &self.experts)
            .field("convert", &self.convert)
            .field("batching", &self.batching)
            .field("shards", &self.shards)
            .finish()
    }
}

/// What a loaded engine hands back to `load`: the pieces the request path
/// needs without touching the engine thread.
type Ready = Result<
    (
        Arc<Tokenizer>,
        ChatTemplate,
        ChatFormat,
        usize,
        u64,
        Option<Arc<model::tier::TierShared>>,
        Arc<engine::ServingCounters>,
    ),
    CucaError,
>;

/// State shared by every clone of a [`LocalModel`].
pub(crate) struct LocalModelInner {
    /// Queue into the engine thread; dropping the last sender ends it.
    jobs: mpsc::Sender<engine::Job>,
    tokenizer: Arc<Tokenizer>,
    template: ChatTemplate,
    format: ChatFormat,
    context_length: usize,
    kv_cache_bytes: u64,
    draft_stats: Arc<engine::DraftCounters>,
    serving: Arc<engine::ServingCounters>,
    /// Counters and heat of the expert tiers; `None` unless
    /// [`ExpertPlacement::Tiered`].
    tiers: Option<Arc<model::tier::TierShared>>,
}

/// A loaded in-process model, cheap to clone and share.
///
/// Load it with [`LocalModel::load`], attach it with
/// `CucaClientBuilder::with_local_model`, and select
/// `ProviderEndpoint::Local`. The model runs in this process, on the CPU or
/// on the GPU [`LocalModelConfig::device`] selects: no HTTP endpoint is
/// involved. The engine thread ends once the last clone is dropped.
///
/// # Concurrency and memory
///
/// - The engine is a step scheduler ([`BatchingConfig`]): requests are
///   admitted in FIFO order while fewer than `max_sequences` (4 by default)
///   run and the pool can hold their prompts, and every step runs the decode
///   row of each running sequence (and a chunk of each prompt still being
///   prefilled) in one forward pass: the weights are read once per step, not
///   once per request, so aggregate throughput rises with the number of
///   running requests. At most 16 requests queue behind the running ones.
///   Past that, `generate_stream` waits for a free queue slot (backpressure,
///   not an error). A running request streams through a bounded 64-block
///   channel; a consumer that does not keep up stalls its own request only,
///   and dropping the stream stops it and keeps its work in the prefix store.
///   [`LocalModel::serving_stats`] reports the running and waiting counts,
///   free and cached blocks, prefix hits, preemptions and the rows per step.
/// - The K/V cache is paged and preallocated: blocks of 16 positions in
///   pools, `2 * 4 * 16 * n_kv_head * head_dim` bytes per block and layer that
///   owns K/V (each layer's head count and head dimension come from the file;
///   [`LocalModel::kv_cache_bytes`] reports the total). The pool is shared by
///   all running sequences: a full-attention group holds
///   [`BatchingConfig::kv_tokens`] positions (twice `context_length` by
///   default, however many sequences run), a sliding-window group its
///   window plus about 270 positions per running sequence plus cached
///   prefixes. At the default 8192 that is 422 MB for Gemma 4 E2B, 1.27 GB for
///   E4B and 3.76 GB for Qwen3-0.6B (28 full-attention layers, 8 KV heads of
///   128). Blocks are named by the token ids they hold, so a finished
///   request's blocks stay cached (least recently used evicted first) and a
///   prompt that starts with the same ids only prefills what follows its
///   last whole cached block, reported as the response's `prompt_cache_usage`
///   (`read_tokens` is a multiple of 16: the trailing partial block, at most
///   15 tokens, and the last prompt token are always recomputed).
/// - Weights stay quantized (a mixture-of-experts model keeps every expert,
///   split per expert, so memory is the file size; a pass computes only the
///   routed experts; [`ExpertPlacement::Tiered`] bounds the expert memory to
///   configured RAM and device budgets instead). Gemma 4's per-layer embedding table
///   (`per_layer_token_embd`: 1.32 GB in the E2B Q4_0 file, 1.59 GB in E4B
///   Q4_0) is not loaded: one row
///   per input token (5,040 bytes in E2B Q4_0) is read from the GGUF file,
///   usually from the OS page cache. Resident weights are therefore about
///   the file size minus that table (E2B Q4_0: 1.5 GB of 2.84 GB).
/// - On a GPU device the weights and the K/V cache live in device memory.
///   Q4_0 and Q8_0 weights stay quantized there; F32, F16 and BF16 weights
///   (including `per_layer_model_proj`, 92 MB as f32 for E2B) are f32; every
///   other dtype (Q4_1, Q5_x, the K-quants) is requantized to Q8_0 at load
///   (its error is far below the K-quant error), and a weight over 1 GiB
///   is split into row chunks. Embedding rows are still read from the file
///   on the host.
///
/// # Behavior
///
/// - Thinking is binary: a request with thinking enabled opens the model's
///   reasoning channel, and its effort level is ignored.
/// - The engine is text-only: image blocks are rejected.
/// - `UnifiedRequest::model` is not used for routing, since one client holds
///   one model. `temperature`, `max_tokens`, `top_p`, `top_k`, `seed` and
///   `stop` apply per request and override the load-time [`SamplingConfig`]
///   (a request `seed` restarts the generator for that request only);
///   without one the [`SamplingConfig`] value holds. Generation ends at the
///   first occurrence of a stop string (the stop text is not emitted), at
///   `max_tokens`, at the end-of-generation token, or at the context limit.
/// - Token usage in the response is exact, not estimated.
///
/// # Build
///
/// candle selects its quantized SIMD kernels (Q4_0, Q8_0, ...) at compile
/// time, so build for the host CPU, e.g. `RUSTFLAGS="-C target-cpu=native"
/// cargo build --release --features provider-local`; generic builds are
/// several times slower. A GPU device needs its backend feature. Every GPU
/// feature compiles C through CubeCL's dependencies (as `provider-local`
/// already does for candle) and, on Windows, runs the SDK resource compiler;
/// `provider-local-cuda` and `provider-local-rocm` also download and link a
/// prebuilt LLVM on the first build (ROCm on Linux only). The first forward
/// pass of each new input shape on a GPU is slow while CubeCL autotunes and
/// compiles its kernels.
#[derive(Clone)]
pub struct LocalModel {
    inner: Arc<LocalModelInner>,
}

impl fmt::Debug for LocalModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalModel")
            .field("context_length", &self.inner.context_length)
            .finish_non_exhaustive()
    }
}

/// What a loaded model supports, derived from its chat template and its
/// output conventions; see [`LocalModel::capabilities`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct LocalCapabilities {
    /// The model writes a reasoning channel (`<think>…</think>`, Gemma 4's
    /// `<|channel>thought`), which the stream reports as `Thinking` blocks.
    pub reasoning: bool,
    /// A request can switch the reasoning off: the chat template reads
    /// `enable_thinking`. `false` for a model that always reasons (or never
    /// does).
    pub can_disable_thinking: bool,
    /// The model has a tool-call syntax the output parser understands, so
    /// tool calls come back as `ToolUse` blocks; `false` for plain-text
    /// models.
    pub tools: bool,
}

/// Speculative-decoding counters of a [`LocalModel`] with a drafter, summed
/// over every request so far; see [`LocalModel::draft_stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DraftStats {
    /// Verification rounds that carried drafted tokens.
    pub rounds: u64,
    /// Tokens the drafter proposed.
    pub drafted: u64,
    /// Drafted tokens the target model accepted.
    pub accepted: u64,
}

impl DraftStats {
    /// Fraction of drafted tokens that were accepted, `None` before the
    /// first round.
    pub fn acceptance_rate(&self) -> Option<f64> {
        (self.drafted > 0).then(|| self.accepted as f64 / self.drafted as f64)
    }
}

/// A snapshot of the engine's scheduler and K/V pools, see
/// [`LocalModel::serving_stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ServingStats {
    /// Requests running now (prefilling or decoding).
    pub running: usize,
    /// Requests admitted to the engine but waiting for a slot or for K/V
    /// blocks (including a request preempted and put back in the queue).
    pub waiting: usize,
    /// Free blocks in the full-attention pool (or the first pool of a model
    /// without one).
    pub free_blocks: usize,
    /// Unreferenced blocks the prefix store keeps in that pool.
    pub cached_blocks: usize,
    /// Bytes of recurrent-state snapshots the prefix store keeps (hybrid
    /// models; zero for the others).
    pub snapshot_bytes: u64,
    /// Prompt tokens served from the prefix store since load.
    pub prefix_hit_tokens: u64,
    /// Sequences preempted (their blocks released, the request put back in
    /// the queue) since load.
    pub preemptions: u64,
    /// Forward steps the scheduler ran since load.
    pub steps: u64,
    /// Rows (tokens) those steps processed since load; `batched_rows > steps`
    /// means steps carried several rows.
    pub batched_rows: u64,
}

/// How the engine serves concurrent requests: a step scheduler that runs the
/// decode rows of several sequences, and chunks of prompts being prefilled,
/// in one forward pass over one shared K/V block pool (see the local-provider
/// module docs). `FromStr` parses `seqs=4,kv=16384,step=256,prefix=off` (any
/// subset; `kv` and `step` in tokens, `prefix` is `on`/`off`), which the
/// tests and the example read from `CUCA_LOCAL_BATCHING`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchingConfig {
    /// Most sequences running at once, `1..=64` (default 4). Each running
    /// sequence of a hybrid model (Qwen 3.5) also holds its own recurrent
    /// state (20 MB for the 0.8B model).
    pub max_sequences: usize,
    /// Positions in the shared K/V block pool: the pool of every
    /// full-attention layer group holds this many positions for all running
    /// sequences and the cached prefixes together (a sliding-window group
    /// `max_sequences` windows plus cached prefixes). `None` (the default)
    /// is twice `context_length`; an explicit value is at least
    /// `context_length + 16` and rounds up to whole blocks of 16. It is the
    /// K/V memory knob: [`LocalModel::kv_cache_bytes`] reports the result, and
    /// raising `max_sequences` does not raise it.
    pub kv_tokens: Option<usize>,
    /// Most rows (tokens) of one forward step across all sequences, at least
    /// `max_sequences` and at most 2048. `None` (the default) is the prefill
    /// chunk of the context length (256 at 8192 positions).
    pub max_step_tokens: Option<usize>,
    /// Keep the K/V of finished requests, and of prefix blocks leaving a
    /// sliding window, in the prefix store for later prompts (the default).
    /// With `false` nothing is registered, so every prompt is prefilled in
    /// full.
    pub prefix_cache: bool,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            max_sequences: 4,
            kv_tokens: None,
            max_step_tokens: None,
            prefix_cache: true,
        }
    }
}

impl std::str::FromStr for BatchingConfig {
    type Err = CucaError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bad = || {
            CucaError::Config(format!(
                "unknown batching `{s}`; expected a comma-separated subset of seqs=<n>,kv=<tokens>,step=<tokens>,prefix=on|off"
            ))
        };
        let mut config = Self::default();
        let lower = s.trim().to_ascii_lowercase();
        if lower.is_empty() {
            return Ok(config);
        }
        let mut seen = Vec::new();
        for part in lower.split(',') {
            let (key, value) = part.split_once('=').ok_or_else(bad)?;
            if seen.contains(&key) {
                return Err(bad());
            }
            seen.push(key);
            let number = || value.parse::<usize>().map_err(|_| bad());
            match key {
                "seqs" => config.max_sequences = number()?,
                "kv" => config.kv_tokens = Some(number()?),
                "step" => config.max_step_tokens = Some(number()?),
                "prefix" => {
                    config.prefix_cache = match value {
                        "on" | "true" | "1" => true,
                        "off" | "false" | "0" => false,
                        _ => return Err(bad()),
                    }
                }
                _ => return Err(bad()),
            }
        }
        Ok(config)
    }
}

impl LocalModel {
    /// Bytes the K/V block pools occupy, in RAM or device memory: blocks of 16
    /// positions, `2 * 4 * 16 * n_kv_head * head_dim` bytes per block and layer
    /// that owns K/V. A full-attention group of layers holds
    /// [`BatchingConfig::kv_tokens`] positions, shared by every running
    /// sequence (twice `context_length` by default, whatever `max_sequences`
    /// is); a sliding-window group `ceil((window - 1 + rows) / 16) + 2` blocks
    /// per running sequence (rows: one call's, at most 256) plus up to
    /// `kv_tokens - context_length` positions of cached prefixes (see the
    /// local-provider module docs).
    ///
    /// A hybrid model (Qwen 3.5) owns K/V only in its attention layers, and
    /// besides keeps a recurrent state of fixed size per running sequence
    /// (about 20 MB for the 0.8B model, two copies while speculating) plus
    /// one per cached snapshot, none of it counted here.
    pub fn kv_cache_bytes(&self) -> u64 {
        self.inner.kv_cache_bytes
    }

    /// What this model supports: reasoning, switching it off, tool calls.
    pub fn capabilities(&self) -> LocalCapabilities {
        let reasoning = self.inner.format.writes_reasoning();
        LocalCapabilities {
            reasoning,
            can_disable_thinking: reasoning && self.inner.template.reads_enable_thinking(),
            tools: !matches!(
                self.inner.format.tool_syntax(),
                self::chat::ToolSyntax::Plain
            ),
        }
    }

    /// Speculative-decoding counters (all zero without a drafter).
    pub fn draft_stats(&self) -> DraftStats {
        self.inner.draft_stats.snapshot()
    }

    /// The scheduler and K/V pool counters: requests running and waiting,
    /// free and cached blocks, prefix-store hits, preemptions, and how many
    /// rows the forward steps carried (see [`ServingStats`]).
    pub fn serving_stats(&self) -> ServingStats {
        self.inner.serving.snapshot()
    }

    /// Expert tier counters (hits per tier, file reads, resident bytes);
    /// `None` unless the model loaded with [`ExpertPlacement::Tiered`].
    pub fn expert_tier_stats(&self) -> Option<ExpertTierStats> {
        self.inner.tiers.as_ref().map(|t| t.stats())
    }

    /// The expert heat learned so far, for [`ExpertTiers::warm_from`] in a
    /// later session (it serializes with serde); `None` unless the model
    /// loaded with [`ExpertPlacement::Tiered`].
    pub fn expert_heat(&self) -> Option<ExpertHeat> {
        self.inner.tiers.as_ref().map(|t| t.heat())
    }

    /// Load the model described by `config`: resolve or download the file,
    /// read the weights, tokenizer and chat template, and start the engine
    /// thread.
    ///
    /// The heavy work (download, hashing, parsing gigabytes of weights) runs on
    /// the engine thread; this future only awaits its result, so it never
    /// blocks the async runtime.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] for an invalid configuration (zero or oversized
    /// `context_length`, non-finite `temperature`, `top_p` outside `(0, 1]`,
    /// a [`LocalDevice`] whose cargo feature is not enabled, an
    /// [`ExpertPlacement::Tiered`] whose budgets do not fit the device or the
    /// model, or that is set on a model without experts);
    /// [`CucaError::Io`] when the file is missing or unreadable or the thread
    /// cannot start; [`CucaError::Transport`] and [`CucaError::Http`] for
    /// download failures; [`CucaError::Provider`] for an unsupported or
    /// malformed GGUF file (an architecture outside the supported list, a
    /// tensor dtype candle cannot read, a missing tensor or metadata key, or
    /// a drafter that does not match its target, or `DrafterSource::InTrunk`
    /// on a model whose file has no prediction head), a failed checksum, or a
    /// GPU device that fails to initialize (no such adapter, missing driver).
    pub async fn load(config: LocalModelConfig) -> Result<Self, CucaError> {
        validate(&config)?;
        let stats = Arc::new(engine::DraftCounters::default());
        let shared = Arc::clone(&stats);
        Self::start(stats, move || {
            let path = resolve_model_path(&config, &config.source)?;
            let drafter = match &config.drafter {
                None => None,
                Some(DrafterSource::File(source)) => Some(engine::ResolvedDrafter::File(
                    resolve_model_path(&config, source)?,
                )),
                Some(DrafterSource::InTrunk) => Some(engine::ResolvedDrafter::InTrunk),
            };
            let (engine, tokenizer, template, format) =
                engine::Engine::open(&path, drafter.as_ref(), shared, &config)?;
            Ok((engine, tokenizer, template, format))
        })
        .await
    }

    /// Start the engine thread, build its engine with `open` there, and wait
    /// for it to report ready.
    async fn start<F>(draft_stats: Arc<engine::DraftCounters>, open: F) -> Result<Self, CucaError>
    where
        F: FnOnce()
                -> Result<(engine::Engine, Arc<Tokenizer>, ChatTemplate, ChatFormat), CucaError>
            + Send
            + 'static,
    {
        let (jobs_tx, mut jobs_rx) = mpsc::channel::<engine::Job>(JOB_QUEUE);
        let (ready_tx, ready_rx) = oneshot::channel::<Ready>();
        std::thread::Builder::new()
            .name("cuca-local-engine".to_string())
            .spawn(move || {
                // GPU backends report a missing adapter or driver by
                // panicking during device setup: catch it so the caller gets
                // the message instead of a dead thread.
                let opened = match catch_unwind(AssertUnwindSafe(open)) {
                    Ok(opened) => opened,
                    Err(payload) => {
                        let msg = payload
                            .downcast_ref::<&str>()
                            .copied()
                            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                            .unwrap_or("panic");
                        Err(CucaError::provider(
                            ProviderEndpoint::Local,
                            format!("local engine failed to initialize: {msg}"),
                        ))
                    }
                };
                let mut engine = match opened {
                    Ok((engine, tokenizer, template, format)) => {
                        if ready_tx
                            .send(Ok((
                                tokenizer,
                                template,
                                format,
                                engine.context_length(),
                                engine.kv_cache_bytes(),
                                engine.expert_tiers(),
                                engine.serving_counters(),
                            )))
                            .is_err()
                        {
                            return;
                        }
                        engine
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                engine.run_loop(&mut jobs_rx);
            })
            .map_err(|e| CucaError::Io {
                message: format!("failed to start the local engine thread: {e}"),
            })?;

        match ready_rx.await {
            Ok(Ok((
                tokenizer,
                template,
                format,
                context_length,
                kv_cache_bytes,
                tiers,
                serving,
            ))) => Ok(LocalModel {
                inner: Arc::new(LocalModelInner {
                    jobs: jobs_tx,
                    tokenizer,
                    template,
                    format,
                    context_length,
                    kv_cache_bytes,
                    draft_stats,
                    serving,
                    tiers,
                }),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(CucaError::provider(
                ProviderEndpoint::Local,
                "local engine thread exited during load",
            )),
        }
    }
}

/// Reject configurations the engine cannot honor, before any thread starts.
fn validate(config: &LocalModelConfig) -> Result<(), CucaError> {
    if config.context_length == Some(0) {
        return Err(CucaError::Config(
            "context_length must be at least 1".into(),
        ));
    }
    if !config.shards.is_empty() {
        let devices = std::iter::once(config.device).chain(config.shards.iter().copied());
        if devices.clone().any(|d| d == LocalDevice::Cpu) {
            return Err(CucaError::Config(
                "tensor parallelism: `device` and every shard must be a GPU device, got cpu".into(),
            ));
        }
        for device in devices {
            if let Some(e) = device_unavailable(device) {
                return Err(e);
            }
            if std::mem::discriminant(&device) != std::mem::discriminant(&config.device) {
                return Err(CucaError::Config(format!(
                    "tensor parallelism: every shard must use the same GPU backend as `device` ({:?}), got {device:?}",
                    config.device
                )));
            }
        }
        if config.drafter.is_some() {
            return Err(CucaError::Config(
                "tensor parallelism: a drafter is not supported with `shards`".into(),
            ));
        }
        if config.experts != ExpertPlacement::Resident {
            return Err(CucaError::Config(
                "tensor parallelism: tiered experts are not supported with `shards`".into(),
            ));
        }
    }
    let batching = &config.batching;
    if !(1..=64).contains(&batching.max_sequences) {
        return Err(CucaError::Config(format!(
            "batching.max_sequences must be in 1..=64, got {}",
            batching.max_sequences
        )));
    }
    if let Some(step) = batching.max_step_tokens
        && !(batching.max_sequences..=2048).contains(&step)
    {
        return Err(CucaError::Config(format!(
            "batching.max_step_tokens must be in max_sequences..=2048 ({}..=2048), got {step}",
            batching.max_sequences
        )));
    }
    let temperature = config.sampling.temperature;
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(CucaError::Config(format!(
            "sampling.temperature must be a finite value >= 0, got {temperature}"
        )));
    }
    let top_p = config.sampling.top_p;
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(CucaError::Config(format!(
            "sampling.top_p must be in (0, 1], got {top_p}"
        )));
    }
    if let Some(e) = device_unavailable(config.device) {
        return Err(e);
    }
    if config.drafter == Some(DrafterSource::InTrunk) && config.device != LocalDevice::Cpu {
        return Err(CucaError::Config(format!(
            "DrafterSource::InTrunk runs on the CPU only, not on {:?}",
            config.device
        )));
    }
    if config.drafter.is_some() && !(1..=MAX_DRAFT_TOKENS).contains(&config.draft_tokens) {
        return Err(CucaError::Config(format!(
            "draft_tokens must be in 1..={MAX_DRAFT_TOKENS}, got {}",
            config.draft_tokens
        )));
    }
    if let ExpertPlacement::Tiered(t) = &config.experts {
        if !(1..=MAX_LOADER_THREADS).contains(&t.loader_threads) {
            return Err(CucaError::Config(format!(
                "experts.loader_threads must be in 1..={MAX_LOADER_THREADS}, got {}",
                t.loader_threads
            )));
        }
        if config.device == LocalDevice::Cpu {
            if t.device_bytes != 0 {
                return Err(CucaError::Config(
                    "experts.device_bytes is for GPU devices; set it to 0 on LocalDevice::Cpu"
                        .into(),
                ));
            }
            if t.ram_bytes == 0 {
                return Err(CucaError::Config(
                    "experts.ram_bytes must be greater than 0 on LocalDevice::Cpu".into(),
                ));
            }
        } else if t.device_bytes == 0 {
            return Err(CucaError::Config(format!(
                "experts.device_bytes must be greater than 0 on {:?}",
                config.device
            )));
        }
    }
    Ok(())
}

/// The GGUF path for `config.source`, downloading (and, for a checkpoint,
/// converting) it first when needed.
///
/// Runs on the engine thread. The download uses a private current-thread
/// runtime and HTTP client (never the `CucaClient` one), both dropped before
/// returning: the engine loop later blocks on a channel, which must not happen
/// inside a runtime. A checkpoint ([`ModelSource::Checkpoint`],
/// [`ModelSource::Dir`]) is converted to
/// `<cache>/converted/.../model-<quant>.gguf` once; an existing result is
/// reused without touching the network or the shards.
fn resolve_model_path(
    config: &LocalModelConfig,
    source: &ModelSource,
) -> Result<PathBuf, CucaError> {
    let cache_dir =
        || download::resolve_cache_dir(config.cache_dir.as_deref(), &|key| std::env::var_os(key));
    match source {
        ModelSource::File(path) => {
            if !path.is_file() {
                return Err(CucaError::Io {
                    message: format!("model file `{}` does not exist", path.display()),
                });
            }
            let is_part = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(convert::gguf_merge::SplitName::parse)
                .is_some();
            if !is_part {
                return Ok(path.clone());
            }
            let cache_dir = cache_dir()?;
            let Some(plan) = convert::gguf_merge::plan_local(&cache_dir, path)? else {
                return Ok(path.clone());
            };
            convert::gguf_merge::merge(
                &plan.parts,
                &plan.merged,
                config.on_download_progress.as_ref(),
            )?;
            Ok(plan.merged)
        }
        ModelSource::HuggingFace {
            repo,
            revision,
            file,
            sha256,
            parts,
        } => {
            let cache_dir = cache_dir()?;
            let split = convert::gguf_merge::repo_parts(file);
            if split.is_none() && !parts.is_empty() {
                return Err(CucaError::Config(format!(
                    "`{file}` is not named like a split GGUF part (`<base>-00001-of-0000N.gguf`) but the source lists {} parts",
                    parts.len()
                )));
            }
            let runtime = || -> Result<_, CucaError> {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                let client = reqwest::Client::builder()
                    .connect_timeout(CONNECT_TIMEOUT)
                    .build()?;
                Ok((runtime, client))
            };
            if let Some((names, merged_file)) = split {
                let wanted: Vec<(String, Option<String>)> = if parts.is_empty() {
                    names
                        .iter()
                        .enumerate()
                        .map(|(i, n)| (n.clone(), if i == 0 { sha256.clone() } else { None }))
                        .collect()
                } else {
                    let listed: Vec<&str> = parts.iter().map(|p| p.file.as_str()).collect();
                    if listed != names.iter().map(String::as_str).collect::<Vec<_>>() {
                        return Err(CucaError::Config(format!(
                            "the parts of `{file}` must be {names:?} in order, got {listed:?}"
                        )));
                    }
                    parts
                        .iter()
                        .map(|p| (p.file.clone(), p.sha256.clone()))
                        .collect()
                };
                let merged = download::cached_path(&cache_dir, repo, revision, &merged_file)?;
                if !merged.exists() {
                    let (runtime, client) = runtime()?;
                    runtime.block_on(download::ensure_split_downloaded(
                        &client,
                        download::HF_ENDPOINT,
                        &cache_dir,
                        repo,
                        revision,
                        &wanted,
                        &merged,
                        hf_token(config).as_deref(),
                        config.on_download_progress.as_ref(),
                    ))?;
                }
                return Ok(merged);
            }
            let final_path = download::cached_path(&cache_dir, repo, revision, file)?;
            if !final_path.exists() {
                let (runtime, client) = runtime()?;
                runtime.block_on(download::ensure_downloaded(
                    &client,
                    download::HF_ENDPOINT,
                    repo,
                    revision,
                    file,
                    sha256.as_deref(),
                    &final_path,
                    hf_token(config).as_deref(),
                    config.on_download_progress.as_ref(),
                ))?;
            }
            Ok(final_path)
        }
        ModelSource::Checkpoint { repo, revision } => {
            let cache_dir = cache_dir()?;
            let quant = config.convert.quant;
            let out = convert::output_path(&cache_dir, source, quant)?;
            if !out.exists() {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                let client = reqwest::Client::builder()
                    .connect_timeout(CONNECT_TIMEOUT)
                    .build()?;
                let checkpoint = runtime.block_on(convert::hf::fetch(
                    &client,
                    download::HF_ENDPOINT,
                    repo,
                    revision,
                    &cache_dir,
                    hf_token(config).as_deref(),
                    config.on_download_progress.as_ref(),
                ))?;
                drop(runtime);
                convert::convert(
                    &checkpoint,
                    &out,
                    quant,
                    config.on_download_progress.as_ref(),
                )?;
            }
            Ok(out)
        }
        ModelSource::Dir(dir) => {
            if !dir.is_dir() {
                return Err(CucaError::Io {
                    message: format!("checkpoint directory `{}` does not exist", dir.display()),
                });
            }
            let quant = config.convert.quant;
            let out = convert::output_path(&cache_dir()?, source, quant)?;
            if !out.exists() {
                let checkpoint = convert::hf::CheckpointDir::open(dir)?;
                convert::convert(
                    &checkpoint,
                    &out,
                    quant,
                    config.on_download_progress.as_ref(),
                )?;
            }
            Ok(out)
        }
    }
}

/// The Hugging Face token: the configured one, else `HF_TOKEN`.
fn hf_token(config: &LocalModelConfig) -> Option<String> {
    config.hf_token.clone().or_else(|| {
        std::env::var("HF_TOKEN")
            .ok()
            .filter(|token| !token.is_empty())
    })
}

/// The block stream of one local request: the receiving end of the channel the
/// engine thread writes to. Dropping it tells the engine to stop generating.
struct LocalStream {
    rx: mpsc::Receiver<Result<MessageContentBlock, CucaError>>,
}

impl Stream for LocalStream {
    type Item = Result<MessageContentBlock, CucaError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl CucaClient {
    /// Dispatch a unified request to the in-process engine.
    ///
    /// Renders the chat template, tokenizes the prompt, and queues one job for
    /// the engine thread; the returned stream yields the job's blocks. The
    /// response metadata carries the prefix-reuse counters (as prompt-cache
    /// usage) and the exact token usage once generation ends.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when no model is attached or the request's
    /// `temperature` or `top_p` is invalid; [`CucaError::Provider`] when the
    /// request cannot be rendered (for example an image block), the prompt
    /// does not fit the context, or the engine has shut down.
    pub(crate) async fn dispatch_local(
        &self,
        req: UnifiedRequest,
    ) -> Result<ProviderDispatch, CucaError> {
        let model = self
            .local_model()
            .ok_or_else(|| CucaError::Config(MODEL_REQUIRED.into()))?;
        let inner = &model.inner;
        if let Some(t) = req.temperature
            && (!t.is_finite() || t < 0.0)
        {
            return Err(CucaError::Config(format!(
                "temperature must be a finite value >= 0, got {t}"
            )));
        }
        if let Some(p) = req.top_p
            && (!p.is_finite() || p <= 0.0 || p > 1.0)
        {
            return Err(CucaError::Config(format!(
                "top_p must be a finite value in (0, 1], got {p}"
            )));
        }

        let prompt = chat_prompt(&req)?;
        let rendered = inner.template.render(&prompt)?;
        let prompt_ids = inner.tokenizer.encode_prompt(&rendered);
        if prompt_ids.len() >= inner.context_length {
            return Err(CucaError::provider(
                ProviderEndpoint::Local,
                format!(
                    "prompt has {} tokens; context_length is {}",
                    prompt_ids.len(),
                    inner.context_length
                ),
            ));
        }

        let (events, rx) = mpsc::channel(EVENT_BUFFER);
        let metadata = ResponseMetadataHandle::empty();
        let job = engine::Job {
            prompt_ids,
            start_in_thinking: inner.format.starts_in_thinking(&rendered),
            enable_thinking: prompt.enable_thinking,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            top_p: req.top_p,
            top_k: req.top_k,
            seed: req.seed,
            stop: req.stop,
            events,
            metadata: metadata.clone(),
        };
        inner.jobs.send(job).await.map_err(|_| {
            CucaError::provider(ProviderEndpoint::Local, "local engine has shut down")
        })?;
        Ok(ProviderDispatch {
            stream: Box::pin(LocalStream { rx }),
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use tokio_stream::StreamExt;

    use super::*;
    use crate::provider::local::engine::testing::{engine, tiny_format, tiny_tokenizer};
    use crate::provider::local::model::tiny::{TinyModel, TinySpec, write_spec, write_tiny_model};

    /// A template that renders the first message's content after `<bos>`.
    const TEMPLATE: &str = "{{ bos_token }}{{ messages[0]['content'] }}";

    async fn tiny_model(model: &TinyModel, context_length: usize) -> LocalModel {
        let path = model.path.clone();
        LocalModel::start(Arc::default(), move || {
            let tokenizer = Arc::new(tiny_tokenizer());
            let template = ChatTemplate::new(
                TEMPLATE.to_string(),
                "<bos>".to_string(),
                "<eos>".to_string(),
            )?;
            // The engine must be built on the engine thread, from the path.
            let tiny = TinyModel { path };
            let engine = engine(&tiny, context_length, 0.0, None);
            // `tiny` would delete the file on drop; the caller still owns it.
            std::mem::forget(tiny);
            let format = tiny_format(&tokenizer);
            Ok((engine, tokenizer, template, format))
        })
        .await
        .expect("tiny engine starts")
    }

    fn local_client(model: LocalModel) -> CucaClient {
        CucaClient::builder()
            .with_provider(ProviderEndpoint::Local)
            .with_local_model(model)
            .build()
            .expect("client builds")
    }

    #[test]
    fn defaults_match_the_documented_values() {
        let config = LocalModelConfig::default();
        assert_eq!(config.context_length, None);
        assert_eq!(
            resolve_context_length(None, 131_072),
            DEFAULT_CONTEXT_LENGTH
        );
        assert_eq!(resolve_context_length(None, 4096), 4096);
        assert_eq!(resolve_context_length(None, 2048), 2048);
        assert_eq!(resolve_context_length(Some(100_000), 4096), 100_000);
        assert_eq!(config.sampling, SamplingConfig::default());
        assert_eq!(config.sampling.top_k, 64);
        assert_eq!(config.source, ModelSource::default_model());
        let ModelSource::HuggingFace {
            repo,
            revision,
            file,
            sha256,
            ..
        } = ModelSource::default_model()
        else {
            panic!("default source is a Hugging Face file");
        };
        assert_eq!(repo, "ggml-org/gemma-4-E2B-it-GGUF");
        assert_eq!(revision.len(), 40);
        assert_eq!(file, "gemma-4-E2B-it-Q4_0.gguf");
        assert_eq!(sha256.map(|s| s.len()), Some(64));
        assert_eq!(config.experts, ExpertPlacement::Resident);
        assert_eq!(ExpertTiers::default().loader_threads, 4);
    }

    #[test]
    fn presets_are_pinned_named_and_parseable() {
        use std::str::FromStr;
        let mut names = std::collections::HashSet::new();
        for p in ModelSource::PRESETS {
            assert!(names.insert(p.name), "duplicate preset {}", p.name);
            assert_eq!(p.name, p.name.to_ascii_lowercase());
            assert_eq!(p.revision.len(), 40, "{}", p.name);
            assert_eq!(p.sha256.len(), 64, "{}", p.name);
            assert!(
                p.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                "{}",
                p.name
            );
            assert!(p.size_bytes > 100_000_000, "{}", p.name);
            assert!(p.file.ends_with(".gguf"), "{}", p.name);
            assert!(!p.architecture.is_empty(), "{}", p.name);
            let source = p.source();
            assert_eq!(ModelSource::from_str(p.name).unwrap(), source);
            assert_eq!(
                ModelSource::from_str(&p.name.to_ascii_uppercase()).unwrap(),
                source
            );
            assert_eq!(ModelSource::preset(p.name), Some(p));
            let ModelSource::HuggingFace { sha256, .. } = source else {
                panic!("presets are Hugging Face files");
            };
            assert_eq!(sha256.as_deref(), Some(p.sha256));
        }
        // The default stays Gemma 4 E2B Q4_0 and sits first in the table.
        assert_eq!(ModelSource::PRESETS[0].name, "gemma4-e2b-q4_0");
        assert_eq!(ModelSource::PRESETS[0].size_bytes, 2_841_481_184);
        // Exactly one drafter preset; every other preset is a main model.
        let drafters: Vec<_> = ModelSource::PRESETS.iter().filter(|p| p.drafter).collect();
        assert_eq!(drafters.len(), 1);
        assert_eq!(drafters[0].architecture, "gemma4-assistant");
        for name in ["qwen3-0.6b-q8_0", "phi4-mini-q4_0", "olmoe-1b-7b-q4_0"] {
            assert!(ModelSource::preset(name).is_some(), "{name}");
        }
    }

    #[test]
    fn a_file_naming_a_split_part_resolves_to_the_merged_gguf() {
        use super::convert::gguf_merge::fixtures::{load, split, temp_dir};

        let model = super::model::tiny::write_tiny_model();
        let dir = temp_dir("source");
        let parts = split(&model, &dir, 3, 3, |i| i % 3);
        let cache = dir.join("cache");
        let resolve = |path: &std::path::Path| {
            let config = LocalModelConfig {
                source: ModelSource::File(path.to_path_buf()),
                cache_dir: Some(cache.clone()),
                ..LocalModelConfig::default()
            };
            resolve_model_path(&config, &config.source)
        };
        // Any part names the set; the merged file is the same one.
        let merged = resolve(&parts[1]).unwrap();
        assert!(merged.starts_with(cache.join("merged")), "{merged:?}");
        assert_eq!(merged.file_name().unwrap(), "tiny.gguf");
        assert_eq!(load(&merged).1.len(), load(&model.path).1.len());
        assert_eq!(resolve(&parts[0]).unwrap(), merged);
        assert!(parts.iter().all(|p| p.exists()), "the user's parts stay");
        // A file that is not a split part loads as it is.
        assert_eq!(resolve(&model.path).unwrap(), model.path);
        // A missing sibling names the file.
        std::fs::remove_file(&parts[2]).unwrap();
        let err = resolve(&parts[0]).unwrap_err().to_string();
        assert!(err.contains("tiny-00003-of-00003.gguf"), "{err}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_split_preset_pins_every_part() {
        const PARTS: &[PresetPart] = &[
            PresetPart {
                file: "m-Q4_0-00001-of-00002.gguf",
                sha256: "a",
                size_bytes: 10,
            },
            PresetPart {
                file: "m-Q4_0-00002-of-00002.gguf",
                sha256: "b",
                size_bytes: 20,
            },
        ];
        let preset = ModelPreset {
            file: PARTS[0].file,
            sha256: PARTS[0].sha256,
            size_bytes: 30,
            parts: PARTS,
            ..ModelSource::PRESETS[0]
        };
        let ModelSource::HuggingFace {
            file,
            sha256,
            parts,
            ..
        } = preset.source()
        else {
            panic!("presets are Hugging Face files");
        };
        assert_eq!(file, "m-Q4_0-00001-of-00002.gguf");
        assert_eq!(sha256.as_deref(), Some("a"));
        assert_eq!(
            parts,
            [
                ModelPart {
                    file: "m-Q4_0-00001-of-00002.gguf".into(),
                    sha256: Some("a".into())
                },
                ModelPart {
                    file: "m-Q4_0-00002-of-00002.gguf".into(),
                    sha256: Some("b".into())
                }
            ]
        );
        // Only the split presets list parts, and every part is pinned: the
        // first is the preset's file and the sizes add up.
        for p in ModelSource::PRESETS.iter().filter(|p| !p.parts.is_empty()) {
            assert_eq!(p.name, "mistral-small-4-119b-q4_0");
            assert_eq!((p.parts[0].file, p.parts[0].sha256), (p.file, p.sha256));
            assert_eq!(
                p.parts.iter().map(|q| q.size_bytes).sum::<u64>(),
                p.size_bytes
            );
            assert!(p.parts.iter().all(|q| q.sha256.len() == 64));
            // `<base>-0000i-of-0000N.gguf`, in part order.
            let n = p.parts.len();
            for (i, part) in p.parts.iter().enumerate() {
                let tag = format!("-{:05}-of-{n:05}.gguf", i + 1);
                assert!(part.file.ends_with(&tag), "{}: {tag}", part.file);
            }
        }
        assert_eq!(
            ModelSource::PRESETS
                .iter()
                .filter(|p| !p.parts.is_empty())
                .count(),
            1
        );
    }

    #[test]
    fn split_sources_validate_their_parts_before_any_download() {
        let source = |file: &str, parts: &[&str]| ModelSource::HuggingFace {
            repo: "org/model".into(),
            revision: "rev".into(),
            file: file.into(),
            sha256: None,
            parts: parts
                .iter()
                .map(|f| ModelPart {
                    file: (*f).into(),
                    sha256: None,
                })
                .collect(),
        };
        let config = |source| LocalModelConfig {
            source,
            cache_dir: Some(std::env::temp_dir().join("cuca-unused-cache")),
            ..LocalModelConfig::default()
        };
        // Parts listed for a file that is not a split part.
        let c = config(source("model.gguf", &["model.gguf"]));
        let err = resolve_model_path(&c, &c.source).unwrap_err();
        assert!(
            matches!(&err, CucaError::Config(m) if m.contains("model.gguf")),
            "{err:?}"
        );
        // Parts that are not the set `file` names.
        let c = config(source(
            "m-00001-of-00002.gguf",
            &["m-00001-of-00002.gguf", "m-00002-of-00003.gguf"],
        ));
        let err = resolve_model_path(&c, &c.source).unwrap_err();
        assert!(
            matches!(&err, CucaError::Config(m) if m.contains("m-00002-of-00002.gguf")),
            "{err:?}"
        );
    }

    #[test]
    fn sources_parse_from_repo_revision_file() {
        let source: ModelSource = "unsloth/gemma-4-E2B-it-GGUF:0314792d:gemma-4-E2B-it-Q4_1.gguf"
            .parse()
            .unwrap();
        assert_eq!(
            source,
            ModelSource::hugging_face(
                "unsloth/gemma-4-E2B-it-GGUF",
                "0314792d",
                "gemma-4-E2B-it-Q4_1.gguf"
            )
        );
        for bad in ["", "gemma4-e9b-q4_0", "a:b", "a:b:c:d", "a::c", ":b:c"] {
            let err = bad.parse::<ModelSource>().unwrap_err();
            assert!(
                matches!(&err, CucaError::Config(m) if m.contains("gemma4-e4b-q8_0")),
                "{bad:?}: {err:?}"
            );
        }
    }

    #[test]
    fn debug_output_hides_the_token_and_names_the_callback() {
        let config = LocalModelConfig {
            hf_token: Some("hf_secret".into()),
            on_download_progress: Some(DownloadProgressCallback::new(|_| {})),
            ..Default::default()
        };
        let text = format!("{config:?}");
        assert!(!text.contains("hf_secret"), "{text}");
        assert!(text.contains("<redacted>"), "{text}");
        assert!(text.contains("DownloadProgressCallback(..)"), "{text}");
    }

    #[tokio::test]
    async fn load_rejects_invalid_configuration() {
        for (config, needle) in [
            (
                LocalModelConfig {
                    context_length: Some(0),
                    ..Default::default()
                },
                "context_length",
            ),
            (
                LocalModelConfig {
                    sampling: SamplingConfig {
                        temperature: f32::NAN,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                "temperature",
            ),
            (
                LocalModelConfig {
                    sampling: SamplingConfig {
                        temperature: -0.5,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                "temperature",
            ),
            (
                LocalModelConfig {
                    sampling: SamplingConfig {
                        top_p: 0.0,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                "top_p",
            ),
            (
                LocalModelConfig {
                    sampling: SamplingConfig {
                        top_p: 1.5,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                "top_p",
            ),
            (
                LocalModelConfig {
                    experts: ExpertPlacement::Tiered(ExpertTiers {
                        ram_bytes: 1 << 30,
                        loader_threads: 0,
                        ..ExpertTiers::default()
                    }),
                    ..Default::default()
                },
                "experts.loader_threads must be in 1..=64, got 0",
            ),
            (
                LocalModelConfig {
                    experts: ExpertPlacement::Tiered(ExpertTiers {
                        ram_bytes: 1 << 30,
                        device_bytes: 1 << 30,
                        ..ExpertTiers::default()
                    }),
                    ..Default::default()
                },
                "experts.device_bytes is for GPU devices",
            ),
            (
                LocalModelConfig {
                    experts: ExpertPlacement::Tiered(ExpertTiers::default()),
                    ..Default::default()
                },
                "experts.ram_bytes must be greater than 0",
            ),
        ] {
            let err = LocalModel::load(config).await.unwrap_err();
            assert!(
                matches!(&err, CucaError::Config(m) if m.contains(needle)),
                "{err}"
            );
        }
    }

    #[test]
    fn expert_placement_parses_every_form() {
        let tiers = |ram_bytes, device_bytes, loader_threads| {
            ExpertPlacement::Tiered(ExpertTiers {
                ram_bytes,
                device_bytes,
                loader_threads,
                warm_from: None,
            })
        };
        for (text, placement) in [
            ("resident", ExpertPlacement::Resident),
            (" Resident ", ExpertPlacement::Resident),
            ("tiered:ram=4g", tiers(4 << 30, 0, 4)),
            ("tiered:ram=1100M", tiers(1100 << 20, 0, 4)),
            ("tiered:ram=512k,threads=8", tiers(512 << 10, 0, 8)),
            ("tiered:ram=123", tiers(123, 0, 4)),
            ("TIERED:device=6g,ram=12g", tiers(12 << 30, 6 << 30, 4)),
            ("tiered:device=2g,ram=0", tiers(0, 2 << 30, 4)),
            ("tiered:device=1g", tiers(0, 1 << 30, 4)),
        ] {
            assert_eq!(
                text.parse::<ExpertPlacement>().unwrap(),
                placement,
                "{text}"
            );
        }
        for bad in [
            "",
            "tiered",
            "tiered:",
            "tiered:threads=2",
            "tiered:ram=4t",
            "tiered:ram=",
            "tiered:ram=-1",
            "tiered:ram=1g,ram=2g",
            "tiered:ram=1g,disk=2g",
            "tiered:ram=1g,threads=x",
            "tiered:ram=99999999999g",
            "resident:ram=1g",
        ] {
            let err = bad.parse::<ExpertPlacement>().unwrap_err();
            assert!(
                matches!(&err, CucaError::Config(m) if m.contains("unknown expert placement")),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn batching_parses_every_subset_and_validates_its_bounds() {
        let default = BatchingConfig::default();
        assert_eq!(
            (
                default.max_sequences,
                default.kv_tokens,
                default.max_step_tokens
            ),
            (4, None, None)
        );
        assert!(default.prefix_cache);
        for (text, seqs, kv, step, prefix) in [
            ("", 4, None, None, true),
            ("seqs=8", 8, None, None, true),
            (" SEQS=2, kv=16384 ", 2, Some(16384), None, true),
            (
                "seqs=4,kv=8192,step=128,prefix=off",
                4,
                Some(8192),
                Some(128),
                false,
            ),
            ("prefix=on,step=64", 4, None, Some(64), true),
            ("prefix=false", 4, None, None, false),
        ] {
            let parsed = text.replace(' ', "").parse::<BatchingConfig>().unwrap();
            assert_eq!(
                (
                    parsed.max_sequences,
                    parsed.kv_tokens,
                    parsed.max_step_tokens,
                    parsed.prefix_cache
                ),
                (seqs, kv, step, prefix),
                "{text}"
            );
        }
        for bad in [
            "seqs",
            "seqs=x",
            "seqs=-1",
            "seqs=2,seqs=3",
            "slots=2",
            "prefix=maybe",
            "kv=1k",
        ] {
            let err = bad.parse::<BatchingConfig>().unwrap_err();
            assert!(
                matches!(&err, CucaError::Config(m) if m.contains("unknown batching")),
                "{bad}: {err}"
            );
        }
        let with = |batching: BatchingConfig| LocalModelConfig {
            batching,
            ..LocalModelConfig::default()
        };
        let rejected = |text: &str| {
            let err = validate(&with(text.parse().unwrap())).unwrap_err();
            assert!(
                matches!(&err, CucaError::Config(m) if m.contains("batching")),
                "{text}: {err}"
            );
        };
        rejected("seqs=0");
        rejected("seqs=65");
        rejected("seqs=4,step=3");
        rejected("step=2049");
        for ok in ["seqs=1", "seqs=64,step=64", "step=2048", "kv=1"] {
            assert!(validate(&with(ok.parse().unwrap())).is_ok(), "{ok}");
        }
    }

    #[test]
    fn local_device_parses_every_form() {
        use GpuAdapter::{Default as Dflt, Discrete, Integrated};
        for (text, device) in [
            ("cpu", LocalDevice::Cpu),
            (" CPU ", LocalDevice::Cpu),
            ("vulkan", LocalDevice::Vulkan(Dflt)),
            ("Vulkan:1", LocalDevice::Vulkan(Discrete(1))),
            ("vulkan:integrated:0", LocalDevice::Vulkan(Integrated(0))),
            ("metal", LocalDevice::Metal(Dflt)),
            ("metal:2", LocalDevice::Metal(Discrete(2))),
            ("METAL:Integrated:3", LocalDevice::Metal(Integrated(3))),
            ("cuda", LocalDevice::Cuda(0)),
            ("cuda:1", LocalDevice::Cuda(1)),
            ("rocm", LocalDevice::Rocm(0)),
            ("rocm:4", LocalDevice::Rocm(4)),
        ] {
            assert_eq!(text.parse::<LocalDevice>().unwrap(), device, "{text}");
        }
        for bad in [
            "",
            "gpu",
            "cpu:0",
            "vulkan:x",
            "vulkan:integrated",
            "vulkan:discrete:0",
            "cuda:integrated:0",
            "rocm:-1",
            "metal:1:2",
        ] {
            let err = bad.parse::<LocalDevice>().unwrap_err();
            assert!(
                matches!(&err, CucaError::Config(m) if m.contains("unknown local device")),
                "{bad}: {err}"
            );
        }
    }

    #[cfg(not(feature = "provider-local-vulkan"))]
    #[tokio::test]
    async fn validate_rejects_a_compiled_out_device() {
        let config = LocalModelConfig {
            device: LocalDevice::Vulkan(GpuAdapter::Default),
            ..Default::default()
        };
        let err = LocalModel::load(config).await.unwrap_err();
        assert!(
            matches!(&err, CucaError::Config(m) if m.contains("`provider-local-vulkan`")),
            "{err}"
        );
    }

    /// The in-trunk head runs on the CPU only: a GPU device is a
    /// configuration error, naming the missing backend feature when it is
    /// compiled out and the CPU restriction otherwise; the CPU is accepted.
    #[tokio::test]
    async fn validate_rejects_the_in_trunk_head_on_a_gpu() {
        let config = LocalModelConfig {
            device: LocalDevice::Vulkan(GpuAdapter::Default),
            drafter: Some(DrafterSource::InTrunk),
            ..Default::default()
        };
        let err = LocalModel::load(config).await.unwrap_err();
        assert!(
            matches!(&err, CucaError::Config(m) if m.contains("`provider-local-vulkan`") || m.contains("CPU only")),
            "{err}"
        );
        let cpu = LocalModelConfig {
            drafter: Some(DrafterSource::InTrunk),
            ..Default::default()
        };
        assert!(validate(&cpu).is_ok());
    }

    #[test]
    fn devices_parse_from_a_plus_separated_list() {
        let mut config = LocalModelConfig::default();
        config.set_devices("vulkan:0+vulkan:1").unwrap();
        assert_eq!(config.device, LocalDevice::Vulkan(GpuAdapter::Discrete(0)));
        assert_eq!(
            config.shards,
            [LocalDevice::Vulkan(GpuAdapter::Discrete(1))]
        );
        config.set_devices(" cuda + cuda:1 + CUDA:1 ").unwrap();
        assert_eq!(
            (config.device, config.shards.clone()),
            (
                LocalDevice::Cuda(0),
                vec![LocalDevice::Cuda(1), LocalDevice::Cuda(1)]
            )
        );
        config.set_devices("cpu").unwrap();
        assert_eq!((config.device, config.shards.len()), (LocalDevice::Cpu, 0));
        // A bad entry leaves the configuration alone.
        config.set_devices("vulkan").unwrap();
        for bad in ["", "+", "vulkan+", "vulkan+nope", "gpu:0"] {
            let err = config.set_devices(bad).unwrap_err();
            assert!(matches!(err, CucaError::Config(_)), "{bad}: {err}");
            assert_eq!(
                config.device,
                LocalDevice::Vulkan(GpuAdapter::Default),
                "{bad}"
            );
            assert!(config.shards.is_empty(), "{bad}");
        }
    }

    #[test]
    fn validate_enforces_the_tensor_parallel_rules() {
        let config = |device, shards: &[LocalDevice]| LocalModelConfig {
            device,
            shards: shards.to_vec(),
            ..LocalModelConfig::default()
        };
        let vulkan = LocalDevice::Vulkan(GpuAdapter::Default);
        let rejects = |config: LocalModelConfig, what: &str| {
            let err = validate(&config).unwrap_err();
            assert!(
                matches!(&err, CucaError::Config(m) if m.contains(what)),
                "{what}: {err}"
            );
        };
        // No shards: the single-device rules only.
        assert!(validate(&config(LocalDevice::Cpu, &[])).is_ok());
        // The CPU cannot be a shard or a first device of a split model.
        rejects(config(vulkan, &[LocalDevice::Cpu]), "tensor parallelism");
        rejects(config(LocalDevice::Cpu, &[vulkan]), "tensor parallelism");
        // A backend that is compiled out names its feature.
        #[cfg(not(feature = "provider-local-vulkan"))]
        rejects(config(vulkan, &[vulkan]), "`provider-local-vulkan`");
        #[cfg(feature = "provider-local-vulkan")]
        {
            assert!(
                validate(&config(vulkan, &[vulkan])).is_ok(),
                "a device may repeat"
            );
            let two = config(
                vulkan,
                &[vulkan, LocalDevice::Vulkan(GpuAdapter::Discrete(1))],
            );
            assert!(validate(&two).is_ok());
            let mut drafted = config(vulkan, &[vulkan]);
            drafted.drafter = Some(DrafterSource::InTrunk);
            rejects(drafted, "drafter");
            let mut tiered = config(vulkan, &[vulkan]);
            tiered.experts = "tiered:device=1g".parse().unwrap();
            rejects(tiered, "tiered experts");
        }
        // Different backends do not mix (when both are compiled in).
        #[cfg(all(feature = "provider-local-vulkan", feature = "provider-local-metal"))]
        rejects(
            config(vulkan, &[LocalDevice::Metal(GpuAdapter::Default)]),
            "same GPU backend",
        );
    }

    #[test]
    fn a_model_source_converts_to_a_file_drafter() {
        let source = ModelSource::File("drafter.gguf".into());
        assert_eq!(
            DrafterSource::from(source.clone()),
            DrafterSource::File(source)
        );
    }

    /// A GPU backend that cannot find its adapter panics during setup; the
    /// caller must get that message, not an opaque "thread exited".
    #[tokio::test]
    async fn start_reports_a_panic_during_load() {
        let err = LocalModel::start(Arc::default(), || panic!("no Vulkan adapter found"))
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("local engine failed to initialize: no Vulkan adapter found"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn load_reports_a_missing_model_file() {
        let config = LocalModelConfig {
            source: ModelSource::File(PathBuf::from("definitely/not/here.gguf")),
            ..Default::default()
        };
        let err = LocalModel::load(config).await.unwrap_err();
        assert!(matches!(&err, CucaError::Io { message } if message.contains("does not exist")));
    }

    #[tokio::test]
    async fn load_reports_a_file_that_is_not_gguf() {
        let path = std::env::temp_dir().join(format!("cuca-not-gguf-{}.gguf", std::process::id()));
        std::fs::write(&path, b"this is not a GGUF file").unwrap();
        let config = LocalModelConfig {
            source: ModelSource::File(path.clone()),
            ..Default::default()
        };
        let err = LocalModel::load(config).await.unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(&err, CucaError::Provider { message, .. } if message.contains("not a supported GGUF file")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn dispatch_without_a_model_is_a_config_error() {
        // `build` refuses `Local` without a model, so build for another
        // provider and re-select `Local` is impossible; assert the build check.
        let err = CucaClient::builder()
            .with_provider(ProviderEndpoint::Local)
            .build()
            .err()
            .expect("Local without a model must not build");
        assert!(matches!(&err, CucaError::Config(m) if m == MODEL_REQUIRED));
    }

    /// The tiny vocabulary has a reasoning channel and the Gemma 4 tool
    /// syntax; its template never reads `enable_thinking`, so reasoning cannot
    /// be switched off.
    #[tokio::test]
    async fn capabilities_follow_the_format_and_the_template() {
        let file = write_tiny_model();
        let model = tiny_model(&file, 32).await;
        let caps = model.capabilities();
        assert!(caps.reasoning && caps.tools);
        assert!(!caps.can_disable_thinking);
        assert!(
            ChatTemplate::new(
                "{% if enable_thinking %}x{% endif %}".to_string(),
                String::new(),
                String::new()
            )
            .unwrap()
            .reads_enable_thinking()
        );
    }

    #[tokio::test]
    async fn local_requests_stream_blocks_and_report_usage() {
        // A model of 96 positions: the prompt spans two full K/V blocks.
        let file = write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        });
        let client = local_client(tiny_model(&file, 96).await);
        let message = "abcdefg".repeat(6);

        let request = UnifiedRequest::new("ignored")
            .add_user_message(message.clone())
            .set_temperature(0.0)
            .set_max_tokens(5);
        let (mut stream, response) = client.generate_stream_with_response(request).await.unwrap();
        while let Some(item) = stream.next().await {
            // Random weights may emit a malformed tool call, a parse error
            // item that does not end the stream.
            let _ = item;
        }
        let response = response.take().expect("terminal response");
        let usage = response.usage.expect("local engine reports usage");
        // `<bos>` + 42 letters.
        assert_eq!(usage.prompt_tokens, 43);
        assert!(usage.completion_tokens <= 5);
        assert_eq!(usage.reasoning_tokens, None);
        assert_eq!(response.provider, ProviderEndpoint::Local);
        let cache = response.prompt_cache_usage.expect("prefix reuse reported");
        assert_eq!((cache.read_tokens, cache.write_tokens), (0, 43));

        // The same prompt again reuses its full blocks below the last token:
        // 32 of 43 positions (the trailing 11 are recomputed).
        let request = UnifiedRequest::new("ignored")
            .add_user_message(message)
            .set_temperature(0.0)
            .set_max_tokens(2);
        let (mut stream, response) = client.generate_stream_with_response(request).await.unwrap();
        while stream.next().await.is_some() {}
        let cache = response.take().unwrap().prompt_cache_usage.unwrap();
        assert_eq!((cache.read_tokens, cache.write_tokens), (32, 11));
    }

    #[tokio::test]
    async fn queued_requests_all_complete_in_order() {
        let file = write_tiny_model();
        let client = Arc::new(local_client(tiny_model(&file, 32).await));
        let mut streams = Vec::new();
        for i in 0..4u32 {
            let request = UnifiedRequest::new("ignored")
                .add_user_message("abcdef")
                .set_temperature(0.0)
                .set_max_tokens(i + 1);
            streams.push(client.generate_stream_with_response(request).await.unwrap());
        }
        for (i, (mut stream, response)) in streams.into_iter().enumerate() {
            while stream.next().await.is_some() {}
            let usage = response.take().unwrap().usage.unwrap();
            assert!(usage.completion_tokens as usize <= i + 1);
        }
    }

    #[tokio::test]
    async fn invalid_requests_fail_before_queueing() {
        let file = write_tiny_model();
        let client = local_client(tiny_model(&file, 8).await);

        let err = client
            .generate_stream(
                UnifiedRequest::new("m")
                    .add_user_message("abc")
                    .set_temperature(f32::INFINITY),
            )
            .await
            .err()
            .expect("non-finite temperature");
        assert!(matches!(&err, CucaError::Config(m) if m.contains("temperature")));

        for p in [0.0, -0.5, 1.5, f32::NAN] {
            let err = client
                .generate_stream(
                    UnifiedRequest::new("m")
                        .add_user_message("abc")
                        .set_top_p(p),
                )
                .await
                .err()
                .expect("invalid top_p");
            assert!(
                matches!(&err, CucaError::Config(m) if m.contains("top_p")),
                "{err:?}"
            );
        }

        // `<bos>` + 7 letters fills the 8-position context.
        let err = client
            .generate_stream(UnifiedRequest::new("m").add_user_message("abcdefg"))
            .await
            .err()
            .expect("prompt fills the context");
        assert!(
            matches!(&err, CucaError::Provider { message, .. } if message.contains("context_length is 8")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn dropping_the_stream_leaves_the_engine_usable() {
        let file = write_tiny_model();
        let client = local_client(tiny_model(&file, 32).await);
        for _ in 0..3 {
            let request = UnifiedRequest::new("m")
                .add_user_message("abcd")
                .set_temperature(0.0)
                .set_max_tokens(20);
            drop(client.generate_stream(request).await.unwrap());
        }
        let request = UnifiedRequest::new("m")
            .add_user_message("abcd")
            .set_temperature(0.0)
            .set_max_tokens(2);
        let mut stream = client.generate_stream(request).await.unwrap();
        while let Some(item) = stream.next().await {
            let _ = item;
        }
    }

    #[tokio::test]
    async fn debug_prints_the_context_length() {
        let file = write_tiny_model();
        let model = tiny_model(&file, 32).await;
        assert_eq!(
            format!("{model:?}"),
            "LocalModel { context_length: 32, .. }"
        );
    }
}
