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
//! `gemma2`, `llama` (Llama 1-3.x and compatible), `qwen2`, `qwen3`, `phi3`
//! (Phi 3, 3.5, 4-mini), `smollm3`, `granite`, `granitemoe`, `olmo2`,
//! `ernie4_5` or `qwen35` (CPU only) runs: the loader reads every
//! dimension (layers, widths, per-layer head counts and head dimensions,
//! FFN widths or experts, vocabulary, RoPE variants) from the file's
//! metadata and tensor shapes, handles biases, QK-norm, fused QKV and
//! gate/up tensors, tied or separate output heads, sliding windows, NoPE
//! layers, soft-caps and mixtures of experts, and rejects anything else (an
//! unknown architecture, a tensor dtype candle cannot read) with an error
//! naming the cause. [`ModelSource::PRESETS`] lists the verified models,
//! each pinned to a repository commit and SHA-256 (by name through
//! `str::parse`, e.g. `qwen3-0.6b-q8_0`), next to
//! [`ModelSource::hugging_face`] and [`ModelSource::File`] for any other
//! file. The default is Gemma 4 E2B Q4_0 (2.84 GB). Chat template, tool-call
//! syntax and reasoning markers come from the file ([`LocalModel::capabilities`]
//! reports what a model supports).
//!
//! Bounds:
//!
//! - Single-slot FIFO engine: one request runs at a time, at most 16 queue
//!   behind it (further callers wait), 64 output blocks buffered per request.
//! - K/V cache memory is `context_length * 2 * 4 * sum(n_kv_head * head_dim)`
//!   bytes over the layers that own K/V ([`LocalModel::kv_cache_bytes`]), in
//!   RAM or device memory: per position 36,864 bytes for Gemma 4 E2B (302 MB
//!   at 8192), 114,688 for E4B, 229,376 for Qwen3-0.6B and 786,432 for
//!   Phi-3.5-mini (6.44 GB at 8192: a full-attention model with 32 KV heads
//!   needs a smaller `context_length`).
//! - Gemma 4's per-layer embedding rows are read from disk per input token
//!   instead of being held in RAM (5,040 bytes each and 1.32 GB saved in E2B
//!   Q4_0). A mixture-of-experts model keeps every expert resident (file
//!   size); a pass only computes the experts its tokens route to.
//! - Prefill runs in chunks of at most 256 tokens, bounding the score matrix.
//!
//! # Determinism
//!
//! On the CPU a token's logits do not depend on how the prompt was split:
//! prefill and speculative verification run in a row-exact mode (float
//! matmuls per row, attention per query row) that equals single-token decoding
//! bit for bit (per-row attention on scoped threads, float matmuls per row,
//! and a safe Q4_0/Q8_0 matmul kernel whose rows are independent of their
//! batch), at no cost in prefill throughput. A request that
//! reuses a cached prefix therefore generates what a cold request generates.
//! (Quantized matmuls round activations to Q8_0, which otherwise amplifies the
//! batch-shape rounding of a float matmul into visible logit jumps.)
//!
//! # Speculative decoding (opt-in)
//!
//! [`LocalModelConfig::drafter`] attaches Gemma 4's multi-token-prediction
//! assistant (preset `gemma4-e2b-mtp-f16`) to a Gemma 4 E2B target. The
//! drafter proposes up to [`LocalModelConfig::draft_tokens`] tokens from the
//! target's last hidden state, sharing the target's K/V cache; the target
//! verifies them in one batched pass and keeps the longest prefix it agrees
//! with. At temperature 0 the output equals plain greedy decoding; above 0 the
//! standard speculative-sampling acceptance rule (accept with
//! `min(1, p/q)`, resample the rejection from `max(0, p - q)`) preserves the
//! target's sampling distribution. An engine-side governor measures the
//! tokens per second of plain and speculative rounds and drafts only while
//! speculation is faster, re-probing with exponential back-off, so the worst
//! case stays near plain decoding (a seeded sampling run always drafts, so equal
//! prompts still reproduce equal tokens). Measured on Gemma 4 E2B Q4_0 with the
//! E2B assistant (128-token replies, quiet development machine, speed against
//! plain decoding on the same device): CPU 1.17x on a code prompt, 1.37x on a
//! list, 1.00x on prose; Vulkan (RX 9070 XT) 0.97x,
//! 0.95x and 0.96x: GPU single-token decoding is host-bound (35 tok/s), so
//! the drafter's own launches cost as much as the verification saves, and the
//! governor keeps the run within about 5 % of plain decoding. Drafting is off
//! by default; it pays on the CPU.
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
//! replies, two runs on a quiet CPU, plain decoding 52-53 tok/s): a code
//! prompt 1.36-1.40x at one draft per round, 1.21x at two and 1.57-1.63x at
//! three or four; a list 1.05-1.41x; creative prose 0.96-1.07x (its text is
//! hard to predict, so the governor keeps drafting mostly off); the three
//! together 1.11-1.24x for one to four drafts (drafted tokens accepted: 75 %,
//! 60 %, 51 %, 50 %). A draft step costs 1.05 ms against a 19-22 ms trunk
//! step because it ranks only the first 65,536 of the 248,320 token ids: 5x
//! cheaper than the whole vocabulary for 4 % fewer accepted tokens per round.

mod chat;
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
const DEFAULT_DRAFT_TOKENS: usize = 3;
/// Largest accepted [`LocalModelConfig::draft_tokens`].
const MAX_DRAFT_TOKENS: usize = 16;
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
}

impl ModelPreset {
    /// The pinned, checksummed [`ModelSource`] of this preset.
    pub fn source(&self) -> ModelSource {
        ModelSource::HuggingFace {
            repo: self.repo.to_string(),
            revision: self.revision.to_string(),
            file: self.file.to_string(),
            sha256: Some(self.sha256.to_string()),
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
    }
}

const PRESET_TABLE: [ModelPreset; 23] = [
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
        "gemma2-2b-q4_0",
        "second-state/gemma-2-2b-it-GGUF",
        "b7d6a1a55228d89cc129757c687384b96781f97e",
        "gemma-2-2b-it-Q4_0.gguf",
        "2c1a93b3dfaaff7345ef184956f3adc78e37e6a3a4fd7bddb25bff01607ef12c",
        1_629_509_216,
        "gemma2",
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
        "qwen2.5-0.5b-q8_0",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "9217f5db79a29953eb74d5343926648285ec7e67",
        "qwen2.5-0.5b-instruct-q8_0.gguf",
        "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
        675_710_816,
        "qwen2",
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
        "llama3.2-1b-q4_0",
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "067b946cf014b7c697f3654f621d577a3e3afd1c",
        "Llama-3.2-1B-Instruct-Q4_0.gguf",
        "fa0390e7c043f89ae1847bd6682d748041a99d4ef3de0e0b27d33b6af97a8be8",
        773_025_920,
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
        "phi3.5-mini-q4_0",
        "bartowski/Phi-3.5-mini-instruct-GGUF",
        "6d70da17e749a471ccb62ade694486011a75cda3",
        "Phi-3.5-mini-instruct-Q4_0.gguf",
        "6e0a4c93fe5db9408348cd2cbd15eee6f448efe6f91255a208c2d28701259b1d",
        2_182_468_896,
        "phi3",
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
        "smollm2-360m-q8_0",
        "bartowski/SmolLM2-360M-Instruct-GGUF",
        "7be6f65f1db715fe5dc5a4634c0d459b4eed42ec",
        "SmolLM2-360M-Instruct-Q8_0.gguf",
        "c004ac34cce45e03e4452bca472535bc371fcb0d07ae560bba2c57ce4a284b7b",
        386_405_280,
        "llama",
    ),
    preset(
        "tinyllama-1.1b-q4_0",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "52e7645ba7c309695bec7ac98f4f005b139cf465",
        "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556",
        637_699_456,
        "llama",
    ),
    preset(
        "granite3.1-moe-1b-q4_0",
        "bartowski/granite-3.1-1b-a400m-instruct-GGUF",
        "940d2e1f9f65330615c7c8e980e6c5ac73d3360c",
        "granite-3.1-1b-a400m-instruct-Q4_0.gguf",
        "642a83a3e38607015d14ccc0cb8069d02b94a32d4bf8fd18b958ead08d4b6e0e",
        771_466_560,
        "granitemoe",
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

/// Where [`LocalModel::load`] gets the GGUF file from.
///
/// Any GGUF whose architecture the engine supports works (see the module
/// docs): the loader derives every dimension from the file. The verified
/// presets are in [`ModelSource::PRESETS`]; other files are described with
/// [`ModelSource::hugging_face`] or [`ModelSource::File`]. A source also
/// parses from a string (`str::parse`): a preset name (ASCII
/// case-insensitive), or `repo:revision:file`.
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
    },
    /// A GGUF file already on disk; nothing is downloaded.
    File(PathBuf),
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
            _ => Err(CucaError::Config(format!(
                "unknown model `{s}`: expected a preset ({}) or `repo:revision:file`",
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
/// `temperature` may be overridden per request with
/// `UnifiedRequest::set_temperature`; `top_k`, `top_p` and `seed` apply to
/// every request.
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
    /// Maximum tokens of prompt plus completion per request, also the K/V
    /// cache capacity: it costs `context_length * 2 * 4 * sum(n_kv_head *
    /// head_dim)` bytes of RAM over the K/V-owning layers (Gemma 4 E2B
    /// 36,864 per position, Qwen3-0.6B 229,376, Phi-3.5-mini 786,432, i.e.
    /// 6.44 GB at 8192: size it for the model; [`LocalModel::kv_cache_bytes`]
    /// reports the result; see [`LocalModel`]). `None` (the default)
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
    /// Tokens the drafter proposes per verification round, `1..=16`; ignored
    /// without a drafter. Longer rounds win more on predictable text and
    /// waste more verification work on unpredictable text.
    pub draft_tokens: usize,
    /// Optional download progress observer.
    pub on_download_progress: Option<DownloadProgressCallback>,
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
        }
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
            .finish()
    }
}

/// What a loaded engine hands back to `load`: the pieces the request path
/// needs without touching the engine thread.
type Ready = Result<(Arc<Tokenizer>, ChatTemplate, ChatFormat, usize, usize), CucaError>;

/// State shared by every clone of a [`LocalModel`].
pub(crate) struct LocalModelInner {
    /// Queue into the engine thread; dropping the last sender ends it.
    jobs: mpsc::Sender<engine::Job>,
    tokenizer: Arc<Tokenizer>,
    template: ChatTemplate,
    format: ChatFormat,
    context_length: usize,
    kv_bytes_per_position: usize,
    draft_stats: Arc<engine::DraftCounters>,
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
/// - The engine is a single slot: requests run one at a time in FIFO order,
///   with at most 16 queued behind the running one. Past that,
///   `generate_stream` waits for a free queue slot (backpressure, not an
///   error). A running request streams through a bounded 64-block channel, so
///   a slow consumer paces generation, and dropping the stream stops it.
/// - The K/V cache is preallocated for `context_length` positions and costs
///   `context_length * 2 * 4 * sum(n_kv_head * head_dim)` bytes over the layers
///   that own K/V (each layer's head count and head dimension come from the
///   file; [`LocalModel::kv_cache_bytes`] reports the total): Gemma 4 E2B
///   `context_length * 36,864` bytes (302 MB at the default 8192), E4B
///   `context_length * 114,688` (939 MB at 8192), Qwen3-0.6B
///   `context_length * 229,376` (28 layers, 8 KV heads of 128). It is
///   reused across requests: a prompt that shares a prefix with the previous
///   one only prefills the new tokens, reported as the response's
///   `prompt_cache_usage`.
/// - Weights stay quantized (a mixture-of-experts model keeps every expert,
///   split per expert, so memory is the file size; a pass computes only the
///   routed experts). Gemma 4's per-layer embedding table
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
///   one model. `temperature` and `max_tokens` apply per request; `top_k`,
///   `top_p` and `seed` come from [`SamplingConfig`]. Generation also stops
///   at the context limit.
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

impl LocalModel {
    /// Bytes the K/V cache occupies: `context_length * 2 * 4 * sum(n_kv_head
    /// * head_dim)` over the layers that own K/V, in RAM or device memory.
    ///
    /// A hybrid model (Qwen 3.5) owns K/V only in its attention layers, and
    /// besides keeps a recurrent state of fixed size (about 20 MB for the
    /// 0.8B model, three copies while speculating) that is not counted here.
    pub fn kv_cache_bytes(&self) -> u64 {
        (self.inner.context_length as u64) * (self.inner.kv_bytes_per_position as u64)
    }

    /// What this model supports: reasoning, switching it off, tool calls.
    pub fn capabilities(&self) -> LocalCapabilities {
        let reasoning = self.inner.format.reasoning().is_some();
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
    /// a [`LocalDevice`] whose cargo feature is not enabled);
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
                                engine.kv_bytes_per_position(),
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
                while let Some(job) = jobs_rx.blocking_recv() {
                    engine.run(job);
                }
            })
            .map_err(|e| CucaError::Io {
                message: format!("failed to start the local engine thread: {e}"),
            })?;

        match ready_rx.await {
            Ok(Ok((tokenizer, template, format, context_length, kv_bytes_per_position))) => {
                Ok(LocalModel {
                    inner: Arc::new(LocalModelInner {
                        jobs: jobs_tx,
                        tokenizer,
                        template,
                        format,
                        context_length,
                        kv_bytes_per_position,
                        draft_stats,
                    }),
                })
            }
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
    Ok(())
}

/// The GGUF path for `config.source`, downloading it first when needed.
///
/// Runs on the engine thread. The download uses a private current-thread
/// runtime and HTTP client (never the `CucaClient` one), both dropped before
/// returning: the engine loop later blocks on a channel, which must not happen
/// inside a runtime.
fn resolve_model_path(
    config: &LocalModelConfig,
    source: &ModelSource,
) -> Result<PathBuf, CucaError> {
    match source {
        ModelSource::File(path) => {
            if !path.is_file() {
                return Err(CucaError::Io {
                    message: format!("model file `{}` does not exist", path.display()),
                });
            }
            Ok(path.clone())
        }
        ModelSource::HuggingFace {
            repo,
            revision,
            file,
            sha256,
        } => {
            let cache_dir = download::resolve_cache_dir(config.cache_dir.as_deref(), &|key| {
                std::env::var_os(key)
            })?;
            let final_path = download::cached_path(&cache_dir, repo, revision, file)?;
            if !final_path.exists() {
                let token = config.hf_token.clone().or_else(|| {
                    std::env::var("HF_TOKEN")
                        .ok()
                        .filter(|token| !token.is_empty())
                });
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                let client = reqwest::Client::builder()
                    .connect_timeout(CONNECT_TIMEOUT)
                    .build()?;
                runtime.block_on(download::ensure_downloaded(
                    &client,
                    download::HF_ENDPOINT,
                    repo,
                    revision,
                    file,
                    sha256.as_deref(),
                    &final_path,
                    token.as_deref(),
                    config.on_download_progress.as_ref(),
                ))?;
            }
            Ok(final_path)
        }
    }
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
    /// `temperature` is invalid; [`CucaError::Provider`] when the request
    /// cannot be rendered (for example an image block), the prompt does not
    /// fit the context, or the engine has shut down.
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
    use crate::provider::local::model::tiny::{TinyModel, write_tiny_model};

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
        } = ModelSource::default_model()
        else {
            panic!("default source is a Hugging Face file");
        };
        assert_eq!(repo, "ggml-org/gemma-4-E2B-it-GGUF");
        assert_eq!(revision.len(), 40);
        assert_eq!(file, "gemma-4-E2B-it-Q4_0.gguf");
        assert_eq!(sha256.map(|s| s.len()), Some(64));
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
        for name in [
            "qwen3-0.6b-q8_0",
            "llama3.2-1b-q4_0",
            "phi3.5-mini-q4_0",
            "granite3.1-moe-1b-q4_0",
        ] {
            assert!(ModelSource::preset(name).is_some(), "{name}");
        }
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
        ] {
            let err = LocalModel::load(config).await.unwrap_err();
            assert!(
                matches!(&err, CucaError::Config(m) if m.contains(needle)),
                "{err}"
            );
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
        let file = write_tiny_model();
        let client = local_client(tiny_model(&file, 32).await);

        let request = UnifiedRequest::new("ignored")
            .add_user_message("abcdefg")
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
        // `<bos>` + 7 letters.
        assert_eq!(usage.prompt_tokens, 8);
        assert!(usage.completion_tokens <= 5);
        assert_eq!(usage.reasoning_tokens, None);
        assert_eq!(response.provider, ProviderEndpoint::Local);
        let cache = response.prompt_cache_usage.expect("prefix reuse reported");
        assert_eq!((cache.read_tokens, cache.write_tokens), (0, 8));

        // The same prompt again reuses all but the last token.
        let request = UnifiedRequest::new("ignored")
            .add_user_message("abcdefg")
            .set_temperature(0.0)
            .set_max_tokens(2);
        let (mut stream, response) = client.generate_stream_with_response(request).await.unwrap();
        while stream.next().await.is_some() {}
        let cache = response.take().unwrap().prompt_cache_usage.unwrap();
        assert_eq!((cache.read_tokens, cache.write_tokens), (7, 1));
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
