//! The test configuration: `cuca-test.kdl` at the repository root, read once
//! per test binary.
//!
//! Every live or oracle test reads its settings from here instead of the
//! environment. A missing file is the default configuration, under which every
//! live test skips. `cuca-test.example.kdl` lists every key. The file is
//! gitignored, so two runs with different settings cannot overlap.
//!
//! The same file is compiled into the integration tests (`tests/common`,
//! `tests/local_env`) and into the library's unit tests (`crate::test_config`,
//! through `#[path]` and `extern crate self as cuca`), so it names library
//! items as `cuca::...` and imports nothing from `crate::`.
//!
//! `#![allow(dead_code)]`: no test binary uses every key.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use serde::Deserialize;

#[cfg(feature = "provider-local")]
use cuca::config::ConfigValue;
#[cfg(feature = "provider-local")]
use cuca::{DrafterSource, ExpertPlacement, LocalDevice, LocalModelConfig, ModelSource};

/// Where the test configuration lives: the repository root.
pub const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/cuca-test.kdl");

/// The test configuration; [`TestConfig::default`] when the file is absent.
///
/// # Panics
///
/// When the file exists and does not parse, naming the problem.
pub fn get() -> &'static TestConfig {
    static CONFIG: LazyLock<TestConfig> = LazyLock::new(load);
    &CONFIG
}

fn load() -> TestConfig {
    let path = Path::new(PATH);
    if !path.exists() {
        return TestConfig::default();
    }
    read(path)
}

#[cfg(feature = "config-kdl")]
fn read(path: &Path) -> TestConfig {
    let config: TestConfig = cuca::config::load_kdl_file(path).unwrap_or_else(|e| panic!("{e}"));
    // `local.model` is kept raw; read it once here so a typo fails every test
    // that loads the configuration, not only the one that loads a model.
    #[cfg(feature = "provider-local")]
    let _ = config.local_model();
    config
}

#[cfg(not(feature = "config-kdl"))]
fn read(_: &Path) -> TestConfig {
    panic!("cuca-test.kdl exists but the `config-kdl` feature is off: add it to --features")
}

/// The whole file.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TestConfig {
    /// A skip becomes a failure: an unreachable server, a missing model or
    /// oracle panics instead of printing `SKIP:`.
    pub require_live: bool,
    /// The model cache of every test that downloads or converts, unless
    /// `local { model { cache_dir ... } }` names another one.
    pub cache_dir: Option<PathBuf>,
    /// The llama.cpp server of the plugin and service suites.
    pub llamacpp: LlamaCpp,
    /// The in-process model of the `provider_local*` suites.
    #[cfg(feature = "provider-local")]
    pub local: LocalSection,
    /// Oracle dumps and real-model files of the local engine's unit tests.
    #[cfg(feature = "provider-local")]
    pub oracle: OracleSection,
    #[cfg(not(feature = "provider-local"))]
    local: serde::de::IgnoredAny,
    #[cfg(not(feature = "provider-local"))]
    oracle: serde::de::IgnoredAny,
}

/// `llamacpp base_url=... model=...`.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlamaCpp {
    /// The server's OpenAI-compatible base, `/v1` included.
    pub base_url: Option<String>,
    /// The model id to request; the first id the server reports when unset.
    pub model: Option<String>,
}

/// `local { ... }`.
#[cfg(feature = "provider-local")]
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LocalSection {
    /// `model { ... }`: a [`LocalModelConfig`], kept raw so a suite can tell
    /// a written key from a default one.
    model: Option<ConfigValue>,
    /// Without a `source`, run the default preset (downloading it).
    pub download_default: bool,
    /// The reference placement `provider_local_tiers.rs` compares with.
    pub experts_ref: Option<ExpertPlacement>,
    /// The draft lengths `provider_local_mtp.rs` compares; `[3]` when empty.
    #[serde(deserialize_with = "cuca::config::one_or_many")]
    pub draft_tokens: Vec<usize>,
    /// Report a CPU speculative output that differs from plain decoding
    /// instead of failing.
    pub allow_mismatch: bool,
    /// Print each greedy reply as a `REPLY|...` line.
    pub print_replies: bool,
    /// Run the long-prompt speed scenario.
    pub perf: bool,
    /// No round of the speed scenario hits the prefix cache.
    pub perf_cold: bool,
}

/// `oracle { ... }`: reference dumps and real-model files.
#[cfg(feature = "provider-local")]
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OracleSection {
    /// The transformers oracle dumps (`*.json`).
    pub ref_dir: Option<PathBuf>,
    /// Only the converter oracles whose repository contains this text.
    pub convert_only: Option<String>,
    /// The tolerance of the local converter oracle.
    pub ref_tol: Option<f32>,
    /// The checkpoint, GGUF and tensor-name prefix the local converter oracle
    /// compares.
    pub ref_ckpt: Option<String>,
    pub ref_gguf: Option<String>,
    pub ref_prefix: Option<String>,
    /// The tensor the converter probe prints.
    pub convert_probe: Option<String>,
    /// The DFlash transformers oracle directory.
    pub dflash_ref: Option<PathBuf>,
    /// The Hugging Face catalog the tokenizer goldens read; the cache's
    /// `huggingface` directory when unset.
    pub catalog_dir: Option<PathBuf>,
    /// A GGUF and its tokenizer cases for the tokenizer golden test.
    pub tok_gguf: Option<PathBuf>,
    pub tok_cases: Option<PathBuf>,
    /// Most cases the tokenizer golden test may differ on.
    pub tok_max_diff: Option<usize>,
    /// A real Mistral Small 4 GGUF for the latent-attention check.
    pub mistral4_gguf: Option<PathBuf>,
    /// DeepSeek-V4.1.
    pub ds41: Ds41,
    /// Gemma 4 26B-A4B mixture of experts.
    pub g4moe: G4Moe,
}

/// `oracle { ds41 { ... } }`.
#[cfg(feature = "provider-local")]
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Ds41 {
    /// The tiny torch oracle (`make_ref.py`): `target/`, `expected/`, logits.
    pub oracle_dir: Option<PathBuf>,
    /// The reference tokenizer files.
    pub tokenizer_ref_dir: Option<PathBuf>,
    /// The reference `tokenizer.json`.
    pub tokenizer_json: Option<PathBuf>,
    /// The real-weights oracle directory (with the top-level `cache_dir`
    /// holding the converted set).
    pub real_dir: Option<PathBuf>,
    /// Tiered expert RAM in GiB (default 90 on the CPU, 85 on a GPU).
    pub real_ram_gib: Option<u64>,
    /// Offload the last this many layers (3 GiB device tier).
    pub real_gpu_layers: Option<usize>,
    /// The production-path engine's context, sequences and rows.
    pub real_ctx: Option<usize>,
    pub real_seqs: Option<usize>,
    pub real_rows: Option<usize>,
    /// Run the speed probes.
    pub real_speed: bool,
    /// Run the production-path check of the GPU engine.
    pub real_paths: bool,
    /// The production-path check with the quantizers on.
    pub real_qdq: bool,
    /// Write every position's logits here.
    pub real_dump_dir: Option<PathBuf>,
}

/// `oracle { g4moe { ... } }`.
#[cfg(feature = "provider-local")]
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct G4Moe {
    /// The transformers oracle directory.
    pub oracle_dir: Option<PathBuf>,
    /// The real GGUF and its llama.cpp reference.
    pub real: Option<PathBuf>,
    pub real_ref: Option<PathBuf>,
    /// A truncated real GGUF and its llama.cpp reference.
    pub trunc: Option<PathBuf>,
    pub trunc_ref: Option<PathBuf>,
}

#[cfg(feature = "provider-local")]
impl TestConfig {
    /// `local { model { ... } }` as a [`LocalModelConfig`] (the default when
    /// absent), with the top-level `cache_dir` when it names none.
    ///
    /// # Panics
    ///
    /// When the node does not deserialize, naming the file and the problem.
    pub fn local_model(&self) -> LocalModelConfig {
        let mut config = match &self.local.model {
            Some(value) => LocalModelConfig::deserialize(value.clone())
                .unwrap_or_else(|e| panic!("{PATH}: local {{ model {{ ... }} }}: {e}")),
            None => LocalModelConfig::default(),
        };
        if config.cache_dir.is_none() {
            config.cache_dir.clone_from(&self.cache_dir);
        }
        config
    }

    /// `local { model { ... } }` as written (`None` when absent), for a suite
    /// that writes the model into another configuration file.
    pub fn local_model_raw(&self) -> Option<&ConfigValue> {
        self.local.model.as_ref()
    }

    /// Whether `local { model { ... } }` writes `key`.
    pub fn local_has(&self, key: &str) -> bool {
        self.local
            .model
            .as_ref()
            .is_some_and(|model| model.get(key).is_some())
    }

    /// The path of a written `source file=...` or `source dir=...`.
    pub fn local_path(&self) -> Option<PathBuf> {
        if !self.local_has("source") {
            return None;
        }
        match self.local_model().source {
            ModelSource::File(path) | ModelSource::Dir(path) => Some(path),
            _ => None,
        }
    }

    /// The path of a `drafter file=...` or `drafter dir=...`.
    pub fn drafter_path(&self) -> Option<PathBuf> {
        match self.local_model().drafter {
            Some(DrafterSource::File(ModelSource::File(path) | ModelSource::Dir(path))) => {
                Some(path)
            }
            _ => None,
        }
    }

    /// `device` then every shard; `None` without a written `device`.
    pub fn local_devices(&self) -> Option<Vec<LocalDevice>> {
        if !self.local_has("device") {
            return None;
        }
        let config = self.local_model();
        Some(
            std::iter::once(config.device)
                .chain(config.shards)
                .collect(),
        )
    }
}
