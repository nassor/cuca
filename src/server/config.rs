//! Server configuration: listener, API key and the models to load.
//!
//! A model is named by a [`ModelSpec`], which a `--model [name=]source`
//! argument or a `[[model]]` table of the `--config` file produces. Both
//! start from the global [`ModelOptions`] defaults (`--device`,
//! `--context-length`, ...) and a table overrides them per model:
//!
//! ```toml
//! [[model]]
//! name = "chat"
//! source = "qwen3-0.6b-q8_0"
//! batching = "seqs=8"
//!
//! [[model]]
//! source = "/models/gemma-4-E4B-it-Q4_0.gguf"
//! device = "vulkan"
//! gpu_layers = 20
//! ```

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use super::error::ServerError;
use crate::{
    BatchingConfig, DrafterSource, ExpertPlacement, LocalModelConfig, ModelSource, WeightQuant,
};

/// Most models one server loads: each holds its weights and K/V pools in
/// memory, so a longer list is a mistake, not a deployment.
pub const MAX_MODELS: usize = 16;

/// Everything `serve` needs to start.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address of the HTTP listener.
    pub listen: SocketAddr,
    /// The models to load, at least one, with unique names.
    pub models: Vec<ModelSpec>,
    /// When set, every route except `GET /health` needs
    /// `Authorization: Bearer <key>` (vLLM's `--api-key`).
    pub api_key: Option<String>,
    /// Serve the stats API (`/cuca/v1/stats`, `/cuca/v1/stats/stream`).
    pub stats_api: bool,
    /// The sliding window of the latency figures.
    pub latency_window: Duration,
}

impl ServerConfig {
    /// vLLM's default port.
    pub const DEFAULT_LISTEN: &'static str = "127.0.0.1:8000";
    /// Default [`ServerConfig::latency_window`].
    pub const DEFAULT_LATENCY_WINDOW: Duration = Duration::from_secs(60);

    /// A configuration for `models` listening on `listen`, the rest default.
    pub fn new(listen: SocketAddr, models: Vec<ModelSpec>) -> Self {
        Self {
            listen,
            models,
            api_key: None,
            stats_api: false,
            latency_window: Self::DEFAULT_LATENCY_WINDOW,
        }
    }

    /// Check the model list: between 1 and [`MAX_MODELS`] entries, names
    /// non-empty and unique.
    ///
    /// # Errors
    ///
    /// [`ServerError::Config`] naming the problem.
    pub fn validate(&self) -> Result<(), ServerError> {
        if self.models.is_empty() {
            return Err(ServerError::Config(
                "no model to serve: pass --model <[name=]source> or --config <file.toml>".into(),
            ));
        }
        if self.models.len() > MAX_MODELS {
            return Err(ServerError::Config(format!(
                "{} models; at most {MAX_MODELS} per server",
                self.models.len()
            )));
        }
        for (i, spec) in self.models.iter().enumerate() {
            if spec.name.trim().is_empty() || spec.name.chars().any(char::is_control) {
                return Err(ServerError::Config(format!(
                    "model {} has an empty or unprintable name",
                    i + 1
                )));
            }
            if self.models[..i].iter().any(|other| other.name == spec.name) {
                return Err(ServerError::Config(format!(
                    "two models are named `{}`",
                    spec.name
                )));
            }
        }
        if self.api_key.as_deref() == Some("") {
            return Err(ServerError::Config("--api-key is empty".into()));
        }
        Ok(())
    }
}

/// One model to load and the name clients select it by.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// The `model` value of requests and the id in `GET /v1/models`.
    pub name: String,
    /// The source as the user wrote it, for logs and the stats API.
    pub source_label: String,
    /// The device string as the user wrote it (`cpu` when none), for the
    /// stats API.
    pub device_label: String,
    /// What [`LocalModel::load`](crate::LocalModel::load) receives (the
    /// server adds the download-progress callback).
    pub config: LocalModelConfig,
}

/// The per-model options a `--model` argument inherits from the command line
/// and a `[[model]]` table can override. Every field is optional; `None`
/// keeps the engine's default.
#[derive(Debug, Clone, Default)]
pub struct ModelOptions {
    /// `+`-separated device list (`cpu`, `vulkan:0`, `cuda:0+cuda:1`).
    pub device: Option<String>,
    /// Maximum prompt plus completion tokens.
    pub context_length: Option<usize>,
    /// A `BatchingConfig` string (`seqs=8,kv=16384`).
    pub batching: Option<String>,
    /// An `ExpertPlacement` string (`resident`, `tiered:ram=4g`, `host`).
    pub experts: Option<String>,
    /// Layers on the GPU (`-ngl`).
    pub gpu_layers: Option<usize>,
    /// Model cache directory.
    pub cache_dir: Option<PathBuf>,
    /// A Jinja chat template file replacing the model's own.
    pub chat_template_file: Option<PathBuf>,
    /// Checkpoint conversion precision (`f16`, `q8_0`, `q4_0`).
    pub quant: Option<String>,
    /// A drafter: `in-trunk` or a model source.
    pub drafter: Option<String>,
    /// Tokens drafted per round (1 to 16).
    pub draft_tokens: Option<usize>,
}

impl ModelOptions {
    /// These options, with `self`'s set fields winning over `defaults`.
    fn over(&self, defaults: &Self) -> Self {
        Self {
            device: self.device.clone().or_else(|| defaults.device.clone()),
            context_length: self.context_length.or(defaults.context_length),
            batching: self.batching.clone().or_else(|| defaults.batching.clone()),
            experts: self.experts.clone().or_else(|| defaults.experts.clone()),
            gpu_layers: self.gpu_layers.or(defaults.gpu_layers),
            cache_dir: self
                .cache_dir
                .clone()
                .or_else(|| defaults.cache_dir.clone()),
            chat_template_file: self
                .chat_template_file
                .clone()
                .or_else(|| defaults.chat_template_file.clone()),
            quant: self.quant.clone().or_else(|| defaults.quant.clone()),
            drafter: self.drafter.clone().or_else(|| defaults.drafter.clone()),
            draft_tokens: self.draft_tokens.or(defaults.draft_tokens),
        }
    }

    /// Apply the options to a configuration for `source`.
    fn apply(&self, source: ModelSource) -> Result<(LocalModelConfig, String), ServerError> {
        let mut config = LocalModelConfig {
            source,
            ..LocalModelConfig::default()
        };
        let device_label = match &self.device {
            Some(list) => {
                config.set_devices(list)?;
                list.trim().to_string()
            }
            None => "cpu".to_string(),
        };
        config.context_length = self.context_length;
        if let Some(text) = &self.batching {
            config.batching = text.parse::<BatchingConfig>()?;
        }
        if let Some(text) = &self.experts {
            config.experts = text.parse::<ExpertPlacement>()?;
        }
        config.gpu_layers = self.gpu_layers;
        config.cache_dir.clone_from(&self.cache_dir);
        if let Some(path) = &self.chat_template_file {
            config.chat_template = Some(std::fs::read_to_string(path).map_err(|e| {
                ServerError::Config(format!("reading chat template `{}`: {e}", path.display()))
            })?);
        }
        if let Some(text) = &self.quant {
            config.convert.quant = text.parse::<WeightQuant>()?;
        }
        if let Some(text) = &self.drafter {
            config.drafter = Some(if text.trim().eq_ignore_ascii_case("in-trunk") {
                DrafterSource::InTrunk
            } else {
                DrafterSource::File(parse_source(text)?.0)
            });
        }
        if let Some(n) = self.draft_tokens {
            config.draft_tokens = n;
        }
        Ok((config, device_label))
    }
}

/// The `[[model]]` tables of a `--config` file.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    #[serde(default)]
    model: Vec<FileModel>,
}

/// One `[[model]]` table: the source, an optional name and the overrides.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileModel {
    name: Option<String>,
    source: String,
    device: Option<String>,
    context_length: Option<usize>,
    batching: Option<String>,
    experts: Option<String>,
    gpu_layers: Option<usize>,
    cache_dir: Option<PathBuf>,
    chat_template_file: Option<PathBuf>,
    quant: Option<String>,
    drafter: Option<String>,
    draft_tokens: Option<usize>,
}

impl FileModel {
    fn options(&self) -> ModelOptions {
        ModelOptions {
            device: self.device.clone(),
            context_length: self.context_length,
            batching: self.batching.clone(),
            experts: self.experts.clone(),
            gpu_layers: self.gpu_layers,
            cache_dir: self.cache_dir.clone(),
            chat_template_file: self.chat_template_file.clone(),
            quant: self.quant.clone(),
            drafter: self.drafter.clone(),
            draft_tokens: self.draft_tokens,
        }
    }
}

impl ModelSpec {
    /// Parse a `--model` argument: `[name=]source` where `source` is a preset
    /// name, `repo:revision:file` (a GGUF on Hugging Face), `repo:revision`
    /// (a safetensors checkpoint) or a filesystem path (a GGUF file, or a
    /// checkpoint directory). The name defaults to the preset name, the file
    /// stem or the repository name. `defaults` are the global options.
    ///
    /// # Errors
    ///
    /// [`ServerError::Config`] for an unknown source or an invalid option.
    pub fn from_arg(arg: &str, defaults: &ModelOptions) -> Result<Self, ServerError> {
        let (name, source) = match arg.split_once('=') {
            Some((name, source)) => (Some(name.trim().to_string()), source),
            None => (None, arg),
        };
        Self::build(name, source, defaults)
    }

    fn build(
        name: Option<String>,
        source: &str,
        options: &ModelOptions,
    ) -> Result<Self, ServerError> {
        let (source, default_name) = parse_source(source)?;
        let source_label = source_label(&source);
        let (config, device_label) = options.apply(source)?;
        Ok(Self {
            name: name.unwrap_or(default_name),
            source_label,
            device_label,
            config,
        })
    }
}

/// Every model a command line names: the `--model` arguments in order, then
/// the `[[model]]` tables of `--config`, all over the global `defaults`.
///
/// # Errors
///
/// [`ServerError::Config`] from [`ModelSpec::from_arg`] or
/// [`load_config_file`].
pub fn collect_models(
    args: &[String],
    config_file: Option<&Path>,
    defaults: &ModelOptions,
) -> Result<Vec<ModelSpec>, ServerError> {
    let mut models = args
        .iter()
        .map(|arg| ModelSpec::from_arg(arg, defaults))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(path) = config_file {
        models.extend(load_config_file(path, defaults)?);
    }
    Ok(models)
}

/// The models of a TOML `--config` file, each over the `defaults`.
///
/// # Errors
///
/// [`ServerError::Config`] when the file cannot be read, is not valid TOML,
/// has an unknown key, or a model is invalid.
pub fn load_config_file(
    path: &Path,
    defaults: &ModelOptions,
) -> Result<Vec<ModelSpec>, ServerError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ServerError::Config(format!("reading `{}`: {e}", path.display())))?;
    parse_config(&text, defaults)
        .map_err(|e| ServerError::Config(format!("`{}`: {e}", path.display())))
}

/// [`load_config_file`] over the text of the file.
fn parse_config(text: &str, defaults: &ModelOptions) -> Result<Vec<ModelSpec>, ServerError> {
    let file: ConfigFile = toml::from_str(text).map_err(|e| ServerError::Config(e.to_string()))?;
    file.model
        .iter()
        .map(|m| ModelSpec::build(m.name.clone(), &m.source, &m.options().over(defaults)))
        .collect()
}

/// A [`ModelSource`] from user text, and the model name it implies.
fn parse_source(text: &str) -> Result<(ModelSource, String), ServerError> {
    let text = text.trim();
    if let Some(preset) = ModelSource::preset(text) {
        return Ok((preset.source(), preset.name.to_string()));
    }
    let path = Path::new(text);
    if path.is_file() {
        return Ok((ModelSource::File(path.to_path_buf()), file_stem(path)));
    }
    if path.is_dir() {
        let name = path
            .file_name()
            .map_or_else(|| text.to_string(), |n| n.to_string_lossy().into_owned());
        return Ok((ModelSource::Dir(path.to_path_buf()), name));
    }
    let source: ModelSource = text.parse().map_err(|e| {
        ServerError::Config(format!(
            "model source `{text}` is neither an existing path nor a known source ({e})"
        ))
    })?;
    let name = match &source {
        ModelSource::HuggingFace { file, .. } => file_stem(Path::new(file)),
        ModelSource::Checkpoint { repo, .. } => repo.rsplit('/').next().unwrap_or(repo).to_string(),
        ModelSource::File(path) | ModelSource::Dir(path) => file_stem(path),
    };
    Ok((source, name))
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned())
}

/// The source as a short label for logs and the stats API.
fn source_label(source: &ModelSource) -> String {
    match source {
        ModelSource::HuggingFace {
            repo,
            revision,
            file,
            ..
        } => {
            let short = revision.get(..revision.len().min(8)).unwrap_or(revision);
            format!("{repo}:{short}:{file}")
        }
        ModelSource::File(path) | ModelSource::Dir(path) => path.display().to_string(),
        ModelSource::Checkpoint { repo, revision } => {
            let short = revision.get(..revision.len().min(8)).unwrap_or(revision);
            format!("{repo}:{short}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(arg: &str) -> ModelSpec {
        ModelSpec::from_arg(arg, &ModelOptions::default()).unwrap()
    }

    #[test]
    fn a_preset_names_itself_and_an_explicit_name_wins() {
        assert_eq!(spec("qwen3-0.6b-q8_0").name, "qwen3-0.6b-q8_0");
        assert_eq!(spec("small=qwen3-0.6b-q8_0").name, "small");
    }

    #[test]
    fn a_hugging_face_file_is_named_by_its_stem_and_a_checkpoint_by_its_repo() {
        let gguf = spec("org/repo:abcdef0123456789:model-Q4_0.gguf");
        assert_eq!(gguf.name, "model-Q4_0");
        assert_eq!(gguf.source_label, "org/repo:abcdef01:model-Q4_0.gguf");
        assert_eq!(spec("org/Repo:main").name, "Repo");
    }

    #[test]
    fn an_unknown_source_is_a_config_error_naming_it() {
        let err = ModelSpec::from_arg("nonsense", &ModelOptions::default()).unwrap_err();
        assert!(err.to_string().contains("nonsense"), "{err}");
    }

    #[test]
    fn a_table_overrides_the_global_defaults_field_by_field() {
        let defaults = ModelOptions {
            context_length: Some(2048),
            batching: Some("seqs=2".into()),
            ..ModelOptions::default()
        };
        let toml = r#"
            [[model]]
            source = "qwen3-0.6b-q8_0"
            batching = "seqs=8"

            [[model]]
            name = "other"
            source = "phi4-mini-q4_0"
        "#;
        let models = parse_config(toml, &defaults).unwrap();
        assert_eq!(models[0].config.context_length, Some(2048));
        assert_eq!(models[0].config.batching.max_sequences, 8);
        assert_eq!(models[1].name, "other");
        assert_eq!(models[1].config.batching.max_sequences, 2);
    }

    #[test]
    fn an_unknown_key_in_the_config_file_is_rejected() {
        let toml = "[[model]]\nsource = \"qwen3-0.6b-q8_0\"\ncontext = 4\n";
        assert!(parse_config(toml, &ModelOptions::default()).is_err());
    }

    #[test]
    fn the_device_list_sets_the_device_and_the_shards() {
        let options = ModelOptions {
            device: Some("vulkan:0+vulkan:1".into()),
            ..ModelOptions::default()
        };
        let model = ModelSpec::from_arg("qwen3-0.6b-q8_0", &options).unwrap();
        assert_eq!(model.config.shards.len(), 1);
        assert_eq!(model.device_label, "vulkan:0+vulkan:1");
    }

    #[test]
    fn validate_rejects_no_models_and_duplicate_names() {
        let listen = ServerConfig::DEFAULT_LISTEN.parse().unwrap();
        assert!(ServerConfig::new(listen, vec![]).validate().is_err());
        let one = spec("qwen3-0.6b-q8_0");
        let config = ServerConfig::new(listen, vec![one.clone(), one]);
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("qwen3-0.6b-q8_0"), "{err}");
    }
}
