//! The `cuca` configuration file: one KDL document whose sections are the
//! structs below, read by `cuca serve` and `cuca balance` (`--config`, default
//! `cuca.kdl`). Every key is a field name; a `model` node is a
//! [`LocalModelConfig`] with the name clients select it by as its argument.
//!
//! ```kdl
//! variables {
//!     API_KEY "change-me"          // delete the entry to read API_KEY from the environment
//! }
//!
//! log format="pretty" level="info"
//!
//! serve {
//!     listen "127.0.0.1:8000"
//!     api_key "${API_KEY}"
//!     stats_api #true
//!     latency_window "60s"
//!
//!     model "chat" {
//!         source "qwen3-0.6b-q8_0"
//!         batching max_sequences=8
//!     }
//!     model {
//!         source file="/models/gemma-4-E4B-it-Q4_0.gguf"
//!         device "vulkan:0"
//!         gpu_layers 20
//!     }
//! }
//!
//! balance {
//!     listen "127.0.0.1:8080"
//!     upstreams "http://10.0.0.1:8000" "http://10.0.0.2:8000"
//!     upstream_api_key "${API_KEY}"
//!     policy "prefix-affinity"
//!     health_interval "1s"
//!     stats_api #true
//! }
//! ```
//!
//! `${NAME}` and `${NAME:-default}` placeholders resolve against the
//! `variables` block, then the process environment ([`crate::config`]); no
//! other setting is read from the environment. A relative path resolves
//! against the working directory. `examples/configs/cuca.kdl` is this file.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer};

use super::balancer::BalancerConfig;
use super::error::ServerError;
use super::telemetry::LogConfig;
use crate::config::ConfigValue;
use crate::error::CucaError;
use crate::{LocalModelConfig, ModelSource};

/// Most models one server loads: each holds its weights and K/V pools in
/// memory, so a longer list is a mistake, not a deployment.
pub const MAX_MODELS: usize = 16;

/// The whole configuration file. `serve` reads the `serve` section,
/// `balance` the `balance` one; both read `log`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CucaConfig {
    /// Logging of either command.
    pub log: LogConfig,
    /// The model server.
    pub serve: Option<ServerConfig>,
    /// The balancer.
    pub balance: Option<BalancerConfig>,
}

impl CucaConfig {
    /// Read the KDL file at `path`.
    ///
    /// # Errors
    ///
    /// [`ServerError::Config`] naming the file, for an unreadable file, a KDL
    /// syntax error, an unset `${VAR}` without a default, an unknown key or a
    /// value of the wrong type.
    pub fn load(path: &Path) -> Result<Self, ServerError> {
        crate::config::load_kdl_file(path).map_err(config_error)
    }

    /// Read a configuration from KDL `text`.
    ///
    /// # Errors
    ///
    /// As [`CucaConfig::load`], without the file name.
    pub fn from_kdl_str(text: &str) -> Result<Self, ServerError> {
        crate::config::load_kdl_str(text).map_err(config_error)
    }

    /// The `serve` section.
    ///
    /// # Errors
    ///
    /// [`ServerError::Config`] when the file has none.
    pub fn into_serve(self) -> Result<ServerConfig, ServerError> {
        self.serve
            .ok_or_else(|| ServerError::Config("the configuration has no `serve` section".into()))
    }

    /// The `balance` section.
    ///
    /// # Errors
    ///
    /// [`ServerError::Config`] when the file has none.
    pub fn into_balance(self) -> Result<BalancerConfig, ServerError> {
        self.balance
            .ok_or_else(|| ServerError::Config("the configuration has no `balance` section".into()))
    }
}

/// A configuration error of the library as the server's, without a second
/// `configuration error:` prefix.
fn config_error(error: CucaError) -> ServerError {
    match error {
        CucaError::Config(message) => ServerError::Config(message),
        other => other.into(),
    }
}

/// A string parsed with `T`'s `FromStr`, whose error is a [`ServerError`]:
/// the KDL form of [`LogFormat`](super::LogFormat) and the balancer's
/// [`Policy`](super::balancer::routing::Policy).
pub(crate) fn parsed<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr<Err = ServerError>,
{
    let text = String::deserialize(deserializer)?;
    text.parse().map_err(|e| match e {
        ServerError::Config(message) => D::Error::custom(message),
        other => D::Error::custom(other),
    })
}

/// Everything `serve` needs to start: the `serve` section.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Address of the HTTP listener ([`ServerConfig::DEFAULT_LISTEN`]).
    pub listen: SocketAddr,
    /// The models to load, at least one, with unique names: one `model` node
    /// each.
    #[serde(rename = "model", deserialize_with = "crate::config::one_or_many")]
    pub models: Vec<ModelSpec>,
    /// When set, every route except `GET /health` needs
    /// `Authorization: Bearer <key>` (vLLM's `--api-key`).
    pub api_key: Option<String>,
    /// Serve the stats API (`/cuca/v1/stats`, `/cuca/v1/stats/stream`).
    pub stats_api: bool,
    /// The sliding window of the latency figures (`"60s"`).
    #[serde(deserialize_with = "crate::config::duration")]
    pub latency_window: Duration,
}

impl ServerConfig {
    /// vLLM's default port on the loopback interface.
    pub const DEFAULT_LISTEN: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8000);
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
                "no model to serve: add a `model` node to the `serve` section".into(),
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
            return Err(ServerError::Config("api_key is empty".into()));
        }
        Ok(())
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self::new(Self::DEFAULT_LISTEN, Vec::new())
    }
}

/// One model to load and the name clients select it by: a `model` node,
/// `model "name" { source "..."; ... }`, whose children are the
/// [`LocalModelConfig`].
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// The `model` value of requests and the id in `GET /v1/models`.
    pub name: String,
    /// What [`LocalModel::load`](crate::LocalModel::load) receives (the
    /// server adds the download-progress callback).
    pub config: LocalModelConfig,
}

impl ModelSpec {
    /// A model called `name`, or when `None` by its source: the preset name,
    /// the file stem of a Hugging Face file or a local file, the repository
    /// name of a checkpoint, the directory name of a checkpoint directory.
    pub fn new(name: Option<String>, config: LocalModelConfig) -> Self {
        let name = name.unwrap_or_else(|| default_name(&config.source));
        Self { name, config }
    }

    /// The source as a short label for logs and the stats API.
    pub fn source_label(&self) -> String {
        match &self.config.source {
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

    /// The device, then each shard, joined by `+` (`vulkan:0+vulkan:1`).
    pub fn device_label(&self) -> String {
        std::iter::once(&self.config.device)
            .chain(&self.config.shards)
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("+")
    }
}

/// The name a model without one goes by; see [`ModelSpec::new`].
fn default_name(source: &ModelSource) -> String {
    if let Some(preset) = ModelSource::PRESETS.iter().find(|p| p.source() == *source) {
        return preset.name.to_string();
    }
    let stem = |path: &Path| {
        path.file_stem()
            .map_or_else(String::new, |s| s.to_string_lossy().into_owned())
    };
    match source {
        ModelSource::HuggingFace { file, .. } => stem(Path::new(file)),
        ModelSource::Checkpoint { repo, .. } => repo.rsplit('/').next().unwrap_or(repo).to_string(),
        ModelSource::File(path) => stem(path),
        ModelSource::Dir(path) => path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        ),
    }
}

/// A `model` node: a table whose `id` (the node's argument) is the name and
/// whose other keys are a [`LocalModelConfig`]. Written by hand because
/// `flatten` cannot be combined with `deny_unknown_fields`.
impl<'de> Deserialize<'de> for ModelSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let ConfigValue::Object(mut table) = ConfigValue::deserialize(deserializer)? else {
            return Err(D::Error::custom(
                "write `model \"name\" { source \"...\" }`",
            ));
        };
        let name = match table.remove("id") {
            None => None,
            Some(ConfigValue::String(name)) => Some(name),
            Some(other) => {
                return Err(D::Error::custom(format!(
                    "a model's name must be a string, got {other}"
                )));
            }
        };
        let label = name.clone().or_else(|| {
            table
                .get("source")
                .map(|s| s.as_str().map_or_else(|| s.to_string(), str::to_string))
        });
        let config = LocalModelConfig::deserialize(ConfigValue::Object(table)).map_err(|e| {
            D::Error::custom(match &label {
                Some(label) => format!("model `{label}`: {e}"),
                None => format!("model: {e}"),
            })
        })?;
        Ok(Self::new(name, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExpertPlacement;
    use crate::server::balancer::routing::Policy;

    fn spec(name: &str) -> ModelSpec {
        ModelSpec::new(
            Some(name.to_string()),
            LocalModelConfig {
                source: "qwen3-0.6b-q8_0".parse().unwrap(),
                ..Default::default()
            },
        )
    }

    fn serve(text: &str) -> Result<ServerConfig, ServerError> {
        CucaConfig::from_kdl_str(text)?.into_serve()
    }

    #[test]
    fn the_shipped_example_parses() {
        let config =
            CucaConfig::from_kdl_str(include_str!("../../examples/configs/cuca.kdl")).unwrap();
        let balance = config.balance.clone().unwrap();
        let serve = config.into_serve().unwrap();
        serve.validate().unwrap();
        let names: Vec<&str> = serve.models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["chat", "gemma-4-E4B-it-Q4_0", "moe"]);
        assert_eq!(serve.api_key.as_deref(), Some("change-me"));
        assert_eq!(serve.models[0].config.batching.max_sequences, 8);
        assert_eq!(serve.models[1].config.gpu_layers, Some(20));
        let ExpertPlacement::Tiered(tiers) = &serve.models[2].config.experts else {
            panic!("the moe model has tiered experts");
        };
        assert_eq!((tiers.device_bytes, tiers.ram_bytes), (4 << 30, 12 << 30));
        assert_eq!(balance.policy, Policy::PrefixAffinity);
        assert_eq!(balance.health_interval, Duration::from_secs(1));
        assert_eq!(balance.upstream_api_key.as_deref(), Some("change-me"));
        assert_eq!(balance.resolve_upstreams().unwrap().len(), 2);
    }

    #[test]
    fn a_model_is_named_by_its_argument_else_by_its_source() {
        let config = serve(
            r#"
serve {
    model "small" { source "qwen3-0.6b-q8_0"; }
    model { source "qwen3-0.6b-q8_0"; }
    model { source "org/repo:abcdef0123456789:model-Q4_0.gguf"; }
    model { source "org/Repo:main"; }
    model { source dir="/ckpt/Qwen3-0.6B"; }
}
"#,
        )
        .unwrap();
        let names: Vec<&str> = config.models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "small",
                "qwen3-0.6b-q8_0",
                "model-Q4_0",
                "Repo",
                "Qwen3-0.6B"
            ]
        );
        assert_eq!(
            config.models[2].source_label(),
            "org/repo:abcdef01:model-Q4_0.gguf"
        );
        // One `model` node is a list of one.
        let one = serve(r#"serve { model "m" { source "qwen3-0.6b-q8_0"; }; }"#).unwrap();
        assert_eq!(one.models.len(), 1);
    }

    #[test]
    fn an_unknown_key_is_rejected_naming_the_model() {
        let err = serve(r#"serve { model "chat" { source "qwen3-0.6b-q8_0"; context 4; }; }"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("model `chat`"), "{err}");
        assert!(err.contains("unknown field `context`"), "{err}");
        let err = serve(r#"serve { model "qwen3-0.6b-q8_0"; }"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("model \"name\" { source"), "{err}");
        let err = CucaConfig::from_kdl_str("serv { }")
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown field `serv`"), "{err}");
        let err = CucaConfig::from_kdl_str("")
            .unwrap()
            .into_serve()
            .unwrap_err();
        assert!(err.to_string().contains("no `serve` section"), "{err}");
    }

    #[test]
    fn the_device_label_joins_device_and_shards() {
        let config = serve(
            r#"serve { model "m" { source "qwen3-0.6b-q8_0"; device "vulkan:0"; shards "vulkan:1"; }; }"#,
        )
        .unwrap();
        assert_eq!(config.models[0].device_label(), "vulkan:0+vulkan:1");
        assert_eq!(spec("cpu").device_label(), "cpu");
    }

    #[test]
    fn validate_rejects_no_models_and_duplicate_names() {
        let listen = ServerConfig::DEFAULT_LISTEN;
        assert!(ServerConfig::new(listen, vec![]).validate().is_err());
        let one = spec("qwen3-0.6b-q8_0");
        let config = ServerConfig::new(listen, vec![one.clone(), one]);
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("qwen3-0.6b-q8_0"), "{err}");
    }
}
