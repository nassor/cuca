//! A Hugging Face checkpoint on disk: the files the converter reads, and the
//! download of those files for [`super::super::ModelSource::Checkpoint`].

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{cfg_err, eagle};
use crate::error::CucaError;
use crate::provider::local::DownloadProgressCallback;
use crate::provider::local::download;

/// Optional files of a checkpoint, fetched when the repository has them.
const OPTIONAL_FILES: [&str; 8] = [
    "generation_config.json",
    "tekken.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
];
/// The shard index of a sharded checkpoint.
const INDEX_FILE: &str = "model.safetensors.index.json";
/// The weights of an unsharded checkpoint.
const SINGLE_FILE: &str = "model.safetensors";

/// The files of one checkpoint directory.
#[derive(Debug)]
pub(crate) struct CheckpointDir {
    pub(crate) dir: PathBuf,
    /// `config.json`.
    pub(crate) config: Value,
    pub(crate) generation_config: Option<Value>,
    pub(crate) tokenizer_json: Option<Value>,
    pub(crate) tokenizer_config: Option<Value>,
    /// `tokenizer.model` (SentencePiece), when present.
    pub(crate) tokenizer_model: Option<PathBuf>,
    /// `added_tokens.json`: SentencePiece tokens added after training.
    pub(crate) added_tokens: Option<Value>,
    /// The text of `chat_template.jinja`, when present.
    pub(crate) chat_template_file: Option<String>,
    /// The safetensors shards, sorted by name.
    pub(crate) shards: Vec<PathBuf>,
}

fn read_json(path: &Path) -> Result<Value, CucaError> {
    let text = std::fs::read_to_string(path).map_err(|e| CucaError::Io {
        message: format!("cannot read `{}`: {e}", path.display()),
    })?;
    serde_json::from_str(&text).map_err(|e| cfg_err(format!("`{}`: {e}", path.display())))
}

fn read_json_opt(path: &Path) -> Result<Option<Value>, CucaError> {
    path.is_file().then(|| read_json(path)).transpose()
}

/// `config.json`, or, for the Mistral-format EAGLE drafter (which ships
/// `params.json` and no `config.json`), the config [`eagle::config_from_params`]
/// derives. A directory with neither fails reading `config.json`.
fn read_config(dir: &Path) -> Result<Value, CucaError> {
    let config = dir.join("config.json");
    let params = dir.join(eagle::PARAMS_FILE);
    if !config.is_file() && params.is_file() {
        return eagle::config_from_params(&read_json(&params)?);
    }
    read_json(&config)
}

/// The shard files named by a `model.safetensors.index.json`, sorted.
fn index_shards(dir: &Path, index: &Value) -> Result<Vec<PathBuf>, CucaError> {
    let map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| cfg_err(format!("`{INDEX_FILE}` has no `weight_map`")))?;
    let names: BTreeSet<&str> = map.values().filter_map(Value::as_str).collect();
    if names.is_empty() {
        return Err(cfg_err(format!("`{INDEX_FILE}` lists no shard")));
    }
    names
        .into_iter()
        .map(|n| {
            if n.contains(['/', '\\']) || n.starts_with('.') {
                Err(cfg_err(format!(
                    "`{INDEX_FILE}` names the shard `{n}`; expected a plain file name"
                )))
            } else {
                Ok(dir.join(n))
            }
        })
        .collect()
}

impl CheckpointDir {
    /// Read the checkpoint in `dir`: `config.json` is required; the shards
    /// are the `weight_map` values of `model.safetensors.index.json` when
    /// present, else `model.safetensors`, else every `*.safetensors` file.
    pub(crate) fn open(dir: &Path) -> Result<Self, CucaError> {
        let config = read_config(dir)?;
        let shards = if let Some(index) = read_json_opt(&dir.join(INDEX_FILE))? {
            index_shards(dir, &index)?
        } else if dir.join(SINGLE_FILE).is_file() {
            vec![dir.join(SINGLE_FILE)]
        } else {
            let mut found: Vec<PathBuf> = std::fs::read_dir(dir)?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            found.sort();
            found
        };
        if shards.is_empty() {
            return Err(cfg_err(format!(
                "checkpoint `{}` has no {SINGLE_FILE} or {INDEX_FILE}",
                dir.display()
            )));
        }
        for shard in &shards {
            if !shard.is_file() {
                return Err(CucaError::Io {
                    message: format!("checkpoint shard `{}` does not exist", shard.display()),
                });
            }
        }
        let template = dir.join("chat_template.jinja");
        let model = dir.join("tokenizer.model");
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            generation_config: read_json_opt(&dir.join("generation_config.json"))?,
            tokenizer_json: read_json_opt(&dir.join("tokenizer.json"))?,
            tokenizer_config: read_json_opt(&dir.join("tokenizer_config.json"))?,
            tokenizer_model: model.is_file().then_some(model),
            added_tokens: read_json_opt(&dir.join("added_tokens.json"))?,
            chat_template_file: if template.is_file() {
                Some(std::fs::read_to_string(&template)?)
            } else {
                // Older checkpoints keep it as `{"chat_template": "..."}`.
                read_json_opt(&dir.join("chat_template.json"))?.and_then(|j| {
                    j.get("chat_template")
                        .and_then(|t| t.as_str().map(String::from))
                })
            },
            shards,
        })
    }
}

/// A repository revision being downloaded into its cache directory.
struct Source<'a> {
    client: &'a reqwest::Client,
    endpoint: &'a str,
    repo: &'a str,
    revision: &'a str,
    cache_dir: &'a Path,
    token: Option<&'a str>,
    progress: Option<&'a DownloadProgressCallback>,
}

impl Source<'_> {
    /// Download `file` (no checksum); `Ok(false)` when the repository has no
    /// such file.
    async fn optional(&self, file: &str) -> Result<bool, CucaError> {
        download::ensure_downloaded_optional(
            self.client,
            self.endpoint,
            self.repo,
            self.revision,
            file,
            None,
            &download::cached_path(self.cache_dir, self.repo, self.revision, file)?,
            self.token,
            self.progress,
        )
        .await
    }

    /// Download `file` (no checksum); a missing file is an error.
    async fn required(&self, file: &str) -> Result<(), CucaError> {
        download::ensure_downloaded(
            self.client,
            self.endpoint,
            self.repo,
            self.revision,
            file,
            None,
            &download::cached_path(self.cache_dir, self.repo, self.revision, file)?,
            self.token,
            self.progress,
        )
        .await
    }
}

/// Download the files of `repo` at `revision` into the cache directory of the
/// checkpoint and open them.
///
/// `config.json` comes first: an architecture without a converter fails
/// before any weight is downloaded (the Mistral-format EAGLE drafter has
/// `params.json` in its place). Missing optional files are skipped.
pub(crate) async fn fetch(
    client: &reqwest::Client,
    endpoint: &str,
    repo: &str,
    revision: &str,
    cache_dir: &Path,
    token: Option<&str>,
    progress: Option<&DownloadProgressCallback>,
) -> Result<CheckpointDir, CucaError> {
    let source = Source {
        client,
        endpoint,
        repo,
        revision,
        cache_dir,
        token,
        progress,
    };
    let dir = download::cached_path(cache_dir, repo, revision, "config.json")?
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| cfg_err("checkpoint path has no parent"))?;
    let has_config = source.optional("config.json").await?;
    if !has_config && !source.optional(eagle::PARAMS_FILE).await? {
        return Err(cfg_err(format!(
            "checkpoint `{repo}` at `{revision}` has neither config.json nor {}",
            eagle::PARAMS_FILE
        )));
    }
    super::dflash::check_architecture(&read_config(&dir)?)?;

    for file in OPTIONAL_FILES.iter().chain(&[INDEX_FILE]) {
        source.optional(file).await?;
    }
    let shards: Vec<String> = match read_json_opt(&dir.join(INDEX_FILE))? {
        Some(index) => index_shards(&dir, &index)?
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_string))
            .collect(),
        None if has_config => vec![SINGLE_FILE.to_string()],
        None => vec![eagle::WEIGHTS_FILE.to_string()],
    };
    for shard in &shards {
        source.required(shard).await?;
    }
    CheckpointDir::open(&dir)
}

#[cfg(test)]
mod tests {
    use super::super::arch::ConvArch;
    use super::*;
    use crate::provider::local::convert::safetensors::tests::temp_dir;

    #[test]
    fn opens_single_and_sharded_checkpoints() {
        let dir = temp_dir("hf");
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"]}"#,
        )
        .unwrap();
        assert!(CheckpointDir::open(&dir).is_err(), "no weights");
        std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
        let c = CheckpointDir::open(&dir).unwrap();
        assert_eq!(c.shards, [dir.join("model.safetensors")]);
        assert!(c.tokenizer_json.is_none() && c.tokenizer_model.is_none());

        std::fs::write(
            dir.join(INDEX_FILE),
            r#"{"weight_map":{"a":"s-2.safetensors","b":"s-1.safetensors","c":"s-2.safetensors"}}"#,
        )
        .unwrap();
        assert!(
            CheckpointDir::open(&dir).is_err(),
            "listed shards are missing"
        );
        std::fs::write(dir.join("s-1.safetensors"), b"x").unwrap();
        std::fs::write(dir.join("s-2.safetensors"), b"x").unwrap();
        std::fs::write(dir.join("chat_template.jinja"), "{{ x }}").unwrap();
        let c = CheckpointDir::open(&dir).unwrap();
        assert_eq!(
            c.shards,
            [dir.join("s-1.safetensors"), dir.join("s-2.safetensors")]
        );
        assert_eq!(c.chat_template_file.as_deref(), Some("{{ x }}"));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The Mistral-format EAGLE drafter has `params.json` and
    /// `consolidated.safetensors` and no `config.json`: its config is derived.
    #[test]
    fn opens_a_params_json_drafter_without_config_json() {
        let dir = temp_dir("hf-eagle");
        assert!(CheckpointDir::open(&dir).is_err(), "nothing there");
        std::fs::write(dir.join("consolidated.safetensors"), b"x").unwrap();
        std::fs::write(dir.join("params.json"), r#"{"moe": {"num_experts": 4}}"#).unwrap();
        let err = CheckpointDir::open(&dir).unwrap_err().to_string();
        assert!(err.contains("mixture-of-experts"), "{err}");
        std::fs::write(
            dir.join("params.json"),
            r#"{"dim": 64, "n_layers": 2, "hidden_dim": 96, "n_heads": 4, "n_kv_heads": 4,
                "norm_eps": 1e-6, "vocab_size": 16, "q_lora_rank": 32, "kv_lora_rank": 16,
                "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 16,
                "yarn": {"original_max_position_embeddings": 8, "factor": 2, "beta": 32, "alpha": 1},
                "llama_4_scaling": {"original_max_position_embeddings": 8, "beta": 0.3},
                "moe": null}"#,
        )
        .unwrap();
        let c = CheckpointDir::open(&dir).unwrap();
        assert_eq!(c.shards, [dir.join("consolidated.safetensors")]);
        assert_eq!(
            ConvArch::from_config(&c.config).unwrap(),
            ConvArch::Mistral4Eagle
        );
        // A `config.json` wins over `params.json`.
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"]}"#,
        )
        .unwrap();
        let c = CheckpointDir::open(&dir).unwrap();
        assert_eq!(ConvArch::from_config(&c.config).unwrap(), ConvArch::Llama);
        let _ = std::fs::remove_dir_all(dir);
    }
}
