//! The hyperparameters of a DeepSeek-V4.1 `config.json` (`text_config` merged
//! over the top level, as [`Hp`] does), validated the way llama.cpp's
//! `deepseek41.cpp` loader validates them, so a checkpoint the engine could
//! not run fails at conversion naming the rule.

use serde_json::Value as Json;

use super::super::cfg_err;
use super::super::config::Hp;
use crate::error::CucaError;

/// YaRN rope scaling, which applies to the compressed-attention layers only
/// (the others use plain RoPE at `rope_theta`).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Yarn {
    pub(super) factor: f64,
    pub(super) original_context: usize,
    pub(super) beta_fast: f64,
    pub(super) beta_slow: f64,
}

/// The two-level candidate selection of the last index-source run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Candidate {
    pub(super) layer: usize,
    pub(super) blocks: usize,
    pub(super) block_size: usize,
}

/// The Engram n-gram tables.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct EngramConfig {
    /// Layers that own an Engram.
    pub(super) layers: Vec<usize>,
    /// Rows of each layer's table.
    pub(super) rows: Vec<u64>,
    /// Largest n-gram order (orders 2 to this one are hashed).
    pub(super) ngram: usize,
    /// Smallest bucket modulus minus one, per n-gram order.
    pub(super) vocab: u64,
    /// Hash heads per n-gram order.
    pub(super) heads: usize,
    /// Width of a table row.
    pub(super) dim: usize,
    pub(super) pad_token: u32,
    /// Classes of the compressed token map.
    pub(super) compressed_vocab: usize,
}

/// The DSpark drafter stages.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DsparkConfig {
    pub(super) block_size: usize,
    pub(super) noise_token: u32,
    /// Layers whose inputs feed the drafter.
    pub(super) targets: Vec<i32>,
    pub(super) markov_rank: usize,
    pub(super) n_expert: usize,
    pub(super) n_used: usize,
}

/// Everything the converter reads from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Config {
    pub(super) n_layer: usize,
    pub(super) n_embd: usize,
    pub(super) n_vocab: usize,
    pub(super) n_head: usize,
    pub(super) n_head_kv: usize,
    pub(super) head_dim: usize,
    pub(super) rope_dim: usize,
    pub(super) rope_theta: f64,
    pub(super) yarn: Yarn,
    pub(super) n_ctx: usize,
    pub(super) eps: f64,
    pub(super) q_lora: usize,
    pub(super) o_lora: usize,
    pub(super) o_groups: usize,
    pub(super) window: usize,
    /// One entry per backbone layer, then one per drafter stage.
    pub(super) compress_ratios: Vec<i32>,
    pub(super) compress_theta: f64,
    pub(super) kv_sources: Vec<u32>,
    pub(super) index_sources: Vec<u32>,
    pub(super) index_heads: usize,
    pub(super) index_dim: usize,
    pub(super) index_topk: usize,
    pub(super) candidate: Option<Candidate>,
    pub(super) n_expert: usize,
    pub(super) n_shared: usize,
    pub(super) n_used: usize,
    pub(super) moe_ff: usize,
    pub(super) routed_scale: f64,
    pub(super) norm_topk: bool,
    pub(super) swiglu_limit: f64,
    pub(super) hc_mult: usize,
    pub(super) hc_iters: usize,
    pub(super) hc_eps: f64,
    pub(super) engram: Option<EngramConfig>,
    /// Drafter stages the checkpoint holds (`mtp.N.*`).
    pub(super) n_mtp: usize,
    pub(super) dspark: Option<DsparkConfig>,
}

fn float(hp: &Hp, key: &str) -> Result<f64, CucaError> {
    hp.float(&[key])
        .ok_or_else(|| cfg_err(format!("config.json has no number `{key}`")))
}

fn uint(hp: &Hp, key: &str) -> Result<usize, CucaError> {
    hp.req_uint(&[key])
}

fn uints<T: TryFrom<u64>>(hp: &Hp, key: &str) -> Result<Vec<T>, CucaError> {
    hp.get(key)
        .and_then(Json::as_array)
        .and_then(|a| {
            a.iter()
                .map(|v| v.as_u64().and_then(|n| T::try_from(n).ok()))
                .collect()
        })
        .ok_or_else(|| cfg_err(format!("config.json `{key}` is not a list of integers")))
}

impl Config {
    /// Read and validate `hp`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] naming the missing key or the violated rule.
    pub(super) fn read(hp: &Hp) -> Result<Self, CucaError> {
        let quant = hp.get("quantization_config");
        let block = quant
            .and_then(|q| q.get("weight_block_size"))
            .and_then(Json::as_array)
            .map(|b| b.iter().filter_map(Json::as_u64).collect::<Vec<_>>());
        if let Some(block) = block
            && block != [32, 32]
        {
            return Err(cfg_err(format!(
                "FP8 weight_block_size {block:?} is not supported (DeepSeek-V4.1 uses [32, 32]; V4's 128x128 grid would rescale silently)"
            )));
        }
        let scale_fmt = quant
            .and_then(|q| q.get("scale_fmt"))
            .and_then(Json::as_str)
            .unwrap_or("ue8m0");
        if scale_fmt != "ue8m0" {
            return Err(cfg_err(format!(
                "FP8 scale format `{scale_fmt}` is not supported (only ue8m0)"
            )));
        }
        let scoring = hp.get("scoring_func").and_then(Json::as_str);
        if scoring != Some("sqrtsoftplus") {
            return Err(cfg_err(format!(
                "scoring_func {scoring:?} is not supported (only `sqrtsoftplus`)"
            )));
        }
        if hp.float(&["gate_temp"]).is_some_and(|t| t != 1.0) {
            return Err(cfg_err("gate_temp != 1 is not supported"));
        }
        if hp.uint(&["num_hash_layers"]).is_some_and(|n| n != 0) {
            return Err(cfg_err(
                "hash-routed layers (num_hash_layers > 0) are DeepSeek-V4, not V4.1",
            ));
        }

        let rope = hp.rope_all();
        let rope_type = rope
            .get("rope_type")
            .or_else(|| rope.get("type"))
            .and_then(Json::as_str);
        if rope_type != Some("yarn") {
            return Err(cfg_err(format!(
                "rope scaling {rope_type:?} is not supported (DeepSeek-V4.1 uses yarn)"
            )));
        }
        let rope_num = |key: &str| {
            rope.get(key)
                .and_then(Json::as_f64)
                .ok_or_else(|| cfg_err(format!("rope scaling has no `{key}`")))
        };
        let yarn = Yarn {
            factor: rope_num("factor")?,
            original_context: rope_num("original_max_position_embeddings")? as usize,
            beta_fast: rope_num("beta_fast")?,
            beta_slow: rope_num("beta_slow")?,
        };

        let n_layer = hp.n_layer()?;
        let n_mtp = hp.uint(&["num_nextn_predict_layers"]).unwrap_or(0);
        let compress_ratios: Vec<i32> = uints::<u32>(hp, "compress_ratios")?
            .into_iter()
            .map(|r| r as i32)
            .collect();
        if compress_ratios.len() < n_layer {
            return Err(cfg_err(format!(
                "compress_ratios has {} entries for {n_layer} layers",
                compress_ratios.len()
            )));
        }
        let candidate = hp
            .get("candidate_source_layer_id")
            .and_then(Json::as_i64)
            .filter(|&l| l >= 0)
            .map(|layer| {
                Ok::<_, CucaError>(Candidate {
                    layer: layer as usize,
                    blocks: uint(hp, "candidate_topk_blocks")?,
                    block_size: uint(hp, "candidate_block_size")?,
                })
            })
            .transpose()?;

        let engram = match hp.get("engram_layer_ids").and_then(Json::as_array) {
            Some(ids) if !ids.is_empty() => {
                let layers: Vec<usize> = uints(hp, "engram_layer_ids")?;
                let rows: Vec<u64> = uints(hp, "engram_num_embeddings")?;
                if rows.len() != layers.len() {
                    return Err(cfg_err(
                        "engram_num_embeddings does not list one row count per engram layer",
                    ));
                }
                Some(EngramConfig {
                    layers,
                    rows,
                    ngram: uint(hp, "engram_max_ngram_size")?,
                    vocab: uint(hp, "engram_vocab_size")? as u64,
                    heads: uint(hp, "engram_n_heads")?,
                    dim: uint(hp, "engram_head_dim")?,
                    pad_token: uint(hp, "engram_pad_token_id")? as u32,
                    compressed_vocab: uint(hp, "engram_compressed_vocab_size")?,
                })
            }
            _ => None,
        };
        let dspark = (n_mtp > 0 && hp.get("dspark_block_size").is_some())
            .then(|| {
                Ok::<_, CucaError>(DsparkConfig {
                    block_size: uint(hp, "dspark_block_size")?,
                    noise_token: uint(hp, "dspark_noise_token_id")? as u32,
                    targets: uints::<u32>(hp, "dspark_target_layer_ids")?
                        .into_iter()
                        .map(|t| t as i32)
                        .collect(),
                    markov_rank: uint(hp, "dspark_markov_rank")?,
                    n_expert: uint(hp, "dspark_n_routed_experts")?,
                    n_used: uint(hp, "dspark_num_experts_per_tok")?,
                })
            })
            .transpose()?;

        let config = Self {
            n_layer,
            n_embd: hp.n_embd()?,
            n_vocab: uint(hp, "vocab_size")?,
            n_head: hp.n_head()?,
            n_head_kv: hp.uint(&["num_key_value_heads"]).unwrap_or(1),
            head_dim: hp.head_dim()?,
            rope_dim: uint(hp, "qk_rope_head_dim")?,
            rope_theta: float(hp, "rope_theta")?,
            yarn,
            n_ctx: uint(hp, "max_position_embeddings")?,
            eps: float(hp, "rms_norm_eps")?,
            q_lora: uint(hp, "q_lora_rank")?,
            o_lora: uint(hp, "o_lora_rank")?,
            o_groups: uint(hp, "o_groups")?,
            window: uint(hp, "sliding_window")?,
            compress_ratios,
            compress_theta: float(hp, "compress_rope_theta")?,
            kv_sources: uints(hp, "kv_source_layer_ids")?,
            index_sources: uints(hp, "index_source_layer_ids")?,
            index_heads: uint(hp, "index_n_heads")?,
            index_dim: uint(hp, "index_head_dim")?,
            index_topk: uint(hp, "index_topk")?,
            candidate,
            n_expert: uint(hp, "n_routed_experts")?,
            n_shared: uint(hp, "n_shared_experts")?,
            n_used: uint(hp, "num_experts_per_tok")?,
            moe_ff: uint(hp, "moe_intermediate_size")?,
            routed_scale: float(hp, "routed_scaling_factor")?,
            norm_topk: hp
                .get("norm_topk_prob")
                .and_then(Json::as_bool)
                .unwrap_or(true),
            swiglu_limit: float(hp, "swiglu_limit")?,
            hc_mult: uint(hp, "hc_mult")?,
            hc_iters: uint(hp, "hc_sinkhorn_iters")?,
            hc_eps: float(hp, "hc_eps")?,
            engram,
            n_mtp,
            dspark,
        };
        config.validate()?;
        Ok(config)
    }

    /// llama.cpp's loader rules (`deepseek41.cpp`): ratios of at most 2, every
    /// compressed layer reading the latest key/value and index source before
    /// it at its own ratio, the candidate source being an index source, and the
    /// Engram tables matching their buckets.
    fn validate(&self) -> Result<(), CucaError> {
        let n = self.n_layer;
        if let Some(r) = self
            .compress_ratios
            .iter()
            .find(|&&r| !(0..=2).contains(&r))
        {
            return Err(cfg_err(format!("compress ratio {r} is not 0, 1 or 2")));
        }
        for (what, sources) in [
            ("kv_source_layer_ids", &self.kv_sources),
            ("index_source_layer_ids", &self.index_sources),
        ] {
            if let Some(l) = sources.iter().find(|&&l| l as usize >= n) {
                return Err(cfg_err(format!("{what} names layer {l} of {n}")));
            }
        }
        let latest = |sources: &[u32], il: usize| {
            sources
                .iter()
                .map(|&s| s as usize)
                .filter(|&s| s <= il)
                .max()
        };
        for (il, &ratio) in self.compress_ratios[..n].iter().enumerate() {
            if ratio == 0 {
                continue;
            }
            for (what, sources) in [
                ("key/value", &self.kv_sources),
                ("index", &self.index_sources),
            ] {
                let Some(source) = latest(sources, il) else {
                    return Err(cfg_err(format!(
                        "compressed layer {il} has no {what} source at or before it"
                    )));
                };
                if self.compress_ratios[source] != ratio {
                    return Err(cfg_err(format!(
                        "compressed layer {il} (ratio {ratio}) reads the {what} source {source}, whose ratio is {}",
                        self.compress_ratios[source]
                    )));
                }
            }
        }
        if let Some(c) = &self.candidate
            && !self.index_sources.contains(&(c.layer as u32))
        {
            return Err(cfg_err(format!(
                "candidate_source_layer_id {} is not an index source",
                c.layer
            )));
        }
        if let Some(e) = &self.engram {
            if let Some(l) = e.layers.iter().find(|&&l| l >= n) {
                return Err(cfg_err(format!("engram layer {l} of {n}")));
            }
            if !e.dim.is_multiple_of(32) {
                return Err(cfg_err(format!(
                    "engram_head_dim {} is not a multiple of 32 (one scale per 32 columns)",
                    e.dim
                )));
            }
        }
        if !self.moe_ff.is_multiple_of(32) || !self.n_embd.is_multiple_of(32) {
            return Err(cfg_err(
                "hidden_size and moe_intermediate_size must be multiples of 32 (one FP4 scale per 32 columns)",
            ));
        }
        Ok(())
    }
}
