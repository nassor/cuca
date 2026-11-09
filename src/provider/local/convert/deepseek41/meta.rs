//! The GGUF metadata of a DeepSeek-V4.1 conversion: the `deepseek41.*` keys of
//! llama.cpp's `deepseek41` layout, including the Engram hash constants
//! (`ple.*`, the compressed token map) the runtime reads instead of
//! recomputing, and the `dflash.*` keys of the DSpark drafter file.

use super::super::cfg_err;
use super::super::engram;
use super::hparams::{Config, EngramConfig};
use crate::error::CucaError;
use crate::provider::local::WeightQuant;
use crate::provider::local::gguf::Value;

/// llama.cpp's `LlamaFileType` for a file whose routed experts are MXFP4.
const FILE_TYPE_MXFP4_MOE: u32 = 38;
/// `LLAMA_EXPERT_GATING_FUNC_SQRTSOFTPLUS`.
const GATING_SQRTSOFTPLUS: u32 = 4;

/// Ordered key/value pairs under one architecture prefix.
struct Kv {
    prefix: &'static str,
    items: Vec<(String, Value)>,
}

impl Kv {
    fn new(prefix: &'static str) -> Self {
        Self {
            prefix,
            items: Vec::new(),
        }
    }

    fn raw(&mut self, key: &str, value: Value) {
        match self.items.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => self.items.push((key.to_string(), value)),
        }
    }

    fn put(&mut self, key: &str, value: Value) {
        self.raw(&format!("{}.{key}", self.prefix), value);
    }

    fn u32(&mut self, key: &str, v: usize) -> Result<(), CucaError> {
        let v = u32::try_from(v).map_err(|_| cfg_err(format!("{key}: {v} does not fit u32")))?;
        self.put(key, Value::U32(v));
        Ok(())
    }

    fn f32(&mut self, key: &str, v: f64) {
        self.put(key, Value::F32(v as f32));
    }

    fn array(&mut self, key: &str, values: impl IntoIterator<Item = Value>) {
        self.put(key, Value::Array(values.into_iter().collect()));
    }
}

/// Keys the backbone and the drafter share: the attention, the routed-expert
/// and the hyper-connection hyperparameters.
fn common(kv: &mut Kv, cfg: &Config, n_block: usize) -> Result<(), CucaError> {
    kv.u32("context_length", cfg.n_ctx)?;
    kv.u32("embedding_length", cfg.n_embd)?;
    kv.u32("attention.head_count", cfg.n_head)?;
    kv.u32("attention.head_count_kv", cfg.n_head_kv)?;
    kv.u32("attention.key_length", cfg.head_dim)?;
    kv.u32("attention.value_length", cfg.head_dim)?;
    kv.u32("rope.dimension_count", cfg.rope_dim)?;
    kv.f32("rope.freq_base", cfg.rope_theta);
    kv.f32("attention.layer_norm_rms_epsilon", cfg.eps);
    kv.u32("attention.q_lora_rank", cfg.q_lora)?;
    kv.u32("attention.sliding_window", cfg.window)?;
    kv.u32("attention.output_group_count", cfg.o_groups)?;
    kv.u32("attention.output_lora_rank", cfg.o_lora)?;
    kv.u32("expert_shared_count", cfg.n_shared)?;
    kv.u32("expert_feed_forward_length", cfg.moe_ff)?;
    kv.u32("expert_gating_func", GATING_SQRTSOFTPLUS as usize)?;
    kv.f32("expert_weights_scale", cfg.routed_scale);
    kv.put("expert_weights_norm", Value::Bool(cfg.norm_topk));
    let clamp = |_| Value::F32(cfg.swiglu_limit as f32);
    kv.array("swiglu_clamp_exp", (0..n_block).map(clamp));
    kv.array("swiglu_clamp_shexp", (0..n_block).map(clamp));
    kv.u32("hyper_connection.count", cfg.hc_mult)?;
    kv.u32("hyper_connection.sinkhorn_iterations", cfg.hc_iters)?;
    kv.f32("hyper_connection.epsilon", cfg.hc_eps);
    Ok(())
}

fn u32s(values: &[u32]) -> impl Iterator<Item = Value> + '_ {
    values.iter().map(|&v| Value::U32(v))
}

/// The Engram keys: bucket layout (`ple.*`) and the compressed token map.
fn engram_keys(
    kv: &mut Kv,
    e: &EngramConfig,
    tokenizer_json: &serde_json::Value,
    n_vocab: usize,
) -> Result<(), CucaError> {
    let primes = engram::primes(e.layers.len(), e.ngram, e.heads, e.vocab);
    for (layer, (per, &rows)) in primes.iter().zip(&e.rows).enumerate() {
        let sum: u64 = per.iter().flatten().sum();
        if sum != rows {
            return Err(cfg_err(format!(
                "Engram layer {}: its {} hash buckets add up to {sum} rows, config.json says {rows}",
                e.layers[layer],
                per.iter().map(Vec::len).sum::<usize>()
            )));
        }
    }
    let (token_map, classes) = engram::compressed_token_map(tokenizer_json, n_vocab)?;
    if classes != e.compressed_vocab {
        return Err(cfg_err(format!(
            "the compressed token map has {classes} classes, config.json says {}",
            e.compressed_vocab
        )));
    }
    let layer_ids: Vec<u32> = e.layers.iter().map(|&l| l as u32).collect();
    let multipliers = engram::multipliers(
        &e.layers.iter().map(|&l| l as u64).collect::<Vec<_>>(),
        e.ngram,
        classes as u64,
    );
    kv.array("ple.layers", u32s(&layer_ids));
    kv.u32("ple.ngram_size", e.ngram)?;
    kv.u32("ple.heads_per_ngram", e.heads)?;
    kv.u32("embedding_length_per_layer_input", e.dim)?;
    kv.array(
        "ple.layer_multipliers",
        multipliers.iter().flatten().map(|&m| Value::U64(m)),
    );
    kv.array(
        "ple.head_offsets",
        engram::head_offsets(&primes).into_iter().map(Value::U64),
    );
    kv.array(
        "ple.head_vocab_sizes",
        primes.iter().flatten().flatten().map(|&p| Value::U64(p)),
    );
    kv.put("engram.pad_token_id", Value::U32(e.pad_token));
    kv.put(
        "engram.token_map",
        Value::Array(token_map.into_iter().map(Value::I32).collect()),
    );
    Ok(())
}

/// The metadata of the backbone file: `general.*`, `deepseek41.*` and, from
/// `tokenizer`, the vocabulary keys and the chat template.
///
/// # Errors
///
/// A hyperparameter that does not fit its GGUF type, an Engram layout that
/// disagrees with the checkpoint's row counts, or a token map the wrong size.
pub(super) fn main_kv(
    cfg: &Config,
    name: &str,
    quant: WeightQuant,
    tokenizer_json: Option<&serde_json::Value>,
    tokenizer: Vec<(String, Value)>,
) -> Result<Vec<(String, Value)>, CucaError> {
    let mut kv = Kv::new("deepseek41");
    kv.raw("general.architecture", Value::String("deepseek41".into()));
    kv.raw("general.type", Value::String("model".into()));
    kv.raw("general.name", Value::String(name.to_string()));
    kv.u32("block_count", cfg.n_layer)?;
    common(&mut kv, cfg, cfg.n_layer)?;
    kv.put("rope.scaling.type", Value::String("yarn".into()));
    kv.f32("rope.scaling.factor", cfg.yarn.factor);
    kv.u32(
        "rope.scaling.original_context_length",
        cfg.yarn.original_context,
    )?;
    kv.f32("rope.scaling.yarn_beta_fast", cfg.yarn.beta_fast);
    kv.f32("rope.scaling.yarn_beta_slow", cfg.yarn.beta_slow);
    kv.array(
        "attention.compress_ratios",
        cfg.compress_ratios.iter().map(|&r| Value::I32(r)),
    );
    kv.f32("attention.compress_rope_freq_base", cfg.compress_theta);
    kv.array("attention.kv_source_layers", u32s(&cfg.kv_sources));
    kv.array("attention.indexer.source_layers", u32s(&cfg.index_sources));
    kv.u32("attention.indexer.head_count", cfg.index_heads)?;
    kv.u32("attention.indexer.key_length", cfg.index_dim)?;
    kv.u32("attention.indexer.top_k", cfg.index_topk)?;
    if let Some(c) = &cfg.candidate {
        kv.put(
            "attention.indexer.candidate_source_layer",
            Value::I32(c.layer as i32),
        );
        kv.u32("attention.indexer.candidate_top_k_blocks", c.blocks)?;
        kv.u32("attention.indexer.candidate_block_size", c.block_size)?;
    }
    kv.u32("expert_count", cfg.n_expert)?;
    kv.u32("expert_used_count", cfg.n_used)?;
    kv.u32("hash_layer_count", 0)?;
    if let Some(e) = &cfg.engram {
        let tokenizer_json = tokenizer_json
            .ok_or_else(|| cfg_err("the Engram token map needs the checkpoint's tokenizer.json"))?;
        engram_keys(&mut kv, e, tokenizer_json, cfg.n_vocab)?;
    }
    kv.raw("general.quantization_version", Value::U32(2));
    kv.raw("general.file_type", Value::U32(file_type(quant)));
    let mut items = kv.items;
    items.extend(tokenizer);
    Ok(items)
}

/// The metadata of the DSpark drafter file (`general.architecture dflash`):
/// the stages' hyperparameters and `tokenizer.ggml.mask_token_id`, plus the
/// target's vocabulary.
///
/// # Errors
///
/// A checkpoint without the `dspark_*` configuration.
pub(super) fn drafter_kv(
    cfg: &Config,
    name: &str,
    quant: WeightQuant,
    tokenizer: Vec<(String, Value)>,
) -> Result<Vec<(String, Value)>, CucaError> {
    let Some(d) = &cfg.dspark else {
        return Err(cfg_err(
            "the checkpoint has no DSpark drafter (num_nextn_predict_layers and dspark_* in config.json)",
        ));
    };
    let n = cfg.n_mtp;
    let mut kv = Kv::new("dflash");
    kv.raw("general.architecture", Value::String("dflash".into()));
    kv.raw("general.type", Value::String("model".into()));
    kv.raw("general.name", Value::String(format!("{name}-dspark")));
    kv.u32("block_count", n)?;
    common(&mut kv, cfg, n)?;
    kv.u32("feed_forward_length", cfg.moe_ff)?;
    kv.u32("expert_count", d.n_expert)?;
    kv.u32("expert_used_count", d.n_used)?;
    kv.array("attention.compress_ratios", (0..n).map(|_| Value::I32(0)));
    kv.put("hyper_connection.single_pass", Value::Bool(true));
    kv.u32("block_size", d.block_size)?;
    kv.array("target_layers", d.targets.iter().map(|&t| Value::I32(t)));
    kv.put("sample_from_anchor", Value::Bool(true));
    kv.put("has_confidence_head", Value::Bool(true));
    kv.raw("tokenizer.ggml.mask_token_id", Value::U32(d.noise_token));
    kv.raw("general.quantization_version", Value::U32(2));
    kv.raw("general.file_type", Value::U32(file_type(quant)));
    let mut items = kv.items;
    for (key, value) in tokenizer {
        if !items.iter().any(|(k, _)| *k == key) {
            items.push((key, value));
        }
    }
    Ok(items)
}

/// `general.file_type`: MXFP4 experts at every precision but `f16`.
fn file_type(quant: WeightQuant) -> u32 {
    match quant {
        WeightQuant::F16 => 1,
        _ => FILE_TYPE_MXFP4_MOE,
    }
}
