//! Conversion of a DFlash or DSpine drafter checkpoint to the `dflash` GGUF
//! layout.
//!
//! Supported checkpoints:
//!
//! - HuggingFace/transformers `MuseGlimmerAssistantModel`
//!   (`model_type = "muse_glimmer_assistant"`): five all-sliding decoder layers
//!   with `encoder.fc.weight` and `encoder.output_norm_enc.weight`.
//! - Standard DFlash (`DFlashDraftModel`, `model_type` `"dflash"` or `"qwen3"`
//!   with `dflash_config`): trunk tensors `[model.]fc.weight`,
//!   `[model.]hidden_norm.weight`, `[model.]norm.weight`, and
//!   `[model.]layers.N.*`, full or sliding attention, optional
//!   `draft_block_causal`.
//! - Selected-prefix DSpine (`DSpineDraftModel`, `model_type = "qwen3"` with
//!   `dflash_config.selected_prefix_code_bridge`): block-causal DFlash trunk
//!   plus the post-layer code bridge (`dspine.codebook.weight`,
//!   `dspine.blk.*.early_read.weight`, `dspine.read.weight`,
//!   `dspine.gate.{weight,bias}`, `dspine.pred_gate.weight`,
//!   `dspine.prev.weight`, and `dspine.blk.*.write.weight`).
//!
//! Every tensor keeps its values and its row-major `[out, in]` shape: the
//! weights use the half-split (NEOX) rotary layout transformers does, so
//! nothing is permuted, and the norms and biases are stored as `F32`.
//! Matrices follow the requested `WeightQuant` (under mixed formats,
//! `dspine.codebook.weight` is kept at `Q8_0`, or `F16` under
//! `WeightQuant::F16`).
//!
//! `dflash.target_layers` holds llama.cpp's convention, the HF
//! `target_layer_ids` plus one: entry `i` names the layer whose *input* feeds
//! the drafter, which is the output of layer `i - 1` (transformers' ids are
//! outputs: `[1, 13, 25, 37, 49]` becomes `[2, 14, 26, 38, 50]`).

use std::collections::HashSet;
use std::path::Path;

use crate::provider::local::gguf::Value;
use serde_json::{Map, Value as Json};

use super::cfg_err;
use super::config::{Hp, file_type};
use super::hf::CheckpointDir;
use super::safetensors::Shards;
use super::tensors::{Body, Plan, Slice};
use crate::error::CucaError;
use crate::provider::local::model::cpu::dflash::{ARCH, check_metadata};
use crate::provider::local::{DownloadProgressCallback, WeightQuant};

const CLASSES: &[&str] = &[
    "MuseGlimmerAssistantModel",
    "DFlashDraftModel",
    "DSpineDraftModel",
];
const MODEL_TYPES: &[&str] = &["muse_glimmer_assistant", "dflash"];

const VERSION_SCALARS: &[&str] = &[
    "draft_block_causal_version",
    "last_write_refine_version",
    "selected_prefix_bridge.independent_write_version",
    "selected_prefix_bridge.terminal_disabled_version",
];

fn has_arch(config: &Json, name: &str) -> bool {
    config
        .get("architectures")
        .and_then(Json::as_array)
        .is_some_and(|a| a.iter().any(|c| c.as_str() == Some(name)))
}

fn json_usize(v: &Json) -> Option<usize> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .or_else(|| {
            v.as_f64()
                .filter(|f| f.fract() == 0.0 && *f >= 0.0)
                .map(|f| f as usize)
        })
}

fn json_i64(v: &Json) -> Option<i64> {
    v.as_i64()
        .or_else(|| v.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
}

fn is_truthy(v: &Json) -> bool {
    match v {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Json::String(s) => !s.is_empty(),
        Json::Array(a) => !a.is_empty(),
        Json::Object(o) => !o.is_empty(),
    }
}

fn check_unsupported_dflash_keys(config: &Json) -> Result<(), CucaError> {
    let Some(df_val) = config.get("dflash_config") else {
        return Ok(());
    };
    if df_val.is_null() {
        return Ok(());
    }
    let df = df_val
        .as_object()
        .ok_or_else(|| cfg_err("`dflash_config` in config.json must be an object"))?;
    for key in [
        "selector_rank",
        "selector_top_k",
        "projector_type",
        "attention_sink_bias",
    ] {
        if df.get(key).is_some_and(is_truthy) {
            return Err(cfg_err(format!(
                "unsupported DFlash/DSpine draft key `dflash_config.{key}`"
            )));
        }
    }
    for key in [
        "attention_value_scale",
        "selected_prefix_feature_reliability",
    ] {
        if df.get(key).is_some_and(|v| !v.is_null()) {
            return Err(cfg_err(format!(
                "unsupported DFlash/DSpine draft key `dflash_config.{key}`"
            )));
        }
    }
    if df
        .get("selected_prefix_feature_next_token_rows")
        .and_then(Json::as_bool)
        == Some(true)
    {
        return Err(cfg_err(
            "unsupported `dflash_config.selected_prefix_feature_next_token_rows` (DSpark layout, not DSpine)",
        ));
    }
    for key in [
        "conv_kernel_size",
        "conv_group_size",
        "selected_prefix_terminal_residual_rank",
    ] {
        if let Some(v) = df.get(key)
            && !v.is_null()
            && json_i64(v) != Some(0)
        {
            return Err(cfg_err(format!(
                "unsupported DFlash/DSpine draft key `dflash_config.{key}` ({v})"
            )));
        }
    }
    Ok(())
}

/// Whether `config` (a `config.json`) is a DFlash or DSpine drafter this
/// module converts.
pub(crate) fn is_dflash(config: &Json) -> bool {
    CLASSES.iter().any(|c| has_arch(config, c))
        || config
            .get("model_type")
            .and_then(Json::as_str)
            .is_some_and(|t| MODEL_TYPES.contains(&t))
        || config.get("dflash_config").is_some_and(|v| !v.is_null())
}

/// `config` validated as a convertible architecture: what
/// `ConvArch::from_config` is to the other checkpoints.
///
/// # Errors
///
/// [`CucaError::Config`] for an architecture without a converter.
pub(crate) fn check_architecture(config: &Json) -> Result<(), CucaError> {
    if is_dflash(config) {
        if let Some(arches) = config.get("architectures").and_then(Json::as_array)
            && !arches.is_empty()
            && !CLASSES.iter().any(|c| has_arch(config, c))
        {
            return Err(cfg_err(format!(
                "unsupported DFlash drafter architecture {arches:?}"
            )));
        }
        check_unsupported_dflash_keys(config)?;
        Ok(())
    } else {
        super::arch::ConvArch::from_config(config).map(|_| ())
    }
}

/// Code-bridge parameters parsed for `DSpineDraftModel`.
struct DspineCfg {
    vocab_size: usize,
    code_size: usize,
    rank: usize,
    eps: f64,
    head_topk: usize,
}

/// What `config.json` says (transformers' `MuseGlimmerAssistantConfig`,
/// `DFlashDraftModel`, or `DSpineDraftModel`).
struct Config {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_ff: usize,
    eps: f64,
    rope_theta: f64,
    max_positions: usize,
    block_size: usize,
    mask_token: usize,
    /// transformers' `target_layer_ids`.
    target_ids: Vec<usize>,
    /// Whether each layer is a sliding-window layer.
    sliding: Vec<bool>,
    window: Option<usize>,
    block_causal: bool,
    dspine: Option<DspineCfg>,
}

fn parse_layer_types(v: Option<&Json>, n_layer: usize) -> Result<Option<Vec<bool>>, CucaError> {
    match v {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Array(types)) if types.len() == n_layer => types
            .iter()
            .map(|t| match t.as_str() {
                Some("sliding_attention") => Ok(true),
                Some("full_attention") => Ok(false),
                other => Err(cfg_err(format!("layer type {other:?} is not supported"))),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(cfg_err(format!(
            "layer_types does not list {n_layer} layers"
        ))),
    }
}

fn read_dspine_cfg(
    hp: &Hp,
    df: &Map<String, Json>,
    n_layer: usize,
    block_causal: bool,
) -> Result<DspineCfg, CucaError> {
    let vocab_size = hp.req_uint(&["vocab_size"])?;
    if !block_causal {
        return Err(cfg_err(
            "DSpine requires `dflash_config.draft_block_causal = true`",
        ));
    }
    if df.get("selected_prefix_terminal").and_then(Json::as_bool) != Some(false) {
        return Err(cfg_err(
            "DSpine requires `dflash_config.selected_prefix_terminal = false`",
        ));
    }
    if df.get("selected_prefix_decode_mode").and_then(Json::as_str) != Some("serial") {
        return Err(cfg_err(
            "DSpine requires `dflash_config.selected_prefix_decode_mode = \"serial\"`",
        ));
    }
    if df
        .get("selected_prefix_feature_writes")
        .and_then(Json::as_bool)
        != Some(true)
    {
        return Err(cfg_err(
            "DSpine requires `dflash_config.selected_prefix_feature_writes = true`",
        ));
    }
    if df
        .get("selected_prefix_feature_write_mode")
        .and_then(Json::as_str)
        != Some("independent")
    {
        return Err(cfg_err(
            "DSpine requires `dflash_config.selected_prefix_feature_write_mode = \"independent\"`",
        ));
    }

    let feat_layers = df
        .get("selected_prefix_feature_layers")
        .and_then(Json::as_array)
        .ok_or_else(|| cfg_err("missing `dflash_config.selected_prefix_feature_layers`"))?;
    let parsed_layers: Vec<usize> = feat_layers
        .iter()
        .map(|v| {
            json_usize(v)
                .ok_or_else(|| cfg_err("`selected_prefix_feature_layers` holds a non-integer"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected_layers: Vec<usize> = (1..=n_layer).collect();
    if parsed_layers != expected_layers {
        return Err(cfg_err(format!(
            "`dflash_config.selected_prefix_feature_layers` must be {expected_layers:?}, got {parsed_layers:?}"
        )));
    }

    let bridge = df
        .get("selected_prefix_code_bridge")
        .and_then(Json::as_object)
        .ok_or_else(|| cfg_err("missing `dflash_config.selected_prefix_code_bridge`"))?;
    let code_size = bridge
        .get("code_size")
        .and_then(json_usize)
        .filter(|&n| n > 0)
        .ok_or_else(|| cfg_err("`selected_prefix_code_bridge.code_size` must be positive"))?;
    let rank = bridge
        .get("rank")
        .and_then(json_usize)
        .filter(|&n| n > 0)
        .ok_or_else(|| cfg_err("`selected_prefix_code_bridge.rank` must be positive"))?;
    let eps = bridge
        .get("eps")
        .and_then(Json::as_f64)
        .filter(|&e| e.is_finite() && e > 0.0)
        .ok_or_else(|| cfg_err("`selected_prefix_code_bridge.eps` must be positive"))?;

    let mut topks = Vec::new();
    if let Some(second) = hp.get("code_bridge_second_pass").filter(|v| !v.is_null()) {
        let obj = second
            .as_object()
            .ok_or_else(|| cfg_err("`code_bridge_second_pass` must be an object"))?;
        if obj.get("mode").and_then(Json::as_str) != Some("serial_table") {
            return Err(cfg_err(
                "`code_bridge_second_pass.mode` must be `\"serial_table\"`",
            ));
        }
        if obj.get("first_exact").and_then(json_usize) != Some(n_layer - 1) {
            return Err(cfg_err(format!(
                "`code_bridge_second_pass.first_exact` must be {} (the last draft layer)",
                n_layer - 1
            )));
        }
        let tk = obj
            .get("head_topk")
            .and_then(json_usize)
            .ok_or_else(|| cfg_err("missing `code_bridge_second_pass.head_topk`"))?;
        topks.push(tk);
    }
    if let Some(refine) = df
        .get("selected_prefix_last_write_refine")
        .filter(|v| !v.is_null())
    {
        let obj = refine
            .as_object()
            .ok_or_else(|| cfg_err("`selected_prefix_last_write_refine` must be an object"))?;
        if obj.get("kind").and_then(Json::as_str) != Some("exact_sender") {
            return Err(cfg_err(
                "`selected_prefix_last_write_refine.kind` must be `\"exact_sender\"`",
            ));
        }
        if obj.get("version").and_then(json_i64) != Some(1) {
            return Err(cfg_err(
                "`selected_prefix_last_write_refine.version` must be 1",
            ));
        }
        let w = obj
            .get("weight")
            .and_then(Json::as_f64)
            .ok_or_else(|| cfg_err("missing `selected_prefix_last_write_refine.weight`"))?;
        if !w.is_finite() || w < 0.0 {
            return Err(cfg_err(
                "`selected_prefix_last_write_refine.weight` must be non-negative",
            ));
        }
        let tk = obj
            .get("head_topk")
            .and_then(json_usize)
            .ok_or_else(|| cfg_err("missing `selected_prefix_last_write_refine.head_topk`"))?;
        topks.push(tk);
    }
    if topks.is_empty() {
        return Err(cfg_err(
            "DSpine config names neither `code_bridge_second_pass` nor `selected_prefix_last_write_refine`",
        ));
    }
    let head_topk = topks[0];
    if topks.iter().any(|&tk| tk != head_topk) {
        return Err(cfg_err(format!(
            "DSpine `head_topk` sources disagree: {topks:?}"
        )));
    }
    if !(2..=64).contains(&head_topk) || !head_topk.is_power_of_two() {
        return Err(cfg_err(format!(
            "DSpine `head_topk` must be a power of two in 2..=64, got {head_topk}"
        )));
    }

    Ok(DspineCfg {
        vocab_size,
        code_size,
        rank,
        eps,
        head_topk,
    })
}

impl Config {
    fn read(hp: &Hp, raw: &Json) -> Result<Self, CucaError> {
        check_architecture(raw)?;
        let n_layer = hp.n_layer()?;
        let n_embd = hp.n_embd()?;
        let n_head = hp.n_head()?;
        if let Some(act) = hp.get("hidden_act").and_then(Json::as_str)
            && act != "silu"
        {
            return Err(cfg_err(format!(
                "the DFlash drafter's activation `{act}` is not supported (only `silu`)"
            )));
        }
        if hp.get("attention_bias").and_then(Json::as_bool) == Some(true) {
            return Err(cfg_err(
                "the DFlash drafter's `attention_bias` must be false",
            ));
        }
        if hp.get("tie_word_embeddings").and_then(Json::as_bool) == Some(true) {
            return Err(cfg_err(
                "the DFlash drafter's `tie_word_embeddings` must be false (it shares the target head)",
            ));
        }
        if hp.get("rope_scaling").is_some_and(|v| !v.is_null()) {
            return Err(cfg_err(
                "the DFlash drafter's `rope_scaling` is not supported (only default RoPE)",
            ));
        }
        let rope = hp.rope_all();
        if let Some(kind) = rope
            .get("rope_type")
            .or_else(|| rope.get("type"))
            .and_then(Json::as_str)
            && kind != "default"
        {
            return Err(cfg_err(format!(
                "the DFlash drafter's rope type `{kind}` is not supported (only `default`)"
            )));
        }

        let df = match hp.get("dflash_config") {
            None | Some(Json::Null) => None,
            Some(Json::Object(map)) => Some(map),
            Some(_) => return Err(cfg_err("`dflash_config` is not an object")),
        };
        let is_dspine = has_arch(raw, "DSpineDraftModel")
            || df.is_some_and(|d| d.contains_key("selected_prefix_code_bridge"))
            || hp
                .get("code_bridge_second_pass")
                .is_some_and(|v| !v.is_null());

        let block_causal = df
            .and_then(|d| d.get("draft_block_causal"))
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let dspine = if is_dspine {
            let df_map = df.ok_or_else(|| cfg_err("DSpine checkpoint misses `dflash_config`"))?;
            Some(read_dspine_cfg(hp, df_map, n_layer, block_causal)?)
        } else {
            None
        };

        let layer_types = parse_layer_types(hp.get("layer_types"), n_layer)?;
        let use_sw = hp.get("use_sliding_window").and_then(Json::as_bool);
        let (sliding, window) = if use_sw == Some(false) {
            if layer_types.as_ref().is_some_and(|lt| lt.iter().any(|&s| s)) {
                return Err(cfg_err(
                    "`layer_types` declares `sliding_attention` while `use_sliding_window` is false",
                ));
            }
            if hp.get("sliding_window").is_some_and(|v| !v.is_null()) {
                return Err(cfg_err(
                    "`sliding_window` is set while `use_sliding_window` is false",
                ));
            }
            (vec![false; n_layer], None)
        } else if is_dspine {
            let declares = use_sw == Some(true)
                || layer_types.as_ref().is_some_and(|lt| lt.iter().any(|&s| s));
            if !declares {
                if hp.get("sliding_window").is_some_and(|v| !v.is_null()) {
                    return Err(cfg_err(
                        "`sliding_window` is set while sliding attention is off",
                    ));
                }
                (vec![false; n_layer], None)
            } else {
                let all_sliding = layer_types.as_ref().is_some_and(|lt| lt.iter().all(|&s| s));
                let max_win = hp.uint(&["max_window_layers"]).unwrap_or(n_layer);
                let w = hp.uint(&["sliding_window"]).filter(|&w| w > 0);
                match (use_sw == Some(true), all_sliding, max_win == n_layer, w) {
                    (true, true, true, Some(w)) => (vec![true; n_layer], Some(w)),
                    _ => {
                        return Err(cfg_err(
                            "DSpine sliding attention requires `use_sliding_window = true`, all `sliding_attention` layers, `max_window_layers = num_hidden_layers`, and positive `sliding_window`",
                        ));
                    }
                }
            }
        } else {
            let sliding = layer_types.unwrap_or_else(|| vec![true; n_layer]);
            let any_sw = sliding.iter().any(|&s| s);
            let window = if any_sw {
                let w = hp.uint(&["sliding_window"]).unwrap_or(2048);
                if w == 0 {
                    return Err(cfg_err("`sliding_window` must be positive"));
                }
                Some(w)
            } else {
                if hp.get("sliding_window").is_some_and(|v| !v.is_null()) {
                    return Err(cfg_err(
                        "`sliding_window` is set without any `sliding_attention` layer",
                    ));
                }
                None
            };
            (sliding, window)
        };

        let target_ids_val = df
            .and_then(|d| d.get("target_layer_ids"))
            .or_else(|| hp.get("target_layer_ids"));
        let target_ids = match target_ids_val {
            None if df.is_some() || is_dspine => {
                return Err(cfg_err("missing `dflash_config.target_layer_ids`"));
            }
            None => vec![1, 13, 25, 37, 49],
            Some(Json::Array(ids)) => ids
                .iter()
                .map(|v| {
                    json_usize(v).ok_or_else(|| cfg_err("target_layer_ids holds a non-integer"))
                })
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err(cfg_err("target_layer_ids is not a list")),
        };
        if target_ids.is_empty() || target_ids.windows(2).any(|w| w[0] >= w[1]) {
            return Err(cfg_err(
                "target_layer_ids must be a non-empty ascending list",
            ));
        }
        if let Some(nt) = hp.uint(&["num_target_layers"])
            && target_ids.iter().any(|&id| id >= nt)
        {
            return Err(cfg_err(format!(
                "target_layer_ids {target_ids:?} exceeds num_target_layers {nt}"
            )));
        }

        let mask_token = match df
            .and_then(|d| d.get("mask_token_id"))
            .and_then(json_usize)
            .or_else(|| hp.uint(&["mask_token_id"]))
        {
            Some(id) => id,
            None if df.is_some() || is_dspine => {
                return Err(cfg_err("missing `dflash_config.mask_token_id`"));
            }
            None => 201_818,
        };
        if let Some(vocab) = hp.uint(&["vocab_size"]) {
            if mask_token >= vocab {
                return Err(cfg_err(format!(
                    "mask_token_id {mask_token} outside vocab {vocab}"
                )));
            }
            if let Some(dv) = hp.uint(&["draft_vocab_size"])
                && dv != vocab
            {
                return Err(cfg_err(format!(
                    "draft_vocab_size {dv} != vocab_size {vocab}"
                )));
            }
        }

        let rope_theta = hp
            .float(&["rope_theta"])
            .or_else(|| rope.get("rope_theta").and_then(Json::as_f64))
            .unwrap_or(500_000.0);

        Ok(Self {
            n_layer,
            n_embd,
            n_head,
            n_kv_head: hp.uint(&["num_key_value_heads"]).unwrap_or(n_head),
            head_dim: hp.head_dim()?,
            n_ff: hp.req_uint(&["intermediate_size"])?,
            eps: hp.float(&["rms_norm_eps"]).unwrap_or(1e-5),
            rope_theta,
            max_positions: hp.uint(&["max_position_embeddings"]).unwrap_or(131_072),
            block_size: hp.uint(&["block_size"]).unwrap_or(16),
            mask_token,
            target_ids,
            sliding,
            window,
            block_causal,
            dspine,
        })
    }

    /// The `dflash.*` and tokenizer metadata, llama.cpp's plus block-causal
    /// and DSpine extensions.
    fn metadata(&self, name: &str, quant: WeightQuant) -> Result<Vec<(String, Value)>, CucaError> {
        let u = |key: &str, v: usize| -> Result<(String, Value), CucaError> {
            let v =
                u32::try_from(v).map_err(|_| cfg_err(format!("{key}: {v} does not fit u32")))?;
            Ok((key.to_string(), Value::U32(v)))
        };
        let key = |k: &str| format!("{ARCH}.{k}");
        let mut kv = vec![
            (
                "general.architecture".to_string(),
                Value::String(ARCH.into()),
            ),
            ("general.type".to_string(), Value::String("model".into())),
            ("general.name".to_string(), Value::String(name.to_string())),
            u(&key("block_count"), self.n_layer)?,
            u(&key("context_length"), self.max_positions)?,
            u(&key("embedding_length"), self.n_embd)?,
            u(&key("feed_forward_length"), self.n_ff)?,
            u(&key("attention.head_count"), self.n_head)?,
            u(&key("attention.head_count_kv"), self.n_kv_head)?,
            u(&key("attention.key_length"), self.head_dim)?,
            u(&key("attention.value_length"), self.head_dim)?,
            (key("rope.freq_base"), Value::F32(self.rope_theta as f32)),
            (
                key("attention.layer_norm_rms_epsilon"),
                Value::F32(self.eps as f32),
            ),
            u(&key("block_size"), self.block_size)?,
            (
                key("target_layers"),
                // llama.cpp stores the input of layer i + 1 = output of i.
                Value::Array(
                    self.target_ids
                        .iter()
                        .map(|&i| {
                            i32::try_from(i + 1)
                                .map(Value::I32)
                                .map_err(|_| cfg_err("target_layer_ids does not fit i32"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            ),
        ];
        if let Some(window) = self.window {
            kv.push(u(&key("attention.sliding_window"), window)?);
            kv.push((
                key("attention.sliding_window_pattern"),
                Value::Array(self.sliding.iter().map(|&s| Value::Bool(s)).collect()),
            ));
        }
        if self.block_causal {
            kv.push((key("block_causal"), Value::Bool(true)));
        }
        if let Some(ds) = &self.dspine {
            kv.push(u(&key("dspine.code_size"), ds.code_size)?);
            kv.push(u(&key("dspine.rank"), ds.rank)?);
            kv.push((key("dspine.eps"), Value::F32(ds.eps as f32)));
            kv.push(u(&key("dspine.head_topk"), ds.head_topk)?);
        }
        kv.push(u("tokenizer.ggml.mask_token_id", self.mask_token)?);
        kv.push(("general.quantization_version".to_string(), Value::U32(2)));
        kv.push((
            "general.file_type".to_string(),
            Value::U32(file_type(quant)),
        ));
        Ok(kv)
    }

    /// Trunk and 2-D/1-D bridge tensors as `(hf, gguf, shape)`.
    fn expected(
        &self,
        pfx: &str,
        encoder_pair: (String, String),
    ) -> Vec<(String, String, Vec<usize>)> {
        let (n, d) = (self.n_embd, self.head_dim);
        let (qd, kd) = (self.n_head * d, self.n_kv_head * d);
        let t = |hf: String, gguf: String, shape: Vec<usize>| (hf, gguf, shape);
        let (fc_hf, enc_norm_hf) = encoder_pair;
        let mut tensors = vec![
            t(
                fc_hf,
                "fc.weight".into(),
                vec![n, self.target_ids.len() * n],
            ),
            t(enc_norm_hf, "enc.output_norm.weight".into(), vec![n]),
            t(
                format!("{pfx}norm.weight"),
                "output_norm.weight".into(),
                vec![n],
            ),
        ];
        for il in 0..self.n_layer {
            let l = |hf: &str, gguf: &str, shape: Vec<usize>| {
                t(
                    format!("{pfx}layers.{il}.{hf}.weight"),
                    format!("blk.{il}.{gguf}.weight"),
                    shape,
                )
            };
            tensors.extend([
                l("input_layernorm", "attn_norm", vec![n]),
                l("self_attn.q_proj", "attn_q", vec![qd, n]),
                l("self_attn.k_proj", "attn_k", vec![kd, n]),
                l("self_attn.v_proj", "attn_v", vec![kd, n]),
                l("self_attn.o_proj", "attn_output", vec![n, qd]),
                l("self_attn.q_norm", "attn_q_norm", vec![d]),
                l("self_attn.k_norm", "attn_k_norm", vec![d]),
                l("post_attention_layernorm", "ffn_norm", vec![n]),
                l("mlp.gate_proj", "ffn_gate", vec![self.n_ff, n]),
                l("mlp.up_proj", "ffn_up", vec![self.n_ff, n]),
                l("mlp.down_proj", "ffn_down", vec![n, self.n_ff]),
            ]);
        }
        if let Some(ds) = &self.dspine {
            let b = |s: &str| format!("{pfx}selected_prefix_bridge.{s}");
            tensors.extend([
                t(
                    b("codebook"),
                    "dspine.codebook.weight".into(),
                    vec![ds.vocab_size, ds.code_size],
                ),
                t(
                    b("read_weight"),
                    "dspine.read.weight".into(),
                    vec![ds.code_size, n],
                ),
                t(
                    b("gate.weight"),
                    "dspine.gate.weight".into(),
                    vec![ds.rank, ds.code_size],
                ),
                t(b("gate.bias"), "dspine.gate.bias".into(), vec![ds.rank]),
                t(
                    b("predecessor_gate.weight"),
                    "dspine.pred_gate.weight".into(),
                    vec![ds.rank, ds.code_size],
                ),
                t(
                    b("prev.weight"),
                    "dspine.prev.weight".into(),
                    vec![ds.rank, ds.code_size],
                ),
            ]);
        }
        tensors
    }
}

/// Detect the optional `"model."` tensor prefix and encoder naming (`encoder.*`
/// for Muse Glimmer, `[model.]fc` + `[model.]hidden_norm` for DFlash/DSpine).
fn checkpoint_naming(shards: &Shards) -> (&'static str, (String, String)) {
    let pfx = if shards.tensor("model.fc.weight").is_some()
        || shards
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .is_some()
    {
        "model."
    } else {
        ""
    };
    let encoder_pair = if shards.tensor("encoder.fc.weight").is_some() {
        (
            "encoder.fc.weight".to_string(),
            "encoder.output_norm_enc.weight".to_string(),
        )
    } else {
        (
            format!("{pfx}fc.weight"),
            format!("{pfx}hidden_norm.weight"),
        )
    };
    (pfx, encoder_pair)
}

/// One plan per tensor, checked against the config: a missing tensor, an
/// unknown one (an embedding table or head of its own, a variant this layout
/// does not hold) or a wrong shape is an error naming it.
fn plan(cfg: &Config, shards: &Shards) -> Result<Vec<Plan>, CucaError> {
    let (pfx, encoder_pair) = checkpoint_naming(shards);
    let expected = cfg.expected(pfx, encoder_pair);
    let mut allowed: HashSet<String> = expected.iter().map(|(hf, ..)| hf.clone()).collect();

    if cfg.dspine.is_some() {
        for v in VERSION_SCALARS {
            let hf = format!("{pfx}{v}");
            let got = shards.read_i64_scalar(&hf)?;
            if got != 1 {
                return Err(cfg_err(format!(
                    "DSpine tensor `{hf}` has version {got}, expected 1"
                )));
            }
            allowed.insert(hf);
        }
    } else {
        let causal_ver = format!("{pfx}draft_block_causal_version");
        if shards.tensor(&causal_ver).is_some() {
            let got = shards.read_i64_scalar(&causal_ver)?;
            if got != 1 || !cfg.block_causal {
                return Err(cfg_err(format!(
                    "tensor `{causal_ver}` ({got}) requires draft_block_causal = true and version 1"
                )));
            }
            allowed.insert(causal_ver);
        }
    }

    let early_hf = format!("{pfx}selected_prefix_bridge.early_read_weights");
    let write_hf = format!("{pfx}selected_prefix_bridge.feature_write_weights");
    let out_hf = format!("{pfx}selected_prefix_bridge.out.weight");
    if let Some(ds) = &cfg.dspine {
        allowed.insert(early_hf.clone());
        allowed.insert(write_hf.clone());
        let out_meta = shards
            .tensor(&out_hf)
            .ok_or_else(|| cfg_err(format!("the DSpine drafter has no tensor `{out_hf}`")))?;
        let out_want = [ds.code_size, ds.rank];
        if out_meta.shape != out_want {
            return Err(cfg_err(format!(
                "tensor `{out_hf}` has shape {:?}, expected {out_want:?} from config.json",
                out_meta.shape
            )));
        }
        let mut out_vals = Vec::new();
        shards.read_rows(&out_hf, 0, out_meta.rows(), &mut out_vals)?;
        if out_vals.iter().any(|&v| v != 0.0) {
            return Err(cfg_err(format!(
                "DSpine tensor `{out_hf}` is non-zero, but selected_prefix_terminal is disabled"
            )));
        }
        allowed.insert(out_hf);
    }

    for name in shards.names() {
        if !allowed.contains(name) {
            return Err(cfg_err(format!(
                "tensor `{name}` of the DFlash drafter has no GGUF name (the drafter holds no embedding table or head: it reads the target's)"
            )));
        }
    }

    let mut plans = Vec::with_capacity(expected.len() + cfg.n_layer * 2);
    for (hf, name, shape) in expected {
        let meta = shards
            .tensor(&hf)
            .ok_or_else(|| cfg_err(format!("the DFlash drafter has no tensor `{hf}`")))?;
        if meta.shape != shape {
            return Err(cfg_err(format!(
                "tensor `{hf}` has shape {:?}, expected {shape:?} from config.json",
                meta.shape
            )));
        }
        plans.push(Plan {
            name,
            dims: shape,
            body: Body::Slices(vec![Slice {
                tensor: hf,
                row_start: 0,
                rows: meta.rows(),
            }]),
            ops: Vec::new(),
        });
    }

    if let Some(ds) = &cfg.dspine {
        let early_meta = shards
            .tensor(&early_hf)
            .ok_or_else(|| cfg_err(format!("the DSpine drafter has no tensor `{early_hf}`")))?;
        let early_want = [cfg.n_layer - 1, ds.code_size, cfg.n_embd];
        if early_meta.shape != early_want {
            return Err(cfg_err(format!(
                "tensor `{early_hf}` has shape {:?}, expected {early_want:?} from config.json",
                early_meta.shape
            )));
        }
        for il in 0..cfg.n_layer - 1 {
            plans.push(Plan {
                name: format!("dspine.blk.{il}.early_read.weight"),
                dims: vec![ds.code_size, cfg.n_embd],
                body: Body::Slices(vec![Slice {
                    tensor: early_hf.clone(),
                    row_start: il * ds.code_size,
                    rows: ds.code_size,
                }]),
                ops: Vec::new(),
            });
        }

        let write_meta = shards
            .tensor(&write_hf)
            .ok_or_else(|| cfg_err(format!("the DSpine drafter has no tensor `{write_hf}`")))?;
        let write_want = [cfg.n_layer, cfg.n_embd, ds.rank];
        if write_meta.shape != write_want {
            return Err(cfg_err(format!(
                "tensor `{write_hf}` has shape {:?}, expected {write_want:?} from config.json",
                write_meta.shape
            )));
        }
        for il in 0..cfg.n_layer {
            plans.push(Plan {
                name: format!("dspine.blk.{il}.write.weight"),
                dims: vec![cfg.n_embd, ds.rank],
                body: Body::Slices(vec![Slice {
                    tensor: write_hf.clone(),
                    row_start: il * cfg.n_embd,
                    rows: cfg.n_embd,
                }]),
                ops: Vec::new(),
            });
        }
    }

    Ok(plans)
}

/// Write the GGUF of the DFlash or DSpine drafter `ck` to `partial`.
///
/// # Errors
///
/// [`CucaError::Config`] for a config or a tensor set this layout does not
/// hold, [`CucaError::Io`] for file errors.
pub(crate) fn convert(
    ck: &CheckpointDir,
    partial: &Path,
    quant: WeightQuant,
    progress: Option<&DownloadProgressCallback>,
) -> Result<(), CucaError> {
    let hp = Hp::new(&ck.config)?;
    let cfg = Config::read(&hp, &ck.config)?;
    let shards = Shards::open(&ck.shards)?;
    let plans = plan(&cfg, &shards)?;
    let kv = cfg.metadata(&super::model_name(&ck.dir), quant)?;
    // The engine's own reader validates the metadata it will read.
    let md = kv.iter().cloned().collect();
    check_metadata(&md)
        .map_err(|e| cfg_err(format!("the converted metadata does not load: {e}")))?;

    let info = super::quantmix::ModelInfo::default();
    super::write_plans(partial, &kv, &plans, &shards, (quant, &info), progress)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::provider::local::gguf;
    use candle_core::Device;
    use candle_core::quantized::GgmlDType;

    use super::super::safetensors::tests::{f32_bytes, temp_dir, write_file};
    use super::*;
    use crate::provider::local::model::cpu::dflash::DflashNet;
    use crate::provider::local::model::tiny::{
        TinyDflash, TinySpec, load, write_spec, write_tiny_dflash,
    };

    /// The checkpoint name of GGUF tensor `name` (the real
    /// `Muse-Glimmer-30B-assistant` names, written out by hand).
    fn hf_name(name: &str) -> String {
        match name {
            "fc.weight" => return "encoder.fc.weight".into(),
            "enc.output_norm.weight" => return "encoder.output_norm_enc.weight".into(),
            "output_norm.weight" => return "norm.weight".into(),
            _ => {}
        }
        let rest = name.strip_prefix("blk.").unwrap();
        let (il, part) = rest.split_once('.').unwrap();
        let part = match part {
            "attn_norm.weight" => "input_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            other => panic!("no checkpoint name for `{other}`"),
        };
        format!("layers.{il}.{part}")
    }

    fn config() -> Json {
        serde_json::json!({
            "architectures": ["MuseGlimmerAssistantModel"],
            "model_type": "muse_glimmer_assistant",
            "hidden_size": 64, "intermediate_size": 96,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "hidden_act": "silu", "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "max_position_embeddings": 512, "sliding_window": 6,
            "layer_types": ["sliding_attention", "sliding_attention"],
            "block_size": 8, "mask_token_id": 14, "target_layer_ids": [0, 2]
        })
    }

    /// The tensors of a drafter file as `(name, shape, f32 values)`.
    fn read_gguf(path: &Path) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let content = gguf::Content::open(path).unwrap();
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort();
        names
            .into_iter()
            .map(|n| {
                let t = content
                    .reader()
                    .tensor(&n, &Device::Cpu)
                    .unwrap()
                    .dequantize(&Device::Cpu)
                    .unwrap();
                let shape = t.dims().to_vec();
                (n, shape, t.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            })
            .collect()
    }

    /// A checkpoint directory holding `config` and the tensors of `source`
    /// (a drafter file) under their checkpoint names, as F32 shards.
    fn checkpoint(tag: &str, config: &Json, source: &Path) -> PathBuf {
        let dir = temp_dir(tag);
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let tensors: Vec<(String, Vec<usize>, Vec<u8>)> = read_gguf(source)
            .into_iter()
            .map(|(n, shape, v)| (hf_name(&n), shape, f32_bytes(&v)))
            .collect();
        let refs: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors
            .iter()
            .map(|(n, s, b)| (n.as_str(), "F32", s.as_slice(), b.clone()))
            .collect();
        write_file(&dir.join("model.safetensors"), &refs);
        dir
    }

    fn convert_dir(dir: &Path, out: &Path) -> Result<(), CucaError> {
        let ck = CheckpointDir::open(dir)?;
        super::super::convert(&ck, out, WeightQuant::F16, None)
    }

    #[test]
    fn the_checkpoint_is_recognised_by_its_class_or_model_type() {
        assert!(is_dflash(&config()));
        assert!(is_dflash(
            &serde_json::json!({"model_type": "muse_glimmer_assistant"})
        ));
        assert!(is_dflash(
            &serde_json::json!({"architectures": ["MuseGlimmerAssistantModel"]})
        ));
        assert!(is_dflash(&serde_json::json!({
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "dflash_config": {}
        })));
        assert!(is_dflash(&serde_json::json!({
            "architectures": ["DSpineDraftModel"],
            "model_type": "qwen3",
            "dflash_config": {}
        })));
        assert!(!is_dflash(&serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3"
        })));
        assert!(!is_dflash(
            &serde_json::json!({"architectures": ["MuseGlimmerForConditionalGeneration"]})
        ));
        assert!(check_architecture(&config()).is_ok());
        assert!(
            check_architecture(&serde_json::json!({
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3"
            }))
            .is_ok()
        );
        let err = check_architecture(&serde_json::json!({"architectures": ["PixtralModel"]}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("PixtralModel"), "{err}");
        let err = check_architecture(&serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "dflash_config": {}
        }))
        .unwrap_err()
        .to_string();
        assert!(err.contains("DFlash2DraftModel"), "{err}");
        let err = check_architecture(&serde_json::json!({
            "architectures": ["DSpineDraftModel"],
            "dflash_config": {"conv_kernel_size": 3}
        }))
        .unwrap_err()
        .to_string();
        assert!(err.contains("conv_kernel_size"), "{err}");
    }

    /// A converted drafter equals the file it was made from, in llama.cpp's
    /// layout, and drafts exactly like it.
    #[test]
    fn a_drafter_checkpoint_converts_to_the_dflash_layout() {
        let fixture = write_tiny_dflash(&TinyDflash {
            dtype: GgmlDType::F16,
            ..TinyDflash::default()
        });
        let dir = checkpoint("dflash-conv", &config(), &fixture.path);
        let out = dir.join("drafter.gguf");
        convert_dir(&dir, &out).unwrap();

        // Names, shapes, values and storage types.
        let want = read_gguf(&fixture.path);
        let got = read_gguf(&out);
        assert_eq!(got.len(), 3 + 11 * 2);
        assert_eq!(got, want, "every tensor keeps its values and shape");
        let content = gguf::Content::open(&out).unwrap();
        for (name, info) in &content.tensor_infos {
            let norm = name.ends_with("norm.weight");
            assert_eq!(
                info.dtype,
                gguf::GgufType::Ggml(if norm { GgmlDType::F32 } else { GgmlDType::F16 }),
                "{name}"
            );
        }

        // The metadata llama.cpp writes, plus the mask token.
        let md = &content.metadata;
        let get = |k: &str| md.get(k).unwrap_or_else(|| panic!("no `{k}`"));
        assert_eq!(get("general.architecture").to_string().unwrap(), "dflash");
        let ints = |k: &str| -> Vec<i64> {
            get(k)
                .to_vec()
                .unwrap()
                .iter()
                .map(|v| match v {
                    Value::I32(n) => i64::from(*n),
                    other => panic!("{other:?}"),
                })
                .collect()
        };
        assert_eq!(ints("dflash.target_layers"), [1, 3], "ids plus one");
        assert_eq!(get("dflash.block_size").to_u32().unwrap(), 8);
        assert_eq!(get("dflash.block_count").to_u32().unwrap(), 2);
        assert_eq!(get("dflash.embedding_length").to_u32().unwrap(), 64);
        assert_eq!(get("dflash.feed_forward_length").to_u32().unwrap(), 96);
        assert_eq!(get("dflash.attention.head_count").to_u32().unwrap(), 4);
        assert_eq!(get("dflash.attention.head_count_kv").to_u32().unwrap(), 2);
        assert_eq!(get("dflash.attention.key_length").to_u32().unwrap(), 16);
        assert_eq!(get("dflash.attention.sliding_window").to_u32().unwrap(), 6);
        assert_eq!(get("dflash.context_length").to_u32().unwrap(), 512);
        assert_eq!(get("dflash.rope.freq_base").to_f32().unwrap(), 500_000.0);
        assert_eq!(get("tokenizer.ggml.mask_token_id").to_u32().unwrap(), 14);
        assert_eq!(
            get("dflash.attention.sliding_window_pattern")
                .to_vec()
                .unwrap()
                .len(),
            2
        );

        // Both files draft the same tokens.
        let model = write_spec(&TinySpec::muse_glimmer());
        let target = load(&model, 32);
        let open = |path: &Path| {
            let content = gguf::Content::open(path).unwrap();
            DflashNet::open(&content, &target, 32).unwrap()
        };
        let (mut a, mut b) = (open(&fixture.path), open(&out));
        let feats = crate::provider::local::model::tiny::Lcg(3).values(10 * a.feature_width(), 1.5);
        a.observe(0, &feats).unwrap();
        b.observe(0, &feats).unwrap();
        let drafts = |net: &DflashNet| {
            let mut rows = Vec::new();
            net.draft(&target, 5, 10, 4, &mut |l| {
                rows.push(l.to_vec());
                Some(0)
            })
            .unwrap();
            rows
        };
        assert_eq!(drafts(&a), drafts(&b));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The real checkpoint is BF16: its values survive the conversion to
    /// F16 and the norms stay exact.
    #[test]
    fn bf16_weights_convert_with_their_values() {
        let fixture = write_tiny_dflash(&TinyDflash::default());
        let dir = temp_dir("dflash-bf16");
        std::fs::write(dir.join("config.json"), config().to_string()).unwrap();
        // `(checkpoint name, shape, bf16 bytes, the values they hold)`.
        type Tensor = (String, Vec<usize>, Vec<u8>, Vec<f32>);
        let tensors: Vec<Tensor> = read_gguf(&fixture.path)
            .into_iter()
            .map(|(n, shape, v)| {
                let rounded: Vec<f32> = v
                    .iter()
                    .map(|x| half::bf16::from_f32(*x).to_f32())
                    .collect();
                let bytes = rounded
                    .iter()
                    .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
                    .collect();
                (hf_name(&n), shape, bytes, rounded)
            })
            .collect();
        let refs: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors
            .iter()
            .map(|(n, s, b, _)| (n.as_str(), "BF16", s.as_slice(), b.clone()))
            .collect();
        write_file(&dir.join("model.safetensors"), &refs);
        let out = dir.join("drafter.gguf");
        convert_dir(&dir, &out).unwrap();
        let got: std::collections::HashMap<String, Vec<f32>> = read_gguf(&out)
            .into_iter()
            .map(|(n, _, v)| (hf_name(&n), v))
            .collect();
        assert_eq!(got.len(), tensors.len());
        for (name, _, _, want) in &tensors {
            let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            for (w, v) in want.iter().zip(&got[name]) {
                assert!((w - v).abs() <= 2e-3 * scale, "{name}: {w} vs {v}");
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// What the layout does not hold, or holds differently, fails the
    /// conversion naming it.
    #[test]
    fn an_unsupported_checkpoint_is_refused_by_name() {
        let fixture = write_tiny_dflash(&TinyDflash {
            dtype: GgmlDType::F32,
            ..TinyDflash::default()
        });
        let out = |dir: &Path| dir.join("drafter.gguf");
        let refused = |dir: &Path| {
            convert_dir(dir, &out(dir))
                .expect_err("must be refused")
                .to_string()
        };

        // A tensor of the checkpoint the layout has no name for (the
        // drafter reads the target's embeddings).
        let dir = checkpoint("dflash-extra", &config(), &fixture.path);
        let mut tensors = read_gguf(&fixture.path)
            .into_iter()
            .map(|(n, s, v)| (hf_name(&n), s, f32_bytes(&v)))
            .collect::<Vec<_>>();
        tensors.push((
            "embed_tokens.weight".into(),
            vec![16, 64],
            f32_bytes(&[0.0; 1024]),
        ));
        let refs: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors
            .iter()
            .map(|(n, s, b)| (n.as_str(), "F32", s.as_slice(), b.clone()))
            .collect();
        write_file(&dir.join("model.safetensors"), &refs);
        let e = refused(&dir);
        assert!(
            e.contains("embed_tokens.weight") && e.contains("no GGUF name"),
            "{e}"
        );

        // A missing tensor.
        tensors.retain(|(n, ..)| n != "embed_tokens.weight" && n != "layers.1.mlp.up_proj.weight");
        let refs: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors
            .iter()
            .map(|(n, s, b)| (n.as_str(), "F32", s.as_slice(), b.clone()))
            .collect();
        write_file(&dir.join("model.safetensors"), &refs);
        let e = refused(&dir);
        assert!(e.contains("layers.1.mlp.up_proj.weight"), "{e}");

        // A shape the config contradicts: three target layers, `fc` for two.
        let mut cfg = config();
        cfg["target_layer_ids"] = serde_json::json!([0, 1, 2]);
        let dir2 = checkpoint("dflash-shape", &cfg, &fixture.path);
        let e = refused(&dir2);
        assert!(
            e.contains("encoder.fc.weight") && e.contains("shape"),
            "{e}"
        );

        // Descending layers, another activation, another rope.
        for (key, value, needle) in [
            ("target_layer_ids", serde_json::json!([2, 0]), "ascending"),
            ("hidden_act", serde_json::json!("gelu"), "activation"),
            (
                "rope_parameters",
                serde_json::json!({"rope_theta": 1e4, "rope_type": "yarn"}),
                "rope type",
            ),
        ] {
            let mut cfg = config();
            cfg[key] = value;
            let dir3 = checkpoint("dflash-cfg", &cfg, &fixture.path);
            let e = refused(&dir3);
            assert!(e.contains(needle), "{key}: {e}");
            let _ = std::fs::remove_dir_all(dir3);
        }
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(dir2);
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect()
    }

    fn qwen_hf_name(gguf_name: &str, prefix: &str) -> Option<String> {
        match gguf_name {
            "fc.weight" => Some(format!("{prefix}fc.weight")),
            "enc.output_norm.weight" => Some(format!("{prefix}hidden_norm.weight")),
            "output_norm.weight" => Some(format!("{prefix}norm.weight")),
            "dspine.codebook.weight" => Some(format!("{prefix}selected_prefix_bridge.codebook")),
            "dspine.read.weight" => Some(format!("{prefix}selected_prefix_bridge.read_weight")),
            "dspine.gate.weight" => Some(format!("{prefix}selected_prefix_bridge.gate.weight")),
            "dspine.gate.bias" => Some(format!("{prefix}selected_prefix_bridge.gate.bias")),
            "dspine.pred_gate.weight" => Some(format!(
                "{prefix}selected_prefix_bridge.predecessor_gate.weight"
            )),
            "dspine.prev.weight" => Some(format!("{prefix}selected_prefix_bridge.prev.weight")),
            other if other.starts_with("dspine.blk.") => None,
            _ => Some(format!("{prefix}{}", hf_name(gguf_name))),
        }
    }

    fn dspine_checkpoint_tensors(
        source: &Path,
        prefix: &str,
        n_layer: usize,
        out_zero: bool,
        causal_ver: i64,
    ) -> Vec<(String, &'static str, Vec<usize>, Vec<u8>)> {
        let gguf = read_gguf(source);
        let mut out = Vec::new();
        let mut early_reads: Vec<(usize, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut writes: Vec<(usize, Vec<usize>, Vec<f32>)> = Vec::new();
        for (name, shape, vals) in &gguf {
            if let Some(rest) = name.strip_prefix("dspine.blk.") {
                let (il_s, part) = rest.split_once('.').unwrap();
                let il: usize = il_s.parse().unwrap();
                if part == "early_read.weight" {
                    early_reads.push((il, shape.clone(), vals.clone()));
                } else if part == "write.weight" {
                    writes.push((il, shape.clone(), vals.clone()));
                } else {
                    panic!("unexpected dspine.blk tensor `{name}`");
                }
                continue;
            }
            let hf = qwen_hf_name(name, prefix).unwrap();
            out.push((hf, "BF16", shape.clone(), bf16_bytes(vals)));
        }

        early_reads.sort_by_key(|(il, _, _)| *il);
        assert_eq!(early_reads.len(), n_layer - 1);
        let (_, e_shape, _) = &early_reads[0];
        let mut early_all = Vec::new();
        for (_, _, vals) in &early_reads {
            early_all.extend_from_slice(vals);
        }
        out.push((
            format!("{prefix}selected_prefix_bridge.early_read_weights"),
            "BF16",
            vec![n_layer - 1, e_shape[0], e_shape[1]],
            bf16_bytes(&early_all),
        ));

        writes.sort_by_key(|(il, _, _)| *il);
        assert_eq!(writes.len(), n_layer);
        let (_, w_shape, _) = &writes[0];
        let mut writes_all = Vec::new();
        for (_, _, vals) in &writes {
            writes_all.extend_from_slice(vals);
        }
        out.push((
            format!("{prefix}selected_prefix_bridge.feature_write_weights"),
            "BF16",
            vec![n_layer, w_shape[0], w_shape[1]],
            bf16_bytes(&writes_all),
        ));

        let (code_size, rank) = (e_shape[0], w_shape[1]);
        let mut out_w = vec![0.0f32; code_size * rank];
        if !out_zero {
            out_w[0] = 0.25;
        }
        out.push((
            format!("{prefix}selected_prefix_bridge.out.weight"),
            "BF16",
            vec![code_size, rank],
            bf16_bytes(&out_w),
        ));

        for (ver_name, ver_val) in [
            ("draft_block_causal_version", causal_ver),
            ("last_write_refine_version", 1),
            ("selected_prefix_bridge.independent_write_version", 1),
            ("selected_prefix_bridge.terminal_disabled_version", 1),
        ] {
            out.push((
                format!("{prefix}{ver_name}"),
                "I64",
                vec![],
                ver_val.to_le_bytes().to_vec(),
            ));
        }
        out
    }

    fn dspine_config_full_attn() -> Json {
        serde_json::json!({
            "architectures": ["DSpineDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 16,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000,
            "max_position_embeddings": 512,
            "use_sliding_window": false,
            "sliding_window": null,
            "max_window_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "block_size": 8,
            "num_target_layers": 4,
            "tie_word_embeddings": false,
            "dflash_config": {
                "target_layer_ids": [0, 2],
                "mask_token_id": 14,
                "draft_block_causal": true,
                "selected_prefix_terminal": false,
                "selected_prefix_decode_mode": "serial",
                "selected_prefix_feature_writes": true,
                "selected_prefix_feature_write_mode": "independent",
                "selected_prefix_feature_layers": [1, 2],
                "selected_prefix_code_bridge": {
                    "code_size": 32,
                    "rank": 32,
                    "eps": 1e-6,
                    "kappa": 0.5
                },
                "selected_prefix_last_write_refine": {
                    "kind": "exact_sender",
                    "version": 1,
                    "head_topk": 4,
                    "weight": 1.0
                }
            }
        })
    }

    #[test]
    fn dflash_draft_model_full_attention_converts_and_drafts() {
        let fixture = write_tiny_dflash(&TinyDflash {
            window: None,
            block_causal: true,
            dtype: GgmlDType::F16,
            ..TinyDflash::default()
        });
        let cfg = serde_json::json!({
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 16,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "rope_theta": 500_000.0,
            "max_position_embeddings": 512,
            "use_sliding_window": false,
            "sliding_window": null,
            "layer_types": ["full_attention", "full_attention"],
            "block_size": 8,
            "num_target_layers": 4,
            "tie_word_embeddings": false,
            "dflash_config": {
                "target_layer_ids": [0, 2],
                "mask_token_id": 14,
                "draft_block_causal": true
            }
        });
        let dir = temp_dir("dflash-std-full");
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let mut tensors: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> =
            read_gguf(&fixture.path)
                .into_iter()
                .map(|(n, s, v)| (qwen_hf_name(&n, "").unwrap(), "BF16", s, bf16_bytes(&v)))
                .collect();
        tensors.push((
            "draft_block_causal_version".into(),
            "I64",
            vec![],
            1i64.to_le_bytes().to_vec(),
        ));
        let refs: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors
            .iter()
            .map(|(n, dt, s, b)| (n.as_str(), *dt, s.as_slice(), b.clone()))
            .collect();
        write_file(&dir.join("model.safetensors"), &refs);
        let out = dir.join("drafter.gguf");
        convert_dir(&dir, &out).unwrap();

        let content = gguf::Content::open(&out).unwrap();
        assert!(matches!(
            content.metadata.get("dflash.block_causal"),
            Some(Value::Bool(true))
        ));
        assert!(
            !content
                .metadata
                .contains_key("dflash.attention.sliding_window")
        );
        let model = write_spec(&TinySpec::qwen3());
        let target = load(&model, 32);
        let mut net = DflashNet::open(&content, &target, 32).unwrap();
        let feats =
            crate::provider::local::model::tiny::Lcg(5).values(6 * net.feature_width(), 1.0);
        net.observe(0, &feats).unwrap();
        let got = net
            .draft(&target, 2, 6, 4, &mut |l| {
                Some(crate::provider::local::sampler::greedy(l))
            })
            .unwrap();
        assert_eq!(got.len(), 4);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn dspine_checkpoint_converts_both_variants_and_drafts() {
        use crate::provider::local::model::tiny::TinyDspine;
        let target_model = write_spec(&TinySpec::qwen3());
        let target = load(&target_model, 32);

        // Full-attention 8B-style DSpine (only selected_prefix_last_write_refine).
        let fixture = write_tiny_dflash(&TinyDflash {
            window: None,
            block_causal: true,
            dtype: GgmlDType::F16,
            dspine: Some(TinyDspine::for_vocab(target.vocab())),
            ..TinyDflash::default()
        });
        let dir_a = temp_dir("dspine-8b");
        std::fs::write(
            dir_a.join("config.json"),
            dspine_config_full_attn().to_string(),
        )
        .unwrap();
        let tensors_a = dspine_checkpoint_tensors(&fixture.path, "", 2, true, 1);
        let refs_a: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors_a
            .iter()
            .map(|(n, dt, s, b)| (n.as_str(), *dt, s.as_slice(), b.clone()))
            .collect();
        write_file(&dir_a.join("model.safetensors"), &refs_a);

        for (quant, want_cb) in [
            (WeightQuant::F16, gguf::GgufType::Ggml(GgmlDType::F16)),
            (WeightQuant::Q4KM, gguf::GgufType::Ggml(GgmlDType::Q8_0)),
        ] {
            let out = dir_a.join(format!("dspine-{quant:?}.gguf"));
            let ck = CheckpointDir::open(&dir_a).unwrap();
            super::super::convert(&ck, &out, quant, None).unwrap();
            let content = gguf::Content::open(&out).unwrap();
            assert_eq!(
                content.tensor_infos["dspine.codebook.weight"].dtype,
                want_cb
            );
            assert_eq!(
                content
                    .metadata
                    .get("dflash.dspine.head_topk")
                    .and_then(|v| v.to_u32().ok()),
                Some(4)
            );
            assert!(
                !content
                    .metadata
                    .contains_key("dflash.attention.sliding_window")
            );
            assert!(
                !content
                    .tensor_infos
                    .contains_key("selected_prefix_bridge.out.weight")
            );
            let mut net = DflashNet::open(&content, &target, 32).unwrap();
            let feats =
                crate::provider::local::model::tiny::Lcg(7).values(6 * net.feature_width(), 0.75);
            net.observe(0, &feats).unwrap();
            let drafts = net
                .draft(&target, 3, 6, 4, &mut |l| {
                    Some(crate::provider::local::sampler::greedy(l))
                })
                .unwrap();
            assert_eq!(drafts.len(), 4);
        }

        // Sliding-window 27B-style DSpine with `model.` tensor prefix and both
        // code_bridge_second_pass and selected_prefix_last_write_refine.
        let mut cfg_b = dspine_config_full_attn();
        cfg_b["use_sliding_window"] = serde_json::json!(true);
        cfg_b["sliding_window"] = serde_json::json!(8);
        cfg_b["layer_types"] = serde_json::json!(["sliding_attention", "sliding_attention"]);
        cfg_b["code_bridge_second_pass"] = serde_json::json!({
            "mode": "serial_table",
            "first_exact": 1,
            "head_topk": 4
        });
        let dir_b = temp_dir("dspine-27b");
        std::fs::write(dir_b.join("config.json"), cfg_b.to_string()).unwrap();
        let tensors_b = dspine_checkpoint_tensors(&fixture.path, "model.", 2, true, 1);
        let refs_b: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors_b
            .iter()
            .map(|(n, dt, s, b)| (n.as_str(), *dt, s.as_slice(), b.clone()))
            .collect();
        write_file(&dir_b.join("model.safetensors"), &refs_b);
        let out_b = dir_b.join("drafter.gguf");
        convert_dir(&dir_b, &out_b).unwrap();
        let content_b = gguf::Content::open(&out_b).unwrap();
        assert_eq!(
            content_b
                .metadata
                .get("dflash.attention.sliding_window")
                .and_then(|v| v.to_u32().ok()),
            Some(8)
        );
        let mut net_b = DflashNet::open(&content_b, &target, 32).unwrap();
        let feats =
            crate::provider::local::model::tiny::Lcg(9).values(12 * net_b.feature_width(), 0.75);
        net_b.observe(0, &feats).unwrap();
        assert_eq!(
            net_b
                .draft(&target, 2, 12, 4, &mut |l| {
                    Some(crate::provider::local::sampler::greedy(l))
                })
                .unwrap()
                .len(),
            4
        );

        let _ = std::fs::remove_dir_all(dir_a);
        let _ = std::fs::remove_dir_all(dir_b);
    }

    #[test]
    fn dspine_invalid_versions_non_zero_out_and_bad_configs_are_refused() {
        use crate::provider::local::model::tiny::TinyDspine;
        let fixture = write_tiny_dflash(&TinyDflash {
            window: None,
            block_causal: true,
            dtype: GgmlDType::F16,
            dspine: Some(TinyDspine::for_vocab(16)),
            ..TinyDflash::default()
        });

        let convert_err = |tag: &str, cfg: &Json, out_zero: bool, causal_ver: i64| -> String {
            let dir = temp_dir(tag);
            std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
            let tensors = dspine_checkpoint_tensors(&fixture.path, "", 2, out_zero, causal_ver);
            let refs: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors
                .iter()
                .map(|(n, dt, s, b)| (n.as_str(), *dt, s.as_slice(), b.clone()))
                .collect();
            write_file(&dir.join("model.safetensors"), &refs);
            let msg = convert_dir(&dir, &dir.join("drafter.gguf"))
                .expect_err("must be refused")
                .to_string();
            let _ = std::fs::remove_dir_all(dir);
            msg
        };

        let base = dspine_config_full_attn();
        let e = convert_err("dspine-nonzero-out", &base, false, 1);
        assert!(e.contains("out.weight") && e.contains("non-zero"), "{e}");

        let e = convert_err("dspine-bad-ver", &base, true, 2);
        assert!(
            e.contains("draft_block_causal_version") && e.contains("version 2"),
            "{e}"
        );

        let mut disagree = base.clone();
        disagree["code_bridge_second_pass"] = serde_json::json!({
            "mode": "serial_table",
            "first_exact": 1,
            "head_topk": 8
        });
        let e = convert_err("dspine-disagree", &disagree, true, 1);
        assert!(e.contains("disagree"), "{e}");

        let mut bad_tk = base.clone();
        bad_tk["dflash_config"]["selected_prefix_last_write_refine"]["head_topk"] =
            serde_json::json!(6);
        let e = convert_err("dspine-bad-tk", &bad_tk, true, 1);
        assert!(e.contains("power of two"), "{e}");

        let mut sw_false = base.clone();
        sw_false["sliding_window"] = serde_json::json!(8);
        let e = convert_err("dspine-sw-false", &sw_false, true, 1);
        assert!(
            e.contains("sliding_window") && e.contains("use_sliding_window") && e.contains("false"),
            "{e}"
        );

        let mut tied = base;
        tied["tie_word_embeddings"] = serde_json::json!(true);
        let e = convert_err("dspine-tied", &tied, true, 1);
        assert!(e.contains("tie_word_embeddings"), "{e}");
    }
}
