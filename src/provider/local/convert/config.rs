//! GGUF metadata from `config.json`.
//!
//! A port of the pinned llama.cpp converter's `set_gguf_parameters` (the
//! generic `TextModel` one and each family's override) and of the extra
//! tensors it generates (`generate_extra_tensors`: Llama 3 and Gemma 4
//! `rope_freqs`, Phi 3 LongRoPE factors). Every key `Hparams::from_metadata`
//! reads is produced; [`check`] loads the result through it so a gap fails the
//! conversion naming the key.

use crate::provider::local::gguf::Value;
use serde_json::{Map, Value as Json};

use super::arch::ConvArch;
use super::cfg_err;
use crate::error::CucaError;
use crate::provider::local::WeightQuant;
use crate::provider::local::model::params::Hparams;

/// The merged hyperparameters of a checkpoint, as the converter's
/// `self.hparams` and `self.rope_parameters`.
#[derive(Debug, Clone)]
pub(crate) struct Hp {
    map: Map<String, Json>,
    /// `rope_parameters` (or `rope_scaling`) with the converter's mirrored
    /// entries.
    rope: Map<String, Json>,
}

impl Hp {
    /// The top-level config with `text_config` merged over it (multimodal
    /// checkpoints), and the converter's rope-parameter normalisation.
    pub(crate) fn new(config: &Json) -> Result<Self, CucaError> {
        let Json::Object(top) = config else {
            return Err(cfg_err("config.json is not a JSON object"));
        };
        let mut map = top.clone();
        if let Some(Json::Object(text)) = top.get("text_config") {
            for (k, v) in text {
                map.insert(k.clone(), v.clone());
            }
        }
        let mut hp = Self {
            map,
            rope: Map::new(),
        };
        let mut rope = match hp
            .map
            .get("rope_parameters")
            .or_else(|| hp.map.get("rope_scaling"))
        {
            Some(Json::Object(m)) => m.clone(),
            _ => Map::new(),
        };
        let rope_theta = hp
            .find(&[
                "global_rope_theta",
                "rope_global_theta",
                "rope_theta_global",
                "rope_theta",
                "rotary_emb_base",
            ])
            .cloned();
        let local_theta = hp
            .find(&[
                "local_rope_theta",
                "rope_local_theta",
                "rope_theta_local",
                "swa_rope_theta",
                "rope_local_base_freq",
            ])
            .cloned();
        let rope_type = rope.get("type").filter(|t| !t.is_null()).cloned();
        if !rope.contains_key("full_attention") && !rope.contains_key("sliding_attention") {
            if let Some(t) = local_theta {
                rope.insert(
                    "sliding_attention".into(),
                    serde_json::json!({ "rope_theta": t }),
                );
            }
            let mirrored = [
                ("rope_theta", rope_theta),
                ("rope_type", rope_type),
                (
                    "partial_rotary_factor",
                    hp.find(&["partial_rotary_factor", "rope_pct", "rope_percent"])
                        .cloned(),
                ),
                (
                    "original_max_position_embeddings",
                    hp.find(&["original_max_position_embeddings"]).cloned(),
                ),
            ];
            for (key, value) in mirrored {
                if let Some(v) = value {
                    rope.entry(key).or_insert(v);
                }
            }
        }
        hp.rope = rope;
        Ok(hp)
    }

    /// Fill what `transformers`' config classes default and the converter
    /// therefore sees (it loads `AutoConfig(...).to_dict()`): the head width
    /// of Llama-family and Qwen 3 configs and the RMSNorm epsilon.
    pub(crate) fn with_defaults(mut self, arch: ConvArch) -> Self {
        let (head_dim, eps) = match arch {
            ConvArch::Llama | ConvArch::Mistral3 => (
                match (self.n_embd(), self.n_head()) {
                    (Ok(e), Ok(h)) if h > 0 => Some(e / h),
                    _ => None,
                },
                1e-6,
            ),
            ConvArch::Qwen3 | ConvArch::Qwen3Moe => (Some(128), 1e-6),
            ConvArch::Smollm3 | ConvArch::Qwen2 | ConvArch::Granite | ConvArch::Mistral4 => {
                (None, 1e-6)
            }
            ConvArch::Phi3 | ConvArch::Olmo2 | ConvArch::Olmoe | ConvArch::Ernie4_5 => (None, 1e-5),
            _ => return self,
        };
        for (key, value) in [
            ("head_dim", head_dim.map(Json::from)),
            ("rms_norm_eps", Some(Json::from(eps))),
        ] {
            if let Some(value) = value {
                let slot = self.map.entry(key).or_insert(Json::Null);
                if slot.is_null() {
                    *slot = value;
                }
            }
        }
        self
    }

    /// The value of the first of `keys` present (`find_hparam`); a present
    /// `null` is `None`.
    pub(crate) fn find(&self, keys: &[&str]) -> Option<&Json> {
        keys.iter()
            .find_map(|k| self.map.get(*k))
            .filter(|v| !v.is_null())
    }

    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        self.map.get(key).filter(|v| !v.is_null())
    }

    fn flag(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(Json::as_bool)
    }

    pub(crate) fn uint(&self, keys: &[&str]) -> Option<usize> {
        self.find(keys).and_then(as_usize)
    }

    pub(crate) fn float(&self, keys: &[&str]) -> Option<f64> {
        self.find(keys).and_then(Json::as_f64)
    }

    pub(crate) fn req_uint(&self, keys: &[&str]) -> Result<usize, CucaError> {
        self.uint(keys)
            .ok_or_else(|| cfg_err(format!("config.json has none of {keys:?}")))
    }

    /// A truthy value as the converter's `if value:` sees it.
    fn truthy_f(&self, key: &str) -> Option<f64> {
        self.float(&[key]).filter(|v| *v != 0.0)
    }

    /// `rope_parameters.get("full_attention", rope_parameters)`.
    pub(crate) fn rope_full(&self) -> &Map<String, Json> {
        match self.rope.get("full_attention") {
            Some(Json::Object(m)) => m,
            _ => &self.rope,
        }
    }

    /// `rope_parameters["full_attention"]["rope_theta"]`.
    fn rope_theta(&self) -> Option<f64> {
        self.rope_full().get("rope_theta").and_then(Json::as_f64)
    }

    /// `rope_parameters` itself (the converter's `self.rope_parameters`).
    pub(crate) fn rope_all(&self) -> &Map<String, Json> {
        &self.rope
    }

    /// `rope_parameters["sliding_attention"]["rope_theta"]`.
    fn rope_sliding_theta(&self) -> Option<f64> {
        self.rope
            .get("sliding_attention")?
            .get("rope_theta")?
            .as_f64()
    }

    /// Number of transformer blocks of the checkpoint, the multi-token
    /// prediction blocks not included.
    pub(crate) fn n_layer(&self) -> Result<usize, CucaError> {
        self.req_uint(&["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
    }

    pub(crate) fn n_head(&self) -> Result<usize, CucaError> {
        self.req_uint(&["num_attention_heads", "n_head", "n_heads"])
    }

    pub(crate) fn n_embd(&self) -> Result<usize, CucaError> {
        self.req_uint(&["hidden_size", "n_embd", "dim"])
    }

    /// `head_dim`, or the hidden size over the head count.
    pub(crate) fn head_dim(&self) -> Result<usize, CucaError> {
        match self.uint(&["head_dim"]) {
            Some(d) => Ok(d),
            None => Ok(self.n_embd()? / self.n_head()?.max(1)),
        }
    }
}

fn as_usize(v: &Json) -> Option<usize> {
    v.as_u64().map(|n| n as usize).or_else(|| {
        v.as_f64()
            .filter(|f| f.fract() == 0.0 && *f >= 0.0)
            .map(|f| f as usize)
    })
}

/// Ordered GGUF key/value pairs under one architecture prefix.
struct Kv {
    prefix: &'static str,
    items: Vec<(String, Value)>,
}

impl Kv {
    fn new(arch: ConvArch) -> Self {
        Self {
            prefix: arch.gguf_name(),
            items: Vec::new(),
        }
    }

    /// Set a full key, replacing an earlier value (the converter's later
    /// override of an earlier write).
    fn raw(&mut self, key: &str, value: Value) {
        match self.items.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => self.items.push((key.to_string(), value)),
        }
    }

    fn arch_key(&mut self, key: &str, value: Value) {
        let full = format!("{}.{key}", self.prefix);
        self.raw(&full, value);
    }

    fn u32(&mut self, key: &str, v: usize) -> Result<(), CucaError> {
        let v = u32::try_from(v).map_err(|_| cfg_err(format!("{key}: {v} does not fit u32")))?;
        self.arch_key(key, Value::U32(v));
        Ok(())
    }

    /// [`Kv::u32`] of a value the config may not have.
    fn u32_opt(&mut self, key: &str, v: Option<usize>) -> Result<(), CucaError> {
        v.map_or(Ok(()), |v| self.u32(key, v))
    }

    fn f32(&mut self, key: &str, v: f64) {
        self.arch_key(key, Value::F32(v as f32));
    }

    /// [`Kv::f32`] of a value the config may not have.
    fn f32_opt(&mut self, key: &str, v: Option<f64>) {
        if let Some(v) = v {
            self.f32(key, v);
        }
    }

    fn bool(&mut self, key: &str, v: bool) {
        self.arch_key(key, Value::Bool(v));
    }

    fn i32s(&mut self, key: &str, v: &[i64]) {
        self.arch_key(
            key,
            Value::Array(v.iter().map(|x| Value::I32(*x as i32)).collect()),
        );
    }

    fn bools(&mut self, key: &str, v: &[bool]) {
        self.arch_key(
            key,
            Value::Array(v.iter().map(|x| Value::Bool(*x)).collect()),
        );
    }
}

/// A tensor the converter generates rather than reads.
#[derive(Debug)]
pub(crate) struct ExtraTensor {
    pub(crate) name: &'static str,
    pub(crate) data: Vec<f32>,
}

/// The result of [`build`].
#[derive(Debug)]
pub(crate) struct ModelMeta {
    pub(crate) kv: Vec<(String, Value)>,
    pub(crate) extra: Vec<ExtraTensor>,
}

/// `LlamaFileType` of a weight precision.
pub(crate) fn file_type(q: WeightQuant) -> u32 {
    match q {
        WeightQuant::F16 => 1,
        WeightQuant::Q4_0 => 2,
        WeightQuant::Q8_0 => 7,
    }
}

/// Which rope scaling types are written to the file and which are not.
fn rope_scaling(arch: ConvArch, hp: &Hp, kv: &mut Kv) -> Result<(), CucaError> {
    let p = hp.rope_full();
    let Some(kind) = p.get("rope_type").and_then(Json::as_str) else {
        return Ok(());
    };
    let num = |key: &str| p.get(key).and_then(Json::as_f64);
    match (kind, num("factor")) {
        ("default" | "" | "llama3", _) => {}
        ("linear", Some(factor)) => {
            kv.arch_key("rope.scaling.type", Value::String("linear".into()));
            kv.f32("rope.scaling.factor", factor);
        }
        ("yarn", Some(factor)) => {
            kv.arch_key("rope.scaling.type", Value::String("yarn".into()));
            kv.f32("rope.scaling.factor", factor);
            let orig = p
                .get("original_max_position_embeddings")
                .and_then(as_usize)
                .ok_or_else(|| {
                    cfg_err("YaRN rope scaling lacks original_max_position_embeddings")
                })?;
            kv.u32("rope.scaling.original_context_length", orig)?;
            if let Some(t) = p.get("truncate").and_then(Json::as_bool) {
                kv.bool("rope.scaling.yarn_truncate", t);
            }
            kv.f32_opt("rope.scaling.yarn_ext_factor", num("extrapolation_factor"));
            if p.contains_key("attention_factor") || p.contains_key("attn_factor") {
                return Err(cfg_err(
                    "an explicit YaRN attention_factor is not supported by the engine",
                ));
            }
            kv.f32_opt("rope.scaling.yarn_beta_fast", num("beta_fast"));
            kv.f32_opt("rope.scaling.yarn_beta_slow", num("beta_slow"));
        }
        ("su" | "longrope", _) => {
            kv.arch_key("rope.scaling.type", Value::String("longrope".into()));
        }
        ("proportional", _) if arch == ConvArch::Gemma4 || arch == ConvArch::Gemma4Assistant => {}
        (other, _) => {
            return Err(cfg_err(format!(
                "rope scaling type `{other}` is not supported"
            )));
        }
    }
    Ok(())
}

/// The generic `TextModel.set_gguf_parameters`.
fn base(arch: ConvArch, hp: &Hp, kv: &mut Kv, n_block: usize) -> Result<(), CucaError> {
    kv.u32("block_count", n_block)?;
    kv.u32_opt(
        "context_length",
        hp.uint(&[
            "max_position_embeddings",
            "n_ctx",
            "n_positions",
            "max_length",
            "max_sequence_length",
            "model_max_length",
        ]),
    )?;
    kv.u32_opt(
        "embedding_length",
        hp.uint(&["hidden_size", "n_embd", "dim"]),
    )?;
    kv.u32_opt(
        "feed_forward_length",
        hp.uint(&[
            "prefix_dense_intermediate_size",
            "dense_intermediate_size",
            "intermediate_size",
            "n_inner",
            "hidden_dim",
        ]),
    )?;
    kv.u32_opt(
        "attention.head_count",
        hp.uint(&["num_attention_heads", "n_head", "n_heads"]),
    )?;
    kv.u32_opt(
        "attention.head_count_kv",
        hp.uint(&["num_key_value_heads", "n_kv_heads"]),
    )?;
    rope_scaling(arch, hp, kv)?;
    if let Some(Json::Array(sections)) = hp.rope_all().get("mrope_section") {
        let mut s: Vec<i64> = sections.iter().filter_map(Json::as_i64).collect();
        while s.len() < 4 {
            s.push(0);
        }
        kv.i32s("rope.dimension_sections", &s[..4]);
    }
    kv.f32_opt("rope.freq_base", hp.rope_theta());
    kv.f32_opt("rope.freq_base_swa", hp.rope_sliding_theta());
    kv.f32_opt(
        "attention.layer_norm_rms_epsilon",
        hp.float(&["rms_norm_eps", "norm_eps"]),
    );
    kv.f32_opt(
        "attention.layer_norm_epsilon",
        hp.float(&["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"]),
    );
    kv.u32_opt(
        "expert_count",
        hp.uint(&["num_local_experts", "num_experts", "n_routed_experts"]),
    )?;
    kv.u32_opt(
        "expert_used_count",
        hp.uint(&[
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
        ]),
    )?;
    kv.u32_opt("expert_group_count", hp.uint(&["n_group"]))?;
    kv.u32_opt("expert_group_used_count", hp.uint(&["topk_group"]))?;
    if let Some(d) = hp.uint(&["head_dim"]) {
        kv.u32("attention.key_length", d)?;
        kv.u32("attention.value_length", d)?;
    }
    Ok(())
}

/// `DeepseekV2Model.set_gguf_parameters` as `Mistral4Model` inherits it:
/// multi-head latent attention stored as multi-query attention over the
/// compressed cache (one KV head, `kv_lora + rope` wide keys, `kv_lora` wide
/// values), with the per-head sizes of the uncompressed attention in the
/// `*_mla` keys.
fn mla(arch: ConvArch, hp: &Hp, kv: &mut Kv, n_block: usize) -> Result<(), CucaError> {
    base(arch, hp, kv, n_block)?;
    kv.u32("attention.head_count_kv", 1)?;
    let has_moe = hp.get("n_routed_experts").is_some();
    // No MoE: every layer is dense; MoE: none is unless the config says.
    let dense = match hp.uint(&["first_k_dense_replace"]) {
        Some(n) => n,
        None if has_moe => 0,
        None => hp.n_layer()?,
    };
    kv.u32("leading_dense_block_count", dense)?;
    kv.u32("vocab_size", hp.req_uint(&["vocab_size"])?)?;
    kv.u32_opt("attention.q_lora_rank", hp.uint(&["q_lora_rank"]))?;
    let lora = hp.uint(&["kv_lora_rank"]).unwrap_or(512);
    let nope = hp.req_uint(&["qk_nope_head_dim"])?;
    let rope = hp.req_uint(&["qk_rope_head_dim"])?;
    kv.u32("attention.kv_lora_rank", lora)?;
    kv.u32("attention.key_length", lora + rope)?;
    kv.u32("attention.value_length", lora)?;
    kv.u32("attention.key_length_mla", nope + rope)?;
    kv.u32("attention.value_length_mla", hp.req_uint(&["v_head_dim"])?)?;
    kv.u32(
        "expert_feed_forward_length",
        hp.req_uint(&["moe_intermediate_size", "intermediate_size"])?,
    )?;
    kv.u32_opt("expert_count", hp.uint(&["n_routed_experts"]))?;
    kv.u32(
        "expert_shared_count",
        hp.uint(&["n_shared_experts"]).unwrap_or(0),
    )?;
    kv.f32_opt("expert_weights_scale", hp.float(&["routed_scaling_factor"]));
    if hp.flag("norm_topk_prob") == Some(true) {
        kv.bool("expert_weights_norm", true);
    }
    kv.u32("rope.dimension_count", rope)?;
    // llama.cpp's DeepSeek convention: the multiplier the engine applies is
    // `0.1 * mscale_all_dim`, not `mscale_all_dim` itself.
    if let Some(m) = hp.rope_all().get("mscale_all_dim").and_then(Json::as_f64) {
        kv.f32("rope.scaling.yarn_log_multiplier", 0.1 * m);
    }
    Ok(())
}

/// `Ministral3Model.set_gguf_parameters`: for a `ministral3` text model the
/// YaRN `mscale_all_dim` becomes `yarn_log_multiplier` and
/// `llama_4_scaling_beta` the attention temperature scale.
fn mistral3_rope(hp: &Hp, kv: &mut Kv) -> Result<(), CucaError> {
    if hp.get("model_type").and_then(Json::as_str) != Some("ministral3") {
        return Ok(());
    }
    let p = hp.rope_full();
    if p.get("rope_type").and_then(Json::as_str) != Some("yarn") {
        return Err(cfg_err("ministral3 must have a yarn rope_type"));
    }
    let get = |key: &str| {
        p.get(key)
            .and_then(Json::as_f64)
            .ok_or_else(|| cfg_err(format!("ministral3 rope_parameters lack `{key}`")))
    };
    kv.f32("rope.scaling.yarn_log_multiplier", get("mscale_all_dim")?);
    kv.f32("attention.temperature_scale", get("llama_4_scaling_beta")?);
    Ok(())
}

/// Llama-style extras: `vocab_size` and the rotary dimension count.
fn llama_extras(hp: &Hp, kv: &mut Kv) -> Result<(), CucaError> {
    kv.u32("vocab_size", hp.req_uint(&["vocab_size"])?)?;
    kv.u32("rope.dimension_count", hp.head_dim()?)
}

/// `LlamaModel.generate_extra_tensors`: the `rope_freqs` factors of Llama 3
/// rope scaling.
fn llama3_freqs(hp: &Hp) -> Result<Vec<ExtraTensor>, CucaError> {
    let p = hp.rope_full();
    if !p
        .get("rope_type")
        .and_then(Json::as_str)
        .is_some_and(|t| t.eq_ignore_ascii_case("llama3"))
    {
        return Ok(Vec::new());
    }
    let f = |k: &str, d: f64| p.get(k).and_then(Json::as_f64).unwrap_or(d);
    let base = f("rope_theta", 10_000.0);
    let dim = hp.head_dim()?;
    let factor = f("factor", 8.0);
    let low = f("low_freq_factor", 1.0);
    let high = f("high_freq_factor", 4.0);
    let old = f("original_max_position_embeddings", 8192.0);
    let low_wavelen = old / low;
    let high_wavelen = old / high;
    let data = (0..dim / 2)
        .map(|i| {
            // The converter computes the frequencies in f32.
            let freq = 1.0f32 / (base as f32).powf((2 * i) as f32 / dim as f32);
            let wavelen = 2.0 * std::f64::consts::PI / f64::from(freq);
            let r = if wavelen < high_wavelen {
                1.0
            } else if wavelen > low_wavelen {
                factor
            } else {
                let smooth = (old / wavelen - low) / (high - low);
                1.0 / ((1.0 - smooth) / factor + smooth)
            };
            r as f32
        })
        .collect();
    Ok(vec![ExtraTensor {
        name: "rope_freqs.weight",
        data,
    }])
}

/// `final_logit_softcapping`, when set and non-zero.
fn final_softcap(hp: &Hp, kv: &mut Kv) {
    kv.f32_opt(
        "final_logit_softcapping",
        hp.truthy_f("final_logit_softcapping"),
    );
}

/// `per_layer_config[i][key]` of the first layer of `kind`, as an integer.
fn first_layer_uint(hp: &Hp, kind: &str, key: &str) -> Option<usize> {
    let layer_types = hp.get("layer_types")?.as_array()?;
    let Json::Object(cfg) = hp.get("per_layer_config")? else {
        return None;
    };
    for (idx, layer) in cfg {
        let i: usize = idx.parse().ok()?;
        if layer_types.get(i).and_then(Json::as_str) == Some(kind)
            && let Some(v) = layer.get(key)
        {
            return as_usize(v);
        }
    }
    None
}

/// `Gemma4Model.set_gguf_parameters` over the Gemma 3 base; returns the
/// extra tensors (`rope_freqs` of the full-attention layers).
fn gemma4(
    arch: ConvArch,
    hp: &Hp,
    kv: &mut Kv,
    n_block: usize,
) -> Result<Vec<ExtraTensor>, CucaError> {
    if hp.flag("enable_moe_block").unwrap_or(false)
        || hp.uint(&["num_experts"]).is_some_and(|n| n > 0)
    {
        return Err(cfg_err(
            "Gemma 4 mixture-of-experts checkpoints are not supported by the engine",
        ));
    }
    base(arch, hp, kv, n_block)?;
    gemma3_overrides(hp, kv)?;

    let shared = hp.req_uint(&["num_kv_shared_layers"])?;
    kv.u32("attention.shared_kv_layers", shared)?;
    kv.u32(
        "embedding_length_per_layer_input",
        hp.uint(&["hidden_size_per_layer_input"]).unwrap_or(0),
    )?;
    let swa: Vec<bool> = hp
        .get("layer_types")
        .and_then(Json::as_array)
        .ok_or_else(|| cfg_err("config.json has no `layer_types`"))?
        .iter()
        .filter_map(Json::as_str)
        .map(|t| t == "sliding_attention")
        .collect();
    kv.bools("attention.sliding_window_pattern", &swa);

    let head_dim_full = hp
        .uint(&["global_head_dim"])
        .or_else(|| first_layer_uint(hp, "full_attention", "head_dim"))
        .ok_or_else(|| cfg_err("cannot find the full-attention head dimension"))?;
    let head_dim_swa = hp.req_uint(&["head_dim"])?;
    kv.u32("attention.key_length", head_dim_full)?;
    kv.u32("attention.value_length", head_dim_full)?;
    kv.u32("attention.key_length_swa", head_dim_swa)?;
    kv.u32("attention.value_length_swa", head_dim_swa)?;

    kv.u32_opt(
        "expert_feed_forward_length",
        hp.uint(&["expert_intermediate_size", "moe_intermediate_size"]),
    )?;
    if hp.flag("use_double_wide_mlp").unwrap_or(false) {
        let n_ff = hp.req_uint(&["intermediate_size"])?;
        let first_shared = n_block - shared;
        let arr: Vec<i64> = (0..n_block)
            .map(|il| {
                if il < first_shared {
                    n_ff as i64
                } else {
                    2 * n_ff as i64
                }
            })
            .collect();
        kv.i32s("feed_forward_length", &arr);
    }
    let kv_full = hp
        .uint(&["num_global_key_value_heads"])
        .or_else(|| first_layer_uint(hp, "full_attention", "num_key_value_heads"));
    if let (Some(full), Some(swa_heads)) = (kv_full, hp.uint(&["num_key_value_heads"])) {
        let arr: Vec<i64> = swa
            .iter()
            .map(|&s| if s { swa_heads as i64 } else { full as i64 })
            .collect();
        kv.i32s("attention.head_count_kv", &arr);
    }
    let partial_swa = hp
        .rope_all()
        .get("partial_rotary_factor")
        .and_then(Json::as_f64)
        .unwrap_or(1.0);
    kv.u32("rope.dimension_count", head_dim_full)?;
    kv.u32(
        "rope.dimension_count_swa",
        (head_dim_swa as f64 * partial_swa) as usize,
    )?;

    // `generate_extra_tensors`: the full layers use "proportional" RoPE with a
    // partial rotary factor; ggml's NEOX rotates the first dims, so the
    // unrotated ones get a huge frequency factor.
    let full = hp
        .rope_all()
        .get("full_attention")
        .ok_or_else(|| cfg_err("Gemma 4 rope_parameters have no full_attention entry"))?;
    if full.get("rope_type").and_then(Json::as_str) != Some("proportional") {
        return Err(cfg_err(
            "Gemma 4 full-attention rope_type must be `proportional`",
        ));
    }
    let partial_full = full
        .get("partial_rotary_factor")
        .and_then(Json::as_f64)
        .ok_or_else(|| cfg_err("Gemma 4 full-attention rope has no partial_rotary_factor"))?;
    let n_rot = (head_dim_full as f64 * partial_full / 2.0) as usize;
    let n_unrot = head_dim_full / 2 - n_rot;
    let mut data = vec![1.0f32; n_rot];
    data.extend(std::iter::repeat_n(1e30f32, n_unrot));
    Ok(vec![ExtraTensor {
        name: "rope_freqs.weight",
        data,
    }])
}

/// The overrides `Gemma3Model.set_gguf_parameters` applies over the base.
fn gemma3_overrides(hp: &Hp, kv: &mut Kv) -> Result<(), CucaError> {
    kv.u32(
        "context_length",
        hp.uint(&["max_position_embeddings"]).unwrap_or(131_072),
    )?;
    kv.u32(
        "attention.head_count",
        hp.uint(&["num_attention_heads"]).unwrap_or(8),
    )?;
    kv.f32(
        "attention.layer_norm_rms_epsilon",
        hp.float(&["rms_norm_eps"]).unwrap_or(1e-6),
    );
    let head_dim = hp.uint(&["head_dim"]).unwrap_or(256);
    kv.u32("attention.key_length", head_dim)?;
    kv.u32("attention.value_length", head_dim)?;
    kv.f32("rope.freq_base", hp.rope_theta().unwrap_or(1_000_000.0));
    if hp.get("attn_logit_softcapping").is_some() {
        return Err(cfg_err(
            "Gemma 3 checkpoints have no attn_logit_softcapping",
        ));
    }
    final_softcap(hp, kv);
    if hp.uint(&["sliding_window_pattern"]) != Some(1) {
        kv.u32(
            "attention.sliding_window",
            hp.req_uint(&["sliding_window"])?,
        )?;
    }
    kv.u32(
        "attention.head_count_kv",
        hp.uint(&["num_key_value_heads"]).unwrap_or(4),
    )
}

/// `Phi3MiniModel.set_gguf_parameters` and its LongRoPE factors.
fn phi3(hp: &Hp, kv: &mut Kv, n_block: usize) -> Result<Vec<ExtraTensor>, CucaError> {
    let n_embd = hp.n_embd()?;
    let n_head = hp.n_head()?;
    let rope = hp.rope_full();
    let orig = rope
        .get("original_max_position_embeddings")
        .and_then(as_usize)
        .ok_or_else(|| cfg_err("Phi 3 rope parameters lack original_max_position_embeddings"))?;
    let max_pos = hp.req_uint(&["n_positions", "max_position_embeddings"])?;
    let rot_pct = rope
        .get("partial_rotary_factor")
        .and_then(Json::as_f64)
        .unwrap_or(1.0);
    let rope_dims = (rot_pct * n_embd as f64) as usize / n_head;

    kv.u32("context_length", max_pos)?;
    kv.u32("rope.scaling.original_context_length", orig)?;
    kv.u32("embedding_length", n_embd)?;
    kv.u32("feed_forward_length", hp.req_uint(&["intermediate_size"])?)?;
    kv.u32("block_count", n_block)?;
    kv.u32("attention.head_count", n_head)?;
    kv.u32(
        "attention.head_count_kv",
        hp.req_uint(&["num_key_value_heads", "n_head_kv"])?,
    )?;
    kv.f32(
        "attention.layer_norm_rms_epsilon",
        hp.float(&["rms_norm_eps"])
            .ok_or_else(|| cfg_err("config.json has no rms_norm_eps"))?,
    );
    kv.u32("rope.dimension_count", rope_dims)?;
    kv.f32(
        "rope.freq_base",
        hp.rope_theta()
            .ok_or_else(|| cfg_err("Phi 3 rope parameters lack rope_theta"))?,
    );
    kv.u32(
        "attention.sliding_window",
        hp.uint(&["sliding_window"]).unwrap_or(0),
    )?;

    let floats = |k: &str| -> Option<Vec<f32>> {
        rope.get(k).and_then(Json::as_array).map(|a| {
            a.iter()
                .filter_map(Json::as_f64)
                .map(|x| x as f32)
                .collect()
        })
    };
    let Some(long) = floats("long_factor").filter(|l| !l.is_empty()) else {
        return Ok(Vec::new());
    };
    let short =
        floats("short_factor").ok_or_else(|| cfg_err("Phi 3 rope parameters lack short_factor"))?;
    let scale = max_pos as f64 / orig as f64;
    let kind = rope
        .get("rope_type")
        .and_then(Json::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let attn_factor = match kind.as_str() {
        "su" | "longrope" if scale > 1.0 => (1.0 + scale.ln() / (orig as f64).ln()).sqrt(),
        "yarn" if scale > 1.0 => 0.1 * scale.ln() + 1.0,
        "su" | "longrope" | "yarn" => 1.0,
        "" => return Err(cfg_err("Phi 3 rope parameters lack rope_type")),
        other => {
            return Err(cfg_err(format!(
                "Phi 3 rope scaling type `{other}` is not supported"
            )));
        }
    };
    kv.f32("rope.scaling.attn_factor", attn_factor);
    if long.len() != short.len() || long.len() != rope_dims / 2 {
        return Err(cfg_err(format!(
            "Phi 3 rope factor lengths {}/{} must be {}",
            long.len(),
            short.len(),
            rope_dims / 2
        )));
    }
    Ok(vec![
        ExtraTensor {
            name: "rope_factors_long.weight",
            data: long,
        },
        ExtraTensor {
            name: "rope_factors_short.weight",
            data: short,
        },
    ])
}

/// The remaining `MuseGlimmerModel.set_gguf_parameters` keys, after refusing
/// configurations whose behaviour the engine fixes per architecture.
fn muse_glimmer(hp: &Hp, kv: &mut Kv) -> Result<(), CucaError> {
    let n_layer = hp.n_layer()?;
    let window = hp.req_uint(&["sliding_window"])?;
    let Some(Json::Array(types)) = hp.get("layer_types") else {
        return Err(cfg_err("config.json has no layer_types"));
    };
    if types.len() != n_layer {
        return Err(cfg_err(format!(
            "layer_types has {} entries, expected num_hidden_layers ({n_layer})",
            types.len()
        )));
    }
    let sliding: Vec<bool> = types
        .iter()
        .map(|t| t.as_str() == Some("sliding_attention"))
        .collect();
    if types
        .iter()
        .any(|t| !matches!(t.as_str(), Some("sliding_attention" | "full_attention")))
    {
        return Err(cfg_err(
            "Muse Glimmer layer_types other than sliding_attention and full_attention",
        ));
    }
    // The engine rotates exactly the sliding layers, with the global base.
    let theta = hp.rope_theta();
    if let Some(Json::Array(thetas)) = hp.get("layer_rope_theta") {
        let ok = thetas.len() == n_layer
            && thetas.iter().zip(&sliding).all(|(t, s)| {
                let t = t.as_f64().unwrap_or(f64::NAN);
                if *s { Some(t) == theta } else { t == 0.0 }
            });
        if !ok {
            return Err(cfg_err(
                "Muse Glimmer layer_rope_theta differs from rotary embeddings on the sliding layers only, with rope_theta",
            ));
        }
    }
    if hp.rope_full().get("rope_type").and_then(Json::as_str) != Some("default") {
        return Err(cfg_err("Muse Glimmer rope scaling is not supported"));
    }
    if hp
        .get("hidden_activation")
        .or_else(|| hp.get("hidden_act"))
        .and_then(Json::as_str)
        .is_some_and(|a| a != "silu")
    {
        return Err(cfg_err("Muse Glimmer activations other than silu"));
    }
    if hp.flag("attention_bias") == Some(true) {
        return Err(cfg_err("Muse Glimmer attention biases are not supported"));
    }
    if hp.float(&["post_norm_eps"]).is_some_and(|e| e != 1e-8) {
        return Err(cfg_err(
            "Muse Glimmer post_norm_eps other than 1e-8 is not supported: the engine fixes it",
        ));
    }
    final_softcap(hp, kv);
    kv.f32(
        "logit_scale",
        hp.float(&["output_multiplier"])
            .ok_or_else(|| cfg_err("config.json has no output_multiplier"))?,
    );
    kv.u32("attention.sliding_window", window)?;
    kv.bools("attention.sliding_window_pattern", &sliding);
    Ok(())
}

/// Refuse features the engine's architecture does not implement, so a
/// converted file never silently computes something else.
fn guard(arch: ConvArch, hp: &Hp) -> Result<(), CucaError> {
    let sliding_layers = || {
        hp.get("layer_types")
            .and_then(Json::as_array)
            .is_some_and(|l| l.iter().any(|t| t.as_str() == Some("sliding_attention")))
    };
    // FP8 weights are dequantized by one scale per tensor (or per expert);
    // refuse block scales before any weight is read.
    if hp
        .get("quantization_config")
        .and_then(|q| q.get("weight_block_size"))
        .is_some_and(|b| !b.is_null())
    {
        return Err(cfg_err(
            "block-scaled FP8 checkpoints (quantization_config.weight_block_size) are not supported",
        ));
    }
    match arch {
        ConvArch::Olmo2 if sliding_layers() => {
            return Err(cfg_err(
                "OLMo 3 sliding-window layers are not supported by the engine",
            ));
        }
        ConvArch::Qwen3Moe => {
            if hp.flag("norm_topk_prob") != Some(true) {
                return Err(cfg_err(
                    "Qwen3 MoE with norm_topk_prob = false is not supported: the engine renormalises the selected experts",
                ));
            }
            if hp
                .uint(&["shared_expert_intermediate_size"])
                .is_some_and(|n| n > 0)
            {
                return Err(cfg_err("shared experts are not supported for Qwen3 MoE"));
            }
            if hp.uint(&["mlp_only_layers"]).is_some()
                || hp
                    .get("mlp_only_layers")
                    .and_then(Json::as_array)
                    .is_some_and(|a| !a.is_empty())
            {
                return Err(cfg_err("Qwen3 MoE mlp_only_layers are not supported"));
            }
            if hp.uint(&["decoder_sparse_step"]).is_some_and(|n| n != 1) {
                return Err(cfg_err("Qwen3 MoE decoder_sparse_step must be 1"));
            }
        }
        ConvArch::Olmoe if hp.flag("norm_topk_prob") == Some(true) => {
            return Err(cfg_err(
                "OLMoE with norm_topk_prob = true is not supported: the engine keeps the raw softmax weights",
            ));
        }
        ConvArch::Smollm3 => {
            // The engine fixes the NoPE pattern: every fourth layer.
            if let Some(Json::Array(layers)) = hp.get("no_rope_layers") {
                let expect = |i: usize| i64::from(!(i + 1).is_multiple_of(4));
                if layers
                    .iter()
                    .enumerate()
                    .any(|(i, v)| v.as_i64() != Some(expect(i)))
                {
                    return Err(cfg_err(
                        "SmolLM 3 no_rope_layers differ from every fourth layer",
                    ));
                }
            }
        }
        ConvArch::Gemma3 => {
            if let Some(Json::Array(types)) = hp.get("layer_types") {
                let ok = types.iter().enumerate().all(|(i, t)| {
                    (t.as_str() == Some("full_attention")) == (i + 1).is_multiple_of(6)
                });
                if !ok {
                    return Err(cfg_err(
                        "Gemma 3 layer_types differ from every sixth layer being full attention",
                    ));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Build the metadata of the checkpoint (everything but the tokenizer).
///
/// `name` becomes `general.name`; `n_mtp` is the number of
/// multi-token-prediction blocks whose tensors the checkpoint holds (Qwen 3.5
/// only).
pub(crate) fn build(
    arch: ConvArch,
    hp: &Hp,
    name: &str,
    quant: WeightQuant,
    n_mtp: usize,
) -> Result<ModelMeta, CucaError> {
    guard(arch, hp)?;
    let mut kv = Kv::new(arch);
    kv.raw(
        "general.architecture",
        Value::String(arch.gguf_name().into()),
    );
    kv.raw("general.type", Value::String("model".into()));
    kv.raw("general.name", Value::String(name.to_string()));
    let n_layer = hp.n_layer()?;
    let n_block = if arch.is_qwen35() {
        n_layer + n_mtp
    } else {
        n_layer
    };
    let mut extra = Vec::new();

    match arch {
        ConvArch::Llama | ConvArch::Smollm3 | ConvArch::Granite | ConvArch::Mistral3 => {
            // Granite ignores head_dim: the engine derives the head size
            // from the embedding width.
            let granite = (arch == ConvArch::Granite).then(|| {
                let mut g = hp.clone();
                g.map.remove("head_dim");
                g
            });
            let llama_hp = granite.as_ref().unwrap_or(hp);
            base(arch, llama_hp, &mut kv, n_block)?;
            llama_extras(llama_hp, &mut kv)?;
            if arch == ConvArch::Granite {
                for (src, key) in [
                    ("attention_multiplier", "attention.scale"),
                    ("embedding_multiplier", "embedding_scale"),
                    ("residual_multiplier", "residual_scale"),
                    ("logits_scaling", "logit_scale"),
                ] {
                    kv.f32_opt(key, llama_hp.truthy_f(src));
                }
            }
            extra = llama3_freqs(hp)?;
            if arch == ConvArch::Mistral3 {
                mistral3_rope(hp, &mut kv)?;
            }
        }
        ConvArch::Qwen2 | ConvArch::Qwen3 | ConvArch::Ernie4_5 => base(arch, hp, &mut kv, n_block)?,
        ConvArch::Qwen3Moe => {
            base(arch, hp, &mut kv, n_block)?;
            kv.u32_opt(
                "expert_feed_forward_length",
                hp.uint(&["moe_intermediate_size"]),
            )?;
        }
        ConvArch::Qwen35 | ConvArch::Qwen35Moe => {
            base(arch, hp, &mut kv, n_block)?;
            kv.u32_opt(
                "expert_feed_forward_length",
                hp.uint(&["moe_intermediate_size"]),
            )?;
            kv.u32_opt(
                "expert_shared_feed_forward_length",
                hp.uint(&["shared_expert_intermediate_size"]),
            )?;
            if n_mtp > 0 {
                kv.u32("nextn_predict_layers", n_mtp)?;
            }
            kv.u32("ssm.conv_kernel", hp.req_uint(&["linear_conv_kernel_dim"])?)?;
            kv.u32("ssm.state_size", hp.req_uint(&["linear_key_head_dim"])?)?;
            kv.u32("ssm.group_count", hp.req_uint(&["linear_num_key_heads"])?)?;
            let n_v = hp.req_uint(&["linear_num_value_heads"])?;
            kv.u32("ssm.time_step_rank", n_v)?;
            kv.u32(
                "ssm.inner_size",
                hp.req_uint(&["linear_value_head_dim"])? * n_v,
            )?;
            if let Some(Json::Array(types)) = hp.get("layer_types") {
                if types.len() != n_layer {
                    return Err(cfg_err(format!(
                        "layer_types has {} entries, expected num_hidden_layers ({n_layer})",
                        types.len()
                    )));
                }
                let mut rec: Vec<bool> = types
                    .iter()
                    .map(|t| t.as_str() == Some("linear_attention"))
                    .collect();
                rec.resize(n_block, false);
                kv.bools("attention.recurrent_layers", &rec);
            }
            kv.u32(
                "full_attention_interval",
                hp.uint(&["full_attention_interval"]).unwrap_or(4),
            )?;
            let rope_dim = hp.head_dim()?;
            let partial = hp
                .rope_all()
                .get("partial_rotary_factor")
                .and_then(Json::as_f64)
                .unwrap_or(0.25);
            kv.u32("rope.dimension_count", (rope_dim as f64 * partial) as usize)?;
            if !hp.rope_all().contains_key("mrope_section") {
                kv.i32s("rope.dimension_sections", &[11, 11, 10, 0]);
            }
        }
        ConvArch::Mistral4 => mla(arch, hp, &mut kv, n_block)?,
        // Converted by `deepseek41`, which writes its own metadata.
        ConvArch::Deepseek41 => {
            return Err(cfg_err(
                "DeepSeek-V4.1 checkpoints convert through convert::deepseek41",
            ));
        }
        ConvArch::Mistral4Eagle => {
            mla(arch, hp, &mut kv, n_block)?;
            kv.u32("embedding_length_out", hp.n_embd()?)?;
            // The Llama 4 query scaling is written out (the target's file
            // leaves it to the engine's default of the same two numbers): the
            // drafter must scale exactly as its target, which the engine
            // checks when it pairs them.
            let rope = hp.rope_full();
            if let (Some(beta), Some(length)) = (
                rope.get("llama_4_scaling_beta").and_then(Json::as_f64),
                rope.get("original_max_position_embeddings")
                    .and_then(as_usize),
            ) {
                kv.f32("attention.temperature_scale", beta);
                kv.u32("attention.temperature_length", length)?;
            }
        }
        ConvArch::Phi3 => extra = phi3(hp, &mut kv, n_block)?,
        ConvArch::Gemma3 => {
            base(arch, hp, &mut kv, n_block)?;
            gemma3_overrides(hp, &mut kv)?;
        }
        ConvArch::Gemma4 => extra = gemma4(arch, hp, &mut kv, n_block)?,
        ConvArch::Gemma4Assistant => {
            extra = gemma4(arch, hp, &mut kv, n_block)?;
            kv.u32(
                "embedding_length_out",
                hp.req_uint(&["backbone_hidden_size"])?,
            )?;
            kv.u32("nextn_predict_layers", n_block)?;
        }
        ConvArch::Olmo2 => {
            base(arch, hp, &mut kv, n_block)?;
            kv.u32_opt("attention.sliding_window", hp.uint(&["sliding_window"]))?;
        }
        ConvArch::Olmoe => {
            base(arch, hp, &mut kv, n_block)?;
            kv.f32("attention.layer_norm_rms_epsilon", 1e-5);
        }
        ConvArch::MuseGlimmer => {
            base(arch, hp, &mut kv, n_block)?;
            muse_glimmer(hp, &mut kv)?;
        }
    }
    kv.raw("general.quantization_version", Value::U32(2));
    kv.raw("general.file_type", Value::U32(file_type(quant)));
    Ok(ModelMeta {
        kv: kv.items,
        extra,
    })
}

/// `general.sampling.*` from `generation_config.json`, the way gguf-py's
/// `Metadata.load` and `Metadata.set_gguf_meta_model` write them (llama.cpp
/// master, and the pinned commit): `sequence` (string), `top_k`,
/// `penalty_last_n` and `mirostat` (int32), `top_p`, `min_p`,
/// `xtc_probability`, `xtc_threshold`, `temperature` (as `temp`),
/// `penalty_repeat` (or `repetition_penalty`), `mirostat_tau` and
/// `mirostat_eta` (float32). A key absent or `null` in the file writes
/// nothing, an empty file writes nothing, and so does a value of the wrong
/// type (the Python converter would fail on it). The engine itself reads
/// `temp`, `top_k` and `top_p` only (`sampling.rs`).
pub(crate) fn sampling_defaults(generation_config: Option<&Json>) -> Vec<(String, Value)> {
    let Some(Json::Object(gen_cfg)) = generation_config else {
        return Vec::new();
    };
    let int = |key: &str| -> Option<Value> {
        let n = gen_cfg.get(key)?.as_i64()?;
        i32::try_from(n).ok().map(Value::I32)
    };
    let float = |keys: &[&str]| -> Option<Value> {
        // The first key present decides, even when it is null (Python's
        // `get(a, get(b))`).
        keys.iter()
            .find_map(|k| gen_cfg.get(*k))?
            .as_f64()
            .map(|f| Value::F32(f as f32))
    };
    let text = |key: &str| -> Option<Value> {
        gen_cfg
            .get(key)?
            .as_str()
            .map(|s| Value::String(s.to_string()))
    };
    [
        ("general.sampling.sequence", text("sequence")),
        ("general.sampling.top_k", int("top_k")),
        ("general.sampling.top_p", float(&["top_p"])),
        ("general.sampling.min_p", float(&["min_p"])),
        (
            "general.sampling.xtc_probability",
            float(&["xtc_probability"]),
        ),
        ("general.sampling.xtc_threshold", float(&["xtc_threshold"])),
        ("general.sampling.temp", float(&["temperature"])),
        ("general.sampling.penalty_last_n", int("penalty_last_n")),
        (
            "general.sampling.penalty_repeat",
            float(&["penalty_repeat", "repetition_penalty"]),
        ),
        ("general.sampling.mirostat", int("mirostat")),
        ("general.sampling.mirostat_tau", float(&["mirostat_tau"])),
        ("general.sampling.mirostat_eta", float(&["mirostat_eta"])),
    ]
    .into_iter()
    .filter_map(|(key, value)| Some((key.to_string(), value?)))
    .collect()
}

/// Load the metadata through the engine's own reader, so a key the engine
/// needs but the converter did not write fails the conversion, naming it.
///
/// `has_tensor` reports the names of the tensors the file will hold.
pub(crate) fn check(
    kv: &[(String, Value)],
    has_tensor: &dyn Fn(&str) -> bool,
) -> Result<(), CucaError> {
    let md: std::collections::HashMap<String, Value> = kv.iter().cloned().collect();
    Hparams::from_metadata(&md, has_tensor)
        .map(|_| ())
        .map_err(|e| cfg_err(format!("the converted metadata does not load: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hp(v: Json) -> Hp {
        Hp::new(&v).unwrap()
    }

    fn get<'a>(m: &'a ModelMeta, key: &str) -> &'a Value {
        &m.kv
            .iter()
            .find(|(k, _)| k == key)
            .unwrap_or_else(|| panic!("no {key}"))
            .1
    }

    #[test]
    fn llama3_rope_scaling_writes_freqs_not_metadata() {
        let h = hp(serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2,
            "num_key_value_heads": 2, "max_position_embeddings": 4096,
            "intermediate_size": 128, "rms_norm_eps": 1e-5, "vocab_size": 100,
            "rope_theta": 500000.0,
            "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
        }));
        let m = build(ConvArch::Llama, &h, "t", WeightQuant::F16, 0).unwrap();
        assert!(m.kv.iter().all(|(k, _)| !k.contains("rope.scaling")));
        assert_eq!(m.extra.len(), 1);
        assert_eq!(m.extra[0].data.len(), 8);
        assert_eq!(m.extra[0].data[0], 1.0, "highest frequency is unscaled");
        assert_eq!(
            m.extra[0].data[7], 32.0,
            "lowest frequency is scaled by the factor"
        );
        assert!(matches!(
            get(&m, "llama.rope.dimension_count"),
            Value::U32(16)
        ));
        assert!(matches!(get(&m, "llama.rope.freq_base"), Value::F32(f) if *f == 500000.0));
    }

    /// `generation_config.json` becomes `general.sampling.*` like gguf-py's
    /// `Metadata`: names, types, `repetition_penalty` as a fallback, nothing
    /// for absent or null keys, and nothing at all for no file.
    #[test]
    fn generation_config_writes_the_sampling_keys() {
        let kv = sampling_defaults(Some(&serde_json::json!({
            "bos_token_id": 2, "do_sample": true, "max_length": 4096,
            "temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.05,
            "repetition_penalty": 1.1, "penalty_last_n": 64, "top_a": null
        })));
        let get = |key: &str| kv.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
        assert!(matches!(get("general.sampling.temp"), Some(Value::F32(t)) if t == 0.6f32));
        assert!(matches!(
            get("general.sampling.top_k"),
            Some(Value::I32(20))
        ));
        assert!(matches!(get("general.sampling.top_p"), Some(Value::F32(p)) if p == 0.95f32));
        assert!(matches!(get("general.sampling.min_p"), Some(Value::F32(p)) if p == 0.05f32));
        assert!(
            matches!(get("general.sampling.penalty_repeat"), Some(Value::F32(p)) if p == 1.1f32)
        );
        assert!(matches!(
            get("general.sampling.penalty_last_n"),
            Some(Value::I32(64))
        ));
        assert_eq!(kv.len(), 6, "{kv:?}: no key for the other fields");
        // `penalty_repeat` wins over `repetition_penalty`; an integer is a float.
        let kv = sampling_defaults(Some(&serde_json::json!({
            "penalty_repeat": 1.2, "repetition_penalty": 1.1, "top_p": 1
        })));
        assert!(matches!(&kv[..], [(a, Value::F32(p)), (b, Value::F32(r))]
            if a == "general.sampling.top_p" && *p == 1.0
            && b == "general.sampling.penalty_repeat" && *r == 1.2f32));
        // Mistral-style files name no sampling field: no keys.
        assert!(
            sampling_defaults(Some(
                &serde_json::json!({"bos_token_id": 1, "max_length": 262144})
            ))
            .is_empty()
        );
        assert!(sampling_defaults(Some(&serde_json::json!({}))).is_empty());
        assert!(sampling_defaults(None).is_empty());
    }

    #[test]
    fn text_config_merges_over_the_top_level() {
        let h = hp(serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"num_hidden_layers": 3, "hidden_size": 8},
            "hidden_size": 99
        }));
        assert_eq!(h.n_layer().unwrap(), 3);
        assert_eq!(h.n_embd().unwrap(), 8);
    }

    #[test]
    fn gemma3_mirrors_the_local_rope_base() {
        let h = hp(serde_json::json!({
            "architectures": ["Gemma3ForCausalLM"],
            "num_hidden_layers": 6, "hidden_size": 32, "intermediate_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
            "sliding_window": 512, "rope_local_base_freq": 10000.0, "rope_theta": 1000000.0,
            "max_position_embeddings": 32768, "rms_norm_eps": 1e-6, "vocab_size": 100
        }));
        let m = build(ConvArch::Gemma3, &h, "t", WeightQuant::Q8_0, 0).unwrap();
        assert!(matches!(get(&m, "gemma3.rope.freq_base"), Value::F32(f) if *f == 1e6));
        assert!(matches!(get(&m, "gemma3.rope.freq_base_swa"), Value::F32(f) if *f == 1e4));
        assert!(matches!(
            get(&m, "gemma3.attention.sliding_window"),
            Value::U32(512)
        ));
        assert!(matches!(get(&m, "general.file_type"), Value::U32(7)));
    }

    fn muse_config() -> Json {
        serde_json::json!({
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "text_config": {
                "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
                "sliding_window": 8, "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0],
                "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
                "output_multiplier": 0.35, "final_logit_softcapping": 20.0,
                "qk_scale_factor": 3.87, "post_norm_eps": 1e-8, "hidden_activation": "silu",
                "vocab_size": 300
            }
        })
    }

    #[test]
    fn muse_glimmer_writes_the_pattern_multiplier_and_softcap() {
        let m = build(
            ConvArch::MuseGlimmer,
            &hp(muse_config()),
            "t",
            WeightQuant::F16,
            0,
        )
        .unwrap();
        assert!(matches!(
            get(&m, "muse-glimmer.logit_scale"),
            Value::F32(f) if *f == 0.35f32
        ));
        assert!(matches!(
            get(&m, "muse-glimmer.final_logit_softcapping"),
            Value::F32(f) if *f == 20.0
        ));
        assert!(matches!(
            get(&m, "muse-glimmer.attention.sliding_window"),
            Value::U32(8)
        ));
        let Value::Array(pattern) = get(&m, "muse-glimmer.attention.sliding_window_pattern") else {
            panic!("pattern");
        };
        let bools: Vec<bool> = pattern
            .iter()
            .map(|v| matches!(v, Value::Bool(true)))
            .collect();
        assert_eq!(bools, [true, true, true, false]);
        assert!(matches!(
            get(&m, "muse-glimmer.rope.freq_base"),
            Value::F32(f) if *f == 500000.0
        ));
        assert!(matches!(
            get(&m, "muse-glimmer.attention.key_length"),
            Value::U32(32)
        ));
    }

    #[test]
    fn muse_glimmer_refuses_what_the_engine_fixes() {
        let refuse = |edit: &dyn Fn(&mut serde_json::Map<String, Json>)| {
            let mut cfg = muse_config();
            edit(cfg["text_config"].as_object_mut().unwrap());
            let h = hp(cfg);
            build(ConvArch::MuseGlimmer, &h, "t", WeightQuant::F16, 0).unwrap_err()
        };
        // A rotary embedding on a full layer (or none on a sliding one).
        refuse(&|t| {
            t.insert(
                "layer_rope_theta".into(),
                serde_json::json!([500000.0, 500000.0, 500000.0, 500000.0]),
            );
        });
        refuse(&|t| {
            t.insert(
                "layer_rope_theta".into(),
                serde_json::json!([500000.0, 0, 500000.0, 0]),
            );
        });
        refuse(&|t| {
            t.insert("post_norm_eps".into(), serde_json::json!(1e-5));
        });
        refuse(&|t| {
            t.insert("hidden_activation".into(), serde_json::json!("gelu"));
        });
        refuse(&|t| {
            t.insert("attention_bias".into(), serde_json::json!(true));
        });
        refuse(&|t| {
            t.remove("output_multiplier");
        });
    }

    #[test]
    fn mistral4_metadata_follows_the_deepseek_converter() {
        // The text model of mistralai/Mistral-Small-4-119B-2603's config.json.
        let h = hp(serde_json::json!({
            "architectures": ["Mistral3ForConditionalGeneration"],
            "model_type": "mistral3",
            "text_config": {
                "first_k_dense_replace": 0, "head_dim": 128, "hidden_size": 4096,
                "intermediate_size": 12288, "kv_lora_rank": 256,
                "max_position_embeddings": 1048576, "model_type": "mistral4",
                "moe_intermediate_size": 2048, "n_group": 1, "n_routed_experts": 128,
                "n_shared_experts": 1, "norm_topk_prob": true, "num_attention_heads": 32,
                "num_experts_per_tok": 4, "num_hidden_layers": 36, "num_key_value_heads": 32,
                "q_lora_rank": 1024, "qk_nope_head_dim": 64, "qk_rope_head_dim": 64,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "beta_fast": 32.0, "beta_slow": 1.0, "factor": 128.0,
                    "llama_4_scaling_beta": 0.1, "mscale": 1.0, "mscale_all_dim": 1.0,
                    "original_max_position_embeddings": 8192, "rope_theta": 10000.0,
                    "rope_type": "yarn", "type": "yarn"
                },
                "routed_scaling_factor": 1.0, "topk_group": 1, "v_head_dim": 128,
                "vocab_size": 131072
            }
        }))
        .with_defaults(ConvArch::Mistral4);
        let m = build(ConvArch::Mistral4, &h, "t", WeightQuant::F16, 0).unwrap();
        let u32s = |key: &str| match get(&m, &format!("mistral4.{key}")) {
            Value::U32(n) => *n,
            other => panic!("{key}: {other:?}"),
        };
        // Multi-query attention over the compressed cache.
        assert_eq!(u32s("attention.head_count"), 32);
        assert_eq!(u32s("attention.head_count_kv"), 1);
        assert_eq!(u32s("attention.key_length"), 256 + 64);
        assert_eq!(u32s("attention.value_length"), 256);
        assert_eq!(u32s("attention.key_length_mla"), 64 + 64);
        assert_eq!(u32s("attention.value_length_mla"), 128);
        assert_eq!(u32s("attention.q_lora_rank"), 1024);
        assert_eq!(u32s("attention.kv_lora_rank"), 256);
        assert_eq!(u32s("rope.dimension_count"), 64);
        assert_eq!(u32s("leading_dense_block_count"), 0);
        assert_eq!(u32s("expert_count"), 128);
        assert_eq!(u32s("expert_used_count"), 4);
        assert_eq!(u32s("expert_shared_count"), 1);
        assert_eq!(u32s("expert_feed_forward_length"), 2048);
        assert_eq!(u32s("expert_group_count"), 1);
        assert_eq!(u32s("expert_group_used_count"), 1);
        assert_eq!(u32s("rope.scaling.original_context_length"), 8192);
        assert!(matches!(get(&m, "mistral4.expert_weights_scale"), Value::F32(f) if *f == 1.0));
        assert!(matches!(
            get(&m, "mistral4.expert_weights_norm"),
            Value::Bool(true)
        ));
        // llama.cpp's DeepSeek convention: 0.1 * mscale_all_dim.
        assert!(
            matches!(get(&m, "mistral4.rope.scaling.yarn_log_multiplier"), Value::F32(f) if *f == 0.1)
        );
        assert!(matches!(get(&m, "mistral4.rope.scaling.factor"), Value::F32(f) if *f == 128.0));
        assert!(m.kv.iter().all(|(k, _)| !k.contains("temperature")));
    }

    #[test]
    fn block_scaled_fp8_is_refused_up_front() {
        let h = hp(serde_json::json!({
            "num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 2,
            "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
        }));
        let err = build(ConvArch::Qwen2, &h, "t", WeightQuant::F16, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("weight_block_size"), "{err}");
    }

    #[test]
    fn unsupported_variants_are_refused() {
        let h = hp(serde_json::json!({
            "num_hidden_layers": 1, "norm_topk_prob": false, "hidden_size": 8,
            "num_attention_heads": 2, "num_experts": 4
        }));
        assert!(build(ConvArch::Qwen3Moe, &h, "t", WeightQuant::F16, 0).is_err());
        let h = hp(serde_json::json!({
            "num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 2,
            "rope_scaling": {"rope_type": "dynamic", "factor": 2.0}
        }));
        let err = build(ConvArch::Qwen2, &h, "t", WeightQuant::F16, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dynamic"), "{err}");
    }
}
