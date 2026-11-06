//! GGUF metadata from `config.json`.
//!
//! A port of the pinned llama.cpp converter's `set_gguf_parameters` (the
//! generic `TextModel` one and each family's override) and of the extra
//! tensors it generates (`generate_extra_tensors`: Llama 3 and Gemma 4
//! `rope_freqs`, Phi 3 LongRoPE factors). Every key `Hparams::from_metadata`
//! reads is produced; [`check`] loads the result through it so a gap fails the
//! conversion naming the key.

use candle_core::quantized::gguf_file::Value;
use serde_json::{Map, Value as Json};

use super::arch::ConvArch;
use crate::error::CucaError;
use crate::provider::local::WeightQuant;
use crate::provider::local::model::params::Hparams;

fn cfg_err(msg: impl Into<String>) -> CucaError {
    CucaError::Config(msg.into())
}

/// The merged hyperparameters of a checkpoint, as the converter's
/// `self.hparams` and `self.rope_parameters`.
#[derive(Debug)]
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
        let partial = hp
            .find(&["partial_rotary_factor", "rope_pct", "rope_percent"])
            .cloned();
        let orig = hp.find(&["original_max_position_embeddings"]).cloned();
        if !rope.contains_key("full_attention") && !rope.contains_key("sliding_attention") {
            if let Some(t) = local_theta {
                rope.insert(
                    "sliding_attention".into(),
                    serde_json::json!({ "rope_theta": t }),
                );
            }
            if !rope.contains_key("rope_theta")
                && let Some(t) = rope_theta
            {
                rope.insert("rope_theta".into(), t);
            }
            if !rope.contains_key("rope_type")
                && let Some(t) = rope.get("type").filter(|t| !t.is_null()).cloned()
            {
                rope.insert("rope_type".into(), t);
            }
            if !rope.contains_key("partial_rotary_factor")
                && let Some(p) = partial
            {
                rope.insert("partial_rotary_factor".into(), p);
            }
            if !rope.contains_key("original_max_position_embeddings")
                && let Some(o) = orig
            {
                rope.insert("original_max_position_embeddings".into(), o);
            }
        }
        hp.rope = rope;
        Ok(hp)
    }

    /// Fill what `transformers`' config classes default and the converter
    /// therefore sees (it loads `AutoConfig(...).to_dict()`): the head width
    /// of Llama-family and Qwen 3 configs and the RMSNorm epsilon.
    pub(crate) fn with_defaults(mut self, arch: ConvArch) -> Self {
        let set = |map: &mut Map<String, Json>, key: &str, value: Json| {
            if map.get(key).is_none_or(Json::is_null) {
                map.insert(key.to_string(), value);
            }
        };
        let head_dim = match (self.n_embd(), self.n_head()) {
            (Ok(e), Ok(h)) if h > 0 => Some(e / h),
            _ => None,
        };
        match arch {
            ConvArch::Llama => {
                if let Some(d) = head_dim {
                    set(&mut self.map, "head_dim", Json::from(d));
                }
                set(&mut self.map, "rms_norm_eps", Json::from(1e-6));
            }
            ConvArch::Smollm3 => set(&mut self.map, "rms_norm_eps", Json::from(1e-6)),
            ConvArch::Qwen3 | ConvArch::Qwen3Moe => {
                set(&mut self.map, "head_dim", Json::from(128));
                set(&mut self.map, "rms_norm_eps", Json::from(1e-6));
            }
            ConvArch::Qwen2 | ConvArch::Granite => {
                set(&mut self.map, "rms_norm_eps", Json::from(1e-6));
            }
            ConvArch::Phi3 | ConvArch::Olmo2 | ConvArch::Olmoe | ConvArch::Ernie4_5 => {
                set(&mut self.map, "rms_norm_eps", Json::from(1e-5));
            }
            _ => {}
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
pub(crate) struct Kv {
    prefix: &'static str,
    pub(crate) items: Vec<(String, Value)>,
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
    pub(crate) fn raw(&mut self, key: &str, value: Value) {
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

    fn f32(&mut self, key: &str, v: f64) {
        self.arch_key(key, Value::F32(v as f32));
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
fn file_type(q: WeightQuant) -> u32 {
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
    let factor = p.get("factor").and_then(Json::as_f64);
    match kind {
        "default" | "" => {}
        "linear" if factor.is_some() => {
            kv.arch_key("rope.scaling.type", Value::String("linear".into()));
            kv.f32("rope.scaling.factor", factor.unwrap_or(1.0));
        }
        "yarn" if factor.is_some() => {
            kv.arch_key("rope.scaling.type", Value::String("yarn".into()));
            kv.f32("rope.scaling.factor", factor.unwrap_or(1.0));
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
            if let Some(e) = p.get("extrapolation_factor").and_then(Json::as_f64) {
                kv.f32("rope.scaling.yarn_ext_factor", e);
            }
            if p.get("attention_factor")
                .or_else(|| p.get("attn_factor"))
                .is_some()
            {
                return Err(cfg_err(
                    "an explicit YaRN attention_factor is not supported by the engine",
                ));
            }
            if let Some(b) = p.get("beta_fast").and_then(Json::as_f64) {
                kv.f32("rope.scaling.yarn_beta_fast", b);
            }
            if let Some(b) = p.get("beta_slow").and_then(Json::as_f64) {
                kv.f32("rope.scaling.yarn_beta_slow", b);
            }
        }
        "su" | "longrope" => {
            kv.arch_key("rope.scaling.type", Value::String("longrope".into()));
        }
        "llama3" => {}
        "proportional" if arch == ConvArch::Gemma4 || arch == ConvArch::Gemma4Assistant => {}
        other => {
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
    if let Some(n) = hp.uint(&[
        "max_position_embeddings",
        "n_ctx",
        "n_positions",
        "max_length",
        "max_sequence_length",
        "model_max_length",
    ]) {
        kv.u32("context_length", n)?;
    }
    if let Some(n) = hp.uint(&["hidden_size", "n_embd", "dim"]) {
        kv.u32("embedding_length", n)?;
    }
    if let Some(n) = hp.uint(&[
        "prefix_dense_intermediate_size",
        "dense_intermediate_size",
        "intermediate_size",
        "n_inner",
        "hidden_dim",
    ]) {
        kv.u32("feed_forward_length", n)?;
    }
    if let Some(n) = hp.uint(&["num_attention_heads", "n_head", "n_heads"]) {
        kv.u32("attention.head_count", n)?;
    }
    if let Some(n) = hp.uint(&["num_key_value_heads", "n_kv_heads"]) {
        kv.u32("attention.head_count_kv", n)?;
    }
    rope_scaling(arch, hp, kv)?;
    if let Some(Json::Array(sections)) = hp.rope_all().get("mrope_section") {
        let mut s: Vec<i64> = sections.iter().filter_map(Json::as_i64).collect();
        while s.len() < 4 {
            s.push(0);
        }
        kv.i32s("rope.dimension_sections", &s[..4]);
    }
    if let Some(t) = hp.rope_full().get("rope_theta").and_then(Json::as_f64) {
        kv.f32("rope.freq_base", t);
    }
    if let Some(t) = hp.rope_sliding_theta() {
        kv.f32("rope.freq_base_swa", t);
    }
    if let Some(e) = hp.float(&["rms_norm_eps", "norm_eps"]) {
        kv.f32("attention.layer_norm_rms_epsilon", e);
    }
    if let Some(e) = hp.float(&["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"]) {
        kv.f32("attention.layer_norm_epsilon", e);
    }
    if let Some(n) = hp.uint(&["num_local_experts", "num_experts", "n_routed_experts"]) {
        kv.u32("expert_count", n)?;
    }
    if let Some(n) = hp.uint(&[
        "num_experts_per_tok",
        "num_experts_per_token",
        "top_k_experts",
    ]) {
        kv.u32("expert_used_count", n)?;
    }
    if let Some(d) = hp.uint(&["head_dim"]) {
        kv.u32("attention.key_length", d)?;
        kv.u32("attention.value_length", d)?;
    }
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

fn gemma_softcaps(hp: &Hp, kv: &mut Kv) {
    if let Some(c) = hp.truthy_f("final_logit_softcapping") {
        kv.f32("final_logit_softcapping", c);
    }
}

/// `per_layer_config[i]` of the layers of `kind`, first hit of `key`.
fn first_layer_value(hp: &Hp, kind: &str, key: &str) -> Option<Json> {
    let layer_types = hp.get("layer_types")?.as_array()?;
    let Json::Object(cfg) = hp.get("per_layer_config")? else {
        return None;
    };
    for (idx, layer) in cfg {
        let i: usize = idx.parse().ok()?;
        if layer_types.get(i).and_then(Json::as_str) == Some(kind)
            && let Some(v) = layer.get(key)
        {
            return Some(v.clone());
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
    if hp
        .get("enable_moe_block")
        .and_then(Json::as_bool)
        .unwrap_or(false)
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
    let layer_types: Vec<&str> = hp
        .get("layer_types")
        .and_then(Json::as_array)
        .ok_or_else(|| cfg_err("config.json has no `layer_types`"))?
        .iter()
        .filter_map(Json::as_str)
        .collect();
    let swa: Vec<bool> = layer_types
        .iter()
        .map(|t| *t == "sliding_attention")
        .collect();
    kv.bools("attention.sliding_window_pattern", &swa);

    let head_dim_full = match hp.uint(&["global_head_dim"]) {
        Some(d) => d,
        None => first_layer_value(hp, "full_attention", "head_dim")
            .as_ref()
            .and_then(as_usize)
            .ok_or_else(|| cfg_err("cannot find the full-attention head dimension"))?,
    };
    let head_dim_swa = hp.req_uint(&["head_dim"])?;
    kv.u32("attention.key_length", head_dim_full)?;
    kv.u32("attention.value_length", head_dim_full)?;
    kv.u32("attention.key_length_swa", head_dim_swa)?;
    kv.u32("attention.value_length_swa", head_dim_swa)?;

    if let Some(n) = hp.uint(&["expert_intermediate_size", "moe_intermediate_size"]) {
        kv.u32("expert_feed_forward_length", n)?;
    }
    if hp
        .get("use_double_wide_mlp")
        .and_then(Json::as_bool)
        .unwrap_or(false)
    {
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
    let kv_full = match hp.uint(&["num_global_key_value_heads"]) {
        Some(n) => Some(n),
        None => first_layer_value(hp, "full_attention", "num_key_value_heads")
            .as_ref()
            .and_then(as_usize),
    };
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
    kv.f32(
        "rope.freq_base",
        hp.rope_full()
            .get("rope_theta")
            .and_then(Json::as_f64)
            .unwrap_or(1_000_000.0),
    );
    if hp.get("attn_logit_softcapping").is_some() {
        return Err(cfg_err(
            "Gemma 3 checkpoints have no attn_logit_softcapping",
        ));
    }
    gemma_softcaps(hp, kv);
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
        rope.get("rope_theta")
            .and_then(Json::as_f64)
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

/// Refuse features the engine's architecture does not implement, so a
/// converted file never silently computes something else.
fn guard(arch: ConvArch, hp: &Hp) -> Result<(), CucaError> {
    let sliding_layers = |hp: &Hp| {
        hp.get("layer_types")
            .and_then(Json::as_array)
            .is_some_and(|l| l.iter().any(|t| t.as_str() == Some("sliding_attention")))
    };
    match arch {
        ConvArch::Olmo2 if sliding_layers(hp) => {
            return Err(cfg_err(
                "OLMo 3 sliding-window layers are not supported by the engine",
            ));
        }
        ConvArch::Qwen3Moe => {
            if hp.get("norm_topk_prob").and_then(Json::as_bool) != Some(true) {
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
        ConvArch::Olmoe if hp.get("norm_topk_prob").and_then(Json::as_bool) == Some(true) => {
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
        ConvArch::Llama | ConvArch::Smollm3 | ConvArch::Granite => {
            if arch == ConvArch::Granite {
                // Granite ignores head_dim: the engine derives the head size
                // from the embedding width.
                let mut granite = Hp {
                    map: hp.map.clone(),
                    rope: hp.rope.clone(),
                };
                granite.map.remove("head_dim");
                base(arch, &granite, &mut kv, n_block)?;
                llama_extras(&granite, &mut kv)?;
                for (src, key) in [
                    ("attention_multiplier", "attention.scale"),
                    ("embedding_multiplier", "embedding_scale"),
                    ("residual_multiplier", "residual_scale"),
                    ("logits_scaling", "logit_scale"),
                ] {
                    if let Some(v) = granite.truthy_f(src) {
                        kv.f32(key, v);
                    }
                }
            } else {
                base(arch, hp, &mut kv, n_block)?;
                llama_extras(hp, &mut kv)?;
            }
            extra = llama3_freqs(hp)?;
        }
        ConvArch::Qwen2 | ConvArch::Qwen3 | ConvArch::Ernie4_5 => base(arch, hp, &mut kv, n_block)?,
        ConvArch::Qwen3Moe => {
            base(arch, hp, &mut kv, n_block)?;
            if let Some(n) = hp.uint(&["moe_intermediate_size"]) {
                kv.u32("expert_feed_forward_length", n)?;
            }
        }
        ConvArch::Qwen35 | ConvArch::Qwen35Moe => {
            base(arch, hp, &mut kv, n_block)?;
            if let Some(n) = hp.uint(&["moe_intermediate_size"]) {
                kv.u32("expert_feed_forward_length", n)?;
            }
            if let Some(n) = hp.uint(&["shared_expert_intermediate_size"]) {
                kv.u32("expert_shared_feed_forward_length", n)?;
            }
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
            if let Some(w) = hp.uint(&["sliding_window"]) {
                kv.u32("attention.sliding_window", w)?;
            }
        }
        ConvArch::Olmoe => {
            base(arch, hp, &mut kv, n_block)?;
            kv.f32("attention.layer_norm_rms_epsilon", 1e-5);
        }
    }
    kv.raw("general.quantization_version", Value::U32(2));
    kv.raw("general.file_type", Value::U32(file_type(quant)));
    Ok(ModelMeta {
        kv: kv.items,
        extra,
    })
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
