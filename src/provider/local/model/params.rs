//! Hyperparameters read from the `{general.architecture}.*` GGUF metadata.
//!
//! [`Hparams::from_metadata`] resolves the architecture ([`Arch`]) and turns
//! the keys plus the architecture table ([`Arch::behavior`]) into per-layer
//! specifications ([`LayerSpec`]) and the distinct rotary embeddings they use
//! ([`RopeSpec`]): the loaders and both forward passes consume only this
//! structure, never raw keys.

use std::collections::HashMap;

use candle_core::quantized::gguf_file::Value;

use super::arch::{Act, Arch, AttnScale, Behavior, EmbdScale, Swa};
use super::rope::{Factors, RopeSpec, Yarn};
use super::{err, meta_err};
use crate::error::CucaError;

/// Metadata of a GGUF file.
pub(crate) type Metadata = HashMap<String, Value>;

/// An unsigned integer value of any width.
fn value_usize(v: &Value) -> Option<usize> {
    match v {
        Value::U8(n) => Some(usize::from(*n)),
        Value::U16(n) => Some(usize::from(*n)),
        Value::U32(n) => usize::try_from(*n).ok(),
        Value::U64(n) => usize::try_from(*n).ok(),
        Value::I8(n) => usize::try_from(*n).ok(),
        Value::I16(n) => usize::try_from(*n).ok(),
        Value::I32(n) => usize::try_from(*n).ok(),
        Value::I64(n) => usize::try_from(*n).ok(),
        _ => None,
    }
}

fn value_f64(v: &Value) -> Option<f64> {
    match v {
        Value::F32(n) => Some(f64::from(*n)),
        Value::F64(n) => Some(*n),
        other => value_usize(other).map(|n| n as f64),
    }
}

/// Typed access to the `{arch}.`-prefixed keys of a metadata map.
pub(crate) struct Keys<'a> {
    md: &'a Metadata,
    prefix: &'a str,
}

impl<'a> Keys<'a> {
    pub(crate) fn new(md: &'a Metadata, prefix: &'a str) -> Self {
        Self { md, prefix }
    }

    fn full(&self, key: &str) -> String {
        format!("{}.{key}", self.prefix)
    }

    fn value(&self, key: &str) -> Option<&'a Value> {
        self.md.get(&self.full(key))
    }

    pub(crate) fn usize_opt(&self, key: &str) -> Result<Option<usize>, CucaError> {
        self.value(key)
            .map(|v| {
                value_usize(v).ok_or_else(|| meta_err(&self.full(key), "not an unsigned integer"))
            })
            .transpose()
    }

    pub(crate) fn usize(&self, key: &str) -> Result<usize, CucaError> {
        self.usize_opt(key)?
            .ok_or_else(|| meta_err(&self.full(key), "missing"))
    }

    pub(crate) fn f64_opt(&self, key: &str) -> Result<Option<f64>, CucaError> {
        self.value(key)
            .map(|v| value_f64(v).ok_or_else(|| meta_err(&self.full(key), "not a number")))
            .transpose()
    }

    pub(crate) fn f64(&self, key: &str) -> Result<f64, CucaError> {
        self.f64_opt(key)?
            .ok_or_else(|| meta_err(&self.full(key), "missing"))
    }

    pub(crate) fn str_opt(&self, key: &str) -> Option<&'a str> {
        match self.value(key) {
            Some(Value::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// A key that is either one value for every layer or one per layer.
    fn per_layer(&self, key: &str, n_layer: usize) -> Result<Option<Vec<usize>>, CucaError> {
        let Some(value) = self.value(key) else {
            return Ok(None);
        };
        let name = self.full(key);
        if let Value::Array(items) = value {
            let counts = items
                .iter()
                .map(|v| value_usize(v).ok_or_else(|| meta_err(&name, "not an unsigned integer")))
                .collect::<Result<Vec<_>, _>>()?;
            if counts.len() != n_layer {
                return Err(meta_err(
                    &name,
                    format!("expected {n_layer} entries, found {}", counts.len()),
                ));
            }
            return Ok(Some(counts));
        }
        let n = value_usize(value).ok_or_else(|| meta_err(&name, "not an unsigned integer"))?;
        Ok(Some(vec![n; n_layer]))
    }
}

/// What a transformer block mixes tokens with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    /// Softmax attention over the K/V cache (the [`LayerSpec`] attention
    /// fields apply).
    Attention,
    /// A gated delta-net (linear attention) layer: a convolution and a
    /// recurrent matrix state per sequence instead of a K/V cache; the
    /// attention fields of [`LayerSpec`] only echo the head counts.
    DeltaNet(DeltaSpec),
}

/// Dimensions of a gated delta-net layer, from `{arch}.ssm.*` (llama.cpp
/// names in parentheses).
///
/// A layer keeps `n_v_heads` states of shape `[head_k_dim, head_v_dim]`; the
/// `n_k_heads` query/key heads are shared by `n_v_heads / n_k_heads`
/// consecutive value heads (repeat-interleave).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeltaSpec {
    /// Taps of the causal depthwise convolution (`ssm.conv_kernel`); the
    /// convolution state holds the last `conv_kernel - 1` inputs.
    pub(crate) conv_kernel: usize,
    /// Query/key heads (`ssm.group_count`).
    pub(crate) n_k_heads: usize,
    /// Value heads, one recurrent state each (`ssm.time_step_rank`).
    pub(crate) n_v_heads: usize,
    /// Query/key head width (`ssm.state_size`).
    pub(crate) head_k_dim: usize,
    /// Value head width (`ssm.inner_size / ssm.time_step_rank`).
    pub(crate) head_v_dim: usize,
}

impl DeltaSpec {
    /// Width of the query (and of the key) projection.
    pub(crate) fn key_dim(&self) -> usize {
        self.n_k_heads * self.head_k_dim
    }

    /// Width of the value (and gate) projection, `ssm.inner_size`.
    pub(crate) fn value_dim(&self) -> usize {
        self.n_v_heads * self.head_v_dim
    }

    /// Channels of the convolution: query, key and value concatenated.
    pub(crate) fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// f32 values of one layer's recurrent state: the convolution history
    /// (`conv_kernel - 1` rows of `conv_dim`) and the `n_v_heads` matrices.
    pub(crate) fn state_floats(&self) -> usize {
        (self.conv_kernel - 1) * self.conv_dim()
            + self.n_v_heads * self.head_k_dim * self.head_v_dim
    }
}

/// One transformer block's mixer kind and attention geometry.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayerSpec {
    pub(crate) kind: LayerKind,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    /// Sliding-window length in tokens (the query included); `None` for full
    /// attention.
    pub(crate) window: Option<usize>,
    /// Index into [`Hparams::ropes`]; `None` for a NoPE layer and for a
    /// delta-net layer (which has no positional encoding).
    pub(crate) rope: Option<usize>,
    /// The layer whose K/V cache this layer reads: itself when it owns K/V;
    /// `usize::MAX` for a drafter layer, which reads the target's cache, and
    /// for a delta-net layer, which has no K/V.
    pub(crate) kv_source: usize,
    /// Multiplier on the attention scores.
    pub(crate) scale: f32,
}

impl LayerSpec {
    /// Whether this layer computes and stores its own K/V.
    pub(crate) fn owns_kv(&self, il: usize) -> bool {
        self.kv_source == il
    }

    /// The delta-net dimensions when this layer is a delta-net layer.
    pub(crate) fn delta(&self) -> Option<&DeltaSpec> {
        match &self.kind {
            LayerKind::DeltaNet(d) => Some(d),
            LayerKind::Attention => None,
        }
    }
}

/// Top-k mixture-of-experts FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MoeSpec {
    pub(crate) n_expert: usize,
    pub(crate) n_used: usize,
    /// Renormalise the selected experts' softmax weights to sum to one.
    pub(crate) norm_weights: bool,
}

/// Hyperparameters of a checkpoint.
#[derive(Debug, Clone)]
pub(crate) struct Hparams {
    pub(crate) arch: Arch,
    pub(crate) behavior: Behavior,
    /// Trunk layers: `block_count` minus the trailing multi-token-prediction
    /// blocks ([`Hparams::nextn_predict_layers`]).
    pub(crate) n_layer: usize,
    pub(crate) n_embd: usize,
    /// Width of the hidden state the model exchanges with its partner: the
    /// target's `n_embd` for a drafter (`embedding_length_out`), else `n_embd`.
    pub(crate) n_embd_out: usize,
    /// Maximum context length the checkpoint was trained for.
    pub(crate) n_ctx_train: usize,
    /// RMSNorm epsilon.
    pub(crate) eps: f64,
    pub(crate) layers: Vec<LayerSpec>,
    /// The distinct rotary embeddings of the layers.
    pub(crate) ropes: Vec<RopeSpec>,
    /// Context length LongRoPE compares against to pick short or long
    /// factors.
    pub(crate) n_ctx_orig: usize,
    /// Multiplier on the looked-up token embeddings.
    pub(crate) embd_scale: f32,
    /// Multiplier on each sub-block output before the residual add.
    pub(crate) residual_scale: f32,
    /// Divisor of the final logits.
    pub(crate) logit_scale: f32,
    pub(crate) logit_softcap: Option<f64>,
    pub(crate) attn_softcap: Option<f64>,
    /// Width of one layer's per-layer-embedding slice (Gemma 4).
    pub(crate) ple_dim: Option<usize>,
    pub(crate) moe: Option<MoeSpec>,
    /// Trailing blocks of the file (`{arch}.nextn_predict_layers`, only read
    /// for architectures that embed them) that hold a multi-token-prediction
    /// head instead of trunk layers: blocks `n_layer..n_layer +
    /// nextn_predict_layers`. The trunk loaders ignore them.
    pub(crate) nextn_predict_layers: usize,
    /// Geometry of the first prediction block, block `n_layer` (an
    /// attention block that owns its K/V; `None` without prediction
    /// blocks). It is not part of [`Hparams::layers`], so the trunk's K/V
    /// accounting ([`Hparams::kv_bytes_per_position`]) leaves it out.
    pub(crate) nextn: Option<LayerSpec>,
}

impl Hparams {
    pub(crate) fn act(&self) -> Act {
        self.behavior.act
    }

    /// Layers that compute their own K/V.
    pub(crate) fn kv_owners(&self) -> impl Iterator<Item = (usize, &LayerSpec)> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(il, l)| l.owns_kv(*il))
    }

    /// K/V cache bytes per context position (f32 keys and values).
    pub(crate) fn kv_bytes_per_position(&self) -> usize {
        self.kv_owners()
            .map(|(_, l)| 2 * 4 * l.n_kv_head * l.head_dim)
            .sum()
    }

    /// Whether any layer keeps a recurrent state (gated delta-net layers).
    /// Such a model cannot roll its cache back to an arbitrary position:
    /// see the `Memory` and `Cache rollback` sections of [`super`].
    pub(crate) fn is_hybrid(&self) -> bool {
        self.layers.iter().any(|l| l.delta().is_some())
    }

    /// Bytes of the recurrent state of one sequence: every delta-net layer's
    /// convolution history and state matrices in f32. Independent of the
    /// context length (Qwen 3.5 0.8B: 18 layers, 1.12 MB each).
    #[allow(dead_code)] // Reported by the memory docs and tests; no runtime consumer.
    pub(crate) fn recurrent_state_bytes(&self) -> usize {
        self.layers
            .iter()
            .filter_map(LayerSpec::delta)
            .map(|d| 4 * d.state_floats())
            .sum()
    }

    /// Read and validate the hyperparameters. `has_tensor` reports whether
    /// the file holds a tensor, which decides whether per-pair RoPE factors
    /// apply.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] naming the unsupported architecture or the
    /// missing, mistyped or inconsistent key.
    pub(crate) fn from_metadata(
        md: &Metadata,
        has_tensor: &dyn Fn(&str) -> bool,
    ) -> Result<Self, CucaError> {
        let arch_name = match md.get("general.architecture") {
            Some(Value::String(s)) => s.as_str(),
            _ => return Err(meta_err("general.architecture", "missing")),
        };
        let arch = Arch::from_name(arch_name)?;
        let behavior = arch.behavior();
        let k = Keys::new(md, arch.name());

        let n_block = k.usize("block_count")?;
        let n_embd = k.usize("embedding_length")?;
        let n_ctx_train = k.usize("context_length")?;
        let eps = k.f64("attention.layer_norm_rms_epsilon")?;
        if n_block == 0 || n_embd == 0 || n_ctx_train == 0 {
            return Err(err(
                "GGUF metadata: layer, embedding and context sizes must be non-zero",
            ));
        }
        let nextn_predict_layers = if behavior.nextn_blocks {
            k.usize_opt("nextn_predict_layers")?.unwrap_or(0)
        } else {
            0
        };
        if nextn_predict_layers >= n_block {
            return Err(meta_err(
                &format!("{}.nextn_predict_layers", arch.name()),
                format!(
                    "{nextn_predict_layers} prediction blocks leave no trunk layer among {n_block} blocks"
                ),
            ));
        }
        let n_layer = n_block - nextn_predict_layers;

        // Per-layer arrays cover every block of the file, the trailing
        // prediction blocks included.
        let n_head = k
            .per_layer("attention.head_count", n_block)?
            .ok_or_else(|| meta_err(&format!("{}.attention.head_count", arch.name()), "missing"))?;
        let n_kv_head = k
            .per_layer("attention.head_count_kv", n_block)?
            .unwrap_or_else(|| n_head.clone());

        // Head dimensions: Gemma 4 has one per attention type.
        let key_length = k.usize_opt("attention.key_length")?;
        let key_length_swa = if behavior.gemma4_extras {
            Some(k.usize("attention.key_length_swa")?)
        } else {
            None
        };
        for (key, expected) in [
            ("attention.value_length", key_length),
            ("attention.value_length_swa", key_length_swa),
        ] {
            if let (Some(v), Some(e)) = (k.usize_opt(key)?, expected)
                && v != e
            {
                return Err(err(format!(
                    "GGUF metadata `{}.{key}`: {v} differs from the key length {e}; only equal key and value lengths are supported",
                    arch.name()
                )));
            }
        }

        let sliding = sliding_layers(&k, &behavior, n_layer)?;
        let window = match (behavior.swa, k.usize_opt("attention.sliding_window")?) {
            (Swa::None, _) => None,
            (_, Some(w)) if w > 0 => Some(w),
            _ => {
                return Err(meta_err(
                    &format!("{}.attention.sliding_window", arch.name()),
                    "missing or zero",
                ));
            }
        };

        // Shared K/V layers (Gemma 4): the last `shared_kv_layers` read the
        // cache of the last owner of their own attention type.
        let shared_kv = if behavior.gemma4_extras && !arch.is_drafter() {
            k.usize_opt("attention.shared_kv_layers")?.unwrap_or(0)
        } else {
            0
        };
        let n_layer_kv = n_layer
            .checked_sub(shared_kv)
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                meta_err(
                    &format!("{}.attention.shared_kv_layers", arch.name()),
                    format!("{shared_kv} shared layers leave no K/V owner among {n_layer} layers"),
                )
            })?;
        // A drafter owns no K/V at all: every layer reads the target's cache.
        let kv_source = if arch.is_drafter() {
            vec![usize::MAX; n_layer]
        } else {
            kv_sources(&sliding, n_layer_kv)?
        };

        // Hybrid stacks: the full-attention interval and the delta-net
        // dimensions. Layer `il` is full attention when `(il + 1)` is a
        // multiple of the interval, a delta-net layer otherwise.
        let nonzero = |key: &str| -> Result<usize, CucaError> {
            match k.usize(key)? {
                0 => Err(meta_err(
                    &format!("{}.{key}", arch.name()),
                    "must be non-zero",
                )),
                n => Ok(n),
            }
        };
        let hybrid = if behavior.hybrid_delta_net {
            let interval = nonzero("full_attention_interval")?;
            let conv_kernel = nonzero("ssm.conv_kernel")?;
            let head_k_dim = nonzero("ssm.state_size")?;
            let n_k_heads = nonzero("ssm.group_count")?;
            let n_v_heads = nonzero("ssm.time_step_rank")?;
            let inner = nonzero("ssm.inner_size")?;
            if !n_v_heads.is_multiple_of(n_k_heads) {
                return Err(meta_err(
                    &format!("{}.ssm.time_step_rank", arch.name()),
                    format!(
                        "{n_v_heads} value heads are not a multiple of the {n_k_heads} key heads (ssm.group_count)"
                    ),
                ));
            }
            if !inner.is_multiple_of(n_v_heads) {
                return Err(meta_err(
                    &format!("{}.ssm.inner_size", arch.name()),
                    format!(
                        "{inner} is not a multiple of the {n_v_heads} value heads (ssm.time_step_rank)"
                    ),
                ));
            }
            let delta = DeltaSpec {
                conv_kernel,
                n_k_heads,
                n_v_heads,
                head_k_dim,
                head_v_dim: inner / n_v_heads,
            };
            Some((interval, delta))
        } else {
            None
        };

        // Rotary embeddings.
        let n_ctx_orig = k
            .usize_opt("rope.scaling.original_context_length")?
            .unwrap_or(n_ctx_train);
        let rope_full = |head_dim: usize| -> Result<RopeSpec, CucaError> {
            let n_dims = k.usize_opt("rope.dimension_count")?.unwrap_or(head_dim);
            let scaling = k.str_opt("rope.scaling.type").unwrap_or("linear");
            let factor = k.f64_opt("rope.scaling.factor")?.unwrap_or(0.0);
            let yarn = match scaling {
                "yarn" => Some(Yarn {
                    ext_factor: 1.0,
                    beta_fast: k.f64_opt("rope.scaling.yarn_beta_fast")?.unwrap_or(32.0),
                    beta_slow: k.f64_opt("rope.scaling.yarn_beta_slow")?.unwrap_or(1.0),
                    n_ctx_orig,
                }),
                "linear" | "none" | "longrope" => None,
                other => {
                    return Err(meta_err(
                        &format!("{}.rope.scaling.type", arch.name()),
                        format!("unsupported scaling type `{other}`"),
                    ));
                }
            };
            let factors = if has_tensor("rope_factors_long.weight")
                && has_tensor("rope_factors_short.weight")
            {
                Factors::LongShort
            } else if has_tensor("rope_freqs.weight") {
                Factors::Freqs
            } else {
                Factors::None
            };
            Ok(RopeSpec {
                kind: behavior.rope,
                n_dims,
                base: k.f64_opt("rope.freq_base")?.unwrap_or(10_000.0),
                freq_scale: if factor > 0.0 && scaling != "none" {
                    1.0 / factor
                } else {
                    1.0
                },
                yarn,
                attn_factor: k.f64_opt("rope.scaling.attn_factor")?.unwrap_or(1.0),
                factors,
            })
        };
        let rope_swa = |head_dim: usize| -> Result<RopeSpec, CucaError> {
            let n_dims = if behavior.gemma4_extras {
                k.usize("rope.dimension_count_swa")?
            } else {
                k.usize_opt("rope.dimension_count")?.unwrap_or(head_dim)
            };
            Ok(RopeSpec::plain(
                behavior.rope,
                n_dims,
                k.f64_opt("rope.freq_base_swa")?.unwrap_or(10_000.0),
            ))
        };

        let mut ropes: Vec<RopeSpec> = Vec::new();
        // The geometry of attention block `il` (a trunk layer, or the
        // prediction block that follows the trunk), reading its K/V from
        // `kv_source`.
        let mut attention_block = |il: usize,
                                   is_sliding: bool,
                                   kv_source: usize|
         -> Result<LayerSpec, CucaError> {
            let head_dim = match (is_sliding, key_length_swa, key_length) {
                (true, Some(d), _) => d,
                (_, _, Some(d)) => d,
                _ => {
                    if n_head[il] == 0 || !n_embd.is_multiple_of(n_head[il]) {
                        return Err(err(format!(
                            "GGUF metadata: embedding length {n_embd} is not divisible by {} heads",
                            n_head[il]
                        )));
                    }
                    n_embd / n_head[il]
                }
            };
            if head_dim == 0 || !head_dim.is_multiple_of(2) {
                return Err(err(
                    "GGUF metadata: attention key lengths must be even and non-zero",
                ));
            }
            if n_head[il] == 0 || n_kv_head[il] == 0 || !n_head[il].is_multiple_of(n_kv_head[il]) {
                return Err(err(format!(
                    "GGUF metadata `{}.attention.head_count_kv`: {} KV heads do not divide {} query heads (layer {il})",
                    arch.name(),
                    n_kv_head[il],
                    n_head[il]
                )));
            }
            let nope = behavior
                .no_rope_interval
                .is_some_and(|n| (il + 1).is_multiple_of(n));
            let rope = if nope {
                None
            } else {
                let spec = if is_sliding {
                    rope_swa(head_dim)?
                } else {
                    rope_full(head_dim)?
                };
                if spec.n_dims == 0 || !spec.n_dims.is_multiple_of(2) || spec.n_dims > head_dim {
                    return Err(err(format!(
                        "GGUF metadata: rope dimension count {} must be even and within the head dimension {head_dim}",
                        spec.n_dims
                    )));
                }
                Some(match ropes.iter().position(|r| *r == spec) {
                    Some(i) => i,
                    None => {
                        ropes.push(spec);
                        ropes.len() - 1
                    }
                })
            };
            let scale = match behavior.attn_scale {
                AttnScale::HeadDim => 1.0 / (head_dim as f64).sqrt(),
                AttnScale::One => 1.0,
                AttnScale::Gemma3 if n_layer == 62 => {
                    1.0 / ((n_embd / n_head[il].max(1)) as f64).sqrt()
                }
                AttnScale::Gemma3 => 1.0 / (head_dim as f64).sqrt(),
                AttnScale::Gemma2 if n_layer == 46 => {
                    1.0 / ((n_embd / n_head[il].max(1)) as f64).sqrt()
                }
                AttnScale::Gemma2 => 1.0 / (head_dim as f64).sqrt(),
                AttnScale::Key => k.f64("attention.scale")?,
            };
            Ok(LayerSpec {
                kind: LayerKind::Attention,
                n_head: n_head[il],
                n_kv_head: n_kv_head[il],
                head_dim,
                window: if is_sliding { window } else { None },
                rope,
                kv_source,
                scale: scale as f32,
            })
        };
        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            if let Some((interval, delta)) = hybrid
                && !(il + 1).is_multiple_of(interval)
            {
                layers.push(LayerSpec {
                    kind: LayerKind::DeltaNet(delta),
                    n_head: delta.n_v_heads,
                    n_kv_head: delta.n_k_heads,
                    head_dim: delta.head_k_dim,
                    window: None,
                    rope: None,
                    kv_source: usize::MAX,
                    scale: 1.0,
                });
                continue;
            }
            layers.push(attention_block(il, sliding[il], kv_source[il])?);
        }
        // The prediction block is full attention with its own K/V.
        let nextn = if nextn_predict_layers > 0 {
            Some(attention_block(n_layer, false, n_layer)?)
        } else {
            None
        };

        let embd_scale = match behavior.embd_scale {
            EmbdScale::None => 1.0,
            EmbdScale::SqrtEmbd => (n_embd as f64).sqrt(),
            EmbdScale::Key => k.f64("embedding_scale")?,
        } as f32;
        let (residual_scale, logit_scale) = if behavior.granite_scales {
            (
                k.f64("residual_scale")? as f32,
                k.f64("logit_scale")? as f32,
            )
        } else {
            (1.0, 1.0)
        };
        let positive = |key: &str| -> Result<Option<f64>, CucaError> {
            match k.f64_opt(key)? {
                Some(v) if !v.is_finite() || v <= 0.0 => Err(meta_err(
                    &format!("{}.{key}", arch.name()),
                    "soft-cap must be positive",
                )),
                other => Ok(other),
            }
        };
        let ple_dim = if behavior.gemma4_extras {
            k.usize_opt("embedding_length_per_layer_input")?
                .filter(|&d| d > 0)
        } else {
            None
        };
        let moe = if behavior.moe {
            let n_expert = k.usize("expert_count")?;
            let n_used = k.usize("expert_used_count")?;
            if n_expert == 0 || n_used == 0 || n_used > n_expert {
                return Err(err(format!(
                    "GGUF metadata: {n_used} used experts of {n_expert} is not a valid top-k"
                )));
            }
            Some(MoeSpec {
                n_expert,
                n_used,
                norm_weights: true,
            })
        } else if k.usize_opt("expert_count")?.is_some_and(|n| n > 0) {
            return Err(err(format!(
                "`{}` checkpoints with experts are not supported",
                arch.name()
            )));
        } else {
            None
        };

        Ok(Self {
            arch,
            behavior,
            n_layer,
            n_embd,
            n_embd_out: if arch.is_drafter() {
                k.usize("embedding_length_out")?
            } else {
                n_embd
            },
            n_ctx_train,
            eps,
            layers,
            ropes,
            n_ctx_orig,
            embd_scale,
            residual_scale,
            logit_scale,
            logit_softcap: positive("final_logit_softcapping")?,
            attn_softcap: positive("attn_logit_softcapping")?,
            ple_dim,
            moe,
            nextn_predict_layers,
            nextn,
        })
    }
}

/// Which layers use sliding-window attention.
fn sliding_layers(
    k: &Keys<'_>,
    behavior: &Behavior,
    n_layer: usize,
) -> Result<Vec<bool>, CucaError> {
    const PATTERN: &str = "attention.sliding_window_pattern";
    match behavior.swa {
        Swa::None => Ok(vec![false; n_layer]),
        Swa::PatternArray => {
            let key = format!("{}.{PATTERN}", k.prefix);
            let Some(Value::Array(items)) = k.value(PATTERN) else {
                return Err(meta_err(&key, "missing or not an array"));
            };
            let flags = items
                .iter()
                .map(|v| v.to_bool().map_err(|e| meta_err(&key, e)))
                .collect::<Result<Vec<bool>, CucaError>>()?;
            if flags.len() != n_layer {
                return Err(meta_err(
                    &key,
                    format!("expected {n_layer} entries, found {}", flags.len()),
                ));
            }
            Ok(flags)
        }
        Swa::Every(default) => {
            let n = match k.value(PATTERN) {
                Some(v) => value_usize(v).filter(|&n| n > 0).ok_or_else(|| {
                    meta_err(&format!("{}.{PATTERN}", k.prefix), "not a positive integer")
                })?,
                None => default,
            };
            Ok((0..n_layer).map(|il| il % n < n - 1).collect())
        }
    }
}

/// For each layer, the layer whose K/V cache it reads: itself below
/// `n_layer_kv`, otherwise the last layer below `n_layer_kv` with the same
/// attention type (sliding or full).
pub(crate) fn kv_sources(is_sliding: &[bool], n_layer_kv: usize) -> Result<Vec<usize>, CucaError> {
    if n_layer_kv == 0 || n_layer_kv > is_sliding.len() {
        return Err(err(format!(
            "{n_layer_kv} K/V-owning layers do not fit {} layers",
            is_sliding.len()
        )));
    }
    (0..is_sliding.len())
        .map(|il| {
            if il < n_layer_kv {
                return Ok(il);
            }
            (0..n_layer_kv)
                .rev()
                .find(|&j| is_sliding[j] == is_sliding[il])
                .ok_or_else(|| {
                    let kind = if is_sliding[il] { "sliding" } else { "full" };
                    err(format!(
                        "layer {il} shares K/V but no earlier {kind}-attention layer owns any"
                    ))
                })
        })
        .collect()
}

/// Build a metadata map from `(key, value)` pairs (test helper).
#[cfg(test)]
pub(crate) fn md(entries: &[(&str, Value)]) -> Metadata {
    entries
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use candle_core::quantized::gguf_file::Value::{Array, Bool, F32, String as Str, U32};

    use super::super::rope::RopeKind;
    use super::*;

    fn none(_: &str) -> bool {
        false
    }

    /// Gemma 4 E2B sliding pattern: every fifth layer (4, 9, ...) is full.
    fn e2b_pattern() -> Vec<bool> {
        (0..35).map(|il| il % 5 != 4).collect()
    }

    pub(crate) fn e2b_metadata() -> Metadata {
        md(&[
            ("general.architecture", Str("gemma4".into())),
            ("gemma4.block_count", U32(35)),
            ("gemma4.context_length", U32(131_072)),
            ("gemma4.embedding_length", U32(1536)),
            ("gemma4.attention.head_count", U32(8)),
            ("gemma4.attention.head_count_kv", U32(1)),
            ("gemma4.attention.key_length", U32(512)),
            ("gemma4.attention.key_length_swa", U32(256)),
            ("gemma4.attention.sliding_window", U32(512)),
            ("gemma4.attention.shared_kv_layers", U32(20)),
            ("gemma4.embedding_length_per_layer_input", U32(256)),
            ("gemma4.rope.freq_base", F32(1e6)),
            ("gemma4.rope.freq_base_swa", F32(1e4)),
            ("gemma4.rope.dimension_count", U32(512)),
            ("gemma4.rope.dimension_count_swa", U32(256)),
            ("gemma4.attention.layer_norm_rms_epsilon", F32(1e-6)),
            ("gemma4.final_logit_softcapping", F32(30.0)),
            (
                "gemma4.attention.sliding_window_pattern",
                Array(e2b_pattern().into_iter().map(Bool).collect()),
            ),
        ])
    }

    fn has_rope_freqs(name: &str) -> bool {
        name == "rope_freqs.weight"
    }

    #[test]
    fn kv_sources_follow_the_e2b_pattern() {
        let sources = kv_sources(&e2b_pattern(), 15).unwrap();
        assert_eq!(sources.len(), 35);
        for (il, &src) in sources.iter().enumerate().take(15) {
            assert_eq!(src, il);
        }
        let pattern = e2b_pattern();
        for il in 15..35 {
            let expected = if pattern[il] { 13 } else { 14 };
            assert_eq!(sources[il], expected, "layer {il}");
        }
    }

    #[test]
    fn kv_sources_reject_a_type_with_no_owner() {
        // Layers 1.. are full but the only K/V owner (layer 0) is sliding.
        let err = kv_sources(&[true, false, false], 1).unwrap_err();
        assert!(err.to_string().contains("no earlier full-attention layer"));
    }

    #[test]
    fn gemma4_e2b_metadata_resolves() {
        let p = Hparams::from_metadata(&e2b_metadata(), &has_rope_freqs).unwrap();
        assert_eq!(p.arch, Arch::Gemma4);
        assert_eq!(p.n_layer, 35);
        assert_eq!(p.kv_owners().count(), 15);
        assert_eq!(p.ple_dim, Some(256));
        assert_eq!(p.logit_softcap, Some(30.0));
        assert!((p.embd_scale - 1536f32.sqrt()).abs() < 1e-3);
        // Layer 0 sliding (head 256, window 512), layer 4 full (head 512).
        assert_eq!(p.layers[0].head_dim, 256);
        assert_eq!(p.layers[0].window, Some(512));
        assert_eq!(p.layers[4].head_dim, 512);
        assert_eq!(p.layers[4].window, None);
        assert_eq!(p.layers[0].scale, 1.0);
        // Two distinct ropes: swa (no factors) and full (rope_freqs).
        assert_eq!(p.ropes.len(), 2);
        let full = &p.ropes[p.layers[4].rope.unwrap()];
        assert_eq!(full.factors, Factors::Freqs);
        assert_eq!(full.n_dims, 512);
        let swa = &p.ropes[p.layers[0].rope.unwrap()];
        assert_eq!(swa.factors, Factors::None);
        assert_eq!(swa.base, 1e4);
        assert_eq!(p.layers[20].kv_source, 13);
        // E2B: 12 sliding (256) and 3 full (512) owners, one KV head.
        assert_eq!(p.kv_bytes_per_position(), 36_864);
    }

    #[test]
    fn missing_and_mistyped_keys_are_named() {
        let mut meta = e2b_metadata();
        meta.remove("gemma4.block_count");
        let msg = Hparams::from_metadata(&meta, &none)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("gemma4.block_count") && msg.contains("missing"),
            "{msg}"
        );

        let mut meta = e2b_metadata();
        meta.insert("gemma4.embedding_length".into(), Str("x".into()));
        let msg = Hparams::from_metadata(&meta, &none)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("gemma4.embedding_length"), "{msg}");
    }

    #[test]
    fn unknown_architecture_is_rejected_by_name() {
        let mut meta = e2b_metadata();
        meta.insert("general.architecture".into(), Str("rwkv6".into()));
        let msg = Hparams::from_metadata(&meta, &none)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("`rwkv6`"), "{msg}");
    }

    #[test]
    fn per_layer_kv_head_arrays_are_accepted() {
        let mut meta = e2b_metadata();
        let counts = (0..35)
            .map(|il| U32(if il % 2 == 0 { 1 } else { 2 }))
            .collect();
        meta.insert("gemma4.attention.head_count_kv".into(), Array(counts));
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        assert_eq!(p.layers[0].n_kv_head, 1);
        assert_eq!(p.layers[1].n_kv_head, 2);
    }

    #[test]
    fn inconsistent_metadata_is_rejected() {
        let reject = |key: &str, value: Value, needle: &str| {
            let mut meta = e2b_metadata();
            meta.insert(key.into(), value);
            let msg = Hparams::from_metadata(&meta, &none)
                .unwrap_err()
                .to_string();
            assert!(msg.contains(needle), "{key}: {msg}");
        };
        reject("gemma4.attention.head_count_kv", U32(3), "do not divide");
        reject(
            "gemma4.attention.value_length",
            U32(128),
            "only equal key and value lengths",
        );
        reject("gemma4.attention.sliding_window", U32(0), "sliding_window");
        reject("gemma4.attention.shared_kv_layers", U32(35), "no K/V owner");
        reject("gemma4.final_logit_softcapping", F32(0.0), "soft-cap");
        reject("gemma4.attention.key_length_swa", U32(255), "even");
        reject("gemma4.rope.dimension_count", U32(1024), "rope dimension");
        reject(
            "gemma4.attention.sliding_window_pattern",
            Array(vec![Bool(true)]),
            "expected 35 entries",
        );
        reject("gemma4.expert_count", U32(8), "experts are not supported");
    }

    fn llama_metadata() -> Metadata {
        md(&[
            ("general.architecture", Str("llama".into())),
            ("llama.block_count", U32(4)),
            ("llama.context_length", U32(1024)),
            ("llama.embedding_length", U32(64)),
            ("llama.attention.head_count", U32(8)),
            ("llama.attention.head_count_kv", U32(2)),
            ("llama.attention.layer_norm_rms_epsilon", F32(1e-5)),
            ("llama.rope.freq_base", F32(500_000.0)),
        ])
    }

    #[test]
    fn llama_defaults_derive_head_dim_and_rope_from_the_embedding() {
        let p = Hparams::from_metadata(&llama_metadata(), &has_rope_freqs).unwrap();
        assert_eq!(p.arch, Arch::Llama);
        assert_eq!(p.layers[0].head_dim, 8);
        assert!((p.layers[0].scale - 1.0 / 8f32.sqrt()).abs() < 1e-6);
        assert_eq!(p.ropes.len(), 1);
        assert_eq!(p.ropes[0].kind, RopeKind::Norm);
        assert_eq!(p.ropes[0].n_dims, 8);
        assert_eq!(p.ropes[0].factors, Factors::Freqs);
        assert!(
            p.layers
                .iter()
                .all(|l| l.window.is_none() && l.rope == Some(0))
        );
        assert_eq!(p.kv_bytes_per_position(), 4 * 2 * 4 * 2 * 8);
    }

    #[test]
    fn gemma3_pattern_scaling_and_swa_rope_defaults() {
        let mut meta = md(&[
            ("general.architecture", Str("gemma3".into())),
            ("gemma3.block_count", U32(12)),
            ("gemma3.context_length", U32(32768)),
            ("gemma3.embedding_length", U32(640)),
            ("gemma3.attention.head_count", U32(4)),
            ("gemma3.attention.head_count_kv", U32(1)),
            ("gemma3.attention.key_length", U32(256)),
            ("gemma3.attention.sliding_window", U32(512)),
            ("gemma3.attention.layer_norm_rms_epsilon", F32(1e-6)),
            ("gemma3.rope.freq_base", F32(1e6)),
        ]);
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        // Layers 5 and 11 are full; the others sliding with the default base.
        for il in 0..12 {
            assert_eq!(p.layers[il].window.is_some(), il % 6 != 5, "layer {il}");
        }
        assert_eq!(p.ropes[p.layers[0].rope.unwrap()].base, 10_000.0);
        assert_eq!(p.ropes[p.layers[5].rope.unwrap()].base, 1e6);
        assert_eq!(p.ropes[p.layers[5].rope.unwrap()].freq_scale, 1.0);
        assert!((p.layers[0].scale - 1.0 / 16.0).abs() < 1e-7);
        assert_eq!(p.logit_softcap, None);
        // 4B+ checkpoints scale full-attention RoPE linearly; sliding layers don't.
        meta.insert("gemma3.rope.scaling.type".into(), Str("linear".into()));
        meta.insert("gemma3.rope.scaling.factor".into(), F32(8.0));
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        assert_eq!(p.ropes[p.layers[5].rope.unwrap()].freq_scale, 0.125);
        assert_eq!(p.ropes[p.layers[0].rope.unwrap()].freq_scale, 1.0);
    }

    #[test]
    fn gemma2_alternates_windows_and_reads_the_soft_caps() {
        let mut meta = md(&[
            ("general.architecture", Str("gemma2".into())),
            ("gemma2.block_count", U32(26)),
            ("gemma2.context_length", U32(8192)),
            ("gemma2.embedding_length", U32(2304)),
            ("gemma2.attention.head_count", U32(8)),
            ("gemma2.attention.head_count_kv", U32(4)),
            ("gemma2.attention.key_length", U32(256)),
            ("gemma2.attention.sliding_window", U32(4096)),
            ("gemma2.attention.layer_norm_rms_epsilon", F32(1e-6)),
            ("gemma2.attn_logit_softcapping", F32(50.0)),
            ("gemma2.final_logit_softcapping", F32(30.0)),
            ("gemma2.rope.freq_base", F32(10_000.0)),
        ]);
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        for il in 0..26 {
            assert_eq!(p.layers[il].window.is_some(), il % 2 == 0, "layer {il}");
        }
        assert_eq!(p.attn_softcap, Some(50.0));
        assert_eq!(p.logit_softcap, Some(30.0));
        assert!((p.layers[0].scale - 1.0 / 16.0).abs() < 1e-7);
        assert_eq!(p.ropes.len(), 1, "both layer types share one rotary base");
        // The 46-layer 27B model scales by the embedding per head instead.
        meta.insert("gemma2.block_count".into(), U32(46));
        meta.insert("gemma2.embedding_length".into(), U32(4608));
        meta.insert("gemma2.attention.head_count".into(), U32(32));
        meta.insert("gemma2.attention.head_count_kv".into(), U32(16));
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        assert!((p.layers[0].scale - 1.0 / 144f32.sqrt()).abs() < 1e-7);
    }

    #[test]
    fn olmo2_has_only_post_norms_and_full_width_qk_norm() {
        let b = Arch::Olmo2.behavior();
        assert!(!b.pre_norms && b.post_norms && b.qk_norm_full);
        assert_eq!(b.rope, RopeKind::Neox);
        for arch in [Arch::Llama, Arch::Gemma2, Arch::Granite, Arch::Ernie4_5] {
            let b = arch.behavior();
            assert!(b.pre_norms && !b.qk_norm_full, "{arch:?}");
        }
    }

    #[test]
    fn granite_dense_reads_the_scales_without_experts() {
        let mut meta = md(&[
            ("general.architecture", Str("granite".into())),
            ("granite.block_count", U32(2)),
            ("granite.context_length", U32(4096)),
            ("granite.embedding_length", U32(64)),
            ("granite.attention.head_count", U32(8)),
            ("granite.attention.head_count_kv", U32(2)),
            ("granite.attention.layer_norm_rms_epsilon", F32(1e-5)),
            ("granite.attention.scale", F32(0.0078125)),
            ("granite.embedding_scale", F32(12.0)),
            ("granite.residual_scale", F32(0.22)),
            ("granite.logit_scale", F32(8.0)),
        ]);
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        assert!(p.moe.is_none());
        assert_eq!(
            (p.embd_scale, p.residual_scale, p.logit_scale),
            (12.0, 0.22, 8.0)
        );
        assert!((p.layers[0].scale - 0.0078125).abs() < 1e-9);
        meta.insert("granite.expert_count".into(), U32(8));
        assert!(Hparams::from_metadata(&meta, &none).is_err());
    }

    #[test]
    fn smollm3_marks_every_fourth_layer_nope() {
        let meta = md(&[
            ("general.architecture", Str("smollm3".into())),
            ("smollm3.block_count", U32(8)),
            ("smollm3.context_length", U32(1024)),
            ("smollm3.embedding_length", U32(64)),
            ("smollm3.attention.head_count", U32(8)),
            ("smollm3.attention.head_count_kv", U32(2)),
            ("smollm3.attention.layer_norm_rms_epsilon", F32(1e-6)),
        ]);
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        for il in 0..8 {
            assert_eq!(p.layers[il].rope.is_none(), il % 4 == 3, "layer {il}");
        }
    }

    #[test]
    fn longrope_and_yarn_keys_resolve() {
        let mut meta = md(&[
            ("general.architecture", Str("phi3".into())),
            ("phi3.block_count", U32(2)),
            ("phi3.context_length", U32(131_072)),
            ("phi3.rope.scaling.original_context_length", U32(4096)),
            ("phi3.embedding_length", U32(96)),
            ("phi3.attention.head_count", U32(3)),
            ("phi3.attention.head_count_kv", U32(3)),
            ("phi3.attention.layer_norm_rms_epsilon", F32(1e-5)),
            ("phi3.rope.dimension_count", U32(24)),
            ("phi3.attention.sliding_window", U32(262_144)),
            ("phi3.rope.scaling.attn_factor", F32(1.19)),
        ]);
        let long_short = |n: &str| n.starts_with("rope_factors_");
        let p = Hparams::from_metadata(&meta, &long_short).unwrap();
        let r = &p.ropes[0];
        assert_eq!(
            (r.kind, r.n_dims, r.factors),
            (RopeKind::Neox, 24, Factors::LongShort)
        );
        assert!((r.attn_factor - 1.19).abs() < 1e-6);
        assert!(
            p.layers.iter().all(|l| l.window.is_none()),
            "Phi SWA stays disabled"
        );
        assert_eq!(p.n_ctx_orig, 4096);

        meta.insert("phi3.rope.scaling.type".into(), Str("yarn".into()));
        meta.insert("phi3.rope.scaling.factor".into(), F32(4.0));
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        let r = &p.ropes[0];
        assert_eq!(r.freq_scale, 0.25);
        assert_eq!(
            r.yarn.map(|y| (y.n_ctx_orig, y.beta_fast)),
            Some((4096, 32.0))
        );
        meta.insert("phi3.rope.scaling.type".into(), Str("dynamic".into()));
        assert!(Hparams::from_metadata(&meta, &none).is_err());
    }

    #[test]
    fn granitemoe_reads_scales_and_experts() {
        let meta = md(&[
            ("general.architecture", Str("granitemoe".into())),
            ("granitemoe.block_count", U32(2)),
            ("granitemoe.context_length", U32(4096)),
            ("granitemoe.embedding_length", U32(64)),
            ("granitemoe.attention.head_count", U32(4)),
            ("granitemoe.attention.head_count_kv", U32(2)),
            ("granitemoe.attention.layer_norm_rms_epsilon", F32(1e-6)),
            ("granitemoe.attention.scale", F32(0.015625)),
            ("granitemoe.embedding_scale", F32(12.0)),
            ("granitemoe.residual_scale", F32(0.22)),
            ("granitemoe.logit_scale", F32(6.0)),
            ("granitemoe.expert_count", U32(8)),
            ("granitemoe.expert_used_count", U32(2)),
        ]);
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        assert_eq!(p.layers[0].scale, 0.015625);
        assert_eq!(
            (p.embd_scale, p.residual_scale, p.logit_scale),
            (12.0, 0.22, 6.0)
        );
        assert_eq!(
            p.moe,
            Some(MoeSpec {
                n_expert: 8,
                n_used: 2,
                norm_weights: true
            })
        );
        let mut bad = meta.clone();
        bad.insert("granitemoe.expert_used_count".into(), U32(9));
        assert!(Hparams::from_metadata(&bad, &none).is_err());
    }

    /// The published Qwen3.5-0.8B metadata.
    fn qwen35_metadata() -> Metadata {
        md(&[
            ("general.architecture", Str("qwen35".into())),
            ("qwen35.block_count", U32(25)),
            ("qwen35.context_length", U32(262_144)),
            ("qwen35.embedding_length", U32(1024)),
            ("qwen35.feed_forward_length", U32(3584)),
            ("qwen35.attention.head_count", U32(8)),
            ("qwen35.attention.head_count_kv", U32(2)),
            ("qwen35.attention.key_length", U32(256)),
            ("qwen35.attention.value_length", U32(256)),
            ("qwen35.attention.layer_norm_rms_epsilon", F32(1e-6)),
            (
                "qwen35.rope.dimension_sections",
                Array(vec![U32(11), U32(11), U32(10), U32(0)]),
            ),
            ("qwen35.rope.freq_base", F32(1e7)),
            ("qwen35.rope.dimension_count", U32(64)),
            ("qwen35.ssm.conv_kernel", U32(4)),
            ("qwen35.ssm.state_size", U32(128)),
            ("qwen35.ssm.group_count", U32(16)),
            ("qwen35.ssm.time_step_rank", U32(16)),
            ("qwen35.ssm.inner_size", U32(2048)),
            ("qwen35.full_attention_interval", U32(4)),
            ("qwen35.nextn_predict_layers", U32(1)),
        ])
    }

    #[test]
    fn qwen35_resolves_hybrid_layers_and_drops_the_prediction_block() {
        let p = Hparams::from_metadata(&qwen35_metadata(), &none).unwrap();
        assert_eq!(p.arch, Arch::Qwen35);
        // 25 blocks, the last one is the MTP head.
        assert_eq!((p.n_layer, p.nextn_predict_layers), (24, 1));
        assert_eq!(p.layers.len(), 24);
        for (il, l) in p.layers.iter().enumerate() {
            let attention = (il + 1) % 4 == 0;
            assert_eq!(l.delta().is_none(), attention, "layer {il}");
            assert_eq!(l.owns_kv(il), attention, "layer {il}");
            assert_eq!(l.rope.is_some(), attention, "layer {il}");
        }
        let delta = *p.layers[0].delta().unwrap();
        assert_eq!(
            (
                delta.n_k_heads,
                delta.n_v_heads,
                delta.head_k_dim,
                delta.head_v_dim
            ),
            (16, 16, 128, 128)
        );
        assert_eq!(
            (delta.key_dim(), delta.value_dim(), delta.conv_dim()),
            (2048, 2048, 6144)
        );
        assert!(p.is_hybrid());
        // Attention layers: 8 heads of 256 over 2 KV heads, scale 1/16.
        let attn = &p.layers[3];
        assert_eq!((attn.n_head, attn.n_kv_head, attn.head_dim), (8, 2, 256));
        assert!((attn.scale - 1.0 / 16.0).abs() < 1e-9);
        // Partial NEOX rope over 64 of the 256 dimensions, sections ignored.
        assert_eq!(p.ropes.len(), 1);
        assert_eq!(p.ropes[0].kind, RopeKind::Neox);
        assert_eq!((p.ropes[0].n_dims, p.ropes[0].base), (64, 1e7));
        // Only the six attention layers store K/V: 6 x (2 x 4 x 2 x 256).
        assert_eq!(p.kv_bytes_per_position(), 6 * 4096);
        // The prediction block (block 24) is a full-attention block of the
        // same geometry that owns its K/V and shares the trunk's rope; it is
        // not a trunk layer, so the count above leaves it out.
        let head = p.nextn.as_ref().unwrap();
        assert_eq!(head.delta(), None);
        assert_eq!((head.n_head, head.n_kv_head, head.head_dim), (8, 2, 256));
        assert!(head.owns_kv(24));
        assert_eq!((head.rope, head.window), (Some(0), None));
        assert!((head.scale - 1.0 / 16.0).abs() < 1e-9);
        // 18 delta layers x ((4 - 1) x 6144 + 16 x 128 x 128) f32.
        assert_eq!(delta.state_floats(), 3 * 6144 + 16 * 128 * 128);
        assert_eq!(p.recurrent_state_bytes(), 18 * 4 * delta.state_floats());
        assert_eq!(p.recurrent_state_bytes(), 20_201_472);
    }

    #[test]
    fn qwen35_without_prediction_blocks_keeps_every_block() {
        let mut meta = qwen35_metadata();
        meta.remove("qwen35.nextn_predict_layers");
        meta.insert("qwen35.block_count".into(), U32(24));
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        assert_eq!((p.n_layer, p.nextn_predict_layers), (24, 0));
        assert!(p.nextn.is_none());
    }

    #[test]
    fn per_layer_head_arrays_may_cover_the_prediction_block() {
        let mut meta = qwen35_metadata();
        // Recurrent layers may carry 0 heads; the trailing block is covered.
        let kv_heads = (0..25)
            .map(|il| U32(if il % 4 == 3 || il == 24 { 2 } else { 0 }))
            .collect();
        meta.insert("qwen35.attention.head_count_kv".into(), Array(kv_heads));
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        assert_eq!(p.n_layer, 24);
        assert_eq!(p.layers[3].n_kv_head, 2);
        let short = (0..24).map(|_| U32(2)).collect();
        meta.insert("qwen35.attention.head_count_kv".into(), Array(short));
        let msg = Hparams::from_metadata(&meta, &none)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("expected 25 entries"), "{msg}");
    }

    #[test]
    fn inconsistent_qwen35_metadata_is_rejected_by_key() {
        let reject = |key: &str, value: Option<Value>, needle: &str| {
            let mut meta = qwen35_metadata();
            match value {
                Some(v) => meta.insert(key.into(), v),
                None => meta.remove(key),
            };
            let msg = Hparams::from_metadata(&meta, &none)
                .unwrap_err()
                .to_string();
            assert!(msg.contains(key), "{key}: {msg}");
            assert!(msg.contains(needle), "{key}: {msg}");
        };
        for key in [
            "qwen35.full_attention_interval",
            "qwen35.ssm.conv_kernel",
            "qwen35.ssm.state_size",
            "qwen35.ssm.group_count",
            "qwen35.ssm.time_step_rank",
            "qwen35.ssm.inner_size",
        ] {
            reject(key, None, "missing");
            reject(key, Some(U32(0)), "non-zero");
        }
        reject("qwen35.ssm.time_step_rank", Some(U32(24)), "not a multiple");
        reject("qwen35.ssm.inner_size", Some(U32(2050)), "not a multiple");
        reject(
            "qwen35.nextn_predict_layers",
            Some(U32(25)),
            "no trunk layer",
        );
        reject(
            "qwen35.nextn_predict_layers",
            Some(Str("x".into())),
            "unsigned",
        );
        // A drafter-less architecture ignores the key it does not embed.
        let mut llama = llama_metadata();
        llama.insert("llama.nextn_predict_layers".into(), U32(9));
        assert_eq!(Hparams::from_metadata(&llama, &none).unwrap().n_layer, 4);
    }

    #[test]
    fn value_heads_may_outnumber_key_heads() {
        let mut meta = qwen35_metadata();
        meta.insert("qwen35.ssm.group_count".into(), U32(4));
        let p = Hparams::from_metadata(&meta, &none).unwrap();
        let delta = p.layers[0].delta().unwrap();
        assert_eq!((delta.n_k_heads, delta.n_v_heads), (4, 16));
        assert_eq!(delta.conv_dim(), 2 * 4 * 128 + 2048);
        // Value width follows inner_size / time_step_rank, not state_size.
        meta.insert("qwen35.ssm.inner_size".into(), U32(4096));
        let delta = *Hparams::from_metadata(&meta, &none).unwrap().layers[0]
            .delta()
            .unwrap();
        assert_eq!((delta.head_k_dim, delta.head_v_dim), (128, 256));
    }
}
