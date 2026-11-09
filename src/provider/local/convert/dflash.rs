//! Conversion of a DFlash drafter checkpoint to the `dflash` GGUF layout.
//!
//! The checkpoint is transformers' `MuseGlimmerAssistantModel`
//! (`meta-models/Muse-Glimmer-30B-assistant`, `model_type`
//! `muse_glimmer_assistant`): five all-sliding decoder layers that read a
//! block of target tokens and the target's hidden features, with no
//! embedding table and no LM head (see `model::cpu::dflash`). The output is
//! the layout of llama.cpp's `MuseGlimmerAssistantModel` converter
//! (`conversion/muse_glimmer.py`, arch `dflash`), minus the target's
//! vocabulary, which the engine takes from the target model instead:
//!
//! | GGUF | checkpoint |
//! |---|---|
//! | `fc.weight` | `encoder.fc.weight` `[n_embd, n_features * n_embd]` |
//! | `enc.output_norm.weight` | `encoder.output_norm_enc.weight` |
//! | `output_norm.weight` | `norm.weight` |
//! | `blk.N.attn_norm.weight` | `layers.N.input_layernorm.weight` |
//! | `blk.N.attn_q`, `attn_k`, `attn_v`, `attn_output` | `layers.N.self_attn.{q,k,v,o}_proj.weight` |
//! | `blk.N.attn_q_norm`, `attn_k_norm` | `layers.N.self_attn.{q,k}_norm.weight` |
//! | `blk.N.ffn_norm.weight` | `layers.N.post_attention_layernorm.weight` |
//! | `blk.N.ffn_gate`, `ffn_up`, `ffn_down` | `layers.N.mlp.{gate,up,down}_proj.weight` |
//!
//! Every tensor keeps its values and its row-major `[out, in]` shape: the
//! weights use the half-split (NEOX) rotary layout transformers does, so
//! nothing is permuted, and the norms are plain weights (no `+ 1`, no Q/K
//! scale). Matrices are stored at the requested precision, norms as `F32`.
//!
//! The metadata is `general.*`, `dflash.{block_count, context_length,
//! embedding_length, feed_forward_length, attention.{head_count,
//! head_count_kv, key_length, value_length, layer_norm_rms_epsilon,
//! sliding_window, sliding_window_pattern}, rope.freq_base, block_size,
//! target_layers}` and `tokenizer.ggml.mask_token_id`.
//!
//! `dflash.target_layers` holds llama.cpp's convention, the HF
//! `target_layer_ids` plus one: entry `i` names the layer whose *input* feeds
//! the drafter, which is the output of layer `i - 1` (transformers' ids are
//! outputs: `[1, 13, 25, 37, 49]` becomes `[2, 14, 26, 38, 50]`).

use std::path::Path;

use crate::provider::local::gguf::Value;
use serde_json::Value as Json;

use super::cfg_err;
use super::config::{Hp, file_type};
use super::hf::CheckpointDir;
use super::safetensors::Shards;
use super::tensors::{Body, Plan, Slice};
use crate::error::CucaError;
use crate::provider::local::model::cpu::dflash::{ARCH, check_metadata};
use crate::provider::local::{DownloadProgressCallback, WeightQuant};

/// The class of the checkpoints this converter reads.
const CLASS: &str = "MuseGlimmerAssistantModel";
/// Its `model_type`.
const MODEL_TYPE: &str = "muse_glimmer_assistant";

/// Whether `config` (a `config.json`) is a DFlash drafter this module
/// converts.
pub(crate) fn is_dflash(config: &Json) -> bool {
    config
        .get("architectures")
        .and_then(Json::as_array)
        .is_some_and(|a| a.iter().any(|c| c.as_str() == Some(CLASS)))
        || config.get("model_type").and_then(Json::as_str) == Some(MODEL_TYPE)
}

/// `config` validated as a convertible architecture: what
/// `ConvArch::from_config` is to the other checkpoints.
///
/// # Errors
///
/// [`CucaError::Config`] for an architecture without a converter.
pub(crate) fn check_architecture(config: &Json) -> Result<(), CucaError> {
    if is_dflash(config) {
        Ok(())
    } else {
        super::arch::ConvArch::from_config(config).map(|_| ())
    }
}

/// What `config.json` says (transformers' `MuseGlimmerAssistantConfig`
/// defaults filled in).
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
}

impl Config {
    fn read(hp: &Hp) -> Result<Self, CucaError> {
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
        let rope = hp.rope_all();
        if let Some(kind) = rope.get("rope_type").and_then(Json::as_str)
            && kind != "default"
        {
            return Err(cfg_err(format!(
                "the DFlash drafter's rope type `{kind}` is not supported (only `default`)"
            )));
        }
        let sliding = match hp.get("layer_types") {
            None => vec![true; n_layer],
            Some(Json::Array(types)) if types.len() == n_layer => types
                .iter()
                .map(|t| match t.as_str() {
                    Some("sliding_attention") => Ok(true),
                    Some("full_attention") => Ok(false),
                    other => Err(cfg_err(format!("layer type {other:?} is not supported"))),
                })
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => {
                return Err(cfg_err(format!(
                    "layer_types does not list {n_layer} layers"
                )));
            }
        };
        let target_ids = match hp.get("target_layer_ids") {
            None => vec![1, 13, 25, 37, 49],
            Some(Json::Array(ids)) => ids
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| cfg_err("target_layer_ids holds a non-integer"))
                })
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err(cfg_err("target_layer_ids is not a list")),
        };
        if target_ids.is_empty() || target_ids.windows(2).any(|w| w[0] >= w[1]) {
            return Err(cfg_err(
                "target_layer_ids must be a non-empty ascending list",
            ));
        }
        Ok(Self {
            n_layer,
            n_embd,
            n_head,
            n_kv_head: hp.uint(&["num_key_value_heads"]).unwrap_or(n_head),
            head_dim: hp.head_dim()?,
            n_ff: hp.req_uint(&["intermediate_size"])?,
            eps: hp.float(&["rms_norm_eps"]).unwrap_or(1e-5),
            rope_theta: rope
                .get("rope_theta")
                .and_then(Json::as_f64)
                .unwrap_or(500_000.0),
            max_positions: hp.uint(&["max_position_embeddings"]).unwrap_or(131_072),
            block_size: hp.uint(&["block_size"]).unwrap_or(16),
            mask_token: hp.uint(&["mask_token_id"]).unwrap_or(201_818),
            target_ids,
            window: sliding
                .iter()
                .any(|&s| s)
                .then(|| hp.uint(&["sliding_window"]).unwrap_or(2048)),
            sliding,
        })
    }

    /// The `dflash.*` and tokenizer metadata, llama.cpp's.
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
        kv.push(u("tokenizer.ggml.mask_token_id", self.mask_token)?);
        kv.push(("general.quantization_version".to_string(), Value::U32(2)));
        kv.push((
            "general.file_type".to_string(),
            Value::U32(file_type(quant)),
        ));
        Ok(kv)
    }

    /// Every tensor of the checkpoint as `(checkpoint name, GGUF name, shape)`.
    fn expected(&self) -> Vec<(String, String, Vec<usize>)> {
        let (n, d) = (self.n_embd, self.head_dim);
        let (qd, kd) = (self.n_head * d, self.n_kv_head * d);
        let t = |hf: &str, gguf: &str, shape: Vec<usize>| (hf.to_string(), gguf.to_string(), shape);
        let mut tensors = vec![
            t(
                "encoder.fc.weight",
                "fc.weight",
                vec![n, self.target_ids.len() * n],
            ),
            t(
                "encoder.output_norm_enc.weight",
                "enc.output_norm.weight",
                vec![n],
            ),
            t("norm.weight", "output_norm.weight", vec![n]),
        ];
        for il in 0..self.n_layer {
            let l = |hf: &str, gguf: &str, shape: Vec<usize>| {
                t(
                    &format!("layers.{il}.{hf}.weight"),
                    &format!("blk.{il}.{gguf}.weight"),
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
        tensors
    }
}

/// One plan per tensor, checked against the config: a missing tensor, an
/// unknown one (an embedding table or head of its own, a variant this layout
/// does not hold) or a wrong shape is an error naming it.
fn plan(cfg: &Config, shards: &Shards) -> Result<Vec<Plan>, CucaError> {
    let expected = cfg.expected();
    for name in shards.names() {
        if !expected.iter().any(|(hf, ..)| hf == name) {
            return Err(cfg_err(format!(
                "tensor `{name}` of the DFlash drafter has no GGUF name (the drafter holds no embedding table or head: it reads the target's)"
            )));
        }
    }
    let mut plans = Vec::with_capacity(expected.len());
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
    Ok(plans)
}

/// Write the GGUF of the DFlash drafter `ck` to `partial`.
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
    let cfg = Config::read(&hp)?;
    let shards = Shards::open(&ck.shards)?;
    let plans = plan(&cfg, &shards)?;
    let kv = cfg.metadata(&super::model_name(&ck.dir), quant)?;
    // The engine's own reader validates the metadata it will read.
    let md = kv.iter().cloned().collect();
    check_metadata(&md)
        .map_err(|e| cfg_err(format!("the converted metadata does not load: {e}")))?;

    super::write_plans(partial, &kv, &plans, &shards, quant, progress)
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
        assert!(!is_dflash(
            &serde_json::json!({"architectures": ["MuseGlimmerForConditionalGeneration"]})
        ));
        assert!(check_architecture(&config()).is_ok());
        let err = check_architecture(&serde_json::json!({"architectures": ["PixtralModel"]}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("PixtralModel"), "{err}");
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
}
