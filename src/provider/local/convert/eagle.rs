//! The Mistral Small 4 EAGLE drafter in Mistral's own checkpoint format.
//!
//! `mistralai/Mistral-Small-4-119B-2603-eagle` is published as
//! `consolidated.safetensors` (Mistral tensor names, FP8 `e4m3` weights with a
//! per-tensor `qscale_weight`/`qscale_act` pair) next to a `params.json`; it has
//! no `config.json`. [`config_from_params`] derives the Hugging Face style
//! config the rest of the converter reads, exactly as Mistral's own
//! `convert_mistral4_weight_to_hf.py` maps `params.json` for the full model
//! (including its constants: `mscale = mscale_all_dim = 1`, so the softmax
//! scale carries the YaRN magnitude twice, as in the target), and refuses
//! every other Mistral-format checkpoint.
//!
//! The drafter is two DeepSeek-style latent-attention layers over a *dense*
//! gated FFN (`moe` is `null`), a `2 * dim -> dim` fusion projection
//! (`eagle_linear`) and a final norm; its embedding and LM head are the
//! target's. The GGUF it converts to (architecture `mistral4-eagle`) keeps
//! the tensor names the engine reads (`model::layout`, `nextn.eh_proj.weight`
//! for the fusion projection); `plan_eagle` in `tensors.rs` maps them.

use serde_json::{Map, Value, json};

use crate::error::CucaError;

/// `architectures[0]` of the config [`config_from_params`] derives: the name
/// of the class vLLM runs this checkpoint with.
pub(crate) const ARCHITECTURE: &str = "EagleMistralLarge3ForCausalLM";

/// The Mistral-format hyperparameter file.
pub(crate) const PARAMS_FILE: &str = "params.json";

/// The weights file of a Mistral-format checkpoint.
pub(crate) const WEIGHTS_FILE: &str = "consolidated.safetensors";

/// `max_position_embeddings` of the Mistral Small 4 family, for a
/// `params.json` that names none.
const DEFAULT_MAX_POSITIONS: u64 = 1_048_576;

fn cfg_err(msg: impl Into<String>) -> CucaError {
    CucaError::Config(format!("params.json: {}", msg.into()))
}

fn field<'a>(p: &'a Map<String, Value>, key: &str) -> Result<&'a Value, CucaError> {
    p.get(key)
        .filter(|v| !v.is_null())
        .ok_or_else(|| cfg_err(format!("`{key}` is missing")))
}

fn uint(p: &Map<String, Value>, key: &str) -> Result<u64, CucaError> {
    field(p, key)?
        .as_u64()
        .ok_or_else(|| cfg_err(format!("`{key}` is not an unsigned integer")))
}

fn float(p: &Map<String, Value>, key: &str) -> Result<f64, CucaError> {
    field(p, key)?
        .as_f64()
        .ok_or_else(|| cfg_err(format!("`{key}` is not a number")))
}

fn object<'a>(p: &'a Map<String, Value>, key: &str) -> Result<&'a Map<String, Value>, CucaError> {
    field(p, key)?
        .as_object()
        .ok_or_else(|| cfg_err(format!("`{key}` is not an object")))
}

/// The Hugging Face style `config.json` of the EAGLE drafter described by the
/// parsed `params.json`.
///
/// # Errors
///
/// [`CucaError::Config`] when `params` is not the dense EAGLE drafter (a
/// mixture-of-experts model, a vision tower, a YaRN `apply_scale` or a Llama 4
/// scaling length that differ from what the engine implements) or lacks a
/// field.
pub(crate) fn config_from_params(params: &Value) -> Result<Value, CucaError> {
    let Value::Object(p) = params else {
        return Err(cfg_err("not a JSON object"));
    };
    if p.get("moe").is_some_and(|m| !m.is_null()) {
        return Err(cfg_err(
            "describes a mixture-of-experts model: the Mistral-format files of a full Mistral Small 4 are not converted (use its Hugging Face files, config.json and model-*.safetensors); only the dense EAGLE drafter is read from params.json",
        ));
    }
    if p.get("vision_encoder").is_some_and(|v| !v.is_null()) {
        return Err(cfg_err(
            "describes a vision tower, which a drafter has none",
        ));
    }
    let yarn = object(p, "yarn")?;
    if yarn.get("apply_scale").and_then(Value::as_bool) == Some(true) {
        return Err(cfg_err(
            "`yarn.apply_scale` is true: Mistral's conversion applies no cos/sin scale for these checkpoints, and the engine implements only that",
        ));
    }
    let original = uint(yarn, "original_max_position_embeddings")?;
    // The Llama 4 query scaling takes the YaRN original context as its
    // length (transformers and vLLM read no other).
    let scaling = object(p, "llama_4_scaling")?;
    if uint(scaling, "original_max_position_embeddings")? != original {
        return Err(cfg_err(
            "`llama_4_scaling.original_max_position_embeddings` differs from `yarn.original_max_position_embeddings`",
        ));
    }
    let dim = uint(p, "dim")?;
    let rope_theta = p
        .get("rope_theta")
        .and_then(Value::as_f64)
        .unwrap_or(10_000.0);
    let mut config = json!({
        "architectures": [ARCHITECTURE],
        "model_type": "mistral4",
        "hidden_size": dim,
        "num_hidden_layers": uint(p, "n_layers")?,
        "intermediate_size": uint(p, "hidden_dim")?,
        "num_attention_heads": uint(p, "n_heads")?,
        "num_key_value_heads": uint(p, "n_kv_heads")?,
        "rms_norm_eps": float(p, "norm_eps")?,
        "vocab_size": uint(p, "vocab_size")?,
        "tie_word_embeddings": p.get("tied_embeddings").and_then(Value::as_bool).unwrap_or(false),
        "max_position_embeddings": p
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_MAX_POSITIONS),
        "q_lora_rank": uint(p, "q_lora_rank")?,
        "kv_lora_rank": uint(p, "kv_lora_rank")?,
        "qk_nope_head_dim": uint(p, "qk_nope_head_dim")?,
        "qk_rope_head_dim": uint(p, "qk_rope_head_dim")?,
        "v_head_dim": uint(p, "v_head_dim")?,
        "rope_parameters": {
            "type": "yarn",
            "rope_type": "yarn",
            "rope_theta": rope_theta,
            "factor": float(yarn, "factor")?,
            "original_max_position_embeddings": original,
            "beta_fast": float(yarn, "beta")?,
            "beta_slow": float(yarn, "alpha")?,
            "mscale_all_dim": 1.0,
            "mscale": 1.0,
            "llama_4_scaling_beta": float(scaling, "beta")?,
        },
    });
    if let (Some(head_dim), Value::Object(map)) = (p.get("head_dim"), &mut config) {
        map.insert("head_dim".into(), head_dim.clone());
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, gguf_file};

    use super::super::hf::CheckpointDir;
    use super::super::safetensors::Shards;
    use super::*;
    use crate::provider::local::WeightQuant;

    /// The published drafter's `params.json` (`mistralai/Mistral-Small-4-119B-2603-eagle`).
    fn published() -> Value {
        json!({
            "dim": 4096, "n_layers": 2, "head_dim": 128, "hidden_dim": 12288,
            "n_heads": 32, "n_kv_heads": 32, "rope_theta": 10000.0, "norm_eps": 1e-06,
            "vocab_size": 131072, "tied_embeddings": false, "max_position_embeddings": 1048576,
            "llama_4_scaling": {"original_max_position_embeddings": 8192, "beta": 0.1},
            "q_lora_rank": 1024, "qk_rope_head_dim": 64, "qk_nope_head_dim": 64,
            "kv_lora_rank": 256, "v_head_dim": 128,
            "quantization": {"qformat_weight": "fp8_e4m3", "qscheme_act": "TENSOR"},
            "yarn": {"original_max_position_embeddings": 8192, "factor": 128,
                     "apply_scale": false, "beta": 32, "alpha": 1},
            "moe": null, "vision_encoder": null
        })
    }

    #[test]
    fn the_published_params_become_the_targets_config_values() {
        let c = config_from_params(&published()).unwrap();
        assert_eq!(c["architectures"][0], ARCHITECTURE);
        assert_eq!(
            (c["hidden_size"].as_u64(), c["intermediate_size"].as_u64()),
            (Some(4096), Some(12288))
        );
        assert_eq!(c["num_hidden_layers"], 2);
        assert_eq!(c["head_dim"], 128);
        // The same rope parameters Mistral's converter writes for the target.
        let r = &c["rope_parameters"];
        assert_eq!(r["factor"], 128.0);
        assert_eq!(r["original_max_position_embeddings"], 8192);
        assert_eq!(
            (r["beta_fast"].as_f64(), r["beta_slow"].as_f64()),
            (Some(32.0), Some(1.0))
        );
        assert_eq!(
            (r["mscale"].as_f64(), r["mscale_all_dim"].as_f64()),
            (Some(1.0), Some(1.0))
        );
        assert_eq!(r["llama_4_scaling_beta"], 0.1);
        assert_eq!(r["rope_theta"], 10000.0);
    }

    #[test]
    fn anything_but_the_dense_drafter_is_refused() {
        let with = |key: &str, value: Value| {
            let mut p = published();
            p[key] = value;
            config_from_params(&p).unwrap_err().to_string()
        };
        assert!(with("moe", json!({"num_experts": 128})).contains("mixture-of-experts"));
        assert!(with("vision_encoder", json!({"hidden_size": 1024})).contains("vision"));
        let mut p = published();
        p["yarn"]["apply_scale"] = json!(true);
        assert!(
            config_from_params(&p)
                .unwrap_err()
                .to_string()
                .contains("apply_scale")
        );
        let mut p = published();
        p["llama_4_scaling"]["original_max_position_embeddings"] = json!(4096);
        assert!(
            config_from_params(&p)
                .unwrap_err()
                .to_string()
                .contains("llama_4_scaling")
        );
        let mut p = published();
        p.as_object_mut().unwrap().remove("q_lora_rank");
        assert!(
            config_from_params(&p)
                .unwrap_err()
                .to_string()
                .contains("q_lora_rank")
        );
        assert!(config_from_params(&json!([1])).is_err());
    }

    fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/provider/local/testdata/eagle")
    }

    fn convert_fixture(tag: &str, quant: WeightQuant) -> (gguf_file::Content, std::path::PathBuf) {
        let out = std::env::temp_dir().join(format!(
            "cuca-eagle-{tag}-{}-{}.gguf",
            std::process::id(),
            quant.name()
        ));
        let ck = CheckpointDir::open(&fixture_dir()).unwrap();
        super::super::convert(&ck, &out, quant, None).unwrap();
        let content = gguf_file::Content::read(&mut std::fs::File::open(&out).unwrap()).unwrap();
        (content, out)
    }

    /// The GGUF layout of `mistral4-eagle`: metadata, tensor names,
    /// shapes and dtypes at each precision (the norms are `f32`; the per-head
    /// up-projections, whose rows are narrower than a quantization block,
    /// stay `f32` under a block format and are `f16` under `f16`).
    #[test]
    fn the_fixture_converts_to_the_documented_layout() {
        for (quant, matrix) in [
            (WeightQuant::F16, GgmlDType::F16),
            (WeightQuant::Q8_0, GgmlDType::Q8_0),
            (WeightQuant::Q4_0, GgmlDType::Q4_0),
        ] {
            let (content, out) = convert_fixture("layout", quant);
            let md = &content.metadata;
            let p = "mistral4-eagle";
            assert_eq!(
                md["general.architecture"].to_string().unwrap(),
                &String::from(p)
            );
            let uint = |key: &str| md[&format!("{p}.{key}")].to_u32().unwrap();
            let float = |key: &str| md[&format!("{p}.{key}")].to_f32().unwrap();
            for (key, want) in [
                ("block_count", 2),
                ("embedding_length", 64),
                ("embedding_length_out", 64),
                ("vocab_size", 16),
                ("feed_forward_length", 96),
                ("attention.head_count", 4),
                ("attention.head_count_kv", 1),
                ("attention.key_length", 24),
                ("attention.value_length", 16),
                ("attention.q_lora_rank", 32),
                ("attention.kv_lora_rank", 16),
                ("attention.key_length_mla", 16),
                ("attention.value_length_mla", 16),
                ("rope.dimension_count", 8),
                ("rope.scaling.original_context_length", 8),
                ("attention.temperature_length", 8),
                ("leading_dense_block_count", 2),
            ] {
                assert_eq!(uint(key), want, "{key}");
            }
            for (key, want) in [
                ("rope.freq_base", 10_000.0),
                ("rope.scaling.factor", 2.0),
                ("rope.scaling.yarn_beta_fast", 32.0),
                ("rope.scaling.yarn_beta_slow", 1.0),
                ("rope.scaling.yarn_log_multiplier", 0.1),
                ("attention.temperature_scale", 0.3),
                ("attention.layer_norm_rms_epsilon", 1e-6),
            ] {
                assert_eq!(float(key), want, "{key}");
            }
            assert!(
                md.keys().all(|k| !k.starts_with("tokenizer.")),
                "no tokenizer"
            );

            let narrow = if quant == WeightQuant::F16 {
                GgmlDType::F16
            } else {
                GgmlDType::F32
            };
            let mut want: Vec<(String, Vec<usize>, GgmlDType)> = vec![
                ("nextn.eh_proj.weight".into(), vec![64, 128], matrix),
                ("output_norm.weight".into(), vec![64], GgmlDType::F32),
            ];
            for il in 0..2 {
                let b = |part: &str| format!("blk.{il}.{part}.weight");
                want.extend([
                    (b("attn_norm"), vec![64], GgmlDType::F32),
                    (b("attn_q_a"), vec![32, 64], matrix),
                    (b("attn_q_a_norm"), vec![32], GgmlDType::F32),
                    (b("attn_q_b"), vec![64, 32], matrix),
                    (b("attn_kv_a_mqa"), vec![24, 64], matrix),
                    (b("attn_kv_a_norm"), vec![16], GgmlDType::F32),
                    (b("attn_k_b"), vec![4, 16, 8], narrow),
                    (b("attn_v_b"), vec![4, 16, 16], narrow),
                    (b("attn_output"), vec![64, 64], matrix),
                    (b("ffn_norm"), vec![64], GgmlDType::F32),
                    (b("ffn_gate"), vec![96, 64], matrix),
                    (b("ffn_up"), vec![96, 64], matrix),
                    (b("ffn_down"), vec![64, 96], matrix),
                ]);
            }
            let mut got: Vec<(String, Vec<usize>, GgmlDType)> = content
                .tensor_infos
                .iter()
                .map(|(n, i)| (n.clone(), i.shape.dims().to_vec(), i.ggml_dtype))
                .collect();
            got.sort_by(|a, b| a.0.cmp(&b.0));
            want.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(got, want, "{quant:?}");
            let _ = std::fs::remove_file(&out);
            let _ = std::fs::remove_file(out.with_extension("gguf.lock"));
        }
    }

    /// The values: every matrix is the FP8 value times its `qscale_weight`
    /// (rounded to f16 here), the fusion projection is stored as it is, and
    /// `wkv_b` splits into the per-head `k_b` (each head transposed) and
    /// `v_b`.
    #[test]
    fn the_fixture_values_are_the_fp8_values_times_their_scales() {
        let (content, out) = convert_fixture("values", WeightQuant::F16);
        let shards = Shards::open(&[fixture_dir().join(WEIGHTS_FILE)]).unwrap();
        let mut file = std::fs::File::open(&out).unwrap();
        let mut tensor = |name: &str| -> Vec<f32> {
            content
                .tensor(&mut file, name, &Device::Cpu)
                .unwrap()
                .dequantize(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let f16 = |v: &[f32]| -> Vec<f32> {
            v.iter().map(|x| half::f16::from_f32(*x).to_f32()).collect()
        };
        let fc = shards.f32("eagle_linear.weight").unwrap();
        assert!(fc.iter().any(|&v| v != 0.0));
        assert_eq!(tensor("nextn.eh_proj.weight"), f16(&fc));
        let down = shards.f32("layers.1.feed_forward.w2.weight").unwrap();
        assert_eq!(tensor("blk.1.ffn_down.weight"), f16(&down));
        let up = shards.f32("layers.0.feed_forward.w3.weight").unwrap();
        assert_eq!(tensor("blk.0.ffn_up.weight"), f16(&up));
        // Norms are BF16 in the checkpoint and exact in f32.
        assert_eq!(
            tensor("output_norm.weight"),
            shards.f32("norm.weight").unwrap()
        );
        // `kv_b_proj` rows are `[heads][nope + v][kv_lora]`: `k_b[h][c][n]`
        // is `wkv_b[h * (nope + v) + n][c]`, `v_b[h][j][c]` the row `nope + j`.
        let kv_b = shards.f32("layers.0.attention.wkv_b.weight").unwrap();
        let (k_b, v_b) = (
            tensor("blk.0.attn_k_b.weight"),
            tensor("blk.0.attn_v_b.weight"),
        );
        let (heads, nope, v, lora) = (4, 8, 16, 16);
        for h in 0..heads {
            for c in 0..lora {
                for n in 0..nope {
                    assert_eq!(
                        k_b[(h * lora + c) * nope + n],
                        kv_b[(h * (nope + v) + n) * lora + c]
                    );
                }
                for j in 0..v {
                    assert_eq!(
                        v_b[(h * v + j) * lora + c],
                        kv_b[(h * (nope + v) + nope + j) * lora + c]
                    );
                }
            }
        }
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(out.with_extension("gguf.lock"));
    }

    /// A checkpoint tensor the drafter has no GGUF name for is an error
    /// naming it, and a missing weight scale is one naming the companion.
    #[test]
    fn an_unmapped_tensor_or_a_missing_scale_fails_the_conversion() {
        use crate::provider::local::convert::safetensors::tests::{temp_dir, write_file};
        let dir = temp_dir("eagle-bad");
        std::fs::copy(fixture_dir().join(PARAMS_FILE), dir.join(PARAMS_FILE)).unwrap();
        let bf = |v: f32| half::bf16::from_f32(v).to_le_bytes().to_vec();
        write_file(
            &dir.join(WEIGHTS_FILE),
            &[
                ("norm.weight", "BF16", &[2], [bf(1.0), bf(1.0)].concat()),
                ("layers.0.attention.rotary.weight", "BF16", &[1], bf(1.0)),
            ],
        );
        let ck = CheckpointDir::open(&dir).unwrap();
        let out = dir.join("out.gguf");
        let e = super::super::convert(&ck, &out, WeightQuant::F16, None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("layers.0.attention.rotary.weight"), "{e}");

        write_file(
            &dir.join(WEIGHTS_FILE),
            &[("eagle_linear.weight", "F8_E4M3", &[1, 1], vec![0x38])],
        );
        let ck = CheckpointDir::open(&dir).unwrap();
        let e = super::super::convert(&ck, &out, WeightQuant::F16, None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("eagle_linear.weight_scale_inv"), "{e}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
