//! A tiny Llama checkpoint converted at each K-quant mix, end to end: the
//! tensor types the GGUF records follow llama.cpp's table, the bytes are the
//! reference quantizer's, and the engine loads the file and runs it.
//!
//! The checkpoint is 8 layers of width 256 with an untied head; at FFN width 256
//! every row is a multiple of 256, so the K-quants apply without a fallback.

use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::GgmlDType;
use serde_json::json;

use super::hf::CheckpointDir;
use super::refquant::{Kind, quantize_row};
use super::safetensors::tests::{f32_bytes, temp_dir, write_file};
use crate::provider::local::gguf;
use crate::provider::local::model::Want;
use crate::provider::local::model::cpu::Transformer;
use crate::provider::local::model::paged::Sizing;
use crate::provider::local::model::tiny::Lcg;
use crate::provider::local::{EngramCache, ExpertPlacement, WeightQuant};

const HIDDEN: usize = 256;
const LAYERS: usize = 8;
const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
/// 256 byte tokens, one merge (`he`), `<|bos|>` and `<|eos|>`.
const VOCAB: usize = 259;

/// GPT-2's `bytes_to_unicode`: the character that spells each byte.
fn byte_chars() -> Vec<char> {
    let printable: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut next = 0;
    (0..256u32)
        .map(|b| {
            let code = if printable.contains(&b) {
                b
            } else {
                next += 1;
                255 + next
            };
            char::from_u32(code).expect("a scalar value")
        })
        .collect()
}

/// A checkpoint directory and the values of every weight, by checkpoint name.
struct Fixture {
    dir: std::path::PathBuf,
    weights: HashMap<String, Vec<f32>>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The checkpoint with FFN width `ff`.
fn fixture(ff: usize) -> Fixture {
    let dir = temp_dir("quant-e2e");
    let mut rng = Lcg(0x5EED);
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![VOCAB, HIDDEN]),
        ("lm_head.weight".into(), vec![VOCAB, HIDDEN]),
        ("model.norm.weight".into(), vec![HIDDEN]),
    ];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}.");
        tensors.extend([
            (format!("{p}input_layernorm.weight"), vec![HIDDEN]),
            (format!("{p}post_attention_layernorm.weight"), vec![HIDDEN]),
            (
                format!("{p}self_attn.q_proj.weight"),
                vec![HEADS * HEAD_DIM, HIDDEN],
            ),
            (
                format!("{p}self_attn.k_proj.weight"),
                vec![KV_HEADS * HEAD_DIM, HIDDEN],
            ),
            (
                format!("{p}self_attn.v_proj.weight"),
                vec![KV_HEADS * HEAD_DIM, HIDDEN],
            ),
            (
                format!("{p}self_attn.o_proj.weight"),
                vec![HIDDEN, HEADS * HEAD_DIM],
            ),
            (format!("{p}mlp.gate_proj.weight"), vec![ff, HIDDEN]),
            (format!("{p}mlp.up_proj.weight"), vec![ff, HIDDEN]),
            (format!("{p}mlp.down_proj.weight"), vec![HIDDEN, ff]),
        ]);
    }
    let mut weights = HashMap::new();
    let mut files = Vec::new();
    for (name, shape) in &tensors {
        let n: usize = shape.iter().product();
        let values = if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
            rng.values(n, 0.1).iter().map(|v| 1.0 + v).collect()
        } else {
            rng.values(n, 0.15)
        };
        files.push((name.as_str(), "F32", shape.as_slice(), f32_bytes(&values)));
        weights.insert(name.clone(), values);
    }
    write_file(&dir.join("model.safetensors"), &files);

    std::fs::write(
        dir.join("config.json"),
        json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": HIDDEN, "intermediate_size": ff, "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
            "vocab_size": VOCAB, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "max_position_embeddings": 512, "tie_word_embeddings": false,
            "bos_token_id": 257, "eos_token_id": 258
        })
        .to_string(),
    )
    .unwrap();
    let chars = byte_chars();
    let mut vocab: serde_json::Map<String, serde_json::Value> = chars
        .iter()
        .enumerate()
        .map(|(id, c)| (c.to_string(), json!(id)))
        .collect();
    vocab.insert("he".into(), json!(256));
    std::fs::write(
        dir.join("tokenizer.json"),
        json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": ["h e"]},
            "added_tokens": [
                {"id": 257, "content": "<|bos|>", "special": true},
                {"id": 258, "content": "<|eos|>", "special": true}
            ],
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
            "decoder": {"type": "ByteLevel"}
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        json!({
            "bos_token": "<|bos|>", "eos_token": "<|eos|>",
            "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"
        })
        .to_string(),
    )
    .unwrap();
    Fixture { dir, weights }
}

fn convert_at(fx: &Fixture, quant: WeightQuant) -> (gguf::Content, std::path::PathBuf) {
    let ck = CheckpointDir::open(&fx.dir).unwrap();
    let out = fx.dir.join(format!("model-{}.gguf", quant.name()));
    super::convert(&ck, &out, quant, None).unwrap();
    (gguf::Content::open(&out).unwrap(), out)
}

/// The type the table gives each tensor of the fixture: `promoted` layers hold
/// the attention values and the FFN down projections at `high`, the others at
/// `base`; `head` is the output head.
struct Expect {
    base: GgmlDType,
    high: GgmlDType,
    promoted: &'static [usize],
    head: GgmlDType,
    embd: GgmlDType,
}

fn check_types(content: &gguf::Content, w: &Expect, what: &str) {
    let ty = |name: &str| content.ggml_type(name).unwrap();
    assert_eq!(ty("output.weight"), w.head, "{what} output");
    assert_eq!(ty("token_embd.weight"), w.embd, "{what} token_embd");
    assert_eq!(
        ty("output_norm.weight"),
        GgmlDType::F32,
        "{what} output_norm"
    );
    for l in 0..LAYERS {
        let pick = if w.promoted.contains(&l) {
            w.high
        } else {
            w.base
        };
        for (part, want) in [
            ("attn_v", pick),
            ("ffn_down", pick),
            ("attn_q", w.base),
            ("attn_k", w.base),
            ("attn_output", w.base),
            ("ffn_gate", w.base),
            ("ffn_up", w.base),
            ("attn_norm", GgmlDType::F32),
            ("ffn_norm", GgmlDType::F32),
        ] {
            let name = format!("blk.{l}.{part}.weight");
            assert_eq!(ty(&name), want, "{what} {name}");
        }
    }
}

/// The engine loads the converted file and returns finite, varying logits.
fn run(path: &Path) -> Vec<f32> {
    let content = gguf::Content::open(path).unwrap();
    let mut model = Transformer::load_placed(
        &content,
        64,
        &ExpertPlacement::Resident,
        &Sizing::single(64),
        &EngramCache::default(),
    )
    .unwrap();
    let logits = model
        .forward(&[257, 40, 80, 120, 7], 0, Want::Last)
        .unwrap()
        .unwrap();
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
    let (lo, hi) = logits
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    assert!(hi - lo > 1e-4, "logits vary across the vocabulary");
    logits
}

/// The cosine similarity of two logit vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

/// The stored bytes of `name` against the reference quantizer over the source
/// values (no permutation applies: the tensor is not a Q or K projection).
fn check_bytes(fx: &Fixture, content: &gguf::Content, hf: &str, gguf_name: &str, kind: Kind) {
    let values = &fx.weights[hf];
    let block = if matches!(kind, Kind::Q5_0 | Kind::Q5_1) {
        32
    } else {
        256
    };
    let bytes = match kind {
        Kind::Q5_0 => 22,
        Kind::Q5_1 => 24,
        Kind::Q4K => 144,
        Kind::Q5K => 176,
        Kind::Q6K => 210,
    };
    let mut want = vec![0u8; values.len() / block * bytes];
    quantize_row(kind, values, &mut want);
    let got = content.reader().bytes(gguf_name).unwrap();
    assert!(
        got == want,
        "{gguf_name} differs from the reference quantizer"
    );
}

#[test]
fn q4_k_m_writes_llama_cpps_mix_and_the_engine_runs_it() {
    let fx = fixture(256);
    let (content, out) = convert_at(&fx, WeightQuant::Q4KM);
    // 8 layers: use_more_bits is true for layers 0, 3, 6 and 7.
    check_types(
        &content,
        &Expect {
            base: GgmlDType::Q4K,
            high: GgmlDType::Q6K,
            promoted: &[0, 3, 6, 7],
            head: GgmlDType::Q6K,
            embd: GgmlDType::Q4K,
        },
        "q4_k_m",
    );
    assert_eq!(
        content.metadata["general.file_type"].to_u32().unwrap(),
        15,
        "LLAMA_FTYPE_MOSTLY_Q4_K_M"
    );
    check_bytes(
        &fx,
        &content,
        "model.layers.1.mlp.up_proj.weight",
        "blk.1.ffn_up.weight",
        Kind::Q4K,
    );
    check_bytes(
        &fx,
        &content,
        "model.layers.3.mlp.down_proj.weight",
        "blk.3.ffn_down.weight",
        Kind::Q6K,
    );
    check_bytes(
        &fx,
        &content,
        "model.layers.0.self_attn.v_proj.weight",
        "blk.0.attn_v.weight",
        Kind::Q6K,
    );
    check_bytes(&fx, &content, "lm_head.weight", "output.weight", Kind::Q6K);
    run(&out);
}

#[test]
fn q5_k_m_and_q4_k_s_and_q6_k_write_their_mixes_and_run() {
    let fx = fixture(256);
    let (content, out) = convert_at(&fx, WeightQuant::Q5KM);
    check_types(
        &content,
        &Expect {
            base: GgmlDType::Q5K,
            high: GgmlDType::Q6K,
            promoted: &[0, 3, 6, 7],
            head: GgmlDType::Q6K,
            embd: GgmlDType::Q5K,
        },
        "q5_k_m",
    );
    assert_eq!(content.metadata["general.file_type"].to_u32().unwrap(), 17);
    check_bytes(
        &fx,
        &content,
        "model.layers.2.mlp.gate_proj.weight",
        "blk.2.ffn_gate.weight",
        Kind::Q5K,
    );
    run(&out);

    // Q4_K_S: the first four attention values and the first eighth (one
    // layer) of the FFN down projections are Q5_K.
    let (content, out) = convert_at(&fx, WeightQuant::Q4KS);
    let ty = |name: &str| content.ggml_type(name).unwrap();
    for l in 0..LAYERS {
        let v = if l < 4 {
            GgmlDType::Q5K
        } else {
            GgmlDType::Q4K
        };
        let down = if l < 1 {
            GgmlDType::Q5K
        } else {
            GgmlDType::Q4K
        };
        assert_eq!(ty(&format!("blk.{l}.attn_v.weight")), v, "attn_v {l}");
        assert_eq!(
            ty(&format!("blk.{l}.ffn_down.weight")),
            down,
            "ffn_down {l}"
        );
        assert_eq!(ty(&format!("blk.{l}.ffn_up.weight")), GgmlDType::Q4K);
    }
    assert_eq!(ty("output.weight"), GgmlDType::Q6K);
    assert_eq!(content.metadata["general.file_type"].to_u32().unwrap(), 14);
    run(&out);

    // Q6_K: every matrix is Q6_K, and its logits track the f16 conversion's.
    let (content, out) = convert_at(&fx, WeightQuant::Q6K);
    check_types(
        &content,
        &Expect {
            base: GgmlDType::Q6K,
            high: GgmlDType::Q6K,
            promoted: &[],
            head: GgmlDType::Q6K,
            embd: GgmlDType::Q6K,
        },
        "q6_k",
    );
    assert_eq!(content.metadata["general.file_type"].to_u32().unwrap(), 18);
    let q6 = run(&out);
    let (_, f16_path) = convert_at(&fx, WeightQuant::F16);
    let f16 = run(&f16_path);
    // Random weights amplify rounding through 8 layers, so the comparison is
    // the direction of the logits: Q8_0 measures 0.998 against f16, Q6_K 0.987.
    let c = cosine(&q6, &f16);
    assert!(c > 0.97, "Q6_K logits point where f16's do: cosine {c}");
}

#[test]
fn each_quant_is_cached_under_its_own_name() {
    let names: Vec<_> = [
        WeightQuant::F16,
        WeightQuant::Q8_0,
        WeightQuant::Q4_0,
        WeightQuant::Q4KS,
        WeightQuant::Q4KM,
        WeightQuant::Q5KM,
        WeightQuant::Q6K,
    ]
    .into_iter()
    .map(|quant| {
        let source = crate::provider::local::ModelSource::Checkpoint {
            repo: "o/m".into(),
            revision: "r".into(),
        };
        let config = crate::provider::local::ConvertConfig {
            quant,
            ..Default::default()
        };
        super::output_path(Path::new("cache"), &source, config)
            .unwrap()
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_owned)
            .unwrap()
    })
    .collect();
    let unique: std::collections::HashSet<_> = names.iter().collect();
    assert_eq!(unique.len(), names.len(), "{names:?}");
    assert_eq!(names[4], "model-q4_k_m.gguf");
}

#[test]
fn rows_that_do_not_fill_256_fall_back_in_the_file_and_still_run() {
    // FFN width 96: `ffn_down` rows are 96 long, a multiple of 32 only.
    let fx = fixture(96);
    let (content, out) = convert_at(&fx, WeightQuant::Q4KM);
    let ty = |name: &str| content.ggml_type(name).unwrap();
    for l in 0..LAYERS {
        // Q4_K falls back to Q5_0, the promoted Q6_K to Q8_0.
        let down = if [0, 3, 6, 7].contains(&l) {
            GgmlDType::Q8_0
        } else {
            GgmlDType::Q5_0
        };
        assert_eq!(
            ty(&format!("blk.{l}.ffn_down.weight")),
            down,
            "ffn_down {l}"
        );
        // Their input rows are 256 long: the rest of the layer is unaffected.
        assert_eq!(ty(&format!("blk.{l}.ffn_up.weight")), GgmlDType::Q4K);
    }
    check_bytes(
        &fx,
        &content,
        "model.layers.1.mlp.down_proj.weight",
        "blk.1.ffn_down.weight",
        Kind::Q5_0,
    );
    run(&out);
    let (content, out) = convert_at(&fx, WeightQuant::Q5KM);
    assert_eq!(
        content.ggml_type("blk.1.ffn_down.weight").unwrap(),
        GgmlDType::Q5_1
    );
    check_bytes(
        &fx,
        &content,
        "model.layers.1.mlp.down_proj.weight",
        "blk.1.ffn_down.weight",
        Kind::Q5_1,
    );
    run(&out);

    // Width 80 is not even a multiple of 32: half precision.
    let fx = fixture(80);
    let (content, out) = convert_at(&fx, WeightQuant::Q4KM);
    assert_eq!(
        content.ggml_type("blk.1.ffn_down.weight").unwrap(),
        GgmlDType::F16
    );
    assert_eq!(
        content.ggml_type("blk.1.ffn_up.weight").unwrap(),
        GgmlDType::Q4K
    );
    run(&out);
}
