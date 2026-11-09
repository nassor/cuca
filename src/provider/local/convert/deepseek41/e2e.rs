//! The converter end to end on a micro DeepSeek-V4.1 checkpoint written in the
//! release format (FP8 e4m3 weights on 32x32 `ue8m0` grids, packed FP4 experts,
//! Engram tables, bf16 and f32 small tensors, vision tensors the converter must
//! skip, one DSpark stage): the conversion at `quant = f16` with a bf16 Engram
//! table must give back, tensor by tensor, the exact float values the checkpoint
//! defines, and the metadata the format lists. The values are chosen so that
//! every dequantized number is exact in half precision.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::json;

use super::super::engram;
use super::super::hf::CheckpointDir;
use super::super::safetensors::tests::{temp_dir, write_file};
use super::super::safetensors::{E4M3, e8m0};
use super::plan::Role;
use crate::provider::local::convert::{convert_set, converted};
use crate::provider::local::gguf::{Content, GgufType, Value};
use crate::provider::local::model::quant::dequant_row;
use crate::provider::local::tokenizer::byte_to_char;
use crate::provider::local::{ConvertConfig, EngramQuant, WeightQuant};

const HIDDEN: usize = 64;
const HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const Q_LORA: usize = 32;
const O_GROUPS: usize = 2;
const O_LORA: usize = 32;
const MOE_FF: usize = 32;
const EXPERTS: usize = 3;
const HC: usize = 2;
const INDEX_HEADS: usize = 2;
const INDEX_DIM: usize = 32;
const ENGRAM_DIM: usize = 32;
const NGRAM: usize = 3;
const ENGRAM_HEADS: usize = 2;
const ENGRAM_VOCAB: u64 = 20;
const VOCAB: usize = 262;
const BOS: usize = 258;
const EOS: usize = 259;
const MARKOV: usize = 16;
const DRAFT_EXPERTS: usize = 2;

type Tensor = (String, &'static str, Vec<usize>, Vec<u8>);

/// Deterministic pseudo-random bytes.
fn bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

/// The checkpoint under construction and the float values it defines, by GGUF
/// name.
#[derive(Default)]
struct Micro {
    tensors: Vec<Tensor>,
    want: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    seed: u64,
}

impl Micro {
    fn next_seed(&mut self) -> u64 {
        self.seed += 1;
        self.seed
    }

    fn add(&mut self, name: &str, dtype: &'static str, shape: &[usize], data: Vec<u8>) {
        self.tensors
            .push((name.into(), dtype, shape.to_vec(), data));
    }

    fn expect(&mut self, gguf: &str, shape: &[usize], values: Vec<f32>) {
        self.want.insert(gguf.into(), (shape.to_vec(), values));
    }

    /// An FP8 matrix with its 32x32 scale grid; the expected GGUF tensor is the
    /// dequantized matrix.
    fn fp8(&mut self, module: &str, gguf: &str, rows: usize, cols: usize) {
        let seed = self.next_seed();
        let weight: Vec<u8> = bytes(rows * cols, seed)
            .into_iter()
            .map(|b| if b & 0x7F == 0x7F { b & 0x80 | 0x38 } else { b })
            .collect();
        let grid_cols = cols.div_ceil(32);
        let grid: Vec<u8> = bytes(rows.div_ceil(32) * grid_cols, seed + 1000)
            .iter()
            .map(|b| 124 + b % 7)
            .collect();
        let values = (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                let e = grid[(r / 32) * grid_cols + c / 32];
                (f64::from(E4M3[usize::from(weight[i])]) * 2f64.powi(i32::from(e) - 127)) as f32
            })
            .collect();
        self.add(
            &format!("{module}.weight"),
            "F8_E4M3",
            &[rows, cols],
            weight,
        );
        self.add(
            &format!("{module}.scale"),
            "F8_E8M0",
            &[rows.div_ceil(32), grid_cols],
            grid,
        );
        self.expect(gguf, &[rows, cols], values);
    }

    /// An Engram table: e4m3 rows with one scale per row and 32 columns.
    fn engram_table(&mut self, module: &str, gguf: &str, rows: usize, dim: usize) {
        let seed = self.next_seed();
        let weight: Vec<u8> = bytes(rows * dim, seed)
            .into_iter()
            .map(|b| if b & 0x7F == 0x7F { b & 0x80 | 0x38 } else { b })
            .collect();
        let groups = dim / 32;
        let scale: Vec<u8> = bytes(rows * groups, seed + 2000)
            .iter()
            .map(|b| 124 + b % 7)
            .collect();
        let values = (0..rows * dim)
            .map(|i| {
                let e = scale[i / dim * groups + i % dim / 32];
                (f64::from(E4M3[usize::from(weight[i])]) * 2f64.powi(i32::from(e) - 127)) as f32
            })
            .collect();
        self.add(&format!("{module}.weight"), "F8_E4M3", &[rows, dim], weight);
        self.add(
            &format!("{module}.scale"),
            "F8_E8M0",
            &[rows, groups],
            scale,
        );
        self.expect(gguf, &[rows, dim], values);
    }

    /// Values exact in bf16 and in half precision: multiples of 1/16 in [-0.5, 0.4375].
    fn small_values(&mut self, n: usize) -> Vec<f32> {
        let seed = self.next_seed();
        bytes(n, seed)
            .into_iter()
            .map(|b| f32::from(b % 16) / 16.0 - 0.5)
            .collect()
    }

    /// A bf16 tensor; the expected GGUF tensor holds its values.
    fn bf16(&mut self, name: &str, gguf: Option<&str>, shape: &[usize]) {
        let values = self.small_values(shape.iter().product());
        let data = values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        self.add(name, "BF16", shape, data);
        if let Some(gguf) = gguf {
            self.expect(gguf, shape, values);
        }
    }

    /// An f32 tensor; the expected GGUF tensor holds its values.
    fn f32(&mut self, name: &str, gguf: &str, shape: &[usize]) {
        let values = self.small_values(shape.iter().product());
        let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add(name, "F32", shape, data);
        self.expect(gguf, shape, values);
    }

    /// `n` experts' `proj` (`[rows, cols]` FP4 weights, one scale per 32 columns);
    /// the expected GGUF tensor is the stack of their dequantized values.
    fn experts(
        &mut self,
        prefix: &str,
        proj: &str,
        gguf: &str,
        n: usize,
        rows: usize,
        cols: usize,
    ) {
        const FP4: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut stack = Vec::new();
        for e in 0..n {
            let seed = self.next_seed();
            let weight = bytes(rows * cols / 2, seed);
            let scale: Vec<u8> = bytes(rows * cols / 32, seed + 5000)
                .iter()
                .map(|b| 123 + b % 9)
                .collect();
            for r in 0..rows {
                for c in 0..cols {
                    let byte = weight[r * cols / 2 + c / 2];
                    let code = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    let s = e8m0(scale[r * cols / 32 + c / 32]);
                    stack.push(FP4[usize::from(code)] * s);
                }
            }
            let base = format!("{prefix}.ffn.experts.{e}.{proj}");
            self.add(&format!("{base}.weight"), "I8", &[rows, cols / 2], weight);
            self.add(
                &format!("{base}.scale"),
                "F8_E8M0",
                &[rows, cols / 32],
                scale,
            );
        }
        self.expect(gguf, &[n, rows, cols], stack);
    }

    /// A transformer block of the checkpoint (`prefix` is `layers.N` or
    /// `mtp.N`) and of the GGUF (`blk.M`).
    fn block(&mut self, prefix: &str, blk: &str, experts: usize, compressed: bool) {
        self.q_norm(prefix, blk, "attn_norm.weight", "attn_norm.weight", HIDDEN);
        self.q_norm(prefix, blk, "ffn_norm.weight", "ffn_norm.weight", HIDDEN);
        self.f32(
            &format!("{prefix}.attn.attn_sink"),
            &format!("{blk}.attn_sinks.weight"),
            &[HEADS],
        );
        self.fp8(
            &format!("{prefix}.attn.wq_a"),
            &format!("{blk}.attn_q_a.weight"),
            Q_LORA,
            HIDDEN,
        );
        self.q_norm(
            prefix,
            blk,
            "attn.q_norm.weight",
            "attn_q_a_norm.weight",
            Q_LORA,
        );
        self.fp8(
            &format!("{prefix}.attn.wq_b"),
            &format!("{blk}.attn_q_b.weight"),
            HEADS * HEAD_DIM,
            Q_LORA,
        );
        self.fp8(
            &format!("{prefix}.attn.wkv"),
            &format!("{blk}.attn_kv.weight"),
            HEAD_DIM,
            HIDDEN,
        );
        self.q_norm(
            prefix,
            blk,
            "attn.kv_norm.weight",
            "attn_kv_a_norm.weight",
            HEAD_DIM,
        );
        self.fp8(
            &format!("{prefix}.attn.wo_a"),
            &format!("{blk}.attn_output_a.weight"),
            O_GROUPS * O_LORA,
            HEADS * HEAD_DIM / O_GROUPS,
        );
        self.fp8(
            &format!("{prefix}.attn.wo_b"),
            &format!("{blk}.attn_output_b.weight"),
            HIDDEN,
            O_GROUPS * O_LORA,
        );
        for site in ["attn", "ffn"] {
            self.f32(
                &format!("{prefix}.hc_{site}_fn"),
                &format!("{blk}.hc_{site}_fn.weight"),
                &[(2 + HC) * HC, HC * HIDDEN],
            );
            self.f32(
                &format!("{prefix}.hc_{site}_base"),
                &format!("{blk}.hc_{site}_base.weight"),
                &[(2 + HC) * HC],
            );
            self.f32(
                &format!("{prefix}.hc_{site}_scale"),
                &format!("{blk}.hc_{site}_scale.weight"),
                &[3],
            );
        }
        if compressed {
            for (name, gguf) in [
                ("wkv", "attn_compressor_kv"),
                ("wgate", "attn_compressor_gate"),
            ] {
                self.bf16(
                    &format!("{prefix}.attn.compressor.{name}.weight"),
                    Some(&format!("{blk}.{gguf}.weight")),
                    &[HEAD_DIM, HIDDEN],
                );
            }
            self.q_norm(
                prefix,
                blk,
                "attn.compressor.norm.weight",
                "attn_compressor_norm.weight",
                HEAD_DIM,
            );
            self.bf16(
                &format!("{prefix}.attn.indexer.wk.weight"),
                Some(&format!("{blk}.indexer.attn_k.weight")),
                &[INDEX_DIM, HEAD_DIM],
            );
            self.q_norm(
                prefix,
                blk,
                "attn.indexer.k_norm.weight",
                "indexer.k_norm.weight",
                INDEX_DIM,
            );
            self.fp8(
                &format!("{prefix}.attn.indexer.wq_b"),
                &format!("{blk}.indexer.attn_q_b.weight"),
                INDEX_HEADS * INDEX_DIM,
                Q_LORA,
            );
            self.bf16(
                &format!("{prefix}.attn.indexer.weights_proj.weight"),
                Some(&format!("{blk}.indexer.proj.weight")),
                &[INDEX_HEADS, HIDDEN],
            );
        }
        let gate = format!("{prefix}.ffn.gate.weight");
        self.bf16(&gate, None, &[experts, HIDDEN]);
        let t = self.want_last_f32(&gate);
        self.expect(&format!("{blk}.ffn_gate_inp.weight"), &[experts, HIDDEN], t);
        self.f32(
            &format!("{prefix}.ffn.gate.bias"),
            &format!("{blk}.exp_probs_b.bias"),
            &[experts],
        );
        // The image-token routing bias is not converted.
        let skipped = self.small_values(experts);
        self.add(
            &format!("{prefix}.ffn.gate.bias_vl"),
            "F32",
            &[experts],
            skipped.iter().flat_map(|v| v.to_le_bytes()).collect(),
        );
        for (proj, gguf, rows, cols) in [
            ("w1", "ffn_gate_shexp", MOE_FF, HIDDEN),
            ("w3", "ffn_up_shexp", MOE_FF, HIDDEN),
            ("w2", "ffn_down_shexp", HIDDEN, MOE_FF),
        ] {
            self.fp8(
                &format!("{prefix}.ffn.shared_experts.{proj}"),
                &format!("{blk}.{gguf}.weight"),
                rows,
                cols,
            );
        }
        for (proj, gguf, rows, cols) in [
            ("w1", "ffn_gate_exps", MOE_FF, HIDDEN),
            ("w3", "ffn_up_exps", MOE_FF, HIDDEN),
            ("w2", "ffn_down_exps", HIDDEN, MOE_FF),
        ] {
            self.experts(
                prefix,
                proj,
                &format!("{blk}.{gguf}.weight"),
                experts,
                rows,
                cols,
            );
        }
    }

    /// The f32 values of the last tensor added under `name` (bf16 or f32).
    fn want_last_f32(&self, name: &str) -> Vec<f32> {
        let (_, dtype, _, data) = self
            .tensors
            .iter()
            .rev()
            .find(|t| t.0 == name)
            .expect("added");
        match *dtype {
            "BF16" => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::bf16::from_le_bytes(*b).to_f32())
                .collect(),
            _ => data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        }
    }

    /// A bf16 vector the GGUF holds as `f32`.
    fn norm(&mut self, checkpoint: &str, gguf: &str, n: usize) {
        self.bf16(checkpoint, None, &[n]);
        let values = self.want_last_f32(checkpoint);
        self.expect(gguf, &[n], values);
    }

    fn q_norm(&mut self, prefix: &str, blk: &str, name: &str, gguf: &str, n: usize) {
        self.norm(&format!("{prefix}.{name}"), &format!("{blk}.{gguf}"), n);
    }
}

/// The tokenizer files: byte-level BPE over the 256 byte characters, two merged
/// words and four added tokens, with DeepSeek's three-`Split` pre-tokenizer.
fn write_tokenizer(dir: &Path) {
    let splits = [
        r"\p{N}{1,3}",
        "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
    ];
    let mut pre: Vec<serde_json::Value> = splits
        .iter()
        .map(|r| json!({"type": "Split", "pattern": {"Regex": r}, "behavior": "Isolated", "invert": false}))
        .collect();
    pre.push(json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}));
    let mut vocab = serde_json::Map::new();
    for b in 0..=255u8 {
        vocab.insert(byte_to_char(b).to_string(), json!(b));
    }
    vocab.insert("he".into(), json!(256));
    vocab.insert("hel".into(), json!(257));
    let added = [
        (
            BOS,
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>",
            true,
        ),
        (EOS, "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>", true),
        (260, "<\u{ff5c}\u{2581}pad\u{2581}\u{ff5c}>", true),
        (261, "<think>", false),
    ];
    let added_json: Vec<_> = added
        .iter()
        .map(|(id, content, special)| {
            json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false, "normalized": !special, "special": special})
        })
        .collect();
    let tokenizer = json!({
        "version": "1.0",
        "added_tokens": added_json,
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": pre},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        "model": {"type": "BPE", "vocab": vocab, "merges": ["h e", "he l"]},
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    let decoder: serde_json::Map<String, serde_json::Value> = added
        .iter()
        .map(|(id, content, special)| {
            (
                id.to_string(),
                json!({"content": content, "special": special, "normalized": !special}),
            )
        })
        .collect();
    let config = json!({
        "add_bos_token": false,
        "add_eos_token": false,
        "bos_token": added[0].1,
        "eos_token": added[1].1,
        "pad_token": added[1].1,
        "clean_up_tokenization_spaces": false,
        "added_tokens_decoder": decoder,
    });
    std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
}

/// What the micro checkpoint's metadata says about the Engram tables.
struct EngramSizes {
    rows: u64,
    classes: usize,
}

fn engram_sizes(dir: &Path) -> EngramSizes {
    let primes = engram::primes(1, NGRAM, ENGRAM_HEADS, ENGRAM_VOCAB);
    let tokenizer =
        serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json")).unwrap())
            .unwrap();
    let (_, classes) = engram::compressed_token_map(&tokenizer, VOCAB).unwrap();
    EngramSizes {
        rows: primes.iter().flatten().flatten().sum(),
        classes,
    }
}

/// A micro checkpoint on disk and the values it defines.
struct Fixture {
    dir: PathBuf,
    /// The backbone's GGUF tensors.
    main: Micro,
    /// The DSpark stage's GGUF tensors.
    drafter: Micro,
    sizes: EngramSizes,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Write the micro checkpoint into a fresh directory: layer 0 (no compressor,
/// the Engram) and layer 1 (compressor and indexer sources), the roots, and one
/// DSpark stage; the vision tensors, the aligner and the image tokens ride
/// along. `without` removes the checkpoint tensors it names (for the error
/// tests).
fn checkpoint(with_engram: bool, without: &[&str]) -> Fixture {
    let dir = temp_dir("ds41-micro");
    write_tokenizer(&dir);
    let sizes = engram_sizes(&dir);
    let mut main = Micro::default();
    main.bf16("embed.weight", Some("token_embd.weight"), &[VOCAB, HIDDEN]);
    main.bf16("head.weight", Some("output.weight"), &[VOCAB, HIDDEN]);
    main.norm("norm.weight", "output_norm.weight", HIDDEN);
    for il in 0..2 {
        main.block(
            &format!("layers.{il}"),
            &format!("blk.{il}"),
            EXPERTS,
            il == 1,
        );
    }
    if with_engram {
        main.fp8(
            "layers.0.engram.wkv",
            "blk.0.engram_kv.weight",
            (HC + 1) * HIDDEN,
            (NGRAM - 1) * ENGRAM_HEADS * ENGRAM_DIM,
        );
        main.bf16("layers.0.engram.q_weight", None, &[HC, HIDDEN]);
        main.bf16("layers.0.engram.k_weight", None, &[HC, HIDDEN]);
        let (q, k) = (
            main.want_last_f32("layers.0.engram.q_weight"),
            main.want_last_f32("layers.0.engram.k_weight"),
        );
        let gate = q.iter().zip(&k).map(|(a, b)| a * b).collect();
        main.expect("blk.0.engram_gate.weight", &[HC, HIDDEN], gate);
        main.engram_table(
            "layers.0.engram.embed",
            "per_layer_token_embd.weight",
            sizes.rows as usize,
            ENGRAM_DIM,
        );
    }

    let mut drafter = Micro {
        seed: 500,
        ..Micro::default()
    };
    drafter.block("mtp.0", "blk.0", DRAFT_EXPERTS, false);
    drafter.fp8("mtp.0.main_proj", "fc.weight", HIDDEN, HIDDEN);
    drafter.norm("mtp.0.main_norm.weight", "enc.output_norm.weight", HIDDEN);
    drafter.norm("mtp.0.norm.weight", "output_norm.weight", HIDDEN);
    drafter.bf16(
        "mtp.0.markov_head.embed.weight",
        Some("markov_w1.weight"),
        &[VOCAB, MARKOV],
    );
    drafter.bf16(
        "mtp.0.markov_head.head.weight",
        Some("markov_w2.weight"),
        &[VOCAB, MARKOV],
    );
    drafter.bf16(
        "mtp.0.confidence_head.proj.weight",
        Some("conf_proj.weight"),
        &[1, HIDDEN + MARKOV],
    );

    let mut junk = Micro {
        seed: 900,
        ..Micro::default()
    };
    for name in [
        "vision.patch_embed.proj.weight",
        "aligner.w1.weight",
        "image_start",
    ] {
        junk.bf16(name, None, &[4, 4]);
    }

    let mut tensors: Vec<&Tensor> = main
        .tensors
        .iter()
        .chain(&drafter.tensors)
        .chain(&junk.tensors)
        .filter(|t| !without.contains(&t.0.as_str()))
        .collect();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut weight_map = serde_json::Map::new();
    for (shard, part) in tensors.chunks(tensors.len().div_ceil(2)).enumerate() {
        let name = format!("model-0000{}-of-00002.safetensors", shard + 1);
        let listed: Vec<(&str, &str, &[usize], Vec<u8>)> = part
            .iter()
            .map(|t| (t.0.as_str(), t.1, t.2.as_slice(), t.3.clone()))
            .collect();
        write_file(&dir.join(&name), &listed);
        for t in part {
            weight_map.insert(t.0.clone(), json!(name));
        }
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        json!({"metadata": {}, "weight_map": weight_map}).to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("config.json"),
        config_json(with_engram, &sizes).to_string(),
    )
    .unwrap();
    Fixture {
        dir,
        main,
        drafter,
        sizes,
    }
}

fn config_json(with_engram: bool, sizes: &EngramSizes) -> serde_json::Value {
    let mut text: serde_json::Map<String, serde_json::Value> = [
        ("model_type", json!("deepseek_v41_text")),
        ("vocab_size", json!(VOCAB)),
        ("hidden_size", json!(HIDDEN)),
        ("moe_intermediate_size", json!(MOE_FF)),
        ("num_hidden_layers", json!(2)),
        ("num_attention_heads", json!(HEADS)),
        ("num_key_value_heads", json!(1)),
        ("head_dim", json!(HEAD_DIM)),
        ("qk_rope_head_dim", json!(16)),
        ("q_lora_rank", json!(Q_LORA)),
        ("o_lora_rank", json!(O_LORA)),
        ("o_groups", json!(O_GROUPS)),
        ("swiglu_limit", json!(10.0)),
        ("rms_norm_eps", json!(1e-20)),
        ("max_position_embeddings", json!(4096)),
        ("rope_theta", json!(10000)),
        ("rope_scaling", json!({"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 1024})),
        ("n_routed_experts", json!(EXPERTS)),
        ("n_shared_experts", json!(1)),
        ("num_experts_per_tok", json!(2)),
        ("scoring_func", json!("sqrtsoftplus")),
        ("norm_topk_prob", json!(true)),
        ("routed_scaling_factor", json!(1.5)),
        ("sliding_window", json!(16)),
        ("compress_ratios", json!([0, 2, 0])),
        ("compress_rope_theta", json!(160000)),
        ("kv_source_layer_ids", json!([1])),
        ("index_source_layer_ids", json!([1])),
        ("index_n_heads", json!(INDEX_HEADS)),
        ("index_head_dim", json!(INDEX_DIM)),
        ("index_topk", json!(8)),
        ("candidate_source_layer_id", json!(-1)),
        ("hc_mult", json!(HC)),
        ("hc_sinkhorn_iters", json!(20)),
        ("hc_eps", json!(1e-6)),
        ("num_nextn_predict_layers", json!(1)),
        ("dspark_block_size", json!(3)),
        ("dspark_noise_token_id", json!(261)),
        ("dspark_target_layer_ids", json!([1])),
        ("dspark_markov_rank", json!(MARKOV)),
        ("dspark_n_routed_experts", json!(DRAFT_EXPERTS)),
        ("dspark_num_experts_per_tok", json!(1)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    if with_engram {
        let map = &mut text;
        map.insert("engram_layer_ids".into(), json!([0]));
        map.insert("engram_num_embeddings".into(), json!([sizes.rows]));
        map.insert("engram_max_ngram_size".into(), json!(NGRAM));
        map.insert("engram_vocab_size".into(), json!(ENGRAM_VOCAB));
        map.insert("engram_n_heads".into(), json!(ENGRAM_HEADS));
        map.insert("engram_head_dim".into(), json!(ENGRAM_DIM));
        map.insert("engram_pad_token_id".into(), json!(260));
        map.insert("engram_compressed_vocab_size".into(), json!(sizes.classes));
    }
    json!({
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "bos_token_id": BOS,
        "eos_token_id": EOS,
        "pad_token_id": 260,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
        "text_config": text,
    })
}

/// A tensor of `content` as its shape and `f32` values.
fn values(content: &Content, name: &str) -> (GgufType, Vec<usize>, Vec<f32>) {
    let info = content.info(name).unwrap();
    let dims = info.shape.dims().to_vec();
    let cols = *dims.last().unwrap();
    let data = content.reader().bytes(name).unwrap();
    let row_bytes = info.dtype.byte_len(cols).unwrap();
    let mut out = vec![0.0f32; dims.iter().product()];
    for (bytes, row) in data.chunks_exact(row_bytes).zip(out.chunks_exact_mut(cols)) {
        dequant_row(info.dtype, bytes, 1.0, row).unwrap();
    }
    (info.dtype, dims, out)
}

fn convert_fixture(
    fixture: &Fixture,
    out_dir: &str,
    config: ConvertConfig,
    role: Role,
) -> (PathBuf, PathBuf) {
    let ck = CheckpointDir::open(&fixture.dir).unwrap();
    let out = fixture
        .dir
        .join(out_dir)
        .join(format!("model-{}.gguf", config.quant.name()));
    let path = convert_set(&ck, &out, config, role, None).unwrap();
    (out, path)
}

fn f16_exact() -> ConvertConfig {
    ConvertConfig {
        quant: WeightQuant::F16,
        engram_quant: EngramQuant::Bf16,
    }
}

fn string(content: &Content, key: &str) -> String {
    content.metadata[key].to_string().unwrap().clone()
}

fn u32s(content: &Content, key: &str) -> Vec<u32> {
    content.metadata[key]
        .to_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_u32().unwrap())
        .collect()
}

#[test]
fn the_micro_checkpoint_converts_to_exactly_the_values_it_defines() {
    let fixture = checkpoint(true, &[]);
    let (out, first) = convert_fixture(&fixture, "f16", f16_exact(), Role::Main);
    assert_eq!(
        first.file_name().and_then(|n| n.to_str()),
        Some("model-f16-00001-of-00003.gguf")
    );
    assert_eq!(converted(&out, Role::Main), Some(first.clone()));
    assert!(!out.exists(), "the nominal path is not a file of the set");
    let content = Content::open(&first).unwrap();
    assert_eq!(content.files().len(), 3);

    let mut have: Vec<&str> = content.tensor_infos.keys().map(String::as_str).collect();
    have.sort_unstable();
    let want: Vec<&str> = fixture.main.want.keys().map(String::as_str).collect();
    assert_eq!(
        have, want,
        "the tensors of the set are exactly the planned ones"
    );
    for (name, (shape, expected)) in &fixture.main.want {
        let (dtype, dims, got) = values(&content, name);
        assert_eq!(&dims, shape, "{name}");
        assert_eq!(&got, expected, "{name} ({})", dtype.name());
    }
    // The storage types of design section 4.2 at quant f16.
    let dtype = |name: &str| content.info(name).unwrap().dtype.name().to_string();
    assert_eq!(dtype("blk.1.hc_attn_fn.weight"), "F32");
    assert_eq!(dtype("blk.0.ffn_gate_inp.weight"), "F32");
    assert_eq!(dtype("blk.0.engram_gate.weight"), "F32");
    assert_eq!(dtype("blk.1.attn_compressor_kv.weight"), "F16");
    assert_eq!(dtype("blk.0.attn_q_a.weight"), "F16");
    assert_eq!(dtype("blk.0.ffn_gate_exps.weight"), "F16");
    assert_eq!(dtype("per_layer_token_embd.weight"), "BF16");
    assert!(content.tensor_infos.keys().all(|n| !n.contains("bias_vl")));
}

#[test]
fn the_metadata_is_llama_cpps_deepseek41_dialect() {
    let fixture = checkpoint(true, &[]);
    let (_, first) = convert_fixture(&fixture, "meta", f16_exact(), Role::Main);
    let md = Content::open(&first).unwrap().metadata;
    let get = |key: &str| &md[key];
    assert_eq!(
        get("general.architecture").to_string().unwrap(),
        "deepseek41"
    );
    for (key, want) in [
        ("deepseek41.block_count", 2),
        ("deepseek41.context_length", 4096),
        ("deepseek41.embedding_length", HIDDEN as u32),
        ("deepseek41.attention.head_count", HEADS as u32),
        ("deepseek41.attention.head_count_kv", 1),
        ("deepseek41.attention.key_length", HEAD_DIM as u32),
        ("deepseek41.attention.q_lora_rank", Q_LORA as u32),
        ("deepseek41.attention.sliding_window", 16),
        ("deepseek41.attention.output_group_count", O_GROUPS as u32),
        ("deepseek41.attention.output_lora_rank", O_LORA as u32),
        ("deepseek41.attention.indexer.top_k", 8),
        ("deepseek41.expert_count", EXPERTS as u32),
        ("deepseek41.expert_used_count", 2),
        ("deepseek41.expert_shared_count", 1),
        ("deepseek41.expert_feed_forward_length", MOE_FF as u32),
        ("deepseek41.expert_gating_func", 4),
        ("deepseek41.hyper_connection.count", HC as u32),
        ("deepseek41.hyper_connection.sinkhorn_iterations", 20),
        ("deepseek41.hash_layer_count", 0),
        ("deepseek41.ple.ngram_size", NGRAM as u32),
        ("deepseek41.ple.heads_per_ngram", ENGRAM_HEADS as u32),
        (
            "deepseek41.embedding_length_per_layer_input",
            ENGRAM_DIM as u32,
        ),
        ("deepseek41.engram.pad_token_id", 260),
        ("tokenizer.ggml.bos_token_id", BOS as u32),
        ("tokenizer.ggml.eos_token_id", EOS as u32),
    ] {
        assert_eq!(get(key).to_u32().unwrap(), want, "{key}");
    }
    assert_eq!(get("split.count").to_u64().unwrap(), 3);
    assert_eq!(get("split.no").to_u64().unwrap(), 0);
    assert_eq!(get("deepseek41.rope.freq_base").to_f32().unwrap(), 10000.0);
    assert_eq!(
        get("deepseek41.attention.compress_rope_freq_base")
            .to_f32()
            .unwrap(),
        160000.0
    );
    assert_eq!(
        get("deepseek41.rope.scaling.type").to_string().unwrap(),
        "yarn"
    );
    assert_eq!(get("deepseek41.rope.scaling.factor").to_f32().unwrap(), 4.0);
    assert_eq!(
        get("deepseek41.rope.scaling.original_context_length")
            .to_u32()
            .unwrap(),
        1024
    );
    assert_eq!(
        get("deepseek41.expert_weights_scale").to_f32().unwrap(),
        1.5
    );
    assert!(get("deepseek41.expert_weights_norm").to_bool().unwrap());
    assert!(!md.contains_key("deepseek41.attention.indexer.candidate_source_layer"));
    let ints = |key: &str| -> Vec<i64> {
        md[key]
            .to_vec()
            .unwrap()
            .iter()
            .map(|v| match v {
                Value::I32(n) => i64::from(*n),
                other => panic!("{key}: {other:?}"),
            })
            .collect()
    };
    assert_eq!(ints("deepseek41.attention.compress_ratios"), [0, 2, 0]);
    let content_md = Content::open(&first).unwrap();
    assert_eq!(
        u32s(&content_md, "deepseek41.attention.kv_source_layers"),
        [1]
    );
    assert_eq!(
        u32s(&content_md, "deepseek41.attention.indexer.source_layers"),
        [1]
    );
    assert_eq!(u32s(&content_md, "deepseek41.ple.layers"), [0]);
    let floats = md["deepseek41.swiglu_clamp_exp"].to_vec().unwrap();
    assert_eq!(floats.len(), 2);
    assert!(floats.iter().all(|v| v.to_f32().unwrap() == 10.0));

    // The Engram constants are the reference's.
    let primes = engram::primes(1, NGRAM, ENGRAM_HEADS, ENGRAM_VOCAB);
    let sizes: Vec<u64> = md["deepseek41.ple.head_vocab_sizes"]
        .to_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_u64().unwrap())
        .collect();
    assert_eq!(
        sizes,
        primes
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
    );
    let offsets: Vec<u64> = md["deepseek41.ple.head_offsets"]
        .to_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_u64().unwrap())
        .collect();
    assert_eq!(offsets, engram::head_offsets(&primes));
    let multipliers: Vec<u64> = md["deepseek41.ple.layer_multipliers"]
        .to_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_u64().unwrap())
        .collect();
    assert_eq!(
        multipliers,
        engram::multipliers(&[0], NGRAM, fixture.sizes.classes as u64)[0]
    );
    let map = md["deepseek41.engram.token_map"].to_vec().unwrap();
    assert_eq!(map.len(), VOCAB);

    // The tokenizer: the three-stage chain's name, no BOS added, the builtin template.
    assert_eq!(get("tokenizer.ggml.model").to_string().unwrap(), "gpt2");
    assert_eq!(
        get("tokenizer.ggml.pre").to_string().unwrap(),
        "deepseek-v3"
    );
    assert!(!get("tokenizer.ggml.add_bos_token").to_bool().unwrap());
    assert_eq!(md["tokenizer.ggml.tokens"].to_vec().unwrap().len(), VOCAB);
    assert_eq!(
        get("tokenizer.chat_template").to_string().unwrap(),
        crate::provider::local::chat::expand_template_name("deepseek-v41")
            .unwrap()
            .as_ref()
    );
    // The tokenizer loads the way the engine loads it.
    crate::provider::local::tokenizer::Tokenizer::from_gguf(&md).unwrap();
}

#[test]
fn at_the_default_precision_experts_stay_mxfp4_and_dense_weights_are_q8_0() {
    let fixture = checkpoint(true, &[]);
    let (_, first) = convert_fixture(&fixture, "q8", ConvertConfig::default(), Role::Main);
    let content = Content::open(&first).unwrap();
    for (name, (_, expected)) in &fixture.main.want {
        let (dtype, _, got) = values(&content, name);
        match dtype.name().as_str() {
            // A byte shuffle: exactly the checkpoint's values.
            "MXFP4" => assert_eq!(&got, expected, "{name}"),
            "Q8_0" => {
                // Half a step of the block's 8-bit grid, per 32 values.
                for (g, w) in got.chunks(32).zip(expected.chunks(32)) {
                    let amax = w.iter().fold(0f32, |m, x| m.max(x.abs()));
                    for (a, b) in g.iter().zip(w) {
                        assert!(
                            (a - b).abs() <= amax / 127.0 * 0.51 + 1e-6,
                            "{name}: {a} vs {b}"
                        );
                    }
                }
            }
            "F32" | "F16" => assert_eq!(&got, expected, "{name}"),
            other => panic!("{name} is stored as {other}"),
        }
    }
    let dtype = |name: &str| content.info(name).unwrap().dtype.name().to_string();
    assert_eq!(dtype("blk.1.ffn_up_exps.weight"), "MXFP4");
    assert_eq!(dtype("blk.0.ffn_down_exps.weight"), "MXFP4");
    assert_eq!(dtype("blk.0.attn_output_a.weight"), "Q8_0");
    assert_eq!(dtype("token_embd.weight"), "Q8_0");
    assert_eq!(dtype("per_layer_token_embd.weight"), "Q8_0");
    // The compressor and the indexer's small matrices stay half precision.
    assert_eq!(dtype("blk.1.attn_compressor_gate.weight"), "F16");
    assert_eq!(dtype("blk.1.indexer.proj.weight"), "F16");
}

#[test]
fn the_engram_table_follows_engram_quant() {
    let fixture = checkpoint(true, &[]);
    for (engram_quant, name) in [(EngramQuant::Q4_0, "Q4_0"), (EngramQuant::Q8_0, "Q8_0")] {
        let config = ConvertConfig {
            quant: WeightQuant::F16,
            engram_quant,
        };
        let (_, first) = convert_fixture(&fixture, name, config, Role::Main);
        let content = Content::open(&first).unwrap();
        let (dtype, dims, got) = values(&content, "per_layer_token_embd.weight");
        assert_eq!(dtype.name(), name);
        assert_eq!(dims, [fixture.sizes.rows as usize, ENGRAM_DIM]);
        // Per-row relative L2 error of the quantized table against the checkpoint's.
        let want = &fixture.main.want["per_layer_token_embd.weight"].1;
        let worst = got
            .chunks(ENGRAM_DIM)
            .zip(want.chunks(ENGRAM_DIM))
            .map(|(g, w)| {
                let num: f32 = g.iter().zip(w).map(|(a, b)| (a - b) * (a - b)).sum();
                let den: f32 = w.iter().map(|b| b * b).sum();
                (num / den.max(1e-30)).sqrt()
            })
            .fold(0f32, f32::max);
        eprintln!("ENGRAM|{name}|worst per-row relative L2 error {worst:.4}");
        assert!(
            worst < if name == "Q8_0" { 0.02 } else { 0.25 },
            "{name}: {worst}"
        );
    }
}

#[test]
fn a_model_without_engram_layers_writes_two_parts() {
    let fixture = checkpoint(false, &[]);
    let (out, first) = convert_fixture(&fixture, "two", f16_exact(), Role::Main);
    assert_eq!(
        first.file_name().and_then(|n| n.to_str()),
        Some("model-f16-00001-of-00002.gguf")
    );
    assert_eq!(converted(&out, Role::Main), Some(first.clone()));
    let content = Content::open(&first).unwrap();
    assert_eq!(content.files().len(), 2);
    assert!(
        !content
            .tensor_infos
            .contains_key("per_layer_token_embd.weight")
    );
    assert!(!content.metadata.contains_key("deepseek41.ple.layers"));
}

#[test]
fn a_finished_conversion_is_not_redone() {
    let fixture = checkpoint(true, &[]);
    let (out, first) = convert_fixture(&fixture, "again", f16_exact(), Role::Main);
    let stamp = |p: &Path| std::fs::metadata(p).unwrap().modified().unwrap();
    let before = stamp(&first);
    let ck = CheckpointDir::open(&fixture.dir).unwrap();
    let again = convert_set(&ck, &out, f16_exact(), Role::Main, None).unwrap();
    assert_eq!(again, first);
    assert_eq!(stamp(&first), before);
    let leftovers: Vec<_> = std::fs::read_dir(out.parent().unwrap())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().ends_with(".partial"))
        .collect();
    assert!(leftovers.is_empty());
}

#[test]
fn the_drafter_converts_to_a_dflash_file() {
    let fixture = checkpoint(true, &[]);
    let (out, path) = convert_fixture(&fixture, "drafter", f16_exact(), Role::Drafter);
    assert_eq!(
        path.file_name().and_then(|n| n.to_str()),
        Some("dspark-model-f16.gguf")
    );
    assert_eq!(converted(&out, Role::Drafter), Some(path.clone()));
    assert_eq!(
        converted(&out, Role::Main),
        None,
        "the drafter is not the backbone"
    );
    let content = Content::open(&path).unwrap();
    let mut have: Vec<&str> = content.tensor_infos.keys().map(String::as_str).collect();
    have.sort_unstable();
    let want: Vec<&str> = fixture.drafter.want.keys().map(String::as_str).collect();
    assert_eq!(have, want);
    for (name, (shape, expected)) in &fixture.drafter.want {
        let (_, dims, got) = values(&content, name);
        // The checkpoint's `(1, n)` confidence head is a vector in the GGUF.
        let flat;
        let shape = if name == "conf_proj.weight" {
            flat = vec![shape.iter().product()];
            &flat
        } else {
            shape
        };
        assert_eq!(&dims, shape, "{name}");
        assert_eq!(&got, expected, "{name}");
    }
    assert_eq!(
        content.info("conf_proj.weight").unwrap().dtype.name(),
        "F32"
    );
    assert_eq!(string(&content, "general.architecture"), "dflash");
    let md = &content.metadata;
    for (key, want) in [
        ("dflash.block_count", 1),
        ("dflash.block_size", 3),
        ("dflash.expert_count", DRAFT_EXPERTS as u32),
        ("dflash.expert_used_count", 1),
        ("dflash.feed_forward_length", MOE_FF as u32),
        ("dflash.hyper_connection.count", HC as u32),
        ("tokenizer.ggml.mask_token_id", 261),
    ] {
        assert_eq!(md[key].to_u32().unwrap(), want, "{key}");
    }
    assert!(md["dflash.hyper_connection.single_pass"].to_bool().unwrap());
    assert!(md["dflash.has_confidence_head"].to_bool().unwrap());
    assert!(md["dflash.sample_from_anchor"].to_bool().unwrap());
    // llama.cpp stores the layer ids as signed integers.
    let targets: Vec<i32> = md["dflash.target_layers"]
        .to_vec()
        .unwrap()
        .iter()
        .map(|v| match v {
            Value::I32(n) => *n,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(targets, [1]);
    // The target's vocabulary rides along.
    assert_eq!(md["tokenizer.ggml.tokens"].to_vec().unwrap().len(), VOCAB);
}

#[test]
fn a_missing_checkpoint_tensor_fails_the_conversion_naming_it() {
    for missing in [
        "layers.1.attn.wkv.scale",
        "layers.0.ffn.experts.2.w3.weight",
        "layers.1.hc_ffn_fn",
    ] {
        let fixture = checkpoint(true, &[missing]);
        let ck = CheckpointDir::open(&fixture.dir).unwrap();
        let out = fixture.dir.join("missing").join("model-f16.gguf");
        let e = convert_set(&ck, &out, f16_exact(), Role::Main, None)
            .unwrap_err()
            .to_string();
        assert!(e.contains(missing), "{missing}: {e}");
        assert!(converted(&out, Role::Main).is_none());
        let partials = std::fs::read_dir(out.parent().unwrap())
            .map(|d| d.filter_map(Result::ok).count())
            .unwrap_or(0);
        assert_eq!(partials, 1, "only the lock file remains: {missing}");
    }
}

#[test]
fn a_directory_source_converts_in_place_and_the_drafter_resolves_to_its_own_file() {
    use crate::provider::local::{
        DrafterSource, LocalModelConfig, ModelSource, resolve_drafter_path, resolve_model_path,
    };

    let fixture = checkpoint(true, &[]);
    let cache = fixture.dir.join("cache");
    let source = ModelSource::Dir(fixture.dir.clone());
    let config = LocalModelConfig {
        source: source.clone(),
        cache_dir: Some(cache.clone()),
        convert: f16_exact(),
        drafter: Some(DrafterSource::File(source.clone())),
        ..LocalModelConfig::default()
    };
    // The checkpoint directory is read where it is: nothing is fetched, and the
    // result lands in the cache next to nothing else of the checkpoint.
    let target = resolve_model_path(&config, &source).unwrap();
    let drafter = resolve_drafter_path(&config, &source).unwrap();
    let name = |p: &Path| p.file_name().and_then(|n| n.to_str()).unwrap().to_string();
    assert_eq!(name(&target), "model-f16-engram-bf16-00001-of-00003.gguf");
    assert_eq!(name(&drafter), "dspark-model-f16-engram-bf16.gguf");
    assert!(target.starts_with(&cache) && drafter.starts_with(&cache));
    assert_eq!(Content::open(&target).unwrap().files().len(), 3);
    assert_eq!(
        string(&Content::open(&drafter).unwrap(), "general.architecture"),
        "dflash"
    );
    // A second resolution reuses both files.
    let stamp = |p: &Path| std::fs::metadata(p).unwrap().modified().unwrap();
    let before = (stamp(&target), stamp(&drafter));
    assert_eq!(resolve_model_path(&config, &source).unwrap(), target);
    assert_eq!(resolve_drafter_path(&config, &source).unwrap(), drafter);
    assert_eq!((stamp(&target), stamp(&drafter)), before);
}
