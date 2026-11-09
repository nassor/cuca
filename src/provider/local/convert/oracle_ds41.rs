//! The DeepSeek-V4.1 converter against the torch oracle's tiny release-format
//! checkpoint (`.omp/local-models/ref/ds41/make_ref.py`, design §9.1).
//!
//! `CUCA_LOCAL_DS41_REF` names the oracle's output directory: `target/` is the
//! checkpoint (FP8 e4m3 on 32x32 `ue8m0` grids, packed FP4 experts, Engram
//! tables in e4m3 with a scale per 32 columns, bf16/f32 small tensors, a DSpark
//! drafter, vision tensors to skip), `expected/main.safetensors` and
//! `expected/dspark.safetensors` hold the exactly dequantized values every GGUF
//! tensor must have (computed in Python from the checkpoint bytes, by llama.cpp's
//! name map), `expected/tensors.json` the storage type of each tensor and
//! `expected/metadata_{main,dspark}.json` every metadata key with its type and
//! value (the `ple.*` constants from numpy and the reference `engram.py`).
//! Without the variable (or the directory) each test prints `SKIP:` and passes;
//! `CUCA_REQUIRE_LIVE=1` makes it fail.
//!
//! At `quant = f16` with a bf16 Engram table every value is exact, so the
//! converted tensors must equal the oracle's bit for bit; at the default
//! precision the f32, f16 and MXFP4 tensors are exact still and the Q8_0 ones
//! stay within half a step of their block.

use std::path::{Path, PathBuf};

use super::deepseek41::plan::Role;
use super::hf::CheckpointDir;
use super::safetensors::Shards;
use super::safetensors::tests::temp_dir;
use super::{convert_set, converted};
use crate::provider::local::gguf::{Content, Value};
use crate::provider::local::model::quant::dequant_row;
use crate::provider::local::tokenizer::golden::skip;
use crate::provider::local::{ConvertConfig, EngramQuant, WeightQuant};

/// The oracle directory, or `None` after a `SKIP:` line.
fn reference() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("CUCA_LOCAL_DS41_REF").map(PathBuf::from) else {
        skip("set CUCA_LOCAL_DS41_REF to the ds41 oracle directory");
        return None;
    };
    if !dir.join("target").is_dir() || !dir.join("expected").is_dir() {
        skip(&format!(
            "`{}` has no target/ and expected/ (run make_ref.py)",
            dir.display()
        ));
        return None;
    }
    Some(dir)
}

/// The converted set in a directory removed on drop.
struct Converted {
    dir: PathBuf,
    content: Content,
}

impl Drop for Converted {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn convert_oracle(root: &Path, config: ConvertConfig, role: Role) -> Converted {
    let ck = CheckpointDir::open(&root.join("target")).expect("the oracle checkpoint opens");
    let dir = temp_dir("ds41-oracle");
    let out = dir.join(format!("model-{}.gguf", config.quant.name()));
    let first = convert_set(&ck, &out, config, role, None).expect("the oracle checkpoint converts");
    assert_eq!(converted(&out, role), Some(first.clone()));
    let content = Content::open(&first).expect("the converted set opens");
    Converted { dir, content }
}

/// The storage class the oracle names (`f32`, `f16`, `bf16`, `q8_0`, `q4_0`,
/// `mxfp4`) of a GGUF type.
fn class(dtype: crate::provider::local::gguf::GgufType) -> String {
    dtype.name().to_lowercase()
}

/// A tensor of `content` as its shape and `f32` values.
fn values(content: &Content, name: &str) -> (Vec<usize>, Vec<f32>) {
    let info = content.info(name).unwrap();
    let dims = info.shape.dims().to_vec();
    let cols = *dims.last().unwrap();
    let data = content.reader().bytes(name).unwrap();
    let row_bytes = info.dtype.byte_len(cols).unwrap();
    let mut out = vec![0.0f32; dims.iter().product()];
    for (bytes, row) in data.chunks_exact(row_bytes).zip(out.chunks_exact_mut(cols)) {
        dequant_row(info.dtype, bytes, 1.0, row).unwrap();
    }
    (dims, out)
}

/// The oracle's expected tensors (`expected/<which>.safetensors`).
fn expected(root: &Path, which: &str) -> Shards {
    Shards::open(&[root.join("expected").join(format!("{which}.safetensors"))]).unwrap()
}

fn tensor_table(root: &Path, which: &str) -> serde_json::Map<String, serde_json::Value> {
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("expected/tensors.json")).unwrap())
            .unwrap();
    json[which].as_object().unwrap().clone()
}

/// Whether `value` is the metadata value the oracle lists (`type`: `u32`,
/// `i32`, `u64`, `f32`, `bool`, `str` or `arr:<type>`).
fn same(value: &Value, ty: &str, want: &serde_json::Value) -> Result<(), String> {
    let wrong = |what: &str| Err(format!("{what}: got {value:?}, want {ty} {want}"));
    match (ty, value) {
        ("u32", Value::U32(v)) if want.as_u64() == Some(u64::from(*v)) => Ok(()),
        ("i32", Value::I32(v)) if want.as_i64() == Some(i64::from(*v)) => Ok(()),
        ("u64", Value::U64(v)) if want.as_u64() == Some(*v) => Ok(()),
        ("f32", Value::F32(v)) if want.as_f64().map(|w| w as f32) == Some(*v) => Ok(()),
        ("bool", Value::Bool(v)) if want.as_bool() == Some(*v) => Ok(()),
        ("str", Value::String(v)) if want.as_str() == Some(v.as_str()) => Ok(()),
        (arr, Value::Array(items)) if arr.starts_with("arr:") => {
            let want = want.as_array().ok_or("expected an array")?;
            if items.len() != want.len() {
                return Err(format!("{} items, want {}", items.len(), want.len()));
            }
            let element = &arr["arr:".len()..];
            for (i, (item, w)) in items.iter().zip(want).enumerate() {
                same(item, element, w).map_err(|e| format!("item {i}: {e}"))?;
            }
            Ok(())
        }
        _ => wrong("type or value differs"),
    }
}

/// Every key of `expected/<file>` is in `content` with its type and value, and
/// the file has no other key than the ones the converter adds on its own.
fn check_metadata(root: &Path, file: &str, content: &Content) {
    let list: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(root.join("expected").join(file)).unwrap())
            .unwrap();
    let mut failures = Vec::new();
    for entry in &list {
        let key = entry["key"].as_str().unwrap();
        match content.metadata.get(key) {
            None => failures.push(format!("{key}: missing")),
            Some(v) => {
                if let Err(e) = same(v, entry["type"].as_str().unwrap(), &entry["value"]) {
                    failures.push(format!("{key}: {e}"));
                }
            }
        }
    }
    let listed: Vec<&str> = list.iter().filter_map(|e| e["key"].as_str()).collect();
    for key in content.metadata.keys() {
        let own = key.starts_with("general.")
            || key.starts_with("split.")
            || key == "tokenizer.chat_template";
        if !own && !listed.contains(&key.as_str()) {
            failures.push(format!("{key}: not in the oracle's list"));
        }
    }
    assert!(failures.is_empty(), "{file}:\n{}", failures.join("\n"));
}

/// Compare the tensors of `content` with `which`'s: the same names and shapes,
/// the storage classes of `class_of`, and values equal (exact classes) or within
/// half a step of their 32-block (`q8_0`).
fn check_tensors(
    root: &Path,
    which: &str,
    content: &Content,
    class_of: &dyn Fn(&str, &serde_json::Value) -> String,
) {
    let want = expected(root, which);
    let table = tensor_table(root, which);
    let mut have: Vec<&str> = content.tensor_infos.keys().map(String::as_str).collect();
    have.sort_unstable();
    assert_eq!(have, want.names(), "{which}: the tensors of the set");
    let mut checked = 0;
    for name in want.names() {
        let (dims, got) = values(content, name);
        let meta = want.tensor(name).unwrap();
        assert_eq!(dims, meta.shape, "{name}: shape");
        let wanted = want.f32(name).unwrap();
        let dtype = class(content.info(name).unwrap().dtype);
        assert_eq!(dtype, class_of(name, &table[name]), "{name}: storage class");
        if dtype == "q8_0" || dtype == "q4_0" {
            let steps = if dtype == "q8_0" { 127.0 } else { 7.0 };
            for (g, w) in got.chunks(32).zip(wanted.chunks(32)) {
                let amax = w.iter().fold(0f32, |m, x| m.max(x.abs()));
                for (a, b) in g.iter().zip(w) {
                    assert!(
                        (a - b).abs() <= amax / steps * 0.6 + 1e-6,
                        "{name}: {a} vs {b}"
                    );
                }
            }
        } else {
            assert_eq!(got, wanted, "{name} ({dtype})");
        }
        checked += 1;
    }
    assert_eq!(checked, table.len());
}

fn exact(quant: WeightQuant) -> ConvertConfig {
    ConvertConfig {
        quant,
        engram_quant: EngramQuant::Bf16,
    }
}

#[test]
fn the_oracle_checkpoint_converts_to_exactly_its_float_tensors_and_metadata() {
    let Some(root) = reference() else {
        return;
    };
    let set = convert_oracle(&root, exact(WeightQuant::F16), Role::Main);
    check_metadata(&root, "metadata_main.json", &set.content);
    // The table is stored as bf16 here (exact), every other tensor as the
    // oracle says for quant f16.
    check_tensors(&root, "main", &set.content, &|name, entry| {
        if name == "per_layer_token_embd.weight" {
            "bf16".into()
        } else {
            entry["dtype_f16"].as_str().unwrap().into()
        }
    });
    assert_eq!(set.content.files().len(), 3);
}

#[test]
fn at_the_default_precision_the_experts_are_native_mxfp4() {
    let Some(root) = reference() else {
        return;
    };
    let set = convert_oracle(&root, ConvertConfig::default(), Role::Main);
    check_tensors(&root, "main", &set.content, &|_, entry| {
        entry["dtype_default"].as_str().unwrap().into()
    });
    // The MXFP4 stacks are the checkpoint's own FP4 values, not requantized.
    let experts = class(
        set.content
            .info("blk.1.ffn_gate_exps.weight")
            .unwrap()
            .dtype,
    );
    assert_eq!(experts, "mxfp4");
}

/// How the Engram table fares at the lossy precisions on the oracle's table
/// (per-row relative L2 error of the quantized rows against the exact ones;
/// the design's expectation for real rows is below 1 % at Q8_0).
#[test]
fn the_engram_table_follows_engram_quant_and_reports_its_error() {
    let Some(root) = reference() else {
        return;
    };
    let want = expected(&root, "main")
        .f32("per_layer_token_embd.weight")
        .unwrap();
    for (engram_quant, name, bound) in [
        (EngramQuant::Q8_0, "q8_0", 0.02),
        (EngramQuant::Q4_0, "q4_0", 0.25),
        (EngramQuant::Bf16, "bf16", 0.0),
    ] {
        let config = ConvertConfig {
            quant: WeightQuant::F16,
            engram_quant,
        };
        let set = convert_oracle(&root, config, Role::Main);
        let info = set.content.info("per_layer_token_embd.weight").unwrap();
        assert_eq!(class(info.dtype), name);
        let (dims, got) = values(&set.content, "per_layer_token_embd.weight");
        assert_eq!(dims[0] * dims[1], want.len());
        let cols = dims[1];
        let worst = got
            .chunks(cols)
            .zip(want.chunks(cols))
            .map(|(g, w)| {
                let num: f32 = g.iter().zip(w).map(|(a, b)| (a - b) * (a - b)).sum();
                let den: f32 = w.iter().map(|b| b * b).sum();
                (num / den.max(1e-30)).sqrt()
            })
            .fold(0f32, f32::max);
        eprintln!(
            "ENGRAM|{name}|{} rows|worst per-row relative L2 error {worst:.5}",
            dims[0]
        );
        assert!(worst <= bound, "{name}: {worst}");
    }
}

#[test]
fn the_oracles_drafter_converts_to_exactly_its_float_tensors_and_metadata() {
    let Some(root) = reference() else {
        return;
    };
    let set = convert_oracle(&root, exact(WeightQuant::F16), Role::Drafter);
    check_metadata(&root, "metadata_dspark.json", &set.content);
    check_tensors(&root, "dspark", &set.content, &|_, entry| {
        entry["dtype_f16"].as_str().unwrap().into()
    });
    assert_eq!(set.content.files().len(), 1);
    let default = convert_oracle(&root, ConvertConfig::default(), Role::Drafter);
    check_tensors(&root, "dspark", &default.content, &|_, entry| {
        entry["dtype_default"].as_str().unwrap().into()
    });
}

/// The ceil-division scale grid of an FP8 weight whose rows and columns end in
/// partial 32-blocks (no model dimension of the oracle can, see its README):
/// `extra/fp8_ceil_grid.safetensors` holds the weight, its grid and the
/// dequantized values.
#[test]
fn an_fp8_weight_with_partial_blocks_dequantizes_as_the_oracle_does() {
    let Some(root) = reference() else {
        return;
    };
    let shards = Shards::open(&[root.join("extra/fp8_ceil_grid.safetensors")]).unwrap();
    let want = shards.f32("w.expected").unwrap();
    let rows = shards.tensor("w.weight").unwrap().rows();
    for block_rows in [1, 7, 32, 33, rows] {
        let mut got = Vec::new();
        let mut first = 0;
        while first < rows {
            let n = block_rows.min(rows - first);
            let mut part = Vec::new();
            shards
                .block_scaled_rows("w.weight", first, n, &mut part)
                .unwrap();
            got.extend(part);
            first += n;
        }
        assert_eq!(got, want, "blocks of {block_rows} rows");
    }
}
