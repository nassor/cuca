//! Fixtures for the tests of split GGUF sets: a tiny model cut into parts named
//! like llama.cpp's `llama-gguf-split` output.

use crate::provider::local::gguf::{self, Content, Value};
use candle_core::Device;
use candle_core::quantized::QTensor;

use super::SplitName;
use crate::provider::local::model::tiny::TinyModel;
use std::fs::File;
use std::path::{Path, PathBuf};

pub(crate) fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cuca-merge-{name}-{}-{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos())
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub(crate) type Loaded = (Vec<(String, Value)>, Vec<(String, QTensor)>);

/// The metadata and tensors of a GGUF file, read through candle.
pub(crate) fn load(path: &Path) -> Loaded {
    let content = Content::open(path).unwrap();
    let mut metadata: Vec<_> = content
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    metadata.sort_by(|a, b| a.0.cmp(&b.0));
    let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    names.sort();
    let mut reader = content.reader();
    let tensors = names
        .into_iter()
        .map(|n| {
            let t = reader.tensor(&n, &Device::Cpu).unwrap();
            (n, t)
        })
        .collect();
    (metadata, tensors)
}

/// Cut `model` into parts named like llama.cpp's, tensor `i` going to part
/// `assign(i)`; the first part keeps the model's metadata, the others only
/// `split.*` keys. `count` is what every part claims.
pub(crate) fn split(
    model: &TinyModel,
    dir: &Path,
    parts: usize,
    claimed: usize,
    assign: impl Fn(usize) -> usize,
) -> Vec<PathBuf> {
    let (metadata, tensors) = load(&model.path);
    let split_name = SplitName {
        base: "tiny".into(),
        no: 1,
        count: parts as u32,
    };
    let total = tensors.len() as i32;
    (0..parts)
        .map(|p| {
            let path = dir.join(split_name.part_name(p as u32 + 1));
            let mut kv: Vec<(String, Value)> = if p == 0 { metadata.clone() } else { Vec::new() };
            kv.push(("split.no".into(), Value::U16(p as u16)));
            kv.push(("split.count".into(), Value::U16(claimed as u16)));
            kv.push(("split.tensors.count".into(), Value::I32(total)));
            let kv_refs: Vec<(&str, &Value)> = kv.iter().map(|(k, v)| (k.as_str(), v)).collect();
            let ts: Vec<(&str, &QTensor)> = tensors
                .iter()
                .enumerate()
                .filter(|(i, _)| assign(*i) == p)
                .map(|(_, (n, t))| (n.as_str(), t))
                .collect();
            let mut file = File::create(&path).unwrap();
            gguf::write(&mut file, &kv_refs, &ts).unwrap();
            path
        })
        .collect()
}
