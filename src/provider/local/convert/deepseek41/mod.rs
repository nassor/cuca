//! Conversion of a DeepSeek-V4.1 checkpoint (`DeepseekV41ForCausalLM`, the
//! release format of `deepseek-ai/DeepSeek-V4.1-Flash-1017`) to the GGUF files
//! the engine loads: llama.cpp's `deepseek41` layout, and the DSpark drafter as
//! its `dflash` file.
//!
//! - [`hparams`]: `config.json`, validated by llama.cpp's loader rules;
//! - [`plan`]: every GGUF tensor's name, shape, storage type and source, from
//!   the checkpoint's tensor headers alone;
//! - [`stream`]: the values, streamed: FP8 weights dequantized exactly from
//!   their 32x32 `ue8m0` scale grid in row blocks, FP4 experts byte-repacked into
//!   MXFP4 blocks (or dequantized exactly under `quant = f16`), `engram_gate` as
//!   `q_weight * k_weight`, the two Engram tables as one `per_layer_token_embd`
//!   in chunks of rows;
//! - [`meta`]: the `deepseek41.*` / `dflash.*` metadata including the Engram
//!   hash constants ([`super::engram`]).
//!
//! The backbone converts to a split set `<stem>-00001-of-0000N.gguf`: part 1
//! holds the metadata only, part 2 every tensor but the Engram table, part 3
//! the table (`N` is 2 for a model without Engram layers). The parts can be
//! moved to different volumes (the loader looks for a part next to the first
//! one, or in the directories it is told). Parts 2 and 3 are renamed into place
//! before part 1, so the existence of part 1 marks a finished conversion
//! ([`finished_set`], [`super::converted`]). The conversion reports bytes written
//! and bytes to write through the progress callback. Before it writes anything it
//! compares the free space of the output volume (`sysinfo`'s disk list) with the
//! bytes to write and fails with [`CucaError::Config`] naming the sizes; a platform
//! that cannot report free space skips the check.
//!
//! Peak memory is two block buffers of at most 32 MiB of `f32` plus the encoded
//! block (see [`stream`]), never a tensor: the 196.6 billion values of the Engram
//! tables stream through.

#[cfg(test)]
mod e2e;
mod hparams;
mod meta;
pub(crate) mod plan;
mod stream;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use self::hparams::Config;
use self::plan::{Item, Part, Plan, Role};
use super::arch::ConvArch;
use super::config::Hp;
use super::gguf_writer::{GgufWriter, TensorSpec};
use super::hf::CheckpointDir;
use super::safetensors::Shards;
use super::vocab;
use super::{WRITE_BUFFER, cfg_err};
use crate::error::CucaError;
use crate::provider::local::gguf::{SplitName, Value};
use crate::provider::local::tokenizer::Tokenizer;
use crate::provider::local::{ConvertConfig, DownloadProgressCallback, download, download::report};

/// Slack on top of the tensor data for the metadata and the GGUF headers.
const HEADER_SLACK: u64 = 256 << 20;

/// The path of part `no` of `count` of the split set whose nominal single-file
/// path is `out` (`model-q8_0.gguf` -> `model-q8_0-00002-of-00003.gguf`).
fn part_path(out: &Path, no: u32, count: u32) -> PathBuf {
    let base = out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let name = SplitName { base, no, count }.part_name(no);
    out.with_file_name(name)
}

/// The first part of a finished conversion to the nominal path `out`: the
/// three-part set, or the two-part one of a model without Engram layers. The
/// first part is renamed into place last, so its existence marks a complete set.
pub(crate) fn finished_set(out: &Path) -> Option<PathBuf> {
    [3, 2]
        .into_iter()
        .map(|count| part_path(out, 1, count))
        .find(|first| first.exists())
}

/// Where the DSpark drafter converted from the same checkpoint as the
/// backbone of the nominal path `out` is written: `dspark-<file name>`.
pub(crate) fn drafter_path(out: &Path) -> PathBuf {
    let name = out
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.gguf");
    out.with_file_name(format!("dspark-{name}"))
}

/// The checkpoint, planned: everything a conversion needs before it writes.
struct Prepared {
    shards: Shards,
    plan: Plan,
    kv: Vec<(String, Value)>,
}

/// Open the checkpoint and plan `role`.
fn prepare(ck: &CheckpointDir, convert: ConvertConfig, role: Role) -> Result<Prepared, CucaError> {
    let hp = Hp::new(&ck.config)?;
    let config = Config::read(&hp)?;
    let shards = Shards::open(&ck.shards)?;
    let plan = plan::plan(&shards, &config, convert.quant, convert.engram_quant, role)?;
    let name = super::model_name(&ck.dir);
    let vocab = vocab::build(ConvArch::Deepseek41, ck, &hp)?
        .ok_or_else(|| cfg_err("DeepSeek-V4.1 checkpoints carry a vocabulary"))?;
    // The tokenizer must load the way the engine will load it.
    let md = vocab.kv.iter().cloned().collect();
    Tokenizer::from_gguf(&md)
        .map_err(|e| cfg_err(format!("the converted tokenizer does not load: {e}")))?;
    let kv = match role {
        Role::Main => {
            let mut kv = meta::main_kv(
                &config,
                &name,
                convert.quant,
                ck.tokenizer_json.as_ref(),
                vocab.kv,
            )?;
            kv.extend(super::config::sampling_defaults(
                ck.generation_config.as_ref(),
            ));
            kv
        }
        Role::Drafter => meta::drafter_kv(&config, &name, convert.quant, vocab.kv)?,
    };
    Ok(Prepared { shards, plan, kv })
}

/// Free bytes of the volume holding `dir`, if the platform can tell.
fn available_space(dir: &Path) -> Option<u64> {
    let canonical = dir.canonicalize().ok()?;
    let text = canonical.to_string_lossy();
    // Windows' `\\?\C:\...` form does not match a disk's `C:\` mount point.
    let plain = PathBuf::from(text.strip_prefix(r"\\?\").unwrap_or(&text));
    let disks = sysinfo::Disks::new_with_refreshed_list().ok()?;
    disks
        .iter()
        .filter(|d| plain.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(sysinfo::Disk::available_space)
}

/// Fail before writing when `dir` cannot hold `need` bytes.
///
/// # Errors
///
/// [`CucaError::Config`] naming the sizes and what to change.
fn preflight(dir: &Path, need: u64, what: &str) -> Result<(), CucaError> {
    std::fs::create_dir_all(dir)?;
    let Some(free) = available_space(dir) else {
        return Ok(());
    };
    check_space(free, need, dir, what)
}

fn check_space(free: u64, need: u64, dir: &Path, what: &str) -> Result<(), CucaError> {
    if free >= need {
        return Ok(());
    }
    let gb = |b: u64| b as f64 / 1e9;
    Err(cfg_err(format!(
        "converting {what} writes {:.1} GB into `{}`, which has {:.1} GB free; free {:.1} GB more, choose another cache directory, or convert with a smaller precision (`ConvertConfig::quant`, `engram_quant: q4_0` halves the Engram table)",
        gb(need),
        dir.display(),
        gb(free),
        gb(need - free)
    )))
}

/// The conversion of the backbone to the nominal path `out` (see the module
/// docs), returning the first part of the set.
///
/// An existing finished set is left alone. `progress` receives bytes written
/// and bytes to write.
///
/// # Errors
///
/// [`CucaError::Config`] for a checkpoint the converter does not support, or
/// for too little free space; [`CucaError::Io`] for file errors.
pub(crate) fn convert(
    ck: &CheckpointDir,
    out: &Path,
    convert: ConvertConfig,
    progress: Option<&DownloadProgressCallback>,
) -> Result<PathBuf, CucaError> {
    let _lock = download::lock_file(out)?;
    if let Some(done) = finished_set(out) {
        return Ok(done);
    }
    let prepared = prepare(ck, convert, Role::Main)?;
    let backbone = prepared.plan.part(Part::Backbone);
    let engram = prepared.plan.part(Part::Engram);
    let count: u32 = if engram.is_empty() { 2 } else { 3 };
    let total = prepared.plan.bytes(Part::Backbone) + prepared.plan.bytes(Part::Engram);
    let dir = out.parent().unwrap_or(Path::new("."));
    preflight(dir, total + HEADER_SLACK, "DeepSeek-V4.1")?;

    let paths: Vec<PathBuf> = (1..=count).map(|no| part_path(out, no, count)).collect();
    let partials: Vec<PathBuf> = paths
        .iter()
        .map(|p| download::with_suffix(p, ".partial"))
        .collect();
    let n_tensors = backbone.len() + engram.len();
    let result = (|| {
        let split = |no: u16| {
            vec![
                ("split.no".to_string(), Value::U16(no)),
                ("split.count".to_string(), Value::U16(count as u16)),
                (
                    "split.tensors.count".to_string(),
                    Value::I32(n_tensors as i32),
                ),
            ]
        };
        let mut done = 0u64;
        report(progress, 0, total);
        let mut first = prepared.kv.clone();
        first.extend(split(0));
        write_part(
            &partials[0],
            &first,
            &[],
            &prepared.shards,
            &mut done,
            total,
            progress,
        )?;
        write_part(
            &partials[1],
            &split(1),
            &backbone,
            &prepared.shards,
            &mut done,
            total,
            progress,
        )?;
        if !engram.is_empty() {
            write_part(
                &partials[2],
                &split(2),
                &engram,
                &prepared.shards,
                &mut done,
                total,
                progress,
            )?;
        }
        // The first part last: its presence says the set is complete.
        for no in (0..partials.len()).rev() {
            std::fs::rename(&partials[no], &paths[no])?;
        }
        Ok(())
    })();
    if let Err(e) = result {
        for partial in &partials {
            let _ = std::fs::remove_file(partial);
        }
        return Err(e);
    }
    Ok(paths
        .into_iter()
        .next()
        .unwrap_or_else(|| out.to_path_buf()))
}

/// The conversion of the DSpark drafter (`mtp.*` of the same checkpoint) to the
/// single `dflash` file [`drafter_path`]`(out)`.
///
/// # Errors
///
/// As [`convert`]; also a checkpoint without `dspark_*` configuration.
pub(crate) fn convert_drafter(
    ck: &CheckpointDir,
    out: &Path,
    convert: ConvertConfig,
    progress: Option<&DownloadProgressCallback>,
) -> Result<PathBuf, CucaError> {
    let out = drafter_path(out);
    let out = out.as_path();
    let _lock = download::lock_file(out)?;
    if out.exists() {
        return Ok(out.to_path_buf());
    }
    let prepared = prepare(ck, convert, Role::Drafter)?;
    let items = prepared.plan.part(Part::Backbone);
    let total = prepared.plan.bytes(Part::Backbone);
    preflight(
        out.parent().unwrap_or(Path::new(".")),
        total + HEADER_SLACK,
        "the DeepSeek-V4.1 DSpark drafter",
    )?;
    let partial = download::with_suffix(out, ".partial");
    let mut done = 0u64;
    report(progress, 0, total);
    let result = write_part(
        &partial,
        &prepared.kv,
        &items,
        &prepared.shards,
        &mut done,
        total,
        progress,
    )
    .and_then(|()| std::fs::rename(&partial, out).map_err(CucaError::from));
    if let Err(e) = result {
        let _ = std::fs::remove_file(&partial);
        return Err(e);
    }
    Ok(out.to_path_buf())
}

/// Whether `config` is a DeepSeek-V4.1 checkpoint.
pub(crate) fn is_deepseek41(config: &serde_json::Value) -> bool {
    matches!(ConvArch::from_config(config), Ok(ConvArch::Deepseek41))
}

/// Write one GGUF file: `kv`, then `items` streamed from `shards`.
fn write_part(
    path: &Path,
    kv: &[(String, Value)],
    items: &[&Item],
    shards: &Shards,
    done: &mut u64,
    total: u64,
    progress: Option<&DownloadProgressCallback>,
) -> Result<(), CucaError> {
    let specs: Vec<TensorSpec> = items.iter().map(|i| i.spec()).collect();
    let file = File::create(path)?;
    let mut writer = GgufWriter::start(BufWriter::with_capacity(WRITE_BUFFER, file), kv, &specs)?;
    for item in items {
        stream::write_item(item, shards, &mut |bytes| {
            writer.write(bytes)?;
            *done += bytes.len() as u64;
            report(progress, *done, total);
            Ok(())
        })?;
    }
    let mut sink = writer.finish()?;
    sink.flush()?;
    sink.get_ref().sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_are_named_like_llama_cpps_split_files() {
        let out = Path::new("cache").join("model-q8_0.gguf");
        assert_eq!(
            part_path(&out, 2, 3),
            Path::new("cache").join("model-q8_0-00002-of-00003.gguf")
        );
        assert_eq!(
            drafter_path(&out),
            Path::new("cache").join("dspark-model-q8_0.gguf")
        );
        let engram = Path::new("model-q8_0-engram-q4_0.gguf");
        assert_eq!(
            part_path(engram, 1, 3).file_name().and_then(|n| n.to_str()),
            Some("model-q8_0-engram-q4_0-00001-of-00003.gguf")
        );
    }

    #[test]
    fn too_little_space_is_refused_with_the_sizes() {
        let dir = Path::new("cache");
        assert!(check_space(10, 10, dir, "x").is_ok());
        let e = check_space(100_000_000_000, 298_000_000_000, dir, "DeepSeek-V4.1")
            .unwrap_err()
            .to_string();
        assert!(e.contains("298.0 GB") && e.contains("100.0 GB free"), "{e}");
        assert!(
            e.contains("198.0 GB more") && e.contains("engram_quant"),
            "{e}"
        );
    }

    #[test]
    fn the_free_space_of_the_output_volume_is_read_and_enforced() {
        let dir = super::super::safetensors::tests::temp_dir("space");
        let free = available_space(&dir).expect("this platform reports free space");
        assert!(free > 0);
        assert!(preflight(&dir, 1, "a test model").is_ok());
        let e = preflight(&dir, u64::MAX, "a test model")
            .unwrap_err()
            .to_string();
        assert!(e.contains("a test model") && e.contains("GB free"), "{e}");
        // The directory is created when it is missing.
        let nested = dir.join("a").join("b");
        assert!(preflight(&nested, 1, "x").is_ok() && nested.is_dir());
        let _ = std::fs::remove_dir_all(dir);
    }
}
