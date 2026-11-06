//! Conversion of a Hugging Face safetensors checkpoint to a GGUF file the
//! engine loads unchanged.
//!
//! A port of the pinned llama.cpp converter (`conversion/*.py` at commit
//! `67107ac34f62882d8b53ea41ad68010ea635b9be`) for the architectures of
//! `model/arch.rs`, in pure Rust:
//!
//! - [`hf`]: the checkpoint's files (`config.json`, tokenizer files, shards),
//!   and their download;
//! - [`safetensors`]: shard headers and positional reads, one row block at a
//!   time (the crate forbids `unsafe`, so nothing is memory-mapped);
//! - [`config`]: the `{arch}.*` metadata and generated tensors
//!   (`set_gguf_parameters`, `generate_extra_tensors`);
//! - [`tensors`]: tensor names and values (`modify_tensors`), planned from the
//!   shard headers alone, then streamed and quantized one tensor at a time;
//! - [`vocab`], [`spm`], [`chktxt`]: the tokenizer metadata;
//! - [`gguf_writer`]: the streaming GGUF v3 writer.
//!
//! The header (metadata plus every tensor's name, shape and type) is known
//! before any weight is read, so the writer streams: peak memory is one
//! block of one tensor (at most 32 MiB of `f32`) plus its quantized form,
//! whatever the model's size. The result is checked before it is renamed into
//! place: the metadata must load through the engine's own `Hparams` and
//! tokenizer readers.

pub(crate) mod arch;
pub(crate) mod chktxt;
pub(crate) mod config;
pub(crate) mod gguf_merge;
pub(crate) mod gguf_writer;
pub(crate) mod hf;
#[cfg(test)]
mod oracle;
pub(crate) mod safetensors;
pub(crate) mod spm;
pub(crate) mod tensors;
pub(crate) mod vocab;

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use self::arch::ConvArch;
use self::gguf_writer::GgufWriter;
use self::hf::CheckpointDir;
use super::tokenizer::Tokenizer;
use super::{DownloadProgress, DownloadProgressCallback, ModelSource, WeightQuant, download};
use crate::error::CucaError;

/// Write buffer of the output file.
const WRITE_BUFFER: usize = 1 << 20;

/// Where the converted GGUF of `source` at `quant` lives under `cache_dir`:
/// `<cache>/converted/<repo / as -->/<revision>/model-<quant>.gguf` for a
/// checkpoint, `<cache>/converted/dir-<key>/model-<quant>.gguf` for a
/// directory (the key covers its path, `config.json` and shards).
pub(crate) fn output_path(
    cache_dir: &Path,
    source: &ModelSource,
    quant: WeightQuant,
) -> Result<PathBuf, CucaError> {
    let file = format!("model-{}.gguf", quant.name());
    match source {
        ModelSource::Checkpoint { repo, revision } => {
            // Validates the components the way the download cache does.
            download::cached_path(cache_dir, repo, revision, &file)?;
            Ok(cache_dir
                .join("converted")
                .join(repo.replace('/', "--"))
                .join(revision)
                .join(file))
        }
        ModelSource::Dir(dir) => Ok(cache_dir
            .join("converted")
            .join(format!("dir-{}", dir_key(dir)?))
            .join(file)),
        _ => Err(CucaError::Config(
            "only a checkpoint source is converted".into(),
        )),
    }
}

/// The first 16 hex digits of SHA-256 over the canonical path of `dir` and,
/// for `config.json` and every `*.safetensors` file in name order, its name,
/// byte length and modification time in seconds.
fn dir_key(dir: &Path) -> Result<String, CucaError> {
    let canonical = dir.canonicalize().map_err(|e| CucaError::Io {
        message: format!(
            "cannot resolve checkpoint directory `{}`: {e}",
            dir.display()
        ),
    })?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(&canonical)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n == "config.json" || n.ends_with(".safetensors"))
        })
        .collect();
    files.sort();
    let mut hasher = Sha256::new();
    hasher.update(canonical.to_string_lossy().as_bytes());
    for file in files {
        let meta = std::fs::metadata(&file)?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        hasher.update(b"\0");
        hasher.update(
            file.file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default()
                .as_bytes(),
        );
        hasher.update(meta.len().to_le_bytes());
        hasher.update(mtime.to_le_bytes());
    }
    Ok(hasher
        .finalize()
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// `general.name` for a checkpoint directory: the repository's name for a
/// cached download (`.../Owner--Name/<revision>`), else the directory's.
fn model_name(dir: &Path) -> String {
    let leaf = dir.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    let is_revision = leaf.len() == 40 && leaf.bytes().all(|b| b.is_ascii_hexdigit());
    let repo = if is_revision {
        dir.parent()
            .and_then(Path::file_name)
            .and_then(|n| n.to_str())
            .unwrap_or(leaf)
    } else {
        leaf
    };
    repo.rsplit("--").next().unwrap_or(repo).to_string()
}

fn report(progress: Option<&DownloadProgressCallback>, done: u64, total: u64) {
    if let Some(cb) = progress {
        (cb.0)(DownloadProgress {
            downloaded_bytes: done,
            total_bytes: Some(total),
        });
    }
}

/// Convert `checkpoint` to the GGUF file `out`.
///
/// Writes `<out>.partial` and renames it into place under an exclusive
/// `<out>.lock` file (the pattern of the model download), so concurrent
/// loaders convert once and a present `out` is always complete. An existing
/// `out` is left alone. `progress` receives the tensors converted so far in
/// [`DownloadProgress::downloaded_bytes`] and the tensor count in
/// [`DownloadProgress::total_bytes`].
///
/// # Errors
///
/// [`CucaError::Config`] for an architecture, tensor, rope scaling or
/// tokenizer the converter does not support, or a checkpoint the engine could
/// not load after conversion (naming the missing metadata key);
/// [`CucaError::Io`] for file errors.
pub(crate) fn convert(
    checkpoint: &CheckpointDir,
    out: &Path,
    quant: WeightQuant,
    progress: Option<&DownloadProgressCallback>,
) -> Result<(), CucaError> {
    write_atomically(out, |partial| {
        convert_locked(checkpoint, partial, quant, progress)
    })
}

/// The exclusive `<out>.lock` file of an output, released on drop.
pub(crate) struct OutputLock {
    _file: File,
}

/// Take the exclusive `<out>.lock` file of `out` (creating the directory and
/// the empty lock file), waiting for another loader that holds it.
pub(crate) fn lock_output(out: &Path) -> Result<OutputLock, CucaError> {
    let parent = out.parent().ok_or_else(|| {
        CucaError::Config(format!(
            "output `{}` has no parent directory",
            out.display()
        ))
    })?;
    std::fs::create_dir_all(parent)?;
    let mut lock_name = out.as_os_str().to_owned();
    lock_name.push(".lock");
    let lock = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(PathBuf::from(lock_name))?;
    lock.lock()?;
    Ok(OutputLock { _file: lock })
}

/// Produce the file `out` with `build`, which writes the path it is given.
///
/// Writes `<out>.partial` and renames it into place under an exclusive
/// `<out>.lock` file (the pattern of the model download), so concurrent
/// loaders build once and a present `out` is always complete. An existing
/// `out` is left alone and `build` is not called; a failed build removes the
/// partial file.
pub(crate) fn write_atomically(
    out: &Path,
    build: impl FnOnce(&Path) -> Result<(), CucaError>,
) -> Result<(), CucaError> {
    let lock = lock_output(out)?;
    build_locked(&lock, out, build)
}

/// [`write_atomically`] for a caller that already holds `out`'s lock.
pub(crate) fn build_locked(
    _lock: &OutputLock,
    out: &Path,
    build: impl FnOnce(&Path) -> Result<(), CucaError>,
) -> Result<(), CucaError> {
    // Another loader may have built the file while this one waited.
    if out.exists() {
        return Ok(());
    }
    let mut partial_name = out.as_os_str().to_owned();
    partial_name.push(".partial");
    let partial = PathBuf::from(partial_name);
    match build(&partial) {
        Ok(()) => std::fs::rename(&partial, out)?,
        Err(e) => {
            let _ = std::fs::remove_file(&partial);
            return Err(e);
        }
    }
    Ok(())
}

fn convert_locked(
    ck: &CheckpointDir,
    partial: &Path,
    quant: WeightQuant,
    progress: Option<&DownloadProgressCallback>,
) -> Result<(), CucaError> {
    let arch = ConvArch::from_config(&ck.config)?;
    let hp = config::Hp::new(&ck.config)?.with_defaults(arch);
    let shards = safetensors::Shards::open(&ck.shards)?;
    let n_mtp = if arch.is_qwen35() {
        tensors::mtp_blocks(&shards)
    } else {
        0
    };
    let meta = config::build(arch, &hp, &model_name(&ck.dir), quant, n_mtp)?;
    let vocab = vocab::build(arch, ck, &hp)?;
    // Gemma 3 and 4 drop the embedding rows past the real vocabulary.
    let n_vocab = vocab
        .as_ref()
        .filter(|_| matches!(arch, ConvArch::Gemma3 | ConvArch::Gemma4))
        .map(|v| v.n_vocab);
    let plans = tensors::plan(arch, &hp, &shards, meta.extra, n_vocab)?;

    let mut kv = meta.kv;
    if let Some(v) = vocab {
        kv.extend(v.kv);
    }
    let names: HashSet<&str> = plans.iter().map(|p| p.name.as_str()).collect();
    config::check(&kv, &|n| names.contains(n))?;
    if arch != ConvArch::Gemma4Assistant {
        let md: HashMap<String, _> = kv.iter().cloned().collect();
        Tokenizer::from_gguf(&md).map_err(|e| {
            CucaError::Config(format!("the converted tokenizer does not load: {e}"))
        })?;
    }

    let specs: Vec<_> = plans.iter().map(|p| p.spec(quant)).collect();
    let file = File::create(partial)?;
    let mut writer = GgufWriter::start(BufWriter::with_capacity(WRITE_BUFFER, file), &kv, &specs)?;
    let total = plans.len() as u64;
    report(progress, 0, total);
    for (i, (plan, spec)) in plans.iter().zip(&specs).enumerate() {
        plan.write(&shards, spec.dtype, &mut |bytes| writer.write(bytes))?;
        report(progress, i as u64 + 1, total);
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
    fn output_paths_follow_the_cache_layout() {
        let cache = Path::new("cache");
        let ck = ModelSource::Checkpoint {
            repo: "Qwen/Qwen3-0.6B".into(),
            revision: "abc".into(),
        };
        assert_eq!(
            output_path(cache, &ck, WeightQuant::Q8_0).unwrap(),
            cache
                .join("converted")
                .join("Qwen--Qwen3-0.6B")
                .join("abc")
                .join("model-q8_0.gguf")
        );
        let bad = ModelSource::Checkpoint {
            repo: "a/b".into(),
            revision: "../x".into(),
        };
        assert!(output_path(cache, &bad, WeightQuant::F16).is_err());
        assert!(output_path(cache, &ModelSource::File("x".into()), WeightQuant::F16).is_err());
    }

    #[test]
    fn directory_keys_track_the_files() {
        let dir = safetensors::tests::temp_dir("key");
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"abc").unwrap();
        std::fs::write(dir.join("tokenizer.json"), "x").unwrap();
        let a = dir_key(&dir).unwrap();
        assert_eq!(a.len(), 16);
        assert_eq!(a, dir_key(&dir).unwrap());
        std::fs::write(dir.join("tokenizer.json"), "yy").unwrap();
        assert_eq!(a, dir_key(&dir).unwrap(), "tokenizer files do not matter");
        std::fs::write(dir.join("model.safetensors"), b"abcd").unwrap();
        assert_ne!(a, dir_key(&dir).unwrap(), "a changed shard does");
        assert!(dir_key(&dir.join("missing")).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn model_names_come_from_the_repository() {
        let rev = "a".repeat(40);
        let dir = Path::new("cache")
            .join("huggingface")
            .join("Qwen--Qwen3-0.6B")
            .join(&rev);
        assert_eq!(model_name(&dir), "Qwen3-0.6B");
        assert_eq!(model_name(Path::new("some/My Model")), "My Model");
    }
}
