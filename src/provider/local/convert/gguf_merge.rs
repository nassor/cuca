//! Merging the parts of a llama.cpp split GGUF into one file.
//!
//! `llama-gguf-split` cuts a large model into `<base>-00001-of-0000N.gguf`
//! files. Every part is a GGUF that carries `split.no` (0-based),
//! `split.count` and `split.tensors.count` (the tensors of the whole set); the
//! first part holds the model's metadata (and, in some sets, no tensor at
//! all), the others hold only their tensors. The engine reads one file, so a
//! set is merged once into one cached GGUF, like `llama-gguf-split --merge`:
//!
//! - the metadata of the first part, without the `split.*` keys;
//! - the tensor infos of every part, in part order and, inside a part, in
//!   data order, with offsets recomputed;
//! - the tensor data copied part by part, never more than one chunk in memory.
//!
//! The header is parsed here instead of through `gguf_file::Content::read`
//! because a part may hold a tensor type candle does not know (the IQ family,
//! MXFP4, ...): a merge copies those bytes untouched, and it is the loader
//! that rejects them. A tensor's size is the distance to the next tensor's
//! offset (the last one runs to the end of the file), so the copy includes the
//! alignment padding the source file had.
//!
//! Output goes through the converter's `.partial` + rename + `.lock` pattern
//! ([`super::write_atomically`]).

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file::Value;
use sha2::{Digest, Sha256};

use super::gguf_writer::{ALIGNMENT, GgufWriter, RawTensorSpec};
use crate::error::CucaError;
use crate::provider::local::{DownloadProgress, DownloadProgressCallback};

/// Write buffer of the merged file.
const WRITE_BUFFER: usize = 1 << 20;
/// Bytes copied per read.
const CHUNK: usize = 8 << 20;
/// Bytes between two progress reports inside a tensor.
const REPORT_EVERY: u64 = 64 << 20;
/// Most tensor dimensions GGUF stores.
const MAX_DIMS: u32 = 4;
/// Longest key or string value accepted.
const MAX_STRING: u64 = 1 << 30;
/// The metadata keys of a split part that the merged file drops.
const SPLIT_KEYS: [&str; 3] = ["split.no", "split.count", "split.tensors.count"];

/// A split file name, `<base>-<no>-of-<count>.gguf` with 5-digit numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SplitName {
    /// The name without the split suffix (and without `.gguf`).
    pub(crate) base: String,
    /// The part's number, from 1.
    pub(crate) no: u32,
    /// The number of parts.
    pub(crate) count: u32,
}

impl SplitName {
    /// Parse a file name; `None` when it does not follow the split pattern
    /// (or its numbers are 0 or `no` exceeds `count`).
    pub(crate) fn parse(file_name: &str) -> Option<Self> {
        let stem = file_name.strip_suffix(".gguf")?;
        let (rest, count) = stem.rsplit_once("-of-")?;
        let (base, no) = rest.rsplit_once('-')?;
        let digits = |s: &str| {
            (s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit()))
                .then(|| s.parse::<u32>().ok())
                .flatten()
        };
        let (no, count) = (digits(no)?, digits(count)?);
        (no >= 1 && no <= count && !base.is_empty()).then(|| Self {
            base: base.to_string(),
            no,
            count,
        })
    }

    /// The file name of part `no` (from 1) of the set.
    pub(crate) fn part_name(&self, no: u32) -> String {
        format!("{}-{no:05}-of-{:05}.gguf", self.base, self.count)
    }

    /// The name of the merged file: `<base>.gguf`.
    pub(crate) fn merged_name(&self) -> String {
        format!("{}.gguf", self.base)
    }
}

/// The parts of a split set named by a repository file path
/// (`dir/<base>-00001-of-00003.gguf`), as repository paths in part order, and
/// the repository path of the merged file; `None` for a file that is not a
/// split part.
pub(crate) fn repo_parts(file: &str) -> Option<(Vec<String>, String)> {
    let (dir, name) = match file.rsplit_once('/') {
        Some((dir, name)) => (format!("{dir}/"), name),
        None => (String::new(), file),
    };
    let split = SplitName::parse(name)?;
    let parts = (1..=split.count)
        .map(|no| format!("{dir}{}", split.part_name(no)))
        .collect();
    Some((parts, format!("{dir}{}", split.merged_name())))
}

/// The merge a `ModelSource::File` that names a split part asks for: the parts
/// next to it and the merged file in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LocalMerge {
    pub(crate) parts: Vec<PathBuf>,
    pub(crate) merged: PathBuf,
}

/// Plan the merge of the split set `path` belongs to, or `None` when `path`
/// is not named like a split part. The merged file is
/// `<cache>/merged/<key>/<base>.gguf`; the key covers the directory and every
/// part's name, length and modification time, so a changed part merges again.
///
/// # Errors
///
/// [`CucaError::Io`] naming a part that does not exist.
pub(crate) fn plan_local(cache_dir: &Path, path: &Path) -> Result<Option<LocalMerge>, CucaError> {
    let Some(split) = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(SplitName::parse)
    else {
        return Ok(None);
    };
    let dir = path.parent().unwrap_or(Path::new("."));
    let dir = dir.canonicalize().map_err(|e| CucaError::Io {
        message: format!("cannot resolve the directory of `{}`: {e}", path.display()),
    })?;
    let mut hasher = Sha256::new();
    hasher.update(dir.to_string_lossy().as_bytes());
    let mut parts = Vec::new();
    for no in 1..=split.count {
        let part = dir.join(split.part_name(no));
        let meta = std::fs::metadata(&part).map_err(|e| CucaError::Io {
            message: format!(
                "part {no} of {} of the split GGUF `{}` is missing: `{}`: {e}",
                split.count,
                path.display(),
                part.display()
            ),
        })?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        hasher.update(b"\0");
        hasher.update(split.part_name(no).as_bytes());
        hasher.update(meta.len().to_le_bytes());
        hasher.update(mtime.to_le_bytes());
        parts.push(part);
    }
    let key: String = hasher
        .finalize()
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect();
    let merged = cache_dir.join("merged").join(key).join(split.merged_name());
    Ok(Some(LocalMerge { parts, merged }))
}

/// One tensor of a part.
#[derive(Debug)]
struct PartTensor {
    name: String,
    gguf_dims: Vec<u64>,
    ggml_type: u32,
    /// Offset in the data section.
    offset: u64,
    /// Bytes from `offset` to the next tensor (or the end of the file).
    extent: u64,
}

/// What a merge needs from one part's header.
#[derive(Debug)]
struct PartHeader {
    metadata: Vec<(String, Value)>,
    /// Tensors in data order.
    tensors: Vec<PartTensor>,
    /// Absolute offset of the data section.
    data_start: u64,
}

impl PartHeader {
    fn int(&self, key: &str) -> Option<u64> {
        let value = &self.metadata.iter().find(|(k, _)| k == key)?.1;
        match value {
            Value::U8(v) => Some(u64::from(*v)),
            Value::U16(v) => Some(u64::from(*v)),
            Value::U32(v) => Some(u64::from(*v)),
            Value::U64(v) => Some(*v),
            Value::I8(v) => u64::try_from(*v).ok(),
            Value::I16(v) => u64::try_from(*v).ok(),
            Value::I32(v) => u64::try_from(*v).ok(),
            Value::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }
}

/// A bounded little-endian reader over a part file that names the file in
/// every error.
struct Reader<'a> {
    inner: BufReader<File>,
    path: &'a Path,
    len: u64,
}

impl Reader<'_> {
    fn fail(&self, what: impl std::fmt::Display) -> CucaError {
        CucaError::Config(format!(
            "`{}` is not a readable GGUF file: {what}",
            self.path.display()
        ))
    }

    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, CucaError> {
        let mut buf = vec![0u8; n];
        self.inner.read_exact(&mut buf).map_err(|e| CucaError::Io {
            message: format!("`{}`: {e}", self.path.display()),
        })?;
        Ok(buf)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], CucaError> {
        let mut buf = [0u8; N];
        self.inner.read_exact(&mut buf).map_err(|e| CucaError::Io {
            message: format!("`{}`: {e}", self.path.display()),
        })?;
        Ok(buf)
    }

    fn u8(&mut self) -> Result<u8, CucaError> {
        Ok(self.array::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, CucaError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, CucaError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String, CucaError> {
        let n = self.u64()?;
        if n > MAX_STRING || n > self.len {
            return Err(self.fail(format!("a string of {n} bytes")));
        }
        let bytes = self.bytes(n as usize)?;
        String::from_utf8(bytes).map_err(|_| self.fail("a string that is not UTF-8"))
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, CucaError> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(i8::from_le_bytes(self.array()?)),
            2 => Value::U16(u16::from_le_bytes(self.array()?)),
            3 => Value::I16(i16::from_le_bytes(self.array()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.array()?)),
            6 => Value::F32(f32::from_le_bytes(self.array()?)),
            7 => match self.u8()? {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                b => return Err(self.fail(format!("the boolean value {b}"))),
            },
            8 => Value::String(self.string()?),
            9 => {
                if depth > 0 {
                    return Err(self.fail("a nested array"));
                }
                let element = self.u32()?;
                let n = self.u64()?;
                if n > self.len {
                    return Err(self.fail(format!("an array of {n} elements")));
                }
                let mut items = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    items.push(self.value(element, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.array()?)),
            12 => Value::F64(f64::from_le_bytes(self.array()?)),
            other => return Err(self.fail(format!("the value type {other}"))),
        })
    }
}

/// Read the header of one part: metadata, tensor infos and where the data
/// starts.
fn read_header(path: &Path) -> Result<PartHeader, CucaError> {
    let file = File::open(path).map_err(|e| CucaError::Io {
        message: format!("cannot open `{}`: {e}", path.display()),
    })?;
    let len = file
        .metadata()
        .map_err(|e| CucaError::Io {
            message: format!("`{}`: {e}", path.display()),
        })?
        .len();
    let mut r = Reader {
        inner: BufReader::with_capacity(1 << 20, file),
        path,
        len,
    };
    if r.array::<4>()? != *b"GGUF" {
        return Err(r.fail("it does not start with the GGUF magic"));
    }
    // Versions 2 and 3 share one little-endian layout (candle writes 2,
    // llama.cpp 3); the merged file is written as 3.
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(r.fail(format!("GGUF version {version}, only 2 and 3 are merged")));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_tensors > len || n_kv > len {
        return Err(r.fail("counts larger than the file"));
    }
    let mut metadata = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        let value = r.value(ty, 0)?;
        metadata.push((key, value));
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            return Err(r.fail(format!("tensor `{name}` with {n_dims} dimensions")));
        }
        let gguf_dims = (0..n_dims)
            .map(|_| r.u64())
            .collect::<Result<Vec<_>, _>>()?;
        let ggml_type = r.u32()?;
        let offset = r.u64()?;
        tensors.push(PartTensor {
            name,
            gguf_dims,
            ggml_type,
            offset,
            extent: 0,
        });
    }
    let header_end = r.inner.stream_position().map_err(|e| CucaError::Io {
        message: format!("`{}`: {e}", path.display()),
    })?;
    let mut header = PartHeader {
        metadata,
        tensors,
        data_start: 0,
    };
    if let Some(alignment) = header.int("general.alignment")
        && alignment != ALIGNMENT
    {
        return Err(CucaError::Config(format!(
            "`{}` has general.alignment {alignment}; only {ALIGNMENT} is merged",
            path.display()
        )));
    }
    header.data_start = header_end.div_ceil(ALIGNMENT) * ALIGNMENT;
    if header.data_start > len && !header.tensors.is_empty() {
        return Err(r.fail("the file ends inside the header"));
    }
    let data_len = len.saturating_sub(header.data_start);
    header.tensors.sort_by_key(|t| t.offset);
    for i in 0..header.tensors.len() {
        let end = header.tensors.get(i + 1).map_or(data_len, |t| t.offset);
        let t = &mut header.tensors[i];
        if end < t.offset || end > data_len {
            return Err(CucaError::Config(format!(
                "`{}`: tensor `{}` at offset {} runs past the end of the file",
                path.display(),
                t.name,
                t.offset
            )));
        }
        t.extent = end - t.offset;
    }
    Ok(header)
}

/// Check that `headers` are the consecutive parts of one split set.
fn validate(parts: &[PathBuf], headers: &[PartHeader]) -> Result<(), CucaError> {
    let count = parts.len() as u64;
    let mut total = 0u64;
    let expected_total = headers.first().and_then(|h| h.int("split.tensors.count"));
    for (i, (path, header)) in parts.iter().zip(headers).enumerate() {
        let name = path.display();
        let Some(no) = header.int("split.no") else {
            return Err(CucaError::Config(format!(
                "`{name}` has no split.no: it is not a part of a split GGUF"
            )));
        };
        if no != i as u64 {
            return Err(CucaError::Config(format!(
                "`{name}` has split.no {no}, expected {i} for part {} of {count}",
                i + 1
            )));
        }
        match header.int("split.count") {
            Some(c) if c == count => {}
            other => {
                return Err(CucaError::Config(format!(
                    "`{name}` has split.count {}, but {count} parts were given",
                    other.map_or("none".to_string(), |c| c.to_string())
                )));
            }
        }
        if header.int("split.tensors.count") != expected_total {
            return Err(CucaError::Config(format!(
                "`{name}` disagrees with the first part on split.tensors.count"
            )));
        }
        total += header.tensors.len() as u64;
    }
    if let Some(expected) = expected_total
        && expected != total
    {
        return Err(CucaError::Config(format!(
            "the parts hold {total} tensors but split.tensors.count says {expected} (`{}`)",
            parts
                .first()
                .map_or(String::new(), |p| p.display().to_string())
        )));
    }
    Ok(())
}

fn report(progress: Option<&DownloadProgressCallback>, done: u64, total: u64) {
    if let Some(cb) = progress {
        (cb.0)(DownloadProgress {
            downloaded_bytes: done,
            total_bytes: Some(total),
        });
    }
}

/// Merge `parts` (in part order) into the file `out`, atomically.
///
/// `progress` receives the data bytes copied so far in
/// [`DownloadProgress::downloaded_bytes`] and their total in
/// [`DownloadProgress::total_bytes`]. An existing `out` is left alone.
///
/// # Errors
///
/// [`CucaError::Config`] naming the offending file for a part that is not a
/// GGUF, has the wrong `split.no` or `split.count`, a tensor count that
/// disagrees with `split.tensors.count`, a tensor that appears twice, or a
/// tensor that runs past its file's end; [`CucaError::Io`] for a file that
/// cannot be read or written.
pub(crate) fn merge(
    parts: &[PathBuf],
    out: &Path,
    progress: Option<&DownloadProgressCallback>,
) -> Result<(), CucaError> {
    super::write_atomically(out, |partial| merge_to(parts, partial, progress))
}

/// [`merge`] into exactly `partial`, without locking or renaming.
pub(crate) fn merge_to(
    parts: &[PathBuf],
    partial: &Path,
    progress: Option<&DownloadProgressCallback>,
) -> Result<(), CucaError> {
    if parts.is_empty() {
        return Err(CucaError::Config("no GGUF parts to merge".into()));
    }
    let headers = parts
        .iter()
        .map(|p| read_header(p))
        .collect::<Result<Vec<_>, _>>()?;
    validate(parts, &headers)?;

    let metadata: Vec<(String, Value)> = headers[0]
        .metadata
        .iter()
        .filter(|(k, _)| !SPLIT_KEYS.contains(&k.as_str()))
        .cloned()
        .collect();
    let mut seen: HashSet<&str> = HashSet::new();
    let mut specs = Vec::new();
    let mut total = 0u64;
    for (path, header) in parts.iter().zip(&headers) {
        for t in &header.tensors {
            if !seen.insert(t.name.as_str()) {
                return Err(CucaError::Config(format!(
                    "tensor `{}` appears twice in the split set (again in `{}`)",
                    t.name,
                    path.display()
                )));
            }
            specs.push(RawTensorSpec {
                name: t.name.clone(),
                gguf_dims: t.gguf_dims.clone(),
                ggml_type: t.ggml_type,
                bytes: t.extent,
            });
            total += t.extent;
        }
    }

    let file = File::create(partial)?;
    let mut writer = GgufWriter::start_raw(
        BufWriter::with_capacity(WRITE_BUFFER, file),
        &metadata,
        &specs,
    )?;
    report(progress, 0, total);
    let mut done = 0u64;
    let mut reported = 0u64;
    let mut buf = vec![0u8; CHUNK];
    for (path, header) in parts.iter().zip(&headers) {
        let mut file = File::open(path).map_err(|e| CucaError::Io {
            message: format!("cannot open `{}`: {e}", path.display()),
        })?;
        for t in &header.tensors {
            file.seek(SeekFrom::Start(header.data_start + t.offset))
                .map_err(|e| CucaError::Io {
                    message: format!("`{}`: {e}", path.display()),
                })?;
            let mut left = t.extent;
            while left > 0 {
                let n = usize::try_from(left).map_or(CHUNK, |l| l.min(CHUNK));
                file.read_exact(&mut buf[..n]).map_err(|e| CucaError::Io {
                    message: format!("`{}`: tensor `{}`: {e}", path.display(), t.name),
                })?;
                writer.write(&buf[..n])?;
                left -= n as u64;
                done += n as u64;
                if done - reported >= REPORT_EVERY {
                    reported = done;
                    report(progress, done, total);
                }
            }
        }
    }
    let mut sink = writer.finish()?;
    sink.flush()?;
    sink.get_ref().sync_all()?;
    report(progress, total, total);
    Ok(())
}

/// Test fixtures shared with the download and source tests: a tiny model cut
/// into split parts.
#[cfg(test)]
pub(crate) mod fixtures {
    use candle_core::Device;
    use candle_core::quantized::QTensor;
    use candle_core::quantized::gguf_file::{self, Content, Value};

    use super::SplitName;
    use crate::provider::local::model::tiny::TinyModel;
    use std::fs::File;
    use std::io::BufReader;
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
        let mut reader = BufReader::new(File::open(path).unwrap());
        let content = Content::read(&mut reader).unwrap();
        let mut metadata: Vec<_> = content
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        metadata.sort_by(|a, b| a.0.cmp(&b.0));
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort();
        let tensors = names
            .into_iter()
            .map(|n| {
                let t = content.tensor(&mut reader, &n, &Device::Cpu).unwrap();
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
                let mut kv: Vec<(String, Value)> =
                    if p == 0 { metadata.clone() } else { Vec::new() };
                kv.push(("split.no".into(), Value::U16(p as u16)));
                kv.push(("split.count".into(), Value::U16(claimed as u16)));
                kv.push(("split.tensors.count".into(), Value::I32(total)));
                let kv_refs: Vec<(&str, &Value)> =
                    kv.iter().map(|(k, v)| (k.as_str(), v)).collect();
                let ts: Vec<(&str, &QTensor)> = tensors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assign(*i) == p)
                    .map(|(_, (n, t))| (n.as_str(), t))
                    .collect();
                let mut file = File::create(&path).unwrap();
                gguf_file::write(&mut file, &kv_refs, &ts).unwrap();
                path
            })
            .collect()
    }

    pub(crate) fn tensor_bytes(t: &QTensor) -> Vec<u8> {
        t.data().unwrap().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use candle_core::quantized::gguf_file;

    use super::fixtures::{load, split, temp_dir, tensor_bytes};
    use super::*;
    use crate::provider::local::model::tiny::write_tiny_model;

    #[test]
    fn split_names_parse_and_round_trip() {
        let s = SplitName::parse("Mistral-Q4_0-00002-of-00003.gguf").unwrap();
        assert_eq!(
            s,
            SplitName {
                base: "Mistral-Q4_0".into(),
                no: 2,
                count: 3
            }
        );
        assert_eq!(s.part_name(1), "Mistral-Q4_0-00001-of-00003.gguf");
        assert_eq!(s.merged_name(), "Mistral-Q4_0.gguf");
        for bad in [
            "model.gguf",
            "m-1-of-2.gguf",
            "m-00000-of-00002.gguf",
            "m-00003-of-00002.gguf",
            "-00001-of-00002.gguf",
            "m-00001-of-00002.bin",
        ] {
            assert!(SplitName::parse(bad).is_none(), "{bad}");
        }
        let (parts, merged) = repo_parts("UD-Q4/m-00001-of-00002.gguf").unwrap();
        assert_eq!(
            parts,
            ["UD-Q4/m-00001-of-00002.gguf", "UD-Q4/m-00002-of-00002.gguf"]
        );
        assert_eq!(merged, "UD-Q4/m.gguf");
        assert!(repo_parts("plain.gguf").is_none());
    }

    #[test]
    fn three_parts_with_a_metadata_only_first_merge_into_the_original() {
        let model = write_tiny_model();
        let dir = temp_dir("three");
        // Part 1 holds no tensor, part 2 the first half, part 3 the rest.
        let n = load(&model.path).1.len();
        let parts = split(&model, &dir, 3, 3, |i| if i < n / 2 { 1 } else { 2 });
        let out = dir.join("merged").join("tiny.gguf");
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = events.clone();
        let cb = DownloadProgressCallback::new(move |p| sink.lock().unwrap().push(p));
        merge(&parts, &out, Some(&cb)).unwrap();

        let (want_meta, want_tensors) = load(&model.path);
        let (got_meta, got_tensors) = load(&out);
        let strip = |m: Vec<(String, Value)>| -> Vec<(String, Value)> {
            m.into_iter()
                .filter(|(k, _)| !k.starts_with("split.") && k != "general.alignment")
                .collect()
        };
        assert_eq!(
            format!("{:?}", strip(got_meta.clone())),
            format!("{:?}", strip(want_meta))
        );
        assert!(got_meta.iter().all(|(k, _)| !k.starts_with("split.")));
        assert_eq!(got_tensors.len(), want_tensors.len());
        for ((wn, wt), (gn, gt)) in want_tensors.iter().zip(&got_tensors) {
            assert_eq!(wn, gn);
            assert_eq!(wt.shape(), gt.shape(), "{wn}");
            assert_eq!(wt.dtype(), gt.dtype(), "{wn}");
            assert_eq!(tensor_bytes(wt), tensor_bytes(gt), "{wn}");
        }
        let events = events.lock().unwrap();
        let first = events.first().unwrap();
        let last = events.last().unwrap();
        assert_eq!(first.downloaded_bytes, 0);
        assert_eq!(last.downloaded_bytes, last.total_bytes.unwrap());
        assert!(!out.with_extension("gguf.partial").exists());
        // An existing merged file is left alone.
        std::fs::write(&out, b"x").unwrap();
        merge(&parts, &out, None).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"x");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_split_with_tensors_in_every_part_merges_too() {
        let model = write_tiny_model();
        let dir = temp_dir("every");
        let parts = split(&model, &dir, 2, 2, |i| i % 2);
        let out = dir.join("out.gguf");
        merge(&parts, &out, None).unwrap();
        let (_, want) = load(&model.path);
        let (_, got) = load(&out);
        assert_eq!(want.len(), got.len());
        for (w, g) in want.iter().zip(&got) {
            assert_eq!(tensor_bytes(&w.1), tensor_bytes(&g.1), "{}", w.0);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    fn config_error(parts: &[PathBuf], out: &Path) -> String {
        let err = merge(parts, out, None).unwrap_err();
        assert!(matches!(err, CucaError::Config(_)), "{err:?}");
        assert!(!out.exists(), "no merged file after a failure");
        assert!(
            !out.with_extension("gguf.partial").exists(),
            "no partial file after a failure"
        );
        err.to_string()
    }

    #[test]
    fn a_wrong_split_count_is_a_config_error_naming_the_file() {
        let model = write_tiny_model();
        let dir = temp_dir("count");
        let parts = split(&model, &dir, 2, 3, |i| i % 2);
        let msg = config_error(&parts, &dir.join("out.gguf"));
        assert!(msg.contains("split.count 3"), "{msg}");
        assert!(msg.contains("tiny-00001-of-00002.gguf"), "{msg}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn parts_in_the_wrong_order_are_rejected() {
        let model = write_tiny_model();
        let dir = temp_dir("order");
        let mut parts = split(&model, &dir, 2, 2, |i| i % 2);
        parts.reverse();
        let msg = config_error(&parts, &dir.join("out.gguf"));
        assert!(msg.contains("split.no"), "{msg}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_tensor_in_two_parts_is_rejected() {
        let model = write_tiny_model();
        let dir = temp_dir("dup");
        // Tensor 0 lands in both parts.
        let (metadata, tensors) = load(&model.path);
        let total = tensors.len() as i32 + 1;
        let paths: Vec<PathBuf> = (0..2)
            .map(|p| {
                let path = dir.join(format!("tiny-{:05}-of-00002.gguf", p + 1));
                let mut kv = if p == 0 { metadata.clone() } else { Vec::new() };
                kv.push(("split.no".into(), Value::U16(p as u16)));
                kv.push(("split.count".into(), Value::U16(2)));
                kv.push(("split.tensors.count".into(), Value::I32(total)));
                let kv_refs: Vec<_> = kv.iter().map(|(k, v)| (k.as_str(), v)).collect();
                let ts: Vec<_> = tensors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i == 0 || *i % 2 == p)
                    .map(|(_, (n, t))| (n.as_str(), t))
                    .collect();
                gguf_file::write(&mut File::create(&path).unwrap(), &kv_refs, &ts).unwrap();
                path
            })
            .collect();
        let msg = config_error(&paths, &dir.join("out.gguf"));
        assert!(msg.contains("appears twice"), "{msg}");
        assert!(msg.contains("tiny-00002-of-00002.gguf"), "{msg}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_file_that_is_not_a_split_part_is_rejected() {
        let model = write_tiny_model();
        let dir = temp_dir("plain");
        let msg = config_error(std::slice::from_ref(&model.path), &dir.join("out.gguf"));
        assert!(msg.contains("split.no"), "{msg}");
        let junk = dir.join("junk-00001-of-00001.gguf");
        std::fs::write(&junk, b"not a gguf file at all").unwrap();
        let msg = config_error(std::slice::from_ref(&junk), &dir.join("out2.gguf"));
        assert!(msg.contains("junk-00001-of-00001.gguf"), "{msg}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_wrong_tensor_total_is_rejected() {
        let model = write_tiny_model();
        let dir = temp_dir("total");
        let parts = split(&model, &dir, 2, 2, |i| i % 2);
        // Drop every tensor of part 2 from the set by merging part 1 alone is a
        // count error too, but here the file itself lies: part 2 is rewritten
        // without its tensors while still claiming the full total.
        let (_, tensors) = load(&model.path);
        let kv = [
            ("split.no", Value::U16(1)),
            ("split.count", Value::U16(2)),
            ("split.tensors.count", Value::I32(tensors.len() as i32)),
        ];
        let kv_refs: Vec<_> = kv.iter().map(|(k, v)| (*k, v)).collect();
        gguf_file::write(&mut File::create(&parts[1]).unwrap(), &kv_refs, &[]).unwrap();
        let msg = config_error(&parts, &dir.join("out.gguf"));
        assert!(msg.contains("split.tensors.count"), "{msg}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_missing_part_is_an_io_error_naming_it() {
        let model = write_tiny_model();
        let dir = temp_dir("missing");
        let parts = split(&model, &dir, 3, 3, |i| i % 3);
        std::fs::remove_file(&parts[2]).unwrap();
        let cache = dir.join("cache");
        let err = plan_local(&cache, &parts[0]).unwrap_err();
        assert!(matches!(err, CucaError::Io { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("tiny-00003-of-00003.gguf"), "{msg}");
        let err = merge(&parts, &dir.join("out.gguf"), None).unwrap_err();
        assert!(matches!(err, CucaError::Io { .. }), "{err:?}");
        assert!(err.to_string().contains("tiny-00003-of-00003.gguf"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn local_plans_key_the_merged_file_by_the_parts() {
        let model = write_tiny_model();
        let dir = temp_dir("plan");
        let parts = split(&model, &dir, 2, 2, |i| i % 2);
        let cache = dir.join("cache");
        assert!(plan_local(&cache, &model.path).unwrap().is_none());
        // Any part names the same set.
        let a = plan_local(&cache, &parts[0]).unwrap().unwrap();
        let b = plan_local(&cache, &parts[1]).unwrap().unwrap();
        assert_eq!(a, b);
        assert_eq!(a.parts.len(), 2);
        assert_eq!(a.merged.file_name().unwrap(), "tiny.gguf");
        assert!(a.merged.starts_with(cache.join("merged")));
        // A changed part changes the key.
        let mut bytes = std::fs::read(&parts[1]).unwrap();
        bytes.push(0);
        std::fs::write(&parts[1], bytes).unwrap();
        let c = plan_local(&cache, &parts[0]).unwrap().unwrap();
        assert_ne!(a.merged, c.merged);
        let _ = std::fs::remove_dir_all(dir);
    }
}
