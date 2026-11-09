//! The engine's own GGUF reader.
//!
//! candle's `gguf_file::Content::read` fails at header parse on any tensor
//! type candle does not implement (MXFP4, IQ4_NL, I8, the IQ family), so no
//! file holding such a tensor could even be opened. This module reads the
//! header itself:
//!
//! - [`GgufType`] keeps every type id, including ones the engine cannot run;
//!   [`Content::tensor`] and [`TensorInfo::byte_len`] fail only for the tensor
//!   that is asked for, naming it and its type;
//! - a **file set** (`<base>-00001-of-0000N.gguf`, llama.cpp's split layout:
//!   `split.no`, `split.count`, the metadata in part 1) opens as one
//!   [`Content`] whose tensors know the part that holds them, so a model never
//!   has to be merged into one file (and its parts may sit on different
//!   volumes: a part is found next to the file that was named, or in the
//!   directories of [`Content::open_with`]);
//! - [`TensorReader`] reads tensors from the parts (one buffered handle per
//!   part), [`Content::location`] gives positional readers (tiered experts,
//!   row tables) the file and absolute offset of a tensor.
//!
//! Metadata keeps candle's [`Value`] shape. Versions 1 to 3 are read;
//! `general.alignment` is honoured.

mod dtype;
#[cfg(test)]
pub(crate) mod fixtures;
mod value;
#[cfg(test)]
mod write;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::quantized::QTensor;
use candle_core::quantized::ggml_file::qtensor_from_ggml;
use candle_core::{Device, Result, Shape, bail};

pub(crate) use self::dtype::GgufType;
pub(crate) use self::value::{Value, ValueType};
#[cfg(test)]
pub(crate) use self::write::{RawTensor, write, write_raw};

/// Alignment of the data section when the file does not say.
pub(crate) const DEFAULT_ALIGNMENT: u64 = 32;
/// Longest string accepted.
const MAX_STRING_LENGTH: u64 = 1 << 30;
/// Most tensor dimensions accepted.
const MAX_TENSOR_DIMS: u32 = 4;

/// A GGUF format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Version {
    V1,
    V2,
    V3,
}

impl Version {
    /// Bytes of a length prefix.
    fn length_size(self) -> u64 {
        match self {
            Self::V1 => 4,
            Self::V2 | Self::V3 => 8,
        }
    }
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_length<R: Read>(r: &mut R, version: Version) -> Result<u64> {
    match version {
        Version::V1 => read_u32(r).map(u64::from),
        Version::V2 | Version::V3 => read_u64(r),
    }
}

fn remaining_bytes<R: Seek>(r: &mut R, file_size: u64) -> Result<u64> {
    Ok(file_size.saturating_sub(r.stream_position()?))
}

fn read_string<R: Read + Seek>(r: &mut R, version: Version, file_size: u64) -> Result<String> {
    let len = read_length(r, version)?;
    if len > MAX_STRING_LENGTH {
        bail!("gguf: string length {len} exceeds max {MAX_STRING_LENGTH}")
    }
    let remaining = remaining_bytes(r, file_size)?;
    if len > remaining {
        bail!("gguf: string length {len} exceeds remaining file bytes {remaining}")
    }
    let mut v = vec![0u8; len as usize];
    r.read_exact(&mut v)?;
    // Strings should not be NUL terminated but some files are.
    while v.last() == Some(&0) {
        v.pop();
    }
    Ok(String::from_utf8_lossy(&v).into_owned())
}

#[cfg(test)]
fn write_string<W: std::io::Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

/// What a tensor info says about one tensor.
#[derive(Debug, Clone)]
pub(crate) struct TensorInfo {
    /// The tensor type (possibly one the engine cannot run).
    pub dtype: GgufType,
    /// Dimensions, slowest first (the reverse of the file's `ne` order).
    pub shape: Shape,
    /// Offset in the data section of the part that holds the tensor.
    pub offset: u64,
    /// The part (index into [`Content::files`]) that holds the tensor.
    pub file: usize,
}

impl TensorInfo {
    /// Bytes of the tensor's data.
    ///
    /// # Errors
    ///
    /// When the type is unknown or the element count is not a whole number of
    /// blocks.
    pub(crate) fn byte_len(&self) -> Result<usize> {
        let elems = self.shape.elem_count();
        self.dtype.byte_len(elems).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "{} values do not form whole blocks of {}",
                elems,
                self.dtype.name()
            ))
        })
    }
}

/// One file of a set.
#[derive(Debug, Clone)]
pub(crate) struct DataFile {
    /// Where the part lives.
    pub path: PathBuf,
    /// Absolute offset of its data section.
    pub data_offset: u64,
}

/// The index of a GGUF file or file set.
#[derive(Debug)]
pub(crate) struct Content {
    /// Key-value metadata (of part 1 for a set).
    pub metadata: HashMap<String, Value>,
    /// Every tensor of the file or set, by name.
    pub tensor_infos: HashMap<String, TensorInfo>,
    files: Vec<DataFile>,
}

/// One parsed header.
struct Header {
    metadata: HashMap<String, Value>,
    tensors: Vec<(String, TensorInfo)>,
    data_offset: u64,
}

fn read_header(path: &Path, file: usize) -> Result<Header> {
    let f = File::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("cannot open `{}`: {e}", path.display())))?;
    parse_header(&mut BufReader::with_capacity(1 << 20, f), file)
}

fn parse_header<R: Read + Seek>(r: &mut R, file: usize) -> Result<Header> {
    let file_size = r.seek(SeekFrom::End(0))?;
    r.rewind()?;
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" && &magic != b"FUGG" {
        bail!("unknown magic {magic:02x?}");
    }
    let version = match read_u32(r)? {
        1 => Version::V1,
        2 => Version::V2,
        3 => Version::V3,
        v => bail!("gguf: unsupported version {v}"),
    };
    let n_tensors = read_length(r, version)?;
    let n_kv = read_length(r, version)?;
    if n_tensors > value::MAX_ARRAY_ELEMENTS || n_kv > value::MAX_ARRAY_ELEMENTS {
        bail!("gguf: header counts {n_tensors} tensors and {n_kv} entries exceed the limits");
    }
    let prefix = version.length_size();
    let needed = n_kv
        .saturating_mul(prefix + 4 + 1)
        .saturating_add(n_tensors.saturating_mul(prefix + 4 + 4 + 8));
    let remaining = remaining_bytes(r, file_size)?;
    if needed > remaining {
        bail!(
            "gguf: header declares {n_tensors} tensors and {n_kv} metadata entries, needs at least {needed} bytes, only {remaining} remaining"
        )
    }
    let mut metadata = HashMap::new();
    for _ in 0..n_kv {
        let key = read_string(r, version, file_size)?;
        let ty = ValueType::read_id(r)?;
        metadata.insert(key, Value::read(r, ty, version, 0, file_size)?);
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_string(r, version, file_size)?;
        let n_dims = read_u32(r)?;
        if n_dims > MAX_TENSOR_DIMS {
            bail!("gguf: tensor '{name}' has {n_dims} dimensions, max is {MAX_TENSOR_DIMS}")
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(match version {
                Version::V1 => read_u32(r)? as usize,
                Version::V2 | Version::V3 => read_u64(r)? as usize,
            });
        }
        dims.reverse();
        let dtype = GgufType::from_id(read_u32(r)?);
        let offset = read_u64(r)?;
        tensors.push((
            name,
            TensorInfo {
                dtype,
                shape: Shape::from(dims),
                offset,
                file,
            },
        ));
    }
    let position = r.stream_position()?;
    let alignment = match metadata.get("general.alignment") {
        Some(Value::U8(v)) => u64::from(*v),
        Some(Value::U16(v)) => u64::from(*v),
        Some(Value::U32(v)) => u64::from(*v),
        Some(Value::I8(v)) if *v > 0 => *v as u64,
        Some(Value::I16(v)) if *v > 0 => *v as u64,
        Some(Value::I32(v)) if *v > 0 => *v as u64,
        _ => DEFAULT_ALIGNMENT,
    };
    Ok(Header {
        metadata,
        tensors,
        data_offset: position.div_ceil(alignment) * alignment,
    })
}

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

    /// The file names of every part of the set, in part order.
    pub(crate) fn part_names(&self) -> Vec<String> {
        (1..=self.count).map(|no| self.part_name(no)).collect()
    }
}

impl Content {
    /// Open the GGUF file at `path`; when it is a part of a split set
    /// (`split.count` > 1 and a split file name) every part is opened, each
    /// looked for next to `path`.
    ///
    /// # Errors
    ///
    /// An unreadable file, a malformed header, a missing part or a set whose
    /// parts disagree. (The engine opens through [`Content::open_with`]; this
    /// is the tests' shorthand.)
    #[cfg(test)]
    pub(crate) fn open(path: &Path) -> Result<Self> {
        Self::open_with(path, &[])
    }

    /// [`Content::open`] that also looks for a missing part in `extra_dirs`.
    pub(crate) fn open_with(path: &Path, extra_dirs: &[PathBuf]) -> Result<Self> {
        let first = read_header(path, 0)?;
        let count = first
            .metadata
            .get("split.count")
            .and_then(|v| v.to_u64().ok())
            .unwrap_or(1);
        let split = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(SplitName::parse);
        let (Some(split), true) = (split, count > 1) else {
            return Ok(Self::single(path, first));
        };
        if u64::from(split.count) != count {
            bail!(
                "`{}` says it is part {} of {} but its split.count is {count}",
                path.display(),
                split.no,
                split.count
            );
        }
        let dir = path.parent().unwrap_or(Path::new("."));
        let mut files = Vec::new();
        let mut tensors: Vec<(String, TensorInfo)> = Vec::new();
        let mut meta: Option<HashMap<String, Value>> = None;
        for no in 1..=split.count {
            let name = split.part_name(no);
            let part = std::iter::once(dir)
                .chain(extra_dirs.iter().map(PathBuf::as_path))
                .map(|d| d.join(&name))
                .find(|p| p.is_file())
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "part {no} of {} of the split GGUF `{}` is missing: `{name}`",
                        split.count,
                        path.display()
                    ))
                })?;
            let header = read_header(&part, files.len())?;
            let declared = header
                .metadata
                .get("split.no")
                .and_then(|v| v.to_u64().ok());
            if declared.is_some_and(|d| d != u64::from(no - 1)) {
                bail!(
                    "`{}` is named part {no} but its split.no is {}",
                    part.display(),
                    declared.unwrap_or(0)
                );
            }
            files.push(DataFile {
                path: part,
                data_offset: header.data_offset,
            });
            tensors.extend(header.tensors);
            if no == 1 {
                meta = Some(header.metadata);
            }
        }
        let metadata = meta.ok_or_else(|| candle_core::Error::Msg("empty set".into()))?;
        let total = metadata
            .get("split.tensors.count")
            .and_then(|v| v.to_u64().ok());
        if total.is_some_and(|t| t != tensors.len() as u64) {
            bail!(
                "the split GGUF `{}` declares {} tensors but its parts hold {}",
                path.display(),
                total.unwrap_or(0),
                tensors.len()
            );
        }
        let mut tensor_infos = HashMap::with_capacity(tensors.len());
        for (name, info) in tensors {
            if tensor_infos.insert(name.clone(), info).is_some() {
                bail!(
                    "tensor `{name}` is in two parts of the split GGUF `{}`",
                    path.display()
                );
            }
        }
        Ok(Self {
            metadata,
            tensor_infos,
            files,
        })
    }

    fn single(path: &Path, header: Header) -> Self {
        Self {
            metadata: header.metadata,
            tensor_infos: header.tensors.into_iter().collect(),
            files: vec![DataFile {
                path: path.to_path_buf(),
                data_offset: header.data_offset,
            }],
        }
    }

    /// The index of GGUF bytes read from `reader` (no file behind it).
    #[cfg(test)]
    pub(crate) fn from_reader<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let header = parse_header(reader, 0)?;
        Ok(Self::single(Path::new(""), header))
    }

    /// The files of the set, in part order.
    pub(crate) fn files(&self) -> &[DataFile] {
        &self.files
    }

    /// The info of tensor `name`.
    ///
    /// # Errors
    ///
    /// When the file has no such tensor.
    pub(crate) fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor_infos
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("cannot find tensor info for {name}")))
    }

    /// The file and the absolute offset of the first byte of a tensor.
    pub(crate) fn location(&self, info: &TensorInfo) -> (&Path, u64) {
        let f = &self.files[info.file];
        (&f.path, f.data_offset + info.offset)
    }

    /// A reader for tensor data (opens each part lazily).
    pub(crate) fn reader(&self) -> TensorReader<'_> {
        TensorReader {
            content: self,
            handles: (0..self.files.len()).map(|_| None).collect(),
        }
    }

    /// The type of tensor `name` when candle can run it.
    ///
    /// # Errors
    ///
    /// A missing tensor, or a type the engine cannot run (named in the error).
    pub(crate) fn ggml_type(&self, name: &str) -> Result<candle_core::quantized::GgmlDType> {
        let info = self.info(name)?;
        info.dtype.ggml().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "tensor `{name}` has type {}, which this engine cannot run",
                info.dtype.name()
            ))
        })
    }
}

/// Reads tensor data from the files of a [`Content`], one buffered handle per
/// part opened on first use.
pub(crate) struct TensorReader<'a> {
    content: &'a Content,
    handles: Vec<Option<BufReader<File>>>,
}

impl TensorReader<'_> {
    fn handle(&mut self, file: usize) -> Result<&mut BufReader<File>> {
        let slot = &mut self.handles[file];
        if slot.is_none() {
            let path = &self.content.files[file].path;
            let f = File::open(path).map_err(|e| {
                candle_core::Error::Msg(format!("cannot open `{}`: {e}", path.display()))
            })?;
            *slot = Some(BufReader::new(f));
        }
        slot.as_mut()
            .ok_or_else(|| candle_core::Error::Msg("no handle".into()))
    }

    /// The raw bytes of tensor `name`.
    ///
    /// # Errors
    ///
    /// A missing tensor, an unknown type, or data that runs past the file.
    pub(crate) fn bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self.content.info(name)?;
        let len = info.byte_len().map_err(|e| {
            candle_core::Error::Msg(format!("tensor `{name}` ({}): {e}", info.dtype.name()))
        })?;
        let (_, start) = self.content.location(info);
        let r = self.handle(info.file)?;
        let file_size = r.seek(SeekFrom::End(0))?;
        let remaining = file_size.saturating_sub(start);
        if len as u64 > remaining {
            bail!(
                "tensor `{name}` needs {len} bytes at offset {start}, only {remaining} remaining in file"
            )
        }
        let mut raw = vec![0u8; len];
        r.seek(SeekFrom::Start(start))?;
        r.read_exact(&mut raw)?;
        Ok(raw)
    }

    /// Fill `buf` from `offset` (absolute) of part `file`.
    ///
    /// # Errors
    ///
    /// An I/O error, or `buf` running past the end of the part.
    pub(crate) fn read_at(&mut self, file: usize, offset: u64, buf: &mut [u8]) -> Result<()> {
        let r = self.handle(file)?;
        r.seek(SeekFrom::Start(offset))?;
        r.read_exact(buf)?;
        Ok(())
    }

    /// Tensor `name` as a candle tensor.
    ///
    /// # Errors
    ///
    /// As [`TensorReader::bytes`]; also a type candle cannot hold.
    pub(crate) fn tensor(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        let dtype = self.content.ggml_type(name)?;
        let dims = self.content.info(name)?.shape.dims().to_vec();
        let raw = self.bytes(name)?;
        qtensor_from_ggml(dtype, &raw, dims, device)
    }
}

/// A candle tensor of `dtype` and shape `dims` from its file bytes.
///
/// # Errors
///
/// When `bytes` is not the size of `dims` in `dtype`.
pub(crate) fn qtensor_from_bytes(
    dtype: candle_core::quantized::GgmlDType,
    dims: &[usize],
    bytes: &[u8],
    device: &Device,
) -> Result<QTensor> {
    qtensor_from_ggml(dtype, bytes, dims.to_vec(), device)
}

#[cfg(test)]
mod tests;
