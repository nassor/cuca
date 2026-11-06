//! Safetensors shards read with positional file reads.
//!
//! The crate forbids `unsafe`, so the shards are not memory-mapped
//! (`MmapedSafetensors::multi` is an `unsafe fn` because a file written while
//! mapped is undefined behaviour). [`Shards`] parses each shard's header (an
//! 8-byte little-endian length and a JSON object of `dtype`, `shape` and
//! `data_offsets` per tensor) and reads tensor bytes on demand with
//! `read_exact_at`, in row blocks: converting a tensor never holds more than
//! one block of it in memory.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::CucaError;
use crate::provider::local::model::tier::read_exact_at;

/// Largest header accepted (the format's own convention is 100 MB).
const MAX_HEADER: u64 = 100_000_000;

/// A stored element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StDtype {
    F32,
    F16,
    Bf16,
    /// Any other safetensors dtype, by name; not convertible.
    Other(&'static str),
}

impl StDtype {
    fn parse(s: &str) -> Self {
        match s {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F64" => Self::Other("F64"),
            "I64" => Self::Other("I64"),
            "I32" => Self::Other("I32"),
            "I16" => Self::Other("I16"),
            "I8" => Self::Other("I8"),
            "U8" => Self::Other("U8"),
            "BOOL" => Self::Other("BOOL"),
            "F8_E4M3" => Self::Other("F8_E4M3"),
            "F8_E5M2" => Self::Other("F8_E5M2"),
            _ => Self::Other("unknown"),
        }
    }

    /// Bytes per element, `None` for dtypes that cannot be read as `f32`.
    fn size(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::Bf16 => Some(2),
            Self::Other(_) => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Other(n) => n,
        }
    }
}

/// Where one tensor lives.
#[derive(Debug, Clone)]
pub(crate) struct TensorMeta {
    shard: usize,
    pub(crate) dtype: StDtype,
    pub(crate) shape: Vec<usize>,
    /// Absolute byte offset of the tensor's data in its shard.
    start: u64,
    /// Data length in bytes.
    len: u64,
}

impl TensorMeta {
    /// Number of elements.
    pub(crate) fn elems(&self) -> usize {
        self.shape.iter().product()
    }

    /// Elements of one row: the last dimension of a matrix or stack, 1 for a
    /// vector (whose rows are its elements).
    pub(crate) fn row_len(&self) -> usize {
        if self.shape.len() <= 1 {
            1
        } else {
            self.shape.last().copied().unwrap_or(1)
        }
    }

    /// Rows: every dimension but the last flattened (a `[E, R, C]` stack has
    /// `E * R` rows of `C`), the element count for a vector.
    pub(crate) fn rows(&self) -> usize {
        self.elems().checked_div(self.row_len()).unwrap_or(0)
    }
}

/// The tensors of one or more safetensors files.
pub(crate) struct Shards {
    files: Vec<File>,
    paths: Vec<PathBuf>,
    index: HashMap<String, TensorMeta>,
}

fn bad(path: &Path, why: impl std::fmt::Display) -> CucaError {
    CucaError::Config(format!("safetensors file `{}`: {why}", path.display()))
}

impl Shards {
    /// Open every shard and index its tensors. A tensor name in two shards is
    /// an error.
    ///
    /// The shards must not be modified while a [`Shards`] over them is alive:
    /// a read would see a mix of old and new bytes (never a memory-safety
    /// problem, since nothing is mapped).
    pub(crate) fn open(paths: &[PathBuf]) -> Result<Self, CucaError> {
        let mut files = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for (shard, path) in paths.iter().enumerate() {
            let file = File::open(path)?;
            let size = file.metadata()?.len();
            let mut len = [0u8; 8];
            read_exact_at(&file, 0, &mut len).map_err(|_| bad(path, "shorter than its header"))?;
            let header_len = u64::from_le_bytes(len);
            if header_len > MAX_HEADER || 8 + header_len > size {
                return Err(bad(path, format!("header length {header_len} is invalid")));
            }
            let mut header = vec![0u8; header_len as usize];
            read_exact_at(&file, 8, &mut header)?;
            let json: Value = serde_json::from_slice(&header)
                .map_err(|e| bad(path, format!("header is not JSON: {e}")))?;
            let Value::Object(map) = json else {
                return Err(bad(path, "header is not a JSON object"));
            };
            let base = 8 + header_len;
            for (name, entry) in map {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = entry
                    .get("dtype")
                    .and_then(Value::as_str)
                    .map(StDtype::parse)
                    .ok_or_else(|| bad(path, format!("tensor `{name}` has no dtype")))?;
                let shape = entry
                    .get("shape")
                    .and_then(Value::as_array)
                    .and_then(|a| a.iter().map(|d| d.as_u64().map(|d| d as usize)).collect())
                    .ok_or_else(|| bad(path, format!("tensor `{name}` has no shape")))?;
                let offsets = entry
                    .get("data_offsets")
                    .and_then(Value::as_array)
                    .and_then(|a| Some((a.first()?.as_u64()?, a.get(1)?.as_u64()?)))
                    .ok_or_else(|| bad(path, format!("tensor `{name}` has no data_offsets")))?;
                if offsets.0 > offsets.1 || base + offsets.1 > size {
                    return Err(bad(path, format!("tensor `{name}` lies outside the file")));
                }
                let meta = TensorMeta {
                    shard,
                    dtype,
                    shape,
                    start: base + offsets.0,
                    len: offsets.1 - offsets.0,
                };
                if let Some(size) = dtype.size()
                    && meta.elems() as u64 * size as u64 != meta.len
                {
                    return Err(bad(
                        path,
                        format!("tensor `{name}`: shape and data size disagree"),
                    ));
                }
                if index.insert(name.clone(), meta).is_some() {
                    return Err(bad(path, format!("tensor `{name}` appears in two shards")));
                }
            }
            files.push(file);
        }
        Ok(Self {
            files,
            paths: paths.to_vec(),
            index,
        })
    }

    /// Every tensor name, sorted.
    pub(crate) fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.index.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    pub(crate) fn tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.index.get(name)
    }

    fn require(&self, name: &str) -> Result<&TensorMeta, CucaError> {
        self.tensor(name)
            .ok_or_else(|| CucaError::Config(format!("checkpoint has no tensor `{name}`")))
    }

    /// Read `n_rows` rows starting at row `first` of tensor `name` into `out`
    /// as `f32` (`F32`, `F16` and `BF16` sources); `out` is cleared first.
    /// Any other dtype is a [`CucaError::Config`] naming the tensor and the
    /// dtype.
    pub(crate) fn read_rows(
        &self,
        name: &str,
        first: usize,
        n_rows: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), CucaError> {
        let meta = self.require(name)?;
        let Some(size) = meta.dtype.size() else {
            return Err(CucaError::Config(format!(
                "tensor `{name}` has dtype {}, which the converter cannot read",
                meta.dtype.name()
            )));
        };
        if first + n_rows > meta.rows() {
            return Err(CucaError::Config(format!(
                "tensor `{name}`: rows {first}..{} exceed its {} rows",
                first + n_rows,
                meta.rows()
            )));
        }
        let row_len = meta.row_len();
        let n = n_rows * row_len;
        let mut bytes = vec![0u8; n * size];
        let offset = meta.start + (first * row_len * size) as u64;
        read_exact_at(&self.files[meta.shard], offset, &mut bytes)
            .map_err(|e| bad(&self.paths[meta.shard], format!("reading `{name}`: {e}")))?;
        out.clear();
        out.reserve(n);
        match meta.dtype {
            StDtype::F32 => out.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b)),
            ),
            StDtype::F16 => out.extend(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| half::f16::from_le_bytes(*b).to_f32()),
            ),
            StDtype::Bf16 => out.extend(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| half::bf16::from_le_bytes(*b).to_f32()),
            ),
            StDtype::Other(_) => {}
        }
        Ok(())
    }

    /// The whole tensor as `f32`.
    #[cfg(test)]
    pub(crate) fn f32(&self, name: &str) -> Result<Vec<f32>, CucaError> {
        let rows = self.require(name)?.rows();
        let mut out = Vec::new();
        self.read_rows(name, 0, rows, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Write a safetensors file holding `tensors` (name, dtype, shape, raw
    /// bytes) at `path`.
    pub(crate) fn write_file(path: &Path, tensors: &[(&str, &str, &[usize], Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, dtype, shape, bytes) in tensors {
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes.len()],
                }),
            );
            offset += bytes.len();
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        for (_, _, _, bytes) in tensors {
            file.extend_from_slice(bytes);
        }
        std::fs::write(path, file).unwrap();
    }

    pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    pub(crate) fn temp_dir(tag: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "cuca-convert-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reads_dtypes_and_row_blocks_across_shards() {
        let dir = temp_dir("st");
        let a = dir.join("a.safetensors");
        let b = dir.join("b.safetensors");
        let bf: Vec<u8> = [1.5f32, -2.0, 0.25, 8.0]
            .iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
            .collect();
        let f16: Vec<u8> = [0.5f32, 3.0]
            .iter()
            .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
            .collect();
        write_file(
            &a,
            &[
                ("w", "F32", &[3, 2], f32_bytes(&[1., 2., 3., 4., 5., 6.])),
                ("n", "BF16", &[4], bf),
            ],
        );
        write_file(
            &b,
            &[("h", "F16", &[2, 1], f16), ("ids", "I64", &[1], vec![0; 8])],
        );
        let shards = Shards::open(&[a, b]).unwrap();
        assert_eq!(shards.names(), ["h", "ids", "n", "w"]);
        assert_eq!(shards.f32("w").unwrap(), [1., 2., 3., 4., 5., 6.]);
        let mut rows = Vec::new();
        shards.read_rows("w", 1, 2, &mut rows).unwrap();
        assert_eq!(rows, [3., 4., 5., 6.]);
        assert_eq!(shards.f32("n").unwrap(), [1.5, -2.0, 0.25, 8.0]);
        assert_eq!(shards.f32("h").unwrap(), [0.5, 3.0]);
        let err = shards.f32("ids").unwrap_err().to_string();
        assert!(err.contains("`ids`") && err.contains("I64"), "{err}");
        assert!(shards.read_rows("w", 2, 2, &mut rows).is_err());
        assert!(shards.f32("missing").is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_corrupt_files() {
        let dir = temp_dir("st-bad");
        let p = dir.join("x.safetensors");
        std::fs::write(&p, [1u8, 2, 3]).unwrap();
        assert!(Shards::open(std::slice::from_ref(&p)).is_err());
        let mut huge = u64::MAX.to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        std::fs::write(&p, huge).unwrap();
        assert!(Shards::open(std::slice::from_ref(&p)).is_err());
        write_file(&p, &[("w", "F32", &[2], f32_bytes(&[1.0]))]);
        // Shape says 2 elements, data holds 1.
        assert!(Shards::open(&[p]).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
