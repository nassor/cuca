//! GGUF metadata values: the typed union of the format, read and written
//! little-endian.

#[cfg(test)]
use std::collections::HashSet;
#[cfg(test)]
use std::io::Write;
use std::io::{Read, Seek};

use candle_core::Result;
use candle_core::bail;

use super::{Version, read_length, read_string, remaining_bytes};

/// Deepest nesting of arrays accepted (bounds the recursion of a crafted file).
const MAX_DEPTH: usize = 64;
/// Most elements of one array accepted.
pub(super) const MAX_ARRAY_ELEMENTS: u64 = 1 << 30;

/// The type tag of a metadata value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    String,
    Array,
}

impl ValueType {
    fn from_id(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            v => bail!("unrecognized value-type {v:#08x}"),
        })
    }

    pub(super) fn read_id<R: Read>(reader: &mut R) -> Result<Self> {
        let mut b = [0u8; 4];
        reader.read_exact(&mut b)?;
        Self::from_id(u32::from_le_bytes(b))
    }

    #[cfg(test)]
    pub(super) fn id(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }

    /// Fewest bytes one value of this type occupies, to reject an array
    /// length that exceeds the file before allocating.
    fn min_disk_size(self, version: Version) -> u64 {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String => version.length_size(),
            Self::Array => 4 + version.length_size(),
        }
    }
}

/// A metadata value.
#[derive(Debug, Clone)]
pub(crate) enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

#[cfg(test)]
macro_rules! getter {
    ($name:ident, $variant:ident, $ty:ty) => {
        pub(crate) fn $name(&self) -> Result<$ty> {
            match self {
                Self::$variant(v) => Ok(*v),
                v => bail!("not a {} {v:?}", stringify!($ty)),
            }
        }
    };
}

impl Value {
    #[cfg(test)]
    pub(crate) fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    #[cfg(test)]
    getter!(to_f32, F32, f32);

    #[cfg(test)]
    pub(crate) fn to_vec(&self) -> Result<&Vec<Value>> {
        match self {
            Self::Array(v) => Ok(v),
            v => bail!("not a vec {v:?}"),
        }
    }

    pub(crate) fn to_u32(&self) -> Result<u32> {
        match self {
            Self::U32(v) => Ok(*v),
            v => bail!("not a u32 {v:?}"),
        }
    }

    pub(crate) fn to_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            v => bail!("not a bool {v:?}"),
        }
    }

    /// An unsigned integer; narrower unsigned types and booleans widen.
    pub(crate) fn to_u64(&self) -> Result<u64> {
        match self {
            Self::U64(v) => Ok(*v),
            Self::U8(v) => Ok(u64::from(*v)),
            Self::U16(v) => Ok(u64::from(*v)),
            Self::U32(v) => Ok(u64::from(*v)),
            Self::Bool(v) => Ok(u64::from(*v)),
            v => bail!("not a u64 or upcastable to u64 {v:?}"),
        }
    }

    pub(crate) fn to_string(&self) -> Result<&String> {
        match self {
            Self::String(v) => Ok(v),
            v => bail!("not a string {v:?}"),
        }
    }

    pub(super) fn read<R: Read + Seek>(
        reader: &mut R,
        ty: ValueType,
        version: Version,
        depth: usize,
        file_size: u64,
    ) -> Result<Self> {
        if depth > MAX_DEPTH {
            bail!("gguf: value nesting depth exceeds max {MAX_DEPTH}")
        }
        fn arr<const N: usize, R: Read>(r: &mut R) -> Result<[u8; N]> {
            let mut b = [0u8; N];
            r.read_exact(&mut b)?;
            Ok(b)
        }
        Ok(match ty {
            ValueType::U8 => Self::U8(arr::<1, _>(reader)?[0]),
            ValueType::I8 => Self::I8(i8::from_le_bytes(arr(reader)?)),
            ValueType::U16 => Self::U16(u16::from_le_bytes(arr(reader)?)),
            ValueType::I16 => Self::I16(i16::from_le_bytes(arr(reader)?)),
            ValueType::U32 => Self::U32(u32::from_le_bytes(arr(reader)?)),
            ValueType::I32 => Self::I32(i32::from_le_bytes(arr(reader)?)),
            ValueType::U64 => Self::U64(u64::from_le_bytes(arr(reader)?)),
            ValueType::I64 => Self::I64(i64::from_le_bytes(arr(reader)?)),
            ValueType::F32 => Self::F32(f32::from_le_bytes(arr(reader)?)),
            ValueType::F64 => Self::F64(f64::from_le_bytes(arr(reader)?)),
            ValueType::Bool => match arr::<1, _>(reader)?[0] {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => bail!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader, version, file_size)?),
            ValueType::Array => {
                let element = ValueType::read_id(reader)?;
                let len = read_length(reader, version)?;
                if len > MAX_ARRAY_ELEMENTS {
                    bail!("gguf: array length {len} exceeds max {MAX_ARRAY_ELEMENTS}")
                }
                let needed = len.saturating_mul(element.min_disk_size(version));
                let remaining = remaining_bytes(reader, file_size)?;
                if needed > remaining {
                    bail!(
                        "gguf: array of {len} elements needs at least {needed} bytes, only {remaining} remaining"
                    )
                }
                let mut items = Vec::new();
                for _ in 0..len {
                    items.push(Self::read(reader, element, version, depth + 1, file_size)?);
                }
                Self::Array(items)
            }
        })
    }

    #[cfg(test)]
    pub(super) fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        match self {
            Self::U8(v) => w.write_all(&[*v])?,
            Self::I8(v) => w.write_all(&v.to_le_bytes())?,
            Self::U16(v) => w.write_all(&v.to_le_bytes())?,
            Self::I16(v) => w.write_all(&v.to_le_bytes())?,
            Self::U32(v) => w.write_all(&v.to_le_bytes())?,
            Self::I32(v) => w.write_all(&v.to_le_bytes())?,
            Self::U64(v) => w.write_all(&v.to_le_bytes())?,
            Self::I64(v) => w.write_all(&v.to_le_bytes())?,
            Self::F32(v) => w.write_all(&v.to_le_bytes())?,
            Self::F64(v) => w.write_all(&v.to_le_bytes())?,
            Self::Bool(v) => w.write_all(&[u8::from(*v)])?,
            Self::String(v) => super::write_string(w, v)?,
            Self::Array(v) => {
                let types: HashSet<_> = v.iter().map(Self::value_type).collect();
                if types.len() > 1 {
                    bail!("multiple value-types in the same array {types:?}")
                }
                let element = types.into_iter().next().unwrap_or(ValueType::U32);
                w.write_all(&element.id().to_le_bytes())?;
                w.write_all(&(v.len() as u64).to_le_bytes())?;
                for item in v {
                    item.write(w)?;
                }
            }
        }
        Ok(())
    }
}
