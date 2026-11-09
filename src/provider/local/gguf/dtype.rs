//! The tensor types a GGUF file can name.
//!
//! candle's [`GgmlDType`] is a closed set of the types candle implements, and
//! its GGUF reader fails on any other id. The engine reads the type ids of the
//! file itself: [`GgufType`] adds the types the engine knows without candle
//! (MXFP4, IQ4_NL, I8) and keeps an unrecognised id as [`GgufType::Unknown`],
//! so a header that mentions a type the engine cannot run still opens and the
//! error names the tensor and the type only when that tensor is requested.

use candle_core::quantized::GgmlDType;

/// ggml type id of IQ4_NL.
const IQ4_NL: u32 = 20;
/// ggml type id of I8.
const I8: u32 = 24;
/// ggml type id of MXFP4.
const MXFP4: u32 = 39;

/// A tensor type of a GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GgufType {
    /// A type candle implements (float, legacy and K-quant blocks).
    Ggml(GgmlDType),
    /// OCP MXFP4: 32 E2M1 values per block as 16 bytes (byte `j` holds element
    /// `j` in the low nibble and element `j + 16` in the high one) behind one
    /// E8M0 scale byte, 17 bytes per block.
    Mxfp4,
    /// IQ4_NL: 32 values as 16 bytes of nibbles indexing a 16-entry table,
    /// behind one f16 scale, 18 bytes per block.
    Iq4Nl,
    /// One signed byte per value.
    I8,
    /// A type id this engine does not know.
    Unknown(u32),
}

impl From<GgmlDType> for GgufType {
    fn from(dtype: GgmlDType) -> Self {
        Self::Ggml(dtype)
    }
}

impl GgufType {
    /// The type named by the id of a GGUF tensor info.
    pub(crate) fn from_id(id: u32) -> Self {
        let ggml = match id {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            30 => GgmlDType::BF16,
            IQ4_NL => return Self::Iq4Nl,
            I8 => return Self::I8,
            MXFP4 => return Self::Mxfp4,
            _ => return Self::Unknown(id),
        };
        Self::Ggml(ggml)
    }

    /// The id written to a GGUF tensor info.
    #[cfg(test)]
    pub(crate) fn id(self) -> u32 {
        match self {
            Self::Ggml(d) => match d {
                GgmlDType::F32 => 0,
                GgmlDType::F16 => 1,
                GgmlDType::Q4_0 => 2,
                GgmlDType::Q4_1 => 3,
                GgmlDType::Q5_0 => 6,
                GgmlDType::Q5_1 => 7,
                GgmlDType::Q8_0 => 8,
                GgmlDType::Q8_1 => 9,
                GgmlDType::Q2K => 10,
                GgmlDType::Q3K => 11,
                GgmlDType::Q4K => 12,
                GgmlDType::Q5K => 13,
                GgmlDType::Q6K => 14,
                GgmlDType::Q8K => 15,
                GgmlDType::BF16 => 30,
            },
            Self::Iq4Nl => IQ4_NL,
            Self::I8 => I8,
            Self::Mxfp4 => MXFP4,
            Self::Unknown(id) => id,
        }
    }

    /// The candle type, when candle implements it.
    pub(crate) fn ggml(self) -> Option<GgmlDType> {
        match self {
            Self::Ggml(d) => Some(d),
            _ => None,
        }
    }

    /// Values per block, `None` for an unknown type.
    pub(crate) fn block_size(self) -> Option<usize> {
        match self {
            Self::Ggml(d) => Some(d.block_size()),
            Self::Mxfp4 | Self::Iq4Nl => Some(32),
            Self::I8 => Some(1),
            Self::Unknown(_) => None,
        }
    }

    /// Bytes per block, `None` for an unknown type.
    pub(crate) fn type_size(self) -> Option<usize> {
        match self {
            Self::Ggml(d) => Some(d.type_size()),
            Self::Mxfp4 => Some(17),
            Self::Iq4Nl => Some(18),
            Self::I8 => Some(1),
            Self::Unknown(_) => None,
        }
    }

    /// Bytes of `elems` values, `None` for an unknown type or a count that is
    /// not a whole number of blocks.
    pub(crate) fn byte_len(self, elems: usize) -> Option<usize> {
        let block = self.block_size()?;
        elems
            .is_multiple_of(block)
            .then(|| elems / block * self.type_size().unwrap_or(0))
    }

    /// The name used in error messages.
    pub(crate) fn name(self) -> String {
        match self {
            Self::Ggml(d) => format!("{d:?}"),
            Self::Mxfp4 => "MXFP4".into(),
            Self::Iq4Nl => "IQ4_NL".into(),
            Self::I8 => "I8".into(),
            Self::Unknown(id) => format!("ggml type {id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_unknown_ids_are_kept() {
        for id in [
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 24, 30, 39,
        ] {
            let t = GgufType::from_id(id);
            assert!(!matches!(t, GgufType::Unknown(_)), "{id}");
            assert_eq!(t.id(), id);
        }
        assert_eq!(GgufType::from_id(16), GgufType::Unknown(16));
        assert_eq!(GgufType::Unknown(16).id(), 16);
        assert_eq!(GgufType::Unknown(16).byte_len(32), None);
    }

    #[test]
    fn block_geometry_matches_llama_cpp() {
        assert_eq!(GgufType::Mxfp4.byte_len(5120), Some(160 * 17));
        assert_eq!(GgufType::Iq4Nl.byte_len(256), Some(8 * 18));
        assert_eq!(GgufType::I8.byte_len(7), Some(7));
        assert_eq!(GgufType::Mxfp4.byte_len(33), None);
        assert_eq!(GgufType::Ggml(GgmlDType::Q8_0).byte_len(64), Some(2 * 34));
    }
}
