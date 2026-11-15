//! The tensor types a GGUF file can name.
//!
//! candle's [`GgmlDType`] is a closed set of the types candle implements, and
//! its GGUF reader fails on any other id. The engine reads the type ids of the
//! file itself: [`GgufType`] adds the types the engine knows without candle
//! (the I-quants IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL
//! and IQ4_XS, the ternary types TQ1_0 and TQ2_0, MXFP4, NVFP4 and I8) and
//! keeps an unrecognised id as [`GgufType::Unknown`], so a header that
//! mentions a type the engine cannot run still opens and the error names the
//! tensor and the type only when that tensor is requested.
//!
//! Ids, block sizes and byte sizes are llama.cpp's (`ggml.h` `ggml_type`,
//! `ggml-common.h` block structs). Block layouts are documented with their
//! decoders in `model/quant/`.

use candle_core::quantized::GgmlDType;

/// Values per super-block of the K-quant and I-quant types (`QK_K`).
const QK_K: usize = 256;

/// ggml type id of IQ2_XXS.
const IQ2_XXS: u32 = 16;
/// ggml type id of IQ2_XS.
const IQ2_XS: u32 = 17;
/// ggml type id of IQ3_XXS.
const IQ3_XXS: u32 = 18;
/// ggml type id of IQ1_S.
const IQ1_S: u32 = 19;
/// ggml type id of IQ4_NL.
const IQ4_NL: u32 = 20;
/// ggml type id of IQ3_S.
const IQ3_S: u32 = 21;
/// ggml type id of IQ2_S.
const IQ2_S: u32 = 22;
/// ggml type id of IQ4_XS.
const IQ4_XS: u32 = 23;
/// ggml type id of I8.
const I8: u32 = 24;
/// ggml type id of IQ1_M.
const IQ1_M: u32 = 29;
/// ggml type id of TQ1_0.
const TQ1_0: u32 = 34;
/// ggml type id of TQ2_0.
const TQ2_0: u32 = 35;
/// ggml type id of MXFP4.
const MXFP4: u32 = 39;
/// ggml type id of NVFP4.
const NVFP4: u32 = 40;

/// A tensor type of a GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GgufType {
    /// A type candle implements (float, legacy and K-quant blocks).
    Ggml(GgmlDType),
    /// OCP MXFP4: 32 E2M1 values per block as 16 bytes (byte `j` holds element
    /// `j` in the low nibble and element `j + 16` in the high one) behind one
    /// E8M0 scale byte, 17 bytes per block.
    Mxfp4,
    /// NVFP4: 64 E2M1 values per block as four 16-value sub-blocks, each with
    /// its own UE4M3 scale byte (the four scale bytes come first), 36 bytes
    /// per block.
    Nvfp4,
    /// IQ4_NL: 32 values as 16 bytes of nibbles indexing a 16-entry table,
    /// behind one f16 scale, 18 bytes per block.
    Iq4Nl,
    /// IQ4_XS: 256 values as 128 bytes of nibbles indexing the IQ4_NL table,
    /// with an f16 scale and a 6-bit scale per 32 values, 136 bytes per
    /// block.
    Iq4Xs,
    /// IQ2_XXS: 256 values from a 256-entry grid of 8 values (2.0625 bits per
    /// value), 66 bytes per block.
    Iq2Xxs,
    /// IQ2_XS: 256 values from a 512-entry grid of 8 values, 74 bytes per
    /// block.
    Iq2Xs,
    /// IQ2_S: 256 values from a 1024-entry grid of 8 values with explicit
    /// sign bytes, 82 bytes per block.
    Iq2S,
    /// IQ3_XXS: 256 values from a 256-entry grid of 4 values (3.0625 bits per
    /// value), 98 bytes per block.
    Iq3Xxs,
    /// IQ3_S: 256 values from a 512-entry grid of 4 values with explicit sign
    /// bytes, 110 bytes per block.
    Iq3S,
    /// IQ1_S: 256 values from a 2048-entry grid of 8 ternary values (1.5625
    /// bits per value), 50 bytes per block.
    Iq1S,
    /// IQ1_M: like IQ1_S with a delta sign per 16 values and the f16 scale
    /// spread over the four 16-bit scale words, 56 bytes per block.
    Iq1M,
    /// TQ1_0: 256 ternary values packed five per byte (base 3) behind an f16
    /// scale, 54 bytes per block.
    Tq1_0,
    /// TQ2_0: 256 ternary values at two bits each behind an f16 scale, 66
    /// bytes per block.
    Tq2_0,
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
            IQ2_XXS => return Self::Iq2Xxs,
            IQ2_XS => return Self::Iq2Xs,
            IQ3_XXS => return Self::Iq3Xxs,
            IQ1_S => return Self::Iq1S,
            IQ4_NL => return Self::Iq4Nl,
            IQ3_S => return Self::Iq3S,
            IQ2_S => return Self::Iq2S,
            IQ4_XS => return Self::Iq4Xs,
            I8 => return Self::I8,
            IQ1_M => return Self::Iq1M,
            TQ1_0 => return Self::Tq1_0,
            TQ2_0 => return Self::Tq2_0,
            MXFP4 => return Self::Mxfp4,
            NVFP4 => return Self::Nvfp4,
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
            Self::Iq2Xxs => IQ2_XXS,
            Self::Iq2Xs => IQ2_XS,
            Self::Iq3Xxs => IQ3_XXS,
            Self::Iq1S => IQ1_S,
            Self::Iq4Nl => IQ4_NL,
            Self::Iq3S => IQ3_S,
            Self::Iq2S => IQ2_S,
            Self::Iq4Xs => IQ4_XS,
            Self::I8 => I8,
            Self::Iq1M => IQ1_M,
            Self::Tq1_0 => TQ1_0,
            Self::Tq2_0 => TQ2_0,
            Self::Mxfp4 => MXFP4,
            Self::Nvfp4 => NVFP4,
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
            Self::Nvfp4 => Some(64),
            Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq2S
            | Self::Iq3Xxs
            | Self::Iq3S
            | Self::Iq1S
            | Self::Iq1M
            | Self::Iq4Xs
            | Self::Tq1_0
            | Self::Tq2_0 => Some(QK_K),
            Self::I8 => Some(1),
            Self::Unknown(_) => None,
        }
    }

    /// Bytes per block, `None` for an unknown type.
    pub(crate) fn type_size(self) -> Option<usize> {
        match self {
            Self::Ggml(d) => Some(d.type_size()),
            Self::Mxfp4 => Some(17),
            Self::Nvfp4 => Some(36),
            Self::Iq4Nl => Some(18),
            Self::Iq4Xs => Some(136),
            Self::Iq2Xxs => Some(66),
            Self::Iq2Xs => Some(74),
            Self::Iq2S => Some(82),
            Self::Iq3Xxs => Some(98),
            Self::Iq3S => Some(110),
            Self::Iq1S => Some(50),
            Self::Iq1M => Some(56),
            Self::Tq1_0 => Some(54),
            Self::Tq2_0 => Some(66),
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
        let name = match self {
            Self::Ggml(d) => return format!("{d:?}"),
            Self::Mxfp4 => "MXFP4",
            Self::Nvfp4 => "NVFP4",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq4Xs => "IQ4_XS",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq2S => "IQ2_S",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq3S => "IQ3_S",
            Self::Iq1S => "IQ1_S",
            Self::Iq1M => "IQ1_M",
            Self::Tq1_0 => "TQ1_0",
            Self::Tq2_0 => "TQ2_0",
            Self::I8 => "I8",
            Self::Unknown(id) => return format!("ggml type {id}"),
        };
        name.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every id the engine reads, with the llama.cpp name.
    const KNOWN: [(u32, &str); 29] = [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (6, "Q5_0"),
        (7, "Q5_1"),
        (8, "Q8_0"),
        (9, "Q8_1"),
        (10, "Q2K"),
        (11, "Q3K"),
        (12, "Q4K"),
        (13, "Q5K"),
        (14, "Q6K"),
        (15, "Q8K"),
        (16, "IQ2_XXS"),
        (17, "IQ2_XS"),
        (18, "IQ3_XXS"),
        (19, "IQ1_S"),
        (20, "IQ4_NL"),
        (21, "IQ3_S"),
        (22, "IQ2_S"),
        (23, "IQ4_XS"),
        (24, "I8"),
        (29, "IQ1_M"),
        (30, "BF16"),
        (34, "TQ1_0"),
        (35, "TQ2_0"),
        (39, "MXFP4"),
        (40, "NVFP4"),
    ];

    #[test]
    fn ids_round_trip_and_unknown_ids_are_kept() {
        for (id, name) in KNOWN {
            let t = GgufType::from_id(id);
            assert!(!matches!(t, GgufType::Unknown(_)), "{id}");
            assert_eq!(t.id(), id);
            assert_eq!(t.name(), name, "{id}");
        }
        // 4 and 5 are the removed Q4_2 and Q4_3, 25 is I16, 31 is a removed
        // repacked type and 41 is Q1_0: none is read.
        for id in [4, 5, 25, 31, 41, 99] {
            assert_eq!(GgufType::from_id(id), GgufType::Unknown(id));
            assert_eq!(GgufType::Unknown(id).id(), id);
            assert_eq!(GgufType::Unknown(id).byte_len(256), None);
        }
    }

    #[test]
    fn block_geometry_matches_llama_cpp() {
        assert_eq!(GgufType::Mxfp4.byte_len(5120), Some(160 * 17));
        assert_eq!(GgufType::Nvfp4.byte_len(128), Some(2 * 36));
        assert_eq!(GgufType::Iq4Nl.byte_len(256), Some(8 * 18));
        assert_eq!(GgufType::Iq4Xs.byte_len(512), Some(2 * 136));
        assert_eq!(GgufType::Iq1M.byte_len(256), Some(56));
        assert_eq!(GgufType::Tq1_0.byte_len(256), Some(54));
        assert_eq!(GgufType::I8.byte_len(7), Some(7));
        assert_eq!(GgufType::Mxfp4.byte_len(33), None);
        assert_eq!(GgufType::Nvfp4.byte_len(32), None);
        assert_eq!(GgufType::Iq2Xxs.byte_len(128), None);
        assert_eq!(GgufType::Ggml(GgmlDType::Q8_0).byte_len(64), Some(2 * 34));
    }
}
