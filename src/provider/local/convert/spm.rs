//! A reader for the pieces of a SentencePiece `tokenizer.model`.
//!
//! The file is a protobuf `ModelProto`; only `pieces` (field 1, repeated
//! `SentencePiece { piece = 1: string, score = 2: float, type = 3: enum }`) is
//! needed, so this is a hand-written wire-format walker rather than a protobuf
//! dependency. Every other field (trainer and normalizer specs) is skipped.

use crate::error::CucaError;

/// `SentencePiece.Type`; also the values of `tokenizer.ggml.token_type` for a
/// `llama` vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub(crate) enum PieceType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl PieceType {
    fn from_i32(v: i32) -> Option<Self> {
        Some(match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => return None,
        })
    }
}

/// One vocabulary entry of the model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Piece {
    pub(crate) text: String,
    pub(crate) score: f32,
    pub(crate) kind: PieceType,
}

fn bad(why: &str) -> CucaError {
    CucaError::Config(format!(
        "tokenizer.model is not a SentencePiece model: {why}"
    ))
}

/// A cursor over protobuf wire bytes.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn varint(&mut self) -> Result<u64, CucaError> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let (&byte, rest) = self
                .0
                .split_first()
                .ok_or_else(|| bad("truncated varint"))?;
            self.0 = rest;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(bad("varint longer than 10 bytes"))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], CucaError> {
        if n > self.0.len() {
            return Err(bad("truncated field"));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn len_delimited(&mut self) -> Result<&'a [u8], CucaError> {
        let n = usize::try_from(self.varint()?).map_err(|_| bad("field too long"))?;
        self.take(n)
    }

    /// Skip a field of wire type `wire`.
    fn skip(&mut self, wire: u64) -> Result<(), CucaError> {
        match wire {
            0 => self.varint().map(|_| ()),
            1 => self.take(8).map(|_| ()),
            2 => self.len_delimited().map(|_| ()),
            5 => self.take(4).map(|_| ()),
            _ => Err(bad("unsupported wire type")),
        }
    }
}

fn parse_piece(bytes: &[u8]) -> Result<Piece, CucaError> {
    let mut r = Reader(bytes);
    let mut piece = Piece {
        text: String::new(),
        score: 0.0,
        kind: PieceType::Normal,
    };
    while !r.0.is_empty() {
        let tag = r.varint()?;
        match (tag >> 3, tag & 7) {
            (1, 2) => {
                piece.text = std::str::from_utf8(r.len_delimited()?)
                    .map_err(|_| bad("a piece is not UTF-8"))?
                    .to_string();
            }
            (2, 5) => {
                let b = r.take(4)?;
                piece.score = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            (3, 0) => {
                let v = r.varint()? as i32;
                piece.kind = PieceType::from_i32(v).ok_or_else(|| bad("unknown piece type"))?;
            }
            (_, wire) => r.skip(wire)?,
        }
    }
    Ok(piece)
}

/// The vocabulary of a `tokenizer.model`, in id order.
pub(crate) fn parse_pieces(model: &[u8]) -> Result<Vec<Piece>, CucaError> {
    let mut r = Reader(model);
    let mut pieces = Vec::new();
    while !r.0.is_empty() {
        let tag = r.varint()?;
        match (tag >> 3, tag & 7) {
            (1, 2) => pieces.push(parse_piece(r.len_delimited()?)?),
            (_, wire) => r.skip(wire)?,
        }
    }
    if pieces.is_empty() {
        return Err(bad("it has no pieces"));
    }
    Ok(pieces)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn field(num: u64, wire: u64, body: &[u8], out: &mut Vec<u8>) {
        varint(num << 3 | wire, out);
        if wire == 2 {
            varint(body.len() as u64, out);
        }
        out.extend_from_slice(body);
    }

    /// A hand-built `ModelProto`: `pieces` plus fields the reader must skip.
    pub(crate) fn model(pieces: &[(&str, f32, Option<i32>)]) -> Vec<u8> {
        let mut out = Vec::new();
        // trainer_spec (field 2) comes first in real files.
        field(2, 2, &[0x08, 0x01, 0x12, 0x03, b'a', b'b', b'c'], &mut out);
        for (text, score, kind) in pieces {
            let mut p = Vec::new();
            field(1, 2, text.as_bytes(), &mut p);
            field(2, 5, &score.to_le_bytes(), &mut p);
            if let Some(kind) = kind {
                let mut v = Vec::new();
                varint(*kind as u64, &mut v);
                field(3, 0, &v, &mut p);
            }
            field(1, 2, &p, &mut out);
        }
        // normalizer_spec and a fixed64 field after the pieces.
        field(3, 2, b"identity", &mut out);
        field(9, 1, &[0; 8], &mut out);
        out
    }

    #[test]
    fn decodes_pieces_scores_and_types() {
        let bytes = model(&[
            ("<unk>", 0.0, Some(2)),
            ("<s>", 0.0, Some(3)),
            ("\u{2581}the", -3.5, None),
            ("<0x0A>", 0.0, Some(6)),
            ("<extra>", -1000.0, Some(4)),
            ("<unused>", 0.0, Some(5)),
        ]);
        let pieces = parse_pieces(&bytes).unwrap();
        assert_eq!(pieces.len(), 6);
        assert_eq!(pieces[0].kind, PieceType::Unknown);
        assert_eq!(pieces[1].kind, PieceType::Control);
        assert_eq!(
            pieces[2],
            Piece {
                text: "\u{2581}the".into(),
                score: -3.5,
                kind: PieceType::Normal
            }
        );
        assert_eq!(pieces[3].kind, PieceType::Byte);
        assert_eq!(pieces[4].kind, PieceType::UserDefined);
        assert_eq!(pieces[5].kind, PieceType::Unused);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_pieces(&[]).is_err());
        assert!(parse_pieces(&[0x0a, 0x05, b'a']).is_err());
        assert!(parse_pieces(&[0xff; 12]).is_err());
    }
}
