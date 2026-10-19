//! Hand-written pre-tokenizers for byte-level BPE vocabularies.
//!
//! llama.cpp chooses a chain of regular-expression stages from
//! `tokenizer.ggml.pre` and applies them with `unicode_regex_split`: stage 1
//! splits the whole text into the pieces it matches and the unmatched gaps
//! between them, stage 2 runs over each of those fragments separately, and so on;
//! what remains after the last stage are the words that BPE merges
//! independently. This module reproduces that chain without a regex engine.
//! Every stage is a linear scanner that tries the alternatives of its pattern in
//! order at each position and resolves greedy matching plus backtracking by
//! hand (`\s+(?!\S)`, `\s*[\r\n]+`, `[^\r\n\p{L}\p{N}]?\p{L}+`, and the GPT-4o
//! pair of overlapping case classes), so the cost is linear in the text and
//! never exponential.
//!
//! Character classes follow Unicode: `\p{L}` is the five letter categories,
//! `\p{M}` the three mark categories, `\p{P}` the punctuation categories,
//! `\p{N}` is [`char::is_numeric`] and `\s` is [`char::is_whitespace`]. The
//! GPT-4o stages distinguish upper and lower case letters with the real
//! `\p{Lu}`/`\p{Ll}` categories; llama.cpp approximates that split with ASCII
//! case only (every non-ASCII letter counts as both), so the two can differ on
//! non-ASCII mixed-case runs such as `éÉ`.
//!
//! Because stages run on the fragments the previous stage left, a chain is not
//! the same as one regex over the whole text: in the starcoder family (`\p{N}`
//! then GPT-2) a whitespace run right before a digit ends its fragment and so
//! stays whole, where a single GPT-2 regex would give its last space to the
//! digits. That is llama.cpp's behaviour and is what this module reproduces.
//!
//! Allocation: a split call fills two caller-owned range vectors; stages
//! allocate nothing.

use std::ops::Range;

use unicode_general_category::{GeneralCategory, get_general_category};

/// Parameters of the shared "word" regex family
/// (`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
/// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+` and its variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WordRule {
    /// A standalone case-insensitive contraction alternative opens the pattern
    /// (llama3, qwen2); GPT-4o instead appends contractions to a word.
    standalone_contractions: bool,
    /// Longest digit group: 3 for llama3 and GPT-4o, 1 for qwen2.
    max_digits: usize,
    /// Combining marks join letter runs and are excluded from the punctuation
    /// class (qwen35).
    marks_are_letters: bool,
    /// Words split on the `[\p{Lu}...]*[\p{Ll}...]+` / `[\p{Lu}...]+[\p{Ll}...]*`
    /// case pair and the punctuation tail also takes `/` (GPT-4o).
    case_split: bool,
}

/// The llama3 pattern (also smaug-bpe and dbrx).
const LLAMA3: WordRule = WordRule {
    standalone_contractions: true,
    max_digits: 3,
    marks_are_letters: false,
    case_split: false,
};
/// The qwen2 pattern: single digits.
const QWEN2: WordRule = WordRule {
    max_digits: 1,
    ..LLAMA3
};
/// The qwen35 pattern: qwen2 with `\p{M}` in letter runs.
const QWEN35: WordRule = WordRule {
    marks_are_letters: true,
    ..QWEN2
};
/// The GPT-4o pattern.
const GPT4O: WordRule = WordRule {
    standalone_contractions: false,
    max_digits: 3,
    marks_are_letters: false,
    case_split: true,
};

/// One stage of a pre-tokenizer chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    /// `\p{N}`: every numeric character is a piece of its own.
    Digit,
    /// `\p{N}+`.
    DigitRun,
    /// `[0-9][0-9][0-9]`.
    ThreeAsciiDigits,
    /// `[\p{P}\$\+<=>\^~\|]+`.
    PunctRun,
    /// The GPT-2 pattern `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
    /// ?[^\s\p{L}\p{N}]+|\s+(?!\S)`.
    Gpt2,
    /// A member of the llama3 / qwen / GPT-4o word family.
    Word(WordRule),
}

/// llama.cpp's `default` chain, used for a missing `tokenizer.ggml.pre`.
const DEFAULT_STAGES: &[Stage] = &[
    Stage::PunctRun,
    Stage::Gpt2,
    Stage::DigitRun,
    Stage::ThreeAsciiDigits,
];
/// A single GPT-2 stage.
const GPT2_STAGES: &[Stage] = &[Stage::Gpt2];
/// The starcoder family: single digits, then GPT-2.
const STARCODER_STAGES: &[Stage] = &[Stage::Digit, Stage::Gpt2];
/// The llama3 pattern.
const LLAMA3_STAGES: &[Stage] = &[Stage::Word(LLAMA3)];
/// The qwen2 pattern.
const QWEN2_STAGES: &[Stage] = &[Stage::Word(QWEN2)];
/// The qwen35 pattern.
const QWEN35_STAGES: &[Stage] = &[Stage::Word(QWEN35)];
/// The GPT-4o pattern.
const GPT4O_STAGES: &[Stage] = &[Stage::Word(GPT4O)];

/// What a `tokenizer.ggml.pre` value selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pre {
    /// A regex chain over GPT-2 byte-encoded words.
    Words(Pretokenizer),
    /// The space-escaped, newline-run-split scheme of `gemma4`.
    Gemma4,
}

/// A stage chain plus the llama.cpp flags that travel with the pre name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Pretokenizer {
    /// Stages applied in order.
    stages: &'static [Stage],
    /// A word that is already a token is emitted without merging.
    ignore_merges: bool,
    /// llama.cpp's default for `add_bos` under this pre.
    add_bos: bool,
}

impl Pre {
    /// Resolve a `tokenizer.ggml.pre` value (`None` and `default` select the
    /// default chain). Unknown names yield `None`.
    pub(super) fn from_name(name: Option<&str>) -> Option<Self> {
        let words = |stages, ignore_merges, add_bos| {
            Some(Self::Words(Pretokenizer {
                stages,
                ignore_merges,
                add_bos,
            }))
        };
        match name.unwrap_or("default") {
            "default" => words(DEFAULT_STAGES, false, false),
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral"
            | "midm-2.0" | "lfm2" | "jina-v5-nano" => words(LLAMA3_STAGES, true, true),
            "smaug-bpe" | "dbrx" => words(LLAMA3_STAGES, false, false),
            "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez" | "stablelm2"
            | "hunyuan" | "solar-open" => words(QWEN2_STAGES, false, false),
            "qwen35" => words(QWEN35_STAGES, false, false),
            "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es"
            | "jina-v2-de" | "a.x-4.0" | "mellum" | "modern-bert" | "mpt" | "olmo" | "jais"
            | "trillion" | "granite-docling" | "exaone4" => words(GPT2_STAGES, false, false),
            "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone"
            | "minerva-7b" | "mellum2" => words(STARCODER_STAGES, false, false),
            "gpt-4o" | "llama4" | "kanana2" | "talkie" | "minimax-m2" => {
                words(GPT4O_STAGES, false, false)
            }
            "gemma4" => Some(Self::Gemma4),
            _ => None,
        }
    }
}

impl Pretokenizer {
    /// Whether whole-word vocabulary hits skip merging.
    pub(super) fn ignore_merges(&self) -> bool {
        self.ignore_merges
    }

    /// llama.cpp's `add_bos` default for this pre.
    pub(super) fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// Split `text` into words, appending their byte ranges to `out`.
    ///
    /// `scratch` is the second range buffer the stage chain ping-pongs with; both
    /// vectors are cleared first, so callers reuse them across calls.
    pub(super) fn split(
        &self,
        text: &str,
        out: &mut Vec<Range<usize>>,
        scratch: &mut Vec<Range<usize>>,
    ) {
        out.clear();
        scratch.clear();
        if text.is_empty() {
            return;
        }
        out.push(0..text.len());
        for &stage in self.stages {
            scratch.clear();
            split_stage(stage, text, out, scratch);
            std::mem::swap(out, scratch);
        }
    }
}

/// Apply one stage to every fragment in `input`, appending matches and the gaps
/// between them to `output` in text order.
fn split_stage(stage: Stage, text: &str, input: &[Range<usize>], output: &mut Vec<Range<usize>>) {
    for fragment in input {
        let piece = &text[fragment.clone()];
        let base = fragment.start;
        let mut pos = 0;
        let mut gap_start: Option<usize> = None;
        while pos < piece.len() {
            match match_at(stage, piece, pos) {
                Some(end) if end > pos => {
                    if let Some(start) = gap_start.take() {
                        output.push(base + start..base + pos);
                    }
                    output.push(base + pos..base + end);
                    pos = end;
                }
                _ => {
                    gap_start.get_or_insert(pos);
                    pos += char_len_at(piece, pos);
                }
            }
        }
        if let Some(start) = gap_start {
            output.push(base + start..base + piece.len());
        }
    }
}

/// Try the stage's pattern at byte offset `pos` of `s`, returning the end of the
/// match.
fn match_at(stage: Stage, s: &str, pos: usize) -> Option<usize> {
    let c = char_at(s, pos)?;
    match stage {
        Stage::Digit => is_number(c).then(|| pos + c.len_utf8()),
        Stage::DigitRun => is_number(c).then(|| run_end(s, pos, is_number)),
        Stage::ThreeAsciiDigits => {
            let bytes = s.as_bytes().get(pos..pos + 3)?;
            bytes.iter().all(u8::is_ascii_digit).then_some(pos + 3)
        }
        Stage::PunctRun => is_punct_run_char(c).then(|| run_end(s, pos, is_punct_run_char)),
        Stage::Gpt2 => gpt2_at(s, pos, c),
        Stage::Word(rule) => word_at(rule, s, pos, c),
    }
}

/// The character starting at byte offset `pos`.
fn char_at(s: &str, pos: usize) -> Option<char> {
    s.get(pos..)?.chars().next()
}

/// UTF-8 length of the character at `pos` (1 past the end).
fn char_len_at(s: &str, pos: usize) -> usize {
    char_at(s, pos).map_or(1, char::len_utf8)
}

/// End of the maximal run of characters satisfying `pred` starting at `pos`.
fn run_end(s: &str, pos: usize, pred: impl Fn(char) -> bool) -> usize {
    let mut end = pos;
    for c in s[pos..].chars() {
        if !pred(c) {
            break;
        }
        end += c.len_utf8();
    }
    end
}

/// `\p{L}`.
fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

/// `\p{M}`.
fn is_mark(c: char) -> bool {
    !c.is_ascii()
        && matches!(
            get_general_category(c),
            GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
        )
}

/// `\p{N}`.
fn is_number(c: char) -> bool {
    c.is_numeric()
}

/// `\s`.
fn is_space(c: char) -> bool {
    c.is_whitespace()
}

/// `\p{Lu}` or `\p{Lt}`: the letters only the "upper" class of GPT-4o holds.
fn is_upper_only(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_uppercase();
    }
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter | GeneralCategory::TitlecaseLetter
    )
}

/// `\p{Lm}`, `\p{Lo}` or `\p{M}`: members of both GPT-4o case classes.
fn is_case_neutral(c: char) -> bool {
    !c.is_ascii()
        && matches!(
            get_general_category(c),
            GeneralCategory::ModifierLetter
                | GeneralCategory::OtherLetter
                | GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
        )
}

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`.
fn is_upper_class(c: char) -> bool {
    is_upper_only(c) || is_case_neutral(c)
}

/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`.
fn is_lower_class(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_lowercase();
    }
    get_general_category(c) == GeneralCategory::LowercaseLetter || is_case_neutral(c)
}

/// `\p{P}`.
fn is_punctuation(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_punctuation()
            && !matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '`' | '|' | '~');
    }
    matches!(
        get_general_category(c),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
    )
}

/// `[\p{P}\$\+<=>\^~\|]`.
fn is_punct_run_char(c: char) -> bool {
    is_punctuation(c) || matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '~' | '|')
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)` (or the case-sensitive GPT-2 form) starting at
/// the apostrophe at `pos`.
fn contraction(s: &str, pos: usize, case_insensitive: bool) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.get(pos) != Some(&b'\'') {
        return None;
    }
    let fold = |b: u8| {
        if case_insensitive {
            b.to_ascii_lowercase()
        } else {
            b
        }
    };
    let first = fold(*bytes.get(pos + 1)?);
    if matches!(first, b's' | b't' | b'm' | b'd') {
        return Some(pos + 2);
    }
    let second = fold(*bytes.get(pos + 2)?);
    matches!((first, second), (b'r', b'e') | (b'v', b'e') | (b'l', b'l')).then_some(pos + 3)
}

/// The GPT-2 pattern at `pos` (`c` is the character there).
fn gpt2_at(s: &str, pos: usize, c: char) -> Option<usize> {
    if c == '\''
        && let Some(end) = contraction(s, pos, false)
    {
        return Some(end);
    }
    let body = if c == ' ' { pos + 1 } else { pos };
    if let Some(d) = char_at(s, body) {
        if is_letter(d) {
            return Some(run_end(s, body, is_letter));
        }
        if is_number(d) {
            return Some(run_end(s, body, is_number));
        }
        if !is_space(d) {
            return Some(run_end(s, body, |x| {
                !is_space(x) && !is_letter(x) && !is_number(x)
            }));
        }
    }
    // `\s+(?!\S)`: the whole run at the end of the fragment, else all but the
    // last whitespace character (which the next match may claim as its space).
    if is_space(c) {
        let run = whitespace_run(s, pos);
        if run.end == s.len() {
            return Some(run.end);
        }
        if run.last_start > pos {
            return Some(run.last_start);
        }
    }
    None
}

/// Shape of a whitespace run, scanned once for the alternatives that depend on it.
struct WhitespaceRun {
    /// End of the run.
    end: usize,
    /// Start of the run's last character.
    last_start: usize,
    /// End of the last `\r` or `\n` in the run, if any.
    last_newline_end: Option<usize>,
}

/// Scan the whitespace run starting at `pos`.
fn whitespace_run(s: &str, pos: usize) -> WhitespaceRun {
    let mut run = WhitespaceRun {
        end: pos,
        last_start: pos,
        last_newline_end: None,
    };
    for c in s[pos..].chars() {
        if !is_space(c) {
            break;
        }
        run.last_start = run.end;
        run.end += c.len_utf8();
        if c == '\r' || c == '\n' {
            run.last_newline_end = Some(run.end);
        }
    }
    run
}

/// `[^\r\n\p{L}\p{N}]`: characters allowed as the optional prefix of a word.
fn is_word_prefix(c: char) -> bool {
    c != '\r' && c != '\n' && !is_letter(c) && !is_number(c)
}

/// The word-family pattern of `rule` at `pos`.
fn word_at(rule: WordRule, s: &str, pos: usize, c: char) -> Option<usize> {
    if rule.standalone_contractions
        && c == '\''
        && let Some(end) = contraction(s, pos, true)
    {
        return Some(end);
    }
    if rule.case_split {
        if let Some(end) = case_split_word(s, pos, c) {
            return Some(end);
        }
    } else if let Some(end) = letter_word(rule, s, pos, c) {
        return Some(end);
    }

    // `\p{N}{1,3}` / `\p{N}`
    if is_number(c) {
        let mut end = pos;
        for d in s[pos..].chars().take(rule.max_digits) {
            if !is_number(d) {
                break;
            }
            end += d.len_utf8();
        }
        return Some(end);
    }

    // ` ?[^\s\p{L}\p{N}]+[\r\n]*` (`[\r\n/]*` for GPT-4o; `\p{M}` excluded too
    // for qwen35).
    let punct = |x: char| {
        !is_space(x) && !is_letter(x) && !is_number(x) && !(rule.marks_are_letters && is_mark(x))
    };
    let body = if c == ' ' { pos + 1 } else { pos };
    if char_at(s, body).is_some_and(punct) {
        let end = run_end(s, body, punct);
        let tail = |x: char| x == '\r' || x == '\n' || (rule.case_split && x == '/');
        return Some(run_end(s, end, tail));
    }

    if !is_space(c) {
        return None;
    }
    let run = whitespace_run(s, pos);
    // `\s*[\r\n]+`
    if let Some(end) = run.last_newline_end {
        return Some(end);
    }
    // `\s+(?!\S)`, then `\s+`
    if run.end == s.len() || run.last_start == pos {
        return Some(run.end);
    }
    Some(run.last_start)
}

/// `[^\r\n\p{L}\p{N}]?\p{L}+` (letters widened to `[\p{L}\p{M}]` for qwen35).
fn letter_word(rule: WordRule, s: &str, pos: usize, c: char) -> Option<usize> {
    let letter = |x: char| is_letter(x) || (rule.marks_are_letters && is_mark(x));
    if is_word_prefix(c) {
        let next = pos + c.len_utf8();
        if char_at(s, next).is_some_and(letter) {
            return Some(run_end(s, next, letter));
        }
    }
    letter(c).then(|| run_end(s, pos, letter))
}

/// The two GPT-4o word alternatives, each with and without the optional prefix:
/// `[^\r\n\p{L}\p{N}]?[\p{Lu}..]*[\p{Ll}..]+(?i:contraction)?` first, then
/// `[^\r\n\p{L}\p{N}]?[\p{Lu}..]+[\p{Ll}..]*(?i:contraction)?`.
fn case_split_word(s: &str, pos: usize, c: char) -> Option<usize> {
    for lower_tail in [true, false] {
        if is_word_prefix(c)
            && let Some(end) = case_body(s, pos + c.len_utf8(), lower_tail)
        {
            return Some(end);
        }
        if let Some(end) = case_body(s, pos, lower_tail) {
            return Some(end);
        }
    }
    None
}

/// Match `U*L+` (`lower_tail`) or `U+L*` at `start`, then an optional
/// contraction suffix, where `U` and `L` are the overlapping upper and lower
/// classes.
fn case_body(s: &str, start: usize, lower_tail: bool) -> Option<usize> {
    // Maximal upper-class prefix, remembering the last character that is also
    // lower-class (the point regex backtracking would settle on).
    let mut upper_end = start;
    let mut last_neutral: Option<(usize, usize)> = None;
    for c in s.get(start..)?.chars() {
        if !is_upper_class(c) {
            break;
        }
        if is_lower_class(c) {
            last_neutral = Some((upper_end, upper_end + c.len_utf8()));
        }
        upper_end += c.len_utf8();
    }
    let end = if lower_tail {
        if char_at(s, upper_end).is_some_and(is_lower_class) {
            run_end(s, upper_end, is_lower_class)
        } else {
            last_neutral?.1
        }
    } else {
        if upper_end == start {
            return None;
        }
        run_end(s, upper_end, is_lower_class)
    };
    Some(contraction(s, end, true).unwrap_or(end))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Split `text` with the chain named `pre` and return the words.
    fn words(pre: &str, text: &str) -> Vec<String> {
        let Some(Pre::Words(p)) = Pre::from_name(Some(pre)) else {
            panic!("`{pre}` is not a word pre-tokenizer");
        };
        let (mut out, mut scratch) = (Vec::new(), Vec::new());
        p.split(text, &mut out, &mut scratch);
        let joined: String = out.iter().map(|r| &text[r.clone()]).collect();
        assert_eq!(joined, text, "words must tile the text");
        out.iter().map(|r| text[r.clone()].to_string()).collect()
    }

    /// Check a table of `(input, expected words)` against one pre name.
    fn check(pre: &str, cases: &[(&str, &[&str])]) {
        for (text, expected) in cases {
            assert_eq!(words(pre, text), *expected, "{pre}: {text:?}");
        }
    }

    #[test]
    fn llama3_words() {
        check(
            "llama-bpe",
            &[
                ("Hello world", &["Hello", " world"]),
                (
                    "I'm HE'S we'LL they'd 'VE",
                    &[
                        "I", "'m", " HE", "'S", " we", "'LL", " they", "'d", " '", "VE",
                    ],
                ),
                ("don't", &["don", "'t"]),
                ("rock 'n' roll'", &["rock", " '", "n", "'", " roll", "'"]),
                ("12345 6789", &["123", "45", " ", "678", "9"]),
                ("x1234y", &["x", "123", "4", "y"]),
                ("  two", &[" ", " two"]),
                ("a   b", &["a", "  ", " b"]),
                ("a \n b", &["a", " \n", " b"]),
                ("a\n\n\nb", &["a", "\n\n\n", "b"]),
                ("a\r\nb", &["a", "\r\n", "b"]),
                ("a \n  \n b", &["a", " \n  \n", " b"]),
                ("end  ", &["end", "  "]),
                ("tab\tbed\t\tx", &["tab", "\tbed", "\t", "\tx"]),
                ("!?\n\nok", &["!?\n\n", "ok"]),
                (" !!hi", &[" !!", "hi"]),
                ("$%^hello", &["$%^", "hello"]),
                (
                    "\u{4e16}\u{754c}\u{4f60}\u{597d}!",
                    &["\u{4e16}\u{754c}\u{4f60}\u{597d}", "!"],
                ),
                ("e\u{301}x", &["e", "\u{301}x"]),
                ("a\u{1f980}b", &["a", "\u{1f980}b"]),
                ("trailing ", &["trailing", " "]),
                (" ", &[" "]),
                ("\n", &["\n"]),
                ("", &[]),
            ],
        );
    }

    #[test]
    fn llama3_unicode_numbers_group_by_three() {
        // Arabic-Indic digits are `\p{N}`; superscripts too.
        check(
            "llama3",
            &[(
                "\u{661}\u{662}\u{663}\u{664}",
                &["\u{661}\u{662}\u{663}", "\u{664}"],
            )],
        );
    }

    #[test]
    fn qwen2_single_digits_and_whitespace() {
        check(
            "qwen2",
            &[
                (
                    "12345 6789",
                    &["1", "2", "3", "4", "5", " ", "6", "7", "8", "9"],
                ),
                ("It's 5pm", &["It", "'s", " ", "5", "pm"]),
                ("line1\n\nline2", &["line", "1", "\n\n", "line", "2"]),
                ("x  \ny", &["x", "  \n", "y"]),
                ("x\n  y", &["x", "\n", " ", " y"]),
                ("a\u{3000}b", &["a", "\u{3000}b"]),
            ],
        );
    }

    #[test]
    fn qwen35_marks_join_letters() {
        check(
            "qwen35",
            &[
                ("e\u{301}x", &["e\u{301}x"]),
                ("\u{e9}t\u{e9}", &["\u{e9}t\u{e9}"]),
                ("a\u{301} 1", &["a\u{301}", " ", "1"]),
                ("!\u{301}", &["!\u{301}"]),
            ],
        );
        // Under qwen2 the mark is not a letter: it only prefixes the next letters.
        check("qwen2", &[("e\u{301}x", &["e", "\u{301}x"])]);
    }

    #[test]
    fn gpt2_words() {
        check(
            "gpt-2",
            &[
                ("Hello world", &["Hello", " world"]),
                ("don't DON'T", &["don", "'t", " DON", "'", "T"]),
                (
                    "it's we're they've I'm you'll he'd",
                    &[
                        "it", "'s", " we", "'re", " they", "'ve", " I", "'m", " you", "'ll", " he",
                        "'d",
                    ],
                ),
                ("12345 6789", &["12345", " 6789"]),
                ("a  b", &["a", " ", " b"]),
                ("a   ", &["a", "   "]),
                ("a \t b", &["a", " \t", " b"]),
                ("a\tb", &["a", "\t", "b"]),
                ("a\n\nb", &["a", "\n", "\n", "b"]),
                ("end.\n", &["end", ".", "\n"]),
                ("  lead", &[" ", " lead"]),
                ("x\u{1f600}\u{1f600}y", &["x", "\u{1f600}\u{1f600}", "y"]),
                (" \u{4e2d}\u{6587}", &[" \u{4e2d}\u{6587}"]),
            ],
        );
    }

    #[test]
    fn starcoder_group_splits_digits_first() {
        check(
            "refact",
            &[
                ("abc123 def", &["abc", "1", "2", "3", " def"]),
                (
                    "12345 6789",
                    &["1", "2", "3", "4", "5", " ", "6", "7", "8", "9"],
                ),
                ("x  1", &["x", "  ", "1"]),
            ],
        );
        // llama.cpp semantics, which differ from a single GPT-2 regex over the whole
        // text (HuggingFace's refact/granite pre-tokenizer): the digit stage runs
        // first, so the fragment before a digit ends there, `\s+(?!\S)` sees that
        // end, and two spaces stay together instead of splitting off one for `1`.
        assert_eq!(words("starcoder", "a  1"), ["a", "  ", "1"]);
    }

    #[test]
    fn default_chain_splits_punctuation_then_gpt2_then_digits() {
        check(
            "default",
            &[
                ("Hello, world!", &["Hello", ",", " world", "!"]),
                ("a+b=c", &["a", "+", "b", "=", "c"]),
                ("x ... y", &["x", " ", "...", " y"]),
                ("12345 6789", &["123", "45", " ", "678", "9"]),
                ("a_b", &["a", "_", "b"]),
            ],
        );
        assert_eq!(
            Pre::from_name(None),
            Pre::from_name(Some("default")),
            "a missing pre is the default"
        );
    }

    #[test]
    fn gpt4o_case_split_words() {
        check(
            "gpt-4o",
            &[
                ("Hello world", &["Hello", " world"]),
                ("HelloWorld", &["Hello", "World"]),
                ("camelCaseWord", &["camel", "Case", "Word"]),
                ("HTTPServer", &["HTTPServer"]),
                ("ABC", &["ABC"]),
                ("ABCdef", &["ABCdef"]),
                ("ABc Def", &["ABc", " Def"]),
                ("don't DON'T Don'T", &["don't", " DON'T", " Don'T"]),
                ("It's", &["It's"]),
                ("'s", &["'s"]),
                ("12345", &["123", "45"]),
                ("x1", &["x", "1"]),
                ("a/b//\n", &["a", "/b", "//\n"]),
                ("foo.bar()\n\n", &["foo", ".bar", "()\n\n"]),
                ("a  b", &["a", " ", " b"]),
                ("end  \n x", &["end", "  \n", " x"]),
                ("\u{c9}t\u{e9}", &["\u{c9}t\u{e9}"]),
                ("\u{e9}\u{c9}", &["\u{e9}", "\u{c9}"]),
                ("\u{4e2d}\u{6587}x", &["\u{4e2d}\u{6587}x"]),
                ("e\u{301}x", &["e\u{301}x"]),
                ("\u{201c}Quote\u{201d}", &["\u{201c}Quote", "\u{201d}"]),
                ("-Hello", &["-Hello"]),
                ("\tHello", &["\tHello"]),
                ("\nHello", &["\n", "Hello"]),
            ],
        );
    }

    #[test]
    fn pre_names_resolve() {
        assert!(matches!(Pre::from_name(Some("gemma4")), Some(Pre::Gemma4)));
        assert!(Pre::from_name(Some("definitely-not-a-pre")).is_none());
        for name in [
            "default",
            "llama3",
            "llama-v3",
            "llama-bpe",
            "falcon3",
            "falcon-h1",
            "pixtral",
            "midm-2.0",
            "lfm2",
            "jina-v5-nano",
            "smaug-bpe",
            "dbrx",
            "qwen2",
            "deepseek-r1-qwen",
            "kormo",
            "f2llmv2",
            "megrez",
            "stablelm2",
            "hunyuan",
            "solar-open",
            "qwen35",
            "gpt-2",
            "phi-2",
            "mpt",
            "olmo",
            "jais",
            "trillion",
            "granite-docling",
            "exaone4",
            "jina-es",
            "starcoder",
            "refact",
            "command-r",
            "smollm",
            "codeshell",
            "exaone",
            "minerva-7b",
            "mellum2",
            "gpt-4o",
            "llama4",
            "kanana2",
            "talkie",
            "minimax-m2",
        ] {
            assert!(Pre::from_name(Some(name)).is_some(), "{name}");
        }
        let Some(Pre::Words(llama3)) = Pre::from_name(Some("llama-bpe")) else {
            panic!("llama-bpe resolves");
        };
        assert!(llama3.ignore_merges() && llama3.add_bos());
        let Some(Pre::Words(smaug)) = Pre::from_name(Some("smaug-bpe")) else {
            panic!("smaug-bpe resolves");
        };
        assert!(!smaug.ignore_merges() && !smaug.add_bos());
    }

    #[test]
    fn long_inputs_are_linear() {
        // A pathological whitespace run and a long uppercase run must not blow up.
        let spaces = " ".repeat(200_000) + "x";
        assert_eq!(words("llama3", &spaces).len(), 2);
        let upper = "A".repeat(200_000) + "1";
        assert_eq!(words("gpt-4o", &upper).len(), 2);
    }
}
