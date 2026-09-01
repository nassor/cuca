//! Outbound PII/secret scrubbing (`plugin-redaction`).
//!
//! [`RedactionPlugin`] implements [`CucaPlugin`] and rewrites the outbound
//! [`UnifiedRequest`] in `on_request`, the last pipeline stage before a provider
//! adapter puts the request on the wire
//! ([`CucaClient::generate_stream`](crate::client::CucaClient::generate_stream)).
//! Every match is replaced by `[REDACTED:{kind}]`, so a prompt assembled from
//! local files, tool output, or user input cannot carry a caller-owned API key,
//! email address, or card/ID number to a third-party endpoint.
//!
//! [`RedactionPlugin::scrub_str`] is the very same matcher the hook runs, and it
//! is public on purpose: a caller can scrub a value before it ever enters a
//! request, or reject on `count > 0` instead of rewriting.
//!
//! # Field map
//!
//! Every rewritten field is listed here, with the `field` label that identifies
//! it in the `tracing` event:
//!
//! | Location | `field` label | Action |
//! | --- | --- | --- |
//! | [`MessageContentBlock::Text`] | `"message_text"` | scrub the text |
//! | [`MessageContentBlock::Thinking`] `reasoning` | `"message_thinking"` | scrub the reasoning |
//! | [`MessageContentBlock::ToolCall`] `arguments` | `"tool_call_arguments"` | scrub every JSON **string leaf**, recursively, depth-bounded by [`RedactionConfig::MAX_JSON_DEPTH`] |
//! | [`MessageContentBlock::ToolResult`] `output` | `"tool_result_output"` | scrub the output |
//! | [`UnifiedMessage::name`](crate::types::UnifiedMessage::name) | `"message_name"` | scrub the `Some` payload |
//! | [`ToolDefinition::description`](crate::types::ToolDefinition::description) | `"tool_description"` | scrub when [`RedactionConfig::scrub_tool_definitions`] |
//!
//! Everything else is left byte-identical, each for a reason. A "missing" field
//! is a decision, not an oversight:
//!
//! - `Thinking::signature` — a provider signature authenticating the reasoning;
//!   rewriting it invalidates it.
//! - `ToolCall::id`, `ToolResult::tool_call_id`, `UnifiedMessage::tool_call_id`
//!   — correlation ids; scrubbing them breaks call/result matching.
//! - `ToolCall::name`, `ToolDefinition::name` — tool identity; a renamed tool is
//!   an unroutable tool.
//! - `ToolDefinition::input_schema` — JSON Schema keywords (`const`, `enum`,
//!   `pattern`) are the tool's contract; rewriting them silently changes tool
//!   semantics.
//! - `ImageBase64 { media_type, data }` — a literal hit inside base64 is a
//!   coincidence, and rewriting the payload corrupts the image.
//! - JSON **object keys** inside `arguments` — part of the tool's input
//!   contract, not user data.
//! - `model`, `provider`, `temperature`, `max_tokens`, `stream`, `thinking`,
//!   `prompt_cache` — routing and sampling knobs, not text. `provider` in
//!   particular is authoritative (see the
//!   [field-usage contract](crate::request)).
//!
//! # The matcher, and why there is no regex
//!
//! [`RedactionRule::Literal`] and [`RedactionRule::Prefixed`] compile to a
//! [`Finder`] once in [`RedactionPlugin::new`] — SIMD-accelerated substring
//! search over the crate's existing `memchr` dependency, with no per-request
//! searcher construction. [`RedactionRule::EmailLike`] and
//! [`RedactionRule::DigitRun`] are hand-rolled ASCII scans. Overlaps resolve
//! **leftmost-longest**, breaking an exact tie by policy order, so the output
//! string is a deterministic function of the policy and the input.
//!
//! There is deliberately no pattern language. `regex` would buy arbitrary
//! caller-authored detectors, at the price of three transitive crates
//! (`regex-syntax`, `regex-automata`, `aho-corasick`) in a crate whose identity
//! is five core dependencies, plus an untrusted pattern language on the
//! outbound hot path where a pathological pattern becomes per-request latency.
//! [`RedactionRule`] is the extension seam instead: a future `Regex` variant
//! behind its own feature is additive and does not change this plugin's shape.
//!
//! # Inbound text is not scrubbed
//!
//! `on_stream_chunk` is **not** implemented; the [`CucaPlugin`] default applies.
//! Three reasons:
//!
//! 1. *Threat model.* The leak this plugin exists to stop is caller-side data
//!    crossing the wire. An inbound block was produced by the provider, which
//!    has already seen whatever it emitted.
//! 2. *A per-chunk matcher would be silently unreliable.* Providers stream text
//!    incrementally, so a secret straddling two blocks would be missed. A
//!    matcher that works only when token boundaries happen to align is worse
//!    than none, and making it reliable needs cross-chunk buffering, which
//!    contradicts both the streaming latency story and the crate's bounded
//!    memory rules.
//! 3. *Inbound text is scrubbed on its next outbound pass.* Assistant turns and
//!    tool results re-enter the conversation as `messages` on the following
//!    request, where `on_request` scrubs them. Scrubbing is idempotent — a
//!    `Literal` value containing [`RedactionConfig::REDACTION_MARKER`] is
//!    rejected at construction, so a replacement token cannot re-match — and a
//!    multi-turn conversation therefore converges.
//!
//! # Order is observable, never required
//!
//! The plugin requires no registration position relative to any peer: it names
//! no peer plugin, and both orders are legal. Its effect *is* observable to
//! plugins registered after it, because `on_request` hooks run in registration
//! order over one shared request:
//!
//! - **A session store.** Registered *before* a store, the trajectory records
//!   scrubbed content; registered *after*, the store persists the raw value — to
//!   disk, when the store writes through a file backend, whose append-only
//!   format never rewrites an existing frame. Independently of order,
//!   **model-output records are never scrubbed**, because inbound scrubbing is
//!   deferred (above).
//! - **A prompt cache.** The cache key is digested from the request *after*
//!   every `on_request` hook, so enabling redaction changes every key: a
//!   snapshot imported from a pre-redaction run no longer matches, and is
//!   rejected as a digest mismatch rather than served. Two requests differing
//!   only inside a fully redacted secret converge onto one key — accepted on
//!   purpose, since the provider saw neither secret and the response cannot
//!   depend on which one it was.
//!
//! # Bounds and allocation
//!
//! Nothing here grows with traffic:
//!
//! - `rules` is a `Box<[_]>` fixed at construction, capped by
//!   [`RedactionConfig::MAX_RULES`], each pattern by
//!   [`RedactionConfig::MAX_PATTERN_BYTES`] and each `kind` by
//!   [`RedactionConfig::MAX_KIND_BYTES`]. An over-cap or empty policy is
//!   rejected before any allocation.
//! - The stats slot is two `u64` and one most-recent event tuple. The tuple is
//!   *replaced*, never appended; the total uses `saturating_add`. Per-`kind`
//!   aggregation belongs on the caller's OTel meter, fed by the `kind` field of
//!   the `tracing` event.
//! - The per-call match buffer is a stack local capped by
//!   [`RedactionConfig::max_matches_per_text`], checked while pushing. Past the
//!   cap the call fails: a capped scrub cannot promise the value is clean, so it
//!   refuses instead of truncating.
//! - `ToolCall::arguments` recursion is bounded by
//!   [`RedactionConfig::MAX_JSON_DEPTH`] frames, checked at frame entry.
//!
//! Allocation on the scan path: searchers and replacement tokens are built once
//! in [`RedactionPlugin::new`]. A clean string allocates **nothing** —
//! [`RedactionPlugin::scrub_str`] hands back a [`Cow::Borrowed`] and neither
//! per-call buffer is ever pushed to. A dirty string allocates one output
//! `String`, sized exactly from the input length and the net replacement delta,
//! plus the bounded match buffer and one `rules.len()`-entry candidate cache
//! that keeps the scan linear in the input instead of re-searching every rule
//! after every replacement. UTF-8 correctness needs no `unsafe`: a valid UTF-8
//! needle cannot hit at a non-`char` boundary inside a valid UTF-8 haystack, and
//! the prefixed/email/digit scanners only ever accept ASCII bytes, so every
//! match boundary is a `char` boundary and the output is assembled with
//! `push_str` over borrowed segments.
//!
//! **There is no near-cap warning, by construction.** No structure here fills up
//! over time, so there is nothing to warn about approaching; the two
//! cap-adjacent conditions are hard errors instead, and the per-redaction
//! `tracing::warn!` already gives an operator a rate signal.
//!
//! # Non-guarantees
//!
//! This is a byte matcher, not a classifier: **a secret it has no rule for
//! crosses the wire.** There is no built-in rule set and no "looks like a
//! secret" heuristic — every rule is caller-authored, so a false positive is a
//! policy the caller wrote and can see in the `tracing` event. It does not
//! encrypt anything, and it redacts nothing already persisted or exported: the
//! sensitivity contract on [`UnifiedRequest`] and on `cuca-export` is unchanged,
//! because redaction happens upstream of storage and is not an export-time
//! guarantee. The `tracing` event carries the rule's `kind`, the field label,
//! and an index — **never** the matched bytes or their surroundings.

use std::borrow::Cow;
use std::sync::Mutex;

use memchr::memmem::Finder;

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::request::UnifiedRequest;
use crate::types::MessageContentBlock;

/// This plugin's [`CucaPlugin::name`], reused in every [`PluginError`] it
/// raises.
const NAME: &str = "redaction";

/// The `schema` key on every policy rejection, so a caller can tell a config
/// rejection from a hook failure without matching on message text.
const CONFIG_SCHEMA: &str = "redaction-config";

/// `field` label for [`MessageContentBlock::Text`].
const FIELD_MESSAGE_TEXT: &str = "message_text";
/// `field` label for [`MessageContentBlock::Thinking`]'s `reasoning`.
const FIELD_MESSAGE_THINKING: &str = "message_thinking";
/// `field` label for a string leaf of [`MessageContentBlock::ToolCall`]'s
/// `arguments`.
const FIELD_TOOL_CALL_ARGUMENTS: &str = "tool_call_arguments";
/// `field` label for [`MessageContentBlock::ToolResult`]'s `output`.
const FIELD_TOOL_RESULT_OUTPUT: &str = "tool_result_output";
/// `field` label for `UnifiedMessage::name`.
const FIELD_MESSAGE_NAME: &str = "message_name";
/// `field` label for `ToolDefinition::description`.
const FIELD_TOOL_DESCRIPTION: &str = "tool_description";

/// What a rule looks for. `kind` labels the replacement token and the tracing
/// event; it is never the matched value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedactionRule {
    /// An exact, byte-for-byte occurrence of a known secret value.
    Literal {
        /// Slug naming the secret class, e.g. `"api-key"`.
        kind: String,
        /// The exact bytes to find.
        value: String,
    },
    /// `prefix` followed by `min_len..=max_len` bytes from the token alphabet
    /// (ASCII alphanumeric, `-`, `_`), e.g. `prefix = "sk-"`.
    Prefixed {
        /// Slug naming the secret class, e.g. `"api-key"`.
        kind: String,
        /// The literal prefix that opens the token.
        prefix: String,
        /// Fewest token bytes after `prefix` that still count as a match.
        min_len: usize,
        /// Most token bytes consumed after `prefix`; the match truncates here.
        max_len: usize,
    },
    /// A `local@domain.tld`-shaped run (ASCII local part, at least one dot in
    /// the domain, no trailing separator).
    EmailLike {
        /// Slug naming the secret class, e.g. `"email"`.
        kind: String,
    },
    /// A run of `min_digits..=max_digits` ASCII digits, optionally interrupted
    /// by single `-` or ` ` separators (card / phone / national-ID shapes).
    DigitRun {
        /// Slug naming the secret class, e.g. `"card"`.
        kind: String,
        /// Fewest digits in the run that still count as a match.
        min_digits: usize,
        /// Most digits consumed; the match truncates on this digit.
        max_digits: usize,
    },
}

impl RedactionRule {
    /// The rule's `kind` slug, whichever variant this is.
    fn kind(&self) -> &str {
        match self {
            RedactionRule::Literal { kind, .. }
            | RedactionRule::Prefixed { kind, .. }
            | RedactionRule::EmailLike { kind }
            | RedactionRule::DigitRun { kind, .. } => kind,
        }
    }
}

/// Caller-supplied redaction policy. Bounds are validated by
/// [`RedactionConfig::new`] and again by [`RedactionPlugin::new`].
#[derive(Debug, Clone)]
pub struct RedactionConfig {
    /// The rules, applied in this order for overlap tie-breaks. Must be
    /// non-empty: an empty policy is an inert no-op, not a valid config.
    pub rules: Vec<RedactionRule>,
    /// Per-string cap on matches; exceeding it fails the hook loudly rather
    /// than partially redacting (default 1024).
    pub max_matches_per_text: usize,
    /// Whether `ToolDefinition::description` is scrubbed (default `true`).
    pub scrub_tool_definitions: bool,
}

impl Default for RedactionConfig {
    /// An **unusable** default: `rules` is empty, so [`RedactionPlugin::new`]
    /// rejects it. Exists for `..Default::default()` struct-update ergonomics
    /// only, as with `MemoryConfig`.
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            max_matches_per_text: 1024,
            scrub_tool_definitions: true,
        }
    }
}

impl RedactionConfig {
    /// Maximum rules in one policy.
    pub const MAX_RULES: usize = 256;
    /// Maximum bytes in a `Literal::value` or `Prefixed::prefix`.
    pub const MAX_PATTERN_BYTES: usize = 512;
    /// Maximum bytes in a `kind` slug.
    pub const MAX_KIND_BYTES: usize = 32;
    /// Ceiling on `max_matches_per_text`.
    pub const MAX_MATCHES_PER_TEXT: usize = 4096;
    /// Maximum `serde_json::Value` nesting walked inside `ToolCall::arguments`.
    pub const MAX_JSON_DEPTH: usize = 64;
    /// Prefix of every replacement token; a `Literal::value` containing it is
    /// rejected, which is what makes scrubbing idempotent.
    pub const REDACTION_MARKER: &'static str = "[REDACTED:";

    /// Rejecting constructor: validated bounds, defaults for the rest.
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] with `schema: "redaction-config"` for every
    /// case listed in [`Self::validate`].
    pub fn new(rules: Vec<RedactionRule>) -> Result<Self, PluginError> {
        let config = Self {
            rules,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    /// Reject a policy that cannot be honored: empty `rules`; more than
    /// [`Self::MAX_RULES`]; an empty or over-long pattern; a `kind` that is not
    /// a non-empty ASCII slug (`[a-z0-9_-]`) within [`Self::MAX_KIND_BYTES`];
    /// a `Literal::value` containing [`Self::REDACTION_MARKER`];
    /// `min_len > max_len`; `min_digits == 0` or `min_digits > max_digits`;
    /// `max_matches_per_text` of `0` or above [`Self::MAX_MATCHES_PER_TEXT`].
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] with `schema: "redaction-config"`; the
    /// message names the offending field (`rules[3].kind`,
    /// `max_matches_per_text`, ...) so a caller can find it.
    pub fn validate(&self) -> Result<(), PluginError> {
        if self.rules.is_empty() {
            return Err(invalid(
                "rules must not be empty: an empty policy is an inert no-op, not a redaction policy",
            ));
        }
        if self.rules.len() > Self::MAX_RULES {
            return Err(invalid(format!(
                "rules holds {} entries, over MAX_RULES ({})",
                self.rules.len(),
                Self::MAX_RULES
            )));
        }
        for (i, rule) in self.rules.iter().enumerate() {
            validate_kind(i, rule.kind())?;
            match rule {
                RedactionRule::Literal { value, .. } => {
                    validate_pattern(i, "value", value)?;
                    if value.contains(Self::REDACTION_MARKER) {
                        return Err(invalid(format!(
                            "rules[{i}].value contains the replacement marker {:?}, which would \
                             make scrubbing non-idempotent",
                            Self::REDACTION_MARKER
                        )));
                    }
                }
                RedactionRule::Prefixed {
                    prefix,
                    min_len,
                    max_len,
                    ..
                } => {
                    validate_pattern(i, "prefix", prefix)?;
                    if min_len > max_len {
                        return Err(invalid(format!(
                            "rules[{i}].min_len ({min_len}) is over rules[{i}].max_len ({max_len})"
                        )));
                    }
                }
                RedactionRule::EmailLike { .. } => {}
                RedactionRule::DigitRun {
                    min_digits,
                    max_digits,
                    ..
                } => {
                    if *min_digits == 0 {
                        return Err(invalid(format!(
                            "rules[{i}].min_digits must be non-zero: a zero-digit run matches \
                             everywhere"
                        )));
                    }
                    if min_digits > max_digits {
                        return Err(invalid(format!(
                            "rules[{i}].min_digits ({min_digits}) is over rules[{i}].max_digits \
                             ({max_digits})"
                        )));
                    }
                }
            }
        }
        if self.max_matches_per_text == 0 {
            return Err(invalid(
                "max_matches_per_text must be non-zero: a zero cap refuses every dirty string",
            ));
        }
        if self.max_matches_per_text > Self::MAX_MATCHES_PER_TEXT {
            return Err(invalid(format!(
                "max_matches_per_text ({}) is over MAX_MATCHES_PER_TEXT ({})",
                self.max_matches_per_text,
                Self::MAX_MATCHES_PER_TEXT
            )));
        }
        Ok(())
    }
}

/// A policy rejection, always carrying the `"redaction-config"` schema key.
fn invalid(message: impl Into<String>) -> PluginError {
    PluginError::Validation {
        schema: CONFIG_SCHEMA.to_string(),
        message: message.into(),
    }
}

/// A `kind` slug: non-empty, within [`RedactionConfig::MAX_KIND_BYTES`], and
/// made only of `[a-z0-9_-]` so it is safe inside a replacement token and in a
/// `tracing` field.
fn validate_kind(index: usize, kind: &str) -> Result<(), PluginError> {
    if kind.is_empty() {
        return Err(invalid(format!("rules[{index}].kind must not be empty")));
    }
    if kind.len() > RedactionConfig::MAX_KIND_BYTES {
        return Err(invalid(format!(
            "rules[{index}].kind is {} bytes, over MAX_KIND_BYTES ({})",
            kind.len(),
            RedactionConfig::MAX_KIND_BYTES
        )));
    }
    if !kind
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
    {
        return Err(invalid(format!(
            "rules[{index}].kind must be an ASCII slug of [a-z0-9_-]; got {kind:?}"
        )));
    }
    Ok(())
}

/// A literal value or a rule prefix: non-empty and within
/// [`RedactionConfig::MAX_PATTERN_BYTES`].
fn validate_pattern(index: usize, field: &str, pattern: &str) -> Result<(), PluginError> {
    if pattern.is_empty() {
        return Err(invalid(format!("rules[{index}].{field} must not be empty")));
    }
    if pattern.len() > RedactionConfig::MAX_PATTERN_BYTES {
        return Err(invalid(format!(
            "rules[{index}].{field} is {} bytes, over MAX_PATTERN_BYTES ({})",
            pattern.len(),
            RedactionConfig::MAX_PATTERN_BYTES
        )));
    }
    Ok(())
}

/// The result of scrubbing one string: borrowed and `count == 0` when clean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Redacted<'a> {
    /// The scrubbed text; `Cow::Borrowed` iff nothing matched.
    pub text: Cow<'a, str>,
    /// Number of replacements applied.
    pub count: usize,
}

/// Outbound PII/secret scrubbing over every text-bearing request field.
///
/// `Send + Sync` via the [`CucaPlugin`] supertrait, so one instance is shared as
/// `Arc<dyn CucaPlugin>` across `await` points in the client pipeline.
///
/// # Growth
///
/// Nothing here grows with traffic: `rules` is fixed at construction, the stats
/// slot holds two counters and one most-recent event tuple, and the per-call
/// match buffer is a stack local capped by `max_matches_per_text`. See the
/// module header's *Bounds and allocation* section.
pub struct RedactionPlugin {
    /// Compiled rules in policy order; fixed at construction.
    rules: Box<[CompiledRule]>,
    /// Per-string match cap (see [`RedactionConfig::max_matches_per_text`]).
    max_matches_per_text: usize,
    /// Whether `ToolDefinition::description` is scrubbed.
    scrub_tool_definitions: bool,
    /// Counters plus the most recent event tuple; one lock per `on_request`,
    /// taken after the scan, never inside it.
    stats: Mutex<RedactionStats>,
}

/// A rule with its searcher and replacement token built once.
struct CompiledRule {
    /// The rule's `kind` slug, as reported in the `tracing` event.
    kind: Box<str>,
    /// `"[REDACTED:{kind}]"`, allocated at construction, never per match.
    replacement: Box<str>,
    /// The compiled searcher.
    matcher: CompiledMatcher,
}

/// The four matcher shapes. `Finder<'static>` is built once via
/// `Finder::new(..).into_owned()`; no per-request searcher construction.
enum CompiledMatcher {
    /// [`RedactionRule::Literal`]: the whole needle is the match.
    Literal(Finder<'static>),
    /// [`RedactionRule::Prefixed`]: the needle opens a token run.
    Prefixed {
        /// Searcher over the rule's `prefix`.
        finder: Finder<'static>,
        /// Fewest token bytes after the prefix.
        min_len: usize,
        /// Most token bytes after the prefix.
        max_len: usize,
    },
    /// [`RedactionRule::EmailLike`]: an ASCII `local@domain.tld` scan.
    EmailLike,
    /// [`RedactionRule::DigitRun`]: an ASCII digit-run scan.
    DigitRun {
        /// Fewest digits in a qualifying run.
        min_digits: usize,
        /// Most digits consumed by one match.
        max_digits: usize,
    },
}

/// Fixed-size stats: no per-rule or per-kind map, so nothing grows with the
/// number of distinct secrets seen. Per-kind aggregation belongs on the
/// caller's OTel meter, fed by the `kind` field of the tracing event.
#[derive(Debug, Default)]
struct RedactionStats {
    /// Redactions applied by the most recent `on_request`.
    last_request: u64,
    /// Saturating cumulative total across every request.
    total: u64,
    /// Most recent `(kind, field, count)` tuple, mirroring the tracing event.
    last_event: Option<(String, &'static str, usize)>,
}

/// One selected, non-overlapping replacement site.
///
/// `start`/`end` are byte offsets into the scrubbed input and always land on
/// `char` boundaries (see the module header's *Bounds and allocation* section).
#[derive(Debug, Clone, Copy)]
struct Match {
    /// Byte offset of the first replaced byte.
    start: usize,
    /// Byte offset one past the last replaced byte.
    end: usize,
    /// Index into [`RedactionPlugin::rules`] of the rule that matched.
    rule: usize,
}

/// A dirty scrub: the rewritten string plus the replacements that produced it,
/// in input order. Only ever built when at least one rule matched, which is what
/// keeps the clean path allocation-free.
struct ScrubHit {
    /// The rewritten string.
    text: String,
    /// The applied replacements, left to right.
    matches: Vec<Match>,
}

/// Per-hook accumulator, so the `stats` mutex is taken exactly once, after the
/// scan rather than inside it.
#[derive(Default)]
struct HookTally {
    /// Redactions applied so far in this hook.
    count: usize,
    /// Most recent `(kind, field, count)` tuple.
    event: Option<(String, &'static str, usize)>,
}

impl RedactionPlugin {
    /// Compile `config` into searchers and replacement tokens.
    ///
    /// # Errors
    ///
    /// The [`RedactionConfig::validate`] errors.
    pub fn new(config: RedactionConfig) -> Result<Self, PluginError> {
        config.validate()?;
        Ok(Self {
            rules: config.rules.iter().map(CompiledRule::compile).collect(),
            max_matches_per_text: config.max_matches_per_text,
            scrub_tool_definitions: config.scrub_tool_definitions,
            stats: Mutex::new(RedactionStats::default()),
        })
    }

    /// Scrub one string without allocating when nothing matches.
    ///
    /// The single matcher entry point; the hook and any caller-side use share
    /// it. Matches resolve leftmost-longest, breaking ties by policy order.
    ///
    /// # Errors
    ///
    /// [`PluginError::HookFailure`] (`stage: "request"`) when the string holds
    /// more than [`Self::match_cap`] matches: a capped scrub cannot promise the
    /// value is clean, so it refuses instead of truncating.
    pub fn scrub_str<'a>(&self, text: &'a str) -> Result<Redacted<'a>, PluginError> {
        match self.scrub(text)? {
            None => Ok(Redacted {
                text: Cow::Borrowed(text),
                count: 0,
            }),
            Some(hit) => Ok(Redacted {
                count: hit.matches.len(),
                text: Cow::Owned(hit.text),
            }),
        }
    }

    /// Rules in the compiled policy. Cheap: reads an immutable field.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// The per-string match cap. Cheap: reads an immutable field.
    pub fn match_cap(&self) -> usize {
        self.max_matches_per_text
    }

    /// Redactions applied during the most recent `on_request` (the per-request
    /// accessor). `0` before the first request.
    pub fn last_request_redactions(&self) -> u64 {
        self.stats().last_request
    }

    /// Saturating cumulative redactions across every request.
    pub fn total_redactions(&self) -> u64 {
        self.stats().total
    }

    /// Most recent `(kind, field, count)` event, mirroring the `tracing` event
    /// emitted alongside it. Never carries a matched value.
    pub fn last_redaction_event(&self) -> Option<(String, &'static str, usize)> {
        self.stats().last_event.clone()
    }

    /// The stats slot, recovering a mutex poisoned by a panicking holder: this
    /// is diagnostics, and the scrub has already happened, so bookkeeping never
    /// fails a request.
    fn stats(&self) -> std::sync::MutexGuard<'_, RedactionStats> {
        self.stats.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The shared matcher body: `None` when `text` is clean, so
    /// [`Self::scrub_str`] can hand back a borrow and the hook can skip the
    /// assignment entirely.
    ///
    /// # Errors
    ///
    /// [`PluginError::HookFailure`] past [`Self::match_cap`] matches.
    fn scrub(&self, text: &str) -> Result<Option<ScrubHit>, PluginError> {
        // Cold pass: one search per rule over the whole string. A clean string
        // stops here, having allocated nothing at all.
        let mut pending: Vec<Option<(usize, usize)>> = Vec::new();
        let mut best: Option<Match> = None;
        for (rule, compiled) in self.rules.iter().enumerate() {
            let Some(span) = compiled.matcher.find_from(text, 0) else {
                continue;
            };
            if pending.is_empty() {
                // Rules ahead of this one matched nowhere in `text`, so `None`
                // is their final answer and the cache can be seeded here — one
                // allocation the clean path never reaches.
                pending = vec![None; self.rules.len()];
            }
            pending[rule] = Some(span);
            best = Some(preferred(best, span, rule));
        }
        let Some(mut selected) = best else {
            return Ok(None);
        };

        // Warm pass: consume the selected match, then re-search only the rules
        // whose cached candidate now starts behind the cursor. Every rule's
        // searches therefore advance monotonically, keeping the whole scan
        // linear in the input rather than quadratic in the match count.
        let mut matches: Vec<Match> = Vec::new();
        loop {
            if matches.len() == self.max_matches_per_text {
                return Err(PluginError::hook(
                    NAME,
                    "request",
                    format!(
                        "a single value holds more than max_matches_per_text ({}) matches; the \
                         request is refused rather than partially scrubbed",
                        self.max_matches_per_text
                    ),
                ));
            }
            matches.push(selected);
            let cursor = selected.end;
            let mut next: Option<Match> = None;
            for (rule, slot) in pending.iter_mut().enumerate() {
                if slot.is_some_and(|(start, _)| start < cursor) {
                    *slot = self.rules[rule].matcher.find_from(text, cursor);
                }
                if let Some(span) = *slot {
                    next = Some(preferred(next, span, rule));
                }
            }
            match next {
                Some(candidate) => selected = candidate,
                None => break,
            }
        }

        let mut removed = 0usize;
        let mut added = 0usize;
        for hit in &matches {
            removed += hit.end - hit.start;
            added += self.rules[hit.rule].replacement.len();
        }
        let mut scrubbed = String::with_capacity(text.len() - removed + added);
        let mut cursor = 0usize;
        for hit in &matches {
            scrubbed.push_str(&text[cursor..hit.start]);
            scrubbed.push_str(&self.rules[hit.rule].replacement);
            cursor = hit.end;
        }
        scrubbed.push_str(&text[cursor..]);
        Ok(Some(ScrubHit {
            text: scrubbed,
            matches,
        }))
    }

    /// Scrub one owned string field in place, recording the hit.
    ///
    /// # Errors
    ///
    /// The [`Self::scrub_str`] error; the field is left untouched.
    fn scrub_field(
        &self,
        target: &mut String,
        field: &'static str,
        index: usize,
        tally: &mut HookTally,
    ) -> Result<(), PluginError> {
        if let Some(hit) = self.scrub(target)? {
            self.record(&hit, field, index, tally);
            *target = hit.text;
        }
        Ok(())
    }

    /// Scrub every string leaf of a `ToolCall::arguments` value in place.
    ///
    /// Object *keys* are part of the tool's input contract and are never
    /// rewritten. Depth is already bounded by the `on_request` pre-pass, which
    /// runs over every tool call before the first rewrite.
    ///
    /// # Errors
    ///
    /// The [`Self::scrub_str`] error.
    fn scrub_json(
        &self,
        value: &mut serde_json::Value,
        index: usize,
        tally: &mut HookTally,
    ) -> Result<(), PluginError> {
        match value {
            serde_json::Value::String(text) => {
                self.scrub_field(text, FIELD_TOOL_CALL_ARGUMENTS, index, tally)
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    self.scrub_json(item, index, tally)?;
                }
                Ok(())
            }
            serde_json::Value::Object(map) => {
                for leaf in map.values_mut() {
                    self.scrub_json(leaf, index, tally)?;
                }
                Ok(())
            }
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
                Ok(())
            }
        }
    }

    /// Emit one structural event per replacement and fold the hit into `tally`.
    ///
    /// The event carries the rule's `kind`, the field label, and `index` — the
    /// message's position in `messages`, or the tool's position in `tools` for
    /// the `tool_description` field. Never the matched bytes.
    fn record(&self, hit: &ScrubHit, field: &'static str, index: usize, tally: &mut HookTally) {
        for applied in &hit.matches {
            tracing::warn!(
                target: "cuca::redaction",
                kind = %self.rules[applied.rule].kind,
                field = field,
                message_index = index,
                "outbound value redacted"
            );
        }
        tally.count += hit.matches.len();
        if let Some(last) = hit.matches.last() {
            tally.event = Some((
                self.rules[last.rule].kind.to_string(),
                field,
                hit.matches.len(),
            ));
        }
    }
}

impl CompiledRule {
    /// Build the searcher and the replacement token once, at construction.
    fn compile(rule: &RedactionRule) -> Self {
        let matcher = match rule {
            RedactionRule::Literal { value, .. } => {
                CompiledMatcher::Literal(Finder::new(value.as_bytes()).into_owned())
            }
            RedactionRule::Prefixed {
                prefix,
                min_len,
                max_len,
                ..
            } => CompiledMatcher::Prefixed {
                finder: Finder::new(prefix.as_bytes()).into_owned(),
                min_len: *min_len,
                max_len: *max_len,
            },
            RedactionRule::EmailLike { .. } => CompiledMatcher::EmailLike,
            RedactionRule::DigitRun {
                min_digits,
                max_digits,
                ..
            } => CompiledMatcher::DigitRun {
                min_digits: *min_digits,
                max_digits: *max_digits,
            },
        };
        let kind = rule.kind();
        Self {
            replacement: format!("{}{kind}]", RedactionConfig::REDACTION_MARKER).into_boxed_str(),
            kind: kind.into(),
            matcher,
        }
    }
}

impl CompiledMatcher {
    /// Leftmost match at or after `from`, as a `(start, end)` byte range, or
    /// `None` when this rule has no further match. The longest run wins at a
    /// given start.
    fn find_from(&self, text: &str, from: usize) -> Option<(usize, usize)> {
        let bytes = text.as_bytes();
        match self {
            CompiledMatcher::Literal(finder) => {
                let hit = from + finder.find(&bytes[from..])?;
                Some((hit, hit + finder.needle().len()))
            }
            CompiledMatcher::Prefixed {
                finder,
                min_len,
                max_len,
            } => {
                let prefix_len = finder.needle().len();
                let mut cursor = from;
                while let Some(hit) = finder.find(&bytes[cursor..]) {
                    let start = cursor + hit;
                    let body = start + prefix_len;
                    let mut end = body;
                    while end < bytes.len() && end - body < *max_len && is_token_byte(bytes[end]) {
                        end += 1;
                    }
                    if end - body >= *min_len {
                        return Some((start, end));
                    }
                    // Too short to be a token: keep looking, allowing an
                    // overlapping occurrence of the prefix itself.
                    cursor = start + 1;
                }
                None
            }
            CompiledMatcher::EmailLike => find_email(bytes, from),
            CompiledMatcher::DigitRun {
                min_digits,
                max_digits,
            } => find_digit_run(bytes, from, *min_digits, *max_digits),
        }
    }
}

/// Leftmost-longest with a policy-order tie-break: a candidate wins only by
/// starting earlier, or by being strictly longer at the same start. Rules are
/// offered in policy order, so the earlier rule keeps an exact tie.
fn preferred(current: Option<Match>, span: (usize, usize), rule: usize) -> Match {
    let candidate = Match {
        start: span.0,
        end: span.1,
        rule,
    };
    match current {
        Some(best)
            if best.start < candidate.start
                || (best.start == candidate.start && best.end >= candidate.end) =>
        {
            best
        }
        _ => candidate,
    }
}

/// A byte from the [`RedactionRule::Prefixed`] token alphabet.
fn is_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'
}

/// A byte allowed in an email local part.
fn is_email_local_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'%' | b'+' | b'-')
}

/// A byte allowed in an email domain.
fn is_email_domain_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-')
}

/// Leftmost `local@domain.tld`-shaped ASCII run at or after `from`.
///
/// The local part is expanded leftward from the `@` but never past `from`, so a
/// match can never reach back into an already-replaced region. It must open on
/// an alphanumeric, so a leading separator (`.foo@bar.baz`) stays outside the
/// match. The domain must hold a dot that is not its first byte, and is trimmed
/// of a trailing `.`/`-` so a match never ends on a separator.
///
/// Every match boundary is an ASCII byte, hence a `char` boundary.
fn find_email(bytes: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut cursor = from;
    while let Some(hit) = memchr::memchr(b'@', &bytes[cursor..]) {
        let at = cursor + hit;
        let mut start = at;
        while start > from && is_email_local_byte(bytes[start - 1]) {
            start -= 1;
        }
        while start < at && !bytes[start].is_ascii_alphanumeric() {
            start += 1;
        }
        let mut end = at + 1;
        while end < bytes.len() && is_email_domain_byte(bytes[end]) {
            end += 1;
        }
        while end > at + 1 && !bytes[end - 1].is_ascii_alphanumeric() {
            end -= 1;
        }
        let domain = &bytes[at + 1..end];
        if start < at && domain.len() >= 3 && domain[0] != b'.' && domain.contains(&b'.') {
            return Some((start, end));
        }
        cursor = at + 1;
    }
    None
}

/// Leftmost run of `min_digits..=max_digits` ASCII digits at or after `from`,
/// tolerating a single `-` or ` ` between digits.
///
/// The run ends on its last digit, never on a separator. Reaching the
/// `max_digits`-th digit returns immediately: `max_digits >= min_digits` is
/// validated, so the cap is already a match and there is no reason to walk the
/// rest of a long digit string only to truncate it. A run that falls short of
/// `min_digits` is skipped whole, since every run starting inside it is a strict
/// suffix and therefore shorter.
///
/// Every match boundary is an ASCII digit, hence a `char` boundary.
fn find_digit_run(
    bytes: &[u8],
    from: usize,
    min_digits: usize,
    max_digits: usize,
) -> Option<(usize, usize)> {
    let mut start = from;
    while start < bytes.len() {
        if !bytes[start].is_ascii_digit() {
            start += 1;
            continue;
        }
        let mut cursor = start;
        let mut digits = 0usize;
        let mut run_end = start;
        while cursor < bytes.len() {
            if bytes[cursor].is_ascii_digit() {
                digits += 1;
                cursor += 1;
                run_end = cursor;
                if digits == max_digits {
                    return Some((start, cursor));
                }
            } else if matches!(bytes[cursor], b'-' | b' ')
                && bytes.get(cursor + 1).is_some_and(u8::is_ascii_digit)
            {
                cursor += 1;
            } else {
                break;
            }
        }
        if digits >= min_digits {
            return Some((start, run_end));
        }
        start = run_end;
    }
    None
}

/// Refuse a `ToolCall::arguments` value nested deeper than
/// [`RedactionConfig::MAX_JSON_DEPTH`] containers.
///
/// The bound is checked at frame entry, so the recursion stops one frame past
/// the cap instead of riding an adversarial `Value` down the stack.
///
/// # Errors
///
/// [`PluginError::HookFailure`] (`stage: "request"`) past the depth bound.
fn check_json_depth(value: &serde_json::Value, depth: usize) -> Result<(), PluginError> {
    if depth > RedactionConfig::MAX_JSON_DEPTH {
        return Err(PluginError::hook(
            NAME,
            "request",
            format!(
                "tool_call_arguments nests deeper than MAX_JSON_DEPTH ({}); the request is refused \
                 rather than partially scrubbed",
                RedactionConfig::MAX_JSON_DEPTH
            ),
        ));
    }
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                check_json_depth(item, depth + 1)?;
            }
        }
        serde_json::Value::Object(map) => {
            for leaf in map.values() {
                check_json_depth(leaf, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

impl CucaPlugin for RedactionPlugin {
    fn name(&self) -> &'static str {
        NAME
    }

    /// Scrub every field in the module header's field map, in message order.
    ///
    /// # Errors
    ///
    /// [`PluginError::HookFailure`] (`stage: "request"`) for a `ToolCall`
    /// `arguments` value nested past [`RedactionConfig::MAX_JSON_DEPTH`], or a
    /// string holding more than [`RedactionPlugin::match_cap`] matches. Either
    /// way the request is refused before provider dispatch and later plugins'
    /// hooks never run. The depth bound is checked over the whole request
    /// before the first rewrite, so that refusal leaves the request untouched;
    /// an over-cap refusal may leave earlier fields already scrubbed, which is
    /// strictly safer than the input and never reaches the wire.
    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        for msg in &req.messages {
            for block in &msg.content {
                if let MessageContentBlock::ToolCall { arguments, .. } = block {
                    check_json_depth(arguments, 0)?;
                }
            }
        }

        let mut tally = HookTally::default();
        for (index, msg) in req.messages.iter_mut().enumerate() {
            if let Some(name) = msg.name.as_mut() {
                self.scrub_field(name, FIELD_MESSAGE_NAME, index, &mut tally)?;
            }
            for block in &mut msg.content {
                match block {
                    MessageContentBlock::Text(text) => {
                        self.scrub_field(text, FIELD_MESSAGE_TEXT, index, &mut tally)?;
                    }
                    MessageContentBlock::Thinking { reasoning, .. } => {
                        self.scrub_field(reasoning, FIELD_MESSAGE_THINKING, index, &mut tally)?;
                    }
                    MessageContentBlock::ToolCall { arguments, .. } => {
                        self.scrub_json(arguments, index, &mut tally)?;
                    }
                    MessageContentBlock::ToolResult { output, .. } => {
                        self.scrub_field(output, FIELD_TOOL_RESULT_OUTPUT, index, &mut tally)?;
                    }
                    // A MIME label and a base64 payload: see the module
                    // header's not-scrubbed list.
                    MessageContentBlock::ImageBase64 { .. } => {}
                }
            }
        }
        if self.scrub_tool_definitions {
            for (index, tool) in req.tools.iter_mut().enumerate() {
                self.scrub_field(
                    &mut tool.description,
                    FIELD_TOOL_DESCRIPTION,
                    index,
                    &mut tally,
                )?;
            }
        }

        let applied = tally.count as u64;
        let mut stats = self.stats();
        stats.last_request = applied;
        stats.total = stats.total.saturating_add(applied);
        if let Some(event) = tally.event {
            stats.last_event = Some(event);
        }
        Ok(())
    }

    // `execute_local_tool`, `on_stream_chunk`, `on_response_complete`: trait
    // defaults. See the module header's *Inbound text is not scrubbed*.
}

#[cfg(all(test, feature = "plugin-redaction"))]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::types::{MessageRole, ToolDefinition, UnifiedMessage};

    /// The fake secret every fixture carries.
    const SECRET: &str = "sk-live-4242";

    fn literal(kind: &str, value: &str) -> RedactionRule {
        RedactionRule::Literal {
            kind: kind.to_string(),
            value: value.to_string(),
        }
    }

    fn plugin_with(rules: Vec<RedactionRule>) -> RedactionPlugin {
        RedactionPlugin::new(RedactionConfig::new(rules).expect("policy must validate"))
            .expect("plugin must build")
    }

    fn block_text(req: &UnifiedRequest, message: usize, block: usize) -> String {
        match &req.messages[message].content[block] {
            MessageContentBlock::Text(text) => text.clone(),
            other => panic!("expected a Text block, got {other:?}"),
        }
    }

    /// A request touching every field in the map and every field deliberately
    /// left alone, each carrying [`SECRET`].
    fn kitchen_sink_request() -> UnifiedRequest {
        UnifiedRequest::new(format!("model-{SECRET}"))
            .add_message(UnifiedMessage {
                role: MessageRole::Assistant,
                content: vec![
                    MessageContentBlock::Text(format!("text {SECRET}")),
                    MessageContentBlock::Thinking {
                        reasoning: format!("reasoning {SECRET}"),
                        signature: Some(format!("signature {SECRET}")),
                    },
                    MessageContentBlock::ImageBase64 {
                        media_type: "image/png".to_string(),
                        data: format!("data {SECRET}"),
                    },
                    MessageContentBlock::ToolCall {
                        id: format!("call-{SECRET}"),
                        name: format!("tool-{SECRET}"),
                        arguments: serde_json::json!({ "path": format!("arg {SECRET}") }),
                    },
                    MessageContentBlock::ToolResult {
                        tool_call_id: format!("call-{SECRET}"),
                        output: format!("output {SECRET}"),
                    },
                ],
                name: Some(format!("name {SECRET}")),
                tool_call_id: Some(format!("call-{SECRET}")),
            })
            .add_tool(ToolDefinition {
                name: format!("tool-{SECRET}"),
                description: format!("description {SECRET}"),
                input_schema: serde_json::json!({ "const": SECRET }),
            })
    }

    // 1
    #[test]
    fn literal_rule_replaces_every_occurrence_in_message_text() {
        let plugin = plugin_with(vec![literal("api-key", SECRET)]);
        let mut req = UnifiedRequest::new("m").add_user_message(format!("a {SECRET} b {SECRET} c"));

        plugin.on_request(&mut req).expect("hook must succeed");

        assert_eq!(
            block_text(&req, 0, 0),
            "a [REDACTED:api-key] b [REDACTED:api-key] c"
        );
        assert_eq!(plugin.last_request_redactions(), 2);
    }

    // 2
    #[test]
    fn a_clean_string_is_borrowed_and_counts_nothing() {
        let plugin = plugin_with(vec![literal("api-key", SECRET)]);

        let redacted = plugin
            .scrub_str("nothing sensitive in here")
            .expect("scrub must succeed");

        assert!(
            matches!(redacted.text, Cow::Borrowed(_)),
            "a clean string must be handed back borrowed, not reallocated"
        );
        assert_eq!(redacted.count, 0);
    }

    // 3
    #[test]
    fn prefixed_rule_stops_at_the_first_byte_outside_the_token_alphabet() {
        let plugin = plugin_with(vec![RedactionRule::Prefixed {
            kind: "token".to_string(),
            prefix: "sk-".to_string(),
            min_len: 4,
            max_len: 32,
        }]);

        let redacted = plugin
            .scrub_str("key sk-abc_DEF-123! trailing")
            .expect("scrub must succeed");

        assert_eq!(redacted.text, "key [REDACTED:token]! trailing");
        assert_eq!(redacted.count, 1);
    }

    // 4
    #[test]
    fn prefixed_rule_honors_min_len_and_truncates_at_max_len() {
        let plugin = plugin_with(vec![RedactionRule::Prefixed {
            kind: "token".to_string(),
            prefix: "sk-".to_string(),
            min_len: 4,
            max_len: 6,
        }]);

        let short = plugin
            .scrub_str("sk-ab is too short")
            .expect("scrub must succeed");
        assert_eq!(short.count, 0, "a token under min_len is not a match");
        assert_eq!(short.text, "sk-ab is too short");

        let long = plugin
            .scrub_str("sk-abcdefghij")
            .expect("scrub must succeed");
        assert_eq!(
            long.text, "[REDACTED:token]ghij",
            "the match truncates at max_len token bytes"
        );
        assert_eq!(long.count, 1);
    }

    // 5
    #[test]
    fn email_like_needs_a_dotted_domain() {
        let plugin = plugin_with(vec![RedactionRule::EmailLike {
            kind: "email".to_string(),
        }]);

        let hit = plugin
            .scrub_str("mail a.b+c@example.co now")
            .expect("scrub must succeed");
        assert_eq!(hit.text, "mail [REDACTED:email] now");

        let miss = plugin
            .scrub_str("not-an-email@ has no domain")
            .expect("scrub must succeed");
        assert_eq!(miss.count, 0);
        assert_eq!(miss.text, "not-an-email@ has no domain");
    }

    // 6
    #[test]
    fn digit_run_redacts_a_separated_card_and_leaves_a_year_alone() {
        let plugin = plugin_with(vec![RedactionRule::DigitRun {
            kind: "card".to_string(),
            min_digits: 13,
            max_digits: 19,
        }]);

        let redacted = plugin
            .scrub_str("card 4111-1111-1111-1111 in 2024")
            .expect("scrub must succeed");

        assert_eq!(redacted.text, "card [REDACTED:card] in 2024");
        assert_eq!(redacted.count, 1);
    }

    // 7
    #[test]
    fn overlaps_resolve_leftmost_longest_then_by_policy_order() {
        let longest = plugin_with(vec![
            literal("short", "secret"),
            literal("long", "secret-value"),
        ]);
        assert_eq!(
            longest
                .scrub_str("a secret-value b")
                .expect("scrub must succeed")
                .text,
            "a [REDACTED:long] b",
            "the longer match wins at a shared start"
        );

        let leftmost = plugin_with(vec![
            literal("later", "value-tail"),
            literal("earlier", "secret"),
        ]);
        assert_eq!(
            leftmost
                .scrub_str("a secret-value-tail b")
                .expect("scrub must succeed")
                .text,
            "a [REDACTED:earlier]-[REDACTED:later] b",
            "leftmost beats longer-but-later, whatever the policy order"
        );

        let tied = plugin_with(vec![literal("first", "dup"), literal("second", "dup")]);
        assert_eq!(
            tied.scrub_str("x dup y").expect("scrub must succeed").text,
            "x [REDACTED:first] y",
            "an exact tie goes to the earlier rule in policy order"
        );
    }

    // 8
    #[test]
    fn on_request_scrubs_every_field_in_the_map() {
        let plugin = plugin_with(vec![literal("api-key", SECRET)]);
        let mut req = kitchen_sink_request();

        plugin.on_request(&mut req).expect("hook must succeed");

        let msg = &req.messages[0];
        assert_eq!(msg.name.as_deref(), Some("name [REDACTED:api-key]"));
        assert_eq!(block_text(&req, 0, 0), "text [REDACTED:api-key]");
        match &msg.content[1] {
            MessageContentBlock::Thinking { reasoning, .. } => {
                assert_eq!(reasoning, "reasoning [REDACTED:api-key]");
            }
            other => panic!("expected a Thinking block, got {other:?}"),
        }
        match &msg.content[4] {
            MessageContentBlock::ToolResult { output, .. } => {
                assert_eq!(output, "output [REDACTED:api-key]");
            }
            other => panic!("expected a ToolResult block, got {other:?}"),
        }
        assert_eq!(req.tools[0].description, "description [REDACTED:api-key]");
    }

    // 9
    #[test]
    fn on_request_leaves_every_non_text_field_byte_identical() {
        let plugin = plugin_with(vec![literal("api-key", SECRET)]);
        let mut req = kitchen_sink_request();

        plugin.on_request(&mut req).expect("hook must succeed");

        assert_eq!(req.model, format!("model-{SECRET}"), "routing knob");
        let msg = &req.messages[0];
        assert_eq!(
            msg.tool_call_id.as_deref(),
            Some(format!("call-{SECRET}").as_str()),
            "correlation id"
        );
        match &msg.content[1] {
            MessageContentBlock::Thinking { signature, .. } => assert_eq!(
                signature.as_deref(),
                Some(format!("signature {SECRET}").as_str()),
                "rewriting a provider signature invalidates it"
            ),
            other => panic!("expected a Thinking block, got {other:?}"),
        }
        match &msg.content[2] {
            MessageContentBlock::ImageBase64 { media_type, data } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(data, &format!("data {SECRET}"), "base64 payload");
            }
            other => panic!("expected an ImageBase64 block, got {other:?}"),
        }
        match &msg.content[3] {
            MessageContentBlock::ToolCall { id, name, .. } => {
                assert_eq!(id, &format!("call-{SECRET}"), "correlation id");
                assert_eq!(name, &format!("tool-{SECRET}"), "tool identity");
            }
            other => panic!("expected a ToolCall block, got {other:?}"),
        }
        match &msg.content[4] {
            MessageContentBlock::ToolResult { tool_call_id, .. } => {
                assert_eq!(tool_call_id, &format!("call-{SECRET}"), "correlation id");
            }
            other => panic!("expected a ToolResult block, got {other:?}"),
        }
        assert_eq!(req.tools[0].name, format!("tool-{SECRET}"), "tool identity");
        assert_eq!(
            req.tools[0].input_schema,
            serde_json::json!({ "const": SECRET }),
            "a schema keyword is the tool's contract"
        );
    }

    // 10
    #[test]
    fn on_request_scrubs_tool_call_argument_string_leaves_but_not_keys() {
        let plugin = plugin_with(vec![literal("api-key", SECRET)]);
        let mut req = UnifiedRequest::new("m").add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: "call-1".to_string(),
                name: "write".to_string(),
                arguments: serde_json::json!({
                    SECRET: format!("top {SECRET}"),
                    "nested": {
                        "deep": [format!("in-array {SECRET}"), 7, null, { "leaf": SECRET }]
                    }
                }),
            }],
            name: None,
            tool_call_id: None,
        });

        plugin.on_request(&mut req).expect("hook must succeed");

        let arguments = match &req.messages[0].content[0] {
            MessageContentBlock::ToolCall { arguments, .. } => arguments,
            other => panic!("expected a ToolCall block, got {other:?}"),
        };
        assert_eq!(
            arguments,
            &serde_json::json!({
                SECRET: "top [REDACTED:api-key]",
                "nested": {
                    "deep": ["in-array [REDACTED:api-key]", 7, null, { "leaf": "[REDACTED:api-key]" }]
                }
            }),
            "string leaves at every level are scrubbed; object keys are the tool's contract"
        );
        assert_eq!(plugin.last_request_redactions(), 3);
    }

    // 11
    #[test]
    fn arguments_past_the_depth_bound_refuse_the_whole_request() {
        let plugin = plugin_with(vec![literal("api-key", SECRET)]);
        let mut arguments = serde_json::Value::String(format!("deep {SECRET}"));
        for _ in 0..=RedactionConfig::MAX_JSON_DEPTH {
            arguments = serde_json::Value::Array(vec![arguments]);
        }
        let mut req = UnifiedRequest::new("m").add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![
                MessageContentBlock::Text(format!("text {SECRET}")),
                MessageContentBlock::ToolCall {
                    id: "call-1".to_string(),
                    name: "write".to_string(),
                    arguments,
                },
            ],
            name: None,
            tool_call_id: None,
        });
        let before = req.clone();

        let err = plugin
            .on_request(&mut req)
            .expect_err("an over-deep arguments value must refuse the request");

        assert!(
            matches!(
                err,
                PluginError::HookFailure {
                    plugin: "redaction",
                    stage: "request",
                    ..
                }
            ),
            "expected a request-stage hook failure, got {err:?}"
        );
        assert_eq!(
            req, before,
            "the depth bound is checked before the first rewrite, so nothing is partially scrubbed"
        );
    }

    // 12
    #[test]
    fn a_string_over_the_match_cap_is_refused_instead_of_truncated() {
        let plugin = RedactionPlugin::new(RedactionConfig {
            rules: vec![literal("api-key", SECRET)],
            max_matches_per_text: 2,
            ..Default::default()
        })
        .expect("plugin must build");
        assert_eq!(plugin.match_cap(), 2);

        assert_eq!(
            plugin
                .scrub_str(&format!("{SECRET} {SECRET}"))
                .expect("exactly the cap must still scrub")
                .count,
            2
        );

        let err = plugin
            .scrub_str(&format!("{SECRET} {SECRET} {SECRET}"))
            .expect_err("over the cap must refuse");
        assert!(
            matches!(
                err,
                PluginError::HookFailure {
                    plugin: "redaction",
                    stage: "request",
                    ..
                }
            ),
            "expected a request-stage hook failure, got {err:?}"
        );
    }

    // 13
    #[test]
    fn an_empty_policy_is_rejected() {
        let err = RedactionConfig::new(Vec::new()).expect_err("an empty policy is not a policy");

        match err {
            PluginError::Validation { schema, message } => {
                assert_eq!(schema, "redaction-config");
                assert!(message.contains("rules"), "message was {message:?}");
            }
            other => panic!("expected a validation rejection, got {other:?}"),
        }
    }

    // 14
    #[test]
    fn validate_rejects_every_out_of_bounds_policy() {
        fn rejects(config: RedactionConfig, needle: &str) {
            match config.validate() {
                Err(PluginError::Validation { schema, message }) => {
                    assert_eq!(schema, "redaction-config");
                    assert!(
                        message.contains(needle),
                        "message {message:?} must mention {needle:?}"
                    );
                }
                other => panic!("expected a validation rejection for {needle:?}, got {other:?}"),
            }
        }
        fn config(rules: Vec<RedactionRule>) -> RedactionConfig {
            RedactionConfig {
                rules,
                ..Default::default()
            }
        }
        fn prefixed(min_len: usize, max_len: usize) -> RedactionRule {
            RedactionRule::Prefixed {
                kind: "token".to_string(),
                prefix: "sk-".to_string(),
                min_len,
                max_len,
            }
        }
        fn digits(min_digits: usize, max_digits: usize) -> RedactionRule {
            RedactionRule::DigitRun {
                kind: "card".to_string(),
                min_digits,
                max_digits,
            }
        }

        rejects(
            config(vec![
                literal("api-key", SECRET);
                RedactionConfig::MAX_RULES + 1
            ]),
            "MAX_RULES",
        );
        rejects(config(vec![literal("api-key", "")]), "value");
        rejects(
            config(vec![literal(
                "api-key",
                &"a".repeat(RedactionConfig::MAX_PATTERN_BYTES + 1),
            )]),
            "MAX_PATTERN_BYTES",
        );
        rejects(config(vec![literal("", SECRET)]), "kind");
        rejects(config(vec![literal("Api-Key", SECRET)]), "slug");
        rejects(config(vec![literal("api key", SECRET)]), "slug");
        rejects(
            config(vec![literal(
                &"a".repeat(RedactionConfig::MAX_KIND_BYTES + 1),
                SECRET,
            )]),
            "MAX_KIND_BYTES",
        );
        rejects(
            config(vec![literal("api-key", "[REDACTED:api-key]")]),
            "marker",
        );
        rejects(config(vec![prefixed(9, 4)]), "min_len");
        rejects(config(vec![digits(0, 19)]), "min_digits");
        rejects(config(vec![digits(19, 13)]), "min_digits");
        rejects(
            RedactionConfig {
                rules: vec![literal("api-key", SECRET)],
                max_matches_per_text: 0,
                ..Default::default()
            },
            "max_matches_per_text",
        );
        rejects(
            RedactionConfig {
                rules: vec![literal("api-key", SECRET)],
                max_matches_per_text: RedactionConfig::MAX_MATCHES_PER_TEXT + 1,
                ..Default::default()
            },
            "MAX_MATCHES_PER_TEXT",
        );
    }

    // 15
    #[test]
    fn last_request_is_per_request_while_the_total_accumulates() {
        let plugin = plugin_with(vec![literal("api-key", SECRET)]);
        assert_eq!(plugin.last_request_redactions(), 0);
        assert_eq!(plugin.total_redactions(), 0);
        assert_eq!(plugin.rule_count(), 1);

        let mut two = UnifiedRequest::new("m").add_user_message(format!("{SECRET} {SECRET}"));
        plugin.on_request(&mut two).expect("hook must succeed");
        assert_eq!(plugin.last_request_redactions(), 2);
        assert_eq!(plugin.total_redactions(), 2);

        let mut one = UnifiedRequest::new("m").add_user_message(format!("only {SECRET}"));
        plugin.on_request(&mut one).expect("hook must succeed");
        assert_eq!(plugin.last_request_redactions(), 1);
        assert_eq!(plugin.total_redactions(), 3);

        let mut clean = UnifiedRequest::new("m").add_user_message("nothing to redact");
        plugin.on_request(&mut clean).expect("hook must succeed");
        assert_eq!(
            plugin.last_request_redactions(),
            0,
            "the per-request counter reflects only the most recent request"
        );
        assert_eq!(plugin.total_redactions(), 3);
    }

    // 16
    #[test]
    fn the_last_event_names_the_kind_and_never_the_matched_value() {
        let plugin = plugin_with(vec![literal("api-key", SECRET)]);
        assert!(plugin.last_redaction_event().is_none());

        let mut req = UnifiedRequest::new("m").add_user_message(format!("a {SECRET} b {SECRET}"));
        plugin.on_request(&mut req).expect("hook must succeed");

        let (kind, field, count) = plugin
            .last_redaction_event()
            .expect("a redaction must record an event");
        assert_eq!(kind, "api-key");
        assert_eq!(field, "message_text");
        assert_eq!(count, 2);
        assert!(
            !kind.contains(SECRET),
            "the event names the secret class, never the secret"
        );
    }

    // 17
    #[test]
    fn name_and_send_sync() {
        let plugin: Arc<dyn CucaPlugin> = Arc::new(plugin_with(vec![literal("api-key", SECRET)]));

        assert_eq!(plugin.name(), "redaction");
    }

    // 18
    #[test]
    fn non_ascii_text_around_a_match_survives_byte_identically() {
        let plugin = plugin_with(vec![
            literal("api-key", SECRET),
            RedactionRule::EmailLike {
                kind: "email".to_string(),
            },
        ]);

        let text = format!("héllo 🌍 {SECRET} — señor a@b.co ✅");
        let redacted = plugin.scrub_str(&text).expect("scrub must succeed");

        assert_eq!(
            redacted.text,
            "héllo 🌍 [REDACTED:api-key] — señor [REDACTED:email] ✅"
        );
        assert_eq!(redacted.count, 2);
    }

    // 19
    #[test]
    fn on_stream_chunk_is_the_documented_no_op() {
        let plugin = plugin_with(vec![literal("api-key", SECRET)]);
        let mut chunk = MessageContentBlock::Text(format!("the model said {SECRET}"));
        let before = chunk.clone();

        plugin
            .on_stream_chunk(&mut chunk)
            .expect("the default hook must succeed");

        assert_eq!(
            chunk, before,
            "inbound scrubbing is deferred; implementing it later must be deliberate"
        );
    }

    // 20
    #[test]
    fn scrubbing_is_idempotent_across_two_passes() {
        let plugin = plugin_with(vec![
            literal("api-key", SECRET),
            RedactionRule::EmailLike {
                kind: "email".to_string(),
            },
            RedactionRule::DigitRun {
                kind: "card".to_string(),
                min_digits: 13,
                max_digits: 19,
            },
        ]);
        let mut req = UnifiedRequest::new("m")
            .add_user_message(format!("{SECRET} a@b.co 4111-1111-1111-1111"));

        plugin.on_request(&mut req).expect("hook must succeed");
        let once = req.clone();
        assert_eq!(plugin.last_request_redactions(), 3);

        plugin.on_request(&mut req).expect("hook must succeed");

        assert_eq!(
            plugin.last_request_redactions(),
            0,
            "a replacement token cannot re-match, so the second pass is a no-op"
        );
        assert_eq!(req, once);
        assert_eq!(plugin.total_redactions(), 3);
    }

    // 21
    #[test]
    fn scrub_tool_definitions_gates_only_the_tool_description() {
        let rules = vec![literal("api-key", SECRET)];
        let off = RedactionPlugin::new(RedactionConfig {
            rules: rules.clone(),
            scrub_tool_definitions: false,
            ..Default::default()
        })
        .expect("plugin must build");
        let on = plugin_with(rules);

        let mut left_alone = kitchen_sink_request();
        off.on_request(&mut left_alone).expect("hook must succeed");
        assert_eq!(
            left_alone.tools[0].description,
            format!("description {SECRET}"),
            "the knob is off, so an author-written description is untouched"
        );
        assert_eq!(
            block_text(&left_alone, 0, 0),
            "text [REDACTED:api-key]",
            "message text is scrubbed either way"
        );

        let mut scrubbed = kitchen_sink_request();
        on.on_request(&mut scrubbed).expect("hook must succeed");
        assert_eq!(
            scrubbed.tools[0].description,
            "description [REDACTED:api-key]"
        );
    }
}
