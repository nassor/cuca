//! Append-only session trajectory store.
//!
//! [`SessionLogPlugin`] implements the [`SessionStorePlugin`] contract on the
//! [`SessionEvent`] model: every interaction class, system instructions,
//! reasoning, outputs, tool executions (with stdout/stderr/exit codes when a
//! block carries them), latency, token usage, and model swaps, is recorded as
//! one append-only [`SessionRecord`] per session.
//!
//! # Append-only paradigm
//!
//! Records are only ever *added* to a session's trajectory; nothing is rewritten
//! or removed. Each session's records carry a 0-based `sequence` assigned by the
//! store on append, so replay order is the append order. [`SessionBackend`] is
//! the storage seam; this module ships a capped in-memory [`InMemoryBackend`]
//! and an append-only framed file backend [`FileBackend`].
//!
//! # Per-session sequencing
//!
//! The plugin is authoritative over sequence numbers: [`SessionLogPlugin`]
//! tracks a per-session next-sequence map and re-sequences every record on
//! `append_log`. Backends handle raw [`SessionRecord`]s and never assign
//! sequences themselves: the one exception is the audit [`SessionEvent::Fork`]
//! record a backend appends during `fork`, which is sequenced at the session's
//! current tail (the store-authoritative rule, since `fork` receives no sequence
//! from the caller).
//!
//! # Fork semantics
//!
//! [`SessionStorePlugin::fork_session`] branches from any historical `point_id`
//! (`"{session_id}:{sequence}"`). The new session's trajectory is the prefix of
//! the original up to and including the fork point, re-labelled with the new
//! session id so its records belong to the branch; the original session gains a
//! [`SessionEvent::Fork`] record for auditability. After a successful fork the
//! plugin realigns the original session's next-sequence counter to its replayed
//! length so later appends cannot collide with the audit record.
//!
//! # File layout
//!
//! [`FileBackend`] stores one session per file `{dir}/{session_id}.cslog`. Each
//! file opens with the 8-byte magic `CUCASLOG` and a one-byte format version,
//! followed by one COBS-framed postcard record per append. Files are opened
//! with `append(true)` (never truncated), so existing frames always survive
//! further appends. Session ids containing path separators are rejected rather
//! than silently mapped into subdirectories.
//!
//! # Why framed postcard
//!
//! postcard is not self-delimiting, so an append-only file needs framing. COBS
//! delimits every frame with a zero byte, which costs two bytes per record and
//! leaves a reader able to find the next frame boundary after a torn or
//! corrupt one; a length prefix cannot resynchronize. Records reach postcard
//! through the storage mirrors in this module, because postcard cannot decode
//! the internally tagged [`SessionEvent`] or the adjacently tagged
//! [`MessageContentBlock`], and cannot decode a `serde_json::Value` at all.
//! The mirrors are also the versioned storage schema: the format version in
//! the file header changes when their shape does, so a stale file is rejected
//! rather than misread.
//!
//! # Model-swap contract
//!
//! When both `plugin-speculative` and `plugin-session-log` are enabled, the
//! orchestrator calls [`SessionStorePlugin::append_log`] to record
//! [`SessionEvent::ModelSwap`] events against the registered store plugin.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

use crate::error::PluginError;
use crate::plugin::{CucaPlugin, SessionStorePlugin};
use crate::request::{UnifiedRequest, UnifiedResponse};
use crate::session::{SessionEvent, SessionRecord};
use crate::types::{MessageContentBlock, MessageRole};

/// The storage seam for append-only session trajectories.
///
/// A backend owns the durable representation of records and the fork operation;
/// it never invents sequence numbers except for the audit `Fork` record it
/// appends to the original session during `fork` (sequenced at the session's
/// current tail).
pub trait SessionBackend: Send + Sync {
    /// Persist one record. The record already carries its authoritative
    /// `sequence`; the backend must store it verbatim and never rewrite prior
    /// records.
    fn append(&self, record: &SessionRecord) -> Result<(), PluginError>;

    /// Replay the full trajectory in append order (ordered by `sequence`).
    fn replay(&self, session_id: &str) -> Result<Vec<SessionRecord>, PluginError>;

    /// Fork from a historical `point_id` (`"{session_id}:{sequence}"`), returning
    /// a NEW session id whose trajectory is the prefix of `session_id` up to and
    /// including `point_id`. The original session gains a [`SessionEvent::Fork`]
    /// record for auditability.
    fn fork(&self, session_id: &str, point_id: &str) -> Result<String, PluginError>;
}

/// In-memory trajectory store: `HashMap<session_id, Vec<SessionRecord>>`.
///
/// `append` pushes to the session's vector (append-only by construction); the
/// plugin assigns sequences. `fork` derives the branch in place and records the
/// audit `Fork` event on the original session. Not persisted; for a durable
/// store use [`FileBackend`].
///
/// Growth is capped: at most [`Self::max_records`] records in total across
/// sessions ([`Self::new`] uses [`Self::DEFAULT_MAX_RECORDS`];
/// [`Self::with_max_records`] validates a custom bound). At the cap, `append`
/// and `fork` fail with a [`PluginError`] instead of evicting: this is an
/// audit log, and dropping records would silently corrupt replay and fork.
/// [`Self::len`] is the O(1) usage gauge. For growth bounded by disk instead
/// of process memory, use [`FileBackend`].
pub struct InMemoryBackend {
    inner: Mutex<InMemoryStore>,
    fork_counter: Mutex<u64>,
    max_records: usize,
}

/// Mutex-guarded store: the session map plus a running total so the cap check
/// and the [`InMemoryBackend::len`] gauge are O(1), never a scan.
struct InMemoryStore {
    sessions: HashMap<String, Vec<SessionRecord>>,
    total: usize,
}

impl InMemoryBackend {
    /// Total-record cap used by [`Self::new`].
    pub const DEFAULT_MAX_RECORDS: usize = 65_536;

    /// Create an empty backend capped at [`Self::DEFAULT_MAX_RECORDS`] records.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(InMemoryStore {
                sessions: HashMap::new(),
                total: 0,
            }),
            fork_counter: Mutex::new(0),
            max_records: Self::DEFAULT_MAX_RECORDS,
        }
    }

    /// Create an empty backend holding at most `max_records` records in total
    /// across sessions.
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] when `max_records` is zero.
    pub fn with_max_records(max_records: usize) -> Result<Self, PluginError> {
        if max_records == 0 {
            return Err(PluginError::Validation {
                schema: "max_records".to_string(),
                message: "max_records must be non-zero".to_string(),
            });
        }
        let mut backend = Self::new();
        backend.max_records = max_records;
        Ok(backend)
    }

    /// Total records currently stored across sessions: the O(1) usage gauge
    /// against [`Self::max_records`].
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).total
    }

    /// Whether the store holds no records.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The configured total-record cap.
    pub fn max_records(&self) -> usize {
        self.max_records
    }

    /// The at-cap refusal for an operation needing `needed` more records.
    fn full_error(&self, needed: usize, total: usize) -> PluginError {
        PluginError::Internal(format!(
            "in-memory session log full: {total} of {} records stored and {needed} more needed; \
             raise max_records or use FileBackend for disk-bound growth",
            self.max_records
        ))
    }
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionBackend for InMemoryBackend {
    fn append(&self, record: &SessionRecord) -> Result<(), PluginError> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.total >= self.max_records {
            return Err(self.full_error(1, inner.total));
        }
        inner
            .sessions
            .entry(record.session_id.clone())
            .or_default()
            .push(record.clone());
        inner.total += 1;
        Ok(())
    }

    fn replay(&self, session_id: &str) -> Result<Vec<SessionRecord>, PluginError> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        Ok(inner.sessions.get(session_id).cloned().unwrap_or_default())
    }

    fn fork(&self, session_id: &str, point_id: &str) -> Result<String, PluginError> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let records = inner.sessions.get(session_id).cloned().unwrap_or_default();
        let index = records
            .iter()
            .position(|r| r.point_id() == point_id)
            .ok_or_else(|| PluginError::Validation {
                schema: "point_id".to_string(),
                message: format!("no record with point_id `{point_id}` in session `{session_id}`"),
            })?;

        // All-or-nothing cap check before any mutation: the branch prefix
        // (index + 1 records) plus the Fork audit record on the original.
        let needed = index + 2;
        if inner.total.saturating_add(needed) > self.max_records {
            return Err(self.full_error(needed, inner.total));
        }

        let mut counter = self.fork_counter.lock().unwrap_or_else(|p| p.into_inner());
        let n = *counter;
        *counter += 1;
        let new_id = format!("{session_id}:fork:{point_id}:{n}");

        // The new trajectory is the prefix up to and including the fork point,
        // re-labelled so every record belongs to the new session.
        let prefix: Vec<SessionRecord> = records[..=index]
            .iter()
            .map(|r| SessionRecord {
                session_id: new_id.clone(),
                ..r.clone()
            })
            .collect();
        let replaced = inner.sessions.insert(new_id.clone(), prefix);
        debug_assert!(
            replaced.is_none(),
            "fork ids are unique per backend: the fork counter is monotonic"
        );

        // Auditability: sequence the Fork record at the original's current tail.
        let fork_rec = SessionRecord {
            session_id: session_id.to_string(),
            sequence: records.len() as u64,
            timestamp_ms: now_ms(),
            event: SessionEvent::Fork {
                from_point: point_id.to_string(),
                to_session: new_id.clone(),
            },
        };
        if let Some(v) = inner.sessions.get_mut(session_id) {
            v.push(fork_rec);
        }
        inner.total += needed;
        Ok(new_id)
    }
}

/// Magic at the head of every [`FileBackend`] file.
const FILE_MAGIC: &[u8; 8] = b"CUCASLOG";

/// Storage schema version for the mirrors below, written after [`FILE_MAGIC`].
///
/// postcard encodes struct fields positionally and carries no field names, so
/// it has no schema evolution: any change to [`StoredRecord`], [`StoredEvent`],
/// or [`StoredBlock`] REQUIRES bumping this constant.
const FORMAT_VERSION: u8 = 1;

/// Bytes of file header: [`FILE_MAGIC`] plus [`FORMAT_VERSION`].
const HEADER_LEN: usize = FILE_MAGIC.len() + 1;

/// Scratch capacity [`FileBackend`] keeps between appends.
const MAX_RETAINED_SCRATCH: usize = 64 * 1024;

/// Storage mirror of [`MessageContentBlock`].
///
/// Externally tagged, so postcard can decode it, with the `Value` leaf carried
/// as canonical JSON text. The wire type is adjacently tagged and postcard
/// refuses to decode a variant identifier, so a mirror is the only way to
/// round-trip it without changing the public JSON representation.
#[derive(serde::Serialize, serde::Deserialize)]
enum StoredBlock {
    Text(String),
    ImageBase64 {
        media_type: String,
        data: String,
    },
    Thinking {
        reasoning: String,
        signature: Option<String>,
    },
    ToolCall {
        id: String,
        name: String,
        #[serde(with = "crate::canonical::value_as_canonical_json")]
        arguments: serde_json::Value,
    },
    ToolResult {
        tool_call_id: String,
        output: String,
    },
}

impl From<&MessageContentBlock> for StoredBlock {
    /// Exhaustive by construction: a new [`MessageContentBlock`] variant fails
    /// to compile here instead of silently dropping out of storage.
    fn from(block: &MessageContentBlock) -> Self {
        match block {
            MessageContentBlock::Text(text) => Self::Text(text.clone()),
            MessageContentBlock::ImageBase64 { media_type, data } => Self::ImageBase64 {
                media_type: media_type.clone(),
                data: data.clone(),
            },
            MessageContentBlock::Thinking {
                reasoning,
                signature,
            } => Self::Thinking {
                reasoning: reasoning.clone(),
                signature: signature.clone(),
            },
            MessageContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => Self::ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            },
            MessageContentBlock::ToolResult {
                tool_call_id,
                output,
            } => Self::ToolResult {
                tool_call_id: tool_call_id.clone(),
                output: output.clone(),
            },
        }
    }
}

impl From<StoredBlock> for MessageContentBlock {
    fn from(block: StoredBlock) -> Self {
        match block {
            StoredBlock::Text(text) => Self::Text(text),
            StoredBlock::ImageBase64 { media_type, data } => Self::ImageBase64 { media_type, data },
            StoredBlock::Thinking {
                reasoning,
                signature,
            } => Self::Thinking {
                reasoning,
                signature,
            },
            StoredBlock::ToolCall {
                id,
                name,
                arguments,
            } => Self::ToolCall {
                id,
                name,
                arguments,
            },
            StoredBlock::ToolResult {
                tool_call_id,
                output,
            } => Self::ToolResult {
                tool_call_id,
                output,
            },
        }
    }
}

/// Storage mirror of [`SessionEvent`]: externally tagged for postcard, with the
/// `Value` leaf carried as canonical JSON text.
#[derive(serde::Serialize, serde::Deserialize)]
enum StoredEvent {
    SystemPrompt {
        text: String,
    },
    Message {
        role: MessageRole,
        content: Vec<StoredBlock>,
    },
    Reasoning {
        reasoning: String,
        signature: Option<String>,
    },
    Output {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        #[serde(with = "crate::canonical::value_as_canonical_json")]
        arguments: serde_json::Value,
    },
    ToolResult {
        tool_call_id: String,
        output: String,
        stdout: Option<String>,
        stderr: Option<String>,
        exit_code: Option<i32>,
    },
    ModelSwap {
        from: String,
        to: String,
        reason: String,
    },
    Latency {
        duration_ms: u64,
    },
    TokenUsage {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    Fork {
        from_point: String,
        to_session: String,
    },
}

impl From<&SessionEvent> for StoredEvent {
    /// Exhaustive by construction: a new [`SessionEvent`] variant fails to
    /// compile here instead of silently dropping out of storage.
    fn from(event: &SessionEvent) -> Self {
        match event {
            SessionEvent::SystemPrompt { text } => Self::SystemPrompt { text: text.clone() },
            SessionEvent::Message { role, content } => Self::Message {
                role: *role,
                content: content.iter().map(StoredBlock::from).collect(),
            },
            SessionEvent::Reasoning {
                reasoning,
                signature,
            } => Self::Reasoning {
                reasoning: reasoning.clone(),
                signature: signature.clone(),
            },
            SessionEvent::Output { text } => Self::Output { text: text.clone() },
            SessionEvent::ToolCall {
                id,
                name,
                arguments,
            } => Self::ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            },
            SessionEvent::ToolResult {
                tool_call_id,
                output,
                stdout,
                stderr,
                exit_code,
            } => Self::ToolResult {
                tool_call_id: tool_call_id.clone(),
                output: output.clone(),
                stdout: stdout.clone(),
                stderr: stderr.clone(),
                exit_code: *exit_code,
            },
            SessionEvent::ModelSwap { from, to, reason } => Self::ModelSwap {
                from: from.clone(),
                to: to.clone(),
                reason: reason.clone(),
            },
            SessionEvent::Latency { duration_ms } => Self::Latency {
                duration_ms: *duration_ms,
            },
            SessionEvent::TokenUsage {
                prompt_tokens,
                completion_tokens,
            } => Self::TokenUsage {
                prompt_tokens: *prompt_tokens,
                completion_tokens: *completion_tokens,
            },
            SessionEvent::Fork {
                from_point,
                to_session,
            } => Self::Fork {
                from_point: from_point.clone(),
                to_session: to_session.clone(),
            },
        }
    }
}

impl From<StoredEvent> for SessionEvent {
    fn from(event: StoredEvent) -> Self {
        match event {
            StoredEvent::SystemPrompt { text } => Self::SystemPrompt { text },
            StoredEvent::Message { role, content } => Self::Message {
                role,
                content: content.into_iter().map(MessageContentBlock::from).collect(),
            },
            StoredEvent::Reasoning {
                reasoning,
                signature,
            } => Self::Reasoning {
                reasoning,
                signature,
            },
            StoredEvent::Output { text } => Self::Output { text },
            StoredEvent::ToolCall {
                id,
                name,
                arguments,
            } => Self::ToolCall {
                id,
                name,
                arguments,
            },
            StoredEvent::ToolResult {
                tool_call_id,
                output,
                stdout,
                stderr,
                exit_code,
            } => Self::ToolResult {
                tool_call_id,
                output,
                stdout,
                stderr,
                exit_code,
            },
            StoredEvent::ModelSwap { from, to, reason } => Self::ModelSwap { from, to, reason },
            StoredEvent::Latency { duration_ms } => Self::Latency { duration_ms },
            StoredEvent::TokenUsage {
                prompt_tokens,
                completion_tokens,
            } => Self::TokenUsage {
                prompt_tokens,
                completion_tokens,
            },
            StoredEvent::Fork {
                from_point,
                to_session,
            } => Self::Fork {
                from_point,
                to_session,
            },
        }
    }
}

/// Storage mirror of [`SessionRecord`].
#[derive(serde::Serialize, serde::Deserialize)]
struct StoredRecord {
    session_id: String,
    sequence: u64,
    timestamp_ms: u64,
    event: StoredEvent,
}

impl From<&SessionRecord> for StoredRecord {
    /// Destructures exhaustively: a new [`SessionRecord`] field fails to
    /// compile here instead of silently dropping out of storage.
    fn from(record: &SessionRecord) -> Self {
        let SessionRecord {
            session_id,
            sequence,
            timestamp_ms,
            event,
        } = record;
        Self {
            session_id: session_id.clone(),
            sequence: *sequence,
            timestamp_ms: *timestamp_ms,
            event: StoredEvent::from(event),
        }
    }
}

impl From<StoredRecord> for SessionRecord {
    fn from(record: StoredRecord) -> Self {
        let StoredRecord {
            session_id,
            sequence,
            timestamp_ms,
            event,
        } = record;
        Self {
            session_id,
            sequence,
            timestamp_ms,
            event: event.into(),
        }
    }
}

/// postcard storage flavor appending into a caller-owned buffer.
///
/// Lets [`FileBackend`] frame every record into one reusable scratch buffer
/// instead of allocating a fresh `Vec` per append. `Index`/`IndexMut` are
/// required by postcard's COBS flavor, which backpatches its header bytes, and
/// are offset by the frame's start so a file header may already occupy the
/// front of the buffer.
struct ScratchFlavor<'a> {
    buffer: &'a mut Vec<u8>,
    start: usize,
}

impl<'a> ScratchFlavor<'a> {
    fn new(buffer: &'a mut Vec<u8>) -> Self {
        let start = buffer.len();
        Self { buffer, start }
    }
}

impl postcard::ser_flavors::Flavor for ScratchFlavor<'_> {
    type Output = ();

    #[inline]
    fn try_push(&mut self, byte: u8) -> postcard::Result<()> {
        self.buffer.push(byte);
        Ok(())
    }

    #[inline]
    fn try_extend(&mut self, data: &[u8]) -> postcard::Result<()> {
        self.buffer.extend_from_slice(data);
        Ok(())
    }

    fn finalize(self) -> postcard::Result<Self::Output> {
        Ok(())
    }
}

impl std::ops::Index<usize> for ScratchFlavor<'_> {
    type Output = u8;

    fn index(&self, index: usize) -> &u8 {
        &self.buffer[self.start + index]
    }
}

impl std::ops::IndexMut<usize> for ScratchFlavor<'_> {
    fn index_mut(&mut self, index: usize) -> &mut u8 {
        let start = self.start;
        &mut self.buffer[start + index]
    }
}

/// Append-only framed file backend: one session per `{dir}/{session_id}.cslog`.
///
/// Each record is one COBS-framed postcard frame appended with
/// `OpenOptions::append(true)`, so prior frames are never rewritten or
/// truncated (append-only invariant). Replay validates the file header, then
/// decodes frames in append order; a missing file replays as empty. A session
/// whose only file is a legacy `{session_id}.jsonl` is reported as
/// [`PluginError::Validation`], never replayed as empty. Session ids containing
/// path separators (`/` or `\`) are rejected with [`PluginError::Validation`]
/// rather than silently mapping into a subdirectory.
///
/// Trajectory growth is bounded by disk, not process memory. The one in-process
/// buffer is the framing scratch: it is capped at 64 KiB of retained capacity
/// and shrunk back to that bound after any larger record, so a single oversized
/// record cannot pin memory for the process lifetime. [`Self::scratch_capacity`]
/// is the usage gauge.
pub struct FileBackend {
    dir: std::path::PathBuf,
    fork_counter: Mutex<u64>,
    scratch: Mutex<Vec<u8>>,
}

impl FileBackend {
    /// Create (and, if needed, `create_dir_all`) the backing directory.
    ///
    /// # Errors
    ///
    /// [`PluginError::Io`] when `dir` cannot be created.
    pub fn new(dir: impl Into<std::path::PathBuf>) -> Result<Self, PluginError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| PluginError::Io(e.to_string()))?;
        Ok(Self {
            dir,
            fork_counter: Mutex::new(0),
            scratch: Mutex::new(Vec::new()),
        })
    }

    /// Currently retained framing-scratch capacity in bytes, at most 64 KiB
    /// between appends.
    pub fn scratch_capacity(&self) -> usize {
        self.scratch
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .capacity()
    }

    /// Resolve a session id to its record file path.
    fn path(&self, session_id: &str) -> std::path::PathBuf {
        self.dir.join(format!("{session_id}.cslog"))
    }

    /// Reject ids that would escape the directory or cross into subdirectories.
    fn check_safe_id(id: &str) -> Result<(), PluginError> {
        if id.contains('/') || id.contains('\\') {
            return Err(PluginError::Validation {
                schema: "session_id".to_string(),
                message: format!(
                    "session id `{id}` contains a path separator; refusing to map it to a file"
                ),
            });
        }
        Ok(())
    }

    /// Decode the frames following the header of a whole file.
    fn decode_frames(
        bytes: &mut [u8],
        path: &std::path::Path,
    ) -> Result<Vec<SessionRecord>, PluginError> {
        let invalid = |message: String| PluginError::Validation {
            schema: "cslog".to_string(),
            message,
        };

        if bytes.len() < HEADER_LEN {
            return Err(invalid(format!(
                "`{}` is {} bytes, too short for a {HEADER_LEN}-byte header",
                path.display(),
                bytes.len()
            )));
        }
        let (header, mut rest) = bytes.split_at_mut(HEADER_LEN);
        if &header[..FILE_MAGIC.len()] != FILE_MAGIC.as_slice() {
            return Err(invalid(format!(
                "`{}` does not start with the CUCASLOG magic",
                path.display()
            )));
        }
        let version = header[FILE_MAGIC.len()];
        if version != FORMAT_VERSION {
            return Err(invalid(format!(
                "`{}` has format version {version}; this build reads version {FORMAT_VERSION}",
                path.display()
            )));
        }

        let mut records = Vec::new();
        while !rest.is_empty() {
            let (stored, tail) =
                postcard::take_from_bytes_cobs::<StoredRecord>(rest).map_err(|e| {
                    invalid(format!(
                        "invalid record frame {} in `{}`: {e}",
                        records.len(),
                        path.display()
                    ))
                })?;
            records.push(SessionRecord::from(stored));
            rest = tail;
        }
        Ok(records)
    }
}

impl SessionBackend for FileBackend {
    fn append(&self, record: &SessionRecord) -> Result<(), PluginError> {
        Self::check_safe_id(&record.session_id)?;
        let path = self.path(&record.session_id);
        let stored = StoredRecord::from(record);

        // The scratch lock also serializes same-process appends, so the header
        // check and the write below cannot interleave with another append.
        let mut scratch = self.scratch.lock().unwrap_or_else(|p| p.into_inner());

        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .map_err(|e| PluginError::Io(e.to_string()))?;
        let is_new = file
            .metadata()
            .map_err(|e| PluginError::Io(e.to_string()))?
            .len()
            == 0;

        scratch.clear();
        if is_new {
            scratch.extend_from_slice(FILE_MAGIC);
            scratch.push(FORMAT_VERSION);
        }
        let flavor = postcard::ser_flavors::Cobs::try_new(ScratchFlavor::new(&mut scratch))
            .map_err(|e| PluginError::Io(e.to_string()))?;
        postcard::serialize_with_flavor(&stored, flavor)
            .map_err(|e| PluginError::Io(e.to_string()))?;

        // One write_all: a new file's header and its first frame land together.
        let write = file
            .write_all(&scratch)
            .map_err(|e| PluginError::Io(e.to_string()));
        // Release the frame, then cap retained capacity: one oversized record
        // must not pin memory for the process lifetime. `shrink_to` cannot drop
        // below the current length, so the clear has to come first.
        scratch.clear();
        if scratch.capacity() > MAX_RETAINED_SCRATCH {
            scratch.shrink_to(MAX_RETAINED_SCRATCH);
        }
        write
    }

    fn replay(&self, session_id: &str) -> Result<Vec<SessionRecord>, PluginError> {
        Self::check_safe_id(session_id)?;
        let path = self.path(session_id);
        let mut bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let legacy = self.dir.join(format!("{session_id}.jsonl"));
                if legacy.exists() {
                    return Err(PluginError::Validation {
                        schema: "cslog".to_string(),
                        message: format!(
                            "session `{session_id}` has no `{}`, but a JSON-lines file `{}` exists; \
                             this backend reads framed postcard records and will not decode it",
                            path.display(),
                            legacy.display()
                        ),
                    });
                }
                return Ok(Vec::new());
            }
            Err(e) => return Err(PluginError::Io(e.to_string())),
        };
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        Self::decode_frames(&mut bytes, &path)
    }

    fn fork(&self, session_id: &str, point_id: &str) -> Result<String, PluginError> {
        Self::check_safe_id(session_id)?;
        let records = self.replay(session_id)?;
        let index = records
            .iter()
            .position(|r| r.point_id() == point_id)
            .ok_or_else(|| PluginError::Validation {
                schema: "point_id".to_string(),
                message: format!("no record with point_id `{point_id}` in session `{session_id}`"),
            })?;

        let mut counter = self.fork_counter.lock().unwrap_or_else(|p| p.into_inner());
        let n = *counter;
        *counter += 1;
        drop(counter);
        let new_id = format!("{session_id}:fork:{point_id}:{n}");
        Self::check_safe_id(&new_id)?;

        // Write the new branch (fresh file, created via the append-only path).
        let prefix: Vec<SessionRecord> = records[..=index]
            .iter()
            .map(|r| SessionRecord {
                session_id: new_id.clone(),
                ..r.clone()
            })
            .collect();
        for rec in &prefix {
            self.append(rec)?;
        }

        // Auditability: append the Fork record to the original, at its tail.
        let fork_rec = SessionRecord {
            session_id: session_id.to_string(),
            sequence: records.len() as u64,
            timestamp_ms: now_ms(),
            event: SessionEvent::Fork {
                from_point: point_id.to_string(),
                to_session: new_id.clone(),
            },
        };
        self.append(&fork_rec)?;
        Ok(new_id)
    }
}

/// Epoch millis at call time (mirrors [`SessionRecord::new`]).
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Append-only session-log plugin wiring the [`CucaPlugin`] hooks to a
/// [`SessionBackend`].
///
/// The hooks record to the configured `session_id` (default `"default"`).
/// `on_request` logs a [`SessionEvent::SystemPrompt`] for every System message
/// (always) and a [`SessionEvent::Message`] for user/assistant messages on
/// their first appearance by position; `on_stream_chunk` maps each content
/// block to its event; `on_response_complete` logs latency and token usage.
/// Append failures propagate as [`PluginError`] from the hook that hit them.
///
/// # Growth
///
/// The trajectory itself lives in the backend, which owns the bound
/// ([`InMemoryBackend::max_records`], or disk for [`FileBackend`]). The
/// plugin keeps only two bookkeeping maps, and they do not grow with traffic:
/// each holds one small entry (a session id plus a counter) per *distinct
/// session id* ever passed to [`SessionStorePlugin::append_log`] or created by
/// [`SessionStorePlugin::fork_session`], and every later record for that
/// session reuses its entry. They are deliberately uncapped because evicting
/// one would restart that session's sequence numbering at zero and corrupt
/// replay and forking; a caller that mints unbounded session ids owns that
/// bound.
pub struct SessionLogPlugin {
    backend: Arc<dyn SessionBackend>,
    /// Per-session next sequence (the store is authoritative over ordering).
    sequence: Mutex<HashMap<String, u64>>,
    /// Per-session count of user/assistant message positions already recorded
    /// (first-appearance-by-position dedupe for `on_request`).
    recorded_messages: Mutex<HashMap<String, usize>>,
    /// Default session the hooks write to.
    session_id: String,
}

impl SessionLogPlugin {
    /// Create the plugin over a caller-supplied backend.
    pub fn new(backend: Arc<dyn SessionBackend>) -> Self {
        Self {
            backend,
            sequence: Mutex::new(HashMap::new()),
            recorded_messages: Mutex::new(HashMap::new()),
            session_id: "default".to_string(),
        }
    }

    /// Convenience constructor backed by a fresh [`InMemoryBackend`] (capped
    /// at [`InMemoryBackend::DEFAULT_MAX_RECORDS`] records).
    pub fn new_in_memory() -> Self {
        Self::new(Arc::new(InMemoryBackend::new()))
    }

    /// Set the session the hooks write to (builder-style, consumes `self`).
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = id.into();
        self
    }

    /// Accessor for the underlying backend (useful for tests and diagnostics).
    pub fn backend(&self) -> &Arc<dyn SessionBackend> {
        &self.backend
    }

    /// Sequence one event onto `session_id` and append it, taking the event by
    /// value.
    ///
    /// [`SessionStorePlugin::append_log`] can only borrow its record, so it has
    /// to clone the event; the hooks build theirs locally and hand ownership
    /// over instead. That clone would otherwise land on every streamed chunk,
    /// alongside the `SessionRecord` the hook no longer has to build.
    fn append_event(
        &self,
        session_id: &str,
        timestamp_ms: u64,
        event: SessionEvent,
    ) -> Result<(), PluginError> {
        // The store is authoritative over sequence numbers: the record lands
        // at the session's next value and keeps the caller's timestamp.
        let mut seqs = self.sequence.lock().unwrap_or_else(|p| p.into_inner());
        let next = seqs.get(session_id).copied().unwrap_or(0);
        let sequenced = SessionRecord {
            session_id: session_id.to_string(),
            sequence: next,
            timestamp_ms,
            event,
        };
        seqs.insert(session_id.to_string(), next + 1);
        drop(seqs);
        self.backend.append(&sequenced)
    }
}

impl CucaPlugin for SessionLogPlugin {
    fn name(&self) -> &'static str {
        "session-log"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        let mut recorded = self
            .recorded_messages
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut new_count = recorded.get(&self.session_id).copied().unwrap_or(0);

        for (i, msg) in req.messages.iter().enumerate() {
            match msg.role {
                MessageRole::System => {
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            MessageContentBlock::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    self.append_event(
                        &self.session_id,
                        now_ms(),
                        SessionEvent::SystemPrompt { text },
                    )?;
                }
                MessageRole::User | MessageRole::Assistant => {
                    // First-appearance-by-position: record only messages whose
                    // position we have not yet covered, then advance the cursor.
                    if i >= new_count {
                        self.append_event(
                            &self.session_id,
                            now_ms(),
                            SessionEvent::Message {
                                role: msg.role,
                                content: msg.content.clone(),
                            },
                        )?;
                        new_count = i + 1;
                    }
                }
                MessageRole::Tool => {}
            }
        }

        recorded.insert(self.session_id.clone(), new_count);
        Ok(())
    }

    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        let event = match chunk {
            MessageContentBlock::Thinking {
                reasoning,
                signature,
            } => Some(SessionEvent::Reasoning {
                reasoning: reasoning.clone(),
                signature: signature.clone(),
            }),
            MessageContentBlock::Text(text) => Some(SessionEvent::Output { text: text.clone() }),
            MessageContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => Some(SessionEvent::ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            }),
            MessageContentBlock::ToolResult {
                tool_call_id,
                output,
            } => {
                // The unified block carries only call id + output; streams and
                // exit code are unavailable here, so they stay None.
                Some(SessionEvent::ToolResult {
                    tool_call_id: tool_call_id.clone(),
                    output: output.clone(),
                    stdout: None,
                    stderr: None,
                    exit_code: None,
                })
            }
            MessageContentBlock::ImageBase64 { .. } => None,
        };
        if let Some(event) = event {
            self.append_event(&self.session_id, now_ms(), event)?;
        }
        Ok(())
    }

    fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
        self.append_event(
            &self.session_id,
            now_ms(),
            SessionEvent::Latency {
                duration_ms: (res.duration_secs * 1000.0) as u64,
            },
        )?;
        self.append_event(
            &self.session_id,
            now_ms(),
            SessionEvent::TokenUsage {
                prompt_tokens: res.prompt_tokens,
                completion_tokens: res.completion_tokens,
            },
        )?;
        Ok(())
    }
}

impl SessionStorePlugin for SessionLogPlugin {
    fn append_log(&self, session_id: &str, record: &SessionRecord) -> Result<(), PluginError> {
        // Only the event is cloned: the trait hands us a borrowed record, and
        // the sequence/session fields are rewritten anyway.
        self.append_event(session_id, record.timestamp_ms, record.event.clone())
    }

    fn replay_session(&self, session_id: &str) -> Result<Vec<SessionRecord>, PluginError> {
        self.backend.replay(session_id)
    }

    fn fork_session(&self, session_id: &str, point_id: &str) -> Result<String, PluginError> {
        let new_id = self.backend.fork(session_id, point_id)?;
        // The backend appended a Fork audit record to the original; realign its
        // next-sequence to the replayed length so later appends don't collide.
        let len = self.backend.replay(session_id)?.len() as u64;
        let mut seqs = self.sequence.lock().unwrap_or_else(|p| p.into_inner());
        seqs.insert(session_id.to_string(), len);
        Ok(new_id)
    }
}

#[cfg(all(test, feature = "plugin-session-log"))]
mod tests {
    use super::*;
    use crate::request::UnifiedResponse;
    use crate::types::ProviderEndpoint;

    /// Removes the temp directory on drop so tests leave no files behind.
    struct TestDir(std::path::PathBuf);
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fresh_temp_dir() -> TestDir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        TestDir(std::env::temp_dir().join(format!(
            "cuca-session-log-test-{}-{nanos}",
            std::process::id()
        )))
    }

    fn event_kind(e: &SessionEvent) -> &'static str {
        match e {
            SessionEvent::SystemPrompt { .. } => "SystemPrompt",
            SessionEvent::Message { .. } => "Message",
            SessionEvent::Reasoning { .. } => "Reasoning",
            SessionEvent::Output { .. } => "Output",
            SessionEvent::ToolCall { .. } => "ToolCall",
            SessionEvent::ToolResult { .. } => "ToolResult",
            SessionEvent::ModelSwap { .. } => "ModelSwap",
            SessionEvent::Latency { .. } => "Latency",
            SessionEvent::TokenUsage { .. } => "TokenUsage",
            SessionEvent::Fork { .. } => "Fork",
        }
    }

    #[test]
    fn append_log_assigns_increasing_sequences() {
        let plugin = SessionLogPlugin::new_in_memory();
        let rec = |e| SessionRecord::new("s", e);
        plugin
            .append_log("s", &rec(SessionEvent::SystemPrompt { text: "a".into() }))
            .unwrap();
        plugin
            .append_log("s", &rec(SessionEvent::Output { text: "b".into() }))
            .unwrap();
        plugin
            .append_log("s", &rec(SessionEvent::Output { text: "c".into() }))
            .unwrap();

        let replay = plugin.replay_session("s").unwrap();
        let seqs: Vec<u64> = replay.iter().map(|r| r.sequence).collect();
        assert_eq!(seqs, vec![0, 1, 2]);
        assert!(replay.iter().all(|r| r.session_id == "s"));
    }

    #[test]
    fn fork_session_branches_prefix_and_audits_original() {
        let plugin = SessionLogPlugin::new_in_memory();
        for i in 0..4 {
            plugin
                .append_log(
                    "s",
                    &SessionRecord::new(
                        "s",
                        SessionEvent::Output {
                            text: format!("o{i}"),
                        },
                    ),
                )
                .unwrap();
        }

        let new_id = plugin.fork_session("s", "s:2").unwrap();
        assert!(new_id.starts_with("s:fork:s:2:"));

        // New session = prefix [0, 1, 2], re-labelled to the branch id.
        let new_replay = plugin.replay_session(&new_id).unwrap();
        assert_eq!(new_replay.len(), 3);
        let new_seqs: Vec<u64> = new_replay.iter().map(|r| r.sequence).collect();
        assert_eq!(new_seqs, vec![0, 1, 2]);
        assert!(new_replay.iter().all(|r| r.session_id == new_id));

        // Original gains a Fork audit record at its tail.
        let orig = plugin.replay_session("s").unwrap();
        assert_eq!(orig.len(), 5);
        match &orig.last().unwrap().event {
            SessionEvent::Fork {
                from_point,
                to_session,
            } => {
                assert_eq!(from_point, "s:2");
                assert_eq!(to_session, &new_id);
            }
            other => panic!("expected Fork, got {other:?}"),
        }

        // Sequence counter realigned: next append lands after the Fork record.
        plugin
            .append_log(
                "s",
                &SessionRecord::new(
                    "s",
                    SessionEvent::Output {
                        text: "after".into(),
                    },
                ),
            )
            .unwrap();
        let after = plugin.replay_session("s").unwrap();
        assert_eq!(after.last().unwrap().sequence, 5);

        // Unknown point ids error.
        assert!(plugin.fork_session("s", "s:99").is_err());
        assert!(plugin.fork_session("s", "missing").is_err());
    }

    #[test]
    fn file_backend_appends_without_truncation_and_replays() {
        let guard = fresh_temp_dir();
        let dir = guard.0.clone();
        let backend = FileBackend::new(&dir).unwrap();

        let rec =
            |seq: u64, text: String| SessionRecord::at("j", seq, 1, SessionEvent::Output { text });
        for i in 0..3 {
            backend.append(&rec(i, format!("line{i}"))).unwrap();
        }

        let path = dir.join("j.cslog");
        let bytes = std::fs::read(&path).unwrap();
        // Header once, then one zero-delimited frame per record.
        assert_eq!(&bytes[..FILE_MAGIC.len()], FILE_MAGIC.as_slice());
        assert_eq!(bytes[FILE_MAGIC.len()], FORMAT_VERSION);
        assert_eq!(bytes.iter().filter(|b| **b == 0).count(), 3);
        let prefix = bytes.clone();

        // Re-opening the backend and appending adds frames without truncating.
        let backend2 = FileBackend::new(&dir).unwrap();
        backend2.append(&rec(3, "line3".into())).unwrap();
        let bytes2 = std::fs::read(&path).unwrap();
        assert_eq!(&bytes2[..prefix.len()], prefix.as_slice());
        assert_eq!(bytes2.iter().filter(|b| **b == 0).count(), 4);
        // The header is written once, not per append.
        assert_eq!(
            bytes2
                .windows(FILE_MAGIC.len())
                .filter(|w| *w == FILE_MAGIC.as_slice())
                .count(),
            1
        );

        // A fresh instance replays every record in order.
        let fresh = FileBackend::new(&dir).unwrap();
        let replay = fresh.replay("j").unwrap();
        assert_eq!(replay.len(), 4);
        let seqs: Vec<u64> = replay.iter().map(|r| r.sequence).collect();
        assert_eq!(seqs, vec![0, 1, 2, 3]);

        // Missing session replays as empty.
        assert!(fresh.replay("nope").unwrap().is_empty());

        // Path-separator ids are rejected.
        let bad = SessionRecord::at("bad/id", 0, 1, SessionEvent::Output { text: "x".into() });
        assert!(fresh.append(&bad).is_err());
        assert!(fresh.replay("bad/id").is_err());
    }

    #[test]
    fn file_fork_writes_branch_and_audits_original() {
        let guard = fresh_temp_dir();
        let dir = guard.0.clone();
        let backend = FileBackend::new(&dir).unwrap();
        let rec =
            |seq: u64, text: String| SessionRecord::at("s", seq, 1, SessionEvent::Output { text });
        for i in 0..4 {
            backend.append(&rec(i, format!("o{i}"))).unwrap();
        }

        let new_id = backend.fork("s", "s:2").unwrap();
        let branch = backend.replay(&new_id).unwrap();
        assert_eq!(branch.len(), 3);
        assert!(branch.iter().all(|r| r.session_id == new_id));

        let orig = backend.replay("s").unwrap();
        assert_eq!(orig.len(), 5);
        assert!(matches!(
            orig.last().unwrap().event,
            SessionEvent::Fork { .. }
        ));
    }

    /// One record per `SessionEvent` variant, with every
    /// `MessageContentBlock` variant inside the `Message` one.
    fn every_event() -> Vec<SessionEvent> {
        vec![
            SessionEvent::SystemPrompt {
                text: "be concise".into(),
            },
            SessionEvent::Message {
                role: MessageRole::Assistant,
                content: vec![
                    MessageContentBlock::Text("hi".into()),
                    MessageContentBlock::ImageBase64 {
                        media_type: "image/png".into(),
                        data: "AAAA".into(),
                    },
                    MessageContentBlock::Thinking {
                        reasoning: "weighing".into(),
                        signature: Some("sig".into()),
                    },
                    MessageContentBlock::ToolCall {
                        id: "call-1".into(),
                        name: "search".into(),
                        arguments: serde_json::json!({ "z": [null, true, 1.5, "s", {}], "a": 1 }),
                    },
                    MessageContentBlock::ToolResult {
                        tool_call_id: "call-1".into(),
                        output: "found".into(),
                    },
                ],
            },
            SessionEvent::Reasoning {
                reasoning: "because".into(),
                signature: None,
            },
            SessionEvent::Output {
                text: "answer".into(),
            },
            SessionEvent::ToolCall {
                id: "tc-1".into(),
                name: "run".into(),
                arguments: serde_json::json!([1, { "k": "v" }]),
            },
            SessionEvent::ToolResult {
                tool_call_id: "tc-1".into(),
                output: "out".into(),
                stdout: Some("stdout-bytes".into()),
                stderr: Some("stderr-bytes".into()),
                exit_code: Some(3),
            },
            SessionEvent::ModelSwap {
                from: "fast".into(),
                to: "slow".into(),
                reason: "malformed".into(),
            },
            SessionEvent::Latency { duration_ms: 42 },
            SessionEvent::TokenUsage {
                prompt_tokens: 11,
                completion_tokens: 22,
            },
            SessionEvent::Fork {
                from_point: "s:0".into(),
                to_session: "s2".into(),
            },
        ]
    }

    #[test]
    fn every_event_variant_round_trips_through_the_file_backend() {
        let guard = fresh_temp_dir();
        let backend = FileBackend::new(&guard.0).unwrap();
        let events = every_event();
        // Guards the mirror against a variant added to SessionEvent.
        assert_eq!(events.len(), 10, "cover every SessionEvent variant");

        let records: Vec<SessionRecord> = events
            .into_iter()
            .enumerate()
            .map(|(i, event)| {
                SessionRecord::at("all", i as u64, 1_700_000_000_000 + i as u64, event)
            })
            .collect();
        for record in &records {
            backend.append(record).unwrap();
        }
        assert_eq!(backend.replay("all").unwrap(), records);
    }

    #[test]
    fn public_json_representation_is_unchanged() {
        // The mirrors are internal: SessionRecord's own JSON shape stays
        // internally tagged, with tool arguments as a JSON object.
        let record = SessionRecord::at(
            "s",
            0,
            7,
            SessionEvent::ToolCall {
                id: "tc-1".into(),
                name: "run".into(),
                arguments: serde_json::json!({ "a": 1 }),
            },
        );
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "session_id": "s",
                "sequence": 0,
                "timestamp_ms": 7,
                "event": {
                    "type": "tool_call",
                    "id": "tc-1",
                    "name": "run",
                    "arguments": { "a": 1 }
                }
            })
        );
        let back: SessionRecord = serde_json::from_value(json).unwrap();
        assert_eq!(back, record);
    }

    #[test]
    fn replay_rejects_an_unknown_format_version() {
        let guard = fresh_temp_dir();
        let dir = guard.0.clone();
        let backend = FileBackend::new(&dir).unwrap();
        backend
            .append(&SessionRecord::at(
                "v",
                0,
                1,
                SessionEvent::Output { text: "x".into() },
            ))
            .unwrap();

        let path = dir.join("v.cslog");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[FILE_MAGIC.len()] = FORMAT_VERSION + 1;
        std::fs::write(&path, &bytes).unwrap();

        let err = backend.replay("v").expect_err("unknown version must fail");
        let message = format!("{err}");
        assert!(message.contains("format version"), "{message}");
    }

    #[test]
    fn replay_rejects_a_file_without_the_magic() {
        let guard = fresh_temp_dir();
        let dir = guard.0.clone();
        let backend = FileBackend::new(&dir).unwrap();
        std::fs::write(dir.join("m.cslog"), b"not-a-cuca-log-file").unwrap();
        let err = backend.replay("m").expect_err("bad magic must fail");
        assert!(format!("{err}").contains("magic"));

        // Too short for a header at all.
        std::fs::write(dir.join("t.cslog"), b"CUCA").unwrap();
        let err = backend.replay("t").expect_err("short file must fail");
        assert!(format!("{err}").contains("too short"));
    }

    #[test]
    fn replay_reports_a_legacy_jsonl_file_instead_of_replaying_empty() {
        let guard = fresh_temp_dir();
        let dir = guard.0.clone();
        let backend = FileBackend::new(&dir).unwrap();
        std::fs::write(
            dir.join("old.jsonl"),
            b"{\"session_id\":\"old\",\"sequence\":0,\"timestamp_ms\":1,\
              \"event\":{\"type\":\"output\",\"text\":\"x\"}}\n",
        )
        .unwrap();

        let err = backend
            .replay("old")
            .expect_err("a legacy file must not replay as empty");
        let message = format!("{err}");
        assert!(message.contains("old.jsonl"), "{message}");

        // A session with neither file still replays as empty.
        assert!(backend.replay("absent").unwrap().is_empty());
    }

    #[test]
    fn replay_detects_a_torn_trailing_frame() {
        let guard = fresh_temp_dir();
        let dir = guard.0.clone();
        let backend = FileBackend::new(&dir).unwrap();
        for i in 0..3 {
            backend
                .append(&SessionRecord::at(
                    "torn",
                    i,
                    1,
                    SessionEvent::Output {
                        text: format!("o{i}"),
                    },
                ))
                .unwrap();
        }

        let path = dir.join("torn.cslog");
        let bytes = std::fs::read(&path).unwrap();
        // Drop the final delimiter and one payload byte: a crash mid-write.
        std::fs::write(&path, &bytes[..bytes.len() - 2]).unwrap();
        let err = backend.replay("torn").expect_err("torn frame must fail");
        assert!(format!("{err}").contains("record frame"));
    }

    #[test]
    fn empty_file_replays_as_empty() {
        let guard = fresh_temp_dir();
        let dir = guard.0.clone();
        let backend = FileBackend::new(&dir).unwrap();
        std::fs::write(dir.join("e.cslog"), b"").unwrap();
        assert!(backend.replay("e").unwrap().is_empty());
    }

    #[test]
    fn framing_scratch_stays_within_its_bound() {
        let guard = fresh_temp_dir();
        let backend = FileBackend::new(&guard.0).unwrap();
        // One oversized record must not pin capacity for the process lifetime.
        backend
            .append(&SessionRecord::at(
                "big",
                0,
                1,
                SessionEvent::Output {
                    text: "x".repeat(MAX_RETAINED_SCRATCH * 2),
                },
            ))
            .unwrap();
        assert!(backend.scratch_capacity() <= MAX_RETAINED_SCRATCH);

        backend
            .append(&SessionRecord::at(
                "big",
                1,
                1,
                SessionEvent::Output {
                    text: "small".into(),
                },
            ))
            .unwrap();
        assert!(backend.scratch_capacity() <= MAX_RETAINED_SCRATCH);
        assert_eq!(backend.replay("big").unwrap().len(), 2);
    }

    #[test]
    fn hooks_record_all_interaction_classes_in_append_order() {
        let plugin = SessionLogPlugin::new_in_memory();

        let mut req = UnifiedRequest::new("m")
            .add_system_message("be concise")
            .add_user_message("hello");
        plugin.on_request(&mut req).unwrap();

        let mut text = MessageContentBlock::Text("hi there".into());
        plugin.on_stream_chunk(&mut text).unwrap();
        let mut call = MessageContentBlock::ToolCall {
            id: "call-1".into(),
            name: "search".into(),
            arguments: serde_json::json!({ "q": "x" }),
        };
        plugin.on_stream_chunk(&mut call).unwrap();
        let mut result = MessageContentBlock::ToolResult {
            tool_call_id: "call-1".into(),
            output: "found".into(),
        };
        plugin.on_stream_chunk(&mut result).unwrap();

        let res = UnifiedResponse {
            model: "m".into(),
            provider: ProviderEndpoint::OpenAi,
            duration_secs: 0.5,
            prompt_tokens: 10,
            completion_tokens: 20,
            finish_reason: Some("stop".into()),
            content: Vec::new(),
            prompt_cache_usage: None,
        };
        plugin.on_response_complete(&res).unwrap();

        let replay = plugin.replay_session("default").unwrap();
        let kinds: Vec<&str> = replay.iter().map(|r| event_kind(&r.event)).collect();
        assert_eq!(
            kinds,
            vec![
                "SystemPrompt",
                "Message",
                "Output",
                "ToolCall",
                "ToolResult",
                "Latency",
                "TokenUsage",
            ]
        );

        match &replay[5].event {
            SessionEvent::Latency { duration_ms } => assert_eq!(*duration_ms, 500),
            other => panic!("expected Latency, got {other:?}"),
        }
        match &replay[6].event {
            SessionEvent::TokenUsage {
                prompt_tokens,
                completion_tokens,
            } => {
                assert_eq!((*prompt_tokens, *completion_tokens), (10, 20));
            }
            other => panic!("expected TokenUsage, got {other:?}"),
        }
        match &replay[4].event {
            SessionEvent::ToolResult { tool_call_id, .. } => assert_eq!(tool_call_id, "call-1"),
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn thinking_chunk_records_reasoning() {
        let plugin = SessionLogPlugin::new_in_memory();
        let mut chunk = MessageContentBlock::Thinking {
            reasoning: "think".into(),
            signature: Some("sig".into()),
        };
        plugin.on_stream_chunk(&mut chunk).unwrap();
        let replay = plugin.replay_session("default").unwrap();
        assert_eq!(replay.len(), 1);
        match &replay[0].event {
            SessionEvent::Reasoning {
                reasoning,
                signature,
            } => {
                assert_eq!(reasoning, "think");
                assert_eq!(signature.as_deref(), Some("sig"));
            }
            other => panic!("expected Reasoning, got {other:?}"),
        }
    }

    #[test]
    fn name_and_send_sync() {
        let plugin = SessionLogPlugin::new_in_memory();
        assert_eq!(plugin.name(), "session-log");

        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SessionLogPlugin>();
        assert_send_sync::<InMemoryBackend>();
        assert_send_sync::<FileBackend>();
    }

    // --- InMemoryBackend record cap ---

    #[test]
    fn in_memory_backend_rejects_zero_cap() {
        assert!(matches!(
            InMemoryBackend::with_max_records(0),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn in_memory_append_fails_loudly_at_the_cap() {
        let backend = InMemoryBackend::with_max_records(2).unwrap();
        assert!(backend.is_empty());
        let rec = |seq: u64| {
            SessionRecord::at(
                "s",
                seq,
                1,
                SessionEvent::Output {
                    text: format!("o{seq}"),
                },
            )
        };
        backend.append(&rec(0)).unwrap();
        backend.append(&rec(1)).unwrap();
        assert_eq!(backend.len(), 2);

        let err = backend.append(&rec(2)).unwrap_err();
        assert!(matches!(err, PluginError::Internal(_)));
        // No eviction, no partial write: the log is exactly as before.
        assert_eq!(backend.len(), 2);
        assert_eq!(backend.replay("s").unwrap().len(), 2);
    }

    #[test]
    fn in_memory_fork_is_all_or_nothing_at_the_cap() {
        // Cap 4: three records fit, but a fork at the second record needs
        // three more (two branched plus one Fork audit) and must not fit.
        let backend = InMemoryBackend::with_max_records(4).unwrap();
        let rec = |seq: u64| {
            SessionRecord::at(
                "s",
                seq,
                1,
                SessionEvent::Output {
                    text: format!("o{seq}"),
                },
            )
        };
        for seq in 0..3 {
            backend.append(&rec(seq)).unwrap();
        }
        let point = backend.replay("s").unwrap()[1].point_id();

        let err = backend.fork("s", &point).unwrap_err();
        assert!(matches!(err, PluginError::Internal(_)));
        // Nothing changed: no branch session, no Fork audit record, gauge flat.
        assert_eq!(backend.len(), 3);
        let replay = backend.replay("s").unwrap();
        assert_eq!(replay.len(), 3);
        assert!(
            replay
                .iter()
                .all(|r| !matches!(r.event, SessionEvent::Fork { .. }))
        );

        // With headroom the same fork succeeds and the gauge counts the
        // branch prefix plus the audit record.
        let roomy = InMemoryBackend::with_max_records(8).unwrap();
        for seq in 0..3 {
            roomy.append(&rec(seq)).unwrap();
        }
        let new_id = roomy.fork("s", &point).unwrap();
        assert_eq!(roomy.len(), 6);
        assert_eq!(roomy.replay(&new_id).unwrap().len(), 2);
    }

    #[test]
    fn with_session_id_retargets_hooks() {
        let plugin = SessionLogPlugin::new_in_memory().with_session_id("custom");
        let mut req = UnifiedRequest::new("m").add_user_message("hi");
        plugin.on_request(&mut req).unwrap();

        let replay = plugin.replay_session("custom").unwrap();
        assert_eq!(replay.len(), 1);
        assert!(matches!(
            replay[0].event,
            SessionEvent::Message {
                role: MessageRole::User,
                ..
            }
        ));
        // Default session untouched.
        assert!(plugin.replay_session("default").unwrap().is_empty());
    }

    #[test]
    fn hook_append_failures_propagate() {
        struct FailingBackend;
        impl SessionBackend for FailingBackend {
            fn append(&self, _r: &SessionRecord) -> Result<(), PluginError> {
                Err(PluginError::Internal("boom".into()))
            }
            fn replay(&self, _s: &str) -> Result<Vec<SessionRecord>, PluginError> {
                Ok(vec![])
            }
            fn fork(&self, _s: &str, _p: &str) -> Result<String, PluginError> {
                Ok("x".into())
            }
        }
        let plugin = SessionLogPlugin::new(Arc::new(FailingBackend));
        let res = UnifiedResponse {
            model: "m".into(),
            provider: ProviderEndpoint::OpenAi,
            duration_secs: 0.5,
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: None,
            content: Vec::new(),
            prompt_cache_usage: None,
        };
        assert!(plugin.on_response_complete(&res).is_err());
    }
}
