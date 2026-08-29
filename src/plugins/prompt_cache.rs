//! Client-owned local response cache (`plugin-prompt-cache`).
//!
//! [`PromptCache`] is a bounded, TTL-and-LRU-evicting cache of complete
//! `UnifiedRequest` -> `UnifiedResponse` pairs, keyed by the lowercase
//! SHA-256 hex digest of the canonicalized effective request
//! ([`digest_request`]). It is a plain client-level service (see
//! `CucaClient`), never a `CucaPlugin`: callers compute the lookup key
//! themselves (after provider selection and every `on_request` hook runs)
//! and pass it to [`PromptCache::lookup`]; a miss is completed by calling
//! [`PromptCache::insert`] once the response finishes.
//!
//! # Determinism
//!
//! The cache never reads the wall clock directly inside its public methods:
//! every method goes through the crate-private [`CacheClock`] the instance
//! was built with (the production instance reads UNIX milliseconds; tests
//! substitute a deterministic clock). LRU order is tracked independently of
//! hash-map iteration order via an oldest-first key list (`lru_order`), so
//! eviction and exported ranks are reproducible across runs.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::request::{UnifiedRequest, UnifiedResponse};

/// Error returned by [`PromptCache`] and [`PromptCacheConfig`] operations.
#[derive(Debug, Clone)]
pub enum PromptCacheError {
    /// Invalid cache configuration (e.g. zero capacity or non-positive TTL).
    Config(String),
    /// A cache entry or an imported snapshot failed a structural check.
    Validation {
        /// The field (or entry key) that failed validation.
        field: String,
        /// Human-readable validation detail.
        message: String,
    },
    /// JSON serialization/deserialization failure while digesting a request.
    Json(String),
    /// The internal state mutex was poisoned by a panicking lock holder.
    Lock(String),
}

impl std::fmt::Display for PromptCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptCacheError::Config(msg) => write!(f, "prompt cache configuration error: {msg}"),
            PromptCacheError::Validation { field, message } => {
                write!(f, "prompt cache validation failed for {field}: {message}")
            }
            PromptCacheError::Json(msg) => write!(f, "prompt cache JSON error: {msg}"),
            PromptCacheError::Lock(msg) => write!(f, "prompt cache lock error: {msg}"),
        }
    }
}

impl std::error::Error for PromptCacheError {}

/// Bounded capacity and time-to-live for a [`PromptCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptCacheConfig {
    /// Maximum number of live entries retained; must be non-zero.
    pub capacity: usize,
    /// Time-to-live for a stored entry, starting at successful insertion;
    /// must be positive.
    pub ttl: Duration,
}

impl PromptCacheConfig {
    /// Build a validated cache configuration.
    ///
    /// # Errors
    ///
    /// [`PromptCacheError::Config`] for a zero `capacity` or a zero `ttl` (a
    /// [`Duration`] cannot be negative, so zero is the only non-positive value
    /// to guard against).
    pub fn new(capacity: usize, ttl: Duration) -> Result<Self, PromptCacheError> {
        if capacity == 0 {
            return Err(PromptCacheError::Config(
                "capacity must be non-zero".to_string(),
            ));
        }
        if ttl.is_zero() {
            return Err(PromptCacheError::Config("ttl must be positive".to_string()));
        }
        Ok(Self { capacity, ttl })
    }
}

/// One stored request/response pair.
///
/// **Sensitive full-fidelity export:** `cuca-export` intentionally includes the
/// complete memory graph and local-cache request/response values. It may
/// contain confidential system prompts, user messages, tool arguments and
/// results, base64 image data, model output, signatures, and graph properties.
/// Treat the JSON as sensitive data; do not log or publish it. CUCA does not
/// encrypt, redact, or write it. The caller owns access control, encryption,
/// storage, and deletion.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptCacheEntry {
    /// Lowercase SHA-256 hex digest of the canonicalized effective request.
    pub key: String,
    /// The effective request that produced `response`.
    pub request: UnifiedRequest,
    /// The normalized response stored for `request`.
    pub response: UnifiedResponse,
    /// UNIX milliseconds at which the entry was stored.
    pub stored_at_unix_ms: u64,
    /// UNIX milliseconds at which the entry expires.
    pub expires_at_unix_ms: u64,
    /// Recency rank at export time; `0` is least recently used.
    pub lru_rank: usize,
}

/// A point-in-time export of every live cache entry, sorted by `key`.
///
/// **Sensitive full-fidelity export:** `cuca-export` intentionally includes the
/// complete memory graph and local-cache request/response values. It may
/// contain confidential system prompts, user messages, tool arguments and
/// results, base64 image data, model output, signatures, and graph properties.
/// Treat the JSON as sensitive data; do not log or publish it. CUCA does not
/// encrypt, redact, or write it. The caller owns access control, encryption,
/// storage, and deletion.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptCacheSnapshot {
    /// Live entries, sorted by `key`.
    pub entries: Vec<PromptCacheEntry>,
}

/// Outcome of [`PromptCache::replace_snapshot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptCacheImportReport {
    /// Number of entries actually installed into the cache.
    pub imported_entries: usize,
    /// Number of snapshot entries skipped because they had already expired.
    pub expired_entries: usize,
    /// Number of live entries evicted to fit the destination capacity.
    pub capacity_evictions: usize,
}

/// A source of UNIX-millisecond timestamps for [`PromptCache`].
///
/// Crate-private: the production instance reads the wall clock; tests
/// substitute a deterministic value so TTL/LRU behavior is reproducible.
trait CacheClock: Send + Sync {
    fn now_unix_ms(&self) -> u64;
}

/// Production clock: the real wall-clock time in UNIX milliseconds.
#[derive(Debug, Default)]
struct SystemClock;

impl CacheClock for SystemClock {
    fn now_unix_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// Internal mutable state guarded by [`PromptCache`]'s mutex.
struct PromptCacheState {
    /// Entries keyed by digest.
    entries: HashMap<String, PromptCacheEntry>,
    /// Digest keys ordered oldest-first; the back is most recently used.
    /// Exported `lru_rank`s are rebuilt from this order, never from hash-map
    /// iteration.
    lru_order: VecDeque<String>,
}

/// Client-owned bounded response cache: TTL expiry plus deterministic LRU
/// eviction over complete effective-request digests.
///
/// `Send + Sync`; safe to share behind an `Arc` across concurrent lookups.
/// Never accepts a path, writer, or persistence backend.
pub struct PromptCache {
    config: PromptCacheConfig,
    state: Mutex<PromptCacheState>,
    clock: Arc<dyn CacheClock>,
}

impl PromptCache {
    /// Build an empty cache from a validated configuration, using the real
    /// wall clock.
    ///
    /// # Errors
    ///
    /// Infallible today — `config` is already validated by
    /// [`PromptCacheConfig::new`] — but the `Result` is part of the published
    /// signature so a future construction-time check stays non-breaking.
    pub fn new(config: PromptCacheConfig) -> Result<Self, PromptCacheError> {
        Ok(Self::with_clock(config, Arc::new(SystemClock)))
    }

    /// Build an empty cache with an injected clock (crate-private test seam;
    /// visible to `mod tests` below as a descendant module).
    fn with_clock(config: PromptCacheConfig, clock: Arc<dyn CacheClock>) -> Self {
        Self {
            config,
            clock,
            state: Mutex::new(PromptCacheState {
                entries: HashMap::new(),
                lru_order: VecDeque::new(),
            }),
        }
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, PromptCacheState>, PromptCacheError> {
        self.state
            .lock()
            .map_err(|e| PromptCacheError::Lock(e.to_string()))
    }

    /// Look up a previously computed digest (see [`digest_request`]).
    ///
    /// A live hit refreshes recency (moves the key to the most-recently-used
    /// position) and returns the stored entry with a freshly computed
    /// `lru_rank`. An expired entry is removed from the cache and treated as
    /// a miss; a miss never changes recency for any other entry.
    ///
    /// # Errors
    ///
    /// [`PromptCacheError::Lock`] when the state mutex is poisoned.
    pub fn lookup(&self, key: &str) -> Result<Option<PromptCacheEntry>, PromptCacheError> {
        let now = self.clock.now_unix_ms();
        let mut guard = self.lock_state()?;
        // Reborrow through the guard once so `entries` and `lru_order` are
        // borrowed as disjoint fields; without it every `state.<field>` access
        // is a separate `DerefMut` and the entry has to be looked up twice.
        let state = &mut *guard;
        let Some(stored) = state.entries.get(key) else {
            return Ok(None);
        };
        if stored.expires_at_unix_ms <= now {
            state.entries.remove(key);
            remove_from_order(&mut state.lru_order, key);
            return Ok(None);
        }
        let mut entry = stored.clone();
        remove_from_order(&mut state.lru_order, key);
        state.lru_order.push_back(key.to_string());
        entry.lru_rank = rank_of(&state.lru_order, key);
        Ok(Some(entry))
    }

    /// Compute the digest of `request`, prune expired entries, and store the
    /// pair. Inserting under an already-present digest refreshes its
    /// timestamps and recency without growing the cache. If capacity is
    /// exceeded, the least-recently-used entry (rank `0`) is evicted.
    ///
    /// # Errors
    ///
    /// The [`digest_request`] errors for `request`; [`PromptCacheError::Lock`]
    /// when the state mutex is poisoned.
    pub fn insert(
        &self,
        request: UnifiedRequest,
        response: UnifiedResponse,
    ) -> Result<(), PromptCacheError> {
        let key = digest_request(&request)?;
        let now = self.clock.now_unix_ms();
        let mut state = self.lock_state()?;
        prune_expired(&mut state, now);

        let expires_at = now.saturating_add(self.config.ttl.as_millis() as u64);
        let entry = PromptCacheEntry {
            key: key.clone(),
            request,
            response,
            stored_at_unix_ms: now,
            expires_at_unix_ms: expires_at,
            lru_rank: 0,
        };
        state.entries.insert(key.clone(), entry);
        remove_from_order(&mut state.lru_order, &key);
        state.lru_order.push_back(key);

        while state.lru_order.len() > self.config.capacity {
            if let Some(oldest) = state.lru_order.pop_front() {
                state.entries.remove(&oldest);
            }
        }
        Ok(())
    }

    /// Number of resident entries, without cloning any entry.
    ///
    /// The cheap usage gauge for a caller watching the cache fill: one lock
    /// hold and a `HashMap::len`, no request/response clones. The count
    /// includes entries that have expired but have not yet been pruned
    /// (pruning happens in [`Self::insert`] and [`Self::lookup`]), so it is
    /// an upper bound on the live count. Compare it against
    /// [`Self::capacity`]; [`Self::snapshot`] gives the exact live set and
    /// clones every entry.
    ///
    /// # Errors
    ///
    /// [`PromptCacheError::Lock`] when the state mutex is poisoned.
    pub fn len(&self) -> Result<usize, PromptCacheError> {
        Ok(self.lock_state()?.entries.len())
    }

    /// Whether the cache holds no resident entries (see [`Self::len`]).
    ///
    /// # Errors
    ///
    /// [`PromptCacheError::Lock`] when the state mutex is poisoned.
    pub fn is_empty(&self) -> Result<bool, PromptCacheError> {
        Ok(self.lock_state()?.entries.is_empty())
    }

    /// The configured entry bound: the value [`Self::len`] is measured
    /// against. Reads the immutable config, so it takes no lock.
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Export every live entry, sorted by key, using the real wall clock.
    ///
    /// Expired entries are excluded. The returned snapshot carries complete
    /// request/response values: see [`PromptCacheSnapshot`] for the
    /// sensitive-data warning. No path, file, or writer is involved.
    ///
    /// # Errors
    ///
    /// [`PromptCacheError::Lock`] when the cache's state mutex is poisoned.
    pub fn snapshot(&self) -> Result<PromptCacheSnapshot, PromptCacheError> {
        self.snapshot_at(self.clock.now_unix_ms())
    }

    /// Crate-private test seam: [`Self::snapshot`] pinned to `now_unix_ms`.
    pub(crate) fn snapshot_at(
        &self,
        now_unix_ms: u64,
    ) -> Result<PromptCacheSnapshot, PromptCacheError> {
        let state = self.lock_state()?;
        // Renumber ranks contiguously from 0 over the live entries only,
        // ordered by their position in `lru_order` (oldest live first),
        // mirroring `stage_snapshot`'s install renumbering. Deriving ranks
        // straight from `rank_of` over the unpruned `lru_order` would leave
        // gaps wherever an expired entry (excluded below) still occupies a
        // slot in that order, producing ranks `validate_snapshot` rejects.
        let mut entries: Vec<PromptCacheEntry> = state
            .lru_order
            .iter()
            .filter_map(|key| state.entries.get(key))
            .filter(|e| e.expires_at_unix_ms > now_unix_ms)
            .cloned()
            .collect();
        for (rank, entry) in entries.iter_mut().enumerate() {
            entry.lru_rank = rank;
        }
        entries.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(PromptCacheSnapshot { entries })
    }

    /// Validate and stage `snapshot`, then atomically replace the live
    /// state, using the real wall clock to decide which entries are still
    /// live.
    ///
    /// Every entry is validated (key shape, digest match, timestamp order,
    /// rank uniqueness/contiguity, no duplicate keys) before any lock is
    /// held; a single validation failure rejects the whole import with no
    /// state change. Expired entries are then skipped and counted. If more
    /// live entries remain than `capacity`, only the newest (highest-rank)
    /// ones are kept and the rest are counted as capacity evictions.
    /// Installed ranks are renumbered contiguously from `0`.
    ///
    /// # Errors
    ///
    /// [`PromptCacheError::Validation`] naming the offending entry field for a
    /// malformed key, digest mismatch, bad timestamp order, or duplicate
    /// key/rank; [`PromptCacheError::Json`] when digesting an entry's request
    /// fails; [`PromptCacheError::Lock`] when the state mutex is poisoned.
    pub fn replace_snapshot(
        &self,
        snapshot: PromptCacheSnapshot,
    ) -> Result<PromptCacheImportReport, PromptCacheError> {
        self.replace_snapshot_at(snapshot, self.clock.now_unix_ms())
    }

    /// Crate-private test seam: [`Self::replace_snapshot`] pinned to
    /// `now_unix_ms`.
    pub(crate) fn replace_snapshot_at(
        &self,
        snapshot: PromptCacheSnapshot,
        now_unix_ms: u64,
    ) -> Result<PromptCacheImportReport, PromptCacheError> {
        let staged = self.stage_snapshot(snapshot, now_unix_ms)?;
        self.commit_staged(staged)
    }

    /// Validate `snapshot` and build the exact state it would install,
    /// without locking or touching the live cache.
    ///
    /// Staging seam for the combined `cuca-export` coordinator: it lets every
    /// component be validated before any component commits. All structural
    /// checks (key shape, digest match, timestamp order, rank
    /// uniqueness/contiguity, duplicate keys) run here, before expiration
    /// filtering and capacity trimming, so a duplicate among
    /// already-expired entries is still rejected.
    pub(crate) fn stage_snapshot(
        &self,
        snapshot: PromptCacheSnapshot,
        now_unix_ms: u64,
    ) -> Result<StagedCacheState, PromptCacheError> {
        validate_snapshot(&snapshot)?;

        let total = snapshot.entries.len();
        let mut live: Vec<PromptCacheEntry> = snapshot
            .entries
            .into_iter()
            .filter(|e| e.expires_at_unix_ms > now_unix_ms)
            .collect();
        let expired_entries = total - live.len();

        // Ascending rank: oldest first, so the tail is most-recently-used.
        live.sort_by_key(|e| e.lru_rank);

        let capacity_evictions = live.len().saturating_sub(self.config.capacity);
        if capacity_evictions > 0 {
            live.drain(0..capacity_evictions);
        }

        let mut lru_order = VecDeque::with_capacity(live.len());
        let mut entries = HashMap::with_capacity(live.len());
        for (rank, mut entry) in live.into_iter().enumerate() {
            entry.lru_rank = rank;
            lru_order.push_back(entry.key.clone());
            entries.insert(entry.key.clone(), entry);
        }
        let imported_entries = entries.len();

        Ok(StagedCacheState {
            state: PromptCacheState { entries, lru_order },
            report: PromptCacheImportReport {
                imported_entries,
                expired_entries,
                capacity_evictions,
            },
        })
    }

    /// Install an already staged state under a single lock hold.
    ///
    /// Commit seam for the combined `cuca-export` coordinator: staging can
    /// fail freely, but this step only replaces state and can fail solely on
    /// a poisoned lock.
    pub(crate) fn commit_staged(
        &self,
        staged: StagedCacheState,
    ) -> Result<PromptCacheImportReport, PromptCacheError> {
        let mut state = self.lock_state()?;
        *state = staged.state;
        Ok(staged.report)
    }
}

/// A fully validated cache state that is ready to install, plus the report
/// [`PromptCache::commit_staged`] will return once it is installed.
///
/// Opaque by design: the coordinator carries it between the staging and
/// commit phases without reaching into [`PromptCache`]'s private state.
pub(crate) struct StagedCacheState {
    state: PromptCacheState,
    report: PromptCacheImportReport,
}

/// Remove `key` from `order` wherever it currently sits (no-op if absent).
fn remove_from_order(order: &mut VecDeque<String>, key: &str) {
    if let Some(pos) = order.iter().position(|k| k == key) {
        order.remove(pos);
    }
}

/// Current oldest-first position of `key` in `order`; `0` if absent (should
/// only be queried for keys known to be present).
fn rank_of(order: &VecDeque<String>, key: &str) -> usize {
    order.iter().position(|k| k == key).unwrap_or(0)
}

/// Remove every entry whose `expires_at_unix_ms` is at or before `now_unix_ms`.
fn prune_expired(state: &mut PromptCacheState, now_unix_ms: u64) {
    let expired: Vec<String> = state
        .entries
        .iter()
        .filter(|(_, e)| e.expires_at_unix_ms <= now_unix_ms)
        .map(|(k, _)| k.clone())
        .collect();
    for key in expired {
        state.entries.remove(&key);
        remove_from_order(&mut state.lru_order, &key);
    }
}

/// Structural validation for an imported snapshot: run before any lock is
/// held or state is touched, so a single bad entry rejects the whole import
/// without any partial mutation. Checks every entry (including ones that
/// will later be filtered as expired), so duplicate keys and rank conflicts
/// are rejected before expiration filtering runs.
fn validate_snapshot(snapshot: &PromptCacheSnapshot) -> Result<(), PromptCacheError> {
    let mut seen_keys = HashSet::with_capacity(snapshot.entries.len());
    let mut seen_ranks = HashSet::with_capacity(snapshot.entries.len());
    for entry in &snapshot.entries {
        if !is_lowercase_sha256_hex(&entry.key) {
            return Err(PromptCacheError::Validation {
                field: format!("entries[{}].key", entry.key),
                message: "key must be a lowercase 64-character SHA-256 hex digest".to_string(),
            });
        }
        if !seen_keys.insert(entry.key.clone()) {
            return Err(PromptCacheError::Validation {
                field: format!("entries[{}].key", entry.key),
                message: "duplicate key in imported snapshot".to_string(),
            });
        }
        if entry.stored_at_unix_ms >= entry.expires_at_unix_ms {
            return Err(PromptCacheError::Validation {
                field: format!("entries[{}].stored_at_unix_ms", entry.key),
                message: "stored_at_unix_ms must be less than expires_at_unix_ms".to_string(),
            });
        }
        let expected_key = digest_request(&entry.request)?;
        if expected_key != entry.key {
            return Err(PromptCacheError::Validation {
                field: format!("entries[{}].key", entry.key),
                message: "key does not match the digest of its request".to_string(),
            });
        }
        if !seen_ranks.insert(entry.lru_rank) {
            return Err(PromptCacheError::Validation {
                field: format!("entries[{}].lru_rank", entry.key),
                message: "duplicate lru_rank in imported snapshot".to_string(),
            });
        }
    }
    // Unique ranks, all < n, with n of them: by pigeonhole they are exactly
    // the set {0, .., n-1}, i.e. unique implies contiguous here.
    let n = snapshot.entries.len();
    if seen_ranks.iter().any(|&r| r >= n) {
        return Err(PromptCacheError::Validation {
            field: "entries[].lru_rank".to_string(),
            message: format!("ranks must be contiguous from 0..{n}"),
        });
    }
    Ok(())
}

/// Whether `s` is exactly 64 lowercase hex characters (a SHA-256 digest).
fn is_lowercase_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

/// Compute the lowercase SHA-256 hex digest of the canonical JSON form of the
/// complete effective request.
///
/// "Effective" means the request exactly as it will cross the wire: after
/// provider selection and every `on_request` hook. Canonicalization
/// recursively sorts JSON object keys and preserves array order and scalar
/// values, so semantically identical requests digest identically regardless
/// of struct field-declaration or hash-map iteration order. `UnifiedRequest`
/// never carries client credentials, bearer tokens, base URLs, or HTTP
/// clients (those live on `CucaClient`), so the digest never depends on them.
///
/// # Errors
///
/// [`PromptCacheError::Validation`] for a non-finite `temperature`
/// (`NaN`/`inf`), which is rejected rather than silently digested
/// (`serde_json` would otherwise encode it as JSON `null`, colliding with
/// an absent temperature); [`PromptCacheError::Json`] when the request
/// cannot be serialized.
pub fn digest_request(request: &UnifiedRequest) -> Result<String, PromptCacheError> {
    if let Some(t) = request.temperature
        && !t.is_finite()
    {
        return Err(PromptCacheError::Validation {
            field: "temperature".to_string(),
            message: "temperature must be finite".to_string(),
        });
    }
    let value = serde_json::to_value(request).map_err(|e| PromptCacheError::Json(e.to_string()))?;
    let canonical = canonicalize_json(value);
    let bytes =
        serde_json::to_vec(&canonical).map_err(|e| PromptCacheError::Json(e.to_string()))?;
    let digest = Sha256::digest(&bytes);
    Ok(to_lower_hex(&digest))
}

/// Recursively rebuild `value` with every object's keys sorted; arrays and
/// scalars are preserved as-is (array order is significant, object key order
/// is not).
fn canonicalize_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(String, serde_json::Value)> = map
                .into_iter()
                .map(|(k, v)| (k, canonicalize_json(v)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let mut sorted = serde_json::Map::with_capacity(entries.len());
            for (k, v) in entries {
                sorted.insert(k, v);
            }
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(canonicalize_json).collect())
        }
        other => other,
    }
}

/// Format `bytes` as lowercase hex, independent of any `GenericArray`
/// hex-formatting impl.
///
/// Nibble lookup rather than `format!("{b:02x}")` per byte: the digest is
/// computed once per request, and the `format!` form allocated one throwaway
/// `String` for each of the 32 digest bytes.
fn to_lower_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::types::{MessageContentBlock, ProviderEndpoint, ToolDefinition};

    /// Deterministic clock for LRU/TTL tests: starts at `start` and only
    /// moves when explicitly advanced or set.
    #[derive(Debug)]
    struct FixedClock(AtomicU64);

    impl FixedClock {
        fn new(start: u64) -> Self {
            Self(AtomicU64::new(start))
        }

        fn set(&self, value: u64) {
            self.0.store(value, Ordering::SeqCst);
        }
    }

    impl CacheClock for FixedClock {
        fn now_unix_ms(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn cache_with_clock(
        capacity: usize,
        ttl_ms: u64,
        start: u64,
    ) -> (PromptCache, Arc<FixedClock>) {
        let clock = Arc::new(FixedClock::new(start));
        let config = PromptCacheConfig::new(capacity, Duration::from_millis(ttl_ms)).unwrap();
        let cache = PromptCache::with_clock(config, clock.clone());
        (cache, clock)
    }

    fn sample_request(model: &str) -> UnifiedRequest {
        UnifiedRequest::new(model)
            .add_system_message("be concise")
            .add_user_message("hello")
    }

    fn sample_response(model: &str) -> UnifiedResponse {
        UnifiedResponse {
            model: model.to_string(),
            provider: ProviderEndpoint::OpenAi,
            duration_secs: 1.0,
            prompt_tokens: 10,
            completion_tokens: 5,
            finish_reason: Some("stop".to_string()),
            content: vec![MessageContentBlock::Text("ok".to_string())],
            prompt_cache_usage: None,
        }
    }

    fn entry_for(
        request: UnifiedRequest,
        response: UnifiedResponse,
        stored_at: u64,
        ttl_ms: u64,
        rank: usize,
    ) -> PromptCacheEntry {
        let key = digest_request(&request).expect("digest should succeed");
        PromptCacheEntry {
            key,
            request,
            response,
            stored_at_unix_ms: stored_at,
            expires_at_unix_ms: stored_at + ttl_ms,
            lru_rank: rank,
        }
    }

    // --- digest_request ---

    #[test]
    fn digest_is_64_lowercase_hex_chars() {
        let digest = digest_request(&sample_request("model-a")).unwrap();
        assert_eq!(digest.len(), 64);
        assert!(
            digest
                .chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        );
    }

    #[test]
    fn digest_is_deterministic_for_the_same_request() {
        let a = digest_request(&sample_request("model-a")).unwrap();
        let b = digest_request(&sample_request("model-a")).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn digest_changes_with_every_effective_request_field() {
        let base = sample_request("model-a");
        let base_digest = digest_request(&base).unwrap();

        let mut provider_changed = base.clone();
        provider_changed.provider = ProviderEndpoint::Anthropic;
        assert_ne!(digest_request(&provider_changed).unwrap(), base_digest);

        let messages_changed = base.clone().add_user_message("one more turn");
        assert_ne!(digest_request(&messages_changed).unwrap(), base_digest);

        let temperature_changed = base.clone().set_temperature(0.9);
        assert_ne!(digest_request(&temperature_changed).unwrap(), base_digest);

        let max_tokens_changed = base.clone().set_max_tokens(64);
        assert_ne!(digest_request(&max_tokens_changed).unwrap(), base_digest);

        let stream_changed = base.clone().with_stream(false);
        assert_ne!(digest_request(&stream_changed).unwrap(), base_digest);

        let thinking_changed = base.clone().enable_thinking(None);
        assert_ne!(digest_request(&thinking_changed).unwrap(), base_digest);

        let tools_changed = base.clone().add_tool(ToolDefinition {
            name: "search".to_string(),
            description: "search the web".to_string(),
            input_schema: serde_json::json!({ "type": "object" }),
        });
        assert_ne!(digest_request(&tools_changed).unwrap(), base_digest);

        let directive_changed =
            base.clone()
                .with_prompt_cache(crate::request::PromptCacheDirective::Ephemeral {
                    breakpoints: vec![crate::request::PromptCacheBreakpoint {
                        message_index: 0,
                        block_index: 0,
                    }],
                });
        assert_ne!(digest_request(&directive_changed).unwrap(), base_digest);
    }

    #[test]
    fn digest_ignores_nested_object_key_order_but_preserves_array_order() {
        // Same tool schema, differently ordered keys at parse time.
        let schema_a: serde_json::Value =
            serde_json::from_str(r#"{"type":"object","properties":{"q":{"type":"string"}}}"#)
                .unwrap();
        let schema_b: serde_json::Value =
            serde_json::from_str(r#"{"properties":{"q":{"type":"string"}},"type":"object"}"#)
                .unwrap();

        let tool_a = ToolDefinition {
            name: "search".to_string(),
            description: "d".to_string(),
            input_schema: schema_a,
        };
        let tool_b = ToolDefinition {
            name: "search".to_string(),
            description: "d".to_string(),
            input_schema: schema_b,
        };
        let request_a = sample_request("model-a").add_tool(tool_a.clone());
        let request_b = sample_request("model-a").add_tool(tool_b.clone());
        assert_eq!(
            digest_request(&request_a).unwrap(),
            digest_request(&request_b).unwrap(),
            "object key order must not affect the digest"
        );

        // Same two tools, reversed order: array order IS significant.
        let other_tool = ToolDefinition {
            name: "other".to_string(),
            description: "d".to_string(),
            input_schema: serde_json::json!({}),
        };
        let request_forward = sample_request("model-a")
            .add_tool(tool_a.clone())
            .add_tool(other_tool.clone());
        let request_reversed = sample_request("model-a")
            .add_tool(other_tool)
            .add_tool(tool_a);
        assert_ne!(
            digest_request(&request_forward).unwrap(),
            digest_request(&request_reversed).unwrap(),
            "array order must affect the digest"
        );
    }

    #[test]
    fn digest_is_unaffected_by_client_only_values() {
        // UnifiedRequest never carries credentials/base URLs/HTTP clients;
        // two structurally-equal requests always digest the same regardless
        // of what a caller's CucaClient is separately configured with.
        let a = sample_request("model-a");
        let b = sample_request("model-a");
        assert_eq!(digest_request(&a).unwrap(), digest_request(&b).unwrap());
    }

    #[test]
    fn digest_rejects_non_finite_temperature() {
        let request = sample_request("model-a").set_temperature(f32::NAN);
        let err = digest_request(&request).unwrap_err();
        assert!(
            matches!(err, PromptCacheError::Validation { field, .. } if field == "temperature")
        );

        let request = sample_request("model-a").set_temperature(f32::INFINITY);
        assert!(digest_request(&request).is_err());
    }

    // --- config validation ---

    #[test]
    fn config_rejects_zero_capacity_and_zero_ttl() {
        assert!(matches!(
            PromptCacheConfig::new(0, Duration::from_millis(100)),
            Err(PromptCacheError::Config(_))
        ));
        assert!(matches!(
            PromptCacheConfig::new(1, Duration::ZERO),
            Err(PromptCacheError::Config(_))
        ));
        assert!(PromptCacheConfig::new(1, Duration::from_millis(1)).is_ok());
    }

    // --- LRU / TTL policy ---

    #[test]
    fn expiry_starts_at_insertion_time() {
        let (cache, _clock) = cache_with_clock(2, 100, 1_000);
        cache
            .insert(sample_request("a"), sample_response("a"))
            .unwrap();
        let key = digest_request(&sample_request("a")).unwrap();
        let state = cache.state.lock().unwrap();
        let entry = state.entries.get(&key).unwrap();
        assert_eq!(entry.stored_at_unix_ms, 1_000);
        assert_eq!(entry.expires_at_unix_ms, 1_100);
    }

    #[test]
    fn expired_lookup_misses_and_physically_removes_the_entry() {
        let (cache, clock) = cache_with_clock(2, 100, 0);
        cache
            .insert(sample_request("a"), sample_response("a"))
            .unwrap();
        let key = digest_request(&sample_request("a")).unwrap();

        clock.set(200); // past the 100ms ttl
        assert_eq!(cache.lookup(&key).unwrap(), None);

        let state = cache.state.lock().unwrap();
        assert!(
            !state.entries.contains_key(&key),
            "expired entry must be physically removed by lookup, not just filtered"
        );
        assert!(!state.lru_order.contains(&key));
    }

    #[test]
    fn insertion_prunes_expired_entries() {
        let (cache, clock) = cache_with_clock(5, 50, 0);
        cache
            .insert(sample_request("a"), sample_response("a"))
            .unwrap();
        let key_a = digest_request(&sample_request("a")).unwrap();

        clock.set(1_000); // far past a's ttl; no lookup performed on a
        cache
            .insert(sample_request("b"), sample_response("b"))
            .unwrap();

        let state = cache.state.lock().unwrap();
        assert!(
            !state.entries.contains_key(&key_a),
            "insert must prune expired entries even without a lookup"
        );
    }

    #[test]
    fn hit_recency_changes_only_for_a_live_key() {
        let (cache, _clock) = cache_with_clock(3, 1_000, 0);
        cache
            .insert(sample_request("a"), sample_response("a"))
            .unwrap();
        cache
            .insert(sample_request("b"), sample_response("b"))
            .unwrap();
        let key_a = digest_request(&sample_request("a")).unwrap();
        let key_b = digest_request(&sample_request("b")).unwrap();

        {
            let state = cache.state.lock().unwrap();
            assert_eq!(state.lru_order, vec![key_a.clone(), key_b.clone()]);
        }

        // Live hit on a: a moves to the back (most recently used).
        cache.lookup(&key_a).unwrap();
        {
            let state = cache.state.lock().unwrap();
            assert_eq!(state.lru_order, vec![key_b.clone(), key_a.clone()]);
        }

        // Miss on an unknown key changes nothing.
        cache.lookup("does-not-exist").unwrap();
        let state = cache.state.lock().unwrap();
        assert_eq!(state.lru_order, vec![key_b, key_a]);
    }

    #[test]
    fn replacement_refreshes_the_key_without_exceeding_capacity() {
        let (cache, clock) = cache_with_clock(2, 1_000, 0);
        cache
            .insert(sample_request("a"), sample_response("a"))
            .unwrap();
        cache
            .insert(sample_request("b"), sample_response("b"))
            .unwrap();
        let key_a = digest_request(&sample_request("a")).unwrap();
        let key_b = digest_request(&sample_request("b")).unwrap();

        clock.set(500);
        cache
            .insert(sample_request("a"), sample_response("a-v2"))
            .unwrap();

        let state = cache.state.lock().unwrap();
        assert_eq!(
            state.entries.len(),
            2,
            "replacement must not grow the cache"
        );
        assert_eq!(state.lru_order, vec![key_b, key_a.clone()]);
        let refreshed = state.entries.get(&key_a).unwrap();
        assert_eq!(refreshed.stored_at_unix_ms, 500);
        assert_eq!(refreshed.response.model, "a-v2");
    }

    #[test]
    fn eviction_removes_rank_zero() {
        let (cache, _clock) = cache_with_clock(2, 1_000, 0);
        cache
            .insert(sample_request("a"), sample_response("a"))
            .unwrap();
        cache
            .insert(sample_request("b"), sample_response("b"))
            .unwrap();
        cache
            .insert(sample_request("c"), sample_response("c"))
            .unwrap();

        let key_a = digest_request(&sample_request("a")).unwrap();
        let key_b = digest_request(&sample_request("b")).unwrap();
        let key_c = digest_request(&sample_request("c")).unwrap();

        let state = cache.state.lock().unwrap();
        assert!(
            !state.entries.contains_key(&key_a),
            "rank-0 (oldest) must be evicted"
        );
        assert_eq!(state.lru_order, vec![key_b, key_c]);
    }

    #[test]
    fn rankings_are_deterministic_regardless_of_hash_map_order() {
        let (cache, _clock) = cache_with_clock(10, 1_000, 0);
        let models = ["m1", "m2", "m3", "m4", "m5"];
        for model in models {
            cache
                .insert(sample_request(model), sample_response(model))
                .unwrap();
        }
        let snapshot = cache.snapshot_at(0).unwrap();
        // Sorted by key (independent of hash-map iteration order).
        let mut sorted_keys = snapshot
            .entries
            .iter()
            .map(|e| e.key.clone())
            .collect::<Vec<_>>();
        sorted_keys.sort();
        assert_eq!(
            snapshot
                .entries
                .iter()
                .map(|e| e.key.clone())
                .collect::<Vec<_>>(),
            sorted_keys
        );
        // Ranks match insertion order (0..=4) independent of hash-map order.
        let mut ranks: Vec<usize> = snapshot.entries.iter().map(|e| e.lru_rank).collect();
        ranks.sort();
        assert_eq!(ranks, vec![0, 1, 2, 3, 4]);
    }

    // --- usage gauge ---

    #[test]
    fn len_tracks_resident_entries_against_capacity() {
        let (cache, clock) = cache_with_clock(2, 100, 0);
        assert_eq!(cache.capacity(), 2);
        assert!(cache.is_empty().unwrap());
        assert_eq!(cache.len().unwrap(), 0);

        cache
            .insert(sample_request("a"), sample_response("a"))
            .unwrap();
        assert_eq!(cache.len().unwrap(), 1);
        assert!(!cache.is_empty().unwrap());

        // Capacity is a hard bound: a third insert evicts rather than grows.
        cache
            .insert(sample_request("b"), sample_response("b"))
            .unwrap();
        cache
            .insert(sample_request("c"), sample_response("c"))
            .unwrap();
        assert_eq!(cache.len().unwrap(), 2);

        // Resident-but-expired entries still count (documented upper bound);
        // a lookup prunes the one it touches.
        clock.set(1_000);
        assert_eq!(cache.len().unwrap(), 2);
        let key_c = digest_request(&sample_request("c")).unwrap();
        assert_eq!(cache.lookup(&key_c).unwrap(), None);
        assert_eq!(cache.len().unwrap(), 1);
    }

    // --- snapshot / replace_snapshot ---

    #[test]
    fn snapshot_entries_are_sorted_by_key() {
        let (cache, _clock) = cache_with_clock(10, 1_000, 0);
        for model in ["zeta", "alpha", "mu"] {
            cache
                .insert(sample_request(model), sample_response(model))
                .unwrap();
        }
        let snapshot = cache.snapshot_at(0).unwrap();
        let keys: Vec<String> = snapshot.entries.iter().map(|e| e.key.clone()).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn snapshot_at_ranks_are_contiguous_when_an_entry_is_resident_but_expired() {
        // Regression test: `snapshot_at` must not derive exported `lru_rank`
        // values from positions in the UNPRUNED `lru_order` while filtering
        // expired entries out of the exported set — that produces
        // non-contiguous ranks that `validate_snapshot` (and therefore
        // `replace_snapshot_at`) rejects. Build three entries so that the
        // oldest is resident-but-expired (no `insert`/`lookup` call has run
        // since it crossed its TTL) while the other two remain live.
        let (cache, _clock) = cache_with_clock(5, 1_000, 0);
        cache
            .insert(sample_request("a"), sample_response("a"))
            .unwrap();
        _clock.set(300);
        cache
            .insert(sample_request("b"), sample_response("b"))
            .unwrap();
        _clock.set(600);
        cache
            .insert(sample_request("c"), sample_response("c"))
            .unwrap();

        // "a" (stored at 0, ttl 1_000) has expired by 1_100, but neither of
        // the later inserts triggered a prune of it (both ran before it
        // expired), so it stays physically resident in `lru_order`/`entries`.
        let now = 1_100;
        let b_key = digest_request(&sample_request("b")).unwrap();
        let c_key = digest_request(&sample_request("c")).unwrap();

        let snapshot = cache.snapshot_at(now).unwrap();
        assert_eq!(
            snapshot.entries.len(),
            2,
            "the resident-but-expired entry must be excluded from the export"
        );
        let mut ranks: Vec<usize> = snapshot.entries.iter().map(|e| e.lru_rank).collect();
        ranks.sort_unstable();
        assert_eq!(
            ranks,
            vec![0, 1],
            "exported ranks must be contiguous over the live subset, not over \
             the unpruned lru_order positions"
        );

        let report = cache
            .replace_snapshot_at(snapshot, now)
            .expect("a live export must always re-import cleanly");
        assert_eq!(report.imported_entries, 2);
        assert_eq!(report.expired_entries, 0);
        assert_eq!(report.capacity_evictions, 0);

        // Live entries and their relative recency order are preserved: "b"
        // (older) then "c" (newer).
        let state = cache.state.lock().unwrap();
        assert_eq!(state.lru_order, vec![b_key.clone(), c_key.clone()]);
        assert_eq!(state.entries.get(&b_key).unwrap().lru_rank, 0);
        assert_eq!(state.entries.get(&c_key).unwrap().lru_rank, 1);
    }

    #[test]
    fn replace_snapshot_rejects_malformed_key() {
        let (cache, _clock) = cache_with_clock(2, 1_000, 0);
        let mut entry = entry_for(sample_request("a"), sample_response("a"), 0, 1_000, 0);
        entry.key = "not-a-valid-hex-digest".to_string();
        let result = cache.replace_snapshot_at(
            PromptCacheSnapshot {
                entries: vec![entry],
            },
            0,
        );
        assert!(matches!(result, Err(PromptCacheError::Validation { .. })));
    }

    #[test]
    fn replace_snapshot_rejects_stored_at_not_before_expires_at() {
        let (cache, _clock) = cache_with_clock(2, 1_000, 0);
        let mut entry = entry_for(sample_request("a"), sample_response("a"), 500, 1_000, 0);
        entry.expires_at_unix_ms = 500; // stored_at == expires_at
        let result = cache.replace_snapshot_at(
            PromptCacheSnapshot {
                entries: vec![entry],
            },
            0,
        );
        assert!(matches!(result, Err(PromptCacheError::Validation { .. })));
    }

    #[test]
    fn replace_snapshot_rejects_non_contiguous_or_duplicate_ranks() {
        let (cache, _clock) = cache_with_clock(3, 1_000, 0);
        let entry_a = entry_for(sample_request("a"), sample_response("a"), 0, 1_000, 0);
        let entry_b = entry_for(sample_request("b"), sample_response("b"), 0, 1_000, 2); // gap: no rank 1
        let result = cache.replace_snapshot_at(
            PromptCacheSnapshot {
                entries: vec![entry_a, entry_b],
            },
            0,
        );
        assert!(matches!(result, Err(PromptCacheError::Validation { .. })));

        let entry_a = entry_for(sample_request("a"), sample_response("a"), 0, 1_000, 0);
        let entry_b = entry_for(sample_request("b"), sample_response("b"), 0, 1_000, 0); // duplicate rank
        let result = cache.replace_snapshot_at(
            PromptCacheSnapshot {
                entries: vec![entry_a, entry_b],
            },
            0,
        );
        assert!(matches!(result, Err(PromptCacheError::Validation { .. })));
    }

    #[test]
    fn replace_snapshot_rejects_duplicate_keys_before_expiration_filtering() {
        let (cache, _clock) = cache_with_clock(3, 1_000, 0);
        let live = entry_for(sample_request("a"), sample_response("a"), 0, 1_000, 0);
        let mut already_expired = entry_for(sample_request("a"), sample_response("a"), 0, 1_000, 1);
        already_expired.expires_at_unix_ms = 1; // would be pruned as expired at now=999999
        let result = cache.replace_snapshot_at(
            PromptCacheSnapshot {
                entries: vec![live, already_expired],
            },
            999_999,
        );
        assert!(
            matches!(result, Err(PromptCacheError::Validation { .. })),
            "duplicate keys must reject even when one copy would later be filtered as expired"
        );
    }

    #[test]
    fn replace_snapshot_rejects_digest_mismatch() {
        let (cache, _clock) = cache_with_clock(2, 1_000, 0);
        let mut entry = entry_for(sample_request("a"), sample_response("a"), 0, 1_000, 0);
        entry.key = "0".repeat(64); // well-formed but wrong
        let result = cache.replace_snapshot_at(
            PromptCacheSnapshot {
                entries: vec![entry],
            },
            0,
        );
        assert!(matches!(result, Err(PromptCacheError::Validation { .. })));
    }

    #[test]
    fn replace_snapshot_skips_and_reports_expired_entries() {
        let (cache, _clock) = cache_with_clock(5, 1_000, 0);
        let live = entry_for(sample_request("a"), sample_response("a"), 0, 1_000, 0);
        let expired = entry_for(sample_request("b"), sample_response("b"), 0, 100, 1);
        let report = cache
            .replace_snapshot_at(
                PromptCacheSnapshot {
                    entries: vec![live.clone(), expired],
                },
                500, // past b's expiry (100ms ttl), before a's (1000ms ttl)
            )
            .unwrap();
        assert_eq!(report.expired_entries, 1);
        assert_eq!(report.imported_entries, 1);
        assert_eq!(report.capacity_evictions, 0);

        let snapshot = cache.snapshot_at(500).unwrap();
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].key, live.key);
    }

    #[test]
    fn replace_snapshot_retains_newest_ranks_when_capacity_shrinks() {
        let (cache, _clock) = cache_with_clock(2, 1_000, 0); // destination capacity = 2
        let oldest = entry_for(sample_request("a"), sample_response("a"), 0, 1_000, 0);
        let middle = entry_for(sample_request("b"), sample_response("b"), 0, 1_000, 1);
        let newest = entry_for(sample_request("c"), sample_response("c"), 0, 1_000, 2);
        let report = cache
            .replace_snapshot_at(
                PromptCacheSnapshot {
                    entries: vec![oldest.clone(), middle.clone(), newest.clone()],
                },
                0,
            )
            .unwrap();
        assert_eq!(report.capacity_evictions, 1);
        assert_eq!(report.imported_entries, 2);
        assert_eq!(report.expired_entries, 0);

        let state = cache.state.lock().unwrap();
        assert!(!state.entries.contains_key(&oldest.key));
        assert!(state.entries.contains_key(&middle.key));
        assert!(state.entries.contains_key(&newest.key));
        assert_eq!(
            state.lru_order,
            vec![middle.key.clone(), newest.key.clone()]
        );
        // Ranks renumbered contiguously from 0.
        assert_eq!(state.entries.get(&middle.key).unwrap().lru_rank, 0);
        assert_eq!(state.entries.get(&newest.key).unwrap().lru_rank, 1);
    }

    #[test]
    fn replace_snapshot_is_all_or_nothing() {
        let (cache, _clock) = cache_with_clock(2, 1_000, 0);
        cache
            .insert(
                sample_request("pre-existing"),
                sample_response("pre-existing"),
            )
            .unwrap();

        let mut bad_entry = entry_for(sample_request("a"), sample_response("a"), 0, 1_000, 0);
        bad_entry.key = "bad".to_string();
        let result = cache.replace_snapshot_at(
            PromptCacheSnapshot {
                entries: vec![bad_entry],
            },
            0,
        );
        assert!(result.is_err());

        // The pre-existing entry must still be present: no partial mutation.
        let snapshot = cache.snapshot_at(0).unwrap();
        assert_eq!(snapshot.entries.len(), 1);
    }

    // --- deny_unknown_fields ---

    #[test]
    fn prompt_cache_entry_deserialize_rejects_unknown_field() {
        let mut value = serde_json::to_value(entry_for(
            sample_request("a"),
            sample_response("a"),
            0,
            1_000,
            0,
        ))
        .unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("bogus".to_string(), serde_json::json!(true));
        let result: Result<PromptCacheEntry, _> = serde_json::from_value(value);
        assert!(
            result.is_err(),
            "an unknown field must be rejected, not silently ignored"
        );
    }

    #[test]
    fn prompt_cache_snapshot_deserialize_rejects_unknown_field() {
        let mut value = serde_json::to_value(PromptCacheSnapshot::default()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("bogus".to_string(), serde_json::json!(true));
        let result: Result<PromptCacheSnapshot, _> = serde_json::from_value(value);
        assert!(
            result.is_err(),
            "an unknown field must be rejected, not silently ignored"
        );
    }
}
