//! Cumulative token and currency accounting with a hard budget cap.
//!
//! [`CostPlugin`] implements [`CucaPlugin`] and is registered on the builder
//! with `register_plugin`. Four direct-call accessors read and roll the ledger:
//! [`CostPlugin::usage`] (one cheap `Copy` reading), [`CostPlugin::breakdown`]
//! (the per-model slice), [`CostPlugin::reset`] (billing-period rollover), and
//! [`CostPlugin::estimate_request_tokens`] (price a turn before building a
//! client).
//!
//! With `plugin-telemetry` co-enabled,
//! [`OtelCostObserver`](crate::cost_otel::OtelCostObserver) is a ready-made
//! [`CostObserver`] that records every reading to the caller's OpenTelemetry
//! meter provider; it lives in core rather than here, because a plugin must
//! not name a peer.
//!
//! # What the core reports
//!
//! [`UnifiedResponse::prompt_tokens`] is always `0`, and
//! [`UnifiedResponse::completion_tokens`] is a per-block count rather than a
//! token count: the client adds `1` per `Text`, `Thinking`, or `ToolCall`
//! block. No provider adapter parses an upstream `usage` object into either
//! field. [`UnifiedResponse::prompt_cache_usage`] carries the only
//! provider-reported token numbers in the crate, and only the Anthropic adapter
//! populates it. Every figure this plugin records is therefore a tiktoken
//! estimate, reconciled against provider truth for cache read and write tokens
//! alone.
//!
//! # Token counting
//!
//! The estimator is an approximation. It encodes, with `encode_ordinary`: the
//! role label of every message, every `Text` block, every `Thinking.reasoning`,
//! every `ToolResult.output`, every `ToolCall` name plus its JSON-stringified
//! arguments, and every [`ToolDefinition`](crate::types::ToolDefinition) in
//! `UnifiedRequest::tools` (name, description, and stringified input schema,
//! which providers bill as prompt tokens). It does not encode `ImageBase64`
//! blocks, the provider's per-message framing tokens (such as the
//! `<|im_start|>` delimiters tiktoken-rs applies for chat completions), or
//! `max_tokens`/`temperature`. Images are the one deliberate undercount:
//! providers bill them by tile and dimension, which no text tokenizer
//! reproduces, so each skipped prompt block increments
//! [`CostUsage::untokenized_image_blocks`], the visible signal for it.
//!
//! # Synchronous hooks
//!
//! No `await` anywhere: both hooks, the [`PricingResolver`] seam, and the
//! [`CostObserver`] seam are synchronous by construction. Mid-stream abort is
//! impossible, so `on_stream_chunk` is not implemented: an error from that hook
//! rejects one block and the client keeps polling the provider stream. Only
//! `on_request` can refuse a turn before dispatch.
//!
//! # Two-phase charge and correct
//!
//! `on_request` charges the prompt estimate and enforces the cap against the
//! projected total, so a cap is never crossed rather than merely detected.
//! `on_response_complete` charges the completion estimate and applies the cache
//! correction. Two over-counts follow, both conservative:
//!
//! 1. A turn that fails after `on_request` (a provider error, or a later
//!    plugin's `on_request` error) keeps its prompt charge, because no terminal
//!    hook fires on that path. A budget cap therefore fails safe.
//! 2. A locally replayed response still runs every `on_response_complete` hook
//!    against the stored response, so a client-owned local response cache makes
//!    this ledger read as gross, pre-cache spend.
//!
//! # Bounds
//!
//! The per-model breakdown is capped by [`CostConfig::max_tracked_models`]
//! (default `64`, validated non-zero). A turn for an untracked model when the
//! map is full folds into one reserved overflow bucket and increments
//! [`CostUsage::overflow_turns`]. Totals stay exact; only per-model attribution
//! degrades, and the degradation is counted. Nothing is evicted, so no recorded
//! spend disappears, and [`CostPlugin::breakdown`] returns at most
//! `max_tracked_models + 1` entries. The [`PricingTable`] and the observer list
//! are caller-owned and immutable after [`CostPlugin::new`]: no hook writes
//! them. Per-request transient allocation is one `String` per `ToolCall` block
//! and one per tool-definition schema, from `serde_json::to_string`, dropped
//! immediately after encoding; the role label is encoded as its own short call,
//! so no joined per-message `String` is built.
//!
//! # No baked-in prices
//!
//! [`PricingTable`] is caller-supplied and this crate ships no vendor rates.
//! Prices change without a crate release, vary by region, tier, commitment,
//! batch mode, and context-length bracket, and a `ProviderEndpoint::Custom`
//! gateway has pricing a library cannot know. A stale hardcoded table produces
//! confidently wrong billing numbers. The shape matches
//! `ContextWindowResolver` plus a configured fallback instead of a baked-in
//! window registry, and `OpenTelemetryPlugin::new` taking the caller's meter
//! provider rather than installing a global one. This plugin installs nothing
//! global either.
//!
//! # Currency
//!
//! Money is `u64` micro-units of a caller-defined currency, with `u128`
//! intermediates and integer division. The crate never names a currency and
//! never converts. Pass `3_000_000` for US$3.00 per million tokens if the
//! currency is USD.
//!
//! # Hook order
//!
//! The near-cap warning mutates the request, so it changes the effective
//! request a client-level response cache digests, exactly as the memory
//! plugin's injections do. Neither plugin requires a registration position, but
//! the digest and the estimate both differ with the order.
//!
//! # Lock discipline
//!
//! Two mutexes: the tiktoken encoder and the ledger. Encoding finishes and
//! releases the encoder before the ledger lock is taken, and no function holds
//! both guards.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::request::{UnifiedRequest, UnifiedResponse};
use crate::types::{MessageContentBlock, MessageRole, UnifiedMessage};

/// Prefix that makes the near-cap warning injection idempotent within one
/// prompt: `on_request` scans for it and never injects a second warning while
/// one is present.
const COST_WARNING_MARKER: &str = "CUCA cost warning:";

/// Model bucket every turn beyond [`CostConfig::max_tracked_models`] folds into.
///
/// The leading NUL keeps the reserved key disjoint from every real model id.
const OVERFLOW_MODEL: &str = "\u{0}cuca-cost-overflow";

/// Divisor of every price computation: rates are quoted per million tokens.
const TOKENS_PER_MTOK: u128 = 1_000_000;

/// Per-model prices, in micro-units of the caller's currency per million tokens.
///
/// "Micro-units" are caller-defined: pass `3_000_000` for US$3.00/Mtok if the
/// currency is USD. The crate never names a currency and never converts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelRates {
    /// Full-price prompt tokens.
    pub input_micros_per_mtok: u64,
    /// Model-generated completion tokens.
    pub output_micros_per_mtok: u64,
    /// Prompt tokens served from the provider cache
    /// ([`PromptCacheUsage::read_tokens`](crate::request::PromptCacheUsage::read_tokens)).
    /// Replaces the input rate for those tokens.
    pub cache_read_micros_per_mtok: u64,
    /// Prompt tokens written to the provider cache
    /// ([`PromptCacheUsage::write_tokens`](crate::request::PromptCacheUsage::write_tokens)).
    /// Charged on top of the input rate as a cache-creation surcharge.
    pub cache_write_micros_per_mtok: u64,
}

/// Caller-owned model to rate map. Fixed at construction; never grows from
/// traffic.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PricingTable {
    rates: HashMap<String, ModelRates>,
}

impl PricingTable {
    /// An empty table: every model is unpriced until [`Self::with_model`] adds
    /// one.
    pub fn new() -> Self {
        Self::default()
    }

    /// Chainable insert; a repeated model id replaces its rates.
    pub fn with_model(mut self, model: impl Into<String>, rates: ModelRates) -> Self {
        self.rates.insert(model.into(), rates);
        self
    }

    /// Rates for `model`, or `None` when the table does not price it.
    pub fn get(&self, model: &str) -> Option<ModelRates> {
        self.rates.get(model).copied()
    }

    /// Number of priced models.
    pub fn len(&self) -> usize {
        self.rates.len()
    }

    /// Whether the table prices no model at all.
    pub fn is_empty(&self) -> bool {
        self.rates.is_empty()
    }
}

/// Extension seam: live per-model rates, consulted before [`PricingTable`].
///
/// The static table cannot change after construction ([`CostPlugin`] hooks take
/// `&self`); this is the seam for a caller refreshing prices at runtime.
pub trait PricingResolver: Send + Sync {
    /// Rates for `model`, or `None` to fall through to the configured table.
    fn resolve_rates(&self, model: &str) -> Option<ModelRates>;
}

/// What to do with a turn whose model has no rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnpricedModelPolicy {
    /// Refuse the turn in `on_request` with a [`PluginError::HookFailure`].
    Reject,
    /// Charge tokens to the ledger, charge no currency, and count the turn in
    /// [`CostUsage::unpriced_turns`]. Never silent: the counter is the signal.
    CountTokensOnly,
}

/// Configuration for the cost plugin.
pub struct CostConfig {
    /// tiktoken-rs encoder name, e.g. `"cl100k_base"` (a base encoder name) or
    /// a model name like `"gpt-4o"` that maps to a tokenizer.
    pub encoder_name: String,
    /// Static per-model rates.
    pub pricing: PricingTable,
    /// Optional live rate source, consulted before [`Self::pricing`].
    pub pricing_resolver: Option<Arc<dyn PricingResolver>>,
    /// Cumulative token cap; `None` disables token enforcement. `Some(0)` is
    /// rejected at construction.
    pub max_total_tokens: Option<u64>,
    /// Cumulative currency cap in micro-units; `None` disables. `Some(0)` is
    /// rejected at construction.
    pub max_total_micros: Option<u64>,
    /// Inject a near-cap warning system message at this fraction of the
    /// tightest cap. `None` disables. Must be in `(0.0, 1.0]` and requires a
    /// cap.
    pub warn_fraction: Option<f32>,
    /// Cap on distinct model keys in the per-model breakdown; must be non-zero.
    /// A turn beyond it folds into one reserved overflow bucket.
    pub max_tracked_models: usize,
    /// Handling for a model with no rates.
    pub on_unpriced_model: UnpricedModelPolicy,
    /// Observers handed a reading on every charge and commit.
    pub observers: Vec<Arc<dyn CostObserver>>,
}

impl Default for CostConfig {
    /// `"cl100k_base"`, empty pricing, no resolver, no caps, no warning, 64
    /// tracked models, [`UnpricedModelPolicy::CountTokensOnly`], no observers.
    fn default() -> Self {
        Self {
            encoder_name: "cl100k_base".to_string(),
            pricing: PricingTable::new(),
            pricing_resolver: None,
            max_total_tokens: None,
            max_total_micros: None,
            warn_fraction: None,
            max_tracked_models: 64,
            on_unpriced_model: UnpricedModelPolicy::CountTokensOnly,
            observers: Vec::new(),
        }
    }
}

impl CostConfig {
    /// Reject configurations that cannot be honored.
    ///
    /// # Errors
    ///
    /// [`PluginError::Internal`] for: `max_tracked_models == 0`; a `Some(0)`
    /// token or currency cap; a `warn_fraction` that is non-finite, `<= 0.0`,
    /// or `> 1.0`; a `warn_fraction` with no cap set; or
    /// [`UnpricedModelPolicy::Reject`] with an empty `pricing` table and no
    /// `pricing_resolver`, which would refuse every turn.
    fn validate(&self) -> Result<(), PluginError> {
        if self.max_tracked_models == 0 {
            return Err(PluginError::Internal(
                "max_tracked_models must be non-zero".to_string(),
            ));
        }
        if self.max_total_tokens == Some(0) {
            return Err(PluginError::Internal(
                "max_total_tokens must be non-zero when set".to_string(),
            ));
        }
        if self.max_total_micros == Some(0) {
            return Err(PluginError::Internal(
                "max_total_micros must be non-zero when set".to_string(),
            ));
        }
        if let Some(fraction) = self.warn_fraction
            && (!fraction.is_finite() || fraction <= 0.0 || fraction > 1.0)
        {
            return Err(PluginError::Internal(format!(
                "warn_fraction must be in (0.0, 1.0], got {fraction}"
            )));
        }
        if self.warn_fraction.is_some()
            && self.max_total_tokens.is_none()
            && self.max_total_micros.is_none()
        {
            return Err(PluginError::Internal(
                "warn_fraction requires max_total_tokens or max_total_micros".to_string(),
            ));
        }
        if self.on_unpriced_model == UnpricedModelPolicy::Reject
            && self.pricing.is_empty()
            && self.pricing_resolver.is_none()
        {
            return Err(PluginError::Internal(
                "UnpricedModelPolicy::Reject with no pricing table and no pricing_resolver \
                 would refuse every turn"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Cost of `tokens` at `micros_per_mtok`, in micro-units.
///
/// Integer only: the `u128` intermediate holds the product exactly and the
/// division truncates, so no `f64` rounding accumulates across turns. The cast
/// back saturates rather than wrapping.
fn price_micros(tokens: u64, micros_per_mtok: u64) -> u64 {
    let product = u128::from(tokens) * u128::from(micros_per_mtok);
    u64::try_from(product / TOKENS_PER_MTOK).unwrap_or(u64::MAX)
}

/// One ledger reading. `Copy`, allocation-free, computed under one lock.
///
/// Token counts are tiktoken estimates, not provider-reported usage: only
/// `cache_read_tokens` and `cache_write_tokens` come from the provider, and
/// only when the adapter reports them. See the
/// [module docs](crate::plugins::cost).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostUsage {
    /// Estimated prompt tokens charged since construction or the last
    /// [`CostPlugin::reset`].
    pub prompt_tokens: u64,
    /// Estimated completion tokens charged over the same period.
    pub completion_tokens: u64,
    /// Provider-reported prompt tokens served from the provider cache; a
    /// subset of `prompt_tokens`.
    pub cache_read_tokens: u64,
    /// Provider-reported prompt tokens written to the provider cache.
    pub cache_write_tokens: u64,
    /// Cumulative spend in micro-units of the caller's currency.
    pub spent_micros: u64,
    /// Turns committed since construction or the last [`CostPlugin::reset`].
    pub turns: u64,
    /// Turns charged with no rates (only under
    /// [`UnpricedModelPolicy::CountTokensOnly`]).
    pub unpriced_turns: u64,
    /// Prompt `ImageBase64` blocks skipped by the estimator: a known
    /// undercount.
    pub untokenized_image_blocks: u64,
    /// Turns whose model folded into the overflow bucket at
    /// [`CostConfig::max_tracked_models`].
    pub overflow_turns: u64,
    /// The configured cumulative token cap, if any.
    pub max_total_tokens: Option<u64>,
    /// The configured cumulative currency cap, if any.
    pub max_total_micros: Option<u64>,
    /// True when a configured [`CostConfig::warn_fraction`] is met on the
    /// tightest cap.
    pub near_cap: bool,
}

impl CostUsage {
    /// `prompt_tokens + completion_tokens` (cache counters are a subset of
    /// prompt).
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// Per-model slice of the ledger, returned by [`CostPlugin::breakdown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CostEntry {
    /// Estimated prompt tokens charged to this model.
    pub prompt_tokens: u64,
    /// Estimated completion tokens charged to this model.
    pub completion_tokens: u64,
    /// Provider-reported cache-read tokens; a subset of `prompt_tokens`.
    pub cache_read_tokens: u64,
    /// Provider-reported cache-write tokens.
    pub cache_write_tokens: u64,
    /// Spend attributed to this model, in micro-units.
    pub spent_micros: u64,
    /// Turns committed for this model.
    pub turns: u64,
}

impl CostEntry {
    /// `prompt_tokens + completion_tokens`, saturating.
    fn total_tokens(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    /// Field-wise saturating accumulate.
    fn add(&mut self, delta: &CostEntry) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(delta.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(delta.completion_tokens);
        self.cache_read_tokens = self
            .cache_read_tokens
            .saturating_add(delta.cache_read_tokens);
        self.cache_write_tokens = self
            .cache_write_tokens
            .saturating_add(delta.cache_write_tokens);
        self.spent_micros = self.spent_micros.saturating_add(delta.spent_micros);
        self.turns = self.turns.saturating_add(delta.turns);
    }

    /// Field-wise saturating undo of [`Self::add`].
    fn sub(&mut self, delta: &CostEntry) {
        self.prompt_tokens = self.prompt_tokens.saturating_sub(delta.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_sub(delta.completion_tokens);
        self.cache_read_tokens = self
            .cache_read_tokens
            .saturating_sub(delta.cache_read_tokens);
        self.cache_write_tokens = self
            .cache_write_tokens
            .saturating_sub(delta.cache_write_tokens);
        self.spent_micros = self.spent_micros.saturating_sub(delta.spent_micros);
        self.turns = self.turns.saturating_sub(delta.turns);
    }

    /// [`Self::add`] with `refund_micros` taken off the spend first, so the
    /// cache correction can lower what `on_request` already charged.
    fn commit(&mut self, delta: &CostEntry, refund_micros: u64) {
        self.spent_micros = self.spent_micros.saturating_sub(refund_micros);
        self.add(delta);
    }
}

/// Observes the ledger without changing it (reporting/UI gauge seam).
pub trait CostObserver: Send + Sync {
    /// Handed a reading after every `on_request` charge and every
    /// `on_response_complete` commit.
    ///
    /// # Errors
    ///
    /// An `Err` from the `on_request` call aborts the turn and its charge is
    /// rolled back. An `Err` from the `on_response_complete` call is logged by
    /// the client and never surfaces, so that commit stands.
    fn observe(&self, usage: &CostUsage) -> Result<(), PluginError>;
}

/// Private cumulative totals; [`CostUsage`] is the public reading of them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CostTotals {
    entry: CostEntry,
    unpriced_turns: u64,
    untokenized_image_blocks: u64,
    overflow_turns: u64,
}

/// One `on_request` charge: the ledger delta plus the counters riding with it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RequestCharge {
    entry: CostEntry,
    unpriced: bool,
    image_blocks: u64,
}

/// Private ledger state.
struct CostLedger {
    totals: CostTotals,
    per_model: HashMap<String, CostEntry>,
}

impl CostLedger {
    /// An empty ledger.
    fn new() -> Self {
        Self {
            totals: CostTotals::default(),
            per_model: HashMap::new(),
        }
    }

    /// The key `model` charges to, and whether it folded into the reserved
    /// overflow bucket because `max_tracked` distinct models are already
    /// tracked.
    ///
    /// The overflow bucket never consumes a tracked slot, so `per_model` holds
    /// at most `max_tracked + 1` entries.
    fn key_for<'a>(&self, model: &'a str, max_tracked: usize) -> (&'a str, bool) {
        if self.per_model.contains_key(model) {
            return (model, false);
        }
        let tracked =
            self.per_model.len() - usize::from(self.per_model.contains_key(OVERFLOW_MODEL));
        if tracked >= max_tracked {
            (OVERFLOW_MODEL, true)
        } else {
            (model, false)
        }
    }

    /// Add `delta` to `key`'s bucket, creating it on the first charge.
    ///
    /// `HashMap::entry` would allocate an owned key on every charge; only the
    /// first charge for a model needs one.
    fn add_to_bucket(&mut self, key: &str, delta: &CostEntry) {
        if let Some(bucket) = self.per_model.get_mut(key) {
            bucket.add(delta);
            return;
        }
        self.per_model.insert(key.to_string(), *delta);
    }

    /// Apply an `on_request` charge to the totals and to `model`'s bucket.
    ///
    /// A fold is not counted here: `overflow_turns` counts committed turns, and
    /// the same model folds again at commit time.
    fn charge_request(&mut self, model: &str, max_tracked: usize, charge: &RequestCharge) {
        self.totals.entry.add(&charge.entry);
        if charge.unpriced {
            self.totals.unpriced_turns = self.totals.unpriced_turns.saturating_add(1);
        }
        self.totals.untokenized_image_blocks = self
            .totals
            .untokenized_image_blocks
            .saturating_add(charge.image_blocks);
        let (key, _) = self.key_for(model, max_tracked);
        self.add_to_bucket(key, &charge.entry);
    }

    /// Undo [`Self::charge_request`] after an observer aborted the turn.
    ///
    /// Subtracting the same delta stays exact under concurrency: another turn
    /// may have charged in between, and the remaining total is still correct.
    /// A bucket left at zero is dropped so it stops consuming a tracked slot.
    fn refund_request(&mut self, model: &str, max_tracked: usize, charge: &RequestCharge) {
        self.totals.entry.sub(&charge.entry);
        if charge.unpriced {
            self.totals.unpriced_turns = self.totals.unpriced_turns.saturating_sub(1);
        }
        self.totals.untokenized_image_blocks = self
            .totals
            .untokenized_image_blocks
            .saturating_sub(charge.image_blocks);
        let (key, _) = self.key_for(model, max_tracked);
        let emptied = match self.per_model.get_mut(key) {
            Some(bucket) => {
                bucket.sub(&charge.entry);
                *bucket == CostEntry::default()
            }
            None => false,
        };
        if emptied {
            self.per_model.remove(key);
        }
    }

    /// Prompt tokens already charged to `model`'s bucket: the clamp that keeps
    /// the cache correction from crediting more than this ledger charged.
    fn charged_prompt_tokens(&self, model: &str, max_tracked: usize) -> u64 {
        let (key, _) = self.key_for(model, max_tracked);
        self.per_model.get(key).map_or(0, |b| b.prompt_tokens)
    }

    /// Commit a completed turn, taking `refund_micros` off the spend before
    /// `delta` goes on, and counting the turn in `overflow_turns` when it folds
    /// into the overflow bucket.
    fn commit_response(
        &mut self,
        model: &str,
        max_tracked: usize,
        delta: &CostEntry,
        refund_micros: u64,
    ) {
        let (key, folded) = self.key_for(model, max_tracked);
        if folded {
            self.totals.overflow_turns = self.totals.overflow_turns.saturating_add(1);
        }
        self.totals.entry.commit(delta, refund_micros);
        if let Some(bucket) = self.per_model.get_mut(key) {
            bucket.commit(delta, refund_micros);
            return;
        }
        let mut fresh = CostEntry::default();
        fresh.commit(delta, refund_micros);
        self.per_model.insert(key.to_string(), fresh);
    }

    /// The public reading, with `near_cap` derived rather than latched.
    fn usage(&self, config: &CostConfig) -> CostUsage {
        let mut usage = CostUsage {
            prompt_tokens: self.totals.entry.prompt_tokens,
            completion_tokens: self.totals.entry.completion_tokens,
            cache_read_tokens: self.totals.entry.cache_read_tokens,
            cache_write_tokens: self.totals.entry.cache_write_tokens,
            spent_micros: self.totals.entry.spent_micros,
            turns: self.totals.entry.turns,
            unpriced_turns: self.totals.unpriced_turns,
            untokenized_image_blocks: self.totals.untokenized_image_blocks,
            overflow_turns: self.totals.overflow_turns,
            max_total_tokens: config.max_total_tokens,
            max_total_micros: config.max_total_micros,
            near_cap: false,
        };
        usage.near_cap = config
            .warn_fraction
            .is_some_and(|fraction| cap_fraction(&usage) >= f64::from(fraction));
        usage
    }
}

/// Fraction of the tightest configured cap this reading has reached; `0.0` when
/// no cap is configured.
///
/// The float exists only for this comparison; money and tokens stay integral
/// everywhere else. A zero cap is rejected at construction, so neither division
/// can be by zero.
fn cap_fraction(usage: &CostUsage) -> f64 {
    let tokens = usage
        .max_total_tokens
        .map_or(0.0, |cap| usage.total_tokens() as f64 / cap as f64);
    let micros = usage
        .max_total_micros
        .map_or(0.0, |cap| usage.spent_micros as f64 / cap as f64);
    tokens.max(micros)
}

/// Cumulative token/currency ledger with a hard budget cap.
///
/// Holds the tiktoken encoder behind its own `Mutex` (the mutex serializes
/// access so the counter can be shared across `await` points) and the ledger
/// behind a second one. Lock discipline: encode first, release, then take the
/// ledger lock. The two are never held together.
pub struct CostPlugin {
    config: CostConfig,
    encoder: Mutex<tiktoken_rs::CoreBPE>,
    ledger: Mutex<CostLedger>,
}

impl CostPlugin {
    /// Build a validated plugin.
    ///
    /// # Errors
    ///
    /// [`PluginError::Internal`] when [`CostConfig`] validation fails (zero
    /// `max_tracked_models`, a `Some(0)` cap, a `warn_fraction` outside
    /// `(0.0, 1.0]` or without a cap, or [`UnpricedModelPolicy::Reject`] with
    /// no rate source) or the tiktoken encoder cannot be loaded.
    pub fn new(config: CostConfig) -> Result<Self, PluginError> {
        config.validate()?;
        let encoder = crate::tokenize::load_encoder(&config.encoder_name)?;
        Ok(Self {
            config,
            encoder: Mutex::new(encoder),
            ledger: Mutex::new(CostLedger::new()),
        })
    }

    /// Cheap reading: one lock, one `Copy` struct, no allocation.
    ///
    /// # Errors
    ///
    /// [`PluginError::Internal`] when the ledger lock is poisoned.
    pub fn usage(&self) -> Result<CostUsage, PluginError> {
        Ok(self.ledger()?.usage(&self.config))
    }

    /// Per-model breakdown, sorted by model id. Allocates; bounded by
    /// [`CostConfig::max_tracked_models`] plus the overflow bucket.
    ///
    /// # Errors
    ///
    /// [`PluginError::Internal`] when the ledger lock is poisoned.
    pub fn breakdown(&self) -> Result<Vec<(String, CostEntry)>, PluginError> {
        let mut out: Vec<(String, CostEntry)> = {
            let ledger = self.ledger()?;
            ledger
                .per_model
                .iter()
                .map(|(model, entry)| (model.clone(), *entry))
                .collect()
        };
        out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// Zero the ledger (billing-period rollover). Configuration is unchanged.
    ///
    /// # Errors
    ///
    /// [`PluginError::Internal`] when the ledger lock is poisoned.
    pub fn reset(&self) -> Result<(), PluginError> {
        let mut ledger = self.ledger()?;
        ledger.totals = CostTotals::default();
        ledger.per_model.clear();
        Ok(())
    }

    /// Estimated chargeable prompt tokens for `req`, using the same
    /// approximation the hooks use. Public so a caller can price a turn before
    /// building a client.
    ///
    /// # Errors
    ///
    /// [`PluginError::Internal`] when the encoder lock is poisoned or a
    /// `ToolCall`'s arguments cannot be serialized.
    pub fn estimate_request_tokens(&self, req: &UnifiedRequest) -> Result<u64, PluginError> {
        Ok(self.estimate_prompt(req)?.tokens)
    }

    /// Rates for `model`: resolver first, then the static table.
    pub fn rates_for(&self, model: &str) -> Option<ModelRates> {
        if let Some(resolver) = &self.config.pricing_resolver
            && let Some(rates) = resolver.resolve_rates(model)
        {
            return Some(rates);
        }
        self.config.pricing.get(model)
    }

    /// The tiktoken encoder guard.
    fn encoder(&self) -> Result<MutexGuard<'_, tiktoken_rs::CoreBPE>, PluginError> {
        self.encoder
            .lock()
            .map_err(|e| PluginError::Internal(format!("tiktoken encoder lock poisoned: {e}")))
    }

    /// The ledger guard.
    fn ledger(&self) -> Result<MutexGuard<'_, CostLedger>, PluginError> {
        self.ledger
            .lock()
            .map_err(|e| PluginError::Internal(format!("cost ledger lock poisoned: {e}")))
    }

    /// Chargeable prompt tokens plus the image blocks the estimator skipped.
    fn estimate_prompt(&self, req: &UnifiedRequest) -> Result<PromptEstimate, PluginError> {
        let encoder = self.encoder()?;
        let mut estimate = PromptEstimate::default();
        let count = |text: &str| encoder.encode_ordinary(text).len() as u64;
        for msg in &req.messages {
            estimate.tokens = estimate.tokens.saturating_add(count(role_prefix(msg.role)));
            for block in &msg.content {
                match block {
                    MessageContentBlock::Text(text) => {
                        estimate.tokens = estimate.tokens.saturating_add(count(text));
                    }
                    MessageContentBlock::Thinking { reasoning, .. } => {
                        estimate.tokens = estimate.tokens.saturating_add(count(reasoning));
                    }
                    MessageContentBlock::ToolCall {
                        name, arguments, ..
                    } => {
                        let args = serialize_json(arguments, "tool call arguments")?;
                        estimate.tokens = estimate
                            .tokens
                            .saturating_add(count(name))
                            .saturating_add(count(&args));
                    }
                    MessageContentBlock::ToolResult { output, .. } => {
                        estimate.tokens = estimate.tokens.saturating_add(count(output));
                    }
                    // Providers bill images by tile and dimension, which no
                    // text tokenizer reproduces; the counter is the signal.
                    MessageContentBlock::ImageBase64 { .. } => {
                        estimate.image_blocks = estimate.image_blocks.saturating_add(1);
                    }
                }
            }
        }
        // Tool schemas are prompt tokens the provider bills for.
        for tool in &req.tools {
            let schema = serialize_json(&tool.input_schema, "tool input schema")?;
            estimate.tokens = estimate
                .tokens
                .saturating_add(count(&tool.name))
                .saturating_add(count(&tool.description))
                .saturating_add(count(&schema));
        }
        Ok(estimate)
    }

    /// Estimated completion tokens of an aggregated response.
    ///
    /// Counts the block partition the client counts blocks over: `Text`,
    /// `Thinking.reasoning`, and `ToolCall` name plus arguments. `ToolResult`
    /// and `ImageBase64` carry no generated tokens.
    fn estimate_completion(&self, content: &[MessageContentBlock]) -> Result<u64, PluginError> {
        let encoder = self.encoder()?;
        let count = |text: &str| encoder.encode_ordinary(text).len() as u64;
        let mut tokens = 0u64;
        for block in content {
            match block {
                MessageContentBlock::Text(text) => {
                    tokens = tokens.saturating_add(count(text));
                }
                MessageContentBlock::Thinking { reasoning, .. } => {
                    tokens = tokens.saturating_add(count(reasoning));
                }
                MessageContentBlock::ToolCall {
                    name, arguments, ..
                } => {
                    let args = serialize_json(arguments, "tool call arguments")?;
                    tokens = tokens
                        .saturating_add(count(name))
                        .saturating_add(count(&args));
                }
                MessageContentBlock::ToolResult { .. }
                | MessageContentBlock::ImageBase64 { .. } => {}
            }
        }
        Ok(tokens)
    }
}

/// Chargeable prompt tokens plus the image blocks the estimator skipped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PromptEstimate {
    tokens: u64,
    image_blocks: u64,
}

/// Role prefix encoded ahead of a message's blocks.
///
/// The trailing space is part of the constant, so no per-message `String` is
/// built to join the role to its text.
fn role_prefix(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system ",
        MessageRole::User => "user ",
        MessageRole::Assistant => "assistant ",
        MessageRole::Tool => "tool ",
    }
}

/// Compact JSON text of `value`, for token counting.
///
/// A serialization failure is an error rather than a zero count: an uncounted
/// payload would silently understate the charge.
fn serialize_json(value: &serde_json::Value, what: &str) -> Result<String, PluginError> {
    serde_json::to_string(value)
        .map_err(|e| PluginError::Internal(format!("failed to serialize {what}: {e}")))
}

impl CucaPlugin for CostPlugin {
    fn name(&self) -> &'static str {
        "cost-accounting"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        // Encode first and drop the encoder guard before the ledger lock is
        // taken: the two mutexes are never held together.
        let estimate = self.estimate_prompt(req)?;
        let rates = self.rates_for(&req.model);
        if rates.is_none() && self.config.on_unpriced_model == UnpricedModelPolicy::Reject {
            return Err(PluginError::hook(
                "cost-accounting",
                "request",
                format!("no rates for model '{}'", req.model),
            ));
        }
        let turn_micros = rates.map_or(0, |r| {
            price_micros(estimate.tokens, r.input_micros_per_mtok)
        });
        let charge = RequestCharge {
            entry: CostEntry {
                prompt_tokens: estimate.tokens,
                spent_micros: turn_micros,
                ..Default::default()
            },
            unpriced: rates.is_none(),
            image_blocks: estimate.image_blocks,
        };

        let usage = {
            let mut ledger = self.ledger()?;
            // Enforcement is against the projected total, so a cap is never
            // crossed rather than merely detected afterwards. Nothing is
            // committed on the refusal path.
            let projected_tokens = ledger
                .totals
                .entry
                .total_tokens()
                .saturating_add(estimate.tokens);
            if let Some(cap) = self.config.max_total_tokens
                && projected_tokens > cap
            {
                return Err(PluginError::hook(
                    "cost-accounting",
                    "request",
                    format!(
                        "token budget exceeded: this turn would reach {projected_tokens} of \
                         {cap} tokens"
                    ),
                ));
            }
            let projected_micros = ledger.totals.entry.spent_micros.saturating_add(turn_micros);
            if let Some(cap) = self.config.max_total_micros
                && projected_micros > cap
            {
                return Err(PluginError::hook(
                    "cost-accounting",
                    "request",
                    format!(
                        "currency budget exceeded: this turn would reach {projected_micros} of \
                         {cap} micros"
                    ),
                ));
            }
            ledger.charge_request(&req.model, self.config.max_tracked_models, &charge);
            ledger.usage(&self.config)
        };

        // The marker scan works block by block: the marker rides in a single
        // `Text` block, so nothing joins a message's text into a fresh
        // `String` per message per request.
        if usage.near_cap
            && !req.messages.iter().any(|m| {
                m.content.iter().any(
                    |block| matches!(block, MessageContentBlock::Text(text) if text.contains(COST_WARNING_MARKER)),
                )
            })
        {
            let percent = cap_fraction(&usage) * 100.0;
            req.messages.push(UnifiedMessage::system(format!(
                "{COST_WARNING_MARKER} This client has used {percent:.0}% of its budget cap; \
                 wrap up soon.",
            )));
        }

        for observer in &self.config.observers {
            if let Err(e) = observer.observe(&usage) {
                // The turn is aborted, so its charge must not outlive it.
                self.ledger()?
                    .refund_request(&req.model, self.config.max_tracked_models, &charge);
                return Err(e);
            }
        }
        Ok(())
    }

    fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
        let completion_tokens = self.estimate_completion(&res.content)?;
        let rates = self.rates_for(&res.model);

        let usage = {
            let mut ledger = self.ledger()?;
            let mut delta = CostEntry {
                completion_tokens,
                turns: 1,
                ..Default::default()
            };
            let mut spend = rates.map_or(0, |r| {
                price_micros(completion_tokens, r.output_micros_per_mtok)
            });
            let mut refund_micros = 0u64;
            // The only provider-truth correction available: re-price the
            // cached portion of the prompt this ledger already charged at the
            // input rate. The clamp keeps the cache counter a subset of
            // `prompt_tokens` when the provider's tokenizer and tiktoken
            // disagree.
            if let Some(cache) = res.prompt_cache_usage {
                let charged =
                    ledger.charged_prompt_tokens(&res.model, self.config.max_tracked_models);
                let read = u64::from(cache.read_tokens).min(charged);
                let write = u64::from(cache.write_tokens);
                delta.cache_read_tokens = read;
                delta.cache_write_tokens = write;
                if let Some(r) = rates {
                    refund_micros = price_micros(read, r.input_micros_per_mtok);
                    spend = spend
                        .saturating_add(price_micros(read, r.cache_read_micros_per_mtok))
                        .saturating_add(price_micros(write, r.cache_write_micros_per_mtok));
                }
            }
            delta.spent_micros = spend;
            ledger.commit_response(
                &res.model,
                self.config.max_tracked_models,
                &delta,
                refund_micros,
            );
            ledger.usage(&self.config)
        };

        for observer in &self.config.observers {
            observer.observe(&usage)?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "plugin-cost"))]
mod tests {
    use super::*;
    use crate::request::PromptCacheUsage;
    use crate::types::{ProviderEndpoint, ToolDefinition};
    use serde_json::json;

    /// Rates that price input and output only.
    fn text_rates(input: u64, output: u64) -> ModelRates {
        ModelRates {
            input_micros_per_mtok: input,
            output_micros_per_mtok: output,
            ..Default::default()
        }
    }

    /// Resolver with a hit for one model name.
    struct FakeResolver;

    impl PricingResolver for FakeResolver {
        fn resolve_rates(&self, model: &str) -> Option<ModelRates> {
            (model == "resolved").then(|| text_rates(7, 7))
        }
    }

    /// Observer that records every reading handed to it.
    struct FakeObserver(Mutex<Vec<CostUsage>>);

    impl CostObserver for FakeObserver {
        fn observe(&self, usage: &CostUsage) -> Result<(), PluginError> {
            self.0.lock().unwrap().push(*usage);
            Ok(())
        }
    }

    /// Observer that always fails.
    struct FailingObserver;

    impl CostObserver for FailingObserver {
        fn observe(&self, _usage: &CostUsage) -> Result<(), PluginError> {
            Err(PluginError::Internal("observer failure".to_string()))
        }
    }

    /// A request whose single user message is long enough for a stable,
    /// double-digit token estimate.
    fn user_request(model: &str) -> UnifiedRequest {
        UnifiedRequest::new(model).add_user_message(
            "the quick brown fox jumps over the lazy dog, repeatedly and with enthusiasm",
        )
    }

    /// A terminal response whose `completion_tokens` is a block count, exactly
    /// as the client writes it.
    fn response_with(model: &str, content: Vec<MessageContentBlock>) -> UnifiedResponse {
        UnifiedResponse {
            model: model.to_string(),
            provider: ProviderEndpoint::Custom(String::new()),
            duration_secs: 0.0,
            prompt_tokens: 0,
            completion_tokens: 1,
            finish_reason: None,
            content,
            prompt_cache_usage: None,
        }
    }

    /// tiktoken count of `text` under the default encoder.
    fn encoded(text: &str) -> u64 {
        crate::tokenize::load_encoder("cl100k_base")
            .unwrap()
            .encode_ordinary(text)
            .len() as u64
    }

    /// Prompt estimate of `req` under a default plugin.
    fn estimate_of(req: &UnifiedRequest) -> u64 {
        CostPlugin::new(CostConfig::default())
            .unwrap()
            .estimate_request_tokens(req)
            .unwrap()
    }

    /// The `Internal` message of `err`, or a panic naming what arrived instead.
    fn internal_message(err: PluginError) -> String {
        match err {
            PluginError::Internal(msg) => msg,
            other => panic!("expected PluginError::Internal, got {other:?}"),
        }
    }

    /// The `HookFailure` message of `err`, asserting the plugin and stage.
    fn hook_message(err: PluginError) -> String {
        match err {
            PluginError::HookFailure {
                plugin,
                stage,
                message,
            } => {
                assert_eq!(plugin, "cost-accounting");
                assert_eq!(stage, "request");
                message
            }
            other => panic!("expected PluginError::HookFailure, got {other:?}"),
        }
    }

    /// The `PluginError` a rejected configuration produces.
    ///
    /// `CostPlugin` is deliberately not `Debug` (its `CoreBPE` is not), so
    /// `Result::unwrap_err` is unavailable here.
    fn rejection(config: CostConfig) -> PluginError {
        match CostPlugin::new(config) {
            Ok(_) => panic!("the configuration must be rejected"),
            Err(e) => e,
        }
    }

    #[test]
    fn default_config_is_valid() {
        let plugin = CostPlugin::new(CostConfig::default()).expect("the default config builds");
        let usage = plugin.usage().unwrap();
        assert_eq!(usage.total_tokens(), 0);
        assert_eq!(usage.max_total_tokens, None);
        assert!(!usage.near_cap);
    }

    #[test]
    fn zero_tracked_models_is_rejected() {
        let err = rejection(CostConfig {
            max_tracked_models: 0,
            ..Default::default()
        });
        assert!(internal_message(err).contains("max_tracked_models"));
    }

    #[test]
    fn zero_token_cap_is_rejected() {
        let err = rejection(CostConfig {
            max_total_tokens: Some(0),
            ..Default::default()
        });
        assert!(internal_message(err).contains("max_total_tokens"));
    }

    #[test]
    fn zero_currency_cap_is_rejected() {
        let err = rejection(CostConfig {
            max_total_micros: Some(0),
            ..Default::default()
        });
        assert!(internal_message(err).contains("max_total_micros"));
    }

    #[test]
    fn warn_fraction_outside_the_unit_interval_is_rejected() {
        for fraction in [0.0f32, 1.5, f32::NAN] {
            let err = rejection(CostConfig {
                max_total_tokens: Some(1_000),
                warn_fraction: Some(fraction),
                ..Default::default()
            });
            assert!(
                internal_message(err).contains("warn_fraction"),
                "{fraction} must be rejected by the warn_fraction guard"
            );
        }
        // The interval is closed at the top: exactly 1.0 is legal.
        assert!(
            CostPlugin::new(CostConfig {
                max_total_tokens: Some(1_000),
                warn_fraction: Some(1.0),
                ..Default::default()
            })
            .is_ok()
        );
    }

    #[test]
    fn warn_fraction_without_any_cap_is_rejected() {
        let err = rejection(CostConfig {
            warn_fraction: Some(0.8),
            ..Default::default()
        });
        assert!(internal_message(err).contains("warn_fraction requires"));
    }

    #[test]
    fn reject_policy_without_any_rate_source_is_rejected() {
        let err = rejection(CostConfig {
            on_unpriced_model: UnpricedModelPolicy::Reject,
            ..Default::default()
        });
        assert!(internal_message(err).contains("refuse every turn"));

        // A resolver alone is a rate source, so the same policy is accepted.
        assert!(
            CostPlugin::new(CostConfig {
                on_unpriced_model: UnpricedModelPolicy::Reject,
                pricing_resolver: Some(Arc::new(FakeResolver)),
                ..Default::default()
            })
            .is_ok()
        );
    }

    #[test]
    fn unknown_encoder_name_is_rejected() {
        let err = rejection(CostConfig {
            encoder_name: "not-an-encoder".to_string(),
            ..Default::default()
        });
        let msg = internal_message(err);
        assert!(msg.contains("not-an-encoder"), "{msg}");
    }

    #[test]
    fn resolver_rates_win_over_the_static_table() {
        let plugin = CostPlugin::new(CostConfig {
            pricing: PricingTable::new()
                .with_model("resolved", text_rates(1, 1))
                .with_model("table-only", text_rates(2, 2)),
            pricing_resolver: Some(Arc::new(FakeResolver)),
            ..Default::default()
        })
        .unwrap();

        assert_eq!(plugin.rates_for("resolved"), Some(text_rates(7, 7)));
        assert_eq!(plugin.rates_for("table-only"), Some(text_rates(2, 2)));
        assert_eq!(plugin.rates_for("unknown"), None);
    }

    #[test]
    fn pricing_is_integer_and_saturates() {
        assert_eq!(price_micros(1_000_000, 3_000_000), 3_000_000);
        assert_eq!(price_micros(0, u64::MAX), 0);
        // The u128 product is exact; only the cast back saturates.
        assert_eq!(price_micros(2_000_000, u64::MAX), u64::MAX);
        assert_eq!(price_micros(u64::MAX, u64::MAX), u64::MAX);
    }

    #[test]
    fn sub_million_token_counts_round_down_deterministically() {
        assert_eq!(price_micros(1, 3_000_000), 3);
        assert_eq!(price_micros(999_999, 3_000_000), 2_999_997);
        // Under one micro-unit truncates to zero rather than rounding up.
        assert_eq!(price_micros(1, 1), 0);
        assert_eq!(price_micros(999_999, 1), 0);
        assert_eq!(price_micros(1_000_000, 1), 1);
    }

    #[test]
    fn estimate_counts_text_thinking_tool_call_and_tool_result_blocks() {
        let plugin = CostPlugin::new(CostConfig::default()).unwrap();
        let bare = UnifiedRequest::new("m").add_message(UnifiedMessage {
            role: MessageRole::User,
            content: Vec::new(),
            name: None,
            tool_call_id: None,
        });
        let base = plugin.estimate_request_tokens(&bare).unwrap();
        assert!(base > 0, "the role prefix alone is chargeable");

        let blocks = [
            MessageContentBlock::Text("a chargeable sentence".to_string()),
            MessageContentBlock::Thinking {
                reasoning: "a chargeable reasoning trace".to_string(),
                signature: None,
            },
            MessageContentBlock::ToolCall {
                id: "call-1".to_string(),
                name: "read_file".to_string(),
                arguments: json!({ "path": "/tmp/report.txt" }),
            },
            MessageContentBlock::ToolResult {
                tool_call_id: "call-1".to_string(),
                output: "a chargeable tool output".to_string(),
            },
        ];
        for block in blocks {
            let req = UnifiedRequest::new("m").add_message(UnifiedMessage {
                role: MessageRole::User,
                content: vec![block.clone()],
                name: None,
                tool_call_id: None,
            });
            assert!(
                plugin.estimate_request_tokens(&req).unwrap() > base,
                "{block:?} must be charged"
            );
        }
    }

    #[test]
    fn estimate_counts_tool_definitions_from_the_request() {
        let plugin = CostPlugin::new(CostConfig::default()).unwrap();
        let without = user_request("m");
        let with = user_request("m").add_tool(ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file from disk and return its contents".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
            }),
        });

        assert!(
            plugin.estimate_request_tokens(&with).unwrap()
                > plugin.estimate_request_tokens(&without).unwrap()
        );
    }

    #[test]
    fn estimate_skips_images_and_counts_them() {
        let plugin = CostPlugin::new(CostConfig::default()).unwrap();
        let text_only = user_request("m");
        let mut with_image = user_request("m");
        with_image.messages[0]
            .content
            .push(MessageContentBlock::ImageBase64 {
                media_type: "image/png".to_string(),
                data: "aGVsbG8gaW1hZ2UgcGF5bG9hZA==".to_string(),
            });

        assert_eq!(
            plugin.estimate_request_tokens(&with_image).unwrap(),
            plugin.estimate_request_tokens(&text_only).unwrap(),
            "an image block contributes no tokens"
        );

        let mut req = with_image;
        plugin.on_request(&mut req).unwrap();
        assert_eq!(plugin.usage().unwrap().untokenized_image_blocks, 1);
    }

    #[test]
    fn a_turn_within_the_token_cap_is_charged() {
        let req = user_request("m");
        let estimate = estimate_of(&req);
        let plugin = CostPlugin::new(CostConfig {
            pricing: PricingTable::new().with_model("m", text_rates(3_000_000, 0)),
            max_total_tokens: Some(estimate * 10),
            ..Default::default()
        })
        .unwrap();

        let mut req = req;
        plugin.on_request(&mut req).unwrap();

        let usage = plugin.usage().unwrap();
        assert_eq!(usage.prompt_tokens, estimate);
        assert_eq!(usage.spent_micros, price_micros(estimate, 3_000_000));
        assert_eq!(usage.turns, 0, "turns count commits, not charges");
        assert_eq!(usage.unpriced_turns, 0);
    }

    #[test]
    fn a_turn_that_would_cross_the_token_cap_is_refused_and_charges_nothing() {
        let plugin = CostPlugin::new(CostConfig {
            max_total_tokens: Some(1),
            ..Default::default()
        })
        .unwrap();
        let before = plugin.usage().unwrap();

        let mut req = user_request("m");
        let message = hook_message(plugin.on_request(&mut req).unwrap_err());
        assert!(message.contains("token budget exceeded"), "{message}");

        assert_eq!(plugin.usage().unwrap(), before);
        assert!(plugin.breakdown().unwrap().is_empty());
        assert_eq!(req.messages.len(), 1, "the refused turn is not mutated");
    }

    #[test]
    fn a_turn_that_would_cross_the_currency_cap_is_refused() {
        let plugin = CostPlugin::new(CostConfig {
            pricing: PricingTable::new().with_model("m", text_rates(1_000_000_000, 0)),
            max_total_micros: Some(1),
            ..Default::default()
        })
        .unwrap();
        let before = plugin.usage().unwrap();

        let mut req = user_request("m");
        let message = hook_message(plugin.on_request(&mut req).unwrap_err());
        assert!(message.contains("currency budget exceeded"), "{message}");
        assert_eq!(plugin.usage().unwrap(), before);
    }

    #[test]
    fn an_unpriced_model_is_refused_under_reject_policy() {
        let plugin = CostPlugin::new(CostConfig {
            pricing: PricingTable::new().with_model("priced", text_rates(1, 1)),
            on_unpriced_model: UnpricedModelPolicy::Reject,
            ..Default::default()
        })
        .unwrap();

        let mut req = user_request("other");
        let message = hook_message(plugin.on_request(&mut req).unwrap_err());
        assert!(message.contains("no rates for model 'other'"), "{message}");
        assert_eq!(plugin.usage().unwrap().prompt_tokens, 0);

        // The priced model still goes through under the same policy.
        let mut priced = user_request("priced");
        plugin.on_request(&mut priced).unwrap();
        assert!(plugin.usage().unwrap().prompt_tokens > 0);
    }

    #[test]
    fn an_unpriced_model_is_charged_in_tokens_only_under_count_tokens_only() {
        let plugin = CostPlugin::new(CostConfig::default()).unwrap();
        let mut req = user_request("unpriced");
        plugin.on_request(&mut req).unwrap();

        let usage = plugin.usage().unwrap();
        assert!(usage.prompt_tokens > 0);
        assert_eq!(usage.spent_micros, 0);
        assert_eq!(usage.unpriced_turns, 1);
    }

    #[test]
    fn crossing_warn_fraction_injects_one_system_message() {
        let req = user_request("m");
        let estimate = estimate_of(&req);
        let plugin = CostPlugin::new(CostConfig {
            max_total_tokens: Some(estimate * 2),
            warn_fraction: Some(0.5),
            ..Default::default()
        })
        .unwrap();

        let mut req = req;
        plugin.on_request(&mut req).unwrap();

        assert_eq!(req.messages.len(), 2);
        let warning = &req.messages[1];
        assert_eq!(warning.role, MessageRole::System);
        match &warning.content[0] {
            MessageContentBlock::Text(text) => {
                assert!(text.starts_with(COST_WARNING_MARKER), "{text}");
            }
            other => panic!("expected a Text block, got {other:?}"),
        }
        assert!(plugin.usage().unwrap().near_cap);
    }

    #[test]
    fn a_second_request_carrying_the_marker_is_not_warned_again() {
        let req = user_request("m");
        let estimate = estimate_of(&req);
        let plugin = CostPlugin::new(CostConfig {
            max_total_tokens: Some(estimate * 8),
            warn_fraction: Some(0.1),
            ..Default::default()
        })
        .unwrap();

        let mut req = req;
        plugin.on_request(&mut req).unwrap();
        plugin.on_request(&mut req).unwrap();

        let markers = req
            .messages
            .iter()
            .filter(|m| {
                m.content.iter().any(|block| {
                    matches!(block, MessageContentBlock::Text(t) if t.contains(COST_WARNING_MARKER))
                })
            })
            .count();
        assert_eq!(markers, 1);
        assert_eq!(req.messages.len(), 2);
    }

    #[test]
    fn no_warning_below_warn_fraction() {
        let req = user_request("m");
        let estimate = estimate_of(&req);
        let plugin = CostPlugin::new(CostConfig {
            max_total_tokens: Some(estimate * 10),
            warn_fraction: Some(0.9),
            ..Default::default()
        })
        .unwrap();

        let mut req = req;
        plugin.on_request(&mut req).unwrap();

        assert_eq!(req.messages.len(), 1);
        assert!(!plugin.usage().unwrap().near_cap);
    }

    #[test]
    fn completion_tokens_are_estimated_from_content_not_read_from_the_response() {
        let plugin = CostPlugin::new(CostConfig::default()).unwrap();
        let answer = "a deliberately long completion whose tiktoken count is far above one block";
        let res = response_with("m", vec![MessageContentBlock::Text(answer.to_string())]);
        assert_eq!(
            res.completion_tokens, 1,
            "the client writes a block count here, not a token count"
        );

        plugin.on_response_complete(&res).unwrap();

        let expected = encoded(answer);
        assert!(expected > 1);
        let usage = plugin.usage().unwrap();
        assert_eq!(usage.completion_tokens, expected);
        assert_eq!(usage.turns, 1);
    }

    #[test]
    fn tool_result_blocks_in_the_response_are_not_charged_as_completion() {
        let plugin = CostPlugin::new(CostConfig::default()).unwrap();
        let res = response_with(
            "m",
            vec![
                MessageContentBlock::ToolResult {
                    tool_call_id: "call-1".to_string(),
                    output: "a long tool output that carries no generated tokens".to_string(),
                },
                MessageContentBlock::ImageBase64 {
                    media_type: "image/png".to_string(),
                    data: "aGVsbG8gaW1hZ2UgcGF5bG9hZA==".to_string(),
                },
            ],
        );

        plugin.on_response_complete(&res).unwrap();

        let usage = plugin.usage().unwrap();
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.turns, 1);
    }

    #[test]
    fn prompt_cache_usage_reprices_prompt_tokens_at_the_cache_rates() {
        let rates = ModelRates {
            input_micros_per_mtok: 3_000_000,
            output_micros_per_mtok: 0,
            cache_read_micros_per_mtok: 300_000,
            cache_write_micros_per_mtok: 3_750_000,
        };
        let plugin = CostPlugin::new(CostConfig {
            pricing: PricingTable::new().with_model("m", rates),
            ..Default::default()
        })
        .unwrap();

        let mut req = user_request("m");
        let estimate = estimate_of(&req);
        plugin.on_request(&mut req).unwrap();
        let charged = plugin.usage().unwrap().spent_micros;
        assert_eq!(charged, price_micros(estimate, 3_000_000));

        let read = estimate / 2;
        let write = 4u32;
        let mut res = response_with("m", Vec::new());
        res.prompt_cache_usage = Some(PromptCacheUsage {
            read_tokens: read as u32,
            write_tokens: write,
        });
        plugin.on_response_complete(&res).unwrap();

        let usage = plugin.usage().unwrap();
        assert_eq!(usage.cache_read_tokens, read);
        assert_eq!(usage.cache_write_tokens, u64::from(write));
        let expected = charged - price_micros(read, 3_000_000)
            + price_micros(read, 300_000)
            + price_micros(u64::from(write), 3_750_000);
        assert_eq!(usage.spent_micros, expected);
        assert!(
            usage.spent_micros < charged,
            "cached prompt tokens cost less than full-price input tokens"
        );
    }

    #[test]
    fn a_response_without_prompt_cache_usage_leaves_the_prompt_charge_alone() {
        let plugin = CostPlugin::new(CostConfig {
            pricing: PricingTable::new().with_model("m", text_rates(3_000_000, 0)),
            ..Default::default()
        })
        .unwrap();

        let mut req = user_request("m");
        plugin.on_request(&mut req).unwrap();
        let charged = plugin.usage().unwrap();

        let res = response_with("m", Vec::new());
        assert!(res.prompt_cache_usage.is_none());
        plugin.on_response_complete(&res).unwrap();

        let usage = plugin.usage().unwrap();
        assert_eq!(usage.spent_micros, charged.spent_micros);
        assert_eq!(usage.prompt_tokens, charged.prompt_tokens);
        assert_eq!(usage.cache_read_tokens, 0);
        assert_eq!(usage.cache_write_tokens, 0);
        assert_eq!(usage.turns, 1);
    }

    #[test]
    fn cache_read_tokens_exceeding_the_estimate_saturate_instead_of_underflowing() {
        let plugin = CostPlugin::new(CostConfig {
            pricing: PricingTable::new().with_model("m", text_rates(3_000_000, 0)),
            ..Default::default()
        })
        .unwrap();

        let mut req = user_request("m");
        let estimate = estimate_of(&req);
        plugin.on_request(&mut req).unwrap();

        // The provider's tokenizer disagrees wildly: it reports far more cached
        // prompt tokens than this ledger ever charged.
        let mut res = response_with("m", Vec::new());
        res.prompt_cache_usage = Some(PromptCacheUsage {
            read_tokens: u32::MAX,
            write_tokens: 0,
        });
        plugin.on_response_complete(&res).unwrap();

        let usage = plugin.usage().unwrap();
        assert_eq!(
            usage.cache_read_tokens, estimate,
            "the cache counter is clamped to the prompt tokens actually charged"
        );
        assert_eq!(usage.spent_micros, 0, "the refund cannot exceed the charge");
    }

    #[test]
    fn models_beyond_max_tracked_models_fold_into_the_overflow_bucket() {
        let max_tracked = 2usize;
        let plugin = CostPlugin::new(CostConfig {
            max_tracked_models: max_tracked,
            ..Default::default()
        })
        .unwrap();

        let models = ["a", "b", "c", "d"];
        let mut charged_prompt = 0u64;
        let mut charged_completion = 0u64;
        for model in models {
            let mut req = user_request(model);
            charged_prompt += estimate_of(&req);
            plugin.on_request(&mut req).unwrap();

            let answer = "a short answer";
            charged_completion += encoded(answer);
            plugin
                .on_response_complete(&response_with(
                    model,
                    vec![MessageContentBlock::Text(answer.to_string())],
                ))
                .unwrap();
        }

        let breakdown = plugin.breakdown().unwrap();
        assert!(breakdown.len() <= max_tracked + 1);
        assert!(breakdown.iter().any(|(model, _)| model == OVERFLOW_MODEL));

        let usage = plugin.usage().unwrap();
        assert_eq!(usage.overflow_turns, 2, "two models beyond the cap folded");
        assert_eq!(usage.turns, models.len() as u64);
        // Totals stay exact: only per-model attribution degrades.
        assert_eq!(usage.prompt_tokens, charged_prompt);
        assert_eq!(usage.completion_tokens, charged_completion);
        let summed: u64 = breakdown
            .iter()
            .map(|(_, entry)| entry.total_tokens())
            .sum();
        assert_eq!(summed, usage.total_tokens());
    }

    #[test]
    fn breakdown_is_sorted_by_model_id() {
        let plugin = CostPlugin::new(CostConfig::default()).unwrap();
        for model in ["zeta", "alpha", "mu"] {
            let mut req = user_request(model);
            plugin.on_request(&mut req).unwrap();
        }

        let models: Vec<String> = plugin
            .breakdown()
            .unwrap()
            .into_iter()
            .map(|(model, _)| model)
            .collect();
        assert_eq!(models, vec!["alpha", "mu", "zeta"]);
    }

    #[test]
    fn reset_zeroes_the_ledger_and_keeps_the_configuration() {
        let req = user_request("m");
        let estimate = estimate_of(&req);
        let plugin = CostPlugin::new(CostConfig {
            max_total_tokens: Some(estimate),
            ..Default::default()
        })
        .unwrap();

        let mut first = req.clone();
        plugin.on_request(&mut first).unwrap();
        // The cap is now exactly spent, so a second turn is refused.
        let mut second = req.clone();
        assert!(plugin.on_request(&mut second).is_err());

        plugin.reset().unwrap();

        let usage = plugin.usage().unwrap();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.spent_micros, 0);
        assert_eq!(usage.turns, 0);
        assert!(plugin.breakdown().unwrap().is_empty());
        assert_eq!(
            usage.max_total_tokens,
            Some(estimate),
            "reset rolls the ledger, not the configuration"
        );

        let mut third = req;
        plugin
            .on_request(&mut third)
            .expect("the cap has headroom again");
    }

    #[test]
    fn usage_near_cap_flips_exactly_at_warn_fraction() {
        let req = user_request("m");
        let estimate = estimate_of(&req);
        // Two turns of `estimate` tokens against a cap of `4 * estimate`: the
        // first reading sits at 0.25, the second at exactly 0.5.
        let plugin = CostPlugin::new(CostConfig {
            max_total_tokens: Some(estimate * 4),
            warn_fraction: Some(0.5),
            ..Default::default()
        })
        .unwrap();

        let mut first = req.clone();
        plugin.on_request(&mut first).unwrap();
        assert!(!plugin.usage().unwrap().near_cap, "0.25 is below 0.5");

        let mut second = req;
        plugin.on_request(&mut second).unwrap();
        assert!(plugin.usage().unwrap().near_cap, "0.5 meets 0.5");
    }

    #[test]
    fn observers_see_a_reading_on_request_and_on_response_complete() {
        let observer = Arc::new(FakeObserver(Mutex::new(Vec::new())));
        let plugin = CostPlugin::new(CostConfig {
            observers: vec![observer.clone()],
            ..Default::default()
        })
        .unwrap();

        let mut req = user_request("m");
        let estimate = estimate_of(&req);
        plugin.on_request(&mut req).unwrap();
        let answer = "a short answer";
        plugin
            .on_response_complete(&response_with(
                "m",
                vec![MessageContentBlock::Text(answer.to_string())],
            ))
            .unwrap();

        let readings = observer.0.lock().unwrap();
        assert_eq!(readings.len(), 2);
        assert_eq!(readings[0].prompt_tokens, estimate);
        assert_eq!(readings[0].turns, 0);
        assert_eq!(readings[1].completion_tokens, encoded(answer));
        assert_eq!(readings[1].turns, 1);
    }

    #[test]
    fn an_observer_error_in_on_request_aborts_the_turn_and_the_charge_is_not_visible() {
        let plugin = CostPlugin::new(CostConfig {
            observers: vec![Arc::new(FailingObserver)],
            ..Default::default()
        })
        .unwrap();
        let before = plugin.usage().unwrap();

        let mut req = user_request("m");
        let err = plugin.on_request(&mut req).unwrap_err();
        assert_eq!(internal_message(err), "observer failure");

        assert_eq!(plugin.usage().unwrap(), before);
        assert!(plugin.breakdown().unwrap().is_empty());
    }

    #[test]
    fn an_observer_error_in_on_response_complete_is_returned_by_the_hook() {
        let plugin = CostPlugin::new(CostConfig {
            observers: vec![Arc::new(FailingObserver)],
            ..Default::default()
        })
        .unwrap();

        let res = response_with("m", vec![MessageContentBlock::Text("answer".to_string())]);
        let err = plugin.on_response_complete(&res).unwrap_err();
        assert_eq!(internal_message(err), "observer failure");

        // The client logs this error and never surfaces it, so the commit that
        // preceded the observer stands.
        assert_eq!(plugin.usage().unwrap().turns, 1);
    }

    #[test]
    fn name_is_the_stable_identifier() {
        let plugin = CostPlugin::new(CostConfig::default()).unwrap();
        assert_eq!(plugin.name(), "cost-accounting");
    }
}
