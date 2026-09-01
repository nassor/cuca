//! Client-side outbound throttle (`plugin-rate-limit`): an integer token
//! bucket over request rate, plus a hard cap on concurrently in-flight turns.
//!
//! [`RateLimiter`] paces a caller that fans out over
//! [`CucaClient::generate_stream`](crate::client::CucaClient::generate_stream)
//! so it stays inside a provider's published quota instead of discovering the
//! quota as [`CucaError::Http`](crate::error::CucaError::Http) with status 429.
//!
//! # Explicit-call contract
//!
//! `RateLimiter` is not a [`CucaPlugin`](crate::plugin::CucaPlugin). It has no
//! request or stream hooks, so passing it to `register_plugin` is a compile
//! error rather than an inert registration. The entry points are
//! [`RateLimiter::acquire`], [`RateLimiter::try_acquire`], the returned
//! [`RateLimitPermit`], and [`RateLimiter::usage`].
//!
//! The hook shape cannot carry this capability.
//! [`CucaPlugin::on_request`](crate::plugin::CucaPlugin::on_request) is
//! synchronous, so a hook can only reject a request, never pace it. Pacing is
//! the capability. A hook-acquired concurrency permit also leaks on three real
//! paths: `generate_stream` returning `Err` after the `on_request` hooks ran,
//! a consumer dropping the stream before its end, and the speculative arm that
//! returns the orchestrator stream unwrapped. None of those reach a terminal
//! hook. An RAII [`RateLimitPermit`] releases on all of them.
//!
//! # Mandatory hand-off
//!
//! The permit's lifetime is the whole turn: hold it across `generate_stream`
//! *and* the stream drain, then drop it.
//!
//! ```ignore
//! let permit = limiter.acquire().await?;
//! let mut stream = client.generate_stream(request).await?;
//! while let Some(block) = stream.next().await { /* ... */ }
//! drop(permit);
//! ```
//!
//! Dropping the permit before the stream is drained under-counts concurrency;
//! [`RateLimiter::usage`] is the gauge that makes the mistake observable.
//!
//! # Ordering and quota semantics
//!
//! [`RateLimiter::acquire`] takes a concurrency slot first and a token second.
//! A caller that gives up waiting for a slot has therefore spent no quota, and
//! the total number of parked callers is already bounded by `max_concurrent +
//! max_waiters`. A token is spent, never returned: dropping a permit returns
//! its slot only.
//!
//! # Bounds
//!
//! [`RateLimitConfig::burst`] caps accumulated credit, so a long idle never
//! banks unbounded tokens. [`RateLimitConfig::max_concurrent`] caps granted
//! permits. [`RateLimitConfig::max_waiters`] caps parked callers and rejects
//! at the cap with [`RateLimitError::QueueFull`] rather than queueing, which
//! is what bounds the semaphore's own waiter list. No per-request data is
//! retained: no request clone, no key, no log, no map. The near-cap seam is
//! [`RateLimitConfig::warn_fraction`] plus [`RateLimitObserver`], the same
//! shape the memory plugin's `ContextUsageObserver` uses.
//!
//! # Determinism
//!
//! All bucket arithmetic is integer thousandths of a token with a preserved
//! sub-token remainder, driven through the crate-private `RateClock`. The
//! production clock is monotonic (`Instant`-based), unlike the prompt cache's
//! wall-clock `CacheClock`: a limiter exports nothing across processes, so it
//! has no reason to read the system clock and every reason not to, since a
//! system-time jump would otherwise mint a burst. The bucket mutex is never
//! held across an `.await`.
//!
//! # Runtime coupling
//!
//! [`RateLimiter::acquire`] needs a Tokio reactor (`tokio::time` for the wait
//! budget, `tokio::sync::Semaphore` for the slots).
//! [`RateLimiter::try_acquire`] and [`RateLimiter::usage`] are runtime-free,
//! so a non-Tokio caller still gets admission control.
//!
//! # Not covered
//!
//! No 429 or `Retry-After` feedback loop. A non-2xx response becomes
//! [`CucaError::Http`](crate::error::CucaError::Http) inside the provider
//! adapters and is propagated by `?` before any stream wrapper exists, so no
//! plugin-visible surface ever sees a status; response headers are never
//! captured at all. A caller that still hits a 429 widens `interval` or lowers
//! `max_requests`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::PluginError;

/// One whole token, in the thousandths the bucket counts in.
const MILLI_TOKEN: u64 = 1_000;

/// Error returned by [`RateLimiter`] and [`RateLimitConfig`] operations.
#[derive(Debug, Clone)]
pub enum RateLimitError {
    /// Invalid configuration (zero rate, interval, burst, concurrency, queue
    /// bound, wait budget, or an out-of-range warn fraction).
    Config(String),
    /// No concurrency slot is free and the waiter queue is already at
    /// `max_waiters`: the acquire is refused instead of queueing further.
    QueueFull {
        /// Callers already parked in [`RateLimiter::acquire`].
        waiters: usize,
        /// The configured queue bound.
        max_waiters: usize,
    },
    /// No concurrency slot is free right now. [`RateLimiter::try_acquire`]
    /// only; [`RateLimiter::acquire`] waits instead.
    Busy {
        /// Permits currently held.
        in_flight: usize,
        /// The configured concurrency cap.
        max_concurrent: usize,
    },
    /// No token was available within the wait budget; `retry_after` is the
    /// remaining time until the bucket holds one whole token.
    Exhausted {
        /// Time until the bucket refills to one whole token.
        retry_after: Duration,
    },
    /// A [`RateLimitObserver`] refused the acquire.
    Observer(PluginError),
    /// The internal bucket mutex was poisoned by a panicking lock holder.
    Lock(String),
}

impl std::fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLimitError::Config(msg) => write!(f, "rate limit configuration error: {msg}"),
            RateLimitError::QueueFull {
                waiters,
                max_waiters,
            } => write!(
                f,
                "rate limit queue is full: {waiters} of {max_waiters} waiters are already parked"
            ),
            RateLimitError::Busy {
                in_flight,
                max_concurrent,
            } => write!(
                f,
                "rate limit concurrency cap reached: {in_flight} of {max_concurrent} slots in flight"
            ),
            RateLimitError::Exhausted { retry_after } => write!(
                f,
                "rate limit token bucket is exhausted; retry after {} ms",
                retry_after.as_millis()
            ),
            RateLimitError::Observer(err) => {
                write!(f, "rate limit observer refused the acquire: {err}")
            }
            RateLimitError::Lock(msg) => write!(f, "rate limit lock error: {msg}"),
        }
    }
}

impl std::error::Error for RateLimitError {}

/// Lossy interop edge into the crate's plugin error contract.
///
/// `Config` becomes a validation failure; `Observer` round-trips the wrapped
/// error unchanged; the throttle variants collapse into
/// [`PluginError::Internal`] and keep their numbers only in the message,
/// because no existing variant carries a retry-after and the crate reuses
/// variants rather than adding one. Callers that need the typed `retry_after`
/// match on [`RateLimitError`] before converting.
impl From<RateLimitError> for PluginError {
    fn from(error: RateLimitError) -> Self {
        match error {
            RateLimitError::Config(message) => PluginError::Validation {
                schema: "rate-limit-config".to_string(),
                message,
            },
            RateLimitError::Observer(inner) => inner,
            other => PluginError::Internal(other.to_string()),
        }
    }
}

/// A cheap usage reading of a [`RateLimiter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitUsage {
    /// Whole tokens currently in the bucket (after refilling to "now").
    pub available_tokens: u32,
    /// Permits currently held: `max_concurrent - available_permits()`.
    pub in_flight: usize,
    /// Callers currently parked in [`RateLimiter::acquire`].
    pub waiting: usize,
}

/// Observes limiter pressure without changing admission (reporting/UI gauge
/// seam).
pub trait RateLimitObserver: Send + Sync {
    /// Handed a reading when the waiter queue crosses
    /// [`RateLimitConfig::warn_fraction`]; an `Err` aborts that acquire.
    ///
    /// # Errors
    ///
    /// Whatever the implementation reports; the acquire fails with
    /// [`RateLimitError::Observer`] carrying it, and the concurrency slot the
    /// acquire had already taken is released.
    fn observe(&self, usage: &RateLimitUsage) -> Result<(), PluginError>;
}

/// Validated bounds for a [`RateLimiter`].
#[derive(Clone)]
pub struct RateLimitConfig {
    /// Tokens replenished per `interval`; must be non-zero.
    pub max_requests: u32,
    /// Refill window; must be non-zero.
    pub interval: Duration,
    /// Bucket capacity, i.e. the largest instantaneous burst; must be
    /// non-zero. Defaults to `max_requests`.
    pub burst: u32,
    /// Hard cap on concurrently held permits; must be non-zero and at most
    /// [`Semaphore::MAX_PERMITS`].
    pub max_concurrent: usize,
    /// Hard cap on callers parked in `acquire`; must be non-zero. At the cap
    /// an acquire is refused with [`RateLimitError::QueueFull`] rather than
    /// queued.
    pub max_waiters: usize,
    /// Per-acquire wait budget; must be non-zero. Defaults to `interval`.
    pub max_wait: Duration,
    /// Warn when `waiting / max_waiters` reaches this fraction; must be in
    /// `(0.0, 1.0]` when set. `None` disables.
    pub warn_fraction: Option<f32>,
    /// Observers handed a reading when `warn_fraction` is crossed.
    pub observers: Vec<Arc<dyn RateLimitObserver>>,
}

/// Every bound plus the observer count: `dyn RateLimitObserver` is not
/// [`Debug`](std::fmt::Debug), so the observers themselves cannot be printed.
impl std::fmt::Debug for RateLimitConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimitConfig")
            .field("max_requests", &self.max_requests)
            .field("interval", &self.interval)
            .field("burst", &self.burst)
            .field("max_concurrent", &self.max_concurrent)
            .field("max_waiters", &self.max_waiters)
            .field("max_wait", &self.max_wait)
            .field("warn_fraction", &self.warn_fraction)
            .field("observers", &self.observers.len())
            .finish()
    }
}

impl RateLimitConfig {
    /// Build a validated configuration: `burst = max_requests`,
    /// `max_wait = interval`, no warn fraction, no observers.
    ///
    /// # Errors
    ///
    /// [`RateLimitError::Config`] for a zero `max_requests`, `interval`,
    /// `max_concurrent`, or `max_waiters`, or a `max_concurrent` above
    /// [`Semaphore::MAX_PERMITS`].
    pub fn new(
        max_requests: u32,
        interval: Duration,
        max_concurrent: usize,
        max_waiters: usize,
    ) -> Result<Self, RateLimitError> {
        let config = Self {
            max_requests,
            interval,
            burst: max_requests,
            max_concurrent,
            max_waiters,
            max_wait: interval,
            warn_fraction: None,
            observers: Vec::new(),
        };
        config.validate()?;
        Ok(config)
    }

    /// Override the burst capacity.
    ///
    /// # Errors
    ///
    /// [`RateLimitError::Config`] when `burst` is zero.
    pub fn with_burst(mut self, burst: u32) -> Result<Self, RateLimitError> {
        self.burst = burst;
        self.validate()?;
        Ok(self)
    }

    /// Override the per-acquire wait budget.
    ///
    /// # Errors
    ///
    /// [`RateLimitError::Config`] when `max_wait` is zero.
    pub fn with_max_wait(mut self, max_wait: Duration) -> Result<Self, RateLimitError> {
        self.max_wait = max_wait;
        self.validate()?;
        Ok(self)
    }

    /// Set the near-cap warning fraction.
    ///
    /// # Errors
    ///
    /// [`RateLimitError::Config`] when `fraction` is outside `(0.0, 1.0]`.
    pub fn with_warn_fraction(mut self, fraction: f32) -> Result<Self, RateLimitError> {
        self.warn_fraction = Some(fraction);
        self.validate()?;
        Ok(self)
    }

    /// Register an observer for near-cap warnings.
    #[must_use]
    pub fn with_observer(mut self, observer: Arc<dyn RateLimitObserver>) -> Self {
        self.observers.push(observer);
        self
    }

    /// Re-check every bound. Public because the fields are public, so a struct
    /// literal can bypass [`Self::new`]; [`RateLimiter::new`] calls it.
    ///
    /// # Errors
    ///
    /// [`RateLimitError::Config`] for any violated bound above.
    pub fn validate(&self) -> Result<(), RateLimitError> {
        if self.max_requests == 0 {
            return Err(config_error("max_requests must be non-zero"));
        }
        if self.interval.is_zero() {
            return Err(config_error("interval must be non-zero"));
        }
        if self.burst == 0 {
            return Err(config_error("burst must be non-zero"));
        }
        if self.max_concurrent == 0 {
            return Err(config_error("max_concurrent must be non-zero"));
        }
        if self.max_concurrent > Semaphore::MAX_PERMITS {
            return Err(RateLimitError::Config(format!(
                "max_concurrent must be at most {}, got {}",
                Semaphore::MAX_PERMITS,
                self.max_concurrent
            )));
        }
        if self.max_waiters == 0 {
            return Err(config_error("max_waiters must be non-zero"));
        }
        if self.max_wait.is_zero() {
            return Err(config_error("max_wait must be non-zero"));
        }
        if let Some(fraction) = self.warn_fraction
            && (!fraction.is_finite() || fraction <= 0.0 || fraction > 1.0)
        {
            return Err(RateLimitError::Config(format!(
                "warn_fraction must be in (0.0, 1.0], got {fraction}"
            )));
        }
        Ok(())
    }

    /// Refill window in whole milliseconds, floored at 1.
    ///
    /// The bucket's arithmetic is millisecond-resolution, so a sub-millisecond
    /// interval rounds up to the smallest window the clock can express instead
    /// of dividing by zero.
    fn interval_millis(&self) -> u64 {
        u64::try_from(self.interval.as_millis())
            .unwrap_or(u64::MAX)
            .max(1)
    }
}

fn config_error(message: &str) -> RateLimitError {
    RateLimitError::Config(message.to_string())
}

/// Monotonic millisecond source for [`RateLimiter`].
///
/// Crate-private test seam in the mold of the prompt cache's `CacheClock`, but
/// **monotonic**: the production impl reads `Instant::elapsed` from the
/// limiter's construction instant, not the wall clock, so a system-time jump
/// can never mint a burst.
trait RateClock: Send + Sync {
    fn elapsed_millis(&self) -> u64;
}

/// Production clock: milliseconds elapsed since the limiter was built.
struct MonotonicClock {
    origin: Instant,
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl RateClock for MonotonicClock {
    fn elapsed_millis(&self) -> u64 {
        // Saturating rather than wrapping: u64 milliseconds covers ~584
        // million years of process uptime, and a saturated clock stalls the
        // refill instead of jumping backwards.
        u64::try_from(self.origin.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// Two integers; fixed size, never grows.
struct BucketState {
    /// Tokens in thousandths, capped at `burst * 1_000`.
    tokens_milli: u64,
    /// Monotonic ms at which `tokens_milli` was last brought current.
    last_refill_millis: u64,
}

impl BucketState {
    /// Bring the bucket current with `now_millis`.
    ///
    /// Integer-only and remainder-preserving: the clock mark advances by only
    /// the time the granted tokens consumed, so a sub-token remainder carries
    /// into the next refill and a slow trickle cannot starve. When the gain
    /// would overflow the cap the mark jumps to `now_millis` instead: credit
    /// past `burst` is discarded, never banked as elapsed time that a later
    /// refill could redeem.
    fn refill(&mut self, config: &RateLimitConfig, now_millis: u64) {
        let capacity_milli = u64::from(config.burst) * MILLI_TOKEN;
        // `saturating_sub` keeps a non-advancing clock at zero gain.
        let elapsed = now_millis.saturating_sub(self.last_refill_millis);
        let interval_millis = u128::from(config.interval_millis());
        let per_interval_milli = u128::from(config.max_requests) * u128::from(MILLI_TOKEN);
        // u128 intermediate: u32 tokens x 1_000 x u64 milliseconds cannot
        // overflow it.
        let gain_milli = u128::from(elapsed) * per_interval_milli / interval_millis;
        let headroom_milli = u128::from(capacity_milli.saturating_sub(self.tokens_milli));

        if gain_milli >= headroom_milli {
            self.tokens_milli = capacity_milli;
            self.last_refill_millis = now_millis;
            return;
        }
        // Below the cap, so the gain fits the u64 headroom.
        let gain_milli = gain_milli as u64;
        self.tokens_milli += gain_milli;
        let consumed_millis = u128::from(gain_milli) * interval_millis / per_interval_milli;
        self.last_refill_millis = self
            .last_refill_millis
            .saturating_add(consumed_millis as u64);
    }

    /// Time until the bucket holds one whole token, assuming it is current.
    fn time_to_next_token(&self, config: &RateLimitConfig) -> Duration {
        let missing_milli = MILLI_TOKEN.saturating_sub(self.tokens_milli);
        if missing_milli == 0 {
            return Duration::ZERO;
        }
        let interval_millis = u128::from(config.interval_millis());
        let per_interval_milli = u128::from(config.max_requests) * u128::from(MILLI_TOKEN);
        // Ceiling division: a wait that rounds down to zero would spin.
        let millis = (u128::from(missing_milli) * interval_millis).div_ceil(per_interval_milli);
        // Bounded by one interval, which is a `u64` count of milliseconds.
        Duration::from_millis(millis as u64)
    }
}

/// An admitted request: one concurrency slot, held until drop.
///
/// The token is spent, not returned; only the slot comes back. There is no
/// manual `Drop` impl: dropping the wrapped [`OwnedSemaphorePermit`] releases
/// the slot, which is what makes release correct on every caller exit path,
/// including a panic or an early `?`.
#[derive(Debug)]
pub struct RateLimitPermit(
    #[expect(
        dead_code,
        reason = "held only for its Drop, which returns the concurrency slot"
    )]
    OwnedSemaphorePermit,
);

/// Decrements the admission counter on every exit path of
/// [`RateLimiter::acquire`], including an early `?` and a panic.
struct WaiterGuard<'a>(&'a AtomicUsize);

impl Drop for WaiterGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Client-side outbound throttle: an integer token bucket plus a concurrency
/// semaphore.
///
/// `Send + Sync`; share one instance behind an `Arc` across every task that
/// dispatches to the throttled provider. Not a
/// [`CucaPlugin`](crate::plugin::CucaPlugin); see the module docs.
pub struct RateLimiter {
    config: RateLimitConfig,
    clock: Arc<dyn RateClock>,
    bucket: Mutex<BucketState>,
    slots: Arc<Semaphore>,
    /// Callers admitted into `acquire` and not yet finished. Bumped before the
    /// semaphore is ever awaited, so it, not the semaphore's internal list, is
    /// what bounds parked callers.
    waiting: AtomicUsize,
}

impl RateLimiter {
    /// Build a limiter with a full bucket (`burst` tokens) and every
    /// concurrency slot free.
    ///
    /// # Errors
    ///
    /// The [`RateLimitConfig::validate`] errors for `config`.
    pub fn new(config: RateLimitConfig) -> Result<Self, RateLimitError> {
        Self::with_clock(config, Arc::new(MonotonicClock::default()))
    }

    /// Build a limiter with an injected clock (crate-private test seam;
    /// visible to `mod tests` below as a descendant module).
    fn with_clock(
        config: RateLimitConfig,
        clock: Arc<dyn RateClock>,
    ) -> Result<Self, RateLimitError> {
        config.validate()?;
        let bucket = BucketState {
            tokens_milli: u64::from(config.burst) * MILLI_TOKEN,
            last_refill_millis: clock.elapsed_millis(),
        };
        Ok(Self {
            slots: Arc::new(Semaphore::new(config.max_concurrent)),
            bucket: Mutex::new(bucket),
            clock,
            waiting: AtomicUsize::new(0),
            config,
        })
    }

    /// Wait for a concurrency slot and then for a token, up to
    /// [`RateLimitConfig::max_wait`].
    ///
    /// Slot first, token second, on purpose: a caller that gives up waiting
    /// for a slot has spent no quota, and total parked callers are already
    /// bounded by `max_concurrent + max_waiters`.
    ///
    /// # Errors
    ///
    /// [`RateLimitError::QueueFull`] when the waiter queue is at
    /// `max_waiters`; [`RateLimitError::Exhausted`] when the wait budget
    /// elapses (carrying the remaining `retry_after`);
    /// [`RateLimitError::Observer`] when an observer refuses;
    /// [`RateLimitError::Lock`] when the bucket mutex is poisoned.
    pub async fn acquire(&self) -> Result<RateLimitPermit, RateLimitError> {
        let deadline = tokio::time::Instant::now() + self.config.max_wait;

        // Admission before parking: the counter is the queue bound, so it is
        // bumped before the semaphore is ever awaited.
        let admitted = self.waiting.fetch_add(1, Ordering::AcqRel) + 1;
        let _waiter = WaiterGuard(&self.waiting);
        if admitted > self.config.max_waiters {
            return Err(RateLimitError::QueueFull {
                waiters: admitted - 1,
                max_waiters: self.config.max_waiters,
            });
        }

        // The semaphore is private and never closed, so the only reachable
        // failure is the elapsed wait budget; both mean no slot was granted.
        let Ok(Ok(permit)) =
            tokio::time::timeout_at(deadline, Arc::clone(&self.slots).acquire_owned()).await
        else {
            return Err(RateLimitError::Exhausted {
                retry_after: self.peek_retry_after()?,
            });
        };

        // Near-cap warning, outside the bucket lock and before any token is
        // spent: a refusal costs the caller nothing but the slot, which the
        // in-progress permit releases as it drops on the way out.
        if !self.config.observers.is_empty() && self.crosses_warn_fraction(admitted) {
            let usage = self.usage()?;
            for observer in &self.config.observers {
                observer.observe(&usage).map_err(RateLimitError::Observer)?;
            }
        }

        loop {
            match self.take_token()? {
                None => return Ok(RateLimitPermit(permit)),
                Some(retry_after) => {
                    let now = tokio::time::Instant::now();
                    if now >= deadline {
                        return Err(RateLimitError::Exhausted { retry_after });
                    }
                    tokio::time::sleep_until(deadline.min(now + retry_after)).await;
                }
            }
        }
    }

    /// Non-blocking admission: succeed only if a slot and a whole token are
    /// both free right now. Runtime-free: no `tokio::time` involvement.
    ///
    /// # Errors
    ///
    /// [`RateLimitError::Busy`] when no slot is free;
    /// [`RateLimitError::Exhausted`] when the bucket is empty (carrying the
    /// time until the next whole token); [`RateLimitError::Lock`] on a
    /// poisoned mutex.
    pub fn try_acquire(&self) -> Result<RateLimitPermit, RateLimitError> {
        let permit =
            Arc::clone(&self.slots)
                .try_acquire_owned()
                .map_err(|_| RateLimitError::Busy {
                    in_flight: self.in_flight(),
                    max_concurrent: self.config.max_concurrent,
                })?;
        // The permit drops on either error arm below, so a refused acquire
        // never strands a slot.
        match self.take_token()? {
            None => Ok(RateLimitPermit(permit)),
            Some(retry_after) => Err(RateLimitError::Exhausted { retry_after }),
        }
    }

    /// Cheap combined gauge; refills the bucket to "now" before reading it.
    ///
    /// # Errors
    ///
    /// [`RateLimitError::Lock`] on a poisoned mutex.
    pub fn usage(&self) -> Result<RateLimitUsage, RateLimitError> {
        let available_tokens =
            self.with_bucket(|bucket| (bucket.tokens_milli / MILLI_TOKEN) as u32)?;
        Ok(RateLimitUsage {
            available_tokens,
            in_flight: self.in_flight(),
            waiting: self.waiting.load(Ordering::Acquire),
        })
    }

    /// The validated bounds this limiter was built with.
    #[must_use]
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }

    /// Permits currently held, in O(1).
    fn in_flight(&self) -> usize {
        self.config
            .max_concurrent
            .saturating_sub(self.slots.available_permits())
    }

    /// Whether this admission pushed the waiter queue to or past the
    /// configured warn fraction.
    fn crosses_warn_fraction(&self, admitted: usize) -> bool {
        self.config
            .warn_fraction
            .is_some_and(|fraction| admitted as f32 / self.config.max_waiters as f32 >= fraction)
    }

    /// Refill to "now" and hand the current bucket to `read`, releasing the
    /// lock before returning.
    ///
    /// Every bucket access goes through here, which is what keeps the guard
    /// off every `.await` in [`Self::acquire`].
    fn with_bucket<T>(
        &self,
        read: impl FnOnce(&mut BucketState) -> T,
    ) -> Result<T, RateLimitError> {
        let now = self.clock.elapsed_millis();
        let mut bucket = self
            .bucket
            .lock()
            // Fail closed: a poisoned bucket may hold half-updated token
            // state, and recovering it could admit a burst.
            .map_err(|e| RateLimitError::Lock(e.to_string()))?;
        bucket.refill(&self.config, now);
        Ok(read(&mut bucket))
    }

    /// Spend one whole token (`None`), or report how long until the bucket
    /// holds one (`Some`).
    fn take_token(&self) -> Result<Option<Duration>, RateLimitError> {
        self.with_bucket(|bucket| {
            if bucket.tokens_milli >= MILLI_TOKEN {
                bucket.tokens_milli -= MILLI_TOKEN;
                None
            } else {
                Some(bucket.time_to_next_token(&self.config))
            }
        })
    }

    /// How long until the bucket holds one whole token, spending nothing.
    fn peek_retry_after(&self) -> Result<Duration, RateLimitError> {
        self.with_bucket(|bucket| bucket.time_to_next_token(&self.config))
    }
}

#[cfg(all(test, feature = "plugin-rate-limit"))]
mod tests {
    use std::sync::atomic::AtomicU64;

    use super::*;

    /// Deterministic clock for the bucket arithmetic tests.
    #[derive(Default)]
    struct TestClock {
        millis: AtomicU64,
    }

    impl TestClock {
        fn advance(&self, millis: u64) {
            self.millis.fetch_add(millis, Ordering::SeqCst);
        }
    }

    impl RateClock for TestClock {
        fn elapsed_millis(&self) -> u64 {
            self.millis.load(Ordering::SeqCst)
        }
    }

    /// `max_requests` per `interval_ms`, one slot short of unlimited waiters.
    fn config(max_requests: u32, interval_ms: u64, max_concurrent: usize) -> RateLimitConfig {
        RateLimitConfig::new(
            max_requests,
            Duration::from_millis(interval_ms),
            max_concurrent,
            8,
        )
        .expect("test config must validate")
    }

    fn limiter(config: RateLimitConfig) -> RateLimiter {
        RateLimiter::new(config).expect("test limiter must build")
    }

    /// A limiter driven by an injected clock that starts at zero.
    fn clocked(config: RateLimitConfig) -> (RateLimiter, Arc<TestClock>) {
        let clock = Arc::new(TestClock::default());
        let limiter = RateLimiter::with_clock(config, Arc::clone(&clock) as Arc<dyn RateClock>)
            .expect("test limiter must build");
        (limiter, clock)
    }

    fn config_message(error: RateLimitError) -> String {
        match error {
            RateLimitError::Config(message) => message,
            other => panic!("expected a Config error, got {other}"),
        }
    }

    /// Drain every token the bucket holds, dropping each permit so the
    /// concurrency slot comes back but the token does not.
    fn drain_bucket(limiter: &RateLimiter) {
        while limiter.try_acquire().is_ok() {}
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            0
        );
    }

    #[test]
    fn config_new_rejects_zero_max_requests() {
        let error = RateLimitConfig::new(0, Duration::from_secs(1), 1, 1)
            .expect_err("a zero rate must be rejected");
        assert!(config_message(error).contains("max_requests"));
    }

    #[test]
    fn config_new_rejects_zero_interval() {
        let error =
            RateLimitConfig::new(1, Duration::ZERO, 1, 1).expect_err("a zero interval is rejected");
        assert!(config_message(error).contains("interval"));
    }

    #[test]
    fn config_new_rejects_zero_max_concurrent() {
        let error = RateLimitConfig::new(1, Duration::from_secs(1), 0, 1)
            .expect_err("a zero concurrency cap is rejected");
        assert!(config_message(error).contains("max_concurrent"));
    }

    #[test]
    fn config_new_rejects_zero_max_waiters() {
        let error = RateLimitConfig::new(1, Duration::from_secs(1), 1, 0)
            .expect_err("a zero queue bound is rejected");
        assert!(config_message(error).contains("max_waiters"));
    }

    #[test]
    fn config_new_defaults_burst_to_max_requests_and_max_wait_to_interval() {
        let config = RateLimitConfig::new(7, Duration::from_millis(250), 3, 9)
            .expect("a valid config must build");
        assert_eq!(config.burst, 7);
        assert_eq!(config.max_wait, Duration::from_millis(250));
        assert_eq!(config.warn_fraction, None);
        assert!(config.observers.is_empty());
    }

    #[test]
    fn config_with_burst_rejects_zero() {
        let error = config(4, 1_000, 1)
            .with_burst(0)
            .expect_err("a zero burst is rejected");
        assert!(config_message(error).contains("burst"));
        assert_eq!(
            config(4, 1_000, 1)
                .with_burst(9)
                .expect("a non-zero burst is accepted")
                .burst,
            9
        );
    }

    #[test]
    fn config_with_max_wait_rejects_zero() {
        let error = config(4, 1_000, 1)
            .with_max_wait(Duration::ZERO)
            .expect_err("a zero wait budget is rejected");
        assert!(config_message(error).contains("max_wait"));
    }

    #[test]
    fn config_with_warn_fraction_rejects_zero_and_above_one() {
        for fraction in [0.0f32, -0.5, 1.5, f32::NAN, f32::INFINITY] {
            let error = config(4, 1_000, 1)
                .with_warn_fraction(fraction)
                .expect_err("an out-of-range warn fraction is rejected");
            assert!(
                config_message(error).contains("warn_fraction"),
                "{fraction} must be refused by the warn_fraction guard"
            );
        }
        // The interval is closed at the top.
        assert_eq!(
            config(4, 1_000, 1)
                .with_warn_fraction(1.0)
                .expect("1.0 is in range")
                .warn_fraction,
            Some(1.0)
        );
    }

    #[test]
    fn validate_rejects_a_struct_literal_config_that_bypassed_new() {
        let bypassed = RateLimitConfig {
            max_requests: 1,
            interval: Duration::from_secs(1),
            burst: 0,
            max_concurrent: 1,
            max_waiters: 1,
            max_wait: Duration::from_secs(1),
            warn_fraction: None,
            observers: Vec::new(),
        };
        assert!(config_message(bypassed.validate().expect_err("burst is zero")).contains("burst"));
        assert!(matches!(
            RateLimiter::new(bypassed),
            Err(RateLimitError::Config(_))
        ));
    }

    #[test]
    fn limiter_new_starts_with_a_full_bucket_of_burst_tokens() {
        let limiter = limiter(config(3, 60_000, 2).with_burst(7).expect("burst is valid"));
        assert_eq!(
            limiter.usage().expect("usage must read"),
            RateLimitUsage {
                available_tokens: 7,
                in_flight: 0,
                waiting: 0,
            }
        );
    }

    #[test]
    fn try_acquire_spends_exactly_one_token() {
        let limiter = limiter(config(3, 60_000, 2));
        let _permit = limiter.try_acquire().expect("a full bucket admits");
        assert_eq!(
            limiter.usage().expect("usage must read"),
            RateLimitUsage {
                available_tokens: 2,
                in_flight: 1,
                waiting: 0,
            }
        );
    }

    #[test]
    fn try_acquire_on_an_empty_bucket_reports_exhausted_with_time_to_next_token() {
        // 2 tokens per second, so one whole token takes 500 ms.
        let (limiter, _clock) = clocked(config(2, 1_000, 4).with_burst(1).expect("burst is valid"));
        let _permit = limiter.try_acquire().expect("the one token admits");
        match limiter.try_acquire() {
            Err(RateLimitError::Exhausted { retry_after }) => {
                assert_eq!(retry_after, Duration::from_millis(500));
            }
            other => panic!("an empty bucket must report Exhausted, got {other:?}"),
        }
    }

    #[test]
    fn try_acquire_with_every_slot_held_reports_busy() {
        let limiter = limiter(config(5, 60_000, 1));
        let _held = limiter.try_acquire().expect("the only slot is free");
        match limiter.try_acquire() {
            Err(RateLimitError::Busy {
                in_flight,
                max_concurrent,
            }) => {
                assert_eq!((in_flight, max_concurrent), (1, 1));
            }
            other => panic!("a held slot must report Busy, got {other:?}"),
        }
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            4,
            "a slot refusal must not spend a token"
        );
    }

    #[test]
    fn refill_grants_exactly_max_requests_per_interval() {
        // 3 tokens per second into a bucket that holds 6.
        let (limiter, clock) = clocked(config(3, 1_000, 1).with_burst(6).expect("burst is valid"));
        drain_bucket(&limiter);

        clock.advance(1_000);
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            3
        );
        clock.advance(1_000);
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            6
        );
    }

    #[test]
    fn refill_clamps_at_burst_after_a_long_idle() {
        let (limiter, clock) = clocked(config(3, 1_000, 1).with_burst(6).expect("burst is valid"));
        drain_bucket(&limiter);

        clock.advance(1_000_000);
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            6,
            "a long idle banks at most `burst`"
        );

        // Idle time past the cap is discarded, not banked: spending the burst
        // must leave the bucket empty until the clock advances again.
        drain_bucket(&limiter);
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            0
        );
    }

    #[test]
    fn refill_carries_the_sub_token_remainder_across_two_half_interval_advances() {
        // 1 token per 100 ms: each advance below is worth exactly half a token.
        let (limiter, clock) = clocked(config(1, 100, 1));
        drain_bucket(&limiter);

        clock.advance(50);
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            0,
            "half a token is not a whole token"
        );
        clock.advance(50);
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            1,
            "the two halves must sum instead of being rounded away"
        );
    }

    #[test]
    fn refill_grants_nothing_when_the_clock_does_not_advance() {
        let (limiter, _clock) = clocked(config(4, 1_000, 4));
        let _permit = limiter.try_acquire().expect("a full bucket admits");
        for _ in 0..3 {
            assert_eq!(
                limiter.usage().expect("usage must read").available_tokens,
                3,
                "a frozen clock must not mint tokens"
            );
        }
    }

    #[test]
    fn dropping_a_permit_returns_its_concurrency_slot() {
        let limiter = limiter(config(5, 60_000, 1));
        let permit = limiter.try_acquire().expect("the only slot is free");
        assert_eq!(limiter.usage().expect("usage must read").in_flight, 1);
        drop(permit);
        assert_eq!(limiter.usage().expect("usage must read").in_flight, 0);
        limiter
            .try_acquire()
            .expect("the released slot must be reusable");
    }

    #[test]
    fn dropping_a_permit_does_not_return_its_token() {
        let (limiter, _clock) = clocked(config(2, 60_000, 1));
        let permit = limiter.try_acquire().expect("a full bucket admits");
        drop(permit);
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            1,
            "a spent token is gone; only the slot comes back"
        );
    }

    #[tokio::test]
    async fn acquire_waits_for_a_refill_instead_of_rejecting() {
        // One token per 50 ms, a wait budget far past it.
        let limiter = limiter(
            config(1, 50, 4)
                .with_max_wait(Duration::from_secs(5))
                .expect("wait budget is valid"),
        );
        drop(limiter.try_acquire().expect("the one token admits"));

        let started = Instant::now();
        let _permit = limiter
            .acquire()
            .await
            .expect("the refill must arrive inside the budget");
        assert!(
            started.elapsed() >= Duration::from_millis(45),
            "acquire must have waited for the refill, waited {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn acquire_past_the_wait_budget_reports_exhausted_with_retry_after() {
        // One token per second, but only 50 ms of budget to wait for it.
        let limiter = limiter(
            config(1, 1_000, 4)
                .with_max_wait(Duration::from_millis(50))
                .expect("wait budget is valid"),
        );
        drop(limiter.try_acquire().expect("the one token admits"));

        let started = Instant::now();
        match limiter.acquire().await {
            Err(RateLimitError::Exhausted { retry_after }) => {
                assert!(
                    retry_after > Duration::ZERO,
                    "the refusal must carry the remaining wait"
                );
            }
            other => panic!("an elapsed budget must report Exhausted, got {other:?}"),
        }
        assert!(started.elapsed() >= Duration::from_millis(45));
        assert_eq!(
            limiter.usage().expect("usage must read").in_flight,
            0,
            "a refused acquire must not strand its slot"
        );
    }

    #[tokio::test]
    async fn acquire_at_the_waiter_cap_reports_queue_full_without_parking() {
        let limiter = Arc::new(limiter(
            RateLimitConfig::new(8, Duration::from_secs(60), 1, 1)
                .expect("config must validate")
                .with_max_wait(Duration::from_secs(10))
                .expect("wait budget is valid"),
        ));
        let held = limiter.try_acquire().expect("the only slot is free");

        let parked = tokio::spawn({
            let limiter = Arc::clone(&limiter);
            async move { limiter.acquire().await.map(|_| ()) }
        });
        for _ in 0..16 {
            if limiter.usage().expect("usage must read").waiting == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(limiter.usage().expect("usage must read").waiting, 1);

        let started = Instant::now();
        match limiter.acquire().await {
            Err(RateLimitError::QueueFull {
                waiters,
                max_waiters,
            }) => assert_eq!((waiters, max_waiters), (1, 1)),
            other => panic!("a full queue must report QueueFull, got {other:?}"),
        }
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "the refusal must be immediate, took {:?}",
            started.elapsed()
        );

        drop(held);
        parked
            .await
            .expect("the parked task must not panic")
            .expect("the parked waiter must be admitted once the slot frees");
    }

    #[tokio::test]
    async fn acquire_that_times_out_waiting_for_a_slot_leaves_available_tokens_unchanged() {
        let limiter = limiter(
            RateLimitConfig::new(5, Duration::from_secs(60), 1, 8)
                .expect("config must validate")
                .with_max_wait(Duration::from_millis(50))
                .expect("wait budget is valid"),
        );
        let _held = limiter.try_acquire().expect("the only slot is free");
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            4
        );

        assert!(matches!(
            limiter.acquire().await,
            Err(RateLimitError::Exhausted { .. })
        ));
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            4,
            "a caller that gave up waiting for a slot must not have spent a token"
        );
        assert_eq!(limiter.usage().expect("usage must read").waiting, 0);
    }

    #[tokio::test]
    async fn usage_reports_available_tokens_in_flight_and_waiting() {
        let limiter = Arc::new(limiter(
            RateLimitConfig::new(4, Duration::from_secs(60), 1, 4)
                .expect("config must validate")
                .with_max_wait(Duration::from_secs(10))
                .expect("wait budget is valid"),
        ));
        let held = limiter.try_acquire().expect("the only slot is free");

        let parked = tokio::spawn({
            let limiter = Arc::clone(&limiter);
            async move { limiter.acquire().await.map(|_| ()) }
        });
        for _ in 0..16 {
            if limiter.usage().expect("usage must read").waiting == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }

        assert_eq!(
            limiter.usage().expect("usage must read"),
            RateLimitUsage {
                available_tokens: 3,
                in_flight: 1,
                waiting: 1,
            }
        );

        drop(held);
        parked
            .await
            .expect("the parked task must not panic")
            .expect("the parked waiter must be admitted");
        assert_eq!(
            limiter.usage().expect("usage must read"),
            RateLimitUsage {
                available_tokens: 2,
                in_flight: 0,
                waiting: 0,
            }
        );
    }

    #[derive(Default)]
    struct RecordingObserver {
        seen: Mutex<Vec<RateLimitUsage>>,
    }

    impl RateLimitObserver for RecordingObserver {
        fn observe(&self, usage: &RateLimitUsage) -> Result<(), PluginError> {
            self.seen
                .lock()
                .expect("observer log must not be poisoned")
                .push(*usage);
            Ok(())
        }
    }

    struct RefusingObserver;

    impl RateLimitObserver for RefusingObserver {
        fn observe(&self, _usage: &RateLimitUsage) -> Result<(), PluginError> {
            Err(PluginError::Internal("observer says no".to_string()))
        }
    }

    #[tokio::test]
    async fn crossing_warn_fraction_hands_a_usage_reading_to_every_observer() {
        let first = Arc::new(RecordingObserver::default());
        let second = Arc::new(RecordingObserver::default());
        // One admitted waiter out of two is exactly the configured fraction.
        let limiter = limiter(
            RateLimitConfig::new(5, Duration::from_secs(60), 2, 2)
                .expect("config must validate")
                .with_warn_fraction(0.5)
                .expect("fraction is in range")
                .with_observer(Arc::clone(&first) as Arc<dyn RateLimitObserver>)
                .with_observer(Arc::clone(&second) as Arc<dyn RateLimitObserver>),
        );

        let _permit = limiter.acquire().await.expect("a full bucket admits");

        for observer in [&first, &second] {
            let seen = observer
                .seen
                .lock()
                .expect("observer log must read")
                .clone();
            assert_eq!(
                seen.len(),
                1,
                "every observer gets the reading exactly once"
            );
            assert_eq!(
                seen[0],
                RateLimitUsage {
                    // The slot is taken before the warning; the token is not.
                    available_tokens: 5,
                    in_flight: 1,
                    waiting: 1,
                }
            );
        }
    }

    #[tokio::test]
    async fn observer_error_aborts_the_acquire_and_releases_the_slot() {
        let limiter = limiter(
            RateLimitConfig::new(5, Duration::from_secs(60), 1, 1)
                .expect("config must validate")
                .with_warn_fraction(1.0)
                .expect("fraction is in range")
                .with_observer(Arc::new(RefusingObserver) as Arc<dyn RateLimitObserver>),
        );

        match limiter.acquire().await {
            Err(RateLimitError::Observer(PluginError::Internal(message))) => {
                assert_eq!(message, "observer says no");
            }
            other => panic!("a refusing observer must abort the acquire, got {other:?}"),
        }
        assert_eq!(
            limiter.usage().expect("usage must read"),
            RateLimitUsage {
                available_tokens: 5,
                in_flight: 0,
                waiting: 0,
            },
            "the refused acquire must release its slot and spend no token"
        );
    }

    #[test]
    fn poisoned_bucket_mutex_reports_lock_instead_of_admitting() {
        let limiter = Arc::new(limiter(config(5, 60_000, 2)));
        let poisoner = Arc::clone(&limiter);
        let panicked = std::thread::spawn(move || {
            let _guard = poisoner
                .bucket
                .lock()
                .expect("a fresh mutex is not poisoned");
            panic!("poison the bucket");
        })
        .join();
        assert!(panicked.is_err(), "the helper thread must have panicked");

        assert!(matches!(
            limiter.try_acquire(),
            Err(RateLimitError::Lock(_))
        ));
        assert!(matches!(limiter.usage(), Err(RateLimitError::Lock(_))));
        assert_eq!(
            limiter.slots.available_permits(),
            2,
            "a fail-closed refusal must not strand a slot"
        );
    }

    #[test]
    fn config_error_maps_to_plugin_error_validation() {
        let error = RateLimitConfig::new(0, Duration::from_secs(1), 1, 1)
            .expect_err("a zero rate must be rejected");
        match PluginError::from(error) {
            PluginError::Validation { schema, message } => {
                assert_eq!(schema, "rate-limit-config");
                assert!(message.contains("max_requests"));
            }
            other => panic!("a Config error must map to Validation, got {other:?}"),
        }
    }

    #[test]
    fn observer_error_round_trips_through_the_plugin_error_conversion() {
        let inner = PluginError::hook("rate-limit", "acquire", "refused");
        let converted = PluginError::from(RateLimitError::Observer(inner.clone()));
        assert_eq!(converted.to_string(), inner.to_string());
        assert!(matches!(converted, PluginError::HookFailure { .. }));
    }

    #[test]
    fn rate_limiter_and_permit_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RateLimiter>();
        assert_send_sync::<RateLimitPermit>();
        assert_send_sync::<RateLimitUsage>();
        assert_send_sync::<RateLimitError>();
    }
}
