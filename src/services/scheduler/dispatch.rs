//! The dispatcher: placement over several backends, failover, rebalance, the
//! circuit breaker and lazy client builds.
//!
//! A [`Dispatcher`] holds one backend list. Each backend owns a
//! [`Scheduler`], a circuit breaker and a [`CucaClient`] (built up front, or
//! on the first call by a [`ClientFactory`]). [`Dispatcher::call`] places a
//! call on the least-loaded backend whose breaker admits it, streams the
//! reply, and tries every other admitting backend once when a backend cannot
//! be reached.

use std::future::poll_fn;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::{Duration, Instant};

use tracing::Instrument;

use super::queue::{
    BackendId, CallMeta, CallOutcome, CallReport, CallSlot, DispatchGate, Load, Scheduler,
    SchedulerConfig, SchedulerError, SchedulerPool, SystemClock, lowers_the_worst, move_waiting,
};
use crate::client::CucaClient;
use crate::error::CucaError;
use crate::request::{ThinkingConfig, TokenUsage, UnifiedRequest};
use crate::types::MessageContentBlock;

/// The default time a backend that could not be reached is skipped.
const UNREACHABLE_COOLDOWN: Duration = Duration::from_secs(20);

/// The error of a [`Dispatcher`] operation.
#[derive(Debug, Clone)]
pub enum DispatchError {
    /// Invalid configuration: an invalid scheduler configuration, a zero
    /// cooldown, a backend listed twice in one list, or more than 64
    /// backends in one registry.
    Config(String),
    /// The backend the call was placed on already holds
    /// [`SchedulerConfig::max_waiting`] waiting calls; the call was refused
    /// and nothing is queued.
    QueueFull {
        /// The backend whose queue is full.
        backend: BackendId,
    },
    /// No backend could take the call: each one that admitted it could not be
    /// reached, or every breaker is open, or the list is empty.
    Unreachable {
        /// Why: the transport error of the last refusal, or which breakers
        /// are open and for how long.
        detail: String,
    },
    /// The call ran past [`DispatcherConfig::timeout`]. The deadline starts
    /// when the call leaves the queue, so it measures the model.
    Timeout {
        /// The deadline, in whole seconds.
        seconds: u64,
    },
    /// The stream ended without any text and without any reasoning.
    Empty,
    /// The stream held reasoning but no text, as when the token limit is
    /// reached mid-reasoning.
    ReasoningOnly {
        /// Characters of reasoning received.
        reasoning_chars: usize,
    },
    /// The [`DispatchGate`] reported the call's task stale while it waited.
    InputsChanged,
    /// The backend answered with an error: an HTTP status, a provider
    /// error, or a failure mid-stream. The call is not retried elsewhere.
    Call(CucaError),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(message) => write!(f, "dispatcher configuration error: {message}"),
            Self::QueueFull { backend } => {
                write!(f, "the queue of backend {} is full", backend.as_str())
            }
            Self::Unreachable { detail } => write!(f, "no backend is reachable: {detail}"),
            Self::Timeout { seconds } => write!(f, "the call timed out after {seconds}s"),
            Self::Empty => write!(f, "the backend returned no text"),
            Self::ReasoningOnly { reasoning_chars } => write!(
                f,
                "the backend returned only reasoning ({reasoning_chars} characters) and no text"
            ),
            Self::InputsChanged => write!(f, "the call's inputs changed while it waited"),
            Self::Call(error) => write!(f, "the backend call failed: {error}"),
        }
    }
}

impl std::error::Error for DispatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Call(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SchedulerError> for DispatchError {
    fn from(error: SchedulerError) -> Self {
        match error {
            SchedulerError::Config(message) => Self::Config(message),
            SchedulerError::QueueFull { backend, .. } => Self::QueueFull { backend },
            SchedulerError::InputsChanged => Self::InputsChanged,
        }
    }
}

/// Builds a backend's [`CucaClient`] on demand, for a client that is slow or
/// costly to make, such as one that loads a model in process.
pub trait ClientFactory: Send + Sync + std::fmt::Debug {
    /// Builds the backend's client. The dispatcher calls it once per
    /// backend, before the first slot is reserved and never under the call
    /// deadline; an `Err` opens the backend's breaker and the next call
    /// tries again after the cooldown.
    fn build(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Arc<CucaClient>, CucaError>> + Send + '_>>;
}

/// Where a backend's [`CucaClient`] comes from.
pub enum BackendClient {
    /// A client that already exists.
    Ready(Arc<CucaClient>),
    /// A client the factory builds before the backend's first call.
    Lazy(Arc<dyn ClientFactory>),
}

impl std::fmt::Debug for BackendClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready(_) => f.write_str("Ready"),
            Self::Lazy(factory) => f.debug_tuple("Lazy").field(factory).finish(),
        }
    }
}

/// How to reach one backend.
#[derive(Debug)]
pub struct BackendSpec {
    /// The backend's identity. A spec whose id the registry already holds
    /// joins that backend and is otherwise dropped.
    pub id: BackendId,
    /// Where the client comes from.
    pub client: BackendClient,
    /// `Some(tag)` sends the tag as the request's model and tags the reply
    /// with it. `None` leaves the request's model as it is, and the reply
    /// carries that model.
    pub model: Option<Arc<str>>,
    /// `Some` overwrites the request's thinking configuration before it is
    /// sent. `None` leaves the request's own.
    pub thinking: Option<ThinkingConfig>,
}

/// One backend: its client, its [`Scheduler`] and its circuit breaker.
///
/// A registry builds each backend once per [`BackendId`], so every dispatcher
/// that lists it shares one queue, tuner and breaker.
pub struct Backend {
    client: BackendClient,
    built: tokio::sync::OnceCell<Arc<CucaClient>>,
    model: Option<Arc<str>>,
    thinking: Option<ThinkingConfig>,
    scheduler: Arc<Scheduler>,
    breaker: Breaker,
    /// The bit of [`CallMeta::allowed`] naming this backend's scheduler in its
    /// pool ([`SchedulerPool::insert`]).
    bit: u64,
}

impl Backend {
    /// The backend's identity.
    #[must_use]
    pub fn id(&self) -> &BackendId {
        self.scheduler.backend()
    }

    /// The scheduler its calls wait in.
    #[must_use]
    pub const fn scheduler(&self) -> &Arc<Scheduler> {
        &self.scheduler
    }

    /// Its client, building it first when it is lazy and not built yet.
    async fn client(&self) -> Result<&Arc<CucaClient>, CucaError> {
        match &self.client {
            BackendClient::Ready(client) => Ok(client),
            BackendClient::Lazy(factory) => self.built.get_or_try_init(|| factory.build()).await,
        }
    }

    /// Whether it has a client to send with without building one first.
    fn has_client(&self) -> bool {
        matches!(self.client, BackendClient::Ready(_)) || self.built.initialized()
    }

    /// Opens its breaker for `cooldown` from `now`.
    fn trip(&self, now: Instant, cooldown: Duration) {
        // CONTEXT: one WARN when the backend goes away and a DEBUG line per
        // re-trip after that: a failed re-attempt is routine.
        if self.breaker.trip(now, cooldown) {
            tracing::warn!(
                target: "cuca::scheduler",
                "backend {} unreachable; its calls go elsewhere for {}s at a time until it answers",
                self.id().as_str(),
                cooldown.as_secs()
            );
        } else {
            tracing::debug!(
                target: "cuca::scheduler",
                "backend {} still unreachable",
                self.id().as_str()
            );
        }
    }

    /// Closes its breaker: it answered.
    fn close(&self) {
        if self.breaker.close() {
            tracing::info!(
                target: "cuca::scheduler",
                "backend {} reachable again",
                self.id().as_str()
            );
        }
    }
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backend")
            .field("id", self.id())
            .field("client", &self.client)
            .field("breaker", &self.breaker)
            .finish_non_exhaustive()
    }
}

/// A one-deadline circuit breaker over "the backend could not be reached".
///
/// Closed (`None`, or a deadline already past) lets every call through.
/// Open (a deadline still ahead) refuses them. There is no half-open state:
/// once the deadline passes, the next call is a real attempt, and its outcome
/// either closes the breaker or moves the deadline. Time is always passed in,
/// so the state machine is testable with explicit instants.
#[derive(Debug, Default)]
struct Breaker {
    open_until: Mutex<Option<Instant>>,
}

impl Breaker {
    /// How long it stays open after `now`; `None` while it is closed.
    fn remaining(&self, now: Instant) -> Option<Duration> {
        let open_until = *self.lock();
        open_until
            .map(|until| until.saturating_duration_since(now))
            .filter(|remaining| !remaining.is_zero())
    }

    /// Lets a call through at `now`, or refuses it while the breaker is open.
    fn admit(&self, now: Instant) -> Result<(), DispatchError> {
        match self.remaining(now) {
            Some(remaining) => Err(DispatchError::Unreachable {
                detail: format!(
                    "a connect was refused; next attempt in {}s",
                    remaining.as_secs().max(1)
                ),
            }),
            None => Ok(()),
        }
    }

    /// Opens the breaker for `cooldown` from `now`; `true` when it had never
    /// been opened or was closed since.
    fn trip(&self, now: Instant, cooldown: Duration) -> bool {
        self.lock().replace(now + cooldown).is_none()
    }

    /// Closes the breaker; `true` when it was open (or its cooldown had
    /// passed without an answer).
    fn close(&self) -> bool {
        self.lock().take().is_some()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Instant>> {
        // CONTEXT: the guarded value is a plain deadline, never left
        // half-written, so a poisoned lock still holds a usable one.
        self.open_until
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// One dispatcher's backends, in the order its list names them, and the bits of
/// [`CallMeta::allowed`] that name them: what every call it makes is stamped
/// with, so a rebalance hands a call only to a backend the dispatcher lists.
#[derive(Debug, Default)]
struct Joined {
    backends: Vec<Arc<Backend>>,
    mask: u64,
}

/// Every backend of a process, each built once per [`BackendId`].
///
/// One scheduler per backend. A dispatcher created after another joins the
/// backends the first already listed and queues in the same schedulers; the
/// first dispatcher to list a backend sets its configuration. Every scheduler
/// shares the registry's [`DispatchGate`] and change channel, and every one
/// belongs to the registry's [`SchedulerPool`], whose placement lock
/// serializes placement across all dispatchers.
///
/// A backend is never removed from a registry: a list that stops naming one
/// leaves it to drain the calls already waiting there, and a list that names
/// it again gets the same scheduler, breaker and built client back. A pool
/// holds at most 64 backends.
#[derive(Debug)]
pub struct BackendRegistry {
    gate: Arc<dyn DispatchGate>,
    changes: Arc<tokio::sync::watch::Sender<u64>>,
    pool: Arc<SchedulerPool>,
    backends: Vec<Arc<Backend>>,
}

impl BackendRegistry {
    /// An empty registry whose schedulers share `gate` and report their
    /// changes to `changes`.
    #[must_use]
    pub fn new(gate: Arc<dyn DispatchGate>, changes: Arc<tokio::sync::watch::Sender<u64>>) -> Self {
        Self {
            gate,
            changes,
            pool: Arc::new(SchedulerPool::new()),
            backends: Vec::new(),
        }
    }

    /// Every backend's scheduler, once each, in the order dispatchers listed
    /// them.
    #[must_use]
    pub fn schedulers(&self) -> Vec<Arc<Scheduler>> {
        self.backends
            .iter()
            .map(|backend| Arc::clone(&backend.scheduler))
            .collect()
    }

    /// The backends `specs` lists, in its order, building each one no
    /// dispatcher has listed yet with `config`, and the bits of
    /// [`CallMeta::allowed`] that name them.
    fn join(
        &mut self,
        specs: Vec<BackendSpec>,
        config: &SchedulerConfig,
    ) -> Result<Joined, DispatchError> {
        for (index, spec) in specs.iter().enumerate() {
            if specs[..index].iter().any(|earlier| earlier.id == spec.id) {
                return Err(DispatchError::Config(format!(
                    "backend listed twice: {}",
                    spec.id.as_str()
                )));
            }
        }
        let mut backends = Vec::with_capacity(specs.len());
        let mut mask = 0u64;
        for spec in specs {
            if let Some(known) = self.backends.iter().find(|known| *known.id() == spec.id) {
                mask |= known.bit;
                backends.push(Arc::clone(known));
                continue;
            }
            let scheduler = Arc::new(Scheduler::new(
                spec.id,
                config.clone(),
                Arc::new(SystemClock),
                Arc::clone(&self.gate),
                Arc::clone(&self.changes),
                Arc::clone(&self.pool),
            ));
            let Some(index) = self.pool.insert(&scheduler) else {
                return Err(DispatchError::Config("more than 64 backends".to_owned()));
            };
            let backend = Arc::new(Backend {
                client: spec.client,
                built: tokio::sync::OnceCell::new(),
                model: spec.model,
                thinking: spec.thinking,
                scheduler,
                breaker: Breaker::default(),
                bit: 1u64 << index,
            });
            tracing::info!(
                target: "cuca::scheduler",
                "backend {} (at most {} calls at once)",
                backend.id().as_str(),
                backend.scheduler.ceiling()
            );
            mask |= backend.bit;
            self.backends.push(Arc::clone(&backend));
            backends.push(backend);
        }
        Ok(Joined { backends, mask })
    }
}

/// The configuration of a [`Dispatcher`].
#[derive(Debug, Clone)]
pub struct DispatcherConfig {
    /// What a backend this dispatcher is the first to list gets.
    pub scheduler: SchedulerConfig,
    /// The most a call may run once it has left the queue, or `None` to wait
    /// however long the backend takes. Default `None`.
    pub timeout: Option<Duration>,
    /// How long a backend that could not be reached is skipped. Default 20
    /// seconds.
    pub unreachable_cooldown: Duration,
}

impl DispatcherConfig {
    /// A configuration over `scheduler` with no timeout and a 20 second
    /// unreachable cooldown.
    #[must_use]
    pub const fn new(scheduler: SchedulerConfig) -> Self {
        Self {
            scheduler,
            timeout: None,
            unreachable_cooldown: UNREACHABLE_COOLDOWN,
        }
    }

    /// Sets the per-call deadline.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets how long an unreachable backend is skipped.
    ///
    /// # Errors
    ///
    /// [`DispatchError::Config`] when `cooldown` is zero.
    pub fn with_unreachable_cooldown(mut self, cooldown: Duration) -> Result<Self, DispatchError> {
        if cooldown.is_zero() {
            return Err(DispatchError::Config(
                "the unreachable cooldown must be above zero".to_owned(),
            ));
        }
        self.unreachable_cooldown = cooldown;
        Ok(self)
    }
}

/// What a finished call returns.
#[derive(Debug, Clone)]
pub struct Completion {
    /// The reply text.
    pub text: String,
    /// The backend's tag ([`BackendSpec::model`]) or, with none, the model the
    /// request named.
    pub model: Arc<str>,
    /// The backend that served the call.
    pub backend: BackendId,
    /// The backend's parallelism level when the call started.
    pub level: u32,
    /// Milliseconds from reserving a place to getting a slot.
    pub wait_ms: u64,
    /// Milliseconds from getting the slot to the end of the stream.
    pub run_ms: u64,
    /// Characters of text and reasoning streamed back.
    pub output_chars: u64,
    /// Characters of reasoning among them.
    pub reasoning_chars: usize,
    /// The backend's own token accounting; `None` when it reported none.
    pub usage: Option<TokenUsage>,
}

/// Places calls over a list of backends.
///
/// Created with [`Dispatcher::new`] from a [`BackendRegistry`]; the list is
/// swapped whole by [`Dispatcher::set_backends`]. A call reads the list once
/// and keeps that snapshot for every placement and failover.
#[derive(Debug)]
pub struct Dispatcher {
    joined: RwLock<Arc<Joined>>,
    config: DispatcherConfig,
    pool: Arc<SchedulerPool>,
    /// When [`Self::rebalance`] last ran, so it runs at its interval however
    /// often its caller's beat asks.
    last_rebalance: Mutex<Option<Instant>>,
}

impl Dispatcher {
    /// A dispatcher over `specs`, joining the backends `registry` already
    /// holds.
    ///
    /// # Errors
    ///
    /// [`DispatchError::Config`] when `specs` lists one id twice or the
    /// registry would hold more than 64 backends.
    pub fn new(
        registry: &mut BackendRegistry,
        specs: Vec<BackendSpec>,
        config: DispatcherConfig,
    ) -> Result<Self, DispatchError> {
        let joined = registry.join(specs, &config.scheduler)?;
        Ok(Self {
            joined: RwLock::new(Arc::new(joined)),
            config,
            pool: Arc::clone(&registry.pool),
            last_rebalance: Mutex::new(None),
        })
    }

    /// Points this dispatcher at `specs`, joining the backends `registry`
    /// already holds. Calls made from now on go to the new list; a call
    /// already waiting on a backend the list drops finishes there.
    ///
    /// # Errors
    ///
    /// As [`Self::new`]. The old list stays in place.
    pub fn set_backends(
        &self,
        registry: &mut BackendRegistry,
        specs: Vec<BackendSpec>,
    ) -> Result<(), DispatchError> {
        let joined = registry.join(specs, &self.config.scheduler)?;
        *self.joined.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(joined);
        Ok(())
    }

    /// The current list.
    #[must_use]
    pub fn backends(&self) -> Vec<Arc<Backend>> {
        self.snapshot().backends.clone()
    }

    /// The bit set of [`CallMeta::allowed`] naming exactly the backends of
    /// the current list: what every call is stamped with.
    #[must_use]
    pub fn allowed_backends(&self) -> u64 {
        self.snapshot().mask
    }

    /// The most calls the current list's backends may run at once, all
    /// together.
    #[must_use]
    pub fn ceiling(&self) -> u32 {
        self.snapshot()
            .backends
            .iter()
            .map(|backend| backend.scheduler.ceiling())
            .sum()
    }

    fn snapshot(&self) -> Arc<Joined> {
        Arc::clone(&self.joined.read().unwrap_or_else(PoisonError::into_inner))
    }

    /// Hands waiting calls from a busier backend to one that would start them
    /// sooner, at most once every `interval`.
    ///
    /// Placement happens when a call is made and nothing moves it afterwards,
    /// so a burst that lands on one backend sits there while another drains.
    /// This is the second half: a call that has not started is moved, keeping
    /// its place in the queue's aging, so the waiting work follows the idle
    /// backends. The caller drives it on its own beat.
    ///
    /// Only backends whose breakers admit are involved, and only a backend
    /// the call is allowed on ([`CallMeta::allowed`]), so a dispatcher
    /// pinned to one backend still holds its calls there.
    pub fn rebalance(&self, interval: Duration) {
        let joined = self.snapshot();
        if joined.backends.len() < 2 {
            return;
        }
        let now = Instant::now();
        {
            let mut last = self
                .last_rebalance
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if last.is_some_and(|at| now.saturating_duration_since(at) < interval) {
                return;
            }
            *last = Some(now);
        }
        let live: Vec<&Arc<Backend>> = joined
            .backends
            .iter()
            .filter(|backend| backend.breaker.remaining(now).is_none())
            .collect();
        if live.len() < 2 {
            return;
        }
        // CONTEXT: the calls still waiting are the bound on moves (a move
        // takes one), and each move accepted below strictly lowers the
        // longest estimated start, so the pass settles instead of shuffling
        // calls between backends forever over a difference no move can close.
        let budget: usize = live
            .iter()
            .map(|backend| backend.scheduler.load().waiting)
            .sum();
        let mut moved_total = 0usize;
        for _ in 0..budget {
            let loads: Vec<Load> = live
                .iter()
                .map(|backend| backend.scheduler.load())
                .collect();
            let Some(target) = (0..live.len()).min_by_key(|&index| loads[index].start_ms()) else {
                break;
            };
            let mut sources: Vec<usize> =
                (0..live.len()).filter(|&index| index != target).collect();
            sources.sort_by_key(|&index| std::cmp::Reverse(loads[index].start_ms()));
            let mut moved = false;
            for source in sources {
                if !lowers_the_worst(&loads, source, target) {
                    continue;
                }
                if move_waiting(
                    &live[source].scheduler,
                    &live[target].scheduler,
                    live[target].bit,
                ) {
                    moved = true;
                    break;
                }
            }
            if !moved {
                break;
            }
            moved_total += 1;
        }
        if moved_total > 0 {
            tracing::debug!(
                target: "cuca::scheduler",
                "rebalanced {moved_total} waiting call(s) over {} backend(s)",
                live.len()
            );
        }
    }

    /// Sends `request` to the least-loaded backend whose breaker admits it
    /// and returns the streamed reply.
    ///
    /// Before a slot is reserved, every lazy backend of the list whose
    /// breaker admits and whose client is not built yet is built, all at
    /// once; one that fails opens its breaker. When the chosen backend then
    /// proves unreachable, the call goes once to each other backend that
    /// still admits; every other failure is the call's own. The backend's
    /// [`BackendSpec::model`] and [`BackendSpec::thinking`] overwrite the
    /// request's fields before it is sent; nothing else on the request
    /// changes. `meta.allowed` is replaced by the bits of the current list.
    ///
    /// # Errors
    ///
    /// [`DispatchError::QueueFull`] when the chosen backend's queue is full;
    /// [`DispatchError::Unreachable`] when every admitting backend failed to
    /// connect or none admits; [`DispatchError::Timeout`] past the
    /// configured deadline; [`DispatchError::Empty`] and
    /// [`DispatchError::ReasoningOnly`] for a reply without text;
    /// [`DispatchError::InputsChanged`] when the gate reports the task stale;
    /// [`DispatchError::Call`] for any other backend error.
    pub async fn call(
        &self,
        request: UnifiedRequest,
        mut meta: CallMeta,
    ) -> Result<Completion, DispatchError> {
        let joined = self.snapshot();
        let backends = &joined.backends[..];
        meta.allowed = joined.mask;
        self.build_lazy(backends).await;
        let mut pending = Some((request, meta));
        // CONTEXT: `Vec::new` does not allocate; only a failover pushes.
        let mut tried: Vec<usize> = Vec::new();
        let mut refused: Option<DispatchError> = None;
        loop {
            let now = Instant::now();
            // CONTEXT: the scheduler keeps the meta it queues and the
            // attempt consumes the request, so both are cloned only while
            // another backend could still take a failover.
            let next = if tried.len() + 1 < backends.len() {
                pending.clone()
            } else {
                pending.take()
            };
            let Some((request, meta)) = next else {
                return Err(refused.unwrap_or_else(|| unreachable(backends, now)));
            };
            // CONTEXT: the slot is reserved before the clock starts, so a
            // deadline measures the model rather than the queue.
            let queued = Instant::now();
            let Some((index, backend, id)) = self.reserve(backends, meta, &tried)? else {
                return Err(refused.unwrap_or_else(|| unreachable(backends, now)));
            };
            let wait = tracing::debug_span!(
                target: "cuca::scheduler",
                "scheduler.wait",
                backend = backend.id().as_str(),
                level = tracing::field::Empty,
                wait_ms = tracing::field::Empty,
            );
            let slot = backend
                .scheduler
                .await_placement(id)
                .instrument(wait.clone())
                .await?;
            let wait_ms = elapsed_ms(queued);
            wait.record("level", slot.level());
            wait.record("wait_ms", wait_ms);
            match self.attempt(&backend, request, slot, wait_ms).await {
                Err(error @ DispatchError::Unreachable { .. }) => {
                    tried.push(index);
                    refused = Some(error);
                }
                other => return other,
            }
        }
    }

    /// Builds every lazy backend of `backends` whose breaker admits and that
    /// has no client yet, all at once; one that fails opens its breaker, so
    /// [`pick`] passes it over for a cooldown.
    ///
    /// CONTEXT: before the call reserves a slot rather than inside
    /// [`Self::attempt`], so a long first build (a model download and load)
    /// is neither counted in the backend's measured pace nor under the
    /// deadline. Every waiting call shares the one build, and two lazy
    /// backends build side by side rather than one after the other.
    async fn build_lazy(&self, backends: &[Arc<Backend>]) {
        let now = Instant::now();
        let mut builds = tokio::task::JoinSet::new();
        for backend in backends {
            if backend.has_client() || backend.breaker.remaining(now).is_some() {
                continue;
            }
            let backend = Arc::clone(backend);
            let cooldown = self.config.unreachable_cooldown;
            builds.spawn(async move {
                if backend.client().await.is_err() {
                    backend.trip(Instant::now(), cooldown);
                }
            });
        }
        while builds.join_next().await.is_some() {}
    }

    /// Selects the least-loaded admitting backend of `backends` not yet
    /// `tried` and enqueues `meta` on it, atomically with respect to every
    /// other placement through the shared pool. Answers the backend's index,
    /// the backend, and the call id to await with
    /// [`Scheduler::await_placement`]; `None` when no backend admits.
    ///
    /// CONTEXT: without the pool's placement lock, the many calls of one
    /// burst each read every backend as idle before any enqueue lands, and
    /// the whole burst piles onto the first backend.
    fn reserve(
        &self,
        backends: &[Arc<Backend>],
        meta: CallMeta,
        tried: &[usize],
    ) -> Result<Option<(usize, Arc<Backend>, u64)>, DispatchError> {
        let _guard = self.pool.lock_placement();
        let Some(index) = pick(backends, Instant::now(), tried) else {
            return Ok(None);
        };
        let backend = Arc::clone(&backends[index]);
        let id = backend.scheduler.enqueue(meta)?;
        Ok(Some((index, backend, id)))
    }

    /// The index of the backend a call goes to at `now`, among the current
    /// list: see [`pick`].
    #[cfg(all(test, feature = "provider-openai"))]
    fn pick(&self, now: Instant, tried: &[usize]) -> Option<usize> {
        pick(&self.snapshot().backends, now, tried)
    }

    /// One attempt of a call on `backend`: its slot, its stream, its report.
    async fn attempt(
        &self,
        backend: &Backend,
        mut request: UnifiedRequest,
        slot: CallSlot,
        wait_ms: u64,
    ) -> Result<Completion, DispatchError> {
        let cooldown = self.config.unreachable_cooldown;
        let model: Arc<str> = match &backend.model {
            Some(tag) => {
                request.model = String::from(&**tag);
                Arc::clone(tag)
            }
            None => Arc::from(request.model.as_str()),
        };
        if let Some(thinking) = &backend.thinking {
            request.thinking = Some(thinking.clone());
        }

        // CONTEXT: the breaker is asked again after the queue, so a call that
        // waited behind one whose connect was just refused does not pay the
        // same refusal.
        if let Err(error) = backend.breaker.admit(Instant::now()) {
            slot.finish(CallReport::empty(CallOutcome::Failed));
            return Err(error);
        }
        let client = match backend.client().await {
            Ok(client) => client,
            Err(error) => {
                backend.trip(Instant::now(), cooldown);
                slot.finish(CallReport::empty(CallOutcome::Failed));
                return Err(DispatchError::Unreachable {
                    detail: error.to_string(),
                });
            }
        };

        let started = Instant::now();
        let span = tracing::debug_span!(
            target: "cuca::scheduler",
            "scheduler.stream",
            backend = backend.id().as_str(),
            model = &*model,
            run_ms = tracing::field::Empty,
            output_chars = tracing::field::Empty,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
        );
        let streamed = async {
            match self.config.timeout {
                Some(deadline) => {
                    match tokio::time::timeout(deadline, collect_stream(client, request)).await {
                        Ok(streamed) => streamed,
                        Err(_elapsed) => Err(DispatchError::Timeout {
                            seconds: deadline.as_secs(),
                        }),
                    }
                }
                None => collect_stream(client, request).await,
            }
        }
        .instrument(span.clone())
        .await;
        let run_ms = elapsed_ms(started);
        span.record("run_ms", run_ms);
        let reply = match streamed {
            Ok(reply) => {
                backend.close();
                reply
            }
            Err(error) => {
                if matches!(error, DispatchError::Unreachable { .. }) {
                    backend.trip(Instant::now(), cooldown);
                }
                slot.finish(CallReport::empty(CallOutcome::Failed));
                return Err(error);
            }
        };
        let output_chars =
            u64::try_from(reply.text.chars().count() + reply.reasoning_chars).unwrap_or(u64::MAX);
        span.record("output_chars", output_chars);
        // CONTEXT: `Ok` exactly when this call returns `Ok`: a reply that
        // only reasoned is a failed call, for a tuner sample as for the
        // totals.
        let answered = !reply.text.trim().is_empty();
        let report = CallReport {
            outcome: if answered {
                CallOutcome::Ok
            } else {
                CallOutcome::Failed
            },
            prompt_tokens: reply.usage.map(|usage| u64::from(usage.prompt_tokens)),
            completion_tokens: reply.usage.map(|usage| u64::from(usage.completion_tokens)),
            output_chars,
        };
        // Unknown stays unrecorded: an absent field, never a zero.
        if let Some(tokens) = report.prompt_tokens {
            span.record("prompt_tokens", tokens);
        }
        if let Some(tokens) = report.completion_tokens {
            span.record("completion_tokens", tokens);
        }
        let level = slot.level();
        slot.finish(report);

        if !answered {
            // CONTEXT: a reply with reasoning blocks and no text, as when
            // the token limit is hit mid-reasoning, looks identical to "the
            // server returned nothing" unless it is named separately.
            if reply.reasoning_chars > 0 {
                return Err(DispatchError::ReasoningOnly {
                    reasoning_chars: reply.reasoning_chars,
                });
            }
            return Err(DispatchError::Empty);
        }
        Ok(Completion {
            text: reply.text,
            model,
            backend: backend.id().clone(),
            level,
            wait_ms,
            run_ms,
            output_chars,
            reasoning_chars: reply.reasoning_chars,
            usage: reply.usage,
        })
    }
}

/// The index of the backend a call goes to at `now`: among `backends` not yet
/// `tried` whose breaker admits, the one where a call queued now would start
/// soonest (the calls waiting and running at that backend's measured pace,
/// over the calls it runs at once), the first listed on a tie.
fn pick(backends: &[Arc<Backend>], now: Instant, tried: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, Load)> = None;
    for (index, backend) in backends.iter().enumerate() {
        if tried.contains(&index) || backend.breaker.remaining(now).is_some() {
            continue;
        }
        let load = backend.scheduler.load();
        if best.is_none_or(|(_, lightest)| load.is_lighter_than(lightest)) {
            best = Some((index, load));
        }
    }
    best.map(|(index, _)| index)
}

/// The error of a call none of `backends` admits at `now`.
fn unreachable(backends: &[Arc<Backend>], now: Instant) -> DispatchError {
    let soonest = backends
        .iter()
        .filter_map(|backend| backend.breaker.remaining(now))
        .min();
    DispatchError::Unreachable {
        detail: match soonest {
            Some(remaining) => format!(
                "every backend refused a connect; next attempt in {}s",
                remaining.as_secs().max(1)
            ),
            None => "no backend is configured".to_owned(),
        },
    }
}

/// Milliseconds since `since`, saturating.
fn elapsed_ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}

struct StreamedReply {
    text: String,
    reasoning_chars: usize,
    /// The backend's token usage for the call; `None` when it reported none.
    usage: Option<TokenUsage>,
}

/// Sends `request` and drains the stream into one reply.
///
/// A [`CucaError::Transport`] returned before any stream exists means no
/// server answered the request (a refused or failed connect, or a connection
/// dropped before the response headers): [`DispatchError::Unreachable`]. Any
/// other error from the send, and every error item of the stream, is
/// [`DispatchError::Call`].
async fn collect_stream(
    client: &CucaClient,
    request: UnifiedRequest,
) -> Result<StreamedReply, DispatchError> {
    let (mut stream, response) = match client.generate_stream_with_response(request).await {
        Ok(opened) => opened,
        Err(CucaError::Transport { message }) => {
            return Err(DispatchError::Unreachable { detail: message });
        }
        Err(other) => return Err(DispatchError::Call(other)),
    };
    let mut reply = StreamedReply {
        text: String::new(),
        reasoning_chars: 0,
        usage: None,
    };
    while let Some(block) = poll_fn(|cx| stream.as_mut().poll_next(cx)).await {
        match block.map_err(DispatchError::Call)? {
            MessageContentBlock::Text(text) => reply.text.push_str(&text),
            MessageContentBlock::Thinking { reasoning, .. } => {
                reply.reasoning_chars += reasoning.chars().count();
            }
            _ => {}
        }
    }
    // CONTEXT: read `usage` itself. `prompt_tokens`/`completion_tokens` fall
    // back to a content-block count when the backend reports nothing, and a
    // block count is not a token count.
    reply.usage = response.take().and_then(|response| response.usage);
    Ok(reply)
}

#[cfg(all(test, feature = "service-scheduler", feature = "provider-openai"))]
mod tests {
    use std::io::ErrorKind;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::sync::watch;

    use super::*;
    use crate::services::scheduler::{OpenGate, Priority};
    use crate::types::{ProviderEndpoint, UnifiedMessage};

    /// CONTEXT: port 1 on loopback has no listener, so a connect is refused
    /// rather than reaching whatever server runs on the machine.
    const DEAD_URL: &str = "http://127.0.0.1:1/v1";

    fn registry() -> BackendRegistry {
        BackendRegistry::new(
            Arc::new(OpenGate::default()),
            Arc::new(watch::channel(0_u64).0),
        )
    }

    fn openai(url: &str) -> Arc<CucaClient> {
        Arc::new(
            CucaClient::builder()
                .with_provider(ProviderEndpoint::OpenAi)
                .with_base_url(url)
                .with_api_key("test")
                .build()
                .expect("a client with a base URL builds"),
        )
    }

    fn spec(url: &str) -> BackendSpec {
        BackendSpec {
            id: BackendId::parse(url),
            client: BackendClient::Ready(openai(url)),
            model: None,
            thinking: None,
        }
    }

    fn config() -> DispatcherConfig {
        DispatcherConfig::new(SchedulerConfig::new(1).expect("one call at a time is valid"))
    }

    fn specs(urls: &[&str]) -> Vec<BackendSpec> {
        urls.iter().map(|url| spec(url)).collect()
    }

    fn dispatcher(urls: &[&str]) -> Dispatcher {
        Dispatcher::new(&mut registry(), specs(urls), config()).expect("the list joins")
    }

    fn call() -> CallMeta {
        CallMeta::new("test", Priority::Medium, "subject")
    }

    fn request() -> UnifiedRequest {
        UnifiedRequest::new("model").add_message(UnifiedMessage::user("Say hi."))
    }

    fn is_unreachable<T: std::fmt::Debug>(reply: &Result<T, DispatchError>) -> bool {
        matches!(reply, Err(DispatchError::Unreachable { .. }))
    }

    /// A loopback URL nothing listens on: a port the OS just handed out and
    /// took back.
    fn closed_url() -> String {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("loopback has a free port")
            .local_addr()
            .expect("a bound address")
            .port();
        format!("http://127.0.0.1:{port}/v1")
    }

    const ANSWER: &[u8] =
        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n\
                            data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                            data: [DONE]\n\n";

    /// A loopback server that answers `count` completions with "hi", each
    /// once `delay` has passed, and its URL.
    async fn serving(count: usize, delay: Duration) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback has a free port");
        let port = listener.local_addr().expect("a bound address").port();
        let server = tokio::spawn(async move {
            for _ in 0..count {
                let (mut socket, _) = listener.accept().await.expect("the client connects");
                read_request(&mut socket).await;
                tokio::time::sleep(delay).await;
                socket
                    .write_all(ANSWER)
                    .await
                    .expect("the reply is written");
                socket.shutdown().await.expect("the reply is finished");
            }
        });
        (format!("http://127.0.0.1:{port}/v1"), server)
    }

    async fn answering_server() -> (String, tokio::task::JoinHandle<()>) {
        serving(1, Duration::ZERO).await
    }

    /// A loopback listener that never accepts, to tell whether a call
    /// connected to it, and its URL.
    fn silent_listener() -> (String, std::net::TcpListener) {
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("loopback has a free port");
        listener
            .set_nonblocking(true)
            .expect("a listener can be non-blocking");
        let port = listener.local_addr().expect("a bound address").port();
        (format!("http://127.0.0.1:{port}/v1"), listener)
    }

    fn connected(listener: &std::net::TcpListener) -> bool {
        match listener.accept() {
            Ok(_) => true,
            Err(error) if error.kind() == ErrorKind::WouldBlock => false,
            Err(error) => panic!("the listener failed: {error}"),
        }
    }

    /// Reads one request's headers and its `content-length` body, so the
    /// reply is not answered over unread bytes.
    async fn read_request(socket: &mut tokio::net::TcpStream) {
        let mut received = Vec::new();
        let mut chunk = [0_u8; 4096];
        loop {
            let read = socket.read(&mut chunk).await.expect("the request arrives");
            assert_ne!(read, 0, "the client hung up mid-request");
            received.extend_from_slice(&chunk[..read]);
            let Some(end) = received.windows(4).position(|w| w == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&received[..end]).to_ascii_lowercase();
            let length: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .map_or(0, |value| value.trim().parse().expect("a numeric length"));
            if received.len() >= end + 4 + length {
                return;
            }
        }
    }

    mod breaker {
        use super::*;

        #[test]
        fn admits_a_call_when_nothing_has_been_refused() {
            let breaker = Breaker::default();

            assert!(breaker.admit(Instant::now()).is_ok());
        }

        #[test]
        fn refuses_a_call_inside_the_cooldown_after_a_trip() {
            let breaker = Breaker::default();
            let tripped = Instant::now();
            breaker.trip(tripped, UNREACHABLE_COOLDOWN);

            let admitted = breaker.admit(tripped + UNREACHABLE_COOLDOWN / 2);

            assert!(
                matches!(admitted, Err(DispatchError::Unreachable { .. })),
                "expected unreachable, got {admitted:?}"
            );
        }

        #[test]
        fn admits_a_real_attempt_once_the_cooldown_has_passed() {
            let breaker = Breaker::default();
            let tripped = Instant::now();
            breaker.trip(tripped, UNREACHABLE_COOLDOWN);

            assert!(breaker.admit(tripped + UNREACHABLE_COOLDOWN).is_ok());
        }

        #[test]
        fn admits_calls_inside_the_cooldown_once_closed() {
            let breaker = Breaker::default();
            let tripped = Instant::now();
            breaker.trip(tripped, UNREACHABLE_COOLDOWN);

            breaker.close();

            assert!(breaker.admit(tripped).is_ok());
        }

        #[test]
        fn reports_a_first_trip_once() {
            let breaker = Breaker::default();

            let first = breaker.trip(Instant::now(), UNREACHABLE_COOLDOWN);
            let again = breaker.trip(Instant::now(), UNREACHABLE_COOLDOWN);

            assert_eq!((first, again), (true, false));
        }
    }

    mod config {
        use super::*;

        #[test]
        fn rejects_a_zero_cooldown() {
            let rejected = config().with_unreachable_cooldown(Duration::ZERO);

            assert!(matches!(rejected, Err(DispatchError::Config(_))));
        }

        #[test]
        fn defaults_to_twenty_seconds_and_no_timeout() {
            let config = config();

            assert_eq!(
                (config.unreachable_cooldown, config.timeout),
                (Duration::from_secs(20), None)
            );
        }
    }

    mod call {
        use super::*;

        #[tokio::test]
        async fn trips_the_breaker_on_a_refused_connect() {
            let dispatcher = dispatcher(&[DEAD_URL]);

            let reply = dispatcher.call(request(), call()).await;

            assert!(
                is_unreachable(&reply),
                "a refused connect must read as unreachable, got {reply:?}"
            );
            assert!(
                dispatcher.backends()[0]
                    .breaker
                    .admit(Instant::now())
                    .is_err(),
                "the refusal must open the breaker"
            );
        }

        #[tokio::test]
        async fn returns_unreachable_without_connecting_while_the_breaker_is_open() {
            let (url, listener) = silent_listener();
            let dispatcher = dispatcher(&[&url]);
            dispatcher.backends()[0]
                .breaker
                .trip(Instant::now(), UNREACHABLE_COOLDOWN);

            let reply = dispatcher.call(request(), call()).await;

            assert!(
                is_unreachable(&reply),
                "expected unreachable, got {reply:?}"
            );
            assert!(
                !connected(&listener),
                "an open breaker must not open a socket"
            );
        }

        #[tokio::test]
        async fn tries_the_backend_again_once_the_cooldown_has_passed() {
            let dispatcher = dispatcher(&[DEAD_URL]);
            let tripped = Instant::now()
                .checked_sub(UNREACHABLE_COOLDOWN)
                .expect("the clock is past one cooldown");
            dispatcher.backends()[0]
                .breaker
                .trip(tripped, UNREACHABLE_COOLDOWN);

            let _ = dispatcher.call(request(), call()).await;

            let reopened = (*dispatcher.backends()[0].breaker.lock())
                .expect("still unreachable, so open again");
            assert!(
                reopened > tripped + UNREACHABLE_COOLDOWN,
                "a fresh refusal must move the deadline, which only a real attempt can"
            );
        }

        #[tokio::test]
        async fn closes_the_breaker_when_the_backend_answers() {
            let (url, server) = answering_server().await;
            let dispatcher = dispatcher(&[&url]);
            let tripped = Instant::now()
                .checked_sub(UNREACHABLE_COOLDOWN)
                .expect("the clock is past one cooldown");
            dispatcher.backends()[0]
                .breaker
                .trip(tripped, UNREACHABLE_COOLDOWN);

            let reply = dispatcher
                .call(request(), call())
                .await
                .expect("the backend answered");
            server.await.expect("the server finished");

            assert_eq!(reply.text, "hi");
            assert_eq!(
                *dispatcher.backends()[0].breaker.lock(),
                None,
                "an answer closes the breaker"
            );
        }

        #[tokio::test]
        async fn describes_the_reply() {
            let (url, server) = answering_server().await;
            let dispatcher = dispatcher(&[&url]);

            let reply = dispatcher
                .call(request(), call())
                .await
                .expect("the backend answered");
            server.await.expect("the server finished");

            assert_eq!(
                (
                    reply.text.as_str(),
                    &*reply.model,
                    reply.backend.as_str(),
                    reply.level,
                    reply.output_chars,
                    reply.reasoning_chars
                ),
                ("hi", "model", BackendId::parse(&url).as_str(), 1, 2, 0)
            );
        }

        #[tokio::test]
        async fn tags_the_reply_with_the_backends_model() {
            let (url, server) = answering_server().await;
            let mut tagged = spec(&url);
            tagged.model = Some(Arc::from("local/tag"));
            let dispatcher =
                Dispatcher::new(&mut registry(), vec![tagged], config()).expect("the list joins");

            let reply = dispatcher
                .call(request(), call())
                .await
                .expect("the backend answered");
            server.await.expect("the server finished");

            assert_eq!(&*reply.model, "local/tag");
        }

        #[tokio::test]
        async fn refuses_a_call_past_the_queue_bound() {
            let (url, _listener) = silent_listener();
            let scheduler = SchedulerConfig::new(1)
                .and_then(|config| config.with_max_waiting(1))
                .expect("a bound of one is valid");
            let dispatcher = Dispatcher::new(
                &mut registry(),
                specs(&[&url]),
                DispatcherConfig::new(scheduler),
            )
            .expect("the list joins");
            let scheduler = Arc::clone(dispatcher.backends()[0].scheduler());
            let _running = scheduler
                .acquire(call())
                .await
                .expect("the first call starts");
            let _waiting = scheduler.enqueue(call()).expect("one call may wait");

            let reply = dispatcher.call(request(), call()).await;

            assert!(
                matches!(reply, Err(DispatchError::QueueFull { .. })),
                "got {reply:?}"
            );
        }
    }

    mod pick {
        use super::*;

        #[tokio::test]
        async fn sends_a_call_to_the_least_loaded_backend() {
            let dispatcher = dispatcher(&[DEAD_URL, &closed_url()]);
            let _busy = dispatcher.backends()[0]
                .scheduler
                .acquire(call())
                .await
                .expect("an idle scheduler dispatches at once");

            let picked = dispatcher.pick(Instant::now(), &[]);

            assert_eq!(picked, Some(1), "the idle second backend is lighter");
        }

        #[tokio::test]
        async fn prefers_idle_backend_over_busy_backend_with_faster_pace() {
            let dispatcher = dispatcher(&[DEAD_URL, &closed_url()]);
            // Backend 0 completes a fast call, lowering its measured pace.
            let slot = dispatcher.backends()[0]
                .scheduler
                .acquire(call())
                .await
                .expect("slot");
            slot.finish(CallReport {
                outcome: CallOutcome::Ok,
                prompt_tokens: Some(10),
                completion_tokens: Some(10),
                output_chars: 50,
            });
            assert!(
                dispatcher.backends()[0].scheduler.load().pace_ms
                    < dispatcher.backends()[1].scheduler.load().pace_ms
            );
            // Backend 0 now has one call in flight.
            let _busy = dispatcher.backends()[0]
                .scheduler
                .acquire(call())
                .await
                .expect("slot");

            let picked = dispatcher.pick(Instant::now(), &[]);

            assert_eq!(
                picked,
                Some(1),
                "the idle backend must be picked over the busy one even if the busy one has a faster measured pace"
            );
        }

        #[tokio::test]
        async fn weighs_a_backend_by_how_long_its_answers_take() {
            let (quick_url, quick) = answering_server().await;
            let quick_dispatcher = dispatcher(&[&quick_url]);
            quick_dispatcher
                .call(request(), call())
                .await
                .expect("the backend answered");
            quick.await.expect("the server finished");

            let (slow_url, slow) = serving(1, Duration::from_millis(200)).await;
            let slow_dispatcher = dispatcher(&[&slow_url]);
            slow_dispatcher
                .call(request(), call())
                .await
                .expect("the backend answered");
            slow.await.expect("the server finished");

            let quick_pace = quick_dispatcher.backends()[0].scheduler.load().pace_ms;
            let slow_pace = slow_dispatcher.backends()[0].scheduler.load().pace_ms;

            assert!(
                slow_pace > quick_pace,
                "200 ms a call must read slower than an immediate answer: {slow_pace} against {quick_pace}"
            );
        }

        #[tokio::test]
        async fn breaks_a_tie_by_list_order() {
            let dispatcher = dispatcher(&[&closed_url(), DEAD_URL]);

            let picked = dispatcher.pick(Instant::now(), &[]);

            assert_eq!(picked, Some(0));
        }

        #[tokio::test]
        async fn skips_a_backend_whose_breaker_is_open() {
            let dispatcher = dispatcher(&[DEAD_URL, &closed_url()]);
            dispatcher.backends()[0]
                .breaker
                .trip(Instant::now(), UNREACHABLE_COOLDOWN);

            let picked = dispatcher.pick(Instant::now(), &[]);

            assert_eq!(picked, Some(1));
        }

        #[tokio::test]
        async fn fails_unreachable_without_a_socket_when_every_breaker_is_open() {
            let (first_url, first) = silent_listener();
            let (second_url, second) = silent_listener();
            let dispatcher = dispatcher(&[&first_url, &second_url]);
            for backend in &dispatcher.backends() {
                backend.breaker.trip(Instant::now(), UNREACHABLE_COOLDOWN);
            }

            let reply = dispatcher.call(request(), call()).await;

            assert!(is_unreachable(&reply), "got {reply:?}");
            assert_eq!(
                (connected(&first), connected(&second)),
                (false, false),
                "an open breaker must not open a socket"
            );
        }
    }

    mod failover {
        use super::*;

        #[tokio::test]
        async fn tries_the_next_backend_when_one_refuses_a_connect() {
            let (answering, server) = answering_server().await;
            let dispatcher = dispatcher(&[DEAD_URL, &answering]);

            let reply = dispatcher
                .call(request(), call())
                .await
                .expect("the second backend answered");
            server.await.expect("the server finished");

            assert_eq!(
                (
                    reply.text.as_str(),
                    dispatcher.backends()[0]
                        .breaker
                        .remaining(Instant::now())
                        .is_some()
                ),
                ("hi", true),
                "the refusing backend's breaker opens and the call is served elsewhere"
            );
        }

        #[tokio::test]
        async fn returns_unreachable_once_every_backend_refused() {
            let dispatcher = dispatcher(&[DEAD_URL, &closed_url()]);

            let reply = dispatcher.call(request(), call()).await;

            assert!(is_unreachable(&reply), "got {reply:?}");
            assert!(
                dispatcher
                    .backends()
                    .iter()
                    .all(|backend| backend.breaker.remaining(Instant::now()).is_some()),
                "each backend was tried once and tripped"
            );
        }

        #[tokio::test]
        async fn keeps_an_http_error_on_the_backend_that_answered_it() {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("loopback has a free port");
            let port = listener.local_addr().expect("a bound address").port();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.expect("the client connects");
                read_request(&mut socket).await;
                socket
                    .write_all(
                        b"HTTP/1.1 500 Internal Server Error\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    )
                    .await
                    .expect("the reply is written");
                socket.shutdown().await.expect("the reply is finished");
            });
            let (spare_url, spare) = silent_listener();
            let dispatcher = dispatcher(&[&format!("http://127.0.0.1:{port}/v1"), &spare_url]);

            let reply = dispatcher.call(request(), call()).await;
            server.await.expect("the server finished");

            assert!(
                matches!(reply, Err(DispatchError::Call(_))),
                "an error status is the call's own failure, got {reply:?}"
            );
            assert!(!connected(&spare), "an HTTP error must not fail over");
        }
    }

    mod backend_registry {
        use super::*;

        #[tokio::test]
        async fn shares_each_backend_between_dispatchers_over_the_same_ids() {
            let mut registry = registry();
            let main = Dispatcher::new(
                &mut registry,
                specs(&["http://127.0.0.1:1234/v1", "http://192.168.68.77:1234/v1"]),
                config(),
            )
            .expect("the main list joins");
            let story = Dispatcher::new(
                &mut registry,
                specs(&["http://localhost:1234/v1/", "http://192.168.68.77:1234/v1"]),
                config(),
            )
            .expect("the story list joins");

            let shared: Vec<bool> = main
                .backends()
                .iter()
                .zip(&story.backends())
                .map(|(a, b)| Arc::ptr_eq(&a.scheduler, &b.scheduler))
                .collect();
            assert_eq!((shared, registry.schedulers().len()), (vec![true, true], 2));
        }

        #[tokio::test]
        async fn keeps_dispatchers_over_disjoint_ids_apart() {
            let mut registry = registry();
            let main = Dispatcher::new(&mut registry, specs(&[DEAD_URL]), config())
                .expect("the main list joins");
            let story = Dispatcher::new(&mut registry, specs(&["http://127.0.0.1:2/v1"]), config())
                .expect("the story list joins");

            assert!(!Arc::ptr_eq(
                &main.backends()[0].scheduler,
                &story.backends()[0].scheduler
            ));
            assert_eq!(registry.schedulers().len(), 2);
        }

        #[tokio::test]
        async fn refuses_one_id_listed_twice() {
            let rejected = Dispatcher::new(&mut registry(), specs(&[DEAD_URL, DEAD_URL]), config());

            assert!(matches!(rejected, Err(DispatchError::Config(_))));
        }

        #[tokio::test]
        async fn refuses_a_sixty_fifth_backend() {
            let urls: Vec<String> = (1..=65)
                .map(|port| format!("http://127.0.0.1:{port}/v1"))
                .collect();
            let list: Vec<&str> = urls.iter().map(String::as_str).collect();

            let rejected = Dispatcher::new(&mut registry(), specs(&list), config());

            assert!(matches!(rejected, Err(DispatchError::Config(_))));
        }
    }

    mod set_backends {
        use super::*;

        fn ids(dispatcher: &Dispatcher) -> Vec<String> {
            dispatcher
                .backends()
                .iter()
                .map(|backend| backend.id().as_str().to_owned())
                .collect()
        }

        #[tokio::test]
        async fn points_new_calls_at_the_new_list() {
            let mut registry = registry();
            let dispatcher =
                Dispatcher::new(&mut registry, specs(&["http://127.0.0.1:1/v1"]), config())
                    .expect("the list joins");

            dispatcher
                .set_backends(
                    &mut registry,
                    specs(&["http://127.0.0.1:2/v1", "http://127.0.0.1:3/v1"]),
                )
                .expect("both entries join");

            assert_eq!(
                (ids(&dispatcher), registry.schedulers().len()),
                (
                    vec![
                        BackendId::parse("http://127.0.0.1:2/v1")
                            .as_str()
                            .to_owned(),
                        BackendId::parse("http://127.0.0.1:3/v1")
                            .as_str()
                            .to_owned(),
                    ],
                    3
                ),
                "the dropped backend stays in the registry"
            );
        }

        #[tokio::test]
        async fn gets_back_the_same_scheduler_for_a_backend_listed_again() {
            let mut registry = registry();
            let dispatcher =
                Dispatcher::new(&mut registry, specs(&["http://127.0.0.1:1/v1"]), config())
                    .expect("the list joins");
            let first = Arc::clone(&dispatcher.backends()[0]);
            dispatcher
                .set_backends(&mut registry, specs(&["http://127.0.0.1:2/v1"]))
                .expect("the list swaps");

            dispatcher
                .set_backends(&mut registry, specs(&["http://127.0.0.1:1/v1"]))
                .expect("the first backend joins again");

            assert!(Arc::ptr_eq(&first, &dispatcher.backends()[0]));
        }

        #[tokio::test]
        async fn keeps_the_old_list_when_an_entry_cannot_join() {
            let mut registry = registry();
            let dispatcher =
                Dispatcher::new(&mut registry, specs(&["http://127.0.0.1:1/v1"]), config())
                    .expect("the list joins");
            let before = ids(&dispatcher);

            let rejected = dispatcher.set_backends(
                &mut registry,
                specs(&["http://127.0.0.1:2/v1", "http://127.0.0.1:2/v1"]),
            );

            assert!(rejected.is_err());
            assert_eq!(ids(&dispatcher), before);
        }
    }

    mod rebalance {
        use super::*;

        const URL_A: &str = "http://127.0.0.1:1/v1";
        const URL_B: &str = "http://127.0.0.1:2/v1";

        /// A task waiting for `scheduler`'s bucket, holding whatever slot it
        /// gets until it is aborted: what a backlog actually looks like.
        fn waiting_on(scheduler: &Arc<Scheduler>, call: CallMeta) -> tokio::task::JoinHandle<()> {
            let scheduler = Arc::clone(scheduler);
            tokio::spawn(async move {
                let Ok(_slot) = scheduler.acquire(call).await else {
                    return;
                };
                std::future::pending::<()>().await;
            })
        }

        /// Waits (without sleeping) until `scheduler` holds `expected` calls,
        /// so a spawned acquirer has queued.
        async fn settle(scheduler: &Arc<Scheduler>, expected: usize) {
            for _ in 0..10_000 {
                if scheduler.load().calls == expected {
                    return;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(scheduler.load().calls, expected, "the call never queued");
        }

        #[tokio::test]
        async fn hands_a_waiting_call_to_an_idle_backend() {
            let dispatcher = dispatcher(&[URL_A, URL_B]);
            let backends = dispatcher.backends();
            let held = backends[0]
                .scheduler()
                .acquire(call())
                .await
                .expect("the first call starts");
            let waiting = waiting_on(backends[0].scheduler(), call());
            settle(backends[0].scheduler(), 2).await;

            dispatcher.rebalance(Duration::ZERO);

            assert_eq!(
                backends[0].scheduler().load().calls,
                1,
                "only the held call is left"
            );
            assert_eq!(
                backends[1].scheduler().load().calls,
                1,
                "the idle backend took the waiting call"
            );
            assert!(
                backends[1].scheduler().snapshot().buckets[0].call.is_some(),
                "the moved call is running on the backend that took it"
            );
            held.finish(CallReport::empty(CallOutcome::Ok));
            waiting.abort();
        }

        #[tokio::test]
        async fn never_hands_a_call_to_a_backend_its_dispatcher_does_not_list() {
            let mut registry = registry();
            let main = Dispatcher::new(&mut registry, specs(&[URL_A, URL_B]), config())
                .expect("the main list joins");
            let story = Dispatcher::new(&mut registry, specs(&[URL_A]), config())
                .expect("the story list joins");
            let (main_backends, story_backends) = (main.backends(), story.backends());
            assert_eq!(story.allowed_backends(), story_backends[0].bit);
            assert_eq!(
                main.allowed_backends(),
                story_backends[0].bit | main_backends[1].bit
            );
            assert_eq!(
                story.allowed_backends() & main_backends[1].bit,
                0,
                "the story dispatcher is pinned to the first backend"
            );
            let held = main_backends[0]
                .scheduler()
                .acquire(call())
                .await
                .expect("the first call starts");
            let pinned = {
                let mut pinned = call();
                pinned.allowed = story.allowed_backends();
                waiting_on(main_backends[0].scheduler(), pinned)
            };
            settle(main_backends[0].scheduler(), 2).await;

            main.rebalance(Duration::ZERO);

            assert_eq!(
                main_backends[1].scheduler().load().calls,
                0,
                "the pinned call is not moved to a backend its dispatcher never lists"
            );
            assert_eq!(main_backends[0].scheduler().load().calls, 2);
            held.finish(CallReport::empty(CallOutcome::Ok));
            pinned.abort();
        }

        #[tokio::test]
        async fn never_hands_a_call_to_a_backend_that_refused_a_connect() {
            let dispatcher = dispatcher(&[URL_A, URL_B]);
            let backends = dispatcher.backends();
            let held = backends[0]
                .scheduler()
                .acquire(call())
                .await
                .expect("the first call starts");
            let waiting = waiting_on(backends[0].scheduler(), call());
            settle(backends[0].scheduler(), 2).await;
            backends[1]
                .breaker
                .trip(Instant::now(), UNREACHABLE_COOLDOWN);

            dispatcher.rebalance(Duration::ZERO);

            assert_eq!(
                backends[1].scheduler().load().calls,
                0,
                "a backend whose breaker is open is left alone"
            );
            assert_eq!(backends[0].scheduler().load().calls, 2);
            held.finish(CallReport::empty(CallOutcome::Ok));
            waiting.abort();
        }

        #[tokio::test]
        async fn waits_out_its_interval_before_moving_anything_again() {
            let dispatcher = dispatcher(&[URL_A, URL_B]);
            let backends = dispatcher.backends();
            let held_a = backends[0]
                .scheduler()
                .acquire(call())
                .await
                .expect("a starts");
            let held_b = backends[1]
                .scheduler()
                .acquire(call())
                .await
                .expect("b starts");
            let first = waiting_on(backends[0].scheduler(), call());
            let second = waiting_on(backends[0].scheduler(), call());
            settle(backends[0].scheduler(), 3).await;
            let interval = Duration::from_secs(3_600);

            dispatcher.rebalance(interval);

            assert_eq!(
                backends[1].scheduler().load().calls,
                2,
                "one waiter moved to the backend with the shorter queue"
            );
            let third = waiting_on(backends[0].scheduler(), call());
            settle(backends[0].scheduler(), 3).await;
            dispatcher.rebalance(interval);

            assert_eq!(
                backends[1].scheduler().load().calls,
                2,
                "inside the interval nothing moves"
            );
            held_a.finish(CallReport::empty(CallOutcome::Ok));
            held_b.finish(CallReport::empty(CallOutcome::Ok));
            first.abort();
            second.abort();
            third.abort();
        }
    }

    mod lazy {
        use super::*;

        /// Builds a client over `url`, counting its builds, or fails.
        #[derive(Debug)]
        struct CountingFactory {
            url: String,
            builds: AtomicUsize,
            fail: bool,
        }

        impl CountingFactory {
            fn new(url: &str, fail: bool) -> Arc<Self> {
                Arc::new(Self {
                    url: url.to_owned(),
                    builds: AtomicUsize::new(0),
                    fail,
                })
            }
        }

        impl ClientFactory for CountingFactory {
            fn build(
                &self,
            ) -> Pin<Box<dyn Future<Output = Result<Arc<CucaClient>, CucaError>> + Send + '_>>
            {
                Box::pin(async move {
                    self.builds.fetch_add(1, Ordering::SeqCst);
                    tokio::task::yield_now().await;
                    if self.fail {
                        Err(CucaError::Config("the model did not load".to_owned()))
                    } else {
                        Ok(openai(&self.url))
                    }
                })
            }
        }

        fn lazy(url: &str, factory: &Arc<CountingFactory>) -> Dispatcher {
            let lazy = BackendSpec {
                id: BackendId::parse(url),
                client: BackendClient::Lazy(Arc::clone(factory) as Arc<dyn ClientFactory>),
                model: None,
                thinking: None,
            };
            Dispatcher::new(&mut registry(), vec![lazy], config()).expect("the list joins")
        }

        #[tokio::test]
        async fn builds_a_lazy_backend_once_before_the_first_slot() {
            let (url, server) = serving(2, Duration::ZERO).await;
            let factory = CountingFactory::new(&url, false);
            let dispatcher = lazy(&url, &factory);
            assert_eq!(
                factory.builds.load(Ordering::SeqCst),
                0,
                "nothing builds at join"
            );

            let (first, second) = tokio::join!(
                dispatcher.call(request(), call()),
                dispatcher.call(request(), call())
            );
            server.await.expect("the server finished");

            assert_eq!(
                (
                    first.is_ok(),
                    second.is_ok(),
                    factory.builds.load(Ordering::SeqCst)
                ),
                (true, true, 1),
                "two concurrent calls share one build"
            );
        }

        #[tokio::test]
        async fn trips_the_breaker_when_the_factory_fails() {
            let factory = CountingFactory::new(DEAD_URL, true);
            let dispatcher = lazy(DEAD_URL, &factory);

            let reply = dispatcher.call(request(), call()).await;
            let again = dispatcher.call(request(), call()).await;

            assert!(is_unreachable(&reply), "got {reply:?}");
            assert!(is_unreachable(&again), "got {again:?}");
            assert_eq!(
                factory.builds.load(Ordering::SeqCst),
                1,
                "the open breaker keeps the second call from building again"
            );
            assert!(
                dispatcher.backends()[0]
                    .breaker
                    .remaining(Instant::now())
                    .is_some()
            );
        }
    }
}
