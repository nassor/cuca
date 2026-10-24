use std::borrow::Cow;
use std::collections::VecDeque;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::{Notify, watch};

use super::tuner::{Sample, Tuner, TunerSnapshot};

/// The default level from which bucket 0 becomes the reserved bucket.
///
/// CONTEXT: below it, reserving one of two or three slots for the reserved
/// lane would halve the throughput of every other call.
const RESERVED_LANE_FROM_LEVEL: u32 = 4;

/// The default number of calls that may wait at one scheduler.
const MAX_WAITING: usize = 256;

/// The largest ceiling a scheduler accepts.
const MAX_CEILING: u32 = 64;

/// Distinct call kinds counted apart; later ones fold into [`OTHER_KIND`].
const MAX_KINDS: usize = 32;

/// The kind the totals of every kind past [`MAX_KINDS`] are counted under.
const OTHER_KIND: &str = "other";

/// Completed calls kept for the snapshot.
const RECENT_CALLS: usize = 64;

/// Where time comes from: the real clock in production, a hand-moved one in
/// tests.
pub trait Clock: Send + Sync + std::fmt::Debug {
    /// The current instant.
    fn now(&self) -> Instant;
}

/// [`Instant::now`].
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// How soon a call should go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// Background work that can wait.
    Low,
    /// Ordinary work.
    Medium,
    /// Work something else is blocked on.
    High,
    /// A rerun of a call whose inputs changed under it; see
    /// [`CallMeta::reruns`].
    Urgent,
}

impl Priority {
    /// The value a call of this priority starts waiting with.
    #[must_use]
    pub const fn initial_value(self) -> u64 {
        match self {
            Self::Low => 1,
            Self::Medium => 10,
            Self::High => 100,
            Self::Urgent => 10_000,
        }
    }

    /// Its wire spelling, as serialised.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Urgent => "urgent",
        }
    }
}

/// Which lane a call queues in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Lane {
    /// The ordinary queue: every bucket serves it.
    Shared,
    /// The reserved lane. From [`SchedulerConfig::reserved_lane_from_level`]
    /// up, bucket 0 serves the best reserved call first and any shared call
    /// only when no reserved call waits, and the other buckets never serve a
    /// reserved call. Below that level it behaves as [`Lane::Shared`].
    Reserved,
}

/// Who a model call is for, as the scheduler queues and shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallMeta {
    /// A free label for what the call is for. The scheduler keeps totals per
    /// kind, for at most 32 distinct kinds; later ones are counted as
    /// `"other"`.
    pub kind: Cow<'static, str>,
    /// How soon the call goes when it is a first attempt.
    pub priority: Priority,
    /// The lane it queues in.
    pub lane: Lane,
    /// What the call is about, shown in snapshots.
    pub subject: String,
    /// An opaque caller reference (a ticket id, say), shown in snapshots.
    pub reference: Option<String>,
    /// How many times the call's task has rerun; anything above 0 is urgent.
    pub reruns: u32,
    /// The call's task as the [`DispatchGate`] knows it: a waiting call whose
    /// task went stale leaves the queue instead of being sent.
    pub task: Option<u64>,
    /// Which backends it may run on, as the bit set
    /// [`SchedulerPool::insert`] assigns: bit `i` is the `i`-th scheduler the
    /// pool holds. A dispatcher stamps its own list, so a rebalance never
    /// hands a call to a backend the caller excluded; [`u64::MAX`] means
    /// anywhere, which is what a call made straight on a [`Scheduler`] gets.
    pub allowed: u64,
}

impl CallMeta {
    /// A first attempt in the shared lane with no reference and no task,
    /// allowed on any backend.
    #[must_use]
    pub fn new(
        kind: impl Into<Cow<'static, str>>,
        priority: Priority,
        subject: impl Into<String>,
    ) -> Self {
        Self {
            kind: kind.into(),
            priority,
            lane: Lane::Shared,
            subject: subject.into(),
            reference: None,
            reruns: 0,
            task: None,
            allowed: u64::MAX,
        }
    }

    /// The same call in `lane`.
    #[must_use]
    pub const fn with_lane(mut self, lane: Lane) -> Self {
        self.lane = lane;
        self
    }

    /// The same call with an opaque caller reference.
    #[must_use]
    pub fn with_reference(mut self, reference: impl Into<String>) -> Self {
        self.reference = Some(reference.into());
        self
    }

    /// The same call tied to `task` in the [`DispatchGate`].
    #[must_use]
    pub const fn with_task(mut self, task: u64) -> Self {
        self.task = Some(task);
        self
    }

    /// The same call marked as the `reruns`-th rerun of its task.
    #[must_use]
    pub const fn with_reruns(mut self, reruns: u32) -> Self {
        self.reruns = reruns;
        self
    }

    /// The priority it waits at: [`Priority::Urgent`] for a rerun, its own
    /// priority otherwise.
    #[must_use]
    pub const fn effective_priority(&self) -> Priority {
        if self.reruns > 0 {
            Priority::Urgent
        } else {
            self.priority
        }
    }
}

/// The caller's say over dispatch: a pause, and which tasks went stale.
///
/// Every scheduler of a pool shares one gate and waits on its
/// [`wake`](Self::wake) notifier, so a caller that changes the gate's state
/// calls `wake().notify_waiters()` to make the queues look again.
pub trait DispatchGate: Send + Sync + std::fmt::Debug {
    /// Whether only [`Priority::Urgent`] calls are dispatched; calls already
    /// running finish.
    fn paused(&self) -> bool;

    /// Whether `task` went stale: a waiting call of it is withdrawn and its
    /// caller gets [`SchedulerError::InputsChanged`].
    fn is_stale(&self, task: u64) -> bool;

    /// The notifier every scheduler of the pool waits on.
    fn wake(&self) -> &Arc<Notify>;
}

/// A [`DispatchGate`] that is never paused and never stale.
#[derive(Debug, Default)]
pub struct OpenGate {
    wake: Arc<Notify>,
}

impl DispatchGate for OpenGate {
    fn paused(&self) -> bool {
        false
    }

    fn is_stale(&self, _task: u64) -> bool {
        false
    }

    fn wake(&self) -> &Arc<Notify> {
        &self.wake
    }
}

/// How one [`Scheduler`] queues.
///
/// Built by [`SchedulerConfig::new`] and checked by [`Self::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// The most calls the backend runs at once; 1 to 64. The tuner starts at
    /// 1 and never exceeds it.
    pub ceiling: u32,
    /// The most calls that may wait at the scheduler; at the cap
    /// [`Scheduler::enqueue`] refuses with [`SchedulerError::QueueFull`].
    /// Default 256.
    pub max_waiting: usize,
    /// The parallelism level from which bucket 0 serves the reserved lane
    /// (see [`Lane::Reserved`]). Default 4.
    pub reserved_lane_from_level: u32,
}

impl SchedulerConfig {
    /// A config with `ceiling` and the defaults (`max_waiting` 256,
    /// `reserved_lane_from_level` 4).
    ///
    /// # Errors
    ///
    /// [`SchedulerError::Config`] when `ceiling` is 0 or above 64.
    pub fn new(ceiling: u32) -> Result<Self, SchedulerError> {
        Self {
            ceiling,
            max_waiting: MAX_WAITING,
            reserved_lane_from_level: RESERVED_LANE_FROM_LEVEL,
        }
        .validated()
    }

    /// The same config with `max_waiting` calls allowed to wait.
    ///
    /// # Errors
    ///
    /// [`SchedulerError::Config`] when `max_waiting` is 0.
    pub fn with_max_waiting(mut self, max_waiting: usize) -> Result<Self, SchedulerError> {
        self.max_waiting = max_waiting;
        self.validated()
    }

    /// Checks every field.
    ///
    /// # Errors
    ///
    /// [`SchedulerError::Config`] when `ceiling` is 0 or above 64,
    /// `max_waiting` is 0, or `reserved_lane_from_level` is 0.
    pub fn validate(&self) -> Result<(), SchedulerError> {
        if self.ceiling == 0 || self.ceiling > MAX_CEILING {
            return Err(SchedulerError::Config(format!(
                "ceiling must be 1 to {MAX_CEILING}, got {}",
                self.ceiling
            )));
        }
        if self.max_waiting == 0 {
            return Err(SchedulerError::Config(
                "max_waiting must be at least 1".into(),
            ));
        }
        if self.reserved_lane_from_level == 0 {
            return Err(SchedulerError::Config(
                "reserved_lane_from_level must be at least 1".into(),
            ));
        }
        Ok(())
    }

    fn validated(self) -> Result<Self, SchedulerError> {
        self.validate()?;
        Ok(self)
    }
}

/// Why a scheduler call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    /// A [`SchedulerConfig`] field is out of range.
    Config(String),
    /// The backend already has `max_waiting` calls waiting; the call was not
    /// queued.
    QueueFull {
        /// The backend that refused.
        backend: BackendId,
        /// Its [`SchedulerConfig::max_waiting`].
        max_waiting: usize,
    },
    /// The call's task went stale while it waited, so it was withdrawn
    /// unsent; the caller reruns.
    InputsChanged,
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(message) => write!(f, "scheduler config: {message}"),
            Self::QueueFull {
                backend,
                max_waiting,
            } => write!(
                f,
                "backend {} already has {max_waiting} calls waiting",
                backend.as_str()
            ),
            Self::InputsChanged => f.write_str("the call's inputs changed while it waited"),
        }
    }
}

impl std::error::Error for SchedulerError {}

/// How a call ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CallOutcome {
    /// The backend answered with text.
    Ok,
    /// A timeout, an unreachable backend, an error status, or an empty
    /// reply: a trial in progress is reverted.
    Failed,
    /// The caller went away before reporting.
    Cancelled,
}

/// What one completed call cost, as the backend reported it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallReport {
    /// How it ended.
    pub outcome: CallOutcome,
    /// Prompt tokens, when the backend reported them; `None` is unknown,
    /// never an estimate.
    pub prompt_tokens: Option<u64>,
    /// Completion tokens, likewise.
    pub completion_tokens: Option<u64>,
    /// Characters streamed back (answer and reasoning), as measured.
    pub output_chars: u64,
}

impl CallReport {
    /// A call that ended without output.
    #[must_use]
    pub const fn empty(outcome: CallOutcome) -> Self {
        Self {
            outcome,
            prompt_tokens: None,
            completion_tokens: None,
            output_chars: 0,
        }
    }
}

/// A backend's identity: two base URLs with the same identity share one
/// scheduler.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct BackendId(String);

impl BackendId {
    /// Normalises a base URL: scheme and host lowercased, the scheme's
    /// default port filled in, the path without a trailing `/`, and every
    /// loopback spelling (`localhost`, any `127.x.x.x`, `::1`, `[::1]`) one
    /// host.
    #[must_use]
    pub fn parse(base_url: &str) -> Self {
        let trimmed = base_url.trim();
        let Some((scheme, rest)) = trimmed.split_once("://") else {
            return Self(trimmed.trim_end_matches('/').to_ascii_lowercase());
        };
        let scheme = scheme.to_ascii_lowercase();
        let (authority, path) = rest.find('/').map_or((rest, ""), |at| rest.split_at(at));
        let authority = authority
            .rsplit_once('@')
            .map_or(authority, |(_, host)| host);
        let (host, port) = split_host_port(authority);
        let host = canonical_host(host);
        let port = port.or(match scheme.as_str() {
            "http" => Some(80),
            "https" => Some(443),
            _ => None,
        });
        let path = path.trim_end_matches('/');
        Self(match port {
            Some(port) => format!("{scheme}://{host}:{port}{path}"),
            None => format!("{scheme}://{host}{path}"),
        })
    }

    /// The normalised identity.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Splits `host[:port]`, `[v6][:port]` or a bare IPv6 literal.
fn split_host_port(authority: &str) -> (&str, Option<u16>) {
    if let Some(bracketed) = authority.strip_prefix('[') {
        return match bracketed.split_once(']') {
            Some((host, after)) => (
                host,
                after.strip_prefix(':').and_then(|port| port.parse().ok()),
            ),
            None => (authority, None),
        };
    }
    match authority.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') => match port.parse() {
            Ok(port) => (host, Some(port)),
            Err(_) => (authority, None),
        },
        _ => (authority, None),
    }
}

fn canonical_host(host: &str) -> String {
    let host = host.to_ascii_lowercase();
    if host == "localhost" || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback()) {
        "loopback".to_owned()
    } else {
        host
    }
}

/// The next waiting call's id.
///
/// CONTEXT: one counter for the process, not one per scheduler, because a
/// rebalance moves a call between schedulers and the future that queued it
/// finds it again by this id. Ids still increase with arrival, so the lower
/// id is the earlier one.
static NEXT_WAITER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct Waiter {
    id: u64,
    meta: CallMeta,
    value: u64,
    arrived: Instant,
}

#[derive(Debug)]
struct Runner {
    id: u64,
    meta: CallMeta,
    value: u64,
    waited: Duration,
    started: Instant,
    bucket: u32,
    level: u32,
}

/// The waiting and running calls, and the dispatch rules over them.
#[derive(Debug, Default)]
struct Queue {
    waiting: Vec<Waiter>,
    running: Vec<Runner>,
}

impl Queue {
    fn push(&mut self, meta: CallMeta, now: Instant) -> u64 {
        let id = NEXT_WAITER.fetch_add(1, Ordering::Relaxed);
        self.waiting.push(Waiter {
            id,
            value: meta.effective_priority().initial_value(),
            meta,
            arrived: now,
        });
        id
    }

    /// Whether `id` is waiting here (rather than running, or elsewhere).
    fn is_waiting(&self, id: u64) -> bool {
        self.waiting.iter().any(|waiter| waiter.id == id)
    }

    /// The best eligible waiter: biggest value, earliest arrival.
    fn best(&self, eligible: impl Fn(&Waiter) -> bool) -> Option<usize> {
        self.waiting
            .iter()
            .enumerate()
            .filter(|(_, waiter)| eligible(waiter))
            .max_by(|(_, a), (_, b)| a.value.cmp(&b.value).then(b.id.cmp(&a.id)))
            .map(|(index, _)| index)
    }

    fn bucket_free(&self, bucket: u32) -> bool {
        !self.running.iter().any(|runner| runner.bucket == bucket)
    }

    /// The next waiter to start and the bucket it starts in.
    fn next_placement(
        &self,
        level: u32,
        reserved_from: u32,
        eligible: &impl Fn(&Waiter) -> bool,
    ) -> Option<(usize, u32)> {
        if level < reserved_from {
            let bucket = (0..level).find(|bucket| self.bucket_free(*bucket))?;
            return self.best(eligible).map(|index| (index, bucket));
        }
        let reserved_free = self.bucket_free(0);
        if reserved_free
            && let Some(index) =
                self.best(|waiter| waiter.meta.lane == Lane::Reserved && eligible(waiter))
        {
            return Some((index, 0));
        }
        let index = self.best(|waiter| waiter.meta.lane != Lane::Reserved && eligible(waiter))?;
        let bucket = (1..level)
            .find(|bucket| self.bucket_free(*bucket))
            .or_else(|| reserved_free.then_some(0))?;
        Some((index, bucket))
    }

    /// Starts as many waiters as `level` allows; returns their ids.
    fn dispatch(
        &mut self,
        level: u32,
        reserved_from: u32,
        now: Instant,
        eligible: impl Fn(&Waiter) -> bool,
    ) -> Vec<u64> {
        let mut started = Vec::new();
        while u32::try_from(self.running.len()).unwrap_or(u32::MAX) < level {
            let Some((index, bucket)) = self.next_placement(level, reserved_from, &eligible) else {
                break;
            };
            let waiter = self.waiting.remove(index);
            started.push(waiter.id);
            self.running.push(Runner {
                id: waiter.id,
                waited: now.saturating_duration_since(waiter.arrived),
                meta: waiter.meta,
                value: waiter.value,
                started: now,
                bucket,
                level,
            });
        }
        started
    }

    /// Ends a running call and ages every waiter by one.
    fn complete(&mut self, id: u64) -> Option<Runner> {
        let position = self.running.iter().position(|runner| runner.id == id)?;
        let runner = self.running.remove(position);
        for waiter in &mut self.waiting {
            waiter.value += 1;
        }
        Some(runner)
    }

    fn withdraw(&mut self, id: u64) -> bool {
        let before = self.waiting.len();
        self.waiting.retain(|waiter| waiter.id != id);
        self.waiting.len() != before
    }

    fn is_running(&self, id: u64) -> bool {
        self.running.iter().any(|runner| runner.id == id)
    }

    /// The level a running call was dispatched under; `None` if it is not
    /// running.
    fn running_level(&self, id: u64) -> Option<u32> {
        self.running
            .iter()
            .find(|runner| runner.id == id)
            .map(|runner| runner.level)
    }
}

/// The schedulers of one pool of backends.
///
/// A call queues in one of them and keeps its identity — a process-wide call
/// id — wherever it waits, so a rebalance can hand it to a backend that
/// would start it sooner and the future that queued it still finds it. The
/// pool is what makes that search possible; nothing else lives here.
#[derive(Debug, Default)]
pub struct SchedulerPool {
    members: Mutex<Vec<Weak<Scheduler>>>,
    /// Serializes placement. A caller holds this while it selects a
    /// scheduler from a load read and enqueues on it, so concurrent
    /// placements (across dispatchers, and the many calls of one burst) cannot
    /// all read the same idle load and pile onto a single backend. The
    /// members lock is separate: [`Self::place`] and [`Self::abandon`] take
    /// it only long enough to snapshot the members.
    place: Mutex<()>,
}

/// Where one scheduler holds a call, as [`Scheduler::place`] sees it.
#[derive(Debug)]
enum MemberPlacement {
    /// Running there, in this bucket.
    Running(u32),
    /// Waiting there, with no bucket free for it yet.
    Waiting,
    /// Waiting there, and withdrawn because its task went stale.
    Stale,
}

/// Where the pool holds a call, as [`SchedulerPool::place`] sees it.
#[derive(Debug)]
enum Placement {
    /// Running on this member, in this bucket.
    Running { home: Arc<Scheduler>, level: u32 },
    /// Waiting on a member.
    Waiting,
    /// Withdrawn: its task went stale while it waited.
    Stale,
    /// In no member's queue. Only a rebalance's own instant between taking a
    /// call from one member and pushing it onto another; the call's future
    /// simply keeps waiting.
    Gone,
}

impl SchedulerPool {
    /// An empty pool; every scheduler is added by [`Self::insert`] as its
    /// backend is built.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The placement guard: hold it while selecting a scheduler and
    /// enqueueing on it, so placements serialize.
    pub(crate) fn lock_placement(&self) -> MutexGuard<'_, ()> {
        self.place.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Registers `scheduler`, answering the bit of [`CallMeta::allowed`] that
    /// names it; `None` once a pool has more than 64 backends, which then
    /// can still give calls away but never take any.
    pub fn insert(&self, scheduler: &Arc<Scheduler>) -> Option<u32> {
        let mut members = self.lock();
        let index = u32::try_from(members.len()).unwrap_or(u32::MAX);
        members.push(Arc::downgrade(scheduler));
        (index < 64).then_some(index)
    }

    /// The live members, in the order their backends were built.
    fn members(&self) -> Vec<Arc<Scheduler>> {
        self.lock().iter().filter_map(Weak::upgrade).collect()
    }

    /// Where `id` stands right now.
    fn place(&self, id: u64) -> Placement {
        for member in self.members() {
            match member.place(id) {
                Some(MemberPlacement::Running(level)) => {
                    return Placement::Running {
                        home: member,
                        level,
                    };
                }
                Some(MemberPlacement::Waiting) => return Placement::Waiting,
                Some(MemberPlacement::Stale) => return Placement::Stale,
                None => {}
            }
        }
        Placement::Gone
    }

    /// Takes `id` out of whichever member holds it: unsent if it was waiting,
    /// cancelled if it had started. Answers whether any member held it.
    fn abandon(&self, id: u64) -> bool {
        self.members()
            .iter()
            .any(|member| member.abandon(id).is_some())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Weak<Scheduler>>> {
        // CONTEXT: the guarded value is a list of handles, never left
        // half-written, so a poisoned lock still holds a usable one.
        self.members.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Hands one waiting call of `source` to `target`.
///
/// The call taken is the one that has waited longest among those whose
/// [`CallMeta::allowed`] admits `target` (bit `target_bit`) and whose task has
/// not gone stale. A call that has started never moves: only a call that has
/// not been sent can be sent somewhere else. Answers whether it moved one.
///
/// CONTEXT: the clock beat's rebalance drives this one call at a time, so the
/// two locks below are only ever taken source-then-target, by one task; a
/// waiting call's own future holds at most one scheduler's lock at a time.
pub(crate) fn move_waiting(source: &Scheduler, target: &Scheduler, target_bit: u64) -> bool {
    let now = target.clock.now();
    let mut from = source.lock();
    let Some(position) = from.queue.best(|waiter| {
        waiter.meta.allowed & target_bit != 0
            && !waiter
                .meta
                .task
                .is_some_and(|task| source.gate.is_stale(task))
    }) else {
        return false;
    };
    let waiter = from.queue.waiting.remove(position);
    let mut to = target.lock();
    to.queue.waiting.push(waiter);
    target.step(&mut to, now);
    drop(to);
    drop(from);
    target.wake.notify_waiters();
    source.changed();
    target.changed();
    true
}

/// Whether moving one waiting call from `source` to `target` would lower the
/// longest estimated start among `loads`.
///
/// CONTEXT: min-max, not "toward equal counts". Two backends of close but
/// unequal pace have no move that equalises them — the last call would come
/// straight back — so a rebalance that only compares counts oscillates for as
/// long as it is given budget, moving calls between servers every interval
/// forever. Accepting a move only when it lowers the worst wait makes the
/// rebalance settle: the worst wait is a whole number that strictly decreases
/// on every move, so there are only finitely many. The one exception, a move
/// onto an idle backend, dispatches the call there at once, so it is no
/// longer waiting and cannot be moved back.
pub(crate) fn lowers_the_worst(loads: &[Load], source: usize, target: usize) -> bool {
    // An idle target starts the moved call immediately, so it always takes
    // one from a busier backend. The assumed pace of a backend that has never
    // answered must not make it look slower than the queue it is emptying:
    // without this, the first server to answer fast is judged cheaper than an
    // idle one and rebalance piles everything onto it.
    if loads[target].calls == 0 {
        return loads[source].calls > 0;
    }
    let start = |load: &Load, calls: usize| -> u128 {
        u128::try_from(calls).unwrap_or(u128::MAX) * u128::from(load.pace_ms)
            / u128::from(load.level.max(1))
    };
    let worst_now = loads.iter().map(|load| load.start_ms()).max().unwrap_or(0);
    let worst_after = loads
        .iter()
        .enumerate()
        .map(|(index, load)| {
            let calls = if index == source {
                load.calls.saturating_sub(1)
            } else if index == target {
                load.calls.saturating_add(1)
            } else {
                load.calls
            };
            start(load, calls)
        })
        .max()
        .unwrap_or(0);
    worst_after < worst_now
}

/// A call that is running, as the dashboard shows it.
#[derive(Debug, Clone, Serialize)]
pub struct RunningCall {
    /// What it is for.
    pub kind: String,
    /// Who it is about.
    pub subject: String,
    /// Its caller reference.
    pub reference: Option<String>,
    /// The priority it waited at.
    pub priority: Priority,
    /// Its value when it was dispatched.
    pub value: u64,
    /// How long it waited, in milliseconds.
    pub waited_ms: u64,
    /// How long it has been running, in milliseconds.
    pub elapsed_ms: u64,
    /// How many times its task has rerun.
    pub reruns: u32,
}

/// One bucket (slot) of the current level.
#[derive(Debug, Clone, Serialize)]
pub struct BucketSnapshot {
    /// Its index; 0 is the reserved bucket from the reserved-lane level up.
    pub index: u32,
    /// `shared`, `reserved_first` or `shared_only`.
    pub serves: &'static str,
    /// The call it is running, if any.
    pub call: Option<RunningCall>,
}

/// A call waiting for a bucket.
#[derive(Debug, Clone, Serialize)]
pub struct WaitingCall {
    /// What it is for.
    pub kind: String,
    /// Who it is about.
    pub subject: String,
    /// Its caller reference.
    pub reference: Option<String>,
    /// The priority it entered at.
    pub priority: Priority,
    /// Its current value, aged by every completion since it arrived.
    pub value: u64,
    /// How long it has waited, in milliseconds.
    pub waited_ms: u64,
    /// How many times its task has rerun.
    pub reruns: u32,
}

/// One completed call.
#[derive(Debug, Clone, Serialize)]
pub struct CompletedCall {
    /// What it was for.
    pub kind: String,
    /// Who it was about.
    pub subject: String,
    /// Its caller reference.
    pub reference: Option<String>,
    /// The priority it waited at.
    pub priority: Priority,
    /// The parallelism level it was dispatched under.
    pub level: u32,
    /// The bucket it ran in.
    pub bucket: u32,
    /// How long it waited, in milliseconds.
    pub waited_ms: u64,
    /// How long it ran, in milliseconds.
    pub run_ms: u64,
    /// Prompt tokens, when the backend reported them.
    pub prompt_tokens: Option<u64>,
    /// Completion tokens, when the backend reported them.
    pub completion_tokens: Option<u64>,
    /// Characters streamed back.
    pub output_chars: u64,
    /// How it ended.
    pub outcome: CallOutcome,
    /// How many times its task had rerun.
    pub reruns: u32,
    /// Milliseconds since the scheduler started, at completion.
    pub completed_at_ms: u64,
}

/// What one kind of call has cost since boot.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct KindTotals {
    /// Completed calls.
    pub calls: u64,
    /// Of those, failed.
    pub failed: u64,
    /// Prompt tokens over the calls that reported them.
    pub prompt_tokens: u64,
    /// Completion tokens over the calls that reported them.
    pub completion_tokens: u64,
    /// Calls that reported no token usage.
    pub calls_without_usage: u64,
    /// Characters streamed back.
    pub output_chars: u64,
    /// Time spent running, in milliseconds.
    pub run_ms: u64,
    /// Time spent waiting, in milliseconds.
    pub waited_ms: u64,
}

/// One kind's totals, named.
#[derive(Debug, Clone, Serialize)]
pub struct KindTotalsEntry {
    /// The kind.
    pub kind: String,
    /// Its totals.
    #[serde(flatten)]
    pub totals: KindTotals,
}

/// Everything one scheduler is doing, for `GET /api/llm/scheduler` and the
/// `scheduler` SSE frame.
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerSnapshot {
    /// The normalised backend identity.
    pub backend: BackendId,
    /// The configured maximum parallel calls.
    pub ceiling: u32,
    /// The parallelism level in force.
    pub level: u32,
    /// The mean duration of the calls this backend has finished, in
    /// milliseconds: what a pool weighs its queue by.
    pub pace_ms: u64,
    /// Milliseconds since the scheduler started.
    pub uptime_ms: u64,
    /// One entry per bucket of the level (plus any a lowered level left
    /// running).
    pub buckets: Vec<BucketSnapshot>,
    /// Waiting calls, next to be dispatched first.
    pub waiting: Vec<WaitingCall>,
    /// Whether only urgent calls are being dispatched, because the
    /// [`DispatchGate`] is paused.
    pub paused: bool,
    /// The parallelism tuner.
    pub tuner: TunerSnapshot,
    /// The most recent completed calls, newest first.
    pub recent: Vec<CompletedCall>,
    /// What each kind has cost since boot.
    pub totals: Vec<KindTotalsEntry>,
}

#[derive(Debug)]
struct State {
    queue: Queue,
    tuner: Tuner,
    recent: VecDeque<CompletedCall>,
    totals: Vec<(Cow<'static, str>, KindTotals)>,
    /// The mean duration of the calls this backend has finished, which the
    /// pool weighs its queue by.
    pace: Pace,
    reserved_hold_until: Option<Instant>,
}

impl State {
    /// The totals `kind` is counted under: its own entry, or past
    /// [`MAX_KINDS`] distinct kinds the entry of [`OTHER_KIND`].
    fn totals_for(&mut self, kind: &str) -> &mut KindTotals {
        let at = match self.totals.iter().position(|(seen, _)| &**seen == kind) {
            Some(at) => at,
            None if self.totals.len() < MAX_KINDS => {
                self.totals
                    .push((Cow::Owned(kind.to_owned()), KindTotals::default()));
                self.totals.len() - 1
            }
            None => match self
                .totals
                .iter()
                .position(|(seen, _)| &**seen == OTHER_KIND)
            {
                Some(at) => at,
                None => {
                    self.totals
                        .push((Cow::Borrowed(OTHER_KIND), KindTotals::default()));
                    self.totals.len() - 1
                }
            },
        };
        &mut self.totals[at].1
    }
}

/// The scheduler of one backend.
#[derive(Debug)]
pub struct Scheduler {
    backend: BackendId,
    config: SchedulerConfig,
    clock: Arc<dyn Clock>,
    started: Instant,
    gate: Arc<dyn DispatchGate>,
    /// The schedulers this one shares a pool with, so a call the pool hands
    /// to another backend is still found by the future that queued it.
    pool: Arc<SchedulerPool>,
    wake: Arc<Notify>,
    changes: Arc<watch::Sender<u64>>,
    state: Mutex<State>,
}

/// A scheduler's load: what a pool of backends compares to pick one.
///
/// CONTEXT: the queue is weighed by the backend's own pace, not by its count
/// alone. Two servers of one pool are rarely equal — a second endpoint is
/// usually a different machine — and counting calls alone hands a server that
/// takes minutes per call the same share as one that takes a second, so the
/// queue waits on the slow one while the fast one sits idle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Load {
    /// Calls waiting or running.
    pub calls: usize,
    /// Of those, the ones waiting for a bucket: what a rebalance can move.
    pub waiting: usize,
    /// The parallelism level in force (at least 1).
    pub level: u32,
    /// The mean duration of the calls this backend has finished, in
    /// milliseconds (2000 before it has finished one).
    pub pace_ms: u64,
}

impl Load {
    /// How long a call queued now waits before it starts: the calls ahead of
    /// it at this backend's measured pace, spread over the calls it runs at
    /// once.
    #[must_use]
    pub fn start_ms(self) -> u128 {
        u128::try_from(self.calls).unwrap_or(u128::MAX) * u128::from(self.pace_ms)
            / u128::from(self.level.max(1))
    }

    /// Whether a call queued now would start sooner here than on `other`,
    /// compared by cross-multiplying so no division rounds a tie away.
    #[must_use]
    pub fn is_lighter_than(self, other: Self) -> bool {
        // An idle backend with no calls in flight is always lighter than a busy
        // one with calls already queued or running, even if the busy one's
        // measured pace is lower than the default assumed pace.
        if self.calls == 0 && other.calls > 0 {
            return true;
        }
        if self.calls > 0 && other.calls == 0 {
            return false;
        }
        let weigh = |calls: usize, pace_ms: u64, level: u32| {
            u128::try_from(calls).unwrap_or(u128::MAX)
                * u128::from(pace_ms)
                * u128::from(level.max(1))
        };
        weigh(self.calls, self.pace_ms, other.level) < weigh(other.calls, other.pace_ms, self.level)
    }
}

/// The mean call duration a backend is assumed to have before it has finished
/// one, in milliseconds.
///
/// CONTEXT: a pool picks its backend before any of them has answered
/// anything, so *some* figure is needed then. It never decides much: every
/// completion replaces it with the measured pace.
const ASSUMED_CALL_MS: u64 = 2_000;

/// How fast a backend's measured pace forgets, as a power of two: one part in
/// `1 << PACE_SHIFT` of the gap to the call just finished.
///
/// CONTEXT: a mean over every call ever made would take a hundred calls to
/// notice a server that has just started stalling, and the first burst after
/// it does is exactly when that matters: at a window of eight, one
/// 140-second call moves a two-second pace to nineteen.
const PACE_SHIFT: u32 = 3;

/// How long a backend's calls take, as a mean of the ones it has finished.
#[derive(Debug, Clone, Copy)]
struct Pace {
    mean_ms: u64,
}

impl Pace {
    /// The assumed pace, before the backend has finished a call.
    const fn new() -> Self {
        Self {
            mean_ms: ASSUMED_CALL_MS,
        }
    }

    /// The current mean, in milliseconds.
    const fn mean_ms(&self) -> u64 {
        self.mean_ms
    }

    /// Folds one finished call into the mean, rounded to the nearest
    /// millisecond so the mean converges instead of freezing a few
    /// milliseconds away from it.
    fn record(&mut self, run_ms: u64) {
        let gap = i64::try_from(run_ms).unwrap_or(i64::MAX)
            - i64::try_from(self.mean_ms).unwrap_or(i64::MAX);
        let step = (gap + (1 << (PACE_SHIFT - 1))) >> PACE_SHIFT;
        let next = i64::try_from(self.mean_ms).unwrap_or(i64::MAX) + step;
        self.mean_ms = u64::try_from(next).unwrap_or(1).max(1);
    }
}

impl Scheduler {
    /// A scheduler for `backend` under `config`, starting at one call at a
    /// time, a member of `pool` once [`SchedulerPool::insert`] adds it.
    ///
    /// `config` is taken as built: [`SchedulerConfig::new`] already validated
    /// it, and a hand-edited one is checked by [`SchedulerConfig::validate`].
    #[must_use]
    pub fn new(
        backend: BackendId,
        config: SchedulerConfig,
        clock: Arc<dyn Clock>,
        gate: Arc<dyn DispatchGate>,
        changes: Arc<watch::Sender<u64>>,
        pool: Arc<SchedulerPool>,
    ) -> Self {
        let started = clock.now();
        Self {
            backend,
            started,
            wake: Arc::clone(gate.wake()),
            gate,
            pool,
            changes,
            state: Mutex::new(State {
                queue: Queue::default(),
                tuner: Tuner::new(config.ceiling, started),
                recent: VecDeque::with_capacity(RECENT_CALLS),
                totals: Vec::new(),
                pace: Pace::new(),
                reserved_hold_until: None,
            }),
            config,
            clock,
        }
    }

    /// The backend this scheduler serves.
    #[must_use]
    pub const fn backend(&self) -> &BackendId {
        &self.backend
    }

    /// The configured maximum parallel calls.
    #[must_use]
    pub const fn ceiling(&self) -> u32 {
        self.config.ceiling
    }

    /// How busy this backend is: its waiting and running calls, the level in
    /// force and how long its calls take, read under the lock without
    /// building a snapshot.
    #[must_use]
    pub fn load(&self) -> Load {
        let state = self.lock();
        let waiting = state.queue.waiting.len();
        Load {
            calls: waiting + state.queue.running.len(),
            waiting,
            level: state.tuner.level(),
            pace_ms: state.pace.mean_ms(),
        }
    }

    /// Where `id` stands in this scheduler, under one lock: running with its
    /// bucket, waiting (dispatched here if a bucket is free), or withdrawn
    /// because its task went stale. `None` when this scheduler does not hold
    /// it at all.
    ///
    /// CONTEXT: checking running, waiting and staleness under the *same* lock
    /// is what keeps a dispatch from racing a withdrawal: a call this
    /// scheduler has started cannot be taken out from under its own future.
    fn place(&self, id: u64) -> Option<MemberPlacement> {
        let now = self.clock.now();
        let mut state = self.lock();
        if let Some(level) = state.queue.running_level(id) {
            return Some(MemberPlacement::Running(level));
        }
        if !state.queue.is_waiting(id) {
            return None;
        }
        let stale = state
            .queue
            .waiting
            .iter()
            .find(|waiter| waiter.id == id)
            .is_some_and(|waiter| {
                waiter
                    .meta
                    .task
                    .is_some_and(|task| self.gate.is_stale(task))
            });
        if stale {
            state.queue.withdraw(id);
            return Some(MemberPlacement::Stale);
        }
        self.step(&mut state, now);
        Some(match state.queue.running_level(id) {
            Some(level) => MemberPlacement::Running(level),
            None => MemberPlacement::Waiting,
        })
    }

    /// Takes `id` out of this scheduler: unsent if it was waiting, cancelled
    /// if it had started. `None` when it was not here.
    fn abandon(&self, id: u64) -> Option<bool> {
        let running = {
            let mut state = self.lock();
            if state.queue.withdraw(id) {
                false
            } else if state.queue.is_running(id) {
                true
            } else {
                return None;
            }
        };
        if running {
            self.finish(id, CallReport::empty(CallOutcome::Cancelled));
        }
        Some(running)
    }

    /// Waits for a bucket for one call, in this scheduler or in whichever
    /// member of its pool a rebalance hands it to.
    ///
    /// # Errors
    ///
    /// [`SchedulerError::QueueFull`] when the queue is at `max_waiting`;
    /// [`SchedulerError::InputsChanged`] when the call's task went stale
    /// while it waited: it is withdrawn unsent, and the caller reruns.
    pub async fn acquire(self: &Arc<Self>, meta: CallMeta) -> Result<CallSlot, SchedulerError> {
        let id = self.enqueue(meta)?;
        self.await_placement(id).await
    }

    /// Pushes `meta` onto this scheduler's queue and dispatches it if a
    /// bucket is free, answering its call id. The push and dispatch are one
    /// critical section, so a caller that selected this scheduler from a load
    /// read taken under the pool's placement lock enqueues atomically with
    /// respect to that selection.
    ///
    /// # Errors
    ///
    /// [`SchedulerError::QueueFull`] when `max_waiting` calls already wait
    /// here; the call is not queued.
    pub fn enqueue(&self, meta: CallMeta) -> Result<u64, SchedulerError> {
        let id = {
            let mut state = self.lock();
            if state.queue.waiting.len() >= self.config.max_waiting {
                return Err(SchedulerError::QueueFull {
                    backend: self.backend.clone(),
                    max_waiting: self.config.max_waiting,
                });
            }
            let now = self.clock.now();
            let id = state.queue.push(meta, now);
            self.step(&mut state, now);
            id
        };
        self.changed();
        Ok(id)
    }

    /// Waits for call `id` to be dispatched, answering its slot.
    ///
    /// # Errors
    ///
    /// [`SchedulerError::InputsChanged`] when the call's task went stale
    /// while it waited.
    pub async fn await_placement(self: &Arc<Self>, id: u64) -> Result<CallSlot, SchedulerError> {
        let pool = Arc::clone(&self.pool);
        let mut pending = PendingCall {
            scheduler: Arc::clone(self),
            pool: Arc::clone(&pool),
            changes: Arc::clone(&self.changes),
            id,
            armed: true,
        };
        loop {
            let notified = self.wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            // CONTEXT: `self` first, so a scheduler that was never inserted
            // into a pool (a test that builds one alone) still dispatches its
            // own calls, and so the common case — the call never moved —
            // costs one lock instead of a scan of the pool.
            let placement = match self.place(id) {
                Some(MemberPlacement::Running(level)) => Placement::Running {
                    home: Arc::clone(self),
                    level,
                },
                Some(MemberPlacement::Waiting) => Placement::Waiting,
                Some(MemberPlacement::Stale) => Placement::Stale,
                None => pool.place(id),
            };
            match placement {
                Placement::Running { home, level } => {
                    pending.armed = false;
                    return Ok(CallSlot {
                        scheduler: home,
                        id,
                        level,
                        done: false,
                    });
                }
                Placement::Stale => {
                    pending.armed = false;
                    self.changed();
                    return Err(SchedulerError::InputsChanged);
                }
                // Waiting somewhere the pool knows about, or a rebalance's
                // instant between two members: either way, keep waiting.
                Placement::Waiting | Placement::Gone => {}
            }
            notified.await;
        }
    }

    /// Everything this scheduler is doing.
    #[must_use]
    pub fn snapshot(&self) -> SchedulerSnapshot {
        let now = self.clock.now();
        let paused = self.gate.paused();
        let state = self.lock();
        let level = state.tuner.level();
        let millis = |duration: Duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
        let shown = state
            .queue
            .running
            .iter()
            .map(|runner| runner.bucket + 1)
            .max()
            .unwrap_or(0)
            .max(level);
        let buckets = (0..shown)
            .map(|index| BucketSnapshot {
                index,
                serves: if level < self.config.reserved_lane_from_level {
                    "shared"
                } else if index == 0 {
                    "reserved_first"
                } else {
                    "shared_only"
                },
                call: state
                    .queue
                    .running
                    .iter()
                    .find(|runner| runner.bucket == index)
                    .map(|runner| RunningCall {
                        kind: runner.meta.kind.to_string(),
                        subject: runner.meta.subject.clone(),
                        reference: runner.meta.reference.clone(),
                        priority: runner.meta.effective_priority(),
                        value: runner.value,
                        waited_ms: millis(runner.waited),
                        elapsed_ms: millis(now.saturating_duration_since(runner.started)),
                        reruns: runner.meta.reruns,
                    }),
            })
            .collect();
        let mut waiting: Vec<&Waiter> = state.queue.waiting.iter().collect();
        waiting.sort_by(|a, b| b.value.cmp(&a.value).then(a.id.cmp(&b.id)));
        SchedulerSnapshot {
            backend: self.backend.clone(),
            ceiling: self.config.ceiling,
            level,
            pace_ms: state.pace.mean_ms(),
            uptime_ms: millis(now.saturating_duration_since(self.started)),
            buckets,
            waiting: waiting
                .into_iter()
                .map(|waiter| WaitingCall {
                    kind: waiter.meta.kind.to_string(),
                    subject: waiter.meta.subject.clone(),
                    reference: waiter.meta.reference.clone(),
                    priority: waiter.meta.effective_priority(),
                    value: waiter.value,
                    waited_ms: millis(now.saturating_duration_since(waiter.arrived)),
                    reruns: waiter.meta.reruns,
                })
                .collect(),
            paused,
            tuner: state.tuner.snapshot(now),
            recent: state.recent.iter().rev().cloned().collect(),
            totals: state
                .totals
                .iter()
                .map(|(kind, totals)| KindTotalsEntry {
                    kind: kind.to_string(),
                    totals: *totals,
                })
                .collect(),
        }
    }

    /// Starts a due experiment, then dispatches what the level allows.
    fn step(&self, state: &mut State, now: Instant) {
        let paused = self.gate.paused();
        let backlog = state.queue.waiting.len() + state.queue.running.len();
        state.tuner.poll(now, backlog, paused);
        let reserved_held = state.reserved_hold_until.is_some_and(|until| now < until);
        if state.reserved_hold_until.is_some_and(|until| now >= until) {
            state.reserved_hold_until = None;
        }
        let stale = |waiter: &Waiter| {
            waiter
                .meta
                .task
                .is_some_and(|task| self.gate.is_stale(task))
        };
        let reserved_from = self.config.reserved_lane_from_level;
        let started = state
            .queue
            .dispatch(state.tuner.level(), reserved_from, now, |waiter| {
                let urgent = waiter.meta.effective_priority() == Priority::Urgent;
                if waiter.meta.lane == Lane::Reserved && reserved_held && !urgent {
                    return false;
                }
                (!paused || urgent) && !stale(waiter)
            });
        for id in &started {
            state.tuner.on_dispatch(*id);
        }
        if !started.is_empty() {
            self.wake.notify_waiters();
        }
    }

    /// The scheduler's wake notifier: the [`DispatchGate`]'s, so an owner
    /// that changes the gate's state calls `notify_waiters()` on it.
    #[must_use]
    pub const fn wake(&self) -> &Arc<Notify> {
        &self.wake
    }

    /// Delays dispatching [`Lane::Reserved`] calls until `duration` has
    /// passed or until [`Self::clear_reserved_hold`] is called. Urgent calls
    /// are not held.
    ///
    /// CONTEXT: a caller that expects a burst of shared calls soon holds the
    /// reserved lane first, so a long reserved call does not start just
    /// before them and block the backend.
    ///
    /// Needs a running Tokio runtime: a spawned timer wakes the queues when
    /// the hold ends.
    pub fn hold_reserved_for(&self, duration: Duration) {
        let now = self.clock.now();
        let until = now + duration;
        {
            let mut state = self.lock();
            state.reserved_hold_until = Some(until);
        }
        let wake = Arc::clone(&self.wake);
        tokio::spawn(async move {
            tokio::time::sleep(duration).await;
            wake.notify_waiters();
        });
    }

    /// Releases any pending hold on the reserved lane immediately.
    pub fn clear_reserved_hold(&self) {
        let mut state = self.lock();
        if state.reserved_hold_until.take().is_some() {
            let now = self.clock.now();
            self.step(&mut state, now);
            drop(state);
            self.wake.notify_waiters();
            self.changed();
        }
    }

    fn finish(&self, id: u64, report: CallReport) {
        let now = self.clock.now();
        {
            let mut state = self.lock();
            let Some(runner) = state.queue.complete(id) else {
                return;
            };
            let millis =
                |duration: Duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
            let run = now.saturating_duration_since(runner.started);
            // CONTEXT: only a call that answered says how fast this backend
            // is; a refusal is instant and a timeout measures the deadline,
            // so neither is folded into the pace the pool weighs it by.
            if report.outcome == CallOutcome::Ok {
                state.pace.record(millis(run));
            }
            state.tuner.on_complete(Sample {
                id,
                level: runner.level,
                dispatched: runner.started,
                completed: now,
                chars: report.output_chars,
                tokens: report.completion_tokens,
                outcome: report.outcome,
            });
            let totals = state.totals_for(&runner.meta.kind);
            totals.calls += 1;
            totals.failed += u64::from(report.outcome == CallOutcome::Failed);
            totals.prompt_tokens += report.prompt_tokens.unwrap_or(0);
            totals.completion_tokens += report.completion_tokens.unwrap_or(0);
            totals.calls_without_usage +=
                u64::from(report.prompt_tokens.is_none() && report.completion_tokens.is_none());
            totals.output_chars += report.output_chars;
            totals.run_ms += millis(run);
            totals.waited_ms += millis(runner.waited);
            if state.recent.len() == RECENT_CALLS {
                state.recent.pop_front();
            }
            state.recent.push_back(CompletedCall {
                priority: runner.meta.effective_priority(),
                kind: runner.meta.kind.into_owned(),
                subject: runner.meta.subject,
                reference: runner.meta.reference,
                level: runner.level,
                bucket: runner.bucket,
                waited_ms: millis(runner.waited),
                run_ms: millis(run),
                prompt_tokens: report.prompt_tokens,
                completion_tokens: report.completion_tokens,
                output_chars: report.output_chars,
                outcome: report.outcome,
                reruns: runner.meta.reruns,
                completed_at_ms: millis(now.saturating_duration_since(self.started)),
            });
            self.step(&mut state, now);
        }
        self.wake.notify_waiters();
        self.changed();
    }

    fn changed(&self) {
        self.changes
            .send_modify(|version| *version = version.wrapping_add(1));
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        // CONTEXT: every update is whole under the lock, so a poisoned guard
        // still holds a consistent queue.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A waiting call's place in the pool; dropping it unsent withdraws it,
/// whichever member holds it.
struct PendingCall {
    scheduler: Arc<Scheduler>,
    pool: Arc<SchedulerPool>,
    changes: Arc<watch::Sender<u64>>,
    id: u64,
    armed: bool,
}

impl Drop for PendingCall {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let abandoned = self.scheduler.abandon(self.id).is_some() || self.pool.abandon(self.id);
        if abandoned {
            self.changes
                .send_modify(|version| *version = version.wrapping_add(1));
        }
    }
}

/// One dispatched call's bucket. Report how it went with
/// [`CallSlot::finish`]; dropping it unreported counts as cancelled.
///
/// The slot owns its scheduler rather than borrowing it, because a rebalance
/// can hand a waiting call to another backend and this is where it ends up.
#[derive(Debug)]
pub struct CallSlot {
    scheduler: Arc<Scheduler>,
    id: u64,
    level: u32,
    done: bool,
}

impl CallSlot {
    /// The parallelism level the call was dispatched under.
    #[must_use]
    pub const fn level(&self) -> u32 {
        self.level
    }

    /// Frees the bucket and records how the call went.
    pub fn finish(mut self, report: CallReport) {
        self.done = true;
        self.scheduler.finish(self.id, report);
    }
}

impl Drop for CallSlot {
    fn drop(&mut self) {
        if !self.done {
            self.scheduler
                .finish(self.id, CallReport::empty(CallOutcome::Cancelled));
        }
    }
}

impl std::fmt::Debug for PendingCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingCall").field("id", &self.id).finish()
    }
}

#[cfg(all(test, feature = "service-scheduler"))]
mod tests {
    use std::collections::HashSet;
    use std::sync::atomic::AtomicBool;

    use super::*;

    /// The three flavours of call the ordering tests mix.
    #[derive(Debug, Clone, Copy)]
    enum Kind {
        High,
        Medium,
        Reserved,
    }

    fn meta(kind: Kind, subject: &str) -> CallMeta {
        match kind {
            Kind::High => CallMeta::new("urgent-work", Priority::High, subject),
            Kind::Medium => CallMeta::new("event", Priority::Medium, subject),
            Kind::Reserved => {
                CallMeta::new("background", Priority::Low, subject).with_lane(Lane::Reserved)
            }
        }
    }

    fn config(ceiling: u32) -> SchedulerConfig {
        SchedulerConfig::new(ceiling).expect("a valid ceiling")
    }

    /// A gate the tests pause and make stale by hand.
    #[derive(Debug, Default)]
    struct FakeGate {
        paused: AtomicBool,
        stale: Mutex<HashSet<u64>>,
        wake: Arc<Notify>,
    }

    impl FakeGate {
        fn set_paused(&self, paused: bool) {
            self.paused.store(paused, Ordering::SeqCst);
            self.wake.notify_waiters();
        }

        fn make_stale(&self, task: u64) {
            self.stale
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(task);
            self.wake.notify_waiters();
        }
    }

    impl DispatchGate for FakeGate {
        fn paused(&self) -> bool {
            self.paused.load(Ordering::SeqCst)
        }

        fn is_stale(&self, task: u64) -> bool {
            self.stale
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .contains(&task)
        }

        fn wake(&self) -> &Arc<Notify> {
            &self.wake
        }
    }

    /// Queues calls in order and returns the queue.
    fn queued(calls: &[(Kind, &str)]) -> (Queue, Instant) {
        let now = Instant::now();
        let mut queue = Queue::default();
        for (kind, subject) in calls {
            queue.push(meta(*kind, subject), now);
        }
        (queue, now)
    }

    fn subjects(queue: &Queue, ids: &[u64]) -> Vec<String> {
        ids.iter()
            .filter_map(|id| queue.running.iter().find(|runner| runner.id == *id))
            .map(|runner| runner.meta.subject.clone())
            .collect()
    }

    mod order {
        use super::*;

        #[test]
        fn dispatches_the_biggest_value_first() {
            let (mut queue, now) = queued(&[
                (Kind::Reserved, "reserved"),
                (Kind::Medium, "medium"),
                (Kind::High, "high"),
            ]);

            let started = queue.dispatch(1, 4, now, |_| true);

            assert_eq!(subjects(&queue, &started), ["high"]);
        }

        #[test]
        fn breaks_a_tie_by_the_earliest_arrival() {
            let (mut queue, now) = queued(&[(Kind::Reserved, "first"), (Kind::Reserved, "second")]);

            let started = queue.dispatch(1, 4, now, |_| true);

            assert_eq!(subjects(&queue, &started), ["first"]);
        }

        #[test]
        fn ages_every_waiting_call_by_one_per_completion() {
            let (mut queue, now) = queued(&[(Kind::Reserved, "running"), (Kind::Reserved, "old")]);
            let started = queue.dispatch(1, 4, now, |_| true);
            queue.complete(started[0]);

            assert_eq!(queue.waiting[0].value, 2);
        }

        #[test]
        fn lets_an_aged_low_call_overtake_a_fresh_medium_one() {
            let (mut queue, now) = queued(&[(Kind::Reserved, "reserved")]);
            for round in 0..9 {
                queue.push(meta(Kind::High, &format!("high {round}")), now);
                let started = queue.dispatch(1, 4, now, |_| true);
                queue.complete(started[0]);
            }
            // The reserved call has waited through nine completions: 1 + 9 = 10,
            // and it arrived before the medium call.
            queue.push(meta(Kind::Medium, "medium"), now);

            let started = queue.dispatch(1, 4, now, |_| true);

            assert_eq!(subjects(&queue, &started), ["reserved"]);
        }

        #[test]
        fn dispatches_an_urgent_rerun_ahead_of_everything() {
            let (mut queue, now) = queued(&[(Kind::High, "high")]);
            queue.push(
                CallMeta {
                    reruns: 1,
                    ..meta(Kind::Reserved, "rerun")
                },
                now,
            );

            let started = queue.dispatch(1, 4, now, |_| true);

            assert_eq!(subjects(&queue, &started), ["rerun"]);
        }

        #[test]
        fn never_runs_more_calls_than_the_level() {
            let (mut queue, now) =
                queued(&[(Kind::High, "a"), (Kind::High, "b"), (Kind::High, "c")]);

            let started = queue.dispatch(2, 4, now, |_| true);

            assert_eq!(started.len(), 2);
        }
    }

    mod buckets {
        use super::*;

        #[test]
        fn shares_every_bucket_at_three_or_fewer() {
            let (mut queue, now) = queued(&[
                (Kind::High, "a"),
                (Kind::High, "b"),
                (Kind::High, "c"),
                (Kind::Reserved, "reserved"),
            ]);

            let started = queue.dispatch(3, 4, now, |_| true);

            assert_eq!(subjects(&queue, &started), ["a", "b", "c"]);
        }

        #[test]
        fn keeps_the_reserved_bucket_for_a_reserved_call_above_three() {
            let (mut queue, now) = queued(&[
                (Kind::High, "a"),
                (Kind::High, "b"),
                (Kind::High, "c"),
                (Kind::High, "d"),
                (Kind::Reserved, "reserved"),
            ]);

            let started = queue.dispatch(4, 4, now, |_| true);

            assert_eq!(subjects(&queue, &started), ["reserved", "a", "b", "c"]);
            let reserved = queue.running.iter().find(|r| r.meta.subject == "reserved");
            assert_eq!(reserved.map(|r| r.bucket), Some(0));
        }

        #[test]
        fn lends_the_reserved_bucket_to_a_shared_call_when_no_reserved_call_waits() {
            let (mut queue, now) = queued(&[
                (Kind::High, "a"),
                (Kind::High, "b"),
                (Kind::High, "c"),
                (Kind::High, "d"),
            ]);

            let started = queue.dispatch(4, 4, now, |_| true);

            assert_eq!(started.len(), 4);
            let lent = queue.running.iter().find(|r| r.bucket == 0);
            assert_eq!(
                lent.map(|r| r.meta.subject.as_str()),
                Some("d"),
                "filled last"
            );
        }

        #[test]
        fn never_puts_a_reserved_call_in_a_shared_bucket() {
            let (mut queue, now) = queued(&[(Kind::Reserved, "one"), (Kind::Reserved, "two")]);

            let started = queue.dispatch(4, 4, now, |_| true);

            assert_eq!(subjects(&queue, &started), ["one"]);
        }
    }

    mod load {
        use super::*;

        /// A load of `calls` calls at the assumed pace, at level `level`.
        fn load(calls: usize, level: u32) -> Load {
            Load {
                calls,
                waiting: calls,
                level,
                pace_ms: ASSUMED_CALL_MS,
            }
        }

        #[test]
        fn weighs_calls_against_the_level_in_force() {
            let busy_wide = load(3, 4);
            let idle_narrow = load(1, 1);

            assert!(busy_wide.is_lighter_than(idle_narrow));
        }

        #[test]
        fn is_not_lighter_than_an_equal_ratio() {
            assert!(!load(2, 2).is_lighter_than(load(1, 1)));
        }

        #[test]
        fn prefers_an_idle_backend_even_with_unmeasured_pace() {
            let busy_measured = Load {
                calls: 1,
                waiting: 0,
                level: 1,
                pace_ms: 500,
            };
            let idle_unmeasured = Load {
                calls: 0,
                waiting: 0,
                level: 1,
                pace_ms: ASSUMED_CALL_MS,
            };

            assert!(idle_unmeasured.is_lighter_than(busy_measured));
            assert!(!busy_measured.is_lighter_than(idle_unmeasured));
        }

        #[test]
        fn prefers_a_faster_backend_behind_a_longer_queue() {
            // Ten calls of half a second wait five seconds; one call of two
            // minutes waits two minutes.
            let fast = Load {
                calls: 10,
                waiting: 10,
                level: 1,
                pace_ms: 500,
            };
            let slow = Load {
                calls: 1,
                waiting: 1,
                level: 1,
                pace_ms: 120_000,
            };

            assert!(fast.is_lighter_than(slow));
            assert!(!slow.is_lighter_than(fast));
        }

        #[test]
        fn reads_the_wait_in_milliseconds() {
            let queued = Load {
                calls: 4,
                waiting: 4,
                level: 2,
                pace_ms: 1_000,
            };

            assert_eq!(queued.start_ms(), 2_000);
        }

        #[test]
        fn accepts_a_move_that_shortens_the_longest_wait() {
            // One backend holds two calls at a second each; the other is
            // idle. Moving one takes the worst wait from two seconds to one.
            let busy = Load {
                calls: 2,
                waiting: 1,
                level: 1,
                pace_ms: 1_000,
            };
            let idle = Load {
                calls: 0,
                waiting: 0,
                level: 1,
                pace_ms: 1_000,
            };

            assert!(lowers_the_worst(&[busy, idle], 0, 1));
            assert!(!lowers_the_worst(&[busy, idle], 1, 0));
        }

        #[test]
        fn moves_to_an_idle_backend_even_when_its_assumed_pace_is_slower() {
            // The busy backend measured 500 ms a call; the idle one has never
            // answered, so it still reads at ASSUMED_CALL_MS (2000 ms).
            // Without the idle-target short-circuit the assumed pace makes the
            // move look like it raises the worst wait, so rebalance refuses it
            // and every call piles onto the fast backend while the idle
            // servers stay idle.
            let busy = Load {
                calls: 3,
                waiting: 2,
                level: 1,
                pace_ms: 500,
            };
            let idle = Load {
                calls: 0,
                waiting: 0,
                level: 1,
                pace_ms: ASSUMED_CALL_MS,
            };

            assert!(lowers_the_worst(&[busy, idle], 0, 1));
            assert!(!lowers_the_worst(&[busy, idle], 1, 0));
        }

        #[test]
        fn refuses_the_move_that_would_only_swap_the_worst_back() {
            // Close but unequal paces: one call here and two there is already
            // the best of both arrangements, so the move back must be
            // refused. Equalising the counts is what oscillates.
            let fast = Load {
                calls: 1,
                waiting: 1,
                level: 1,
                pace_ms: 2_000,
            };
            let slow = Load {
                calls: 2,
                waiting: 2,
                level: 1,
                pace_ms: 1_730,
            };

            assert!(!lowers_the_worst(&[fast, slow], 0, 1));
            assert!(!lowers_the_worst(&[fast, slow], 1, 0));
        }
    }

    mod pace {
        use super::*;

        #[test]
        fn starts_from_the_assumed_call() {
            assert_eq!(Pace::new().mean_ms(), ASSUMED_CALL_MS);
        }

        #[test]
        fn one_slow_call_moves_the_mean_a_long_way() {
            let mut pace = Pace::new();

            pace.record(140_000);

            assert_eq!(pace.mean_ms(), 19_250);
        }

        #[test]
        fn settles_next_to_the_calls_it_is_fed() {
            let mut pace = Pace::new();

            for _ in 0..64 {
                pace.record(800);
            }

            // Within one step of the window: the mean stops moving once a
            // step would be less than half a millisecond.
            assert!(
                pace.mean_ms().abs_diff(800) <= 8,
                "settled at {}",
                pace.mean_ms()
            );
        }

        #[test]
        fn a_faster_call_than_the_mean_lowers_it() {
            let mut pace = Pace::new();

            pace.record(100);

            assert!(pace.mean_ms() < ASSUMED_CALL_MS);
        }
    }

    mod backend_id {
        use super::*;

        fn same(a: &str, b: &str) -> bool {
            BackendId::parse(a) == BackendId::parse(b)
        }

        #[test]
        fn counts_localhost_and_127_0_0_1_as_one_backend() {
            assert!(same("http://localhost:1234/v1", "http://127.0.0.1:1234/v1"));
        }

        #[test]
        fn counts_the_ipv6_loopback_as_the_same_backend() {
            assert!(same("http://[::1]:1234/v1", "http://127.0.0.1:1234/v1"));
        }

        #[test]
        fn counts_any_127_address_as_loopback() {
            assert!(same("http://127.1.2.3:1234/v1", "http://localhost:1234/v1"));
        }

        #[test]
        fn ignores_a_trailing_slash() {
            assert!(same(
                "http://127.0.0.1:1234/v1/",
                "http://127.0.0.1:1234/v1"
            ));
        }

        #[test]
        fn fills_in_the_default_port() {
            assert!(same("http://example.org/v1", "http://example.org:80/v1"));
        }

        #[test]
        fn ignores_case_in_scheme_and_host() {
            assert!(same("HTTP://LocalHost:1234/v1", "http://127.0.0.1:1234/v1"));
        }

        #[test]
        fn tells_two_ports_apart() {
            assert!(!same(
                "http://127.0.0.1:1234/v1",
                "http://127.0.0.1:4321/v1"
            ));
        }

        #[test]
        fn tells_two_paths_apart() {
            assert!(!same(
                "http://127.0.0.1:1234/v1",
                "http://127.0.0.1:1234/v2"
            ));
        }
    }

    mod pause {
        use super::*;

        #[derive(Debug)]
        struct FixedClock(Instant);

        impl Clock for FixedClock {
            fn now(&self) -> Instant {
                self.0
            }
        }

        fn scheduler(gate: &Arc<FakeGate>) -> Arc<Scheduler> {
            let gate: Arc<dyn DispatchGate> = Arc::clone(gate) as Arc<dyn DispatchGate>;
            Arc::new(Scheduler::new(
                BackendId::parse("http://127.0.0.1:1234/v1"),
                config(4),
                Arc::new(FixedClock(Instant::now())),
                gate,
                Arc::new(watch::Sender::new(0)),
                Arc::new(SchedulerPool::new()),
            ))
        }

        #[tokio::test]
        async fn holds_new_non_urgent_calls_while_the_gate_is_paused() {
            let gate = Arc::new(FakeGate::default());
            gate.set_paused(true);
            let scheduler = scheduler(&gate);

            let waiting = tokio::time::timeout(
                Duration::from_millis(50),
                scheduler.acquire(meta(Kind::High, "held")),
            )
            .await;

            assert!(waiting.is_err(), "the call must wait for the pause to end");
        }

        #[tokio::test]
        async fn dispatches_an_urgent_rerun_during_the_pause() {
            let gate = Arc::new(FakeGate::default());
            gate.set_paused(true);
            let scheduler = scheduler(&gate);

            let slot = tokio::time::timeout(
                Duration::from_millis(50),
                scheduler.acquire(meta(Kind::Reserved, "rerun").with_reruns(1)),
            )
            .await;

            assert!(matches!(slot, Ok(Ok(_))), "the rerun goes at once");
        }

        #[tokio::test]
        async fn resumes_normal_dispatch_when_the_pause_ends() {
            let gate = Arc::new(FakeGate::default());
            gate.set_paused(true);
            let scheduler = scheduler(&gate);
            let waiting = {
                let scheduler = Arc::clone(&scheduler);
                tokio::spawn(async move {
                    scheduler
                        .acquire(meta(Kind::High, "held"))
                        .await
                        .map(|slot| slot.finish(CallReport::empty(CallOutcome::Ok)))
                })
            };
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert!(!waiting.is_finished(), "paused while the gate is paused");

            gate.set_paused(false);

            let resumed = tokio::time::timeout(Duration::from_secs(1), waiting).await;
            assert!(
                matches!(resumed, Ok(Ok(Ok(())))),
                "dispatched once the pause is over"
            );
        }

        #[tokio::test]
        async fn withdraws_a_waiting_call_whose_task_went_stale() {
            let gate = Arc::new(FakeGate::default());
            gate.make_stale(7);
            let scheduler = scheduler(&gate);

            let sent = scheduler
                .acquire(meta(Kind::Reserved, "stale").with_task(7))
                .await;

            assert_eq!(sent.err(), Some(SchedulerError::InputsChanged));
        }
    }

    mod reserved_hold {
        use super::*;

        #[tokio::test]
        async fn delays_reserved_dispatch_while_held() {
            let scheduler = Arc::new(Scheduler::new(
                BackendId::parse("http://127.0.0.1:1234"),
                config(1),
                Arc::new(SystemClock),
                Arc::new(OpenGate::default()),
                Arc::new(watch::Sender::new(0)),
                Arc::new(SchedulerPool::new()),
            ));
            scheduler.hold_reserved_for(Duration::from_secs(10));

            let fut = scheduler.acquire(meta(Kind::Reserved, "char_01"));
            tokio::pin!(fut);

            assert!(
                tokio::time::timeout(Duration::from_millis(50), &mut fut)
                    .await
                    .is_err()
            );

            scheduler.clear_reserved_hold();
            let slot = tokio::time::timeout(Duration::from_millis(100), fut)
                .await
                .expect("completes")
                .expect("acquired slot");
            assert_eq!(slot.level(), 1);
        }
    }

    mod queue_full {
        use super::*;

        #[tokio::test]
        async fn refuses_the_call_past_max_waiting() {
            let config = config(1).with_max_waiting(1).expect("a valid cap");
            let scheduler = Arc::new(Scheduler::new(
                BackendId::parse("http://127.0.0.1:1234"),
                config,
                Arc::new(SystemClock),
                Arc::new(OpenGate::default()),
                Arc::new(watch::Sender::new(0)),
                Arc::new(SchedulerPool::new()),
            ));
            let running = scheduler
                .enqueue(meta(Kind::High, "runs"))
                .expect("the first call starts");
            let waiting = scheduler
                .enqueue(meta(Kind::High, "waits"))
                .expect("the second call waits");

            let refused = scheduler.enqueue(meta(Kind::High, "refused"));

            assert_eq!(
                refused,
                Err(SchedulerError::QueueFull {
                    backend: BackendId::parse("http://127.0.0.1:1234"),
                    max_waiting: 1,
                })
            );
            assert_eq!(scheduler.load().calls, 2, "the refused call was not queued");
            assert!(scheduler.lock().queue.is_running(running));
            assert!(scheduler.lock().queue.is_waiting(waiting));
        }

        #[test]
        fn rejects_a_config_out_of_range() {
            assert!(SchedulerConfig::new(0).is_err());
            assert!(SchedulerConfig::new(65).is_err());
            assert!(config(1).with_max_waiting(0).is_err());
            let mut zero_level = config(1);
            zero_level.reserved_lane_from_level = 0;
            assert!(zero_level.validate().is_err());
            assert!(SchedulerConfig::new(64).is_ok());
        }
    }

    mod totals {
        use super::*;

        #[tokio::test]
        async fn folds_the_thirty_third_kind_into_other() {
            let scheduler = Arc::new(Scheduler::new(
                BackendId::parse("http://127.0.0.1:1234"),
                config(1),
                Arc::new(SystemClock),
                Arc::new(OpenGate::default()),
                Arc::new(watch::Sender::new(0)),
                Arc::new(SchedulerPool::new()),
            ));
            for index in 0..MAX_KINDS + 2 {
                let meta = CallMeta::new(format!("kind-{index}"), Priority::Low, "subject");
                let slot = scheduler.acquire(meta).await.expect("an idle scheduler");
                slot.finish(CallReport::empty(CallOutcome::Ok));
            }

            let snapshot = scheduler.snapshot();

            assert_eq!(snapshot.totals.len(), MAX_KINDS + 1);
            let counted: u64 = snapshot.totals.iter().map(|entry| entry.totals.calls).sum();
            assert_eq!(counted, 34, "every call is counted once");
            let other = snapshot.totals.last().expect("an other entry");
            assert_eq!(other.kind, OTHER_KIND);
            assert_eq!(other.totals.calls, 2);
            assert_eq!(snapshot.totals[0].kind, "kind-0", "first-seen order");
        }
    }

    /// A pool of the schedulers for `urls`, sharing one gate the way a
    /// dispatcher's backends do.
    fn pool_of(urls: &[&str]) -> (Arc<SchedulerPool>, Vec<Arc<Scheduler>>) {
        let pool = Arc::new(SchedulerPool::new());
        let gate: Arc<dyn DispatchGate> = Arc::new(OpenGate::default());
        let schedulers = urls
            .iter()
            .map(|url| {
                let scheduler = Arc::new(Scheduler::new(
                    BackendId::parse(url),
                    config(4),
                    Arc::new(SystemClock),
                    Arc::clone(&gate),
                    Arc::new(watch::Sender::new(0)),
                    Arc::clone(&pool),
                ));
                assert!(pool.insert(&scheduler).is_some(), "the pool takes a bit");
                scheduler
            })
            .collect();
        (pool, schedulers)
    }

    /// Waits (without sleeping) until `scheduler` holds `expected` calls, so a
    /// spawned acquirer has queued.
    async fn settle(scheduler: &Arc<Scheduler>, expected: usize) {
        for _ in 0..10_000 {
            if scheduler.load().calls == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(scheduler.load().calls, expected, "the call never queued");
    }

    mod rebalance {
        use super::*;

        #[tokio::test]
        async fn hands_a_waiting_call_to_an_idle_sibling() {
            let (_pool, schedulers) = pool_of(&["http://127.0.0.1:1/v1", "http://127.0.0.1:2/v1"]);
            let held = schedulers[0]
                .acquire(meta(Kind::Medium, "held"))
                .await
                .expect("the first call starts");
            let queued = {
                let scheduler = Arc::clone(&schedulers[0]);
                tokio::spawn(async move {
                    scheduler
                        .acquire(meta(Kind::Reserved, "moved"))
                        .await
                        .map(|slot| slot.finish(CallReport::empty(CallOutcome::Ok)))
                })
            };
            settle(&schedulers[0], 2).await;

            assert!(move_waiting(&schedulers[0], &schedulers[1], 1 << 1));

            assert_eq!(schedulers[0].load().calls, 1, "only the held call is left");
            assert_eq!(schedulers[1].load().calls, 1, "the sibling took it");
            let dispatched = tokio::time::timeout(Duration::from_millis(200), queued)
                .await
                .expect("the moved call is dispatched")
                .expect("the task did not panic");
            assert!(dispatched.is_ok(), "the moved call got a slot");
            held.finish(CallReport::empty(CallOutcome::Ok));
        }

        #[tokio::test]
        async fn never_hands_over_a_running_call() {
            let (_pool, schedulers) = pool_of(&["http://127.0.0.1:1/v1", "http://127.0.0.1:2/v1"]);
            let held = schedulers[0]
                .acquire(meta(Kind::Medium, "held"))
                .await
                .expect("the call starts");

            assert!(!move_waiting(&schedulers[0], &schedulers[1], 1 << 1));

            assert_eq!(schedulers[0].load().calls, 1);
            assert_eq!(schedulers[1].load().calls, 0);
            held.finish(CallReport::empty(CallOutcome::Ok));
        }

        #[tokio::test]
        async fn never_hands_a_call_to_a_backend_it_may_not_use() {
            let (_pool, schedulers) = pool_of(&["http://127.0.0.1:1/v1", "http://127.0.0.1:2/v1"]);
            let mut pinned = meta(Kind::Reserved, "pinned");
            pinned.allowed = 1 << 0;
            let id = {
                let mut state = schedulers[0].lock();
                state.queue.push(pinned, Instant::now())
            };

            assert!(!move_waiting(&schedulers[0], &schedulers[1], 1 << 1));

            assert_eq!(
                schedulers[0].load().calls,
                1,
                "it stays where it was queued"
            );
            assert_eq!(schedulers[1].load().calls, 0);
            std::assert_matches!(
                schedulers[0].place(id),
                Some(MemberPlacement::Running(_)),
                "the pinned call still runs on the backend it queued at"
            );
        }

        #[tokio::test]
        async fn finds_a_waiting_call_after_a_move() {
            let (pool, schedulers) = pool_of(&["http://127.0.0.1:1/v1", "http://127.0.0.1:2/v1"]);
            // Both schedulers busy, so the moved call has to wait where it
            // lands rather than being dispatched at once.
            let held_a = schedulers[0]
                .acquire(meta(Kind::Medium, "held a"))
                .await
                .expect("a starts");
            let held_b = schedulers[1]
                .acquire(meta(Kind::Medium, "held b"))
                .await
                .expect("b starts");
            let id = {
                let mut state = schedulers[0].lock();
                state
                    .queue
                    .push(meta(Kind::Reserved, "moved"), Instant::now())
            };

            assert!(move_waiting(&schedulers[0], &schedulers[1], 1 << 1));

            assert!(matches!(pool.place(id), Placement::Waiting));
            assert_eq!(schedulers[1].load().calls, 2, "waiting behind b's bucket");
            held_a.finish(CallReport::empty(CallOutcome::Ok));
            held_b.finish(CallReport::empty(CallOutcome::Ok));
        }
    }

    mod pool_abandon {
        use super::*;

        #[tokio::test]
        async fn withdraws_a_moved_call_whichever_member_holds_it() {
            let (pool, schedulers) = pool_of(&["http://127.0.0.1:1/v1", "http://127.0.0.1:2/v1"]);
            let held_a = schedulers[0]
                .acquire(meta(Kind::Medium, "held a"))
                .await
                .expect("a starts");
            let held_b = schedulers[1]
                .acquire(meta(Kind::Medium, "held b"))
                .await
                .expect("b starts");
            let dropped = {
                let scheduler = Arc::clone(&schedulers[0]);
                tokio::spawn(
                    async move { scheduler.acquire(meta(Kind::Reserved, "dropped")).await },
                )
            };
            settle(&schedulers[0], 2).await;
            let id = {
                let state = schedulers[0].lock();
                state.queue.waiting[0].id
            };
            assert!(move_waiting(&schedulers[0], &schedulers[1], 1 << 1));
            assert!(matches!(pool.place(id), Placement::Waiting));

            dropped.abort();
            let _ = dropped.await;

            assert_eq!(
                schedulers[1].load().calls,
                1,
                "the sibling keeps only its own"
            );
            assert_eq!(schedulers[0].load().calls, 1);
            held_a.finish(CallReport::empty(CallOutcome::Ok));
            held_b.finish(CallReport::empty(CallOutcome::Ok));
        }
    }
}
