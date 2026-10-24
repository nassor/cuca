//! Priority call scheduler over one or more backends (`service-scheduler`):
//! a queue per backend with aging, an adaptive parallelism level, and a
//! [`Dispatcher`] that places each call on the backend that would start it
//! soonest.
//!
//! A [`Scheduler`] decides which waiting call goes next and how many run at
//! once. A [`Dispatcher`] owns one scheduler per backend, picks a backend for
//! every [`CallMeta`], fails over when a backend is unreachable, and streams
//! the call through a [`CucaClient`](crate::client::CucaClient). The pieces
//! are generic: a call's kind is a free label, and the caller's own rules
//! reach the queue through [`DispatchGate`].
//!
//! # Explicit-call contract
//!
//! An explicit-call service, never a [`CucaPlugin`](crate::plugin::CucaPlugin)
//! ([`crate::services`] owns that contract). The entry points are
//! [`Dispatcher::call`] for a whole call, [`Scheduler::acquire`] for a bare
//! slot, [`Dispatcher::rebalance`] and [`Dispatcher::set_backends`].
//!
//! Why no hook could carry this capability:
//! [`CucaPlugin::on_request`](crate::plugin::CucaPlugin::on_request) is
//! synchronous, so a hook can only inspect or reject a request, never park it
//! behind other calls. Parking is the capability.
//!
//! # Mandatory hand-off
//!
//! Two duties stay with the caller:
//!
//! - Run [`Dispatcher::rebalance`] on the caller's own beat (a timer, a
//!   periodic tick). Nothing in the service spawns it, so a call that waits
//!   behind a slow backend moves to an idle one only when the caller says so.
//! - Finish every [`CallSlot`] taken from [`Scheduler::acquire`] with
//!   [`CallSlot::finish`]. Dropping one counts as [`CallOutcome::Cancelled`]
//!   and teaches the tuner and the pace estimate nothing. [`Dispatcher::call`]
//!   finishes its own slot.
//!
//! A caller whose [`DispatchGate`] changes state (a pause begins or ends, a
//! task goes stale) calls `notify_waiters()` on the gate's
//! [`wake`](DispatchGate::wake) notifier so the queues look again.
//!
//! # Ordering
//!
//! Each waiting call has a value that starts at its [`Priority`]
//! ([`Priority::initial_value`]: low 1, medium 10, high 100, urgent 10 000;
//! [`CallMeta::reruns`] above 0 makes any call urgent). Every completion at
//! the backend adds 1 to every call still waiting, so a low call cannot
//! starve forever. The waiting call with the biggest value goes next; ties go
//! to the earliest arrival.
//!
//! The parallelism level `P` is the number of buckets. From
//! [`SchedulerConfig::reserved_lane_from_level`] (default 4) up, bucket 0
//! serves [`Lane::Reserved`] calls first and a [`Lane::Shared`] call only
//! when no reserved call waits, and the other `P - 1` buckets never serve a
//! reserved call. Below that level every bucket serves one shared queue. The
//! tuner starts each scheduler at `P = 1` and raises it up to the configured
//! ceiling only after measuring a gain (see [`TunerSnapshot`]).
//!
//! While [`DispatchGate::paused`] is true only urgent calls are dispatched.
//! A waiting call whose task [`DispatchGate::is_stale`] is withdrawn unsent
//! and its caller gets [`SchedulerError::InputsChanged`].
//!
//! # Bounds
//!
//! - [`SchedulerConfig::max_waiting`] (default 256) caps the calls waiting at
//!   one backend; past it [`Scheduler::enqueue`] refuses with
//!   [`SchedulerError::QueueFull`].
//! - [`SchedulerConfig::ceiling`] is at most 64.
//! - A pool holds at most 64 backends; the 65th is rejected as a
//!   [`DispatchError::Config`].
//! - Totals are kept for at most 32 distinct [`CallMeta::kind`] labels; the
//!   33rd and later are counted under `"other"`.
//! - [`SchedulerSnapshot::recent`] is a ring of the last 64 completed calls.
//! - The tuner keeps 128 throughput samples and a history of 32 level
//!   changes.
//!
//! # Determinism
//!
//! All time passes through the [`Clock`] given to [`Scheduler::new`]
//! ([`SystemClock`] in production), so queue order, aging and the tuner are
//! testable without sleeping. Ties break by call id, which rises with
//! arrival. State locks are never held across an `.await`.
//!
//! # Runtime coupling
//!
//! Needs a Tokio reactor: waiting calls park on a `tokio::sync::Notify`,
//! [`Scheduler::hold_reserved_for`] spawns a timer, and
//! [`Dispatcher::call`] builds lazy clients on a `JoinSet`.
//!
//! # Not covered
//!
//! No retry of a failed call: [`Dispatcher::call`] fails over only when the
//! backend was unreachable before any stream existed, and returns every other
//! error to the caller. No 429 or `Retry-After` feedback: a rate-limited
//! response is a [`CucaError::Http`](crate::error::CucaError::Http) that
//! reaches the caller as [`DispatchError::Call`], and the tuner sees it only
//! as a failed call.

mod dispatch;
mod queue;
mod tuner;

pub use dispatch::{
    Backend, BackendClient, BackendRegistry, BackendSpec, ClientFactory, Completion, DispatchError,
    Dispatcher, DispatcherConfig,
};

pub use queue::{
    BackendId, BucketSnapshot, CallMeta, CallOutcome, CallReport, CallSlot, Clock, CompletedCall,
    DispatchGate, KindTotals, KindTotalsEntry, Lane, Load, OpenGate, Priority, RunningCall,
    Scheduler, SchedulerConfig, SchedulerError, SchedulerPool, SchedulerSnapshot, SystemClock,
    WaitingCall,
};
pub use tuner::{ChangeReason, Direction, LevelChange, Metric, TunerPhase, TunerSnapshot};
