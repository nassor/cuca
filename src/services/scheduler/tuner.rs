//! The parallelism tuner: how many model calls one backend runs at once.
//!
//! Every boot starts at one call at a time. The tuner has to *earn* each
//! extra level of parallelism by measuring it, one level at a time.
//!
//! CONTEXT: overlapping requests to one local LM Studio server whose
//! `max_tokens` summed past roughly 1.5k to 2k stalled 35 to 110 s before the
//! first token, so a backend is never assumed to take parallel calls.
//!
//! An experiment moves the level one step (up or down, clamped to
//! `[1, ceiling]`), measures the trial over the next `max(trial − 1, 1)`
//! calls dispatched under it, and compares that against the incumbent level's
//! most recent interval of the same size:
//!
//! - a step up is kept only on a clear gain ([`GAIN_TO_GO_UP`]);
//! - a step down is kept unless throughput falls more than a small margin
//!   ([`LOSS_TO_STAY_DOWN`]), since fewer parallel calls for the same
//!   throughput is the healthier backend;
//! - any failed call during a trial (a timeout, an unreachable backend, an
//!   error status) reverts it at once.
//!
//! It hill-climbs: an accepted step keeps its direction, a rejected one
//! flips it. The first experiment is due [`FIRST_EXPERIMENT`] after boot and
//! each later interval doubles, up to [`EXPERIMENT_CAP`]. A due experiment is
//! postponed, and runs later, while the backlog is smaller than the trial
//! level (there would be nothing to run in parallel), while the dispatch gate
//! is paused (the pause would measure the pause), or while the
//! incumbent has not yet completed enough calls to compare against.
//!
//! Time is always passed in, so every rule here is testable with explicit
//! instants and no sleeping.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::Serialize;

use super::queue::CallOutcome;

/// When the first experiment is due, counted from boot.
///
/// CONTEXT: fifteen minutes of real calls at one-at-a-time gives the
/// incumbent a baseline from the whole mix of call kinds before anything is
/// tried against it.
pub(super) const FIRST_EXPERIMENT: Duration = Duration::from_secs(15 * 60);

/// The longest interval between two experiments.
///
/// CONTEXT: the interval doubles after every experiment (15 min, 30 min,
/// 1 h, 2 h, …), so a backend that has found its level is left alone, while
/// one experiment a day still notices a model swap or a backend that got
/// faster or slower.
pub(super) const EXPERIMENT_CAP: Duration = Duration::from_secs(24 * 60 * 60);

/// Throughput a trial one level up must reach, as a multiple of the
/// incumbent's, to be kept.
///
/// CONTEXT: the gain has to stand clear of noise. One more call in flight
/// costs the backend memory and risks the overlap stall, so it has to pay
/// for itself by at least a tenth.
pub(super) const GAIN_TO_GO_UP: f64 = 1.10;

/// Throughput a trial one level down must keep, as a multiple of the
/// incumbent's, to be kept.
///
/// CONTEXT: fewer calls in flight for the same throughput is the healthier
/// backend, so a step down is kept unless it measurably costs throughput:
/// more than five percent, about the noise between two windows of the same
/// level.
pub(super) const LOSS_TO_STAY_DOWN: f64 = 0.95;

/// Completed calls remembered for baselines.
///
/// CONTEXT: a trial window is at most `ceiling − 1` calls and the scheduler
/// caps its ceiling at 64, so the incumbent's most recent window always
/// fits with room for the calls of other levels interleaved with it.
const SAMPLE_CAPACITY: usize = 128;

/// Level changes kept for the snapshot.
const HISTORY_CAPACITY: usize = 32;

/// The shortest interval a throughput is divided by.
///
/// CONTEXT: a window of calls that all completed in the same instant (an
/// unreachable backend answers in microseconds) must not divide by zero;
/// a millisecond is below anything a real model call takes.
const MIN_INTERVAL_SECS: f64 = 0.001;

/// One completed call, as the tuner measures it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Sample {
    /// The scheduler's call id.
    pub id: u64,
    /// The parallelism level in force when the call was dispatched.
    pub level: u32,
    /// When it was dispatched.
    pub dispatched: Instant,
    /// When it completed.
    pub completed: Instant,
    /// Output characters it streamed back, always measured.
    pub chars: u64,
    /// Completion tokens the backend reported for it; `None` when it
    /// reported none (see [`Metric`]).
    pub tokens: Option<u64>,
    /// How it ended.
    pub outcome: CallOutcome,
}

/// What a throughput is measured in.
///
/// CONTEXT: completion tokens a second when the backend reports them, since
/// a token is the unit the backend decodes. Characters per token differ
/// between prose, JSON and names, so a trial that happened to run the
/// JSON-heavy calls would read slower in characters for the same work.
/// Characters stay the fallback: every call measures them, and a backend
/// that reports no usage (or one call of a window that reported none) is
/// still comparable. One comparison never mixes the two: a trial is
/// compared in tokens only when every successful call of its own window
/// *and* of the baseline window reported tokens, otherwise both sides are
/// compared in characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Completion tokens a second, as the backend reported them.
    CompletionTokensPerSecond,
    /// Output characters a second, as streamed back.
    OutputCharsPerSecond,
}

/// Output summed over successful calls, in both units.
#[derive(Debug, Clone, Copy)]
struct Units {
    chars: u64,
    /// `None` once any summed call reported no tokens.
    tokens: Option<u64>,
}

impl Units {
    /// Nothing summed yet: every (no) call reported tokens.
    const NONE: Self = Self {
        chars: 0,
        tokens: Some(0),
    };

    fn add(&mut self, sample: &Sample) {
        self.chars = self.chars.saturating_add(sample.chars);
        self.tokens = self
            .tokens
            .zip(sample.tokens)
            .map(|(sum, tokens)| sum.saturating_add(tokens));
    }

    fn per_second(self, interval: Duration) -> Rate {
        Rate {
            chars: throughput(self.chars, interval),
            tokens: self.tokens.map(|tokens| throughput(tokens, interval)),
        }
    }
}

/// A throughput in both units; tokens only when every call reported them.
#[derive(Debug, Clone, Copy)]
struct Rate {
    chars: f64,
    tokens: Option<f64>,
}

impl Rate {
    /// This rate in its best unit.
    fn preferred(self) -> (Metric, f64) {
        self.tokens
            .map_or((Metric::OutputCharsPerSecond, self.chars), |tokens| {
                (Metric::CompletionTokensPerSecond, tokens)
            })
    }

    /// The one unit both `baseline` and `trial` are measured in, with each
    /// one's value in it: tokens only when both have them.
    fn common(baseline: Self, trial: Self) -> (Metric, f64, f64) {
        match (baseline.tokens, trial.tokens) {
            (Some(baseline), Some(trial)) => (Metric::CompletionTokensPerSecond, baseline, trial),
            _ => (Metric::OutputCharsPerSecond, baseline.chars, trial.chars),
        }
    }
}

/// Which way the next experiment steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// One more call in flight.
    Up,
    /// One fewer.
    Down,
}

impl Direction {
    const fn flipped(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
        }
    }
}

/// Why the level changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeReason {
    /// An experiment began at the trial level.
    Trial,
    /// The trial measured at least as well as it had to; it stays.
    Kept,
    /// The trial measured worse than it had to; back to the incumbent.
    Rejected,
    /// A call failed during the trial; back to the incumbent at once.
    Failed,
}

/// One change of the parallelism level.
#[derive(Debug, Clone, Serialize)]
pub struct LevelChange {
    /// Milliseconds since the scheduler started.
    pub at_ms: u64,
    /// The level before.
    pub from: u32,
    /// The level after (equal to `from` for [`ChangeReason::Kept`]).
    pub to: u32,
    /// Why.
    pub reason: ChangeReason,
    /// Incumbent throughput the trial was compared against, in
    /// [`Self::metric`]; `None` on the change that began the trial.
    pub baseline: Option<f64>,
    /// The trial's measured throughput, when it was measured to the end.
    pub trial: Option<f64>,
    /// The unit `baseline` and `trial` are in; `None` on the change that
    /// began the trial.
    pub metric: Option<Metric>,
}

/// The tuner's phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TunerPhase {
    /// Running at the incumbent level, waiting for the next experiment.
    Steady,
    /// Measuring a trial level.
    Trial,
}

/// What the tuner is doing, for the dashboard.
#[derive(Debug, Clone, Serialize)]
pub struct TunerSnapshot {
    /// Steady or measuring a trial.
    pub phase: TunerPhase,
    /// The level in force: the trial level during a trial.
    pub level: u32,
    /// The level a trial is measured against (the current level when
    /// steady).
    pub incumbent: u32,
    /// The level being tried, during a trial.
    pub trial_level: Option<u32>,
    /// Calls a trial is measured over: `max(trial − 1, 1)`.
    pub window: Option<u32>,
    /// Calls dispatched under the trial level so far (counted to `window`).
    pub trial_dispatched: Option<u32>,
    /// Of those, how many have completed.
    pub trial_completed: Option<u32>,
    /// The incumbent's throughput the running trial is compared against,
    /// in [`Self::metric`].
    pub baseline_throughput: Option<f64>,
    /// The running trial's throughput so far, in [`Self::metric`].
    pub trial_throughput: Option<f64>,
    /// Which way the next experiment steps.
    pub direction: Direction,
    /// Milliseconds since the scheduler started at which the next
    /// experiment is due; `None` when the ceiling is one (nothing to try).
    pub next_experiment_at_ms: Option<u64>,
    /// The current interval between experiments, in milliseconds.
    pub interval_ms: u64,
    /// The unit the running trial is compared in; `None` when steady.
    pub metric: Option<Metric>,
    /// The most recent level changes, oldest first.
    pub history: Vec<LevelChange>,
}

#[derive(Debug, Clone)]
struct Trial {
    incumbent: u32,
    level: u32,
    began: Instant,
    window: u32,
    /// Ids of the first `window` calls dispatched under the trial level.
    dispatched: Vec<u64>,
    /// How many of those have completed.
    completed: u32,
    /// Output of every successful call completed since the trial began.
    units: Units,
    baseline: Rate,
}

/// The hill-climbing parallelism tuner of one backend.
#[derive(Debug)]
pub(super) struct Tuner {
    ceiling: u32,
    level: u32,
    direction: Direction,
    interval: Duration,
    next_due: Instant,
    started: Instant,
    trial: Option<Trial>,
    samples: VecDeque<Sample>,
    history: VecDeque<LevelChange>,
}

impl Tuner {
    /// A tuner at level 1 whose first experiment is due
    /// [`FIRST_EXPERIMENT`] after `started`.
    pub(super) fn new(ceiling: u32, started: Instant) -> Self {
        Self {
            ceiling: ceiling.max(1),
            level: 1,
            direction: Direction::Up,
            interval: FIRST_EXPERIMENT,
            next_due: started + FIRST_EXPERIMENT,
            started,
            trial: None,
            samples: VecDeque::with_capacity(SAMPLE_CAPACITY),
            history: VecDeque::with_capacity(HISTORY_CAPACITY),
        }
    }

    /// The parallelism level in force.
    pub(super) const fn level(&self) -> u32 {
        self.level
    }

    /// Starts a due experiment if nothing postpones it.
    ///
    /// `backlog` is every call waiting or running on the backend; `paused`
    /// says the dispatch gate is paused.
    pub(super) fn poll(&mut self, now: Instant, backlog: usize, paused: bool) {
        if self.trial.is_some() || self.ceiling <= 1 || now < self.next_due || paused {
            return;
        }
        let target = self.target();
        if backlog < usize::try_from(target).unwrap_or(usize::MAX) {
            return;
        }
        let window = target.saturating_sub(1).max(1);
        let Some(baseline) = self.baseline(window) else {
            return;
        };
        self.change(now, target, ChangeReason::Trial, None, None);
        self.trial = Some(Trial {
            incumbent: self.level,
            level: target,
            began: now,
            window,
            dispatched: Vec::with_capacity(usize::try_from(window).unwrap_or(0)),
            completed: 0,
            units: Units::NONE,
            baseline,
        });
        self.level = target;
    }

    /// Notes a call dispatched at the current level.
    pub(super) fn on_dispatch(&mut self, id: u64) {
        if let Some(trial) = &mut self.trial
            && trial.dispatched.len() < usize::try_from(trial.window).unwrap_or(usize::MAX)
        {
            trial.dispatched.push(id);
        }
    }

    /// Records a completed call and concludes a trial it completes or fails.
    pub(super) fn on_complete(&mut self, sample: Sample) {
        if self.samples.len() == SAMPLE_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        let Some(trial) = &mut self.trial else {
            return;
        };
        if sample.outcome == CallOutcome::Failed {
            self.conclude(sample.completed, false, None, ChangeReason::Failed);
            return;
        }
        if sample.outcome == CallOutcome::Ok {
            trial.units.add(&sample);
        }
        if trial.dispatched.contains(&sample.id) {
            trial.completed += 1;
        }
        if trial.completed < trial.window {
            return;
        }
        let (metric, baseline, measured) = Rate::common(
            trial.baseline,
            trial.units.per_second(sample.completed - trial.began),
        );
        let accepted = if trial.level > trial.incumbent {
            measured >= baseline * GAIN_TO_GO_UP
        } else {
            measured >= baseline * LOSS_TO_STAY_DOWN
        };
        let reason = if accepted {
            ChangeReason::Kept
        } else {
            ChangeReason::Rejected
        };
        self.conclude(
            sample.completed,
            accepted,
            Some((metric, baseline, measured)),
            reason,
        );
    }

    pub(super) fn snapshot(&self, now: Instant) -> TunerSnapshot {
        let trial = self.trial.as_ref();
        let compared = trial.map(|trial| {
            Rate::common(
                trial.baseline,
                trial
                    .units
                    .per_second(now.saturating_duration_since(trial.began)),
            )
        });
        TunerSnapshot {
            phase: if trial.is_some() {
                TunerPhase::Trial
            } else {
                TunerPhase::Steady
            },
            level: self.level,
            incumbent: trial.map_or(self.level, |trial| trial.incumbent),
            trial_level: trial.map(|trial| trial.level),
            window: trial.map(|trial| trial.window),
            trial_dispatched: trial
                .map(|trial| u32::try_from(trial.dispatched.len()).unwrap_or(u32::MAX)),
            trial_completed: trial.map(|trial| trial.completed),
            baseline_throughput: compared.map(|(_, baseline, _)| baseline),
            trial_throughput: compared.map(|(_, _, trial)| trial),
            direction: self.direction,
            next_experiment_at_ms: (self.ceiling > 1).then(|| self.millis(self.next_due)),
            interval_ms: u64::try_from(self.interval.as_millis()).unwrap_or(u64::MAX),
            metric: compared.map(|(metric, _, _)| metric),
            history: self.history.iter().cloned().collect(),
        }
    }

    /// The next experiment's level: one step in the current direction,
    /// flipping it at either end of `[1, ceiling]`.
    fn target(&mut self) -> u32 {
        let step = |direction, level: u32| match direction {
            Direction::Up => level.checked_add(1).filter(|next| *next <= self.ceiling),
            Direction::Down => level.checked_sub(1).filter(|next| *next >= 1),
        };
        if let Some(next) = step(self.direction, self.level) {
            return next;
        }
        self.direction = self.direction.flipped();
        step(self.direction, self.level).unwrap_or(self.level)
    }

    /// The incumbent's throughput over its most recent `window` successful
    /// calls: the output of every successful call completed between the
    /// first of them being dispatched and the last completing, over that
    /// interval.
    fn baseline(&self, window: u32) -> Option<Rate> {
        let window = usize::try_from(window).ok()?;
        let recent: Vec<&Sample> = self
            .samples
            .iter()
            .rev()
            .filter(|sample| sample.level == self.level && sample.outcome == CallOutcome::Ok)
            .take(window)
            .collect();
        if recent.len() < window {
            return None;
        }
        let start = recent.iter().map(|sample| sample.dispatched).min()?;
        let end = recent.iter().map(|sample| sample.completed).max()?;
        let mut units = Units::NONE;
        for sample in self.samples.iter().filter(|sample| {
            sample.outcome == CallOutcome::Ok
                && sample.completed >= start
                && sample.completed <= end
        }) {
            units.add(sample);
        }
        Some(units.per_second(end - start))
    }

    /// Ends the trial. `measured` is the comparison it was decided on (unit,
    /// baseline, trial), or `None` when a failure ended it early.
    fn conclude(
        &mut self,
        now: Instant,
        accepted: bool,
        measured: Option<(Metric, f64, f64)>,
        reason: ChangeReason,
    ) {
        let Some(trial) = self.trial.take() else {
            return;
        };
        let to = if accepted {
            trial.level
        } else {
            trial.incumbent
        };
        let (metric, baseline, measured) = measured.map_or_else(
            || {
                let (metric, baseline) = trial.baseline.preferred();
                (metric, baseline, None)
            },
            |(metric, baseline, measured)| (metric, baseline, Some(measured)),
        );
        self.change(now, to, reason, Some((metric, baseline)), measured);
        self.level = to;
        if !accepted {
            self.direction = self.direction.flipped();
        }
        self.interval = (self.interval * 2).min(EXPERIMENT_CAP);
        self.next_due = now + self.interval;
    }

    fn change(
        &mut self,
        now: Instant,
        to: u32,
        reason: ChangeReason,
        baseline: Option<(Metric, f64)>,
        trial: Option<f64>,
    ) {
        if self.history.len() == HISTORY_CAPACITY {
            self.history.pop_front();
        }
        self.history.push_back(LevelChange {
            at_ms: self.millis(now),
            from: self.level,
            to,
            reason,
            baseline: baseline.map(|(_, baseline)| baseline),
            trial,
            metric: baseline.map(|(metric, _)| metric),
        });
    }

    fn millis(&self, at: Instant) -> u64 {
        u64::try_from(at.saturating_duration_since(self.started).as_millis()).unwrap_or(u64::MAX)
    }
}

/// Units (characters or tokens) a second over `interval`.
#[expect(
    clippy::cast_precision_loss,
    reason = "a throughput is a ratio for comparison; counts past 2^53 are not real"
)]
fn throughput(units: u64, interval: Duration) -> f64 {
    units as f64 / interval.as_secs_f64().max(MIN_INTERVAL_SECS)
}

#[cfg(test)]
mod test {
    use super::*;

    fn sample(id: u64, level: u32, dispatched: Instant, completed: Instant, units: u64) -> Sample {
        Sample {
            id,
            level,
            dispatched,
            completed,
            chars: units,
            tokens: None,
            outcome: CallOutcome::Ok,
        }
    }

    /// A tuner at level 1, ceiling 4, with one completed call of `units`
    /// characters over ten seconds, and its first experiment due.
    fn due_tuner(units: u64) -> (Tuner, Instant) {
        let start = Instant::now();
        let mut tuner = Tuner::new(4, start);
        let at = start + FIRST_EXPERIMENT - Duration::from_secs(10);
        tuner.on_complete(sample(0, 1, at, at + Duration::from_secs(10), units));
        (tuner, start + FIRST_EXPERIMENT)
    }

    mod schedule {
        use super::*;

        #[test]
        fn starts_every_boot_at_one_call_at_a_time() {
            let tuner = Tuner::new(4, Instant::now());

            assert_eq!(tuner.level(), 1);
        }

        #[test]
        fn holds_the_first_experiment_until_fifteen_minutes_after_boot() {
            let (mut tuner, due) = due_tuner(1000);

            tuner.poll(due - Duration::from_secs(1), 8, false);
            assert_eq!(tuner.level(), 1, "a second early");
            tuner.poll(due, 8, false);
            assert_eq!(tuner.level(), 2, "on time");
        }

        #[test]
        fn doubles_the_interval_after_each_experiment_up_to_a_day() {
            let (mut tuner, mut due) = due_tuner(1000);
            let mut intervals = Vec::new();
            let mut id = 1;
            for _ in 0..9 {
                // Re-arm the incumbent's baseline at whatever level it is now.
                let level = tuner.level();
                tuner.on_complete(sample(
                    id,
                    level,
                    due - Duration::from_secs(10),
                    due - Duration::from_secs(1),
                    1000,
                ));
                id += 1;
                tuner.poll(due, 8, false);
                // One failure ends the trial at once.
                tuner.on_complete(Sample {
                    outcome: CallOutcome::Failed,
                    ..sample(id, tuner.level(), due, due, 0)
                });
                id += 1;
                intervals.push(tuner.interval);
                due = tuner.next_due;
            }

            let minutes: Vec<u64> = intervals.iter().map(|d| d.as_secs() / 60).collect();
            assert_eq!(minutes, [30, 60, 120, 240, 480, 960, 1440, 1440, 1440]);
        }

        #[test]
        fn postpones_a_trial_while_the_backlog_is_smaller_than_the_trial_level() {
            let (mut tuner, due) = due_tuner(1000);

            tuner.poll(due, 1, false);
            assert_eq!(tuner.level(), 1, "one waiting call cannot run two at once");
            tuner.poll(due + Duration::from_secs(60), 2, false);
            assert_eq!(tuner.level(), 2, "postponed, not skipped");
        }

        #[test]
        fn postpones_a_trial_while_the_gate_is_paused() {
            let (mut tuner, due) = due_tuner(1000);

            tuner.poll(due, 8, true);

            assert_eq!(tuner.level(), 1);
        }

        #[test]
        fn never_experiments_with_a_ceiling_of_one() {
            let start = Instant::now();
            let mut tuner = Tuner::new(1, start);
            tuner.on_complete(sample(0, 1, start, start + Duration::from_secs(1), 100));

            tuner.poll(start + EXPERIMENT_CAP * 2, 8, false);

            assert_eq!(tuner.level(), 1);
            assert_eq!(tuner.snapshot(start).next_experiment_at_ms, None);
        }
    }

    mod trial {
        use super::*;

        /// Starts the 1 → 2 trial (window 1) and completes its one call
        /// after `seconds` with `units` characters.
        fn run_up_trial(units: u64, seconds: u64) -> Tuner {
            let (mut tuner, due) = due_tuner(1000);
            tuner.poll(due, 8, false);
            tuner.on_dispatch(10);
            tuner.on_complete(sample(
                10,
                2,
                due,
                due + Duration::from_secs(seconds),
                units,
            ));
            tuner
        }

        #[test]
        fn keeps_a_step_up_that_clearly_raises_throughput() {
            // Baseline 100 chars/s; the trial runs 150 chars/s.
            let tuner = run_up_trial(1500, 10);

            assert_eq!(tuner.level(), 2);
            assert_eq!(tuner.direction, Direction::Up, "keeps climbing");
        }

        #[test]
        fn rejects_a_step_up_that_gains_less_than_a_tenth() {
            // Baseline 100 chars/s; the trial runs 105 chars/s.
            let tuner = run_up_trial(1050, 10);

            assert_eq!(tuner.level(), 1);
            assert_eq!(tuner.direction, Direction::Down, "flips after a rejection");
        }

        #[test]
        fn reverts_a_stalled_step_up() {
            // The overlap stall: two calls in flight, the first token after
            // a minute.
            let tuner = run_up_trial(1000, 70);

            assert_eq!(tuner.level(), 1);
        }

        #[test]
        fn reverts_a_trial_the_moment_a_call_fails() {
            let (mut tuner, due) = due_tuner(1000);
            tuner.poll(due, 8, false);
            tuner.on_dispatch(10);

            tuner.on_complete(Sample {
                outcome: CallOutcome::Failed,
                ..sample(
                    11,
                    1,
                    due - Duration::from_secs(5),
                    due + Duration::from_secs(1),
                    0,
                )
            });

            assert_eq!(tuner.level(), 1);
            assert_eq!(
                tuner.history.back().map(|change| change.reason),
                Some(ChangeReason::Failed)
            );
        }

        #[test]
        fn measures_the_trial_over_trial_level_minus_one_calls() {
            // At 3 → 4 the window is three calls.
            let start = Instant::now();
            let mut tuner = Tuner::new(4, start);
            tuner.level = 3;
            let due = start + FIRST_EXPERIMENT;
            for id in 0..3 {
                tuner.on_complete(sample(id, 3, due - Duration::from_secs(10), due, 1000));
            }
            tuner.poll(due, 8, false);
            for id in 10..13 {
                tuner.on_dispatch(id);
            }
            tuner.on_complete(sample(10, 4, due, due + Duration::from_secs(5), 5000));
            tuner.on_complete(sample(11, 4, due, due + Duration::from_secs(6), 5000));
            assert_eq!(
                tuner.trial.as_ref().map(|t| t.completed),
                Some(2),
                "still measuring"
            );

            tuner.on_complete(sample(12, 4, due, due + Duration::from_secs(7), 5000));

            assert!(tuner.trial.is_none(), "concluded on the third");
            assert_eq!(tuner.level(), 4);
        }

        #[test]
        fn keeps_a_step_down_that_costs_no_throughput() {
            let start = Instant::now();
            let mut tuner = Tuner::new(4, start);
            tuner.level = 3;
            tuner.direction = Direction::Down;
            let due = start + FIRST_EXPERIMENT;
            // Incumbent window (2 calls): 2000 chars in 10 s = 200/s.
            tuner.on_complete(sample(
                0,
                3,
                due - Duration::from_secs(10),
                due - Duration::from_secs(1),
                1000,
            ));
            tuner.on_complete(sample(1, 3, due - Duration::from_secs(10), due, 1000));
            tuner.poll(due, 8, false);
            assert_eq!(tuner.level(), 2);
            tuner.on_dispatch(10);

            // 1960 chars in 10 s = 196/s: within the margin.
            tuner.on_complete(sample(10, 2, due, due + Duration::from_secs(10), 1960));

            assert_eq!(tuner.level(), 2);
        }

        #[test]
        fn rejects_a_step_down_that_loses_throughput() {
            let start = Instant::now();
            let mut tuner = Tuner::new(4, start);
            tuner.level = 3;
            tuner.direction = Direction::Down;
            let due = start + FIRST_EXPERIMENT;
            tuner.on_complete(sample(
                0,
                3,
                due - Duration::from_secs(10),
                due - Duration::from_secs(1),
                1000,
            ));
            tuner.on_complete(sample(1, 3, due - Duration::from_secs(10), due, 1000));
            tuner.poll(due, 8, false);
            tuner.on_dispatch(10);

            tuner.on_complete(sample(10, 2, due, due + Duration::from_secs(10), 1500));

            assert_eq!(tuner.level(), 3);
            assert_eq!(tuner.direction, Direction::Up);
        }

        #[test]
        fn tries_up_from_one_even_when_the_climb_was_heading_down() {
            let (mut tuner, due) = due_tuner(1000);
            tuner.direction = Direction::Down;

            tuner.poll(due, 8, false);

            assert_eq!(tuner.level(), 2);
        }
    }

    mod metric {
        use super::*;

        /// Starts the 1 → 2 trial (window 1) over a ten-second incumbent
        /// call of 1000 chars and `baseline_tokens`, then completes the
        /// trial's one call after ten seconds with `chars` and `tokens`.
        fn run_up_trial(baseline_tokens: Option<u64>, chars: u64, tokens: Option<u64>) -> Tuner {
            let start = Instant::now();
            let mut tuner = Tuner::new(4, start);
            let due = start + FIRST_EXPERIMENT;
            tuner.on_complete(Sample {
                tokens: baseline_tokens,
                ..sample(0, 1, due - Duration::from_secs(10), due, 1000)
            });
            tuner.poll(due, 8, false);
            tuner.on_dispatch(10);
            tuner.on_complete(Sample {
                tokens,
                ..sample(10, 2, due, due + Duration::from_secs(10), chars)
            });
            tuner
        }

        fn last_change(tuner: &Tuner) -> Option<(Option<Metric>, Option<f64>, Option<f64>)> {
            tuner
                .history
                .back()
                .map(|change| (change.metric, change.baseline, change.trial))
        }

        #[test]
        fn compares_in_completion_tokens_when_both_windows_report_them() {
            // 100 chars/s on both sides would reject; 20 → 30 tok/s is a
            // clear gain in the unit the backend decodes.
            let tuner = run_up_trial(Some(200), 1000, Some(300));

            assert_eq!(tuner.level(), 2);
            assert_eq!(
                last_change(&tuner),
                Some((
                    Some(Metric::CompletionTokensPerSecond),
                    Some(20.0),
                    Some(30.0)
                ))
            );
        }

        #[test]
        fn falls_back_to_chars_on_both_sides_when_the_trial_reports_no_tokens() {
            // Trial 105 chars/s against 100: rejected. Mixing units (105
            // chars/s against the baseline's 20 tok/s) would keep it.
            let tuner = run_up_trial(Some(200), 1050, None);

            assert_eq!(tuner.level(), 1);
            assert_eq!(
                last_change(&tuner),
                Some((Some(Metric::OutputCharsPerSecond), Some(100.0), Some(105.0)))
            );
        }

        #[test]
        fn falls_back_to_chars_on_both_sides_when_the_baseline_reports_no_tokens() {
            // Trial 150 chars/s against 100: kept. Mixing units (1 tok/s
            // against the baseline's 100 chars/s) would reject it.
            let tuner = run_up_trial(None, 1500, Some(10));

            assert_eq!(tuner.level(), 2);
            assert_eq!(
                last_change(&tuner),
                Some((Some(Metric::OutputCharsPerSecond), Some(100.0), Some(150.0)))
            );
        }

        #[test]
        fn one_call_without_tokens_puts_its_whole_window_in_chars() {
            // The 3 → 2 trial's baseline window is the most recent call,
            // which reported tokens; an interleaved call completed inside
            // that interval did not, so the baseline has no token rate.
            let start = Instant::now();
            let mut tuner = Tuner::new(4, start);
            tuner.level = 3;
            tuner.direction = Direction::Down;
            let due = start + FIRST_EXPERIMENT;
            tuner.on_complete(sample(0, 3, due - Duration::from_secs(8), due, 1000));
            tuner.on_complete(Sample {
                tokens: Some(500),
                ..sample(1, 3, due - Duration::from_secs(10), due, 1000)
            });
            tuner.poll(due, 8, false);
            assert_eq!(tuner.level(), 2);
            tuner.on_dispatch(10);

            let snapshot = tuner.snapshot(due + Duration::from_secs(10));
            assert_eq!(snapshot.metric, Some(Metric::OutputCharsPerSecond));
            assert_eq!(snapshot.baseline_throughput, Some(200.0));
        }

        #[test]
        fn a_failed_trial_records_the_baseline_in_its_own_unit() {
            let start = Instant::now();
            let mut tuner = Tuner::new(4, start);
            let due = start + FIRST_EXPERIMENT;
            tuner.on_complete(Sample {
                tokens: Some(200),
                ..sample(0, 1, due - Duration::from_secs(10), due, 1000)
            });
            tuner.poll(due, 8, false);

            tuner.on_complete(Sample {
                outcome: CallOutcome::Failed,
                ..sample(10, 2, due, due, 0)
            });

            assert_eq!(
                last_change(&tuner),
                Some((Some(Metric::CompletionTokensPerSecond), Some(20.0), None))
            );
        }
    }
}
