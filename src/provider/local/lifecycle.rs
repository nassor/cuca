//! How long a [`LocalModel`](super::LocalModel) outlives its last handle: the
//! end of its engine thread.
//!
//! A model on a GPU gives its device memory back when its engine is dropped,
//! and the backend does that on a device thread of its own (`hipFree` through
//! the CubeCL device handle). A process that returns from `main` while that
//! call is in flight runs the runtime's exit handlers underneath it: HIP
//! then aborts (`pure virtual method called`) or crashes. The engine thread
//! therefore reports when it has dropped everything it built
//! ([`SignalOnDrop`], a guard that outlives the engine), and the two places
//! where a model's life can end wait for that, for a bounded time:
//!
//! - the last clone of the model ([`EngineStop`], the last field of the shared
//!   state, so the job queue has closed by the time it waits);
//! - a load that is cancelled ([`LoadGuard`], the `start` future dropped before
//!   the engine was reported ready), when the engine thread has begun to open
//!   the device. The thread asks [`EngineLife::proceed`] before it does, so a
//!   load cancelled earlier (a download, a conversion) never touches the
//!   device and nothing waits for it.
//!
//! Nothing here depends on a backend: a CPU model signals the same way and
//! its wait ends as soon as the thread does.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

/// The longest the last clone of a model waits for the engine thread. The
/// thread ends when the job queue is closed and the requests that were
/// running are done; their consumers are gone at shutdown, so in practice it
/// takes the time of the device teardown (about 0.1 s for a 0.6B model, a
/// second for 8B).
pub(super) const ENGINE_STOP_WAIT: Duration = Duration::from_secs(10);

/// The longest a cancelled load waits for an engine thread that has begun to
/// open the device. The weights of the model that is loading upload first,
/// and a large one takes longer than a stop: past the bound the process exits
/// as it did before.
pub(super) const LOAD_CANCEL_WAIT: Duration = Duration::from_secs(30);

/// Reading, downloading or converting: the device is untouched.
const LOADING: u8 = 0;
/// The engine thread has begun to open the device.
const DEVICE: u8 = 1;
/// The caller went away before the engine was ready.
const CANCELLED: u8 = 2;

/// The shared life of one engine thread: its load phase and whether it has
/// ended.
#[derive(Debug)]
pub(super) struct EngineLife {
    phase: AtomicU8,
    /// The engine thread, so that a wait on it from itself (the last clone of
    /// the model dropped inside a call that runs there) returns at once.
    thread: OnceLock<ThreadId>,
    /// Set once the thread has dropped everything it built.
    done: Mutex<bool>,
    ended: Condvar,
}

impl EngineLife {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            phase: AtomicU8::new(LOADING),
            thread: OnceLock::new(),
            done: Mutex::new(false),
            ended: Condvar::new(),
        })
    }

    /// The calling thread is the engine thread.
    pub(super) fn enter(&self) {
        let _ = self.thread.set(std::thread::current().id());
    }

    /// Whether the engine thread may go on to build the engine: `false` once
    /// the load was cancelled. With `device`, a `true` answer also records that
    /// the device is about to be touched, so that a cancellation from now on
    /// waits for the thread to end (and one that came earlier is seen here,
    /// before the device is).
    pub(super) fn proceed(&self, device: bool) -> bool {
        if device {
            match self
                .phase
                .compare_exchange(LOADING, DEVICE, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => true,
                Err(phase) => phase == DEVICE,
            }
        } else {
            self.phase.load(Ordering::Acquire) != CANCELLED
        }
    }

    /// Mark the load cancelled; `true` when the engine thread had begun to
    /// open the device, which the caller must wait out.
    fn cancel(&self) -> bool {
        self.phase.swap(CANCELLED, Ordering::AcqRel) == DEVICE
    }

    /// The engine thread has dropped everything it built.
    fn finish(&self) {
        *self.done.lock().unwrap_or_else(PoisonError::into_inner) = true;
        self.ended.notify_all();
    }

    /// Wait up to `bound` for the engine thread to end; whether it did. A
    /// wait on the engine thread itself returns `false` at once.
    pub(super) fn wait(&self, bound: Duration) -> bool {
        if self.thread.get() == Some(&std::thread::current().id()) {
            return false;
        }
        let deadline = Instant::now() + bound;
        let mut done = self.done.lock().unwrap_or_else(PoisonError::into_inner);
        while !*done {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            done = self
                .ended
                .wait_timeout(done, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        true
    }
}

/// Reports the end of the engine thread when dropped. Declare it first in
/// the thread's closure: locals drop in reverse order, so it fires after the
/// engine (and with it the GPU device) is gone, on every path out, a panic
/// included.
pub(super) struct SignalOnDrop(pub(super) Arc<EngineLife>);

impl Drop for SignalOnDrop {
    fn drop(&mut self) {
        self.0.finish();
    }
}

/// The last field of the shared state of a model: dropping the last clone
/// waits for the engine thread (the job queue, a field declared before it,
/// has closed by then).
pub(super) struct EngineStop(pub(super) Arc<EngineLife>);

impl Drop for EngineStop {
    fn drop(&mut self) {
        self.0.wait(ENGINE_STOP_WAIT);
    }
}

/// Held by the `start` future until the engine is ready: when the future is
/// dropped first (a shutdown that cancels the load), the engine thread is told
/// to stop at its next check and the drop waits for it if it has begun to
/// open the device.
pub(super) struct LoadGuard(Option<Arc<EngineLife>>);

impl LoadGuard {
    pub(super) fn new(life: Arc<EngineLife>) -> Self {
        Self(Some(life))
    }

    /// The engine is ready: nothing to cancel.
    pub(super) fn ready(mut self) {
        self.0 = None;
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        if let Some(life) = self.0.take()
            && life.cancel()
        {
            life.wait(LOAD_CANCEL_WAIT);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;

    /// A thread that signals its end `after` the guard is created.
    fn thread_ending_after(life: &Arc<EngineLife>, after: Duration) -> std::thread::JoinHandle<()> {
        let life = Arc::clone(life);
        std::thread::spawn(move || {
            life.enter();
            let _end = SignalOnDrop(Arc::clone(&life));
            std::thread::sleep(after);
        })
    }

    #[test]
    fn a_wait_ends_with_the_thread() {
        let life = EngineLife::new();
        let thread = thread_ending_after(&life, Duration::from_millis(150));
        let started = Instant::now();
        assert!(life.wait(Duration::from_secs(5)));
        assert!(started.elapsed() >= Duration::from_millis(100));
        thread.join().unwrap();
        // once ended, every wait returns at once
        assert!(life.wait(Duration::ZERO));
    }

    #[test]
    fn a_wait_is_bounded() {
        let life = EngineLife::new();
        let started = Instant::now();
        assert!(!life.wait(Duration::from_millis(100)));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn the_signal_fires_when_the_thread_panics() {
        let life = EngineLife::new();
        let l = Arc::clone(&life);
        let thread = std::thread::spawn(move || {
            let _end = SignalOnDrop(Arc::clone(&l));
            panic!("the engine thread failed");
        });
        assert!(thread.join().is_err());
        assert!(life.wait(Duration::from_secs(5)));
    }

    #[test]
    fn the_engine_thread_does_not_wait_for_itself() {
        let life = EngineLife::new();
        let l = Arc::clone(&life);
        let waited = std::thread::spawn(move || {
            l.enter();
            let started = Instant::now();
            let ended = l.wait(Duration::from_secs(30));
            (ended, started.elapsed())
        })
        .join()
        .unwrap();
        assert!(!waited.0);
        assert!(waited.1 < Duration::from_secs(5), "{:?}", waited.1);
    }

    #[test]
    fn a_cancel_before_the_device_is_not_waited_for() {
        let life = EngineLife::new();
        let guard = LoadGuard::new(Arc::clone(&life));
        let started = Instant::now();
        drop(guard); // the thread never began to open the device
        assert!(started.elapsed() < Duration::from_secs(5));
        // and the thread, when it asks, is told to stop before it touches the device
        assert!(!life.proceed(true));
        assert!(!life.proceed(false));
    }

    #[test]
    fn a_cancel_after_the_device_began_waits_for_the_thread() {
        let life = EngineLife::new();
        let l = Arc::clone(&life);
        let began = Arc::new(AtomicBool::new(false));
        let b = Arc::clone(&began);
        let thread = std::thread::spawn(move || {
            l.enter();
            let _end = SignalOnDrop(Arc::clone(&l));
            assert!(l.proceed(true));
            b.store(true, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(300)); // building the engine
        });
        while !began.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        let guard = LoadGuard::new(Arc::clone(&life));
        let started = Instant::now();
        drop(guard);
        assert!(
            started.elapsed() >= Duration::from_millis(200),
            "the cancelled load returned while the device was still being opened: {:?}",
            started.elapsed()
        );
        assert!(life.wait(Duration::ZERO), "the thread had ended");
        thread.join().unwrap();
    }

    #[test]
    fn a_ready_load_is_not_cancelled() {
        let life = EngineLife::new();
        LoadGuard::new(Arc::clone(&life)).ready();
        assert!(life.proceed(true));
        assert!(life.proceed(true));
    }
}
