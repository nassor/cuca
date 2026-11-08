//! The models a server holds and their load state.
//!
//! The registry is built once from the configured [`ModelSpec`]s and never
//! changes size ([`MAX_MODELS`](super::MAX_MODELS) at most). Every model
//! starts `Loading`; [`ModelRegistry::start`] loads them concurrently, each
//! on its own task (the heavy work runs on the engine's own thread, see
//! [`LocalModel::load`]), and a slot turns `Ready` or `Failed` when its load
//! ends. Requests and `/health` read the slots; nothing in a slot grows with
//! traffic.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use tokio::task::{JoinHandle, JoinSet};

use super::config::ModelSpec;
use super::error::ServerError;
use super::metrics::ModelMetrics;
use super::openai::unix_secs;
use crate::{DownloadProgressCallback, LocalCapabilities, LocalModel};

/// A model that finished loading, with the figures the server reports.
#[derive(Debug)]
pub struct LoadedModel {
    /// The engine handle; cheap to clone.
    pub model: LocalModel,
    /// Most prompt plus completion tokens per request.
    pub context_length: usize,
    /// Bytes of the K/V pools.
    pub kv_bytes: u64,
    /// Size of the GGUF the model runs from.
    pub weights_bytes: u64,
    /// What the model supports.
    pub capabilities: LocalCapabilities,
    /// When the load finished, seconds since the Unix epoch.
    pub loaded_at: u64,
}

/// Where a slot's model is in its life.
#[derive(Debug, Clone)]
pub enum SlotStatus {
    /// Downloading, converting or reading the weights.
    Loading,
    /// Serving requests.
    Ready(Arc<LoadedModel>),
    /// The load failed with this message.
    Failed(Arc<str>),
}

/// Progress a load reports (download, conversion or merge): `done` of
/// `total` units, whatever the unit is.
#[derive(Debug, Default)]
struct LoadProgress {
    done: AtomicU64,
    /// `0` when the total is unknown.
    total: AtomicU64,
    /// The last logged step, so the log carries about twenty lines per load.
    logged_step: AtomicU64,
}

/// One configured model.
#[derive(Debug)]
pub struct ModelSlot {
    /// The name clients select the model by.
    pub name: Arc<str>,
    /// Where it comes from.
    pub source: String,
    /// The device it runs on.
    pub device: String,
    /// Its counters and histograms.
    pub metrics: ModelMetrics,
    status: RwLock<SlotStatus>,
    progress: LoadProgress,
}

impl ModelSlot {
    fn new(spec: &ModelSpec) -> Self {
        Self {
            name: Arc::from(spec.name.as_str()),
            source: spec.source_label.clone(),
            device: spec.device_label.clone(),
            metrics: ModelMetrics::default(),
            status: RwLock::new(SlotStatus::Loading),
            progress: LoadProgress::default(),
        }
    }

    /// The current status (a cheap clone).
    pub fn status(&self) -> SlotStatus {
        self.status
            .read()
            .map_or(SlotStatus::Loading, |status| status.clone())
    }

    /// The model, once it is ready.
    pub fn ready(&self) -> Option<Arc<LoadedModel>> {
        match self.status() {
            SlotStatus::Ready(model) => Some(model),
            _ => None,
        }
    }

    fn set(&self, status: SlotStatus) {
        // A poisoned lock still holds a valid status; replace it anyway.
        let mut guard = self
            .status
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = status;
    }

    /// The fraction of the current load step done, when it reports one.
    pub fn load_progress(&self) -> Option<f64> {
        let total = self.progress.total.load(Ordering::Relaxed);
        let done = self.progress.done.load(Ordering::Relaxed);
        (total > 0).then(|| (done as f64 / total as f64).min(1.0))
    }

    /// A callback for [`LocalModelConfig::on_download_progress`](crate::LocalModelConfig::on_download_progress)
    /// that records the progress and logs about every 5%.
    fn progress_callback(self: &Arc<Self>) -> DownloadProgressCallback {
        let slot = Arc::clone(self);
        DownloadProgressCallback::new(move |p| {
            let total = p.total_bytes.unwrap_or(0);
            slot.progress
                .done
                .store(p.downloaded_bytes, Ordering::Relaxed);
            slot.progress.total.store(total, Ordering::Relaxed);
            let step = p
                .downloaded_bytes
                .saturating_mul(20)
                .checked_div(total)
                .unwrap_or(p.downloaded_bytes >> 28);
            if slot.progress.logged_step.fetch_max(step, Ordering::Relaxed) < step {
                tracing::info!(
                    model = %slot.name,
                    done = p.downloaded_bytes,
                    total = (total > 0).then_some(total),
                    percent = (total > 0).then(|| p.downloaded_bytes * 100 / total),
                    "loading: download, conversion or merge progress"
                );
            }
        })
    }
}

/// Every model of the server, by position.
#[derive(Debug)]
pub struct ModelRegistry {
    slots: Vec<Arc<ModelSlot>>,
}

impl ModelRegistry {
    /// A registry with every model `Loading`; nothing is loaded.
    pub fn pending(specs: &[ModelSpec]) -> Arc<Self> {
        Arc::new(Self {
            slots: specs.iter().map(|s| Arc::new(ModelSlot::new(s))).collect(),
        })
    }

    /// A registry for `specs` and the task that loads them all concurrently.
    /// The task ends `Ok` when every model is ready and with the first
    /// failure otherwise (the slot of a failed model reads `Failed`).
    pub fn start(specs: Vec<ModelSpec>) -> (Arc<Self>, JoinHandle<Result<(), ServerError>>) {
        let registry = Self::pending(&specs);
        let loader = {
            let registry = Arc::clone(&registry);
            tokio::spawn(async move {
                let mut loads = JoinSet::new();
                for (slot, spec) in registry.slots.iter().zip(specs) {
                    loads.spawn(load_one(Arc::clone(slot), spec));
                }
                while let Some(joined) = loads.join_next().await {
                    joined
                        .map_err(|e| ServerError::Io(format!("a model loader panicked: {e}")))??;
                }
                Ok(())
            })
        };
        (registry, loader)
    }

    /// The slots, in configuration order.
    pub fn slots(&self) -> &[Arc<ModelSlot>] {
        &self.slots
    }

    /// Whether every model is ready.
    pub fn all_ready(&self) -> bool {
        self.slots
            .iter()
            .all(|s| matches!(s.status(), SlotStatus::Ready(_)))
    }

    /// The slot a request's `model` field selects: with one model, any (or no)
    /// name selects it, as llama.cpp's server does; with several the name must
    /// match one exactly.
    ///
    /// # Errors
    ///
    /// [`ResolveError::Missing`] when several models are loaded and the
    /// request names none, [`ResolveError::Unknown`] when the name matches
    /// none.
    pub fn resolve(&self, requested: Option<&str>) -> Result<&Arc<ModelSlot>, ResolveError> {
        if let [only] = self.slots.as_slice() {
            return Ok(only);
        }
        let name = requested.ok_or(ResolveError::Missing)?;
        self.slots
            .iter()
            .find(|s| &*s.name == name)
            .ok_or_else(|| ResolveError::Unknown(name.to_string()))
    }
}

/// Why [`ModelRegistry::resolve`] found no model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// Several models are loaded and the request names none.
    Missing,
    /// The requested name matches no model.
    Unknown(String),
}

/// Load one model into its slot.
async fn load_one(slot: Arc<ModelSlot>, spec: ModelSpec) -> Result<(), ServerError> {
    let mut config = spec.config;
    config.on_download_progress = Some(slot.progress_callback());
    tracing::info!(model = %slot.name, source = %slot.source, device = %slot.device, "loading model");
    let started = Instant::now();
    match LocalModel::load(config).await {
        Ok(model) => {
            let loaded = LoadedModel {
                context_length: model.context_length(),
                kv_bytes: model.kv_cache_bytes(),
                weights_bytes: model.weights_file_bytes(),
                capabilities: model.capabilities(),
                loaded_at: unix_secs(),
                model,
            };
            tracing::info!(
                model = %slot.name,
                seconds = started.elapsed().as_secs_f64(),
                context_length = loaded.context_length,
                kv_mib = loaded.kv_bytes >> 20,
                weights_mib = loaded.weights_bytes >> 20,
                max_sequences = loaded.model.serving_stats().max_sequences,
                reasoning = loaded.capabilities.reasoning,
                tools = loaded.capabilities.tools,
                "model ready"
            );
            slot.set(SlotStatus::Ready(Arc::new(loaded)));
            Ok(())
        }
        Err(source) => {
            tracing::error!(model = %slot.name, error = %source, "model failed to load");
            slot.set(SlotStatus::Failed(Arc::from(source.to_string())));
            Err(ServerError::ModelLoad {
                model: slot.name.to_string(),
                source,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::config::ModelOptions;

    fn registry(names: &[&str]) -> Arc<ModelRegistry> {
        let specs: Vec<ModelSpec> = names
            .iter()
            .map(|n| {
                ModelSpec::from_arg(&format!("{n}=qwen3-0.6b-q8_0"), &ModelOptions::default())
                    .unwrap()
            })
            .collect();
        ModelRegistry::pending(&specs)
    }

    #[test]
    fn a_single_model_answers_to_any_name_or_none() {
        let r = registry(&["only"]);
        assert_eq!(&*r.resolve(None).unwrap().name, "only");
        assert_eq!(&*r.resolve(Some("gpt-4")).unwrap().name, "only");
    }

    #[test]
    fn several_models_need_an_exact_name() {
        let r = registry(&["a", "b"]);
        assert_eq!(&*r.resolve(Some("b")).unwrap().name, "b");
        assert_eq!(
            r.resolve(Some("c")).unwrap_err(),
            ResolveError::Unknown("c".into())
        );
        assert_eq!(r.resolve(None).unwrap_err(), ResolveError::Missing);
    }

    #[test]
    fn a_pending_registry_is_not_ready_and_reports_progress_once_known() {
        let r = registry(&["a"]);
        assert!(!r.all_ready());
        let slot = &r.slots()[0];
        assert_eq!(slot.load_progress(), None);
        slot.progress.total.store(200, Ordering::Relaxed);
        slot.progress.done.store(50, Ordering::Relaxed);
        assert_eq!(slot.load_progress(), Some(0.25));
    }
}
