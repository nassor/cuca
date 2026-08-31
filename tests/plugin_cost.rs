//! Integration tests for the cost-accounting plugin (`plugin-cost`).
//!
//! Three tests drive a real llama.cpp turn and assert the ledger moved; they
//! skip when the server is unreachable. The other three are deterministic: a
//! budget refusal never reaches a provider at all, and dispatch counts plus the
//! exact prompt a provider receives can only be asserted against
//! `common::spawn_counting_sse_server`, not against a real model.
#![cfg(all(feature = "provider-llamacpp", feature = "plugin-cost"))]

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cuca::plugin::CucaPlugin;
use cuca::types::MessageContentBlock;
use cuca::{
    CostConfig, CostPlugin, CucaClient, CucaError, ModelRates, PluginError, PricingTable,
    UnifiedRequest,
};

/// Records the request every hook registered after it observes.
///
/// Hooks run in registration order over one shared `UnifiedRequest`, so a
/// recorder placed after a mutating plugin captures that plugin's edits.
#[derive(Default)]
struct RequestRecorder {
    seen: Mutex<Vec<UnifiedRequest>>,
}

impl CucaPlugin for RequestRecorder {
    fn name(&self) -> &'static str {
        "test-request-recorder"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        self.seen.lock().unwrap().push(req.clone());
        Ok(())
    }
}

/// Rates that make one small turn cost a visible, non-zero number of micros.
fn rates() -> ModelRates {
    ModelRates {
        input_micros_per_mtok: 3_000_000,
        output_micros_per_mtok: 15_000_000,
        cache_read_micros_per_mtok: 300_000,
        cache_write_micros_per_mtok: 3_750_000,
    }
}

/// A plugin with no caps and no pricing: it counts tokens and nothing else.
fn counting_plugin() -> Arc<CostPlugin> {
    Arc::new(CostPlugin::new(CostConfig::default()).expect("cost plugin must build"))
}

/// A client at `addr` with `plugins` registered, in order.
fn client_at(addr: &str, plugins: Vec<Arc<dyn CucaPlugin>>) -> CucaClient {
    let mut builder = common::llamacpp_builder(addr.to_string());
    for plugin in plugins {
        builder = builder.register_plugin(plugin);
    }
    builder.build().expect("client build must succeed")
}

/// The request the deterministic tests dispatch.
fn mock_request() -> UnifiedRequest {
    UnifiedRequest::new("cost-model")
        .add_system_message("You are concise.")
        .add_user_message("Reply with the single word: ok")
}

/// A live turn's ledger is moved by both hooks: `on_request` charges the prompt
/// estimate and `on_response_complete` charges the completion estimate and
/// commits the turn.
#[tokio::test]
async fn live_turn_charges_prompt_and_completion_tokens() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let cost = counting_plugin();
    let client = common::client_with_plugins(vec![Arc::clone(&cost) as Arc<dyn CucaPlugin>]);

    let stream = client
        .generate_stream(common::live_request(
            "Reply with the single word: ok",
            &model,
        ))
        .await
        .expect("generate_stream must start");
    let blocks = common::drain_timeout(stream, 60).await;
    assert!(
        blocks
            .iter()
            .any(|b| matches!(b, MessageContentBlock::Text(_))),
        "expected at least one Text block, got {blocks:?}"
    );

    let usage = cost.usage().expect("ledger lock must not be poisoned");
    assert!(
        usage.prompt_tokens > 0,
        "the prompt estimate must be charged, got {usage:?}"
    );
    assert!(
        usage.completion_tokens > 0,
        "the completion estimate must be charged, got {usage:?}"
    );
    assert_eq!(usage.turns, 1, "one committed turn, got {usage:?}");
    assert_eq!(
        usage.spent_micros, 0,
        "no rates were configured, so no currency is charged"
    );
    assert_eq!(
        usage.unpriced_turns, 1,
        "the unpriced turn is counted, never silent"
    );
}

/// Pricing the live model turns the same turn into currency: both hooks charge
/// micros, and the total matches the model's own breakdown bucket.
#[tokio::test]
async fn live_turn_charges_currency_when_the_model_is_priced() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let cost = Arc::new(
        CostPlugin::new(CostConfig {
            pricing: PricingTable::new().with_model(model.clone(), rates()),
            ..Default::default()
        })
        .expect("cost plugin must build"),
    );
    let client = common::client_with_plugins(vec![Arc::clone(&cost) as Arc<dyn CucaPlugin>]);

    let stream = client
        .generate_stream(common::live_request(
            "Reply with the single word: ok",
            &model,
        ))
        .await
        .expect("generate_stream must start");
    common::drain_timeout(stream, 60).await;

    let usage = cost.usage().expect("ledger lock must not be poisoned");
    assert!(
        usage.spent_micros > 0,
        "a priced model must charge currency, got {usage:?}"
    );
    assert_eq!(
        usage.unpriced_turns, 0,
        "the model is priced, got {usage:?}"
    );

    let breakdown = cost.breakdown().expect("ledger lock must not be poisoned");
    assert_eq!(
        breakdown.len(),
        1,
        "one model was charged, got {breakdown:?}"
    );
    assert_eq!(breakdown[0].0, model);
    assert_eq!(breakdown[0].1.spent_micros, usage.spent_micros);
}

/// A cap with no headroom refuses the turn in `on_request`, so
/// `generate_stream` fails before any provider dispatch and the ledger keeps
/// nothing. No server is needed: the refusal happens before the client opens a
/// connection, which is why this test is not gated on the live harness.
#[tokio::test]
async fn a_zero_headroom_budget_refuses_the_live_turn_before_dispatch() {
    let cost = Arc::new(
        CostPlugin::new(CostConfig {
            max_total_tokens: Some(1),
            ..Default::default()
        })
        .expect("cost plugin must build"),
    );
    let client = common::client_with_plugins(vec![Arc::clone(&cost) as Arc<dyn CucaPlugin>]);
    let before = cost.usage().expect("ledger lock must not be poisoned");

    let err = client
        .generate_stream(common::live_request("Reply with ok", "cost-model"))
        .await
        .err()
        .expect("the turn must be refused before dispatch");
    match err {
        CucaError::Plugin(PluginError::HookFailure {
            plugin,
            stage,
            message,
        }) => {
            assert_eq!(plugin, "cost-accounting");
            assert_eq!(stage, "request");
            assert!(message.contains("token budget exceeded"), "{message}");
        }
        other => panic!("expected CucaError::Plugin(HookFailure), got {other:?}"),
    }

    assert_eq!(
        cost.usage().expect("ledger lock must not be poisoned"),
        before,
        "a refused turn commits nothing"
    );
    assert!(
        cost.breakdown()
            .expect("ledger lock must not be poisoned")
            .is_empty()
    );
}

/// The near-cap warning is a real request mutation: a plugin registered after
/// the cost plugin sees the injected system message, so it is part of the
/// prompt the provider receives.
#[tokio::test]
async fn the_near_cap_warning_reaches_the_provider_prompt() {
    let dispatches = Arc::new(AtomicUsize::new(0));
    let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;

    // Half the cap is spent by this one turn, which meets a quarter-cap warning
    // threshold; the cap itself still has headroom, so the turn is not refused.
    let estimate = counting_plugin()
        .estimate_request_tokens(&mock_request())
        .expect("encoder lock must not be poisoned");
    let cost = Arc::new(
        CostPlugin::new(CostConfig {
            max_total_tokens: Some(estimate * 2),
            warn_fraction: Some(0.25),
            ..Default::default()
        })
        .expect("cost plugin must build"),
    );
    let recorder = Arc::new(RequestRecorder::default());
    let client = client_at(
        &format!("http://{addr}/v1"),
        vec![
            Arc::clone(&cost) as Arc<dyn CucaPlugin>,
            Arc::clone(&recorder) as Arc<dyn CucaPlugin>,
        ],
    );

    common::drain_timeout(
        client
            .generate_stream(mock_request())
            .await
            .expect("generate_stream must start"),
        10,
    )
    .await;

    assert!(
        cost.usage()
            .expect("ledger lock must not be poisoned")
            .near_cap
    );
    let seen = recorder.seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "one request observed, got {}", seen.len());
    let warnings: Vec<&str> = seen[0]
        .messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|block| match block {
            MessageContentBlock::Text(text) if text.starts_with("CUCA cost warning:") => {
                Some(text.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        warnings.len(),
        1,
        "exactly one warning must reach the later hook, got {:?}",
        seen[0].messages
    );
    assert_eq!(
        seen[0].messages.len(),
        mock_request().messages.len() + 1,
        "the warning is the only message the plugin adds"
    );
}

/// The ledger is cumulative across turns, and `reset` rolls it back to zero
/// without touching the configuration.
#[tokio::test]
async fn two_live_turns_accumulate_and_reset_clears_them() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let cost = Arc::new(
        CostPlugin::new(CostConfig {
            pricing: PricingTable::new().with_model(model.clone(), rates()),
            ..Default::default()
        })
        .expect("cost plugin must build"),
    );
    let client = common::client_with_plugins(vec![Arc::clone(&cost) as Arc<dyn CucaPlugin>]);

    for prompt in ["Reply with the single word: ok", "Reply with the digit: 1"] {
        let stream = client
            .generate_stream(common::live_request(prompt, &model))
            .await
            .expect("generate_stream must start");
        common::drain_timeout(stream, 60).await;
    }

    let usage = cost.usage().expect("ledger lock must not be poisoned");
    assert_eq!(usage.turns, 2, "two committed turns, got {usage:?}");
    assert!(usage.prompt_tokens > 0 && usage.spent_micros > 0);

    cost.reset().expect("ledger lock must not be poisoned");

    let cleared = cost.usage().expect("ledger lock must not be poisoned");
    assert_eq!(cleared.turns, 0);
    assert_eq!(cleared.prompt_tokens, 0);
    assert_eq!(cleared.completion_tokens, 0);
    assert_eq!(cleared.spent_micros, 0);
    assert!(
        cost.breakdown()
            .expect("ledger lock must not be poisoned")
            .is_empty()
    );
    assert_eq!(
        cost.rates_for(&model),
        Some(rates()),
        "reset rolls the ledger, not the configuration"
    );
}

/// The plugin adds no dispatch of its own: an accepted turn dispatches exactly
/// once, and a refused turn dispatches not at all.
#[tokio::test]
async fn dispatch_count_is_unchanged_by_the_plugin() {
    let dispatches = Arc::new(AtomicUsize::new(0));
    let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
    let addr = format!("http://{addr}/v1");

    let accepted = counting_plugin();
    let client = client_at(&addr, vec![Arc::clone(&accepted) as Arc<dyn CucaPlugin>]);
    common::drain_timeout(
        client
            .generate_stream(mock_request())
            .await
            .expect("generate_stream must start"),
        10,
    )
    .await;
    assert_eq!(
        dispatches.load(Ordering::SeqCst),
        1,
        "one accepted turn, one dispatch"
    );
    assert_eq!(
        accepted
            .usage()
            .expect("ledger lock must not be poisoned")
            .turns,
        1
    );

    let refusing = Arc::new(
        CostPlugin::new(CostConfig {
            max_total_tokens: Some(1),
            ..Default::default()
        })
        .expect("cost plugin must build"),
    );
    let capped = client_at(&addr, vec![Arc::clone(&refusing) as Arc<dyn CucaPlugin>]);
    assert!(
        capped.generate_stream(mock_request()).await.is_err(),
        "the capped turn must be refused"
    );
    assert_eq!(
        dispatches.load(Ordering::SeqCst),
        1,
        "a cap refusal never reaches the provider"
    );
}
