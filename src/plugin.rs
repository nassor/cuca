//! The "everything-is-a-plugin" seam: [`CucaPlugin`] and [`SessionStorePlugin`].
//!
//! [`CucaPlugin`] is the base hook surface every feature-gated capability
//! implements; [`SessionStorePlugin`] extends it with the append-only session
//! trajectory operations. Clients hold plugins as `Vec<Arc<dyn CucaPlugin>>`
//! and invoke the hooks around every request/stream.
//!
//! # Design notes
//!
//! The default hook bodies return `Ok(())` and ignore their inputs. The
//! telemetry and guardrails plugins implement only subsets of the hooks, so
//! the defaults keep partial implementations ergonomic; behavior is identical
//! whenever a hook is overridden.
//!
//! `Send + Sync` is a supertrait so the plugin list,
//! `Vec<Arc<dyn CucaPlugin>>`, can be shared across `await` points in the
//! async client pipeline.

use crate::error::PluginError;
use crate::request::{UnifiedRequest, UnifiedResponse};
use crate::session::SessionRecord;
use crate::types::MessageContentBlock;

/// A feature-gated agentic capability. Instances are registered on the client
/// builder as `Arc<dyn CucaPlugin>` and invoked around every request/stream.
pub trait CucaPlugin: Send + Sync {
    /// Stable, unique plugin name (e.g. "opentelemetry-observability").
    fn name(&self) -> &'static str;

    /// Pre-dispatch hook: mutate the unified request (inject context, enforce
    /// policies, count tokens). Runs in registration order.
    ///
    /// # Errors
    ///
    /// Return `Err` to refuse the request: the client wraps it in
    /// [`crate::error::CucaError::Plugin`] and `generate_stream` fails before
    /// any provider dispatch, so later plugins' hooks never run.
    fn on_request(&self, _req: &mut UnifiedRequest) -> Result<(), PluginError> {
        Ok(())
    }

    /// Optionally execute a streamed tool call locally before stream hooks run.
    ///
    /// The client invokes this only for [`MessageContentBlock::ToolCall`] blocks.
    /// A replacement must be a `ToolResult` for the same call id.
    ///
    /// # Errors
    ///
    /// Return `Err` when the call was this plugin's to run and it failed: the
    /// client yields it as [`crate::error::CucaError::Plugin`] in place of the
    /// block and keeps polling the provider stream. A tool that is not this
    /// plugin's concern must return `Ok(None)`, not an error.
    fn execute_local_tool(
        &self,
        _call: &MessageContentBlock,
    ) -> Result<Option<MessageContentBlock>, PluginError> {
        Ok(None)
    }

    /// Per-block hook on every streamed `MessageContentBlock`. A plugin may
    /// replace the block (e.g. guardrails swap an invalid ToolCall for a
    /// ToolResult carrying the validation error).
    ///
    /// # Errors
    ///
    /// Return `Err` to reject the block: the client yields
    /// [`crate::error::CucaError::Plugin`] instead of it, the block is neither
    /// accumulated into the response nor token-counted, later plugins do not
    /// see it, and polling continues with the next block. Recoverable
    /// conditions belong in the block itself (the guardrails and tool plugins
    /// re-inject a `ToolResult`) rather than in an error.
    fn on_stream_chunk(&self, _chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        Ok(())
    }

    /// Terminal hook after the stream completes (telemetry, session logging).
    ///
    /// # Errors
    ///
    /// Return `Err` to report a bookkeeping failure. The stream has already
    /// ended, so the client logs it and moves on to the next plugin: a failure
    /// here never surfaces to the consumer and never suppresses another
    /// plugin's terminal hook.
    fn on_response_complete(&self, _res: &UnifiedResponse) -> Result<(), PluginError> {
        Ok(())
    }
}

/// Plugins that maintain the append-only session trajectory and support forking.
pub trait SessionStorePlugin: CucaPlugin {
    /// Append one record to the session's trajectory (must be append-only).
    ///
    /// # Errors
    ///
    /// Implementation-defined: a backend that is full, unreachable, or handed
    /// an unusable `session_id` returns the matching [`PluginError`].
    fn append_log(&self, session_id: &str, record: &SessionRecord) -> Result<(), PluginError>;
    /// Replay the full trajectory in append order.
    ///
    /// # Errors
    ///
    /// Implementation-defined: an unreadable or corrupt trajectory returns the
    /// matching [`PluginError`]. An unknown `session_id` replays as empty
    /// rather than failing.
    fn replay_session(&self, session_id: &str) -> Result<Vec<SessionRecord>, PluginError>;
    /// Fork from a historical `point_id`; returns the NEW session id. The new
    /// trajectory contains the prefix up to and including `point_id`.
    ///
    /// # Errors
    ///
    /// Implementation-defined: a `point_id` that names no recorded position,
    /// or a backend that cannot record the branch, returns the matching
    /// [`PluginError`].
    fn fork_session(&self, session_id: &str, point_id: &str) -> Result<String, PluginError>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::Mutex;

    use super::*;
    use crate::session::SessionEvent;
    use crate::types::{ProviderEndpoint, UnifiedMessage};

    #[test]
    fn default_hooks_are_noops_that_leave_inputs_untouched() {
        // A plugin overriding only `name()`: every hook falls back to the
        // default `Ok(())` body, which must not modify the borrowed inputs.
        struct NameOnly;
        impl CucaPlugin for NameOnly {
            fn name(&self) -> &'static str {
                "name-only"
            }
        }

        let plugin = NameOnly;
        assert_eq!(plugin.name(), "name-only");

        let mut req = UnifiedRequest::new("gpt-4o")
            .add_user_message("hello")
            .set_temperature(0.5)
            .set_max_tokens(128);
        let before = req.clone();
        plugin.on_request(&mut req).unwrap();
        assert_eq!(req, before);

        let mut chunk = MessageContentBlock::Text("hi".to_string());
        let chunk_before = chunk.clone();
        plugin.on_stream_chunk(&mut chunk).unwrap();
        assert_eq!(chunk, chunk_before);

        let call = MessageContentBlock::ToolCall {
            id: "call_1".into(),
            name: "read_tool_result".into(),
            arguments: serde_json::json!({}),
        };
        assert_eq!(plugin.execute_local_tool(&call).unwrap(), None);

        let res = UnifiedResponse {
            model: "gpt-4o".into(),
            provider: ProviderEndpoint::OpenAi,
            duration_secs: 1.0,
            prompt_tokens: 10,
            completion_tokens: 20,
            finish_reason: Some("stop".into()),
            content: vec![MessageContentBlock::Text("answer".into())],
            prompt_cache_usage: None,
        };
        plugin.on_response_complete(&res).unwrap();
    }

    #[test]
    fn full_mock_runs_hooks_in_order_with_mutating_effects() {
        // Interior mutability lets a `&self` hook record its invocations and
        // effects across the three call sites. `tokio::sync::Mutex` returns
        // guards directly (no poisoning/unwrap), and the mock is only driven
        // synchronously here via `blocking_lock`.
        struct RecordingPlugin {
            calls: Mutex<Vec<&'static str>>,
            completed_models: Mutex<Vec<String>>,
        }
        impl RecordingPlugin {
            fn new() -> Self {
                Self {
                    calls: Mutex::new(Vec::new()),
                    completed_models: Mutex::new(Vec::new()),
                }
            }

            fn record(&self, call: &'static str) {
                self.calls.blocking_lock().push(call);
            }

            fn record_model(&self, model: String) {
                self.completed_models.blocking_lock().push(model);
            }
        }
        impl CucaPlugin for RecordingPlugin {
            fn name(&self) -> &'static str {
                "recording"
            }

            fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
                self.record("on_request");
                req.messages.push(UnifiedMessage::user("injected-by-hook"));
                Ok(())
            }

            fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
                self.record("on_stream_chunk");
                *chunk = MessageContentBlock::Text("guarded".to_string());
                Ok(())
            }

            fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
                self.record("on_response_complete");
                self.record_model(res.model.clone());
                Ok(())
            }
        }

        let plugin = RecordingPlugin::new();

        let mut req = UnifiedRequest::new("gpt-4o").add_user_message("hi");
        plugin.on_request(&mut req).unwrap();
        assert_eq!(req.message_count(), 2);
        assert_eq!(
            req.messages[1].content[0],
            MessageContentBlock::Text("injected-by-hook".to_string())
        );

        let mut chunk = MessageContentBlock::Text("raw".to_string());
        plugin.on_stream_chunk(&mut chunk).unwrap();
        assert_eq!(chunk, MessageContentBlock::Text("guarded".to_string()));

        let res = UnifiedResponse {
            model: "gpt-4o".into(),
            provider: ProviderEndpoint::OpenAi,
            duration_secs: 1.0,
            prompt_tokens: 10,
            completion_tokens: 20,
            finish_reason: Some("stop".into()),
            content: vec![MessageContentBlock::Text("answer".into())],
            prompt_cache_usage: None,
        };
        plugin.on_response_complete(&res).unwrap();

        assert_eq!(
            plugin.calls.blocking_lock().clone(),
            vec!["on_request", "on_stream_chunk", "on_response_complete"]
        );
        assert_eq!(
            plugin.completed_models.blocking_lock().clone(),
            vec!["gpt-4o".to_string()]
        );
    }

    #[test]
    fn trait_objects_and_plugin_list_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        // Box the trait objects to get sized type arguments for the helper.
        assert_send_sync::<Box<dyn CucaPlugin>>();
        assert_send_sync::<Box<dyn SessionStorePlugin>>();
        assert_send_sync::<Vec<Arc<dyn CucaPlugin>>>();
    }

    #[test]
    fn session_store_plugin_bound_holds_and_fork_returns_new_id() {
        // `SessionStorePlugin: CucaPlugin` requires a mock to implement both;
        // the `S: SessionStorePlugin` bound below then proves the relationship.
        struct StoreMock;
        impl CucaPlugin for StoreMock {
            fn name(&self) -> &'static str {
                "store-mock"
            }
        }
        impl SessionStorePlugin for StoreMock {
            fn append_log(
                &self,
                _session_id: &str,
                _record: &SessionRecord,
            ) -> Result<(), PluginError> {
                Ok(())
            }

            fn replay_session(&self, _session_id: &str) -> Result<Vec<SessionRecord>, PluginError> {
                Ok(vec![])
            }

            fn fork_session(
                &self,
                _session_id: &str,
                _point_id: &str,
            ) -> Result<String, PluginError> {
                Ok("forked-session-1".to_string())
            }
        }

        fn takes_store<S: SessionStorePlugin>(_s: &S) {}
        let store = StoreMock;
        takes_store(&store);

        let record = SessionRecord::at(
            "s1",
            0,
            1_000,
            SessionEvent::Output {
                text: "hi".to_string(),
            },
        );
        store.append_log("s1", &record).unwrap();
        assert!(store.replay_session("s1").unwrap().is_empty());
        assert_eq!(
            store.fork_session("s1", "pt-3").unwrap(),
            "forked-session-1"
        );
    }
}
