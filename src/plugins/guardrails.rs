//! JSON Schema output guardrails and bounded retry injection.
//!
//! [`JsonGuardrailPlugin`] validates model tool arguments and response payloads
//! against caller-registered JSON Schemas before those values commit. When a
//! value fails its schema the plugin does not error the stream; instead it
//! replaces the offending block with a `ToolResult` (or `Text`) carrying a
//! formatted error JSON and re-injects it, so the model sees the diagnostic and
//! retries on its next turn. Retries are bounded by `max_attempts`; past the
//! cap the plugin emits a hard `"guardrail_exhausted"` result and stops
//! allowing further retries (the documented self-correcting-loop bound).
//!
//! Every retry injection also emits a `tracing::warn!` structural event
//! (`target: "cuca::guardrails"`) carrying `schema_name`, `error_type`, and
//! `attempt_count`. Full OTel metric emission remains plugin-telemetry's
//! concern; the plugin only records the most recent event tuple for test
//! assertions (see [`JsonGuardrailPlugin::last_retry_event`]).
//!
//! Schemas are keyed by tool name; the reserved `"response"` key (when
//! registered) guards model text responses that look like JSON objects. A tool
//! with no registered schema passes through untouched: guardrails apply only
//! where schemas exist.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::types::MessageContentBlock;

/// JSON Schema validation + bounded retry-injection plugin.
///
/// `Send + Sync` via the [`CucaPlugin`] supertrait, so an instance can be shared
/// as `Arc<dyn CucaPlugin>` across `await` points in the client pipeline. All
/// hook bodies return `Ok(())`: validation failures are re-injected into the
/// stream rather than surfaced as plugin errors.
///
/// # Growth
///
/// The only state that grows with traffic is the per-tool-call attempt
/// counter, capped at [`Self::MAX_TRACKED_CALLS`] entries. At the cap the
/// oldest tracked id is evicted in insertion order: tool-call ids are unique
/// per call, so an evicted id belongs to a call that is long finished, and a
/// (practically impossible) reuse of that id simply starts back at attempt 1.
pub struct JsonGuardrailPlugin {
    /// Tool name (or the reserved `"response"` key) -> compiled JSON Schema
    /// validator. Tools absent from this map pass through unvalidated. Fixed
    /// at construction.
    schemas: HashMap<String, jsonschema::Validator>,
    /// Per-tool-call attempt counters, bounded by [`Self::MAX_TRACKED_CALLS`].
    attempts: Mutex<AttemptCounters>,
    /// Upper bound on retries per tool call before `"guardrail_exhausted"`.
    max_attempts: u32,
    /// Most recent `(schema_name, error_type, attempt_count)` retry event,
    /// mirroring the `tracing` event emitted alongside it. Read through
    /// [`Self::last_retry_event`].
    last_retry_event: Mutex<Option<(String, String, u32)>>,
}

/// Mutex-guarded attempt counters: the map plus the insertion order that makes
/// eviction at the cap deterministic (the `lru_order` shape `PromptCache`
/// uses).
#[derive(Default)]
struct AttemptCounters {
    /// Tool-call id -> current attempt count. Incremented on each invalid
    /// re-injection, cleared on a valid pass-through so a later failure starts
    /// back at attempt 1.
    counts: HashMap<String, u32>,
    /// Tracked ids in first-failure order; the front is evicted at the cap.
    order: VecDeque<String>,
}

impl JsonGuardrailPlugin {
    /// Loads a JSON Schema document from `schema_path`.
    ///
    /// The document holds a JSON object mapping tool name -> JSON Schema. Each
    /// value is compiled into a validator.
    ///
    /// The file-loading path uses a bounded retry default of 3
    /// (`max_attempts = 3`); callers wanting an explicit bound use
    /// [`Self::with_schemas`].
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Io`] if the file cannot be opened or read, and
    /// [`PluginError::Validation`] if the document is not a valid JSON object or
    /// any embedded schema fails to compile.
    pub fn new(schema_path: impl AsRef<std::path::Path>) -> Result<Self, PluginError> {
        let path = schema_path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| PluginError::Io(e.to_string()))?;
        let schemas: HashMap<String, serde_json::Value> =
            serde_json::from_reader(file).map_err(|e| PluginError::Validation {
                schema: path.display().to_string(),
                message: format!("failed to parse schema document: {e}"),
            })?;
        Self::compile(schemas, 3)
    }

    /// Programmatic schema registration without a file (tests, in-code config).
    ///
    /// `max_attempts` bounds how many times a failing tool call may be
    /// re-injected before the plugin emits `"guardrail_exhausted"`.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Validation`] if any embedded schema fails to
    /// compile.
    pub fn with_schemas(
        schemas: HashMap<String, serde_json::Value>,
        max_attempts: u32,
    ) -> Result<Self, PluginError> {
        Self::compile(schemas, max_attempts)
    }

    /// Compiles the schema map into validators and assembles the plugin.
    ///
    /// `max_attempts` bounds the retry loop: [`Self::new`] supplies a sensible
    /// default of 3 for the file-loading path, while [`Self::with_schemas`]
    /// lets callers pass an explicit limit.
    fn compile(
        schemas: HashMap<String, serde_json::Value>,
        max_attempts: u32,
    ) -> Result<Self, PluginError> {
        let mut compiled = HashMap::with_capacity(schemas.len());
        for (name, schema) in schemas {
            let validator =
                jsonschema::validator_for(&schema).map_err(|e| PluginError::Validation {
                    schema: name.clone(),
                    message: e.to_string(),
                })?;
            compiled.insert(name, validator);
        }
        Ok(Self {
            schemas: compiled,
            attempts: Mutex::new(AttemptCounters::default()),
            max_attempts,
            last_retry_event: Mutex::new(None),
        })
    }

    /// Maximum number of tool-call ids whose attempt counters are retained.
    ///
    /// See the type's *Growth* section for the eviction policy.
    pub const MAX_TRACKED_CALLS: usize = 4096;

    /// Number of tool-call ids currently tracked: the O(1) usage gauge against
    /// [`Self::MAX_TRACKED_CALLS`].
    pub fn tracked_calls(&self) -> usize {
        self.attempts
            .lock()
            .map_or(0, |attempts| attempts.counts.len())
    }

    /// Validates `value` against the schema registered for `tool_name`.
    ///
    /// A tool with no registered schema returns `Ok(())` (pass-through:
    /// guardrails apply only where schemas exist). On failure, returns `Err`
    /// with one human-readable issue string per schema violation.
    pub fn validate(&self, tool_name: &str, value: &serde_json::Value) -> Result<(), Vec<String>> {
        let Some(validator) = self.schemas.get(tool_name) else {
            return Ok(());
        };
        let issues: Vec<String> = validator
            .iter_errors(value)
            .map(|e| e.to_string())
            .collect();
        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    /// Most recent `(schema_name, error_type, attempt_count)` retry event.
    ///
    /// Test-support accessor: lets tests assert the emitted retry event
    /// without installing a tracing subscriber. `None` until the first retry
    /// injection.
    pub fn last_retry_event(&self) -> Option<(String, String, u32)> {
        self.last_retry_event
            .lock()
            .ok()
            .and_then(|event| event.clone())
    }
}

impl CucaPlugin for JsonGuardrailPlugin {
    /// Stable plugin name: `"json-guardrails"` is the fixed identifier clients
    /// use to address this plugin.
    fn name(&self) -> &'static str {
        "json-guardrails"
    }

    /// Per-block guardrail pass over streamed content.
    ///
    /// `ToolCall` blocks are validated against the schema keyed by the tool
    /// name; on failure the block is replaced with a `ToolResult` carrying the
    /// formatted error JSON (bounded by `max_attempts`). `Text` blocks that
    /// look like a JSON object (leading `{`) are validated against the reserved
    /// `"response"` schema when one is registered, and invalid text is replaced
    /// with the re-injected error text. All other blocks pass through. Always
    /// returns `Ok(())`: validation failures are re-injected, not errors.
    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        match chunk {
            MessageContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                match self.validate(name, arguments) {
                    Ok(()) => {
                        let mut attempts = self.attempts.lock().map_err(|_| {
                            PluginError::Internal("guardrail attempts lock poisoned".into())
                        })?;
                        if attempts.counts.remove(id).is_some() {
                            attempts.order.retain(|tracked| tracked != id);
                        }
                    }
                    Err(issues) => {
                        let mut attempts = self.attempts.lock().map_err(|_| {
                            PluginError::Internal("guardrail attempts lock poisoned".into())
                        })?;
                        let n = match attempts.counts.get_mut(id) {
                            Some(count) => {
                                *count += 1;
                                *count
                            }
                            None => {
                                // First failure for this id: evict the oldest
                                // tracked id when the map is at its cap, so the
                                // counters cannot grow with traffic.
                                while attempts.order.len() >= Self::MAX_TRACKED_CALLS {
                                    if let Some(oldest) = attempts.order.pop_front() {
                                        attempts.counts.remove(&oldest);
                                    }
                                }
                                attempts.counts.insert(id.clone(), 1);
                                attempts.order.push_back(id.clone());
                                1
                            }
                        };
                        drop(attempts);
                        let error = if n <= self.max_attempts {
                            "schema_validation_failed"
                        } else {
                            "guardrail_exhausted"
                        };
                        let output =
                            serde_json::json!({ "error": error, "tool": name, "issues": issues })
                                .to_string();
                        tracing::warn!(target: "cuca::guardrails", schema_name = %name, error_type = error, attempt_count = n, "guardrail result injected");
                        // `name` and `id` are moved out rather than cloned:
                        // the assignment below replaces the block they were
                        // borrowed from, so neither is read again.
                        *self.last_retry_event.lock().map_err(|_| {
                            PluginError::Internal("guardrail event lock poisoned".into())
                        })? = Some((std::mem::take(name), error.to_string(), n));
                        *chunk = MessageContentBlock::ToolResult {
                            tool_call_id: std::mem::take(id),
                            output,
                        };
                    }
                }
                Ok(())
            }
            MessageContentBlock::Text(text) if text.trim_start().starts_with('{') => {
                match serde_json::from_str::<serde_json::Value>(text) {
                    Ok(value) => match self.validate("response", &value) {
                        Ok(()) => Ok(()),
                        Err(issues) => {
                            *text = serde_json::json!({ "error": "schema_validation_failed", "tool": "response", "issues": issues }).to_string();
                            Ok(())
                        }
                    },
                    Err(_) => Ok(()),
                }
            }
            _ => Ok(()),
        }
    }

    /// Terminal no-op: this plugin's work is entirely per-block.
    fn on_response_complete(
        &self,
        _res: &crate::request::UnifiedResponse,
    ) -> Result<(), PluginError> {
        Ok(())
    }
}

#[cfg(all(test, feature = "plugin-guardrails"))]
mod tests {
    use super::*;

    fn schema() -> HashMap<String, serde_json::Value> {
        HashMap::from([(
            "get_weather".to_string(),
            serde_json::json!({
                "type": "object",
                "required": ["city"],
                "properties": { "city": { "type": "string" } }
            }),
        )])
    }

    #[test]
    fn valid_tool_call_passes_through_unchanged() {
        let plugin = JsonGuardrailPlugin::with_schemas(schema(), 3).unwrap();
        let mut block = MessageContentBlock::ToolCall {
            id: "call-1".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "city": "Berlin" }),
        };
        plugin.on_stream_chunk(&mut block).unwrap();
        assert!(matches!(
            block,
            MessageContentBlock::ToolCall { ref name, .. } if name == "get_weather"
        ));
    }

    #[test]
    fn invalid_tool_call_replaced_with_retry_result() {
        let plugin = JsonGuardrailPlugin::with_schemas(schema(), 3).unwrap();
        let mut block = MessageContentBlock::ToolCall {
            id: "call-2".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "temp": 1 }),
        };
        plugin.on_stream_chunk(&mut block).unwrap();
        let MessageContentBlock::ToolResult {
            tool_call_id,
            output,
        } = block
        else {
            panic!("expected ToolResult");
        };
        assert_eq!(tool_call_id, "call-2");
        assert!(output.contains("schema_validation_failed"));
        // The missing required field should be reported as an issue.
        assert!(output.contains("required") || output.contains("city"));
    }

    #[test]
    fn repeated_injection_exhausts_attempts() {
        let plugin = JsonGuardrailPlugin::with_schemas(schema(), 2).unwrap();
        // Each iteration re-injects a fresh ToolCall carrying the same id, so the
        // plugin's per-id attempt counter accumulates across them.
        for _ in 0..2 {
            let mut block = MessageContentBlock::ToolCall {
                id: "call-3".to_string(),
                name: "get_weather".to_string(),
                arguments: serde_json::json!({ "temp": 1 }),
            };
            plugin.on_stream_chunk(&mut block).unwrap();
            // Attempts 1 and 2 (<= cap 2) -> retry, not yet exhausted.
            assert!(matches!(
                block,
                MessageContentBlock::ToolResult { ref output, .. }
                    if output.contains("schema_validation_failed")
            ));
        }
        // Third injection of the same id (attempt 3 > cap 2) -> exhausted.
        let mut block = MessageContentBlock::ToolCall {
            id: "call-3".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "temp": 1 }),
        };
        plugin.on_stream_chunk(&mut block).unwrap();
        assert!(matches!(
            block,
            MessageContentBlock::ToolResult { ref output, .. } if output.contains("guardrail_exhausted")
        ));
    }

    #[test]
    fn valid_pass_through_clears_attempt_counter() {
        let plugin = JsonGuardrailPlugin::with_schemas(schema(), 1).unwrap();
        let mut block = MessageContentBlock::ToolCall {
            id: "call-4".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "temp": 1 }),
        };
        // First failure: retry, attempt 1.
        plugin.on_stream_chunk(&mut block).unwrap();
        assert!(matches!(
            block,
            MessageContentBlock::ToolResult { ref output, .. }
                if output.contains("schema_validation_failed")
        ));
        // Now a valid call with the same id passes through and clears the counter.
        block = MessageContentBlock::ToolCall {
            id: "call-4".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "city": "Berlin" }),
        };
        plugin.on_stream_chunk(&mut block).unwrap();
        assert!(matches!(
            block,
            MessageContentBlock::ToolCall { ref name, .. } if name == "get_weather"
        ));
        // A later failure with the same id starts back at attempt 1, so with
        // max_attempts = 1 it is again a retry, not exhausted.
        block = MessageContentBlock::ToolCall {
            id: "call-4".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "temp": 1 }),
        };
        plugin.on_stream_chunk(&mut block).unwrap();
        assert!(matches!(
            block,
            MessageContentBlock::ToolResult { ref output, .. }
                if output.contains("schema_validation_failed")
        ));
    }

    #[test]
    fn attempt_counters_stop_growing_at_the_tracked_call_cap() {
        let plugin = JsonGuardrailPlugin::with_schemas(schema(), 3).unwrap();
        // One first-failure per distinct id, one past the cap.
        for i in 0..=JsonGuardrailPlugin::MAX_TRACKED_CALLS {
            let mut block = MessageContentBlock::ToolCall {
                id: format!("call-{i}"),
                name: "get_weather".to_string(),
                arguments: serde_json::json!({ "temp": 1 }),
            };
            plugin.on_stream_chunk(&mut block).unwrap();
        }
        assert_eq!(
            plugin.tracked_calls(),
            JsonGuardrailPlugin::MAX_TRACKED_CALLS,
            "the counter map is capped, not unbounded"
        );
    }

    #[test]
    fn evicted_tool_call_id_starts_back_at_the_first_attempt() {
        let plugin = JsonGuardrailPlugin::with_schemas(schema(), 1).unwrap();
        let invalid = |id: String| MessageContentBlock::ToolCall {
            id,
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "temp": 1 }),
        };
        // `call-0` is the oldest tracked id, so filling the map past the cap
        // evicts exactly it.
        for i in 0..=JsonGuardrailPlugin::MAX_TRACKED_CALLS {
            let mut block = invalid(format!("call-{i}"));
            plugin.on_stream_chunk(&mut block).unwrap();
        }
        let mut block = invalid("call-0".to_string());
        plugin.on_stream_chunk(&mut block).unwrap();
        assert!(
            matches!(
                &block,
                MessageContentBlock::ToolResult { output, .. }
                    if output.contains("schema_validation_failed")
            ),
            "an evicted id retries from attempt 1 instead of reporting exhaustion: {block:?}"
        );
    }

    #[test]
    fn last_retry_event_tracks_retry_and_exhaustion() {
        let plugin = JsonGuardrailPlugin::with_schemas(schema(), 1).unwrap();
        // First failure: retry, attempt 1.
        let mut block = MessageContentBlock::ToolCall {
            id: "call-5".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "temp": 1 }),
        };
        plugin.on_stream_chunk(&mut block).unwrap();
        assert_eq!(
            plugin.last_retry_event(),
            Some((
                "get_weather".to_string(),
                "schema_validation_failed".to_string(),
                1
            ))
        );
        // Second injection of the same id exceeds cap 1 -> exhausted, attempt 2.
        let mut block = MessageContentBlock::ToolCall {
            id: "call-5".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "temp": 1 }),
        };
        plugin.on_stream_chunk(&mut block).unwrap();
        assert_eq!(
            plugin.last_retry_event(),
            Some((
                "get_weather".to_string(),
                "guardrail_exhausted".to_string(),
                2
            ))
        );
    }

    #[test]
    fn validate_returns_expected_results() {
        let plugin = JsonGuardrailPlugin::with_schemas(schema(), 3).unwrap();
        // Valid input.
        assert_eq!(
            plugin.validate("get_weather", &serde_json::json!({ "city": "Berlin" })),
            Ok(())
        );
        // Invalid input yields issue strings.
        match plugin.validate("get_weather", &serde_json::json!({ "temp": 1 })) {
            Err(issues) => {
                assert!(!issues.is_empty());
                assert!(issues.iter().any(|i| i.contains("city")));
            }
            Ok(()) => panic!("expected validation error"),
        }
        // Unknown schema name passes through.
        assert_eq!(
            plugin.validate("no_such_tool", &serde_json::json!({ "temp": 1 })),
            Ok(())
        );
    }

    #[test]
    fn response_text_guardrail_reinjects_invalid_json() {
        let mut schemas = schema();
        schemas.insert(
            "response".to_string(),
            serde_json::json!({
                "type": "object",
                "required": ["city"],
                "properties": { "city": { "type": "string" } }
            }),
        );
        let plugin = JsonGuardrailPlugin::with_schemas(schemas, 3).unwrap();

        // Object-looking JSON that fails the response schema -> re-injected error text.
        let mut block = MessageContentBlock::Text("{\"temp\": 1}".to_string());
        plugin.on_stream_chunk(&mut block).unwrap();
        let MessageContentBlock::Text(text) = block else {
            panic!("expected Text");
        };
        assert!(text.contains("schema_validation_failed"));
        assert!(text.contains("city"));

        // Valid object text passes through unchanged.
        let mut block = MessageContentBlock::Text("{\"city\": \"Berlin\"}".to_string());
        plugin.on_stream_chunk(&mut block).unwrap();
        let MessageContentBlock::Text(text) = block else {
            panic!("expected Text");
        };
        assert_eq!(text, "{\"city\": \"Berlin\"}");
    }

    #[test]
    fn name_is_stable_and_type_is_send_sync() {
        let plugin = JsonGuardrailPlugin::with_schemas(schema(), 3).unwrap();
        assert_eq!(plugin.name(), "json-guardrails");

        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<JsonGuardrailPlugin>();
    }

    #[test]
    fn new_missing_path_is_error() {
        let err = JsonGuardrailPlugin::new("./definitely-missing-schemas.json");
        assert!(matches!(
            err,
            Err(PluginError::Io(_)) | Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn new_file_loaded_path_has_bounded_retries() {
        // Write a temp schema document, load it via new(), and confirm the
        // file path's default max_attempts = 3 yields a retry loop (first
        // failure -> "schema_validation_failed"), exhausting only on the 4th.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("cuca-guardrails-test-{}.json", std::process::id()));
        std::fs::write(
            &path,
            serde_json::json!({
                "get_weather": {
                    "type": "object",
                    "required": ["city"],
                    "properties": { "city": { "type": "string" } }
                }
            })
            .to_string(),
        )
        .unwrap();
        let plugin = JsonGuardrailPlugin::new(&path).unwrap();

        // Attempts 1-3 (<= default cap 3) -> retry, not exhausted.
        for _ in 0..3 {
            let mut block = MessageContentBlock::ToolCall {
                id: "call-file".to_string(),
                name: "get_weather".to_string(),
                arguments: serde_json::json!({ "temp": 1 }),
            };
            plugin.on_stream_chunk(&mut block).unwrap();
            assert!(matches!(
                block,
                MessageContentBlock::ToolResult { ref output, .. }
                    if output.contains("schema_validation_failed")
            ));
        }
        // 4th injection of the same id (attempt 4 > cap 3) -> exhausted.
        let mut block = MessageContentBlock::ToolCall {
            id: "call-file".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "temp": 1 }),
        };
        plugin.on_stream_chunk(&mut block).unwrap();
        assert!(matches!(
            block,
            MessageContentBlock::ToolResult { ref output, .. } if output.contains("guardrail_exhausted")
        ));

        let _ = std::fs::remove_file(&path);
    }
}
