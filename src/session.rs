//! Append-only session audit-trail model.
//!
//! One serializable [`SessionRecord`] per logged event (system instructions,
//! reasoning, outputs, tool executions, model swaps, latency, token usage, and
//! forks). Storage, replay, and forking behavior are owned by the session-log
//! plugin; this module only defines the record type.

use crate::types::{MessageContentBlock, MessageRole};

/// One auditable interaction event (append-only trajectory).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    /// System instructions / contextual conditioning / dynamic prompt injection.
    SystemPrompt {
        /// The system prompt text.
        text: String,
    },
    /// A full user/assistant message (reasoning, tool calls, outputs).
    Message {
        /// The speaker.
        role: MessageRole,
        /// The ordered content blocks making up the message.
        content: Vec<MessageContentBlock>,
    },
    /// Model reasoning/thinking block.
    Reasoning {
        /// The reasoning text.
        reasoning: String,
        /// An optional provider signature authenticating the reasoning.
        signature: Option<String>,
    },
    /// Raw generation output delta.
    Output {
        /// The generated text delta.
        text: String,
    },
    /// Executed tool request.
    ToolCall {
        /// Provider-assigned id of the call.
        id: String,
        /// Name of the tool being invoked.
        name: String,
        /// JSON-encoded arguments passed to the tool.
        arguments: serde_json::Value,
    },
    /// Tool result incl. streams and exit code.
    ToolResult {
        /// Id of the tool call this result answers.
        tool_call_id: String,
        /// String output produced by the tool.
        output: String,
        /// Captured stdout of the tool process, if any.
        stdout: Option<String>,
        /// Captured stderr of the tool process, if any.
        stderr: Option<String>,
        /// Process exit code, if the tool terminated with one.
        exit_code: Option<i32>,
    },
    /// Fast/slow model swap trigger.
    ModelSwap {
        /// The model being swapped away from.
        from: String,
        /// The model being swapped to.
        to: String,
        /// Why the swap was triggered.
        reason: String,
    },
    /// Latency measurement.
    Latency {
        /// Duration of the measured operation, in milliseconds.
        duration_ms: u64,
    },
    /// Token accounting for a generation.
    TokenUsage {
        /// Prompt tokens consumed.
        prompt_tokens: u32,
        /// Completion tokens produced.
        completion_tokens: u32,
    },
    /// Trajectory fork: `from_point` = the forked record id, `to_session` = new session.
    Fork {
        /// The `point_id()` of the record the new session forked from.
        from_point: String,
        /// The id of the newly forked session.
        to_session: String,
    },
}

/// A single append-only audit record: one logged event at a moment in time.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SessionRecord {
    /// The id of the session this record belongs to.
    pub session_id: String,
    /// 0-based append order within the session.
    pub sequence: u64,
    /// Epoch millis at which the event occurred.
    pub timestamp_ms: u64,
    /// The audited event itself.
    pub event: SessionEvent,
}

impl SessionRecord {
    /// Build a record with fully explicit positioning fields.
    pub fn at(
        session_id: impl Into<String>,
        sequence: u64,
        timestamp_ms: u64,
        event: SessionEvent,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            sequence,
            timestamp_ms,
            event,
        }
    }

    /// Convenience constructor with `sequence: 0` and `timestamp_ms` from
    /// `SystemTime::now()`. The store re-sequences on append, so the caller need
    /// not manage ordering here.
    pub fn new(session_id: impl Into<String>, event: SessionEvent) -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self::at(session_id, 0, timestamp_ms, event)
    }

    /// The stable point identifier `"<session_id>:<sequence>"`.
    ///
    /// This is the string `fork_session` takes as `point_id`.
    pub fn point_id(&self) -> String {
        format!("{}:{}", self.session_id, self.sequence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn round_trip<
        T: serde::Serialize + for<'de> serde::Deserialize<'de> + PartialEq + std::fmt::Debug,
    >(
        value: &T,
    ) {
        let json = serde_json::to_value(value).expect("serialize should succeed");
        let back: T = serde_json::from_value(json).expect("deserialize should succeed");
        assert_eq!(*value, back);
    }

    #[test]
    fn system_prompt_round_trips() {
        round_trip(&SessionEvent::SystemPrompt {
            text: "You are CUCA.".into(),
        });
        assert_eq!(
            serde_json::to_value(&SessionEvent::SystemPrompt { text: "hi".into() })
                .expect("serialize should succeed"),
            json!({ "type": "system_prompt", "text": "hi" })
        );
    }

    #[test]
    fn message_round_trips() {
        let event = SessionEvent::Message {
            role: MessageRole::User,
            content: vec![MessageContentBlock::Text("hello".into())],
        };
        round_trip(&event);
        assert_eq!(
            serde_json::to_value(&event).expect("serialize should succeed"),
            json!({
                "type": "message",
                "role": "user",
                "content": [{ "type": "text", "value": "hello" }]
            })
        );
    }

    #[test]
    fn reasoning_round_trips() {
        round_trip(&SessionEvent::Reasoning {
            reasoning: "think step by step".into(),
            signature: None,
        });
        round_trip(&SessionEvent::Reasoning {
            reasoning: "think step by step".into(),
            signature: Some("sig-123".into()),
        });
    }

    #[test]
    fn output_round_trips() {
        round_trip(&SessionEvent::Output {
            text: "delta".into(),
        });
    }

    #[test]
    fn tool_call_round_trips() {
        round_trip(&SessionEvent::ToolCall {
            id: "call_1".into(),
            name: "search".into(),
            arguments: json!({ "query": "cuca", "limit": 5 }),
        });
    }

    #[test]
    fn tool_result_round_trips() {
        round_trip(&SessionEvent::ToolResult {
            tool_call_id: "call_1".into(),
            output: "42".into(),
            stdout: None,
            stderr: None,
            exit_code: None,
        });
        round_trip(&SessionEvent::ToolResult {
            tool_call_id: "call_1".into(),
            output: "42".into(),
            stdout: Some("stdout here".into()),
            stderr: Some("stderr here".into()),
            exit_code: Some(1),
        });
    }

    #[test]
    fn model_swap_round_trips() {
        round_trip(&SessionEvent::ModelSwap {
            from: "fast".into(),
            to: "slow".into(),
            reason: "escalated".into(),
        });
    }

    #[test]
    fn latency_round_trips() {
        round_trip(&SessionEvent::Latency { duration_ms: 1234 });
    }

    #[test]
    fn token_usage_round_trips() {
        round_trip(&SessionEvent::TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
        });
    }

    #[test]
    fn fork_round_trips() {
        round_trip(&SessionEvent::Fork {
            from_point: "sess-a:3".into(),
            to_session: "sess-b".into(),
        });
    }

    #[test]
    fn record_constructors_populate_fields_and_point_id() {
        let at = SessionRecord::at(
            "sess-a",
            7,
            1_700_000_000_000,
            SessionEvent::Output { text: "x".into() },
        );
        assert_eq!(at.session_id, "sess-a");
        assert_eq!(at.sequence, 7);
        assert_eq!(at.timestamp_ms, 1_700_000_000_000);
        assert_eq!(at.point_id(), "sess-a:7");

        let new = SessionRecord::new("sess-a", SessionEvent::Output { text: "x".into() });
        assert_eq!(new.session_id, "sess-a");
        assert_eq!(new.sequence, 0);
        assert_eq!(new.point_id(), "sess-a:0");
        assert!(new.timestamp_ms > 0);
    }

    #[test]
    fn trajectory_round_trips_in_order() {
        let trajectory = vec![
            SessionRecord::new(
                "sess-a".to_string(),
                SessionEvent::SystemPrompt {
                    text: "be helpful".into(),
                },
            ),
            SessionRecord::new(
                "sess-a".to_string(),
                SessionEvent::Message {
                    role: MessageRole::User,
                    content: vec![MessageContentBlock::Text("what is 2+2?".into())],
                },
            ),
            SessionRecord::new(
                "sess-a".to_string(),
                SessionEvent::ToolCall {
                    id: "call_1".into(),
                    name: "calc".into(),
                    arguments: json!({ "expr": "2+2" }),
                },
            ),
            SessionRecord::new(
                "sess-a".to_string(),
                SessionEvent::ToolResult {
                    tool_call_id: "call_1".into(),
                    output: "4".into(),
                    stdout: Some("4".into()),
                    stderr: None,
                    exit_code: Some(0),
                },
            ),
            SessionRecord::new(
                "sess-a".to_string(),
                SessionEvent::Latency { duration_ms: 15 },
            ),
            SessionRecord::new(
                "sess-a".to_string(),
                SessionEvent::TokenUsage {
                    prompt_tokens: 20,
                    completion_tokens: 3,
                },
            ),
        ];
        let json = serde_json::to_value(&trajectory).expect("serialize should succeed");
        let back: Vec<SessionRecord> =
            serde_json::from_value(json).expect("deserialize should succeed");
        assert_eq!(back, trajectory);
    }
}
