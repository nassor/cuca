//! OpenAI provider dispatch.
//!
//! Routes `provider-openai` requests through the shared
//! [`openai_compat`](crate::provider::openai_compat) adapter, defaulting the
//! base URL to the OpenAI API (`https://api.openai.com/v1`) when the builder
//! did not set one.

use crate::client::{CucaClient, ProviderDispatch};
use crate::error::CucaError;
use crate::provider::openai_compat::{OpenAiCompatConfig, openai_compat_stream};
use crate::request::UnifiedRequest;

#[cfg(feature = "provider-openai")]
impl CucaClient {
    /// Dispatch a unified request to the OpenAI `/chat/completions` endpoint.
    ///
    /// Uses the client's base URL when set, otherwise the OpenAI API default;
    /// the configured API key becomes the bearer token. Called by
    /// `generate_stream` under the `provider-openai` feature.
    pub(crate) async fn dispatch_openai(
        &self,
        req: UnifiedRequest,
    ) -> Result<ProviderDispatch, CucaError> {
        let cfg = OpenAiCompatConfig {
            base_url: if self.base_url().is_empty() {
                "https://api.openai.com/v1".into()
            } else {
                self.base_url().to_string()
            },
            api_key: self.api_key().map(str::to_string),
            model: req.model.clone(),
        };
        openai_compat_stream(self.http_client(), &cfg, req).await
    }
}

#[cfg(all(test, feature = "provider-openai"))]
mod tests {
    use std::sync::{Arc, mpsc};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_stream::StreamExt;

    use crate::client::CucaClient;
    use crate::error::PluginError;
    use crate::plugin::CucaPlugin;
    use crate::request::{UnifiedRequest, UnifiedResponse};
    use crate::types::{MessageContentBlock, ProviderEndpoint};

    /// Records every `on_response_complete` payload for assertion.
    struct RecordingPlugin {
        /// The completion hook runs synchronously inside `poll_next`; an
        /// unbounded mpsc sender records each response without any locking
        /// (a tokio mutex's blocking_lock would panic inside the runtime).
        tx: mpsc::Sender<UnifiedResponse>,
    }

    impl RecordingPlugin {
        fn new() -> (Self, mpsc::Receiver<UnifiedResponse>) {
            let (tx, rx) = mpsc::channel();
            (Self { tx }, rx)
        }
    }

    impl CucaPlugin for RecordingPlugin {
        fn name(&self) -> &'static str {
            "recording-openai"
        }

        fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
            self.tx.send(res.clone()).map_err(|_| {
                PluginError::Internal("recording channel closed before completion".into())
            })
        }
    }

    /// Canned OpenAI-shaped SSE frames: text delta, reasoning, tool-call
    /// accumulation, finish_reason, then `[DONE]`.
    fn canned_frames() -> Vec<&'static str> {
        vec![
            r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
            r#"data: {"choices":[{"delta":{"reasoning_content":"let me think"}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"NYC\"}"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"data: [DONE]"#,
        ]
    }

    #[tokio::test]
    async fn end_to_end_stream_translates_openai_sse_and_completes_plugin() {
        // In-process stub: an OpenAI-shaped SSE server on an ephemeral port.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let frames = canned_frames();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            // Consume the request head; the response is written regardless.
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf).await.unwrap();
            let mut response = String::from(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            );
            for frame in &frames {
                // Each SSE frame needs its terminating blank line inside the
                // chunk body for the parser to complete it.
                let body = format!("{frame}\n\n");
                response.push_str(&format!("{:x}\r\n{body}\r\n", body.len()));
            }
            response.push_str("0\r\n\r\n");
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        });

        let (plugin, rx) = RecordingPlugin::new();
        let plugin = Arc::new(plugin);
        // Register through the trait-object type the builder expects; the
        // concrete handle is kept for asserting on the recorded responses.
        let plugin_dyn = Arc::clone(&plugin) as Arc<dyn CucaPlugin>;
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::OpenAi)
            .with_base_url(format!("http://{addr}/v1"))
            .register_plugin(plugin_dyn)
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));

        let stream = client
            .generate_stream(UnifiedRequest::new("gpt-4o").add_user_message("hi"))
            .await
            .unwrap_or_else(|e| panic!("generate_stream must succeed: {e}"));
        let mut blocks = Vec::new();
        let mut stream = stream;
        while let Some(block) = stream.next().await {
            blocks.push(block.unwrap_or_else(|e| panic!("stream block must be Ok: {e}")));
        }
        server.await.unwrap();

        assert_eq!(
            blocks,
            vec![
                MessageContentBlock::Text("Hello".into()),
                MessageContentBlock::Thinking {
                    reasoning: "let me think".into(),
                    signature: None,
                },
                MessageContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: serde_json::json!({ "location": "NYC" }),
                },
            ]
        );

        // The completion hook fired exactly once, with the aggregated response.
        let completed: Vec<UnifiedResponse> = rx.try_iter().collect();
        assert_eq!(
            completed.len(),
            1,
            "on_response_complete must fire exactly once"
        );
        assert!(completed[0].completion_tokens >= 1);
        assert_eq!(completed[0].content, blocks);
        assert_eq!(completed[0].model, "gpt-4o");
        assert_eq!(completed[0].provider, ProviderEndpoint::OpenAi);
        // Non-Anthropic adapters always leave the response metadata handle
        // empty: OpenAI reports no provider prompt-cache usage.
        assert_eq!(completed[0].prompt_cache_usage, None);
    }
}
