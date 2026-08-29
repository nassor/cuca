//! vLLM provider dispatch.
//!
//! Routes `provider-vllm` requests through the shared
//! [`openai_compat`](crate::provider::openai_compat) adapter: vLLM exposes the
//! same `/v1/chat/completions` `data:`-SSE contract as OpenAI. It differs only
//! in the default base URL (`http://127.0.0.1:8000`) and that the API key is
//! an optional header rather than a required bearer token.
//!
//! vLLM emits `reasoning_content` for reasoning models; that field flows
//! through the shared translator's `Thinking` mapping at no extra cost.

use crate::client::{CucaClient, ProviderDispatch};
use crate::error::CucaError;
use crate::provider::openai_compat::{OpenAiCompatConfig, openai_compat_stream};
use crate::request::UnifiedRequest;

/// Resolve the effective base URL for a vLLM dispatch.
///
/// An empty builder `base_url` falls back to vLLM's default
/// (`http://127.0.0.1:8000/v1`); any configured value passes through
/// unchanged. The default carries the `/v1` suffix because
/// [`openai_compat_stream`] appends `/chat/completions` to the base URL, so
/// the full request path becomes `/v1/chat/completions`. (The spec's provider
/// table lists the bare host `http://127.0.0.1:8000` without `/v1` , an
/// inconsistency with the OpenAI/DeepSeek table rows, which include `/v1`;
/// documented here rather than silently "fixed".)
pub(crate) fn resolve_base_url(client_base: &str) -> String {
    if client_base.is_empty() {
        "http://127.0.0.1:8000/v1".into()
    } else {
        client_base.to_string()
    }
}

#[cfg(feature = "provider-vllm")]
impl CucaClient {
    /// Dispatch a unified request to the vLLM `/chat/completions` endpoint.
    ///
    /// Uses the client's base URL when set, otherwise vLLM's local default;
    /// the configured API key becomes the bearer token (vLLM treats it as an
    /// optional header). Called by `generate_stream` under the
    /// `provider-vllm` feature.
    pub(crate) async fn dispatch_vllm(
        &self,
        req: UnifiedRequest,
    ) -> Result<ProviderDispatch, CucaError> {
        let cfg = OpenAiCompatConfig {
            base_url: resolve_base_url(self.base_url()),
            api_key: self.api_key().map(str::to_string), // vLLM: optional header
            model: req.model.clone(),
        };
        openai_compat_stream(self.http_client(), &cfg, req).await
    }
}

#[cfg(all(test, feature = "provider-vllm"))]
mod tests {
    use super::resolve_base_url;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_stream::StreamExt;

    use crate::provider::openai_compat::{ChatCompletionTranslator, translate_sse};
    use crate::sse::SseStreamParser;
    use crate::types::{MessageContentBlock, ProviderEndpoint};

    /// Empty builder base URL resolves to the vLLM default.
    #[test]
    fn empty_base_resolves_to_vllm_default() {
        assert_eq!(resolve_base_url(""), "http://127.0.0.1:8000/v1");
    }

    /// A configured base URL is respected verbatim (override wins).
    #[test]
    fn configured_base_is_respected() {
        assert_eq!(
            resolve_base_url("http://127.0.0.1:9999/v1"),
            "http://127.0.0.1:9999/v1"
        );
    }

    /// Reuse proof: `translate_sse` over vLLM-shaped `data:` frames yields
    /// `Text`/`ToolCall` blocks through the shared adapter.
    #[test]
    fn translate_sse_reuse_yields_text_and_tool_call_blocks() {
        let mut parser = SseStreamParser::new();
        let mut translator = ChatCompletionTranslator::new();
        let mut blocks = Vec::new();
        for frame in [
            r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"NYC\"}"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"data: [DONE]"#,
        ] {
            let out = translate_sse(
                &mut parser,
                &mut translator,
                format!("{frame}\n\n").as_bytes(),
            )
            .unwrap();
            blocks.extend(out.into_iter().flatten());
        }
        // finish_reason/[DONE] flush the tool call into the translator's pending
        // queue without returning it; drain it with one final translate, mirroring
        // how `openai_compat_stream` drains pending at stream end.
        blocks.extend(translator.translate("{}").unwrap());
        assert_eq!(
            blocks,
            vec![
                MessageContentBlock::Text("Hello".into()),
                MessageContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: serde_json::json!({ "location": "NYC" }),
                },
            ]
        );
    }

    /// End-to-end: a stub vLLM-shaped SSE server on an ephemeral port, with the
    /// request path captured so we can assert it hit `/v1/chat/completions`.
    #[tokio::test]
    async fn end_to_end_stream_hits_v1_chat_completions_in_order() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            // Read the request head and capture the request-line path.
            let mut buf = [0u8; 1024];
            let n = socket.read(&mut buf).await.unwrap();
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = head
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or_default()
                .to_string();
            let mut response = String::from(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            );
            for frame in [
                r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
                r#"data: {"choices":[{"delta":{"reasoning_content":"let me think"}}]}"#,
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
                r#"data: [DONE]"#,
            ] {
                let body = format!("{frame}\n\n");
                response.push_str(&format!("{:x}\r\n{body}\r\n", body.len()));
            }
            response.push_str("0\r\n\r\n");
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
            path
        });

        let client = crate::client::CucaClient::builder()
            .with_provider(ProviderEndpoint::Vllm)
            .with_base_url(format!("http://127.0.0.1:{}/v1", addr.port()))
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));

        let stream = client
            .generate_stream(
                crate::request::UnifiedRequest::new("test-model").add_user_message("hi"),
            )
            .await
            .unwrap_or_else(|e| panic!("generate_stream must succeed: {e}"));
        let mut blocks = Vec::new();
        let mut stream = stream;
        while let Some(block) = stream.next().await {
            blocks.push(block.unwrap_or_else(|e| panic!("stream block must be Ok: {e}")));
        }
        let path = server.await.unwrap();

        assert_eq!(path, "/v1/chat/completions");
        assert_eq!(
            blocks,
            vec![
                MessageContentBlock::Text("Hello".into()),
                MessageContentBlock::Thinking {
                    reasoning: "let me think".into(),
                    signature: None,
                },
            ]
        );
    }
}
