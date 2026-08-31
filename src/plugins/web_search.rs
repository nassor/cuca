//! Real-time web search and live retrieval plugin.
//!
//! [`WebSearchPlugin`] hooks live search endpoints (Firecrawl, Tavily, or the
//! DeepSeek Web Search endpoint) into the agent's stream pipeline. It resolves
//! model-issued `web_search` and `web_extract` tool calls to normalized
//! [`ToolResult`](crate::types::MessageContentBlock::ToolResult) blocks so the
//! model can query live documentation, specs, and indexes during inference.
//!
//! # Bridge design: sync hook → async HTTP client
//!
//! [`CucaPlugin::on_stream_chunk`] is synchronous, but the HTTP calls behind
//! [`WebSearchPlugin::search`] / [`WebSearchPlugin::extract_page`] are async.
//! Blocking the caller's executor to wait on a runtime-managed future would
//! deadlock a current-thread tokio runtime. Unlike the MCP plugin, which
//! keeps a long-lived worker thread alive across calls, this plugin has no
//! persistent connection to keep warm. Each tool call therefore spawns a
//! **short-lived dedicated OS thread** ([`std::thread::spawn`]) that builds its
//! own current-thread tokio runtime, runs the search/extraction *inside* that
//! runtime via [`tokio::runtime::Runtime::block_on`], and reports the result
//! back over a `std::sync::mpsc` channel, which the hook awaits with a plain
//! std blocking `recv()`.
//!
//! Pause semantics: the stream pipeline **pauses** while the search runs (the
//! model's stream stops producing blocks until the tool result is emitted),
//! the same documented pause as the MCP and HITL plugins. The per-call cost
//! is bounded by the search latency plus one thread-spawn; the thread is
//! dropped when the call completes. Because the wait uses std primitives (no
//! tokio `blocking_send`/`blocking_recv`, which panic inside any runtime), the
//! hook may block even on a runtime worker thread such as the pipeline's
//! `poll_next`. This is safe here precisely because no persistent state must
//! outlive the call: the per-call thread exclusively owns its runtime.

use serde_json::{Value, json};

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::types::MessageContentBlock;

/// A backend that powers [`WebSearchPlugin`] search calls.
///
/// Each provider maps to a distinct request shape (see
/// [`WebSearchPlugin::build_request`]) and response shape (see
/// [`WebSearchPlugin::parse_response`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSearchProvider {
    /// Firecrawl search (`POST {base}/v1/search`) and scrape
    /// (`POST {base}/v1/scrape`). The only provider with page extraction.
    Firecrawl,
    /// Tavily search (`POST https://api.tavily.com/search`). Snippets only;
    /// no page extraction.
    Tavily,
    /// DeepSeek Web Search (`POST {base}/web-search`). Snippets only; no page
    /// extraction.
    DeepSeekWebSearch,
}

/// Configuration for a [`WebSearchPlugin`].
///
/// [`api_key`](Self::api_key) is required for live calls; [`Default`] leaves it
/// empty so the caller must supply one. [`max_results`](Self::max_results)
/// defaults to 5.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSearchConfig {
    /// The search backend to use.
    pub provider: WebSearchProvider,
    /// API key for the configured provider.
    pub api_key: String,
    /// Optional base URL override. When `None`, a per-provider documented
    /// default is used: `https://api.firecrawl.dev` (Firecrawl),
    /// `https://api.tavily.com` (Tavily), `https://api.deepseek.com`
    /// (DeepSeek Web Search).
    pub base_url: Option<String>,
    /// Maximum number of results to request (and return) per search.
    pub max_results: usize,
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        WebSearchConfig {
            // Firecrawl is the default because it is the only provider that
            // also supports page extraction.
            provider: WebSearchProvider::Firecrawl,
            api_key: String::new(),
            base_url: None,
            max_results: 5,
        }
    }
}

/// A single normalized search hit.
///
/// [`WebSearchPlugin::parse_response`] projects each provider's raw response
/// items onto this shape so downstream tool results are provider-agnostic.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    /// Result title.
    pub title: String,
    /// Result URL.
    pub url: String,
    /// Short textual excerpt (Firecrawl `description`, Tavily `content`,
    /// DeepSeek `snippet`).
    pub snippet: String,
}

/// Web search plugin: resolves `web_search` / `web_extract` tool calls in the
/// stream pipeline against a configured search backend.
#[derive(Clone)]
pub struct WebSearchPlugin {
    config: WebSearchConfig,
    http: reqwest::Client,
}

impl WebSearchPlugin {
    /// Build a plugin for the given configuration with a fresh
    /// [`reqwest::Client`].
    pub fn new(config: WebSearchConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        WebSearchPlugin { config, http }
    }

    /// The resolved base URL for the configured provider (override applied).
    fn base_url(&self) -> String {
        match &self.config.base_url {
            Some(override_url) => override_url.clone(),
            None => match self.config.provider {
                WebSearchProvider::Firecrawl => "https://api.firecrawl.dev".to_string(),
                WebSearchProvider::Tavily => "https://api.tavily.com".to_string(),
                WebSearchProvider::DeepSeekWebSearch => "https://api.deepseek.com".to_string(),
            },
        }
    }

    /// Build the provider-specific request as `(method, url, headers, body)`.
    ///
    /// This is the testable seam for request construction; it performs no I/O.
    /// - Firecrawl: `POST {base}/v1/search`, `Authorization: Bearer <key>`,
    ///   body `{ query, limit }`.
    /// - Tavily: `POST {base}/search`, no auth header (the key travels in the
    ///   body), body `{ api_key, query, max_results }`.
    /// - DeepSeek Web Search: `POST {base}/web-search`,
    ///   `Authorization: Bearer <key>`, body `{ query, max_results }`.
    ///
    /// The endpoint shapes follow the documented defaults and are
    /// overridable wholesale via [`WebSearchConfig::base_url`].
    pub fn build_request(&self, query: &str) -> (String, String, Vec<(String, String)>, Value) {
        let base = self.base_url();
        let (path, headers, body) = match self.config.provider {
            WebSearchProvider::Firecrawl => (
                format!("{base}/v1/search"),
                vec![(
                    "Authorization".to_string(),
                    format!("Bearer {}", self.config.api_key),
                )],
                json!({ "query": query, "limit": self.config.max_results }),
            ),
            WebSearchProvider::Tavily => (
                format!("{base}/search"),
                Vec::new(),
                json!({
                    "api_key": self.config.api_key,
                    "query": query,
                    "max_results": self.config.max_results,
                }),
            ),
            WebSearchProvider::DeepSeekWebSearch => (
                format!("{base}/web-search"),
                vec![(
                    "Authorization".to_string(),
                    format!("Bearer {}", self.config.api_key),
                )],
                json!({ "query": query, "max_results": self.config.max_results }),
            ),
        };
        ("POST".to_string(), path, headers, body)
    }

    /// Parse a provider response body into normalized [`SearchResult`]s.
    ///
    /// - Firecrawl: `data[].{ title, url, description }` → snippet =
    ///   `description`.
    /// - Tavily: `results[].{ title, url, content }` → snippet = `content`.
    /// - DeepSeek Web Search: `results[].{ title, url, snippet }`.
    ///
    /// Malformed JSON yields [`PluginError::Validation`]; a well-formed body
    /// with missing/empty results yields `Ok(vec![])` (documented behavior).
    pub fn parse_response(&self, body: &str) -> Result<Vec<SearchResult>, PluginError> {
        let value: Value = serde_json::from_str(body).map_err(|e| PluginError::Validation {
            schema: "web_search response".to_string(),
            message: format!("malformed JSON response: {e}"),
        })?;
        let results = match self.config.provider {
            WebSearchProvider::Firecrawl => value
                .get("data")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|item| {
                    Some(SearchResult {
                        title: item.get("title")?.as_str()?.to_string(),
                        url: item.get("url")?.as_str()?.to_string(),
                        snippet: item
                            .get("description")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    })
                })
                .collect(),
            WebSearchProvider::Tavily => value
                .get("results")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|item| {
                    Some(SearchResult {
                        title: item.get("title")?.as_str()?.to_string(),
                        url: item.get("url")?.as_str()?.to_string(),
                        snippet: item
                            .get("content")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    })
                })
                .collect(),
            WebSearchProvider::DeepSeekWebSearch => value
                .get("results")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|item| {
                    Some(SearchResult {
                        title: item.get("title")?.as_str()?.to_string(),
                        url: item.get("url")?.as_str()?.to_string(),
                        snippet: item
                            .get("snippet")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    })
                })
                .collect(),
        };
        Ok(results)
    }

    /// Run a live search against the configured provider.
    ///
    /// This is the live-network path; the unit suite never calls it directly
    /// (request construction and response parsing are exercised via
    /// [`Self::build_request`] / [`Self::parse_response`]). Non-2xx responses
    /// and transport errors map to [`PluginError::Internal`].
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>, PluginError> {
        let (method, url, headers, body) = self.build_request(query);
        let method = reqwest::Method::from_bytes(method.as_bytes())
            .map_err(|e| PluginError::Internal(format!("invalid HTTP method: {e}")))?;
        let mut request = self.http.request(method, &url);
        for (key, value) in &headers {
            request = request.header(key, value);
        }
        let response = request
            .json(&body)
            .send()
            .await
            .map_err(|e| PluginError::Internal(format!("web search request failed: {e}")))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| PluginError::Internal(format!("reading search response failed: {e}")))?;
        if !status.is_success() {
            return Err(PluginError::Internal(format!(
                "web search returned HTTP {status}: {text}"
            )));
        }
        self.parse_response(&text)
    }

    /// Whether page extraction is supported for the configured provider.
    ///
    /// Only Firecrawl exposes a scrape endpoint; Tavily and DeepSeek Web Search
    /// return snippets only, so [`Self::extract_page`] reports
    /// [`PluginError::NotSupported`] for them.
    pub(crate) fn extract_supported(&self) -> bool {
        matches!(self.config.provider, WebSearchProvider::Firecrawl)
    }

    /// Extract readable page text via the provider's scrape endpoint.
    ///
    /// Firecrawl: `POST {base}/v1/scrape`, `Authorization: Bearer <key>`, body
    /// `{ url }`; the markdown (or raw `content`) field of the response
    /// `data` object is returned. Non-Firecrawl providers return
    /// [`PluginError::NotSupported`] before any network I/O, so this path is
    /// testable without a live connection.
    pub async fn extract_page(&self, url: &str) -> Result<String, PluginError> {
        if !self.extract_supported() {
            return Err(PluginError::NotSupported(
                "page extraction is only supported by Firecrawl; Tavily and DeepSeek Web Search return search snippets only".to_string(),
            ));
        }
        let target = format!("{}/v1/scrape", self.base_url());
        let response = self
            .http
            .post(&target)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&json!({ "url": url }))
            .send()
            .await
            .map_err(|e| PluginError::Internal(format!("scrape request failed: {e}")))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| PluginError::Internal(format!("reading scrape response failed: {e}")))?;
        if !status.is_success() {
            return Err(PluginError::Internal(format!(
                "scrape returned HTTP {status}: {text}"
            )));
        }
        let value: Value = serde_json::from_str(&text).map_err(|e| PluginError::Validation {
            schema: "web scrape response".to_string(),
            message: format!("malformed JSON response: {e}"),
        })?;
        let data = value.get("data");
        if let Some(markdown) = data.and_then(|d| d.get("markdown")).and_then(Value::as_str) {
            return Ok(markdown.to_string());
        }
        if let Some(content) = data.and_then(|d| d.get("content")).and_then(Value::as_str) {
            return Ok(content.to_string());
        }
        Err(PluginError::Internal(
            "scrape response contained no extractable content".to_string(),
        ))
    }

    /// Run a search synchronously on a short-lived dedicated thread.
    ///
    /// Spawns one OS thread, builds a current-thread tokio runtime on it, and
    /// runs [`Self::search`] inside that runtime, returning the result over a
    /// std mpsc channel. The thread is dropped when the call completes, so no
    /// persistent state or runtime-guard hazard exists; the pipeline pauses for
    /// the search latency (see module docs).
    fn search_sync(&self, query: &str) -> Result<Vec<SearchResult>, PluginError> {
        let plugin = self.clone();
        let query = query.to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let result = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| PluginError::Internal(format!("failed to build runtime: {e}")))
                .and_then(|rt| rt.block_on(plugin.search(&query)));
            // The receiver may be dropped if the caller bails; ignore send
            // failure here: the caller handles a missing result.
            let _ = tx.send(result);
        });
        match handle.join() {
            Ok(()) => rx.recv().unwrap_or_else(|_| {
                Err(PluginError::Internal(
                    "search worker thread failed to return a result".to_string(),
                ))
            }),
            Err(_) => Err(PluginError::Internal(
                "search worker thread panicked".to_string(),
            )),
        }
    }

    /// Run page extraction synchronously on a short-lived dedicated thread.
    ///
    /// Same per-call thread bridge as [`Self::search_sync`]; see its docs.
    fn extract_sync(&self, url: &str) -> Result<String, PluginError> {
        let plugin = self.clone();
        let url = url.to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let result = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| PluginError::Internal(format!("failed to build runtime: {e}")))
                .and_then(|rt| rt.block_on(plugin.extract_page(&url)));
            let _ = tx.send(result);
        });
        match handle.join() {
            Ok(()) => rx.recv().unwrap_or_else(|_| {
                Err(PluginError::Internal(
                    "extract worker thread failed to return a result".to_string(),
                ))
            }),
            Err(_) => Err(PluginError::Internal(
                "extract worker thread panicked".to_string(),
            )),
        }
    }
}

/// Validate the `web_search` tool arguments and extract the required `query`.
///
/// A missing, non-string, or blank `query` yields [`PluginError::Validation`].
/// Testable seam for the stream hook's argument validation (no network).
pub(crate) fn parse_web_search_args(arguments: &Value) -> Result<String, PluginError> {
    let query = arguments.get("query").and_then(Value::as_str).unwrap_or("");
    if query.trim().is_empty() {
        return Err(PluginError::Validation {
            schema: "web_search".to_string(),
            message: "web_search requires a non-empty string `query` argument".to_string(),
        });
    }
    Ok(query.to_string())
}

/// Validate the `web_extract` tool arguments and extract the required `url`.
///
/// A missing, non-string, or blank `url` yields [`PluginError::Validation`].
/// Testable seam for the stream hook's argument validation (no network).
pub(crate) fn parse_web_extract_args(arguments: &Value) -> Result<String, PluginError> {
    let url = arguments.get("url").and_then(Value::as_str).unwrap_or("");
    if url.trim().is_empty() {
        return Err(PluginError::Validation {
            schema: "web_extract".to_string(),
            message: "web_extract requires a non-empty string `url` argument".to_string(),
        });
    }
    Ok(url.to_string())
}

/// Serialize a set of results into the JSON array string used as a
/// [`ToolResult`](crate::types::MessageContentBlock::ToolResult) `output`.
///
/// [`SearchResult`] is serializable, so the output round-trips through
/// [`serde_json`]. Serialization of this shape cannot fail; the fallback
/// preserves the message rather than panicking.
pub(crate) fn format_results(results: Vec<SearchResult>) -> String {
    serde_json::to_string(&results)
        .unwrap_or_else(|e| format!("failed to serialize search results: {e}"))
}

impl CucaPlugin for WebSearchPlugin {
    fn name(&self) -> &'static str {
        "web-search"
    }

    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        if let MessageContentBlock::ToolCall {
            id,
            name,
            arguments,
        } = chunk
        {
            let output = match name.as_str() {
                "web_search" => match parse_web_search_args(arguments) {
                    Ok(query) => self.search_sync(&query).map(format_results),
                    Err(e) => Err(e),
                },
                "web_extract" => match parse_web_extract_args(arguments) {
                    Ok(url) => self.extract_sync(&url),
                    Err(e) => Err(e),
                },
                // Unknown tools are not this plugin's responsibility; leave
                // the block untouched.
                _ => return Ok(()),
            };
            // Validation and transport errors surface inside the ToolResult
            // output rather than failing the hook, so the model can react to
            // them in the conversation.
            let output = output.unwrap_or_else(|e| e.to_string());
            // `id` is moved, not cloned: this assignment replaces the block it
            // was borrowed from.
            *chunk = MessageContentBlock::ToolResult {
                tool_call_id: std::mem::take(id),
                output,
            };
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "plugin-web-search"))]
mod tests {
    use super::*;

    fn firecrawl_plugin() -> WebSearchPlugin {
        WebSearchPlugin::new(WebSearchConfig {
            provider: WebSearchProvider::Firecrawl,
            api_key: "k-firecrawl".to_string(),
            base_url: None,
            max_results: 5,
        })
    }

    fn tavily_plugin() -> WebSearchPlugin {
        WebSearchPlugin::new(WebSearchConfig {
            provider: WebSearchProvider::Tavily,
            api_key: "k-tavily".to_string(),
            base_url: None,
            max_results: 7,
        })
    }

    fn deepseek_plugin() -> WebSearchPlugin {
        WebSearchPlugin::new(WebSearchConfig {
            provider: WebSearchProvider::DeepSeekWebSearch,
            api_key: "k-deepseek".to_string(),
            base_url: None,
            max_results: 3,
        })
    }

    #[test]
    fn firecrawl_request_shape() {
        let (method, url, headers, body) = firecrawl_plugin().build_request("cuca");
        assert_eq!(method, "POST");
        assert!(url.starts_with("https://api.firecrawl.dev"));
        assert!(url.ends_with("/v1/search"));
        assert_eq!(
            headers,
            vec![(
                "Authorization".to_string(),
                "Bearer k-firecrawl".to_string()
            )]
        );
        assert_eq!(body, json!({ "query": "cuca", "limit": 5 }));
    }

    #[test]
    fn tavily_request_shape() {
        let (method, url, headers, body) = tavily_plugin().build_request("cuca");
        assert_eq!(method, "POST");
        assert!(url.starts_with("https://api.tavily.com"));
        assert!(url.ends_with("/search"));
        // Tavily carries its key in the body, not an Authorization header.
        assert!(headers.is_empty());
        assert_eq!(body["api_key"], "k-tavily");
        assert_eq!(body["query"], "cuca");
        assert_eq!(body["max_results"], 7);
    }

    #[test]
    fn deepseek_request_shape() {
        let (method, url, headers, body) = deepseek_plugin().build_request("cuca");
        assert_eq!(method, "POST");
        assert!(url.starts_with("https://api.deepseek.com"));
        assert!(url.ends_with("/web-search"));
        assert_eq!(
            headers,
            vec![("Authorization".to_string(), "Bearer k-deepseek".to_string())]
        );
        assert_eq!(body, json!({ "query": "cuca", "max_results": 3 }));
    }

    #[test]
    fn base_url_override_applies() {
        let plugin = WebSearchPlugin::new(WebSearchConfig {
            provider: WebSearchProvider::Firecrawl,
            api_key: "k".to_string(),
            base_url: Some("http://localhost:8000".to_string()),
            max_results: 5,
        });
        let (_, url, _, _) = plugin.build_request("q");
        assert!(url.starts_with("http://localhost:8000"));
        assert!(url.ends_with("/v1/search"));
    }

    #[test]
    fn firecrawl_parse_response() {
        let body = r#"{
            "data": [
                { "title": "Firecrawl Docs", "url": "https://docs.firecrawl.dev", "description": "Scrape anything" },
                { "title": "CUCA", "url": "https://cuca.dev", "description": "Compact client" }
            ]
        }"#;
        let results = firecrawl_plugin().parse_response(body).unwrap();
        assert_eq!(
            results,
            vec![
                SearchResult {
                    title: "Firecrawl Docs".to_string(),
                    url: "https://docs.firecrawl.dev".to_string(),
                    snippet: "Scrape anything".to_string(),
                },
                SearchResult {
                    title: "CUCA".to_string(),
                    url: "https://cuca.dev".to_string(),
                    snippet: "Compact client".to_string(),
                },
            ]
        );
    }

    #[test]
    fn tavily_parse_response() {
        let body = r#"{
            "results": [
                { "title": "Tavily", "url": "https://tavily.com", "content": "Search API" }
            ]
        }"#;
        let results = tavily_plugin().parse_response(body).unwrap();
        assert_eq!(
            results,
            vec![SearchResult {
                title: "Tavily".to_string(),
                url: "https://tavily.com".to_string(),
                snippet: "Search API".to_string(),
            }]
        );
    }

    #[test]
    fn deepseek_parse_response() {
        let body = r#"{
            "results": [
                { "title": "DeepSeek", "url": "https://deepseek.com", "snippet": "Web search" }
            ]
        }"#;
        let results = deepseek_plugin().parse_response(body).unwrap();
        assert_eq!(
            results,
            vec![SearchResult {
                title: "DeepSeek".to_string(),
                url: "https://deepseek.com".to_string(),
                snippet: "Web search".to_string(),
            }]
        );
    }

    #[test]
    fn parse_response_malformed_is_validation() {
        assert!(matches!(
            firecrawl_plugin().parse_response("not json"),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn parse_response_empty_results_ok() {
        assert!(
            firecrawl_plugin()
                .parse_response(r#"{"data": []}"#)
                .unwrap()
                .is_empty()
        );
        assert!(tavily_plugin().parse_response(r#"{}"#).unwrap().is_empty());
    }

    #[test]
    fn parse_web_search_args_requires_query() {
        assert_eq!(
            parse_web_search_args(&json!({ "query": "cuca" })).unwrap(),
            "cuca"
        );
        assert!(matches!(
            parse_web_search_args(&json!({})),
            Err(PluginError::Validation { .. })
        ));
        assert!(matches!(
            parse_web_search_args(&json!({ "query": "  " })),
            Err(PluginError::Validation { .. })
        ));
        assert!(matches!(
            parse_web_search_args(&json!({ "query": 42 })),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn parse_web_extract_args_requires_url() {
        assert_eq!(
            parse_web_extract_args(&json!({ "url": "https://docs.firecrawl.dev" })).unwrap(),
            "https://docs.firecrawl.dev"
        );
        assert!(matches!(
            parse_web_extract_args(&json!({})),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn format_results_round_trips() {
        let results = vec![
            SearchResult {
                title: "A".to_string(),
                url: "https://a".to_string(),
                snippet: "sa".to_string(),
            },
            SearchResult {
                title: "B".to_string(),
                url: "https://b".to_string(),
                snippet: "sb".to_string(),
            },
        ];
        let output = format_results(results.clone());
        let parsed: Vec<SearchResult> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed, results);
    }

    #[test]
    fn web_search_missing_query_replaced_with_error_result() {
        // Validation fails before any network I/O, so the hook is safe to
        // exercise end-to-end here.
        let plugin = firecrawl_plugin();
        let mut chunk = MessageContentBlock::ToolCall {
            id: "call_1".to_string(),
            name: "web_search".to_string(),
            arguments: json!({}),
        };
        plugin.on_stream_chunk(&mut chunk).unwrap();
        match chunk {
            MessageContentBlock::ToolResult {
                tool_call_id,
                output,
            } => {
                assert_eq!(tool_call_id, "call_1");
                assert!(output.contains("requires"), "output: {output}");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn unknown_tool_passes_through() {
        let plugin = firecrawl_plugin();
        let original = MessageContentBlock::ToolCall {
            id: "call_2".to_string(),
            name: "some_other_tool".to_string(),
            arguments: json!({ "x": 1 }),
        };
        let mut chunk = original.clone();
        plugin.on_stream_chunk(&mut chunk).unwrap();
        assert_eq!(chunk, original);
    }

    #[test]
    fn extract_supported_only_for_firecrawl() {
        assert!(firecrawl_plugin().extract_supported());
        assert!(!tavily_plugin().extract_supported());
        assert!(!deepseek_plugin().extract_supported());
    }

    #[test]
    fn extract_page_not_supported_for_non_firecrawl() {
        // The provider guard runs before any network I/O, so a current-thread
        // runtime can drive it without a live connection.
        let plugin = tavily_plugin();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = runtime
            .block_on(plugin.extract_page("https://example.com"))
            .unwrap_err();
        assert!(matches!(err, PluginError::NotSupported(_)));
    }

    #[test]
    fn name_is_web_search() {
        assert_eq!(firecrawl_plugin().name(), "web-search");
    }

    #[test]
    fn plugin_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WebSearchPlugin>();
    }
}
