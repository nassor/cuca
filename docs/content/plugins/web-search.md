+++
title = "Web search"
description = "The live web search plugin: the three provider backends, the web_search and web_extract tools, and their request shapes."
template = "page.html"
weight = 8
+++

# Web search

<dl class="page-facts">
<dt>In one line</dt>
<dd>Resolves <code>web_search</code> and <code>web_extract</code> tool calls against a configured search backend.</dd>
<dt>You need</dt>
<dd>The <code>plugin-web-search</code> feature and an API key for the configured backend.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>WebSearchPlugin</code> or choosing a <code>WebSearchProvider</code>.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-web-search` (enables `dep:reqwest`, `dep:tokio`, `reqwest/rustls`) |
| Entry type | `WebSearchPlugin` |
| Supporting types | `WebSearchConfig`, `WebSearchProvider`, `SearchResult` |
| Plugin name | `"web-search"` |
| Attachment | `register_plugin` |
| Peer features | None; the feature enables its own TLS backend, so no provider feature is needed for HTTPS |
| Needs | An API key for the configured backend |

## Hooks

`WebSearchPlugin` overrides `on_stream_chunk` only. A `ToolCall` named `web_search` or `web_extract` is replaced with a `ToolResult`. Any other block passes through untouched.

- `web_search` takes a required, non-blank `query`. The output is a JSON array of `SearchResult` objects, `{title, url, snippet}`. A body with no results gives `[]`.
- `web_extract` takes a required, non-blank `url`. The output is the page text. Only Firecrawl exposes a scrape endpoint, so on Tavily and DeepSeek Web Search the call fails with `PluginError::NotSupported` before any network call.

Failures never fail the hook. The error text becomes the `ToolResult` output, so the model can react to it in the conversation. For example: `internal plugin error: web search returned HTTP 401 Unauthorized: ...`.

Each call runs on a short-lived OS thread with its own current-thread Tokio runtime, and the hook waits on a std channel. The stream pauses until the result is ready, then the thread ends. This is safe on a current-thread runtime and on a runtime worker.

The plugin also exposes `search(query)`, `extract_page(url)` and `build_request(query)`. `build_request` returns the method, URL, headers and body without doing any I/O.

## Configuration

`WebSearchConfig` defaults:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `provider` | `WebSearchProvider` | `Firecrawl` | The search backend. |
| `api_key` | `String` | `""` | Backend API key; set it for live calls. |
| `base_url` | `Option<String>` | `None` | Overrides the backend base URL. |
| `max_results` | `usize` | `5` | Results requested per search. |

`WebSearchPlugin::new(config)` builds a `reqwest::Client` with a 30 second timeout. It does not validate `api_key`.

### Backends

| `WebSearchProvider` | Search route | Auth | Default base URL | Extraction |
|---|---|---|---|---|
| `Firecrawl` | `POST {base}/v1/search` | `Authorization: Bearer` | `https://api.firecrawl.dev` | `POST {base}/v1/scrape` |
| `Tavily` | `POST {base}/search` | `api_key` in the body | `https://api.tavily.com` | None |
| `DeepSeekWebSearch` | `POST {base}/web-search` | `Authorization: Bearer` | `https://api.deepseek.com` | None |

Request bodies and response fields per backend:

- Firecrawl sends `{query, limit}` and reads `data[].{title, url, description}`, with `description` as the snippet.
- Tavily sends `{api_key, query, max_results}` and reads `results[].{title, url, content}`, with `content` as the snippet.
- DeepSeek Web Search sends `{query, max_results}` and reads `results[].{title, url, snippet}`.
- An item without a string `title` or `url` is dropped; a missing snippet becomes the empty string.
- Firecrawl extraction sends `{url}` and returns `data.markdown`, or `data.content` when there is no markdown. A response with neither is an error.

## Caps and at-cap policy

No cap on growth: the plugin holds its configuration and an HTTP client, and nothing accumulates between calls.

`max_results` bounds each search, not the plugin. The plugin passes it to the backend as `limit` or `max_results` and does not truncate the response. The usage gauge is none.

## Errors

The hook returns no error. Each variant below becomes `ToolResult` output text. `search` and `extract_page` return them to a direct caller.

- `PluginError::Validation` when `query` or `url` is missing, not a string, or blank, and when a response body is not valid JSON.
- `PluginError::NotSupported` for `web_extract` on Tavily or DeepSeek Web Search.
- `PluginError::Internal` for a transport failure, a non-2xx HTTP status (the message carries the status and body), a failure to build the worker runtime, a worker thread that panics or returns nothing, and a Firecrawl scrape response with no extractable content.

## Example

`examples/web_search.rs` searches through Firecrawl. It needs a `llama-server` on port 1234 with the demo model loaded and `CUCA_SEARCH_API_KEY` holding a Firecrawl key; without the key it prints one line and exits. `CUCA_SEARCH_BASE_URL` overrides the backend endpoint. `CUCA_BASE_URL` and `CUCA_MODEL` retarget the model at any OpenAI-compatible server.

```rust,name=Configure and register WebSearchPlugin, from examples/web_search.rs
let config = WebSearchConfig {
    provider: WebSearchProvider::Firecrawl,
    api_key,
    base_url: std::env::var("CUCA_SEARCH_BASE_URL").ok(),
    max_results: 3,
};
let web_search = Arc::new(WebSearchPlugin::new(config));
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url.clone())
    .register_plugin(Arc::clone(&web_search) as Arc<dyn CucaPlugin>)
    .build()?;
```

```bash,name=Runs the same on all three platforms
cargo run --example web_search --features "provider-llamacpp plugin-web-search"
```

```text,name=Expected output with an invalid key
Backend Firecrawl, at most 3 results per search

Turn 1: the model searches, and the pipeline pauses for the HTTP call
  tool result: internal plugin error: web search returned HTTP 401 Unauthorized: {"success":false,"error":"Unauthorized: Invalid token"}
  thinking blocks: 60

No results, so there is no second turn
  a valid key makes this a JSON array of {title, url, snippet} objects
```

The thinking-block count depends on the model; this output came from `google/gemma-4-12b-qat`.

## Related pages

- [Plugins](@/plugins/_index.md) lists every plugin, its hooks and its caps.
- [Write a custom plugin](@/guides/custom-plugin.md) walks through the hook contract.
- [Error types](@/reference/errors.md) lists `PluginError`.
