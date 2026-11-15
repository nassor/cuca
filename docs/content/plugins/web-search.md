+++
title = "Web search"
description = "Give the model live web search and page extraction through Firecrawl, Tavily or DeepSeek Web Search."
template = "page.html"
weight = 8
+++

# Web search

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>WebSearchPlugin</code> answers the model's <code>web_search</code> and <code>web_extract</code> tool calls using a search service.</dd>
<dt>You need</dt>
<dd>The <code>plugin-web-search</code> feature and an API key for the service you choose.</dd>
<dt>Read this if</dt>
<dd>You want the model to look things up on the web.</dd>
</dl>

The plugin sends a search request to one of three services and returns the results as a `ToolResult`. It brings its own HTTPS support, so you do not need a provider feature for it.

## Register it

```rust,name=Configure and register WebSearchPlugin
let config = WebSearchConfig {
    provider: WebSearchProvider::Firecrawl,
    api_key,
    base_url: None,
    max_results: 3,
};
let web_search = Arc::new(WebSearchPlugin::new(config));
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url.clone())
    .register_plugin(Arc::clone(&web_search) as Arc<dyn CucaPlugin>)
    .build()?;
```

Expected effect: a `web_search` call with a `query` becomes a `ToolResult` holding a JSON array of `{title, url, snippet}` objects. The plugin does not add the two tools to your request, so declare `web_search` and `web_extract` as tools yourself.

## Run the example

`examples/web_search.rs` searches through Firecrawl. It needs a `llama-server` on port 1234 with a tool-calling model loaded, and a Firecrawl key after `--` as `--search-api-key <KEY>`. Without a key it prints one line and exits. `--search-base-url` changes the search endpoint. `--base-url` and `--model` point the model at another OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example web_search --features "provider-llamacpp plugin-web-search" -- --search-api-key <KEY>
```

```text,name=Expected output with an invalid key
Backend Firecrawl, at most 3 results per search

Turn 1: the model searches, and the pipeline pauses for the HTTP call
  tool result: internal plugin error: web search returned HTTP 401 Unauthorized: {"success":false,"error":"Unauthorized: Invalid token"}
  thinking blocks: 60

No results, so there is no second turn
  a valid key makes this a JSON array of {title, url, snippet} objects
```

A bad key shows the error text as the tool result, so the model can react to it.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-web-search` (brings its own `reqwest` with `rustls`) |
| Entry type | `WebSearchPlugin` |
| Supporting types | `WebSearchConfig`, `WebSearchProvider`, `SearchResult` |
| Plugin name | `"web-search"` |
| Attachment | `register_plugin` |
| Needs | An API key for the chosen service |

## Hooks

`WebSearchPlugin` acts on `on_stream_chunk` only. A `ToolCall` named `web_search` or `web_extract` becomes a `ToolResult`. Other blocks pass through.

- `web_search` takes a required, non-blank `query`. The output is a JSON array of `{title, url, snippet}`. No results gives `[]`.
- `web_extract` takes a required, non-blank `url`. The output is the page text. Only Firecrawl can extract pages. On Tavily and DeepSeek Web Search the call fails with `PluginError::NotSupported` before any network call.

A failure never fails the hook. The error text becomes the `ToolResult` output, so the model can see it. For example: `internal plugin error: web search returned HTTP 401 Unauthorized: ...`.

The stream pauses until the HTTP call finishes. This is safe on a current-thread runtime and on a runtime worker.

Direct methods: `search(query)`, `extract_page(url)` and `build_request(query)`. `build_request` returns the method, URL, headers and body without sending anything.

## Configuration

`WebSearchConfig` keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `provider` | `WebSearchProvider` | `Firecrawl` | The search service. |
| `api_key` | `String` | `""` | The service API key. Set it for live calls. |
| `base_url` | `Option<String>` | `None` | Replaces the service's base URL. |
| `max_results` | `usize` | `5` | Results requested per search. |

`WebSearchPlugin::new(config)` builds an HTTP client with a 30 second timeout. It does not check `api_key`.

### Services

- `Firecrawl`: search `POST {base}/v1/search`, auth `Authorization: Bearer`, default base URL `https://api.firecrawl.dev`, page extraction `POST {base}/v1/scrape`.
- `Tavily`: search `POST {base}/search`, auth `api_key` in the body, default base URL `https://api.tavily.com`, no page extraction.
- `DeepSeekWebSearch`: search `POST {base}/web-search`, auth `Authorization: Bearer`, default base URL `https://api.deepseek.com`, no page extraction.

Request and response fields:

- Firecrawl sends `{query, limit}` and reads `data[].{title, url, description}`. `description` becomes the snippet.
- Tavily sends `{api_key, query, max_results}` and reads `results[].{title, url, content}`. `content` becomes the snippet.
- DeepSeek Web Search sends `{query, max_results}` and reads `results[].{title, url, snippet}`.
- An item without a string `title` or `url` is dropped. A missing snippet becomes an empty string.
- Firecrawl extraction sends `{url}` and returns `data.markdown`, or `data.content` if there is no markdown. A response with neither is an error.

## Caps and at-cap policy

No cap. The plugin holds its configuration and an HTTP client, and nothing builds up between calls. There is no usage gauge.

`max_results` limits each search, not the plugin. It is passed to the service as `limit` or `max_results`. The plugin does not cut the response.

## Errors

The hook returns no error. Each variant below becomes `ToolResult` text. `search` and `extract_page` return them to a direct caller.

- `PluginError::Validation` when `query` or `url` is missing, not a string or blank, and when a response body is not valid JSON.
- `PluginError::NotSupported` for `web_extract` on Tavily or DeepSeek Web Search.
- `PluginError::Internal` for a network failure, an HTTP status outside 2xx (the message holds the status and body), a worker that fails to start, panics or returns nothing, and a Firecrawl response with no usable content.

## Related pages

- [Plugins](@/plugins/_index.md): the hooks, their order and the error rules.
- [Write a custom plugin](@/guides/custom-plugin.md): the hook contract.
- [Error types](@/reference/errors.md): `PluginError`.
