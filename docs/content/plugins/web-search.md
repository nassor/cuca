+++
title = "Web search"
description = "The live web search plugin: the three provider backends, the web_search and web_extract tools, and their request shapes."
template = "page.html"
weight = 8
+++

# Web search

<dl class="page-facts">
<dt>In one line</dt>
<dd>Resolves web_search and web_extract tool calls against a configured search backend.</dd>
<dt>You need</dt>
<dd>The <code>plugin-web-search</code> feature and an API key for the configured backend.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>WebSearchPlugin</code> or choosing a <code>WebSearchProvider</code>.</dd>
</dl>

`WebSearchPlugin` resolves `web_search` and `web_extract` tool calls a model issues against one configured backend: Firecrawl, Tavily, or DeepSeek Web Search. Only Firecrawl exposes a scrape endpoint, so `web_extract` returns `PluginError::NotSupported` on the other two before any network call. Reach for it to give a model live retrieval without wiring a bespoke HTTP tool for each search API.

```rust,name=Register the plugin and validate a tool call before any network call
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, WebSearchConfig, WebSearchPlugin, WebSearchProvider};
use serde_json::json;

let web_search = Arc::new(WebSearchPlugin::new(WebSearchConfig {
    provider: WebSearchProvider::Firecrawl,
    api_key: std::env::var("FIRECRAWL_API_KEY").unwrap_or_default(),
    base_url: None,
    max_results: 5,
}));

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .register_plugin(Arc::clone(&web_search) as Arc<dyn CucaPlugin>)
    .build()?;

let mut call = MessageContentBlock::ToolCall {
    id: "call_1".into(),
    name: "web_search".into(),
    arguments: json!({}),
};
web_search.on_stream_chunk(&mut call)?;
```

```text,name=A missing query argument never reaches the network
ToolResult { tool_call_id: "call_1", output: "validation failed for schema web_search: web_search requires a non-empty string `query` argument" }
```

## Entry types

`WebSearchPlugin`, `WebSearchConfig`, `WebSearchProvider`, `SearchResult`.

## `CucaPlugin`

`WebSearchPlugin` implements `CucaPlugin` with the plugin name `"web-search"`. It overrides `on_stream_chunk` only.

## Config

`WebSearchConfig` defaults: `provider: Firecrawl`, `api_key: ""` (must be set for live calls), `base_url: None`, `max_results: 5`.

## Providers

| `WebSearchProvider` | Search route | Auth | Page extraction |
|---|---|---|---|
| `Firecrawl` | `POST {base}/v1/search` | `Authorization: Bearer` | Yes, `POST {base}/v1/scrape` |
| `Tavily` | `POST {base}/search` | `api_key` field in the request body | No |
| `DeepSeekWebSearch` | `POST {base}/web-search` | `Authorization: Bearer` | No |

Default base URLs when `WebSearchConfig::base_url` is `None`: `https://api.firecrawl.dev` for Firecrawl, `https://api.tavily.com` for Tavily, `https://api.deepseek.com` for DeepSeek Web Search.

## Tools

| Tool | Arguments | Behavior |
|---|---|---|
| `web_search` | `query` (required, non-blank) | Returns normalized `SearchResult { title, url, snippet }` items as a `ToolResult` |
| `web_extract` | `url` (required, non-blank) | Returns extracted page text on Firecrawl; on Tavily and DeepSeek Web Search it returns `PluginError::NotSupported` before any network call, since only Firecrawl exposes a scrape endpoint |

## Capacity

No growth cap. The plugin holds only its configuration and an HTTP client; nothing accumulates between calls.
