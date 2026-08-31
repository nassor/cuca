//! DeepSeek provider adapter, gated to `provider-deepseek`.
//!
//! DeepSeek serves two wire protocols behind one base-URL decision:
//!
//! - **Native OpenAI-compatible** endpoint (`https://api.deepseek.com/v1`):
//!   reuses the shared [`openai_compat`](crate::provider::openai_compat)
//!   adapter, so DeepSeek's `reasoning_content` fields flow through the shared
//!   translator's `Thinking` handling.
//! - **Anthropic bridge** (`https://api.deepseek.com/anthropic`): speaks the
//!   Anthropic Messages API
//!   ([`build_anthropic_request`]/[`anthropic_stream`]) and expects Claude-style
//!   model ids, which are translated to DeepSeek ids by
//!   [`translate_bridge_model`].
//!
//! The route is chosen by [`is_anthropic_bridge`] on the configured base URL.

#![cfg(feature = "provider-deepseek")]

use crate::client::{CucaClient, ProviderDispatch};
use crate::error::CucaError;
use crate::provider::anthropic::{AnthropicAuth, anthropic_stream, build_anthropic_request};
use crate::provider::openai_compat::{OpenAiCompatConfig, openai_compat_stream};
use crate::request::{PromptCacheDirective, UnifiedRequest};

/// Map a Claude-style model id to its DeepSeek counterpart for the bridge.
///
/// `claude-opus` -> `deepseek-v4-pro`; `claude-sonnet` and
/// `claude-haiku` -> `deepseek-v4-flash`; anything else passes through
/// unchanged.
pub fn translate_bridge_model(model: &str) -> String {
    match model {
        "claude-opus" => "deepseek-v4-pro".into(),
        "claude-sonnet" | "claude-haiku" => "deepseek-v4-flash".into(),
        other => other.into(),
    }
}

/// True when the configured base URL targets the Anthropic bridge.
///
/// Both the DeepSeek-served bridge host (`api.deepseek.com/anthropic`) and a
/// generic path that ends in `/anthropic` (e.g. a local proxy mirroring the
/// bridge) count as the bridge.
pub fn is_anthropic_bridge(base_url: &str) -> bool {
    base_url.contains("api.deepseek.com/anthropic") || base_url.ends_with("/anthropic")
}

/// Stage a request for the bridge, consuming it.
///
/// Translates the Claude-style model id to its DeepSeek counterpart and forces
/// [`PromptCacheDirective::Disabled`]. DeepSeek reuses the Anthropic protocol
/// module, but module sharing is not prompt-cache support: the directive is
/// dropped here explicitly so no bridge request can acquire a beta header,
/// `cache_control` block, or breakpoint validation error, regardless of what
/// the caller asked for.
///
/// Takes `req` by value: the caller owns it and never needs the pre-staging
/// version, so staging is two field writes instead of a full clone of every
/// message and content block.
pub(crate) fn bridge_unified_request(mut req: UnifiedRequest) -> UnifiedRequest {
    req.model = translate_bridge_model(&req.model);
    req.prompt_cache = PromptCacheDirective::Disabled;
    req
}

/// Build the Anthropic-shape bridge body for a [`UnifiedRequest`].
///
/// Testable seam used by [`CucaClient::dispatch_deepseek`]: stages the request
/// through [`bridge_unified_request`], then builds the Anthropic-shape body.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "test-only seam: dispatch calls bridge_unified_request and anthropic_stream directly"
    )
)]
pub(crate) fn bridge_request(req: UnifiedRequest) -> Result<serde_json::Value, CucaError> {
    build_anthropic_request(&bridge_unified_request(req))
}

impl CucaClient {
    /// Dispatch a unified request to DeepSeek.
    ///
    /// Defaults the base URL to the native OpenAI-compatible endpoint
    /// (`https://api.deepseek.com/v1`) when the builder did not set one. When
    /// the base URL targets the Anthropic bridge ([`is_anthropic_bridge`]), the
    /// request is sent through [`anthropic_stream`] with the model
    /// id translated and `x-api-key` auth (an api key is required); otherwise
    /// it goes through [`openai_compat_stream`]. Called by
    /// `generate_stream` under the `provider-deepseek` feature.
    pub(crate) async fn dispatch_deepseek(
        &self,
        req: UnifiedRequest,
    ) -> Result<ProviderDispatch, CucaError> {
        let base = if self.base_url().is_empty() {
            "https://api.deepseek.com/v1".to_string()
        } else {
            self.base_url().to_string()
        };

        if is_anthropic_bridge(&base) {
            let key = self
                .api_key()
                .ok_or_else(|| {
                    CucaError::Config("deepseek anthropic bridge requires an api key".into())
                })?
                .to_string();
            let auth = AnthropicAuth::ApiKey(key);
            // The bridge URL is used as-is: no `/v1` suffix logic.
            let bridged = bridge_unified_request(req);
            anthropic_stream(self.http_client(), &base, &auth, bridged).await
        } else {
            openai_compat_stream(
                self.http_client(),
                &OpenAiCompatConfig {
                    base_url: base,
                    api_key: self.api_key().map(str::to_string),
                    model: req.model.clone(),
                },
                req,
            )
            .await
        }
    }
}

#[cfg(all(test, feature = "provider-deepseek"))]
mod tests {
    use serde_json::json;

    use crate::provider::openai_compat::build_chat_completion_body;
    use crate::request::{PromptCacheBreakpoint, ThinkingEffort, UnifiedRequest};
    use crate::types::ProviderEndpoint;

    use super::*;

    // --- translate_bridge_model ---

    #[test]
    fn translates_claude_opus_to_deepseek_v4_pro() {
        assert_eq!(translate_bridge_model("claude-opus"), "deepseek-v4-pro");
    }

    #[test]
    fn translates_claude_sonnet_and_haiku_to_deepseek_v4_flash() {
        assert_eq!(translate_bridge_model("claude-sonnet"), "deepseek-v4-flash");
        assert_eq!(translate_bridge_model("claude-haiku"), "deepseek-v4-flash");
    }

    #[test]
    fn passes_unknown_models_through_unchanged() {
        assert_eq!(translate_bridge_model("deepseek-v4-pro"), "deepseek-v4-pro");
        assert_eq!(translate_bridge_model("gpt-4o"), "gpt-4o");
        assert_eq!(translate_bridge_model(""), "");
    }

    // --- is_anthropic_bridge ---

    #[test]
    fn detects_deepseek_bridge_host() {
        assert!(is_anthropic_bridge("https://api.deepseek.com/anthropic"));
    }

    #[test]
    fn does_not_mistake_native_or_local_urls_for_bridge() {
        assert!(!is_anthropic_bridge("https://api.deepseek.com/v1"));
        assert!(!is_anthropic_bridge("http://localhost:11434/v1"));
    }

    #[test]
    fn detects_generic_trailing_anthropic_path() {
        assert!(is_anthropic_bridge("http://localhost:8080/anthropic"));
    }

    // --- bridge_request ---

    #[test]
    fn bridge_request_translates_model_and_builds_anthropic_body() {
        let req = UnifiedRequest::new("claude-sonnet").add_user_message("hi");
        let body = bridge_request(req).expect("bridge body must build");

        assert_eq!(body["model"], json!("deepseek-v4-flash"));
        assert!(body["messages"].is_array());
        assert_eq!(body["max_tokens"], json!(1024));
        assert_eq!(body["stream"], json!(true));
    }

    #[test]
    fn bridge_request_translates_opus_model() {
        let req = UnifiedRequest::new("claude-opus").add_user_message("hi");
        let body = bridge_request(req).expect("bridge body must build");

        assert_eq!(body["model"], json!("deepseek-v4-pro"));
    }

    // --- native path ---

    #[test]
    fn native_body_is_openai_shaped_with_deepseek_model() {
        let req = UnifiedRequest::new("deepseek-v4-pro").add_user_message("hi");
        let body = build_chat_completion_body(&req);

        assert_eq!(body["model"], json!("deepseek-v4-pro"));
        assert_eq!(body["stream"], json!(true));
        assert!(body["messages"].is_array());
    }

    // --- thinking ---

    #[test]
    fn native_body_thinking_enabled_emits_thinking_mode() {
        let mut req = UnifiedRequest::new("deepseek-reasoner")
            .add_user_message("hi")
            .enable_thinking(Some(ThinkingEffort::High));
        req.provider = ProviderEndpoint::DeepSeek;
        let body = build_chat_completion_body(&req);

        assert_eq!(body["thinking"], json!({ "type": "enabled" }));
        // DeepSeek has no effort knob.
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn native_body_thinking_disabled_emits_disabled_mode() {
        let mut req = UnifiedRequest::new("deepseek-reasoner")
            .add_user_message("hi")
            .disable_thinking();
        req.provider = ProviderEndpoint::DeepSeek;
        let body = build_chat_completion_body(&req);

        assert_eq!(body["thinking"], json!({ "type": "disabled" }));
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn bridge_request_unchanged_when_thinking_unset() {
        let req = UnifiedRequest::new("claude-sonnet").add_user_message("hi");
        let body = bridge_request(req).expect("bridge body must build");

        assert_eq!(body["model"], json!("deepseek-v4-flash"));
        assert!(body.get("thinking").is_none());
    }

    // --- prompt cache: DeepSeek is unsupported on both paths ---

    /// A DeepSeek request that asks for ephemeral caching at coordinates that
    /// would be valid (and one that would be an error) on Anthropic.
    fn ephemeral_request(model: &str) -> UnifiedRequest {
        let mut req = UnifiedRequest::new(model)
            .add_system_message("policy")
            .add_user_message("hi")
            .with_prompt_cache(PromptCacheDirective::Ephemeral {
                breakpoints: vec![
                    PromptCacheBreakpoint {
                        message_index: 0,
                        block_index: 0,
                    },
                    PromptCacheBreakpoint {
                        message_index: 99,
                        block_index: 99,
                    },
                ],
            });
        req.provider = ProviderEndpoint::DeepSeek;
        req
    }

    /// The bridge reuses the Anthropic protocol module, but module sharing is
    /// not prompt-cache support: the bridge body carries no `cache_control`
    /// and the request asks for no beta header, even with out-of-range
    /// breakpoints.
    #[test]
    fn bridge_request_never_emits_cache_control_or_betas() {
        let req = ephemeral_request("claude-sonnet");
        let body =
            bridge_request(req.clone()).expect("an unsupported directive must not fail the bridge");

        assert_eq!(body["model"], json!("deepseek-v4-flash"));
        assert!(
            !serde_json::to_string(&body)
                .unwrap()
                .contains("cache_control"),
            "deepseek bridge must not emit cache_control: {body}"
        );
        // System stays the Anthropic scalar form, never array-form blocks.
        assert_eq!(body["system"], json!("policy"));
        assert!(
            crate::provider::anthropic::prompt_cache_betas(&req).is_empty(),
            "no anthropic-beta for a DeepSeek endpoint"
        );
    }

    /// The bridge stages its own request clone through the disabled directive,
    /// so nothing downstream can observe cache intent.
    #[test]
    fn bridge_request_forces_the_directive_to_disabled() {
        let req = ephemeral_request("claude-opus");
        let staged = bridge_unified_request(req.clone());

        assert_eq!(staged.model, "deepseek-v4-pro");
        assert_eq!(staged.prompt_cache, PromptCacheDirective::Disabled);
        assert_eq!(staged.provider, ProviderEndpoint::DeepSeek);
        assert!(crate::provider::anthropic::prompt_cache_betas(&staged).is_empty());
        assert_eq!(
            staged.messages, req.messages,
            "the wire messages are otherwise unchanged"
        );
    }

    #[test]
    fn native_body_never_emits_prompt_cache_fields() {
        let req = ephemeral_request("deepseek-v4-pro");
        let body = build_chat_completion_body(&req);
        let text = serde_json::to_string(&body).unwrap();

        assert!(
            !text.contains("cache_control") && !text.contains("prompt_cache"),
            "the native OpenAI-compatible body must be unchanged: {body}"
        );
        // Byte-identical to the same request without a directive.
        let mut plain = req.clone();
        plain.prompt_cache = PromptCacheDirective::Disabled;
        assert_eq!(body, build_chat_completion_body(&plain));
    }

    /// DeepSeek never asks for provider caching, so a bridge response reports
    /// no normalized usage: the frame shape the bridge actually returns (token
    /// counts without cache counters) yields `None`.
    #[test]
    fn deepseek_reports_no_normalized_prompt_cache_usage() {
        let mut translator = crate::provider::anthropic::AnthropicTranslator::new();
        let block = translator
            .translate(&crate::sse::SseEvent {
                event: "message_start".to_string(),
                data: r#"{"type":"message_start","message":{"role":"assistant","content":[],"usage":{"input_tokens":7,"output_tokens":3}}}"#
                    .to_string(),
                id: None,
                retry: None,
            })
            .expect("a control frame never fails");

        assert!(block.is_none());
        assert_eq!(translator.take_prompt_cache_usage(), None);
    }
}
