//! K/V transfer between model servers: the export route a prefill node
//! serves and the import a decode node runs before it generates.
//!
//! A balancer with a `disaggregate` section names a prefill node in the
//! `x-cuca-prefill` header of a request it sends to a decode node. When that
//! model has a `kv_import` section, [`remote_prefill`] asks the peer for the
//! K/V rows of the prompt (`POST /cuca/v1/kv/export`) and files them in the
//! engine's prefix store, so the request then reads them as a cache hit. It
//! never fails the request: whatever goes wrong, the decode node prefills the
//! prompt itself.
//!
//! The frames are those of [`kvframe`]; the body of an export is the frames
//! back to back and at most one frame is buffered besides the network chunk
//! being read.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::extract::rejection::BytesRejection;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use super::balancer::config::normalize_upstream;
use super::config::KvImportConfig;
use super::http::{ApiError, body_rejection};
use super::registry::{LoadedModel, ModelSlot, ResolveError};
use super::state::ServerState;
use crate::provider::local::KvImport;
use crate::provider::local::kvframe;
use crate::{CucaError, PromptTokens};

/// The largest frame an export may carry: a frame holds the rows of at most
/// one chunk, far below this.
const MAX_FRAME_BYTES: u64 = 1 << 30;

/// How a remote prefill ended; the discriminant is the slot of
/// `ModelMetrics::kv_imports`, in the order of `KV_IMPORT_OUTCOMES`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// The peer's rows are in the prefix store.
    Ok = 0,
    /// The prompt is too short beyond the local cache to be worth a transfer.
    Skipped = 1,
    /// The model's drafter replays the prompt, so imported rows would be
    /// computed again.
    Drafter = 2,
    /// The named peer is not in the model's `kv_import.peers`.
    Unlisted = 3,
    /// The transfer failed; the request prefills locally.
    Failed = 4,
}

/// Import the K/V of `prompt` from `peer` into `loaded`'s prefix store.
/// `authorization` is the request's own `Authorization` value, sent on to the
/// peer. Counts the outcome in the model's metrics and never fails.
pub(crate) async fn remote_prefill(
    state: &ServerState,
    slot: &ModelSlot,
    loaded: &LoadedModel,
    cfg: &KvImportConfig,
    peer: &str,
    prompt: &PromptTokens,
    authorization: Option<&HeaderValue>,
) -> Outcome {
    let metrics = &slot.metrics;
    let listed = normalize_upstream(peer)
        .ok()
        .filter(|(_, url)| cfg.peers.contains(url));
    let (outcome, moved) = match listed {
        None => (Outcome::Unlisted, None),
        Some((_, url)) => match plan(loaded, cfg, prompt).await {
            Err(error) => failed(slot, &url, &error),
            Ok(Err(outcome)) => (outcome, None),
            Ok(Ok(have)) => {
                let pull = Pull {
                    peer: &url,
                    prompt,
                    have,
                    authorization,
                };
                match fetch(state, slot, loaded, cfg, pull).await {
                    Ok(moved) => (Outcome::Ok, Some(moved)),
                    Err(error) => failed(slot, &url, &error),
                }
            }
        },
    };
    metrics.kv_imports[outcome as usize].fetch_add(1, Ordering::Relaxed);
    if let Some((tokens, bytes)) = moved {
        metrics
            .kv_import_tokens
            .fetch_add(tokens as u64, Ordering::Relaxed);
        metrics.kv_import_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
    outcome
}

fn failed(slot: &ModelSlot, peer: &str, error: &str) -> (Outcome, Option<(usize, u64)>) {
    tracing::warn!(
        model = %slot.name,
        peer,
        error,
        "K/V import failed; the request prefills locally"
    );
    (Outcome::Failed, None)
}

/// Whether a transfer is worth making: `Ok(Ok(have))` with the tokens the
/// prefix store already holds, `Ok(Err(outcome))` for a skip, `Err` when the
/// engine cannot answer.
async fn plan(
    loaded: &LoadedModel,
    cfg: &KvImportConfig,
    prompt: &PromptTokens,
) -> Result<Result<usize, Outcome>, String> {
    if !loaded.model.imports_kv() {
        return Ok(Err(Outcome::Drafter));
    }
    let ids = prompt.ids();
    let have = loaded
        .model
        .kv_hit(ids.to_vec())
        .await
        .map_err(|e| e.to_string())?;
    if ids.len().saturating_sub(1).saturating_sub(have) < cfg.min_tokens {
        return Ok(Err(Outcome::Skipped));
    }
    Ok(Ok(have))
}

/// What one pull asks of a peer.
struct Pull<'a> {
    /// The peer's normalized base URL.
    peer: &'a str,
    prompt: &'a PromptTokens,
    /// Tokens the prefix store already holds.
    have: usize,
    authorization: Option<&'a HeaderValue>,
}

/// Ask the peer for the rows of the prompt after the first `have` tokens and
/// write them to the engine. Returns the tokens and bytes moved.
async fn fetch(
    state: &ServerState,
    slot: &ModelSlot,
    loaded: &LoadedModel,
    cfg: &KvImportConfig,
    pull: Pull<'_>,
) -> Result<(usize, u64), String> {
    let Pull {
        peer,
        prompt,
        have,
        authorization,
    } = pull;
    let ids = prompt.ids();
    let mut request = state
        .peer_client
        .post(format!("{peer}/cuca/v1/kv/export"))
        .timeout(cfg.timeout)
        .json(&json!({
            "model": &*slot.name,
            "fingerprint": format!("{:016x}", loaded.model.fingerprint()),
            "from": have,
            "ids": ids,
        }));
    if let Some(authorization) = authorization {
        request = request.header(AUTHORIZATION, authorization);
    }
    let response = request
        .send()
        .await
        .map_err(|e| format!("the peer is not reachable: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        let text: String = text.chars().take(300).collect();
        return Err(format!("the peer answered {status}: {text}"));
    }

    let mut stream = response.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut import: Option<KvImport> = None;
    let (mut tokens, mut bytes, mut filed) = (0usize, 0u64, false);
    'read: while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk.map_err(|e| format!("the transfer broke: {e}"))?);
        while let Some(frame) = take_frame(&mut buf)? {
            bytes += frame.len() as u64;
            if import.is_none() {
                let (kind, payload) = kvframe::parse(&frame).map_err(|e| e.to_string())?;
                if kind != kvframe::HEADER {
                    return Err("the peer's first frame is not the header".into());
                }
                let header = kvframe::decode_header(payload).map_err(|e| e.to_string())?;
                if header.from != have {
                    return Err(format!(
                        "the peer exports from {} but {have} tokens were asked for",
                        header.from
                    ));
                }
                tokens = header
                    .to
                    .checked_sub(header.from)
                    .ok_or("the peer's range is empty")?;
                import = Some(
                    loaded
                        .model
                        .kv_import(ids.to_vec(), header.from, header.to)
                        .await
                        .map_err(|e| e.to_string())?,
                );
            }
            if let Some(import) = import.as_mut() {
                filed = import
                    .write_frame(&frame)
                    .await
                    .map_err(|e| e.to_string())?;
                if filed {
                    break 'read;
                }
            }
        }
    }
    if !filed {
        return Err("the peer's export ended before its end frame".into());
    }
    Ok((tokens, bytes))
}

/// Cut the first whole frame off `buf`: `None` while it is incomplete.
fn take_frame(buf: &mut Vec<u8>) -> Result<Option<Vec<u8>>, String> {
    let Some(envelope) = buf.get(..kvframe::ENVELOPE) else {
        return Ok(None);
    };
    let len = <[u8; 8]>::try_from(&envelope[1..])
        .map(u64::from_le_bytes)
        .map_err(|_| "a K/V frame is truncated".to_string())?;
    if len > MAX_FRAME_BYTES {
        return Err(format!(
            "a K/V frame announces {len} bytes; at most {MAX_FRAME_BYTES}"
        ));
    }
    let total = kvframe::ENVELOPE + len as usize;
    if buf.len() < total {
        return Ok(None);
    }
    Ok(Some(buf.drain(..total).collect()))
}

/// The body of `POST /cuca/v1/kv/export`.
#[derive(Deserialize)]
struct ExportRequest {
    model: String,
    /// The importer's [`LocalModel::fingerprint`](crate::LocalModel::fingerprint)
    /// as 16 hex digits.
    fingerprint: String,
    /// Tokens the importer already holds.
    from: usize,
    ids: Vec<u32>,
}

/// `POST /cuca/v1/kv/export`: prefill `ids` as a request would and stream
/// the K/V rows after the first `from` tokens, as frames, to a decode node.
pub(crate) async fn export(
    State(state): State<Arc<ServerState>>,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    let body = body.map_err(body_rejection)?;
    let request: ExportRequest = serde_json::from_slice(&body)
        .map_err(|e| ApiError::invalid_request(format!("invalid request body: {e}")))?;
    let slot = state
        .registry
        .resolve(Some(&request.model))
        .map_err(|e| match e {
            ResolveError::Missing => ApiError::invalid_request("`model` is required"),
            ResolveError::Unknown(name) => ApiError::model_not_found(&name),
        })?
        .clone();
    let loaded = slot
        .ready()
        .ok_or_else(|| ApiError::model_unavailable(&slot))?;
    let theirs = u64::from_str_radix(&request.fingerprint, 16)
        .map_err(|_| ApiError::invalid_request("`fingerprint` must be 16 hex digits"))?;
    if theirs != loaded.model.fingerprint() {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "kv_fingerprint_mismatch",
            "the importer's model is not the same file or cache layout".into(),
        ));
    }
    let mut export = loaded
        .model
        .kv_export(request.ids, request.from)
        .await
        .map_err(|e| export_error(&e))?;
    let (from, to) = export.range();
    let metrics = &slot.metrics;
    metrics.kv_exports.fetch_add(1, Ordering::Relaxed);
    metrics
        .kv_export_tokens
        .fetch_add(to.saturating_sub(from) as u64, Ordering::Relaxed);

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(2);
    tokio::spawn(async move {
        while let Some(frame) = export.next_frame().await {
            let item = match frame {
                Ok(frame) => {
                    slot.metrics
                        .kv_export_bytes
                        .fetch_add(frame.len() as u64, Ordering::Relaxed);
                    Ok(Bytes::from(frame))
                }
                Err(error) => Err(std::io::Error::other(error.to_string())),
            };
            let failed = item.is_err();
            // A closed channel is a peer that went away: the export is
            // dropped here and its blocks go back to the engine.
            if tx.send(item).await.is_err() || failed {
                break;
            }
        }
    });
    let mut response = Response::new(Body::from_stream(ReceiverStream::new(rx)));
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    Ok(response.into_response())
}

fn export_error(error: &CucaError) -> ApiError {
    match error {
        CucaError::Config(message) => ApiError::new(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "kv_export_disabled",
            message.clone(),
        ),
        CucaError::Provider { message, .. } if message.contains("K/V export slots are busy") => {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "unavailable_error",
                "kv_export_busy",
                message.clone(),
            )
        }
        CucaError::Provider { message, .. } if message.contains("shut down") => ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable_error",
            "engine_unavailable",
            message.clone(),
        ),
        other => ApiError::invalid_request(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::{Value, json};

    use super::*;
    use crate::LocalModelConfig;
    use crate::provider::local::engine::testing::{engine_at, tiny_file, tiny_local_model};
    use crate::server::config::{ModelSpec, ServerConfig};
    use crate::server::http::router;
    use crate::server::registry::ModelRegistry;
    use crate::sysmon::SystemSampler;

    /// A server over the tiny model `file` serving `m`, exporting K/V when
    /// `exports`, importing from `peers` when given.
    struct Node {
        base: String,
        fingerprint: u64,
    }

    async fn node(path: &std::path::Path, exports: bool, peers: &[&str]) -> Node {
        let model = tiny_local_model(path.to_path_buf(), 32, move |path| {
            let engine = engine_at(path, 32);
            if exports {
                engine.with_exports(2)
            } else {
                engine
            }
        })
        .await;
        let fingerprint = model.fingerprint();
        let mut spec = ModelSpec::new(Some("m".into()), LocalModelConfig::default());
        if !peers.is_empty() {
            spec.kv_import = Some(KvImportConfig {
                peers: peers.iter().map(|p| (*p).to_string()).collect(),
                min_tokens: 16,
                ..KvImportConfig::default()
            });
        }
        let config = ServerConfig::new("127.0.0.1:0".parse().unwrap(), vec![spec.clone()]);
        let registry = ModelRegistry::ready_spec(&spec, model);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let state = Arc::new(ServerState::new(
            &config,
            addr,
            registry,
            SystemSampler::start(Duration::from_secs(1)),
            crate::server::state::peer_client().unwrap(),
        ));
        tokio::spawn(async move {
            let _ = axum::serve(listener, router(state)).await;
        });
        Node {
            base: format!("http://{addr}"),
            fingerprint,
        }
    }

    /// A URL nothing listens on.
    fn dead_url() -> String {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    }

    /// 21 characters of its alphabet: the tiny tokenizer makes a token of each, plus the
    /// start token, so one block of 16 can move.
    const LONG: &str = "abcdefgabcdefgabcdefg";

    async fn chat(base: &str, prompt: &str, prefill: Option<&str>) -> Value {
        let mut request = crate::tls::http_client()
            .post(format!("{base}/v1/chat/completions"))
            .json(&json!({
                "model": "m",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2,
            }));
        if let Some(prefill) = prefill {
            request = request.header("x-cuca-prefill", prefill);
        }
        let response = request.send().await.unwrap();
        assert_eq!(response.status(), 200);
        response.json().await.unwrap()
    }

    async fn metrics(base: &str) -> String {
        crate::tls::http_client()
            .get(format!("{base}/metrics"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }

    fn imports(text: &str, outcome: &str) -> String {
        let prefix = format!("cuca_kv_imports_total{{model=\"m\",outcome=\"{outcome}\"}} ");
        let line = text.lines().find(|l| l.starts_with(&prefix));
        line.map_or_else(String::new, |l| l[prefix.len()..].to_string())
    }

    async fn export_call(base: &str, fingerprint: &str, ids: &[u32]) -> reqwest::Response {
        crate::tls::http_client()
            .post(format!("{base}/cuca/v1/kv/export"))
            .json(&json!({"model": "m", "fingerprint": fingerprint, "from": 0, "ids": ids}))
            .send()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn a_decode_node_imports_the_prefill_nodes_rows_and_answers_alike() {
        let file = tiny_file();
        let p = node(&file.path, true, &[]).await;
        let d = node(&file.path, false, &[&p.base]).await;
        let plain = node(&file.path, false, &[]).await;

        let want = chat(&plain.base, LONG, None).await;
        let got = chat(&d.base, LONG, Some(&p.base)).await;
        assert_eq!(
            got["choices"][0]["message"]["content"],
            want["choices"][0]["message"]["content"]
        );
        assert_eq!(
            got["usage"]["prompt_tokens"],
            want["usage"]["prompt_tokens"]
        );

        let d_text = metrics(&d.base).await;
        assert_eq!(imports(&d_text, "ok"), "1", "{d_text}");
        assert!(!d_text.contains("cuca_kv_import_tokens_total{model=\"m\"} 0\n"));
        assert!(!d_text.contains("cuca_kv_import_bytes_total{model=\"m\"} 0\n"));
        let p_text = metrics(&p.base).await;
        assert!(
            p_text.contains("cuca_kv_exports_total{model=\"m\"} 1\n"),
            "{p_text}"
        );
        assert!(!p_text.contains("cuca_kv_export_bytes_total{model=\"m\"} 0\n"));

        // The imported rows were read as a cache hit.
        assert!(
            !d_text.contains("cuca_prompt_tokens_cached_total{model=\"m\"} 0\n"),
            "{d_text}"
        );

        // A short prompt is prefilled locally.
        chat(&d.base, "abc", Some(&p.base)).await;
        assert_eq!(imports(&metrics(&d.base).await, "skipped"), "1");

        // Without a `kv_import` section the header is ignored.
        let got = chat(&plain.base, LONG, Some(&p.base)).await;
        assert_eq!(
            got["choices"][0]["message"]["content"],
            want["choices"][0]["message"]["content"]
        );
        assert_eq!(imports(&metrics(&plain.base).await, "ok"), "0");
    }

    #[tokio::test]
    async fn a_peer_that_is_not_listed_or_is_gone_costs_nothing_but_a_count() {
        let file = tiny_file();
        let p = node(&file.path, true, &[]).await;
        let gone = dead_url();
        let d = node(&file.path, false, &[&p.base, &gone]).await;
        let plain = node(&file.path, false, &[]).await;
        let want = chat(&plain.base, LONG, None).await;

        // A cold prompt, so the transfer is attempted and fails.
        let got = chat(&d.base, LONG, Some(&gone)).await;
        assert_eq!(
            got["choices"][0]["message"]["content"],
            want["choices"][0]["message"]["content"]
        );
        let got = chat(&d.base, LONG, Some("http://127.0.0.1:1")).await;
        assert_eq!(
            got["choices"][0]["message"]["content"],
            want["choices"][0]["message"]["content"]
        );

        let text = metrics(&d.base).await;
        assert_eq!(imports(&text, "unlisted"), "1", "{text}");
        assert_eq!(imports(&text, "failed"), "1", "{text}");
        assert_eq!(imports(&text, "ok"), "0", "{text}");
        assert!(
            metrics(&p.base)
                .await
                .contains("cuca_kv_exports_total{model=\"m\"} 0\n")
        );
    }

    #[tokio::test]
    async fn the_export_route_refuses_a_foreign_model_and_a_node_that_does_not_export() {
        let file = tiny_file();
        let p = node(&file.path, true, &[]).await;
        let plain = node(&file.path, false, &[]).await;
        let ids: Vec<u32> = (0..20).map(|i| 4 + i % 8).collect();
        let right = format!("{:016x}", p.fingerprint);

        let r = export_call(&p.base, &format!("{:016x}", p.fingerprint ^ 1), &ids).await;
        assert_eq!(r.status(), 409);
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["code"], "kv_fingerprint_mismatch");

        let r = export_call(&plain.base, &right, &ids).await;
        assert_eq!(r.status(), 409);
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["code"], "kv_export_disabled");

        let r = export_call(&p.base, "not-hex", &ids).await;
        assert_eq!(r.status(), 400);

        let r = export_call(&p.base, &right, &ids).await;
        assert_eq!(r.status(), 200);
        assert_eq!(r.headers()["content-type"], "application/octet-stream");
        let body = r.bytes().await.unwrap();
        assert_eq!(body[0], kvframe::HEADER);
        assert_eq!(body[body.len() - kvframe::ENVELOPE], kvframe::END);
    }

    #[test]
    fn frames_are_cut_from_a_buffer_and_oversized_ones_refused() {
        let mut frame = vec![kvframe::END];
        frame.extend_from_slice(&3u64.to_le_bytes());
        frame.extend_from_slice(&[7, 8, 9]);
        let mut buf = frame[..4].to_vec();
        assert_eq!(take_frame(&mut buf), Ok(None));
        buf.extend_from_slice(&frame[4..]);
        buf.push(0xAA);
        assert_eq!(take_frame(&mut buf), Ok(Some(frame)));
        assert_eq!(buf, [0xAA]);

        let mut huge = vec![kvframe::ROWS];
        huge.extend_from_slice(&(MAX_FRAME_BYTES + 1).to_le_bytes());
        assert!(take_frame(&mut huge).unwrap_err().contains("at most"));
    }
}
