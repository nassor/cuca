//! The stats stream consumer.
//!
//! A tokio task follows `GET /cuca/v1/stats/stream`, decodes each SSE event
//! into a [`StatsSnapshot`] and sends it to the UI loop over a bounded channel
//! next to the connection state. A broken stream reconnects with a capped
//! exponential backoff; the UI keeps showing the last snapshot meanwhile.

use std::time::{Duration, Instant};

use reqwest::{Client, StatusCode};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;

use super::TuiOptions;
use crate::server::error::ServerError;
use crate::server::stats::StatsSnapshot;

/// Updates buffered between the network task and the UI loop.
const CHANNEL_CAPACITY: usize = 16;
/// Longest wait between two connection attempts.
const MAX_BACKOFF: Duration = Duration::from_secs(15);
/// Longest SSE line or event accepted; a larger one is a protocol error.
const MAX_EVENT_BYTES: usize = 8 << 20;

/// Where the stream is.
#[derive(Debug, Clone, PartialEq)]
pub enum Connection {
    /// First attempt in progress.
    Connecting,
    /// Events are arriving.
    Connected,
    /// The stream broke; the next attempt starts at `retry_at`.
    Reconnecting {
        /// Attempts failed in a row.
        attempt: u32,
        /// When the next attempt starts.
        retry_at: Instant,
        /// Why the last attempt or stream failed.
        error: String,
    },
}

/// What the network task tells the UI loop.
#[derive(Debug)]
pub enum Update {
    /// A new snapshot.
    Snapshot(Box<StatsSnapshot>),
    /// A change of connection state.
    Connection(Connection),
}

/// Starts the network task. The task ends when the receiver is dropped or
/// the returned handle is aborted.
///
/// # Errors
///
/// [`ServerError::Config`] for a URL that is not `http://` or `https://`, an
/// interval below 100 ms, or an API key the HTTP client cannot send.
pub fn spawn(
    options: &TuiOptions,
) -> Result<(mpsc::Receiver<Update>, JoinHandle<()>), ServerError> {
    let url = stream_url(&options.url, options.interval_ms)?;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .build()
        .map_err(|error| ServerError::Config(format!("HTTP client: {error}")))?;
    let api_key = options.api_key.clone();
    // No data for this long on a live stream means the connection is dead.
    let stall =
        Duration::from_millis(options.interval_ms.saturating_mul(3)).max(Duration::from_secs(10));
    let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
    let handle = tokio::spawn(async move {
        if tx
            .send(Update::Connection(Connection::Connecting))
            .await
            .is_err()
        {
            return;
        }
        let mut attempt = 0u32;
        loop {
            let Err(failure) =
                follow(&client, &url, api_key.as_deref(), stall, &tx, &mut attempt).await;
            let error = match failure {
                // The receiver is gone: stop.
                Failure::Closed => return,
                Failure::Stream(error) => error,
            };
            attempt = attempt.saturating_add(1);
            let wait = backoff(attempt);
            let update = Update::Connection(Connection::Reconnecting {
                attempt,
                retry_at: Instant::now() + wait,
                error,
            });
            if tx.send(update).await.is_err() {
                return;
            }
            tokio::time::sleep(wait).await;
        }
    });
    Ok((rx, handle))
}

/// `<base>/cuca/v1/stats/stream?interval_ms=<n>`.
fn stream_url(base: &str, interval_ms: u64) -> Result<String, ServerError> {
    if interval_ms < 100 {
        return Err(ServerError::Config(
            "interval_ms must be at least 100".into(),
        ));
    }
    let base = base.trim().trim_end_matches('/');
    if !(base.starts_with("http://") || base.starts_with("https://")) {
        return Err(ServerError::Config(format!(
            "url `{base}` must start with http:// or https://"
        )));
    }
    Ok(format!(
        "{base}/cuca/v1/stats/stream?interval_ms={interval_ms}"
    ))
}

/// Seconds to wait after `attempt` failures in a row: 1, 2, 4, ... capped.
pub fn backoff(attempt: u32) -> Duration {
    let exponent = attempt.saturating_sub(1).min(16);
    Duration::from_secs(1u64 << exponent).min(MAX_BACKOFF)
}

enum Failure {
    /// The UI loop dropped its receiver.
    Closed,
    /// The stream could not be opened or broke.
    Stream(String),
}

/// One connection: opens the stream and forwards events until it fails.
/// Resets `attempt` on the first event, so a stream that works restarts the
/// backoff from a second.
async fn follow(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
    stall: Duration,
    tx: &mpsc::Sender<Update>,
    attempt: &mut u32,
) -> Result<std::convert::Infallible, Failure> {
    let mut request = client.get(url).header("accept", "text/event-stream");
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .map_err(|error| Failure::Stream(describe(&error)))?;
    let status = response.status();
    if !status.is_success() {
        return Err(Failure::Stream(describe_status(status)));
    }

    let mut stream = std::pin::pin!(response.bytes_stream());
    let mut decoder = SseDecoder::default();
    let mut events = Vec::new();
    let mut announced = false;
    loop {
        let chunk = match tokio::time::timeout(stall, stream.next()).await {
            Err(_) => return Err(Failure::Stream(format!("no data for {}s", stall.as_secs()))),
            Ok(None) => return Err(Failure::Stream("stream closed by the server".into())),
            Ok(Some(Err(error))) => return Err(Failure::Stream(describe(&error))),
            Ok(Some(Ok(chunk))) => chunk,
        };
        events.clear();
        decoder.push(&chunk, &mut events).map_err(Failure::Stream)?;
        for data in &events {
            let snapshot: StatsSnapshot = serde_json::from_str(data)
                .map_err(|error| Failure::Stream(format!("bad snapshot: {error}")))?;
            if !announced {
                announced = true;
                *attempt = 0;
                tx.send(Update::Connection(Connection::Connected))
                    .await
                    .map_err(|_| Failure::Closed)?;
            }
            tx.send(Update::Snapshot(Box::new(snapshot)))
                .await
                .map_err(|_| Failure::Closed)?;
        }
    }
}

fn describe(error: &reqwest::Error) -> String {
    if error.is_connect() {
        "connection refused or unreachable".to_string()
    } else if error.is_timeout() {
        "timed out".to_string()
    } else if error.is_body() || error.is_decode() {
        "connection lost while streaming".to_string()
    } else {
        error.to_string()
    }
}

fn describe_status(status: StatusCode) -> String {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            format!("HTTP {status}: the server wants an API key (--api-key)")
        }
        StatusCode::NOT_FOUND => {
            format!("HTTP {status}: stats API not enabled (start the server with --stats-api)")
        }
        _ => format!("HTTP {status}"),
    }
}

/// Splits an SSE byte stream into the `data` payload of each event.
///
/// Follows the SSE rules the stats stream needs: `\n`, `\r\n` and `\r` line
/// ends, `:` comment lines ignored, one optional space after `data:`, several
/// `data:` lines of one event joined with `\n`, an empty line ending the
/// event. Other fields (`event:`, `id:`, `retry:`) are ignored.
#[derive(Debug, Default)]
pub struct SseDecoder {
    line: Vec<u8>,
    data: String,
    has_data: bool,
    /// The previous byte was `\r`, so a `\n` right after it is the same line end.
    after_cr: bool,
}

impl SseDecoder {
    /// Feeds bytes; every event completed by them is appended to `events`.
    ///
    /// # Errors
    ///
    /// A message when one line or event outgrows the cap.
    pub fn push(&mut self, bytes: &[u8], events: &mut Vec<String>) -> Result<(), String> {
        for &byte in bytes {
            match byte {
                b'\n' if self.after_cr => self.after_cr = false,
                b'\n' | b'\r' => {
                    self.after_cr = byte == b'\r';
                    self.end_line(events);
                }
                _ => {
                    self.after_cr = false;
                    if self.line.len() + self.data.len() >= MAX_EVENT_BYTES {
                        return Err("event larger than 8 MiB".into());
                    }
                    self.line.push(byte);
                }
            }
        }
        Ok(())
    }

    fn end_line(&mut self, events: &mut Vec<String>) {
        if self.line.is_empty() {
            if self.has_data {
                events.push(std::mem::take(&mut self.data));
                self.has_data = false;
            }
            return;
        }
        let line = String::from_utf8_lossy(&self.line);
        if let Some(value) = line.strip_prefix("data:") {
            if self.has_data {
                self.data.push('\n');
            }
            self.data.push_str(value.strip_prefix(' ').unwrap_or(value));
            self.has_data = true;
        }
        drop(line);
        self.line.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(chunks: &[&[u8]]) -> Vec<String> {
        let mut decoder = SseDecoder::default();
        let mut events = Vec::new();
        for chunk in chunks {
            decoder.push(chunk, &mut events).unwrap();
        }
        events
    }

    #[test]
    fn an_event_split_across_chunks_decodes_once() {
        let events = decode(&[b"data: {\"a\"", b":1}\n", b"\ndata: 2\n\n"]);
        assert_eq!(events, ["{\"a\":1}", "2"]);
    }

    #[test]
    fn crlf_comments_and_other_fields_are_handled() {
        let events = decode(&[b": keep-alive\r\nevent: snapshot\r\ndata:x\r\ndata: y\r\n\r\n"]);
        assert_eq!(events, ["x\ny"]);
    }

    #[test]
    fn a_line_without_a_blank_line_is_not_an_event_yet() {
        assert!(decode(&[b"data: partial\n"]).is_empty());
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let seconds: Vec<u64> = (1..=7).map(|attempt| backoff(attempt).as_secs()).collect();
        assert_eq!(seconds, [1, 2, 4, 8, 15, 15, 15]);
        assert_eq!(backoff(u32::MAX), MAX_BACKOFF);
    }

    #[test]
    fn the_stream_url_needs_a_scheme_and_a_sane_interval() {
        assert_eq!(
            stream_url("http://127.0.0.1:8000/", 1000).unwrap(),
            "http://127.0.0.1:8000/cuca/v1/stats/stream?interval_ms=1000"
        );
        assert!(stream_url("127.0.0.1:8000", 1000).is_err());
        assert!(stream_url("http://h", 10).is_err());
    }

    /// A one-shot HTTP server: answers one request with two snapshots as an
    /// event stream, then closes. Returns its address and the request head.
    async fn serve_two_snapshots() -> (std::net::SocketAddr, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut head = vec![0u8; 4096];
            let read = socket.read(&mut head).await.unwrap();
            let snapshot =
                serde_json::to_string(&crate::server::tui::tests::server_snapshot()).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n\
                 data: {snapshot}\n\n: keep-alive\n\ndata: {snapshot}\n\n"
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            String::from_utf8_lossy(&head[..read]).into_owned()
        });
        (address, handle)
    }

    #[tokio::test]
    async fn the_stream_delivers_snapshots_with_the_key_then_reports_the_loss() {
        let (address, server) = serve_two_snapshots().await;
        let options = TuiOptions {
            url: format!("http://{address}"),
            api_key: Some("secret".into()),
            interval_ms: 250,
        };
        let (mut updates, task) = spawn(&options).unwrap();

        let mut seen = Vec::new();
        while let Some(update) = updates.recv().await {
            let done = matches!(update, Update::Connection(Connection::Reconnecting { .. }));
            seen.push(update);
            if done {
                break;
            }
        }
        task.abort();

        let head = server.await.unwrap().to_ascii_lowercase();
        assert!(
            head.starts_with("get /cuca/v1/stats/stream?interval_ms=250 "),
            "{head}"
        );
        assert!(head.contains("authorization: bearer secret"), "{head}");

        let kinds: Vec<&str> = seen
            .iter()
            .map(|update| match update {
                Update::Connection(Connection::Connecting) => "connecting",
                Update::Connection(Connection::Connected) => "connected",
                Update::Connection(Connection::Reconnecting { error, .. }) => {
                    assert!(error.contains("closed"), "{error}");
                    "reconnecting"
                }
                Update::Snapshot(_) => "snapshot",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "connecting",
                "connected",
                "snapshot",
                "snapshot",
                "reconnecting"
            ]
        );
    }
}
