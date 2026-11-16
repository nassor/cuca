//! The HTTP transport of a layer stage: one `POST` per frame to the worker's
//! model server.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{StageCounters, StageTransport};
use crate::error::CucaError;
use crate::provider::local::{RemoteStage, STAGE_COMPUTE_HEADER};

/// How long a connect may take before the stage counts as unreachable.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

/// Posts frames to `{url}/cuca/v1/stage[/{model}]` and blocks the calling
/// thread on the answer. It owns a single-threaded runtime, so it must be
/// created and called from a plain thread (the engine thread), never from
/// inside another runtime.
pub(crate) struct HttpTransport {
    /// The base URL as configured, without a trailing `/`; also the name of
    /// the stage in messages.
    url: String,
    /// The full endpoint the frames go to.
    endpoint: String,
    key: Option<String>,
    timeout: Duration,
    client: reqwest::Client,
    rt: tokio::runtime::Runtime,
    stats: Arc<StageCounters>,
}

impl HttpTransport {
    /// A transport for `stage`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the HTTP client or its runtime cannot be
    /// built.
    pub(crate) fn new(stage: &RemoteStage) -> Result<Self, CucaError> {
        let url = stage.url.trim_end_matches('/').to_owned();
        let endpoint = match &stage.model {
            Some(model) => format!("{url}/cuca/v1/stage/{}", percent_encode(model)),
            None => format!("{url}/cuca/v1/stage"),
        };
        let client = crate::tls::http_client_builder()
            .connect_timeout(CONNECT_TIMEOUT)
            // Passes are far apart, and the runtime drives no connection
            // between calls, so an idle connection would die unnoticed and
            // fail the next pass: every call opens its own (a TCP handshake
            // costs nothing next to a forward pass).
            .pool_max_idle_per_host(0)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| super::super::err(format!("stage {url}: cannot build client: {e}")))?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| super::super::err(format!("stage {url}: cannot build runtime: {e}")))?;
        Ok(Self {
            url,
            endpoint,
            key: stage.api_key.clone(),
            timeout: stage.timeout,
            client,
            rt,
            stats: Arc::default(),
        })
    }

    /// The error a failed request or body read becomes.
    fn transport_err(&self, e: &reqwest::Error) -> CucaError {
        let url = &self.url;
        if e.is_connect() {
            super::super::err(format!("stage {url} is not reachable: {e}"))
        } else if e.is_timeout() {
            super::super::err(format!(
                "stage {url} timed out after {}s",
                self.timeout.as_secs()
            ))
        } else {
            super::super::err(format!("stage {url}: {e}"))
        }
    }

    /// The error a non-2xx answer becomes: 409 is a mismatch of
    /// configuration, anything else a provider failure.
    fn status_err(&self, status: reqwest::StatusCode, body: &[u8]) -> CucaError {
        let message = serde_json::from_slice::<serde_json::Value>(body)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
            .unwrap_or_else(|| format!("HTTP {}", status.as_u16()));
        let text = format!("stage {}: {message}", self.url);
        if status == reqwest::StatusCode::CONFLICT {
            CucaError::Config(text)
        } else {
            super::super::err(text)
        }
    }
}

impl StageTransport for HttpTransport {
    fn call(&self, frame: Vec<u8>) -> Result<Vec<u8>, CucaError> {
        let sent = frame.len();
        let started = Instant::now();
        let outcome = self.rt.block_on(async {
            let mut request = self
                .client
                .post(&self.endpoint)
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .timeout(self.timeout)
                .body(frame);
            if let Some(key) = &self.key {
                request = request.bearer_auth(key);
            }
            let response = request.send().await.map_err(|e| self.transport_err(&e))?;
            let status = response.status();
            let compute = response
                .headers()
                .get(STAGE_COMPUTE_HEADER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(Duration::ZERO, Duration::from_nanos);
            let body = response.bytes().await.map_err(|e| self.transport_err(&e))?;
            if status.is_success() {
                Ok((body, compute))
            } else {
                Err(self.status_err(status, &body))
            }
        });
        match outcome {
            Ok((body, compute)) => {
                self.stats
                    .record(started.elapsed(), compute, sent, body.len());
                Ok(body.to_vec())
            }
            Err(e) => {
                self.stats.record_error(sent);
                Err(e)
            }
        }
    }

    fn name(&self) -> String {
        format!("stage {}", self.url)
    }

    fn counters(&self) -> Arc<StageCounters> {
        Arc::clone(&self.stats)
    }
}

/// `s` with every byte outside the URL path's unreserved set as `%XX`.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread::JoinHandle;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    /// What the responder saw of the request.
    struct Seen {
        request_line: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    impl Seen {
        fn header(&self, name: &str) -> Option<&str> {
            self.headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.as_str())
        }
    }

    /// What the responder does after reading the request.
    enum Reply {
        Answer(u16, Vec<u8>),
        /// A 200 whose headers report the nanoseconds the worker computed.
        Computed(Vec<u8>, u64),
        /// Never answers; holds the connection open for the given time.
        Hang(Duration),
    }

    /// Read one request off `sock`.
    async fn read_request(sock: &mut tokio::net::TcpStream) -> Seen {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        let head_end = loop {
            if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break at;
            }
            let n = sock.read(&mut chunk).await.unwrap();
            assert!(n > 0, "the client closed before its headers ended");
            buf.extend_from_slice(&chunk[..n]);
        };
        let head = String::from_utf8(buf[..head_end].to_vec()).unwrap();
        let mut lines = head.split("\r\n");
        let request_line = lines.next().unwrap().to_owned();
        let headers: Vec<(String, String)> = lines
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
            .collect();
        let length: usize = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .map(|(_, v)| v.parse().unwrap())
            .unwrap_or(0);
        let mut body = buf[head_end + 4..].to_vec();
        while body.len() < length {
            let n = sock.read(&mut chunk).await.unwrap();
            assert!(n > 0, "the client closed before its body ended");
            body.extend_from_slice(&chunk[..n]);
        }
        Seen {
            request_line,
            headers,
            body,
        }
    }

    /// A one-connection HTTP/1.1 responder on its own thread and runtime.
    fn serve(reply: Reply) -> (String, mpsc::Receiver<Seen>, JoinHandle<()>) {
        let std_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        std_listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", std_listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let listener = TcpListener::from_std(std_listener).unwrap();
                let (mut sock, _) = listener.accept().await.unwrap();
                let _ = tx.send(read_request(&mut sock).await);
                match reply {
                    Reply::Answer(status, payload) => {
                        let head = format!(
                            "HTTP/1.1 {status} X\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            payload.len()
                        );
                        sock.write_all(head.as_bytes()).await.unwrap();
                        sock.write_all(&payload).await.unwrap();
                        sock.shutdown().await.unwrap();
                    }
                    Reply::Computed(payload, ns) => {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\n{STAGE_COMPUTE_HEADER}: {ns}\r\nconnection: close\r\n\r\n",
                            payload.len()
                        );
                        sock.write_all(head.as_bytes()).await.unwrap();
                        sock.write_all(&payload).await.unwrap();
                        sock.shutdown().await.unwrap();
                    }
                    Reply::Hang(time) => tokio::time::sleep(time).await,
                }
            });
        });
        (url, rx, handle)
    }

    /// A responder that echoes each request's body as a keep-alive answer and
    /// then closes the connection, as a server does when its idle timeout
    /// fires between two passes. Serves up to `rounds` connections, giving up
    /// on one that does not come within 3 s; returns how many it accepted.
    fn serve_then_close(rounds: usize) -> (String, JoinHandle<usize>) {
        let std_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        std_listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", std_listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let listener = TcpListener::from_std(std_listener).unwrap();
                let mut accepted = 0;
                for _ in 0..rounds {
                    let next = tokio::time::timeout(Duration::from_secs(3), listener.accept());
                    let Ok(Ok((mut sock, _))) = next.await else {
                        break;
                    };
                    accepted += 1;
                    let seen = read_request(&mut sock).await;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-length: {}\r\n\r\n",
                        seen.body.len()
                    );
                    sock.write_all(head.as_bytes()).await.unwrap();
                    sock.write_all(&seen.body).await.unwrap();
                    sock.flush().await.unwrap();
                    // The idle timeout fires after the client has its answer
                    // and holds the connection as reusable.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                accepted
            })
        });
        (url, handle)
    }

    fn stage(url: &str) -> RemoteStage {
        RemoteStage {
            url: url.to_owned(),
            layers: 1,
            timeout: Duration::from_secs(10),
            ..RemoteStage::default()
        }
    }

    #[test]
    fn it_posts_the_frame_to_the_stage_path() {
        let frame = vec![2, 0, 255, 0, 0, 7];
        let (url, seen, handle) = serve(Reply::Answer(200, vec![0, 1, 0, 0, 9]));
        // A trailing slash on the URL does not double up.
        let t = HttpTransport::new(&stage(&format!("{url}/"))).unwrap();
        let answer = t.call(frame.clone()).unwrap();
        handle.join().unwrap();
        assert_eq!(answer, vec![0, 1, 0, 0, 9]);
        let seen = seen.recv().unwrap();
        assert_eq!(seen.request_line, "POST /cuca/v1/stage HTTP/1.1");
        assert_eq!(seen.body, frame);
        assert_eq!(
            seen.header("content-type"),
            Some("application/octet-stream")
        );
        assert_eq!(seen.header("authorization"), None);
    }

    #[test]
    fn calls_count_time_bytes_and_the_workers_compute() {
        let (url, _seen, handle) = serve(Reply::Computed(vec![9; 40], 1_500_000));
        let t = HttpTransport::new(&stage(&url)).unwrap();
        t.call(vec![1; 100]).unwrap();
        handle.join().unwrap();
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.errors), (1, 0));
        assert_eq!((s.sent_bytes, s.received_bytes), (100, 40));
        assert_eq!(s.compute_ns, 1_500_000);
        assert!(s.wall_ns > 0);
        assert_eq!(s.network_ns(), s.wall_ns.saturating_sub(1_500_000));

        // A worker that reports nothing counts no compute; a refused call
        // counts as an error and adds no time.
        let (url, _seen, handle) = serve(Reply::Answer(200, vec![1]));
        let t = HttpTransport::new(&stage(&url)).unwrap();
        t.call(vec![1]).unwrap();
        handle.join().unwrap();
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.compute_ns), (1, 0));
        let (url, _seen, handle) = serve(Reply::Answer(500, Vec::new()));
        let t = HttpTransport::new(&stage(&url)).unwrap();
        t.call(vec![1; 7]).unwrap_err();
        handle.join().unwrap();
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.errors, s.wall_ns, s.sent_bytes), (0, 1, 0, 7));
    }

    #[test]
    fn it_posts_to_the_model_path_with_a_bearer_key() {
        let (url, seen, handle) = serve(Reply::Answer(200, Vec::new()));
        let t = HttpTransport::new(&RemoteStage {
            model: Some("org/model b".to_owned()),
            api_key: Some("sk-secret".to_owned()),
            ..stage(&url)
        })
        .unwrap();
        assert!(t.call(vec![1]).unwrap().is_empty());
        handle.join().unwrap();
        let seen = seen.recv().unwrap();
        assert_eq!(
            seen.request_line,
            "POST /cuca/v1/stage/org%2Fmodel%20b HTTP/1.1"
        );
        assert_eq!(seen.header("authorization"), Some("Bearer sk-secret"));
    }

    #[test]
    fn a_plain_model_name_is_not_escaped() {
        assert_eq!(percent_encode("gemma-4_E2B.v1~x"), "gemma-4_E2B.v1~x");
        assert_eq!(percent_encode("a:b"), "a%3Ab");
        assert_eq!(percent_encode("é"), "%C3%A9");
    }

    #[test]
    fn a_409_is_a_configuration_error_with_the_servers_message() {
        let body = br#"{"error":{"message":"fingerprint differs","type":"x"}}"#.to_vec();
        let (url, _seen, handle) = serve(Reply::Answer(409, body));
        let t = HttpTransport::new(&stage(&url)).unwrap();
        let e = t.call(vec![1]).unwrap_err();
        handle.join().unwrap();
        match e {
            CucaError::Config(text) => {
                assert!(text.contains("fingerprint differs"), "{text}");
                assert!(text.contains(&url), "{text}");
            }
            other => panic!("expected a configuration error, got {other}"),
        }
    }

    #[test]
    fn other_failures_are_provider_errors() {
        let (url, _seen, handle) = serve(Reply::Answer(500, b"oops <html>".to_vec()));
        let t = HttpTransport::new(&stage(&url)).unwrap();
        let e = t.call(vec![1]).unwrap_err();
        handle.join().unwrap();
        assert!(matches!(e, CucaError::Provider { .. }), "{e}");
        assert!(e.to_string().contains("HTTP 500"), "{e}");

        let body = br#"{"error":{"message":"out of blocks"}}"#.to_vec();
        let (url, _seen, handle) = serve(Reply::Answer(503, body));
        let t = HttpTransport::new(&stage(&url)).unwrap();
        let e = t.call(vec![1]).unwrap_err();
        handle.join().unwrap();
        assert!(matches!(e, CucaError::Provider { .. }), "{e}");
        assert!(e.to_string().contains("out of blocks"), "{e}");
    }

    #[test]
    fn a_closed_port_is_not_reachable() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let t = HttpTransport::new(&stage(&format!("http://127.0.0.1:{port}"))).unwrap();
        let e = t.call(vec![1]).unwrap_err();
        assert!(matches!(e, CucaError::Provider { .. }), "{e}");
        assert!(e.to_string().contains("is not reachable"), "{e}");
    }

    #[test]
    fn a_silent_stage_times_out() {
        let (url, _seen, handle) = serve(Reply::Hang(Duration::from_secs(4)));
        let t = HttpTransport::new(&RemoteStage {
            timeout: Duration::from_secs(1),
            ..stage(&url)
        })
        .unwrap();
        let started = std::time::Instant::now();
        let e = t.call(vec![1]).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(4));
        assert!(matches!(e, CucaError::Provider { .. }), "{e}");
        assert!(e.to_string().contains("timed out after 1s"), "{e}");
        handle.join().unwrap();
    }

    /// Passes are seconds apart and a server may close an idle connection
    /// between two of them. The transport opens a connection per call, so
    /// the second call never meets the dead one a pool would hand it.
    #[test]
    fn calls_survive_a_server_that_closes_idle_connections() {
        let (url, handle) = serve_then_close(3);
        let t = HttpTransport::new(&stage(&url)).unwrap();
        for (i, frame) in [vec![1u8, 2], vec![3], vec![4, 5, 6]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(t.call(frame.clone()).unwrap(), frame, "call {i}");
            // Long enough for the close to reach the client's socket.
            std::thread::sleep(Duration::from_millis(300));
        }
        assert_eq!(handle.join().unwrap(), 3, "one connection per call");
    }
}
