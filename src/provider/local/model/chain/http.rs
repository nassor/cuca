//! The transport of a layer stage to the worker's model server: a frame
//! stream when the server takes the upgrade, one `POST` per frame otherwise.
//!
//! Over plain `http://` the transport first upgrades a connection to the
//! frame stream of [`stream`](super::stream), which costs a blocking write
//! and read per call. It falls back to `POST` for `https://` (the stream does
//! not speak TLS), for a server that does not know the upgrade (an older
//! cuca, a proxy that drops it) and, for that call, for a refusal such as a
//! busy worker.
//!
//! The `POST` path keeps a connection between calls. The engine thread
//! blocks on a current-thread runtime that drives the connections only while
//! a call runs, so a connection that sits in the pool between two passes can
//! be closed by the server (or a proxy in front of it) without the client
//! noticing. The next call then fails on it before any answer arrives. The
//! transport sends such a call again, once, on another connection, as the
//! stream does on a new one: that is safe because the worker runs a numbered
//! call once and answers a repeat with the reply it kept (see
//! [`stageframe`](crate::provider::local::stageframe)).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;

use super::stream::{Outcome, StreamLink};
use super::{StageCounters, StageTransport};
use crate::error::CucaError;
use crate::provider::local::{CarryDtype, RemoteStage, STAGE_COMPUTE_HEADER};

/// How long a connect may take before the stage counts as unreachable.
pub(super) const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

/// How long a connection may sit unused in the pool before the client drops
/// it: under the idle timeouts servers and proxies commonly close at (nginx
/// 75 s, cloud load balancers 60 s), so most connections are dropped here
/// before the other end closes them. A shorter one (a server at 5 s) is the
/// retry's case.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

/// A failed attempt: the error it becomes, and whether the connection was
/// to blame (the call may then go again on another one).
struct Failed {
    error: CucaError,
    stale: bool,
}

/// Whether `e`, from sending a request, says the connection failed before
/// any answer arrived: a request error that is not a connect error (the
/// stage is unreachable: a second attempt would only wait again) and not a
/// timeout (the stage is slow, or gone: a second attempt would not help).
/// The answer's head had not arrived, so a retried call never retries an
/// answer that was under way.
fn lost_connection(e: &reqwest::Error) -> bool {
    e.is_request() && !e.is_connect() && !e.is_timeout()
}

/// Sends frames to a worker's model server and blocks the calling thread on
/// the answer: over the frame stream at `{url}/cuca/v1/stage-stream[/{model}]`
/// for a plain `http://` worker that takes the upgrade, else by `POST` to
/// `{url}/cuca/v1/stage[/{model}]`. The `POST` path owns a single-threaded
/// runtime, so the transport must be created and called from a plain thread
/// (the engine thread), never from inside another runtime.
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
    /// The frame stream, for a plain `http://` worker.
    stream: Option<StreamLink>,
    /// The worker has no stream: every call goes by `POST`.
    post_only: AtomicBool,
    /// The number type of the carries, as configured.
    carry: CarryDtype,
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
            // reqwest 0.13 already defaults to it; a frame is one small
            // write each way and must not wait for an ACK (Nagle).
            .tcp_nodelay(true)
            // One call at a time: one connection is all the pool needs. A
            // call that finds it closed is sent again (see the module docs).
            .pool_max_idle_per_host(1)
            .pool_idle_timeout(POOL_IDLE_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| super::super::err(format!("stage {url}: cannot build client: {e}")))?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| super::super::err(format!("stage {url}: cannot build runtime: {e}")))?;
        let model_path = stage
            .model
            .as_ref()
            .map_or_else(String::new, |m| format!("/{}", percent_encode(m)));
        let stream = StreamLink::new(&url, &model_path, stage.api_key.as_deref(), stage.timeout);
        Ok(Self {
            url,
            endpoint,
            key: stage.api_key.clone(),
            timeout: stage.timeout,
            client,
            rt,
            stats: Arc::default(),
            stream,
            post_only: AtomicBool::new(false),
            carry: stage.carry,
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

    /// One attempt: post `frame` and read the whole answer, with the compute
    /// time the worker reported.
    async fn attempt(&self, frame: &Bytes) -> Result<(Bytes, Duration), Failed> {
        let mut request = self
            .client
            .post(&self.endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
            .timeout(self.timeout)
            .body(frame.clone());
        if let Some(key) = &self.key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(|e| Failed {
            stale: lost_connection(&e),
            error: self.transport_err(&e),
        })?;
        let status = response.status();
        let compute = response
            .headers()
            .get(STAGE_COMPUTE_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(Duration::ZERO, Duration::from_nanos);
        let body = response.bytes().await.map_err(|e| Failed {
            stale: false,
            error: self.transport_err(&e),
        })?;
        if status.is_success() {
            Ok((body, compute))
        } else {
            Err(Failed {
                stale: false,
                error: self.status_err(status, &body),
            })
        }
    }
}

impl StageTransport for HttpTransport {
    fn call(&self, frame: Vec<u8>) -> Result<Vec<u8>, CucaError> {
        if let Some(link) = &self.stream
            && !self.post_only.load(Ordering::Relaxed)
        {
            let started = Instant::now();
            match link.call(&frame, &self.stats) {
                Ok(Outcome::Answer { body, compute }) => {
                    self.stats
                        .record(started.elapsed(), compute, frame.len(), body.len());
                    return Ok(body);
                }
                Ok(Outcome::Fallback { permanent }) => {
                    if permanent {
                        self.post_only.store(true, Ordering::Relaxed);
                    }
                }
                Err(e) => {
                    self.stats.record_error(frame.len());
                    return Err(e);
                }
            }
        }
        let sent = frame.len();
        let frame = Bytes::from(frame);
        let started = Instant::now();
        let outcome = self.rt.block_on(async {
            match self.attempt(&frame).await {
                Err(Failed { stale: true, .. }) => {
                    self.stats.record_retry(sent);
                    self.attempt(&frame).await
                }
                other => other,
            }
        });
        match outcome {
            Ok((body, compute)) => {
                self.stats
                    .record(started.elapsed(), compute, sent, body.len());
                Ok(body.to_vec())
            }
            Err(failed) => {
                self.stats.record_error(sent);
                Err(failed.error)
            }
        }
    }

    fn name(&self) -> String {
        format!("stage {}", self.url)
    }

    fn counters(&self) -> Arc<StageCounters> {
        Arc::clone(&self.stats)
    }

    fn carry(&self) -> CarryDtype {
        self.carry
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

    /// A responder that reads a request and closes the connection without
    /// answering, `drops` times (what a server does that processed a frame
    /// and died, or a pooled connection it had closed), then answers the next
    /// request with its body. Returns the body of every request it read; it
    /// stops after the answer, or when no connection comes within 1 s.
    fn serve_dropping(drops: usize) -> (String, JoinHandle<Vec<Vec<u8>>>) {
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
                let mut bodies = Vec::new();
                loop {
                    let next = tokio::time::timeout(Duration::from_secs(1), listener.accept());
                    let Ok(Ok((mut sock, _))) = next.await else {
                        break;
                    };
                    let seen = read_request(&mut sock).await;
                    bodies.push(seen.body.clone());
                    if bodies.len() <= drops {
                        continue;
                    }
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        seen.body.len()
                    );
                    sock.write_all(head.as_bytes()).await.unwrap();
                    sock.write_all(&seen.body).await.unwrap();
                    sock.shutdown().await.unwrap();
                    break;
                }
                bodies
            })
        });
        (url, handle)
    }

    /// A responder that echoes `requests` requests on the one connection it
    /// accepts; returns how many connections came (one, if the client keeps
    /// its connection between calls).
    fn serve_keepalive(requests: usize) -> (String, JoinHandle<usize>) {
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
                let next = tokio::time::timeout(Duration::from_secs(3), listener.accept());
                let Ok(Ok((mut sock, _))) = next.await else {
                    return 0;
                };
                let mut accepted = 1;
                for _ in 0..requests {
                    let read =
                        tokio::time::timeout(Duration::from_secs(3), read_request(&mut sock));
                    let Ok(seen) = read.await else {
                        break;
                    };
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-length: {}\r\n\r\n",
                        seen.body.len()
                    );
                    sock.write_all(head.as_bytes()).await.unwrap();
                    sock.write_all(&seen.body).await.unwrap();
                    sock.flush().await.unwrap();
                }
                let more = tokio::time::timeout(Duration::from_millis(200), listener.accept());
                if matches!(more.await, Ok(Ok(_))) {
                    accepted += 1;
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

    /// A transport that posts every frame: the responders below speak HTTP
    /// only, and the frame stream has its own tests.
    fn post_only(stage: &RemoteStage) -> HttpTransport {
        let t = HttpTransport::new(stage).unwrap();
        t.post_only.store(true, Ordering::Relaxed);
        t
    }

    #[test]
    fn it_posts_the_frame_to_the_stage_path() {
        let frame = vec![2, 0, 255, 0, 0, 7];
        let (url, seen, handle) = serve(Reply::Answer(200, vec![0, 1, 0, 0, 9]));
        // A trailing slash on the URL does not double up.
        let t = post_only(&stage(&format!("{url}/")));
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
        let t = post_only(&stage(&url));
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
        let t = post_only(&stage(&url));
        t.call(vec![1]).unwrap();
        handle.join().unwrap();
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.compute_ns), (1, 0));
        let (url, _seen, handle) = serve(Reply::Answer(500, Vec::new()));
        let t = post_only(&stage(&url));
        t.call(vec![1; 7]).unwrap_err();
        handle.join().unwrap();
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.errors, s.wall_ns, s.sent_bytes), (0, 1, 0, 7));
    }

    #[test]
    fn it_posts_to_the_model_path_with_a_bearer_key() {
        let (url, seen, handle) = serve(Reply::Answer(200, Vec::new()));
        let t = post_only(&RemoteStage {
            model: Some("org/model b".to_owned()),
            api_key: Some("sk-secret".to_owned()),
            ..stage(&url)
        });
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
        let t = post_only(&stage(&url));
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
        let t = post_only(&stage(&url));
        let e = t.call(vec![1]).unwrap_err();
        handle.join().unwrap();
        assert!(matches!(e, CucaError::Provider { .. }), "{e}");
        assert!(e.to_string().contains("HTTP 500"), "{e}");

        let body = br#"{"error":{"message":"out of blocks"}}"#.to_vec();
        let (url, _seen, handle) = serve(Reply::Answer(503, body));
        let t = post_only(&stage(&url));
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
        assert_eq!(t.counters().snapshot(t.name()).retries, 0, "no retry");
        assert!(e.to_string().contains("is not reachable"), "{e}");
    }

    #[test]
    fn a_silent_stage_times_out() {
        let (url, _seen, handle) = serve(Reply::Hang(Duration::from_secs(4)));
        let t = post_only(&RemoteStage {
            timeout: Duration::from_secs(1),
            ..stage(&url)
        });
        let started = std::time::Instant::now();
        let e = t.call(vec![1]).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(4));
        assert!(matches!(e, CucaError::Provider { .. }), "{e}");
        assert!(e.to_string().contains("timed out after 1s"), "{e}");
        assert_eq!(t.counters().snapshot(t.name()).retries, 0, "no retry");
        handle.join().unwrap();
    }

    /// Passes are seconds apart and a server may close an idle connection
    /// between two of them, which the client does not see until it sends on
    /// it. The calls still succeed: on a new connection, whether the client
    /// finds the old one dead before it writes or only after.
    #[test]
    fn calls_survive_a_server_that_closes_idle_connections() {
        let (url, handle) = serve_then_close(3);
        let t = post_only(&stage(&url));
        for (i, frame) in [vec![1u8, 2], vec![3], vec![4, 5, 6]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(t.call(frame.clone()).unwrap(), frame, "call {i}");
            // Long enough for the close to reach the client's socket.
            std::thread::sleep(Duration::from_millis(300));
        }
        assert_eq!(handle.join().unwrap(), 3, "each connection served one call");
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.errors), (3, 0));
        assert!(s.retries <= 2, "at most one retry per call: {}", s.retries);
    }

    /// The connection is kept between calls: three calls, one connection.
    #[test]
    fn calls_reuse_their_connection() {
        let (url, handle) = serve_keepalive(3);
        let t = post_only(&stage(&url));
        for frame in [vec![1u8, 2], vec![3], vec![4, 5, 6]] {
            assert_eq!(t.call(frame.clone()).unwrap(), frame);
        }
        assert_eq!(handle.join().unwrap(), 1, "one connection for three calls");
        assert_eq!(t.counters().snapshot(t.name()).retries, 0);
    }

    /// A connection that dies before any answer arrives (a server that
    /// processed the frame and went down before it answered, or one that
    /// closed the pooled connection) gets the same frame sent again, once.
    #[test]
    fn a_connection_that_dies_before_the_answer_gets_the_frame_again() {
        let frame = vec![7u8, 8, 9, 10];
        let (url, handle) = serve_dropping(1);
        let t = post_only(&stage(&url));
        assert_eq!(t.call(frame.clone()).unwrap(), frame);
        assert_eq!(handle.join().unwrap(), vec![frame.clone(), frame]);
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.errors, s.retries), (1, 0, 1));
        assert_eq!(s.sent_bytes, 8, "both attempts count");
        assert_eq!(s.received_bytes, 4);
    }

    /// The retry is one: a second failure is the call's error.
    #[test]
    fn a_second_failure_is_the_calls_error() {
        let (url, handle) = serve_dropping(2);
        let t = post_only(&stage(&url));
        let e = t.call(vec![1, 2]).unwrap_err();
        assert!(matches!(e, CucaError::Provider { .. }), "{e}");
        assert_eq!(handle.join().unwrap().len(), 2, "two attempts, no third");
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.errors, s.retries), (0, 1, 1));
    }

    /// An answer, even an error status, is final: it is no lost connection.
    #[test]
    fn an_answered_call_is_never_sent_again() {
        let (url, seen, handle) = serve(Reply::Answer(500, Vec::new()));
        let t = post_only(&stage(&url));
        t.call(vec![1]).unwrap_err();
        handle.join().unwrap();
        drop(seen);
        assert_eq!(t.counters().snapshot(t.name()).retries, 0);
    }

    // ---- the frame stream ----

    use crate::provider::local::stagestream::{
        MAX_STAGE_FRAME_BYTES, REPLY_FAILED, REPLY_MISMATCH, REPLY_OK, write_framed,
    };

    /// What the stream responder does with one accepted connection.
    enum Conn {
        /// Answer the upgrade with `101`, then do the steps, one per frame.
        Stream(Vec<Step>),
        /// Answer the upgrade with this status and close.
        Refuse(u16),
        /// Answer one `POST` with its body.
        Post,
    }

    enum Step {
        /// Answer with the frame itself, reporting this compute time.
        Echo(u64),
        /// Answer with this status and message.
        Status(u8, &'static str),
        /// Read the frame and close the connection.
        Drop,
        /// Read the frame and stay silent for this long.
        Hang(Duration),
        /// Write these bytes as the answer.
        Raw(Vec<u8>),
    }

    #[derive(Default)]
    struct Heard {
        /// The head of every request, in order.
        heads: Vec<String>,
        /// The frame of every stream request.
        frames: Vec<Vec<u8>>,
    }

    fn read_head(sock: &mut std::net::TcpStream) -> String {
        use std::io::Read;
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            sock.read_exact(&mut byte).unwrap();
            head.push(byte[0]);
        }
        String::from_utf8(head).unwrap()
    }

    fn reply(status: u8, compute_ns: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![status];
        out.extend_from_slice(&compute_ns.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// A responder that accepts one connection per entry of `script`, in
    /// order, and returns what it heard once the script is done.
    fn serve_script(script: Vec<Conn>) -> (String, JoinHandle<Heard>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut heard = Heard::default();
            for conn in script {
                let (mut sock, _) = listener.accept().unwrap();
                sock.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let head = read_head(&mut sock);
                heard.heads.push(head.clone());
                match conn {
                    Conn::Refuse(status) => {
                        let body = b"{}";
                        let text = format!(
                            "HTTP/1.1 {status} X\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            body.len()
                        );
                        sock.write_all(text.as_bytes()).unwrap();
                        sock.write_all(body).unwrap();
                    }
                    Conn::Post => {
                        let length: usize = head
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(str::to_owned)
                            })
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                        let mut body = vec![0u8; length];
                        sock.read_exact(&mut body).unwrap();
                        let text = format!(
                            "HTTP/1.1 200 OK\r\ncontent-length: {length}\r\nconnection: close\r\n\r\n"
                        );
                        sock.write_all(text.as_bytes()).unwrap();
                        sock.write_all(&body).unwrap();
                    }
                    Conn::Stream(steps) => {
                        sock.write_all(
                            b"HTTP/1.1 101 Switching Protocols\r\nconnection: upgrade\r\nupgrade: cuca-stage/1\r\n\r\n",
                        )
                        .unwrap();
                        for step in steps {
                            let mut len = [0u8; 8];
                            sock.read_exact(&mut len).unwrap();
                            let mut frame = vec![0u8; u64::from_le_bytes(len) as usize];
                            sock.read_exact(&mut frame).unwrap();
                            heard.frames.push(frame.clone());
                            match step {
                                Step::Echo(ns) => {
                                    sock.write_all(&reply(REPLY_OK, ns, &frame)).unwrap();
                                }
                                Step::Status(status, message) => {
                                    sock.write_all(&reply(status, 0, message.as_bytes()))
                                        .unwrap();
                                }
                                Step::Drop => break,
                                Step::Hang(time) => std::thread::sleep(time),
                                Step::Raw(bytes) => sock.write_all(&bytes).unwrap(),
                            }
                        }
                    }
                }
            }
            heard
        });
        (url, handle)
    }

    #[test]
    fn calls_use_one_upgraded_stream() {
        let (url, handle) = serve_script(vec![Conn::Stream(vec![
            Step::Echo(1000),
            Step::Echo(1000),
            Step::Echo(1000),
        ])]);
        let t = HttpTransport::new(&RemoteStage {
            model: Some("m b".to_owned()),
            api_key: Some("sk".to_owned()),
            ..stage(&url)
        })
        .unwrap();
        for frame in [vec![1u8, 2], vec![3], vec![4, 5, 6]] {
            assert_eq!(t.call(frame.clone()).unwrap(), frame);
        }
        let heard = handle.join().unwrap();
        assert_eq!(heard.heads.len(), 1, "one connection for three calls");
        let head = &heard.heads[0];
        assert!(
            head.starts_with("GET /cuca/v1/stage-stream/m%20b HTTP/1.1\r\n"),
            "{head}"
        );
        assert!(head.contains("Connection: Upgrade\r\n"), "{head}");
        assert!(head.contains("Upgrade: cuca-stage/1\r\n"), "{head}");
        assert!(head.contains("Authorization: Bearer sk\r\n"), "{head}");
        assert_eq!(heard.frames.len(), 3);
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.errors, s.retries), (3, 0, 0));
        assert_eq!(s.compute_ns, 3000);
        // Frame bytes, as over `POST`: the length prefix and the reply head
        // are not counted.
        assert_eq!((s.sent_bytes, s.received_bytes), (6, 6));
    }

    /// A stream that dies before any byte of the answer is replaced by a new
    /// upgraded connection, which gets the same frame.
    #[test]
    fn a_stream_that_drops_before_the_answer_is_replaced_and_the_frame_sent_again() {
        let frame = vec![7u8, 8, 9, 10];
        let (url, handle) = serve_script(vec![
            Conn::Stream(vec![Step::Drop]),
            Conn::Stream(vec![Step::Echo(5)]),
        ]);
        let t = HttpTransport::new(&stage(&url)).unwrap();
        assert_eq!(t.call(frame.clone()).unwrap(), frame);
        let heard = handle.join().unwrap();
        assert_eq!(heard.frames, vec![frame.clone(), frame]);
        assert_eq!(heard.heads.len(), 2);
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.errors, s.retries), (1, 0, 1));
    }

    #[test]
    fn a_second_dropped_stream_is_the_calls_error() {
        let (url, handle) = serve_script(vec![
            Conn::Stream(vec![Step::Drop]),
            Conn::Stream(vec![Step::Drop]),
        ]);
        let t = HttpTransport::new(&stage(&url)).unwrap();
        let e = t.call(vec![1, 2]).unwrap_err();
        assert!(matches!(e, CucaError::Provider { .. }), "{e}");
        assert_eq!(
            handle.join().unwrap().frames.len(),
            2,
            "two attempts, no third"
        );
        let s = t.counters().snapshot(t.name());
        assert_eq!((s.calls, s.errors, s.retries), (0, 1, 1));
    }

    /// A refusal the worker answers is no lost connection: the stream stays,
    /// and a mismatch is a configuration error as it is over `POST`.
    #[test]
    fn stream_failures_keep_the_connection_and_map_to_errors() {
        let (url, handle) = serve_script(vec![Conn::Stream(vec![
            Step::Status(REPLY_MISMATCH, "fingerprint differs"),
            Step::Status(REPLY_FAILED, "out of blocks"),
            Step::Echo(0),
        ])]);
        let t = HttpTransport::new(&stage(&url)).unwrap();
        match t.call(vec![1]).unwrap_err() {
            CucaError::Config(text) => {
                assert!(text.contains("fingerprint differs"), "{text}");
                assert!(text.contains(&url), "{text}");
            }
            other => panic!("expected a configuration error, got {other}"),
        }
        let e = t.call(vec![2]).unwrap_err();
        assert!(matches!(e, CucaError::Provider { .. }), "{e}");
        assert!(e.to_string().contains("out of blocks"), "{e}");
        assert_eq!(t.call(vec![3]).unwrap(), vec![3]);
        assert_eq!(handle.join().unwrap().heads.len(), 1, "one connection");
        assert_eq!(t.counters().snapshot(t.name()).retries, 0);
    }

    /// A server that does not know the upgrade (an older cuca) is posted to,
    /// and not asked again.
    #[test]
    fn a_server_without_the_stream_is_posted_to_from_then_on() {
        let (url, handle) = serve_script(vec![Conn::Refuse(404), Conn::Post, Conn::Post]);
        let t = HttpTransport::new(&stage(&url)).unwrap();
        assert_eq!(t.call(vec![1, 2]).unwrap(), vec![1, 2]);
        assert_eq!(t.call(vec![3]).unwrap(), vec![3]);
        let heard = handle.join().unwrap();
        let methods: Vec<&str> = heard
            .heads
            .iter()
            .map(|h| h.split(' ').next().unwrap_or(""))
            .collect();
        assert_eq!(methods, ["GET", "POST", "POST"]);
    }

    /// A refusal that may pass (a busy worker) costs that call a `POST`, and
    /// the next call tries the stream again.
    #[test]
    fn a_busy_worker_is_posted_to_for_that_call_only() {
        let (url, handle) = serve_script(vec![
            Conn::Refuse(503),
            Conn::Post,
            Conn::Stream(vec![Step::Echo(0)]),
        ]);
        let t = HttpTransport::new(&stage(&url)).unwrap();
        assert_eq!(t.call(vec![1]).unwrap(), vec![1]);
        assert_eq!(t.call(vec![2]).unwrap(), vec![2]);
        let heard = handle.join().unwrap();
        assert_eq!(
            heard.frames,
            vec![vec![2u8]],
            "the second call went by stream"
        );
    }

    #[test]
    fn a_silent_stream_times_out_and_is_not_sent_again() {
        let (url, handle) =
            serve_script(vec![Conn::Stream(vec![Step::Hang(Duration::from_secs(3))])]);
        let t = HttpTransport::new(&RemoteStage {
            timeout: Duration::from_secs(1),
            ..stage(&url)
        })
        .unwrap();
        let started = std::time::Instant::now();
        let e = t.call(vec![1]).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(3));
        assert!(e.to_string().contains("timed out after 1s"), "{e}");
        assert_eq!(t.counters().snapshot(t.name()).retries, 0);
        assert_eq!(handle.join().unwrap().frames.len(), 1);
    }

    #[test]
    fn an_answer_above_the_frame_limit_is_refused_unread() {
        let big = reply(REPLY_OK, 0, &[]);
        let mut raw = big;
        let at = raw.len() - 8;
        raw[at..].copy_from_slice(&((MAX_STAGE_FRAME_BYTES as u64) + 1).to_le_bytes());
        let (url, handle) = serve_script(vec![Conn::Stream(vec![Step::Raw(raw)])]);
        let t = HttpTransport::new(&stage(&url)).unwrap();
        let e = t.call(vec![1]).unwrap_err();
        assert!(e.to_string().contains("exceeds the limit"), "{e}");
        handle.join().unwrap();
    }

    #[test]
    fn only_plain_http_gets_a_stream() {
        assert!(
            HttpTransport::new(&stage("http://127.0.0.1:1"))
                .unwrap()
                .stream
                .is_some()
        );
        assert!(
            HttpTransport::new(&stage("https://127.0.0.1:1"))
                .unwrap()
                .stream
                .is_none()
        );
        // Framing helper: a small frame is one write.
        let mut out = Vec::new();
        write_framed(&mut out, &[1, 2], &[3]).unwrap();
        assert_eq!(out, [1, 2, 3]);
    }
}
