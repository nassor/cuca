//! The coordinator's side of the frame stream of a layer stage (see
//! [`stagestream`](crate::provider::local::stagestream) for the framing).
//!
//! The link upgrades one connection of the worker's server and keeps it:
//! every call is a blocking write and a blocking read on the calling (engine)
//! thread, with no runtime in between. A connection that fails before the
//! first byte of an answer is replaced and the frame sent again, once; the
//! worker runs a numbered call once and answers a repeat from memory, so this
//! is safe (see [`stageframe`](crate::provider::local::stageframe)). A worker
//! that does not take the upgrade is left to the `POST` transport.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use super::StageCounters;
use super::http::CONNECT_TIMEOUT;
use crate::error::CucaError;
use crate::provider::local::stagestream::{
    MAX_STAGE_FRAME_BYTES, REPLY_HEAD, REPLY_MISMATCH, REPLY_OK, STAGE_STREAM_UPGRADE, write_framed,
};

/// The head of the worker's answer to the upgrade is at most this long.
const MAX_HEAD: usize = 16 * 1024;

/// How one call went.
pub(super) enum Outcome {
    /// The worker answered: its reply frame and the nanoseconds its engine
    /// thread took.
    Answer { body: Vec<u8>, compute: Duration },
    /// The worker does not serve the stream (now, or at all when
    /// `permanent`): the call goes by `POST`.
    Fallback { permanent: bool },
}

/// One stream to one worker.
pub(super) struct StreamLink {
    /// What messages call the stage: `stage {url}`.
    name: String,
    host: String,
    port: u16,
    /// The `Host` header.
    authority: String,
    /// The request target of the upgrade.
    target: String,
    key: Option<String>,
    timeout: Duration,
    sock: Mutex<Option<TcpStream>>,
}

/// What reading an answer found.
enum Exchange {
    Answer {
        status: u8,
        compute: Duration,
        body: Vec<u8>,
    },
    /// The connection failed before a byte of the answer arrived.
    Lost,
    Failed(CucaError),
}

impl StreamLink {
    /// A link to `url` (the stage's name in messages), for a plain `http://`
    /// URL; `None` for any other scheme, which only the `POST` transport
    /// serves (it speaks TLS).
    pub(super) fn new(
        url: &str,
        model_path: &str,
        key: Option<&str>,
        timeout: Duration,
    ) -> Option<Self> {
        let parsed = reqwest::Url::parse(url).ok()?;
        if parsed.scheme() != "http" {
            return None;
        }
        let host = parsed.host_str()?.trim_matches(['[', ']']).to_owned();
        let port = parsed.port_or_known_default()?;
        let prefix = parsed.path().trim_end_matches('/');
        Some(Self {
            name: format!("stage {url}"),
            authority: format!("{}:{port}", parsed.host_str()?),
            target: format!("{prefix}/cuca/v1/stage-stream{model_path}"),
            host,
            port,
            key: key.map(str::to_owned),
            timeout,
            sock: Mutex::new(None),
        })
    }

    fn not_reachable(&self, e: &std::io::Error) -> CucaError {
        super::super::err(format!("{} is not reachable: {e}", self.name))
    }

    /// Connect and upgrade.
    fn connect(&self) -> Result<Result<TcpStream, bool>, CucaError> {
        let addrs = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|e| self.not_reachable(&e))?;
        let mut last = None;
        let mut found = None;
        for addr in addrs {
            match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
                Ok(s) => {
                    found = Some(s);
                    break;
                }
                Err(e) => last = Some(e),
            }
        }
        let Some(mut sock) = found else {
            let e = last.unwrap_or_else(|| std::io::Error::other("no address"));
            return Err(self.not_reachable(&e));
        };
        let io = |e: std::io::Error| super::super::err(format!("{}: {e}", self.name));
        sock.set_nodelay(true).map_err(io)?;
        sock.set_write_timeout(Some(self.timeout)).map_err(io)?;
        sock.set_read_timeout(Some(CONNECT_TIMEOUT)).map_err(io)?;
        let mut request = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: Upgrade\r\nUpgrade: {STAGE_STREAM_UPGRADE}\r\n",
            self.target, self.authority
        );
        if let Some(key) = &self.key {
            request.push_str(&format!("Authorization: Bearer {key}\r\n"));
        }
        request.push_str("\r\n");
        sock.write_all(request.as_bytes()).map_err(io)?;

        let mut head = Vec::new();
        let mut chunk = [0u8; 512];
        while !head.windows(4).any(|w| w == b"\r\n\r\n") {
            if head.len() > MAX_HEAD {
                return Ok(Err(true));
            }
            match sock.read(&mut chunk) {
                Ok(0) => return Ok(Err(false)),
                Ok(n) => head.extend_from_slice(&chunk[..n]),
                Err(e) => return Err(io(e)),
            }
        }
        let text = String::from_utf8_lossy(&head).to_ascii_lowercase();
        let status = text.split_whitespace().nth(1).unwrap_or("");
        if status == "101" && text.contains(&format!("upgrade: {STAGE_STREAM_UPGRADE}")) {
            sock.set_read_timeout(Some(self.timeout)).map_err(io)?;
            return Ok(Ok(sock));
        }
        // A server that has no such route or does not know the upgrade will
        // not have one later; any other refusal (busy, loading, a key the
        // `POST` route reports better) is for this call only.
        Ok(Err(matches!(
            status,
            "400" | "404" | "405" | "426" | "501" | ""
        )))
    }

    /// Send `frame` and read the answer on `sock`.
    fn exchange(&self, sock: &mut TcpStream, frame: &[u8]) -> Exchange {
        let failed = |e: &std::io::Error| match e.kind() {
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => {
                Exchange::Failed(super::super::err(format!(
                    "{} timed out after {}s",
                    self.name,
                    self.timeout.as_secs()
                )))
            }
            _ => Exchange::Failed(super::super::err(format!("{}: {e}", self.name))),
        };
        let lost = |e: &std::io::Error| {
            matches!(
                e.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::UnexpectedEof
                    | std::io::ErrorKind::NotConnected
            )
        };
        if let Err(e) = write_framed(&mut *sock, &(frame.len() as u64).to_le_bytes(), frame) {
            return if lost(&e) { Exchange::Lost } else { failed(&e) };
        }
        let mut head = [0u8; REPLY_HEAD];
        match sock.read(&mut head[..1]) {
            Ok(0) => return Exchange::Lost,
            Ok(_) => {}
            Err(e) => return if lost(&e) { Exchange::Lost } else { failed(&e) },
        }
        if let Err(e) = sock.read_exact(&mut head[1..]) {
            return failed(&e);
        }
        let status = head[0];
        let compute =
            Duration::from_nanos(u64::from_le_bytes(head[1..9].try_into().unwrap_or([0; 8])));
        let len = u64::from_le_bytes(head[9..].try_into().unwrap_or([0; 8]));
        let Some(len) = usize::try_from(len)
            .ok()
            .filter(|&n| n <= MAX_STAGE_FRAME_BYTES)
        else {
            return Exchange::Failed(super::super::err(format!(
                "{}: an answer of {len} bytes exceeds the limit",
                self.name
            )));
        };
        let mut body = vec![0u8; len];
        if let Err(e) = sock.read_exact(&mut body) {
            return failed(&e);
        }
        Exchange::Answer {
            status,
            compute,
            body,
        }
    }

    /// One call: `frame` out, the answer back, on the stream (opened first
    /// if there is none).
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] for a worker that answers a mismatch,
    /// [`CucaError::Provider`] for any other failure, an unreachable worker
    /// included.
    pub(super) fn call(&self, frame: &[u8], stats: &StageCounters) -> Result<Outcome, CucaError> {
        let mut retried = false;
        loop {
            let mut guard = self.sock.lock().unwrap_or_else(PoisonError::into_inner);
            if guard.is_none() {
                match self.connect()? {
                    Ok(sock) => *guard = Some(sock),
                    Err(permanent) => return Ok(Outcome::Fallback { permanent }),
                }
            }
            let Some(sock) = guard.as_mut() else {
                return Err(super::super::err(format!("{}: no connection", self.name)));
            };
            match self.exchange(sock, frame) {
                Exchange::Answer {
                    status,
                    compute,
                    body,
                } => {
                    return match status {
                        REPLY_OK => Ok(Outcome::Answer { body, compute }),
                        other => {
                            let text = format!("{}: {}", self.name, String::from_utf8_lossy(&body));
                            Err(if other == REPLY_MISMATCH {
                                CucaError::Config(text)
                            } else {
                                super::super::err(text)
                            })
                        }
                    };
                }
                Exchange::Lost => {
                    *guard = None;
                    if retried {
                        return Err(super::super::err(format!(
                            "{}: the connection closed before an answer arrived",
                            self.name
                        )));
                    }
                    retried = true;
                    stats.record_retry(frame.len());
                }
                Exchange::Failed(e) => {
                    *guard = None;
                    return Err(e);
                }
            }
        }
    }
}
