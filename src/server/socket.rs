//! The model server's listener: every accepted connection has `TCP_NODELAY`,
//! and a layer-stage frame stream can take its socket over after its HTTP
//! upgrade.
//!
//! axum hands a connection to hyper, and after an upgrade the socket is
//! reachable only as hyper's wrapped, type-erased stream (hyper-util keeps
//! the wrapper private, so it cannot be downcast). The stream's blocking
//! threads need the socket itself. The listener's IO type ([`ServedIo`])
//! therefore carries a [`RawSocket`] claim that the request sees as
//! connection info: the upgrade handler marks it wanted before it answers
//! `101`, and when hyper lets go of the upgraded stream, dropping the
//! `ServedIo`, the tcp stream moves into the claim instead of closing. A
//! connection nobody claims pays nothing but the `Arc`.

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Router;
use axum::extract::connect_info::{Connected, IntoMakeServiceWithConnectInfo};
use axum::serve::{IncomingStream, Listener};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};

/// The socket of a connection, for an upgrade to claim.
#[derive(Default)]
struct Claim {
    /// Set before the `101`: keep the stream when the connection's IO drops.
    wanted: AtomicBool,
    /// The stream, once the IO has dropped.
    kept: Mutex<Option<TcpStream>>,
}

/// The connection an HTTP request arrived on, for an upgrade to take over.
#[derive(Clone)]
pub(crate) struct RawSocket(Arc<Claim>);

impl RawSocket {
    /// Keep the socket when hyper lets the connection go: call it before the
    /// response that upgrades the connection is returned.
    pub(crate) fn want(&self) {
        self.0.wanted.store(true, Ordering::Release);
    }

    /// The socket, blocking, for a thread of its own; call it once hyper has
    /// given the upgraded connection up (its `OnUpgrade` resolved and the
    /// `Upgraded` dropped), from inside the runtime that serves it.
    ///
    /// # Errors
    ///
    /// [`std::io::ErrorKind::NotFound`] when the connection was not kept,
    /// else the error of taking the stream out of the runtime.
    pub(crate) fn take(&self) -> std::io::Result<std::net::TcpStream> {
        let stream = self
            .0
            .kept
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "the connection was not kept")
            })?;
        let stream = stream.into_std()?;
        stream.set_nonblocking(false)?;
        Ok(stream)
    }
}

/// An accepted connection: the stream hyper serves and its claim.
pub(crate) struct ServedIo {
    stream: Option<TcpStream>,
    claim: Arc<Claim>,
}

impl ServedIo {
    fn new(stream: TcpStream) -> Self {
        // axum leaves Nagle's algorithm on, which holds back the small
        // trailing segment of a response (one SSE event) until the peer
        // acknowledges the one before it.
        if let Err(e) = stream.set_nodelay(true) {
            tracing::debug!("cannot set TCP_NODELAY on an accepted connection: {e}");
        }
        Self {
            stream: Some(stream),
            claim: Arc::default(),
        }
    }

    fn live(&mut self) -> std::io::Result<Pin<&mut TcpStream>> {
        self.stream
            .as_mut()
            .map(Pin::new)
            .ok_or_else(|| std::io::Error::other("the connection was handed over"))
    }
}

impl Drop for ServedIo {
    fn drop(&mut self) {
        if self.claim.wanted.load(Ordering::Acquire)
            && let Some(stream) = self.stream.take()
        {
            *self
                .claim
                .kept
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(stream);
        }
    }
}

impl AsyncRead for ServedIo {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut().live() {
            Ok(stream) => stream.poll_read(cx, buf),
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

impl AsyncWrite for ServedIo {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut().live() {
            Ok(stream) => stream.poll_write(cx, buf),
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut().live() {
            Ok(stream) => stream.poll_flush(cx),
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut().live() {
            Ok(stream) => stream.poll_shutdown(cx),
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut().live() {
            Ok(stream) => stream.poll_write_vectored(cx, bufs),
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    fn is_write_vectored(&self) -> bool {
        self.stream
            .as_ref()
            .is_some_and(TcpStream::is_write_vectored)
    }
}

/// The listener of the model server.
pub(crate) struct ServedListener(TcpListener);

impl Listener for ServedListener {
    type Io = ServedIo;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (ServedIo, SocketAddr) {
        loop {
            match self.0.accept().await {
                Ok((stream, addr)) => return (ServedIo::new(stream), addr),
                // The peer gave up before it was accepted: take the next.
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::ConnectionRefused
                            | std::io::ErrorKind::ConnectionAborted
                            | std::io::ErrorKind::ConnectionReset
                    ) => {}
                // Out of file descriptors and the like: back off, as axum does.
                Err(e) => {
                    tracing::error!("accept failed: {e}");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.0.local_addr()
    }
}

impl Connected<IncomingStream<'_, ServedListener>> for RawSocket {
    fn connect_info(stream: IncomingStream<'_, ServedListener>) -> Self {
        Self(Arc::clone(&stream.io().claim))
    }
}

/// What `axum::serve` takes to serve `app` on `listener` with
/// connections that can be upgraded to a frame stream.
pub(crate) fn serving(
    listener: TcpListener,
    app: Router,
) -> (
    ServedListener,
    IntoMakeServiceWithConnectInfo<Router, RawSocket>,
) {
    (
        ServedListener(listener),
        app.into_make_service_with_connect_info::<RawSocket>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn accepted_connections_have_nodelay() {
        let bound = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = bound.local_addr().unwrap();
        let mut listener = ServedListener(bound);
        let _client = std::net::TcpStream::connect(addr).unwrap();
        let (io, _) = listener.accept().await;
        assert!(io.stream.as_ref().unwrap().nodelay().unwrap());
    }

    /// A connection nobody wants closes with its IO; one that is wanted
    /// outlives it, and can be taken, blocking, and used.
    #[tokio::test]
    async fn a_wanted_connection_outlives_its_io_and_others_close() {
        use std::io::{Read, Write};
        let bound = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = bound.local_addr().unwrap();
        let mut listener = ServedListener(bound);

        let mut plain = std::net::TcpStream::connect(addr).unwrap();
        let (io, _) = listener.accept().await;
        drop(io);
        let mut byte = [0u8; 1];
        assert_eq!(plain.read(&mut byte).unwrap(), 0, "an unclaimed one closes");

        let mut client = std::net::TcpStream::connect(addr).unwrap();
        let (io, _) = listener.accept().await;
        let socket = RawSocket(Arc::clone(&io.claim));
        socket.want();
        drop(io);
        let mut taken = socket.take().unwrap();
        client.write_all(b"ping").unwrap();
        let mut got = [0u8; 4];
        taken.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"ping");
        taken.write_all(b"pong").unwrap();
        client.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"pong");
        assert!(socket.take().is_err(), "a connection is taken once");
    }
}
