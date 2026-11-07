//! Model cache resolution and the resumable, verified Hugging Face download.
//!
//! [`resolve_cache_dir`] picks the cache root (explicit setting, `CUCA_CACHE_DIR`,
//! then the platform cache directory) and [`cached_path`] maps a repository
//! file to `<cache>/huggingface/<repo>/<revision>/<file>`, rejecting path
//! traversal. [`ensure_downloaded`] fetches
//! `{endpoint}/{repo}/resolve/{revision}/{file}` into `<final>.partial`:
//!
//! - an exclusive `<final>.lock` file serialises concurrent loaders (this
//!   process or another); the empty lock file is intentionally left in the
//!   cache, because deleting it after unlock would race with a process still
//!   blocked on it;
//! - an existing partial file is resumed with an HTTP `Range` request, falling
//!   back to a restart from byte 0 (once per call) when the server ignores or
//!   rejects the range;
//! - the SHA-256 is computed while streaming, covering any resumed prefix, and
//!   a mismatch deletes the partial file;
//! - the verified partial file is renamed atomically, so a present final file is
//!   always complete and verified.
//!
//! The functions use blocking `std::fs` calls: they run on the engine's
//! dedicated loader thread inside a current-thread runtime. Memory use is
//! bounded by a 1 MiB write buffer and one network chunk.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::{DownloadProgress, DownloadProgressCallback, convert};
use crate::error::CucaError;
use crate::types::ProviderEndpoint;

/// Default Hugging Face endpoint.
pub(crate) const HF_ENDPOINT: &str = "https://huggingface.co";

/// Write-buffer capacity and progress-report granularity.
const MIB: usize = 1024 * 1024;
/// How long a download may go without receiving a chunk.
const STALL_TIMEOUT: Duration = Duration::from_secs(60);
/// Longest error-response body kept in a [`CucaError::Http`], in characters.
const ERROR_BODY_CHARS: usize = 512;
/// Most bytes read from an error response body.
const ERROR_BODY_READ_BYTES: usize = 4096;

/// A non-empty env value, if set.
fn env_nonempty(env: &dyn Fn(&str) -> Option<OsString>, key: &str) -> Option<OsString> {
    env(key).filter(|v| !v.is_empty())
}

/// Resolve the model cache root.
///
/// Precedence (first wins): `explicit`; `CUCA_CACHE_DIR` as given; on Windows
/// `%LOCALAPPDATA%\cuca`, else `%USERPROFILE%\AppData\Local\cuca`; on macOS
/// `$HOME/Library/Caches/cuca`; elsewhere `$XDG_CACHE_HOME/cuca`, else
/// `$HOME/.cache/cuca`. Empty environment values count as unset. `env` reads
/// environment variables, so tests can inject a fake one.
pub(crate) fn resolve_cache_dir(
    explicit: Option<&Path>,
    env: &dyn Fn(&str) -> Option<OsString>,
) -> Result<PathBuf, CucaError> {
    if let Some(dir) = explicit {
        return Ok(dir.to_path_buf());
    }
    if let Some(dir) = env_nonempty(env, "CUCA_CACHE_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let platform = if cfg!(windows) {
        env_nonempty(env, "LOCALAPPDATA")
            .map(|d| PathBuf::from(d).join("cuca"))
            .or_else(|| {
                env_nonempty(env, "USERPROFILE")
                    .map(|d| PathBuf::from(d).join("AppData").join("Local").join("cuca"))
            })
    } else if cfg!(target_os = "macos") {
        env_nonempty(env, "HOME")
            .map(|d| PathBuf::from(d).join("Library").join("Caches").join("cuca"))
    } else {
        env_nonempty(env, "XDG_CACHE_HOME")
            .map(|d| PathBuf::from(d).join("cuca"))
            .or_else(|| {
                env_nonempty(env, "HOME").map(|d| PathBuf::from(d).join(".cache").join("cuca"))
            })
    };
    platform.ok_or_else(|| {
        CucaError::Config(
            "cannot resolve a model cache directory; set LocalModelConfig::cache_dir or CUCA_CACHE_DIR"
                .into(),
        )
    })
}

/// Validate one repository path component (`repo`, `revision` or `file`).
fn check_component(label: &str, value: &str) -> Result<(), CucaError> {
    let bad = value.is_empty()
        || value.contains('\\')
        || value.starts_with('/')
        || value.split('/').any(|segment| {
            segment.is_empty() || segment == "." || segment == ".." || segment.contains(':')
        });
    if bad {
        return Err(CucaError::Config(format!(
            "invalid model {label} `{value}`: it must be a relative `/`-separated path of plain segments"
        )));
    }
    Ok(())
}

/// Cache location of a repository file:
/// `<cache>/huggingface/<repo with '/' as "--">/<revision>/<file>`.
///
/// Fails with [`CucaError::Config`] when `repo`, `revision` or `file` is empty,
/// has an empty, `.`, `..` or `:`-containing segment, contains a backslash, or
/// starts with `/`.
pub(crate) fn cached_path(
    cache_dir: &Path,
    repo: &str,
    revision: &str,
    file: &str,
) -> Result<PathBuf, CucaError> {
    check_component("repo", repo)?;
    check_component("revision", revision)?;
    check_component("file", file)?;
    let mut path = cache_dir
        .join("huggingface")
        .join(repo.replace('/', "--"))
        .join(revision);
    for segment in file.split('/') {
        path.push(segment);
    }
    Ok(path)
}

/// `path` with `suffix` appended to its file name (`.partial`, `.lock`).
pub(crate) fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// An exclusive `<target>.lock` file, released on drop.
///
/// The empty lock file is left in place: deleting it after unlock would race
/// with a process still blocked on it.
pub(crate) struct FileLock {
    _file: File,
}

/// Take the exclusive `<target>.lock` file (creating `target`'s directory and
/// the lock file), waiting for another loader that holds it.
pub(crate) fn lock_file(target: &Path) -> Result<FileLock, CucaError> {
    let parent = target.parent().ok_or_else(|| {
        CucaError::Config(format!("`{}` has no parent directory", target.display()))
    })?;
    std::fs::create_dir_all(parent)?;
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(with_suffix(target, ".lock"))?;
    file.lock()?;
    Ok(FileLock { _file: file })
}

/// Lowercase hex of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Hash the whole of `path` into `hasher`.
fn hash_file(path: &Path, hasher: &mut Sha256) -> Result<(), CucaError> {
    let mut file = File::open(path)?;
    let mut buf = vec![0u8; MIB];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        hasher.update(&buf[..n]);
    }
}

/// Report progress when a callback is installed; `total` is a byte count or
/// `None` while unknown.
pub(crate) fn report(
    progress: Option<&DownloadProgressCallback>,
    downloaded: u64,
    total: impl Into<Option<u64>>,
) {
    if let Some(cb) = progress {
        (cb.0)(DownloadProgress {
            downloaded_bytes: downloaded,
            total_bytes: total.into(),
        });
    }
}

/// Parse `bytes START-END/TOTAL` into `(start, total)`; `total` is `None` for `*`.
fn parse_content_range(value: &str) -> Option<(u64, Option<u64>)> {
    let rest = value.trim().strip_prefix("bytes")?.trim_start();
    let (range, total) = rest.split_once('/')?;
    let (start, _end) = range.split_once('-')?;
    let start = start.trim().parse().ok()?;
    let total = match total.trim() {
        "*" => None,
        t => Some(t.parse().ok()?),
    };
    Some((start, total))
}

/// Read up to a few KiB of an error response and truncate it for diagnostics.
async fn error_body(mut response: reqwest::Response) -> String {
    let mut bytes = Vec::new();
    while bytes.len() < ERROR_BODY_READ_BYTES {
        match tokio::time::timeout(STALL_TIMEOUT, response.chunk()).await {
            Ok(Ok(Some(chunk))) => bytes.extend_from_slice(&chunk),
            _ => break,
        }
    }
    String::from_utf8_lossy(&bytes)
        .chars()
        .take(ERROR_BODY_CHARS)
        .collect()
}

/// Make sure `final_path` holds the verified contents of `file` from
/// `{endpoint}/{repo}/resolve/{revision}/{file}`, downloading (or resuming) it
/// when absent.
///
/// `sha256` is the expected lowercase hex digest, checked before the atomic
/// rename; `token` is sent as a bearer token; `progress` is called at the start,
/// every 1 MiB, and at the end. A download that receives no data for 60 seconds
/// fails with [`CucaError::Transport`], keeping the partial file for a later
/// resume.
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is an independent input of one download; a parameter struct would only restate them"
)]
pub(crate) async fn ensure_downloaded(
    client: &reqwest::Client,
    endpoint: &str,
    repo: &str,
    revision: &str,
    file: &str,
    sha256: Option<&str>,
    final_path: &Path,
    token: Option<&str>,
    progress: Option<&DownloadProgressCallback>,
) -> Result<(), CucaError> {
    if final_path.exists() {
        return Ok(());
    }
    let _lock = lock_file(final_path)?;
    // Another loader may have finished while this one waited for the lock.
    if final_path.exists() {
        return Ok(());
    }
    let url = format!(
        "{}/{repo}/resolve/{revision}/{file}",
        endpoint.trim_end_matches('/')
    );
    download_locked(client, &url, file, sha256, final_path, token, progress).await
}

/// Make sure `merged` holds the merge of a split GGUF set: download and verify
/// every `(file, sha256)` part (resuming partial files), merge them into
/// `merged` and delete the parts, so the merged file is the cache entry.
///
/// An exclusive `<merged>.lock` is held for the whole operation, so a
/// concurrent loader waits and then finds `merged` instead of downloading
/// parts that the first loader is about to delete. A failed merge leaves the
/// verified parts for a retry. `progress` receives each part's download, then
/// the merge's copied bytes.
///
/// # Errors
///
/// Those of [`ensure_downloaded`] and of the merge ([`CucaError::Config`]
/// naming the offending part).
#[expect(
    clippy::too_many_arguments,
    reason = "same inputs as ensure_downloaded, plus the cache and the parts"
)]
pub(crate) async fn ensure_split_downloaded(
    client: &reqwest::Client,
    endpoint: &str,
    cache_dir: &Path,
    repo: &str,
    revision: &str,
    parts: &[(String, Option<String>)],
    merged: &Path,
    token: Option<&str>,
    progress: Option<&DownloadProgressCallback>,
) -> Result<(), CucaError> {
    if merged.exists() {
        return Ok(());
    }
    let lock = lock_file(merged)?;
    // Another loader may have merged while this one waited for the lock.
    if merged.exists() {
        return Ok(());
    }
    let mut paths = Vec::with_capacity(parts.len());
    for (file, sha256) in parts {
        let path = cached_path(cache_dir, repo, revision, file)?;
        ensure_downloaded(
            client,
            endpoint,
            repo,
            revision,
            file,
            sha256.as_deref(),
            &path,
            token,
            progress,
        )
        .await?;
        paths.push(path);
    }
    convert::build_locked(&lock, merged, |partial| {
        convert::gguf_merge::merge_to(&paths, partial, progress)
    })?;
    for path in &paths {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(with_suffix(path, ".lock"));
    }
    Ok(())
}

/// [`ensure_downloaded`] for a file the repository may not have: `Ok(false)`
/// when the server answers HTTP 404 (nothing is written), `Ok(true)` when
/// `final_path` holds the file.
#[expect(
    clippy::too_many_arguments,
    reason = "same inputs as ensure_downloaded"
)]
pub(crate) async fn ensure_downloaded_optional(
    client: &reqwest::Client,
    endpoint: &str,
    repo: &str,
    revision: &str,
    file: &str,
    sha256: Option<&str>,
    final_path: &Path,
    token: Option<&str>,
    progress: Option<&DownloadProgressCallback>,
) -> Result<bool, CucaError> {
    match ensure_downloaded(
        client, endpoint, repo, revision, file, sha256, final_path, token, progress,
    )
    .await
    {
        Ok(()) => Ok(true),
        Err(CucaError::Http { status: 404, .. }) => Ok(false),
        Err(e) => Err(e),
    }
}

/// The download proper, run while holding the lock file.
async fn download_locked(
    client: &reqwest::Client,
    url: &str,
    file: &str,
    sha256: Option<&str>,
    final_path: &Path,
    token: Option<&str>,
    progress: Option<&DownloadProgressCallback>,
) -> Result<(), CucaError> {
    let partial = with_suffix(final_path, ".partial");
    let new_hasher = || sha256.map(|_| Sha256::new());
    let mut hasher = new_hasher();
    let mut offset = std::fs::metadata(&partial).map_or(0, |m| m.len());
    if offset > 0
        && let Some(h) = hasher.as_mut()
    {
        hash_file(&partial, h)?;
    }

    let mut restarted = false;
    let (mut response, total) = loop {
        let mut request = client
            .get(url)
            .header(reqwest::header::ACCEPT_ENCODING, "identity");
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        if offset > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }
        let response = request.send().await?;
        let status = response.status().as_u16();
        let content_range = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_content_range);

        // Every arm either breaks out with a usable response, returns an error,
        // or falls through to the single restart below.
        match status {
            206 => match content_range {
                Some((start, total)) if start == offset => {
                    let total = total.or_else(|| response.content_length().map(|l| offset + l));
                    break (response, total);
                }
                _ => {}
            },
            200 => {
                // Without a range request this is the whole file; with one,
                // the server ignored it and its body is the whole file too.
                offset = 0;
                hasher = new_hasher();
                let total = response.content_length();
                break (response, total);
            }
            416 if !restarted => {}
            _ if (200..300).contains(&status) => {
                let total = response.content_length().map(|l| offset + l);
                break (response, total);
            }
            _ => {
                return Err(CucaError::Http {
                    status,
                    body: error_body(response).await,
                });
            }
        }
        if restarted {
            return Err(CucaError::Transport {
                message: format!("server returned an unusable range response for {file}"),
            });
        }
        restarted = true;
        offset = 0;
        hasher = new_hasher();
        match std::fs::remove_file(&partial) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    };

    let file_handle = if offset > 0 {
        OpenOptions::new().append(true).open(&partial)?
    } else {
        File::create(&partial)?
    };
    let mut writer = BufWriter::with_capacity(MIB, file_handle);
    let mut downloaded = offset;
    let mut last_reported = offset;
    report(progress, downloaded, total);
    loop {
        let chunk = tokio::time::timeout(STALL_TIMEOUT, response.chunk())
            .await
            .map_err(|_| CucaError::Transport {
                message: "download stalled for 60s".to_string(),
            })??;
        let Some(chunk) = chunk else { break };
        writer.write_all(&chunk)?;
        if let Some(h) = hasher.as_mut() {
            h.update(&chunk);
        }
        downloaded += chunk.len() as u64;
        if downloaded - last_reported >= MIB as u64 {
            report(progress, downloaded, total);
            last_reported = downloaded;
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    report(progress, downloaded, total);

    if let Some(total) = total
        && downloaded != total
    {
        return Err(CucaError::Transport {
            message: format!("download of {file} ended at {downloaded} of {total} bytes"),
        });
    }
    if let (Some(expected), Some(h)) = (sha256, hasher) {
        let got = hex(&h.finalize());
        let expected = expected.to_ascii_lowercase();
        if got != expected {
            let _ = std::fs::remove_file(&partial);
            return Err(CucaError::provider(
                ProviderEndpoint::Local,
                format!("sha256 mismatch for {file}: expected {expected}, got {got}"),
            ));
        }
    }
    std::fs::rename(&partial, final_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn fake_env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key| {
            pairs
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| OsString::from(v))
        }
    }

    #[test]
    fn explicit_cache_dir_wins() {
        let env = fake_env(&[
            ("CUCA_CACHE_DIR", "/env"),
            ("HOME", "/home/u"),
            ("LOCALAPPDATA", "C:/l"),
        ]);
        let dir = resolve_cache_dir(Some(Path::new("/explicit")), &env).expect("resolves");
        assert_eq!(dir, PathBuf::from("/explicit"));
    }

    #[test]
    fn env_override_is_used_as_is() {
        let env = fake_env(&[("CUCA_CACHE_DIR", "/env/cache"), ("HOME", "/home/u")]);
        assert_eq!(
            resolve_cache_dir(None, &env).expect("resolves"),
            PathBuf::from("/env/cache")
        );
    }

    #[test]
    fn platform_directory_precedence() {
        if cfg!(windows) {
            let env = fake_env(&[("LOCALAPPDATA", "C:/Local"), ("USERPROFILE", "C:/Users/u")]);
            assert_eq!(
                resolve_cache_dir(None, &env).expect("resolves"),
                PathBuf::from("C:/Local").join("cuca")
            );
            let env = fake_env(&[("USERPROFILE", "C:/Users/u")]);
            assert_eq!(
                resolve_cache_dir(None, &env).expect("resolves"),
                PathBuf::from("C:/Users/u")
                    .join("AppData")
                    .join("Local")
                    .join("cuca")
            );
        } else if cfg!(target_os = "macos") {
            let env = fake_env(&[("HOME", "/Users/u"), ("XDG_CACHE_HOME", "/xdg")]);
            assert_eq!(
                resolve_cache_dir(None, &env).expect("resolves"),
                PathBuf::from("/Users/u")
                    .join("Library")
                    .join("Caches")
                    .join("cuca")
            );
        } else {
            let env = fake_env(&[("XDG_CACHE_HOME", "/xdg"), ("HOME", "/home/u")]);
            assert_eq!(
                resolve_cache_dir(None, &env).expect("resolves"),
                PathBuf::from("/xdg").join("cuca")
            );
            let env = fake_env(&[("HOME", "/home/u")]);
            assert_eq!(
                resolve_cache_dir(None, &env).expect("resolves"),
                PathBuf::from("/home/u").join(".cache").join("cuca")
            );
        }
    }

    #[test]
    fn unresolvable_cache_dir_is_a_config_error() {
        let env = fake_env(&[("CUCA_CACHE_DIR", "")]);
        let err = resolve_cache_dir(None, &env).expect_err("nothing to resolve from");
        assert!(matches!(&err, CucaError::Config(m) if m.contains("CUCA_CACHE_DIR")));
    }

    #[test]
    fn cached_path_layout() {
        let path = cached_path(Path::new("cache"), "org/model", "abc123", "dir/model.gguf")
            .expect("valid components");
        let expected: PathBuf = [
            "cache",
            "huggingface",
            "org--model",
            "abc123",
            "dir",
            "model.gguf",
        ]
        .iter()
        .collect();
        assert_eq!(path, expected);
    }

    #[test]
    fn cached_path_rejects_traversal_and_bad_components() {
        let cache = Path::new("cache");
        let bad = [
            ("", "rev", "f.gguf"),
            ("org/m", "", "f.gguf"),
            ("org/m", "rev", ""),
            ("../m", "rev", "f.gguf"),
            ("org/m", "..", "f.gguf"),
            ("org/m", "rev", "../f.gguf"),
            ("org/m", "rev", "a/../../f.gguf"),
            ("org/m", "rev", "a\\b.gguf"),
            ("org\\m", "rev", "f.gguf"),
            ("org/m", "rev", "/abs.gguf"),
            ("org/m", "rev", "a//b.gguf"),
            ("org/m", "rev", "C:evil.gguf"),
        ];
        for (repo, rev, file) in bad {
            assert!(
                matches!(
                    cached_path(cache, repo, rev, file),
                    Err(CucaError::Config(_))
                ),
                "should reject ({repo:?}, {rev:?}, {file:?})"
            );
        }
    }

    #[test]
    fn content_range_parsing() {
        assert_eq!(parse_content_range("bytes 5-9/10"), Some((5, Some(10))));
        assert_eq!(parse_content_range("bytes 0-9/*"), Some((0, None)));
        assert_eq!(parse_content_range("bytes */10"), None);
        assert_eq!(parse_content_range("garbage"), None);
    }

    /// How the stub server treats a `Range` request.
    #[derive(Clone, Copy)]
    enum Mode {
        /// Serve 206 with `Content-Range`.
        Honour,
        /// Ignore `Range` and serve 200 with the full body.
        Ignore,
        /// Answer any ranged request with 416; full body otherwise.
        Reject416,
        /// Answer every request with 404.
        NotFound,
    }

    struct Stub {
        base: String,
        ranges: Arc<Mutex<Vec<Option<String>>>>,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for Stub {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    /// Serve every connection from `respond(request head)`: the status text,
    /// extra header lines (each ending in `\r\n`) and the payload.
    async fn serve(
        respond: impl Fn(&str) -> (&'static str, String, Vec<u8>) + Send + 'static,
    ) -> Stub {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind stub");
        let base = format!(
            "http://127.0.0.1:{}",
            listener.local_addr().expect("addr").port()
        );
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let mut raw = Vec::new();
                let mut buf = [0u8; 2048];
                while !raw.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => raw.extend_from_slice(&buf[..n]),
                    }
                }
                let (status, extra, payload) = respond(&String::from_utf8_lossy(&raw));
                let response = format!(
                    "HTTP/1.1 {status}\r\n{extra}content-length: {}\r\nconnection: close\r\n\r\n",
                    payload.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.write_all(&payload).await;
                let _ = socket.shutdown().await;
            }
        });
        Stub {
            base,
            ranges: Arc::default(),
            task,
        }
    }

    /// A stub serving `body` as `mode` says, recording every `Range` header.
    async fn spawn_stub(body: Vec<u8>, mode: Mode) -> Stub {
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let seen = ranges.clone();
        let mut stub = serve(move |head| {
            let range = head.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("range")
                    .then(|| value.trim().to_string())
            });
            seen.lock().expect("lock").push(range.clone());
            let start = range
                .as_deref()
                .and_then(|r| r.strip_prefix("bytes="))
                .and_then(|r| r.strip_suffix('-'))
                .and_then(|r| r.parse::<usize>().ok());
            let len = body.len();
            match (mode, start) {
                (Mode::NotFound, _) => ("404 Not Found", String::new(), b"nope".to_vec()),
                (Mode::Honour, Some(s)) => (
                    "206 Partial Content",
                    format!("content-range: bytes {s}-{}/{len}\r\n", len - 1),
                    body[s..].to_vec(),
                ),
                (Mode::Reject416, Some(_)) => (
                    "416 Range Not Satisfiable",
                    format!("content-range: bytes */{len}\r\n"),
                    Vec::new(),
                ),
                _ => ("200 OK", String::new(), body.clone()),
            }
        })
        .await;
        stub.ranges = ranges;
        stub
    }

    /// Temp directory removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static NEXT: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "cuca-download-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).expect("create temp dir");
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn body() -> Vec<u8> {
        (0..100 * 1024u32)
            .map(|i| (i.wrapping_mul(31) % 251) as u8)
            .collect()
    }

    fn sha_of(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    fn client() -> reqwest::Client {
        reqwest::Client::builder().build().expect("client builds")
    }

    async fn fetch(
        stub: &Stub,
        dir: &TempDir,
        sha: Option<&str>,
        progress: Option<&DownloadProgressCallback>,
    ) -> Result<PathBuf, CucaError> {
        let final_path = cached_path(&dir.0, "org/model", "rev1", "model.gguf").expect("path");
        ensure_downloaded(
            &client(),
            &stub.base,
            "org/model",
            "rev1",
            "model.gguf",
            sha,
            &final_path,
            Some("tok"),
            progress,
        )
        .await
        .map(|()| final_path)
    }

    fn seed_partial(final_path: &Path, bytes: &[u8]) {
        std::fs::create_dir_all(final_path.parent().expect("parent")).expect("mkdir");
        std::fs::write(with_suffix(final_path, ".partial"), bytes).expect("seed partial");
    }

    fn final_path(dir: &TempDir) -> PathBuf {
        cached_path(&dir.0, "org/model", "rev1", "model.gguf").expect("path")
    }

    #[tokio::test]
    async fn fresh_download_verifies_and_reports_progress() {
        let data = body();
        let stub = spawn_stub(data.clone(), Mode::Honour).await;
        let dir = TempDir::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        let cb = DownloadProgressCallback::new(move |p| sink.lock().expect("lock").push(p));
        let path = fetch(&stub, &dir, Some(&sha_of(&data)), Some(&cb))
            .await
            .expect("download succeeds");
        assert_eq!(std::fs::read(&path).expect("final file"), data);
        assert!(!with_suffix(&path, ".partial").exists());
        let events = events.lock().expect("lock");
        assert_eq!(
            events.first().copied(),
            Some(DownloadProgress {
                downloaded_bytes: 0,
                total_bytes: Some(data.len() as u64)
            })
        );
        assert_eq!(
            events.last().copied(),
            Some(DownloadProgress {
                downloaded_bytes: data.len() as u64,
                total_bytes: Some(data.len() as u64)
            })
        );
        assert_eq!(stub.ranges.lock().expect("lock").as_slice(), [None]);
    }

    #[tokio::test]
    async fn existing_final_file_skips_the_network() {
        let dir = TempDir::new();
        let path = final_path(&dir);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(&path, b"already here").expect("write final");
        let stub = Stub {
            base: "http://127.0.0.1:1".to_string(),
            ranges: Arc::default(),
            task: tokio::spawn(async {}),
        };
        fetch(&stub, &dir, Some("00"), None)
            .await
            .expect("no download needed");
        assert_eq!(std::fs::read(&path).expect("final"), b"already here");
    }

    #[tokio::test]
    async fn resumes_a_partial_file_with_a_range_request() {
        let data = body();
        let stub = spawn_stub(data.clone(), Mode::Honour).await;
        let dir = TempDir::new();
        let half = data.len() / 2;
        seed_partial(&final_path(&dir), &data[..half]);
        let path = fetch(&stub, &dir, Some(&sha_of(&data)), None)
            .await
            .expect("resume succeeds");
        assert_eq!(std::fs::read(&path).expect("final"), data);
        assert!(!with_suffix(&path, ".partial").exists());
        assert_eq!(
            stub.ranges.lock().expect("lock").as_slice(),
            [Some(format!("bytes={half}-"))]
        );
    }

    #[tokio::test]
    async fn server_ignoring_range_restarts_from_zero() {
        let data = body();
        let stub = spawn_stub(data.clone(), Mode::Ignore).await;
        let dir = TempDir::new();
        seed_partial(&final_path(&dir), &data[..4096]);
        let path = fetch(&stub, &dir, Some(&sha_of(&data)), None)
            .await
            .expect("restart succeeds");
        assert_eq!(std::fs::read(&path).expect("final"), data);
    }

    #[tokio::test]
    async fn range_not_satisfiable_discards_the_partial_and_retries() {
        let data = body();
        let stub = spawn_stub(data.clone(), Mode::Reject416).await;
        let dir = TempDir::new();
        seed_partial(&final_path(&dir), b"garbage that is not a prefix");
        let path = fetch(&stub, &dir, Some(&sha_of(&data)), None)
            .await
            .expect("restart succeeds");
        assert_eq!(std::fs::read(&path).expect("final"), data);
        assert_eq!(stub.ranges.lock().expect("lock").len(), 2);
    }

    /// A stub that serves `routes` (`(path suffix, body)`) with plain 200s.
    async fn spawn_routes(routes: Vec<(String, Vec<u8>)>) -> Stub {
        serve(move |head| {
            let path = head
                .lines()
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .unwrap_or("");
            match routes
                .iter()
                .find(|(suffix, _)| path.ends_with(suffix.as_str()))
            {
                Some((_, body)) => ("200 OK", String::new(), body.clone()),
                None => ("404 Not Found", String::new(), b"nope".to_vec()),
            }
        })
        .await
    }

    /// A tiny model cut into three parts (the first without tensors): the
    /// part names with their bytes.
    fn split_parts() -> (Vec<(String, Vec<u8>)>, usize) {
        use super::super::convert::gguf_merge::fixtures::{load, split, temp_dir};
        let model = crate::provider::local::model::tiny::write_tiny_model();
        let src = temp_dir("dl-src");
        let paths = split(&model, &src, 3, 3, |i| 1 + i % 2);
        let parts = paths
            .iter()
            .map(|p| {
                (
                    p.file_name().expect("name").to_string_lossy().into_owned(),
                    std::fs::read(p).expect("part"),
                )
            })
            .collect();
        let _ = std::fs::remove_dir_all(src);
        (parts, load(&model.path).1.len())
    }

    #[tokio::test]
    async fn a_split_set_downloads_verifies_merges_and_deletes_its_parts() {
        use super::super::convert::gguf_merge::fixtures::load;
        let (parts, n_tensors) = split_parts();
        let stub = spawn_routes(
            parts
                .iter()
                .map(|(name, body)| (format!("/{name}"), body.clone()))
                .collect(),
        )
        .await;
        let dir = TempDir::new();
        let merged = cached_path(&dir.0, "org/model", "rev1", "tiny.gguf").expect("path");
        let wanted: Vec<(String, Option<String>)> = parts
            .iter()
            .map(|(name, body)| (name.clone(), Some(sha_of(body))))
            .collect();
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        let cb = DownloadProgressCallback::new(move |p| sink.lock().expect("lock").push(p));
        ensure_split_downloaded(
            &client(),
            &stub.base,
            &dir.0,
            "org/model",
            "rev1",
            &wanted,
            &merged,
            None,
            Some(&cb),
        )
        .await
        .expect("split download succeeds");
        assert_eq!(load(&merged).1.len(), n_tensors);
        for (name, _) in &parts {
            let part = cached_path(&dir.0, "org/model", "rev1", name).expect("path");
            assert!(!part.exists(), "{name} is deleted after the merge");
            assert!(!with_suffix(&part, ".partial").exists());
        }
        assert!(!with_suffix(&merged, ".partial").exists());
        let last = events
            .lock()
            .expect("lock")
            .last()
            .copied()
            .expect("events");
        assert_eq!(
            Some(last.downloaded_bytes),
            last.total_bytes,
            "merge reports its end"
        );
        // The merged file is the cache entry: nothing is fetched again.
        drop(stub);
        ensure_split_downloaded(
            &client(),
            "http://127.0.0.1:1",
            &dir.0,
            "org/model",
            "rev1",
            &wanted,
            &merged,
            None,
            None,
        )
        .await
        .expect("a present merged file skips the network");
    }

    #[tokio::test]
    async fn a_part_with_the_wrong_digest_fails_before_the_merge() {
        let (parts, _) = split_parts();
        let stub = spawn_routes(
            parts
                .iter()
                .map(|(name, body)| (format!("/{name}"), body.clone()))
                .collect(),
        )
        .await;
        let dir = TempDir::new();
        let merged = cached_path(&dir.0, "org/model", "rev1", "tiny.gguf").expect("path");
        let mut wanted: Vec<(String, Option<String>)> = parts
            .iter()
            .map(|(name, body)| (name.clone(), Some(sha_of(body))))
            .collect();
        wanted[1].1 = Some("0".repeat(64));
        let err = ensure_split_downloaded(
            &client(),
            &stub.base,
            &dir.0,
            "org/model",
            "rev1",
            &wanted,
            &merged,
            None,
            None,
        )
        .await
        .expect_err("digest mismatch must fail");
        assert!(err.to_string().contains(&parts[1].0), "{err}");
        assert!(!merged.exists());
        assert!(!with_suffix(&merged, ".partial").exists());
        // The verified first part stays for the retry.
        let first = cached_path(&dir.0, "org/model", "rev1", &parts[0].0).expect("path");
        assert!(first.exists());
    }

    #[tokio::test]
    async fn wrong_sha_fails_and_leaves_no_files() {
        let data = body();
        let stub = spawn_stub(data.clone(), Mode::Honour).await;
        let dir = TempDir::new();
        let wrong = "0".repeat(64);
        let err = fetch(&stub, &dir, Some(&wrong), None)
            .await
            .expect_err("digest mismatch must fail");
        let message = err.to_string();
        assert!(
            message.contains("sha256 mismatch for model.gguf"),
            "{message}"
        );
        assert!(
            message.contains(&wrong) && message.contains(&sha_of(&data)),
            "{message}"
        );
        let path = final_path(&dir);
        assert!(!path.exists());
        assert!(!with_suffix(&path, ".partial").exists());
    }

    #[tokio::test]
    async fn http_error_status_is_reported_with_body() {
        let stub = spawn_stub(body(), Mode::NotFound).await;
        let dir = TempDir::new();
        let err = fetch(&stub, &dir, None, None)
            .await
            .expect_err("404 must fail");
        assert!(matches!(&err, CucaError::Http { status: 404, body } if body == "nope"));
        assert!(!final_path(&dir).exists());
    }

    #[tokio::test]
    async fn download_without_sha_skips_verification() {
        let data = body();
        let stub = spawn_stub(data.clone(), Mode::Honour).await;
        let dir = TempDir::new();
        let path = fetch(&stub, &dir, None, None)
            .await
            .expect("download succeeds");
        assert_eq!(std::fs::read(&path).expect("final"), data);
    }
}
