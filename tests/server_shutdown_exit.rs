//! Process-level check of `cuca run`'s exit: after a model is loaded, SIGTERM
//! and SIGINT, with the server idle and after a request, end the process with
//! status 0.
//!
//! A model on a GPU gives its device memory back when it is dropped, on a
//! thread of the backend, and a process that exits while that is in flight
//! aborts (ROCm: `pure virtual method called`, status 134). The library waits
//! for the engine thread when the last handle of a model is dropped (see
//! `provider::local::lifecycle`); this test runs the real binary to prove it.
//!
//! The model is `local { model { ... } }` of `cuca-test.kdl` (`source` and the
//! other keys, `device` included, are written into the server's
//! configuration), so the run covers the backend the file names: build with
//! the feature of that device (`--features provider-local-rocm`, ...). Without
//! a model the test prints `SKIP` and passes, unless `require_live #true`.
//! Unix only: the signals are sent with `kill(1)`.
//!
//! ```sh
//! # cuca-test.kdl: local { model { source "qwen3-0.6b-q8_0"; device "rocm:0"; }; }
//! RUSTFLAGS="-C target-cpu=native" cargo test --release --no-default-features --features server,provider-local-rocm --test server_shutdown_exit -- --nocapture
//! ```

#![cfg(all(unix, feature = "server"))]

mod local_env;

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

/// How often each signal and mode is repeated.
const ROUNDS: usize = 3;
/// The longest the model may take to load.
const READY_TIMEOUT: Duration = Duration::from_secs(600);
/// The longest the process may take to end after the signal.
const EXIT_TIMEOUT: Duration = Duration::from_secs(60);

/// `value` as a KDL argument or property value.
fn scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(format!(
            "\"{}\"",
            s.replace('\\', "\\\\").replace('"', "\\\"")
        )),
        Value::Bool(b) => Some(format!("#{b}")),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// One KDL node for `key` holding `value`: a scalar or a list of scalars as
/// arguments, an object of scalars as properties, any other object as a block
/// of children.
fn node(key: &str, value: &Value, indent: usize, out: &mut String) {
    let pad = " ".repeat(indent);
    if let Some(s) = scalar(value) {
        let _ = writeln!(out, "{pad}{key} {s}");
    } else if let Value::Array(items) = value {
        let args: Vec<String> = items.iter().filter_map(scalar).collect();
        let _ = writeln!(out, "{pad}{key} {}", args.join(" "));
    } else if let Value::Object(map) = value {
        if map.values().all(|v| scalar(v).is_some()) {
            let props: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{k}={}", scalar(v).unwrap_or_default()))
                .collect();
            let _ = writeln!(out, "{pad}{key} {}", props.join(" "));
        } else {
            let _ = writeln!(out, "{pad}{key} {{");
            for (k, v) in map {
                node(k, v, indent + 4, out);
            }
            let _ = writeln!(out, "{pad}}}");
        }
    }
}

/// The server's configuration: `local { model { ... } }` of `cuca-test.kdl`
/// as the one model of a `serve` section on `addr`; `None` without a model.
fn server_config(addr: SocketAddr) -> Option<String> {
    let config = local_env::test_config::get();
    let Value::Object(model) = config.local_model_raw()? else {
        return None;
    };
    if !model.contains_key("source") {
        return None;
    }
    let mut body = String::new();
    for (key, value) in model {
        node(key, value, 8, &mut body);
    }
    if !model.contains_key("cache_dir")
        && let Some(dir) = &config.cache_dir
    {
        node(
            "cache_dir",
            &Value::String(dir.display().to_string()),
            8,
            &mut body,
        );
    }
    Some(format!(
        "serve {{\n    listen \"{addr}\"\n    model \"m\" {{\n{body}    }}\n}}\n"
    ))
}

/// A free loopback address (bound and released: another process could take it
/// before the server does, which fails the run loudly at the bind).
fn free_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a free port");
    listener.local_addr().expect("its address")
}

/// One HTTP/1.1 exchange over a fresh connection: the status line's code and
/// the body.
fn http(addr: SocketAddr, request: &str) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .ok()?;
    stream.write_all(request.as_bytes()).ok()?;
    let mut reply = String::new();
    stream.read_to_string(&mut reply).ok()?;
    let status = reply.split_whitespace().nth(1)?.parse().ok()?;
    let body = reply
        .split_once("\r\n\r\n")
        .map_or("", |(_, b)| b)
        .to_string();
    Some((status, body))
}

fn health(addr: SocketAddr) -> bool {
    http(
        addr,
        "GET /health HTTP/1.1\r\nHost: cuca\r\nConnection: close\r\n\r\n",
    )
    .is_some_and(|(status, _)| status == 200)
}

fn chat(addr: SocketAddr) -> bool {
    let body = r#"{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"temperature":0}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: cuca\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    http(addr, &request).is_some_and(|(status, _)| status == 200)
}

/// Start `cuca run -c <config>` with its output in `dir`.
fn spawn(config: &Path, dir: &Path) -> Child {
    let log = std::fs::File::create(dir.join("cuca.log")).expect("a log file");
    Command::new(env!("CARGO_BIN_EXE_cuca"))
        .args(["run", "-c"])
        .arg(config)
        .stdin(Stdio::null())
        .stdout(log.try_clone().expect("a second handle"))
        .stderr(log)
        .spawn()
        .expect("`cuca run` starts")
}

/// Wait for `/health` to answer 200 while the process lives.
fn wait_ready(child: &mut Child, addr: SocketAddr, log: &Path) {
    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        if health(addr) {
            return;
        }
        if let Some(status) = child.try_wait().expect("the child can be polled") {
            let output = std::fs::read_to_string(log).unwrap_or_default();
            panic!("`cuca run` ended before it was ready ({status}):\n{output}");
        }
        assert!(Instant::now() < deadline, "the model did not load in time");
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Send `signal` (`TERM`, `INT`) to `child` and wait for it to end.
fn stop(child: &mut Child, signal: &str) -> std::process::ExitStatus {
    let sent = Command::new("kill")
        .arg(format!("-{signal}"))
        .arg(child.id().to_string())
        .status()
        .expect("kill(1) runs");
    assert!(sent.success(), "kill -{signal} failed");
    let deadline = Instant::now() + EXIT_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait().expect("the child can be polled") {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("`cuca run` did not end within {EXIT_TIMEOUT:?} of SIG{signal}");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn cuca_run_exits_cleanly_on_sigterm_and_sigint() {
    let addr = free_addr();
    let Some(config) = server_config(addr) else {
        return local_env::skip_no_model();
    };
    let dir = std::env::temp_dir().join(format!("cuca-shutdown-exit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a work directory");
    let path = dir.join("cuca.kdl");
    std::fs::write(&path, &config).expect("the configuration is written");
    println!("configuration:\n{config}");

    let mut failures = Vec::new();
    for signal in ["TERM", "INT"] {
        for served in [false, true] {
            for round in 1..=ROUNDS {
                let mut child = spawn(&path, &dir);
                wait_ready(&mut child, addr, &dir.join("cuca.log"));
                if served {
                    assert!(chat(addr), "the chat request failed");
                }
                let started = Instant::now();
                let status = stop(&mut child, signal);
                let what = format!(
                    "SIG{signal} {} round {round}",
                    if served { "after a request" } else { "idle" }
                );
                println!("{what}: {status} after {:.2?}", started.elapsed());
                if !status.success() {
                    let log = std::fs::read_to_string(dir.join("cuca.log")).unwrap_or_default();
                    failures.push(format!("{what}: {status}\n{log}"));
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "`cuca run` did not exit with status 0:\n{}",
        failures.join("\n")
    );
}
