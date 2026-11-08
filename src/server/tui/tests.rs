//! Render tests over ratatui's `TestBackend` with synthetic snapshots, and the
//! fixtures the other modules' tests share.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;

use super::app::{App, Pane};
use super::client::{Connection, Update};
use super::ui;
use crate::server::stats::{
    Cause, CpuStats, Diagnosis, DiskIo, DiskStats, GpuStats, KvStats, LatencyStats, MemoryStats,
    ModelMemory, ModelState, ModelStats, NodeInfo, NodeRole, ProcessStats, Quantiles, RequestStats,
    Severity, StatsSnapshot, SystemSnapshot, TokensPerSecond, UpstreamStats,
};

const GIB: u64 = 1 << 30;
const MIB: f64 = 1_048_576.0;

pub fn press(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn quantiles(p50: f64, p95: f64, p99: f64, mean: f64, count: u64) -> Quantiles {
    Quantiles {
        p50,
        p95,
        p99,
        mean,
        count,
    }
}

fn slow_latency() -> LatencyStats {
    LatencyStats {
        window_s: 60.0,
        ttft: quantiles(0.8, 3.4, 5.1, 1.2, 42),
        itl: quantiles(0.03, 0.09, 0.12, 0.04, 42),
        e2e: quantiles(6.0, 24.0, 31.0, 9.0, 42),
        queue: quantiles(0.4, 1.9, 2.4, 0.7, 42),
    }
}

fn model(name: &str, device: &str, busy: bool) -> ModelStats {
    ModelStats {
        name: name.to_string(),
        source: "preset".to_string(),
        device: device.to_string(),
        context_length: 8192,
        state: ModelState::Ready,
        max_sequences: 4,
        running: if busy { 4 } else { 1 },
        waiting: if busy { 5 } else { 0 },
        kv: KvStats {
            total_blocks: 512,
            free_blocks: 12,
            cached_blocks: 20,
            used_blocks: 480,
            bytes: GIB * 3 / 2,
            usage_ratio: if busy { 0.94 } else { 0.1 },
        },
        prefix_hit_ratio: Some(0.62),
        preemptions_per_min: if busy { 2.5 } else { 0.0 },
        tokens_per_s: TokensPerSecond {
            prompt: 820.0,
            generation: if busy { 94.3 } else { 31.0 },
        },
        steps_per_s: 18.0,
        rows_per_s: 60.0,
        mean_rows_per_step: 3.5,
        prefill_row_share: 0.4,
        mean_step_ms: if busy { 55.0 } else { 12.0 },
        last_step_ms: 60.0,
        mean_queue_ms: 1800.0,
        memory: ModelMemory {
            weights_bytes: GIB * 16 / 5,
            kv_bytes: GIB * 3 / 2,
            device: device.to_string(),
        },
        requests: RequestStats {
            in_flight: 8,
            total: 1280,
            errors: 3,
            rate_per_s: 2.4,
        },
        latency: slow_latency(),
        ..ModelStats::default()
    }
}

fn diagnosis() -> Vec<Diagnosis> {
    let entry = |severity, cause, message: &str, evidence: &[(&str, f64)]| Diagnosis {
        severity,
        cause,
        model: Some("gemma4-e2b-q4_0".to_string()),
        upstream: None,
        message: message.to_string(),
        evidence: evidence.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
    };
    vec![
        entry(
            Severity::Warn,
            Cause::KvPoolFull,
            "K/V pool is 94% full, requests wait for blocks",
            &[("kv_usage_ratio", 0.94)],
        ),
        entry(
            Severity::Critical,
            Cause::QueueSaturated,
            "All 4 sequence slots are busy and 5 requests wait",
            &[("waiting", 5.0), ("max_sequences", 4.0)],
        ),
        Diagnosis {
            severity: Severity::Info,
            cause: Cause::MemoryPressure,
            message: "Swap is in use".to_string(),
            ..Diagnosis::default()
        },
    ]
}

pub fn server_snapshot() -> StatsSnapshot {
    let gpu = |index, name: &str, vendor: &str, util, used_gib: u64, temp| GpuStats {
        index,
        name: name.to_string(),
        vendor: vendor.to_string(),
        util_pct: Some(util),
        vram_total_bytes: Some(24 * GIB),
        vram_used_bytes: Some(used_gib * GIB),
        process_vram_bytes: Some(used_gib * GIB * 3 / 4),
        shared_vram_used_bytes: Some(GIB / 4),
        temperature_c: temp,
        power_w: temp.map(|_| 410.0),
        source: "nvml".to_string(),
    };
    StatsSnapshot {
        node: NodeInfo {
            id: "127.0.0.1:8000".to_string(),
            role: NodeRole::Server,
            version: "0.1.0".to_string(),
            listen: "127.0.0.1:8000".to_string(),
            uptime_s: 3725.0,
            ..NodeInfo::default()
        },
        models: vec![
            model("gemma4-e2b-q4_0", "vulkan:0", true),
            model("qwen3-0.6b-q8_0", "cpu", false),
        ],
        latency: slow_latency(),
        requests: RequestStats {
            in_flight: 8,
            total: 1280,
            errors: 3,
            rate_per_s: 2.4,
        },
        system: SystemSnapshot {
            cpu: CpuStats {
                usage_pct: 91.5,
                per_core_pct: (0..16).map(|i| 20.0 + i as f32 * 5.0).collect(),
                cores: 16,
                load_avg: Some([12.1, 9.8, 7.2]),
                brand: "AMD Ryzen 9 7950X 16-Core Processor".to_string(),
            },
            memory: MemoryStats {
                total_bytes: 64 * GIB,
                used_bytes: 58 * GIB,
                available_bytes: 6 * GIB,
                swap_total_bytes: 16 * GIB,
                swap_used_bytes: 3 * GIB,
                swap_in_bytes_per_s: Some(0.0),
                swap_out_bytes_per_s: Some(6.0 * MIB),
            },
            disks: vec![
                DiskStats {
                    mount: "C:\\".to_string(),
                    name: "nvme0".to_string(),
                    total_bytes: 1000 * GIB,
                    available_bytes: 120 * GIB,
                    kind: "SSD".to_string(),
                },
                DiskStats {
                    mount: "D:\\".to_string(),
                    name: "sda".to_string(),
                    total_bytes: 4000 * GIB,
                    available_bytes: 300 * GIB,
                    kind: "HDD".to_string(),
                },
            ],
            disk_io: DiskIo {
                read_bytes_per_s: 350.0 * MIB,
                write_bytes_per_s: 12.0 * MIB,
            },
            gpus: vec![
                gpu(0, "NVIDIA GeForce RTX 4090", "nvidia", 97.0, 21, Some(78.0)),
                gpu(1, "AMD Radeon RX 7900 XT", "amd", 12.0, 2, None),
            ],
            process: ProcessStats {
                pid: 4242,
                cpu_pct: 640.0,
                rss_bytes: 6 * GIB,
                virtual_bytes: 30 * GIB,
                threads: Some(48),
                read_bytes_per_s: 300.0 * MIB,
                write_bytes_per_s: 0.0,
            },
            sampled_at_ms: 1_700_000_000_000,
        },
        diagnosis: diagnosis(),
        upstreams: Vec::new(),
    }
}

pub fn balancer_snapshot() -> StatsSnapshot {
    let mut first = server_snapshot();
    first.node.id = "10.0.0.11:8000".to_string();
    let mut second = server_snapshot();
    second.node.id = "10.0.0.12:8000".to_string();
    second.system.gpus.clear();
    second.diagnosis.clear();
    let upstream =
        |id: &str, url: &str, healthy: bool, snapshot: StatsSnapshot, error: Option<&str>| {
            UpstreamStats {
                id: id.to_string(),
                url: url.to_string(),
                healthy,
                in_flight: 5,
                latency: slow_latency(),
                error: error.map(str::to_string),
                routed_total: 5120,
                errors_total: 12,
                snapshot: Some(Box::new(snapshot)),
                snapshot_age_s: Some(0.5),
            }
        };
    StatsSnapshot {
        node: NodeInfo {
            id: "0.0.0.0:9000".to_string(),
            role: NodeRole::Balancer,
            version: "0.1.0".to_string(),
            uptime_s: 90_000.0,
            ..NodeInfo::default()
        },
        latency: slow_latency(),
        requests: RequestStats {
            in_flight: 10,
            total: 10_240,
            errors: 24,
            rate_per_s: 5.5,
        },
        diagnosis: vec![Diagnosis {
            severity: Severity::Warn,
            upstream: Some("cpu-box-2".to_string()),
            message: "1 of 2 upstreams is down".to_string(),
            ..Diagnosis::default()
        }],
        upstreams: vec![
            upstream("gpu-box-1", "http://10.0.0.11:8000", true, first, None),
            upstream(
                "cpu-box-2",
                "http://10.0.0.12:8000",
                false,
                second,
                Some("connection refused"),
            ),
        ],
        ..StatsSnapshot::default()
    }
}

fn buffer_text(buffer: &Buffer) -> String {
    let area = buffer.area;
    (0..area.height)
        .map(|y| {
            let row: String = (0..area.width).map(|x| buffer[(x, y)].symbol()).collect();
            row.trim_end().to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Draws one frame of `app` at `width` x `height` and returns its text.
pub fn render(app: &mut App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| ui::draw(frame, app)).unwrap();
    buffer_text(terminal.backend().buffer())
}

fn app_with(snapshot: StatsSnapshot) -> App {
    let mut app = App::new("http://127.0.0.1:8000", 1000);
    app.apply(Update::Connection(Connection::Connected));
    for _ in 0..3 {
        app.apply(Update::Snapshot(Box::new(snapshot.clone())));
    }
    app
}

fn assert_has(frame: &str, needles: &[&str]) {
    for needle in needles {
        assert!(frame.contains(needle), "missing `{needle}` in:\n{frame}");
    }
}

#[test]
fn a_server_shows_every_panel_at_120x40() {
    let mut app = app_with(server_snapshot());
    let frame = render(&mut app, 120, 40);
    assert_has(
        &frame,
        &[
            "SERVER",
            "127.0.0.1:8000",
            "● connected",
            "CPU",
            "RAM",
            "Swap",
            "VRAM0",
            "RTX 4090",
            "weights",
            "Disk read",
            "C:",
            "Latency",
            "p95",
            "TTFT",
            "Why is it slow?",
            "All 4 sequence slots are busy",
            "waiting=5",
            "Models (2)",
            "gemma4-e2b-q4_0",
            "qwen3-0.6b-q8_0",
        ],
    );
    let critical = frame.find("All 4 sequence slots").unwrap();
    let warning = frame.find("K/V pool is 94% full").unwrap();
    assert!(
        critical < warning,
        "the critical reason ranks above the warning"
    );
}

#[test]
fn a_server_stays_usable_at_80x24_one_pane_at_a_time() {
    let mut app = app_with(server_snapshot());
    let frame = render(&mut app, 80, 24);
    assert_has(
        &frame,
        &["CPU", "RAM", "VRAM0", "[Resources]", "All 4 sequence slots"],
    );

    app.on_key(press(KeyCode::Tab));
    assert_eq!(app.ui.focus, Pane::Latency);
    let frame = render(&mut app, 80, 24);
    assert_has(
        &frame,
        &["p95", "TTFT", "Queue", "All 4 sequence slots are busy"],
    );

    app.on_key(press(KeyCode::Tab));
    let frame = render(&mut app, 80, 24);
    assert_has(&frame, &["gemma4-e2b-q4_0", "KV usage"]);
}

#[test]
fn the_model_details_popup_lists_the_per_model_latency() {
    let mut app = app_with(server_snapshot());
    app.on_key(press(KeyCode::Tab));
    app.on_key(press(KeyCode::Tab));
    app.on_key(press(KeyCode::Enter));
    let frame = render(&mut app, 120, 40);
    assert_has(
        &frame,
        &[
            "Model details",
            "Latency of this model",
            "prefix hit",
            "K/V cache and memory",
        ],
    );
}

#[test]
fn a_balancer_lists_its_upstreams_and_drills_into_one() {
    let mut app = app_with(balancer_snapshot());
    let frame = render(&mut app, 120, 40);
    assert_has(
        &frame,
        &[
            "BALANCER",
            "Fleet resources (1/2 upstreams up)",
            "Upstreams (1/2 up)",
            "http://10.0.0.11:8000",
            "http://10.0.0.12:8000",
            "✗ down",
            "connection refused",
            "p95",
            "[cpu-box-2] 1 of 2 upstreams is down",
        ],
    );

    app.ui.focus = Pane::List;
    app.on_key(press(KeyCode::Enter));
    let frame = render(&mut app, 120, 40);
    assert_has(
        &frame,
        &[
            "upstream gpu-box-1",
            "RTX 4090",
            "Models (2)",
            "All 4 sequence slots",
        ],
    );
    app.on_key(press(KeyCode::Esc));
    assert!(render(&mut app, 120, 40).contains("Upstreams (1/2 up)"));
}

#[test]
fn a_balancer_fits_80x24() {
    let mut app = app_with(balancer_snapshot());
    app.ui.focus = Pane::List;
    let frame = render(&mut app, 80, 24);
    assert_has(&frame, &["gpu-box-1", "cpu-box-2", "[Upstreams]"]);
}

#[test]
fn waiting_and_reconnecting_keep_the_last_data() {
    let mut app = App::new("http://127.0.0.1:8000", 1000);
    assert_has(
        &render(&mut app, 120, 40),
        &["Waiting for http://127.0.0.1:8000", "--stats-api"],
    );

    let mut app = app_with(server_snapshot());
    app.apply(Update::Connection(Connection::Reconnecting {
        attempt: 2,
        retry_at: std::time::Instant::now() + std::time::Duration::from_secs(4),
        error: "connection refused or unreachable".to_string(),
    }));
    let frame = render(&mut app, 120, 40);
    assert_has(&frame, &["reconnecting in", "VRAM0", "connection refused"]);
}

#[test]
fn no_size_panics() {
    let mut servers = app_with(server_snapshot());
    let mut balancers = app_with(balancer_snapshot());
    for (width, height) in [
        (1, 1),
        (39, 9),
        (40, 10),
        (60, 14),
        (80, 24),
        (100, 33),
        (110, 34),
        (130, 60),
        (200, 12),
    ] {
        for app in [&mut servers, &mut balancers] {
            for pane in [Pane::Resources, Pane::Latency, Pane::List] {
                app.ui.focus = pane;
                render(app, width, height);
            }
        }
    }
}
