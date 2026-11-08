//! The resources panel: CPU, RAM and swap, disks, GPUs and VRAM, the process.
//!
//! The panel is a list of one-row items. When the area is shorter than the
//! list, the items of the highest priority number go first, so CPU, RAM and
//! each GPU's utilization and VRAM stay on screen longest.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::widgets::{
    ACCENT, DIM, OK, Trend, WARN, fit, gauge_row, level, panel, ratio, spark, usage_color,
};
use crate::server::stats::{GpuStats, ModelState, ModelStats, SystemSnapshot};
use crate::server::tui::format;
use crate::server::tui::history::History;

/// What one panel row shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Item {
    Cpu,
    Cores,
    Ram,
    Swap,
    Paging,
    NoGpu,
    GpuUtil(usize),
    GpuVram(usize),
    GpuInfo(usize),
    GpuMem(usize),
    DiskRead,
    DiskWrite,
    DiskSpace(usize),
    Process,
    Gap,
}

/// The rows in display order with their drop priority (higher goes first).
fn plan(system: &SystemSnapshot) -> Vec<(u8, Item)> {
    let mut items = vec![(0, Item::Cpu)];
    if !system.cpu.per_core_pct.is_empty() {
        items.push((3, Item::Cores));
    }
    items.extend([
        (0, Item::Ram),
        (1, Item::Swap),
        (1, Item::Paging),
        (9, Item::Gap),
    ]);
    if system.gpus.is_empty() {
        items.push((2, Item::NoGpu));
    }
    for index in 0..system.gpus.len() {
        items.push((0, Item::GpuUtil(index)));
        items.push((0, Item::GpuVram(index)));
        items.push((2, Item::GpuInfo(index)));
        items.push((2, Item::GpuMem(index)));
    }
    items.push((9, Item::Gap));
    items.push((1, Item::DiskRead));
    items.push((1, Item::DiskWrite));
    for index in 0..system.disks.len() {
        items.push((4, Item::DiskSpace(index)));
    }
    if system.process.pid != 0 {
        items.push((9, Item::Gap));
        items.push((3, Item::Process));
    }
    items
}

/// What the panel draws.
pub struct Resources<'a> {
    /// Panel title.
    pub title: &'a str,
    /// The machine (or the fleet).
    pub system: &'a SystemSnapshot,
    /// Models, for the per-GPU memory breakdown.
    pub models: &'a [ModelStats],
    /// Series behind the sparklines.
    pub history: &'a History,
}

/// Draws the panel.
pub fn draw(frame: &mut Frame, area: Rect, resources: &Resources<'_>, focused: bool) {
    let title = if resources.system.cpu.brand.is_empty() {
        format!(" {} ", resources.title)
    } else {
        format!(
            " {} · {} ",
            resources.title,
            resources.system.cpu.brand.trim()
        )
    };
    let block = panel(title, focused);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }

    let mut items = plan(resources.system);
    while items.len() > usize::from(inner.height) {
        let worst = items
            .iter()
            .map(|(priority, _)| *priority)
            .max()
            .unwrap_or(0);
        let Some(at) = items.iter().rposition(|(priority, _)| *priority == worst) else {
            break;
        };
        items.remove(at);
    }

    let label_width = if inner.width >= 44 { 20 } else { 17 };
    for (row, (_, item)) in items.into_iter().enumerate() {
        let at = Rect {
            y: inner.y + row as u16,
            height: 1,
            ..inner
        };
        draw_item(frame, at, item, resources, label_width);
    }
}

fn draw_item(
    frame: &mut Frame,
    area: Rect,
    item: Item,
    resources: &Resources<'_>,
    label_width: usize,
) {
    let system = resources.system;
    let history = resources.history;
    let limit = usize::from(area.width);
    match item {
        Item::Gap => {}
        Item::Cpu => {
            let cpu = &system.cpu;
            let label = format!(
                "CPU {:>6} {}c",
                format::percent(f64::from(cpu.usage_pct)),
                cpu.cores
            );
            let values = history.series(limit, |s| s.cpu_pct);
            gauge_row(
                frame,
                area,
                fit(&label, label_width),
                f64::from(cpu.usage_pct) / 100.0,
                usage_color(f64::from(cpu.usage_pct) / 100.0),
                Some(Trend {
                    values: &values,
                    max: Some(100.0),
                    color: ACCENT,
                }),
            );
        }
        Item::Cores => frame.render_widget(Paragraph::new(core_line(system, limit)), area),
        Item::Ram => {
            let memory = &system.memory;
            let label = format!(
                "RAM {}",
                format::bytes_pair(memory.used_bytes, memory.total_bytes)
            );
            let used = ratio(memory.used_bytes as f64, memory.total_bytes as f64);
            let values = history.series(limit, |s| s.ram_used);
            gauge_row(
                frame,
                area,
                fit(&label, label_width),
                used,
                usage_color(used),
                Some(Trend {
                    values: &values,
                    max: Some(memory.total_bytes as f64),
                    color: ACCENT,
                }),
            );
        }
        Item::Swap => draw_swap(frame, area, resources, label_width),
        Item::NoGpu => frame.render_widget(
            Paragraph::new(Span::styled("GPU none reported", Style::new().fg(DIM))),
            area,
        ),
        Item::GpuUtil(index) => {
            let Some(gpu) = system.gpus.get(index) else {
                return;
            };
            let util = gpu.util_pct.map(f64::from);
            let label = match util {
                Some(util) => format!("GPU{index} {:>6}", format::percent(util)),
                None => format!("GPU{index} n/a"),
            };
            let values = history.gpu_series(index, limit, |g| g.util_pct);
            let ratio = util.unwrap_or(0.0) / 100.0;
            gauge_row(
                frame,
                area,
                fit(&label, label_width),
                ratio,
                usage_color(ratio),
                Some(Trend {
                    values: &values,
                    max: Some(100.0),
                    color: ACCENT,
                }),
            );
        }
        Item::GpuVram(index) => {
            let Some(gpu) = system.gpus.get(index) else {
                return;
            };
            let total = gpu.vram_total_bytes.unwrap_or(0);
            let used = gpu.vram_used_bytes.unwrap_or(0);
            let label = match gpu.vram_used_bytes {
                Some(used) => format!("VRAM{index} {}", format::bytes_pair(used, total)),
                None => format!("VRAM{index} n/a"),
            };
            let used_ratio = ratio(used as f64, total as f64);
            let values = history.gpu_series(index, limit, |g| g.vram_used);
            gauge_row(
                frame,
                area,
                fit(&label, label_width),
                used_ratio,
                usage_color(used_ratio),
                Some(Trend {
                    values: &values,
                    max: Some(total as f64),
                    color: Color::Magenta,
                }),
            );
        }
        Item::GpuInfo(index) => {
            if let Some(gpu) = system.gpus.get(index) {
                frame.render_widget(Paragraph::new(gpu_info(gpu)), area);
            }
        }
        Item::GpuMem(index) => {
            if let Some(gpu) = system.gpus.get(index) {
                frame.render_widget(Paragraph::new(gpu_memory(gpu, resources.models)), area);
            }
        }
        Item::Paging => draw_paging(frame, area, resources, label_width),
        Item::DiskRead | Item::DiskWrite => {
            let reading = item == Item::DiskRead;
            let (name, value) = if reading {
                ("read ", system.disk_io.read_bytes_per_s)
            } else {
                ("write", system.disk_io.write_bytes_per_s)
            };
            let label = Line::from(format!("Disk {name} {}", format::rate(value)));
            let values =
                history.series(limit, |s| if reading { s.disk_read } else { s.disk_write });
            text_spark_row(
                frame,
                area,
                label,
                label_width + 4,
                &values,
                if reading { OK } else { WARN },
            );
        }
        Item::DiskSpace(index) => {
            let Some(disk) = system.disks.get(index) else {
                return;
            };
            let used = disk.total_bytes.saturating_sub(disk.available_bytes);
            let used_ratio = ratio(used as f64, disk.total_bytes as f64);
            let label = format!(
                "{} {} {} free",
                disk.mount.trim_end_matches(['\\', '/']),
                disk.kind,
                format::bytes_compact(disk.available_bytes)
            );
            let width = if area.width >= 60 {
                label_width + 8
            } else {
                label_width
            };
            gauge_row(
                frame,
                area,
                fit(&label, width),
                used_ratio,
                level(used_ratio, 0.8, 0.92),
                None,
            );
        }
        Item::Process => frame.render_widget(Paragraph::new(process_line(system)), area),
    }
}

/// `cores ▁▂▅█▁...`: one glyph per logical core, or per group of cores when
/// the row is narrower than the core count.
fn core_line(system: &SystemSnapshot, width: usize) -> Line<'static> {
    const GLYPHS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let cores = &system.cpu.per_core_pct;
    let mut spans = vec![Span::styled("cores ", Style::new().fg(DIM))];
    if cores.is_empty() {
        spans.push(Span::styled("n/a", Style::new().fg(DIM)));
        return Line::from(spans);
    }
    let room = width.saturating_sub(6).max(1);
    let group = cores.len().div_ceil(room);
    for chunk in cores.chunks(group) {
        let mean = f64::from(chunk.iter().sum::<f32>()) / chunk.len() as f64;
        let glyph = GLYPHS[((mean / 100.0 * 7.0).round().clamp(0.0, 7.0)) as usize];
        spans.push(Span::styled(
            glyph,
            Style::new().fg(usage_color(mean / 100.0)),
        ));
    }
    if let Some([one, five, fifteen]) = system.cpu.load_avg {
        let load = format!("  load {one:.2} {five:.2} {fifteen:.2}");
        if spans.len() + load.len() < width {
            spans.push(Span::styled(load, Style::new().fg(DIM)));
        }
    }
    Line::from(spans)
}

fn draw_swap(frame: &mut Frame, area: Rect, resources: &Resources<'_>, label_width: usize) {
    let memory = &resources.system.memory;
    if memory.swap_total_bytes == 0 {
        frame.render_widget(
            Paragraph::new(Span::styled("Swap none", Style::new().fg(DIM))),
            area,
        );
        return;
    }
    let label = format!(
        "Swap {}",
        format::bytes_pair(memory.swap_used_bytes, memory.swap_total_bytes)
    );
    let used = ratio(
        memory.swap_used_bytes as f64,
        memory.swap_total_bytes as f64,
    );
    // Swap in use is a cost even when nothing pages right now.
    let color = if used >= 0.5 || memory.swap_used_bytes > 0 {
        WARN
    } else {
        OK
    };
    gauge_row(frame, area, fit(&label, label_width), used, color, None);
}

/// Page-in and page-out rates; any paging is highlighted, because a model
/// whose weights page in and out of RAM is slow.
fn draw_paging(frame: &mut Frame, area: Rect, resources: &Resources<'_>, label_width: usize) {
    let memory = &resources.system.memory;
    let (Some(page_in), Some(page_out)) = (memory.swap_in_bytes_per_s, memory.swap_out_bytes_per_s)
    else {
        frame.render_widget(
            Paragraph::new(Span::styled("Paging n/a", Style::new().fg(DIM))),
            area,
        );
        return;
    };
    let paging = page_in > 0.0 || page_out > 0.0;
    let style = if paging {
        Style::new().fg(WARN).add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(DIM)
    };
    let label = Line::from(Span::styled(
        format!(
            "Paging in {} out {}",
            format::rate(page_in),
            format::rate(page_out)
        ),
        style,
    ));
    let values = resources
        .history
        .series(usize::from(area.width), |s| s.swap_in + s.swap_out);
    text_spark_row(frame, area, label, label_width.max(32), &values, WARN);
}

/// A label of up to `label_width` cells, then a sparkline over the rest of
/// the row.
fn text_spark_row(
    frame: &mut Frame,
    area: Rect,
    label: Line<'static>,
    label_width: usize,
    values: &[f64],
    color: Color,
) {
    let label_width = label_width.min(usize::from(area.width / 2)) as u16;
    frame.render_widget(
        Paragraph::new(label),
        Rect {
            width: label_width,
            ..area
        },
    );
    let at = Rect {
        x: area.x + label_width + 1,
        width: area.width.saturating_sub(label_width + 1),
        ..area
    };
    spark(frame, at, values, None, color);
}

/// `└ RTX 4090 · 78°C · 410 W · shared 256M`: what the adapter is and how hot.
fn gpu_info(gpu: &GpuStats) -> Line<'static> {
    let dim = Style::new().fg(DIM);
    let mut spans = vec![Span::styled(
        format!("  └ {}", short_gpu_name(&gpu.name)),
        Style::new().fg(ACCENT),
    )];
    if let Some(temperature) = gpu.temperature_c {
        let color = level(f64::from(temperature), 75.0, 85.0);
        spans.push(Span::styled(" · ", dim));
        spans.push(Span::styled(
            format!("{temperature:.0}°C"),
            Style::new().fg(color),
        ));
    }
    if let Some(power) = gpu.power_w {
        spans.push(Span::styled(" · ", dim));
        spans.push(Span::raw(format!("{power:.0} W")));
    }
    if let Some(shared) = gpu.shared_vram_used_bytes {
        spans.push(Span::styled(" · ", dim));
        spans.push(Span::raw(format!(
            "shared {}",
            format::bytes_compact(shared)
        )));
    }
    Line::from(spans)
}

/// `weights 3.2G · KV 1.5G · cuca 15G · others 5.2G`: where the VRAM went.
/// Weights and K/V come from the models placed on this adapter, `cuca` is
/// everything this process holds, `others` is what other processes hold.
fn gpu_memory(gpu: &GpuStats, models: &[ModelStats]) -> Line<'static> {
    let dim = Style::new().fg(DIM);
    let mut spans = vec![Span::raw("    ")];
    let (weights, kv) = model_bytes_on_gpu(models, gpu.index);
    let mut part = |label: &str, bytes: u64| {
        if spans.len() > 1 {
            spans.push(Span::styled(" · ", dim));
        }
        spans.push(Span::styled(format!("{label} "), dim));
        spans.push(Span::raw(format::bytes_compact(bytes)));
    };
    if weights > 0 || kv > 0 {
        part("weights", weights);
        part("KV", kv);
    }
    if let Some(process) = gpu.process_vram_bytes {
        part("cuca", process);
        if let Some(used) = gpu.vram_used_bytes {
            part("others", used.saturating_sub(process));
        }
    }
    Line::from(spans)
}

fn short_gpu_name(name: &str) -> String {
    name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
}

/// Weights and K/V bytes of the models running on GPU `index`. A model
/// spread over several devices (`cuda:0+cuda:1`) counts for an even share.
fn model_bytes_on_gpu(models: &[ModelStats], index: usize) -> (u64, u64) {
    let mut weights = 0;
    let mut kv = 0;
    for model in models
        .iter()
        .filter(|model| model.state != ModelState::Failed)
    {
        let devices = gpu_indices(&model.memory.device);
        if devices.contains(&index) {
            let share = devices.len() as u64;
            weights += model.memory.weights_bytes / share;
            kv += model.memory.kv_bytes / share;
        }
    }
    (weights, kv)
}

/// GPU indices a device string names: `vulkan:1` is `[1]`, `cuda:0+cuda:1`
/// is `[0, 1]`, `cpu` is empty, a bare backend name means its first adapter.
pub fn gpu_indices(device: &str) -> Vec<usize> {
    device
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty() && *part != "cpu")
        .map(|part| {
            part.split_once(':')
                .and_then(|(_, index)| index.parse().ok())
                .unwrap_or(0)
        })
        .collect()
}

fn process_line(system: &SystemSnapshot) -> Line<'static> {
    let process = &system.process;
    let dim = Style::new().fg(DIM);
    let mut spans = vec![
        Span::styled("proc ", dim),
        Span::raw(format!("pid {}", process.pid)),
        Span::styled(" · ", dim),
        Span::raw(format!("cpu {:.0}%", process.cpu_pct)),
        Span::styled(" · ", dim),
        Span::raw(format!("rss {}", format::bytes(process.rss_bytes))),
    ];
    if let Some(threads) = process.threads {
        spans.push(Span::styled(" · ", dim));
        spans.push(Span::raw(format!("{threads} thr")));
    }
    spans.push(Span::styled(" · ", dim));
    spans.push(Span::raw(format!(
        "io R {} W {}",
        format::rate(process.read_bytes_per_s),
        format::rate(process.write_bytes_per_s)
    )));
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_strings_name_their_gpus() {
        assert_eq!(gpu_indices("cpu"), Vec::<usize>::new());
        assert_eq!(gpu_indices("vulkan:1"), [1]);
        assert_eq!(gpu_indices("cuda:0+cuda:1"), [0, 1]);
        assert_eq!(gpu_indices("metal"), [0]);
    }

    #[test]
    fn a_short_panel_drops_the_least_important_rows_first() {
        let system = SystemSnapshot {
            gpus: vec![GpuStats::default()],
            disks: vec![Default::default()],
            ..SystemSnapshot::default()
        };
        let mut items = plan(&system);
        while items.len() > 4 {
            let worst = items.iter().map(|(p, _)| *p).max().unwrap();
            let at = items.iter().rposition(|(p, _)| *p == worst).unwrap();
            items.remove(at);
        }
        let kept: Vec<Item> = items.into_iter().map(|(_, item)| item).collect();
        assert_eq!(
            kept,
            [Item::Cpu, Item::Ram, Item::GpuUtil(0), Item::GpuVram(0)]
        );
    }
}
