//! Human formatting of sizes, rates and durations.
//!
//! Pure functions, no allocation beyond the returned `String`. A non-finite or
//! negative input prints as zero, so a half-filled snapshot never shows `NaN`.

const KIB: f64 = 1024.0;
const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];

fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Bytes as `512 B`, `1.5 KiB`, `12.3 MiB`, `1.25 GiB`.
pub fn bytes(value: u64) -> String {
    // A u64 above 2^53 loses low bits in f64; irrelevant at one decimal.
    bytes_f(value as f64)
}

/// [`bytes()`] for a fractional size.
pub fn bytes_f(value: f64) -> String {
    let value = finite(value);
    if value < KIB {
        return format!("{value:.0} B");
    }
    let mut scaled = value;
    let mut unit = 0;
    while scaled >= KIB && unit < UNITS.len() - 1 {
        scaled /= KIB;
        unit += 1;
    }
    if scaled < 10.0 {
        format!("{scaled:.2} {}", UNITS[unit])
    } else {
        format!("{scaled:.1} {}", UNITS[unit])
    }
}

/// `used/total` in the unit of `total`: `58.0/64.0 GiB`.
pub fn bytes_pair(used: u64, total: u64) -> String {
    let mut unit = 0;
    let mut scale = 1.0;
    while total as f64 / scale >= KIB && unit < UNITS.len() - 1 {
        scale *= KIB;
        unit += 1;
    }
    if unit == 0 {
        return format!("{used}/{total} B");
    }
    format!(
        "{:.1}/{:.1} {}",
        used as f64 / scale,
        total as f64 / scale,
        UNITS[unit]
    )
}

/// Bytes in a few characters: `100B`, `256M`, `3.0G`.
pub fn bytes_compact(value: u64) -> String {
    let value = value as f64;
    if value < KIB {
        format!("{value:.0}B")
    } else if value < KIB * KIB {
        format!("{:.0}K", value / KIB)
    } else if value < KIB * KIB * KIB {
        format!("{:.0}M", value / (KIB * KIB))
    } else if value < KIB * KIB * KIB * KIB {
        format!("{:.1}G", value / (KIB * KIB * KIB))
    } else {
        format!("{:.1}T", value / (KIB * KIB * KIB * KIB))
    }
}

/// A byte rate: `3.4 MiB/s`.
pub fn rate(bytes_per_s: f64) -> String {
    format!("{}/s", bytes_f(bytes_per_s))
}

/// A latency given in seconds: `400 us`, `4.2 ms`, `120 ms`, `1.25 s`, `2m05s`.
pub fn seconds(value: f64) -> String {
    let value = finite(value);
    if value <= 0.0 {
        "0 ms".to_string()
    } else if value < 0.001 {
        format!("{:.0} us", value * 1e6)
    } else if value < 0.01 {
        format!("{:.1} ms", value * 1e3)
    } else if value < 1.0 {
        format!("{:.0} ms", value * 1e3)
    } else if value < 60.0 {
        format!("{value:.2} s")
    } else {
        let total = value.round() as u64;
        format!("{}m{:02}s", total / 60, total % 60)
    }
}

/// Milliseconds as [`seconds`] does.
pub fn millis(value: f64) -> String {
    seconds(finite(value) / 1e3)
}

/// An uptime: `9s`, `4m 12s`, `2h 05m`, `3d 4h`.
pub fn uptime(value: f64) -> String {
    let total = finite(value) as u64;
    let (days, hours, minutes, secs) = (
        total / 86_400,
        total % 86_400 / 3_600,
        total % 3_600 / 60,
        total % 60,
    );
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m {secs:02}s")
    } else {
        format!("{secs}s")
    }
}

/// A percentage given as `0..=100`: `37.2%`.
pub fn percent(value: f64) -> String {
    format!("{:.1}%", finite(value))
}

/// A ratio `0..=1` as a whole percentage: `73%`. `None` prints `-`.
pub fn ratio_percent(value: Option<f64>) -> String {
    value.map_or_else(
        || "-".to_string(),
        |ratio| format!("{:.0}%", finite(ratio) * 100.0),
    )
}

/// A counter, compact above a thousand: `999`, `1.2k`, `3.4M`.
pub fn count(value: u64) -> String {
    match value {
        0..1_000 => value.to_string(),
        1_000..1_000_000 => format!("{:.1}k", value as f64 / 1e3),
        1_000_000..1_000_000_000 => format!("{:.1}M", value as f64 / 1e6),
        _ => format!("{:.1}G", value as f64 / 1e9),
    }
}

/// A per-second figure with a precision that follows its size:
/// `0.42`, `12.3`, `1234`.
pub fn per_second(value: f64) -> String {
    let value = finite(value);
    if value >= 1_000.0 {
        format!("{value:.0}")
    } else if value >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

/// A free-form number (a diagnosis evidence value): integers print plain,
/// others with at most three decimals and no trailing zeros.
pub fn number(value: f64) -> String {
    if !value.is_finite() {
        return "-".to_string();
    }
    if value.fract() == 0.0 && value.abs() < 1e15 {
        return format!("{value:.0}");
    }
    let text = if value.abs() >= 100.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.3}")
    };
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Cuts `text` to at most `max` characters, ending in `…` when it was cut.
pub fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let keep = max.saturating_sub(1);
    let mut cut: String = text.chars().take(keep).collect();
    if max > 0 {
        cut.push('…');
    }
    cut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_pick_the_binary_unit() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(1023), "1023 B");
        assert_eq!(bytes(1536), "1.50 KiB");
        assert_eq!(bytes(12 * 1024 * 1024 + 300_000), "12.3 MiB");
        assert_eq!(bytes(24 << 30), "24.0 GiB");
        assert_eq!(bytes(3 << 50), "3.00 PiB");
        assert_eq!(bytes_pair(58 << 30, 64 << 30), "58.0/64.0 GiB");
        assert_eq!(bytes_pair(512 << 20, 24 << 30), "0.5/24.0 GiB");
        assert_eq!(bytes_pair(0, 0), "0/0 B");
        assert_eq!(bytes_compact(3 << 30), "3.0G");
        assert_eq!(bytes_compact(256 << 20), "256M");
        assert_eq!(bytes_compact(100), "100B");
    }

    #[test]
    fn rates_append_per_second() {
        assert_eq!(rate(3.5 * 1024.0 * 1024.0), "3.50 MiB/s");
        assert_eq!(rate(f64::NAN), "0 B/s");
        assert_eq!(rate(-5.0), "0 B/s");
    }

    #[test]
    fn latencies_scale_from_microseconds_to_minutes() {
        assert_eq!(seconds(0.0), "0 ms");
        assert_eq!(seconds(0.0004), "400 us");
        assert_eq!(seconds(0.0042), "4.2 ms");
        assert_eq!(seconds(0.12), "120 ms");
        assert_eq!(seconds(1.25), "1.25 s");
        assert_eq!(seconds(125.0), "2m05s");
        assert_eq!(millis(250.0), "250 ms");
    }

    #[test]
    fn uptimes_keep_two_units() {
        assert_eq!(uptime(9.4), "9s");
        assert_eq!(uptime(252.0), "4m 12s");
        assert_eq!(uptime(7_500.0), "2h 05m");
        assert_eq!(uptime(3.0 * 86_400.0 + 4.0 * 3_600.0), "3d 4h");
    }

    #[test]
    fn counts_and_ratios_are_compact() {
        assert_eq!(count(999), "999");
        assert_eq!(count(1_234), "1.2k");
        assert_eq!(count(3_400_000), "3.4M");
        assert_eq!(ratio_percent(Some(0.734)), "73%");
        assert_eq!(ratio_percent(None), "-");
        assert_eq!(percent(37.24), "37.2%");
    }

    #[test]
    fn evidence_numbers_drop_trailing_zeros() {
        assert_eq!(number(3.0), "3");
        assert_eq!(number(0.97), "0.97");
        assert_eq!(number(0.123_456), "0.123");
        assert_eq!(number(1234.56), "1234.6");
        assert_eq!(number(f64::INFINITY), "-");
    }

    #[test]
    fn truncate_ends_in_an_ellipsis_only_when_it_cut() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("abcdefghij", 5), "abcd…");
        assert_eq!(truncate("abc", 0), "");
    }
}
