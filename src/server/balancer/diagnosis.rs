//! Balancer diagnosis: reasons the traffic through the balancer is slow or at
//! risk, from what it measures about its upstreams. Like the model server's
//! rules (`server::diagnosis`) each rule is a function over plain signals
//! with thresholds from [`Thresholds`], and every entry carries its
//! evidence. The balancer's snapshot lists these first, then each upstream's
//! own diagnoses re-labelled with the upstream ([`relabel`]).

use std::collections::BTreeMap;

use crate::server::stats::{Cause, Diagnosis, Severity};

/// What the rules know about one upstream.
#[derive(Debug, Clone, Default)]
pub struct UpstreamSignal {
    /// The upstream's id.
    pub id: String,
    /// Whether it takes requests.
    pub healthy: bool,
    /// The last failure.
    pub error: Option<String>,
    /// How long it has been failing.
    pub down_for_s: Option<f64>,
    /// Its routing load score.
    pub score: f64,
    /// Whether its stats snapshot is available (the next three need it).
    pub has_stats: bool,
    /// Requests it reports waiting.
    pub waiting: u64,
    /// Requests it reports running.
    pub running: u64,
    /// Its sequence slots.
    pub max_sequences: u64,
    /// Balancer-measured 95th percentile of the time to the first byte.
    pub ttft_p95_s: f64,
    /// Requests that figure comes from.
    pub ttft_count: u64,
}

/// The limits the rules compare against.
#[derive(Debug, Clone, PartialEq)]
pub struct Thresholds {
    /// An upstream's score above `ratio * min_score + floor` is an imbalance.
    pub imbalance: (f64, f64),
    /// An upstream's p95 above `ratio` times the others' mean, and above
    /// the absolute floor in seconds, is slow; the requests each needs.
    pub slow: (f64, f64, u64),
    /// Waiting requests per slot that make saturation critical.
    pub saturated_critical_per_slot: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            imbalance: (3.0, 4.0),
            slow: (3.0, 0.5, 5),
            saturated_critical_per_slot: 2.0,
        }
    }
}

fn round(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn entry(
    severity: Severity,
    cause: Cause,
    upstream: Option<&str>,
    message: String,
    evidence: &[(&str, f64)],
) -> Diagnosis {
    Diagnosis {
        severity,
        cause,
        model: None,
        upstream: upstream.map(str::to_string),
        message,
        evidence: evidence
            .iter()
            .map(|&(k, v)| (k.to_string(), round(v)))
            .collect::<BTreeMap<_, _>>(),
    }
}

/// The balancer's own diagnoses, most severe first. `affinity` says the
/// policy is prefix affinity, which pins conversations to upstreams.
pub fn diagnose(signals: &[UpstreamSignal], affinity: bool, th: &Thresholds) -> Vec<Diagnosis> {
    let mut out = Vec::new();
    let healthy: Vec<&UpstreamSignal> = signals.iter().filter(|s| s.healthy).collect();

    if !signals.is_empty() && healthy.is_empty() {
        out.push(entry(
            Severity::Critical,
            Cause::NoHealthyUpstream,
            None,
            format!(
                "none of the {} upstream(s) is healthy: every request gets a 503",
                signals.len()
            ),
            &[("upstreams", signals.len() as f64), ("healthy", 0.0)],
        ));
    }

    for s in signals.iter().filter(|s| !s.healthy) {
        out.push(entry(
            if healthy.is_empty() {
                Severity::Critical
            } else {
                Severity::Warn
            },
            Cause::UpstreamDown,
            Some(&s.id),
            format!(
                "upstream {} is down{}: its traffic goes to the others",
                s.id,
                s.error
                    .as_deref()
                    .map(|e| format!(" ({e})"))
                    .unwrap_or_default()
            ),
            &[("down_for_s", s.down_for_s.unwrap_or(0.0))],
        ));
    }

    let with_stats: Vec<&&UpstreamSignal> = healthy.iter().filter(|s| s.has_stats).collect();
    if !healthy.is_empty()
        && with_stats.len() == healthy.len()
        && with_stats
            .iter()
            .all(|s| s.waiting > 0 && s.max_sequences > 0 && s.running >= s.max_sequences)
    {
        let waiting: u64 = with_stats.iter().map(|s| s.waiting).sum();
        let slots: u64 = with_stats.iter().map(|s| s.max_sequences).sum();
        let critical = waiting as f64 >= th.saturated_critical_per_slot * slots as f64;
        out.push(entry(
            if critical { Severity::Critical } else { Severity::Warn },
            Cause::UpstreamSaturated,
            None,
            format!(
                "every healthy upstream is full: {waiting} request(s) wait for one of {slots} sequence slots; add upstreams or raise `seqs`"
            ),
            &[
                ("waiting", waiting as f64),
                ("slots", slots as f64),
                ("healthy", healthy.len() as f64),
            ],
        ));
    }

    if affinity && healthy.len() >= 2 {
        let min = healthy
            .iter()
            .map(|s| s.score)
            .fold(f64::INFINITY, f64::min);
        if let Some(top) = healthy.iter().max_by(|a, b| a.score.total_cmp(&b.score))
            && top.score > th.imbalance.0 * min + th.imbalance.1
        {
            out.push(entry(
                Severity::Warn,
                Cause::LoadImbalance,
                Some(&top.id),
                format!(
                    "upstream {} carries a load of {:.1} while the lightest has {:.1}: prefix affinity keeps pinning conversations to it",
                    top.id, top.score, min
                ),
                &[("score", top.score), ("min_score", min)],
            ));
        }
    }

    let (ratio, floor, min_count) = th.slow;
    let measured: Vec<&&UpstreamSignal> = healthy
        .iter()
        .filter(|s| s.ttft_count >= min_count)
        .collect();
    if measured.len() >= 2 {
        for s in &measured {
            let others: Vec<f64> = measured
                .iter()
                .filter(|o| o.id != s.id)
                .map(|o| o.ttft_p95_s)
                .collect();
            let mean = others.iter().sum::<f64>() / others.len() as f64;
            if s.ttft_p95_s > floor && s.ttft_p95_s > ratio * mean {
                out.push(entry(
                    Severity::Warn,
                    Cause::UpstreamSlow,
                    Some(&s.id),
                    format!(
                        "upstream {} answers slowly: p95 time to first byte {:.2} s against {:.2} s on the others",
                        s.id, s.ttft_p95_s, mean
                    ),
                    &[("ttft_p95_s", s.ttft_p95_s), ("others_ttft_p95_s", mean)],
                ));
            }
        }
    }

    out.sort_by_key(|d| std::cmp::Reverse(d.severity));
    out
}

/// An upstream's own diagnoses as the balancer reports them: tagged with the
/// upstream and prefixed in the message.
pub fn relabel(upstream: &str, diagnoses: &[Diagnosis]) -> Vec<Diagnosis> {
    diagnoses
        .iter()
        .map(|d| Diagnosis {
            upstream: Some(upstream.to_string()),
            message: format!("[{upstream}] {}", d.message),
            ..d.clone()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn up(id: &str) -> UpstreamSignal {
        UpstreamSignal {
            id: id.into(),
            healthy: true,
            has_stats: true,
            max_sequences: 4,
            running: 1,
            ..UpstreamSignal::default()
        }
    }

    fn causes(d: &[Diagnosis]) -> Vec<Cause> {
        d.iter().map(|d| d.cause).collect()
    }

    fn run(signals: &[UpstreamSignal]) -> Vec<Diagnosis> {
        diagnose(signals, true, &Thresholds::default())
    }

    #[test]
    fn healthy_balanced_upstreams_have_no_diagnosis() {
        assert!(run(&[up("a"), up("b")]).is_empty());
    }

    #[test]
    fn no_healthy_upstream_is_critical_and_each_down_one_is_listed() {
        let mut a = up("a");
        a.healthy = false;
        a.error = Some("connection refused".into());
        a.down_for_s = Some(12.0);
        let mut b = up("b");
        b.healthy = false;
        let d = run(&[a, b]);
        assert_eq!(
            causes(&d),
            [
                Cause::NoHealthyUpstream,
                Cause::UpstreamDown,
                Cause::UpstreamDown
            ]
        );
        assert!(d.iter().all(|d| d.severity == Severity::Critical));
        assert_eq!(d[1].upstream.as_deref(), Some("a"));
        assert!(d[1].message.contains("connection refused"));
        assert_eq!(d[1].evidence["down_for_s"], 12.0);
    }

    #[test]
    fn one_down_upstream_among_healthy_ones_is_a_warning() {
        let mut b = up("b");
        b.healthy = false;
        let d = run(&[up("a"), b]);
        assert_eq!(causes(&d), [Cause::UpstreamDown]);
        assert_eq!(d[0].severity, Severity::Warn);
    }

    #[test]
    fn saturation_needs_every_healthy_upstream_full_and_queueing() {
        let full = |id: &str, waiting| UpstreamSignal {
            running: 4,
            waiting,
            ..up(id)
        };
        let d = run(&[full("a", 2), full("b", 1)]);
        assert_eq!(causes(&d), [Cause::UpstreamSaturated]);
        assert_eq!(
            (
                d[0].severity,
                d[0].evidence["waiting"],
                d[0].evidence["slots"]
            ),
            (Severity::Warn, 3.0, 8.0)
        );
        // 16 waiting over 8 slots is critical.
        assert_eq!(
            run(&[full("a", 8), full("b", 8)])[0].severity,
            Severity::Critical
        );
        // One upstream with room: no saturation.
        assert!(
            run(&[full("a", 5), up("b")])
                .iter()
                .all(|d| d.cause != Cause::UpstreamSaturated)
        );
        // Without stats the balancer cannot tell.
        let mut blind = full("a", 5);
        blind.has_stats = false;
        assert!(run(&[blind]).is_empty());
    }

    #[test]
    fn imbalance_fires_only_under_affinity_and_beyond_the_threshold() {
        let scored = |id: &str, score| UpstreamSignal { score, ..up(id) };
        // 3 * 1 + 4 = 7: a score of 7 holds, 8 fires.
        assert!(run(&[scored("a", 7.0), scored("b", 1.0)]).is_empty());
        let d = run(&[scored("a", 8.0), scored("b", 1.0)]);
        assert_eq!(causes(&d), [Cause::LoadImbalance]);
        assert_eq!(d[0].upstream.as_deref(), Some("a"));
        assert_eq!(d[0].evidence["min_score"], 1.0);
        // Round robin or least loaded do not pin traffic.
        assert!(
            diagnose(
                &[scored("a", 8.0), scored("b", 1.0)],
                false,
                &Thresholds::default()
            )
            .is_empty()
        );
    }

    #[test]
    fn a_slow_upstream_needs_enough_requests_and_a_big_gap() {
        let timed = |id: &str, p95, count| UpstreamSignal {
            ttft_p95_s: p95,
            ttft_count: count,
            ..up(id)
        };
        let d = run(&[timed("a", 2.0, 10), timed("b", 0.2, 10)]);
        assert_eq!(causes(&d), [Cause::UpstreamSlow]);
        assert_eq!(d[0].upstream.as_deref(), Some("a"));
        // Too few requests on one side: no comparison.
        assert!(run(&[timed("a", 2.0, 10), timed("b", 0.2, 2)]).is_empty());
        // Slow in ratio but under the absolute floor.
        assert!(run(&[timed("a", 0.3, 10), timed("b", 0.05, 10)]).is_empty());
    }

    #[test]
    fn an_upstreams_own_diagnoses_are_relabelled_with_its_id() {
        let own = Diagnosis {
            cause: Cause::KvPoolFull,
            message: "pool is full".into(),
            ..Diagnosis::default()
        };
        let r = relabel("a:1", &[own]);
        assert_eq!(r[0].upstream.as_deref(), Some("a:1"));
        assert_eq!(r[0].message, "[a:1] pool is full");
        assert_eq!(r[0].cause, Cause::KvPoolFull);
    }
}
