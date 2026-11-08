//! Routing: which upstream gets a chat request. Pure functions over plain data,
//! so the policies are unit-tested without a network.
//!
//! # Candidates
//!
//! [`resolve_model`] picks the model a request is for and the healthy
//! upstreams that serve it. With a `model` field the candidates are the
//! healthy upstreams listing that name; without one the request works only
//! when every healthy upstream serves the same single model (the model
//! server's own single-model rule), otherwise it is a 400.
//!
//! # Policies
//!
//! Each candidate has a load score (see [`load_score`]):
//! `in_flight + waiting + running / max_sequences`, where `in_flight` is what
//! the balancer has outstanding to that upstream and the rest is what the
//! upstream last reported.
//!
//! * `least-loaded`: the lowest score; ties rotate.
//! * `round-robin`: candidates in turn.
//! * `prefix-affinity` (the default): rendezvous (highest-random-weight)
//!   hashing of the model and the request's prefix key over the candidates, so
//!   a conversation keeps reaching the upstream whose prefix cache holds it,
//!   and when an upstream leaves only the keys it held move. The affinity
//!   target is passed over for the least-loaded candidate when its score
//!   exceeds `2 * min_score + 2` (min over the candidates); those
//!   fall-backs are counted. A request with no prefix key (no text) has no
//!   affinity and goes least-loaded.
//!
//! The prefix key is the canonical text of the leading system/developer
//! messages and the first user message, at most [`PREFIX_KEY_BYTES`] bytes.

use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::server::error::ServerError;
use crate::server::stats::StatsSnapshot;

/// Bytes of canonical prompt text the affinity key keeps.
pub const PREFIX_KEY_BYTES: usize = 4096;

/// How a request picks its upstream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Policy {
    /// Rendezvous hashing on the prompt prefix, falling back to least loaded.
    #[default]
    PrefixAffinity,
    /// The upstream with the lowest load score.
    LeastLoaded,
    /// Upstreams in turn.
    RoundRobin,
}

impl Policy {
    /// The name `--policy` takes.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PrefixAffinity => "prefix-affinity",
            Self::LeastLoaded => "least-loaded",
            Self::RoundRobin => "round-robin",
        }
    }
}

impl FromStr for Policy {
    type Err = ServerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "prefix-affinity" | "affinity" => Ok(Self::PrefixAffinity),
            "least-loaded" => Ok(Self::LeastLoaded),
            "round-robin" => Ok(Self::RoundRobin),
            other => Err(ServerError::Config(format!(
                "unknown policy `{other}`: expected prefix-affinity, least-loaded or round-robin"
            ))),
        }
    }
}

/// What the router needs to know about one upstream.
#[derive(Debug, Clone)]
pub struct UpstreamView<'a> {
    /// The upstream's id (`host:port`).
    pub id: &'a str,
    /// Whether it takes requests now.
    pub healthy: bool,
    /// The models it last reported.
    pub models: &'a [String],
}

/// Why no candidate was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoCandidate {
    /// The request names no model and the healthy upstreams do not all serve
    /// the same single one.
    MissingModel,
    /// No upstream, healthy or not, has reported this model.
    UnknownModel(String),
    /// The model exists but no healthy upstream serves it (or none is
    /// healthy at all).
    NoneHealthy,
}

/// The model of a request and the indexes of the healthy upstreams serving
/// it, in upstream order.
///
/// `warmed_up` says every upstream has reported its models at least once; an
/// unknown model is a 404 only then, before it a 503 (the answer may simply
/// not have arrived).
///
/// # Errors
///
/// [`NoCandidate`] when none qualifies.
pub fn resolve_model(
    requested: Option<&str>,
    upstreams: &[UpstreamView<'_>],
    warmed_up: bool,
) -> Result<(String, Vec<usize>), NoCandidate> {
    let healthy = || upstreams.iter().enumerate().filter(|(_, u)| u.healthy);
    let model = match requested {
        Some(name) => name.to_string(),
        None => {
            let mut names: Vec<&str> = healthy()
                .flat_map(|(_, u)| u.models.iter().map(String::as_str))
                .collect();
            names.sort_unstable();
            names.dedup();
            match names.as_slice() {
                [only] => (*only).to_string(),
                [] => return Err(NoCandidate::NoneHealthy),
                _ => return Err(NoCandidate::MissingModel),
            }
        }
    };
    let serves = |u: &UpstreamView<'_>| u.models.contains(&model);
    let candidates: Vec<usize> = healthy()
        .filter(|(_, u)| serves(u))
        .map(|(i, _)| i)
        .collect();
    if !candidates.is_empty() {
        return Ok((model, candidates));
    }
    if upstreams.iter().any(serves) || !warmed_up {
        return Err(NoCandidate::NoneHealthy);
    }
    if requested.is_some() {
        Err(NoCandidate::UnknownModel(model))
    } else {
        Err(NoCandidate::NoneHealthy)
    }
}

/// The load score of an upstream for `model`: the balancer's in-flight
/// requests, plus the requests the upstream reports waiting (for that model
/// when its snapshot lists it, else all), plus its running fraction of
/// `max_sequences` as a tie-breaker below one request.
pub fn load_score(in_flight: u64, snapshot: Option<&StatsSnapshot>, model: &str) -> f64 {
    let mut score = in_flight as f64;
    if let Some(snapshot) = snapshot {
        let listed = snapshot.models.iter().any(|m| m.name == model);
        let (mut waiting, mut running, mut slots) = (0u64, 0u64, 0u64);
        for m in snapshot
            .models
            .iter()
            .filter(|m| !listed || m.name == model)
        {
            waiting += m.waiting;
            running += m.running;
            slots += m.max_sequences;
        }
        score += waiting as f64;
        if slots > 0 {
            score += running as f64 / slots as f64;
        }
    }
    score
}

/// One candidate of a choice.
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
    /// Index into the balancer's upstreams.
    pub index: usize,
    /// The upstream's id, the hash identity of the node.
    pub id: &'a str,
    /// Its [`load_score`].
    pub score: f64,
}

/// How a choice was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// The prefix's rendezvous target.
    Affinity,
    /// The affinity target was too loaded; the least-loaded one was used.
    Fallback,
    /// The least-loaded candidate (policy, or no prefix key).
    LeastLoaded,
    /// The next candidate in turn.
    RoundRobin,
}

impl Decision {
    /// The label of the metric and the span field.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Affinity => "affinity",
            Self::Fallback => "fallback",
            Self::LeastLoaded => "least_loaded",
            Self::RoundRobin => "round_robin",
        }
    }
}

/// The affinity target's score above which it is passed over:
/// `2 * min_score + 2`.
pub fn fallback_limit(min_score: f64) -> f64 {
    2.0 * min_score + 2.0
}

/// Chooses among candidates; the only state is a rotation counter.
#[derive(Debug)]
pub struct Router {
    policy: Policy,
    turn: AtomicU64,
}

impl Router {
    /// A router for `policy`.
    pub fn new(policy: Policy) -> Self {
        Self {
            policy,
            turn: AtomicU64::new(0),
        }
    }

    /// The configured policy.
    pub fn policy(&self) -> Policy {
        self.policy
    }

    /// Pick one of `candidates` for `model` and the prefix `key` (`None` or
    /// empty: no affinity). `None` when there are no candidates.
    pub fn choose(
        &self,
        model: &str,
        key: Option<&str>,
        candidates: &[Candidate<'_>],
    ) -> Option<(usize, Decision)> {
        if candidates.is_empty() {
            return None;
        }
        match self.policy {
            Policy::RoundRobin => {
                let n = self.turn.fetch_add(1, Ordering::Relaxed) as usize % candidates.len();
                Some((candidates[n].index, Decision::RoundRobin))
            }
            Policy::LeastLoaded => Some((self.least_loaded(candidates), Decision::LeastLoaded)),
            Policy::PrefixAffinity => match key.filter(|k| !k.is_empty()) {
                None => Some((self.least_loaded(candidates), Decision::LeastLoaded)),
                Some(key) => {
                    let target = rendezvous(model, key, candidates)?;
                    let min = candidates
                        .iter()
                        .map(|c| c.score)
                        .fold(f64::INFINITY, f64::min);
                    if target.score > fallback_limit(min) {
                        Some((self.least_loaded(candidates), Decision::Fallback))
                    } else {
                        Some((target.index, Decision::Affinity))
                    }
                }
            },
        }
    }

    /// The lowest score; candidates within a hair of it share the turns.
    fn least_loaded(&self, candidates: &[Candidate<'_>]) -> usize {
        let min = candidates
            .iter()
            .map(|c| c.score)
            .fold(f64::INFINITY, f64::min);
        let tied: Vec<&Candidate<'_>> = candidates
            .iter()
            .filter(|c| c.score <= min + 1e-9)
            .collect();
        let n = self.turn.fetch_add(1, Ordering::Relaxed) as usize % tied.len();
        tied[n].index
    }
}

/// FNV-1a over `parts`, a zero byte between them.
fn fnv(parts: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        for &b in *part {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// SplitMix64's finalizer: spreads FNV's weak low bits.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The candidate with the highest rendezvous weight for `model` and `key`
/// (`None` without candidates). A weight depends only on the key and that
/// candidate's id, so removing a candidate moves only the keys it won.
fn rendezvous<'a>(
    model: &str,
    key: &str,
    candidates: &'a [Candidate<'a>],
) -> Option<&'a Candidate<'a>> {
    let key_hash = fnv(&[model.as_bytes(), key.as_bytes()]);
    candidates
        .iter()
        .max_by_key(|c| mix(key_hash ^ mix(fnv(&[c.id.as_bytes()]))))
}

/// The text of a message's `content` (a string, or the `text` parts joined by
/// a newline).
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// The affinity key of a parsed chat body: the leading system and developer
/// messages and the first user message, as `role\0text\0` records, cut to
/// [`PREFIX_KEY_BYTES`] at a character boundary. `None` when the request has
/// no user text.
pub fn prefix_key(messages: &[Value]) -> Option<String> {
    let mut key = String::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        let text = content_text(message.get("content"));
        if !matches!(role, "system" | "developer" | "user") {
            continue;
        }
        key.push_str(role);
        key.push('\0');
        key.push_str(&text);
        key.push('\0');
        if key.len() >= PREFIX_KEY_BYTES || role == "user" {
            break;
        }
    }
    if !key.contains("user\0") {
        return None;
    }
    let mut end = key.len().min(PREFIX_KEY_BYTES);
    while !key.is_char_boundary(end) {
        end -= 1;
    }
    key.truncate(end);
    Some(key)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::server::stats::ModelStats;

    fn cands<'a>(ids: &'a [&'a str], scores: &[f64]) -> Vec<Candidate<'a>> {
        ids.iter()
            .zip(scores)
            .enumerate()
            .map(|(index, (id, &score))| Candidate { index, id, score })
            .collect()
    }

    fn view<'a>(id: &'a str, healthy: bool, models: &'a [String]) -> UpstreamView<'a> {
        UpstreamView {
            id,
            healthy,
            models,
        }
    }

    #[test]
    fn removing_an_upstream_moves_only_the_keys_it_held() {
        let router = Router::new(Policy::PrefixAffinity);
        let ids = ["a:1", "b:1", "c:1", "d:1"];
        let all = cands(&ids, &[0.0; 4]);
        let without_c: Vec<Candidate<'_>> = all.iter().copied().filter(|c| c.id != "c:1").collect();
        let (mut held_by_c, mut counts) = (0, [0usize; 4]);
        for i in 0..2000 {
            let key = format!("user\0conversation {i}\0");
            let (before, _) = router.choose("m", Some(&key), &all).unwrap();
            let (after, _) = router.choose("m", Some(&key), &without_c).unwrap();
            counts[before] += 1;
            if ids[before] == "c:1" {
                held_by_c += 1;
                assert_ne!(ids[after], "c:1");
            } else {
                assert_eq!(
                    before, after,
                    "a key not held by the removed upstream moved"
                );
            }
        }
        assert!(held_by_c > 0);
        assert!(
            counts.iter().all(|&n| n > 2000 / 4 * 7 / 10),
            "keys spread roughly evenly: {counts:?}"
        );
    }

    #[test]
    fn the_same_key_and_model_always_pick_the_same_upstream() {
        let router = Router::new(Policy::PrefixAffinity);
        let ids = ["a:1", "b:1", "c:1"];
        let c = cands(&ids, &[0.0; 3]);
        let first = router.choose("m", Some("k"), &c).unwrap();
        for _ in 0..20 {
            assert_eq!(router.choose("m", Some("k"), &c).unwrap(), first);
        }
        assert_eq!(first.1, Decision::Affinity);
    }

    #[test]
    fn an_overloaded_affinity_target_falls_back_to_the_least_loaded() {
        let router = Router::new(Policy::PrefixAffinity);
        let ids = ["a:1", "b:1"];
        let idle = cands(&ids, &[0.0, 0.0]);
        let (target, _) = router.choose("m", Some("k"), &idle).unwrap();
        let other = 1 - target;
        // min score 1: the limit is 4; a score of exactly 4 still holds.
        let mut scores = [0.0; 2];
        scores[other] = 1.0;
        scores[target] = 4.0;
        let at_limit = router
            .choose("m", Some("k"), &cands(&ids, &scores))
            .unwrap();
        assert_eq!(at_limit, (target, Decision::Affinity));
        scores[target] = 4.5;
        let over = router
            .choose("m", Some("k"), &cands(&ids, &scores))
            .unwrap();
        assert_eq!(over, (other, Decision::Fallback));
        assert_eq!(fallback_limit(1.0), 4.0);
    }

    #[test]
    fn a_request_without_a_prefix_key_goes_least_loaded() {
        let router = Router::new(Policy::PrefixAffinity);
        let ids = ["a:1", "b:1"];
        let c = cands(&ids, &[3.0, 1.0]);
        assert_eq!(
            router.choose("m", None, &c),
            Some((1, Decision::LeastLoaded))
        );
        assert_eq!(
            router.choose("m", Some(""), &c),
            Some((1, Decision::LeastLoaded))
        );
    }

    #[test]
    fn least_loaded_picks_the_minimum_and_rotates_between_ties() {
        let router = Router::new(Policy::LeastLoaded);
        let ids = ["a:1", "b:1", "c:1"];
        assert_eq!(
            router
                .choose("m", None, &cands(&ids, &[2.0, 0.5, 1.0]))
                .unwrap()
                .0,
            1
        );
        let tied = cands(&ids, &[1.0, 1.0, 5.0]);
        let picks: Vec<usize> = (0..4)
            .map(|_| router.choose("m", None, &tied).unwrap().0)
            .collect();
        assert!(
            picks.contains(&0) && picks.contains(&1) && !picks.contains(&2),
            "{picks:?}"
        );
    }

    #[test]
    fn round_robin_visits_every_candidate_in_turn() {
        let router = Router::new(Policy::RoundRobin);
        let ids = ["a:1", "b:1", "c:1"];
        let c = cands(&ids, &[9.0, 0.0, 0.0]);
        let picks: Vec<usize> = (0..6)
            .map(|_| router.choose("m", None, &c).unwrap().0)
            .collect();
        assert_eq!(picks, [0, 1, 2, 0, 1, 2]);
        assert_eq!(router.choose("m", None, &[]), None);
    }

    #[test]
    fn candidates_are_the_healthy_upstreams_serving_the_model() {
        let (a, b) = (
            vec!["m1".to_string()],
            vec!["m1".to_string(), "m2".to_string()],
        );
        let up = [
            view("a", true, &a),
            view("b", true, &b),
            view("c", false, &a),
        ];
        assert_eq!(
            resolve_model(Some("m1"), &up, true).unwrap(),
            ("m1".into(), vec![0, 1])
        );
        assert_eq!(
            resolve_model(Some("m2"), &up, true).unwrap(),
            ("m2".into(), vec![1])
        );
        // Known only on an unhealthy upstream: none healthy, not a 404.
        let up = [view("a", false, &a), view("b", true, &b)];
        assert_eq!(
            resolve_model(Some("m1"), &up, true),
            Ok(("m1".into(), vec![1]))
        );
        let only_down = [view("a", false, &a), view("b", true, &[])];
        assert_eq!(
            resolve_model(Some("m1"), &only_down, true),
            Err(NoCandidate::NoneHealthy)
        );
        assert_eq!(
            resolve_model(Some("zzz"), &up, true),
            Err(NoCandidate::UnknownModel("zzz".into()))
        );
        // Before every upstream reported, an unknown name may just be early.
        assert_eq!(
            resolve_model(Some("zzz"), &up, false),
            Err(NoCandidate::NoneHealthy)
        );
    }

    #[test]
    fn a_missing_model_works_only_when_the_healthy_upstreams_serve_one_model() {
        let (a, b) = (vec!["m1".to_string()], vec!["m2".to_string()]);
        let same = [view("a", true, &a), view("b", true, &a)];
        assert_eq!(
            resolve_model(None, &same, true).unwrap(),
            ("m1".into(), vec![0, 1])
        );
        let differ = [view("a", true, &a), view("b", true, &b)];
        assert_eq!(
            resolve_model(None, &differ, true),
            Err(NoCandidate::MissingModel)
        );
        // An unhealthy upstream's models do not count.
        let one_down = [view("a", true, &a), view("b", false, &b)];
        assert_eq!(
            resolve_model(None, &one_down, true).unwrap(),
            ("m1".into(), vec![0])
        );
        assert_eq!(
            resolve_model(None, &[], true),
            Err(NoCandidate::NoneHealthy)
        );
    }

    #[test]
    fn the_load_score_adds_in_flight_waiting_and_the_running_fraction() {
        let model = |name: &str, running, waiting, max| ModelStats {
            name: name.into(),
            running,
            waiting,
            max_sequences: max,
            ..ModelStats::default()
        };
        let snap = StatsSnapshot {
            models: vec![model("m", 2, 3, 4), model("other", 4, 9, 4)],
            ..StatsSnapshot::default()
        };
        assert_eq!(load_score(5, None, "m"), 5.0);
        assert_eq!(load_score(1, Some(&snap), "m"), 1.0 + 3.0 + 0.5);
        // A model the snapshot does not list sums every model.
        assert_eq!(load_score(0, Some(&snap), "x"), 12.0 + 6.0 / 8.0);
    }

    #[test]
    fn the_prefix_key_is_the_system_text_and_the_first_user_message() {
        let msgs = |v: Value| v.as_array().unwrap().clone();
        let first = msgs(json!([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "second"},
        ]));
        let later_turn = msgs(json!([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "a\nb"},
            {"role": "assistant", "content": "different answer"},
            {"role": "user", "content": "other follow-up"},
        ]));
        assert_eq!(
            prefix_key(&first),
            prefix_key(&later_turn),
            "a conversation keeps its key"
        );
        assert_eq!(
            prefix_key(&first).unwrap(),
            "system\0Be brief.\0user\0a\nb\0"
        );
        assert_eq!(
            prefix_key(&msgs(json!([{"role": "assistant", "content": "x"}]))),
            None
        );
        assert_ne!(
            prefix_key(&msgs(json!([{"role": "user", "content": "one"}]))),
            prefix_key(&msgs(json!([{"role": "user", "content": "two"}])))
        );
    }

    #[test]
    fn a_long_prompt_is_cut_at_a_character_boundary() {
        let long = "é".repeat(5000);
        let key = prefix_key(&[json!({"role": "user", "content": long})]).unwrap();
        assert!(key.len() <= PREFIX_KEY_BYTES);
        assert!(key.is_char_boundary(key.len()));
        assert!(key.len() > PREFIX_KEY_BYTES - 4);
    }

    #[test]
    fn policies_parse_by_name() {
        assert_eq!(
            "prefix-affinity".parse::<Policy>().unwrap(),
            Policy::PrefixAffinity
        );
        assert_eq!(
            "Least-Loaded".parse::<Policy>().unwrap(),
            Policy::LeastLoaded
        );
        assert!("random".parse::<Policy>().is_err());
    }
}
