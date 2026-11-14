//! Balancer configuration: the `balance` section of the configuration file
//! ([`crate::server::CucaConfig`]).
//!
//! ```kdl
//! balance {
//!     listen "127.0.0.1:8080"
//!     upstreams "http://10.0.0.1:8000" "http://10.0.0.2:8000"
//!     upstream_api_key "${API_KEY}"
//!     api_key "${CLIENT_KEY}"
//!     policy "prefix-affinity"
//!     health_interval "1s"
//!     latency_window "60s"
//!     stats_api #true
//! }
//! ```

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use serde::{Deserialize, Deserializer};

use super::routing::Policy;
use crate::server::error::ServerError;

/// Most upstreams one balancer fronts.
pub const MAX_UPSTREAMS: usize = 256;

/// Everything `balance` needs to start: the `balance` section.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BalancerConfig {
    /// Address of the HTTP listener ([`BalancerConfig::DEFAULT_LISTEN`]).
    pub listen: SocketAddr,
    /// Base URLs of the model servers, at least one (several arguments of
    /// one `upstreams` node).
    #[serde(deserialize_with = "crate::config::one_or_many")]
    pub upstreams: Vec<String>,
    /// Sent to every upstream as `Authorization: Bearer <key>`.
    pub upstream_api_key: Option<String>,
    /// Required from clients on every route but `/health`.
    pub api_key: Option<String>,
    /// Serve the stats API.
    pub stats_api: bool,
    /// How requests pick an upstream: `prefix-affinity` (the default),
    /// `least-loaded` or `round-robin`.
    pub policy: Policy,
    /// The interval between polls of each upstream, 100 ms to 60 s (`"1s"`).
    #[serde(deserialize_with = "crate::config::duration")]
    pub health_interval: Duration,
    /// The sliding window of the latency figures (`"60s"`).
    #[serde(deserialize_with = "crate::config::duration")]
    pub latency_window: Duration,
}

impl<'de> Deserialize<'de> for Policy {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        crate::server::config::parsed(deserializer)
    }
}

impl BalancerConfig {
    /// The default listen address.
    pub const DEFAULT_LISTEN: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8000);
    /// The default health interval.
    pub const DEFAULT_HEALTH_INTERVAL: Duration = Duration::from_millis(1000);
    /// The default latency window.
    pub const DEFAULT_LATENCY_WINDOW: Duration = Duration::from_secs(60);

    /// A configuration for `upstreams` listening on `listen`, the rest
    /// default.
    pub fn new(listen: SocketAddr, upstreams: Vec<String>) -> Self {
        Self {
            listen,
            upstreams,
            upstream_api_key: None,
            api_key: None,
            stats_api: false,
            policy: Policy::default(),
            health_interval: Self::DEFAULT_HEALTH_INTERVAL,
            latency_window: Self::DEFAULT_LATENCY_WINDOW,
        }
    }

    /// Validate and normalize: the upstreams as `(id, base URL)` pairs.
    ///
    /// # Errors
    ///
    /// [`ServerError::Config`] for no upstream, more than [`MAX_UPSTREAMS`],
    /// a URL that is not `http(s)://host[:port][/path]`, a duplicate, an empty
    /// key, or a health interval outside 100 ms to 60 s.
    pub fn resolve_upstreams(&self) -> Result<Vec<(String, String)>, ServerError> {
        if self.upstreams.is_empty() {
            return Err(ServerError::Config(
                "no upstream: list at least one URL in `upstreams`".into(),
            ));
        }
        if self.upstreams.len() > MAX_UPSTREAMS {
            return Err(ServerError::Config(format!(
                "{} upstreams; at most {MAX_UPSTREAMS}",
                self.upstreams.len()
            )));
        }
        if !(Duration::from_millis(100)..=Duration::from_secs(60)).contains(&self.health_interval) {
            return Err(ServerError::Config(
                "health_interval must be between 100ms and 60s".into(),
            ));
        }
        if self.api_key.as_deref() == Some("") || self.upstream_api_key.as_deref() == Some("") {
            return Err(ServerError::Config("an API key is empty".into()));
        }
        let mut out: Vec<(String, String)> = Vec::with_capacity(self.upstreams.len());
        for raw in &self.upstreams {
            let (id, url) = normalize_upstream(raw)?;
            if out.iter().any(|(other, _)| *other == id) {
                return Err(ServerError::Config(format!(
                    "upstream `{id}` is listed twice"
                )));
            }
            out.push((id, url));
        }
        Ok(out)
    }
}

impl Default for BalancerConfig {
    fn default() -> Self {
        Self::new(Self::DEFAULT_LISTEN, Vec::new())
    }
}

/// `(host:port, base URL without a trailing slash)` of an upstream URL.
fn normalize_upstream(raw: &str) -> Result<(String, String), ServerError> {
    let bad = |why: &str| ServerError::Config(format!("upstream `{raw}`: {why}"));
    let url = raw.trim().trim_end_matches('/');
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .ok_or_else(|| bad("expected http://host:port or https://host:port"))?;
    let authority = rest.split('/').next().unwrap_or("");
    if authority.is_empty() || url.chars().any(char::is_whitespace) || authority.contains('@') {
        return Err(bad(
            "expected http(s)://host[:port][/path] with no spaces or credentials",
        ));
    }
    Ok((authority.to_string(), url.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::CucaConfig;

    fn config(upstreams: &[&str]) -> BalancerConfig {
        BalancerConfig::new(
            BalancerConfig::DEFAULT_LISTEN,
            upstreams.iter().map(|s| (*s).to_string()).collect(),
        )
    }

    fn balance(text: &str) -> Result<BalancerConfig, ServerError> {
        CucaConfig::from_kdl_str(text)?.into_balance()
    }

    #[test]
    fn upstream_urls_normalize_to_an_id_and_a_base_without_a_trailing_slash() {
        let r = config(&["http://10.0.0.1:8000/", "https://gpu.example/v1prefix/"])
            .resolve_upstreams()
            .unwrap();
        assert_eq!(
            r[0],
            ("10.0.0.1:8000".into(), "http://10.0.0.1:8000".into())
        );
        assert_eq!(
            r[1],
            ("gpu.example".into(), "https://gpu.example/v1prefix".into())
        );
    }

    #[test]
    fn bad_upstream_lists_are_rejected() {
        assert!(config(&[]).resolve_upstreams().is_err());
        assert!(config(&["10.0.0.1:8000"]).resolve_upstreams().is_err());
        assert!(config(&["http://"]).resolve_upstreams().is_err());
        assert!(config(&["http://user:pw@h:1"]).resolve_upstreams().is_err());
        let dup = config(&["http://a:1", "http://a:1/"])
            .resolve_upstreams()
            .unwrap_err();
        assert!(dup.to_string().contains("a:1"), "{dup}");
        let many: Vec<String> = (0..=MAX_UPSTREAMS)
            .map(|i| format!("http://h{i}:1"))
            .collect();
        assert!(
            BalancerConfig::new("127.0.0.1:1".parse().unwrap(), many)
                .resolve_upstreams()
                .is_err()
        );
        let mut fast = config(&["http://a:1"]);
        fast.health_interval = Duration::from_millis(10);
        assert!(fast.resolve_upstreams().is_err());
    }

    #[test]
    fn an_unknown_key_or_policy_is_rejected() {
        let one = balance(r#"balance { upstreams "http://a:1"; policy "least-loaded"; }"#).unwrap();
        assert_eq!(one.upstreams, ["http://a:1"]);
        assert_eq!(one.policy, Policy::LeastLoaded);
        assert_eq!(one.listen, BalancerConfig::DEFAULT_LISTEN);

        let err = balance(r#"balance { upstream "http://a:1"; }"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown field `upstream`"), "{err}");
        let err = balance(r#"balance { policy "random"; }"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown policy `random`"), "{err}");
        let err = balance("balance { health_interval 1000; }")
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected a duration"), "{err}");
    }
}
