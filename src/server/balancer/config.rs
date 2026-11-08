//! Balancer configuration: the command line and an optional `[balancer]` table.
//!
//! ```toml
//! [balancer]
//! listen = "0.0.0.0:8000"
//! upstreams = ["http://10.0.0.1:8000", "http://10.0.0.2:8000"]
//! upstream_api_key = "secret"
//! policy = "prefix-affinity"
//! health_interval_ms = 1000
//! stats_api = true
//! ```
//!
//! A value on the command line wins over the file.

use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use super::routing::Policy;
use crate::server::error::ServerError;

/// Most upstreams one balancer fronts.
pub const MAX_UPSTREAMS: usize = 256;

/// Everything `balance` needs to start.
#[derive(Debug, Clone)]
pub struct BalancerConfig {
    /// Address of the HTTP listener.
    pub listen: SocketAddr,
    /// Base URLs of the model servers, at least one.
    pub upstreams: Vec<String>,
    /// Sent to every upstream as `Authorization: Bearer <key>`.
    pub upstream_api_key: Option<String>,
    /// Required from clients on every route but `/health`.
    pub api_key: Option<String>,
    /// Serve the stats API.
    pub stats_api: bool,
    /// How requests pick an upstream.
    pub policy: Policy,
    /// The interval between polls of each upstream.
    pub health_interval: Duration,
    /// The sliding window of the latency figures.
    pub latency_window: Duration,
}

impl BalancerConfig {
    /// The default listen address.
    pub const DEFAULT_LISTEN: &'static str = "127.0.0.1:8000";
    /// The default health interval.
    pub const DEFAULT_HEALTH_INTERVAL: Duration = Duration::from_millis(1000);

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
            latency_window: Duration::from_secs(60),
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
                "no upstream: pass --upstream <URL> at least once".into(),
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
                "--health-interval-ms must be between 100 and 60000".into(),
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

/// The balancer options a command line carries; unset ones fall to the file,
/// then to the defaults.
#[derive(Debug, Clone, Default)]
pub struct BalancerArgs {
    /// `--listen`.
    pub listen: Option<SocketAddr>,
    /// `--upstream`, repeated.
    pub upstreams: Vec<String>,
    /// `--upstream-api-key`.
    pub upstream_api_key: Option<String>,
    /// `--api-key`.
    pub api_key: Option<String>,
    /// `--stats-api`.
    pub stats_api: bool,
    /// `--policy`.
    pub policy: Option<String>,
    /// `--health-interval-ms`.
    pub health_interval_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    balancer: Option<FileBalancer>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileBalancer {
    listen: Option<SocketAddr>,
    #[serde(default)]
    upstreams: Vec<String>,
    upstream_api_key: Option<String>,
    api_key: Option<String>,
    stats_api: Option<bool>,
    policy: Option<String>,
    health_interval_ms: Option<u64>,
}

impl BalancerConfig {
    /// The configuration of a command line, over an optional TOML file with a
    /// `[balancer]` table: `args` win, the upstreams of both are joined (the
    /// command line's first).
    ///
    /// # Errors
    ///
    /// [`ServerError::Config`] for an unreadable or invalid file, an unknown
    /// policy or an invalid combination.
    pub fn from_args(args: BalancerArgs, file: Option<&Path>) -> Result<Self, ServerError> {
        let from_file = match file {
            None => FileBalancer::default(),
            Some(path) => {
                let text = std::fs::read_to_string(path).map_err(|e| {
                    ServerError::Config(format!("reading `{}`: {e}", path.display()))
                })?;
                let parsed: ConfigFile = toml::from_str(&text)
                    .map_err(|e| ServerError::Config(format!("`{}`: {e}", path.display())))?;
                parsed.balancer.unwrap_or_default()
            }
        };
        Self::merge(args, from_file)
    }

    fn merge(args: BalancerArgs, file: FileBalancer) -> Result<Self, ServerError> {
        let listen = match args.listen.or(file.listen) {
            Some(addr) => addr,
            None => Self::DEFAULT_LISTEN
                .parse()
                .map_err(|e| ServerError::Config(format!("default listen address: {e}")))?,
        };
        let mut upstreams = args.upstreams;
        upstreams.extend(file.upstreams);
        let mut config = Self::new(listen, upstreams);
        config.upstream_api_key = args.upstream_api_key.or(file.upstream_api_key);
        config.api_key = args.api_key.or(file.api_key);
        config.stats_api = args.stats_api || file.stats_api.unwrap_or(false);
        if let Some(policy) = args.policy.or(file.policy) {
            config.policy = policy.parse()?;
        }
        if let Some(ms) = args.health_interval_ms.or(file.health_interval_ms) {
            config.health_interval = Duration::from_millis(ms);
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(upstreams: &[&str]) -> BalancerConfig {
        BalancerConfig::new(
            BalancerConfig::DEFAULT_LISTEN.parse().unwrap(),
            upstreams.iter().map(|s| (*s).to_string()).collect(),
        )
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
    fn the_command_line_wins_over_the_file_and_the_upstreams_join() {
        let file: ConfigFile = toml::from_str(
            r#"[balancer]
               listen = "0.0.0.0:9000"
               upstreams = ["http://file:1"]
               policy = "round-robin"
               health_interval_ms = 500
               stats_api = true"#,
        )
        .unwrap();
        let args = BalancerArgs {
            upstreams: vec!["http://cli:1".into()],
            policy: Some("least-loaded".into()),
            ..BalancerArgs::default()
        };
        let c = BalancerConfig::merge(args, file.balancer.unwrap()).unwrap();
        assert_eq!(c.listen.port(), 9000);
        assert_eq!(c.upstreams, ["http://cli:1", "http://file:1"]);
        assert_eq!(c.policy, Policy::LeastLoaded);
        assert_eq!(c.health_interval, Duration::from_millis(500));
        assert!(c.stats_api);
    }

    #[test]
    fn an_unknown_key_or_policy_is_rejected() {
        assert!(toml::from_str::<ConfigFile>("[balancer]\nupstream = []\n").is_err());
        let args = BalancerArgs {
            policy: Some("random".into()),
            ..BalancerArgs::default()
        };
        assert!(BalancerConfig::merge(args, FileBalancer::default()).is_err());
    }
}
