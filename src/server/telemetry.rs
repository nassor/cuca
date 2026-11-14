//! Logging: a `tracing` subscriber on stdout, human readable or JSON lines.
//!
//! The server's own events carry the request id, model, token counts and
//! timings as fields (`cuca::access` is the per-request line), so the JSON
//! format is directly ingestible by a log pipeline. The `log` node of the
//! configuration file ([`LogConfig`]) sets the format and the filter; no
//! environment variable is read.

use std::str::FromStr;

use serde::{Deserialize, Deserializer};
use tracing_subscriber::EnvFilter;

use super::error::ServerError;

/// How log lines are written.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LogFormat {
    /// Human readable lines.
    #[default]
    Pretty,
    /// One JSON object per line.
    Json,
}

impl FromStr for LogFormat {
    type Err = ServerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pretty" | "text" => Ok(Self::Pretty),
            "json" => Ok(Self::Json),
            other => Err(ServerError::Config(format!(
                "unknown log format `{other}`: expected pretty or json"
            ))),
        }
    }
}

impl<'de> Deserialize<'de> for LogFormat {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        super::config::parsed(deserializer)
    }
}

/// The `log` node: `log format="json" level="info,cuca=debug"`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LogConfig {
    /// `pretty` (the default) or `json`.
    pub format: LogFormat,
    /// A level or a full filter such as `info,cuca=debug` (default `info`).
    pub level: String,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            format: LogFormat::Pretty,
            level: "info".into(),
        }
    }
}

/// Install the global subscriber with `config`'s format and filter.
///
/// # Errors
///
/// [`ServerError::Config`] for a filter that does not parse or a subscriber
/// already installed.
pub fn init(config: &LogConfig) -> Result<(), ServerError> {
    let filter = EnvFilter::try_new(&config.level)
        .map_err(|e| ServerError::Config(format!("invalid log level: {e}")))?;
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    let installed = match config.format {
        LogFormat::Pretty => builder.try_init(),
        LogFormat::Json => builder.json().try_init(),
    };
    installed.map_err(|e| ServerError::Config(format!("logging is already set up: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_formats_parse_case_insensitively_and_reject_the_rest() {
        assert_eq!("JSON".parse::<LogFormat>().unwrap(), LogFormat::Json);
        assert_eq!("pretty".parse::<LogFormat>().unwrap(), LogFormat::Pretty);
        assert!("xml".parse::<LogFormat>().is_err());
    }
}
