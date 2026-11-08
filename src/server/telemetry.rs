//! Logging: a `tracing` subscriber on stdout, human readable or JSON lines.
//!
//! The server's own events carry the request id, model, token counts and
//! timings as fields (`cuca::access` is the per-request line), so the JSON
//! format is directly ingestible by a log pipeline.

use std::str::FromStr;

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

/// Install the global subscriber. `level` (a level or a full filter such as
/// `info,cuca=debug`) wins over the `RUST_LOG` environment variable, which
/// wins over the default `info`.
///
/// # Errors
///
/// [`ServerError::Config`] for a filter that does not parse or a subscriber
/// already installed.
pub fn init(format: LogFormat, level: Option<&str>) -> Result<(), ServerError> {
    let filter = match level {
        Some(level) => EnvFilter::try_new(level),
        None => EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info")),
    }
    .map_err(|e| ServerError::Config(format!("invalid log filter: {e}")))?;
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    let installed = match format {
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
