//! The error type of `cuca serve`.

use std::fmt;

use crate::error::CucaError;

/// Why the server could not start or stopped.
///
/// A request that fails is not a `ServerError`: it is answered with an
/// OpenAI-style error body (see the `openai` module) and the server goes on.
#[derive(Debug)]
#[non_exhaustive]
pub enum ServerError {
    /// A command-line option, a `--config` file or a model spec is invalid.
    Config(String),
    /// The listen address could not be bound or the HTTP server failed.
    Io(String),
    /// A model failed to load; the server stops rather than serve without it.
    ModelLoad {
        /// The model's name.
        model: String,
        /// The loader's error.
        source: CucaError,
    },
}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "invalid server configuration: {message}"),
            Self::Io(message) => write!(f, "server I/O error: {message}"),
            Self::ModelLoad { model, source } => {
                write!(f, "model `{model}` failed to load: {source}")
            }
        }
    }
}

impl std::error::Error for ServerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ModelLoad { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<CucaError> for ServerError {
    /// A configuration error from the local provider (`--device vulkan` in a
    /// build without the feature) is a configuration error of the server.
    fn from(error: CucaError) -> Self {
        Self::Config(error.to_string())
    }
}
