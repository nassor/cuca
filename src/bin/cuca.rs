//! The `cuca` command: `cuca serve` runs an OpenAI-compatible model server over
//! the in-process local engine, `cuca balance` balances several of them. Both
//! read one KDL file (`--config`, default `cuca.kdl`; see
//! `cuca::server::config`). This file only parses arguments; the server is
//! `cuca::server`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use cuca::server::{CucaConfig, ServerError, balance, init_logging, serve};

#[derive(Debug, Parser)]
#[command(name = "cuca", version, about = "cuca: an in-process LLM server")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the models of the file's `serve` section over an
    /// OpenAI-compatible HTTP API (vLLM and llama.cpp compatible).
    Serve(ConfigArgs),

    /// Balance chat requests over the `cuca serve` instances of the file's
    /// `balance` section, with prefix-cache affinity, passive and active
    /// health checks and a stats API of its own.
    Balance(ConfigArgs),

    /// Watch a running server or balancer in the terminal: resources,
    /// latency and the reasons for it (needs `stats_api #true` on the node).
    #[cfg(feature = "server-tui")]
    Tui(TuiArgs),
}

/// Options of `cuca serve` and `cuca balance`.
#[derive(Debug, Args)]
struct ConfigArgs {
    /// The KDL configuration file.
    #[arg(long, short = 'c', value_name = "FILE", default_value = "cuca.kdl")]
    config: PathBuf,
}

/// Options of `cuca tui`.
#[cfg(feature = "server-tui")]
#[derive(Debug, Args)]
struct TuiArgs {
    /// Base URL of the server or balancer to watch; it must serve the stats
    /// API (`stats_api #true`).
    #[arg(long, value_name = "URL", default_value = "http://127.0.0.1:8000")]
    url: String,

    /// Bearer token, when the node was started with an API key.
    #[arg(long, value_name = "KEY")]
    api_key: Option<String>,

    /// Milliseconds between two snapshots (at least 100).
    #[arg(long, value_name = "MS", default_value_t = 1000)]
    interval_ms: u64,
}

/// Runs `cuca serve`.
async fn run_serve(args: ConfigArgs) -> Result<(), ServerError> {
    let config = CucaConfig::load(&args.config)?;
    init_logging(&config.log)?;
    serve(config.into_serve()?).await
}

/// Runs `cuca balance`.
async fn run_balance(args: ConfigArgs) -> Result<(), ServerError> {
    let config = CucaConfig::load(&args.config)?;
    init_logging(&config.log)?;
    balance(config.into_balance()?).await
}

#[tokio::main]
async fn main() -> ExitCode {
    let outcome = match Cli::parse().command {
        Command::Serve(args) => run_serve(args).await,
        Command::Balance(args) => run_balance(args).await,
        #[cfg(feature = "server-tui")]
        Command::Tui(args) => {
            cuca::server::tui::run(cuca::server::tui::TuiOptions {
                url: args.url,
                api_key: args.api_key,
                interval_ms: args.interval_ms,
            })
            .await
        }
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
