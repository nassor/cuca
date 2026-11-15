//! The `cuca` command: `cuca run` starts what one KDL file holds (`--config`,
//! default `cuca.kdl`; see `cuca::server::config`): a `serve` section is an
//! OpenAI-compatible model server over the in-process local engine, a
//! `balance` section a load balancer over several of them, and a file with
//! both runs both in one process. This file only parses arguments; the
//! server is `cuca::server`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use cuca::server::{CucaConfig, ServerError, init_logging, run};

#[derive(Debug, Parser)]
#[command(name = "cuca", version, about = "cuca: an in-process LLM server")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the file's sections: `serve` loads models behind an
    /// OpenAI-compatible HTTP API, `balance` spreads chat requests over
    /// model servers with prefix-cache affinity and health checks, and with
    /// both the balancer also routes to the local models (two listeners).
    Run(RunArgs),

    /// Watch a running server or balancer in the terminal: resources,
    /// latency and the reasons for it (needs `stats_api #true` on the node).
    #[cfg(feature = "server-tui")]
    Tui(TuiArgs),
}

/// Options of `cuca run`.
#[derive(Debug, Args)]
struct RunArgs {
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

/// Runs `cuca run`.
async fn run_node(args: RunArgs) -> Result<(), ServerError> {
    let CucaConfig {
        log,
        serve,
        balance,
    } = CucaConfig::load(&args.config)?;
    init_logging(&log)?;
    run(serve, balance).await
}

#[tokio::main]
async fn main() -> ExitCode {
    let outcome = match Cli::parse().command {
        Command::Run(args) => run_node(args).await,
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
