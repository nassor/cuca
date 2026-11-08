//! The `cuca` command: `cuca serve` runs an OpenAI-compatible model server over
//! the in-process local engine. This file only parses arguments; the server is
//! `cuca::server`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use cuca::server::{
    BalancerArgs, BalancerConfig, LogFormat, ModelOptions, ServerConfig, ServerError, balance,
    collect_models, init_logging, serve,
};

#[derive(Debug, Parser)]
#[command(name = "cuca", version, about = "cuca: an in-process LLM server")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve one or more models over an OpenAI-compatible HTTP API (vLLM and
    /// llama.cpp compatible).
    Serve(Box<ServeArgs>),

    /// Balance chat requests over several `cuca serve` instances, with
    /// prefix-cache affinity, passive and active health checks and a stats
    /// API of its own.
    Balance(Box<BalanceArgs>),

    /// Watch a running server or balancer in the terminal: resources,
    /// latency and the reasons for it (needs `--stats-api` on the node).
    #[cfg(feature = "server-tui")]
    Tui(TuiArgs),
}

#[derive(Debug, Args)]
struct ServeArgs {
    /// Address to listen on (vLLM's default port).
    #[arg(long, default_value = ServerConfig::DEFAULT_LISTEN)]
    listen: SocketAddr,

    /// A model to serve, `[name=]source`; repeat for several. A source is a
    /// preset name, `repo:revision:file` (a GGUF on Hugging Face),
    /// `repo:revision` (a safetensors checkpoint) or a path to a GGUF file or
    /// a checkpoint directory. The name defaults to the preset name, the file
    /// stem or the repository name.
    #[arg(long = "model", value_name = "[NAME=]SOURCE")]
    models: Vec<String>,

    /// A TOML file of `[[model]]` tables (`name`, `source`, `device`,
    /// `context_length`, `batching`, `experts`, `gpu_layers`, `cache_dir`,
    /// `chat_template_file`, `quant`, `drafter`, `draft_tokens`) overriding
    /// the options below per model.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Device of every model: `cpu`, `vulkan[:n]`, `metal`, `cuda[:n]`,
    /// `rocm[:n]`; join several with `+` to split one model over GPUs. A GPU
    /// needs its cargo feature (`provider-local-vulkan`, ...).
    #[arg(long, value_name = "DEVICE")]
    device: Option<String>,

    /// Maximum prompt plus completion tokens per request (default: the
    /// smaller of 8192 and the model's own maximum).
    #[arg(long, value_name = "TOKENS")]
    context_length: Option<usize>,

    /// Batching: any of `seqs=4,kv=16384,step=256,prefix=off`.
    #[arg(long, value_name = "SPEC")]
    batching: Option<String>,

    /// Mixture-of-experts placement: `resident`, `tiered:ram=4g,device=6g` or
    /// `host[:layers=n]`.
    #[arg(long, value_name = "SPEC")]
    experts: Option<String>,

    /// Layers to run on the GPU; the rest run on the CPU (llama.cpp's `-ngl`).
    #[arg(long, value_name = "N")]
    gpu_layers: Option<usize>,

    /// Directory models are downloaded and converted into (default:
    /// `CUCA_CACHE_DIR`, then the platform cache directory).
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,

    /// Require `Authorization: Bearer <KEY>` on every route except `/health`.
    #[arg(long, value_name = "KEY", env = "CUCA_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// Serve the stats API: `GET /cuca/v1/stats` and
    /// `GET /cuca/v1/stats/stream`.
    #[arg(long)]
    stats_api: bool,

    /// Log format: `pretty` or `json`.
    #[arg(long, value_name = "FORMAT", default_value = "pretty")]
    log_format: String,

    /// Log filter, a level or `target=level,...` (default: `RUST_LOG`, else
    /// `info`).
    #[arg(long, value_name = "LEVEL")]
    log_level: Option<String>,
}

/// Options of `cuca balance`.
#[derive(Debug, Args)]
struct BalanceArgs {
    /// Address to listen on (default 127.0.0.1:8000).
    #[arg(long)]
    listen: Option<SocketAddr>,

    /// Base URL of a `cuca serve` instance (`http://host:port`); repeat for
    /// several. At least one, here or in the config file.
    #[arg(long = "upstream", value_name = "URL")]
    upstreams: Vec<String>,

    /// Sent to every upstream as `Authorization: Bearer <KEY>`.
    #[arg(
        long,
        value_name = "KEY",
        env = "CUCA_UPSTREAM_API_KEY",
        hide_env_values = true
    )]
    upstream_api_key: Option<String>,

    /// Require `Authorization: Bearer <KEY>` from clients on every route
    /// except `/health`.
    #[arg(long, value_name = "KEY", env = "CUCA_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// Serve the stats API: `GET /cuca/v1/stats` and
    /// `GET /cuca/v1/stats/stream` (`role: balancer`, upstream snapshots
    /// nested).
    #[arg(long)]
    stats_api: bool,

    /// How a request picks an upstream: `prefix-affinity` (default),
    /// `least-loaded` or `round-robin`.
    #[arg(long, value_name = "POLICY")]
    policy: Option<String>,

    /// Milliseconds between polls of each upstream (100 to 60000, default
    /// 1000).
    #[arg(long, value_name = "MS")]
    health_interval_ms: Option<u64>,

    /// A TOML file with a `[balancer]` table (`listen`, `upstreams`,
    /// `upstream_api_key`, `api_key`, `stats_api`, `policy`,
    /// `health_interval_ms`); command-line options win.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Log format: `pretty` or `json`.
    #[arg(long, value_name = "FORMAT", default_value = "pretty")]
    log_format: String,

    /// Log filter, a level or `target=level,...` (default: `RUST_LOG`, else
    /// `info`).
    #[arg(long, value_name = "LEVEL")]
    log_level: Option<String>,
}

/// Options of `cuca tui`.
#[cfg(feature = "server-tui")]
#[derive(Debug, Args)]
struct TuiArgs {
    /// Base URL of the server or balancer to watch; it must serve the stats
    /// API (`--stats-api`).
    #[arg(long, value_name = "URL", default_value = "http://127.0.0.1:8000")]
    url: String,

    /// Bearer token, when the node was started with an API key.
    #[arg(long, value_name = "KEY", env = "CUCA_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// Milliseconds between two snapshots (at least 100).
    #[arg(long, value_name = "MS", default_value_t = 1000)]
    interval_ms: u64,
}

fn serve_config(args: &ServeArgs) -> Result<ServerConfig, ServerError> {
    let defaults = ModelOptions {
        device: args.device.clone(),
        context_length: args.context_length,
        batching: args.batching.clone(),
        experts: args.experts.clone(),
        gpu_layers: args.gpu_layers,
        cache_dir: args.cache_dir.clone(),
        ..ModelOptions::default()
    };
    let models = collect_models(&args.models, args.config.as_deref(), &defaults)?;
    let mut config = ServerConfig::new(args.listen, models);
    config.api_key.clone_from(&args.api_key);
    config.stats_api = args.stats_api;
    Ok(config)
}

/// Runs `cuca serve`.
async fn run_serve(args: ServeArgs) -> Result<(), ServerError> {
    init_logging(
        args.log_format.parse::<LogFormat>()?,
        args.log_level.as_deref(),
    )?;
    serve(serve_config(&args)?).await
}

/// Runs `cuca balance`.
async fn run_balance(args: BalanceArgs) -> Result<(), ServerError> {
    init_logging(
        args.log_format.parse::<LogFormat>()?,
        args.log_level.as_deref(),
    )?;
    let config = BalancerConfig::from_args(
        BalancerArgs {
            listen: args.listen,
            upstreams: args.upstreams,
            upstream_api_key: args.upstream_api_key,
            api_key: args.api_key,
            stats_api: args.stats_api,
            policy: args.policy,
            health_interval_ms: args.health_interval_ms,
        },
        args.config.as_deref(),
    )?;
    balance(config).await
}

#[tokio::main]
async fn main() -> ExitCode {
    let outcome = match Cli::parse().command {
        Command::Serve(args) => run_serve(*args).await,
        Command::Balance(args) => run_balance(*args).await,
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
