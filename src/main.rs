use gazellm::config::EngineConfig;
use gazellm::engine_version;
use gazellm::net::start_http_server;
use std::env;
use std::path::{Path, PathBuf};
use std::process;
use tracing::{error, info};

fn parse_config_path() -> PathBuf {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--config" {
            if let Some(val) = args.next() {
                return PathBuf::from(val);
            }
        }
    }
    PathBuf::from("config/engine_config.json")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();
    info!("Starting gazeLLM Engine v{}", engine_version());

    let config_path = parse_config_path();
    info!("Loading configuration from '{}'", config_path.display());

    let config = match EngineConfig::load_from_file(&config_path) {
        Ok(loaded_config) => loaded_config,
        Err(err) => {
            if !Path::new(&config_path).exists() {
                info!("Config file not found, using default engine configuration");
                EngineConfig::default()
            } else {
                error!(
                    "Failed to load configuration from '{}': {}",
                    config_path.display(),
                    err
                );
                process::exit(1);
            }
        }
    };

    info!(
        "Engine configured: host={}:{}, max_batch_size={}, block_size={}",
        config.host, config.port, config.max_batch_size, config.block_size
    );

    start_http_server(&config).await?;

    Ok(())
}
