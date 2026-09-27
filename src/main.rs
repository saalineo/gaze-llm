use gazellm::config::EngineConfig;
use gazellm::engine_version;
use gazellm::net::start_http_server;
use std::env;
use std::process;
use tracing::{error, info};

fn parse_config_path() -> String {
    let args: Vec<String> = env::args().collect();
    args.windows(2)
        .find(|pair| pair[0] == "--config")
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| "config/engine_config.json".to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    info!("Starting gazeLLM Engine v{}", engine_version());

    let config_path = parse_config_path();
    info!("Loading configuration from '{}'", config_path);

    let config = match EngineConfig::load_from_file(&config_path) {
        Ok(loaded_config) => loaded_config,
        Err(err) => {
            error!("Failed to load configuration from '{}': {}", config_path, err);
            process::exit(1);
        }
    };

    info!(
        "Engine configured: host={}:{}, max_batch_size={}, block_size={}",
        config.host, config.port, config.max_batch_size, config.block_size
    );

    start_http_server(config).await?;

    Ok(())
}
