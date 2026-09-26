use gazellm::engine_version;
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    info!("Starting gazeLLM Engine v{}", engine_version());
}
