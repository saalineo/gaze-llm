//! Axum asynchronous HTTP API server.

use crate::config::EngineConfig;
use axum::{
    routing::{get, post},
    Json, Router,
};
use std::net::SocketAddr;

/// Constructs the Axum HTTP router with all engine endpoints configured.
pub fn create_router() -> Router {
    Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/v1/completions", post(handle_completion))
}


/// Starts the Axum HTTP REST server bound to the address specified in `config`.
///
/// # Errors
/// Returns an error if the host/port address fails to parse, the socket cannot be bound,
/// or the underlying server fails during execution.
pub async fn start_http_server(
    config: &EngineConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = create_router();

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    tracing::info!("HTTP server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_completion(Json(_payload): Json<serde_json::Value>) -> &'static str {
    r#"{"text": "stub", "tokens": []}"#
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .expect("failed to build HTTP request"),
            )
            .await
            .expect("service execution failed");

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read response body");
        assert_eq!(&body[..], b"OK");
    }

    #[tokio::test]
    async fn test_completions_endpoint() {
        let app = create_router();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt": "Hello"}"#))
                    .expect("failed to build HTTP request"),
            )
            .await
            .expect("service execution failed");

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read response body");
        assert_eq!(&body[..], b"{\"text\": \"stub\", \"tokens\": []}");
    }
}

