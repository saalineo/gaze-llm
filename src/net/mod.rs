//! Network server infrastructure (gRPC and HTTP REST frontend).

pub mod grpc;
pub mod http;
pub mod request_processor;
pub mod sse;

pub use grpc::EngineInferenceService;
pub use http::{
    create_router, start_http_server, HttpCompletionRequest, HttpCompletionResponse,
    HttpServerError,
};
pub use request_processor::{
    validate_request, InferenceRequest, InferenceRequestBuilder, RequestValidationError,
};
pub use sse::{create_token_stream, create_token_stream_with_keepalive, SseTokenPayload};

/// Network frontend server stub.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NetworkServer;

impl NetworkServer {
    /// Creates a new [`NetworkServer`] instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}
