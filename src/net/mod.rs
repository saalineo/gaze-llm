//! Network server infrastructure (gRPC and HTTP REST frontend).

pub mod grpc;
pub mod http;
pub mod request_processor;

pub use grpc::EngineInferenceService;
pub use http::start_http_server;
pub use request_processor::{validate_request, InferenceRequest, RequestValidationError};

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

