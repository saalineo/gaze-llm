//! Network server infrastructure (gRPC and HTTP REST frontend).

pub mod grpc;
pub mod http;
pub mod request_processor;

pub use http::start_http_server;
pub use request_processor::{validate_request, InferenceRequest};

/// Network frontend server stub.
#[derive(Default)]
pub struct NetworkServer;

impl NetworkServer {
    pub fn new() -> Self {
        Self
    }
}
