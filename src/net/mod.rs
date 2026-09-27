//! Network server infrastructure (gRPC and HTTP REST frontend).

pub mod grpc;
pub mod http;

pub use http::start_http_server;

/// Network frontend server stub.
#[derive(Default)]
pub struct NetworkServer;

impl NetworkServer {
    pub fn new() -> Self {
        Self
    }
}
