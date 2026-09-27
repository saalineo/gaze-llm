//! Network server infrastructure (gRPC and HTTP REST frontend).

pub mod http;

pub use http::start_http_server;

/// Network frontend server stub.
pub struct NetworkServer;

impl NetworkServer {
    pub fn new() -> Self {
        Self
    }
}
impl Default for NetworkServer {
    fn default() -> Self {
        Self::new()
    }
}
