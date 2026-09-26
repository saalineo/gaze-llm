//! gazeLLM Core-Engine Library
//!
//! zero-Python high-throughput LLM serving infrastructure

pub mod config;
pub mod ffi;
pub mod memory;
pub mod net;
pub mod scheduler;
pub mod weights;

/// Returns the engine version string.
pub fn engine_version() -> &'static str {
    "0.1.0-alpha"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_version() {
        assert_eq!(engine_version(), "0.1.0-alpha");
    }
}
