//! Engine configuration management module.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// High-performance engine configuration settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineConfig {
    /// Host IP address to bind network listeners.
    pub host: String,
    /// Port number for HTTP/REST and gRPC services.
    pub port: u16,
    /// Maximum number of concurrent requests in a batch.
    pub max_batch_size: usize,
    /// Number of tokens per `PagedAttention` memory block.
    pub block_size: usize,

    /// Total number of memory blocks allocated in host RAM.
    pub max_num_blocks: usize,
    /// Total number of memory blocks allocated on GPU VRAM.
    pub num_gpu_blocks: usize,
    /// File path to the compiled Mojo shared library (`.so` / `.dylib`).
    pub mojo_library_path: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_batch_size: 32,
            block_size: 16,
            max_num_blocks: 1024,
            num_gpu_blocks: 512,
            mojo_library_path: "target/libmojo_core.so".to_string(),
        }
    }
}

/// Errors encountered when loading engine configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// I/O failure when reading the configuration file.
    #[error("failed to read configuration file at '{path}': {source}")]
    Io {
        /// Target file path.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// JSON deserialization failure.
    #[error("failed to parse JSON configuration in '{path}': {source}")]
    Json {
        /// Target file path.
        path: String,
        /// Underlying serde JSON error.
        #[source]
        source: serde_json::Error,
    },
}

impl EngineConfig {
    /// Loads configuration from a JSON file.
    ///
    /// # Errors
    /// Returns [`ConfigError::Io`] if the file cannot be read, or [`ConfigError::Json`]
    /// if the file contains invalid JSON structure.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path_ref = path.as_ref();
        let path_str = path_ref.to_string_lossy().to_string();
        let content = fs::read_to_string(path_ref).map_err(|source| ConfigError::Io {
            path: path_str.clone(),
            source,
        })?;
        let config: Self =
            serde_json::from_str(&content).map_err(|source| ConfigError::Json {
                path: path_str,
                source,
            })?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_valid_config() {
        let json = r#"{
            "host": "127.0.0.1",
            "port": 8080,
            "max_batch_size": 32,
            "block_size": 16,
            "max_num_blocks": 1024,
            "num_gpu_blocks": 512,
            "mojo_library_path": "target/libmojo_core.so"
        }"#;

        let config: EngineConfig =
            serde_json::from_str(json).expect("valid JSON should deserialize");
        assert_eq!(config, EngineConfig::default());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let load_result = EngineConfig::load_from_file("nonexistent_config_path_12345.json");
        assert!(load_result.is_err());
        let err = load_result.unwrap_err();
        assert!(matches!(err, ConfigError::Io { .. }));
        assert!(err
            .to_string()
            .contains("failed to read configuration file"));
    }

    #[test]
    fn test_load_invalid_json_file() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("invalid_config_test.json");
        let mut file = fs::File::create(&file_path).expect("failed to create temporary file");
        writeln!(file, "{{ invalid json ").expect("failed to write invalid json");

        let load_result = EngineConfig::load_from_file(&file_path);
        assert!(load_result.is_err());
        let err = load_result.unwrap_err();
        assert!(matches!(err, ConfigError::Json { .. }));
        assert!(err
            .to_string()
            .contains("failed to parse JSON configuration"));
        let _ = fs::remove_file(file_path);
    }
}

