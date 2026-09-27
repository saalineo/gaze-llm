//! Engine configuration management module.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;

/// High-performance engine configuration settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineConfig {
    pub host: String,
    pub port: u16,
    pub max_batch_size: usize,
    pub block_size: usize,
    pub max_num_blocks: usize,
    pub num_gpu_blocks: usize,
    pub mojo_library_path: String,
}

/// Errors encountered when loading engine configuration.
#[derive(Debug)]
pub enum ConfigError {
    Io { path: String, source: std::io::Error },
    Json { path: String, source: serde_json::Error },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io { path, source } => {
                write!(f, "Failed to read configuration file at '{}': {}", path, source)
            }
            ConfigError::Json { path, source } => {
                write!(f, "Failed to parse JSON configuration in '{}': {}", path, source)
            }
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io { source, .. } => Some(source),
            ConfigError::Json { source, .. } => Some(source),
        }
    }
}

impl EngineConfig {
    /// Loads configuration from a JSON file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path_ref = path.as_ref();
        let path_str = path_ref.to_string_lossy().to_string();
        let content = fs::read_to_string(path_ref).map_err(|source| ConfigError::Io {
            path: path_str.clone(),
            source,
        })?;
        let config: EngineConfig = serde_json::from_str(&content).map_err(|source| ConfigError::Json {
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

        let config: EngineConfig = serde_json::from_str(json).expect("Valid JSON should deserialize");
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.block_size, 16);
        assert_eq!(config.max_num_blocks, 1024);
        assert_eq!(config.num_gpu_blocks, 512);
        assert_eq!(config.mojo_library_path, "target/libmojo_core.so");
    }

    #[test]
    fn test_load_nonexistent_file() {
        let load_result = EngineConfig::load_from_file("nonexistent_config_path_12345.json");
        assert!(load_result.is_err());
        let err = load_result.unwrap_err();
        assert!(err.to_string().contains("Failed to read configuration file"));
    }

    #[test]
    fn test_load_invalid_json_file() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("invalid_config_test.json");
        let mut file = fs::File::create(&file_path).expect("Failed to create temporary file");
        writeln!(file, "{{ invalid json ").expect("Failed to write invalid json");

        let load_result = EngineConfig::load_from_file(&file_path);
        assert!(load_result.is_err());
        let err = load_result.unwrap_err();
        assert!(err.to_string().contains("Failed to parse JSON configuration"));
        let _ = fs::remove_file(file_path);
    }
}
