//! Weight loader and mmap model parser module.

/// Weight loader stub for safetensors / GGUF model files.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WeightLoader;

impl WeightLoader {
    /// Creates a new [`WeightLoader`] instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

