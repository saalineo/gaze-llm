//! Engine configuration management module.

pub struct EngineConfig {
    pub max_batch_size: usize,
    pub max_num_seqs: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_num_seqs: 256,
        }
    }
}
