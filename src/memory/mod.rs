//! Memory management, KV-cache allocation, and pinned memory buffers.

pub mod raw_buffer;

pub use raw_buffer::RawMemoryBuffer;

/// KV-cache allocator stub.
#[derive(Default)]
pub struct MemoryManager;

impl MemoryManager {
    pub fn new() -> Self {
        Self
    }
}
