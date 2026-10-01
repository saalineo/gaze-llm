//! Memory management, KV-cache allocation, and pinned memory buffers.

pub mod raw_buffer;

pub use raw_buffer::{AllocationError, RawMemoryBuffer};

/// KV-cache allocator stub.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MemoryManager;

impl MemoryManager {
    /// Creates a new [`MemoryManager`] instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

