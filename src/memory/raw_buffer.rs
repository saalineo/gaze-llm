//! Raw memory buffer allocation and lifecycle management.

use std::alloc::{alloc, dealloc, Layout, LayoutError};
use std::fmt;
use std::ptr::NonNull;


/// Errors that can occur when allocating a [`RawMemoryBuffer`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AllocationError {
    /// Attempted to allocate a zero-byte buffer.
    #[error("cannot allocate zero-sized buffer")]
    ZeroSize,
    /// Invalid layout requested (e.g. non-power-of-two alignment or integer overflow).
    #[error("invalid memory layout: {0}")]
    InvalidLayout(#[from] LayoutError),
    /// The global system allocator returned a null pointer.
    #[error("system allocator failed to allocate memory")]
    OutOfMemory,
}

/// RAII-managed aligned raw memory buffer for zero-overhead cross-runtime (Mojo/CUDA) FFI execution.
pub struct RawMemoryBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl RawMemoryBuffer {
    /// Allocates an uninitialized memory buffer of `size_bytes` aligned to `align_bytes`.
    ///
    /// # Errors
    /// Returns [`AllocationError::ZeroSize`] if `size_bytes` is 0, [`AllocationError::InvalidLayout`]
    /// if `align_bytes` is not a valid power of two, or [`AllocationError::OutOfMemory`] if
    /// the underlying system allocator fails.
    pub fn allocate(size_bytes: usize, align_bytes: usize) -> Result<Self, AllocationError> {
        if size_bytes == 0 {
            return Err(AllocationError::ZeroSize);
        }
        let layout = Layout::from_size_align(size_bytes, align_bytes)?;

        // SAFETY: `layout` has non-zero size, satisfying the allocator contract.
        let raw_ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw_ptr).ok_or(AllocationError::OutOfMemory)?;

        Ok(Self { ptr, layout })
    }

    /// Returns a raw mutable pointer to the underlying buffer.
    #[must_use]
    #[inline]
    pub const fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns a raw immutable pointer to the underlying buffer.
    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns the buffer allocation size in bytes.
    #[must_use]
    #[inline]
    pub const fn size(&self) -> usize {
        self.layout.size()
    }

    /// Returns the buffer memory alignment in bytes.
    #[must_use]
    #[inline]
    pub const fn align(&self) -> usize {
        self.layout.align()
    }

    /// Returns `true` if the buffer has a size of 0 bytes.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.layout.size() == 0
    }

    /// Returns an immutable slice view of the buffer memory.
    ///
    /// # Safety
    /// The caller must ensure that the memory within the buffer has been properly initialized
    /// before reading through the returned slice.
    #[must_use]
    #[inline]
    pub const unsafe fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` is non-null and valid for reads of `self.layout.size()` bytes
        // if initialized by caller.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.layout.size()) }
    }

    /// Returns a mutable slice view of the buffer memory.
    ///
    /// # Safety
    /// The caller must ensure that reading uninitialized memory through this slice is avoided
    /// until initialized, or that the slice is used only for writing.
    #[must_use]
    #[inline]
    pub const unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `self.ptr` is non-null and valid for writes of `self.layout.size()` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }

}

impl Drop for RawMemoryBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated with `self.layout` via `std::alloc::alloc`.
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

impl fmt::Debug for RawMemoryBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawMemoryBuffer")
            .field("ptr", &self.ptr)
            .field("size", &self.layout.size())
            .field("align", &self.layout.align())
            .finish()
    }
}

// SAFETY: `RawMemoryBuffer` uniquely owns the allocated memory block and its layout metadata.
// Transferring ownership across thread boundaries does not introduce data races.
unsafe impl Send for RawMemoryBuffer {}

// SAFETY: `RawMemoryBuffer` internal metadata is immutable after construction.
// Dereferencing the raw pointer returned by `as_mut_ptr` or `as_ptr` requires unsafe code;
// callers are responsible for coordinating concurrent access across threads.
unsafe impl Sync for RawMemoryBuffer {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_allocate_and_access() {
        let size = 1024;
        let align = 64;
        let mut buffer =
            RawMemoryBuffer::allocate(size, align).expect("allocation should succeed");

        assert_eq!(buffer.size(), size);
        assert_eq!(buffer.align(), align);
        assert!(!buffer.is_empty());
        let ptr = buffer.as_mut_ptr();
        assert!(!ptr.is_null());
        assert_eq!(
            (ptr as usize) % align,
            0,
            "pointer must be 64-byte aligned"
        );

        // SAFETY: Testing raw pointer writes to allocated memory.
        unsafe {
            ptr.write(0xAA);
            assert_eq!(ptr.read(), 0xAA);
            ptr.add(size - 1).write(0x55);
            assert_eq!(ptr.add(size - 1).read(), 0x55);

            let slice = buffer.as_mut_slice();
            slice[0] = 0x11;
            slice[size - 1] = 0x22;
            assert_eq!(buffer.as_slice()[0], 0x11);
            assert_eq!(buffer.as_slice()[size - 1], 0x22);
        }
    }

    #[test]
    fn test_allocate_zero_size_fails() {
        let err = RawMemoryBuffer::allocate(0, 64).expect_err("0-byte allocation must fail");
        assert_eq!(err, AllocationError::ZeroSize);
    }

    #[test]
    fn test_allocate_invalid_alignment_fails() {
        let err =
            RawMemoryBuffer::allocate(1024, 3).expect_err("non-power-of-two alignment must fail");
        assert!(matches!(err, AllocationError::InvalidLayout(_)));

        let err_zero = RawMemoryBuffer::allocate(1024, 0).expect_err("zero alignment must fail");
        assert!(matches!(err_zero, AllocationError::InvalidLayout(_)));
    }

    #[test]
    fn test_allocate_10mb_64byte_aligned() {
        const TEN_MB: usize = 10 * 1024 * 1024;
        const ALIGN_64: usize = 64;

        let buffer = RawMemoryBuffer::allocate(TEN_MB, ALIGN_64)
            .expect("10MB 64-byte aligned allocation should succeed");

        assert_eq!(buffer.size(), TEN_MB);
        assert_eq!(buffer.align(), ALIGN_64);

        let ptr = buffer.as_mut_ptr();
        assert_eq!(
            (ptr as usize) % ALIGN_64,
            0,
            "buffer pointer must satisfy 64-byte alignment"
        );

        // SAFETY: Pointer offsets within allocated range.
        unsafe {
            ptr.write(42);
            ptr.add(TEN_MB / 2).write(84);
            ptr.add(TEN_MB - 1).write(126);

            assert_eq!(ptr.read(), 42);
            assert_eq!(ptr.add(TEN_MB / 2).read(), 84);
            assert_eq!(ptr.add(TEN_MB - 1).read(), 126);
        }
    }

    #[test]
    fn test_thread_send() {
        let buffer = RawMemoryBuffer::allocate(512, 16).unwrap();
        let expected_addr = buffer.as_ptr() as usize;

        let handle = thread::spawn(move || {
            assert_eq!(buffer.as_ptr() as usize, expected_addr);
            buffer.size()
        });

        let returned_size = handle.join().expect("thread join should succeed");
        assert_eq!(returned_size, 512);
    }

    #[test]
    fn test_thread_sync() {
        let buffer = Arc::new(RawMemoryBuffer::allocate(256, 32).unwrap());
        let mut handles = Vec::new();

        for _ in 0..4 {
            let buf_clone = Arc::clone(&buffer);
            handles.push(thread::spawn(move || {
                assert_eq!(buf_clone.size(), 256);
                assert_eq!(buf_clone.align(), 32);
            }));
        }

        for handle in handles {
            handle.join().expect("thread join should succeed");
        }
    }

    #[test]
    fn test_debug_formatting() {
        let buffer = RawMemoryBuffer::allocate(128, 16).unwrap();
        let debug_str = format!("{buffer:?}");
        assert!(debug_str.contains("RawMemoryBuffer"));
        assert!(debug_str.contains("size: 128"));
        assert!(debug_str.contains("align: 16"));
    }
}

