use std::alloc::{alloc, dealloc, Layout};
use std::fmt;
use std::ptr::NonNull;

/// RAII-managed aligned raw memory buffer for zero-overhead cross-runtime (Mojo/CUDA) FFI execution.
pub struct RawMemoryBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl RawMemoryBuffer {
    /// Allocates an uninitialized memory buffer of `size_bytes` aligned to `align_bytes`.
    ///
    /// # Errors
    /// Returns an error if `size_bytes` is 0, if `align_bytes` is not a valid power of two,
    /// or if the underlying system allocator fails.
    pub fn allocate(size_bytes: usize, align_bytes: usize) -> Result<Self, String> {
        if size_bytes == 0 {
            return Err("Cannot allocate 0-byte buffer".into());
        }
        let layout = Layout::from_size_align(size_bytes, align_bytes)
            .map_err(|e| format!("Invalid layout: {}", e))?;

        // SAFETY: `layout` has non-zero size, satisfying the allocator contract.
        let ptr = NonNull::new(unsafe { alloc(layout) })
            .ok_or_else(|| "Memory allocation failed".to_string())?;

        Ok(Self { ptr, layout })
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn size(&self) -> usize {
        self.layout.size()
    }

    pub fn align(&self) -> usize {
        self.layout.align()
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
        let buffer = RawMemoryBuffer::allocate(size, align).expect("Allocation should succeed");

        assert_eq!(buffer.size(), size);
        assert_eq!(buffer.align(), align);
        let ptr = buffer.as_mut_ptr();
        assert!(!ptr.is_null());
        assert_eq!((ptr as usize) % align, 0, "Pointer must be 64-byte aligned");

        unsafe {
            ptr.write(0xAA);
            assert_eq!(ptr.read(), 0xAA);
            ptr.add(size - 1).write(0x55);
            assert_eq!(ptr.add(size - 1).read(), 0x55);
        }
    }

    #[test]
    fn test_allocate_zero_size_fails() {
        let err = RawMemoryBuffer::allocate(0, 64).expect_err("0-byte allocation must fail");
        assert_eq!(err, "Cannot allocate 0-byte buffer");
    }

    #[test]
    fn test_allocate_invalid_alignment_fails() {
        let err =
            RawMemoryBuffer::allocate(1024, 3).expect_err("Non-power-of-two alignment must fail");
        assert!(err.contains("Invalid layout"));

        let err_zero = RawMemoryBuffer::allocate(1024, 0).expect_err("Zero alignment must fail");
        assert!(err_zero.contains("Invalid layout"));
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
            "Buffer pointer must satisfy 64-byte alignment"
        );

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

        let returned_size = handle.join().expect("Thread join should succeed");
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
            handle.join().expect("Thread join should succeed");
        }
    }

    #[test]
    fn test_debug_formatting() {
        let buffer = RawMemoryBuffer::allocate(128, 16).unwrap();
        let debug_str = format!("{:?}", buffer);
        assert!(debug_str.contains("RawMemoryBuffer"));
        assert!(debug_str.contains("size: 128"));
        assert!(debug_str.contains("align: 16"));
    }
}
