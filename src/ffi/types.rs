//! Shared C ABI FFI type definitions for Rust and Mojo compute execution.

use std::ffi::c_char;

/// Supported tensor data types for Mojo compute kernels.
pub mod dtype {
    pub const FP32: i32 = 0;
    pub const FP16: i32 = 1;
    pub const BF16: i32 = 2;
    pub const INT8: i32 = 3;
}

/// C-compatible representation of a Mojo tensor memory buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MojoTensorBuffer {
    pub data_ptr: *mut std::ffi::c_void,
    pub num_elements: usize,
    pub element_size_bytes: usize,
    pub dtype: i32,
}

/// C-compatible representation of a block table for PagedAttention memory management.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MojoBlockTable {
    pub block_table_ptr: *mut i32,
    pub num_blocks: usize,
    pub block_size: usize,
}

/// C-compatible status result returned from Mojo FFI kernel invocations.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiResult {
    pub status_code: i32,
    pub error_message: *const c_char,
}

impl FfiResult {
    /// Constructs a success result (`status_code == 0`) with a null error pointer.
    pub const fn ok() -> Self {
        Self {
            status_code: 0,
            error_message: std::ptr::null(),
        }
    }

    /// Constructs an error result with the specified non-zero status code and message pointer.
    pub const fn err(status_code: i32, message: *const c_char) -> Self {
        Self {
            status_code,
            error_message: message,
        }
    }

    /// Returns `true` if the operation succeeded (`status_code == 0`).
    pub fn is_ok(&self) -> bool {
        self.status_code == 0
    }
}

// Compile-time static assertions ensuring memory alignment and layout match C ABI.
const _: () = {
    assert!(std::mem::size_of::<MojoTensorBuffer>() == 32);
    assert!(std::mem::align_of::<MojoTensorBuffer>() == 8);
    assert!(std::mem::offset_of!(MojoTensorBuffer, data_ptr) == 0);
    assert!(std::mem::offset_of!(MojoTensorBuffer, num_elements) == 8);
    assert!(std::mem::offset_of!(MojoTensorBuffer, element_size_bytes) == 16);
    assert!(std::mem::offset_of!(MojoTensorBuffer, dtype) == 24);

    assert!(std::mem::size_of::<MojoBlockTable>() == 24);
    assert!(std::mem::align_of::<MojoBlockTable>() == 8);
    assert!(std::mem::offset_of!(MojoBlockTable, block_table_ptr) == 0);
    assert!(std::mem::offset_of!(MojoBlockTable, num_blocks) == 8);
    assert!(std::mem::offset_of!(MojoBlockTable, block_size) == 16);

    assert!(std::mem::size_of::<FfiResult>() == 16);
    assert!(std::mem::align_of::<FfiResult>() == 8);
    assert!(std::mem::offset_of!(FfiResult, status_code) == 0);
    assert!(std::mem::offset_of!(FfiResult, error_message) == 8);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_buffer_layout() {
        assert_eq!(std::mem::size_of::<MojoTensorBuffer>(), 32);
        assert_eq!(std::mem::align_of::<MojoTensorBuffer>(), 8);
        assert_eq!(std::mem::offset_of!(MojoTensorBuffer, data_ptr), 0);
        assert_eq!(std::mem::offset_of!(MojoTensorBuffer, num_elements), 8);
        assert_eq!(
            std::mem::offset_of!(MojoTensorBuffer, element_size_bytes),
            16
        );
        assert_eq!(std::mem::offset_of!(MojoTensorBuffer, dtype), 24);
    }

    #[test]
    fn test_block_table_layout() {
        assert_eq!(std::mem::size_of::<MojoBlockTable>(), 24);
        assert_eq!(std::mem::align_of::<MojoBlockTable>(), 8);
        assert_eq!(std::mem::offset_of!(MojoBlockTable, block_table_ptr), 0);
        assert_eq!(std::mem::offset_of!(MojoBlockTable, num_blocks), 8);
        assert_eq!(std::mem::offset_of!(MojoBlockTable, block_size), 16);
    }

    #[test]
    fn test_ffi_result_layout() {
        assert_eq!(std::mem::size_of::<FfiResult>(), 16);
        assert_eq!(std::mem::align_of::<FfiResult>(), 8);
        assert_eq!(std::mem::offset_of!(FfiResult, status_code), 0);
        assert_eq!(std::mem::offset_of!(FfiResult, error_message), 8);
    }

    #[test]
    fn test_ffi_result_helpers() {
        let ok_res = FfiResult::ok();
        assert!(ok_res.is_ok());
        assert_eq!(ok_res.status_code, 0);
        assert!(ok_res.error_message.is_null());

        let err_res = FfiResult::err(1, std::ptr::null());
        assert!(!err_res.is_ok());
        assert_eq!(err_res.status_code, 1);
    }
}
