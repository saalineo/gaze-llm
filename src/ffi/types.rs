//! Shared C ABI FFI type definitions for Rust and Mojo compute execution.

use std::ffi::c_char;

/// Supported tensor data types for Mojo compute kernels.
pub mod dtype {
    /// 32-bit floating point.
    pub const FP32: i32 = 0;
    /// 16-bit floating point.
    pub const FP16: i32 = 1;
    /// Brain floating point (16-bit).
    pub const BF16: i32 = 2;
    /// 8-bit signed integer.
    pub const INT8: i32 = 3;
}

/// Strongly-typed enumeration of tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DType {
    /// 32-bit floating point.
    Fp32 = dtype::FP32,
    /// 16-bit floating point.
    Fp16 = dtype::FP16,
    /// Brain floating point (16-bit).
    Bf16 = dtype::BF16,
    /// 8-bit signed integer.
    Int8 = dtype::INT8,
}

impl From<DType> for i32 {
    #[inline]
    fn from(dt: DType) -> Self {
        dt as Self
    }
}

/// Error type when converting an invalid integer into a [`DType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown dtype code: {0}")]
pub struct InvalidDTypeError(pub i32);

impl TryFrom<i32> for DType {
    type Error = InvalidDTypeError;

    #[inline]
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            dtype::FP32 => Ok(Self::Fp32),
            dtype::FP16 => Ok(Self::Fp16),
            dtype::BF16 => Ok(Self::Bf16),
            dtype::INT8 => Ok(Self::Int8),
            other => Err(InvalidDTypeError(other)),
        }
    }
}

/// C-compatible representation of a Mojo tensor memory buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MojoTensorBuffer {
    /// Pointer to the contiguous raw memory buffer.
    pub data_ptr: *mut std::ffi::c_void,
    /// Number of elements in the tensor.
    pub num_elements: usize,
    /// Byte size of each individual element.
    pub element_size_bytes: usize,
    /// Data type identifier (matching [`DType`] or [`dtype`]).
    pub dtype: i32,
}

/// C-compatible representation of a block table for `PagedAttention` memory management.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MojoBlockTable {


    /// Pointer to the block indices array.
    pub block_table_ptr: *mut i32,
    /// Total number of mapped blocks.
    pub num_blocks: usize,
    /// Token capacity per block.
    pub block_size: usize,
}

/// C-compatible status result returned from Mojo FFI kernel invocations.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FfiResult {
    /// Exit status code (0 for success, non-zero for error).
    pub status_code: i32,
    /// Pointer to null-terminated C string if an error occurred, or null on success.
    pub error_message: *const c_char,
}

impl FfiResult {
    /// Constructs a success result (`status_code == 0`) with a null error pointer.
    #[must_use]
    pub const fn ok() -> Self {
        Self {
            status_code: 0,
            error_message: std::ptr::null(),
        }
    }

    /// Constructs an error result with the specified non-zero status code and message pointer.
    #[must_use]
    pub const fn err(status_code: i32, message: *const c_char) -> Self {
        Self {
            status_code,
            error_message: message,
        }
    }

    /// Returns `true` if the operation succeeded (`status_code == 0`).
    #[must_use]
    #[inline]
    pub const fn is_ok(&self) -> bool {
        self.status_code == 0
    }

    /// Returns `true` if the operation failed (`status_code != 0`).
    #[must_use]
    #[inline]
    pub const fn is_err(&self) -> bool {
        self.status_code != 0
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
        let ok_result = FfiResult::ok();
        assert!(ok_result.is_ok());
        assert!(!ok_result.is_err());
        assert_eq!(ok_result.status_code, 0);
        assert!(ok_result.error_message.is_null());

        let err_result = FfiResult::err(1, std::ptr::null());
        assert!(!err_result.is_ok());
        assert!(err_result.is_err());
        assert_eq!(err_result.status_code, 1);
    }

    #[test]
    fn test_dtype_conversions() {
        assert_eq!(DType::try_from(0), Ok(DType::Fp32));
        assert_eq!(DType::try_from(1), Ok(DType::Fp16));
        assert_eq!(DType::try_from(2), Ok(DType::Bf16));
        assert_eq!(DType::try_from(3), Ok(DType::Int8));
        assert_eq!(DType::try_from(99), Err(InvalidDTypeError(99)));

        assert_eq!(i32::from(DType::Fp32), 0);
        assert_eq!(i32::from(DType::Int8), 3);
    }
}

