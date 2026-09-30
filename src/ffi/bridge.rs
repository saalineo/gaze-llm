use crate::ffi::types::{FfiResult, MojoTensorBuffer};
use std::ffi::CStr;

extern "C" {
    pub fn mojo_execute_kernel(
        input: *const MojoTensorBuffer,
        output: *mut MojoTensorBuffer,
    ) -> FfiResult;
}

/// Executes a Mojo compute kernel across the C-ABI FFI boundary.
pub fn safe_mojo_execute(
    input: &MojoTensorBuffer,
    output: &mut MojoTensorBuffer,
) -> Result<(), String> {
    // SAFETY: Coercing valid Rust references to raw pointers for the C-ABI call.
    let kernel_result = unsafe { mojo_execute_kernel(input, output) };
    if kernel_result.is_ok() {
        return Ok(());
    }

    if !kernel_result.error_message.is_null() {
        // SAFETY: Error pointer is non-null and points to a valid null-terminated C string.
        let msg = unsafe { CStr::from_ptr(kernel_result.error_message) };
        Err(msg.to_string_lossy().into_owned())
    } else {
        Err("Unknown Mojo execution error".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::ffi::CString;

    thread_local! {
        static MOCK_STATUS: Cell<i32> = const { Cell::new(0) };
        static MOCK_ERROR_MSG: RefCell<Option<CString>> = const { RefCell::new(None) };
    }

    #[no_mangle]
    pub extern "C" fn mojo_execute_kernel(
        input: *const MojoTensorBuffer,
        output: *mut MojoTensorBuffer,
    ) -> FfiResult {
        if input.is_null() || output.is_null() {
            return FfiResult::err(1, c"Null buffer pointer".as_ptr());
        }

        let status = MOCK_STATUS.with(|s| s.get());
        if status == 0 {
            FfiResult::ok()
        } else {
            let msg_ptr = MOCK_ERROR_MSG.with(|m| {
                m.borrow()
                    .as_ref()
                    .map_or(std::ptr::null(), |cstr| cstr.as_ptr())
            });
            FfiResult::err(status, msg_ptr)
        }
    }

    fn set_mock_behavior(status: i32, msg: Option<&str>) {
        MOCK_STATUS.with(|s| s.set(status));
        MOCK_ERROR_MSG.with(|m| {
            *m.borrow_mut() = msg.map(|s| CString::new(s).unwrap());
        });
    }

    fn dummy_buffer() -> MojoTensorBuffer {
        MojoTensorBuffer {
            data_ptr: std::ptr::null_mut(),
            num_elements: 10,
            element_size_bytes: 4,
            dtype: 0,
        }
    }

    #[test]
    fn test_safe_mojo_execute_success() {
        set_mock_behavior(0, None);
        let input = dummy_buffer();
        let mut output = dummy_buffer();
        assert!(safe_mojo_execute(&input, &mut output).is_ok());
    }

    #[test]
    fn test_safe_mojo_execute_error_with_message() {
        set_mock_behavior(42, Some("CUDA out of memory in Mojo kernel"));
        let input = dummy_buffer();
        let mut output = dummy_buffer();
        assert_eq!(
            safe_mojo_execute(&input, &mut output).unwrap_err(),
            "CUDA out of memory in Mojo kernel"
        );
    }

    #[test]
    fn test_safe_mojo_execute_error_null_message() {
        set_mock_behavior(-1, None);
        let input = dummy_buffer();
        let mut output = dummy_buffer();
        assert_eq!(
            safe_mojo_execute(&input, &mut output).unwrap_err(),
            "Unknown Mojo execution error"
        );
    }
}
