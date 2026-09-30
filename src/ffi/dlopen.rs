//! Dynamic library loader for runtime hot-swapping of Mojo compute kernels.

use crate::ffi::types::{FfiResult, MojoTensorBuffer};
use libloading::{Library, Symbol};
use std::ffi::CStr;
use std::path::Path;
use std::sync::Arc;

pub type MojoKernelFn =
    unsafe extern "C" fn(*const MojoTensorBuffer, *mut MojoTensorBuffer) -> FfiResult;

/// RAII wrapper managing the lifecycle and dynamic symbol table of a compiled Mojo shared object.
pub struct MojoLibrary {
    _lib: Library,
    pub execute_kernel: Symbol<'static, MojoKernelFn>,
}

impl MojoLibrary {
    /// Dynamically loads a Mojo shared object (`.so` / `.dylib`) from the given path
    /// and resolves the entry kernel symbol `mojo_execute_kernel`.
    pub fn load<P: AsRef<Path>>(
        path: P,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
        let path_ref = path.as_ref();
        // SAFETY: Loading external shared objects assumes standard C-ABI calling conventions
        // and matching `MojoTensorBuffer` / `FfiResult` layouts.
        let lib = unsafe { Library::new(path_ref)? };
        let kernel_symbol: Symbol<MojoKernelFn> = unsafe { lib.get(b"mojo_execute_kernel\0")? };

        // SAFETY: The symbol pointer remains valid for the lifetime of `_lib`,
        // which is co-owned inside the returned `MojoLibrary` struct.
        let execute_kernel_static: Symbol<'static, MojoKernelFn> =
            unsafe { std::mem::transmute(kernel_symbol) };

        Ok(Arc::new(Self {
            _lib: lib,
            execute_kernel: execute_kernel_static,
        }))
    }

    /// Safely invokes the dynamically loaded `mojo_execute_kernel` symbol.
    pub fn execute(
        &self,
        input: &MojoTensorBuffer,
        output: &mut MojoTensorBuffer,
    ) -> Result<(), String> {
        let kernel_result = unsafe { (self.execute_kernel)(input, output) };
        if kernel_result.is_ok() {
            return Ok(());
        }

        if !kernel_result.error_message.is_null() {
            let error_msg = unsafe { CStr::from_ptr(kernel_result.error_message) };
            Err(error_msg.to_string_lossy().into_owned())
        } else {
            Err("Unknown Mojo execution error".to_string())
        }
    }
}

// SAFETY: `MojoLibrary` owns the loaded library mapping and immutable symbol pointer.
// Function symbols in compiled Mojo shared objects are re-entrant and thread-safe.
unsafe impl Send for MojoLibrary {}
unsafe impl Sync for MojoLibrary {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::raw_buffer::RawMemoryBuffer;
    use std::path::PathBuf;
    use std::thread;

    fn get_mojo_lib_path() -> PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest_dir).join("target/libmojo_kernel.so")
    }

    #[test]
    fn test_dlopen_valid_library() {
        let lib_path = get_mojo_lib_path();
        if !lib_path.exists() {
            return;
        }

        let mojo_lib = MojoLibrary::load(&lib_path).expect("Failed to load Mojo library");
        let input_buf = RawMemoryBuffer::allocate(1024, 64).expect("Alloc failed");
        let output_buf = RawMemoryBuffer::allocate(1024, 64).expect("Alloc failed");

        let in_tensor = MojoTensorBuffer {
            data_ptr: input_buf.as_mut_ptr() as *mut _,
            num_elements: 256,
            element_size_bytes: 4,
            dtype: 0,
        };
        let mut out_tensor = MojoTensorBuffer {
            data_ptr: output_buf.as_mut_ptr() as *mut _,
            num_elements: 256,
            element_size_bytes: 4,
            dtype: 0,
        };

        let kernel_status = unsafe { (mojo_lib.execute_kernel)(&in_tensor, &mut out_tensor) };
        assert_eq!(kernel_status.status_code, 0);
    }

    #[test]
    fn test_dlopen_execute_method() {
        let lib_path = get_mojo_lib_path();
        if !lib_path.exists() {
            return;
        }

        let mojo_lib = MojoLibrary::load(&lib_path).expect("Failed to load Mojo library");
        let input_buf = RawMemoryBuffer::allocate(512, 64).expect("Alloc failed");
        let output_buf = RawMemoryBuffer::allocate(512, 64).expect("Alloc failed");

        let in_tensor = MojoTensorBuffer {
            data_ptr: input_buf.as_mut_ptr() as *mut _,
            num_elements: 128,
            element_size_bytes: 4,
            dtype: 0,
        };
        let mut out_tensor = MojoTensorBuffer {
            data_ptr: output_buf.as_mut_ptr() as *mut _,
            num_elements: 128,
            element_size_bytes: 4,
            dtype: 0,
        };

        let execution_result = mojo_lib.execute(&in_tensor, &mut out_tensor);
        assert!(execution_result.is_ok());
    }

    #[test]
    fn test_dlopen_nonexistent_path_fails() {
        let load_result = MojoLibrary::load("target/nonexistent_lib_invalid.so");
        assert!(load_result.is_err());
    }

    #[test]
    fn test_dlopen_missing_symbol_fails() {
        let libc_candidates = ["libc.so.6", "/usr/lib/libc.so.6", "/lib/x86_64-linux-gnu/libc.so.6"];
        for candidate in &libc_candidates {
            if Path::new(candidate).exists() || *candidate == "libc.so.6" {
                if let Ok(lib) = unsafe { Library::new(candidate) } {
                    let missing_sym: Result<Symbol<MojoKernelFn>, _> =
                        unsafe { lib.get(b"mojo_execute_kernel\0") };
                    assert!(missing_sym.is_err());
                    break;
                }
            }
        }
    }

    #[test]
    fn test_dlopen_concurrent_execution() {
        let lib_path = get_mojo_lib_path();
        if !lib_path.exists() {
            return;
        }

        let mojo_lib = MojoLibrary::load(&lib_path).expect("Failed to load Mojo library");
        let mut thread_handles = Vec::new();

        for _ in 0..4 {
            let shared_lib = Arc::clone(&mojo_lib);
            thread_handles.push(thread::spawn(move || {
                let input_buf = RawMemoryBuffer::allocate(256, 64).expect("Alloc failed");
                let output_buf = RawMemoryBuffer::allocate(256, 64).expect("Alloc failed");

                let in_tensor = MojoTensorBuffer {
                    data_ptr: input_buf.as_mut_ptr() as *mut _,
                    num_elements: 64,
                    element_size_bytes: 4,
                    dtype: 0,
                };
                let mut out_tensor = MojoTensorBuffer {
                    data_ptr: output_buf.as_mut_ptr() as *mut _,
                    num_elements: 64,
                    element_size_bytes: 4,
                    dtype: 0,
                };

                let execution_result = shared_lib.execute(&in_tensor, &mut out_tensor);
                assert!(execution_result.is_ok());
            }));
        }

        for handle in thread_handles {
            handle.join().expect("Thread join failed");
        }
    }
}
