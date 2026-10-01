//! Foreign Function Interface (FFI) bindings for external compute runtime (Mojo C-ABI).

pub mod bridge;
pub mod dlopen;
pub mod types;

pub use bridge::{safe_mojo_execute, FfiBridgeError};
pub use dlopen::{MojoKernelFn, MojoLibrary, MojoLibraryError};
pub use types::{dtype, DType, FfiResult, InvalidDTypeError, MojoBlockTable, MojoTensorBuffer};

/// Scaffolding function for initializing FFI compute bindings.
pub fn init_ffi() {
    tracing::info!("Initializing FFI compute bindings");
}

