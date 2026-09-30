//! Foreign Function Interface (FFI) bindings for external compute runtime (Mojo C-ABI).

pub mod bridge;
pub mod dlopen;
pub mod types;

pub use bridge::safe_mojo_execute;
pub use dlopen::MojoLibrary;

/// Scaffolding function for initializing FFI compute bindings.
pub fn init_ffi() {
    tracing::info!("Initializing FFI compute bindings");
}
