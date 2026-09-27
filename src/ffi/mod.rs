//! Foreign Function Interface (FFI) bindings for external compute runtime (Mojo C-ABI).

pub mod types;

/// Scaffolding function for initializing FFI compute bindings.
pub fn init_ffi() {
    tracing::info!("Initializing FFI compute bindings");
}
