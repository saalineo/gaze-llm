fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure().compile(&["proto/engine.proto"], &["proto"])?;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    println!("cargo:rustc-link-search=native={}/target", manifest_dir);
    println!("cargo:rustc-link-lib=dylib=mojo_kernel");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/target", manifest_dir);
    println!("cargo:rerun-if-changed=mojo/entry.mojo");
    println!("cargo:rerun-if-changed=mojo/types.mojo");

    Ok(())
}
