extern crate bindgen;
extern crate cc;

use std::env;
use std::path::PathBuf;

fn generate_bindings() {
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=src/gpu/cuda/gpu_test.cu");
    println!("cargo:rerun-if-changed=src/gpu/cuda/fields/fft.cuh");
    println!("cargo:rerun-if-changed=src/gpu/cuda/fields/field_structs.hpp");
    println!("cargo:rerun-if-changed=src/gpu/cuda/fields/field_types.cuh");
    println!("cargo:rerun-if-changed=src/gpu/cuda/fields/fields.cuh");
    println!("cargo:rerun-if-changed=src/gpu/cuda/fields/multiexp.cuh");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/gpu/cuda/interface.hpp")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn compile_libraries() {
    if cfg!(feature = "cuda") {
        compile_cuda_libraries();
    }
}

// cuda toolkit 11 build will never finish with profile release.
// Use cuda toolkit 10 instead, which comes with Ubuntu 20.04.
#[allow(dead_code)]
fn compile_cuda_libraries() {
    // By default, cc add `-ccbin=c++` argument to nvcc. This is undesirable for high version gcc,
    // as cuda usually does not support the latest version of gcc.
    // https://stackoverflow.com/questions/49342835/changing-the-compilation-arguments-passed-to-nvcc-by-rust-using-cc
    env::set_var("CXX", env::var("CXX").unwrap_or("cuda-g++".to_string()));
    cc::Build::new()
        .cuda(true)
        .file("src/gpu/cuda/gpu_test.cu")
        .flag("-lcuda")
        .flag("-Xptxas")
        .flag("-v")
        //.flag("-t16")
        .flag("-std=c++11")
        .flag("-gencode=arch=compute_75,code=compute_75")
        .flag("--maxrregcount=128")
        .include("src/gpu/cuda/fields")
        .compile("bellpersoncuda");
    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
}

fn main() {
    generate_bindings();
    compile_libraries();
}
