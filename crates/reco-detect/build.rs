fn main() {
    // Only compile the TensorRT C++ wrapper and link native libraries
    // when the `tensorrt-native` feature is enabled. On systems without
    // TensorRT (CI runners, desktops without NVIDIA GPU), the feature is
    // off and this build script is a no-op.
    if std::env::var("CARGO_FEATURE_TENSORRT_NATIVE").is_ok() {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .file("csrc/tensorrt_wrapper.cpp")
            .include("csrc")
            .flag("-std=c++14")
            .flag("-w"); // Suppress deprecation warnings from NVIDIA headers

        // CUDA include paths (check standard locations)
        let cuda_include_paths = [
            "/opt/cuda/include",       // Arch Linux
            "/usr/local/cuda/include", // Ubuntu / Jetson
            "/usr/include",            // System-wide
        ];
        for path in &cuda_include_paths {
            if std::path::Path::new(path)
                .join("cuda_runtime_api.h")
                .exists()
            {
                build.include(path);
                break;
            }
        }

        // Allow override via CUDA_HOME
        if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
            build.include(format!("{cuda_home}/include"));
        }

        build.compile("tensorrt_wrapper");

        println!("cargo:rustc-link-lib=nvinfer");
        println!("cargo:rustc-link-lib=cudart");

        // Standard library search paths (Jetson + desktop)
        println!("cargo:rustc-link-search=/usr/lib/aarch64-linux-gnu");
        println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=/opt/cuda/lib64");

        // Allow override via environment variables
        if let Ok(trt_lib) = std::env::var("TENSORRT_LIB_DIR") {
            println!("cargo:rustc-link-search={trt_lib}");
        }
        if let Ok(cuda_lib) = std::env::var("CUDA_LIB_DIR") {
            println!("cargo:rustc-link-search={cuda_lib}");
        }
    }

    println!("cargo:rerun-if-changed=csrc/tensorrt_wrapper.cpp");
    println!("cargo:rerun-if-changed=csrc/tensorrt_wrapper.h");
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
