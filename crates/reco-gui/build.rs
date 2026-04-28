fn main() {
    let config = slint_build::CompilerConfiguration::new().with_style("fluent-dark".to_string());
    slint_build::compile_with_config("ui/main.slint", config).unwrap();

    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        let hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !hash.is_empty() {
            println!("cargo:rustc-env=GIT_HASH={hash}");
        }
    }
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
