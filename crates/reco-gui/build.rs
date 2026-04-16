fn main() {
    let config = slint_build::CompilerConfiguration::new().with_style("material-dark".to_string());
    slint_build::compile_with_config("ui/main.slint", config).unwrap();
}
