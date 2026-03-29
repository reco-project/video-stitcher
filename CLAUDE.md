# Reco Video Stitcher

Open-source GPU-accelerated panoramic sports camera software.

## v2 Architecture (Rust + wgpu) — active development on `v2` branch

- `crates/reco-core/` — GPU stitching engine (library crate)
- `crates/reco-cli/` — CLI binary (`reco stitch`, `reco info`)
- `crates/reco-io/` — Pluggable I/O backends (FFmpeg decode/encode, GStreamer live cameras)

## Key commands

```bash
cargo build                   # Build all crates
cargo test --all              # Run all tests
cargo clippy --all-targets -- -D warnings   # Lint
cargo fmt --all -- --check    # Format check
cargo fmt --all               # Auto-format
cargo doc --no-deps --open    # Generate and open docs
cargo run -p reco-cli -- info # Show GPU info
cargo run -p reco-cli -- stitch left.mp4 right.mp4 -c match.json -o out.mp4
cargo run -p reco-cli -- preview left.mp4 right.mp4 -c match.json
cargo run --release -p reco-cli --features profiling -- stitch left.mp4 right.mp4 -c match.json -o out.mp4 --max-frames 300  # Profile 300 frames → reco-trace.json (open in ui.perfetto.dev)
```

## Code standards

- `rustfmt` formatting (config in `rustfmt.toml`)
- `clippy` linting with `-D warnings` (zero warnings policy)
- Doc comments (`///`) on all public items
- Module-level docs (`//!`) explaining purpose
- Tests in each module (`#[cfg(test)] mod tests`)
- All PRs must pass: `cargo fmt --check && cargo clippy && cargo test`
- Clippy must also pass with `--features profiling`
- `profiling` feature: opt-in `tracing` + `tracing-chrome` instrumentation (zero-cost when off)

## Context
- Public open-source project (AGPL-3.0) with a growing community and forum
- Users include football clubs, amateur sports teams — prioritize UX clarity
- Open alternative to proprietary sports camera solutions
- v2 targets: desktop (Win/macOS/Linux), NVIDIA Jetson, cloud, mobile

## When writing code
- Production-grade: handle errors, validate inputs at API boundaries
- Cross-platform (Windows/macOS/Linux) — avoid platform-specific assumptions
- Performance matters: this processes video frames in real time
- Modular: reco-core must be usable as a standalone Rust crate
- Explicit over implicit: no hidden defaults, no magic
