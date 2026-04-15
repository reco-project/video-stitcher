# Contributing to Reco Video Stitcher

Thank you for your interest in contributing! This document covers the process for contributing code, reporting issues, and the legal terms.

## Getting Started

1. Fork the repository
2. Clone your fork and create a branch: `git checkout -b feat/your-feature` or `fix/your-bug`
3. Install dependencies: FFmpeg dev libraries, Rust stable toolchain
4. Build: `cargo build`
5. Test: `cargo test --all`

## Development

### Before submitting

```bash
cargo fmt --all               # Format
cargo clippy --all-targets -- -D warnings   # Lint (zero warnings)
cargo test --all              # Tests
cargo doc --no-deps           # Documentation builds
```

### Branch naming

- `feat/short-description` for new features
- `fix/short-description` for bug fixes
- `refactor/short-description` for refactoring

### Commits

Write clear commit messages. Use conventional prefixes:
- `feat:` new feature
- `fix:` bug fix
- `refactor:` code restructuring
- `chore:` maintenance (deps, CI, docs)
- `perf:` performance improvement

### Pull requests

- One PR per feature/fix
- PRs must pass CI (fmt, clippy, test, doc)
- Squash merge to main
- Reference the issue number: `Closes #123`

## Architecture

The project is a Cargo workspace with six crates:

- **reco-core** - GPU stitching engine (library, no I/O deps)
- **reco-cli** - CLI binary
- **reco-io** - FFmpeg/GStreamer/libcamera backends
- **reco-detect** - AI detection (ORT, TensorRT, NCNN, CoreML)
- **reco-autocam** - AI camera control
- **reco-calibrate** - Stereo calibration

`reco-core` must remain a pure library with no I/O dependencies. Detection, encoding, and camera backends live in their respective crates.

## Reporting Issues

- Use GitHub Issues
- Include: steps to reproduce, expected vs actual behavior, camera model, OS
- For performance issues: include frame count, resolution, GPU model, fps numbers

## Contributor License Agreement

By submitting a pull request, you agree to the following terms:

1. You grant Mohamed Taha GUELZIM (the project maintainer) a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license to use, reproduce, modify, distribute, sublicense, and relicense your contribution in any form, including under proprietary licenses.

2. You represent that you have the right to grant this license and that your contribution is your original work.

3. This agreement allows the maintainer to offer the software under dual licensing (AGPL-3.0 for open source, commercial license for proprietary use) without requiring further permission from contributors.

4. Your contribution remains attributed to you in the git history.

By opening a pull request, you acknowledge that you have read and agree to these terms.

## License

This project is licensed under [AGPL-3.0](LICENSE).
