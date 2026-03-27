# Reco Video Stitcher

Open-source GPU-accelerated panoramic sports camera software.

## v2 Architecture (Rust + wgpu) — active development on `v2` branch

- `crates/reco-core/` — GPU stitching engine (library crate)
- `crates/reco-cli/` — CLI binary (`reco stitch`, `reco info`)
- `crates/reco-ffmpeg/` — FFmpeg decode/encode plugin

## v1 Architecture (Electron + React + Python) — legacy, on `main` branch

- `electron/` — Electron shell, packaging, IPC
- `frontend/` — React UI (Vite)
- `backend/` — FastAPI Python server (video processing, stitching)
- `scripts/` — build/dev utilities

## Key commands

### v2 (Rust)
```bash
cargo build                   # Build all crates
cargo test --all              # Run all tests
cargo clippy --all-targets -- -D warnings   # Lint
cargo fmt --all -- --check    # Format check
cargo fmt --all               # Auto-format
cargo doc --no-deps --open    # Generate and open docs
cargo run -p reco-cli -- info # Show GPU info
cargo run -p reco-cli -- stitch left.mp4 right.mp4 -c match.json -o out.mp4
```

### v1 (Node/Python)
```bash
npm run dev              # Start full stack (electron + frontend + backend)
npm run backend-dev      # FastAPI only
npm run frontend-dev     # React only
npm run build            # Production build
npm run test             # Run all tests
npm run format           # Format all (JS + Python)
npm run lint             # Lint JS
```

## Code standards

### Rust (v2)
- `rustfmt` formatting (config in `rustfmt.toml`)
- `clippy` linting with `-D warnings` (zero warnings policy)
- Doc comments (`///`) on all public items
- Module-level docs (`//!`) explaining purpose
- Tests in each module (`#[cfg(test)] mod tests`)
- All PRs must pass: `cargo fmt --check && cargo clippy && cargo test`

### JavaScript/Python (v1)
- Python: black formatting, type hints required on public functions
- JS/React: ESLint config in repo, follow existing component patterns
- GPU/OpenCV code: comment non-obvious stitching math

## Context
- Public open-source project (AGPL-3.0) with a growing community and forum
- Users include football clubs, amateur sports teams — prioritize UX clarity
- Competitors: Veo, Pixellot (proprietary, expensive) — Reco is the open alternative
- v2 targets: desktop (Win/macOS/Linux), NVIDIA Jetson, cloud, mobile
- Always use `sudo -A` for any system commands

## When writing code
- Production-grade: handle errors, validate inputs at API boundaries
- Cross-platform (Windows/macOS/Linux) — avoid platform-specific assumptions
- Performance matters: this processes video frames in real time
- Modular: reco-core must be usable as a standalone Rust crate
- Explicit over implicit: no hidden defaults, no magic
