SHELL := /bin/zsh

.PHONY: help mac-intel-init mac-intel-doctor gui-build-mac-intel gui-run-mac-intel

help:
	@echo "Reco Video Stitcher Make targets"
	@echo "  make mac-intel-init      Install Intel macOS build dependencies"
	@echo "  make mac-intel-doctor    Print toolchain and pkg-config status"
	@echo "  make gui-build-mac-intel Build reco-gui release binary (no ORT defaults)"
	@echo "  make gui-run-mac-intel   Run reco-gui release binary"

mac-intel-init:
	@command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
	brew update
	brew install ffmpeg pkgconf rustup-init
	@RUSTUP_BIN="$$(command -v rustup 2>/dev/null || true)"; \
	if [ -z "$$RUSTUP_BIN" ] && [ -x "$$(brew --prefix rustup)/bin/rustup" ]; then \
		RUSTUP_BIN="$$(brew --prefix rustup)/bin/rustup"; \
	fi; \
	if [ -z "$$RUSTUP_BIN" ] && [ -x "$$HOME/.cargo/bin/rustup" ]; then \
		RUSTUP_BIN="$$HOME/.cargo/bin/rustup"; \
	fi; \
	if [ -z "$$RUSTUP_BIN" ]; then \
		echo "rustup is not available in PATH. Add $$(brew --prefix rustup)/bin to PATH and retry."; \
		exit 1; \
	fi; \
	"$$RUSTUP_BIN" toolchain install 1.92.0; \
	"$$RUSTUP_BIN" override set 1.92.0
	@echo "Intel macOS environment initialized."

mac-intel-doctor:
	@echo "=== Toolchain ==="
	@rustc --version || true
	@cargo --version || true
	@if command -v rustup >/dev/null; then \
		rustup show active-toolchain; \
	elif [ -x "$$(brew --prefix rustup)/bin/rustup" ]; then \
		"$$(brew --prefix rustup)/bin/rustup" show active-toolchain; \
	elif [ -x "$$HOME/.cargo/bin/rustup" ]; then \
		"$$HOME/.cargo/bin/rustup" show active-toolchain; \
	else \
		echo "rustup not found in PATH"; \
	fi
	@echo "=== Brew ==="
	@brew --version | head -n 1 || true
	@echo "=== FFmpeg pkgconfig path ==="
	@echo "$$(brew --prefix ffmpeg)/lib/pkgconfig"
	@echo "=== pkg-config libavutil ==="
	@PKG_CONFIG_PATH="$$(brew --prefix ffmpeg)/lib/pkgconfig:$${PKG_CONFIG_PATH}" \
		pkg-config --modversion libavutil || { echo "libavutil not found (run: make mac-intel-init)"; exit 1; }

gui-build-mac-intel:
	@RUSTUP_BIN="$$(command -v rustup 2>/dev/null || true)"; \
	if [ -z "$$RUSTUP_BIN" ] && [ -x "$$(brew --prefix rustup)/bin/rustup" ]; then \
		RUSTUP_BIN="$$(brew --prefix rustup)/bin/rustup"; \
	fi; \
	if [ -z "$$RUSTUP_BIN" ] && [ -x "$$HOME/.cargo/bin/rustup" ]; then \
		RUSTUP_BIN="$$HOME/.cargo/bin/rustup"; \
	fi; \
	if [ -z "$$RUSTUP_BIN" ]; then \
		echo "rustup is required (brew install rustup-init)."; \
		exit 1; \
	fi; \
	PKG_CONFIG_PATH="$$(brew --prefix ffmpeg)/lib/pkgconfig:$${PKG_CONFIG_PATH}" \
		"$$RUSTUP_BIN" run 1.92.0 cargo build --release -p reco-gui --no-default-features

gui-run-mac-intel: gui-build-mac-intel
	./target/release/reco-gui
