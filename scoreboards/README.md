# Reco HTML scoreboards

Reco scoreboards are trusted local HTML/CSS/JavaScript packages rendered over the final camera image. Packages are discovered from the `scoreboards/` directory next to the Reco GUI executable, from the per-user data directory, or from `RECO_SCOREBOARDS_DIR`. Reco never interprets sport rules or hardcodes sport names.

The bundled `basketball/` package is a reference implementation, not part of the Rust API. See [DESIGNER_GUIDE.md](DESIGNER_GUIDE.md) to create a package and [AGENTS.md](AGENTS.md) for constraints that apply when an agent adds a sport.

## Runtime architecture

The first implementation uses an installed Chrome or Chromium executable through the Chrome DevTools Protocol. It was selected because it provides the same JavaScript and browser networking primitives on Windows, macOS, and Linux, supports transparent PNG capture, and adds no bundled Chromium binary to Reco releases. The browser runs on an independent worker and is started only while the overlay is enabled.

The worker observes DOM changes and active web animations, captures at no more than 30 fps, and publishes only changed RGBA surfaces. Video rendering never waits for the browser. The last surface remains in a cached wgpu texture and is composed after Stitching and Autocam, before the shared recording/stream encoding boundary.

Alternatives considered:

- Wry uses different system web engines and requires a native window/event loop; it does not expose one consistent cross-platform offscreen pixel API.
- CEF has a strong invalidation-driven offscreen API but would add a large binary payload and substantially more platform-specific build/packaging work.
- WebKitGTK is a Linux system dependency and does not solve the Windows/macOS engine path.

Chrome/Chromium must currently be installed. `CHROME` can point to a nonstandard executable. A later runtime can implement the same RGBA producer contract without changing manifests, scoreboards, or the GPU compositor.

## Security model

Scoreboard packages are executable content and are trusted to the same extent as locally installed scripts. Reco loads only validated local package entries; it does not accept arbitrary remote page URLs. While a scoreboard is active, a local-network server exposes only files contained by that package plus the embedded SDK, rejects traversal, and provides no directory listing. Packages with an optional editor may exchange generic JSON state through a dedicated endpoint protected by a random, runtime-only token. The QR button in Reco shows that temporary address so another device on the same Wi-Fi or LAN can control the scoreboard. Browser sandboxing stays enabled, and Reco exposes no process/filesystem APIs to JavaScript.

Normal browser security still applies. Network requests use standard CORS, mixed-content, and certificate rules. WebSocket and EventSource connections are available. Relative package assets work offline. Review community packages before installing them and do not install packages from an untrusted source.

## Installation locations

In priority order, Reco searches:

1. paths in `RECO_SCOREBOARDS_DIR`;
2. `scoreboards/` next to the executable;
3. `Contents/Resources/scoreboards/` for a macOS app bundle;
4. the platform user-data location (`Reco/scoreboards`);
5. the repository `scoreboards/` directory in development builds.

Invalid packages are skipped, logged, and reported in the GUI. IDs must remain unique across all discovered locations.
