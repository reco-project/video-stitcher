# Native FFmpeg SDK and packaging checks

`ffmpeg-sdk.json` selects the native source release, checksum and explicit Linux build features. Rust's `ffmpeg-next` version does not select the native library version. The SDK builder installs only into a new caller-provided directory; it does not modify system FFmpeg or drivers. Source builds of Reco can still use their own native libraries.

The scripts require Python 3.12+, a C/C++ toolchain, make, nasm and pkg-config. Linux development dependencies are x264, x265, aom, dav1d, GnuTLS, zlib, bzip2, xz, libdrm, libva and nv-codec headers. The packaging tests additionally require patchelf and GNU readelf. `build_ffmpeg_sdk.py` defaults to one compile job. Use `--archive` to reuse a downloaded source tarball; its checksum is always verified.

The recipe checks its pinned VA API and nv-codec header versions so a different build machine cannot silently raise graphics runtime requirements. Other dependency versions come from the build environment; this source/configuration pin is not a claim of bit-for-bit reproducibility. CI fingerprints the installed packages before caching the SDK.

```sh
python3 scripts/build_ffmpeg_sdk.py --work-dir /tmp/reco-ffmpeg-build --prefix /tmp/reco-ffmpeg-sdk
FFMPEG_DIR=/tmp/reco-ffmpeg-sdk LD_LIBRARY_PATH=/tmp/reco-ffmpeg-sdk/lib PATH=/tmp/reco-ffmpeg-sdk/bin:$PATH cargo test -p reco-io --no-default-features --features ffmpeg
python3 -m unittest discover -s scripts/tests -v
python3 scripts/bundle_linux.py --library-dir /tmp/reco-ffmpeg-sdk/lib --library-dir /usr/lib/x86_64-linux-gnu --output /tmp/reco-ffmpeg-bundle /tmp/reco-ffmpeg-sdk/bin/ffmpeg /tmp/reco-ffmpeg-sdk/bin/ffprobe
python3 scripts/verify_ffmpeg_bundle.py /tmp/reco-ffmpeg-bundle
```

`bundle_linux.py` inspects trusted ELF build outputs and copies their transitive DT_NEEDED dependencies from explicitly supplied library directories. It does not execute input binaries or search the current directory, loader environment or loader cache. Each staged ELF gets an `$ORIGIN` runpath. The output must not exist. The original SDK and binaries are unchanged. `bundle.json` records staged file hashes and required system-library names. The platform C runtime and graphics driver libraries remain external; this is not a fully static application or proof of compatibility with an older OS.

The receipt also records each staged ELF's glibc version references. `--max-glibc 2.39` rejects newer or nonnumeric/private references, including transitive dependencies; `test_bundle_linux.py` tests numeric ordering, dependency rejection and separation of required versus exported versions. This is a conservative static check, not an assertion that every machine with that glibc version supports the application or its dynamically loaded drivers. CI uses the Ubuntu 24.04 glibc ceiling so a dependency upgrade cannot silently raise it.

`test_bundle_linux.py` exercises transitive relocation after hiding the original build directory, missing libraries, architecture/SONAME mismatches, input preservation and external driver dependencies. `verify_ffmpeg_bundle.py` checks actual loader resolution, the pinned version, TLS/RTMPS protocol availability, PNG support and exact pixels through a short lossless encode/decode round trip. When the package includes the I/O lib test executable as `reco-io-tests`, the verifier also requires and runs the exact FIFO A/V recovery regression using the packaged libraries and tools. Protocol availability is not TLS certificate/authentication or server interoperability testing. These checks do not benchmark performance or certify GPU drivers.

When `reco` is staged, the verifier also runs an actual CPU cylindrical export of 12 distinct solid-colour frames. The generated input explicitly uses BT.709 limited-range YUV. Every decoded RGB channel must match its frame's expected colour within 8/255, checking ordering, frame count and exterior black pixels without treating codec rounding as a geometry failure. This is a packaging smoke test, not general source-colour, projection, ONNX Runtime or GUI acceptance.

`stage_linux_desktop.py` stages one CLI or GUI executable, FFmpeg tools/libraries and the Linux x86-64 CPU ONNX Runtime selected by `linux-desktop.json`. It verifies the CPU wheel and SDK source checksums, then bundles the native library without installing Python bindings. ONNX keeps the exact `libonnxruntime.so` name used by dynamic loading. The application license, FFmpeg source/build recipe/licenses and ONNX license/third-party notices are retained. Outputs must be fresh; failed collection does not publish partial staging.

`linux_notices.py` identifies every other copied native library through the installed Debian package database, checking the library against its package checksum before attributing notices or source versions. Unknown owners, diversions and modified libraries are rejected. `notices/dependencies.json` maps the copied closure to components and package identities; `notices/required-sources.json` lists exact distribution source versions. `test_linux_notices.py` covers rejection and source-download failure paths. The SDK and ONNX receipts are separate from distribution package ownership.

```sh
python3 scripts/stage_linux_desktop.py --sdk /tmp/reco-ffmpeg-sdk --ort-archive /tmp/onnxruntime.whl --library-dir /usr/lib/x86_64-linux-gnu --output /tmp/reco-desktop target/release/reco
python3 scripts/fetch_linux_sources.py /tmp/reco-desktop/notices/required-sources.json --output /tmp/reco-dependency-sources
```

`fetch_linux_sources.py` uses APT's authenticated source indexes and exact `package=version` selection in download-only mode. Configure matching `deb-src` indexes first, or use an isolated APT configuration via `APT_CONFIG`; the script does not enable repositories or install packages. Unavailable exact versions fail rather than selecting newer sources. Its source directory includes distribution patches/build files and a SHA256 receipt and must accompany the corresponding binary distribution. This is native dependency provenance, not a complete license audit of Reco's statically linked Rust dependencies or a claim of bit-for-bit reproducibility.

These tools do not by themselves publish or certify a desktop release. The release must also provide Reco's corresponding source/build inputs and applicable notices, define and test its OS support matrix and validate dynamically loaded platform components. This guide describes SDK CI and staging checks, not the release support matrix. Native macOS and Windows packaging require their own loader and acceptance checks.

The Linux x86-64 path in [Release](../.github/workflows/release.yml) uses the same [SDK setup](../.github/actions/linux-desktop-sdk/action.yml) as CI and requires both optimized application builds to succeed. It stages CLI and GUI separately, combines their exact native source requirements, and creates a matching source archive containing the Git revision, Rust toolchain identity, FFmpeg source/recipe, native distribution sources and Reco with `cargo vendor --locked` dependencies. Vendored dependency resolution is checked offline. Only tracked Reco source is archived, and a dirty tracked checkout fails the source-packaging step. No local ignored configuration, review files or build directories are included.

Before creating the three Linux archives, the release path hides its original SDK and checks relocated loader resolution, the CLI export oracle and mandatory CPU ONNX inference. GUI loader checks are not an interactive preview/export test, and these checks do not establish every graphics-driver combination. The source archive must remain distributed alongside the two binary archives. The native ARM cross-build, Windows and macOS paths use their own packaging recipes.

The `ffmpeg-sdk` job in [Rust CI](../.github/workflows/rust.yml) builds the pinned SDK and CLI on Ubuntu 24.04, stages notices and retrieves matching dependency sources, and tests the relocated runtime and CLI export while the original SDK path is unavailable. It explicitly executes the FIFO recovery regression and the mandatory CPU identity-model inference test in `ort_session::provider_tests`; the latter fails instead of skipping if ONNX cannot load. The verifier clears `ORT_*` and loader overrides. These checks do not certify GPU inference or GUI interaction. This runner choice defines the test environment, not a new minimum OS for all Reco consumers. The SDK cache includes the build recipe and installed package fingerprint.

## Windows cross-build inputs

`windows-sdk-sources.json` pins the Windows cross-toolchain image by digest and records the dependency source revisions, archive checksums and required notices. Its upstream build link records the recipe commit and exported image digest. The FFmpeg source pin is shared with `ffmpeg-sdk.json`. The compiler runtime sources, crosstool patches and upstream dependency build scripts accompany the codec sources; AMD and NVIDIA headers are included without redistributing graphics drivers.

```sh
python3 scripts/fetch_windows_sources.py --cache /tmp/reco-windows-source-cache --output /tmp/reco-windows-sources
```

The collector supports `--offline` for a populated cache. It verifies every archive before collecting regular notice files, retains the complete original archives and publishes only a complete output directory. It never follows source archive links when collecting notices. Required notice paths are explicit, including header-embedded NVIDIA notices. `test_windows_sources.py` covers modified archives, omitted or modified notice receipts, unsafe archive paths, missing inputs and preservation of existing outputs. Collected source archives are not executed by the collector.

Run `build_windows_ffmpeg_sdk.py` inside the manifest's Linux cross-toolchain image, with the scripts and collected sources mounted read-only and a separate writable work directory:

```sh
python3 /recipe/build_windows_ffmpeg_sdk.py --sources /sources --work-dir /work/build --prefix /work/sdk --jobs 2
```

The builder needs no network. It rechecks the source and notice receipts, installs the pinned NVIDIA headers in its private build directory, and uses explicit FFmpeg features rather than the image's broad default configuration. Host compiler/include flags are not inherited. MSVC import libraries are placed in `lib`, where `ffmpeg-sys-next` searches; runtime DLLs stay in `bin`. `share/reco-ffmpeg` contains notices, the source receipt, build configuration and hashes of SDK outputs. The separate collected source directory must accompany redistribution of this SDK.

The reusable [Windows SDK workflow](../.github/workflows/windows-sdk.yml) builds inside the digest-pinned image with no network, a read-only root, two CPUs and a 6 GiB memory limit. Cache identity includes both manifests and build/collection scripts. SDK and corresponding-source artifacts are separate. The [Windows setup action](../.github/actions/windows-desktop-sdk/action.yml) checks recipe identity, required import libraries and receipt-named bytes before exporting `FFMPEG_DIR`. Release and test builds use this shared path instead of selecting a floating upstream download.

`stage_windows_desktop.py` requires the application, verified SDK and the checksum-pinned ONNX Runtime/DirectML packages from `windows-desktop.json`. It copies the x64 runtime DLLs, FFmpeg tools, component licenses and notices into a fresh directory, recording output hashes in `files.json`. Only selected NuGet members are read, not the full multi-platform package. `test_windows_desktop.py` exercises missing DLLs/notices, corrupt SDK/archive bytes, absent offline runtimes and preservation of existing outputs. Both Windows applications must build and stage successfully in the release workflow; its matching FFmpeg source archive accompanies the binary archives. Intermediate SDK artifacts are excluded from release asset selection.

```sh
python scripts/stage_windows_desktop.py --sdk C:/sdk --verify-only
python scripts/stage_windows_desktop.py target/release/reco.exe --sdk C:/sdk --runtime-cache C:/runtime-cache --output C:/reco-package
```

Windows drivers and the Microsoft Visual C++ runtime remain platform prerequisites. Staging is not a transitive PE dependency audit, hardware acceptance, GUI interaction test or a complete license audit of statically linked Rust dependencies. These checks do not certify a desktop release or bit-for-bit reproducibility.

The [Windows package verification action](../.github/actions/verify-windows-desktop/action.yml) reuses `verify_ffmpeg_bundle.py` for codec/protocol checks and the CPU colour export oracle. It checks the packaged file receipt, restricts child `PATH` to the package and Windows system directory, and clears FFmpeg/ORT loader overrides. It also requires the native identity-model inference test to execute successfully with the packaged ONNX runtime. A GUI-only bundle exercises native runtimes but does not launch or validate the GUI; hardware decode/encode and interactive preview need native acceptance separately.

## Native macOS SDK and relocation

`build_macos_ffmpeg_sdk.py` uses the shared FFmpeg and codec source pins, building with Apple's compiler and a macOS 14.0 deployment target. It requires Python 3.12+, CMake, Ninja, Meson, Make and pkg-config; Intel builds also require NASM for codec assembly. Tools must already be available on `PATH`. The builder installs only into its requested fresh prefix and defaults to two build jobs. A populated source cache permits `--offline` builds.

```sh
python3 scripts/build_macos_ffmpeg_sdk.py --source-cache /tmp/reco-source-cache --work-dir /tmp/reco-sdk-build --prefix /tmp/reco-sdk --jobs 2
python3 scripts/bundle_macos.py --arch arm64 --max-macos 14.0 --library-dir /tmp/reco-sdk/lib --output /tmp/reco-ffmpeg-bundle /tmp/reco-sdk/bin/ffmpeg /tmp/reco-sdk/bin/ffprobe
python3 scripts/verify_ffmpeg_bundle.py /tmp/reco-ffmpeg-bundle
```

The SDK retains all selected source archives, notices, configure commands and output hashes under `share/reco-ffmpeg`. `verify_sdk` rechecks those inputs and required outputs before staging. `test_macos_sdk.py` covers mismatched recipes, architectures, missing outputs, modified archives/notices/libraries and escaping symlinks. This is source provenance and build-configuration validation, not bit-for-bit reproducibility across Apple toolchains.

`bundle_macos.py` accepts only explicitly trusted library directories. It checks each Mach-O's architecture and deployment target, gathers its non-system dependency closure, rewrites load commands to `@loader_path`, ad-hoc signs and verifies the results, and publishes a fresh directory with a hash receipt. `test_bundle_macos.py` covers rejected foreign paths, ambiguous dependencies, filename collisions, architecture/OS mismatches and atomic failure. On macOS, the shared runtime verifier clears loader overrides and checks actual dyld paths during its codec/protocol and lossless-roundtrip checks. Ad-hoc signing is not Developer ID signing, notarization, Gatekeeper acceptance or an application bundle.

`macos-desktop.json` pins the Apple Silicon ONNX Runtime archive and the matching source release for Intel. `build_macos_onnxruntime.py` builds the x86-64 shared library with upstream's CMake recipe, disabling package discovery for fetched dependencies and retaining the pinned source archive and third-party notices. Upstream fetches dependencies with the hashes in its pinned `cmake/deps.txt`, so this build needs network access. The builder neither installs Python packages nor modifies system tools. `test_macos_onnxruntime.py` checks destination safety, failure atomicity and the selected Intel configuration.

```sh
python3 scripts/build_macos_onnxruntime.py --source-cache /tmp/reco-source-cache --work-dir /tmp/reco-ort-build --output /tmp/reco-ort --jobs 2
```

The Intel runtime can be cross-built on Apple Silicon, but architecture inspection does not replace native inference and application acceptance. Runtime staging must preserve `libonnxruntime.dylib`, its license and third-party notices. Relocation checks do not establish an OS support matrix or certify GUI interaction.
