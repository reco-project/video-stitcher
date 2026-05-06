# wgpu 29 Metal Interop Reference

The `crates/reco-core/src/interop/metal.rs` and
`crates/reco-detect/src/metal_compute.rs` modules were ported from
the objc2-metal API (wgpu 29-era) to the metal crate 0.33 API
(wgpu 28) in PR #217 as part of the wgpu version downgrade.

## Why this doc exists

The two API shapes are non-obviously different (different type names,
different call conventions, different ownership semantics). When we
eventually move back to wgpu 29+ (likely after Slint upstream lands
`unstable-wgpu-29` — see slint-ui/slint#11378), we'll need to port
these modules back.

Rather than lose the old objc2-metal code to git history alone, this
file points at the exact commits so the next port is a straightforward
reversal.

## Where to find the wgpu 29 code

The last commit with the full objc2-metal-based implementation is
just before commit `9160c51` ("spike: downgrade wgpu 29 fork to
wgpu 28 from crates.io"). To recover it:

```bash
# View the old code:
git show 9160c51^:crates/reco-core/src/metal_interop.rs
git show 9160c51^:crates/reco-core/src/metal_compute.rs

# Or restore on a branch:
git checkout -b restore-wgpu29-metal 9160c51^ -- \
    crates/reco-core/src/metal_interop.rs \
    crates/reco-core/src/metal_compute.rs
```

Earlier reference points (in case 9160c51 is squashed out):
- `c5d07a1` — "pre-v2 release hardening" (late wgpu 29 state)
- `952c69a` — "feat: Metal GPU detection pipeline for macOS zero-copy path"
- `714e79f` — "feat: add macOS Metal/VideoToolbox zero-copy decode interop"

## Key API differences (summary)

| Concern | wgpu 28 (metal crate 0.33) | wgpu 29 (objc2-metal 0.3) |
|---|---|---|
| Device type | `metal::Device` / `&metal::DeviceRef` | `Retained<ProtocolObject<dyn MTLDevice>>` |
| Method style | snake_case (`new_command_queue()`) | objc-style (`newCommandQueue()`) |
| Shader source | `&str` direct | `NSString::from_str(&str)` |
| Error handling | `Result<T, String>` | `Result<T, Retained<NSError>>` |
| Retain / clone | `texture_ref.to_owned()` | `Retained::retain(ptr)` |
| Raw pointer | `.as_ptr()` on ForeignTypeRef | `Retained::as_ptr(&retained)` |
| HAL raw handle | `&metal::Texture` | `&ProtocolObject<dyn MTLTexture>` |
| set_texture signature | `(index, Option<&TextureRef>)` | `(Option<&...>, index)` (swapped) |
| MTLTextureType variant | `MTLTextureType::D2` | `MTLTextureType::Type2D` |

## Related FRICTION docs

See `crates/reco-gui/FRICTION.md` #12 (Slint `Image::try_from`
ownership), which is the upstream blocker keeping us on wgpu 28.
Once Slint supports wgpu 29, upgrading goes: bump Slint features,
bump reco-core wgpu, restore these two files from git, drop the
`metal` + `foreign-types` deps from reco-core's `[target.'cfg(
target_os = "macos"))]'.dependencies]`.
