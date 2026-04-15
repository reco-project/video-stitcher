# Deep Review Summary - 2026-03-28

**Branch:** v2
**Baseline:** clippy clean, fmt clean, 11 unit tests + 5 doctests passing
**Lint suppressions:** 2 (`#[allow(clippy::too_many_arguments)]` in encoder.rs and main.rs)
**Methodology:** 11 parallel subagents (6 automated scans + 5 adversarial reviewers)

## Findings by Severity

| Severity | Count | Issues |
|----------|-------|--------|
| Critical | 8 | #42, #43, #44, #45, #46, #47, #48, #49 |
| High | 8 | #50, #51, #52, #53, #54, #55, #56, #57, #58, #59, #60 |
| Medium | 6 | #61, #62, #63, #64, #65, #66 |
| Low | 1 | #67 |
| **Total** | **20 issues** | Milestone: `deep-review-2026-03-28` |

## Top 5 Actions (by impact)

1. **Fix x264 encoder tuning** (#49) - Add `tune=zerolatency`, increase CRF, use 4 threads. Estimated 20-35% encode speedup on Jetson. Easiest high-impact change.

2. **Cache bind groups** (#46) - Per-frame bind group creation will OOM on Vulkan after ~60 seconds. Fix: create once in `Nv12Converter::new()`. Quick fix, critical severity.

3. **Eliminate GPU shader waste** (#48) - sRGB round-trip, unconditional color correction, unnecessary depth buffer. Combined 2-5ms/frame savings at 4K. Moderate effort.

4. **Fix Surface transmute** (#43) - Use `Arc<Window>` for safe lifetime. Use-after-free on exit. Quick fix.

5. **Fix aarch64 == Jetson assumption** (#45) - Replace `cfg!(target_arch = "aarch64")` with runtime `is_jetson()`. Would break Apple Silicon. Quick fix.

## Performance Budget (4K@30fps on Jetson Orin Nano Super)

Current: 29.9fps (33.4ms/frame)

| Optimization | Expected savings | Issue |
|-------------|-----------------|-------|
| x264 tuning | 10-20ms | #49 |
| sRGB + depth + LAB elimination | 2-5ms | #48 |
| Per-frame allocations | ~2ms | #55 |
| GPU pipeline overlap | ~2ms | #65 |
| Frame dropping (latency) | 0ms (latency fix) | #60 |
| **Total recoverable** | **16-29ms** | |
| **Projected frame time** | **4.4-17.4ms** | |
| **Projected fps** | **57-227fps** | |

The pipeline is far from hardware limits. The x264 encoder is the dominant bottleneck, and the GPU shader waste is significant.

## Architecture Assessment

- **Extensibility: 3/10** - Trait interfaces exist but are bypassed in all hot paths
- **FrameSource** can't represent NV12 or GPU-resident frames (#53)
- **CLI is coupled to GPU internals** via pub fields and direct wgpu dependency (#54)
- **Code duplication** between stitch/camera commands and between CLI/reco-io (#56, #62)
- **Test coverage**: reco-core has 11 unit tests, reco-io and reco-cli have zero (#56)

## Security Assessment

- **1 critical injection** - GStreamer pipeline injection via device string (#42)
- **Integer overflow** in buffer size calculations (#50)
- **No input validation** on calibration JSON, output paths, or blend parameter (#51, #64)
- **CUDA/Vulkan sync race** in zero-copy path (#44)
- **Use-after-free** from Surface lifetime transmute (#43)

## Dependency Assessment

- **wgpu-hal** phantom dependency (never imported)
- **log + tracing** dual logging creates confusion
- **nvarguscamerasrc** crashes after ~40min with dual cameras (#47)
- **Vulkan corruption** reported on Jetson Orin Nano (#66)
- **wgpu-hal API** will break on next version upgrade (#59)

## Kanban Workflow

All issues are labeled `backlog` and assigned to milestone `deep-review-2026-03-28`.

Flow: `backlog` -> `in-progress` -> `review` -> closed (Done)

**Recommended prioritization:**
1. Critical security fixes (#42, #43, #44) - immediate
2. Critical correctness bugs (#45, #46) - immediate
3. Critical performance (#48, #49) - next sprint
4. High architecture (#53, #54) - planned refactor
5. Everything else - backlog triage
