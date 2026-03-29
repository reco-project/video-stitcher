#!/usr/bin/env bash
# Quick test commands for reco v2 stitcher
# Usage: ./scripts/test.sh [command]
#
# Commands:
#   stitch      Stitch 30s of 4K video (default)
#   stitch1080  Stitch 30s of 1080p video
#   quick       Quick 10-frame smoke test
#   bench       Benchmark: zero-copy vs CPU upload (300 frames each)
#   profile     Profile 300 frames, output trace to reco-trace.json
#   info        Show GPU and system info
#   compare     Stitch same clip with both paths, print comparison
#
# Environment:
#   RECO_RES      Output resolution: 720, 1080, 2k, 4k (default: 1080)
#   RECO_FRAMES   Override frame count (default varies per command)
#   RECO_QUALITY  Quality preset: fast, balanced, high (default: balanced)
#   RECO_OUT      Output directory (default: /tmp)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ── Config ───────────────────────────────────────────────────────────
RES=~/Videos/video-stitcher-res
LEFT_4K="$RES/gopro10_left(4K).mp4"
RIGHT_4K="$RES/gopro10_right(4K).mp4"
LEFT_1080="$RES/gopro10_left(1080pHQ).mp4"
RIGHT_1080="$RES/gopro10_right(1080pHQ).mp4"
CAL="$RES/match.json"
QUALITY="${RECO_QUALITY:-balanced}"
OUT="${RECO_OUT:-/tmp}"

# Resolve output resolution
case "${RECO_RES:-1080}" in
    720)  OUT_W=1280  OUT_H=720  ;;
    1080) OUT_W=1920  OUT_H=1080 ;;
    2k|2K|1440) OUT_W=2560 OUT_H=1440 ;;
    4k|4K) OUT_W=3840 OUT_H=2160 ;;
    *) echo "Unknown resolution: $RECO_RES (use 720, 1080, 2k, 4k)"; exit 1 ;;
esac
RES_TAG="${RECO_RES:-1080}p"

RECO="cargo run --release -p reco-cli --"
RECO_PROF="cargo run --release -p reco-cli --features profiling --"

# ── Commands ─────────────────────────────────────────────────────────
cmd_stitch() {
    local frames="${RECO_FRAMES:-900}"
    local out_file="$OUT/reco_4k_${RES_TAG}.mp4"
    echo "==> Stitching 4K input → ${OUT_W}x${OUT_H}, $frames frames, quality=$QUALITY"
    RUST_LOG=warn $RECO stitch "$LEFT_4K" "$RIGHT_4K" \
        -c "$CAL" -o "$out_file" \
        --width "$OUT_W" --height "$OUT_H" \
        --quality "$QUALITY" --max-frames "$frames"
    echo -e "\nOutput: $out_file"
    ffprobe -v error -show_entries stream=width,height,nb_frames,duration,bit_rate \
        -of compact "$out_file"
}

cmd_stitch1080() {
    local frames="${RECO_FRAMES:-900}"
    local out_file="$OUT/reco_1080p_${RES_TAG}.mp4"
    echo "==> Stitching 1080p input → ${OUT_W}x${OUT_H}, $frames frames, quality=$QUALITY"
    RUST_LOG=warn $RECO stitch "$LEFT_1080" "$RIGHT_1080" \
        -c "$CAL" -o "$out_file" \
        --width "$OUT_W" --height "$OUT_H" \
        --quality "$QUALITY" --max-frames "$frames"
    echo -e "\nOutput: $out_file"
    ffprobe -v error -show_entries stream=width,height,nb_frames,duration,bit_rate \
        -of compact "$out_file"
}

cmd_quick() {
    local frames="${RECO_FRAMES:-10}"
    echo "==> Quick smoke test ($frames frames, ${OUT_W}x${OUT_H})"
    RUST_LOG=warn $RECO stitch "$LEFT_4K" "$RIGHT_4K" \
        -c "$CAL" -o "$OUT/reco_quick.mp4" \
        --width "$OUT_W" --height "$OUT_H" \
        --max-frames "$frames"
    echo "OK"
}

cmd_bench() {
    local frames="${RECO_FRAMES:-300}"
    echo "==> Benchmark: $frames frames, 4K input → ${OUT_W}x${OUT_H}"
    echo ""
    echo "--- Zero-copy (CUDA/Vulkan interop) ---"
    RUST_LOG=warn $RECO stitch "$LEFT_4K" "$RIGHT_4K" \
        -c "$CAL" -o "$OUT/reco_bench_zc.mp4" \
        --width "$OUT_W" --height "$OUT_H" --max-frames "$frames"
    echo ""
    echo "--- CPU upload (baseline) ---"
    RECO_NO_HWACCEL=1 RUST_LOG=warn $RECO stitch "$LEFT_4K" "$RIGHT_4K" \
        -c "$CAL" -o "$OUT/reco_bench_cpu.mp4" \
        --width "$OUT_W" --height "$OUT_H" --max-frames "$frames"
}

cmd_profile() {
    local frames="${RECO_FRAMES:-300}"
    echo "==> Profiling $frames frames → reco-trace.json (${OUT_W}x${OUT_H})"
    RUST_LOG=warn $RECO_PROF stitch "$LEFT_4K" "$RIGHT_4K" \
        -c "$CAL" -o "$OUT/reco_profiled.mp4" \
        --width "$OUT_W" --height "$OUT_H" --max-frames "$frames"
    echo -e "\nTrace: $(pwd)/reco-trace.json (open in ui.perfetto.dev)"
}

cmd_info() {
    $RECO info
}

cmd_compare() {
    local frames="${RECO_FRAMES:-300}"
    echo "==> Comparing both paths, $frames frames, 4K input → ${OUT_W}x${OUT_H}"
    echo ""

    echo "--- Zero-copy ---"
    RUST_LOG=warn $RECO stitch "$LEFT_4K" "$RIGHT_4K" \
        -c "$CAL" -o "$OUT/reco_cmp_zc.mp4" \
        --width "$OUT_W" --height "$OUT_H" --max-frames "$frames"

    echo ""
    echo "--- CPU upload ---"
    RECO_NO_HWACCEL=1 RUST_LOG=warn $RECO stitch "$LEFT_4K" "$RIGHT_4K" \
        -c "$CAL" -o "$OUT/reco_cmp_cpu.mp4" \
        --width "$OUT_W" --height "$OUT_H" --max-frames "$frames"

    echo ""
    echo "==> Output comparison:"
    echo "Zero-copy:"
    ffprobe -v error -show_entries stream=width,height,nb_frames,duration,bit_rate \
        -of compact "$OUT/reco_cmp_zc.mp4"
    echo "CPU upload:"
    ffprobe -v error -show_entries stream=width,height,nb_frames,duration,bit_rate \
        -of compact "$OUT/reco_cmp_cpu.mp4"
}

# ── Dispatch ─────────────────────────────────────────────────────────
case "${1:-stitch}" in
    stitch)     cmd_stitch ;;
    stitch1080) cmd_stitch1080 ;;
    quick)      cmd_quick ;;
    bench)      cmd_bench ;;
    profile)    cmd_profile ;;
    info)       cmd_info ;;
    compare)    cmd_compare ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: $0 {stitch|stitch1080|quick|bench|profile|info|compare}"
        exit 1
        ;;
esac
