//! Integration tests for the stacked-video encoder / source.
//!
//! Covers M6.5 item 3's load-bearing requirement: a reader opened
//! on the same path while the writer is still writing must see
//! already-flushed fragments. Enabled only when the `stacked-output`
//! feature is on; run with:
//!
//! ```bash
//! cargo test -p reco-io --features stacked-output --test stacked_video_roundtrip
//! ```

#![cfg(feature = "stacked-output")]

use reco_core::source::YuvFrame;
use reco_io::ffmpeg::encoder::{Container, Quality};
use reco_io::stacked_video::encoder::{StackedEncoder, StackedEncoderConfig};
use reco_io::stacked_video::source::StackedSource;
use reco_io::stacked_video::{GridLayout, pack_yuv420p, unpack_yuv420p};

const TILE_W: u32 = 320;
const TILE_H: u32 = 240;
const N_FRAMES: usize = 60;

/// Synthetic tile: fills planes with per-frame/per-tile constant
/// values so we can recover the written index after decode even
/// after H.264 lossy compression rounds off.
fn synthetic_tile(frame_idx: usize, tile_idx: usize) -> YuvFrame {
    let y_val = 64u8.wrapping_add((frame_idx as u8) % 128);
    let u_val = if tile_idx == 0 { 96u8 } else { 160u8 };
    let v_val = if tile_idx == 0 { 160u8 } else { 96u8 };
    let yw = TILE_W as usize;
    let yh = TILE_H as usize;
    let uvw = yw / 2;
    let uvh = yh / 2;
    YuvFrame {
        y: vec![y_val; yw * yh],
        u: vec![u_val; uvw * uvh],
        v: vec![v_val; uvw * uvh],
        width: TILE_W,
        height: TILE_H,
        timestamp_us: (frame_idx as i64) * 33_333,
    }
}

fn mean(bytes: &[u8]) -> f64 {
    let sum: u64 = bytes.iter().map(|b| *b as u64).sum();
    sum as f64 / bytes.len() as f64
}

/// In-memory pack -> unpack round trip. Validates the pure
/// primitives are byte-perfect without going through ffmpeg. The
/// encoder tests below add lossy H.264 on top of this.
#[test]
fn pack_unpack_is_byte_perfect() {
    let layout = GridLayout::vstack(TILE_W, TILE_H, 2).expect("even dims");
    let left = synthetic_tile(7, 0);
    let right = synthetic_tile(7, 1);
    let packed = pack_yuv420p(&layout, &[Some(&left), Some(&right)]).expect("pack");
    let tiles = unpack_yuv420p(&layout, &packed).expect("unpack");
    assert_eq!(tiles.len(), 2);
    assert_eq!(tiles[0].y, left.y);
    assert_eq!(tiles[0].u, left.u);
    assert_eq!(tiles[0].v, left.v);
    assert_eq!(tiles[1].y, right.y);
    assert_eq!(tiles[1].u, right.u);
    assert_eq!(tiles[1].v, right.v);
}

fn encoder_config(container: Container) -> StackedEncoderConfig {
    let base = StackedEncoderConfig::default();
    StackedEncoderConfig {
        inner: reco_io::ffmpeg::encoder::EncoderConfig {
            container,
            quality: Quality::Fast,
            gop_size: Some(10),
            ..base.inner
        },
        fps: base.fps,
    }
}

/// Encode N frames to a fragmented MP4, then decode the whole file
/// and verify we recover the left/right chroma distinction for each
/// frame. End-to-end sanity: pack -> encode -> decode -> unpack.
/// Fragmented MP4 finalize is currently broken under our stack
/// (write_trailer returns AVERROR -162 even with the canonical
/// movflags recipe that ffmpeg CLI accepts). Kept `#[ignore]`d as
/// the regression fixture for the eventual fix; see the
/// `Container` docstring and the architecture note
/// `stacked-video-replay-2026-04-19.md` for context.
#[test]
#[ignore]
fn roundtrip_fmp4_recovers_tiles() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("stacked.mp4");
    let layout = GridLayout::vstack(TILE_W, TILE_H, 2).expect("even dims");
    let mut enc =
        StackedEncoder::new(layout, &path, encoder_config(Container::Mp4Fragmented)).expect("open");
    for i in 0..N_FRAMES {
        let l = synthetic_tile(i, 0);
        let r = synthetic_tile(i, 1);
        enc.push(&[Some(&l), Some(&r)]).expect("push");
    }
    enc.finish().expect("finish fmp4");

    let mut src = StackedSource::open(layout, &path).expect("open source");
    let mut decoded = 0usize;
    while src.next_tuple().expect("decode").is_some() {
        decoded += 1;
    }
    assert_eq!(decoded, N_FRAMES, "all frames should round-trip fMP4");
}

#[test]
fn roundtrip_matroska_recovers_tiles() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("stacked.mkv");
    let layout = GridLayout::vstack(TILE_W, TILE_H, 2).expect("even dims");

    let mut enc =
        StackedEncoder::new(layout, &path, encoder_config(Container::Matroska)).expect("open");
    for i in 0..N_FRAMES {
        let l = synthetic_tile(i, 0);
        let r = synthetic_tile(i, 1);
        enc.push(&[Some(&l), Some(&r)]).expect("push");
    }
    enc.finish().expect("finish");

    let mut src = StackedSource::open(layout, &path).expect("open source");
    let mut decoded = 0usize;
    while let Some(tiles) = src.next_tuple().expect("decode") {
        assert_eq!(tiles.len(), 2);
        // H.264 4:2:0 lossy, but a flat chroma plane should survive
        // well enough that `left` stays distinctly "redder" (high V,
        // low U) and `right` stays distinctly "bluer" (low V, high U)
        // relative to each other. Tolerance is generous to accommodate
        // x264's chroma rounding.
        let lu = mean(&tiles[0].u);
        let lv = mean(&tiles[0].v);
        let ru = mean(&tiles[1].u);
        let rv = mean(&tiles[1].v);
        assert!(
            lv > lu,
            "frame {decoded}: left tile should read V>U, got V={lv:.1} U={lu:.1}"
        );
        assert!(
            ru > rv,
            "frame {decoded}: right tile should read U>V, got U={ru:.1} V={rv:.1}"
        );
        decoded += 1;
    }
    assert_eq!(decoded, N_FRAMES, "all frames should round-trip");
}

/// Write-while-read: open a reader on the same file mid-write and
/// verify it sees already-flushed fragments. This is the core
/// guarantee behind replay-during-recording.
///
/// Fragmented MP4 flushes fragments on keyframes. `libx264` with
/// `tune=zerolatency` (our Quality::Fast default) uses short GOPs,
/// so we expect the first fragment to appear within a handful of
/// frames.
/// Matroska write-while-read: the load-bearing M6.5-item-3 guarantee.
/// A reader opened on the same file while the writer is still pushing
/// frames must see already-flushed clusters. Matroska flushes clusters
/// periodically (roughly every keyframe with a short GOP), and the
/// container has no central index that needs the trailer to be
/// readable - so a concurrent reader sees content as soon as it
/// hits disk.
#[test]
fn matroska_reader_sees_partial_writes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("stacked.mkv");
    let layout = GridLayout::vstack(TILE_W, TILE_H, 2).expect("even dims");

    let mut enc =
        StackedEncoder::new(layout, &path, encoder_config(Container::Matroska)).expect("open");

    // Push half the frames and flush so the AVIO layer writes to
    // disk. Without flush(), ffmpeg buffers several clusters-worth
    // of packets in memory before a single write.
    for i in 0..(N_FRAMES / 2) {
        let l = synthetic_tile(i, 0);
        let r = synthetic_tile(i, 1);
        enc.push(&[Some(&l), Some(&r)]).expect("push");
    }
    enc.flush().expect("flush");

    // Reader opens with a separate file handle while the writer
    // holds its own. No file locks, no mmap; the OS lets both
    // coexist.
    let mut src = StackedSource::open(layout, &path)
        .expect("reader should open Matroska while writer is still running");

    // Drain whatever's readable right now. With the drain-on-EOF
    // decoder fix, this returns every cluster the writer has
    // flushed.
    let mut seen = 0usize;
    while let Some(tiles) = src.next_tuple().expect("decode") {
        assert_eq!(tiles.len(), 2);
        seen += 1;
    }
    assert!(seen > 0, "reader should see at least one flushed cluster");

    // Drop the reader's handle before finalizing. Matroska doesn't
    // need this for correctness, but it mirrors the real replay
    // consumer pattern where the reader closes when the user
    // stops scrubbing.
    drop(src);

    // Writer keeps pushing and finalizes cleanly afterwards.
    for i in (N_FRAMES / 2)..N_FRAMES {
        let l = synthetic_tile(i, 0);
        let r = synthetic_tile(i, 1);
        enc.push(&[Some(&l), Some(&r)]).expect("push");
    }
    enc.finish().expect("writer finishes after read");

    // Final file has all frames.
    let mut final_src = StackedSource::open(layout, &path).expect("reopen final");
    let mut total = 0usize;
    while final_src.next_tuple().expect("decode").is_some() {
        total += 1;
    }
    assert_eq!(total, N_FRAMES, "final file should hold all pushed frames");
}

/// Push-API recorder (FRICTION A18 close): verifies
/// `SessionStackedRecorder` accepts `YuvPlanes<'_>` feeds and
/// produces a replayable stacked-video file.
///
/// This is the key correctness test for the M6.5 item 3 push side:
/// consumers call `record_yuv` per submit, `finish` at session end,
/// and the output file must be openable by `StackedSource` with all
/// pushed frames recoverable.
#[test]
fn session_recorder_records_planes() {
    use reco_core::pipeline::YuvPlanes;

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("session_replay.mkv");
    let layout = GridLayout::vstack(TILE_W, TILE_H, 2).expect("even dims");

    let mut rec = reco_io::stacked_video::replay::SessionStackedRecorder::open(
        &path,
        encoder_config(Container::Matroska),
        TILE_W,
        TILE_H,
    )
    .expect("open session recorder");

    for i in 0..N_FRAMES {
        let l = synthetic_tile(i, 0);
        let r = synthetic_tile(i, 1);
        let left = YuvPlanes {
            y: &l.y,
            u: &l.u,
            v: &l.v,
        };
        let right = YuvPlanes {
            y: &r.y,
            u: &r.u,
            v: &r.v,
        };
        rec.record_yuv(&left, &right, TILE_W, TILE_H);
    }
    rec.finish();
    drop(rec);

    // Readback: open the recorded file as a StackedSource and
    // verify we get every frame back.
    let mut src = StackedSource::open(layout, &path).expect("open recorded file");
    let mut count = 0usize;
    while src.next_tuple().expect("decode").is_some() {
        count += 1;
    }
    assert_eq!(
        count, N_FRAMES,
        "session recorder should produce a fully round-trippable file"
    );
}

/// Plain MP4 (moov-at-end) cannot be opened until the writer
/// finishes. This is the negative case that motivates the fMP4
/// default - it documents why "just use MP4" is wrong for replay.
#[test]
fn plain_mp4_reader_cannot_open_mid_write() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("stacked.mp4");
    let layout = GridLayout::vstack(TILE_W, TILE_H, 2).expect("even dims");

    let mut enc = StackedEncoder::new(layout, &path, encoder_config(Container::Mp4)).expect("open");
    for i in 0..N_FRAMES {
        let l = synthetic_tile(i, 0);
        let r = synthetic_tile(i, 1);
        enc.push(&[Some(&l), Some(&r)]).expect("push");
    }

    // Plain MP4: the moov atom is written at finish, so a reader
    // opened before finish sees no seekable structure and
    // StackedSource::open fails.
    let open_result = StackedSource::open(layout, &path);
    assert!(
        open_result.is_err(),
        "plain MP4 should not be openable before writer finishes"
    );

    enc.finish().expect("finish");

    // After finish, the file is readable.
    let mut src = StackedSource::open(layout, &path).expect("open after finish");
    let mut count = 0;
    while src.next_tuple().expect("decode").is_some() {
        count += 1;
    }
    assert_eq!(count, N_FRAMES);
}
