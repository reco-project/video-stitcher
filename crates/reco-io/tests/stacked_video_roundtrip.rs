//! Integration tests for the stacked-video encoder / source.
//!
//! Covers replay-during-recording's load-bearing requirement: a reader
//! opened on the same path while the writer is still writing must see
//! already-flushed fragments. Enabled whenever the `ffmpeg` feature
//! is on (default); run directly with:
//!
//! ```bash
//! cargo test -p reco-io --test stacked_video_roundtrip
//! ```

#![cfg(feature = "ffmpeg")]

use reco_core::source::Yuv420pFrame;
use reco_io::output::{Container, Quality, QualityPreset};
use reco_io::stacked_video::encoder::{StackedEncoder, StackedEncoderConfig};
use reco_io::stacked_video::source::StackedSource;
use reco_io::stacked_video::{GridLayout, pack_yuv420p, unpack_yuv420p};

const TILE_W: u32 = 320;
const TILE_H: u32 = 240;
const N_FRAMES: usize = 60;

#[test]
fn raw_replay_preserves_range_and_planes_without_assigning_a_matrix() {
    use reco_core::source::FrameSource;
    use reco_io::stacked_video::replay::ReplayRecordingSource;
    let dir = tempfile::tempdir().unwrap();
    let layout = GridLayout::vstack(TILE_W, TILE_H, 2).unwrap();
    for name in ["libx264", "libx265"] {
        for full_range in [false, true] {
            let path = dir.path().join(format!("{name}-{full_range}.mkv"));
            let mut config = encoder_config(Container::Mkv);
            config.inner.encoder_name = Some(name.into());
            config.inner.codec = None;
            config.full_range = full_range;
            let mut writer = StackedEncoder::new(layout, &path, config.clone()).unwrap();
            let left = synthetic_tile(0, 0);
            let right = synthetic_tile(0, 1);
            writer.push_all(&[&left, &right]).unwrap();
            writer.finish().unwrap();
            drop(writer);
            let source = StackedSource::open(layout, &path).unwrap();
            assert_eq!(source.is_full_range(), full_range);
            let replay_path = dir.path().join(format!("{name}-{full_range}-replay.mkv"));
            config.full_range = !full_range;
            let mut recording =
                ReplayRecordingSource::wrap(Box::new(source), &replay_path, config).unwrap();
            assert!(recording.next_frame().unwrap().is_some());
            assert!(recording.next_frame().unwrap().is_none());
            drop(recording);
            let input = ffmpeg_next::format::input(&replay_path).unwrap();
            let parameters = input
                .streams()
                .best(ffmpeg_next::media::Type::Video)
                .unwrap()
                .parameters();
            // SAFETY: the parameters are owned and remain live during inspection.
            assert_eq!(
                unsafe { (*parameters.as_ptr()).color_space },
                ffmpeg_next::ffi::AVColorSpace::AVCOL_SPC_UNSPECIFIED
            );
            let mut replay = StackedSource::open(layout, &replay_path).unwrap();
            assert_eq!(replay.is_full_range(), full_range);
            let frames = replay.next_tuple().unwrap().unwrap();
            for (actual, expected) in frames.iter().zip([&left, &right]) {
                for (a, e) in [
                    (&actual.y, &expected.y),
                    (&actual.u, &expected.u),
                    (&actual.v, &expected.v),
                ] {
                    assert!((mean(a) - mean(e)).abs() <= 3.0);
                }
            }
        }
    }
}

/// Synthetic tile: fills planes with per-frame/per-tile constant
/// values so we can recover the written index after decode even
/// after H.264 lossy compression rounds off.
fn synthetic_tile(frame_idx: usize, tile_idx: usize) -> Yuv420pFrame {
    let y_val = 64u8.wrapping_add((frame_idx as u8) % 128);
    let u_val = if tile_idx == 0 { 96u8 } else { 160u8 };
    let v_val = if tile_idx == 0 { 160u8 } else { 96u8 };
    let yw = TILE_W as usize;
    let yh = TILE_H as usize;
    let uvw = yw / 2;
    let uvh = yh / 2;
    Yuv420pFrame {
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
            encoder_name: Some("libx264".into()),
            container,
            quality: Quality::Preset(QualityPreset::Fast),
            gop_size: Some(10),
            ..base.inner
        },
        fps: base.fps,
        full_range: false,
    }
}

/// Encode N frames to a fragmented MP4, then decode the whole file
/// and verify we recover the left/right chroma distinction for each
/// frame. End-to-end sanity: pack -> encode -> decode -> unpack.
#[test]
fn roundtrip_fmp4_recovers_tiles() {
    roundtrip_recovers_tiles(Container::Mp4Fragmented, "stacked.mp4");
}

#[test]
fn roundtrip_matroska_recovers_tiles() {
    roundtrip_recovers_tiles(Container::Mkv, "stacked.mkv");
}

fn roundtrip_recovers_tiles(container: Container, filename: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(filename);
    let layout = GridLayout::vstack(TILE_W, TILE_H, 2).expect("even dims");

    let mut enc = StackedEncoder::new(layout, &path, encoder_config(container)).expect("open");
    for i in 0..N_FRAMES {
        let l = synthetic_tile(i, 0);
        let r = synthetic_tile(i, 1);
        enc.push(&[Some(&l), Some(&r)]).expect("push");
    }
    enc.finish().expect("finish");
    drop(enc);

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

/// Explicit flush publishes decodable packets before the writer finishes.
#[test]
fn matroska_reader_sees_partial_writes() {
    reader_sees_partial_writes(Container::Mkv, "stacked.mkv");
}

#[test]
fn fmp4_reader_sees_partial_writes() {
    reader_sees_partial_writes(Container::Mp4Fragmented, "stacked.mp4");
}

fn reader_sees_partial_writes(container: Container, filename: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(filename);
    let layout = GridLayout::vstack(TILE_W, TILE_H, 2).expect("even dims");

    let mut enc = StackedEncoder::new(layout, &path, encoder_config(container)).expect("open");

    // Flush both muxer and I/O buffers without closing the writer.
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
        .expect("reader should open while writer is still running");

    // Drain the packets published so far.
    let mut seen = 0usize;
    while let Some(tiles) = src.next_tuple().expect("decode") {
        assert_eq!(tiles.len(), 2);
        assert!(tiles[0].y[0].abs_diff(synthetic_tile(seen, 0).y[0]) <= 4);
        assert!(tiles[0].u[0].abs_diff(96) <= 4);
        assert!(tiles[1].u[0].abs_diff(160) <= 4);
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
    enc.flush().expect("flush after finish is harmless");
    drop(enc);

    // Final file has all frames.
    let mut final_src = StackedSource::open(layout, &path).expect("reopen final");
    let mut total = 0usize;
    while final_src.next_tuple().expect("decode").is_some() {
        total += 1;
    }
    assert_eq!(total, N_FRAMES, "final file should hold all pushed frames");
}

/// Push-API recorder: verifies
/// `SessionStackedRecorder` accepts `Yuv420pPlanes<'_>` feeds and
/// produces a replayable stacked-video file.
///
/// This is the key correctness test for the push-side recorder:
/// consumers call `record_yuv` per submit, `finish` at session end,
/// and the output file must be openable by `StackedSource` with all
/// pushed frames recoverable.
#[test]
fn session_recorder_records_planes() {
    use reco_core::render::planes::Yuv420pPlanes;

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("session_replay.mkv");
    let layout = GridLayout::vstack(TILE_W, TILE_H, 2).expect("even dims");

    let mut rec = reco_io::stacked_video::replay::SessionStackedRecorder::open(
        &path,
        encoder_config(Container::Mkv),
        TILE_W,
        TILE_H,
    )
    .expect("open session recorder");

    for i in 0..N_FRAMES {
        let l = synthetic_tile(i, 0);
        let r = synthetic_tile(i, 1);
        let left = Yuv420pPlanes {
            y: &l.y,
            u: &l.u,
            v: &l.v,
        };
        let right = Yuv420pPlanes {
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
