//! Browser ROI editor document built from the accepted preview images.

use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use anyhow::{Context, ensure};
use base64::Engine;
use reco_core::calibration::FieldRoi;
use reco_core::render::color::bt709_yuv_to_rgb;
use reco_core::source::{FrameSet, Yuv420pData};

/// One bounded image-encoding job, tied to its accepted document and inputs.
pub(super) struct PendingEditor {
    bookmark: super::PreviewBookmark,
    worker: Option<JoinHandle<anyhow::Result<EditorFile>>>,
}

/// Retain the current browser document without accumulating camera images in the cache.
pub(super) struct EditorFile {
    pub path: PathBuf,
}

impl EditorFile {
    fn write(directory: &Path, html: &str) -> anyhow::Result<Self> {
        std::fs::create_dir_all(directory)?;
        let path = directory.join(format!("roi-{}.html", uuid::Uuid::new_v4()));
        let mut file = std::fs::File::create_new(&path)?;
        let document = Self { path };
        let result = file.write_all(html.as_bytes());
        drop(file);
        result?;
        Ok(document)
    }
}

impl Drop for EditorFile {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.path) {
            log::warn!("Could not remove ROI editor document: {error}");
        }
    }
}

impl PendingEditor {
    pub fn start(state: &super::AppState) -> anyhow::Result<Self> {
        let bookmark = super::PreviewBookmark::capture(state)
            .context("Load camera inputs and calibration first")?;
        let Some(FrameSet::Yuv420p(cameras)) = state.playback.current_frame() else {
            anyhow::bail!("No decoded preview images available");
        };
        ensure!(cameras.len() == 2, "ROI editor needs two camera images");
        let cameras = cameras.clone();
        let full_range = state.playback.is_full_range();
        let roi = bookmark.calibration.field_roi.clone();
        let worker = std::thread::Builder::new()
            .name("roi-editor".into())
            .spawn(move || {
                let html = document(&cameras, full_range, roi.as_ref())?;
                // Snap-packaged browsers cannot read Linux's private /tmp namespace.
                let cache = std::env::var_os("XDG_CACHE_HOME")
                    .map(PathBuf::from)
                    .or_else(|| {
                        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache"))
                    })
                    .unwrap_or_else(std::env::temp_dir);
                EditorFile::write(&cache.join("reco").join("roi"), &html)
            })?;
        Ok(Self {
            bookmark,
            worker: Some(worker),
        })
    }

    pub fn is_finished(&self) -> bool {
        self.worker.as_ref().is_some_and(JoinHandle::is_finished)
    }

    pub fn finish(mut self, state: &super::AppState) -> Option<anyhow::Result<EditorFile>> {
        let result = self
            .worker
            .take()?
            .join()
            .unwrap_or_else(|_| Err(anyhow::anyhow!("ROI editor worker panicked")));
        self.bookmark.matches(state).then_some(result)
    }
}

impl Drop for PendingEditor {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Encode the two raw camera images without re-seeking or reloading calibration.
pub(super) fn document(
    cameras: &[Arc<Yuv420pData>],
    full_range: bool,
    roi: Option<&FieldRoi>,
) -> anyhow::Result<String> {
    ensure!(cameras.len() == 2, "ROI editor needs two camera images");
    let left = image_data(&cameras[0], full_range)?;
    let right = image_data(&cameras[1], full_range)?;
    let roi = serde_json::to_string(&roi)?;
    Ok(include_str!("../../../resources/roi_editor.html")
        .replace("{{LEFT_IMAGE_DATA}}", &left)
        .replace("{{RIGHT_IMAGE_DATA}}", &right)
        .replace("{{ROI_JSON}}", &roi))
}

fn image_data(frame: &Yuv420pData, full_range: bool) -> anyhow::Result<String> {
    let (width, height) = frame.size;
    let (w, h) = (width as usize, height as usize);
    let pixels = w.checked_mul(h).context("ROI image dimensions overflow")?;
    let rgb_len = pixels
        .checked_mul(3)
        .context("ROI image dimensions overflow")?;
    ensure!(
        width > 0
            && height > 0
            && width.is_multiple_of(2)
            && height.is_multiple_of(2)
            && frame.y.len() == pixels
            && frame.u.len() == pixels / 4
            && frame.v.len() == pixels / 4,
        "ROI image must contain tightly packed, even-sized YUV420P planes"
    );
    let mut rgb = vec![0; rgb_len];
    for (i, pixel) in rgb.chunks_exact_mut(3).enumerate() {
        let chroma = (i / w / 2) * (w / 2) + (i % w / 2);
        let colour = bt709_yuv_to_rgb(
            frame.y[i] as f64 / 255.0,
            frame.u[chroma] as f64 / 255.0,
            frame.v[chroma] as f64 / 255.0,
            full_range,
        );
        pixel.copy_from_slice(&colour.map(|c| (c * 255.0).round() as u8));
    }
    let image = image::RgbImage::from_raw(width, height, rgb).context("invalid ROI image size")?;
    let mut png = Cursor::new(Vec::new());
    image.write_to(&mut png, image::ImageFormat::Png)?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png.into_inner())
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(size: (u32, u32), y: u8) -> Arc<Yuv420pData> {
        let pixels = (size.0 * size.1) as usize;
        Arc::new(Yuv420pData {
            size,
            y: vec![y; pixels],
            u: vec![128; pixels / 4],
            v: vec![128; pixels / 4],
        })
    }

    fn decode_data(data: &str) -> image::RgbImage {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data.strip_prefix("data:image/png;base64,").unwrap())
            .unwrap();
        image::load_from_memory(&bytes).unwrap().to_rgb8()
    }

    #[test]
    fn each_camera_keeps_its_own_dimensions_and_current_roi() {
        let cameras = [frame((64, 32), 16), frame((32, 48), 235)];
        let roi = FieldRoi {
            left: vec![[0.1, 0.2], [0.8, 0.3], [0.5, 0.9]],
            right: vec![],
        };
        let html = document(&cameras, false, Some(&roi)).unwrap();
        for (side, size, expected) in [("LEFT", (64, 32), 0), ("RIGHT", (32, 48), 255)] {
            let prefix = format!("const {side}_IMAGE_DATA = '");
            let data = html
                .split(&prefix)
                .nth(1)
                .unwrap()
                .split('\'')
                .next()
                .unwrap();
            let image = decode_data(data);
            assert_eq!(image.dimensions(), size);
            assert!(image.pixels().all(|p| p.0 == [expected; 3]));
        }
        let json = html
            .split("const INITIAL_ROI = ")
            .nth(1)
            .unwrap()
            .split(';')
            .next()
            .unwrap();
        assert_eq!(serde_json::from_str::<FieldRoi>(json).unwrap(), roi);
        assert!(!html.contains("{{ROI_JSON}}"));
    }

    #[test]
    fn source_range_changes_decoding_without_changing_coordinates() {
        for (y, full_range, expected) in [
            (16, false, 0),
            (235, false, 255),
            (16, true, 16),
            (235, true, 235),
        ] {
            let image = decode_data(&image_data(&frame((4, 2), y), full_range).unwrap());
            assert_eq!(image.dimensions(), (4, 2));
            assert!(
                image
                    .pixels()
                    .all(|p| p.0.iter().all(|c| c.abs_diff(expected) <= 1))
            );
        }
    }

    #[test]
    fn invalid_images_fail_without_a_stale_fallback() {
        assert!(document(&[], false, None).is_err());
        assert!(document(&[frame((2, 2), 16)], false, None).is_err());
        for size in [(0, 2), (3, 2), (2, 3)] {
            assert!(image_data(&frame(size, 16), false).is_err());
        }
        let mut bad = (*frame((2, 2), 16)).clone();
        bad.u.clear();
        assert!(image_data(&bad, false).is_err());
        bad.size = (u32::MAX - 1, u32::MAX - 1);
        assert!(image_data(&bad, false).is_err());
    }

    fn state() -> super::super::AppState {
        use reco_core::calibration::{Calibration, Framing, Lens};
        let mut state = super::super::AppState::new(Default::default());
        state.left_input = Some(reco_io::InputPath::Single("left.mp4".into()));
        state.right_input = Some(reco_io::InputPath::Single("right.mp4".into()));
        state.calibration = Some(Calibration::new(
            vec![Lens::flat(64, 32); 2],
            reco_core::projection::LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
        ));
        state
    }

    #[test]
    fn worker_results_are_bound_to_the_accepted_document_and_inputs() {
        let changes: [fn(&mut super::super::AppState); 3] = [
            |state| state.calibration.as_mut().unwrap().framing.roll = 0.1,
            |state| state.right_input = Some(reco_io::InputPath::Single("replacement.mp4".into())),
            |state| state.calibration = None,
        ];
        for change in changes {
            let mut state = state();
            let job = PendingEditor {
                bookmark: super::super::PreviewBookmark::capture(&state).unwrap(),
                worker: Some(std::thread::spawn(|| {
                    EditorFile::write(&std::env::temp_dir(), "prepared")
                })),
            };
            change(&mut state);
            assert!(job.finish(&state).is_none());
        }
        let state = state();
        let job = PendingEditor {
            bookmark: super::super::PreviewBookmark::capture(&state).unwrap(),
            worker: Some(std::thread::spawn(|| {
                EditorFile::write(&std::env::temp_dir(), "prepared")
            })),
        };
        let document = job.finish(&state).unwrap().unwrap();
        let path = document.path.clone();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "prepared");
        drop(document);
        assert!(!path.exists());
        assert!(
            PendingEditor::start(&state).is_err(),
            "no decoded frame must fail"
        );
    }

    #[test]
    fn worker_failure_is_reported_and_drop_joins_owned_work() {
        let state = state();
        let job = PendingEditor {
            bookmark: super::super::PreviewBookmark::capture(&state).unwrap(),
            worker: Some(std::thread::spawn(|| panic!("test ROI worker failure"))),
        };
        assert!(job.finish(&state).unwrap().is_err());
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_completed = Arc::clone(&completed);
        let job = PendingEditor {
            bookmark: super::super::PreviewBookmark::capture(&state).unwrap(),
            worker: Some(std::thread::spawn(move || {
                worker_completed.store(true, std::sync::atomic::Ordering::Release);
                EditorFile::write(&std::env::temp_dir(), "prepared")
            })),
        };
        drop(job);
        assert!(completed.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    #[ignore = "requires RECO_PLAYBACK_TEST_VIDEO and an isolated XDG_CACHE_HOME"]
    fn editor_uses_the_displayed_pair_across_sync_and_segment_boundaries() {
        let path = PathBuf::from(std::env::var("RECO_PLAYBACK_TEST_VIDEO").unwrap());
        let cache = PathBuf::from(std::env::var("XDG_CACHE_HOME").unwrap());
        let mut state = state();
        let input = reco_io::InputPath::Chained(vec![path.clone(), path]);
        state.left_input = Some(input.clone());
        state.right_input = Some(input.clone());
        state.calibration.as_mut().unwrap().field_roi = Some(FieldRoi {
            left: vec![[0.1, 0.1], [0.8, 0.2], [0.5, 0.9]],
            right: vec![],
        });
        for offset in [-7, 7] {
            state
                .playback
                .open(
                    &[input.clone(), input.clone()],
                    offset,
                    &state.decoder_cleanup,
                )
                .unwrap();
            let target = state.playback.total_frames().unwrap() * 3 / 4;
            state.playback.seek_frame(target).unwrap();
            state.playback.advance_blocking().unwrap();
            let Some(FrameSet::Yuv420p(cameras)) = state.playback.current_frame() else {
                panic!("YUV420P preview")
            };
            let expected = document(
                cameras,
                state.playback.is_full_range(),
                state.calibration.as_ref().unwrap().field_roi.as_ref(),
            )
            .unwrap();
            let index = state.playback.frame_index();
            let job = PendingEditor::start(&state).unwrap();
            state.playback.advance_blocking().unwrap();
            assert_eq!(state.playback.frame_index(), index + 1);
            let document = job.finish(&state).unwrap().unwrap();
            assert!(document.path.starts_with(&cache));
            assert_eq!(std::fs::read_to_string(&document.path).unwrap(), expected);
        }
    }
}
