//! Native lifetime, pairing and session delivery through actual loopback RTSP.

use super::*;
use crate::ffmpeg::decoder::VideoDecoder;
use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use reco_core::gpu::GpuContext;
use reco_core::interop::cuda::{cuda_2d_copy_dtoh, cuda_ensure_context};

fn options() -> RtspConfig {
    RtspConfig {
        software_decode: false,
        open_timeout: Duration::from_secs(5),
        ..config()
    }
}

fn gpu() -> GpuContext {
    let gpu = GpuContext::new_blocking().unwrap();
    assert!(
        gpu.supports_zero_copy(),
        "NVIDIA Vulkan required: {}",
        gpu.gpu_name()
    );
    gpu
}

fn luma(ptr: u64, pitch: usize, size: (u32, u32)) -> Vec<u8> {
    cuda_ensure_context().unwrap();
    let mut bytes = vec![0; (size.0 * size.1) as usize];
    cuda_2d_copy_dtoh(
        bytes.as_mut_ptr().cast(),
        size.0 as usize,
        ptr,
        pitch,
        size.0 as usize,
        size.1 as usize,
    )
    .unwrap();
    bytes
}

#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC and Vulkan"]
fn retained_rtsp_images_survive_pairing_reuse_and_source_destruction() {
    let gpu = gpu();
    let peers = [Peer::ramp(35, 0, None), Peer::ramp(35, 7_000_000, Some(8))];
    let stop = Arc::new(AtomicBool::new(false));
    let mut source = RtspSource::open_with_audio(
        vec![
            input(&peers[0]),
            RtspInput {
                timestamp_offset_us: -7_000_000,
                ..input(&peers[1])
            },
        ],
        options(),
        stop.clone(),
        None,
        Some(&gpu),
    )
    .unwrap();
    assert!(source.is_gpu_resident());
    assert_eq!(source.info().camera_sizes, [(128, 96); 2]);
    let retained = match &source.shared.lock().inputs[0].images.front().unwrap().image {
        Image::Cuda(frame) => frame.clone(),
        _ => panic!("native owner required"),
    };
    let before = luma(retained.y_ptr, retained.y_pitch, (128, 96));
    let mut previous: Option<Vec<i64>> = None;
    let mut reused = false;
    for _ in 0..14 {
        let frame = source.next_frame().unwrap().unwrap();
        let capture = frame.capture.unwrap();
        reused |= previous
            .as_ref()
            .is_some_and(|previous| previous.iter().zip(&capture.times_us).any(|(a, b)| a == b));
        previous = Some(capture.times_us.clone());
        let FrameSet::GpuResident { slots } = frame.images else {
            panic!("native frame required")
        };
        let native = source.native.as_ref().unwrap();
        for (camera, slot) in slots.into_iter().enumerate() {
            let surface = &native.shared.cameras[camera];
            let bytes = luma(
                surface.buf.y_ptr[usize::from(slot)],
                surface.buf.y_pitch[usize::from(slot)],
                (128, 96),
            );
            let index = (capture.times_us[camera] + source.media_origin_us) / 100_000;
            let expected = 32 + index as u8 * 4;
            assert!(
                bytes.iter().all(|&v| v.abs_diff(expected) <= 2),
                "camera {camera}: index {index}, got {}",
                bytes[0]
            );
            surface.slot_free_tx.send(slot).unwrap();
        }
    }
    assert!(reused, "missing capture must exercise retained-image reuse");
    stop.store(true, Ordering::Relaxed);
    assert!(source.next_frame().unwrap().is_none());
    drop(source);
    assert_eq!(luma(retained.y_ptr, retained.y_pitch, (128, 96)), before);
    let weak = Arc::downgrade(&retained);
    drop(retained);
    assert!(weak.upgrade().is_none());
}

struct Probe(Arc<Mutex<Vec<Vec<u8>>>>);
impl Detector for Probe {
    fn name(&self) -> &'static str {
        "native-luma"
    }
    fn prefers_cuda_nv12(&self) -> bool {
        true
    }
    fn detect(
        &mut self,
        camera: usize,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let DetectorFrame::CudaNv12(frame) = frame else {
            panic!("CUDA input required")
        };
        let values = luma(frame.y_ptr, frame.y_pitch, (frame.width, frame.height));
        assert!(values.iter().all(|&v| v.abs_diff(values[0]) <= 2));
        self.0.lock().unwrap()[camera].push(values[0]);
        Ok(Vec::new())
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC and Vulkan"]
fn native_rtsp_job_preserves_detection_and_encoded_lookahead() {
    for depth in [0.0, 0.3] {
        let peers = [Peer::ramp(40, 0, None), Peer::ramp(40, 0, Some(8))];
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("native.mp4");
        let seen = Arc::new(Mutex::new(vec![Vec::new(); 2]));
        let samples = seen.clone();
        let result = crate::StitchJob::from_rtsp(
            peers.iter().map(input).collect(),
            options(),
            pipeline::calibration(),
            &output,
        )
        .resolution(64, 32)
        .encoder_name("libx264")
        .lookahead(depth)
        .lookahead_memory_limit(16 * 1024 * 1024)
        .max_frames(14)
        .on_session(move |session, source| {
            assert!(source.is_gpu_resident());
            session.core_mut().set_constrained_look(false);
            session.set_detection_interval(1);
            session.set_detector(Box::new(Probe(samples)));
            Ok(())
        })
        .run(&Arc::default())
        .unwrap();
        assert_eq!(result.decode_mode, "RTSP CUDA/Vulkan");
        assert_eq!(result.frames_processed, 14);
        let observations = seen.lock().unwrap();
        assert_eq!(
            observations[0].len(),
            observations[1].len() + 1,
            "only the camera with a missing capture must skip an inference: {observations:?}"
        );
        assert!(observations[0].contains(&64));
        assert!(!observations[1].contains(&64));
        for samples in observations.iter() {
            assert!(
                samples.len() >= 13,
                "held images must not trigger inference twice: {samples:?}"
            );
            assert!(
                samples.windows(2).all(|v| v[1] > v[0]),
                "detector saw recycled pixels: {samples:?}"
            );
        }
        let mut decoder = VideoDecoder::open_software(&output).unwrap();
        let mut frames = 0;
        let mut previous = None;
        while let Some(frame) = decoder.next_frame().unwrap() {
            assert!(previous.is_none_or(|pts| frame.timestamp_us > pts));
            previous = Some(frame.timestamp_us);
            assert!(frame.y.iter().any(|&y| y > 32));
            frames += 1;
        }
        assert_eq!(frames, 14);
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC and Vulkan"]
fn native_negotiation_cancels_when_another_camera_never_opens() {
    let gpu = gpu();
    let peers = [Peer::ramp(30, 0, None), Peer::silent()];
    let stop = Arc::new(AtomicBool::new(false));
    let token = stop.clone();
    let cancel = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        token.store(true, Ordering::Relaxed);
    });
    let start = Instant::now();
    assert!(
        RtspSource::open_with_audio(
            peers.iter().map(input).collect(),
            options(),
            stop,
            None,
            Some(&gpu)
        )
        .is_err()
    );
    cancel.join().unwrap();
    assert!(start.elapsed() < Duration::from_secs(2));
}

#[test]
#[ignore = "requires NVDEC and a non-matching Vulkan adapter"]
fn rtsp_host_fallback_with_incompatible_renderer() {
    let gpu = GpuContext::new_blocking().unwrap();
    assert!(
        !gpu.supports_zero_copy(),
        "select a non-CUDA Vulkan adapter"
    );
    let peer = Peer::ramp(30, 0, None);
    let mut source = RtspSource::open_with_audio(
        vec![input(&peer)],
        options(),
        Arc::default(),
        None,
        Some(&gpu),
    )
    .unwrap();
    assert!(!source.is_gpu_resident());
    assert!(matches!(
        source.next_frame().unwrap().unwrap().images,
        FrameSet::Yuv420p(_)
    ));
}
