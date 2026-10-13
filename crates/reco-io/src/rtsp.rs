//! RTSP inputs with bounded timestamp selection and cancellable TCP reads.
//!
//! Explicit per-input offsets map decoded media timestamps onto one common timeline. They must be measured or supplied by the producer; connection order and packet arrival do not establish camera synchronization. The playout clock is anchored once from initial receipt, so age limits concern that media timeline, not independently certified exposure age. Untimed startup images are discarded within the opening deadline. After startup, clock resets, missing timestamps, format changes, unexpected end-of-stream and expired inputs fail the source without automatic reconnection.
//! Direct opening delivers host images. GPU-aware jobs can retain CUDA images through pairing and stage selected images into existing CUDA/Vulkan surfaces without a host round trip.

#[cfg(all(feature = "gpu", target_os = "linux"))]
mod native;
#[cfg(test)]
mod tests;
mod worker;

use crate::decoder_cleanup::DecoderTask;
use crate::ffmpeg::live_audio::LiveAudio;
use crate::frame_sync::{Capture, Config, Selector};
use reco_core::source::{
    CaptureTiming, FrameSet, FrameSource, SourceError, SourceFrame, SourceInfo, Yuv420pData,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// RTSP video input and its mapping to the source's common media timeline.
#[derive(Clone)]
pub struct RtspInput {
    /// RTSP URL. Credentials are not included in Reco errors or source labels.
    pub url: String,
    /// Microseconds added to decoded media timestamps. Equal offsets only synchronize streams that already have a common timestamp origin/rate. This is not clock-drift estimation.
    pub timestamp_offset_us: i64,
}

/// Explicit resource, cadence and matching policy for RTSP capture.
#[derive(Clone, Copy, Debug)]
pub struct RtspConfig {
    /// Output cadence as `(numerator, denominator)`, including rates such as `(30000, 1001)`.
    pub fps: (u32, u32),
    /// Matching/reuse limits on the common media timeline. Capacity bounds both each worker queue and each selector queue separately.
    pub synchronization: Config,
    /// Deadline for connecting, probing and receiving the initial image from every input.
    pub open_timeout: Duration,
    /// Deadline for obtaining another decoded image. Includes demux reads, not only socket silence.
    pub read_timeout: Duration,
    /// Disable hardware decode probes. Rendering is selected separately by the consumer.
    pub software_decode: bool,
}

#[derive(Clone, Copy)]
struct Metadata {
    size: (u32, u32),
    full_range: bool,
    first_time_us: i64,
    first_arrived_us: i64,
    rotation: i32,
}

#[derive(Clone)]
enum Image {
    Host(Arc<Yuv420pData>),
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    Cuda(Arc<crate::ffmpeg::decoder::GpuFrame>),
}

struct Decoded {
    time_us: i64,
    arrived_us: i64,
    image: Image,
}

#[derive(Default)]
struct InputState {
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    layout: Option<native::Layout>,
    metadata: Option<Metadata>,
    audio: Option<LiveAudio>,
    images: VecDeque<Decoded>,
    terminal: Option<Result<(), SourceError>>,
}

struct State {
    inputs: Vec<InputState>,
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    delivery: Option<native::Layout>,
}

struct Shared {
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    native_requested: bool,
    state: Mutex<State>,
    wake: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Camera-ordered RTSP video frames selected on one explicitly mapped media timeline.
/// Drop cancels all owned decoder workers and joins them unless [`Self::defer_cleanup`] assigned a cleanup scope. FFmpeg interrupts are cooperative; system DNS resolution is not a hard real-time deadline.
pub struct RtspSource {
    shared: Arc<Shared>,
    workers: Vec<DecoderTask>,
    stop: Arc<AtomicBool>,
    interrupted: Arc<AtomicBool>,
    start: Instant,
    config: RtspConfig,
    info: SourceInfo,
    full_range: bool,
    selector: Selector<Image>,
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    native: Option<native::Delivery>,
    rotations: [i32; 2],
    media_origin_us: i64,
    wall_shift_us: i64,
    slot: Option<i64>,
    waiting: bool,
    last_read: Instant,
    remainder: u64,
    output_origin_us: Option<i64>,
    audio: Option<LiveAudio>,
    exhausted: bool,
    failure: Option<SourceError>,
}

fn read_error(reason: impl ToString) -> SourceError {
    SourceError::Read {
        reason: reason.to_string(),
    }
}

fn elapsed_us(start: Instant) -> Result<i64, SourceError> {
    i64::try_from(start.elapsed().as_micros())
        .map_err(|_| read_error("RTSP running clock overflow"))
}

fn advance_slot(
    slot: i64,
    remainder: u64,
    fps: (u32, u32),
    after_us: i64,
) -> Result<(i64, u64), SourceError> {
    let distance = u128::try_from(i128::from(after_us) - i128::from(slot))
        .map_err(|_| read_error("RTSP cadence cannot move backwards"))?;
    let scale = 1_000_000u128 * u128::from(fps.1);
    let rate = u128::from(fps.0);
    let ticks = ((distance + 1) * rate - u128::from(remainder)).div_ceil(scale);
    let numerator = ticks * scale + u128::from(remainder);
    let step = i64::try_from(numerator / rate).map_err(|_| read_error("RTSP cadence overflow"))?;
    let next = slot
        .checked_add(step)
        .ok_or_else(|| read_error("RTSP cadence overflow"))?;
    Ok((next, (numerator % rate) as u64))
}

impl RtspSource {
    /// Open all camera inputs concurrently. Metadata is fixed for the source's lifetime; this call waits for an initial decoded image from every camera.
    /// Only `rtsp://` over TCP is accepted. Audio is not part of this video source.
    pub fn open(
        inputs: Vec<RtspInput>,
        config: RtspConfig,
        interrupted: Arc<AtomicBool>,
    ) -> Result<Self, SourceError> {
        Self::open_with_audio(
            inputs,
            config,
            interrupted,
            None,
            #[cfg(feature = "gpu")]
            None,
        )
    }

    pub(crate) fn open_with_audio(
        inputs: Vec<RtspInput>,
        config: RtspConfig,
        interrupted: Arc<AtomicBool>,
        audio_camera: Option<usize>,
        #[cfg(feature = "gpu")] gpu: Option<&reco_core::gpu::GpuContext>,
    ) -> Result<Self, SourceError> {
        #[cfg(all(feature = "gpu", not(target_os = "linux")))]
        let _ = gpu;
        let invalid = |reason: &str| SourceError::Init {
            path: "RTSP capture".into(),
            reason: reason.into(),
        };
        if inputs.is_empty()
            || inputs.iter().any(|input| {
                !input.url.starts_with("rtsp://")
                    || input.url.len() == 7
                    || input.url.bytes().any(|b| b == 0 || b.is_ascii_control())
            })
        {
            return Err(invalid(
                "expected nonempty rtsp:// inputs without control characters",
            ));
        }
        if audio_camera.is_some_and(|camera| camera >= inputs.len()) {
            return Err(invalid("audio camera index exceeds the RTSP input count"));
        }
        if config.fps.0 == 0
            || config.fps.1 == 0
            || config.fps.0 > i32::MAX as u32
            || config.fps.1 > i32::MAX as u32
            || u64::from(config.fps.0) > 1_000_000 * u64::from(config.fps.1)
        {
            return Err(invalid("invalid RTSP output cadence"));
        }
        if [config.open_timeout, config.read_timeout]
            .iter()
            .any(|duration| duration.as_micros() == 0 || duration.as_micros() > i64::MAX as u128)
        {
            return Err(invalid(
                "RTSP timeouts must be positive representable microseconds",
            ));
        }
        let selector = Selector::new(inputs.len(), config.synchronization).map_err(read_error)?;
        if interrupted.load(Ordering::Relaxed) {
            return Err(invalid("RTSP opening cancelled"));
        }
        let shared = Arc::new(Shared {
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            native_requested: !config.software_decode
                && matches!(inputs.len(), 1 | 2)
                && gpu.is_some_and(|gpu| gpu.supports_zero_copy()),
            state: Mutex::new(State {
                inputs: (0..inputs.len()).map(|_| InputState::default()).collect(),
                #[cfg(all(feature = "gpu", target_os = "linux"))]
                delivery: None,
            }),
            wake: Condvar::new(),
        });
        let mut source = Self {
            shared,
            workers: Vec::new(),
            stop: Arc::new(AtomicBool::new(false)),
            interrupted,
            start: Instant::now(),
            config,
            info: SourceInfo {
                camera_sizes: Vec::new(),
                fps: f64::from(config.fps.0) / f64::from(config.fps.1),
                fps_rational: Some((config.fps.0 as i32, config.fps.1 as i32)),
                total_frames: None,
            },
            full_range: false,
            selector,
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            native: None,
            rotations: [0; 2],
            media_origin_us: 0,
            wall_shift_us: 0,
            slot: None,
            waiting: false,
            last_read: Instant::now(),
            remainder: 0,
            output_origin_us: None,
            audio: None,
            exhausted: false,
            failure: None,
        };
        for (camera, input) in inputs.into_iter().enumerate() {
            let shared = source.shared.clone();
            let stop = source.stop.clone();
            let interrupted = source.interrupted.clone();
            let start = source.start;
            let handle = std::thread::Builder::new()
                .name(format!("rtsp_{camera}"))
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        worker::decode(
                            camera,
                            input,
                            config,
                            audio_camera == Some(camera),
                            shared.clone(),
                            stop,
                            interrupted,
                            start,
                        )
                    }))
                    .unwrap_or_else(|_| {
                        Err(read_error(format!("RTSP input {camera} worker panicked")))
                    });
                    shared.lock().inputs[camera].terminal = Some(result);
                    shared.wake.notify_one();
                })
                .map_err(|e| invalid(&format!("cannot start RTSP worker: {e}")))?;
            source.workers.push(DecoderTask::new(handle));
        }
        let metadata = {
            let mut state = source.shared.lock();
            loop {
                source.check_errors(&state)?;
                if source.interrupted.load(Ordering::Relaxed) {
                    return Err(invalid("RTSP opening cancelled"));
                }
                if source.start.elapsed() >= config.open_timeout {
                    return Err(invalid("RTSP initial image deadline exceeded"));
                }
                if state.inputs.iter().all(|input| input.metadata.is_some()) {
                    break state
                        .inputs
                        .iter()
                        .map(|input| input.metadata.unwrap())
                        .collect::<Vec<_>>();
                }
                if state
                    .inputs
                    .iter()
                    .any(|input| input.terminal.is_some() && input.metadata.is_none())
                {
                    return Err(invalid("RTSP input ended before its initial image"));
                }
                state = source
                    .shared
                    .wake
                    .wait_timeout(state, Duration::from_millis(10))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        };
        source.full_range = metadata[0].full_range;
        source.audio =
            audio_camera.and_then(|camera| source.shared.lock().inputs[camera].audio.clone());
        if metadata
            .iter()
            .any(|info| info.full_range != source.full_range)
        {
            return Err(invalid("RTSP camera color ranges differ"));
        }
        source.info.camera_sizes = metadata.iter().map(|info| info.size).collect();
        for (rotation, info) in source.rotations.iter_mut().zip(&metadata) {
            *rotation = info.rotation;
        }
        #[cfg(all(feature = "gpu", target_os = "linux"))]
        let delivery = source.shared.lock().delivery;
        #[cfg(all(feature = "gpu", target_os = "linux"))]
        if let Some(native::Layout::Cuda(format)) = delivery {
            source.native = Some(native::Delivery::new(
                gpu.unwrap(),
                &source.info.camera_sizes,
                format,
            )?);
        }
        source.media_origin_us = metadata
            .iter()
            .map(|info| info.first_time_us)
            .min()
            .unwrap();
        // Only establish local playout phase. Relative camera times remain those
        // supplied by the explicit offsets, never aligned by their arrival order.
        let shift = metadata
            .iter()
            .map(|info| {
                i128::from(info.first_time_us)
                    - i128::from(source.media_origin_us)
                    - i128::from(info.first_arrived_us)
            })
            .max()
            .unwrap();
        source.wall_shift_us =
            i64::try_from(shift).map_err(|_| invalid("RTSP clock origin overflow"))?;
        Ok(source)
    }

    /// Cancel immediately on drop but let the supplied scope join retired decoder workers.
    /// Interactive consumers can replace a source without waiting for native teardown. Drop the source before the scope.
    pub fn defer_cleanup(&mut self, cleanup: &crate::DecoderCleanup) {
        for worker in &mut self.workers {
            worker.defer_to(cleanup);
        }
    }

    pub(crate) fn audio(&self) -> Option<LiveAudio> {
        self.audio.clone()
    }

    fn check_errors(&self, state: &State) -> Result<(), SourceError> {
        for input in &state.inputs {
            if let Some(Err(error)) = &input.terminal {
                return Err(error.clone());
            }
        }
        Ok(())
    }

    fn now_us(&self) -> Result<i64, SourceError> {
        elapsed_us(self.start)?
            .checked_add(self.wall_shift_us)
            .ok_or_else(|| read_error("RTSP running clock overflow"))
    }

    fn advance(&mut self, after_us: i64) -> Result<(), SourceError> {
        let (slot, remainder) = advance_slot(
            self.slot.unwrap(),
            self.remainder,
            self.config.fps,
            after_us,
        )?;
        self.slot = Some(slot);
        self.remainder = remainder;
        Ok(())
    }

    fn receive(
        &mut self,
        blocking: bool,
        mut resuming: bool,
    ) -> Result<Option<SourceFrame>, SourceError> {
        loop {
            let shared = self.shared.clone();
            let mut state = shared.lock();
            self.check_errors(&state)?;
            if self.interrupted.load(Ordering::Relaxed) {
                self.exhausted = true;
                return Ok(None);
            }
            if self.slot.is_none() {
                // Discovery can decode a buffered burst faster than media time.
                // Establish one common playout phase from its freshest receipt,
                // preserving every camera's explicit relative timestamp mapping.
                for frame in state.inputs.iter().flat_map(|input| &input.images) {
                    let shift = i128::from(frame.time_us)
                        - i128::from(self.media_origin_us)
                        - i128::from(frame.arrived_us);
                    self.wall_shift_us = self.wall_shift_us.max(
                        i64::try_from(shift)
                            .map_err(|_| read_error("RTSP clock origin overflow"))?,
                    );
                }
            }
            let mut latest = None::<i64>;
            for (camera, input) in state.inputs.iter_mut().enumerate() {
                while let Some(frame) = input.images.pop_front() {
                    let time_us = frame
                        .time_us
                        .checked_sub(self.media_origin_us)
                        .ok_or_else(|| read_error("RTSP mapped time overflow"))?;
                    let arrived = frame
                        .arrived_us
                        .checked_add(self.wall_shift_us)
                        .ok_or_else(|| read_error("RTSP receipt time overflow"))?;
                    latest = Some(latest.map_or(time_us, |last| last.max(time_us)));
                    // An early decoded burst can run ahead of local playout.
                    // It becomes eligible at its media time, never sooner.
                    self.selector
                        .push(
                            camera,
                            Capture {
                                time_us,
                                image: frame.image,
                            },
                            arrived.max(time_us),
                        )
                        .map_err(read_error)?;
                }
            }
            if state.inputs.iter().enumerate().any(|(camera, input)| {
                input.terminal.is_some() && !self.selector.has_pending(camera)
            }) {
                self.exhausted = true;
                return Ok(None);
            }
            let now = self.now_us()?;
            if self.slot.is_none() {
                self.slot = latest;
            }
            let mut wait_us = 10_000;
            if let Some(mut slot) = self.slot {
                if resuming || !self.waiting {
                    if now.saturating_sub(slot) > self.config.synchronization.wait_us {
                        if self.waiting {
                            // An interactive consumer may pause between polls. Expire
                            // that pending deadline before resuming at a fresh slot.
                            match self.selector.poll(slot, now) {
                                Ok(Some(_)) => {}
                                Err(
                                    crate::frame_sync::Error::Clock
                                    | crate::frame_sync::Error::Config,
                                ) => {
                                    return Err(read_error("RTSP selector clock changed"));
                                }
                                _ => self.selector.skip_pending().map_err(read_error)?,
                            }
                        }
                        self.advance(now - 1)?;
                        slot = self.slot.unwrap();
                    }
                    self.waiting = true;
                    resuming = false;
                }
                match self.selector.poll(slot, now).map_err(read_error)? {
                    Some(selection) => {
                        drop(state);
                        self.waiting = false;
                        self.advance(selection.at_us)?;
                        let origin = *self.output_origin_us.get_or_insert(selection.at_us);
                        if let Some(audio) = &self.audio {
                            audio.set_origin(
                                self.media_origin_us.checked_add(origin).ok_or_else(|| {
                                    read_error("RTSP audio clock origin overflow")
                                })?,
                            );
                        }
                        return Ok(Some(SourceFrame {
                            pts_us: Some(selection.at_us - origin),
                            capture: Some(CaptureTiming {
                                at_us: selection.at_us,
                                times_us: selection.captures.iter().map(|c| c.time_us).collect(),
                                max_age_us: Some(self.config.synchronization.max_age_us as u64),
                            }),
                            images: self.deliver(selection.captures)?,
                        }));
                    }
                    None => {
                        let deadline = slot
                            .checked_add(self.config.synchronization.wait_us)
                            .ok_or_else(|| read_error("RTSP slot deadline overflow"))?;
                        wait_us = if now < slot {
                            slot - now
                        } else {
                            deadline - now
                        };
                    }
                }
            }
            if !blocking {
                return Ok(None);
            }
            drop(
                shared
                    .wake
                    .wait_timeout(
                        state,
                        Duration::from_micros(wait_us.clamp(1, 10_000) as u64),
                    )
                    .unwrap_or_else(|e| e.into_inner()),
            );
        }
    }

    fn read(&mut self, blocking: bool) -> Result<Option<SourceFrame>, SourceError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if self.exhausted {
            return Ok(None);
        }
        let interval_us =
            (1_000_000u64 * u64::from(self.config.fps.1)).div_ceil(u64::from(self.config.fps.0));
        let resuming = self.last_read.elapsed()
            > Duration::from_micros(interval_us.max(self.config.synchronization.wait_us as u64));
        self.last_read = Instant::now();
        let result = self.receive(blocking, resuming);
        match &result {
            Ok(None) if self.exhausted => {
                self.stop.store(true, Ordering::Relaxed);
            }
            Err(error) => {
                self.failure = Some(error.clone());
                self.stop.store(true, Ordering::Relaxed);
            }
            Ok(_) => {}
        }
        result
    }

    fn deliver(&self, captures: Vec<Capture<Image>>) -> Result<FrameSet, SourceError> {
        #[cfg(all(feature = "gpu", target_os = "linux"))]
        if let Some(native) = &self.native {
            return native.stage(&captures);
        }
        let images = captures
            .into_iter()
            .map(|capture| match capture.image {
                Image::Host(image) => Ok(image),
                #[cfg(all(feature = "gpu", target_os = "linux"))]
                Image::Cuda(_) => Err(read_error("RTSP host image residency changed")),
            })
            .collect::<Result<_, _>>()?;
        Ok(FrameSet::Yuv420p(images))
    }

    #[cfg(all(feature = "gpu", target_os = "linux"))]
    pub(crate) fn shared_texture_set(&self) -> Option<&reco_core::interop::SharedTextureSet> {
        self.native.as_ref().map(|native| &native.shared)
    }

    pub(crate) fn decode_mode(&self) -> &'static str {
        if self.is_gpu_resident() {
            "RTSP CUDA/Vulkan"
        } else {
            "RTSP host YUV420P"
        }
    }
}

impl FrameSource for RtspSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }
    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        self.read(true)
    }
    fn try_next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        if self.is_gpu_resident() {
            return Err(read_error(
                "native RTSP delivery requires session-driven processing",
            ));
        }
        self.read(false)
    }
    fn is_gpu_resident(&self) -> bool {
        #[cfg(all(feature = "gpu", target_os = "linux"))]
        return self.native.is_some();
        #[cfg(not(all(feature = "gpu", target_os = "linux")))]
        false
    }
    #[cfg(feature = "gpu")]
    fn gpu_pixel_format(&self) -> reco_core::render::renderer::GpuPixelFormat {
        #[cfg(target_os = "linux")]
        if let Some(native) = &self.native {
            return native.format;
        }
        reco_core::render::renderer::GpuPixelFormat::Nv12
    }
    fn left_rotation(&self) -> i32 {
        self.rotations[0]
    }
    fn right_rotation(&self) -> i32 {
        self.rotations[1]
    }
    fn is_full_range(&self) -> bool {
        self.full_range
    }
    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

impl Drop for RtspSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.shared.wake.notify_all();
    }
}
