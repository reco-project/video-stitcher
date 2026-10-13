//! Cancellable RTSP decode with bounded latest-image delivery.

use super::*;
use crate::ffmpeg::decoder::{DecodeError, VideoDecoder};
use ffmpeg_next::{Dictionary, format};
use std::path::Path;
use std::sync::atomic::AtomicU64;

struct Deadline {
    start: Instant,
    at_us: AtomicU64,
    stop: Arc<AtomicBool>,
    interrupted: Arc<AtomicBool>,
}

impl Deadline {
    fn arm(&self, timeout: Duration) {
        self.at_us.store(
            self.start
                .elapsed()
                .saturating_add(timeout)
                .as_micros()
                .min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
    }

    fn expired(&self) -> bool {
        self.start.elapsed().as_micros() >= u128::from(self.at_us.load(Ordering::Relaxed))
    }

    fn cancelled(&self) -> bool {
        self.stop.load(Ordering::Relaxed) || self.interrupted.load(Ordering::Relaxed)
    }
}

pub(super) fn decode(
    camera: usize,
    input: RtspInput,
    config: RtspConfig,
    capture_audio: bool,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    interrupted: Arc<AtomicBool>,
    start: Instant,
) -> Result<(), SourceError> {
    let control = Arc::new(Deadline {
        start,
        at_us: AtomicU64::new(0),
        stop,
        interrupted,
    });
    control.arm(config.open_timeout);
    let callback = control.clone();
    let mut options = Dictionary::new();
    options.set("rtsp_transport", "tcp");
    options.set(
        "allowed_media_types",
        if capture_audio {
            "video+audio"
        } else {
            "video"
        },
    );
    // Discover tracks without buffering a long frame-rate sample. Missing codec
    // dimensions are established by the first decoded image within open_timeout.
    options.set("analyzeduration", "100000");
    options.set("probesize", "1048576");
    options.set("fpsprobesize", "0");
    options.set("timeout", &config.read_timeout.as_micros().to_string());
    crate::init();
    let context = format::input_with_interrupt_and_dictionary(
        Path::new(&input.url),
        move || callback.cancelled() || callback.expired(),
        options,
    );
    let context = match context {
        Ok(context) => context,
        Err(_) if control.cancelled() => return Ok(()),
        Err(error) => {
            return Err(read_error(format!(
                "RTSP input {camera} open failed: {error}"
            )));
        }
    };
    let audio = capture_audio
        .then(|| LiveAudio::from_input(&context, input.timestamp_offset_us))
        .flatten();
    shared.lock().inputs[camera].audio = audio.clone();
    let mut decoder = VideoDecoder::from_live_input(context, config.software_decode, audio)
        .map_err(|e| read_error(format!("RTSP input {camera} decoder: {e}")))?;
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    let layout = native::negotiate(&shared, camera, &decoder, || {
        control.cancelled() || control.expired()
    })?;
    let delivery = "host YUV420P";
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    let delivery = if matches!(layout, native::Layout::Cuda(_)) {
        "retained CUDA"
    } else {
        delivery
    };
    log::info!(
        "RTSP input {camera}: {} decode, {delivery} delivery",
        decoder.backend()
    );
    let mut previous = None;
    loop {
        if control.cancelled() {
            return Ok(());
        }
        control.arm(if previous.is_none() {
            config.open_timeout.saturating_sub(start.elapsed())
        } else {
            config.read_timeout
        });
        let mut decode_image = || {
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            if let native::Layout::Cuda(format) = layout {
                return decoder
                    .next_frame_gpu()?
                    .map(|frame| {
                        if frame.pixel_format != format {
                            return Err(DecodeError::ConversionError(
                                "RTSP native plane format changed".into(),
                            ));
                        }
                        Ok((
                            frame.timestamp_us,
                            (frame.width, frame.height),
                            Image::Cuda(Arc::new(frame)),
                        ))
                    })
                    .transpose();
            }
            decoder.next_frame().map(|frame| {
                frame.map(|frame| {
                    let size = (frame.width, frame.height);
                    (
                        frame.timestamp_us,
                        size,
                        Image::Host(Arc::new(Yuv420pData {
                            size,
                            y: frame.y,
                            u: frame.u,
                            v: frame.v,
                        })),
                    )
                })
            })
        };
        let frame = match decode_image() {
            Ok(frame) => frame,
            Err(_) if control.cancelled() => return Ok(()),
            // RTSP probing may leave an untimed bootstrap image. Wait for a
            // timed image within the opening deadline; never fabricate its PTS.
            Err(DecodeError::MissingTimestamp)
                if previous.is_none() && start.elapsed() < config.open_timeout =>
            {
                continue;
            }
            Err(error) => {
                return Err(read_error(format!(
                    "RTSP input {camera} read failed: {error}"
                )));
            }
        };
        let Some((timestamp_us, size, image)) = frame else {
            return if control.cancelled() {
                Ok(())
            } else {
                Err(read_error(format!(
                    "RTSP input {camera} ended unexpectedly"
                )))
            };
        };
        let time_us = timestamp_us
            .checked_add(input.timestamp_offset_us)
            .ok_or_else(|| read_error(format!("RTSP input {camera} timestamp overflow")))?;
        if previous.is_some_and(|last| time_us <= last) {
            return Err(read_error(format!(
                "RTSP input {camera} presentation clock reset or stopped"
            )));
        }
        previous = Some(time_us);
        let arrived_us = elapsed_us(start)?;
        let mut state = shared.lock();
        let state = &mut state.inputs[camera];
        state.metadata.get_or_insert(Metadata {
            size,
            full_range: decoder.is_full_range(),
            first_time_us: time_us,
            first_arrived_us: arrived_us,
            rotation: decoder.rotation(),
        });
        // Validate before eviction: a clock reset must not disappear with its image.
        if state.images.len() == config.synchronization.capacity {
            state.images.pop_front();
        }
        state.images.push_back(Decoded {
            time_us,
            arrived_us,
            image,
        });
        shared.wake.notify_one();
    }
}
