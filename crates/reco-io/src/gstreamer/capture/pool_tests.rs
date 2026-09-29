//! Finite GStreamer pools exercise mapped-buffer ownership without native import.

use super::*;
use crate::frame_sync::Config;
use gstreamer_app::AppSrc;
use std::sync::Arc;

type Mapped = Arc<gst::MappedBuffer<gst::buffer::Readable>>;

struct Fixture {
    pair: CapturePair<Mapped>,
    sources: [AppSrc; 2],
    pools: [gst::BufferPool; 2],
    size: u32,
}

impl Fixture {
    fn new(size: u32) -> Self {
        let input = "appsrc name=source is-live=true format=time caps=video/x-raw,format=NV12,width=4,height=2,framerate=30/1 ! appsink name=sink sync=false";
        let pair = CapturePair::start(
            [input; 2],
            |sample| {
                sample
                    .buffer()
                    .unwrap()
                    .to_owned()
                    .into_mapped_buffer_readable()
                    .map(Arc::new)
                    .map_err(|_| SourceError::Read {
                        reason: "map failed".into(),
                    })
            },
            Some((
                Config {
                    wait_us: 20_000,
                    skew_us: 20_000,
                    max_age_us: 150_000,
                    capacity: 2,
                },
                30,
            )),
        )
        .unwrap();
        let sources = pair.test_app_sources();
        let pools = std::array::from_fn(|_| {
            let pool = gst::BufferPool::new();
            let mut config = pool.config();
            config.set_params(None, 12, size, size);
            pool.set_config(config).unwrap();
            pool.set_active(true).unwrap();
            pool
        });
        Self {
            pair,
            sources,
            pools,
            size,
        }
    }

    fn acquire(&self, camera: usize) -> Option<gst::Buffer> {
        let options =
            gst::BufferPoolAcquireParams::with_flags(gst::BufferPoolAcquireFlags::DONTWAIT);
        match self.pools[camera].acquire_buffer(Some(&options)) {
            Ok(buffer) => Some(buffer),
            Err(gst::FlowError::Eos) => None,
            Err(error) => panic!("pool acquisition failed: {error}"),
        }
    }

    fn push(&self, camera: usize, marker: u8, time: i64) {
        let mut buffer = self.acquire(camera).expect("pool starved");
        let writable = buffer.get_mut().unwrap();
        writable.set_pts(gst::ClockTime::from_useconds(time as u64));
        writable.map_writable().unwrap().as_mut_slice().fill(marker);
        self.sources[camera].push_buffer(buffer).unwrap();
    }

    fn all_recycled(&mut self) {
        self.pair.stop().unwrap();
        for camera in 0..2 {
            let buffers: Vec<_> = (0..self.size)
                .map(|_| self.acquire(camera).expect("unreleased pool buffer"))
                .collect();
            assert!(self.acquire(camera).is_none());
            drop(buffers);
            self.pools[camera].set_active(false).unwrap();
        }
    }
}

#[test]
fn finite_pools_survive_reuse_and_release_every_mapping_on_stop() {
    for size in [2, 4] {
        for missing in 0..2 {
            let mut fixture = Fixture::new(size);
            let mut previous: Option<CapturedPair<Mapped>> = None;
            for index in 0..12u8 {
                let time = sync::clock_us(&fixture.pair.capture.pipeline).unwrap();
                for camera in 0..2 {
                    if index % 4 != 1 || camera != missing {
                        fixture.push(camera, index * 2 + camera as u8, time);
                    }
                }
                let frame = fixture.pair.next(&AtomicBool::new(false)).unwrap().unwrap();
                for camera in 0..2 {
                    let held = index % 4 == 1 && camera == missing;
                    let expected = (if held { index - 1 } else { index }) * 2 + camera as u8;
                    assert!(
                        frame.images[camera]
                            .as_slice()
                            .iter()
                            .all(|&byte| byte == expected)
                    );
                    if held {
                        assert!(Arc::ptr_eq(
                            &frame.images[camera],
                            &previous.as_ref().unwrap().images[camera]
                        ));
                        assert_eq!(
                            frame.capture.times_us[camera],
                            previous.as_ref().unwrap().capture.times_us[camera]
                        );
                    }
                }
                previous = Some(frame);
            }
            drop(previous);
            fixture.all_recycled();
        }
    }
}

#[test]
fn a_single_buffer_cannot_refresh_a_retained_image_and_expires_cleanly() {
    let mut fixture = Fixture::new(1);
    let time = sync::clock_us(&fixture.pair.capture.pipeline).unwrap();
    for camera in 0..2 {
        fixture.push(camera, 42, time);
    }
    drop(fixture.pair.next(&AtomicBool::new(false)).unwrap().unwrap());
    assert!(fixture.acquire(0).is_none());
    assert!(fixture.acquire(1).is_none());
    let error = loop {
        match fixture.pair.next(&AtomicBool::new(false)) {
            Ok(Some(_)) => {}
            Err(error) => break error,
            Ok(None) => panic!("unexpected EOS"),
        }
    };
    assert!(error.to_string().contains("age limit"), "{error}");
    fixture.all_recycled();
}

#[test]
fn four_buffer_pools_progress_with_three_downstream_images_retained() {
    let mut fixture = Fixture::new(4);
    let mut reads = std::collections::VecDeque::new();
    for index in 0..18u8 {
        let time = sync::clock_us(&fixture.pair.capture.pipeline).unwrap();
        for camera in 0..2 {
            fixture.push(camera, index, time);
        }
        let frame = fixture.pair.next(&AtomicBool::new(false)).unwrap().unwrap();
        assert!(
            frame
                .images
                .iter()
                .all(|image| image.as_slice().iter().all(|&byte| byte == index))
        );
        reads.push_back(frame);
        if reads.len() > 3 {
            reads.pop_front();
        }
    }
    drop(reads);
    fixture.all_recycled();
}

#[test]
fn capture_does_not_keep_an_unused_last_sample_cache() {
    let mut fixture = Fixture::new(2);
    let time = sync::clock_us(&fixture.pair.capture.pipeline).unwrap();
    for camera in 0..2 {
        fixture.push(camera, 42, time);
    }
    drop(fixture.pair.next(&AtomicBool::new(false)).unwrap().unwrap());
    for sink in &fixture.pair.capture.sinks {
        assert!(!sink.property::<bool>("enable-last-sample"));
        assert!(
            sink.property::<Option<gst::Sample>>("last-sample")
                .is_none()
        );
    }
    fixture.all_recycled();
}

#[test]
fn source_stop_does_not_recycle_a_downstream_mapping_early() {
    let mut fixture = Fixture::new(2);
    let time = sync::clock_us(&fixture.pair.capture.pipeline).unwrap();
    for camera in 0..2 {
        fixture.push(camera, 42, time);
    }
    let frame = fixture.pair.next(&AtomicBool::new(false)).unwrap().unwrap();
    fixture.pair.stop().unwrap();
    for camera in 0..2 {
        let spare = fixture.acquire(camera).unwrap();
        assert!(
            fixture.acquire(camera).is_none(),
            "displayed image must remain retained"
        );
        assert!(
            frame.images[camera]
                .as_slice()
                .iter()
                .all(|&byte| byte == 42)
        );
        drop(spare);
    }
    drop(frame);
    fixture.all_recycled();
}

#[test]
fn finite_pools_recover_after_capture_and_consumer_backpressure() {
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};
    struct Producer {
        stop: Arc<AtomicBool>,
        worker: Option<JoinHandle<()>>,
    }
    impl Drop for Producer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            self.worker.take().unwrap().join().unwrap();
        }
    }
    for size in [2, 4] {
        let mut fixture = Fixture::new(size);
        let stop = Arc::new(AtomicBool::new(false));
        let cancelled = stop.clone();
        let sources = fixture.sources.clone();
        let pools = fixture.pools.clone();
        let pipeline = fixture.pair.capture.pipeline.clone();
        let producer = Producer {
            stop,
            worker: Some(std::thread::spawn(move || {
                let options =
                    gst::BufferPoolAcquireParams::with_flags(gst::BufferPoolAcquireFlags::DONTWAIT);
                let mut next = Instant::now();
                while !cancelled.load(Ordering::Relaxed) {
                    let now = Instant::now();
                    if now < next {
                        std::thread::sleep((next - now).min(Duration::from_millis(1)));
                        continue;
                    }
                    next = now + Duration::from_micros(33_333);
                    for (camera, pool) in pools.iter().enumerate() {
                        match pool.acquire_buffer(Some(&options)) {
                            Ok(mut buffer) => {
                                let time = sync::clock_us(&pipeline).unwrap();
                                let buffer_ref = buffer.get_mut().unwrap();
                                buffer_ref.set_pts(gst::ClockTime::from_useconds(time as u64));
                                buffer_ref
                                    .map_writable()
                                    .unwrap()
                                    .as_mut_slice()
                                    .fill(camera as u8 + 42);
                                sources[camera].push_buffer(buffer).unwrap();
                            }
                            // A finite camera pool cannot capture until ownership returns.
                            Err(gst::FlowError::Eos) => {}
                            Err(error) => panic!("pool failed: {error}"),
                        }
                    }
                }
            })),
        };
        let mut previous = None;
        for delay_ms in [0, 350, 0, 180, 0] {
            std::thread::sleep(Duration::from_millis(delay_ms));
            let resumed = sync::clock_us(&fixture.pair.capture.pipeline).unwrap();
            for _ in 0..3 {
                let frame = fixture.pair.next(&AtomicBool::new(false)).unwrap().unwrap();
                assert!(previous.is_none_or(|last| frame.pts_us > last));
                previous = Some(frame.pts_us);
                assert!(frame.capture.at_us >= resumed - 55_000);
                for camera in 0..2 {
                    assert!(
                        frame.images[camera]
                            .as_slice()
                            .iter()
                            .all(|&byte| byte == camera as u8 + 42)
                    );
                }
            }
        }
        drop(producer);
        fixture.all_recycled();
    }
}
