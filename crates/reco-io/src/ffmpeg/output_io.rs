//! Owned muxer I/O cancellation and asynchronous transport failure retention.

use std::ffi::{CString, c_char, c_int, c_void};
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ffmpeg_next::{self as ffmpeg, Dictionary, Packet, format, sys};

use super::encoder::{EncodeError, EncoderConfig};

// Native failures are negative; reserve a positive value for caller cancellation.
const CANCELLED: i32 = 1;

struct Control {
    cancelled: Option<Arc<AtomicBool>>,
    stopped: AtomicBool,
    start: Instant,
    timeout: Duration,
    operation_deadline: AtomicU64,
    finish_deadline: AtomicU64,
    error: AtomicI32,
}

impl Control {
    fn deadline(&self) -> u64 {
        self.start
            .elapsed()
            .saturating_add(self.timeout)
            .as_millis()
            .min(u64::MAX as u128) as u64
    }

    fn record(&self, result: c_int) {
        if result < 0 {
            let _ = self
                .error
                .compare_exchange(0, result, Ordering::Relaxed, Ordering::Relaxed);
        }
    }

    fn status(&self) -> Result<(), EncodeError> {
        let error = self.error.load(Ordering::Relaxed);
        if error == 0 {
            Ok(())
        } else if error == CANCELLED {
            Err(EncodeError::Cancelled)
        } else {
            Err(ffmpeg::Error::from(error).into())
        }
    }
}

extern "C" fn interrupted(opaque: *mut c_void) -> c_int {
    // SAFETY: OutputIo owns the boxed control until the context and FIFO worker are destroyed.
    let control = unsafe { &*opaque.cast::<Control>() };
    if control
        .cancelled
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
    {
        let _ = control
            .error
            .compare_exchange(0, CANCELLED, Ordering::Relaxed, Ordering::Relaxed);
        return 1;
    }
    let deadline = control
        .operation_deadline
        .load(Ordering::Relaxed)
        .min(control.finish_deadline.load(Ordering::Relaxed));
    let stop = control.stopped.load(Ordering::Relaxed)
        || control.start.elapsed().as_millis() >= u128::from(deadline);
    if stop {
        control.record(ffmpeg::Error::Exit.into());
    }
    stop as c_int
}

unsafe extern "C" fn open_io(
    context: *mut sys::AVFormatContext,
    io: *mut *mut sys::AVIOContext,
    url: *const c_char,
    flags: c_int,
    options: *mut *mut sys::AVDictionary,
) -> c_int {
    // SAFETY: FFmpeg passes a live context, including for FIFO's nested context, with our opaque control.
    unsafe {
        let control = &*(*context).opaque.cast::<Control>();
        control
            .operation_deadline
            .store(control.deadline(), Ordering::Relaxed);
        let result = sys::avio_open2(io, url, flags, &(*context).interrupt_callback, options);
        control
            .operation_deadline
            .store(u64::MAX, Ordering::Relaxed);
        control.record(result);
        result
    }
}

unsafe extern "C" fn close_io(
    context: *mut sys::AVFormatContext,
    io: *mut sys::AVIOContext,
) -> c_int {
    // SAFETY: FFmpeg transfers this opened I/O context for a single close. Read its error before freeing it.
    unsafe {
        let control = &*(*context).opaque.cast::<Control>();
        if !io.is_null() {
            control.record((*io).error);
        }
        let result = sys::avio_close(io);
        control.record(result);
        result
    }
}

pub(super) struct OutputIo {
    // Drop order keeps callbacks alive through AVIO close and native worker teardown.
    context: format::context::Output,
    control: Box<Control>,
    network: bool,
    fifo: bool,
    header_written: bool,
}

impl OutputIo {
    pub(super) fn open(
        path: &Path,
        muxer: &str,
        config: &EncoderConfig,
        fifo: bool,
    ) -> Result<Self, EncodeError> {
        let network = crate::output::is_network_destination(path);
        let path = path
            .to_str()
            .ok_or_else(|| EncodeError::InvalidConfig("output path is not UTF-8".into()))?;
        let url = CString::new(path)
            .map_err(|_| EncodeError::InvalidConfig("output contains NUL".into()))?;
        let muxer = CString::new(if fifo { "fifo" } else { muxer }).expect("static muxer name");
        let control = Box::new(Control {
            cancelled: network.then(|| config.interrupted.clone()).flatten(),
            stopped: AtomicBool::new(false),
            start: Instant::now(),
            timeout: config.network_timeout.unwrap_or(Duration::from_secs(10)),
            operation_deadline: AtomicU64::new(u64::MAX),
            finish_deadline: AtomicU64::new(u64::MAX),
            error: AtomicI32::new(0),
        });
        // SAFETY: allocate a unique context, wrap immediately so failed opens also release it.
        unsafe {
            let mut raw = ptr::null_mut();
            let result = sys::avformat_alloc_output_context2(
                &mut raw,
                ptr::null(),
                muxer.as_ptr(),
                url.as_ptr(),
            );
            if result < 0 {
                return Err(ffmpeg::Error::from(result).into());
            }
            let mut output = Self {
                context: format::context::Output::wrap(raw),
                control,
                network,
                fifo,
                header_written: false,
            };
            if network {
                (*raw).opaque = (&mut *output.control as *mut Control).cast();
                (*raw).interrupt_callback = sys::AVIOInterruptCB {
                    callback: Some(interrupted),
                    opaque: (*raw).opaque,
                };
                (*raw).io_open = Some(open_io);
                (*raw).io_close2 = Some(close_io);
            }
            if (*(*raw).oformat).flags & sys::AVFMT_NOFILE == 0 {
                let result = if network {
                    if interrupted((*raw).opaque) != 0 {
                        output.control.record(ffmpeg::Error::Exit.into());
                        return output.control.status().map(|()| output);
                    }
                    open_io(
                        raw,
                        &mut (*raw).pb,
                        url.as_ptr(),
                        sys::AVIO_FLAG_WRITE,
                        ptr::null_mut(),
                    )
                } else {
                    sys::avio_open(&mut (*raw).pb, url.as_ptr(), sys::AVIO_FLAG_WRITE)
                };
                output.control.record(result);
                output.control.status()?;
            }
            Ok(output)
        }
    }

    fn operation<T>(
        &mut self,
        operation: impl FnOnce(&mut format::context::Output) -> Result<T, ffmpeg::Error>,
    ) -> Result<T, EncodeError> {
        self.control.status()?;
        if self.network && !self.fifo {
            self.control
                .operation_deadline
                .store(self.control.deadline(), Ordering::Relaxed);
        }
        let result = operation(&mut self.context);
        if self.network && !self.fifo {
            self.control
                .operation_deadline
                .store(u64::MAX, Ordering::Relaxed);
        }
        if let Err(error) = &result {
            self.control.record((*error).into());
        }
        self.control.status()?;
        result.map_err(Into::into)
    }

    pub(super) fn write_header(&mut self, mut options: Dictionary<'_>) -> Result<(), EncodeError> {
        if self.fifo {
            options.set("fifo_format", "flv");
            options.set("queue_size", "60");
            options.set("drop_pkts_on_overflow", "1");
            options.set("restart_with_keyframe", "1");
            options.set("attempt_recovery", "0");
        }
        if self.network && !self.fifo {
            self.control
                .operation_deadline
                .store(self.control.deadline(), Ordering::Relaxed);
        }
        let result = self.context.write_header_with(options);
        // A successful FIFO header starts a worker even if it immediately reports an I/O failure.
        self.header_written = result.is_ok();
        if self.network && !self.fifo {
            self.control
                .operation_deadline
                .store(u64::MAX, Ordering::Relaxed);
        }
        if let Err(error) = result {
            self.control.record(error.into());
        }
        self.control.status()
    }

    pub(super) fn write_packet(&mut self, packet: &mut Packet) -> Result<(), EncodeError> {
        self.operation(|context| packet.write_interleaved(context))
    }

    pub(super) fn flush(&mut self) -> Result<(), EncodeError> {
        self.operation(|context| {
            // SAFETY: unique open muxer; null packets request flush, not finalization.
            unsafe {
                let raw = context.as_mut_ptr();
                let result = sys::av_interleaved_write_frame(raw, ptr::null_mut());
                if result < 0 {
                    return Err(result.into());
                }
                let result = sys::av_write_frame(raw, ptr::null_mut());
                if result < 0 {
                    return Err(result.into());
                }
                if !(*raw).pb.is_null() {
                    sys::avio_flush((*raw).pb);
                    if (*(*raw).pb).error < 0 {
                        return Err((*(*raw).pb).error.into());
                    }
                }
            }
            Ok(())
        })
    }

    pub(super) fn finish(&mut self) -> Result<(), EncodeError> {
        if std::mem::take(&mut self.header_written) {
            if self.network {
                self.control
                    .finish_deadline
                    .store(self.control.deadline(), Ordering::Relaxed);
            }
            // Always join/finalize even after a sticky worker error. The close callback retains native failures FIFO's trailer can omit.
            if let Err(error) = self.context.write_trailer() {
                self.control.record(error.into());
            }
        }
        self.control.status()
    }
}

impl Deref for OutputIo {
    type Target = format::context::Output;
    fn deref(&self) -> &Self::Target {
        &self.context
    }
}
impl DerefMut for OutputIo {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.context
    }
}
impl Drop for OutputIo {
    fn drop(&mut self) {
        self.control.stopped.store(true, Ordering::Relaxed);
        let _ = self.finish();
    }
}

#[cfg(test)]
mod tests;
