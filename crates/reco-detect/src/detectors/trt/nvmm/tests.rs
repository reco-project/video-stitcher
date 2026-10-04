//! Injected NvBuf ABI with real CUDA storage. This does not exercise Jetson transforms.

use super::*;
use crate::detectors::trt::cuda::CudaBuffer;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

static SERIAL: Mutex<()> = Mutex::new(());
static CREATED: AtomicUsize = AtomicUsize::new(0);
static DESTROYED: AtomicUsize = AtomicUsize::new(0);
static FAILURE: AtomicUsize = AtomicUsize::new(0);
std::thread_local! {
    static SESSION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[repr(C)]
struct Surface {
    header: NvBufSurfaceHeader,
    _plane: Box<NvBufSurfaceParamsHeader>,
    _pixels: CudaBuffer,
}

unsafe extern "C" fn create(
    out: *mut *mut c_void,
    _: u32,
    params: *const NvBufSurfaceCreateParams,
) -> i32 {
    let params = unsafe { &*params };
    let pitch = params.width * 4 + 32;
    let Ok(pixels) = CudaBuffer::new((pitch * params.height) as usize) else {
        return -1;
    };
    let mut plane = Box::new(NvBufSurfaceParamsHeader {
        _width: params.width,
        _height: params.height,
        pitch,
        _color_format: NVBUF_COLOR_FORMAT_RGBA as u32,
        _layout: 0,
        _pad0: 0,
        _buffer_desc: -1,
        _data_size: pitch * params.height,
        _pad1: 0,
        data_ptr: pixels.ptr() as *mut c_void,
    });
    match FAILURE.load(Ordering::SeqCst) {
        4 => plane.data_ptr = std::ptr::null_mut(),
        5 => plane.pitch = params.width * 4 - 1,
        _ => {}
    }
    let mut surface = Box::new(Surface {
        header: NvBufSurfaceHeader {
            _gpu_id: 0,
            _batch_size: 1,
            num_filled: 0,
            _is_contiguous: 1,
            _pad0: [0; 3],
            _mem_type: NVBUF_MEM_CUDA_DEVICE as u32,
            _pad1: 0,
            surface_list: &mut *plane,
        },
        _plane: plane,
        _pixels: pixels,
    });
    if FAILURE.load(Ordering::SeqCst) == 3 {
        surface.header.surface_list = std::ptr::null_mut();
    }
    CREATED.fetch_add(1, Ordering::SeqCst);
    unsafe {
        *out = Box::into_raw(surface).cast();
    }
    0
}

unsafe extern "C" fn destroy(surface: *mut c_void) -> i32 {
    unsafe {
        drop(Box::from_raw(surface.cast::<Surface>()));
    }
    DESTROYED.fetch_add(1, Ordering::SeqCst);
    0
}

unsafe extern "C" fn set_session(params: *const NvBufSurfTransformSessionParams) -> i32 {
    let params = unsafe { &*params };
    if params.compute_mode != NVBUF_TRANSFORM_COMPUTE_GPU || FAILURE.load(Ordering::SeqCst) == 1 {
        return -1;
    }
    SESSION.set(true);
    0
}

unsafe extern "C" fn transform(
    source: *mut c_void,
    destination: *mut c_void,
    params: *const NvBufSurfTransformParams,
) -> i32 {
    if !SESSION.replace(false) || FAILURE.load(Ordering::SeqCst) == 2 {
        return -1;
    }
    let value = unsafe { *source.cast::<u8>() };
    let destination = unsafe { &*destination.cast::<Surface>() };
    let plane = &destination._plane;
    let rect = unsafe { &*(*params).dst_rect };
    for row in rect.top..rect.top + rect.height {
        let ptr = plane.data_ptr as u64 + u64::from(row * plane.pitch + rect.left * 4);
        if cuda_memset_d8(ptr, value, rect.width as usize * 4).is_err() {
            return -1;
        }
    }
    0
}

static FUNCTIONS: NvBufFunctions = NvBufFunctions {
    _libraries: None,
    create,
    destroy,
    set_session_params: set_session,
    transform,
};

#[test]
#[ignore = "requires CUDA; injected NvBuf functions, not Jetson validation"]
fn allocation_is_owned_before_grey_fill_can_fail() {
    let _serial = SERIAL.lock().unwrap();
    FAILURE.store(0, Ordering::SeqCst);
    let created = CREATED.load(Ordering::SeqCst);
    let destroyed = DESTROYED.load(Ordering::SeqCst);
    for size in [16, 640, 1280] {
        let result =
            NvmmPreprocessor::allocate(size, &FUNCTIONS, |_| Err("injected fill failure".into()));
        assert!(result.err().unwrap().to_string().contains("fill failure"));
    }
    for failure in [3, 4, 5] {
        FAILURE.store(failure, Ordering::SeqCst);
        assert!(
            NvmmPreprocessor::allocate(16, &FUNCTIONS, |_| panic!(
                "invalid metadata must fail before fill"
            ))
            .is_err()
        );
    }
    FAILURE.store(0, Ordering::SeqCst);
    assert_eq!(CREATED.load(Ordering::SeqCst) - created, 6);
    assert_eq!(DESTROYED.load(Ordering::SeqCst) - destroyed, 6);
}

#[test]
#[ignore = "requires CUDA; injected NvBuf functions, not Jetson validation"]
fn reuses_one_surface_with_clean_padding_and_thread_local_setup() {
    let _serial = SERIAL.lock().unwrap();
    FAILURE.store(0, Ordering::SeqCst);
    let created = CREATED.load(Ordering::SeqCst);
    let destroyed = DESTROYED.load(Ordering::SeqCst);
    let mut prepared =
        NvmmPreprocessor::allocate(16, &FUNCTIONS, NvmmPreprocessor::fill_grey).unwrap();
    let ptr = prepared.data_ptr;
    let pitch = prepared.pitch;
    let mut value = 200u8;
    for size in [(8, 4), (4, 8), (8, 4)] {
        let layout = InputLayout::new(size, 16);
        // The injected transform accepts an immutable byte as its source, not a native image.
        unsafe {
            prepared
                .transform((&mut value as *mut u8).cast(), layout)
                .unwrap();
        }
        let output = CudaBuffer::new(3 * 16 * 16 * 4).unwrap();
        crate::cuda_kernels::normalize_rgba_to_chw(ptr, pitch, output.ptr(), 16, 16).unwrap();
        let stream = crate::detectors::trt::cuda::CudaStream::new().unwrap();
        let mut pixels = vec![0.0; 3 * 16 * 16];
        output.copy_to_host(&mut pixels, &stream).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let inside = (layout.padding.0..layout.padding.0 + layout.resized.0).contains(&x)
                    && (layout.padding.1..layout.padding.1 + layout.resized.1).contains(&y);
                let expected = if inside { value } else { 114 } as f32 / 255.0;
                for channel in 0..3 {
                    assert!(
                        (pixels[channel * 256 + y as usize * 16 + x as usize] - expected).abs()
                            < 1e-6
                    );
                }
            }
        }
        assert_eq!(prepared.data_ptr, ptr);
        value -= 40;
    }
    for failure in [1, 2] {
        FAILURE.store(failure, Ordering::SeqCst);
        assert!(
            unsafe {
                prepared.transform((&mut value as *mut u8).cast(), InputLayout::new((8, 4), 16))
            }
            .is_err()
        );
    }
    FAILURE.store(0, Ordering::SeqCst);
    std::thread::spawn(move || {
        let mut value = 50u8;
        unsafe {
            prepared
                .transform((&mut value as *mut u8).cast(), InputLayout::new((8, 4), 16))
                .unwrap();
        }
        drop(prepared);
    })
    .join()
    .unwrap();
    assert_eq!(CREATED.load(Ordering::SeqCst) - created, 1);
    assert_eq!(DESTROYED.load(Ordering::SeqCst) - destroyed, 1);
}

#[test]
fn invalid_model_sizes_do_not_allocate() {
    for size in [0, u32::MAX] {
        assert!(NvmmPreprocessor::allocate(size, &FUNCTIONS, NvmmPreprocessor::fill_grey).is_err());
    }
}
