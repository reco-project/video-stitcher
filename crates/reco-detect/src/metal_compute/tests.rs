//! Native Metal tensor checks using IOSurface-backed constant-colour inputs.

use super::*;
use reco_core::interop::metal::RetainedCVPixelBuffer;
use std::ffi::c_void;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFDictionaryCreate(
        allocator: *const c_void,
        keys: *const *const c_void,
        values: *const *const c_void,
        count: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> *const c_void;
    fn CFRelease(value: *const c_void);
    // Only the addresses of these opaque callback tables are passed to CoreFoundation.
    static kCFTypeDictionaryKeyCallBacks: u8;
    static kCFTypeDictionaryValueCallBacks: u8;
    static kCFBooleanTrue: *const c_void;
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    static kCVPixelBufferIOSurfacePropertiesKey: *const c_void;
    static kCVPixelBufferMetalCompatibilityKey: *const c_void;
    fn CVPixelBufferCreate(
        allocator: *const c_void,
        width: usize,
        height: usize,
        format: u32,
        attributes: *const c_void,
        output: *mut CVPixelBufferRef,
    ) -> i32;
    fn CVPixelBufferRelease(buffer: CVPixelBufferRef);
    fn CVPixelBufferLockBaseAddress(buffer: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(buffer: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddressOfPlane(buffer: CVPixelBufferRef, plane: usize) -> *mut u8;
    fn CVPixelBufferGetBytesPerRowOfPlane(buffer: CVPixelBufferRef, plane: usize) -> usize;
}

fn image(size: (u32, u32), format: u32, ten_bit: bool) -> RetainedCVPixelBuffer {
    unsafe {
        let keys_cb = std::ptr::addr_of!(kCFTypeDictionaryKeyCallBacks).cast();
        let values_cb = std::ptr::addr_of!(kCFTypeDictionaryValueCallBacks).cast();
        let empty = CFDictionaryCreate(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            keys_cb,
            values_cb,
        );
        assert!(!empty.is_null());
        let keys = [
            kCVPixelBufferIOSurfacePropertiesKey,
            kCVPixelBufferMetalCompatibilityKey,
        ];
        let values = [empty, kCFBooleanTrue];
        let attributes = CFDictionaryCreate(
            std::ptr::null(),
            keys.as_ptr(),
            values.as_ptr(),
            2,
            keys_cb,
            values_cb,
        );
        assert!(!attributes.is_null());
        let mut ptr = std::ptr::null_mut();
        let status = CVPixelBufferCreate(
            std::ptr::null(),
            size.0 as usize,
            size.1 as usize,
            format,
            attributes,
            &mut ptr,
        );
        CFRelease(attributes);
        CFRelease(empty);
        assert_eq!(status, 0);
        assert!(!ptr.is_null());
        let buffer = RetainedCVPixelBuffer::retain(ptr);
        CVPixelBufferRelease(ptr);
        assert_eq!(CVPixelBufferLockBaseAddress(ptr, 0), 0);
        for plane in 0..2 {
            let base = CVPixelBufferGetBaseAddressOfPlane(ptr, plane);
            let stride = CVPixelBufferGetBytesPerRowOfPlane(ptr, plane);
            assert!(!base.is_null());
            let height = size.1 as usize / if plane == 0 { 1 } else { 2 };
            for y in 0..height {
                for x in 0..size.0 as usize {
                    let value = if plane == 0 { 64u8 } else { 128u8 };
                    if ten_bit {
                        base.add(y * stride + x * 2)
                            .cast::<u16>()
                            .write(u16::from(value) << 8);
                    } else {
                        base.add(y * stride + x).write(value);
                    }
                }
            }
        }
        assert_eq!(CVPixelBufferUnlockBaseAddress(ptr, 0), 0);
        buffer
    }
}

#[test]
#[ignore = "requires Metal and IOSurface-backed CoreVideo NV12/P010 buffers"]
fn native_preprocessing_reuses_tensor_and_matches_integer_padding_for_each_camera() {
    let gpu = GpuContext::new_blocking().unwrap();
    let mut pipeline = MetalPreprocessPipeline::new(&gpu, 8).unwrap();
    let mut address = None;
    for (format, ten_bit, full_range) in [
        (0x34323066, false, true),
        (0x34323076, false, false),
        (0x78663230, true, true),
        (0x78343230, true, false),
    ] {
        for size in [(6, 4), (4, 6), (8, 8), (16, 8), (6, 4)] {
            let image = image(size, format, ten_bit);
            let layout = crate::letterbox::Letterbox::new(size, 8).unwrap();
            let tensor = unsafe { pipeline.preprocess(image.as_ptr(), size, &gpu) }.unwrap();
            if let Some(previous) = address {
                assert_eq!(tensor.as_ptr(), previous);
            }
            address = Some(tensor.as_ptr());
            let normalizer = if ten_bit {
                256.0 / 65535.0
            } else {
                1.0 / 255.0
            };
            let (mut y, mut chroma) = (64.0 * normalizer, 128.0 * normalizer - 0.5);
            if !full_range {
                y = (y - 0.0627451) * 1.164384;
                chroma *= 1.138393;
            }
            let rgb: [f32; 3] = [
                y + 1.5748 * chroma,
                y - (0.1873 + 0.4681) * chroma,
                y + 1.8556 * chroma,
            ];
            for row in 0..8 {
                for col in 0..8 {
                    let content = (layout.pad_x..layout.pad_x + layout.width).contains(&col)
                        && (layout.pad_y..layout.pad_y + layout.height).contains(&row);
                    for channel in 0..3 {
                        let expected = if content {
                            rgb[channel].clamp(0.0, 1.0)
                        } else {
                            114.0 / 255.0
                        };
                        assert!(
                            (tensor[channel * 64 + (row * 8 + col) as usize] - expected).abs()
                                < 0.0002,
                            "format={format:x}, size={size:?}, pixel=({col},{row}), channel={channel}"
                        );
                    }
                }
            }
            assert!(
                unsafe { pipeline.preprocess(image.as_ptr(), (size.0 + 2, size.1), &gpu) }.is_err()
            );
        }
    }
    assert!(MetalPreprocessPipeline::new(&gpu, 0).is_err());
    assert!(MetalPreprocessPipeline::new(&gpu, u32::MAX).is_err());
}
