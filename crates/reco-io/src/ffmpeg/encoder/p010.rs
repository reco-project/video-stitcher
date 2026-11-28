//! Pack P010LE input into the reusable 10-bit FFmpeg staging frame.

use ffmpeg_next::{format::Pixel, frame::Video};

pub(super) fn stage(input: &[u8], frame: &mut Video) {
    let row_bytes = frame.width() as usize * 2;
    let (y, uv) = input.split_at(row_bytes * frame.height() as usize);
    match frame.format() {
        Pixel::P010LE => {
            for (plane, source) in [(0, y), (1, uv)] {
                let stride = frame.stride(plane);
                let dst = frame.data_mut(plane);
                if stride == row_bytes {
                    dst[..source.len()].copy_from_slice(source);
                } else {
                    for (src, dst) in source
                        .chunks_exact(row_bytes)
                        .zip(dst.chunks_exact_mut(stride))
                    {
                        dst[..row_bytes].copy_from_slice(src);
                    }
                }
            }
        }
        Pixel::YUV420P10LE => {
            // Planar 10-bit samples occupy the low bits of each LE word.
            stage_planar(frame, 0, y, row_bytes, 0, 1);
            stage_planar(frame, 1, uv, row_bytes, 0, 2);
            stage_planar(frame, 2, uv, row_bytes, 1, 2);
        }
        _ => unreachable!("expected 10-bit staging format"),
    }
}

fn stage_planar(
    frame: &mut Video,
    plane: usize,
    source: &[u8],
    row_bytes: usize,
    start: usize,
    step: usize,
) {
    let stride = frame.stride(plane);
    for (src, dst) in source
        .chunks_exact(row_bytes)
        .zip(frame.data_mut(plane).chunks_exact_mut(stride))
    {
        let src = src.as_chunks::<2>().0;
        let dst = dst.as_chunks_mut::<2>().0;
        for (sample, out) in src.iter().skip(start).step_by(step).zip(dst) {
            *out = (u16::from_le_bytes(*sample) >> 6).to_le_bytes();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LUMA: &[u16] = &[0, 64, 512, 940, 1023, 1, 7, 3, 1022, 511, 65, 939];
    const UV: &[u16] = &[0, 1023, 64, 940, 512, 511];

    fn packed_input() -> Vec<u8> {
        LUMA.iter()
            .chain(UV)
            .flat_map(|sample| (sample << 6).to_le_bytes())
            .collect()
    }

    fn check_plane(frame: &Video, plane: usize, row_bytes: usize, expected: &[u8]) {
        let stride = frame.stride(plane);
        assert!(stride > row_bytes);
        for (actual, expected) in frame
            .data(plane)
            .chunks_exact(stride)
            .zip(expected.chunks_exact(row_bytes))
        {
            assert_eq!(&actual[..row_bytes], expected);
            assert!(actual[row_bytes..].iter().all(|&byte| byte == 0xab));
        }
    }

    #[test]
    fn stages_p010le_with_padded_strides() {
        let mut frame = Video::new(Pixel::P010LE, 6, 2);
        frame.data_mut(0).fill(0xab);
        frame.data_mut(1).fill(0xab);
        let input = packed_input();
        stage(&input, &mut frame);
        check_plane(&frame, 0, 12, &input[..24]);
        check_plane(&frame, 1, 12, &input[24..]);
    }

    #[test]
    fn stages_planar_10bit_with_lsb_alignment_and_separate_chroma() {
        let mut frame = Video::new(Pixel::YUV420P10LE, 6, 2);
        for plane in 0..3 {
            frame.data_mut(plane).fill(0xab);
        }
        stage(&packed_input(), &mut frame);
        let as_bytes = |samples: &[u16]| -> Vec<u8> {
            samples
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect()
        };
        check_plane(&frame, 0, 12, &as_bytes(LUMA));
        check_plane(&frame, 1, 6, &as_bytes(&[0, 64, 512]));
        check_plane(&frame, 2, 6, &as_bytes(&[1023, 940, 511]));
    }
}
