//! Extract one frame from left+right videos, GPU-undistort, run AKAZE
//! with overlap region filtering, match features, save side-by-side PNG.
//!
//! Usage: cargo run -p reco-calibrate --example dump_undistorted -- \
//!   <left.mp4> <right.mp4> <match.json> <output_dir> [sync_offset] [frame_num]

use reco_calibrate::features::{self, DetectRegion};
use reco_core::calibration::MatchCalibration;
use reco_core::gpu::GpuContext;
use reco_core::undistort::GpuUndistort;

fn main() {
    reco_io::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "Usage: {} <left.mp4> <right.mp4> <match.json> <output_dir> [sync_offset] [frame_num]",
            args[0]
        );
        std::process::exit(1);
    }

    let sync_offset: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(85);
    let target_frame: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(300);

    let json_str = std::fs::read_to_string(&args[3]).unwrap();
    let cal: MatchCalibration = serde_json::from_str(&json_str).expect("invalid match.json");

    // Decode target frame from each video
    let mut left_dec =
        reco_io::ffmpeg::decoder::VideoDecoder::open(std::path::Path::new(&args[1])).unwrap();
    let mut right_dec =
        reco_io::ffmpeg::decoder::VideoDecoder::open(std::path::Path::new(&args[2])).unwrap();

    // Skip to target frame on left, target+offset on right
    for _ in 0..target_frame {
        left_dec.next_frame().unwrap();
    }
    for _ in 0..(target_frame + sync_offset) {
        right_dec.next_frame().unwrap();
    }
    println!(
        "Left frame {target_frame}, Right frame {} (sync_offset={sync_offset})",
        target_frame + sync_offset
    );

    let left_yuv = left_dec.next_frame().unwrap().expect("no left frames");
    let right_yuv = right_dec.next_frame().unwrap().expect("no right frames");
    let (lw, lh) = (left_yuv.width, left_yuv.height);
    let (rw, rh) = (right_yuv.width, right_yuv.height);
    println!("Left: {}x{}, Right: {}x{}", lw, lh, rw, rh);

    // GPU undistort both frames
    let gpu = pollster::block_on(GpuContext::new()).expect("no GPU");
    let left_undistort = GpuUndistort::new(&gpu, lw, lh);
    let right_undistort = GpuUndistort::new(&gpu, rw, rh);
    let left_rgba =
        left_undistort.undistort(&gpu, &left_yuv.y, &left_yuv.u, &left_yuv.v, &cal.left);
    let right_rgba =
        right_undistort.undistort(&gpu, &right_yuv.y, &right_yuv.u, &right_yuv.v, &cal.right);
    println!("GPU undistort done");

    // AKAZE detect with overlap region filtering (same as calibration pipeline)
    let left_region = DetectRegion {
        x_min: 0.4,
        x_max: 1.0,
        y_min: 0.3,
        y_max: 1.0,
    };
    let right_region = DetectRegion {
        x_min: 0.0,
        x_max: 0.6,
        y_min: 0.3,
        y_max: 1.0,
    };

    let (kp_left, desc_left) = features::detect(&left_rgba, lw, lh, Some(left_region), 2000);
    let (kp_right, desc_right) = features::detect(&right_rgba, rw, rh, Some(right_region), 2000);
    println!(
        "AKAZE: {} left keypoints (overlap zone), {} right keypoints (overlap zone)",
        kp_left.len(),
        kp_right.len()
    );

    // Match with Lowe's ratio test + cross-check
    let matches = features::match_descriptors(&desc_left, &desc_right, 0.7);
    println!("Matches: {} (ratio=0.7, cross-checked)", matches.len());

    // Generate colors for each match
    let match_colors: Vec<image::Rgba<u8>> = matches
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let hue = (i as f32 / matches.len().max(1) as f32) * 360.0;
            let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
            image::Rgba([r, g, b, 255])
        })
        .collect();

    let mut left_img = image::RgbaImage::from_raw(lw, lh, left_rgba).unwrap();
    let mut right_img = image::RgbaImage::from_raw(rw, rh, right_rgba).unwrap();

    for (i, m) in matches.iter().enumerate() {
        let color = match_colors[i];
        let lp = &kp_left[m.left_idx];
        let rp = &kp_right[m.right_idx];
        draw_circle(&mut left_img, lp.x as i32, lp.y as i32, 8, color);
        draw_circle(&mut right_img, rp.x as i32, rp.y as i32, 8, color);
    }

    // Side-by-side with match lines
    let total_w = lw + rw;
    let total_h = lh.max(rh);
    let mut sbs = image::RgbaImage::new(total_w, total_h);
    for y in 0..lh {
        for x in 0..lw {
            sbs.put_pixel(x, y, *left_img.get_pixel(x, y));
        }
    }
    for y in 0..rh {
        for x in 0..rw {
            sbs.put_pixel(lw + x, y, *right_img.get_pixel(x, y));
        }
    }
    for (i, m) in matches.iter().enumerate() {
        let color = match_colors[i];
        let lp = &kp_left[m.left_idx];
        let rp = &kp_right[m.right_idx];
        draw_line(
            &mut sbs,
            lp.x as i32,
            lp.y as i32,
            lw as i32 + rp.x as i32,
            rp.y as i32,
            color,
        );
    }

    let out_dir = &args[4];
    std::fs::create_dir_all(out_dir).unwrap();
    sbs.save(format!("{out_dir}/matches_side_by_side.png"))
        .unwrap();
    println!("Saved to {out_dir}/matches_side_by_side.png");
}

fn draw_circle(img: &mut image::RgbaImage, cx: i32, cy: i32, r: i32, color: image::Rgba<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    img.put_pixel(px as u32, py as u32, color);
                }
            }
        }
    }
}

fn draw_line(
    img: &mut image::RgbaImage,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: image::Rgba<u8>,
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    let (w, h) = (img.width() as i32, img.height() as i32);
    loop {
        if x >= 0 && x < w && y >= 0 && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 {
        0..60 => (c, x, 0.0),
        60..120 => (x, c, 0.0),
        120..180 => (0.0, c, x),
        180..240 => (0.0, x, c),
        240..300 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}
