//! Pure FFmpeg encoder option construction and AVOption validation.
//!
//! Translates [`Quality`] and optional preset overrides into an FFmpeg
//! [`ffmpeg::Dictionary`] by interpreting [`QualityModel`] from the declarative
//! catalog (with graceful FFmpeg-default + suffix fallback for custom
//! user-supplied encoder names).

extern crate ffmpeg_next as ffmpeg;

use super::EncodeError;
use super::catalog::{EncoderFamily, QualityModel, find_encoder_spec};
use crate::output::{BitDepth, Quality, QualityPreset};

/// Scale a 1080p-tuned bitrate ceiling (Mbps) by output pixel count, so
/// higher resolutions get a proportionally higher cap. Clamped to
/// `[0.5x, 8x]` of the 1080p baseline; always >= 1 Mbps.
pub(super) fn scale_bitrate_mbps(base_mbps: u32, width: u32, height: u32) -> u32 {
    let scale = (f64::from(width) * f64::from(height) / (1920.0 * 1080.0)).clamp(0.5, 8.0);
    ((f64::from(base_mbps) * scale).round() as u32).max(1)
}

/// Map normalized numeric quality (`0..=100`, higher = better) to VideoToolbox
/// `(global_quality, maxrate_1080p_mbps)`.
///
/// VideoToolbox `global_quality` maps to `kVTCompressionPropertyKey_Quality`
/// (`0.0..=1.0`, FFmpeg divides `global_quality` by 100). Because VT's bitrate
/// grows steeply near `1.0` (near-lossless), `0..=100` maps linearly onto
/// `30..=90`: `q = 50` matches `Balanced` (`60`), `q = 75` reaches `75`, and
/// `q > 75` continues climbing to `90` without hitting the `1.0` cliff.
///
/// The 1080p `maxrate` ceiling (scaled by output resolution via
/// [`scale_bitrate_mbps`]) is piecewise-linear through the codec's `Balanced`
/// (`q = 50`) and `High` (`q = 75`) ceilings and expands smoothly up to `2x`
/// the `High` ceiling at `q = 100`.
pub(super) fn videotoolbox_quality_params(q: u8, is_hevc: bool) -> (u32, u32) {
    let q = f64::from(q.min(100));
    let global_quality = (30.0 + (q / 100.0) * 60.0).round() as u32;

    let (min_mbps, balanced_mbps, high_mbps, max_mbps) = if is_hevc {
        (6.0, 14.0, 22.0, 44.0)
    } else {
        (8.0, 18.0, 30.0, 60.0)
    };
    let maxrate_mbps = if q <= 50.0 {
        min_mbps + (q / 50.0) * (balanced_mbps - min_mbps)
    } else if q <= 75.0 {
        balanced_mbps + ((q - 50.0) / 25.0) * (high_mbps - balanced_mbps)
    } else {
        high_mbps + ((q - 75.0) / 25.0) * (max_mbps - high_mbps)
    };

    (global_quality, maxrate_mbps.round() as u32)
}

fn numeric_quality_crf(q: u8) -> f64 {
    40.0 - (f64::from(q) / 100.0) * 28.0
}

fn format_crf_decimal(q: u8) -> String {
    format!("{:.2}", numeric_quality_crf(q))
}

fn format_crf_integer(q: u8) -> String {
    format!("{:.0}", numeric_quality_crf(q))
}

fn is_tegra_platform() -> bool {
    std::path::Path::new("/etc/nv_tegra_release").exists()
        || std::fs::read_to_string("/proc/device-tree/compatible")
            .unwrap_or_default()
            .contains("nvidia,tegra")
}

fn apply_videotoolbox_numeric_quality(
    opts: &mut ffmpeg::Dictionary<'static>,
    name: &str,
    q: u8,
    is_hevc: bool,
    mbps: &impl Fn(u32) -> String,
) {
    let (vt_q, maxrate_base) = videotoolbox_quality_params(q, is_hevc);
    let val = vt_q.to_string();
    let maxrate = mbps(maxrate_base);
    opts.set("global_quality", &val);
    opts.set("maxrate", &maxrate);
    log::info!(
        "Encoder quality: {q} -> {name} global_quality={val} (VT {:.2}), maxrate={maxrate}",
        f64::from(vt_q) / 100.0
    );
}

fn apply_quality_model(
    opts: &mut ffmpeg::Dictionary<'static>,
    name: &str,
    family: EncoderFamily,
    model: QualityModel,
    effective_preset: QualityPreset,
    quality_override: Option<u8>,
    mbps: &impl Fn(u32) -> String,
) {
    match model {
        QualityModel::NvencVbr {
            fast,
            balanced,
            high,
            profile,
            adaptive_quantization,
        } => {
            // NVENC VBR with per-quality bitrate ceiling. Prior defaults
            // (10M / 15M across all quality presets) artificially capped
            // High output at ~15 Mbps even with cq=19. Scale the ceilings
            // with quality so "high" actually delivers the visual bump.
            let tier = match effective_preset {
                QualityPreset::Fast => fast,
                QualityPreset::Balanced => balanced,
                QualityPreset::High => high,
            };
            opts.set("preset", tier.preset);
            opts.set("tune", "hq");
            opts.set("rc", "vbr");
            opts.set("cq", tier.cq);
            opts.set("b", &mbps(tier.bitrate_mbps));
            opts.set("maxrate", &mbps(tier.maxrate_mbps));
            if let Some(profile) = profile {
                opts.set("profile", profile);
            }
            if adaptive_quantization {
                opts.set("spatial-aq", "1");
                opts.set("temporal-aq", "1");
            }
            if let Some(q) = quality_override {
                opts.set("cq", &format_crf_decimal(q));
            }
        }
        QualityModel::Qsv { profile } => {
            let gq = match effective_preset {
                QualityPreset::Fast => "28",
                QualityPreset::Balanced => "23",
                QualityPreset::High => "19",
            };
            opts.set("preset", "medium");
            opts.set("global_quality", gq);
            if let Some(profile) = profile {
                opts.set("profile", profile);
            }
            if let Some(q) = quality_override {
                opts.set("global_quality", &format_crf_integer(q));
            }
        }
        QualityModel::VideoToolbox {
            fast_maxrate_mbps,
            balanced_maxrate_mbps,
            high_maxrate_mbps,
            profile,
            is_hevc,
        } => {
            // VideoToolbox quality maps to kVTCompressionPropertyKey_Quality
            // (0.0-1.0 float, FFmpeg divides global_quality by 100). Pair
            // global_quality with a resolution-scaled maxrate ceiling via
            // kVTCompressionPropertyKey_DataRateLimits matching NVENC.
            let (q_str, maxrate) = match effective_preset {
                QualityPreset::Fast => ("50", fast_maxrate_mbps),
                QualityPreset::Balanced => ("60", balanced_maxrate_mbps),
                QualityPreset::High => ("70", high_maxrate_mbps),
            };
            opts.set("global_quality", q_str);
            opts.set("maxrate", &mbps(maxrate));
            opts.set("profile", profile);
            if let Some(q) = quality_override {
                apply_videotoolbox_numeric_quality(opts, name, q, is_hevc, mbps);
            }
        }
        QualityModel::VaapiQp { profile } => {
            let qp = match effective_preset {
                QualityPreset::Fast => "28",
                QualityPreset::Balanced => "23",
                QualityPreset::High => "19",
            };
            opts.set("qp", qp);
            if let Some(profile) = profile {
                opts.set("profile", profile);
            }
            if let Some(q) = quality_override {
                opts.set("qp", &format_crf_integer(q));
            }
        }
        QualityModel::AmfCqp => {
            let q_str = match effective_preset {
                QualityPreset::Fast => "22",
                QualityPreset::Balanced => "18",
                QualityPreset::High => "14",
            };
            opts.set("quality", "balanced");
            opts.set("qp_i", q_str);
            opts.set("qp_p", q_str);
            opts.set("rc", "cqp");
            if let Some(q) = quality_override {
                let integer = format_crf_integer(q);
                opts.set("qp_i", &integer);
                opts.set("qp_p", &integer);
            }
        }
        QualityModel::SoftwareCrf {
            fast,
            balanced,
            high,
            profile,
            integer_crf,
            tegra_thread_cap,
        } => {
            let tier = match effective_preset {
                QualityPreset::Fast => fast,
                QualityPreset::Balanced => balanced,
                QualityPreset::High => high,
            };
            opts.set("preset", tier.preset);
            opts.set("crf", tier.crf);
            if let Some(profile) = profile {
                opts.set("profile", profile);
            }
            if tegra_thread_cap && is_tegra_platform() {
                // On Tegra (Jetson), limit threads to avoid starving other
                // pipeline stages (capture + pairing threads need CPU too).
                // x264 defaults to num_cpus which over-subscribes.
                let threads = std::env::var("RECO_X264_THREADS")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(4);
                log::info!("Tegra platform detected, limiting libx264 to {threads} threads");
                opts.set("threads", &threads.to_string());
            }
            if let Some(q) = quality_override {
                let value = if integer_crf {
                    format_crf_integer(q)
                } else {
                    format_crf_decimal(q)
                };
                opts.set("crf", &value);
            }
        }
        QualityModel::SoftwareAomCrf {
            fast,
            balanced,
            high,
        } => {
            let tier = match effective_preset {
                QualityPreset::Fast => fast,
                QualityPreset::Balanced => balanced,
                QualityPreset::High => high,
            };
            opts.set("crf", tier.crf);
            opts.set("cpu-used", tier.cpu_used);
            opts.set("row-mt", "1");
            if let Some(q) = quality_override {
                opts.set("crf", &format_crf_integer(q));
            }
        }
        QualityModel::DriverDefault => {
            if family == EncoderFamily::V4l2M2m {
                // The wrapper exposes no portable constant-quality control.
                log::info!("Encoder '{name}' uses driver quality defaults");
            } else {
                log::info!(
                    "Encoder '{name}' has no quality presets configured, using FFmpeg defaults"
                );
            }
        }
    }
}

fn apply_fallback_numeric_quality(
    opts: &mut ffmpeg::Dictionary<'static>,
    name: &str,
    q: u8,
    mbps: &impl Fn(u32) -> String,
) {
    match EncoderFamily::from_suffix(name) {
        Some(EncoderFamily::VideoToolbox) => {
            apply_videotoolbox_numeric_quality(opts, name, q, name.starts_with("hevc_"), mbps);
        }
        Some(EncoderFamily::Nvenc) => opts.set("cq", &format_crf_decimal(q)),
        Some(EncoderFamily::Qsv) => opts.set("global_quality", &format_crf_integer(q)),
        Some(EncoderFamily::Vaapi) => opts.set("qp", &format_crf_integer(q)),
        Some(EncoderFamily::Amf) => {
            let integer = format_crf_integer(q);
            opts.set("qp_i", &integer);
            opts.set("qp_p", &integer);
        }
        _ => {}
    }
}

/// Build encoder-specific FFmpeg options.
pub(super) fn build_encoder_opts(
    name: &str,
    quality: Quality,
    preset_override: Option<&str>,
    width: u32,
    height: u32,
    bit_depth: BitDepth,
) -> ffmpeg::Dictionary<'static> {
    let mut opts = ffmpeg::Dictionary::new();

    // Bitrate ceilings below are tuned for 1080p. Scale them with pixel
    // count so higher resolutions (2K/4K) aren't capped far below the
    // quality the CQ setting actually wants. Returns an "NM" string.
    let mbps = |base: u32| format!("{}M", scale_bitrate_mbps(base, width, height));

    // When a quality override is set, derive the effective preset from
    // the quality value so NVENC preset/maxrate scale with the CQ.
    let quality_override = match quality {
        Quality::Value(value) => Some(value),
        Quality::Preset(_) => None,
    };
    let effective_preset = match quality {
        Quality::Value(75..) => QualityPreset::High,
        Quality::Value(40..) => QualityPreset::Balanced,
        Quality::Value(_) => QualityPreset::Fast,
        Quality::Preset(preset) => preset,
    };

    if let Some(spec) = find_encoder_spec(name) {
        apply_quality_model(
            &mut opts,
            name,
            spec.family,
            spec.quality_model,
            effective_preset,
            quality_override,
            &mbps,
        );
    } else {
        log::info!("Encoder '{name}' has no quality presets configured, using FFmpeg defaults");
        if let Some(q) = quality_override {
            apply_fallback_numeric_quality(&mut opts, name, q, &mbps);
        }
    }

    if bit_depth == BitDepth::Ten {
        opts.set("profile", "main10");
    }

    // Apply preset override.
    if let Some(preset) = preset_override {
        opts.set("preset", preset);
    }

    opts
}

/// Reject option keys not recognized by the encoder or its private AVOptions.
pub(super) fn validate_encoder_options(
    encoder: &ffmpeg::encoder::video::Video,
    options: &ffmpeg::Dictionary<'_>,
) -> Result<(), EncodeError> {
    for (key, _) in options.iter() {
        let key_c = std::ffi::CString::new(key)
            .map_err(|_| EncodeError::InvalidConfig("encoder option contains a NUL byte".into()))?;
        // SAFETY: the encoder and its private AVOptions context are alive; this only looks up an option.
        let option = unsafe {
            ffmpeg::sys::av_opt_find(
                encoder.as_ptr().cast_mut().cast(),
                key_c.as_ptr(),
                std::ptr::null(),
                0,
                ffmpeg::sys::AV_OPT_SEARCH_CHILDREN,
            )
        };
        if option.is_null() {
            return Err(EncodeError::InvalidConfig(format!(
                "encoder does not support option {key}"
            )));
        }
    }
    Ok(())
}
