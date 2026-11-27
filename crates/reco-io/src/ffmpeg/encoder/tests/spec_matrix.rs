//! Table-driven matrix tests over the declarative `ALL_ENCODERS` catalog,
//! `QualityModel`, `StagingMode`, and `ContainerPolicy`.

use std::collections::{BTreeMap, HashSet};

use super::super::catalog::{
    ALL_ENCODERS, EncoderFamily, StagingMode, find_encoder_spec, specs_for_codec,
    vaapi_peer_encoder, validate_selection,
};
use super::super::container::{ContainerPolicy, HVC1_CODEC_TAG, OutputColor};
use super::super::options::{build_encoder_opts, videotoolbox_quality_params};
use super::super::{Codec, Container, EncoderConfig, Pixel, Quality};
use crate::ffmpeg::ColorMetadata;
use crate::output::QualityPreset;
use ffmpeg_next as ffmpeg;

fn opts_map(dict: &ffmpeg::Dictionary<'_>) -> BTreeMap<String, String> {
    dict.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn expected_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

#[test]
fn catalog_invariants_unique_names_codec_coverage_and_hw_before_sw() {
    let mut seen = HashSet::new();
    for spec in ALL_ENCODERS {
        assert!(
            seen.insert(spec.name),
            "duplicate encoder name in ALL_ENCODERS: {}",
            spec.name
        );
        assert_eq!(
            find_encoder_spec(spec.name),
            Some(spec),
            "find_encoder_spec mismatch for {}",
            spec.name
        );
    }
    assert_eq!(find_encoder_spec("nonexistent_encoder"), None);

    for codec in [Codec::H264, Codec::HEVC, Codec::AV1] {
        let specs: Vec<_> = specs_for_codec(codec).collect();
        assert!(
            !specs.is_empty(),
            "codec {codec:?} must have at least one encoder spec"
        );
        assert!(
            specs.iter().any(|s| s.is_hardware()),
            "codec {codec:?} must include hardware encoders"
        );
        assert!(
            specs.iter().any(|s| !s.is_hardware()),
            "codec {codec:?} must include software fallback encoders"
        );

        let mut saw_software = false;
        for spec in &specs {
            if !spec.is_hardware() {
                saw_software = true;
            } else {
                assert!(
                    !saw_software,
                    "hardware encoder {} appears after software encoder in {codec:?}",
                    spec.name
                );
            }
        }
    }

    // Preserve exact candidate preference order for each codec.
    assert_eq!(
        specs_for_codec(Codec::H264)
            .map(|s| s.name)
            .collect::<Vec<_>>(),
        vec![
            "h264_nvenc",
            "h264_qsv",
            "h264_videotoolbox",
            "h264_amf",
            "h264_vaapi",
            "h264_v4l2m2m",
            "h264_mediacodec",
            "h264_mf",
            "libx264",
        ]
    );
    assert_eq!(
        specs_for_codec(Codec::HEVC)
            .map(|s| s.name)
            .collect::<Vec<_>>(),
        vec![
            "hevc_nvenc",
            "hevc_qsv",
            "hevc_videotoolbox",
            "hevc_amf",
            "hevc_vaapi",
            "hevc_v4l2m2m",
            "hevc_mediacodec",
            "hevc_mf",
            "libx265",
        ]
    );
    assert_eq!(
        specs_for_codec(Codec::AV1)
            .map(|s| s.name)
            .collect::<Vec<_>>(),
        vec![
            "av1_nvenc",
            "av1_qsv",
            "av1_amf",
            "av1_vaapi",
            "libsvtav1",
            "libaom-av1",
        ]
    );

    // Linux QSV -> VAAPI peer mapping covers every QSV encoder in the catalog.
    for spec in ALL_ENCODERS {
        if spec.family == EncoderFamily::Qsv {
            let peer = vaapi_peer_encoder(spec.name)
                .unwrap_or_else(|| panic!("missing VAAPI peer for QSV encoder {}", spec.name));
            let peer_spec = find_encoder_spec(peer).expect("VAAPI peer must exist in ALL_ENCODERS");
            assert_eq!(peer_spec.family, EncoderFamily::Vaapi);
            assert_eq!(peer_spec.codec, spec.codec);
        } else {
            assert_eq!(vaapi_peer_encoder(spec.name), None);
        }
    }
}

#[test]
fn staging_mode_invariants_across_all_encoders() {
    for spec in ALL_ENCODERS {
        match spec.family {
            EncoderFamily::Vaapi => {
                assert_eq!(
                    spec.staging,
                    StagingMode::VaapiSurface,
                    "{} must use VaapiSurface staging",
                    spec.name
                );
                assert_eq!(spec.staging.encoder_pixel_format(), Pixel::VAAPI);
                assert_eq!(spec.staging.staging_pixel_format(), Pixel::NV12);
                assert!(spec.staging.uses_vaapi_upload());
            }
            EncoderFamily::Software => {
                assert_eq!(
                    spec.staging,
                    StagingMode::Yuv420p,
                    "{} must use Yuv420p staging",
                    spec.name
                );
                assert_eq!(spec.staging.encoder_pixel_format(), Pixel::YUV420P);
                assert_eq!(spec.staging.staging_pixel_format(), Pixel::YUV420P);
                assert!(!spec.staging.uses_vaapi_upload());
            }
            _ => {
                assert_eq!(
                    spec.staging,
                    StagingMode::Nv12,
                    "{} must use Nv12 staging",
                    spec.name
                );
                assert_eq!(spec.staging.encoder_pixel_format(), Pixel::NV12);
                assert_eq!(spec.staging.staging_pixel_format(), Pixel::NV12);
                assert!(!spec.staging.uses_vaapi_upload());
            }
        }
    }
}

#[test]
fn all_encoders_options_and_selection_matrix() {
    let presets = [
        QualityPreset::Fast,
        QualityPreset::Balanced,
        QualityPreset::High,
    ];
    let resolutions = [(1920u32, 1080u32), (3840u32, 2160u32)];
    let numeric_qualities = [0u8, 50, 75, 100];

    for spec in ALL_ENCODERS {
        // Every spec must accept preset qualities and produce well-formed options at 1080p and 4K.
        for preset in presets {
            let cfg = EncoderConfig {
                quality: Quality::Preset(preset),
                ..Default::default()
            };
            validate_selection(spec.name, &cfg).unwrap_or_else(|e| {
                panic!("{} rejected preset {preset:?}: {e}", spec.name);
            });

            for (w, h) in resolutions {
                let opts = build_encoder_opts(spec.name, Quality::Preset(preset), None, w, h);
                assert!(
                    opts.get("b:v").is_none(),
                    "{}: must not emit CLI option b:v",
                    spec.name
                );
                if spec.quality_model.supports_numeric_quality() {
                    assert!(
                        opts.iter().next().is_some(),
                        "{}: expected non-empty options for {preset:?} @ {w}x{h}",
                        spec.name
                    );
                } else {
                    assert!(
                        opts.iter().next().is_none(),
                        "{}: DriverDefault spec must emit no preset options",
                        spec.name
                    );
                }

                // Explicit preset override always sets `preset`.
                let overridden = build_encoder_opts(
                    spec.name,
                    Quality::Preset(preset),
                    Some("custom_preset"),
                    w,
                    h,
                );
                assert_eq!(overridden.get("preset"), Some("custom_preset"));
            }
        }

        // Numeric quality support matches `QualityModel::supports_numeric_quality()`.
        for q in numeric_qualities {
            let cfg = EncoderConfig {
                quality: Quality::Value(q),
                ..Default::default()
            };
            if spec.quality_model.supports_numeric_quality() {
                validate_selection(spec.name, &cfg).unwrap_or_else(|e| {
                    panic!("{} rejected numeric quality {q}: {e}", spec.name);
                });
                for (w, h) in resolutions {
                    let opts = build_encoder_opts(spec.name, Quality::Value(q), None, w, h);
                    assert!(
                        opts.iter().next().is_some(),
                        "{}: expected non-empty options for q={q} @ {w}x{h}",
                        spec.name
                    );
                    assert!(opts.get("b:v").is_none());
                }
            } else {
                assert!(
                    validate_selection(spec.name, &cfg).is_err(),
                    "{}: must reject Quality::Value({q})",
                    spec.name
                );
            }
        }

        // Device selection rules per family.
        let nvenc_dev = EncoderConfig {
            device: Some("0".into()),
            ..Default::default()
        };
        let vaapi_dev = EncoderConfig {
            device: Some("/dev/dri/renderD128".into()),
            ..Default::default()
        };
        let empty_dev = EncoderConfig {
            device: Some(String::new()),
            ..Default::default()
        };
        let nul_dev = EncoderConfig {
            device: Some("0\0".into()),
            ..Default::default()
        };

        assert!(validate_selection(spec.name, &empty_dev).is_err());
        assert!(validate_selection(spec.name, &nul_dev).is_err());

        match spec.family {
            EncoderFamily::Nvenc => {
                assert!(validate_selection(spec.name, &nvenc_dev).is_ok());
                assert!(validate_selection(spec.name, &vaapi_dev).is_err());
            }
            EncoderFamily::Vaapi => {
                assert!(validate_selection(spec.name, &vaapi_dev).is_ok());
                assert!(validate_selection(spec.name, &nvenc_dev).is_ok());
            }
            _ => {
                assert!(validate_selection(spec.name, &nvenc_dev).is_err());
                assert!(validate_selection(spec.name, &vaapi_dev).is_err());
            }
        }
    }
}

#[test]
fn custom_non_catalog_encoder_suffix_fallbacks_preserve_behavior() {
    // Suffix-based numeric quality and device validation for custom user-supplied names.
    for (name, key, expected_q50) in [
        ("vp9_nvenc", "cq", "26.00"),
        ("vp9_qsv", "global_quality", "26"),
        ("vp9_vaapi", "qp", "26"),
        ("vp9_amf", "qp_i", "26"),
        ("custom_videotoolbox", "global_quality", "60"),
    ] {
        assert!(find_encoder_spec(name).is_none());
        let cfg = EncoderConfig {
            quality: Quality::Value(50),
            ..Default::default()
        };
        validate_selection(name, &cfg).unwrap();

        let preset_opts = build_encoder_opts(
            name,
            Quality::Preset(QualityPreset::Balanced),
            None,
            1920,
            1080,
        );
        assert!(preset_opts.iter().next().is_none());

        let num_opts = build_encoder_opts(name, Quality::Value(50), None, 1920, 1080);
        assert_eq!(num_opts.get(key), Some(expected_q50), "{name}");
    }

    // Custom AMF sets both qp_i and qp_p.
    let amf = build_encoder_opts("vp9_amf", Quality::Value(50), None, 1920, 1080);
    assert_eq!(amf.get("qp_i"), Some("26"));
    assert_eq!(amf.get("qp_p"), Some("26"));

    // Custom VideoToolbox uses HEVC curve when name starts with `hevc_`.
    let (vt_q, vt_maxrate) = videotoolbox_quality_params(75, true);
    let custom_hevc_vt = build_encoder_opts(
        "hevc_custom_videotoolbox",
        Quality::Value(75),
        None,
        1920,
        1080,
    );
    assert_eq!(
        custom_hevc_vt.get("global_quality"),
        Some(vt_q.to_string().as_str())
    );
    assert_eq!(
        custom_hevc_vt.get("maxrate"),
        Some(format!("{vt_maxrate}M").as_str())
    );

    // Custom NVENC / VAAPI device selection works via suffix fallback.
    assert!(
        validate_selection(
            "vp9_nvenc",
            &EncoderConfig {
                device: Some("1".into()),
                ..Default::default()
            }
        )
        .is_ok()
    );
    assert!(
        validate_selection(
            "vp9_vaapi",
            &EncoderConfig {
                device: Some("/dev/dri/renderD129".into()),
                ..Default::default()
            }
        )
        .is_ok()
    );
}

#[test]
fn representative_encoder_options_exact_parity() {
    // 1. h264_nvenc: preset + profile + spatial/temporal AQ + decimal cq override + 4K scaling
    assert_eq!(
        opts_map(&build_encoder_opts(
            "h264_nvenc",
            Quality::Preset(QualityPreset::Balanced),
            None,
            1920,
            1080
        )),
        expected_map(&[
            ("preset", "p4"),
            ("tune", "hq"),
            ("rc", "vbr"),
            ("cq", "23"),
            ("b", "12M"),
            ("maxrate", "18M"),
            ("profile", "high"),
            ("spatial-aq", "1"),
            ("temporal-aq", "1"),
        ])
    );
    assert_eq!(
        opts_map(&build_encoder_opts(
            "h264_nvenc",
            Quality::Value(75),
            None,
            3840,
            2160
        )),
        expected_map(&[
            ("preset", "p5"),
            ("tune", "hq"),
            ("rc", "vbr"),
            ("cq", "19.00"),
            ("b", "80M"),
            ("maxrate", "120M"),
            ("profile", "high"),
            ("spatial-aq", "1"),
            ("temporal-aq", "1"),
        ])
    );

    // 2. hevc_nvenc: main profile + lower ceilings + AQ
    assert_eq!(
        opts_map(&build_encoder_opts(
            "hevc_nvenc",
            Quality::Preset(QualityPreset::High),
            None,
            1920,
            1080
        )),
        expected_map(&[
            ("preset", "p5"),
            ("tune", "hq"),
            ("rc", "vbr"),
            ("cq", "19"),
            ("b", "15M"),
            ("maxrate", "22M"),
            ("profile", "main"),
            ("spatial-aq", "1"),
            ("temporal-aq", "1"),
        ])
    );

    // 3. av1_nvenc: MUST NOT set profile, spatial-aq, or temporal-aq
    assert_eq!(
        opts_map(&build_encoder_opts(
            "av1_nvenc",
            Quality::Preset(QualityPreset::Fast),
            None,
            1920,
            1080
        )),
        expected_map(&[
            ("preset", "p3"),
            ("tune", "hq"),
            ("rc", "vbr"),
            ("cq", "32"),
            ("b", "5M"),
            ("maxrate", "8M"),
        ])
    );
    let av1_nvenc_num = build_encoder_opts("av1_nvenc", Quality::Value(50), None, 1920, 1080);
    assert_eq!(av1_nvenc_num.get("cq"), Some("26.00"));
    assert_eq!(av1_nvenc_num.get("profile"), None);
    assert_eq!(av1_nvenc_num.get("spatial-aq"), None);
    assert_eq!(av1_nvenc_num.get("temporal-aq"), None);

    // 4. h264_videotoolbox & hevc_videotoolbox
    assert_eq!(
        opts_map(&build_encoder_opts(
            "h264_videotoolbox",
            Quality::Preset(QualityPreset::High),
            None,
            1920,
            1080
        )),
        expected_map(&[
            ("global_quality", "70"),
            ("maxrate", "30M"),
            ("profile", "high"),
        ])
    );
    assert_eq!(
        opts_map(&build_encoder_opts(
            "hevc_videotoolbox",
            Quality::Preset(QualityPreset::High),
            None,
            1920,
            1080
        )),
        expected_map(&[
            ("global_quality", "70"),
            ("maxrate", "22M"),
            ("profile", "main"),
        ])
    );

    // 5. h264_qsv sets profile=high; hevc_qsv/av1_qsv omit profile; numeric is integer
    assert_eq!(
        opts_map(&build_encoder_opts(
            "h264_qsv",
            Quality::Preset(QualityPreset::Balanced),
            None,
            1920,
            1080
        )),
        expected_map(&[
            ("preset", "medium"),
            ("global_quality", "23"),
            ("profile", "high"),
        ])
    );
    assert_eq!(
        opts_map(&build_encoder_opts(
            "h264_qsv",
            Quality::Value(50),
            None,
            1920,
            1080
        )),
        expected_map(&[
            ("preset", "medium"),
            ("global_quality", "26"),
            ("profile", "high"),
        ])
    );
    for qsv_no_profile in ["hevc_qsv", "av1_qsv"] {
        let opts = build_encoder_opts(
            qsv_no_profile,
            Quality::Preset(QualityPreset::Fast),
            None,
            1920,
            1080,
        );
        assert_eq!(
            opts_map(&opts),
            expected_map(&[("preset", "medium"), ("global_quality", "28")])
        );
    }

    // 6. h264_vaapi sets profile=high; hevc_vaapi/av1_vaapi omit profile; numeric is integer
    assert_eq!(
        opts_map(&build_encoder_opts(
            "h264_vaapi",
            Quality::Preset(QualityPreset::High),
            None,
            1920,
            1080
        )),
        expected_map(&[("qp", "19"), ("profile", "high")])
    );
    assert_eq!(
        opts_map(&build_encoder_opts(
            "h264_vaapi",
            Quality::Value(50),
            None,
            1920,
            1080
        )),
        expected_map(&[("qp", "26"), ("profile", "high")])
    );
    for vaapi_no_profile in ["hevc_vaapi", "av1_vaapi"] {
        let opts = build_encoder_opts(
            vaapi_no_profile,
            Quality::Preset(QualityPreset::Balanced),
            None,
            1920,
            1080,
        );
        assert_eq!(opts_map(&opts), expected_map(&[("qp", "23")]));
    }

    // 7. h264_amf: cqp + balanced quality + integer qp_i/qp_p
    assert_eq!(
        opts_map(&build_encoder_opts(
            "h264_amf",
            Quality::Preset(QualityPreset::Balanced),
            None,
            1920,
            1080
        )),
        expected_map(&[
            ("quality", "balanced"),
            ("qp_i", "18"),
            ("qp_p", "18"),
            ("rc", "cqp"),
        ])
    );
    assert_eq!(
        opts_map(&build_encoder_opts(
            "h264_amf",
            Quality::Value(50),
            None,
            1920,
            1080
        )),
        expected_map(&[
            ("quality", "balanced"),
            ("qp_i", "26"),
            ("qp_p", "26"),
            ("rc", "cqp"),
        ])
    );

    // 8. h264_v4l2m2m: driver defaults (empty dict)
    assert!(
        opts_map(&build_encoder_opts(
            "h264_v4l2m2m",
            Quality::Preset(QualityPreset::High),
            None,
            1920,
            1080
        ))
        .is_empty()
    );

    // 9. libx264: decimal crf formatting + profile=high (ignoring optional Tegra `threads`)
    let mut x264_balanced = opts_map(&build_encoder_opts(
        "libx264",
        Quality::Preset(QualityPreset::Balanced),
        None,
        1920,
        1080,
    ));
    x264_balanced.remove("threads");
    assert_eq!(
        x264_balanced,
        expected_map(&[("preset", "veryfast"), ("crf", "28"), ("profile", "high")])
    );
    let mut x264_num = opts_map(&build_encoder_opts(
        "libx264",
        Quality::Value(50),
        None,
        1920,
        1080,
    ));
    x264_num.remove("threads");
    assert_eq!(
        x264_num,
        expected_map(&[
            ("preset", "veryfast"),
            ("crf", "26.00"),
            ("profile", "high"),
        ])
    );

    // 10. libx265: decimal crf formatting + profile=main
    assert_eq!(
        opts_map(&build_encoder_opts(
            "libx265",
            Quality::Preset(QualityPreset::High),
            None,
            1920,
            1080
        )),
        expected_map(&[("preset", "medium"), ("crf", "21"), ("profile", "main")])
    );
    assert_eq!(
        opts_map(&build_encoder_opts(
            "libx265",
            Quality::Value(25),
            None,
            1920,
            1080
        )),
        expected_map(&[
            ("preset", "veryfast"),
            ("crf", "33.00"),
            ("profile", "main"),
        ])
    );

    // 11. libsvtav1: integer crf formatting, no profile
    assert_eq!(
        opts_map(&build_encoder_opts(
            "libsvtav1",
            Quality::Preset(QualityPreset::Balanced),
            None,
            1920,
            1080
        )),
        expected_map(&[("preset", "7"), ("crf", "30")])
    );
    assert_eq!(
        opts_map(&build_encoder_opts(
            "libsvtav1",
            Quality::Value(50),
            None,
            1920,
            1080
        )),
        expected_map(&[("preset", "7"), ("crf", "26")])
    );

    // 12. libaom-av1: crf + cpu-used + row-mt=1, integer crf formatting, no preset
    assert_eq!(
        opts_map(&build_encoder_opts(
            "libaom-av1",
            Quality::Preset(QualityPreset::High),
            None,
            1920,
            1080
        )),
        expected_map(&[("crf", "25"), ("cpu-used", "4"), ("row-mt", "1")])
    );
    assert_eq!(
        opts_map(&build_encoder_opts(
            "libaom-av1",
            Quality::Value(50),
            None,
            1920,
            1080
        )),
        expected_map(&[("crf", "26"), ("cpu-used", "6"), ("row-mt", "1")])
    );
}

#[test]
fn container_policy_matrix_across_containers_codecs_and_color_modes() {
    let sdr_untagged = OutputColor::default();
    let sdr_rendered = OutputColor {
        metadata: ColorMetadata {
            primaries: None,
            transfer: None,
            space: Some(ffmpeg::color::Space::BT709),
        },
        full_range: Some(false),
        rendered_rgba_scaler: true,
    };
    let hdr_pq = OutputColor {
        metadata: ColorMetadata {
            primaries: Some(ffmpeg::color::Primaries::BT2020),
            transfer: Some(ffmpeg::color::TransferCharacteristic::SMPTE2084),
            space: Some(ffmpeg::color::Space::BT2020NCL),
        },
        full_range: Some(false),
        rendered_rgba_scaler: true,
    };

    assert!(!sdr_untagged.has_tags());
    assert!(sdr_rendered.has_tags());
    assert!(hdr_pq.has_tags());

    let containers = [
        (Container::Mp4, "mp4", true, false),
        (Container::Mp4Fragmented, "mp4", true, true),
        (Container::Mov, "mov", true, false),
        (Container::Mkv, "matroska", false, false),
        (Container::Flv, "flv", false, false),
    ];

    let color_cases = [
        ("sdr_untagged", sdr_untagged),
        ("sdr_rendered", sdr_rendered),
        ("hdr_pq", hdr_pq),
    ];

    for (container, expected_muxer, expected_isobmff, expected_fragmented) in containers {
        let policy = ContainerPolicy::for_container(container);
        assert_eq!(policy.muxer_name(), expected_muxer);
        assert_eq!(policy.is_isobmff(), expected_isobmff);
        assert_eq!(policy.is_fragmented(), expected_fragmented);

        for (color_label, color) in color_cases {
            let expected_movflags = match (container, color.has_tags()) {
                (Container::Mp4 | Container::Mov, false) => None,
                (Container::Mp4 | Container::Mov, true) => Some("write_colr"),
                (Container::Mp4Fragmented, false) => Some("empty_moov+frag_keyframe"),
                (Container::Mp4Fragmented, true) => Some("empty_moov+frag_keyframe+write_colr"),
                (Container::Mkv | Container::Flv, _) => None,
            };

            let actual_movflags = policy.movflags(color);
            assert_eq!(
                actual_movflags.as_deref(),
                expected_movflags,
                "{container:?} / {color_label}: unexpected movflags"
            );
            // Zero behavioral change: plain MP4 never injects `faststart`.
            if let Some(ref flags) = actual_movflags {
                assert!(
                    !flags.contains("faststart"),
                    "{container:?} / {color_label}: unexpected faststart flag"
                );
            }

            let dict = policy.muxer_options(color);
            assert_eq!(dict.get("movflags"), expected_movflags);
        }

        for codec in [Codec::H264, Codec::HEVC, Codec::AV1] {
            for has_global_header in [false, true] {
                let expected_tag = if codec == Codec::HEVC && has_global_header && expected_isobmff
                {
                    Some(HVC1_CODEC_TAG)
                } else {
                    None
                };
                assert_eq!(
                    policy.codec_tag(codec.id(), has_global_header),
                    expected_tag,
                    "{container:?} / {codec:?} / global_header={has_global_header}"
                );
            }
        }
    }
}
