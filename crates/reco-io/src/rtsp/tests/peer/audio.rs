//! Encoded AAC fixture with independently phased RTP timestamps and common RTCP sender reports.

use super::*;
use ffmpeg_next::{self as ffmpeg, Rescale};

pub(super) struct Audio {
    pub(super) enabled: bool,
    config: String,
    packets: Vec<(i64, Vec<u8>)>,
    next: usize,
    sequence: u16,
    offset_us: i64,
}

impl Audio {
    pub(super) fn new(frames: usize, offset_us: i64) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audio.m4a");
        let result = std::process::Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=997:sample_rate=48000",
                "-t",
                &(frames as f64 / 10.0).to_string(),
                "-c:a",
                "aac",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let mut input = ffmpeg::format::input(&path).unwrap();
        let parameters = input
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .unwrap()
            .parameters();
        // SAFETY: parameters own live codec initialization bytes while copied into SDP text.
        let config = unsafe {
            let parameters = &*parameters.as_ptr();
            assert!((1..128).contains(&parameters.extradata_size));
            std::slice::from_raw_parts(parameters.extradata, parameters.extradata_size as usize)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        };
        let packets = input
            .packets()
            .filter_map(|(stream, packet)| {
                let time = packet
                    .pts()
                    .unwrap()
                    .rescale(stream.time_base(), (1, 1_000_000));
                // AAC priming precedes the test clock; retained packets identify the transmitted tone.
                (time >= 0).then(|| (time, packet.data().unwrap().to_vec()))
            })
            .collect();
        Self {
            enabled: false,
            config,
            packets,
            next: 0,
            sequence: 0,
            offset_us,
        }
    }

    pub(super) fn sdp(&self) -> String {
        format!(
            "m=audio 0 RTP/AVP 97\r\na=rtpmap:97 MPEG4-GENERIC/48000/1\r\na=fmtp:97 streamtype=5;profile-level-id=1;mode=AAC-hbr;config={};SizeLength=13;IndexLength=3;IndexDeltaLength=3\r\na=control:trackID=1\r\n",
            self.config
        )
    }

    pub(super) fn send_reports(&self, stream: &mut TcpStream) -> std::io::Result<()> {
        for (channel, ssrc, timestamp) in [(1, 1234u32, 90000u32), (3, 5678, 48000)] {
            let mut report = vec![0x80, 200, 0, 6];
            report.extend_from_slice(&ssrc.to_be_bytes());
            report.extend_from_slice(&3_910_000_000u32.to_be_bytes());
            report.extend_from_slice(&0u32.to_be_bytes());
            report.extend_from_slice(&timestamp.to_be_bytes());
            report.extend_from_slice(&[0; 8]);
            interleaved(stream, channel, &report)?;
        }
        Ok(())
    }

    pub(super) fn send_until(
        &mut self,
        stream: &mut TcpStream,
        until_us: i64,
    ) -> std::io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        while let Some((time, _)) = self.packets.get(self.next) {
            let time = time + self.offset_us;
            if time > until_us {
                break;
            }
            let timestamp = 48000 + time.rescale((1, 1_000_000), (1, 48000));
            // Exercise both single-AU and aggregated RFC 3640 packets. FFmpeg
            // must derive subsequent AU times from the shared packet clock.
            let units: Vec<_> = self.packets[self.next..]
                .iter()
                .take(2)
                .take_while(|(pts, _)| pts + self.offset_us <= until_us)
                .collect();
            let mut packet = vec![0x80, 97 | 128];
            packet.extend_from_slice(&self.sequence.to_be_bytes());
            packet.extend_from_slice(&u32::try_from(timestamp).unwrap().to_be_bytes());
            packet.extend_from_slice(&5678u32.to_be_bytes());
            packet.extend_from_slice(&(16 * units.len() as u16).to_be_bytes());
            for (_, data) in &units {
                packet.extend_from_slice(&(u16::try_from(data.len()).unwrap() << 3).to_be_bytes());
            }
            for (_, data) in &units {
                packet.extend_from_slice(data);
            }
            interleaved(stream, 2, &packet)?;
            self.next += units.len();
            self.sequence = self.sequence.wrapping_add(1);
        }
        Ok(())
    }
}
