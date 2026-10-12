//! Bounded encoded audio from the same demux connection and media clock as live video.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use ffmpeg_next::{self as ffmpeg, codec};

const MAX_BYTES: usize = 4 * 1024 * 1024;
const MAX_PACKETS: usize = 4096;
pub(crate) const TIME_BASE: ffmpeg::Rational = ffmpeg::Rational(1, 1_000_000);

#[derive(Default)]
struct Queue {
    packets: VecDeque<ffmpeg::Packet>,
    bytes: usize,
    origin_us: Option<i64>,
    previous_us: Option<i64>,
}

/// Private bridge between the live demux worker, source clock and encoded output.
#[derive(Clone)]
pub(crate) struct LiveAudio {
    pub(crate) parameters: codec::Parameters,
    pub(crate) stream_index: usize,
    input_time_base: ffmpeg::Rational,
    offset_us: i64,
    queue: Arc<Mutex<Queue>>,
}

impl LiveAudio {
    pub(crate) fn from_input(
        input: &ffmpeg::format::context::Input,
        offset_us: i64,
    ) -> Option<Self> {
        let stream = input.streams().best(ffmpeg::media::Type::Audio)?;
        Some(Self {
            parameters: stream.parameters().clone(),
            stream_index: stream.index(),
            input_time_base: stream.time_base(),
            offset_us,
            queue: Arc::new(Mutex::new(Queue::default())),
        })
    }

    pub(crate) fn push(&self, mut packet: ffmpeg::Packet) -> Result<(), String> {
        packet.rescale_ts(self.input_time_base, TIME_BASE);
        let time = packet
            .dts()
            .or_else(|| packet.pts())
            .ok_or("live audio packet has no timestamp")?;
        let time = time
            .checked_add(self.offset_us)
            .ok_or("live audio timestamp overflow")?;
        let pts = match packet.pts() {
            Some(pts) => pts
                .checked_add(self.offset_us)
                .ok_or("live audio timestamp overflow")?,
            None => time,
        };
        packet.set_pts(Some(pts));
        packet.set_dts(Some(time));
        let mut queue = self.queue.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(previous) = queue.previous_us
            && time <= previous
        {
            return Err(format!(
                "live audio clock reset or stopped: {previous} -> {time} us"
            ));
        }
        queue.previous_us = Some(time);
        if queue.origin_us.is_some_and(|origin| time < origin) {
            return Ok(());
        }
        if packet.size() > MAX_BYTES {
            return Err("live audio packet exceeds the bounded buffer".into());
        }
        while queue.bytes + packet.size() > MAX_BYTES || queue.packets.len() == MAX_PACKETS {
            if queue.origin_us.is_some() {
                return Err("live audio output fell behind its bounded buffer".into());
            }
            let old = queue
                .packets
                .pop_front()
                .expect("nonempty full audio queue");
            queue.bytes -= old.size();
        }
        queue.bytes += packet.size();
        queue.packets.push_back(packet);
        Ok(())
    }

    pub(crate) fn set_origin(&self, origin_us: i64) {
        let mut queue = self.queue.lock().unwrap_or_else(|error| error.into_inner());
        if queue.origin_us.is_none() {
            queue.origin_us = Some(origin_us);
            while queue
                .packets
                .front()
                .is_some_and(|packet| packet.dts().unwrap() < origin_us)
            {
                let old = queue.packets.pop_front().unwrap();
                queue.bytes -= old.size();
            }
        }
    }

    /// Pop one packet on the output timeline, never waiting on a decoder worker.
    pub(crate) fn next_until(&self, time_us: i64) -> Result<Option<ffmpeg::Packet>, String> {
        let mut queue = self.queue.lock().unwrap_or_else(|error| error.into_inner());
        let Some((origin, first)) = queue.origin_us.zip(queue.packets.front()) else {
            return Ok(None);
        };
        if i128::from(first.dts().unwrap()) - i128::from(origin) > i128::from(time_us) {
            return Ok(None);
        }
        let pts = first
            .pts()
            .unwrap()
            .checked_sub(origin)
            .ok_or("live audio timestamp overflow")?;
        let dts = first
            .dts()
            .unwrap()
            .checked_sub(origin)
            .ok_or("live audio timestamp overflow")?;
        let mut packet = queue.packets.pop_front().unwrap();
        queue.bytes -= packet.size();
        packet.set_pts(Some(pts));
        packet.set_dts(Some(dts));
        Ok(Some(packet))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn capture(offset_us: i64) -> LiveAudio {
        LiveAudio {
            parameters: codec::Parameters::new(),
            stream_index: 1,
            input_time_base: TIME_BASE,
            offset_us,
            queue: Arc::new(Mutex::new(Queue::default())),
        }
    }

    fn packet(time: i64, size: usize) -> ffmpeg::Packet {
        let mut packet = ffmpeg::Packet::new(size);
        packet.set_pts(Some(time));
        packet.set_dts(Some(time));
        packet.set_duration(20_000);
        packet
    }

    #[test]
    fn audio_uses_video_origin_and_preserves_phase_gaps_and_duration() {
        let audio = capture(-7_000_000);
        for time in [7_100_000, 7_125_000, 7_145_000, 7_205_000] {
            audio.push(packet(time, 8)).unwrap();
        }
        assert!(audio.next_until(i64::MAX).unwrap().is_none());
        audio.set_origin(110_000);
        audio.set_origin(500_000); // Only the first selected slot anchors output.
        assert!(audio.next_until(14_999).unwrap().is_none());
        for time in [15_000, 35_000, 95_000] {
            let packet = audio.next_until(time).unwrap().unwrap();
            assert_eq!(packet.pts(), Some(time));
            assert_eq!(packet.dts(), Some(time));
            assert_eq!(packet.duration(), 20_000);
        }
        assert!(audio.next_until(i64::MAX).unwrap().is_none());
        assert_eq!(audio.queue.lock().unwrap().bytes, 0);
    }

    #[test]
    fn startup_is_rolling_but_active_overflow_is_an_error() {
        for packet_size in [1, MAX_BYTES / 2] {
            let audio = capture(0);
            let count = MAX_PACKETS.min(MAX_BYTES / packet_size);
            for index in 0..count + 2 {
                audio.push(packet(index as i64, packet_size)).unwrap();
            }
            let queue = audio.queue.lock().unwrap();
            assert_eq!(queue.packets.len(), count);
            assert_eq!(queue.bytes, count * packet_size);
            assert_eq!(queue.packets.front().unwrap().dts(), Some(2));
            drop(queue);
            audio.set_origin(2);
            assert!(
                audio
                    .push(packet(count as i64 + 2, packet_size))
                    .unwrap_err()
                    .contains("fell behind")
            );
        }
    }

    #[test]
    fn reset_missing_timestamp_and_oversized_packet_are_rejected() {
        let audio = capture(0);
        assert!(
            audio
                .push(ffmpeg::Packet::new(1))
                .unwrap_err()
                .contains("no timestamp")
        );
        audio.push(packet(20, 1)).unwrap();
        audio.set_origin(30);
        assert!(
            audio
                .push(packet(10, 1))
                .unwrap_err()
                .contains("clock reset")
        );
        assert!(
            audio
                .push(packet(40, MAX_BYTES + 1))
                .unwrap_err()
                .contains("exceeds")
        );
    }

    #[test]
    fn source_and_output_clock_overflows_are_errors() {
        assert!(
            capture(i64::MAX)
                .push(packet(10, 1))
                .unwrap_err()
                .contains("overflow")
        );
        let audio = capture(0);
        audio.set_origin(i64::MIN + 1);
        audio.push(packet(10, 1)).unwrap();
        // A far-future packet is not eligible, so no overflowing subtraction occurs.
        assert!(audio.next_until(i64::MAX).unwrap().is_none());
        let mut unusual = packet(11, 1);
        unusual.set_pts(Some(i64::MAX));
        let other = capture(0);
        other.set_origin(-1);
        other.push(unusual).unwrap();
        assert!(matches!(other.next_until(20), Err(error) if error.contains("overflow")));
    }
}
