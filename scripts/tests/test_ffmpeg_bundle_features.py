"""The packaged runtime must include more than RTSP's underlying TCP protocol."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_ffmpeg_bundle import verify_media_features


class MediaFeaturesTests(unittest.TestCase):
    def test_every_required_transport_codec_and_muxer_is_checked(self):
        inventory = {
            "protocols": "file tcp rtmp tls https rtmps",
            "demuxers": " D rtsp RTSP input\n D mov,mp4,m4a QuickTime input\n",
            "muxers": " E fifo bounded queue\n E flv FLV output\n E mp4 MP4 output\n",
            "decoders": " V..... h264 H264 input\n A..... aac AAC input\n",
            "encoders": " V..... libx264 H264 output\n A..... aac AAC output\n",
        }

        def run(binary, *args):
            self.assertEqual(binary, "ffmpeg")
            return inventory[args[-1].lstrip("-")].encode()

        verify_media_features(run, "ffmpeg")
        for category, names in (
            ("protocols", ["rtmp", "rtmps", "tls", "tcp"]),
            ("demuxers", ["rtsp", "mov"]),
            ("muxers", ["fifo", "flv", "mp4"]),
            ("decoders", ["h264", "aac"]),
            ("encoders", ["libx264", "aac"]),
        ):
            original = inventory[category]
            for name in names:
                with self.subTest(category=category, name=name):
                    inventory[category] = original.replace(name, "unavailable")
                    with self.assertRaisesRegex(ValueError, name):
                        verify_media_features(run, "ffmpeg")
            inventory[category] = original
