"""Live streaming control endpoints.

Spawns FFmpeg to ingest UDP MPEG-TS and output HLS segments/playlist.
"""

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urljoin, quote
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from fastapi.responses import Response, PlainTextResponse

from app.data_paths import VIDEOS_DIR, get_ffmpeg_path
from app.utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/live", tags=["live"])

FFMPEG = get_ffmpeg_path()
LIVE_HLS_URL = os.environ.get("LIVE_HLS_URL", "http://myrpi5:8080/live/index.m3u8")

LIVE_DIR = VIDEOS_DIR / "live"
PLAYLIST_PATH = LIVE_DIR / "index.m3u8"
SEGMENT_TEMPLATE = LIVE_DIR / "seg_%05d.ts"

_live_process = None
_live_lock = threading.Lock()
_last_error = None


def _is_running() -> bool:
    global _live_process
    if _live_process is None:
        return False
    if _live_process.poll() is None:
        return True
    _live_process = None
    return False


def _clear_live_dir() -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    for child in LIVE_DIR.iterdir():
        if child.is_dir():
            for nested in child.iterdir():
                if nested.is_file() or nested.is_symlink():
                    nested.unlink(missing_ok=True)
                else:
                    _remove_dir(nested)
            child.rmdir()
        else:
            child.unlink(missing_ok=True)


def _remove_dir(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _remove_dir(child)
        else:
            child.unlink(missing_ok=True)
    path.rmdir()


def _build_ffmpeg_command() -> list[str]:
    return [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-i",
        "udp://0.0.0.0:5000?fifo_size=5000000&overrun_nonfatal=1",
        "-c",
        "copy",
        "-f",
        "hls",
        "-hls_time",
        "0.5",
        "-hls_list_size",
        "10",
        "-hls_flags",
        "delete_segments+append_list+omit_endlist+program_date_time",
        "-hls_segment_type",
        "mpegts",
        "-hls_segment_filename",
        str(SEGMENT_TEMPLATE.resolve()),
        str(PLAYLIST_PATH.resolve()),
    ]


def _fetch_url(url: str, fallback_urls: Optional[list[str]] = None) -> bytes:
    urls = [url]
    if fallback_urls:
        urls.extend(fallback_urls)

    last_exc = None
    for candidate in urls:
        try:
            req = Request(candidate, headers={"User-Agent": "VideoStitcher/1.0"})
            with urlopen(req, timeout=10) as response:
                return response.read()
        except HTTPError as exc:
            last_exc = exc
            if exc.code == 404:
                continue
            break
        except Exception as exc:
            last_exc = exc
            break

    raise HTTPException(status_code=502, detail=f"Failed to fetch live URL: {last_exc}")


def _rewrite_playlist(playlist_text: str, base_url: str) -> str:
    lines = []
    for raw_line in playlist_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            lines.append(raw_line)
            continue

        absolute = urljoin(base_url, line)
        if absolute.lower().endswith(".m3u8"):
            proxied = f"/live/playlist?url={quote(absolute, safe='')}"
        else:
            proxied = f"/live/segment?url={quote(absolute, safe='')}"
        lines.append(proxied)

    return "\n".join(lines) + "\n"


def _build_playlist_response(playlist_url: str) -> PlainTextResponse:
    playlist_bytes = _fetch_url(playlist_url)
    try:
        playlist_text = playlist_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=502, detail="Invalid playlist encoding")

    base_url = playlist_url.rsplit("/", 1)[0] + "/"
    rewritten = _rewrite_playlist(playlist_text, base_url)
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }
    return PlainTextResponse(rewritten, media_type="application/vnd.apple.mpegurl", headers=headers)


@router.post("/start")
def start_live_stream():
    global _live_process, _last_error

    with _live_lock:
        if _is_running():
            return {"running": True, "message": "Live stream already running", "pid": _live_process.pid}

        try:
            _clear_live_dir()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to clear live directory: {exc}")

        command = _build_ffmpeg_command()
        logger.info("Starting live FFmpeg: %s", " ".join(command))

        try:
            _live_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            _last_error = None
        except Exception as exc:
            _live_process = None
            _last_error = str(exc)
            raise HTTPException(status_code=500, detail=f"Failed to start live stream: {exc}")

        return {"running": True, "message": "Live stream started", "pid": _live_process.pid}


@router.post("/stop")
def stop_live_stream():
    global _live_process

    with _live_lock:
        if not _is_running():
            return {"running": False, "message": "Live stream not running"}

        process = _live_process

        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)
        except Exception as exc:
            logger.warning("Failed to stop live FFmpeg: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to stop live stream: {exc}")
        finally:
            _live_process = None

        return {"running": False, "message": "Live stream stopped"}


@router.get("/status")
def live_status():
    running = _is_running()
    pid = _live_process.pid if running and _live_process else None

    return {
        "running": running,
        "pid": pid,
        "playlist": str(PLAYLIST_PATH) if PLAYLIST_PATH.exists() else None,
        "last_error": _last_error,
    }


@router.get("/playlist")
def live_playlist(url: Optional[str] = Query(default=None)):
    playlist_url = url or LIVE_HLS_URL
    return _build_playlist_response(playlist_url)


@router.get("/playlist.m3u8")
def live_playlist_m3u8():
    return _build_playlist_response(LIVE_HLS_URL)


@router.get("/segment")
def live_segment(url: str):
    fallbacks = []
    if "/live/" in url:
        fallbacks.append(url.replace("/live/", "/"))
    segment_bytes = _fetch_url(url, fallback_urls=fallbacks)
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }
    return Response(segment_bytes, media_type="video/MP2T", headers=headers)
