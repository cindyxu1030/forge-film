"""Abstract Music Provider layer.

Supports: Sonilo video-to-music, Mock.
All providers expose a common generate_for_video(video_path) -> local_path
interface returning an audio file generated for the given video.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

SONILO_API_URL = "https://api.sonilo.com"
SONILO_VIDEO_TO_MUSIC_PATH = "/v1/video-to-music"
SONILO_API_KEYS_URL = "https://platform.sonilo.com/dashboard/api-keys"

# Matches the backend's generation read timeout: a long generation can keep
# running (and charging) on the backend for up to 600s, so timing out the
# client sooner would orphan a paid request.
GENERATION_TIMEOUT_SECONDS = 600


class MusicProvider(ABC):
    """Abstract base for music generation providers used by MusicSink."""

    @abstractmethod
    async def generate_for_video(
        self, video_path: str, output_dir: str, prompt: str | None = None
    ) -> str:
        """Generate a music track for the video, save locally, return file path."""
        ...


def _extract_detail(body: str) -> str:
    """Pull the `detail` field out of an API error body, falling back to the raw body."""
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            detail = parsed.get("detail") or parsed.get("error") or parsed.get("message")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
        return body
    except (json.JSONDecodeError, TypeError):
        return body


async def consume_ndjson_stream(lines: AsyncIterator[str]) -> tuple[bytes, str | None]:
    """Consume the NDJSON event stream returned by Sonilo generation endpoints.

    Event types: `audio_chunk` (base64 audio bytes per stream_index), `title`,
    `complete` (terminal success), `error` (terminal failure). Progress events
    such as `stage_start`/`stage_complete` and malformed lines are ignored.

    Returns:
        (audio_bytes, title_or_none) — audio bytes of the first stream index.
    Raises:
        RuntimeError on an `error` event or if the stream ends without `complete`.
    """
    streams: dict[int, bytearray] = {}
    title: str | None = None
    completed = False
    error_msg: str | None = None

    async for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "audio_chunk":
            try:
                idx = int(event.get("stream_index", 0))
            except (TypeError, ValueError):
                continue
            data = event.get("data")
            if idx < 0 or not isinstance(data, str):
                continue
            try:
                decoded = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                continue
            streams.setdefault(idx, bytearray()).extend(decoded)
        elif etype == "title":
            value = event.get("title")
            if isinstance(value, str) and value.strip():
                title = value.strip()
        elif etype == "complete":
            completed = True
        elif etype == "error":
            error_msg = event.get("message") or event.get("code") or "stream error"
            break

    if error_msg:
        raise RuntimeError(f"Sonilo stream error: {error_msg}")
    if not completed:
        raise RuntimeError("Sonilo stream ended without a `complete` event")

    first_index = min(streams.keys(), default=None)
    if first_index is None or not streams[first_index]:
        raise RuntimeError("Sonilo stream completed but no audio data was received")
    return bytes(streams[first_index]), title


class SoniloMusicProvider(MusicProvider):
    """Sonilo video-to-music: an original track generated from the finished cut.

    The track is generated from the video itself, so its length matches the
    video automatically. Output is licensed and safe for commercial use
    (terms apply). Returns an .m4a file (AAC in an MP4 container).
    """

    def __init__(self, api_key: str = "", api_url: str = ""):
        self.api_key = api_key or os.environ.get("SONILO_API_KEY", "")
        self.api_url = (api_url or os.environ.get("SONILO_API_URL", SONILO_API_URL)).rstrip("/")
        if not self.api_key:
            raise ValueError(
                "SONILO_API_KEY is not set. The music sink needs a Sonilo API key — "
                "add SONILO_API_KEY to .env (or music.api_key in forge.yaml). "
                f"Keys: {SONILO_API_KEYS_URL}"
            )

    async def generate_for_video(
        self, video_path: str, output_dir: str, prompt: str | None = None
    ) -> str:
        import httpx

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise RuntimeError(f"Music input video not found or empty: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        data = {"prompt": prompt} if prompt else None
        url = f"{self.api_url}{SONILO_VIDEO_TO_MUSIC_PATH}"

        async with httpx.AsyncClient(timeout=GENERATION_TIMEOUT_SECONDS) as client:
            with open(video_path, "rb") as fh:
                files = {"video": (os.path.basename(video_path), fh, "video/mp4")}
                async with client.stream(
                    "POST",
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=data,
                    files=files,
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        raise RuntimeError(self._http_error(resp.status_code, body))
                    audio_bytes, _title = await consume_ndjson_stream(resp.aiter_lines())

        out_path = os.path.join(output_dir, "soundtrack.m4a")
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        return out_path

    @staticmethod
    def _http_error(status: int, body: str) -> str:
        detail = _extract_detail(body)
        if status == 401:
            return f"Sonilo API key was rejected — verify the key at {SONILO_API_KEYS_URL}"
        if status == 402:
            return detail or "Sonilo account has no remaining credits"
        if status == 413:
            return f"Sonilo upload too large: {detail}"
        if status == 429:
            return f"Sonilo rate limit exceeded: {detail}"
        return f"Sonilo API error ({status}): {detail}"


class MockMusicProvider(MusicProvider):
    """Mock music provider — writes a silent track via ffmpeg (no API calls)."""

    async def generate_for_video(
        self, video_path: str, output_dir: str, prompt: str | None = None
    ) -> str:
        import subprocess

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "soundtrack.m4a")
        duration = 3.0
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
                check=True,
                capture_output=True,
            )
            duration = float(json.loads(probe.stdout)["format"]["duration"])
        except (subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            pass
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                 "-t", f"{duration:.2f}", "-c:a", "aac", out_path],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: write empty file so downstream warnings are visible
            with open(out_path, "wb") as f:
                f.write(b"")
        return out_path
