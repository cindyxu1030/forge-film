"""Abstract SFX Provider layer.

Supports: Sonilo video-to-sfx, Mock.
All providers expose a common generate_for_video(video_path) -> local_path
interface returning a sound-effects audio file generated for the given video.

Unlike the music endpoint (an NDJSON stream, see forge.providers.music), the
SFX endpoint is task-based: POST /v1/video-to-sfx returns a task_id, and
GET /v1/tasks/{task_id} is polled until the task reaches a terminal state.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from abc import ABC, abstractmethod

SONILO_API_URL = "https://api.sonilo.com"
SONILO_VIDEO_TO_SFX_PATH = "/v1/video-to-sfx"
SONILO_TASKS_PATH = "/v1/tasks"
SONILO_API_KEYS_URL = "https://platform.sonilo.com/dashboard/api-keys"

# The backend rejects videos longer than this on /v1/video-to-sfx, so we
# pre-check locally to fail fast and skip a wasted upload. Keep this matched
# to the backend's limit.
SFX_MAX_VIDEO_DURATION_SECONDS = 180

# Video uploads are slow, and the submit request is only safe once the
# backend has fully received it — mirror the music provider's budget.
SUBMIT_TIMEOUT_SECONDS = 600
# Total budget for polling the task after submission.
POLL_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 5.0

# Test seam: monkeypatch sfx._poll_sleep to avoid real 5s waits in tests.
_poll_sleep = asyncio.sleep

# content_type -> extension for the SFX audio artifact. The backend sets
# content_type from the requested audio format; default is AAC in an MP4
# container (.m4a).
_AUDIO_CONTENT_TYPE_EXTS = {
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/flac": ".flac",
}


class SfxProvider(ABC):
    """Abstract base for sound-effects generation providers used by SfxSink."""

    @abstractmethod
    async def generate_for_video(
        self, video_path: str, output_dir: str, prompt: str | None = None
    ) -> str:
        """Generate a sound-effects track for the video, save locally, return file path."""
        ...


class TaskNotFoundError(RuntimeError):
    """The polled task_id does not exist on the backend (HTTP 404).

    /v1/tasks serves SFX tasks only, so a 404 means a bad task id or the id
    of a non-SFX (e.g. music) task — retrying can never help, unlike every
    other poll error.
    """


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


def _probe_duration(path: str) -> float | None:
    """Read a media file's duration in seconds via ffprobe (None if unavailable)."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            check=True,
            capture_output=True,
        )
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _check_video_duration(
    path: str, max_seconds: int = SFX_MAX_VIDEO_DURATION_SECONDS
) -> None:
    """Best-effort pre-check of the video duration against the backend cap.

    /v1/video-to-sfx rejects videos longer than the cap, so failing fast here
    skips a wasted upload. Fail-open: when ffprobe is unavailable or the
    duration cannot be read, the backend makes the final call.
    """
    duration = _probe_duration(path)
    if duration is not None and duration > max_seconds:
        raise RuntimeError(
            f"SFX input video is {duration:.1f}s long — "
            f"the SFX endpoint accepts up to {max_seconds}s"
        )


def _ext_from_content_type(content_type: object) -> str:
    if not isinstance(content_type, str):
        return ".m4a"
    return _AUDIO_CONTENT_TYPE_EXTS.get(content_type.lower(), ".m4a")


class SoniloSfxProvider(SfxProvider):
    """Sonilo video-to-sfx: royalty-free sound effects generated from the
    finished cut, timed to the picture.

    Task pipeline: the video is uploaded once (the request is charged on
    acceptance, so the submit is never retried), the backend returns a
    task_id, and GET /v1/tasks/{task_id} is polled until the task succeeds
    or fails. The terminal body carries the sound-effects audio artifact
    (plus a backend-mixed video, which is not used here — the sink muxes
    the audio locally so the video stream stays stream-copied and an
    existing music track is preserved).
    """

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "",
        poll_timeout_seconds: float = POLL_TIMEOUT_SECONDS,
    ):
        self.api_key = api_key or os.environ.get("SONILO_API_KEY", "")
        self.api_url = (api_url or os.environ.get("SONILO_API_URL", SONILO_API_URL)).rstrip("/")
        self.poll_timeout_seconds = poll_timeout_seconds
        if not self.api_key:
            raise ValueError(
                "SONILO_API_KEY is not set. The SFX sink needs a Sonilo API key — "
                "add SONILO_API_KEY to .env (or sfx.api_key in forge.yaml). "
                f"Keys: {SONILO_API_KEYS_URL}"
            )

    async def generate_for_video(
        self, video_path: str, output_dir: str, prompt: str | None = None
    ) -> str:
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise RuntimeError(f"SFX input video not found or empty: {video_path}")
        _check_video_duration(video_path)
        os.makedirs(output_dir, exist_ok=True)

        task_id = await self._submit_task(video_path, prompt)
        body = await self._poll_task(task_id)
        return await self._save_audio_artifact(body, output_dir, task_id)

    async def _submit_task(self, video_path: str, prompt: str | None) -> str:
        """POST the video to /v1/video-to-sfx; expect a body with a task_id.

        No retry: the backend creates (and charges) the task as soon as it
        accepts the request, so a blind resubmit could double-charge.
        """
        import httpx

        data = {"prompt": prompt} if prompt else None
        url = f"{self.api_url}{SONILO_VIDEO_TO_SFX_PATH}"
        async with httpx.AsyncClient(timeout=SUBMIT_TIMEOUT_SECONDS) as client:
            with open(video_path, "rb") as fh:
                files = {"video": (os.path.basename(video_path), fh, "video/mp4")}
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=data,
                    files=files,
                )
        if resp.status_code >= 400:
            raise RuntimeError(self._http_error(resp.status_code, resp.text))
        try:
            body = resp.json()
        except json.JSONDecodeError:
            body = None
        task_id = body.get("task_id") if isinstance(body, dict) else None
        if not task_id:
            raise RuntimeError(
                f"Sonilo accepted the SFX request (status {resp.status_code}) "
                "but returned no task_id"
            )
        return str(task_id)

    async def _get_task(self, task_id: str) -> dict:
        """Single GET of /v1/tasks/{task_id}; returns the task body.

        Raises TaskNotFoundError on 404 (see the class docstring) and
        RuntimeError on any other error.
        """
        import httpx

        url = f"{self.api_url}{SONILO_TASKS_PATH}/{task_id}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
        if resp.status_code == 404:
            raise TaskNotFoundError(
                f"Sonilo SFX task {task_id} was not found — "
                "bad task id, or the id of a non-SFX task"
            )
        if resp.status_code >= 400:
            raise RuntimeError(self._http_error(resp.status_code, resp.text))
        try:
            body = resp.json()
        except json.JSONDecodeError:
            body = None
        if not isinstance(body, dict):
            raise RuntimeError(
                f"Unexpected response for Sonilo SFX task {task_id} (expected a JSON object)"
            )
        return body

    async def _poll_task(self, task_id: str) -> dict:
        """Poll /v1/tasks/{task_id} until the task is terminal.

        A 404 means the task id itself is bad, so it fails immediately.
        Every other poll error is retried until the deadline: the poll GET
        is free and read-only, and the submitted task keeps running on the
        backend, so a transient failure here should not abandon the result.
        """
        deadline = time.monotonic() + self.poll_timeout_seconds
        last_error: Exception | None = None
        while True:
            body = None
            try:
                body = await self._get_task(task_id)
            except TaskNotFoundError:
                raise
            except Exception as e:
                last_error = e
            if body is not None:
                status = body.get("status")
                if status == "succeeded":
                    return body
                if status == "failed":
                    raise RuntimeError(self._failure_message(body, task_id))
            if time.monotonic() >= deadline:
                detail = f" (last poll error: {last_error})" if last_error else ""
                raise RuntimeError(
                    f"Timed out after {self.poll_timeout_seconds:.0f}s waiting "
                    f"for Sonilo SFX task {task_id}{detail}"
                )
            await _poll_sleep(POLL_INTERVAL_SECONDS)

    async def _save_audio_artifact(self, body: dict, output_dir: str, task_id: str) -> str:
        audio = body.get("audio")
        if not isinstance(audio, dict) or not audio.get("url"):
            raise RuntimeError(
                f"Sonilo SFX task {task_id} succeeded but returned no audio artifact"
            )
        ext = _ext_from_content_type(audio.get("content_type"))
        out_path = os.path.join(output_dir, f"sfx{ext}")
        await self._download_artifact(audio["url"], out_path)
        return out_path

    async def _download_artifact(self, url: str, dest: str) -> None:
        """Stream-download a presigned artifact URL to dest.

        Sends no Authorization header: presigned URLs carry their own auth,
        and the API key must never be sent to the storage host. A partial
        file is removed on any failure.
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=SUBMIT_TIMEOUT_SECONDS) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code >= 400:
                        raise RuntimeError(
                            f"SFX artifact download failed (status {resp.status_code})"
                        )
                    with open(dest, "wb") as fh:
                        async for chunk in resp.aiter_bytes():
                            fh.write(chunk)
        except httpx.RequestError as e:
            if os.path.exists(dest):
                os.unlink(dest)
            raise RuntimeError(f"SFX artifact download failed: {e}") from e
        except BaseException:
            if os.path.exists(dest):
                os.unlink(dest)
            raise

    @staticmethod
    def _failure_message(body: dict, task_id: str) -> str:
        err = body.get("error")
        if isinstance(err, dict):
            code = err.get("code") or "GENERATION_FAILED"
            message = err.get("message") or "generation failed"
        elif isinstance(err, str) and err:
            code, message = "GENERATION_FAILED", err
        else:
            code, message = "GENERATION_FAILED", "generation failed"
        if body.get("refunded") is True:
            refund = "the charge was reversed"
        else:
            refund = "check your Sonilo usage to reconcile the charge"
        return f"Sonilo SFX task {task_id} failed ({code}): {message} — {refund}"

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


class MockSfxProvider(SfxProvider):
    """Mock SFX provider — writes a silent track via ffmpeg (no API calls)."""

    async def generate_for_video(
        self, video_path: str, output_dir: str, prompt: str | None = None
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "sfx.m4a")
        duration = _probe_duration(video_path) or 3.0
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
