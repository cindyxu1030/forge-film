import asyncio

import pytest

from forge.assembler import sfx_sink as sfx_sink_module
from forge.assembler.sfx_sink import SfxSink
from forge.providers import sfx as sfx_module
from forge.providers.sfx import (
    SfxProvider,
    SoniloSfxProvider,
    TaskNotFoundError,
)

SUCCEEDED_BODY = {
    "status": "succeeded",
    "audio": {"url": "https://storage.example/sfx", "content_type": "audio/mp4"},
}


class FakeSfxProvider(SfxProvider):
    """Test double: writes a fake audio file without any API or ffmpeg calls."""

    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    async def generate_for_video(self, video_path, output_dir, prompt=None):
        self.calls.append((video_path, prompt))
        out = f"{output_dir}/sfx.m4a"
        with open(out, "wb") as f:
            f.write(b"fake-sfx")
        return out


async def _no_sleep(_seconds):
    return None


def _sonilo_provider(monkeypatch, tmp_path, task_bodies, duration=10.0, poll_timeout=60.0):
    """Build a SoniloSfxProvider with every network seam replaced.

    task_bodies: sequence of dicts (poll responses) and/or exceptions
    (raised by that poll). The submit and artifact download are recorded,
    never performed.
    """
    monkeypatch.setattr(sfx_module, "_probe_duration", lambda path: duration)
    monkeypatch.setattr(sfx_module, "_poll_sleep", _no_sleep)
    provider = SoniloSfxProvider(api_key="test-key", poll_timeout_seconds=poll_timeout)
    calls = {"submits": [], "polls": 0, "downloads": []}

    async def fake_submit(video_path, prompt):
        calls["submits"].append((video_path, prompt))
        return "task-123"

    responses = iter(task_bodies)

    async def fake_get_task(task_id):
        calls["polls"] += 1
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    async def fake_download(url, dest):
        calls["downloads"].append(url)
        with open(dest, "wb") as f:
            f.write(b"sfx-bytes")

    monkeypatch.setattr(provider, "_submit_task", fake_submit)
    monkeypatch.setattr(provider, "_get_task", fake_get_task)
    monkeypatch.setattr(provider, "_download_artifact", fake_download)
    return provider, calls


def _video(tmp_path, name="final.mp4", content=b"video-bytes"):
    video = tmp_path / name
    video.write_bytes(content)
    return video


# ── Gating ──────────────────────────────────────────────────────────────────

def test_config_sfx_disabled_by_default(tmp_path):
    from forge.config import ForgeConfig

    cfg = ForgeConfig(tmp_path / "missing.yaml")
    assert cfg.sfx_enabled is False
    assert cfg.sfx_provider == "sonilo"
    assert cfg.sfx_prompt is None


def test_sonilo_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("SONILO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SONILO_API_KEY"):
        SoniloSfxProvider(api_key="")


# ── Duration cap ────────────────────────────────────────────────────────────

def test_duration_cap_rejects_long_video_before_upload(monkeypatch, tmp_path):
    provider, calls = _sonilo_provider(monkeypatch, tmp_path, [], duration=200.0)
    video = _video(tmp_path)
    with pytest.raises(RuntimeError, match="180"):
        asyncio.run(provider.generate_for_video(str(video), str(tmp_path)))
    assert calls["submits"] == []


def test_duration_cap_fails_open_when_probe_unavailable(monkeypatch, tmp_path):
    provider, calls = _sonilo_provider(
        monkeypatch, tmp_path, [SUCCEEDED_BODY], duration=None
    )
    video = _video(tmp_path)
    result = asyncio.run(provider.generate_for_video(str(video), str(tmp_path)))
    assert result.endswith("sfx.m4a")
    assert calls["submits"] == [(str(video), None)]


# ── Poll paths ──────────────────────────────────────────────────────────────

def test_poll_waits_through_processing_then_downloads_audio(monkeypatch, tmp_path):
    provider, calls = _sonilo_provider(
        monkeypatch, tmp_path, [{"status": "processing"}, SUCCEEDED_BODY]
    )
    video = _video(tmp_path)
    result = asyncio.run(
        provider.generate_for_video(str(video), str(tmp_path), prompt="rain on glass")
    )
    assert result == str(tmp_path / "sfx.m4a")
    assert (tmp_path / "sfx.m4a").read_bytes() == b"sfx-bytes"
    assert calls["polls"] == 2
    assert calls["submits"] == [(str(video), "rain on glass")]
    assert calls["downloads"] == ["https://storage.example/sfx"]


def test_poll_task_not_found_fails_immediately(monkeypatch, tmp_path):
    provider, calls = _sonilo_provider(
        monkeypatch, tmp_path, [TaskNotFoundError("task not found"), SUCCEEDED_BODY]
    )
    video = _video(tmp_path)
    with pytest.raises(TaskNotFoundError):
        asyncio.run(provider.generate_for_video(str(video), str(tmp_path)))
    assert calls["polls"] == 1
    assert calls["downloads"] == []


def test_poll_retries_transient_errors(monkeypatch, tmp_path):
    provider, calls = _sonilo_provider(
        monkeypatch,
        tmp_path,
        [RuntimeError("Sonilo API error (502): upstream"), SUCCEEDED_BODY],
    )
    video = _video(tmp_path)
    result = asyncio.run(provider.generate_for_video(str(video), str(tmp_path)))
    assert result.endswith("sfx.m4a")
    assert calls["polls"] == 2


def test_poll_failed_task_raises_with_backend_message(monkeypatch, tmp_path):
    failed = {
        "status": "failed",
        "error": {"code": "GENERATION_FAILED", "message": "scene too dark"},
        "refunded": True,
    }
    provider, _ = _sonilo_provider(monkeypatch, tmp_path, [failed])
    video = _video(tmp_path)
    with pytest.raises(RuntimeError, match="scene too dark"):
        asyncio.run(provider.generate_for_video(str(video), str(tmp_path)))


def test_poll_timeout_raises_with_task_id(monkeypatch, tmp_path):
    provider, _ = _sonilo_provider(
        monkeypatch, tmp_path, [{"status": "processing"}], poll_timeout=0.0
    )
    video = _video(tmp_path)
    with pytest.raises(RuntimeError, match="task-123"):
        asyncio.run(provider.generate_for_video(str(video), str(tmp_path)))


def test_succeeded_without_audio_artifact_raises(monkeypatch, tmp_path):
    provider, _ = _sonilo_provider(monkeypatch, tmp_path, [{"status": "succeeded"}])
    video = _video(tmp_path)
    with pytest.raises(RuntimeError, match="no audio"):
        asyncio.run(provider.generate_for_video(str(video), str(tmp_path)))


# ── Sink gating and failure fallback ────────────────────────────────────────

def test_sfx_sink_skips_missing_video(tmp_path):
    provider = FakeSfxProvider()
    sink = SfxSink(provider, output_path=str(tmp_path / "final_with_sfx.mp4"))
    result = asyncio.run(sink.finalize(str(tmp_path / "does_not_exist.mp4")))
    assert result is None
    assert provider.calls == []


def test_sfx_sink_provider_failure_leaves_prior_output_intact(tmp_path):
    class FailingProvider(SfxProvider):
        async def generate_for_video(self, video_path, output_dir, prompt=None):
            raise RuntimeError("Sonilo API error (502): upstream")

    video = _video(tmp_path, name="final_with_music.mp4")
    sink = SfxSink(FailingProvider(), output_path=str(tmp_path / "final_with_music_and_sfx.mp4"))
    result = asyncio.run(sink.finalize(str(video)))
    assert result is None
    assert video.read_bytes() == b"video-bytes"
    assert not (tmp_path / "final_with_music_and_sfx.mp4").exists()


# ── Mux correctness ─────────────────────────────────────────────────────────

def _run_mux(tmp_path, monkeypatch, has_audio, out_name):
    """Run the sink with a fake ffmpeg run and return the recorded command."""
    recorded: dict = {}

    def fake_run(cmd, check, capture_output):
        recorded["cmd"] = cmd
        (tmp_path / out_name).write_bytes(b"muxed")
        return None

    monkeypatch.setattr(sfx_sink_module.subprocess, "run", fake_run)
    monkeypatch.setattr(sfx_sink_module, "_has_audio_stream", lambda path: has_audio)

    video = _video(tmp_path)
    provider = FakeSfxProvider()
    out_path = str(tmp_path / out_name)
    sink = SfxSink(provider, output_path=out_path, prompt="rain on glass")
    result = asyncio.run(sink.finalize(str(video)))

    assert result == out_path
    assert provider.calls == [(str(video), "rain on glass")]
    return recorded["cmd"], out_path


def test_sfx_sink_mux_mixes_when_input_has_audio(tmp_path, monkeypatch):
    """Chained after the music sink: video stream-copied, music + effects mixed."""
    cmd, out_path = _run_mux(
        tmp_path, monkeypatch, has_audio=True, out_name="final_with_music_and_sfx.mp4"
    )
    assert cmd[:2] == ["ffmpeg", "-y"]
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert "amix=inputs=2" in cmd[cmd.index("-filter_complex") + 1]
    assert "[aout]" in cmd
    # -shortest would clip tail frames off the stream-copied video; the
    # amix duration=first bound makes it unnecessary here.
    assert "-shortest" not in cmd
    assert cmd[-1] == out_path


def test_sfx_sink_mux_maps_effects_when_input_is_silent(tmp_path, monkeypatch):
    """Without a music track, the effects become the audio track directly."""
    cmd, out_path = _run_mux(
        tmp_path, monkeypatch, has_audio=False, out_name="final_with_sfx.mp4"
    )
    assert cmd[:2] == ["ffmpeg", "-y"]
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert "-filter_complex" not in cmd
    assert cmd[cmd.index("-map", cmd.index("-map") + 1) + 1] == "1:a"
    assert "-shortest" in cmd
    assert cmd[-1] == out_path
