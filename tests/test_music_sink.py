import asyncio
import base64
import json

import pytest

from forge.assembler import music_sink as music_sink_module
from forge.assembler.music_sink import MusicSink
from forge.providers.music import (
    MusicProvider,
    SoniloMusicProvider,
    consume_ndjson_stream,
)


async def _lines(items: list[str]):
    for item in items:
        yield item


def _chunk(data: bytes, stream_index: int = 0) -> str:
    return json.dumps(
        {"type": "audio_chunk", "stream_index": stream_index, "data": base64.b64encode(data).decode()}
    )


class FakeMusicProvider(MusicProvider):
    """Test double: writes a fake audio file without any API or ffmpeg calls."""

    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    async def generate_for_video(self, video_path, output_dir, prompt=None):
        self.calls.append((video_path, prompt))
        out = f"{output_dir}/soundtrack.m4a"
        with open(out, "wb") as f:
            f.write(b"fake-audio")
        return out


def test_consume_ndjson_stream_assembles_audio():
    events = [
        json.dumps({"type": "stage_start", "stage": "analysis"}),  # ignored
        _chunk(b"hello "),
        _chunk(b"world"),
        json.dumps({"type": "title", "title": "Night Drive"}),
        json.dumps({"type": "complete"}),
    ]
    audio, title = asyncio.run(consume_ndjson_stream(_lines(events)))
    assert audio == b"hello world"
    assert title == "Night Drive"


def test_consume_ndjson_stream_uses_first_stream_index():
    events = [
        _chunk(b"second", stream_index=1),
        _chunk(b"first", stream_index=0),
        json.dumps({"type": "complete"}),
    ]
    audio, _ = asyncio.run(consume_ndjson_stream(_lines(events)))
    assert audio == b"first"


def test_consume_ndjson_stream_ignores_malformed_lines():
    events = [
        "not json at all",
        json.dumps(["a", "list"]),
        _chunk(b"data"),
        json.dumps({"type": "complete"}),
    ]
    audio, title = asyncio.run(consume_ndjson_stream(_lines(events)))
    assert audio == b"data"
    assert title is None


def test_consume_ndjson_stream_error_event_raises():
    events = [_chunk(b"partial"), json.dumps({"type": "error", "message": "quota exhausted"})]
    with pytest.raises(RuntimeError, match="quota exhausted"):
        asyncio.run(consume_ndjson_stream(_lines(events)))


def test_consume_ndjson_stream_requires_complete_event():
    events = [_chunk(b"data")]
    with pytest.raises(RuntimeError, match="complete"):
        asyncio.run(consume_ndjson_stream(_lines(events)))


def test_consume_ndjson_stream_requires_audio():
    events = [json.dumps({"type": "complete"})]
    with pytest.raises(RuntimeError, match="no audio"):
        asyncio.run(consume_ndjson_stream(_lines(events)))


def test_sonilo_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("SONILO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SONILO_API_KEY"):
        SoniloMusicProvider(api_key="")


def test_config_music_disabled_by_default(tmp_path):
    from forge.config import ForgeConfig

    cfg = ForgeConfig(tmp_path / "missing.yaml")
    assert cfg.music_enabled is False
    assert cfg.music_provider == "sonilo"
    assert cfg.music_prompt is None


def test_music_sink_skips_missing_video(tmp_path):
    provider = FakeMusicProvider()
    sink = MusicSink(provider, output_path=str(tmp_path / "final_with_music.mp4"))
    result = asyncio.run(sink.finalize(str(tmp_path / "does_not_exist.mp4")))
    assert result is None
    assert provider.calls == []


def test_music_sink_provider_failure_leaves_master_untouched(tmp_path):
    class FailingProvider(MusicProvider):
        async def generate_for_video(self, video_path, output_dir, prompt=None):
            raise RuntimeError("Sonilo API error (502): upstream")

    video = tmp_path / "final.mp4"
    video.write_bytes(b"video-bytes")
    sink = MusicSink(FailingProvider(), output_path=str(tmp_path / "final_with_music.mp4"))
    result = asyncio.run(sink.finalize(str(video)))
    assert result is None
    assert video.read_bytes() == b"video-bytes"


def test_music_sink_mux_copies_video_stream(tmp_path, monkeypatch):
    """The mux must stream-copy the video, encode AAC audio, and fade the tail."""
    recorded: dict = {}

    def fake_run(cmd, check, capture_output):
        recorded["cmd"] = cmd
        (tmp_path / "final_with_music.mp4").write_bytes(b"muxed")
        return None

    monkeypatch.setattr(music_sink_module.subprocess, "run", fake_run)
    monkeypatch.setattr(music_sink_module, "_probe_duration", lambda path: 10.0)

    video = tmp_path / "final.mp4"
    video.write_bytes(b"video-bytes")
    provider = FakeMusicProvider()
    out_path = str(tmp_path / "final_with_music.mp4")
    sink = MusicSink(provider, output_path=out_path, prompt="warm strings")

    result = asyncio.run(sink.finalize(str(video)))

    assert result == out_path
    assert provider.calls == [(str(video), "warm strings")]
    cmd = recorded["cmd"]
    assert cmd[:2] == ["ffmpeg", "-y"]
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert "-af" in cmd
    assert cmd[cmd.index("-af") + 1] == "afade=t=out:st=8.00:d=2.00"
    assert cmd[-1] == out_path


def test_music_sink_skips_fade_when_duration_unknown(tmp_path, monkeypatch):
    recorded: dict = {}

    def fake_run(cmd, check, capture_output):
        recorded["cmd"] = cmd
        (tmp_path / "final_with_music.mp4").write_bytes(b"muxed")
        return None

    monkeypatch.setattr(music_sink_module.subprocess, "run", fake_run)
    monkeypatch.setattr(music_sink_module, "_probe_duration", lambda path: None)

    video = tmp_path / "final.mp4"
    video.write_bytes(b"video-bytes")
    sink = MusicSink(FakeMusicProvider(), output_path=str(tmp_path / "final_with_music.mp4"))
    asyncio.run(sink.finalize(str(video)))
    assert "-af" not in recorded["cmd"]
