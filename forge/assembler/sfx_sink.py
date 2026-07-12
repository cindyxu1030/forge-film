import json
import os
import subprocess

from rich.console import Console

from forge.providers.sfx import SfxProvider

AUDIO_BITRATE = "192k"


def _has_audio_stream(path: str) -> bool:
    """Whether a media file already carries an audio stream (via ffprobe).

    Best-effort: an unprobeable input counts as silent, so the mux falls
    back to mapping the effects track directly.
    """
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            check=True,
            capture_output=True,
        )
        data = json.loads(proc.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    streams = data.get("streams") if isinstance(data, dict) else None
    if not isinstance(streams, list):
        return False
    return any(isinstance(s, dict) and s.get("codec_type") == "audio" for s in streams)


class SfxSink:
    """Terminal sink node: adds generated sound effects to the assembled cut.

    Consumes the latest assembled video (final.mp4, or final_with_music.mp4
    when the music sink ran first), asks the configured SfxProvider for a
    sound-effects track generated from the video and timed to the picture,
    and muxes it onto a copy of the input. The video stream is always
    stream-copied; when the input already carries audio (a music track),
    the effects are mixed into it, otherwise they become the audio track.
    Writes final_with_sfx.mp4 (or final_with_music_and_sfx.mp4 after the
    music sink); the input file is never modified.

    Opt-in: only runs when `sfx.enabled` is set in forge.yaml or the CLI
    is invoked with `--sfx`. A provider failure leaves the prior output
    intact.
    """

    def __init__(
        self,
        provider: SfxProvider,
        output_path: str = "./output/final_with_sfx.mp4",
        prompt: str | None = None,
        console: Console | None = None,
    ):
        self.provider = provider
        self.output_path = output_path
        self.prompt = prompt
        self.console = console or Console()

    async def finalize(self, video_path: str) -> str | None:
        if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            self.console.print("[yellow]SFX sink: no assembled video to work with — skipping.[/yellow]")
            return None

        out_dir = os.path.dirname(self.output_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        self.console.print("[bold]SFX sink:[/bold] generating sound effects from the final cut...")
        try:
            audio_path = await self.provider.generate_for_video(
                video_path, out_dir, prompt=self.prompt
            )
        except Exception as e:
            self.console.print(f"[red]SFX sink failed:[/red] {e}")
            self.console.print(f"[yellow]{os.path.basename(video_path)} is untouched.[/yellow]")
            return None

        result = self._mux(video_path, audio_path)
        if result:
            size_kb = os.path.getsize(result) / 1024
            self.console.print(f"[green]Final video with sound effects:[/green] {result} ({size_kb:.1f} KB)")
        return result

    def _mux(self, video_path: str, audio_path: str) -> str | None:
        """Mux the effects onto the video: stream-copy video, encode AAC audio.
        When the input already carries audio (the music sink's output), the
        effects are mixed into it at original levels instead of replacing it."""
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path]
        if _has_audio_stream(video_path):
            # duration=first already bounds the mix to the existing audio
            # track (itself trimmed to the video by the music sink), so
            # -shortest is not needed — adding it can clip tail frames off
            # the stream-copied video.
            cmd += [
                "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[aout]",
                "-map", "0:v", "-map", "[aout]",
            ]
        else:
            cmd += ["-map", "0:v", "-map", "1:a", "-shortest"]
        cmd += [
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            self.output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return self.output_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.console.print(f"[yellow]SFX mux warning:[/yellow] {e}")
            return None
