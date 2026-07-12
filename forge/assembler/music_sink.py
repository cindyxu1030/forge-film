import json
import os
import subprocess

from rich.console import Console

from forge.providers.music import MusicProvider

FADE_OUT_SECONDS = 2.0
AUDIO_BITRATE = "192k"


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


class MusicSink:
    """Terminal sink node: adds a generated music track to the assembled cut.

    Consumes the concatenated final.mp4 produced by StreamAssembler, asks the
    configured MusicProvider for a track generated from the finished cut, and
    muxes it onto a copy of the master (video stream copied untouched, AAC
    audio, short tail fade). Writes final_with_music.mp4 next to final.mp4;
    final.mp4 itself is never modified.

    Opt-in: only runs when `music.enabled` is set in forge.yaml or the CLI
    is invoked with `--music`. A provider failure leaves final.mp4 intact.
    """

    def __init__(
        self,
        provider: MusicProvider,
        output_path: str = "./output/final_with_music.mp4",
        prompt: str | None = None,
        fade_seconds: float = FADE_OUT_SECONDS,
        console: Console | None = None,
    ):
        self.provider = provider
        self.output_path = output_path
        self.prompt = prompt
        self.fade_seconds = fade_seconds
        self.console = console or Console()

    async def finalize(self, video_path: str) -> str | None:
        if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            self.console.print("[yellow]Music sink: no assembled video to work with — skipping.[/yellow]")
            return None

        out_dir = os.path.dirname(self.output_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        self.console.print("[bold]Music sink:[/bold] generating a track from the final cut...")
        try:
            audio_path = await self.provider.generate_for_video(
                video_path, out_dir, prompt=self.prompt
            )
        except Exception as e:
            self.console.print(f"[red]Music sink failed:[/red] {e}")
            self.console.print("[yellow]final.mp4 is untouched.[/yellow]")
            return None

        result = self._mux(video_path, audio_path)
        if result:
            size_kb = os.path.getsize(result) / 1024
            self.console.print(f"[green]Final video with music:[/green] {result} ({size_kb:.1f} KB)")
        return result

    def _mux(self, video_path: str, audio_path: str) -> str | None:
        """Mux the track onto the video: stream-copy video, encode AAC audio,
        apply a short tail fade, stop at whichever stream ends first."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        ]
        duration = _probe_duration(video_path)
        if duration and duration > self.fade_seconds:
            fade_start = duration - self.fade_seconds
            cmd += ["-af", f"afade=t=out:st={fade_start:.2f}:d={self.fade_seconds:.2f}"]
        cmd += ["-shortest", self.output_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return self.output_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.console.print(f"[yellow]Music mux warning:[/yellow] {e}")
            return None
