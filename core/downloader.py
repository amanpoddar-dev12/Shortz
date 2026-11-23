"""Video ingestion, lossless stream merging, audio extraction, and heatmap parsing.

Uses yt-dlp for YouTube metadata/heatmap extraction and separate high-quality
stream downloads, followed by native FFmpeg subprocesses for lossless MP4 merging
and 16kHz mono WAV extraction for Whisper transcription.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from core.models import HeatmapPoint, IngestionResult

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when video ingestion, stream merging, or audio extraction fails."""


def _require_binary(name: str) -> str:
    """Return path to a required CLI binary or raise a clear DownloadError."""
    path = shutil.which(name)
    if not path:
        raise DownloadError(
            f"Required binary '{name}' was not found on PATH. "
            "Install FFmpeg (including ffprobe) and ensure it is available on your system PATH."
        )
    return path


def _sanitize_id(raw_id: str) -> str:
    """Sanitize a video ID or filename stem for safe directory creation."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", raw_id).strip("_")
    return cleaned or "video"


def _run_subprocess(cmd: list[str], description: str) -> subprocess.CompletedProcess[str]:
    """Execute a subprocess command and raise DownloadError on non-zero exit."""
    logger.debug("Running %s: %s", description, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise DownloadError(f"Failed to execute {description}: {exc}") from exc

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-800:]
        raise DownloadError(
            f"{description} failed (exit code {proc.returncode}): {stderr_tail}"
        )
    return proc


def _probe_media(video_path: Path) -> dict[str, Any]:
    """Probe duration, width, height, and FPS from a media file using ffprobe."""
    ffprobe_bin = _require_binary("ffprobe")
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    proc = _run_subprocess(cmd, f"ffprobe on {video_path.name}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise DownloadError(f"Invalid ffprobe JSON output for {video_path}: {exc}") from exc

    duration = float(data.get("format", {}).get("duration") or 0.0)
    width = 0
    height = 0
    fps = 0.0

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            rate_str = str(stream.get("r_frame_rate") or "0/1")
            if "/" in rate_str:
                num_s, den_s = rate_str.split("/", 1)
                num, den = float(num_s), float(den_s)
                fps = (num / den) if den > 0 else 0.0
            else:
                fps = float(rate_str)
            break

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
    }


def _extract_wav_16k(video_path: Path, wav_path: Path) -> None:
    """Extract a clean 16kHz mono PCM WAV file from a video using FFmpeg."""
    ffmpeg_bin = _require_binary("ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    _run_subprocess(cmd, f"16kHz mono WAV extraction to {wav_path.name}")


def _merge_streams(video_stream: Path, audio_stream: Path, output_mp4: Path) -> None:
    """Losslessly merge separate video and audio streams into a single MP4 via FFmpeg."""
    ffmpeg_bin = _require_binary("ffmpeg")
    copy_cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_stream),
        "-i",
        str(audio_stream),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_mp4),
    ]
    try:
        _run_subprocess(copy_cmd, "lossless FFmpeg stream merge")
    except DownloadError:
        logger.warning(
            "Direct audio stream copy failed for %s; falling back to AAC audio remux.",
            output_mp4.name,
        )
        aac_cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_stream),
            "-i",
            str(audio_stream),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_mp4),
        ]
        _run_subprocess(aac_cmd, "FFmpeg stream merge with AAC audio")


def _parse_heatmap(raw_heatmap: Optional[list[dict[str, Any]]]) -> list[HeatmapPoint]:
    """Parse and normalize YouTube 'Most Replayed' heatmap entries."""
    if not raw_heatmap:
        return []

    raw_values = [float(item.get("value", 0.0)) for item in raw_heatmap]
    max_val = max(raw_values) if raw_values else 1.0
    scale = max_val if max_val > 1.0 else 1.0

    points: list[HeatmapPoint] = []
    for item in raw_heatmap:
        start_t = float(item.get("start_time", 0.0))
        end_t = float(item.get("end_time", 0.0))
        val = float(item.get("value", 0.0)) / scale if scale > 0 else 0.0
        clamped = max(0.0, min(1.0, val))
        if end_t > start_t >= 0.0:
            points.append(
                HeatmapPoint(
                    start_time=round(start_t, 3),
                    end_time=round(end_t, 3),
                    intensity=round(clamped, 4),
                )
            )
    return points


class YouTubeDownloader:
    """Handles YouTube video/heatmap ingestion and local video bypass."""

    def __init__(self, output_dir: str = "downloads") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_and_extract(self, input_source: str) -> IngestionResult:
        """Process a YouTube URL or local video file path.

        Args:
            input_source: YouTube video URL or path to a local video file.

        Returns:
            IngestionResult containing paths to the MP4 video, 16kHz mono WAV,
            and parsed YouTube engagement heatmap points.
        """
        source_path = Path(input_source)
        if source_path.is_file():
            return self._process_local_file(source_path)
        return self._process_youtube_url(input_source)

    def _process_local_file(self, local_path: Path) -> IngestionResult:
        """Bypass yt-dlp for a local video file and extract 16kHz mono WAV."""
        logger.info("Local file detected, bypassing yt-dlp: %s", local_path)
        short_hash = hashlib.sha1(str(local_path.resolve()).encode("utf-8")).hexdigest()[:8]
        video_id = f"{_sanitize_id(local_path.stem)}_{short_hash}"
        work_dir = self.output_dir / video_id
        work_dir.mkdir(parents=True, exist_ok=True)

        wav_path = work_dir / f"{video_id}_16k.wav"
        _extract_wav_16k(local_path, wav_path)
        probe = _probe_media(local_path)

        if probe["duration"] <= 0.0:
            raise DownloadError(f"Could not determine valid duration for local file: {local_path}")

        return IngestionResult(
            video_path=str(local_path.resolve()),
            audio_wav_path=str(wav_path.resolve()),
            heatmap_points=[],
            video_id=video_id,
            duration=probe["duration"],
            title=local_path.stem,
            channel="Local File",
            width=probe["width"],
            height=probe["height"],
            fps=probe["fps"],
            source_url=str(local_path.resolve()),
        )

    def _process_youtube_url(self, url: str) -> IngestionResult:
        """Download separate video/audio streams, merge via FFmpeg, and extract heatmap."""
        try:
            import yt_dlp
        except ImportError as exc:
            raise DownloadError("yt-dlp is not installed. Run: pip install yt-dlp") from exc

        logger.info("Extracting metadata and heatmap for URL: %s", url)
        info_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            raise DownloadError(f"Failed to fetch YouTube metadata for {url}: {exc}") from exc

        if not info or not info.get("id"):
            raise DownloadError(f"Invalid metadata returned by yt-dlp for {url}")

        video_id = _sanitize_id(str(info["id"]))
        work_dir = self.output_dir / video_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Save raw info JSON for debugging/caching
        info_json_path = work_dir / f"{video_id}.info.json"
        info_json_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

        heatmap_points = _parse_heatmap(info.get("heatmap"))
        logger.info("Parsed %d heatmap points for video %s", len(heatmap_points), video_id)

        merged_mp4 = work_dir / f"{video_id}.mp4"
        wav_path = work_dir / f"{video_id}_16k.wav"

        if not merged_mp4.exists():
            raw_video_path = self._download_stream(
                url=url,
                fmt="bestvideo[ext=mp4]/bestvideo",
                out_template=str(work_dir / "raw_video.%(ext)s"),
                label="video stream",
            )
            raw_audio_path = self._download_stream(
                url=url,
                fmt="bestaudio[ext=m4a]/bestaudio",
                out_template=str(work_dir / "raw_audio.%(ext)s"),
                label="audio stream",
            )
            _merge_streams(raw_video_path, raw_audio_path, merged_mp4)

            # Clean up temporary unmerged stream files
            for temp_file in (raw_video_path, raw_audio_path):
                if temp_file.exists():
                    temp_file.unlink()
        else:
            logger.info("Using cached merged video: %s", merged_mp4)

        if not wav_path.exists():
            _extract_wav_16k(merged_mp4, wav_path)

        probe = _probe_media(merged_mp4)
        duration = probe["duration"] or float(info.get("duration") or 0.0)
        if duration <= 0.0:
            raise DownloadError(f"Could not determine valid duration for {video_id}")

        return IngestionResult(
            video_path=str(merged_mp4.resolve()),
            audio_wav_path=str(wav_path.resolve()),
            heatmap_points=heatmap_points,
            video_id=video_id,
            duration=duration,
            title=str(info.get("title") or video_id),
            channel=str(info.get("uploader") or info.get("channel") or "Unknown"),
            width=probe["width"] or int(info.get("width") or 0),
            height=probe["height"] or int(info.get("height") or 0),
            fps=probe["fps"] or float(info.get("fps") or 0.0),
            source_url=url,
        )

    def _download_stream(self, url: str, fmt: str, out_template: str, label: str) -> Path:
        """Download a single stream using yt-dlp and return the resulting file path."""
        import yt_dlp

        opts: dict[str, Any] = {
            "format": fmt,
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "overwrites": True,
        }
        logger.info("Downloading %s...", label)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_path = Path(ydl.prepare_filename(info))
        except Exception as exc:
            raise DownloadError(f"Failed to download {label} for {url}: {exc}") from exc

        if not downloaded_path.exists():
            raise DownloadError(f"Expected {label} at {downloaded_path}, but file was not created.")
        return downloaded_path
