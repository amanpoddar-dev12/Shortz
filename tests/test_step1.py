"""Unit tests for Step 1: core.downloader and core.transcriber."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.downloader import (
    DownloadError,
    YouTubeDownloader,
    _parse_heatmap,
)
from core.transcriber import Transcriber, TranscriptionError


def test_parse_heatmap_normalizes_and_filters() -> None:
    raw = [
        {"start_time": 0.0, "end_time": 5.0, "value": 0.25},
        {"start_time": 5.0, "end_time": 10.0, "value": 2.0},
        {"start_time": 10.0, "end_time": 10.0, "value": 0.5},  # invalid zero-length
    ]
    points = _parse_heatmap(raw)
    assert len(points) == 2
    assert points[0].intensity == 0.125
    assert points[1].intensity == 1.0


def test_local_file_bypass_extracts_wav_and_empty_heatmap(tmp_path: Path) -> None:
    local_mp4 = tmp_path / "sample_talk.mp4"
    local_mp4.write_bytes(b"fake-mp4-data")

    downloader = YouTubeDownloader(output_dir=str(tmp_path / "downloads"))

    with (
        patch("core.downloader._extract_wav_16k") as mock_wav,
        patch(
            "core.downloader._probe_media",
            return_value={"duration": 42.5, "width": 1920, "height": 1080, "fps": 30.0},
        ),
    ):
        result = downloader.download_and_extract(str(local_mp4))

    mock_wav.assert_called_once()
    assert result.video_path == str(local_mp4.resolve())
    assert result.audio_wav_path.endswith("_16k.wav")
    assert result.heatmap_points == []
    assert result.duration == 42.5
    assert result.width == 1920


def test_youtube_download_merges_streams_and_parses_heatmap(tmp_path: Path) -> None:
    downloader = YouTubeDownloader(output_dir=str(tmp_path / "downloads"))
    fake_info = {
        "id": "abc123XYZ",
        "title": "Test Video",
        "uploader": "Test Channel",
        "duration": 60.0,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "heatmap": [
            {"start_time": 0.0, "end_time": 10.0, "value": 0.4},
            {"start_time": 10.0, "end_time": 20.0, "value": 1.0},
        ],
    }

    fake_ydl = MagicMock()
    fake_ydl.__enter__.return_value = fake_ydl
    fake_ydl.extract_info.return_value = fake_info

    def fake_download_stream(self: YouTubeDownloader, url: str, fmt: str, out_template: str, label: str) -> Path:
        p = Path(out_template.replace("%(ext)s", "mp4" if "video" in label else "m4a"))
        p.write_bytes(b"stream-bytes")
        return p

    with (
        patch("yt_dlp.YoutubeDL", return_value=fake_ydl),
        patch.object(YouTubeDownloader, "_download_stream", fake_download_stream),
        patch("core.downloader._merge_streams") as mock_merge,
        patch("core.downloader._extract_wav_16k") as mock_wav,
        patch(
            "core.downloader._probe_media",
            return_value={"duration": 60.0, "width": 1920, "height": 1080, "fps": 30.0},
        ),
    ):
        result = downloader.download_and_extract("https://www.youtube.com/watch?v=abc123XYZ")

    mock_merge.assert_called_once()
    mock_wav.assert_called_once()
    assert result.video_id == "abc123XYZ"
    assert len(result.heatmap_points) == 2
    assert result.heatmap_points[1].intensity == 1.0
    assert json.loads((tmp_path / "downloads" / "abc123XYZ" / "abc123XYZ.info.json").read_text())["id"] == "abc123XYZ"


def test_transcriber_missing_audio_raises(tmp_path: Path) -> None:
    transcriber = Transcriber(engine="faster-whisper")
    with pytest.raises(TranscriptionError, match="Audio file not found"):
        transcriber.transcribe(tmp_path / "missing.wav", video_id="vid1")


def test_transcriber_faster_whisper_returns_word_timestamps(tmp_path: Path) -> None:
    wav = tmp_path / "test_16k.wav"
    wav.write_bytes(b"RIFFfake")

    fake_words = [
        SimpleNamespace(word=" Hello", start=0.1, end=0.45, probability=0.98),
        SimpleNamespace(word=" world", start=0.46, end=0.9, probability=0.95),
    ]
    fake_seg = SimpleNamespace(start=0.1, end=0.9, text=" Hello world", words=fake_words)
    fake_info = SimpleNamespace(language="en", duration=1.5)

    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([fake_seg], fake_info)

    transcriber = Transcriber(engine="faster-whisper", model_name="tiny")
    transcriber._local_model = fake_model

    result = transcriber.transcribe(wav, video_id="vid1")
    assert result.engine == "faster-whisper"
    assert result.language == "en"
    assert result.full_text == "Hello world"
    assert len(result.words) == 2
    assert result.words[0].word == "Hello"
    assert result.words[0].start == 0.1
    assert result.words[1].end == 0.9


def test_transcriber_groq_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    wav = tmp_path / "test_16k.wav"
    wav.write_bytes(b"RIFFfake")

    transcriber = Transcriber(engine="groq", groq_api_key=None)
    with pytest.raises(TranscriptionError, match="GROQ_API_KEY"):
        transcriber.transcribe(wav, video_id="vid1")
