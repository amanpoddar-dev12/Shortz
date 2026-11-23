"""Word-level transcription engine supporting faster-whisper (local/Colab GPU) and Groq Free API.

Produces millisecond-accurate WordTimestamp and TranscriptSegment structures
conforming to core.models.TranscriptResult.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from core.models import (
    TranscriptResult,
    TranscriptSegment,
    TranscriptionEngine,
    WordTimestamp,
)

logger = logging.getLogger(__name__)

GROQ_MAX_BYTES = 24 * 1024 * 1024  # 24 MB safety margin under Groq's 25 MB limit
GROQ_CHUNK_SECONDS = 600  # 10-minute chunks for very long audio


class TranscriptionError(RuntimeError):
    """Raised when transcription fails due to configuration, audio, or API errors."""


def _detect_device_and_compute() -> tuple[str, str]:
    """Detect CUDA availability via ctranslate2 and choose optimal compute type."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _probe_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds using ffprobe, or 0.0 if unavailable."""
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return 0.0
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            return float(proc.stdout.strip())
        except ValueError:
            return 0.0
    return 0.0


class Transcriber:
    """Unified word-level transcription interface for local GPU/CPU and Groq Free API."""

    def __init__(
        self,
        engine: TranscriptionEngine = "faster-whisper",
        model_name: str = "large-v3-turbo",
        groq_api_key: Optional[str] = None,
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self.engine: TranscriptionEngine = engine
        self.model_name = model_name
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.device = device
        self.compute_type = compute_type
        self._local_model: Any = None

    def transcribe(
        self,
        audio_path: str | Path,
        video_id: str,
        language: Optional[str] = None,
    ) -> TranscriptResult:
        """Transcribe a 16kHz mono WAV file into word-level timestamps.

        Args:
            audio_path: Path to the audio file (typically 16kHz mono WAV).
            video_id: Identifier of the source video.
            language: Optional ISO-639-1 language code (e.g. 'en'). Auto-detected if None.

        Returns:
            TranscriptResult with segments and flattened WordTimestamp list.
        """
        path = Path(audio_path)
        if not path.is_file():
            raise TranscriptionError(f"Audio file not found: {path}")

        if self.engine == "faster-whisper":
            return self._transcribe_faster_whisper(path, video_id, language)
        if self.engine == "groq":
            return self._transcribe_groq(path, video_id, language)
        if self.engine == "whisperx":
            return self._transcribe_whisperx(path, video_id, language)
        raise TranscriptionError(f"Unsupported transcription engine: {self.engine}")

    def _get_faster_whisper_model(self) -> Any:
        """Lazily load and cache the faster-whisper model."""
        if self._local_model is not None:
            return self._local_model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from exc

        device, compute_type = self.device, self.compute_type
        if device == "auto" or compute_type == "auto":
            auto_dev, auto_comp = _detect_device_and_compute()
            device = auto_dev if device == "auto" else device
            compute_type = auto_comp if compute_type == "auto" else compute_type

        logger.info(
            "Loading faster-whisper model '%s' on device=%s compute_type=%s",
            self.model_name,
            device,
            compute_type,
        )
        self._local_model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute_type,
        )
        return self._local_model

    def _transcribe_faster_whisper(
        self, audio_path: Path, video_id: str, language: Optional[str]
    ) -> TranscriptResult:
        """Run local/Colab GPU transcription via faster-whisper."""
        model = self._get_faster_whisper_model()
        logger.info("Transcribing %s with faster-whisper...", audio_path.name)

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        segments: list[TranscriptSegment] = []
        all_words: list[WordTimestamp] = []
        full_text_parts: list[str] = []

        for idx, seg in enumerate(segments_iter):
            seg_words: list[WordTimestamp] = []
            for w in getattr(seg, "words", None) or []:
                text = (w.word or "").strip()
                if not text:
                    continue
                start_t = max(0.0, round(float(w.start), 3))
                end_t = max(start_t, round(float(w.end), 3))
                prob = max(0.0, min(1.0, float(getattr(w, "probability", 1.0))))
                wt = WordTimestamp(
                    word=text,
                    start=start_t,
                    end=end_t,
                    probability=round(prob, 4),
                )
                seg_words.append(wt)
                all_words.append(wt)

            seg_text = (seg.text or "").strip()
            if seg_text:
                full_text_parts.append(seg_text)

            seg_start = max(0.0, round(float(seg.start), 3))
            seg_end = max(seg_start, round(float(seg.end), 3))
            segments.append(
                TranscriptSegment(
                    id=idx,
                    start=seg_start,
                    end=seg_end,
                    text=seg_text,
                    words=seg_words,
                )
            )

        duration = float(getattr(info, "duration", 0.0)) or _probe_audio_duration(audio_path)
        if duration <= 0.0:
            duration = all_words[-1].end if all_words else 0.001

        return TranscriptResult(
            video_id=video_id,
            language=str(getattr(info, "language", language or "en")),
            duration=round(duration, 3),
            engine="faster-whisper",
            full_text=" ".join(full_text_parts).strip(),
            segments=segments,
            words=all_words,
        )

    def _transcribe_groq(
        self, audio_path: Path, video_id: str, language: Optional[str]
    ) -> TranscriptResult:
        """Run ultra-fast cloud transcription via Groq Free API (whisper-large-v3-turbo)."""
        if not self.groq_api_key:
            raise TranscriptionError(
                "GROQ_API_KEY is not set. Add it to your .env file or pass groq_api_key."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise TranscriptionError("groq package is not installed. Run: pip install groq") from exc

        client = Groq(api_key=self.groq_api_key)
        groq_model = self.model_name if "whisper" in self.model_name else "whisper-large-v3-turbo"
        total_duration = _probe_audio_duration(audio_path)

        with tempfile.TemporaryDirectory(prefix="ovc_groq_") as tmp_dir:
            upload_chunks = self._prepare_groq_chunks(audio_path, Path(tmp_dir))

            segments: list[TranscriptSegment] = []
            all_words: list[WordTimestamp] = []
            full_text_parts: list[str] = []
            detected_lang = language or "en"
            seg_counter = 0

            for chunk_path, offset_sec in upload_chunks:
                logger.info(
                    "Uploading %s (offset=%.1fs) to Groq (%s)...",
                    chunk_path.name,
                    offset_sec,
                    groq_model,
                )
                with chunk_path.open("rb") as audio_file:
                    kwargs: dict[str, Any] = {
                        "file": (chunk_path.name, audio_file.read()),
                        "model": groq_model,
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["word", "segment"],
                    }
                    if language:
                        kwargs["language"] = language

                    resp = client.audio.transcriptions.create(**kwargs)

                raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                detected_lang = str(raw.get("language") or detected_lang)
                chunk_text = str(raw.get("text") or "").strip()
                if chunk_text:
                    full_text_parts.append(chunk_text)

                chunk_words: list[WordTimestamp] = []
                for w in raw.get("words") or []:
                    word_str = str(w.get("word") or "").strip()
                    if not word_str:
                        continue
                    w_start = max(0.0, round(float(w.get("start", 0.0)) + offset_sec, 3))
                    w_end = max(w_start, round(float(w.get("end", 0.0)) + offset_sec, 3))
                    wt = WordTimestamp(word=word_str, start=w_start, end=w_end)
                    chunk_words.append(wt)
                    all_words.append(wt)

                raw_segments = raw.get("segments") or []
                if raw_segments:
                    for s in raw_segments:
                        s_start = max(0.0, round(float(s.get("start", 0.0)) + offset_sec, 3))
                        s_end = max(s_start, round(float(s.get("end", 0.0)) + offset_sec, 3))
                        seg_words = [w for w in chunk_words if s_start <= w.start <= s_end]
                        segments.append(
                            TranscriptSegment(
                                id=seg_counter,
                                start=s_start,
                                end=s_end,
                                text=str(s.get("text") or "").strip(),
                                words=seg_words,
                            )
                        )
                        seg_counter += 1
                elif chunk_words:
                    segments.append(
                        TranscriptSegment(
                            id=seg_counter,
                            start=chunk_words[0].start,
                            end=chunk_words[-1].end,
                            text=chunk_text,
                            words=chunk_words,
                        )
                    )
                    seg_counter += 1

        duration = total_duration or (all_words[-1].end if all_words else 0.001)
        return TranscriptResult(
            video_id=video_id,
            language=detected_lang,
            duration=round(duration, 3),
            engine="groq",
            full_text=" ".join(full_text_parts).strip(),
            segments=segments,
            words=all_words,
        )

    def _prepare_groq_chunks(
        self, audio_path: Path, tmp_dir: Path
    ) -> list[tuple[Path, float]]:
        """Compress or chunk audio so every upload stays under Groq's 25 MB limit."""
        if audio_path.stat().st_size <= GROQ_MAX_BYTES:
            return [(audio_path, 0.0)]

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise TranscriptionError(
                f"Audio file ({audio_path.stat().st_size / 1e6:.1f} MB) exceeds Groq's 25 MB limit "
                "and FFmpeg is not available to compress or chunk it."
            )

        # Compress to 16kHz mono MP3 (48kbps) first — fits ~65 minutes in <24 MB
        compressed_path = tmp_dir / "compressed.mp3"
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(audio_path),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-b:a",
                "48k",
                str(compressed_path),
            ],
            capture_output=True,
            check=True,
        )
        if compressed_path.stat().st_size <= GROQ_MAX_BYTES:
            return [(compressed_path, 0.0)]

        # Still > 24 MB: split into fixed-duration chunks
        total_dur = _probe_audio_duration(compressed_path)
        chunks: list[tuple[Path, float]] = []
        offset = 0.0
        idx = 0
        while offset < total_dur:
            chunk_file = tmp_dir / f"chunk_{idx:03d}.mp3"
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-ss",
                    str(offset),
                    "-t",
                    str(GROQ_CHUNK_SECONDS),
                    "-i",
                    str(compressed_path),
                    "-c",
                    "copy",
                    str(chunk_file),
                ],
                capture_output=True,
                check=True,
            )
            if chunk_file.exists() and chunk_file.stat().st_size > 0:
                chunks.append((chunk_file, offset))
            offset += GROQ_CHUNK_SECONDS
            idx += 1
        return chunks

    def _transcribe_whisperx(
        self, audio_path: Path, video_id: str, language: Optional[str]
    ) -> TranscriptResult:
        """Run forced-alignment transcription via WhisperX (optional Colab GPU engine)."""
        try:
            import whisperx
        except ImportError as exc:
            raise TranscriptionError(
                "whisperx is not installed. Install it or use engine='faster-whisper' / 'groq'."
            ) from exc

        device, compute_type = _detect_device_and_compute()
        model = whisperx.load_model(self.model_name, device, compute_type=compute_type)
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=16, language=language)

        lang = result.get("language", language or "en")
        align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
        aligned = whisperx.align(
            result["segments"], align_model, metadata, audio, device, return_char_alignments=False
        )

        segments: list[TranscriptSegment] = []
        all_words: list[WordTimestamp] = []
        full_text_parts: list[str] = []

        for idx, seg in enumerate(aligned.get("segments", [])):
            seg_words: list[WordTimestamp] = []
            for w in seg.get("words", []):
                word_str = str(w.get("word") or "").strip()
                if not word_str or "start" not in w or "end" not in w:
                    continue
                w_start = max(0.0, round(float(w["start"]), 3))
                w_end = max(w_start, round(float(w["end"]), 3))
                wt = WordTimestamp(
                    word=word_str,
                    start=w_start,
                    end=w_end,
                    probability=round(float(w.get("score", 1.0)), 4),
                )
                seg_words.append(wt)
                all_words.append(wt)

            seg_text = str(seg.get("text") or "").strip()
            if seg_text:
                full_text_parts.append(seg_text)
            segments.append(
                TranscriptSegment(
                    id=idx,
                    start=max(0.0, round(float(seg.get("start", 0.0)), 3)),
                    end=max(0.0, round(float(seg.get("end", 0.0)), 3)),
                    text=seg_text,
                    words=seg_words,
                )
            )

        duration = _probe_audio_duration(audio_path) or (all_words[-1].end if all_words else 0.001)
        return TranscriptResult(
            video_id=video_id,
            language=lang,
            duration=round(duration, 3),
            engine="whisperx",
            full_text=" ".join(full_text_parts).strip(),
            segments=segments,
            words=all_words,
        )
