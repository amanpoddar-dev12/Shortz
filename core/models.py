"""Core data contracts and Pydantic schemas for OpenViralClip.

Defines all inter-module data structures passed between ingestion, transcription,
viral LLM analysis, smart reframing, subtitle generation, rendering, and YouTube upload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

DurationBucket = Literal[30, 60, 90]
AspectRatio = Literal["9:16", "1:1", "16:9"]
CropMode = Literal["single_speaker", "split_screen", "center_fallback"]
PrivacyStatus = Literal["private", "unlisted", "public"]
TranscriptionEngine = Literal["faster-whisper", "whisperx", "groq"]
LLMProvider = Literal["gemini", "ollama"]


class HeatmapPoint(BaseModel):
    """Represents a single segment of YouTube's 'Most Replayed' heatmap."""

    start_time: float = Field(..., ge=0.0, description="Segment start in seconds.")
    end_time: float = Field(..., gt=0.0, description="Segment end in seconds.")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Normalized replay intensity [0.0, 1.0].")

    @model_validator(mode="after")
    def validate_times(self) -> HeatmapPoint:
        if self.end_time <= self.start_time:
            raise ValueError("HeatmapPoint end_time must be greater than start_time.")
        return self


class IngestionResult(BaseModel):
    """Output contract of core.downloader."""

    video_path: str = Field(..., min_length=1)
    audio_wav_path: str = Field(..., min_length=1)
    heatmap_points: list[HeatmapPoint] = Field(default_factory=list)
    video_id: str = Field(..., min_length=1)
    duration: float = Field(..., gt=0.0, description="Total video duration in seconds.")
    title: str = "Untitled"
    channel: str = "Unknown"
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    fps: float = Field(default=0.0, ge=0.0)
    source_url: str = ""


# Backward-compatible alias for downstream modules
VideoMetadata = IngestionResult


class WordTimestamp(BaseModel):
    """Millisecond-accurate word-level timestamp."""

    word: str = Field(..., min_length=1)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    speaker: Optional[str] = None

    @model_validator(mode="after")
    def validate_times(self) -> WordTimestamp:
        if self.end < self.start:
            raise ValueError(f"WordTimestamp end ({self.end}) < start ({self.start}) for '{self.word}'.")
        return self


class TranscriptSegment(BaseModel):
    """Sentence or phrase-level transcript segment containing word timestamps."""

    id: int = Field(..., ge=0)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)
    speaker: Optional[str] = None


class TranscriptResult(BaseModel):
    """Output contract of core.transcriber."""

    video_id: str
    language: str = "en"
    duration: float = Field(..., gt=0.0)
    engine: TranscriptionEngine
    full_text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    words: list[WordTimestamp] = Field(default_factory=list)


class ViralClipCandidate(BaseModel):
    """Represents a single AI-identified viral clip candidate."""

    clip_id: str = Field(..., min_length=1, description="Unique identifier, e.g. 'clip_01_60s'.")
    target_duration_bucket: DurationBucket = Field(
        ..., description="Target duration bucket in seconds (30, 60, or 90)."
    )
    virality_score: int = Field(..., ge=1, le=100, description="Predicted virality score (1-100).")
    hook_reason: str = Field(..., min_length=5, description="Why the opening hook and payoff retain viewers.")
    start_time: float = Field(..., ge=0.0, description="Word-snapped start timestamp in seconds.")
    end_time: float = Field(..., gt=0.0, description="Word-snapped end timestamp in seconds.")
    transcript_excerpt: str = Field(..., min_length=1)
    youtube_title: str = Field(..., min_length=3, max_length=100)
    youtube_description: str = Field(default="")
    hashtags: list[str] = Field(default_factory=list)
    thumbnail_hook_text: str = Field(
        ..., min_length=2, max_length=60, description="Short 2-5 word punchy text overlay for thumbnail."
    )
    heatmap_overlap_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average YouTube Most Replayed intensity over this window."
    )

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 3)

    @model_validator(mode="after")
    def validate_window(self) -> ViralClipCandidate:
        if self.end_time <= self.start_time:
            raise ValueError("ViralClipCandidate end_time must be greater than start_time.")
        return self


class ViralAnalysisResponse(BaseModel):
    """Structured output schema enforced on Gemini / Ollama responses."""

    candidates: list[ViralClipCandidate] = Field(default_factory=list)


class CropKeyframe(BaseModel):
    """Single smoothed virtual-camera crop rectangle at a given frame/timestamp."""

    frame_idx: int = Field(..., ge=0)
    timestamp: float = Field(..., ge=0.0)
    crop_x: int = Field(..., ge=0)
    crop_y: int = Field(..., ge=0)
    crop_w: int = Field(..., gt=0)
    crop_h: int = Field(..., gt=0)
    mode: CropMode = "single_speaker"
    secondary_crop_x: Optional[int] = Field(default=None, ge=0)
    secondary_crop_y: Optional[int] = Field(default=None, ge=0)


class CropTrajectory(BaseModel):
    """Output contract of core.cropper for a single clip and target aspect ratio."""

    clip_id: str
    aspect_ratio: AspectRatio
    source_width: int = Field(..., gt=0)
    source_height: int = Field(..., gt=0)
    target_width: int = Field(..., gt=0)
    target_height: int = Field(..., gt=0)
    fps: float = Field(..., gt=0.0)
    mode: CropMode
    keyframes: list[CropKeyframe] = Field(..., min_length=1)


class RenderedClip(BaseModel):
    """Output contract of core.renderer + core.thumbnail for a finished Short."""

    clip_id: str
    aspect_ratio: AspectRatio
    duration_bucket: DurationBucket
    video_path: Path
    subtitle_path: Path
    thumbnail_path: Path
    candidate: ViralClipCandidate
    file_size_bytes: int = Field(default=0, ge=0)
    render_time_seconds: float = Field(default=0.0, ge=0.0)


class UploadResult(BaseModel):
    """Output contract of uploader.youtube_agent."""

    clip_id: str
    youtube_video_id: str
    youtube_url: str
    privacy_status: PrivacyStatus
    thumbnail_uploaded: bool = False


class SubtitleStyleConfig(BaseModel):
    """Styling configuration for ASS word-by-word animated subtitles."""

    font_name: str = "Montserrat"
    font_size: int = Field(default=68, ge=16, le=200)
    primary_color: str = "&H00FFFFFF"
    highlight_color: str = "&H0000FFFF"
    outline_color: str = "&H00000000"
    outline_width: int = Field(default=4, ge=0, le=12)
    shadow_depth: int = Field(default=2, ge=0, le=10)
    words_per_line: int = Field(default=3, ge=1, le=8)
    uppercase: bool = True
    alignment: int = Field(default=2, ge=1, le=9)
    margin_v: int = Field(default=260, ge=0, le=800)


class AppConfig(BaseModel):
    """Root application configuration loaded from config.yaml."""

    clip_durations: list[DurationBucket] = Field(default_factory=lambda: [30, 60, 90])
    aspect_ratios: list[AspectRatio] = Field(default_factory=lambda: ["9:16", "1:1", "16:9"])
    max_clips_per_bucket: int = Field(default=2, ge=1, le=10)
    transcription_engine: TranscriptionEngine = "faster-whisper"
    whisper_model: str = "large-v3-turbo"
    llm_provider: LLMProvider = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    ollama_model: str = "qwen2.5:14b"
    ollama_base_url: str = "http://localhost:11434"
    crop_smoothing_window: int = Field(default=15, ge=1, le=120)
    enable_split_screen: bool = True
    ffmpeg_crf: int = Field(default=18, ge=0, le=51)
    ffmpeg_preset: str = "medium"
    youtube_privacy_status: PrivacyStatus = "private"
    youtube_category_id: str = "22"
    subtitles: SubtitleStyleConfig = Field(default_factory=SubtitleStyleConfig)
