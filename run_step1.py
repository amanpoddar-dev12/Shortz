"""CLI runner for testing Step 1 (Video Ingestion + Word-Level Transcription).

Usage:
    python run_step1.py --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --engine faster-whisper
    python run_step1.py --source path/to/local.mp4 --engine groq
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from core.downloader import YouTubeDownloader
from core.transcriber import Transcriber


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Test OpenViralClip Step 1 (Download + Transcribe)")
    parser.add_argument("--source", required=True, help="YouTube URL or local .mp4 file path")
    parser.add_argument(
        "--output-dir",
        default="downloads",
        help="Workspace output directory (default: downloads)",
    )
    parser.add_argument(
        "--engine",
        choices=["faster-whisper", "groq", "whisperx"],
        default="faster-whisper",
        help="Transcription engine",
    )
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper model name (e.g. tiny, base, large-v3-turbo)",
    )
    args = parser.parse_args()

    downloader = YouTubeDownloader(output_dir=args.output_dir)
    ingestion = downloader.download_and_extract(args.source)
    print("\n=== INGESTION RESULT ===")
    print(f"Video ID   : {ingestion.video_id}")
    print(f"Duration   : {ingestion.duration:.2f}s")
    print(f"Video Path : {ingestion.video_path}")
    print(f"Audio WAV  : {ingestion.audio_wav_path}")
    print(f"Heatmap Pts: {len(ingestion.heatmap_points)}")

    transcriber = Transcriber(engine=args.engine, model_name=args.model)
    transcript = transcriber.transcribe(ingestion.audio_wav_path, video_id=ingestion.video_id)

    out_json = Path(args.output_dir) / ingestion.video_id / f"{ingestion.video_id}_transcript.json"
    out_json.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    print("\n=== TRANSCRIPTION RESULT ===")
    print(f"Engine     : {transcript.engine}")
    print(f"Language   : {transcript.language}")
    print(f"Words      : {len(transcript.words)}")
    print(f"Segments   : {len(transcript.segments)}")
    print(f"Saved JSON : {out_json}")
    if transcript.words:
        sample = [f"{w.word} [{w.start:.2f}-{w.end:.2f}]" for w in transcript.words[:10]]
        print(f"First 10w  : {' | '.join(sample)}")


if __name__ == "__main__":
    main()
