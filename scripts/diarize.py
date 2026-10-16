#!/usr/bin/env python3
"""Offline speaker diarization for 16 kHz mono WAV files using sherpa-onnx."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "work" / "models" / "diarization"
SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMBEDDING_MODEL = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
SAMPLE_RATE = 16_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="16 kHz mono PCM WAV input")
    parser.add_argument("--output", required=True, type=Path, help="JSON output path")
    parser.add_argument("--track-name", help="Human-readable track label for output metadata")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=-1,
        help="Known speaker count; -1 (default) estimates the count from clustering",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=6,
        help=(
            "For automatic counting, rerun with this count if the first pass exceeds it "
            "(default: 6)"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Automatic clustering threshold; lower values tend to create more speakers",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="CPU inference threads (default: 4)"
    )
    parser.add_argument(
        "--window-shift-ratio",
        type=float,
        default=0.1,
        help="Segmentation window shift ratio; smaller improves temporal coverage at added cost",
    )
    parser.add_argument(
        "--block-minutes",
        type=float,
        help=(
            "Optional memory-bounded mode. Diarize blocks independently; speaker IDs are local "
            "to each block and are not globally reconciled. Default processes the full recording."
        ),
    )
    return parser


def model_paths(model_dir: Path) -> tuple[Path, Path]:
    segmentation = model_dir / SEGMENTATION_MODEL
    embedding = model_dir / EMBEDDING_MODEL
    missing = [path for path in (segmentation, embedding) if not path.is_file()]
    if missing:
        urls = {
            segmentation: "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
            embedding: "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        }
        details = "\n".join(f"  {path}\n    Download: {urls[path]}" for path in missing)
        raise FileNotFoundError(f"Required diarization model files are missing:\n{details}")
    return segmentation, embedding


def make_config(
    segmentation: Path,
    embedding: Path,
    *,
    num_speakers: int,
    threshold: float,
    num_threads: int,
    window_shift_ratio: float,
) -> sherpa_onnx.OfflineSpeakerDiarizationConfig:
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation), window_shift_ratio=window_shift_ratio
            ),
            num_threads=num_threads,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding), num_threads=num_threads, provider="cpu"
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError("Invalid sherpa-onnx diarization config; check model files and options")
    return config


def audio_info(path: Path) -> tuple[int, int, int]:
    info = sf.info(path)
    if info.channels != 1:
        raise ValueError(f"Expected mono input, got {info.channels} channels: {path}")
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"Expected 16000 Hz input, got {info.samplerate} Hz: {path}")
    if info.subtype not in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}:
        raise ValueError(f"Unsupported WAV subtype {info.subtype!r}: {path}")
    return info.samplerate, info.frames, info.channels


def read_audio(path: Path, start: int = 0, frames: int = -1) -> np.ndarray:
    with sf.SoundFile(path) as source:
        source.seek(start)
        samples = source.read(frames=frames, dtype="float32", always_2d=True)
    return np.ascontiguousarray(samples[:, 0])


def segments_to_turns(segments: Any, *, block_index: int | None = None) -> list[dict[str, Any]]:
    turns = []
    for segment in segments:
        speaker = int(segment.speaker)
        turn: dict[str, Any] = {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "speaker_id": speaker,
            "speaker": (
                f"speaker_{speaker:02d}"
                if block_index is None
                else f"block_{block_index:03d}:speaker_{speaker:02d}"
            ),
        }
        if block_index is not None:
            turn["block_index"] = block_index
        turns.append(turn)
    return turns


def process_with_progress(
    diarizer: sherpa_onnx.OfflineSpeakerDiarization, audio: np.ndarray
) -> Any:
    last_percent = -5

    def progress(processed: int, total: int) -> int:
        nonlocal last_percent
        percent = int(processed * 100 / max(total, 1))
        if percent >= last_percent + 5 or percent == 100:
            print(f"Diarization: {percent}%", file=sys.stderr, flush=True)
            last_percent = percent
        return 0

    return diarizer.process(audio, callback=progress).sort_by_start_time()


def run(args: argparse.Namespace) -> dict[str, Any]:
    segmentation, embedding = model_paths(args.model_dir)
    sample_rate, frame_count, _ = audio_info(args.input)
    duration = frame_count / sample_rate
    track_name = args.track_name or args.input.stem

    base_config = make_config(
        segmentation,
        embedding,
        num_speakers=args.num_speakers,
        threshold=args.threshold,
        num_threads=args.num_threads,
        window_shift_ratio=args.window_shift_ratio,
    )
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(base_config)
    if diarizer.sample_rate != sample_rate:
        raise ValueError(
            f"Models expect {diarizer.sample_rate} Hz audio, but input is {sample_rate} Hz"
        )

    common_metadata = {
        "track": track_name,
        "input": str(args.input),
        "duration_seconds": round(duration, 3),
        "sample_rate": sample_rate,
        "models": {
            "segmentation": SEGMENTATION_MODEL,
            "embedding": EMBEDDING_MODEL,
        },
        "threshold": args.threshold,
        "num_threads": args.num_threads,
        "window_shift_ratio": args.window_shift_ratio,
    }

    if args.block_minutes is not None:
        if args.block_minutes <= 0:
            raise ValueError("--block-minutes must be positive")
        block_frames = int(args.block_minutes * 60 * sample_rate)
        if block_frames < sample_rate:
            raise ValueError("--block-minutes must be at least 1/60 minute (1 second)")
        block_turns = []
        speaker_counts: dict[str, int] = {}
        speaker_counts_before_cap: dict[str, int] = {}
        block_index = 0
        for start_frame in range(0, frame_count, block_frames):
            audio = read_audio(
                args.input, start_frame, min(block_frames, frame_count - start_frame)
            )
            result = process_with_progress(diarizer, audio)
            offset_seconds = start_frame / sample_rate
            turns = segments_to_turns(result, block_index=block_index)
            first_pass_count = len({turn["speaker_id"] for turn in turns})
            speaker_counts_before_cap[str(block_index)] = first_pass_count
            if args.num_speakers == -1 and first_pass_count > args.max_speakers:
                base_config.clustering.num_clusters = args.max_speakers
                if not base_config.validate():
                    raise ValueError("Invalid clustering config when applying --max-speakers")
                diarizer.set_config(base_config)
                result = process_with_progress(diarizer, audio)
                turns = segments_to_turns(result, block_index=block_index)
                base_config.clustering.num_clusters = -1
                diarizer.set_config(base_config)
            for turn in turns:
                turn["start"] = round(turn["start"] + offset_seconds, 3)
                turn["end"] = round(turn["end"] + offset_seconds, 3)
            speaker_counts[str(block_index)] = len({t["speaker_id"] for t in turns})
            block_turns.extend(turns)
            block_index += 1
        return {
            "metadata": {
                **common_metadata,
                "speaker_scope": "block_local",
                "speaker_count": None,
                "speaker_counts_by_block": speaker_counts,
                "auto_speaker_counts_before_cap_by_block": speaker_counts_before_cap,
                "block_minutes": args.block_minutes,
                "block_count": block_index,
                "num_speakers_requested": args.num_speakers,
                "max_speakers": args.max_speakers,
            },
            "turns": block_turns,
        }

    # The whole waveform is intentionally processed as one recording so clustering
    # has a single speaker space. Float32 audio costs about 3.2 GiB for 15 hours;
    # use --block-minutes when host memory is insufficient, accepting local IDs.
    audio = read_audio(args.input)
    result = process_with_progress(diarizer, audio)
    turns = segments_to_turns(result)
    first_pass_count = len({turn["speaker_id"] for turn in turns})
    rerun_with_cap = args.num_speakers == -1 and first_pass_count > args.max_speakers
    if rerun_with_cap:
        base_config.clustering.num_clusters = args.max_speakers
        if not base_config.validate():
            raise ValueError("Invalid clustering config when applying --max-speakers")
        diarizer.set_config(base_config)
        result = process_with_progress(diarizer, audio)
        turns = segments_to_turns(result)

    return {
        "metadata": {
            **common_metadata,
            "speaker_scope": "recording_global",
            "speaker_count": len({turn["speaker_id"] for turn in turns}),
            "num_speakers_requested": args.num_speakers,
            "auto_speaker_count_before_cap": first_pass_count,
            "max_speakers": args.max_speakers,
            "rerun_with_max_speakers": rerun_with_cap,
        },
        "turns": turns,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.num_speakers == 0 or args.num_speakers < -1:
        print("--num-speakers must be -1 (auto) or a positive integer", file=sys.stderr)
        return 2
    if args.max_speakers < 1:
        print("--max-speakers must be positive", file=sys.stderr)
        return 2
    if args.num_threads < 1:
        print("--num-threads must be positive", file=sys.stderr)
        return 2
    if not 0 < args.window_shift_ratio <= 1:
        print("--window-shift-ratio must be greater than 0 and at most 1", file=sys.stderr)
        return 2
    try:
        payload = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output.with_name(args.output.name + ".tmp")
        temporary_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_output, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"diarize.py: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote {len(payload['turns'])} turns to {args.output} "
        f"(speaker scope: {payload['metadata']['speaker_scope']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
