"""Prepare full or ASR-sparse audio for diarization and restore its timeline."""

from __future__ import annotations

import copy
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000
SPARSE_COVERAGE_THRESHOLD = 0.25
REGION_PADDING_SECONDS = 0.3
REGION_SEPARATOR_SECONDS = 0.5
STREAM_FRAMES = SAMPLE_RATE * 30


def merge_speech_regions(
    raw: dict[str, Any], duration: float, padding: float = REGION_PADDING_SECONDS
) -> list[tuple[float, float]]:
    """Collect padded word spans (or segment spans without words) and merge them."""
    spans: list[tuple[float, float]] = []
    for segment in raw.get("segments", []):
        words = segment.get("words") or []
        candidates = words if words else [segment]
        for item in candidates:
            try:
                start = max(0.0, float(item["start"]) - padding)
                end = min(duration, float(item["end"]) + padding)
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                spans.append((start, end))
    spans.sort()
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _gain_schedule(raw: dict[str, Any]) -> list[tuple[float, float, float]]:
    schedule = []
    for chunk in raw.get("chunks", []):
        try:
            start = float(chunk["offset"])
            end = start + float(chunk["duration"])
            gain_db = float(chunk.get("gain_db", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if end > start and math.isfinite(gain_db):
            schedule.append((start, end, 10 ** (gain_db / 20)))
    return sorted(schedule)


def _gain_at(time_seconds: float, schedule: list[tuple[float, float, float]]) -> float:
    for start, end, gain in schedule:
        if start <= time_seconds < end:
            return gain
    return 1.0


def _write_interval(
    source: sf.SoundFile,
    target: sf.SoundFile,
    start: float,
    end: float,
    schedule: list[tuple[float, float, float]],
) -> int:
    """Stream a source-time interval with its recorded ASR gain into target."""
    rate = source.samplerate
    begin_frame = max(0, min(source.frames, round(start * rate)))
    end_frame = max(begin_frame, min(source.frames, round(end * rate)))
    source.seek(begin_frame)
    frames_written = 0
    while source.tell() < end_frame:
        current = source.tell()
        current_time = current / rate
        gain = _gain_at(current_time, schedule)
        next_boundary = end_frame
        for chunk_start, chunk_end, _ in schedule:
            boundary = round(chunk_end * rate)
            if current < boundary < next_boundary:
                next_boundary = boundary
            if current < round(chunk_start * rate) < next_boundary:
                next_boundary = round(chunk_start * rate)
        count = min(STREAM_FRAMES, next_boundary - current)
        samples = source.read(count, dtype="float32", always_2d=True)
        if not len(samples):
            break
        if gain != 1.0:
            samples *= gain
            np.clip(samples, -1.0, 1.0, out=samples)
        target.write(samples)
        frames_written += len(samples)
    return frames_written


def prepare_diarization_input(
    source_audio: str | Path,
    raw: dict[str, Any],
    output_wav: str | Path,
) -> dict[str, Any]:
    """Decode to 16 kHz mono PCM16, apply ASR gains, and compact sparse tracks."""
    source_audio = Path(source_audio)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    schedule = _gain_schedule(raw)

    with tempfile.TemporaryDirectory(prefix="diarization-audio-", dir=output_wav.parent) as temp_dir:
        decoded = Path(temp_dir) / "decoded.wav"
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source_audio),
            "-map_metadata", "-1", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le", "-fflags", "+bitexact", "-flags:a", "+bitexact",
            str(decoded),
        ]
        subprocess.run(command, check=True, capture_output=True)
        with sf.SoundFile(decoded) as audio:
            if audio.channels != 1 or audio.samplerate != SAMPLE_RATE:
                raise RuntimeError("ffmpeg did not produce mono 16 kHz audio")
            duration = audio.frames / SAMPLE_RATE
            regions = merge_speech_regions(raw, duration)
            coverage = sum(end - start for start, end in regions) / duration if duration else 0.0
            sparse = coverage < SPARSE_COVERAGE_THRESHOLD
            mode = "sparse_asr_compact" if sparse else "full_recording"
            region_map: list[dict[str, float]] = []
            with sf.SoundFile(
                output_wav,
                mode="w",
                samplerate=SAMPLE_RATE,
                channels=1,
                subtype="PCM_16",
                format="WAV",
            ) as output:
                if sparse:
                    compact_time = 0.0
                    for index, (start, end) in enumerate(regions):
                        if index:
                            silence = np.zeros(
                                (round(REGION_SEPARATOR_SECONDS * SAMPLE_RATE), 1), dtype=np.float32
                            )
                            output.write(silence)
                            compact_time += len(silence) / SAMPLE_RATE
                        compact_start = compact_time
                        frames_written = _write_interval(audio, output, start, end, schedule)
                        compact_time += frames_written / SAMPLE_RATE
                        region_map.append({
                            "compact_start": compact_start,
                            "compact_end": compact_time,
                            "original_start": start,
                            "original_end": end,
                        })
                else:
                    _write_interval(audio, output, 0.0, duration, schedule)

    output_duration = sf.info(output_wav).duration
    return {
        "mode": mode,
        "source_audio": str(source_audio),
        "coverage": coverage,
        "regions": region_map,
        "original_duration_seconds": duration,
        "prepared_duration_seconds": output_duration,
        "sample_rate": SAMPLE_RATE,
        "gain_applied": any(gain != 1.0 for _, _, gain in schedule),
    }


def clip_turns_to_duration(diarization: dict[str, Any], duration: float) -> dict[str, Any]:
    """Clip already-restored turn lists without remapping sparse recording times."""
    result = copy.deepcopy(diarization)
    duration = float(duration)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Original audio duration must be finite and nonnegative")
    for key in ("turns", "overlapping_turns"):
        if key not in result:
            continue
        clipped = []
        for turn in result[key]:
            try:
                start, end = float(turn["start"]), float(turn["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            turn["start"] = min(duration, max(0.0, start))
            turn["end"] = min(duration, max(0.0, end))
            if turn["end"] > turn["start"]:
                clipped.append(turn)
        result[key] = clipped
    return result


def restore_turns(
    diarization: dict[str, Any], preprocessing: dict[str, Any]
) -> dict[str, Any]:
    """Map compact diarization turns back onto the original recording timeline."""
    result = copy.deepcopy(diarization)
    def restore_list(turns):
        if preprocessing.get("mode") != "sparse_asr_compact":
            return copy.deepcopy(turns)
        restored = []
        for turn in turns:
            try:
                start, end = float(turn["start"]), float(turn["end"])
            except (KeyError, TypeError, ValueError):
                continue
            for region in preprocessing.get("regions", []):
                left = max(start, float(region["compact_start"]))
                right = min(end, float(region["compact_end"]))
                if right <= left:
                    continue
                mapped = copy.deepcopy(turn)
                offset = float(region["original_start"]) - float(region["compact_start"])
                mapped["start"] = round(left + offset, 3)
                mapped["end"] = round(right + offset, 3)
                if mapped["end"] > mapped["start"]:
                    restored.append(mapped)
        return sorted(restored, key=lambda turn: (turn["start"], turn["end"]))

    for key in ("turns", "overlapping_turns"):
        if key in diarization:
            result[key] = restore_list(diarization[key])

    metadata = dict(diarization.get("metadata", {}))
    prepared_input = metadata.get("input")
    if preprocessing.get("source_audio"):
        metadata["input"] = preprocessing["source_audio"]
    if prepared_input and prepared_input != metadata.get("input"):
        metadata["diarization_input"] = prepared_input
    metadata["input_duration_seconds"] = preprocessing.get("original_duration_seconds")
    metadata["duration_seconds"] = preprocessing.get("original_duration_seconds")
    metadata["preprocessing"] = copy.deepcopy(preprocessing)
    metadata["speaker_scope"] = "recording_global"
    result["metadata"] = metadata
    duration = preprocessing.get("original_duration_seconds")
    return clip_turns_to_duration(result, duration) if duration is not None else result
