#!/usr/bin/env python3
"""Assign diarization speakers to ASR words and render labeled transcripts."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any

try:  # Support both direct script execution and ``import scripts.label_transcripts``.
    from .transcribe import atomic_json, render
except ImportError:  # pragma: no cover - exercised by the CLI entry point
    from transcribe import atomic_json, render


CONFIDENCE_FIELDS = (
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
    "temperature",
    "review_recommended",
)
MAX_GROUP_GAP_SECONDS = 1.5
MAX_GROUP_DURATION_SECONDS = 20.0
SENTENCE_BREAK_MIN_SECONDS = 10.0


class TurnIndex:
    """Overlap index: O(log turns + overlapping turns) queries for sorted data."""

    def __init__(self, turns: list[dict[str, Any]]):
        valid = []
        for turn in turns:
            try:
                start, end = float(turn["start"]), float(turn["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                valid.append((start, end, _speaker_label(turn)))
        valid.sort(key=lambda item: (item[0], item[1], item[2]))
        self.turns = valid
        self.starts = [item[0] for item in valid]
        size = len(valid)
        self.tree_size = 1
        while self.tree_size < size:
            self.tree_size *= 2
        self.max_ends = [float("-inf")] * (2 * self.tree_size)
        for i, (_, end, _) in enumerate(valid):
            self.max_ends[self.tree_size + i] = end
        for i in range(self.tree_size - 1, 0, -1):
            self.max_ends[i] = max(self.max_ends[2 * i], self.max_ends[2 * i + 1])

    def overlaps(self, start: float, end: float) -> list[tuple[float, float, str]]:
        if end <= start or not self.turns:
            return []
        # Any turn beginning at/after the word's end cannot intersect it.
        limit = bisect.bisect_left(self.starts, end)
        found: list[tuple[float, float, str]] = []

        def visit(node: int, left: int, right: int) -> None:
            if left >= limit or self.max_ends[node] <= start:
                return
            if right - left == 1:
                if left < len(self.turns):
                    turn = self.turns[left]
                    found.append(turn)
                return
            middle = (left + right) // 2
            visit(node * 2, left, middle)
            if middle < limit:
                visit(node * 2 + 1, middle, right)

        visit(1, 0, self.tree_size)
        return found


def _speaker_label(turn: dict[str, Any]) -> str:
    value = turn.get("speaker")
    if value is None:
        value = turn.get("speaker_id")
    return str(value) if value is not None else "unknown"


def overlap_speakers(
    start: float, end: float, turns: list[dict[str, Any]], index: TurnIndex | None = None
) -> list[str]:
    """Return all overlapping speakers, ordered by their total covered duration."""
    return speaker_overlap_details(start, end, turns, index)[0]


def speaker_overlap_details(
    start: float, end: float, turns: list[dict[str, Any]], index: TurnIndex | None = None
) -> tuple[list[str], bool]:
    """Return ordered speaker coverage and whether distinct turns overlap in time."""
    if end <= start:
        # Point timestamps count only when strictly inside a turn.
        point_index = index or TurnIndex(turns)
        speakers = sorted({speaker for a, b, speaker in point_index.overlaps(start, start + 1e-12)
                           if a <= start < b})
        return speakers, len(speakers) > 1
    turn_index = index or TurnIndex(turns)
    matches = turn_index.overlaps(start, end)
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    events: list[tuple[float, int, str]] = []
    for turn_start, turn_end, speaker in matches:
        clipped = (max(start, turn_start), min(end, turn_end))
        by_speaker.setdefault(speaker, []).append(clipped)
        events.append((clipped[0], 1, speaker))
        events.append((clipped[1], -1, speaker))
    coverage = {}
    for speaker, intervals in by_speaker.items():
        intervals.sort()
        total = 0.0
        left, right = intervals[0]
        for next_left, next_right in intervals[1:]:
            if next_left <= right:
                right = max(right, next_right)
            else:
                total += right - left
                left, right = next_left, next_right
        coverage[speaker] = total + right - left
    active: dict[str, int] = {}
    concurrent_speakers = False
    for _, delta, speaker in sorted(events, key=lambda event: (event[0], event[1])):
        if delta < 0:
            active[speaker] -= 1
            if active[speaker] <= 0:
                del active[speaker]
        else:
            active[speaker] = active.get(speaker, 0) + 1
            if len(active) > 1:
                concurrent_speakers = True
                break
    return sorted(coverage, key=lambda speaker: (-coverage[speaker], speaker)), concurrent_speakers


def _label_fields(
    start: float, end: float, turns: list[dict[str, Any]], index: TurnIndex
) -> dict[str, Any]:
    speakers, concurrent = speaker_overlap_details(start, end, turns, index)
    return {
        "speaker": speakers[0] if speakers else "unknown",
        "speakers": speakers if speakers else ["unknown"],
        "overlapping_speech": concurrent,
    }


def _segment_confidence(segment: dict[str, Any]) -> dict[str, Any]:
    return {key: segment[key] for key in CONFIDENCE_FIELDS if key in segment}


def _word_text(words: list[dict[str, Any]], fallback: str = "") -> str:
    rendered = "".join(str(word.get("word", "")) for word in words).strip()
    return rendered or fallback.strip()


def _make_group(words: list[dict[str, Any]], sources: list[dict[str, Any]]) -> dict[str, Any]:
    # Preserve source order while recording each original segment only once.
    unique_sources: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for source in sources:
        identity = source.get("id", (source["start"], source["end"]))
        marker = (type(identity).__name__, repr(identity))
        if marker not in seen:
            seen.add(marker)
            unique_sources.append(source)
    speaker = words[0].get("speaker", "unknown")
    labels = list(dict.fromkeys(
        label for word in words for label in (word.get("speakers") or [word.get("speaker", "unknown")])
    ))
    result: dict[str, Any] = {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": _word_text(words),
        "speaker": speaker,
        "words": words,
        "speakers": labels,
        "overlapping_speech": any(word.get("overlapping_speech", False) for word in words),
        "source_segments": unique_sources,
        "review_recommended": any(s.get("review_recommended", False) for s in unique_sources),
    }
    if len(labels) > 1:
        result["speakers"] = labels
    # Keep the original ASR confidence fields available for straightforward review.
    for key in CONFIDENCE_FIELDS:
        if key == "review_recommended":
            continue
        values = [s[key] for s in unique_sources if key in s]
        if values:
            result[key] = values[0] if len(values) == 1 else values
    return result


def label_result(asr: dict[str, Any], diarization: dict[str, Any]) -> dict[str, Any]:
    """Return ASR data with word-level speakers and coherent grouped utterances."""
    turns = diarization.get("turns", [])
    if not isinstance(turns, list):
        raise ValueError("diarization JSON must contain a 'turns' list")

    turn_index = TurnIndex(turns)
    overlap_turns = diarization.get("overlapping_turns")
    overlap_index = TurnIndex(overlap_turns) if isinstance(overlap_turns, list) else None

    def labels(start, end):
        fields = _label_fields(start, end, turns, turn_index)
        if overlap_index is not None:
            # Exclusive turns choose the word owner; ordinary turns retain overlap evidence.
            speakers, concurrent = speaker_overlap_details(start, end, overlap_turns, overlap_index)
            primary = fields["speaker"]
            fields["speakers"] = list(dict.fromkeys(
                ([primary] if primary != "unknown" else []) + speakers)) or ["unknown"]
            fields["overlapping_speech"] = concurrent
        return fields

    # Flatten timestamped words, retaining original segment confidence for review.
    timed_words: list[tuple[dict[str, Any], dict[str, Any]]] = []
    atomic_segments: list[dict[str, Any]] = []
    segments = asr.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("ASR JSON must contain a 'segments' list")
    for index, segment in enumerate(segments):
        source = {"id": segment.get("id", index), "start": segment["start"], "end": segment["end"]}
        source.update(_segment_confidence(segment))
        words = segment.get("words") or []
        if not words:
            item = dict(segment)
            item.update(labels(float(segment["start"]), float(segment["end"])))
            item["source_segments"] = [source]
            atomic_segments.append(item)
            continue
        for word in words:
            item = dict(word)
            start = float(item.get("start", segment["start"]))
            end = float(item.get("end", segment["end"]))
            item.setdefault("start", start)
            item.setdefault("end", end)
            item.update(labels(start, end))
            timed_words.append((item, source))

    groups: list[dict[str, Any]] = []
    current_words: list[dict[str, Any]] = []
    current_sources: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current_words, current_sources
        if current_words:
            groups.append(_make_group(current_words, current_sources))
        current_words, current_sources = [], []

    events: list[tuple[float, float, str, Any]] = [
        (float(word["start"]), float(word["end"]), "word", (word, source))
        for word, source in timed_words
    ]
    events.extend(
        (float(segment["start"]), float(segment["end"]), "atomic", segment)
        for segment in atomic_segments
    )
    events.sort(key=lambda event: (event[0], event[1], event[2]))
    for _, _, kind, payload in events:
        if kind == "atomic":
            flush()
            groups.append(payload)
            continue
        word, source = payload
        key = word.get("speaker", "unknown")
        if current_words:
            previous = current_words[-1]
            previous_key = previous.get("speaker", "unknown")
            gap = float(word["start"]) - float(previous["end"])
            group_duration = float(previous["end"]) - float(current_words[0]["start"])
            previous_text = str(previous.get("word", "")).rstrip()
            sentence_end = previous_text.endswith((".", "!", "?", "…"))
            if (key != previous_key or gap > MAX_GROUP_GAP_SECONDS
                    or float(word["end"]) - float(current_words[0]["start"]) > MAX_GROUP_DURATION_SECONDS
                    or (sentence_end and group_duration >= SENTENCE_BREAK_MIN_SECONDS)):
                flush()
        current_words.append(word)
        current_sources.append(source)
    flush()

    result = dict(asr)
    result["segments"] = groups
    result["diarization"] = dict(diarization.get("metadata", {}))
    result["diarization"]["turn_count"] = len(turns)
    # Rejected near-silent candidates are carried through without inferred speakers.
    rejected = []
    for segment in asr.get("rejected_segments", []):
        item = dict(segment)
        item.pop("speaker", None)
        item.pop("speakers", None)
        rejected.append(item)
    if "rejected_segments" in asr:
        result["rejected_segments"] = rejected
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr", required=True, type=Path, help="ASR result JSON")
    parser.add_argument("--diarization", required=True, type=Path, help="Diarization JSON")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    args = parser.parse_args()
    try:
        asr = json.loads(args.asr.read_text(encoding="utf-8"))
        diarization = json.loads(args.diarization.read_text(encoding="utf-8"))
        result = label_result(asr, diarization)
        args.output.mkdir(parents=True, exist_ok=True)
        target = args.output / (Path(result["source"]).stem + ".json")
        render(result, args.output)
        atomic_json(target, result)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Wrote labeled transcript to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
