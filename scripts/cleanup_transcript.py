#!/usr/bin/env python3
"""Conservatively propose edits to a labeled transcript through LM Studio.

The source JSON is never modified. By default, only segments marked
``review_recommended`` are sent to the local model. ``--all`` is intended for
small, explicitly reviewed samples.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "work" / "llm_config.json"
POWERSHELL_BRIDGE = ROOT / "scripts" / "lmstudio_chat.ps1"
WORD_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)
RESPONSE_FIELDS = {"id", "start", "end", "speaker", "source_ids", "text"}
REPEATED_VOWEL_RE = re.compile(r"([аеёиоуыэюяaeiou])\1{31,}", re.IGNORECASE)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def cleanup_fingerprint(
    source_path: Path, config: dict[str, Any], process_all: bool = False,
    source_bytes: bytes | None = None,
) -> str:
    """Identity shared by the checkpoint, final audit, and batch watcher."""
    identity = {
        "source": str(source_path.resolve()),
        "source_sha256": digest(source_path.read_bytes() if source_bytes is None else source_bytes),
        "config": config,
        "process_all": process_all,
        "code_sha256": digest(Path(__file__).read_bytes()),
        "bridge_sha256": digest(POWERSHELL_BRIDGE.read_bytes()),
    }
    return digest(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def cleanup_complete(
    source_path: Path, output_dir: Path, config: dict[str, Any], process_all: bool = False,
) -> bool:
    """Check the last-published completion marker without loading a model."""
    try:
        audit = json.loads((output_dir / f"{source_path.stem}.audit.json").read_text(encoding="utf-8"))
        if (not isinstance(audit, dict) or audit.get("complete") is not True
                or audit.get("fingerprint") != cleanup_fingerprint(source_path, config, process_all)
                or audit.get("errors") != 0):
            return False
        hashes = audit.get("artifact_sha256")
        if not isinstance(hashes, dict) or set(hashes) != {"json", "txt", "srt"}:
            return False
        return all(
            digest((output_dir / f"{source_path.stem}.{extension}").read_bytes()) == expected
            for extension, expected in hashes.items()
        )
    except (OSError, ValueError, TypeError):
        return False


@contextmanager
def source_lock(source_path: Path):
    """OS releases this advisory lock even if a cleanup worker is killed."""
    directory = ROOT / "work" / "cleanup-locks"
    directory.mkdir(parents=True, exist_ok=True)
    key = digest(str(source_path.resolve()).encode("utf-8"))
    with (directory / f"{key}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"cleanup already running for {source_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def timestamp(seconds: float, srt: bool = False) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    whole, ms = divmod(milliseconds, 1000)
    minutes, sec = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{sec:02}" + (f",{ms:03}" if srt else "")


def render(data: dict[str, Any]) -> tuple[str, str]:
    text_lines: list[str] = []
    subtitles: list[str] = []
    for number, segment in enumerate(data["segments"], start=1):
        text = str(segment.get("text", "")).strip()
        speaker = segment.get("speaker")
        if speaker:
            text = f"{speaker}: {text}"
        start, end = float(segment["start"]), float(segment["end"])
        text_lines.append(f"[{timestamp(start)}–{timestamp(end)}] {text}")
        subtitles.append(
            f"{number}\n{timestamp(start, True)} --> {timestamp(end, True)}\n{text}\n"
        )
    return "\n".join(text_lines) + "\n", "\n".join(subtitles) + "\n"


def windows_path(path: Path) -> str:
    """Convert a WSL path for Windows PowerShell without shell interpolation."""
    resolved = path.resolve()
    if sys.platform.startswith("linux") and shutil.which("wslpath"):
        result = subprocess.run(
            ["wslpath", "-w", str(resolved)], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    return str(resolved)


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"Missing local config {path}. Create it with endpoint and model only; "
            "do not put API tokens in this file."
        ) from exc
    if not isinstance(config, dict):
        raise ValueError("LLM config must be a JSON object")
    for key in ("api_url", "model"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"LLM config requires a non-empty {key!r}")
    endpoint = urlsplit(config["api_url"])
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("LM Studio endpoint must be an HTTP loopback address")
    if endpoint.path != "/api/v1/chat":
        raise ValueError("LM Studio endpoint must use the native /api/v1/chat API")
    return config


def call_lmstudio(
    config: dict[str, Any],
    record: dict[str, Any],
    work_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system = (
        "Ты осторожный редактор русской расшифровки настольной ролевой игры. "
        "Исправляй только пунктуацию, регистр букв и пробелы. "
        "Сохраняй все слова, имена, повторы, порядок и смысл; не заменяй, "
        "не добавляй и не удаляй слова. Не угадывай имена, термины или события. "
        "Если исправление не очевидно — оставь исходный текст. "
        "Верни только один JSON-объект с теми же полями id, start, end, speaker, "
        "source_ids и с отредактированным text. Все поля, кроме text, копируй дословно. "
        "Ответ должен начинаться с { и заканчиваться }. "
        "Не используй Markdown и блоки кода с тройными обратными кавычками. "
        "Не добавляй объяснений, заголовков или новых фраз."
    )
    payload = {
        "model": config["model"],
        "system_prompt": system,
        "input": json.dumps(record, ensure_ascii=False, separators=(",", ":")),
        "temperature": 0,
        "reasoning": "off",
        "max_output_tokens": min(2000, max(100, int(config.get("max_output_tokens", 2000)))),
        "store": False,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path: Path | None = None
    response_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", suffix=".json",
            prefix="cleanup-request-", dir=work_dir, delete=False
        ) as request_file:
            request_path = Path(request_file.name)
            json.dump(payload, request_file, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".json", prefix="cleanup-response-",
            dir=work_dir, delete=False
        ) as response_file:
            response_path = Path(response_file.name)

        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if not powershell:
            raise RuntimeError("PowerShell is required to reach LM Studio from WSL")
        helper = windows_path(POWERSHELL_BRIDGE)
        args = [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            helper,
            windows_path(request_path),
            windows_path(response_path),
            str(config["api_url"]),
            str(config["model"]),
            str(int(config.get("timeout_seconds", 300))),
        ]
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"LM Studio request failed: {detail or 'PowerShell bridge error'}")
        response = json.loads(response_path.read_text(encoding="utf-8"))
    finally:
        for temporary in (request_path, response_path):
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    if not isinstance(response, dict):
        raise ValueError("LM Studio response was not an object")
    stats = response.get("stats") or {}
    output = response.get("output", [])
    if not isinstance(stats, dict) or not isinstance(output, list) or any(
        not isinstance(item, dict) for item in output
    ):
        raise ValueError("LM Studio response has invalid stats or output")
    reasoning_ok, reasoning_error = validate_reasoning_output(stats, output)
    if not reasoning_ok:
        raise ValueError(reasoning_error)
    messages = [
        str(item.get("content", ""))
        for item in output
        if item.get("type") == "message"
    ]
    if len(messages) != 1 or not messages[0].strip():
        raise ValueError("expected exactly one non-empty message in the model response")
    try:
        proposal = json.loads(messages[0])
    except json.JSONDecodeError as exc:
        raise ValueError("model response was not a JSON object") from exc
    if not isinstance(proposal, dict):
        raise ValueError("model response JSON was not an object")
    return proposal, stats


def validate_reasoning_output(stats: dict[str, Any], output: list[dict[str, Any]]) -> tuple[bool, str]:
    reasoning_tokens = int(stats.get("reasoning_output_tokens", 0) or 0)
    if reasoning_tokens:
        return False, f"model returned {reasoning_tokens} reasoning tokens despite reasoning=off"
    if any(item.get("type") == "reasoning" for item in output if isinstance(item, dict)):
        return False, "model returned a reasoning block; proposal rejected"
    return True, "reasoning off"


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for col, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[col] + 1,
                    previous[col - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def validate_proposal(
    expected: dict[str, Any], proposal: dict[str, Any]
) -> tuple[bool, str, int]:
    if set(proposal) != RESPONSE_FIELDS:
        return False, "response fields differ from the required schema", 0
    for key in RESPONSE_FIELDS - {"text"}:
        if proposal.get(key) != expected.get(key):
            return False, f"{key} changed", 0
    text = proposal.get("text")
    if not isinstance(text, str) or not text or text != text.strip():
        return False, "text is empty or has surrounding whitespace", 0
    if "\n" in text or "\r" in text:
        return False, "model added a line break", 0

    original = str(expected["text"])
    if text == original:
        return True, "unchanged", 0
    distance = levenshtein(original, text)
    max_distance = max(4, min(20, math.ceil(len(original) * 0.15)))
    if distance > max_distance:
        return False, f"edit distance {distance} exceeds budget {max_distance}", distance
    old_words = WORD_RE.findall(original)
    new_words = WORD_RE.findall(text)
    if len(old_words) != len(new_words):
        return False, "word count changed", distance
    if old_words and all(
        old.casefold() == new.casefold()
        for old, new in zip(old_words[:-1], new_words[:-1])
    ):
        old_last, new_last = old_words[-1].casefold(), new_words[-1].casefold()
        if len(new_last) < len(old_last) and old_last.startswith(new_last):
            return False, "last word appears truncated", distance
    if any(old.casefold() != new.casefold() for old, new in zip(old_words, new_words)):
        return False, "lexical word changed; manual audio review required", distance
    return True, "accepted", distance


def stable_record(index: int, segment: dict[str, Any]) -> dict[str, Any]:
    source_segments = segment.get("source_segments") or []
    source_ids = [item.get("id") for item in source_segments if isinstance(item, dict)]
    return {
        "id": index,
        "start": segment.get("start"),
        "end": segment.get("end"),
        "speaker": segment.get("speaker"),
        "source_ids": source_ids,
        "text": segment.get("text", ""),
    }


def write_outputs(
    source_path: Path,
    output_dir: Path,
    data: dict[str, Any],
    audit: dict[str, Any],
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    outputs = {
        output_dir / f"{stem}.json": json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    }
    text_content, srt_content = render(data)
    outputs[output_dir / f"{stem}.txt"] = text_content
    outputs[output_dir / f"{stem}.srt"] = srt_content
    audit_path = output_dir / f"{stem}.audit.json"
    existing = [path for path in [*outputs, audit_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )
    # Retire any old marker before replacing artifacts. A crash midway cannot
    # advertise an incomplete set as complete; the checkpoint remains resumable.
    audit_path.unlink(missing_ok=True)
    for path, content in outputs.items():
        atomic_write(path, content)
    audit = {
        **audit,
        "complete": True,
        "artifact_sha256": {
            path.suffix.lstrip("."): digest(content.encode("utf-8"))
            for path, content in outputs.items()
        },
    }
    atomic_write(audit_path, json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


def cleanup_source(
    source_path: Path, output_dir: Path, config: dict[str, Any],
    process_all: bool = False, overwrite: bool = False, dry_run: bool = False,
) -> int:
    if (output_dir / f"{source_path.stem}.json").resolve() == source_path.resolve():
        raise ValueError("cleanup output JSON must differ from the source transcript")
    with source_lock(source_path):
        source_bytes = source_path.read_bytes()
        data = json.loads(source_bytes)
        if (not isinstance(data, dict) or not isinstance(data.get("segments"), list)
                or any(not isinstance(segment, dict) for segment in data["segments"])):
            raise ValueError("input JSON must be a labeled transcript with a segments list")
        if data.get("complete") is not True:
            raise ValueError("input transcript is not marked complete")
        candidates = [
            (index, segment)
            for index, segment in enumerate(data["segments"])
            if process_all or segment.get("review_recommended") is True
        ]
        mode = "all" if process_all else "review_recommended only"
        print(f"Selected {len(candidates)} of {len(data['segments'])} segments ({mode}).")
        if dry_run:
            return 0
        if cleanup_complete(source_path, output_dir, config, process_all):
            print("Matching complete cleanup already exists.")
            return 0
        if not overwrite and any(
            (output_dir / f"{source_path.stem}.{extension}").exists()
            for extension in ("json", "txt", "srt", "audit.json")
        ):
            raise FileExistsError("Output already exists; pass --overwrite to replace stale or incomplete outputs")

        fingerprint = cleanup_fingerprint(source_path, config, process_all, source_bytes)
        checkpoint_path = output_dir / f".{source_path.stem}.cleanup.checkpoint.json"
        checkpoint: dict[str, Any] = {}
        try:
            cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
                checkpoint = cached
        except (OSError, ValueError):
            pass
        cached_items = checkpoint.get("proposals", [])
        if not isinstance(cached_items, list):
            cached_items = []
        saved = {
            item["id"]: item for item in cached_items
            if isinstance(item, dict) and type(item.get("id")) is int
            and isinstance(item.get("status"), str)
            and item["status"] in {"accepted", "unchanged", "rejected"}
        }
        cleaned = copy.deepcopy(data)
        audit_items: list[dict[str, Any]] = []
        for index, segment in candidates:
            expected = stable_record(index, segment)
            audit_item = saved.get(index)
            if audit_item is not None:
                proposal = audit_item.get("proposal")
                if not isinstance(proposal, dict):
                    audit_item = None
                else:
                    accepted, reason, distance = validate_proposal(expected, proposal)
                    status = "unchanged" if accepted and reason == "unchanged" else (
                        "accepted" if accepted else "rejected"
                    )
                    if (audit_item["status"] != status
                            or any(audit_item.get(key) != value for key, value in expected.items())):
                        audit_item = None
                    else:
                        audit_item = {**audit_item, "reason": reason, "edit_distance": distance}
            # Extreme vowel runs cannot be repaired within the lexical guardrail.
            # Preserve them for audio review rather than sending a copying request
            # that can run to the model's output limit. Recompute this every run.
            if REPEATED_VOWEL_RE.search(str(expected["text"])):
                audit_item = {
                    **expected, "original_text": str(segment.get("text", "")),
                    "proposal": None, "status": "manual_review", "request_sent": False,
                    "reason": "extreme repeated-vowel run; manual audio review required",
                }
            if audit_item is None:
                audit_item = {
                    **expected, "original_text": str(segment.get("text", "")),
                    "proposal": None, "status": "error", "reason": "not processed",
                }
                try:
                    proposal, stats = call_lmstudio(config, expected, ROOT / "work")
                    audit_item["proposal"] = proposal
                    audit_item["stats"] = {
                        key: stats[key]
                        for key in ("input_tokens", "total_output_tokens", "reasoning_output_tokens", "tokens_per_second")
                        if key in stats
                    }
                    accepted, reason, distance = validate_proposal(expected, proposal)
                    status = "unchanged" if accepted and reason == "unchanged" else (
                        "accepted" if accepted else "rejected"
                    )
                    audit_item.update(status=status, reason=reason, edit_distance=distance)
                except (RuntimeError, ValueError, OSError, TypeError) as exc:
                    audit_item["reason"] = str(exc)
            if audit_item["status"] == "accepted":
                cleaned["segments"][index]["text"] = audit_item["proposal"]["text"]
            audit_items.append(audit_item)
            # Include later cached successes too, so an interrupted retry does
            # not discard decisions reached after an earlier failed request.
            retained = {**saved, **{item["id"]: item for item in audit_items}}
            atomic_write(checkpoint_path, json.dumps({
                "fingerprint": fingerprint, "complete": False,
                "proposals": list(retained.values()),
            }, ensure_ascii=False, indent=2) + "\n")
            print(f"segment {index}: {audit_item['status']} ({audit_item['reason']})")

        errors = sum(item["status"] == "error" for item in audit_items)
        if errors:
            print(f"cleanup_transcript: {errors} retryable segment error(s); checkpoint saved", file=sys.stderr)
            return 2
        if cleanup_fingerprint(source_path, config, process_all) != fingerprint:
            raise RuntimeError("source or cleanup implementation changed during processing; retry cleanup")
        now = datetime.now(timezone.utc).isoformat()
        cleaned["cleanup"] = {
            "model": config["model"], "created_at": now, "mode": mode,
            "fingerprint": fingerprint,
            "accepted_edits": sum(item["status"] == "accepted" for item in audit_items),
            "unchanged": sum(item["status"] == "unchanged" for item in audit_items),
            "rejected": sum(item["status"] == "rejected" for item in audit_items),
            "manual_review": sum(item["status"] == "manual_review" for item in audit_items),
        }
        audit = {
            "source": str(source_path), "model": config["model"],
            "created_at": now, "mode": mode, "fingerprint": fingerprint,
            "errors": 0, "manual_review": cleaned["cleanup"]["manual_review"],
            "proposals": audit_items,
        }
        write_outputs(source_path, output_dir, cleaned, audit, overwrite)
        print(f"Wrote cleaned JSON, audit JSON, TXT, and SRT to {output_dir}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="labeled transcript JSON")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "transcripts" / "cleaned"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--all", action="store_true", help="process all segments, including unflagged ones"
    )
    parser.add_argument("--dry-run", action="store_true", help="show selected segment count only")
    parser.add_argument("--overwrite", action="store_true", help="replace existing cleaned outputs")
    args = parser.parse_args()

    try:
        return cleanup_source(
            args.input.resolve(), args.output_dir.resolve(), load_config(args.config),
            process_all=args.all, overwrite=args.overwrite, dry_run=args.dry_run,
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"cleanup_transcript: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
