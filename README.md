# dndtranscribe

Local Russian transcription and speaker labeling for recorded tabletop sessions.
Each audio track keeps its own timeline. The pipeline supports resumable Whisper
recognition, sherpa-onnx CPU diarization, optional pyannote Community-1 GPU
speaker labeling, and conservative cleanup through a local LM Studio model.

This repository contains source code, dependency locks, synthetic tests, and
example configuration only. Recordings, personal conversations, transcripts,
campaign references, credentials, local reports, and model weights are excluded.

## Requirements and setup

The primary environment uses Linux or WSL, Python 3.13, `uv`, system
`ffmpeg`/`ffprobe`, and an NVIDIA GPU for Whisper FP16 inference. The optional
Community-1 environment uses Python 3.12 and its separate dependency lock.
LM Studio integration uses Windows PowerShell from WSL.

```sh
scripts/setup.sh
scripts/run-python scripts/download_diarization_models.py
```

Download the pinned Whisper model into the ignored local cache:

```sh
scripts/run-python -c 'from faster_whisper import download_model; download_model("large-v3", cache_dir="work/models", revision="edaa852ec7e145841d8ffdb056a99866b5f0a478")'
```

Model downloads require network access and remain subject to their upstream
licenses and access conditions. Subsequent Whisper recognition uses the cache.
The diarization downloader records model provenance locally.

## Transcription and CPU speaker labeling

Place your own `.m4a` tracks under `audio/`. For paired tracks, use names such as
`session_track-01.m4a` and `session_track-02.m4a`. Keep source recordings outside
Git. Run these commands in separate terminals:

```sh
scripts/run-python scripts/transcribe.py
scripts/run-python scripts/watch_diarization.py --window-shift-ratio 0.25
```

Whisper uses Russian large-v3, FP16, batched inference, VAD, word timestamps,
and overlapping processing chunks. Quiet tracks receive bounded amplification.
Completed chunks are checkpointed for resumption. Rejected caption-like or
near-silent candidates remain in local JSON for review.

The CPU speaker-labeling stage estimates up to six speakers. It preserves word
metadata, records overlaps, and labels uncovered words `unknown`. Speaker IDs
are local estimates, not verified identities.

Generated files stay under ignored directories:

| Directory | Contents |
| --- | --- |
| `transcripts/raw/` | ASR JSON, readable TXT, and SRT subtitles |
| `transcripts/diarization/` | Estimated speaker turns and metadata |
| `transcripts/labeled/` | Speaker-labeled JSON, TXT, and SRT |
| `work/` | Checkpoints, status, logs, model cache, and local configuration |

Stop and restart the same command to resume. Review recognition accuracy,
subtitle timing, speaker assignments, and uncertain terms against the audio.

## Optional Community-1 speaker labeling

Create the isolated environment:

```sh
uv venv .venv-diarize --python 3.12
uv pip install --python .venv-diarize/bin/python -r requirements-diarize.lock
```

Accept the upstream [Community-1 model conditions](https://huggingface.co/pyannote/speaker-diarization-community-1)
and set `HF_TOKEN` in your local environment or an ignored `.env` file. Keep
credentials out of tracked files. Once other GPU inference is idle:

```sh
scripts/run-diarization-python scripts/watch_pyannote.py --once
```

Results are separate under `transcripts/diarization-community1/` and
`transcripts/labeled-community1/`. Exclusive turns align transcript words;
ordinary overlapping turns are retained separately. The watcher fingerprints
inputs/configuration, checkpoints progress, holds a process lock, and reports
per-track failures. The dependency lock targets Linux/Python 3.12 CUDA 12.8.

## Optional local cleanup

Copy `examples/llm_config.example.json` to `work/llm_config.json` and set the
identifier of an installed LM Studio model. The endpoint must remain an HTTP
loopback URL ending in `/api/v1/chat`. Do not add tokens to this configuration.
Keep GPU transcription/diarization idle while cleanup runs.

The watcher manages model loading, batch progress, and unloading:

```sh
scripts/run-python scripts/watch_cleanup.py --once --output-dir transcripts/cleaned
```

For an individual file, inspect the selected segment count without inference:

```sh
scripts/run-python scripts/cleanup_transcript.py transcripts/labeled/session_track-01.json --dry-run
```

A real individual run requires the configured model to be loaded first using
`scripts/start_cleanup_gpu.ps1`; unload it afterward using
`scripts/stop_cleanup_gpu.ps1`.

Cleanup processes review-flagged segments by default. Its dedicated system
prompt requests plain JSON, reasoning disabled, and temperature zero. Validation
preserves identities, timestamps, speakers, word counts, lexical words, and
source references. Accepted changes are limited to punctuation, spacing, and
capitalization; rejected proposals retain the source text. Extreme repeated-vowel
runs are retained without a request and explicitly marked `manual_review`.

Per-segment checkpoints retry request/invalid-JSON errors and revalidate cached
proposals. Errors prevent completed output publication. JSON/TXT/SRT are written
atomically, followed by a completion audit containing their hashes. Source files
cannot be overwritten. The watcher holds a lifecycle lock and stops the POSIX
cleanup process group before unloading on interruption; Windows bridge and
server-side cancellation through WSL has not been verified.

## Optional recording glossary hints

`examples/glossary.example.json` documents the glossary shape using an invented
example. Keep your own glossary under ignored `references/glossary.json`.
The selector discovers candidates from both raw tracks of a recording and uses
an optional local Decider runtime to choose bounded canonical hints. It does not
edit transcripts or start recognition.

CPU-only candidate discovery:

```sh
scripts/run-python scripts/select_recording_hotwords.py --recording session --glossary references/glossary.json --dry-run
```

Actual selection additionally requires a separate environment with the official
Decider runtime and local model weights; those are not installed by the base
setup or included in this repository. Supply them using `--model` and run the
same command without `--dry-run`. Profiles and evidence audits stay under
`work/recording-hotwords/`.

Profiles have an actual Whisper-token budget. Empty profiles are valid; missing
or oversized profiles fail validation. Relevance scores do not establish audio
accuracy. Review hints on short excerpts before a separately saved guided run:

```sh
scripts/run-python scripts/transcribe.py --hotwords-dir work/recording-hotwords audio/session_track-01.m4a audio/session_track-02.m4a
```

Guided outputs default to `transcripts/raw-glossary/` and separate progress.
Original baseline outputs remain intact. Recording-specific experiments and
private comparison reports are excluded from the public source release.

## Tests

```sh
scripts/run-python -W error::ResourceWarning -m unittest discover -s tests -q
```

Tests use synthetic fixtures and temporary files. They cover transcript guards,
timelines, speaker alignment, hotword selection, resumability, artifact integrity,
process locking, and interruption behavior. They do not load a model or provide
an accuracy benchmark. Keep generated test files and recordings out of commits;
the source-only `.gitignore` deliberately excludes unrecognized root content.
