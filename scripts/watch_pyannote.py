"""Resume Community-1 diarization without modifying raw or sherpa transcripts."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

try:
    from .transcribe import ROOT, atomic_json, render
    from .label_transcripts import label_result
    from .diarization_audio import prepare_diarization_input, restore_turns
    from .diarize_pyannote import CommunityDiarizer, MODEL, MODEL_REVISION
except ImportError:
    from transcribe import ROOT, atomic_json, render
    from label_transcripts import label_result
    from diarization_audio import prepare_diarization_input, restore_turns
    from diarize_pyannote import CommunityDiarizer, MODEL, MODEL_REVISION


def source_fingerprint(path):
    """Hash source bytes as well as location, so changed audio invalidates reuse."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'size': path.stat().st_size,
            'sha256': digest.hexdigest()}


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def matches(metadata, config, fingerprint, audio):
    return (metadata.get('pipeline_config') == config
            and metadata.get('raw_content_sha256') == fingerprint
            and metadata.get('source_fingerprint') == audio)


def batch_state(expected, completed, failed):
    outstanding = sorted(set(expected) - set(completed) - set(failed))
    if outstanding:
        return 'waiting_for_asr', outstanding
    return ('complete_with_errors' if failed else 'complete'), outstanding


def run_batch(root, config, once=False, diarizer_factory=CommunityDiarizer):
    raw_dir = root / 'transcripts' / 'raw'
    diar_dir = root / 'transcripts' / 'diarization-community1'
    labeled_dir = root / 'transcripts' / 'labeled-community1'
    scratch = root / 'work' / 'community1-input'
    status_path = root / 'work' / 'community1_status.json'
    for path in (diar_dir, labeled_dir, scratch):
        path.mkdir(parents=True, exist_ok=True)
    expected = {path.stem: path for path in (root / 'audio').glob('*.m4a')}
    if not expected:
        raise RuntimeError('No input audio found')
    completed, failed = {}, {}
    diarizer = None

    def status(state, outstanding=None, current=None):
        value = {'state': state, 'completed': completed, 'failed': failed,
                 'expected': len(expected), 'outstanding': outstanding if outstanding is not None else
                 sorted(set(expected) - set(completed) - set(failed)),
                 'pipeline_config': config,
                 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if current is not None:
            value['file'] = current
        atomic_json(status_path, value)

    while True:
        # Start with short recordings while preserving a deterministic order.
        ready = [(name, read_json(raw_dir / (name + '.json'))) for name in expected]
        ready.sort(key=lambda item: (item[1].get('duration', float('inf')), item[0]))
        for name, raw in ready:
            if name in completed or name in failed or not raw.get('complete'):
                continue
            wav = scratch / (name + '.wav')
            status('running', current=name)
            try:
                raw_path = raw_dir / (name + '.json')
                fingerprint = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                source = expected[name]
                audio = source_fingerprint(source)
                target = labeled_dir / raw_path.name
                diar_path = diar_dir / raw_path.name
                previous = read_json(target)
                diar = read_json(diar_path)
                metadata = diar.get('metadata', {})
                cached = (matches(metadata, config, fingerprint, audio)
                          and isinstance(diar.get('turns'), list)
                          and isinstance(diar.get('overlapping_turns'), list))
                if cached and matches(previous.get('diarization', {}), config, fingerprint, audio):
                    # Rendering is cheap and repairs missing/interrupted TXT/SRT files.
                    render(previous, labeled_dir)
                    completed[name] = metadata['speaker_count']
                    print('SKIP Community-1', name, flush=True)
                    continue
                if not cached:
                    if not raw.get('segments'):
                        preparation = None
                        diar = {'metadata': {'track': name, 'model': MODEL,
                                'model_revision': MODEL_REVISION, 'input': str(source),
                                'duration_seconds': raw['duration'], 'speaker_scope': 'recording_global',
                                'alignment': 'exclusive_speaker_diarization', 'speaker_count': 0,
                                'reason': 'No usable speech in ASR result'},
                                'turns': [], 'overlapping_turns': []}
                    else:
                        preparation = prepare_diarization_input(source, raw, wav)
                        if preparation['prepared_duration_seconds'] <= 0:
                            raise ValueError('ASR has no usable timestamped speech regions')
                        if diarizer is None:
                            diarizer = diarizer_factory(device=config['device'],
                                                       max_speakers=config['max_speakers'])
                        diar = restore_turns(diarizer.diarize(wav, name), preparation)
                    diar['metadata'].update(pipeline_config=config, raw_content_sha256=fingerprint,
                                            source_fingerprint=audio)
                    atomic_json(diar_path, diar)
                result = label_result(raw, diar)
                render(result, labeled_dir)
                atomic_json(target, result)
                completed[name] = diar['metadata']['speaker_count']
                print('LABELED Community-1', name, 'speakers:', completed[name], flush=True)
            except Exception as error:
                failed[name] = f'{type(error).__name__}: {error}'
                print('FAILED Community-1', name, failed[name], flush=True)
            finally:
                wav.unlink(missing_ok=True)
            status('running')
        state, outstanding = batch_state(expected, completed, failed)
        asr_status = read_json(root / 'work' / 'asr_status.json')
        if state == 'waiting_for_asr' and (once or asr_status.get('state') == 'complete'):
            state = 'incomplete'
        status(state, outstanding)
        if once or not outstanding or asr_status.get('state') == 'complete':
            print('COMMUNITY-1 WORKER EXIT', state, flush=True)
            return 1 if failed or outstanding else 0
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    config = {'version': 1, 'model': MODEL, 'model_revision': MODEL_REVISION,
              'pyannote_version': importlib.metadata.version('pyannote.audio'),
              'torch_version': importlib.metadata.version('torch'), 'device': args.device,
              'max_speakers': 6, 'alignment': 'exclusive_speaker_diarization',
              'sparse_asr_coverage_threshold': 0.25, 'region_padding_seconds': 0.3,
              'region_separator_seconds': 0.5, 'gain': 'asr_chunk_gain', 'label_version': 2,
              'timeline_bounds_version': 1}
    work = ROOT / 'work'
    work.mkdir(parents=True, exist_ok=True)
    # The OS releases this lock even after a crash; stale PID files cannot block resume.
    with (work / 'community1.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('A Community-1 worker is already running')
        pid_path = work / 'community1.pid'
        pid_path.write_text(str(os.getpid()) + '\n')
        try:
            return run_batch(ROOT, config, args.once)
        finally:
            pid_path.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
