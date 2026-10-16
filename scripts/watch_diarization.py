"""Process completed ASR files while GPU transcription continues independently."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from transcribe import ROOT, atomic_json, render
from label_transcripts import label_result
from diarization_audio import prepare_diarization_input, restore_turns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-shift-ratio', type=float, default=0.1)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    config = {'version': 2, 'window_shift_ratio': args.window_shift_ratio,
              'clustering_threshold': 0.7, 'max_speakers': 6, 'sparse_asr_coverage_threshold': 0.25}
    directory = ROOT / 'transcripts'
    raw_dir = directory / 'raw'
    diar_dir = directory / 'diarization'
    labeled_dir = directory / 'labeled'
    scratch = ROOT / 'work' / 'diarization_input'
    for path in [diar_dir, labeled_dir, scratch]:
        path.mkdir(parents=True, exist_ok=True)
    completed, failed = {}, {}
    expected_names = {path.stem for path in (ROOT / 'audio').glob('*.m4a')}
    expected = len(expected_names)
    while True:
        ready_paths = sorted(raw_dir.glob('*.json'), key=lambda p: (json.loads(p.read_text())['duration'] >= 600, p.name))
        for raw_path in ready_paths:
            raw = json.loads(raw_path.read_text())
            if not raw.get('complete'):
                continue
            name = raw_path.stem
            target = labeled_dir / raw_path.name
            fingerprint = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            if target.exists():
                previous = json.loads(target.read_text())
                if (isinstance(previous.get('diarization'), dict)
                    and previous['diarization'].get('pipeline_config') == config
                    and previous['diarization'].get('raw_content_sha256') == fingerprint):
                    completed[name] = previous['diarization'].get('speaker_count', 0)
                    continue
            if name in failed:
                continue
            print('DIARIZE', name, flush=True)
            temporary_wav = scratch / (name + '.wav')
            diar_path = diar_dir / raw_path.name
            atomic_json(ROOT / 'work' / 'diarization_status.json', {
                'state': 'running', 'file': name, 'completed': completed, 'failed': failed,
                'expected': expected, 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            try:
                if not raw['segments']:
                    diar = {'metadata': {'track': name, 'speaker_scope': 'recording_global',
                            'speaker_count': 0, 'reason': 'No usable speech in ASR result'}, 'turns': []}
                    atomic_json(diar_path, diar)
                else:
                    cached = json.loads(diar_path.read_text()) if diar_path.exists() else {}
                    if (cached.get('metadata', {}).get('pipeline_config') != config
                        or cached.get('metadata', {}).get('raw_content_sha256') != fingerprint):
                        temporary_wav.unlink(missing_ok=True)
                        preparation = prepare_diarization_input(Path(raw['source']), raw, temporary_wav)
                        compact_result = scratch / (name + '.compact.json')
                        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'diarize.py'),
                                        str(temporary_wav), '--output', str(compact_result),
                                        '--track-name', name, '--max-speakers', '6', '--num-threads', '4',
                                        '--threshold', '0.7', '--window-shift-ratio', str(args.window_shift_ratio)], check=True)
                        diar = restore_turns(json.loads(compact_result.read_text()), preparation)
                        diar['metadata']['input'] = raw['source']
                        atomic_json(diar_path, diar)
                        compact_result.unlink(missing_ok=True)
                    diar = json.loads(diar_path.read_text())
                diar['metadata']['pipeline_config'] = config
                diar['metadata']['raw_content_sha256'] = fingerprint
                atomic_json(diar_path, diar)
                result = label_result(raw, diar)
                render(result, labeled_dir)
                atomic_json(target, result)
                completed[name] = diar['metadata']['speaker_count']
                print('LABELED', name, 'speakers:', completed[name], flush=True)
            except Exception as error:
                failed[name] = f'{type(error).__name__}: {error}'
                print('FAILED', name, failed[name], flush=True)
            finally:
                temporary_wav.unlink(missing_ok=True)
        asr_status_path = ROOT / 'work' / 'asr_status.json'
        asr_status = json.loads(asr_status_path.read_text()) if asr_status_path.exists() else {}
        finished = asr_status.get('state') == 'complete' and asr_status.get('total_files') == expected
        outstanding = expected_names - set(completed) - set(failed)
        state = ('complete_with_errors' if failed else 'complete') if finished and not outstanding else (
            'incomplete' if finished else 'waiting_for_asr'
        )
        status = {'state': state,
                  'completed': completed, 'failed': failed, 'expected': expected,
                  'outstanding': sorted(outstanding),
                  'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        atomic_json(ROOT / 'work' / 'diarization_status.json', status)
        if args.once or finished:
            print('DIARIZATION WORKER EXIT', status['state'], flush=True)
            return 1 if failed or outstanding else 0
        # Do not wait forever after an ASR crash. Completed transcripts remain usable.
        pid_path = ROOT / 'work' / 'asr.pid'
        if pid_path.exists() and asr_status.get('state') != 'temporarily_stopped_for_gpu_llm_test':
            try:
                os.kill(int(pid_path.read_text()), 0)
            except ProcessLookupError:
                status['state'] = 'asr_process_stopped_before_completion'
                atomic_json(ROOT / 'work' / 'diarization_status.json', status)
                return 1
        time.sleep(10)


if __name__ == '__main__':
    raise SystemExit(main())
