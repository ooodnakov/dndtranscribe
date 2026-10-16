#!/usr/bin/env python3
"""Wait for ASR and diarization, then clean completed labeled transcripts."""

import argparse
import datetime
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    from .cleanup_transcript import cleanup_complete, load_config
except ImportError:
    from cleanup_transcript import cleanup_complete, load_config

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def windows_path(path: Path) -> str:
    resolved = path.resolve()
    if sys.platform.startswith('linux') and shutil.which('wslpath'):
        result = subprocess.run(
            ['wslpath', '-w', str(resolved)], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    return str(resolved)


def powershell() -> str:
    executable = shutil.which('powershell.exe') or shutil.which('powershell')
    if not executable:
        raise RuntimeError('PowerShell is required to manage LM Studio from WSL')
    return executable


def run_powershell(script: Path) -> None:
    subprocess.run(
        [powershell(), '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', windows_path(script)],
        cwd=ROOT, check=True,
    )


def run_cleanup_child(command: list[str]) -> subprocess.CompletedProcess:
    # Keep POSIX request helpers in the publisher's own process group so interruption
    # cannot leave them running when the watcher unloads the model.
    # WSL interoperability does not guarantee cancellation of Windows processes.
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
        def stop_group(signum):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

        try:
            stdout, stderr = process.communicate()
        except BaseException:
            stop_group(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop_group(signal.SIGKILL)
                process.wait()
            finally:
                # A helper may ignore SIGTERM or outlive an already-exited publisher.
                stop_group(signal.SIGKILL)
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def workers_ready() -> tuple[bool, dict]:
    audio_names = {path.stem for path in (ROOT / 'audio').glob('*.m4a')}
    asr = read_json(ROOT / 'work' / 'asr_status.json')
    diar = read_json(ROOT / 'work' / 'diarization_status.json')
    # Counts alone can hide a missing expected track behind an unrelated stale output.
    asr_completed = asr.get('completed', [])
    diar_completed = diar.get('completed', {})
    asr_names = {Path(name).stem for name in asr_completed if isinstance(name, str)} if isinstance(asr_completed, (list, dict)) else set()
    diar_names = set(diar_completed) if isinstance(diar_completed, dict) else set()
    labeled = {
        name for name in audio_names
        if read_json(ROOT / 'transcripts' / 'labeled' / f'{name}.json').get('complete') is True
    }
    asr_ready = asr.get('state') == 'complete' and audio_names <= asr_names and not asr.get('failed')
    diar_ready = (
        diar.get('state') == 'complete' and audio_names <= diar_names and not diar.get('failed')
    )
    missing = sorted(audio_names - labeled)
    return bool(audio_names) and asr_ready and diar_ready and not missing, {
        'asr_state': asr.get('state'), 'diarization_state': diar.get('state'),
        'expected': len(audio_names), 'labeled': len(labeled), 'missing_labeled': missing,
        'missing_asr': sorted(audio_names - asr_names),
        'missing_diarization': sorted(audio_names - diar_names),
    }


def run_batch(output_dir: Path, once: bool = False, interval: int = 30) -> int:
    status_path = ROOT / 'work' / 'cleanup_status.json'
    failures, completed, details = [], [], {}
    start_attempted = False
    interrupted = False

    def status(state, current=None):
        value = {
            'state': state, **details, 'completed': completed, 'failed': failures,
            'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if current is not None:
            value['current_file'] = current
        atomic_json(status_path, value)

    try:
        while True:
            ready, details = workers_ready()
            if ready:
                break
            status('waiting_for_transcription_and_diarization')
            if once:
                print('Cleanup is waiting for complete ASR, diarization, and labeled outputs.')
                return 1
            time.sleep(interval)

        config_path = ROOT / 'work' / 'llm_config.json'
        config = load_config(config_path)
        sources = [ROOT / 'transcripts' / 'labeled' / f'{path.stem}.json'
                   for path in sorted((ROOT / 'audio').glob('*.m4a'))]
        pending = []
        for source in sources:
            if cleanup_complete(source, output_dir, config):
                completed.append(source.stem)
            else:
                pending.append(source)
        if pending:
            status('starting_model')
            start_attempted = True
            run_powershell(ROOT / 'scripts' / 'start_cleanup_gpu.ps1')
        for index, source in enumerate(pending, start=1):
            status('running', source.stem)
            result = run_cleanup_child(
                [sys.executable, str(ROOT / 'scripts' / 'cleanup_transcript.py'),
                 str(source), '--output-dir', str(output_dir), '--config', str(config_path),
                 '--overwrite'],
            )
            complete = result.returncode == 0 and cleanup_complete(source, output_dir, config)
            if not complete:
                error = (result.stderr or result.stdout).strip()[-1000:]
                failures.append({'file': source.stem, 'error': error or (
                    f'worker exited with status {result.returncode}' if result.returncode
                    else 'worker exited successfully without complete matching output artifacts'
                )})
            else:
                completed.append(source.stem)
            print(f'Cleanup {index}/{len(pending)}: {source.stem} '
                  f'{"complete" if complete else "failed"}', flush=True)
        if start_attempted:
            status('stopping_model')
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        failures.append({'error': f'{type(error).__name__}: {error}'})
    finally:
        if start_attempted:
            try:
                run_powershell(ROOT / 'scripts' / 'stop_cleanup_gpu.ps1')
            except KeyboardInterrupt:
                interrupted = True
            except Exception as error:
                failures.append({'error': f'model unload failed: {type(error).__name__}: {error}'})

    state = 'interrupted' if interrupted else ('complete_with_errors' if failures else 'complete')
    status(state)
    print('CLEANUP WORKER EXIT', state, flush=True)
    return 130 if interrupted else (1 if failures else 0)


def interrupt_worker(signum, frame):
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=int, default=30, help='poll interval in seconds')
    parser.add_argument('--once', action='store_true', help='check readiness without waiting')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'transcripts' / 'cleaned')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be positive')

    work = ROOT / 'work'
    work.mkdir(parents=True, exist_ok=True)
    # Hold an OS lock for the entire wait/batch/model lifecycle, even with a stale PID file.
    with (work / 'cleanup.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('A cleanup watcher is already running')
        pid_path = work / 'cleanup.pid'
        previous_handler = signal.signal(signal.SIGTERM, interrupt_worker)
        try:
            pid_path.write_text(str(os.getpid()) + '\n', encoding='utf-8')
            return run_batch(args.output_dir.resolve(), args.once, args.interval)
        finally:
            pid_path.unlink(missing_ok=True)
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
