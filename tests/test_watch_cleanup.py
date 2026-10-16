import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import cleanup_transcript, watch_cleanup


class CleanupWatcherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(watch_cleanup, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        (self.root / 'audio').mkdir()
        (self.root / 'work').mkdir()
        (self.root / 'transcripts' / 'labeled').mkdir(parents=True)
        self.output_dir = self.root / 'transcripts' / 'cleaned-v2'
        self.config = {'api_url': 'http://localhost:1234/api/v1/chat', 'model': 'current-model'}
        self.write('work/llm_config.json', self.config)
        self.fixture()

    def write(self, path, data):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data), encoding='utf-8')

    def fixture(self, names=('one',)):
        for name in names:
            (self.root / 'audio' / f'{name}.m4a').touch()
            self.write(f'transcripts/labeled/{name}.json', {'complete': True, 'segments': []})
        self.write('work/asr_status.json', {'state': 'complete', 'completed': [f'{name}.m4a' for name in names]})
        self.write('work/diarization_status.json', {'state': 'complete', 'completed': dict.fromkeys(names, 1), 'failed': {}})

    def status(self):
        return json.loads((self.root / 'work' / 'cleanup_status.json').read_text())

    def run_batch(self):
        return watch_cleanup.run_batch(self.output_dir, once=True)

    def test_readiness_matches_expected_track_identities_and_labeled_completion(self):
        self.assertTrue(watch_cleanup.workers_ready()[0])
        self.write('work/asr_status.json', {'state': 'complete', 'completed': ['unrelated.m4a']})
        self.assertFalse(watch_cleanup.workers_ready()[0])
        self.fixture()
        self.write('work/diarization_status.json', {'state': 'complete', 'completed': {'unrelated': 1}})
        self.assertFalse(watch_cleanup.workers_ready()[0])
        self.fixture()
        self.write('transcripts/labeled/one.json', {'complete': False, 'segments': []})
        self.assertFalse(watch_cleanup.workers_ready()[0])
        (self.root / 'audio' / 'one.m4a').unlink()
        self.assertFalse(watch_cleanup.workers_ready()[0])

    def test_incomplete_once_exits_unsuccessfully_without_starting_model(self):
        (self.root / 'transcripts/labeled/one.json').unlink()
        with patch.object(watch_cleanup, 'run_powershell') as manage:
            self.assertEqual(self.run_batch(), 1)
        manage.assert_not_called()
        self.assertEqual(self.status()['state'], 'waiting_for_transcription_and_diarization')

    def test_matching_artifacts_skip_model_and_ignore_unrelated_labeled_files(self):
        self.write('transcripts/labeled/unrelated.json', {'complete': True, 'segments': []})
        with patch.object(watch_cleanup, 'cleanup_complete', return_value=True) as complete, \
                patch.object(watch_cleanup, 'run_powershell') as manage, \
                patch.object(watch_cleanup, 'run_cleanup_child') as child:
            self.assertEqual(self.run_batch(), 0)
        complete.assert_called_once_with(self.root / 'transcripts/labeled/one.json', self.output_dir, self.config)
        manage.assert_not_called()
        child.assert_not_called()
        self.assertEqual(self.status()['completed'], ['one'])
        self.assertEqual(self.status()['state'], 'complete')

    def test_existing_failed_audit_does_not_skip_retry_and_output_is_verified(self):
        self.write('transcripts/cleaned-v2/one.audit.json', {'proposals': [{'reason': 'request failed'}]})
        with patch.object(watch_cleanup, 'cleanup_complete', side_effect=[False, True]) as complete, \
                patch.object(watch_cleanup, 'run_powershell') as manage, \
                patch.object(watch_cleanup, 'run_cleanup_child', return_value=subprocess.CompletedProcess([], 0, '', '')) as child:
            self.assertEqual(self.run_batch(), 0)
        self.assertEqual(complete.call_count, 2)
        self.assertEqual([call.args[0].name for call in manage.call_args_list],
                         ['start_cleanup_gpu.ps1', 'stop_cleanup_gpu.ps1'])
        self.assertIn('--overwrite', child.call_args.args[0])
        self.assertIn(str(self.output_dir), child.call_args.args[0])
        self.assertEqual(self.status()['state'], 'complete')

    def test_zero_child_exit_without_complete_artifacts_stays_unsuccessful(self):
        with patch.object(watch_cleanup, 'cleanup_complete', return_value=False), \
                patch.object(watch_cleanup, 'run_powershell'), \
                patch.object(watch_cleanup, 'run_cleanup_child', return_value=subprocess.CompletedProcess([], 0, '', '')):
            self.assertEqual(self.run_batch(), 1)
        self.assertEqual(self.status()['state'], 'complete_with_errors')
        self.assertEqual(self.status()['completed'], [])
        self.assertIn('complete matching output artifacts', self.status()['failed'][0]['error'])

    def test_nonzero_child_exit_stays_failed_even_if_outputs_exist_and_batch_continues(self):
        self.fixture(('one', 'two'))
        with patch.object(watch_cleanup, 'cleanup_complete', side_effect=[False, False, True]), \
                patch.object(watch_cleanup, 'run_powershell'), \
                patch.object(watch_cleanup, 'run_cleanup_child', side_effect=[
                    subprocess.CompletedProcess([], 2, '', 'request failed'),
                    subprocess.CompletedProcess([], 0, '', '')]):
            self.assertEqual(self.run_batch(), 1)
        self.assertEqual(self.status()['completed'], ['two'])
        self.assertEqual(self.status()['failed'], [{'file': 'one', 'error': 'request failed'}])

    def test_start_failure_attempts_unload_and_is_unsuccessful(self):
        with patch.object(watch_cleanup, 'cleanup_complete', return_value=False), \
                patch.object(watch_cleanup, 'run_powershell', side_effect=[RuntimeError('start failure'), None]) as manage:
            self.assertEqual(self.run_batch(), 1)
        self.assertEqual(manage.call_count, 2)
        self.assertEqual(self.status()['state'], 'complete_with_errors')

    def test_interrupt_marks_status_and_unloads_model(self):
        with patch.object(watch_cleanup, 'cleanup_complete', return_value=False), \
                patch.object(watch_cleanup, 'run_powershell') as manage, \
                patch.object(watch_cleanup, 'run_cleanup_child', side_effect=KeyboardInterrupt):
            self.assertEqual(self.run_batch(), 130)
        self.assertEqual(manage.call_count, 2)
        self.assertEqual(self.status()['state'], 'interrupted')

    def test_os_lock_rejects_second_worker_without_removing_owners_pid(self):
        pid_path = self.root / 'work' / 'cleanup.pid'
        pid_path.write_text('existing owner\n')
        with (self.root / 'work' / 'cleanup.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(sys, 'argv', ['watch_cleanup.py', '--once']), \
                    patch.object(watch_cleanup, 'run_batch') as run:
                with self.assertRaises(SystemExit) as error:
                    watch_cleanup.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
        self.assertEqual(pid_path.read_text(), 'existing owner\n')

    def test_stale_pid_does_not_block_and_sigterm_unwinds_pid_and_lock(self):
        pid_path = self.root / 'work' / 'cleanup.pid'
        pid_path.write_text('999999999\n')
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(*args):
            self.assertEqual(pid_path.read_text().strip(), str(os.getpid()))
            os.kill(os.getpid(), signal.SIGTERM)

        with patch.object(sys, 'argv', ['watch_cleanup.py', '--once']), \
                patch.object(watch_cleanup, 'run_batch', side_effect=terminate):
            with self.assertRaises(KeyboardInterrupt):
                watch_cleanup.main()
        self.assertFalse(pid_path.exists())
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        with (self.root / 'work' / 'cleanup.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_real_engine_completion_invalidates_missing_or_changed_artifacts_config_and_source(self):
        source = self.root / 'transcripts/labeled/one.json'
        with patch.object(cleanup_transcript, 'ROOT', self.root):
            self.assertEqual(cleanup_transcript.cleanup_source(source, self.output_dir, self.config), 0)
            with patch.object(watch_cleanup, 'run_powershell') as manage:
                self.assertEqual(self.run_batch(), 0)
            manage.assert_not_called()
            for change in ('txt', 'srt', 'json', 'audit', 'config', 'source'):
                with self.subTest(change=change):
                    if change in ('txt', 'srt', 'json'):
                        (self.output_dir / f'one.{change}').unlink()
                    elif change == 'audit':
                        self.write('transcripts/cleaned-v2/one.audit.json', {'complete': False, 'errors': 1})
                    elif change == 'config':
                        self.config = {**self.config, 'model': 'replacement-model'}
                        self.write('work/llm_config.json', self.config)
                    else:
                        self.write('transcripts/labeled/one.json', {'complete': True, 'segments': [], 'changed': True})
                    with patch.object(watch_cleanup, 'run_powershell') as manage, \
                            patch.object(watch_cleanup, 'run_cleanup_child', return_value=subprocess.CompletedProcess([], 2, '', 'synthetic failure')) as child:
                        self.assertEqual(self.run_batch(), 1)
                    self.assertEqual(manage.call_count, 2)
                    child.assert_called_once()
                    self.assertEqual(cleanup_transcript.cleanup_source(source, self.output_dir, self.config, overwrite=True), 0)

    def test_real_child_is_reaped_before_model_unload_after_parent_only_sigterm(self):
        child_pid = self.root / 'child.pid'
        original_run = watch_cleanup.run_cleanup_child
        code = ("import os, pathlib, signal, sys, time; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                "os.kill(os.getppid(), signal.SIGTERM); time.sleep(60)")

        def child(*args, **kwargs):
            return original_run([sys.executable, '-c', code, str(child_pid)])

        def manage(script):
            if script.name == 'stop_cleanup_gpu.ps1':
                pid = int(child_pid.read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

        with patch.object(sys, 'argv', ['watch_cleanup.py', '--once', '--output-dir', str(self.output_dir)]), \
                patch.object(watch_cleanup, 'cleanup_complete', return_value=False), \
                patch.object(watch_cleanup, 'run_powershell', side_effect=manage), \
                patch.object(watch_cleanup, 'run_cleanup_child', side_effect=child):
            self.assertEqual(watch_cleanup.main(), 130)
        self.assertEqual(self.status()['state'], 'interrupted')
        self.assertEqual(self.status()['failed'], [])
        self.assertFalse((self.root / 'work' / 'cleanup.pid').exists())

    def test_sigterm_during_batch_marks_interrupted_and_unloads(self):
        with patch.object(sys, 'argv', ['watch_cleanup.py', '--once', '--output-dir', str(self.output_dir)]), \
                patch.object(watch_cleanup, 'cleanup_complete', return_value=False), \
                patch.object(watch_cleanup, 'run_powershell') as manage, \
                patch.object(watch_cleanup, 'run_cleanup_child', side_effect=lambda *args, **kwargs: os.kill(os.getpid(), signal.SIGTERM)):
            self.assertEqual(watch_cleanup.main(), 130)
        self.assertEqual(self.status()['state'], 'interrupted')
        self.assertEqual(manage.call_count, 2)
        self.assertFalse((self.root / 'work' / 'cleanup.pid').exists())

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'requires Linux process state inspection')
    def test_parent_only_sigterm_stops_request_descendant_before_model_unload(self):
        helper_pid = self.root / 'helper.pid'
        original_run = watch_cleanup.run_cleanup_child
        helper_code = (
            "import os, pathlib, signal, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
        )
        publisher_code = '''import os, pathlib, signal, subprocess, sys, time
subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
deadline = time.monotonic() + 5
while not pathlib.Path(sys.argv[1]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
assert pathlib.Path(sys.argv[1]).exists(), 'request helper did not start'
os.kill(os.getppid(), signal.SIGTERM)
time.sleep(60)
'''

        def child(*args, **kwargs):
            return original_run([sys.executable, '-c', publisher_code, str(helper_pid), helper_code])

        def manage(script):
            if script.name == 'stop_cleanup_gpu.ps1':
                pid = int(helper_pid.read_text())
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    try:
                        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
                    except (FileNotFoundError, ProcessLookupError):
                        return
                    if state == 'Z':  # An orphaned zombie is already terminated.
                        return
                    time.sleep(0.01)
                self.fail('request helper is still running when the model is unloaded')

        try:
            with patch.object(sys, 'argv', ['watch_cleanup.py', '--once', '--output-dir', str(self.output_dir)]), \
                    patch.object(watch_cleanup, 'cleanup_complete', return_value=False), \
                    patch.object(watch_cleanup, 'run_powershell', side_effect=manage), \
                    patch.object(watch_cleanup, 'run_cleanup_child', side_effect=child):
                self.assertEqual(watch_cleanup.main(), 130)
            self.assertEqual(self.status()['state'], 'interrupted')
            self.assertEqual(self.status()['failed'], [])
        finally:
            if helper_pid.exists():
                try:
                    os.kill(int(helper_pid.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == '__main__':
    unittest.main()
