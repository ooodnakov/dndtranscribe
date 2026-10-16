import unittest
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from scripts import cleanup_transcript as cleanup
from scripts.cleanup_transcript import validate_proposal, validate_reasoning_output


class CleanupValidationTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            'id': 7,
            'start': 1.25,
            'end': 2.5,
            'speaker': 'speaker_01',
            'source_ids': [3, 4],
            'text': 'привет мир',
        }

    def proposal(self, **changes):
        result = dict(self.expected)
        result.update(changes)
        return result

    def test_punctuation_and_case_edits_are_accepted(self):
        accepted, reason, _ = validate_proposal(
            self.expected, self.proposal(text='Привет, мир!')
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, 'accepted')

    def test_lexical_changes_are_rejected(self):
        accepted, reason, _ = validate_proposal(
            self.expected, self.proposal(text='привет мыр')
        )
        self.assertFalse(accepted)
        self.assertIn('lexical word changed', reason)

    def test_identity_and_timing_fields_are_immutable(self):
        accepted, reason, _ = validate_proposal(
            self.expected, self.proposal(start=1.3)
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, 'start changed')

    def test_exact_response_schema_is_required(self):
        accepted, reason, _ = validate_proposal(
            self.expected, self.proposal(extra='field')
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, 'response fields differ from the required schema')

    def test_reasoning_tokens_are_rejected(self):
        valid, reason = validate_reasoning_output(
            {'reasoning_output_tokens': 1}, []
        )
        self.assertFalse(valid)
        self.assertIn('reasoning tokens', reason)

    def test_reasoning_blocks_are_rejected(self):
        valid, reason = validate_reasoning_output(
            {}, [{'type': 'reasoning'}]
        )
        self.assertFalse(valid)
        self.assertIn('reasoning block', reason)


class CleanupBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.config = {'model': 'test-model', 'api_url': 'http://localhost:1234/api/v1/chat'}
        self.record = cleanup.stable_record(0, {
            'start': 0, 'end': 1, 'speaker': 'speaker_01', 'text': 'привет мир',
        })

    def call_with_message(self, message):
        def bridge(args, **kwargs):
            self.payload = json.loads(Path(args[6]).read_text(encoding='utf-8'))
            Path(args[7]).write_text(json.dumps({
                'output': [{'type': 'message', 'content': message}],
                'stats': {'reasoning_output_tokens': 0},
            }), encoding='utf-8')
            return subprocess.CompletedProcess(args, 0, '', '')

        with patch.object(cleanup.shutil, 'which', return_value='powershell'), \
                patch.object(cleanup, 'windows_path', side_effect=str), \
                patch.object(cleanup.subprocess, 'run', side_effect=bridge):
            return cleanup.call_lmstudio(self.config, self.record, self.work)

    def test_native_prompt_is_separate_from_record_and_sampling_is_greedy(self):
        proposal, stats = self.call_with_message(json.dumps(self.record))
        self.assertEqual(proposal, self.record)
        self.assertEqual(stats['reasoning_output_tokens'], 0)
        self.assertEqual(json.loads(self.payload['input']), self.record)
        self.assertIn('только пунктуацию, регистр букв и пробелы', self.payload['system_prompt'])
        self.assertIn('Не используй Markdown', self.payload['system_prompt'])
        self.assertEqual(self.payload['temperature'], 0)
        self.assertEqual(self.payload['reasoning'], 'off')
        self.assertFalse(self.payload['store'])
        self.assertEqual(list(self.work.iterdir()), [])

    def test_markdown_fenced_json_remains_a_retryable_parse_error(self):
        with self.assertRaisesRegex(ValueError, 'model response was not a JSON object'):
            self.call_with_message('```json\n' + json.dumps(self.record) + '\n```')
        self.assertEqual(list(self.work.iterdir()), [])


class CleanupRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source.json'
        self.output = self.root / 'cleaned'
        self.config = {'model': 'test-model', 'api_url': 'http://localhost:1234/api/v1/chat'}
        self.data = {
            'complete': True,
            'segments': [
                {'start': index, 'end': index + 1, 'text': 'привет мир',
                 'review_recommended': True, 'speaker': 'speaker_01'}
                for index in range(2)
            ],
        }
        self.write_source()
        self.root_patch = patch.object(cleanup, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def write_source(self):
        self.source.write_text(json.dumps(self.data, ensure_ascii=False), encoding='utf-8')

    def response(self, config, record, work_dir):
        return {**record, 'text': 'Привет, мир!'}, {}

    def run_cleanup(self, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cleanup.cleanup_source(self.source, self.output, self.config, **kwargs)

    def test_interrupted_resume_keeps_successful_segment(self):
        with patch.object(cleanup, 'call_lmstudio', side_effect=[
            self.response(self.config, cleanup.stable_record(0, self.data['segments'][0]), self.root),
            KeyboardInterrupt(),
        ]):
            with self.assertRaises(KeyboardInterrupt):
                self.run_cleanup()
        self.assertFalse(cleanup.cleanup_complete(self.source, self.output, self.config))
        with patch.object(cleanup, 'call_lmstudio', side_effect=self.response) as request:
            self.assertEqual(self.run_cleanup(), 0)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[1]['id'], 1)
        result = json.loads((self.output / 'source.json').read_text(encoding='utf-8'))
        self.assertEqual([s['text'] for s in result['segments']], ['Привет, мир!'] * 2)
        self.assertTrue(cleanup.cleanup_complete(self.source, self.output, self.config))
        self.assertEqual(json.loads(self.source.read_text(encoding='utf-8')), self.data)

    def test_malformed_cached_decisions_are_retried_before_applying_edits(self):
        def remove_proposal(item):
            item.pop('proposal')

        def remove_text(item):
            item['proposal'].pop('text')

        def change_words(item):
            item['proposal']['text'] = 'совсем другое'

        def change_timing(item):
            item['proposal']['start'] = 100

        def change_record(item):
            item['text'] = 'поврежденный кеш'

        def malformed_status(item):
            item['status'] = ['accepted']

        for index, corrupt in enumerate((remove_proposal, remove_text, change_words,
                                         change_timing, change_record, malformed_status)):
            with self.subTest(corruption=corrupt.__name__):
                self.output = self.root / f'corrupt-{index}'
                first_response = self.response(
                    self.config, cleanup.stable_record(0, self.data['segments'][0]), self.root
                )
                with patch.object(cleanup, 'call_lmstudio', side_effect=[first_response, KeyboardInterrupt()]):
                    with self.assertRaises(KeyboardInterrupt):
                        self.run_cleanup()
                checkpoint_path = self.output / '.source.cleanup.checkpoint.json'
                checkpoint = json.loads(checkpoint_path.read_text())
                corrupt(checkpoint['proposals'][0])
                checkpoint_path.write_text(json.dumps(checkpoint))
                with patch.object(cleanup, 'call_lmstudio', side_effect=self.response) as request:
                    self.assertEqual(self.run_cleanup(), 0)
                self.assertEqual(request.call_count, 2)
                result = json.loads((self.output / 'source.json').read_text())
                self.assertEqual([s['text'] for s in result['segments']], ['Привет, мир!'] * 2)
                self.assertTrue(cleanup.cleanup_complete(self.source, self.output, self.config))

    def test_cached_decision_reason_is_reconstructed_on_resume(self):
        with patch.object(cleanup, 'call_lmstudio', side_effect=[
            self.response(self.config, cleanup.stable_record(0, self.data['segments'][0]), self.root),
            KeyboardInterrupt(),
        ]):
            with self.assertRaises(KeyboardInterrupt):
                self.run_cleanup()
        checkpoint_path = self.output / '.source.cleanup.checkpoint.json'
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint['proposals'][0].pop('reason')
        checkpoint_path.write_text(json.dumps(checkpoint))
        with patch.object(cleanup, 'call_lmstudio', side_effect=self.response) as request:
            self.assertEqual(self.run_cleanup(), 0)
        self.assertEqual(request.call_count, 1)
        audit = json.loads((self.output / 'source.audit.json').read_text())
        self.assertEqual(audit['proposals'][0]['reason'], 'accepted')

    def test_extreme_vowel_runs_are_preserved_for_manual_review_without_requests(self):
        self.data['segments'] = [{
            'start': 2.75, 'end': 9.5, 'speaker': 'speaker_01',
            'text': 'Г' + 'о' * 223, 'review_recommended': True,
            'source_segments': [{'id': 17}],
        }]
        self.write_source()
        source_bytes = self.source.read_bytes()
        with patch.object(cleanup, 'call_lmstudio') as request:
            self.assertEqual(self.run_cleanup(), 0)
            # Remove the published set to exercise checkpoint recovery as well.
            for extension in ('json', 'txt', 'srt', 'audit.json'):
                (self.output / f'source.{extension}').unlink()
            self.assertEqual(self.run_cleanup(), 0)
        request.assert_not_called()
        self.assertEqual(self.source.read_bytes(), source_bytes)
        cleaned = json.loads((self.output / 'source.json').read_text())
        self.assertEqual(cleaned['segments'], self.data['segments'])
        self.assertEqual(cleaned['cleanup']['manual_review'], 1)
        audit = json.loads((self.output / 'source.audit.json').read_text())
        self.assertEqual(audit['manual_review'], 1)
        self.assertEqual(audit['errors'], 0)
        decision = audit['proposals'][0]
        self.assertEqual(decision['status'], 'manual_review')
        self.assertIsNone(decision['proposal'])
        self.assertFalse(decision['request_sent'])
        self.assertIn('manual audio review required', decision['reason'])
        self.assertTrue(cleanup.cleanup_complete(self.source, self.output, self.config))

    def test_vowel_run_policy_is_narrow_and_case_insensitive(self):
        for text in ('а' * 32, 'Аа' * 16, 'e' * 32, 'Ёё' * 16):
            with self.subTest(text=text):
                self.assertIsNotNone(cleanup.REPEATED_VOWEL_RE.search(text))
        ordinary = ['да да да', 'о' * 31, 'р' * 40]
        self.data['segments'] = [
            {'start': index, 'end': index + 1, 'text': text, 'review_recommended': True}
            for index, text in enumerate(ordinary)
        ]
        self.write_source()
        with patch.object(cleanup, 'call_lmstudio', side_effect=lambda config, record, work: (record, {})) as request:
            self.assertEqual(self.run_cleanup(), 0)
        self.assertEqual(request.call_count, len(ordinary))
        cleaned = json.loads((self.output / 'source.json').read_text())
        self.assertEqual(cleaned['cleanup']['manual_review'], 0)

    def test_errors_retry_but_successes_and_guardrail_rejections_do_not(self):
        def first(config, record, work_dir):
            if record['id'] == 0:
                raise RuntimeError('model lookup failed')
            return {**record, 'text': 'привет мыр'}, {}

        with patch.object(cleanup, 'call_lmstudio', side_effect=first):
            self.assertEqual(self.run_cleanup(), 2)
        self.assertFalse((self.output / 'source.audit.json').exists())
        checkpoint = json.loads((self.output / '.source.cleanup.checkpoint.json').read_text())
        self.assertEqual([p['status'] for p in checkpoint['proposals']], ['error', 'rejected'])
        with patch.object(cleanup, 'call_lmstudio', side_effect=self.response) as request:
            self.assertEqual(self.run_cleanup(), 0)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[1]['id'], 0)
        audit = json.loads((self.output / 'source.audit.json').read_text())
        self.assertEqual([p['status'] for p in audit['proposals']], ['accepted', 'rejected'])
        self.assertEqual(audit['errors'], 0)

    def test_invalid_json_is_retryable_and_main_returns_error(self):
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(self.config))
        with patch.object(cleanup, 'call_lmstudio', side_effect=ValueError('model response was not a JSON object')):
            with patch.object(sys, 'argv', ['cleanup_transcript', str(self.source),
                                           '--output-dir', str(self.output), '--config', str(config_path)]):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(cleanup.main(), 2)
        with patch.object(cleanup, 'call_lmstudio', side_effect=self.response) as request:
            self.assertEqual(self.run_cleanup(), 0)
        self.assertEqual(request.call_count, 2)

    def test_matching_completed_output_is_reused(self):
        with patch.object(cleanup, 'call_lmstudio', side_effect=self.response):
            self.assertEqual(self.run_cleanup(), 0)
        with patch.object(cleanup, 'call_lmstudio') as request:
            self.assertEqual(self.run_cleanup(), 0)
        request.assert_not_called()

    def test_source_model_config_and_mode_changes_invalidate_decisions(self):
        with patch.object(cleanup, 'call_lmstudio', side_effect=self.response) as request:
            self.assertEqual(self.run_cleanup(), 0)
            self.config['model'] = 'new-model'
            self.assertFalse(cleanup.cleanup_complete(self.source, self.output, self.config))
            self.assertEqual(self.run_cleanup(overwrite=True), 0)
            self.config['max_output_tokens'] = 1500
            self.assertFalse(cleanup.cleanup_complete(self.source, self.output, self.config))
            self.assertEqual(self.run_cleanup(overwrite=True), 0)
            self.data['segments'][0]['text'] = 'привет мир!'
            self.write_source()
            self.assertFalse(cleanup.cleanup_complete(self.source, self.output, self.config))
            self.assertEqual(self.run_cleanup(overwrite=True), 0)
            self.assertFalse(cleanup.cleanup_complete(self.source, self.output, self.config, True))
            self.assertEqual(self.run_cleanup(overwrite=True, process_all=True), 0)
        self.assertEqual(request.call_count, 10)

    def test_code_and_bridge_changes_invalidate_fingerprint(self):
        code = self.root / 'cleanup.py'
        bridge = self.root / 'bridge.ps1'
        code.write_text('code v1')
        bridge.write_text('bridge v1')
        with patch.object(cleanup, '__file__', str(code)), patch.object(cleanup, 'POWERSHELL_BRIDGE', bridge):
            first = cleanup.cleanup_fingerprint(self.source, self.config)
            code.write_text('code v2')
            second = cleanup.cleanup_fingerprint(self.source, self.config)
            self.assertNotEqual(first, second)
            bridge.write_text('bridge v2')
            self.assertNotEqual(second, cleanup.cleanup_fingerprint(self.source, self.config))

    def test_missing_or_modified_artifact_is_incomplete(self):
        with patch.object(cleanup, 'call_lmstudio', side_effect=self.response):
            self.assertEqual(self.run_cleanup(), 0)
        for extension in ('json', 'txt', 'srt'):
            artifact = self.output / f'source.{extension}'
            original = artifact.read_bytes()
            artifact.unlink()
            self.assertFalse(cleanup.cleanup_complete(self.source, self.output, self.config))
            artifact.write_bytes(original + b'changed')
            self.assertFalse(cleanup.cleanup_complete(self.source, self.output, self.config))
            artifact.write_bytes(original)
            self.assertTrue(cleanup.cleanup_complete(self.source, self.output, self.config))

    def test_legacy_and_error_audits_are_incomplete(self):
        self.output.mkdir()
        audit_path = self.output / 'source.audit.json'
        audit_path.write_text('{}')
        self.assertFalse(cleanup.cleanup_complete(self.source, self.output, self.config))
        with patch.object(cleanup, 'call_lmstudio', side_effect=self.response):
            self.assertEqual(self.run_cleanup(overwrite=True), 0)
        audit = json.loads(audit_path.read_text())
        audit['errors'] = 1
        audit_path.write_text(json.dumps(audit))
        self.assertFalse(cleanup.cleanup_complete(self.source, self.output, self.config))

    def test_publication_failure_retires_old_marker_and_resume_reuses_decisions(self):
        with patch.object(cleanup, 'call_lmstudio', side_effect=self.response):
            self.assertEqual(self.run_cleanup(), 0)
        self.config['model'] = 'replacement'
        original_write = cleanup.atomic_write
        writes = []

        def fail_srt(path, content):
            writes.append(path.name)
            if path.suffix == '.srt':
                raise OSError('simulated disk failure')
            original_write(path, content)

        with patch.object(cleanup, 'call_lmstudio', side_effect=self.response):
            with patch.object(cleanup, 'atomic_write', side_effect=fail_srt):
                with self.assertRaises(OSError):
                    self.run_cleanup(overwrite=True)
        self.assertFalse((self.output / 'source.audit.json').exists())
        self.assertFalse(cleanup.cleanup_complete(self.source, self.output, self.config))
        with patch.object(cleanup, 'call_lmstudio') as request:
            with patch.object(cleanup, 'atomic_write', wraps=original_write) as write:
                self.assertEqual(self.run_cleanup(overwrite=True), 0)
        request.assert_not_called()
        self.assertEqual(write.call_args.args[0].name, 'source.audit.json')

    def test_existing_outputs_require_overwrite_before_requests(self):
        self.output.mkdir()
        (self.output / 'source.txt').write_text('preserved')
        with patch.object(cleanup, 'call_lmstudio') as request:
            with self.assertRaises(FileExistsError):
                self.run_cleanup()
        request.assert_not_called()
        self.assertEqual((self.output / 'source.txt').read_text(), 'preserved')

    def test_output_cannot_replace_source_even_with_overwrite(self):
        original = self.source.read_bytes()
        alias = self.root / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        for output_dir in (self.source.parent, alias):
            with self.subTest(output_dir=output_dir):
                with patch.object(cleanup, 'call_lmstudio') as request:
                    with patch.object(cleanup, 'source_lock') as lock:
                        with self.assertRaisesRegex(ValueError, 'must differ from the source'):
                            cleanup.cleanup_source(self.source, output_dir, self.config, overwrite=True)
                request.assert_not_called()
                lock.assert_not_called()
                self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse((self.root / 'work').exists())

    def test_source_lock_blocks_another_process_and_is_released(self):
        code = (
            'from pathlib import Path\n'
            'from scripts import cleanup_transcript as c\n'
            f'c.ROOT = Path({str(self.root)!r})\n'
            'try:\n'
            f'    with c.source_lock(Path({str(self.source)!r})):\n'
            '        pass\n'
            'except RuntimeError:\n'
            '    raise SystemExit(7)\n'
        )
        with cleanup.source_lock(self.source):
            blocked = subprocess.run([sys.executable, '-c', code], cwd=Path(cleanup.__file__).parents[1], capture_output=True)
        self.assertEqual(blocked.returncode, 7, blocked.stderr)
        released = subprocess.run([sys.executable, '-c', code], cwd=Path(cleanup.__file__).parents[1], capture_output=True)
        self.assertEqual(released.returncode, 0, released.stderr)


if __name__ == '__main__':
    unittest.main()
