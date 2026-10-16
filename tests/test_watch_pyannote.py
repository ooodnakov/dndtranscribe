import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.watch_pyannote import batch_state, run_batch


class CommunityBatchTests(unittest.TestCase):
    def fixture(self, root, names=('one', 'two')):
        (root / 'audio').mkdir()
        (root / 'transcripts' / 'raw').mkdir(parents=True)
        for name in names:
            (root / 'audio' / (name + '.m4a')).write_bytes(b'original audio')
            raw = {'complete': True, 'duration': 100, 'source': str(root / 'audio' / (name + '.m4a')),
                   'segments': [{'start': 10, 'end': 11, 'text': 'слово', 'words': [
                       {'start': 10, 'end': 11, 'word': 'слово'}]}]}
            (root / 'transcripts' / 'raw' / (name + '.json')).write_text(json.dumps(raw))

    @staticmethod
    def prepare(source, raw, wav):
        wav.touch()
        return {'mode': 'sparse_asr_compact', 'source_audio': str(source),
                'prepared_duration_seconds': 1, 'original_duration_seconds': 100,
                'regions': [{'compact_start': 0, 'compact_end': 1, 'original_start': 10}]}

    def test_pipeline_reused_resume_skips_and_changed_source_reprocesses(self):
        calls, loads = [], []

        class FakeDiarizer:
            def __init__(self, **kwargs):
                loads.append(kwargs)

            def diarize(self, wav, name):
                calls.append(name)
                return {'metadata': {'speaker_count': 1},
                        'turns': [{'start': 0, 'end': 1, 'speaker': 'A'}],
                        'overlapping_turns': [{'start': 0, 'end': 1, 'speaker': 'A'}]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            config = {'device': 'cpu', 'max_speakers': 6}
            with patch('scripts.watch_pyannote.prepare_diarization_input', self.prepare):
                self.assertEqual(run_batch(root, config, True, FakeDiarizer), 0)
                self.assertEqual(len(loads), 1)
                self.assertEqual(calls, ['one', 'two'])
                self.assertEqual(run_batch(root, config, True, FakeDiarizer), 0)
                self.assertEqual(len(loads), 1)
                (root / 'audio' / 'one.m4a').write_bytes(b'changed audio')
                self.assertEqual(run_batch(root, config, True, FakeDiarizer), 0)
                self.assertEqual(calls, ['one', 'two', 'one'])
                # Recover labeling without loading the model again.
                (root / 'transcripts/labeled-community1/two.json').unlink()
                self.assertEqual(run_batch(root, config, True, FakeDiarizer), 0)
                self.assertEqual(calls, ['one', 'two', 'one'])
                # A raw transcript change invalidates even unchanged audio.
                raw_path = root / 'transcripts/raw/one.json'
                raw = json.loads(raw_path.read_text())
                raw['segments'][0]['words'][0]['word'] = 'исправлено'
                raw_path.write_text(json.dumps(raw))
                self.assertEqual(run_batch(root, config, True, FakeDiarizer), 0)
                self.assertEqual(calls[-1], 'one')
                self.assertEqual(len(calls), 4)
                # Configuration changes invalidate both track caches.
                changed_config = {**config, 'model_revision': 'new revision'}
                self.assertEqual(run_batch(root, changed_config, True, FakeDiarizer), 0)
                self.assertEqual(len(calls), 6)
            result = json.loads((root / 'transcripts/labeled-community1/one.json').read_text())
            self.assertEqual(result['segments'][0]['words'][0]['start'], 10)
            diar = json.loads((root / 'transcripts/diarization-community1/one.json').read_text())
            self.assertEqual(diar['overlapping_turns'][0]['start'], 10)
            self.assertFalse((root / 'transcripts' / 'labeled').exists())
            self.assertFalse(list((root / 'work/community1-input').glob('*.wav')))

    def test_failures_do_not_stop_other_files_and_are_retried_on_restart(self):
        class BrokenDiarizer:
            def __init__(self, **kwargs):
                pass

            def diarize(self, wav, name):
                raise RuntimeError('synthetic failure')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch('scripts.watch_pyannote.prepare_diarization_input', self.prepare):
                self.assertEqual(run_batch(root, {'device': 'cpu', 'max_speakers': 6},
                                           True, BrokenDiarizer), 1)
            status = json.loads((root / 'work/community1_status.json').read_text())
            self.assertEqual(status['state'], 'complete_with_errors')
            self.assertEqual(set(status['failed']), {'one', 'two'})
            self.assertEqual(status['outstanding'], [])

    def test_missing_asr_cannot_report_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, ('one',))
            (root / 'transcripts/raw/one.json').unlink()
            self.assertEqual(run_batch(root, {'device': 'cpu', 'max_speakers': 6}, True), 1)
            status = json.loads((root / 'work/community1_status.json').read_text())
            self.assertEqual(status['state'], 'incomplete')
            self.assertEqual(status['outstanding'], ['one'])

    def test_complete_only_when_all_expected_inputs_accounted_for(self):
        self.assertEqual(batch_state({'a', 'b'}, {'a': 1}, {})[0], 'waiting_for_asr')
        self.assertEqual(batch_state({'a', 'b'}, {'a': 1}, {'b': 'error'})[0], 'complete_with_errors')
        self.assertEqual(batch_state({'a', 'b'}, {'a': 1, 'b': 0}, {})[0], 'complete')


if __name__ == '__main__':
    unittest.main()
