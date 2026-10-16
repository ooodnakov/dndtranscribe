import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.transcribe import asr_signature, load_hotword_profiles, output_paths


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        # Deterministic token boundaries for testing the cutoff without a model.
        return SimpleNamespace(ids=text.split())


class GuidedTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'session_track-01.m4a'
        self.source.write_bytes(b'audio')

    def test_unguided_signature_matches_original_serialization(self):
        source = self.source
        original = {'source': str(source), 'size': source.stat().st_size,
                    'mtime_ns': source.stat().st_mtime_ns, 'chunk': 600,
                    'model': 'large-v3', 'gain_db': 20, 'beam': 5, 'pipeline_version': 2}
        expected = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        self.assertEqual(asr_signature(source, 600, 20), expected)

    def test_guidance_content_and_policy_invalidate_cache(self):
        baseline = asr_signature(self.source, 600, 20)
        guidance = {'text': 'Грэм', 'sha256': 'a', 'token_count': 2, 'policy_version': 1}
        guided = asr_signature(self.source, 600, 20, guidance)
        self.assertNotEqual(baseline, guided)
        for key, value in [('text', 'Мира'), ('sha256', 'b'), ('policy_version', 2)]:
            self.assertNotEqual(guided, asr_signature(self.source, 600, 20,
                                                     dict(guidance, **{key: value})))

    def test_paired_tracks_share_profile_and_empty_control_is_allowed(self):
        second = self.root / 'session_track-02.m4a'
        (self.root / 'session.txt').write_text(' Грэм\n', encoding='utf-8')
        profiles = load_hotword_profiles([self.source, second], self.root, Tokenizer())
        self.assertEqual(profiles[self.source], profiles[second])
        self.assertEqual(profiles[self.source]['text'], 'Грэм')
        (self.root / 'session.txt').write_text('', encoding='utf-8')
        empty = load_hotword_profiles([self.source], self.root, Tokenizer())[self.source]
        self.assertEqual(empty['token_count'], 0)
        self.assertEqual(empty['text'], '')

    def test_missing_profile_fails_instead_of_silently_omitting_hints(self):
        with self.assertRaises(FileNotFoundError):
            load_hotword_profiles([self.source], self.root, Tokenizer())

    def test_token_budget_boundary(self):
        path = self.root / 'session.txt'
        path.write_text(' '.join(['name'] * 223), encoding='utf-8')
        profiles = load_hotword_profiles([self.source], self.root, Tokenizer())
        self.assertEqual(profiles[self.source]['token_count'], 223)
        path.write_text(' '.join(['name'] * 224), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '224 Whisper tokens'):
            load_hotword_profiles([self.source], self.root, Tokenizer())

    def test_guided_defaults_and_baseline_protection(self):
        output, status = output_paths(None, None, True, self.root)
        self.assertEqual(output, self.root / 'transcripts' / 'raw-glossary')
        self.assertEqual(status, self.root / 'work' / 'asr_glossary_status.json')
        baseline = self.root / 'transcripts' / 'raw'
        for path in [baseline, baseline / 'nested']:
            with self.assertRaisesRegex(ValueError, 'outside transcripts/raw'):
                output_paths(path, None, True, self.root)
        with self.assertRaisesRegex(ValueError, 'separate status'):
            output_paths(None, self.root / 'work' / 'asr_status.json', True, self.root)
        self.assertEqual(output_paths(None, None, False, self.root),
                         (baseline, self.root / 'work' / 'asr_status.json'))


if __name__ == '__main__':
    unittest.main()
