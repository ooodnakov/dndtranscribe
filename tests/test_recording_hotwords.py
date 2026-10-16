import json
from pathlib import Path
import tempfile
import unittest

from scripts.select_recording_hotwords import (
    apply_decision, candidate_evidence, group_recordings, pack_budget, reusable_audit,
)


class RecordingHotwordsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def raw(self, name, text, complete=True):
        path = self.root / f'{name}.json'
        path.write_text(json.dumps({'complete': complete, 'segments': [
            {'id': 7, 'start': 2.0, 'end': 5.0, 'text': text}]}), encoding='utf-8')
        return path

    def test_group_paired_tracks_without_merging_other_recordings(self):
        self.raw('session_track-01', '')
        self.raw('session_track-02', '')
        self.raw('another_track-01', '')
        groups = group_recordings(self.root)
        self.assertEqual(len(groups['session']), 2)
        self.assertEqual(len(groups['another']), 1)

    def test_exact_fuzzy_and_semantic_alias_evidence_are_distinct(self):
        paths = [self.raw('a_track-01', 'Королевич здесь. Волшебнику нужна помощь. Город Луны далеко.')]
        entries = [{'термин': 'Королевич'}, {'термин': 'Волшебник'},
                   {'термин': 'Луноград', 'варианты': ['Город Луны']}]
        candidates = candidate_evidence(entries, paths)
        self.assertEqual(candidates[0]['exact_occurrences'], 1)
        self.assertEqual(candidates[1]['evidence'][0]['match'], 'near_spelling')
        self.assertEqual(candidates[2]['evidence'], [])

    def test_long_case_ending_on_short_name_is_retrieved(self):
        candidate = candidate_evidence([{'термин': 'Грэм'}],
                                      [self.raw('a', 'Мы знакомы с Грэмами.')])[0]
        self.assertEqual(candidate['occurrences'], 1)
        self.assertEqual(candidate['exact_occurrences'], 0)
        self.assertEqual(candidate['evidence'][0]['observed'], 'Грэмами')
        self.assertEqual(apply_decision(candidate, .79, .8), 'reject_model')

    def test_short_cyrillic_e_confusion_is_narrow_and_requires_model_gate(self):
        candidate = candidate_evidence([{'термин': 'Грэм'}],
                                      [self.raw('a', 'Грем здесь. Время пришло, берём вещи.')])[0]
        self.assertEqual(candidate['occurrences'], 1)
        self.assertEqual(candidate['exact_occurrences'], 0)
        self.assertEqual(candidate['evidence'][0]['observed'], 'Грем')
        self.assertEqual(apply_decision(candidate, .79, .8), 'reject_model')
        self.assertEqual(apply_decision(candidate, .81, .8), 'accept_model')

    def test_numbers_are_required_in_canonical_entity_names(self):
        entries = [{'термин': 'Объект 41'}]
        absent = candidate_evidence(entries, [self.raw('a', 'Объект 42 рядом с объектом.')])[0]
        self.assertEqual(absent['evidence'], [])
        found = candidate_evidence(entries, [self.raw('b', 'Мы видели Объект 41.')])[0]
        self.assertEqual(found['exact_occurrences'], 1)

    def test_short_name_inflection_and_word_boundaries(self):
        candidate = candidate_evidence([{'термин': 'Грэм'}],
                                      [self.raw('a', 'Говорю с Грэмом. Грэма не видно.')])[0]
        self.assertEqual(candidate['occurrences'], 2)
        self.assertEqual(candidate['exact_occurrences'], 0)
        absent = candidate_evidence([{'термин': 'Иона'}],
                                   [self.raw('b', 'Эта региональная группа.')])[0]
        self.assertEqual(absent['evidence'], [])

    def test_evidence_is_bounded_but_preserves_matched_word(self):
        text = 'начало ' * 100 + 'Волшебник' + ' конец' * 100
        candidate = candidate_evidence([{'термин': 'Волшебник'}], [self.raw('a', text)])[0]
        evidence = candidate['evidence'][0]
        self.assertIn('Волшебник', evidence['context'])
        self.assertLessEqual(len(evidence['context']), 370)
        self.assertEqual(evidence['segment_id'], 7)

    def test_incomplete_asr_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            candidate_evidence([], [self.raw('a', '', complete=False)])

    def test_exact_evidence_uses_lower_threshold_but_still_requires_model(self):
        self.assertEqual(apply_decision({'exact_occurrences': 1}, .1, .8), 'reject_exact_model')
        self.assertEqual(apply_decision({'exact_occurrences': 1}, .6404, .8), 'reject_exact_model')
        self.assertEqual(apply_decision({'exact_occurrences': 1}, .6404, .8, .5), 'accept_exact_model')
        self.assertEqual(apply_decision({'exact_occurrences': 1}, .71, .8), 'accept_exact_model')
        self.assertEqual(apply_decision({'exact_occurrences': 0}, .79, .8), 'reject_model')
        with self.assertRaises(ValueError):
            apply_decision({'exact_occurrences': 1}, float('nan'), .8)

    def test_budget_accounts_for_full_join_and_audits_exclusions(self):
        decisions = [dict(term='name', exact_occurrences=1, probability=.9,
                          occurrences=2, decision='retain_exact_evidence'),
                     dict(term='other', exact_occurrences=0, probability=.99,
                          occurrences=1, decision='accept_model')]
        text, count = pack_budget(decisions, len, 8)
        self.assertEqual((text, count), ('name', 5))
        self.assertEqual(decisions[1]['selection'], 'token_budget_excluded')

    def test_cache_requires_all_fingerprints_to_match(self):
        path = self.root / 'audit.json'
        identity = {'raw': 'a', 'selector': 'x', 'tokenizer': 'y', 'threshold': .8}
        path.write_text(json.dumps({'fingerprint': identity, 'complete': True}))
        self.assertIsNotNone(reusable_audit(path, dict(identity)))
        for key in identity:
            changed = dict(identity, **{key: 'changed'})
            self.assertIsNone(reusable_audit(path, changed))
        path.write_text('{broken')
        self.assertIsNone(reusable_audit(path, identity))


if __name__ == '__main__':
    unittest.main()
