import unittest

from scripts.label_transcripts import label_result, overlap_speakers


class LabelTranscriptTests(unittest.TestCase):
    def test_exclusive_assignment_keeps_ordinary_overlap_and_word_timestamps(self):
        raw = {"source": "track.m4a", "segments": [{"start": 1.0, "end": 2.0,
               "text": "слово", "words": [{"start": 1.2, "end": 1.8, "word": "слово"}]}]}
        diarization = {"turns": [{"start": 1.0, "end": 2.0, "speaker": "A"}],
                       "overlapping_turns": [
                           {"start": 1.0, "end": 2.0, "speaker": "A"},
                           {"start": 1.1, "end": 1.9, "speaker": "B"}]}
        result = label_result(raw, diarization)
        word = result["segments"][0]["words"][0]
        self.assertEqual(word["speaker"], "A")
        self.assertEqual(word["speakers"], ["A", "B"])
        self.assertTrue(word["overlapping_speech"])
        self.assertEqual((word["start"], word["end"]), (1.2, 1.8))

    def test_word_gets_speaker_with_greatest_overlap_and_reports_overlap(self):
        turns = [
            {"start": 0.0, "end": 1.0, "speaker": "speaker_01"},
            {"start": 0.4, "end": 0.9, "speaker": "speaker_02"},
        ]
        self.assertEqual(overlap_speakers(0.2, 0.8, turns), ["speaker_01", "speaker_02"])
        result = label_result(
            {"source": "recording.wav", "segments": [{"start": 0.2, "end": 0.8,
                "text": "hello", "words": [{"start": 0.2, "end": 0.8, "word": " hello"}]}]},
            {"turns": turns},
        )
        word = result["segments"][0]["words"][0]
        self.assertEqual(word["speaker"], "speaker_01")
        self.assertEqual(word["speakers"], ["speaker_01", "speaker_02"])
        self.assertEqual(result["segments"][0]["speaker"], "speaker_01")
        self.assertTrue(word["overlapping_speech"])
        self.assertTrue(result["segments"][0]["overlapping_speech"])

    def test_primary_uses_total_covered_duration_per_speaker(self):
        turns = [
            {"start": 0.0, "end": 0.2, "speaker": "speaker_A"},
            {"start": 0.2, "end": 0.4, "speaker": "speaker_A"},
            {"start": 0.1, "end": 0.4, "speaker": "speaker_B"},
        ]
        self.assertEqual(overlap_speakers(0.0, 0.5, turns), ["speaker_A", "speaker_B"])

    def test_turn_handoff_inside_word_is_not_overlap_speech(self):
        result = label_result(
            {"source": "recording.wav", "segments": [{"start": 0.8, "end": 1.2,
                "text": "word", "words": [{"start": 0.8, "end": 1.2, "word": " word"}]}]},
            {"turns": [
                {"start": 0, "end": 1, "speaker": "A"},
                {"start": 1, "end": 2, "speaker": "B"},
            ]},
        )
        word = result["segments"][0]["words"][0]
        self.assertEqual(word["speakers"], ["A", "B"])
        self.assertFalse(word["overlapping_speech"])

    def test_turn_boundaries_do_not_claim_touching_words(self):
        self.assertEqual(overlap_speakers(1.0, 1.2, [
            {"start": 0.0, "end": 1.0, "speaker": "speaker_01"},
            {"start": 1.2, "end": 2.0, "speaker": "speaker_02"},
        ]), [])

    def test_words_without_turns_are_unknown_and_form_a_group(self):
        result = label_result(
            {"source": "recording.wav", "segments": [{"start": 0.0, "end": 0.5,
                "text": "hi", "words": [{"start": 0.0, "end": 0.2, "word": " hi"},
                                             {"start": 0.2, "end": 0.5, "word": " there"}]}]},
            {"turns": []},
        )
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["speaker"], "unknown")
        self.assertEqual([word["speaker"] for word in result["segments"][0]["words"]],
                         ["unknown", "unknown"])

    def test_grouping_respects_speaker_changes_and_large_gaps(self):
        words = [
            {"start": 0.0, "end": 0.2, "word": " one"},
            {"start": 0.3, "end": 0.5, "word": " two"},
            {"start": 0.6, "end": 0.8, "word": " three"},
            {"start": 2.5, "end": 2.8, "word": " four"},
        ]
        asr = {"source": "recording.wav", "segments": [
            {"id": 1, "start": 0, "end": 0.2, "text": "one", "avg_logprob": -0.2,
             "review_recommended": False, "words": words[:2]},
            {"id": 2, "start": 0.6, "end": 0.8, "text": "three", "avg_logprob": -1.2,
             "review_recommended": True, "words": words[2:3]},
            {"id": 3, "start": 2.5, "end": 2.8, "text": "four", "words": words[3:]},
        ]}
        turns = [
            {"start": 0, "end": 0.5, "speaker": "A"},
            {"start": 0.5, "end": 1, "speaker": "B"},
            {"start": 2.5, "end": 3, "speaker": "A"},
        ]
        groups = label_result(asr, {"turns": turns})["segments"]
        self.assertEqual([g["text"] for g in groups], ["one two", "three", "four"])
        self.assertEqual([g["speaker"] for g in groups], ["A", "B", "A"])
        self.assertEqual(groups[0]["source_segments"][0]["avg_logprob"], -0.2)
        self.assertFalse(groups[0]["review_recommended"])
        self.assertTrue(groups[1]["review_recommended"])
        self.assertEqual(groups[1]["source_segments"][0]["id"], 2)

    def test_review_marker_remains_boolean_or_when_groups_merge(self):
        result = label_result(
            {"source": "recording.wav", "segments": [
                {"id": 1, "start": 0, "end": 0.2, "text": "a", "review_recommended": False,
                 "avg_logprob": -0.1, "words": [{"start": 0, "end": 0.2, "word": " a"}]},
                {"id": 2, "start": 0.3, "end": 0.5, "text": "b", "review_recommended": True,
                 "avg_logprob": -1.2, "words": [{"start": 0.3, "end": 0.5, "word": " b"}]},
            ]},
            {"turns": [{"start": 0, "end": 1, "speaker": "A"}]},
        )
        group = result["segments"][0]
        self.assertIs(group["review_recommended"], True)
        self.assertEqual(group["avg_logprob"], [-0.1, -1.2])

    def test_long_groups_break_at_sentence_end_after_ten_seconds(self):
        words = [
            {"start": 0, "end": 5, "word": " First"},
            {"start": 5, "end": 10.1, "word": " sentence."},
            {"start": 10.2, "end": 10.5, "word": " Next"},
        ]
        result = label_result(
            {"source": "recording.wav", "segments": [{"start": 0, "end": 10.5,
                "text": "First sentence. Next", "words": words}]},
            {"turns": [{"start": 0, "end": 30, "speaker": "A"}]},
        )
        self.assertEqual([s["text"] for s in result["segments"]], ["First sentence.", "Next"])

    def test_max_group_duration_caps_long_continuous_speech(self):
        words = [{"start": i * 2.0, "end": i * 2.0 + 0.5, "word": f" w{i}"}
                 for i in range(12)]
        result = label_result(
            {"source": "recording.wav", "segments": [{"start": 0, "end": 23.5,
                "text": "many words", "words": words}]},
            {"turns": [{"start": 0, "end": 30, "speaker": "A"}]},
        )
        self.assertTrue(all(s["end"] - s["start"] <= 20 for s in result["segments"]))

    def test_segment_without_words_uses_segment_timing_and_keeps_review_fields(self):
        result = label_result(
            {"source": "recording.wav", "segments": [{"id": 7, "start": 1.0, "end": 2.0,
                "text": "fallback", "avg_logprob": -0.9, "review_recommended": True,
                "words": []}]},
            {"turns": [{"start": 1.5, "end": 2.5, "speaker": "speaker_03"}]},
        )
        segment = result["segments"][0]
        self.assertEqual(segment["speaker"], "speaker_03")
        self.assertEqual(segment["avg_logprob"], -0.9)
        self.assertTrue(segment["review_recommended"])
        self.assertEqual(segment["source_segments"][0]["id"], 7)

    def test_no_word_fallback_segment_breaks_neighboring_word_groups(self):
        result = label_result(
            {"source": "recording.wav", "segments": [
                {"id": 1, "start": 0, "end": 0.2, "text": "first",
                 "words": [{"start": 0, "end": 0.2, "word": " first"}]},
                {"id": 2, "start": 0.3, "end": 0.4, "text": "fallback", "words": []},
                {"id": 3, "start": 0.5, "end": 0.7, "text": "last",
                 "words": [{"start": 0.5, "end": 0.7, "word": " last"}]},
            ]},
            {"turns": [{"start": 0, "end": 1, "speaker": "speaker_01"}]},
        )
        self.assertEqual([s["text"] for s in result["segments"]], ["first", "fallback", "last"])

    def test_rejected_segments_are_never_labeled(self):
        result = label_result(
            {"source": "recording.wav", "segments": [],
             "rejected_segments": [{"start": 0, "end": 1, "speaker": "wrong",
                                    "speakers": ["wrong"], "reason": "near_silent"}]},
            {"turns": [{"start": 0, "end": 1, "speaker": "speaker_01"}]},
        )
        self.assertNotIn("speaker", result["rejected_segments"][0])
        self.assertNotIn("speakers", result["rejected_segments"][0])

    def test_full_diarization_metadata_is_retained(self):
        result = label_result(
            {"source": "recording.wav", "segments": []},
            {"metadata": {"speaker_count": 3, "models": {"embedding": "model.onnx"},
                           "speaker_scope": "recording_global"}, "turns": []},
        )
        self.assertEqual(result["diarization"]["speaker_count"], 3)
        self.assertEqual(result["diarization"]["models"]["embedding"], "model.onnx")
        self.assertEqual(result["diarization"]["speaker_scope"], "recording_global")


if __name__ == "__main__":
    unittest.main()
