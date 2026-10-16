import tempfile
import unittest
from pathlib import Path

from scripts.transcribe import prepare_export, render, timestamp


def segment(text, no_speech_prob, *, start=0.0, end=1.0, words=None):
    return {
        "start": start,
        "end": end,
        "text": text,
        "no_speech_prob": no_speech_prob,
        "words": words or [],
    }


class PrepareExportTests(unittest.TestCase):
    def test_high_no_speech_stock_captions_move_to_rejected(self):
        captions = [
            segment("Субтитры создавал DimaTorzok", 0.835),
            segment("Продолжение следует...", 0.916),
        ]
        result = {"duration": 10.0, "segments": captions}

        prepare_export(result)

        self.assertEqual(result["segments"], [])
        self.assertEqual(result["rejected_segments"], captions)
        self.assertTrue(all(s["rejection_reason"] ==
                            "suspected_whisper_stock_caption_on_non_speech"
                            for s in result["rejected_segments"]))

    def test_same_stock_caption_with_low_no_speech_stays_in_transcript(self):
        result = {"duration": 10.0,
                  "segments": [segment("Субтитры создавал DimaTorzok", 0.2)]}

        prepare_export(result)

        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["rejected_segments"] if "rejected_segments" in result else [], [])
        self.assertFalse(result["segments"][0]["review_recommended"])

    def test_high_no_speech_real_dialogue_is_kept_and_flagged_for_review(self):
        result = {"duration": 10.0,
                  "segments": [segment("Я сейчас вернусь.", 0.835)]}

        prepare_export(result)

        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["text"], "Я сейчас вернусь.")
        self.assertIs(result["segments"][0]["review_recommended"], True)
        self.assertEqual(result.get("rejected_segments", []), [])

    def test_existing_review_flag_is_normalized_to_boolean(self):
        result = {"duration": 10.0,
                  "segments": [{**segment("Текст", 0.1), "review_recommended": 1}]}
        prepare_export(result)
        self.assertIs(result["segments"][0]["review_recommended"], True)


class TimestampAndRenderTests(unittest.TestCase):
    def test_clamps_segment_and_final_word_to_media_duration(self):
        result = {"duration": 12.0, "segments": [{
            "start": 11.9,
            "end": 12.2,
            "text": "конец",
            "words": [
                {"start": 11.9, "end": 12.15, "word": " конец"},
                {"start": 12.15, "end": 12.4, "word": "."},
            ],
        }]}

        prepare_export(result)

        output = result["segments"][0]
        self.assertEqual(output["end"], 12.0)
        self.assertEqual(output["words"][-1]["end"], 12.0)
        self.assertEqual(output["words"][-1]["start"], 12.0)

    def test_srt_timestamp_rounding_carries_across_minute_and_hour(self):
        self.assertEqual(timestamp(59.9996, srt=True), "00:01:00,000")
        self.assertEqual(timestamp(3599.9996, srt=True), "01:00:00,000")

    def test_render_writes_rounded_srt_boundaries(self):
        result = {"source": "/recordings/session.wav", "segments": [{
            "start": 59.9996, "end": 60.9996, "text": "переход",
        }]}
        with tempfile.TemporaryDirectory() as directory:
            render(result, Path(directory))
            subtitle = (Path(directory) / "session.srt").read_text()
        self.assertIn("00:01:00,000 --> 00:01:01,000", subtitle)
        self.assertIn("переход", subtitle)


if __name__ == "__main__":
    unittest.main()
