import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.diarization_audio import (
    SAMPLE_RATE,
    clip_turns_to_duration,
    merge_speech_regions,
    prepare_diarization_input,
    restore_turns,
)


class RegionTests(unittest.TestCase):
    def test_pads_clips_and_merges_word_spans_with_segment_fallback(self):
        raw = {"segments": [
            {"start": 0.0, "end": 0.2, "words": [
                {"start": 0.1, "end": 0.2},
                {"start": 0.4, "end": 0.8},
            ]},
            {"start": 1.4, "end": 1.5, "words": []},
            {"start": 2.8, "end": 3.2, "words": [{"start": 2.8, "end": 3.2}]},
        ]}

        regions = merge_speech_regions(raw, duration=3.0)

        self.assertEqual(regions, [(0.0, 1.8), (2.5, 3.0)])


class PrepareDiarizationInputTests(unittest.TestCase):
    def make_source(self, directory: str, duration: float = 20.0) -> Path:
        path = Path(directory) / "source.wav"
        samples = np.full((round(duration * SAMPLE_RATE),), 0.05, dtype=np.float32)
        sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")
        return path

    def test_sparse_track_is_compacted_and_reports_region_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(directory)
            output = Path(directory) / "compact.wav"
            raw = {
                "chunks": [{"offset": 0, "duration": 20, "gain_db": 6.0206}],
                "segments": [{"words": [
                    {"start": 2.0, "end": 2.5},
                    {"start": 15.0, "end": 15.5},
                ]}],
            }

            metadata = prepare_diarization_input(source, raw, output)

            info = sf.info(output)
            self.assertEqual((info.samplerate, info.channels, info.subtype),
                             (SAMPLE_RATE, 1, "PCM_16"))
            self.assertEqual(metadata["mode"], "sparse_asr_compact")
            self.assertAlmostEqual(metadata["coverage"], 2.2 / 20, places=3)
            self.assertEqual(len(metadata["regions"]), 2)
            self.assertAlmostEqual(metadata["regions"][0]["original_start"], 1.7)
            self.assertAlmostEqual(metadata["regions"][0]["original_end"], 2.8)
            self.assertAlmostEqual(metadata["regions"][1]["compact_start"], 1.6)
            self.assertAlmostEqual(metadata["prepared_duration_seconds"], 2.7, places=3)
            samples, _ = sf.read(output, dtype="float32")
            self.assertAlmostEqual(float(np.mean(samples[:1000])), 0.1, places=2)

    def test_full_coverage_keeps_entire_waveform_and_applies_chunk_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(directory, duration=10.0)
            output = Path(directory) / "full.wav"
            raw = {
                "chunks": [{"offset": 0, "duration": 10, "gain_db": 6.0206}],
                "segments": [{"words": [{"start": 1.0, "end": 4.0}]}],
            }

            metadata = prepare_diarization_input(source, raw, output)

            self.assertEqual(metadata["mode"], "full_recording")
            self.assertEqual(metadata["regions"], [])
            self.assertAlmostEqual(metadata["coverage"], 3.6 / 10.0, places=3)
            info = sf.info(output)
            self.assertAlmostEqual(info.duration, 10.0, places=3)
            samples, _ = sf.read(output, dtype="float32")
            self.assertAlmostEqual(float(np.mean(samples)), 0.1, places=2)


class RestoreTurnTests(unittest.TestCase):
    def test_sparse_turns_map_to_original_and_split_around_separator(self):
        diarization = {
            "metadata": {"input": "/tmp/compact.wav", "speaker_count": 2,
                         "models": {"embedding": "embed.onnx"}},
            "turns": [{"start": 0.8, "end": 1.9, "speaker": "speaker_01", "speaker_id": 1}],
        }
        preprocessing = {
            "mode": "sparse_asr_compact",
            "source_audio": "/recordings/original.m4a",
            "original_duration_seconds": 20.0,
            "regions": [
                {"compact_start": 0.0, "compact_end": 1.1,
                 "original_start": 1.7, "original_end": 2.8},
                {"compact_start": 1.6, "compact_end": 2.7,
                 "original_start": 14.7, "original_end": 15.8},
            ],
        }

        restored = restore_turns(diarization, preprocessing)

        self.assertEqual([(t["start"], t["end"]) for t in restored["turns"]],
                         [(2.5, 2.8), (14.7, 15.0)])
        self.assertEqual([t["speaker_id"] for t in restored["turns"]], [1, 1])
        self.assertEqual(restored["metadata"]["speaker_count"], 2)
        self.assertEqual(restored["metadata"]["input"], "/recordings/original.m4a")
        self.assertEqual(restored["metadata"]["diarization_input"], "/tmp/compact.wav")
        self.assertEqual(restored["metadata"]["models"]["embedding"], "embed.onnx")
        self.assertEqual(restored["metadata"]["duration_seconds"], 20.0)
        self.assertEqual(restored["metadata"]["speaker_scope"], "recording_global")
        self.assertEqual(restored["metadata"]["preprocessing"], preprocessing)
        self.assertEqual(diarization["turns"][0]["start"], 0.8)  # input remains unchanged

    def test_sparse_restores_both_exclusive_and_overlapping_turns(self):
        diarization = {"metadata": {},
                       "turns": [{"start": 0.0, "end": 0.5, "speaker": "A"}],
                       "overlapping_turns": [
                           {"start": 0.0, "end": 1.0, "speaker": "A"},
                           {"start": 0.2, "end": 1.0, "speaker": "B"}]}
        preprocessing = {"mode": "sparse_asr_compact", "regions": [
            {"compact_start": 0.0, "compact_end": 0.6, "original_start": 30.0},
            {"compact_start": 0.9, "compact_end": 1.2, "original_start": 80.0}]}
        restored = restore_turns(diarization, preprocessing)
        self.assertEqual(restored["turns"][0]["start"], 30.0)
        self.assertEqual([(t["start"], t["end"]) for t in restored["overlapping_turns"]],
                         [(30.0, 30.6), (30.2, 30.6), (80.0, 80.1), (80.0, 80.1)])
        self.assertEqual(diarization["overlapping_turns"][0]["start"], 0.0)

    def test_full_recording_clips_both_turn_lists_and_removes_empty_turns(self):
        diarization = {"metadata": {}, "turns": [
            {"start": -0.03, "end": 2.0, "speaker": "A"},
            {"start": 3.0, "end": 3.727, "speaker": "A"},
            {"start": 3.7, "end": 4.0, "speaker": "B"}],
            "overlapping_turns": [
                {"start": 3.0, "end": 3.727, "speaker": "A"},
                {"start": 3.7, "end": 4.0, "speaker": "B"},
                {"start": 1.0, "end": 1.0, "speaker": "B"}]}
        restored = restore_turns(diarization, {
            "mode": "full_recording", "original_duration_seconds": 3.648})
        self.assertEqual([(t["start"], t["end"]) for t in restored["turns"]],
                         [(0.0, 2.0), (3.0, 3.648)])
        self.assertEqual([(t["start"], t["end"]) for t in restored["overlapping_turns"]],
                         [(3.0, 3.648)])
        self.assertEqual(diarization["turns"][0]["start"], -0.03)

    def test_clip_already_restored_sparse_times_without_mapping_again(self):
        diarization = {"metadata": {"preprocessing": {"mode": "sparse_asr_compact",
                       "regions": [{"compact_start": 0, "compact_end": 1,
                                    "original_start": 80, "original_end": 81}]}},
                       "turns": [{"start": 80, "end": 81.01, "speaker": "A"}],
                       "overlapping_turns": [{"start": 80, "end": 81.01, "speaker": "A"}]}
        clipped = clip_turns_to_duration(diarization, 81)
        self.assertEqual(clipped["turns"], [{"start": 80, "end": 81, "speaker": "A"}])
        self.assertEqual(clipped["overlapping_turns"], clipped["turns"])
        self.assertEqual(clipped["metadata"], diarization["metadata"])

    def test_full_recording_preserves_turn_times(self):
        diarization = {"metadata": {"threshold": 0.7},
                       "turns": [{"start": 3.1, "end": 4.2, "speaker": "speaker_00"}]}
        preprocessing = {"mode": "full_recording", "original_duration_seconds": 9.0}

        restored = restore_turns(diarization, preprocessing)

        self.assertEqual(restored["turns"], diarization["turns"])
        self.assertEqual(restored["metadata"]["threshold"], 0.7)
        self.assertEqual(restored["metadata"]["input_duration_seconds"], 9.0)
        self.assertEqual(restored["metadata"]["speaker_scope"], "recording_global")


if __name__ == "__main__":
    unittest.main()
