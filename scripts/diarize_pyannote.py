"""Community-1 diarization in the isolated PyTorch environment."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'pyannote/speaker-diarization-community-1'
MODEL_REVISION = '3533c8cf8e369892e6b79ff1bf80f7b0286a54ee'
os.environ['PYANNOTE_METRICS_ENABLED'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / 'work' / 'huggingface'))


class CommunityDiarizer:
    """Load once and reuse the same local pipeline for each recording."""

    def __init__(self, device='cuda', max_speakers=6, revision=MODEL_REVISION):
        from dotenv import dotenv_values
        import torch
        from pyannote.audio import Pipeline

        values = {**dotenv_values(ROOT / '.env'), **os.environ}
        token = next((v for k, v in values.items()
                      if k.lower() in {'hf_token', 'hugging_face_hub_token', 'huggingface_token'} and v), None)
        if not token:
            raise RuntimeError('A valid HF_TOKEN is required for Community-1 setup')
        torch.set_num_threads(4)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('PyTorch CUDA is unavailable in the diarization environment')
        self.pipeline = Pipeline.from_pretrained(MODEL, token=token, revision=revision)
        self.pipeline.to(torch.device(device))
        self.device, self.max_speakers, self.revision = device, max_speakers, revision

    def diarize(self, input_path, track_name=None):
        import soundfile as sf
        import torch

        started = time.monotonic()
        samples, rate = sf.read(input_path, dtype='float32', always_2d=True)
        if len(samples) == 0:
            raise ValueError('Input audio is empty')
        if rate != 16000 or samples.shape[1] != 1:
            raise ValueError('Input must be mono 16 kHz WAV')
        print(f'Community-1: {len(samples) / rate:.1f} seconds on {self.device}', flush=True)
        output = self.pipeline({'waveform': torch.from_numpy(samples.T), 'sample_rate': rate},
                               min_speakers=1, max_speakers=self.max_speakers)
        labels = sorted(output.speaker_diarization.labels())
        label_ids = {label: index for index, label in enumerate(labels)}

        def turns(annotation):
            return [{'start': round(turn.start, 3), 'end': round(turn.end, 3),
                     'speaker': str(speaker), 'speaker_id': label_ids[speaker]}
                    for turn, _, speaker in annotation.itertracks(yield_label=True)]

        result = {'metadata': {
            'track': track_name or Path(input_path).stem, 'input': str(input_path),
            'duration_seconds': len(samples) / rate, 'sample_rate': rate,
            'speaker_scope': 'recording_global', 'speaker_count': len(labels),
            'model': MODEL, 'model_revision': self.revision,
            'pyannote_version': importlib.metadata.version('pyannote.audio'),
            'device': self.device, 'max_speakers': self.max_speakers,
            'alignment': 'exclusive_speaker_diarization',
            'elapsed_seconds': time.monotonic() - started,
        }, 'turns': turns(output.exclusive_speaker_diarization),
           'overlapping_turns': turns(output.speaker_diarization)}
        print(f'Community-1 complete: {len(labels)} speakers in {time.monotonic() - started:.1f}s', flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--track-name')
    parser.add_argument('--max-speakers', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    from transcribe import atomic_json
    diarizer = CommunityDiarizer(args.device, args.max_speakers)
    atomic_json(args.output, diarizer.diarize(args.input, args.track_name))


if __name__ == '__main__':
    main()
