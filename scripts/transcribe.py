"""Resumable local Russian ASR. Each input track retains its own timeline."""
import argparse
import dataclasses
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOTWORDS_POLICY_VERSION = 1


def asr_signature(source, chunk_seconds, max_gain_db, guidance=None):
    configuration = {
        'source': str(source), 'size': source.stat().st_size,
        'mtime_ns': source.stat().st_mtime_ns, 'chunk': chunk_seconds,
        'model': 'large-v3', 'gain_db': max_gain_db, 'beam': 5, 'pipeline_version': 2,
    }
    if guidance is not None:
        configuration['hotwords'] = guidance
    return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()


def output_paths(output, status, guided, root=ROOT):
    output = output or root / 'transcripts' / ('raw-glossary' if guided else 'raw')
    status = status or root / 'work' / ('asr_glossary_status.json' if guided else 'asr_status.json')
    if guided:
        baseline = (root / 'transcripts' / 'raw').resolve()
        if output.resolve().is_relative_to(baseline):
            raise ValueError('Guided ASR output must be outside transcripts/raw')
        if status.resolve() == (root / 'work' / 'asr_status.json').resolve():
            raise ValueError('Guided ASR must use a separate status file')
    return output, status


def load_hotword_profiles(inputs, directory, tokenizer):
    profiles = {}
    for source in inputs:
        recording = re.sub(r'_track-\d+$', '', source.stem)
        path = directory / (recording + '.txt')
        text = path.read_text(encoding='utf-8').strip()
        count = len(tokenizer.encode(' ' + text, add_special_tokens=False).ids) if text else 0
        if count > 223:
            raise ValueError(f'{path}: {count} Whisper tokens exceeds the 223-token hotwords limit')
        profiles[source] = {
            'text': text, 'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'token_count': count, 'policy_version': HOTWORDS_POLICY_VERSION,
        }
    return profiles


def local_whisper_tokenizer():
    # Tokenization needs no Whisper model or CUDA allocation.
    from tokenizers import Tokenizer
    cache = ROOT / 'work' / 'models' / 'models--Systran--faster-whisper-large-v3'
    revision = (cache / 'refs' / 'main').read_text().strip()
    return Tokenizer.from_file(str(cache / 'snapshots' / revision / 'tokenizer.json'))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def timestamp(seconds, srt=False):
    milliseconds = max(0, round(seconds * 1000))
    sec, ms = divmod(milliseconds, 1000)
    minutes, sec = divmod(sec, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{sec:02}' + (f',{ms:03}' if srt else '')


def render(result, directory):
    """Render both raw and speaker-labeled transcripts from the same JSON schema."""
    stem = Path(result['source']).stem
    lines, subtitles = [], []
    for n, s in enumerate(result['segments'], 1):
        text = s['text'].strip()
        label = s.get('speaker')
        text = f'{label}: {text}' if label else text
        lines.append(f'[{timestamp(s["start"])}–{timestamp(s["end"])}] {text}')
        subtitles.append(f'{n}\n{timestamp(s["start"], True)} --> {timestamp(s["end"], True)}\n{text}\n')
    for suffix, content in [('.txt', '\n'.join(lines)), ('.srt', '\n'.join(subtitles))]:
        path = directory / (stem + suffix)
        temporary = path.with_suffix(suffix + '.tmp')
        temporary.write_text(content + '\n')
        temporary.replace(path)


def clamp_timestamps(result):
    """Whisper may place its final word slightly beyond the actual media end."""
    limit = result['duration']
    for segment in result['segments'] + result.get('rejected_segments', []):
        segment['end'] = min(limit, max(0, segment['end']))
        segment['start'] = min(segment['end'], max(0, segment['start']))
        for word in segment.get('words') or []:
            word['end'] = min(limit, max(0, word['end']))
            word['start'] = min(word['end'], max(0, word['start']))


def prepare_export(result):
    clamp_timestamps(result)
    kept = []
    for segment in result['segments']:
        segment['review_recommended'] = bool(segment.get('review_recommended')) or segment.get('no_speech_prob', 0) > 0.6
        normalized = segment['text'].strip().casefold().rstrip('.…! ')
        # These high-confidence stock captions occurred in actual non-speech
        # samples. Preserve them for review, outside the usable transcript.
        stock_caption = 'dimatorzok' in normalized or normalized == 'продолжение следует'
        if stock_caption and segment.get('no_speech_prob', 0) > 0.8:
            segment['rejection_reason'] = 'suspected_whisper_stock_caption_on_non_speech'
            result.setdefault('rejected_segments', []).append(segment)
        else:
            kept.append(segment)
    result['segments'] = kept


def decode(path, offset, length):
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(offset), '-i', str(path),
               '-t', str(length), '-ar', '16000', '-ac', '1', '-f', 'f32le', 'pipe:1']
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype='<f4').copy()


def duration(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                             '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    return float(json.loads(result.stdout)['format']['duration'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='*', type=Path)
    parser.add_argument('--chunk-seconds', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--status', type=Path)
    parser.add_argument('--hotwords-dir', type=Path,
                        help='Directory of recording-name.txt profiles shared by paired tracks')
    parser.add_argument('--max-gain-db', type=float, default=20)
    args = parser.parse_args()
    inputs = args.inputs or sorted((ROOT / 'audio').glob('*.m4a'))
    try:
        args.output, status_path = output_paths(args.output, args.status, args.hotwords_dir is not None)
        profiles = (load_hotword_profiles([p.resolve() for p in inputs], args.hotwords_dir,
                                         local_whisper_tokenizer())
                    if args.hotwords_dir is not None else {})
    except (OSError, ValueError) as error:
        parser.error(str(error))
    jobs = [(p.resolve(), duration(p)) for p in inputs]
    # Finish short clips early, then process substantial sessions in date order.
    jobs.sort(key=lambda item: (item[1] >= 600, item[0].name))
    args.output.mkdir(parents=True, exist_ok=True)
    model = None
    completed = []
    for job_index, (source, length) in enumerate(jobs, 1):
        target = args.output / (source.stem + '.json')
        guidance = profiles.get(source)
        signature = asr_signature(source, args.chunk_seconds, args.max_gain_db, guidance)
        if target.exists():
            previous = json.loads(target.read_text())
            if previous.get('signature') == signature and previous.get('complete'):
                prepare_export(previous)
                render(previous, args.output)
                atomic_json(target, previous)
                completed.append(source.name)
                print(f'SKIP completed {source.name}', flush=True)
                continue
        if model is None:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
            print('Loading large-v3 on CUDA (float16)', flush=True)
            model = WhisperModel('large-v3', device='cuda', compute_type='float16',
                                 cpu_threads=4, download_root=str(ROOT / 'work' / 'models'),
                                 local_files_only=True)
            asr = BatchedInferencePipeline(model=model)
        print(f'FILE {job_index}/{len(jobs)} {source.name} ({length / 60:.1f} min)', flush=True)
        chunks = []
        rejected = []
        segments = []
        started = time.monotonic()
        for index in range(math.ceil(length / args.chunk_seconds)):
            offset = index * args.chunk_seconds
            chunk_path = ROOT / 'work' / 'asr_chunks' / signature / f'{index:05}.json'
            if chunk_path.exists():
                chunk = json.loads(chunk_path.read_text())
            else:
                decode_start = max(0, offset - 1.5)
                owned_end = min(offset + args.chunk_seconds, length)
                samples = decode(source, decode_start, min(owned_end + 1.5, length) - decode_start)
                if not len(samples):
                    raise RuntimeError(f'Empty decoded chunk {source} at {offset}')
                peak = float(np.max(np.abs(samples)))
                rms = float(np.sqrt(np.mean(samples ** 2)))
                # Quiet independent microphones need gain; cap amplification to avoid
                # turning near-silent electrical noise into the dominant signal.
                gain = min(10 ** (args.max_gain_db / 20), 0.70795 / peak) if 0 < peak < 0.1 else 1.0
                original_samples = samples.copy()
                samples *= gain
                generated, info = asr.transcribe(
                    samples, language='ru', task='transcribe', batch_size=args.batch_size,
                    beam_size=5, word_timestamps=True, vad_filter=True,
                    vad_parameters={'min_silence_duration_ms': 500, 'speech_pad_ms': 300},
                    condition_on_previous_text=False,
                    **({'hotwords': guidance['text']} if guidance is not None else {}),
                )
                decoded = []
                rejected_chunk = []
                for segment in generated:
                    item = dataclasses.asdict(segment)
                    # Overlap chunks to preserve words at decode boundaries, then
                    # give each timestamped word to exactly one chunk.
                    words = item.get('words') or []
                    for word in words:
                        word['start'] += decode_start
                        word['end'] += decode_start
                    words = [w for w in words if offset <= (w['start'] + w['end']) / 2 < owned_end]
                    if not words:
                        continue
                    item['words'] = words
                    item['text'] = ''.join(w['word'] for w in words)
                    item['start'], item['end'] = words[0]['start'], words[-1]['end']
                    local_start = max(0, int((item['start'] - decode_start) * 16000))
                    local_end = min(len(original_samples), int((item['end'] - decode_start) * 16000))
                    signal = original_samples[local_start:local_end]
                    signal_peak = float(np.max(np.abs(signal))) if len(signal) else 0
                    signal_rms = float(np.sqrt(np.mean(signal ** 2))) if len(signal) else 0
                    if signal_peak < 0.01 and signal_rms < 0.001:
                        item['rejection_reason'] = 'near_silent_signal: peak below -40 dBFS and RMS below -60 dBFS'
                        rejected_chunk.append(item)
                        continue
                    item['id'] = len(segments) + len(decoded)
                    item['review_recommended'] = item['avg_logprob'] < -1 or item['compression_ratio'] > 2.4
                    decoded.append(item)
                chunk = {'offset': offset, 'duration': owned_end - offset,
                         'speech_seconds': info.duration_after_vad, 'gain_db': 20 * math.log10(gain),
                         'original_peak_dbfs': 20 * math.log10(max(peak, 1e-12)),
                         'original_rms_dbfs': 20 * math.log10(max(rms, 1e-12)),
                         'segments': decoded, 'rejected_segments': rejected_chunk}
                atomic_json(chunk_path, chunk)
            chunks.append({k: v for k, v in chunk.items() if k not in {'segments', 'rejected_segments'}})
            segments.extend(chunk['segments'])
            rejected.extend(chunk.get('rejected_segments', []))
            atomic_json(status_path, {'state': 'running', 'file': source.name,
                        'file_index': job_index, 'total_files': len(jobs),
                        'audio_seconds_processed': min(offset + args.chunk_seconds, length),
                        'file_duration': length, 'completed': completed,
                        'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            print(f'  {min(offset + args.chunk_seconds, length) / 60:.1f}/{length / 60:.1f} min; '
                  f'{len(segments)} segments; elapsed {time.monotonic() - started:.1f}s', flush=True)
        result = {'source': str(source), 'signature': signature, 'complete': True,
                  'model': 'Systran/faster-whisper-large-v3', 'language': 'ru',
                  'duration': length, 'diarization': 'pending', 'chunks': chunks,
                  'segments': segments, 'rejected_segments': rejected}
        if guidance is not None:
            result['hotwords'] = guidance
        prepare_export(result)
        render(result, args.output)
        atomic_json(target, result)
        completed.append(source.name)
        print(f'DONE {source.name}: {len(segments)} segments', flush=True)
    atomic_json(status_path, {'state': 'complete', 'completed': completed,
                             'total_files': len(jobs), 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(f'ALL ASR COMPLETE: {len(completed)} tracks', flush=True)


if __name__ == '__main__':
    main()
