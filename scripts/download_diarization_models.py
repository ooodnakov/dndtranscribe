#!/usr/bin/env python3
"""Download pinned, public sherpa-onnx speaker-diarization model artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "work" / "models" / "diarization"
ASSETS = (
    {
        "name": "pyannote-segmentation-3.0 ONNX",
        "repo_id": "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
        "revision": "9403a6902bb58e3d5ae8c7e77c3422de279db2e0",
        "filename": "model.onnx",
        "local_path": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
        "sha256": "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
        "upstream_source": "https://huggingface.co/pyannote/segmentation-3.0",
        "license": "MIT (CNRS)",
        "license_url": "https://huggingface.co/pyannote/segmentation-3.0/blob/main/LICENSE",
        "access_note": (
            "The original Hugging Face repository is gated for direct downloads; this pinned "
            "ONNX conversion is publicly downloadable from the sherpa-onnx model mirror."
        ),
    },
    {
        "name": "3D-Speaker CAM++ VoxCeleb embedding",
        "repo_id": "csukuangfj/speaker-embedding-models",
        "revision": "0743f301363dec56491a490f6d6cbc9d67f9a3bf",
        "filename": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "local_path": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "sha256": "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b",
        "upstream_source": "https://github.com/modelscope/3D-Speaker",
        "license": "Apache-2.0",
        "license_url": "https://github.com/modelscope/3D-Speaker/blob/main/LICENSE",
        "access_note": "English VoxCeleb speaker-embedding model; no Hugging Face token is needed.",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(model_dir: Path) -> dict[str, object]:
    assets = []
    for asset in ASSETS:
        local_path = model_dir / asset["local_path"]
        item = {key: value for key, value in asset.items() if key != "sha256"}
        item["local_path"] = asset["local_path"]
        item["local_sha256"] = sha256(local_path)
        assets.append(item)
    return {
        "created_by": "scripts/download_diarization_models.py",
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "assets": assets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    for asset in ASSETS:
        destination = args.model_dir / asset["local_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and sha256(destination) == asset["sha256"]:
            print(f"Verified existing {destination}")
            continue

        cached_file = Path(
            hf_hub_download(
                repo_id=asset["repo_id"],
                filename=asset["filename"],
                revision=asset["revision"],
                token=False,
            )
        )
        actual_hash = sha256(cached_file)
        if actual_hash != asset["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {asset['name']}: expected {asset['sha256']}, "
                f"received {actual_hash}"
            )
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copyfile(cached_file, temporary)
        os.replace(temporary, destination)
        print(f"Downloaded and verified {destination}")

    details = provenance(args.model_dir)
    provenance_path = args.model_dir / "provenance.json"
    temporary_provenance = provenance_path.with_name(provenance_path.name + ".tmp")
    temporary_provenance.write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_provenance, provenance_path)
    print(f"Wrote model provenance to {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
