#!/usr/bin/env python3
"""Measure an upstream Decider checkpoint on saved transcript edit cases.

Uses the official eager PyTorch API and its stored calibration. This writes
only a probe report; it never modifies transcripts or contacts LM Studio.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[1]
OPTION_SUFFIX = "\nOptions:\n(A) no\n(B) yes\nAnswer: ("


def load_cases(path):
    """Recover the original inputs exactly, refusing incompatible fixtures."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty JSON list")
    prepared = []
    for case in cases:
        if not isinstance(case.get("expected"), bool):
            raise ValueError(f"{case.get('name')}: expected must be a boolean")
        prompt = case["request"]["prompt"]
        state = f"Original: {case['original']}\nProposed edit: {case['proposed']}"
        prefix = f"Context:\n{state}\n\nQuestion: "
        if not prompt.startswith(prefix) or not prompt.endswith(OPTION_SUFFIX):
            raise ValueError(f"{case['name']}: prompt does not match saved inputs")
        question = prompt[len(prefix):-len(OPTION_SUFFIX)]
        if not question.strip():
            raise ValueError(f"{case['name']}: empty question")
        prepared.append((case, state, question))
    return prepared


def summarize(results):
    harmless = [r for r in results if r["expected"]]
    harmful = [r for r in results if not r["expected"]]
    correct = sum(r["correct"] for r in results)
    return {
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "harmless_total": len(harmless),
        "harmless_accepted": sum(r["predicted"] for r in harmless),
        "harmful_total": len(harmful),
        "harmful_rejected": sum(not r["predicted"] for r in harmful),
        "median_elapsed_ms": statistics.median(r["elapsed_ms"] for r in results),
        "brier_score": statistics.mean(
            (r["probabilities"]["yes"] - r["expected"]) ** 2 for r in results
        ),
    }


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local upstream HF snapshot")
    parser.add_argument("--cases", type=Path, default=ROOT / "work/decider-validation/cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "work/decider-validation/upstream-results.json")
    args = parser.parse_args()
    prepared = load_cases(args.cases)
    model_path = args.model.resolve(strict=True)
    config = json.loads((model_path / "decider_config.json").read_text(encoding="utf-8"))
    if args.output.resolve().is_relative_to(model_path) or args.output.resolve() == args.cases.resolve():
        parser.error("Output must be outside the model directory and differ from cases")

    # Import after argument/input validation so --help works without GPU packages.
    import torch
    from decider.infer import Decider

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")
    started = time.perf_counter()
    decider = Decider(str(model_path), device="cuda", dtype=torch.bfloat16, use_graphs=False)
    torch.cuda.synchronize()
    load_elapsed_ms = (time.perf_counter() - started) * 1000

    def decide(state, question):
        return decider.system_one(
            state,
            {"preserves_transcript": {"type": "noul", "instructions": question}},
            layout="state_first",
        )

    # Exclude lazy initialization from the measured case times.
    warm_started = time.perf_counter()
    decide(prepared[0][1], prepared[0][2])
    torch.cuda.synchronize()
    warmup_elapsed_ms = (time.perf_counter() - warm_started) * 1000
    results = []
    for case, state, question in prepared:
        torch.cuda.synchronize()
        started = time.perf_counter()
        response = decide(state, question)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        probability = float(response["answers"]["preserves_transcript"]["noul"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"{case['name']}: invalid Noul probability {probability}")
        predicted = probability >= 0.5
        results.append({
            "name": case["name"], "original": case["original"], "proposed": case["proposed"],
            "state": state, "question": question, "expected": case["expected"],
            "predicted": predicted, "correct": predicted == case["expected"],
            "probabilities": {"no": round(1 - probability, 4), "yes": probability},
            "elapsed_ms": elapsed_ms, "response": response,
        })
        print(f"{case['name']}: p(yes)={probability:.4f}, correct={predicted == case['expected']}, {elapsed_ms:.1f} ms", flush=True)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "Official decider.infer.Decider.system_one; typed Noul; eager CUDA BF16; state-first",
        "model_path": str(model_path), "model_name": decider.name, "model_config": config,
        "cases_path": str(args.cases.resolve()), "threshold": 0.5,
        "tie_rule": "Accept when reported P(yes) >= 0.5",
        "probability_precision": "Upstream system_one rounds Noul to four decimal places; no custom readout",
        "temperature": decider.T, "temperature_by_type": decider.T_by_type,
        "device": torch.cuda.get_device_name(),
        "packages": {name: package_version(name) for name in ("decider-ai", "torch", "transformers", "flash-linear-attention")},
        "timing": {"load_elapsed_ms": load_elapsed_ms, "warmup_elapsed_ms": warmup_elapsed_ms,
                   "description": "One warmup; each timed call synchronized before and after; model loading excluded"},
        "summary": summarize(results), "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
