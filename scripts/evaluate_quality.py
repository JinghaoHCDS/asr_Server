"""Reproducible Qwen development/acceptance runs; never treats mock as quality evidence."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import re
import sys
import tempfile
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidfuzz.distance import Levenshtein

from asr_server.config import Settings
from asr_server.transcription import TranscriptionRequest, run_transcription_path


PROFILES: dict[str, dict[str, Any]] = {
    "legacy": {"quality_profile": "legacy"},
    "legacy_budget4096": {"quality_profile": "legacy", "max_new_tokens": 4096},
    **{
        f"quality_{seconds}_{overlap}": {
            "quality_profile": "quality_v1", "quality_chunk_seconds": seconds,
            "overlap_seconds": overlap,
        }
        for seconds in (30, 60, 90, 120, 180) for overlap in (0, 1)
    },
}
PROFILES["quality_60_0_no_review"] = {**PROFILES["quality_60_0"], "quality_review_enabled": False}
PROFILES["quality_60_0_no_glossary"] = {**PROFILES["quality_60_0"], "hotwords": None, "context": "", "glossary_snapshot": None}


def normalize(text: str) -> str:
    """Declared CER normalization: Unicode NFKC, casefold, remove whitespace/punctuation."""
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def error_counts(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = normalize(reference), normalize(hypothesis)
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    for edit in Levenshtein.editops(ref, hyp):
        key = {"replace": "substitutions", "delete": "deletions", "insert": "insertions"}[edit.tag]
        counts[key] += 1
    distance = sum(counts.values())
    return {**counts, "reference_chars": len(ref), "distance": distance,
            "cer": distance / len(ref) if ref else (0.0 if not hyp else None)}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(math.ceil(len(values) * q) - 1, 0)]


def score_sample(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    text = str(prediction.get("text", ""))
    raw_text = str(prediction.get("raw_text", text))
    ref = case["reference_text"]
    terms = case.get("terms", [])
    expected = sum(normalize(ref).count(normalize(term)) for term in terms)
    recovered = sum(min(normalize(ref).count(normalize(term)), normalize(text).count(normalize(term))) for term in terms)
    forbidden_insertions = sum(normalize(text).count(normalize(term)) for term in case.get("absent_terms", []))
    # Ground truth labels identify correct automatic edits; do not infer correctness from a matched alias.
    quality = prediction.get("quality", {})
    changes = prediction.get("changes", quality.get("changes", []))
    approved = {(item["before"], item["after"]) for item in case.get("approved_changes", [])}
    correct = sum((str(item.get("before", item.get("from", ""))),
                   str(item.get("after", item.get("to", "")))) in approved for item in changes)
    alignment_errors: list[float] = []
    predicted = prediction.get("timestamps", [])
    for truth in case.get("alignment_reference", []):
        # Index is explicit annotation, preventing nearest-match reuse that hides alignment errors.
        index = int(truth["index"])
        if index < len(predicted) and normalize(predicted[index]["text"]) == normalize(truth["text"]):
            alignment_errors.extend([abs(float(predicted[index]["start"]) - float(truth["start"])),
                                     abs(float(predicted[index]["end"]) - float(truth["end"]))])
    boundaries = []
    for region in case.get("boundary_regions", []):
        observed = "".join(str(t["text"]) for t in predicted
                           if float(region["start"]) <= (float(t["start"]) + float(t["end"])) / 2 < float(region["end"]))
        boundaries.append(error_counts(region["reference_text"], observed))
    return {
        "id": case["id"], "duration_seconds": case["duration_seconds"],
        "raw": error_counts(ref, raw_text), "normalized": error_counts(ref, text),
        "expected_terms": expected, "recovered_terms": recovered,
        "absent_term_insertions": forbidden_insertions,
        "automatic_changes": len(changes), "correct_automatic_changes": correct,
        "alignment_errors_seconds": alignment_errors,
        "alignment_expected_units": len(case.get("alignment_reference", [])),
        "alignment_matched_units": len(alignment_errors) // 2,
        "boundary": boundaries,
        "complete": bool(prediction.get("text") is not None and not prediction.get("error")
                         and not quality.get("incomplete", False)
                         and quality.get("status") != "incomplete"
                         and not prediction.get("generation", {}).get("truncated", False)),
        "issues": quality.get("issues", []), "timings": prediction.get("timings", {}),
    }


def read_manifest(path: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    seen: set[str] = set()
    selected = []
    for item in manifest["items"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", item["id"]):
            raise ValueError("sample id must contain only ASCII letters, digits, underscore or hyphen")
        if item["id"] in seen:
            raise ValueError("duplicate sample id")
        seen.add(item["id"])
        if item.get("split") not in {"dev", "acceptance"}:
            raise ValueError("each sample requires dev or acceptance split")
        if not item.get("license") or not item.get("source"):
            raise ValueError("each sample requires source and license, including privately permitted recordings")
        if item.get("split") != split:
            continue
        audio = (path.parent / item["audio"]).resolve()
        reference = (path.parent / item["reference"]).resolve()
        if not audio.is_file() or not reference.is_file():
            raise ValueError(f"missing audio/reference for {item['id']}")
        if len(item.get("sha256", "")) != 64:
            raise ValueError("each audio requires a frozen SHA-256")
        digest = hashlib.sha256()
        with audio.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"audio fingerprint mismatch: {item['id']}")
        item = {**item, "audio_path": str(audio), "reference_text": reference.read_text()}
        if not normalize(item["reference_text"]) or not 0 < item.get("duration_seconds", 0) <= 21600:
            raise ValueError("human lexical reference and a duration within six hours are required")
        selected.append(item)
    if not selected:
        raise ValueError(f"no annotated {split} samples; cannot report quality acceptance")
    return selected


async def run_cases(cases: list[dict[str, Any]], output: Path, profiles: list[str], mock: bool) -> None:
    from asr_server.main import create_app
    if not mock:
        if platform.system() != "Linux":
            raise ValueError("real Qwen evaluation must run in WSL Linux")
        import importlib
        torch = importlib.import_module("torch")
        if torch.version.cuda is None or not torch.cuda.is_available():
            raise ValueError("CUDA torch acceptance required before real evaluation")
    settings = Settings(adapter="mock" if mock else "qwen", enable_qwen_alignment=True,
                        workbench_enabled=False)
    app = create_app(settings)
    manager = app.state.manager
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise ValueError("output already contains a run; use a new directory to keep experiments separate")
    inventory = manager.list_models()
    (output / "run.json").write_text(json.dumps({"mock": mock, "models": inventory,
        "profiles": {name: PROFILES[name] for name in profiles},
        "cases": [{"id": c["id"], "audio_sha256": c["sha256"],
                   "reference_sha256": hashlib.sha256(c["reference_text"].encode()).hexdigest()}
                  for c in cases],
        "normalization": "NFKC/casefold/no whitespace or punctuation"}, ensure_ascii=False, indent=2))
    try:
        for case in cases:
            base_request = TranscriptionRequest(model="qwen3-asr-1.7b", language=case.get("language", "auto"),
                response_format="verbose_json", timestamps="auto", backend="transformers", max_new_tokens=None,
                context=case.get("context", ""), hotwords=json.dumps(case.get("terms", []) + case.get("absent_terms", [])),
                split_strategy="auto", max_chunk_seconds=None, overlap_seconds=None, preserve_segments=True,
                glossary_snapshot=case.get("glossary_snapshot"))
            for name in profiles:
                path = output / name / f"{case['id']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                request = replace(base_request, **PROFILES[name])
                with tempfile.TemporaryDirectory(prefix="asr_eval_") as tmp:
                    try:
                        result = await run_transcription_path(Path(case["audio_path"]), workspace=Path(tmp),
                            manager=manager, settings=settings, request=request)
                    except Exception as exc:
                        result = {"error": {"type": type(exc).__name__, "message": str(exc)}, "text": ""}
                    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(f"{case['id']} {name}: {'failed' if result.get('error') else 'finished'}", flush=True)
            # A genuine paired short run; same human reference, reset model context per excerpt.
            for number, region in enumerate(case.get("paired_regions", [])):
                with tempfile.TemporaryDirectory(prefix="asr_eval_pair_") as tmp:
                    clip = Path(tmp) / "clip.wav"
                    process = await asyncio.create_subprocess_exec("ffmpeg", "-v", "error", "-y", "-ss", str(region["start"]),
                        "-i", case["audio_path"], "-t", str(region["end"] - region["start"]), "-vn", "-ac", "1", "-ar", "16000", str(clip))
                    if await process.wait() != 0:
                        raise ValueError("paired excerpt extraction failed")
                    for name in profiles:
                        short_request = replace(base_request, **PROFILES[name])
                        work = Path(tmp) / name
                        work.mkdir()
                        try:
                            result = await run_transcription_path(clip, workspace=work, manager=manager,
                                settings=settings, request=short_request)
                        except Exception as exc:
                            result = {"error": {"type": type(exc).__name__, "message": str(exc)}, "text": ""}
                        (output / name / f"{case['id']}.pair{number}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await manager.shutdown()


def score_run(cases: list[dict[str, Any]], output: Path, profiles: list[str], split: str) -> dict[str, Any]:
    metadata = json.loads((output / "run.json").read_text())
    summaries = {}
    for name in profiles:
        scores = []
        pairs: list[dict[str, Any]] = []
        for case in cases:
            prediction = json.loads((output / name / f"{case['id']}.json").read_text())
            scores.append(score_sample(case, prediction))
            for number, region in enumerate(case.get("paired_regions", [])):
                short = json.loads((output / name / f"{case['id']}.pair{number}.json").read_text())
                long_text = "".join(str(item["text"]) for item in prediction.get("timestamps", [])
                    if region["start"] <= (item["start"] + item["end"]) / 2 < region["end"])
                pairs.append({"duration_seconds": case["duration_seconds"],
                    "long": error_counts(region["reference_text"], long_text),
                    "short": error_counts(region["reference_text"], str(short.get("text", "")))})
        chars = sum(s["raw"]["reference_chars"] for s in scores)
        errors = sum(s["raw"]["distance"] for s in scores)
        total_terms = sum(s["expected_terms"] for s in scores)
        changes = sum(s["automatic_changes"] for s in scores)
        align_errors = [error for s in scores for error in s["alignment_errors_seconds"]]
        expected_alignment = sum(s["alignment_expected_units"] for s in scores)
        buckets = {}
        for label, low, high in (("30m", 1200, 3600), ("2h", 3600, 10800), ("4h", 10800, 18000), ("6h", 18000, 21601)):
            group = [p for p in pairs if low <= p["duration_seconds"] < high]
            n = sum(p["long"]["reference_chars"] for p in group)
            buckets[label] = (sum(p["long"]["distance"] - p["short"]["distance"] for p in group) / n) if n else None
        summaries[name] = {"raw_cer": errors / chars if chars else None,
            "normalized_cer": sum(s["normalized"]["distance"] for s in scores) / chars if chars else None,
            "term_recall": sum(s["recovered_terms"] for s in scores) / total_terms if total_terms else None,
            "normalization_precision": sum(s["correct_automatic_changes"] for s in scores) / changes if changes else None,
            "absent_term_insertions": sum(s["absent_term_insertions"] for s in scores),
            "alignment_p95_seconds": percentile(align_errors, .95),
            "alignment_coverage": sum(s["alignment_matched_units"] for s in scores) / expected_alignment if expected_alignment else None,
            "duration_bucket_cer_delta": buckets, "complete": all(s["complete"] for s in scores),
            "boundary_deletions": sum(b["deletions"] for s in scores for b in s["boundary"]),
            "boundary_insertions": sum(b["insertions"] for s in scores for b in s["boundary"]),
            "timing_total_ms": sum(s["timings"].get("total_ms", 0) for s in scores),
            "samples": scores}
    baseline = summaries.get("legacy")
    eligible = []
    for name, summary in summaries.items():
        checks = {
            "real_acceptance": split == "acceptance" and not metadata["mock"],
            "overall_cer": baseline is not None and summary["raw_cer"] <= baseline["raw_cer"],
            "length_stability": all(v is not None and v <= .01 for v in summary["duration_bucket_cer_delta"].values()),
            "term_recall": summary["term_recall"] is not None and summary["term_recall"] >= .95,
            "normalization_precision": summary["normalization_precision"] is not None and summary["normalization_precision"] >= .99,
            "no_absent_terms": summary["absent_term_insertions"] == 0,
            "alignment": summary["alignment_p95_seconds"] is not None and summary["alignment_p95_seconds"] <= .5 and summary["alignment_coverage"] == 1,
            "full_six_hour_reference": any(c["duration_seconds"] >= 19800 and c.get("full_reference") for c in cases),
            "complete": summary["complete"],
            "boundary_annotations": any(c.get("boundary_regions") for c in cases),
        }
        summary["checks"] = checks
        if all(checks.values()) and name.startswith("quality_"):
            eligible.append(name)
    ranked = sorted(eligible, key=lambda n: (summaries[n]["raw_cer"], 1 - summaries[n]["term_recall"],
        summaries[n]["boundary_deletions"] + summaries[n]["boundary_insertions"], summaries[n]["timing_total_ms"]))
    return {"status": "passed" if ranked else "not_accepted", "selected_candidate": ranked[0] if ranked else None,
            "mock": metadata["mock"], "split": split, "profiles": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "run", "score"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=["dev", "acceptance"], default="dev")
    parser.add_argument("--output", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    try:
        cases = read_manifest(args.manifest, args.split)
        if args.command == "validate":
            print(json.dumps({"samples": len(cases), "split": args.split}))
            return 0
        if args.command == "run":
            asyncio.run(run_cases(cases, args.output, args.profiles, args.mock))
            return 0
        report = score_run(cases, args.output, args.profiles, args.split)
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({"status": report["status"], "selected_candidate": report["selected_candidate"]}))
        return 0 if report["status"] == "passed" else 2
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
