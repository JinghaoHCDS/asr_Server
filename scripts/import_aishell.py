"""Freeze an extracted AISHELL-1 partition into a local quality-evaluation manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path


SOURCE = "https://www.openslr.org/33/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="extracted data_aishell directory; this tool does not download 15GB")
    parser.add_argument("--partition", choices=["dev", "test"], default="dev")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("limit must be positive")
    if args.output.exists():
        parser.error("output must be a new directory, so frozen references cannot be overwritten")
    transcript_path = args.data_dir / "transcript" / "aishell_transcript_v0.8.txt"
    audio_root = args.data_dir / "wav" / args.partition
    if not transcript_path.is_file() or not audio_root.is_dir():
        parser.error("expected transcript/aishell_transcript_v0.8.txt and extracted wav/PARTITION")
    transcripts = {}
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            transcripts[fields[0]] = "".join(fields[1].split())
    selected = [p for p in sorted(audio_root.rglob("*.wav")) if p.stem in transcripts][:args.limit]
    if not selected:
        parser.error("partition contains no extracted WAV files with reference transcripts")
    args.output.mkdir(parents=True)
    references = args.output / "reference"
    references.mkdir()
    items: list[dict[str, object]] = []
    for path in selected:
        reference = references / f"{path.stem}.txt"
        reference.write_text(transcripts[path.stem] + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with wave.open(str(path), "rb") as source:
            duration = source.getnframes() / source.getframerate()
        items.append({"id": f"aishell-{path.stem}", "split": "dev" if args.partition == "dev" else "acceptance",
            "audio": str(path.resolve()), "reference": str(reference.relative_to(args.output)),
            "sha256": digest, "duration_seconds": duration, "source": SOURCE,
            "license": "Apache-2.0 (OpenSLR SLR33)", "full_reference": True,
            "language": "auto", "terms": [], "absent_terms": []})
    manifest = {"schema_version": 1, "dataset": "AISHELL-1", "partition": args.partition,
        "source": SOURCE, "items": items,
        "limitations": "Clean short Mandarin utterances; does not replace natural long-form and domain-term evaluation."}
    (args.output / "corpus.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.output / "corpus.json"), "samples": len(items)}))


if __name__ == "__main__":
    main()
