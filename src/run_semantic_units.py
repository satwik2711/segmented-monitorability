"""Experiment 1: freeze LEX, FIX4, and ENTROPY units for the MRT sample."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from core_chunker import PPL_MODEL_ID, segment_trace_with_ppl_debug, split_into_sentences
from representations import chunks_from_cuts, fixed_four_cuts, lexical_cuts


DEFAULT_INPUT = Path("data/mrt_shade_tool_use_sample.jsonl")
DEFAULT_OUTPUT = Path("data/mrt_semantic_units.jsonl")


def make_units(row: dict[str, Any], model_id: str, device: str | None) -> dict[str, Any]:
    blocks = []
    for text in row["reasoning_blocks"]:
        sentences = split_into_sentences(text)
        if not sentences:
            raise ValueError(f"Empty reasoning block in {row['source_path']}")
        entropy_chunks, scores, _, diagnostics = segment_trace_with_ppl_debug(
            text, model_id=model_id, device=device
        )
        entropy_cuts = diagnostics["cut_after_sentence_indices"]
        if chunks_from_cuts(sentences, entropy_cuts) != entropy_chunks and len(sentences) > 1:
            raise ValueError("ENTROPY cuts did not reconstruct the chunker output")
        methods = {}
        for name, cuts in (
            ("RAW", []),
            ("LEX", lexical_cuts(sentences)),
            ("FIX4", fixed_four_cuts(sentences)),
            ("ENTROPY", entropy_cuts),
        ):
            methods[name] = {"cuts": cuts, "chunks": chunks_from_cuts(sentences, cuts)}
        blocks.append({"sentences": sentences, "methods": methods, "entropy_scores": scores})
    return {"source_path": row["source_path"], "model_id": model_id, "blocks": blocks}


def boundary_f1(units: list[dict[str, Any]], annotations_path: Path) -> dict[str, float | None]:
    indexed = {row["source_path"]: row for row in units}
    counts = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})
    reviewed = 0
    with annotations_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if not item.get("reviewed"):
                continue
            source_path = item["source_path"]
            block_index = item["block_index"]
            block = indexed[source_path]["blocks"][block_index]
            gold = item["gold_cuts"]
            chunks_from_cuts(block["sentences"], gold)
            reviewed += 1
            for method in ("LEX", "FIX4", "ENTROPY"):
                predictions = block["methods"][method]["cuts"]
                matched = 0
                prediction_index = 0
                gold_index = 0
                while prediction_index < len(predictions) and gold_index < len(gold):
                    prediction = predictions[prediction_index]
                    truth = gold[gold_index]
                    if abs(prediction - truth) <= 1:
                        matched += 1
                        prediction_index += 1
                        gold_index += 1
                    elif prediction < truth:
                        prediction_index += 1
                    else:
                        gold_index += 1
                counts[method]["tp"] += matched
                counts[method]["pred"] += len(predictions)
                counts[method]["gold"] += len(gold)
    if not reviewed:
        raise ValueError("No reviewed annotation blocks found")
    return {
        method: (2 * values["tp"] / (values["pred"] + values["gold"]))
        if values["pred"] + values["gold"] else None
        for method, values in counts.items()
    }


def write_annotation_template(units: list[dict[str, Any]], path: Path, trace_count: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in units[:trace_count]:
            for block_index, block in enumerate(row["blocks"]):
                if len(block["sentences"]) < 2:
                    continue
                item = {
                    "source_path": row["source_path"],
                    "block_index": block_index,
                    "sentences": block["sentences"],
                    "gold_cuts": [],
                    "reviewed": False,
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def summarize(units: list[dict[str, Any]], annotations: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {"trajectories": len(units), "methods": {}}
    f1 = boundary_f1(units, annotations) if annotations else {}
    for method in ("LEX", "FIX4", "ENTROPY"):
        chunk_words = [
            len(chunk.split())
            for row in units
            for block in row["blocks"]
            for chunk in block["methods"][method]["chunks"]
        ]
        cut_count = sum(
            len(block["methods"][method]["cuts"])
            for row in units for block in row["blocks"]
        )
        result["methods"][method] = {
            "boundary_f1_tolerance_1_sentence": f1.get(method),
            "chunks": len(chunk_words),
            "within_block_cuts": cut_count,
            "median_words_per_chunk": statistics.median(chunk_words) if chunk_words else None,
            "fraction_chunks_under_10_words": (
                sum(words < 10 for words in chunk_words) / len(chunk_words) if chunk_words else None
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=PPL_MODEL_ID)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--annotations", type=Path, help="Reviewed boundary annotations as JSONL")
    parser.add_argument("--annotation-template", type=Path, help="Write an unreviewed template for 12 traces")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    units = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if args.limit is not None and len(units) >= args.limit:
                break
            row = make_units(json.loads(line), args.model_id, args.device)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            units.append(row)
            print(f"Processed {len(units)}: {row['source_path']}", flush=True)
    if args.annotation_template:
        write_annotation_template(units, args.annotation_template)
    summary = summarize(units, args.annotations)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(units)} trajectories to {args.output} and summary to {summary_path}")


if __name__ == "__main__":
    main()
