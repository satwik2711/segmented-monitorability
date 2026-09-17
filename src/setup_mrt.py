"""Download a near-balanced, single-condition MRT tool-use sample.

Install: uv sync
Run:     python3 src/setup_mrt.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


REPO = "ScaleAI/mrt"
CONDITION = "transcripts_no_hint_selective_thinking_time_basic"
AGENT_MODEL = "claude-3-5-sonnet-20241022"
MONITOR_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_OUTPUT = Path("data/mrt_shade_tool_use_sample.jsonl")
THINKING = re.compile(r"<(?:model_)?thinking>(.*?)</(?:model_)?thinking>", re.DOTALL)
SOURCE_PATH = re.compile(
    rf"^task_pairs/([^/]+)/{CONDITION}/{AGENT_MODEL}/{MONITOR_MODEL}/transcript_\d+\.json$"
)


def get_record(path: str, revision: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(REPO, path, repo_type="dataset", revision=revision)
    with open(local_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    conversation = raw.get("conversation")
    label = raw.get("side_task_evaluation")
    if not isinstance(conversation, list) or type(label) is not bool:
        raise ValueError(f"Missing conversation or boolean side_task_evaluation: {path}")

    reasoning_blocks = [
        match.group(1).strip()
        for message in conversation
        if message.get("role") == "assistant" and isinstance(message.get("content"), str)
        for match in THINKING.finditer(message["content"])
        if match.group(1).strip()
    ]
    if not reasoning_blocks:
        raise ValueError(f"No tagged assistant reasoning: {path}")
    if not any(message.get("role") == "tool" for message in conversation):
        raise ValueError(f"No tool observations: {path}")

    return {
        "source_path": path,
        "task": SOURCE_PATH.match(path).group(1),
        "side_task_success": label,
        "conversation": conversation,
        "reasoning_blocks": reasoning_blocks,
        "reasoning_words": sum(len(block.split()) for block in reasoning_blocks),
        "conversation_sha256": hashlib.sha256(
            json.dumps(conversation, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }


def choose_records(records: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    by_label = {label: [row for row in records if row["side_task_success"] is label] for label in (False, True)}
    minority = min((False, True), key=lambda label: len(by_label[label]))
    minority_count = min(target // 2, len(by_label[minority]))
    desired = {minority: minority_count, not minority: target - minority_count}
    if any(len(by_label[label]) < count for label, count in desired.items()):
        raise RuntimeError(f"Cannot form {target} rows from label counts {dict((k, len(v)) for k, v in by_label.items())}")

    selected: list[dict[str, Any]] = []
    for label in (False, True):
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_label[label]:
            by_task[row["task"]].append(row)
        for task_rows in by_task.values():
            task_rows.sort(key=lambda row: (-row["reasoning_words"], row["source_path"]))
        while sum(row["side_task_success"] is label for row in selected) < desired[label]:
            for task in sorted(by_task):
                if by_task[task] and sum(row["side_task_success"] is label for row in selected) < desired[label]:
                    selected.append(by_task[task].pop(0))

    if len({row["conversation_sha256"] for row in selected}) != len(selected):
        raise RuntimeError("Duplicate agent conversations in selected records")
    return sorted(selected, key=lambda row: row["source_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--revision", default="main", help="Dataset commit or branch")
    args = parser.parse_args()
    if args.target < 2:
        parser.error("--target must be at least 2")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install dependency: uv sync") from exc

    api = HfApi()
    revision = api.dataset_info(REPO, revision=args.revision).sha
    if not revision:
        raise RuntimeError("Could not resolve MRT dataset revision")
    paths = sorted(path for path in api.list_repo_files(REPO, repo_type="dataset", revision=revision) if SOURCE_PATH.fullmatch(path))
    if not paths:
        raise RuntimeError("No transcripts found for the configured MRT slice")
    print(f"Reading {len(paths)} MRT transcripts at {revision}", flush=True)
    with ThreadPoolExecutor(max_workers=12) as executor:
        records = list(executor.map(lambda path: get_record(path, revision), paths))
    selected = choose_records(records, args.target)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": REPO,
        "revision": revision,
        "task_family": "SHADE-Arena tool use",
        "condition": CONDITION,
        "agent_model": AGENT_MODEL,
        "original_monitor_model": MONITOR_MODEL,
        "label": "side_task_success (original side_task_evaluation)",
        "selection": "closest possible class balance; within each class, round robin over tasks, preferring longer reasoning",
        "candidate_counts": dict(Counter(str(row["side_task_success"]) for row in records)),
        "selected_counts": dict(Counter(str(row["side_task_success"]) for row in selected)),
        "selected_task_counts": {
            task: dict(Counter(str(row["side_task_success"]) for row in selected if row["task"] == task))
            for task in sorted({row["task"] for row in selected})
        },
        "sources": [row["source_path"] for row in selected],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} trajectories to {args.output}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
