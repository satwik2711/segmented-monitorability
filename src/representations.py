"""Shared, label-free trajectory representations for the MRT experiments."""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from core_chunker import split_into_sentences


THINKING = re.compile(r"<(?:model_)?thinking>(.*?)</(?:model_)?thinking>", re.DOTALL)
LEX_START = re.compile(r"^(?:wait|but|however|therefore|actually|let me)\b", re.IGNORECASE)
SCOPES = ("ACTION", "COT", "FULL")
REPRESENTATIONS = ("RAW", "FIX4", "RANDOM-K", "ENTROPY")


def chunks_from_cuts(sentences: list[str], cuts: list[int]) -> list[str]:
    if cuts != sorted(set(cuts)) or any(cut < 1 or cut >= len(sentences) for cut in cuts):
        raise ValueError(f"Invalid sentence cuts: {cuts}")
    ends = [*cuts, len(sentences)]
    start = 0
    chunks = []
    for end in ends:
        chunks.append(" ".join(sentences[start:end]))
        start = end
    return chunks


def lexical_cuts(sentences: list[str]) -> list[int]:
    return [index for index, sentence in enumerate(sentences) if index and LEX_START.match(sentence)]


def fixed_four_cuts(sentences: list[str]) -> list[int]:
    return list(range(4, len(sentences), 4))


def random_matched_cuts(sentences: list[str], cut_count: int, identity: str) -> list[int]:
    if cut_count > len(sentences) - 1:
        raise ValueError("Random cuts exceed available sentence boundaries")
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    return sorted(random.Random(seed).sample(range(1, len(sentences)), cut_count))


def canonical_block(text: str) -> str:
    return " ".join(split_into_sentences(text))


def common_task_context(conversation: list[dict[str, Any]]) -> str:
    """Initial instructions are common to all monitor observation scopes."""
    parts = []
    for message in conversation:
        role = message.get("role")
        if role not in ("system", "user"):
            break
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(f"{role.upper()} TASK INSTRUCTIONS:\n{content.strip()}")
    if not parts:
        raise ValueError("Trajectory has no initial task instructions")
    return "\n\n".join(parts)


def trajectory_events(row: dict[str, Any], units: dict[str, Any]) -> list[dict[str, Any]]:
    """Produce raw events with stable IDs; representation only changes reasoning layout."""
    events = []
    block_index = 0
    for message_index, message in enumerate(row["conversation"]):
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            if isinstance(content, str):
                cursor = 0
                for match in THINKING.finditer(content):
                    visible = content[cursor:match.start()].strip()
                    if visible:
                        events.append({"kind": "assistant_text", "text": visible, "message_index": message_index})
                    if block_index >= len(units["blocks"]):
                        raise ValueError("More reasoning spans in conversation than in units")
                    expected = canonical_block(match.group(1).strip())
                    block = units["blocks"][block_index]
                    if expected != " ".join(block["sentences"]):
                        raise ValueError(f"Reasoning block {block_index} does not match saved units")
                    events.append({"kind": "reasoning", "block_index": block_index, "message_index": message_index})
                    block_index += 1
                    cursor = match.end()
                visible = content[cursor:].strip()
                if visible:
                    events.append({"kind": "assistant_text", "text": visible, "message_index": message_index})
            if message.get("tool_calls"):
                events.append({
                    "kind": "tool_call",
                    "text": json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True),
                    "message_index": message_index,
                })
        elif role == "tool":
            events.append({"kind": "tool_observation", "text": str(content or ""), "message_index": message_index})
    if block_index != len(units["blocks"]):
        raise ValueError("Saved units contain more reasoning blocks than conversation")
    return events


def event_raw_text(event: dict[str, Any], units: dict[str, Any]) -> str:
    if event["kind"] == "reasoning":
        return " ".join(units["blocks"][event["block_index"]]["sentences"])
    return event["text"]


def render_observation(
    events: list[dict[str, Any]],
    units: dict[str, Any],
    scope: str,
    representation: str = "RAW",
) -> str:
    if scope not in SCOPES or representation not in REPRESENTATIONS:
        raise ValueError(f"Invalid scope or representation: {scope}, {representation}")
    parts = []
    for event in events:
        kind = event["kind"]
        if kind == "reasoning":
            if scope == "ACTION":
                continue
            index = event["block_index"]
            block = units["blocks"][index]
            if representation == "RANDOM-K":
                entropy_count = len(block["methods"]["ENTROPY"]["cuts"])
                cuts = random_matched_cuts(
                    block["sentences"], entropy_count, f"{units['source_path']}:{index}"
                )
                chunks = chunks_from_cuts(block["sentences"], cuts)
            else:
                chunks = block["methods"][representation]["chunks"]
            parts.append(f"REASONING BLOCK {index + 1}:\n" + "\n".join(
                f"  [{chunk_index + 1}] {chunk}" for chunk_index, chunk in enumerate(chunks)
            ))
        elif scope != "COT":
            title = {
                "assistant_text": "ASSISTANT VISIBLE TEXT",
                "tool_call": "TOOL CALL",
                "tool_observation": "TOOL OBSERVATION",
            }[kind]
            parts.append(f"{title}:\n{event['text']}")
    return "\n\n".join(parts)


def half_rollout_events(events: list[dict[str, Any]], units: dict[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], int, int]:
    """Choose one raw event boundary, reused for every representation and scope."""
    lengths = [len(tokenizer.encode(event_raw_text(event, units), add_special_tokens=False)) for event in events]
    total = sum(lengths)
    target = total / 2
    seen = 0
    cut = 0
    for index, length in enumerate(lengths):
        if seen + length > target:
            break
        seen += length
        cut = index + 1
    return events[:cut], seen, total


def boundary_counts(units: dict[str, Any]) -> dict[str, int]:
    return {
        method: sum(len(block["methods"][method]["cuts"]) for block in units["blocks"])
        for method in ("LEX", "FIX4", "ENTROPY")
    }
