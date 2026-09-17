"""Segment reasoning with top-20 predictive entropy from a local Qwen model.

Run on the MRT sample: python3 src/core_chunker.py
Dependencies: uv sync
"""

import argparse
import json
import math
import re
from collections import Counter
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

PPL_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
PPL_TOP_LOGPROBS = 20
PPL_DEFAULT_MAX_WINDOW_TOKENS = 768
PPL_WINDOW_OVERLAP_RATIO = 0.12

_SENTINEL = "\x00"

_ABBREVIATIONS = sorted(
    [
        # Titles
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Jr.",
        "Sr.",
        "Gen.",
        "Gov.",
        "Sen.",
        "Rep.",
        "Sgt.",
        "Cpl.",
        "Pvt.",
        # Months
        "Jan.",
        "Feb.",
        "Mar.",
        "Apr.",
        "Jun.",
        "Jul.",
        "Aug.",
        "Sep.",
        "Sept.",
        "Oct.",
        "Nov.",
        "Dec.",
        # Days
        "Mon.",
        "Tue.",
        "Wed.",
        "Thu.",
        "Fri.",
        "Sat.",
        "Sun.",
        # General / Academic
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "approx.",
        "dept.",
        "govt.",
        "corp.",
        "inc.",
        "ltd.",
        "assoc.",
        # Units
        "ft.",
        "in.",
        "oz.",
        "lb.",
        "lbs.",
        "mi.",
        "km.",
        "cm.",
        "mm.",
        "hr.",
        "hrs.",
        "min.",
        "sec.",
        "sq.",
        # References
        "No.",
        "Fig.",
        "Eq.",
        "Vol.",
        "Ch.",
        "Sec.",
        "p.",
        "pp.",
        # Place
        "St.",
        "Ave.",
        "Blvd.",
        "Rd.",
        "Ct.",
        # Geographic / Time
        "U.S.",
        "U.K.",
        "a.m.",
        "p.m.",
    ],
    key=len,
    reverse=True,
)


@dataclass(frozen=True)
class _ChunkingRuntimeProbe:
    model_id: str
    device: str
    model: Any
    tokenizer: Any
    tokenizer_ref: str
    max_window_tokens: int
    overlap_tokens: int


def _mask_abbreviations(text: str) -> str:
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr[:-1] + _SENTINEL)
    return text


def _unmask_abbreviations(text: str) -> str:
    return text.replace(_SENTINEL, ".")


def split_into_sentences(text: str) -> list[str]:
    masked = _mask_abbreviations(text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", masked)
    return [_unmask_abbreviations(s).strip() for s in parts if s.strip()]


@lru_cache(maxsize=2)
def _startup_probe(model_id: str = PPL_MODEL_ID, device: str | None = None) -> _ChunkingRuntimeProbe:
    """Load the proxy LM once and cap windows to keep logits memory bounded."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Direct scoring needs torch and transformers. Install with "
            "`uv sync`."
        ) from exc

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Boundary mapping requires a fast tokenizer with offset_mapping")
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    context_limit = getattr(model.config, "max_position_embeddings", None)
    max_window_tokens = min(PPL_DEFAULT_MAX_WINDOW_TOKENS, context_limit or PPL_DEFAULT_MAX_WINDOW_TOKENS)
    return _ChunkingRuntimeProbe(
        model_id=model_id,
        device=device,
        model=model,
        tokenizer=tokenizer,
        tokenizer_ref=model_id,
        max_window_tokens=max_window_tokens,
        overlap_tokens=max(1, int(max_window_tokens * PPL_WINDOW_OVERLAP_RATIO)),
    )


def probe_chunking_runtime(
    model_id: str = PPL_MODEL_ID,
    device: str | None = None,
) -> dict[str, Any]:
    probe = _startup_probe(model_id=model_id, device=device)
    return {
        "model_id": probe.model_id,
        "device": probe.device,
        "tokenizer_ref": probe.tokenizer_ref,
        "max_window_tokens": probe.max_window_tokens,
        "overlap_tokens": probe.overlap_tokens,
    }


def _sentence_boundary_char_offsets(sentences: list[str]) -> list[int]:
    if len(sentences) < 2:
        return []

    offsets: list[int] = []
    prefix_len = len(sentences[0])
    for sentence in sentences[1:]:
        offsets.append(prefix_len)
        prefix_len += 1 + len(sentence)
    return offsets


def _is_valid_boundary_positions(
    positions: list[int],
    token_count: int,
    expected_count: int,
) -> bool:
    if len(positions) != expected_count:
        return False
    if token_count <= 1:
        return False

    prev = -1
    for pos in positions:
        if not isinstance(pos, int):
            return False
        if pos <= 0 or pos >= token_count:
            return False
        if pos <= prev:
            return False
        prev = pos
    return True


def _coerce_boundary_positions(
    raw_positions: list[int],
    token_count: int,
    expected_count: int,
) -> list[int]:
    if expected_count == 0:
        return []
    if token_count <= 1:
        raise ValueError("Token count too small for boundary mapping")
    if expected_count >= token_count:
        raise ValueError(
            f"Too many boundaries ({expected_count}) for token_count={token_count}"
        )

    positions: list[int] = []
    prev = 0
    for idx, raw in enumerate(raw_positions):
        remaining = expected_count - idx - 1
        min_allowed = prev + 1
        max_allowed = token_count - 1 - remaining
        candidate = int(raw)
        if candidate < min_allowed:
            candidate = min_allowed
        if candidate > max_allowed:
            candidate = max_allowed
        if candidate < min_allowed or candidate > max_allowed:
            raise ValueError("Unable to coerce boundary token positions")
        positions.append(candidate)
        prev = candidate

    return positions


def _map_boundaries_with_offsets(
    boundary_offsets: list[int],
    token_offsets: list[int],
    token_count: int,
) -> list[int]:
    expected_count = len(boundary_offsets)
    if expected_count == 0:
        return []
    if not token_offsets:
        raise ValueError("Missing tokenizer offsets for boundary mapping")

    raw_positions: list[int] = []
    for boundary in boundary_offsets:
        pos = bisect_left(token_offsets, boundary)
        while pos + 1 < len(token_offsets) and token_offsets[pos + 1] == token_offsets[pos]:
            pos += 1
        if pos >= len(token_offsets):
            # Some servers return offsets that do not fully cover final prompt chars.
            pos = len(token_offsets) - 1
        raw_positions.append(pos)

    positions = _coerce_boundary_positions(
        raw_positions=raw_positions,
        token_count=token_count,
        expected_count=expected_count,
    )

    if not _is_valid_boundary_positions(positions, token_count, expected_count):
        raise ValueError("Invalid boundary token positions from tokenizer offsets")
    return positions


def _score_window_boundaries(
    sentences_window: list[str],
    probe: _ChunkingRuntimeProbe,
) -> tuple[list[tuple[int, int, float]], int, str]:
    """Score each boundary using the distribution for its next prompt token."""
    if not sentences_window:
        return [], 0, "tokenizer_offsets"

    import torch

    prompt = " ".join(sentences_window)
    encoded = probe.tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    token_count = encoded["input_ids"].shape[1]
    if token_count > probe.max_window_tokens:
        raise ValueError(f"Window has {token_count} tokens; limit is {probe.max_window_tokens}")
    if len(sentences_window) < 2:
        return [], token_count, "tokenizer_offsets"

    boundary_positions = _map_boundaries_with_offsets(
        _sentence_boundary_char_offsets(sentences_window),
        [start for start, _ in offsets],
        token_count,
    )
    input_ids = encoded["input_ids"].to(probe.device)
    attention_mask = encoded["attention_mask"].to(probe.device)
    with torch.inference_mode():
        logits = probe.model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
        scores: list[tuple[int, int, float]] = []
        for local_idx, token_pos in enumerate(boundary_positions):
            # Causal logits at token_pos - 1 predict the token at token_pos.
            top_logits = torch.topk(logits[token_pos - 1].float(), k=PPL_TOP_LOGPROBS).values
            probabilities = torch.softmax(top_logits, dim=-1)
            entropy = -(probabilities * torch.log2(probabilities)).sum()
            scores.append((local_idx, token_pos, float(torch.exp2(entropy).item())))
    return scores, token_count, "tokenizer_offsets"


def _count_tokens_with_tokenizer(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _build_sentence_windows(
    sentences: list[str],
    max_window_tokens: int,
    overlap_tokens: int,
    tokenizer: Any,
) -> list[tuple[int, int]]:
    n = len(sentences)
    if n <= 1:
        return [(0, n)]

    token_count_cache: dict[tuple[int, int], int] = {}

    def span_token_count(start: int, end: int) -> int:
        key = (start, end)
        if key in token_count_cache:
            return token_count_cache[key]
        prompt = " ".join(sentences[start:end])
        count = _count_tokens_with_tokenizer(tokenizer, prompt)
        token_count_cache[key] = count
        return count

    windows: list[tuple[int, int]] = []
    start = 0
    while start < n:
        low = start + 1
        high = n
        best_end: int | None = None

        while low <= high:
            mid = (low + high) // 2
            count = span_token_count(start, mid)
            if count <= max_window_tokens:
                best_end = mid
                low = mid + 1
            else:
                high = mid - 1

        if best_end is None:
            raise ValueError(
                f"Sentence span starting at index {start} exceeds max_window_tokens={max_window_tokens}"
            )

        windows.append((start, best_end))
        if best_end >= n:
            break

        if best_end - start <= 1:
            start = best_end
            continue

        target_overlap_tokens = max(1, overlap_tokens)
        next_low = start + 1
        next_high = best_end - 1
        next_start = best_end - 1
        while next_low <= next_high:
            mid = (next_low + next_high) // 2
            overlap_count = span_token_count(mid, best_end)
            if overlap_count > target_overlap_tokens:
                next_low = mid + 1
            else:
                next_start = mid
                next_high = mid - 1

        start = max(start + 1, min(next_start, best_end - 1))

    return windows


def _compute_boundary_perplexity(
    sentences: list[str],
    model_id: str = PPL_MODEL_ID,
    device: str | None = None,
) -> tuple[list[float], dict[str, Any]]:
    if len(sentences) < 2:
        return [], {
            "window_count": 0,
            "windowing_used": False,
            "boundary_mapping_mode_counts": {},
            "used_boundary_mapping_fallback": False,
        }

    probe = _startup_probe(model_id=model_id, device=device)
    windows = _build_sentence_windows(
        sentences=sentences,
        max_window_tokens=probe.max_window_tokens,
        overlap_tokens=probe.overlap_tokens,
        tokenizer=probe.tokenizer,
    )

    boundary_count = len(sentences) - 1
    best_left_context = [-1] * boundary_count
    ppl_scores: list[float | None] = [None] * boundary_count
    mapping_mode_counts: Counter[str] = Counter()

    for start, end in windows:
        sentences_window = sentences[start:end]
        scored_boundaries, token_count, mapping_mode = _score_window_boundaries(
            sentences_window=sentences_window,
            probe=probe,
        )
        mapping_mode_counts[mapping_mode] += 1

        if token_count > probe.max_window_tokens:
            raise ValueError(
                f"Window ({start}, {end}) returned {token_count} prompt tokens, "
                f"exceeding max_window_tokens={probe.max_window_tokens}"
            )

        for local_idx, left_context_tokens, ppl in scored_boundaries:
            global_boundary_idx = start + local_idx
            if left_context_tokens > best_left_context[global_boundary_idx]:
                best_left_context[global_boundary_idx] = left_context_tokens
                ppl_scores[global_boundary_idx] = ppl

    missing = [idx for idx, value in enumerate(ppl_scores) if value is None]
    if missing:
        raise ValueError(f"Window stitching missed boundary indices: {missing[:10]}")

    diagnostics = {
        "window_count": len(windows),
        "windowing_used": len(windows) > 1,
        "boundary_mapping_mode_counts": dict(mapping_mode_counts),
        "used_boundary_mapping_fallback": False,
    }
    return [float(value) for value in ppl_scores if value is not None], diagnostics


def _find_cognitive_peaks(ppl_scores: list[float]) -> set[int]:
    if not ppl_scores:
        return set()

    smooth = np.convolve(ppl_scores, np.ones(3) / 3, mode="same") if len(ppl_scores) > 3 else ppl_scores

    median = np.median(smooth)
    q75, q25 = np.percentile(smooth, [75, 25])
    iqr = q75 - q25

    threshold = max(median + (0.5 * iqr), median + 0.1)

    peaks, _ = find_peaks(smooth, height=threshold, distance=1)
    return set(peaks)


def segment_trace(
    trace: str,
    model_id: str = PPL_MODEL_ID,
    device: str | None = None,
) -> list[str]:
    chunks, _, _ = segment_trace_with_ppl(
        trace,
        model_id=model_id,
        device=device,
    )
    return chunks


def segment_trace_with_ppl(
    trace: str,
    model_id: str = PPL_MODEL_ID,
    device: str | None = None,
) -> tuple[list[str], list[float], list[list[float]]]:
    chunks, ppl_scores, chunk_ppl, _ = segment_trace_with_ppl_debug(
        trace,
        model_id=model_id,
        device=device,
    )
    return chunks, ppl_scores, chunk_ppl


def segment_trace_with_ppl_debug(
    trace: str,
    model_id: str = PPL_MODEL_ID,
    device: str | None = None,
) -> tuple[list[str], list[float], list[list[float]], dict[str, Any]]:
    sentences = split_into_sentences(trace)
    if len(sentences) < 2:
        diagnostics = {
            "sentence_count": len(sentences),
            "window_count": 0,
            "windowing_used": False,
            "boundary_mapping_mode_counts": {},
            "used_boundary_mapping_fallback": False,
            "boundary_mapping_strategy": "tokenizer_offsets",
            "fallback_path_used": None,
            "cut_after_sentence_indices": [],
        }
        return [trace], [], [[]], diagnostics

    ppl_scores, boundary_diagnostics = _compute_boundary_perplexity(
        sentences,
        model_id=model_id,
        device=device,
    )
    peaks = _find_cognitive_peaks(ppl_scores)

    chunks = []
    current = [sentences[0]]
    chunk_ppl: list[list[float]] = []
    current_ppl: list[float] = []
    cut_after_sentence_indices: list[int] = []

    for i in range(len(ppl_scores)):
        next_sent = sentences[i + 1]
        word_count = sum(len(s.split()) for s in current)
        if i in peaks and word_count >= 15:
            chunks.append(" ".join(current))
            chunk_ppl.append(current_ppl)
            cut_after_sentence_indices.append(i + 1)
            current = [next_sent]
            current_ppl = []
        else:
            current.append(next_sent)
            current_ppl.append(ppl_scores[i])

    if current:
        chunks.append(" ".join(current))
        chunk_ppl.append(current_ppl)

    diagnostics = {
        "sentence_count": len(sentences),
        "window_count": int(boundary_diagnostics.get("window_count", 0)),
        "windowing_used": bool(boundary_diagnostics.get("windowing_used", False)),
        "boundary_mapping_mode_counts": boundary_diagnostics.get("boundary_mapping_mode_counts", {}),
        "used_boundary_mapping_fallback": False,
        "boundary_mapping_strategy": "tokenizer_offsets",
        "fallback_path_used": None,
        "model_id": model_id,
        "device": _startup_probe(model_id=model_id, device=device).device,
        "cut_after_sentence_indices": cut_after_sentence_indices,
    }
    return chunks, ppl_scores, chunk_ppl, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/mrt_shade_tool_use_sample.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/mrt_shade_tool_use_segmented.jsonl"))
    parser.add_argument("--model-id", default=PPL_MODEL_ID)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N trajectories")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if args.limit is not None and count >= args.limit:
                break
            row = json.loads(line)
            blocks = row.get("reasoning_blocks")
            if not isinstance(blocks, list):
                raise ValueError(f"Missing reasoning_blocks in input row {count + 1}")
            segmented_blocks = []
            for block in blocks:
                chunks, scores, _, diagnostics = segment_trace_with_ppl_debug(
                    block, model_id=args.model_id, device=args.device
                )
                segmented_blocks.append(
                    {"chunks": chunks, "boundary_scores": scores, "diagnostics": diagnostics}
                )
            result = {
                "source_path": row["source_path"],
                "task": row["task"],
                "side_task_success": row["side_task_success"],
                "model_id": args.model_id,
                "segmented_reasoning_blocks": segmented_blocks,
            }
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1
            print(f"Segmented {count}: {row['source_path']}", flush=True)
    print(f"Wrote {count} segmented trajectories to {args.output}")


if __name__ == "__main__":
    main()
