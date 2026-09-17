import math
import re
from collections import Counter
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
from scipy.signal import find_peaks

PPL_MODEL_PORT = 8081
PPL_TOP_LOGPROBS = 20
PPL_REQUEST_TIMEOUT = 300
PPL_DEFAULT_MAX_WINDOW_TOKENS = 12000
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
    base_url: str
    model_id: str | None
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


def _normalize_top_logprobs(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        normalized: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            logprob = entry.get("logprob")
            if not isinstance(logprob, (int, float)):
                continue
            token = entry.get("token")
            if token is None:
                token = entry.get("decoded_token")
            normalized.append({"token": "" if token is None else str(token), "logprob": float(logprob)})
        return normalized

    if isinstance(raw, dict):
        normalized: list[dict[str, Any]] = []
        for token, logprob in raw.items():
            if isinstance(logprob, (int, float)):
                normalized.append({"token": str(token), "logprob": float(logprob)})
                continue
            if isinstance(logprob, dict):
                value = logprob.get("logprob")
                if not isinstance(value, (int, float)):
                    continue
                decoded = logprob.get("decoded_token")
                if decoded is None:
                    decoded = logprob.get("token")
                if decoded is None:
                    decoded = token
                normalized.append({"token": str(decoded), "logprob": float(value)})
        return normalized

    return []


def _normalize_base_url(port: int, base_url: str | None) -> str:
    if base_url:
        trimmed = base_url.strip().rstrip("/")
    else:
        trimmed = f"http://localhost:{port}"

    if trimmed.endswith("/v1/completions"):
        return trimmed[:-len("/v1/completions")]
    if trimmed.endswith("/v1"):
        return trimmed

    parsed = urlparse(trimmed)
    if parsed.scheme and parsed.netloc:
        return trimmed
    return f"http://localhost:{port}"


def _completions_url(base_url: str) -> str:
    return f"{base_url}/v1/completions"


def _extract_prompt_logprob_payload(
    data: dict[str, Any],
) -> tuple[list[str], list[list[dict[str, Any]]], list[int]]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return [], [], []

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return [], [], []

    logprobs = first_choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return [], [], []

    tokens = logprobs.get("tokens")
    top_logprobs = logprobs.get("top_logprobs")
    text_offsets = logprobs.get("text_offset")
    if not isinstance(tokens, list) or not isinstance(top_logprobs, list) or not isinstance(text_offsets, list):
        return [], [], []

    n = min(len(tokens), len(top_logprobs), len(text_offsets))
    if n <= 0:
        return [], [], []

    prompt_tokens = ["" if token is None else str(token) for token in tokens[:n]]
    top_per_token = [_normalize_top_logprobs(entry) for entry in top_logprobs[:n]]
    offsets: list[int] = []
    for value in text_offsets[:n]:
        if isinstance(value, int):
            offsets.append(value)
            continue
        if isinstance(value, float) and value.is_integer():
            offsets.append(int(value))
            continue
        return [], [], []

    return prompt_tokens, top_per_token, offsets


def _post_completions(
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    response = httpx.post(
        _completions_url(base_url),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Invalid completions response payload")
    return data


def _request_prompt_logprobs(
    prompt: str,
    base_url: str,
    model_id: str | None,
    timeout: int,
    top_logprobs: int = PPL_TOP_LOGPROBS,
) -> tuple[list[str], list[list[dict[str, Any]]], list[int], dict[str, Any]]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": 0,
        "echo": True,
        "logprobs": top_logprobs,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model_id:
        payload["model"] = model_id

    data = _post_completions(base_url=base_url, payload=payload, timeout=timeout)
    prompt_tokens, top_per_token, token_offsets = _extract_prompt_logprob_payload(data)
    if not prompt_tokens or not top_per_token or not token_offsets:
        raise ValueError(
            "Prompt logprobs were not returned. "
            "This deployment must support /v1/completions with echo=true and max_tokens=0."
        )
    return prompt_tokens, top_per_token, token_offsets, data


def _parse_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _resolve_model_metadata(
    base_url: str,
    requested_model_id: str | None,
) -> tuple[str | None, int, list[str]]:
    response = httpx.get(f"{base_url}/v1/models", timeout=PPL_REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Invalid /v1/models response")

    models = data.get("data")
    if not isinstance(models, list) or not models:
        raise ValueError("No models listed by /v1/models")

    selected: dict[str, Any] | None = None
    if requested_model_id:
        for model in models:
            if isinstance(model, dict) and model.get("id") == requested_model_id:
                selected = model
                break
        if selected is None:
            available = [model.get("id") for model in models if isinstance(model, dict)]
            raise ValueError(f"Requested model '{requested_model_id}' not in /v1/models: {available}")
    else:
        first = models[0]
        if isinstance(first, dict):
            selected = first

    resolved_model_id: str | None = requested_model_id
    max_model_len = PPL_DEFAULT_MAX_WINDOW_TOKENS
    if isinstance(selected, dict):
        selected_id = selected.get("id")
        if isinstance(selected_id, str) and selected_id.strip():
            resolved_model_id = selected_id
        parsed_max = _parse_int(selected.get("max_model_len"))
        if parsed_max is not None and parsed_max > 0:
            max_model_len = parsed_max

    tokenizer_refs: list[str] = []

    def add_tokenizer_ref(value: Any) -> None:
        if not isinstance(value, str):
            return
        cleaned = value.strip()
        if not cleaned:
            return
        if cleaned not in tokenizer_refs:
            tokenizer_refs.append(cleaned)

    add_tokenizer_ref(requested_model_id)
    if isinstance(selected, dict):
        add_tokenizer_ref(selected.get("id"))
        add_tokenizer_ref(selected.get("root"))
        add_tokenizer_ref(selected.get("parent"))
        add_tokenizer_ref(selected.get("model"))
        add_tokenizer_ref(selected.get("tokenizer"))
        add_tokenizer_ref(selected.get("tokenizer_id"))
    add_tokenizer_ref(resolved_model_id)

    # Common served-model-name to HF-id conversion for local Qwen3 servers.
    for tokenizer_ref in list(tokenizer_refs):
        match = re.fullmatch(r"qwen3-([0-9]+(?:\.[0-9]+)?)b", tokenizer_ref.lower())
        if match:
            add_tokenizer_ref(f"Qwen/Qwen3-{match.group(1)}B")

    if not tokenizer_refs:
        raise ValueError("Unable to derive tokenizer reference from /v1/models response")
    return resolved_model_id, max_model_len, tokenizer_refs


@lru_cache(maxsize=16)
def _load_tokenizer(tokenizer_refs: tuple[str, ...]) -> tuple[Any, str]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for tokenizer-based window sizing. "
            "Install dependencies with `uv sync`."
        ) from exc

    errors: list[str] = []
    for tokenizer_ref in tokenizer_refs:
        for local_files_only in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_ref,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                )
                return tokenizer, tokenizer_ref
            except Exception as exc:
                scope = "local" if local_files_only else "remote"
                errors.append(f"{tokenizer_ref} ({scope}): {exc}")

    raise RuntimeError(
        "Failed to load tokenizer for chunking model. Tried refs: "
        f"{list(tokenizer_refs)}. Last errors: {errors[-3:]}"
    )


@lru_cache(maxsize=16)
def _startup_probe(
    port: int,
    base_url: str | None,
    model_id: str | None,
) -> _ChunkingRuntimeProbe:
    normalized_base_url = _normalize_base_url(port=port, base_url=base_url)
    resolved_model_id, model_context_window, tokenizer_refs = _resolve_model_metadata(
        base_url=normalized_base_url,
        requested_model_id=model_id,
    )
    tokenizer, tokenizer_ref = _load_tokenizer(tuple(tokenizer_refs))
    max_window_tokens = max(2,model_context_window)
    overlap_tokens = max(1, int(max_window_tokens * PPL_WINDOW_OVERLAP_RATIO))

    return _ChunkingRuntimeProbe(
        base_url=normalized_base_url,
        model_id=resolved_model_id,
        tokenizer=tokenizer,
        tokenizer_ref=tokenizer_ref,
        max_window_tokens=max_window_tokens,
        overlap_tokens=overlap_tokens,
    )


def probe_chunking_runtime(
    port: int = PPL_MODEL_PORT,
    base_url: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    probe = _startup_probe(port=port, base_url=base_url, model_id=model_id)
    return {
        "base_url": probe.base_url,
        "model_id": probe.model_id,
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
        raise ValueError("Missing text_offset data for boundary mapping")

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
        raise ValueError("Invalid boundary token positions from text_offset mapping")
    return positions


def _map_boundaries_with_tokenizer(
    sentences_window: list[str],
    tokenizer: Any,
    token_count: int,
) -> list[int]:
    boundary_count = len(sentences_window) - 1
    if boundary_count <= 0:
        return []

    raw_positions: list[int] = []
    prefix = sentences_window[0]
    for sentence in sentences_window[1:]:
        raw_positions.append(_count_tokens_with_tokenizer(tokenizer, prefix))
        prefix = f"{prefix} {sentence}"

    positions = _coerce_boundary_positions(
        raw_positions=raw_positions,
        token_count=token_count,
        expected_count=boundary_count,
    )
    if not _is_valid_boundary_positions(positions, token_count, boundary_count):
        raise ValueError("Invalid boundary token positions from tokenizer fallback mapping")
    return positions


def _perplexity_from_top_logprobs(top_logprobs: list[dict[str, Any]]) -> float:
    logprobs = [
        float(entry["logprob"])
        for entry in top_logprobs
        if isinstance(entry, dict) and isinstance(entry.get("logprob"), (int, float))
    ]
    if not logprobs:
        return 1.0

    max_logprob = max(logprobs)
    exp_shifted = [math.exp(logprob - max_logprob) for logprob in logprobs]
    total = sum(exp_shifted)
    if total <= 0:
        return 1.0

    probs = [value / total for value in exp_shifted]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return 2 ** entropy


def _score_window_boundaries(
    sentences_window: list[str],
    base_url: str,
    model_id: str | None,
    tokenizer: Any,
) -> tuple[list[tuple[int, int, float]], int, str]:
    if not sentences_window:
        return [], 0, "text_offset"

    prompt = " ".join(sentences_window)
    prompt_tokens, top_per_token, token_offsets, _ = _request_prompt_logprobs(
        prompt=prompt,
        base_url=base_url,
        model_id=model_id,
        timeout=PPL_REQUEST_TIMEOUT,
    )
    token_count = len(prompt_tokens)

    boundary_offsets = _sentence_boundary_char_offsets(sentences_window)
    if not boundary_offsets:
        return [], token_count, "text_offset"

    try:
        boundary_positions = _map_boundaries_with_offsets(
            boundary_offsets=boundary_offsets,
            token_offsets=token_offsets,
            token_count=token_count,
        )
        mapping_mode = "text_offset"
    except ValueError as offset_exc:
        try:
            boundary_positions = _map_boundaries_with_tokenizer(
                sentences_window=sentences_window,
                tokenizer=tokenizer,
                token_count=token_count,
            )
            mapping_mode = "tokenizer_fallback"
        except ValueError as tokenizer_exc:
            raise ValueError(
                "Boundary mapping failed via text_offset and tokenizer fallback. "
                f"text_offset_error={offset_exc}; tokenizer_error={tokenizer_exc}"
            ) from tokenizer_exc

    scored_boundaries: list[tuple[int, int, float]] = []
    for local_idx, token_pos in enumerate(boundary_positions):
        ppl = _perplexity_from_top_logprobs(top_per_token[token_pos])
        scored_boundaries.append((local_idx, token_pos, ppl))
    return scored_boundaries, token_count, mapping_mode


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
    port: int,
    base_url: str | None = None,
    model_id: str | None = None,
) -> tuple[list[float], dict[str, Any]]:
    if len(sentences) < 2:
        return [], {
            "window_count": 0,
            "windowing_used": False,
            "boundary_mapping_mode_counts": {},
            "used_boundary_mapping_fallback": False,
        }

    probe = _startup_probe(port=port, base_url=base_url, model_id=model_id)
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
            base_url=probe.base_url,
            model_id=probe.model_id,
            tokenizer=probe.tokenizer,
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
        "used_boundary_mapping_fallback": mapping_mode_counts.get("tokenizer_fallback", 0) > 0,
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
    port: int = PPL_MODEL_PORT,
    base_url: str | None = None,
    model_id: str | None = None,
) -> list[str]:
    chunks, _, _ = segment_trace_with_ppl(
        trace,
        port=port,
        base_url=base_url,
        model_id=model_id,
    )
    return chunks


def segment_trace_with_ppl(
    trace: str,
    port: int = PPL_MODEL_PORT,
    base_url: str | None = None,
    model_id: str | None = None,
) -> tuple[list[str], list[float], list[list[float]]]:
    chunks, ppl_scores, chunk_ppl, _ = segment_trace_with_ppl_debug(
        trace,
        port=port,
        base_url=base_url,
        model_id=model_id,
    )
    return chunks, ppl_scores, chunk_ppl


def segment_trace_with_ppl_debug(
    trace: str,
    port: int = PPL_MODEL_PORT,
    base_url: str | None = None,
    model_id: str | None = None,
) -> tuple[list[str], list[float], list[list[float]], dict[str, Any]]:
    sentences = split_into_sentences(trace)
    if len(sentences) < 2:
        diagnostics = {
            "sentence_count": len(sentences),
            "window_count": 0,
            "windowing_used": False,
            "boundary_mapping_mode_counts": {},
            "used_boundary_mapping_fallback": False,
            "boundary_mapping_strategy": "text_offset",
            "fallback_path_used": None,
        }
        return [trace], [], [[]], diagnostics

    ppl_scores, boundary_diagnostics = _compute_boundary_perplexity(
        sentences,
        port=port,
        base_url=base_url,
        model_id=model_id,
    )
    peaks = _find_cognitive_peaks(ppl_scores)

    chunks = []
    current = [sentences[0]]
    chunk_ppl: list[list[float]] = []
    current_ppl: list[float] = []

    for i in range(len(ppl_scores)):
        next_sent = sentences[i + 1]
        word_count = sum(len(s.split()) for s in current)
        if i in peaks and word_count >= 15:
            chunks.append(" ".join(current))
            chunk_ppl.append(current_ppl)
            current = [next_sent]
            current_ppl = []
        else:
            current.append(next_sent)
            current_ppl.append(ppl_scores[i])

    if current:
        chunks.append(" ".join(current))
        chunk_ppl.append(current_ppl)

    used_fallback = bool(boundary_diagnostics.get("used_boundary_mapping_fallback"))
    diagnostics = {
        "sentence_count": len(sentences),
        "window_count": int(boundary_diagnostics.get("window_count", 0)),
        "windowing_used": bool(boundary_diagnostics.get("windowing_used", False)),
        "boundary_mapping_mode_counts": boundary_diagnostics.get("boundary_mapping_mode_counts", {}),
        "used_boundary_mapping_fallback": used_fallback,
        "boundary_mapping_strategy": (
            "text_offset_then_tokenizer_fallback" if used_fallback else "text_offset"
        ),
        "fallback_path_used": "tokenizer_boundary_mapping" if used_fallback else None,
    }
    return chunks, ppl_scores, chunk_ppl, diagnostics
