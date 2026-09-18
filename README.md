# segmented-monitorability

## MRT trajectory sample

Install the project dependencies and run:

```sh
uv sync
uv run python src/setup_mrt.py
```

The script downloads one [ScaleAI MRT](https://huggingface.co/datasets/ScaleAI/mrt) SHADE-Arena tool-use run family: Claude 3.5 Sonnet as the agent, the `transcripts_no_hint_selective_thinking_time_basic` condition, and the repository's `side_task_evaluation` as an external success label. It writes 50 trajectories to `data/mrt_shade_tool_use_sample.jsonl` and a source manifest beside it. This run family has 24 failures, so the nearest 50-row balance is 24 failures and 26 successes. The output preserves the complete `conversation` and extracts ordered `<thinking>` / `<model_thinking>` spans into `reasoning_blocks`. It excludes the original `monitoring_response` to avoid leaking an earlier monitor's judgment into new experiments.

The manifest records the exact dataset commit, source paths, and task-by-outcome counts. Outcomes in this run family vary strongly by task, so compare monitor representations on the same frozen 50 traces and report the task mix. For monitor inputs, omit the original system prompt and the `side_task_success` field; use that field only for scoring. MRT is licensed CC BY-NC 4.0. Use `--target`, `--output`, or `--revision` to change the defaults.

## Entropy segmentation

The chunker scores each reasoning block with the local `Qwen/Qwen3-0.6B-Base` model. Run:

```sh
uv run python src/core_chunker.py
```

This reads the MRT JSONL and writes `data/mrt_shade_tool_use_segmented.jsonl`. Use `--limit 1` for a first run, or `--device cpu|mps|cuda` to select hardware. The model loads once, and reasoning blocks remain separate. The boundary score is the effective branching factor of Qwen's top 20 next-token probabilities; it is a proxy model's score, not the original agent's internal uncertainty.

## Blog experiments

Generate the saved reasoning representations and a template for manual boundary labels:

```sh
uv run python src/run_semantic_units.py --annotation-template data/mrt_boundary_annotations.jsonl
```

The runner writes `data/mrt_semantic_units.jsonl` and a `.summary.json` with chunk counts, median words per chunk, and the fraction under 10 words. In the annotation template, set `reviewed` to `true` for each inspected reasoning block and fill `gold_cuts` with sentence indices **after** which a turn ends (for example, `[2, 5]`). Then rerun with `--annotations data/mrt_boundary_annotations.jsonl` to add boundary F1 within ±1 sentence. The template contains the first 12 traces and does not show outcome labels. Use a separate `--output` when testing with `--limit 1`.

Check monitor inputs without sending any API calls:

```sh
uv run python src/run_monitorability.py --dry-run --output data/mrt_monitor_prompts.jsonl
```

For OpenRouter monitoring, put your API key after `MONITOR_API_KEY=` in the ignored `.env` file at the repo root. Set `MONITOR_API_BASE=https://openrouter.ai/api/v1` and `MONITOR_MODEL=typesafe/jev-1.13`. Jev returns a typed yes/no probability through OpenRouter's decisions endpoint; the runner thresholds it at 0.5. The version is pinned so the monitor stays fixed for all conditions. Then run:

```sh
uv run python src/run_monitorability.py
```

The runner writes one verdict per trajectory and condition to `data/mrt_jev_verdicts.jsonl`, followed by a `.summary.json` with TPR, TNR, and G². Earlier Qwen and DeepSeek verdict files remain separate. Each Jev verdict records the number and sizes of input windows, their probabilities, providers, returned model versions, and usage. Use `--resume` after an interrupted run. Experiment 2 uses ACTION, COT, and FULL; Experiment 3 compares RAW, FIX4, RANDOM-K, and ENTROPY under FULL at complete and halfway prefixes. The halfway cutoff is fixed once per raw trajectory using the Qwen tokenizer, then reused for every representation. No reasoning block is split across the cutoff.

Jev has a 32K-token context. The runner greedily packs complete trajectory events into windows under a 28K Qwen proxy-token budget; FULL uses the same event boundaries under every reasoning representation. Every event is retained, and the maximum yes probability across windows gives one verdict. This max aggregation is a heuristic for long trajectories: a task that requires combining evidence across windows may be misjudged, so report the window count alongside results. The preview of the frozen sample requires 559 calls: 451 conditions fit in one window, 39 need two, and 10 need three. The largest preview window used 28,163 Jev input tokens in a live check. At Jev's listed $0.042/M input tokens with free output, roughly 6.59 million Qwen proxy input tokens imply about $0.28 input cost, subject to Jev tokenization. Requests are paced at least one second apart; use `--request-interval SECONDS` to change this. `--jev-input-budget` controls window size (at most 30,000 proxy tokens). Provider 429 responses back off for at least 30, 60, 120, then 120 seconds (or longer if `Retry-After` requests it).
