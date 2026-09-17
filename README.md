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

For actual monitoring, use one fixed model at an OpenAI-compatible chat endpoint:

```sh
MONITOR_API_BASE=http://localhost:1234 uv run python src/run_monitorability.py --model YOUR_MONITOR_MODEL
```

Set `MONITOR_API_KEY` if the endpoint requires a bearer token. The runner writes one verdict per trajectory and condition to `data/mrt_monitor_verdicts.jsonl`, followed by a `.summary.json` with TPR, TNR, and G². Use `--resume` after an interrupted run. Experiment 2 uses ACTION, COT, and FULL; Experiment 3 compares RAW, FIX4, RANDOM-K, and ENTROPY under FULL at complete and halfway prefixes. The halfway cutoff is fixed once per raw trajectory using the Qwen tokenizer, then reused for every representation. No reasoning block is split across the cutoff.
