# segmented-monitorability

## MRT trajectory sample

Install the downloader dependencies and run:

```sh
python3 -m pip install huggingface_hub
python3 src/setup_mrt.py
```

The script downloads one [ScaleAI MRT](https://huggingface.co/datasets/ScaleAI/mrt) SHADE-Arena tool-use run family: Claude 3.5 Sonnet as the agent, the `transcripts_no_hint_selective_thinking_time_basic` condition, and the repository's `side_task_evaluation` as an external success label. It writes 50 trajectories to `data/mrt_shade_tool_use_sample.jsonl` and a source manifest beside it. This run family has 24 failures, so the nearest 50-row balance is 24 failures and 26 successes. The output preserves the complete `conversation` and extracts ordered `<thinking>` / `<model_thinking>` spans into `reasoning_blocks`. It excludes the original `monitoring_response` to avoid leaking an earlier monitor's judgment into new experiments.

The manifest records the exact dataset commit, source paths, and task-by-outcome counts. Outcomes in this run family vary strongly by task, so compare monitor representations on the same frozen 50 traces and report the task mix. For monitor inputs, omit the original system prompt and the `side_task_success` field; use that field only for scoring. MRT is licensed CC BY-NC 4.0. Use `--target`, `--output`, or `--revision` to change the defaults.
