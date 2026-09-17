Dataset selection

Pick 50 public agentic trajectories, balanced 25/25 on an externally provided binary outcome. The important requirement is not the benchmark name; it is that the traces actually contain enough reasoning for segmentation to matter. I’d pre-filter for something like ≥800 reasoning tokens, ≥3 distinct assistant reasoning blocks, ≥2 tool/action calls, complete trajectory, and a reliable binary outcome label. Avoid traces where the assistant mostly emits tool calls with one-line thoughts. Freeze the 50 before looking at segmentation results.

Experiment 1 — does entropy chunking produce better semantic units?

This is just the chunking-quality sanity check.

Compare only:

LEX — lexical markers such as Wait / But / However / Therefore / Actually / Let me...
FIX4 — every 4 sentences
ENTROPY — your current predictive-entropy / effective-branching-factor peaks

Tool calls and observations are preserved as structural separators; these methods operate only inside reasoning blocks.

For 10–12 traces, manually mark semantic-turn boundaries. Then calculate only:

$$ \text{Boundary F1} = \frac{2PR}{P+R} $$

with a prediction counted as correct if it lies within ±1 sentence of a human boundary.

Across all 50, report just one descriptive statistic:

$$ \text{median words per chunk} $$

and optionally % tiny chunks (<10 words) if lexical markers visibly fragment the text.

That is enough. One table:

Method	Boundary F1 ↑	Median words/chunk	<10-word chunks ↓
LEX			
FIX4			
ENTROPY			

This replaces all the old MMLU chunk-quality numbers.

Experiment 2 — establish that this dataset actually has CoT monitorability

Use one fixed LLM monitor and the trajectory's external binary label as ground truth. The LLM is the monitor, not the judge.

Run the three standard observation scopes:

ACTION — actions/tool calls + observations, no reasoning
COT — reasoning only
FULL — complete rollout

Ask the monitor for a binary judgment using the same prompt everywhere.

Keep only the metric already used in current monitorability evaluation:

$$ G^2 = TPR \times TNR $$

where

$$ TPR=\frac{TP}{TP+FN}, \qquad TNR=\frac{TN}{TN+FP}. $$

One table:

Monitor input	\(G^2\) ↑
ACTION	
COT	
FULL	

This answers one thing: does the reasoning trace actually contain useful information beyond the observable trajectory?

No Brier, AUROC, calibration curves, etc. needed for this blog.

Experiment 3 — does segmentation make that monitorable evidence easier to expose?

Now keep FULL information fixed, and only change how reasoning is segmented.

Compare:

RAW — original reasoning blocks
FIX4 — four-sentence chunks
RANDOM-K — random sentence boundaries, matched to the number of ENTROPY chunks
ENTROPY — your semantic boundaries

I’d drop UNIFORM-K; RANDOM-K already gives the important causal control.

Run the same monitor twice:

Full rollout

Give the whole trajectory under each representation and report:

$$ G^2_{\text{full}} $$

This tells us whether segmentation helps ordinary finished-rollout monitoring.

Half rollout

Take the trajectory prefix ending at the last available boundary before 50% of raw trajectory tokens, and run the same monitor:

$$ G^2_{50\%} $$

This is the simplest possible version of “early monitorability.”

Now the whole result is:

Representation	\(G^2_{50\%}\) ↑	\(G^2_{\text{full}}\) ↑
RAW		
FIX4		
RANDOM-K		
ENTROPY		

If ENTROPY beats FIX4/RANDOM at 50% but everything converges at the end, that is actually a very nice result:

Semantic segmentation did not add information to the rollout; it exposed useful evidence earlier.

If everything is equal, that's also useful: semantic chunks may be nicer representations without improving monitorability.