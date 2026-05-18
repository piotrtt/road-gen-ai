# HybridGenerator — Pilot Run

> Companion note to *Prompt Optimization* (`experiments/prompt_optimization/`) and *Temperature Sweep* (`experiments/temperature_sweep/`). Documents the first end-to-end run of the new LLM + deterministic-diversity-metric feedback loop on Gemini 2.5 Pro.

## 1. What this approach is

The previous "hybrid" generator only combined random candidate sampling with a similarity-based picker — no LLM was involved. To make that distinction explicit it has been renamed **DiversityDrivenGenerator** (`src/generators/diversity_driven_generator.py`, key `diversity_driven`).

The new **HybridGenerator** (`src/generators/hybrid_generator.py`, key `hybrid`) is the actual hybrid: an LLM proposes candidates, the project's deterministic similarity metric scores them, and the LLM chooses the winner via a tool call. Across rounds, a rising threshold keeps the loop running until further progress can't be made.

## 2. Algorithm (per network)

For each of `max_rounds = 3` rounds:

1. **Propose** — One LLM round-trip forces a single call to the `propose_candidates` tool, returning **3 distinct candidate road networks** (`NUM_CANDIDATES = 3`).
2. **Score** — Server-side, each candidate's mean `combined_similarity` is computed against the networks already committed in this batch (50% Levenshtein topological + 50% normalized parameter geometric, identical to the metric used everywhere else).
3. **Select** — A second LLM round-trip presents the scores and forces a call to `select_best(index, justification)`. Out-of-range indices fall back to server-side argmin with a logged warning.
4. **Threshold update** — The round's second-best similarity becomes the threshold the *next* round must satisfy (`chosen_score ≤ threshold`) to keep iterating. The globally-best-seen candidate is tracked separately and is what we ultimately commit, so intermediate degradations don't replace earlier wins.
5. **Stop** — When the threshold check fails, the loop exits and the running best is committed. The threshold resets between networks.

Round 1 always commits (no prior threshold). After committing, the next network's loop starts fresh.

## 3. Pilot result — Gemini 2.5 Pro, N = 10

Command:
```
uv run python src/run_evaluation.py --quantity 10 --components 7 \
  --approaches hybrid --llm-models gemini-2.5-pro --repeats 1
```

Output (`outputs/evaluation/20260518_032845/`):

| Metric | Value |
|---|---:|
| Mean combined similarity | **0.126** |
| Topological mean | 0.121 |
| Geometric mean | 0.131 |
| Wall-clock | 978 s (~16 min) |
| Reject rate | 0 % |

## 4. Comparison vs prior Gemini results

| Approach | Gemini mean similarity | Note |
|---|---:|---|
| `llm:gemini` (full benchmark, PR #2) | 0.183 | Single-candidate-per-network LLM, n=20 |
| `gemini` temperature sweep best | 0.139 | Fine-grid optimum at T=1.1, n=10 |
| **`hybrid:gemini`** (this pilot) | **0.126** | New feedback loop, n=10 |

The hybrid run improves on plain LLM Gemini by ~31 % and on the best single-candidate Gemini setup by ~9 %. With only 45 pairs (n=10), this is directional rather than conclusive, but the gap is wide enough that the mechanism is clearly doing useful work.

## 5. Loop behaviour observations

The 30 rounds of logs from this run show all four signatures we wanted to see:

- **Mid-loop improvement** — `round=1 chosen=0.1750 → round=2 chosen=0.0945 ✓ → round=3 chosen=0.1197` (round 2 was a clean win; round 3 was worse but still ≤ threshold).
- **Threshold correctly stopping the loop** — e.g. `round=2 chosen=0.1739 threshold=0.1487 → stop` and `round=3 chosen=0.1702 threshold=0.1580 → stop`.
- **`running_best` preserved across noisy rounds** — `round=3 chosen=0.1748 running_best=0.1236 → pass` (the round-2 win wasn't lost when round 3 came back worse).
- **No fallback argmin warnings** — Gemini always returned 3 valid candidates and picked a legitimate index; the `select_best` validation never had to override the model's choice.

Roughly seven of the ten networks ran the full three rounds; three exited early via threshold check — a healthy mix indicating the feedback loop isn't trivially passing or trivially stopping.

## 6. Cost & speed

About 98 s per network on Gemini (~3.3× plain-LLM Gemini, ~16 min for 10 networks). The factor is bounded above by `2 round-trips × max_rounds = 6` LLM calls per network in the worst case, and in this run averaged ~5 round-trips per network. For GPT-5.1 and Claude (faster than Gemini in the sweep results) the cost factor will be lower in absolute wall-clock terms.

## 7. Limitations & next steps

- **Sample size**: n=10 → 45 pairs. Re-run with `--repeats 3 --quantity 20` for a thesis-grade variance estimate.
- **Single model so far**: validate `hybrid:gpt5` and `hybrid:claude` at the same scale.
- **No ablation on `max_rounds`**: 3 was picked from the user's spec; a sweep over `{1, 2, 3, 5}` would quantify how much each extra round buys.
- **No ablation on `NUM_CANDIDATES`**: fixed at 3 per the user's choice. A 5-candidate variant would cost more tokens but might give the LLM more headroom to find a clear winner.
- **LLM-vs-server picker**: the LLM currently picks the index after seeing the scores, but for the cases where it just argmins, the second round-trip is pure overhead. A future ablation comparing "LLM picks" vs "server picks (argmin)" would say whether the LLM's judgement adds anything beyond the deterministic score.
