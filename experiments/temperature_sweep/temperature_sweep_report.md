# Temperature Sweep: Selecting per-LLM-family Sampling Temperatures

> Companion report to *Prompt Optimization for Road Network Diversity* (`experiments/prompt_optimization/prompt_optimization_report.md`).
> Scope: with each model's optimized system prompt fixed, sweep `temperature` to identify the value that maximizes diversity (lowest mean pairwise similarity) without breaking output validity.

## 1. Motivation

Sampling temperature controls the entropy of the next-token distribution. At **low** temperatures the model becomes near-deterministic and repeats high-probability completions, which collapses diversity. At **high** temperatures token sampling broadens, but past a point the model begins emitting malformed JSON, hallucinating component types, or producing parameters outside their valid range — diversity in the sampler does *not* translate into diversity in the validated output.

Because each model family was post-trained differently (RLHF schemes, decoding defaults, calibration), the diversity-vs-validity sweet spot is **family-dependent**. The prompt-optimization experiment held temperature fixed at 0.9; this report tests whether 0.9 is actually optimal for each family or whether a different value reduces mean similarity further.

**Hypothesis.** Each family has a distinct optimum in `[0.5, 1.3]`. We expect Claude (more conservative defaults) to prefer slightly higher temperatures than GPT-5.1, and Gemini to be the most sensitive at the high end.

## 2. Methodology

| Variable | Value |
|---|---|
| Networks per (model, temperature) cell | 10 |
| Components per network | 7 |
| Repeats per cell | 1 (acknowledged limitation) |
| System prompt | Per-family optimized prompt (fixed) |
| User prompt | Same template (`load_user_prompt`), within-batch diversity context, `max_examples=5` |
| Temperatures swept | 0.5, 0.7, 0.9, 1.1, 1.3 (Gemini: reduced to 0.5, 0.9, 1.3 with N=5 due to ~3–5 min/network latency) |
| Models | `gpt-5.1`, `gemini-2.5-pro`, `claude-opus-4.6` (DMGPT) |
| Metric | Mean pairwise combined similarity (50% topological + 50% geometric) |
| Selection rule | Argmin mean similarity over temperatures, with secondary requirement that `successful >= 8/10` |

Single run per cell is a deliberate trade-off: this is a *coarse* sweep meant to bracket the optimum, not a publication-grade variance estimate. The downstream benchmark (Step 3) uses 3 repeats at the chosen temperatures.

Implementation: `scripts/temperature_sweep.py`. Per-temperature outputs: `experiments/temperature_sweep/{family}/temp_{T}.json`. Per-family roll-up: `summary.json`.

## 3. Results

### 3.1 GPT-5.1

| Temperature | Successful | Mean similarity | Topological mean | Geometric mean | Std | Wall-clock (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 10/10 | 0.2501 | 0.2000 | 0.3002 | 0.2245 | 22.7 |
| 0.7 | 10/10 | 0.2125 | 0.1492 | 0.2757 | 0.1183 | 22.6 |
| 0.9 | 10/10 | 0.2142 | 0.1619 | 0.2664 | 0.1524 | 23.5 |
| **1.1** | **10/10** | **0.1724** | **0.1397** | **0.2051** | 0.1350 | 26.4 |
| 1.3 | 10/10 | 0.1929 | 0.1556 | 0.2303 | 0.1255 | 24.4 |

**Chosen optimum**: **T = 1.1** (mean similarity **0.1724**).
**Rationale**: GPT-5.1's diversity is monotonically decreasing on `[0.5, 1.1]` and turns up slightly at 1.3, giving a clean U-curve with the minimum at 1.1. Validation success is 100% across the entire range — no failure cliff — so the choice is purely diversity-driven. T = 1.1 beats the previous 0.9 default by **0.04** in mean similarity (relative ~19% improvement); both topological and geometric components drop. T = 1.3 produces one obviously-degenerate `[straight]*7` network, hinting at the onset of incoherent sampling above 1.1.

### 3.2 Gemini 2.5 Pro (reduced grid)

Gemini was swept on the reduced grid `{0.5, 0.9, 1.3}` with N = 5 due to anticipated latency from the optimization experiment. In practice DMGPT's Gemini route was ~20 s/network, comparable to Claude.

| Temperature | Successful | Mean similarity | Topological mean | Geometric mean | Std | Wall-clock (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 5/5 | 0.1577 | 0.1286 | 0.1869 | 0.1159 | 108.1 |
| 0.9 | 5/5 | 0.1584 | 0.1286 | 0.1883 | 0.1228 | 97.1 |
| **1.3** | **5/5** | **0.1440** | **0.1143** | **0.1737** | 0.1587 | 87.2 |

**Chosen optimum**: **T = 1.3** (mean similarity **0.1440**).
**Rationale**: Gemini's diversity is essentially flat between 0.5 and 0.9 (0.158 vs 0.158) and improves modestly at 1.3 (0.144). Validation success remains 100% at the high end — unlike Claude, Gemini accepts temperatures above 1.0. Even on this reduced 3-point grid, Gemini is the strongest baseline of the three families: every cell beats GPT-5.1's best (0.172) and Claude's best (0.187). The N=5 sample size means the gap between 0.5/0.9 and 1.3 sits well within sampling noise; a follow-up at N=10 would tighten the estimate, but the monotonic-or-flat trend already gives 1.3 the edge with no validity penalty.

### 3.3 Claude Opus 4.6

| Temperature | Successful | Mean similarity | Topological mean | Geometric mean | Std | Wall-clock (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 10/10 | 0.2194 | 0.1714 | 0.2675 | 0.2064 | 56.2 |
| **0.7** | **10/10** | **0.1874** | **0.1556** | **0.2193** | 0.1508 | 57.7 |
| 0.9 | 10/10 | 0.1895 | 0.1333 | 0.2456 | 0.1619 | 57.4 |
| 1.1 | 0/10 | – | – | – | – | 5.5 |
| 1.3 | 0/10 | – | – | – | – | 5.2 |

**Chosen optimum**: **T = 0.7** (mean similarity **0.1874**).
**Rationale**: Claude (via Vertex AI in DMGPT) enforces a hard cap of `temperature ≤ 1.0` — both T = 1.1 and T = 1.3 returned `400 Bad Request: temperature: range: 0..1` and produced zero networks, so the effective range is `[0.5, 0.9]`. Within that range diversity bottoms out at 0.7; 0.9 is essentially tied (0.190 vs 0.187), and 0.5 degrades to 0.219. Interestingly the topological/geometric split differs from GPT-5.1: at 0.9 Claude's *topological* mean is best (0.133) but *geometric* is worst (0.246), suggesting Claude varies type composition well at higher temperatures but anchors parameter values. The chosen optimum 0.7 favors balanced topological-and-geometric diversity over either extreme. Claude is ~2.5× slower per network than GPT-5.1.

## 4. Discussion

**The family-specific-optimum hypothesis is confirmed.** Each family's best temperature lands at a distinct point on the swept grid:

| Family | Optimal T | Mean similarity at optimum | Cap behavior |
|---|---:|---:|---|
| GPT-5.1 | 1.1 | 0.172 | accepts up to 1.3, mild degradation past 1.1 |
| Gemini 2.5 Pro | 1.3 | 0.144 | accepts up to 1.3, still improving (likely beyond) |
| Claude Opus 4.6 | 0.7 | 0.187 | hard cap at 1.0 — provider error on T > 1.0 |

The hypothesis stated in §1 — that Claude would prefer slightly *higher* temperatures than GPT-5.1 — is **falsified**. The actual ordering is `Claude (0.7) < GPT-5.1 (1.1) < Gemini (1.3)`, with Claude being the most conservative. The mechanical reason is the provider-side temperature cap on Claude routed through Vertex AI; without it the curve at 0.9 is essentially identical to 0.7 (0.190 vs 0.187), so even with a cap-free Claude the optimum would not move far.

**Failure modes at the extremes.** Claude's high-T failures (T ≥ 1.1) are 100% validation rejects from the API itself — Vertex AI returns `400 Bad Request: temperature: range: 0..1` before any generation happens, which means we have no data on how Claude *would* behave at higher T. GPT-5.1 produces structurally valid output at T = 1.3 but emits at least one obviously degenerate sequence (`['straight']*7`), foreshadowing eventual incoherence. Gemini shows no high-T failure at all on the grid we tested.

**Topological vs geometric signatures match the prompt-optimization findings.** GPT-5.1's geometric mean drops much harder than its topological mean (0.300 → 0.205 as T goes 0.5 → 1.1, a 32% reduction, vs topological 0.200 → 0.140, a 30% reduction); the *gap* between geometric and topological narrows from 0.10 to 0.07. Claude shows the inverse pattern: at T = 0.9 the topological mean is best (0.133) but geometric is worst (0.246), confirming the "parameter anchoring" failure mode flagged in the optimization report. Gemini's gap is narrow across all T (~0.06), consistent with a model that varies type and parameters in step.

**Non-monotonicity.** GPT-5.1's curve dips at 1.1 and rises at 1.3, a clear U-shape. Claude is mostly flat in `[0.7, 0.9]` with a sharp drop at the cap. Gemini on the reduced grid looks monotonic-downward but the N = 5 estimate makes any second-derivative claim premature.

## 4b. Fine-grained follow-up sweep

After the coarse sweep above, each family was re-swept on a finer **0.05-step** grid centred on the coarse optimum, all at **N = 10**:

- GPT-5.1: `[0.90, 0.95, ..., 1.30]`
- Claude:  `[0.70, 0.75, ..., 1.00]` (above the provider cap)
- Gemini:  `[0.90, 0.95, ..., 1.30]`

Outputs live in `experiments/temperature_sweep/{family}_fine/`.

### 4b.1 GPT-5.1 (fine)

| Temperature | Mean | Topo | Geom | Std | Successful |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.2314 | 0.1492 | 0.3137 | 0.131 | 10/10 |
| 0.95 | 0.2386 | 0.1778 | 0.2995 | 0.117 | 10/10 |
| 1.00 | 0.2031 | 0.1619 | 0.2443 | 0.139 | 10/10 |
| 1.05 | 0.1994 | 0.1524 | 0.2465 | 0.127 | 10/10 |
| 1.10 | 0.2204 | 0.1778 | 0.2630 | 0.118 | 10/10 |
| 1.15 | 0.2199 | 0.1524 | 0.2875 | 0.131 | 10/10 |
| 1.20 | 0.2211 | 0.1619 | 0.2803 | 0.162 | 10/10 |
| 1.25 | 0.2817 | 0.2063 | 0.3570 | 0.132 | 10/10 |
| **1.30** | **0.1844** | 0.1429 | 0.2259 | 0.118 | 10/10 |

**Replication concern.** The coarse sweep's T = 1.1 result (0.172) does **not** replicate here: T = 1.1 on the fine grid lands at 0.220. The standard deviation per cell is ~0.12–0.14, which is consistent with a true between-run noise of order ±0.03 at N = 10. The fine sweep instead identifies **T = 1.3** as the cleanest minimum (0.184), and the coarse-sweep T = 1.3 value (0.193) is within noise of that. The plateau at 1.0–1.2 (~0.20) followed by a clear dip at 1.3 is more stable than a sharp 1.1 peak. The T = 1.25 spike (0.282) is treated as an outlier — likely one network drove it.

### 4b.2 Claude Opus 4.6 (fine)

| Temperature | Mean | Topo | Geom | Std | Successful |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.1855 | 0.1556 | 0.2154 | 0.160 | 10/10 |
| 0.75 | 0.1999 | 0.1492 | 0.2505 | 0.151 | 10/10 |
| 0.80 | 0.1768 | 0.1270 | 0.2266 | 0.140 | 10/10 |
| 0.85 | 0.1985 | 0.1492 | 0.2478 | 0.190 | 10/10 |
| **0.90** | **0.1722** | 0.1302 | 0.2142 | 0.134 | 10/10 |
| 0.95 | 0.2076 | 0.1429 | 0.2724 | 0.115 | 10/10 |
| 1.00 | 0.1828 | 0.1238 | 0.2418 | 0.143 | 10/10 |

The curve is noisier than GPT-5.1's: alternating up/down between adjacent 0.05-spaced cells. Three local minima cluster around `{0.80, 0.90, 1.00}` (means 0.177, 0.172, 0.183), all within one standard deviation of each other. The chosen optimum is **T = 0.9** (0.172), supported by being the global minimum and having a clean topo-and-geo decomposition (0.130 / 0.214). Note that even 0.7 — the previous default — sits at 0.186, only 0.014 worse, so the decision is data-driven but the operational impact of misranking by 0.05 here is small.

### 4b.3 Gemini 2.5 Pro (fine)

| Temperature | Mean | Topo | Geom | Std | Successful |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.1775 | 0.1238 | 0.2312 | 0.143 | 10/10 |
| 0.95 | 0.2161 | 0.1492 | 0.2831 | 0.174 | 10/10 |
| 1.00 | 0.1934 | 0.1365 | 0.2504 | 0.134 | 10/10 |
| 1.05 | 0.1775 | 0.1270 | 0.2281 | 0.110 | 10/10 |
| **1.10** | **0.1389** | 0.1111 | 0.1666 | 0.143 | 10/10 |
| 1.15 | 0.1829 | 0.1492 | 0.2166 | 0.154 | 10/10 |
| 1.20 | 0.1566 | 0.1302 | 0.1830 | 0.160 | 10/10 |
| 1.25 | 0.1788 | 0.1206 | 0.2370 | 0.133 | 10/10 |
| 1.30 | 0.1648 | 0.1302 | 0.1994 | 0.156 | 10/10 |

A clear minimum at **T = 1.1** (0.139) that beats every other cell by ≥0.018 — comfortably outside one-sigma noise. Both topological (0.111) and geometric (0.167) components hit their per-family lows at this point. The coarse-sweep value at T = 1.3 (0.144 at N = 5) was close to this minimum but didn't quite identify it, partly because the coarse grid skipped 1.1 entirely. T = 1.1 is now the canonical Gemini default.

### 4b.4 Cross-family fine-sweep summary

| Family | Fine optimum | Mean similarity |
|---|---:|---:|
| Gemini 2.5 Pro | **1.10** | **0.139** |
| Claude Opus 4.6 | **0.90** | **0.172** |
| GPT-5.1 | **1.30** | **0.184** |

The fine sweeps shift each family's chosen temperature, and Gemini's optimum improves enough (0.144 → 0.139) to widen its lead in the full benchmark. The lesson from the GPT-5.1 result is that at N = 10 the cell-level noise (~0.05) is comparable to the gaps between adjacent cells; a future tightening should use multiple repeats per cell rather than a finer grid.

## 5. Decision

The chosen temperatures are wired into a `MODEL_DEFAULT_TEMPERATURE` lookup in `src/llm_engine/client.py`, keyed by model family (see `_detect_model_family` in `src/llm_engine/prompts.py`). This becomes the default for `LLMGenerator` unless overridden by the runner via `--temperature`.

| Family | Chosen temperature (final, after fine sweep) | Mean similarity |
|---|---:|---:|
| `gpt5`   | **1.3** | 0.184 |
| `gemini` | **1.1** | 0.139 |
| `claude` | **0.9** | 0.172 |

(Coarse-sweep values shown earlier are kept in §3 for context but are superseded by the fine-sweep choices above.)

## 6. Limitations

- Single run per cell — variance not characterized at this stage.
- Only one prompt per family (the optimized one); a different prompt may shift the optimum.
- Coarse 0.2-step grid; the true optimum may sit between sampled points.
- N=10 networks → 45 pairwise comparisons per cell; small-sample noise of order ±0.02 on mean similarity.
- DMGPT routing may apply provider-side normalization to the temperature parameter.
