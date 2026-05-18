#!/usr/bin/env python3
"""
Run comprehensive evaluation across all road network generation approaches.

Supports:
- Multiple LLM models in one run (one bucket per model: ``llm:gpt5``, ``llm:gemini``, ``llm:claude``).
- Repeated runs per (approach, model) for variance estimation (``--repeats``).
- Per-family default temperature, overridable via ``--temperature``.

Output layout (per invocation, ``run_id`` = timestamp):

    outputs/evaluation/{run_id}/
        config.json
        raw/{method}__rep{i}.json
        aggregated.json

Example:
    uv run python src/run_evaluation.py --quantity 50 --components 7 \
        --approaches random least_generated hybrid llm \
        --llm-models gpt-5.1 gemini-2.5-pro claude-opus-4.6 \
        --repeats 3
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.generators import RandomGenerator, LeastGeneratedGenerator
from src.generators.hybrid_generator import HybridGenerator
from src.generators.llm_generator import LLMGenerator
from src.llm_engine.prompts import _detect_model_family
from src.metrics.evaluation import EvaluationRunner


# Per-family default temperature, picked by the temperature sweep
# (experiments/temperature_sweep/<family>/summary.json). See the report at
# experiments/temperature_sweep/temperature_sweep_report.md for rationale.
MODEL_DEFAULT_TEMPERATURE: dict[str, Optional[float]] = {
    "gpt5": 1.3,        # fine-sweep optimum: mean similarity 0.184 at T=1.3
    "claude": 0.9,      # fine-sweep optimum: mean similarity 0.172 at T=0.9 (T>1.0 rejected by provider)
    "gemini": 1.1,      # fine-sweep optimum: mean similarity 0.139 at T=1.1
}


def _model_short_key(model_name: str) -> str:
    """Map a full model name to a short family-aware key (e.g. ``gpt-5.1`` -> ``gpt5``)."""
    family = _detect_model_family(model_name)
    return family or model_name.replace("/", "_")


def _make_generator(approach: str, model_name: Optional[str], temperature: Optional[float]):
    """Construct a fresh generator for a single repeat. Clears stored networks for fairness."""
    if approach == "random":
        return RandomGenerator()
    if approach == "least_generated":
        gen = LeastGeneratedGenerator()
        gen.reset_usage_counts()
        return gen
    if approach == "hybrid":
        gen = HybridGenerator(num_candidates=5, base_approach="random")
        gen.storage.clear(approach="hybrid")
        return gen
    if approach == "llm":
        gen = LLMGenerator(
            model_name=model_name,
            include_existing=True,
            temperature=temperature,
        )
        gen.storage.clear(approach="llm")
        return gen
    raise ValueError(f"Unknown approach: {approach}")


def _aggregate_method(repeats_reports: list[dict]) -> dict:
    """Aggregate per-repeat reports into mean/std for the headline metrics."""
    def collect(path):
        out = []
        for r in repeats_reports:
            cur = r
            for p in path:
                cur = cur.get(p) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None:
                out.append(cur)
        return out

    def m_std(values):
        if not values:
            return {"mean": None, "std": None, "n": 0}
        if len(values) == 1:
            return {"mean": values[0], "std": 0.0, "n": 1}
        return {"mean": statistics.mean(values), "std": statistics.pstdev(values), "n": len(values)}

    return {
        "repeats": len(repeats_reports),
        "time_to_quantity_seconds": m_std(collect(["summary", "time_to_quantity_seconds"])),
        "reject_rate": m_std(collect(["summary", "reject_rate"])),
        "diversity_combined_mean": m_std(collect(["diversity", "combined", "mean"])),
        "diversity_topological_mean": m_std(collect(["diversity", "topological", "mean"])),
        "diversity_geometric_mean": m_std(collect(["diversity", "geometric", "mean"])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation of road network generators"
    )
    parser.add_argument("--quantity", type=int, default=10,
                        help="Number of networks to generate per (approach, repeat)")
    parser.add_argument("--components", type=int, default=7,
                        help="Number of components per network")
    parser.add_argument("--approaches", type=str, nargs="+",
                        default=["random", "least_generated", "hybrid", "llm"],
                        choices=["random", "least_generated", "hybrid", "llm"])
    parser.add_argument("--llm-models", type=str, nargs="+",
                        default=["gpt-5.1"],
                        help="LLM model names (one bucket per model when 'llm' is in --approaches)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override the per-family default temperature for all LLM models")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeats per (approach, model) for variance")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Parent directory for evaluation runs (default: outputs/evaluation/)")

    args = parser.parse_args()

    # Build a per-run output directory
    parent_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent.parent / "outputs" / "evaluation"
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = parent_dir / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of (method_key, approach, model_name) to evaluate
    method_specs: list[tuple[str, str, Optional[str]]] = []
    for approach in args.approaches:
        if approach == "llm":
            for m in args.llm_models:
                method_specs.append((f"llm:{_model_short_key(m)}", "llm", m))
        else:
            method_specs.append((approach, approach, None))

    # Persist run config
    config = {
        "run_id": run_id,
        "quantity": args.quantity,
        "components": args.components,
        "approaches": args.approaches,
        "llm_models": args.llm_models,
        "repeats": args.repeats,
        "temperature_override": args.temperature,
        "model_default_temperature": MODEL_DEFAULT_TEMPERATURE,
        "methods": [{"key": k, "approach": a, "model": m} for k, a, m in method_specs],
        "started_at": datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print("=" * 72)
    print(f"ROAD NETWORK GENERATOR EVALUATION  (run_id={run_id})")
    print("=" * 72)
    print(f"Quantity={args.quantity}, components={args.components}, repeats={args.repeats}")
    print(f"Methods: {[k for k, _, _ in method_specs]}")
    print(f"Output:  {run_dir}")

    # Per-method aggregated results (built up as repeats finish)
    aggregated: dict[str, dict] = {"run_id": run_id, "config": config, "methods": {}}

    for method_key, approach, model_name in method_specs:
        print(f"\n--- {method_key} ---")
        # Resolve temperature: explicit override > family default > None
        if approach == "llm":
            family = _detect_model_family(model_name) or ""
            temp = (args.temperature
                    if args.temperature is not None
                    else MODEL_DEFAULT_TEMPERATURE.get(family))
        else:
            temp = None

        per_repeat_reports: list[dict] = []
        for rep in range(args.repeats):
            print(f"  repeat {rep + 1}/{args.repeats}")
            # Fresh runner state per repeat so timings/results don't bleed
            runner = EvaluationRunner(
                output_dir=raw_dir,
                target_quantity=args.quantity,
                num_components=args.components,
            )
            try:
                generator = _make_generator(approach, model_name, temp)
                metrics = runner.evaluate_generator(generator, method_key)
                report = metrics.generate_report()
                report["method_key"] = method_key
                report["approach"] = approach
                report["model"] = model_name
                report["temperature"] = temp
                report["repeat"] = rep
                # Save raw per-repeat
                fname = f"{method_key.replace(':', '_')}__rep{rep}.json"
                (raw_dir / fname).write_text(json.dumps(report, indent=2, default=str))
                per_repeat_reports.append(report)
            except Exception as e:
                print(f"    FAILED: {e}")
                per_repeat_reports.append({
                    "error": str(e),
                    "method_key": method_key,
                    "approach": approach,
                    "model": model_name,
                    "temperature": temp,
                    "repeat": rep,
                })

        # Aggregate this method
        successful_reports = [r for r in per_repeat_reports if "error" not in r]
        aggregated["methods"][method_key] = {
            "approach": approach,
            "model": model_name,
            "temperature": temp,
            "raw_files": [f"raw/{method_key.replace(':', '_')}__rep{i}.json"
                          for i in range(args.repeats)],
            "stats": _aggregate_method(successful_reports) if successful_reports else None,
            "errors": [r for r in per_repeat_reports if "error" in r],
        }

    # Persist aggregated.json
    aggregated["finished_at"] = datetime.now().isoformat()
    (run_dir / "aggregated.json").write_text(json.dumps(aggregated, indent=2, default=str))

    # Pretty-print final comparison table
    print("\n" + "=" * 88)
    print("FINAL COMPARISON  (mean ± std across repeats; lower diversity = more diverse)")
    print("=" * 88)
    print(f"{'method':<22} {'time(s)':>16} {'reject%':>14} {'div_combined':>16} {'div_topo':>14} {'div_geo':>12}")
    print("-" * 88)
    for k, m in aggregated["methods"].items():
        s = m.get("stats")
        if not s:
            print(f"{k:<22}  (no successful repeats)")
            continue
        def fmt(stat, scale=1.0, prec=3):
            if stat["mean"] is None:
                return "n/a"
            return f"{stat['mean']*scale:.{prec}f}±{stat['std']*scale:.{prec}f}"
        print(
            f"{k:<22} "
            f"{fmt(s['time_to_quantity_seconds'], prec=2):>16} "
            f"{fmt(s['reject_rate'], scale=100, prec=1):>14} "
            f"{fmt(s['diversity_combined_mean']):>16} "
            f"{fmt(s['diversity_topological_mean']):>14} "
            f"{fmt(s['diversity_geometric_mean']):>12}"
        )

    print(f"\nWrote: {run_dir / 'aggregated.json'}")
    print(f"Wrote: {run_dir / 'config.json'}")


if __name__ == "__main__":
    main()
