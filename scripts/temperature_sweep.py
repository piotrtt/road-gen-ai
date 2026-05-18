#!/usr/bin/env python3
"""
Temperature Sweep for Road Network Diversity

For a given LLM family, generate N networks at each of several temperatures
using the model's optimized system prompt, measure diversity, and write a
per-temperature JSON plus a `summary.json` mapping temperature -> mean similarity.

Lightweight: small N (default 10), single repeat per cell, fast turnaround.

Usage:
    uv run python scripts/temperature_sweep.py \
        --model gpt-5.1 --model-key gpt5 \
        --prompt-file prompts/system_prompt_gpt5.txt \
        --temperatures 0.5 0.7 0.9 1.1 1.3 \
        --num-networks 10 --num-components 7 \
        --output-dir experiments/temperature_sweep/gpt5
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_engine.prompts import get_function_schema, load_user_prompt
from src.metrics.similarity import calculate_pairwise_similarities

VALID_TYPES = {
    "straight", "curve", "lane_switch", "fork",
    "t_intersection", "intersection", "roundabout", "u_shape"
}
REQUIRED_FIELDS = {"id", "sequence_index", "type", "lane_width", "right_lanes", "left_lanes"}


def validate_network(network: dict, expected_components: int) -> tuple[bool, str]:
    if not isinstance(network, dict):
        return False, "not a dict"
    road_network = network.get("road_network")
    if not road_network or not isinstance(road_network, list):
        return False, "missing or empty road_network"
    if abs(len(road_network) - expected_components) > 2:
        return False, f"component count {len(road_network)} too far from {expected_components}"
    for i, comp in enumerate(road_network):
        if not isinstance(comp, dict):
            return False, f"component {i} is not a dict"
        missing = REQUIRED_FIELDS - set(comp.keys())
        if missing:
            return False, f"component {i} missing fields: {missing}"
        if comp["type"] not in VALID_TYPES:
            return False, f"component {i} invalid type: {comp['type']}"
    return True, "ok"


def fix_sequence_indices(network: dict) -> dict:
    for i, comp in enumerate(network.get("road_network", [])):
        comp["sequence_index"] = i
        comp["id"] = f"{comp['type']}_{i}"
    return network


def generate_single_network(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
    temperature: float,
) -> tuple[dict | None, str]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "generate_road_network"}},
            temperature=temperature,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return None, "no tool calls in response"
        args = json.loads(message.tool_calls[0].function.arguments)
        return args, "ok"
    except Exception as e:
        return None, f"api_error: {str(e)}"


def run_one_temperature(
    client: OpenAI,
    model: str,
    model_key: str,
    system_prompt: str,
    prompt_hash: str,
    temperature: float,
    num_networks: int,
    num_components: int,
    output_dir: Path,
) -> dict:
    schema = get_function_schema()
    tools = [{"type": "function", "function": schema}]

    generated_networks = []
    failures = []
    start_time = time.time()

    for i in range(num_networks):
        existing = [{"road_network": n["road_network"]} for n in generated_networks]
        user_prompt = load_user_prompt(
            num_components=num_components,
            existing_networks=existing if existing else None,
            max_examples=5,
        )
        result, reason = generate_single_network(
            client, model, system_prompt, user_prompt, tools, temperature
        )
        if result is None:
            failures.append(reason)
            print(f"  T={temperature} net {i+1}/{num_networks}: FAILED ({reason})", file=sys.stderr)
            continue
        valid, val_reason = validate_network(result, num_components)
        if not valid:
            failures.append(f"validation: {val_reason}")
            print(f"  T={temperature} net {i+1}/{num_networks}: INVALID ({val_reason})", file=sys.stderr)
            continue
        result = fix_sequence_indices(result)
        generated_networks.append(result)
        types = [c["type"] for c in result["road_network"]]
        print(f"  T={temperature} net {i+1}/{num_networks}: OK - {types}", file=sys.stderr)

    elapsed = time.time() - start_time

    diversity = {
        "mean_similarity": None, "std": None, "min": None, "max": None,
        "topological_mean": None, "geometric_mean": None,
        "num_networks": len(generated_networks), "num_pairs": 0,
    }
    if len(generated_networks) >= 2:
        _, stats = calculate_pairwise_similarities(generated_networks)
        diversity.update({
            "mean_similarity": round(stats["mean"], 4),
            "std": round(stats["std"], 4),
            "min": round(stats["min"], 4),
            "max": round(stats["max"], 4),
            "num_pairs": stats["count"],
        })
        _, topo = calculate_pairwise_similarities(generated_networks, topo_weight=1.0, geom_weight=0.0)
        diversity["topological_mean"] = round(topo["mean"], 4)
        _, geom = calculate_pairwise_similarities(generated_networks, topo_weight=0.0, geom_weight=1.0)
        diversity["geometric_mean"] = round(geom["mean"], 4)

    type_distribution = dict(Counter(
        c["type"] for net in generated_networks for c in net["road_network"]
    ))

    out = {
        "model": model,
        "model_key": model_key,
        "temperature": temperature,
        "prompt_hash": prompt_hash,
        "elapsed_seconds": round(elapsed, 2),
        "generation": {
            "total_attempts": num_networks,
            "successful": len(generated_networks),
            "failed": len(failures),
            "failure_reasons": failures,
        },
        "diversity": diversity,
        "type_distribution": type_distribution,
        "networks": generated_networks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    # Use 2 decimals to keep filenames sortable: temp_0.50.json, temp_1.30.json
    fname = f"temp_{temperature:.2f}.json"
    (output_dir / fname).write_text(json.dumps(out, indent=2))
    print(f"  -> wrote {output_dir / fname}", file=sys.stderr)

    return out


def main():
    parser = argparse.ArgumentParser(description="Temperature sweep for road network diversity")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--num-networks", type=int, default=10)
    parser.add_argument("--num-components", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not os.getenv("DMGPT_PAT"):
        print("Error: DMGPT_PAT environment variable not set", file=sys.stderr)
        sys.exit(1)

    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {args.prompt_file}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text()
    prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:16]

    client = OpenAI(
        base_url="https://api.dmgpt.dm-drogeriemarkt.com/api/v1/openai",
        api_key=os.getenv("DMGPT_PAT"),
    )

    output_dir = Path(args.output_dir)
    summary_rows = []

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Temperature sweep | Model: {args.model} ({args.model_key})", file=sys.stderr)
    print(f"Temperatures: {args.temperatures}", file=sys.stderr)
    print(f"N={args.num_networks}, components={args.num_components}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    for t in args.temperatures:
        print(f"\n--- temperature = {t} ---", file=sys.stderr)
        out = run_one_temperature(
            client=client,
            model=args.model,
            model_key=args.model_key,
            system_prompt=system_prompt,
            prompt_hash=prompt_hash,
            temperature=t,
            num_networks=args.num_networks,
            num_components=args.num_components,
            output_dir=output_dir,
        )
        d = out["diversity"]
        summary_rows.append({
            "temperature": t,
            "successful": out["generation"]["successful"],
            "failed": out["generation"]["failed"],
            "mean_similarity": d["mean_similarity"],
            "topological_mean": d["topological_mean"],
            "geometric_mean": d["geometric_mean"],
            "std": d["std"],
            "elapsed_seconds": out["elapsed_seconds"],
        })

    # Build summary, picking the temperature with lowest mean_similarity (best diversity)
    valid_rows = [r for r in summary_rows if r["mean_similarity"] is not None]
    best = min(valid_rows, key=lambda r: r["mean_similarity"]) if valid_rows else None

    summary = {
        "model": args.model,
        "model_key": args.model_key,
        "prompt_file": str(args.prompt_file),
        "prompt_hash": prompt_hash,
        "num_networks": args.num_networks,
        "num_components": args.num_components,
        "temperatures": args.temperatures,
        "rows": summary_rows,
        "best_temperature": best["temperature"] if best else None,
        "best_mean_similarity": best["mean_similarity"] if best else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SWEEP SUMMARY - {args.model_key}", file=sys.stderr)
    print(f"{'temp':>6} {'mean':>8} {'topo':>8} {'geom':>8} {'std':>8} {'ok':>4} {'fail':>4} {'sec':>7}", file=sys.stderr)
    for r in summary_rows:
        print(
            f"{r['temperature']:>6.2f} "
            f"{(r['mean_similarity'] if r['mean_similarity'] is not None else 0):>8.4f} "
            f"{(r['topological_mean'] if r['topological_mean'] is not None else 0):>8.4f} "
            f"{(r['geometric_mean'] if r['geometric_mean'] is not None else 0):>8.4f} "
            f"{(r['std'] if r['std'] is not None else 0):>8.4f} "
            f"{r['successful']:>4d} {r['failed']:>4d} {r['elapsed_seconds']:>7.1f}",
            file=sys.stderr,
        )
    if best:
        print(f"\nBest temperature: {best['temperature']} (mean_similarity={best['mean_similarity']})", file=sys.stderr)
    print(f"Summary saved to: {output_dir / 'summary.json'}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
