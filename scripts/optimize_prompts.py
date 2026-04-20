#!/usr/bin/env python3
"""
Prompt Optimization Tool for Road Network Diversity

Generates road networks using a specified LLM model via the DMGPT API,
measures diversity using the project's custom similarity metric, and
outputs results as JSON for iterative prompt optimization.

Usage:
    python scripts/optimize_prompts.py \
        --model gpt-5.1 \
        --model-key gpt5 \
        --prompt-file prompts/system_prompt_gpt5.txt \
        --iteration 1 \
        --output-dir experiments/prompt_optimization/history/gpt5
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_engine.prompts import get_function_schema, load_user_prompt
from src.metrics.similarity import calculate_pairwise_similarities

# Valid component types
VALID_TYPES = {
    "straight", "curve", "lane_switch", "fork",
    "t_intersection", "intersection", "roundabout", "u_shape"
}

REQUIRED_FIELDS = {"id", "sequence_index", "type", "lane_width", "right_lanes", "left_lanes"}


def validate_network(network: dict, expected_components: int) -> tuple[bool, str]:
    """Validate a generated road network structure."""
    if not isinstance(network, dict):
        return False, "not a dict"

    road_network = network.get("road_network")
    if not road_network or not isinstance(road_network, list):
        return False, "missing or empty road_network"

    if len(road_network) == 0:
        return False, "empty component list"

    # Allow +-2 tolerance on component count
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
    """Fix sequence indices and IDs to ensure correct ordering."""
    components = network.get("road_network", [])
    for i, comp in enumerate(components):
        comp["sequence_index"] = i
        comp["id"] = f"{comp['type']}_{i}"
    return network


def generate_single_network(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
) -> tuple[dict | None, str]:
    """Generate a single road network via function calling."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "generate_road_network"}},
            temperature=0.9,
        )

        message = response.choices[0].message
        if not message.tool_calls:
            return None, "no tool calls in response"

        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        return args, "ok"

    except Exception as e:
        return None, f"api_error: {str(e)}"


def run_iteration(
    model: str,
    model_key: str,
    prompt_file: str,
    num_networks: int,
    num_components: int,
    iteration: int,
    output_dir: str,
) -> dict:
    """Run a single optimization iteration: generate networks and measure diversity."""

    # Load system prompt
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {prompt_file}", file=sys.stderr)
        sys.exit(1)

    system_prompt = prompt_path.read_text()
    prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:16]

    # Initialize client
    client = OpenAI(
        base_url="https://api.dmgpt.dm-drogeriemarkt.com/api/v1/openai",
        api_key=os.getenv("DMGPT_PAT"),
    )

    # Build function calling tools
    schema = get_function_schema()
    tools = [{"type": "function", "function": schema}]

    # Generate networks
    generated_networks = []
    failures = []
    start_time = time.time()

    for i in range(num_networks):
        # Build user prompt with within-batch diversity context
        existing = [{"road_network": n["road_network"]} for n in generated_networks]
        user_prompt = load_user_prompt(
            num_components=num_components,
            existing_networks=existing if existing else None,
            max_examples=5,
        )

        # Generate
        result, reason = generate_single_network(client, model, system_prompt, user_prompt, tools)

        if result is None:
            failures.append(reason)
            print(f"  Network {i+1}/{num_networks}: FAILED ({reason})", file=sys.stderr)
            continue

        # Validate
        valid, val_reason = validate_network(result, num_components)
        if not valid:
            failures.append(f"validation: {val_reason}")
            print(f"  Network {i+1}/{num_networks}: INVALID ({val_reason})", file=sys.stderr)
            continue

        # Fix indices and add
        result = fix_sequence_indices(result)
        generated_networks.append(result)
        types = [c["type"] for c in result["road_network"]]
        print(f"  Network {i+1}/{num_networks}: OK - {types}", file=sys.stderr)

    elapsed = time.time() - start_time

    # Compute diversity
    diversity_stats = {
        "mean_similarity": None,
        "std": None,
        "min": None,
        "max": None,
        "topological_mean": None,
        "geometric_mean": None,
        "target_met": False,
        "num_networks": len(generated_networks),
        "num_pairs": 0,
    }

    if len(generated_networks) >= 2:
        # Combined similarity
        sims, stats = calculate_pairwise_similarities(generated_networks)
        diversity_stats["mean_similarity"] = round(stats["mean"], 4)
        diversity_stats["std"] = round(stats["std"], 4)
        diversity_stats["min"] = round(stats["min"], 4)
        diversity_stats["max"] = round(stats["max"], 4)
        diversity_stats["num_pairs"] = stats["count"]
        diversity_stats["target_met"] = stats["mean"] < 0.3

        # Topological similarity separately
        topo_sims, topo_stats = calculate_pairwise_similarities(
            generated_networks, topo_weight=1.0, geom_weight=0.0
        )
        diversity_stats["topological_mean"] = round(topo_stats["mean"], 4)

        # Geometric similarity separately
        geom_sims, geom_stats = calculate_pairwise_similarities(
            generated_networks, topo_weight=0.0, geom_weight=1.0
        )
        diversity_stats["geometric_mean"] = round(geom_stats["mean"], 4)

    # Type distribution across all networks
    all_types = []
    for net in generated_networks:
        for comp in net["road_network"]:
            all_types.append(comp["type"])
    type_distribution = dict(Counter(all_types))

    # Build result
    result = {
        "iteration": iteration,
        "model": model,
        "model_key": model_key,
        "prompt_hash": prompt_hash,
        "elapsed_seconds": round(elapsed, 2),
        "generation": {
            "total_attempts": num_networks,
            "successful": len(generated_networks),
            "failed": len(failures),
            "failure_reasons": failures,
        },
        "diversity": diversity_stats,
        "type_distribution": type_distribution,
        "networks": generated_networks,
    }

    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"iteration_{iteration:03d}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {result_file}", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(description="Prompt Optimization Tool for Road Network Diversity")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-5.1)")
    parser.add_argument("--model-key", required=True, help="Short model key (e.g., gpt5)")
    parser.add_argument("--prompt-file", required=True, help="Path to system prompt file")
    parser.add_argument("--num-networks", type=int, default=10, help="Networks per iteration")
    parser.add_argument("--num-components", type=int, default=7, help="Components per network")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    args = parser.parse_args()

    if not os.getenv("DMGPT_PAT"):
        print("Error: DMGPT_PAT environment variable not set", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Iteration {args.iteration} | Model: {args.model} ({args.model_key})", file=sys.stderr)
    print(f"Prompt: {args.prompt_file}", file=sys.stderr)
    print(f"Generating {args.num_networks} networks with {args.num_components} components each", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    result = run_iteration(
        model=args.model,
        model_key=args.model_key,
        prompt_file=args.prompt_file,
        num_networks=args.num_networks,
        num_components=args.num_components,
        iteration=args.iteration,
        output_dir=args.output_dir,
    )

    # Print summary to stderr
    div = result["diversity"]
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"RESULTS - Iteration {args.iteration}", file=sys.stderr)
    print(f"  Networks generated: {result['generation']['successful']}/{result['generation']['total_attempts']}", file=sys.stderr)
    print(f"  Mean similarity:    {div['mean_similarity']}", file=sys.stderr)
    print(f"  Topological mean:   {div['topological_mean']}", file=sys.stderr)
    print(f"  Geometric mean:     {div['geometric_mean']}", file=sys.stderr)
    print(f"  Target met (<0.3):  {div['target_met']}", file=sys.stderr)
    print(f"  Type distribution:  {result['type_distribution']}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # Print JSON to stdout for machine parsing
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
