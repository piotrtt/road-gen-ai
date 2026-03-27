#!/usr/bin/env python3
"""
Evaluate multiple LLM models for road network generation.

Tests models from OpenAI, Google, and Anthropic, comparing:
- Generation time
- Reject rate
- Diversity score

Results saved to CSV and top 3 displayed in terminal.

Usage:
    python src/evaluate_llms.py --quantity 5 --components 7
    python src/evaluate_llms.py --quantity 10 --components 7 --output results.csv
"""

import argparse
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

from src.generators.llm_generator import LLMGenerator
from src.generators.network_storage import NetworkStorage
from src.metrics.similarity import calculate_pairwise_similarities


# Models to evaluate (using litellm naming convention)
# See: https://docs.litellm.ai/docs/providers
LLM_MODELS = [
    # OpenAI models (no prefix needed)
    {"provider": "OpenAI", "model": "gpt-5-mini", "name": "GPT-5 Mini"},
    {"provider": "OpenAI", "model": "gpt-5", "name": "GPT-5"},
    {"provider": "OpenAI", "model": "gpt-5.4", "name": "GPT-5.4"},
    # Google models (use google/ prefix for Vertex AI or gemini/ for AI Studio)
    {"provider": "Google", "model": "gemini/gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    {"provider": "Google", "model": "gemini/gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
    {"provider": "Google", "model": "gemini/gemini-3-flash-preview", "name": "Gemini 3.0 Flash"},
    {"provider": "Google", "model": "gemini/gemini-3-pro-preview", "name": "Gemini 3.0 Pro"},
    {"provider": "Google", "model": "gemini/gemini/gemini-3.1-flash-lite-preview", "name": "Gemini 3.1 Flash Lite"},
    {"provider": "Google", "model": "gemini/gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro"},
    # Anthropic models (use anthropic/ prefix)
    {"provider": "Anthropic", "model": "anthropic/claude-haiku-4-5-20251001", "name": "Claude 4.5 Haiku"},
    {"provider": "Anthropic", "model": "anthropic/claude-sonnet-4-6", "name": "Claude 4.6 Sonnet"},
    {"provider": "Anthropic", "model": "anthropic/claude-opus-4-6", "name": "Claude 4.6 Opus"},
]


@dataclass
class ModelResult:
    """Results from evaluating a single model."""
    provider: str
    model_id: str
    model_name: str
    num_networks: int
    num_components: int
    total_time_seconds: float
    avg_time_per_network: float
    successful: int
    failed: int
    reject_rate: float
    diversity_mean: float
    diversity_std: float
    error: str = ""


def evaluate_model(
    model_config: Dict[str, str],
    num_networks: int,
    num_components: int,
    storage: NetworkStorage
) -> ModelResult:
    """
    Evaluate a single LLM model.

    Args:
        model_config: Model configuration with provider, model, name
        num_networks: Number of networks to generate
        num_components: Components per network
        storage: Network storage instance

    Returns:
        ModelResult with evaluation metrics
    """
    provider = model_config["provider"]
    model_id = model_config["model"]
    model_name = model_config["name"]

    print(f"\n  Testing {model_name} ({model_id})...")

    # Clear previous networks for this model
    storage.clear(approach=f"llm_{model_id.replace('/', '_')}")

    generated_networks = []
    successful = 0
    failed = 0
    error_msg = ""

    start_time = time.time()

    try:
        generator = LLMGenerator(
            model_name=model_id,
            include_existing=True,  
        )

        for i in range(num_networks):
            try:
                network = generator.generate(num_components)
                generated_networks.append(network)
                successful += 1
                print(f"    [{i+1}/{num_networks}] ✓", end="", flush=True)
            except Exception as e:
                failed += 1
                error_msg = str(e)[:50]
                print(f"    [{i+1}/{num_networks}] ✗", end="", flush=True)

        print()  # New line after progress

    except Exception as e:
        error_msg = str(e)[:100]
        print(f"    ❌ Failed to initialize: {error_msg}")

    total_time = time.time() - start_time

    # Calculate diversity if we have enough networks
    diversity_mean = 0.0
    diversity_std = 0.0

    if len(generated_networks) >= 2:
        _, stats = calculate_pairwise_similarities(generated_networks)
        diversity_mean = stats.get("mean", 0.0)
        diversity_std = stats.get("std", 0.0)

    return ModelResult(
        provider=provider,
        model_id=model_id,
        model_name=model_name,
        num_networks=num_networks,
        num_components=num_components,
        total_time_seconds=round(total_time, 2),
        avg_time_per_network=round(total_time / num_networks, 2) if num_networks > 0 else 0,
        successful=successful,
        failed=failed,
        reject_rate=round(failed / num_networks, 3) if num_networks > 0 else 0,
        diversity_mean=round(diversity_mean, 4),
        diversity_std=round(diversity_std, 4),
        error=error_msg
    )


def save_results_to_csv(results: List[ModelResult], output_path: Path):
    """Save results to CSV file."""
    fieldnames = [
        "provider", "model_id", "model_name", "num_networks", "num_components",
        "total_time_seconds", "avg_time_per_network", "successful", "failed",
        "reject_rate", "diversity_mean", "diversity_std", "error"
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "provider": result.provider,
                "model_id": result.model_id,
                "model_name": result.model_name,
                "num_networks": result.num_networks,
                "num_components": result.num_components,
                "total_time_seconds": result.total_time_seconds,
                "avg_time_per_network": result.avg_time_per_network,
                "successful": result.successful,
                "failed": result.failed,
                "reject_rate": result.reject_rate,
                "diversity_mean": result.diversity_mean,
                "diversity_std": result.diversity_std,
                "error": result.error
            })


def print_top_results(results: List[ModelResult]):
    """Print top 3 models by different metrics."""
    # Filter out complete failures
    valid_results = [r for r in results if r.successful > 0]

    if not valid_results:
        print("\n❌ No models completed successfully")
        return

    print("\n" + "=" * 80)
    print("TOP 3 MODELS BY CATEGORY")
    print("=" * 80)

    # Top 3 by Speed (lowest avg time)
    print("\n🚀 FASTEST (by avg time per network):")
    by_speed = sorted(valid_results, key=lambda x: x.avg_time_per_network)[:3]
    for i, r in enumerate(by_speed, 1):
        print(f"   {i}. {r.model_name:25s} - {r.avg_time_per_network:.2f}s/network")

    # Top 3 by Diversity (lowest similarity = more diverse)
    print("\n🎯 MOST DIVERSE (lowest similarity score):")
    by_diversity = sorted(valid_results, key=lambda x: x.diversity_mean)[:3]
    for i, r in enumerate(by_diversity, 1):
        print(f"   {i}. {r.model_name:25s} - {r.diversity_mean:.4f} similarity")

    # Top 3 by Reliability (lowest reject rate)
    print("\n✅ MOST RELIABLE (lowest reject rate):")
    by_reliability = sorted(valid_results, key=lambda x: x.reject_rate)[:3]
    for i, r in enumerate(by_reliability, 1):
        print(f"   {i}. {r.model_name:25s} - {r.reject_rate*100:.1f}% reject rate")

    # Overall best (composite score: speed + diversity + reliability)
    print("\n🏆 OVERALL BEST (balanced score):")
    # Normalize and combine: lower is better for all metrics
    if len(valid_results) >= 2:
        max_time = max(r.avg_time_per_network for r in valid_results)
        max_div = max(r.diversity_mean for r in valid_results) or 1

        def composite_score(r):
            time_norm = r.avg_time_per_network / max_time if max_time > 0 else 0
            div_norm = r.diversity_mean / max_div if max_div > 0 else 0
            reject_norm = r.reject_rate
            return (time_norm * 0.3) + (div_norm * 0.4) + (reject_norm * 0.3)

        by_overall = sorted(valid_results, key=composite_score)[:3]
        for i, r in enumerate(by_overall, 1):
            score = composite_score(r)
            print(f"   {i}. {r.model_name:25s} - score: {score:.4f}")


def print_full_results_table(results: List[ModelResult]):
    """Print full results table."""
    print("\n" + "=" * 100)
    print("FULL RESULTS")
    print("=" * 100)

    header = f"{'Model':<30} | {'Provider':<10} | {'Time/Net':<10} | {'Success':<8} | {'Reject%':<8} | {'Diversity':<10}"
    print(header)
    print("-" * 100)

    for r in sorted(results, key=lambda x: x.avg_time_per_network):
        status = "✓" if r.successful > 0 else "✗"
        print(f"{r.model_name:<30} | {r.provider:<10} | {r.avg_time_per_network:>7.2f}s | "
              f"{r.successful:>3}/{r.num_networks:<3} | {r.reject_rate*100:>6.1f}% | {r.diversity_mean:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple LLM models for road network generation"
    )

    parser.add_argument(
        "--quantity",
        type=int,
        default=5,
        help="Number of networks to generate per model (default: 5)"
    )

    parser.add_argument(
        "--components",
        type=int,
        default=7,
        help="Number of components per network (default: 7)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: outputs/evaluation/llm_comparison_<timestamp>.csv)"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific model IDs to test (default: all models)"
    )

    args = parser.parse_args()

    # Set up output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("outputs/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"llm_comparison_{timestamp}.csv"

    # Filter models if specified
    models_to_test = LLM_MODELS
    if args.models:
        models_to_test = [m for m in LLM_MODELS if m["model"] in args.models]
        if not models_to_test:
            print(f"❌ No matching models found for: {args.models}")
            print(f"Available models: {[m['model'] for m in LLM_MODELS]}")
            return

    print("=" * 80)
    print("LLM MODEL EVALUATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Networks per model: {args.quantity}")
    print(f"  Components per network: {args.components}")
    print(f"  Models to test: {len(models_to_test)}")
    print(f"  Output: {output_path}")

    # Storage for generated networks
    storage = NetworkStorage()

    # Evaluate each model
    results = []
    for model_config in models_to_test:
        result = evaluate_model(
            model_config,
            num_networks=args.quantity,
            num_components=args.components,
            storage=storage
        )
        results.append(result)

    # Save to CSV
    save_results_to_csv(results, output_path)
    print(f"\n📄 Results saved to: {output_path}")

    # Print full table
    print_full_results_table(results)

    # Print top 3
    print_top_results(results)


if __name__ == "__main__":
    main()
