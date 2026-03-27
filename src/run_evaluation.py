#!/usr/bin/env python3
"""
Run comprehensive evaluation across all road network generation approaches.

Usage:
    python src/run_evaluation.py --quantity 10 --components 7
    python src/run_evaluation.py --quantity 50 --components 7 --approaches random hybrid
"""

import argparse
from pathlib import Path
from src.generators import RandomGenerator, LeastGeneratedGenerator
from src.generators.hybrid_generator import HybridGenerator
from src.generators.llm_generator import LLMGenerator
from src.metrics.evaluation import EvaluationRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation of road network generators"
    )

    parser.add_argument(
        "--quantity",
        type=int,
        default=10,
        help="Number of networks to generate per approach (default: 10)"
    )

    parser.add_argument(
        "--components",
        type=int,
        default=7,
        help="Number of components per network (default: 7)"
    )

    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        default=["random", "least_generated", "hybrid", "llm"],
        choices=["random", "least_generated", "hybrid", "llm"],
        help="Approaches to evaluate (default: all)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation reports"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )

    args = parser.parse_args()

    # Set up evaluation runner
    output_dir = Path(args.output_dir) if args.output_dir else None
    runner = EvaluationRunner(
        output_dir=output_dir,
        target_quantity=args.quantity,
        num_components=args.components
    )

    print("=" * 70)
    print("ROAD NETWORK GENERATOR EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Target quantity: {args.quantity} networks")
    print(f"  Components per network: {args.components}")
    print(f"  Approaches: {args.approaches}")

    # Create generators for each approach
    generators = {}

    if "random" in args.approaches:
        generators["random"] = RandomGenerator()

    if "least_generated" in args.approaches:
        gen = LeastGeneratedGenerator()
        gen.reset_usage_counts()  # Start fresh
        generators["least_generated"] = gen

    if "hybrid" in args.approaches:
        generators["hybrid"] = HybridGenerator(
            num_candidates=5,
            base_approach="random"
        )
        # Clear existing hybrid networks for fair comparison
        generators["hybrid"].storage.clear(approach="hybrid")

    if "llm" in args.approaches:
        generators["llm"] = LLMGenerator(
            model_name=args.model,
            include_existing=True
        )
        # Clear existing LLM networks for fair comparison
        generators["llm"].storage.clear(approach="llm")

    # Run evaluation for each approach
    for name, generator in generators.items():
        try:
            runner.evaluate_generator(generator, name)
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # Print comparison
    runner.print_comparison()

    # Save reports
    print("\n📄 Saving reports...")
    runner.save_all_reports()

    # Print individual detailed reports
    for name, metrics in runner.results.items():
        metrics.print_report()


if __name__ == "__main__":
    main()
