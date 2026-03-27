#!/usr/bin/env python3
"""
CLI tool for generating road networks using different approaches.

Usage:
    python src/generate_roads.py --approach random --num-maps 10 --components 7
    python src/generate_roads.py --approach least_generated --num-maps 10 --components 7
    python src/generate_roads.py --approach hybrid --num-maps 10 --components 7 --candidates 5
    python src/generate_roads.py --approach llm --num-maps 5 --components 7 --model gpt-4o-mini
"""

import argparse
from pathlib import Path
from src.generators import RandomGenerator, LeastGeneratedGenerator
from src.generators.hybrid_generator import HybridGenerator
from src.generators.llm_generator import LLMGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate road networks for autonomous vehicle testing"
    )

    parser.add_argument(
        "--approach",
        type=str,
        choices=["random", "least_generated", "hybrid", "llm"],
        required=True,
        help="Generation approach to use"
    )

    parser.add_argument(
        "--num-maps",
        type=int,
        default=10,
        help="Number of road networks to generate (default: 10)"
    )

    parser.add_argument(
        "--components",
        type=int,
        default=7,
        help="Number of components per road network (default: 7)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated networks (default: outputs/graphs/)"
    )

    parser.add_argument(
        "--reset-usage",
        action="store_true",
        help="Reset usage counters before generation (for testing)"
    )

    # Hybrid-specific arguments
    parser.add_argument(
        "--candidates",
        type=int,
        default=5,
        help="Number of candidates to generate per selection (hybrid only, default: 5)"
    )

    parser.add_argument(
        "--base-approach",
        type=str,
        choices=["random", "least_generated"],
        default="random",
        help="Base generator for hybrid approach (default: random)"
    )

    parser.add_argument(
        "--topo-weight",
        type=float,
        default=0.5,
        help="Weight for topological similarity (default: 0.5)"
    )

    parser.add_argument(
        "--geom-weight",
        type=float,
        default=0.5,
        help="Weight for geometric similarity (default: 0.5)"
    )

    parser.add_argument(
        "--clear-hybrid",
        action="store_true",
        help="Clear existing hybrid networks before generation"
    )

    # LLM-specific arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (default: gpt-4o-mini or LLM_MODEL env var)"
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Max existing networks to include in LLM prompt (default: 5)"
    )

    parser.add_argument(
        "--clear-llm",
        action="store_true",
        help="Clear existing LLM-generated networks before generation"
    )

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Create generator based on approach
    if args.approach == "random":
        print("🎲 Using Random Generator")
        generator = RandomGenerator(output_dir=output_dir)

    elif args.approach == "least_generated":
        print("📊 Using Least Generated (Count-Based) Generator")
        generator = LeastGeneratedGenerator(output_dir=output_dir)

        if args.reset_usage:
            print("  Resetting usage counters...")
            generator.reset_usage_counts()

        print(f"  Current usage counts: {generator.get_usage_statistics()}")

    elif args.approach == "hybrid":
        print("🔀 Using Hybrid Generator (Similarity-Based Selection)")
        print(f"  Base approach: {args.base_approach}")
        print(f"  Candidates per selection: {args.candidates}")
        print(f"  Weights: topo={args.topo_weight}, geom={args.geom_weight}")

        generator = HybridGenerator(
            output_dir=output_dir,
            num_candidates=args.candidates,
            base_approach=args.base_approach,
            topo_weight=args.topo_weight,
            geom_weight=args.geom_weight,
        )

        if args.clear_hybrid:
            print("  Clearing existing hybrid networks...")
            generator.storage.clear(approach="hybrid")

        existing_count = generator.storage.count(approach="hybrid")
        print(f"  Existing hybrid networks: {existing_count}")

    elif args.approach == "llm":
        print("🤖 Using LLM Generator (Function Calling)")
        model = args.model or "gpt-4o-mini"
        print(f"  Model: {model}")
        print(f"  Max examples in prompt: {args.max_examples}")

        generator = LLMGenerator(
            output_dir=output_dir,
            model_name=model,
            max_examples=args.max_examples,
            include_existing=True,
        )

        if args.clear_llm:
            print("  Clearing existing LLM networks...")
            generator.storage.clear(approach="llm")

        existing_count = generator.storage.count(approach="llm")
        print(f"  Existing LLM networks: {existing_count}")

    else:
        raise ValueError(f"Unknown approach: {args.approach}")

    # Generate networks
    print(f"\n🚗 Generating {args.num_maps} road networks with {args.components} components each...")
    print(f"📁 Output directory: {generator.output_dir}\n")

    saved_files = generator.generate_multiple(
        num_networks=args.num_maps,
        num_components=args.components
    )

    print(f"\n✅ Successfully generated {len(saved_files)} road networks!")
    print(f"📂 Saved to: {generator.output_dir}")

    # Show final usage stats for least_generated approach
    if args.approach == "least_generated":
        print(f"\n📊 Final usage statistics:")
        usage_stats = generator.get_usage_statistics()
        for comp_type, count in sorted(usage_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {comp_type:20s}: {count:3d}")


if __name__ == "__main__":
    main()
