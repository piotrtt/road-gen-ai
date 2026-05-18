#!/usr/bin/env python3
"""
CLI tool for generating road networks using different approaches.

Usage:
    python src/generate_roads.py --approach random --num-maps 10 --components 7
    python src/generate_roads.py --approach least_generated --num-maps 10 --components 7
    python src/generate_roads.py --approach diversity_driven --num-maps 10 --components 7 --candidates 5
    python src/generate_roads.py --approach llm --num-maps 5 --components 7 --model gpt-4o-mini

    # Generate and immediately convert to .xodr:
    python src/generate_roads.py --approach random --num-maps 5 --components 7 --to-xodr
    python src/generate_roads.py --approach random --num-maps 5 --components 7 --to-xodr --xodr-dir outputs/xodr/

    # Generate, convert, and open the first map in esmini:
    python src/generate_roads.py --approach random --num-maps 5 --components 7 --to-xodr --visualize
    python src/generate_roads.py --approach hybrid --num-maps 5 --components 7 --model gpt-5.1
"""

import argparse
import subprocess
from pathlib import Path
from src.generators import RandomGenerator, LeastGeneratedGenerator
from src.generators.diversity_driven_generator import DiversityDrivenGenerator
from src.generators.hybrid_generator import HybridGenerator
from src.generators.llm_generator import LLMGenerator
from src.json_to_xodr import convert as json_to_xodr_convert

# Path to the bundled esmini odrviewer binary.
_ESMINI_BIN = Path(__file__).parent / "esmini" / "bin" / "odrviewer"


def main():
    parser = argparse.ArgumentParser(
        description="Generate road networks for autonomous vehicle testing"
    )

    parser.add_argument(
        "--approach",
        type=str,
        choices=["random", "least_generated", "diversity_driven", "llm", "hybrid"],
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

    # Diversity-driven-specific arguments
    parser.add_argument(
        "--candidates",
        type=int,
        default=5,
        help="Number of candidates to generate per selection (diversity_driven only, default: 5)"
    )

    parser.add_argument(
        "--base-approach",
        type=str,
        choices=["random", "least_generated"],
        default="random",
        help="Base generator for diversity-driven approach (default: random)"
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
        "--clear-storage",
        action="store_true",
        help="Clear existing networks for this approach before generation"
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

    # XODR conversion arguments
    parser.add_argument(
        "--to-xodr",
        action="store_true",
        help="Convert every generated JSON network to OpenDRIVE (.xodr) after generation"
    )

    parser.add_argument(
        "--xodr-dir",
        type=str,
        default=None,
        help="Output directory for .xodr files (default: outputs/xodr/)"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Open the first generated .xodr file in esmini odrviewer after conversion. "
            "Implies --to-xodr."
        )
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

    elif args.approach == "diversity_driven":
        print("🔀 Using DiversityDriven Generator (Similarity-Based Selection)")
        print(f"  Base approach: {args.base_approach}")
        print(f"  Candidates per selection: {args.candidates}")
        print(f"  Weights: topo={args.topo_weight}, geom={args.geom_weight}")

        generator = DiversityDrivenGenerator(
            output_dir=output_dir,
            num_candidates=args.candidates,
            base_approach=args.base_approach,
            topo_weight=args.topo_weight,
            geom_weight=args.geom_weight,
        )

        if args.clear_storage:
            print("  Clearing existing diversity_driven networks...")
            generator.storage.clear(approach="diversity_driven")

        existing_count = generator.storage.count(approach="diversity_driven")
        print(f"  Existing diversity_driven networks: {existing_count}")

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

    elif args.approach == "hybrid":
        print("🤝 Using HybridGenerator (LLM + diversity metric feedback loop)")
        model = args.model or "gpt-4o-mini"
        print(f"  Model: {model}")

        generator = HybridGenerator(
            output_dir=output_dir,
            model_name=model,
            include_existing=True,
        )

        if args.clear_storage:
            print("  Clearing existing hybrid networks...")
            generator.storage.clear(approach="hybrid")

        existing_count = generator.storage.count(approach="hybrid")
        print(f"  Existing hybrid networks: {existing_count}")

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

    # Optionally convert all generated JSON files to OpenDRIVE (.xodr)
    if args.to_xodr or args.visualize:
        xodr_dir = Path(args.xodr_dir) if args.xodr_dir else generator.output_dir.parent / "xodr"
        xodr_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n🗺️  Converting {len(saved_files)} networks to OpenDRIVE (.xodr)...")
        print(f"📁 XODR output directory: {xodr_dir}\n")

        converted, failed = 0, 0
        xodr_files: list[Path] = []
        for json_path in saved_files:
            xodr_path = xodr_dir / json_path.with_suffix(".xodr").name
            try:
                json_to_xodr_convert(json_path, xodr_path)
                print(f"  ✅ {json_path.name} → {xodr_path.name}")
                xodr_files.append(xodr_path)
                converted += 1
            except Exception as exc:
                print(f"  ❌ {json_path.name} failed: {exc}")
                failed += 1

        print(f"\n{'✅' if failed == 0 else '⚠️ '} Converted {converted}/{len(saved_files)} networks to .xodr")
        if failed:
            print(f"   {failed} conversion(s) failed — see errors above.")

        # Launch esmini odrviewer on the first successfully converted file.
        if args.visualize:
            if not _ESMINI_BIN.exists():
                print(f"\n⚠️  esmini odrviewer not found at {_ESMINI_BIN}")
            elif xodr_files:
                first_xodr = xodr_files[0]
                print(f"\n🖥️  Launching esmini odrviewer → {first_xodr.name}")
                subprocess.run([str(_ESMINI_BIN), "--odr", str(first_xodr)])
            else:
                print("\n⚠️  No .xodr files were produced; nothing to visualize.")


if __name__ == "__main__":
    main()
