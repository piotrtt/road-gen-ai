#!/usr/bin/env python3
"""
Analyze diversity of generated road networks.

Quick verification tool for Phase 0.
"""

import json
from pathlib import Path
from collections import Counter
import argparse


def analyze_network(network_file: Path) -> dict:
    """Analyze a single road network."""
    with open(network_file, 'r') as f:
        data = json.load(f)

    components = data.get('road_network', [])
    types = [c['type'] for c in components]

    return {
        'filename': network_file.name,
        'num_components': len(components),
        'types': types,
        'type_counts': Counter(types),
        'unique_types': len(set(types))
    }


def analyze_all_networks(graphs_dir: Path, prefix: str = None):
    """Analyze all networks matching prefix."""
    pattern = f"{prefix}*.json" if prefix else "*.json"
    network_files = sorted(graphs_dir.glob(pattern))

    if not network_files:
        print(f"❌ No network files found matching: {pattern}")
        return

    print(f"📊 Analyzing {len(network_files)} networks...\n")

    all_type_counts = Counter()
    all_analyses = []

    for network_file in network_files:
        analysis = analyze_network(network_file)
        all_analyses.append(analysis)
        all_type_counts.update(analysis['type_counts'])

    # Print summary
    print(f"{'Filename':<35} | Components | Unique Types | Type Sequence")
    print("-" * 110)

    for analysis in all_analyses:
        types_str = ', '.join(analysis['types'])
        print(f"{analysis['filename']:<35} | {analysis['num_components']:10d} | {analysis['unique_types']:12d} | {types_str}")

    print(f"\n{'=' * 110}")
    print(f"OVERALL STATISTICS (across {len(network_files)} networks)")
    print(f"{'=' * 110}")
    print(f"\nTotal component type usage:")
    for comp_type, count in sorted(all_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {comp_type:20s}: {count:3d}")

    print(f"\n  Total unique types used: {len(all_type_counts)}")
    print(f"  Total components generated: {sum(all_type_counts.values())}")
    print(f"  Average components per network: {sum(all_type_counts.values()) / len(network_files):.1f}")

    # Diversity metrics
    type_usage_variance = sum((count - sum(all_type_counts.values())/len(all_type_counts))**2
                              for count in all_type_counts.values()) / len(all_type_counts)
    print(f"  Type usage variance: {type_usage_variance:.2f} (lower = more balanced)")


def main():
    parser = argparse.ArgumentParser(description="Analyze diversity of generated road networks")
    parser.add_argument(
        "--approach",
        type=str,
        choices=["random", "least_generated", "all"],
        default="all",
        help="Which approach to analyze"
    )
    parser.add_argument(
        "--graphs-dir",
        type=str,
        default="outputs/graphs",
        help="Directory containing generated networks"
    )

    args = parser.parse_args()

    graphs_dir = Path(args.graphs_dir)

    if not graphs_dir.exists():
        print(f"❌ Directory not found: {graphs_dir}")
        return

    if args.approach == "all":
        print("\n🎲 RANDOM GENERATOR ANALYSIS")
        print("=" * 110)
        analyze_all_networks(graphs_dir, prefix="random_")

        print("\n\n📊 LEAST GENERATED GENERATOR ANALYSIS")
        print("=" * 110)
        analyze_all_networks(graphs_dir, prefix="least_generated_")

    elif args.approach == "random":
        analyze_all_networks(graphs_dir, prefix="random_")

    elif args.approach == "least_generated":
        analyze_all_networks(graphs_dir, prefix="least_generated_")


if __name__ == "__main__":
    main()
