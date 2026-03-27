#!/usr/bin/env python3
"""
Analyze similarity between generated road networks.

Uses the custom similarity metric to compare networks from
different generation approaches.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from src.metrics.similarity import (
    topological_similarity,
    geometric_similarity,
    combined_similarity,
    calculate_pairwise_similarities,
    extract_type_sequence,
)


def load_networks(graphs_dir: Path, prefix: str = None) -> List[Dict]:
    """Load all road networks matching prefix."""
    pattern = f"{prefix}*.json" if prefix else "*.json"
    networks = []

    for filepath in sorted(graphs_dir.glob(pattern)):
        with open(filepath, 'r') as f:
            networks.append(json.load(f))

    return networks


def analyze_approach(graphs_dir: Path, prefix: str):
    """Analyze similarity metrics for a single approach."""
    networks = load_networks(graphs_dir, prefix)

    if len(networks) < 2:
        print(f"  Not enough networks to compare (found {len(networks)})")
        return

    print(f"\n  Analyzing {len(networks)} networks...")

    # Calculate all pairwise similarities
    all_sims, stats = calculate_pairwise_similarities(networks)

    # Separate topological and geometric
    topo_sims = []
    geom_sims = []
    for i in range(len(networks)):
        for j in range(i + 1, len(networks)):
            topo_sims.append(topological_similarity(networks[i], networks[j]))
            geom_sims.append(geometric_similarity(networks[i], networks[j]))

    print("\n  COMBINED SIMILARITY (50/50 weighting)")
    print(f"    Mean:  {stats['mean']:.3f}")
    print(f"    Std:   {stats['std']:.3f}")
    print(f"    Min:   {stats['min']:.3f}")
    print(f"    Max:   {stats['max']:.3f}")

    print("\n  TOPOLOGICAL SIMILARITY (Edit Distance based)")
    print(f"    Mean:  {sum(topo_sims)/len(topo_sims):.3f}")
    print(f"    Min:   {min(topo_sims):.3f}")
    print(f"    Max:   {max(topo_sims):.3f}")

    print("\n  GEOMETRIC SIMILARITY (Parameter based)")
    print(f"    Mean:  {sum(geom_sims)/len(geom_sims):.3f}")
    print(f"    Min:   {min(geom_sims):.3f}")
    print(f"    Max:   {max(geom_sims):.3f}")

    # Show most similar and most different pairs
    print("\n  MOST SIMILAR PAIR:")
    max_idx = all_sims.index(max(all_sims))
    show_pair_comparison(networks, max_idx, len(networks), max(all_sims))

    print("\n  MOST DIFFERENT PAIR:")
    min_idx = all_sims.index(min(all_sims))
    show_pair_comparison(networks, min_idx, len(networks), min(all_sims))


def show_pair_comparison(networks: List[Dict], pair_idx: int, n: int, similarity: float):
    """Show details about a specific pair."""
    # Convert linear index back to i, j
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx == pair_idx:
                seq_i = extract_type_sequence(networks[i])
                seq_j = extract_type_sequence(networks[j])
                print(f"    Network {i+1}: {seq_i}")
                print(f"    Network {j+1}: {seq_j}")
                print(f"    Similarity: {similarity:.3f}")
                return
            idx += 1


def _get_approach_stats(networks: List[Dict]) -> Optional[float]:
    """Calculate mean pairwise similarity if enough networks exist."""
    if networks and len(networks) >= 2:
        _, stats = calculate_pairwise_similarities(networks)
        return stats['mean']
    return None


def compare_approaches(graphs_dir: Path):
    """Compare similarity distributions between approaches."""
    print("\n" + "=" * 70)
    print("INTER-APPROACH COMPARISON")
    print("=" * 70)

    # Load all networks
    all_networks = {
        'random': load_networks(graphs_dir, "random_"),
        'least_generated': load_networks(graphs_dir, "least_generated_"),
        'hybrid': load_networks(graphs_dir, "hybrid_"),
        'llm': load_networks(graphs_dir, "llm_"),
    }

    # Calculate stats for each approach
    approaches = {}
    for name, nets in all_networks.items():
        stat = _get_approach_stats(nets)
        if stat is not None:
            approaches[name] = stat

    if len(approaches) < 2:
        print("  Need at least 2 approaches with 2+ networks for comparison")
        return

    print("\n  Within-approach mean similarity (lower = more diverse):")
    for approach, mean_sim in sorted(approaches.items(), key=lambda x: x[1]):
        print(f"    {approach:20s}: {mean_sim:.3f}")

    # Find most diverse approach
    most_diverse = min(approaches.items(), key=lambda x: x[1])
    print(f"\n  Most diverse approach: {most_diverse[0]} (similarity: {most_diverse[1]:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Analyze similarity between road networks")
    parser.add_argument(
        "--approach",
        type=str,
        choices=["random", "least_generated", "hybrid", "llm", "all"],
        default="all",
        help="Which approach to analyze"
    )
    parser.add_argument(
        "--graphs-dir",
        type=str,
        default="outputs/graphs",
        help="Directory containing networks"
    )

    args = parser.parse_args()
    graphs_dir = Path(args.graphs_dir)

    if not graphs_dir.exists():
        print(f"Directory not found: {graphs_dir}")
        return

    if args.approach in ("random", "all"):
        print("\n" + "=" * 70)
        print("RANDOM GENERATOR - Similarity Analysis")
        print("=" * 70)
        analyze_approach(graphs_dir, "random_")

    if args.approach in ("least_generated", "all"):
        print("\n" + "=" * 70)
        print("LEAST GENERATED GENERATOR - Similarity Analysis")
        print("=" * 70)
        analyze_approach(graphs_dir, "least_generated_")

    if args.approach in ("hybrid", "all"):
        print("\n" + "=" * 70)
        print("HYBRID GENERATOR - Similarity Analysis")
        print("=" * 70)
        analyze_approach(graphs_dir, "hybrid_")

    if args.approach in ("llm", "all"):
        print("\n" + "=" * 70)
        print("LLM GENERATOR - Similarity Analysis")
        print("=" * 70)
        analyze_approach(graphs_dir, "llm_")

    if args.approach == "all":
        compare_approaches(graphs_dir)


if __name__ == "__main__":
    main()
