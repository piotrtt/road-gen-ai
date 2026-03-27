"""
Hybrid road network generator.

Combines candidate generation (from Random or Least Generated) with
custom similarity-based selection to maximize diversity.

Uses Professor Bade's recommended custom similarity metric:
- Topological similarity: Edit Distance on component type sequences
- Geometric similarity: Parameter distribution comparison

No vector database or embeddings are used.
"""

import random
from typing import Dict, List, Optional
from src.generators.base_generator import BaseGenerator
from src.generators.random_generator import RandomGenerator
from src.generators.least_generated import LeastGeneratedGenerator
from src.generators.network_storage import NetworkStorage
from src.metrics.similarity import combined_similarity


class HybridGenerator(BaseGenerator):
    """
    Generate road networks by selecting most diverse candidates.

    Algorithm (Diversity-Maximizing Selection):
    1. Generate N candidate road networks using base generator
    2. Load all existing hybrid networks from storage
    3. For each candidate, compute similarity to all existing networks
       using custom metric (edit distance + parameter comparison)
    4. Select the candidate with LOWEST average similarity (most diverse)
    5. Save selected network to storage

    This ensures each new network maximizes diversity from existing ones.
    """

    def __init__(
        self,
        output_dir=None,
        num_candidates: int = 5,
        base_approach: str = "random",
        topo_weight: float = 0.5,
        geom_weight: float = 0.5,
    ):
        """
        Initialize hybrid generator.

        Args:
            output_dir: Directory for saving generated networks
            num_candidates: Number of candidate networks to generate per selection
            base_approach: Which generator to use for candidates ('random' or 'least_generated')
            topo_weight: Weight for topological similarity (default 0.5)
            geom_weight: Weight for geometric similarity (default 0.5)
        """
        super().__init__(output_dir)
        self.num_candidates = num_candidates
        self.base_approach = base_approach
        self.topo_weight = topo_weight
        self.geom_weight = geom_weight

        # Initialize base generator
        if base_approach == "random":
            self.base_generator = RandomGenerator(output_dir=output_dir)
        elif base_approach == "least_generated":
            self.base_generator = LeastGeneratedGenerator(output_dir=output_dir)
        else:
            raise ValueError(f"Unknown base approach: {base_approach}")

        # Network storage for similarity comparison
        self.storage = NetworkStorage(storage_dir=output_dir)

    def get_name(self) -> str:
        return f"hybrid_{self.base_approach}"

    def _calculate_diversity_score(
        self,
        candidate: Dict,
        existing_networks: List[Dict]
    ) -> float:
        """
        Calculate diversity score for a candidate.

        Higher score = more different from existing networks.

        Args:
            candidate: Candidate road network
            existing_networks: List of existing networks to compare against

        Returns:
            Diversity score (1 - average similarity)
        """
        if not existing_networks:
            return 1.0  # First network is maximally diverse

        similarities = [
            combined_similarity(
                candidate,
                existing,
                topo_weight=self.topo_weight,
                geom_weight=self.geom_weight
            )
            for existing in existing_networks
        ]

        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity  # Convert to diversity

    def _select_most_diverse(
        self,
        candidates: List[Dict],
        existing_networks: List[Dict]
    ) -> tuple:
        """
        Select the most diverse candidate.

        Args:
            candidates: List of candidate networks
            existing_networks: List of existing networks

        Returns:
            Tuple of (selected_network, diversity_score, all_scores)
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        if not existing_networks:
            # First generation - random selection
            return candidates[0], 1.0, [1.0] * len(candidates)

        # Calculate diversity score for each candidate
        diversity_scores = [
            self._calculate_diversity_score(candidate, existing_networks)
            for candidate in candidates
        ]

        # Select candidate with highest diversity (lowest similarity)
        best_idx = diversity_scores.index(max(diversity_scores))

        return candidates[best_idx], diversity_scores[best_idx], diversity_scores

    def generate(self, num_components: int) -> Dict:
        """
        Generate a road network by selecting the most diverse candidate.

        Args:
            num_components: Number of components in the network

        Returns:
            The most diverse road network dictionary
        """
        # Generate candidates
        candidates = [
            self.base_generator.generate(num_components)
            for _ in range(self.num_candidates)
        ]

        # Load existing networks for comparison
        existing = self.storage.load_all(approach="hybrid")

        # Select most diverse
        selected, diversity_score, all_scores = self._select_most_diverse(
            candidates, existing
        )

        # Store selection metadata
        selected["_selection_metadata"] = {
            "diversity_score": diversity_score,
            "candidate_scores": all_scores,
            "num_candidates": self.num_candidates,
            "num_existing": len(existing),
        }

        return selected

    def generate_multiple(self, num_networks: int, num_components: int) -> list:
        """
        Generate multiple diverse road networks.

        Each network is selected to maximize diversity from all previously
        generated networks (including those generated in this batch).

        Args:
            num_networks: Number of networks to generate
            num_components: Number of components per network

        Returns:
            List of Paths to saved network files
        """
        saved_files = []

        for i in range(num_networks):
            print(f"Generating network {i+1}/{num_networks}...")

            # Generate and select most diverse
            network = self.generate(num_components)

            # Extract selection info before saving
            selection_info = network.pop("_selection_metadata", {})

            # Save to storage
            filepath = self.storage.save(
                network,
                approach="hybrid",
                metadata={
                    "base_approach": self.base_approach,
                    "diversity_score": selection_info.get("diversity_score", 1.0),
                    "num_candidates": self.num_candidates,
                }
            )
            saved_files.append(filepath)

            diversity = selection_info.get("diversity_score", 1.0)
            print(f"  Diversity score: {diversity:.3f}")
            print(f"  Saved to {filepath.name}")

        return saved_files

    def get_statistics(self) -> Dict:
        """Get statistics about hybrid generation."""
        hybrid_networks = self.storage.load_all(approach="hybrid")

        return {
            "total_hybrid_networks": len(hybrid_networks),
            "base_approach": self.base_approach,
            "num_candidates": self.num_candidates,
            "weights": {
                "topological": self.topo_weight,
                "geometric": self.geom_weight,
            }
        }


if __name__ == "__main__":
    # Test hybrid generator
    print("Testing Hybrid Generator...")
    print("=" * 60)

    generator = HybridGenerator(
        num_candidates=5,
        base_approach="random",
        topo_weight=0.5,
        geom_weight=0.5,
    )

    print(f"Base approach: {generator.base_approach}")
    print(f"Candidates per selection: {generator.num_candidates}")
    print()

    # Generate a few networks
    print("Generating 3 test networks...")
    for i in range(3):
        network = generator.generate(num_components=7)
        selection_info = network.pop("_selection_metadata", {})
        types = [c["type"] for c in network["road_network"]]

        print(f"\nNetwork {i+1}:")
        print(f"  Types: {types}")
        print(f"  Diversity: {selection_info.get('diversity_score', 'N/A'):.3f}")
        print(f"  Candidate scores: {[f'{s:.3f}' for s in selection_info.get('candidate_scores', [])]}")
