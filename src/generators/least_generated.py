"""
Least Generated (Count-Based) road network generator.

Implements the RoadGen paper's count-based approach with random tie-breaking
for components that have the same minimum usage count.
"""

import random
from typing import Dict
from src.generators.base_generator import BaseGenerator
from src.generators.component_library import get_all_component_types, generate_random_component
from src.generators.usage_tracker import UsageTracker


class LeastGeneratedGenerator(BaseGenerator):
    """
    Generate road networks by selecting least-used component types.

    Algorithm (from RoadGen paper):
    1. Maintain global usage counter for each component type
    2. For each component position:
       - Find all component types with minimum usage count
       - RANDOMLY select among ties (critical tie-breaking logic!)
       - Generate component with random parameters
       - Increment usage counter
    3. Persist counts globally across all generation runs

    This ensures long-term diversity across many generated maps.
    """

    def __init__(self, output_dir=None, usage_tracker: UsageTracker = None):
        """
        Initialize least generated generator.

        Args:
            output_dir: Directory for saving generated networks
            usage_tracker: Optional custom usage tracker. If None, uses default.
        """
        super().__init__(output_dir)
        self.usage_tracker = usage_tracker or UsageTracker()

    def get_name(self) -> str:
        return "least_generated"

    def _select_least_used_type(self) -> str:
        """
        Select component type with minimum usage count.

        Critical: When multiple types have same min count, randomly choose!
        This is the RoadGen paper's tie-breaking logic.

        Returns:
            Selected component type
        """
        least_used_types = self.usage_tracker.get_least_used_types()

        if not least_used_types:
            # Fallback to all types if tracker is empty
            least_used_types = get_all_component_types()

        # RANDOM selection among ties (RoadGen's approach)
        return random.choice(least_used_types)

    def generate(self, num_components: int) -> Dict:
        """
        Generate a road network using least-generated component selection.

        Args:
            num_components: Number of components to include

        Returns:
            Dictionary with "road_network" key containing list of components
        """
        if num_components <= 0:
            raise ValueError("num_components must be positive")

        components = []

        for i in range(num_components):
            # Select least-used component type (with random tie-breaking)
            component_type = self._select_least_used_type()

            # Generate component with random parameters
            component = generate_random_component(component_type, sequence_index=i)

            # Increment usage counter (persists to file automatically)
            self.usage_tracker.increment(component_type)

            components.append(component)

        return {"road_network": components}

    def get_usage_statistics(self) -> Dict[str, int]:
        """Get current usage counts for all component types."""
        return self.usage_tracker.get_all_counts()

    def reset_usage_counts(self):
        """Reset usage counts to zero (for testing purposes)."""
        self.usage_tracker.reset()


if __name__ == "__main__":
    # Test least generated generator
    generator = LeastGeneratedGenerator()

    print("Initial usage counts:")
    print(generator.get_usage_statistics())
    print()

    print("Generating 3 road networks with 8 components each...\n")

    for i in range(3):
        network = generator.generate(num_components=8)
        component_types = [c['type'] for c in network['road_network']]

        print(f"\nNetwork {i+1}:")
        print(f"  Component sequence: {component_types}")
        print(f"  Type distribution: {dict((t, component_types.count(t)) for t in set(component_types))}")

    print("\n\nFinal usage counts after generating 3 networks:")
    print(generator.get_usage_statistics())
