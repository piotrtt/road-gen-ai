"""
Random road network generator.

Implements pure random selection of components and parameters,
based on the RoadGen paper's random baseline approach.
"""

import random
from typing import Dict
from src.generators.base_generator import BaseGenerator
from src.generators.component_library import get_all_component_types, generate_random_component


class RandomGenerator(BaseGenerator):
    """
    Generate road networks with purely random component selection.

    Algorithm:
    1. Select random component type from available library
    2. Generate random parameters within defined ranges
    3. Repeat until target number of components reached

    Note: No spatial validation or collision detection in Phase 0.
    """

    def get_name(self) -> str:
        return "random"

    def generate(self, num_components: int) -> Dict:
        """
        Generate a road network with random components.

        Args:
            num_components: Number of components to include

        Returns:
            Dictionary with "road_network" key containing list of components
        """
        if num_components <= 0:
            raise ValueError("num_components must be positive")

        components = []
        available_types = get_all_component_types()

        for i in range(num_components):
            # Randomly select component type
            component_type = random.choice(available_types)

            # Generate component with random parameters
            component = generate_random_component(component_type, sequence_index=i)

            components.append(component)

        return {"road_network": components}


if __name__ == "__main__":
    # Test random generator
    generator = RandomGenerator()

    print("Generating 3 random road networks with 5 components each...\n")

    for i in range(3):
        network = generator.generate(num_components=5)
        print(f"\nNetwork {i+1}:")
        print(f"  Component sequence: {[c['type'] for c in network['road_network']]}")

        # Show first component details
        if network['road_network']:
            print(f"  First component: {network['road_network'][0]}")
