"""
Validated road network generator with spatial overlap detection.

Uses the RoadGen-style approach:
1. Generate candidate component with random parameters
2. Check for overlap with existing components
3. If overlap, try different parameters
4. If all parameters fail, try different component type
5. Continue until target component count reached or max attempts exceeded
"""

import random
from typing import Dict, List, Optional, Tuple
from src.generators.base_generator import BaseGenerator
from src.generators.component_library import (
    COMPONENT_LIBRARY,
    get_all_component_types,
    generate_random_component
)
from src.generators.spatial_validator import SpatialValidator, PlacedComponent


class ValidatedRandomGenerator(BaseGenerator):
    """
    Random generator with spatial validation.

    Like RandomGenerator but validates each component placement
    to ensure no overlaps occur.
    """

    def __init__(
        self,
        safety_margin: float = 1.0,
        max_placement_attempts: int = 50,
        max_type_retries: int = 5
    ):
        """
        Initialize validated generator.

        Args:
            safety_margin: Extra padding between components (meters)
            max_placement_attempts: Max parameter combinations per component
            max_type_retries: Max different component types to try if all fail
        """
        super().__init__()
        self.safety_margin = safety_margin
        self.max_placement_attempts = max_placement_attempts
        self.max_type_retries = max_type_retries
        self.validator = SpatialValidator(safety_margin=safety_margin)

    def get_name(self) -> str:
        return "validated_random"

    def generate(self, num_components: int) -> Dict:
        """
        Generate a road network with validated placements.

        Args:
            num_components: Target number of components

        Returns:
            Dictionary with "road_network" key
        """
        if num_components <= 0:
            raise ValueError("num_components must be positive")

        self.validator.reset()
        components = []
        available_types = get_all_component_types()

        for i in range(num_components):
            placed = self._place_next_component(i, available_types)

            if placed is None:
                # Could not place any component - stop early
                print(f"Warning: Could not place component {i+1}, stopping at {len(components)}")
                break

            components.append(placed.component)

        return {"road_network": components}

    def _place_next_component(
        self,
        sequence_index: int,
        available_types: List[str]
    ) -> Optional[PlacedComponent]:
        """
        Try to place the next component.

        Args:
            sequence_index: Index in the network sequence
            available_types: List of component types to try

        Returns:
            PlacedComponent if successful, None if all failed
        """
        # Shuffle types for randomness
        types_to_try = random.sample(available_types, len(available_types))

        for retry in range(self.max_type_retries):
            if retry >= len(types_to_try):
                break

            component_type = types_to_try[retry]
            param_options = COMPONENT_LIBRARY[component_type]["params"]

            # Try to find valid placement with different parameters
            valid_component = self.validator.try_parameters(
                component_type,
                param_options,
                max_attempts=self.max_placement_attempts
            )

            if valid_component is not None:
                # Update with proper ID and sequence index
                valid_component["id"] = f"{component_type}_{sequence_index + 1}"
                valid_component["sequence_index"] = sequence_index

                # Place it
                placed = self.validator.place_component(valid_component, validate=False)
                return placed

        return None


class ValidatedLeastGeneratedGenerator(BaseGenerator):
    """
    Least Generated approach with spatial validation.

    Combines count-based diversity with overlap detection.
    """

    def __init__(
        self,
        safety_margin: float = 1.0,
        max_placement_attempts: int = 50
    ):
        super().__init__()
        self.safety_margin = safety_margin
        self.max_placement_attempts = max_placement_attempts
        self.validator = SpatialValidator(safety_margin=safety_margin)

        # Import usage tracker
        from src.generators.usage_tracker import UsageTracker
        self.usage_tracker = UsageTracker()

    def get_name(self) -> str:
        return "validated_least_generated"

    def generate(self, num_components: int) -> Dict:
        """
        Generate a road network using least-used component types with validation.
        """
        if num_components <= 0:
            raise ValueError("num_components must be positive")

        self.validator.reset()
        components = []

        for i in range(num_components):
            placed = self._place_least_used_component(i)

            if placed is None:
                print(f"Warning: Could not place component {i+1}, stopping at {len(components)}")
                break

            components.append(placed.component)

        return {"road_network": components}

    def _place_least_used_component(
        self,
        sequence_index: int
    ) -> Optional[PlacedComponent]:
        """
        Try to place the least-used component type that doesn't overlap.
        """
        # Get types sorted by usage (least used first)
        least_used_types = self.usage_tracker.get_least_used_types()

        # Try each type in order of least usage
        for component_type in least_used_types:
            param_options = COMPONENT_LIBRARY[component_type]["params"]

            valid_component = self.validator.try_parameters(
                component_type,
                param_options,
                max_attempts=self.max_placement_attempts
            )

            if valid_component is not None:
                valid_component["id"] = f"{component_type}_{sequence_index + 1}"
                valid_component["sequence_index"] = sequence_index

                placed = self.validator.place_component(valid_component, validate=False)

                if placed:
                    # Update usage count
                    self.usage_tracker.increment(component_type)
                    return placed

        return None

    def reset_usage_counts(self):
        """Reset component usage counts."""
        self.usage_tracker.reset()


def test_validated_generator():
    """Test the validated generator with visualization."""
    import math

    print("=" * 60)
    print("Testing ValidatedRandomGenerator")
    print("=" * 60)

    generator = ValidatedRandomGenerator(
        safety_margin=0.5,
        max_placement_attempts=30,
        max_type_retries=4
    )

    # Generate a few networks
    for i in range(3):
        print(f"\nGenerating network {i+1}...")
        network = generator.generate(num_components=7)

        components = network["road_network"]
        print(f"  Components: {len(components)}")
        print(f"  Types: {[c['type'] for c in components]}")

        if len(components) > 0:
            # Visualize
            from src.generators.spatial_validator import visualize_network
            visualize_network(
                generator.validator,
                f"outputs/validated_network_{i+1}.png"
            )

    print("\n" + "=" * 60)
    print("Testing ValidatedLeastGeneratedGenerator")
    print("=" * 60)

    lg_generator = ValidatedLeastGeneratedGenerator(
        safety_margin=0.5,
        max_placement_attempts=30
    )

    network = lg_generator.generate(num_components=7)
    components = network["road_network"]
    print(f"Components: {len(components)}")
    print(f"Types: {[c['type'] for c in components]}")

    from src.generators.spatial_validator import visualize_network
    visualize_network(
        lg_generator.validator,
        "outputs/validated_least_generated.png"
    )


if __name__ == "__main__":
    test_validated_generator()
