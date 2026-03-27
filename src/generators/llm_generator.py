"""
LLM-based road network generator.

Uses function calling to generate road networks directly from an LLM.
Prompts are loaded from explicit template files as recommended by Professor Bade.
"""

import json
import os
from typing import Dict, List, Optional
from src.generators.base_generator import BaseGenerator
from src.generators.network_storage import NetworkStorage
from src.llm_engine.client import LLMClient
from src.llm_engine.prompts import (
    load_system_prompt,
    load_user_prompt,
    get_function_schema,
)


class LLMGenerator(BaseGenerator):
    """
    Generate road networks using LLM with function calling.

    Algorithm:
    1. Load existing networks from storage (for diversity context)
    2. Build prompt with system instructions and existing network examples
    3. Call LLM with function calling to get structured output
    4. Validate and return the generated network

    The LLM is instructed to generate networks that are different from
    existing ones, providing natural language-based diversity.
    """

    def __init__(
        self,
        output_dir=None,
        model_name: str = None,
        max_examples: int = 5,
        include_existing: bool = True,
    ):
        """
        Initialize LLM generator.

        Args:
            output_dir: Directory for saving generated networks
            model_name: LLM model to use (default: from env or gpt-4)
            max_examples: Max existing networks to include in prompt
            include_existing: Whether to include existing networks for diversity
        """
        super().__init__(output_dir)

        # Get model from env or default
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.max_examples = max_examples
        self.include_existing = include_existing

        # Initialize LLM client
        self.client = LLMClient(model_name=self.model_name)

        # Network storage for loading existing networks
        self.storage = NetworkStorage(storage_dir=output_dir)

        # Load prompts
        self.system_prompt = load_system_prompt()

    def get_name(self) -> str:
        return "llm"

    def _build_tool_definition(self) -> List[Dict]:
        """Build the tool definition for function calling."""
        schema = get_function_schema()
        return [{
            "type": "function",
            "function": schema
        }]

    def _validate_network(self, network: Dict) -> bool:
        """
        Validate that the generated network has required fields.

        Args:
            network: Generated network dictionary

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(network, dict):
            return False

        road_network = network.get("road_network", [])
        if not isinstance(road_network, list) or len(road_network) == 0:
            return False

        required_fields = ["id", "sequence_index", "type", "lane_width", "right_lanes", "left_lanes"]
        valid_types = {"straight", "curve", "lane_switch", "fork", "t_intersection",
                      "intersection", "roundabout", "u_shape"}

        for component in road_network:
            # Check required fields
            for field in required_fields:
                if field not in component:
                    print(f"  Missing field: {field}")
                    return False

            # Check valid type
            if component["type"] not in valid_types:
                print(f"  Invalid type: {component['type']}")
                return False

        return True

    def _fix_sequence_indices(self, network: Dict) -> Dict:
        """Ensure sequence indices are correct."""
        for i, component in enumerate(network.get("road_network", [])):
            component["sequence_index"] = i
            # Fix ID if needed
            comp_type = component["type"]
            component["id"] = f"{comp_type}_{i + 1}"
        return network

    def generate(self, num_components: int, max_retries: int = 3) -> Dict:
        """
        Generate a road network using the LLM.

        Args:
            num_components: Number of components to generate
            max_retries: Maximum retry attempts if generation fails

        Returns:
            Generated road network dictionary
        """
        # Load existing networks for diversity context
        existing = []
        if self.include_existing:
            existing = self.storage.load_all(approach="llm")

        # Build user prompt
        user_prompt = load_user_prompt(
            num_components=num_components,
            existing_networks=existing,
            max_examples=self.max_examples
        )

        # Get tool definition
        tools = self._build_tool_definition()

        # Try to generate with retries
        for attempt in range(max_retries):
            try:
                # Call LLM with function calling
                result = self.client.query_structured(
                    prompt=user_prompt,
                    tools=tools,
                    system_prompt=self.system_prompt
                )

                if result is None:
                    print(f"  Attempt {attempt + 1}: No function call returned")
                    continue

                # Validate result
                if not self._validate_network(result):
                    print(f"  Attempt {attempt + 1}: Invalid network structure")
                    continue

                # Fix sequence indices
                result = self._fix_sequence_indices(result)

                # Check component count
                actual_count = len(result.get("road_network", []))
                if actual_count != num_components:
                    print(f"  Attempt {attempt + 1}: Wrong component count ({actual_count} vs {num_components})")
                    # Accept if close enough
                    if abs(actual_count - num_components) <= 2:
                        return result
                    continue

                return result

            except Exception as e:
                print(f"  Attempt {attempt + 1}: Error - {e}")
                continue

        # If all retries failed, raise error
        raise RuntimeError(f"Failed to generate valid network after {max_retries} attempts")

    def generate_multiple(self, num_networks: int, num_components: int) -> list:
        """
        Generate multiple road networks.

        Each network is saved to storage, so subsequent generations
        can see previous ones for diversity.

        Args:
            num_networks: Number of networks to generate
            num_components: Number of components per network

        Returns:
            List of Paths to saved network files
        """
        saved_files = []

        for i in range(num_networks):
            print(f"Generating network {i+1}/{num_networks}...")

            try:
                network = self.generate(num_components)

                # Save to storage
                filepath = self.storage.save(
                    network,
                    approach="llm",
                    metadata={
                        "model": self.model_name,
                        "num_components": len(network.get("road_network", [])),
                    }
                )
                saved_files.append(filepath)

                types = [c["type"] for c in network["road_network"]]
                print(f"  Types: {types}")
                print(f"  Saved to {filepath.name}")

            except Exception as e:
                print(f"  Failed: {e}")
                continue

        return saved_files


if __name__ == "__main__":
    # Test LLM generator
    print("Testing LLM Generator...")
    print("=" * 60)

    generator = LLMGenerator(include_existing=False)

    print(f"Model: {generator.model_name}")
    print(f"Max examples: {generator.max_examples}")
    print()

    # Generate a single network
    print("Generating single test network...")
    try:
        network = generator.generate(num_components=5)
        print(f"\nGenerated network:")
        print(json.dumps(network, indent=2))
    except Exception as e:
        print(f"Error: {e}")
