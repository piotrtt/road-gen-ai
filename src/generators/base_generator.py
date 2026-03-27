"""
Base generator class with common functionality for all road network generators.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class BaseGenerator(ABC):
    """Abstract base class for road network generators."""

    def __init__(self, output_dir: Path = None):
        """
        Initialize base generator.

        Args:
            output_dir: Directory for storing generated road networks.
                       Defaults to outputs/graphs/
        """
        default_output = Path(__file__).parent.parent.parent / "outputs" / "graphs"
        self.output_dir = output_dir or default_output
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate(self, num_components: int) -> Dict:
        """
        Generate a road network with specified number of components.

        Args:
            num_components: Number of road components to include

        Returns:
            Dictionary with "road_network" key containing list of components
        """
        pass

    def save_network(self, road_network: Dict, filename: str = None) -> Path:
        """
        Save road network to JSON file.

        Args:
            road_network: The road network dictionary
            filename: Optional custom filename. If None, auto-generates timestamp-based name.

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"road_network_{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(road_network, f, indent=2)

        return filepath

    def load_network(self, filepath: Path) -> Dict:
        """Load road network from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def generate_multiple(self, num_networks: int, num_components: int) -> List[Path]:
        """
        Generate multiple road networks.

        Args:
            num_networks: Number of networks to generate
            num_components: Number of components per network

        Returns:
            List of Paths to saved network files
        """
        saved_files = []

        for i in range(num_networks):
            print(f"Generating network {i+1}/{num_networks}...")
            network = self.generate(num_components)
            filepath = self.save_network(network, f"{self.get_name()}_network_{i+1:03d}.json")
            saved_files.append(filepath)
            print(f"  Saved to {filepath.name}")

        return saved_files

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this generator (e.g., 'random', 'least_generated')."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(output_dir={self.output_dir})"
