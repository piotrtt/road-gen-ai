"""
Filesystem-based storage for generated road networks.

Replaces vector database storage with simple JSON file storage.
Networks are stored as individual JSON files and can be loaded
for similarity comparison.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class NetworkStorage:
    """
    Manage storage and retrieval of generated road networks.

    Stores networks as individual JSON files with metadata.
    Supports loading all networks for similarity comparison.
    """

    def __init__(self, storage_dir: Path = None):
        """
        Initialize network storage.

        Args:
            storage_dir: Directory for storing networks.
                        Defaults to outputs/graphs/
        """
        default_dir = Path(__file__).parent.parent.parent / "outputs" / "graphs"
        self.storage_dir = storage_dir or default_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        road_network: Dict,
        approach: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save a road network to storage.

        Args:
            road_network: The road network dictionary
            approach: Generation approach used (e.g., 'hybrid', 'random')
            metadata: Optional additional metadata

        Returns:
            Path to saved file
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{approach}_network_{timestamp}.json"
        filepath = self.storage_dir / filename

        # Add metadata to network
        data = {
            "road_network": road_network.get("road_network", []),
            "metadata": {
                "approach": approach,
                "generated_at": datetime.now().isoformat(),
                "num_components": len(road_network.get("road_network", [])),
                **(metadata or {})
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def load(self, filepath: Path) -> Dict:
        """Load a single road network from file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def load_all(self, approach: Optional[str] = None) -> List[Dict]:
        """
        Load all networks from storage.

        Args:
            approach: Optional filter by approach name

        Returns:
            List of road network dictionaries
        """
        pattern = f"{approach}_*.json" if approach else "*.json"
        networks = []

        for filepath in sorted(self.storage_dir.glob(pattern)):
            try:
                networks.append(self.load(filepath))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {filepath}: {e}")

        return networks

    def count(self, approach: Optional[str] = None) -> int:
        """Count networks in storage."""
        pattern = f"{approach}_*.json" if approach else "*.json"
        return len(list(self.storage_dir.glob(pattern)))

    def clear(self, approach: Optional[str] = None):
        """
        Clear networks from storage.

        Args:
            approach: Optional filter - only clear networks from this approach
        """
        pattern = f"{approach}_*.json" if approach else "*.json"
        for filepath in self.storage_dir.glob(pattern):
            filepath.unlink()

    def get_recent(self, n: int = 10, approach: Optional[str] = None) -> List[Dict]:
        """
        Get the N most recently generated networks.

        Args:
            n: Number of networks to retrieve
            approach: Optional filter by approach

        Returns:
            List of road network dictionaries (most recent first)
        """
        pattern = f"{approach}_*.json" if approach else "*.json"
        files = sorted(self.storage_dir.glob(pattern), reverse=True)

        networks = []
        for filepath in files[:n]:
            try:
                networks.append(self.load(filepath))
            except (json.JSONDecodeError, IOError):
                continue

        return networks


if __name__ == "__main__":
    # Test storage
    storage = NetworkStorage()
    print(f"Storage directory: {storage.storage_dir}")
    print(f"Total networks: {storage.count()}")
    print(f"Random networks: {storage.count('random')}")
    print(f"Least generated: {storage.count('least_generated')}")
