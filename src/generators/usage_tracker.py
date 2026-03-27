"""
Usage tracker for component types across all generation runs.

Implements RoadGen's approach of persistent global counters for diversity.
"""

import json
from pathlib import Path
from typing import Dict
from src.generators.component_library import get_all_component_types


# Default storage location
DEFAULT_USAGE_FILE = Path(__file__).parent.parent.parent / "outputs" / "component_usage.json"


class UsageTracker:
    """
    Track component type usage globally across all generation runs.

    Persists counts to file to ensure long-term diversity.
    """

    def __init__(self, storage_path: Path = None):
        """
        Initialize usage tracker.

        Args:
            storage_path: Path to JSON file for storing counts.
                         Defaults to outputs/component_usage.json
        """
        self.storage_path = storage_path or DEFAULT_USAGE_FILE
        self.usage_counts = self._load_counts()

    def _load_counts(self) -> Dict[str, int]:
        """Load usage counts from file, or initialize if not exists."""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        else:
            # Initialize all component types to 0
            return {comp_type: 0 for comp_type in get_all_component_types()}

    def save(self):
        """Save current usage counts to file."""
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.storage_path, 'w') as f:
            json.dump(self.usage_counts, f, indent=2)

    def increment(self, component_type: str):
        """
        Increment usage count for a component type.

        Args:
            component_type: Type of component that was used
        """
        if component_type not in self.usage_counts:
            self.usage_counts[component_type] = 0

        self.usage_counts[component_type] += 1
        self.save()  # Auto-save after each increment

    def get_count(self, component_type: str) -> int:
        """Get usage count for a component type."""
        return self.usage_counts.get(component_type, 0)

    def get_all_counts(self) -> Dict[str, int]:
        """Get all usage counts."""
        return self.usage_counts.copy()

    def get_least_used_types(self) -> list:
        """
        Get list of component types with minimum usage count.

        Returns list (not single item) to handle ties.
        """
        if not self.usage_counts:
            return []

        min_count = min(self.usage_counts.values())
        return [comp_type for comp_type, count in self.usage_counts.items()
                if count == min_count]

    def reset(self):
        """Reset all counts to zero (for testing)."""
        self.usage_counts = {comp_type: 0 for comp_type in get_all_component_types()}
        self.save()

    def __repr__(self):
        return f"UsageTracker(counts={self.usage_counts})"


if __name__ == "__main__":
    # Test usage tracker
    tracker = UsageTracker()
    print("Initial counts:")
    print(tracker.get_all_counts())

    # Simulate some usage
    tracker.increment("straight")
    tracker.increment("curve")
    tracker.increment("straight")

    print("\nAfter increments:")
    print(tracker.get_all_counts())
    print(f"\nLeast used types: {tracker.get_least_used_types()}")
