"""
Component library defining available road components and their parameter ranges.

Based on existing Pydantic models in src/road_components/definitions.py
and inspired by RoadGen paper's approach.
"""

import math
import random
from typing import Dict, List, Any
from src.road_components.definitions import (
    StraightRoad,
    Curve,
    LaneSwitch,
    Fork,
    TIntersection,
    Intersection,
    Roundabout,
    UShapedRoad,
    RoadComponent
)


# Component library with discrete parameter ranges
COMPONENT_LIBRARY: Dict[str, Dict[str, Any]] = {
    "straight": {
        "class": StraightRoad,
        "params": {
            "lane_width": [2.5, 3.0, 3.5, 4.0],
            "right_lanes": [1, 2, 3],
            "left_lanes": [1, 2, 3],
            "length": [20, 50, 100, 150, 200]
        }
    },
    "curve": {
        "class": Curve,
        "params": {
            "lane_width": [2.5, 3.0, 3.5, 4.0],
            "right_lanes": [1, 2, 3],
            "left_lanes": [1, 2, 3],
            "radius": [10, 15, 20, 30, 40],
            "angle": [
                math.pi / 6,      # 30°
                math.pi / 4,      # 45°
                math.pi / 3,      # 60°
                math.pi / 2,      # 90°
                math.pi * 2 / 3,  # 120°
                math.pi * 3 / 4,  # 135°
                math.pi           # 180°
            ]
        }
    },
    "lane_switch": {
        "class": LaneSwitch,
        "params": {
            "lane_width": [2.5, 3.0, 3.5, 4.0],
            "right_lanes": [1, 2, 3],
            "left_lanes": [1, 2, 3],
            "left_lanes_out": [1, 2, 3],
            "right_lanes_out": [1, 2, 3]
        }
    },
    "fork": {
        "class": Fork,
        "params": {
            "lane_width": [2.5, 3.0, 3.5, 4.0],
            "right_lanes": [1, 2, 3],
            "left_lanes": [1, 2, 3],
            "angle": [math.pi / 4, math.pi / 3, math.pi / 2, math.pi * 2 / 3]
        }
    },
    "t_intersection": {
        "class": TIntersection,
        "params": {
            "lane_width": [2.5, 3.0, 3.5, 4.0],
            "right_lanes": [1, 2, 3],
            "left_lanes": [1, 2, 3],
            "angle": [math.pi / 4, math.pi / 3, math.pi / 2, math.pi * 2 / 3]
        }
    },
    "intersection": {
        "class": Intersection,
        "params": {
            "lane_width": [2.5, 3.0, 3.5, 4.0],
            "right_lanes": [1, 2, 3],
            "left_lanes": [1, 2, 3],
            "spacing": [5, 10, 15, 20]
        }
    },
    "roundabout": {
        "class": Roundabout,
        "params": {
            "lane_width": [2.5, 3.0, 3.5, 4.0],
            "right_lanes": [1, 2, 3],
            "left_lanes": [1, 2, 3],
            "radius": [10, 15, 20, 25],
            "num_exits": [3, 4, 5],
            "arm_length": [5, 10, 15, 20]
        }
    },
    "u_shape": {
        "class": UShapedRoad,
        "params": {
            "lane_width": [2.5, 3.0, 3.5, 4.0],
            "right_lanes": [1, 2, 3],
            "left_lanes": [1, 2, 3],
            "length": [10, 20, 30, 40],
            "distance": [10, 15, 20, 25],
            "direction": ["left", "right"]
        }
    }
}


def get_all_component_types() -> List[str]:
    """Return list of all available component type names."""
    return list(COMPONENT_LIBRARY.keys())


def generate_random_component(component_type: str, sequence_index: int = 0) -> Dict[str, Any]:
    """
    Generate a random component of the specified type with random parameters.

    Args:
        component_type: Type of component (e.g., "straight", "curve")
        sequence_index: Index in the road network sequence

    Returns:
        Dictionary representation of the component
    """
    if component_type not in COMPONENT_LIBRARY:
        raise ValueError(f"Unknown component type: {component_type}")

    config = COMPONENT_LIBRARY[component_type]
    params = {}

    # Randomly select value for each parameter
    for param_name, param_values in config["params"].items():
        params[param_name] = random.choice(param_values)

    # Create component instance
    component_class = config["class"]
    component = component_class(
        id=f"{component_type}_{sequence_index + 1}",
        sequence_index=sequence_index,
        **params
    )

    # Return as dictionary
    return component.model_dump()


def create_component_from_params(
    component_type: str,
    params: Dict[str, Any],
    sequence_index: int = 0
) -> Dict[str, Any]:
    """
    Create a component of specified type with given parameters.

    Args:
        component_type: Type of component
        params: Parameter dictionary
        sequence_index: Index in sequence

    Returns:
        Dictionary representation of the component
    """
    if component_type not in COMPONENT_LIBRARY:
        raise ValueError(f"Unknown component type: {component_type}")

    config = COMPONENT_LIBRARY[component_type]
    component_class = config["class"]

    component = component_class(
        id=f"{component_type}_{sequence_index + 1}",
        sequence_index=sequence_index,
        **params
    )

    return component.model_dump()


if __name__ == "__main__":
    # Test component generation
    print("Available component types:", get_all_component_types())
    print("\nSample components:")

    for comp_type in get_all_component_types():
        comp = generate_random_component(comp_type, 0)
        print(f"\n{comp_type}:")
        print(comp)
