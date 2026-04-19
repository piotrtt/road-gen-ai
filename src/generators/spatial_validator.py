"""
Spatial validation for road components using Shapely.

Implements overlap detection based on the RoadGen paper approach:
- Each component has a bounding polygon
- New components are validated against all existing polygons
- Uses Shapely for efficient polygon intersection checks

Reference: RoadGen paper Algorithm 1, func/judge.py

Install shapely: pip install shapely>=2.0
"""

import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

try:
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    # Fallback: simple polygon class for basic operations
    class Polygon:
        def __init__(self, coords):
            self.coords = list(coords)
            self.exterior = type('obj', (object,), {'coords': self.coords})()

        def intersection(self, _other):
            # Simplified: return empty intersection (no real collision detection)
            return type('obj', (object,), {'area': 0})()

        def buffer(self, _distance):
            return self

        @property
        def bounds(self):
            xs = [c[0] for c in self.coords]
            ys = [c[1] for c in self.coords]
            return (min(xs), min(ys), max(xs), max(ys))

    Point = None

    def unary_union(geoms):
        if not geoms:
            return Polygon([(0,0), (0,0), (0,0)])
        return geoms[0]

    print("Warning: shapely not installed. Spatial validation will be limited.")
    print("Install with: pip install shapely>=2.0")


@dataclass
class PlacedComponent:
    """A component that has been placed in the road network with spatial info."""
    component: Dict[str, Any]
    polygon: Polygon
    endpoint: Tuple[float, float]  # Where next component connects
    direction: float  # Direction in radians (0 = +X, π/2 = +Y)


class SpatialValidator:
    """
    Validates spatial placement of road components.

    Tracks bounding polygons of all placed components and checks
    new candidates for overlap.
    """

    def __init__(self, safety_margin: float = 1.0):
        """
        Initialize validator.

        Args:
            safety_margin: Extra padding around polygons in meters
        """
        self.safety_margin = safety_margin
        self.placed_components: List[PlacedComponent] = []
        self.covered_areas: List[Polygon] = []
        self.current_endpoint: Tuple[float, float] = (0.0, 0.0)
        self.current_direction: float = 0.0  # radians, 0 = +X direction

    def reset(self):
        """Clear all placed components and reset to origin."""
        self.placed_components = []
        self.covered_areas = []
        self.current_endpoint = (0.0, 0.0)
        self.current_direction = 0.0

    def check_overlap(self, candidate_polygon: Polygon) -> bool:
        """
        Check if candidate polygon overlaps with any existing component.

        Args:
            candidate_polygon: Shapely Polygon of the candidate component

        Returns:
            True if overlap detected (invalid placement), False if valid
        """
        if not self.covered_areas:
            return False

        # Add safety margin to candidate
        if self.safety_margin > 0:
            candidate_polygon = candidate_polygon.buffer(self.safety_margin)

        for existing_polygon in self.covered_areas:
            intersection = candidate_polygon.intersection(existing_polygon)
            if intersection.area > 0:
                return True

        return False

    def compute_bounding_polygon(
        self,
        component: Dict[str, Any],
        start: Tuple[float, float],
        direction: float
    ) -> Polygon:
        """
        Compute the bounding polygon for a component at given position/direction.

        Args:
            component: Component dictionary with type and parameters
            start: Starting point (x, y)
            direction: Direction in radians

        Returns:
            Shapely Polygon representing the component's footprint
        """
        comp_type = component.get("type", "straight")
        lane_width = component.get("lane_width", 3.5)
        right_lanes = component.get("right_lanes", 2)
        left_lanes = component.get("left_lanes", 2)
        total_width = (right_lanes + left_lanes) * lane_width

        if comp_type == "straight":
            return self._compute_straight_polygon(
                component, start, direction, total_width
            )
        elif comp_type == "curve":
            return self._compute_curve_polygon(
                component, start, direction, total_width
            )
        elif comp_type == "u_shape":
            return self._compute_ushape_polygon(
                component, start, direction, total_width
            )
        elif comp_type in ("fork", "t_intersection"):
            return self._compute_junction_polygon(
                component, start, direction, total_width
            )
        elif comp_type == "intersection":
            return self._compute_intersection_polygon(
                component, start, direction, total_width
            )
        elif comp_type == "roundabout":
            return self._compute_roundabout_polygon(
                component, start, direction, total_width
            )
        elif comp_type == "lane_switch":
            return self._compute_lane_switch_polygon(
                component, start, direction, total_width
            )
        else:
            # Default: simple rectangle
            return self._compute_straight_polygon(
                {"length": 20}, start, direction, total_width
            )

    def _rotate_point(
        self,
        point: Tuple[float, float],
        origin: Tuple[float, float],
        angle: float
    ) -> Tuple[float, float]:
        """Rotate a point around origin by angle (radians)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dx = point[0] - origin[0]
        dy = point[1] - origin[1]
        new_x = dx * cos_a - dy * sin_a + origin[0]
        new_y = dx * sin_a + dy * cos_a + origin[1]
        return (new_x, new_y)

    def _rotate_polygon(
        self,
        points: List[Tuple[float, float]],
        origin: Tuple[float, float],
        angle: float
    ) -> List[Tuple[float, float]]:
        """Rotate all polygon points around origin."""
        return [self._rotate_point(p, origin, angle) for p in points]

    def _compute_straight_polygon(
        self,
        component: Dict[str, Any],
        start: Tuple[float, float],
        direction: float,
        total_width: float
    ) -> Polygon:
        """Compute bounding box for straight road segment."""
        length = component.get("length", 20)
        half_width = total_width / 2

        # Rectangle in local coords (extending in +X direction)
        points = [
            (start[0], start[1] - half_width),          # bottom-left
            (start[0] + length, start[1] - half_width), # bottom-right
            (start[0] + length, start[1] + half_width), # top-right
            (start[0], start[1] + half_width),          # top-left
        ]

        # Rotate to actual direction
        rotated = self._rotate_polygon(points, start, direction)
        return Polygon(rotated)

    def _compute_curve_polygon(
        self,
        component: Dict[str, Any],
        start: Tuple[float, float],
        direction: float,
        total_width: float
    ) -> Polygon:
        """
        Compute bounding polygon for curve.
        Uses arc approximation with inner/outer radius.
        """
        radius = component.get("radius", 20)
        angle = component.get("angle", math.pi / 2)
        half_width = total_width / 2

        inner_r = radius - half_width
        outer_r = radius + half_width

        # Approximate arc with segments
        num_segments = max(8, int(abs(angle) / (math.pi / 16)))
        points = []

        # Determine curve direction (positive angle = left turn)
        curve_sign = 1 if angle > 0 else -1
        abs_angle = abs(angle)

        # Center of curve circle (perpendicular to start direction)
        center_x = start[0]
        center_y = start[1] + curve_sign * radius

        # Outer arc points
        for i in range(num_segments + 1):
            t = (i / num_segments) * abs_angle
            theta = -curve_sign * math.pi / 2 + curve_sign * t
            px = center_x + outer_r * math.cos(theta)
            py = center_y + outer_r * math.sin(theta)
            points.append((px, py))

        # Inner arc points (reverse order)
        for i in range(num_segments, -1, -1):
            t = (i / num_segments) * abs_angle
            theta = -curve_sign * math.pi / 2 + curve_sign * t
            px = center_x + inner_r * math.cos(theta)
            py = center_y + inner_r * math.sin(theta)
            points.append((px, py))

        # Rotate entire polygon to match direction
        rotated = self._rotate_polygon(points, start, direction)
        return Polygon(rotated)

    def _compute_ushape_polygon(
        self,
        component: Dict[str, Any],
        start: Tuple[float, float],
        direction: float,
        total_width: float
    ) -> Polygon:
        """Compute bounding polygon for U-shaped road (180° turn)."""
        length = component.get("length", 20)
        distance = component.get("distance", 15)
        turn_dir = component.get("direction", "right")
        half_width = total_width / 2

        # U-shape consists of: entry straight + arc + exit straight
        # Simplified as bounding rectangle
        dir_sign = 1 if turn_dir == "right" else -1

        points = [
            (start[0], start[1] - half_width),
            (start[0] + length, start[1] - half_width),
            (start[0] + length + distance, start[1] - half_width),
            (start[0] + length + distance, start[1] + dir_sign * (distance + total_width)),
            (start[0], start[1] + dir_sign * (distance + total_width)),
            (start[0], start[1] + half_width),
        ]

        rotated = self._rotate_polygon(points, start, direction)
        return Polygon(rotated)

    def _compute_junction_polygon(
        self,
        component: Dict[str, Any],
        start: Tuple[float, float],
        direction: float,
        total_width: float
    ) -> Polygon:
        """Compute bounding polygon for fork/T-junction."""
        # Note: angle affects the branch direction but we use simplified bounding box
        _angle = component.get("angle", math.pi / 2)
        junction_length = total_width * 2  # Approximate junction size
        half_width = total_width / 2

        # T-shape approximation
        points = [
            (start[0], start[1] - half_width),
            (start[0] + junction_length, start[1] - half_width),
            (start[0] + junction_length, start[1] - total_width),
            (start[0] + junction_length * 1.5, start[1]),
            (start[0] + junction_length, start[1] + total_width),
            (start[0] + junction_length, start[1] + half_width),
            (start[0], start[1] + half_width),
        ]

        rotated = self._rotate_polygon(points, start, direction)
        return Polygon(rotated)

    def _compute_intersection_polygon(
        self,
        component: Dict[str, Any],
        start: Tuple[float, float],
        direction: float,
        total_width: float
    ) -> Polygon:
        """Compute bounding polygon for 4-way intersection."""
        spacing = component.get("spacing", 10)
        size = total_width + spacing

        # Square intersection area
        points = [
            (start[0], start[1] - size / 2),
            (start[0] + size, start[1] - size / 2),
            (start[0] + size, start[1] + size / 2),
            (start[0], start[1] + size / 2),
        ]

        rotated = self._rotate_polygon(points, start, direction)
        return Polygon(rotated)

    def _compute_roundabout_polygon(
        self,
        component: Dict[str, Any],
        start: Tuple[float, float],
        direction: float,
        total_width: float
    ) -> Polygon:
        """Compute bounding polygon for roundabout (circular)."""
        radius = component.get("radius", 20)
        arm_length = component.get("arm_length", 10)

        # Total radius including arms
        total_radius = radius + arm_length + total_width

        # Circular polygon approximation
        num_points = 32
        center_x = start[0] + arm_length + radius
        center_y = start[1]

        points = []
        for i in range(num_points):
            theta = (i / num_points) * 2 * math.pi
            px = center_x + total_radius * math.cos(theta)
            py = center_y + total_radius * math.sin(theta)
            points.append((px, py))

        rotated = self._rotate_polygon(points, start, direction)
        return Polygon(rotated)

    def _compute_lane_switch_polygon(
        self,
        component: Dict[str, Any],
        start: Tuple[float, float],
        direction: float,
        total_width: float
    ) -> Polygon:
        """Compute bounding polygon for lane switch (tapered section)."""
        # Lane switch is essentially a trapezoid
        left_out = component.get("left_lanes_out", 2)
        right_out = component.get("right_lanes_out", 2)
        lane_width = component.get("lane_width", 3.5)

        width_in = total_width
        width_out = (left_out + right_out) * lane_width
        length = max(width_in, width_out) * 2  # Transition length

        half_in = width_in / 2
        half_out = width_out / 2

        points = [
            (start[0], start[1] - half_in),
            (start[0] + length, start[1] - half_out),
            (start[0] + length, start[1] + half_out),
            (start[0], start[1] + half_in),
        ]

        rotated = self._rotate_polygon(points, start, direction)
        return Polygon(rotated)

    def compute_endpoint(
        self,
        component: Dict[str, Any],
        start: Tuple[float, float],
        direction: float
    ) -> Tuple[Tuple[float, float], float]:
        """
        Compute the endpoint position and direction after a component.

        Args:
            component: Component dictionary
            start: Starting point
            direction: Starting direction in radians

        Returns:
            Tuple of (endpoint_position, endpoint_direction)
        """
        comp_type = component.get("type", "straight")

        if comp_type == "straight":
            length = component.get("length", 20)
            end_x = start[0] + length * math.cos(direction)
            end_y = start[1] + length * math.sin(direction)
            return ((end_x, end_y), direction)

        elif comp_type == "curve":
            radius = component.get("radius", 20)
            angle = component.get("angle", math.pi / 2)

            # Curve changes direction
            new_direction = direction + angle

            # Endpoint calculation (simplified)
            # For a curve, end position depends on radius and angle
            curve_sign = 1 if angle > 0 else -1
            center_x = start[0] - curve_sign * radius * math.sin(direction)
            center_y = start[1] + curve_sign * radius * math.cos(direction)

            end_x = center_x + curve_sign * radius * math.sin(new_direction)
            end_y = center_y - curve_sign * radius * math.cos(new_direction)

            return ((end_x, end_y), new_direction)

        elif comp_type == "u_shape":
            # length is used for the arm segments, distance for perpendicular offset
            _length = component.get("length", 20)  # Not used in simplified endpoint calc
            distance = component.get("distance", 15)
            turn_dir = component.get("direction", "right")

            # U-turn reverses direction
            dir_sign = 1 if turn_dir == "right" else -1
            new_direction = direction + math.pi

            # Perpendicular offset
            perp_x = -dir_sign * distance * math.sin(direction)
            perp_y = dir_sign * distance * math.cos(direction)

            end_x = start[0] + perp_x
            end_y = start[1] + perp_y

            return ((end_x, end_y), new_direction)

        elif comp_type == "lane_switch":
            # Lane switch is like a straight
            lane_width = component.get("lane_width", 3.5)
            total_lanes = component.get("right_lanes", 2) + component.get("left_lanes", 2)
            length = total_lanes * lane_width * 2

            end_x = start[0] + length * math.cos(direction)
            end_y = start[1] + length * math.sin(direction)
            return ((end_x, end_y), direction)

        else:
            # For junctions, roundabouts, etc. - simplified
            # Continue in same direction with offset
            offset = 20  # Default junction size
            end_x = start[0] + offset * math.cos(direction)
            end_y = start[1] + offset * math.sin(direction)
            return ((end_x, end_y), direction)

    def place_component(
        self,
        component: Dict[str, Any],
        validate: bool = True
    ) -> Optional[PlacedComponent]:
        """
        Attempt to place a component at the current endpoint.

        Args:
            component: Component to place
            validate: Whether to check for overlap

        Returns:
            PlacedComponent if successful, None if overlap detected
        """
        polygon = self.compute_bounding_polygon(
            component,
            self.current_endpoint,
            self.current_direction
        )

        if validate and self.check_overlap(polygon):
            return None

        # Compute new endpoint
        new_endpoint, new_direction = self.compute_endpoint(
            component,
            self.current_endpoint,
            self.current_direction
        )

        placed = PlacedComponent(
            component=component,
            polygon=polygon,
            endpoint=new_endpoint,
            direction=new_direction
        )

        self.placed_components.append(placed)
        self.covered_areas.append(polygon)
        self.current_endpoint = new_endpoint
        self.current_direction = new_direction

        return placed

    def try_parameters(
        self,
        component_type: str,
        param_options: Dict[str, List[Any]],
        max_attempts: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Try different parameter combinations to find valid placement.

        Similar to RoadGen's update() function - shuffles parameters
        and tries combinations until finding one that doesn't overlap.

        Args:
            component_type: Type of component
            param_options: Dict of parameter name -> list of values to try
            max_attempts: Maximum number of parameter combinations to try

        Returns:
            Valid component dict if found, None if all overlap
        """
        import random

        # Shuffle parameter lists
        shuffled_params = {
            k: random.sample(v, len(v))
            for k, v in param_options.items()
        }

        attempts = 0

        # Try combinations
        for params in self._generate_param_combinations(shuffled_params, max_attempts):
            attempts += 1

            component = {
                "type": component_type,
                "id": f"{component_type}_test",
                "sequence_index": len(self.placed_components),
                **params
            }

            polygon = self.compute_bounding_polygon(
                component,
                self.current_endpoint,
                self.current_direction
            )

            if not self.check_overlap(polygon):
                return component

            if attempts >= max_attempts:
                break

        return None

    def _generate_param_combinations(
        self,
        param_dict: Dict[str, List[Any]],
        max_count: int
    ):
        """Generate parameter combinations up to max_count."""
        import itertools

        keys = list(param_dict.keys())
        values = [param_dict[k] for k in keys]

        count = 0
        for combo in itertools.product(*values):
            if count >= max_count:
                break
            yield dict(zip(keys, combo))
            count += 1


def visualize_network(validator: SpatialValidator, output_path: str = None):
    """
    Visualize the placed road network using matplotlib.

    Args:
        validator: SpatialValidator with placed components
        output_path: If provided, save figure to this path
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
    except ImportError:
        print("matplotlib not available for visualization")
        return

    _, ax = plt.subplots(figsize=(12, 12))

    patches = []
    colors = []

    color_map = {
        "straight": "lightblue",
        "curve": "lightgreen",
        "fork": "orange",
        "t_intersection": "yellow",
        "intersection": "red",
        "roundabout": "purple",
        "u_shape": "pink",
        "lane_switch": "cyan",
    }

    for placed in validator.placed_components:
        comp_type = placed.component.get("type", "straight")
        coords = list(placed.polygon.exterior.coords)
        patch = MplPolygon(coords, closed=True)
        patches.append(patch)
        colors.append(color_map.get(comp_type, "gray"))

    collection = PatchCollection(patches, alpha=0.6)
    collection.set_facecolors(colors)
    collection.set_edgecolors("black")
    ax.add_collection(collection)

    # Auto-scale
    if validator.covered_areas:
        all_bounds = unary_union(validator.covered_areas).bounds
        margin = 20
        ax.set_xlim(all_bounds[0] - margin, all_bounds[2] + margin)
        ax.set_ylim(all_bounds[1] - margin, all_bounds[3] + margin)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Road Network ({len(validator.placed_components)} components)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=comp_type, alpha=0.6)
        for comp_type, color in color_map.items()
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Test the spatial validator
    validator = SpatialValidator(safety_margin=0.5)

    # Place some test components
    components = [
        {"type": "straight", "length": 50, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2},
        {"type": "curve", "radius": 25, "angle": math.pi/2, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2},
        {"type": "straight", "length": 30, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2},
        {"type": "t_intersection", "angle": math.pi/2, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2},
        {"type": "straight", "length": 40, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2},
    ]

    print("Testing spatial validator...")
    for i, comp in enumerate(components):
        comp["id"] = f"{comp['type']}_{i}"
        comp["sequence_index"] = i

        result = validator.place_component(comp, validate=True)
        if result:
            print(f"  Placed {comp['type']} at {result.endpoint}")
        else:
            print(f"  OVERLAP: Could not place {comp['type']}")

    print(f"\nTotal placed: {len(validator.placed_components)}")

    # Visualize
    visualize_network(validator, "test_network.png")
